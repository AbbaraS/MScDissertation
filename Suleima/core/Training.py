from ctypes.util import test
from time import time
import torch
from core.mydataloader import * 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.CNNmodel import MultiViewCNN
import torch.nn as nn
from zmq import device


class TrainModel:
    def __init__(self, model, dataloader_module, pos_weight, WD=1e-3, LR=1e-5, patience=5, epochs=50, threshold_cutoff=0.5):
        self.model = model.to(self.get_device())
        self.dataloader_module = dataloader_module
        self.epochs = epochs
        self.patience = patience
        self.threshold_cutoff = threshold_cutoff  # threshold for binary classification
        self.device = self.get_device()
        
        self.optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, factor=0.5, verbose=True)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        
    def get_device(self):
        """Returns the device to be used for training."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_loaders(self):
        return self.dataloader_module.get_train_loader(), self.dataloader_module.get_val_loader(), self.dataloader_module.get_test_loader()



    def train1(self):
        train_loader, val_loader, test_loader = self.get_loaders()
        train_losses = []
        val_losses = []
        patience_counter = 0
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M")
        for epoch in range(self.epochs):
            
            train_loss, train_accuracy = self.train(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = torch.save(self.model.state_dict(), f"{self.timestamp}.pt")  # save best model
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            print(f"Epoch {epoch+1}/{self.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
                  f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        test_loss, test_accuracy = self.test(test_loader)
        
        print(f"Best Validation Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
                
    def train(self, loader):
        self.model.train()
        loss=0.0
        correct = 0
        total = 0
        
        for batch in loader:
            axial = batch["axial"].to(self.device)
            sagittal = batch["sagittal"].to(self.device)
            coronal = batch["coronal"].to(self.device)
            meta = batch["meta"].to(self.device)
            label = batch["label"].to(self.device)
            #pid = batch["pid"].to(self.device)
            
            self.optimizer.zero_grad()
            
            out = self.model(axial, sagittal, coronal, meta)
            loss = self.criterion(out, batch["label"].to(self.device))
            
            loss.backward()
            self.optimizer.step()
            
            total += label.size(0)
            loss += loss.item()
            predicted = (torch.sigmoid(out).cpu().detach().numpy() > self.threshold_cutoff).astype(int)
            correct += (predicted == label.cpu().numpy()).sum()
        
        accuracy = 100 * correct / total
        loss /= len(loader)
        print(f"Training Loss: {loss:.4f}, Training Accuracy: {accuracy:.2f}%")
        return loss, accuracy
        
    def validate(self, loader):
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                axial = batch["axial"].to(self.device)
                sagittal = batch["sagittal"].to(self.device)
                coronal = batch["coronal"].to(self.device)
                meta = batch["meta"].to(self.device)
                label = batch["label"].to(self.device)
                #pid = batch["pid"].to(self.device)
                
                out = self.model(axial, sagittal, coronal, meta)
                loss = self.criterion(out, label)
                loss += loss.item()
                total += label.size(0)
                predicted = (torch.sigmoid(out).cpu().detach().numpy() > self.threshold_cutoff).astype(int)
                correct += (predicted == label.cpu().numpy()).sum()
        
        accuracy = 100 * correct / total
        loss /= len(loader)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        return loss, accuracy

    def test(self, loader):
        self.model.load_state_dict(torch.load(f"{self.timestamp}.pt"))
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_prob = []
        y_pred = []
        
        with torch.no_grad():
            for batch in loader:
                axial = batch["axial"].to(self.device)
                sagittal = batch["sagittal"].to(self.device)
                coronal = batch["coronal"].to(self.device)
                meta = batch["meta"].to(self.device)
                label = batch["label"].to(self.device)
                
                out = self.model(axial, sagittal, coronal, meta)
                loss += self.criterion(out, label).item()
                
                total += label.size(0)
                predicted = (torch.sigmoid(out).cpu().detach().numpy() > self.threshold_cutoff).astype(int)
                correct += (predicted == label.cpu().numpy()).sum()
                y_true.extend(label.cpu().numpy())
                y_prob.extend(torch.sigmoid(out).cpu().detach().numpy())
                y_pred.extend(predicted)
        
        # Convert lists to numpy arrays for further analysis if needed
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.array(y_pred)
        
        accuracy = 100 * correct / total
        loss /= len(loader)

        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        return loss, accuracy
    
    def print_loader_cases(self, loader):
        """Prints the cases in the loader."""
        idx = 0
        for batch in loader:
            #x_axial = batch["axial"]
            #x_sagittal = batch["sagittal"]
            #x_coronal = batch["coronal"]
            #x_meta = batch["meta"]
            y = batch["label"]
            pid = batch["pid"]
    
            print(f"{idx} - {pid} - {y.item()}")
            idx += 1
            #break