import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import copy
from core.CNNmodel import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from core.globals import *
from tqdm import tqdm
logger = logging.getLogger('root')
fold_log = logging.getLogger('folds')
from core.CVsplits import *


def run_INNER_folds(DL, outer_id, hypers, inner_folds=INNER_FOLDS):
	'''DONE. ONLY UPDATE IF ERROR OCCURS.'''
	inner_loop_results = []
	#for inner_id in range(inner_folds):
	for INN in tqdm(range(inner_folds), desc=f"Inner Fold training..."):
		train_loader, val_loader = DL.create_inner_loaders(
			outer_fold_id=outer_id,
			inner_fold_id=INN)
		fold_log.info(f"F-{INN} | Data loaders created. Starting model training...")
		DR = hypers['DR']
		model = MultiViewCNN(DR)
		best_val_loss = train_INNER_model(model, train_loader, val_loader, hypers)
		inner_loop_results.append(best_val_loss)
		fold_log.info(f"best_val_loss: {best_val_loss:.4f}")
	fold_log.info(f"inner_loop_results: {inner_loop_results}")
	return inner_loop_results



def run_OUTER_hp_search(hp_search_results, completed_tasks, DL):
	'''DONE. ONLY UPDATE IF ERROR OCCURS.'''
	print(f"4-fold Nested CV Hyperparameter Search")
	for CURRENT_OUT_ID in range(OUTER_FOLDS):
		for set in tqdm(PARAM_GRID, desc=f"Fold {CURRENT_OUT_ID} | Hyperparameter Search..."):
			fold_log.info(f"{set}")
			paramID = int(set['paramID'])
			if (CURRENT_OUT_ID, paramID) in completed_tasks: continue
			inner_val_losses = run_INNER_folds(DL, CURRENT_OUT_ID, set, INNER_FOLDS)
			#inner_val_losses = [0.5535, 0.5540, 0.5520]  # Placeholder for actual inner fold results
			hp_search_results.append({
				"OUTER_fold_id": CURRENT_OUT_ID,
				"paramID": paramID,
				"avg_val_loss": np.mean(inner_val_losses)})
			save_hp_search_results(hp_search_results)
			fold_log.info(f"Fold {CURRENT_OUT_ID} | ParamID {paramID} | Avg Val Loss: {np.mean(inner_val_losses):.4f}")
			completed_tasks.add((CURRENT_OUT_ID, paramID))
	return completed_tasks



def train_INNER_model(model, train_loader, val_loader, hypers):
	''' DONE. DONT CHANGE IT EVER..'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	LR = hypers['LR']
	WD = hypers['WD']

	P = 3
	epochs = hypers['epochs']
	#labels = [d['label'] for d in train_loader.dataset.data_dicts]
	#pos_weight = torch.tensor(labels.count(0) / labels.count(1), dtype=torch.float32)
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=P, factor=0.5)
	#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
	criterion = nn.BCEWithLogitsLoss()
	best_val_loss = float('inf')
	P_counter = 0
	for epoch in range(epochs):
		model.train()
		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad()
			outputs = model(axi, sag, cor, met)
			loss = criterion(outputs, lbl)
			loss.backward()
			optimizer.step()

		model.eval()
		running_loss = 0.0
		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				outputs = model(axi, sag, cor, met)
				loss = criterion(outputs, lbl)
				running_loss += loss.item() * lbl.size(0)

		val_loss = running_loss / len(val_loader.dataset)
		scheduler.step(val_loss)

		fold_log.info(f"	E: {epoch} | Val loss: {val_loss} | P_counter: {P_counter}")
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			P_counter = 0
		else:
			P_counter += 1
			if P_counter >= P:
				logger.info(f"Early stopping with Val Loss: {best_val_loss:.4f} at patience: {P_counter}/10")
				break
	return best_val_loss






def train_model(model, train_loader, val_loader, hypers):
	epoch_history = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	LR= hypers['learning_rate']
	WD= hypers['weight_decay']

	P = 10

	# Calculate pos_weight for handling class imbalance
	labels = [d['label'] for d in train_loader.dataset.data_dicts]
	pos_weight = torch.tensor(labels.count(0) / labels.count(1), dtype=torch.float32)

	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=P, factor=0.5)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

	best_val_loss = float('inf')
	best_model_state = None
	P_counter = 0

	#fold_log.info(f"Epoch | TrnLss | TrnAcc | VlLss | VlAcc | LR  ")
	for epoch in range(50):
		train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
		val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

		current_lr = optimizer.param_groups[0]['lr']
		fold_log.info(f"{epoch+1}/{hypers['epochs']} | "
					 f"{train_loss:.4f} | {train_acc:.4f} | "
					 f"{val_loss:.4f} | {val_acc:.4f} | {current_lr:.4f}")

		epoch_history.append({
			'epoch': epoch + 1,
			'train_loss': train_loss,
			'train_acc': train_acc,
			'val_loss': val_loss,
			'val_acc': val_acc})

		scheduler.step(val_loss)

		#logger.info(f"Epoch {epoch+1}: Best Val Loss: {best_val_loss:.4f}, Patience: {P_counter}/{hypers['patience']}")
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			# deepcopy to save the state in memory, not to disk
			best_model_state = copy.deepcopy(model.state_dict())
			P_counter = 0
		else:
			P_counter += 1
			if P_counter >= P:
				fold_log.info(f"Early stopping with Val Loss: {best_val_loss:.4f} at patience: {P_counter}/{hypers['patience']}")
				break

	model.load_state_dict(best_model_state)
	return model, epoch_history, best_val_loss

def train_epoch(model, loader, optimizer, criterion, return_loss_acc=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.train()
	running_loss = 0.0
	all_labels = []
	all_preds = []

	for i, batch in enumerate(loader):
		axi = batch["axial_image"].to(device)
		cor = batch["coronal_image"].to(device)
		sag = batch["sagittal_image"].to(device)
		met = batch["meta"].to(device)
		lbl = batch["label"].to(device).unsqueeze(1)


		optimizer.zero_grad()
		outputs = model(axi, sag, cor, meta=met)
		loss = criterion(outputs, lbl)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * lbl.size(0)

		if return_loss_acc:
			preds = torch.sigmoid(outputs) > 0.5
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())
	if return_loss_acc:
		epoch_loss = running_loss / len(loader.dataset)
		epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		return epoch_loss, epoch_acc
	else:
		return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	all_labels = []
	all_preds = []

	with torch.no_grad():
		for batch in loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			outputs = model(axi, sag, cor, meta=met)
			loss = criterion(outputs, lbl)

			running_loss += loss.item() * lbl.size(0)
			preds = torch.sigmoid(outputs) > 0.5
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

	epoch_loss = running_loss / len(loader.dataset)
	epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
	return epoch_loss, epoch_acc





def evaluate_model(model, test_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	threshold = 0.4
	labels = [d['label'] for d in test_loader.dataset.data_dicts]
	pos_weight = torch.tensor(labels.count(0) / labels.count(1), dtype=torch.float32)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

	running_loss = 0.0
	y_true = []
	y_prob = []

	with torch.no_grad():
		for batch in test_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			outputs = model(axi, sag, cor, meta=met)
			loss = criterion(outputs, lbl)
			running_loss += loss.item() * lbl.size(0)

			# Store probabilities for AUC and true labels
			probs = torch.sigmoid(outputs)
			y_prob.extend(probs.cpu().numpy())
			y_true.extend(lbl.cpu().numpy())

	y_true = np.array(y_true)
	y_prob = np.array(y_prob)
	y_pred = (y_prob > threshold).astype(int)

	final_loss = running_loss / len(test_loader.dataset)
	final_acc = np.mean(y_pred == y_true)
	final_auc = roc_auc_score(y_true, y_prob) # Calculate AUC

	return {"loss": final_loss,
		    "accuracy": final_acc,
		    "auc": final_auc}




