import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import copy
from core.CNNmodel import *
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from monai.transforms import (
	Compose, EnsureChannelFirstd, Orientationd, Lambdad,
	CropForegroundd, NormalizeIntensityd, Resized, EnsureTyped,
	RandAffined, RandGaussianNoiseD)
from core.CardiacCTdataset import *

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger('root')
fold_log = logging.getLogger('folds')

def clamp_hu_values(image_tensor):
	return torch.clamp(image_tensor, min=-175.0, max=250.0)

def get_data_loaders(batch_size, train_datalist, val_datalist, test_datalist, stats):
	shape = (64, 64, 64)
	orientation = "RAS"

	base_transforms_list = [
		EnsureChannelFirstd(keys=["image", "mask"]),
		Orientationd(keys=["image", "mask"], axcodes=orientation),
		Lambdad(keys=["image"], func=clamp_hu_values),
		CropForegroundd(keys=["image", "mask"], source_key="mask"),
		NormalizeIntensityd(keys=["image"], subtrahend=stats['HUmean'], divisor=stats['HUstd']),
		Resized(keys=["image", "mask"], spatial_size=shape, mode=("bilinear", "nearest")),
		EnsureTyped(keys=["image", "mask"])]
	val_transforms = Compose(base_transforms_list) # Validation/Test transforms have no augmentation

	train_transforms_list = base_transforms_list + [
		RandAffined(keys=['image', 'mask'], prob=0.5, rotate_range=(0, 0, np.pi/12), scale_range=(0.1, 0.1, 0.1)),
		RandGaussianNoiseD(keys=['image'], prob=0.1)]
	train_transforms = Compose(train_transforms_list)

	train_dataset = CardiacCTDataset(train_datalist, train_transforms, stats)
	val_dataset   = CardiacCTDataset(val_datalist, val_transforms, stats)
	test_dataset  = CardiacCTDataset(test_datalist, val_transforms, stats)

	num_workers = 4
	trn_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	tst_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


	return trn_loader, val_loader, tst_loader


def get_Kfold_stats(datalist):
	stats = [case["stats"] for case in datalist]
	ages = [case['age'] for case in datalist]

	count = sum(s['count'] for s in stats)
	sum_global = sum(s['sum'] for s in stats)
	sum_sq_global = sum(s['sum_sq'] for s in stats)

	UHmean = sum_global / count
	HUvar = (sum_sq_global / count) - (UHmean ** 2)
	HUstd = np.sqrt(max(0.0, HUvar))

	return {'HUmean':  float(UHmean),
			'HUstd':   float(HUstd),
			'AGEmean': float(np.mean(ages)),
			'AGEstd':  float(np.std(ages))}

def run_k_fold(train_datalist, val_datalist, test_datalist, fold_idx):
	fold_log.info(f"---------- FOLD {fold_idx + 1} ----------")
	Kstats = get_Kfold_stats(train_datalist)

	# --- Define Hyperparameters ---
	hypers = {"LR": 1e-4, "WD": 1e-5, "epochs": 20,
			  "patience": 10, "batch_size": 8, "threshold_cutoff": 0.5, "DR": 0.4}

	train_loader, val_loader, test_loader = get_data_loaders(hypers["batch_size"], train_datalist, val_datalist, test_datalist, Kstats)
	# --- Initialize Model ---
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MultiViewCNN(dropout_rate=hypers["DR"]).to(device)

	best_model, epoch_history = train_model(model, train_loader, val_loader, hypers, fold_idx)
	final_scores = evaluate_model(best_model, test_loader, hypers)

	final_scores['learning_rate'] = hypers['LR']

	fold_log.info(f"Fold {fold_idx + 1} Stats: {Kstats}")
	fold_log.info(f"Fold {fold_idx + 1} Final scores: {final_scores}")
	return final_scores, epoch_history


def train_epoch(model, loader, optimizer, criterion, device):
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

		preds = torch.sigmoid(outputs) > 0.5
		all_preds.extend(preds.cpu().numpy())
		all_labels.extend(lbl.cpu().numpy())

	epoch_loss = running_loss / len(loader.dataset)
	epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
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


def train_model(model, train_loader, val_loader, hypers, fold_idx):
	epoch_history = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	# Calculate pos_weight for handling class imbalance
	labels = [d['label'] for d in train_loader.dataset.data_dicts]
	pos_weight = torch.tensor(labels.count(0) / labels.count(1), dtype=torch.float32)

	optimizer = optim.Adam(model.parameters(), lr=hypers['LR'], weight_decay=hypers['WD'])
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hypers['patience'],
							   factor=0.5)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

	best_val_loss = float('inf')
	best_model_state = None
	patience_counter = 0

	fold_log.info("---- Model training ----")
	fold_log.info(f"Epoch | TrnLss | TrnAcc | VlLss | VlAcc | LR  ")
	for epoch in range(hypers['epochs']):
		train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
		val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

		current_lr = optimizer.param_groups[0]['lr']
		fold_log.info(f"{epoch+1}/{hypers['epochs']} | "
					 f"{train_loss:.4f} | {train_acc:.4f} | "
					 f"{val_loss:.4f} | {val_acc:.4f} | {current_lr:.4f}")

		epoch_history.append({
			'fold_id': fold_idx + 1,
			'epoch': epoch + 1,
			'train_loss': train_loss,
			'train_acc': train_acc,
			'val_loss': val_loss,
			'val_acc': val_acc
		})
		scheduler.step(val_loss)
		#print(optimizer.param_groups[0]['lr'])
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			# deepcopy to save the state in memory, not to disk
			best_model_state = copy.deepcopy(model.state_dict())
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= hypers['patience']:
				fold_log.info("Early stopping triggered.")
				break

	model.load_state_dict(best_model_state)
	return model, epoch_history

def evaluate_model(model, test_loader, hypers):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

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
	y_pred = (y_prob > hypers['threshold_cutoff']).astype(int)

	final_loss = running_loss / len(test_loader.dataset)
	final_acc = np.mean(y_pred == y_true)
	final_auc = roc_auc_score(y_true, y_prob) # Calculate AUC

	return {"loss": final_loss,
		 "accuracy": final_acc,
		 "auc": final_auc,
		 }


