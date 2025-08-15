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
from core.Log import *
logger = logging.getLogger('root')
fold_log = logging.getLogger('folds')
from core.CVsplits import *
from core.globals import *





def TRAIN_MODEL(model, train_loader, val_loader, hypers):
	log = logging.getLogger('train')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	LR = hypers['LR']
	WD = hypers['WD']
	TH = hypers['TH']
	P = 5
	epochs = hypers['epochs']
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=P, factor=0.5)
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		all_labels = []
		all_preds = []

		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad()
			outputs = model(axi, sag, cor, met)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > TH
			all_preds.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

		T_loss = running_loss / train_N
		T_acc = np.mean(np.array(all_preds) == np.array(all_labels))

		model.eval()
		running_loss = 0.0
		all_labels = []
		all_preds = []
		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				outputs = model(axi, sag, cor, met)
				V_loss = criterion(outputs, lbl)

				running_loss += V_loss.item() * lbl.size(0)
				prediction = torch.sigmoid(outputs) > TH
				all_preds.extend(prediction.cpu().numpy())
				all_labels.extend(lbl.cpu().numpy())

		V_loss = running_loss / val_N
		V_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		scheduler.step(V_loss)

		if V_loss < best_V_loss:
			best_V_loss = V_loss

			best_model_state = copy.deepcopy(model.state_dict())
			P_counter = 0
		else:
			P_counter += 1

		log.info(f"		{hypers["paramID"]}; {hypers["Fold"]}; {epoch}; {T_loss}; {T_acc}; {V_loss}; {V_acc}; {optimizer.param_groups[0]['lr']}")
		if P_counter >= P: break
	model.load_state_dict(best_model_state)
	return model, best_model_state





def EVALUATE_MODEL(model, test_loader, hypers):
	TH = hypers['TH']
	log = logging.getLogger('evaluate')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	criterion = nn.BCEWithLogitsLoss()
	eval_N = len(test_loader.dataset)

	running_loss = 0.0
	all_predictions = []
	all_probabilities = []
	all_labels = []
	with torch.no_grad():
		for batch in test_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			CaseID = batch["CaseID"]

			outputs = model(axi, sag, cor, meta=met)
			loss = criterion(outputs, lbl)
			running_loss += loss.item() * lbl.size(0)

			probability = torch.sigmoid(outputs)
			all_probabilities.extend(probability.cpu().numpy())
			prediction = probability > TH

			all_predictions.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())
			log.info(f"		{hypers["paramID"]}; {hypers["Fold"]}; {CaseID}; {prediction}; {probability};")

	final_loss = running_loss / eval_N
	return final_loss, all_probabilities, all_labels, all_predictions





def run_OUTERS_topParamSets(experiments_results, completed_tasks, DL):
	print(completed_tasks)
	print(experiments_results)
	print(f"4-fold NCV Top HP Sets Experiment")
	for CURRENT_OUT_ID in range(OUTER_FOLDS):
		fold_log.info(f"\n\n\n############################## OUTER FOLD -{CURRENT_OUT_ID}- ##############################")
		logger.info(f"FOLD {CURRENT_OUT_ID} | Top HP Sets Experiment")

		train_loader, val_loader, _ = DL.create_outer_loaders(CURRENT_OUT_ID)
		val_N=len(val_loader.dataset)
		train_N=len(train_loader.dataset)
		fold_log.info(f"train_N: {train_N}, val_N: {val_N}")
		logger.info(f"--FOLD ID--,,,---paramID---,,,,-----Best Val Loss ")
		for hypers in OUTER_FOLDS_PARAMS:
			fold_log.info(f"---------------------------------  PARAM ID  -{hypers['paramID']}-  ---------------------------------")
			fold_log.info(f"{hypers}")
			paramID = hypers['paramID']

			if (CURRENT_OUT_ID, paramID) in completed_tasks: continue

			DR = hypers['DR']
			model = MultiViewCNN(DR)
			best_val_loss = train_INNER_model(model, train_loader, val_loader, hypers)

			logger.info(f"     {CURRENT_OUT_ID}     ,,,      {paramID}      ,,,,     {best_val_loss}    ")

			experiments_results.append({
				'OUTER_fold_id': CURRENT_OUT_ID,
				'paramID': paramID,
				'best_val_loss': best_val_loss
			})

			save_experiments_results(experiments_results)
			fold_log.info(f"FOLD {CURRENT_OUT_ID} | ParamID {paramID} | Best Val Loss: {best_val_loss:.4f}  --------")
			completed_tasks.add((CURRENT_OUT_ID, paramID))

	return completed_tasks

def run_OUTER_hp_search(hp_search_results, completed_tasks, DL):
	'''DONE. ONLY UPDATE IF ERROR OCCURS.'''
	print(f"4-fold Nested CV Hyperparameter Search")
	for CURRENT_OUT_ID in range(OUTER_FOLDS):
		fold_log.info(f"\n\n\n############################## OUTER FOLD -{CURRENT_OUT_ID}- ##############################")
		pbar_params = tqdm(PARAM_GRID, desc=f"Fold {CURRENT_OUT_ID} | HP Search", position=CURRENT_OUT_ID, leave=True)

		for set_params in pbar_params:
			fold_log.info(f"---------------------------------  PARAM ID  -{set_params['paramID']}-  ---------------------------------")
			fold_log.info(f"{set_params}")
			#for set in tqdm(PARAM_GRID, desc=f"OUTER FOLD {CURRENT_OUT_ID} | HP Search...", position=CURRENT_OUT_ID, leave=True):

			paramID = set_params['paramID']
			if (CURRENT_OUT_ID, paramID) in completed_tasks: continue
			inner_val_losses = run_INNER_folds(DL, CURRENT_OUT_ID, set_params, INNER_FOLDS)  # 1
			#inner_val_losses = [0.5535, 0.5540, 0.5520]  # Placeholder for actual inner fold results
			hp_search_results.append({
				"OUTER_fold_id": CURRENT_OUT_ID,
				"paramID": paramID,
				"avg_val_loss": np.mean(inner_val_losses)})
			save_hp_search_results(hp_search_results)
			fold_log.info(f"FOLD {CURRENT_OUT_ID} | ParamID {paramID} | Avg Val Loss: {np.mean(inner_val_losses):.4f}###############")
			completed_tasks.add((CURRENT_OUT_ID, paramID))
	return completed_tasks

def run_INNER_folds(DL, outer_id, hypers, inner_folds):
	'''DONE. ONLY UPDATE IF ERROR OCCURS.'''
	pbar_inner = tqdm(range(inner_folds), desc=f"  ↳ Inner Folds (ParamID {hypers['paramID']})", position=OUTER_FOLDS + 1, leave=False)
	inner_loop_results = []
	for INN in pbar_inner:
		train_loader, val_loader = DL.create_inner_loaders(
			outer_fold_id=outer_id,
			inner_fold_id=INN)

		DR = hypers['DR']
		model = MultiViewCNN(DR)

		best_val_loss = train_INNER_model(model, train_loader, val_loader, hypers)
		inner_loop_results.append(best_val_loss)
		fold_log.info(f"best_val_loss: {best_val_loss:.4f}")
	fold_log.info(f"inner_loop_results: {inner_loop_results}")
	return inner_loop_results


def train_INNER_model(model, train_loader, val_loader, hypers):
	''' DONE. DONT CHANGE IT EVER..'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	LR = hypers['LR']
	WD = hypers['WD']
	P = 3
	epochs = hypers['epochs']
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=P, factor=0.5)
	criterion = nn.BCEWithLogitsLoss()
	best_val_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)

	for epoch in range(epochs):
		model.train()
		train_running_loss = 0.0
		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad()
			outputs = model(axi, sag, cor, met)
			Tloss = criterion(outputs, lbl)
			Tloss.backward()
			train_running_loss += Tloss.item() * lbl.size(0)
			optimizer.step()
		T_loss = train_running_loss / train_N

		model.eval()
		val_running_loss = 0.0
		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				outputs = model(axi, sag, cor, met)
				Vloss = criterion(outputs, lbl)
				val_running_loss += Vloss.item() * lbl.size(0)

		val_loss = val_running_loss / val_N
		scheduler.step(val_loss)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			P_counter = 0
		else:
			P_counter += 1
			if P_counter >= P:
				#logger.info(f"Early stopping with Val Loss: {best_val_loss:.4f} at patience: {P_counter}/10")
				fold_log.info(f"	E: {epoch} | TL: {T_loss} |	VL: {val_loss} | P: {P_counter} | Early stopping triggered.")
				break
		fold_log.info(f"	E: {epoch} | TL: {T_loss} |	VL: {val_loss} | P: {P_counter}")
	return best_val_loss





def evaluate_model(model, test_loader, TH):
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
	y_pred = (y_prob > TH).astype(int)

	final_loss = running_loss / len(test_loader.dataset)
	final_acc = np.mean(y_pred == y_true)
	final_auc = roc_auc_score(y_true, y_prob) # Calculate AUC

	return {"loss": final_loss,
		    "accuracy": final_acc,
		    "auc": final_auc}




