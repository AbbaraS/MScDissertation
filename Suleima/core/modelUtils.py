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
from tqdm.notebook import tqdm
from core.Log import *
logger = logging.getLogger('root')
fold_log = logging.getLogger('folds')
from core.CVsplits import *
from core.globals import *





def TRAIN_MODEL(model, train_loader, val_loader, hypers):
	log = logging.getLogger('OUTER_train')
	log.info(f"		 ExpID; HP_Set;  Fold;   Epoch;   TrainLoss;   		 TrainAcc;             ValLoss;              	ValAcc; 	LR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	LR = hypers['LR']
	WD = hypers['WD']
	TH = hypers['TH']
	ExpID = hypers['ExpID']
	P = 5
	epochs = hypers['epochs']
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=P, factor=0.5)
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	pbar_epochs = tqdm(range(epochs), desc=f"	↳ Experiment {ExpID} | Training model... ", position=ExpID, leave=True)
	for epoch in pbar_epochs:
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
		log.info(f"		{ExpID}; {hypers['HPset']}; {hypers['Fold']}; {epoch}; {T_loss}; {T_acc}; {V_loss}; {V_acc}; {optimizer.param_groups[0]['lr']}")
		if P_counter >= P: break
	model.load_state_dict(best_model_state)
	return model, best_model_state





def EVALUATE_MODEL(model, test_loader, hypers):
	TH = hypers['TH']
	ExpID = hypers['ExpID']
	log = logging.getLogger('OUTER_evaluate')
	log.info(f"		 ExpID; HP_Set;  Fold;   CaseID;   prediction;   		 probability    ")
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
		pbar_eval = tqdm(test_loader, desc=f"	↳ Experiment {ExpID} | Evaluating model... ", position=ExpID, leave=True)
		for batch in pbar_eval:
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
			log.info(f"		{ExpID}; {hypers['HPset']}; {hypers['Fold']}; {CaseID}; {prediction}; {probability};")

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

			save_to_json(experiments_results, filename="training/OUTERSexperiments_results.json")
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
			save_to_json(hp_search_results, filename="training/hp_search_results.json")
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



def train_INNER_model(model, train_loader, val_loader, experiment):
	''' DONE. DONT CHANGE IT EVER..'''
	log = logging.getLogger('INNER_train')
	#log.info(f"		 ExpID; OUTER_FOLD; INNER_FOLD;	HP_Set;   Epoch;  TrainLoss;  ValLoss;  P;  LR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	hypers = experiment['hypers']
	#LR = hypers['LR']
	#WD = hypers['WD']
	#ExpID = experiment['ExpID']
	#P = hypers['P']
	epochs = hypers['Epochs']
	optimizer = optim.Adam(model.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	pbar_epochs = tqdm(range(epochs), desc=f"	↳ Experiment {experiment['ExpID']} | Training model... ", position=experiment['ExpID'], leave=True)
	for epoch in pbar_epochs:
		model.train()
		running_loss = 0.0
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
		T_loss = running_loss / train_N
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
				V_loss = criterion(outputs, lbl)
				running_loss += V_loss.item() * lbl.size(0)

		V_loss = running_loss / val_N
		scheduler.step(V_loss)
		if V_loss < best_V_loss:
			best_V_loss = V_loss
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"		{experiment['ExpID']}; 	{experiment['OUTER_FOLD']}; 	{experiment['INNER_FOLD']}; 	{hypers['HPset']};  	{epoch}; 	{T_loss}; 	{V_loss};  	{P_counter}; 	{optimizer.param_groups[0]['lr']}")
		if P_counter >= hypers['P']: break
	return best_V_loss

#def train_INNER_RESNET(model, train_loader, val_loader, experiment):
#	''' DONE. DONT CHANGE IT EVER..'''
#	log = logging.getLogger('INNER_train')
#	#log.info(f"		 ExpID; OUTER_FOLD; INNER_FOLD;	HP_Set;   Epoch;  TrainLoss;  ValLoss;  P;  LR")
#	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#	feature_extractor, classifier = create_adapted_resnet18(device)
#	model.to(device)
#	hypers = experiment['hypers']
#	#LR = hypers['LR']
#	#WD = hypers['WD']
#	#ExpID = experiment['ExpID']
#	#P = hypers['P']
#	epochs = hypers['Epochs']
#	optimizer = optim.Adam(model.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
#	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
#	criterion = nn.BCEWithLogitsLoss()
#	best_V_loss = float('inf')
#	P_counter = 0
#	val_N=len(val_loader.dataset)
#	train_N=len(train_loader.dataset)
#	pbar_epochs = tqdm(range(epochs), desc=f"	↳ Experiment {experiment['ExpID']} | Training model... ", position=experiment['ExpID'], leave=True)
#	for epoch in pbar_epochs:
#		model.train()
#		running_loss = 0.0
#		for batch in train_loader:
#			axi = batch["axial_image"].to(device)
#			cor = batch["coronal_image"].to(device)
#			sag = batch["sagittal_image"].to(device)
#			met = batch["meta"].to(device)
#			lbl = batch["label"].to(device).unsqueeze(1)
#			axi_features = feature_extractor(axi)
#			cor_features = feature_extractor(cor)
#			sag_features = feature_extractor(sag)
#			combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)
#			optimizer.zero_grad()
#			outputs = classifier(combined_input)
#			T_loss = criterion(outputs, lbl)
#			T_loss.backward()
#			optimizer.step()
#			running_loss += T_loss.item() * lbl.size(0)
#		T_loss = running_loss / train_N
#		model.eval()
#		running_loss = 0.0
#		with torch.no_grad():
#			for batch in val_loader:
#				axi = batch["axial_image"].to(device)
#				cor = batch["coronal_image"].to(device)
#				sag = batch["sagittal_image"].to(device)
#				met = batch["meta"].to(device)
#				lbl = batch["label"].to(device).unsqueeze(1)
#				axi_features = feature_extractor(axi)
#				cor_features = feature_extractor(cor)
#				sag_features = feature_extractor(sag)
#				combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)
#				outputs = classifier(combined_input)
#				V_loss = criterion(outputs, lbl)
#				running_loss += V_loss.item() * lbl.size(0)
#
#		V_loss = running_loss / val_N
#		scheduler.step(V_loss)
#		if V_loss < best_V_loss:
#			best_V_loss = V_loss
#			P_counter = 0
#		else:
#			P_counter += 1
#		log.info(f"	 RESNET;	{experiment['ExpID']}; 	{experiment['OUTER_FOLD']}; 	{experiment['INNER_FOLD']}; 	{hypers['HPset']};  	{epoch}; 	{T_loss}; 	{V_loss};  	{P_counter}; 	{optimizer.param_groups[0]['lr']}")
#		if P_counter >= hypers['P']: break
#	return best_V_loss


def train_INNER_MLP2(model, train_loader, val_loader, experiment):
	''' DONE. DONT CHANGE IT EVER..'''
	log = logging.getLogger('INNER_train')
	#log.info(f"		 ExpID; OUTER_FOLD; INNER_FOLD;	HP_Set;   Epoch;  TrainLoss;  ValLoss;  P;  LR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	hypers = experiment['hypers']
	#LR = hypers['LR']
	#WD = hypers['WD']
	#ExpID = experiment['ExpID']
	#P = hypers['P']
	epochs = hypers['Epochs']
	optimizer = optim.Adam(model.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	pbar_epochs = tqdm(range(epochs), desc=f"	↳ Experiment {experiment['ExpID']} | Training model... ", position=experiment['ExpID'], leave=True)
	for epoch in pbar_epochs:
		model.train()
		running_loss = 0.0
		for batch in train_loader:
			#axi = batch["axial_image"].to(device)
			#cor = batch["coronal_image"].to(device)
			#sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			optimizer.zero_grad()
			outputs = model(met)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()
			running_loss += T_loss.item() * lbl.size(0)
		T_loss = running_loss / train_N
		model.eval()
		running_loss = 0.0
		with torch.no_grad():
			for batch in val_loader:
				#axi = batch["axial_image"].to(device)
				#cor = batch["coronal_image"].to(device)
				#sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)
				outputs = model(met)
				V_loss = criterion(outputs, lbl)
				running_loss += V_loss.item() * lbl.size(0)

		V_loss = running_loss / val_N
		scheduler.step(V_loss)
		if V_loss < best_V_loss:
			best_V_loss = V_loss
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"	 MLP;	{experiment['ExpID']}; 	{experiment['OUTER_FOLD']}; 	{experiment['INNER_FOLD']}; 	{hypers['HPset']};  	{epoch}; 	{T_loss}; 	{V_loss};  	{P_counter}; 	{optimizer.param_groups[0]['lr']}")
		if P_counter >= hypers['P']: break
	return best_V_loss





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




