import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import copy
from core.CNNmodel import *
from core.benchmarks import create_adapted_resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import logging
from core.globals import *
from tqdm.notebook import tqdm
from core.Log import *
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
	P = hypers['P']
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

def EVALUATE_MODEL1(model, test_loader, hypers):
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
			current_batch_size = len(CaseID)
			for i in range(current_batch_size):
				#print(c)
				#c += 1
				# Use .item() to get the Python scalar value from the tensor
				#print(i)
				case_id = CaseID[i]
				pred = prediction[i].item()
				prob = probability[i].item()
				log.info(f"     {ExpID}; {hypers['HPset']}; {hypers['Fold']}; {case_id}; {pred}; {prob:.4f};")


			#log.info(f"		{ExpID}; {hypers['HPset']}; {hypers['Fold']}; {CaseID}; {prediction}; {probability};")

	final_loss = running_loss / eval_N
	return final_loss, all_probabilities, all_labels, all_predictions

def train_INNER_model(model, train_loader, val_loader, experiment):
	log = logging.getLogger('INNER_5Ktrain')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	hypers = experiment['hypers']
	ExpID = experiment['ExpID']
	epochs = hypers['Epochs']

	LR = hypers['LR']
	WD = hypers['WD']
	P = hypers['P']

	optimizer = optim.Adam(model.parameters(), lr= LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2, factor=0.5,
								  threshold=1e-3, threshold_mode='rel',
								  cooldown=0, min_lr=1e-6)
	criterion = nn.BCEWithLogitsLoss()

	best_auc = -np.inf
	best_loss_at_best_auc = np.inf
	best_th = 0.5

	no_improve = 0

	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)


	print(f"	↳ Experiment {ExpID} | Training model... ")
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0

		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad(set_to_none=True)

			logits = model(axi, sag, cor, met)
			T_loss = criterion(logits, lbl)

			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)

		T_loss = running_loss / train_N

		model.eval()
		running_loss = 0.0
		all_labels = []  #y_true
		all_probs = []   #y_pred

		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				logits = model(axi, sag, cor, met)
				V_loss = criterion(logits, lbl)
				running_loss += V_loss.item() * lbl.size(0)

				all_labels.extend(lbl.cpu())
				all_probs.extend(torch.sigmoid(logits).cpu())

		V_loss = running_loss / val_N
		all_labels = torch.cat(all_labels).numpy().reshape(-1)
		all_probs = torch.cat(all_probs).numpy().reshape(-1)
		val_auc = roc_auc_score(all_labels, all_probs)

		scheduler.step(V_loss)
		th_star = best_threshold(all_labels, all_probs)
		improved = val_auc > best_auc + 1e-6

		log.info(f"{experiment['Model']};    {ExpID};    {experiment['OUTER_FOLD']};    {experiment['INNER_FOLD']};    {hypers['HPset']};    {epoch:02d};    {T_loss:.4f};    {V_loss:.4f}    {val_auc:.4f};    {th_star:.6f};    {optimizer.param_groups[0]['lr']};    {no_improve:02d};")

		if improved:
			best_auc = val_auc
			best_loss_at_best_auc = V_loss
			best_th = th_star
			best_epoch = epoch
			no_improve = 0
		else:
			no_improve += 1
		if no_improve >= P: break

	results = {
		"best_val_auc": float(best_auc),
		"best_val_loss": float(best_loss_at_best_auc),
		"best_threshold": float(best_th),
		"best_epoch": int(best_epoch),
		"epochs_ran": int(best_epoch + 1)}

	return results





def train_INNER_modelRESNET(model, train_loader, val_loader, experiment):
	log = logging.getLogger('INNER_5KtrainBENCHMARKS')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	feature_extractor, model = create_adapted_resnet18(device)

	hypers = experiment['hypers']
	ExpID = experiment['ExpID']
	epochs = hypers['Epochs']

	LR = hypers['LR']
	WD = hypers['WD']
	P = hypers['P']

	optimizer = optim.Adam(model.parameters(), lr= LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2, factor=0.5,
								  threshold=1e-3, threshold_mode='rel',
								  cooldown=0, min_lr=1e-6)
	criterion = nn.BCEWithLogitsLoss()

	best_auc = -np.inf
	best_loss_at_best_auc = np.inf
	best_th = 0.5

	no_improve = 0

	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)


	print(f"	↳ Experiment {ExpID} | Training model... ")
	for epoch in range(epochs):
		model.train()
		feature_extractor.eval()
		running_loss = 0.0

		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			with torch.no_grad():
				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)
			combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)

			optimizer.zero_grad(set_to_none=True)

			logits = model(combined_input)
			T_loss = criterion(logits, lbl)

			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)

		T_loss = running_loss / train_N

		model.eval()
		running_loss = 0.0
		all_labels = []  #y_true
		all_probs = []   #y_pred

		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)

				combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)

				logits = model(combined_input)
				V_loss = criterion(logits, lbl)
				running_loss += V_loss.item() * lbl.size(0)

				all_labels.extend(lbl.cpu())
				all_probs.extend(torch.sigmoid(logits).cpu())

		V_loss = running_loss / val_N
		all_labels = torch.cat(all_labels).numpy().reshape(-1)
		all_probs = torch.cat(all_probs).numpy().reshape(-1)
		val_auc = roc_auc_score(all_labels, all_probs)

		scheduler.step(V_loss)
		th_star = best_threshold(all_labels, all_probs)
		improved = val_auc > best_auc + 1e-6

		log.info(f"{experiment['Model']};    {ExpID};    {experiment['OUTER_FOLD']};    {experiment['INNER_FOLD']};    {hypers['HPset']};    {epoch:02d};    {T_loss:.4f};    {V_loss:.4f}    {val_auc:.4f};    {th_star:.6f};    {optimizer.param_groups[0]['lr']};    {no_improve:02d};")

		if improved:
			best_auc = val_auc
			best_loss_at_best_auc = V_loss
			best_th = th_star
			best_epoch = epoch
			no_improve = 0
		else:
			no_improve += 1
		if no_improve >= 3: break

	results = {
		"ExpID": ExpID,
		"Model": experiment['Model'],
		"best_val_auc": float(best_auc),
		"best_val_loss": float(best_loss_at_best_auc),
		"best_threshold": float(best_th),
		"best_epoch": int(best_epoch),
		"epochs_ran": int(epoch)}

	return results









def best_threshold(all_labels, all_probs, utility="youden"):
	"""
	Pick a post-hoc decision threshold on validation predictions.
	utility: "youden" (maximize TPR-FPR) or "f1".
	"""
	if utility == "f1":
		# scan unique probabilities for F1
		# (for speed you can sample a subset if very large)
		thr = np.unique(all_probs)
		f1s = [f1_score(all_labels, all_probs >= t) for t in thr]
		idx = int(np.argmax(f1s))
		return float(thr[idx])
	else:
		fpr, tpr, thr = roc_curve(all_labels, all_probs)
		j = tpr - fpr
		idx = int(np.argmax(j))
		return float(thr[idx])  # may be outside [0,1] if degenerate; fine.






def train_MLP(model, train_loader, val_loader, experiment):
	log = logging.getLogger('OUTER_train')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	hypers = experiment['hypers']
	epochs = hypers['Epochs']
	optimizer = optim.Adam(model.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2,
								  factor=0.9,
								  threshold=0.01,
								  min_lr=8e-4)   # 0.0008 < 0.0009
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	print(f"	↳ Experiment {experiment['ExpID']} | Training model... ")
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		all_labels = []
		all_preds = []

		for batch in train_loader:
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			optimizer.zero_grad()
			outputs = model(met)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > hypers['TH']
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
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)
				outputs = model(met)
				V_loss = criterion(outputs, lbl)

				running_loss += V_loss.item() * lbl.size(0)
				prediction = torch.sigmoid(outputs) > hypers['TH']
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
		log.info(f"	 MLP_META;	{experiment['ExpID']};	  {experiment['OUTER_FOLD']};		 {hypers['HPset']};  				 {epoch}; 			{T_loss}; 			{T_acc};					{V_loss}; 	 	{V_acc};		{P_counter}; 		{optimizer.param_groups[0]['lr']}")
		if P_counter >= hypers['P']: break
	model.load_state_dict(best_model_state)
	return model, best_model_state

def train_RESNET(train_loader, val_loader, experiment):
	log = logging.getLogger('OUTER_train')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	feature_extractor, classifier = create_adapted_resnet18(device)
	#model.to(device)
	hypers = experiment['hypers']
	epochs = hypers['Epochs']
	TH = hypers['TH']
	optimizer = optim.Adam(classifier.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
	P_counter = 0
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2,
								  factor=0.9,
								  threshold=0.01,
								  min_lr=8e-4)   # 0.0008 < 0.0009
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)

	print(f"	↳ RESNET Experiment {experiment['ExpID']} | Training model... ")
	for epoch in range(epochs):

		classifier.train()
		feature_extractor.eval()

		all_labels = []
		all_preds = []

		running_loss = 0.0
		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			with torch.no_grad():
				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)

			combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)

			optimizer.zero_grad()
			outputs = classifier(combined_input)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > TH
			all_preds.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

		T_loss = running_loss / train_N
		T_acc = np.mean(np.array(all_preds) == np.array(all_labels))

		classifier.eval()
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

				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)

				combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)
				outputs = classifier(combined_input)
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
			best_model_state = copy.deepcopy(classifier.state_dict())
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"	 RESNET;	{experiment['ExpID']};	  {experiment['OUTER_FOLD']};		 {hypers['HPset']};  				 {epoch}; 			{T_loss}; 			{T_acc};					{V_loss}; 	 	{V_acc};		{P_counter}; 		{optimizer.param_groups[0]['lr']}")
		if P_counter >= hypers['P']: break
	classifier.load_state_dict(best_model_state)
	return classifier, best_model_state





def train_SINGLEVIEW(model, train_loader, val_loader, experiment):
	log = logging.getLogger('OUTER_train')
	#log.info(f"		 ExpID; HP_Set;  Fold;   Epoch;   TrainLoss;   		 TrainAcc;             ValLoss;              	ValAcc; 	LR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	hypers = experiment['hypers']
	LR = hypers['LR']
	WD = hypers['WD']
	TH = hypers['TH']
	ExpID = experiment['ExpID']
	P = hypers['P']
	epochs = hypers['Epochs']

	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
							   patience=2, factor=0.9, threshold=0.01)
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	print(f"	↳ Single View Experiment {ExpID} | Training model... ")
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		all_labels = []
		all_preds = []

		for batch in train_loader:
			#axi = batch["axial_image"].to(device)
			#cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad()
			#outputs = model(axi, met)
			outputs = model(sag, met)
			#outputs = model(cor, met)
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
				#axi = batch["axial_image"].to(device)
				#cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				#outputs = model(axi, met)
				outputs = model(sag, met)
				#outputs = model(cor, met)
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
		log.info(f"		SAGITTAL_VIEW; 	{ExpID};	 {hypers['HPset']}; 	{experiment['OUTER_FOLD']};		 {epoch}; 		{T_loss}; 		{T_acc}; 			{V_loss}; 		{V_acc}; 		{P_counter};		{optimizer.param_groups[0]['lr']}	")
		if P_counter >= P: break
	model.load_state_dict(best_model_state)
	return model, best_model_state





