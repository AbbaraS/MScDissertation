import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import copy
from core.CNNmodel import *
from core.benchmarks import create_adapted_resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
		if P_counter >= hypers['P']:  return best_V_loss
	return best_V_loss

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





