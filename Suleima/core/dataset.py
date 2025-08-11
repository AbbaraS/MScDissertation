
import json
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from core.preprocessing import *
from core.mydataloader import *


def run_one_fold(train_datalist, val_datalist, test_datalist, fold_idx):
	"""
	Executes the entire pipeline for a single fold of the cross-validation.
	This includes:
	1. Calculating normalization stats for the fold's training data.
	2. Creating DataLoaders.
	3. Initializing and training the model.
	4. Evaluating the model on the fold's test set.
	Args:
		train_datalist (list): The list of training cases for this fold.
		val_datalist (list): The list of validation cases for this fold.
		test_datalist (list): The list of test cases for this fold.
		fold_idx (int): The index of the current fold (for logging).
	Returns:
		float: The performance score (e.g., AUC) for this fold.
	"""
	logging.info(f"--- Starting Fold {fold_idx + 1} ---")
	logging.info(f"Fold Split: {len(train_datalist)} train, {len(val_datalist)} val, {len(test_datalist)} test.")


	HUstats = [case["stats"] for case in train_datalist]
	UH_mean, HU_std = calculate_HU_stats(HUstats)

	ages = [case['age'] for case in train_datalist]
	AGE_mean = np.mean(ages); AGE_std = np.std(ages)
	logging.info(f"Fold {fold_idx + 1} | HU_mean={UH_mean:.2f}, HU_std={HU_std:.2f}, AGE_mean={AGE_mean:.2f}, AGE_std={AGE_std:.2f}")
	fold_stats = {
		'HU_mean': UH_mean,
		'HU_std': HU_std,
		'AGE_mean': AGE_mean,
		'AGE_std': AGE_std}

	train_loader, val_loader, test_loader = get_data_loaders(
	     				train_datalist,
						val_datalist,
						test_datalist,
						fold_stats)

	# 3. Initialize model and train it
	# This function would contain your epoch loop, training, validation, and saving the best model
	# trained_model = train_and_evaluate_model(train_loader, val_loader)

	# 4. Evaluate the final model on the held-out test set
	# score = evaluate_final_model(trained_model, test_loader)
	score = np.random.rand() # Placeholder for the actual score
	logging.info(f"--- Fold {fold_idx + 1} Score: {score:.4f} ---")

	return score



