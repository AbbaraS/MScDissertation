from pathlib import Path
import json
import logging
import sys
import os



def root_logger():
	"""Configures both the root logger and the dedicated folds logger."""

	# --- Configure Root Logger (for general model.log and console) ---
	root_logger = logging.getLogger('root')
	root_logger.setLevel(logging.INFO)
	root_logger.propagate = False
	if not root_logger.hasHandlers():
		root_logger.propagate = False
		app_handler = logging.FileHandler('model.log', mode='a')
		app_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))
		root_logger.addHandler(app_handler)
		root_logger.addHandler(console_handler)


def folds_logger():
	# --- Configure Folds Logger (for folds.log) ---
	folds_logger = logging.getLogger('folds') # Give it a descriptive name
	folds_logger.setLevel(logging.INFO)
	folds_logger.propagate = False

	if not folds_logger.hasHandlers():
		folds_logger.propagate = False
		folds_handler = logging.FileHandler('folds.log', mode='a')
		folds_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))
		folds_logger.addHandler(folds_handler)



def setup_loggers():
	# --- Configure Logger
	train = logging.getLogger('OUTER_train')
	train.setLevel(logging.INFO)
	train.propagate = False

	evaluate = logging.getLogger('OUTER_evaluate')
	evaluate.setLevel(logging.INFO)
	evaluate.propagate = False

	if not train.hasHandlers():
		train.propagate = False
		train_handler = logging.FileHandler('OUTER_training.log', mode='a')		#REMEMBER: 'a' mode to append logs
		train_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		train.addHandler(train_handler)

	if not evaluate.hasHandlers():
		evaluate.propagate = False
		evaluate_handler = logging.FileHandler('OUTER_evaluating.log', mode='a')		#REMEMBER: 'a' mode to append logs
		evaluate_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		evaluate.addHandler(evaluate_handler)

	train = logging.getLogger('INNER_train')
	train.setLevel(logging.INFO)
	train.propagate = False

	if not train.hasHandlers():
		train.propagate = False
		train_handler = logging.FileHandler('INNER_training.log', mode='a')		#REMEMBER: 'a' mode to append logs
		train_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		train.addHandler(train_handler)



logger = logging.getLogger('root')

'''
per epoch metrics:
[fold_id, epoch_number , train_loss,  train_accuracy, val_loss, val_accuracy]

Final Fold Performance:
[fold_id, model_name, val_loss at which early stopping occurred] +
	key classifier metrics:
	AUC (Area Under the ROC Curve),
	F1-Score, Precision, Recall, Accuracy,
	positive predictive value, negative predictive value,
	false positive rate, false negative rate,
	uncalibrated_brier, calibrated_brier,
Hyperparameters and Metadata:
[learning_rate, weight_decay, batch_size, dropout_rate, image_size]


Does the 64x64x64 volume still clearly show the anatomical features needed for diagnosis & classification in this set up?

'''

training = ['model', 'OUT_K', 'params_id', 'INNER_K', '', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']

testing = ['model', 'fold_id', 'AUC', 'F1-Score', 'Precision',
		   'Recall', 'Accuracy', 'Specificity', 'Sensitivity']


training_filename = 'training_logs.csv'

RESULTS_FILEPATH = Path("training/hp_search_results.json")




def load_dataset(pool="main"):
	full_dl = load_dataset_info()
	return [i for i in full_dl if i["pool"] == pool]

json_files = [
	"training/hp_search_results.json",
	"training/OUTERSexperiments_results.json",
	"training/parameter_grid.json",

	]




def save_to_json(data, filename="training/OUTERSexperiments_results.json"):
	#print(f"Saving to {filename}.")
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, 'w') as f:
		json.dump(data, f, indent=2)

def load_from_json(filename="training/OUTERSexperiments_results.json"):
	if not os.path.exists(filename):
		print(f"No data found in {filename}.")
		return []
	with open(filename, 'r') as f:
		results = json.load(f)
		print(f"Loaded {filename}.")
		return results






json_path = Path("data/data_info.json")

def save_dataset_info(dataset, file="data/data_info.json"):
	"""Save dataset information to a JSON file."""
	try:
		Path(file).parent.mkdir(parents=True, exist_ok=True)
		with open(file, 'w') as jf:
			json.dump(dataset, jf, indent=4)
		print(f"Successfully saved data to: {file}")
	except Exception as e:
		print(f"An error occurred: {e}")

def load_dataset_info(file="data/data_info.json"):
	"""Load dataset information from a JSON file."""
	try:
		if Path(file).exists():
			with open(file, 'r') as jf:
				return json.load(jf)
		else:
			print(f"File not found: {file}")
			return []
	except Exception as e:
		print(f"An error occurred: {e}")
		return []

def filter_json_keys(input_path: str, output_path: str):
	"""
	Loads data from a JSON file, filters it to keep only specified keys,
	and saves the result to a new JSON file.
	"""

	keys_to_keep = ["ID", "label", "age", "gender", "directory",  "originals",  "cropped" ]
	try:
		# Load the original data from the input file
		with open(input_path, 'r') as f:
			data = json.load(f)
		if not isinstance(data, list):
			if isinstance(data, dict):
				data = [data]
			else:
				logger.error("Error: JSON data is not a list of dictionaries or a single dictionary.")
				return

		filtered_data = []
		for item in data:
			new_item = {key: item[key] for key in keys_to_keep if key in item}
			filtered_data.append(new_item)

		# Save the new, smaller dataset to the output file
		with open(output_path, 'w') as f:
			# Use indent for readability of the output JSON
			json.dump(filtered_data, f, indent=4)

		logger.info(f"Successfully filtered data and saved to {output_path}")

	except FileNotFoundError:
		logger.error(f"Error: The file {input_path} was not found.")
	except KeyError as e:
		logger.error(f"Error: A key {e} was not found in one of the data entries.")
	except Exception as e:
		logger.error(f"An unexpected error occurred: {e}")







