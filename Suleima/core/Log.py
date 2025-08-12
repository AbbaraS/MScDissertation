from pathlib import Path
import json
import logging
import sys

def root_logger():
	"""Configures both the root logger and the dedicated folds logger."""

	# --- Configure Root Logger (for general model.log and console) ---
	root_logger = logging.getLogger('root')
	root_logger.setLevel(logging.INFO)
	if not root_logger.hasHandlers():

		# Clear existing handlers if any
		root_logger.handlers.clear()
		# File handler for app.log
		app_handler = logging.FileHandler('model.log', mode='w')
		app_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))

		# Console handler
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
		root_logger.addHandler(app_handler)
		root_logger.addHandler(console_handler)


def folds_logger():
	# --- Configure Folds Logger (for folds.log) ---
	folds_logger = logging.getLogger('folds') # Give it a descriptive name
	folds_logger.setLevel(logging.INFO)

	if not folds_logger.hasHandlers():

		# Clear existing handlers if any
		folds_logger.handlers.clear()

		folds_logger.propagate = False
		# File handler for folds.log
		folds_handler = logging.FileHandler('folds.log', mode='w')
		folds_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
		folds_logger.addHandler(folds_handler)



logger = logging.getLogger('root')

json_path = Path("data/data_info.json")

def save_dataset_info(dataset):
	"""Save dataset information to a JSON file."""
	try:
		json_path.parent.mkdir(parents=True, exist_ok=True)
		with open(json_path, 'w') as json_file:
			json.dump(dataset, json_file, indent=4)
		logger.info(f"Successfully saved data to: {json_path}")
		# Reload updated dataset info
		return load_dataset_info()
	except Exception as e:
		logger.error(f"An error occurred: {e}")

def load_dataset_info():
	"""Load dataset information from a JSON file."""
	try:
		if json_path.exists():
			with open(json_path, 'r') as json_file:
				dataset = json.load(json_file)
			return dataset
		else:
			logger.warning(f"File not found: {json_path}")
			return None
	except Exception as e:
		logger.error(f"An error occurred: {e}")
		return None

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


'''
per epoch metrics:
[fold_id, epoch_number ,train_loss,  train_accuracy, val_loss, val_accuracy]

Final Fold Performance:
[fold_id, model_name, val_loss at which early stopping occurred] +
	key classifier metrics:
	[AUC (Area Under the ROC Curve), F1-Score, Precision, Recall, Accuracy]
Hyperparameters and Metadata:
[learning_rate, weight_decay, batch_size, dropout_rate, image_size]

'''




