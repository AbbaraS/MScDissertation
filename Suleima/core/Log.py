from pathlib import Path
import json
import logging
import sys
import os



def setup_loggers():
	root_logger = logging.getLogger('root')
	INNERtrain = logging.getLogger('INNER_train')
	OUTtrain = logging.getLogger('OUTER_train')
	evaluate = logging.getLogger('OUTER_evaluate')

	root_logger.setLevel(logging.INFO)
	INNERtrain.setLevel(logging.INFO)
	OUTtrain.setLevel(logging.INFO)
	evaluate.setLevel(logging.INFO)

	root_logger.propagate = False
	evaluate.propagate = False
	INNERtrain.propagate = False
	OUTtrain.propagate = False

	if not root_logger.hasHandlers():
		root_logger.propagate = False
		app_handler = logging.FileHandler('main.log', mode='a')
		app_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))
		root_logger.addHandler(app_handler)
		root_logger.addHandler(console_handler)

	if not INNERtrain.hasHandlers():
		INNERtrain.propagate = False
		train_handler = logging.FileHandler('INNER_training.log', mode='a')		#REMEMBER: 'a' mode to append logs
		train_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		INNERtrain.addHandler(train_handler)

	if not OUTtrain.hasHandlers():
		OUTtrain.propagate = False
		train_handler = logging.FileHandler('OUTER_training.log', mode='a')		#REMEMBER: 'a' mode to append logs
		train_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		OUTtrain.addHandler(train_handler)

	if not evaluate.hasHandlers():
		evaluate.propagate = False
		evaluate_handler = logging.FileHandler('OUTER_evaluating.log', mode='a')		#REMEMBER: 'a' mode to append logs
		evaluate_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		evaluate.addHandler(evaluate_handler)




def setup_loggers5K():
	root_logger = logging.getLogger('root')

	OUTtrain = logging.getLogger('OUTER_5Ktrain')
	evaluate = logging.getLogger('OUTER_5Kevaluate')

	root_logger.setLevel(logging.INFO)

	OUTtrain.setLevel(logging.INFO)
	evaluate.setLevel(logging.INFO)

	root_logger.propagate = False
	evaluate.propagate = False

	OUTtrain.propagate = False

	if not root_logger.hasHandlers():
		root_logger.propagate = False
		app_handler = logging.FileHandler('main.log', mode='a')
		app_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M'))
		root_logger.addHandler(app_handler)
		root_logger.addHandler(console_handler)

	if not OUTtrain.hasHandlers():
		OUTtrain.propagate = False
		train_handler = logging.FileHandler('OUTER_5Ktrain.log', mode='a')		#REMEMBER: 'a' mode to append logs
		train_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		OUTtrain.addHandler(train_handler)

	if not evaluate.hasHandlers():
		evaluate.propagate = False
		evaluate_handler = logging.FileHandler('OUTER_5Kevaluate.log', mode='a')		#REMEMBER: 'a' mode to append logs
		evaluate_handler.setFormatter(logging.Formatter('%(asctime)s ;		 %(message)s', datefmt='%H:%M'))
		evaluate.addHandler(evaluate_handler)




training = ['model', 'OUT_K', 'params_id', 'INNER_K', '', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']

testing = ['model', 'fold_id', 'AUC', 'F1-Score', 'Precision',
		   'Recall', 'Accuracy', 'Specificity', 'Sensitivity']

PATHS = [
	"training/OUT_5_HPSearch_EXPERIMENTS.json",
	"training/OUTERS_4_experiments_results.json",
	"training/outer4_hp_search_results.json",
	"training/outer_5_INNER_experiments.json",
	"training/outers_4_trainHistory.txt",
	"training/outer5_innerHistory.txt",
	"OUTERs_4_evaluating.txt",
	"training/OUT_4parameter_grid.json"

]



filepath = Path("")

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

def load_dataset(pool="main"):
	full_dl = load_dataset_info()
	return [i for i in full_dl if i["pool"] == pool]



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
	logger = logging.getLogger('root')

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







