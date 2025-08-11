from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

output_json_path = Path("data/dataset_info.json")

def save_dataset_info(dataset):
	"""Save dataset information to a JSON file."""
	try:
		output_json_path.parent.mkdir(parents=True, exist_ok=True)
		with open(output_json_path, 'w') as json_file:
			json.dump(dataset, json_file, indent=4)
		logging.info(f"Successfully saved data to: {output_json_path}")
	except Exception as e:
		logging.error(f"An error occurred: {e}")


def load_dataset_info():
	"""Load dataset information from a JSON file."""
	try:
		if output_json_path.exists():
			with open(output_json_path, 'r') as json_file:
				dataset = json.load(json_file)
			return dataset
		else:
			logging.warning(f"File not found: {output_json_path}")
			return None
	except Exception as e:
		logging.error(f"An error occurred: {e}")
		return None


