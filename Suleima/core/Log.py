from pathlib import Path
import json
import logging


json_path = Path("data/data_info.json")

def save_dataset_info(dataset):
	"""Save dataset information to a JSON file."""
	try:
		json_path.parent.mkdir(parents=True, exist_ok=True)
		with open(json_path, 'w') as json_file:
			json.dump(dataset, json_file, indent=4)
		logging.info(f"Successfully saved data to: {json_path}")
		# Reload updated dataset info
		return load_dataset_info()
	except Exception as e:
		logging.error(f"An error occurred: {e}")


def load_dataset_info():
	"""Load dataset information from a JSON file."""
	try:
		if json_path.exists():
			with open(json_path, 'r') as json_file:
				dataset = json.load(json_file)
			return dataset
		else:
			logging.warning(f"File not found: {json_path}")
			return None
	except Exception as e:
		logging.error(f"An error occurred: {e}")
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
				logging.error("Error: JSON data is not a list of dictionaries or a single dictionary.")
				return

		filtered_data = []
		for item in data:
			new_item = {key: item[key] for key in keys_to_keep if key in item}
			filtered_data.append(new_item)

		# Save the new, smaller dataset to the output file
		with open(output_path, 'w') as f:
			# Use indent for readability of the output JSON
			json.dump(filtered_data, f, indent=4)

		print(f"Successfully filtered data and saved to {output_path}")

	except FileNotFoundError:
		print(f"Error: The file {input_path} was not found.")
	except KeyError as e:
		print(f"Error: A key {e} was not found in one of the data entries.")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
