import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from core.Log import *
from core.globals import *
pools = ["holdout", "main"]

def load_dataset(pool="main"):
	full_dl = load_dataset_info()
	return [i for i in full_dl if i["pool"] == pool]

json_files = [
	"training/hp_search_results.json",
	"training/OUTERSexperiments_results.json",
	"training/param_grid.json",

	]




def save_to_json(data, filename="training/OUTERSexperiments_results.json"):
	print(f"Saving to {filename}.")
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








def get_dataset_stats(datalist):
	'''DONE. ONLY UPDATE IF ERROR OCCURS.'''
	stats = [case["stats"] for case in datalist]
	ages = [case['age'] for case in datalist]

	count = sum(s['count'] for s in stats)
	sum_global = sum(s['sum'] for s in stats)
	sum_sq_global = sum(s['sum_sq'] for s in stats)

	UHmean = sum_global / count
	HUvar = (sum_sq_global / count) - (UHmean ** 2)
	HUstd = np.sqrt(max(0.0, HUvar))

	return {'HUmean':  float(UHmean),
			'HUstd':   float(HUstd),
			'AGEmean': float(np.mean(ages)),
			'AGEstd':  float(np.std(ages))}




def create_folds_stats(OUTER_K=4, INNER_K=3):

	OUTER_cv = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=42)
	INNER_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=42)
	main_training_set = load_dataset("main")
	main_training_labels = [i["label"] for i in main_training_set]
	all_folds_data = []

	for outer_fold_idx, (outer_train_indices, outer_test_indices) in enumerate(OUTER_cv.split(main_training_set, main_training_labels)):
		print("Generating and saving fold indices...")
		outer_train_pool = [main_training_set[i] for i in outer_train_indices]
		outer_stats=get_dataset_stats(outer_train_pool)
		outer_fold_data = {
			"outer_fold_id": outer_fold_idx,
			"outer_train_indices": outer_train_indices,
			"outer_test_indices": outer_test_indices,
			"outer_fold_stats": outer_stats,
			"inner_folds": []
		}

		inner_loop_labels = [main_training_labels[i] for i in outer_train_indices]

		for inner_fold_idx, (inner_train_indices_local, inner_val_indices_local) in enumerate(INNER_cv.split(outer_train_pool, inner_loop_labels)):
			inner_train_data = [outer_train_pool[i] for i in inner_train_indices_local]
			inner_stats = get_dataset_stats(inner_train_data)

			inner_fold_data = {
				"inner_fold_id": inner_fold_idx,
				"inner_train_indices": inner_train_indices_local,
				"inner_val_indices": inner_val_indices_local,
				"inner_fold_stats": inner_stats
			}
			outer_fold_data["inner_folds"].append(inner_fold_data)

		all_folds_data.append(outer_fold_data)

	with open("training/folds_indices_stats.pkl", "wb") as f:
		pickle.dump(all_folds_data, f)



def get_fold_indices(outer_fold_id, inner_fold_id):
	"""
	Returns a tuple containing
		(inner_train_indices, inner_val_indices)
	for the specified fold.
	"""
	with open("training/fold_indices.pkl", "rb") as f:
		all_folds_data = pickle.load(f)
	outer_fold_data = all_folds_data[outer_fold_id]
	inner_fold_data = outer_fold_data['inner_folds'][inner_fold_id]

	train_indices = inner_fold_data['inner_train_indices']
	val_indices = inner_fold_data['inner_val_indices']

	return train_indices, val_indices


def get_fold_stats(outer_fold_id=None, inner_fold_id=None):
	"""
	Returns a tuple containing
		(outer_fold_stats, inner_fold_stats)
	for the specified fold.
	"""
	with open("training/folds_indices_stats.pkl", "rb") as f:
		all_folds_data = pickle.load(f)
	if outer_fold_id is None and inner_fold_id is None:
		return all_folds_data
	elif outer_fold_id is None:
		# Return all outer folds stats
		return [fold['outer_fold_stats'] for fold in all_folds_data]
	elif inner_fold_id is None:
		# Return all inner folds stats for the specified outer fold
		return [fold['inner_fold_stats'] for fold in all_folds_data[outer_fold_id]['inner_folds']]
	else:
		# Return stats for the specified outer and inner fold
		outer_fold_data = all_folds_data[outer_fold_id]
		inner_fold_data = outer_fold_data['inner_folds'][inner_fold_id]
		return outer_fold_data['outer_fold_stats'], inner_fold_data['inner_fold_stats']


def get_all_tasks(grid=PARAM_GRID):
	all_possible_tasks = []
	for outer_id in range(OUTER_FOLDS):
		for params in grid:
			param_id = params['paramID']
			all_possible_tasks.append((outer_id, param_id))
	return all_possible_tasks