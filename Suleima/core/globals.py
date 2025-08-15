from core.Log import *
#PBAR_POSITION = 0

OUTER_FOLDS = 4
INNER_FOLDS = 3
T=True
F=False

LABEL_MAP = {
	"other_heart": 1,
	"myocardium": 2,
	"left_ventricle": 3,
}

'''

0.0001
0.0005
0.00001

'''

LR_SWEEP = [6e-4, 5e-4, 4e-4, 3e-4]
DR_SWEEP = [0.3]
TH_SWEEP = [0.4, 0.5]
WD_SWEEP = [1e-4, 1e-6]

fixed_dropout_rate = 0.2             # A reasonable, fixed value

DONE_PARAM_GRID = [
	{"paramID": 1, "LR": 5e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},
	{"paramID": 2, "LR": 5e-4, "WD": 1e-4, "DR": 0.3, "epochs":10},
	{"paramID": 3, "LR": 5e-4, "WD": 1e-4, "DR": 0.4, "epochs":10},
	{"paramID": 4, "LR": 5e-4, "WD": 1e-6, "DR": 0.2, "epochs":10},
	{"paramID": 5, "LR": 2e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},
	{"paramID": 6, "LR": 8e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},]

PARAM_GRID = [
	{"paramID": 9, "LR": 0.0008, "WD": 0.000001, "DR": 0.2, "epochs":10},
	#{"paramID": 10, "LR": 0.0002, "WD": 0.0001, "DR": 0.4, "epochs":10},
	#{"paramID": 11, "LR": 0.001, "WD": 0.0001, "DR": 0.2, "epochs":10},
	#{"paramID": 12, "LR": 0.001, "WD": 0.000001, "DR": 0.2, "epochs":10},
	{"paramID": 1, "LR": 5e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},
	{"paramID": 2, "LR": 5e-4, "WD": 1e-4, "DR": 0.3, "epochs":10},
	{"paramID": 3, "LR": 5e-4, "WD": 1e-4, "DR": 0.4, "epochs":10},
	{"paramID": 4, "LR": 5e-4, "WD": 1e-6, "DR": 0.2, "epochs":10},
	{"paramID": 5, "LR": 2e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},
	{"paramID": 6, "LR": 8e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},
]

OUTER_FOLDS_PARAMS=[
	{"paramID": 9, "LR": 0.0008, "WD": 0.000001, "DR": 0.2, "epochs":10},
	{"paramID": 6, "LR": 8e-4, "WD": 1e-4, "DR": 0.2, "epochs":10}
]






CASE_INFO = ["ID", "directory", "volumes"]
MONAI_KEYS = ["image", "mask"]

SEGMENT_FILES = {
	"myocardium":       	"heart_myocardium.nii.gz",
	"left_ventricle":   	"heart_ventricle_left.nii.gz",
	"right_ventricle":  	"heart_ventricle_right.nii.gz",
	"left_atrium":      	"heart_atrium_left.nii.gz",
	"right_atrium":     	"heart_atrium_right.nii.gz",
}

HUMIN = -175.0
HUMAX = 250.0
TARGETVOL = (64, 64, 64)
TARGETORIEN = "RAS"

MEAN= 151.40625
STD= 86.16580963134766#


















'''
import itertools
LR_SWEEP = [6e-4, 5e-4, 4e-4, 3e-4]
DR_SWEEP = [0.3]
TH_SWEEP = [0.4]
WD_SWEEP = [1e-4]
# 1. Define all model-specific hyperparameter sweeps in one dictionary
model_configs = {
	"MultiViewCNN": {
		"LR_SWEEP": [6e-4, 5e-4, 4e-4, 3e-4],
		"DR_SWEEP": [0.3],
		"WD_SWEEP": [1e-4]
	},
	"MLP_META": {
		"LR_SWEEP": [6e-4, 5e-4, 4e-4, 3e-4],
		"DR_SWEEP": [0.3],
		"WD_SWEEP": [1e-4]
	},
	"RESNET_18": {
		"LR_SWEEP": [6e-4, 5e-4, 4e-4, 3e-4],
		"DR_SWEEP": [0.3],
		"WD_SWEEP": [1e-4]
	},
	"Single_ViewCNN": {
		"LR_SWEEP": [6e-4, 5e-4, 4e-4, 3e-4],
		"DR_SWEEP": [0.3],
		"WD_SWEEP": [1e-4]
	}
}

# 2. Define global parameters that are the same for all models
GLOBAL_PARAMS = {
	"TH_SWEEP": [0.4],
	"P": 5,
	"Epochs": 30
}


INNER_CV_parameters = []
ID = 1

# Iterate through each model and its specific configuration
for model_name, config in model_configs.items():

	# Generate all unique combinations of the model's hyperparameters
	# e.g., for MultiViewCNN, this will create (1e-3, 0.3, 1e-4), (1e-3, 0.4, 1e-4), etc.
	hp_combinations = list(itertools.product(
		config['LR_SWEEP'],
		config['DR_SWEEP'],
		config['WD_SWEEP'],
		GLOBAL_PARAMS['TH_SWEEP'] # Include global sweeps here too
	))

	# Loop through outer and inner folds
	for outer_fold_idx in range(0, 5):
		for inner_fold_idx in range(0, 3):
			# Loop through each hyperparameter combination for this model
			for i, (lr, dr, wd, th) in enumerate(hp_combinations):
				item = {
					"ExpID": ID,
					"Model": model_name,
					'OUTER_FOLD': outer_fold_idx,
					'INNER_FOLD': inner_fold_idx,
					"hypers": {
						"HPset": i + 1,
						"LR": lr,
						"WD": wd,
						"DR": dr,
						"TH": th,
						"P": GLOBAL_PARAMS['P'],
						"Epochs": GLOBAL_PARAMS['Epochs'],
					},
					"trained": False,
				}
				INNER_CV_parameters.append(item)
				ID += 1

print(f"Total combinations generated: {len(INNER_CV_parameters)}")
#save_to_json(INNER_CV_parameters, filename="training/INNER_FOLDS.json")

# Print the first few items to see the structure
for param in INNER_CV_parameters[:3]:
	print(param)





'''