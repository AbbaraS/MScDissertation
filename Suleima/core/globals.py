from core.Log import *

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

LR_SWEEP = [2e-4, 5e-4, 8e-4]
DR_SWEEP = [0.2, 0.3, 0.4]
WD_SWEEP = [1e-4, 5e-4, 1e-5]

fixed_dropout_rate = 0.2             # A reasonable, fixed value

PARAM_GRID = [
	{"paramID": 1, "LR": 5e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},
	{"paramID": 2, "LR": 5e-4, "WD": 1e-4, "DR": 0.3, "epochs":10},
	{"paramID": 3, "LR": 5e-4, "WD": 1e-4, "DR": 0.4, "epochs":10},
	{"paramID": 4, "LR": 5e-4, "WD": 1e-6, "DR": 0.2, "epochs":10},   # --- Confirm with your other best WD value ---
	{"paramID": 5, "LR": 2e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},   # --- Check a slightly lower LR just in case ---
	{"paramID": 6, "LR": 8e-4, "WD": 1e-4, "DR": 0.2, "epochs":10},    # --- Check a slightly higher LR just in case ---
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



