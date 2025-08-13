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
PARAM_GRID = [
{'paramID': 1, 'learning_rate': 0.0001, 'weight_decay': 1e-05,  'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 2, 'learning_rate': 0.0001, 'weight_decay': 0.0001, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 3, 'learning_rate': 0.0001, 'weight_decay': 0.001, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 4, 'learning_rate': 0.001, 'weight_decay': 1e-05, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 5, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 6, 'learning_rate': 0.001, 'weight_decay': 0.001, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 7, 'learning_rate': 0.005, 'weight_decay': 1e-05, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 8, 'learning_rate': 0.005, 'weight_decay': 0.0001, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8},
{'paramID': 9, 'learning_rate': 0.005, 'weight_decay': 0.001, 'threshold': 0.4, 'DR': 0.4, 'epochs': 50, 'patience': 10, 'batch_size': 8}]

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



