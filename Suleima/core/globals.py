

import logging


T=True
F=False

LABEL_MAP = {
	"other_heart": 1,
	"myocardium": 2,
	"left_ventricle": 3,
}

CASE_INFO = ["ID", "directory", "volumes"]
KEYS = ["image", "mask"]

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


'''
"ID": Unique identifier for the case,
"directory": Path to the case directory,
"original_paths": []

'''





logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
