
from core.SamplePath import SamplePath
from core.SampleInfo import SampleInfo
from core.processing import crop

import os

group = "CNTRL" #"TTS"
CNTRL_dicom_root = "../Takotsubo-Syndrome/data/Inputs/normal_cases/"
TTS_dicom_root = "../Takotsubo-Syndrome/data/Inputs/takotsubo_cases/"
base_root = "data/cases/"

for caseID in os.listdir(base_root):
    if caseID.startswith('.'):
        continue



    
    
    
    
    #path = SamplePath(patient_id=caseID, base_root=base_root)
    #case = SampleInfo(patient_id=caseID, path=path)
    #
    #case.load_segments()
    #_ = case.ct_data  # triggers load
    #crop(case)
