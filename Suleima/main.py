
from core.SamplePath import SamplePath
from core.SampleInfo import SampleInfo
from core.processing import crop
import os

group = "TTS"
dicom_root = "../Takotsubo-Syndrome/data/Inputs/TTS_cases"
base_root = "data/cases"

for patientID in os.listdir(dicom_root):
    if patientID.startswith('.'):
        continue

    case_id = f"{group}_{patientID}"
    path = SamplePath(patient_id=case_id, base_root=base_root, dicom_root=dicom_root)
    case = SampleInfo(patient_id=case_id, path=path)
    
    case.load_segments()
    _ = case.ct_data  # triggers load
    crop(case)
