from Suleima.Utilities.old_utils import *
from SampleInfos import SampleInfo 
from Suleima.core.SamplePath import SamplePath


# Constants
CNTRL = 'CNTRL'
TTS = 'TTS'
CNTRL_DICOM_ROOT = "../Takotsubo-Syndrome/data/Inputs/normal_cases"
TTS_DICOM_ROOT = "../Takotsubo-Syndrome/data/Inputs/TTS_cases"
BASE_ROOT = "data/cases/"

# Group to process
GROUP = TTS
DICOM_ROOT = TTS_DICOM_ROOT


def create_directories(path_obj):
    """Creates necessary folders for a given SamplePath object."""
    for dir_path in [
        path_obj.patient_dir,
        path_obj.segments_dir,
        path_obj.cropped_dir,
        path_obj.resampled_dir,
        path_obj.ctSlices_dir,
        path_obj.segmentSlices_dir,
        path_obj.pngSlices_dir
    ]:
        os.makedirs(dir_path, exist_ok=True)


def load_full_ct(case_obj):
    """Loads full CT from DICOM if not already converted, and stores metadata."""
    if not os.path.exists(case_obj.path.fullCT):
        dicom2nifti.convert_dicom.dicom_series_to_nifti(case_obj.path.dicom_dir, case_obj.path.fullCT)

    if os.path.exists(case_obj.path.fullCT):
        case_obj.load_fullCT()
    else:
        case_obj.flog("CT conversion failed or file missing.")


def process_case(patientID, group, dicom_root, base_root):
    """Handles one full patient case creation and metadata load."""
    case_id = f"{group}_{patientID}"
    path = SamplePath(patient_id=case_id, base_root=base_root, dicom_root=dicom_root)
    case = SampleInfo(patient_id=case_id, path=path)

    create_directories(path)
    load_full_ct(case)

    return case  # Optional: return to add to a list


def main():
    for patientID in os.listdir(DICOM_ROOT):
        if patientID.startswith('.'):
            continue  # skip .DS_Store or hidden files

        print(f"Processing patient: {patientID}")
        case = process_case(patientID, GROUP, DICOM_ROOT, BASE_ROOT)

        # You can now access attributes or methods
        print(case.fullCTshape)
        break  # remove to process all patients


if __name__ == "__main__":
    main()