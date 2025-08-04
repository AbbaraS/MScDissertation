from core.CT_3D import *
import os


def cropping_volumes(case: CardiacCT):
    """
    Executes the cropping pipeline for a CardiacCT case:
    - Skips if croppedCT already exists.
    - Loads fullCT and segments.
    - Computes bounding box.
    - Crops fullCT and segments.
    - Saves cropped results.
    """
    # === Check if croppedCT already exists
    cropped_path = os.path.join(case.casePath, "croppedCT.nii.gz")
    if os.path.exists(cropped_path):
        case.flog("Cropped CT already exists. Skipping cropping.")
        return

    # === Load full CT and segments
    fullCT = case.get_fullCT()
    segments = load_segments(case)

    if fullCT is None or not segments:
        case.flog("Cannot crop: missing fullCT or segments.")
        return

    # === Crop and Save
    crop_to_bbox(case)
    save_cropped(case)
    case.flog("Cropping completed and saved.")
    