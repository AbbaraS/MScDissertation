import nibabel as nib
import os
import numpy as np


class CardiacCT:
    def __init__(self, caseID, ):
        self.caseID = caseID
        self.casePath = f"data/cases/{caseID}"
        self.fullCT = None
        self.croppedCT = None
        self.resampledCT = None
        self.log = []

        self.LV_totalseg = None
        self.LA_totalseg = None  
        self.RV_totalseg = None  
        self.RA_totalseg = None  
        self.MYO_totalseg = None 
        #self.dicomPath = f"../Takotsubo-Syndrome/data/Inputs/{caseID}/dicom"
        
        self.segmentPath = os.path.join(self.casePath, "segments")
        self.missing = []
    
    def flog(self, msg):
        self.log.append(msg)
        
    # === Volume Getters ===

    def get_fullCT(self):
        if self.fullCT is not None:
            return self.fullCT

        path = os.path.join(self.casePath, "fullCT.nii.gz")
        if os.path.exists(path):
            self.fullCT = nib.load(path).get_fdata()
        else:
            msg = "fullCT.nii.gz missing"
            self.flog(msg)
            self.missing.append("fullCT")
        return self.fullCT

    def get_croppedCT(self):
        if self.croppedCT is not None:
            return self.croppedCT

        path = os.path.join(self.casePath, "croppedCT.nii.gz")
        if os.path.exists(path):
            self.croppedCT = nib.load(path).get_fdata()
        else:
            msg = "croppedCT.nii.gz missing"
            self.flog(msg)
            self.missing.append("croppedCT")
        return self.croppedCT

    def get_resampledCT(self):
        if self.resampledCT is not None:
            return self.resampledCT

        path = os.path.join(self.casePath, "resampledCT.nii.gz")
        if os.path.exists(path):
            self.resampledCT = nib.load(path).get_fdata()
        else:
            msg = "resampledCT.nii.gz missing"
            self.flog(msg)
            self.missing.append("resampledCT")
        return self.resampledCT

    # === Segment Getters ===

    def get_segment(self, name, filename):
        attr_name = f"{name}_totalseg"
        if getattr(self, attr_name) is not None:
            return getattr(self, attr_name)

        path = os.path.join(self.casePath, "segments", filename)
        if os.path.exists(path):
            data = nib.load(path).get_fdata()
            setattr(self, attr_name, data)
        else:
            msg = f"{filename} missing"
            self.flog(msg)
            self.missing.append(name)
            return None

        return getattr(self, attr_name)

    def get_LV_totalseg(self):
        return self.get_segment("LV", "heart_ventricle_left.nii.gz")

    def get_LA_totalseg(self):
        return self.get_segment("LA", "heart_atrium_left.nii.gz")

    def get_RV_totalseg(self):
        return self.get_segment("RV", "heart_ventricle_right.nii.gz")

    def get_RA_totalseg(self):
        return self.get_segment("RA", "heart_atrium_right.nii.gz")

    def get_MYO_totalseg(self):
        return self.get_segment("MYO", "heart_myocardium.nii.gz")
    
    # === Loaders ===

def load_fullCT(case: CardiacCT):
    """Loads the full CT volume for the case."""
    return case.get_fullCT()

def load_croppedCT(case: CardiacCT):
    """
    Loads the cropped CT volume for the case.
    Returns the cropped CT data or None if not found.
    """
    return case.get_croppedCT()

def load_resampledCT(case: CardiacCT):
    """
    Loads the resampled CT volume for the case.
    Returns the resampled CT data or None if not found.
    """
    return case.get_resampledCT()

def load_segments(case: CardiacCT):
    """
    Loads all available cardiac segments from disk.
    Returns a dictionary of segment_name: volume_array for segments that exist.
    """
    segments = {
        "LV": case.get_LV_totalseg(),
        "LA": case.get_LA_totalseg(),
        "RV": case.get_RV_totalseg(),
        "RA": case.get_RA_totalseg(),
        "MYO": case.get_MYO_totalseg(),
    }

    # Filter out missing segments
    available_segments = {k: v for k, v in segments.items() if v is not None}
    return available_segments

def export_to_csv(case: CardiacCT, csv_path: str):
    """Writes log and metadata to CSV (e.g. shape, voxel size)."""
    # Optional: extract case info and write
    pass
            
def cardiac_bounding_box(case):
    """
    Computes the bounding box of available cardiac segments.
    Returns slice objects: (slice_x, slice_y, slice_z) or None if no segments are loaded.
    """
    segments = load_segments(case)
    if not segments:
        case.flog("No segments available for bounding box.")
        return None

    combined_mask = np.sum([seg > 0 for seg in segments.values()], axis=0) > 0

    coords = np.array(np.where(combined_mask))
    if coords.size == 0:
        case.flog("Bounding box: No positive voxels found.")
        return None

    x_min, y_min, z_min = coords.min(axis=1)
    x_max, y_max, z_max = coords.max(axis=1)

    # Ensure bounding box fits within fullCT shape
    shape = case.get_fullCT().shape
    x0, x1 = max(x_min, 0), min(x_max + 1, shape[0])
    y0, y1 = max(y_min, 0), min(y_max + 1, shape[1])
    z0, z1 = max(z_min, 0), min(z_max + 1, shape[2])

    return slice(x0, x1), slice(y0, y1), slice(z0, z1)

def crop_to_bbox(case):
    """
    Applies bounding box cropping to fullCT and available segments.
    Stores cropped volumes in case.croppedCT, case.LV_croppedseg, etc.
    """
    bbox = cardiac_bounding_box(case)
    if bbox is None:
        return

    case.croppedCT = case.get_fullCT()[bbox]

    segments = load_segments(case)
    for key, volume in segments.items():
        cropped = volume[bbox]
        setattr(case, f"{key}_croppedseg", cropped)

def save_cropped(case):
    """
    Saves cropped CT and cropped segments to appropriate folders.
    - croppedCT → data/cases/{caseID}/croppedCT.nii.gz
    - cropped segments → data/cases/{caseID}/cropped/{segment}.nii.gz
    """
    # Prepare paths
    base_dir = case.casePath
    cropped_dir = os.path.join(base_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    # Save cropped CT
    if case.croppedCT is not None:
        path = os.path.join(base_dir, "croppedCT.nii.gz")
        nib.save(nib.Nifti1Image(case.croppedCT, affine=np.eye(4)), path)
    else:
        case.flog("No croppedCT found to save.")

    # Save cropped segments
    for key in ['LV', 'LA', 'RV', 'RA', 'MYO']:
        cropped_attr = getattr(case, f"{key}_croppedseg", None)
        if cropped_attr is not None:
            path = os.path.join(cropped_dir, f"{key}.nii.gz")
            nib.save(nib.Nifti1Image(cropped_attr, affine=np.eye(4)), path)
        else:
            case.flog(f"No cropped segment for {key} found to save.")
    
    