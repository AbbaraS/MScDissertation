
import os

class SamplePath:
    def __init__(self, patient_id, base_root, dicom_root):
        self.patient_id = patient_id
        self.patient_dir = os.path.join(base_root, patient_id)
        self.dicom_dir = os.path.join(dicom_root, patient_id)
        
        self.segments_dir = os.path.join(self.patient_dir, "segments")
        self.cropped_dir = os.path.join(self.patient_dir, "cropped")
        self.resampled_dir = os.path.join(self.patient_dir, "resampled")
        self.ctSlices_dir = os.path.join(self.patient_dir, "ctSlices")
        self.segmentSlices_dir = os.path.join(self.patient_dir, "segmentSlices")
        self.pngSlices_dir = os.path.join(self.patient_dir, "pngSlices")

        # File paths
        self.fullCT = os.path.join(self.patient_dir, "fullCT.nii.gz")
        self.croppedCT = os.path.join(self.patient_dir, "croppedCT.nii.gz")
        self.resampledCT = os.path.join(self.patient_dir, "resampledCT.nii.gz")

        # Segment paths
        self.segments = {
            "LV": os.path.join(self.segments_dir, "heart_ventricle_left.nii.gz"),
            "RV": os.path.join(self.segments_dir, "heart_ventricle_right.nii.gz"),
            "LA": os.path.join(self.segments_dir, "heart_atrium_left.nii.gz"),
            "RA": os.path.join(self.segments_dir, "heart_atrium_right.nii.gz"),
            "MYO": os.path.join(self.segments_dir, "heart_myocardium.nii.gz"),
        }

        # Resampled paths
        self.resampled_segments = {
            k: os.path.join(self.resampled_dir, f"{k}.nii.gz") for k in self.segments
        }

        self.heart_mask = os.path.join(self.segments_dir, "heart_mask.nii.gz")





class SamplePath:
    def __init__(self
                 , patient_id
                 , dicom_dir
                 , patient_dir
                 , segments_dir
                 , cropped_dir 
                 , resampled_dir
                 , ctSlices_dir
                 , segmentSlices_dir
                 , pngSlices_dir
                 ):
        self.patient_id = patient_id
        self.dicom_dir = dicom_dir
        self.patient_dir = patient_dir
        self.segments_dir = segments_dir
        self.cropped_dir = cropped_dir
        self.resampled_dir = resampled_dir
        self.ctSlices_dir = ctSlices_dir
        self.segmentSlices_dir = segmentSlices_dir
        self.pngSlices_dir = pngSlices_dir
        
        
        self.fullCT = os.path.join(patient_dir, "fullCT.nii.gz")
        self.croppedCT = os.path.join(patient_dir, "croppedCT.nii.gz")
        self.resampledCT = os.path.join(patient_dir, "resampledCT.nii.gz")
        self.TS_LV = os.path.join(segments_dir, "heart_ventricle_left.nii.gz")
        self.TS_RV = os.path.join(segments_dir, "heart_ventricle_right.nii.gz")
        self.TS_LA = os.path.join(segments_dir, "heart_atrium_left.nii.gz")
        self.TS_RA = os.path.join(segments_dir, "heart_atrium_right.nii.gz")
        self.TS_MYO = os.path.join(segments_dir, "heart_myocardium.nii.gz")
        self.segments = {
            "LV": self.TS_LV,
            "RV": self.TS_RV,
            "LA": self.TS_LA,
            "RA": self.TS_RA,
            "MYO": self.TS_MYO
        }
        self.resampled_LV = os.path.join(resampled_dir, "LV.nii.gz")
        self.resampled_RV = os.path.join(resampled_dir, "RV.nii.gz")
        self.resampled_LA = os.path.join(resampled_dir, "LA.nii.gz")
        self.resampled_RA = os.path.join(resampled_dir, "RA.nii.gz")
        self.resampled_MYO = os.path.join(resampled_dir, "MYO.nii.gz")
        self.heart_mask = os.path.join(segments_dir, "heart_mask.nii.gz")
        
        
        
        
        