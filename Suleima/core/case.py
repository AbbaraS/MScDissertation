import nibabel as nib
import os
import numpy as np



import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage
from datetime import datetime
from nibabel.orientations import aff2axcodes

############################################
# NIFTI VOLUME WRAPPER
############################################
class NiftiVolume:
    def __init__(self, path, case=None):
        self.path = path
        self.case = case
        if not os.path.exists(path):
            raise FileNotFoundError(f"NIfTI file not found: {path}")
        self.image = nib.load(path)

    @property
    def data(self):
        return self.image.get_fdata()

    @property
    def shape(self):
        return self.image.shape

    @property
    def affine(self):
        return self.image.affine

    @property
    def spacing(self):
        return self.image.header.get_zooms()

    @property
    def origin(self):
        return self.affine[:3, 3]

    def save(self, path=None):
        nib.save(self.image, path or self.path)
        if self.case:
            self.case.log_message(f"Saved: {path or self.path}")

    @classmethod
    def from_array(cls, array, affine, path, case=None):
        obj = cls.__new__(cls)
        obj.image = nib.Nifti1Image(array, affine)
        obj.path = path
        obj.case = case
        return obj


############################################
# CARDIAC CT CLASS
############################################
class CardiacCT:
    def __init__(self, caseID):
        self.caseID = caseID
        self.casePath = f"data/cases/{caseID}"
        self.segmentNames = ["LV", "LA", "RV", "RA", "MYO"]
        self.log = []
        self.missing = []

        self.fullCT = self.load_nifti("fullCT.nii.gz")
        self.totalsegs = self.load_segments("segments")
        self.croppedCT = None
        self.croppedsegs = {}
        self.resampledCT = None
        self.resampledsegs = {}

    def log_message(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        print(entry)
        self.log.append(entry)
        with open(os.path.join(self.casePath, "info.txt"), "a") as f:
            f.write(entry + "\n")

    def load_nifti(self, filename):
        path = os.path.join(self.casePath, filename)
        try:
            vol = NiftiVolume(path, case=self)
            self.log_message(f"Loaded {filename}: shape={vol.shape}, spacing={vol.spacing}")
            return vol
        except FileNotFoundError:
            self.log_message(f"Missing: {filename}")
            self.missing.append(filename)
            return None

    def load_segments(self, folder):
        segs = {}
        for name in self.segmentNames:
            path = os.path.join(self.casePath, folder, f"{name}.nii.gz")
            try:
                segs[name] = NiftiVolume(path, case=self)
            except FileNotFoundError:
                self.log_message(f"Missing segment: {name} in {folder}")
        return segs

    def cropped_exists(self):
        return os.path.exists(os.path.join(self.casePath, "croppedCT.nii.gz"))

    def resampled_exists(self):
        return os.path.exists(os.path.join(self.casePath, "resampledCT.nii.gz"))

    def crop_pipeline(self):
        bbox = self.get_bounding_box()
        if bbox is None:
            self.log_message("No bounding box could be determined.")
            return

        self.croppedCT = self.crop_volume(self.fullCT, bbox, "croppedCT.nii.gz")

        for name, vol in self.totalsegs.items():
            cropped = self.crop_volume(vol, bbox, f"cropped/{name}.nii.gz")
            self.croppedsegs[name] = cropped

    def crop_volume(self, vol, bbox, filename):
        cropped_data = vol.data[bbox]
        cropped_path = os.path.join(self.casePath, filename)
        cropped_vol = NiftiVolume.from_array(cropped_data, vol.affine, cropped_path, self)
        cropped_vol.save()
        return cropped_vol

    def get_bounding_box(self):
        masks = [seg.data > 0 for seg in self.totalsegs.values() if seg]
        if not masks:
            return None

        combined = np.any(np.stack(masks), axis=0)
        coords = np.array(np.where(combined))
        x0, y0, z0 = coords.min(axis=1)
        x1, y1, z1 = coords.max(axis=1) + 1
        self.log_message(f"Bounding box: x=({x0}:{x1}), y=({y0}:{y1}), z=({z0}:{z1})")
        return slice(x0, x1), slice(y0, y1), slice(z0, z1)

    def trim_empty_slices(self):
        mask = np.sum([(s.data > 0).astype(np.uint8) for s in self.croppedsegs.values() if s], axis=0) > 0
        x = np.any(mask, axis=(1,2))
        y = np.any(mask, axis=(0,2))
        z = np.any(mask, axis=(0,1))

        if np.all(x) and np.all(y) and np.all(z):
            self.log_message("No empty slices to trim.")
            return

        x0, x1 = np.where(x)[0][[0, -1]]
        y0, y1 = np.where(y)[0][[0, -1]]
        z0, z1 = np.where(z)[0][[0, -1]]
        bbox = slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1)

        self.croppedCT = self.crop_volume(self.croppedCT, bbox, "croppedCT.nii.gz")
        for name, vol in self.croppedsegs.items():
            self.croppedsegs[name] = self.crop_volume(vol, bbox, f"cropped/{name}.nii.gz")

    def resample_pipeline(self, target_spacing=[1.0]*3, target_shape=(64,64,64)):
        self.resampledCT = self.resample_volume(self.croppedCT, target_spacing, target_shape, "resampledCT.nii.gz", linear=True)
        for name, vol in self.croppedsegs.items():
            self.resampledsegs[name] = self.resample_volume(vol, target_spacing, target_shape, f"resampled/{name}.nii.gz", linear=False)

    def resample_volume(self, vol, spacing, shape, filename, linear=True):
        img = sitk.GetImageFromArray(vol.data.transpose(2,1,0))
        img.SetSpacing(vol.spacing[::-1])

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear if linear else sitk.sitkNearestNeighbor)
        resample.SetOutputSpacing(spacing[::-1])

        orig_size = np.array(img.GetSize(), dtype=np.int32)
        orig_spacing = np.array(img.GetSpacing())
        new_size = np.round(orig_size * (orig_spacing / np.array(spacing[::-1]))).astype(int).tolist()
        resample.SetSize(new_size)
        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputOrigin(img.GetOrigin())

        new_img = resample.Execute(img)
        arr = sitk.GetArrayFromImage(new_img).transpose(2,1,0)

        if shape != arr.shape:
            arr = scipy.ndimage.zoom(arr, [t/s for t, s in zip(shape, arr.shape)], order=1 if linear else 0)

        affine = np.eye(4)
        affine[:3, :3] = np.array(new_img.GetDirection()).reshape(3,3) * np.array(new_img.GetSpacing())[:, None]
        affine[:3, 3] = new_img.GetOrigin()

        path = os.path.join(self.casePath, filename)
        vol = NiftiVolume.from_array(arr, affine, path, self)
        vol.save()
        return vol

