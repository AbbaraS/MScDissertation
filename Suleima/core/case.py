



import os
from unittest import case
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage
from datetime import datetime
from nibabel.orientations import aff2axcodes

import matplotlib.pyplot as plt

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
		#self.data = self.image.get_fdata()

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

	def show_scrollable(self, axis=2):
		"""Displays a scrollable image viewer for the 3D volume along a given axis."""
		data = np.moveaxis(self.data, axis, 0)

		class IndexTracker:
			def __init__(self, ax, data):
				self.ax = ax
				self.data = data
				self.slices = data.shape[0]
				self.ind = self.slices // 2

				self.im = ax.imshow(self.data[self.ind, :, :], cmap="gray")
				self.update()

			def on_scroll(self, event):
				if event.button == 'up':
					self.ind = (self.ind + 1) % self.slices
				elif event.button == 'down':
					self.ind = (self.ind - 1) % self.slices
				self.update()

			def update(self):
				self.im.set_data(self.data[self.ind, :, :])
				self.ax.set_title(f"Slice {self.ind+1}/{self.slices}")
				self.im.axes.figure.canvas.draw()

		fig, ax = plt.subplots(1, 1)
		tracker = IndexTracker(ax, data)
		fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
		plt.show()

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
		self.segmentMap = {
				"LV": "heart_ventricle_left.nii.gz",
				"LA": "heart_atrium_left.nii.gz",
				"RV": "heart_ventricle_right.nii.gz",
				"RA": "heart_atrium_right.nii.gz",
				"MYO": "heart_myocardium.nii.gz"
			}
		self.fullCT = self.load_nifti("fullCT.nii.gz")
		self.totalsegs = self.load_TSsegments()
		self.croppedCT = None
		self.croppedsegs = {}
		self.resampledCT = None
		self.resampledsegs = {}
		self.cropped_mask= None
		self.resampled_mask = None

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

	def load_TSsegments(self):
		segs = {}
		for name, filename in self.segmentMap.items():
			path = os.path.join(self.casePath, "segments", filename)
			try:
				segs[name] = NiftiVolume(path, case=self)
			except FileNotFoundError:
				self.log_message(f"Missing segment: {name} in segments")
		return segs

	def load_segments(self, folder):
		segs = {}
		for name in self.segmentNames:
			path = os.path.join(self.casePath, folder, f"{name}.nii.gz")
			try:
				segs[name] = NiftiVolume(path, case=self)
			except FileNotFoundError:
				self.log_message(f"Missing segment: {name} in {folder}")
		return segs

	def croppedCT_exists(self):
		return os.path.exists(os.path.join(self.casePath, "croppedCT.nii.gz"))

	def resampledCT_exists(self):
		return os.path.exists(os.path.join(self.casePath, "resampledCT.nii.gz"))

	def cropped_heart_mask_exists(self):
		return os.path.exists(os.path.join(self.casePath, "cropped", "heart_mask.nii.gz"))
	
	def resampled_heart_mask_exists(self):
		return os.path.exists(os.path.join(self.casePath, "resampled", "heart_mask.nii.gz"))

	def crop_pipeline(self):
		if not self.croppedCT_exists():
			bbox = self.get_bounding_box()
			if bbox is None:
				self.log_message("No bounding box could be determined.")
				return

			self.croppedCT = self.crop_volume(self.fullCT, bbox, "croppedCT.nii.gz")

			for name, vol in self.totalsegs.items():
				cropped = self.crop_volume(vol, bbox, f"cropped/{name}.nii.gz")
				self.croppedsegs[name] = cropped
		else:
			self.log_message("Cropped volumes exist. Skipping cropping.")
			self.croppedCT = self.load_nifti("croppedCT.nii.gz")
			self.croppedsegs = self.load_segments("cropped")
		
		if not self.cropped_heart_mask_exists():
			self.cropped_mask = self.create_heart_mask(self.croppedsegs, os.path.join(self.casePath, "cropped", "heart_mask.nii.gz"))
		else:
			self.cropped_mask = self.load_nifti("cropped/heart_mask.nii.gz")
			
	def create_heart_mask(self, segments, path):
		if segments:
			seg_arrays = [seg.data.astype(bool) for seg in segments.values()]
			combined_array = np.any(seg_arrays, axis=0).astype(np.uint8)
			affine = segments["LV"].affine
			header = segments["LV"].image.header
			# Use metadata from one of the cropped segmentations to create new NIfTI
			
			combined_img = nib.Nifti1Image(combined_array, affine=affine, header=header)

			# Save combined mask
			nib.save(combined_img, path)

			self.log_message(f"Combined mask saved in {path}")
			return NiftiVolume(path, case=self)
		else:
			self.log_message("No segments found to combine.")


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
		self.log_message(f"Trimming empty slices to bbox: {bbox}")
		self.croppedCT = self.crop_volume(self.croppedCT, bbox, "croppedCT.nii.gz")
		for name, vol in self.croppedsegs.items():
			self.croppedsegs[name] = self.crop_volume(vol, bbox, f"cropped/{name}.nii.gz")
		self.cropped_mask = self.crop_volume(self.cropped_mask, bbox, "cropped/heart_mask.nii.gz")

	def resample_pipeline(self, target_spacing=[1.0]*3, target_shape=(64,64,64)):
		if not self.resampledCT_exists():
			self.resampledCT = self.resample_volume(self.croppedCT, target_spacing, target_shape, "resampledCT.nii.gz", linear=True)
			for name, vol in self.croppedsegs.items():
				self.resampledsegs[name] = self.resample_volume(vol, target_spacing, target_shape, f"resampled/{name}.nii.gz", linear=False)
		else:
			self.log_message("Resampled volumes exist. Skipping resampling.")
			self.resampledCT = self.load_nifti("resampledCT.nii.gz")
			self.resampledsegs = self.load_segments("resampled")

		
		if not self.resampled_heart_mask_exists():
			self.resampled_mask = self.create_heart_mask(self.resampledsegs, os.path.join(self.casePath, "resampled", "heart_mask.nii.gz"))
		else:
			self.resampled_mask = self.load_nifti("resampled/heart_mask.nii.gz")

  
	def resample_volume(self, vol, spacing, shape, filename, linear=True):
		img = sitk.GetImageFromArray(vol.data.transpose(2,1,0))
		img.SetSpacing([float(s) for s in vol.spacing[::-1]])  # explicit float conversion

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


  
  
  