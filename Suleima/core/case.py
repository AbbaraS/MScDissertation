



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
	def __init__(self, path):
		"""
		Initializes a NiftiVolume object from a path.
		  """
		self.path = path
		if os.path.exists(path):
			self.obj = nib.load(path)
		else:
			self.obj = None

	@property
	def data(self):
		return self.obj.get_fdata()

	@property
	def shape(self):
		return self.obj.shape

	@property
	def affine(self):
		return self.obj.affine

	@property
	def header(self):
		return self.obj.header

	@property
	def spacing(self):
		return self.obj.header.get_zooms()

	@property
	def origin(self):
		return self.affine[:3, 3]

	def save(self, path=None):
		nib.save(self.obj, path)

	@classmethod
	def init_from_array(cls, array, affine, header, path=None):
		'''
		initialises object from array and affine.
		This is a class method, not an instance method.

		It receives the class itself (cls) instead of an instance.
		This creates a new instance without calling __init__
		which avoids the FileNotFoundError check in your constructor 
		that expects a file to exist at a path.
		'''
		vol = cls.__new__(cls)
		# Ensure header shape matches array shape
		header.set_data_shape(array.shape)
		vol.obj = nib.Nifti1Image(array, affine, header)
		if path:
			vol.path = path
			nib.save(vol.obj, path)
		else:
			vol.path = None
		return vol


############################################
# CARDIAC CT CLASS
'''
segmentMap = {
				"LV": "heart_ventricle_left.nii.gz",
				"LA": "heart_atrium_left.nii.gz",
				"RV": "heart_ventricle_right.nii.gz",
				"RA": "heart_atrium_right.nii.gz",
				"MYO": "heart_myocardium.nii.gz"
			}

'''
############################################
class Case:
	def __init__(self, caseID):
		self.caseID = caseID
		self.casePath = f"data/cases/{caseID}"
		self.missing = []
		self.log = []

	def log_message(self, msg):
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"[{timestamp}] {msg}"
		#print(entry)
		self.log.append(entry)
		with open(os.path.join(self.casePath, "info.txt"), "a") as f:
			f.write(entry + "\n")

	def load_nifti(self, filename):
		path = os.path.join(self.casePath, filename)
		try:
			vol = NiftiVolume(path)
			#self.log_message(f"Loaded {filename}: shape={vol.shape}, spacing={vol.spacing}")
			return vol
		except FileNotFoundError:
			#self.log_message(f"Missing: {filename}")
			self.missing.append(filename)
			return None

	def CT_exists(self, type="full"):
		# types = ['full', 'cropped', 'resampled']
		return os.path.exists(os.path.join(self.casePath, f"{type}CT.nii.gz"))

	def segment_exist(self, folder, name):
		# folder = 'cropped' or 'resampled'
		# name = 'LV', 'LA', 'RV', 'RA', 'MYO',  'heart_mask'
		return os.path.exists(os.path.join(self.casePath, folder, f"{name}.nii.gz"))


	@property
	def fullCT(self):
		if not self.CT_exists("full"):
			self.log_message("Full CT not found.")
			self.missing.append("fullCT.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "fullCT.nii.gz"))

	@fullCT.setter
	def fullCT(self, value):
		if isinstance(value, NiftiVolume):
			self.fullCT = value
		else:
			raise ValueError("fullCT must be a NiftiVolume object.")


	@property
	def LVtotalseg(self):
		if not self.segment_exist("segments", "heart_ventricle_left"):
			self.log_message("LV total segment not found.")
			self.missing.append("segments/heart_ventricle_left.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "segments", "heart_ventricle_left.nii.gz"))
	@LVtotalseg.setter
	def LVtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self.LVtotalseg = value
		else:
			raise ValueError("LVtotalseg must be a NiftiVolume object.")

	@property
	def LAtotalseg(self):
		if not self.segment_exist("segments", "heart_atrium_left"):
			self.log_message("LA total segment not found.")
			self.missing.append("segments/heart_atrium_left.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "segments", "heart_atrium_left.nii.gz"))
	@LAtotalseg.setter
	def LAtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self.LAtotalseg = value
		else:
			raise ValueError("LAtotalseg must be a NiftiVolume object.")

	@property
	def RVtotalseg(self):
		if not self.segment_exist("segments", "heart_ventricle_right"):
			self.log_message("RV total segment not found.")
			self.missing.append("segments/heart_ventricle_right.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "segments", "heart_ventricle_right.nii.gz"))
	@RVtotalseg.setter
	def RVtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self.RVtotalseg = value
		else:
			raise ValueError("RVtotalseg must be a NiftiVolume object.")

	@property
	def RAtotalseg(self):
		if not self.segment_exist("segments", "heart_atrium_right"):
			self.log_message("RA total segment not found.")
			return None
		return NiftiVolume(os.path.join(self.casePath, "segments", "heart_atrium_right.nii.gz"))
	@RAtotalseg.setter
	def RAtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self.RAtotalseg = value
		else:
			raise ValueError("RAtotalseg must be a NiftiVolume object.")

	@property
	def MYOtotalseg(self):
		if not self.segment_exist("segments", "heart_myocardium"):
			self.log_message("MYO total segment not found.")
			self.missing.append("segments/heart_myocardium.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "segments", "heart_myocardium.nii.gz"))
	@MYOtotalseg.setter
	def MYOtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self.MYOtotalseg = value
		else:
			raise ValueError("MYOtotalseg must be a NiftiVolume object.")

	@property
	def totalsegs(self):
		return {k: v for k, v in {
				"LV": self.LVtotalseg,
				"LA": self.LAtotalseg,
				"RV": self.RVtotalseg,
				"RA": self.RAtotalseg,
				"MYO": self.MYOtotalseg
			}.items() if v is not None}
	@totalsegs.setter
	def totalsegs(self, segments):
		if not isinstance(segments, dict):
			raise ValueError("totalsegs must be a dictionary.")

		for key, volume in segments.items():
			if not isinstance(volume, NiftiVolume):
				raise ValueError(f"Segment '{key}' must be a NiftiVolume object.")
	
			if key == "LV":
				self.LVtotalseg = volume
			elif key == "LA":
				self.LAtotalseg = volume
			elif key == "RV":
				self.RVtotalseg = volume
			elif key == "RA":
				self.RAtotalseg = volume
			elif key == "MYO":
				self.MYOtotalseg = volume
			else:
				raise KeyError(f"Unknown segment key: {key}")

	@property
	def croppedCT(self):
		if not self.CT_exists("cropped"):
			self.log_message("croppedCT not found.")
			self.missing.append("croppedCT.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "croppedCT.nii.gz"))
	@fullCT.setter
	def fullCT(self, value):
		if isinstance(value, NiftiVolume):
			self.fullCT = value
		else:
			raise ValueError("fullCT must be a NiftiVolume object.")

	@property
	def LVcropped(self):
		if not self.segment_exist("cropped", "LV"):
			self.log_message("LV cropped segment not found.")
			self.missing.append("cropped/LV.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "cropped", "LV.nii.gz"))

	@LVcropped.setter
	def LVcropped(self, value):
		self.LVcropped = value

	@property
	def LAcropped(self):
		if not self.segment_exist("cropped", "LA"):
			self.log_message("LA cropped segment not found.")
			self.missing.append("cropped/LA.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "cropped", "LA.nii.gz"))
	@LAcropped.setter
	def LAcropped(self, value):
		if isinstance(value, NiftiVolume):
			self.LAcropped = value
		else:
			raise ValueError("LAcropped must be a NiftiVolume object.")

	@property
	def RVcropped(self):
		if not self.segment_exist("cropped", "RV"):
			self.log_message("RV cropped segment not found.")
			self.missing.append("cropped/RV.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "cropped", "RV.nii.gz"))
	@RVcropped.setter
	def RVcropped(self, value):
		if isinstance(value, NiftiVolume):
			self.RVcropped = value
		else:
			raise ValueError("RVcropped must be a NiftiVolume object.")

	@property
	def RAcropped(self):
		if not self.segment_exist("cropped", "RA"):
			self.log_message("RA cropped segment not found.")
			self.missing.append("cropped/RA.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "cropped", "RA.nii.gz"))
	@RAcropped.setter
	def RAcropped(self, value):
		if isinstance(value, NiftiVolume):
			self.RAcropped = value
		else:
			raise ValueError("RAcropped must be a NiftiVolume object.")


	@property
	def MYOcropped(self):
		if not self.segment_exist("cropped", "MYO"):
			self.log_message("MYO cropped segment not found.")
			self.missing.append("cropped/MYO.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "cropped", "MYO.nii.gz"))
	@MYOcropped.setter
	def MYOcropped(self, value):
		if isinstance(value, NiftiVolume):
			self.MYOcropped = value
		else:
			raise ValueError("MYOcropped must be a NiftiVolume object.")

	@property
	def croppedsegs(self):
		return {k: v for k, v in {
			"LV": self.LVcropped,
			"LA": self.LAcropped,
			"RV": self.RVcropped,
			"RA": self.RAcropped,
			"MYO": self.MYOcropped
		}.items() if v is not None}
	@croppedsegs.setter
	def croppedsegs(self, segments):
		if not isinstance(segments, dict):
			raise ValueError("croppedsegs must be a dictionary.")

		for key, volume in segments.items():
			if not isinstance(volume, NiftiVolume):
				raise ValueError(f"Segment '{key}' must be a NiftiVolume object.")
	
			if key == "LV":
				self.LVcropped = volume
			elif key == "LA":
				self.LAcropped = volume
			elif key == "RV":
				self.RVcropped = volume
			elif key == "RA":
				self.RAcropped = volume
			elif key == "MYO":
				self.MYOcropped = volume
			else:
				raise KeyError(f"Unknown segment key: {key}")

	@property
	def cropped_mask(self):
		if not self.segment_exist("cropped", "heart_mask"):
			self.log_message("Cropped heart mask not found.")
			self.missing.append("cropped/heart_mask.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "cropped", "heart_mask.nii.gz"))
	@cropped_mask.setter
	def cropped_mask(self, value):
		if isinstance(value, NiftiVolume):
			self.cropped_mask = value
		else:
			raise ValueError("cropped_mask must be a NiftiVolume object.")

	@property
	def resampledCT(self):
		if not self.CT_exists("resampled"):
			self.log_message("resampledCT not found.")
			self.missing.append("resampledCT.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampledCT.nii.gz"))
	@resampledCT.setter
	def resampledCT(self, value):
		if isinstance(value, NiftiVolume):
			self.resampledCT = value
		else:
			raise ValueError("resampledCT must be a NiftiVolume object.")

	@property
	def LVresampled(self):
		if not self.segment_exist("resampled", "LV"):
			self.log_message("LV resampled segment not found.")
			self.missing.append("resampled/LV.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampled", "LV.nii.gz"))
	@LVresampled.setter
	def LVresampled(self, value):
		if isinstance(value, NiftiVolume):
			self.LVresampled = value
		else:
			raise ValueError("LVresampled must be a NiftiVolume object.")

	@property
	def RAresampled(self):
		if not self.segment_exist("resampled", "RA"):
			self.log_message("RA resampled segment not found.")
			self.missing.append("resampled/RA.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampled", "RA.nii.gz"))
	@RAresampled.setter
	def RAresampled(self, value):
		if isinstance(value, NiftiVolume):
			self.RAresampled = value
		else:
			raise ValueError("RAresampled must be a NiftiVolume object.")


	@property
	def RVresampled(self):
		if not self.segment_exist("resampled", "RV"):
			self.log_message("RV resampled segment not found.")
			self.missing.append("resampled/RV.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampled", "RV.nii.gz"))
	@RVresampled.setter
	def RVresampled(self, value):
		if isinstance(value, NiftiVolume):
			self.RVresampled = value
		else:
			raise ValueError("RVresampled must be a NiftiVolume object.")

	@property
	def LAresampled(self):	
		if not self.segment_exist("resampled", "LA"):
			self.log_message("LA resampled segment not found.")
			self.missing.append("resampled/LA.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampled", "LA.nii.gz"))
	@LAresampled.setter
	def LAresampled(self, value):
		if isinstance(value, NiftiVolume):
			self.LAresampled = value
		else:
			raise ValueError("LAresampled must be a NiftiVolume object.")

	@property
	def MYOresampled(self):
		if not self.segment_exist("resampled", "MYO"):
			self.log_message("MYO resampled segment not found.")
			self.missing.append("resampled/MYO.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampled", "MYO.nii.gz"))
	@MYOresampled.setter
	def MYOresampled(self, value):
		if isinstance(value, NiftiVolume):
			self.MYOresampled = value
		else:
			raise ValueError("MYOresampled must be a NiftiVolume object.")


	@property
	def resampledsegs(self):
		return {k: v for k, v in {
			"LV": self.LVresampled,
			"LA": self.LAresampled,
			"RV": self.RVresampled,
			"RA": self.RAresampled,
			"MYO": self.MYOresampled
		}.items() if v is not None}
	@resampledsegs.setter
	def resampledsegs(self, segments):
		if not isinstance(segments, dict):
			raise ValueError("resampledsegs must be a dictionary.")

		for key, volume in segments.items():
			if not isinstance(volume, NiftiVolume):
				raise ValueError(f"Segment '{key}' must be a NiftiVolume object.")
	
			if key == "LV":
				self.LVresampled = volume
			elif key == "LA":
				self.LAresampled = volume
			elif key == "RV":
				self.RVresampled = volume
			elif key == "RA":
				self.RAresampled = volume
			elif key == "MYO":
				self.MYOresampled = volume
			else:
				raise KeyError(f"Unknown segment key: {key}")


	@property
	def resampled_mask(self):
		if not self.segment_exist("resampled", "heart_mask"):
			self.log_message("Resampled heart mask not found.")
			self.missing.append("resampled/heart_mask.nii.gz")
			return None
		return NiftiVolume(os.path.join(self.casePath, "resampled", "heart_mask.nii.gz"))
	@resampled_mask.setter
	def resampled_mask(self, value):
		if isinstance(value, NiftiVolume):
			self.resampled_mask = value
		else:
			raise ValueError("resampled_mask must be a NiftiVolume object.")



	def create_mask(self, segments, path):
		if segments:
			seg_arrays = [seg.data.astype(bool) for seg in segments.values()]
			array = np.any(seg_arrays, axis=0).astype(np.uint8)
			affine = segments["LV"].affine
			header = segments["LV"].header
			# Use metadata from one of the cropped segmentations to create new NIfTI
			return NiftiVolume.init_from_array(array, affine, header, path)
		else:
			self.log_message("No segments found to combine.")
			return None


	def get_bounding_box(self):
		masks = [seg.data > 0 for seg in self.totalsegs.values() if seg]
		if not masks:
			self.log_message(f"No bounding box could be determined. \nmask = {masks}")
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
			#self.log_message("Resampled volumes exist. Skipping resampling.")
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

	def get_LV_indices(self):
		lv = self.resampledsegs["LV"].data
		sagittal_indices = np.where(np.any(lv > 0, axis=(1, 2)))[0]
		coronal_indices = np.where(np.any(lv > 0, axis=(0, 2)))[0]
		axial_indices = np.where(np.any(lv > 0, axis=(0, 1)))[0]
		return {
			"sag": sagittal_indices,
			"cor": coronal_indices,
			"axi": axial_indices
		}

	def get_three_slices_within(indices):
		indices = sorted(indices)
		if len(indices) < 3:
			return list(indices)
		return [
			indices[len(indices) // 4],
			indices[len(indices) // 2],
			indices[3 * len(indices) // 4]
		]
	#view = ['sag', 'cor', 'axi']
	def select_slices(self):
		lv_indices = self.get_LV_indices()
		print(f"LV indices: {lv_indices['sag']}")
		slices = {
			"sagittal": self.get_three_slices_within(lv_indices["sag"]),
			"coronal": self.get_three_slices_within(lv_indices["cor"]),
			"axial": self.get_three_slices_within(lv_indices["axi"])
		}
		self.log_message(f"Selected slices: {slices}")



def crop_bbox(case, segments):
	masks = [seg.data > 0 for seg in segments.values() if seg]
	if not masks:
		case.log_message(f"No bounding box determined. \nmask = {masks}")
		return None
	combined = np.any(np.stack(masks), axis=0)
	coords = np.array(np.where(combined))
	x0, y0, z0 = coords.min(axis=1)
	x1, y1, z1 = coords.max(axis=1) + 1
	case.log_message(f"Cropping bbox: \nX=({x0}:{x1}), Y=({y0}:{y1}), Z=({z0}:{z1})")
	return slice(x0, x1), slice(y0, y1), slice(z0, z1)

def trim_bbox(case, segments):
	mask = np.sum([(s.data > 0).astype(np.uint8) for s in segments.values() if s], axis=0) > 0
	x = np.any(mask, axis=(1,2))
	y = np.any(mask, axis=(0,2))
	z = np.any(mask, axis=(0,1))
	if np.all(x) and np.all(y) and np.all(z):
		case.log_message("No empty slices to trim.")
		return
	x0, x1 = np.where(x)[0][[0, -1]]
	y0, y1 = np.where(y)[0][[0, -1]]
	z0, z1 = np.where(z)[0][[0, -1]]
	case.log_message(f"Trimming bbox: \nX=({x0}:{x1+1})", 
                  f"Y=({y0}:{y1+1})", 
                  f"Z=({z0}:{z1+1})")
	return slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1)
	


def crop_volume(vol, bbox):
	return NiftiVolume.init_from_array(vol.data[bbox], 
										vol.affine, 
										vol.header)


def crop(case):
	#files =["croppedCT", "LV", "LA", "RV", "RA", "MYO", "heart_mask"]
	bbox = case.get_bounding_box()
	ct = crop_volume(case.fullCT, bbox)
	lv = crop_volume(case.LVtotalseg, bbox)
	la = crop_volume(case.LAtotalseg, bbox)
	rv = crop_volume(case.RVtotalseg, bbox)
	ra = crop_volume(case.RAtotalseg, bbox)
	myo = crop_volume(case.MYOtotalseg, bbox)
	segs={
		"lv": lv,
		"la": la,
		"rv": rv,
		"ra": ra,
		"myo": myo
	}
	trim_bbox = trim_bbox(case, segs)
	if not trim_bbox:
		case.log_message("No trimming needed. saving with Cropping bbox.")
		case.CTcropped = NiftiVolume.init_from_array(ct.data, ct.affine, ct.header, os.path.join(case.casePath, "croppedCT.nii.gz"))
		case.LVcropped = NiftiVolume.init_from_array(lv.data, lv.affine, lv.header, os.path.join(case.casePath, "cropped", "LV.nii.gz"))
		case.LAcropped = NiftiVolume.init_from_array(la.data, la.affine, la.header, os.path.join(case.casePath, "cropped", "LA.nii.gz"))
		case.RVcropped = NiftiVolume.init_from_array(rv.data, rv.affine, rv.header, os.path.join(case.casePath, "cropped", "RV.nii.gz"))
		case.RAcropped = NiftiVolume.init_from_array(ra.data, ra.affine, ra.header, os.path.join(case.casePath, "cropped", "RA.nii.gz"))
		case.MYOcropped = NiftiVolume.init_from_array(myo.data, myo.affine, myo.header, os.path.join(case.casePath, "cropped", "MYO.nii.gz"))
	else:
		case.CTcropped = NiftiVolume.init_from_array(ct.data[trim_bbox], ct.affine, ct.header, os.path.join(case.casePath, "croppedCT.nii.gz"))
		case.LVcropped = NiftiVolume.init_from_array(lv.data[trim_bbox], lv.affine, lv.header, os.path.join(case.casePath, "cropped", "LV.nii.gz"))
		case.LAcropped = NiftiVolume.init_from_array(la.data[trim_bbox], la.affine, la.header, os.path.join(case.casePath, "cropped", "LA.nii.gz"))
		case.RVcropped = NiftiVolume.init_from_array(rv.data[trim_bbox], rv.affine, rv.header, os.path.join(case.casePath, "cropped", "RV.nii.gz"))
		case.RAcropped = NiftiVolume.init_from_array(ra.data[trim_bbox], ra.affine, ra.header, os.path.join(case.casePath, "cropped", "RA.nii.gz"))
		case.MYOcropped = NiftiVolume.init_from_array(myo.data[trim_bbox], myo.affine, myo.header, os.path.join(case.casePath, "cropped", "MYO.nii.gz"))	
	
	if not case.cropped_mask:
		case.cropped_mask = case.create_mask(case.croppedsegs, os.path.join(case.casePath, "cropped", "heart_mask.nii.gz"))

