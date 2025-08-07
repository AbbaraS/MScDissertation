

from core.Log import log

import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from datetime import datetime
from core.NiftiVolume import NiftiVolume

############################################
# NIFTI VOLUME WRAPPER
############################################



############################################
class Case:
	def __init__(self, caseID):
		self.caseID = caseID
		self.casePath = f"data/cases/{caseID}"
		self.missing = []
		self.log = []
		# Internal storage for dynamic assignment
		self._fullCT = None
		self._croppedCT = None
		self._resampledCT = None
		self._cropped_mask = None
		self._resampled_mask = None

		# Segments
		self._LVtotalseg = None
		self._LAtotalseg = None
		self._RVtotalseg = None
		self._RAtotalseg = None
		self._MYOtotalseg = None

		self._LVcropped = None
		self._LAcropped = None
		self._RVcropped = None
		self._RAcropped = None
		self._MYOcropped = None

		self._LVresampled = None
		self._LAresampled = None
		self._RVresampled = None
		self._RAresampled = None
		self._MYOresampled = None
		self.slices = None
		os.makedirs(os.path.join(self.casePath, "cropped"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "resampled"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "ctSlices"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "maskSlices"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "pngSlices"), exist_ok=True)
	# ==== Full CT ====

	@property
	def fullCT(self):
		if self._fullCT is not None:
			return self._fullCT
		path = os.path.join(self.casePath, "fullCT.nii.gz")
		if not self.CT_exists("full"):
			log("Full CT not found.")

			return None
		self._fullCT = NiftiVolume(path)
		return self._fullCT
	@fullCT.setter
	def fullCT(self, value):
		if isinstance(value, NiftiVolume):
			self._fullCT = value
			#value.save()
		else:
			raise ValueError(f"fullCT must be a NiftiVolume object. Got {type(value)}")

	# ==== Total Segment Properties ====

	@property
	def LVtotalseg(self):
		if self._LVtotalseg is not None:
			return self._LVtotalseg
		path = os.path.join(self.casePath, "segments", "heart_ventricle_left.nii.gz")
		if not self.segment_exist("segments", "heart_ventricle_left"):
			log("LV total segment not found.")

			return None
		self._LVtotalseg = NiftiVolume(path)
		return self._LVtotalseg

	@LVtotalseg.setter
	def LVtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self._LVtotalseg = value
			 #value.save()
		else:
			raise ValueError("LVtotalseg must be a NiftiVolume object.")

	@property
	def LAtotalseg(self):
		if self._LAtotalseg is not None:
			return self._LAtotalseg
		path = os.path.join(self.casePath, "segments", "heart_atrium_left.nii.gz")
		if not self.segment_exist("segments", "heart_atrium_left"):
			log("LA total segment not found.")

			return None
		self._LAtotalseg = NiftiVolume(path)
		return self._LAtotalseg

	@LAtotalseg.setter
	def LAtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self._LAtotalseg = value
			 #value.save()
		else:
			raise ValueError("LAtotalseg must be a NiftiVolume object.")

	@property
	def RVtotalseg(self):
		if self._RVtotalseg is not None:
			return self._RVtotalseg
		path = os.path.join(self.casePath, "segments", "heart_ventricle_right.nii.gz")
		if not self.segment_exist("segments", "heart_ventricle_right"):
			log("RV total segment not found.")

			return None
		self._RVtotalseg = NiftiVolume(path)
		return self._RVtotalseg

	@RVtotalseg.setter
	def RVtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self._RVtotalseg = value
			 #value.save()
		else:
			raise ValueError("RVtotalseg must be a NiftiVolume object.")

	@property
	def RAtotalseg(self):
		if self._RAtotalseg is not None:
			return self._RAtotalseg
		path = os.path.join(self.casePath, "segments", "heart_atrium_right.nii.gz")
		if not self.segment_exist("segments", "heart_atrium_right"):
			log("RA total segment not found.")

			return None
		self._RAtotalseg = NiftiVolume(path)
		return self._RAtotalseg

	@RAtotalseg.setter
	def RAtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self._RAtotalseg = value
			 #value.save()
		else:
			raise ValueError("RAtotalseg must be a NiftiVolume object.")

	@property
	def MYOtotalseg(self):
		if self._MYOtotalseg is not None:
			return self._MYOtotalseg
		path = os.path.join(self.casePath, "segments", "heart_myocardium.nii.gz")
		if not self.segment_exist("segments", "heart_myocardium"):
			log("MYO total segment not found.")

			return None
		self._MYOtotalseg = NiftiVolume(path)
		return self._MYOtotalseg

	@MYOtotalseg.setter
	def MYOtotalseg(self, value):
		if isinstance(value, NiftiVolume):
			self._MYOtotalseg = value
			 #value.save()
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
				 #volume.save()
			elif key == "LA":
				self.LAtotalseg = volume
				 #volume.save()
			elif key == "RV":
				self.RVtotalseg = volume
				 #volume.save()
			elif key == "RA":
				self.RAtotalseg = volume
				 #volume.save()
			elif key == "MYO":
				self.MYOtotalseg = volume
				 #volume.save()
			else:
				raise KeyError(f"Unknown segment key: {key}")

	# ==== Cropped CT ====

	@property
	def croppedCT(self):
		if self._croppedCT is not None:
			return self._croppedCT
		path = os.path.join(self.casePath, "croppedCT.nii.gz")
		if not self.CT_exists("cropped"):
			log("croppedCT not found.")

			return None
		self._croppedCT = NiftiVolume(path)
		return self._croppedCT

	@croppedCT.setter
	def croppedCT(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedCT = value
			 #value.save()
		else:
			raise ValueError(f"croppedCT must be a NiftiVolume object. Got {type(value)}")


	# ==== Cropped Segments ====

	@property
	def LVcropped(self):
		if self._LVcropped is not None:
			return self._LVcropped
		path = os.path.join(self.casePath, "cropped", "LV.nii.gz")
		if not self.segment_exist("cropped", "LV"):
			log("LV cropped segment not found.")

			return None
		self._LVcropped = NiftiVolume(path)
		return self._LVcropped

	@LVcropped.setter
	def LVcropped(self, value):
		if isinstance(value, NiftiVolume):
			self._LVcropped = value
			 #value.save()
		else:
			raise ValueError("LVcropped must be a NiftiVolume object.")


	@property
	def LAcropped(self):
		if self._LAcropped is not None:
			return self._LAcropped
		path = os.path.join(self.casePath, "cropped", "LA.nii.gz")
		if not self.segment_exist("cropped", "LA"):
			log("LA cropped segment not found.")

			return None
		self._LAcropped = NiftiVolume(path)
		return self._LAcropped

	@LAcropped.setter
	def LAcropped(self, value):
		if isinstance(value, NiftiVolume):
			self._LAcropped = value
			 #value.save()
		else:
			raise ValueError("LAcropped must be a NiftiVolume object.")


	@property
	def RVcropped(self):
		if self._RVcropped is not None:
			return self._RVcropped
		path = os.path.join(self.casePath, "cropped", "RV.nii.gz")
		if not self.segment_exist("cropped", "RV"):
			log("RV cropped segment not found.")

			return None
		self._RVcropped = NiftiVolume(path)
		return self._RVcropped

	@RVcropped.setter
	def RVcropped(self, value):
		if isinstance(value, NiftiVolume):
			self._RVcropped = value
			 #value.save()
		else:
			raise ValueError("RVcropped must be a NiftiVolume object.")


	@property
	def RAcropped(self):
		if self._RAcropped is not None:
			return self._RAcropped
		path = os.path.join(self.casePath, "cropped", "RA.nii.gz")
		if not self.segment_exist("cropped", "RA"):
			log("RA cropped segment not found.")

			return None
		self._RAcropped = NiftiVolume(path)
		return self._RAcropped

	@RAcropped.setter
	def RAcropped(self, value):
		if isinstance(value, NiftiVolume):
			self._RAcropped = value
			 #value.save()
		else:
			raise ValueError("RAcropped must be a NiftiVolume object.")


	@property
	def MYOcropped(self):
		if self._MYOcropped is not None:
			return self._MYOcropped
		path = os.path.join(self.casePath, "cropped", "MYO.nii.gz")
		if not self.segment_exist("cropped", "MYO"):
			log("MYO cropped segment not found.")

			return None
		self._MYOcropped = NiftiVolume(path)
		return self._MYOcropped

	@MYOcropped.setter
	def MYOcropped(self, value):
		if isinstance(value, NiftiVolume):
			self._MYOcropped = value
			 #value.save()
		else:
			raise ValueError("MYOcropped must be a NiftiVolume object.")


	# ==== Cropped Heart Mask ====

	@property
	def cropped_mask(self):
		if self._cropped_mask is not None:
			return self._cropped_mask
		path = os.path.join(self.casePath, "cropped", "heart_mask.nii.gz")
		if not self.segment_exist("cropped", "heart_mask"):
			log("Cropped heart mask not found.")

			return None
		self._cropped_mask = NiftiVolume(path)
		return self._cropped_mask

	@cropped_mask.setter
	def cropped_mask(self, value):
		if isinstance(value, NiftiVolume):
			self._cropped_mask = value
			 #value.save()
		else:
			raise ValueError("cropped_mask must be a NiftiVolume object.")


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
				 #volume.save()
			elif key == "LA":
				self.LAcropped = volume
				 #volume.save()
			elif key == "RV":
				self.RVcropped = volume
				 #volume.save()
			elif key == "RA":
				self.RAcropped = volume
				 #volume.save()
			elif key == "MYO":
				self.MYOcropped = volume
				 #volume.save()
			else:
				raise KeyError(f"Unknown segment key: {key}")

	# ==== Resampled CT ====

	@property
	def resampledCT(self):
		if self._resampledCT is not None:
			return self._resampledCT
		path = os.path.join(self.casePath, "resampledCT.nii.gz")
		if not self.CT_exists("resampled"):
			log("resampledCT not found.")

			return None
		self._resampledCT = NiftiVolume(path)
		return self._resampledCT

	@resampledCT.setter
	def resampledCT(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledCT = value
			 #value.save()
		else:
			raise ValueError("resampledCT must be a NiftiVolume object.")


	# ==== Resampled Segments ====

	@property
	def LVresampled(self):
		if self._LVresampled is not None:
			return self._LVresampled
		path = os.path.join(self.casePath, "resampled", "LV.nii.gz")
		if not self.segment_exist("resampled", "LV"):
			log("LV resampled segment not found.")

			return None
		self._LVresampled = NiftiVolume(path)
		return self._LVresampled

	@LVresampled.setter
	def LVresampled(self, value):
		if isinstance(value, NiftiVolume):
			self._LVresampled = value
			 #value.save()
		else:
			raise ValueError("LVresampled must be a NiftiVolume object.")


	@property
	def LAresampled(self):
		if self._LAresampled is not None:
			return self._LAresampled
		path = os.path.join(self.casePath, "resampled", "LA.nii.gz")
		if not self.segment_exist("resampled", "LA"):
			log("LA resampled segment not found.")
			return None
		self._LAresampled = NiftiVolume(path)
		return self._LAresampled

	@LAresampled.setter
	def LAresampled(self, value):
		if isinstance(value, NiftiVolume):
			self._LAresampled = value
			 #value.save()
		else:
			raise ValueError("LA resampled must be a NiftiVolume object.")


	@property
	def RVresampled(self):
		if self._RVresampled is not None:
			return self._RVresampled
		path = os.path.join(self.casePath, "resampled", "RV.nii.gz")
		if not self.segment_exist("resampled", "RV"):
			log("RV resampled segment not found.")

			return None
		self._RVresampled = NiftiVolume(path)
		return self._RVresampled

	@RVresampled.setter
	def RVresampled(self, value):
		if isinstance(value, NiftiVolume):
			self._RVresampled = value
			 #value.save()
		else:
			raise ValueError("RVresampled must be a NiftiVolume object.")


	@property
	def RAresampled(self):
		if self._RAresampled is not None:
			return self._RAresampled
		path = os.path.join(self.casePath, "resampled", "RA.nii.gz")
		if not self.segment_exist("resampled", "RA"):
			log("RA resampled segment not found.")

			return None
		self._RAresampled = NiftiVolume(path)
		return self._RAresampled

	@RAresampled.setter
	def RAresampled(self, value):
		if isinstance(value, NiftiVolume):
			self._RAresampled = value
			 #value.save()
		else:
			raise ValueError("RAresampled must be a NiftiVolume object.")


	@property
	def MYOresampled(self):
		if self._MYOresampled is not None:
			return self._MYOresampled
		path = os.path.join(self.casePath, "resampled", "MYO.nii.gz")
		if not self.segment_exist("resampled", "MYO"):
			log("MYO resampled segment not found.")

			return None
		self._MYOresampled = NiftiVolume(path)
		return self._MYOresampled

	@MYOresampled.setter
	def MYOresampled(self, value):
		if isinstance(value, NiftiVolume):
			self._MYOresampled = value
			 #value.save()
		else:
			raise ValueError("MYOresampled must be a NiftiVolume object.")


	# ==== Resampled Heart Mask ====

	@property
	def resampled_mask(self):
		if self._resampled_mask is not None:
			return self._resampled_mask
		path = os.path.join(self.casePath, "resampled", "heart_mask.nii.gz")
		if not self.segment_exist("resampled", "heart_mask"):
			log("Resampled heart mask not found.")

			return None
		self._resampled_mask = NiftiVolume(path)
		return self._resampled_mask

	@resampled_mask.setter
	def resampled_mask(self, value):
		if isinstance(value, NiftiVolume):
			self._resampled_mask = value
			 #value.save()
		else:
			raise ValueError("resampled_mask must be a NiftiVolume object.")

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
				 #volume.save()
			elif key == "LA":
				self.LAresampled = volume
				 #volume.save()
			elif key == "RV":
				self.RVresampled = volume
				 #volume.save()
			elif key == "RA":
				self.RAresampled = volume
				 #volume.save()
			elif key == "MYO":
				self.MYOresampled = volume
				 #volume.save()
			else:
				raise KeyError(f"Unknown segment key: {key}")

	def log_message(self, msg):
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"[{timestamp}] {msg}"
		#print(entry)
		self.log.append(entry)
		with open(os.path.join(self.casePath, "info.txt"), "a") as f:
			f.write(entry + "\n")

	def CT_exists(self, type="full"):
		# types = ['full', 'cropped', 'resampled']
		return os.path.exists(os.path.join(self.casePath, f"{type}CT.nii.gz"))

	def segment_exist(self, folder, name):
		# folder = 'cropped' or 'resampled'
		# name = 'LV', 'LA', 'RV', 'RA', 'MYO',  'heart_mask'
		return os.path.exists(os.path.join(self.casePath, folder, f"{name}.nii.gz"))


	def create_mask(self, segments, path):
		if segments:
			seg_arrays = [seg.data.astype(bool) for seg in segments.values()]
			array = np.any(seg_arrays, axis=0).astype(np.uint8)
			affine = segments["LV"].affine
			header = segments["LV"].header
			# Use metadata from one of the cropped segmentations to create new NIfTI
			return NiftiVolume.init_from_array(array, affine, header, path)
		else:
			log("No segments found to combine.")
			return None


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
		return NiftiVolume.init_from_array(arr, affine, vol.header, path)


	def crop_volume(self, vol, bbox, filename):
		data = vol.data[bbox]
		path = os.path.join(self.casePath, filename)
		cropped_vol = NiftiVolume.init_from_array(data,
											vol.affine,
											vol.header,
											path)
		#cropped_vol.save()
		log(f"Cropped {path}")
		return cropped_vol






