

from core.Log import *

import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
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
		self._normalisedCT = None
		self._croppedMask = None
		self._resampledMask = None
		self._labelMask	= None
		# Segments
		self._totalsegLV  = None
		self._totalsegLA  = None
		self._totalsegRV  = None
		self._totalsegRA  = None
		self._totalsegMYO = None

		self._croppedLV = None
		self._croppedLA = None
		self._croppedRV = None
		self._croppedRA = None
		self._croppedMYO = None

		self._resampledLV = None
		self._resampledLA = None
		self._resampledRV = None
		self._resampledRA = None
		self._resampledMYO = None
		self.slices = None
		os.makedirs(os.path.join(self.casePath, "cropped"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "resampled"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "ctSlices"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "maskSlices"), exist_ok=True)
		os.makedirs(os.path.join(self.casePath, "pngSlices"), exist_ok=True)


		set_log(caseID)
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
		else:
			raise ValueError(f"fullCT must be a NiftiVolume object. Got {type(value)}")

	# ==== Total Segment Properties ====

	@property
	def totalsegLV(self):
		if self._totalsegLV is not None:
			return self._totalsegLV
		path = os.path.join(self.casePath, "segments", "heart_ventricle_left.nii.gz")
		if not self.segment_exist("segments", "heart_ventricle_left"):
			log("LV total segment not found.")

			return None
		self._totalsegLV = NiftiVolume(path)
		return self._totalsegLV

	@totalsegLV.setter
	def set_totalsegLV(self, value):
		if isinstance(value, NiftiVolume):
			self._totalsegLV = value
			 #value.save()
		else:
			raise ValueError("totalsegLV must be a NiftiVolume object.")

	@property
	def totalsegLA(self):
		if self._totalsegLA is not None:
			return self._totalsegLA
		path = os.path.join(self.casePath, "segments", "heart_atrium_left.nii.gz")
		if not self.segment_exist("segments", "heart_atrium_left"):
			log("LA total segment not found.")

			return None
		self._totalsegLA = NiftiVolume(path)
		return self._totalsegLA

	@totalsegLA.setter
	def totalsegLA(self, value):
		if isinstance(value, NiftiVolume):
			self._totalsegLA = value
			 #value.save()
		else:
			raise ValueError("totalsegLA must be a NiftiVolume object.")

	@property
	def totalsegRV(self):
		if self._totalsegRV is not None:
			return self._totalsegRV
		path = os.path.join(self.casePath, "segments", "heart_ventricle_right.nii.gz")
		if not self.segment_exist("segments", "heart_ventricle_right"):
			log("RV total segment not found.")

			return None
		self._totalsegRV = NiftiVolume(path)
		return self._totalsegRV

	@totalsegRV.setter
	def set_totalsegRV(self, value):
		if isinstance(value, NiftiVolume):
			self._totalsegRV = value
			 #value.save()
		else:
			raise ValueError("totalsegRV must be a NiftiVolume object.")

	@property
	def totalsegRA(self):
		if self._totalsegRA is not None:
			return self._totalsegRA
		path = os.path.join(self.casePath, "segments", "heart_atrium_right.nii.gz")
		if not self.segment_exist("segments", "heart_atrium_right"):
			log("RA total segment not found.")

			return None
		self._totalsegRA = NiftiVolume(path)
		return self._totalsegRA

	@totalsegRA.setter
	def totalsegRA(self, value):
		if isinstance(value, NiftiVolume):
			self._totalsegRA = value
			 #value.save()
		else:
			raise ValueError("totalsegRA must be a NiftiVolume object.")

	@property
	def totalsegMYO(self):
		if self._totalsegMYO is not None:
			return self._totalsegMYO
		path = os.path.join(self.casePath, "segments", "heart_myocardium.nii.gz")
		if not self.segment_exist("segments", "heart_myocardium"):
			log("MYO total segment not found.")

			return None
		self._totalsegMYO = NiftiVolume(path)
		return self._totalsegMYO

	@totalsegMYO.setter
	def totalsegMYO(self, value):
		if isinstance(value, NiftiVolume):
			self._totalsegMYO = value
			 #value.save()
		else:
			raise ValueError("totalsegMYO must be a NiftiVolume object.")


	@property
	def totalsegs(self):
		return {k: v for k, v in {
				"LV": self.totalsegLV,
				"LA": self.totalsegLA,
				"RV": self.totalsegRV,
				"RA": self.totalsegRA,
				"MYO": self.totalsegMYO
			}.items() if v is not None}

	@totalsegs.setter
	def totalsegs(self, segments):
		if not isinstance(segments, dict):
			raise ValueError("totalsegs must be a dictionary.")

		for key, volume in segments.items():
			if not isinstance(volume, NiftiVolume):
				raise ValueError(f"Segment '{key}' must be a NiftiVolume object.")

			if key == "LV":
				self.totalsegLV = volume
				 #volume.save()
			elif key == "LA":
				self.totalsegLA = volume
				 #volume.save()
			elif key == "RV":
				self.totalsegRV = volume
				 #volume.save()
			elif key == "RA":
				self.totalsegRA = volume
				 #volume.save()
			elif key == "MYO":
				self.totalsegMYO = volume
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
	def croppedLV(self):
		if self._croppedLV is not None:
			return self._croppedLV
		path = os.path.join(self.casePath, "cropped", "LV.nii.gz")
		if not self.segment_exist("cropped", "LV"):
			log("LV cropped segment not found.")

			return None
		self._croppedLV = NiftiVolume(path)
		return self._croppedLV

	@croppedLV.setter
	def croppedLV(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedLV = value
			 #value.save()
		else:
			raise ValueError("croppedLV must be a NiftiVolume object.")


	@property
	def croppedLA(self):
		if self._croppedLA is not None:
			return self._croppedLA
		path = os.path.join(self.casePath, "cropped", "LA.nii.gz")
		if not self.segment_exist("cropped", "LA"):
			log("LA cropped segment not found.")

			return None
		self._croppedLA = NiftiVolume(path)
		return self._croppedLA

	@croppedLA.setter
	def croppedLA(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedLA = value
			 #value.save()
		else:
			raise ValueError("croppedLA must be a NiftiVolume object.")


	@property
	def croppedRV(self):
		if self._croppedRV is not None:
			return self._croppedRV
		path = os.path.join(self.casePath, "cropped", "RV.nii.gz")
		if not self.segment_exist("cropped", "RV"):
			log("RV cropped segment not found.")

			return None
		self._croppedRV = NiftiVolume(path)
		return self._croppedRV

	@croppedRV.setter
	def croppedRV(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedRV = value
			 #value.save()
		else:
			raise ValueError("croppedRV must be a NiftiVolume object.")


	@property
	def croppedRA(self):
		if self._croppedRA is not None:
			return self._croppedRA
		path = os.path.join(self.casePath, "cropped", "RA.nii.gz")
		if not self.segment_exist("cropped", "RA"):
			log("RA cropped segment not found.")

			return None
		self._croppedRA = NiftiVolume(path)
		return self._croppedRA

	@croppedRA.setter
	def croppedRA(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedRA = value
			 #value.save()
		else:
			raise ValueError("croppedRA must be a NiftiVolume object.")


	@property
	def croppedMYO(self):
		if self._croppedMYO is not None:
			return self._croppedMYO
		path = os.path.join(self.casePath, "cropped", "MYO.nii.gz")
		if not self.segment_exist("cropped", "MYO"):
			log("MYO cropped segment not found.")

			return None
		self._croppedMYO = NiftiVolume(path)
		return self._croppedMYO

	@croppedMYO.setter
	def croppedMYO(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedMYO = value
			 #value.save()
		else:
			raise ValueError("croppedMYO must be a NiftiVolume object.")


	# ==== Cropped Heart Mask ====

	@property
	def croppedMask(self):
		if self._croppedMask is not None:
			return self._croppedMask
		path = os.path.join(self.casePath, "cropped", "heart_mask.nii.gz")
		if not self.segment_exist("cropped", "heart_mask"):
			log("Cropped heart mask not found.")

			return None
		self._croppedMask = NiftiVolume(path)
		return self._croppedMask

	@croppedMask.setter
	def croppedMask(self, value):
		if isinstance(value, NiftiVolume):
			self._croppedMask = value
			 #value.save()
		else:
			raise ValueError("croppedMask must be a NiftiVolume object.")


	@property
	def croppedsegs(self):
		return {k: v for k, v in {
			"LV": self.croppedLV,
			"LA": self.croppedLA,
			"RV": self.croppedRV,
			"RA": self.croppedRA,
			"MYO": self.croppedMYO
		}.items() if v is not None}

	@croppedsegs.setter
	def croppedsegs(self, segments):
		if not isinstance(segments, dict):
			raise ValueError("croppedsegs must be a dictionary.")

		for key, volume in segments.items():
			if not isinstance(volume, NiftiVolume):
				raise ValueError(f"Segment '{key}' must be a NiftiVolume object.")

			if key == "LV":
				self.croppedLV = volume
				 #volume.save()
			elif key == "LA":
				self.croppedLA = volume
				 #volume.save()
			elif key == "RV":
				self.croppedRV = volume
				 #volume.save()
			elif key == "RA":
				self.croppedRA = volume
				 #volume.save()
			elif key == "MYO":
				self.croppedMYO = volume
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
	def resampledLV(self):
		if self._resampledLV is not None:
			return self._resampledLV
		path = os.path.join(self.casePath, "resampled", "LV.nii.gz")
		if not self.segment_exist("resampled", "LV"):
			log("LV resampled segment not found.")

			return None
		self._resampledLV = NiftiVolume(path)
		return self._resampledLV

	@resampledLV.setter
	def resampledLV(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledLV = value
			 #value.save()
		else:
			raise ValueError("resampledLV must be a NiftiVolume object.")


	@property
	def resampledLA(self):
		if self._resampledLA is not None:
			return self._resampledLA
		path = os.path.join(self.casePath, "resampled", "LA.nii.gz")
		if not self.segment_exist("resampled", "LA"):
			log("LA resampled segment not found.")
			return None
		self._resampledLA = NiftiVolume(path)
		return self._resampledLA

	@resampledLA.setter
	def resampledLA(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledLA = value
			 #value.save()
		else:
			raise ValueError("LA resampled must be a NiftiVolume object.")


	@property
	def resampledRV(self):
		if self._resampledRV is not None:
			return self._resampledRV
		path = os.path.join(self.casePath, "resampled", "RV.nii.gz")
		if not self.segment_exist("resampled", "RV"):
			log("RV resampled segment not found.")

			return None
		self._resampledRV = NiftiVolume(path)
		return self._resampledRV

	@resampledRV.setter
	def resampledRV(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledRV = value
			 #value.save()
		else:
			raise ValueError("resampledRV must be a NiftiVolume object.")


	@property
	def resampledRA(self):
		if self._resampledRA is not None:
			return self._resampledRA
		path = os.path.join(self.casePath, "resampled", "RA.nii.gz")
		if not self.segment_exist("resampled", "RA"):
			log("RA resampled segment not found.")

			return None
		self._resampledRA = NiftiVolume(path)
		return self._resampledRA

	@resampledRA.setter
	def resampledRA(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledRA = value
			 #value.save()
		else:
			raise ValueError("resampledRA must be a NiftiVolume object.")


	@property
	def resampledMYO(self):
		if self._resampledMYO is not None:
			return self._resampledMYO
		path = os.path.join(self.casePath, "resampled", "MYO.nii.gz")
		if not self.segment_exist("resampled", "MYO"):
			log("MYO resampled segment not found.")

			return None
		self._resampledMYO = NiftiVolume(path)
		return self._resampledMYO

	@resampledMYO.setter
	def resampledMYO(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledMYO = value
			 #value.save()
		else:
			raise ValueError("resampledMYO must be a NiftiVolume object.")


	# ==== Resampled Heart Mask ====

	@property
	def resampledMask(self):
		if self._resampledMask is not None:
			return self._resampledMask
		path = os.path.join(self.casePath, "resampled", "heart_mask.nii.gz")
		if not self.segment_exist("resampled", "heart_mask"):
			log("Resampled heart mask not found.")

			return None
		self._resampledMask = NiftiVolume(path)
		return self._resampledMask

	@resampledMask.setter
	def resampledMask(self, value):
		if isinstance(value, NiftiVolume):
			self._resampledMask = value
			 #value.save()
		else:
			raise ValueError("resampledMask must be a NiftiVolume object.")

	@property
	def resampledsegs(self):
		return {k: v for k, v in {
			"LV": self.resampledLV,
			"LA": self.resampledLA,
			"RV": self.resampledRV,
			"RA": self.resampledRA,
			"MYO": self.resampledMYO
		}.items() if v is not None}

	@resampledsegs.setter
	def resampledsegs(self, segments):
		if not isinstance(segments, dict):
			raise ValueError("resampledsegs must be a dictionary.")

		for key, volume in segments.items():
			if not isinstance(volume, NiftiVolume):
				raise ValueError(f"Segment '{key}' must be a NiftiVolume object.")
			if key == "LV":
				self.resampledLV = volume
				 #volume.save()
			elif key == "LA":
				self.resampledLA = volume
				 #volume.save()
			elif key == "RV":
				self.resampledRV = volume
				 #volume.save()
			elif key == "RA":
				self.resampledRA = volume
				 #volume.save()
			elif key == "MYO":
				self.resampledMYO = volume
				 #volume.save()
			else:
				raise KeyError(f"Unknown segment key: {key}")


	@property
	def labelMask(self):
		if self._labelMask is not None:
			return self._labelMask
		path = os.path.join(self.casePath, "labelMask.nii.gz")
		if not os.path.exists(path):
			log("labelMask not found.")
			return None
		self._labelMask = NiftiVolume(path)
		return self._labelMask
	@labelMask.setter
	def labelMask(self, value):
		if isinstance(value, NiftiVolume):
			self._labelMask = value
		else:
			raise ValueError(f"labelMask must be a NiftiVolume object. Got {type(value)}")



	def load(self):
		"""Load the case data."""
		self.fullCT = self.load_volume("fullCT.nii.gz")
		self.croppedCT = self.load_volume("croppedCT.nii.gz")
		self.resampledCT = self.load_volume("resampledCT.nii.gz")
		self.totalsegs = {k: v for k, v in {
				"LV": self.load_volume("segments/heart_ventricle_left.nii.gz"),
				"LA": self.load_volume("segments/heart_atrium_left.nii.gz"),
				"RV": self.load_volume("segments/heart_ventricle_right.nii.gz"),
				"RA": self.load_volume("segments/heart_atrium_right.nii.gz"),
				"MYO": self.load_volume("segments/heart_myocardium.nii.gz")
			}.items() if v is not None}

	def load_volume(self, p):
		'''Load a volume from the case directory.'''
		path = os.path.join(self.casePath, p)
		if not os.path.exists(path):
			log(f"volume not found.")
			return None
		return NiftiVolume(path)

	def CT_exists(self, type="full"):
		# types = ['full', 'cropped', 'resampled']
		return os.path.exists(os.path.join(self.casePath, f"{type}CT.nii.gz"))

	def segment_exist(self, folder, name):
		# folder = 'cropped' or 'resampled'
		# name = 'LV', 'LA', 'RV', 'RA', 'MYO',  'heart_mask'
		return os.path.exists(os.path.join(self.casePath, folder, f"{name}.nii.gz"))


	def create_binary_mask(self, segments, path):
		if segments:
			seg_arrays = [seg.data.astype(bool) for seg in segments.values()]
			array = np.any(seg_arrays, axis=0).astype(np.uint8)
			affine = segments["LV"].affine
			# Use metadata from one of the cropped segmentations to create new NIfTI
			return NiftiVolume.init_from_array(array, affine, path)
		else:
			log("No segments found to combine.")
			return None

	def create_labelMask(self, segments, path):
		if segments and self.croppedCT is not None:
			mask = np.zeros_like(self.croppedCT.data, dtype=np.uint8)
			for name, val in segments.items():
				# LV: 1, LA: 2, RV: 3, RA: 4, MYO: 5
				print(f"ct shape: {self.croppedCT.data.shape} - segment {name} shape: {val.shape}")
				label = {"LV": 1, "LA": 2, "RV": 3, "RA": 4, "MYO": 5}.get(name, 0)
				mask[val.data > 0] = label
			NiftiVolume.init_from_array(mask, self.croppedCT.affine, path)
		else:
			log("No segments found to create label mask.")


	def onehot_mask(self, segments, path):
		"""Create a one-hot encoded mask from the segmentation volumes."""
		if not segments:
			log("No segments found for one-hot encoding.")
			return None

		# Get the shape from one of the segmentations
		shape = segments["LV"].shape
		num_classes = len(segments)
		print(f"classes: {num_classes}")
		# Create an empty one-hot encoded array
		onehot = np.zeros((*shape, num_classes), dtype=np.uint8)

		# Fill the one-hot encoded array
		for i, (key, seg) in enumerate(segments.items()):
			onehot[..., i] = (seg.data > 0).astype(np.uint8)

		# Save the one-hot encoded mask
		onehot_vol = NiftiVolume.init_from_array(onehot, segments["LV"].affine, path)
		return onehot_vol

	def oldresample_volume(self, vol, spacing, shape, filename, linear=True):
		print(f"vol.data.shape: {vol.data.shape} - ")

		img = sitk.GetImageFromArray(vol.data.transpose(2,1,0))
		img.SetSpacing([float(s) for s in vol.spacing[::-1]])  # explicit float conversion
		resample = sitk.ResampleImageFilter()
		resample.SetInterpolator(sitk.sitkLinear if linear else sitk.sitkNearestNeighbor)
		resample.SetOutputSpacing(spacing[::-1])
		orig_size = np.array(img.GetSize(), dtype=np.int32)
		orig_spacing = np.array(img.GetSpacing())
		new_size = np.round(orig_size * (orig_spacing / np.array(spacing[::-1]))).astype(int).tolist()
		resample.SetSize(new_size)
		print(f" img.GetDirection(): {img.GetDirection()}")
		resample.SetOutputDirection(img.GetDirection())
		resample.SetOutputOrigin(img.GetOrigin())
		new_img = resample.Execute(img)
		arr = sitk.GetArrayFromImage(new_img).transpose(2,1,0)
		if shape != arr.shape:
			arr = scipy.ndimage.zoom(arr, [t/s for t, s in zip(shape, arr.shape)], order=1 if linear else 0)
		affine = np.eye(4)
		affine[:3, :3] = np.array(new_img.GetDirection()).reshape(3,3) * np.array(new_img.GetSpacing())[:, None]
		affine[:3, 3] = new_img.GetOrigin()
		#print(f"resampled orientation: {}")
		#path = os.path.join(self.casePath, filename)
		#return NiftiVolume.init_from_array(arr, affine, path)


	def crop_volume(self, vol, bbox, filename):
		data = vol.data[bbox]
		path = os.path.join(self.casePath, filename)
		cropped_vol = NiftiVolume.init_from_array(data,
											vol.affine,
											path)
		#cropped_vol.save()
		log(f"Cropped {path}")
		return cropped_vol


	#def HU_normalisation(self):



