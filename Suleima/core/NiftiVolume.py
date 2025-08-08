import nibabel as nib
import os
from core.Log import log
from nibabel.orientations import aff2axcodes
from core.globals import *


class NiftiVolume:
	def __init__(self, path):
		"""
		Initializes a NiftiVolume object from a path.
		"""
		self.path = path
		if os.path.exists(path):
			self.obj = nib.load(path)
			#log(f"{path}: \nshape:{self.obj.shape}, spacing:{self.obj.header.get_zooms()},\norientation:{aff2axcodes(self.obj.affine)}, \norigin:{self.affine[:3, 3]}")
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
	def orientation(self):
		return aff2axcodes(self.affine)

	@property
	def header(self):
		return self.obj.header

	@property
	def spacing(self):
		return self.obj.header.get_zooms()

	@property
	def origin(self):
		return self.affine[:3, 3]

	@property
	def exists(self):
		return self.path is not None and os.path.exists(self.path)

	def save(self, path=None):
		"""
		Saves the current volume to the specified path or to its own path.
		"""
		save_path = path or self.path
		if not save_path:
			raise ValueError("No path specified and no internal path to save to.")
		nib.save(self.obj, save_path)
		#log(f"Saved: {save_path}", False)

	@classmethod
	def init_from_array(cls, array, affine, header, path):
		img = cls.__new__(cls)
		try:
			header.set_data_shape(array.shape)
		except Exception as e:
			log(f"Failed to set header shape: {e}")
			header = nib.Nifti1Header()  # fallback

		img.obj = nib.Nifti1Image(array, affine, header)
		img.path = path

		try:
			nib.save(img.obj, path)
			#log(f"Saved NIfTI to: {path}", False)
		except Exception as e:
			log(f"Failed to save NIfTI to {path}: {e}")
		return img
	def __str__(self):
		return f"NiftiVolume(path={self.path}, shape={self.shape}, spacing={self.spacing}, orientation={self.orientation})"
	def __repr__(self):
		return f"NiftiVolume(path={self.path}, shape={self.shape}, spacing={self.spacing}, orientation={self.orientation})"