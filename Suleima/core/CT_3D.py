from unittest import case
import nibabel as nib
import os
import numpy as np
from datetime import datetime
from nibabel.orientations import aff2axcodes
from Suleima.core.case import NiftiVolume

class CardiacCT:
	def __init__(self, caseID):
		self.caseID = caseID
		self.casePath = f"data/cases/{caseID}"
		self.log = []
		self.segmentPath = os.path.join(self.casePath, "segments")
		self.croppedPath = os.path.join(self.casePath, "cropped")
		self.resampledPath = os.path.join(self.casePath, "resampled")	
		#self.dicomPath = f"../Takotsubo-Syndrome/data/Inputs/{caseID}/dicom"
		self.missing = []
		self.cropped_mask = None  # Placeholder for cropped mask, to be set later
  

		self.fullCT                = self.load_fullCT()  
		self.LV_totalseg           = self.load_LV_totalseg()
		self.LA_totalseg           = self.load_LA_totalseg()
		self.RV_totalseg           = self.load_RV_totalseg()
		self.RA_totalseg           = self.load_RA_totalseg()
		self.MYO_totalseg          = self.load_MYO_totalseg()
		#self.TSsegments            = self.load_TS_segments()  # Placeholder for actual segments loading logic

		self.croppedCT             = None
		self.LV_croppedseg         = None
		self.LA_croppedseg         = None
		self.RV_croppedseg         = None
		self.RA_croppedseg         = None
		self.MYO_croppedseg        = None
		#self.cropped_segments      = self.load_cropped_segments()

		self.resampledCT           = None
		self.LV_resampledseg       = None
		self.LA_resampledseg       = None
		self.RV_resampledseg       = None
		self.RA_resampledseg       = None
		self.MYO_resampledseg      = None
		#self.resampled_segments    = self.load_resampled_segments()

		
	
	def flog(self, msg):
		self.log.append(msg)
	

	#### LOADERS ####	

	def load_fullCT(self):
		path = os.path.join(self.casePath, "fullCT.nii.gz")
		try:
			volume = NiftiVolume(path, self)
			self.log_message(f"fullCT:"
				f"shape: {volume.shape}, spacing: {volume.spacing}, origin: {volume.origin}, "
				f"orientation: {str(aff2axcodes(volume.affine))}")
			return volume
		except FileNotFoundError:
			self.log_message("fullCT.nii.gz missing")
			self.missing.append("fullCT")
			return None

	def load_croppedCT(self):
		path = os.path.join(self.casePath, "croppedCT.nii.gz")
		try:
			volume = NiftiVolume(path, self)
			self.log_message(f"croppedCT:"
				f"shape: {volume.shape}, spacing: {volume.spacing}, origin: {volume.origin}, "
				f"orientation: {str(aff2axcodes(volume.affine))}")
			return volume
		except FileNotFoundError:
			self.log_message("croppedCT.nii.gz missing")
			self.missing.append("croppedCT")
		return None

	def load_resampledCT(self):
		path = os.path.join(self.casePath, "resampledCT.nii.gz")
		try:
			volume = NiftiVolume(path, self)
			self.log_message(f"resampledCT:"
				f"shape: {volume.shape}, spacing: {volume.spacing}, origin: {volume.origin}, "
				f"orientation: {str(aff2axcodes(volume.affine))}")
			return volume
		except FileNotFoundError:
			self.log_message("resampledCT.nii.gz missing")
			self.missing.append("resampledCT")
		return None

	def load_segment(self, name, filename, segment_type="original"):
		"""
		Loads a single segment volume from disk into memory.
	
		Parameters:
			name (str): e.g., "LV"
			filename (str): e.g., "heart_ventricle_left.nii.gz"
			segment_type (str): one of "original", "cropped", "resampled"
	
		Returns:
			np.ndarray or None
		"""
		attr_name = f"{name}_{segment_type}"
		if getattr(self, attr_name, None) is not None:
			return getattr(self, attr_name)
	
		subfolder = {
			"original": "segments",
			"cropped": "cropped",
			"resampled": "resampled"
		}.get(segment_type)
	
		if subfolder is None:
			raise ValueError(f"Unknown segment_type: {segment_type}")
	
		path = os.path.join(self.casePath, subfolder, filename)
		try:
			volume = NiftiVolume(path, self)
			self.log_message(f"{name} {segment_type}:"
				f"shape: {volume.shape}, spacing: {volume.spacing}, origin: {volume.origin}, "
				f"orientation: {str(aff2axcodes(volume.affine))}")
			setattr(self, attr_name, volume)
			return volume
		except FileNotFoundError:
			self.log_message(f"{filename} missing in {subfolder}")
			self.missing.append(attr_name)
			return None

	# === ORIGINAL (totalseg) ===
	def load_LV_totalseg(self): return self.load_segment("LV",   "heart_ventricle_left.nii.gz"    , "original")
	def load_LA_totalseg(self): return self.load_segment("LA",   "heart_atrium_left.nii.gz"       , "original")
	def load_RV_totalseg(self): return self.load_segment("RV",   "heart_ventricle_right.nii.gz"   , "original")
	def load_RA_totalseg(self): return self.load_segment("RA",   "heart_atrium_right.nii.gz"      , "original")
	def load_MYO_totalseg(self): return self.load_segment("MYO", "heart_myocardium.nii.gz"        , "original")

	# === CROPPED ===
	def load_LV_cropped(self): return self.load_segment("LV", "LV.nii.gz", "cropped")
	def load_LA_cropped(self): return self.load_segment("LA", "LA.nii.gz", "cropped")
	def load_RV_cropped(self): return self.load_segment("RV", "RV.nii.gz", "cropped")
	def load_RA_cropped(self): return self.load_segment("RA", "RA.nii.gz", "cropped")
	def load_MYO_cropped(self): return self.load_segment("MYO", "MYO.nii.gz", "cropped")

	# === RESAMPLED ===
	def load_LV_resampled(self): return self.load_segment("LV", "LV.nii.gz", "resampled")
	def load_LA_resampled(self): return self.load_segment("LA", "LA.nii.gz", "resampled")
	def load_RV_resampled(self): return self.load_segment("RV", "RV.nii.gz", "resampled")
	def load_RA_resampled(self): return self.load_segment("RA", "RA.nii.gz", "resampled")
	def load_MYO_resampled(self): return self.load_segment("MYO", "MYO.nii.gz", "resampled")


	def load_TS_segments(self):
		"""Load original TotalSegmentator masks."""
		return {
			"LV": self.load_LV_totalseg(),
			"LA": self.load_LA_totalseg(),
			"RV": self.load_RV_totalseg(),
			"RA": self.load_RA_totalseg(),
			"MYO": self.load_MYO_totalseg(),
		}
	
	def load_cropped_segments(self):
		"""Load cropped segment volumes."""
		return {
			"LV": self.load_LV_cropped(),
			"LA": self.load_LA_cropped(),
			"RV": self.load_RV_cropped(),
			"RA": self.load_RA_cropped(),
			"MYO": self.load_MYO_cropped(),
		}
	
	def load_resampled_segments(self):
		"""Load resampled segment volumes."""
		return {
			"LV": self.load_LV_resampled(),
			"LA": self.load_LA_resampled(),
			"RV": self.load_RV_resampled(),
			"RA": self.load_RA_resampled(),
			"MYO": self.load_MYO_resampled(),
		}
	
	def save_cropped(self):
		"""
		Saves the cropped CT and available segment volumes to disk.
		Logs shape, spacing, origin, and orientation for each saved volume.
		"""
		os.makedirs(self.casePath, exist_ok=True)
		cropped_seg_dir = os.path.join(self.casePath, "cropped")
		os.makedirs(cropped_seg_dir, exist_ok=True)

		# Save cropped CT
		if self.croppedCT and isinstance(self.croppedCT, NiftiVolume):
			ct_path = os.path.join(self.casePath, "croppedCT.nii.gz")
			self.croppedCT.save(ct_path)
			self.log_message(
				f"croppedCT:"
				f" shape: {self.croppedCT.shape}, spacing: {self.croppedCT.spacing}, origin: {self.croppedCT.origin}, "
				f"orientation: {str(aff2axcodes(self.croppedCT.affine))}"
			)

		# Save cropped segments
		cropped_segments = {
			"LV": self.LV_croppedseg,
			"LA": self.LA_croppedseg,
			"RV": self.RV_croppedseg,
			"RA": self.RA_croppedseg,
			"MYO": self.MYO_croppedseg,
		}

		for name, seg in cropped_segments.items():
			if seg and isinstance(seg, NiftiVolume):
				save_path = os.path.join(cropped_seg_dir, f"{name}.nii.gz")
				seg.save(save_path)
				self.log_message(
					f"{name}_cropped:"
					f" shape: {seg.shape}, spacing: {seg.spacing}, origin: {seg.origin}, "
					f"orientation: {str(aff2axcodes(seg.affine))}"
				)
			else:
				self.log_message(f"{name}_cropped: Not available or not a NiftiVolume instance")

	def log_message(self, message, to_console=False):
		"""
		Logs a timestamped message to the case's info.txt file and optionally prints it.
	
		Parameters:
			message (str): The log message to store.
			to_console (bool): Whether to also print the message to the terminal.
		"""
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		log_entry = f"[{timestamp}] {message}"
	
		# Save in-memory
		self.log.append(log_entry)
	
		# Print to console
		if to_console:
			print(log_entry)
	
		# Save to file
		log_path = os.path.join(self.casePath, "info.txt")
		try:
			with open(log_path, "a") as f:
				f.write(log_entry + "\n")
		except Exception as e:
			print(f"[Logger Error] Failed to write to {log_path}: {e}")
   
   
   