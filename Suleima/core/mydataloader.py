import nibabel as nib
from sklearn.model_selection import StratifiedShuffleSplit
import os
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	Orientationd,
	Lambdad,
	CropForegroundd,
	Resized,
	EnsureTyped,
	ScaleIntensityRanged,
	NormalizeIntensityd
)
from pathlib import Path
import pandas as pd
import logging
from core.globals import *
from pathlib import Path
import numpy as np
from core.Log import *
from methods.preprocessing import save_metatensor_as_nifti

from tqdm import tqdm
class TakotsuboDataset(Dataset):
	"""
	Custom PyTorch Dataset for loading Takotsubo CT scans and masks.
	It takes a list of data dictionaries and applies MONAI transforms.
	"""
	def __init__(self, data_dicts, transforms=None):
		"""
		Args:
			data_dicts (list of dict): List where each dict has 'image', 'mask', and 'label'.
			transforms (monai.transforms.Compose): The MONAI transform pipeline.
		"""
		self.data_dicts = data_dicts
		self.transforms = transforms

	def __len__(self):
		return len(self.data_dicts)

	def __getitem__(self, idx):
		# Get the data dictionary for the current index
		data_dict = self.data_dicts[idx]

		# Apply the MONAI transforms to load and process the image/mask
		if self.transforms is not None:
			processed_data = self.transforms(data_dict)
		else:
			processed_data = {
				"image": data_dict["image"],
				"mask": data_dict["mask"]
			}

		# The label is already in the dictionary, just ensure it's a tensor
		label = torch.tensor(data_dict['label'], dtype=torch.float32)

		# Return the processed image, mask, and the label
		return {
			"image": processed_data["image"],
			"mask": processed_data["mask"],
			"label": label
		}

DATA_KEYS = ["image", "mask"]
tensor_loader = Compose([
    LoadImaged(keys=DATA_KEYS),
    EnsureChannelFirstd(keys=DATA_KEYS),
    EnsureTyped(keys=DATA_KEYS)
])

def get_vols(itm, loader=tensor_loader):
	paths = {"image": Path(itm["image"]),
		     "mask":  Path(itm["mask"])}
	return loader(paths)

def get_dataloaders(train_datalist,
					val_datalist,
					test_datalist,
					batch_size,
					train_transforms=None,
					val_transforms=None):

	## --- Create Dataset and DataLoader instances ---
	train_dataset = TakotsuboDataset(data_dicts=train_datalist, transforms=train_transforms)
	val_dataset = TakotsuboDataset(data_dicts=val_datalist, transforms=val_transforms)
	test_dataset = TakotsuboDataset(data_dicts=test_datalist, transforms=val_transforms) # Use val_transforms for test (no augmentation)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	return train_loader, val_loader, test_loader

def resample_dataset(dataset, train_mean, train_std, dim=64):
	"""
	Preprocess the cases and return a dictionary of processed volumes.
	"""
	if dim == 64:
		target_shape = (64, 64, 64)
	elif dim == 128:
		target_shape = (128, 128, 128)
	elif dim == 96:
		target_shape = (96, 96, 96)

	target_orientation = "RAS"
	preproc_transforms = Compose([
		LoadImaged(keys=KEYS),
		EnsureChannelFirstd(keys=KEYS),
		Orientationd(keys=KEYS, axcodes=target_orientation),
		Lambdad(keys=["image"], func=lambda x: torch.clamp(x, min=-175.0, max=250.0)),
		CropForegroundd(keys=KEYS, source_key="mask"),
		NormalizeIntensityd(keys=["image"], subtrahend=train_mean, divisor=train_std),
		Resized(keys=KEYS, spatial_size=target_shape, mode=("bilinear", "nearest")),
		EnsureTyped(keys=KEYS)])

	for item in  tqdm(dataset, desc=f"Resampling to {dim} dataset..."):
		paths = {"image": Path(item["originals"]["image"]),
		   		 "mask": Path(item["originals"]["mask"])}

		volumes = preproc_transforms(paths)

		image = volumes["image"]
		mask = volumes["mask"]

		p1 = Path(item["directory"], f"resCT_{dim}.nii.gz" )
		p2 = Path(item["directory"], f"resMASK_{dim}.nii.gz")

		save_metatensor_as_nifti(image, p1)
		save_metatensor_as_nifti(mask, p2)
		item[f"resampled{dim}"] = {"image": str(p1), "mask": str(p2)}

def create_multilabel_heart_mask(casepath: Path):
	"""
	Loads individual TotalSegmentator files, combines them
	into a single multi-label NIfTI file.
	The labels are applied with priority: LV > Myocardium > Other Heart.
	Args:
		segments_dir: Path to the directory containing the .nii.gz segment files.
		output_path: Path to save the final combined_mask.nii.gz file.
	"""
	segments_dir = casepath / "segments"
	if not segments_dir.exists():
		return

	segments_data = {}
	reference_nii = None
	missing = []
	for key, filename in SEGMENT_FILES.items():
		file_path = segments_dir / filename
		if file_path.exists():
			nii_obj = nib.load(file_path)
			if reference_nii is None:
				reference_nii = nii_obj
			segments_data[key] = (nii_obj.get_fdata() > 0)
		else:
			missing.append(key)

	if missing:
		logging.error(f"Missing segments:{missing}. Aborting.")
		return

	if reference_nii is None:
		logging.error("No valid segment files were loaded. Cannot create mask.")
		return
	shape = reference_nii.shape
	combined_mask = np.zeros(shape, dtype=np.uint8)
	other_heart_mask = np.zeros(shape, dtype=bool)
	for key in ["right_atrium", "right_ventricle", "left_atrium"]:
		if key in segments_data:
			other_heart_mask = np.logical_or(other_heart_mask, segments_data[key])
	combined_mask[other_heart_mask] = LABEL_MAP["other_heart"]
	if "myocardium" in segments_data:
		combined_mask[segments_data["myocardium"]] = LABEL_MAP["myocardium"]
	if "left_ventricle" in segments_data:
		combined_mask[segments_data["left_ventricle"]] = LABEL_MAP["left_ventricle"]
	output_nii = nib.Nifti1Image(combined_mask, affine=reference_nii.affine, header=reference_nii.header)
	output_path = casepath / "label_mask.nii.gz"
	nib.save(output_nii, output_path)












