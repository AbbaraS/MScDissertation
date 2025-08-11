from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	Orientationd,
	CropForegroundd,
	Lambdad)
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from core.Log import *
from pathlib import Path
import os
import logging
from tqdm import tqdm
import torch
import nibabel as nib


def get_demographics(ID: str):
	try:
		parts = ID.split("_")
		type = parts[0]

		if len(parts) == 3: # e.g., "CNTRL_AAP50415783_61F"
			info = parts[2]
		elif len(parts) == 4: # e.g., "CNTRL_AAP_50415783_61F"
			info = parts[3]
		label = 0 if type == "CNTRL" else 1
		return {
			"label": label,         # control = 0, tts = 1
			"age": int(info[:2]),   # age = first two digits of info
			"gender": info[2]       # 'F' or 'M'
			}
	except Exception as e:
		logging.error(f"Error parsing: '{parts}' - should be AgeGender")
		return None

def stratified_datalists(
	full_datalist,
	test_size=0.2,
	random_state=42):
	"""
	Splits the data into train, validation, and test sets using stratification
	and returns the corresponding PyTorch DataLoaders.
	"""
	if not full_datalist:
		raise ValueError("Input datalist is empty. Cannot create DataLoaders.")

	# Get indices and labels for stratification
	indices = np.arange(len(full_datalist))
	labels = [d['label'] for d in full_datalist]

	# --- Outer Split: (Train+Val) vs. Test ---
	# We only need one split to create a single held-out test set.
	sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
	train_val_idx, test_idx = next(sss_test.split(indices, labels))

	# Create the test set datalist
	test_datalist = [full_datalist[i] for i in test_idx]

	# Get the data and labels for the remaining (train+val) set
	train_val_datalist = [full_datalist[i] for i in train_val_idx]
	train_val_labels = [train_val_datalist[i]['label'] for i in range(len(train_val_datalist))]

	# --- Inner Split: Train & Validation ---
	# Split the remaining data. test_size here refers to the proportion of the *remaining*
	# data that will become the validation set. 0.25 of 80% is 20% of the total.
	sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
	train_idx, val_idx = next(sss_val.split(np.arange(len(train_val_datalist)), train_val_labels))

	# Create the final train and validation datalists
	train_datalist = [train_val_datalist[i] for i in train_idx]
	val_datalist = [train_val_datalist[i] for i in val_idx]

	logging.info(f"Dataset split: {len(train_datalist)} training, {len(val_datalist)} validation, {len(test_datalist)} test cases.")
	return train_datalist, val_datalist, test_datalist

def save_metatensor_as_nifti(tensor, output_path: Path):
	"""
	Saves a MONAI MetaTensor to a .nii.gz file.
	Args:
		tensor (torch.Tensor): The MetaTensor to save (image or mask).
		output_path (Path): The file path for the output .nii.gz file.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	tensor_np = tensor.detach().cpu().numpy()
	if tensor_np.ndim == 4:
		tensor_np = tensor_np.squeeze(0)
	affine = tensor.meta['affine']
	nifti_image = nib.Nifti1Image(tensor_np, affine)
	nib.save(nifti_image, output_path)

def crop_training(datalist):
	transforms = Compose([
		LoadImaged(keys=["image", "mask"]),
		EnsureChannelFirstd(keys=["image", "mask"]),
		Orientationd(keys=["image", "mask"], axcodes="RAS"),
		Lambdad(keys=["image"], func=lambda x: torch.clamp(x, min=-175.0, max=250.0)),
		CropForegroundd(keys=["image", "mask"], source_key="mask")
	])
	for case_data in tqdm(datalist, desc="Cropping training data"):
		paths = {"image": Path(case_data["originals"]["image"]), "mask": Path(case_data["originals"]["mask"])}
		transformed = transforms(paths)
		cropped_image = transformed['image']
		cropped_mask = transformed['mask']
		p1 = Path(case_data["directory"], "cropped", "CTcrop.nii.gz")
		p2 = Path(case_data["directory"], "cropped", "MSKcrop.nii.gz")
		save_metatensor_as_nifti(cropped_image, p1)
		save_metatensor_as_nifti(cropped_mask, p2)
		case_data["cropped"] = {"image": str(p1),"mask": str(p2)}

def get_HU_stats(datalist):
	stats_transforms = Compose([
		LoadImaged(keys=["image"]),
		EnsureChannelFirstd(keys=["image"])])
	stats = []
	for item in tqdm(datalist, desc="Loading cropped images"):
		paths = {"image": Path(item["cropped"]["image"]), "mask": Path(item["cropped"]["mask"])}
		loaded = stats_transforms(paths)
		image = loaded["image"]
		fg_voxs = image[image > 0]
		if fg_voxs.numel() == 0: continue
		stats.append({
			"count": fg_voxs.numel(),
			"sum": torch.sum(fg_voxs),
			"sum_sq": torch.sum(fg_voxs ** 2)
		})
	return stats

def calculate_HU_stats(cropped):
	stats = get_HU_stats(cropped)
	count = sum(s['count'] for s in stats)
	sum_global = sum(s['sum'] for s in stats)
	sum_sq_global = sum(s['sum_sq'] for s in stats)
	if count == 0:
			logging.error("No foreground voxels found.")
			return 0.0, 1.0
	#E[X] = sum(X) / N
	global_mean = sum_global / count
	global_variance = (sum_sq_global / count) - (global_mean ** 2)
	if global_variance < 0:
		global_variance = torch.tensor(0.0)
	global_std_dev = torch.sqrt(global_variance)
	mean = global_mean.item()
	std_dev = global_std_dev.item()
	logging.info(f"Global mean: {mean}, Global std dev: {std_dev}")
	return mean, std_dev

def select_slices(volumes):
	"""
	Selects slices from the volumes based on the mask.
	"""
	if not isinstance(volumes, dict) or len(volumes)!=2:
		raise ValueError("volumes must be a dictionary with 'mask' and 'image' keys")
	lv = volumes["mask"]
	lv_mask = (lv == 3)
	h_indices, w_indices, d_indices = torch.where(lv_mask.squeeze(0))
	h_min, h_max = h_indices.min(), h_indices.max()
	w_min, w_max = w_indices.min(), w_indices.max()
	d_min, d_max = d_indices.min(), d_indices.max()
	#print(f"H: ({h_min}, {h_max}), W: ({w_min}, {w_max}), D: ({d_min}, {d_max})")
	d_base = int(d_min + int(0.2 * (d_max - d_min)))
	d_mid = int((d_min + d_max) // 2)
	d_apex = int(d_min + int(0.8 * (d_max - d_min)))
	w_optimal = 0
	max_area = 0
	for w in range(w_min, w_max + 1):
		area = lv_mask[:, :, w, :].sum()
		if area > max_area:
			max_area = area
			w_optimal = w
	h_optimal = 0
	max_area = 0
	for h in range(h_min, h_max + 1):
		area = lv_mask[:, h, :, :].sum()
		if area > max_area:
			max_area = area
			h_optimal = h
	return {
			"Axial": (d_base, d_mid, d_apex),
			"Sagittal": (h_optimal-1, h_optimal, h_optimal+2),
			"Coronal": (w_optimal-1, w_optimal, w_optimal+2)}


def save_slice_as_nifti(
	volume_tensor,
	slice_index: int,
	slice_axis: int,
	output_path: Path
):
	"""
	Extracts a 2D slice from a 3D volume, correctly adjusts its affine matrix,
	and saves it as a spatially-aware 3D NIfTI file with a thickness of 1.

	Args:
		volume_tensor (torch.Tensor): The 3D MetaTensor (image or mask) from which to extract the slice.
									  Must have metadata (e.g., from MONAI LoadImaged).
		slice_index (int): The index of the slice to extract.
		slice_axis (int): The axis from which to extract the slice (0=sagittal, 1=coronal, 2=axial).
		output_path (Path): The full path to save the output .nii.gz file.
	"""
	if 'affine' not in volume_tensor.meta:
		raise ValueError("Input tensor must be a MetaTensor with an 'affine' key in its metadata.")

	# --- 1. Extract the 2D slice data ---
	slice_data = np.take(volume_tensor.cpu().numpy().squeeze(0), slice_index, axis=slice_axis)

	# A NIfTI file must be at least 3D. We add a new axis of size 1.
	# The new axis should correspond to the one we sliced from.
	slice_data_3d = np.expand_dims(slice_data, axis=slice_axis)

	# --- 2. Calculate the new affine matrix for the slice ---
	original_affine = volume_tensor.meta['affine'].cpu().numpy()

	# Create a new affine matrix, starting as a copy
	new_affine = original_affine.copy()

	# The new origin is the physical coordinate of the corner of our slice.
	# We find this by transforming the voxel coordinate (e.g., [0, 0, slice_index, 1])
	# by the original affine matrix.
	corner_voxel_coord = np.array([0, 0, 0, 1])
	corner_voxel_coord[slice_axis] = slice_index

	new_physical_origin = np.dot(original_affine, corner_voxel_coord)

	# Update the origin (the last column) of our new affine matrix
	new_affine[:, 3] = new_physical_origin

	# --- 3. Save the new NIfTI file ---
	output_path.parent.mkdir(parents=True, exist_ok=True)
	nifti_img = nib.Nifti1Image(slice_data_3d, new_affine)
	nib.save(nifti_img, output_path)
	# print(f"Saved slice to: {output_path}")











