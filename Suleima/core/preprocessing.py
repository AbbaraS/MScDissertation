'''
METHODS IN FILE:
get_vols(itm, loader=tensor_loader) --->  return loader(paths)
resample_dataset(dataset, train_mean, train_std, dim=64) ---> NO RETURN | saves volumes in directory

'''


from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	Orientationd,
	CropForegroundd,
	EnsureTyped,
	Resized,
	NormalizeIntensityd,
	Lambdad)
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from core.Log import *
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import nibabel as nib
from tqdm import tqdm
from core.globals import *

stats_transforms = Compose([
		LoadImaged(keys=["image"]),
		EnsureChannelFirstd(keys=["image"])])

tensor_loader = Compose([
	LoadImaged(keys=MONAI_KEYS),
	EnsureChannelFirstd(keys=MONAI_KEYS),
	EnsureTyped(keys=MONAI_KEYS)
])

def calculate_age_stats(datalist):
	# 1. Extract all 'age' values from the list of dictionaries
	ages = [case['age'] for case in datalist]

	# 2. Handle the edge case of an empty list
	if not ages:
		logging.error("The provided training datalist is empty.")
		return

	# 3. Calculate mean and standard deviation using NumPy
	age_mean = np.mean(ages)
	age_std = np.std(ages)

	logging.info(f"Calculated Age Stats: Mean={age_mean:.2f}, Std Dev={age_std:.2f}")

	return age_mean, age_std

def get_vols(itm, loader=tensor_loader):
	paths = {"image": Path(itm["image"]),
			 "mask":  Path(itm["mask"])}
	return loader(paths)

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
		LoadImaged(keys=MONAI_KEYS),
		EnsureChannelFirstd(keys=MONAI_KEYS),
		Orientationd(keys=MONAI_KEYS, axcodes=target_orientation),
		Lambdad(keys=["image"], func=lambda x: torch.clamp(x, min=-175.0, max=250.0)),
		CropForegroundd(keys=MONAI_KEYS, source_key="mask"),
		NormalizeIntensityd(keys=["image"], subtrahend=train_mean, divisor=train_std),
		Resized(keys=MONAI_KEYS, spatial_size=target_shape, mode=("bilinear", "nearest")),
		EnsureTyped(keys=MONAI_KEYS)])

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
		#item[f"resampled{dim}"] = {"image": str(p1), "mask": str(p2)}

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
	for item in tqdm(datalist, desc="Loading cropped images"):
		paths = {"image": Path(item["cropped"]["image"]), "mask": Path(item["cropped"]["mask"])}
		loaded = stats_transforms(paths)
		image = loaded["image"]
		fg_voxs = image[image > image.min()]
		if fg_voxs.numel() == 0: continue
		item["stats"]={
			"count": fg_voxs.numel(),
			"sum": torch.sum(fg_voxs).item(),
			"sum_sq": torch.sum(fg_voxs ** 2).item()}

def calculate_HU_stats(stats):
	count = sum(s['count'] for s in stats)
	sum_global = sum(s['sum'] for s in stats)
	sum_sq_global = sum(s['sum_sq'] for s in stats)
	if count == 0:
			logging.error("No foreground voxels found.")
			return 0.0, 1.0
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
	_, H, W, D = volumes["mask"].shape
	h_indices, w_indices, d_indices = torch.where(lv_mask.squeeze(0))

	h_min, h_max = h_indices.min(), h_indices.max()
	w_min, w_max = w_indices.min(), w_indices.max()
	d_min, d_max = d_indices.min(), d_indices.max()

	# --- Axial ---
	d_span = d_max - d_min
	d_base = d_min + int(0.20 * d_span)
	d_mid =  d_min + int(0.50 * d_span)
	d_apex = d_min + int(0.80 * d_span)
	# --- Coronal & Sagittal  ---

	w_optimal = max(range(w_min, w_max + 1), key=lambda w: lv_mask[:, :, w, :].sum())
	h_optimal = max(range(h_min, h_max + 1), key=lambda h: lv_mask[:, h, :, :].sum())

	# --- Proportional Gap Calculation ---
	h_span = h_max - h_min
	w_span = w_max - w_min

	h_gap = max(1, int(0.15 * h_span))
	w_gap = max(1, int(0.15 * w_span))

	sagittal_slices = (
		max(0, h_optimal - h_gap),
		h_optimal,
		min(H - 1, h_optimal + h_gap)
	)
	coronal_slices = (
		max(0, w_optimal - w_gap),
		w_optimal,
		min(W - 1, w_optimal + w_gap)
	)
	return {"Axial": (d_base, d_mid, d_apex),
			"Sagittal": sagittal_slices,
			"Coronal": coronal_slices}

def save_slice_as_nifti(
	volume_tensor,
	slice_index: int,
	slice_axis: int,
	output_path: Path):
	"""
	Extracts a 2D slice from a 3D volume, correctly adjusts its affine matrix,
	and saves it as a spatially-aware 3D NIfTI file with a thickness of 1.
	"""
	if 'affine' not in volume_tensor.meta:
		raise ValueError("Input tensor must be a MetaTensor with an 'affine' key in its metadata.")
	slice_data = np.take(volume_tensor.cpu().numpy().squeeze(0), slice_index, axis=slice_axis)
	slice_data_3d = np.expand_dims(slice_data, axis=slice_axis)
	original_affine = volume_tensor.meta['affine'].cpu().numpy()
	new_affine = original_affine.copy()
	corner_voxel_coord = np.array([0, 0, 0, 1])
	corner_voxel_coord[slice_axis] = slice_index
	new_physical_origin = np.dot(original_affine, corner_voxel_coord)
	new_affine[:, 3] = new_physical_origin
	output_path.parent.mkdir(parents=True, exist_ok=True)
	nifti_img = nib.Nifti1Image(slice_data_3d, new_affine)
	nib.save(nifti_img, output_path)
	# print(f"Saved slice to: {output_path}")

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

