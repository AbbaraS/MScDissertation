
from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	Orientationd,
	CropForegroundd,
	EnsureTyped,
	Lambdad)
import numpy as np

from pathlib import Path
import logging
from tqdm import tqdm
import torch
import nibabel as nib
from tqdm import tqdm
from core.globals import *

logger = logging.getLogger('root')
stats_transforms = Compose([
		LoadImaged(keys=["image"]),
		EnsureChannelFirstd(keys=["image"])])

tensor_loader = Compose([
	LoadImaged(keys=MONAI_KEYS),
	EnsureChannelFirstd(keys=MONAI_KEYS),
	EnsureTyped(keys=MONAI_KEYS)
])

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


def get_vols(itm, loader=tensor_loader):
	paths = {"image": Path(itm["image"]),
			 "mask":  Path(itm["mask"])}
	return loader(paths)

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
		logger.error(f"Error parsing: '{parts}' - should be AgeGender")
		return None

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
		logger.error(f"Missing segments:{missing}. Aborting.")
		return

	if reference_nii is None:
		logger.error("No valid segment files were loaded. Cannot create mask.")
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



