from anyio import key
from networkx import volume
from core.CT_3D import *
import os
from unittest import case
import SimpleITK as sitk
from matplotlib import image
import numpy as np
import scipy.ndimage




def preprocessing_volumes(case: CardiacCT):
	"""
	Executes the cropping pipeline for a CardiacCT case:
	- Computes bounding box.
	- Crops fullCT and segments.
	- Saves cropped results.
	"""
	# ==== Check for existing cropped files ====
	cropped_ct_path = os.path.join(case.casePath, "croppedCT.nii.gz")
	cropped_dir = os.path.join(case.casePath, "cropped")
	cropped_exists = os.path.exists(cropped_ct_path)

	if cropped_exists:
		case.log_message("Cropping skipped (already exists).")
		case.load_cropped()  # You must define this method
	else:
		# === Crop pipeline ===
		bbox = cardiac_bounding_box(case)
		crop_to_bbox(case, bbox)
		trim_empty_slices(case)
		case.save_cropped()
		case.log_message("Cropping completed and saved.")


	# === Crop and Save
	bbox = cardiac_bounding_box(case)
	crop_to_bbox(case, bbox)
	trim_empty_slices(case)
	#case.save_cropped()
	case.log_message("Cropping completed and saved.")
	# === Resample and Save
	resample_CT(case, target_spacing=[1.0, 1.0, 1.0], target_shape=(64, 64, 64))
	resample_segments(case, target_spacing=[1.0, 1.0, 1.0], target_shape=(64, 64, 64))
	#case.save_resampled()
	case.log_message("Resampling completed and saved.")







def resample_CT(case, target_spacing=[1.0, 1.0, 1.0], target_shape=(64, 64, 64)):
	sitk_img = sitk.GetImageFromArray(case.croppedCT.data.transpose(2, 1, 0))
	spacing = [float(s) for s in case.croppedCT.spacing[::-1]]
	sitk_img.SetSpacing(spacing)
	
	original_spacing = np.array(sitk_img.GetSpacing())
	original_size = np.array(sitk_img.GetSize(), dtype=np.int32)

	new_spacing = np.array(target_spacing[::-1])
	new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()

	resampler = sitk.ResampleImageFilter()
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetSize(new_size)
	resampler.SetOutputSpacing(new_spacing.tolist())
	resampler.SetOutputDirection(sitk_img.GetDirection())
	resampler.SetOutputOrigin(sitk_img.GetOrigin())

	resampled_sitk = resampler.Execute(sitk_img)
	resampled_np = sitk.GetArrayFromImage(resampled_sitk).transpose(2, 1, 0)

	if target_shape and target_shape != resampled_np.shape:
		zoom_factors = [t / s for t, s in zip(target_shape, resampled_np.shape)]
		resampled_np = scipy.ndimage.zoom(resampled_np, zoom=zoom_factors, order=1)

	direction = np.array(resampled_sitk.GetDirection()).reshape(3, 3)
	spacing_arr = np.array(resampled_sitk.GetSpacing())
	origin = np.array(resampled_sitk.GetOrigin())

	affine = np.eye(4)
	affine[:3, :3] = direction * spacing_arr[:, None]
	affine[:3, 3] = origin

	case.resampledCT = NiftiVolume.from_array(resampled_np, affine, os.path.join(case.casePath, "resampledCT.nii.gz"), case=case)



def resample_segments(case, target_spacing=[1.0, 1.0, 1.0], target_shape=(64, 64, 64)):
	cropped_segments = {
		"LV": case.LV_croppedseg,
		"LA": case.LA_croppedseg,
		"RV": case.RV_croppedseg,
		"RA": case.RA_croppedseg,
		"MYO": case.MYO_croppedseg,
	}

	for name, seg in cropped_segments.items():
		if seg is None:
			continue

		sitk_img = sitk.GetImageFromArray(seg.data.transpose(2, 1, 0))
		sitk_img.SetSpacing(seg.spacing[::-1])

		original_spacing = np.array(sitk_img.GetSpacing())
		original_size = np.array(sitk_img.GetSize(), dtype=np.int32)

		new_spacing = np.array(target_spacing[::-1])
		new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()

		resampler = sitk.ResampleImageFilter()
		resampler.SetInterpolator(sitk.sitkNearestNeighbor)
		resampler.SetSize(new_size)
		resampler.SetOutputSpacing(new_spacing.tolist())
		resampler.SetOutputDirection(sitk_img.GetDirection())
		resampler.SetOutputOrigin(sitk_img.GetOrigin())

		resampled = resampler.Execute(sitk_img)
		resampled_np = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)

		if target_shape and target_shape != resampled_np.shape:
			zoom_factors = [t / s for t, s in zip(target_shape, resampled_np.shape)]
			resampled_np = scipy.ndimage.zoom(resampled_np, zoom=zoom_factors, order=0)

		direction = np.array(resampled.GetDirection()).reshape(3, 3)
		spacing_arr = np.array(resampled.GetSpacing())
		origin = np.array(resampled.GetOrigin())

		affine = np.eye(4)
		affine[:3, :3] = direction * spacing_arr[:, None]
		affine[:3, 3] = origin

		resampled_vol = NiftiVolume.from_array(resampled_np, affine, os.path.join(case.resampledPath, f"resampled{name}.nii.gz"), case=case)
		setattr(case, f"{name}_resampledseg", resampled_vol)



def cardiac_bounding_box(case):
	"""
	Computes the bounding box of available cardiac segments.
	Returns slice objects: (slice_x, slice_y, slice_z) or None if no segments are loaded.
	"""
	segments = {
		"LV": getattr(case, "LV_totalseg", None),
		"RV": getattr(case, "RV_totalseg", None),
		"LA": getattr(case, "LA_totalseg", None),
		"RA": getattr(case, "RA_totalseg", None),
		"MYO": getattr(case, "MYO_totalseg", None)
	}
	masks = [
		seg.data > 0
		for seg in segments.values()
		if seg is not None
	]
	if not masks:
		print("Bounding box: No segments loaded.")
		return None

	# Combine masks to form a full heart region
	combined_mask = np.sum(masks, axis=0) > 0
	coords = np.array(np.where(combined_mask))

	if coords.size == 0:
		print("Bounding box: No positive voxels found.")
		return None

	x_min, y_min, z_min = coords.min(axis=1)
	x_max, y_max, z_max = coords.max(axis=1)

	# Ensure bounding box fits within fullCT shape
	shape = case.fullCT.shape
	x0, x1 = max(x_min, 0), min(x_max + 1, shape[0])
	y0, y1 = max(y_min, 0), min(y_max + 1, shape[1])
	z0, z1 = max(z_min, 0), min(z_max + 1, shape[2])
	case.log_message(f"Bounding box: x({x0}:{x1}), y({y0}:{y1}), z({z0}:{z1})")
	return slice(x0, x1), slice(y0, y1), slice(z0, z1)

def crop_to_bbox(case, bbox):
	"""
	Applies bounding box cropping to fullCT and available segments.
	Stores cropped volumes in case.croppedCT, case.LV_croppedseg, etc.
	"""
	
	if bbox is None:
		return

	full_ct_crop = case.fullCT.data[bbox]
	
	case.croppedCT = NiftiVolume.from_array(
		array=full_ct_crop,
		affine=case.fullCT.affine,
		path=os.path.join(case.casePath, "croppedCT.nii.gz"),
		case=case
	)
	# Crop each segment
	segments = {
		"LV": getattr(case, "LV_totalseg", None),
		"RV": getattr(case, "RV_totalseg", None),
		"LA": getattr(case, "LA_totalseg", None),
		"RA": getattr(case, "RA_totalseg", None),
		"MYO": getattr(case, "MYO_totalseg", None)
	}

	for key, volume in segments.items():
		if volume is not None:
			crop = volume.data[bbox]
			#cropped_img = nib.Nifti1Image(crop, affine=volume.affine)
			# Construct the save path: e.g., data/cases/{caseID}/cropped/LV.nii.gz
			save_path = os.path.join(case.casePath, "cropped", f"{key}.nii.gz")

			# Create NiftiVolume directly from crop array + affine
			cropped_volume = NiftiVolume.from_array(
				array=crop,                  # numpy array
				affine=volume.affine,        # affine matrix
				path=save_path,              # full path for saving
				case=case
			)

			# Save and assign
			cropped_volume.save(save_path)
			setattr(case, f"{key}_croppedseg", cropped_volume)

			# Log
			case.log_message(
				f"{key} cropped: shape: {cropped_volume.shape}, spacing: {cropped_volume.spacing}, "
				f"origin: {cropped_volume.origin}, orientation: {str(aff2axcodes(cropped_volume.affine))}"
			)
	


def trim_empty_slices(case: CardiacCT):
	
	"""
	Trims all-zero slices from cropped CT and cropped segments along each axis.
	Updates the volumes in-place and logs the shapes before and after trimming.
	"""
	
	segments = {
		"LV": getattr(case, "LV_croppedseg", None),
		"RV": getattr(case, "RV_croppedseg", None),
		"LA": getattr(case, "LA_croppedseg", None),
		"RA": getattr(case, "RA_croppedseg", None),
		"MYO": getattr(case, "MYO_croppedseg", None)
	}


	original_shape = case.croppedCT.shape
	case.cropped_mask = np.sum([
			(s.data > 0).astype(np.uint8)
				for s in segments.values()
				if s is not None and s.data is not None
				], axis=0) > 0
	x_nonzero = np.any(case.cropped_mask, axis=(1, 2))
	y_nonzero = np.any(case.cropped_mask, axis=(0, 2))
	z_nonzero = np.any(case.cropped_mask, axis=(0, 1))

	if not np.all(x_nonzero) or not np.all(y_nonzero) or not np.all(z_nonzero):
		x_min, x_max = np.where(x_nonzero)[0][[0, -1]]
		y_min, y_max = np.where(y_nonzero)[0][[0, -1]]
		z_min, z_max = np.where(z_nonzero)[0][[0, -1]]

		x_rng = slice(x_min, x_max + 1)
		y_rng = slice(y_min, y_max + 1)
		z_rng = slice(z_min, z_max + 1)

		#case.flog("Trimming empty slices detected:")
		print("Trimming empty slices detected:")

		new_ct = case.croppedCT.data[x_rng, y_rng, z_rng]
		case.croppedCT = new_ct
		case.log_message(f"- CT shape before: {original_shape} after:  {new_ct.shape}")
		#case.flog(f"- CT shape before: {original_shape} after:  {new_ct.shape}")


		for key, seg in segments.items():
			if seg is not None:
				orig_seg_shape = seg.shape
				trimmed = seg.data[x_rng, y_rng, z_rng]
				setattr(case, f"{key}_croppedseg", trimmed)
				#case.flog(f"- {key} shape: {orig_seg_shape} ➝ {trimmed.shape}")
				print(f"- {key} shape: {orig_seg_shape} ➝ {trimmed.shape}")
		
		original_mask_shape = case.cropped_mask.shape
		case.cropped_mask = case.cropped_mask[x_rng, y_rng, z_rng]
		#case.flog(f"mask shape before: {original_mask_shape} after: {case.cropped_mask.shape}")
		case.log_message(f"mask shape before: {original_mask_shape} after: {case.cropped_mask.shape}")
  
		#case.flog("mask shape before: {original_mask_shape} after: {case.cropped_mask.shape}")
	else:
		#case.flog("No empty slices detected. No trimming applied.")
		case.log_message("No empty slices detected. No trimming applied.")
