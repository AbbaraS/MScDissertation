from Suleima.Utilities.imports import *
from Suleima.Utilities.old_utils import get_cropped_paths, \
	load_nifti,\
	get_segments_paths, \
	get_resampled_paths







def get_three_slices_within(indices):
	indices = sorted(indices)
	if len(indices) < 3:
		return list(indices)
	return [
		indices[len(indices) // 4],
		indices[len(indices) // 2],
		indices[3 * len(indices) // 4]
	]

def slice_CT(output_folder):
	cropped_paths = get_resampled_paths(output_folder)
	slices_dir = os.path.join(output_folder, "nii_slices")
	os.makedirs(slices_dir, exist_ok=True)

	ct = nib.load(cropped_paths["CT"])
	affine = ct.affine
	header = ct.header.copy()
	ct_np = ct.get_fdata()
	mask_np = nib.load(cropped_paths["Mask"]).get_fdata()

	lv = nib.load(cropped_paths["LV"]).get_fdata()
	sagittal_indices = np.where(np.any(lv > 0, axis=(1, 2)))[0]
	coronal_indices = np.where(np.any(lv > 0, axis=(0, 2)))[0]
	axial_indices = np.where(np.any(lv > 0, axis=(0, 1)))[0]

	for axis_label, indices, ct_slicer, mask_slicer in zip(
		['X', 'Y', 'Z'],
		[sagittal_indices, coronal_indices, axial_indices],
		[lambda i: ct_np[i, :, :], lambda i: ct_np[:, i, :], lambda i: ct_np[:, :, i]],
		[lambda i: mask_np[i, :, :], lambda i: mask_np[:, i, :], lambda i: mask_np[:, :, i]]):

		for idx in get_three_slices_within(indices):
			ct_slice = ct_slicer(idx)
			mask_slice = mask_slicer(idx)

			ct_nifti = nib.Nifti1Image(ct_slice, affine=affine, header=header)
			mask_nifti = nib.Nifti1Image(mask_slice, affine=affine, header=header)

			nib.save(ct_nifti, os.path.join(slices_dir, f"ct{axis_label}_{idx}.nii.gz"))
			nib.save(mask_nifti, os.path.join(slices_dir, f"mask{axis_label}_{idx}.nii.gz"))

	#print(f"✅ Slices saved in: {slices_dir}")

def trim_empty_slices(volume):
	x_nonzero = np.any(volume, axis=(1, 2))
	y_nonzero = np.any(volume, axis=(0, 2))
	z_nonzero = np.any(volume, axis=(0, 1))
	x_min, x_max = np.where(x_nonzero)[0][[0, -1]]
	y_min, y_max = np.where(y_nonzero)[0][[0, -1]]
	z_min, z_max = np.where(z_nonzero)[0][[0, -1]]
	return (x_min, x_max+1), (y_min, y_max+1), (z_min, z_max+1)

def resample_volume_shape(volume_np,
						  spacing,
						  new_spacing=[1.0, 1.0, 1.0],
						  target_shape=(64, 64, 64),
						  is_label=False,
						  reference_image=None,
						  original_affine=None):
	"""
	Resamples a 3D volume to the given spacing, optionally matching a reference image for perfect alignment.
	Returns the resampled NumPy array and its affine.
	"""
	# Convert to SimpleITK image (z, y, x) format expected by SimpleITK
	sitk_img = sitk.GetImageFromArray(np.transpose(volume_np, (2, 1, 0)))
	sitk_img.SetSpacing([float(s) for s in spacing[::-1]])

	if reference_image is not None:
		# Use reference image's properties to ensure perfect alignment
		resampler = sitk.ResampleImageFilter()
		resampler.SetReferenceImage(reference_image)
		resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
		resampled = resampler.Execute(sitk_img)
	else:
		# Compute new size for desired spacing
		original_size = np.array(sitk_img.GetSize(), dtype=np.int32)
		original_spacing = np.array(sitk_img.GetSpacing())
		new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()

		resampler = sitk.ResampleImageFilter()
		resampler.SetSize(new_size)
		resampler.SetOutputSpacing(new_spacing)
		resampler.SetOutputDirection(sitk_img.GetDirection())
		resampler.SetOutputOrigin(sitk_img.GetOrigin())
		resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
		resampled = resampler.Execute(sitk_img)

	# Recover affine (nibabel-style)
	direction = np.array(resampled.GetDirection()).reshape(3, 3)
	spacing_arr = np.array(resampled.GetSpacing())

	if original_affine is not None:
		origin = original_affine[:3, 3]  # use original origin
	else:
		origin = np.array(resampled.GetOrigin())

	affine = np.eye(4)
	affine[:3, :3] = direction * spacing_arr[:, None]
	affine[:3, 3] = origin

	# Convert back to NumPy in (x, y, z)
	resampled_np = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
	# --- Step 2: Resize to target shape using SciPy ---
	zoom_factors = [t / s for t, s in zip(target_shape, resampled_np.shape)]
	order = 0 if is_label else 1
	volume_final = scipy.ndimage.zoom(resampled_np, zoom=zoom_factors, order=order)
	return volume_final, affine, resampled  # Return SimpleITK image for reference use



def crop_trim_resample_heart(input_folder, output_folder):
	# === Step 3 - Crop, trim, and resample heart ===

	OG_file = load_nifti(get_segments_paths(input_folder))
	lv = OG_file["LV"]["data"]
	rv = OG_file["RV"]["data"]
	la = OG_file["LA"]["data"]
	ra = OG_file["RA"]["data"]
	myo = OG_file["MYO"]["data"]
	og_ct_data = OG_file["CT"]["data"]
	og_ct_voxel = OG_file["CT"]["voxel"]
	og_affine=OG_file["CT"]["affine"]
	# Combine masks to find heart bounding box
	binary_mask = ((lv + rv + la + ra + myo) > 0).astype(np.uint8)   # (sum of positive labels across all masks)  & turn the summed array into a binary mask
	x_min, y_min, z_min = np.array(np.where(binary_mask)).min(axis=1)
	x_max, y_max, z_max = np.array(np.where(binary_mask)).max(axis=1)

	# calculate crop boundaries (3D bounding box)
	# identify min and max voxel coordinates where heart structures are present
	x0, x1 = max(x_min, 0), min(x_max, og_ct_data.shape[0])
	y0, y1 = max(y_min, 0), min(y_max, og_ct_data.shape[1])
	z0, z1 = max(z_min, 0), min(z_max, og_ct_data.shape[2])

	# crop volumes with bounding box coordinates
	ct_crop = og_ct_data[x0:x1, y0:y1, z0:z1]
	lv_crop = lv[x0:x1, y0:y1, z0:z1]
	rv_crop = rv[x0:x1, y0:y1, z0:z1]
	la_crop = la[x0:x1, y0:y1, z0:z1]
	ra_crop = ra[x0:x1, y0:y1, z0:z1]
	myo_crop = myo[x0:x1, y0:y1, z0:z1]

	# Further trim empty slices
	# binary_mask = (sum of positive labels across all masks)  & turn the summed array back into a binary mask
	binary_mask = ((lv_crop + rv_crop + la_crop + ra_crop + myo_crop) > 0).astype(np.uint8)
	x_has_empty = not np.all(np.any(binary_mask, axis=(1, 2)))
	y_has_empty = not np.all(np.any(binary_mask, axis=(0, 2)))
	z_has_empty = not np.all(np.any(binary_mask, axis=(0, 1)))

	print(f"cropped CT shape: {ct_crop.shape}")
	if x_has_empty or y_has_empty or z_has_empty:
		x_rng, y_rng, z_rng = trim_empty_slices(binary_mask)
		ct_crop = ct_crop[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
		lv_crop = lv_crop[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
		rv_crop = rv_crop[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
		la_crop = la_crop[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
		ra_crop = ra_crop[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
		myo_crop = myo_crop[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]

		print(f"trimmed CT shape: {ct_crop.shape}")
	else:
		print("skipping trim step.")


	#print("Original CT crop shape:", ct_crop.shape, "max:", ct_crop.max(), "min:", ct_crop.min())
	# === Resample each ===
	ct_res, ct_affine, ct_sitk = resample_volume_shape(ct_crop, og_ct_voxel, is_label=False, original_affine=og_affine)
	# Resample all masks using CT as reference
	lv_res, _, _ = resample_volume_shape(lv_crop, og_ct_voxel, is_label=True, reference_image=ct_sitk)
	rv_res, _, _ = resample_volume_shape(rv_crop, og_ct_voxel, is_label=True, reference_image=ct_sitk)
	la_res, _, _ = resample_volume_shape(la_crop, og_ct_voxel, is_label=True, reference_image=ct_sitk)
	ra_res, _, _ = resample_volume_shape(ra_crop, og_ct_voxel, is_label=True, reference_image=ct_sitk)
	myo_res, _, _ = resample_volume_shape(myo_crop, og_ct_voxel, is_label=True, reference_image=ct_sitk)

	resampled_file_path = get_resampled_paths(output_folder)
	# Save NIfTI
	nib.save(nib.Nifti1Image(ct_res, ct_affine), resampled_file_path["CT"])
	nib.save(nib.Nifti1Image(lv_res, ct_affine), resampled_file_path["LV"])
	nib.save(nib.Nifti1Image(rv_res, ct_affine), resampled_file_path["RV"])
	nib.save(nib.Nifti1Image(la_res, ct_affine), resampled_file_path["LA"])
	nib.save(nib.Nifti1Image(ra_res, ct_affine), resampled_file_path["RA"])
	nib.save(nib.Nifti1Image(myo_res, ct_affine), resampled_file_path["MYO"])

	cropped_file = load_nifti(resampled_file_path)

	# === Load cropped masks and CT scan ===
	cropped_data =  cropped_file["CT"]["data"]

	combined_mask = np.zeros_like(cropped_data, dtype=np.uint8)
	combined_mask[cropped_file["LV"]["data"] > 0] = 1   # LV
	combined_mask[cropped_file["RV"]["data"] > 0] = 2   # RV
	combined_mask[cropped_file["MYO"]["data"] > 0] = 3  # Myo
	combined_mask[cropped_file["LA"]["data"] > 0] = 4   # LA
	combined_mask[cropped_file["RA"]["data"] > 0] = 5   # RA

	nib.save(nib.Nifti1Image(combined_mask, ct_affine), resampled_file_path["Mask"])

'''
def process_case(root_dir):
	for caseID in os.listdir(root_dir):
		print(f"Patient ID: {caseID}")
		case_dir = os.path.join(root_dir, caseID)


		if not os.path.exists(root_dir):
			continue

		try:
			# === Step 1 (DONE) - Convert DICOM to NIfTI ===
			if not os.path.exists(os.path.join(patient_dir, "OG_CT.nii.gz")):
				dicom2nifti.convert_dicom.dicom_series_to_nifti(dicom_dir, os.path.join(patient_dir, "OG_CT.nii.gz"),)

			# === Step 2 (DONE) - Segment with TotalSegmentator ===
			if not all(os.path.exists(f) for f in OG_paths.values()):
				_ = totalsegmentator(
					input_path=dicom_dir,
					output_path=segments_dir,
					license_number="aca_BWYHC6UQQFDU8A",
					task="heartchambers_highres",
					body_seg=True,
					preview=True,

				)

			# === Step 3 (DONE) - Crop, trim, and resample heart ===
			if not all(os.path.exists(f) for f in cropped_paths.values()):
				crop_trim_resample_heart(input_folder, output_folder)


			# === Step 4 - Slice CT and masks ===
			#if len(os.listdir(os.path.join(output_folder, "nii_slices"))) < 18:
		except Exception as e:
			print(f"Failed for {patientID}: {e}")
'''
