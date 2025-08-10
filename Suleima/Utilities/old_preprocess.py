from Suleima.Utilities.imports import *
from Suleima.Utilities.old_utils import get_cropped_paths, \
	load_nifti,\
	get_segments_paths, \
	get_resampled_paths
import os
from core.Case import *
from core.Log import log











def bounding_box(segments: dict):
	# Filter valid segment masks
	masks = [seg.data > 0 for seg in segments.values() if seg is not None]
	if not masks:
		log("No valid segment masks found for bounding box computation.")
		return None


	# Combine all masks to form a unified foreground region
	combined = np.any(np.stack(masks), axis=0)

	# Identify non-empty indices along each axis
	x_nonzero = np.any(combined, axis=(1,2))
	y_nonzero = np.any(combined, axis=(0,2))
	z_nonzero = np.any(combined, axis=(0,1))

	# Check if the mask is fully empty
	if not (np.any(x_nonzero) and np.any(y_nonzero) and np.any(z_nonzero)):
		log("All slices are empty — no bounding box computed.")
		return None

	# Get min and max indices for bounding box along each axis
	x0, x1 = np.where(x_nonzero)[0][[0, -1]]
	y0, y1 = np.where(y_nonzero)[0][[0, -1]]
	z0, z1 = np.where(z_nonzero)[0][[0, -1]]

	# Add +1 to include the last slice (Python slicing is exclusive)
	x1 += 1
	y1 += 1
	z1 += 1

	log(f"Final bbox (crop + trim):\n"
					 f"X=({x0}:{x1}), Y=({y0}:{y1}), Z=({z0}:{z1})")
	return (slice(x0, x1), slice(y0, y1), slice(z0, z1))




def crop(case: Case):
	if case.croppedCT is None or not case.croppedsegs:
		log("CroppedCT is None")
		bbox = bounding_box(case.totalsegs)
		if bbox is not None:
			# CT
			path = os.path.join(case.casePath, "croppedCT.nii.gz")
			case.croppedCT = NiftiVolume.init_from_array(
				case.fullCT.data[bbox],
				case.fullCT.affine,
				path)
			log(f"CoppedCT saved.")

			assignments = {
				"LV": case.LVtotalseg,
				"LA": case.LAtotalseg,
				"RV": case.RVtotalseg,
				"RA": case.RAtotalseg,
				"MYO": case.MYOtotalseg
			}
			for key, vol in assignments.items():
				cropped = NiftiVolume.init_from_array(
					vol.data[bbox],
					vol.affine,
					os.path.join(case.casePath, "cropped", f"{key}.nii.gz")
				)
				setattr(case, f"{key}cropped", cropped)
				log(f"{key}cropped saved.")
			#log("Cropped CT and segments.", False)
			if case.cropped_mask is None:
				mask = case.create_binary_mask(case.croppedsegs, os.path.join(case.casePath, "cropped", "heart_mask.nii.gz"))
				case.cropped_mask = mask
				log("cropped_mask saved.")
	#else:
	#	case.croppedCT.save()

def resample_old(case: Case, target_spacing=[1.0]*3, target_shape=(64,64,64)):
	try:
		croppedCT = case.croppedCT
		if case.resampledCT is None:
			case.resampledCT = case.resample_volume(
				croppedCT,
				target_spacing,
				target_shape,
				"resampledCT.nii.gz",
				linear=True
			)
			#case.resampledCT.save()
			log("resampledCT saved.")

		if case.resampled_mask is None:
			# Map segment names to property names
			name_to_attr = {
				"LV": "LVresampled",
				"LA": "LAresampled",
				"RV": "RVresampled",
				"RA": "RAresampled",
				"MYO": "MYOresampled"
			}

			for name, vol in case.croppedsegs.items():
				log(f"Resampling segment: {name} - {vol.data.shape}")
				resampled_vol = case.resample_volume(
					vol,
					target_spacing,
					target_shape,
					f"resampled/{name}.nii.gz",
					linear=False
				)
				setattr(case, name_to_attr[name], resampled_vol)

				log(f"{name_to_attr[name]} saved.")

			# Create and save resampled mask
			case.resampled_mask = case.create_binary_mask(
				case.resampledsegs,
				os.path.join(case.casePath, "resampled", "heart_mask.nii.gz")
			)
			log("resampled_mask saved.")
	except Exception as e:
		log(f"Error resampling: {e}", False)


def sitk_from_nifti(vol):
	"""
	vol.data: numpy array in XYZ index order
	vol.affine: 4x4 RAS affine from NiBabel
	Returns: SimpleITK Image with correct LPS geometry.
	"""
	A_ras = np.asarray(vol.affine, dtype=np.float64)
	M = A_ras[:3, :3]                       # voxel-to-world (RAS) linear part
	t_ras = A_ras[:3, 3]                    # world origin (RAS)

	# spacing = column norms; rotation = normalised columns
	sp = np.linalg.norm(M, axis=0)
	R_ras = M / sp

	# convert to LPS (ITK world)
	ras2lps = np.diag([-1.0, -1.0, 1.0])
	R_lps = ras2lps @ R_ras                 # rotate axes into LPS
	t_lps = ras2lps @ t_ras                 # origin into LPS

	# build SITK image; keep array in XYZ and let SITK own (x,y,z) indexing
	img = sitk.GetImageFromArray(vol.data.transpose(2,1,0))  # z,y,x memory view
	img.SetSpacing(tuple(sp.tolist()))                        # (x,y,z) spacings
	img.SetDirection(tuple(R_lps.reshape(-1).tolist()))       # 3x3 row-major
	img.SetOrigin(tuple(t_lps.tolist()))                      # (x,y,z) origin
	return img

def sitk_img(vol: NiftiVolume):
	img = sitk.GetImageFromArray(vol.data.transpose(2,1,0))
	img.SetDirection(tuple(vol.affine[:3, :3].flatten()))
	img.SetOrigin(tuple(vol.affine[:3, 3]))
	img.SetSpacing(tuple(np.sqrt((vol.affine[:3, :3]**2).sum(axis=0))))
	return img

def clip_HU(vol: NiftiVolume, hu_clip=(-1000.0, 2500.0)):
	img = sitk_img(vol)
	img = sitk.Cast(img, sitk.sitkFloat32) # ensure floating point to avoid overflow surprises on clamp
	img = sitk.Clamp(img, lowerBound=float(hu_clip[0]), upperBound=float(hu_clip[1]))
	return img


def space_resample(vol: NiftiVolume):
	# keeps same shape + physical extent & field of view
	# spacing becomes [1.0, 1.0, 1.0]
	target_spacing=np.array([1.0]*3, dtype=np.float32)
	interp = sitk.sitkLinear
	img = clip_HU(vol)
	img_size = np.array(img.GetSize())
	img_spacing = np.array(img.GetSpacing())
	# derive new voxel counts to keep physical extent constant
	# (167, 136, 41) x (0.68, 0.68, 2.0)
	phys_size = img_size * img_spacing          # mm per axis
	new_size = np.maximum(np.round(phys_size / target_spacing), 1).astype(int)


	resample = sitk.ResampleImageFilter()
	resample.SetSize(new_size.tolist())
	resample.SetOutputSpacing(target_spacing.tolist())
	resample.SetOutputDirection(img.GetDirection())
	resample.SetOutputOrigin(img.GetOrigin())
	resample.SetInterpolator(interp)
	img = resample.Execute(img)
	print(f"Size: {img.GetSize()}, Spacing: {img.GetSpacing()}")
	return img

def size_resample(vol: NiftiVolume, path: str):
	interp = sitk.sitkLinear # linear for intensities
	target_shape=(64,64,64)

	# 1) resample to spacing first - returns sitk.Image
	#img = space_resample(vol)
	img = clip_HU(vol)

	# 2) force final size with a second resample
	cur_size = np.array(img.GetSize(), dtype=np.int64)
	cur_spacing = np.array(img.GetSpacing(), dtype=np.float32)

	cur_phys = cur_size * cur_spacing
	out_size = np.array(target_shape, dtype=int)
	out_spacing = (cur_phys / out_size).astype(np.float32)
	print(f"cur_spacing: {cur_spacing}, out_spacing: {out_spacing}")
	rs2 = sitk.ResampleImageFilter()
	rs2.SetSize(out_size.tolist())
	rs2.SetOutputSpacing(out_spacing.tolist())
	rs2.SetOutputDirection(img.GetDirection())
	rs2.SetOutputOrigin(img.GetOrigin())
	rs2.SetInterpolator(interp)
	img = rs2.Execute(img)

	# 3) convert sitk.Image -> NumPy (XYZ order)
	arr_xyz = sitk.GetArrayFromImage(img).transpose(2,1,0).astype(np.float32)

	# 4) build affine: R @ diag(spacing)
	R = np.array(img.GetDirection(), dtype=np.float64).reshape(3,3)
	sp = np.array(img.GetSpacing(),  dtype=np.float64)
	A = np.eye(4, dtype=np.float64)
	A[:3,:3] = R @ np.diag(sp)                       # <- column scaling
	A[:3, 3] = np.array(img.GetOrigin(), dtype=np.float64)
	return NiftiVolume.init_from_array(arr_xyz, A, path)


	#print(f"Size: {img.GetSize()}, Spacing: {img.GetSpacing()}")

def size_pad_crop(vol: NiftiVolume, path: str):
	# symmetric pad or center-crop without further interpolation
	# (spacing unchanged; FOV changes)
	from math import floor
	target_shape=(64,64,64)
	def pad_to(img, target_shape):
		size = np.array(img.GetSize())
		pad_lower = np.maximum((np.array(target_shape) - size)//2, 0)
		pad_upper = np.maximum(np.array(target_shape) - size - pad_lower, 0)
		print("Padding lower:", pad_lower, "upper:", pad_upper)
		return sitk.ConstantPad(img,
					padList=pad_lower.tolist(),
					padUpperBound=pad_upper.tolist(),
					constant=0)

	def crop_to(img, target_shape):
		size = np.array(img.GetSize())
		start = np.maximum((size - np.array(target_shape))//2, 0).astype(int)
		extractor = sitk.RegionOfInterestImageFilter()
		extractor.SetIndex(start.tolist())
		extractor.SetSize(np.minimum(size, np.array(target_shape)).astype(int).tolist())
		print("")
		out = extractor.Execute(img)
		if tuple(out.GetSize()) != tuple(target_shape):
			out = pad_to(out, target_shape)  # pad if we cropped too tight on an axis
		return out

	# decide per axis
	size_now = np.array(img.GetSize())
	print(f"Current size: {size_now}, target shape: {target_shape}")
	if np.any(size_now < np.array(target_shape)):
		img = pad_to(img, target_shape)

	if tuple(img.GetSize()) != tuple(target_shape):
		img = crop_to(img, target_shape)
'''

def segmenting_volumes1(case: Case, skipSegmentation=False):
	"""
	Generates preview images and radiomics features for an already-segmented cardiac CT.
	Assumes that segmentation results are already present in case.segmentPath.
	"""
	path = os.path.join(case.casePath, "fullCT.nii.gz")
	segmentPath = case.segmentPath  # Should point to the folder with segmentation output


	_ = totalsegmentator(
		input_path=path,
		output_path=segmentPath,
		license_number="aca_BWYHC6UQQFDU8A",
		task="heartchambers_highres",
		body_seg=True,
		preview=True,
		radiomics=True,
		skip_segmentation=skipSegmentation)



def segmenting_volumes(case: Case, skipSegmentation=True):
	ctPath = os.path.join(case.casePath, "fullCT.nii.gz")
	segmentPath = case.segmentPath

	if not os.path.exists(ctPath):
		print(f"❌ Full CT for {case.caseID} does not exist.")
		return

	os.makedirs(segmentPath, exist_ok=True)

	cmd = [
		"totalsegmentator",
		ctPath,
		segmentPath,
		"--task", "heartchambers_highres",
		"--body_seg",
		"--radiomics",
		"--preview",
	]

	if skipSegmentation:
		cmd.append("--skip_segmentation")

	subprocess.run(cmd, check=True)

'''





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
