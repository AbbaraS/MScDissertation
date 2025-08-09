#from totalsegmentator.python_api import totalsegmentator
#import subprocess
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
	img = space_resample(vol)

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
	#return NiftiVolume.init_from_array(arr_xyz, A, path)
	return None

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