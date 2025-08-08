#from totalsegmentator.python_api import totalsegmentator
#import subprocess
import os
from core.Case import *
from core.Log import log

def bounding_box(segments: dict):
	"""
	Compute the bounding box that crops to foreground voxels
	across all segment masks and trims any empty slices.

	Parameters:
	- case object
	- segments: dict of segment volumes (must have .data attribute).

	Returns:
	- tuple of slice objects (x_slice, y_slice, z_slice), or None
	"""
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
				case.fullCT.header,
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
					vol.header,
					os.path.join(case.casePath, "cropped", f"{key}.nii.gz")
				)
				setattr(case, f"{key}cropped", cropped)
				log(f"{key}cropped saved.")
			#log("Cropped CT and segments.", False)
			if case.cropped_mask is None:
				mask = case.create_mask(case.croppedsegs, os.path.join(case.casePath, "cropped", "heart_mask.nii.gz"))
				case.cropped_mask = mask
				log("cropped_mask saved.")
	#else:
	#	case.croppedCT.save()

def resample(case: Case, target_spacing=[1.0]*3, target_shape=(64,64,64)):
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
			case.resampled_mask = case.create_mask(
				case.resampledsegs,
				os.path.join(case.casePath, "resampled", "heart_mask.nii.gz")
			)
			log("resampled_mask saved.")
	except Exception as e:
		log(f"Error resampling: {e}", False)





















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