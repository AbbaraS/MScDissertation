import os
from core.Case import *
from core.Log import log

'''
	LV + MYO focus areas: Apical, Mid, Basal.
		- Apical:  bottom 1/3 of slices.
		- Mid:     middle 1/3 of slices.
		- Basal:   top 1/3 of LV slices.

	Others: ## if not included in model, add in LIMITATIONS
		- RV involvement cases?
		- isolated RV cases?

	- LV volume: 3D volume of LV myocardium.
		## can i get 3D LV shape?

'''



def get_three_slices_within(indices):
	"""Pick ~25%, 50%, 75% positions within a sorted list of indices (unique)."""
	if indices is None:
		return []
	indices = np.unique(indices)
	if indices.size == 0:
		return []
	if not np.all(indices[:-1] <= indices[1:]):
		indices = np.sort(indices)

	if indices.size < 3:
		# Return what we have (unique, sorted)
		return indices.tolist()

	# Quartile-ish picks
	q1 = indices[len(indices) // 4]
	q2 = indices[len(indices) // 2]
	q3 = indices[(3 * len(indices)) // 4]
	return [int(q1), int(q2), int(q3)]

def volume_slicer(vol):
	"""Return callables for sagittal (X), coronal (Y), axial (Z) slices for RAS-oriented volumes."""
	sagittal = lambda i: vol[i, :, :]   # X
	coronal  = lambda i: vol[:, i, :]   # Y
	axial    = lambda i: vol[:, :, i]   # Z
	return [sagittal, coronal, axial]

def sliceVol(case):
	"""
	From LVresampled, compute three slice indices per axis, then extract
	those slices from resampledCT and resampled_mask.
	Returns a dict:
	  {
		'X': {'indices': [...], 'ct': [np2d,...], 'mask': [np2d,...]},
		'Y': {...},
		'Z': {...}
	  }
	"""
	lv   = case.LVresampled.data
	ct   = case.resampledCT.data
	mask = case.resampled_mask.data

	# Safety checks
	if lv is None or ct is None or mask is None:
		log("One or more resampled volumes are None; aborting slice()")
		return {}

	# Slicers aligned to RAS axes: X, Y, Z
	ct_slicers   = volume_slicer(ct)
	mask_slicers = volume_slicer(mask)

	# For LV, determine which planes contain any non-zero voxels per axis:
	# X-axis ⇒ collapse Y,Z; Y-axis ⇒ collapse X,Z; Z-axis ⇒ collapse X,Y
	nonzero_X = np.where(np.any(lv > 0, axis=(1, 2)))[0]  # sagittal planes
	nonzero_Y = np.where(np.any(lv > 0, axis=(0, 2)))[0]  # coronal planes
	nonzero_Z = np.where(np.any(lv > 0, axis=(0, 1)))[0]  # axial planes

	axis_info = [
		('X', 0, nonzero_X),
		('Y', 1, nonzero_Y),
		('Z', 2, nonzero_Z),
	]

	results = {}

	for axis_label, axis_idx, nz in axis_info:
		#print(f"Processing {axis_label}: \n\n")
		idxs = get_three_slices_within(nz)

		# Fallback if LV is empty along this axis: pick quartiles of full volume
		if len(idxs) == 0:
			dim = ct.shape[axis_idx]
			idxs = [dim // 4, dim // 2, (3 * dim) // 4]
			log(f"[{axis_label}] LV empty on this axis; using fallback indices: {idxs}", False)

		# Clip to bounds just in case
		dim = ct.shape[axis_idx]
		idxs = [int(np.clip(i, 0, dim - 1)) for i in idxs]

		# Extract paired CT/mask slices using the axis-specific slicers
		ct_slicer   = ct_slicers[axis_idx]
		mask_slicer = mask_slicers[axis_idx]

		ct_slices   = [ct_slicer(i)   for i in idxs]
		mask_slices = [mask_slicer(i) for i in idxs]

		log(f"axis: {axis_label}, indices: {idxs}", False)
		for i in idxs:
			log(f"  slice {axis_label}={i}", False)

		results[axis_label] = {
			"indices": idxs,
			"ct": ct_slices,
			"mask": mask_slices
		}
	save_slices(case, results)
	#return results

def save_slices(case, results):
	for axis, data in results.items():
		idxes = data["indices"]
		ct_slices = data["ct"]
		mask_slices = data["mask"]
		ct_affine = case.resampledCT.affine
		ct_header = case.resampledCT.header
		mask_affine = case.resampled_mask.affine
		mask_header = case.resampled_mask.header
		for i, (idx, ct_slice, mask_slice) in enumerate(zip(idxes, ct_slices, mask_slices)):
			log(f"Saving CT & mask slice {axis}_{idx}")
			ct_filename = f"data/cases/{case.caseID}/ctSlices/{axis}_{idx}.nii.gz"
			mask_filename = f"data/cases/{case.caseID}/maskSlices/{axis}_{idx}.nii.gz"
			_=NiftiVolume.init_from_array(ct_slice, ct_affine, ct_header, ct_filename)
			_=NiftiVolume.init_from_array(mask_slice, mask_affine, mask_header, mask_filename)


def slices_exist(case: Case):
	"""
	Check if slice files exist for the given case.
	Returns True if all slices exist, False otherwise.
	"""
	for axis in ['X', 'Y', 'Z']:
		for i in range(3):  # Assuming 3 slices per axis
			ct_path = f"data/cases/{case.caseID}/ctSlices/{axis}_{i}.nii.gz"
			mask_path = f"data/cases/{case.caseID}/maskSlices/{axis}_{i}.nii.gz"
			if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
				return False
	return True