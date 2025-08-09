#from totalsegmentator.python_api import totalsegmentator
#import subprocess
import os
from core.Case import *
from core.Log import log

import SimpleITK as sitk
import numpy as np
from typing import Optional, Dict



# --- helpers ---
def ensure_float32(img: sitk.Image) -> sitk.Image:
	return sitk.Cast(img, sitk.sitkFloat32)

def ensure_uint8(img: sitk.Image) -> sitk.Image:
	return sitk.Cast(img, sitk.sitkUInt8)


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
	#log("same_geometry")
	return (a.GetSize()      == b.GetSize() and
		np.allclose(a.GetSpacing(),   b.GetSpacing()) and
		np.allclose(a.GetOrigin(),    b.GetOrigin())  and
		np.allclose(a.GetDirection(), b.GetDirection()))

def resample_to_reference_nn(img: sitk.Image, reference: sitk.Image, default=0) -> sitk.Image:
	"""Nearest-neighbor resample onto 'reference' grid."""
	log("resample_to_reference_nn bc not same geometry")
	return sitk.Resample(
		img,
		reference,
		sitk.Transform(),                 # identity
		sitk.sitkNearestNeighbor,
		default,                          # background
		img.GetPixelID()
	)

def to_uint8_binary(img: sitk.Image) -> sitk.Image:
	#log("to_uint8_binary")
	"""Map nonzero→1, zero→0, stored as uint8."""
	arr = sitk.GetArrayFromImage(img)     # z,y,x
	bin_arr = (arr > 0).astype(np.uint8)
	out = sitk.GetImageFromArray(bin_arr)
	out.CopyInformation(img)              # keep geometry
	return out

def union_binary_masks(masks: list[sitk.Image], reference: sitk.Image) -> sitk.Image:
	#log("union_binary_masks")
	"""
		Union a list of (binary) masks onto the reference grid.
		Returns acc: an empty binary mask with same shape and geometry as CT, used to collect the combined union of all input masks.
	"""
	acc = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
	acc.SetOrigin(reference.GetOrigin());
	acc.SetSpacing(reference.GetSpacing());
	acc.SetDirection(reference.GetDirection())
	for m in masks:
		m_ref = m if same_geometry(m, reference) else resample_to_reference_nn(m, reference, default=0)
		m_bin = to_uint8_binary(m_ref)
		acc = sitk.Or(acc, m_bin)
	return acc

def union_binary_masks(masks: list[sitk.Image], reference: sitk.Image) -> sitk.Image:
	if not masks:
		raise ValueError("union_binary_masks: empty input list")
	out = sitk.Image(reference.GetSize(), sitk.sitkUInt8);
	out.CopyInformation(reference)
	for m in masks:
		m_ref = resample_like(m, reference, interp=sitk.sitkNearestNeighbor, default_value=0)
		out = out | binarise(m_ref, 0.5)
	return ensure_uint8(out)

def make_heart_mask_from_binaries(
	reference_ct: sitk.Image,
	totalseg_parts: dict[str, sitk.Image],
	include=("myocardium","left_ventricle","right_ventricle","left_atrium","right_atrium")) -> sitk.Image:
	"""
	totalseg_parts: mapping name->binary mask image (from TotalSegmentator per-structure files)
	include:       which keys to union
	returns        uint8 binary mask aligned to reference_ct
	"""
	log("make_heart_mask_from_binaries")
	masks = []
	for k in include:
		img = totalseg_parts.get(k)
		if img is None:
			continue
		masks.append(img)
	if not masks:
		raise ValueError("No included TotalSegmentator structures were found.")
	heart_mask = union_binary_masks(masks, reference_ct)
	return heart_mask

def make_lv_myo_masks_from_binaries(
	reference_ct: sitk.Image,
	totalseg_parts: dict[str, sitk.Image],
	lv_keys=("left_ventricle",),              # adapt if your keys differ
	myo_keys=("myocardium",),
	clean_islands=True,
	min_voxels=50,
	constrain_to_heart=True,
	precomputed_heart: sitk.Image | None = None,
) -> tuple[sitk.Image, sitk.Image, sitk.Image]:
	"""
	Returns (lv_mask, myo_mask, heart_mask) as uint8 binaries aligned to reference_ct.
	- Enforces MYO ∩ LV = ∅  (myocardium excludes LV cavity)
	- Optionally constrains both to the heart mask (safety)
	"""
	# 1) Build heart if not provided
	if precomputed_heart is None:
		heart_mask = make_heart_mask_from_binaries(
			reference_ct, totalseg_parts,
			include=("myocardium","left_ventricle","right_ventricle","left_atrium","right_atrium")
		)
	else:
		heart_mask = resample_like(precomputed_heart, reference_ct, sitk.sitkNearestNeighbor, 0)
		heart_mask = binarise(heart_mask, 0.5)

	# 2) Union selected keys for LV and MYO (usually single key each)
	def _union(keys):
		masks = [totalseg_parts[k] for k in keys if k in totalseg_parts]
		if not masks:
			# Generate empty aligned mask if missing
			empty = sitk.Image(reference_ct.GetSize(), sitk.sitkUInt8); empty.CopyInformation(reference_ct)
			return empty
		return union_binary_masks(masks, reference_ct)

	lv_mask  = _union(lv_keys)
	myo_mask = _union(myo_keys)

	# 3) Optional cleanup
	if clean_islands:
		lv_mask  = remove_tiny_islands(lv_mask,  min_voxels=min_voxels)
		myo_mask = remove_tiny_islands(myo_mask, min_voxels=min_voxels)

	# 4) Enforce mutual exclusivity (no LV voxels inside MYO and vice versa)
	# Anatomically: MYO is the wall, LV is the cavity -> remove LV from MYO
	myo_mask = myo_mask & sitk.BinaryNot(lv_mask)

	# 5) Constrain to heart for safety
	if constrain_to_heart:
		lv_mask  = lv_mask  & heart_mask
		myo_mask = myo_mask & heart_mask

	# 6) Final typing and metadata are already correct, but ensure:
	lv_mask  = ensure_uint8(lv_mask);  lv_mask.CopyInformation(reference_ct)
	myo_mask = ensure_uint8(myo_mask); myo_mask.CopyInformation(reference_ct)
	heart_mask = ensure_uint8(heart_mask); heart_mask.CopyInformation(reference_ct)

	return lv_mask, myo_mask, heart_mask








def centroid_phys_from_mask_xyz(mask_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
	"""
	mask_xyz: boolean or {0,1} mask aligned to CT, array order (X,Y,Z)
	affine:  4x4 NIfTI affine mapping index→physical in mm
	returns: (3,) centroid (X,Y,Z) in mm
	"""
	log("centroid_phys_from_mask_xyz")
	idx = np.argwhere(mask_xyz > 0)             # shape (N,3) in (x,y,z)
	if idx.size == 0:
		raise ValueError("Mask empty; cannot compute centroid.")
	c_idx = idx.mean(axis=0)                    # (x̄, ȳ, z̄) in index coords
	c_h = np.r_[c_idx, 1.0]
	c_phys = (affine @ c_h)[:3]
	log(f"centroid_phys_from_mask_xyz: c_phys = {c_phys}")               # (X,Y,Z) mm
	return c_phys

def origin_to_center_on_point(c_phys, size, spacing, affine):
	A = affine
	# Extract current direction and replace spacing with desired output spacing
	in_spacing = np.linalg.norm(A[:3,:3], axis=0)           # (sx,sy,sz)
	D = A[:3,:3] @ np.diag(1.0 / in_spacing)                # 3x3 orthonormal (up to sign)
	size = np.asarray(size, float)
	spacing = np.asarray(spacing, float)
	half_extent = (size - 1) * spacing / 2.0                # in voxel-extent mm (no rotation)
	origin_out = c_phys - D @ half_extent
	log(f"origin_to_center_on_point: origin_out = c_phys - D @ half_extent ---> {origin_out} = {c_phys} - {D} @ {half_extent}")
	return origin_out, D

def resample_centered_on_centroid(ct_img: sitk.Image, mask_img: sitk.Image,
								  size=(64,64,64), spacing=(1.0,1.0,1.0)):
	log("resample_centered_on_centroid")
	# 1) centroid from mask in physical space
	ls = sitk.LabelShapeStatisticsImageFilter()
	ls.Execute(mask_img>0)
	c_phys = ls.GetCentroid(1)  # (X,Y,Z) mm

	# 2) build direction matrix D from ct_img
	dir_flat = list(ct_img.GetDirection())       # length 9
	D = np.array(dir_flat, dtype=float).reshape(3,3)

	# 3) compute origin that centers c_phys
	size_arr = np.array(size, float)
	spacing_arr = np.array(spacing, float)
	half_extent = (size_arr - 1) * spacing_arr / 2.0
	origin_out = np.asarray(c_phys) - D @ half_extent

	# 4) anti-alias CT if downsampling then resample onto target grid
	gauss_sigma = 0.75  # mm (heuristic)
	ct_smooth = sitk.SmoothingRecursiveGaussian(ct_img, gauss_sigma)

	ct_out = sitk.Resample(
		ct_smooth,
		size=list(map(int,size_arr)),
		transform=sitk.Transform(),
		interpolator=sitk.sitkLinear,
		outputOrigin=tuple(origin_out),
		outputSpacing=tuple(spacing_arr),
		outputDirection=tuple(dir_flat),
		defaultPixelValue=float(sitk.GetArrayFromImage(ct_img).dtype.type(0))
	)
	mask_out = sitk.Resample(
		mask_img,
		referenceImage=ct_out,                   # identical grid
		transform=sitk.Transform(),
		interpolator=sitk.sitkNearestNeighbor,
		defaultPixelValue=0
	)
	return ct_out, mask_out

def heart_extents_mm(mask_img: sitk.Image, percentile=99.0) -> tuple:
	"""Return robust half-extents (Hx, Hy, Hz) in mm about the centroid."""
	# 1) voxel indices where mask > 0
	arr = sitk.GetArrayFromImage(mask_img)          # z,y,x
	idx_zyx = np.argwhere(arr > 0)                  # (N,3)
	if idx_zyx.size == 0:
		raise ValueError("Mask empty.")
	# convert to (x,y,z) index order expected by affine math below
	idx_xyz = idx_zyx[:, ::-1].astype(float)

	# 2) index→physical mapping
	origin = np.array(mask_img.GetOrigin())         # (X0,Y0,Z0)
	spacing = np.array(mask_img.GetSpacing())       # (sx,sy,sz)
	D = np.array(mask_img.GetDirection()).reshape(3,3)  # 3x3

	# physical coordinates of all foreground voxels
	phys = origin + (D @ (idx_xyz * spacing).T).T   # (N,3) mm

	# 3) centroid in physical space
	c_phys = phys.mean(axis=0)

	# 4) robust half-extents (percentile of absolute deviation)
	dev = np.abs(phys - c_phys)                     # (N,3)
	Hx, Hy, Hz = np.percentile(dev, percentile, axis=0)
	log(f"heart_extents_mm: (Hx, Hy, Hz), c_phys --> ({Hx}, {Hy}, {Hz}), {c_phys}")
	return (Hx, Hy, Hz), c_phys



def spacing_from_extents(H,
						 size=(64,64,64),
						 margin=0.15,
						 isotropic=True,
						 eps_mm=0.05,
						 pad_vox=2, extra_half_vox=True):
	H = np.array(H, float)                          # (Hx,Hy,Hz)
	N = np.array(size, float)
	half_vox = (N - 1.0)/2.0
	pad_term = float(pad_vox) + (0.5 if extra_half_vox else 0.0)
	denom = half_vox - pad_term                        # account for pad
	if np.any(denom <= 0):
		log(f"pad_vox 2 too large for size={size}. Need pad_vox < {(N-1)/2}.")
		raise ValueError(f"pad too large.")         # account for pad
	s_axes = H*(1.0 + margin) / denom
	if isotropic:
		s = float(np.max(s_axes)) + float(eps_mm)
		log(f"spacing_from_extents: (s, s, s): {(s, s, s)}")
		return (s, s, s)
	log(f"spacing_from_extents: (s_axes {s_axes} + float(eps_mm)): {tuple((s_axes + float(eps_mm)).tolist())}")
	return tuple((s_axes + float(eps_mm)).tolist())

def spacing_from_extents_for_96(H, margin=0.15, isotropic=True):
	log("spacing_from_extents_for_96")
	import numpy as np
	N = 96.0
	half_vox = (N - 1.0) / 2.0
	H = np.array(H, float)
	s_axes = H * (1.0 + margin) / half_vox
	if isotropic:
		s = float(np.max(s_axes))
		return (s, s, s)
	return tuple(s_axes.tolist())

def touches_border(mask_resampled: sitk.Image, pad_vox=2, relax_one=True) -> bool:
	arr = sitk.GetArrayFromImage(mask_resampled)  # z,y,x
	zmax, ymax, xmax = np.array(arr.shape) - 1
	# A pad of 'pad_vox' means we flag if any foreground exists at indices <pad or >dim-1-pad.
	z, y, x = np.where(arr > 0)
	if z.size == 0:
		return False
	pad = pad_vox - 1 if relax_one else pad_vox
	return ( (z <= pad).any() or (z >= zmax - pad).any()
		  or (y <= pad).any() or (y >= ymax - pad).any()
		  or (x <= pad).any() or (x >= xmax - pad).any() )

def adaptive_resample_centered(ct_img, heart_mask, size=(64,64,64),
							   base_spacing=(2.0,2.0,2.0), max_tries=3, growth=1.15):
	log("adaptive_resample_centered")
	"""Center on centroid; if mask touches border, inflate spacing and retry."""
	spacing = np.array(base_spacing, float)
	dir_flat = ct_img.GetDirection()
	D = np.array(dir_flat).reshape(3,3)
	ls = sitk.LabelShapeStatisticsImageFilter(); ls.Execute(heart_mask>0)
	c_phys = np.array(ls.GetCentroid(1))  # mm

	for _ in range(max_tries):
		# compute origin to center c_phys
		half_extent = (np.array(size,float) - 1) * spacing / 2.0
		origin_out = c_phys - D @ half_extent

		# smooth + resample CT
		ct_s = sitk.SmoothingRecursiveGaussian(ct_img, 0.75)
		ct_out = sitk.Resample(ct_s, size=list(map(int,size)),
							   transform=sitk.Transform(),
							   interpolator=sitk.sitkLinear,
							   outputOrigin=tuple(origin_out),
							   outputSpacing=tuple(spacing),
							   outputDirection=tuple(dir_flat),
							   defaultPixelValue=0.0)
		# resample mask and check borders
		mask_out = sitk.Resample(heart_mask, ct_out, sitk.Transform(),
								 sitk.sitkNearestNeighbor, 0)
		if not touches_border(mask_out, pad_vox=2):
			return ct_out, mask_out, tuple(spacing)
		spacing *= growth  # inflate and retry

	return ct_out, mask_out, tuple(spacing)  # last attempt result

def adaptive_center_resample_96(ct_img, heart_mask, percentile=99.0,
								margin=0.15, pad_vox=2,
								growth=1.12, max_tries=2):
	log("adaptive_center_resample_96")

	(Hx,Hy,Hz), c_phys = heart_extents_mm(heart_mask, percentile)
	spacing = np.array(spacing_from_extents_for_96((Hx,Hy,Hz), margin), float)
	size = (96,96,96)

	D = np.array(ct_img.GetDirection()).reshape(3,3)
	dir_flat = ct_img.GetDirection()

	for _ in range(max_tries + 1):
		half_extent = (np.array(size,float) - 1) * spacing / 2.0
		origin_out = c_phys - D @ half_extent

		ct_s = sitk.SmoothingRecursiveGaussian(ct_img, 0.75)  # anti-alias if downsampling
		ct_out = sitk.Resample(ct_s, size, sitk.Transform(), sitk.sitkLinear,
							   tuple(origin_out), tuple(spacing), tuple(dir_flat), 0.0)
		mask_out = sitk.Resample(heart_mask, ct_out, sitk.Transform(),
								 sitk.sitkNearestNeighbor, 0)

		if not touches_border(mask_out, pad_vox=pad_vox):
			return ct_out, mask_out, tuple(spacing)

		spacing *= growth  # enlarge FOV and retry

	# Return last attempt if still touching (very rare)
	return ct_out, mask_out, tuple(spacing)









# --- helpers ---


def resample_like(moving: sitk.Image,
				  reference: sitk.Image,
				  interp=sitk.sitkNearestNeighbor,
				  default_value: float = 0) -> sitk.Image:
	"""Resample 'moving' to match 'reference' geometry."""
	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(reference)
	resampler.SetInterpolator(interp)
	resampler.SetDefaultPixelValue(default_value)
	# Preserve orientation/spacing/origin from reference via SetReferenceImage
	return resampler.Execute(moving)

def binarise(img: sitk.Image, thr: float = 0.5) -> sitk.Image:
	# Handles 0/1, probabilities, or stray intensities the same way
	out = sitk.BinaryThreshold(img, lowerThreshold=thr, upperThreshold=1e9,
							   insideValue=1, outsideValue=0)
	return ensure_uint8(out)

def remove_tiny_islands(mask: sitk.Image, min_voxels: int = 50) -> sitk.Image:
	"""Optional: clean spurious speckles."""
	cc = sitk.ConnectedComponent(mask)
	stats = sitk.LabelShapeStatisticsImageFilter()
	stats.Execute(cc)
	keep = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
	keep.CopyInformation(mask)
	for lbl in stats.GetLabels():
		if stats.GetNumberOfPixels(lbl) >= min_voxels:
			keep = keep | sitk.Equal(cc, lbl)
	return ensure_uint8(keep)



# --- main hook you add into your pipeline ---
def make_lv_myo_masks(
	ct_resampled: sitk.Image,
	# EITHER a single multi-label map (0=bg, 1=LV, 2=MYO, ...)
	labelmap_resampled: Optional[sitk.Image] = None,
	# OR individual (already resampled) volumes for LV and MYO (binary or soft)
	per_struct_resampled: Optional[Dict[str, sitk.Image]] = None,
	lv_label: int = 1, myo_label: int = 2,
	clean_small_islands: bool = True
):
	"""
	Returns: (lv_mask_uint8, myo_mask_uint8) aligned to ct_resampled (0/1).
	Makes masks mutually exclusive (MYO wins only where LV is absent).
	"""
	assert (labelmap_resampled is not None) ^ (per_struct_resampled is not None), \
		"Provide either a labelmap OR per-structure volumes, not both."

	if labelmap_resampled is not None:
		# Ensure geometry alignment
		labelmap_resampled = resample_like(labelmap_resampled, ct_resampled,
										   interp=sitk.sitkNearestNeighbor, default_value=0)
		lv = binary_from_labelmap(labelmap_resampled, lv_label)
		myo = binary_from_labelmap(labelmap_resampled, myo_label)
	else:
		# Expect keys: 'LV', 'MYO'
		lv = resample_like(per_struct_resampled['LV'], ct_resampled,
						   interp=sitk.sitkNearestNeighbor, default_value=0)
		myo = resample_like(per_struct_resampled['MYO'], ct_resampled,
							interp=sitk.sitkNearestNeighbor, default_value=0)
		# Convert to hard 0/1 if not already
		lv = binary_from_prob_or_binary(lv, thresh=0.5)
		myo = binary_from_prob_or_binary(myo, thresh=0.5)

	# Enforce mutual exclusivity (no overlapping voxels)
	# If there is overlap, prefer myocardium to remain outside the cavity:
	# remove LV from MYO, or vice versa depending on your anatomical convention.
	overlap = lv & myo
	if sitk.StatisticsImageFilter().Execute(overlap) or True:  # always resolve just in case
		myo = myo & sitk.BinaryNot(lv)

	# Optional: denoise tiny speckles
	if clean_small_islands:
		lv = remove_tiny_islands(lv, min_voxels=50)
		myo = remove_tiny_islands(myo, min_voxels=50)

	# Ensure meta matches CT
	lv.CopyInformation(ct_resampled)
	myo.CopyInformation(ct_resampled)
	return lv, myo

# --- one-hot view (usually do this in the dataloader) ---
def masks_to_one_hot(lv_mask: sitk.Image, myo_mask: sitk.Image, as_numpy=True):
	"""
	Returns a 2-channel one-hot (C=2, Z, Y, X) if as_numpy,
	otherwise a 4D NIfTI (Z,Y,X,C). Prefer doing this at training time.
	"""
	zyx_lv = sitk.GetArrayFromImage(lv_mask).astype(np.uint8)
	zyx_myo = sitk.GetArrayFromImage(myo_mask).astype(np.uint8)
	onehot = np.stack([zyx_lv, zyx_myo], axis=0)  # (C, Z, Y, X)
	if as_numpy:
		return onehot
	# If you really want a NIfTI 4D, SimpleITK expects (t,z,y,x) ordering from numpy:
	onehot_4d = np.moveaxis(onehot, 0, -1)  # (Z, Y, X, C)
	out = sitk.GetImageFromArray(onehot_4d)  # 4D
	out.CopyInformation(lv_mask)             # copies spacing/origin/direction for first 3 dims
	return out
