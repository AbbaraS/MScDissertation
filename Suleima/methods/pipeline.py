from pathlib import Path
import json
import numpy as np
import SimpleITK as sitk
from methods.preprocessing import *
#from methods.preprocessing import make_lv_myo_masks_from_binaries
#assert callable(make_lv_myo_masks_from_binaries)

import os
from core.Case import *
from core.Log import log

import SimpleITK as sitk
import numpy as np
from typing import Optional, Dict

from typing import Optional, Tuple, Dict, Sequence, List


INCLUDE_FILES = [
	"heart_ventricle_left.nii.gz",
	"heart_ventricle_right.nii.gz",
	"heart_atrium_left.nii.gz",
	"heart_atrium_right.nii.gz",
	"heart_myocardium.nii.gz",
]

# Map TS filenames to canonical keys used by make_heart_mask_from_binaries(include=...)
TS_KEYMAP = {
	"heart_myocardium.nii.gz":     	  "myocardium",
	"heart_ventricle_left.nii.gz": 	  "left_ventricle",
	"heart_ventricle_right.nii.gz":	  "right_ventricle",
	"heart_atrium_left.nii.gz":    	  "left_atrium",
	"heart_atrium_right.nii.gz":   	  "right_atrium",
}

def load_ct(ct_path: Path) -> sitk.Image:
	if not ct_path.exists():
		return None
	img = sitk.ReadImage(str(ct_path))
	# NOTE: The CT is *not* intensity-clipped here, per your instruction.
	log(f"load_ct: shape: {img.GetSize()}, spacing: {img.GetSpacing()}, origin: {img.GetOrigin()}")
	return img

def load_available_totalseg_masks(segments_dir: Path) -> dict:
	"""
	Returns {canonical_key: sitk.Image} for any of the INCLUDE_FILES that exist.
	"""
	out = {}
	files = []
	if not segments_dir.exists():
		return out
	for fname in INCLUDE_FILES:
		f = segments_dir / fname
		if f.exists():
			key = TS_KEYMAP[fname]
			out[key] = sitk.ReadImage(str(f))
			files.append(fname)
	log(f"_load_available_totalseg_masks: {files}")
	return out



# --- helpers ---
def ensure_float32(img: sitk.Image) -> sitk.Image:
	return sitk.Cast(img, sitk.sitkFloat32)

def ensure_uint8(img: sitk.Image) -> sitk.Image:
	return sitk.Cast(img, sitk.sitkUInt8)



def union_binary_masks(masks: list[sitk.Image], reference: sitk.Image) -> sitk.Image:
	if not masks:
		raise ValueError("union_binary_masks: empty input list")
	out = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
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

def make_lv_myo_masks_from_binaries(
    reference_ct: sitk.Image,
    totalseg_parts: Dict[str, sitk.Image],
    lv_keys: Sequence[str] = ("left_ventricle",),
    myo_keys: Sequence[str] = ("myocardium",),
    clean_islands: bool = True,
    min_voxels: int = 50,
    constrain_to_heart: bool = True,
    precomputed_heart: Optional[sitk.Image] = None,
	) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
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
	def union(keys):
		masks = [totalseg_parts[k] for k in keys if k in totalseg_parts]
		if not masks:
			# Generate empty aligned mask if missing
			empty = sitk.Image(reference_ct.GetSize(), sitk.sitkUInt8); empty.CopyInformation(reference_ct)
			return empty
		return union_binary_masks(masks, reference_ct)

	lv_mask  = union(lv_keys)
	myo_mask = union(myo_keys)

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




def gaussian_if_downsampling(ct_img: sitk.Image, target_spacing_xyz: tuple[float,float,float]) -> sitk.Image:
	"""
	Apply a light anti-alias Gaussian if any target spacing axis is *larger* than input spacing
	(i.e., we are downsampling along that axis). Otherwise, return as-is.
	"""
	in_sp = np.array(ct_img.GetSpacing(), float)
	tgt = np.array(target_spacing_xyz, float)
	if np.any(tgt > in_sp + 1e-6):
		log(f"_gaussian_if_downsampling: tgt {tgt} > in_sp {in_sp} + 1e-6 , applying Gaussian")
		# Heuristic sigma; safe and effective for ~1–3 mm spacings
		return sitk.SmoothingRecursiveGaussian(ct_img, 0.75)
	log(f"_gaussian_if_downsampling: tgt {tgt} <= in_sp {in_sp} + 1e-6 , no Gaussian applied")
	return ct_img

def gaussian_if_downsampling_axes(img: sitk.Image, target_spacing):
	in_sp = np.array(img.GetSpacing(), float)
	tgt   = np.array(target_spacing, float)
	out = img
	for axis in range(3):
		if tgt[axis] > in_sp[axis] + 1e-6:          # downsampling on this axis
			out = sitk.RecursiveGaussian(out, sigma=0.75, direction=axis)
	return out
	# then use ct_pre = gaussian_if_downsampling_axes(ct_img, spacing_arr)

def resample_centered(ct_img: sitk.Image,
					   heart_mask: sitk.Image,
					   size=(64,64,64),
					   spacing=(2.0,2.0,2.0)):
	"""
	Center grid on the heart centroid, resample CT (linear) + mask (NN) to (size, spacing).
	"""
	# 1) centroid in physical space (mm)
	ls = sitk.LabelShapeStatisticsImageFilter()
	ls.Execute(heart_mask > 0)
	if not ls.HasLabel(1):
		log("resample_centered: Heart mask is empty after union/alignment.")
		raise ValueError("resample_centered: Heart mask is empty after union/alignment.")
	c_phys = np.array(ls.GetCentroid(1), float)  # (X,Y,Z) mm

	# 2) output geometry: keep orientation of CT
	D = np.array(ct_img.GetDirection()).reshape(3,3)
	dir_flat = ct_img.GetDirection()
	N = np.array(size, float)
	spacing_arr = np.array(spacing, float)
	half_extent = (N - 1.0) * spacing_arr / 2.0
	origin_out = c_phys - D @ half_extent
	log(f"resample_centered: ( N {N} - 1.0) * spacing_arr {spacing_arr} / 2.0  = half_extent {half_extent}")
	log(f"resample_centered: \nc_phys {c_phys} - D {D} @ half_extent ---> origin_out = \n{origin_out}")

	# 3) anti-alias only if downsampling
	#ct_pre = gaussian_if_downsampling(ct_img, tuple(spacing_arr))
	ct_pre = gaussian_if_downsampling_axes(ct_img, tuple(spacing_arr))

	AIR_HU = -1024.0

	# 4) resample CT, then mask to CT grid
	#interp = sitk.sitkBSpline
	interp = sitk.sitkLinear

	ct_out = sitk.Resample(
		ct_pre,
		size=[int(x) for x in N],
		transform=sitk.Transform(),
		interpolator=interp,
		outputOrigin=tuple(origin_out),
		outputSpacing=tuple(spacing_arr),
		outputDirection=tuple(dir_flat),
		defaultPixelValue=AIR_HU
	)
	mask_out = sitk.Resample(
		heart_mask,
		ct_out,
		sitk.Transform(),
		sitk.sitkNearestNeighbor,
		0
	)
	return ct_out, mask_out, tuple(spacing_arr), tuple(origin_out)

def adaptive_resample_centered(ct_img: sitk.Image,
								heart_mask: sitk.Image,
								base_spacing: tuple[float,float,float],
								size=(64,64,64),
								pad_vox=2,
								growth=1.08,
								max_tries=2):
	"""
	Center on centroid; if heart touches border, inflate spacing and retry.
	"""
	spacing = np.array(base_spacing, float)
	last = None
	total_attempts = int(max_tries) + 1
	for attempt in range(1, total_attempts + 1):
		ct_out, mask_out, spacing_used, origin_used = resample_centered(
			ct_img, heart_mask, size=size, spacing=tuple(spacing)
		)

		# store this attempt as 'last' in case we exhaust retries
		last = (ct_out, mask_out, spacing_used, origin_used)
		touches_b = touches_border(mask_out, pad_vox=pad_vox)

		if not touches_b:
			log(f"adaptive_resample_centered: attempt {attempt} PASS | touches_border: {touches_b} spacing_used={spacing_used}, origin_used={origin_used}")
			return ct_out, mask_out, spacing_used, origin_used

		# Failed the border check:
		log(f"adaptive_resample_centered: attempt {attempt} FAIL | touches_border: {touches_b} spacing_used={spacing_used}, origin_used={origin_used}")

		# If more attempts remain, grow spacing for the *next* attempt
		if attempt < total_attempts:
			spacing = spacing * float(growth)  # enlarge FOV and retry

	# Rare: return the last (failed) attempt result
	log(f"adaptive_resample_centered: exhausted retries; returning last attempt with spacing={last[2]}")
	return last

def process_case_centroid_FOV(case_path,
							  size=(64,64,64),
							  percentile=99.0,
							  margin=0.15,
							  pad_vox=2,
							  growth=1.12,
							  max_tries=2,
							  outdir_name="centered64"):

	log("process_case_centroid_FOV")
	# --- resolve case path ---
	if not isinstance(case_path, (str, Path)):
		raise ValueError("Provide a case Path with accessible path attributes.")

	ct_path = case_path / "fullCT.nii.gz"
	seg_dir = case_path / "segments"
	out_dir = case_path / outdir_name
	ct_out_path   = out_dir / "ct_centered.nii.gz"
	heart_mask_path = out_dir / "heartmask_centered.nii.gz"
	lv_mask_path    = out_dir / "lv_mask.nii.gz"
	myo_mask_path   = out_dir / "myo_mask.nii.gz"
	meta_path     = out_dir / "centered_meta.json"
	out_dir.mkdir(parents=True, exist_ok=True)

	# --- load CT & TS masks ---
	ct_img = load_ct(ct_path)
	ct_img = sitk.Cast(ct_img, sitk.sitkFloat32)

	totalseg_parts = load_available_totalseg_masks(seg_dir)
	if not totalseg_parts:
		raise FileNotFoundError(f"No TotalSegmentator masks found in: {seg_dir}")

	# --- build binary heart mask aligned to CT ---
	heart_mask = make_heart_mask_from_binaries(ct_img, totalseg_parts)  # uses union_binary_masks -> alignment+NN
	heart_mask = sitk.Cast(heart_mask, sitk.sitkUInt8)
	# --- extents (mm) + centroid (mm) ---
	(Hx, Hy, Hz), c_phys = heart_extents_mm(heart_mask, percentile=percentile)

	# --- per-case isotropic spacing for requested size ---
	spacing = spacing_from_extents((Hx,Hy,Hz), size=size, margin=margin, isotropic=True)

	# --- centered resample with runtime safety check ---
	ct_out, mask_out, spacing_used, origin_used = adaptive_resample_centered(
		ct_img, heart_mask,
		base_spacing=spacing,
		size=size,
		pad_vox=pad_vox,
		growth=growth,
		max_tries=max_tries)

	ct_out = ensure_float32(ct_out)


	'''
	Fixed, cardiac-appropriate HU window --> clinically meaningful HU range for cardiac CT [-1000, 1200]
	'''
	# 1) Clip HU on the resampled image
	lower_HU, upper_HU = -1000.0, 1200.0
	ct_clipped = sitk.Clamp(ct_out, lowerBound=lower_HU, upperBound=upper_HU)


	lv_mask, myo_mask, heart_mask = make_lv_myo_masks_from_binaries(
			reference_ct=ct_clipped,
			totalseg_parts=totalseg_parts,
			lv_keys=("left_ventricle",),
			myo_keys=("myocardium",),
			clean_islands=True,
			min_voxels=50,
			constrain_to_heart=True,
			precomputed_heart=mask_out  )

	sitk.WriteImage(ct_clipped,   str(ct_out_path), False)
	#sitk.WriteImage(mask_out, str(mask_out_path), True)
	# Save (uint8, 0/1)
	sitk.WriteImage(heart_mask, str(heart_mask_path), useCompression=False)
	sitk.WriteImage(lv_mask,    str(lv_mask_path),    useCompression=False)
	sitk.WriteImage(myo_mask,   str(myo_mask_path),   useCompression=False)
	'''
	Global z‑score normalization (train‑set statistics only).
	Compute μ, σ on the training set only (after clipping).
	Apply to train/val/test: x_norm = (x_clip − μ_train) / σ_train
	Store μ, σ in metadata to guarantee reproducibility and avoid data leakage.
	'''

	meta = {
		"size": list(map(int, size)),
		"spacing_initial": list(map(float, spacing)),
		"spacing_final": list(map(float, spacing_used)),
		"origin_final_mm": list(map(float, origin_used)),
		"centroid_mm": list(map(float, c_phys)),
		"percentile": float(percentile),
		"margin": float(margin),
		"pad_vox": int(pad_vox),
		"growth": float(growth),
		"max_tries": int(max_tries),
		"inputs": {
			"ct_path": str(ct_path),
			"segments_dir": str(seg_dir),
			"used_structures": [k for k in totalseg_parts.keys()]
		}
	}
	#with open(meta_path, "w") as f:
	#	json.dump(meta, f, indent=2)

	return {
		"ct_out_path": ct_out_path,
		"mask_out_path": totalseg_parts,
		"meta_path": meta_path,
		"spacing_final": spacing_used,
		"origin_final": origin_used,
		"centroid_mm": tuple(c_phys),
	}
