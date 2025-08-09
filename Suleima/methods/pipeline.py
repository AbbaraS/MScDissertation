from pathlib import Path
import json
import numpy as np
import SimpleITK as sitk
from methods.preprocessing import *
# --------------------------------------------------------------------
# Expected helpers (from your library; not redefined here):
#   same_geometry(a, b) -> bool
#   resample_to_reference_nn(img, reference, default=0) -> sitk.Image
#   to_uint8_binary(img) -> sitk.Image
#   union_binary_masks(masks, reference) -> sitk.Image
#   make_heart_mask_from_binaries(reference_ct, totalseg_parts, include=...) -> sitk.Image
#   heart_extents_mm(mask_img, percentile=99.0) -> ((Hx,Hy,Hz), c_phys)
#   spacing_from_extents(H, size=(64,64,64), margin=0.15, isotropic=True) -> (sx,sy,sz)
#   touches_border(mask_resampled, pad_vox=2) -> bool
# --------------------------------------------------------------------

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
		log("_resample_centered: Heart mask is empty after union/alignment.")
		raise ValueError("_resample_centered: Heart mask is empty after union/alignment.")
	c_phys = np.array(ls.GetCentroid(1), float)  # (X,Y,Z) mm

	# 2) output geometry: keep orientation of CT
	D = np.array(ct_img.GetDirection()).reshape(3,3)
	dir_flat = ct_img.GetDirection()
	N = np.array(size, float)
	spacing_arr = np.array(spacing, float)
	half_extent = (N - 1.0) * spacing_arr / 2.0
	origin_out = c_phys - D @ half_extent
	log(f"_resample_centered: ( N {N} - 1.0) * spacing_arr {spacing_arr} / 2.0  = half_extent {half_extent}")
	log(f"_resample_centered: \nc_phys {c_phys} - D {D} @ half_extent ---> origin_out = \n{origin_out}")

	# 3) anti-alias only if downsampling
	#ct_pre = gaussian_if_downsampling(ct_img, tuple(spacing_arr))
	ct_pre = gaussian_if_downsampling_axes(ct_img, tuple(spacing_arr))

	AIR_HU = -1024.0

	# 4) resample CT, then mask to CT grid
	interp = sitk.sitkBSpline
	#interp = sitk.sitkLinear

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
			log(f"_adaptive_resample_centered: attempt {attempt} PASS | touches_border: {touches_b} spacing_used={spacing_used}, origin_used={origin_used}")
			return ct_out, mask_out, spacing_used, origin_used

		# Failed the border check:
		log(f"_adaptive_resample_centered: attempt {attempt} FAIL | touches_border: {touches_b} spacing_used={spacing_used}, origin_used={origin_used}")

		# If more attempts remain, grow spacing for the *next* attempt
		if attempt < total_attempts:
			spacing = spacing * float(growth)  # enlarge FOV and retry

	# Rare: return the last (failed) attempt result
	log(f"_adaptive_resample_centered: exhausted retries; returning last attempt with spacing={last[2]}")
	return last

def process_case_centroid_FOV(case_path,
							  size=(64,64,64),
							  percentile=99.0,
							  margin=0.15,
							  pad_vox=2,
							  growth=1.12,
							  max_tries=2,
							  outdir_name="centered64"):
	"""
	Pipeline:
	  - Load CT and available TS masks (from the five specified files).
	  - Align masks to CT, union → heart mask (uint8).
	  - Compute robust half-extents (mm) and centroid (mm).
	  - Compute isotropic spacing for the requested 'size' using H + margin.
	  - Centered resample (CT: linear + anti-alias if downsampling; mask: NN).
	  - Runtime border safety check; inflate spacing if needed and retry.
	  - Save CT, mask, and a small JSON metadata file.
	"""
	log("process_case_centroid_FOV")
	# --- resolve case path ---
	if not isinstance(case_path, (str, Path)):
		raise ValueError("Provide a case Path with accessible path attributes.")

	ct_path = case_path / "fullCT.nii.gz"
	seg_dir = case_path / "segments"
	out_dir = case_path / outdir_name
	out_dir.mkdir(parents=True, exist_ok=True)

	# --- load CT & TS masks ---
	ct_img = load_ct(ct_path)
	ct_img = sitk.Cast(ct_img, sitk.sitkFloat32)
	parts = load_available_totalseg_masks(seg_dir)
	if not parts:
		raise FileNotFoundError(f"No TotalSegmentator masks found in: {seg_dir}")

	# --- build binary heart mask aligned to CT ---
	heart_mask = make_heart_mask_from_binaries(ct_img, parts)  # uses union_binary_masks -> alignment+NN
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
		max_tries=max_tries
	)

	ct_out = ensure_float32(ct_out)
	# --- save outputs ---
	ct_out_path   = out_dir / "ct_centered.nii.gz"
	mask_out_path = out_dir / "heartmask_centered.nii.gz"
	meta_path     = out_dir / "centered_meta.json"


	'''
	Fixed, cardiac-appropriate HU window --> clinically meaningful HU range for cardiac CT [-1000, 1200]
	'''
	# 1) Clip HU on the resampled image
	lower_HU, upper_HU = -1000.0, 1200.0
	ct_clipped = sitk.Clamp(ct_out, lowerBound=lower_HU, upperBound=upper_HU)

	#sitk.WriteImage(ct_clipped,   str(ct_out_path), True)
	#sitk.WriteImage(mask_out, str(mask_out_path), True)

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
			"used_structures": [k for k in parts.keys()]
		}
	}
	#with open(meta_path, "w") as f:
	#	json.dump(meta, f, indent=2)

	return {
		"ct_out_path": ct_out_path,
		"mask_out_path": mask_out_path,
		"meta_path": meta_path,
		"spacing_final": spacing_used,
		"origin_final": origin_used,
		"centroid_mm": tuple(c_phys),
	}
