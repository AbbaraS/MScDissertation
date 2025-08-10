from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional, List

import json
import numpy as np
import SimpleITK as sitk


# ----------------------------- Logging ----------------------------------------
def log(msg: str) -> None:
	print(msg)


# ----------------------------- Config -----------------------------------------
INCLUDE_FILES = [
	"heart_ventricle_left.nii.gz",
	"heart_ventricle_right.nii.gz",
	"heart_atrium_left.nii.gz",
	"heart_atrium_right.nii.gz",
	"heart_myocardium.nii.gz",
]

TS_KEYMAP = {
	"heart_myocardium.nii.gz":      "myocardium",
	"heart_ventricle_left.nii.gz":  "left_ventricle",
	"heart_ventricle_right.nii.gz": "right_ventricle",
	"heart_atrium_left.nii.gz":     "left_atrium",
	"heart_atrium_right.nii.gz":    "right_atrium",
}

AIR_HU = -1024.0

import nibabel as nib
from core.NiftiVolume import NiftiVolume
# ----------------------------- I/O --------------------------------------------

def sitk_img(vol: NiftiVolume):
	img = sitk.GetImageFromArray(vol.data.transpose(2,1,0))
	img.SetDirection(tuple(vol.affine[:3, :3].flatten()))
	img.SetOrigin(tuple(vol.affine[:3, 3]))
	img.SetSpacing(tuple(np.sqrt((vol.affine[:3, :3]**2).sum(axis=0))))
	return img

def load_ct(ct_path: Path) -> sitk.Image:
	if not ct_path.exists():
		raise FileNotFoundError(f"Missing CT: {ct_path}")
	obj = NiftiVolume(ct_path)
	img = sitk_img(obj)
	# work in float32 exactly once
	img = sitk.Cast(img, sitk.sitkFloat32)
	log(f"CT: size={img.GetSize()}, spacing={img.GetSpacing()}, origin={img.GetOrigin()}")
	return img


def load_totalseg_as_dict(segments_dir: Path) -> Dict[str, sitk.Image]:
	out: Dict[str, sitk.Image] = {}
	if not segments_dir.exists():
		return out
	present = []
	for fname in INCLUDE_FILES:
		f = segments_dir / fname
		if f.exists():
			obj = NiftiVolume(f)
			out[TS_KEYMAP[fname]] = sitk_img(obj)
			present.append(fname)
	log(f"TS parts found: {present}")
	return out


# ----------------------------- Small utils ------------------------------------
def binarise(img: sitk.Image, thr: float = 0.5) -> sitk.Image:
	return sitk.BinaryThreshold(img, thr, 1e9, 1, 0)  # uint8 inside, 0 outside by default


def ensure_uint8_like(ref: sitk.Image) -> sitk.Image:
	out = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
	out.CopyInformation(ref)
	return out


def resample_like(moving: sitk.Image,
				  reference: sitk.Image,
				  interp=sitk.sitkNearestNeighbor,
				  default_value: float = 0) -> sitk.Image:
	R = sitk.ResampleImageFilter()
	R.SetReferenceImage(reference)
	R.SetInterpolator(interp)
	R.SetDefaultPixelValue(default_value)
	return R.Execute(moving)


def smooth_if_downsampling(ct_img: sitk.Image, target_spacing_xyz: Tuple[float, float, float]) -> sitk.Image:
	in_sp = np.array(ct_img.GetSpacing(), float)
	tgt = np.array(target_spacing_xyz, float)
	if np.any(tgt > in_sp + 1e-6):
		# light, separable, stable
		out = ct_img
		for axis in range(3):
			if tgt[axis] > in_sp[axis] + 1e-6:
				out = sitk.RecursiveGaussian(out, sigma=0.75, direction=axis)
		return out
	return ct_img


# ----------------------------- Masks & extents --------------------------------
def align_ts_to_ct(ts_parts: Dict[str, sitk.Image], ct: sitk.Image) -> Dict[str, sitk.Image]:
	if not ts_parts:
		raise FileNotFoundError("No TotalSegmentator masks provided.")
	aligned = {}
	for k, m in ts_parts.items():
		# single NN alignment, then binarise once
		m_ct = resample_like(m, ct, interp=sitk.sitkNearestNeighbor, default_value=0)
		aligned[k] = binarise(m_ct, 0.5)
	return aligned


def union_masks(masks: Sequence[sitk.Image], reference: sitk.Image) -> sitk.Image:
	if not masks:
		raise ValueError("union_masks: empty list.")
	out = ensure_uint8_like(reference)
	for m in masks:
		out = out | m  # already aligned + binary
	return out


def make_heart_lv_myo(ct: sitk.Image,
					  ts_aligned: Dict[str, sitk.Image],
					  lv_keys: Sequence[str] = ("left_ventricle",),
					  myo_keys: Sequence[str] = ("myocardium",),
					  island_min_voxels: int = 50) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
	# heart = union of selected structures
	keys_heart = ("myocardium", "left_ventricle", "right_ventricle", "left_atrium", "right_atrium")
	heart = union_masks([ts_aligned[k] for k in keys_heart if k in ts_aligned], ct)

	def _union(keys):
		present = [ts_aligned[k] for k in keys if k in ts_aligned]
		return union_masks(present, ct) if present else ensure_uint8_like(ct)

	lv = _union(lv_keys)
	myo = _union(myo_keys)

	# island cleanup (optional but cheap)
	if island_min_voxels > 0:
		for name, var in [("lv", lv), ("myo", myo)]:
			cc = sitk.ConnectedComponent(var)
			stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
			keep = ensure_uint8_like(ct)
			for lbl in stats.GetLabels():
				if stats.GetNumberOfPixels(lbl) >= island_min_voxels:
					keep = keep | sitk.Equal(cc, lbl)
			if name == "lv":  lv = keep
			else:             myo = keep

	# enforce MYO excludes LV cavity
	myo = myo & sitk.BinaryNot(lv)

	# constrain to heart
	lv  = lv  & heart
	myo = myo & heart
	return lv, myo, heart

def mask_sanity(mask: sitk.Image, max_allowed_extent_mm: float = 150.0) -> None:
	arr = sitk.GetArrayFromImage(mask)
	if np.count_nonzero(arr) == 0:
		raise ValueError("Heart mask empty.")
	frac = np.count_nonzero(arr) / arr.size
	if frac > 0.10:  # 10% of the volume foreground is suspicious for a heart
		log(f"WARNING: heart mask foreground fraction unusually large: {frac:.3f}")

	# Optional: fast AABB in index coords as a cheap extent bound
	stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(mask > 0)
	x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
	sp = np.array(mask.GetSpacing(), float)
	lengths = sp * np.array([max(sx - 1, 1), max(sy - 1, 1), max(sz - 1, 1)], float)
	if np.max(lengths) > 2 * max_allowed_extent_mm:
		log(f"WARNING: AABB extent too large ({lengths} mm) — likely far‑away islands or misalignment.")

def half_extents_and_centroid_from_bbox(mask: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
	"""Return (half_extents_mm[3], centroid_mm[3]) from labelstats bounding box."""
	lab = sitk.BinaryThreshold(mask, 0.5, 1e9, 1, 0)  # ensure binary
	stats = sitk.LabelShapeStatisticsImageFilter()
	stats.Execute(lab)
	if not stats.HasLabel(1):
		raise ValueError("Mask empty for bbox computation.")

	# Bounding box in index space: (x, y, z, sizeX, sizeY, sizeZ)
	x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
	# centroid (mm) is already provided in physical space
	c_mm = np.array(stats.GetCentroid(1), dtype=np.float64)

	sp = np.asarray(mask.GetSpacing(), dtype=np.float64)
	# size in *voxels* along each axis → physical length ≈ (size-1)*spacing
	lengths_mm = sp * np.array([max(sx - 1, 1), max(sy - 1, 1), max(sz - 1, 1)], dtype=np.float64)
	H_mm = 0.5 * lengths_mm
	return H_mm, c_mm


def robust_extents_mm(mask: sitk.Image, percentile: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (half-extents_mm[3], centroid_mm[3]) w.r.t. image physical space, respecting direction."""
	arr = sitk.GetArrayFromImage(mask)  # z,y,x
	if np.count_nonzero(arr) == 0:
		raise ValueError("robust_extents_mm: empty mask.")
	idx_zyx = np.argwhere(arr > 0)
	idx_xyz = idx_zyx[:, ::-1].astype(np.float64)

	origin = np.asarray(mask.GetOrigin(), dtype=np.float64)
	spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
	D = np.asarray(mask.GetDirection(), dtype=np.float64).reshape(3, 3)

	phys = origin + (D @ (idx_xyz * spacing).T).T
	c = phys.mean(axis=0)
	dev = np.abs(phys - c)
	Hx, Hy, Hz = np.percentile(dev, percentile, axis=0)
	return np.array([Hx, Hy, Hz], dtype=np.float64), c


def spacing_for_fill(H_mm: np.ndarray,
					 out_size=(64, 64, 64),
					 margin: float = 0.10,
					 pad_vox: int = 1,
					 eps_mm: float = 0.05,
					 isotropic: bool = True) -> Tuple[float, float, float]:
	"""
	Choose spacing so (heart + margin) fits within the interior voxels after a 'pad' on each side.
	Larger pad_vox gives more safety near the borders; smaller margin pushes tighter filling.
	"""
	N = np.asarray(out_size, dtype=np.float64)
	half_vox = (N - 1.0) / 2.0
	pad_term = float(pad_vox) + 0.5  # reserve pad and half-voxel
	denom = half_vox - pad_term
	if np.any(denom <= 0):
		raise ValueError(f"pad too large for out_size={out_size}")

	# per-axis spacing needed
	s_axes = H_mm * (1.0 + margin) / denom
	if isotropic:
		s = float(np.max(s_axes)) + eps_mm
		return (s, s, s)
	return tuple((s_axes + eps_mm).tolist())


def touches_border(mask_resampled: sitk.Image, pad_vox: int = 1) -> bool:
	arr = sitk.GetArrayFromImage(mask_resampled)  # z,y,x
	if np.count_nonzero(arr) == 0:
		return False
	z, y, x = np.where(arr > 0)
	zmax, ymax, xmax = np.array(arr.shape) - 1
	p = pad_vox
	return ((z <= p).any() or (z >= zmax - p).any() or
			(y <= p).any() or (y >= ymax - p).any() or
			(x <= p).any() or (x >= xmax - p).any())


# ----------------------------- Output geometry --------------------------------
def build_output_ref(ct_img: sitk.Image,
					 centroid_mm: np.ndarray,
					 out_size=(64, 64, 64),
					 out_spacing=(2.0, 2.0, 2.0)) -> sitk.Image:
	"""
	Create a blank reference image centered on centroid with CT orientation.
	"""
	N = np.asarray(out_size, dtype=np.int64)
	s = np.asarray(out_spacing, dtype=np.float64)
	D = np.asarray(ct_img.GetDirection(), dtype=np.float64).reshape(3, 3)
	half_extent = (N - 1.0) * s / 2.0
	origin_out = centroid_mm - D @ half_extent

	ref = sitk.Image(int(N[0]), int(N[1]), int(N[2]), sitk.sitkFloat32)
	ref.SetSpacing(tuple(s.tolist()))
	ref.SetDirection(tuple(D.reshape(-1).tolist()))
	ref.SetOrigin(tuple(origin_out.tolist()))
	return ref

def align_and_binarise_to_ct(ts_parts: Dict[str, sitk.Image], ct: sitk.Image) -> Dict[str, sitk.Image]:
    out = {}
    for k, m in ts_parts.items():
        m_al = sitk.Resample(m, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        m_bin = sitk.BinaryThreshold(m_al, 0, 1e9, 1, 0)  # >0 → 1
        out[k] = m_bin
    return out


def keep_largest(bin_img: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(bin_img)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    if not stats.GetLabels(): return bin_img
    L = max(stats.GetLabels(), key=lambda x: stats.GetNumberOfPixels(x))
    return sitk.Equal(cc, L)

def build_heart_lv_myo_strict(ct: sitk.Image, ts_aligned_bin: Dict[str, sitk.Image]):
    keys_heart = ("myocardium","left_ventricle","right_ventricle","left_atrium","right_atrium")
    heart = sitk.Image(ct.GetSize(), sitk.sitkUInt8); heart.CopyInformation(ct)
    for k in keys_heart:
        if k in ts_aligned_bin: heart = heart | ts_aligned_bin[k]

    # sanity & safeguard
    arr = sitk.GetArrayFromImage(heart)
    frac = np.count_nonzero(arr) / arr.size
    if frac > 0.20:   # too big for a heart
        heart = keep_largest(heart)

    lv  = ts_aligned_bin.get("left_ventricle", heart*0) & heart
    myo = ts_aligned_bin.get("myocardium",     heart*0) & heart
    myo = myo & sitk.BinaryNot(lv)

    return lv, myo, heart


def half_extents_from_bbox(mask: sitk.Image):
    lab = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(lab)
    if not stats.HasLabel(1):
        raise ValueError("Empty heart after cleanup.")
    x,y,z,sx,sy,sz = stats.GetBoundingBox(1)
    sp = np.array(mask.GetSpacing(), float)
    lengths = sp * np.array([max(sx-1,1), max(sy-1,1), max(sz-1,1)], float)
    H = 0.5*lengths
    c = np.array(stats.GetCentroid(1), float)
    return H, c

# ----------------------------- Main pipeline ----------------------------------
def process_case_centroid_FOV(case_path: Path,
							  out_size=(64, 64, 64),
							  percentile=99.9,
							  margin=0.010,
							  pad_vox=1,
							  growth=1.08,
							  max_tries=2,
							  outdir_name="centered64") -> Dict:
	"""
	Minimal-resample, tight-fill pipeline:
	  1) Load CT (float32) and align TS masks once (NN+binarise).
	  2) Compute robust half-extents + centroid (mm).
	  3) Choose **isotropic** spacing so heart fills volume with margin & pad.
	  4) Build output grid centered on heart; resample CT (linear+anti-alias if downsampling),
		 and the heart mask (NN). If it still touches a border, grow spacing and retry.
	  5) Resample LV/MYO only once into the final grid; HU clip to cardiac window.
	"""
	case_path = Path(case_path)
	ct_path = case_path / "fullCT.nii.gz"
	seg_dir = case_path / "segments"
	out_dir = case_path / outdir_name
	out_dir.mkdir(parents=True, exist_ok=True)

	ct = load_ct(ct_path)
	ts = load_totalseg_as_dict(seg_dir)
	ts_ct = align_and_binarise_to_ct(ts, ct)

	# heart, lv, myo on CT grid (aligned)
	lv_ct, myo_ct, heart_ct = build_heart_lv_myo_strict(ct, ts_ct)
	mask_sanity(heart_ct)
	# robust extents in physical space
	#H_mm, c_mm = robust_extents_mm(heart_ct, percentile=percentile)
	H_mm, c_mm = half_extents_from_bbox(heart_ct)
	# choose spacing to fill the target grid
	base_spacing = spacing_for_fill(H_mm, out_size=out_size, margin=margin, pad_vox=pad_vox, isotropic=True)

	# adaptive retry if the heart still touches borders (rare, but safe)
	spacing = np.array(base_spacing, float)
	final_ref = None
	heart_final = None
	for attempt in range(1, int(max_tries) + 2):
		ref = build_output_ref(ct, c_mm, out_size=out_size, out_spacing=tuple(spacing.tolist()))

		# CT with anti-alias *iff* downsampling
		ct_pre = smooth_if_downsampling(ct, ref.GetSpacing())
		ct_out = sitk.Resample(ct_pre, ref, sitk.Transform(), sitk.sitkLinear, AIR_HU)

		# heart into ref (NN)
		heart_out = resample_like(heart_ct, ref, sitk.sitkNearestNeighbor, 0)

		if not touches_border(heart_out, pad_vox=pad_vox):
			final_ref = ct_out  # geometry holder
			heart_final = heart_out
			log(f"attempt {attempt} PASS | spacing={ref.GetSpacing()}, origin={ref.GetOrigin()}")
			break

		log(f"attempt {attempt} FAIL (touches border) | spacing={ref.GetSpacing()}")
		spacing *= float(growth)

	if final_ref is None:
		# last attempt geometry
		final_ref = ct_out
		heart_final = heart_out
		log("Exhausted retries; using last geometry.")

	# Resample LV/MYO once into the *final* grid (avoid multiple passes)
	lv_final  = resample_like(lv_ct,  final_ref, sitk.sitkNearestNeighbor, 0)
	myo_final = resample_like(myo_ct, final_ref, sitk.sitkNearestNeighbor, 0)

	# Clip HU after final resample (cardiac window)
	ct_clipped = sitk.Clamp(final_ref, lowerBound=-1000.0, upperBound=1200.0)

	# Save
	p_ct   = out_dir / "ct_centered.nii.gz"
	p_heart = out_dir / "heartmask_centered.nii.gz"
	p_lv    = out_dir / "lv_mask.nii.gz"
	p_myo   = out_dir / "myo_mask.nii.gz"

	sitk.WriteImage(ct_clipped, str(p_ct), useCompression=True)
	sitk.WriteImage(heart_final, str(p_heart), useCompression=True)
	sitk.WriteImage(lv_final, str(p_lv), useCompression=True)
	sitk.WriteImage(myo_final, str(p_myo), useCompression=True)

	meta = {
		"out_size": list(map(int, out_size)),
		"spacing_final": list(map(float, ct_clipped.GetSpacing())),
		"origin_final_mm": list(map(float, ct_clipped.GetOrigin())),
		"direction_final": list(map(float, ct_clipped.GetDirection())),
		"centroid_mm": list(map(float, c_mm)),
		"percentile": float(percentile),
		"margin": float(margin),
		"pad_vox": int(pad_vox),
		"growth": float(growth),
		"max_tries": int(max_tries),
		"inputs": {
			"ct_path": str(ct_path),
			"segments_dir": str(seg_dir),
			"used_structures": sorted(list(ts.keys())),
		},
		"notes": "Global z-score normalisation should use train-set μ,σ after clipping.",
	}
	with open(out_dir / "centered_meta.json", "w") as f:
		json.dump(meta, f, indent=2)

	return {
		"ct_out_path": p_ct,
		"heart_mask_path": p_heart,
		"lv_mask_path": p_lv,
		"myo_mask_path": p_myo,
		"meta_path": out_dir / "centered_meta.json",
		"spacing_final": ct_clipped.GetSpacing(),
		"origin_final": ct_clipped.GetOrigin(),
		"centroid_mm": tuple(c_mm),
	}
