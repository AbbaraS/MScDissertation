

'''


def case_info():

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








'''