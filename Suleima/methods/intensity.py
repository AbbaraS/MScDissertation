import nibabel as nib
import numpy as np
import os
import pandas as pd
from core.Log import log, loginfo
from core.Case import *
from core.globals import *
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

'''
	Clip using percentiles (e.g., p0.5–p99.5) per volume to reduce scan-to-scan variability.
	Clip to [-1000, 1000] HU for soft tissue-focused tasks.
	A safe choice might be clipping to [−1000, 2500] HU — this keeps air, soft tissue, and high-density contrast while removing extreme spikes.

	normalise for CNN:
	a. Global Normalisation (recommended for your dataset)
		1. Clip to fixed HU range (e.g., [-1000, 2500]).
		2. Apply: normalised = HU-𝜇 / 𝜎  [what was this formula called?]
			where μ and σ are global mean and std computed across the whole dataset.
		- Pros: Keeps intensities comparable across cases.
		- Cons: If acquisition protocols differ drastically, some contrast differences may remain.

	b. Per-Volume Z-Score Normalisation
		1. Clip per scan.
		2. For each scan: normalised = HU-𝜇_scan / 𝜎_scan
		- Pros: Each scan gets the same scale regardless of acquisition.
		- Cons: Removes absolute HU meaning (contrast vs non-contrast differences vanish).
		## do i care about keeping or removing contrast vs non-contrast differences?

	c. Min–Max Scaling
		1. Clip.
		2. Scale to [0, 1] or [−1, 1]: scaled=HU−min / max−min
		- Pros: Easy to interpret, stable for training.
		- Cons: Same issue as above — removes HU comparability.
'''


def intensity_stats(img: NiftiVolume):

	if img is None:
		log("No vol for intensity statistics.")
		return
	data = img.data
	#log(f"resampled intensity stats:  {{\nmin: {data.min()}, \nmax: {data.max()}, \nmean: {data.mean()}, \nstd: {data.std()}, \np0.5: {np.percentile(data, 0.5)}, \np99.5: {np.percentile(data, 99.5)}\n	 }}")
	return {
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "p0.5": np.percentile(data, 0.5),
        "p99.5": np.percentile(data, 99.5)
    }

def plot_intensity_distribution(df):
	Q1, Q3 = df["p99.5"].quantile([0.25, 0.75])
	IQR = Q3 - Q1
	upper_limit = Q3 + 1.5 * IQR
	lower_limit = Q1 - 1.5 * IQR
	outliers = df[(df["p99.5"] > upper_limit) | (df["p0.5"] < lower_limit)]
	#loginfo(outliers, "data/intensity_outliers.csv")
	plt.figure(figsize=(10,40))
	for _, row in df.iterrows():
		plt.plot([row["p0.5"], row["p99.5"]], [row["caseID"], row["caseID"]], 'o-', alpha=0.7)
	plt.xlabel("Intensity (HU)")
	plt.ylabel("Case ID")
	plt.title("CT Intensity Ranges (0.5–99.5 percentile)")
	plt.show()

def find_outliers(df):
	median_range = np.median(df["p99.5"] - df["p0.5"])
	mad_range = median_abs_deviation(df["p99.5"] - df["p0.5"])
	threshold = median_range + 3 * mad_range
	med_p99 = np.median(df["p99.5"])
	med_p0 = np.median(df["p0.5"])
	df["is_outlier"] = (
		(df["p0.5"] < med_p0 - 500) |
		(df["p99.5"] > med_p99 + 500)
	)
	return df[df["is_outlier"]].reset_index(drop=True)

def detect_outliers(df):
	Q1, Q3 = df["p99.5"].quantile([0.25, 0.75])
	IQR = Q3 - Q1
	upper_limit = Q3 + 1.5 * IQR
	lower_limit = Q1 - 1.5 * IQR
	return df[(df["p99.5"] > upper_limit) | (df["p0.5"] < lower_limit)]
