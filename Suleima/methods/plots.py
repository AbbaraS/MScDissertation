
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from ipywidgets import interact


def scrollable_ct_mask(ct, mask):
	"""
	Displays a scrollable CT image with an overlay of the heart chamber mask.

	Parameters:
	- ct_path (str): Path to the cropped CT NIfTI file.
	- mask_path (str): Path to the combined cropped mask NIfTI file.
	"""
	# Load the CT and mask images
	ct_img = ct.data
	mask_img = mask.data

	if ct is None or mask is None:
		print("Error: CT or mask not loaded. Run `crop_pipeline()` first.")
		return

	# === Define colormap and legend ===
	colors = ['black', 'red', 'blue', 'green', 'yellow', 'magenta']  # label 0 to 5
	cmap = ListedColormap(colors)
	legend_patches = [
		Patch(color=colors[1], label='LV'),
		Patch(color=colors[2], label='RV'),
		Patch(color=colors[3], label='LA'),
		Patch(color=colors[4], label='RA'),
		Patch(color=colors[5], label='Myocardium')
	]
	# === Axial 'Z' view ===
	def display_slice(slice_index):
		plt.figure(figsize=(6, 6))
		plt.imshow(ct_img[:, :, slice_index], cmap='gray', origin='lower')
		plt.imshow(mask_img[:, :, slice_index], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		plt.axis("off")
		plt.title(f"Slice {slice_index + 1} / {ct_img.shape[2]}")
		plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
		plt.show()

	interact(display_slice, slice_index=(0, ct_img.shape[2] - 1))




def scrollable_ct_mask_compare(case1, case2):
	"""
	Displays two scrollable CT+mask images side by side for comparison.

	Parameters:
	- ct1, mask1: First case's CT and mask NiftiVolume objects
	- caseID1: ID for the first case
	- ct2, mask2: Second case's CT and mask NiftiVolume objects
	- caseID2: ID for the second case
	"""
	caseID1, ct_img1, mask_img1 = case1.caseID, case1.resampledCT.data, case1.LVresampled.data
	caseID2, ct_img2, mask_img2 = case2.caseID, case2.resampledCT.data, case2.LVresampled.data

	# Define colormap and legend
	colors = ['black', 'red', 'blue', 'green', 'yellow', 'magenta']  # label 0 to 5
	cmap = ListedColormap(colors)
	legend_patches = [
		Patch(color=colors[1], label='LV'),
		Patch(color=colors[2], label='RV'),
		Patch(color=colors[3], label='LA'),
		Patch(color=colors[4], label='RA'),
		Patch(color=colors[5], label='Myocardium')
	]

	def display_slice(slice_index):
		fig, axes = plt.subplots(1, 2, figsize=(12, 6))

		# First case
		axes[0].imshow(ct_img1[:, :, slice_index], cmap='gray', origin='lower')
		axes[0].imshow(mask_img1[:, :, slice_index], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		axes[0].set_title(f"{caseID1}\nSlice {slice_index+1}/{ct_img1.shape[2]}")
		axes[0].axis("off")

		# Second case
		axes[1].imshow(ct_img2[:, :, slice_index], cmap='gray', origin='lower')
		axes[1].imshow(mask_img2[:, :, slice_index], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		axes[1].set_title(f"{caseID2}\nSlice {slice_index+1}/{ct_img2.shape[2]}")
		axes[1].axis("off")

		# Legend below both plots
		fig.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
		plt.show()

	interact(display_slice, slice_index=(0, min(ct_img1.shape[2], ct_img2.shape[2]) - 1))





