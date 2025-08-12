import logging
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from ipywidgets import interact
import ipywidgets as widgets
from ipywidgets import IntSlider, HBox, VBox, Output
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
logger = logging.getLogger(__name__)

colors = [
		(0, 0, 0, 0),       # 0: Background (fully transparent)
		(1, 1, 0, 0.3),     # 1: Other heart structures (semi-transparent yellow)
		(0, 0, 1, 0.3),     # 2: Myocardium (semi-transparent blue)
		(1, 0, 0, 0.3)      # 3: Left Ventricle (semi-transparent red)
	]
custom_cmap = ListedColormap(colors)
legend_patches = [
		mpatches.Patch(color=colors[1], label='Heart'),
		mpatches.Patch(color=colors[2], label='Myocardium'),
		mpatches.Patch(color=colors[3], label='LV'),
	]

def display_selected_slices(volumes, slices):
	plt.close('all')
	mask_np = volumes["mask"].cpu().numpy().squeeze(0)
	img_np = volumes["image"].cpu().numpy().squeeze(0)

	axial_indices = slices["Axial"]
	coronal_indices = slices["Coronal"]
	sagittal_indices = slices["Sagittal"]

	fig, axes = plt.subplots(3, 3, figsize=(8, 8))
	fig.suptitle("Selected Slices for Diagnosis", fontsize=6)
	# --- Plot Axial Slices ---
	for i, slice_idx in enumerate(axial_indices):
		ax = axes[0, i]
		ax.imshow(np.take(img_np, slice_idx, axis=2).T, cmap='gray', origin='lower')
		ax.imshow(np.take(mask_np, slice_idx, axis=2).T, cmap=custom_cmap, origin='lower', vmin=0, vmax=len(colors)-1)
		ax.set_title(f'Axial (Short-Axis) Slice: {slice_idx}', fontsize=6)
		ax.axis('off')
	# --- Plot Coronal Slices ---
	for i, slice_idx in enumerate(coronal_indices):
		ax = axes[1, i]
		ax.imshow(np.take(img_np, slice_idx, axis=1).T, cmap='gray', origin='lower')
		ax.imshow(np.take(mask_np, slice_idx, axis=1).T, cmap=custom_cmap, origin='lower', vmin=0, vmax=len(colors)-1)
		ax.set_title(f'Coronal (4-Chamber View) Slice: {slice_idx}', fontsize=6)
		ax.axis('off')

	# --- Plot Sagittal Slices ---
	for i, slice_idx in enumerate(sagittal_indices):
		ax = axes[2, i]
		ax.imshow(np.take(img_np, slice_idx, axis=0).T, cmap='gray', origin='lower')
		ax.imshow(np.take(mask_np, slice_idx, axis=0).T, cmap=custom_cmap, origin='lower', vmin=0, vmax=len(colors)-1)
		ax.set_title(f'Sagittal (2-Chamber View) Slice: {slice_idx}', fontsize=6)
		ax.axis('off')

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()



def scrollable_ct_mask(image_tensor, mask_tensor, view_axis=2):
	plt.close('all')
	image_np = image_tensor.cpu().numpy()
	mask_np = mask_tensor.cpu().numpy()

	if image_np.ndim == 4: image_np = image_np.squeeze(0)
	if mask_np.ndim == 4: mask_np = mask_np.squeeze(0)

	def display_axial(slice_index):
		plt.figure(figsize=(4, 4))
		plt.imshow(image_np[:, :, slice_index], cmap='gray', origin='lower')
		plt.imshow(mask_np[:, :, slice_index], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		plt.axis("off")
		plt.title(f"Slice {slice_index + 1} / {image_np.shape[2]}")
		plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
		plt.show()
	def display_sagittal(slice_index):
		plt.figure(figsize=(4, 4))
		plt.imshow(image_np[slice_index,:, :], cmap='gray', origin='lower')
		plt.imshow(mask_np[slice_index,:, :], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		plt.axis("off")
		plt.title(f"Slice {slice_index + 1} / {image_np.shape[0]}")
		plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
		plt.show()
	def display_coronal(slice_index):
		plt.figure(figsize=(4, 4))
		plt.imshow(image_np[:, slice_index, :], cmap='gray', origin='lower')
		plt.imshow(mask_np[:, slice_index, :], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		plt.axis("off")
		plt.title(f"Slice {slice_index + 1} / {image_np.shape[1]}")
		plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
		plt.show()
	if view_axis == 2: interact(display_axial, slice_index=(0, image_np.shape[2] - 1))
	elif view_axis == 0: interact(display_sagittal, slice_index=(0, image_np.shape[0] - 1))
	elif view_axis == 1: interact(display_coronal, slice_index=(0, image_np.shape[1] - 1))
	else:
		print("Invalid view axis. Choose 0 (sagittal), 1 (coronal), or 2 (axial).")


def scrollable_planes(image_tensor, mask_tensor):
	plt.close('all')
	img = image_tensor.detach().cpu().numpy()
	msk = mask_tensor.detach().cpu().numpy()
	if img.ndim == 4: img = img.squeeze(0)
	if msk.ndim == 4: msk = msk.squeeze(0)
	assert img.shape == msk.shape and img.ndim == 3, "Expect matching 3D volumes"
	X, Y, Z = img.shape
	#vmax = int(np.max(msk)) if np.max(msk) > 0 else 1
	vmax = min(3, int(msk.max()))

	with plt.ioff(): fig, axes = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
	for ax in axes: ax.axis("off")

	i_sag, i_cor, i_ax = X//2, Y//2, Z//2

	sag_im = axes[0].imshow(img[i_sag, :, :], cmap='gray', origin='lower')
	sag_ov = axes[0].imshow(msk[i_sag, :, :], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=vmax)
	axes[0].set_title(f"Sagittal  {i_sag+1}/{X}")

	cor_im = axes[1].imshow(img[:, i_cor, :], cmap='gray', origin='lower')
	cor_ov = axes[1].imshow(msk[:, i_cor, :], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=vmax)
	axes[1].set_title(f"Coronal   {i_cor+1}/{Y}")
	axes[1].legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))

	ax_im = axes[2].imshow(img[:, :, i_ax], cmap='gray', origin='lower')
	ax_ov = axes[2].imshow(msk[:, :, i_ax], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=vmax)
	axes[2].set_title(f"Axial     {i_ax+1}/{Z}")

	s_sag = IntSlider(description='Sagittal', min=0, max=X-1, value=i_sag, continuous_update=False)
	s_cor = IntSlider(description='Coronal',  min=0, max=Y-1, value=i_cor, continuous_update=False)
	s_axl = IntSlider(description='Axial',    min=0, max=Z-1, value=i_ax,  continuous_update=False)
	live = "widget" in matplotlib.get_backend().lower() or "nbagg" in matplotlib.get_backend().lower()
	out = None
	#with out: display(fig)
	if live :
		def redraw(): fig.canvas.draw_idle()
		#fig_widget = fig.canvas
	else:
		out = Output()
		with out:
			display(fig)
		def redraw():
			with out:
				out.clear_output(wait=True)
				display(fig)
		fig_widget = out

	def on_sag(change):
		i = change["new"]
		sag_im.set_data(img[i, :, :]);	sag_ov.set_data(msk[i, :, :])
		axes[0].set_title(f"Sagittal  {i+1}/{X}")
		redraw()
	def on_cor(change):
		i = change["new"]
		cor_im.set_data(img[:, i, :]);	cor_ov.set_data(msk[:, i, :])
		axes[1].set_title(f"Coronal   {i+1}/{Y}")
		redraw()
	def on_ax(change):
		i = change["new"]
		ax_im.set_data(img[:, :, i]);	ax_ov.set_data(msk[:, :, i])
		axes[2].set_title(f"Axial     {i+1}/{Z}")
		redraw()
	s_sag.observe(on_sag, names="value")
	s_cor.observe(on_cor, names="value")
	s_axl.observe(on_ax,  names="value")

	return VBox([fig.canvas, HBox([s_sag, s_cor, s_axl])])


def scrollable_ct_mask_compare(view, caseID1, ct_img1, mask_img1, caseID2, ct_img2, mask_img2):
	"""
	Displays two scrollable CT+mask images side by side for comparison.

	Parameters:
	- ct1, mask1: First case's CT and mask NiftiVolume objects
	- caseID1: ID for the first case
	- ct2, mask2: Second case's CT and mask NiftiVolume objects
	- caseID2: ID for the second case
	"""
	#caseID1, ct_img1, mask_img1 = case1.caseID, case1.croppedCT.data, case1.LVcropped.data
	#caseID2, ct_img2, mask_img2 = case2.caseID, case2.croppedCT.data, case2.LVcropped.data

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

	def display_axial(slice_index):

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



	def display_sagittal(slice_index):
		fig, axes = plt.subplots(1, 2, figsize=(12, 6))

		# First case
		axes[0].imshow(ct_img1[slice_index, :, :], cmap='gray', origin='lower')
		axes[0].imshow(mask_img1[slice_index, :, :], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		axes[0].set_title(f"{caseID1}\nSlice {slice_index+1}/{ct_img1.shape[0]}")
		axes[0].axis("off")

		# Second case
		axes[1].imshow(ct_img2[slice_index, :, :], cmap='gray', origin='lower')
		axes[1].imshow(mask_img2[slice_index, :, :], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		axes[1].set_title(f"{caseID2}\nSlice {slice_index+1}/{ct_img2.shape[0]}")
		axes[1].axis("off")

		# Legend below both plots
		fig.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
		plt.show()


	def display_coronal(slice_index):
		fig, axes = plt.subplots(1, 2, figsize=(12, 6))

		# First case
		axes[0].imshow(ct_img1[:, slice_index, :], cmap='gray', origin='lower')
		axes[0].imshow(mask_img1[:, slice_index, :], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		axes[0].set_title(f"{caseID1}\nSlice {slice_index+1}/{ct_img1.shape[1]}")
		axes[0].axis("off")

		# Second case
		axes[1].imshow(ct_img2[:, slice_index, :], cmap='gray', origin='lower')
		axes[1].imshow(mask_img2[:, slice_index, :], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		axes[1].set_title(f"{caseID2}\nSlice {slice_index+1}/{ct_img2.shape[1]}")
		axes[1].axis("off")

		# Legend below both plots
		fig.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
		plt.show()


	if view == "X": interact(display_sagittal, slice_index=(0, min(ct_img1.shape[0], ct_img2.shape[0]) - 1))
	elif view == "Y": interact(display_coronal, slice_index=(0, min(ct_img1.shape[1], ct_img2.shape[1]) - 1))
	elif view == "Z": interact(display_axial, slice_index=(0, min(ct_img1.shape[2], ct_img2.shape[2]) - 1))
	else:
		print("Invalid view. Choose 'X', 'Y', or 'Z'.")



import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from ipywidgets import interact, IntSlider










def scrollable_ct_mask1(ct_img: sitk.Image, mask_img: sitk.Image = None):
	"""
	Displays a scrollable CT image with an optional mask overlay.

	Parameters:
	- ct_img (sitk.Image): The SimpleITK CT image.
	- mask_img (sitk.Image): The SimpleITK mask image (optional).
	"""
	# 1. Convert sitk.Image to a NumPy array for plotting
	# The array will have shape (depth, height, width)
	ct_array = sitk.GetArrayFromImage(ct_img)

	# Do the same for the mask if it exists
	mask_array = None
	if mask_img:
		mask_array = sitk.GetArrayFromImage(mask_img)

	# === Define colormap and legend ===
	colors = ['transparent', 'red', 'blue', 'green', 'yellow', 'magenta'] # label 0 is now transparent
	cmap = ListedColormap(colors)
	legend_patches = [
		Patch(color=colors[1], label='LV'),
		Patch(color=colors[2], label='RV'),
		Patch(color=colors[3], label='LA'),
		Patch(color=colors[4], label='RA'),
		Patch(color=colors[5], label='Myocardium')
	]

	# 2. Get the number of slices from the NumPy array's shape
	num_slices = ct_array.shape[0]

	# === Define the plotting function for interact ===
	def display_slice(slice_index):
		fig, ax = plt.subplots(figsize=(7, 7))

		# 3. Slice the NumPy array, not the sitk.Image
		# Slicing is on the first axis (z-axis)
		ax.imshow(ct_array[slice_index, :, :], cmap='gray', origin='lower')

		# Overlay the mask if it's available
		if mask_array is not None:
			ax.imshow(mask_array[slice_index, :, :], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)

		ax.axis("off")
		ax.set_title(f"Slice {slice_index + 1} / {num_slices}")

		# Place legend outside the plot
		fig.legend(handles=legend_patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))
		plt.tight_layout(rect=[0, 0.08, 1, 1]) # Adjust layout to make space for legend
		plt.show()

	# Use a widget for a better user experience
	slice_slider = IntSlider(min=0, max=num_slices - 1, step=1, value=num_slices // 2, description='Slice:')
	interact(display_slice, slice_index=slice_slider)

# Example call to the function
#scrollable_ct_mask(ct_img, mask_img)