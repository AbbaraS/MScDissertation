
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from ipywidgets import interact


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

def view_slice_in_3d(image_tensor, mask_tensor, view_axis=2):
	"""
	Displays an interactive, scrollable plot for a 3D image and its mask.

	Args:
		image_tensor (torch.Tensor): The processed 3D image tensor (C, H, W, D).
		mask_tensor (torch.Tensor): The processed 3D multi-label mask (C, H, W, D).
									Assumes labels: 0=BG, 1=Other, 2=Myo, 3=LV.
	"""
	# --- 1. Data Preparation ---
	# Ensure tensors are on CPU and convert to NumPy arrays
	image_np = image_tensor.cpu().numpy()
	mask_np = mask_tensor.cpu().numpy()

	# Remove the channel dimension (C) -> (H, W, D)
	if image_np.ndim == 4:
		image_np = image_np.squeeze(0)
	if mask_np.ndim == 4:
		mask_np = mask_np.squeeze(0)

	# For display, let's view along the last axis (axial view in RAS orientation)
	# You can change this to axis 0 or 1 for sagittal or coronal views

	num_slices = image_np.shape[view_axis]

	# --- 2. Create Custom Colormap for Mask ---
	# Define colors: 0=Transparent, 1=Yellow(semi-transparent), 2=Blue, 3=Red
	# RGBA format (Red, Green, Blue, Alpha)
	colors = [
		(0, 0, 0, 0),       # 0: Background (fully transparent)
		(1, 1, 0, 0.3),     # 1: Other heart structures (semi-transparent yellow)
		(0, 0, 1, 0.5),     # 2: Myocardium (semi-transparent blue)
		(1, 0, 0, 0.5)      # 3: Left Ventricle (semi-transparent red)
	]
	custom_cmap = ListedColormap(colors)
	legend_patches = [
		Patch(color=colors[1], label='Heart'),
		Patch(color=colors[2], label='Myocardium'),
		Patch(color=colors[3], label='LV'),
	]
	# --- 3. Setup the Plot ---
	# Start at the middle slice
	initial_slice_idx = num_slices // 2

	fig, ax = plt.subplots(1, 1, figsize=(8, 8))
	plt.subplots_adjust(bottom=0.15) # Make space for the slider

	# Display the initial CT slice in grayscale
	ct_display = ax.imshow(
		np.take(image_np, initial_slice_idx, axis=view_axis).T,
		cmap='gray',
		origin='lower'
	)

	# Overlay the mask with our custom colormap and transparency
	mask_display = ax.imshow(
		np.take(mask_np, initial_slice_idx, axis=view_axis).T,
		cmap=custom_cmap,
		origin='lower',
		vmin=0,
		vmax=len(colors)-1
	)

	ax.set_title(f'Axial Slice: {initial_slice_idx}/{num_slices-1}')
	ax.axis('off') # Hide axes ticks

	# --- 4. Create the Slider ---
	slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03]) # [left, bottom, width, height]
	slice_slider = Slider(
		ax=slider_ax,
		label='Slice',
		valmin=0,
		valmax=num_slices - 1,
		valinit=initial_slice_idx,
		valstep=1
	)

	# --- 5. Update Function for the Slider ---
	def update(val):
		slice_idx = int(val)

		# Update the data for both the CT and the mask plots
		ct_display.set_data(np.take(image_np, slice_idx, axis=view_axis).T)
		mask_display.set_data(np.take(mask_np, slice_idx, axis=view_axis).T)

		ax.set_title(f'Axial Slice: {slice_idx}/{num_slices-1}')
		fig.canvas.draw_idle() # Redraw the plot

	# Register the update function with the slider
	slice_slider.on_changed(update)

	plt.show()






def scrollable_ct_mask(image_tensor, mask_tensor, view_axis=2):
	"""
	Displays a scrollable CT image with an overlay of the heart chamber mask.

	Parameters:
	- ct_path (str): Path to the cropped CT NIfTI file.
	- mask_path (str): Path to the combined cropped mask NIfTI file.
	"""
	# Load the CT and mask images
	image_np = image_tensor.cpu().numpy()
	mask_np = mask_tensor.cpu().numpy()

	# Remove the channel dimension (C) -> (H, W, D)
	if image_np.ndim == 4:
		image_np = image_np.squeeze(0)
	if mask_np.ndim == 4:
		mask_np = mask_np.squeeze(0)

	# For display, let's view along the last axis (axial view in RAS orientation)
	# You can change this to axis 0 or 1 for sagittal or coronal views



	# === Define colormap and legend ===

	colors = [
		(0, 0, 0, 0),       # 0: Background (fully transparent)
		(1, 1, 0, 0.8),     # 1: Other heart structures (semi-transparent yellow)
		(0, 0, 1, 0.5),     # 2: Myocardium (semi-transparent blue)
		(1, 0, 0, 0.5)      # 3: Left Ventricle (semi-transparent red)
	]
	custom_cmap = ListedColormap(colors)
	legend_patches = [
		Patch(color=colors[1], label='Heart'),
		Patch(color=colors[2], label='Myocardium'),
		Patch(color=colors[3], label='LV'),
	]
	# === Axial 'Z' view ===
	def display_axial(slice_index):
		plt.figure(figsize=(6, 6))
		plt.imshow(image_np[:, :, slice_index], cmap='gray', origin='lower')
		plt.imshow(mask_np[:, :, slice_index], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		plt.axis("off")
		plt.title(f"Slice {slice_index + 1} / {image_np.shape[2]}")
		plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
		plt.show()
	def display_sagittal(slice_index):
		plt.figure(figsize=(6, 6))
		plt.imshow(image_np[slice_index,:, :], cmap='gray', origin='lower')
		plt.imshow(mask_np[slice_index,:, :], cmap=custom_cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
		plt.axis("off")
		plt.title(f"Slice {slice_index + 1} / {image_np.shape[0]}")
		plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
		plt.show()
	def display_coronal(slice_index):
		plt.figure(figsize=(6, 6))
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