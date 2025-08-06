

from totalsegmentator.python_api import totalsegmentator

import nibabel as nib
from nibabel.orientations import aff2axcodes
#from nibabel import Nifti1Image

import dicom2nifti
import dicom2nifti.convert_dicom

import SimpleITK as sitk
import scipy.ndimage
from skimage import measure
import imageio

import os
import pandas as pd
import numpy as np
import csv


import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from ipywidgets import interact

'''
labels = { "patientID": {"label": 0/1}, ...}
slices = {"patientID":{
            "Axial":
                {"ct": [{"idx":"…"," slice": []},
                        {"idx":"…"," slice": []},		
                        {"idx":"…"," slice": []}
                        ],		
                "mask": [{"idx":"…"," slice": []},
                        {"idx":"…"," slice": []},		
                        {"idx":"…"," slice": []}
                        ]},
            "Coronal":
                {"ct": [{"idx":"…"," slice": []},
                        {"idx":"…"," slice": []},		
                        {"idx":"…"," slice": []}
                        ],		
                "mask": [{"idx":"…"," slice": []},
                        {"idx":"…"," slice": []},		
                        {"idx":"…"," slice": []}
                        ]},
            "Sagittal":
                {"ct": [{"idx":"…"," slice": []},
                        {"idx":"…"," slice": []},		
                        {"idx":"…"," slice": []}
                        ],		
                "mask": [{"idx":"…"," slice": []},
                        {"idx":"…"," slice": []},		
                        {"idx":"…"," slice": []}
                        ]}
            }, ...}	
metadata = {"patientID": {"age": 0, "gender": "M/F"},...}


'''

# === Plots ===
                
def plot_slices(ct_cropped, combined_mask, x_slices, y_slices, z_slices):
    # === Define colormap ===
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'magenta']  # label 0 to 5
    cmap = ListedColormap(colors)
    legend_patches = [
        Patch(color=colors[1], label='LV'),
        Patch(color=colors[2], label='RV'),
        Patch(color=colors[3], label='LA'),
        Patch(color=colors[4], label='RA'),
        Patch(color=colors[5], label='Myocardium')
    ]

    # === Helper to extract slices ===
    def extract_slice(image, mask, axis, index):
        if axis == 'x':
            img = image[index, :, :]
            msk = mask[index, :, :]
        elif axis == 'y':
            img = image[:, index, :]
            msk = mask[:, index, :]
        elif axis == 'z':
            img = image[:, :, index]
            msk = mask[:, :, index]
        return np.rot90(img), np.rot90(msk)

    # === Plot all slices in a 3×3 grid ===
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axes = {'x': x_slices, 'y': y_slices, 'z': z_slices}
    axis_labels = {'x': 'Sagittal (X)', 'y': 'Coronal (Y)', 'z': 'Axial (Z)'}

    for row, axis in enumerate(['x', 'y', 'z']):
        for col, idx in enumerate(axes[axis]):
            img, msk = extract_slice(ct_cropped, combined_mask, axis, idx)
            axs[row, col].imshow(img, cmap='gray', origin='lower')
            axs[row, col].imshow(msk, cmap=cmap, alpha=0.4, origin='lower', interpolation='none', vmin=0, vmax=5)
            axs[row, col].set_title(f"{axis_labels[axis]} Slice {idx}")
            axs[row, col].axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize='large', bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def plot_slices_from_png(png_slices_folder):
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

    # === Collect CT slice indices from filenames ===
    pattern = re.compile(r"(ct|mask)([XYZ])_(\d+)\.png", re.IGNORECASE)
    slice_dict = {'x': [], 'y': [], 'z': []}
    
    for fname in os.listdir(png_slices_folder):
        match = pattern.match(fname)
        if match:
            kind = match.group(1).lower()  # 'slice' or 'mask' (if needed later)
            axis = match.group(2).lower()  # 'x', 'y', or 'z'
            idx = int(match.group(3))
            slice_dict[axis].append(idx)

    axes = {}
    for axis in ['x', 'y', 'z']:
        axes[axis] = sorted(set(slice_dict[axis]))

    axis_labels = {'x': 'Sagittal (X)', 'y': 'Coronal (Y)', 'z': 'Axial (Z)'}
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    for row, axis in enumerate(['x', 'y', 'z']):
        axis_label = axis.upper()
        for col, idx in enumerate(axes[axis]):
            ct_path = os.path.join(png_slices_folder, f"ct{axis_label}_{idx}.png")
            mask_path = os.path.join(png_slices_folder, f"mask{axis_label}_{idx}.png")

            if not os.path.exists(ct_path):
                axs[row, col].set_title(f"{axis_labels[axis]} Slice {idx} (missing)")
                axs[row, col].axis('off')
                continue

            img = imageio.imread(ct_path)
            axs[row, col].imshow(img, cmap='gray')

            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path)
                axs[row, col].imshow(mask, cmap=cmap, alpha=0.4, interpolation='none', vmin=0, vmax=5)

            axs[row, col].set_title(f"{axis_labels[axis]} Slice {idx}")
            axs[row, col].axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize='large', bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    
def plot_slices_masks(png_slices_folder, ct_path, mask_path):
    # === Load CT and combined mask volumes ===
    ct_volume = nib.load(ct_path).get_fdata()
    mask_volume = nib.load(mask_path).get_fdata().astype(int)
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)
    print("CT shape:", ct_img.shape)
    print("Mask shape:", mask_img.shape)
    print("CT affine:\n", ct_img.affine)
    print("Mask affine:\n", mask_img.affine)
    # === Define colormap and legend for segmentation ===
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'magenta']  # label 0 to 5
    cmap = ListedColormap(colors)
    legend_patches = [
        Patch(color=colors[1], label='LV'),
        Patch(color=colors[2], label='RV'),
        Patch(color=colors[3], label='LA'),
        Patch(color=colors[4], label='RA'),
        Patch(color=colors[5], label='Myocardium')
    ]

    # === Collect slice indices from PNG filenames ===
    pattern = re.compile(r"slice([XYZ])_(\d+)\.png", re.IGNORECASE)
    slice_dict = {'x': [], 'y': [], 'z': []}
    
    for fname in os.listdir(png_slices_folder):
        match = pattern.match(fname)
        if match:
            axis = match.group(1).lower()
            idx = int(match.group(2))
            slice_dict[axis].append(idx)

    # === Select 3 slices per axis ===
    axes = {}
    for axis in ['x', 'y', 'z']:
        sorted_indices = sorted(slice_dict[axis])
        if len(sorted_indices) >= 3:
            mid = len(sorted_indices) // 2
            axes[axis] = [sorted_indices[mid - 1], sorted_indices[mid], sorted_indices[mid + 1]]
        else:
            axes[axis] = sorted_indices

    axis_labels = {'x': 'Sagittal (X)', 'y': 'Coronal (Y)', 'z': 'Axial (Z)'}
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    for row, axis in enumerate(['x', 'y', 'z']):
        for col, idx in enumerate(axes[axis]):
            # Extract CT and mask slices directly from the volume
            if axis == 'x':
                ct_slice = ct_volume[idx, :, :]
                mask_slice = mask_volume[idx, :, :]
            elif axis == 'y':
                ct_slice = ct_volume[:, idx, :]
                mask_slice = mask_volume[:, idx, :]
            elif axis == 'z':
                ct_slice = ct_volume[:, :, idx]
                mask_slice = mask_volume[:, :, idx]

            # Rotate both slices the same way
            ct_slice = np.rot90(ct_slice)
            mask_slice = np.rot90(mask_slice)

            axs[row, col].imshow(ct_slice, cmap='gray')
            axs[row, col].set_title(f"{axis_labels[axis]} Slice {idx}")
            axs[row, col].axis('off')

            # Draw slice location lines across views
            height, width = ct_slice.shape
            if axis == 'x':
                axs[row, col].axhline(y=ct_volume.shape[1] // 2, color='white', linestyle='--', linewidth=1)
                axs[row, col].axvline(x=ct_volume.shape[2] // 2, color='white', linestyle='--', linewidth=1)
            elif axis == 'y':
                axs[row, col].axhline(y=ct_volume.shape[0] // 2, color='white', linestyle='--', linewidth=1)
                axs[row, col].axvline(x=ct_volume.shape[2] // 2, color='white', linestyle='--', linewidth=1)
            elif axis == 'z':
                axs[row, col].axhline(y=ct_volume.shape[0] // 2, color='white', linestyle='--', linewidth=1)
                axs[row, col].axvline(x=ct_volume.shape[1] // 2, color='white', linestyle='--', linewidth=1)

            # Draw contour outlines for each label
            for label_val in range(1, 6):  # skip 0 (background)
                mask_binary = (mask_slice == label_val).astype(np.uint8)
                contours = measure.find_contours(mask_binary, 0.5)
                for contour in contours:
                    axs[row, col].plot(contour[:, 1], contour[:, 0], color=colors[label_val], linewidth=1)

    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize='large', bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def scrollable_ct_mask(ct_path, mask_path):
    """
    Displays a scrollable CT image with an overlay of the heart chamber mask.

    Parameters:
    - ct_path (str): Path to the cropped CT NIfTI file.
    - mask_path (str): Path to the combined cropped mask NIfTI file.
    """
    # Load the CT and mask images
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)
    ct_data = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

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

    def display_slice(slice_index):
        plt.figure(figsize=(6, 6))
        plt.imshow(ct_data[:, :, slice_index], cmap='gray', origin='lower')
        plt.imshow(mask_data[:, :, slice_index], cmap=cmap, alpha=0.4, origin='lower', vmin=0, vmax=5)
        plt.axis("off")
        plt.title(f"Slice {slice_index + 1} / {ct_data.shape[2]}")
        plt.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
        plt.show()

    interact(display_slice, slice_index=(0, ct_data.shape[2] - 1))    

# === Extras ===

def check_original_mask_alignment(ct_path, mask_path, world_mm, slice_axis='z', slice_index=None):
    # Load CT and mask volumes
    img = nib.load(ct_path)
    ct = img.get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    if world_mm:
        aff=img.affine
        spacing = img.header.get_zooms()
        origin = aff[:3, 3]
        axis=2
        spacing_mm = spacing[axis]
        origin_mm = origin[axis]
        slice_index = round((world_mm - origin_mm) / spacing_mm)
        
    

    # Choose a slice index if not provided
    if slice_index is None:
        slice_index = ct.shape['xyz'.index(slice_axis)] // 2

    # Extract slice and rotate if needed for correct view
    if slice_axis == 'x':
        ct_slice = ct[slice_index, :, :]
        mask_slice = mask[slice_index, :, :]
    elif slice_axis == 'y':
        ct_slice = ct[:, slice_index, :]
        mask_slice = mask[:, slice_index, :]
    elif slice_axis == 'z':
        ct_slice = ct[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]
    else:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")

    # Display CT with contour
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_slice.T, cmap='gray', origin='lower')
    plt.contour(mask_slice.T, levels=[0.5], colors='r')  # red contour for mask
    plt.title(f"Original mask overlay ({slice_axis.upper()} = {slice_index})")
    plt.axis('off')
    plt.show()    

def get_slices_idx(png_slices_folder):
    """
    Extracts slice indices from filenames of the form:
    - sliceX_42.png
    - maskY_57.png
    Returns a dictionary with axis keys ('x', 'y', 'z') and list of unique slice indices.
    """
    pattern = re.compile(r"(slice|mask)([XYZ])_(\d+)\.png", re.IGNORECASE)
    slice_dict = {'x': [], 'y': [], 'z': []}

    for fname in os.listdir(png_slices_folder):
        match = pattern.match(fname)
        if match:
            axis = match.group(2).lower()
            idx = int(match.group(3))
            if idx not in slice_dict[axis]:
                slice_dict[axis].append(idx)

    # sort each axis's list
    for axis in slice_dict:
        slice_dict[axis].sort()
    
    return slice_dict

def save_missing_filesCSV(case):
    # folder paths
    base_dicom_root = f"../Takotsubo-Syndrome/data/Inputs/{case}"  # {patient}/DICOM
    base_input_root = f"data/Inputs/{case}"  # /{patient}
    base_output_root = f"data/Outputs/{case}"  # /{patient}
    output_filename = f'missing_files_{case}.csv'
    missing_files_dic = {}

    seg_files = [
        "heart_ventricle_left.nii.gz",
        "heart_ventricle_right.nii.gz",
        "heart_atrium_left.nii.gz",
        "heart_atrium_right.nii.gz",
        "heart_myocardium.nii.gz",
        "ct_image.nii.gz"
    ]

    cropped_files = [
        "cropped_lv.nii.gz",
        "cropped_rv.nii.gz",
        "cropped_la.nii.gz",
        "cropped_ra.nii.gz",
        "cropped_myo.nii.gz",
        "cropped_ct.nii.gz",
        "cropped_mask.nii.gz"
    ]

    for patientID in os.listdir(base_dicom_root):
        missing = []

        input_folder = os.path.join(base_input_root, patientID)
        output_folder = os.path.join(base_output_root, patientID)

        try:
            # Check segmentation files
            for seg_file in seg_files:
                seg_path = os.path.join(input_folder, seg_file)
                if not os.path.exists(seg_path):
                    missing.append({'file': seg_file, 'path': seg_path})

            # Check cropped files
            for cropped_file in cropped_files:
                cropped_path = os.path.join(output_folder, cropped_file)
                if not os.path.exists(cropped_path):
                    missing.append({'file': cropped_file, 'path': cropped_path})

            if missing:
                missing_files_dic[patientID] = missing

        except Exception as e:
            print(f"❌ Error processing {patientID}: {e}")
            continue
    
    missing_files_dic = missing_files_dic
    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['Patient ID', 'Missing File', 'File Path'])

        # Write each patient ID and their missing files with paths
        for patient_id, files in missing_files_dic.items():
            for file_info in files:
                writer.writerow([patient_id, file_info['file'], file_info['path']])

    print(f"CSV file '{output_filename}' has been created successfully.") 
    
def resample_volume(volume_np, spacing, new_spacing=[1.0, 1.0, 1.0], is_label=False, reference_image=None):
    """
    Resamples a 3D volume to the given spacing, optionally matching a reference image for perfect alignment.
    Returns the resampled NumPy array and its affine.
    """
    # Convert to SimpleITK image (z, y, x) format expected by SimpleITK
    sitk_img = sitk.GetImageFromArray(np.transpose(volume_np, (2, 1, 0)))
    sitk_img.SetSpacing([float(s) for s in spacing[::-1]])

    if reference_image is not None:
        # Use reference image's properties to ensure perfect alignment
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        resampled = resampler.Execute(sitk_img)
    else:
        # Compute new size for desired spacing
        original_size = np.array(sitk_img.GetSize(), dtype=np.int32)
        original_spacing = np.array(sitk_img.GetSpacing())
        new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        resampled = resampler.Execute(sitk_img)

    # Recover affine (nibabel-style)
    direction = np.array(resampled.GetDirection()).reshape(3, 3)
    spacing_arr = np.array(resampled.GetSpacing())
    origin = np.array(resampled.GetOrigin())
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing_arr[:, None]
    affine[:3, 3] = origin

    # Convert back to NumPy in (x, y, z)
    resampled_np = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
    return resampled_np, affine, resampled  # Return SimpleITK image for reference use

def resample_shape(volume, target_shape = (64, 64, 64), is_label=False):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    order = 0 if is_label else 1  # nearest-neighbor for mask, linear for CT
    return scipy.ndimage.zoom(volume, zoom=zoom_factors, order=order)

# === In Use ===

class DataInfo:
    def __init__(self, patient_id, cropped=False, original_shape=None, new_shape=None, original_spacing=None, new_spacing=None, original_orientation=None, new_orientation=None, origin=None):
        self.patient_id = patient_id
        self.cropped = cropped
        self.original_shape = original_shape
        self.new_shape = new_shape
        self.original_spacing = original_spacing
        self.new_spacing = new_spacing
        self.original_orientation = original_orientation
        self.new_orientation = new_orientation
        self.origin = origin



def update_info_csv(case):
    metadata_path = f"data/CSVs/{case}_info.csv"
    output_dir = f"data/Outputs/{case}"	
    input_dir = f"data/Inputs/{case}"

    # Get all patient folder names
    patient_ids = sorted([
        folder for folder in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, folder))
    ])

    df = pd.DataFrame({
        "PatientID": patient_ids,
        "Cropped": False,
        "Original_Shape": np.nan,
        #"Cropped_Shape": np.nan,
        "New_Shape": np.nan,
        "Original_Spacing": np.nan,
        "New_Spacing": np.nan,
        "Original_Orientation": np.nan,
        "New_Orientation": np.nan,
        "Origin": np.nan,
        "selected_Z_slcs": "",
        "selected_X_slcs": "",
        "selected_Y_slcs": "",
    })

    for idx, patient_id in enumerate(patient_ids):
        try:
            patient_input = os.path.join(input_dir, patient_id)
            patient_output = os.path.join(output_dir, patient_id)

            orig_ct_path = os.path.join(patient_input, "ct_image.nii.gz")
            cropped_ct_path = os.path.join(patient_output, "cropped_ct.nii.gz")
            #mask_ct_path = os.path.join(patient_output, "cropped_mask.nii.gz")
            png_slice_dir = os.path.join(patient_output, "png_slices")

            # Load original CT
            if os.path.exists(orig_ct_path):
                orig_img = nib.load(orig_ct_path)
                df.at[idx, "Original_Shape"] = str(orig_img.shape)
                df.at[idx, "Original_Spacing"] = str(tuple(round(s, 3) for s in orig_img.header.get_zooms()))
                df.at[idx, "Original_Orientation"] = str(aff2axcodes(orig_img.affine))
                df.at[idx, "Origin"] = str(tuple(round(s, 3) for s in orig_img.affine[:3, 3]))
                
            #df.at[idx, "Cropped_Shape"] =
            
            # Load cropped CT
            if os.path.exists(cropped_ct_path):
                #cropped_img = nib.load(cropped_ct_path)
                df.at[idx, "Cropped"] = True
                df.at[idx, "New_Shape"] = "(64, 64, 64)"
                df.at[idx, "New_Spacing"] = "(1.0, 1.0, 1.0)"
                df.at[idx, "New_Orientation"] = "('R', 'A', 'S')"

            # Load slice indices
            if os.path.exists(png_slice_dir):
                slice_dict = get_slices_idx(png_slice_dir)
                df.at[idx, "selected_Z_slcs"] = str(sorted(slice_dict['z']))
                df.at[idx, "selected_X_slcs"] = str(sorted(slice_dict['x']))
                df.at[idx, "selected_Y_slcs"] = str(sorted(slice_dict['y']))

        except Exception as e:
            print(f"[Warning] Failed to process patient {patient_id}: {e}")
            continue

    # Save updated CSV
    df.to_csv(metadata_path, index=False)
    print(f"Metadata CSV saved at: {metadata_path}")

def get_segments_paths(segment_dir): 
    # Define the paths to the original files
    return {
        #"CT": os.path.join(segment_dir, "OG_CT.nii.gz"),
        "LV": "heart_ventricle_left.nii.gz",
        "RV": "heart_ventricle_right.nii.gz",
        "LA": "heart_atrium_left.nii.gz",
        "RA": "heart_atrium_right.nii.gz",
        "MYO" : "heart_myocardium.nii.gz",
    }
    #return OG_file_paths
    
def get_resampled_paths(resampled_dir):
	# Define the paths to the resampled files
	return {
		"CT":       os.path.join(resampled_dir, "resampled_ct.nii.gz"),
		"LV":       os.path.join(resampled_dir, "resampled_lv.nii.gz"),
		"RV":       os.path.join(resampled_dir, "resampled_rv.nii.gz"),
		"LA":       os.path.join(resampled_dir, "resampled_la.nii.gz"),
		"RA":       os.path.join(resampled_dir, "resampled_ra.nii.gz"),
		"MYO":      os.path.join(resampled_dir, "resampled_myo.nii.gz"),
		"Mask":     os.path.join(resampled_dir, "resampled_mask.nii.gz"),
	}

def get_cropped_paths(cropped_dir):
    # Define the paths to the resampled files
    return {
        "CT":       os.path.join(cropped_dir, "cropped_ct.nii.gz"),
        "LV":       os.path.join(cropped_dir, "cropped_lv.nii.gz"),
        "RV":       os.path.join(cropped_dir, "cropped_rv.nii.gz"),
        "LA":       os.path.join(cropped_dir, "cropped_la.nii.gz"),
        "RA":       os.path.join(cropped_dir, "cropped_ra.nii.gz"),
        "MYO":      os.path.join(cropped_dir, "cropped_myo.nii.gz"),
        "Mask":     os.path.join(cropped_dir, "cropped_mask.nii.gz"),
    }


def load_nifti(path):
    result = {}
    if os.path.exists(path):
        nib_obj = nib.load(path)
        result[name] = {
            "name": name,
            "path": path,
            "obj": nib_obj,  				# Nibabel object
            "data": nib_obj.get_fdata(),  	# Get the data as a NumPy array
            "header": nib_obj.header, 	# Nibabel header
            "spacing_unit": nib_obj.header.get_xyzt_units(),
            "intent" : nib_obj.header.get_intent(),
            "Qform": nib_obj.header.get_qform(),
            "Sform": nib_obj.header.get_sform(),
            "descrip": nib_obj.header.get('descrip', 'none'),  # Description of the image
            "affine": nib_obj.affine,      # Affine transformation matrix
            "voxel": nib_obj.header.get_zooms(),  # Voxel spacing
            "shape XYZ": nib_obj.shape          # Shape of the data
        }
    return result




# === Old Code ===

def segment_and_crop_data(case):
    # folder paths
    base_dicom_root = f"../Takotsubo-Syndrome/data/Inputs/{case}" #{patient}/DICOM
    base_input_root = f"data/Inputs/{case}" #/{patient}
    base_output_root = f"data/Outputs/{case}" #/{patient}
    print(f"\nProcessing {case}")
    # Loop over all patient folders
    for patientID in os.listdir(base_dicom_root):
        print(f"\nPatient ID: {patientID}")

        dicom_folder = os.path.join(base_dicom_root, patientID, "DICOM")
        input_folder = os.path.join(base_input_root, patientID)
        output_folder = os.path.join(base_output_root, patientID)

        # Skip if DICOM folder doesn't exist
        if not os.path.exists(dicom_folder):
            print(f"⚠️ Skipping {patientID}: DICOM folder not found.")
            continue


        # Skip if cropped files already exist
        if all(os.path.exists(os.path.join(output_folder, f"cropped_{ch}.nii.gz")) for ch in ["ct", "lv", "rv", "la", "ra", "myo", "combined_mask"]):
            print(f"✅ Skipping {patientID}: already segmented and cropped.")
            continue

        # Create input/output folders if they don't exist
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        try:
            # Convert DICOM to NIfTI
            dicom2nifti.convert_dicom.dicom_series_to_nifti(
                dicom_folder,
                os.path.join(input_folder, "ct_image.nii.gz"),
                )

            # Segment heart
            _ = totalsegmentator(
                dicom_folder, input_folder,
                license_number="aca_BWYHC6UQQFDU8A",
                task="heartchambers_highres", body_seg=True
            )

            # Load CT and masks
            ct_img = nib.load(os.path.join(input_folder, "ct_image.nii.gz"))
            ct_data = ct_img.get_fdata()
            lv = nib.load(os.path.join(input_folder, "heart_ventricle_left.nii.gz")).get_fdata()
            rv = nib.load(os.path.join(input_folder, "heart_ventricle_right.nii.gz")).get_fdata()
            la = nib.load(os.path.join(input_folder, "heart_atrium_left.nii.gz")).get_fdata()
            ra = nib.load(os.path.join(input_folder, "heart_atrium_right.nii.gz")).get_fdata()
            myo = nib.load(os.path.join(input_folder, "heart_myocardium.nii.gz")).get_fdata()

            # Combine masks to find heart bounding box
            heart_mask = ((lv + rv + la + ra + myo) > 0).astype(np.uint8)
            coords = np.array(np.where(heart_mask))
            x_min, y_min, z_min = coords.min(axis=1)
            x_max, y_max, z_max = coords.max(axis=1)
            
            x0, x1 = max(x_min, 0), min(x_max, ct_data.shape[0])
            y0, y1 = max(y_min, 0), min(y_max, ct_data.shape[1])
            z0, z1 = max(z_min, 0), min(z_max, ct_data.shape[2])
            

            # Further trim empty slices
            trimmed_mask = (lv_crop + rv_crop + la_crop + ra_crop + myo_crop) > 0
            x_rng, y_rng, z_rng = trim_empty_slices(trimmed_mask.astype(np.uint8))
            ct_crop = ct_data[x0:x1, y0:y1, z0:z1][x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
            lv_crop = lv[x0:x1, y0:y1, z0:z1][x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
            rv_crop = rv[x0:x1, y0:y1, z0:z1][x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
            la_crop = la[x0:x1, y0:y1, z0:z1][x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
            ra_crop = ra[x0:x1, y0:y1, z0:z1][x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]
            myo_crop = myo[x0:x1, y0:y1, z0:z1][x_rng[0]:x_rng[1], y_rng[0]:y_rng[1], z_rng[0]:z_rng[1]]

            nib.save(nib.Nifti1Image(ct_crop, ct_img.affine), os.path.join(output_folder, "cropped_ct.nii.gz"))
            nib.save(nib.Nifti1Image(lv_crop, ct_img.affine), os.path.join(output_folder, "cropped_lv.nii.gz"))
            nib.save(nib.Nifti1Image(rv_crop, ct_img.affine), os.path.join(output_folder, "cropped_rv.nii.gz"))
            nib.save(nib.Nifti1Image(la_crop, ct_img.affine), os.path.join(output_folder, "cropped_la.nii.gz"))
            nib.save(nib.Nifti1Image(ra_crop, ct_img.affine), os.path.join(output_folder, "cropped_ra.nii.gz"))
            nib.save(nib.Nifti1Image(myo_crop, ct_img.affine), os.path.join(output_folder, "cropped_myo.nii.gz"))

            # === Load cropped masks and CT scan ===
            ct_img = nib.load(f"{output_folder}/cropped_ct.nii.gz")
            ct_cropped = ct_img.get_fdata()       
            lv_cropped = nib.load(f"{output_folder}/cropped_lv.nii.gz").get_fdata()
            rv_cropped = nib.load(f"{output_folder}/cropped_rv.nii.gz").get_fdata()
            la_cropped = nib.load(f"{output_folder}/cropped_la.nii.gz").get_fdata()
            ra_cropped = nib.load(f"{output_folder}/cropped_ra.nii.gz").get_fdata()
            myo_cropped = nib.load(f"{output_folder}/cropped_myo.nii.gz").get_fdata()

            # === Combine masks into a single label map ===
            combined_mask = np.zeros_like(ct_cropped, dtype=np.uint8)
            combined_mask[lv_cropped > 0] = 1  # LV -             
            combined_mask[rv_cropped > 0] = 2  # RV - 
            combined_mask[la_cropped > 0] = 3  # LA - 
            combined_mask[ra_cropped > 0] = 4  # RA - 
            combined_mask[myo_cropped > 0] = 5  # Myo - 

            nib.save(nib.Nifti1Image(combined_mask, affine=ct_img.affine), os.path.join(output_folder, "combined_cropped_mask.nii.gz"))

            #print(f"✅ Finished: {patientID}")

        except Exception as e:
            print(f"❌ Failed for {patientID}: {e}")
        #break
        
def slicing_data(case):
    # folder paths
    base_output_root = f"data/Outputs/{case}" #/{patient}

    # Loop over all patient folders
    for patientID in os.listdir(base_output_root):
        #print(f"\nPatient ID: {patientID}")
        output_folder = os.path.join(base_output_root, patientID)
        
        # Skip if output folder doesn't exist
        if not os.path.exists(output_folder):
            print(f"⚠️ Skipping {patientID}: Output folder not found.")
            continue
        
        output_dir = f"data/Outputs/{case}/{patientID}/png_slices"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # === Load cropped masks and CT scan ===
            ct = nib.load(f"{output_folder}/cropped_ct.nii.gz").get_fdata()
            #combined_mask = nib.load(f"{output_folder}/combined_cropped_mask.nii.gz").get_fdata()
            lv_cropped = nib.load(f"{output_folder}/cropped_lv.nii.gz").get_fdata()
            
            #z_slices = get_three_slices_within(np.where(np.any(lv_cropped > 0, axis=(0, 1)))[0])
            #y_slices = get_three_slices_within(np.where(np.any(lv_cropped > 0, axis=(0, 2)))[0])
            #x_slices = get_three_slices_within(np.where(np.any(lv_cropped > 0, axis=(1, 2)))[0])
            
            # Save sagittal (X) slices
            for idx in get_three_slices_within(np.where(np.any(lv_cropped > 0, axis=(1, 2)))[0]):
                slice_data = convert_slice_to_PNG(ct[idx, :, :])
                imageio.imwrite(os.path.join(output_dir, f"sliceX_{idx}.png"), np.rot90(slice_data))
            
            # Save coronal (Y) slices
            for idx in get_three_slices_within(np.where(np.any(lv_cropped > 0, axis=(0, 2)))[0]):
                slice_data = convert_slice_to_PNG(ct[:, idx, :])
                imageio.imwrite(os.path.join(output_dir, f"sliceY_{idx}.png"), np.rot90(slice_data))
            
            # Save axial (Z) slices
            for idx in get_three_slices_within(np.where(np.any(lv_cropped > 0, axis=(0, 1)))[0]):
                slice_data = convert_slice_to_PNG(ct[:, :, idx])
                imageio.imwrite(os.path.join(output_dir, f"sliceZ_{idx}.png"), np.rot90(slice_data))

        except Exception as e:
            print(f"⚠️ Error processing {patientID}: {e}")
        
def save_slice(slice_data, slices_dir, axis_label, idx, kind):
    out_path = os.path.join(slices_dir, f"{kind}{axis_label}_{idx}.png")
    if kind == "ct":
        image = convert_slice_to_PNG(slice_data)
        image = np.rot90(image)
    elif kind == "mask":
        image = np.rot90(slice_data.astype(np.uint8))
    else:
        raise ValueError(f"Unknown kind: {kind}")
    imageio.imwrite(out_path, image)        
        
def convert_slice_to_PNG(slice_data):
    min_val = np.min(slice_data)
    ptp_val = np.ptp(slice_data)  # same as max - min
    if ptp_val == 0:
        return np.zeros_like(slice_data, dtype=np.uint8)
    norm = (slice_data - min_val) / ptp_val
    return (norm * 255).astype(np.uint8)         
        
