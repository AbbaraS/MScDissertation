import os
import numpy as np
import torch
from monai.transforms import LoadImage, Resize
from monai.data.meta_tensor import MetaTensor
from monai.transforms import SaveImage
from monai.transforms import Transform
from totalsegmentator.python_api import totalsegmentator
from nibabel.nifti1 import Nifti1Image


def nifti1image_to_metatensor(nifti_img: Nifti1Image) -> MetaTensor:
    """
    Converts a Nifti1Image object to a MetaTensor with metadata.
    Args:
        nifti_img (Nifti1Image): A loaded NIfTI image.
    Returns:
        MetaTensor: A MONAI MetaTensor containing the image data and metadata.
    """
    # Convert voxel data to PyTorch tensor
    image_tensor = torch.tensor(nifti_img.get_fdata(), dtype=torch.float32)

    # Extract metadata
    metadata = {
        "affine": torch.tensor(nifti_img.affine, dtype=torch.float32),  # Affine transformation matrix
        "spacing": nifti_img.header.get_zooms(),  # Voxel spacing (x, y, z)
        "original_shape": image_tensor.shape  # Store original shape for reference
    }

    # Create a MetaTensor with metadata
    meta_tensor = MetaTensor(image_tensor, meta=metadata)

    return meta_tensor

def resize_to_target_slices(image: MetaTensor, heart_mask: MetaTensor, target_width=512, target_height=512, target_slices=128):
    """
    Extracts the heart region from the input scan and resizes it to a target number of slices, 
    preserving metadata while correctly computing voxel spacing.

    Args:
        image (MetaTensor): The input 3D CT/MRI scan.
        heart_mask (MetaTensor): Binary mask indicating heart region.
        target_slices (int, optional): Number of slices to resize to. Default is 128.

    Returns:
        MetaTensor: Resized image with updated metadata.
    """
    if isinstance(heart_mask, MetaTensor):
        heart_mask = heart_mask.as_tensor().cpu().numpy()
    
    z_indices = np.any(heart_mask > 0, axis=(0, 1))
    heart_slices = np.where(z_indices)[0]
    # print(heart_slices)
    if len(heart_slices) == 0:
        raise ValueError("No heart region found in the mask.")
    
    start_slice = heart_slices[0]
    end_slice = heart_slices[-1] + 1
    
    sliced_image = image[:, :, start_slice:end_slice]
    if not isinstance(sliced_image, MetaTensor):
        sliced_image = MetaTensor(sliced_image, meta=image.meta)
    
    height, width, original_depth = image.shape
    # sliced_depth = sliced_image.shape[-1]

    sliced_image = sliced_image.unsqueeze(0)  # Shape becomes (1, H, W, Z)
    
    resizer = Resize(spatial_size=(target_width,target_height,target_slices), mode="trilinear", align_corners=True)
    resized_image = resizer(sliced_image)
    
    resized_image = resized_image.squeeze(0)

    # Update metadata
    new_meta = image.meta.copy()  
    
    original_spacing = image.meta.get("spacing", (1.0, 1.0, 1.0))
    new_spacing = (
        original_spacing[0],
        original_spacing[1],
        original_spacing[2] * (original_depth / target_slices)
    )
    new_meta["spacing"] = new_spacing

    # Keep affine transformation matrix but adjust slice resolution
    if "affine" in new_meta:
        new_meta["affine"][-1, -1] *= (original_depth / target_slices)  # Adjust Z scaling

    # Return resized MetaTensor with updated metadata
    return MetaTensor(resized_image, meta=new_meta)

def align_segmentation_to_image(segmentation):
    """
    Aligns the TotalSegmentator output (segmentation) with the original image
    by flipping the z-axis of the segmentation mask.
    
    Parameters:
        image (torch.Tensor or np.ndarray): Original image (H x W x D).
        segmentation (torch.Tensor or np.ndarray): TotalSegmentator output (H x W x D).
        
    Returns:
        torch.Tensor: Segmentation aligned with the image.
    """
   
    # Flip along the z-axis (last dimension)
    aligned_segmentation = torch.flip(segmentation, dims=[-1])
    
    return aligned_segmentation

# Example usage
# aligned_mask = align_segmentation_to_image(full_image)


def custom_name_formatter(metadict: dict, saver: Transform) -> dict:
    """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
    according to the input metadata and SaveImage transform."""
    subject = "heart_resized"
    patch_index = None
    return {"subject": f"{subject}", "idx": patch_index}


def preprocess_full_heart(input_file_path: str):
    dicom_file_path = os.path.join(input_file_path, "DICOM")
    subfolders = sorted([f for f in os.listdir(dicom_file_path) if os.path.isdir(os.path.join(dicom_file_path, f))])
    
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in {dicom_file_path}")
    
    first_folder = subfolders[0]
    dicom_file_path = os.path.join(dicom_file_path, first_folder)    

    output_folder = input_file_path.replace("Inputs", "Outputs")
    
    if os.path.exists(f"{output_folder}/heart_resized.nii.gz"):
        image_loader = LoadImage(image_only=True, ensure_channel_first=True)
        resized_image = image_loader(f"{output_folder}/heart_resized.nii.gz")
        resized_image = resized_image.squeeze(dim=-1)
    else:
        # output_image = totalsegmentator(dicom_file_path, output_folder, license_number="aca_BWYHC6UQQFDU8A", roi_subset=["heart"], device="mps")
        output_image = totalsegmentator(dicom_file_path, output_folder, roi_subset=["heart"], device="cpu")
        output_image = nifti1image_to_metatensor(output_image)
        resized_image = resize_to_target_slices(output_image, output_image,target_height=64,target_width=64,target_slices=64)
        # nib.save(resized_image, f"{output_folder}/heart_resized.nii.gz")
        print("saving")
        image_saver = SaveImage(output_dir=f"{output_folder}", separate_folder=False, output_postfix="", output_name_formatter=custom_name_formatter)
        image_saver(resized_image)
        print("saved")
        print(resized_image.shape)
    
    return resized_image
    


# preprocess_full_heart("takotsubo_cases/testing_subset/LET 04456356")
