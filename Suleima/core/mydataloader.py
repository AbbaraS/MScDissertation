
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
	Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Lambdad,
	CropForegroundd, NormalizeIntensityd, Resized, EnsureTyped,
	RandAffined, RandGaussianNoise)
import numpy as np
from core.globals import *
from core.Log import *
from core.preprocessing import select_slices

class CardiacCTDataset(Dataset):
	def __init__(self, data_dicts, transforms, stats):
		self.data_dicts = data_dicts
		self.stats = stats
		self.transforms = transforms
		self.loader = LoadImaged(keys=["image", "mask"])

	def __len__(self):
		return len(self.data_dicts)

	def __getitem__(self, idx):
		case_dict = self.data_dicts[idx]
		loaded_dict = self.loader({
			"image": case_dict['cropped']['image'],
			"mask": case_dict['cropped']['mask']
		})

		processed_3D = self.transforms(loaded_dict)
		image_3d = processed_3D["image"]
		mask_3d = processed_3D["mask"]
		slice_indices = select_slices(mask_3d)

		# 4. Extract the 2D slices from the 3D IMAGE tensor
		# MONAI standard orientation (RAS) is (H, W, D) which corresponds to
		# Sagittal, Coronal, and Axial planes.

		# Axial slices (from the Depth axis)
		axial1 = image_3d[0, :, :, slice_indices['Axial'][0]]
		axial2 = image_3d[0, :, :, slice_indices['Axial'][1]]
		axial3 = image_3d[0, :, :, slice_indices['Axial'][2]]

		# Coronal slices (from the Width axis)
		coronal1 = image_3d[0, :, slice_indices['Coronal'][0], :]
		coronal2 = image_3d[0, :, slice_indices['Coronal'][1], :]
		coronal3 = image_3d[0, :, slice_indices['Coronal'][2], :]

		# Sagittal slices (from the Height axis)
		sagittal1 = image_3d[0, slice_indices['Sagittal'][0], :, :]
		sagittal2 = image_3d[0, slice_indices['Sagittal'][1], :, :]
		sagittal3 = image_3d[0, slice_indices['Sagittal'][2], :, :]

		# 5. Stack the slices for each view to create 3-channel 2D inputs
		# The model's Conv2d expects (C, H, W), so we stack on the channel dimension (dim=0)
		axial_image = torch.stack([axial1, axial2, axial3], dim=0)
		coronal_image = torch.stack([coronal1, coronal2, coronal3], dim=0)
		sagittal_image = torch.stack([sagittal1, sagittal2, sagittal3], dim=0)


		# One-hot encode gender ('F' -> [1, 0], 'M' -> [0, 1])
		gender = [1.0, 0.0] if case_dict['gender'] == 'F' else [0.0, 1.0]
		age = (case_dict['age'] - self.stats['AGE_mean']) / self.stats['AGE_std']

		meta = torch.tensor([age] + gender, dtype=torch.float32)
		label = torch.tensor(case_dict['label'], dtype=torch.float32)

		return {
			"axial_image": axial_image,
			"coronal_image": coronal_image,
			"sagittal_image": sagittal_image,
			"label": label,
			"meta": meta}

def get_data_loaders(train_datalist, val_datalist, test_datalist, batch_size, stats):

	shape = (96, 96, 96)
	orientation = "RAS"

	base_transforms_list = [
		EnsureChannelFirstd(keys=["image", "mask"]),
		Orientationd(keys=["image", "mask"], axcodes=orientation),
		Lambdad(keys=["image"], func=lambda x: torch.clamp(x, min=-175.0, max=250.0)),
		CropForegroundd(keys=["image", "mask"], source_key="mask"),
		NormalizeIntensityd(keys=["image"], subtrahend=stats['HU_mean'], divisor=stats['HU_std']),
		Resized(keys=["image", "mask"], spatial_size=shape, mode=("bilinear", "nearest")),
		EnsureTyped(keys=["image", "mask"])
	]

	# Validation/Test transforms have no augmentation
	val_transforms = Compose(base_transforms_list)

	# create training transforms by combining the lists
	train_transforms_list = base_transforms_list + [
		RandAffined(keys=['image', 'mask'], prob=0.5, rotate_range=(0, 0, np.pi/12), scale_range=(0.1, 0.1, 0.1)),
		RandGaussianNoise(keys=['image'], prob=0.1)
	]
	train_transforms = Compose(train_transforms_list)

	# constructor calls
	train_dataset = CardiacCTDataset(train_datalist, train_transforms, stats)
	val_dataset   = CardiacCTDataset(val_datalist, val_transforms, stats)
	test_dataset  = CardiacCTDataset(test_datalist, val_transforms, stats)

	# Use num_workers > 0 to speed up data loading
	num_workers = 4
	train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return train_loader, val_loader, test_loader







