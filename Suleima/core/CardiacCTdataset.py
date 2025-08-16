import torch
from sklearn.model_selection import train_test_split
from monai.transforms import LoadImaged
from monai.transforms import (
	Compose, EnsureChannelFirstd, Orientationd, Lambdad,
	CropForegroundd, NormalizeIntensityd, Resized, EnsureTyped,
	RandAffined, RandGaussianNoiseD)
from core.globals import *
from core.model_utils import *
import logging
import numpy as np
from core.CNNmodel import *
from core.preprocessing import *
from torch.utils.data import Dataset, DataLoader
logger = logging.getLogger('root')



class DataLoaderFactory:
	"""
	Creates PyTorch DataLoaders for specific folds of a pre-computed
	cross-validation setup.
	"""
	def __init__(self, main_dataset, all_folds_data):
		self.main_dataset = main_dataset
		self.all_folds_data = all_folds_data
		# You can also store constants like num_workers here
		self.num_workers = 4

	def create_inner_loaders(self, outer_fold_id, inner_fold_id):
		"""
		Generates train and validation dataloaders for a specific inner fold.
		"""
		# --- 1. Get the correct data for the specified fold ---
		outer_fold_struct = self.all_folds_data[outer_fold_id]
		inner_fold_struct = outer_fold_struct['INNER_FOLDS'][inner_fold_id]

		# The outer_train_pool is the dataset from which inner folds are made
		outer_train_pool = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]

		# Get the specific train/val data for the inner fold
		# Note: The indices are local to the outer_train_pool
		train_fold_data = [outer_train_pool[i] for i in inner_fold_struct['INNER_FOLD_TRAIN_idx']]
		val_fold_data = [outer_train_pool[i] for i in inner_fold_struct['INNER_FOLD_VAL_idx']]

		# --- 2. Get the pre-calculated stats and create transforms ---
		# It's crucial to use the stats calculated from the *inner* training set
		# to avoid any data leakage from the inner validation set.
		inner_stats = inner_fold_struct['INNER_FOLD_stats']

		train_transforms = get_train_transforms(inner_stats)
		val_transforms = get_val_test_transforms(inner_stats)

		# --- 3. Create Dataset and DataLoader objects ---
		train_dataset = CardiacCTDataset(train_fold_data, train_transforms, inner_stats)
		val_dataset = CardiacCTDataset(val_fold_data, val_transforms, inner_stats)

		train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=self.num_workers)
		val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)

		return train_loader, val_loader



	def create_outer_loaders(self, outer_fold_id):
		"""
		Generates train and test dataloaders for a specific outer fold.
		"""
		outer_fold_struct = self.all_folds_data[outer_fold_id]
		fold_pool = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]
		fold_pool_labels = [self.main_dataset[i]['label'] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]
		train_fold_data, val_fold_data = train_test_split(
			fold_pool,
			test_size=0.1,
			random_state=42,
			stratify=fold_pool_labels)
		test_fold_data = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TEST_idx']]
		outer_stats = outer_fold_struct['OUTER_FOLD_stats']

		train_transforms = get_train_transforms(outer_stats)
		train_dataset = CardiacCTDataset(train_fold_data, train_transforms, outer_stats)
		train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=self.num_workers)

		test_transforms = get_val_test_transforms(outer_stats)
		val_dataset = CardiacCTDataset(val_fold_data, test_transforms, outer_stats)
		val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)
		test_dataset = CardiacCTDataset(test_fold_data, test_transforms, outer_stats)
		test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)

		return train_loader, val_loader, test_loader



def clamp_hu_values(image_tensor):
	return torch.clamp(image_tensor, min=-175.0, max=250.0)

def get_base_transforms(stats, shape=(64, 64, 64)):
	"""Returns the list of base transforms without augmentation."""
	return [
		EnsureChannelFirstd(keys=["image", "mask"]),
		Orientationd(keys=["image", "mask"], axcodes="RAS"),
		Lambdad(keys=["image"], func=clamp_hu_values),
		CropForegroundd(keys=["image", "mask"], source_key="mask"),
		NormalizeIntensityd(keys=["image"], subtrahend=stats['HUmean'], divisor=stats['HUstd']),
		Resized(keys=["image", "mask"], spatial_size=shape, mode=("bilinear", "nearest")),
		EnsureTyped(keys=["image", "mask"])
	]

def get_train_transforms(stats):
	"""Applies augmentations on top of the base transforms."""
	base_transforms = get_base_transforms(stats)
	augmentation_transforms = [
		RandAffined(keys=['image', 'mask'], prob=0.5, rotate_range=(0, 0, np.pi/12), scale_range=(0.1, 0.1, 0.1)),
		RandGaussianNoiseD(keys=['image'], prob=0.1)
	]
	return Compose(base_transforms + augmentation_transforms)

def get_val_test_transforms(stats):
	"""Validation/Test transforms have no augmentation."""
	return Compose(get_base_transforms(stats))


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
		#loaded_dict = self.loader({
		#	"image": case_dict['cropped']['image'],
		#	"mask": case_dict['cropped']['mask']})

		#processed_3D = self.transforms(loaded_dict)
		#image_3d = processed_3D["image"]
		#mask_3d = processed_3D["mask"]
		#slice_indices = select_slices(mask_3d)

		# 4. Extract the 2D slices from the 3D IMAGE tensor
		# MONAI standard orientation (RAS) is (H, W, D) which corresponds to
		# Sagittal, Coronal, and Axial planes.

		# Axial slices (from the Depth axis)
		#axial1 = image_3d[0, :, :, slice_indices['Axial'][0]]
		#axial2 = image_3d[0, :, :, slice_indices['Axial'][1]]
		#axial3 = image_3d[0, :, :, slice_indices['Axial'][2]]

		## Coronal slices (from the Width axis)
		#coronal1 = image_3d[0, :, slice_indices['Coronal'][0], :]
		#coronal2 = image_3d[0, :, slice_indices['Coronal'][1], :]
		#coronal3 = image_3d[0, :, slice_indices['Coronal'][2], :]

		## Sagittal slices (from the Height axis)
		#sagittal1 = image_3d[0, slice_indices['Sagittal'][0], :, :]
		#sagittal2 = image_3d[0, slice_indices['Sagittal'][1], :, :]
		#sagittal3 = image_3d[0, slice_indices['Sagittal'][2], :, :]

		# 5. create 3-channel 2D inputs per view
		# Conv2d expects (C, H, W), stack on the channel dim=0
		#axial_image = torch.stack([axial1, axial2, axial3], dim=0)
		#coronal_image = torch.stack([coronal1, coronal2, coronal3], dim=0)
		#sagittal_image = torch.stack([sagittal1, sagittal2, sagittal3], dim=0)

		# One-hot encode gender ('F' -> [1, 0], 'M' -> [0, 1])
		gender = [1.0, 0.0] if case_dict['gender'] == 'F' else [0.0, 1.0]
		age = (case_dict['age'] - self.stats['AGEmean']) / self.stats['AGEstd']

		meta = torch.tensor([age] + gender, dtype=torch.float32)
		label = torch.tensor(case_dict['label'], dtype=torch.float32)

		return {"CaseID": case_dict['ID'],
			#"axial_image": axial_image,
			#"coronal_image": coronal_image,
			#"sagittal_image": sagittal_image,
			"label": label, "meta": meta}

def select_slices(mask):
	lv_mask = (mask == 3)
	_, H, W, D = mask.shape
	h_indices, w_indices, d_indices = torch.where(lv_mask.squeeze(0))
	h_min, h_max = h_indices.min(), h_indices.max()
	w_min, w_max = w_indices.min(), w_indices.max()
	d_min, d_max = d_indices.min(), d_indices.max()

	# --- Axial ---
	d_span = d_max - d_min
	d_base = d_min + int(0.20 * d_span)
	d_mid =  d_min + int(0.50 * d_span)
	d_apex = d_min + int(0.80 * d_span)
	# --- Coronal & Sagittal  ---
	w_optimal = max(range(w_min, w_max + 1), key=lambda w: lv_mask[:, :, w, :].sum())
	h_optimal = max(range(h_min, h_max + 1), key=lambda h: lv_mask[:, h, :, :].sum())
	h_span = h_max - h_min; w_span = w_max - w_min
	h_gap = max(1, int(0.15 * h_span)); w_gap = max(1, int(0.15 * w_span))

	return {"Axial": (d_base, d_mid, d_apex),
			"Sagittal": (max(0, h_optimal - h_gap), h_optimal, min(H - 1, h_optimal + h_gap)),
			"Coronal": (max(0, w_optimal - w_gap), w_optimal, min(W - 1, w_optimal + w_gap))}












