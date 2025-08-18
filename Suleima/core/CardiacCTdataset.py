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




class DataLoaderFactory:
	"""
	Creates PyTorch DataLoaders for specific folds of a pre-computed
	cross-validation setup.
	"""
	def __init__(self, main_dataset):
		self.main_dataset = main_dataset
		#self.all_folds_data = all_folds_data
		# You can also store constants like num_workers here
		self.num_workers = 4

	def create_inner_loaders(self, outer_fold_id, inner_fold_id):
		"""
		Generates train and validation dataloaders for a specific inner fold.
		"""
		with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
			all_folds_data = pickle.load(f)
		# --- 1. Get the correct data for the specified fold ---
		outer_fold_struct = all_folds_data[outer_fold_id]
		#outer_fold_struct = self.all_folds_data[outer_fold_id]
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

	def create_outer_loadersSingleView(self, outer_fold_id):
		"""
		Generates train and test dataloaders for a specific outer fold.
		"""
		with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
			all_folds_data = pickle.load(f)

		outer_fold_struct = all_folds_data[outer_fold_id]
		fold_pool = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]
		fold_pool_labels = [self.main_dataset[i]['label'] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]

		train_fold_data, val_fold_data = train_test_split(
			fold_pool,
			test_size=10,
			random_state=42,
			stratify=fold_pool_labels)

		test_fold_data = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TEST_idx']]
		outer_stats = outer_fold_struct['OUTER_FOLD_stats']

		train_transforms = get_train_transforms(outer_stats)
		train_dataset = SingleViewDataset(train_fold_data, train_transforms, outer_stats)
		train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=self.num_workers)

		test_transforms = get_val_test_transforms(outer_stats)
		val_dataset = SingleViewDataset(val_fold_data, test_transforms, outer_stats)
		val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)

		test_dataset = SingleViewDataset(test_fold_data, test_transforms, outer_stats)
		test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)

		return train_loader, val_loader, test_loader

	def create_outer_loaders(self, outer_fold_id):
		"""
		Generates train and test dataloaders for a specific outer fold.
		"""
		with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
			all_folds_data = pickle.load(f)

		outer_fold_struct = all_folds_data[outer_fold_id]
		fold_pool = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]
		fold_pool_labels = [self.main_dataset[i]['label'] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]

		train_fold_data, val_fold_data = train_test_split(
			fold_pool,
			test_size=10,
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


	def create_outer_loadersMLP(self, outer_fold_id):
		"""
		Generates train and test dataloaders for a specific outer fold.
		"""
		with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
			all_folds_data = pickle.load(f)

		outer_fold_struct = all_folds_data[outer_fold_id]
		fold_pool = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]
		fold_pool_labels = [self.main_dataset[i]['label'] for i in outer_fold_struct['OUTER_FOLD_TRAIN_idx']]

		train_fold_data, val_fold_data = train_test_split(
			fold_pool,
			test_size=10,
			random_state=42,
			stratify=fold_pool_labels)

		test_fold_data = [self.main_dataset[i] for i in outer_fold_struct['OUTER_FOLD_TEST_idx']]
		outer_stats = outer_fold_struct['OUTER_FOLD_stats']

		train_dataset = METADataset(train_fold_data, outer_stats)
		train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=self.num_workers)

		val_dataset = METADataset(val_fold_data, outer_stats)
		val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)

		test_dataset = METADataset(test_fold_data, outer_stats)
		test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=self.num_workers)

		return train_loader, val_loader, test_loader





def get_fold_stats(outer_fold_id=None, inner_fold_id=None):
	"""
	Returns a tuple containing
		(outer_fold_stats, inner_fold_stats)
	for the specified fold.
	"""
	#with open("training/folds_indices_stats.pkl", "rb") as f:
	with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
		all_folds_data = pickle.load(f)
	if outer_fold_id is None and inner_fold_id is None:
		return all_folds_data
	elif outer_fold_id is None:
		# Return all outer folds stats
		return [fold['OUTER_FOLD_stats'] for fold in all_folds_data]
	elif inner_fold_id is None:
		# Return all inner folds stats for the specified outer fold
		return [fold['INNER_FOLD_stats'] for fold in all_folds_data[outer_fold_id]['INNER_FOLDS']]
	else:
		# Return stats for the specified outer and inner fold
		outer_fold_data = all_folds_data[outer_fold_id]
		inner_fold_data = outer_fold_data['INNER_FOLDS'][inner_fold_id]
		return outer_fold_data['OUTER_FOLD_stats'], inner_fold_data['INNER_FOLD_stats']





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
		loaded_dict = self.loader({
			"image": case_dict['cropped']['image'],
			"mask": case_dict['cropped']['mask']})

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

		## Coronal slices (from the Width axis)
		coronal1 = image_3d[0, :, slice_indices['Coronal'][0], :]
		coronal2 = image_3d[0, :, slice_indices['Coronal'][1], :]
		coronal3 = image_3d[0, :, slice_indices['Coronal'][2], :]

		## Sagittal slices (from the Height axis)
		sagittal1 = image_3d[0, slice_indices['Sagittal'][0], :, :]
		sagittal2 = image_3d[0, slice_indices['Sagittal'][1], :, :]
		sagittal3 = image_3d[0, slice_indices['Sagittal'][2], :, :]

		# 5. create 3-channel 2D inputs per view
		# Conv2d expects (C, H, W), stack on the channel dim=0
		axial_image = torch.stack([axial1, axial2, axial3], dim=0)
		coronal_image = torch.stack([coronal1, coronal2, coronal3], dim=0)
		sagittal_image = torch.stack([sagittal1, sagittal2, sagittal3], dim=0)

		# One-hot encode gender ('F' -> [1, 0], 'M' -> [0, 1])
		gender = [1.0, 0.0] if case_dict['gender'] == 'F' else [0.0, 1.0]
		age = (case_dict['age'] - self.stats['AGEmean']) / self.stats['AGEstd']

		meta = torch.tensor([age] + gender, dtype=torch.float32)
		label = torch.tensor(case_dict['label'], dtype=torch.float32)

		return {"CaseID": case_dict['ID'],
			"axial_image": axial_image,
			"coronal_image": coronal_image,
			"sagittal_image": sagittal_image,
			"label": label, "meta": meta}


class SingleViewDataset(Dataset):
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
			"mask": case_dict['cropped']['mask']})

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

		## Coronal slices (from the Width axis)
		#coronal1 = image_3d[0, :, slice_indices['Coronal'][0], :]
		#coronal2 = image_3d[0, :, slice_indices['Coronal'][1], :]
		#coronal3 = image_3d[0, :, slice_indices['Coronal'][2], :]

		## Sagittal slices (from the Height axis)
		sagittal1 = image_3d[0, slice_indices['Sagittal'][0], :, :]
		sagittal2 = image_3d[0, slice_indices['Sagittal'][1], :, :]
		sagittal3 = image_3d[0, slice_indices['Sagittal'][2], :, :]

		# 5. create 3-channel 2D inputs per view
		# Conv2d expects (C, H, W), stack on the channel dim=0
		#axial_image = torch.stack([axial1, axial2, axial3], dim=0)
		#coronal_image = torch.stack([coronal1, coronal2, coronal3], dim=0)
		sagittal_image = torch.stack([sagittal1, sagittal2, sagittal3], dim=0)

		# One-hot encode gender ('F' -> [1, 0], 'M' -> [0, 1])
		gender = [1.0, 0.0] if case_dict['gender'] == 'F' else [0.0, 1.0]
		age = (case_dict['age'] - self.stats['AGEmean']) / self.stats['AGEstd']

		meta = torch.tensor([age] + gender, dtype=torch.float32)
		label = torch.tensor(case_dict['label'], dtype=torch.float32)

		return {"CaseID": case_dict['ID'],
			#"axial_image": axial_image,
			#"coronal_image": coronal_image,
			"sagittal_image": sagittal_image,
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






class METADataset(Dataset):
	def __init__(self, data_dicts, stats):
		self.data_dicts = data_dicts
		self.stats = stats

	def __len__(self):
		return len(self.data_dicts)

	def __getitem__(self, idx):
		case_dict = self.data_dicts[idx]

		# One-hot encode gender ('F' -> [1, 0], 'M' -> [0, 1])
		gender = [1.0, 0.0] if case_dict['gender'] == 'F' else [0.0, 1.0]
		age = (case_dict['age'] - self.stats['AGEmean']) / self.stats['AGEstd']

		meta = torch.tensor([age] + gender, dtype=torch.float32)
		label = torch.tensor(case_dict['label'], dtype=torch.float32)

		return {"CaseID": case_dict['ID'],
			"label": label, "meta": meta}



from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import cv2


class DenseORB:
	def __init__(self, step=6, patch_size=31, nfeatures_per_slice=2000):
		self.step = step
		self.patch_size = patch_size
		# ORB detector/descriptor (rotation + FAST/BRIEF-ish)
		self.orb = cv2.ORB_create(
			nfeatures=nfeatures_per_slice,
			edgeThreshold=patch_size,
			patchSize=patch_size,
			WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE
		)

	def grid_keypoints(self, h, w, mask=None):
		"""
		Create a dense grid of keypoints, optionally masked (boolean mask).
		"""
		kps = []
		for y in range(self.patch_size, h - self.patch_size, self.step):
			# stride over valid region considering patch_size margin
			row = []
			for x in range(self.patch_size, w - self.patch_size, self.step):
				if mask is not None and mask[y, x] == 0:
					continue
				row.append(cv2.KeyPoint(x=float(x), y=float(y), size=float(self.patch_size)))
			kps.extend(row)
		return kps

	def __call__(self, img_uint8, roi_mask=None):
		"""
		img_uint8: (H,W) uint8 image 0..255
		roi_mask: optional (H,W) {0,1} or bool to restrict keypoints
		returns: (N,32) uint8 ORB descriptors (or shape (0,32) if none)
		"""
		h, w = img_uint8.shape[:2]
		kps = self.grid_keypoints(h, w, roi_mask)
		if not kps:
			return np.zeros((0, 32), dtype=np.uint8)
		# compute descriptors at given keypoints
		_, desc = self.orb.compute(img_uint8, kps)
		if desc is None:
			return np.zeros((0, 32), dtype=np.uint8)
		return desc  # uint8 (N,32)


class VLADEncoder:
	"""
	VLAD with power-normalization (signed-sqrt) and L2 norm.
	"""
	def __init__(self, k=64, batch_size=4096, random_state=0):
		self.k = k
		self.batch_size = batch_size
		self.random_state = random_state
		self.kmeans = None
		self.centers_ = None  # (k,D)
		self.D_ = None

	def fit(self, descriptor_iterable):
		"""
		descriptor_iterable: iterable of arrays (Ni, D) (from training set)
		Fits MiniBatchKMeans on stacked descriptors.
		"""
		rs = check_random_state(self.random_state)
		all_desc = []
		for D_i in descriptor_iterable:
			if D_i is not None and len(D_i) > 0:
				all_desc.append(D_i.astype(np.float32))
		if not all_desc:
			raise ValueError("No descriptors provided to fit VLAD codebook.")
		X = np.vstack(all_desc)  # (N_total, D)
		self.D_ = X.shape[1]
		mbk = MiniBatchKMeans(
			n_clusters=self.k, random_state=self.random_state,
			batch_size=self.batch_size, n_init='auto'
		)
		mbk.fit(X)
		self.kmeans = mbk
		self.centers_ = mbk.cluster_centers_.astype(np.float32)
		return self

	def encode(self, desc):
		"""
		desc: (N,D) float32 descriptors for ONE image/slice.
		returns: (k*D,) float32 VLAD vector with power+L2 normalization.
		"""
		if desc is None or len(desc) == 0:
			# zero vector if no features
			return np.zeros(self.k * self.D_, dtype=np.float32)
		X = desc.astype(np.float32)
		idx = self.kmeans.predict(X)  # (N,)
		# residuals accumulation
		V = np.zeros((self.k, self.D_), dtype=np.float32)
		for c in range(self.k):
			if np.any(idx == c):
				residuals = X[idx == c] - self.centers_[c]  # (Nc,D)
				V[c] = residuals.sum(axis=0)
		v = V.reshape(-1)  # (k*D,)

		# Power-normalization (signed-sqrt) then L2
		v = np.sign(v) * np.sqrt(np.abs(v) + 1e-12)
		nrm = np.linalg.norm(v) + 1e-12
		v = v / nrm
		return v.astype(np.float32)


def get_train_transforms(stats):
	# For classical features, geometric augmentation is optional.
	# Keep deterministic preprocessing to maintain descriptor stability.
	return Compose(get_base_transforms(stats))




class ORBVLADDataset(Dataset):
	"""
	Produces a single feature vector per case:
		features: np.float32 of shape (3 * k * D)  [if VLAD per view averaged across its 3 slices]
		label: float scalar (0/1)
		meta: [z-scored age, one-hot gender(2)]
	"""
	def __init__(self, data_dicts, transforms, stats, vlad_encoder: VLADEncoder,
				 pca: PCA = None, orb_step=6, orb_patch=31, to_uint8=True):
		self.data_dicts = data_dicts
		self.stats = stats
		self.transforms = transforms
		self.loader = LoadImaged(keys=["image", "mask"])
		self.vlad = vlad_encoder
		self.pca = pca

		self.extractor = DenseORB(step=orb_step, patch_size=orb_patch)
		self.to_uint8 = to_uint8

	def __len__(self):
		return len(self.data_dicts)

	@staticmethod
	def _to_uint8(img2d: np.ndarray):
		"""
		Map float slice (post-normalization) to uint8 [0,255].
		Stable linear scaling by robust range.
		"""
		x = img2d.astype(np.float32)
		lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
		if hi <= lo:  # degenerate
			hi = lo + 1.0
		x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
		x = (x * 255.0).round().astype(np.uint8)
		return x

	@staticmethod
	def _mask2d(mask_3d, plane, idx):
		if plane == 'Axial':
			m = mask_3d[0, :, :, idx]
		elif plane == 'Coronal':
			m = mask_3d[0, :, idx, :]
		elif plane == 'Sagittal':
			m = mask_3d[0, idx, :, :]
		else:
			raise ValueError("Unknown plane")
		m = (m > 0).cpu().numpy().astype(np.uint8)
		return m

	def _slice2d(self, img3d, plane, idx):
		if plane == 'Axial':
			s = img3d[0, :, :, idx]
		elif plane == 'Coronal':
			s = img3d[0, :, idx, :]
		elif plane == 'Sagittal':
			s = img3d[0, idx, :, :]
		else:
			raise ValueError("Unknown plane")
		return s.cpu().numpy().astype(np.float32)

	def _encode_view(self, img3d, mask3d, plane, indices):
		"""
		Encode 3 slices of a given plane; return mean VLAD over its 3 slices.
		"""
		vlad_list = []
		for idx in indices:
			sl = self._slice2d(img3d, plane, idx)
			roi = self._mask2d(mask3d, plane, idx)

			if self.to_uint8:
				sl_u8 = self._to_uint8(sl)
			else:
				# If you prefer CLAHE, add it here before uint8 cast.
				sl_u8 = self._to_uint8(sl)

			desc = self.extractor(sl_u8, roi_mask=roi)   # (N,32) uint8
			desc = desc.astype(np.float32)

			v = self.vlad.encode(desc)  # (k*D,)
			vlad_list.append(v)
		if len(vlad_list) == 0:
			return np.zeros(self.vlad.k * self.vlad.D_, dtype=np.float32)
		return np.mean(np.stack(vlad_list, axis=0), axis=0).astype(np.float32)

	def __getitem__(self, idx):
		case_dict = self.data_dicts[idx]
		loaded = self.loader({
			"image": case_dict['cropped']['image'],
			"mask": case_dict['cropped']['mask']
		})
		d = self.transforms(loaded)
		img3d = d["image"]  # (1,H,W,D)
		mask3d = d["mask"]  # (1,H,W,D)

		slice_indices = select_slices(mask3d)

		v_ax = self._encode_view(img3d, mask3d, 'Axial',    slice_indices['Axial'])
		v_co = self._encode_view(img3d, mask3d, 'Coronal',  slice_indices['Coronal'])
		v_sa = self._encode_view(img3d, mask3d, 'Sagittal', slice_indices['Sagittal'])

		feat = np.concatenate([v_ax, v_co, v_sa], axis=0)  # (3*k*D,)

		# Optional PCA (fitted on training only)
		if self.pca is not None:
			feat = self.pca.transform(feat.reshape(1, -1)).astype(np.float32).ravel()

		# meta
		gender = [1.0, 0.0] if case_dict['gender'] == 'F' else [0.0, 1.0]
		age = (case_dict['age'] - self.stats['AGEmean']) / (self.stats['AGEstd'] + 1e-8)
		meta = np.asarray([age] + gender, dtype=np.float32)

		label = np.float32(case_dict['label'])
		return {
			"CaseID": case_dict['ID'],
			"features": torch.from_numpy(feat),   # torch.float32
			"meta": torch.from_numpy(meta),
			"label": torch.tensor(label)
		}


def gather_descriptors_for_codebook(dataset: ORBVLADDataset, max_per_case=5000, max_total=200_000, rng=0):
	"""
	Single-pass descriptor gathering over the dataset with its DenseORB,
	BEFORE VLAD is fitted. Uses dataset transforms and slice selection.
	"""
	rs = check_random_state(rng)
	collected = []
	total = 0
	for i in range(len(dataset)):

		case = dataset.data_dicts[i]
		loaded = dataset.loader({
			"image": case['cropped']['image'],
			"mask": case['cropped']['mask']
		})
		d = dataset.transforms(loaded)
		img3d, mask3d = d["image"], d["mask"]
		slice_indices = select_slices(mask3d)

		# helper to sample descriptors for one plane
		def per_plane(plane, indices):
			desc_all = []
			for idx in indices:
				sl = dataset._slice2d(img3d, plane, idx)
				roi = dataset._mask2d(mask3d, plane, idx)
				sl_u8 = dataset._to_uint8(sl)
				desc = dataset.extractor(sl_u8, roi_mask=roi).astype(np.float32)
				if len(desc):
					desc_all.append(desc)
			if not desc_all:
				return np.zeros((0, 32), np.float32)
			return np.vstack(desc_all)

		D_ax = per_plane('Axial',    slice_indices['Axial'])
		D_co = per_plane('Coronal',  slice_indices['Coronal'])
		D_sa = per_plane('Sagittal', slice_indices['Sagittal'])

		D = np.vstack([d for d in [D_ax, D_co, D_sa] if len(d)])  # (Ni,32)
		if len(D) == 0:
			continue
		# Randomly subsample per case to cap
		if len(D) > max_per_case:
			idxs = rs.choice(len(D), size=max_per_case, replace=False)
			D = D[idxs]
		collected.append(D)
		total += len(D)
		if total >= max_total:
			break
	if not collected:
		raise ValueError("No descriptors found in training set to fit VLAD.")
	return collected  # list of (Ni,32)


def fit_vlad_and_pca(train_data_dicts, stats, k=64, pca_dim=256, orb_step=6, orb_patch=31):
	"""
	Convenience function to:
	  1) build a temporary dataset (no PCA, dummy VLAD enc with k but unfitted)
	  2) gather descriptors -> fit VLAD
	  3) build a dataset with fitted VLAD -> compute features -> fit PCA
	Returns: (vlad_encoder, pca, scaler)  # scaler optional if you want standardization
	"""
	# deterministic transforms for features
	transforms = get_train_transforms(stats)
	dummy_vlad = VLADEncoder(k=k)
	tmp_ds = ORBVLADDataset(train_data_dicts, transforms, stats, vlad_encoder=dummy_vlad,
							pca=None, orb_step=orb_step, orb_patch=orb_patch)

	# 1) descriptors → VLAD codebook
	desc_list = gather_descriptors_for_codebook(tmp_ds, rng=0)
	vlad = VLADEncoder(k=k).fit(desc_list)

	# 2) extract raw VLAD features for PCA fit
	ds_for_pca = ORBVLADDataset(train_data_dicts, transforms, stats, vlad_encoder=vlad,
								pca=None, orb_step=orb_step, orb_patch=orb_patch)
	feats = []
	for i in range(len(ds_for_pca)):
		f = ds_for_pca[i]["features"].numpy()
		feats.append(f)
	X = np.vstack(feats).astype(np.float32)

	# Standardize before PCA (common for VLAD)
	scaler = StandardScaler(with_mean=True, with_std=True)
	Xs = scaler.fit_transform(X)

	n_samples, n_features = Xs.shape
	max_allowed = max(1, min(n_samples - 1, n_features))  # sklearn rule
	n_comp = min(pca_dim, max_allowed)

	solver = "randomized" if n_comp < 0.8 * max_allowed else "full"


	pca = PCA(n_components=n_comp, svd_solver=solver, random_state=0)
	pca.fit(Xs)
	return vlad, pca, scaler



class ClassicalFeatureFactory:
	"""
	Parallel to DataLoaderFactory, but produces DataLoaders over ORB+VLAD(+PCA) features.
	"""
	def __init__(self, main_dataset, orb_step=6, orb_patch=31, k=64, pca_dim=256):
		self.main_dataset = main_dataset
		self.num_workers = 0
		self.orb_step = orb_step
		self.orb_patch = orb_patch
		self.k = k
		self.pca_dim = pca_dim

	def _to_pool(self, idxs):
		return [self.main_dataset[i] for i in idxs]

	def create_inner_feature_loaders(self, outer_fold_id, inner_fold_id):
		with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
			all_folds_data = pickle.load(f)

		outer = all_folds_data[outer_fold_id]
		inner = outer['INNER_FOLDS'][inner_fold_id]

		outer_train_pool = self._to_pool(outer['OUTER_FOLD_TRAIN_idx'])
		train_fold_data = [outer_train_pool[i] for i in inner['INNER_FOLD_TRAIN_idx']]
		val_fold_data   = [outer_train_pool[i] for i in inner['INNER_FOLD_VAL_idx']]

		inner_stats = inner['INNER_FOLD_stats']

		# ---- Fit VLAD + PCA (train only) ----
		vlad, pca, scaler = fit_vlad_and_pca(
			train_fold_data, inner_stats,
			k=self.k, pca_dim=self.pca_dim,
			orb_step=self.orb_step, orb_patch=self.orb_patch
		)

		# Wrap scaler+pca into a callable for dataset (scaler first, then PCA)
		class ScaledPCA:
			def __init__(self, scaler, pca):
				self.scaler = scaler
				self.pca = pca
			def transform(self, X):
				Xs = self.scaler.transform(X)
				return self.pca.transform(Xs)

		spca = ScaledPCA(scaler, pca)

		# Datasets with fitted VLAD and PCA
		tr_ds = ORBVLADDataset(train_fold_data, get_train_transforms(inner_stats),
							   inner_stats, vlad_encoder=vlad, pca=spca,
							   orb_step=self.orb_step, orb_patch=self.orb_patch)
		va_ds = ORBVLADDataset(val_fold_data, get_val_test_transforms(inner_stats),
							   inner_stats, vlad_encoder=vlad, pca=spca,
							   orb_step=self.orb_step, orb_patch=self.orb_patch)

		# Simple collate: stack tensors
		def collate(batch):
			feats = torch.stack([b["features"] for b in batch], dim=0)
			meta  = torch.stack([b["meta"] for b in batch], dim=0)
			y     = torch.stack([b["label"] for b in batch], dim=0)
			ids   = [b["CaseID"] for b in batch]
			return {"features": feats, "meta": meta, "label": y, "CaseID": ids}

		train_loader = DataLoader(tr_ds, batch_size=16, shuffle=True,
								  num_workers=self.num_workers, collate_fn=collate)
		val_loader   = DataLoader(va_ds, batch_size=16, shuffle=False,
								  num_workers=self.num_workers, collate_fn=collate)

		# Store encoders for later test use on the same outer/inner setting if needed
		fold_artifacts = {"vlad": vlad, "scaler": scaler, "pca": pca}
		return train_loader, val_loader, fold_artifacts

	def create_outer_feature_loaders(self, outer_fold_id):
		with open("NCV_5_3_folds/folds_indices_stats.pkl", "rb") as f:
			all_folds_data = pickle.load(f)

		outer = all_folds_data[outer_fold_id]
		train_pool = self._to_pool(outer['OUTER_FOLD_TRAIN_idx'])
		train_labels = [self.main_dataset[i]['label'] for i in outer['OUTER_FOLD_TRAIN_idx']]
		train_fold_data, val_fold_data = train_test_split(
			train_pool, test_size=10, random_state=42, stratify=train_labels
		)
		test_fold_data = self._to_pool(outer['OUTER_FOLD_TEST_idx'])

		outer_stats = outer['OUTER_FOLD_stats']

		# Fit encoders on (train) only
		vlad, pca, scaler = fit_vlad_and_pca(
			train_fold_data, outer_stats,
			k=self.k, pca_dim=self.pca_dim,
			orb_step=self.orb_step, orb_patch=self.orb_patch
		)

		class ScaledPCA:
			def __init__(self, scaler, pca):
				self.scaler = scaler; self.pca = pca
			def transform(self, X):
				return self.pca.transform(self.scaler.transform(X))

		spca = ScaledPCA(scaler, pca)

		tr_ds = ORBVLADDataset(train_fold_data, get_train_transforms(outer_stats),
							   outer_stats, vlad_encoder=vlad, pca=spca,
							   orb_step=self.orb_step, orb_patch=self.orb_patch)
		va_ds = ORBVLADDataset(val_fold_data, get_val_test_transforms(outer_stats),
							   outer_stats, vlad_encoder=vlad, pca=spca,
							   orb_step=self.orb_step, orb_patch=self.orb_patch)
		te_ds = ORBVLADDataset(test_fold_data, get_val_test_transforms(outer_stats),
							   outer_stats, vlad_encoder=vlad, pca=spca,
							   orb_step=self.orb_step, orb_patch=self.orb_patch)

		def collate(batch):
			feats = torch.stack([b["features"] for b in batch], dim=0)
			meta  = torch.stack([b["meta"] for b in batch], dim=0)
			y     = torch.stack([b["label"] for b in batch], dim=0)
			ids   = [b["CaseID"] for b in batch]
			return {"features": feats, "meta": meta, "label": y, "CaseID": ids}

		train_loader = DataLoader(tr_ds, batch_size=16, shuffle=True,
								  num_workers=self.num_workers, collate_fn=collate)
		val_loader   = DataLoader(va_ds, batch_size=16, shuffle=False,
								  num_workers=self.num_workers, collate_fn=collate)
		test_loader  = DataLoader(te_ds, batch_size=16, shuffle=False,
								  num_workers=self.num_workers, collate_fn=collate)

		fold_artifacts = {"vlad": vlad, "scaler": scaler, "pca": pca}
		return train_loader, val_loader, test_loader, fold_artifacts





















