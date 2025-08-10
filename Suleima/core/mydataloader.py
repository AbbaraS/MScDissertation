import nibabel as nib
from sklearn.model_selection import StratifiedShuffleSplit
import os
from core.dataset import CTDataset
from core.Log import *
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import monai
from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	Orientationd,
	Lambdad,
	CropForegroundd,
	Resized,
	EnsureTyped,
	NormalizeIntensityd
)
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from core.globals import *
from pathlib import Path
import nibabel as nib
import json
import numpy as np
import logging
from core.globals import *
from core.Log import *
import monai
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	Orientationd,
	ScaleIntensityRanged,
	CropForegroundd,
	Resized,
	EnsureTyped
)
from tqdm import tqdm
class TakotsuboDataset(Dataset):
	"""
	Custom PyTorch Dataset for loading Takotsubo CT scans and masks.
	It takes a list of data dictionaries and applies MONAI transforms.
	"""
	def __init__(self, data_dicts, transforms=None):
		"""
		Args:
			data_dicts (list of dict): List where each dict has 'image', 'mask', and 'label'.
			transforms (monai.transforms.Compose): The MONAI transform pipeline.
		"""
		self.data_dicts = data_dicts
		self.transforms = transforms

	def __len__(self):
		return len(self.data_dicts)

	def __getitem__(self, idx):
		# Get the data dictionary for the current index
		data_dict = self.data_dicts[idx]

		# Apply the MONAI transforms to load and process the image/mask
		if self.transforms is not None:
			processed_data = self.transforms(data_dict)
		else:
			processed_data = {
				"image": data_dict["image"],
				"mask": data_dict["mask"]
			}

		# The label is already in the dictionary, just ensure it's a tensor
		label = torch.tensor(data_dict['label'], dtype=torch.float32)

		# Return the processed image, mask, and the label
		return {
			"image": processed_data["image"],
			"mask": processed_data["mask"],
			"label": label
		}

# --- 2. Data Preparation Function ---
def get_demographics(ID: str):
	try:
		parts = ID.split("_")
		type = parts[0]

		if len(parts) == 3: # e.g., "CNTRL_AAP50415783_61F"
			info = parts[2]
		elif len(parts) == 4: # e.g., "CNTRL_AAP_50415783_61F"
			info = parts[3]
		label = 0 if type == "CNTRL" else 1
		return {
			"label": label,         # control = 0, tts = 1
			"age": int(info[:2]),   # age = first two digits of info
			"gender": info[2]       # 'F' or 'M'
			}
	except Exception as e:
		logging.error(f"Error parsing: '{parts}' - should be AgeGender")
		return None

def get_originals(loop=None):
	root_dir = Path("data/cases")
	cases= []
	if loop is not None: c = 0
	for ID in os.listdir(root_dir):
		if ID.startswith('.'):
			continue
		set_log(ID)
		# split ID into parts

		casepath = root_dir / ID
		ct_path = casepath / "fullCT.nii.gz"
		mask_path = casepath / "label_mask.nii.gz"
		if not ct_path.exists():
			#logging.error(f"CT file not found for case {ID}. Skipping.")
			continue
		if ct_path.exists() and mask_path.exists():
			#demo = get_demographics(ID)
			#if demo is None:
			#	logging.error(f"Incorrect {ID} format. Skipping.")
			#	continue

			cases.append({
				"ID": ID,
				"directory": casepath,
				"originals": {
					"image": ct_path,
					"mask":  mask_path},
				#"label": demo["label"],
				#"age": demo["age"],
				#"gender": demo["gender"],
				#f"volumes{shape}": {
				#	"image": casepath / f"resCT_{shape}.nii.gz",
				#	"mask":  casepath / f"resMask_{shape}.nii.gz"
				#}
				#"directory": casepath,
				#"volumes64": {
				#	"image": casepath / "resCT_64.nii.gz",
				#	"mask":  casepath / "resMask_64.nii.gz"   },
				#"volumes96": {
				#	"image": casepath / "resCT_96.nii.gz",
				#	"mask":  casepath / "resMask_96.nii.gz"     },
				#"volumes128": {
				#	"image": casepath / "resCT_128.nii.gz",
				#	"mask":  casepath / "resMask_128.nii.gz"     }
			})
		else:
			continue
		#print(f"{ID}")
		if loop is not None and type(loop)== int:
			c += 1
			if c >= loop:
				break
	#print(f"cases: {len(cases)}")
	logging.info(f"Successfully prepared datalist with {len(cases)} cases.")
	return cases

# --- 3. Splitting and DataLoader Creation ---
def stratified_datalists(
	full_datalist,
	test_size=0.2,
	random_state=42):
	"""
	Splits the data into train, validation, and test sets using stratification
	and returns the corresponding PyTorch DataLoaders.
	"""
	if not full_datalist:
		raise ValueError("Input datalist is empty. Cannot create DataLoaders.")

	# Get indices and labels for stratification
	indices = np.arange(len(full_datalist))
	labels = [d['label'] for d in full_datalist]

	# --- Outer Split: (Train+Val) vs. Test ---
	# We only need one split to create a single held-out test set.
	sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
	train_val_idx, test_idx = next(sss_test.split(indices, labels))

	# Create the test set datalist
	test_datalist = [full_datalist[i] for i in test_idx]

	# Get the data and labels for the remaining (train+val) set
	train_val_datalist = [full_datalist[i] for i in train_val_idx]
	train_val_labels = [train_val_datalist[i]['label'] for i in range(len(train_val_datalist))]

	# --- Inner Split: Train & Validation ---
	# Split the remaining data. test_size here refers to the proportion of the *remaining*
	# data that will become the validation set. 0.25 of 80% is 20% of the total.
	sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
	train_idx, val_idx = next(sss_val.split(np.arange(len(train_val_datalist)), train_val_labels))

	# Create the final train and validation datalists
	train_datalist = [train_val_datalist[i] for i in train_idx]
	val_datalist = [train_val_datalist[i] for i in val_idx]

	logging.info(f"Dataset split: {len(train_datalist)} training, {len(val_datalist)} validation, {len(test_datalist)} test cases.")
	return train_datalist, val_datalist, test_datalist
	## --- Create Dataset and DataLoader instances ---
	#train_dataset = TakotsuboDataset(data_dicts=train_datalist, transforms=train_transforms)
	#val_dataset = TakotsuboDataset(data_dicts=val_datalist, transforms=val_transforms)
	#test_dataset = TakotsuboDataset(data_dicts=test_datalist, transforms=val_transforms) # Use val_transforms for test (no augmentation)

	#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	#return train_loader, val_loader, test_loader


def calculate_normalization_stats(train_datalist):
	"""
	Calculates the global mean and standard deviation of voxel intensities
	across an entire training dataset for Z-score normalization.

	This function processes one image at a time to remain memory-efficient.
	It applies initial transforms (loading, clipping, cropping) before
	calculating stats to ensure they are based on relevant voxels only.

	Args:
		train_datalist (list of dict): The list of data dictionaries for the
									   TRAINING set only. Each dict must
									   contain 'image' and 'mask' keys.

	Returns:
		A tuple containing the global mean and standard deviation.
	"""
	if not train_datalist:
		raise ValueError("Input training datalist is empty.")

	# --- 1. Define a minimal transform pipeline for stats calculation ---
	# We only need to load, orient, clip, and crop to the foreground.
	# No resizing or normalization should happen here.
	stats_transforms = Compose([
		LoadImaged(keys=["image", "mask"]),
		EnsureChannelFirstd(keys=["image", "mask"]),
		Orientationd(keys=["image", "mask"], axcodes="RAS"),
		Lambdad(keys=["image"], func=lambda x: torch.clamp(x, min=-175.0, max=250.0)),
		CropForegroundd(keys=["image", "mask"], source_key="mask")
	])

	# --- 2. Initialize variables for calculation ---
	# We use Welford's algorithm for a numerically stable one-pass calculation
	# of mean and variance.
	n = 0  # Voxel count
	mean = 0.0
	M2 = 0.0 # Sum of squares of differences from the current mean

	logging.info("Calculating normalization stats from training set...")
	# Use tqdm for a progress bar
	for item in tqdm(train_datalist, desc="Processing training images"):
		# Apply the transforms to get the cropped image
		processed_data = stats_transforms(item)
		image = processed_data['image'] # This is a tensor

		# --- 3. Accumulate stats ---
		# We only want to calculate stats on the foreground voxels (the heart).
		# The cropped image still has zero-padded areas around the heart.
		# Let's create a boolean mask for non-zero voxels.
		foreground_voxels = image[image != 0]

		if foreground_voxels.numel() == 0:
			continue # Skip if the image is empty after cropping

		# Welford's algorithm update step
		for x in foreground_voxels:
			n += 1
			delta = x - mean
			mean += delta / n
			delta2 = x - mean
			M2 += delta * delta2

	if n < 2:
		logging.error("Not enough data to compute variance. Need at least 2 voxels.")
		return 0.0, 1.0 # Return neutral values

	# --- 4. Finalize calculation ---
	variance = M2 / (n - 1) # (n-1) for sample variance
	std_dev = torch.sqrt(variance)

	mean_val = mean.item()
	std_dev_val = std_dev.item()

	logging.info(f"Calculation Complete. Found {n} foreground voxels.")
	logging.info(f"Global Mean: {mean_val:.4f}")
	logging.info(f"Global Std Dev: {std_dev_val:.4f}")

	return mean_val, std_dev_val







def save_metatensor_as_nifti(tensor, output_path: Path):
	"""
	Saves a MONAI MetaTensor to a .nii.gz file.

	It correctly uses the affine transformation stored in the tensor's
	metadata, which is essential for geometric accuracy after
	preprocessing transforms like cropping and resizing.

	Args:
		tensor (torch.Tensor): The MetaTensor to save (image or mask).
		output_path (Path): The file path for the output .nii.gz file.
	"""
	# Ensure the output directory exists
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# 1. Detach tensor from gradient computation and move to CPU
	tensor_np = tensor.detach().cpu().numpy()

	# 2. Remove the channel dimension if it exists (C, H, W, D) -> (H, W, D)
	if tensor_np.ndim == 4:
		tensor_np = tensor_np.squeeze(0)

	# 3. Get the UPDATED affine matrix from the tensor's metadata
	# This is the most critical step!
	affine = tensor.meta['affine']

	# 4. Create a new NIfTI image object
	nifti_image = nib.Nifti1Image(tensor_np, affine)

	# 5. Save the image to disk
	nib.save(nifti_image, output_path)
	#print(f"Successfully saved tensor to: {output_path}")

def create_multilabel_heart_mask(casepath: Path):
	"""
	Loads individual heart segmentations from TotalSegmentator, combines them
	into a single multi-label NIfTI file, and saves it.

	The labels are applied with priority: LV > Myocardium > Other Heart.

	Args:
		segments_dir: Path to the directory containing the .nii.gz segment files.
		output_path: Path to save the final combined_mask.nii.gz file.
	"""
	segments_dir = casepath / "segments"
	if not segments_dir.exists():
		#logging.error(f"Segments directory not found: {segments_dir}")
		log(f"Segments directory not found")
		#log(f"")
		return

	segments_data = {}
	reference_nii = None # To store affine and header from a loaded file
	missing = []
	# --- 1. Load all required segment files ---
	for key, filename in SEGMENT_FILES.items():
		file_path = segments_dir / filename
		if file_path.exists():
			#logging.info(f"Loading segment: {filename}")

			nii_obj = nib.load(file_path)
			# Use the first successfully loaded file as our reference for geometry
			if reference_nii is None:
				reference_nii = nii_obj
			# Use get_fdata() for robust data access, ensure it's boolean
			segments_data[key] = (nii_obj.get_fdata() > 0)
		else:
			#logging.warning(f"Segment file not found, skipping: {file_path}")
			missing.append(key)
			# If a critical file is missing, we can't proceed

	if missing:
		#for key in missing:
		#	if key in ["myocardium", "left_ventricle"]:
		log(f"Missing segments: {missing}")
		logging.error(f"Missing segments:{missing}. Aborting.")
		return

	if reference_nii is None:
		logging.error("No valid segment files were loaded. Cannot create mask.")
		return

	# --- 2. Create the multi-label mask array ---
	# Start with a zero-filled array with the same shape as our inputs
	shape = reference_nii.shape
	combined_mask = np.zeros(shape, dtype=np.uint8)

	# Apply labels in order of INCREASING priority.
	# The later assignments will overwrite the earlier ones where pixels overlap.

	# Label 1: General heart structures (RA, RV, LA)
	other_heart_mask = np.zeros(shape, dtype=bool)
	for key in ["right_atrium", "right_ventricle", "left_atrium"]:
		if key in segments_data:
			other_heart_mask = np.logical_or(other_heart_mask, segments_data[key])
	combined_mask[other_heart_mask] = LABEL_MAP["other_heart"]

	# Label 2: Myocardium
	if "myocardium" in segments_data:
		combined_mask[segments_data["myocardium"]] = LABEL_MAP["myocardium"]

	# Label 3: Left Ventricle (highest priority)
	if "left_ventricle" in segments_data:
		combined_mask[segments_data["left_ventricle"]] = LABEL_MAP["left_ventricle"]

	#logging.info(f"Created multi-label mask with unique values: {np.unique(combined_mask)}")

	# --- 3. Save the new mask as a NIfTI file ---
	# Create a new Nifti1Image object using the combined data array,
	# but with the affine and header from the original file.

	output_nii = nib.Nifti1Image(combined_mask, affine=reference_nii.affine, header=reference_nii.header)
	output_path = casepath / "label_mask.nii.gz"
	# Save to the specified output path
	nib.save(output_nii, output_path)
	log(f"Successfully saved multi-label mask to: {output_path}")






def crop_cases(cases, shape=64, loop=None):
	"""
	Preprocess the cases and return a dictionary of processed volumes.
	"""
	if shape == 64:
		target_shape = (64, 64, 64)
	elif shape == 128:
		target_shape = (128, 128, 128)
	elif shape == 96:
		target_shape = (96, 96, 96)

	target_orientation = "RAS"

	if loop is not None: c = 0
	for case in cases:
		casepath = case["directory"]
		ID = case["ID"]
		set_log(ID)
		originals = case["voluems"]

		preproc_transforms = Compose([
		LoadImaged(keys=KEYS), # 1. Load image and mask from file paths
		EnsureChannelFirstd(keys=KEYS), # 2. Ensure data has a channel dimension (C, H, W, D)
		Orientationd(keys=KEYS, axcodes=target_orientation), # 3. Reorient to a standard orientation
		Lambdad(
			keys=["image"],
			func=lambda x: torch.clamp(x, min=-175.0, max=250.0)# 4. Clip intensity for the IMAGE ONLY.
		),
		CropForegroundd(keys=KEYS, source_key="mask"), # 5. CROP: find the bounding box from the MASK and apply it to both image and mask.
		# 6. RESAMPLE & RESIZE to a fixed size for the model - handles resampling spacing and resizing the array in one go.
		# 'bilinear' is for the continuous image data.
		# 'nearest' is for the discrete segmentation mask to avoid creating new label values.
		Resized(keys=KEYS, spatial_size=target_shape, mode=("bilinear", "nearest")),
		EnsureTyped(keys=KEYS) # 7. Ensure the final output is a PyTorch tensor
		])
		processed_volumes = preproc_transforms(originals)
		image_tensor = processed_volumes["image"]
		mask_tensor = processed_volumes["mask"]
		image_tensor_path = casepath / f"resCT_{shape}.nii.gz"
		mask_tensor_path = casepath / f"resMASK_{shape}.nii.gz"
		save_metatensor_as_nifti(image_tensor, image_tensor_path)
		save_metatensor_as_nifti(mask_tensor, mask_tensor_path)
		if loop is not None and type(loop) == int:
			c += 1
			if c >= loop:
				break








class DataLoaderModule:
	def __init__(self, slices_dict, metadata_dict,
			  labels_dict, batch_size=1,
			  train_split=0.7,
			  val_split=0.15,
			  test_split=0.15,
			  num_workers=0,
			  seed=42):
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.seed = seed # randomise seed!!!

		self.dataset = CTDataset(slices_dict, metadata_dict, labels_dict)
		self.create_split_loaders(train_split, val_split, test_split)

	def create_split_loaders(self, train_split, val_split, test_split):
		labels = np.array([int(self.dataset[i]['label'].item()) for i in range(len(self.dataset))])
		indices = np.arange(len(labels))

		sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_split), random_state=self.seed)
		train_idx, val_test_idx = next(sss1.split(indices, labels))

		val_labels = labels[val_test_idx]
		sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_split / (val_split + test_split), random_state=self.seed)
		val_idx, test_idx = next(sss2.split(np.zeros(len(val_labels)), val_labels))

		val_idx = val_test_idx[val_idx]
		test_idx = val_test_idx[test_idx]

		self.train_loader = DataLoader(Subset(self.dataset, train_idx), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
		self.val_loader = DataLoader(Subset(self.dataset, val_idx), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
		self.test_loader = DataLoader(Subset(self.dataset, test_idx), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

	def get_train_loader(self):
		return self.train_loader

	def get_val_loader(self):
		return self.val_loader

	def get_test_loader(self):
		return self.test_loader


def load_slices(slice_folder):
	data = {
		"Axial": {"ct": [], "mask": []},
		"Coronal": {"ct": [], "mask": []},
		"Sagittal": {"ct": [], "mask": []}
	}

	for fname in os.listdir(slice_folder):
		if fname.endswith(".nii.gz"):
			fpath = os.path.join(slice_folder, fname)
			axis_code = fname.split("_")[0][-1]
			axis_name = {"X": "Sagittal", "Y": "Coronal", "Z": "Axial"}.get(axis_code)
			if not axis_name:
				continue

			try:
				idx = int(fname.split("_")[1].replace(".nii.gz", ""))
			except ValueError:
				continue

			img = nib.load(fpath).get_fdata()
			entry = {"idx": idx, "slice": img}

			if fname.startswith("ct"):
				data[axis_name]["ct"].append(entry)
			elif fname.startswith("mask"):
				data[axis_name]["mask"].append(entry)

	return data

def load_dataset():
	slices, metadata, labels = {}, {}, {}
	for case in ["normal_cases", "takotsubo_cases"]:
		label = 1 if case == "takotsubo_cases" else 0
		base_root = f"data/Outputs/{case}"
		metadata_df = pd.read_csv(f"data/{case}_metadata.csv")

		for folder in os.listdir(base_root):
			try:
				pid = folder.split("_")[0]
				row = metadata_df[metadata_df["PatientID"] == pid]
				if row.empty:
					continue

				age, gender = row["Age"].values[0], row["Gender"].values[0]
				slice_folder = os.path.join(base_root, folder, "nii_slices")
				if not os.path.isdir(slice_folder):
					continue

				slices[pid] = load_slices(slice_folder)
				metadata[pid] = {"age": age, "gender": gender}
				labels[pid] = {"label": label}
			except Exception as e:
				print(f"Error loading {folder}: {e}")
	return slices, metadata, labels


