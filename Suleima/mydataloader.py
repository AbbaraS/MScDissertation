



import nibabel as nib
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset


# === Classes ===


class DataLoaderModule:
    def __init__(self, dataset, batch_size=1, train_split=0.7, val_split=0.15, test_split=0.15, num_workers=0, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._prepare_stratified_loaders(train_split, val_split, test_split)

    def _prepare_stratified_loaders(self, train_split, val_split, test_split):

        # Extract labels
        labels = np.array([int(self.dataset[i]['label'].item()) for i in range(len(self.dataset))])

        # Initial stratified train/val+test split
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_split), random_state=self.seed)
        train_idx, val_test_idx = next(sss1.split(np.zeros(len(labels)), labels))

        # Further stratified split val/test
        val_labels = labels[val_test_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_split / (test_split + val_split), random_state=self.seed)
        val_idx, test_idx = next(sss2.split(np.zeros(len(val_test_idx)), val_labels))
        val_idx = val_test_idx[val_idx]
        test_idx = val_test_idx[test_idx]

        # Subset creation
        train_ds = Subset(self.dataset, train_idx)
        val_ds = Subset(self.dataset, val_idx)
        test_ds = Subset(self.dataset, test_idx)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader



class SliceDataset(Dataset):
    def __init__(self, slices_dict, metadata_dict, labels_dict):
        """
        Each __getitem__ returns:
        axial (Tensor), sagittal (Tensor), coronal (Tensor), metadata (Tensor), label (Tensor)

        Each view shape: (1, 3, H, W)
        """
        self.data = []

        for pid in slices_dict:
            patient_slices = slices_dict[pid]

            # Stack slices per axis as (3, H, W) then add batch dim -> (1, 3, H, W)
            def get_axis_tensor(axis):
                slices = patient_slices[axis]["ct"]
                tensor = torch.stack([
                    torch.tensor(s["slice"], dtype=torch.float32)  # (H, W)
                    for s in slices
                ])  # (3, H, W)
                return tensor  # (3, H, W)

            if all(k in patient_slices for k in ["Axial", "Sagittal", "Coronal"]):
                self.data.append({
                    "axial": get_axis_tensor("Axial"),
                    "sagittal": get_axis_tensor("Sagittal"),
                    "coronal": get_axis_tensor("Coronal"),
                    "meta": torch.tensor([
                        float(metadata_dict[pid]["age"]),
                        0.0 if metadata_dict[pid]["gender"] == "M" else 1.0
                    ], dtype=torch.float32),
                    "label": torch.tensor(labels_dict[pid]["label"], dtype=torch.float32).unsqueeze(0),
                    "pid": pid
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        return {
            "axial": entry["axial"],       # shape: (1, 3, H, W)
            "sagittal": entry["sagittal"], # shape: (1, 3, H, W)
            "coronal": entry["coronal"],   # shape: (1, 3, H, W)
            "meta": entry["meta"],         # shape: (2,)
            "label": entry["label"],        # shape: (1,)
            "pid": entry["pid"]            # Patient ID
            }

# === Methods ===

def load_slices(slice_folder):
    data = {
        "Axial": {"ct": [], "mask": []},
        "Coronal": {"ct": [], "mask": []},
        "Sagittal": {"ct": [], "mask": []}
    }

    for fname in os.listdir(slice_folder):
        if fname.endswith(".nii.gz"):
            fpath = os.path.join(slice_folder, fname)

            # Determine axis and index
            parts = fname.split("_")
            if len(parts) < 2:
                continue  # skip malformed files

            axis_code = parts[0][-1]  # ctX or maskX → 'X'
            axis_name = {"X": "Sagittal", "Y": "Coronal", "Z": "Axial"}.get(axis_code)
            if axis_name is None:
                continue

            try:
                idx = int(parts[1].replace(".nii.gz", ""))
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
    
    metadata = {}
    labels = {}
    slices = {}
    
    NORMAL='normal_cases'
    TAKO='takotsubo_cases'

    cases = [NORMAL, TAKO]
    
    for case in cases:
        label = 1 if case == TAKO else 0
        base_root = f"data/Outputs/{case}"
        print(f"Loading {case}")
        metadata_df = pd.read_csv(f"data/{case}_metadata.csv")  # PatientID, Age, Gender

        count = 0
        for folder_name in os.listdir(base_root):
            try:
                patient_id = folder_name.split("_")[0]
                row = metadata_df[metadata_df["PatientID"] == patient_id]
                if row.empty:
                    print(f"Metadata missing for {patient_id}")
                    continue

                age = row["Age"].values[0]
                gender = row["Gender"].values[0]

                slice_folder = os.path.join(base_root, folder_name, "nii_slices")
                if not os.path.isdir(slice_folder) or not os.listdir(slice_folder):
                    continue

                slices[patient_id] = load_slices(slice_folder)
                metadata[patient_id] = {"age": age, "gender": gender}
                labels[patient_id] = {"label": label}

            except Exception as e:
                print(f"Error processing folder {folder_name}: {e}")
                continue

            count += 1
            #if count >= 5:
            #    break

    return metadata, labels, slices
