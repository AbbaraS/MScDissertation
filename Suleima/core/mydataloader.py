import nibabel as nib
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import os
import pandas as pd
import numpy as np
from core.dataset import CTDataset 

from torch.utils.data import Dataset, DataLoader, Subset

class DataLoaderModule:
    def __init__(self, slices_dict, metadata_dict, labels_dict, batch_size=1, train_split=0.7, val_split=0.15, test_split=0.15, num_workers=0, seed=42):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

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


