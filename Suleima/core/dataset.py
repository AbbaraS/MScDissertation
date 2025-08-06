
from torch.utils.data import Dataset, DataLoader, Subset
import torch

class CTDataset(Dataset):
    def __init__(self, slices_dict, metadata_dict, labels_dict):
        self.data = []
        self._build_dataset(slices_dict, metadata_dict, labels_dict)

    def _get_axis_tensor(self, axis, slices):
        axis_slices = slices[axis]["ct"]
        return torch.stack([
            torch.tensor(s["slice"], dtype=torch.float32)
            for s in axis_slices
        ])  # Shape: (3, H, W)

    def _build_dataset(self, slices_dict, metadata_dict, labels_dict):
        for pid in slices_dict:
            slices = slices_dict[pid]
            if all(k in slices for k in ["Axial", "Sagittal", "Coronal"]):
                self.data.append({
                    "axial": self._get_axis_tensor("Axial", slices),
                    "sagittal": self._get_axis_tensor("Sagittal", slices),
                    "coronal": self._get_axis_tensor("Coronal", slices),
                    "meta": torch.tensor([
                        float(metadata_dict[pid]["age"]),
                        0.0 if metadata_dict[pid]["gender"] == "M" else 1.0
                    ], dtype=torch.float32),
                    "label": torch.tensor(labels_dict[pid]["label"], dtype=torch.float32),
                    "pid": pid
                })

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "axial": item["axial"],       # (3, H, W)
            "sagittal": item["sagittal"], # (3, H, W)
            "coronal": item["coronal"],   # (3, H, W)
            "meta": item["meta"],         # (2,)
            "label": item["label"].unsqueeze(0),  # (1,)
            "pid": item["pid"]
        }

    def __len__(self):
        return len(self.data)