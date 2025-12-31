# EpicKitchensDataset_KFold.py
from torch.utils.data import Dataset
import pandas as pd
import torch
import os

class EpicKitchensDataset(Dataset):
    def __init__(self, csv_path, tensor_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the specific split CSV (e.g., train_fold_0.csv)
            tensor_dir (str): Path to the folder containing .pt tensors
        """
        self.csv_path = csv_path
        self.path_to_tensors = os.path.expandvars(tensor_dir)
        
        # Hardcoded for safety
        self.num_classes = 300 
        
        print(f"Loading annotations from: {self.csv_path}")
        self.annotations = pd.read_csv(self.csv_path)
        
        self.samples = self._build_index()
        print(f"Loaded {len(self.samples)} samples.")

    def _build_index(self):
        samples = []
        if not os.path.exists(self.path_to_tensors):
             # Fallback
             self.path_to_tensors = "/scratch/leuven/380/vsc38040/x3d_train_tensors"
             
        existing_files = set(os.listdir(self.path_to_tensors))
        
        for _, row in self.annotations.iterrows():
            segment_uid = row['narration_id']
            label = int(row['noun_class'])
            
            clip_idx = 0
            while True:
                fname = f"{segment_uid}_clip{clip_idx}.pt"
                if fname in existing_files:
                    full_path = os.path.join(self.path_to_tensors, fname)
                    samples.append((full_path, label))
                    clip_idx += 1
                else:
                    break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            video_tensor = torch.load(path)
            if video_tensor.dtype != torch.float32:
                 video_tensor = video_tensor.float()
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
            return video_tensor, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self))