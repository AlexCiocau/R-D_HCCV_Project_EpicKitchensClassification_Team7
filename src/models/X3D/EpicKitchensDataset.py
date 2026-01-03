"""
==============================================================================
STEP 0: THE DATASET (FOUNDATION)
==============================================================================
filename: EpicKitchensDataset.py

[PURPOSE]
This file defines the custom PyTorch Dataset class. It handles:
1. Loading pre-computed tensors from the disk (created by tensor_gen scripts).
2. Parsing the CSV annotations (train/val splits).
3. Applying Data Augmentation (ColorJitter, RandomCrop, HorizontalFlip).

[USAGE]
You do not run this file directly. 
It is imported by 'X3D.py' and 'x3D_fine_tunning.py' to load data.
==============================================================================
"""

from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as T  # <--- Important alias

class EpicKitchensDataset(Dataset):
    """
    A custom Dataset class for loading Epic Kitchens data.
    """
    def __init__(self, path_to_data, num_frames, testing=False, transform=None):
        self.path_to_data = path_to_data
        self.testing = testing
        self.num_frames = num_frames
        
        # X3D Standard Normalization Values
        self.mean = [0.43216, 0.394666, 0.37645]
        self.std = [0.22803, 0.22145, 0.216989]

        if not self.testing:
            print("Mode: TRAINING (Online Augmentation Enabled)")
            path_to_annotations = path_to_data + '/annotations/custom_train_80.csv'
            TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_train_tensors'
            
            # --- FIX: Initialize ColorJitter here ---
            self.color_jitter = T.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.05
            )
            # ----------------------------------------
        else:
            print("Mode: VALIDATION")
            path_to_annotations = path_to_data + '/annotations/custom_val_20.csv'
            TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_val_tensors'

        self.path_to_tensors = os.path.expandvars(TENSOR_OUTPUT_DIR_RAW)
        
        print(f"Loading annotations from: {path_to_annotations}")
        print(f"Loading tensors from: {self.path_to_tensors}")
        
        all_annotations = pd.read_csv(path_to_annotations)
        self.annotations = self.filter_annotations(all_annotations)
        
        if len(self.annotations) > 0:
            self.num_classes = self.annotations['verb_class'].max() + 1
        else:
            self.num_classes = 0
            
        print(f"Found {self.num_classes} unique verb classes.")
        
        # Build index
        self.samples = self._build_index()
        print(f"Dataset ready. Total samples: {len(self.samples)}")

    def filter_annotations(self, all_annotations):
        # Basic check to ensure we don't crash on empty CSVs
        if all_annotations.empty:
            return pd.DataFrame()
        return all_annotations

    def _build_index(self):
        samples = []
        if not os.path.exists(self.path_to_tensors):
            raise RuntimeError(f"Tensor dir not found: {self.path_to_tensors}")
            
        print("Indexing existing tensors...")
        # Reading directory once is much faster than checking os.path.exists per file
        existing_files = set(os.listdir(self.path_to_tensors))
        
        for _, row in self.annotations.iterrows():
            segment_uid = row['narration_id']
            label = int(row['verb_class'])
            
            # Look for clip0, clip1, ...
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

    def apply_video_transforms(self, video_tensor):
        """
        Input: (C, T, H, W) tensor
        Output: (C, T, H, W) tensor, resized/cropped/normalized
        """
        # 1. Standardize type
        if video_tensor.dtype != torch.float32:
            video_tensor = video_tensor.float()

        C, num_frames, H, W = video_tensor.shape
        
        # TARGET SIZE: 160 for X3D-XS, 224 for X3D-M
        target_size = 224 

        # ---------------- TRAINING AUGMENTATION ----------------
        if not self.testing:
            
            # --- A. MANUAL SYNCHRONIZED COLOR JITTER ---
            # We explicitly sample random factors within the ranges we defined in __init__
            # Brightness: [max(0, 1-0.2), 1+0.2]
            # Contrast:   [max(0, 1-0.2), 1+0.2]
            # Saturation: [max(0, 1-0.2), 1+0.2]
            # Hue:        [-0.05, 0.05]
            
            # 1. Generate random factors
            bright_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            contrast_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            sat_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            hue_factor = torch.empty(1).uniform_(-0.05, 0.05).item()
            
            # 2. Define the order (usually random, but fixed order is fine for stability)
            # We perform adjustments on (T, C, H, W) to treat frames as a batch
            video_tensor = video_tensor.permute(1, 0, 2, 3) # (T, C, H, W)
            
            # Apply consistently to the whole batch (all frames get same adjustment)
            # Note: We must clamp to ensure valid image ranges [0,1] or [0,255] if needed, 
            # but TF.adjust usually handles floats fine.
            video_tensor = TF.adjust_brightness(video_tensor, bright_factor)
            video_tensor = TF.adjust_contrast(video_tensor, contrast_factor)
            video_tensor = TF.adjust_saturation(video_tensor, sat_factor)
            video_tensor = TF.adjust_hue(video_tensor, hue_factor)
            
            video_tensor = video_tensor.permute(1, 0, 2, 3) # Back to (C, T, H, W)

            # --- B. SPATIAL AUGMENTATION (Crop & Flip) ---
            i, j, h, w = T.RandomCrop.get_params(
                torch.zeros(1, H, W), output_size=(target_size, target_size)
            )
            do_flip = torch.rand(1) < 0.5
            
            video_tensor = TF.crop(video_tensor, i, j, h, w)
            if do_flip:
                video_tensor = TF.hflip(video_tensor)

        # ---------------- VALIDATION (DETERMINISTIC) ----------------
        else:
            video_tensor = TF.center_crop(video_tensor, (target_size, target_size))

        # 3. Normalization
        video_tensor = video_tensor.permute(1, 0, 2, 3) 
        video_tensor = TF.normalize(video_tensor, mean=self.mean, std=self.std)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            # Load tensor
            video_tensor = torch.load(path)
            
            # Apply transforms
            video_tensor = self.apply_video_transforms(video_tensor)

            return video_tensor, label

        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Robustness: try next sample if this one is corrupted
            return self.__getitem__((idx + 1) % len(self))