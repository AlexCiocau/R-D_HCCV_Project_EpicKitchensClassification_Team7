"""
==============================================================================
DEBUG: RAM-CACHED DATASET
==============================================================================
filename: sanity_dataset.py

[PURPOSE]
A specialized dataset class for debugging I/O speed.
It uses 'ThreadPoolExecutor' to load samples into RAM (Cache) immediately.
Used by 'sanity_check.py' to test GPU throughput without disk bottlenecks.
==============================================================================
"""
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class EpicKitchensDataset(Dataset):
    """
    Optimized for VSC: Loads all data into RAM (float16) to eliminate Disk I/O.
    Performs frame selection on CPU to minimize GPU bandwidth usage.
    """
    def __init__(self, path_to_data, num_frames, testing=False, transform=None, ram_cache=True):
        self.path_to_data = path_to_data
        self.testing = testing
        self.num_frames = num_frames
        self.num_classes = 300
        self.ram_cache = ram_cache
        self.cached_data = {}

        if not self.testing:
            print("Mode: TRAINING (Participant Split)")
            path_to_annotations = path_to_data + '/annotations/splits_participants/custom_participant_train.csv'
            TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_train_tensors'
        else:
            print("Mode: VALIDATION (Participant Split)")
            path_to_annotations = path_to_data + '/annotations/splits_participants/custom_participant_val.csv'
            TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_train_tensors'

        self.path_to_tensors = os.path.expandvars(TENSOR_OUTPUT_DIR_RAW)
        
        # Fallback for VSC path if env var fails
        if not os.path.exists(self.path_to_tensors):
            fallback = "/scratch/leuven/380/vsc38040/x3d_train_tensors"
            if os.path.exists(fallback):
                self.path_to_tensors = fallback
            else:
                raise RuntimeError(f"Tensor dir not found: {self.path_to_tensors}")

        print(f"Loading annotations from: {path_to_annotations}")
        self.annotations = pd.read_csv(path_to_annotations)
        
        # Build Index
        self.samples = self._build_index()
        print(f"Dataset ready. Total samples: {len(self.samples)}")
        
        # --- OPTIMIZATION: LOAD TO RAM ---
        if self.ram_cache:
            self._cache_dataset()

    def _build_index(self):
        samples = []
        print(f"Indexing tensors from {self.path_to_tensors}...")
        existing_files = set(os.listdir(self.path_to_tensors))
        
        for _, row in self.annotations.iterrows():
            segment_uid = row['narration_id']
            label = int(row['noun_class'])
            
            # Find all clips for this segment
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

    def _load_single_item(self, idx):
        path, _ = self.samples[idx]
        try:
            # Load and keep as Float16 to save RAM (2 bytes per pixel)
            return idx, torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"Error caching {path}: {e}")
            return idx, None

    def _cache_dataset(self):
        print(f"Loading {len(self.samples)} samples into RAM using ThreadPool...")
        # Use threading to overcome I/O latency on network drives
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(self._load_single_item, range(len(self.samples))), total=len(self.samples)))
        
        for idx, tensor in results:
            if tensor is not None:
                self.cached_data[idx] = tensor
        
        print(f"Cached {len(self.cached_data)} samples in RAM.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # 1. Fetch Data
        if self.ram_cache and idx in self.cached_data:
            video_tensor = self.cached_data[idx]
        else:
            try:
                video_tensor = torch.load(path)
            except Exception as e:
                # Robustness fallback
                return self.__getitem__((idx + 1) % len(self))

        # video_tensor is shape (C, T, H, W) -> (3, 16, 256, 256)
        
        # 2. Select Frame (Spatial Optimization)
        # We only need 1 frame for ConvNeXt, not 16.
        # Slicing here prevents sending unused data to GPU.
        channels, total_frames, h, w = video_tensor.shape
        
        if self.testing:
            # Validation: Always pick center frame
            frame_idx = total_frames // 2
        else:
            # Training: Random sampling
            frame_idx = random.randint(0, total_frames - 1)
            
        # Extract specific frame: (C, H, W)
        frame = video_tensor[:, frame_idx, :, :]

        # 3. Process Dtype (Convert 16 -> 32 here)
        if frame.dtype != torch.float32:
            frame = frame.float()

        # 4. Normalize if needed (0-255 -> 0-1)
        if frame.max() > 1.0:
            frame = frame / 255.0

        return frame, label