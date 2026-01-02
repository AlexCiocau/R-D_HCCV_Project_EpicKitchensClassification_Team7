from torch.utils.data import Dataset
import pandas as pd
import torch
import os

class EpicKitchensDataset(Dataset):
    """
    A custom Dataset class for loading Epic Kitchens data.
    Configured for: NOUNS (Retraining)
    """
    def __init__(self, path_to_data, num_frames, testing=False, transform=None):
        self.path_to_data = path_to_data
        self.testing = testing
        self.num_frames = num_frames
        
        # Hardcoded for EPIC-Kitchens Nouns to prevent dynamic counting errors
        self.num_classes = 300

        if not self.testing:
            print("Mode: TRAINING (Using participant split: custom_participant_train.csv)")
            # Make sure this CSV exists from your previous split step
            path_to_annotations = path_to_data + '/annotations/splits_participants/custom_participant_train.csv'
            TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_train_tensors'
        else:
            print("Mode: VALIDATION (Using participant split: custom_participant_val.csv)")
            path_to_annotations = path_to_data + '/annotations/splits_participants/custom_participant_val.csv'
            TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_train_tensors'

        self.path_to_tensors = os.path.expandvars(TENSOR_OUTPUT_DIR_RAW)
        
        print(f"Loading annotations from: {path_to_annotations}")
        
        all_annotations = pd.read_csv(path_to_annotations)
        self.annotations = self.filter_annotations(all_annotations)
        
        print(f"Found {self.num_classes} noun classes (Fixed).")
        
        # Build index
        self.samples = self._build_index()
        print(f"Dataset ready. Total samples: {len(self.samples)}")

    def filter_annotations(self, all_annotations):
        if all_annotations.empty:
            return pd.DataFrame()
        return all_annotations

    def _build_index(self):
        samples = []
        if not os.path.exists(self.path_to_tensors):
            # Fallback for hardcoded path if env var fails
            fallback = "/scratch/leuven/380/vsc38040/x3d_train_tensors"
            if os.path.exists(fallback):
                self.path_to_tensors = fallback
            else:
                raise RuntimeError(f"Tensor dir not found: {self.path_to_tensors}")
            
        print(f"Indexing tensors from {self.path_to_tensors}...")
        existing_files = set(os.listdir(self.path_to_tensors))
        
        for _, row in self.annotations.iterrows():
            segment_uid = row['narration_id']
            label = int(row['noun_class'])
            
            # Find all clips corresponding to this segment
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
            # Load raw tensor
            video_tensor = torch.load(path)
            
            # Ensure Float32
            if video_tensor.dtype != torch.float32:
                 video_tensor = video_tensor.float()
            
            # Safety check for 0-255 vs 0-1
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0

            # Return raw tensor (C, T, H, W) and label
            return video_tensor, label
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Robustness: try next sample
            return self.__getitem__((idx + 1) % len(self))