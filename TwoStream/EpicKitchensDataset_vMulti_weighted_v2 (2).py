from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as T


##############################################################################################
##############################################################################################
# ---------------------------------- VERB AND NOUN VERSION ----------------------------------- 
##############################################################################################
##############################################################################################


class EpicKitchensDataset(Dataset):
    """
    A custom Dataset class for loading Epic Kitchens data.
    """
    def __init__(self, path_to_data, num_frames, testing=False, transform=None):
        self.path_to_data = path_to_data
        self.testing = testing
        self.num_frames = num_frames
        
        # self.mean = [0.43216, 0.394666, 0.37645]
        # self.std = [0.22803, 0.22145, 0.216989]

        # specific to X3D models from PyTorchVideo
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]

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
            self.num_verb_classes = self.annotations['verb_class'].max() + 1
            # --- NEW: Get Noun Classes ---
            self.num_noun_classes = self.annotations['noun_class'].max() + 1
        else:
            self.num_verb_classes = 0
            self.num_noun_classes = 0
            
        print(f"Found {self.num_verb_classes} verbs and {self.num_noun_classes} nouns.")
        
        self.samples = self._build_index()

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
        existing_files = set(os.listdir(self.path_to_tensors))
        
        for _, row in self.annotations.iterrows():
            segment_uid = row['narration_id']
            verb_label = int(row['verb_class'])
            # --- NEW: Read Noun Label ---
            noun_label = int(row['noun_class'])
            
            clip_idx = 0
            while True:
                fname = f"{segment_uid}_clip{clip_idx}.pt"
                if fname in existing_files:
                    full_path = os.path.join(self.path_to_tensors, fname)
                    # --- NEW: Store tuple of labels ---
                    samples.append((full_path, (verb_label, noun_label)))
                    clip_idx += 1
                else:
                    break
        return samples

    def __len__(self):
        return len(self.samples)
    
    def get_sample_weights(self, balance_by='hybrid'):
        """
        Calculates a weight for every sample in the dataset 
        to balance the training batches.
        """
        import numpy as np
        
        # 1. Gather all labels from the index
        # self.samples is a list of tuples: (path, (verb, noun))
        all_verbs = [s[1][0] for s in self.samples]
        all_nouns = [s[1][1] for s in self.samples]
        
        total_samples = len(self.samples)
        
        # # 2. Choose what to balance
        # if balance_by == 'verb':
        #     labels = all_verbs
        #     num_classes = self.num_verb_classes
        # elif balance_by == 'noun':
        #     labels = all_nouns
        #     num_classes = self.num_noun_classes
        
        # # 3. Count frequency of each class in the EXISTING tensors
        # # We use bincount for speed (labels must be ints)
        # class_counts = np.bincount(labels, minlength=num_classes)
        
        # # Avoid division by zero for classes that might be missing entirely
        # class_counts[class_counts == 0] = 1 
        
        # # 4. Calculate Weight per Class (Inverse Frequency)
        # # weight = 1.0 / count
        # class_weights = 1.0 / class_counts
        
        # # 5. Assign weight to each sample
        # sample_weights = [class_weights[l] for l in labels]
        
        # return torch.DoubleTensor(sample_weights)

        # Calculate Verb Weights
        verb_counts = np.bincount(all_verbs, minlength=self.num_verb_classes)
        verb_counts[verb_counts == 0] = 1
        verb_weights = 1.0 / verb_counts
        sample_weights_v = np.array([verb_weights[v] for v in all_verbs])

        # Calculate Noun Weights
        noun_counts = np.bincount(all_nouns, minlength=self.num_noun_classes)
        noun_counts[noun_counts == 0] = 1
        noun_weights = 1.0 / noun_counts
        sample_weights_n = np.array([noun_weights[n] for n in all_nouns])

        # COMBINE THEM
        if balance_by == 'verb':
            final_weights = sample_weights_v
        elif balance_by == 'noun':
            final_weights = sample_weights_n
        elif balance_by == 'hybrid':
            # We average the "rarity" of the verb and the noun.
            # If either is rare, the sample gets a high weight.
            final_weights = sample_weights_v + sample_weights_n
            
        return torch.DoubleTensor(final_weights)

    # def apply_video_transforms(self, video_tensor):
    #     """
    #     Input: (C, T, H, W) tensor
    #     Output: (C, T, H, W) tensor, resized/cropped/normalized
    #     """
    #     # 1. Standardize type
    #     if video_tensor.dtype != torch.float32:
    #         video_tensor = video_tensor.float()


    #     ###########################################################################
    #     # --- CRITICAL SAFETY CHECK ---
    #     # If the max value is > 1.0, we assume it's 0-255 and scale it down.
    #     if video_tensor.max() > 1.0:
    #         video_tensor = video_tensor / 255.0
    #     # -----------------------------
    #     ############################################################################


    #     C, num_frames, H, W = video_tensor.shape
        
    #     # TARGET SIZE: 160 for X3D-XS, 224 for X3D-M
    #     target_size = 224 

    #     # ---------------- TRAINING AUGMENTATION ----------------
    #     if not self.testing:
            
    #         # --- A. MANUAL SYNCHRONIZED COLOR JITTER ---
    #         # We explicitly sample random factors within the ranges we defined in __init__
    #         # Brightness: [max(0, 1-0.2), 1+0.2]
    #         # Contrast:   [max(0, 1-0.2), 1+0.2]
    #         # Saturation: [max(0, 1-0.2), 1+0.2]
    #         # Hue:        [-0.05, 0.05]
            
    #         # 1. Generate random factors
    #         bright_factor = torch.empty(1).uniform_(0.8, 1.2).item()
    #         contrast_factor = torch.empty(1).uniform_(0.8, 1.2).item()
    #         sat_factor = torch.empty(1).uniform_(0.8, 1.2).item()
    #         hue_factor = torch.empty(1).uniform_(-0.05, 0.05).item()
            
    #         # 2. Define the order (usually random, but fixed order is fine for stability)
    #         # We perform adjustments on (T, C, H, W) to treat frames as a batch
    #         video_tensor = video_tensor.permute(1, 0, 2, 3) # (T, C, H, W)
            
    #         # Apply consistently to the whole batch (all frames get same adjustment)
    #         # Note: We must clamp to ensure valid image ranges [0,1] or [0,255] if needed, 
    #         # but TF.adjust usually handles floats fine.
    #         video_tensor = TF.adjust_brightness(video_tensor, bright_factor)
    #         video_tensor = TF.adjust_contrast(video_tensor, contrast_factor)
    #         video_tensor = TF.adjust_saturation(video_tensor, sat_factor)
    #         video_tensor = TF.adjust_hue(video_tensor, hue_factor)
            
    #         video_tensor = video_tensor.permute(1, 0, 2, 3) # Back to (C, T, H, W)

    #         # --- B. SPATIAL AUGMENTATION (Crop & Flip) ---
    #         i, j, h, w = T.RandomCrop.get_params(
    #             torch.zeros(1, H, W), output_size=(target_size, target_size)
    #         )
    #         do_flip = torch.rand(1) < 0.5
            
    #         video_tensor = TF.crop(video_tensor, i, j, h, w)
    #         if do_flip:
    #             video_tensor = TF.hflip(video_tensor)

    #     # ---------------- VALIDATION (DETERMINISTIC) ----------------
    #     else:
    #         video_tensor = TF.center_crop(video_tensor, (target_size, target_size))

    #     # 3. Normalization
    #     video_tensor = video_tensor.permute(1, 0, 2, 3) 
    #     video_tensor = TF.normalize(video_tensor, mean=self.mean, std=self.std)
    #     video_tensor = video_tensor.permute(1, 0, 2, 3)

    #     return video_tensor

    def apply_video_transforms(self, video_tensor):
        """
        Input: (C, T, H, W) tensor
        Output: (C, T, H, W) tensor, raw values [0-1]
        """
        # 1. Standardize type
        if video_tensor.dtype != torch.float32:
            video_tensor = video_tensor.float()

        # 2. Safety Scaling: Ensure range is [0, 1]
        # If max > 1.0, it means it's 0-255 pixels. Divide by 255.
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0

        # 3. Validation Center Crop (Optional, but good for consistency)
        # We only do simple cropping here. NO NORMALIZATION.
        if self.testing:
            video_tensor = TF.center_crop(video_tensor, (224, 224))
        
        # REMOVED: TF.normalize(...) <--- THIS WAS THE BUG
        
        return video_tensor

    def __getitem__(self, idx):
        # 1. Bounds Check (Prevents IndexError)
        if idx >= len(self.samples):
            print(f"Warning: Index {idx} out of bounds (Len: {len(self.samples)}). Resetting to 0.")
            idx = 0
            
        path, (verb_label, noun_label) = self.samples[idx]
        
        try:
            # 2. Load tensor
            video_tensor = torch.load(path)
            
            # 3. Standardize type
            if video_tensor.dtype != torch.float32:
                video_tensor = video_tensor.float()

            # 4. Safety Scaling (0-255 -> 0-1)
            # This is the "Double Norm" prevention we discussed earlier
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
            
            # 5. Return
            return video_tensor, {"verb": verb_label, "noun": noun_label}

        except Exception as e:
            # 6. Robust Error Handling
            print(f"!!! Error loading sample at index {idx}: {path}")
            print(f"!!! Exception: {e}")
            
            # Instead of idx+1 (which can error again or overflow), pick a random safe index
            # This effectively "skips" the bad file without crashing the batch
            new_idx = torch.randint(0, len(self.samples), (1,)).item()
            return self.__getitem__(new_idx)