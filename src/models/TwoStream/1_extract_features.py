"""
==============================================================================
STEP 1: FEATURE EXTRACTION (OFFLINE)
==============================================================================
filename: 1_extract_features.py

[PURPOSE]
This script optimizes training speed by pre-calculating the output of your 
heavy backbones (X3D and ConvNeXt).
1. Loads pre-trained X3D and ConvNeXt weights.
2. Runs the entire dataset through them in 'Eval' mode.
3. Saves the output vectors (2048-dim motion, 768-dim static) to disk.

[WHY?]
By saving these vectors, you can train the Fusion MLP (Step 2) in seconds 
instead of days, because you don't have to run the heavy CNNs every epoch.

[OUTPUT]
Saves .pt files to "$VSC_SCRATCH/feature_vectors_v2"
==============================================================================
"""
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import DataLoader
 
# --- IMPORTS FROM YOUR EXISTING FILES ---
# Ensure these files are in the same directory
from EpicKitchensDataset_vMulti_weighted_v2 import EpicKitchensDataset
from TwoStream_Advanced_v3 import GPUAugmentor
 
# ---------------- CONFIGURATION ----------------
# Use your pre-trained backbone weights here
X3D_CHECKPOINT = "x3d_tmpaug_model_4.pth" 
CONVNEXT_CHECKPOINT = "convnext_tiny_noun_best.pth"
 
# Path to the PIXEL tensors (from tensor_gen_x3d_fixed.py)
DATA_ROOT = './EPIC-KITCHENS'
 
# Where to save the VECTOR tensors (New output)
OUTPUT_DIR = os.path.expandvars("$VSC_SCRATCH/feature_vectors_v2")
 
BATCH_SIZE = 32
NUM_WORKERS = 8
 
# ---------------- MODEL DEFINITION ----------------
class FeatureExtractorModel(nn.Module):
    def __init__(self, x3d_ckpt, convnext_ckpt):
        super().__init__()
        # --- 1. X3D Stream ---
        print("Loading X3D...")
        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
        current_head_in = self.x3d.blocks[5].proj.in_features
        self.x3d.blocks[5].proj = nn.Linear(current_head_in, 97) # Temp head
        if os.path.exists(x3d_ckpt):
            sd = torch.load(x3d_ckpt, map_location='cpu')
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = {k: v for k, v in sd.items() if 'blocks.5.proj' not in k}
            self.x3d.load_state_dict(sd, strict=False)
        # Remove Head -> Output is 2048 dim vector
        self.x3d.blocks[5].proj = nn.Identity()
 
        # --- 2. ConvNeXt Stream ---
        print("Loading ConvNeXt...")
        self.convnext = models.convnext_tiny(weights=None)
        self.convnext.classifier[2] = nn.Linear(768, 300) # Temp head
        if os.path.exists(convnext_ckpt):
            sd = torch.load(convnext_ckpt, map_location='cpu')
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = {k: v for k, v in sd.items() if 'classifier.2' not in k}
            self.convnext.load_state_dict(sd, strict=False)
        # Remove Head -> Output is 768 dim vector
        self.convnext.classifier[2] = nn.Identity()
 
    def forward(self, x):
        # 1. Motion Features (X3D)
        feat_motion = self.x3d(x) # [B, 2048]
        # 2. Static Features (ConvNeXt) - Middle frame
        middle_idx = x.shape[2] // 2
        feat_static = self.convnext(x[:, :, middle_idx, :, :]) # [B, 768]
        return feat_motion, feat_static
 
def run_extraction(mode='train'):
    print(f"\n--- Processing {mode} set ---")
    # Setup Dataset (Use testing=True even for train to force deterministic behavior)
    is_test = True if mode == 'val' else False
    # We pass transform=None because GPUAugmentor handles it
    dataset = EpicKitchensDataset(DATA_ROOT, 16, testing=is_test, transform=None)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)
    # Create Output Dir
    mode_dir = os.path.join(OUTPUT_DIR, mode)
    os.makedirs(mode_dir, exist_ok=True)
    # Models to GPU
    model.eval()
    augmentor.eval() # IMPORTANT: Center Crop only. No random flipping.
    with torch.no_grad():
        for i, (video_batch, labels_dict) in tqdm(enumerate(loader), total=len(loader)):
            video_batch = video_batch.to(device)
            # Apply Normalization/Resize
            video_batch = augmentor(video_batch) 
            # Extract Features
            feat_motion, feat_static = model(video_batch)
            # Save Batch
            chunk_data = {
                'feat_motion': feat_motion.cpu().half(), # FP16 to save space
                'feat_static': feat_static.cpu().half(),
                'verb': labels_dict['verb'],
                'noun': labels_dict['noun']
            }
            torch.save(chunk_data, os.path.join(mode_dir, f"batch_{i}.pt"))
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    # Init Model
    model = FeatureExtractorModel(X3D_CHECKPOINT, CONVNEXT_CHECKPOINT).to(device)
    model.eval() # Strictly Eval
    # Init Augmentor
    # We use training=False to ensure we get clean Center Crops for consistent features
    augmentor = GPUAugmentor(training=False).to(device)
    # Run
    run_extraction(mode='train')
    run_extraction(mode='val')
    print("Feature Extraction Complete! Now run the training script.")