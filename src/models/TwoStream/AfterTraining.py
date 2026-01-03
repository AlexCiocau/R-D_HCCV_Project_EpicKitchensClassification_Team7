"""
==============================================================================
STEP 4: END-TO-END FINE-TUNING
==============================================================================
filename: AfterTraining.py

[PURPOSE]
This script performs the final polish on the assembled model.
1. Loads the full 'TwoStreamModel' (assembled in Step 3).
2. Unfreezes the backbones (X3D and ConvNeXt).
3. Trains the entire system with a very low learning rate (1e-6).
4. Uses 'FocalLoss' to focus on hard/rare classes.

[USAGE]
Run this to squeeze the final few % of accuracy out of the model.
==============================================================================
"""
from EpicKitchensDataset_vMulti_weighted_v2 import EpicKitchensDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import wandb
import os
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from early_stopping import EarlyStopping

# --- 1. SPEED OPTIMIZATION ---
torch.set_float32_matmul_precision('high')

try:
    from torchvision.transforms import v2
except ImportError:
    print("WARNING: torchvision.transforms.v2 not found. Please update torchvision!")
    import torchvision.transforms as v2

# ------------------------------- HELPER: WEIGHT CLEANER ---------------------------
def clean_state_dict(state_dict):
    """Removes 'module.' prefix if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, label_smoothing=0.1):
        super().__init__() 
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ------------------------------- GPU AUGMENTOR (The Normalizer) ------------------------------
class GPUAugmentor(nn.Module):
    def __init__(self, training=True):
        super().__init__()
        self.training = training
        
        # WE USE IMAGENET STATS (Matches 1_extract_features.py)
        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
        
        if training:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True), 
                v2.RandomCrop(224),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ])
        else:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
            ])

    def forward(self, x):
        # Input: Raw [0-1] tensor from Dataset V2
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x = self.geo_transforms(x)
        x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)
        
        # Apply Normalization HERE (Single Norm)
        x = (x - self.mean) / self.std
        return x

# ------------------------------- TWO-STREAM MODEL ------------------------------
class TwoStreamModel(nn.Module):
    def __init__(self, num_verbs, num_nouns, x3d_checkpoint, convnext_checkpoint, dropout_rate=0.5):
        super().__init__()
        
        # --- STREAM 1: X3D-M ---
        print(f"Loading X3D Stream...")
        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
        
        # 1. Structure Match
        current_head_in = self.x3d.blocks[5].proj.in_features
        self.x3d.blocks[5].proj = nn.Linear(current_head_in, 97) # Temp match
        
        # 2. Load Weights (Robust)
        if os.path.exists(x3d_checkpoint):
            print(f"Loading X3D from {x3d_checkpoint}")
            sd = torch.load(x3d_checkpoint, map_location='cpu')
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = clean_state_dict(sd)
            sd = {k: v for k, v in sd.items() if 'blocks.5.proj' not in k}
            self.x3d.load_state_dict(sd, strict=False)
        
        # 3. Strip Head
        self.x3d.blocks[5].proj = nn.Identity()

        # --- STREAM 2: ConvNeXt-Tiny ---
        print(f"Loading ConvNeXt Stream...")
        self.convnext = models.convnext_tiny(weights=None) 
        
        # 1. Structure Match
        in_features_cx = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features_cx, 300))
        
        # 2. Load Weights (Robust)
        if os.path.exists(convnext_checkpoint):
            print(f"Loading ConvNeXt from {convnext_checkpoint}")
            sd = torch.load(convnext_checkpoint, map_location='cpu')
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = clean_state_dict(sd)
            sd = {k: v for k, v in sd.items() if 'classifier.2' not in k}
            self.convnext.load_state_dict(sd, strict=False)
            
        # 3. Strip Head
        self.convnext.classifier[2] = nn.Identity()

        # --- FUSION MLP (Matches 2_train_mlp_fast.py) ---
        self.project_x3d = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.project_cx = nn.Sequential(
            nn.Linear(768, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.verb_fc = nn.Linear(1024, num_verbs)
        self.noun_fc = nn.Linear(1024, num_nouns)

    def freeze_backbones(self):
        for param in self.x3d.parameters(): param.requires_grad = False
        for param in self.convnext.parameters(): param.requires_grad = False
        # Unfreeze MLP
        for m in [self.project_x3d, self.project_cx, self.fusion_mlp, self.verb_fc, self.noun_fc]:
            for param in m.parameters(): param.requires_grad = True

    def unfreeze_backbones(self):
        for param in self.x3d.parameters(): param.requires_grad = True
        for param in self.convnext.parameters(): param.requires_grad = True

    def forward(self, x):
        feat_motion = self.x3d(x) 
        
        middle_idx = x.shape[2] // 2
        feat_static = self.convnext(x[:, :, middle_idx, :, :]) 
        
        p_motion = self.project_x3d(feat_motion)
        p_static = self.project_cx(feat_static)
        
        combined = torch.cat((p_motion, p_static), dim=1) 
        fused = self.fusion_mlp(combined)                 
        
        return self.verb_fc(fused), self.noun_fc(fused)

# ------------------------------- EVALUATION --------------------------------
def evaluate_model(model, dataloader, criterion, device, augmentor):
    model.eval()
    augmentor.eval()
    total_loss = 0.0
    correct_verbs = 0; correct_nouns = 0; correct_actions = 0
    total_samples = 0
    
    with torch.no_grad():
        for video_batch, labels_dict in tqdm(dataloader, desc="Evaluating", leave=False):
            verb_labels = labels_dict['verb'].to(device)
            noun_labels = labels_dict['noun'].to(device)
            video_batch = video_batch.to(device, non_blocking=True)
            
            video_batch = augmentor(video_batch) # Apply Normalization

            verb_logits, noun_logits = model(video_batch)
            
            loss = criterion(verb_logits, verb_labels) + criterion(noun_logits, noun_labels)
            total_loss += loss.item()
            
            _, verb_preds = torch.max(verb_logits, 1)
            _, noun_preds = torch.max(noun_logits, 1)
            
            total_samples += verb_labels.size(0)
            verb_hits = (verb_preds == verb_labels)
            noun_hits = (noun_preds == noun_labels)
            
            correct_verbs += verb_hits.sum().item()
            correct_nouns += noun_hits.sum().item()
            correct_actions += (verb_hits & noun_hits).sum().item()

    return (total_loss/len(dataloader), 
            100*correct_verbs/total_samples, 
            100*correct_nouns/total_samples, 
            100*correct_actions/total_samples)

if __name__ == '__main__':
    # ------------------------------- CONFIGURATION ----------------------------------
    X3D_CHECKPOINT = "x3d_tmpaug_model_4.pth" 
    CONVNEXT_CHECKPOINT = "convnext_tiny_noun_best.pth"
    MODEL_SAVE_PATH = "twostream_ultimate_best.pth"
    MLP_WEIGHTS = "mlp_best_weights.pth"
    
    BATCH_SIZE = 32  
    NUM_WORKERS = 8
    NUM_FRAMES = 16
    
    # ------------------------------- DATASET (V2 IMPORT) ---------------------------
    # Using V2 which returns RAW 0-1 tensors (No internal normalization)
    train_dataset = EpicKitchensDataset('./EPIC-KITCHENS', NUM_FRAMES, testing=False, transform=None)
    val_dataset = EpicKitchensDataset('./EPIC-KITCHENS', NUM_FRAMES, testing=True, transform=None)
    
    NUM_VERB_CLASSES = train_dataset.num_verb_classes
    NUM_NOUN_CLASSES = 300

    try:
        sample_weights = train_dataset.get_sample_weights(balance_by='hybrid')
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
        print("Using Weighted Sampler.")
    except:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, 
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # ---------------------------- MODEL SETUP ---------------------------------------
    print("Initializing Two-Stream Model...")
    model = TwoStreamModel(NUM_VERB_CLASSES, NUM_NOUN_CLASSES, X3D_CHECKPOINT, CONVNEXT_CHECKPOINT).to(device)
    
    # --- LOAD PRE-TRAINED MLP WEIGHTS ---
    if os.path.exists(MLP_WEIGHTS):
        print(f"Loading pre-trained MLP weights from {MLP_WEIGHTS}...")
        mlp_sd = torch.load(MLP_WEIGHTS, map_location='cpu')
        mlp_sd = clean_state_dict(mlp_sd)
        
        # This will load the MLP heads. It will skip X3D/ConvNeXt keys (missing_keys is normal)
        # But 'unexpected_keys' must be empty.
        missing, unexpected = model.load_state_dict(mlp_sd, strict=False)
        
        if len(unexpected) > 0:
            print(f"[CRITICAL ERROR] MLP keys mismatch! Check names. Unexpected: {unexpected}")
        else:
            print(f"MLP Weights successfully integrated.")
    else:
        print(f"WARNING: {MLP_WEIGHTS} not found! Starting MLP from scratch.")

    # GPU Augmentors (Standard ImageNet Normalization)
    train_augmentor = GPUAugmentor(training=True).to(device)
    val_augmentor = GPUAugmentor(training=False).to(device)

    wandb.init(project="R&D-Project", config={"type": "TwoStream_Fixed_Pipeline"})
    
    # ---------------------------- SANITY CHECK -------------------------
    print("\n[SANITY CHECK] Running evaluation...")
    model.eval()
    criterion = FocalLoss(gamma=2.0)
    
    # This should now match your MLP script results (~53% Verb)
    val_loss, acc_v, acc_n, acc_act = evaluate_model(model, val_loader, criterion, device, val_augmentor)
    print(f"Sanity Check | Loss: {val_loss:.4f} | V_Acc: {acc_v:.2f}% | N_Acc: {acc_n:.2f}% | ACT_Acc: {acc_act:.2f}%")
    
    # ---------------------------- TRAINING LOOP -------------------------
    print("\n[PHASE 1] Freezing Backbones. Training Fusion Head (1 Epoch)...")
    model.freeze_backbones()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    
    # ---------------------------- PHASE 2: FINE TUNE ----------------------------
    print("\n[PHASE 2] Unfreezing Backbones. Fine-tuning...")
    model.unfreeze_backbones()
    
    NUM_EPOCHS_FT = 20
    FINE_TUNE_LR = 1e-6 
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=0.05)
    
    warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=2)
    cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_FT - 2, eta_min=1e-6)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])
    
    early_stopper = EarlyStopping(patience=5, verbose=True, path=MODEL_SAVE_PATH)
    
    global_step = 0
    for epoch in range(NUM_EPOCHS_FT):
        model.train()
        train_augmentor.train()
        loop = tqdm(train_loader, desc=f"Fine-Tune {epoch+1}/{NUM_EPOCHS_FT}", leave=False)
        for video_batch, labels_dict in loop:
            video_batch = video_batch.to(device, non_blocking=True)
            v_labels = labels_dict['verb'].to(device)
            n_labels = labels_dict['noun'].to(device)
            
            with torch.no_grad(): video_batch = train_augmentor(video_batch)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                v_out, n_out = model(video_batch)
                loss = criterion(v_out, v_labels) + criterion(n_out, n_labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            wandb.log({"train/batch_loss": loss.item(), "global_step": global_step})
            global_step += 1

        val_loss, acc_v, acc_n, acc_act = evaluate_model(model, val_loader, criterion, device, val_augmentor)
        scheduler.step()
        print(f"Phase 2 | Epoch {epoch+1} | Loss: {val_loss:.4f} | V_Acc: {acc_v:.2f}% | N_Acc: {acc_n:.2f}% | ACT_Acc: {acc_act:.2f}%")
        wandb.log({"val/act_acc": acc_act, "phase": 2, "epoch": epoch+1})
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping.")
            break