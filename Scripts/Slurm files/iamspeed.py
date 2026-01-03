"""
==============================================================================
VARIANT: HIGH-PERFORMANCE TRAINING ("IAMSPEED")
==============================================================================
filename: iamspeed.py

[PURPOSE]
An optimized training script designed for maximum accuracy and speed.
KEY FEATURES:
1. Consensus Evaluation: Uses 'FiveCrop' + Temporal Averaging (Multiple Views)
   during validation for robust metrics.
2. Aggressive Augmentation: Uses 'RandomResizedCrop(0.4, 1.0)' (Zoom in) 
   to force the model to look at object details.
3. Layer-Wise LR Decay: Lowers learning rate for early layers.

[USAGE]
Use this for your best/final training runs.
==============================================================================
"""
from EpicKitchensDataset_vNouns_participant import EpicKitchensDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import wandb
import os
import torchvision.models as models
from early_stopping import EarlyStopping

# --- IMPORT V2 TRANSFORMS ---
try:
    from torchvision.transforms import v2
except ImportError:
    print("WARNING: torchvision.transforms.v2 not found. Please update torchvision >= 0.16")
    import torchvision.transforms as v2

# ------------------------------- HELPER: LAYER-WISE LR DECAY ----------------------
def get_layer_wise_groups(model, base_lr, weight_decay, layer_decay=0.8):
    """
    Assigns strictly lower LR to earlier layers to preserve pre-trained features.
    """
    param_groups = {}
    num_layers = 9 
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if name.startswith("classifier"):
            layer_id = num_layers - 1 
        elif name.startswith("features"):
            try:
                block_idx = int(name.split('.')[1])
                layer_id = block_idx
            except:
                layer_id = 0
        else:
            layer_id = 0 

        scale = layer_decay ** (num_layers - layer_id - 1)
        
        group_name = f"layer_{layer_id}"
        if group_name not in param_groups:
            param_groups[group_name] = {
                "params": [], 
                "lr": base_lr * scale, 
                "weight_decay": weight_decay
            }
        
        param_groups[group_name]["params"].append(param)
        
    return list(param_groups.values())

# ------------------------------- GPU AUGMENTOR (IMPROVED) ------------------------
class GPUAugmentor(nn.Module):
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.num_classes = num_classes
        
        # ImageNet Normalization Stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if training:
            # 1. AGGRESSIVE TRAINING AUGMENTATION
            # Scale changed from (0.8, 1.0) to (0.4, 1.0)
            # This simulates "zooming in" on objects, reducing background bias.
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True), 
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25),
            ])
            
            # 2. MIXUP / CUTMIX
            self.mixup_cutmix = v2.RandomChoice([
                v2.MixUp(num_classes=num_classes, alpha=0.8),
                v2.CutMix(num_classes=num_classes, alpha=1.0)
            ])
        else:
            # VALIDATION AUGMENTATION (Standard Center Crop for single-view check)
            # Note: The Consensus Evaluator uses its own cropping logic (ThreeCrop)
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
            ])

    def forward(self, x, labels=None):
        # Apply Geometry/Color
        if self.training:
            x = self.geo_transforms(x)
        else:
            # For validation in this specific class, we do standard transforms.
            # (The consensus function below handles its own crops manually)
            x = self.geo_transforms(x)
        
        # Normalize
        x = (x - self.mean) / self.std
        
        # Apply Mixup/Cutmix (Training Only)
        if self.training and labels is not None:
            x, labels = self.mixup_cutmix(x, labels)
            
        return x, labels

# ------------------------------- CONSENSUS EVALUATION (NEW) ------------------------
def evaluate_model_consensus(model, dataloader, criterion, device, num_clips=2):
    """
    Performs Multi-View Testing using FiveCrop:
    1. Selects 'num_clips' temporal frames.
    2. Applies FiveCrop (TL, TR, BL, BR, Center) to each frame.
    3. Total Views = num_clips * 5.
    4. Averages logits for robust prediction.
    """
    model.eval() 
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # FIX: Use FiveCrop instead of ThreeCrop
    tta_transform = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.FiveCrop(224), # Returns tuple of 5 tensors
    ])
    
    # Normalization (must apply manually to the crop stack)
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad(): 
        for video_batch, labels_batch in tqdm(dataloader, desc="Consensus Eval", leave=False):
            
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            B, C, T, H, W = video_batch.shape
            
            # --- 1. TEMPORAL SAMPLING ---
            # Pick 'num_clips' frames evenly spaced
            indices = torch.linspace(T//4, 3*T//4, steps=num_clips).long()
            selected_frames = video_batch[:, :, indices, :, :] # [B, C, num_clips, H, W]
            
            # Flatten Batch and Time -> [B * num_clips, C, H, W]
            flat_frames = selected_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            
            # --- 2. SPATIAL CROPS (FiveCrop) ---
            # Returns tuple of 5 tensors: (TL, TR, BL, BR, Center)
            crops = tta_transform(flat_frames) 
            
            # Stack crops -> [B*num_clips, 5, C, 224, 224]
            # Flatten to -> [B*num_clips*5, C, 224, 224]
            input_tensor = torch.stack(crops, dim=1).view(-1, C, 224, 224)
            
            # --- 3. NORMALIZE ---
            input_tensor = (input_tensor - norm_mean) / norm_std
            
            # --- 4. INFERENCE ---
            logits = model(input_tensor) # [B*num_clips*5, Num_Classes]
            
            # --- 5. AGGREGATE (Consensus) ---
            # Reshape back to [B, Total_Views, Num_Classes]
            # Note: 5 crops * num_clips
            logits = logits.view(B, num_clips * 5, -1)
            
            # Average over all views
            consensus_logits = logits.mean(dim=1) 
            
            # Calculate Loss & Accuracy
            loss = criterion(consensus_logits, labels_batch)
            total_loss += loss.item()
            
            _, predictions = torch.max(consensus_logits, 1)
            total_samples += labels_batch.size(0)
            total_correct += (predictions == labels_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy

if __name__ == '__main__':
    # ------------------------------- CONFIGURATION --------------------------------
    torch.backends.cudnn.benchmark = True
    
    # Config
    BASE_LR = 5e-5 
    LAYER_DECAY = 0.8
    # Lowered Batch Size slightly to account for Consensus Eval memory usage if needed
    BATCH_SIZE = 32      
    NUM_WORKERS = 8  
    NUM_EPOCHS = 50
    NUM_FRAMES = 16
    WARMUP_EPOCHS = 5
    
    MODEL_SAVE_PATH = "convnext_tiny_noun_participants_consensus.pth"

    # ------------------------------- DATASET ----------------------------------------
    print("Initializing Datasets...")
    train_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS', num_frames=NUM_FRAMES, testing=False, transform=None 
    )
    NUM_NOUN_CLASSES = train_dataset.num_classes 
    
    # Sampler Logic
    sampler = None
    shuffle = True
    try:
         if hasattr(train_dataset, 'get_sample_weights'):
             print("Calculating weights for sampler...")
             sample_weights = train_dataset.get_sample_weights(balance_by='noun') 
             sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
             shuffle = False
         else:
             print("No 'get_sample_weights' found. Using standard Shuffle.")
    except:
         sampler = None
         shuffle = True

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )

    val_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS', num_frames=16, testing=True, transform=None
    )
    # Note: Validation batch size is same as train, but effectively 6x larger in memory during eval due to crops.
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------- MODEL ---------------------------------------------
    print("Loading ConvNeXt-Tiny...")
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_features, NUM_NOUN_CLASSES)
    )
    model = model.to(device)
    
    # ---------------------------- AUGMENTOR -----------------------------------------
    train_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=True).to(device)
    
    # ---------------------------- OPTIMIZER ----------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"Configuring Layer-Wise LR Decay (Base: {BASE_LR})")
    param_groups = get_layer_wise_groups(model, base_lr=BASE_LR, weight_decay=0.05, layer_decay=LAYER_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS]
    )
    
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=15, verbose=True, path=MODEL_SAVE_PATH)
    
    wandb.init(project="R&D-Project", config={"type": "convnext_noun_consensus_v4"})

    # ---------------------------- TRAINING LOOP ----------------------------------------
    print("Starting Training...")
    global_step = 0 
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_augmentor.train() 
        epoch_loss = 0.0
        
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for video_batch, labels_batch in batch_loop:
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            # Random Frame for Training (Keep single frame for training speed)
            batch_size, channels, frames, h, w = video_batch.shape
            rand_frame = torch.randint(0, frames, (1,)).item()
            video_batch = video_batch[:, :, rand_frame, :, :]

            # GPU Augment (Mixup + Aggressive Cropping)
            with torch.no_grad():
                video_batch, labels_batch = train_augmentor(video_batch, labels_batch)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(video_batch)
                loss = criterion(outputs, labels_batch)
        
            optimizer.zero_grad()
            scaler.scale(loss).backward() 
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)        
            scaler.update()               
            
            epoch_loss += loss.item()
            wandb.log({"train/batch_loss": loss.item(), "global_step": global_step})
            global_step += 1
            
        # Eval
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # NEW: CONSENSUS EVALUATION
        # Pass num_clips=2 to get 2 temporal frames * 3 spatial crops = 6 views total
        avg_val_loss, val_accuracy = evaluate_model_consensus(model, val_loader, criterion, device, num_clips=2)
        
        scheduler.step()
        current_lr = optimizer.param_groups[-1]['lr']

        print(f"Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | Val Loss (Consensus): {avg_val_loss:.4f} | Acc: {val_accuracy:.2f}% | Head LR: {current_lr:.2e}")    

        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "val/epoch_loss": avg_val_loss,
            "val/accuracy": val_accuracy,
            "learning_rate": current_lr,
            "epoch": epoch + 1
        })
        
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    print("Training finished.")