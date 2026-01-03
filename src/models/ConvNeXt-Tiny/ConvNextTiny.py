"""
==============================================================================
STEP 1: SPATIAL TRAINING (CONVNEXT-TINY)
==============================================================================
filename: ConvNextTiny.py

[PURPOSE]
This script trains the Spatial Stream to recognize Objects (Nouns).
It implements a modern training pipeline:
1. Model: Loads 'ConvNeXt-Tiny' with ImageNet-1K pretrained weights.
2. Sampling: Randomly selects a SINGLE frame from the video clip to treat as an image.
3. Augmentation: Uses a 'GPUAugmentor' class to apply MixUp, CutMix, and 
   ColorJitter directly on the GPU (faster than CPU).
4. Scheduler: Uses Cosine Annealing for smooth learning rate decay.

[USAGE]
$ python src/models/ConvNeXt-Tiny/ConvNextTiny.py

[OUTPUT]
Saves the best weights to: "convnext_tiny_noun_best.pth"
==============================================================================
"""
from EpicKitchensDataset_vNouns import EpicKitchensDataset
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
    print("WARNING: torchvision.transforms.v2 not found. Please update torchvision!")
    import torchvision.transforms as v2

# ------------------------------- GPU AUGMENTATION (WITH MIXUP) ------------------------
class GPUAugmentor(nn.Module):
    """
    Applies transforms + Mixup/CutMix on the GPU.
    """
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.num_classes = num_classes
        
        # ImageNet Stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if training:
            # geometry transforms (applied to image only)
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True), 
                v2.RandomCrop(224),
                v2.RandomHorizontalFlip(p=0.5),
                # Stronger Color Jitter for ConvNeXt
                v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ])
            # Mixup/CutMix (applied to image AND label)
            self.mixup_cutmix = v2.RandomChoice([
                v2.MixUp(num_classes=num_classes, alpha=0.8),
                v2.CutMix(num_classes=num_classes, alpha=1.0)
            ])
        else:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
            ])

    def forward(self, x, labels=None):
        # x: [B, 3, H, W]
        # labels: [B] (Indices)
        
        # 1. Apply Geometry
        x = self.geo_transforms(x)
        
        # 2. Apply Mixup/CutMix (Only in Training)
        if self.training and labels is not None:
            x, labels = self.mixup_cutmix(x, labels)
        
        # 3. Normalize
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            
        x = (x - self.mean) / self.std
        
        return x, labels

# ------------------------------- EVALUATION FUNCTION --------------------------------
def evaluate_model(model, dataloader, criterion, device, augmentor):
    model.eval() 
    augmentor.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    eval_loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad(): 
        for video_batch, labels_batch in eval_loop:
            
            valid_indices = labels_batch != -1
            if not valid_indices.any(): continue
            video_batch = video_batch[valid_indices]
            labels_batch = labels_batch[valid_indices]

            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            # --- Middle Frame Logic (Standard Eval) ---
            middle_frame_idx = video_batch.shape[2] // 2
            video_batch = video_batch[:, :, middle_frame_idx, :, :]
            
            # Transform (No Mixup in Val)
            video_batch, _ = augmentor(video_batch, None)
            
            # Forward
            outputs = model(video_batch)
            loss = criterion(outputs, labels_batch)
            
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_samples += labels_batch.size(0)
            total_correct += (predictions == labels_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy

if __name__ == '__main__':
    # ------------------------------- HYPERPARAMETERS --------------------------------
    torch.backends.cudnn.benchmark = True
    
    # ConvNeXt often prefers Cosine Scheduling, so we set a max epoch count
    LEARNING_RATE = 4e-4 # Slightly higher start for AdamW + Cosine
    BATCH_SIZE = 128     # Keep A100 busy
    NUM_WORKERS = 8  
    NUM_EPOCHS = 50
    NUM_FRAMES = 16
    MODEL_SAVE_PATH = "convnext_tiny_noun_best.pth"

    # ------------------------------- DATASET ----------------------------------------
    # Ensure transform=None in dataset so we get raw tensors
    train_dataset = EpicKitchensDataset(
        path_to_data= './EPIC-KITCHENS', num_frames=NUM_FRAMES, testing=False, transform=None 
    )
    NUM_NOUN_CLASSES = train_dataset.num_classes 
    print(f"Detected {NUM_NOUN_CLASSES} noun classes.")
    
    # --- Weighted Sampler ---
    # Re-enable this to help with rare classes if you have the method available
    print("Calculating weights for sampler...")
    try:
         sample_weights = train_dataset.get_sample_weights(balance_by='hybrid') 
         sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
         shuffle = False
         print("Using WeightedRandomSampler.")
    except:
         print("Sampler method not found, defaulting to standard shuffle.")
         sampler = None
         shuffle = True

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )

    val_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS', num_frames=16, testing=True, transform=None
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------- MODEL: CONVNEXT-TINY -------------------------------
    print("Loading ConvNeXt-Tiny...")
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    # ConvNeXt Head: classifier[2] is the linear layer
    num_features = model.classifier[2].in_features
    
    # Replace Head
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5), # Strong Dropout
        nn.Linear(num_features, NUM_NOUN_CLASSES)
    )
    model = model.to(device)
    
    # ---------------------------- GPU AUGMENTOR --------------------------------------
    train_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=True).to(device)
    val_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=False).to(device)
    
    # ---------------------------- OPTIMIZER ------------------------------------------
    criterion = nn.CrossEntropyLoss()
    
    # ConvNeXt Setup: AdamW with 0.05 weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # --- Cosine Scheduler ---
    # Better for ConvNeXt than Plateau. It warms up, then drops smoothly.
    # We use T_max = NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=15, verbose=True, path=MODEL_SAVE_PATH)
    
    wandb.init(project="R&D-Project", config={"type": "convnext_advanced"})

    # ---------------------------- TRAINING LOOP ----------------------------------------
    print("Starting ConvNeXt training...")
    global_step = 0 
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_augmentor.train() 
        epoch_loss = 0.0
        
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for video_batch, labels_batch in batch_loop:
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            # --- Random Frame Selection (TSN Style) ---
            # Instead of always taking middle, pick random frame for training
            batch_size, channels, frames, h, w = video_batch.shape
            rand_frame = torch.randint(0, frames, (1,)).item()
            video_batch = video_batch[:, :, rand_frame, :, :]

            # --- GPU Augment ---
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
            
        # ------------------ EVALUATION ----------------------------
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, val_augmentor)
        
        # Step Scheduler (Cosine steps every epoch, not based on val loss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.2f}% | LR: {current_lr:.2e}")    

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