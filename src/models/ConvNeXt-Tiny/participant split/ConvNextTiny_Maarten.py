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
    print("WARNING: torchvision.transforms.v2 not found. Please update torchvision!")
    import torchvision.transforms as v2

# ------------------------------- GPU AUGMENTATION (CORRECTED) ------------------------
class GPUAugmentor(nn.Module):
    """
    Applies transforms. 
    CORRECTION: Reduced color jitter intensity to preserve food color information.
    """
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.num_classes = num_classes
        
        # ImageNet Stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if training:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.RandomResizedCrop(224, scale=(0.6, 1.0)), 
                v2.RandomHorizontalFlip(p=0.5),
                
                # --- DOMAIN GENERALIZATION AUGMENTATIONS ---
                # CORRECTION: Lowered values. Food color is important! 
                # Old: brightness=0.4... -> New: brightness=0.1...
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                
                v2.RandomGrayscale(p=0.2),
                
                # 3. Gaussian Blur
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25),
            ])
        else:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
            ])

        # Mixup/Cutmix
        if training:
            self.mixup_cutmix = v2.RandomChoice([
                v2.MixUp(alpha=0.8, num_classes=num_classes),
                v2.CutMix(alpha=1.0, num_classes=num_classes)
            ])
    
    def forward(self, x, labels=None):
        # x shape: [B, C, H, W] (Single frame for ConvNeXt)
        
        # 1. Apply Geometric/Color Transforms
        x = self.geo_transforms(x)
        
        # 2. Normalize
        x = (x - self.mean) / self.std
        
        # 3. Apply Mixup/CutMix (Training Only)
        if self.training and labels is not None:
            x, labels = self.mixup_cutmix(x, labels)
            return x, labels
            
        return x

# ------------------------------- EVALUATION FUNCTION (UPGRADED) --------------------------------
def evaluate_model(model, dataloader, criterion, device, augmentor):
    """
    CORRECTION: Now uses Multi-Frame Evaluation (3 views) instead of single-frame.
    This significantly improves validation accuracy.
    """
    model.eval() 
    augmentor.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    eval_loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad(): 
        for video_batch, labels_batch in eval_loop:
            
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            # Get dimensions: [Batch, Channels, Frames, Height, Width]
            b, c, f, h, w = video_batch.shape

            # --- Multi-View Logic ---
            # Select 3 frames: Early (25%), Middle (50%), Late (75%)
            frames_to_test = [f // 4, f // 2, 3 * f // 4]
            
            multi_view_batch = []
            
            for idx in frames_to_test:
                # Extract frame [B, C, H, W]
                frame = video_batch[:, :, idx, :, :]
                
                # Normalize using augmentor (returns ONLY x)
                frame = augmentor(frame, None)
                multi_view_batch.append(frame)
            
            # Stack views into batch dim: [B*3, C, H, W]
            input_tensor = torch.cat(multi_view_batch, dim=0)
            
            # Forward pass (Batch * 3)
            outputs = model(input_tensor)
            
            # Reshape back to [3, Batch, NumClasses] and Average predictions
            outputs = outputs.view(3, b, -1).mean(dim=0)
            
            # Calculate Loss & Accuracy
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
    
    LEARNING_RATE = 5e-5 
    BATCH_SIZE = 128     
    NUM_WORKERS = 8  
    NUM_EPOCHS = 50
    NUM_FRAMES = 16
    MODEL_SAVE_PATH = "convnext_tiny_noun_participants.pth"

    # ------------------------------- DATASET ----------------------------------------
    train_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS', num_frames=NUM_FRAMES, testing=False, transform=None 
    )
    NUM_NOUN_CLASSES = train_dataset.num_classes 
    print(f"Verified Noun Classes: {NUM_NOUN_CLASSES}")
    
    # --- Weighted Sampler ---
    try:
         print("Calculating weights for sampler...")
         sample_weights = train_dataset.get_sample_weights(balance_by='noun')
         sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
         shuffle = False
         print("Using WeightedRandomSampler (Balanced by Noun).")
    except:
         print("Sampler method not found or failed, defaulting to shuffle.")
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
    
    # Replace Head
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_features, NUM_NOUN_CLASSES)
    )
    model = model.to(device)
    
    # ---------------------------- AUGMENTOR --------------------------------------
    train_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=True).to(device)
    val_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=False).to(device)
    
    # ---------------------------- OPTIMIZER ------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler() # Standard scaler call
    early_stopper = EarlyStopping(patience=10, verbose=True, path=MODEL_SAVE_PATH)
    
    wandb.init(project="R&D-Project", config={"type": "convnext_noun_retrain"})

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

            # --- Random Frame Selection (Training Regularization) ---
            batch_size, channels, frames, h, w = video_batch.shape
            rand_frame = torch.randint(0, frames, (1,)).item()
            video_batch = video_batch[:, :, rand_frame, :, :]

            # --- GPU Augment ---
            with torch.no_grad():
                video_batch, labels_batch = train_augmentor(video_batch, labels_batch)

            with torch.cuda.amp.autocast(): # Standard autocast call
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