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

# ------------------------------- HELPER: LAYER-WISE LR DECAY ----------------------
def get_layer_wise_groups(model, base_lr, weight_decay, layer_decay=0.8):
    """
    Assigns strictly lower LR to earlier layers to preserve pre-trained features.
    The 'head' gets base_lr.
    The 'stem' gets base_lr * (layer_decay ^ depth).
    """
    param_groups = {}
    
    # ConvNeXt specific structure mapping
    # features.0 (stem) -> features.1 (stage 1) -> ... -> features.7 (stage 4) -> classifier
    # We treat 'features' as 8 logical blocks + 1 classifier = 9 layers total.
    num_layers = 9 
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Determine Layer ID
        if name.startswith("classifier"):
            layer_id = num_layers - 1 # Top layer
        elif name.startswith("features"):
            try:
                # features.X. ...
                block_idx = int(name.split('.')[1])
                layer_id = block_idx
            except:
                layer_id = 0
        else:
            layer_id = 0 # Default to stem

        # Calculate Scale: Decay moves backwards from the head
        # Head (ID 8) -> Scale 1.0
        # Stem (ID 0) -> Scale 0.8^8 ~= 0.16
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

# ------------------------------- GPU AUGMENTOR (NO MIXUP) ------------------------
class GPUAugmentor(nn.Module):
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
                
                # Zoom Crop: Forces model to look at object details
                v2.RandomResizedCrop(224, scale=(0.8, 1.0)), 
                v2.RandomHorizontalFlip(p=0.5),
                
                # --- DOMAIN GENERALIZATION (Crucial for Participant Split) ---
                # Forces model to ignore specific lighting/countertop colors
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25),
            ])
        else:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
            ])

    def forward(self, x, labels=None):
        # 1. Apply Transforms
        x = self.geo_transforms(x)
        
        # 2. Normalize
        x = (x - self.mean) / self.std
        
        # 3. No Mixup (Return clean labels)
        return x, labels

# ------------------------------- EVALUATION FUNCTION --------------------------------
def evaluate_model(model, dataloader, criterion, device, augmentor):
    model.eval() 
    augmentor.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad(): 
        for video_batch, labels_batch in dataloader:
            
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            # --- Middle Frame Logic (Standard Eval) ---
            middle_frame_idx = video_batch.shape[2] // 2
            video_batch = video_batch[:, :, middle_frame_idx, :, :]
            
            # Transform
            video_batch = augmentor(video_batch, None)
            
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
    
    # Setup for LLRD: Base LR is high (4e-4) for the head, but decays for backbones
    BASE_LR = 4e-4 
    LAYER_DECAY = 0.8
    
    BATCH_SIZE = 128     
    NUM_WORKERS = 8  
    NUM_EPOCHS = 50
    NUM_FRAMES = 16
    WARMUP_EPOCHS = 5
    
    MODEL_SAVE_PATH = "convnext_tiny_noun_participants_v2.pth"

    # ------------------------------- DATASET ----------------------------------------
    print("Initializing Datasets...")
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
         print("Using WeightedRandomSampler.")
    except:
         print("Sampler method not found, defaulting to shuffle.")
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
    val_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=False).to(device)
    
    # ---------------------------- OPTIMIZER (LLRD) ----------------------------------
    criterion = nn.CrossEntropyLoss()
    
    # Use Layer-Wise Groups
    print(f"Configuring Layer-Wise LR Decay (Base: {BASE_LR}, Decay: {LAYER_DECAY})")
    param_groups = get_layer_wise_groups(model, base_lr=BASE_LR, weight_decay=0.05, layer_decay=LAYER_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    
    # Scheduler
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
    # Increased patience because domain generalization takes longer
    early_stopper = EarlyStopping(patience=15, verbose=True, path=MODEL_SAVE_PATH)
    
    wandb.init(project="R&D-Project", config={"type": "convnext_noun_LLRD_NoMixup"})

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

            # Random Frame Selection
            batch_size, channels, frames, h, w = video_batch.shape
            rand_frame = torch.randint(0, frames, (1,)).item()
            video_batch = video_batch[:, :, rand_frame, :, :]

            # GPU Augment
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
        avg_val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, val_augmentor)
        
        scheduler.step()
        # Log LR of the head (group -1)
        current_lr = optimizer.param_groups[-1]['lr']

        print(f"Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.2f}% | Head LR: {current_lr:.2e}")    

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