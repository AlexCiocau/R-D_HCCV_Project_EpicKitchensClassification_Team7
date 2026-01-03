"""
==============================================================================
UTILITY: RESUME TRAINING
==============================================================================
filename: resume_training.py

[PURPOSE]
A dedicated script to continue training from a saved checkpoint.
Use this if a job crashes or times out on the HPC.
1. Loads the model state from 'PREV_MODEL_PATH'.
2. Fast-forwards the LR Scheduler to 'START_EPOCH'.
3. Initializes the EarlyStopper with the previous best loss.

[CONFIG]
Update 'PREV_MODEL_PATH' and 'START_EPOCH' manually before running.
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
    print("WARNING: torchvision.transforms.v2 not found.")
    import torchvision.transforms as v2

# ------------------------------- HELPER: LAYER-WISE LR DECAY ----------------------
def get_layer_wise_groups(model, base_lr, weight_decay, layer_decay=0.8):
    param_groups = {}
    num_layers = 9 
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith("classifier"): layer_id = num_layers - 1 
        elif name.startswith("features"):
            try: block_idx = int(name.split('.')[1]); layer_id = block_idx
            except: layer_id = 0
        else: layer_id = 0 
        scale = layer_decay ** (num_layers - layer_id - 1)
        group_name = f"layer_{layer_id}"
        if group_name not in param_groups:
            param_groups[group_name] = {"params": [], "lr": base_lr * scale, "weight_decay": weight_decay}
        param_groups[group_name]["params"].append(param)
    return list(param_groups.values())

# ------------------------------- GPU AUGMENTOR ------------------------
class GPUAugmentor(nn.Module):
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.num_classes = num_classes
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if training:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True), 
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25),
            ])
            self.mixup_cutmix = v2.RandomChoice([
                v2.MixUp(num_classes=num_classes, alpha=0.8),
                v2.CutMix(num_classes=num_classes, alpha=1.0)
            ])
        else:
            self.geo_transforms = v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224)])

    def forward(self, x, labels=None):
        x = self.geo_transforms(x)
        x = (x - self.mean) / self.std
        if self.training and labels is not None:
            x, labels = self.mixup_cutmix(x, labels)
        return x, labels

# ------------------------------- CONSENSUS EVALUATION ------------------------
def evaluate_model_consensus(model, dataloader, criterion, device, num_clips=2):
    model.eval() 
    total_loss = 0.0; total_correct = 0; total_samples = 0
    tta_transform = v2.Compose([v2.Resize(256, antialias=True), v2.FiveCrop(224)])
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad(): 
        for video_batch, labels_batch in tqdm(dataloader, desc="Consensus Eval", leave=False):
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            B, C, T, H, W = video_batch.shape
            
            indices = torch.linspace(T//4, 3*T//4, steps=num_clips).long()
            selected_frames = video_batch[:, :, indices, :, :]
            flat_frames = selected_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            
            crops = tta_transform(flat_frames) 
            input_tensor = torch.stack(crops, dim=1).view(-1, C, 224, 224)
            input_tensor = (input_tensor - norm_mean) / norm_std
            
            logits = model(input_tensor)
            logits = logits.view(B, num_clips * 5, -1)
            consensus_logits = logits.mean(dim=1) 
            
            loss = criterion(consensus_logits, labels_batch)
            total_loss += loss.item()
            _, predictions = torch.max(consensus_logits, 1)
            total_samples += labels_batch.size(0)
            total_correct += (predictions == labels_batch).sum().item()

    return total_loss / len(dataloader), 100 * total_correct / total_samples

if __name__ == '__main__':
    # ------------------------------- RESUME CONFIGURATION --------------------------------
    torch.backends.cudnn.benchmark = True
    
    # PATH TO THE MODEL THAT JUST FINISHED
    # Ensure this matches the file name from your previous script
    PREV_MODEL_PATH = "convnext_tiny_noun_participants_consensus.pth"
    
    # WHERE TO SAVE THE NEW PROGRESS
    NEW_MODEL_PATH = "convnext_tiny_noun_participants_continued.pth"
    
    # SETUP
    START_EPOCH = 17       # The last completed epoch (from your logs)
    PREV_BEST_LOSS = 4.2369 # The last Val Loss from your logs
    TOTAL_EPOCHS = 50      # Keep total same as before
    
    BASE_LR = 5e-5 
    LAYER_DECAY = 0.8
    BATCH_SIZE = 32      
    NUM_WORKERS = 8  
    NUM_FRAMES = 16
    WARMUP_EPOCHS = 5

    # ------------------------------- DATASET ----------------------------------------
    print("Initializing Datasets...")
    train_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS', num_frames=NUM_FRAMES, testing=False, transform=None 
    )
    NUM_NOUN_CLASSES = train_dataset.num_classes 
    
    sampler = None
    shuffle = True
    try:
         if hasattr(train_dataset, 'get_sample_weights'):
             sample_weights = train_dataset.get_sample_weights(balance_by='noun') 
             sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
             shuffle = False
    except:
         sampler = None; shuffle = True

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    val_dataset = EpicKitchensDataset(path_to_data='./EPIC-KITCHENS', num_frames=16, testing=True, transform=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------- MODEL LOAD ---------------------------------------------
    print(f"Resuming from: {PREV_MODEL_PATH}")
    model = models.convnext_tiny(weights=None) # No need for ImageNet weights, we overwrite them
    
    # Rebuild Head
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, NUM_NOUN_CLASSES))
    
    # Load State Dict
    if os.path.exists(PREV_MODEL_PATH):
        state_dict = torch.load(PREV_MODEL_PATH, map_location=device)
        # Handle if saved as full model vs state_dict
        if isinstance(state_dict, nn.Module):
            model.load_state_dict(state_dict.state_dict())
        else:
            model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"CRITICAL ERROR: Could not find {PREV_MODEL_PATH}")
        exit()

    model = model.to(device)
    
    # ---------------------------- OPTIMIZER & SCHEDULER SYNC -------------------------
    train_augmentor = GPUAugmentor(num_classes=NUM_NOUN_CLASSES, training=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Re-initialize Optimizer
    param_groups = get_layer_wise_groups(model, base_lr=BASE_LR, weight_decay=0.05, layer_decay=LAYER_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    
    # Re-initialize Scheduler
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])
    
    scaler = torch.amp.GradScaler("cuda")
    
    # Initialize Early Stopper with PREV BEST LOSS so we don't save worse models
    early_stopper = EarlyStopping(patience=15, verbose=True, path=NEW_MODEL_PATH)
    early_stopper.val_loss_min = PREV_BEST_LOSS
    early_stopper.best_score = -PREV_BEST_LOSS
    
    print(f"Fast-forwarding scheduler by {START_EPOCH} epochs...")
    for _ in range(START_EPOCH):
        scheduler.step()
        
    print(f"Resumed Learning Rate: {optimizer.param_groups[-1]['lr']:.2e}")

    wandb.init(project="R&D-Project", config={"type": "convnext_noun_RESUMED", "start_epoch": START_EPOCH})

    # ---------------------------- TRAINING LOOP ----------------------------------------
    print("Resuming Training...")
    global_step = START_EPOCH * len(train_loader)
    
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        model.train()
        train_augmentor.train() 
        epoch_loss = 0.0
        
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=False)
        
        for video_batch, labels_batch in batch_loop:
            video_batch = video_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            rand_frame = torch.randint(0, NUM_FRAMES, (1,)).item()
            video_batch = video_batch[:, :, rand_frame, :, :]

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
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_val_loss, val_accuracy = evaluate_model_consensus(model, val_loader, criterion, device, num_clips=2)
        
        scheduler.step()
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