"""
==============================================================================
VARIANT: K-FOLD TRAINING SCRIPT
==============================================================================
filename: ConvNextTiny_kfold.py

[PURPOSE]
Trains the ConvNeXt-Tiny model on a specific fold of the dataset.
It uses command-line arguments to switch between folds (0-4) without 
changing the code manually.

[USAGE]
Run for each fold sequentially or in parallel:
$ python src/models/ConvNeXt-Tiny/ConvNextTiny_kfold.py --fold 0
$ python src/models/ConvNeXt-Tiny/ConvNextTiny_kfold.py --fold 1
...
==============================================================================
"""
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler, DataLoader
import wandb
import os
import numpy as np
import torchvision.models as models
from early_stopping import EarlyStopping

# IMPORTS
from EpicKitchensDataset_KFold import EpicKitchensDataset
# Reuse the Augmentor from your previous file since it works perfectly
from ConvNextTiny_retrained import GPUAugmentor, evaluate_model 

# --- CONFIG ---
BATCH_SIZE = 128
NUM_WORKERS = 8
NUM_EPOCHS = 50
LEARNING_RATE = 4e-4
TENSOR_DIR = "$VSC_SCRATCH/x3d_train_tensors"
SPLIT_DIR = "./EPIC-KITCHENS/annotations/splits"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    return parser.parse_args()

def get_balanced_sampler(dataset):
    # Calculate weights on the fly for the current fold
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=300)
    class_counts[class_counts==0] = 1 # Avoid div/0
    class_weights = 1.0 / class_counts
    weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

if __name__ == '__main__':
    args = get_args()
    fold_idx = args.fold
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING TRAINING FOR FOLD {fold_idx} ---")

    # 1. Define Paths
    train_csv = os.path.join(SPLIT_DIR, f"train_fold_{fold_idx}.csv")
    val_csv = os.path.join(SPLIT_DIR, f"val_fold_{fold_idx}.csv")
    save_path = f"convnext_tiny_fold_{fold_idx}.pth"

    # 2. Datasets
    train_ds = EpicKitchensDataset(train_csv, TENSOR_DIR)
    val_ds = EpicKitchensDataset(val_csv, TENSOR_DIR)
    
    # 3. Loaders
    print("Calculating sampler weights...")
    sampler = get_balanced_sampler(train_ds)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 4. Model
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(model.classifier[2].in_features, 300)
    )
    model = model.to(device)

    # 5. Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=10, verbose=True, path=save_path)
    
    train_aug = GPUAugmentor(num_classes=300, training=True).to(device)
    val_aug = GPUAugmentor(num_classes=300, training=False).to(device)

    # WandB: Group runs together so you can see them overlaid
    wandb.init(project="Epic-KFold", group="ConvNext-KFold", name=f"Fold-{fold_idx}", config={"fold": fold_idx})

    # 6. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train(); train_aug.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        epoch_loss = 0.0
        
        for video, label in loop:
            video, label = video.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            # Random frame sampling
            rand_frame = torch.randint(0, video.shape[2], (1,)).item()
            video = video[:, :, rand_frame, :, :]
            
            with torch.no_grad(): video, label = train_aug(video, label)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(video)
                loss = criterion(out, label)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()
            
        # Eval
        train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, val_aug)
        scheduler.step()
        
        print(f"Fold {fold_idx} | Ep {epoch+1} | T_Loss: {train_loss:.3f} | V_Loss: {val_loss:.3f} | Acc: {val_acc:.2f}%")
        
        wandb.log({"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch+1})
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early Stopping.")
            break
            
    print(f"Fold {fold_idx} Complete.")