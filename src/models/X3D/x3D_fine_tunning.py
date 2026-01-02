from EpicKitchensDataset import EpicKitchensDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os
import torch.optim.lr_scheduler as lr_scheduler
from early_stopping import EarlyStopping

# ------------------------------- CONFIGURATION ----------------------------------
LOAD_CHECKPOINT_PATH = "x3d_tmpaug_model_2.pth" 
SAVE_NEW_MODEL_PATH = "x3d_m_phase2_rescue.pth"
BATCH_SIZE = 32      
NUM_WORKERS = 12     
NUM_EPOCHS = 50      
NUM_FRAMES = 16

# ------------------------------- DATASETS ---------------------------------------
# Assuming tensors are already processed at 224x224 (Standard for X3D-M)
train_dataset = EpicKitchensDataset(
    path_to_data='./EPIC-KITCHENS',
    num_frames=NUM_FRAMES,
    testing=False,
    transform=None
)
NUM_VERB_CLASSES = train_dataset.num_classes

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS       
)

val_dataset = EpicKitchensDataset(
    path_to_data='./EPIC-KITCHENS',
    num_frames=NUM_FRAMES,
    testing=True,
    transform=None
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=NUM_WORKERS
)

# ------------------------------- MODEL SETUP ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Re-building X3D-M Architecture (Vanilla)...")
# 1. Load Skeleton
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False) 

# 2. MATCHING YOUR ORIGINAL CODE:
# We DO NOT replace the pooling layer with AdaptiveAvgPool. 
# We keep the original X3D pooling to match your saved checkpoint weights.

# 3. Replace Classifier Head
# The PytorchVideo X3D Head is located at: model.blocks[5]
num_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(num_features, NUM_VERB_CLASSES)

# 4. Load Your Checkpoint
print(f"Loading weights from {LOAD_CHECKPOINT_PATH}...")
try:
    state_dict = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
    
    # Standard load. Should work perfectly now that architecture matches.
    model.load_state_dict(state_dict)
    print("✓ Checkpoint loaded successfully. Starting Phase 2.")
except FileNotFoundError:
    print(f"ERROR: Could not find {LOAD_CHECKPOINT_PATH}. Check filename.")
    exit()
except RuntimeError as e:
    print(f"ERROR: Model architecture mismatch. Details: {e}")
    exit()

model = model.to(device)

# ------------------------- OPTIMIZATION (THE RESCUE) ----------------------------

# 1. Differential Learning Rates
# Freeze backbone softly (low LR) vs Head (high LR)
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    param.requires_grad = True # Ensure everything is trainable
    if "blocks.5" in name:     # The Head
        head_params.append(param)
    else:                      # The Backbone
        backbone_params.append(param)

optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},  # Very slow updates for body
    {'params': head_params,     'lr': 1e-4}   # Normal updates for head
], weight_decay=1e-3)                         # Strong Gravity (1e-3) to stop drift

# 2. Criterion
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 3. Scheduler (Aggressive Patience for Phase 2)
# FIX: Removed 'verbose=True' as it is deprecated/removed in newer PyTorch versions
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

# 4. Mixed Precision
scaler = torch.cuda.amp.GradScaler()

# 5. Early Stopping (Reset for Phase 2)
early_stopper = EarlyStopping(patience=8, verbose=True, path=SAVE_NEW_MODEL_PATH)

# ------------------------------- WANDB ------------------------------------------
wandb.init(
    project="R&D-Project",  
    name="X3D-M_Phase2_Rescue",
    config={
        "strategy": "Differential LR + High Decay",
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 1e-3,
        "source_checkpoint": LOAD_CHECKPOINT_PATH
    }
)

# ------------------------------- EVAL FUNCTION ----------------------------------
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for video_batch, labels_batch in tqdm(dataloader, desc="Val", leave=False):
            video_batch, labels_batch = video_batch.to(device), labels_batch.to(device)
            outputs = model(video_batch)
            loss = criterion(outputs, labels_batch)
            
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_samples += labels_batch.size(0)
            total_correct += (predictions == labels_batch).sum().item()

    return total_loss / len(dataloader), 100 * total_correct / total_samples

# ------------------------------- TRAINING LOOP ----------------------------------
print("Starting Phase 2 Training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}", leave=False)
    
    for video_batch, labels_batch in loop:
        video_batch, labels_batch = video_batch.to(device), labels_batch.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(video_batch)
            loss = criterion(outputs, labels_batch)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        wandb.log({"train/batch_loss": loss.item()})

    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    wandb.log({
        "train/epoch_loss": avg_train_loss,
        "val/epoch_loss": val_loss,
        "val/accuracy": val_acc,
        "lr_head": optimizer.param_groups[1]['lr']
    })
    
    scheduler.step(val_loss)
    early_stopper(val_loss, model)
    
    if early_stopper.early_stop:
        print("Early stopping triggered in Phase 2.")
        break

wandb.finish()