from EpicKitchensDataset import EpicKitchensDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import wandb
import os
import torchvision.models.video as video_models
import torch.optim.lr_scheduler as lr_scheduler
from early_stopping import EarlyStopping


# ------------------------------- EVALUATION FUNCTION --------------------------------
# It is called after each training epoch to asses the model's generalization 
def evaluate_model(model, dataloader, criterion, device):
    """
    Runs the model on the test/validation set and returns loss & accuracy.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    eval_loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad(): # Disable gradient calculation
        for video_batch, labels_batch in eval_loop:
            
            # --- Skip bad batches (if any) ---
            valid_indices = labels_batch != -1
            if not valid_indices.any():
                continue
            video_batch = video_batch[valid_indices]
            labels_batch = labels_batch[valid_indices]
            # --- End skipping ---

            video_batch = video_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            outputs = model(video_batch)
            loss = criterion(outputs, labels_batch)
            
            # Calculate metrics
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_samples += labels_batch.size(0)
            total_correct += (predictions == labels_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy

if __name__ == '__main__':

    # ------------------------------- HYPERPARAMETERS --------------------------------
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32     
    NUM_WORKERS = 10     
    NUM_EPOCHS = 200
    NUM_FRAMES = 16
    NUM_VERB_CLASSES = 97
    MODEL_SAVE_PATH = "r3d_epic_kitchens_model.pth"

    # ------------------------------- TRAINING DATASET -------------------------------
    # Training Dataset
    train_dataset = EpicKitchensDataset(
        path_to_data= './EPIC-KITCHENS',
        num_frames=NUM_FRAMES,
        testing=False,
        transform=None
    )

    # Inspecting dataset imbalance
    print("--- Training Data Class Balance ---")
    print(train_dataset.annotations['verb_class'].value_counts())
    print("---------------------------------")

    # ------------------------------- Weighted sampler -------------------------------

    print("Calculating dataset weights for sampler...")
    class_counts = train_dataset.annotations['verb_class'].value_counts().sort_index()
    class_counts = class_counts.reindex(range(NUM_VERB_CLASSES), fill_value=1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_dataset.annotations['verb_class']].values
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("Sampler created.")

    # ------------------------------- TRAINING DATALOADER ----------------------------
    # Training DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,      
        num_workers=NUM_WORKERS       
    )

    # ----------------------------- VALIDATION PREREQUISITES --------------------------
    val_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS',
        num_frames=16,
        testing=True,
        transform=None
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS
    )

    # ---------------------------- GET DEVICE -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Using CPU.")

    # ---------------------------- R3D-18 MODEL ---------------------------------------
    # Load pretrained R3D-18
    print("Loading pretrained R3D-18 model...")
    model = video_models.r3d_18(weights="DEFAULT") 

    # --- NEW: Freeze layers ---
    # Freeze the "stem" (first conv) and "layer1" (early features)
    for name, param in model.named_parameters():
        if "stem" in name or "layer1" in name:
            param.requires_grad = False
    print("Frozen Stem and Layer 1 parameters.")
    # ---------------------------------------------------------------------------------
    
    # Replace classifier 
    num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, NUM_VERB_CLASSES)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Drop 50% of neurons randomly during training
        nn.Linear(num_features, NUM_VERB_CLASSES)
    )
    
    # .to(devide) sends the model to the GPU! Very important to add it!
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)       

    # ---------------------------- OPTIMIZATION: MIXED PRECISION ----------------------
    # Initializes a GradScaler object once, outside of the training loop.
    # This is essential for preventing underflow with half-precision floating-point values.
    scaler = torch.amp.GradScaler("cuda")
    print("Enabled CUDA Mixed Precision Training (torch.cuda.amp).")

    # ---------------------------- LR SCHEDULER & EARLY STOPPING ----------------------
    # Scheduler: Reduce LR when validation loss plateaus
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0001)
    # EarlyStopping: Stop if val_loss doesn't improve for 10 epochs
    early_stopper = EarlyStopping(patience=10, verbose=True, path=MODEL_SAVE_PATH)
    # ---------------------------- W&B INITIALIZATION ---------------------------------
    wandb.init(
        project="R&D-Project",  
        config={                     
            "architecture": "R3D-18_pretrained",
            "dataset": "EPIC-Kitchens-100",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "num_frames": NUM_FRAMES,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
        }
    )
    # It will log histograms of weights/gradients every 100 batches.
    wandb.watch(model, criterion, log="all", log_freq=100)


    # ---------------------------- TRAINING LOOP ----------------------------------------
    print("Starting training...")
    global_step = 0  # <-- W&B: To track total batches seen
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        # Start Training mode
        model.train()
        epoch_loss = 0.0
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        # Without btach_loop we would iterate only over train_loader
        for video_batch, labels_batch in batch_loop:
            
            # Move data to the GPU
            video_batch = video_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Pass the 5D batch from the DataLoader into your model
            # outputs = model(video_batch)
            
            # # Calculate loss and update weights
            # loss = criterion(outputs, labels_batch)
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # ---  Wrap forward pass in autocast() ---
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(video_batch)
                loss = criterion(outputs, labels_batch)
        
            # --- Use scaler for backward() and step() ---
            optimizer.zero_grad()
            scaler.scale(loss).backward() # Scale the loss before backward pass

            # --- NEW: Gradient Clipping ---
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Clips gradient norm to 1.0 (standard value for 3D CNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ------------------------------

            scaler.step(optimizer)        # Update the optimizer weights
            scaler.update()               # Update the scale for the next iteration
            
            # -------------- LOG CURRENT BATCH --------------------

            batch_loss = loss.item()
            epoch_loss += batch_loss
            wandb.log({
                "train/batch_loss": batch_loss,
                "global_step": global_step
            })
            batch_loop.set_postfix(loss=batch_loss)
            global_step += 1
        # ------------------ EPOCH END -----------------------------
        avg_epoch_loss = epoch_loss / len(train_loader)

        # ------------------ EVALUATION ----------------------------
        print("Starting evaluation...\n")
        avg_val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print("Evaluation finished.\n")
        print(f"Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%\n")    

        # ------------------ LOG EPOCH -----------------------------
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "val/epoch_loss": avg_val_loss,
            "val/accuracy": val_accuracy,
            "learning_rate": current_lr,
            "weight_decay": 1e-5,
            "epoch": epoch + 1
        })

        # --- SCHEDULER & EARLY STOPPING ---
        scheduler.step(avg_val_loss) # Step scheduler on validation loss
        early_stopper(avg_val_loss, model)
        
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    print("Training finished.")

    # ------------------------------- SAVE TRAINED MODEL -----------------------------------
    # print(f"Saving model to {MODEL_SAVE_PATH}...")
    # torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # print("Model saved successfully.")

    print(f"Loading best model from {MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Best model loaded successfully.")

    # ------------------------------- END WANDB LOGGING -----------------------------------
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}", 
        type="model",
        description="Trained MC3-18_pretrained model for EPIC-Kitchens"
    )
    artifact.add_file(MODEL_SAVE_PATH)
    wandb.run.log_artifact(artifact)

    wandb.finish()
    print("W&B run finished.")