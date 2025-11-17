from EpicKitchensDataset import EpicKitchensDataset
import torch
from ThreeD_CNN import ThreeD_CNN
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import wandb
import os


# ------------------------- CUDA ---------------------------------
# Check if CUDA is availabe and if so, then what GPU am I using
# if torch.cuda.is_available():
#     print("CUDA is available!")
#     print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#     device = torch.device("cuda")
# else:
#     print("CUDA is NOT available. Using CPU.")
#     device = torch.device("cpu")

#--------------------------------------------------------------------


if __name__ == '__main__':

    # ------------------------------- HYPERPARAMETERS --------------------------------
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256     
    NUM_WORKERS = 16     
    NUM_EPOCHS = 200
    NUM_FRAMES = 16
    NUM_VERB_CLASSES = 97
    MODEL_SAVE_PATH = "naive_epic_kitchens_model.pth"

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

    # ------------------------------- DATALOADER -------------------------------
    # Training DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,      
        num_workers=NUM_WORKERS       
    )

    # ---------------------------- PyTorch 3D-CNN MODEL ---------------------------------
    # Set up your device, model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # .to(devide) sends the model to the GPU! Very important to add it!
    model = ThreeD_CNN(num_classes=NUM_VERB_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------- W&B INITIALIZATION ---------------------------------
    wandb.init(
        project="R&D-Project",  
        config={                     
            "architecture": "ThreeD_CNN",
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
            outputs = model(video_batch)
            
            # Calculate loss and update weights
            loss = criterion(outputs, labels_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # -------------- LOG CURRENT BATCH --------------------

            batch_loss = loss.item()
            epoch_loss += batch_loss
            wandb.log({
                "train/batch_loss": batch_loss,
                "global_step": global_step
            })
            batch_loop.set_postfix(loss=batch_loss)
            global_step += 1
            

        # ------------------ LOG EPOCH -----------------------------
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "epoch": epoch + 1
        })
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss}")

    print("Training finished.")

    # ------------------------------- SAVE TRAINED MODEL -----------------------------------
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")

    # ------------------------------- END WANDB LOGGING -----------------------------------
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}", 
        type="model",
        description="Trained 3D-CNN model for EPIC-Kitchens"
    )
    artifact.add_file(MODEL_SAVE_PATH)
    wandb.run.log_artifact(artifact)

    wandb.finish()
    print("W&B run finished.")