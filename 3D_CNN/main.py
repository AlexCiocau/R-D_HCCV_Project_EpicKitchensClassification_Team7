from EpicKitchensDataset import EpicKitchensDataset
import torch
from ThreeD_CNN import ThreeD_CNN
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

# ------------------------- CUDA ---------------------------------

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

    # ------------------------------- TRAINING DATASET -------------------------------
    # Training Dataset
    train_dataset = EpicKitchensDataset(
        path_to_data= './EPIC-KITCHENS',
        num_frames=16,
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
    class_counts = class_counts.reindex(range(97), fill_value=1)
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
        batch_size=32,
        sampler=sampler,      
        num_workers=6       
    )

    # ---------------------------- PyTorch 3D-CNN MODEL ---------------------------------
    # Set up your device, model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Number of distinct Epic Kitchens verbs
    NUM_VERB_CLASSES = 97 
    # .to(devide) sends the model to the GPU! Very important to add it!
    model = ThreeD_CNN(num_classes=NUM_VERB_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # ---------------------------- TRAINING LOOP ----------------------------------------
    print("Starting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        
        model.train()

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
            
            print(f"Loss: {loss.item()}")

    print("Training finished.")

    # ------------------------------- SAVE TRAINED MODEL -----------------------------------
    MODEL_SAVE_PATH = "naive_epic_kitchens_model.pth"
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")