import torch
from ThreeD_CNN import ThreeD_CNN  
from dataloader import EpicKitchensDataset 
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

if __name__ == '__main__':
    MODEL_PATH = "epic_kitchens_model.pth"
    NUM_VERB_CLASSES = 97 # MUST be the same as you used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # You first build the "empty shell" of the model
    print("Creating model structure...")
    model = ThreeD_CNN(num_classes=NUM_VERB_CLASSES).to(device)

    # --- 3. LOAD THE SAVED WEIGHTS ---
    print(f"Loading trained weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH))

    # --- 4. SET MODEL TO EVALUATION MODE (CRITICAL!) ---
    # This turns off dropout, batchnorm updates, etc.
    model.train()

    # --- 5. PREPARE YOUR TEST DATA ---
    train_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS',
        num_frames=16,
        testing=False,
        transform=None
    )

    print("--- Training Data Class Balance ---")
    print(train_dataset.annotations['verb_class'].value_counts())
    print("---------------------------------")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True, 
        num_workers=8
    )

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    MODEL_SAVE_PATH = "epic_kitchens_model.pth"
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")