from dataloader import EpicKitchensDataset
import torch
from ThreeD_CNN import ThreeD_CNN
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

# ------------------------- CUDA ---------------------------------

# if torch.cuda.is_available():
#     print("CUDA is available! Using GPU.")
#     device = torch.device("cuda")
# else:
#     print("CUDA is not available. Using CPU.")
#     device = torch.device("cpu")

# # You can then check the name of your GPU
# if device.type == "cuda":
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available?  {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
#     print(f"Current GPU: {torch.cuda.get_device_name(0)}")

#--------------------------------------------------------------------

def evaluate_model(model, dataloader, device):
    """
    Runs the model on the test set and returns the accuracy.
    """
    # 1. Set model to evaluation mode
    # This turns off dropout and other training-specific layers
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    # 2. Set up the progress bar
    eval_loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    # 3. Disable gradient calculation
    # We don't need gradients for evaluation, so this saves memory
    with torch.no_grad():
        for video_batch, labels_batch in eval_loop:
            
            # Move data to the GPU
            video_batch = video_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # --- Handle bad batches (if any) ---
            # This skips any dummy data your dataloader might return
            # from corrupt files (where label is -1)
            valid_indices = labels_batch != -1
            if not valid_indices.any():
                continue # Skip batch if all samples are bad
                
            video_batch = video_batch[valid_indices]
            labels_batch = labels_batch[valid_indices]
            # --- End of bad batch handling ---

            # 4. Get model predictions (forward pass)
            outputs = model(video_batch)
            
            # 5. Get the predicted class (the one with the highest score)
            # outputs shape is [B, 125]
            # predictions shape is [B]
            _, predictions = torch.max(outputs, 1)
            
            # 6. Count correct predictions
            total_samples += labels_batch.size(0)
            total_correct += (predictions == labels_batch).sum().item()

    # 7. Calculate final accuracy
    accuracy = 100 * total_correct / total_samples
    return accuracy



if __name__ == '__main__':

    # ------------------------------- TRAINING DATASET -------------------------------
    # Training Dataset
    train_dataset = EpicKitchensDataset(
        path_to_data= './EPIC-KITCHENS',
        num_frames=16,
        testing=False,
        transform=None
    )

    # ------------------------------- Weighted sampler -------------------------------

    print("Calculating dataset weights for sampler...")
    # 1. Get the count for each class, ensure all classes are present
    class_counts = train_dataset.annotations['verb_class'].value_counts().sort_index()
    # Reindex to make sure we have an entry for all classes 0-88 (or 0-124)
    # Use train_dataset.num_classes which you already calculated
    class_counts = class_counts.reindex(range(97), fill_value=1) # Fill new classes with 1 to avoid /0

    # 2. Get the weight for each class (1.0 / count)
    class_weights = 1.0 / class_counts

    # 3. Create a weight for EVERY sample in the dataset
    sample_weights = class_weights[train_dataset.annotations['verb_class']].values
    sample_weights = torch.from_numpy(sample_weights).double()

    # 4. Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("Sampler created.")

    # print("--- Training Data Class Balance ---")
    # print(train_dataset.annotations['verb_class'].value_counts())
    # print("---------------------------------")

    # Training DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        # shuffle=True,
        sampler=sampler,      
        num_workers=6       
    )

    # ----------------------------- TESTING DATASET -----------------------------------
    # # Testing Dataset
    # test_dataset = EpicKitchensDataset(
    #     path_to_data='./EPIC-KITCHENS',
    #     num_frames=16,
    #     transform=None
    # )
    
    # # Testing Dataloader
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=4, 
    #     shuffle=False,
    #     num_workers=4
    # )


    # ---------------------------- PyTorch 3D-CNN MODEL ---------------------------------
    # Set up your device, model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Number of distinct Epic Kitchens verbs
    NUM_VERB_CLASSES = 97 
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
    MODEL_SAVE_PATH = "epic_kitchens_model.pth"
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")

    # ------------------------------------ TESTING -----------------------------------------
    # final_accuracy = evaluate_model(model, test_loader, device)
    # print(f"Final Model Accuracy: {final_accuracy:.2f}%")