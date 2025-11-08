from dataloader import EpicKitchensDataset
import torch
from ThreeD_CNN import ThreeD_CNN
import torch.nn as nn

# This is the "main" part of your script.
# Put all the execution logic inside this block.
if __name__ == '__main__':

    # Instantiate dataset class
    train_dataset = EpicKitchensDataset(
        path_to_data= './EPIC-KITCHENS',
        num_frames=32,
        transform=None
    )

    # Create the DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,      
        num_workers=1      
    )

    # Set up your device, model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_VERB_CLASSES = 125 # Example number for Epic Kitchens verbs
    model = ThreeD_CNN(num_classes=NUM_VERB_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Starting training...")
    num_epochs = 1000
    for epoch in range(num_epochs):
        for video_batch, labels_batch in train_loader:
            
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