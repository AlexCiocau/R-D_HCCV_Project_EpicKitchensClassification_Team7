import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sanity_dataset import EpicKitchensDataset
import torchvision.models as models
from tqdm import tqdm

# --- 1. Define a "Mini" Dataset for Debugging ---
class DebugDataset(EpicKitchensDataset):
    """
    Subclass that forces the dataset to only load 200 samples.
    This allows us to test the RAM Cache logic in seconds, not minutes.
    """
    def _build_index(self):
        full_samples = super()._build_index()
        print(f"DEBUG MODE: Truncating dataset from {len(full_samples)} to 200 samples.")
        return full_samples[:200]

def sanity_check():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checking device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: You are running on CPU! GPU utilization cannot be tested.")

    # --- 2. Initialize Data (RAM Cache Mode) ---
    print("\n--- Phase 1: Testing RAM Cache Speed ---")
    start_load = time.time()
    
    # We use the DebugDataset to load only a tiny slice
    dataset = DebugDataset(
        path_to_data='./EPIC-KITCHENS', 
        num_frames=16, 
        testing=False, 
        ram_cache=True 
    )
    
    print(f"RAM Cache Load Time: {time.time() - start_load:.2f} seconds")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, # Use a realistic batch size
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # --- 3. Initialize Dummy Model ---
    print("\n--- Phase 2: Initializing Model ---")
    model = models.convnext_tiny(weights=None) # No need to download weights for speed check
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 300)
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # --- 4. The Speed Test (The crucial part) ---
    print("\n--- Phase 3: Running Throughput Test (Warmup) ---")
    
    # Warmup (1 batch) to wake up GPU
    for img, label in dataloader:
        img, label = img.to(device), label.to(device)
        _ = model(img)
        break
        
    print("--- Phase 4: Measuring Batches/Second ---")
    start_time = time.time()
    num_batches = 0
    
    # Run through the tiny dataset 5 times to simulate sustained load
    for _ in range(5): 
        for img, label in tqdm(dataloader, desc="Simulating Epoch"):
            # Move to GPU
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            # Forward + Backward
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            num_batches += 1
            
    total_time = time.time() - start_time
    avg_batch_time = total_time / num_batches
    
    print(f"\nRESULTS:")
    print(f"Total Batches Processed: {num_batches}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Time Per Batch: {avg_batch_time:.4f}s")
    
    if avg_batch_time < 0.5:
        print("SUCCESS: Pipeline is FAST. GPU is likely fully fed.")
    else:
        print("WARNING: Pipeline is SLOW. Check Dataloader overhead.")

if __name__ == "__main__":
    sanity_check()