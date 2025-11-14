import torch
from ThreeD_CNN import ThreeD_CNN  
from EpicKitchensDataset import EpicKitchensDataset 
from torch.utils.data import DataLoader
from tqdm import tqdm
# You would also import OpenCV (cv2) here for adding overlays

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
    model.eval()

    # --- 5. PREPARE YOUR TEST DATA ---
    test_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS',
        num_frames=16,
        testing=True,
        transform=None
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True, 
        num_workers=0
    )

    # --- 6. RUN INFERENCE ---
    print("Starting inference to generate overlays...")

    # We don't need to calculate gradients, so this saves memory
    with torch.no_grad():
        for video_batch, labels_batch in tqdm(test_loader):
            
            # Move data to GPU
            video_batch = video_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Get prediction
            outputs = model(video_batch)
            # print("Raw model output (logits):", outputs[0])
            _, prediction = torch.max(outputs, 1)

            # Get the numbers
            pred_class = prediction.item()
            ground_truth_class = labels_batch.item()
            
            print(f"Prediction: {pred_class}, Ground Truth: {ground_truth_class}")
            # for i in range(video_batch.size(0)):
            #     # Use [i] to get the i-th element
            #     pred_class = prediction[i].item()
            #     ground_truth_class = labels_batch[i].item()
                
            #     print(f"Prediction: {pred_class}, Ground Truth: {ground_truth_class}")

            # --- YOUR OVERLAY LOGIC HERE ---
            # 1. Get the original video file path (you'll need to modify
            #    your dataloader to return this)
            # 2. Load the video with OpenCV (cv2.VideoCapture)
            # 3. Loop through frames, add text with cv2.putText()
            #    - Text for "Prediction: {pred_class}"
            #    - Text for "Ground Truth: {ground_truth_class}"
            # 4. Save the new video with cv2.VideoWriter
            #
            # if you_want_to_stop_after_one:
            #     break

    print("Inference finished.")