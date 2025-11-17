import torch
from ThreeD_CNN import ThreeD_CNN  
from EpicKitchensDataset import EpicKitchensDataset 
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # -------------------------------- CONFIGURATION -------------------------------
    MODEL_PATH = "naive_epic_kitchens_model.pth"
    NUM_VERB_CLASSES = 97
    NUM_WORKERS = 8
    TEST_BATCH_SIZE = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------- LOAD MODEL ---------------------------------
    print("Creating model structure...")
    model = ThreeD_CNN(num_classes=NUM_VERB_CLASSES).to(device)
    print(f"Loading trained weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH))

    # Enable EVALUATION MODE (this turns off dropout, batchnorm updates, etc.)
    model.eval()

    # ------------------------------- DATASET and DATALOADER -----------------------
    test_dataset = EpicKitchensDataset(
        path_to_data='./EPIC-KITCHENS',
        num_frames=16,
        testing=True,
        transform=None
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS
    )

    # ------------------------ RUN INFERENCE --------------------------------------
    print("Starting inference to generate metrics...")

    all_preds = []
    all_gts = []

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
            # pred_class = prediction.item()
            # ground_truth_class = labels_batch.item()

            preds_list = prediction.cpu().tolist()
            gts_list = labels_batch.cpu().tolist()
            
            # Use extend() to add all items from the batch lists to our master lists
            all_preds.extend(preds_list)
            all_gts.extend(gts_list)
            
            # print(f"Prediction: {pred_class}, Ground Truth: {ground_truth_class}")
            if preds_list: 
                tqdm.write(f"  [Batch Sample] Prediction: {preds_list[0]}, Ground Truth: {gts_list[0]}")

    if all_gts and all_preds:
        accuracy = accuracy_score(all_gts, all_preds)
        print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No predictions or ground truths were collected.")

    print("Inference finished.")