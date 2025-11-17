import cv2
import pandas as pd
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm


print("CV2 version: ", cv2.__version__)


# --- Configuration ---
ANNOTATION_FILE = './EPIC-KITCHENS/annotations/epic_train_split.csv'
VIDEO_DIR = './EPIC-KITCHENS/videos_640x360' 
# Use os.path.expandvars to read the $VSC_SCRATCH variable
TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/tensors' 
TENSOR_OUTPUT_DIR = os.path.expandvars(TENSOR_OUTPUT_DIR_RAW)
NUM_FRAMES = 16 
IMAGE_SIZE = 224 
# ---------------------

# --------------------------- BEST-PRACTICE -----------------------------------
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145, 0.216989]

transform = T.Compose([
    # Training augmentation: Randomly crops a square and resizes it.
    T.RandomResizedCrop((112, 112)), 
    T.ToTensor(),
    T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)
])

# ------------------------- MY TRANSFORMS -------------------------------------
# # --- Define Transforms (Baked into tensors) ---
# transform = T.Compose([
#     T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

def get_frame_indices(total_frames, num_frames_to_sample):
    """Generates a list of evenly-spaced frame indices."""
    if total_frames < num_frames_to_sample:
        indices = np.arange(0, total_frames).tolist()
        indices += [total_frames - 1] * (num_frames_to_sample - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
    return indices.tolist()

def main():
    print("Loading annotations...")
    annotations = pd.read_csv(ANNOTATION_FILE)

    # --------------------- CHECK IF WE'RE IN THE CORRECT LOCATION --------------------------------
    if '$VSC_SCRATCH' in TENSOR_OUTPUT_DIR:
        print(f"Error: $VSC_SCRATCH environment variable not set or not expanded.")
        print(f"Path is still: {TENSOR_OUTPUT_DIR}")
        return # Stop the script
    # --------------------------------------------------------------------------------------------------

    print(f"Ensuring output directory exists: {TENSOR_OUTPUT_DIR}")
    os.makedirs(TENSOR_OUTPUT_DIR, exist_ok=True)
    
    print("Grouping annotations by video_id for efficient processing...")
    grouped = annotations.groupby('video_id')
    
    # Loop over each VIDEO
    for video_id, segments in tqdm(grouped, desc="Processing Videos"):
        
        participant_id = segments.iloc[0]['participant_id']
        video_filename = f"{video_id}.MP4" 
        
        # Build the correct path using the participant_id: .../EPIC-KITCHENS/videos/train/P01/P01_01.MP4"
        video_path = os.path.join(VIDEO_DIR, participant_id, video_filename)
         
        # Handle not founb videos
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found {video_path}, skipping {len(segments)} segments.")
            continue

        # --- Open video file ONCE ---
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue
        except Exception as e:
            print(f"Error opening {video_path}: {e}")
            continue

        # Loop over each SEGMENT within that video
        for _, row in segments.iterrows():
            segment_uid = row['narration_id']
            output_filename = os.path.join(TENSOR_OUTPUT_DIR, f"{segment_uid}.pt")
            
            if os.path.exists(output_filename):
                continue 

            try:
                start_frame = int(row['start_frame'])
                end_frame = int(row['stop_frame'])
                
                if start_frame >= end_frame:
                    continue 

                segment_frames_pil = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame
                
                while current_frame <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break 
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    segment_frames_pil.append(frame_pil)
                    current_frame += 1

                if not segment_frames_pil:
                    continue

                frame_indices = get_frame_indices(len(segment_frames_pil), NUM_FRAMES)
                sampled_frames = [segment_frames_pil[i] for i in frame_indices]
                
                transformed_frames = [transform(frame) for frame in sampled_frames]
                
                video_tensor = torch.stack(transformed_frames, dim=1) 
                
                torch.save(video_tensor, output_filename)

            except Exception as e:
                print(f"Error processing segment {segment_uid}: {e}")

        cap.release() 

    print("Pre-processing to tensors complete!")

if __name__ == "__main__":
    main()