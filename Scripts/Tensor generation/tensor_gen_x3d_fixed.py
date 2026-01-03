"""
==============================================================================
STEP -1: TENSOR GENERATION (PRE-PROCESSING)
==============================================================================
filename: tensor_gen_x3d_fixed.py

[PURPOSE]
This script converts raw MP4 video files into lightweight PyTorch tensors (.pt).
1. Reads videos from 'videos_640x360'.
2. Extracts frames based on 'start_frame'/'stop_frame' from CSVs.
3. Resizes and Crops frames to 256x256.
4. Saves them as Float16 tensors to the Scratch directory.

[USAGE]
Run this ONCE before any training.
$ python scripts/preprocessing/tensor_gen_x3d_fixed.py

[OUTPUT]
Generates thousands of .pt files in "$VSC_SCRATCH/x3d_train_tensors"
==============================================================================
"""
import cv2
import pandas as pd
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm

# ----------------------- CONFIGURATION ---------------------
NUM_FRAMES = 16 
IMAGE_SIZE = 256 
AUGMENTATION_THRESHOLD = 100 
IS_TRAINING = False 

VIDEO_DIR = './EPIC-KITCHENS/videos_640x360'

if IS_TRAINING:
    ANNOTATION_FILE = './EPIC-KITCHENS/annotations/custom_train_80.csv'
    TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_train_tensors'
else:
    # Ensure this is the correct CSV for the validation/test set you are processing
    ANNOTATION_FILE = './EPIC-KITCHENS/annotations/EPIC_100_validation.csv'
    TENSOR_OUTPUT_DIR_RAW = '$VSC_SCRATCH/x3d_codabench_val'
    AUGMENTATION_THRESHOLD = 0 

TENSOR_OUTPUT_DIR = os.path.expandvars(TENSOR_OUTPUT_DIR_RAW)
# -----------------------------------------------------------

print("Using CLEAN Transforms (Resize -> ToTensor)")
transform = T.Compose([
    T.Resize(IMAGE_SIZE),       
    T.CenterCrop(IMAGE_SIZE),   
    T.ToTensor()                
])

# ---------------------- SAMPLING LOGIC --------------------
def get_uniform_indices(total_frames, num_frames_to_sample):
    if total_frames < num_frames_to_sample:
        indices = np.arange(0, total_frames).tolist()
        indices += [total_frames - 1] * (num_frames_to_sample - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int).tolist()
    return [indices]

def main():
    print(f"Ensuring output directory exists: {TENSOR_OUTPUT_DIR}")
    if '$VSC_SCRATCH' in TENSOR_OUTPUT_DIR:
        print("Error: Env variable not expanded.")
        return
    os.makedirs(TENSOR_OUTPUT_DIR, exist_ok=True)

    print(f"Loading annotations from {ANNOTATION_FILE}...")
    annotations = pd.read_csv(ANNOTATION_FILE)
    
    # Track missing files
    missing_report = []

    grouped = annotations.groupby('video_id')
    
    for video_id, segments in tqdm(grouped, desc="Processing Videos"):
            participant_id = segments.iloc[0]['participant_id']
            video_filename = f"{video_id}.MP4" 
            video_path = os.path.join(VIDEO_DIR, participant_id, video_filename)
            
            # --- DIAGNOSTIC 1: Source Video Check ---
            if not os.path.exists(video_path): 
                msg = f"SKIPPED VIDEO: Source file not found: {video_path}"
                print(msg)
                missing_report.append(msg)
                continue
            
            cap = None
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened(): 
                    msg = f"SKIPPED VIDEO: Could not open CV2 capture: {video_path}"
                    print(msg)
                    missing_report.append(msg)
                    continue
            except Exception as e:
                msg = f"SKIPPED VIDEO: CV2 Exception on {video_path}: {e}"
                print(msg)
                missing_report.append(msg)
                continue

            # Get total frames for boundary checks
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _, row in segments.iterrows():
                segment_uid = row['narration_id']
                
                # Check for output existence
                output_filename_0 = os.path.join(TENSOR_OUTPUT_DIR, f"{segment_uid}_clip0.pt")
                if os.path.exists(output_filename_0):
                    continue

                start_frame = int(row['start_frame'])
                end_frame = int(row['stop_frame'])
                
                # --- DIAGNOSTIC 2: Timestamp Logic ---
                if start_frame >= end_frame: 
                    print(f"SKIPPED SEGMENT {segment_uid}: Start ({start_frame}) >= End ({end_frame})")
                    continue 

                # --- DIAGNOSTIC 3: Out of Bounds ---
                # Some annotations might go past the actual video length
                if start_frame >= total_video_frames:
                    print(f"SKIPPED SEGMENT {segment_uid}: Start ({start_frame}) is past video end ({total_video_frames})")
                    continue

                # Extract Frames
                segment_frames_pil = []
                
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    curr = start_frame
                    
                    # Safety clamp: Don't try to read past the video end
                    safe_end_frame = min(end_frame, total_video_frames - 1)
                    
                    while curr <= safe_end_frame:
                        ret, frame = cap.read()
                        if not ret: 
                            # Frame read failed (end of file or corrupt frame)
                            break 
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        segment_frames_pil.append(Image.fromarray(frame_rgb))
                        curr += 1
                except Exception as e:
                    print(f"Error reading frames for {segment_uid}: {e}")

                # --- DIAGNOSTIC 4: Empty Extraction ---
                if not segment_frames_pil: 
                    msg = f"SKIPPED SEGMENT {segment_uid}: No frames extracted. (Start: {start_frame}, End: {end_frame}, VidLen: {total_video_frames})"
                    print(msg)
                    missing_report.append(msg)
                    continue
                
                # Logic: If we extracted FEWER frames than expected, we proceed anyway 
                # (Uniform sampling handles small lists fine)
                
                all_frame_indices = get_uniform_indices(len(segment_frames_pil), NUM_FRAMES)
                
                for clip_idx, frame_indices in enumerate(all_frame_indices):
                    output_filename = os.path.join(TENSOR_OUTPUT_DIR, f"{segment_uid}_clip{clip_idx}.pt")
                    if os.path.exists(output_filename): continue

                    try:
                        sampled_frames = [segment_frames_pil[i] for i in frame_indices]
                        transformed_frames = [transform(frame) for frame in sampled_frames]
                        video_tensor = torch.stack(transformed_frames, dim=1)
                        video_tensor = video_tensor.half() 
                        torch.save(video_tensor, output_filename)
                    except Exception as e:
                        print(f"Error saving {output_filename}: {e}")

            if cap:
                cap.release() 
    
    print("\n--- GENERATION COMPLETE ---")
    if len(missing_report) > 0:
        print(f"WARNING: {len(missing_report)} issues found:")
        for r in missing_report:
            print(r)
    else:
        print("Success: No missing source files or read errors.")

if __name__ == "__main__":
    main()