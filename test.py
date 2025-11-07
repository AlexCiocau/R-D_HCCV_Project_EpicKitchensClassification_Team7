import torch
import json
from pathlib import Path
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.clip_sampling import make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    Normalize,
    PackPathway,  # <-- IMPORTANT: Import PackPathway
)
from torchvision.transforms import Compose, Lambda

# --- 0. Setup Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 1. Load a Pre-trained Model ---
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
model = model.eval()
model = model.to(device)

# --- 2. Define the PREPROCESSING (Corrected for SlowFast) ---
# These are the specific parameters required for the SlowFast R50 model
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4  # The ratio between slow and fast pathway frame rates
num_clips = 10
num_frames_per_clip = num_frames * sampling_rate

# The clip duration in seconds
clip_duration = (num_frames * sampling_rate) / frames_per_second

video_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames * sampling_rate), # Sample 64 frames
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            ShortSideScale(size=side_size),
            UniformCropVideo(size=(crop_size, crop_size)),
            PackPathway(alpha=slowfast_alpha)  # <-- CRITICAL: This splits the video into slow/fast pathways
        ]
    ),
)

# --- 3. Load and Preprocess YOUR Video ---

# Find a video file in your directory
video_dir = Path("videos_640x360")
# Use .rglob to search recursively (in P01, P02, etc.)
# IMPORTANT: Change "*.mp4" if your videos are .avi, .mkv, or another format
video_files = list(video_dir.rglob("*.mp4"))

if not video_files:
    print(f"No video files (.mp4) found in {video_dir.absolute()}")
    print("Please check the path and file extension.")
    exit()

# Load the first video found
video_path = video_files[0]
print(f"--- Loading video: {video_path} ---")

# We'll sample one clip from the center of the video
clip_sampler = make_clip_sampler("uniform", clip_duration)

# Load the video from the path
video_data = EncodedVideo.from_path(video_path)

# Get video duration
video_duration = video_data.duration

# Select a clip
clip_time_points = clip_sampler(video_duration)
start_sec = clip_time_points[0]
end_sec = clip_time_points[1]

# Load, decode, and sample the clip
video_clip = video_data.get_clip(start_sec=start_sec, end_sec=end_sec)

# Apply the transforms
inputs = video_transform(video_clip)

# Move the data to the GPU
# inputs["video"] is now a *list* of 2 tensors: [slow_pathway, fast_pathway]
inputs["video"] = [i.to(device) for i in inputs["video"]]

# --- 4. Get Predictions ---
print("Running model prediction...")
with torch.no_grad():
    preds = model(inputs["video"])

# --- 5. Interpret the Results ---
# The output 'preds' are raw scores. We need to convert them to probabilities
# and map them to the Kinetics dataset's class names.

# Download the label map from here:
# https://huggingface.co/datasets/huggingface/label-files/blob/main/kinetics400-id2label.json
# and save it as "kinetics_400_labels.json" in your project folder.
try:
    with open("kinetics_400_labels.json", "r") as f:
        # The file maps ID (string) to Label (string), convert ID to int
        kinetics_labels_map = json.load(f)
        kinetics_labels = [kinetics_labels_map[str(i)] for i in range(400)]
except FileNotFoundError:
    print("\n[Error] `kinetics_400_labels.json` not found.")
    print("Please download it from: https://huggingface.co/datasets/huggingface/label-files/blob/main/kinetics400-id2label.json")
    kinetics_labels = None

if kinetics_labels:
    # Apply Softmax to get probabilities
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)

    # Get the top 5 predictions
    pred_scores, pred_indices = torch.topk(preds, k=5)

    print("\n--- Top 5 Predictions: ---")
    for i in range(len(pred_scores[0])):
        score = pred_scores[0][i].item()
        label = kinetics_labels[pred_indices[0][i].item()]
        print(f"  {i+1}. {label:<25} (Score: {score:.4f})")