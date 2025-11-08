from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import pandas as pd
import numpy as np
import torch

class EpicKitchensDataset(Dataset):
    """
    A custom Dataset class for loading Epic Kitchens data.
    """
    def __init__(self, path_to_data, num_frames, transform=None):
        # Path to data should point to the root directory containing the dataset
        # Given the parent directory this class will compute the relative location of the annotations
        self.path_to_data = path_to_data
        self.transform = transform
        self.num_frames = num_frames

        # Import annotations
        path_to_annotations = path_to_data + '/annotations/EPIC_100_train.csv'
        print("Here is the path to annotations: ", path_to_annotations)
        self.annotations = pd.read_csv(path_to_annotations)

        # Import videos
        self.path_to_video = path_to_data + "/videos_640x360"

    def __len__(self):
        # Return the total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            # Retrieve a sample: a clip and its label (! a clip is a short sequence; there are multiple in one video)
            annotation = self.annotations.iloc[idx]

            # We retrieve the the start-stop frames and the video_id of the clip of our current iteration
            video_id = annotation['video_id']
            participant = annotation['participant_id']
            start_frame = annotation['start_frame']
            stop_frame = annotation['stop_frame']
            label = torch.tensor(annotation['verb_class'], dtype=torch.long)

            # Retrieve video
            video_path = f"{self.path_to_video}/{participant}/{video_id}.MP4"
            print("Here is the path to the video: ", video_path)
            v_reader = VideoReader(video_path)

            # Select <num_frames> equally spaced frames between start_frame and stop_frame
            frame_indices_to_sample = np.linspace(start_frame, stop_frame, num=self.num_frames, dtype=int)
            frame_indices_to_sample = np.clip(frame_indices_to_sample, 0, len(v_reader) - 1) 
            frames = v_reader.get_batch(frame_indices_to_sample)
            frames_np = frames.asnumpy()
            clip_tensor = torch.from_numpy(frames_np)
            
            # 3D CNN will expect [Channels, Frames, Height, Width]
            clip_tensor = clip_tensor.permute(3, 0, 1, 2) # [C, T, H, W]
            return clip_tensor, label
        except Exception as e:
            pass
    
    def get_annotations(self):
        # Return annotations as panda type
        return self.annotations