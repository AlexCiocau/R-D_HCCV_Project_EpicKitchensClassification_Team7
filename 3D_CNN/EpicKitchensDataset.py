from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional as TF

class EpicKitchensDataset(Dataset):
    """
    A custom Dataset class for loading Epic Kitchens data.
    """
    def __init__(self, path_to_data, num_frames, testing=False, transform=None):
        # Path to data should point to the root directory containing the dataset
        # Given the parent directory this class will compute the relative location of the annotations
        self.path_to_data = path_to_data
        self.transform = transform
        self.num_frames = num_frames
        self.testing = testing

        # Import annotations
        if not self.testing:
            # path_to_annotations = path_to_data + '/annotations/EPIC_100_train.csv'
            path_to_annotations = path_to_data + '/annotations/epic_train_split.csv'
        else:
            # path_to_annotations = path_to_data + '/annotations/EPIC_100_validation.csv'
            path_to_annotations = path_to_data + '/annotations/epic_validation_split.csv'
        print("Here is the path to annotations: ", path_to_annotations)
        all_annotations = pd.read_csv(path_to_annotations)

        # Import videos
        self.path_to_video = path_to_data + "/videos_640x360"

        # Import tensors
        self.path_to_tensors = path_to_data + "/tensors_16frame"

        # Filter annotations: keep only the ones in the reduced dataset
        print("Filtering annotations... this may take a moment.")
        self.annotations = self.filter_annotations(all_annotations)
        print(f"Dataset reduced from {len(all_annotations)} to {len(self.annotations)} available samples.")

        if len(self.annotations) == 0:
            raise RuntimeError("No valid video files found! Check your data paths.")
        
        

    def filter_annotations(self, all_annotations):
        """
        Loops through the dataframe and keeps only the rows
        where the corresponding video file exists.
        """
        valid_annotations = []
        
        # Use tqdm for a nice progress bar
        for _, row in tqdm(all_annotations.iterrows(), total=all_annotations.shape[0]):
            video_id = row['video_id']
            participant = row['participant_id']
            
            video_path = os.path.join(self.path_to_video, participant, f"{video_id}.MP4")
            
            if os.path.exists(video_path):
                valid_annotations.append(row.to_dict())
                
        # Convert the list of valid dicts back to a DataFrame
        return pd.DataFrame(valid_annotations)

    def __len__(self):
        # Return the total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        # ----------------------------------- NORMAL IMPLEMENTATION -------------------------------------------
        # try:
        #     # Retrieve a sample: a clip and its label (! a clip is a short sequence; there are multiple in one video)
        #     annotation = self.annotations.iloc[idx]

        #     # We retrieve the the start-stop frames and the video_id of the clip of our current iteration
        #     video_id = annotation['video_id']
        #     participant = annotation['participant_id']
        #     start_frame = annotation['start_frame']
        #     stop_frame = annotation['stop_frame']
        #     label = torch.tensor(annotation['verb_class'], dtype=torch.long)

        #     # Retrieve video
        #     video_path = f"{self.path_to_video}/{participant}/{video_id}.MP4"
        #     print("Here is the path to the video: ", video_path)
        #     v_reader = VideoReader(video_path, ctx=cpu(0))

        #     # Select <num_frames> equally spaced frames between start_frame and stop_frame
        #     frame_indices_to_sample = np.linspace(start_frame, stop_frame, num=self.num_frames, dtype=int)
        #     frame_indices_to_sample = np.clip(frame_indices_to_sample, 0, len(v_reader) - 1) 
        #     frames = v_reader.get_batch(frame_indices_to_sample)
        #     frames_np = frames.asnumpy()
        #     clip_tensor = torch.from_numpy(frames_np)
            
        #     # 3D CNN will expect [Channels, Frames, Height, Width]
        #     # clip_tensor = clip_tensor.permute(3, 0, 1, 2) # [C, T, H, W]
        #     clip_tensor = clip_tensor.permute(0, 3, 1, 2)

        #     # 2. Resize H and W
        #     #    (from [16, 3, 360, 640] to [16, 3, 224, 224])
        #     clip_tensor = TF.resize(clip_tensor, (224, 224))
            
        #     # 3. Permute to [C, T, H, W] for the 3D CNN model
        #     #    (from [16, 3, 224, 224] to [3, 16, 224, 224])
        #     clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        #     clip_tensor = clip_tensor.float() / 255.0
        #     return clip_tensor, label
        # except Exception as e:
        #     pass

        # ----------------------------------- TENSOR IMPLEMENTATION -------------------------------------------
        row = self.annotations.iloc[idx]
        segment_uid = row['narration_id'] # Changed from 'uid' to 'narration_id'
        label = int(row['verb_class']) 
        tensor_path = os.path.join(self.path_to_tensors, f"{segment_uid}.pt")

        try:
            video_tensor = torch.load(tensor_path)
            return video_tensor, label
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self))
            
        except Exception as e:
            print(f"Error loading tensor at index {idx} ({tensor_path}): {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def get_annotations(self):
        # Return annotations as panda type
        return self.annotations