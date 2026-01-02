import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = './EPIC-KITCHENS/annotations/EPIC_100_train.csv'
VIDEO_DIR = './EPIC-KITCHENS/videos_640x360' # Path to your actual .MP4 files
OUTPUT_DIR = './EPIC-KITCHENS/annotations/'
VAL_SIZE = 0.2
RANDOM_SEED = 42

def filter_existing_videos(df, video_root):
    """
    Keeps only the rows where the video file actually exists on disk.
    """
    valid_indices = []
    missing_count = 0
    
    print(f"Checking existence of videos in {video_root}...")
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        participant_id = row['participant_id']
        video_filename = f"{row['video_id']}.MP4"
        
        # Construct full path
        video_path = os.path.join(video_root, participant_id, video_filename)
        
        if os.path.exists(video_path):
            valid_indices.append(idx)
        else:
            missing_count += 1
            
    print(f"Finished filtering. Found {len(valid_indices)} valid videos.")
    print(f"Discarded {missing_count} entries because video files were missing.")
    
    return df.loc[valid_indices].copy()

def main():
    print(f"Loading {INPUT_CSV}...")
    full_df = pd.read_csv(INPUT_CSV)
    
    # --- STEP 1: FILTER MISSING VIDEOS ---
    # We must do this BEFORE splitting, otherwise we might assign 
    # a 'ghost' video to the validation set.
    df = filter_existing_videos(full_df, VIDEO_DIR)
    
    if len(df) == 0:
        print("Error: No valid videos found. Check your VIDEO_DIR path.")
        return

    # --- STEP 2: IDENTIFY RARE CLASSES ---
    # We count classes based on the FILTERED dataframe.
    class_counts = df['verb_class'].value_counts()
    
    # Classes with < 2 samples cannot be split (need 1 for train, 1 for val)
    rare_classes = class_counts[class_counts < 2].index.tolist()
    
    # --- STEP 3: SEPARATE SINGLETONS ---
    singletons = df[df['verb_class'].isin(rare_classes)]
    splittable = df[~df['verb_class'].isin(rare_classes)]
    
    print(f"Classes with only 1 sample (forced to train): {len(singletons)}")
    print(f"Splitting the remaining {len(splittable)} segments...")
    
    # --- STEP 4: STRATIFIED SPLIT ---
    train_df, val_df = train_test_split(
        splittable,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=splittable['verb_class']
    )
    
    # --- STEP 5: MERGE & SAVE ---
    # Add singletons back to training
    train_df = pd.concat([train_df, singletons])
    
    # Shuffle training set
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Save to disk
    train_path = os.path.join(OUTPUT_DIR, 'custom_train_80.csv')
    val_path = os.path.join(OUTPUT_DIR, 'custom_val_20.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print("-" * 30)
    print(f"Saved Train Split: {train_path} ({len(train_df)} samples)")
    print(f"Saved Val Split:   {val_path} ({len(val_df)} samples)")
    print("-" * 30)

if __name__ == "__main__":
    main()