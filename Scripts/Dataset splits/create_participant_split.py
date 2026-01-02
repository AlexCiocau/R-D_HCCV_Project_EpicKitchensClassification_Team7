import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_CSV = "./EPIC-KITCHENS/annotations/EPIC_100_train.csv"
OUTPUT_DIR = "./EPIC-KITCHENS/annotations/splits_participants"
TRAIN_RATIO = 0.85 # 85% of participants in Train, 15% in Val
RANDOM_STATE = 42

def create_robust_participant_split():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Get Unique Participants
    all_participants = df['participant_id'].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_participants)
    
    n_train = int(len(all_participants) * TRAIN_RATIO)
    train_p = all_participants[:n_train]
    val_p = all_participants[n_train:]
    
    print(f"Total Participants: {len(all_participants)}")
    print(f"Train Participants ({len(train_p)}): {train_p}")
    print(f"Val Participants   ({len(val_p)}): {val_p}")
    
    # 2. Filter Dataframes
    train_df = df[df['participant_id'].isin(train_p)].copy()
    val_df = df[df['participant_id'].isin(val_p)].copy()
    
    # 3. CRITICAL SAFETY CHECK: Zero-Shot Nouns
    # If a noun appears in Val but NOT in Train, the model cannot possibly learn it.
    train_nouns = set(train_df['noun_class'].unique())
    val_nouns = set(val_df['noun_class'].unique())
    
    # Nouns in Val that are missing from Train
    missing_nouns = val_nouns - train_nouns
    
    print("\n" + "="*40)
    print("SPLIT STATISTICS")
    print("="*40)
    print(f"Train Samples: {len(train_df)}")
    print(f"Val Samples:   {len(val_df)}")
    
    if len(missing_nouns) > 0:
        print(f"\n[WARNING] {len(missing_nouns)} Noun classes are missing from the Training Set!")
        print(f"The model will never predict these correctly in Validation: {missing_nouns}")
        print("This is normal for participant splits (some objects are rare), but be aware.")
    else:
        print("\n[SUCCESS] All validation noun classes are present in the training set.")

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "custom_participant_train.csv")
    val_path = os.path.join(OUTPUT_DIR, "custom_participant_val.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print("\n" + "="*40)
    print("SAVED FILES")
    print("="*40)
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print("Update your Dataset class to point to these files for retraining.")

if __name__ == "__main__":
    create_robust_participant_split()