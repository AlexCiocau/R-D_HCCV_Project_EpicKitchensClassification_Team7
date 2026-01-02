import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# CONFIG
INPUT_CSV = "./EPIC-KITCHENS/annotations/EPIC_100_train.csv"
OUTPUT_DIR = "./EPIC-KITCHENS/annotations/splits"
NUM_FOLDS = 3  # 3 splits is a good balance between robustness and cost
RANDOM_STATE = 42

def create_folds():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    
    # Handle Singletons (Classes with 1 sample cannot be split)
    noun_counts = df['noun_class'].value_counts()
    singletons = df[df['noun_class'].isin(noun_counts[noun_counts < NUM_FOLDS].index)]
    data_to_split = df[~df['noun_class'].isin(noun_counts[noun_counts < NUM_FOLDS].index)]
    
    print(f"Singletons forced to train: {len(singletons)}")
    print(f"Data to split: {len(data_to_split)}")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_to_split, data_to_split['noun_class'])):
        # 1. Split
        fold_train = data_to_split.iloc[train_idx]
        fold_val = data_to_split.iloc[val_idx]
        
        # 2. Add singletons to train (to avoid crash)
        fold_train = pd.concat([fold_train, singletons])
        
        # 3. Save
        train_filename = os.path.join(OUTPUT_DIR, f"train_fold_{fold_idx}.csv")
        val_filename = os.path.join(OUTPUT_DIR, f"val_fold_{fold_idx}.csv")
        
        fold_train.to_csv(train_filename, index=False)
        fold_val.to_csv(val_filename, index=False)
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(fold_train)} | Val: {len(fold_val)}")
        print(f"  Saved to {train_filename} / {val_filename}")

if __name__ == "__main__":
    create_folds()