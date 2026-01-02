import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ---------------- CONFIGURATION ----------------
# Path to the original official training CSV
INPUT_CSV = "./EPIC-KITCHENS/annotations/EPIC_100_train.csv"

# Output paths for your new splits
OUTPUT_TRAIN = "./EPIC-KITCHENS/annotations/custom_noun_train_80.csv"
OUTPUT_VAL = "./EPIC-KITCHENS/annotations/custom_noun_val_20.csv"

TEST_SIZE = 0.2  # 20% Validation
RANDOM_STATE = 42
# -----------------------------------------------

def create_noun_stratified_split():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Total samples: {len(df)}")
    
    # 1. Identify "Singleton" Nouns (Nouns that appear only 1 time)
    # We cannot stratify these (can't put 0.8 in train and 0.2 in val).
    # We must force them into Train to avoid crashing.
    noun_counts = df['noun_class'].value_counts()
    singleton_classes = noun_counts[noun_counts < 2].index.tolist()
    
    print(f"Found {len(singleton_classes)} noun classes with only 1 sample.")
    
    # Separate singletons from the rest
    df_singletons = df[df['noun_class'].isin(singleton_classes)]
    df_rest = df[~df['noun_class'].isin(singleton_classes)]
    
    print(f"Samples to split: {len(df_rest)}")
    print(f"Samples forced to train (singletons): {len(df_singletons)}")
    
    # 2. Perform Stratified Split on the Rest
    # stratify=df_rest['noun_class'] ensures Noun proportions are preserved
    train_df, val_df = train_test_split(
        df_rest, 
        test_size=TEST_SIZE, 
        stratify=df_rest['noun_class'], 
        random_state=RANDOM_STATE
    )
    
    # 3. Add Singletons back to Train
    train_df = pd.concat([train_df, df_singletons])
    
    # 4. Save
    print(f"Saving splits...")
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)
    
    print(f"\nDone!")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
    print(f"Val set:   {len(val_df)} samples ({len(val_df)/len(df):.1%})")
    
    # 5. Verification
    print("\n--- Distribution Check (Top 5 Nouns) ---")
    train_dist = train_df['noun_class'].value_counts(normalize=True)
    val_dist = val_df['noun_class'].value_counts(normalize=True)
    
    comparison = pd.DataFrame({'Train %': train_dist, 'Val %': val_dist})
    print(comparison.head(5))

if __name__ == "__main__":
    create_noun_stratified_split()