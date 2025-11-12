# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os

# # --- Configuration ---
# # *** IMPORTANT: Update this path to point to your CSV file ***
# INPUT_CSV_PATH = './EPIC-KITCHENS/annotations/EPIC_100_train.csv' 

# # Column name containing the class labels (for stratification)
# CLASS_COLUMN = 'verb_class' 

# # Proportion of the data to use for the validation set (e.g., 0.2 = 20%)
# VALIDATION_SPLIT_SIZE = 0.2

# # Random state for reproducible splits
# RANDOM_STATE = 42

# # Output file names
# NEW_TRAIN_CSV = 'epic_train_split.csv'
# NEW_VAL_CSV = 'epic_validation_split.csv'
# # ---------------------

# def split_dataset(csv_path):
#     """
#     Loads the EPIC kitchens training CSV, reports on verb classes,
#     and splits it into new stratified train and validation CSVs.
#     """
#     print(f"Attempting to load dataset from: {csv_path}")
    
#     # Check if the file exists
#     if not os.path.exists(csv_path):
#         print(f"---")
#         print(f"Error: File not found at '{csv_path}'.")
#         print("Please update the 'INPUT_CSV_PATH' variable in the script")
#         print("to the correct location of your 'EPIC_100_train.csv' file.")
#         print(f"---")
#         return

#     try:
#         # Load the main training CSV
#         df = pd.read_csv(csv_path)
#         print(f"Successfully loaded {csv_path}. Found {len(df)} total entries.")

#         # Check if the class column exists
#         if CLASS_COLUMN not in df.columns:
#             print(f"---")
#             print(f"Error: The specified class column '{CLASS_COLUMN}' was not found.")
#             print(f"Available columns are: {list(df.columns)}")
#             print("Please update the 'CLASS_COLUMN' variable if needed.")
#             print(f"---")
#             return
            
#         # 1. Find the number of distinct verb classes
#         num_distinct_classes = df[CLASS_COLUMN].nunique()
#         print(f"\nFound {num_distinct_classes} distinct '{CLASS_COLUMN}' classes.")
        
#         print(f"Class distribution in original file:\n{df[CLASS_COLUMN].value_counts(normalize=True).head()}")


#         # 2. Split the data into training and validation sets
#         print(f"\nSplitting data into train ({1 - VALIDATION_SPLIT_SIZE:.0%}) and validation ({VALIDATION_SPLIT_SIZE:.0%})...")
        
#         # We use train_test_split on the entire dataframe
#         # 'stratify=df[CLASS_COLUMN]' ensures both splits have a similar
#         # distribution of verb classes as the original file.
#         train_df, val_df = train_test_split(
#             df,
#             test_size=VALIDATION_SPLIT_SIZE,
#             random_state=RANDOM_STATE,
#             stratify=df[CLASS_COLUMN]
#         )

#         print("Split complete.")
#         print(f"  New training set size: {len(train_df)} entries.")
#         print(f"  New validation set size: {len(val_df)} entries.")

#         # 3. Save the new CSV files
#         print(f"\nSaving new training file to '{NEW_TRAIN_CSV}'...")
#         # index=False avoids saving the old dataframe index as a new column
#         train_df.to_csv(NEW_TRAIN_CSV, index=False) 
        
#         print(f"Saving new validation file to '{NEW_VAL_CSV}'...")
#         val_df.to_csv(NEW_VAL_CSV, index=False)

#         print("\n---")
#         print("All done! You can now use these new CSV files for your 3D CNN.")
#         print(f"Validation class distribution:\n{val_df[CLASS_COLUMN].value_counts(normalize=True).head()}")
#         print("---")

#     except Exception as e:
#         print(f"\nAn unexpected error occurred: {e}")
#         print("Please check your file and script configuration.")

# if __name__ == "__main__":
#     split_dataset(INPUT_CSV_PATH)



import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# *** IMPORTANT: Update this path to point to your CSV file ***
INPUT_CSV_PATH = './EPIC-KITCHENS/annotations/EPIC_100_train.csv' 

# Column name containing the class labels (for stratification)
CLASS_COLUMN = 'verb_class' 

# Proportion of the data to use for the validation set (e.g., 0.2 = 20%)
VALIDATION_SPLIT_SIZE = 0.2

# Random state for reproducible splits
RANDOM_STATE = 42

# Output file names
NEW_TRAIN_CSV = 'epic_train_split.csv'
NEW_VAL_CSV = 'epic_validation_split.csv'
# ---------------------

def split_dataset(csv_path):
    """
    Loads the EPIC kitchens training CSV, reports on verb classes,
    filters out rare classes, and splits it into new stratified
    train and validation CSVs.
    """
    print(f"Attempting to load dataset from: {csv_path}")
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"---")
        print(f"Error: File not found at '{csv_path}'.")
        print("Please update the 'INPUT_CSV_PATH' variable in the script")
        print("to the correct location of your 'EPIC_100_train.csv' file.")
        print(f"---")
        return

    try:
        # Load the main training CSV
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}. Found {len(df)} total entries.")

        # Check if the class column exists
        if CLASS_COLUMN not in df.columns:
            print(f"---")
            print(f"Error: The specified class column '{CLASS_COLUMN}' was not found.")
            print(f"Available columns are: {list(df.columns)}")
            print("Please update the 'CLASS_COLUMN' variable if needed.")
            print(f"---")
            return
            
        # 1. Find the number of distinct verb classes
        num_distinct_classes = df[CLASS_COLUMN].nunique()
        print(f"\nFound {num_distinct_classes} distinct '{CLASS_COLUMN}' classes.")
        
        print(f"Class distribution in original file (top 5):\n{df[CLASS_COLUMN].value_counts(normalize=True).head()}")

        # --- FIX for Stratification Error ---
        # We must remove classes that have only 1 sample, as they cannot be split.
        print("\nChecking for rare classes (with only 1 sample)...")
        class_counts = df[CLASS_COLUMN].value_counts()
        # Get the class labels for classes that appear 1 time
        rare_classes = class_counts[class_counts == 1].index
        num_rare_classes = len(rare_classes)

        if num_rare_classes > 0:
            print(f"Found {num_rare_classes} classes with only 1 sample each.")
            print(f"These {num_rare_classes} rows will be REMOVED to allow for a clean stratified split.")
            
            # Keep only rows where the class is NOT in the rare_classes list
            # The ~ operator inverts the boolean mask
            df_filtered = df[~df[CLASS_COLUMN].isin(rare_classes)]
            
            print(f"Original dataset size: {len(df)} entries.")
            print(f"Filtered dataset size: {len(df_filtered)} entries.")
        else:
            print("No rare (1-sample) classes found. Proceeding with the full dataset.")
            # Create a copy to avoid potential SettingWithCopyWarning
            df_filtered = df.copy() 
        # ------------------------------------

        # 2. Split the *filtered* data into training and validation sets
        print(f"\nSplitting data into train ({1 - VALIDATION_SPLIT_SIZE:.0%}) and validation ({VALIDATION_SPLIT_SIZE:.0%})...")
        
        # Check if filtered dataframe is empty (edge case)
        if df_filtered.empty:
            print("Error: The filtered dataset is empty. Cannot proceed with split.")
            return

        # 'stratify=df_filtered[CLASS_COLUMN]' ensures both splits have a similar
        # distribution of verb classes.
        train_df, val_df = train_test_split(
            df_filtered,
            test_size=VALIDATION_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=df_filtered[CLASS_COLUMN]
        )

        print("Split complete.")
        print(f"  New training set size: {len(train_df)} entries.")
        print(f"  New validation set size: {len(val_df)} entries.")

        # 3. Save the new CSV files
        print(f"\nSaving new training file to '{NEW_TRAIN_CSV}'...")
        # index=False avoids saving the old dataframe index as a new column
        train_df.to_csv(NEW_TRAIN_CSV, index=False) 
        
        print(f"Saving new validation file to '{NEW_VAL_CSV}'...")
        val_df.to_csv(NEW_VAL_CSV, index=False)

        print("\n---")
        print("All done! You can now use these new CSV files for your 3D CNN.")
        print(f"Validation class distribution (top 5):\n{val_df[CLASS_COLUMN].value_counts(normalize=True).head()}")
        print("---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your file and script configuration.")

if __name__ == "__main__":
    split_dataset(INPUT_CSV_PATH)