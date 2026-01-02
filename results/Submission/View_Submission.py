# import torch
# import sys

# FILENAME = 'submission.pt'

# def inspect():
#     print(f"--- Inspecting {FILENAME} ---")
    
#     try:
#         data = torch.load(FILENAME, weights_only=False) # weights_only=False needed for lists
#     except Exception as e:
#         print(f"❌ FATAL: Could not load file using torch.load(). Error: {e}")
#         return

#     # 1. Check Data Type
#     if not isinstance(data, list):
#         print(f"❌ FAIL: Expected a list, got {type(data)}")
#         return
#     print(f"✅ PASS: File is a list.")

#     # 2. Check Length
#     print(f"ℹ️ INFO: Total predictions found: {len(data)}")
#     if len(data) == 0:
#         print("❌ FAIL: List is empty.")
#         return

#     # 3. Inspect First Element Structure
#     first_item = data[0]
#     required_keys = {'narration_id', 'verb_output', 'noun_output'}
#     if not isinstance(first_item, dict):
#         print(f"❌ FAIL: Items in list must be dicts, got {type(first_item)}")
#         return
    
#     keys_present = set(first_item.keys())
#     if not required_keys.issubset(keys_present):
#         print(f"❌ FAIL: Missing keys. Found {keys_present}, need {required_keys}")
#         return
#     print(f"✅ PASS: Dictionary structure is correct.")

#     # 4. Check Shapes & Types
#     verb_shape = first_item['verb_output'].shape
#     noun_shape = first_item['noun_output'].shape
    
#     print(f"ℹ️ INFO: Verb Shape: {verb_shape} (Expected: torch.Size([97]))")
#     print(f"ℹ️ INFO: Noun Shape: {noun_shape} (Expected: torch.Size([300]))")

#     if verb_shape[0] != 97:
#         print(f"❌ FAIL: Verb output must be length 97.")
#     else:
#         print(f"✅ PASS: Verb dimensions correct.")

#     if noun_shape[0] != 300:
#         print(f"❌ FAIL: Noun output must be length 300.")
#     else:
#         print(f"✅ PASS: Noun dimensions correct.")

#     # 5. CRITICAL: Check IDs (Validation vs Test)
#     print("\n--- 🔍 ID CHECK (The most important part) ---")
#     print("First 5 Narration IDs in file:")
#     for i in range(min(5, len(data))):
#         print(f"  {data[i]['narration_id']}")

#     id_example = data[0]['narration_id']
#     if "P01_11" in id_example or "P01_01" in id_example:
#         print("\n⚠️ WARNING: These look like VALIDATION or TRAIN IDs (e.g. P01_11).")
#         print("   If you are submitting to the 'Test' leaderboard, you will get 0.0 accuracy.")
#         print("   You need IDs like 'P01_101', 'P01_102'...")
#     else:
#         print("\n✅ LOOKS GOOD: IDs do not match standard validation start patterns.")

#     # 6. Check for NaNs
#     has_nan = torch.isnan(first_item['verb_output']).any() or torch.isnan(first_item['noun_output']).any()
#     if has_nan:
#         print("❌ FAIL: Found NaN (Not a Number) in predictions.")
#     else:
#         print("✅ PASS: No NaNs detected in first sample.")

# if __name__ == '__main__':
#     inspect()



import torch

FILENAME = 'submission.pt'

def print_all():
    print(f"--- Reading {FILENAME} ---")
    
    try:
        data = torch.load(FILENAME, weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total Predictions: {len(data)}")
    print(f"{'IDX':<6} | {'NARRATION_ID':<20} | {'VERB':<6} {'(PROB)':<8} | {'NOUN':<6} {'(PROB)':<8}")
    print("-" * 70)

    for i, entry in enumerate(data):
        narr_id = entry['narration_id']
        
        # Get class index and probability
        verb_idx = entry['verb_output'].argmax().item()
        verb_prob = entry['verb_output'].max().item()
        
        noun_idx = entry['noun_output'].argmax().item()
        noun_prob = entry['noun_output'].max().item()
        
        print(f"{i:<6} | {narr_id:<20} | {verb_idx:<6} {verb_prob:.1%}   | {noun_idx:<6} {noun_prob:.1%}")

    print("-" * 70)
    print("End of file.")

if __name__ == '__main__':
    print_all()