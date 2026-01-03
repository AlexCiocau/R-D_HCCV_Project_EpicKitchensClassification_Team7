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