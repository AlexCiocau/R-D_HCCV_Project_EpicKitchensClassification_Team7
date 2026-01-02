import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import pandas as pd
import torchvision.models as models

# IMPORTANT: Import the dataset class that knows your custom split logic
from EpicKitchensDataset_vMulti_weighted_v2 import EpicKitchensDataset

# ---------------- CONFIGURATION ----------------
CHECKPOINT_PATH = "twostream_ultimate_best.pth" 
SUBMISSION_FILENAME = "submission.pt"

# Verification Config (Your Custom Split)
DATA_ROOT = './EPIC-KITCHENS' 

# Submission Config (Codabench Split)
CODABENCH_ANNOTATIONS = "./EPIC-KITCHENS/annotations/EPIC_100_validation.csv"
CODABENCH_TENSORS = os.path.expandvars("$VSC_SCRATCH/x3d_codabench_val")
if not os.path.exists(CODABENCH_TENSORS):
    CODABENCH_TENSORS = "/scratch/leuven/380/vsc38040/x3d_codabench_val"

BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_FRAMES = 16

# --- HARDCODED CLASS COUNTS (FIX FOR SIZE MISMATCH) ---
NUM_VERBS_FIXED = 97
NUM_NOUNS_FIXED = 300

# ---------------- 1. MODEL DEFINITION ----------------
class TwoStreamModel(nn.Module):
    def __init__(self, num_verbs, num_nouns):
        super().__init__()
        # X3D
        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
        self.x3d.blocks[5].proj = nn.Identity()
        
        # ConvNeXt
        self.convnext = models.convnext_tiny(weights=None) 
        self.convnext.classifier[2] = nn.Identity()

        # Fusion
        self.project_x3d = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.project_cx = nn.Sequential(nn.Linear(768, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(2048, 1024), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.verb_fc = nn.Linear(1024, num_verbs)
        self.noun_fc = nn.Linear(1024, num_nouns)

    def forward(self, x):
        feat_m = self.x3d(x)
        feat_s = self.convnext(x[:, :, x.shape[2]//2, :, :])
        fused = self.fusion_mlp(torch.cat((self.project_x3d(feat_m), self.project_cx(feat_s)), dim=1))
        return self.verb_fc(fused), self.noun_fc(fused)

# ---------------- 2. AUGMENTOR (VALIDATION) ----------------
class ValAugmentor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torchvision.transforms.v2 as v2
        except ImportError:
            import torchvision.transforms as v2
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
        
        self.transforms = v2.Compose([
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
        ])
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x = self.transforms(x)
        x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)
        return (x - self.mean) / self.std

# ---------------- 3. SUBMISSION DATASET ----------------
class SubmissionDataset(Dataset):
    def __init__(self, annotations_file, tensor_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.tensor_dir = tensor_dir
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        if not os.path.exists(self.tensor_dir):
            raise RuntimeError(f"Tensor dir not found: {self.tensor_dir}")
        existing_files = set(os.listdir(self.tensor_dir))
        for _, row in self.annotations.iterrows():
            segment_uid = row['narration_id']
            clip_idx = 0
            while True:
                fname = f"{segment_uid}_clip{clip_idx}.pt"
                if fname in existing_files:
                    samples.append((os.path.join(self.tensor_dir, fname), segment_uid))
                    clip_idx += 1
                else:
                    break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, narration_id = self.samples[idx]
        try:
            video = torch.load(path, map_location='cpu')
            if video.dtype != torch.float32: video = video.float()
            if video.max() > 1.0: video = video / 255.0
            return video, narration_id
        except:
            return torch.zeros(3, 16, 224, 224), narration_id

# ---------------- 4. MAIN SCRIPT ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # --- PHASE 1: VERIFICATION (Custom Split) ---
    print("\n" + "="*40)
    print("PHASE 1: VERIFYING ON CUSTOM VALIDATION SPLIT")
    print("="*40)
    
    verify_ds = EpicKitchensDataset(DATA_ROOT, NUM_FRAMES, testing=True)
    verify_loader = DataLoader(verify_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # FIX: Use hardcoded constants instead of dataset attributes
    print(f"Initializing model with Fixed Classes: Verbs={NUM_VERBS_FIXED}, Nouns={NUM_NOUNS_FIXED}")
    model = TwoStreamModel(NUM_VERBS_FIXED, NUM_NOUNS_FIXED).to(device)
    augmentor = ValAugmentor().to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        sd = torch.load(CHECKPOINT_PATH, map_location='cpu')
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        # Now shapes should match
        model.load_state_dict(sd, strict=False)
        model.eval()
    else:
        print(f"CRITICAL: Checkpoint {CHECKPOINT_PATH} not found.")
        return

    correct_v, correct_n, correct_a = 0, 0, 0
    total = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for video, labels in tqdm(verify_loader):
            video = video.to(device)
            video = augmentor(video)
            
            v_labels = labels['verb'].to(device)
            n_labels = labels['noun'].to(device)
            
            v_out, n_out = model(video)
            
            v_pred = v_out.argmax(1)
            n_pred = n_out.argmax(1)
            
            correct_v += (v_pred == v_labels).sum().item()
            correct_n += (n_pred == n_labels).sum().item()
            correct_a += ((v_pred == v_labels) & (n_pred == n_labels)).sum().item()
            total += v_labels.size(0)

    acc_v = 100 * correct_v / total
    acc_n = 100 * correct_n / total
    acc_a = 100 * correct_a / total
    
    print("\n" + "-"*30)
    print(f"CUSTOM VALIDATION RESULTS")
    print("-"*30)
    print(f"Verb Accuracy:   {acc_v:.2f}%")
    print(f"Noun Accuracy:   {acc_n:.2f}%")
    print(f"Action Accuracy: {acc_a:.2f}%")
    print("-"*30)
    
    if acc_a < 25.0:
        print("\n[WARNING] Action accuracy is unexpectedly low (<25%).")
        print("Skipping submission generation.")
        return
    else:
        print("\n[SUCCESS] Model verified. Proceeding to submission generation.")

    # --- PHASE 2: SUBMISSION (Codabench Split) ---
    print("\n" + "="*40)
    print("PHASE 2: GENERATING SUBMISSION (CODABENCH)")
    print("="*40)
    
    sub_ds = SubmissionDataset(CODABENCH_ANNOTATIONS, CODABENCH_TENSORS)
    sub_loader = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    aggregated_results = {}
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for video, nids in tqdm(sub_loader):
            video = video.to(device)
            video = augmentor(video)
            
            v_logits, n_logits = model(video)
            
            v_logits = v_logits.cpu()
            n_logits = n_logits.cpu()
            
            for i, nid in enumerate(nids):
                if nid not in aggregated_results:
                    aggregated_results[nid] = {'v': [], 'n': []}
                aggregated_results[nid]['v'].append(v_logits[i])
                aggregated_results[nid]['n'].append(n_logits[i])
                
    print("Aggregating and applying Softmax...")
    final_submission = []
    
    df = pd.read_csv(CODABENCH_ANNOTATIONS)
    for _, row in df.iterrows():
        nid = row['narration_id']
        
        if nid in aggregated_results:
            v_avg = torch.stack(aggregated_results[nid]['v']).mean(dim=0)
            n_avg = torch.stack(aggregated_results[nid]['n']).mean(dim=0)
            
            # Softmax
            v_prob = torch.softmax(v_avg, dim=0)
            n_prob = torch.softmax(n_avg, dim=0)
            
            final_submission.append({
                'narration_id': str(nid),
                'verb_output': v_prob,
                'noun_output': n_prob
            })
        else:
            final_submission.append({
                'narration_id': str(nid),
                'verb_output': torch.ones(97)/97.0,
                'noun_output': torch.ones(300)/300.0
            })
            
    print(f"Saving to {SUBMISSION_FILENAME}...")
    torch.save(final_submission, SUBMISSION_FILENAME)
    print("Done.")

if __name__ == "__main__":
    main()