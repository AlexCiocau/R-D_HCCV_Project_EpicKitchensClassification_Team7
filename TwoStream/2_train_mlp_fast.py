import torch
import torch.nn as nn
import os
import glob
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
 
# ---------------- CONFIGURATION ----------------
FEATURE_DIR = os.path.expandvars("$VSC_SCRATCH/feature_vectors_v2") # Must match output of Script 1
NUM_VERBS = 97  
NUM_NOUNS = 300 
BATCH_SIZE = 256 # Can be very high now
LR = 1e-4
EPOCHS = 50
DROPOUT = 0.5
 
# ---------------- 1. DATASET FOR VECTORS ----------------
class PrecomputedFeatureDataset(Dataset):
    def __init__(self, mode='train'):
        self.files = glob.glob(os.path.join(FEATURE_DIR, mode, "*.pt"))
        self.data_cache = []
        # Load all into RAM (It's small enough, usually <1GB)
        print(f"Loading {mode} features into RAM...")
        for f in tqdm(self.files):
            try:
                d = torch.load(f)
                b_size = d['feat_motion'].shape[0]
                for i in range(b_size):
                    self.data_cache.append({
                        'motion': d['feat_motion'][i],
                        'static': d['feat_static'][i],
                        'verb': d['verb'][i],
                        'noun': d['noun'][i]
                    })
            except Exception as e:
                print(f"Error loading {f}: {e}")
 
    def __len__(self):
        return len(self.data_cache)
 
    def __getitem__(self, idx):
        item = self.data_cache[idx]
        return item['motion'].float(), item['static'].float(), item['verb'], item['noun']
 
# ---------------- 2. MLP MODEL (Matches your V3 Logic) ----------------
class FusionMLP(nn.Module):
    def __init__(self, num_verbs, num_nouns, dropout=0.5):
        super().__init__()
        # Exact structure from TwoStream_Advanced_v3.py
        self.project_x3d = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.project_cx = nn.Sequential(
            nn.Linear(768, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 1024), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.verb_fc = nn.Linear(1024, num_verbs)
        self.noun_fc = nn.Linear(1024, num_nouns)
 
    def forward(self, motion, static):
        p_motion = self.project_x3d(motion)
        p_static = self.project_cx(static)
        combined = torch.cat((p_motion, p_static), dim=1)
        fused = self.fusion_mlp(combined)
        return self.verb_fc(fused), self.noun_fc(fused)
 
# ---------------- 3. TRAINING LOOP ----------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_v = 0; correct_n = 0; correct_a = 0
    total = 0
    with torch.no_grad():
        for m, s, v, n in loader:
            m, s, v, n = m.to(device), s.to(device), v.to(device), n.to(device)
            v_logits, n_logits = model(m, s)
            loss = criterion(v_logits, v) + criterion(n_logits, n)
            total_loss += loss.item()
            _, vp = v_logits.max(1)
            _, np = n_logits.max(1)
            correct_v += (vp == v).sum().item()
            correct_n += (np == n).sum().item()
            correct_a += ((vp == v) & (np == n)).sum().item()
            total += v.size(0)
    return total_loss/len(loader), correct_v/total, correct_n/total, correct_a/total
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="Epic-MLP-Fast")
 
    # Load Data
    train_ds = PrecomputedFeatureDataset('train')
    val_ds = PrecomputedFeatureDataset('val')
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
 
    # Init Model
    model = FusionMLP(NUM_VERBS, NUM_NOUNS, dropout=DROPOUT).to(device)
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-1)
    criterion = nn.CrossEntropyLoss() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
 
    print("Starting Fast Training...")
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for m, s, v, n in loop:
            m, s, v, n = m.to(device), s.to(device), v.to(device), n.to(device)
            v_out, n_out = model(m, s)
            loss = criterion(v_out, v) + criterion(n_out, n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
 
        val_loss, acc_v, acc_n, acc_a = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Ep {epoch+1}: Loss {val_loss:.4f} | V {acc_v:.1%} | N {acc_n:.1%} | Act {acc_a:.1%}")
        wandb.log({"val_loss": val_loss, "acc_verb": acc_v, "acc_noun": acc_n, "acc_action": acc_a})
        if acc_a > best_acc:
            best_acc = acc_a
            torch.save(model.state_dict(), "mlp_best_weights.pth")
            print("Saved Best MLP Weights.")