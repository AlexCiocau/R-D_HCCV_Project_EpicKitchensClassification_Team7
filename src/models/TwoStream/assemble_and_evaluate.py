import torch
import torch.nn as nn
from tqdm import tqdm
import os
import torchvision.models as models

# --- IMPORT DATASET ---
from EpicKitchensDataset_vMulti_weighted_v2 import EpicKitchensDataset

# --- CONFIGURATION ---
X3D_WEIGHTS = "x3d_tmpaug_model_4.pth"
CONVNEXT_WEIGHTS = "convnext_tiny_noun_retrained.pth" # Your new retrained noun model
MLP_WEIGHTS = "mlp_retrained.pth"                     # Your new retrained MLP
OUTPUT_FILENAME = "twostream_retrained.pth"

BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_FRAMES = 16

# --- SPEED OPTIMIZATION ---
torch.set_float32_matmul_precision('high')
try:
    from torchvision.transforms import v2
except ImportError:
    print("WARNING: torchvision.transforms.v2 not found.")
    import torchvision.transforms as v2

# ------------------------------- HELPERS ----------------------------------
def clean_state_dict(state_dict):
    """Removes 'module.' prefix if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, label_smoothing=0.1):
        super().__init__() 
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ------------------------------- AUGMENTOR --------------------------------
class GPUAugmentor(nn.Module):
    def __init__(self, training=True):
        super().__init__()
        self.training = training
        # ImageNet Norm
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
        
        if training:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True), 
                v2.RandomCrop(224),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ])
        else:
            self.geo_transforms = v2.Compose([
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
            ])

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x = self.geo_transforms(x)
        x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)
        x = (x - self.mean) / self.std
        return x

# ------------------------------- MODEL ------------------------------------
class TwoStreamModel(nn.Module):
    def __init__(self, num_verbs, num_nouns, x3d_path, convnext_path):
        super().__init__()
        
        # --- 1. X3D Stream ---
        print(f"Loading X3D Stream...")
        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
        self.x3d.blocks[5].proj = nn.Linear(self.x3d.blocks[5].proj.in_features, 97)
        
        if os.path.exists(x3d_path):
            print(f"  -> Loading weights from {x3d_path}")
            sd = torch.load(x3d_path, map_location='cpu')
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = clean_state_dict(sd)
            # Filter head
            sd = {k: v for k, v in sd.items() if 'blocks.5.proj' not in k}
            self.x3d.load_state_dict(sd, strict=False)
        else:
            print(f"  -> WARNING: {x3d_path} not found. Using random init.")
            
        self.x3d.blocks[5].proj = nn.Identity()

        # --- 2. ConvNeXt Stream ---
        print(f"Loading ConvNeXt Stream...")
        self.convnext = models.convnext_tiny(weights=None) 
        cx_dim = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(cx_dim, 300))
        
        if os.path.exists(convnext_path):
            print(f"  -> Loading weights from {convnext_path}")
            sd = torch.load(convnext_path, map_location='cpu')
            if 'model_state_dict' in sd: sd = sd['model_state_dict']
            sd = clean_state_dict(sd)
            # Filter head
            sd = {k: v for k, v in sd.items() if 'classifier.2' not in k}
            self.convnext.load_state_dict(sd, strict=False)
        else:
            print(f"  -> WARNING: {convnext_path} not found. Using random init.")
            
        self.convnext.classifier[2] = nn.Identity()

        # --- 3. Fusion MLP ---
        self.project_x3d = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.project_cx = nn.Sequential(nn.Linear(768, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5)
        )
        self.verb_fc = nn.Linear(1024, num_verbs)
        self.noun_fc = nn.Linear(1024, num_nouns)

    def forward(self, x):
        feat_motion = self.x3d(x) 
        middle_idx = x.shape[2] // 2
        feat_static = self.convnext(x[:, :, middle_idx, :, :]) 
        
        p_motion = self.project_x3d(feat_motion)
        p_static = self.project_cx(feat_static)
        combined = torch.cat((p_motion, p_static), dim=1) 
        fused = self.fusion_mlp(combined)                 
        return self.verb_fc(fused), self.noun_fc(fused)

# ------------------------------- EVALUATION -------------------------------
def evaluate_model(model, dataloader, criterion, device, augmentor):
    model.eval()
    augmentor.eval()
    total_loss = 0.0
    correct_v = 0; correct_n = 0; correct_a = 0
    total = 0
    
    with torch.no_grad():
        for video, labels in tqdm(dataloader, desc="Sanity Check"):
            video = video.to(device, non_blocking=True)
            v_lbl = labels['verb'].to(device)
            n_lbl = labels['noun'].to(device)
            
            video = augmentor(video)
            v_logits, n_logits = model(video)
            
            loss = criterion(v_logits, v_lbl) + criterion(n_logits, n_lbl)
            total_loss += loss.item()
            
            v_pred = v_logits.argmax(1)
            n_pred = n_logits.argmax(1)
            
            correct_v += (v_pred == v_lbl).sum().item()
            correct_n += (n_pred == n_lbl).sum().item()
            correct_a += ((v_pred == v_lbl) & (n_pred == n_lbl)).sum().item()
            total += v_lbl.size(0)

    return (total_loss/len(dataloader), 
            100*correct_v/total, 100*correct_n/total, 100*correct_a/total)

# ------------------------------- MAIN -------------------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # 1. Dataset (Validation Only)
    print("Initializing Dataset...")
    val_ds = EpicKitchensDataset('./EPIC-KITCHENS', NUM_FRAMES, testing=True)
    val_loader = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 2. Assemble Model
    print("\n--- ASSEMBLING MODEL ---")
    model = TwoStreamModel(val_ds.num_verb_classes, 300, X3D_WEIGHTS, CONVNEXT_WEIGHTS).to(device)
    
    # 3. Load MLP Weights
    if os.path.exists(MLP_WEIGHTS):
        print(f"Loading MLP Fusion weights from {MLP_WEIGHTS}...")
        sd = torch.load(MLP_WEIGHTS, map_location='cpu')
        sd = clean_state_dict(sd)
        
        # This loads only the keys present in mlp_retrained (project_x3d, fusion_mlp, etc.)
        # and ignores the backbones (which we already loaded in __init__)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        
        if len(unexpected) > 0:
            print(f"[WARNING] Unexpected keys in MLP file: {unexpected}")
        else:
            print("MLP weights integrated successfully.")
    else:
        print(f"[ERROR] MLP weights {MLP_WEIGHTS} not found!")
        exit()
        
    # 4. Save Combined Model
    print(f"\nSaving full model to {OUTPUT_FILENAME}...")
    torch.save(model.state_dict(), OUTPUT_FILENAME)
    print("Saved.")
    
    # 5. Sanity Check
    print("\n--- RUNNING SANITY CHECK ---")
    augmentor = GPUAugmentor(training=False).to(device)
    criterion = FocalLoss()
    
    loss, v_acc, n_acc, a_acc = evaluate_model(model, val_loader, criterion, device, augmentor)
    
    print("\n" + "="*30)
    print(f"FINAL ASSEMBLED RESULTS")
    print("="*30)
    print(f"Verb Accuracy:   {v_acc:.2f}%")
    print(f"Noun Accuracy:   {n_acc:.2f}%")
    print(f"Action Accuracy: {a_acc:.2f}%")
    print("="*30)