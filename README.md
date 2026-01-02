# EpicKitchens Video Classification (Team 7)

**Authors:** Joren Van den Vondel, Maarten Luypaert, Alexandru-Mihai Ciocău  
**Course:** R-D HCCV Project  

## Project Overview
This repository contains the implementation of a robust heterogeneous **Two-Stream AI model** for end-to-end egocentric action recognition on the [EPIC-KITCHENS-100 dataset](https://epic-kitchens.com/).

The goal is to predict an **Action** defined as a combination of a **Verb** (Motion) and a **Noun** (Object). Our approach explicitly disentangles these two aspects:
* **Spatial Stream (ConvNeXt-Tiny):** Captures static appearance and object details (Noun focus).
* **Temporal Stream (X3D):** Captures motion dynamics and temporal evolution (Verb focus).
* **Fusion Strategy(TwoStream):** We use a late fusion MLP to combine features from both backbones.

**Key Innovations:**
* **Disentanglement:** Separates temporal dynamics from static appearance.
* **Efficient Fusion:** Concatenates an X3D motion backbone with a ConvNeXt spatial backbone.
* **Robust Training:** Utilizes Hybrid Weighted Random Sampling and GPU-accelerated augmentation to handle severe class imbalance (long-tail distribution).

## 📂 Project Structure
The repository is organized to ensure reproducibility and modularity, separating data, scripts, and source code.

```text
├── src/                    # Source code package
│   ├── models/             # Model architectures
│   │   ├── ConvNeXt-Tiny/  # Spatial backbone implementation
│   │   ├── X3D/            # Temporal backbone implementation
│   │   └── TwoStream/      # Fusion logic (feature extraction & MLP)
│   └── utils/              # Helper functions (early_stopping.py)
├── scripts/                
│   ├── Dataset splits/     # Split generation (create_noun_split.py, create_splits.py)
│   ├── Tensor generation/  
│   └── Slurm files/        
├── results/                
│   ├── Saved models (weights)/  # Saved model checkpoints (.pth)
│   └── submissions/        # Generated submission files for the leaderboard
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation