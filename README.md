# EpicKitchens Video Classification (Team 7)

**Authors:** Joren Van den Vondel, Maarten Luypaert, Alexandru-Mihai Ciocău  
**Course:** R-D HCCV Project  

## Project Overview
This repository contains the implementation of a robust heterogeneous **Two-Stream AI model** for end-to-end egocentric action recognition on the [EPIC-KITCHENS-100 dataset](https://epic-kitchens.github.io/2025).

The goal is to predict an **Action** defined as a combination of a **Verb** (Motion) and a **Noun** (Object). Our approach explicitly disentangles these two aspects:
* **Spatial Stream (ConvNeXt-Tiny):** Captures static appearance and object details (Noun focus).
* **Temporal Stream (X3D):** Captures motion dynamics and temporal evolution (Verb focus).
* **Fusion Strategy(TwoStream):** We use a late fusion MLP to combine features from both backbones.

**Key Innovations:**
* **Disentanglement:** Separates temporal dynamics from static appearance.
* **Efficient Fusion:** Concatenates an X3D motion backbone with a ConvNeXt spatial backbone.
* **Robust Training:** Utilizes Hybrid Weighted Random Sampling and GPU-accelerated augmentation to handle severe class imbalance (long-tail distribution).

## Project Structure
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
├── .gitattributes          
├── .gitignore              
├── requirements.txt        
└── README.md              

## Step-by-Step Guide: Training & Evaluation

This section outlines the complete pipeline for reproducing our results. The training process is modular: each backbone (Spatial and Temporal) is trained independently before being fused via the MLP and finally assembled for end-to-end evaluation.

The workflow consists of four main stages: **Dataset Splitting**, **Tensor Preprocessing**, **Model Training**, and **Submission Generation**.

### 1. Generating Custom Dataset Splits
The EPIC-KITCHENS-100 dataset provides standard train/val splits. However, to address class imbalance and specific modality requirements, we generate custom splits for each branch.

* **Reference Files:** The actual splits used for our trained models are available in `scripts/Dataset splits/Current split`.
* **X3D Split (Verb-Rarity):**
  Run `create_splits.py` to generate a split based on verb rarity. This outputs two CSV files used to balance the training and validation examples for the temporal stream.
* **ConvNeXt Split (Participant-Based):**
  Run `create_participant_split.py` to generate a split based on participant IDs, ensuring the model generalizes across different subjects.

### 2. Tensor Preprocessing
To optimize training speed and reduce computational overhead during epochs, we precompute input tensors with baked-in transformations.

* **Action:** Run the `tensor_gen_x3d.py` script.
* **Outcome:** This generates the pre-processed tensors required for the X3D branch, effectively trading storage space for reduced training complexity.

### 3. Training the Model
The model components are trained in a specific sequence: backbones first, followed by the fusion layer, and finally the model assembly.

#### 3.1 Train the X3D Branch (Temporal)
Run `X3D.py`.
* **Process:** Initializes the X3D model with weights pretrained on **Kinetics400**.
* **Output:** The model is fine-tuned on EPIC-KITCHENS to learn temporal dynamics, saving the final weights for the temporal branch.

#### 3.2 Train the ConvNeXt Branch (Spatial)
Run `ConvNeXtTiny.py`.
* **Output:** Trains the spatial backbone and saves the weights for the ConvNeXt-Tiny model.

#### 3.3 Train the MLP Fusion
Locate the scripts in `src/models/TwoStream/` and execute them in the following order:
1. `1_extract_features.py`: Extracts feature vectors from both trained backbones.
2. `2_train_mlp_fast.py`: Trains the MLP to fuse the spatial and temporal features.
3. `AfterTraining.py`: Finalizes the fusion module configuration.

#### 3.4 Model Assembly & Final Evaluation
Run `assemble_and_evaluate.py`.
* **Function:** Loads the weights from steps 3.1, 3.2, and 3.3 to assemble the full Two-Stream model.
* **Validation:** Performs an evaluation run to confirm the assembly is correct and outputs the final model weights.

### 4. Generating a Submission
To generate inference results for the leaderboard:

1. **Inference:** Run `generate_submission.py` to create the `.pth` file containing predictions on the test set.
2. **Verification:** Use `View_Submission.py` to inspect the contents of the generated submission file.

