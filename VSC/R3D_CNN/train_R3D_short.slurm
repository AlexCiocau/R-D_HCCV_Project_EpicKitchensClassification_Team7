#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=Train_R3D_short

# VSC cluster
module load cluster/wice/interactive

# --- Environment Setup ---
echo "--- Starting VSC Job ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Activate venv and define filepath
source venv/bin/activate
cd $VSC_DATA/RD_Project

# Run actual python script
python main.py

echo "Job finished."