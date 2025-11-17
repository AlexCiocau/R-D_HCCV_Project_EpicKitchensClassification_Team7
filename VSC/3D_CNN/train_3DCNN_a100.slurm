#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
#SBATCH --time=8:00:00
#SBATCH --job-name=Train_3DCNN

# VSC cluster
module load cluster/wice/interactive

# Log is stored locally
# export WANDB_MODE=offline

# Activate venv and define filepath
source venv/bin/activate
cd $VSC_DATA/RD_Project

# Run actual python script
python main.py