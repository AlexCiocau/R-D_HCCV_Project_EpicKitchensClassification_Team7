#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --job-name=Script_Test

# VSC cluster
module load cluster/wice/interactive

# Log is stored locally
# export WANDB_MODE=offline

# Activate venv and define filepath
source venv/bin/activate
cd $VSC_DATA/RD_Project

# Run actual python script
python main.py