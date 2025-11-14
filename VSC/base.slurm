#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --job-name=RD_Project

module load cluster/wice/interactive

source venv/bin/activate

cd $VSC_DATA/RD_Project

# python main.py
python preprocess.py
# You can use the arguments to change the comportment of the script.