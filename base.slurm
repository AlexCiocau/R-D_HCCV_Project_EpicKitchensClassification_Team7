#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --job-name=RD_Project

module load cluster/wice/interactive

source venv/bin/activate

cd $VSC_DATA/RD_Project

python main.py
# You can use the arguments to change the comportment of the script.