#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=Generate TmpAug Tensors

module load cluster/wice/interactive

source venv/bin/activate

cd $VSC_DATA/RD_Project

# python main.py
python preprocess_wtmpaug.py

