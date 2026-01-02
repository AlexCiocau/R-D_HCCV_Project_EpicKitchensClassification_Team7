#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=Gen_Missing__x3d_tensors

module load cluster/wice/interactive

source venv/bin/activate

cd $VSC_DATA/RD_Project

# python tensor_gen_x3d_fixed.py
# python fix_validation_csv.py
python generate_submission.py
