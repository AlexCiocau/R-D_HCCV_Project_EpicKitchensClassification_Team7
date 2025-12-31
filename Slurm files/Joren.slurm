#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --cluster=wice
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:17:00
#SBATCH --job-name=Train_Advanced_TwoStream_v3
#SBATCH --output=train_log_%j.out
#SBATCH --error=train_log_%j.err

# 1. Load Modules
# On WICE GPU partition, we usually load the CUDA module implicitly via the environment
# But loading the interactive module is standard for your workflow.
module load cluster/wice/interactive

# 2. Activate Environment
source venv/bin/activate

# 3. Move to Project Directory
cd $VSC_DATA/RD_Project

# 4. PERFORMANCE TUNING (Critical for A100)
# The A100 is so fast that the CPU often becomes the bottleneck (decoding video).
# These flags force the CPU workers to focus on loading data, not math.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export cv_NUM_THREADS=1

# 5. Debugging Info
echo "Job running on node: $SLURMD_NODENAME"
echo "GPU allocated:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 6. Run Training
# python ResNet50_optimized.py
# python ResNet50_advanced.py
# python ConvNextTiny.py
# python TwoStream_Advanced_v2.py
# python debug_advanced_twostream.py
# python TwoStream_Advanced_v3.py
# python AfterTraining_ValidationOnly.py
# python AfterTraining.py
python generate_submission.py