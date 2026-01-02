#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100_debug     # CRITICAL: The fast "debug" queue
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18             # 1/4 node (standard ratio for 1 GPU)
#SBATCH --mem=60G                      # 60GB is the safe max for debug nodes
#SBATCH --time=00:30:00                # 30 min limit (Debug max is 1 hour)
#SBATCH --job-name=DEBUG_ConvNext
#SBATCH --output=debug_log_%j.out
#SBATCH --error=debug_log_%j.err

# --- 1. Load Modules ---
# Loading the interactive module is standard for wICE
module load cluster/wice/interactive

# --- 2. Activate Environment ---
source venv/bin/activate

# --- 3. Move to Project Directory ---
cd $VSC_DATA/RD_Project

# --- 4. Optimization Flags ---
# Even for debugging, we want to simulate the real environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export cv_NUM_THREADS=1

# --- 5. Diagnostic Info ---
echo "DEBUG SESSION STARTED"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --- 6. Execution ---
# Run the sanity check script we created earlier
# If you haven't created it yet, replace this line with your main script
# but add a "--dry_run" flag or similar if your code supports it.
python sanity_check.py