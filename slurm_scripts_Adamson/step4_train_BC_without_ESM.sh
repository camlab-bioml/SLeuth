#!/bin/bash
#SBATCH --job-name=BC_no_ESM
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/step4_BC_no_ESM_%j.out
#SBATCH --error=logs/step4_BC_no_ESM_%j.err

echo "=================================================="
echo "Step 4: Train BC Dataset WITHOUT ESM"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

# Load modules
module purge
module load gnu14 openmpi5 EasyBuild cmake openblas fftw
source /opt/rh/gcc-toolset-13/enable

# Python path explicitly set - no conda activation needed

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# GPU is automatically set by SLURM via --gres=gpu:1

# Change to code directory
cd ../code_Adamson

# Create output directory
mkdir -p ../outputs_Adamson/BC_without_ESM

# Run BC training without ESM
echo "Starting BC training without ESM embeddings..."
echo "Configuration (matching TensorFlow original):"
echo "  - Epochs: 200"
echo "  - Hidden: 128/64"
echo "  - Dropout: 0.3"
echo "  - Alpha: 0.5, Beta: 2.0, Coe: 1.0"
$PYTHON_PATH train_bc.py \
	--dataset BC \
	--epochs 200 \
	--eva_epochs 100 \
	--cv_folds 5 \
	--hidden1 128 \
	--hidden2 64 \
	--dropout 0.3 \
	--alpha 0.5 \
	--beta 2.0 \
	--coe 1.0 \
	--output_dir ../outputs_Adamson/BC_without_ESM

echo "=================================================="
echo "BC training completed at: $(date)"
echo "Results saved to: ../outputs_Adamson/BC_without_ESM/"
echo "=================================================="
