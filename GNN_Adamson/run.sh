#!/bin/bash
#SBATCH --job-name=gnn_adamson
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/gnn_adamson_%j.out
#SBATCH --error=logs/gnn_adamson_%j.err

# Exit on error
set -e

echo "=================================================="
echo "GNN Adamson Training"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "=================================================="

# Load modules
module purge
module load gnu14 openmpi5 EasyBuild cmake openblas fftw
source /opt/rh/gcc-toolset-13/enable

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# Create logs directory
mkdir -p logs

# Run preprocessing first
echo "Running preprocessing..."
$PYTHON_PATH preprocess.py

# Run training: PPI for conv, gamma for target, go_bp+go_cc as edge features
echo "Running training..."
$PYTHON_PATH train.py \
    --conv ppi \
    --target gamma \
    --epochs 200 \
    --folds 5 \
    --hidden 128 \
    --embed 64 \
    --lr 0.001 \
    --seed 42

echo "=================================================="
echo "Completed at: $(date)"
echo "=================================================="
