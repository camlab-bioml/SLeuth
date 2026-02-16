#!/bin/bash
#SBATCH --job-name=siamese_CV3
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/step2_CV3_%j.out
#SBATCH --error=slurm/logs/step2_CV3_%j.err

# Exit on error
set -e

echo "=================================================="
echo "Siamese SL Training: CV3 (Pair-based)"
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

# Change to siamese_sl directory
cd "$(dirname "$0")/.."

# Create output directories
mkdir -p slurm/logs
mkdir -p results/cv3/checkpoints

# Embeddings path and dynamic input_dim
EMB_PATH="../data/all_genes_esm_go.pt"

if [ ! -f "$EMB_PATH" ]; then
    echo "❌ Combined ESM+GO embeddings not found. Run step1b_generate_go_esm.sh first"
    exit 1
fi

INPUT_DIM=$($PYTHON_PATH -c "import torch; print(torch.load('$EMB_PATH', map_location='cpu', weights_only=False)['embeddings'].shape[1])")

echo "Configuration:"
echo "  - CV Type: CV3 (pair-based, BOTH genes unseen in test)"
echo "  - Model: RKHS Siamese (kernel, default)"
echo "  - Encoder: lowrank (efficient)"
echo "  - Input dim: $INPUT_DIM"
echo "  - Epochs: 300"
echo "  - Folds: 5"
echo "  - Note: This is the hardest CV - tests full generalization"
echo "  - Note: CV splits use --seed (42 by default)"
echo ""

# Run training
$PYTHON_PATH train.py \
    --embeddings_path "$EMB_PATH" \
    --input_dim $INPUT_DIM \
    --sl_path ../data/SL_Human_Approved.txt \
    --output_dir results/cv3 \
    --cv_type cv3 \
    --model_type kernel \
    --encoder_type lowrank \
    --encoder_rank 64 \
    --rff_features 128 \
    --bilinear_rank 128 \
    --hidden_dim 512 \
    --latent_dim 256 \
    --epochs 300 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --l1_lambda 0.1 \
    --patience 20 \
    --num_folds 5 \
    --seed 42

echo "=================================================="
echo "CV3 Training completed at: $(date)"
echo "Results saved to: results/cv3/"
echo "=================================================="
