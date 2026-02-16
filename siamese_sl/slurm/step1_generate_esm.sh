#!/bin/bash
#SBATCH --job-name=siamese_ESM
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=slurm/logs/step1_ESM_%j.out
#SBATCH --error=slurm/logs/step1_ESM_%j.err

# Exit on error
set -e

echo "=================================================="
echo "Step 1: Generate ESM Embeddings with Pool PaRTI"
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

# Create necessary directories
mkdir -p ../data
mkdir -p slurm/logs
mkdir -p results

echo "Configuration:"
echo "  - Pooling: Pool PaRTI (PageRank-based attention)"
echo "  - Model: ESM-2 (650M parameters)"
echo "  - Proteins: Reviewed (Swiss-Prot) only (~20k genes)"
echo "  - Output: ../data/all_genes_esm.pt"
echo ""

# Run ESM embedding generation (reviewed_only is now default)
echo "Starting ESM embedding generation..."
$PYTHON_PATH generate_all_genes_esm.py \
    --output ../data/all_genes_esm.pt \
    --pooling pool_parti \
    --batch_size 8 \
    --device cuda:0

# Check if successful
if [ -f "../data/all_genes_esm.pt" ]; then
    echo ""
    echo "✅ ESM embeddings created successfully"

    # Verify embeddings
    $PYTHON_PATH -c "
import torch
data = torch.load('../data/all_genes_esm.pt', map_location='cpu', weights_only=False)
print(f'Embeddings shape: {data[\"embeddings\"].shape}')
print(f'Gene count: {len(data[\"gene_order\"])}')
print(f'Embedding dim: {data[\"embeddings\"].shape[1]}')
print(f'Pooling method: {data.get(\"pooling\", \"unknown\")}')
print(f'Standardized: {data.get(\"standardized\", False)}')

# Check for NaNs
import numpy as np
emb = data['embeddings'].numpy()
nan_count = np.isnan(emb).sum()
print(f'NaN values: {nan_count}')

# Check norms
norms = np.linalg.norm(emb, axis=1)
print(f'Norm range: [{norms.min():.4f}, {norms.max():.4f}]')
"
else
    echo "❌ Failed to create ESM embeddings"
    exit 1
fi

echo "=================================================="
echo "ESM generation completed at: $(date)"
echo "Next: Run step1b_generate_go_esm.sh, then step2_train_cv{1,2,3}.sh"
echo "=================================================="
