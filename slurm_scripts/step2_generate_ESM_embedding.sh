#!/bin/bash
#SBATCH --job-name=ESM_embeddings
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/step2_ESM_embeddings_%j.out
#SBATCH --error=logs/step2_ESM_embeddings_%j.err

echo "=================================================="
echo "Step 2: Generate ESM Embeddings"
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
cd ../code

# Create necessary directories
mkdir -p ../ESM_embedding
mkdir -p ../logs

# Check if FASTA files exist
if [ ! -f "../protein_seq/main_protein_seq.fasta" ]; then
	echo "❌ Main FASTA not found. Run step1_download_protein_seq.sh first"
	exit 1
fi

# Run ESM embedding generation with GPU optimization
echo "Starting ESM embedding generation on GPU..."
echo "Using batch size: 16 (adjust based on GPU memory)"
echo ""
echo "Processing pipeline:"
echo "  1. Generate amino acid embeddings via ESM-2"
echo "  2. Mean pooling across all amino acids per protein"
echo "  3. Normalization: zero mean, then unit norm"

$PYTHON_PATH generate_esm_embeddings_gpu.py \
	--batch_size 16 \
	--device cuda:0

# Check if successful
if [ -f "../ESM_embedding/main_esm_embeddings.pt" ]; then
	echo "✅ Main ESM embeddings created successfully"

	# Verify embeddings
	$PYTHON_PATH -c "
import torch
import numpy as np
data = torch.load('../ESM_embedding/main_esm_embeddings.pt', map_location='cpu')
print(f'Embeddings shape: {data[\"embeddings\"].shape}')
print(f'Gene count: {len(data[\"gene_order\"])}')
if 'noncoding_genes' in data:
    print(f'Non-coding genes: {len(data[\"noncoding_genes\"])}')
non_zero = (data['embeddings'].sum(dim=1) != 0).sum().item()
print(f'Non-zero embeddings: {non_zero}/{len(data[\"gene_order\"])} ({non_zero/len(data[\"gene_order\"])*100:.1f}%)')

# Check normalization
embeddings = data['embeddings'].numpy()
non_zero_mask = np.any(embeddings != 0, axis=1)
if np.any(non_zero_mask):
    non_zero_emb = embeddings[non_zero_mask]
    norms = np.linalg.norm(non_zero_emb, axis=1)
    print(f'')
    print(f'Normalization check:')
    print(f'  - Mean norm: {np.mean(norms):.6f} (should be ~1.0)')
    print(f'  - Std norm: {np.std(norms):.6f} (should be ~0.0)')
    print(f'  - Mean value: {np.mean(non_zero_emb):.6f} (should be ~0.0)')
    "
else
	echo "❌ Failed to create main ESM embeddings"
	exit 1
fi

echo "=================================================="
echo "ESM generation completed at: $(date)"
echo "Next steps: Run training scripts"
echo "  - step3_train_main_with_ESM.sh"
echo "  - step3_train_main_without_ESM.sh"
echo "=================================================="
