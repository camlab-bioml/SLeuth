#!/bin/bash
#SBATCH --job-name=siamese_GO
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm/logs/step1b_GO_%j.out
#SBATCH --error=slurm/logs/step1b_GO_%j.err

# Combine ESM embeddings with anc2vec GO embeddings (CPU-only, ~minutes)

set -e

echo "=================================================="
echo "Step 1b: Generate Combined ESM + GO Embeddings"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Load modules
module purge
module load gnu14 openmpi5 EasyBuild cmake openblas fftw
source /opt/rh/gcc-toolset-13/enable

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# Change to siamese_sl directory
cd "$(dirname "$0")/.."

# Paths
ESM_PATH="../data/all_genes_esm.pt"
GAF_PATH="../uniprot_GO/goa_human.gaf.gz"
OUTPUT_PATH="../data/all_genes_esm_go.pt"

# Check inputs
if [ ! -f "$ESM_PATH" ]; then
    echo "ESM embeddings not found at $ESM_PATH"
    echo "Run step1_generate_esm.sh first"
    exit 1
fi

if [ ! -f "$GAF_PATH" ]; then
    echo "GAF file not found at $GAF_PATH"
    exit 1
fi

echo "Configuration:"
echo "  - ESM input:  $ESM_PATH"
echo "  - GAF file:   $GAF_PATH"
echo "  - Output:     $OUTPUT_PATH"
echo "  - GO source:  anc2vec (200-dim, sum pooling)"
echo "  - Combined:   ESM + GO (dim inferred dynamically)"
echo ""

# Run GO embedding generation
$PYTHON_PATH generate_go_esm_embeddings.py \
    --esm_embeddings "$ESM_PATH" \
    --gaf "$GAF_PATH" \
    --output "$OUTPUT_PATH"

# Verify output
if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "Combined embeddings created successfully"
    $PYTHON_PATH -c "
import torch
data = torch.load('$OUTPUT_PATH', map_location='cpu', weights_only=False)
print(f'Embeddings shape: {data[\"embeddings\"].shape}')
print(f'Gene count: {len(data[\"gene_order\"])}')
print(f'ESM dim: {data[\"esm_dim\"]}')
print(f'GO dim: {data[\"go_dim\"]}')
print(f'Genes with GO: {data[\"num_genes_with_go\"]}/{data[\"num_genes\"]}')
"
else
    echo "Failed to create combined embeddings"
    exit 1
fi

echo "=================================================="
echo "GO embedding generation completed at: $(date)"
echo "=================================================="
