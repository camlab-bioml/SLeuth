#!/bin/bash
# Master script to run the complete Siamese SL pipeline
# Usage: ./slurm/run_all.sh [--skip-esm]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Create logs directory
mkdir -p slurm/logs

echo "=================================================="
echo "Siamese SL Pipeline - RKHS-based Siamese Network"
echo "=================================================="
echo "Start time: $(date)"
echo ""
echo "Pipeline:"
echo "  1. Generate ESM embeddings (Pool PaRTI pooling)"
echo "  2. Train on CV1 (edge-based)"
echo "  3. Train on CV2 (gene-based)"
echo "  4. Train on CV3 (pair-based)"
echo "=================================================="
echo ""

# Check for --skip-esm flag
SKIP_ESM=false
if [ "$1" == "--skip-esm" ]; then
    SKIP_ESM=true
    echo "Skipping ESM generation (--skip-esm flag set)"
fi

# Step 1: Generate ESM embeddings
if [ "$SKIP_ESM" = false ]; then
    echo "Submitting Step 1: ESM Embedding Generation..."
    JOB1=$(sbatch --parsable slurm/step1_generate_esm.sh)
    echo "  Job ID: $JOB1"
    echo ""

    # Step 2: Train all CVs (depends on Step 1)
    echo "Submitting Step 2: Training (will wait for ESM generation)..."
    JOB_CV1=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/step2_train_cv1.sh)
    JOB_CV2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/step2_train_cv2.sh)
    JOB_CV3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/step2_train_cv3.sh)
else
    # Check if embeddings exist
    if [ ! -f "../data/all_genes_esm.pt" ]; then
        echo "❌ Error: ESM embeddings not found at ../data/all_genes_esm.pt"
        echo "   Run without --skip-esm flag first"
        exit 1
    fi
    echo "Using existing ESM embeddings"
    echo ""

    # Submit training jobs without dependency
    echo "Submitting Training Jobs..."
    JOB_CV1=$(sbatch --parsable slurm/step2_train_cv1.sh)
    JOB_CV2=$(sbatch --parsable slurm/step2_train_cv2.sh)
    JOB_CV3=$(sbatch --parsable slurm/step2_train_cv3.sh)
fi

echo "  CV1 Job ID: $JOB_CV1"
echo "  CV2 Job ID: $JOB_CV2"
echo "  CV3 Job ID: $JOB_CV3"
echo ""

# Submit summary job (depends on all training)
echo "Submitting Summary Job (will run after all training completes)..."
JOB_SUMMARY=$(sbatch --parsable --dependency=afterok:$JOB_CV1:$JOB_CV2:$JOB_CV3 slurm/step3_summarize.sh)
echo "  Summary Job ID: $JOB_SUMMARY"
echo ""

echo "=================================================="
echo "All jobs submitted!"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f slurm/logs/step1_ESM_*.out"
echo "  tail -f slurm/logs/step2_CV*_*.out"
echo ""
echo "Results will be saved to:"
echo "  results/cv1/results.json"
echo "  results/cv2/results.json"
echo "  results/cv3/results.json"
echo "  results/summary.txt"
echo "=================================================="
