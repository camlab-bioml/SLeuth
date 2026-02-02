#!/bin/bash
#SBATCH --job-name=siamese_full
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================================
# Full Siamese SL Benchmark
# ============================================================================
# This script runs all steps:
#   1. Generate ESM embeddings with Pool PaRTI
#   2. Train ALL 3 models (siamese, attention, kernel) on ALL 3 CV types
#   3. Report comprehensive results with AUROC, AUPR, F1
#
# Total runs: 3 models × 3 CV types = 9 training runs
# ============================================================================

set -e

echo "============================================================================"
echo "       FULL SIAMESE SL BENCHMARK"
echo "============================================================================"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""
echo "This benchmark will run:"
echo "  - 3 Models: siamese (MLP), attention (cross-attention), kernel (RKHS)"
echo "  - 3 CV Types: CV1 (edge), CV2 (gene), CV3 (pair)"
echo "  - Total: 9 training runs"
echo "============================================================================"
echo ""

# GPU Info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

# Load modules
module purge 2>/dev/null || true
module load gnu15 2>/dev/null || module load gnu14 2>/dev/null || true
module load openmpi5 2>/dev/null || true
module load EasyBuild 2>/dev/null || true
echo "Loaded modules:"
module list 2>&1 || true

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# Change to siamese_sl directory (absolute path)
WORK_DIR="/ddn_exa/campbell/kaiyang/SLMGAE-pytorch/siamese_sl"
cd "$WORK_DIR"
echo "Working directory: $WORK_DIR"

# Create directories inside siamese_sl
mkdir -p results
mkdir -p slurm/logs

# Results file
RESULTS_FILE="results/full_benchmark_results.txt"
echo "============================================================================" > $RESULTS_FILE
echo "       SIAMESE SL BENCHMARK RESULTS" >> $RESULTS_FILE
echo "============================================================================" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# ============================================================================
# STEP 1: Generate ESM Embeddings (if not exists)
# ============================================================================
echo "============================================================================"
echo "STEP 1: ESM Embedding Generation"
echo "============================================================================"

# ESM embeddings stored inside siamese_sl/results/ (already exists)
ESM_PATH="results/all_genes_esm.pt"

if [ -f "$ESM_PATH" ]; then
    echo "ESM embeddings already exist, skipping generation..."
    $PYTHON_PATH -c "
import torch
data = torch.load('$ESM_PATH', map_location='cpu')
print(f'  Embeddings: {data[\"embeddings\"].shape}')
print(f'  Genes: {len(data[\"gene_order\"])}')
print(f'  Pooling: {data.get(\"pooling\", \"unknown\")}')
"
else
    echo "Generating ESM embeddings with Pool PaRTI..."
    $PYTHON_PATH generate_all_genes_esm.py \
        --output "$ESM_PATH" \
        --pooling pool_parti \
        --batch_size 8 \
        --device cuda:0

    if [ ! -f "$ESM_PATH" ]; then
        echo "FAILED: ESM embeddings not created"
        exit 1
    fi
    echo "ESM embeddings created successfully"
fi

# SL pairs read from parent data directory (read-only)
SL_PATH="/ddn_exa/campbell/kaiyang/SLMGAE-pytorch/data/SL_Human_Approved.txt"

if [ ! -f "$SL_PATH" ]; then
    echo "FAILED: SL pairs file not found at $SL_PATH"
    exit 1
fi
echo "SL pairs file found: $SL_PATH"

echo "" >> $RESULTS_FILE
echo "ESM Embeddings:" >> $RESULTS_FILE
$PYTHON_PATH -c "
import torch
data = torch.load('$ESM_PATH', map_location='cpu')
print(f'  Shape: {data[\"embeddings\"].shape}')
print(f'  Pooling: {data.get(\"pooling\", \"unknown\")}')
print(f'  Standardized: {data.get(\"standardized\", False)}')
" >> $RESULTS_FILE

echo ""
echo "Step 1 completed at: $(date)"
echo ""

# ============================================================================
# STEP 2: Train All Models on All CV Types
# ============================================================================
echo "============================================================================"
echo "STEP 2: Training All Models on All CV Types"
echo "============================================================================"
echo ""

# Model configurations
MODELS=("siamese" "attention" "kernel")
CV_TYPES=("cv1" "cv2" "cv3")

# Common parameters
EPOCHS=300
BATCH_SIZE=256
LR=0.001
PATIENCE=20
NUM_FOLDS=5
SEED=42

# Store results for final summary
declare -A RESULTS_AUROC
declare -A RESULTS_AUPR
declare -A RESULTS_F1
FAILED_RUNS=0
SUCCESSFUL_RUNS=0

for MODEL in "${MODELS[@]}"; do
    for CV in "${CV_TYPES[@]}"; do
        echo "============================================================================"
        echo "Training: Model=$MODEL, CV=$CV"
        echo "============================================================================"
        echo "Start: $(date)"

        OUTPUT_DIR="results/${MODEL}_${CV}"
        mkdir -p "$OUTPUT_DIR/checkpoints"

        # Model-specific parameters
        if [ "$MODEL" == "kernel" ]; then
            MODEL_ARGS="--model_type kernel --encoder_type lowrank --encoder_rank 64 --rff_features 128 --bilinear_rank 128"
        elif [ "$MODEL" == "attention" ]; then
            MODEL_ARGS="--model_type attention --num_heads 4"
        else
            MODEL_ARGS="--model_type siamese --predictor_hidden 32"
        fi

        # Run training (continue on error to complete other runs)
        set +e  # Temporarily disable exit on error
        $PYTHON_PATH train.py \
            --embeddings_path "$ESM_PATH" \
            --sl_path "$SL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --cv_type "$CV" \
            $MODEL_ARGS \
            --hidden_dim 64 \
            --latent_dim 64 \
            --dropout 0.2 \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LR \
            --patience $PATIENCE \
            --num_folds $NUM_FOLDS \
            --seed $SEED
        TRAIN_EXIT_CODE=$?
        set -e  # Re-enable exit on error

        if [ $TRAIN_EXIT_CODE -ne 0 ]; then
            echo "WARNING: Training failed for $MODEL on $CV (exit code: $TRAIN_EXIT_CODE)"
        fi

        # Extract results
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            AUROC=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json')); print(f\"{d['summary']['auroc_mean']:.4f} +/- {d['summary']['auroc_std']:.4f}\")")
            AUPR=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json')); print(f\"{d['summary']['aupr_mean']:.4f} +/- {d['summary']['aupr_std']:.4f}\")")
            F1=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json')); print(f\"{d['summary']['f1_mean']:.4f} +/- {d['summary']['f1_std']:.4f}\")")

            echo ""
            echo "Results for $MODEL on $CV:"
            echo "  AUROC: $AUROC"
            echo "  AUPR:  $AUPR"
            echo "  F1:    $F1"

            # Store for summary
            RESULTS_AUROC["${MODEL}_${CV}"]="$AUROC"
            RESULTS_AUPR["${MODEL}_${CV}"]="$AUPR"
            RESULTS_F1["${MODEL}_${CV}"]="$F1"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        else
            echo "WARNING: Results file not found for $MODEL on $CV"
            RESULTS_AUROC["${MODEL}_${CV}"]="FAILED"
            RESULTS_AUPR["${MODEL}_${CV}"]="FAILED"
            RESULTS_F1["${MODEL}_${CV}"]="FAILED"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi

        echo "Completed: $(date)"
        echo ""
    done
done

# ============================================================================
# STEP 3: Generate Summary Report
# ============================================================================
echo "============================================================================"
echo "STEP 3: Summary Report"
echo "============================================================================"
echo ""

# Print to stdout and file
{
echo ""
echo "============================================================================"
echo "                    BENCHMARK RESULTS SUMMARY"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Early Stopping: $PATIENCE evaluations"
echo "  Folds: $NUM_FOLDS"
echo "  Seed: $SEED"
echo ""
echo "============================================================================"
echo "                           AUROC RESULTS"
echo "============================================================================"
printf "%-15s | %-20s | %-20s | %-20s\n" "Model" "CV1 (Edge)" "CV2 (Gene)" "CV3 (Pair)"
echo "-----------------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    printf "%-15s | %-20s | %-20s | %-20s\n" \
        "$MODEL" \
        "${RESULTS_AUROC[${MODEL}_cv1]}" \
        "${RESULTS_AUROC[${MODEL}_cv2]}" \
        "${RESULTS_AUROC[${MODEL}_cv3]}"
done
echo ""
echo "============================================================================"
echo "                           AUPR RESULTS"
echo "============================================================================"
printf "%-15s | %-20s | %-20s | %-20s\n" "Model" "CV1 (Edge)" "CV2 (Gene)" "CV3 (Pair)"
echo "-----------------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    printf "%-15s | %-20s | %-20s | %-20s\n" \
        "$MODEL" \
        "${RESULTS_AUPR[${MODEL}_cv1]}" \
        "${RESULTS_AUPR[${MODEL}_cv2]}" \
        "${RESULTS_AUPR[${MODEL}_cv3]}"
done
echo ""
echo "============================================================================"
echo "                            F1 RESULTS"
echo "============================================================================"
printf "%-15s | %-20s | %-20s | %-20s\n" "Model" "CV1 (Edge)" "CV2 (Gene)" "CV3 (Pair)"
echo "-----------------------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    printf "%-15s | %-20s | %-20s | %-20s\n" \
        "$MODEL" \
        "${RESULTS_F1[${MODEL}_cv1]}" \
        "${RESULTS_F1[${MODEL}_cv2]}" \
        "${RESULTS_F1[${MODEL}_cv3]}"
done
echo ""
echo "============================================================================"
echo ""
echo "CV Type Descriptions:"
echo "  CV1 (Edge-based): Random split of SL pairs. Tests interpolation."
echo "  CV2 (Gene-based): Hold out genes. Test has >=1 unseen gene. Semi-inductive."
echo "  CV3 (Pair-based): Both genes unseen in test. Fully inductive (hardest)."
echo ""
echo "Model Descriptions:"
echo "  siamese:   Simple MLP encoder with symmetric features (sum, product, abs_diff)"
echo "  attention: Cross-attention between gene pairs before prediction"
echo "  kernel:    RKHS-based with Random Fourier Features and Hilbert space mapping"
echo ""
echo "============================================================================"
} | tee -a $RESULTS_FILE

# Also create a JSON summary
$PYTHON_PATH << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from datetime import datetime

models = ["siamese", "attention", "kernel"]
cv_types = ["cv1", "cv2", "cv3"]

summary = {
    "timestamp": datetime.now().isoformat(),
    "configuration": {
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 0.001,
        "patience": 20,
        "num_folds": 5,
        "seed": 42
    },
    "results": {}
}

for model in models:
    summary["results"][model] = {}
    for cv in cv_types:
        results_path = Path(f"results/{model}_{cv}/results.json")
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            summary["results"][model][cv] = {
                "auroc": f"{data['summary']['auroc_mean']:.4f} +/- {data['summary']['auroc_std']:.4f}",
                "aupr": f"{data['summary']['aupr_mean']:.4f} +/- {data['summary']['aupr_std']:.4f}",
                "f1": f"{data['summary']['f1_mean']:.4f} +/- {data['summary']['f1_std']:.4f}",
                "auroc_mean": data['summary']['auroc_mean'],
                "auroc_std": data['summary']['auroc_std'],
                "aupr_mean": data['summary']['aupr_mean'],
                "aupr_std": data['summary']['aupr_std'],
                "f1_mean": data['summary']['f1_mean'],
                "f1_std": data['summary']['f1_std'],
            }
        else:
            summary["results"][model][cv] = {"error": "Results not found"}

with open("results/full_benchmark_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("JSON summary saved to: results/full_benchmark_summary.json")
PYTHON_SCRIPT

echo ""
echo "============================================================================"
echo "Benchmark completed at: $(date)"
echo ""
echo "Run Summary:"
echo "  - Successful: $SUCCESSFUL_RUNS / 9"
echo "  - Failed: $FAILED_RUNS / 9"
echo ""
echo "Output files:"
echo "  - results/full_benchmark_results.txt (human-readable summary)"
echo "  - results/full_benchmark_summary.json (machine-readable summary)"
echo "  - results/<model>_<cv>/results.json (per-run detailed results)"
echo "============================================================================"

# Exit with error if any runs failed
if [ $FAILED_RUNS -gt 0 ]; then
    echo "WARNING: $FAILED_RUNS training run(s) failed. Check logs for details."
    exit 1
fi
