#!/bin/bash
#SBATCH --job-name=slmgae_adamson_main_noESM
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/train_main_without_ESM_%j.out
#SBATCH --error=logs/train_main_without_ESM_%j.err

# Exit on error
set -e

echo "=================================================="
echo "SLMGAE Main Training WITHOUT ESM (Baseline)"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "=================================================="

# Load modules
module purge
module load gnu13/13.2.0 openmpi5/5.0.3 EasyBuild/4.9.1 cmake/3.24.2 openblas/0.3.21 fftw/3.3.10
source /opt/rh/gcc-toolset-13/enable

# Python path explicitly set - no conda activation needed

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/slmgae/bin/python

# GPU is automatically set by SLURM via --gres=gpu:1

# Change to code directory
cd ../code_Adamson

# Create output directories
mkdir -p ../logs
mkdir -p ../outputs_Adamson/main_without_ESM
mkdir -p ../outputs_Adamson/main_without_ESM/checkpoints
mkdir -p ../outputs_Adamson/main_without_ESM/predictions

echo "Starting training WITHOUT ESM embeddings (baseline)..."
echo "Configuration:"
echo "  - Epochs: 300"
echo "  - Using ESM: NO (adjacency matrix as features)"
echo "  - Pos:Neg ratio: 1:1"
echo "  - Output dir: ../outputs_Adamson/main_without_ESM"

# Run training
$PYTHON_PATH train_slmgae.py \
	--cv_type cv1 \
	--epochs 300 \
	--learning_rate 0.001 \
	--dropout 0.2 \
	--hidden1 512 \
	--hidden2 256 \
	--nn_size 45 \
	--pos_neg_ratio 1 \
	--seed 123 \
	--output_dir ../outputs_Adamson/main_without_ESM

# Check training exit status
if [ $? -ne 0 ]; then
	echo "❌ Training failed with exit code $?"
	exit 1
fi

# Check if predictions were generated
echo ""
echo "Checking output files..."
for fold in 0 1 2 3 4; do
	PRED_FILE="../outputs_Adamson/main_without_ESM/fold_${fold}_cv1_predictions.csv"
	if [ -f "$PRED_FILE" ]; then
		SHAPE=$($PYTHON_PATH -c "import pandas as pd; df = pd.read_csv('$PRED_FILE', index_col=0); print(f'{df.shape[0]}x{df.shape[1]}')")
		echo "✅ Fold $fold: ${SHAPE} matrix saved"
	else
		echo "❌ Fold $fold predictions not found"
	fi
done

# Generate final summary
echo ""
echo "=================================================="
echo "Training Summary (Baseline)"
echo "=================================================="

if [ -f "../outputs_Adamson/main_without_ESM/training_summary.json" ]; then
	$PYTHON_PATH -c "
import json
with open('../outputs_Adamson/main_without_ESM/training_summary.json', 'r') as f:
    log = json.load(f)
    if 'mean_auc' in log:
        print(f'Mean AUC across folds: {log[\"mean_auc\"]:.4f}')
    if 'auc_std' in log:
        print(f'AUC Std across folds: {log[\"auc_std\"]:.4f}')
    if 'training_time' in log:
        print(f'Total training time: {log[\"training_time\"]:.2f} seconds')
    "
fi

# Compare with ESM results if available
if [ -f "../outputs_Adamson/main_with_ESM/training_summary.json" ]; then
	echo ""
	echo "Comparison with ESM results:"
	$PYTHON_PATH -c "
import json
with open('../outputs_Adamson/main_without_ESM/training_summary.json', 'r') as f:
    baseline = json.load(f)
with open('../outputs_Adamson/main_with_ESM/training_summary.json', 'r') as f:
    esm = json.load(f)
if 'mean_auc' in baseline and 'mean_auc' in esm:
    diff = esm['mean_auc'] - baseline['mean_auc']
    print(f'  Baseline AUC: {baseline[\"mean_auc\"]:.4f}')
    print(f'  ESM AUC: {esm[\"mean_auc\"]:.4f}')
    print(f'  Improvement: {diff:.4f} ({diff/baseline[\"mean_auc\"]*100:.1f}%)')
    "
fi

echo "=================================================="
echo "Completed at: $(date)"
echo "All 6375x6375 prediction matrices saved in: ../outputs_Adamson/main_without_ESM/"
echo "=================================================="
