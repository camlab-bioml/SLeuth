#!/bin/bash
#SBATCH --job-name=SLMGAE_CV2_ESM
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/train_CV2_with_ESM_%j.out
#SBATCH --error=logs/train_CV2_with_ESM_%j.err

# Exit on error
set -e

echo "=================================================="
echo "SLMGAE CV2 Training WITH ESM Embeddings"
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

# Python path explicitly set - no conda activation needed

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# GPU is automatically set by SLURM via --gres=gpu:1

# Change to code directory
cd ../code_Adamson

# Create output directories
mkdir -p ../logs
mkdir -p ../outputs/CV2_with_ESM
mkdir -p ../outputs/CV2_with_ESM/checkpoints
mkdir -p ../outputs/CV2_with_ESM/predictions

# Check if ESM embeddings exist
if [ ! -f "../ESM_embedding/main_esm_embeddings.pt" ]; then
	echo "❌ ESM embeddings not found. Run step2_generate_ESM_embedding.sh first"
	exit 1
fi

echo "Starting CV2 training with ESM embeddings..."
echo "Configuration:"
echo "  - CV Type: CV2 (gene-based cross-validation)"
echo "  - Epochs: 300"
echo "  - Using ESM: YES"
echo "  - Pos:Neg ratio: 1:1"
echo "  - Output dir: ../outputs/CV2_with_ESM"

# Run training
$PYTHON_PATH train_slmgae_with_esm.py \
	--use_esm \
	--cv_type cv2 \
	--epochs 300 \
	--learning_rate 0.001 \
	--dropout 0.2 \
	--hidden1 512 \
	--hidden2 256 \
	--nn_size 45 \
	--pos_neg_ratio 1 \
	--seed 123 \
	--output_dir ../outputs/CV2_with_ESM

# Check training exit status
if [ $? -ne 0 ]; then
	echo "❌ Training failed with exit code $?"
	exit 1
fi

# Check if predictions were generated
echo ""
echo "Checking output files..."
for fold in 0 1 2 3 4; do
	PRED_FILE="../outputs/CV2_with_ESM/fold_${fold}_cv2_predictions.csv"
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
echo "Training Summary"
echo "=================================================="

if [ -f "../outputs/CV2_with_ESM/training_summary.json" ]; then
	$PYTHON_PATH -c "
import json
with open('../outputs/CV2_with_ESM/training_summary.json', 'r') as f:
    log = json.load(f)
    if 'auc_mean' in log:
        print(f'Mean AUC across folds: {log[\"auc_mean\"]:.4f}')
    if 'auc_std' in log:
        print(f'AUC Std across folds: {log[\"auc_std\"]:.4f}')
    if 'training_time' in log:
        print(f'Total training time: {log[\"training_time\"]:.2f} seconds')
    "
fi

echo "=================================================="
echo "Completed at: $(date)"
echo "All 6375x6375 prediction matrices saved in: ../outputs/CV2_with_ESM/"
echo "=================================================="
