#!/bin/bash
#SBATCH --job-name=slmgae_adamson_CV3
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/train_CV3_%j.out
#SBATCH --error=logs/train_CV3_%j.err

# Exit on error
set -e

echo "=================================================="
echo "SLMGAE Training with CV3 (Pair-based)"
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

# GPU is automatically set by SLURM via --gres=gpu:1

# Change to code directory
cd ../code_Adamson

# Create output directories
mkdir -p ../logs
mkdir -p ../outputs_Adamson/CV3
mkdir -p ../outputs_Adamson/CV3/checkpoints
mkdir -p ../outputs_Adamson/CV3/predictions

echo "Starting CV3 training (pair-based cross-validation)..."
echo "Configuration:"
echo "  - CV Type: CV3 (test on new pairs of known genes)"
echo "  - Epochs: 300"
echo "  - Pos:Neg ratio: 1:1"
echo "  - Output dir: ../outputs_Adamson/CV3"

# Run training
$PYTHON_PATH train_slmgae.py \
	--cv_type cv3 \
	--epochs 300 \
	--learning_rate 0.001 \
	--dropout 0.2 \
	--hidden1 512 \
	--hidden2 256 \
	--nn_size 45 \
	--pos_neg_ratio 1 \
	--seed 123 \
	--output_dir ../outputs_Adamson/CV3

# Check training exit status
if [ $? -ne 0 ]; then
	echo "❌ Training failed with exit code $?"
	exit 1
fi

# Check if predictions were generated
echo ""
echo "Checking output files..."
for fold in 0 1 2 3 4; do
	PRED_FILE="../outputs_Adamson/CV3/fold_${fold}_cv3_predictions.csv"
	if [ -f "$PRED_FILE" ]; then
		LINES=$(wc -l <"$PRED_FILE")
		ROWS=$((LINES - 1)) # Subtract 1 for header row
		echo "✅ Fold $fold: ${ROWS}x${ROWS} matrix saved"
	else
		echo "❌ Fold $fold predictions not found"
	fi
done

echo "=================================================="
echo "Completed at: $(date)"
echo "Results saved to: ../outputs_Adamson/CV3/"
echo "=================================================="
