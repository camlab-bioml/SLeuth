#!/bin/bash
#SBATCH --job-name=SLMGAE_CV2
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/train_CV2_%j.out
#SBATCH --error=logs/train_CV2_%j.err

# Exit on error
set -e

echo "=================================================="
echo "SLMGAE Training with CV2 (Gene-based)"
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

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/slmgae/bin/python

# GPU is automatically set by SLURM via --gres=gpu:1

# Change to code directory
cd ../code

# Create output directories
mkdir -p ../logs
mkdir -p ../outputs/CV2
mkdir -p ../outputs/CV2/checkpoints
mkdir -p ../outputs/CV2/predictions

echo "Starting CV2 training (gene-based cross-validation)..."
echo "Configuration:"
echo "  - CV Type: CV2 (test on unseen genes)"
echo "  - Epochs: 300"
echo "  - Pos:Neg ratio: 1:1"
echo "  - Output dir: ../outputs/CV2"

# Run training
$PYTHON_PATH train_slmgae.py \
	--cv_type cv2 \
	--epochs 300 \
	--learning_rate 0.001 \
	--dropout 0.2 \
	--hidden1 512 \
	--hidden2 256 \
	--nn_size 45 \
	--pos_neg_ratio 1 \
	--seed 123 \
	--output_dir ../outputs/CV2

# Check training exit status
if [ $? -ne 0 ]; then
	echo "❌ Training failed with exit code $?"
	exit 1
fi

# Check if predictions were generated
echo ""
echo "Checking output files..."
for fold in 0 1 2 3 4; do
	PRED_FILE="../outputs/CV2/fold_${fold}_cv2_predictions.csv"
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
echo "Results saved to: ../outputs/CV2/"
echo "=================================================="
