#!/bin/bash
#SBATCH --partition=gpu_Prosmn        # GPU partition
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3           # GPU nodes
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --job-name=GT_train           # Job name
#SBATCH --mem=256G                    # Memory allocation
#SBATCH --cpus-per-task=4             # CPU cores
#SBATCH --time=168:00:00              # 7 days runtime
#SBATCH --output=logs/graph_transformer_%j.out
#SBATCH --error=logs/graph_transformer_%j.err

# Graph Transformer Training Script
# Uses SLMGAE data: SL_Human_Approved.txt with 1:1 neg/pos ratio for efficiency
# Allows training to run to completion with early stopping

echo "============================================================"
echo "Graph Transformer Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Runtime limit: 7 days"
echo "============================================================"

# Error handling
set -e # Exit on error
trap 'echo "Error occurred at $(date). Exit code: $?"' ERR

# Load required modules
echo "=== Loading Modules ==="
module purge
module load gnu13/13.2.0
module load openmpi5/5.0.3
module load EasyBuild/4.9.1
module load cmake/3.24.2
module load openblas/0.3.21
module load fftw/3.3.10
module list

# Enable GCC-13 toolset for C++ stdlib
echo "=== Enabling GCC-13 Toolset ==="
source /opt/rh/gcc-toolset-13/enable

# Set Python environment
PYTHON_PATH=${PYTHON_PATH:-/ddn_exa/campbell/kaiyang/.conda/envs/slmgae/bin/python}

# Verify environment
echo "=== Environment Check ==="
echo "Python: $PYTHON_PATH"
$PYTHON_PATH --version
echo ""
echo "PyTorch version and GPU check:"
$PYTHON_PATH -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "CUDA devices:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
echo ""

# Create necessary directories
echo "=== Creating Directories ==="
mkdir -p logs
mkdir -p results
mkdir -p checkpoints

# Install/verify Python dependencies
echo "=== Checking Dependencies ==="
$PYTHON_PATH -m pip install -q torch numpy pandas scikit-learn matplotlib tqdm

# Set CUDA visibility
export CUDA_VISIBLE_DEVICES=0

# Main training command
echo "=== Starting Training ==="
echo "Configuration:"
echo "  - Epochs: 50 (with early stopping)"
echo "  - Linearizations: 50"
echo "  - Model: 128 hidden, 4 heads, 2 layers"
echo "  - Batch size: 512"
echo "  - Early stopping patience: 10"
echo "  - Using SLMGAE data: SL_Human_Approved.txt"
echo "  - Using ESM embeddings (meanpool) as node features"
echo "  - Using biological support views (GO-BP, GO-CC, PPI)"
echo "  - Gene universe: 6298 genes with protein sequences (ESM-filtered)"
echo "  - Negative sampling: 1:1 ratio (balanced, ~31K total edges)"
echo ""

$PYTHON_PATH train.py \
	--esm_embedding_path ../data/embeddings_esm2_t33_650M_UR50D_meanpool.pt \
	--node_feature_dim 1280 \
	--n_support_views 3 \
	--support_view_dim 64 \
	--hidden_dim 128 \
	--n_heads 4 \
	--n_layers 2 \
	--dropout 0.1 \
	--num_linearizations 50 \
	--epochs 50 \
	--batch_size 512 \
	--lr 1e-3 \
	--min_lr 1e-6 \
	--weight_decay 1e-5 \
	--patience 10 \
	--seed 42 \
	--neg_to_pos_ratio 1 \
	--use_biological_views \
	--output_dir ./results

echo ""
echo "============================================================"
echo "Training completed at $(date)"

if [ $? -eq 0 ]; then
	echo "Training successful! Results saved to ./results"

	# Generate summary report
	echo ""
	echo "=== Training Summary ==="
	RESULT_DIR=$(find ./results -name "run_*" -type d | head -1)
	if [ -d "$RESULT_DIR" ] && [ -f "$RESULT_DIR/training_history.json" ]; then
		$PYTHON_PATH -c "
import json
with open('$RESULT_DIR/training_history.json', 'r') as f:
    history = json.load(f)
    test_metrics = history.get('test_metrics', {})
    print(f'Best epoch: {history.get(\"best_epoch\", \"N/A\")}')
    print(f'Test AUC: {test_metrics.get(\"auc\", \"N/A\"):.4f}')
    print(f'Test F1: {test_metrics.get(\"f1\", \"N/A\"):.4f}')
    print(f'Test Precision: {test_metrics.get(\"precision\", \"N/A\"):.4f}')
    print(f'Test Recall: {test_metrics.get(\"recall\", \"N/A\"):.4f}')
        "
	fi

	echo ""
	echo "Key outputs generated:"
	echo "  - Model checkpoint: $RESULT_DIR/best_model.pt"
	echo "  - All predictions: $RESULT_DIR/all_predictions.csv"
	echo "  - Gene order: $RESULT_DIR/gene_order.txt"
	echo "  - Training plots: $RESULT_DIR/training_history.png"
else
	echo "Training failed with exit code $?"
	echo "Check error log: logs/graph_transformer_${SLURM_JOB_ID}.err"
fi

echo "============================================================"
