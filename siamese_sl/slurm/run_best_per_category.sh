#!/bin/bash
#SBATCH --job-name=bestcat_emb
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/bestcat_%A_%a.out
#SBATCH --error=slurm/logs/bestcat_%A_%a.err

# ============================================================================
# Best-per-Category Step 1 — SLURM Array Worker
# ============================================================================
# Each array task trains the siamese model on ONE (embedding, CV) combination
# with robust PCA variance-based reduction (PCA_VARIANCE from the .conf file).
#
# SLURM_ARRAY_TASK_ID is mapped to (embedding_index, cv_index) using:
#   embedding_index = task_id / num_cvs
#   cv_index        = task_id % num_cvs
#
# Array size is set dynamically by submit_pipeline.sh based on the number of
# embeddings in EMB_CATALOG[] and CV types in CV_TYPES[].
#
# NOTE: --array is NOT set here. It is passed by submit_pipeline.sh:
#   sbatch --array=0-<N-1>%<MAX_CONCURRENT> slurm/run_best_per_category.sh
#
# After all array tasks complete, run_best_per_category_combo.sh picks the
# best embedding per category and trains multi-modal combinations.
#
# Prerequisites:
#   - Embedding .pt files must exist (run_generate_embeddings.sh)
#   - Submit via submit_pipeline.sh, not directly
#
# Output: results/pcavar<variance>_<embedding_type>_<cv_type>/results.json
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.sh"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"

sleep 10  # let disk settle after prior job

# ============================================================================
# Validate: must be running as an array task
# ============================================================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: This script must be submitted as a SLURM array job."
    echo "Use submit_pipeline.sh to launch the pipeline, or pass --array=0-N."
    exit 1
fi

# ============================================================================
# Map task ID → (embedding, CV)
# ============================================================================
# The task ID encodes both the embedding index and CV index:
#   task_id = embedding_index * num_cvs + cv_index

NUM_CVS=${#CV_TYPES[@]}
EMB_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CVS))
CV_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CVS))

# Bounds check (in case --array range exceeds catalog size)
if [ "$EMB_INDEX" -ge "${#EMB_CATALOG[@]}" ]; then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID exceeds catalog size (${#EMB_CATALOG[@]})"
    exit 0
fi

# Extract embedding info and CV type
ENTRY="${EMB_CATALOG[$EMB_INDEX]}"
CV="${CV_TYPES[$CV_INDEX]}"
IFS=':' read -r ETYPE EFILE ECAT <<< "$ENTRY"
EMB_PATH="$DATA_DIR/$EFILE"

echo "Task $SLURM_ARRAY_TASK_ID: $ETYPE ($ECAT) / $CV  [PCA=$PCA_VARIANCE]"
echo "  Embedding: $EMB_PATH"
echo ""

# ============================================================================
# Skip if embedding file doesn't exist
# ============================================================================
# Some embeddings may fail to generate. Exit cleanly so the combo job runs.

if [ ! -f "$EMB_PATH" ]; then
    echo "SKIP: $EFILE not found in $DATA_DIR"
    exit 0
fi

# ============================================================================
# Clean previous results for this (embedding, CV, PCA) combination
# ============================================================================
OUTPUT_DIR="results/pcavar${PCA_VARIANCE}_${ETYPE}_${CV}"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous results: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# ============================================================================
# Train with PCA
# ============================================================================

# Create output directory
$PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

echo "Training: $ETYPE / $CV  (PCA=$PCA_VARIANCE)"
echo "  Output: $OUTPUT_DIR"
echo "  Config: encoder=[$ENCODER_DIMS], epochs=$EPOCHS, patience=$PATIENCE"
echo ""

# Single-modality: post-PCA would be redundant with the per-modality
# --pca_variance step, so always disable it here. POST_PCA_VARIANCE from
# config.sh applies only to the multi-modal combo script.
set +e
$PYTHON_PATH train.py \
    --embeddings_paths "$EMB_PATH" \
    --sl_path "$SL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --cv_type "$CV" \
    $MODEL_ARGS \
    --encoder_dims $ENCODER_DIMS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --l1_lambdas $L1_LAMBDAS \
    --pd_epsilon $PD_EPSILON \
    --eval_interval $EVAL_INTERVAL \
    --patience $PATIENCE \
    --warmrestart_T0 $WARMRESTART_T0 \
    --warmrestart_Tmult $WARMRESTART_TMULT \
    --num_folds $NUM_FOLDS \
    --pos_neg_ratio $POS_NEG_RATIO \
    --seed $SEED \
    --pca_variance $PCA_VARIANCE \
    --no_post_pca
TRAIN_EXIT=$?
set -e

# ============================================================================
# Report result
# ============================================================================
echo ""
if [ $TRAIN_EXIT -ne 0 ] || [ ! -f "$OUTPUT_DIR/results.json" ]; then
    echo "FAILED: $ETYPE / $CV (exit code $TRAIN_EXIT)"
    exit 1
fi

AUROC=$($PYTHON_PATH -c "
import json
d = json.load(open('$OUTPUT_DIR/results.json'))['summary']
v, s = d['auroc_mean'], d['auroc_std']
print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')
" 2>/dev/null || echo "N/A")

PARAMS=$($PYTHON_PATH -c "
import json
d = json.load(open('$OUTPUT_DIR/results.json'))['summary']
print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")
" 2>/dev/null || echo "N/A")

echo "SUCCESS: $ETYPE / $CV  (PCA=$PCA_VARIANCE)"
echo "  AUROC=$AUROC  Params=$PARAMS"
echo "  Completed at: $(date)"
