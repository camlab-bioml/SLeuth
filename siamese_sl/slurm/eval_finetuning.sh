#!/bin/bash
#SBATCH --job-name=eval_finetuning
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/eval_finetuning_%j.out
#SBATCH --error=slurm/logs/eval_finetuning_%j.err

# ============================================================================
# eval_finetuning — External CRISPR-screen evaluation + fine-tuning
# ============================================================================
# Loads a pretrained multi-modal siamese SL combo (default: CV3 combo),
# evaluates it against three external screens (Adamson, Corn, Gilbert) with
# zero-shot calibrated metrics, then fine-tunes via MSE regression on an
# (a, c) affine head + optional last-layer / full-encoder unfreezes.
#
# Produces under $MODEL_DIR/eval_finetuning/:
#   config_used.json
#   gene_coverage.csv
#   zero_shot_metrics.csv  / zero_shot_summary.csv
#   finetuned_metrics.csv  / finetuned_summary.csv
#   {dataset}_predictions.csv  / {dataset}_finetuned_predictions_{mode}.csv
#
# Usage:
#   # Direct submission (uses defaults from eval_finetuning.conf):
#   sbatch slurm/eval_finetuning.sh
#
#   # Override the target model:
#   EVAL_MODEL_DIR=results/my_combo_cv3 sbatch slurm/eval_finetuning.sh
#
#   # Via the pipeline orchestrator (standalone stage):
#   ./slurm/submit_pipeline.sh --eval-only results/best_cat_..._cv3
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/eval_finetuning.conf"

sleep 5

# ============================================================================
# Validate inputs
# ============================================================================
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR not found: $MODEL_DIR"
    echo "  Set it via eval_finetuning.conf or EVAL_MODEL_DIR env var."
    exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json missing — is this a trained combo?"
    exit 1
fi

# ============================================================================
# Resolve embedding paths from the combo's config.json.
#
# The combo's config.json records the exact modalities (and their order) used
# to fit the preprocessing_transform stored in each checkpoint. Reading those
# paths here avoids drift: category winners change across reruns, so any
# hardcoded EMBEDDINGS_PATHS in eval_finetuning.conf rots silently.
#
# If EMBEDDINGS_PATHS is set in eval_finetuning.conf (legacy override), we
# honor it; otherwise auto-resolve from config.json, rebasing filenames under
# the current $DATA_DIR so paths from a different cluster still resolve.
# ============================================================================
if [ ${#EMBEDDINGS_PATHS[@]} -eq 0 ]; then
    echo "[eval_finetuning.sh] Auto-resolving embeddings_paths from $MODEL_DIR/config.json"
    export DATA_DIR
    mapfile -t EMBEDDINGS_PATHS < <(
        "$PYTHON_PATH" -c "
import json, os, sys
cfg = json.load(open('$MODEL_DIR/config.json'))
data_dir = os.environ['DATA_DIR']
paths = cfg.get('embeddings_paths') or ([cfg['embeddings_path']] if cfg.get('embeddings_path') else [])
if not paths:
    sys.exit('ERROR: config.json has no embeddings_paths / embeddings_path')
for p in paths:
    print(os.path.join(data_dir, os.path.basename(p)))
"
    )
    if [ ${#EMBEDDINGS_PATHS[@]} -eq 0 ]; then
        echo "ERROR: failed to resolve embeddings_paths from config.json"
        exit 1
    fi
fi

# Ensure at least one embedding file exists so we fail fast before loading.
for p in "${EMBEDDINGS_PATHS[@]}"; do
    if [ ! -f "$p" ]; then
        echo "WARNING: embedding file missing: $p"
    fi
done

# ============================================================================
# Build post-PCA argument
# ============================================================================
if [ -n "${POST_PCA_VARIANCE:-}" ]; then
    POST_PCA_ARGS="--post_pca_variance $POST_PCA_VARIANCE"
else
    POST_PCA_ARGS=""
fi

if [ "${REPORT_SLDB_FILTERED:-true}" = "true" ]; then
    SLDB_FLAG="--report_sldb_filtered"
else
    SLDB_FLAG=""
fi

# ============================================================================
# Run
# ============================================================================
echo "MODEL_DIR:        $MODEL_DIR"
echo "EMBEDDINGS (${#EMBEDDINGS_PATHS[@]}):"
for p in "${EMBEDDINGS_PATHS[@]}"; do echo "    $p"; done
echo "PCA_VARIANCE:     $PCA_VARIANCE"
echo "POST_PCA_VARIANCE:${POST_PCA_VARIANCE:-(disabled)}"
echo "DATASETS (${#DATASETS[@]}):"
for d in "${DATASETS[@]}"; do echo "    $d"; done
echo "FT_MODES:         ${FT_MODES[*]}"
echo "FT_EPOCHS:        $FT_EPOCHS  (patience=$FT_PATIENCE)"
echo "FT_LR:            $FT_LR   FT_BATCH_SIZE=$FT_BATCH_SIZE   FT_WD=$FT_WEIGHT_DECAY"
echo "SPLIT:            train=$TRAIN_FRAC  val=$VAL_FRAC  seed=$SPLIT_SEED"
echo "SLDB_FILTER:      ${REPORT_SLDB_FILTERED:-true}"
echo ""

$PYTHON_PATH eval_finetuning.py \
    --model_dir "$MODEL_DIR" \
    --embeddings_paths "${EMBEDDINGS_PATHS[@]}" \
    --pca_variance "$PCA_VARIANCE" \
    $POST_PCA_ARGS \
    --datasets "${DATASETS[@]}" \
    --sl_path "$SL_PATH" \
    --base_dir "$BASE_DIR" \
    --ft_modes "${FT_MODES[@]}" \
    --ft_epochs "$FT_EPOCHS" \
    --ft_patience "$FT_PATIENCE" \
    --ft_lr "$FT_LR" \
    --ft_batch_size "$FT_BATCH_SIZE" \
    --ft_weight_decay "$FT_WEIGHT_DECAY" \
    --train_frac "$TRAIN_FRAC" \
    --val_frac "$VAL_FRAC" \
    --split_seed "$SPLIT_SEED" \
    $SLDB_FLAG

echo ""
echo "Completed at: $(date)"
