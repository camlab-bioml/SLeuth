#!/bin/bash
#SBATCH --job-name=eval_finetuning
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-08-10, by request). gpu3 threw a transient "CUDA
# unknown error" at torch init in 2026-07 and the whole tree was moved to
# gpu2 for that; if it recurs, the symptom is a torch.cuda init failure in
# the very first seconds of the job, not a training-time error.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/eval_finetuning_%A_%a.out
#SBATCH --error=slurm/logs/eval_finetuning_%A_%a.err

# ============================================================================
# eval_finetuning — External CRISPR-screen evaluation + fine-tuning
# ============================================================================
# Loads a pretrained multi-modal siamese SL combo (default: CV3 combo),
# evaluates it against three external screens (Adamson, Corn, Gilbert) with
# zero-shot calibrated metrics, then fine-tunes via MSE regression on an
# (a, c) affine head + optional last-layer / full-encoder unfreezes.
#
# Produces under $MODEL_DIR/eval_finetuning_<cv_type>/ (one directory per
# cv_type in EVAL_CV_TYPES — the script never passes --out_dir, so
# eval_finetuning.py derives the name from the cv_type):
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
# POST_PCA_DIM wins over POST_PCA_VARIANCE. Default is the variance target
# (POST_PCA_DIM empty). Be aware the same fraction resolves to a different k
# under `plain` than under `robust`/`svd` (correlation vs raw spectrum: 1087 vs
# 145 on the 6-modality combo), and that k drifts with the combo's modality
# selection, so the encoder width is not fixed run to run. Set POST_PCA_DIM to
# pin it. See config.conf.
if [ -n "${POST_PCA_DIM:-}" ]; then
    POST_PCA_ARGS="--post_pca_dim $POST_PCA_DIM"
elif [ -n "${POST_PCA_VARIANCE:-}" ]; then
    POST_PCA_ARGS="--post_pca_variance $POST_PCA_VARIANCE"
else
    POST_PCA_ARGS=""
fi

if [ "${REPORT_SLDB_FILTERED:-true}" = "true" ]; then
    SLDB_FLAG="--report_sldb_filtered"
else
    SLDB_FLAG=""
fi

if [ "${PER_HEAD:-false}" = "true" ] || [ "${PER_HEAD:-0}" = "1" ]; then
    PER_HEAD_FLAG="--per_head"
else
    PER_HEAD_FLAG=""
fi

# ============================================================================
# Run
# ============================================================================
if [ ${#EVAL_CV_TYPES[@]} -eq 0 ]; then
    echo "ERROR: EVAL_CV_TYPES is empty. Set it in eval_finetuning.conf"
    echo "  (e.g., EVAL_CV_TYPES=(\"cv1\" \"cv2\" \"cv3\"))"
    exit 1
fi

echo "MODEL_DIR:        $MODEL_DIR"
echo "EMBEDDINGS (${#EMBEDDINGS_PATHS[@]}):"
for p in "${EMBEDDINGS_PATHS[@]}"; do echo "    $p"; done
echo "PCA_VARIANCE:     $PCA_VARIANCE"
echo "POST_PCA_DIM:     ${POST_PCA_DIM:-(unset)}"
echo "POST_PCA_VARIANCE:${POST_PCA_VARIANCE:-(disabled)}${POST_PCA_DIM:+  (ignored: POST_PCA_DIM wins)}"
echo "DATASETS (${#DATASETS[@]}):"
for d in "${DATASETS[@]}"; do echo "    $d"; done
echo "EVAL_CV_TYPES:    ${EVAL_CV_TYPES[*]}"
echo "FT_MODES:         ${FT_MODES[*]}"
echo "FT_EPOCHS:        $FT_EPOCHS  (patience=$FT_PATIENCE)"
echo "FT_LR:            $FT_LR   FT_BATCH_SIZE=$FT_BATCH_SIZE   FT_WD=$FT_WEIGHT_DECAY"
echo "SPLIT:            train=$TRAIN_FRAC  val=$VAL_FRAC  seed=$SPLIT_SEED"
echo "SLDB_FILTER:      ${REPORT_SLDB_FILTERED:-true}"
echo "PER_HEAD:         ${PER_HEAD:-false}"
echo ""

# Array-aware. Submitted with --array=0-N the job runs ONE cv_type per task, so
# the three evaluations overlap instead of running back to back; submitted
# without --array it falls back to the serial loop so a bare
# `sbatch slurm/eval_finetuning.sh` still works. Each cv_type writes its own
# ${MODEL_DIR}/eval_finetuning_<cv>/ directory, so the tasks never collide.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#EVAL_CV_TYPES[@]}" ]; then
        echo "Task $SLURM_ARRAY_TASK_ID is beyond ${#EVAL_CV_TYPES[@]} configured cv_types. Nothing to do."
        exit 0
    fi
    RUN_CV_TYPES=("${EVAL_CV_TYPES[$SLURM_ARRAY_TASK_ID]}")
    echo "Array task $SLURM_ARRAY_TASK_ID -> cv_type ${RUN_CV_TYPES[0]}"
else
    RUN_CV_TYPES=("${EVAL_CV_TYPES[@]}")
    echo "No --array: running all ${#EVAL_CV_TYPES[@]} cv_types serially"
fi
echo ""

for CV_TYPE in "${RUN_CV_TYPES[@]}"; do
    echo "============================================================"
    echo "  Running eval_finetuning with --cv_type $CV_TYPE"
    echo "============================================================"

    $PYTHON_PATH eval_finetuning.py \
        --model_dir "$MODEL_DIR" \
        --embeddings_paths "${EMBEDDINGS_PATHS[@]}" \
        --pca_variance "$PCA_VARIANCE" \
        $POST_PCA_ARGS \
        --datasets "${DATASETS[@]}" \
        --sl_path "$SL_PATH" \
        --base_dir "$BASE_DIR" \
        --cv_type "$CV_TYPE" \
        --ft_modes "${FT_MODES[@]}" \
        --ft_epochs "$FT_EPOCHS" \
        --ft_patience "$FT_PATIENCE" \
        --ft_lr "$FT_LR" \
        --ft_batch_size "$FT_BATCH_SIZE" \
        --ft_weight_decay "$FT_WEIGHT_DECAY" \
        --train_frac "$TRAIN_FRAC" \
        --val_frac "$VAL_FRAC" \
        --split_seed "$SPLIT_SEED" \
        $SLDB_FLAG \
        $PER_HEAD_FLAG

    echo ""
done

echo "Completed all CV types at: $(date)"
