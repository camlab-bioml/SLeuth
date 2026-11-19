#!/bin/bash
#SBATCH --job-name=eval_finetuning
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
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
# Chained from the combo it covers EVERY variant in SIAMESE_VARIANTS
# (config.conf): one array task per (eval cv_type, variant), each evaluating
# that variant's cv3 combo (EVAL_VARIANT -> its latest_combo_cv3.txt marker).
#
# Produces under $MODEL_DIR/eval_finetuning_<cv_type>/ (one directory per
# cv_type in EVAL_CV_TYPES — the script never passes --out_dir, so
# eval_finetuning.py derives the name from the cv_type). MODEL_DIR is the
# variant's own combo dir, so the variants never collide:
#   config_used.json
#   gene_coverage.csv
#   zero_shot_metrics.csv  / zero_shot_summary.csv
#   finetuned_metrics.csv  / finetuned_summary.csv
#   {dataset}_predictions.csv  / {dataset}_finetuned_predictions_{mode}.csv
#   {dataset}_sl_probabilities.csv            every screen pair x every head
#                                             (cell line): zero-shot probability
#   {dataset}_finetuned_allpairs_{mode}.csv   every screen pair, with its split:
#                                             the fine-tuned predicted value
#   finetuned_models/{dataset}_{mode}_fold{k}.pt   the fine-tuned weights
#                                             (export_pair_predictions.sh scores
#                                             all gene pairs with them)
#
# Usage, from siamese_sl/. The job is ONE PCA arm's (config.conf PCA arms):
# a direct sbatch is the arm the submitting shell exports (arm none's where
# it exports none) unless PCA_ARM is passed, and each variant's combo comes
# from the arm's marker <arm root>/<prefix>latest_combo_cv3.txt (<arm root> =
# RESULTS_ROOT: results/ for arm none, results/pca_<arm>/ otherwise), so
# without PCA_ARM, from a shell that exports none, the first two forms below
# evaluate, and rewrite, arm none's combos, and the third is logged as arm
# none's (2026-10-21). The arm goes in sbatch's environment, as the orchestrators
# pass it:
#   # every variant x cv_type of the arm, as the orchestrators chain it
#   # (|EVAL_CV_TYPES| x |SIAMESE_VARIANTS| tasks, 12 by default):
#   PCA_ARM=<arm> sbatch --export=ALL --array=0-11 slurm/eval_finetuning.sh
#
#   # one variant's cv3 combo (serial over EVAL_CV_TYPES):
#   EVAL_VARIANT=bilinear:pooled PCA_ARM=<arm> sbatch --export=ALL slurm/eval_finetuning.sh
#
#   # an explicit combo (PCA_ARM then only names the arm in the banner and
#   # log; the checkpoints carry their own PCA):
#   EVAL_MODEL_DIR=results/pca_<arm>/<combo>_cv3 PCA_ARM=<arm> sbatch --export=ALL slurm/eval_finetuning.sh
#
#   # via the pipeline orchestrator (standalone stage): it takes the arm from
#   # the combo's directory, and queues that combo's pair export after it
#   bash slurm/submit_pipeline.sh --eval-only results/pca_<arm>/best_cat_..._cv3
#
# A re-eval by hand clears and rewrites <combo>/eval_finetuning_<cv>/, the
# fine-tuned models (finetuned_models/) included, so that arm's pair export
# must be rerun after it (PCA_ARM=<arm> sbatch --export=ALL
# slurm/export_pair_predictions.sh): its manifest records the sha256 of the
# models it scored, and open_pair_predictions refuses it once they changed.
# So must the arm's external-transfer tables, which are built from these
# outputs (from the repo root, PCA_ARM=<arm> sbatch --export=ALL
# sl_comparison/slurm/run_external_transfer.sh, then its paper tables:
# paper_tables lists tables built from an earlier eval).
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"

# ============================================================================
# Array task -> (variant, eval cv_type), BEFORE eval_finetuning.conf resolves
# MODEL_DIR from the variant's marker.
#   Chained (no EVAL_MODEL_DIR / EVAL_VARIANT): |EVAL_CV_TYPES| x
#       |SIAMESE_VARIANTS| tasks, task = cv_index * n_variants + variant_index
#   --eval-only / EVAL_MODEL_DIR: one explicit model, task = cv_index
#   EVAL_VARIANT set in the env: that one variant, task = cv_index (the
#       orchestrators never set it; decoding the variant from the task id
#       would silently evaluate other variants than the one asked for)
# Needs only n_variants (config.conf), so the cv bound is checked further down
# once eval_finetuning.conf has defined EVAL_CV_TYPES. With an override, a task
# past the cv list FAILS: no launcher sizes an override array beyond
# |EVAL_CV_TYPES| (--eval-only counts it from the same conf), so such a task
# means an override inherited into the chained |cv| x |variants| array, which
# would otherwise evaluate one model and report success for the rest.
# ============================================================================
EVAL_TASK_CV_INDEX=""
EVAL_OVERRIDE=0
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ -z "${EVAL_MODEL_DIR:-}" ] && [ -z "${EVAL_VARIANT:-}" ]; then
        N_VAR=${#SIAMESE_VARIANTS[@]}
        EVAL_VARIANT="${SIAMESE_VARIANTS[$((SLURM_ARRAY_TASK_ID % N_VAR))]}"
        EVAL_TASK_CV_INDEX=$((SLURM_ARRAY_TASK_ID / N_VAR))
    else
        EVAL_TASK_CV_INDEX=$SLURM_ARRAY_TASK_ID
        EVAL_OVERRIDE=1
    fi
fi
source "$SLURM_SUBMIT_DIR/slurm/eval_finetuning.conf"

sleep 5

# ============================================================================
# Validate inputs
# ============================================================================
if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    if [ -n "${EVAL_MODEL_DIR:-}" ]; then
        echo "ERROR: EVAL_MODEL_DIR not found: $EVAL_MODEL_DIR"
    else
        echo "ERROR: MODEL_DIR not found: ${MODEL_DIR:-(none)}  [variant $EVAL_VARIANT]"
        echo "  Marker $EVAL_MARKER missing or stale — did this variant's cv3 combo train?"
    fi
    echo "  Set it via eval_finetuning.conf or EVAL_MODEL_DIR env var."
    exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json missing — is this a trained combo?"
    exit 1
fi
# train.py writes config.json at start and fold_<k>_best.pt at every val
# improvement, but results.json only after ALL folds finish: without it this is
# a crashed or still-running combo, whose partial checkpoints eval_finetuning.py
# would otherwise score as if the run had finished.
if [ ! -f "$MODEL_DIR/results.json" ]; then
    echo "ERROR: $MODEL_DIR/results.json missing — the combo did not finish training."
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

# Every modality must exist: the checkpoint's per-fold preprocessing_transform
# covers all of them, so fail fast here, before eval_finetuning.py clears the
# previous outputs and loads anything.
for p in "${EMBEDDINGS_PATHS[@]}"; do
    if [ ! -f "$p" ]; then
        echo "ERROR: embedding file missing: $p"
        exit 1
    fi
done

# ============================================================================
# Build post-PCA argument (legacy fallback only)
# ============================================================================
# Every checkpoint train.py writes now carries its fold's own
# preprocessing_transform (including whatever PCA its arm trained with: within
# each modality, on the concatenation, at a variance target or an exact
# PCA_JOINT_DIM count), and eval_finetuning.py applies that. The value below is
# read only for a checkpoint that predates the transform (eval_finetuning.conf
# LEGACY_*). eval_finetuning.py has no --post_pca_dim flag, so an exact count
# is deliberately NOT forwarded: passing one made argparse reject every eval
# task whenever the width was pinned for training.
if [ -n "${LEGACY_POST_PCA_VARIANCE:-}" ]; then
    POST_PCA_ARGS="--post_pca_variance $LEGACY_POST_PCA_VARIANCE"
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

echo "PCA arm:          $PCA_ARM ($(pca_arm_label "$PCA_ARM"))"
echo "MODEL_DIR:        $MODEL_DIR"
if [ -n "${EVAL_MODEL_DIR:-}" ]; then
    echo "VARIANT:          (explicit EVAL_MODEL_DIR)"
else
    echo "VARIANT:          $EVAL_VARIANT  (marker $EVAL_MARKER)"
fi
echo "EMBEDDINGS (${#EMBEDDINGS_PATHS[@]}):"
for p in "${EMBEDDINGS_PATHS[@]}"; do echo "    $p"; done
echo "Legacy PCA:       --pca_variance $LEGACY_PCA_VARIANCE ${POST_PCA_ARGS:-(no post-concat PCA)}  (only for a checkpoint without its own preprocessing_transform; every current one carries it)"
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
    if [ "$EVAL_TASK_CV_INDEX" -ge "${#EVAL_CV_TYPES[@]}" ]; then
        if [ "$EVAL_OVERRIDE" = 1 ]; then
            echo "ERROR: task $SLURM_ARRAY_TASK_ID is beyond the ${#EVAL_CV_TYPES[@]} cv_types with"
            echo "  EVAL_MODEL_DIR/EVAL_VARIANT set: an override inherited from the calling"
            echo "  shell collapsed a chained cv_type x variant array to one model. Unset both"
            echo "  and resubmit (or use --eval-only / a direct sbatch without --array)."
            exit 1
        fi
        echo "Task $SLURM_ARRAY_TASK_ID -> cv index $EVAL_TASK_CV_INDEX, beyond the ${#EVAL_CV_TYPES[@]} configured cv_types. Nothing to do."
        exit 0
    fi
    RUN_CV_TYPES=("${EVAL_CV_TYPES[$EVAL_TASK_CV_INDEX]}")
    echo "Array task $SLURM_ARRAY_TASK_ID -> cv_type ${RUN_CV_TYPES[0]}"
else
    RUN_CV_TYPES=("${EVAL_CV_TYPES[@]}")
    echo "No --array: running all ${#EVAL_CV_TYPES[@]} cv_types serially"
fi
echo ""

# A cv_type that exits non-zero (a skipped screen, or only an optional all-pair
# output missing) must not stop the cv_types after it under `set -e`: each one
# writes its own directory, as the array tasks do. The job still ends with a
# non-zero status.
EVAL_RC=0
for CV_TYPE in "${RUN_CV_TYPES[@]}"; do
    echo "============================================================"
    echo "  Running eval_finetuning with --cv_type $CV_TYPE"
    echo "============================================================"

    $PYTHON_PATH eval_finetuning.py \
        --model_dir "$MODEL_DIR" \
        --embeddings_paths "${EMBEDDINGS_PATHS[@]}" \
        --pca_variance "$LEGACY_PCA_VARIANCE" \
        $POST_PCA_ARGS \
        --datasets "${DATASETS[@]}" \
        --sl_path "$SL_PATH" \
        ${NEG_PATH:+--neg_path "$NEG_PATH"} \
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
        $PER_HEAD_FLAG || EVAL_RC=$?

    echo ""
done

echo "Completed all CV types at: $(date)  (exit status $EVAL_RC)"
exit "$EVAL_RC"
