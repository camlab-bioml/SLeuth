#!/bin/bash
#SBATCH --job-name=bench_emb
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission:
# siamese_sl/slurm/submit_pipeline.sh, the only one that submits this script,
# passes it as --nodelist on the command line, which replaces this line
# (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/bench_%A_%a.out
#SBATCH --error=slurm/logs/bench_%A_%a.err

# ============================================================================
# Embedding Benchmark — SLURM Array Worker
# ============================================================================
# Each array task trains the siamese model on ONE (embedding, CV) combination
# with its PCA arm's single-embedding PCA (config.conf pca_single_args: native
# dimension in arms none and joint, --pca_variance PCA_MODALITY_VARIANCE in
# per_modality and both), into the arm's Step-1 root STEP1_ROOT: the results
# root of the first arm of its pair, (none, joint) or (per_modality, both),
# that PCA_ARMS lists (config.conf pca_step1_arm); results/ and
# results/pca_per_modality/ with all four arms.
#
# SLURM_ARRAY_TASK_ID is mapped to (embedding_index, cv_index) using:
#   embedding_index = task_id / num_cvs
#   cv_index        = task_id % num_cvs
#
# Array size is set dynamically by submit_pipeline.sh based on the number of
# embeddings in ALL_EMBEDDINGS[] and CV types in CV_TYPES[].
#
# NOTE: --array is NOT set here. It is passed by submit_pipeline.sh:
#   sbatch --array=0-<N-1>[%N] slurm/run_embedding_benchmark.sh
#   ([%N] only when MAX_CONCURRENT is set: config.conf array_throttle)
#
# Prerequisites:
#   - Embedding .pt files must exist (run_generate_embeddings.sh)
#   - Submitted as an array by submit_pipeline.sh. A direct sbatch is supported
#     too (slurm/README.md "Change SLURM partition/node", ../README.md Step 2a):
#     `export RUN_TOKEN=...` first — config.conf otherwise mints one per array
#     task and the summary marks every directory STALE — and the submission's
#     PCA_ARM and PCA_ARMS (they decide STEP1_ROOT), pass --array as above,
#     and submit slurm/run_embedding_benchmark_summary.sh yourself with
#     --export=ALL --dependency=afterany:<array job id>
#
# Output: $STEP1_ROOT/<embedding_type>_<cv_type>/results.json (+ run_token.txt)
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_embedding_benchmark.conf"

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
if [ "$EMB_INDEX" -ge "${#ALL_EMBEDDINGS[@]}" ]; then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID exceeds embedding count (${#ALL_EMBEDDINGS[@]})"
    exit 0
fi

# Extract embedding info and CV type
ENTRY="${ALL_EMBEDDINGS[$EMB_INDEX]}"
CV="${CV_TYPES[$CV_INDEX]}"
IFS=':' read -r ETYPE EFILE ENAME <<< "$ENTRY"
EMB_PATH="$DATA_DIR/$EFILE"

echo "Task $SLURM_ARRAY_TASK_ID: $ENAME / $CV"
echo "  Embedding: $EMB_PATH"
echo ""

# ============================================================================
# Clean previous results for this (embedding, CV) combination
# ============================================================================
# BEFORE the missing-embedding skip below, not after (as in
# run_best_per_category.sh): run_generate_embeddings.sh deletes every
# all_genes_*.pt at job start and tolerates a failed OPTIONAL_TYPES modality
# (run_generate_embeddings.conf), so an embedding CAN vanish between runs, and
# skipping first would leave the previous run's results.json for the summary to
# tabulate as fresh. Clearing first makes a missing embedding produce no row (or
# a MISSING cell) instead of an old one.
OUTPUT_DIR="$STEP1_ROOT/${ETYPE}_${CV}"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous results: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# ============================================================================
# Skip if embedding file doesn't exist
# ============================================================================
# Some embeddings may fail to generate (e.g., missing dependencies).
# Exit cleanly so the summary job still runs.

if [ ! -f "$EMB_PATH" ]; then
    echo "SKIP: $EFILE not found in $DATA_DIR"
    exit 0
fi

# ============================================================================
# Train
# ============================================================================

# Create output directory
$PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

echo "Training: $ETYPE / $CV  (PCA arm $PCA_ARM: $PCA_SINGLE_ARGS)"
echo "  Output: $OUTPUT_DIR"
echo "  Config: encoder=[$ENCODER_DIMS], epochs=$EPOCHS, patience=$PATIENCE"
# Single-embedding runs train the SELECTION_VARIANT only (config.conf): modality
# selection happens once and its winners are reused by every variant's combo.
echo "  Variant: $SELECTION_VARIANT (SELECTION_VARIANT)  ARCH_ARGS: ${ARCH_ARGS:-(none; train.py defaults)}"
echo "  Selection split: ${SELECTION_ARGS:-(off: SELECTION_FRAC empty)}"
echo ""

set +e
$PYTHON_PATH train.py \
    --embeddings_paths "$EMB_PATH" \
    --sl_path "$SL_PATH" \
    ${NEG_PATH:+--neg_pairs_path "$NEG_PATH"} \
    $CELL_LINE_ARGS \
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
    --preprocessing_fit_scope "${PREPROCESSING_FIT_SCOPE:-train}" \
    --pca_method "${PCA_METHOD:-plain}" \
    --siamese_encoder_type "${SIAMESE_ENCODER_TYPE:-residual}" \
    $PCA_SINGLE_ARGS \
    $(siamese_variant_args "$SELECTION_VARIANT") \
    $SELECTION_ARGS \
    $ARCH_ARGS
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
# This submission's stamp: the summary tabulates only directories carrying it
# (a task that never started never reached the rm -rf above).
printf '%s\n' "$RUN_TOKEN" > "$OUTPUT_DIR/run_token.txt"

AUROC=$($PYTHON_PATH -c "
import json
d = json.load(open('$OUTPUT_DIR/results.json'))['summary']
v, s = d['auroc_mean'], d['auroc_std']
print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')
" 2>/dev/null || echo "N/A")

# AUPRG: the stratified test AUPRG (cell-line mode; the checkpoint metric only
# under --select_metric auprg); .get() -> 'N/A' for an older results.json
# without it.
AUPRG=$($PYTHON_PATH -c "
import json
d = json.load(open('$OUTPUT_DIR/results.json'))['summary']
v, s = d.get('auprg_mean'), d.get('auprg_std')
print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')
" 2>/dev/null || echo "N/A")

PARAMS=$($PYTHON_PATH -c "
import json
d = json.load(open('$OUTPUT_DIR/results.json'))['summary']
print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")
" 2>/dev/null || echo "N/A")

echo "SUCCESS: $ETYPE / $CV  ($PCA_SINGLE_ARGS; PCA arm $PCA_ARM)"
echo "  AUROC(strat)=$AUROC  AUPRG(strat)=$AUPRG  Params=$PARAMS"
echo "  Completed at: $(date)"
