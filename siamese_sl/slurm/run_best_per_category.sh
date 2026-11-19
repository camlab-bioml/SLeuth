#!/bin/bash
#SBATCH --job-name=bestcat_emb
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
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/bestcat_%A_%a.out
#SBATCH --error=slurm/logs/bestcat_%A_%a.err

# ============================================================================
# Best-per-Category Step 1 — SLURM Array Worker
# ============================================================================
# Each array task trains the siamese model on ONE (embedding, CV) combination,
# under one variant, with its PCA arm's single-embedding PCA (config.conf
# pca_single_args: native dimension in arms none and joint, --pca_variance
# PCA_MODALITY_VARIANCE in per_modality and both; PCA_VARIANCE from the .conf
# file only names the output directory), into the arm's Step-1 root STEP1_ROOT:
# the results root of the arm that owns its pair's runs, the first arm of
# (none, joint) or (per_modality, both) that PCA_ARMS lists (config.conf
# pca_step1_arm). With all four arms that is results/ and
# results/pca_per_modality/; PCA_ARMS="joint both" puts them in
# results/pca_joint/ and results/pca_both/. Both arms of a pair rank the same
# runs. The orchestrators submit one array per owner, with PCA_ARM=<owner> and
# the submission's PCA_ARMS exported.
#
# SLURM_ARRAY_TASK_ID is mapped to (variant block, embedding_index, cv_index):
#   task = (block * num_embeddings + embedding_index) * num_cvs + cv_index
# Block 0 is SELECTION_VARIANT (the runs Step 2 ranks); with
# BESTCAT_EXTRA_VARIANTS=1 blocks 1.. are the other SIAMESE_VARIANTS, trained
# for the single-modality table only (bestcat_task_variants in
# run_best_per_category.conf). With the default 0 there is one block, i.e.
#   embedding_index = task_id / num_cvs,  cv_index = task_id % num_cvs
#
# NOTE: --array is NOT set here. Launchers size it with bestcat_n_tasks
# (run_best_per_category.conf):
#   sbatch --array=0-$(( $(bestcat_n_tasks) - 1 ))[%N] slurm/run_best_per_category.sh
#   ([%N] only when MAX_CONCURRENT is set: config.conf array_throttle)
#
# After all array tasks complete, run_best_per_category_combo.sh picks the
# best embedding per category and trains multi-modal combinations.
#
# Prerequisites:
#   - Embedding .pt files must exist (run_generate_embeddings.sh)
#   - Submitted as an array by submit_pipeline.sh and by the repo-root
#     run_benchmarking_siamese_vs_others.sh. A direct sbatch is supported too
#     (slurm/README.md "Change SLURM partition/node", ../README.md Step 2b):
#     `export RUN_TOKEN=...` first — config.conf otherwise mints one per array
#     task and the combo's Step 2 refuses every directory as another
#     submission's — and the submission's PCA_ARM and PCA_ARMS (together they
#     decide STEP1_ROOT, where the combo of each arm of the pair looks), pass
#     --array as above, and submit
#     slurm/run_best_per_category_combo.sh yourself with --export=ALL
#     --dependency=afterany:<array job id>
#
# Output: $STEP1_ROOT/pcavar<variance>_<embedding_type>_<cv_type>/results.json
#         (+ run_token.txt); other variants under $STEP1_ROOT/<solo prefix>pcavar...
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
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
# Map task ID → (variant, embedding, CV)
# ============================================================================
#   task_id = (block * num_embeddings + embedding_index) * num_cvs + cv_index
# (the same layout bestcat_n_tasks sizes the array for).

bestcat_task_variants
NUM_CVS=${#CV_TYPES[@]}
NUM_EMB=${#EMB_CATALOG[@]}
N_TASKS=$(bestcat_n_tasks)
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_TASKS" ]; then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID is beyond the $N_TASKS configured tasks"
    exit 0
fi
BLOCK=$((SLURM_ARRAY_TASK_ID / (NUM_EMB * NUM_CVS)))
EMB_INDEX=$(((SLURM_ARRAY_TASK_ID % (NUM_EMB * NUM_CVS)) / NUM_CVS))
CV_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CVS))
VARIANT="${BESTCAT_TASK_VARIANTS[$BLOCK]}"
SOLO_PREFIX=$(bestcat_solo_prefix "$VARIANT")

# Extract embedding info and CV type
ENTRY="${EMB_CATALOG[$EMB_INDEX]}"
CV="${CV_TYPES[$CV_INDEX]}"
IFS=':' read -r ETYPE EFILE ECAT <<< "$ENTRY"
EMB_PATH="$DATA_DIR/$EFILE"

echo "Task $SLURM_ARRAY_TASK_ID: $ETYPE ($ECAT) / $CV  [PCA arm $PCA_ARM: $PCA_SINGLE_ARGS; dir tag pcavar$PCA_VARIANCE]"
echo "  Embedding: $EMB_PATH"
if [ "$BLOCK" -gt 0 ]; then
    echo "  Variant block $BLOCK: $VARIANT (BESTCAT_EXTRA_VARIANTS=1; reporting only, never ranked)"
fi
echo ""

# ============================================================================
# Clean previous results for this (variant, embedding, CV, PCA) combination
# ============================================================================
# BEFORE the missing-embedding skip below, not after.
#
# run_generate_embeddings.sh deletes every all_genes_*.pt at job start and
# tolerates a failed OPTIONAL_TYPES modality (run_generate_embeddings.conf), so
# an embedding CAN vanish between runs. If
# the skip came first, this task would exit 0 with the PREVIOUS run's
# results.json still on disk. The combo's Step 2
# (run_best_per_category_combo.sh) ranks only
# <Step-1 root>/pcavar<v>_<type>_<SELECTION_CV>/results.json whose run_token.txt is
# this submission's (written below, after a successful train). A candidate with
# this run's token but another variant / signature / fold stamp always stops
# the job; a missing score (failed task, or embedding present but no result) or
# another run's token stops it unless BEST_CAT_ALLOW_MISSING=1; a
# never-generated embedding only warns.
# Clearing first makes a missing embedding produce NO candidate (a logged
# "not generated") instead of an old one, and a failed train leaves no token.
# (sl_comparison reads only this run's latest_combo markers; only the opt-in
# SIAMESE_BASELINE_DISCOVER=all sweep still rglobs results/.)
OUTPUT_DIR="$STEP1_ROOT/${SOLO_PREFIX}pcavar${PCA_VARIANCE}_${ETYPE}_${CV}"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous results: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# ============================================================================
# Skip if embedding file doesn't exist
# ============================================================================
# Some embeddings may fail to generate. Exit cleanly so the combo job runs.

if [ ! -f "$EMB_PATH" ]; then
    echo "SKIP: $EFILE not found in $DATA_DIR"
    exit 0
fi

# ============================================================================
# Train (the arm's single-embedding PCA)
# ============================================================================

# Create output directory
$PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

echo "Training: $ETYPE / $CV  ($PCA_SINGLE_ARGS: $([ "$PCA_SINGLE_ARGS" = --no_post_pca ] && echo "native dim, no PCA" || echo "PCA within the modality"))"
echo "  Output: $OUTPUT_DIR"
echo "  Config: encoder=[$ENCODER_DIMS], epochs=$EPOCHS, patience=$PATIENCE"
# Modality selection happens once, on SELECTION_VARIANT (config.conf), and its
# winners are reused by every variant's combo; other variants reach this script
# only as BESTCAT_EXTRA_VARIANTS=1 reporting runs.
echo "  Variant: $VARIANT$([ "$VARIANT" = "$SELECTION_VARIANT" ] && echo " (SELECTION_VARIANT)")  ARCH_ARGS: ${ARCH_ARGS:-(none; train.py defaults)}"
echo "  Selection split: ${SELECTION_ARGS:-(off: SELECTION_FRAC empty)}"
echo ""

# Single-embedding runs get the per-modality stage of the arm's PCA only
# (PCA_SINGLE_ARGS, config.conf): the joint PCA acts on a concatenation, and on
# one modality it would repeat the per-modality PCA or, after it, see a flat
# spectrum. The matrix-wise (global-scalar) normalize always applies.
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
    $(siamese_variant_args "$VARIANT") \
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
# This submission's stamp: combo Step 2 ranks only candidates carrying it.
printf '%s\n' "$RUN_TOKEN" > "$OUTPUT_DIR/run_token.txt"

# Reporting-only runs (block > 0) are read for their results.json alone, and
# their 313 MB-per-fold prediction matrices would add up to ~270 GB. Deleted
# only now: nonzero-parameter counting reloads the checkpoints during the run.
if [ "$BLOCK" -gt 0 ] && [ "${BESTCAT_X_KEEP_PREDICTIONS:-0}" != "1" ]; then
    rm -rf "$OUTPUT_DIR/checkpoints"
    rm -f "$OUTPUT_DIR"/fold_*_predictions.npy
    echo "  Checkpoints + prediction matrices removed (BESTCAT_X_KEEP_PREDICTIONS=1 keeps them)"
fi

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

echo "SUCCESS: $ETYPE / $CV  ($PCA_SINGLE_ARGS)  [$VARIANT; PCA arm $PCA_ARM]"
echo "  AUROC(strat)=$AUROC  AUPRG(strat)=$AUPRG  Params=$PARAMS"
echo "  Completed at: $(date)"
