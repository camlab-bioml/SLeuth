#!/bin/bash
#SBATCH --job-name=pair_export
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/pair_export_%j.out
#SBATCH --error=slurm/logs/pair_export_%j.err
# One pair_export job of a user at a time: two exports of one combo would
# clear each other's files (a recovery `submit_pipeline.sh --eval-only` beside
# a queued run, say). A --dependency on the sbatch command line REPLACES this
# line, so both orchestrators spell `,singleton` out after their afterany list.
#SBATCH --dependency=singleton
# ============================================================================
# All-gene-pair predictions — one job per PCA arm, every variant
# ============================================================================
# For every variant in SIAMESE_VARIANTS x PAIR_EXPORT_CVS: take the combo
# named by this run's <arm root>/<prefix>latest_combo_<cv>.txt marker (the
# job's PCA arm, PCA_ARM: <arm root> is config.conf's RESULTS_ROOT, results/ for
# arm none and results/pca_<arm>/ otherwise; the orchestrators queue one
# export per arm, serialised by the singleton dependency) and write
# (export_pair_predictions.py, no retraining) into <combo>/pair_predictions/:
#
#   sl_prob_<HEAD>.npy   predicted SL probability of EVERY gene pair, one
#                        n_genes x n_genes matrix per cell-line head (K562,
#                        JURKAT, A549, HELA, A375, 293T, PC9, OTHER) plus ANY
#                        for a per_head variant; ANY only for a pooled variant
#                        (it has no cell-line embedding)
#   finetuned_<Screen>_<mode>_<split>.npy
#                        the predicted screen value (Adamson: Gamma) of every
#                        gene pair from the models eval_finetuning.py
#                        fine-tuned on that screen, scored through the
#                        screen's cell-line head (K562 for Adamson); cv3 combo
#                        only, the one eval_finetuning.py fine-tunes
#   genes.txt / genes.tsv   row/column order
#   manifest.json        what every file is, written last
#
# What is exported, the gene universe and the disk it takes are set in
# slurm/export_pair_predictions.conf (PAIR_EXPORT_*).
#
# Nothing downstream reads these files: the job is chained LAST by both
# orchestrators (afterany on everything else they queue), so it can neither
# delay a job of that submission nor, should the disk fill, break one: every
# table of the root DAG. submit_pipeline.sh queues neither the cell-line
# ablation nor the sl_comparison jobs: when those are submitted by hand, pass
# it --skip-pair-export and sbatch this job after them (sl_comparison/README.md
# step 9).
#
# manifest.json is removed first and written last, so an export that dies
# reads as unfinished; a finished one records the sha256 of the checkpoints
# (and fine-tuned models) it scored, and
# export_pair_predictions.open_pair_predictions refuses it once they have
# changed. Memory: the fold-mean accumulators take at most PAIR_EXPORT_RAM_GB
# (default 40 of this job's 64G; a larger request runs in several passes).
#
# Submit from siamese_sl/. The job is ONE PCA arm's, the PCA_ARM in sbatch's
# environment as the orchestrators pass it (2026-10-21): a direct sbatch
# without it is the arm the submitting shell exports, arm none's (its markers
# in results/) only where none or nothing is. So for arm none PCA_ARM=none,
# never nothing (an arm's hand recipe leaves PCA_ARM exported), and
# results/<combo> below:
#   PCA_ARM=<arm> sbatch --export=ALL slurm/export_pair_predictions.sh
#   PAIR_EXPORT_CVS="cv1 cv2 cv3" PCA_ARM=<arm> sbatch --export=ALL slurm/export_pair_predictions.sh
#   # one explicit combo, whatever the markers say (PCA_ARM then only names
#   # the arm in the banner and log). It is taken to be an EVALUATED combo
#   # whatever CV it was trained under: the fine-tuned sets of
#   # PAIR_EXPORT_FT_SPLITS are requested from it.
#   PAIR_EXPORT_MODEL_DIR=results/pca_<arm>/<combo>_cv3 PCA_ARM=<arm> sbatch --export=ALL slurm/export_pair_predictions.sh
#   # ... one eval_finetuning.py never ran on (a cv1 / cv2 combo, an ablation,
#   # a --skip-eval run): probabilities only. Without PAIR_EXPORT_FT_SPLITS=none
#   # the job writes the probabilities and the manifest, then exits 1.
#   PAIR_EXPORT_FT_SPLITS=none PAIR_EXPORT_MODEL_DIR=results/pca_<arm>/<combo>_cv1 PCA_ARM=<arm> sbatch --export=ALL slurm/export_pair_predictions.sh
# Exit 0 when every requested (variant, CV) was exported in full, else 1.
# ============================================================================

set -e
# siamese_sl/: the submit dir under sbatch, else this script's parent.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/slurm/config.conf"
source "$SUBMIT_DIR/slurm/export_pair_predictions.conf"
set +e

OUT_SUBDIR="pair_predictions"
# The combo eval_finetuning.py fine-tunes (eval_finetuning.conf EVAL_MARKER).
EVAL_COMBO_CV="cv3"

flag_on() { [ "$1" = "1" ] || [ "$1" = "true" ]; }

# The check both orchestrators also run at submission
# (export_pair_predictions.conf); its ERROR lines go to this job's .out.
check_pair_export 2>&1 || exit 1

COMMON_ARGS=(--universe "$PAIR_EXPORT_UNIVERSE" --dtype "$PAIR_EXPORT_DTYPE"
             --ram_gb "$PAIR_EXPORT_RAM_GB")
if [ "$PAIR_EXPORT_UNIVERSE" = "shared" ]; then
    COMMON_ARGS+=(--shared_genes "${SHARED_FOLDS:-../sl_comparison/folds}/genes.txt")
fi
flag_on "$PAIR_EXPORT_PER_FOLD" && COMMON_ARGS+=(--per_fold)
flag_on "$PAIR_EXPORT_LOGITS" && COMMON_ARGS+=(--also_logits)

echo "PCA arm:   $PCA_ARM ($(pca_arm_label "$PCA_ARM")): combos of $RESULTS_ROOT/"
echo "Variants:  ${SIAMESE_VARIANTS[*]}"
echo "Combos:    $PAIR_EXPORT_CVS   (PAIR_EXPORT_CVS)"
echo "Universe:  $PAIR_EXPORT_UNIVERSE   dtype $PAIR_EXPORT_DTYPE   per-fold ${PAIR_EXPORT_PER_FOLD}   logits ${PAIR_EXPORT_LOGITS}"
echo "Fine-tuned: splits [$PAIR_EXPORT_FT_SPLITS]  datasets [$PAIR_EXPORT_FT_DATASETS]  modes [$PAIR_EXPORT_FT_MODES]  (the $EVAL_COMBO_CV combo only)"
echo ""

# --- the (label, combo dir, training cv) list --------------------------------
LABELS=()
DIRS=()
CVS=()
MISSING=()
if [ -n "${PAIR_EXPORT_MODEL_DIR:-}" ]; then
    LABELS+=("(PAIR_EXPORT_MODEL_DIR)")
    DIRS+=("$PAIR_EXPORT_MODEL_DIR")
    CVS+=("")
else
    for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
        VPREFIX=$(siamese_variant_prefix "$VARIANT")
        for CV in $PAIR_EXPORT_CVS; do
            MARKER="$RESULTS_ROOT/${VPREFIX}latest_combo_${CV}.txt"
            DIR=""
            [ -f "$MARKER" ] && DIR=$(head -n 1 "$MARKER" | tr -d '\r')
            if [ -n "$DIR" ] && [ -d "$DIR" ]; then
                LABELS+=("$VARIANT/$CV")
                DIRS+=("$DIR")
                CVS+=("$CV")
            elif [ -n "$DIR" ]; then
                echo "!! MISSING $VARIANT/$CV: marker $MARKER names $DIR, which does not exist"
                MISSING+=("$VARIANT/$CV")
            else
                echo "!! MISSING $VARIANT/$CV: no marker $MARKER (the combo did not train)"
                MISSING+=("$VARIANT/$CV")
            fi
        done
    done
fi

# GPU by policy (every job of this tree runs on CUDA), not for speed: one
# 19,397^2 head-and-fold matrix through the pair-feature term takes seconds
# even on a CPU, and the job is bounded by writing its matrices (42 GB by
# default). Checked once, here: export_pair_predictions.py refuses the
# --device cuda passed below on a node without CUDA (exit 1, once per combo);
# it falls back to the CPU only when --device is omitted, and then prints it.
if ! "$PYTHON_PATH" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is not available to $PYTHON_PATH on $(hostname); not exporting on CPU."
    exit 1
fi

N_OK=0
FAILED=()
for I in "${!DIRS[@]}"; do
    LABEL="${LABELS[$I]}"
    DIR="${DIRS[$I]}"
    CV="${CVS[$I]}"
    echo "============================================================================"
    echo "$LABEL  ->  $DIR/$OUT_SUBDIR/"
    echo "============================================================================"
    # train.py writes config.json at start and results.json only after every
    # fold saved its checkpoint: without it this is a crashed or still-running
    # combo (the check eval_finetuning.sh makes).
    if [ ! -f "$DIR/config.json" ] || [ ! -f "$DIR/results.json" ]; then
        echo "!! $LABEL: $DIR lacks config.json / results.json (the combo did not finish training)"
        FAILED+=("$LABEL")
        echo ""
        continue
    fi

    # The embedding files the combo was trained on, in the training order,
    # rebased under the current $DATA_DIR (as eval_finetuning.sh resolves them).
    export DATA_DIR
    mapfile -t EMBEDDINGS_PATHS < <(
        "$PYTHON_PATH" -c "
import json, os, sys
cfg = json.load(open(sys.argv[1]))
paths = cfg.get('embeddings_paths') or ([cfg['embeddings_path']] if cfg.get('embeddings_path') else [])
for p in paths:
    print(os.path.join(os.environ['DATA_DIR'], os.path.basename(p)))
" "$DIR/config.json"
    )
    if [ ${#EMBEDDINGS_PATHS[@]} -eq 0 ]; then
        echo "!! $LABEL: could not read embeddings_paths from $DIR/config.json"
        FAILED+=("$LABEL")
        echo ""
        continue
    fi
    EMB_MISSING=""
    for p in "${EMBEDDINGS_PATHS[@]}"; do
        [ -f "$p" ] || EMB_MISSING="$EMB_MISSING $p"
    done
    if [ -n "$EMB_MISSING" ]; then
        echo "!! $LABEL: embedding file(s) missing:$EMB_MISSING"
        FAILED+=("$LABEL")
        echo ""
        continue
    fi

    # Fine-tuned models live under the combo eval_finetuning.py evaluated: the
    # cv3 one (or the explicit PAIR_EXPORT_MODEL_DIR). A requested split
    # without them is reported here and by export_pair_predictions.py, which
    # then exits 1 after writing the rest.
    FT_ARGS=()
    if [ "$PAIR_EXPORT_FT_SPLITS" != "none" ] && { [ "$CV" = "$EVAL_COMBO_CV" ] || [ -z "$CV" ]; }; then
        FT_DIRS=()
        for S in $PAIR_EXPORT_FT_SPLITS; do
            FT_DIRS+=("eval_finetuning_$S")
            [ -d "$DIR/eval_finetuning_$S/finetuned_models" ] \
                || echo "   NOTE: $DIR/eval_finetuning_$S/finetuned_models missing (that eval task failed, was skipped, or ran with --no_save_finetuned)"
        done
        if [ ${#FT_DIRS[@]} -gt 0 ]; then
            FT_ARGS+=(--finetuned_eval_dirs "${FT_DIRS[@]}")
            [ "$PAIR_EXPORT_FT_DATASETS" != "all" ] && FT_ARGS+=(--finetuned_datasets $PAIR_EXPORT_FT_DATASETS)
            [ "$PAIR_EXPORT_FT_MODES" != "all" ] && FT_ARGS+=(--finetuned_modes $PAIR_EXPORT_FT_MODES)
        fi
    fi

    $PYTHON_PATH export_pair_predictions.py \
        --model_dir "$DIR" \
        --embeddings_paths "${EMBEDDINGS_PATHS[@]}" \
        --device cuda \
        "${COMMON_ARGS[@]}" \
        "${FT_ARGS[@]}"
    RC=$?
    if [ $RC -eq 0 ] && [ -f "$DIR/$OUT_SUBDIR/manifest.json" ]; then
        N_OK=$((N_OK + 1))
    else
        echo "!! $LABEL: export_pair_predictions.py exited $RC"
        FAILED+=("$LABEL")
    fi
    echo ""
done

echo "============================================================================"
echo "pair export: $N_OK of $((${#DIRS[@]} + ${#MISSING[@]})) combo(s) exported in full"
echo "Completed at: $(date)"
if [ ${#MISSING[@]} -gt 0 ] || [ ${#FAILED[@]} -gt 0 ]; then
    [ ${#MISSING[@]} -gt 0 ] && echo "No combo to export:        ${MISSING[*]}"
    [ ${#FAILED[@]} -gt 0 ] && echo "Failed or incomplete:      ${FAILED[*]}"
    echo "============================================================================"
    exit 1
fi
echo "============================================================================"
exit 0
