#!/bin/bash
#SBATCH --job-name=cellline_ablation
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: the root
# orchestrator (run_benchmarking_siamese_vs_others.sh), the only one that
# submits this script, passes it as --nodelist on the command line, which
# replaces this line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/cellline_ablation_%j.out
#SBATCH --error=slurm/logs/cellline_ablation_%j.err
# ============================================================================
# Post-hoc cell-line ablation (paper tab:cellline) — one job per PCA arm,
# every variant
# ============================================================================
# For every per_head variant in SIAMESE_VARIANTS x CV_TYPES: take the combo
# named by this run's <arm root>/<prefix>latest_combo_<cv>.txt marker, zero its
# cell_emb.weight (cell_line_ablation/zero_cell_emb.py, no retraining), export
# the zeroed model's 8,844 x 8,844 matrices and grade them with the shared
# evaluator (cell_line_ablation/grade.sh). The PCA arm is the job's PCA_ARM
# (config.conf): <arm root> = RESULTS_ROOT (results/ for arm none,
# results/pca_<arm>/ otherwise, under siamese_sl/) and <arm cmp> = CMP_ROOT
# (sl_comparison/results/ or sl_comparison/results/pca_<arm>/). Outputs:
#
#   siamese_sl/<arm root>/<prefix>cellline_ablation_<cv>/
#       results.json          rows, per-fold values, self-check (README.md)
#       fold_<k>_<cv>_predictions.npy, model_params.json, checkpoints/,
#       config.json           the zeroed model
#   <arm cmp>/<row>_nocell_<cv>.json
#       the graded 'removed' row; <row> = siamese_sl (pd:per_head) or
#       siamese_sl_<kernel>_per_head (e.g. siamese_sl_bilinear_per_head)
#
# pooled variants have no cell-line embedding: not applicable (their combos
# are the retrained no-cell-line counterparts), logged as N/A. With
# USE_CELL_LINES=0 nothing applies.
#
# Every row this job can write is moved to <arm cmp>/stale_siamese/ (and every
# <prefix>cellline_ablation_<cv>/ dir of the arm deleted) BEFORE anything runs,
# so a variant or CV that fails leaves its row MISSING, never a previous
# run's. Another arm's rows and dirs are never touched.
#
# Submit from siamese_sl/ (the root orchestrator does, afterok each arm's combo
# job, with the arm in sbatch's environment and a plain --export=ALL):
#   cd siamese_sl && PCA_ARM=<arm> sbatch --export=ALL cell_line_ablation/run_cell_line_ablation.sh
# Exit 0 when every applicable (variant, CV) row was graded, else 1.
# ============================================================================

set -e
# siamese_sl/: the submit dir under sbatch, else this script's parent's parent.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/slurm/config.conf"
set +e

# The arm's sl_comparison results root, from siamese_sl/ (CMP_ROOT is relative
# to the repo root).
SL_RESULTS="../$CMP_ROOT"
STALE_DIR="$SL_RESULTS/stale_siamese"
TS=$(date +%Y%m%dT%H%M%S)

# row name of a variant in sl_comparison (collect_siamese.py's convention)
variant_row() {
    local tag
    tag=$(siamese_variant_tag "$1")
    echo "siamese_sl${tag:+_$tag}"
}

echo "PCA arm: $PCA_ARM ($(pca_arm_label "$PCA_ARM")): combos of $RESULTS_ROOT/, rows -> $CMP_ROOT/"
echo "Variants: ${SIAMESE_VARIANTS[*]}   CVs: ${CV_TYPES[*]}"
echo "USE_CELL_LINES=$USE_CELL_LINES"
echo ""

# --- move this job's previous outputs aside, first ---------------------------
N_MOVED=0
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    ROW=$(variant_row "$VARIANT")
    for CV in "${CV_TYPES[@]}"; do
        F="$SL_RESULTS/${ROW}_nocell_${CV}.json"
        if [ -f "$F" ]; then
            mkdir -p "$STALE_DIR"
            mv -f "$F" "$STALE_DIR/${ROW}_nocell_${CV}.${TS}.json"
            N_MOVED=$((N_MOVED + 1))
        fi
        D="$RESULTS_ROOT/${VPREFIX}cellline_ablation_${CV}"
        [ -d "$D" ] && rm -rf "$D"
    done
done
[ "$N_MOVED" -gt 0 ] && echo "Moved $N_MOVED previous *_nocell row(s) to $STALE_DIR"

if [ "$USE_CELL_LINES" != "1" ]; then
    echo "N/A: USE_CELL_LINES=$USE_CELL_LINES trains single-output models with"
    echo "     no cell-line embedding; there is nothing to ablate."
    exit 0
fi

N_EXPECTED=0
N_GRADED=0
MISSING=()
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    ROW=$(variant_row "$VARIANT")
    echo "============================================================================"
    echo "Variant $VARIANT  (row ${ROW}_nocell, prefix '${VPREFIX}')"
    echo "============================================================================"
    case "$VARIANT" in
        *:per_head) ;;
        *)
            echo "N/A: $VARIANT has no cell-line embedding (pooled); its combo is"
            echo "     the retrained no-cell-line counterpart of the per_head variant."
            echo ""
            continue ;;
    esac

    ZARGS=()
    CVS_HAVE=()
    for CV in "${CV_TYPES[@]}"; do
        N_EXPECTED=$((N_EXPECTED + 1))
        MARKER="$RESULTS_ROOT/${VPREFIX}latest_combo_${CV}.txt"
        DIR=""
        [ -f "$MARKER" ] && DIR=$(head -n 1 "$MARKER" | tr -d '\r')
        if [ -n "$DIR" ] && [ -d "$DIR" ]; then
            ZARGS+=(--model_dir "$CV=$DIR")
            CVS_HAVE+=("$CV")
            echo "  $CV: $DIR"
        else
            if [ -n "$DIR" ]; then
                echo "  !! MISSING $VARIANT/$CV: marker $MARKER names $DIR, which does not exist"
            else
                echo "  !! MISSING $VARIANT/$CV: no marker $MARKER (the combo did not train)"
            fi
            echo "     -> no ${ROW}_nocell_${CV} row"
            MISSING+=("${ROW}_nocell/$CV")
        fi
    done
    if [ ${#CVS_HAVE[@]} -eq 0 ]; then
        echo ""
        continue
    fi

    echo ""
    $PYTHON_PATH cell_line_ablation/zero_cell_emb.py "${ZARGS[@]}" \
        --save_matrices \
        --out_dir "$RESULTS_ROOT" \
        --cv_dir_name "${VPREFIX}cellline_ablation_%s" \
        --model_name "$ROW"
    ZEXIT=$?
    [ $ZEXIT -ne 0 ] && echo "  WARNING: zero_cell_emb.py exited $ZEXIT for $VARIANT;" \
        "grading whatever CVs it completed (a failed CV has no matrices)"

    # grade.sh cds to the repo root; paths below are relative to it. Run it
    # through bash, not ./grade.sh: on 2026-09-23 a whole-tree SFTP upload
    # recreated it on the cluster without the executable bit (git records
    # 100755), and ./grade.sh failed with "Permission denied" after the
    # ablation itself had finished, leaving every _nocell row ungraded.
    MODEL_NAME="${ROW}_nocell" \
    PRED_DIR_TEMPLATE="siamese_sl/$RESULTS_ROOT/${VPREFIX}cellline_ablation_%s" \
    GRADE_CVS="${CVS_HAVE[*]}" \
    GRADE_OUT_DIR="$CMP_ROOT" \
    PYTHON_PATH="$PYTHON_PATH" \
        bash cell_line_ablation/grade.sh

    for CV in "${CVS_HAVE[@]}"; do
        if [ -f "$SL_RESULTS/${ROW}_nocell_${CV}.json" ]; then
            N_GRADED=$((N_GRADED + 1))
        else
            MISSING+=("${ROW}_nocell/$CV")
        fi
    done
    echo ""
done

echo "============================================================================"
echo "cell-line ablation: $N_GRADED of $N_EXPECTED applicable row(s) graded"
echo "Completed at: $(date)"
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "MISSING rows (the paper tables flag them): ${MISSING[*]}"
    echo "============================================================================"
    exit 1
fi
echo "============================================================================"
exit 0
