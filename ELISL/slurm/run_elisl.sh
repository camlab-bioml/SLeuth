#!/bin/bash
#SBATCH --job-name=ELISL
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request), like every comparison job. The
# root orchestrator passes GPU_NODES as --nodelist; by hand, pass
# --nodelist=<node> on the sbatch command line to move it.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00
# One task per CV: task i trains and grades cv(i+1).
#SBATCH --array=0-2
#SBATCH --output=ELISL/slurm/logs/elisl_%A_%a.out
#SBATCH --error=ELISL/slurm/logs/elisl_%A_%a.err

# ============================================================================
# ELISL (Tepeli et al., Bioinformatics 2024), context-free: Sequence_1024 +
# PPI_ec only — train + grade, one CV per array task.
#
# The authors' model is LightGBM random forests: it runs on the CPUs
# (--cpus-per-task, OMP_NUM_THREADS) and has no GPU code path. The job sits on
# gpu3 like every comparison job; its GPU stays idle. The SeqVec input build
# (ELISL/slurm/build_elisl_inputs.sh) is the step that uses one.
#
# Inputs, built once (ELISL/README.md, Inputs):
#   python ELISL/build_elisl_inputs.py download          # login node, internet
#   sbatch ELISL/slurm/build_elisl_inputs.sh             # GPU: SeqVec, node2vec,
#                                                        # data/elisl/gene_features.npz
# check_shared_inputs.py --models ELISL refuses a missing build or one made
# by other code / from other sources than its meta.json records.
#
# Outputs per CV:
#   ELISL/results/elisl_<cv>/   fold_<k>_<cv>_predictions.npy (8,844 x 8,844
#       float32), model_params.json (fold stamp, sizes, sources), results.json
#   sl_comparison/results/ELISL_<cv>.json   the row, graded by
#       sl_comparison/evaluate_model.py --model_name ELISL exactly as every
#       other row. A shared row (compare.py reads it from sl_comparison/results/
#       for every PCA arm; paper_tables lists it in tab_tiers as a published
#       peer; bootstrap_ci finds its matrices in ELISL/results/elisl_<cv>/).
#
# Stale rows. Each task FIRST moves its previous ELISL_<cv>.json to
# sl_comparison/results/stale_shared/ and deletes the previous matrices,
# before any check can exit, so a failed task leaves the row MISSING. The root
# orchestrator (run_benchmarking_siamese_vs_others.sh, since 2026-11-13)
# submits this array with the shared models (afterok on its folds job and,
# when it rebuilds them, the elisl_inputs job), moves the previous rows aside
# at submission, and chains every arm's compare on it (--skip-elisl leaves it
# out; --skip-shared reuses the rows). submit_pipeline.sh does not submit it.
#
# By hand, from the REPO ROOT:
#   sbatch --export=ALL ELISL/slurm/run_elisl.sh                 # cv1-cv3
#   sbatch --export=ALL --array=2 ELISL/slurm/run_elisl.sh       # cv3 only
#   NO_GRID_SEARCH=1 sbatch --export=ALL ELISL/slurm/run_elisl.sh  # ELRRF defaults
# then refresh each arm's table and paper tables (run_compare.sh,
# run_paper_tables.sh with that PCA_ARM).
# ============================================================================

set -u   # NOT -e: failures are handled below

CVS_ARR=(cv1 cv2 cv3)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    CV="${CVS_ARR[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$CV" ]; then
        echo "ERROR: array task $SLURM_ARRAY_TASK_ID has no CV (0-2)"; exit 1
    fi
else
    CV="${ELISL_CV:-cv3}"
fi
ROW="ELISL"

echo "=================================================="
echo "ELISL (context-free) / $CV"
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================="

module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f ELISL/train_elisl.py ]; then
    echo "FATAL: submit from the repo root (ELISL/train_elisl.py not found here)"
    exit 1
fi
. sl_comparison/slurm/python_path.sh   # the siamese jobs' interpreter (config.conf)
mkdir -p ELISL/slurm/logs sl_comparison/results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
SEED="${SEED:-42}"                     # the comparison's one seed (run_shared_models.sh)

OUT_JSON="sl_comparison/results/${ROW}_$CV.json"
PRED_DIR="ELISL/results/elisl_$CV"
# FIRST, before any check can stop this task: the previous row goes aside and
# the previous matrices go, so neither can be graded or tabulated as this run's.
if [ -f "$OUT_JSON" ]; then
    mkdir -p sl_comparison/results/stale_shared
    STALE_TO="sl_comparison/results/stale_shared/${ROW}_$CV.$(date +%Y%m%dT%H%M%S).json"
    echo "  moving previous row aside: $OUT_JSON -> $STALE_TO"
    mv -f "$OUT_JSON" "$STALE_TO"
fi
for _m in "$PRED_DIR"/fold_*_"$CV"_predictions.npy; do
    [ -f "$_m" ] || continue
    echo "  removing previous score matrix: $_m"
    rm -f -- "$_m"
done

if ! $PYTHON_PATH sl_comparison/slurm/check_shared_inputs.py --models ELISL; then
    echo "FATAL: ELISL inputs missing or stale (see above)."
    exit 1
fi
if ! $PYTHON_PATH -c "import lightgbm, skopt" 2>/dev/null; then
    echo "FATAL: lightgbm / scikit-optimize missing from $PYTHON_PATH's" \
         "environment (siamese_sl/slurm/reset_env.sh installs both)."
    exit 1
fi
if [ ! -d sl_comparison/folds/"$CV"/fold_0 ]; then
    echo "FATAL: sl_comparison/folds/$CV/fold_0 missing."
    echo "  Run sl_comparison/slurm/run_shared_folds.sh first."
    exit 1
fi

GS_ARGS=()
[ "${NO_GRID_SEARCH:-0}" = 1 ] && GS_ARGS=(--no_grid_search)

echo "--- training $ROW / $CV ---"
$PYTHON_PATH ELISL/train_elisl.py --cv_type "$CV" \
    --folds_dir sl_comparison/folds --output_dir "$PRED_DIR" \
    --seed "$SEED" "${GS_ARGS[@]}" \
    || { echo "FAIL: $ROW/$CV training"; exit 1; }

echo "--- scoring $ROW / $CV ---"
$PYTHON_PATH sl_comparison/evaluate_model.py --model_name "$ROW" --cv_type "$CV" \
    --pred_dir "$PRED_DIR" --out "$OUT_JSON" \
    || { echo "FAIL: $ROW/$CV eval"; rm -f "$OUT_JSON"; exit 1; }

echo ""
echo "=================================================="
echo "SUCCESS: $ROW / $CV -> $OUT_JSON"
echo "Completed at: $(date)"
echo "Rerun each arm's compare and paper tables to publish the row."
echo "=================================================="
