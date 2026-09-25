#!/bin/bash
#SBATCH --job-name=sl_compare
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Pinned to gpu3 (2026-08-10, by request). gpu3 threw a transient "CUDA
# unknown error" at torch init in 2026-07 and the whole tree was moved to
# gpu2 for that; if it recurs, the symptom is a torch.cuda init failure in
# the very first seconds of the job, not a training-time error.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00
#SBATCH --output=sl_comparison/slurm/logs/sl_compare_%A_%a.out
#SBATCH --error=sl_comparison/slurm/logs/sl_compare_%A_%a.err

# ============================================================================
# Four-model SL comparison — ARRAY worker, one (model, CV) per task.
#
# Was a single job running a 3x3 serial for-loop (nine trainings back to back,
# 2026-08 wall-clock ~6.5h). Now |MODELS| x |CVS| independent tasks, so SLURM
# overlaps them up to whatever the node and the array throttle allow.
#
#   task = cv_index * n_models + model_index      (CV-major, like the other arrays)
#
# Fold generation moved to run_shared_folds.sh — nine tasks racing to write
# sl_comparison/folds/ would interleave partial writes into the one answer key
# every model is graded against. The degree-prior control moved there too (it is
# CPU-only and must not hold a GPU slot). Collecting siamese and printing the
# table moved to run_compare.sh, which is the terminal job.
#
# RESILIENT: a task that fails is isolated to its own (model, CV) cell. The
# downstream compare is chained `afterany`, so a table is still produced from
# whatever succeeded.
#
# Submit from the REPO ROOT, AFTER run_shared_folds.sh:
#   sbatch --dependency=afterok:<folds_job> --array=0-8%3 \
#          sl_comparison/slurm/run_shared_models.sh
# or let run_benchmarking_siamese_vs_others.sh wire it.
# ============================================================================

set -u   # NOT -e: we handle the training/eval failure ourselves

MODELS=(SLGNN NSF4SL SLMGAE)
CVS_ARR=(${CVS:-cv1 cv2 cv3})
N_MODELS=${#MODELS[@]}
N_CVS=${#CVS_ARR[@]}
TOTAL=$((N_MODELS * N_CVS))

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: this is an ARRAY worker and was submitted without --array."
    echo "  Expected: sbatch --array=0-$((TOTAL - 1))%3 $0"
    echo "  ($N_MODELS models x $N_CVS CVs = $TOTAL tasks)"
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID is beyond the $TOTAL configured tasks. Nothing to do."
    exit 0
fi

CV="${CVS_ARR[$((SLURM_ARRAY_TASK_ID / N_MODELS))]}"
NAME="${MODELS[$((SLURM_ARRAY_TASK_ID % N_MODELS))]}"

echo "=================================================="
echo "SL comparison — task $SLURM_ARRAY_TASK_ID of $TOTAL"
echo "  Model: $NAME"
echo "  CV:    $CV"
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================="

# Match the siamese jobs' module setup exactly (siamese_sl/slurm/config.conf;
# config.sh is a dead, stale duplicate — do not use it as the reference):
# gnu15 (gnu14 was retired from Lmod, 2026-07) and NO gcc-toolset-13 SCL — the
# gnu15 toolchain supplies the compiler and the siamese jobs run fine without it.
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

PYTHON_PATH="${PYTHON_PATH:-/ddn_exa/campbell/kaiyang/pytorch/bin/python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p sl_comparison/slurm/logs sl_comparison/results

EMB="${EMB:-../data/all_genes_kg_complex.pt}"   # SLGNN + NSF4SL features
SEED="${SEED:-42}"                              # one seed for folds + all models

# The folds job must have run. Without this the task would train against a
# missing or half-written answer key and the failure would surface much later,
# as an unexplainable score.
if [ ! -d sl_comparison/folds/"$CV"/fold_0 ]; then
    echo "FATAL: sl_comparison/folds/$CV/fold_0 missing."
    echo "  Run sl_comparison/slurm/run_shared_folds.sh first."
    exit 1
fi

# --- Resolve this task's training command ------------------------------------
case "$NAME" in
    SLGNN)
        MODEL_DIR="SLGNN"
        PRED_DIR="SLGNN/results/slgnn_$CV"
        CMD="$PYTHON_PATH train_slgnn.py --cv_type $CV --folds_dir ../sl_comparison/folds \
             --embedding_path $EMB --output_dir results/slgnn_$CV --seed $SEED"
        ;;
    NSF4SL)
        MODEL_DIR="NSF4SL"
        PRED_DIR="NSF4SL/results/nsf4sl_$CV"
        CMD="$PYTHON_PATH train_nsf4sl.py --cv_type $CV --folds_dir ../sl_comparison/folds \
             --embedding_path $EMB --output_dir results/nsf4sl_$CV --seed $SEED"
        ;;
    SLMGAE)
        MODEL_DIR="SLMGAE-in-pytorch"
        PRED_DIR="SLMGAE-in-pytorch/results/slmgae_shared_$CV"
        CMD="$PYTHON_PATH train_slmgae_shared.py --cv_type $CV --folds_dir ../sl_comparison/folds \
             --output_dir results/slmgae_shared_$CV --seed $SEED"
        ;;
    *)
        echo "FATAL: unknown model '$NAME'"; exit 1 ;;
esac

echo "--- training $NAME / $CV ---"
( cd "$MODEL_DIR" && eval "$CMD" ) || { echo "FAIL: $NAME/$CV training"; exit 1; }

echo "--- scoring $NAME / $CV ---"
$PYTHON_PATH sl_comparison/evaluate_model.py --model_name "$NAME" --cv_type "$CV" \
    --pred_dir "$PRED_DIR" --out "sl_comparison/results/${NAME}_$CV.json" \
    || { echo "FAIL: $NAME/$CV eval"; exit 1; }

echo ""
echo "=================================================="
echo "SUCCESS: $NAME / $CV"
echo "Completed at: $(date)"
echo "=================================================="
