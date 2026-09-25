#!/bin/bash
#SBATCH --job-name=sl_preflight
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-08-10, by request). gpu3 threw a transient "CUDA
# unknown error" at torch init in 2026-07 and the whole tree was moved to
# gpu2 for that; if it recurs, the symptom is a torch.cuda init failure in
# the very first seconds of the job, not a training-time error.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00
#SBATCH --output=sl_comparison/slurm/logs/preflight_%j.out
#SBATCH --error=sl_comparison/slurm/logs/preflight_%j.err

# ============================================================================
# Preflight smoke test — catch shape/dtype/DEVICE/path bugs in all three
# shared-fold models BEFORE the expensive full run. Runs on GPU (the cluster
# runs on GPU), so CUDA device-placement bugs are actually exercised — a
# CPU-only smoke test would miss exactly those.
#
# Builds a tiny fold set (top-300 genes, 2 folds, cv1) and runs one short GPU
# training of each model + the evaluator + the comparison table. Reports
# PASS/FAIL per model.
#
# Submit from the repo root:
#   sbatch sl_comparison/slurm/preflight.sh
# ============================================================================

set -u

echo "=================================================="
echo "PREFLIGHT smoke test (GPU, tiny 300-gene fold set)"
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================="

# Match the siamese jobs' module setup exactly (siamese_sl/slurm/config.conf;
# config.sh is a dead, stale duplicate — do not use it as the reference):
# gnu15 (gnu14 was retired from Lmod, 2026-07) and NO gcc-toolset-13 SCL — the
# gnu15 toolchain supplies the compiler and the siamese jobs run fine without it.
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

PYTHON_PATH="${PYTHON_PATH:-/ddn_exa/campbell/kaiyang/pytorch/bin/python}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
EMB="${EMB:-../data/all_genes_kg_complex.pt}"
SEED="${SEED:-42}"
PF=sl_comparison/folds_preflight
CV=cv1
FAIL=0

echo "--- building tiny fold set (seed $SEED) ---"
$PYTHON_PATH sl_comparison/prepare_folds.py --max_genes 300 --num_folds 2 \
    --cv_types "$CV" --out "$PF" --seed "$SEED" --verify || { echo "FAIL: fold gen"; exit 1; }

run () {   # name  dir  "cmd..."  pred_subdir
    local name="$1" dir="$2" cmd="$3" pred="$4"
    echo ""; echo "--- $name (GPU) ---"
    ( cd "$dir" && eval "$cmd" ); local rc=$?
    if [ $rc -ne 0 ]; then echo "FAIL: $name training (exit $rc)"; FAIL=1; return; fi
    $PYTHON_PATH sl_comparison/evaluate_model.py --model_name "$name" --cv_type "$CV" \
        --folds_dir "$PF" --pred_dir "$pred" \
        --out "sl_comparison/results_preflight/${name}_$CV.json" \
        && echo "PASS: $name" || { echo "FAIL: $name eval"; FAIL=1; }
}

mkdir -p sl_comparison/results_preflight

run SLGNN SLGNN \
  "$PYTHON_PATH train_slgnn.py --cv_type $CV --folds_dir ../$PF --embedding_path $EMB \
     --output_dir results/pf_slgnn --dim 16 --n_hops 1 --n_factors 2 \
     --epochs 2 --eval_interval 1 --early_stop 5 --seed $SEED" \
  "SLGNN/results/pf_slgnn"

run NSF4SL NSF4SL \
  "$PYTHON_PATH train_nsf4sl.py --cv_type $CV --folds_dir ../$PF --embedding_path $EMB \
     --output_dir results/pf_nsf4sl --latent_size 32 --batch_size 64 \
     --epochs 2 --eval_interval 1 --early_stop 5 --seed $SEED" \
  "NSF4SL/results/pf_nsf4sl"

run SLMGAE SLMGAE-in-pytorch \
  "$PYTHON_PATH train_slmgae_shared.py --cv_type $CV --folds_dir ../$PF \
     --output_dir results/pf_slmgae --hidden1 32 --hidden2 16 \
     --epochs 2 --eva_epochs 1 --early_stopping 5 --seed $SEED" \
  "SLMGAE-in-pytorch/results/pf_slmgae"

echo ""; echo "--- comparison table (preflight) ---"
$PYTHON_PATH sl_comparison/compare.py --results_dir sl_comparison/results_preflight || FAIL=1

echo ""
echo "=================================================="
if [ $FAIL -eq 0 ]; then
    echo "PREFLIGHT PASSED — all three models ran + scored on GPU."
    echo "Safe to submit the full run."
else
    echo "PREFLIGHT FAILED — fix the reported model(s) before the full run."
fi
echo "Completed at: $(date)"
echo "=================================================="
exit $FAIL
