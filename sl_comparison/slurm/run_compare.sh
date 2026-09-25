#!/bin/bash
#SBATCH --job-name=sl_final_compare
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-08-10, by request). gpu3 threw a transient "CUDA
# unknown error" at torch init in 2026-07 and the whole tree was moved to
# gpu2 for that; if it recurs, the symptom is a torch.cuda init failure in
# the very first seconds of the job, not a training-time error.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:20:00
#SBATCH --output=sl_comparison/slurm/logs/final_compare_%j.out
#SBATCH --error=sl_comparison/slurm/logs/final_compare_%j.err

# ============================================================================
# Final four-model comparison: auto-discover siamese_sl's *_any results, then
# print the unified table. Runs after BOTH the shared-model job and siamese's
# combo job complete (chained by run_benchmarking_siamese_vs_others.sh).
# ============================================================================

set -e
# Match the siamese jobs' module setup exactly (siamese_sl/slurm/config.conf;
# config.sh is a dead, stale duplicate — do not use it as the reference):
# gnu15 (gnu14 was retired from Lmod, 2026-07) and NO gcc-toolset-13 SCL — the
# gnu15 toolchain supplies the compiler and the siamese jobs run fine without it.
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

PYTHON_PATH="${PYTHON_PATH:-/ddn_exa/campbell/kaiyang/pytorch/bin/python}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CVS="${CVS:-cv1 cv2 cv3}"
. sl_comparison/slurm/collect_and_regrade_siamese.sh

echo "--- four-model comparison table ---"
$PYTHON_PATH sl_comparison/compare.py --results_dir sl_comparison/results \
    --csv sl_comparison/results/comparison.csv

echo "Done: $(date)"
