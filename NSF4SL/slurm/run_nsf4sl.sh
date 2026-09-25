#!/bin/bash
#SBATCH --job-name=NSF4SL
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/nsf4sl_%j.out
#SBATCH --error=slurm/logs/nsf4sl_%j.err

# ============================================================================
# NSF4SL training (negative-sample-free contrastive SL prediction).
#
# Submit FROM the NSF4SL directory so relative paths (../data, results/) resolve:
#   cd NSF4SL && sbatch slurm/run_nsf4sl.sh              # runs cv1, cv2, cv3
#   cd NSF4SL && CV_TYPES="cv3" sbatch slurm/run_nsf4sl.sh   # one CV
#   cd NSF4SL && EMB=../data/all_genes_esm2.pt sbatch slurm/run_nsf4sl.sh
# ============================================================================

set -e

echo "=================================================="
echo "NSF4SL Training"
echo "Start: $(date) | Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "=================================================="

# gnu15 (gnu14 was retired from Lmod, 2026-07) and NO gcc-toolset-13 SCL —
# the gnu15 toolchain supplies the compiler. Under `set -e` a load of the
# retired gnu14 modulefile aborts the job seconds after allocation, before
# any training, leaving a log with nothing but the banner.
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs

EMB="${EMB:-../data/all_genes_kg_complex.pt}"
CV_TYPES="${CV_TYPES:-cv1 cv2 cv3}"

for CV in $CV_TYPES; do
    echo ""
    echo "--- Training NSF4SL / $CV (features: $EMB) ---"
    $PYTHON_PATH train_nsf4sl.py \
        --cv_type "$CV" \
        --embedding_path "$EMB" \
        --output_dir "results/nsf4sl_${CV}" \
        --seed 123
done

echo "=================================================="
echo "Completed at: $(date)"
echo "Results in: results/nsf4sl_cv*/results.json"
echo "=================================================="
