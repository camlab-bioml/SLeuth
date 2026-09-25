#!/bin/bash
#SBATCH --job-name=SLGNN
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/slgnn_%j.out
#SBATCH --error=slurm/logs/slgnn_%j.err

# ============================================================================
# SLGNN training (adapted disentangled factor-aware GAT for SL prediction).
#
# Submit FROM the SLGNN directory so relative paths (../data, results/) resolve:
#   cd SLGNN && sbatch slurm/run_slgnn.sh                # runs cv1, cv2, cv3
#   cd SLGNN && CV_TYPES="cv3" sbatch slurm/run_slgnn.sh     # one CV
#   cd SLGNN && EMB=../data/all_genes_esm2.pt sbatch slurm/run_slgnn.sh
# ============================================================================

set -e

echo "=================================================="
echo "SLGNN Training (adapted, pure-PyTorch)"
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
    echo "--- Training SLGNN / $CV (features: $EMB) ---"
    $PYTHON_PATH train_slgnn.py \
        --cv_type "$CV" \
        --embedding_path "$EMB" \
        --output_dir "results/slgnn_${CV}" \
        --seed 123
done

echo "=================================================="
echo "Completed at: $(date)"
echo "Results in: results/slgnn_cv*/results.json"
echo "=================================================="
