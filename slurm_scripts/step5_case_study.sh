#!/bin/bash
#SBATCH --job-name=case_study_no_ESM
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/step5_case_study_%j.out
#SBATCH --error=logs/step5_case_study_%j.err

echo "=================================================="
echo "Step 5: Case Study WITHOUT ESM"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

# Load modules
module purge
module load gnu14 openmpi5 EasyBuild cmake openblas fftw
source /opt/rh/gcc-toolset-13/enable

# Python path explicitly set - no conda activation needed

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/pytorch/bin/python

# GPU is automatically set by SLURM via --gres=gpu:1

# Change to code directory
cd ../code

# Create output directory
mkdir -p ../outputs/case_study_without_ESM

# Run case study without ESM
echo "Starting case study without ESM embeddings..."
$PYTHON_PATH case_study.py \
	--epochs 300 \
	--hidden1 512 \
	--hidden2 256 \
	--dropout 0.2 \
	--output_dir ../outputs/case_study_without_ESM

echo "=================================================="
echo "Case study completed at: $(date)"
echo "Results saved to: ../outputs/case_study_without_ESM/"
echo "=================================================="
