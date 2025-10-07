#!/bin/bash
#SBATCH --job-name=download_proteins
#SBATCH --partition=gpu_Prosmn
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6-12:00:00
#SBATCH --output=logs/step1_download_proteins_%j.out
#SBATCH --error=logs/step1_download_proteins_%j.err

echo "=================================================="
echo "Step 1: Download Protein Sequences"
echo "=================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Load modules
module purge
module load gnu13/13.2.0 openmpi5/5.0.3 EasyBuild/4.9.1 cmake/3.24.2 openblas/0.3.21 fftw/3.3.10
source /opt/rh/gcc-toolset-13/enable

# Python path explicitly set - no conda activation needed

# Python path
PYTHON_PATH=/ddn_exa/campbell/kaiyang/slmgae/bin/python

# Change to code directory
cd ../code

# Create necessary directories
mkdir -p ../protein_seq
mkdir -p ../logs

# Run protein sequence download
echo "Starting protein sequence download..."
$PYTHON_PATH fetch_all_proteins.py

# Check if successful
if [ -f "../protein_seq/main_protein_seq.fasta" ]; then
	echo "✅ Main FASTA created successfully"
	echo "Counting genes..."
	GENE_COUNT=$(grep -c "^>" ../protein_seq/main_protein_seq.fasta)
	echo "Total genes in FASTA: $GENE_COUNT"
else
	echo "❌ Failed to create main FASTA file"
	exit 1
fi

# Note: BC gene list will be auto-created by train_bc.py if needed
# (first 139 genes from List_Proteins_in_SL.txt)

echo "=================================================="
echo "Download completed at: $(date)"
echo "Next step: Run step2_generate_ESM_embedding.sh"
echo "=================================================="
