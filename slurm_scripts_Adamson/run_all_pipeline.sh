#!/bin/bash

# Master script to run the complete SLMGAE pipeline
# This submits all jobs with proper dependencies

echo "=================================================="
echo "SLMGAE Complete Pipeline Submission"
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="

# Create log directory in project root
mkdir -p ../logs

# Step 1: Download protein sequences
echo "Submitting Step 1: Download protein sequences..."
JOB1=$(sbatch --parsable step1_download_protein_seq.sh)
echo "  Job ID: $JOB1"

# Step 2: Generate ESM embeddings (depends on Step 1)
echo "Submitting Step 2: Generate ESM embeddings..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 step2_generate_ESM_embedding.sh)
echo "  Job ID: $JOB2 (depends on $JOB1)"

# Step 3: Main training - both with and without ESM (depends on Step 2)
echo "Submitting Step 3: Main dataset training..."
JOB3A=$(sbatch --parsable --dependency=afterok:$JOB2 step3_train_main_with_ESM.sh)
echo "  Job ID: $JOB3A - Main with ESM (depends on $JOB2)"

JOB3B=$(sbatch --parsable step3_train_main_without_ESM.sh)
echo "  Job ID: $JOB3B - Main without ESM (can run immediately)"

# CV2 and CV3: Cross-validation experiments (can run immediately)
echo "Submitting CV2 and CV3: Cross-validation experiments..."
JOB3C=$(sbatch --parsable step3_train_CV2.sh)
echo "  Job ID: $JOB3C - CV2 (gene-based, can run immediately)"

JOB3D=$(sbatch --parsable step3_train_CV3.sh)
echo "  Job ID: $JOB3D - CV3 (pair-based, can run immediately)"

# ============================================================================
# OPTIONAL STEPS (Currently disabled - uncomment to enable)
# ============================================================================
# These steps are optional and can be run independently:
#   - BC (Breast Cancer) subset: Trains on 139-gene subset for validation
#   - Case study: Evaluates specific gene pairs for biological insights
# Uncomment below to include in pipeline
# ============================================================================

# Step 4: BC training without ESM (OPTIONAL)
# echo "Submitting Step 4: BC dataset training without ESM..."
# JOB4=$(sbatch --parsable step4_train_BC_without_ESM.sh)
# echo "  Job ID: $JOB4 - BC without ESM (can run immediately)"

# Step 5: Case study without ESM (OPTIONAL)
# echo "Submitting Step 5: Case study without ESM..."
# JOB5=$(sbatch --parsable step5_case_study.sh)
# echo "  Job ID: $JOB5 - Case study without ESM (can run immediately)"

echo ""
echo "=================================================="
echo "All jobs submitted!"
echo "=================================================="
echo "Pipeline structure:"
echo "  1. Download proteins: $JOB1"
echo "  2. Generate ESM: $JOB2 (after $JOB1)"
echo "  3. Main training:"
echo "     - With ESM: $JOB3A (after $JOB2)"
echo "     - Without ESM (CV1): $JOB3B (immediate)"
echo "     - CV2 (gene-based): $JOB3C (immediate)"
echo "     - CV3 (pair-based): $JOB3D (immediate)"
echo "  4. BC training: Optional (see script to enable)"
echo "  5. Case study: Optional (see script to enable)"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in: ../logs/"
echo "Check outputs in: ../outputs_Adamson/"
echo "=================================================="
