#!/bin/bash
# ============================================================================
# Shared Configuration for All SLURM Scripts
# ============================================================================
# Source this at the top of every job script AND from the login-node
# orchestrator (submit_pipeline.sh).
#
# Provides:
#   - Environment variables (TRANSFORMERS_NO_TF)
#   - Python interpreter path
#   - Project directory paths (WORK_DIR, BASE_DIR, DATA_DIR, CACHE_DIR)
#   - Key data file paths (ESM_PATH, GO_PATH, SL_PATH, etc.)
#   - Shared training hyperparameters (architecture, optimiser, regularization)
#   - Cross-validation types (CV_TYPES)
#   - Embedding catalog (ALL_EMBEDDINGS)
#   - Compute-node setup (module loads, cd, GPU info) — only inside SLURM jobs
#
# Per-job .conf files hold job-specific settings only:
#   - run_generate_embeddings.conf  (PRECOMPUTED_TYPES)
#   - run_embedding_benchmark.conf  (overrides, if any)
#   - run_best_per_category.conf    (PCA_VARIANCE, EMB_CATALOG)
#
# Usage:
#   # Inside a SLURM job script:
#   source "$SLURM_SUBMIT_DIR/slurm/config.sh"
#
#   # From the login node (submit_pipeline.sh):
#   source "$(dirname "$0")/config.sh"
# ============================================================================

# Prevent transformers from importing TensorFlow (numpy binary incompatibility)
export TRANSFORMERS_NO_TF=1

# Python environment
# Type: string (absolute path to python3 binary)
PYTHON_PATH="/ddn_exa/campbell/kaiyang/pytorch/bin/python3"

# ============================================================================
# Project directories
# ============================================================================
# Works both inside SLURM jobs (SLURM_SUBMIT_DIR is set by sbatch to the
# directory where the command was run) and from the login node (fall back
# to deriving from this file's location: slurm/config.sh → siamese_sl/).

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_DIR="$(dirname "$WORK_DIR")"          # SLMGAE-pytorch/
DATA_DIR="$BASE_DIR/data"                  # Final .pt embedding files
CACHE_DIR="$DATA_DIR/embeddings_cache"     # Downloaded raw data (STRING PPI, GO, model weights, etc.)

# ============================================================================
# Key data files
# ============================================================================
# Type: string (absolute paths, derived from directories above)

ESM_PATH="$DATA_DIR/all_genes_esm.pt"              # ESM-2 embeddings; also canonical gene list
GO_PATH="$DATA_DIR/all_genes_go.pt"                 # GO anc2vec embeddings (200d)
GAF_PATH="$BASE_DIR/uniprot_GO/goa_human.gaf.gz"    # Gene Ontology annotation file (GAF 2.2)
SL_PATH="$DATA_DIR/SL_SynLethDB_experimental.txt"   # SynLethDB 2.0 experimental SL pairs

# ============================================================================
# Shared training hyperparameters
# ============================================================================
# Used by both the embedding benchmark and best-per-category pipelines.
# Per-job .conf files can override any of these for independent tuning.

# --- Architecture ---
ENCODER_DIMS="256 128 64"
MODEL_ARGS="--model_type siamese --no-last-layer-bias"

# --- Training ---
EPOCHS=2000
BATCH_SIZE=256
LR=0.001
WEIGHT_DECAY=0.00001
DROPOUT=0.2
EVAL_INTERVAL=30
PATIENCE=10
NUM_FOLDS=5
POS_NEG_RATIO=1.0
SEED=42
WARMRESTART_T0=50
WARMRESTART_TMULT=2

# --- Regularization ---
# L1_LAMBDAS: per-layer proximal L1 (one value per ENCODER_DIMS layer)
L1_LAMBDAS="0.02 0.01 0.002"
# PD_EPSILON: positive-definiteness regularizer (0 to disable)
PD_EPSILON=0.001

# --- Cross-validation types ---
# cv1: edge split, cv2: gene split, cv3: pair split (hardest)
CV_TYPES=("cv1" "cv2" "cv3")

# ============================================================================
# Embedding catalog
# ============================================================================
# ALL_EMBEDDINGS: complete list (type:filename:display_name)
# Used by generation summary, embedding benchmark, and submit_pipeline.sh.
ALL_EMBEDDINGS=(
    "go:all_genes_go.pt:GO (anc2vec, 200d)"
    "geneformer:all_genes_geneformer.pt:Geneformer"
    "scgpt:all_genes_scgpt.pt:scGPT (512d)"
    "gene2vec:all_genes_gene2vec.pt:Gene2Vec (200d)"
    "esm2:all_genes_esm.pt:ESM-2 (Pool PaRTI, 1280d)"
    "prot_t5:all_genes_prot_t5.pt:ProtT5-XL (1024d)"
    "esm1b:all_genes_esm1b.pt:ESM-1b (1280d)"
    "esmc:all_genes_esmc.pt:ESM-C 600M (1152d)"
    "scprint:all_genes_scprint.pt:scPRINT"
    "seqvec:all_genes_seqvec.pt:SeqVec (1024d)"
    "genept:all_genes_genept.pt:GenePT (Data Leakage) (1536d)"
    "text_embed:all_genes_text_embed.pt:Text-mxbai (1024d)"
    "bioconceptvec:all_genes_bioconceptvec.pt:BioConceptVec (100d)"
    "node2vec_ppi:all_genes_node2vec_ppi.pt:Node2Vec PPI (128d)"
    "ppi_svd:all_genes_ppi_svd.pt:PPI-SVD (256d)"
    "ppi_raw:all_genes_ppi_raw.pt:PPI-RAW (1024d)"
    "mashup:all_genes_mashup.pt:Mashup (500d)"
    "go2vec:all_genes_go2vec.pt:GO2Vec (128d)"
    "onto2vec:all_genes_onto2vec.pt:Onto2Vec (128d)"
    "kg_complex:all_genes_kg_complex.pt:KG-ComplEx (256d)"
)

# ============================================================================
# Compute-node setup (only inside SLURM jobs)
# ============================================================================
# Skipped when sourced from the login node (submit_pipeline.sh), where
# SLURM_JOB_ID is not set. This avoids module-load and nvidia-smi errors.

if [ -n "$SLURM_JOB_ID" ]; then
    module purge
    module load gnu15 openmpi5 EasyBuild cmake openblas fftw

    cd "$WORK_DIR"

    echo "============================================================================"
    echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID | Node: ${SLURM_NODELIST:-$(hostname)}"
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        echo "Array task: $SLURM_ARRAY_TASK_ID (of job $SLURM_ARRAY_JOB_ID)"
    fi
    echo "Start: $(date)"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true
    echo "============================================================================"
    echo ""
fi
