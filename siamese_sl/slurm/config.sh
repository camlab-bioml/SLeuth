#!/bin/bash
# ============================================================================
# Shared configuration for all SLURM scripts
# ============================================================================
# Source this at the top of every job script:
#   source "$SLURM_SUBMIT_DIR/slurm/config.sh"
#
# This file is the single place to edit hyperparameters, paths, embedding
# lists, and best-per-category settings. Both pipeline scripts read from here:
#   - run_embedding_benchmark.sh  (Phase 1: single-embedding benchmark)
#   - run_best_per_category.sh    (Phase 2: PCA + multi-modal combo)
# ============================================================================

# Prevent transformers from importing TensorFlow (numpy binary incompatibility)
export TRANSFORMERS_NO_TF=1

# Load modules
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

# Python environment
# Type: string (absolute path to python3 binary)
# Example: "/home/user/venv/bin/python3"
PYTHON_PATH="/ddn_exa/campbell/kaiyang/pytorch/bin/python3"

# ============================================================================
# Project directories
# ============================================================================
# Derived from SLURM_SUBMIT_DIR, which SLURM sets to the directory where
# `sbatch` was called. All scripts assume you submit from siamese_sl/:
#   cd ~/SLMGAE-pytorch/siamese_sl && sbatch slurm/run_embedding_benchmark.sh

WORK_DIR="$SLURM_SUBMIT_DIR"        # siamese_sl/
BASE_DIR="$(dirname "$WORK_DIR")"    # SLMGAE-pytorch/
DATA_DIR="$BASE_DIR/data"            # Final .pt embedding files
CACHE_DIR="$DATA_DIR/embeddings_cache"  # Downloaded raw data (STRING PPI, GO OBO, model weights, etc.)

# ============================================================================
# Key data files
# ============================================================================
# Type: string (absolute paths, derived from directories above)

ESM_PATH="$DATA_DIR/all_genes_esm.pt"           # ESM-2 embeddings; also used as the canonical gene list for all other types
GO_PATH="$DATA_DIR/all_genes_go.pt"              # GO-only anc2vec embeddings (200d)
GAF_PATH="$BASE_DIR/uniprot_GO/goa_human.gaf.gz" # Gene Ontology annotation file (GAF 2.2)
SL_PATH="$DATA_DIR/SL_Human_Approved.txt"        # Positive SL pairs (gene1 TAB gene2)

# ============================================================================
# Training hyperparameters
# ============================================================================

# --- Architecture ---

# ENCODER_DIMS
# Type: space-separated ints (one per encoder layer)
# Widths of the shared Siamese encoder layers. Each value is the output
# dimension of one Linear layer.
#   - Hidden layers (all except last): Linear -> LayerNorm -> LeakyReLU -> Dropout
#   - Last layer: bare Linear projection (no activation, Xavier init)
# The final layer output is the embedding dimension for inner product scoring.
# Example: "256 128 64" builds:
#   input -> Linear(in,256) -> LN -> LReLU -> Drop
#         -> Linear(256,128) -> LN -> LReLU -> Drop
#         -> Linear(128,64)                        <- bare projection
#         -> z1^T z2 -> sigmoid -> P(SL)
# Example: "512 256 128 64" for a deeper encoder
ENCODER_DIMS="256 128 64"

# MODEL_ARGS
# Type: string (CLI flags passed directly to train.py)
# Model type and architecture flags.
#   --model_type siamese       Inner product scoring (default)
#   --model_type kernel        RKHS with Random Fourier Features
#   --model_type attention     Cross-attention between gene pairs
#   --no-last-layer-bias       Remove bias from last encoder layer;
#                              bias acts as a gene node-degree prior
#                              (some genes are SL with many partners)
# Example: "--model_type kernel --rff_features 256"
MODEL_ARGS="--model_type siamese --no-last-layer-bias"

# --- Training ---

# EPOCHS
# Type: int
# Maximum training epochs per fold. Training may stop earlier via patience.
# Example: 2000
EPOCHS=2000

# BATCH_SIZE
# Type: int
# Number of gene pairs per mini-batch.
# Example: 256
BATCH_SIZE=256

# LR
# Type: float
# AdamW learning rate.
# Example: 0.001
LR=0.001

# WEIGHT_DECAY
# Type: float
# AdamW weight decay (L2 regularization on all parameters).
# Example: 0.00001
WEIGHT_DECAY=0.00001

# DROPOUT
# Type: float (0.0 to 1.0)
# Dropout rate applied in encoder hidden layers.
# Example: 0.2
DROPOUT=0.2

# EVAL_INTERVAL
# Type: int
# Evaluate on the test set every N epochs. Also the granularity for
# early stopping (patience is counted in eval intervals, not epochs).
# Example: 30
EVAL_INTERVAL=30

# PATIENCE
# Type: int
# Early stopping: stop training after this many consecutive eval intervals
# without AUROC improvement. Effective max stall = EVAL_INTERVAL * PATIENCE.
# Example: 10 (with EVAL_INTERVAL=30 -> 300 epochs max stall)
PATIENCE=10

# NUM_FOLDS
# Type: int
# Number of cross-validation folds.
# Example: 5
NUM_FOLDS=5

# POS_NEG_RATIO
# Type: float
# Ratio of negative to positive samples during training. 1.0 = balanced.
# Example: 1.0
POS_NEG_RATIO=1.0

# SEED
# Type: int
# Random seed. Controls model initialization, CV splitting, and negative
# sampling. Same seed = same splits across runs.
# Example: 42
SEED=42

# WARMRESTART_T0
# Type: int
# CosineAnnealingWarmRestarts: initial cycle length in epochs.
# Example: 50
WARMRESTART_T0=50

# WARMRESTART_TMULT
# Type: int
# CosineAnnealingWarmRestarts: cycle length multiplier after each restart.
# Cycle lengths: T0, T0*Tmult, T0*Tmult^2, ...
# Example: 2 (cycles of 50, 100, 200, 400, ...)
WARMRESTART_TMULT=2

# --- Regularization ---

# L1_LAMBDAS
# Type: space-separated floats (one per encoder layer, must match ENCODER_DIMS count)
# Per-layer proximal L1 soft-thresholding strengths. Applied to weight
# matrices only (biases, input bias, and LayerNorm params are never penalized).
# Higher lambda -> more sparsity -> more aggressive feature selection.
# Recommended: strongest on first layer (input feature selection),
#              weakest on last layer (preserve interaction structure).
# Set all to zero to disable: L1_LAMBDAS="0 0 0"
# Example: "0.02 0.01 0.002" for 3-layer encoder
L1_LAMBDAS="0.02 0.01 0.002"

# PD_EPSILON
# Type: float (>= 0)
# Positive-definiteness regularizer. Ensures the scoring kernel
# K = W^T W + eps*I remains strictly positive definite when L1 pushes
# columns of W toward zero.
# Implemented as: score = z1^T z2 + eps * h1^T h2 (h = hidden layer output).
# Set to 0 to disable.
# Example: 0.001
PD_EPSILON=0.001

# --- Cross-validation types ---

# CV_TYPES
# Type: bash array of strings ("cv1", "cv2", "cv3")
# Which CV strategies to run. Each entry trains NUM_FOLDS folds.
#   cv1: Edge split   -- random split of SL pairs (easiest, interpolation)
#   cv2: Gene split   -- hold out genes, test has >=1 unseen gene (semi-inductive)
#   cv3: Pair split   -- both genes in test are unseen (hardest, fully inductive)
# Example: ("cv1" "cv2" "cv3") for all three, or ("cv2") for gene-split only
CV_TYPES=("cv1" "cv2" "cv3")

# ============================================================================
# Embedding types (used by run_embedding_benchmark.sh)
# ============================================================================

# PRECOMPUTED_TYPES
# Type: bash array of strings
# Embedding types to generate via generate_embeddings.py in Phase 1.
# Each type auto-downloads its source data into CACHE_DIR on first run.
# ESM-2 and GO are generated separately (dedicated steps in run_embedding_benchmark.sh).
# Example: to benchmark only PPI embeddings, set to ("node2vec_ppi" "ppi_svd" "mashup")
PRECOMPUTED_TYPES=(
    # Expression-based (single-cell foundation models)
    "geneformer"      # Geneformer token embeddings (V1: 256d, V2: 1152d)
    "scgpt"           # scGPT token embeddings (512d)
    "gene2vec"        # Gene2Vec co-expression skip-gram (200d)
    # Protein sequence (PLMs -- GPU-heavy, hours for ~20k proteins)
    "prot_t5"         # ProtT5-XL T5-based PLM (1024d)
    "esm1b"           # ESM-1b predecessor to ESM-2 (1280d)
    "scprint"         # scPRINT ESM2-derived gene IDs (auto dim, needs manual checkpoint)
    "seqvec"          # SeqVec ELMo-style (1024d, needs allennlp)
    # Text / literature
    "genept"          # GenePT GPT-3.5 text embeddings (1536d, DATA LEAKAGE with SL)
    "text_embed"      # mxbai-embed-large on NCBI gene descriptions (1024d, no leakage)
    "bioconceptvec"   # BioConceptVec PubMed concept embeddings (100d)
    # PPI network (STRING v12.0, auto-downloaded)
    "node2vec_ppi"    # Node2Vec random walks on PPI (128d)
    "ppi_svd"         # Truncated SVD of PPI adjacency (256d)
    "ppi_raw"         # Higher-dim SVD of PPI adjacency (1024d, strong baseline)
    "mashup"          # Diffusion kernel via spectral decomposition (500d)
    # Gene Ontology (GO OBO + human GAF, auto-downloaded)
    "go2vec"          # Node2Vec on GO DAG, mean-pooled per gene (128d)
    "onto2vec"        # Word2Vec on GO axiom sentences, mean-pooled (128d)
    # Knowledge graph
    "kg_complex"      # ComplEx on STRING PPI + GO annotation triples (256d, needs pykeen)
)

# ALL_EMBEDDINGS
# Type: bash array of colon-separated strings "type:filename:display_name"
# Complete list of embeddings for Phase 1 benchmarking (includes ESM-2 and GO
# which are generated by dedicated steps, not by generate_embeddings.py).
# Used by run_embedding_benchmark.sh to detect available embeddings and train.
# Example entry: "go:all_genes_go.pt:GO (anc2vec, 200d)"
ALL_EMBEDDINGS=(
    # GO-based
    "go:all_genes_go.pt:GO (anc2vec, 200d)"
    # Expression-based
    "geneformer:all_genes_geneformer.pt:Geneformer"
    "scgpt:all_genes_scgpt.pt:scGPT (512d)"
    "gene2vec:all_genes_gene2vec.pt:Gene2Vec (200d)"
    # Protein sequence
    "esm2:all_genes_esm.pt:ESM-2 (Pool PaRTI, 1280d)"
    "prot_t5:all_genes_prot_t5.pt:ProtT5-XL (1024d)"
    "esm1b:all_genes_esm1b.pt:ESM-1b (1280d)"
    "scprint:all_genes_scprint.pt:scPRINT"
    "seqvec:all_genes_seqvec.pt:SeqVec (1024d)"
    # Text / literature
    "genept:all_genes_genept.pt:GenePT (Data Leakage) (1536d)"
    "text_embed:all_genes_text_embed.pt:Text-mxbai (1024d)"
    "bioconceptvec:all_genes_bioconceptvec.pt:BioConceptVec (100d)"
    # PPI network
    "node2vec_ppi:all_genes_node2vec_ppi.pt:Node2Vec PPI (128d)"
    "ppi_svd:all_genes_ppi_svd.pt:PPI-SVD (256d)"
    "ppi_raw:all_genes_ppi_raw.pt:PPI-RAW (1024d)"
    "mashup:all_genes_mashup.pt:Mashup (500d)"
    # GO-based
    "go2vec:all_genes_go2vec.pt:GO2Vec (128d)"
    "onto2vec:all_genes_onto2vec.pt:Onto2Vec (128d)"
    # Knowledge graph
    "kg_complex:all_genes_kg_complex.pt:KG-ComplEx (256d)"
)

# ============================================================================
# Best-per-category settings (used by run_best_per_category.sh)
# ============================================================================

# PCA_DIM
# Type: int
# Number of principal components each modality is projected to before
# comparison and multi-modal concatenation. Using the same PCA dim for all
# modalities ensures apples-to-apples comparison. With N winning categories,
# the multi-modal input is N * PCA_DIM dimensions (e.g., 5 * 64 = 320d).
# Override at submit time: BEST_CAT_PCA_DIM=128 sbatch slurm/run_best_per_category.sh
# Example: 64
PCA_DIM="${BEST_CAT_PCA_DIM:-64}"

# EMB_CATALOG
# Type: bash array of colon-separated strings "type:filename:category"
# Maps each embedding type to its .pt file (relative to DATA_DIR) and its
# biological category. Used in Phase 2 to:
#   1. Train each embedding individually with PCA
#   2. Pick the best AUROC embedding per category per CV
#   3. Concatenate the winners into a multi-modal combo
# Categories: expression, protein_seq, text, ppi, go, kg
# genept is excluded due to data leakage with SL labels.
# Example entry: "geneformer:all_genes_geneformer.pt:expression"
EMB_CATALOG=(
    # Expression
    "geneformer:all_genes_geneformer.pt:expression"
    "scgpt:all_genes_scgpt.pt:expression"
    "gene2vec:all_genes_gene2vec.pt:expression"
    # Protein sequence
    "esm2:all_genes_esm.pt:protein_seq"
    "esm1b:all_genes_esm1b.pt:protein_seq"
    "prot_t5:all_genes_prot_t5.pt:protein_seq"
    "scprint:all_genes_scprint.pt:protein_seq"
    "seqvec:all_genes_seqvec.pt:protein_seq"
    # Text / literature (genept excluded -- data leakage)
    "text_embed:all_genes_text_embed.pt:text"
    "bioconceptvec:all_genes_bioconceptvec.pt:text"
    # PPI network
    "node2vec_ppi:all_genes_node2vec_ppi.pt:ppi"
    "ppi_svd:all_genes_ppi_svd.pt:ppi"
    "ppi_raw:all_genes_ppi_raw.pt:ppi"
    "mashup:all_genes_mashup.pt:ppi"
    # Gene Ontology
    "go:all_genes_go.pt:go"
    "go2vec:all_genes_go2vec.pt:go"
    "onto2vec:all_genes_onto2vec.pt:go"
    # Knowledge graph
    "kg_complex:all_genes_kg_complex.pt:kg"
)

# ============================================================================
# cd to working directory and print header
# ============================================================================

cd "$WORK_DIR"

echo "============================================================================"
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true
echo "============================================================================"
echo ""
