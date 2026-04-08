# Siamese Network for Synthetic Lethality Prediction

A PyTorch implementation of a Siamese neural network for predicting Synthetic Lethality (SL). Supports 18+ gene embedding types across 6 categories with a unified pipeline: every modality is imputed, optionally PCA-reduced, and MAD-normalized before concatenation — the same code path whether you pass one file or many.

## Key Features

- **18+ embedding types**: ESM-2, GO, Geneformer, scGPT, BioConceptVec, PPI network, and more
- **Unified embedding pipeline**: Per-modality Huber imputation, optional robust PCA (ROBPCA; Hubert et al., 2005), and MAD normalization, then concatenation. Same code path for one or many modalities; L1-regularized first layer handles feature selection
- **Pool PaRTI pooling**: PageRank-based pooling for ESM protein representations (Tartici et al., 2025)
- **All genes**: Can work with any human gene that has a protein sequence (not limited to SL dataset genes)
- **Siamese architecture**: Shared encoder ensures consistent representations
- **Reproducible**: Fixed random seeds for deterministic training

## Installation

```bash
pip install torch fair-esm biopython requests tqdm networkx scikit-learn pandas robpy
```

## Quick Start

### 1. Generate ESM embeddings for all human genes

```bash
# Downloads UniProt human proteome and generates ESM embeddings with Pool PaRTI
python generate_all_genes_esm.py --output ../data/all_genes_esm.pt --device cuda:0

# Or use mean pooling instead
python generate_all_genes_esm.py --output ../data/all_genes_esm.pt --pooling mean
```

This creates:
- `../data/all_genes_esm.pt` - ESM embeddings (~20k genes, ~100MB)
- `../data/all_genes_esm.genes.txt` - Gene list for reference

**Pool PaRTI** (default) uses PageRank on attention matrices to weight residue importance,
outperforming mean pooling especially for identifying functionally critical regions.

### 2. Train the model

```bash
# Single modality (same pipeline as multi-modal: impute → normalize)
python train.py \
    --embeddings_paths ../data/all_genes_esm.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --output_dir results/siamese_esm \
    --cv_type cv1 \
    --epochs 200 \
    --seed 42

# Multi-modal (each modality: impute → MAD-normalize → concatenate)
python train.py \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --output_dir results/multi_bio_go_ppi \
    --cv_type cv1

# Multi-modal with PCA (each modality: impute → PCA → normalize → concatenate)
python train.py \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --output_dir results/multi_bio_go_ppi_pca \
    --cv_type cv1 --pca_variance 0.8
```

### 3. Predict SL for gene pairs

```bash
# Single pair (accepts gene symbols or Entrez IDs)
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm.pt \
    --gene1 BRCA1 --gene2 PARP1

# Using Entrez IDs directly
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm.pt \
    --gene1 672 --gene2 142

# Find top SL partners for a gene
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm.pt \
    --gene1 BRCA1 --top_k 100 --output brca1_partners.csv
```

## Architectures

### 1. Siamese (inner product scoring)
```
Gene1 emb ─┬─> SharedEncoder ─> z1, h1 ─┐
           │                             ├─> τ·(z1ᵀz2 + ε·h1ᵀh2) + b ─> sigmoid ─> P(SL)
Gene2 emb ─┴─> SharedEncoder ─> z2, h2 ─┘
```
Inner product with PD kernel K = WᵀW + εI. Symmetric by construction.

### 2. Kernel (RKHS-based)
```
Gene1 emb ─┬─> Encoder ─> z1 ─> φ(z1) ∈ H ─┬─> [φ1+φ2, φ1*φ2, |φ1-φ2|] ─> MLP ─> P(SL)
           │                                 │
Gene2 emb ─┴─> Encoder ─> z2 ─> φ(z2) ∈ H ─┘
```

The kernel model operates in a Reproducing Kernel Hilbert Space (RKHS) using:
- **Random Fourier Features (RFF)**: Approximates Gaussian kernel k(x,y) = exp(-||x-y||²/2σ²)
- **Learned linear projection**: Orthogonal basis in Hilbert space (Mahalanobis-like)
- **Symmetric Hilbert features**: element-wise sum, product, and absolute difference of φ(z1) and φ(z2)

```bash
python train.py --model_type kernel --rff_features 256 --bilinear_rank 64
```

### 3. Attention (cross-attention between genes)
```
Gene1 emb ─> Proj ─> CrossAttn(Gene1, Gene2) ─> z1 ─┬─> [z1+z2, z1*z2, |z1-z2|] ─> MLP ─> P(SL)
Gene2 emb ─> Proj ─> CrossAttn(Gene2, Gene1) ─> z2 ─┘
```

## Cross-Validation Strategies

Three strategies matching SLMGAE benchmark:

- **CV1 (edge-based)**: Randomly splits SL pairs into folds. Tests ability to predict held-out interactions between known genes.
- **CV2 (gene-based)**: Test pairs have at least one unseen gene. Tests generalization to partially new genes.
- **CV3 (pair-based)**: Both genes in test pairs are unseen during training. Tests generalization to completely novel gene pairs.

## Reproducibility

All random seeds are fixed:
- Python: `random.seed(seed)`
- NumPy: `np.random.seed(seed)`
- PyTorch: `torch.manual_seed(seed)`
- CUDA: `torch.cuda.manual_seed_all(seed)`
- CuDNN: `deterministic=True, benchmark=False`

## Running on Server (SLURM)

The pipeline has five SLURM jobs: environment reset, embedding generation, benchmark array, benchmark summary, best-per-category array, and best-per-category combo. The two training branches (benchmark and best-per-category) run in parallel after generation completes.

### Step 0: Reset Environment (`reset_env.sh`)

Recreates the Python venv from scratch on a GPU node. Must run on a GPU node (gpu1/gpu2). Run this first if the environment is missing or broken.

```bash
cd siamese_sl
sbatch slurm/reset_env.sh
# Removes old venv, creates fresh one, installs all dependencies
# (torch, fair-esm, robpy, transformers, pykeen, etc.)
# Verifies all packages and pre-caches the HGNC gene name database
```

### Step 1: Generate Embeddings (`run_generate_embeddings.sh`)

Downloads source data and generates all .pt embedding files. Run this first.

```bash
cd siamese_sl
GEN=$(sbatch --parsable slurm/run_generate_embeddings.sh)
# 1. Download SynLethDB experimental SL pairs
# 2. Generate ESM-2 embeddings (downloads UniProt proteome)
# 3. Generate GO embeddings (from GAF file)
# 4. Generate all 17 other embedding types (auto-downloads STRING PPI, GO OBO, etc.)
```

### Step 2a: Embedding Benchmark (`run_embedding_benchmark.sh`)

Benchmarks each embedding individually at native dimensionality (no PCA) across 3 CV types.

```bash
# Chain after generation completes
sbatch --array=0-56%4 --dependency=afterok:$GEN slurm/run_embedding_benchmark.sh
# Trains each embedding × 3 CVs = up to 57 runs, prints comparison table
# Note: submit_pipeline.sh computes the correct --array range automatically;
# these manual commands are shown for reference only.
```

### Step 2b: Best-per-Category Multi-Modal (`run_best_per_category.sh`)

Retrains each embedding with PCA (default 80% variance), picks the best per category, then trains a multi-modal combo. Can run in parallel with Step 2a.

```bash
# Chain after generation completes (runs in parallel with 2a)
sbatch --array=0-53%4 --dependency=afterok:$GEN slurm/run_best_per_category.sh
# Note: submit_pipeline.sh computes the correct --array range automatically;
# these manual commands are shown for reference only.

# Or override PCA variance target
BEST_CAT_PCA_VARIANCE=0.9 sbatch --array=0-53%4 --dependency=afterok:$GEN slurm/run_best_per_category.sh
```

Step 2b sub-steps:
1. Train each embedding with PCA (80% variance) × 3 CVs (apples-to-apples comparison)
2. Select best embedding per category (expression, protein_seq, text, ppi, go, kg) per CV
3. Train multi-modal combo from the winners (each at 80% variance PCA) × 3 CVs
4. Print summary comparing single-embedding vs multi-modal

### Configuration Structure

All scripts source `slurm/config.sh` (shared environment, training hyperparameters, embedding catalog), then their own `.conf` file for job-specific settings. Per-job `.conf` files can override any shared value if independent tuning is needed.

```
config.sh (shared by all jobs + login-node orchestrator)
├── Environment: TRANSFORMERS_NO_TF, module loads (SLURM-only)
├── Python: PYTHON_PATH
├── Directories: WORK_DIR, BASE_DIR, DATA_DIR, CACHE_DIR
├── Data files: ESM_PATH, GO_PATH, GAF_PATH, SL_PATH
├── Training: ENCODER_DIMS, MODEL_ARGS, EPOCHS, BATCH_SIZE, LR,
│             WEIGHT_DECAY, DROPOUT, EVAL_INTERVAL, PATIENCE,
│             NUM_FOLDS, POS_NEG_RATIO, SEED, WARMRESTART_T0/TMULT
├── Regularization: L1_LAMBDAS, PD_EPSILON
├── CV_TYPES[]
├── ALL_EMBEDDINGS[]      — type:filename:display_name
└── Compute-node header: cd, timestamp, GPU info (guarded by SLURM_JOB_ID)

run_generate_embeddings.conf
└── PRECOMPUTED_TYPES[]   — embedding types for generate_embeddings.py

run_embedding_benchmark.conf (overrides only — shared defaults from config.sh)
└── (empty by default; uncomment to override shared training params)

run_best_per_category.conf (PCA + multi-modal combo)
├── PCA_VARIANCE           — override via BEST_CAT_PCA_VARIANCE env var
└── EMB_CATALOG[]         — type:filename:category (genept excluded)
```

### Supported Embeddings

| Category | Type | Dim | Notes |
|----------|------|-----|-------|
| Expression | `geneformer` | 256/1152 | Geneformer V1/V2 token embeddings |
| Expression | `scgpt` | 512 | scGPT token embeddings |
| Expression | `gene2vec` | 200 | Gene2Vec co-expression skip-gram |
| Protein seq | `esm2` | 1280 | ESM-2 + Pool PaRTI |
| Protein seq | `prot_t5` | 1024 | ProtT5-XL |
| Protein seq | `esm1b` | 1280 | ESM-1b |
| Protein seq | `scprint` | auto | scPRINT ESM2-derived gene IDs |
| Protein seq | `seqvec` | 1024 | SeqVec ELMo-style |
| Text | `genept` | 1536 | GenePT (DATA LEAKAGE with SL) |
| Text | `text_embed` | 1024 | mxbai-embed-large on NCBI descriptions |
| Text | `bioconceptvec` | 100 | BioConceptVec PubMed concepts |
| PPI network | `node2vec_ppi` | 128 | Node2Vec on STRING v12.0 |
| PPI network | `ppi_svd` | 256 | Truncated SVD of PPI adjacency |
| PPI network | `ppi_raw` | 1024 | Higher-dim SVD of PPI |
| PPI network | `mashup` | 500 | Diffusion kernel spectral decomposition |
| GO | `go` | 200 | anc2vec GO term embeddings |
| GO | `go2vec` | 128 | Node2Vec on GO DAG |
| GO | `onto2vec` | 128 | Word2Vec on GO axiom sentences |
| KG | `kg_complex` | 256 | ComplEx on STRING PPI + GO triples |

All embedding data is auto-downloaded by `generate_embeddings.py` and cached in `data/embeddings_cache/`. Gene symbols from external sources are normalized (custom corrections + HGNC) and converted to NCBI Entrez Gene IDs. All `.pt` files use Entrez IDs as the canonical gene identifier.

### Monitor Jobs
```bash
squeue -u $USER                                        # Check job status
tail -f slurm/logs/siamese_gen_emb_*.out               # Watch Step 1 (generation)
tail -f slurm/logs/bench_*.out                         # Watch Step 2a (benchmark)
tail -f slurm/logs/bestcat_*.out                       # Watch Step 2b (best-per-category)
```

### Output
Step 2a results:
- `results/<embedding>_<cv>/results.json` — per-embedding per-CV metrics
- `results/embedding_benchmark_summary.json` — comparison table

Step 2b results:
- `results/pcavar0.8_<embedding>_<cv>/results.json` — PCA single-embedding results
- `results/best_per_category.json` — winning embedding per category per CV
- `results/best_cat_<combo>_<cv>/results.json` — multi-modal combo results

## Files

```
siamese_sl/
├── siamese_esm.py              # Model architecture (Siamese, Kernel, Attention)
├── data_loader.py              # Unified embedding pipeline, CV splits
├── train.py                    # Training with cross-validation
├── predict.py                  # Prediction for gene pairs
├── gene_name_utils.py          # Gene ID utilities: symbol ↔ Entrez ID mapping + HGNC normalization
├── generate_all_genes_esm.py   # ESM embedding generation with Pool PaRTI
├── generate_go_esm_embeddings.py # anc2vec GO embeddings
├── generate_embeddings.py      # 17 embedding types (auto-download + generate)
├── gene_embeddings_reference.md # Reference guide for embedding methods
├── ARCHITECTURE.md             # Detailed model architecture + math
├── SETUP.md                    # Server environment setup notes
├── CLAUDE.md                   # Claude Code guidance
├── README.md
├── slurm/                      # SLURM job scripts
│   ├── config.sh               # Shared config (env, modules, paths, data files)
│   ├── reset_env.sh            # Recreate Python venv from scratch
│   ├── run_generate_embeddings.sh   # Step 1: download + generate all .pt files
│   ├── run_generate_embeddings.conf # ↳ embedding type lists
│   ├── run_embedding_benchmark.sh   # Step 2a: benchmark at native dims (no PCA)
│   ├── run_embedding_benchmark.conf # ↳ training hyperparams, embedding list
│   ├── run_best_per_category.sh     # Step 2b: PCA + best-per-category + multi-modal
│   ├── run_best_per_category.conf   # ↳ training hyperparams, PCA dim, catalog
│   └── logs/                   # Job output logs
└── results/                    # Training outputs
```

## Using Your Own Sequences

If you have custom protein sequences, prepare a FASTA file:

```
>GENE1
MSEQVENCE...
>GENE2
MPROTEIN...
```

Then generate embeddings:

```bash
python generate_all_genes_esm.py \
    --fasta your_sequences.fasta \
    --output your_embeddings.pt
```

## GPU Memory

ESM-2 (650M) requires ~4GB GPU memory for inference. If running out of memory:

```bash
# Reduce batch size
python generate_all_genes_esm.py --batch_size 4 --output out.pt

# Or use CPU (slower)
python generate_all_genes_esm.py --device cpu --batch_size 1 --output out.pt
```
