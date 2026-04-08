# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Siamese neural network for predicting Synthetic Lethality (SL) gene pairs. A shared encoder maps gene embeddings to a latent space; pairs are scored via inner product (or kernel/attention variants). Supports 18+ embedding types across 6 categories and multi-modal concatenation.

## Commands

```bash
# Train single embedding (from siamese_sl/)
python train.py --embeddings_paths ../data/all_genes_go.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --output_dir results/siamese_go --cv_type cv1

# Train multi-modal (concatenated embeddings)
python train.py --embeddings_paths ../data/all_genes_bioconceptvec.pt \
    ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --output_dir results/multi_bio_go_ppi_cv1 --cv_type cv1

# Predict
python predict.py --model results/siamese_go/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_go.pt --gene1 BRCA1 --gene2 PARP1

# SLURM: Full pipeline (generate → benchmark + best-per-category → summaries)
cd siamese_sl && ./slurm/submit_pipeline.sh

# SLURM: Skip generation (embeddings already exist)
./slurm/submit_pipeline.sh --skip-generate

# SLURM: Chain after an existing generation job
./slurm/submit_pipeline.sh --after <JOB_ID>

# SLURM: Only benchmark or only best-per-category
./slurm/submit_pipeline.sh --benchmark-only
./slurm/submit_pipeline.sh --best-cat-only

# SLURM: Limit concurrent GPU tasks (default 4)
MAX_CONCURRENT=8 ./slurm/submit_pipeline.sh
```

## Architecture

**Data pipeline** (`data_loader.py`):
- `load_single_embedding()` — loads .pt, normalizes gene names (custom corrections + HGNC), deduplicates post-normalization collisions
- `load_multimodal_embeddings()` — per-modality: impute (Huber M-estimates) → robust PCA (optional) → normalize (center global median, scale global MAD) → concatenate
- `SLDataManager` — takes `embeddings_paths: List[str]` (unified pipeline: single and multi-modal go through the same path). Loads embeddings, creates CV splits via SLMGAE benchmark's `SLDataSplitter`.
- `--embeddings_paths` (always used): one or more .pt files. Each modality goes through: impute → PCA (optional) → normalize → concatenate. Old `--embeddings_path` (singular) and `--embeddings` are deprecated aliases that convert to a single-element list.
- `--pca_variance V` (optional, recommended): target fraction of variance to explain per modality (e.g., `0.8` for 80%). Each modality gets as many components as needed to reach the target. Uses robpy's ROBPCA (Hubert, Rousseeuw & Vanden Branden, 2005): projection pursuit + MCD on scores for gold-standard robust PCA. If ROBPCA fails (near-singular covariance), retries with more conservative alpha (0.85, 0.90, 0.95) and fewer components before falling back. Falls back to plain PCA (median centering + SVD) if robpy unavailable — Huber M-estimator is used for imputation only, never inside PCA. Rank-deficient embedding dimensions are removed via SVD pre-conditioning before ROBPCA.
- `--pca_dims N1 N2 ...` (optional, alternative): per-modality exact component counts. Mutually exclusive with `--pca_variance`.

**Models** (`siamese_esm.py`):
- `SiameseSL` — main model: shared `SiameseEncoder` → inner product + sigmoid. Encoder: Linear→LayerNorm→LeakyReLU→Dropout per hidden layer, bare Linear projection at end. Configurable via `--encoder_dims`.
- `SiameseSLKernel` — RKHS variant with Random Fourier Features
- `SiameseSLWithAttention` — cross-attention variant

**Regularization**:
- Learnable per-feature input bias on the encoder. Not L1-penalized.
- Proximal L1 soft-thresholding per weight matrix (`--l1_lambdas`), applied outside autograd after each optimizer step. Biases, input bias, and LayerNorm params are never penalized.
- PD epsilon (`--pd_epsilon`): ensures kernel K = W^T W + εI is strictly positive definite when L1 pushes columns to zero.
- `--no-last-layer-bias` removes the projection bias (gene node-degree prior).

**Cross-validation** (from SLMGAE benchmark code in `../code/data_split.py`):
- CV1: edge split (random pairs, easiest)
- CV2: gene split (≥1 unseen gene per test pair)
- CV3: pair split (both genes unseen, hardest)

**Gene identifiers** (`gene_name_utils.py`):
- **NCBI Entrez Gene IDs** (strings, e.g., "672" for BRCA1) are the canonical internal identifier throughout the project.
- Gene symbols are used only at I/O boundaries (CLI input, external data files keyed by symbol).
- Symbol normalization pipeline: custom corrections (`gene_corrections_config.py`, 86 mappings, e.g., date-mangled names like MAR-02→MARCHF2, SEP-01→SEPT1, plus renamed ORFs like C17orf53→HROB) → HGNC current symbol → HGNC alias → uppercased original → map to Entrez ID.
- Static reference: `data/gene_id_mapping.tsv` (44k genes, entrez_id TAB symbol). At runtime, `gene_name_utils.py` downloads the HGNC complete set.
- Key methods: `symbol_to_entrez()`, `entrez_to_symbol()`, `symbols_to_entrez_list()`, `symbols_to_entrez_dict()`.
- Applied at embedding load time; all .pt files store `gene_order` as Entrez ID strings.

## SLURM Pipeline

Training is parallelized via **SLURM array jobs**. Each (embedding, CV)
combination runs as a separate array task on its own GPU. Array sizes are
computed dynamically from the `.conf` files — adding/removing embeddings
requires no script changes.

```
submit_pipeline.sh  (login node — orchestrates everything)
    │
    ├── run_generate_embeddings.sh          (single job)
    │
    ├── run_embedding_benchmark.sh          (array: ALL_EMBEDDINGS × CV_TYPES)
    │   └── run_embedding_benchmark_summary.sh  (afterany)
    │
    └── run_best_per_category.sh            (array: EMB_CATALOG × CV_TYPES)
        └── run_best_per_category_combo.sh  (afterany — selects + trains combos)
```

See `slurm/README.md` for full file reference and customization guide.

## Configuration

All SLURM scripts source `slurm/config.sh` for shared environment, then their
own `.conf` for job-specific settings. Training hyperparameters live in
config.sh (shared); per-job `.conf` files can override if independent tuning
is needed.

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
└── EMB_CATALOG[]          — type:filename:category (genept excluded)
```

## Key Design Decisions

- `--input_dim` defaults to `None` (auto-detected from loaded embeddings). Only pass explicitly to override.
- Normalization preserves within-modality feature scale differences (no per-column z-scoring). Global MAD equalizes cross-modality magnitude. L1 on the first encoder layer handles feature selection.
- Per-modality pipeline order: impute → PCA (optional) → normalize. PCA sees natural variance on complete data; normalization equalizes across modalities after reduction.
- Expression embeddings (geneformer, scgpt, gene2vec) overfit on CV3; functional/network embeddings (bioconceptvec, GO, PPI) generalize better.
