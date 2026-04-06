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
    --sl_path ../data/SL_Human_Approved.txt \
    --output_dir results/siamese_esm \
    --cv_type cv1 \
    --epochs 200 \
    --seed 42

# Multi-modal (each modality: impute → MAD-normalize → concatenate)
python train.py \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_Human_Approved.txt \
    --output_dir results/multi_bio_go_ppi \
    --cv_type cv1

# Multi-modal with PCA (each modality: impute → PCA → normalize → concatenate)
python train.py \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_Human_Approved.txt \
    --output_dir results/multi_bio_go_ppi_pca64 \
    --cv_type cv1 --pca_dims 64 64 64
```

### 3. Predict SL for gene pairs

```bash
# Single pair
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm.pt \
    --gene1 BRCA1 --gene2 PARP1

# Multi-modal (must match training modalities)
python predict.py \
    --model results/multi_bio_go_ppi/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --gene1 BRCA1 --gene2 PARP1

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

The pipeline has two phases, each a single SLURM job:

### Phase 1: Embedding Benchmark (`run_embedding_benchmark.sh`)

Generates all embeddings (auto-downloads data) and benchmarks each one individually at native dimensionality across 3 CV types.

```bash
cd siamese_sl
sbatch slurm/run_embedding_benchmark.sh   # Job does everything:
# 1. Generate ESM-2 embeddings (downloads UniProt proteome)
# 2. Generate GO embeddings (from GAF file)
# 3. Generate all 17 other embedding types (auto-downloads STRING PPI, GO OBO, etc.)
# 4. Train each embedding × 3 CVs = up to 57 runs
# 5. Print comparison table
```

### Phase 2: Best-per-Category Multi-Modal (`run_best_per_category.sh`)

Retrains each embedding with PCA (default 64 dims), picks the best per category, then trains a multi-modal combo.

```bash
# Chain after Phase 1 completes
sbatch --dependency=afterok:<PHASE1_JOB_ID> slurm/run_best_per_category.sh

# Or override PCA dimensionality
BEST_CAT_PCA_DIM=128 sbatch --dependency=afterok:<PHASE1_JOB_ID> slurm/run_best_per_category.sh
```

Phase 2 steps:
1. Train each embedding with PCA=64 × 3 CVs (apples-to-apples comparison)
2. Select best embedding per category (expression, protein_seq, text, ppi, go, kg) per CV
3. Train multi-modal combo from the winners (each at PCA=64) × 3 CVs
4. Print summary comparing single-embedding vs multi-modal

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

All embedding data is auto-downloaded by `generate_embeddings.py` and cached in `data/embeddings_cache/`. Gene names are normalized (custom corrections + HGNC) and aligned to a canonical gene order from the ESM embeddings.

### Monitor Jobs
```bash
squeue -u $USER                              # Check job status
tail -f siamese_emb_bench_*.out              # Watch Phase 1
tail -f slurm/logs/best_per_category_*.out   # Watch Phase 2
```

### Output
Phase 1 results:
- `results/<embedding>_<cv>/results.json` — per-embedding per-CV metrics
- `results/embedding_benchmark_summary.json` — comparison table

Phase 2 results:
- `results/pca64_<embedding>_<cv>/results.json` — PCA single-embedding results
- `results/best_per_category.json` — winning embedding per category per CV
- `results/best_cat_<combo>_<cv>/results.json` — multi-modal combo results

## Files

```
siamese_sl/
├── siamese_esm.py              # Model architecture (Siamese, Kernel, Attention)
├── data_loader.py              # Unified embedding pipeline, CV splits
├── train.py                    # Training with cross-validation
├── predict.py                  # Prediction for gene pairs
├── gene_name_utils.py          # Gene name normalization (custom corrections + HGNC)
├── generate_all_genes_esm.py   # ESM embedding generation with Pool PaRTI
├── generate_go_esm_embeddings.py # anc2vec GO embeddings
├── generate_embeddings.py      # 17 embedding types (auto-download + generate)
├── gene_embeddings_reference.md # Reference guide for embedding methods
├── ARCHITECTURE.md             # Detailed model architecture + math
├── SETUP.md                    # Server environment setup notes
├── CLAUDE.md                   # Claude Code guidance
├── README.md
├── slurm/                      # SLURM job scripts
│   ├── config.sh               # Shared config (hyperparams, embedding lists, combos)
│   ├── reset_env.sh            # Recreate Python venv from scratch
│   ├── run_embedding_benchmark.sh  # Phase 1: generate all embeddings + benchmark
│   ├── run_best_per_category.sh    # Phase 2: PCA retraining + multi-modal combo
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
