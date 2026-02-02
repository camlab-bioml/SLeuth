# Siamese Network for Synthetic Lethality Prediction

A PyTorch implementation of a Siamese neural network for predicting Synthetic Lethality (SL) using ESM protein embeddings as the **only** node features.

## Key Features

- **ESM embeddings only**: Uses ESM-2 protein language model embeddings (1280-dim) as sole features
- **Pool PaRTI pooling**: PageRank-based pooling for better protein representations (Tartici et al., 2025)
- **All genes**: Can work with any human gene that has a protein sequence (not limited to SL dataset genes)
- **Siamese architecture**: Shared encoder ensures consistent representations
- **Reproducible**: Fixed random seeds for deterministic training

## Installation

```bash
pip install -r requirements.txt
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
python train.py \
    --embeddings_path ../data/all_genes_esm.pt \
    --sl_path ../data/SL_Human_Approved.txt \
    --output_dir results/siamese_esm \
    --cv_type cv1 \
    --epochs 200 \
    --seed 42
```

### 3. Predict SL for gene pairs

```bash
# Single pair
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings ../data/all_genes_esm.pt \
    --gene1 BRCA1 --gene2 PARP1

# Find top SL partners for a gene
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings ../data/all_genes_esm.pt \
    --gene1 BRCA1 --top_k 100 --output brca1_partners.csv
```

## Architectures

### 1. Siamese (default)
```
Gene1 ESM (1280) ─┬─> SharedEncoder ─> z1 ─┬─> [z1, z2, z1*z2, |z1-z2|] ─> MLP ─> P(SL)
                  │                        │
Gene2 ESM (1280) ─┴─> SharedEncoder ─> z2 ─┘
```

### 2. Kernel (RKHS-based) - recommended for best theoretical grounding
```
Gene1 ESM ─┬─> Encoder ─> z1 ─┬─> RFF(z1), RFF(z2)      ─┬─> Kernel    ─> P(SL)
           │                  ├─> Bilinear(z1, z2)       │   Features
Gene2 ESM ─┴─> Encoder ─> z2 ─┴─> ||z1-z2||², cos(z1,z2) ─┘
```

The kernel model operates in a Reproducing Kernel Hilbert Space (RKHS) using:
- **Random Fourier Features (RFF)**: Approximates Gaussian kernel k(x,y) = exp(-||x-y||²/2σ²)
- **Low-rank bilinear**: Learns Mahalanobis-like metric M = UᵀU with O(rank × dim) parameters
- **Multiple similarities**: Combines RFF, bilinear, Euclidean, and cosine similarities

```bash
python train.py --model_type kernel --rff_features 256 --bilinear_rank 64
```

### 3. Attention (cross-attention between genes)
```
Gene1 ESM ─> Proj ─> CrossAttn(Gene1, Gene2) ─> z1 ─┬─> Predictor ─> P(SL)
Gene2 ESM ─> Proj ─> CrossAttn(Gene2, Gene1) ─> z2 ─┘
```

## Cross-Validation Strategies

Three strategies matching SLMGAE benchmark:

- **CV1 (edge-based)**: Randomly splits SL pairs into folds. Tests ability to predict held-out interactions between known genes.
- **CV2 (gene-based)**: Test pairs have at least one unseen gene. Tests generalization to partially new genes.
- **CV3 (pair-based)**: Both genes in test pairs are unseen during training. Tests generalization to completely novel gene pairs.

## Reproducibility

All random seeds are fixed:
- PyTorch: `torch.manual_seed(seed)`
- NumPy: `np.random.seed(seed)`
- CUDA: `torch.cuda.manual_seed_all(seed)`
- CuDNN: `deterministic=True, benchmark=False`

## Running on Server (SLURM)

### Quick Start
```bash
cd siamese_sl

# Run complete pipeline (ESM generation + all 3 CVs)
./slurm/run_all.sh

# Or skip ESM if already generated
./slurm/run_all.sh --skip-esm
```

### Individual Steps
```bash
# Step 1: Generate ESM embeddings
sbatch slurm/step1_generate_esm.sh

# Step 2: Train on each CV (after ESM is ready)
sbatch slurm/step2_train_cv1.sh
sbatch slurm/step2_train_cv2.sh
sbatch slurm/step2_train_cv3.sh

# Step 3: Generate summary
sbatch slurm/step3_summarize.sh
```

### Monitor Jobs
```bash
squeue -u $USER                      # Check job status
tail -f slurm/logs/step2_CV1_*.out   # Watch training progress
```

### Output
Results are saved to:
- `results/cv1/results.json` - CV1 metrics (AUROC, AUPR, F1)
- `results/cv2/results.json` - CV2 metrics
- `results/cv3/results.json` - CV3 metrics
- `results/summary.txt` - Comparison table

## Files

```
siamese_sl/
├── siamese_esm.py           # Model architecture (Siamese, Kernel, Attention)
├── data_loader.py           # Data loading and CV splits (CV1/CV2/CV3)
├── train.py                 # Training script
├── predict.py               # Prediction script
├── generate_all_genes_esm.py # ESM embedding generation with Pool PaRTI
├── requirements.txt
├── README.md
├── slurm/                   # SLURM job scripts
│   ├── run_all.sh           # Master pipeline script
│   ├── step1_generate_esm.sh
│   ├── step2_train_cv1.sh
│   ├── step2_train_cv2.sh
│   ├── step2_train_cv3.sh
│   ├── step3_summarize.sh
│   └── logs/                # Job output logs
└── results/                 # Training outputs
    ├── cv1/
    ├── cv2/
    └── cv3/
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
