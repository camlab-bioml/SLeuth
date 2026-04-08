# Siamese Network Architecture for Synthetic Lethality Prediction

## Overview

A Siamese network that maps gene embeddings to a shared latent space and scores gene pairs via a **strictly positive definite kernel** implemented as an inner product of learned projections.

---

## Architecture

```
gene1 embedding (d-dim)                    gene2 embedding (d-dim)
       |                                          |
       v                                          v
  +---------------------------------------------------------+
  |             Shared Encoder (same weights)                |
  |                                                         |
  |  Hidden layers (all except last):                       |
  |    Linear(bias=False) -> LayerNorm -> LeakyReLU(0.2)    |
  |                       -> Dropout                        |
  |                                                         |
  |  Projection layer (last, Xavier init):                  |
  |    Linear (no activation, no norm, optional bias)       |
  +---------+----------+-----------------+----------+-------+
            |          |                 |          |
            v          v                 v          v
       z1 (proj)  h1 (hidden)       z2 (proj)  h2 (hidden)
            |          |                 |          |
            +-----+----+                 +-----+---+
                  |                            |
                  v                            v
         logit = tau * (z1^T z2 + eps * h1^T h2) + b
                              |
                              v
                        sigmoid(logit) = P(SL)
```

### Default: `encoder_dims = [256, 128, 64]`

```
input (d) -> Linear(d, 256, bias=False) -> LN(256) -> LeakyReLU(0.2) -> Dropout(0.2)
          -> Linear(256, 128, bias=False) -> LN(128) -> LeakyReLU(0.2) -> Dropout(0.2)
          -> Linear(128, 64)  (Xavier init, bare projection, optional bias)
```

Hidden layers use `bias=False` because LayerNorm already has a learnable shift (beta).

---

## Scoring Function

$$\text{logit} = \tau \cdot (\mathbf{z}_1^\top \mathbf{z}_2 + \varepsilon \cdot \mathbf{h}_1^\top \mathbf{h}_2) + b$$

$$P(\text{SL}) = \sigma(\text{logit})$$

where:
- $\mathbf{z} = \mathbf{W}\mathbf{h}$ — projection of hidden output through the last Linear layer
- $\mathbf{h}$ — output of the second-to-last hidden layer
- $\tau$ — learnable temperature (log-space), initialized to $1/\sqrt{d_z}$ where $d_z$ = projection dim
- $b$ — learnable scoring bias, shifts decision boundary for SL base rate
- $\varepsilon$ — PD regularizer (default 0.001, not learned)

### Kernel Interpretation

The scoring function computes a **bilinear form** in the hidden feature space:

$$\mathbf{z}_1^\top \mathbf{z}_2 + \varepsilon \cdot \mathbf{h}_1^\top \mathbf{h}_2 = \mathbf{h}_1^\top (\mathbf{W}^\top \mathbf{W} + \varepsilon \mathbf{I}) \mathbf{h}_2 = \mathbf{h}_1^\top \mathbf{K} \mathbf{h}_2$$

where $\mathbf{K} = \mathbf{W}^\top \mathbf{W} + \varepsilon \mathbf{I}$.

- $\mathbf{W}^\top \mathbf{W}$ is positive semi-definite (PSD) by construction
- $\varepsilon \mathbf{I}$ ensures **strict** positive definiteness ($\mathbf{K} \succ 0$)
- L1 regularization on $\mathbf{W}$ encourages low-rank $\mathbf{W}^\top \mathbf{W}$, but $\varepsilon \mathbf{I}$ prevents the kernel from becoming degenerate (singular) when L1 zeroes out columns of $\mathbf{W}$

### Why Strictly Positive Definite?

L1 on $\mathbf{W}$ pushes columns toward zero, making $\mathbf{W}^\top \mathbf{W}$ low-rank. A singular kernel means some directions in hidden feature space are completely ignored — the model cannot distinguish gene pairs that differ only in those directions. The $\varepsilon$ term preserves sensitivity in all directions. With $\varepsilon = 0.001$, the contribution is negligible but prevents degeneracy.

---

## Node Feature Creation

Gene embeddings come from various sources (one embedding type per benchmark run):

| Category | Source | Dim | Method |
|----------|--------|-----|--------|
| Expression | Geneformer | auto | Token embeddings from scRNA-seq transformer |
| Expression | scGPT | 512 | Token embeddings from scRNA-seq generative model |
| Expression | Gene2Vec | 200 | Word2Vec on gene co-expression |
| Protein seq | ESM-2 | 1280 | Pool PaRTI on ESM-2 protein representations |
| Protein seq | ProtT5-XL | 1024 | T5-based protein language model |
| Protein seq | ESM-1b | 1280 | ESM-2 predecessor (650M params) |
| Protein seq | scPRINT | auto | ESM2-derived gene identity tokens |
| Protein seq | SeqVec | 1024 | ELMo-style protein embeddings |
| Text | GenePT | 1536 | GPT text embeddings (data leakage risk) |
| Text | Text-mxbai | 1024 | Open-source text embeddings of gene descriptions |
| Text | BioConceptVec | 100 | Word2Vec on PubMed biomedical concepts |
| PPI | Node2Vec PPI | 128 | Node2Vec on STRING PPI network |
| PPI | PPI-SVD | 256 | SVD of STRING PPI adjacency matrix |
| PPI | PPI-RAW | 1024 | High-dim SVD of PPI adjacency |
| PPI | Mashup | 500 | Diffusion kernel spectral decomposition |
| GO | GO (anc2vec) | 200 | Sum-pool GO term embeddings per gene |
| GO | GO2Vec | 128 | Node2Vec on GO DAG, mean-pooled per gene |
| GO | Onto2Vec | 128 | Word2Vec on GO axiom sentences |
| KG | KG-ComplEx | 256 | ComplEx on STRING PPI + GO triples |

**Gene identifiers**: All `.pt` files use NCBI Entrez Gene IDs (strings) as the canonical identifier in `gene_order`. A static reference mapping is at `data/gene_id_mapping.tsv` (derived from HGNC). At runtime, `gene_name_utils.py` downloads the HGNC complete set for symbol/Entrez mapping.

**Missing gene handling**: Genes without embeddings receive NaN vectors in the `.pt` file. At load time, `data_loader.py` replaces NaN with per-dimension Huber M-estimates across all non-missing genes (robust imputation, c=1.345 for 95% normal efficiency). No per-fold standardization is applied.

---

## Parameters

| Component | Shape (default) | L1 Penalized? | Purpose |
|-----------|----------------|---------------|---------|
| Hidden Linear weight 1 | (256, input_dim) | Yes ($\lambda_1$) | Input feature selection |
| Hidden Linear weight 2 | (128, 256) | Yes ($\lambda_2$) | Feature refinement |
| Projection weight $\mathbf{W}$ | (64, 128) | Yes ($\lambda_3$) | Defines kernel $\mathbf{K} = \mathbf{W}^\top\mathbf{W} + \varepsilon\mathbf{I}$ |
| Input bias $b_0$ | (input_dim,) | No | Learnable per-feature offset |
| Hidden LayerNorm weight + bias | (256,)×2, (128,)×2 | No | Per-layer normalization |
| Projection bias | (64,) optional | No | Gene node-degree prior |
| Temperature $\tau$ | scalar | No | Sharpness of sigmoid |
| Scoring bias $b$ | scalar | No | SL base rate |

Note: hidden Linear layers have `bias=False` (redundant with LayerNorm's learnable shift).

---

## Optimization

### AdamW with Cosine Warm Restarts

$$\boldsymbol{\theta}_{t+1} = (1 - \eta_t \lambda_2)\,\boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

Learning rate: cosine annealing with warm restarts ($T_0=50$, $T_\text{mult}=2$).

### Per-Layer Proximal L1

After each AdamW step, apply **per-layer** soft-thresholding to weight matrices only:

$$w_{ij} \leftarrow \text{sign}(w_{ij}) \cdot \max\!\left(|w_{ij}| - \frac{\lambda_k \, \eta_t}{\max(\sqrt{\hat{v}_{ij}},\, 0.1) + \epsilon},\, 0\right)$$

where $\lambda_k$ is the L1 strength for layer $k$. Biases, LayerNorm params, $\tau$, $b$ are **never** penalized.

Recommended schedule: $\lambda_1 > \lambda_2 > \lambda_3$ (strongest on input layer for feature selection, weakest on projection to preserve kernel structure).

### Regularization Summary

| Component | Type | Scope |
|-----------|------|-------|
| AdamW weight decay | L2 (decoupled) | All parameters |
| Proximal L1 ($\lambda_k$) | Per-layer sparse thresholding | Weight matrices only |
| Dropout | Stochastic | Hidden activations |
| LayerNorm | Implicit normalization | Hidden layers |
| $\varepsilon \mathbf{I}$ | PD kernel regularizer | Scoring function |
| Early stopping | Capacity control | Patience × eval_interval epochs |

---

## CLI Flags

```bash
# Architecture
--encoder_dims 256 128 64       # Layer widths (any number of layers)
--no-last-layer-bias            # Remove projection bias (gene degree prior)
--pd_epsilon 0.001              # PD regularizer epsilon (0 to disable)
--dropout 0.2                   # Dropout rate for hidden layers

# Regularization
--l1_lambdas 0.02 0.01 0.002   # Per-layer L1 (must match encoder_dims count)

# Training
--epochs 200                    # Max training epochs
--batch_size 256                # Gene pairs per mini-batch
--learning_rate 0.001           # AdamW learning rate
--weight_decay 1e-5             # AdamW weight decay (decoupled L2)
--warmrestart_T0 50             # CosineAnnealingWarmRestarts initial cycle length (epochs)
--warmrestart_Tmult 2           # CosineAnnealingWarmRestarts cycle length multiplier
--pos_neg_ratio 1.0             # Positive-to-negative sampling ratio
--eval_interval 10              # Evaluate test set every N epochs
--patience 20                   # Early stopping (in eval intervals)
--num_folds 5                   # Cross-validation folds
--seed 42                       # Random seed
--cpu                           # Force CPU (flag, default: off)
```
