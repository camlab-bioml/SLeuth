# Siamese Network Architecture for Synthetic Lethality Prediction

## Overview

A Siamese network that maps gene embeddings to a shared latent space and scores gene pairs via a **strictly positive definite kernel** implemented as an inner product of learned projections.

---

## Architecture

This section describes the single-output `SiameseSL`. The cell-line model
`SiameseSLMultiCell` (below) shares the encoder and kernel, but its `train.py`
defaults differ: no LayerNorm, a raw (sign-free) $\tau$, a per-head projection
bias and `auprg_any` selection unless the published flags are passed (see
"train.py defaults vs the published model").

```
gene1 embedding (d-dim)                    gene2 embedding (d-dim)
       |                                          |
       v                                          v
  +---------------------------------------------------------+
  |             Shared Encoder (same weights)                |
  |                                                         |
  |  Hidden layers (all except last), each block F(x):      |
  |    Linear(bias=False) -> LayerNorm -> LeakyReLU(0.2)    |
  |                       -> Dropout                        |
  |    wrapped as y = x + F(x) when in/out widths match     |
  |    (default --siamese_encoder_type residual; mlp: none) |
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

### Example: `encoder_dims = [256, 128, 64]` (the `SiameseSL` constructor default)

`train.py` never reaches this default: without `--encoder_dims` it builds
`[--hidden_dim 512, --latent_dim 256]`, i.e. `Linear(d, 512) -> Linear(512, 256)`
(two weight matrices, so two `--l1_lambdas`), and the SLURM tree passes
`--encoder_dims 32 32 32 32 32 32 32` from `slurm/config.conf` (seven; width 16
until 2026-10-05, which is also the published model's width). Under
the default `--siamese_encoder_type residual` that stack is an entry lift
`Linear(D, 32)` with no skip, 5 residual blocks $y = x + F(x)$ at width 32, and
the bare projection. So $\mathbf{h}$ in the $\varepsilon\,\mathbf{h}_1^\top\mathbf{h}_2$
term is the residual sum, and the $\tau = 1/\sqrt{d_z}$ init below is not a
unit-scale calibration for it (see `SiameseSL` in `siamese_esm.py`). The
three-layer stack below is the running example for the rest of this page. It
narrows at every layer, so it has no skips under either encoder type.

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
- $\mathbf{h}$ — output of the last hidden block (the projection's input; with residual skips, the residual sum)
- $\tau$ — learnable temperature (log-space), initialized to $1/\sqrt{d_z}$ where $d_z$ = projection dim
- $b$ — learnable scoring bias, shifts decision boundary for SL base rate
- $\varepsilon$ — PD regularizer (default 0.001, not learned)

### Kernel Interpretation

The scoring function computes a **bilinear form** in the hidden feature space:

$$\mathbf{z}_1^\top \mathbf{z}_2 + \varepsilon \cdot \mathbf{h}_1^\top \mathbf{h}_2 = \mathbf{h}_1^\top (\mathbf{W}^\top \mathbf{W} + \varepsilon \mathbf{I}) \mathbf{h}_2 = \mathbf{h}_1^\top \mathbf{K} \mathbf{h}_2$$

where $\mathbf{K} = \mathbf{W}^\top \mathbf{W} + \varepsilon \mathbf{I}$. The identity holds only for a **bias-free projection**: with $\mathbf{z} = \mathbf{W}\mathbf{h} + \mathbf{b}$ the score gains $\mathbf{b}^\top \mathbf{W}(\mathbf{h}_1 + \mathbf{h}_2) + \lVert \mathbf{b} \rVert^2$ (the gene node-degree prior of `SiameseSL`'s docstring) and is no longer bilinear in $\mathbf{h}$. `MODEL_ARGS` passes `--no-last-layer-bias`, but `SiameseSLMultiCell`'s per-head $\mathbf{c}_h$ is on by default and reinstates it — see the defaults table below.

- $\mathbf{W}^\top \mathbf{W}$ is positive semi-definite (PSD) by construction
- $\varepsilon \mathbf{I}$ ensures **strict** positive definiteness ($\mathbf{K} \succ 0$)
- L1 regularization on $\mathbf{W}$ encourages low-rank $\mathbf{W}^\top \mathbf{W}$, but $\varepsilon \mathbf{I}$ prevents the kernel from becoming degenerate (singular) when L1 zeroes out columns of $\mathbf{W}$

### Why Strictly Positive Definite?

L1 on $\mathbf{W}$ pushes columns toward zero, making $\mathbf{W}^\top \mathbf{W}$ low-rank. A singular kernel means some directions in hidden feature space are completely ignored — the model cannot distinguish gene pairs that differ only in those directions. The $\varepsilon$ term preserves sensitivity in all directions. With $\varepsilon = 0.001$, the contribution is negligible but prevents degeneracy.

---

## Cell-line conditioning (optional, `--use_cell_lines`)

`SiameseSLMultiCell` is a separate model for **masked multi-label, concat-at-input** prediction: instead of one SL score per pair, it predicts SL for every cell-line **head** at once. Heads (`cell_line_vocab.HEADS`, 8): `K562, JURKAT, A549, HELA, A375, 293T, PC9` + an `OTHER` catch-all (absorbs rare lines and pairs with no recorded line).

- **Concat-at-input.** A per-head embedding $\mathbf{c}_h \in \mathbb{R}^{d_c}$ is concatenated onto each gene's feature vector *before* the shared encoder, so the cell context interacts with the biological embedding inside the network. The encoder input widens from $D$ to $D + d_c$ ($d_c$ = `--cell_line_dim`, default 8).
- **Per-head score.** For head $h$: $\text{logit}_h = \tau_h\,(k_h(\mathbf{x}_1,\mathbf{x}_2) + \varepsilon\,\mathbf{h}_1^{(h)\top}\mathbf{h}_2^{(h)}) + b_h$, where $\mathbf{h}^{(h)}$, $\mathbf{z}^{(h)}$ are the hidden output and projection of $[\mathbf{x}\,\Vert\,\mathbf{c}_h]$ and $b_h$ is a per-head bias (cell-line base rate). The kernel term is $k_h = \mathbf{z}_1^{(h)\top}\mathbf{z}_2^{(h)}$ under `--kernel pd` (the default; the PD kernel above, per head) or the symmetrised $\mathbf{z}_1^{(h)\top}\mathbf{A}_{\text{sym}}\mathbf{z}_2^{(h)}$, $\mathbf{A}_{\text{sym}} = (\mathbf{A}+\mathbf{A}^\top)/2$ with $\mathbf{A}$ learnable and initialised to $\mathbf{I}$, under `--kernel bilinear` (not necessarily PD). Symmetric in $(\mathbf{x}_1,\mathbf{x}_2)$ either way. `forward → (B, 8)`.
- **Extra score terms (2026-10-05, on by default; `--no_gene_term` / `--no_pair_features` remove them).** Two terms are added to $\text{logit}_h$ after the temperature, next to $b_h$: `--gene_term`, a per-gene propensity $g(\mathbf{h}_1^{(h)}) + g(\mathbf{h}_2^{(h)})$ with $g(\mathbf{h}) = \mathbf{w}_g^\top\mathbf{h}$ (`gene_w`), and `--pair_features`, $p(\mathbf{z}_1,\mathbf{z}_2) = \mathbf{v}^\top\,\text{LeakyReLU}(\mathbf{U}\,[\,|\mathbf{z}_1-\mathbf{z}_2|\,\Vert\,\mathbf{z}_1\odot\mathbf{z}_2\,] + \mathbf{u})$ on the head's projections (`pair_fc1`, `pair_fc2`; hidden width = latent dim). Both are symmetric and exactly 0 at initialisation (`gene_w` and `pair_fc2` start at zero). The motivation: the trained kernel was measured to be close to a product of per-gene norms, so the gene term gives that propensity its own channel and the pair term adds capacity that is specific to the pair. The pair term does not factorise over the two genes, so `retrieval_metrics.build_any_score_matrix` evaluates it pair by pair in row blocks (a slower export). They apply to every variant; to drop them for a SLURM run set `ARCH_ARGS="--no_gene_term --no_pair_features"`. The published model has neither.
- **train.py defaults vs the published model (defaults changed 2026-09-21).** The model's options, and what `train.py` does by default:

  | Option | train.py default | Published SLeuth model |
  |---|---|---|
  | LayerNorm in each encoder block | **off** (Linear → LeakyReLU → Dropout) | on (`--layernorm`) |
  | Temperature | $\tau$ learned **directly**, any sign | $\tau = \exp(\log\tau) > 0$ (`--log_temperature`) |
  | Temperature sharing | one shared $\tau$ | one shared $\tau$ (`--per_head_temperature` gives $\tau_h$) |
  | Projection bias | **per-head** $\mathbf{z}^{(h)} = \mathbf{W}\mathbf{h} + \mathbf{c}'_h$ | none (`--no_per_head_last_layer_bias` + `--no-last-layer-bias`) |
  | Input offset | shared encoder `input_bias` | same (`--per_head_input_bias` gives one per head) |
  | Kernel standardisation | none | none (`--inner_norm batch`: $(s_h-\mu_h)/\sigma_h$, where $s_h = k_h + \varepsilon\,\mathbf{h}_1^{(h)\top}\mathbf{h}_2^{(h)}$ is the whole bracket and the standardised value replaces it in the logit, over the batch in training, running estimates at eval and for a batch of one pair) |
  | Checkpoint selection | **`auprg_any`** | stratified `auprg` (`--select_metric auprg`) |

  The published model is therefore `--layernorm --log_temperature --no_per_head_last_layer_bias --select_metric auprg --no_gene_term --no_pair_features --no-last-layer-bias`, at encoder width 16 on the post-concat PCA at 0.8, i.e. the SLURM pipeline's PCA arm `joint` (`PCA_ARMS=joint`, `PCA_JOINT_VARIANCE=0.8`); the SLURM tree passes the first six through `ARCH_ARGS` (empty by default) and the last through `MODEL_ARGS` (full recipe: `ARCHITECTURE_CHANGES.md`).
- **Pooled variant (`--cell_lines pooled`).** One head trained on the OR label (SL in any line), no cell-line embedding ($d_c = 0$, `--cell_line_dim` ignored): `forward → (B, 1)`.
- **Masked objective.** Each pair carries `(label, mask)` over the 8 heads. A head asserted SL *and* non-SL resolves by **OR** to label 1, supervised (`cell_line_vocab.pair_label_mask(..., same_head="positive")`, the default since 2026-08-10; `same_head="mask"` restores the older drop-it rule). On the shipped data that recovers 1,877 pairs / 2,172 head-cells — JURKAT 1,461, K562 406, HELA 126, OTHER 107, 293T 36, A549 30, A375 6 — so JURKAT's positives go 1,156 → 2,617 and should be read as largely recovered conflicts. Loss $= \sum (\text{mask}\cdot \text{BCE}) / \sum \text{mask}$. **No `pos_weight` anywhere, by design** — the masked mean is the only weighting. Only supervised cells contribute.
- **Metric.** Early stopping and checkpoint selection use the VALIDATION split and `--select_metric`: **`auprg_any`** by default (AUPRG of the pair-level OR label with the max-over-heads score — the outcome every benchmark model is selected and graded on) or **`auprg`**, the stratified AUPRG pooled over every observed (pair, cell-line) cell (the published rule). `train.py`: `sel_metric = self.args.select_metric` in every mode; single-output mode uses `--select_metric` too, and with one output column (single-output or `--cell_lines pooled`) the two choices are the same number. Both `val_auprg` and `val_auprg_any` are recorded at the selected checkpoint (plus `val_<select_metric>`, `val_auprg_any` by default). The value is computed on the forward logits, not on the exported matrix after the grader's float32 shift that SLMGAE / SLGNN / NSF4SL select on: the export is a blocked GEMM (~1e-6 apart in the logit) shifted twice, which can merge near-ties, so the val AUPRG differs by a few 1e-6 at most (see the comment at `sel_metric` in `train.py`). Macro per-head is reported alongside as a diagnostic. Category winners (`run_best_per_category_combo.sh`) and the `sl_comparison` siamese rows (`collect_siamese.py`) are ranked on `val_<select_metric>_mean` from each run's own config (`auprg` for runs predating the flag), falling back to `val_auprg_mean`.
- The cell embedding (`cell_emb`), the bilinear `kernel_A`, the per-head offsets (`head_input_bias`, `head_proj_bias`) and `gene_w` / `pair_fc1` / `pair_fc2` are **not** L1-penalized, so `--l1_lambdas` still maps one-to-one onto the encoder's weight matrices.
- `predict.py` refuses every multi-head checkpoint, per-head or pooled (single-pair CLI; use `eval_finetuning.load_multicell_model`). `eval_finetuning.py --per_head` **does** consume them, scoring each external screen against its named head — or, for a pooled checkpoint, its single head, recorded as `ANY (pooled)` (`config_used.json` `effective_head`); `PER_HEAD` is auto-enabled in `eval_finetuning.conf` when `USE_CELL_LINES=1`. Fine-tuning keeps `cell_emb`, $\tau$ and $b_h$ frozen, and an `--inner_norm batch` model keeps its running statistics; `gene_w` and the pair-term MLP are trained in Full mode only. `export_pair_predictions.py` (since 2026-10-07) scores every gene pair with every head: $\sigma(\text{logit}_h)$ averaged over the fold models, one matrix per cell line, plus `ANY` (the largest head probability within a fold: the score the benchmark ranks by, not a head trained on the any-line label), and with the models `eval_finetuning.py` fine-tuned on a screen, the predicted screen value of every gene pair through that screen's head (K562 for Adamson).

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
| Protein seq | ESM-C | 1152 | ESM Cambrian 600M via Synthyra/ESMplusplus_large, mean-pooled over residues |
| Protein seq | scPRINT | auto | ESM2-derived gene identity tokens |
| Protein seq | SeqVec | 1024 | ELMo-style protein embeddings |
| Text | GenePT | 1536 | GPT text embeddings (data leakage risk) |
| Text | Text-mxbai | 1024 | Open-source text embeddings of gene descriptions |
| Text | BioConceptVec | 100 | Word2Vec on PubMed biomedical concepts |
| PPI | Node2Vec PPI | 128 | Node2Vec on STRING PPI network |
| PPI | PPI-SVD | 256 | SVD of STRING PPI adjacency matrix |
| PPI | PPI-RAW | 1024 | High-dim SVD of PPI adjacency |
| PPI | Mashup | 500 | Mashup (Cho et al.): RWR + log(Q+1/n) + SVD |
| GO | GO (anc2vec) | 200 | Sum-pool GO term embeddings per gene |
| GO | GO2Vec | 128 | Node2Vec on GO DAG, mean-pooled per gene |
| GO | Onto2Vec | 128 | Word2Vec on GO axiom sentences |
| KG | KG-ComplEx | 512 | ComplEx on STRING PPI + GO triples (real ⊕ imag) |

**Gene identifiers**: All `.pt` files use NCBI Entrez Gene IDs (strings) as the canonical identifier in `gene_order`. A static reference mapping is at `data/gene_id_mapping.tsv` (derived from HGNC). At runtime, `gene_name_utils.py` downloads the HGNC complete set for symbol/Entrez mapping.

**Missing gene handling**: Genes without embeddings receive NaN vectors in the `.pt` file. At fit time, `data_loader.py` fills NaN cells with per-column locations fit on the training subset — either a Huber M-estimate (robust and svd paths, c=1.345 for 95% normal efficiency) or the column mean (plain path). See the Preprocessing Pipeline below for the full sequence.

---

## Preprocessing Pipeline

### Fit/apply split (per-fold)

Every preprocessing statistic — impute locations, per-column standardization (plain only), PCA projection `V`, scalar normalize center/scale — is **fit** on a training-gene subset and **applied** to the full gene matrix. `SLDataManager.get_fold_embeddings(fold_data)` is called once per CV fold; the fitted transform dict is stored alongside each checkpoint so `predict.py` / `eval_finetuning.py` can rebuild the exact same feature space at inference.

The fit subset is controlled by `--preprocessing_fit_scope`:

- `train` *(default, leak-free)*: training-pair genes (pos + neg) ∪ genes in no pair. Genes that appear only in val or test pairs are excluded from the fit, so their embeddings can't shape the PCA basis. That holds in every CV type: on the current shared folds it drops about 1.2k of the 8,844 benchmark genes per CV1 fold (an edge split leaves low-degree genes test- or val-only), 3.0-4.2k per CV2 fold and 0.75-1.3k per CV3 fold.
- `all`: every gene regardless of fold. Reproduces the pre-refactor behavior — reintroduces CV2/CV3 leakage. Kept for A/B comparison.

### Three preprocessing variants (`--pca_method`)

The whole pipeline — impute, PCA, normalize — switches together. No mixing.

| stage | `robust` | `plain` *(default)* | `svd` |
|---|---|---|---|
| Impute NaN | per-column **Huber M-estimate** | per-column **mean** | per-column **Huber M-estimate** |
| Pre-PCA | *(none — ROBPCA handles centering internally)* | per-column **mean + std** (z-score) | *(none)* |
| PCA | **ROBPCA** (median-centered-SVD fallback) | **SVD** on standardized data | **median-centered SVD** (no ROBPCA) |
| Post-PCA normalize | global **median / MAD** | global **mean / std** | global **median / MAD** |

All PCA ops use deterministic `torch.linalg.svd` internally — per-fold `V` is bitwise reproducible across runs.

### Stage order (per fold)

```
  .pt file (raw d-dim) ──┐
                         │  (per modality, in order)
                         ▼
    (1) Load + normalize gene names (HGNC corrections) + dedup collisions
                         │
                         ▼
    (2) Impute NaN   — fit location on fit-subset rows, apply to all rows
                         │    (Huber if robust/svd, mean if plain)
                         ▼
    (3) PCA (optional)  — fit V on fit rows, apply to all rows
                         │    (robust: ROBPCA; svd: median-centered SVD;
                         │     plain: per-column standardize + SVD)
                         ▼
    (4) Normalize — fit scalar center/scale on fit rows, apply to all
                         │    (robust/svd: median/MAD; plain: mean/std)
                         ▼
                         └──► (5) Concatenate modalities along feature axis
                                                │
                                                ▼
                        (6) Post-concat PCA (optional, off by default)
                            fit on fit rows of concat, apply to all rows
                                                │
                                                ▼
                        (7) Re-normalize after post-PCA (global scalar; only with step 6)
                                                │
                                                ▼
                                model input  (n_genes, k_out)
```

### Why two PCA stages

**Neither stage runs by default (2026-10-05).** A direct `train.py` run and the reported model (the SLURM pipeline's PCA arm `none`) feed the encoder the normalized, concatenated features as they are (steps 1, 2, 4, 5); there is then no per-column standardization either, since that belongs to the PCA. The two stages below are opt-in. Since 2026-10-20 the SLURM pipeline runs once per **PCA arm** (`slurm/config.conf`, `PCA_ARMS`, default all four), the 2x2 {step 3 no / yes} × {step 6 no / yes}: `none`, `joint` (step 6 only, at `PCA_JOINT_VARIANCE`, 0.8, or `PCA_JOINT_DIM` components: one count for `joint` and `both`, which must stay below the narrowest concatenation step 6 meets in either arm, since `train.py` refuses a count at or above the width it reduces; since 2026-10-21, as step 6 would then not run under a label that says it did), `per_modality` (step 3 only, at `PCA_MODALITY_VARIANCE`, 0.8) and `both`, each the whole pipeline (selection, combos, eval, ablations, tables) in its own results root (`results/`, `results/pca_<arm>/`). They superseded the one-day fixed-selection PCA ablation of 2026-10-19 (`slurm/run_pca_ablation.sh`, never run on the cluster). Single-embedding runs never get step 6 (on one modality it would repeat step 3, or, after it, cut an already flat spectrum at an arbitrary place): native dimension in `none` / `joint`, step 3 in `per_modality` / `both`, so the two arms of each pair rank the same runs and select the same modalities. The repo-root orchestrator tabulates every arm against the first (`sl_comparison/results/pca_arms/tab_pca.csv`). An arm changes more than the PCA: the encoder's input width, and with it the entry layer's capacity (on the 2026-08-11 cv1 winner set re-fit on the current embedding files and shared folds, cv1 fold 0 at 0.8: `none` 4,864, `per_modality` 1,951, `joint` 1,570, `both` 1,291 encoder inputs; `L1_LAMBDAS` is not retuned for any); under the default `plain` each PCA also z-scores every column, where `none` rescales each modality by one scalar; in `both`, step 6 z-scores the per-modality components again, so every kept component weighs the same; and another width draws another random stream (initialisation, shuffles, dropout masks).

**Per-modality PCA (step 3, optional; on in every SLURM training of PCA arms `per_modality` and `both`, single-embedding runs included, off in `none` and `joint`)** runs on the imputed features — z-scored per column under the default `plain` (correlation spectrum), raw under `robust`/`svd` (covariance spectrum) — and reduces each modality independently. The basis changes k a lot: fit on all genes of the pre-rerun files at 0.8, ESM-2 1280d keeps 80 components under `plain` and 20 under `svd`, Geneformer V2 1152d 504 and 501. Normalization (step 4) happens *after* per-modality PCA so cross-modality scales are equalized on the reduced representation.

**Post-concat PCA (step 6)** runs on the already-normalized, concatenated matrix. Multi-modal combos can still leave 1,000+d inputs after step 5 (summed across modalities), which blows up the first-layer parameter count. A single PCA pass on the joint space compresses redundancy across modalities.

**Re-normalization after post-PCA (step 7)** is required because PCA output has per-component variance structure (leading PCs carry far more variance than trailing). Without it, the first Linear's Kaiming init is mis-calibrated. Step 7 mirrors step 4's scalar rescaling — a uniform rescale that preserves the PCA decomposition.

### ROBPCA details (robust path only)

- Implementation: `robpy.ROBPCA` (Hubert, Rousseeuw & Vanden Branden, 2005).
- Pipeline: Stahel–Donoho outlyingness on a classical-PCA subspace → h-subset eigendecomp → orthogonal-distance (OD) filter → final eigenvectors from cov of OD-filtered subset.
- The optional final FastMCD step is disabled (crashes on high-k score matrices near the rank limit); the step-4 OD-filtered eigenvectors are already robust per Hubert et al. 2005.
- Rank-deficient feature dims are removed via SVD pre-conditioning before ROBPCA.
- If ROBPCA fails (near-singular covariance, NaN components), retries with α ∈ {0.85, 0.90, 0.95}.
- Fallback (still inside the robust path): median-centered SVD when `robpy` is unavailable or all retries raise.

### Plain PCA details (plain path only)

- Per-column mean + (unbiased) std computed on fit rows; the std of an exactly-constant column (std == 0) is set to 1. This guards only constant columns and is not a 1e-8 floor, so tiny but real stds (e.g. mashup's diffusion-tail columns) are kept and the z-score stays scale-invariant.
- Fit rows standardized → deterministic `torch.linalg.svd`; top-k right-singular vectors become `V`.
- Apply: test rows are standardized with the stored fit mean/std, then projected by `V`.

### Default behavior in the SLURM pipelines

Per PCA arm (`slurm/config.conf`, "PCA arms": `pca_single_args` for the single-modality runs, `pca_multi_args` for the multi-modal ones):

| Script | Per-modality PCA (step 3) | Post-concat PCA (step 6) |
|---|---|---|
| `run_embedding_benchmark.sh`, `run_best_per_category.sh` (single-modality) | `--pca_variance $PCA_MODALITY_VARIANCE` in arms `per_modality` / `both`; disabled (raw dims) in `none` / `joint` | disabled in every arm (`--no_post_pca`) |
| `run_best_per_category_combo.sh` (multi-modal), `run_ablation.sh` (leave-one-category-out; it refuses an architecture signature that differs from its combo's) | `--pca_variance $PCA_MODALITY_VARIANCE` in `per_modality` / `both`; disabled in `none` / `joint` | `--post_pca_variance $PCA_JOINT_VARIANCE` (or `--post_pca_dim $PCA_JOINT_DIM`) in `joint` / `both`; disabled (`--no_post_pca`) in `none` / `per_modality` |
| Direct `python train.py` | off unless `--pca_variance` passed | off unless `--post_pca_variance` / `--post_pca_dim` passed |

Arm `none` (no PCA at all) is the reported model since 2026-10-05; from 2026-08-24 to then the combo ran step 6 at 0.8 on native-dimension single-modality winners, which is arm `joint` (the published model). Both targets default to 0.8 and are set for a whole submission; `POST_PCA_VARIANCE` / `POST_PCA_DIM`, the former opt-in, are refused since 2026-10-20, and so is a PCA flag in `ARCH_ARGS`.

Every SLURM training script threads `--preprocessing_fit_scope "${PREPROCESSING_FIT_SCOPE:-train}"` and `--pca_method "${PCA_METHOD:-plain}"`. Override by env var, e.g., `PCA_METHOD=robust bash slurm/submit_pipeline.sh`. **`plain` everywhere since 2026-08-10** — the tree previously ran `robust` for the per-embedding benchmark and `svd` for the combo.

---

## Parameters

| Component | Shape (`[256, 128, 64]` example) | L1 Penalized? | Purpose |
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
--encoder_dims 32 32 32 32 32 32 32  # Layer widths (any number of layers;
                                     # last entry is the projection/latent dim).
                                     # SLURM default (slurm/config.conf): entry
                                     # lift + 5 residual blocks + projection.
                                     # Unset: [--hidden_dim 512, --latent_dim 256].
--siamese_encoder_type residual      # residual (y=x+F(x) skips, default) | mlp (no skips)
--no-last-layer-bias                 # Remove the shared projection bias (gene degree prior); with --use_cell_lines only together with --no_per_head_last_layer_bias (the default per-head c_h replaces it)
--pd_epsilon 0.001                   # PD regularizer epsilon (0 to disable; also acts as score-level skip)
--dropout 0.2                        # Dropout rate for hidden layers
--no_gene_term                       # (--use_cell_lines) drop the per-gene propensity g(h_i) + g(h_j) added to each head's logit (on by default)
--no_pair_features                   # (--use_cell_lines) drop the symmetric MLP of [|z_i - z_j|, z_i * z_j] added to each head's logit (on by default)

# Preprocessing / dimensionality reduction
--pca_variance 0.8              # Per-modality PCA variance target (0,1); SLURM: PCA arms per_modality / both (PCA_MODALITY_VARIANCE)
--pca_dims N1 N2 ...            # Per-modality exact component counts (alt to --pca_variance)
--post_pca_variance 0.8         # Post-concat PCA variance target (default: none, no PCA)
--post_pca_dim K                # Post-concat PCA exact component count (overrides variance).
                                # K at or above the concatenated width it would
                                # reduce is refused (a ValueError at the fold's
                                # fit, 2026-10-21): no step 6 would run.
                                # SLURM passes one only in PCA arms joint / both
                                # (PCA_JOINT_VARIANCE / PCA_JOINT_DIM,
                                # slurm/config.conf). Note a
                                # variance target is not
                                # comparable across --pca_method (plain resolves
                                # k on the correlation spectrum: 1087 vs svd's
                                # 145 at the same 0.8), and k drifts with the
                                # combo's modality selection. Set it to pin the
                                # encoder width.
--no_post_pca                   # Disable the post-concat PCA step (already off by default)
--pca_method plain              # Pipeline variant (default: plain):
                                #   plain : mean-impute + mean+std standardize + SVD + mean/std normalize
                                #   robust: Huber-impute + ROBPCA + median/MAD normalize
                                #   svd   : Huber-impute + median-centered SVD + median/MAD normalize
--preprocessing_fit_scope train # Which gene rows fit the preprocessing, per fold:
                                #   train: training-pair genes (pos + neg) + genes in no pair;
                                #          val-/test-only genes excluded (leak-free, default)
                                #   all  : every gene (legacy, leaky; for A/B comparison)

# Regularization
--l1_lambdas 0.02 0.01 0.002   # Per-layer L1: one value per weight matrix =
                               # len(encoder_dims) (7 for the example above,
                               # which is SLURM's; 3 for the SiameseSL default,
                               # 2 for train.py's unset default)

# Training
--epochs 200                    # Max training epochs
--batch_size 256                # Gene pairs per mini-batch
--learning_rate 0.001           # AdamW learning rate
--weight_decay 1e-5             # AdamW weight decay (decoupled L2)
--warmrestart_T0 50             # CosineAnnealingWarmRestarts initial cycle length (epochs)
--warmrestart_Tmult 2           # CosineAnnealingWarmRestarts cycle length multiplier
--pos_neg_ratio 1.0             # NO effect: the shared folds (--folds_dir, required)
                                # fix the class balance (every screened negative,
                                # about 1:4.8 overall); legacy own-fold path only
--eval_interval 10              # Evaluate the VAL split every N epochs (selection is val-only)
--patience 20                   # Early stopping (in eval intervals)
--num_folds 5                   # Cross-validation folds
--seed 42                       # Random seed
--cpu                           # Force CPU (flag, default: off)
```
