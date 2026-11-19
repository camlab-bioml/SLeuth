# Siamese Network for Synthetic Lethality Prediction

A PyTorch implementation of a Siamese neural network for predicting Synthetic Lethality (SL). Supports 18+ gene embedding types across 6 categories with a unified pipeline: every modality is imputed, optionally PCA-reduced, and normalized before concatenation — the same code path whether you pass one file or many. The preprocessing pipeline is **fit per-fold on training-pair genes (pos + neg) plus genes in no pair**; genes seen only in val/test pairs are excluded, so their embeddings never shape the basis (leak-free in every CV type).

## Key Features

- **18+ embedding types**: ESM-2, GO, Geneformer, scGPT, BioConceptVec, PPI network, and more
- **Extra score terms** (cell-line mode, on by default since 2026-10-05): a per-gene propensity `g(h_i) + g(h_j)` and a symmetric MLP of `[|z_i − z_j|, z_i ⊙ z_j]` are added to each head's logit. Both start at exactly 0; `--no_gene_term` / `--no_pair_features` remove them (in SLURM through `ARCH_ARGS`). See `ARCHITECTURE.md`.
- **Unified embedding pipeline**: Per-modality impute → PCA (optional, `--pca_variance` / `--pca_dims`) → normalize → concat, then an optional post-concat PCA + renormalize (`--post_pca_variance` / `--post_pca_dim`). No PCA runs by default in a direct `train.py` run. The SLURM pipeline runs once per **PCA arm** (since 2026-10-20; `slurm/config.conf`, `PCA_ARMS`): `none` (no PCA, the reported model since 2026-10-05), `joint` (the post-concat PCA), `per_modality` (the per-modality PCA) and `both`, each in its own results root (`results/`, `results/pca_<arm>/`). Three variants, selected by `--pca_method`:
    - `plain` *(default, used by every SLURM job)*: mean-impute + per-column mean/std standardize + SVD + global mean/std normalize.
    - `robust` (opt-in): Huber-impute + ROBPCA (Hubert et al., 2005) + global median/MAD normalize.
    - `svd` (opt-in): Huber-impute + median-centered SVD (no per-column standardize, no ROBPCA) + global median/MAD normalize.
  The plain path is fully classical; the robust and svd paths keep the robust measures (Huber / median / MAD). `robust` and `svd` are kept so earlier runs stay reproducible. Same code path for one or many modalities; L1-regularized first layer handles feature selection.
- **Leak-free per-fold preprocessing**: Every statistic (impute locations, PCA `V`, normalize scalars) is fit on training-pair genes (pos + neg) + genes in no pair (val-/test-only genes excluded) and applied to all genes. The fitted transform is saved with each checkpoint so `predict.py` / `eval_finetuning.py` rebuild the exact same feature space at inference.
- **Pool PaRTI pooling**: PageRank-based pooling for ESM protein representations (Tartici et al., 2025)
- **All genes**: Can work with any human gene that has a protein sequence (not limited to SL dataset genes)
- **Siamese architecture**: Shared encoder ensures consistent representations
- **Reproducible**: Fixed random seeds for deterministic training

## Installation

```bash
pip install torch fair-esm biopython requests tqdm networkx scikit-learn pandas robpy
```

A minimal starting set; the complete dependency list (every embedding
generator, the external-screen eval) is in `SETUP.md` and
`slurm/reset_env.sh`.

## Quick Start

### 1. Generate ESM embeddings for all human genes

```bash
# Resolves one canonical protein sequence per listed Entrez Gene ID (the UniProt
# reviewed canonical entry via HGNC uniprot_ids, both source files tracked in
# data/cache; NCBI RefSeq only for the ~0.5% of genes UniProt does not cover)
# and generates ESM embeddings with Pool PaRTI. A gene list is required:
# ../data/gene_universe.txt is built by STEP 0b of slurm/run_generate_embeddings.sh
# (--sl_path <SL pairs file> works too).
python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \
    --output ../data/all_genes_esm2.pt --device cuda:0

# Or use mean pooling instead. NOT the pipeline build: it is stamped a
# marked-down generator_version that embedding_versions.py reports stale, so
# write it BESIDE the catalog file, never over it.
python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \
    --output ../data/all_genes_esm2_mean.pt --pooling mean
```

`--gene_list` takes a `.pt` file (its `gene_order`), a `.txt` file with one
Entrez Gene ID per line, or a two-column SL pairs file.

This creates:
- `../data/all_genes_esm2.pt` - ESM embeddings (~20k genes, ~200MB: `embeddings` and `raw_embeddings`, plus `protein_accessions`)
- `../data/all_genes_esm2.genes.txt` - Gene list for reference

**Pool PaRTI** (default) uses PageRank on attention matrices to weight residue importance,
outperforming mean pooling especially for identifying functionally critical regions.

### 2. Train the model

```bash
# train.py never creates directories: make <output_dir>/checkpoints/ first.
# Single modality (pipeline: impute → normalize, fit per-fold on
# training-pair + unpaired genes; native dim, no PCA)
mkdir -p results/siamese_esm/checkpoints
python train.py \
    --embeddings_paths ../data/all_genes_esm2.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --folds_dir ../sl_comparison/folds \
    --output_dir results/siamese_esm \
    --cv_type cv1 \
    --epochs 200 \
    --seed 42

# Multi-modal (each modality: impute → normalize; concatenate; no PCA)
mkdir -p results/multi_bio_go_ppi/checkpoints
python train.py \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --folds_dir ../sl_comparison/folds \
    --output_dir results/multi_bio_go_ppi \
    --cv_type cv1

# Multi-modal with PCA (each modality: impute → PCA → normalize; concatenate
# → post-concat PCA 0.8 + renormalize; both are opt-in; the SLURM pipeline's
# PCA arm both at its default targets)
mkdir -p results/multi_bio_go_ppi_pca/checkpoints
python train.py \
    --embeddings_paths ../data/all_genes_bioconceptvec.pt \
        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --folds_dir ../sl_comparison/folds \
    --output_dir results/multi_bio_go_ppi_pca \
    --cv_type cv1 --pca_variance 0.8 --post_pca_variance 0.8

# Opt in to the robust pipeline: Huber-impute + ROBPCA + median/MAD normalize
# (the default --pca_method plain is mean-impute + z-score + SVD + mean/std)
mkdir -p results/siamese_esm_robust/checkpoints
python train.py \
    --embeddings_paths ../data/all_genes_esm2.pt \
    --sl_path ../data/SL_SynLethDB_experimental.txt \
    --folds_dir ../sl_comparison/folds \
    --output_dir results/siamese_esm_robust \
    --cv_type cv1 --pca_variance 0.8 --pca_method robust
```

Every run needs `--folds_dir` (the shared folds built by
`sl_comparison/prepare_folds.py --val_frac`): early stopping and checkpoint
selection are validation-only, and only the shared folds carry a validation
split, so `train.py` exits without it. It also needs
`<output_dir>/checkpoints/` to exist beforehand (the `mkdir -p` lines above;
`train.py` stops with `FileNotFoundError` otherwise). Without `--use_cell_lines`
it trains the single-output `SiameseSL` on the folds' own labels; the SLURM
pipeline runs `--use_cell_lines --neg_pairs_path ...` (see
`slurm/config.conf`).

### 3. Predict SL for gene pairs

```bash
# Single pair (accepts gene symbols or Entrez IDs)
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm2.pt \
    --gene1 BRCA1 --gene2 PARP1

# Using Entrez IDs directly
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm2.pt \
    --gene1 672 --gene2 142

# Find top SL partners for a gene
python predict.py \
    --model results/siamese_esm/checkpoints/fold_0_best.pt \
    --embeddings_paths ../data/all_genes_esm2.pt \
    --gene1 BRCA1 --top_k 100 --output brca1_partners.csv
```

`predict.py` loads single-output checkpoints only. For a cell-line or pooled
model (every `--use_cell_lines` run, i.e. the SLURM default) use the export
below.

### 4. All gene pairs, per cell line

`export_pair_predictions.py` (since 2026-10-07) scores every gene pair with a
trained combo and writes one matrix per output head. It works on any
checkpoint family and retrains nothing.

```bash
python export_pair_predictions.py \
    --model_dir results/<combo>_cv3 \
    --embeddings_paths <the combo's config.json embeddings_paths, same order> \
    --finetuned_eval_dirs eval_finetuning_cv1 --finetuned_datasets Adamson
```

It writes into `<model_dir>/pair_predictions/`:

| File | Content |
| --- | --- |
| `sl_prob_<HEAD>.npy` | n x n, symmetric: sigmoid of that head's logit, averaged over the fold models. A `per_head` model writes `K562`, `JURKAT`, `A549`, `HELA`, `A375`, `293T`, `PC9`, `OTHER` (each read with that head's embedding; `OTHER` is a catch-all of rare lines and pairs with no recorded line) and `ANY`; a pooled or single-output model writes `ANY` only |
| `finetuned_<Screen>_<mode>_<cv>.npy` | with `--finetuned_eval_dirs`: the predicted screen value of every gene pair, in the screen's own sign and units (Adamson: `Gamma`), from the models `eval_finetuning.py` fine-tuned under its `<cv>` screen split with `<mode>` (`LP_last` / `Full`), averaged over the fold models and scored through the head they were fine-tuned with (K562 for Adamson on a `per_head` model) |
| `genes.txt`, `genes.tsv` | Entrez ID (and HGNC symbol) of row and column i |
| `manifest.json` | what every file is, with the sha256 of the checkpoints and fine-tuned models it scored; written last, so a directory without it is an unfinished export |

Options: `--universe shared` (the 8,844 benchmark genes in
`sl_comparison/folds/genes.txt` order instead of every embedding gene),
`--genes_file` (your own list), `--per_fold` (each fold model's matrices too),
`--also_logits`, `--dtype float16`. One matrix is n²·4 bytes: 1.5 GB at 19,397
genes, 0.31 GB at 8,844.

```python
from export_pair_predictions import open_pair_predictions
pp = open_pair_predictions("results/<combo>_cv3/pair_predictions")
pp.matrix("K562")[pp.gene_row("BRCA1"), pp.gene_row("PARP1")]
pp.pair_table([("BRCA1", "PARP1"), ("672", "142")])   # every matrix, one row per pair
```

`open_pair_predictions` refuses an export whose fold checkpoints or fine-tuned
models have changed since (`check_model=False` reads it anyway). A gene is
looked up by Entrez ID, by its current HGNC symbol (`genes.tsv`, and the
`gene1_symbol` / `gene2_symbol` columns of the eval CSVs) or by an older name
the HGNC table resolves: Adamson labels 37 of its 542 genes by one (`MRE11A`
for `MRE11`). That table is this repository's
`data/cache/hgnc_complete_set.txt`, the file the pipeline reads; the reader
finds it by its own path, so the lookup gives the same answer from any working
directory and downloads nothing (without the file an older name is a
`KeyError`). An all-digit name is only ever an Entrez ID. Join tables on the
Entrez columns.

What the numbers are and are not:

- The diagonal is NaN. To rank a gene's partners use `np.argsort(-row)`, which
  puts NaN last.
- `ANY` on a `per_head` model is the largest of the eight head probabilities
  within each fold. It is the score the benchmark ranks by; no head is trained
  on the any-line label, so do not read it as a calibrated probability.
- A head's probability is on the scale of the pairs screened in that line
  (SL fraction 3.5% for `A549` to 43.6% for `A375` on the current labels):
  rank pairs within a head, do not compare levels across heads. `A375`,
  `293T` and `PC9` rest on about 2,500 screened pairs or fewer, and `OTHER`
  is a catch-all of rare lines and pairs with no recorded line, not a cell
  line.
- The fold mean averages probabilities (or predicted values), not logits. It
  is not a held-out prediction for a labelled SynLethDB pair: each fold model
  trained on 72% of the labelled pairs under cv1 and on about half under cv2 /
  cv3 (the SLURM job exports the cv3 combos by default). With `--per_fold`,
  fold k's matrix is held out for the pairs outside fold k's train and val
  splits (`sl_comparison/folds/<cv>/fold_<k>/`). Under cv3 the fold's test
  split is only the pairs with both genes unseen: one labelled pair in five
  over the five folds.
- Every labelled pair the models were trained on lies within the 8,844
  benchmark genes. A pair with a gene outside them (79% of the pairs of a
  19,397-gene matrix) is an extrapolation from the gene embeddings.
- A `finetuned_*` matrix is in-sample at the screen's own pairs: every fold
  model was fine-tuned on the same screen split, so its `train` pairs (fitted)
  and `val` pairs (checkpoint selection) are in-sample in the fold mean and in
  every `--per_fold` file: 72% + 8% of the screen's pairs under `cv1`. Only
  the pairs marked `split == test` in
  `eval_finetuning_<cv>/<Screen>_finetuned_allpairs_<mode>.csv` are held out;
  a screen pair's matrix entry is that file's `pred_outcome_mean`, and
  `manifest.json` records the split and names the file. Every pair the screen
  did not measure (99.9% of an Adamson matrix) is an extrapolation that no
  metric of the run grades.
- The Adamson value (`finetuned_Adamson_*.npy`, `pred_outcome_*`) is the
  `Gamma` column of `external data/pair_sgRNA_phenotypes.txt`, averaged over
  the sgRNA-pair rows of a gene pair (rows with a non-targeting guide are not
  used) and fitted as `-Gamma`. It is the growth phenotype of the double
  knockdown, not an interaction score: 97% of the 146,583 pair means are
  negative and a per-gene additive fit explains 90% of their variance
  (`numbers.json`, `F4_additive_r2`). A very negative predicted value mostly
  marks a pair with a growth-essential gene; the SL prediction is
  `sl_prob_<HEAD>.npy`.
- float32 stores a probability above 1 - 6e-8 (a logit above about 17) as
  exactly 1. `manifest.json` counts those entries per matrix; `--also_logits`
  keeps them ranked.
- Every (fold, head) matrix is checked against `model.forward` on sampled
  pairs before it is used; a mismatch stops the export. A fine-tuned set that
  fails its check, or cannot be read, is left out and reported (exit 1) after
  everything else is written.

The free-space check reads the filesystem, not a per-user quota.

## Architectures

### 1. Siamese (inner product scoring)
```
Gene1 emb ─┬─> SharedEncoder ─> z1, h1 ─┐
           │                             ├─> τ·(z1ᵀz2 + ε·h1ᵀh2) + b ─> sigmoid ─> P(SL)
Gene2 emb ─┴─> SharedEncoder ─> z2, h2 ─┘
```
Inner product with PD kernel K = WᵀW + εI *when the projection has no bias* (`--no-last-layer-bias`, which `slurm/config.conf`'s `MODEL_ARGS` passes); train.py's default bias b adds the per-gene terms bᵀW(h1+h2) + ‖b‖² (a gene node-degree prior), so the score is then no longer bilinear in h (it stays a PSD kernel — the inner product of the feature map φ(h) = (Wh+b, √ε·h) — but is no longer h1ᵀKh2). Symmetric by construction either way.

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

**Reference only, not trainable by `train.py`.** The kernel and attention
models have no dense all-pairs scorer, so they cannot export the shared-fold
prediction matrix that every run now writes: `train.py`'s `--model_type`
accepts only `siamese` (argparse rejects `kernel|attention` in every mode, as
it does the removed `--rff_features` / `--bilinear_rank` / `--kernel_sigma` /
`--num_heads` flags), and `--use_cell_lines` always builds
`SiameseSLMultiCell`. The classes are kept in `siamese_esm.py` for reference
and for loading legacy checkpoints with `predict.py`.

### 3. Attention (cross-attention between genes; reference only, see above)
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

Since 2026-10-20 the pipeline runs once per **PCA arm** of `PCA_ARMS` (`slurm/config.conf`, "PCA arms"; default `none joint per_modality both`, any subset, always in this order): the 2x2 of PCA within each modality (no / yes) × PCA on the concatenation of all modalities (no / yes). The combo and the leave-one-category-out ablation of arm `none` train with `--no_post_pca` (no PCA: the reported model since 2026-10-05), of `joint` with `--post_pca_variance $PCA_JOINT_VARIANCE` (0.8; or `--post_pca_dim $PCA_JOINT_DIM`, which wins when set: one count for `joint` and `both`, to keep below the narrowest concatenation their joint PCA meets, ablations included, since `train.py` refuses a count at or above it, 2026-10-21; the published model's PCA), of `per_modality` with `--pca_variance $PCA_MODALITY_VARIANCE --no_post_pca` (0.8) and of `both` with both PCAs. The single-embedding runs (Steps 2a and 2b) get the per-modality PCA only, so `none` and `joint` share their native-dimension runs and `per_modality` and `both` their per-modality-PCA runs: each pair's runs are trained once, as its OWNER (the pair's first arm that `PCA_ARMS` lists), in the owner's results root, and the two arms of a pair select the same winners. Arm `none` writes `results/` (the paths of before 2026-10-20); every other arm writes the same names under `results/pca_<arm>/`. Each arm's jobs are submitted as `PCA_ARM=<arm> sbatch ... --export=ALL` (the arm in sbatch's own environment; an `--export` list such as `--export=ALL,PCA_ARM=<arm>` would make Slurm imply `--get-user-env`, a login-environment fetch at every task launch that holds the job when it fails) under their usual job names and log to `slurm/logs/<stage>_pca_<arm>_<job id>.out` (arm `none`: the scripts' own names); each job's banner names its arm. `PCA_ARMS=joint bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate` trains one arm (or any subset) and leaves the other arms' outputs as they are (alone, `joint` trains its own single-embedding runs, in `results/pca_joint/`), but for the table across the arms, `../sl_comparison/results/pca_arms/`, which every best-per-category submission removes. `--skip-generate` is what keeps them: a submission that regenerates the embeddings (the default) removes the combo markers of EVERY arm, all four roots, since those combos were trained on the vectors it replaces; `--skip-reset-env` keeps the venv the other arms' jobs run from. Keep pairs whole (`none joint`, `per_modality both`) or list second arms only (`joint`, `both`): a subset that lists `none` (or `per_modality`) without `joint` (or `both`) makes it the pair's owner, whose Step 1 rebuilds the single-embedding runs the left-out partner's selection ranked; `submit_pipeline.sh` prints a WARNING before anything changes, and the partner's `tab:solo` belongs to another run until the partner is rerun. `POST_PCA_VARIANCE` / `POST_PCA_DIM`, the former post-concat PCA switch, are refused (`PCA_ARMS=joint` is that pipeline now), and so is a PCA flag in `ARCH_ARGS`.

By default `slurm/submit_pipeline.sh` submits 28 SLURM jobs, 586 tasks counting every array task (10 jobs and 207 tasks with one arm; Slurm's `MaxSubmitJobs` counts each task, see `slurm/README.md`). Once: the environment reset and the embedding generation. Per owner (`none` and `per_modality` by default): the benchmark array + summary and the best-per-category Step-1 array. Per arm: the combo, the external-screen eval array (`eval_finetuning.sh`, |EVAL_CV_TYPES| × |SIAMESE_VARIANTS| = 12 tasks by default; `--skip-eval` opts out), the leave-one-category-out ablation array + summary (`run_ablation.sh` / `run_ablation_summary.sh`, |ABLATE_CATEGORIES| × |CV_TYPES| × |SIAMESE_VARIANTS| = 72 tasks by default; `--skip-ablation` opts out) and, last, the all-gene-pair export (`export_pair_predictions.sh`, one job over every variant's cv3 combo of the arm; `--skip-pair-export` opts out). The two training branches (benchmark and best-per-category) run in parallel after generation completes; each arm's combo waits for its owner's Step-1 array; the eval and the ablation chain afterok on the combo and run in parallel with each other; the exports wait for every other job of every arm (afterany), because nothing reads their output and each writes 42 GB by default (`slurm/export_pair_predictions.conf`). The ablation does not retrain the combo: it reads it as its reference row, so no reported number can move. It keeps every run's fold score matrices by default, for `paper_tables.py --regrade` (`fold_<k>_<cv>_predictions.npy`, 5 folds × 313 MB on the 8,844 shared genes): about 113 GB for an arm's 72 runs; `ABLATION_PURGE_PREDICTIONS=1` removes them. The banner adds up the jobs and array tasks the submission queues and the disk of the fold matrices and pair export it keeps, up to 1,062 GB by default: per arm 19 GB of combos, 42 GB of pair export and 113 GB of ablation matrices, per owner up to 94 GB of benchmark and 89 GB of Step-1 matrices (a task whose embedding is absent keeps none); the eval's per-pair CSVs (about 5 GB per arm) and the checkpoints come on top, named on its `not counted:` line (since 2026-10-21). It queues neither the post-hoc cell-line ablation nor the `sl_comparison` jobs: whenever it trains combos it ends by printing them as a ready-to-paste block wired to its own job ids (since 2026-10-21; per arm the cell-line ablation, compare, external transfer (with the eval: without it nothing rebuilds `external_transfer/`) and paper tables, then, with two arms or more, the table across the arms (with one arm nothing rebuilds it: one arm has nothing to compare); `slurm/README.md`, `submit_pipeline.sh`).

The paper run is one command from the repo root, `bash run_benchmarking_siamese_vs_others.sh --force-generate`. It submits these siamese stages (minus the separate benchmark branch), for every arm, next to the shared folds and the SLGNN / NSF4SL / SLMGAE array (those two queued right after the generation, ahead of every siamese job, since 2026-10-21), and adds four jobs per arm: the post-hoc cell-line ablation (`cell_line_ablation/run_cell_line_ablation.sh`, afterok on the arm's combo), the external transfer (`sl_comparison/slurm/run_external_transfer.sh`, after the arm's eval tasks and the shared models), the four-model compare, and the paper tables (`sl_comparison/slurm/run_paper_tables.sh` → `sl_comparison/results/paper_tables/` for arm `none`, `sl_comparison/results/pca_<arm>/paper_tables/` for the others), and, with two arms or more, one table across the arms (`run_paper_tables.sh --pca_arms_table` → `sl_comparison/results/pca_arms/`, `tab_pca.csv`). The all-gene-pair exports run after all of them. See the root `README.md`.

Both orchestrators make a fresh `RUN_TOKEN` (replacing one inherited from the shell) and export it before the first `sbatch`, except where a submission joins an earlier run, per arm: `submit_pipeline.sh --ablation-only` takes each ablated arm's token (`results[/pca_<arm>]/best_per_category.json` `_selection_meta.run_token`; the arm's previous ablation dirs are removed at submission), and the repo-root one's `--skip-siamese` each reused arm's. Both first compare each marker-named combo's `config.json` with the PCA this shell gives its arm and refuse a mismatch before anything changes (since 2026-10-21; every ablation task would stop at its signature check, and every reused arm's tables would carry the wrong PCA label), printing what to submit instead, each command complete (the settings taken from the calling shell, the three targets in full): one command when one set of targets trains the combos of every arm (always so for a forgotten or stray target); when no one set does, the repo-root one only WARNs, naming the arms whose tables will be PAPER INCOMPLETE, and `--ablation-only` prints one line per arm, or per group of an arm's variants trained at different targets, to submit one after the other (`slurm/README.md`): export the targets the combos were trained with; arms whose tokens differ get their own in their sbatch's environment (`PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ... --export=ALL`), with a WARNING. Each job's `slurm/config.conf` keeps that exported value and makes its own only when none is set. Every benchmark run (Step 2a), best-per-category single-embedding run (Step 2b) and ablation run writes it to `<output_dir>/run_token.txt` after it trains. The combo's selection step refuses a candidate whose token is missing or differs, the ablation summary marks a mismatched run `MISMATCH` with no delta, and the benchmark summary shows a directory with another token as `STALE` (not tabulated), so nothing left over from an earlier submission is selected or tabulated. A job submitted on its own gets a fresh token (each array task its own), so export one before submitting stages by hand (`export RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"`, as in the Step 1 block below), or export an earlier submission's token to join it (`export RUN_TOKEN="$(cat results/pcavar0.8_<type>_cv1/run_token.txt)"`; a direct `sbatch slurm/run_embedding_benchmark_summary.sh` of an earlier benchmark array needs that array's, from `results/<type>_cv1/run_token.txt`). A job submitted on its own is also arm `none` unless `PCA_ARM` is exported (or passed, `PCA_ARM=<arm> sbatch --export=ALL ...`); export the run's `PCA_ARMS` and PCA targets with it when they were not the defaults: `PCA_ARMS` decides which arm's root holds an arm's single-embedding runs.

### Step 0: Reset Environment (`reset_env.sh`)

Recreates the Python venv from scratch on a GPU node. Must run on a GPU node (gpu3). Run this first if the environment is missing or broken.

```bash
cd siamese_sl
sbatch slurm/reset_env.sh
# Removes old venv, creates fresh one, installs all dependencies
# (torch, fair-esm, robpy, transformers, pykeen, etc.) except the optional
# scprint and allennlp (see Supported Embeddings)
# Verifies all packages and pre-caches the HGNC gene name database
```

### Step 1: Generate Embeddings (`run_generate_embeddings.sh`)

Downloads source data and generates all .pt embedding files. Run this first.

```bash
cd siamese_sl
# One token for every stage submitted from this shell (sbatch exports the
# environment); without it each Step-2a/2b task mints its own, so the combo
# refuses every candidate and the benchmark summary shows every cell STALE.
export RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"
# The PCA arm of every siamese job submitted from this shell (slurm/config.conf,
# "PCA arms"; without it a job is arm none): none (no PCA, results/), joint,
# per_modality or both (results/pca_<arm>/). Steps 2a/2b then train the arm's
# single-embedding runs in its owner's root and its combos in its own.
export PCA_ARM=none
GEN=$(sbatch --parsable slurm/run_generate_embeddings.sh)
# 1. Download SynLethDB experimental SL pairs
# 2. Generate ESM-2 embeddings (UniProt reviewed canonical sequences, RefSeq fallback)
# 3. Generate GO embeddings (from GAF file)
# 4. Generate the 17 other embedding types (PRECOMPUTED_TYPES; esmc is
#    commented out and not generated; auto-downloads STRING PPI, GO OBO, etc.).
#    scgpt is the one exception: it is rebuilt from the scGPT whole-human
#    checkpoint folder (best_model.pt + vocab.json), which is git-ignored and
#    must be copied by hand to ../data/embeddings_cache/scGPT_human/ first
#    (see Supported Embeddings) — it is a REQUIRED catalog type, so without it
#    the job logs "[scgpt] Skipped (not available)" and then fails its
#    end-of-run catalog check (just before step 5), after every other build.
#    scprint and seqvec need packages reset_env.sh does not install, so by
#    default they log "[type] Skipped (not available)" and 15 are built
#    (17 .pt files with esm2 and go; see Supported Embeddings).
# 5. Fail unless every required catalog .pt exists (OPTIONAL_TYPES in
#    slurm/run_generate_embeddings.conf lists the accepted absences) and is a
#    current build (embedding_versions.py).
```

Each generator stamps the `.pt` it writes with `generator_version` (`EMBEDDING_GENERATOR_VERSION` in `generate_all_genes_esm.py`). `python embedding_versions.py --data_dir ../data` lists every EMB_CATALOG embedding (esm2 and go included; `--all` adds ALL_EMBEDDINGS) that is missing, carries another version, has genes that differ from `../data/gene_universe.txt`, or does not list them in `all_genes_esm2.pt`'s gene order, and exits 1 if there is any (2 if the catalog cannot be read). The files of `slurm/run_generate_embeddings.conf` `OPTIONAL_TYPES` (esmc, scprint, seqvec, genept) may be absent (reported ABSENT); present, they must be current. The repo-root orchestrator regenerates when it exits non-zero, and the generation job itself ends with this check.

### Step 2a: Embedding Benchmark (`run_embedding_benchmark.sh`)

Benchmarks each embedding individually across 3 CV types, with the single-embedding PCA of the job's arm: native dimensionality (no PCA) in arms `none` / `joint`, PCA within the modality in `per_modality` / `both`, into the owner's root (`results/`, or `results/pca_per_modality/` by default). No later stage reads it: the selection ranks Step 2b's runs.

```bash
# Chain after generation completes. One task per ALL_EMBEDDINGS x CV_TYPES
# entry (20 x 3 = 60 today), read from the conf so the range cannot go stale.
N=$(bash -c 'unset SLURM_JOB_ID; source slurm/config.conf >/dev/null; echo $(( ${#ALL_EMBEDDINGS[@]} * ${#CV_TYPES[@]} ))')
BENCH=$(sbatch --parsable --array=0-$((N - 1)) --dependency=afterok:$GEN --kill-on-invalid-dep=yes slurm/run_embedding_benchmark.sh)
# Trains each embedding × 3 CVs. The comparison table is a separate job, afterany
# (as in submit_pipeline.sh) so one failed task does not cancel it; the kill flag
# cancels the array if generation fails, instead of leaving it (and the summary)
# pending forever.
sbatch --dependency=afterany:$BENCH slurm/run_embedding_benchmark_summary.sh
# Note: submit_pipeline.sh computes the correct --array range automatically;
# these manual commands are shown for reference only.
```

### Step 2b: Best-per-Category Multi-Modal (`run_best_per_category.sh`)

Retrains each embedding under `SELECTION_VARIANT` (default `pd:per_head`) with the single-embedding PCA of the job's arm (`PCA_SINGLE_ARGS`: native dimension, `--no_post_pca`, in arms `none` / `joint`; `--pca_variance $PCA_MODALITY_VARIANCE --no_post_pca` in `per_modality` / `both`) into its owner's root; then the arm's combo job picks the best embedding per category ONCE on `SELECTION_CV` (default cv1) by validation `val_<select_metric>_mean` (`auprg_any` by default) and trains a multi-modal combo from those winners for every variant × CV with the arm's PCA (`PCA_MULTI_ARGS`; in arm `none` the concatenated features, no PCA, since 2026-10-05) into the arm's root. Can run in parallel with Step 2a.

```bash
# Chain after generation completes (runs in parallel with 2a). bestcat_n_tasks
# (run_best_per_category.conf) prints the array size: one task per
# EMB_CATALOG x CV_TYPES entry (19 x 3 = 57 today), plus the other three
# variants' runs when BESTCAT_EXTRA_VARIANTS=1. A short range silently drops
# the last catalog entries (kg_complex, the only kg candidate, is last).
N=$(bash -c 'unset SLURM_JOB_ID; source slurm/config.conf >/dev/null; source slurm/run_best_per_category.conf >/dev/null; bestcat_n_tasks')
BESTCAT=$(sbatch --parsable --array=0-$((N - 1)) --dependency=afterok:$GEN --kill-on-invalid-dep=yes slurm/run_best_per_category.sh)
# The array only trains the single embeddings. Sub-steps 2-4 below (selection,
# combos, summary) are a separate job, afterany as in submit_pipeline.sh so one
# failed embedding does not cancel it; it writes <arm root>/best_per_category.json
# and <arm root>/<prefix>latest_combo_<cv>.txt (<arm root> = results for
# PCA_ARM=none, results/pca_<arm> otherwise).
sbatch --dependency=afterany:$BESTCAT slurm/run_best_per_category_combo.sh
# Note: submit_pipeline.sh computes the correct --array range automatically;
# these manual commands are shown for reference only.
```

For another arm, export its `PCA_ARM` in the Step 1 block instead. The second arm of a pair (`joint` after `none`, `both` after `per_modality`) ranks the first one's single-embedding runs: submit only its combo job, `PCA_ARM=joint sbatch --export=ALL --dependency=afterany:$BESTCAT slurm/run_best_per_category_combo.sh`, with the `PCA_ARMS` of the run exported (the default lists both arms of each pair; with `PCA_ARMS=joint` alone, arm `joint` ranks runs of its own in `results/pca_joint/`). A hand Step 1 of a pair's first arm rewrites the runs the other arm's selection ranked, with no warning (no orchestrator checks): resubmit that arm's combo afterwards, or its `tab:solo` is another run's.

A combo job submitted by hand (as above, or resubmitted alone to retrain a failed combo) removes what was computed from the combos it replaces in its arm (and the table across the arms) and queues none of the jobs that rebuild it; `../sl_comparison/README.md`, "After a combo job submitted by hand", lists what to rerun.

`PCA_VARIANCE` / `BEST_CAT_PCA_VARIANCE` only names the `pcavar<variance>_*`
directories; it sets no PCA. A single-embedding run takes the PCA of its
owner's arm (none in `results/`, `--pca_variance $PCA_MODALITY_VARIANCE` in
`results/pca_per_modality/`), and a combo or ablation run the PCA of its own
arm (`slurm/config.conf`, "PCA arms"); `PCA_MODALITY_VARIANCE`,
`PCA_JOINT_VARIANCE` (0.8 each) and `PCA_JOINT_DIM` (empty; an exact count,
which wins) set the targets for every arm of a submission. `PCA_JOINT_DIM` is
one count for arms `joint` and `both` alike: keep it below the narrowest
concatenation their joint PCA meets, the ablations' included (in arm `both`
about 1,951 columns for the 2026-08-11 cv1 combo at 0.8, about 1,140 without
`ppi`; in `joint` the native width without the widest category). `train.py`
refuses a count at or above that width (since 2026-10-21: no PCA would run
under a label that says it did) at the fit of the first fold that narrow, in
each combo or ablation run: in arm `joint` fold 0 (its width is the same in
every fold); in arm `both`, whose width moves per fold and CV, after that
run's earlier folds and, in the combo job, earlier combos may have trained (the
refused run writes no marker); no check at submission sees a width.

Step 2b sub-steps:
1. Train each embedding × 3 CVs (apples-to-apples comparison), at native dimension or with the per-modality PCA, as the owner's arm says
2. Select the best embedding per category (expression, protein_seq, text, ppi, go, kg) ONCE per arm, on `SELECTION_CV` validation `val_<select_metric>_mean`; the same winners are reused for every CV and every variant (pooling the five folds' validation sets is not leakage-free; see `run_best_per_category_combo.sh`), and the two arms of a pair select the same ones
3. Train the multi-modal combo from the winners (concatenated, with the arm's PCA) × variants × 3 CVs
4. Print summary comparing single-embedding vs multi-modal

### Configuration Structure

All scripts source `slurm/config.conf` (shared environment, training hyperparameters, embedding catalog), then their own `.conf` file for job-specific settings. Per-job `.conf` files can override any shared value if independent tuning is needed. All config files use the `.conf` extension by project convention even though they are plain bash files sourced directly by bash (the extension is cosmetic).

```
config.conf (shared by all jobs + login-node orchestrator)
├── Environment: TRANSFORMERS_NO_TF, module loads (SLURM-only)
├── Python: PYTHON_PATH
├── Run identity: RUN_TOKEN (kept when the orchestrator already exported one)
├── Directories: WORK_DIR, BASE_DIR, DATA_DIR, CACHE_DIR
├── Data files: ESM_PATH, GO_PATH, GAF_PATH, SL_PATH
├── Training: ENCODER_DIMS, MODEL_ARGS, EPOCHS, BATCH_SIZE, LR,
│             WEIGHT_DECAY, DROPOUT, EVAL_INTERVAL, PATIENCE,
│             NUM_FOLDS, POS_NEG_RATIO, SEED, WARMRESTART_T0/TMULT
├── Regularization: L1_LAMBDAS, PD_EPSILON
├── PCA arms: PCA_ARMS (the arms a run trains; unset: none joint per_modality both;
│             set but empty, blank or naming an unknown arm: refused),
│             PCA_ARM (one job's arm; default none), PCA_MODALITY_VARIANCE (0.8),
│             PCA_JOINT_VARIANCE (0.8), PCA_JOINT_DIM (empty; wins when set; one count
│             for joint and both, kept below the narrowest width it reduces: train.py
│             refuses one at or above it), the
│             pca_* helpers, and per job RESULTS_ROOT / STEP1_ROOT / CMP_ROOT /
│             PCA_MULTI_ARGS / PCA_SINGLE_ARGS (POST_PCA_VARIANCE / POST_PCA_DIM refused)
├── CV_TYPES[]
├── ALL_EMBEDDINGS[]      — type:filename:display_name
└── Compute-node header: cd, timestamp, PCA arm, GPU info (guarded by SLURM_JOB_ID)

run_generate_embeddings.conf
├── PRECOMPUTED_TYPES[]   — embedding types for generate_embeddings.py
│                           (esmc commented out: selection ranks the paper's 16)
└── OPTIONAL_TYPES[]      — accepted absences (esmc scprint seqvec genept); every
                            other EMB_CATALOG type must exist and be current

run_embedding_benchmark.conf (overrides only — shared defaults from config.conf)
└── (empty by default; uncomment to override shared training params)

run_best_per_category.conf (single-embedding runs + multi-modal combo)
├── PCA_VARIANCE           — results-directory tag only (BEST_CAT_PCA_VARIANCE); sets no PCA
│                            (the arm's PCA_SINGLE_ARGS does)
├── SELECTION_CV           — the one CV whose validation scores pick the winners (default cv1)
├── EMB_CATALOG[]         — type:filename:category (genept excluded)
├── BESTCAT_EXTRA_VARIANTS — 1 adds the 3 non-selection variants' single-embedding runs (default 0)
└── bestcat_n_tasks()     — the best-per-category array size for the current env
```

### Supported Embeddings

| Category | Type | Dim | Notes |
|----------|------|-----|-------|
| Expression | `geneformer` | 1152 | Geneformer-V2-316M token embeddings (HF ctheodoris/Geneformer root model, pinned revision) |
| Expression | `scgpt` | 512 | scGPT token embeddings; built from a hand-placed checkpoint folder, never downloaded (see below) |
| Expression | `gene2vec` | 200 | Gene2Vec co-expression skip-gram |
| Protein seq | `esm2` | 1280 | ESM-2 + Pool PaRTI |
| Protein seq | `prot_t5` | 1024 | ProtT5-XL |
| Protein seq | `esm1b` | 1280 | ESM-1b |
| Protein seq | `esmc` | 1152 | ESM Cambrian 600M (Synthyra/ESMplusplus_large), mean-pooled; optional, not generated by default (commented out of `PRECOMPUTED_TYPES` in `slurm/run_generate_embeddings.conf`; uncomment to rank it) |
| Protein seq | `scprint` | auto | scPRINT ESM2-derived gene IDs; optional, not built by default (needs `pip install scprint` + a manually placed checkpoint) |
| Protein seq | `seqvec` | 1024 | SeqVec ELMo-style; optional, not built by default (needs `pip install allennlp`) |
| Text | `genept` | 1536 | GenePT (DATA LEAKAGE with SL) |
| Text | `text_embed` | 1024 | mxbai-embed-large on NCBI descriptions |
| Text | `bioconceptvec` | 100 | BioConceptVec PubMed concepts |
| PPI network | `node2vec_ppi` | 128 | Node2Vec on STRING v12.0 |
| PPI network | `ppi_svd` | 256 | Truncated SVD of PPI adjacency |
| PPI network | `ppi_raw` | 1024 | Higher-dim SVD of PPI |
| PPI network | `mashup` | 500 | Mashup (Cho et al.): RWR + log(Q+1/n) + SVD |
| GO | `go` | 200 | anc2vec GO term embeddings |
| GO | `go2vec` | 128 | Node2Vec on GO DAG |
| GO | `onto2vec` | 128 | Word2Vec on GO axiom sentences |
| KG | `kg_complex` | 512 | ComplEx on STRING PPI + GO triples (real ⊕ imag) |

`generate_embeddings.py` auto-downloads the source data for every type and caches it in `data/embeddings_cache/`, with three exceptions. `scgpt` has no public direct download: copy the scGPT whole-human checkpoint folder (`best_model.pt` + `vocab.json`) to `data/embeddings_cache/scGPT_human/` (git-ignored, so no commit carries it — copy it into every checkout; e.g. `pip install gdown && gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O ../data/embeddings_cache/scGPT_human`). `scgpt` is a **required** catalog type, so without the folder the generation job logs `[scgpt] Skipped (not available)` and then fails its closing catalog check, after every other build; the repo-root orchestrator checks for the checkpoint before it submits anything, `slurm/submit_pipeline.sh` does not. The other two are packages `slurm/reset_env.sh` does not set up: `scprint` needs `pip install scprint` and a `scprint_checkpoint.ckpt` placed by hand in `data/embeddings_cache/`, and `seqvec` needs `pip install allennlp`. By default those two are therefore skipped, and esmc is not generated either, so the protein_seq winner is chosen from esm2 / esm1b / prot_t5, and the combo records the three (esmc, scprint, seqvec) in `best_per_category.json` `_selection_meta.unranked_not_generated`. The generation job deletes the cached `*_gene_embeddings.pkl` files at the top of `data/embeddings_cache/`, so each builder rebuilds them. A hand-placed file under another name (e.g. GenePert's `gene_to_geneformer_embed.pkl`) or in a `data/embeddings_cache/<type>/` subdirectory is kept and used only when a build fails; its `.pt` is then stamped non-current, and the job fails its closing `embedding_versions.py` check. Gene symbols from external sources are normalized (custom corrections + HGNC) and converted to NCBI Entrez Gene IDs. All `.pt` files use Entrez IDs as the canonical gene identifier.

### Monitor Jobs
```bash
squeue -u $USER                                        # Check job status
tail -f slurm/logs/siamese_gen_emb_*.out               # Watch Step 1 (generation)
tail -f slurm/logs/bench_*.out                         # Watch Step 2a (benchmark)
tail -f slurm/logs/bestcat_*.out                       # Watch Step 2b (best-per-category)
tail -f slurm/logs/bestcat_combo_pca_joint_*.out       # ... an arm other than none: <stage>_pca_<arm>_*
```

### Output
Paths below are arm `none`'s (`results/`); every other PCA arm writes the same names under `results/pca_<arm>/`, and the single-embedding runs (Step 2a, Step 2b's single-embedding runs) live in the root of the arm that owns them (`results/` for arms `none` / `joint`, `results/pca_per_modality/` for `per_modality` / `both` by default).

Step 2a results:
- `results/<embedding>_<cv>/results.json` — per-embedding per-CV metrics
- `results/embedding_benchmark_summary.json` — comparison table (`pca_arms` / `pca` / `pca_args`: the arms of this run that read the runs, one or both of the pair, and their PCA)

Step 2b results (`<prefix>` is empty for `pd:per_head` and `var_<kernel>_<cell_lines>__` for the other variants):
- `results/pcavar0.8_<embedding>_<cv>/results.json` — single-embedding results (native dim in `results/`, the per-modality PCA in `results/pca_per_modality/`; `pcavar0.8` is a name tag only); with `BESTCAT_EXTRA_VARIANTS=1` also `results/<prefix>pcavar0.8_<embedding>_<cv>/` for the other three variants (reporting only)
- `results/best_per_category.json` — winning embedding per category (selected once on `SELECTION_CV`, replicated under every CV key with that CV's score for information); `_selection_meta` records the arm (`pca_arm`, `pca`, `pca_args`) and the single-embedding runs it ranked (`step1_root`, `step1_pca`, `step1_pca_args`)
- `results/<prefix>best_cat_<combo>_<cv>/results.json` — multi-modal combo results; `fold_metrics[k]` carries each fold's `input_dim` (the encoder input width after the arm's PCA: in arm `none` the concatenated width) and `total_params`
- `results/<prefix>latest_combo_<cv>.txt` — this run's combo per variant and CV (what eval, the ablations and `sl_comparison` read); its text is the combo's path from `siamese_sl/`, `results[/pca_<arm>]/<prefix>best_cat_<combo>_<cv>`
- `run_token.txt` in every Step-2a benchmark (`<embedding>_<cv>`), Step-2b single-embedding (`<prefix>pcavar0.8_*`) and ablation output dir — the submission's `RUN_TOKEN`

Downstream results:
- `<cv3 combo>/eval_finetuning_<cv>/` — external screens; `config_used.json` records `checkpoint_sha256` and the fold ids used, which `sl_comparison/external_transfer.py` verifies. Per screen, besides the metric CSVs and the two CSVs the tables read (`<Screen>_predictions.csv`, `<Screen>_finetuned_predictions_<mode>.csv`):
  - `<Screen>_sl_probabilities.csv` — every screen pair × every head: `prob_<HEAD>_fold<k>` and `prob_<HEAD>_mean` (sigmoid of the zero-shot logit; the mean is over the folds' probabilities), plus `prob_ANY_*`; a pooled or single-output model has `prob_ANY_*` only
  - `<Screen>_finetuned_allpairs_<mode>.csv` — every screen pair with its `split` (`train` / `val` / `test` / `discarded`): `pred_fold<k>`, `pred_mean` on the scale the model was fitted to (`sign × raw score`, so `-Gamma` for Adamson) and `pred_outcome_fold<k>`, `pred_outcome_mean` in the screen's own sign and units. Only `test` rows are held out
  - `finetuned_models/<Screen>_<mode>_fold<k>.pt` — the fine-tuned weights, their affine head and the head they score through (`--no_save_finetuned` skips them)

  These three are optional extras: a failure in one is reported (the task exits 1) only after every table and metric has been written.
- `<cv3 combo>/pair_predictions/` — the all-gene-pair export (section 4 above): `sl_prob_<HEAD>.npy`, `finetuned_Adamson_<mode>_cv1.npy`, `genes.txt` / `genes.tsv`, `manifest.json`
- `results/<prefix>ablate_no_<category>_<cv>/`, `results/ablation_summary.csv` — leave-one-category-out ablation (the CSV's first column, `pca_arm`, names the arm)
- `results/<prefix>cellline_ablation_<cv>/results.json` and `../sl_comparison/results/<row>_nocell_<cv>.json` (`<row>` = `siamese_sl` or `siamese_sl_bilinear_per_head`; arm X: `../sl_comparison/results/pca_X/`) — cell-line ablation of the per_head combos
- `../sl_comparison/results[/pca_<arm>]/` — the arm's comparison rows, `comparison.csv`, `external_transfer/` and `paper_tables/` (repo-root orchestrator), and `../sl_comparison/results/pca_arms/` — the table across the arms (`tab_pca.csv`)

The eval array reads the screens `slurm/eval_finetuning.conf` `DATASETS` names, two of which are git-ignored, so no commit carries them: `external data/pair_sgRNA_phenotypes.txt` (Adamson) and `external data/mmc5.xlsx` (Gilbert). Copy them into the checkout, or skip that stage (`submit_pipeline.sh --skip-eval`); `eval_finetuning.py` refuses a missing one (`dataset_input_problems` → exit 1, its previous outputs untouched), but only once its task starts, after the combo has trained, and `sl_comparison/external_transfer.py` then drops every variant. The repo-root orchestrator checks these `DATASETS` files on the login node before submitting anything (`run_benchmarking_siamese_vs_others.sh`); `submit_pipeline.sh` does not.

## Files

```
siamese_sl/
├── siamese_esm.py              # Model architecture (Siamese, Kernel, Attention)
├── data_loader.py              # Unified embedding pipeline, CV splits
├── train.py                    # Training with cross-validation
├── predict.py                  # Prediction for gene pairs
├── eval_finetuning.py          # External-screen zero-shot + fine-tuning
├── export_pair_predictions.py  # SL probability of all gene pairs, per cell line
├── arch_signature.py           # Architecture/selection signature (ablation checks)
├── embedding_versions.py       # Embedding provenance check (stale/missing .pt)
├── cell_line_ablation/         # Post-hoc cell_emb=0 ablation + its SLURM job
├── gene_name_utils.py          # Gene ID utilities: symbol ↔ Entrez ID mapping + HGNC normalization
├── generate_all_genes_esm.py   # ESM embedding generation with Pool PaRTI
├── generate_go_esm_embeddings.py # anc2vec GO embeddings
├── generate_embeddings.py      # 18 embedding types (auto-download + generate)
├── gene_embeddings_reference.md # Reference guide for embedding methods
├── ARCHITECTURE.md             # Detailed model architecture + math
├── SETUP.md                    # Server environment setup notes
├── README.md
├── slurm/                      # SLURM job scripts
│   ├── config.conf             # Shared config (env, modules, paths, data files)
│   ├── reset_env.sh            # Recreate Python venv from scratch
│   ├── run_generate_embeddings.sh   # Step 1: download + generate all .pt files
│   ├── run_generate_embeddings.conf # ↳ embedding type lists
│   ├── run_embedding_benchmark.sh   # Step 2a: benchmark (native dim in arms none / joint, PCA within the modality in per_modality / both)
│   ├── run_embedding_benchmark.conf # ↳ per-job overrides only (empty by default)
│   ├── run_best_per_category.sh     # Step 2b: single-embedding runs (same PCA as Step 2a, in the owner arm's root)
│   ├── run_best_per_category_combo.sh # Step 2b: category winners + multi-modal combos
│   ├── run_best_per_category.conf   # ↳ selection CV, catalog, dir tag
│   ├── eval_finetuning.sh           # External screens: zero-shot + fine-tuning
│   ├── eval_finetuning.conf         # ↳ screens, heads, fine-tuning settings
│   ├── run_ablation.sh              # Leave-one-category-out ablation (one run per task)
│   ├── run_ablation_summary.sh      # ↳ <arm root>/ablation_summary.csv
│   ├── run_ablation.conf            # ↳ categories, retention
│   ├── export_pair_predictions.sh   # Last job: all-gene-pair export
│   ├── export_pair_predictions.conf # ↳ combos, gene universe, fine-tuned sets
│   └── logs/                   # Job output logs
└── results/                    # Training outputs: PCA arm none, and pca_<arm>/ per other arm
```

## Choosing the Genes

`generate_all_genes_esm.py` takes no user FASTA: it resolves each Entrez Gene
ID to the UniProt reviewed canonical sequence (`data/cache/human_proteome_reviewed.fasta`
+ `hgnc_complete_set.txt`, both tracked in git), falling back to NCBI RefSeq
(NP_ > YP_ > XP_, then longest) for the genes UniProt does not cover; the
accession used per gene is saved in the output as `protein_accessions`. Choose the genes with
`--gene_list` (a `.pt` file's `gene_order`, a `.txt` file with one Entrez ID per
line, or a two-column SL pairs file) or `--sl_path` (an SL pairs file).

## GPU Memory

ESM-2 (650M) requires ~4GB GPU memory for inference. If running out of memory:

```bash
# Reduce batch size
python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \
    --batch_size 4 --output out.pt

# Or use CPU (slower)
python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \
    --device cpu --batch_size 1 --output out.pt
```
