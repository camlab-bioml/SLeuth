# SLURM Pipeline for Siamese SL Training

## Quick Start

```bash
cd siamese_sl

# Full pipeline: env reset → generate → (benchmark ∥ best-per-category) → combo → external-screen eval
./slurm/submit_pipeline.sh

# Skip generation (embeddings already exist)
./slurm/submit_pipeline.sh --skip-generate

# Skip the chained external-screen eval + fine-tuning
./slurm/submit_pipeline.sh --skip-eval

# Run eval only against an existing combo (no training)
./slurm/submit_pipeline.sh --eval-only results/best_cat_..._cv3

# Monitor jobs
squeue -u $(whoami)
tail -f slurm/logs/bench_<JOB_ID>_<TASK_ID>.out
```

## Pipeline Overview

The pipeline has four stages. Stages 2 and 3 run **in parallel** (independent
output directories). Stage 4 chains after Stage 3 on success. Each training
stage uses **SLURM array jobs** so every (embedding, CV) combination runs as a
separate task on its own GPU.

```
Stage 1: Embedding Generation (single job)
    │
    ├──► Stage 2: Embedding Benchmark (array job → summary job)
    │       Each task: one embedding × one CV at native dimensions
    │       Output: results/<type>_<cv>/results.json
    │
    └──► Stage 3: Best-per-Category (array job → combo job)
            Step 1 (array): one embedding × one CV with PCA
            Step 2 (combo): select best per category
            Step 3 (combo): train multi-modal combinations
                            → writes results/latest_combo_${CV}.txt on success
            Output: results/pcavar<variance>_<type>_<cv>/results.json
                    results/best_cat_<combo>_<cv>/results.json
               │
               └──► Stage 4: External-screen eval + fine-tuning (chained afterok)
                     Zero-shot (closed-form OLS) + LP_last + Full FT against
                     Adamson, Corn, Gilbert CRISPR screens on the CV3 combo.
                     MODEL_DIR auto-resolved from results/latest_combo_cv3.txt.
                     Output: ${MODEL_DIR}/eval_finetuning/*
                     Skip with --skip-eval.
```

Array sizes are **computed dynamically** from the `.conf` files. Adding or
removing embeddings from `ALL_EMBEDDINGS[]` or `EMB_CATALOG[]` automatically
adjusts the number of SLURM tasks — no script changes needed.

## File Reference

### Orchestrator

| File | Description |
|------|-------------|
| `submit_pipeline.sh` | **Login-node script.** Parses `.conf` files to compute array sizes, then submits all SLURM jobs with correct dependencies. Supports `--skip-reset-env`, `--skip-generate`, `--after <JOB_ID>`, `--benchmark-only`, `--best-cat-only`, `--skip-eval`, and `--eval-only <combo_dir>`. Set `MAX_CONCURRENT` env var to limit parallel GPU tasks (default: 4). |

### Shared Configuration

| File | Description |
|------|-------------|
| `config.conf` | **Sourced by every script.** Sets Python path, project directories (`WORK_DIR`, `DATA_DIR`, etc.), key data file paths, shared training hyperparameters (architecture, optimizer, regularization), CV types, the full embedding catalog (`ALL_EMBEDDINGS[]`), and compute-node environment (module loads, GPU info). Works from both login nodes and compute nodes — SLURM-specific setup is guarded by `$SLURM_JOB_ID`. |

#### Negatives & cell-line mode (in `config.conf`)

- `NEG_PATH` — experimentally screened non-SL pairs as the negative pool (default set; `""` reverts to random non-edge sampling).
- `USE_CELL_LINES` (`0`/`1`, **default `1`**) + `CELL_LINE_DIM` (default `8`) — **masked multi-label cell-line conditioning**, on by default. Each gene pair gets a per-head (label, mask) over `[K562, JURKAT, A549, HELA, A375, 293T, PC9, OTHER]`; the model (`SiameseSLMultiCell`) predicts SL per head (selection metric: macro per-head AUPR). Requires `NEG_PATH` and the `.sources.tsv` sidecars (produced by Stage 1). These resolve to `$CELL_LINE_ARGS`, threaded into all three training run-scripts. **The Stage 4 external-screen eval consumes the multi-head checkpoint PER cell-line head** — `PER_HEAD` is auto-enabled in `eval_finetuning.conf` when `USE_CELL_LINES=1`, scoring each screen against its cell-line head (Adamson/Gilbert → K562, Corn → OTHER). Fall back to the single-output SL benchmark with `USE_CELL_LINES=0 ./slurm/submit_pipeline.sh`.

### Stage 1: Embedding Generation

| File | Description |
|------|-------------|
| `run_generate_embeddings.sh` | **SLURM job.** Downloads source data and generates all `.pt` embedding files: SynLethDB SL pairs, ESM-2 (GPU-heavy), GO anc2vec, and all other types via `generate_embeddings.py`. Skips already-existing files. |
| `run_generate_embeddings.conf` | **Config.** Lists embedding types for `generate_embeddings.py` (`PRECOMPUTED_TYPES[]`). The full catalog (`ALL_EMBEDDINGS[]`) comes from `config.conf`. |

### Stage 2: Embedding Benchmark

| File | Description |
|------|-------------|
| `run_embedding_benchmark.sh` | **SLURM array worker.** Each array task trains the siamese model on one (embedding, CV) combination at native dimensions (no PCA). Maps `SLURM_ARRAY_TASK_ID` to `(embedding_index, cv_index)` using `ALL_EMBEDDINGS[]` and `CV_TYPES[]`. Skips gracefully if the embedding file is missing. Resume-safe: skips if `results.json` already exists. |
| `run_embedding_benchmark_summary.sh` | **SLURM job** (runs after array). Collects `results.json` from all benchmark directories, prints an AUROC summary table, and saves `results/embedding_benchmark_summary.json`. |
| `run_embedding_benchmark.conf` | **Config.** Empty by default (shared training hyperparameters, CV types, and `ALL_EMBEDDINGS[]` come from `config.conf`). Uncomment overrides here to tune the benchmark independently. |

### Stage 3: Best-per-Category

| File | Description |
|------|-------------|
| `run_best_per_category.sh` | **SLURM array worker.** Each array task trains the siamese model on one (embedding, CV) combination with per-modality PCA (`PCA_VARIANCE` variance target, `PCA_METHOD` = robust or plain). Preprocessing is fit per-fold on `PREPROCESSING_FIT_SCOPE` rows (default `train`, leak-free). Maps `SLURM_ARRAY_TASK_ID` to `(embedding_index, cv_index)` using `EMB_CATALOG[]` and `CV_TYPES[]`. Resume-safe: skips if `results.json` already exists. |
| `run_best_per_category_combo.sh` | **SLURM job** (runs after array). Step 2: selects the best AUROC embedding per biological category per CV. Step 3: trains multi-modal combos by concatenating winners with per-modality PCA + post-concat PCA. Step 4: prints summary table. |
| `run_best_per_category.conf` | **Config.** Job-specific settings: `PCA_VARIANCE` (overridable via `BEST_CAT_PCA_VARIANCE` env var), `PCA_METHOD` (robust ∣ plain), `PREPROCESSING_FIT_SCOPE` (train ∣ all), and `EMB_CATALOG[]` (type:file:category, genept excluded for data leakage). Shared training hyperparameters come from `config.conf`; uncomment overrides here to tune independently. |

### Stage 4: External-Screen Evaluation + Fine-tuning

| File | Description |
|------|-------------|
| `eval_finetuning.sh` | **SLURM job** (chained after `run_best_per_category_combo.sh` via `afterok`). Runs `eval_finetuning.py` against the CV3 combo: zero-shot (closed-form OLS of an `(a, c)` affine head) + `LP_last` and `Full` SGD fine-tuning modes against Adamson, Corn, Gilbert external CRISPR screens. Per-fold metrics and a 5-fold ensemble. Skippable via `--skip-eval` in the orchestrator. |
| `eval_finetuning.conf` | **Config.** `MODEL_DIR` resolution order: `EVAL_MODEL_DIR` env var → `results/latest_combo_cv3.txt` marker (written by combo script) → hardcoded fallback. Plus `EMBEDDINGS_PATHS[]` (must match combo's modalities), `DATASETS[]`, `FT_MODES[]`, and fine-tuning hyperparameters (`FT_EPOCHS`, `FT_LR`, `TRAIN_FRAC`, `SPLIT_SEED`). |

### Utilities

| File | Description |
|------|-------------|
| `reset_env.sh` | **SLURM job.** Recreates the Python venv from scratch on a GPU node using pip. Installs all dependencies (PyTorch, ESM, robpy, etc.) and verifies imports. Must run on a GPU node. |

### Output Directories

| Directory | Source |
|-----------|--------|
| `slurm/logs/` | SLURM stdout/stderr. Array jobs: `bench_<JOB>_<TASK>.out`, `bestcat_<JOB>_<TASK>.out`. Summary/combo: `bench_summary_<JOB>.out`, `bestcat_combo_<JOB>.out`. |
| `results/<type>_<cv>/` | Benchmark results (native dims) |
| `results/pcavar<variance>_<type>_<cv>/` | Best-per-category Step 1 results (with PCA) |
| `results/best_cat_<combo>_<cv>/` | Multi-modal combo results |
| `results/embedding_benchmark_summary.json` | Benchmark summary JSON |
| `results/best_per_category.json` | Category winner selections |

## How Array Task Mapping Works

Both array workers use the same formula to map `SLURM_ARRAY_TASK_ID` to an
(embedding, CV) pair:

```
embedding_index = SLURM_ARRAY_TASK_ID / num_cv_types
cv_index        = SLURM_ARRAY_TASK_ID % num_cv_types
```

For 19 embeddings and 3 CV types, task IDs 0-56 map to:
- Task 0: embedding[0] / cv1
- Task 1: embedding[0] / cv2
- Task 2: embedding[0] / cv3
- Task 3: embedding[1] / cv1
- ...
- Task 56: embedding[18] / cv3

The submit script computes the array range dynamically:
`--array=0-$((NUM_EMB * NUM_CVS - 1))%MAX_CONCURRENT`

## Customization

**Add a new embedding:** Add an entry to `ALL_EMBEDDINGS[]` (benchmark) or
`EMB_CATALOG[]` (best-per-category) in the `.conf` file. The array size
adjusts automatically on the next `submit_pipeline.sh` run.

**Change PCA variance target:** Either edit `PCA_VARIANCE` in `run_best_per_category.conf`
or override at submit time:
```bash
BEST_CAT_PCA_VARIANCE=0.9 ./slurm/submit_pipeline.sh --best-cat-only
```

**Limit GPU concurrency:** Default is 4 concurrent tasks. Adjust with:
```bash
MAX_CONCURRENT=8 ./slurm/submit_pipeline.sh
```

**Change SLURM partition/node:** Edit the `#SBATCH` headers in each `.sh`
file, or pass overrides on the command line:
```bash
# In submit_pipeline.sh, the sbatch calls can accept extra flags
sbatch --partition=other_partition --array=0-56%4 slurm/run_embedding_benchmark.sh
```
