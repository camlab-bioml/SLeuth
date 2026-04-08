# SLURM Pipeline for Siamese SL Training

## Quick Start

```bash
cd siamese_sl

# Full pipeline: generate embeddings → benchmark + best-per-category → summaries
./slurm/submit_pipeline.sh

# Skip generation (embeddings already exist)
./slurm/submit_pipeline.sh --skip-generate

# Monitor jobs
squeue -u $(whoami)
tail -f slurm/logs/bench_<JOB_ID>_<TASK_ID>.out
```

## Pipeline Overview

The pipeline has three stages. Stages 2 and 3 run **in parallel** (independent
output directories). Each training stage uses **SLURM array jobs** so that
every (embedding, CV) combination runs as a separate task on its own GPU.

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
            Output: results/pcavar<variance>_<type>_<cv>/results.json
                    results/best_cat_<combo>_<cv>/results.json
```

Array sizes are **computed dynamically** from the `.conf` files. Adding or
removing embeddings from `ALL_EMBEDDINGS[]` or `EMB_CATALOG[]` automatically
adjusts the number of SLURM tasks — no script changes needed.

## File Reference

### Orchestrator

| File | Description |
|------|-------------|
| `submit_pipeline.sh` | **Login-node script.** Parses `.conf` files to compute array sizes, then submits all SLURM jobs with correct dependencies. Supports `--skip-generate`, `--after <JOB_ID>`, `--benchmark-only`, `--best-cat-only`. Set `MAX_CONCURRENT` env var to limit parallel GPU tasks (default: 4). |

### Shared Configuration

| File | Description |
|------|-------------|
| `config.sh` | **Sourced by every script.** Sets Python path, project directories (`WORK_DIR`, `DATA_DIR`, etc.), key data file paths, shared training hyperparameters (architecture, optimizer, regularization), CV types, the full embedding catalog (`ALL_EMBEDDINGS[]`), and compute-node environment (module loads, GPU info). Works from both login nodes and compute nodes — SLURM-specific setup is guarded by `$SLURM_JOB_ID`. |

### Stage 1: Embedding Generation

| File | Description |
|------|-------------|
| `run_generate_embeddings.sh` | **SLURM job.** Downloads source data and generates all `.pt` embedding files: SynLethDB SL pairs, ESM-2 (GPU-heavy), GO anc2vec, and all other types via `generate_embeddings.py`. Skips already-existing files. |
| `run_generate_embeddings.conf` | **Config.** Lists embedding types for `generate_embeddings.py` (`PRECOMPUTED_TYPES[]`). The full catalog (`ALL_EMBEDDINGS[]`) comes from `config.sh`. |

### Stage 2: Embedding Benchmark

| File | Description |
|------|-------------|
| `run_embedding_benchmark.sh` | **SLURM array worker.** Each array task trains the siamese model on one (embedding, CV) combination at native dimensions (no PCA). Maps `SLURM_ARRAY_TASK_ID` to `(embedding_index, cv_index)` using `ALL_EMBEDDINGS[]` and `CV_TYPES[]`. Skips gracefully if the embedding file is missing. Resume-safe: skips if `results.json` already exists. |
| `run_embedding_benchmark_summary.sh` | **SLURM job** (runs after array). Collects `results.json` from all benchmark directories, prints an AUROC summary table, and saves `results/embedding_benchmark_summary.json`. |
| `run_embedding_benchmark.conf` | **Config.** Empty by default (shared training hyperparameters, CV types, and `ALL_EMBEDDINGS[]` come from `config.sh`). Uncomment overrides here to tune the benchmark independently. |

### Stage 3: Best-per-Category

| File | Description |
|------|-------------|
| `run_best_per_category.sh` | **SLURM array worker.** Each array task trains the siamese model on one (embedding, CV) combination with robust PCA variance-based reduction (`PCA_VARIANCE`). Maps `SLURM_ARRAY_TASK_ID` to `(embedding_index, cv_index)` using `EMB_CATALOG[]` and `CV_TYPES[]`. Resume-safe: skips if `results.json` already exists. |
| `run_best_per_category_combo.sh` | **SLURM job** (runs after array). Step 2: selects the best AUROC embedding per biological category per CV. Step 3: trains multi-modal combos by concatenating winners with per-modality PCA. Step 4: prints summary table. |
| `run_best_per_category.conf` | **Config.** Job-specific settings: `PCA_VARIANCE` (overridable via `BEST_CAT_PCA_VARIANCE` env var) and `EMB_CATALOG[]` (type:file:category, genept excluded for data leakage). Shared training hyperparameters come from `config.sh`; uncomment overrides here to tune independently. |

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
