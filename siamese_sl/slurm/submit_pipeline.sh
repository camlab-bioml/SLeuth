#!/bin/bash
# ============================================================================
# Pipeline Orchestrator — run from the login node
# ============================================================================
# Submits the full Siamese SL training pipeline as SLURM jobs:
#
#   0. Environment reset          (optional — recreates Python venv)
#   1. Embedding generation      (single job, sequential steps)
#   2. Embedding benchmark       (SLURM array job — one task per embedding×CV)
#   3. Benchmark summary         (single job, collects results after array)
#   4. Best-per-category Step 1  (SLURM array job — one task per embedding×CV)
#   5. Best-per-category combo   (single job — selects winners + trains combos)
#
# Steps 2-3 and 4-5 run in parallel (independent output directories).
# Array sizes are computed dynamically from the .conf files so adding or
# removing embeddings requires no script changes.
#
# Prerequisites:
#   - Must be run from siamese_sl/ (the project working directory)
#   - SLURM cluster must be accessible (sbatch available)
#   - slurm/logs/ directory is created automatically
#
# Usage:
#   cd siamese_sl
#
#   # Full pipeline: generate → (benchmark + best-per-category) → summaries
#   ./slurm/submit_pipeline.sh
#
#   # Full pipeline with env reset first
#   ./slurm/submit_pipeline.sh --reset-env
#
#   # Skip generation (embeddings already exist)
#   ./slurm/submit_pipeline.sh --skip-generate
#
#   # Chain after an existing generation job
#   ./slurm/submit_pipeline.sh --after <JOB_ID>
#
#   # Run only one of the two training pipelines
#   ./slurm/submit_pipeline.sh --benchmark-only
#   ./slurm/submit_pipeline.sh --best-cat-only
#
#
# Environment variables:
#   BEST_CAT_PCA_VARIANCE Override PCA variance target for best-per-category (default 0.8)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source shared config (works from login node — SLURM-specific parts guarded)
source "$SCRIPT_DIR/config.sh"

# ============================================================================
# Parse command-line arguments
# ============================================================================

RESET_ENV=false
SKIP_GENERATE=false
AFTER_JOB=""
RUN_BENCHMARK=true
RUN_BEST_CAT=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset-env)
            RESET_ENV=true; shift ;;
        --skip-generate)
            SKIP_GENERATE=true; shift ;;
        --after)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --after requires a job ID argument"
                exit 1
            fi
            AFTER_JOB="$2"; shift 2 ;;
        --benchmark-only)
            RUN_BEST_CAT=false; shift ;;
        --best-cat-only)
            RUN_BENCHMARK=false; shift ;;
        -h|--help)
            head -45 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# Reject conflicting flags
if [ "$RUN_BENCHMARK" = false ] && [ "$RUN_BEST_CAT" = false ]; then
    echo "ERROR: --benchmark-only and --best-cat-only cannot be used together."
    exit 1
fi
if [ "$SKIP_GENERATE" = true ] && [ -n "$AFTER_JOB" ]; then
    echo "ERROR: --skip-generate and --after are mutually exclusive."
    echo "  --skip-generate assumes embeddings already exist (no dependency)."
    echo "  --after chains training after an existing job (implies generation is running)."
    exit 1
fi


# ============================================================================
# Ensure log directory exists
# ============================================================================
mkdir -p "$WORK_DIR/slurm/logs"

echo "============================================================================"
echo "  Siamese SL Pipeline — Job Submission"
echo "============================================================================"
echo "  Working dir:    $WORK_DIR"
echo "  Data dir:       $DATA_DIR"
echo ""

# ============================================================================
# Step 0: Environment reset (optional)
# ============================================================================
# Recreates the Python venv from scratch on a GPU node.

ENV_DEP=""

if [ "$RESET_ENV" = true ]; then
    echo "--- Submitting environment reset ---"
    ENV_JOB=$(sbatch --parsable "$SCRIPT_DIR/reset_env.sh")
    echo "  Reset env job: $ENV_JOB"
    ENV_DEP="--dependency=afterok:$ENV_JOB"
    echo ""
fi

# ============================================================================
# Step 1: Embedding generation (optional)
# ============================================================================
# Produces all .pt embedding files in DATA_DIR. Must complete before any
# training jobs start.

GEN_DEP=""

if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    echo "--- Submitting embedding generation ---"
    GEN_JOB=$(sbatch --parsable $ENV_DEP "$SCRIPT_DIR/run_generate_embeddings.sh")
    echo "  Generation job: $GEN_JOB"
    GEN_DEP="--dependency=afterok:$GEN_JOB"
elif [ -n "$AFTER_JOB" ]; then
    echo "--- Chaining after existing job $AFTER_JOB ---"
    GEN_DEP="--dependency=afterok:$AFTER_JOB"
else
    echo "--- Skipping generation (--skip-generate) ---"
    # If env reset was requested, training jobs still depend on it
    GEN_DEP="$ENV_DEP"
fi
echo ""

# ============================================================================
# Step 2: Embedding benchmark (array job + summary)
# ============================================================================
# Trains each embedding at native dimensions × all CV types.
# Array size = len(ALL_EMBEDDINGS) × len(CV_TYPES), computed from the .conf.

if [ "$RUN_BENCHMARK" = true ]; then
    # Source benchmark config for any overrides (shared defaults from config.sh)
    source "$SCRIPT_DIR/run_embedding_benchmark.conf"

    NUM_EMB=${#ALL_EMBEDDINGS[@]}
    NUM_CVS=${#CV_TYPES[@]}
    BENCH_TASKS=$((NUM_EMB * NUM_CVS))

    if [ "$BENCH_TASKS" -lt 1 ]; then
        echo "WARNING: No embeddings or CV types defined in config.sh"
        echo "  Skipping benchmark."
        RUN_BENCHMARK=false
    fi
fi

if [ "$RUN_BENCHMARK" = true ]; then
    echo "--- Submitting embedding benchmark ---"
    echo "  $NUM_EMB embeddings x $NUM_CVS CVs = $BENCH_TASKS tasks"

    # Submit array job: each task trains one (embedding, CV) combination
    BENCH_JOB=$(sbatch --parsable \
        $GEN_DEP \
        --array=0-$((BENCH_TASKS - 1)) \
        "$SCRIPT_DIR/run_embedding_benchmark.sh")
    echo "  Benchmark array job: $BENCH_JOB"

    # Submit summary after all array tasks finish (afterany = run even if some failed)
    BENCH_SUMM=$(sbatch --parsable \
        --dependency=afterany:$BENCH_JOB \
        "$SCRIPT_DIR/run_embedding_benchmark_summary.sh")
    echo "  Benchmark summary job: $BENCH_SUMM"
    echo ""
fi

# ============================================================================
# Step 3: Best-per-category (array job + combo)
# ============================================================================
# Step 3a: Train each embedding with PCA × all CV types (array job).
# Step 3b: Select winners + train multi-modal combos (single job after array).
# Array size = len(EMB_CATALOG) × len(CV_TYPES), computed from the .conf.

if [ "$RUN_BEST_CAT" = true ]; then
    # Source best-per-category config for PCA_VARIANCE and EMB_CATALOG.
    source "$SCRIPT_DIR/run_best_per_category.conf"

    NUM_EMB=${#EMB_CATALOG[@]}
    NUM_CVS=${#CV_TYPES[@]}
    BESTCAT_TASKS=$((NUM_EMB * NUM_CVS))

    if [ "$BESTCAT_TASKS" -lt 1 ]; then
        echo "WARNING: No embeddings or CV types defined in run_best_per_category.conf"
        echo "  Skipping best-per-category."
        RUN_BEST_CAT=false
    fi
fi

if [ "$RUN_BEST_CAT" = true ]; then
    echo "--- Submitting best-per-category ---"
    echo "  $NUM_EMB embeddings x $NUM_CVS CVs = $BESTCAT_TASKS tasks"
    echo "  PCA variance: $PCA_VARIANCE"

    # Step 3a: array job for single-embedding PCA training
    BESTCAT_JOB=$(sbatch --parsable \
        $GEN_DEP \
        --array=0-$((BESTCAT_TASKS - 1)) \
        "$SCRIPT_DIR/run_best_per_category.sh")
    echo "  Best-cat array job: $BESTCAT_JOB"

    # Step 3b: combo job after all array tasks finish
    COMBO_JOB=$(sbatch --parsable \
        --dependency=afterany:$BESTCAT_JOB \
        "$SCRIPT_DIR/run_best_per_category_combo.sh")
    echo "  Best-cat combo job: $COMBO_JOB"
    echo ""
fi

# ============================================================================
# Done
# ============================================================================
echo "============================================================================"
echo "All jobs submitted. Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f slurm/logs/<job_file>.out"
echo "============================================================================"
