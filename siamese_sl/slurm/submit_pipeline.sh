#!/bin/bash
# ============================================================================
# Pipeline Orchestrator — run from the login node
# ============================================================================
# Submits the full Siamese SL training pipeline as SLURM jobs:
#
#   0. Environment reset          (default — recreates Python venv; skip with --skip-reset-env)
#   1. Embedding generation      (single job, sequential steps)
#   2. Embedding benchmark       (SLURM array job — one task per embedding×CV)
#   3. Benchmark summary         (single job, collects results after array)
#   4. Best-per-category Step 1  (SLURM array job — one task per embedding×CV)
#   5. Best-per-category combo   (single job — selects winners + trains combos)
#   6. External-screen eval + fine-tuning
#                                (single job — zero-shot + LP_last + Full FT
#                                 against Adamson/Corn/Gilbert on the CV3 combo;
#                                 skip with --skip-eval)
#
# Steps 2-3 and 4-5 run in parallel (independent output directories). Step 6
# chains after step 5 via afterok, reading the CV3 combo path from the marker
# file written by step 5.
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
#   # Full pipeline: env reset → generate → (benchmark + best-per-category)
#   ./slurm/submit_pipeline.sh
#
#   # Skip the env reset (use existing venv as-is)
#   ./slurm/submit_pipeline.sh --skip-reset-env
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
#   # Opt out of the chained external-screen eval (default: on when
#   # best-per-category runs, since the combo trained there is its target).
#   ./slurm/submit_pipeline.sh --skip-eval
#
#   # Eval-only: run eval_finetuning.sh against an existing trained combo.
#   # Skips env reset, generation, benchmark, and best-per-category.
#   ./slurm/submit_pipeline.sh --eval-only results/best_cat_..._cv3
#
#
# Environment variables:
#   BEST_CAT_PCA_VARIANCE Override PCA variance target for best-per-category (default 0.8)
#   EVAL_MODEL_DIR        Used by direct sbatch of eval_finetuning.sh (not by this script)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source shared config (works from login node — SLURM-specific parts guarded)
source "$SCRIPT_DIR/config.conf"

# ============================================================================
# PWD sanity check
# ============================================================================
# Must be invoked from siamese_sl/ so the submit-time PWD equals WORK_DIR.
# sbatch sets SLURM_SUBMIT_DIR to the caller's PWD; on the compute node,
# config.conf uses that as WORK_DIR and `cd`s to it. If the caller's PWD
# isn't siamese_sl/, the job lands in the wrong directory and fails to
# locate the Python scripts. Catch this early, before any job is queued.
if [ "$PWD" != "$WORK_DIR" ]; then
    echo "ERROR: submit_pipeline.sh must be run from the siamese_sl/ directory."
    echo "  Current PWD:  $PWD"
    echo "  Required:     $WORK_DIR"
    echo "  Fix:          cd \"$WORK_DIR\" and re-run."
    exit 1
fi

# ============================================================================
# Parse command-line arguments
# ============================================================================

RESET_ENV=true
SKIP_GENERATE=false
AFTER_JOB=""
RUN_BENCHMARK=true
RUN_BEST_CAT=true
RUN_EVAL=true
EVAL_ONLY_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset-env)
            RESET_ENV=true; shift ;;       # retained for back-compat (now a no-op default)
        --skip-reset-env)
            RESET_ENV=false; shift ;;
        --skip-generate)
            SKIP_GENERATE=true; shift ;;
        --after)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --after requires a job ID argument"
                exit 1
            fi
            AFTER_JOB="$2"; shift 2 ;;
        --benchmark-only)
            RUN_BEST_CAT=false; RUN_EVAL=false; shift ;;
        --best-cat-only)
            RUN_BENCHMARK=false; shift ;;
        --skip-eval)
            RUN_EVAL=false; shift ;;
        --eval-only)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --eval-only requires a combo model directory argument"
                exit 1
            fi
            EVAL_ONLY_DIR="$2"; shift 2 ;;
        -h|--help)
            # Print the file-top doc block (all leading lines that start with '#').
            awk '/^#/{print} !/^#/{exit}' "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# Eval-only takes an independent short path (skips training, but still honors
# RESET_ENV so the eval job runs against a fresh venv unless --skip-reset-env).
if [ -n "$EVAL_ONLY_DIR" ]; then
    if [ "$SKIP_GENERATE" = true ] || [ -n "$AFTER_JOB" ] \
       || [ "$RUN_BENCHMARK" = false ] || [ "$RUN_BEST_CAT" = false ]; then
        echo "ERROR: --eval-only cannot be combined with training flags."
        echo "  --eval-only only submits eval_finetuning.sh against an existing combo."
        exit 1
    fi
    if [ ! -d "$EVAL_ONLY_DIR" ]; then
        echo "ERROR: --eval-only directory not found: $EVAL_ONLY_DIR"
        exit 1
    fi
    if [ ! -f "$EVAL_ONLY_DIR/config.json" ]; then
        echo "ERROR: $EVAL_ONLY_DIR/config.json missing — not a trained combo?"
        exit 1
    fi
    mkdir -p "$WORK_DIR/slurm/logs"

    EVAL_DEP=""
    if [ "$RESET_ENV" = true ]; then
        echo "--- Submitting environment reset ---"
        ENV_JOB=$(sbatch --parsable "$SCRIPT_DIR/reset_env.sh")
        echo "  Reset env job: $ENV_JOB"
        EVAL_DEP="--dependency=afterok:$ENV_JOB"
    fi

    echo "--- Submitting eval_finetuning (target: $EVAL_ONLY_DIR) ---"
    # Export EVAL_MODEL_DIR normally; `--export=ALL` propagates it into the
    # job. This handles special characters in the path safely, unlike the
    # `--export=ALL,KEY=value` form which is bash-word-split on whitespace.
    export EVAL_MODEL_DIR="$EVAL_ONLY_DIR"
    EVAL_JOB=$(sbatch --parsable \
        $EVAL_DEP \
        --export=ALL \
        "$SCRIPT_DIR/eval_finetuning.sh")
    echo "  Eval job: $EVAL_JOB"
    echo "  Tail:   tail -f $WORK_DIR/slurm/logs/eval_finetuning_${EVAL_JOB}.out"
    exit 0
fi

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
    # SLURM `afterok:A:B` waits on both A and B — chain env reset too if set.
    if [ -n "$ENV_DEP" ]; then
        GEN_DEP="--dependency=afterok:$ENV_JOB:$AFTER_JOB"
    else
        GEN_DEP="--dependency=afterok:$AFTER_JOB"
    fi
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
    # Source benchmark config for any overrides (shared defaults from config.conf)
    source "$SCRIPT_DIR/run_embedding_benchmark.conf"

    NUM_EMB=${#ALL_EMBEDDINGS[@]}
    NUM_CVS=${#CV_TYPES[@]}
    BENCH_TASKS=$((NUM_EMB * NUM_CVS))

    if [ "$BENCH_TASKS" -lt 1 ]; then
        echo "WARNING: No embeddings or CV types defined in config.conf"
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
# Step 4: External-screen evaluation + fine-tuning (chained after combo)
# ============================================================================
# Targets the CV3 combo trained in Step 3b. The combo script writes
# `results/latest_combo_cv3.txt` on success, which eval_finetuning.conf
# reads to resolve MODEL_DIR automatically — no need to hard-code the
# combo's name (which depends on which embeddings win per category).
# `afterok` (not afterany) so eval only runs if the combo succeeded.
# Skipped when RUN_BEST_CAT is false (nothing fresh to evaluate) or when
# the user passes --skip-eval.

if [ "$RUN_EVAL" = true ] && [ "$RUN_BEST_CAT" = true ]; then
    echo "--- Submitting external-screen eval + fine-tuning ---"
    echo "  Target combo: resolved at job-start from results/latest_combo_cv3.txt"
    echo "  Datasets:     Adamson, Corn, Gilbert (see eval_finetuning.conf)"
    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:$COMBO_JOB \
        --export=ALL \
        "$SCRIPT_DIR/eval_finetuning.sh")
    echo "  Eval job:     $EVAL_JOB"
    echo "  Tail:         tail -f slurm/logs/eval_finetuning_${EVAL_JOB}.out"
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
