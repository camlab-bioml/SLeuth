#!/bin/bash
# ============================================================================
# run_benchmarking_siamese_vs_others.sh — ROOT pipeline orchestrator
# ============================================================================
# Submits the FULL four-model SL benchmark on the SLDB 3.0 standard as one
# SLURM DAG, then prints the comparison. Run from the login node (repo root);
# it submits jobs with `sbatch --parsable` + `--dependency`, exactly like
# siamese_sl/slurm/submit_pipeline.sh (whose conventions this follows).
#
#   reset_env ──┐
#               └─ generate_embeddings ──┬─ siamese best-per-category ─ combo ─┐
#                                        │        (its OWN embedding, cell-line) │
#                                        └─ shared 3-model job (SLGNN/NSF4SL/  ──┼─ final
#                                             SLMGAE on the shared folds)        │  compare
#                                                                                ▼
#   All four scored on AUPRG / normalized AUPR (prevalence-independent).
#
# The three shared-fold models use the ONE canonical fold set
# (sl_comparison/folds, generated inside their job); siamese runs its own
# pipeline (embedding selection + cell-line) and its OR-collapse (*_any)
# metrics are auto-discovered into the final table.
#
# Prerequisites: run from the repo root, on a node with sbatch. GPU jobs land on
# gpu3 (the cluster runs on GPU).
#
# Generation is AUTO-SKIPPED when the embeddings already exist on disk
# (data/all_genes_kg_complex.pt present) — so a plain resubmit after a cancel
# runs everything without re-triggering the flaky ESM/NCBI/CUDA generation step.
# reset_env still runs and gates the DAG. Pass --force-generate to regenerate.
#
# Usage:
#   ./run_benchmarking_siamese_vs_others.sh                 # full pipeline (auto-skips gen if embeddings exist)
#   ./run_benchmarking_siamese_vs_others.sh --skip-reset-env
#   ./run_benchmarking_siamese_vs_others.sh --skip-generate # force-skip generation
#   ./run_benchmarking_siamese_vs_others.sh --force-generate # force regeneration even if embeddings exist
#   ./run_benchmarking_siamese_vs_others.sh --after <JOB>   # chain after a running gen job
#   ./run_benchmarking_siamese_vs_others.sh --skip-siamese  # only the 3 shared models
#   ./run_benchmarking_siamese_vs_others.sh --skip-shared   # only siamese
#
# Env: MAX_CONCURRENT (default 4), SEED (default 42), EMB (shared-model feature
# file, default ../data/all_genes_kg_complex.pt).
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SIAMESE_DIR="$ROOT/siamese_sl"
SLURM_DIR="$SIAMESE_DIR/slurm"

if [ ! -f "$SLURM_DIR/config.conf" ]; then
    echo "ERROR: $SLURM_DIR/config.conf not found — run from the repo root."
    exit 1
fi
# Sourced only for PYTHON_PATH / CV_TYPES / MAX_CONCURRENT. On a login node the
# compute-node header inside config.conf is guarded by SLURM_JOB_ID (no-op here).
source "$SLURM_DIR/config.conf"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
export SEED="${SEED:-42}"          # same seed everywhere -> fair division

# ---- flags -----------------------------------------------------------------
RESET_ENV=true; SKIP_GENERATE=false; FORCE_GENERATE=false; AFTER_JOB=""; RUN_SIAMESE=true; RUN_SHARED=true
# The ablation and the external-screen eval hang off the siamese combo. They
# lived only in siamese_sl/slurm/submit_pipeline.sh until 2026-08-24, which
# meant a run driven from THIS orchestrator silently produced neither.
RUN_ABLATION=true; RUN_EVAL=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-reset-env) RESET_ENV=false; shift ;;
        --skip-generate)  SKIP_GENERATE=true; shift ;;
        --force-generate) FORCE_GENERATE=true; shift ;;
        --after) [ -z "${2:-}" ] && { echo "ERROR: --after needs a job ID"; exit 1; }
                 AFTER_JOB="$2"; shift 2 ;;
        --skip-siamese)   RUN_SIAMESE=false; shift ;;
        --skip-shared)    RUN_SHARED=false; shift ;;
        --skip-ablation)  RUN_ABLATION=false; shift ;;
        --skip-eval)      RUN_EVAL=false; shift ;;
        -h|--help) awk '/^#/{print} !/^#/{exit}' "$0" | tail -n +2 | sed 's/^# \?//'; exit 0 ;;
        *) echo "ERROR: unknown option: $1 (see --help)"; exit 1 ;;
    esac
done
if [ "$RUN_SIAMESE" = false ] && [ "$RUN_SHARED" = false ]; then
    echo "ERROR: --skip-siamese and --skip-shared together leave nothing to run."; exit 1
fi
if [ "$SKIP_GENERATE" = true ] && [ -n "$AFTER_JOB" ]; then
    echo "ERROR: --skip-generate and --after are mutually exclusive."; exit 1
fi
if [ "$SKIP_GENERATE" = true ] && [ "$FORCE_GENERATE" = true ]; then
    echo "ERROR: --skip-generate and --force-generate are mutually exclusive."; exit 1
fi
if [ "$FORCE_GENERATE" = true ] && [ -n "$AFTER_JOB" ]; then
    echo "ERROR: --force-generate and --after are mutually exclusive."; exit 1
fi

# Auto-skip generation when the embeddings are already on disk, so a resubmit
# after a cancel never re-triggers the flaky ESM/NCBI/CUDA generation step
# (its failure would cascade DependencyNeverSatisfied over the whole DAG).
# kg_complex is the shared-model feature file and a reliable "generation done"
# marker; --force-generate overrides. The check runs where the DAG is submitted
# (cluster login node), which shares the compute nodes' filesystem.
EMB_MARKER="$ROOT/data/all_genes_kg_complex.pt"
if [ "$SKIP_GENERATE" = false ] && [ "$FORCE_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    if [ -f "$EMB_MARKER" ]; then
        N_EMB=$(ls "$ROOT"/data/all_genes_*.pt 2>/dev/null | wc -l | tr -d ' ')
        # EXISTENCE IS NOT ENOUGH. This guard used to skip generation whenever
        # the file was present, so an embedding set built over a DIFFERENT gene
        # universe was silently reused — which is exactly how the benchmark ran
        # on 5,257 genes instead of the full universe, giving siamese a
        # different test-set prevalence from its peers with nothing in the log
        # saying so. Verify the stored gene_order matches gene_universe.txt.
        UNIV="$ROOT/data/gene_universe.txt"
        EMB_OK=$("$PYTHON_PATH" - "$EMB_MARKER" "$UNIV" <<'PYEOF'
import sys
try:
    import torch
    d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    go = d.get('gene_order') if isinstance(d, dict) else None
    if go is None:
        print("no gene_order stored"); raise SystemExit(0)
    want = {l.strip() for l in open(sys.argv[2]) if l.strip()}
    have = set(map(str, go))
    # SET comparison, not ordered. The .pt files store gene_order sorted
    # LEXICOGRAPHICALLY ('1','10','100') while gene_universe.txt is sorted
    # NUMERICALLY ('1','2','9'); data_loader builds gene_to_idx by lookup, so
    # the order is irrelevant. Comparing ordered lists here would report every
    # healthy embedding set as stale and burn a full ESM regeneration on
    # every run.
    if have != want:
        miss, extra = len(want - have), len(have - want)
        print(f"gene universe mismatch: {len(have)} genes stored vs "
              f"{len(want)} expected ({miss} missing, {extra} unexpected)")
        raise SystemExit(0)
    print("")           # empty == matches
except SystemExit:
    raise
except Exception as e:
    print(f"could not verify ({type(e).__name__}: {e})")
PYEOF
) || EMB_OK="verification failed"
        if [ -n "$EMB_OK" ]; then
            echo "--- embeddings present but STALE: $EMB_OK ---"
            echo "    regenerating over the current gene universe"
        else
            echo "--- embeddings already present ($N_EMB all_genes_*.pt, incl kg_complex,"
            echo "    gene_order matches gene_universe.txt) -> auto-skipping generation ---"
            echo "    (pass --force-generate to regenerate anyway)"
            SKIP_GENERATE=true
        fi
    else
        echo "--- no embeddings found ($EMB_MARKER missing) -> generation WILL run ---"
    fi
fi

# Absolute log dirs so SLURM writes each job's stdout/stderr to the right place
# regardless of that job's submit dir. Passed to sbatch as --output/--error
# (a CLI value overrides the script's relative #SBATCH path), and created up
# front because SLURM opens the log file at job start — a missing directory
# makes it silently drop the job's output.
SIA_LOGS="$SIAMESE_DIR/slurm/logs"
CMP_LOGS="$ROOT/sl_comparison/slurm/logs"
mkdir -p "$SIA_LOGS" "$CMP_LOGS"
cd "$ROOT"

echo "============================================================================"
echo "  Four-model SL benchmark (siamese_sl vs SLMGAE / SLGNN / NSF4SL)"
echo "  SLDB 3.0 given pos/neg | shared folds | seed $SEED"
echo "============================================================================"

# ---- 0. shared folds, BEFORE anything else ---------------------------------
# siamese now takes its train/val/test partition from sl_comparison/folds
# (SHARED_FOLDS in siamese_sl/slurm/config.conf), so the fold set must exist
# before the siamese array job starts. The siamese and shared-model jobs are
# submitted as siblings with no ordering edge between them, and folds/ is
# untracked build output that a fresh checkout will not have — so generating it
# inside run_shared_models.sh alone would race, or fail every siamese task with
# "folds_dir has no genes.txt". Doing it here on the login node is deterministic
# and costs seconds (pure numpy, no GPU). run_shared_models.sh then finds it
# fresh and reuses it.
#
# THIS MUST COME BEFORE reset_env IS SUBMITTED. It runs on the login node with
# $PYTHON_PATH, which lives inside the very venv reset_env.sh does `rm -rf` on
# ($VENV_DIR = the parent of PYTHON_PATH). Submitted first, reset_env can be
# allocated and start deleting the interpreter while prepare_folds.py is still
# running — killing it mid-write and leaving a half-rewritten folds/ tree whose
# meta.json (written last) still describes the previous fold set. Generating
# first closes the window entirely.
echo "--- shared folds (login node, seed $SEED) ---"
if ! $PYTHON_PATH sl_comparison/prepare_folds.py --seed "$SEED" --verify; then
    echo "FATAL: fold generation failed — every model reads these folds; aborting"
    exit 1
fi

# ---- 0b. environment reset (shared venv) -----------------------------------
ENV_DEP=""
if [ "$RESET_ENV" = true ]; then
    echo "--- reset_env ---"
    ENV_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable --export=ALL \
        --output="$SIA_LOGS/reset_env_%j.out" --error="$SIA_LOGS/reset_env_%j.err" \
        slurm/reset_env.sh)
    echo "  reset_env job: $ENV_JOB"
    ENV_DEP="--dependency=afterok:$ENV_JOB"
fi

# ---- 1. embedding generation (siamese + the shared models' kg_complex) ------
GEN_DEP="$ENV_DEP"
if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    echo "--- generate embeddings ---"
    GEN_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable --export=ALL $ENV_DEP \
        --output="$SIA_LOGS/siamese_gen_emb_%j.out" --error="$SIA_LOGS/siamese_gen_emb_%j.err" \
        slurm/run_generate_embeddings.sh)
    echo "  generation job: $GEN_JOB"
    GEN_DEP="--dependency=afterok:$GEN_JOB"
elif [ -n "$AFTER_JOB" ]; then
    echo "--- chaining after existing job $AFTER_JOB ---"
    if [ -n "$ENV_DEP" ]; then GEN_DEP="--dependency=afterok:$ENV_JOB:$AFTER_JOB"
    else GEN_DEP="--dependency=afterok:$AFTER_JOB"; fi
else
    echo "--- skipping generation (embeddings assumed present) ---"
fi

# (shared folds were generated in step 0, before reset_env could delete the
# interpreter out from under them.)

# Terminal jobs whose completion gates the final comparison.
TERMINALS=""

# ---- 2. siamese_sl: best-per-category (its own embedding, cell-line) + combo
if [ "$RUN_SIAMESE" = true ]; then
    source "$SLURM_DIR/run_best_per_category.conf"
    N=$(( ${#EMB_CATALOG[@]} * ${#CV_TYPES[@]} ))
    echo "--- siamese_sl best-per-category ($N tasks) + combo ---"
    BESTCAT_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable $GEN_DEP --export=ALL \
        --output="$SIA_LOGS/bestcat_%A_%a.out" --error="$SIA_LOGS/bestcat_%A_%a.err" \
        --array=0-$((N - 1))%${MAX_CONCURRENT} slurm/run_best_per_category.sh)
    echo "  best-cat array job: $BESTCAT_JOB"
    COMBO_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable \
        --dependency=afterany:$BESTCAT_JOB --export=ALL \
        --output="$SIA_LOGS/bestcat_combo_%j.out" --error="$SIA_LOGS/bestcat_combo_%j.err" \
        slurm/run_best_per_category_combo.sh)
    echo "  combo job:          $COMBO_JOB  (siamese's *_any comparison results)"
    TERMINALS="${TERMINALS}:$COMBO_JOB"

    # ---- 2b. External-screen transfer eval, one array task per cv_type ------
    # afterOK on the combo: it loads that combo's checkpoints, so a failed
    # combo leaves nothing to evaluate. Resolves MODEL_DIR at job start from
    # results/latest_combo_cv3.txt, which the combo writes on success.
    if [ "$RUN_EVAL" = true ]; then
        EVAL_N=$( ( set +u; source "$SLURM_DIR/eval_finetuning.conf" >/dev/null 2>&1; \
                    echo "${#EVAL_CV_TYPES[@]}" ) 2>/dev/null )
        case "$EVAL_N" in ''|*[!0-9]*|0) EVAL_N=3 ;; esac
        echo "--- external-screen eval + fine-tuning ($EVAL_N cv_types) ---"
        EVAL_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable \
            --dependency=afterok:$COMBO_JOB --export=ALL \
            --array=0-$((EVAL_N - 1))%${EVAL_CONCURRENT:-3} \
            --output="$SIA_LOGS/eval_finetuning_%A_%a.out" \
            --error="$SIA_LOGS/eval_finetuning_%A_%a.err" \
            slurm/eval_finetuning.sh)
        echo "  eval array job:     $EVAL_JOB  (Adamson / Corn / Gilbert)"
    fi

    # ---- 2c. Leave-one-category-out ablation --------------------------------
    # Also afterOK on the combo, so it runs CONCURRENTLY with the eval above
    # rather than behind it; %MAX_CONCURRENT bounds the contention. The full
    # combo is the ablation's reference row and is never retrained here, so
    # this stage cannot change any number in the comparison table.
    if [ "$RUN_ABLATION" = true ]; then
        source "$SLURM_DIR/run_ablation.conf"
        ABL_N=$(( ${#ABLATE_CATEGORIES[@]} * ${#CV_TYPES[@]} ))
        echo "--- leave-one-category-out ablation ($ABL_N tasks) ---"
        ABL_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable \
            --dependency=afterok:$COMBO_JOB --export=ALL \
            --array=0-$((ABL_N - 1))%${MAX_CONCURRENT} \
            --output="$SIA_LOGS/ablation_%A_%a.out" \
            --error="$SIA_LOGS/ablation_%A_%a.err" \
            slurm/run_ablation.sh)
        echo "  ablation array job: $ABL_JOB"
        ABL_SUMM=$(cd "$SIAMESE_DIR" && sbatch --parsable \
            --dependency=afterany:$ABL_JOB --export=ALL \
            --output="$SIA_LOGS/ablation_summary_%j.out" \
            --error="$SIA_LOGS/ablation_summary_%j.err" \
            slurm/run_ablation_summary.sh)
        echo "  ablation summary:   $ABL_SUMM -> results/ablation_summary.csv"
    fi
fi

# ---- 3. shared models: folds job, then a 3x3 ARRAY (SLGNN/NSF4SL/SLMGAE) ----
# Was one job running nine trainings in a serial for-loop. Split so SLURM can
# overlap them: fold generation and the CPU-only degree prior run once in
# run_shared_folds.sh, then |MODELS| x |CVS| independent GPU tasks.
# The array is afterOK on folds — every task grades against that answer key, so
# there is nothing to salvage if fold generation failed.
if [ "$RUN_SHARED" = true ]; then
    echo "--- shared folds + degree prior (once, CPU-only) ---"
    FOLDS_JOB=$(sbatch --parsable $GEN_DEP --export=ALL \
        --output="$CMP_LOGS/sl_folds_%j.out" --error="$CMP_LOGS/sl_folds_%j.err" \
        sl_comparison/slurm/run_shared_folds.sh)
    echo "  shared-folds job:   $FOLDS_JOB"

    SHARED_TASKS=$(( 3 * $(echo "${CVS:-cv1 cv2 cv3}" | wc -w) ))
    echo "--- shared 3-model array ($SHARED_TASKS tasks: 3 models x CVs) ---"
    SHARED_JOB=$(sbatch --parsable --dependency=afterok:$FOLDS_JOB --export=ALL \
        --array=0-$((SHARED_TASKS - 1))%${SHARED_CONCURRENT:-3} \
        --output="$CMP_LOGS/sl_compare_%A_%a.out" --error="$CMP_LOGS/sl_compare_%A_%a.err" \
        sl_comparison/slurm/run_shared_models.sh)
    echo "  shared-models array: $SHARED_JOB  (throttle ${SHARED_CONCURRENT:-3})"
    TERMINALS="${TERMINALS}:$SHARED_JOB"
fi

# ---- 4. final comparison (afterANY on every terminal) ----------------------
# afterany (not afterok): the shared array runs 9 model×CV tasks and individual
# tasks may fail; siamese's combo may also partially fail. We still want the
# table built from whatever succeeded, so run compare regardless. Depending on
# the array job id covers every task in it.
TERMINALS="${TERMINALS#:}"          # strip leading ':'
echo "--- final comparison (afterany:$TERMINALS) ---"
COMPARE_JOB=$(sbatch --parsable --dependency=afterany:$TERMINALS --export=ALL \
    --output="$CMP_LOGS/final_compare_%j.out" --error="$CMP_LOGS/final_compare_%j.err" \
    sl_comparison/slurm/run_compare.sh)
echo "  final compare job:  $COMPARE_JOB"

echo "============================================================================"
echo "All jobs submitted. Monitor:  squeue -u $(whoami)"
echo "Final table lands in:  sl_comparison/results/comparison.csv"
echo "  tail -f $CMP_LOGS/final_compare_${COMPARE_JOB}.out"
echo "============================================================================"
