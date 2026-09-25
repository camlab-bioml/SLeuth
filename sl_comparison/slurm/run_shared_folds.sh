#!/bin/bash
#SBATCH --job-name=sl_folds
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --output=sl_comparison/slurm/logs/sl_folds_%j.out
#SBATCH --error=sl_comparison/slurm/logs/sl_folds_%j.err

# ============================================================================
# Shared folds + the zero-parameter control. Runs ONCE, before the model array.
#
# Split out of run_shared_models.sh (2026-08-24) so the three shared-fold models
# can train as a SLURM array instead of a serial for-loop. Fold generation
# cannot live in the array: nine tasks racing to write sl_comparison/folds/
# would interleave partial writes into the one answer key every model is graded
# against.
#
# No GPU is requested. Fold generation and the degree prior are CPU-only, and
# holding a GPU here would block a training task on the same node.
#
# Submit from the REPO ROOT:  sbatch sl_comparison/slurm/run_shared_folds.sh
# ============================================================================

set -u   # NOT -e: the degree-prior runs are handled individually below

echo "=================================================="
echo "Shared folds + degree-prior control"
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================="

module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

PYTHON_PATH="${PYTHON_PATH:-/ddn_exa/campbell/kaiyang/pytorch/bin/python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p sl_comparison/slurm/logs sl_comparison/results

CVS="${CVS:-cv1 cv2 cv3}"
SEED="${SEED:-42}"
FAILURES=""

# --- 1. Shared folds: (re)generate unless meta.json matches on EVERY input ---
# Comparing only the seed was not enough: meta.json also records the conflict
# policy, val_frac, fold count and the pos/neg source paths, and a stale folds/
# directory from an earlier run would silently be reused with a DIFFERENT answer
# key. When --conflict_policy flipped from 'drop' to 'cell_aware' that would
# have kept 3,791 cross-cell-line pairs deleted (16,624 positives instead of
# 20,415 — an 18.6% difference) while every log line still claimed cell_aware.
# folds/ is untracked build output, so it survives a git pull.
NEED_FOLDS=1
STALE=""
if [ -f sl_comparison/folds/meta.json ]; then
    STALE=$($PYTHON_PATH - "$SEED" <<'PYEOF'
import json, subprocess, sys
want_seed = int(sys.argv[1])
try:
    have = json.load(open('sl_comparison/folds/meta.json'))
except Exception as e:
    print(f"unreadable meta.json ({e})"); raise SystemExit(0)
# Ask prepare_folds.py itself for its current defaults, so this check can never
# drift from the generator.
r = subprocess.run([sys.executable, 'sl_comparison/prepare_folds.py',
                    '--print_defaults'], capture_output=True, text=True)
if r.returncode != 0:
    print("could not read prepare_folds defaults"); raise SystemExit(0)
want = json.loads(r.stdout)
want['seed'] = want_seed
diffs = [f"{k}: folds have {have.get(k)!r}, run wants {v!r}"
         for k, v in want.items() if have.get(k) != v]
# CONTENT check, not just parameters: meta.json can disagree with the arrays
# actually on disk (a partial or interrupted write, or a --max_genes preflight
# overwriting a real fold set). Compare what meta CLAIMS against what is there.
import os
try:
    g = sum(1 for ln in open('sl_comparison/folds/genes.txt') if ln.strip())
    if g != have.get('num_nodes'):
        diffs.append(f"genes.txt has {g} genes but meta claims "
                     f"{have.get('num_nodes')}")
    ap = 'sl_comparison/folds/all_pos.npy'
    if not os.path.exists(ap):
        diffs.append("all_pos.npy missing (needed to mask known SL pairs)")
    else:
        import numpy as _np
        n_ap = len(_np.load(ap))
        if n_ap != have.get('num_positives'):
            diffs.append(f"all_pos.npy has {n_ap} pairs but meta claims "
                         f"{have.get('num_positives')}")
except Exception as e:
    diffs.append(f"fold content unreadable ({e})")
print("; ".join(diffs))
PYEOF
) || STALE="staleness check failed"
    [ -z "$STALE" ] && NEED_FOLDS=0
fi
if [ "$NEED_FOLDS" = 1 ]; then
    [ -n "$STALE" ] && echo "--- existing folds are STALE ($STALE) ---"
    echo "--- generating shared folds (seed $SEED) ---"
    $PYTHON_PATH sl_comparison/prepare_folds.py --seed "$SEED" --verify \
        || { echo "FATAL: fold generation failed"; exit 1; }
else
    echo "--- reusing existing shared folds (meta.json matches on seed, "
    echo "    num_folds, val_frac and conflict_policy) ---"
fi
# --- 2. Zero-parameter control -----------------------------------------------
# Cheap (seconds, CPU-only) and it belongs here rather than in the model array:
# it needs no GPU slot, and running it before any model guarantees the table has
# a reference row even if every trained model later fails. Under CV3 it is
# provably constant (no test gene has a training degree), so its AUROC is
# exactly 0.5 — that is the calibration point for the whole comparison.
for CV in $CVS; do
    echo ""
    echo "--- DegreePrior / $CV ---"
    $PYTHON_PATH sl_comparison/degree_baseline.py --cv_type "$CV" \
        --out "sl_comparison/results/DegreePrior_$CV.json" \
        || { echo "FAIL: DegreePrior/$CV"; FAILURES="$FAILURES DegreePrior/$CV"; }
done

echo ""
echo "=================================================="
if [ -n "$FAILURES" ]; then
    echo "COMPLETED WITH FAILURES:$FAILURES"
else
    echo "Folds ready and degree prior scored for: $CVS"
fi
echo "Completed at: $(date)"
echo "=================================================="
# Exit non-zero only if FOLDS failed — a degree-prior failure must not block the
# model array, which is chained afterok on this job.
[ -d sl_comparison/folds/cv1/fold_0 ] || { echo "FATAL: folds missing"; exit 1; }
exit 0
