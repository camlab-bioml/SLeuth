#!/bin/bash
#SBATCH --job-name=sl_folds
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: the root
# orchestrator (run_benchmarking_siamese_vs_others.sh), the only one that
# submits this script, passes it as --nodelist on the command line, which
# replaces this line (GPU_EXCLUDE only adds --exclude and cannot lift it).
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

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. sl_comparison/slurm/python_path.sh   # the siamese jobs' interpreter (config.conf)
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
# folds/ is tracked in git and travels with every upload of the tree, so a
# pull or an upload can replace it; the content check below catches a mismatch
# (run_benchmarking_siamese_vs_others.sh also keeps it read-only while its DAG
# runs, so an upload onto it fails instead). The pos/neg source files are fingerprinted too
# (prepare_folds.py --check_sources against meta.json source_sha256): paths
# alone miss a pair file re-downloaded in place.
#
# SHARED_FOLDS_FROZEN=1 (exported by run_benchmarking_siamese_vs_others.sh,
# which generated and verified the folds on the login node moments before):
# other jobs of that DAG read folds/ concurrently, and prepare_folds.py
# deletes the old fold dirs and meta.json before rewriting, so regenerating
# here would pull the answer key out from under them. A stale verdict is then
# fatal instead of a rewrite.
NEED_FOLDS=1
STALE=""
SRC_NOTE="the pos/neg source files match its source_sha256"
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
    # The fold ARRAYS themselves: every cv/fold_k/<split>.npy meta names must
    # exist and hash to meta's per-CV split_digest (prepare_folds.split_digest,
    # the same function that wrote it). The two counts above cannot see a
    # missing fold directory or a partition that differs from the one meta
    # describes, and every model trains and is graded on exactly these arrays.
    import numpy as _np
    sys.path.insert(0, 'sl_comparison')
    from prepare_folds import split_digest
    keys = ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg",
            "test_neg")
    n_folds = have.get('num_folds') or 0
    for cv in have.get('cv_types') or []:
        paths = [f"sl_comparison/folds/{cv}/fold_{j}/{k}.npy"
                 for j in range(n_folds) for k in keys]
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            diffs.append(f"{cv}: {len(miss)} fold arrays missing ({miss[0]})")
            continue
        folds = [{k: _np.load(f"sl_comparison/folds/{cv}/fold_{j}/{k}.npy")
                  for k in keys} for j in range(n_folds)]
        want_d = (have.get('split_digest') or {}).get(cv)
        if split_digest(folds) != want_d:
            diffs.append(f"{cv}: fold arrays do not match meta split_digest "
                         f"{want_d!r}")
except Exception as e:
    diffs.append(f"fold content unreadable ({e})")
print("; ".join(diffs))
PYEOF
) || STALE="staleness check failed"
    # --check_sources: 0 = match, 1 = a recorded source differs (pair files,
    # sidecars, or siamese_sl/cell_line_vocab.py), meta.json has no
    # source_sha256, or its fingerprint predates the vocabulary key;
    # anything else (a crash, an unknown flag) = not checked,
    # which is stale too, except for a set the orchestrator built from these
    # very files in step 0 (SHARED_FOLDS_FROZEN), where refusing would only
    # cost the shared models.
    if [ -z "$STALE" ]; then
        $PYTHON_PATH sl_comparison/prepare_folds.py --check_sources
        _CS=$?
        if [ "$_CS" -eq 1 ]; then
            STALE="a recorded source differs from meta.json source_sha256, or it is absent/incomplete (see the STALE: lines above)"
        elif [ "$_CS" -ne 0 ] && [ "${SHARED_FOLDS_FROZEN:-0}" = 1 ]; then
            echo "WARNING: could not fingerprint the pos/neg source files (prepare_folds.py"
            echo "  --check_sources exit $_CS); the orchestrator built these folds from them in step 0."
            SRC_NOTE="the pos/neg source files were NOT fingerprinted (--check_sources exit $_CS)"
        elif [ "$_CS" -ne 0 ]; then
            STALE="could not fingerprint the pos/neg source files (prepare_folds.py --check_sources exit $_CS)"
        fi
    fi
    [ -z "$STALE" ] && NEED_FOLDS=0
fi
# A set that passed every check above can still differ from the one step 0
# built: genes.txt and all_pos.npy are compared by COUNT, the arrays by
# partition digest, and a whole regenerated set agrees with its own meta.json.
# Under the orchestrator the manifest is the exact record (-s: an empty file
# is no record, as before).
if [ "$NEED_FOLDS" = 0 ] && [ "${SHARED_FOLDS_FROZEN:-0}" = 1 ] && [ -s "${FOLDS_MANIFEST:-}" ]; then
    . sl_comparison/slurm/folds_manifest.sh
    if [ "$(folds_manifest)" != "$(cat "$FOLDS_MANIFEST")" ]; then
        STALE="fold files differ from the set step 0 built ($FOLDS_MANIFEST)"
        NEED_FOLDS=1
    fi
fi
# Under the orchestrator, step 0 built this fold set, made it read-only and
# recorded its manifest (FOLDS_MANIFEST). A stale verdict with the fold files
# byte-identical to that manifest means a fold SOURCE (pair file, sidecar,
# cell_line_vocab.py) or prepare_folds.py changed after step 0: every job of
# this DAG still reads the answer key step 0 built, so the shared models go
# ahead, and paper_tables lists the change (a stale source, or prepare_folds.py
# as code drift against CODE_MANIFEST; PAPER INCOMPLETE either way). A fold
# set that differs from the manifest, or no manifest, stays fatal below.
if [ "$NEED_FOLDS" = 1 ] && [ "${SHARED_FOLDS_FROZEN:-0}" = 1 ] && [ -n "$STALE" ] \
   && [ -f "${FOLDS_MANIFEST:-}" ]; then
    . sl_comparison/slurm/folds_manifest.sh
    if [ "$(folds_manifest)" = "$(cat "$FOLDS_MANIFEST")" ]; then
        echo "WARNING: shared folds are STALE ($STALE),"
        echo "  but every fold file is byte-identical to the set step 0 built for this"
        echo "  DAG ($FOLDS_MANIFEST), so what changed after submission is a fold"
        echo "  source or prepare_folds.py, not the folds. All jobs read the same"
        echo "  answer key, so the shared models go ahead; paper_tables lists the"
        echo "  change (a stale source, or prepare_folds.py as changed code)."
        SRC_NOTE="STALE ($STALE); reused only because every fold file is byte-identical to $FOLDS_MANIFEST"
        NEED_FOLDS=0
    fi
fi
if [ "$NEED_FOLDS" = 1 ] && [ "${SHARED_FOLDS_FROZEN:-0}" = 1 ]; then
    echo "FATAL: shared folds are ${STALE:+STALE ($STALE)}$([ -n "$STALE" ] || echo missing), but"
    echo "  SHARED_FOLDS_FROZEN=1: the orchestrator generated them for this DAG and"
    echo "  other jobs read them now, so they are not rewritten here. Resubmit the"
    echo "  run (step 0 regenerates them on the login node)."
    exit 1
fi
if [ "$NEED_FOLDS" = 1 ]; then
    [ -n "$STALE" ] && echo "--- existing folds are STALE ($STALE) ---"
    echo "--- generating shared folds (seed $SEED) ---"
    # A previous orchestrated run left the set read-only; freeze it again
    # after, as the orchestrator does (no job writes under it).
    [ -d sl_comparison/folds ] && chmod -R u+w sl_comparison/folds
    $PYTHON_PATH sl_comparison/prepare_folds.py --seed "$SEED" --verify \
        || { echo "FATAL: fold generation failed"; exit 1; }
    chmod -R a-w sl_comparison/folds \
        || echo "WARNING: could not make sl_comparison/folds read-only"
    echo "NOTE: sl_comparison/folds was regenerated here; bring it back with the"
    echo "  results and commit it where you commit (the repo's answer key)."
elif [ -n "$STALE" ]; then
    echo "--- reusing existing shared folds: $SRC_NOTE ---"
else
    echo "--- reusing existing shared folds (meta.json matches every prepare_folds"
    echo "    default + seed; genes, positives and each CV's fold arrays match it;"
    echo "    $SRC_NOTE) ---"
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
    # Same rule as run_shared_models.sh: a failed control must leave its row
    # missing, never the previous run's.
    if [ -f "sl_comparison/results/DegreePrior_$CV.json" ]; then
        mkdir -p sl_comparison/results/stale_shared
        mv -f "sl_comparison/results/DegreePrior_$CV.json" \
            "sl_comparison/results/stale_shared/DegreePrior_$CV.$(date +%Y%m%dT%H%M%S).json"
    fi
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
