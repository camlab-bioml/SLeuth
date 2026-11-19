#!/bin/bash
#SBATCH --job-name=sl_ext_transfer
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: the root
# orchestrator (run_benchmarking_siamese_vs_others.sh), the only one that
# submits this script, passes it as --nodelist on the command line, which
# replaces this line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --output=sl_comparison/slurm/logs/external_transfer_%j.out
#SBATCH --error=sl_comparison/slurm/logs/external_transfer_%j.err

# ============================================================================
# External-screen transfer tables (paper tab:external / tab:external_ft) of
# ONE PCA arm (PCA_ARM; siamese_sl/slurm/config.conf, PCA arms):
# sl_comparison/external_transfer.py on every SLeuth variant's cv3 combo of
# that arm next to the SLMGAE / SLGNN / NSF4SL cv3 baselines and DegreePrior,
# all scored on identical screen rows, population sd over folds. The arm's
# roots come from sl_comparison/slurm/pca_arm.sh: its siamese root
# (siamese_sl/results, or siamese_sl/results/pca_<arm>) and its comparison
# root (sl_comparison/results, or sl_comparison/results/pca_<arm>).
#
# Inputs, each gated so a stage that failed THIS run is absent (with a loud
# message) rather than silently replaced by an earlier run's files:
#   SLeuth variant  <siamese root>/<prefix>latest_combo_cv3.txt names the
#                   combo (the combo job deletes the markers when it starts and
#                   writes each after that combo trains), by its path from
#                   siamese_sl/; that combo must be one of the variant's
#                   (<prefix>best_cat_*_cv3) directly in the arm's siamese
#                   root, so another arm's combo (or a hand copy) is never
#                   tabled under this arm's name. Its
#                   eval_finetuning_{cv1,cv2,cv3}/ must hold config_used.json
#                   and every <Screen>_predictions.csv. Row name siamese_sl
#                   (pd:per_head) or siamese_sl_<kernel>_<cell_lines>.
#   baseline        all fold_<k>_cv3_predictions.npy (k over the shared cv3
#                   folds) AND sl_comparison/results/<NAME>_cv3.json, the
#                   shared models' row, the same for every arm. That row
#                   is moved aside at submission when this run retrains the
#                   shared models and rewritten only when the task succeeds;
#                   with --skip-shared the previous run's row and matrices are
#                   reused.
#   both            a fold_stamp (model_params.json) matching the current
#                   sl_comparison/folds: external_transfer.py refuses the WHOLE
#                   run on one stale input, so a model trained on another fold
#                   set (e.g. reused after step 0 rebuilt changed folds) is
#                   left ABSENT here instead.
# Everything else (fold stamps again, checkpoint sha256 vs config_used.json,
# matched row sets) is checked by external_transfer.py itself.
#
# Needs no GPU (numpy/pandas only), so it holds none.
# Chained by run_benchmarking_siamese_vs_others.sh, once per arm, afterany on
# that arm's eval_finetuning tasks and the shared-model array. Standalone,
# from the repo root:
#   PCA_ARM=<arm> sbatch --export=ALL sl_comparison/slurm/run_external_transfer.sh
# with the run's other PCA_* settings exported as at its submission, and
# PCA_ARM=none for arm none (2026-10-21: without PCA_ARM the job is the arm
# the submitting shell exports, arm none only where none or nothing is; this
# said a bare sbatch was arm none's).
# Output: <comparison root>/external_transfer/ (sl_comparison/results/
# external_transfer/ for arm none, sl_comparison/results/pca_<arm>/
# external_transfer/ otherwise; cleared first, the arm's only; compare.py
# reads only the top-level *.json, so the subdirectory is never tabulated as
# model rows). Each file names its arm too (2026-10-21: external_transfer.py
# --pca_arm / --pca_label, label only): a trailing pca_arm column in every
# CSV and 'PCA arm: <arm> (<label>)' under summary.md's title, since arm
# none's directory is the pre-arms path and a copy says nothing otherwise.
# ============================================================================

module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. sl_comparison/slurm/python_path.sh   # the siamese jobs' interpreter (config.conf)
# This job's PCA arm: SLEUTH_ARM / SLEUTH_RESULTS / SLEUTH_CMP (config.conf). A
# setting config.conf refuses stops the job here, before anything is removed.
. sl_comparison/slurm/pca_arm.sh || exit 1
mkdir -p sl_comparison/slurm/logs "$SLEUTH_CMP"

OUT="$SLEUTH_CMP/external_transfer"
MODEL_CV=cv3
SCREENS="Adamson AdamsonGI Corn Gilbert"   # external_transfer.py defaults
# FIXED, not read from eval_finetuning.conf EVAL_CV_TYPES: paper_tables.SPLITS
# is likewise ("cv1","cv2","cv3") and tab:external / tab:external-ft are a
# screens x THESE-three grid, so a run that evaluated fewer cannot fill them.
# Passing the shorter list to external_transfer.py would make build_external
# return "ok" on a third of a table (paper_tables would only list the absent
# cells: PAPER INCOMPLETE, exit 3); refusing here stops the run loudly at the
# source. The gate below names EVAL_CV_TYPES as a cause.
SPLITS="cv1 cv2 cv3"

echo "=================================================="
echo "External-screen transfer tables: PCA arm $SLEUTH_ARM ($SLEUTH_PCA_LABEL)"
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "  combos from $SLEUTH_RESULTS/, tables -> $OUT/"
echo "=================================================="

# A table this run cannot rebuild must not survive from an earlier one.
rm -rf "$OUT"

# One line per variant, KERNEL:CELL_LINES|TAG|PREFIX, from config.conf in a
# bash subshell with SLURM_JOB_ID unset (its compute-node block stays inert;
# its own output is discarded so only the loop's lines are read).
VARIANT_ROWS=$(bash -c '
    unset SLURM_JOB_ID
    source "$1" >/dev/null 2>&1 || exit 1
    for v in "${SIAMESE_VARIANTS[@]}"; do
        echo "$v|$(siamese_variant_tag "$v")|$(siamese_variant_prefix "$v")"
    done
' _ siamese_sl/slurm/config.conf) || VARIANT_ROWS=""
if [ -z "$VARIANT_ROWS" ]; then
    echo "ERROR: could not read SIAMESE_VARIANTS from siamese_sl/slurm/config.conf"
    exit 1
fi

# Mismatches between DIR's fold_stamp and the current folds (empty == current);
# a check that cannot run counts as a mismatch. stdin is /dev/null so the
# variant loop's here-string is never read.
stamp_diffs() {   # DIR
    local out
    out=$($PYTHON_PATH -c '
import sys
sys.path.insert(0, "sl_comparison")
import model_params as mp
st = mp.read_model_params(sys.argv[1]).get("fold_stamp")
print("no fold_stamp in model_params.json" if not st else
      "; ".join(mp.compare_fold_stamp(st, "sl_comparison/folds", sys.argv[2])))
' "$1" "$MODEL_CV" </dev/null 2>&1) || out="fold-stamp check failed: ${out:-no output}"
    printf '%s' "$out"
}

# The arm's siamese root and a directory's parent, physically (symlinks and
# relative or absolute spellings resolved; "" when it does not exist), for the
# marker check below.
phys_dir() {   # DIR
    (CDPATH='' cd -- "$1" 2>/dev/null && pwd -P) || true
}
ARM_ROOT=$(phys_dir "$SLEUTH_RESULTS")

SIAMESE_ARGS=""
INCLUDED=""
ABSENT=""
while IFS='|' read -r V TAG PREFIX; do
    [ -n "$V" ] || continue
    NAME="siamese_sl${TAG:+_$TAG}"
    MARKER="$SLEUTH_RESULTS/${PREFIX}latest_combo_${MODEL_CV}.txt"
    if [ ! -f "$MARKER" ]; then
        echo "WARNING: $NAME ($V) ABSENT from the external-transfer tables: no"
        echo "         $MARKER (its $MODEL_CV combo did not train this run in"
        echo "         PCA arm $SLEUTH_ARM)."
        ABSENT="$ABSENT $NAME"
        continue
    fi
    DIR=$(head -n 1 "$MARKER" | tr -d '\r')
    case "$DIR" in /*) ;; *) DIR="siamese_sl/$DIR" ;; esac   # marker is relative to siamese_sl/
    DIR="${DIR%/}"
    # The marker's text is the combo's path, so a marker of another arm's
    # spelling (results/<combo> is arm none's) or a hand edit would put that
    # combo's numbers under this arm's name. Only one of this variant's cv3
    # combos directly in the arm's siamese root is this arm's (the rule
    # collect_siamese.py applies to the same marker for the arm's rows).
    case "$(basename "$DIR")" in
        "${PREFIX}best_cat_"*"_$MODEL_CV") COMBO_NAME_OK=1 ;;
        *) COMBO_NAME_OK=0 ;;
    esac
    if [ -z "$ARM_ROOT" ] || [ "$COMBO_NAME_OK" != 1 ] \
            || [ "$(phys_dir "$(dirname "$DIR")")" != "$ARM_ROOT" ]; then
        echo "WARNING: $NAME ($V) ABSENT from the external-transfer tables: $MARKER"
        echo "         names $DIR, which is not a ${PREFIX}best_cat_*_$MODEL_CV combo"
        echo "         directly in PCA arm $SLEUTH_ARM's $SLEUTH_RESULTS/ (another"
        echo "         arm's or variant's, or a hand copy); refused."
        ABSENT="$ABSENT $NAME"
        continue
    fi
    MISSING=""
    for S in $SPLITS; do
        [ -f "$DIR/eval_finetuning_$S/config_used.json" ] \
            || MISSING="$MISSING $DIR/eval_finetuning_$S/config_used.json"
        for SC in $SCREENS; do
            [ -f "$DIR/eval_finetuning_$S/${SC}_predictions.csv" ] \
                || MISSING="$MISSING $DIR/eval_finetuning_$S/${SC}_predictions.csv"
        done
    done
    if [ -n "$MISSING" ]; then
        set -- $MISSING
        echo "WARNING: $NAME ($V) ABSENT from the external-transfer tables: $# of"
        echo "         its [$SPLITS] eval output(s) missing (first: $1) — that"
        echo "         eval_finetuning task failed, or eval_finetuning.conf"
        echo "         EVAL_CV_TYPES does not cover all three (the tables need them)."
        ABSENT="$ABSENT $NAME"
        continue
    fi
    SD=$(stamp_diffs "$DIR")
    if [ -n "$SD" ]; then
        echo "WARNING: $NAME ($V) ABSENT from the external-transfer tables: $DIR"
        echo "         is not on the current sl_comparison/folds ($SD)."
        ABSENT="$ABSENT $NAME"
        continue
    fi
    SIAMESE_ARGS="$SIAMESE_ARGS --siamese $NAME=$DIR"
    INCLUDED="$INCLUDED $NAME"
done <<< "$VARIANT_ROWS"

NFOLD=$(ls -d sl_comparison/folds/"$MODEL_CV"/fold_* 2>/dev/null | wc -l | tr -d ' ')
BASELINE_ARGS=""
for ENTRY in "SLMGAE|SLMGAE-in-pytorch/results/slmgae_shared_$MODEL_CV" \
             "SLGNN|SLGNN/results/slgnn_$MODEL_CV" \
             "NSF4SL|NSF4SL/results/nsf4sl_$MODEL_CV"; do
    IFS='|' read -r BN BD <<< "$ENTRY"
    ROW="sl_comparison/results/${BN}_$MODEL_CV.json"
    if [ ! -f "$ROW" ]; then
        echo "WARNING: $BN ABSENT from the external-transfer tables: no $ROW"
        echo "         (its $MODEL_CV task did not succeed this run)."
        ABSENT="$ABSENT $BN"
        continue
    fi
    NM=0
    K=0
    while [ "$K" -lt "$NFOLD" ]; do
        [ -f "$BD/fold_${K}_${MODEL_CV}_predictions.npy" ] && NM=$((NM + 1))
        K=$((K + 1))
    done
    if [ "$NFOLD" -eq 0 ] || [ "$NM" -ne "$NFOLD" ]; then
        echo "WARNING: $BN ABSENT from the external-transfer tables: $NM of $NFOLD"
        echo "         fold_<k>_${MODEL_CV}_predictions.npy in $BD."
        ABSENT="$ABSENT $BN"
        continue
    fi
    SD=$(stamp_diffs "$BD")
    if [ -n "$SD" ]; then
        echo "WARNING: $BN ABSENT from the external-transfer tables: $BD is not on"
        echo "         the current sl_comparison/folds ($SD); retrain it."
        ABSENT="$ABSENT $BN"
        continue
    fi
    BASELINE_ARGS="$BASELINE_ARGS --baseline $BN=$BD"
    INCLUDED="$INCLUDED $BN"
done

if [ -z "$SIAMESE_ARGS" ]; then
    echo "ERROR: no SLeuth variant of PCA arm $SLEUTH_ARM has a complete,"
    echo "       current-fold cv3 combo + eval in $SLEUTH_RESULTS/, so no"
    echo "       external-transfer table can be built (absent:$ABSENT)."
    exit 1
fi
if [ -z "$BASELINE_ARGS" ]; then
    # Without --baseline, external_transfer.py falls back to the default
    # baseline dirs, i.e. exactly the unvetted files the gate above refused.
    if $PYTHON_PATH sl_comparison/external_transfer.py --help 2>/dev/null | grep -q -- '--no_baselines'; then
        BASELINE_ARGS="--no_baselines"
    else
        echo "ERROR: no baseline passed the gate and external_transfer.py has no"
        echo "       --no_baselines flag; refusing to let it fall back to unvetted dirs."
        exit 1
    fi
fi

echo ""
echo "PCA arm:  $SLEUTH_ARM ($SLEUTH_PCA_LABEL)"
echo "Included:$INCLUDED"
echo "Absent:  ${ABSENT:- (none)}"
echo "  (+ DegreePrior, computed from the shared $MODEL_CV training folds)"
echo ""
# Word splitting of the two arg strings is intended: names and repo-relative
# paths never contain spaces. The arm and its label are quoted (the label has
# spaces); --out_dir stays last.
$PYTHON_PATH sl_comparison/external_transfer.py $SIAMESE_ARGS $BASELINE_ARGS \
    --degree_prior --model_cv "$MODEL_CV" \
    --pca_arm "$SLEUTH_ARM" --pca_label "$SLEUTH_PCA_LABEL" --out_dir "$OUT"
RC=$?

echo ""
echo "=================================================="
if [ "$RC" -ne 0 ]; then
    echo "FAILED: external_transfer.py exited $RC (no tables in $OUT)."
else
    echo "External-transfer tables, PCA arm $SLEUTH_ARM -> $OUT/${ABSENT:+  (ABSENT:$ABSENT)}"
fi
echo "Completed at: $(date)"
echo "=================================================="
exit "$RC"
