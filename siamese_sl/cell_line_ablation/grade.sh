#!/bin/bash
# ============================================================================
# Grade the cell-line ablation with the SHARED evaluator
# ============================================================================
# The ablation is only worth anything if it is scored by the same code, on the
# same test edges, over the same 8,844-gene universe as the published run.
# That means sl_comparison/evaluate_model.py on the exported matrices. The
# metrics in zero_cell_emb.py's own results.json are already the OR-collapsed
# (max-over-heads) view on the same test edges (its as-trained row must match
# train.py's *_any within 1e-3), but only evaluate_model.py applies the shared
# shift, universe and answer key every other row is graded with.
#
# Never take the number from the source combo's train.py results.json either:
# in cell-line mode it carries BOTH schemas' keys, and the bare
# `auprg/aupr/auroc` are the STRATIFIED metrics pooled over observed
# (pair, cell-line) cells, while the published table reports the OR-collapsed
# `*_any` view graded by the shared evaluator at the shared prevalence.
# Reading the wrong key gives a number that looks right and is not comparable
# to anything in the paper.
#
# Runs from anywhere (it cds to the repo root, where evaluate_model.py
# resolves sl_comparison/folds):
#
#   bash siamese_sl/cell_line_ablation/grade.sh
#
# Requires zero_cell_emb.py to have been run with --save_matrices.
#
# Writes $GRADE_OUT_DIR/${MODEL_NAME}_<cv>.json (default sl_comparison/results,
# arm none's comparison directory; run_cell_line_ablation.sh passes its PCA
# arm's, sl_comparison/results/pca_<arm>/ for the others). compare.py globs
# *.json and keys rows on summary["model"], so the new row then appears in the
# comparison table with no code change. Once a CV's matrix dir exists, any
# previous ${MODEL_NAME}_<cv>.json is moved to $GRADE_OUT_DIR/stale_siamese/
# before grading, so a partial matrix set or a failed grading leaves the row
# MISSING, never an earlier grading; a CV whose dir does not exist is left
# untouched. Exit 0 only when every CV in GRADE_CVS was graded.
#
# The default MODEL_NAME is not a pipeline row: the pipeline's rows
# (<row>_nocell, written through run_cell_line_ablation.sh, which passes
# MODEL_NAME and PRED_DIR_TEMPLATE) are the ones paper_tables cross-checks
# against <arm root>/<prefix>cellline_ablation_<cv>/results.json, and a hand run
# into models/ writes neither that file nor, by default, those rows. To
# rebuild them, resubmit run_cell_line_ablation.sh.
#
# Environment (all optional):
#   MODEL_NAME         row name (default siamese_sl_manual_nocell)
#   PRED_DIR_TEMPLATE  matrices dir per CV, %s = the CV, relative to the repo
#                      root (default siamese_sl/cell_line_ablation/models/
#                      no_cellline_%s, zero_cell_emb.py's default layout)
#   GRADE_CVS          space-separated CVs (default "cv1 cv2 cv3")
#   GRADE_OUT_DIR      where the rows go, relative to the repo root (default
#                      sl_comparison/results; a PCA arm's is CMP_ROOT,
#                      siamese_sl/slurm/config.conf)
#   PYTHON_PATH        interpreter (else sl_comparison/slurm/python_path.sh)
# ============================================================================

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIAMESE_DIR="$(dirname "$HERE")"
ROOT="$(dirname "$SIAMESE_DIR")"

if [ ! -d "$ROOT/sl_comparison/folds" ]; then
    echo "ERROR: $ROOT/sl_comparison/folds not found." >&2
    exit 1
fi
cd "$ROOT"

# The interpreter every sl_comparison job uses (an exported PYTHON_PATH wins,
# then siamese_sl/slurm/config.conf's); python3 on PATH when that file is
# absent, so the script also runs locally.
if [ -f sl_comparison/slurm/python_path.sh ]; then
    . sl_comparison/slurm/python_path.sh
    if [ ! -x "$PYTHON_PATH" ] && ! command -v "$PYTHON_PATH" >/dev/null 2>&1; then
        PYTHON_PATH=python3
    fi
else
    PYTHON_PATH="${PYTHON_PATH:-python3}"
fi

# Row name in the comparison table (summary["model"]); by default not one of
# the pipeline's <row>_nocell rows (header).
MODEL_NAME="${MODEL_NAME:-siamese_sl_manual_nocell}"
PRED_DIR_TEMPLATE="${PRED_DIR_TEMPLATE:-siamese_sl/cell_line_ablation/models/no_cellline_%s}"
GRADE_CVS="${GRADE_CVS:-cv1 cv2 cv3}"
case "$PRED_DIR_TEMPLATE" in
    *%s*) ;;
    *) echo "ERROR: PRED_DIR_TEMPLATE must contain %s for the CV" >&2; exit 1 ;;
esac

GRADE_OUT_DIR="${GRADE_OUT_DIR:-sl_comparison/results}"
mkdir -p "$GRADE_OUT_DIR"
STALE_DIR="$GRADE_OUT_DIR/stale_siamese"
TS=$(date +%Y%m%dT%H%M%S)

GRADED=0
FAILED=0
for CV in $GRADE_CVS; do
    # shellcheck disable=SC2059
    PRED_DIR=$(printf "$PRED_DIR_TEMPLATE" "$CV")
    OUT="$GRADE_OUT_DIR/${MODEL_NAME}_${CV}.json"

    if [ ! -d "$PRED_DIR" ]; then
        echo "SKIP $CV: $PRED_DIR does not exist (array task not run?)"
        [ -f "$OUT" ] && echo "          $OUT left as it was (not regraded)"
        FAILED=$((FAILED + 1))
        continue
    fi

    # zero_cell_emb.py rewrites or clears this dir on every run, so an earlier
    # grading may describe matrices that are gone: if this grading does not
    # succeed, the row must be missing rather than that one.
    if [ -f "$OUT" ]; then
        mkdir -p "$STALE_DIR"
        mv -f "$OUT" "$STALE_DIR/${MODEL_NAME}_${CV}.${TS}.json"
        echo "  moved previous $OUT aside to $STALE_DIR/"
    fi

    # Refuse a partial fold set explicitly. evaluate_model.py would also
    # refuse, but say so here with the actual count so the cause is obvious.
    NPRED=$(ls "$PRED_DIR"/fold_*_"${CV}"_predictions.npy 2>/dev/null | wc -l | tr -d ' ')
    NFOLD=$(ls -d sl_comparison/folds/"$CV"/fold_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPRED" -ne "$NFOLD" ]; then
        echo "SKIP $CV: $NPRED score matrices for $NFOLD folds — a partial mean"
        echo "          is indistinguishable from a complete one in the table."
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "--- $MODEL_NAME / $CV (shared evaluator, $NPRED folds) ---"
    if $PYTHON_PATH sl_comparison/evaluate_model.py \
            --model_name "$MODEL_NAME" \
            --cv_type "$CV" \
            --pred_dir "$PRED_DIR" \
            --out "$OUT"; then
        GRADED=$((GRADED + 1))
    else
        echo "ERROR: grading failed for $CV"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================================"
echo "graded $GRADED cv(s), $FAILED skipped/failed"
if [ "$FAILED" -gt 0 ]; then
    echo "NOT every CV in GRADE_CVS ($GRADE_CVS) was graded: exit 1."
fi
if [ "$GRADED" -gt 0 ]; then
    echo ""
    echo "To fold the new row into the full four-model table, rerun"
    echo "  sl_comparison/slurm/run_compare.sh (with this arm's PCA_ARM exported): it"
    echo "  reads the SLeuth rows of $GRADE_OUT_DIR and the shared models' rows of"
    echo "  sl_comparison/results."
    echo ""
    echo "NOTE: do NOT run sl_comparison/collect_siamese.py to pick this up."
    echo "      It reads only the combos named by the latest_combo_<cv>.txt markers, and"
    echo "      this ablation's results.json has no summary, so it is never collected."
fi
echo "============================================================================"
[ "$GRADED" -gt 0 ] && [ "$FAILED" -eq 0 ] || exit 1
