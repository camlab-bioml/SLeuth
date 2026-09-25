#!/bin/bash
# ============================================================================
# Grade the cell-line ablation with the SHARED evaluator
# ============================================================================
# The ablation is only worth anything if it is scored by the same code, on the
# same test edges, over the same 8,844-gene universe as the published run.
# That means sl_comparison/evaluate_model.py — NOT the ablation's own
# results.json.
#
# Why not siamese's results.json: in cell-line mode it carries BOTH schemas'
# keys, and the bare `auprg/aupr/auroc` are the STRATIFIED metrics pooled over
# observed (pair, cell-line) cells, while the published table reports the
# OR-collapsed `*_any` view graded by the shared evaluator at the shared
# prevalence. Reading the wrong key gives a number that looks right and is not
# comparable to anything in the paper.
#
# RUN FROM THE REPO ROOT (evaluate_model.py resolves sl_comparison/folds
# relative to cwd):
#
#   ./siamese_sl/cell_line_ablation/grade.sh
#
# Requires zero_cell_emb.py to have been run with --save_matrices.
#
# Writes sl_comparison/results/siamese_sl_nocell_<cv>.json. compare.py globs
# *.json and keys rows on summary["model"], so the new row then appears in the
# four-model table with no code change.
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

# Reuse the cluster interpreter when this runs on the cluster; fall back to
# whatever python3 is on PATH so the script is also usable locally.
if [ -x "/ddn_exa/campbell/kaiyang/pytorch/bin/python3" ]; then
    PYTHON_PATH="/ddn_exa/campbell/kaiyang/pytorch/bin/python3"
else
    PYTHON_PATH="${PYTHON_PATH:-python3}"
fi

# Pull the two names/paths from the ablation config rather than restating them.
MODEL_NAME="${MODEL_NAME:-siamese_sl_nocell}"

mkdir -p sl_comparison/results

GRADED=0
FAILED=0
for CV in cv1 cv2 cv3; do
    PRED_DIR="siamese_sl/cell_line_ablation/models/no_cellline_${CV}"
    OUT="sl_comparison/results/${MODEL_NAME}_${CV}.json"

    if [ ! -d "$PRED_DIR" ]; then
        echo "SKIP $CV: $PRED_DIR does not exist (array task not run?)"
        FAILED=$((FAILED + 1))
        continue
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
if [ "$GRADED" -gt 0 ]; then
    echo ""
    "$PYTHON_PATH" "$HERE/summarise.py" || true
    echo ""
    echo "To fold the new row into the full four-model table:"
    echo "  $PYTHON_PATH sl_comparison/compare.py --results_dir sl_comparison/results \\"
    echo "      --csv sl_comparison/results/comparison.csv"
    echo ""
    echo "NOTE: do NOT run sl_comparison/collect_siamese.py to pick this up."
    echo "      It rediscovers the siamese_sl row by scanning siamese_sl/results,"
    echo "      and it would not see this directory — which is deliberate."
fi
echo "============================================================================"
[ "$GRADED" -gt 0 ] || exit 1
