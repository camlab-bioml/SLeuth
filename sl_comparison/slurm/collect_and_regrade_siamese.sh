#!/bin/bash
# Collect siamese_sl into the comparison results dir, then RE-GRADE it with the
# shared evaluator wherever its prediction matrices exist.
#
# Sourced (not executed) by both run_shared_models.sh and run_compare.sh so the
# two entry points cannot drift. run_compare.sh is the DAG's terminal job and
# re-runs collection; if it only ran collect_siamese.py it would overwrite the
# shared-evaluator JSON that run_shared_models.sh had just written, and the
# published comparison.csv would still show siamese at its own prevalence —
# exactly the mismatch this whole path exists to remove.
#
# Expects: PYTHON_PATH, CVS, and cwd = repo root.

echo "--- collecting siamese_sl results ---"
$PYTHON_PATH sl_comparison/collect_siamese.py \
    --discover_dir "${SIAMESE_DIR:-siamese_sl/results}" \
    --out_dir sl_comparison/results || true

# An explicit per-CV pin overrides discovery. Applied AFTER discovery so it
# wins, and it also refreshes the marker so the re-grade below targets the
# pinned run rather than the auto-discovered one.
for CV in $CVS; do
    # `eval`, not ${!VAR}: indirect expansion is a bash-ism. Under zsh it raises
    # "bad substitution", which aborts the rest of THIS file — including the
    # re-grade loop below — while the collection above has already succeeded.
    # The result is a siamese row scored on its own evaluator, carrying a note
    # that says so, and no error anyone would notice. The SLURM jobs run bash so
    # they were never affected, but anyone sourcing this by hand was.
    eval "SRES=\${SIAMESE_RESULTS_${CV}:-}"
    if [ -n "$SRES" ] && [ -f "$SRES" ]; then
        $PYTHON_PATH sl_comparison/collect_siamese.py --siamese_results "$SRES" \
            --cv_type "$CV" --out "sl_comparison/results/siamese_sl_$CV.json" \
            && dirname "$SRES" > "sl_comparison/results/siamese_selected_$CV.txt"
    fi
done

# Re-grade with the SAME evaluator, test edges and gene universe as the peers.
# This is what makes the `baseline` column identical across all four rows.
for CV in $CVS; do
    MARKER="sl_comparison/results/siamese_selected_$CV.txt"
    [ -f "$MARKER" ] || continue
    SDIR="$(cat "$MARKER")"
    NPRED=$(ls "$SDIR"/fold_*_"$CV"_predictions.npy 2>/dev/null | wc -l | tr -d ' ')
    NFOLD=$(ls -d sl_comparison/folds/"$CV"/fold_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPRED" -eq 0 ]; then
        echo "WARN: no prediction matrices in $SDIR for $CV — siamese_sl/$CV stays"
        echo "      on its OWN folds and evaluator; its baseline will NOT match the"
        echo "      peers. Check the siamese run used --folds_dir (SHARED_FOLDS)."
        continue
    fi
    if [ "$NPRED" -ne "$NFOLD" ]; then
        # A partial fold set would silently give siamese a mean over fewer folds
        # than its peers, with every *_std collapsed toward 0. Refuse rather than
        # publish a row that looks comparable and is not.
        echo "WARN: siamese_sl/$CV has $NPRED of $NFOLD prediction matrices"
        echo "      (retrieval pass likely failed on some folds — grep the siamese"
        echo "      log for 'retrieval metrics skipped'). REFUSING to re-grade a"
        echo "      partial fold set; keeping the collect_siamese row instead."
        continue
    fi
    echo ""; echo "--- siamese_sl / $CV (shared evaluator, $NPRED folds) ---"
    # --carry_from is the SAME path we write: the re-grade replaces every
    # metric but must not lose the annotations collect_siamese.py just wrote
    # (the selection-budget audit and the retrieval note). Dropping them is how
    # compare.py ended up pointing at a `note` field that no longer existed.
    $PYTHON_PATH sl_comparison/evaluate_model.py --model_name siamese_sl \
        --cv_type "$CV" --pred_dir "$SDIR" \
        --carry_from "sl_comparison/results/siamese_sl_$CV.json" \
        --out "sl_comparison/results/siamese_sl_$CV.json" \
        || echo "WARN: shared re-grade failed for siamese_sl/$CV; keeping the
      collect_siamese output (baseline will differ from the peers)"
done
