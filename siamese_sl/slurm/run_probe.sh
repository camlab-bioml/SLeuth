#!/bin/bash
#SBATCH --job-name=bilinear_probe
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request), like every siamese GPU job. No
# orchestrator submits this script, so GPU_NODES never reaches it: pass
# --nodelist on the sbatch command line to move it.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
# One task per CV: task i trains and grades CV_TYPES[i] (config.conf).
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/bilinear_probe_%A_%a.out
#SBATCH --error=slurm/logs/bilinear_probe_%A_%a.err

# ============================================================================
# Bilinear probe (reviewer 22DY) — train + grade, one CV per array task
# ============================================================================
# "A simple bilinear probe on the same features": does SLeuth's encoder / head
# add anything over
#
#     logit(i, j) = tau * (W x_i)ᵀ (W x_j) + b
#
# W one bias-free Linear map from the preprocessed input features x to the
# combo's latent width (32), no hidden layer, no input offset, no projection
# bias, no per-gene / pair term, eps = 0, one pooled head (OR label). It is
# SiameseSLMultiCell trained by train.py with --encoder_dims 32 --pd_epsilon 0
# --no_gene_term --no_pair_features --no_per_head_last_layer_bias
# --no-last-layer-bias --no_input_bias --cell_lines pooled --kernel pd and one
# --l1_lambdas value (the combo's first). Parameters: 32 * k + 2, k the
# fold's input width (149,378 at k = 4,668).
#
# Same pairs, same features, same training settings: for each CV the job reads
# PCA arm none's pd:pooled combo, the one named by
#     results/var_pd_pooled__latest_combo_<cv>.txt
# (config.conf siamese_variant_prefix pd:pooled), and reuses its config.json:
# embeddings_paths (same order), fold_modalities (per-fold selection; absent in
# the global mode), selection_frac / selection_seed, sl / neg paths, folds_dir,
# preprocessing, epochs, lr, batch size, weight decay, eval interval, patience,
# restarts, seed, select_metric, log_temperature (siamese_sl/probe_args.py,
# which refuses a combo that is not arm none's pd:pooled one of that CV).
# A CV whose marker is missing is REFUSED (no training, exit 1): the combo job
# deletes the markers when it starts and writes each one when that combo has
# trained, so a missing marker means there are no current features to probe.
#
# Outputs per CV:
#   siamese_sl/results/bilinear_probe_<cv>/   train.py output dir (config.json,
#       results.json, checkpoints/, fold_<k>_<cv>_predictions.npy,
#       model_params.json); removed and rebuilt by each run
#   sl_comparison/results/BilinearProbe_<cv>.json   the row, graded by
#       sl_comparison/evaluate_model.py --model_name BilinearProbe exactly as
#       every other row, plus trainable_params_fold_* (per-fold sizes) and
#       probe_source (the marker, combo dir and sha256 of its config.json)
# The row is a SHARED row, like DegreePrior's: compare.py reads it from
# sl_comparison/results/ for every arm (compare.is_siamese_row is False, so
# collect_and_regrade_siamese.sh never moves or regrades it), paper_tables.py
# lists it in arm none's tab_tiers as kind "baseline", and bootstrap_ci.py
# finds its matrices in siamese_sl/results/bilinear_probe_<cv>/.
#
# Stale rows. Each task FIRST moves its previous BilinearProbe_<cv>.json to
# sl_comparison/results/stale_shared/ (before any check can exit), so a failed
# or refused task leaves the row MISSING, never an older one. Nothing else
# moves or rebuilds it: not run_shared_models.sh (its own NAME only), not the
# root orchestrator's shared-row cleanup (SLGNN / NSF4SL / SLMGAE /
# DegreePrior), not collect_and_regrade_siamese.sh (the siamese_sl* rows), not
# the combo job. After a retrain of arm none's combos or of the folds the old
# row stays in the table until this job runs again: evaluate_model.py's fold
# stamp and paper_tables' fold check catch a fold change, a combo retrain on
# the same folds is caught only by probe_source (compare it with the marker).
#
# The root orchestrator submits it in arm none (afterok on that arm's combo
# job; 2026-11-13). After submit_pipeline.sh alone, submit BY HAND from
# siamese_sl/ (siamese_sl/slurm/README.md gives the order):
#   cd siamese_sl && sbatch --export=ALL slurm/run_probe.sh                # cv1-cv3
#   cd siamese_sl && sbatch --export=ALL --array=2 slurm/run_probe.sh      # cv3 only
# after a retrain, once arm none's combos have finished (the combo job of arm
# none, bestcat_combo, writes every variant's markers):
#   cd siamese_sl && sbatch --dependency=afterok:<arm none bestcat_combo job id> \
#       --export=ALL slurm/run_probe.sh
# then refresh arm none's table and paper tables, from the repo root:
#   PCA_ARM=none sbatch --dependency=afterany:<probe job id> --export=ALL \
#       sl_comparison/slurm/run_compare.sh
#   PCA_ARM=none sbatch --dependency=afterany:<compare job id> --export=ALL \
#       sl_comparison/slurm/run_paper_tables.sh
# Without SLURM (a GPU node, from siamese_sl/): PROBE_CVS="cv1 cv2 cv3" bash slurm/run_probe.sh
# Exit 0 when every CV of the task was trained and graded, else 1.
# ============================================================================

set -e
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SUBMIT_DIR"
# The probe reads arm none's features whatever arm the submitting shell
# exports (config.conf derives RESULTS_ROOT from PCA_ARM).
if [ -n "${PCA_ARM:-}" ] && [ "$PCA_ARM" != none ]; then
    echo "NOTE: PCA_ARM=$PCA_ARM ignored: the bilinear probe is trained on PCA arm none's features."
fi
export PCA_ARM=none
source "$SUBMIT_DIR/slurm/config.conf"
set +e

ROOT="$(dirname "$SUBMIT_DIR")"
PROBE_ROW="BilinearProbe"
VARIANT="pd:pooled"
MARKER_PREFIX=$(siamese_variant_prefix "$VARIANT")
SHARED_CMP="sl_comparison/results"      # repo-root relative; every arm's shared rows
TS=$(date +%Y%m%dT%H%M%S)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUN_CVS="${CV_TYPES[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$RUN_CVS" ]; then
        echo "ERROR: array task $SLURM_ARRAY_TASK_ID has no CV (CV_TYPES: ${CV_TYPES[*]})"
        exit 1
    fi
else
    RUN_CVS="${PROBE_CVS:-${CV_TYPES[*]}}"
fi

echo "Bilinear probe: tau * (W x_i)^T (W x_j) + b on PCA arm none's $VARIANT combo features"
echo "  CVs: $RUN_CVS   markers: $RESULTS_ROOT/${MARKER_PREFIX}latest_combo_<cv>.txt"
echo ""

FAILED=()
for CV in $RUN_CVS; do
    echo "============================================================================"
    echo "BilinearProbe / $CV"
    echo "============================================================================"
    ROW_JSON="$ROOT/$SHARED_CMP/${PROBE_ROW}_${CV}.json"
    OUT_DIR="$RESULTS_ROOT/bilinear_probe_${CV}"
    MARKER="$RESULTS_ROOT/${MARKER_PREFIX}latest_combo_${CV}.txt"

    # FIRST, before any check can stop this CV: the previous row goes aside.
    if [ -f "$ROW_JSON" ]; then
        mkdir -p "$ROOT/$SHARED_CMP/stale_shared"
        mv -f "$ROW_JSON" "$ROOT/$SHARED_CMP/stale_shared/${PROBE_ROW}_${CV}.${TS}.json"
        echo "  moved previous row aside: $SHARED_CMP/stale_shared/${PROBE_ROW}_${CV}.${TS}.json"
    fi

    if [ ! -f "$MARKER" ]; then
        echo "  REFUSED: no $MARKER — arm none's $VARIANT combo of $CV has not trained"
        echo "           (or its combo job is running). Nothing trained; no $PROBE_ROW/$CV row."
        FAILED+=("$CV")
        continue
    fi

    # The probe's train.py arguments from the combo's config.json (NUL-separated:
    # paths and the fold_modalities JSON stay single words).
    ARGS=()
    ARGS_FILE=$(mktemp)
    if ! $PYTHON_PATH probe_args.py --cv "$CV" --marker "$MARKER" \
            --output_dir "$OUT_DIR" > "$ARGS_FILE" 2> "$ARGS_FILE.err"; then
        echo "  REFUSED: $(cat "$ARGS_FILE.err")"
        echo "           Nothing trained; no $PROBE_ROW/$CV row."
        rm -f "$ARGS_FILE" "$ARGS_FILE.err"
        FAILED+=("$CV")
        continue
    fi
    while IFS= read -r -d '' _w; do ARGS+=("$_w"); done < "$ARGS_FILE"
    rm -f "$ARGS_FILE" "$ARGS_FILE.err"
    if [ ${#ARGS[@]} -eq 0 ]; then
        echo "  REFUSED: probe_args.py printed no arguments"
        FAILED+=("$CV")
        continue
    fi
    COMBO_DIR=$($PYTHON_PATH probe_args.py --cv "$CV" --marker "$MARKER" --print_combo)
    echo "  combo: $COMBO_DIR"
    echo "  train.py ${ARGS[*]}"
    echo ""

    # train.py needs the output dir and its checkpoints/ to exist; the previous
    # probe of this CV goes first, so nothing of it can be graded as this run's.
    rm -rf "$OUT_DIR"
    if ! mkdir -p "$OUT_DIR/checkpoints"; then
        echo "  FAILED: could not create $OUT_DIR/checkpoints"
        FAILED+=("$CV")
        continue
    fi

    $PYTHON_PATH train.py "${ARGS[@]}"
    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ] || [ ! -f "$OUT_DIR/results.json" ]; then
        echo "  FAILED: train.py exited $TRAIN_EXIT (results.json: $([ -f "$OUT_DIR/results.json" ] && echo present || echo absent))"
        FAILED+=("$CV")
        continue
    fi

    # Grade with the shared evaluator, from the repo root (its default
    # --folds_dir sl_comparison/folds), exactly as every other row.
    NPRED=$(ls "$OUT_DIR"/fold_*_"$CV"_predictions.npy 2>/dev/null | wc -l | tr -d ' ')
    NFOLD=$(ls -d "$ROOT/sl_comparison/folds/$CV"/fold_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPRED" -ne "$NFOLD" ] || [ "$NPRED" -eq 0 ]; then
        echo "  FAILED: $NPRED score matrices for $NFOLD folds in $OUT_DIR; not graded"
        FAILED+=("$CV")
        continue
    fi
    echo ""
    echo "--- $PROBE_ROW / $CV (shared evaluator, $NPRED folds) ---"
    if ! (cd "$ROOT" && $PYTHON_PATH sl_comparison/evaluate_model.py \
            --model_name "$PROBE_ROW" --cv_type "$CV" \
            --pred_dir "siamese_sl/$OUT_DIR" \
            --out "$SHARED_CMP/${PROBE_ROW}_${CV}.json"); then
        echo "  FAILED: grading $PROBE_ROW/$CV"
        rm -f "$ROW_JSON"
        FAILED+=("$CV")
        continue
    fi

    # Per-fold sizes (the input width follows the combo's per-fold modalities)
    # and which combo the probe stands next to. A failure costs the row: a
    # probe row that cannot say which features it read is not published.
    if ! $PYTHON_PATH - "$OUT_DIR/results.json" "$ROW_JSON" "$MARKER" "$COMBO_DIR" <<'PYEOF'
import hashlib, json, os, sys
import numpy as np
res_path, row_path, marker, combo = sys.argv[1:5]
res = json.load(open(res_path))
tp = [m.get("total_params") for m in res.get("fold_metrics") or []]
if not tp or not all(isinstance(t, int) and not isinstance(t, bool) for t in tp):
    sys.exit(f"fold_metrics[].total_params missing or not integers: {tp!r}")
row = json.load(open(row_path))
s = row["summary"]
s["trainable_params_fold_mean"] = float(np.mean(tp))
s["trainable_params_fold_std"] = float(np.std(tp))
s["trainable_params_fold_min"] = int(min(tp))
s["trainable_params_fold_max"] = int(max(tp))
s["trainable_params_fold_n"] = len(tp)
cfg = os.path.join(combo, "config.json")
s["probe_source"] = {
    "variant": "pd:pooled", "pca_arm": "none", "marker": marker,
    "combo_dir": combo,
    "combo_config_sha256": hashlib.sha256(open(cfg, "rb").read()).hexdigest(),
    # results.json too (2026-11-13): config.json is the same bytes for a
    # retrain with the same settings (new embeddings, new seeds), so it
    # alone cannot tell that combo from the one the probe read.
    "combo_results_sha256": hashlib.sha256(
        open(os.path.join(combo, "results.json"), "rb").read()).hexdigest(),
    "probe_dir": os.path.dirname(res_path),
}
tmp = row_path + ".tmp"
try:
    with open(tmp, "w") as f:
        json.dump(row, f, indent=2)
    os.replace(tmp, row_path)
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
print(f"  per-fold trainable params: {np.mean(tp):,.1f} ({np.std(tp):,.1f}) over {len(tp)} folds")
PYEOF
    then
        echo "  FAILED: could not record the probe's sizes / source; row removed"
        rm -f "$ROW_JSON"
        FAILED+=("$CV")
        continue
    fi
    echo "  row: $SHARED_CMP/${PROBE_ROW}_${CV}.json"
    echo ""
done

echo "============================================================================"
echo "Completed at: $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "NOT trained / graded: ${FAILED[*]} (no $PROBE_ROW row for them)"
    echo "============================================================================"
    exit 1
fi
echo "Rerun arm none's compare and paper tables to publish the row:"
echo "  PCA_ARM=none sbatch --export=ALL sl_comparison/slurm/run_compare.sh   (from the repo root)"
echo "============================================================================"
exit 0
