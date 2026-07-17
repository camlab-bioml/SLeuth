#!/bin/bash
#SBATCH --job-name=bestcat_combo
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3,gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/logs/bestcat_combo_%j.out
#SBATCH --error=slurm/logs/bestcat_combo_%j.err

# ============================================================================
# Best-per-Category — Combo (Steps 2-4)
# ============================================================================
# Runs after all Step 1 array tasks complete (dependency: afterany).
#
# Step 2: Select the best AUPR embedding per biological category using
#         single-embedding scores on SELECTION_CV (default cv1, leakage-free).
#         The SAME winners are reused for every CV combo, so cv2/cv3 hard-
#         generalization performance reflects the model architecture without
#         model-selection bias from having peeked at the cv2/cv3 test split.
#         Reads results/pcavar<variance>_<type>_<cv>/results.json and saves
#         the selection to results/best_per_category.json.
#
# Step 3: Train multi-modal combos for each CV type by concatenating the
#         winning embeddings (one per category) with per-modality PCA.
#         On each successful combo, writes results/latest_combo_${CV}.txt
#         pointing at the combo's output dir — consumed by the chained
#         eval_finetuning.sh job submitted from submit_pipeline.sh.
#
# Step 4: Print summary table with single-embedding and multi-modal results.
#
# Submitted automatically by submit_pipeline.sh with:
#   sbatch --dependency=afterany:<STEP1_ARRAY_JOB_ID> \
#       slurm/run_best_per_category_combo.sh
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"

sleep 10  # let disk settle after prior job

echo "BEST-PER-CATEGORY — COMBO (Steps 2-4)"
echo "  PCA variance target: $PCA_VARIANCE"
echo ""

# Clean previous combo results and selection JSON
rm -f results/best_per_category.json
# Also clean the per-CV marker files so a stale marker from a prior run
# doesn't leak into the chained eval_finetuning job when this CV's combo
# fails to train.
rm -f results/latest_combo_*.txt
for d in results/best_cat_*; do
    [ -d "$d" ] && rm -rf "$d"
done
echo "Cleaned previous best-per-category results"
echo ""

# ============================================================================
# STEP 2: Select best embedding per category per CV
# ============================================================================
# Reads results/pcavar<variance>_<type>_<cv>/results.json produced by the array job.
# Writes results/best_per_category.json.

# Build Python list literal from bash CV_TYPES array: ["cv1", "cv2", "cv3"]
# Used by all Python heredocs below so cv_types stays in sync with the .conf.
CV_TYPES_PY="[$(printf '"%s", ' "${CV_TYPES[@]}" | sed 's/, $//')]"

echo "--- STEP 2: Select best per category (selection CV: $SELECTION_CV) ---"

$PYTHON_PATH << PYSELECT
import json, os
from pathlib import Path

PCA_VARIANCE = "$PCA_VARIANCE"
SELECTION_CV = "$SELECTION_CV"

# Parse catalog from shell array
catalog = []
for line in """$(printf '%s\n' "${EMB_CATALOG[@]}")""".strip().split('\n'):
    etype, efile, ecat = line.split(':')
    catalog.append((etype, efile, ecat))

cv_types = $CV_TYPES_PY  # Injected from CV_TYPES in run_best_per_category.conf
results_dir = Path("results")

if SELECTION_CV not in cv_types:
    print(f"ERROR: SELECTION_CV={SELECTION_CV!r} not in CV_TYPES={cv_types}.")
    raise SystemExit(1)

# Collect PCA benchmark results.
# Selection metric MUST match train.py's checkpoint-selection metric:
#   cell-line mode  -> macro AUPRG (auprg_mean; baseline-free, comparable across
#                      heads with very different prevalence),
#   single-output   -> raw macro AUPR (aupr_mean).
# USE_CELL_LINES is injected from the sourced config.conf. Single-output
# results.json has NO auprg_mean, so we read SEL_KEY (never a bare key) and skip
# any None, so the two modes never mix metrics in the argmax below.
USE_CELL_LINES = "$USE_CELL_LINES" == "1"
SEL_KEY = "auprg_mean" if USE_CELL_LINES else "aupr_mean"
SEL_LABEL = "AUPRG" if USE_CELL_LINES else "AUPR"
scores = {}
for etype, efile, ecat in catalog:
    for cv in cv_types:
        rfile = results_dir / f"pcavar{PCA_VARIANCE}_{etype}_{cv}" / "results.json"
        if rfile.exists():
            try:
                with open(rfile) as _f:
                    d = json.load(_f)["summary"]
                aupr = d.get(SEL_KEY)
                if aupr is not None:
                    scores[(etype, cv)] = aupr
            except (KeyError, json.JSONDecodeError):
                pass

if not scores:
    print("ERROR: No PCA benchmark results found.")
    print("Step 1 array tasks may have all failed. Check slurm/logs/bestcat_*.out")
    raise SystemExit(1)

# Build lookup
type_to_file = {t: f for t, f, _ in catalog}

# --- Leakage-free single selection on SELECTION_CV ---
# Pick winners ONCE using SELECTION_CV scores (default cv1). Reuse the same
# winners for every CV combo so cv2/cv3 hard-generalization performance is
# not inflated by having seen those test splits at selection time.
ref_candidates = {}
for etype, efile, ecat in catalog:
    if (etype, SELECTION_CV) in scores:
        ref_candidates.setdefault(ecat, []).append(
            (etype, scores[(etype, SELECTION_CV)]))

if not ref_candidates:
    print(f"ERROR: No single-embedding results for SELECTION_CV={SELECTION_CV}.")
    print(f"Check that Step 1 array tasks for {SELECTION_CV} completed.")
    raise SystemExit(1)

ref_winners = {}
for cat, candidates in ref_candidates.items():
    best_type, best_score = max(candidates, key=lambda x: x[1])
    ref_winners[cat] = {
        "type": best_type,
        "file": type_to_file[best_type],
        "aupr": round(best_score, 4),
    }

# Replicate the leakage-free selection across all CV keys (so the combo step,
# which reads selection["$CV"], gets the same winners regardless of CV).
# The "aupr" field shown per CV is the selected winner's score on THAT CV
# (informational diagnostic), while the chosen "type"/"file" are identical.
selection = {}
for cv in cv_types:
    selection[cv] = {}
    for cat, info in ref_winners.items():
        cv_score = scores.get((info["type"], cv))
        selection[cv][cat] = {
            "type": info["type"],
            "file": info["file"],
            "aupr": round(cv_score, 4) if cv_score is not None else None,
        }
selection["_selection_meta"] = {
    "selection_cv": SELECTION_CV,
    "selection_metric": SEL_KEY,
    "rationale": "leakage-free: same winners reused across CV combos",
}

# Print summary. NOTE: the JSON "aupr" fields above hold whatever SEL_KEY
# selected (AUPRG in cell-line mode); the field name is kept "aupr" for
# backward-compatible reads, while the printed label reflects the real metric.
print(f"\nSelected by {SELECTION_CV.upper()} {SEL_LABEL} (PCA variance={PCA_VARIANCE}):")
print(f"  {'Category':15s} -> {'Type':15s}  " +
      "  ".join(f"{SEL_LABEL}_{cv}" for cv in cv_types))
for cat in sorted(ref_winners):
    parts = [f"  {cat:15s} -> {ref_winners[cat]['type']:15s}"]
    for cv in cv_types:
        cv_score = scores.get((ref_winners[cat]['type'], cv))
        parts.append(f"{cv_score:.4f}" if cv_score is not None else "  N/A ")
    print("  ".join(parts))

out = results_dir / "best_per_category.json"
with open(out, "w") as _f:
    json.dump(selection, _f, indent=2)
print(f"\nSaved to {out}")
PYSELECT

[ -f results/best_per_category.json ] || { echo "ERROR: Selection failed"; exit 1; }
echo ""

# ============================================================================
# STEP 3: Train multi-modal combo for each CV
# ============================================================================
# For each CV type, concatenate the best embedding from each available
# category with per-modality PCA.

echo "--- STEP 3: Train multi-modal combo for each CV ---"
echo ""

STEP3_SUCCESSFUL=0
STEP3_SKIPPED=0
STEP3_FAILED=0

for CV in "${CV_TYPES[@]}"; do
    echo "============================================================================"
    echo "Training best-per-category combo for $CV"
    echo "============================================================================"

    # Extract paths from the selection JSON.
    # Fields separated by | so spaces within EMB_PATHS are preserved.
    # We capture stdout AND exit code separately so a PYEXTRACT crash (e.g.,
    # corrupt selection JSON, missing CV key) is treated as a hard failure
    # instead of silently producing empty fields that look like "0 modalities."
    set +e
    EXTRACT_OUT=$($PYTHON_PATH << PYEXTRACT
import json, os, sys

with open("results/best_per_category.json") as _f:
    full = json.load(_f)
if "$CV" not in full:
    sys.exit(f"PYEXTRACT: CV key '$CV' missing from best_per_category.json")
sel = full["$CV"]
cats = sorted(sel.keys())

paths = []
names = []
for cat in cats:
    p = "$DATA_DIR/" + sel[cat]["file"]
    if os.path.isfile(p):
        paths.append(p)
        names.append(sel[cat]["type"])

combo_name = "+".join(names)
print(f"{' '.join(paths)}|{combo_name}|{len(paths)}")
PYEXTRACT
)
    EXTRACT_EXIT=$?
    set -e

    if [ $EXTRACT_EXIT -ne 0 ] || [ -z "$EXTRACT_OUT" ]; then
        echo "  ERROR: PYEXTRACT failed (exit=$EXTRACT_EXIT). Skipping this CV."
        echo "  Output captured: $EXTRACT_OUT"
        STEP3_FAILED=$((STEP3_FAILED + 1))
        continue
    fi
    IFS='|' read -r EMB_PATHS COMBO_NAME NUM_MODALITIES <<< "$EXTRACT_OUT"

    if [ "${NUM_MODALITIES:-0}" -lt 2 ]; then
        echo "  SKIP: fewer than 2 modalities available"
        STEP3_SKIPPED=$((STEP3_SKIPPED + 1))
        continue
    fi

    SAFE_NAME=$(echo "$COMBO_NAME" | tr '+' '_')
    OUTPUT_DIR="results/best_cat_${SAFE_NAME}_${CV}"

    echo "  Modalities ($NUM_MODALITIES): $COMBO_NAME"
    echo "  PCA variance: $PCA_VARIANCE"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # Create output directory
    $PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

    if [ -n "$POST_PCA_VARIANCE" ]; then
        POST_PCA_ARGS="--post_pca_variance $POST_PCA_VARIANCE"
    else
        POST_PCA_ARGS="--no_post_pca"
    fi

    set +e
    $PYTHON_PATH train.py \
        --embeddings_paths $EMB_PATHS \
        --sl_path "$SL_PATH" \
        ${NEG_PATH:+--neg_pairs_path "$NEG_PATH"} \
        $CELL_LINE_ARGS \
        --output_dir "$OUTPUT_DIR" \
        --cv_type "$CV" \
        $MODEL_ARGS \
        --encoder_dims $ENCODER_DIMS \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --l1_lambdas $L1_LAMBDAS \
        --pd_epsilon $PD_EPSILON \
        --eval_interval $EVAL_INTERVAL \
        --patience $PATIENCE \
        --warmrestart_T0 $WARMRESTART_T0 \
        --warmrestart_Tmult $WARMRESTART_TMULT \
        --num_folds $NUM_FOLDS \
        --pos_neg_ratio $POS_NEG_RATIO \
        --seed $SEED \
        --preprocessing_fit_scope "${PREPROCESSING_FIT_SCOPE:-train}" \
        --pca_method "${PCA_METHOD:-svd}" \
        --siamese_encoder_type "${SIAMESE_ENCODER_TYPE:-residual}" \
        $POST_PCA_ARGS
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  FAILED (exit code $TRAIN_EXIT)"
        STEP3_FAILED=$((STEP3_FAILED + 1))
    elif [ -f "$OUTPUT_DIR/results.json" ]; then
        AUROC=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d['auroc_mean']; s=d['auroc_std']; print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        AUPR=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d['aupr_mean']; s=d['aupr_std']; print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        # AUPRG present only in cell-line mode; .get() -> 'N/A' in single-output.
        AUPRG=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d.get('auprg_mean'); s=d.get('auprg_std'); print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        PARAMS=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")" 2>/dev/null || echo "N/A")
        echo "  AUROC=$AUROC  AUPR=$AUPR  AUPRG=$AUPRG  Params=$PARAMS"
        STEP3_SUCCESSFUL=$((STEP3_SUCCESSFUL + 1))
        # Drop a marker file so the chained eval_finetuning job can locate
        # this freshly-trained combo. Per-CV so each eval can target the
        # matching split.
        echo "$OUTPUT_DIR" > "results/latest_combo_${CV}.txt"
    else
        echo "  FAILED"
        STEP3_FAILED=$((STEP3_FAILED + 1))
    fi
    set -e

    echo ""
done

# ============================================================================
# STEP 4: Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "                 BEST-PER-CATEGORY RESULTS (PCA variance=$PCA_VARIANCE)"
echo "============================================================================"

$PYTHON_PATH << PYSUMMARY
import json
from pathlib import Path

PCA_VARIANCE = "$PCA_VARIANCE"
with open("results/best_per_category.json") as _f:
    sel = json.load(_f)
cv_types = $CV_TYPES_PY  # Injected from CV_TYPES in run_best_per_category.conf
_meta = sel.get("_selection_meta", {})
selection_cv = _meta.get("selection_cv", "unknown")
# Label the winners table by the metric actually used (AUPRG in cell-line mode).
# The per-entry field is still named "aupr" for backward compat, but it holds
# the SEL_KEY value, so display the true metric name here.
SEL_LABEL = "AUPRG" if _meta.get("selection_metric") == "auprg_mean" else "AUPR"

# Step 1 results: selected winner's score across CVs.
# After leakage-free selection, the same winner is chosen for every CV, so the
# table shows "winner_type (score_on_this_cv)" — column-to-column differences
# reveal how the SELECTION_CV-picked winner generalizes.
col_w = 15
header = " | ".join(f"{cv.upper():{col_w}s}" for cv in cv_types)
table_w = col_w + len(cv_types) * (3 + col_w)

print(f"\n--- Single-embedding {SEL_LABEL} of selected winners ---")
print(f"    (selection_cv={selection_cv}, PCA variance={PCA_VARIANCE})")
print(f"{'Category':{col_w}s} | {header}")
print("-" * table_w)
cats = sorted({c for cv in cv_types if cv in sel for c in sel[cv]})
for cat in cats:
    row = []
    for cv in cv_types:
        if cv in sel and cat in sel[cv]:
            entry = sel[cv][cat]
            aupr_str = (f"{entry['aupr']:.3f}"
                        if entry['aupr'] is not None else "N/A")
            row.append(f"{entry['type']} ({aupr_str})")
        else:
            row.append("N/A")
    row_str = " | ".join(f"{cell:{col_w}s}" for cell in row)
    print(f"{cat:{col_w}s} | {row_str}")

# Step 3 results: multi-modal combo. AUPRG is the cell-line selection metric
# (absent in single-output results.json -> rendered N/A via .get()).
print(f"\n--- Multi-modal combo results ---")
print(f"{'CV':15s} | {'AUROC':25s} | {'AUPR':25s} | {'AUPRG':25s} | {'Params (nonzero/total)':25s} | {'Sparsity':10s}")
print("-" * 140)
for cv in cv_types:
    found = False
    for d in sorted(Path("results").iterdir()):
        if d.name.startswith("best_cat_") and d.name.endswith(f"_{cv}"):
            rf = d / "results.json"
            if rf.exists():
                with open(rf) as _f:
                    r = json.load(_f)["summary"]
                am, astd = r.get('auroc_mean'), r.get('auroc_std')
                pm, pstd = r.get('aupr_mean'), r.get('aupr_std')
                gm, gstd = r.get('auprg_mean'), r.get('auprg_std')
                auroc = f"{am:.4f} +/- {astd:.4f}" if am is not None else "N/A"
                aupr = f"{pm:.4f} +/- {pstd:.4f}" if pm is not None else "N/A"
                auprg = f"{gm:.4f} +/- {gstd:.4f}" if gm is not None else "N/A"
                nz = r.get('nonzero_params') or 0
                tot = r.get('total_params') or 0
                sp = r.get('weight_sparsity') or 0
                params = f"{nz:,}/{tot:,}"
                sparsity = f"{sp:.1f}%"
                print(f"{cv:15s} | {auroc:25s} | {aupr:25s} | {auprg:25s} | {params:25s} | {sparsity:10s}")
                found = True
    if not found:
        print(f"{cv:15s} | {'FAILED':25s} | {'N/A':25s} | {'N/A':25s} | {'N/A':25s} | {'N/A':10s}")
PYSUMMARY

echo ""
echo "============================================================================"
echo "Step 3 (multi-modal): $STEP3_SUCCESSFUL ok, $STEP3_SKIPPED skip, $STEP3_FAILED fail"
echo "Completed at: $(date)"
echo "============================================================================"

# Fail loudly if any per-CV combo training failed. Without this, the job
# exits 0, the afterok-eval job fires, and eval falls back to a hardcoded
# MODEL_DIR that was wiped at the top of this script (results/best_cat_* rm).
# The user would see "MODEL_DIR not found" in the eval log with no link to
# the actual cause. Exiting non-zero here breaks that chain cleanly.
if [ $STEP3_FAILED -gt 0 ]; then
    echo ""
    echo "ERROR: $STEP3_FAILED combo training(s) failed. Downstream eval job"
    echo "       (if queued via submit_pipeline.sh) will be cancelled by SLURM"
    echo "       since it depends on this job exiting successfully (afterok)."
    exit 1
fi
