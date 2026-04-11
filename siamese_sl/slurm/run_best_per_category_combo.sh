#!/bin/bash
#SBATCH --job-name=bestcat_combo
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
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
# Step 2: Select the best AUROC embedding per biological category per CV type,
#         reading results from results/pcavar<variance>_<type>_<cv>/results.json.
#         Saves selection to results/best_per_category.json.
#
# Step 3: Train multi-modal combos for each CV type by concatenating the
#         winning embeddings (one per category) with per-modality PCA.
#
# Step 4: Print summary table with single-embedding and multi-modal results.
#
# Submitted automatically by submit_pipeline.sh with:
#   sbatch --dependency=afterany:<STEP1_ARRAY_JOB_ID> \
#       slurm/run_best_per_category_combo.sh
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.sh"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"

sleep 10  # let disk settle after prior job

echo "BEST-PER-CATEGORY — COMBO (Steps 2-4)"
echo "  PCA variance target: $PCA_VARIANCE"
echo ""

# Clean previous combo results and selection JSON
rm -f results/best_per_category.json
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

echo "--- STEP 2: Select best per category ---"

$PYTHON_PATH << PYSELECT
import json, os
from pathlib import Path

PCA_VARIANCE = "$PCA_VARIANCE"

# Parse catalog from shell array
catalog = []
for line in """$(printf '%s\n' "${EMB_CATALOG[@]}")""".strip().split('\n'):
    etype, efile, ecat = line.split(':')
    catalog.append((etype, efile, ecat))

cv_types = $CV_TYPES_PY  # Injected from CV_TYPES in run_best_per_category.conf
results_dir = Path("results")

# Collect PCA benchmark results
scores = {}
for etype, efile, ecat in catalog:
    for cv in cv_types:
        rfile = results_dir / f"pcavar{PCA_VARIANCE}_{etype}_{cv}" / "results.json"
        if rfile.exists():
            try:
                with open(rfile) as _f:
                    d = json.load(_f)["summary"]
                auroc = d["auroc_mean"]
                if auroc is not None:
                    scores[(etype, cv)] = auroc
            except (KeyError, json.JSONDecodeError):
                pass

if not scores:
    print("ERROR: No PCA benchmark results found.")
    print("Step 1 array tasks may have all failed. Check slurm/logs/bestcat_*.out")
    raise SystemExit(1)

# Build lookup
type_to_file = {t: f for t, f, _ in catalog}

# Pick best per category per CV
selection = {}
for cv in cv_types:
    selection[cv] = {}
    cat_candidates = {}
    for etype, efile, ecat in catalog:
        if (etype, cv) in scores:
            cat_candidates.setdefault(ecat, []).append((etype, scores[(etype, cv)]))

    for cat, candidates in cat_candidates.items():
        best_type, best_auroc = max(candidates, key=lambda x: x[1])
        selection[cv][cat] = {
            "type": best_type,
            "file": type_to_file[best_type],
            "auroc": round(best_auroc, 4),
        }

# Print summary
for cv in cv_types:
    print(f"\n{cv.upper()} — best per category (PCA variance={PCA_VARIANCE}):")
    for cat in sorted(selection[cv]):
        info = selection[cv][cat]
        print(f"  {cat:15s} -> {info['type']:15s}  AUROC={info['auroc']:.4f}")

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
    IFS='|' read -r EMB_PATHS COMBO_NAME NUM_MODALITIES < <($PYTHON_PATH << PYEXTRACT
import json, os

with open("results/best_per_category.json") as _f:
    sel = json.load(_f)["$CV"]
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

    set +e
    $PYTHON_PATH train.py \
        --embeddings_paths $EMB_PATHS \
        --sl_path "$SL_PATH" \
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
        --pca_variance $PCA_VARIANCE
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  FAILED (exit code $TRAIN_EXIT)"
        STEP3_FAILED=$((STEP3_FAILED + 1))
    elif [ -f "$OUTPUT_DIR/results.json" ]; then
        AUROC=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d['auroc_mean']; s=d['auroc_std']; print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        AUPR=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d['aupr_mean']; s=d['aupr_std']; print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        PARAMS=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")" 2>/dev/null || echo "N/A")
        echo "  AUROC=$AUROC  AUPR=$AUPR  Params=$PARAMS"
        STEP3_SUCCESSFUL=$((STEP3_SUCCESSFUL + 1))
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

# Step 1 results: single-embedding winners
col_w = 15
header = " | ".join(f"{cv.upper():{col_w}s}" for cv in cv_types)
table_w = col_w + len(cv_types) * (3 + col_w)

print(f"\n--- Single-embedding AUROC (PCA variance={PCA_VARIANCE}) ---")
print(f"{'Category':{col_w}s} | {header}")
print("-" * table_w)
cats = sorted(set(c for cv in sel.values() for c in cv))
for cat in cats:
    row = []
    for cv in cv_types:
        if cat in sel[cv]:
            row.append(f"{sel[cv][cat]['type']} ({sel[cv][cat]['auroc']:.3f})")
        else:
            row.append("N/A")
    row_str = " | ".join(f"{cell:{col_w}s}" for cell in row)
    print(f"{cat:{col_w}s} | {row_str}")

# Step 3 results: multi-modal combo
print(f"\n--- Multi-modal combo results ---")
print(f"{'CV':15s} | {'AUROC':25s} | {'AUPR':25s} | {'Params (nonzero/total)':25s} | {'Sparsity':10s}")
print("-" * 105)
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
                auroc = f"{am:.4f} +/- {astd:.4f}" if am is not None else "N/A"
                aupr = f"{pm:.4f} +/- {pstd:.4f}" if pm is not None else "N/A"
                nz = r.get('nonzero_params') or 0
                tot = r.get('total_params') or 0
                sp = r.get('weight_sparsity') or 0
                params = f"{nz:,}/{tot:,}"
                sparsity = f"{sp:.1f}%"
                print(f"{cv:15s} | {auroc:25s} | {aupr:25s} | {params:25s} | {sparsity:10s}")
                found = True
    if not found:
        print(f"{cv:15s} | {'FAILED':25s} |")
PYSUMMARY

echo ""
echo "============================================================================"
echo "Step 3 (multi-modal): $STEP3_SUCCESSFUL ok, $STEP3_SKIPPED skip, $STEP3_FAILED fail"
echo "Completed at: $(date)"
echo "============================================================================"
