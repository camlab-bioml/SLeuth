#!/bin/bash
#SBATCH --job-name=siamese_best_cat
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm/logs/best_per_category_%j.out
#SBATCH --error=slurm/logs/best_per_category_%j.err

# ============================================================================
# Best-per-Category Multi-Modal Training
# ============================================================================
# 1. Train each available embedding individually with PCA (64 PCs) x 3 CVs
# 2. Pick the best AUROC embedding per category per CV from those PCA results
# 3. Train multi-modal combos using the best-per-category selections
#
# This ensures apples-to-apples comparison: embeddings are compared at the
# same dimensionality they'll have in the multi-modal combo.
#
# Configuration: PCA_DIM and EMB_CATALOG are defined in config.sh.
#   Override PCA at submit time: BEST_CAT_PCA_DIM=128 sbatch slurm/run_best_per_category.sh
#
# Prerequisites: Embedding .pt files must exist (run_embedding_benchmark.sh
#   steps 1-3, or generate_embeddings.py).
#
# Usage:
#   cd siamese_sl
#   sbatch --dependency=afterok:<PHASE1_JOB_ID> slurm/run_best_per_category.sh
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

# PCA_DIM and EMB_CATALOG are defined in config.sh

echo "BEST-PER-CATEGORY MULTI-MODAL TRAINING"
echo "  PCA dim per modality: $PCA_DIM"
echo ""

# ============================================================================
# STEP 1: Train each embedding with PCA x 3 CVs
# ============================================================================
# Results go to results/pca<dim>_<type>_<cv>/results.json
# Embeddings whose native dim <= PCA_DIM are trained without PCA (no-op).

echo "--- STEP 1: Single-embedding training with PCA=$PCA_DIM ---"
echo ""

STEP1_SUCCESSFUL=0
STEP1_FAILED=0
STEP1_SKIPPED=0

for entry in "${EMB_CATALOG[@]}"; do
    IFS=':' read -r ETYPE EFILE ECAT <<< "$entry"
    EMB_FILE="$DATA_DIR/$EFILE"

    if [ ! -f "$EMB_FILE" ]; then
        echo "[$ETYPE] SKIP — $EFILE not found"
        STEP1_SKIPPED=$((STEP1_SKIPPED + ${#CV_TYPES[@]}))
        continue
    fi

    echo "[$ETYPE] Training with PCA=$PCA_DIM..."

    for CV in "${CV_TYPES[@]}"; do
        OUTPUT_DIR="results/pca${PCA_DIM}_${ETYPE}_${CV}"

        # Skip if already completed
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            echo "  [$ETYPE/$CV] Already done, skipping"
            STEP1_SUCCESSFUL=$((STEP1_SUCCESSFUL + 1))
            continue
        fi

        echo "  --- $ETYPE / $CV --- $(date)"

        # Pre-create output dirs (server may not have mkdir in PATH)
        $PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

        set +e
        $PYTHON_PATH train.py \
            --embeddings_paths "$EMB_FILE" \
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
            --pca_dims $PCA_DIM

        if [ -f "$OUTPUT_DIR/results.json" ]; then
            AUROC=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d['auroc_mean']:.4f} +/- {d['auroc_std']:.4f}\")" 2>/dev/null)
            PARAMS=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")" 2>/dev/null)
            echo "  AUROC=$AUROC  Params=$PARAMS"
            STEP1_SUCCESSFUL=$((STEP1_SUCCESSFUL + 1))
        else
            echo "  FAILED"
            STEP1_FAILED=$((STEP1_FAILED + 1))
        fi
        set -e

        # Brief pause between runs to let the disk flush
        sleep 10
    done
    echo ""
done

echo "Step 1 complete: $STEP1_SUCCESSFUL successful, $STEP1_FAILED failed, $STEP1_SKIPPED skipped"
echo ""

# ============================================================================
# STEP 2: Select best embedding per category per CV (from PCA results)
# ============================================================================
# Reads results/pca<dim>_<type>_<cv>/results.json
# Writes results/best_per_category.json

echo "--- STEP 2: Select best per category ---"

$PYTHON_PATH << PYSELECT
import json, os
from pathlib import Path

PCA_DIM = $PCA_DIM

# Parse catalog
catalog = []
for line in """$(printf '%s\n' "${EMB_CATALOG[@]}")""".strip().split('\n'):
    etype, efile, ecat = line.split(':')
    catalog.append((etype, efile, ecat))

cv_types = ["cv1", "cv2", "cv3"]
results_dir = Path("results")

# Collect PCA benchmark results
scores = {}
for etype, efile, ecat in catalog:
    for cv in cv_types:
        rfile = results_dir / f"pca{PCA_DIM}_{etype}_{cv}" / "results.json"
        if rfile.exists():
            try:
                d = json.load(open(rfile))["summary"]
                scores[(etype, cv)] = d["auroc_mean"]
            except (KeyError, json.JSONDecodeError):
                pass

if not scores:
    print("ERROR: No PCA benchmark results found.")
    raise SystemExit(1)

# Build lookup
type_to_file = {t: f for t, f, _ in catalog}
type_to_cat  = {t: c for t, _, c in catalog}

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
    print(f"\n{cv.upper()} — best per category (PCA={PCA_DIM}):")
    for cat in sorted(selection[cv]):
        info = selection[cv][cat]
        print(f"  {cat:15s} -> {info['type']:15s}  AUROC={info['auroc']:.4f}")

out = results_dir / "best_per_category.json"
json.dump(selection, open(out, "w"), indent=2)
print(f"\nSaved to {out}")
PYSELECT

[ -f results/best_per_category.json ] || { echo "ERROR: Selection failed"; exit 1; }
echo ""

# ============================================================================
# STEP 3: Train multi-modal combo for each CV
# ============================================================================
# For each CV type, concatenate the best embedding from each available category
# with per-modality PCA.

echo "--- STEP 3: Train multi-modal combo for each CV ---"
echo ""

STEP3_SUCCESSFUL=0
STEP3_FAILED=0

for CV in "${CV_TYPES[@]}"; do
    echo "============================================================================"
    echo "Training best-per-category combo for $CV"
    echo "============================================================================"

    # Extract paths and build PCA dims from the selection JSON
    # Fields separated by | so spaces within EMB_PATHS/PCA_DIMS are preserved
    IFS='|' read -r EMB_PATHS PCA_DIMS COMBO_NAME NUM_MODALITIES < <($PYTHON_PATH << PYEXTRACT
import json, os

sel = json.load(open("results/best_per_category.json"))["$CV"]
cats = sorted(sel.keys())

paths = []
pca_dims = []
names = []
for cat in cats:
    p = "$DATA_DIR/" + sel[cat]["file"]
    if os.path.isfile(p):
        paths.append(p)
        pca_dims.append("$PCA_DIM")
        names.append(sel[cat]["type"])

combo_name = "+".join(names)
print(f"{' '.join(paths)}|{' '.join(pca_dims)}|{combo_name}|{len(paths)}")
PYEXTRACT
    )

    if [ "${NUM_MODALITIES:-0}" -lt 2 ]; then
        echo "  SKIP: fewer than 2 modalities available"
        STEP3_FAILED=$((STEP3_FAILED + 1))
        continue
    fi

    SAFE_NAME=$(echo "$COMBO_NAME" | tr '+' '_')
    OUTPUT_DIR="results/best_cat_${SAFE_NAME}_${CV}"

    echo "  Modalities ($NUM_MODALITIES): $COMBO_NAME"
    echo "  PCA: $PCA_DIMS"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # Pre-create output dirs (server may not have mkdir in PATH)
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
        --pca_dims $PCA_DIMS

    if [ -f "$OUTPUT_DIR/results.json" ]; then
        AUROC=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d['auroc_mean']:.4f} +/- {d['auroc_std']:.4f}\")" 2>/dev/null)
        AUPR=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d['aupr_mean']:.4f} +/- {d['aupr_std']:.4f}\")" 2>/dev/null)
        PARAMS=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")" 2>/dev/null)
        echo "  AUROC=$AUROC  AUPR=$AUPR  Params=$PARAMS"
        STEP3_SUCCESSFUL=$((STEP3_SUCCESSFUL + 1))
    else
        echo "  FAILED"
        STEP3_FAILED=$((STEP3_FAILED + 1))
    fi
    set -e

    # Brief pause between runs to let the disk flush
    sleep 10
    echo ""
done

# ============================================================================
# STEP 4: Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "                 BEST-PER-CATEGORY RESULTS (PCA=$PCA_DIM)"
echo "============================================================================"

$PYTHON_PATH << PYSUMMARY
import json
from pathlib import Path

PCA_DIM = $PCA_DIM
sel = json.load(open("results/best_per_category.json"))
cv_types = ["cv1", "cv2", "cv3"]

# Step 1 results: single-embedding with PCA
print("\n--- Single-embedding AUROC (PCA={}) ---".format(PCA_DIM))
print(f"{'Category':15s} | {'CV1':15s} | {'CV2':15s} | {'CV3':15s}")
print("-" * 70)
cats = sorted(set(c for cv in sel.values() for c in cv))
for cat in cats:
    row = []
    for cv in cv_types:
        if cat in sel[cv]:
            row.append(f"{sel[cv][cat]['type']} ({sel[cv][cat]['auroc']:.3f})")
        else:
            row.append("N/A")
    print(f"{cat:15s} | {row[0]:15s} | {row[1]:15s} | {row[2]:15s}")

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
                r = json.load(open(rf))["summary"]
                auroc = f"{r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}"
                aupr = f"{r['aupr_mean']:.4f} +/- {r['aupr_std']:.4f}"
                nz = r.get('nonzero_params', 0)
                tot = r.get('total_params', 0)
                sp = r.get('weight_sparsity', 0)
                params = f"{nz:,}/{tot:,}"
                sparsity = f"{sp:.1f}%"
                print(f"{cv:15s} | {auroc:25s} | {aupr:25s} | {params:25s} | {sparsity:10s}")
                found = True
    if not found:
        print(f"{cv:15s} | {'FAILED':25s} |")
PYSUMMARY

echo ""
echo "============================================================================"
echo "Step 1 (single PCA): $STEP1_SUCCESSFUL ok, $STEP1_FAILED fail, $STEP1_SKIPPED skip"
echo "Step 3 (multi-modal): $STEP3_SUCCESSFUL ok, $STEP3_FAILED fail"
echo "Completed at: $(date)"
echo "============================================================================"
