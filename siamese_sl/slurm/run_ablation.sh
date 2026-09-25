#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/ablation_%A_%a.out
#SBATCH --error=slurm/logs/ablation_%A_%a.err
# ============================================================================
# Leave-One-Category-Out Ablation — array worker
# ============================================================================
# One array task trains ONE ablated combo: the best-per-category winners with
# a single category REMOVED, on one CV type. Task count is
# |ABLATE_CATEGORIES| x |CV_TYPES|, laid out CV-major:
#
#   task = cv_index * n_categories + category_index
#
# Everything except the modality list is identical to
# run_best_per_category_combo.sh Step 3 — same folds, same seed, same
# hyperparameters, same preprocessing. If that script's train.py invocation
# changes, this one must change with it or the ablation stops being an
# ablation of the reported model.
#
# The full combo (all |ABLATE_CATEGORIES| categories) is NOT retrained here;
# the summary script reads
# results/best_cat_*_<cv> as the reference row.
#
# Submit via:
#   ./slurm/submit_pipeline.sh --ablation-only
# or directly:
#   sbatch --array=0-17%6 slurm/run_ablation.sh
# ============================================================================

set -e

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_ablation.conf"

N_CAT=${#ABLATE_CATEGORIES[@]}
N_CV=${#CV_TYPES[@]}
TOTAL=$((N_CAT * N_CV))

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID is beyond the $TOTAL configured tasks. Nothing to do."
    exit 0
fi

CV_IDX=$((SLURM_ARRAY_TASK_ID / N_CAT))
CAT_IDX=$((SLURM_ARRAY_TASK_ID % N_CAT))
CV="${CV_TYPES[$CV_IDX]}"
DROP="${ABLATE_CATEGORIES[$CAT_IDX]}"

echo "============================================================================"
echo "ABLATION  task $SLURM_ARRAY_TASK_ID of $TOTAL"
echo "  CV:      $CV"
echo "  Dropped: $DROP"
echo "============================================================================"

if [ ! -f results/best_per_category.json ]; then
    echo "ERROR: results/best_per_category.json missing."
    echo "  Run the best-per-category pipeline first (submit_pipeline.sh --best-cat-only)."
    exit 1
fi

# --- Resolve the surviving modality list ------------------------------------
# Fields separated by | so spaces inside the path list survive the read.
# A hard failure here (missing CV key, corrupt JSON, dropped category absent)
# must stop the task rather than silently train a full combo and be recorded
# as an ablation.
set +e
EXTRACT_OUT=$($PYTHON_PATH << PYEXTRACT
import json, os, sys

with open("results/best_per_category.json") as _f:
    full = json.load(_f)
if "$CV" not in full:
    sys.exit(f"PYEXTRACT: CV key '$CV' missing from best_per_category.json")
sel = full["$CV"]
if "$DROP" not in sel:
    sys.exit(f"PYEXTRACT: category '$DROP' not present for $CV "
             f"(have: {sorted(sel)}) — nothing to ablate")

paths, names = [], []
for cat in sorted(sel):
    if cat == "$DROP":
        continue
    p = "$DATA_DIR/" + sel[cat]["file"]
    if os.path.isfile(p):
        paths.append(p)
        names.append(sel[cat]["type"])
    else:
        print(f"PYEXTRACT: missing embedding file for {cat}: {p}", file=sys.stderr)

print(f"{' '.join(paths)}|{'+'.join(names)}|{len(paths)}")
PYEXTRACT
)
EXTRACT_EXIT=$?
set -e

if [ $EXTRACT_EXIT -ne 0 ] || [ -z "$EXTRACT_OUT" ]; then
    echo "ERROR: PYEXTRACT failed (exit=$EXTRACT_EXIT): $EXTRACT_OUT"
    exit 1
fi
IFS='|' read -r EMB_PATHS COMBO_NAME NUM_MODALITIES <<< "$EXTRACT_OUT"

if [ "${NUM_MODALITIES:-0}" -lt 2 ]; then
    echo "SKIP: only $NUM_MODALITIES modality left after dropping $DROP; a combo needs >= 2."
    exit 0
fi

OUTPUT_DIR="results/ablate_no_${DROP}_${CV}"
echo "  Surviving ($NUM_MODALITIES): $COMBO_NAME"
echo "  Output: $OUTPUT_DIR"

# A stale directory from an earlier ablation must not be mistaken for this
# run's output; train.py purges per-fold artifacts, but an aborted run can
# leave a results.json the summary would happily read.
rm -rf "$OUTPUT_DIR"
$PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

# Post-concat PCA: same precedence as the combo script. See run_ablation.conf
# for why the default deliberately inherits the combo's setting (and what it
# confounds).
if [ -n "${POST_PCA_DIM:-}" ]; then
    POST_PCA_ARGS="--post_pca_dim $POST_PCA_DIM"
    echo "  Post-concat PCA: exact dim=$POST_PCA_DIM"
elif [ -n "${POST_PCA_VARIANCE:-}" ]; then
    POST_PCA_ARGS="--post_pca_variance $POST_PCA_VARIANCE"
    echo "  Post-concat PCA: variance=$POST_PCA_VARIANCE (retained k varies with the modality set)"
else
    POST_PCA_ARGS="--no_post_pca"
    echo "  Post-concat PCA: disabled"
fi
echo ""

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
    --pca_method "${PCA_METHOD:-plain}" \
    --siamese_encoder_type "${SIAMESE_ENCODER_TYPE:-residual}" \
    $POST_PCA_ARGS
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "FAILED: train.py exited $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi
if [ ! -f "$OUTPUT_DIR/results.json" ]; then
    echo "FAILED: no results.json written"
    exit 1
fi

# Record what was ablated, so the summary never has to parse it back out of a
# directory name and a re-run with different categories cannot be misread.
$PYTHON_PATH - <<PYSTAMP
import json
p = "$OUTPUT_DIR/results.json"
with open(p) as f:
    d = json.load(f)
d["ablation"] = {
    "dropped_category": "$DROP",
    "surviving_modalities": "$COMBO_NAME".split("+"),
    "num_modalities": $NUM_MODALITIES,
    "cv_type": "$CV",
    "reference": "results/best_cat_*_$CV (full combo, all categories)",
}
with open(p, "w") as f:
    json.dump(d, f, indent=2)
PYSTAMP

# Checkpoints only after results.json exists — nonzero-parameter counting
# reloads them mid-run.
if [ "${ABLATION_PURGE_CHECKPOINTS:-1}" = "1" ]; then
    rm -rf "$OUTPUT_DIR/checkpoints"
    echo "  Checkpoints removed (ABLATION_PURGE_CHECKPOINTS=1)"
fi
if [ "${ABLATION_PURGE_PREDICTIONS:-0}" = "1" ]; then
    rm -f "$OUTPUT_DIR"/fold_*_predictions.npy
    echo "  Prediction matrices removed (ABLATION_PURGE_PREDICTIONS=1)"
fi

AUPRG=$($PYTHON_PATH -c "import json;d=json.load(open('$OUTPUT_DIR/results.json'))['summary'];v=d.get('auprg_mean');s=d.get('auprg_std');print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
AUPR=$($PYTHON_PATH -c "import json;d=json.load(open('$OUTPUT_DIR/results.json'))['summary'];v=d.get('aupr_mean');s=d.get('aupr_std');print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
echo ""
echo "SUCCESS: $CV without '$DROP'  AUPRG=$AUPRG  AUPR=$AUPR"
