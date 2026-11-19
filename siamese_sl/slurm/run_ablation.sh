#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
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
# a single category REMOVED, on one CV type, for one model variant. Task count
# is |ABLATE_CATEGORIES| x |CV_TYPES| x |SIAMESE_VARIANTS|, laid out
# variant-major, then CV:
#
#   task = (variant_index * n_cvs + cv_index) * n_categories + category_index
#
# (with only the baseline variant this is the historical CV-major layout).
# Each variant is ablated against ITS OWN reference combo,
# <arm root>/<prefix>best_cat_*_<cv>, and writes
# <arm root>/<prefix>ablate_no_<category>_<cv> — prefix "" for the baseline
# pd:per_head, "var_<kernel>_<cell_lines>__" otherwise; <arm root> is the PCA
# arm's results root, RESULTS_ROOT (config.conf: results/ for arm none,
# results/pca_<arm>/ otherwise), and the ablated combo trains with the arm's
# PCA (PCA_MULTI_ARGS), like its reference.
#
# Everything except the modality list is identical to
# run_best_per_category_combo.sh Step 3 — same folds, same seed, same
# hyperparameters, same preprocessing. If that script's train.py invocation
# changes, this one must change with it or the ablation stops being an
# ablation of the reported model.
#
# The full combo (all |ABLATE_CATEGORIES| categories) is NOT retrained here;
# the summary script reads the combo named by
# <arm root>/<prefix>latest_combo_<cv>.txt as the variant's reference row.
#
# Submit via:
#   bash slurm/submit_pipeline.sh --ablation-only       (every arm of PCA_ARMS)
# or directly from siamese_sl/ (6 categories x 3 CVs x 4 variants = 72 tasks
# per PCA arm), with ONE RUN_TOKEN exported for the array AND its summary:
# without it every task mints its own (config.conf) and the summary reports
# every row as a run_token MISMATCH with no delta. PCA_ARM (default none) picks
# the arm, for the array and its summary alike.
#   export RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$" PCA_ARM=none
#   ABL=$(sbatch --parsable --export=ALL --array=0-71%6 slurm/run_ablation.sh)
#   sbatch --export=ALL --dependency=afterany:$ABL slurm/run_ablation_summary.sh
# paper_tables.py checks the ablation dirs' run_token.txt against its
# --ablation_run_token, else RUN_TOKEN, else best_per_category.json
# _selection_meta.run_token (the token the single-modality dirs are checked
# against too), and the root
# --skip-siamese rebuild sets RUN_TOKEN to the selection run's token. So for
# tab:loco either give paper_tables.py this token on its command line, from
# the repo root:
#   sbatch sl_comparison/slurm/run_paper_tables.sh --ablation_run_token <it>
# from a shell where RUN_TOKEN is unset or the selection run's (sbatch passes
# the shell's environment on by default, as --export=ALL: the export above
# would reach the tables job), never as RUN_TOKEN=<it>: that is also the
# expected token of the single-modality dirs when the selection records none,
# so those would go stale. Or export the selection run's
# _selection_meta.run_token here instead AND first remove the previous
# <arm root>/<prefix>ablate_no_* dirs (submit_pipeline.sh --ablation-only does
# both): under a shared token a task that never starts would pass its previous
# run's directory off as this run's.
# ============================================================================

set -e

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_ablation.conf"

# The removed ablation-only width pin (see run_ablation.conf): stop rather than
# silently run the variance-target ablation the user did not ask for.
if [ -n "${ABLATION_POST_PCA_DIM:-}" ]; then
    echo "ERROR: ABLATION_POST_PCA_DIM is no longer supported: the arch-signature"
    echo "  check refuses an ablation whose post-concat PCA differs from its"
    echo "  reference combo. Pin one width for the combo AND the ablation with"
    echo "  PCA_JOINT_DIM=<k> in a PCA arm with the joint PCA (joint or both;"
    echo "  config.conf, PCA arms) instead: ONE count for both arms, below the"
    echo "  narrowest width their joint stage meets, the ablations' included"
    echo "  (slurm/run_ablation.conf, Input width); train.py refuses one at or"
    echo "  above it."
    exit 1
fi

N_CAT=${#ABLATE_CATEGORIES[@]}
N_CV=${#CV_TYPES[@]}
N_VAR=${#SIAMESE_VARIANTS[@]}
TOTAL=$((N_CAT * N_CV * N_VAR))

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TOTAL" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID is beyond the $TOTAL configured tasks. Nothing to do."
    exit 0
fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
VAR_IDX=$((TASK / (N_CAT * N_CV)))
CV_IDX=$(((TASK % (N_CAT * N_CV)) / N_CAT))
CAT_IDX=$((TASK % N_CAT))
VARIANT="${SIAMESE_VARIANTS[$VAR_IDX]}"
VPREFIX=$(siamese_variant_prefix "$VARIANT")
VARIANT_ARGS=$(siamese_variant_args "$VARIANT")
CV="${CV_TYPES[$CV_IDX]}"
DROP="${ABLATE_CATEGORIES[$CAT_IDX]}"

echo "============================================================================"
echo "ABLATION  task $SLURM_ARRAY_TASK_ID of $TOTAL"
echo "  PCA arm: $PCA_ARM ($(pca_arm_label "$PCA_ARM")) -> $RESULTS_ROOT/"
echo "  Variant: $VARIANT  ($VARIANT_ARGS)"
echo "  CV:      $CV"
echo "  Dropped: $DROP"
echo "  ARCH_ARGS: ${ARCH_ARGS:-(none; train.py defaults)}"
echo "============================================================================"

if [ ! -f "$RESULTS_ROOT/best_per_category.json" ]; then
    echo "ERROR: $RESULTS_ROOT/best_per_category.json missing (PCA arm $PCA_ARM)."
    # An arm that ranks its pair partner's single-embedding runs (STEP1_ROOT
    # is not its own root) must keep doing so: PCA_ARMS=<this arm> alone made
    # it its own owner, and the printed retrain trained a Step-1 array of its
    # own, so the pair's two arms no longer shared one modality set
    # (2026-10-24).
    # Each line carries this job's PCA targets, variants and USE_CELL_LINES (a
    # retrain at other targets fails the ablation's signature check) and keeps
    # the venv; the combo line ranks on the selection CV the pair's owner was
    # ranked on (its best_per_category.json), not this job's default
    # (2026-10-24).
    _set="$(pca_targets_env) USE_CELL_LINES=$USE_CELL_LINES SIAMESE_VARIANTS_LIST=\"${SIAMESE_VARIANTS[*]}\""
    _arms=$(pca_pair_closed "$PCA_ARMS" "$PCA_ARM")
    if [ "$STEP1_ROOT" != "$RESULTS_ROOT" ]; then
        _scv=$(sed -n 's/.*"selection_cv": *"\([^"]*\)".*/\1/p' "$STEP1_ROOT/best_per_category.json" 2>/dev/null | head -n 1)
        _scv=${_scv:-$SELECTION_CV}
        echo "  PCA arm $PCA_ARM ranks the single-embedding runs in $STEP1_ROOT/ (its pair's)."
        echo "  Retrain its selection and combos on those, from siamese_sl/, by resubmitting its"
        echo "  combo job under the token of those runs (cat $STEP1_ROOT/pcavar*_<type>_<cv>/run_token.txt):"
        echo "    RUN_TOKEN=<that token> PCA_ARM=$PCA_ARM PCA_ARMS='$PCA_ARMS' ${_scv:+SELECTION_CV=$_scv }$_set sbatch --export=ALL slurm/run_best_per_category_combo.sh"
        echo "  then this ablation again; or retrain every arm of the run (pair kept):"
        echo "    $_set PCA_ARMS=\"$PCA_ARMS\" bash slurm/submit_pipeline.sh --best-cat-only --skip-generate --skip-reset-env"
    else
        echo "  Run the best-per-category pipeline of this arm first (with the other arm of its pair, if"
        echo "  the run has it: they share its single-embedding runs), from siamese_sl/:"
        echo "    $_set PCA_ARMS=\"$_arms\" bash slurm/submit_pipeline.sh --best-cat-only --skip-generate --skip-reset-env"
    fi
    exit 1
fi

# An ablation is read against THIS variant's full combo, and that combo must be
# the one built from the CURRENT winners: a variant left out of the latest combo
# run (e.g. --ablation-only after a run that trained other variants) keeps an
# older <arm root>/<prefix>best_cat_* whose modality set need not match
# best_per_category.json, and the summary would pair the two anyway. Require the
# exact directory the combo script writes (its SAFE_NAME rule: every winner, in
# sorted-category order; the combo refuses to train with a winner's file
# missing). An empty REF_NAME (missing CV key, corrupt JSON) falls through to
# PYEXTRACT, which fails hard below.
# The reference combo's name, as the combo job built it (fold_selection.py
# combo_spec: the winners of the sorted categories, or, under a per_fold
# selection, the union of the CV's per-fold winners).
REF_NAME=$($PYTHON_PATH fold_selection.py combo_spec \
    --selection "$RESULTS_ROOT/best_per_category.json" --cv "$CV" \
    2>/dev/null | cut -d'|' -f2 | tr '+' '_' || true)
# Spelled exactly as the combo job writes its OUTPUT_DIR (and so its marker
# text): "$RESULTS_ROOT/<prefix>best_cat_<names>_<cv>", compared as a string
# below.
REF_DIR="$RESULTS_ROOT/${VPREFIX}best_cat_${REF_NAME}_${CV}"
if [ -n "$REF_NAME" ] && [ ! -f "$REF_DIR/results.json" ]; then
    echo "SKIP: no reference combo $REF_DIR for variant $VARIANT"
    echo "  (none trained, or trained on other winners than $RESULTS_ROOT/best_per_category.json)."
    exit 0
fi
# ...and it must be the combo the latest combo job trained: that job exits 0
# when one (variant, CV) fails, and a failed combo leaves no marker but may
# leave an earlier run's $REF_DIR/results.json in place.
REF_MARKER="$RESULTS_ROOT/${VPREFIX}latest_combo_${CV}.txt"
if [ -n "$REF_NAME" ] && { [ ! -f "$REF_MARKER" ] || \
        [ "$(head -n 1 "$REF_MARKER" | tr -d '\r')" != "$REF_DIR" ]; }; then
    echo "SKIP: $REF_MARKER does not name $REF_DIR: variant $VARIANT's $CV combo"
    echo "  did not train in the latest combo job (see its log)."
    exit 0
fi

# --- Resolve the surviving modality list ------------------------------------
# Fields separated by | so spaces inside the path list survive the read.
# A hard failure here (missing CV key, corrupt JSON, dropped category absent, a
# surviving winner's embedding file missing) must stop the task rather than
# silently train some other combo and have it recorded as an ablation.
# The reference's modalities minus $DROP (fold_selection.py combo_spec
# --drop): the other categories' winners, or, under a per_fold selection,
# each fold's own winners minus that category's pick, over their union
# (FOLD_MAP, train.py --fold_modalities). Without every file this run is not
# the reference minus ONE category (--data_dir check).
set +e
EXTRACT_OUT=$($PYTHON_PATH fold_selection.py combo_spec \
    --selection "$RESULTS_ROOT/best_per_category.json" --cv "$CV" \
    --drop "$DROP" --data_dir "$DATA_DIR" 2>&1)
EXTRACT_EXIT=$?
set -e

if [ $EXTRACT_EXIT -ne 0 ] || [ -z "$EXTRACT_OUT" ]; then
    echo "ERROR: PYEXTRACT failed (exit=$EXTRACT_EXIT): $EXTRACT_OUT"
    exit 1
fi
IFS='|' read -r EMB_PATHS COMBO_NAME NUM_MODALITIES NUM_UNION FOLD_MAP <<< "$EXTRACT_OUT"

if [ "${NUM_MODALITIES:-0}" -lt 2 ]; then
    echo "SKIP: only $NUM_MODALITIES modality left after dropping $DROP; a combo needs >= 2."
    exit 0
fi

OUTPUT_DIR="$RESULTS_ROOT/${VPREFIX}ablate_no_${DROP}_${CV}"
if [ -n "$FOLD_MAP" ]; then
    echo "  Surviving (union of the per-fold winners, $NUM_UNION; $NUM_MODALITIES per fold): $COMBO_NAME"
    echo "  Per-fold modalities (positions in that list): $FOLD_MAP"
else
    echo "  Surviving ($NUM_MODALITIES): $COMBO_NAME"
fi
echo "  Selection split: ${SELECTION_ARGS:-(off)}"
echo "  Output: $OUTPUT_DIR"

# The arm's PCA, exactly as the combo script passes it (config.conf
# pca_multi_args). See run_ablation.conf for why the ablation inherits the
# combo's setting (and what it confounds): a variance target resolves k on the
# surviving modalities, so the retained width moves with the dropped category.
echo "  PCA: $PCA_MULTI_ARGS"
echo ""

# The exact train.py arguments, built once: used for the architecture check
# below and for the run itself, so the two cannot drift. Unquoted expansions
# word-split exactly as they did in the plain command line.
TRAIN_ARGS=(
    --embeddings_paths $EMB_PATHS
    --sl_path "$SL_PATH"
    ${NEG_PATH:+--neg_pairs_path "$NEG_PATH"}
    $CELL_LINE_ARGS
    --output_dir "$OUTPUT_DIR"
    --cv_type "$CV"
    $MODEL_ARGS
    --encoder_dims $ENCODER_DIMS
    --dropout $DROPOUT
    --epochs $EPOCHS
    --batch_size $BATCH_SIZE
    --learning_rate $LR
    --weight_decay $WEIGHT_DECAY
    --l1_lambdas $L1_LAMBDAS
    --pd_epsilon $PD_EPSILON
    --eval_interval $EVAL_INTERVAL
    --patience $PATIENCE
    --warmrestart_T0 $WARMRESTART_T0
    --warmrestart_Tmult $WARMRESTART_TMULT
    --num_folds $NUM_FOLDS
    --pos_neg_ratio $POS_NEG_RATIO
    --seed $SEED
    --preprocessing_fit_scope "${PREPROCESSING_FIT_SCOPE:-train}"
    --pca_method "${PCA_METHOD:-plain}"
    --siamese_encoder_type "${SIAMESE_ENCODER_TYPE:-residual}"
    $PCA_MULTI_ARGS
    $VARIANT_ARGS
    ${FOLD_MAP:+--fold_modalities "$FOLD_MAP"}
    $SELECTION_ARGS
    $ARCH_ARGS
)

# The ablation must be the reference combo minus ONE category, nothing else.
# Compare this run's architecture + selection signature (train.py
# --print_arch: encoder, the 2026-09-21 scorer options, select_metric, the
# schedule, the PCA) with the reference's config.json; a mismatch (e.g.
# --ablation-only with today's ARCH_ARGS against a published combo, or another
# arm's PCA) would make the summary read an architecture change as a modality
# effect, so stop before training.
if [ -n "$REF_NAME" ]; then
    EXPECT_SIG=$($PYTHON_PATH train.py "${TRAIN_ARGS[@]}" --print_arch | tail -1)
    set +e
    $PYTHON_PATH - "$REF_DIR/config.json" "$EXPECT_SIG" << 'PYARCH'
import json, sys
from arch_signature import arch_signature
ref = arch_signature(json.load(open(sys.argv[1])))
try:
    run = json.loads(sys.argv[2])
except (ValueError, IndexError):
    sys.exit("ERROR: could not compute this run's signature (train.py --print_arch)")
diff = {k: (ref[k], run.get(k)) for k in ref if ref[k] != run.get(k)}
if diff:
    print(f"ERROR: this ablation would differ from its reference combo "
          f"{sys.argv[1].rsplit('/', 1)[0]} in more than the dropped category:")
    for k, (a, b) in sorted(diff.items()):
        print(f"    {k}: reference={a!r}  this run={b!r}")
    print("  Use the reference's ARCH_ARGS / config, or retrain the combo first.")
    sys.exit(1)
print("  Architecture/selection signature matches the reference combo.")
PYARCH
    ARCH_OK=$?
    set -e
    [ $ARCH_OK -eq 0 ] || exit 1
fi

# A stale directory from an earlier ablation must not be mistaken for this
# run's output; train.py purges per-fold artifacts, but an aborted run can
# leave a results.json the summary would happily read. A task that stops
# before this line (SKIP, PYEXTRACT failure, signature mismatch), or never
# starts, keeps an older directory; run_ablation_summary.sh reports it as a
# MISMATCH, with no delta, by its run_token.txt (written below only after a
# successful train) and its signature / modality / fold-stamp checks.
rm -rf "$OUTPUT_DIR"
$PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

set +e
$PYTHON_PATH train.py "${TRAIN_ARGS[@]}"
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
# This submission's stamp: run_ablation_summary.sh gives no delta to an
# ablation carrying another run's token (a task that never started keeps its
# previous run's directory).
printf '%s\n' "$RUN_TOKEN" > "$OUTPUT_DIR/run_token.txt"

# Record what was ablated and against which combo (provenance; the summary
# locates runs by directory name, checks config.embeddings_paths, and reads
# only num_modalities from here).
$PYTHON_PATH - <<PYSTAMP
import json
p = "$OUTPUT_DIR/results.json"
with open(p) as f:
    d = json.load(f)
d["ablation"] = {
    "dropped_category": "$DROP",
    "surviving_modalities": "$COMBO_NAME".split("+"),
    # per fold (the per_fold selection mode; also in num_modalities, the
    # count each fold trained on): {fold: positions in surviving_modalities}
    "fold_modalities": json.loads('$FOLD_MAP') if '$FOLD_MAP' else None,
    "num_modalities": $NUM_MODALITIES,
    "cv_type": "$CV",
    "variant": "$VARIANT",
    "reference": "$REF_DIR" if "$REF_NAME" else None,
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
echo "SUCCESS: $CV without '$DROP' [$VARIANT; PCA arm $PCA_ARM]  AUPRG=$AUPRG  AUPR=$AUPR"
echo "Completed at: $(date)"
