#!/bin/bash
#SBATCH --job-name=allmod_combo
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request), like every siamese GPU job.
# GPU_NODES (config.conf) moves an orchestrated submission: both orchestrators
# pass it as --nodelist on the command line, which replaces this line. A
# standalone sbatch passes --nodelist itself to move it.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
# The combo's 64G / 12 CPUs / 5 days. The input is about twice the selected
# combo's width (arm none: 9,448 columns for the 16 generated catalog
# embeddings against 4,668 for six winners): the 19,397 x 9,448 matrix is
# 0.7 GB in float32 per copy and arm joint's SVD of it a few GB, far inside
# 64G. Each task trains |SIAMESE_VARIANTS| models (2 in the revision run)
# where the combo job trains |SIAMESE_VARIANTS| x 3 in the same limit.
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=5-00:00:00
# One task per CV: task i trains CV_TYPES[i] (config.conf) for every variant.
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/allmod_combo_%A_%a.out
#SBATCH --error=slurm/logs/allmod_combo_%A_%a.err

# ============================================================================
# All-modalities combo (2026-11-12, review response) — no modality selection
# ============================================================================
# The multi-modal SLeuth model on EVERY embedding of the catalog
# (run_best_per_category.conf EMB_CATALOG, in catalog order; a catalog type
# whose file was not generated is skipped only when it is in OPTIONAL_TYPES,
# run_generate_embeddings.conf, any other missing file refuses the task),
# concatenated by the PCA arm's pipeline (config.conf pca_multi_args: arm
# none rescales each modality and concatenates, arm joint standardises the
# concatenation and keeps PCA_JOINT_VARIANCE of it, ...). No Step-1 run and
# no best_per_category.json is read, so it can be submitted without waiting
# for the single-embedding array or the selection.
#
# Same training as the selected combo of the same (arm, variant): MODEL_ARGS,
# ENCODER_DIMS, DROPOUT, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, L1_LAMBDAS,
# PD_EPSILON, EVAL_INTERVAL, PATIENCE, the warm restarts, NUM_FOLDS, SEED,
# PREPROCESSING_FIT_SCOPE, PCA_METHOD, SIAMESE_ENCODER_TYPE, CELL_LINE_ARGS,
# NEG_PATH, ARCH_ARGS, the arm's PCA_MULTI_ARGS, the variant's --kernel /
# --cell_lines, and SELECTION_ARGS (--selection_frac / --selection_seed), so it
# trains on exactly the pairs the selected combo trains on: the selection
# split is held out and scored, but chooses nothing. No --fold_modalities:
# every fold trains on every modality. train.py --all_modalities records it in
# config.json (an arch_signature key, so the model is never mistaken for the
# selected combo by a signature check).
#
# Outputs per (variant, CV), in the PCA arm's results root (config.conf
# RESULTS_ROOT: results/ for arm none, results/pca_<arm>/ otherwise), with
# <prefix> = allmod__<siamese_variant_prefix> (config.conf allmod_prefix:
# allmod__ for pd:per_head, allmod__var_<kernel>_<cell_lines>__ otherwise):
#   <arm root>/<prefix>all_modalities_<cv>/   train.py output dir, removed and
#       rebuilt by each run; all_modalities.json records the modalities used
#       and the optional catalog types that were absent
#   <arm root>/<prefix>latest_allmod_<cv>.txt  success marker, the only thing
#       sl_comparison/collect_siamese.py --allmod reads (removed when the task
#       starts, written once that model trained)
# No name here matches a selected combo's (<prefix>best_cat_*, *latest_combo_*),
# so no combo job, ablation, eval, export or Step-2 glob ever reaches them.
# The comparison row is siamese_sl_allmod[_<tag>]_<cv>.json in the arm's
# sl_comparison root (config.conf allmod_row), built by run_compare.sh when
# ALLMOD_COMBO=1 (collect_and_regrade_siamese.sh); each task FIRST moves its
# (variant, CV)'s previous row and selected-run marker to that root's
# stale_siamese/, so a failed task leaves the row MISSING, never an older one.
#
# Submitted by both orchestrators when ALLMOD_COMBO=1 (config.conf; default 0),
# once per PCA arm of PCA_ARMS, after the embedding generation (or the env
# reset) only, and the arm's final compare waits for it (afterany).
#
# Standalone, for a run that already exists (from siamese_sl/; the same
# PCA_ARMS / SIAMESE_VARIANTS_LIST / SELECTION_FRAC / SELECTION_SEED / ARCH_ARGS
# / PCA targets as that run, PCA_ARM per job):
#   for a in none joint; do
#     ALLMOD_COMBO=1 PCA_ARM=$a PCA_ARMS="none joint" \
#       SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head" SELECTION_FRAC=0.1 SELECTION_SEED=42 \
#       sbatch --export=ALL --output=slurm/logs/allmod_combo_${a}_%A_%a.out \
#       --error=slurm/logs/allmod_combo_${a}_%A_%a.err slurm/run_allmod_combo.sh
#   done
#   (one CV: --array=2 for cv3)
# then, per arm, from the repo root, the compare and paper tables with
# ALLMOD_COMBO=1 (they collect and tabulate the rows only then) and the run's
# PCA_ARMS / SIAMESE_VARIANTS_LIST:
#   ALLMOD_COMBO=1 PCA_ARM=<arm> PCA_ARMS="none joint" SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head" \
#       sbatch --dependency=afterany:<allmod job> --export=ALL sl_comparison/slurm/run_compare.sh
#   (the same prefix) sbatch --dependency=afterany:<compare job> \
#       --export=ALL sl_comparison/slurm/run_paper_tables.sh
# Without SLURM (a GPU node, from siamese_sl/):
#   PCA_ARM=none ALLMOD_CVS="cv1 cv2 cv3" bash slurm/run_allmod_combo.sh
# Exit 0 when every (variant, CV) of the task trained, else 1.
# ============================================================================

set -e
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/slurm/config.conf"
source "$SUBMIT_DIR/slurm/run_best_per_category.conf"
# OPTIONAL_TYPES: the catalog types whose absence generation accepts.
source "$SUBMIT_DIR/slurm/run_generate_embeddings.conf"
set +e

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUN_CVS="${CV_TYPES[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$RUN_CVS" ]; then
        echo "ERROR: array task $SLURM_ARRAY_TASK_ID has no CV (CV_TYPES: ${CV_TYPES[*]})"
        exit 1
    fi
else
    RUN_CVS="${ALLMOD_CVS:-${CV_TYPES[*]}}"
fi

CMP_RES="$BASE_DIR/$CMP_ROOT"
TS=$(date +%Y%m%dT%H%M%S)
echo "ALL-MODALITIES COMBO (no modality selection)"
echo "  PCA arm: $PCA_ARM ($(pca_arm_label "$PCA_ARM"); $PCA_MULTI_ARGS) -> $RESULTS_ROOT/"
echo "  Variants (${#SIAMESE_VARIANTS[@]}): ${SIAMESE_VARIANTS[*]}   CVs: $RUN_CVS"
echo "  ARCH_ARGS: ${ARCH_ARGS:-(none; train.py defaults)}"
echo "  Selection split (held out, chooses nothing): ${SELECTION_ARGS:-(off: SELECTION_FRAC empty)}"
[ "${ALLMOD_COMBO:-0}" = 1 ] || \
    echo "  NOTE: ALLMOD_COMBO=$ALLMOD_COMBO: run_compare.sh / run_paper_tables.sh collect and tabulate these rows only with ALLMOD_COMBO=1."
echo ""

FAILED=()
for CV in $RUN_CVS; do
    # FIRST, before anything can stop this CV: every variant's previous row,
    # selected-run marker and success marker go, so a task that fails or is
    # refused below leaves the row missing, never an older model's.
    for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
        APREFIX=$(allmod_prefix "$VARIANT")
        AROW=$(allmod_row "$VARIANT")
        rm -f "$RESULTS_ROOT/${APREFIX}latest_allmod_${CV}.txt"
        for f in "$CMP_RES/${AROW}_${CV}.json" "$CMP_RES/${AROW}_selected_${CV}.txt"; do
            [ -f "$f" ] || continue
            mkdir -p "$CMP_RES/stale_siamese"
            _b=$(basename "$f")
            mv -f "$f" "$CMP_RES/stale_siamese/${_b%.*}.$TS.${_b##*.}"
            echo "  moved previous row aside: $CMP_ROOT/stale_siamese/${_b%.*}.$TS.${_b##*.}"
        done
    done

    # Every catalog embedding, in catalog order (fold_selection.py allmod_spec).
    SPEC_OUT=$($PYTHON_PATH fold_selection.py allmod_spec --data_dir "$DATA_DIR" \
        --optional "${OPTIONAL_TYPES[@]}" --catalog "${EMB_CATALOG[@]}" 2>&1)
    SPEC_EXIT=$?
    if [ $SPEC_EXIT -ne 0 ] || [ -z "$SPEC_OUT" ]; then
        echo "  REFUSED ($CV): $SPEC_OUT"
        for VARIANT in "${SIAMESE_VARIANTS[@]}"; do FAILED+=("$VARIANT/$CV"); done
        continue
    fi
    IFS='|' read -r EMB_PATHS EMB_NAMES NUM_MODALITIES ABSENT_OPTIONAL <<< "$SPEC_OUT"

    for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
        APREFIX=$(allmod_prefix "$VARIANT")
        VARIANT_ARGS=$(siamese_variant_args "$VARIANT")
        OUTPUT_DIR="$RESULTS_ROOT/${APREFIX}all_modalities_${CV}"
        MARKER="$RESULTS_ROOT/${APREFIX}latest_allmod_${CV}.txt"
        echo "============================================================================"
        echo "All-modalities combo for $CV  [variant $VARIANT]"
        echo "============================================================================"
        echo "  Variant: $VARIANT  ($VARIANT_ARGS)"
        echo "  Modalities ($NUM_MODALITIES, catalog order): $EMB_NAMES"
        [ -z "$ABSENT_OPTIONAL" ] || echo "  Not generated (OPTIONAL_TYPES, skipped): $ABSENT_OPTIONAL"
        echo "  PCA arm: $PCA_ARM ($(pca_arm_label "$PCA_ARM")): $PCA_MULTI_ARGS"
        echo "  Output: $OUTPUT_DIR"
        echo ""

        # The previous model of this (variant, CV) goes first, so nothing of
        # it can be collected as this run's.
        rm -rf "$OUTPUT_DIR"
        if ! mkdir -p "$OUTPUT_DIR/checkpoints"; then
            echo "  FAILED: could not create $OUTPUT_DIR/checkpoints"
            FAILED+=("$VARIANT/$CV")
            continue
        fi

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
            $PCA_MULTI_ARGS \
            $VARIANT_ARGS \
            $SELECTION_ARGS \
            $ARCH_ARGS \
            --all_modalities
        TRAIN_EXIT=$?
        if [ $TRAIN_EXIT -ne 0 ] || [ ! -f "$OUTPUT_DIR/results.json" ]; then
            echo "  FAILED: train.py exited $TRAIN_EXIT (results.json: $([ -f "$OUTPUT_DIR/results.json" ] && echo present || echo absent))"
            FAILED+=("$VARIANT/$CV")
            continue
        fi
        # What it was trained on, beside config.json (whose all_modalities /
        # embeddings_paths say the same): the record collect_siamese.py --allmod
        # carries into the row. A model that cannot say so gets no marker.
        if ! $PYTHON_PATH - "$OUTPUT_DIR" "$EMB_NAMES" "$ABSENT_OPTIONAL" \
                "${#EMB_CATALOG[@]}" "$PCA_ARM" "$VARIANT" <<'PYEOF'
import json, os, sys
out, names, absent, n_cat, arm, variant = sys.argv[1:7]
cfg = json.load(open(os.path.join(out, "config.json")))
if cfg.get("all_modalities") is not True or cfg.get("fold_modalities"):
    sys.exit(f"{out}/config.json does not record an all-modalities run "
             f"(all_modalities={cfg.get('all_modalities')!r}, "
             f"fold_modalities={cfg.get('fold_modalities')!r})")
types = names.split("+")
used = [os.path.basename(p) for p in cfg.get("embeddings_paths") or []]
if len(used) != len(types):
    sys.exit(f"{out}/config.json has {len(used)} embeddings_paths, the "
             f"catalog spec {len(types)}")
rec = {"all_modalities": True, "selection": "none", "pca_arm": arm,
       "variant": variant, "modalities": types, "files": used,
       "n_modalities": len(types), "catalog_size": int(n_cat),
       "absent_optional": [t for t in absent.split(",") if t],
       "selection_frac": cfg.get("selection_frac"),
       "selection_seed": cfg.get("selection_seed")}
tmp = os.path.join(out, "all_modalities.json.tmp")
with open(tmp, "w") as f:
    json.dump(rec, f, indent=2)
os.replace(tmp, os.path.join(out, "all_modalities.json"))
PYEOF
        then
            echo "  FAILED: $OUTPUT_DIR is not recorded as an all-modalities run; no marker"
            FAILED+=("$VARIANT/$CV")
            continue
        fi
        echo "$OUTPUT_DIR" > "$MARKER"
        echo "  done: $MARKER"
        echo ""
    done
done

echo "============================================================================"
echo "Completed at: $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "NOT trained: ${FAILED[*]} (no marker, no siamese_sl_allmod* row for them)"
    echo "============================================================================"
    exit 1
fi
echo "Rows are collected by run_compare.sh with ALLMOD_COMBO=1 (PCA_ARM=$PCA_ARM)."
echo "============================================================================"
exit 0
