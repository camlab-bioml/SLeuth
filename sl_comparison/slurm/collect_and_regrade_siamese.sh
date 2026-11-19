#!/bin/bash
# Collect siamese_sl into the comparison results dir, then RE-GRADE it with the
# shared evaluator wherever its prediction matrices exist.
#
# Sourced (not executed) by run_compare.sh, the DAG's terminal job
# (run_shared_models.sh no longer sources it). Collection and the re-grade live
# in this one file so collection can never run without the re-grade after it: a
# bare collect_siamese.py overwrites the shared-evaluator JSON with siamese's
# self-scored numbers, and the published comparison.csv would then show siamese
# at its own prevalence — exactly the mismatch this whole path exists to remove.
#
# ONE PCA arm per call (PCA arms, siamese_sl/slurm/config.conf): its siamese
# results root <sia> (SLEUTH_RESULTS: siamese_sl/results for arm none,
# siamese_sl/results/pca_<arm> otherwise) and its comparison root <cmp>
# (SLEUTH_CMP: sl_comparison/results, or sl_comparison/results/pca_<arm>),
# both from sl_comparison/slurm/pca_arm.sh, which run_compare.sh sources
# first. Every row, marker and stale copy below is in <cmp>, so the arms'
# rows (same names in every arm) never meet; the shared models' rows in
# sl_comparison/results are never touched here. Unset, both default to arm
# none's directories.
#
# One siamese row per model variant (SIAMESE_VARIANTS in
# siamese_sl/slurm/config.conf, default the 2x2 kernel {pd, bilinear} x
# cell_lines {per_head, pooled}), each built ONLY from THIS run's combo:
#   baseline pd:per_head -> row siamese_sl, the combo named by
#                           <sia>/latest_combo_<cv>.txt, marker
#                           <cmp>/siamese_selected_<cv>.txt, pin
#                           SIAMESE_RESULTS_<cv>
#   other variant <tag>  -> row siamese_sl_<tag>, the combo named by
#                           <sia>/var_<tag>__latest_combo_<cv>.txt, marker
#                           <cmp>/siamese_sl_<tag>_selected_<cv>.txt, pin
#                           SIAMESE_RESULTS_<tag>_<cv>
# (<tag> = <kernel>_<cell_lines>, e.g. bilinear_pooled.) Every row is its
# variant's combo, so the 2x2 compares like with like; collect_siamese.py also
# drops any candidate whose recorded kernel / cell_lines is not the row's.
# With ALLMOD_COMBO=1 (config.conf, 2026-11-12) each variant also gets its
# all-modalities row (siamese_sl/slurm/run_allmod_combo.sh: every catalog
# embedding, no modality selection): siamese_sl_allmod for pd:per_head,
# siamese_sl_allmod_<tag> otherwise, ONLY from <sia>/allmod__<prefix>
# latest_allmod_<cv>.txt (collect_siamese.py --allmod), marker
# <cmp>/<row>_selected_<cv>.txt, no pin; re-graded like every row. The
# previous allmod rows are moved aside below whatever the switch says.
# A pin is one global variable per (variant, CV), whatever the arm: it is
# applied to the arm this call collects, so a hand loop over arms sets it
# per arm.
# Every row names its arm (2026-10-21): collect_siamese.py --pca_arm (label
# only, SLEUTH_ARM) writes selection_audit.pca_arm, which the re-grade below
# carries, on the combo rows, the pinned ones and the
# SIAMESE_BASELINE_DISCOVER=all sweep alike (a combo row's modality selection
# record also carries the selection's pca_arm / pca): the arms' rows have the
# same names, and arm none's sit at the pre-arms path.
# Why the latest_combo markers, not a scan of best_cat_* directories:
# run_best_per_category_combo.sh deletes the markers when it starts and writes
# each one only after that (variant, CV) combo trained, while a combo job whose
# selection (Step 2) fails, or that never runs, leaves the PREVIOUS run's combo
# directories on disk, and those pass every other check (variant, validation
# key, fold stamp). A CV without a marker therefore gets NO row, never an older
# model (collect_siamese.py --combo_markers; the orchestrator also removes old
# markers at submission).
# SIAMESE_BASELINE_DISCOVER=all restores the old broad baseline discovery (the
# best of the whole pd:per_head siamese sweep under <sia>: single-embedding
# runs, ablations, combos, with NO freshness check; other arms' roots inside
# it are skipped), which is NOT comparable with the variant rows. An explicit
# pin (SIAMESE_RESULTS_<cv> / SIAMESE_RESULTS_<tag>_<cv>) still wins over
# either.
#
# Stale rows: every existing 2x2 siamese row (siamese_sl_cv[123].json,
# siamese_sl_<tag>_cv[123].json for the three tagged variants) and its
# selected-run marker in <cmp> is MOVED, not deleted, into <cmp>/stale_siamese/
# (the arm's own) before collecting, under a per-run timestamp suffix
# (<name>.<YYYYmmddTHHMMSS>.<ext>, as stale_shared/ does), and only rows
# rebuilt below come back. So a variant dropped from SIAMESE_VARIANTS_LIST (the
# baseline pd:per_head included), or a row whose combo did not succeed this
# run, cannot leave a previous run's numbers in the table; compare.py reads
# only the top-level *.json, and each run's moved files stay recoverable
# (stale_siamese/ is git-ignored, so a git-tracked row shows up in git as
# deleted, its committed version in git history). The move never reaches
# another arm: arm none's <cmp> is the parent of the others', and the find
# below stays at its top level.
# A row whose shared re-grade is missing, partial or fails is likewise moved
# there (as <row>_<cv>.<timestamp>.unregraded.json): the table never mixes
# siamese's own evaluator with the shared one.
#
# Expects: PYTHON_PATH, CVS, cwd = repo root, and the arm's SLEUTH_CMP /
# SLEUTH_RESULTS / SLEUTH_ARM (pca_arm.sh; unset = arm none). By hand:
#   export PCA_ARM=<arm>   # (and the run's other PCA_* settings), then
#   . sl_comparison/slurm/python_path.sh && . sl_comparison/slurm/pca_arm.sh \
#       && CVS="cv1 cv2 cv3" . sl_comparison/slurm/collect_and_regrade_siamese.sh
# (as run_compare.sh sets them; without CVS every collect_siamese.py call
# below fails after the arm's rows were moved aside, or simply
# PCA_ARM=<arm> sbatch --export=ALL sl_comparison/slurm/run_compare.sh)
# Every path below spells its default inline, "${SLEUTH_CMP:-...}", rather
# than through a variable set up here: test_comp_contracts.py runs the pin
# loop and the per-fold parameter heredoc on their own.

# The roots must be the arm that was asked for: PCA_ARM names it, SLEUTH_ARM
# the arm pca_arm.sh resolved the roots for (both unset: arm none). A hand
# source with PCA_ARM exported and no pca_arm.sh, or with another arm's roots
# left in this shell, would rebuild one arm's rows from the other's combos;
# refuse before anything moves.
if [ "${PCA_ARM:-none}" != "${SLEUTH_ARM:-none}" ]; then
    echo "ERROR: PCA_ARM=${PCA_ARM:-none}, but this shell's roots are PCA arm ${SLEUTH_ARM:-none}'s:"
    echo "       ${SLEUTH_RESULTS:-siamese_sl/results}/ and ${SLEUTH_CMP:-sl_comparison/results}/."
    echo "       Source sl_comparison/slurm/pca_arm.sh first. Nothing was collected or moved."
    return 1
fi

STALE_DIR="${SLEUTH_CMP:-sl_comparison/results}/stale_siamese"
mkdir -p "$STALE_DIR"
# One suffix per invocation, so a row and its marker share it and a later run
# never overwrites this run's moved copies.
STALE_TS=$(date +%Y%m%dT%H%M%S)
# `find`, not a shell glob: this file may be sourced by hand under zsh, where
# an unmatched glob aborts. Only the rows this file rebuilds, i.e. the 2x2
# variants config.conf accepts (pd:per_head is the untagged siamese_sl row),
# not a broad siamese_sl* glob: other siamese_sl_* rows, such as
# siamese_sl_nocell_<cv>.json from siamese_sl/cell_line_ablation/grade.sh, are
# never rebuilt here and would silently drop out of the table. -maxdepth 1:
# the arm's own rows only, never another arm's <cmp> below arm none's.
find "${SLEUTH_CMP:-sl_comparison/results}" -maxdepth 1 -type f \( \
       -name 'siamese_sl_cv[123].json' \
    -o -name 'siamese_sl_pd_pooled_cv[123].json' \
    -o -name 'siamese_sl_bilinear_per_head_cv[123].json' \
    -o -name 'siamese_sl_bilinear_pooled_cv[123].json' \
    -o -name 'siamese_selected_cv[123].txt' \
    -o -name 'siamese_sl_pd_pooled_selected_cv[123].txt' \
    -o -name 'siamese_sl_bilinear_per_head_selected_cv[123].txt' \
    -o -name 'siamese_sl_bilinear_pooled_selected_cv[123].txt' \
    -o -name 'siamese_sl_allmod_cv[123].json' \
    -o -name 'siamese_sl_allmod_pd_pooled_cv[123].json' \
    -o -name 'siamese_sl_allmod_bilinear_per_head_cv[123].json' \
    -o -name 'siamese_sl_allmod_bilinear_pooled_cv[123].json' \
    -o -name 'siamese_sl_allmod_selected_cv[123].txt' \
    -o -name 'siamese_sl_allmod_pd_pooled_selected_cv[123].txt' \
    -o -name 'siamese_sl_allmod_bilinear_per_head_selected_cv[123].txt' \
    -o -name 'siamese_sl_allmod_bilinear_pooled_selected_cv[123].txt' \) 2>/dev/null \
    | while read -r STALE; do
        _B=$(basename "$STALE")
        STALE_TO="$STALE_DIR/${_B%.*}.$STALE_TS.${_B##*.}"
        echo "  moving previous siamese row aside: $STALE -> $STALE_TO"
        mv -f "$STALE" "$STALE_TO"
    done

# zsh (this file may be sourced by hand) does not word-split unquoted
# parameters, and $CVS, $SIAMESE_VARIANT_TAGS, $SIAMESE_ROWS and
# $SIAMESE_BASELINE_ARGS below all rely on it: without this the pin
# loop's eval hits "bad substitution" and aborts the file before any re-grade.
# Restored at the end of the file; bash never enters the block.
if [ -n "${ZSH_VERSION:-}" ]; then
    [[ -o shwordsplit ]] && _CRS_SWS=1 || _CRS_SWS=0
    setopt SH_WORD_SPLIT
fi

# The variant list, read from siamese_sl/slurm/config.conf in a bash subshell
# with SLURM_JOB_ID unset, so its compute-node block (module purge, cd, banner)
# stays inert. Honors an exported SIAMESE_VARIANTS_LIST override (config.conf
# exports it for every job the orchestrators submit). pd:per_head prints
# BASELINE (its tag is empty); if config.conf cannot be sourced, only the
# baseline row is collected, as before.
SIAMESE_ALL_TAGS=$(bash -c '
    unset SLURM_JOB_ID
    source "$1" >/dev/null 2>&1 \
        || { echo "WARN: could not source $1; only the baseline row" >&2; echo BASELINE; exit 0; }
    for v in "${SIAMESE_VARIANTS[@]}"; do
        t=$(siamese_variant_tag "$v"); echo "${t:-BASELINE}"
    done
' _ siamese_sl/slurm/config.conf || echo BASELINE)
# ALLMOD_COMBO (config.conf, honouring an exported value): 1 also builds the
# all-modalities rows siamese_sl_allmod[_<tag>] (siamese_sl/slurm/
# run_allmod_combo.sh) of every variant; 0 (default, or config.conf not
# sourceable) builds none. Their previous rows were moved aside above either
# way, so a run without the switch never tabulates an earlier run's.
SIAMESE_ALLMOD=$(bash -c '
    unset SLURM_JOB_ID
    source "$1" >/dev/null 2>&1 || { echo 0; exit 0; }
    echo "${ALLMOD_COMBO:-0}"
' _ siamese_sl/slurm/config.conf || echo 0)
[ "$SIAMESE_ALLMOD" = 1 ] || SIAMESE_ALLMOD=0
SIAMESE_HAS_BASELINE=0
SIAMESE_VARIANT_TAGS=""
for TAG in $SIAMESE_ALL_TAGS; do
    if [ "$TAG" = BASELINE ]; then
        SIAMESE_HAS_BASELINE=1
    else
        SIAMESE_VARIANT_TAGS="$SIAMESE_VARIANT_TAGS $TAG"
    fi
done

# Baseline row (only when pd:per_head is in the variant list): this run's
# combos only (the latest_combo markers), unless SIAMESE_BASELINE_DISCOVER=all
# asks for the old sweep-wide discovery. The siamese root is the arm's
# SLEUTH_RESULTS. Without one (a hand source of arm none), SIAMESE_DIR, a
# hand override naming siamese's RESULTS dir (the markers and combo
# directories live directly in it); the root orchestrator uses the same name
# for siamese_sl/ itself, so accept that too rather than find no markers if it
# ever reaches a shell exported. With SLEUTH_RESULTS set SIAMESE_DIR cannot
# move the arm's root (an exported siamese_sl/ would read arm none's combos
# into every arm's rows), and the NOTE says it was not used.
SIAMESE_RESULTS_DIR="${SLEUTH_RESULTS:-${SIAMESE_DIR:-siamese_sl/results}}"
if [ -f "$SIAMESE_RESULTS_DIR/train.py" ] && [ -d "$SIAMESE_RESULTS_DIR/results" ]; then
    SIAMESE_RESULTS_DIR="$SIAMESE_RESULTS_DIR/results"
fi
if [ -n "${SLEUTH_RESULTS:-}" ] && [ -n "${SIAMESE_DIR:-}" ]; then
    echo "NOTE: SIAMESE_DIR=$SIAMESE_DIR is not used: this call collects PCA arm"
    echo "      ${SLEUTH_ARM:-none}'s combos, from $SLEUTH_RESULTS/ (pca_arm.sh)."
fi
echo "--- PCA arm ${SLEUTH_ARM:-none}: combos from $SIAMESE_RESULTS_DIR/, rows into ${SLEUTH_CMP:-sl_comparison/results}/ ---"
if [ "$SIAMESE_HAS_BASELINE" != 1 ]; then
    echo "--- pd:per_head not in SIAMESE_VARIANTS: no siamese_sl baseline row ---"
elif [ "${SIAMESE_BASELINE_DISCOVER:-combo}" = "all" ]; then
    # (--cv_types: the sweep is limited to $CVS like the combo branch; the
    # pin and re-grade loops below cover $CVS only, so a row of another CV
    # would stay self-scored at top level.)
    SIAMESE_BASELINE_ARGS="--cv_types $CVS"
    echo "--- collecting siamese_sl results (SIAMESE_BASELINE_DISCOVER=all: whole pd:per_head sweep of $SIAMESE_RESULTS_DIR/, no freshness check) ---"
else
    SIAMESE_BASELINE_ARGS="--combo_markers --cv_types $CVS"
    echo "--- collecting siamese_sl results (this run's pd:per_head combos: $SIAMESE_RESULTS_DIR/latest_combo_<cv>.txt) ---"
fi
if [ "$SIAMESE_HAS_BASELINE" = 1 ]; then
    $PYTHON_PATH sl_comparison/collect_siamese.py \
        --discover_dir "$SIAMESE_RESULTS_DIR" \
        --variant pd:per_head $SIAMESE_BASELINE_ARGS \
        --pca_arm "${SLEUTH_ARM:-none}" \
        --out_dir "${SLEUTH_CMP:-sl_comparison/results}" || true
fi

for TAG in $SIAMESE_VARIANT_TAGS; do
    echo "--- collecting siamese_sl_$TAG results (this run's combos: $SIAMESE_RESULTS_DIR/var_${TAG}__latest_combo_<cv>.txt) ---"
    # <tag> = <kernel>_<cell_lines>; kernels have no underscore.
    $PYTHON_PATH sl_comparison/collect_siamese.py \
        --discover_dir "$SIAMESE_RESULTS_DIR" \
        --model_name "siamese_sl_$TAG" \
        --variant "${TAG%%_*}:${TAG#*_}" \
        --combo_markers --cv_types $CVS \
        --pca_arm "${SLEUTH_ARM:-none}" \
        --out_dir "${SLEUTH_CMP:-sl_comparison/results}" || true
done

# The all-modalities rows (ALLMOD_COMBO=1): one per variant, from this run's
# <sia>/allmod__<prefix>latest_allmod_<cv>.txt markers only (collect_siamese.py
# --allmod), siamese_sl_allmod for pd:per_head, siamese_sl_allmod_<tag> else.
SIAMESE_ALLMOD_TAGS=""
if [ "$SIAMESE_ALLMOD" = 1 ]; then
    for TAG in $SIAMESE_ALL_TAGS; do
        if [ "$TAG" = BASELINE ]; then
            AMODEL=siamese_sl_allmod; AVAR=pd:per_head
        else
            AMODEL="siamese_sl_allmod_$TAG"; AVAR="${TAG%%_*}:${TAG#*_}"
        fi
        SIAMESE_ALLMOD_TAGS="$SIAMESE_ALLMOD_TAGS $AMODEL"
        echo "--- collecting $AMODEL results (all modalities, no selection: $SIAMESE_RESULTS_DIR/allmod__<prefix>latest_allmod_<cv>.txt) ---"
        $PYTHON_PATH sl_comparison/collect_siamese.py \
            --discover_dir "$SIAMESE_RESULTS_DIR" \
            --model_name "$AMODEL" --variant "$AVAR" --allmod \
            --combo_markers --cv_types $CVS \
            --pca_arm "${SLEUTH_ARM:-none}" \
            --out_dir "${SLEUTH_CMP:-sl_comparison/results}" || true
    done
fi

# Rows as MODEL|MARKER_STEM|PIN_STEM. The baseline keeps its historical marker
# (siamese_selected_<cv>.txt) and pin (SIAMESE_RESULTS_<cv>). An
# all-modalities row has no pin (PIN_STEM -): only its marked run is its row.
SIAMESE_ROWS=""
[ "$SIAMESE_HAS_BASELINE" = 1 ] && SIAMESE_ROWS="siamese_sl|siamese|SIAMESE_RESULTS"
for TAG in $SIAMESE_VARIANT_TAGS; do
    SIAMESE_ROWS="$SIAMESE_ROWS siamese_sl_$TAG|siamese_sl_$TAG|SIAMESE_RESULTS_$TAG"
done
for AMODEL in $SIAMESE_ALLMOD_TAGS; do
    SIAMESE_ROWS="$SIAMESE_ROWS $AMODEL|$AMODEL|-"
done

# An explicit per-CV pin overrides discovery. Applied AFTER discovery so it
# wins, and it also refreshes the marker so the re-grade below targets the
# pinned run rather than the auto-discovered one. Row and marker go into the
# arm's <cmp>, like the discovered ones.
for ROW in $SIAMESE_ROWS; do
    IFS='|' read -r SMODEL SMARK SPIN <<< "$ROW"
    [ "$SPIN" != - ] || continue
    for CV in $CVS; do
        # `eval`, not ${!VAR}: indirect expansion is a bash-ism. Under zsh it
        # raises "bad substitution", which aborts the rest of THIS file —
        # including the re-grade loop below — while the collection above has
        # already succeeded. The result is a siamese row scored on its own
        # evaluator, carrying a note that says so, and no error anyone would
        # notice. The SLURM jobs run bash so they were never affected, but
        # anyone sourcing this by hand was.
        eval "SRES=\${${SPIN}_${CV}:-}"
        if [ -n "$SRES" ] && [ -f "$SRES" ]; then
            $PYTHON_PATH sl_comparison/collect_siamese.py --siamese_results "$SRES" \
                --model_name "$SMODEL" --cv_type "$CV" \
                --pca_arm "${SLEUTH_ARM:-none}" \
                --out "${SLEUTH_CMP:-sl_comparison/results}/${SMODEL}_$CV.json" \
                && dirname "$SRES" > "${SLEUTH_CMP:-sl_comparison/results}/${SMARK}_selected_$CV.txt"
        elif [ -n "$SRES" ]; then
            echo "ERROR: pin ${SPIN}_${CV}=$SRES is not a file (give the path of a"
            echo "       results.json, absolute or from the repo root). The pin is IGNORED:"
            echo "       $SMODEL/$CV is the discovered run (or has no row), NOT the pinned one."
        fi
    done
done

# Re-grade with the SAME evaluator, test edges and gene universe as the peers.
# This is what makes the `baseline` column identical across all four rows.
for ROW in $SIAMESE_ROWS; do
    IFS='|' read -r SMODEL SMARK SPIN <<< "$ROW"
    for CV in $CVS; do
        MARKER="${SLEUTH_CMP:-sl_comparison/results}/${SMARK}_selected_$CV.txt"
        [ -f "$MARKER" ] || continue
        SDIR="$(cat "$MARKER")"
        NPRED=$(ls "$SDIR"/fold_*_"$CV"_predictions.npy 2>/dev/null | wc -l | tr -d ' ')
        NFOLD=$(ls -d sl_comparison/folds/"$CV"/fold_* 2>/dev/null | wc -l | tr -d ' ')
        ROWJSON="${SLEUTH_CMP:-sl_comparison/results}/${SMODEL}_$CV.json"
        if [ "$NPRED" -eq 0 ]; then
            echo "ERROR: no prediction matrices in $SDIR for $CV: $SMODEL/$CV cannot"
            echo "       be graded by the shared evaluator. Row moved to"
            echo "       $STALE_DIR/ (not published). Check the siamese run used"
            echo "       --folds_dir (SHARED_FOLDS)."
            [ -f "$ROWJSON" ] && mv -f "$ROWJSON" "$STALE_DIR/${SMODEL}_$CV.$STALE_TS.unregraded.json"
            continue
        fi
        if [ "$NPRED" -ne "$NFOLD" ]; then
            # A partial fold set would silently give siamese a mean over fewer
            # folds than its peers, with every *_std collapsed toward 0. Refuse
            # rather than publish a row that looks comparable and is not.
            echo "ERROR: $SMODEL/$CV has $NPRED of $NFOLD prediction matrices"
            echo "       (retrieval pass likely failed on some folds — grep the siamese"
            echo "       log for 'retrieval metrics skipped'). REFUSING to re-grade a"
            echo "       partial fold set; row moved to $STALE_DIR/ (not published)."
            [ -f "$ROWJSON" ] && mv -f "$ROWJSON" "$STALE_DIR/${SMODEL}_$CV.$STALE_TS.unregraded.json"
            continue
        fi
        echo ""; echo "--- $SMODEL / $CV (shared evaluator, $NPRED folds) ---"
        # --carry_from is the SAME path we write: the re-grade replaces every
        # metric but must not lose the annotations collect_siamese.py just
        # wrote (evaluate_model.CARRY_KEYS: selection_audit, which holds the
        # modality-selection record, and label_semantics). Dropping them is
        # how compare.py ended up pointing at a `note` field that no longer
        # existed. `note` itself is NOT carried: it describes the self-scored
        # numbers this re-grade replaces.
        if ! $PYTHON_PATH sl_comparison/evaluate_model.py --model_name "$SMODEL" \
            --cv_type "$CV" --pred_dir "$SDIR" \
            --carry_from "$ROWJSON" --out "$ROWJSON"; then
            echo "ERROR: shared re-grade failed for $SMODEL/$CV; row moved to"
            echo "       $STALE_DIR/ (not published: it would mix evaluators)."
            [ -f "$ROWJSON" ] && mv -f "$ROWJSON" "$STALE_DIR/${SMODEL}_$CV.$STALE_TS.unregraded.json"
            continue
        fi
        # Per-fold trainable-parameter counts (a PCA arm's variance target
        # resolves the kept width per fold, hence the entry layer differs by
        # fold; in arm none, no PCA, every fold has the same count, as under
        # an exact PCA_JOINT_DIM in arm joint). trainable_params above is the LAST
        # fold's (model_params.json is rewritten at every fold start); the paper
        # reports mean (sd) over folds. Injected after the re-grade, which keeps
        # only CARRY_KEYS; compare.py then writes a trainable_params_fold
        # mean/std row. Population sd (ddof=0), as every other sd. `|| echo`:
        # run_compare.sh runs under set -e and must still write the table.
        $PYTHON_PATH - "$SDIR/results.json" "$ROWJSON" <<'PYEOF' \
            || echo "WARNING: no per-fold parameter counts for $SMODEL/$CV (from $SDIR/results.json)"
import json, sys
import numpy as np
res = json.load(open(sys.argv[1]))
tp = [m.get("total_params") for m in res.get("fold_metrics") or []]
if not tp or not all(isinstance(t, int) and not isinstance(t, bool) for t in tp):
    sys.exit(f"fold_metrics[].total_params missing or not integers: {tp!r}")
row = json.load(open(sys.argv[2]))
s = row["summary"]
s["trainable_params_fold_mean"] = float(np.mean(tp))
s["trainable_params_fold_std"] = float(np.std(tp))
s["trainable_params_fold_min"] = int(min(tp))
s["trainable_params_fold_max"] = int(max(tp))
s["trainable_params_fold_n"] = len(tp)
# Through <name>.tmp and a rename: sys.argv[2] is the re-graded row itself, and
# a rewrite that fails part-way (a full quota) must cost the counts, as the
# WARNING says, not the row.
import os
tmp = sys.argv[2] + ".tmp"
try:
    with open(tmp, "w") as f:
        json.dump(row, f, indent=2)
    os.replace(tmp, sys.argv[2])
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
print(f"  per-fold trainable params: {np.mean(tp):,.1f} ({np.std(tp):,.1f}) over {len(tp)} folds")
PYEOF
    done
done
# Undo the zsh SH_WORD_SPLIT set near the top (see there); `if`, not `&&`, so
# the file's last status is 0 under run_compare.sh's `set -e`.
if [ -n "${ZSH_VERSION:-}" ] && [ "${_CRS_SWS:-1}" = 0 ]; then
    unsetopt SH_WORD_SPLIT
fi
