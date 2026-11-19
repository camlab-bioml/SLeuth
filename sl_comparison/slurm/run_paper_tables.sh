#!/bin/bash
#SBATCH --job-name=sl_paper_tables
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: the root
# orchestrator (run_benchmarking_siamese_vs_others.sh), the only one that
# submits this script, passes it as --nodelist on the command line, which
# replaces this line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
# 12 h: an arm's job also runs the bootstrap CIs (1000 resamples per fold,
# tier and model, and the external screens) between its two table passes.
#SBATCH --time=0-12:00:00
#SBATCH --output=sl_comparison/slurm/logs/paper_tables_%j.out
#SBATCH --error=sl_comparison/slurm/logs/paper_tables_%j.err

# ============================================================================
# Paper tables of ONE PCA arm (PCA_ARM; siamese_sl/slurm/config.conf, PCA
# arms): the last job of run_benchmarking_siamese_vs_others.sh that one of the
# arm's tables waits for (afterany on the arm's final compare, external
# transfer, cell-line ablation, ablation summary and eval array), so it runs
# whatever failed upstream and reports what is missing instead of hanging.
# (Only that arm's all-gene-pair export, which no table reads, and the
# cross-arm comparison below are chained after it.)
#
# sl_comparison/paper_tables.py --pca_arm <arm> writes the arm's
# paper_tables/ in its comparison root (sl_comparison/slurm/pca_arm.sh:
# sl_comparison/results/paper_tables/ for arm none,
# sl_comparison/results/pca_<arm>/paper_tables/ otherwise), from the arm's
# comparison.csv, external_transfer/ and siamese results root, with the shared
# models' rows from sl_comparison/results: one CSV per paper table (tab_tiers,
# tab_solo, tab_loco, tab_cellline, tab_external, tab_external_ft, each with a
# leading pca_arm column), numbers.json (every quantitative claim of the text
# the pipeline can compute) and README.md (which file and column holds each
# table / number, and every missing input), each naming the arm. Its last line
# is the verdict, PAPER COMPLETE or PAPER INCOMPLETE: <n> item(s) (also
# README.md's line 3, the first under its title, and numbers.json "verdict");
# it certifies the computed tables and numbers, not paper/main.tex, which is
# copied by hand. It exits 1 if a table could not be built, 3 if every table
# was built but an item is missing or the fold sources are stale, 0 only when
# complete (2 is argparse's usage error); this job passes that status on, so
# sacct shows the job FAILED unless the paper tables are complete.
#
# Bootstrap CIs (since 2026-11-13; an arm's job only, PT_BOOTSTRAP=0 skips
# them): after the tables, sl_comparison/bootstrap_ci.py --pca_arm <arm>
# --skip_missing writes <cmp>/bootstrap/ (ci_tiers*.csv, ci_external*.csv,
# ci_sldb_agreement.csv, manifest.json; it reads this pass's tab_tiers.csv and
# skips the rows the tables refused), then the tables are built again, which
# fills their CI columns from it. The arm's previous bootstrap/ is removed
# first (with PT_BOOTSTRAP=0 it is left as it is). The exit status is the
# second pass's; a failed bootstrap turns a 0 into 3 (the CIs are missing).
# Without a tab_tiers.csv from the first pass there is no bootstrap and no
# second pass.
#
# --pca_arms_table, as the FIRST argument, writes the cross-arm comparison
# instead: paper_tables.py --pca_arms_table over the run's arms (PCA_ARMS, at
# least two) into sl_comparison/results/pca_arms/ (tab_pca.csv, numbers.json,
# README.md with the same verdict line 3 and exit status, also requiring every
# arm's own paper tables to be complete). The root orchestrator submits it
# once, afterany on every arm's paper tables.
#
# Needs no GPU. Standalone, from the repo root (the run's PCA_* settings
# exported as at its submission):
#   PCA_ARM=<arm> sbatch --export=ALL sl_comparison/slurm/run_paper_tables.sh [paper_tables.py flags]
#   sbatch sl_comparison/slurm/run_paper_tables.sh --pca_arms_table [paper_tables.py flags]
# with PCA_ARM=none for arm none (2026-10-21: without PCA_ARM the job is the
# arm the submitting shell exports, arm none only where none or nothing is;
# the first form gave PCA_ARM as optional). The table across the arms reads
# PCA_ARMS, not PCA_ARM.
# Extra arguments go to paper_tables.py after the variant settings, e.g.
# --ablation_run_token <T> (CLI only) after submit_pipeline.sh --ablation-only
# on a selection that recorded no run token. The arm and its directories are
# this job's: --pca_arm, --pca_arms, --out_dir, --comparison_csv,
# --external_dir and --root (or an abbreviation) are refused (use PCA_ARM /
# PCA_ARMS, or run paper_tables.py itself for another root).
# ============================================================================

module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. sl_comparison/slurm/python_path.sh   # the siamese jobs' interpreter (config.conf)
# This job's PCA arm and the run's arms: SLEUTH_ARM / SLEUTH_CMP /
# SLEUTH_PCA_ARMS (config.conf). A setting config.conf refuses stops the job
# here, before anything is removed.
. sl_comparison/slurm/pca_arm.sh || exit 1

MODE=arm
if [ "${1:-}" = --pca_arms_table ]; then
    MODE=arms
    shift
fi
# The job clears, lists and grades $OUT: a later copy of one of the options
# it sets would win in argparse and make paper_tables.py read or write
# another arm's directories, or another tree's, under this job's name.
# Matched as argparse does (allow_abbrev: an option is also any prefix of it,
# and --opt=value its own), and refused before anything is removed (exit 2, a
# usage error, like argparse's).
for _a in "$@"; do
    _opt=${_a%%=*}
    case "$_opt" in --?*) ;; *) continue ;; esac
    for _own in --pca_arms_table --pca_arm --pca_arms --out_dir \
                --comparison_csv --external_dir --root; do
        case "$_own" in "$_opt"*) ;; *) continue ;; esac
        if [ "$_a" = --pca_arms_table ]; then
            echo "ERROR: --pca_arms_table must be the FIRST argument of"
            echo "       run_paper_tables.sh (the cross-arm job); nothing was changed."
        else
            echo "ERROR: $_a: the PCA arm and its directories are this job's (PCA_ARM /"
            echo "       PCA_ARMS through sl_comparison/slurm/pca_arm.sh; for another"
            echo "       root run sl_comparison/paper_tables.py itself). Nothing was changed."
        fi
        exit 2
    done
done
# MODE_ARGS is word-split on purpose: one arm per word of $SLEUTH_PCA_ARMS,
# and the paths are repo-relative, never with a space.
if [ "$MODE" = arms ]; then
    _n_arms=0
    for _a in $SLEUTH_PCA_ARMS; do _n_arms=$((_n_arms + 1)); done
    if [ "$_n_arms" -lt 2 ]; then
        echo "ERROR: the cross-arm table compares at least two PCA arms; PCA_ARMS"
        echo "       resolves to '$SLEUTH_PCA_ARMS'. Nothing was changed."
        exit 2
    fi
    OUT=sl_comparison/results/pca_arms
    MODE_ARGS="--pca_arms_table --pca_arms $SLEUTH_PCA_ARMS --out_dir $OUT"
else
    OUT="$SLEUTH_CMP/paper_tables"
    MODE_ARGS="--pca_arm $SLEUTH_ARM --pca_arms $SLEUTH_PCA_ARMS --out_dir $OUT"
    MODE_ARGS="$MODE_ARGS --comparison_csv $SLEUTH_CMP/comparison.csv"
    MODE_ARGS="$MODE_ARGS --external_dir $SLEUTH_CMP/external_transfer"
fi
mkdir -p sl_comparison/slurm/logs "$SLEUTH_CMP" "$(dirname "$OUT")"

echo "=================================================="
if [ "$MODE" = arms ]; then
    echo "Paper tables: the cross-arm comparison of PCA arms $SLEUTH_PCA_ARMS"
    echo "  (this job's own PCA_ARM plays no part)"
else
    echo "Paper tables: PCA arm $SLEUTH_ARM ($SLEUTH_PCA_LABEL)"
fi
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "  run token: ${RUN_TOKEN:-(unset: not submitted by an orchestrator)}"
echo "  output: $OUT/"
echo "=================================================="

# The run's variant settings as config.conf / run_best_per_category.conf
# resolve them (in a bash subshell with SLURM_JOB_ID unset, output discarded):
# the orchestrator exports SIAMESE_VARIANTS_LIST, but SELECTION_VARIANT and the
# pcavar tag only when the caller exported them, and a default edited in
# config.conf would otherwise reach the trainers but not the tables.
SETTINGS=$(bash -c '
    unset SLURM_JOB_ID
    source "$1" >/dev/null 2>&1 || exit 1
    source "$2" >/dev/null 2>&1 || exit 1
    printf "%s|%s|%s|%s\n" "${SIAMESE_VARIANTS[*]}" "$SELECTION_VARIANT" \
        "$PCA_VARIANCE" "${BESTCAT_EXTRA_VARIANTS:-0}"
' _ siamese_sl/slurm/config.conf siamese_sl/slurm/run_best_per_category.conf) || SETTINGS=""
SETTING_ARGS=""
if [ -n "$SETTINGS" ]; then
    IFS='|' read -r S_VARIANTS S_SELECTION S_PCA S_EXTRA <<< "$SETTINGS"
    echo "  variants: $S_VARIANTS | selection: $S_SELECTION | pcavar$S_PCA | BESTCAT_EXTRA_VARIANTS=$S_EXTRA"
    # Word splitting of $S_VARIANTS is intended (one KERNEL:CELL_LINES each).
    SETTING_ARGS="--variants $S_VARIANTS --selection_variant $S_SELECTION"
    SETTING_ARGS="$SETTING_ARGS --pca_variance $S_PCA --bestcat_extra_variants $S_EXTRA"
else
    echo "WARNING: could not read the variant settings from siamese_sl/slurm/*.conf;"
    echo "         paper_tables.py falls back to the environment / its defaults."
fi

# CODE_MANIFEST (exported by the orchestrator at submission): paper_tables.py
# re-hashes the code it lists and reports what changed during the run. A
# caller's own --code_manifest comes later in "$@" and wins.
if [ -n "${CODE_MANIFEST:-}" ]; then
    echo "  code manifest: $CODE_MANIFEST"
    set -- --code_manifest "$CODE_MANIFEST" "$@"
fi

# A table this run cannot rebuild must not survive from an earlier one.
rm -rf "$OUT"
BOOT=""
if [ "$MODE" = arm ] && [ "${PT_BOOTSTRAP:-1}" != 0 ]; then
    BOOT="$SLEUTH_CMP/bootstrap"
    rm -rf "$BOOT"   # nor another run's CIs (the first pass would read them)
fi
$PYTHON_PATH sl_comparison/paper_tables.py $MODE_ARGS $SETTING_ARGS "$@"
RC=$?
BRC=""
if [ -n "$BOOT" ]; then
    if [ -f "$OUT/tab_tiers.csv" ]; then
        echo ""
        echo "--- bootstrap CIs -> $BOOT/ ($(date)) ---"
        export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
        $PYTHON_PATH sl_comparison/bootstrap_ci.py --pca_arm "$SLEUTH_ARM" \
            --out_dir "$BOOT" --skip_missing
        BRC=$?
        [ "$BRC" -eq 0 ] || echo "WARNING: bootstrap_ci.py exited $BRC: the tables below have no (or partial) CIs."
        echo ""
        echo "--- paper tables again, with the CI columns ($(date)) ---"
        rm -rf "$OUT"
        $PYTHON_PATH sl_comparison/paper_tables.py $MODE_ARGS $SETTING_ARGS "$@"
        RC=$?
        [ "$BRC" -eq 0 ] || [ "$RC" -ne 0 ] || RC=3
    else
        echo "NOTE: no $OUT/tab_tiers.csv: no bootstrap CIs (they bootstrap its rows)."
    fi
fi

echo ""
echo "=================================================="
if [ -d "$OUT" ]; then
    echo "Written to $OUT/:"
    ls -1 "$OUT" | sed 's/^/  /'
fi
if [ -f "$OUT/README.md" ]; then
    echo "verdict: $(sed -n 3p "$OUT/README.md")"
    [ -z "$BRC" ] || [ "$BRC" -eq 0 ] || echo "bootstrap CIs FAILED (exit $BRC; above): this job exits 3."
    [ "$RC" -ne 0 ] && echo "paper_tables.py exited $RC; $OUT/README.md lists what is missing."
elif [ "$RC" -ne 0 ]; then
    echo "paper_tables.py failed before writing its outputs (exit $RC; see this job's .err)."
fi
echo "Completed at: $(date)"
echo "=================================================="
exit "$RC"
