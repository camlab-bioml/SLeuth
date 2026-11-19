#!/bin/bash
#SBATCH --job-name=sl_final_compare
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: the root
# orchestrator (run_benchmarking_siamese_vs_others.sh), the only one that
# submits this script, passes it as --nodelist on the command line, which
# replaces this line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
# No --gres: the re-grade and the table are numpy / sklearn only (it held a GPU
# for nothing until 2026-09-29), like the folds, transfer and tables jobs.
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
# 3 h: it re-grades 12 siamese rows (4 variants x 3 CVs = 60 dense 8,844^2
# fold matrices through cal_metrics); a TIMEOUT means compare.py never writes
# comparison.csv (the tab:tiers source).
#SBATCH --time=0-03:00:00
#SBATCH --output=sl_comparison/slurm/logs/final_compare_%j.out
#SBATCH --error=sl_comparison/slurm/logs/final_compare_%j.err

# ============================================================================
# Final four-model comparison of ONE PCA arm (PCA_ARM; siamese_sl/slurm/
# config.conf, PCA arms): collect THIS run's siamese combos of that arm (one
# row per variant, from <siamese root>/<prefix>latest_combo_<cv>.txt), re-grade
# them with the shared evaluator (collect_and_regrade_siamese.sh), then print
# the unified table: the arm's SLeuth rows next to the shared models' rows,
# which every arm reads from sl_comparison/results. The arm's roots come from
# sl_comparison/slurm/pca_arm.sh:
#   arm none  siamese_sl/results/          -> sl_comparison/results/
#   arm X     siamese_sl/results/pca_X/    -> sl_comparison/results/pca_X/
# (the rows, *_selected_<cv>.txt markers, stale_siamese/, comparison.csv and
# comparison_notes.txt; comparison.csv, the notes and the rows this job
# collects also name the arm in the file, the rows in selection_audit.pca_arm,
# while the markers, stale_siamese/ and the cell-line ablation's *_nocell rows
# (which the table also reads from here; zero_cell_emb.py records no arm) are
# told apart by their directory alone; 2026-10-21: this said each named it,
# then that every row did). Runs after the shared-model
# array, the arm's combo job and its cell-line ablation (chained afterany by
# run_benchmarking_siamese_vs_others.sh, once per arm).
#
# Standalone, from the repo root:
#   PCA_ARM=<arm> sbatch --export=ALL sl_comparison/slurm/run_compare.sh
# with the run's other PCA_* settings exported as at its submission, and
# PCA_ARM=none for arm none (2026-10-21: without PCA_ARM the job compares the
# arm the submitting shell exports, arm none only where none or nothing is).
# ============================================================================

set -e
# Match the siamese jobs' module setup exactly (siamese_sl/slurm/config.conf;
# config.sh is a dead, stale duplicate — do not use it as the reference):
# gnu15 (gnu14 was retired from Lmod, 2026-07) and NO gcc-toolset-13 SCL — the
# gnu15 toolchain supplies the compiler and the siamese jobs run fine without it.
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Start: $(date) | Job: ${SLURM_JOB_ID:-none} | Node: ${SLURM_NODELIST:-$(hostname)}"
. sl_comparison/slurm/python_path.sh   # the siamese jobs' interpreter (config.conf)
# This job's PCA arm: SLEUTH_ARM / SLEUTH_RESULTS / SLEUTH_CMP (config.conf). A
# setting config.conf refuses stops the job here (set -e), before any row moves.
. sl_comparison/slurm/pca_arm.sh

CVS="${CVS:-cv1 cv2 cv3}"
. sl_comparison/slurm/collect_and_regrade_siamese.sh

echo "--- four-model comparison table (PCA arm $SLEUTH_ARM) ---"
# The arm's SLeuth rows from its directory, every other row from
# sl_comparison/results (compare.py splits on each row's model name; for arm
# none the two are one directory, today's table).
$PYTHON_PATH sl_comparison/compare.py --results_dir "$SLEUTH_CMP" \
    --shared_results_dir sl_comparison/results \
    --csv "$SLEUTH_CMP/comparison.csv" \
    --pca_arm "$SLEUTH_ARM" --pca_label "$SLEUTH_PCA_LABEL"

# Which rows the table has: on 2026-09-23 this log showed only the four SLeuth
# rows and said nothing about the absent SLGNN / NSF4SL / SLMGAE / DegreePrior
# and *_nocell rows. A row of a stage this run skipped is absent by design;
# paper_tables' verdict lists the ones the paper needs. The variants: the
# exported list (the orchestrated run), else the ones the collect step above
# has just resolved from config.conf (a hand-submitted compare: the subshell
# that sources config.conf there does not export its list to this shell).
# The SLeuth rows (and *_nocell) are looked for in the arm's directory, the
# shared models' rows in sl_comparison/results: arm none's other rows there
# are not this arm's. The default is spelled inline (arm none) because
# test_comp_contracts.py runs this block on its own.
ROW_VARIANTS=""
for TAG in $SIAMESE_ALL_TAGS; do
    if [ "$TAG" = BASELINE ]; then ROW_VARIANTS="$ROW_VARIANTS pd:per_head"
    else ROW_VARIANTS="$ROW_VARIANTS ${TAG%%_*}:${TAG#*_}"; fi
done
$PYTHON_PATH - "$CVS" "${SIAMESE_VARIANTS_LIST:-$ROW_VARIANTS}" \
    "${SLEUTH_CMP:-sl_comparison/results}" sl_comparison/results <<'ROWS'
import os, sys
cvs, variants = sys.argv[1].split(), sys.argv[2].split()
arm_dir, shared_dir = sys.argv[3], sys.argv[4]
def row(v):   # collect_and_regrade_siamese.sh / paper_tables.variant_row
    return "siamese_sl" if v == "pd:per_head" else "siamese_sl_" + v.replace(":", "_")
sleuth = ([row(v) for v in variants]
          + [row(v) + "_nocell" for v in variants if v.endswith(":per_head")])
cells = ([(m, cv, arm_dir) for m in sleuth for cv in cvs]
         + [(m, cv, shared_dir) for m in ("SLMGAE", "SLGNN", "NSF4SL", "DegreePrior")
            for cv in cvs])
absent = [f"{m}/{cv}" for m, cv, d in cells if not os.path.isfile(f"{d}/{m}_{cv}.json")]
where = (f"{arm_dir}/<row>_<cv>.json"
         if os.path.normpath(arm_dir) == os.path.normpath(shared_dir) else
         f"SLeuth rows {arm_dir}/<row>_<cv>.json, shared {shared_dir}/<row>_<cv>.json")
print(f"--- rows: {len(cells) - len(absent)} of {len(cells)} present ({where}) ---")
for i in range(0, len(absent), 6):
    print("  ABSENT: " + "  ".join(absent[i:i + 6]))
ROWS

echo "Done: $(date)"
