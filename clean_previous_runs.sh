#!/bin/bash
# ============================================================================
# Remove the outputs of previous training / evaluation runs before a rerun
# ============================================================================
# Deletes what earlier runs WROTE: result directories, checkpoints, prediction
# matrices, eval/fine-tuning CSVs, comparison rows (with the external-transfer
# and paper tables under sl_comparison/results) and SLURM logs, for siamese_sl,
# its cell-line ablation and the three shared-fold models. Every PCA arm is
# included: the arms' own roots, siamese_sl/results/pca_*/ and
# sl_comparison/results/pca_*/ (pca_arms/, the table across the arms, among
# them), are children of those two and go with the rest of their contents. The
# directories themselves are kept (the SLURM #SBATCH --output paths and the
# trainers need them to exist).
#
# Never touched (inputs, not outputs): data/ (embeddings, SynLethKG, TransE
# features, caches), sl_comparison/folds (the orchestrator regenerates and
# verifies it on the login node), source code, paper/.
#
# Usage (from anywhere; paths are resolved from this script's location):
#   bash clean_previous_runs.sh          # dry run: list what would be removed
#   bash clean_previous_runs.sh --yes    # remove it
# --yes refuses while a pipeline job is queued or running (it would delete
# outputs and markers under it); ALLOW_LIVE_JOBS=1 removes anyway.
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

APPLY=false
case "${1:-}" in
    --yes) APPLY=true ;;
    "") ;;
    *) echo "usage: bash $0 [--yes]" >&2; exit 2 ;;
esac

# Same job names as run_benchmarking_siamese_vs_others.sh's live-job check.
if $APPLY && ! command -v squeue >/dev/null 2>&1; then
    echo "NOTE: squeue is not on PATH, so running pipeline jobs were not checked."
fi
if $APPLY && command -v squeue >/dev/null 2>&1 && [ "${ALLOW_LIVE_JOBS:-0}" != 1 ]; then
    if ! _q=$(squeue -h -u "${USER:-$(whoami)}" -o '%i %100j' 2>&1); then
        echo "ERROR: squeue failed, so running jobs cannot be ruled out; nothing was" >&2
        echo "  removed (ALLOW_LIVE_JOBS=1 removes anyway):" >&2
        printf '%s\n' "$_q" | sed 's/^/  /' >&2
        exit 1
    fi
    _live=$(printf '%s\n' "$_q" | awk -v re='^(reset_env|siamese_gen_emb|bench_emb|bench_summary|bestcat_emb|bestcat_combo|eval_finetuning|ablation|ablation_summary|cellline_ablation|bilinear_probe|allmod_combo|ELISL|elisl_inputs|pair_export|sl_preflight|sl_folds|sl_compare|sl_final_compare|sl_ext_transfer|sl_paper_tables|SLGNN|NSF4SL)$' \
        'NF >= 2 && $2 ~ re { print "  " $1 " (" $2 ")" }')
    if [ -n "$_live" ]; then
        echo "ERROR: pipeline jobs are still queued or running; nothing was removed:" >&2
        printf '%s\n' "$_live" | head -20 >&2
        echo "  Wait for them or scancel them (ALLOW_LIVE_JOBS=1 removes anyway)." >&2
        exit 1
    fi
fi

# Directories whose CONTENTS are previous-run outputs (kept, emptied).
EMPTY_DIRS=(
    siamese_sl/results
    siamese_sl/cell_line_ablation/models
    siamese_sl/slurm/logs
    sl_comparison/results
    sl_comparison/slurm/logs
    NSF4SL/results
    NSF4SL/slurm/logs
    SLGNN/results
    SLGNN/slurm/logs
    SLMGAE-in-pytorch/results
    logs
)
# Paths that are entirely previous-run outputs (removed; globs expand here):
# the preflight trees, and the cell-line ablation's result JSONs, which live
# next to its code rather than in models/.
REMOVE_DIRS=(
    sl_comparison/results_preflight
    sl_comparison/folds_preflight
    siamese_sl/cell_line_ablation/results*.json
)

n=0
for d in "${EMPTY_DIRS[@]}"; do
    [ -d "$d" ] || continue
    while IFS= read -r -d '' item; do
        [ "$(basename "$item")" = ".gitkeep" ] && continue
        echo "  $item"
        n=$((n + 1))
        if $APPLY; then rm -rf -- "$item"; fi
    done < <(find "$d" -mindepth 1 -maxdepth 1 -print0)
done
for d in "${REMOVE_DIRS[@]}"; do
    [ -e "$d" ] || continue
    echo "  $d"
    n=$((n + 1))
    if $APPLY; then rm -rf -- "$d"; fi
done

if $APPLY; then
    echo "Removed $n previous-run output(s)."
else
    echo "Dry run: $n previous-run output(s) listed above. Re-run with --yes to remove them."
fi
