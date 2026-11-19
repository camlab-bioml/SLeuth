#!/usr/bin/env bash
#
# Clean training outputs from previous runs so the next pipeline run starts
# fresh. Downloaded data (data/, data/cache/, data/embeddings_cache/) is
# always preserved. Generated embeddings (data/all_genes_*.pt) are preserved
# unless --embeddings is passed, since they are expensive to regenerate.
# Every PCA arm's results root (results/pca_*/) goes with the rest of
# results/; sl_comparison/results/ and its pca_*/ roots are never touched here
# (clean_previous_runs.sh at the repo root empties them).
#
# Usage:
#   bash slurm/clean.sh                 # clean outputs (results, logs, pycache)
#   bash slurm/clean.sh --embeddings    # also remove generated .pt embeddings
#   bash slurm/clean.sh --dry-run       # preview without deleting
#   bash slurm/clean.sh -h | --help     # show usage
# Refuses to delete while a pipeline job is queued or running
# (ALLOW_LIVE_JOBS=1 overrides).

set -euo pipefail

DRY_RUN=0
CLEAN_EMBEDDINGS=0

usage() {
    sed -n '3,17p' "$0" | sed 's/^# //; s/^#$//'
    exit "${1:-0}"
}

for arg in "$@"; do
    case "$arg" in
        -h|--help)       usage 0 ;;
        -n|--dry-run)    DRY_RUN=1 ;;
        --embeddings)    CLEAN_EMBEDDINGS=1 ;;
        *)               echo "Unknown arg: $arg" >&2; usage 1 ;;
    esac
done

# Resolve paths relative to this script (slurm/clean.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIAMESE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Collect targets ---
TARGETS=()

# results/: everything except the directory itself
if [[ -d "$SIAMESE_DIR/results" ]]; then
    while IFS= read -r -d '' path; do
        TARGETS+=("$path")
    done < <(find "$SIAMESE_DIR/results" -mindepth 1 -maxdepth 1 -print0)
fi

# slurm/logs/: every file except .gitkeep
if [[ -d "$SCRIPT_DIR/logs" ]]; then
    while IFS= read -r -d '' path; do
        TARGETS+=("$path")
    done < <(find "$SCRIPT_DIR/logs" -mindepth 1 ! -name .gitkeep -print0)
fi

# __pycache__/ anywhere under siamese_sl/
while IFS= read -r -d '' path; do
    TARGETS+=("$path")
done < <(find "$SIAMESE_DIR" -type d -name __pycache__ -print0)

# Optional: generated embeddings in data/ and computed caches in embeddings_cache/
if [[ "$CLEAN_EMBEDDINGS" -eq 1 ]]; then
    DATA_DIR="$SIAMESE_DIR/../data"
    if [[ -d "$DATA_DIR" ]]; then
        # Generated aligned embeddings: data/all_genes_*.pt
        while IFS= read -r -d '' path; do
            TARGETS+=("$path")
        done < <(find "$DATA_DIR" -maxdepth 1 -name 'all_genes_*.pt' -print0)
        # Gene-list sidecars for those .pt files
        while IFS= read -r -d '' path; do
            TARGETS+=("$path")
        done < <(find "$DATA_DIR" -maxdepth 1 -name 'all_genes_*.genes.txt' -print0)
        # Computed embedding pickles (downloaded sources use different
        # extensions: .gz, .obo, .tsv, .fasta, .txt, .pickle, .npz, .json).
        if [[ -d "$DATA_DIR/embeddings_cache" ]]; then
            while IFS= read -r -d '' path; do
                TARGETS+=("$path")
            done < <(find "$DATA_DIR/embeddings_cache" -maxdepth 1 \
                \( -name '*_gene_embeddings.pkl' \
                   -o -name '*_gene_embeddings.pkl.generator_version' \) -print0)
        fi
        # Legacy NCBI sequence caches (no longer written). The current
        # protein_sequences_canonical.json checks itself and is kept.
        for f in ncbi_gene_sequences.json ncbi_protein_sequences.fasta; do
            [[ -f "$DATA_DIR/embeddings_cache/$f" ]] && \
                TARGETS+=("$DATA_DIR/embeddings_cache/$f")
        done
    fi
fi

# --- Report & delete ---
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    echo "Nothing to clean."
    exit 0
fi

echo "Will remove ${#TARGETS[@]} paths:"
for path in "${TARGETS[@]}"; do
    echo "  $path"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "[dry-run] No files deleted. Re-run without --dry-run to proceed."
    exit 0
fi

# A running pipeline job reads and writes these paths (markers, checkpoints,
# results): refuse while one is queued or running, as the orchestrators do
# (same job names; ALLOW_LIVE_JOBS=1 removes anyway).
if [ "${ALLOW_LIVE_JOBS:-0}" != 1 ]; then
    if command -v squeue >/dev/null 2>&1; then
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
    else
        echo "NOTE: squeue is not on PATH, so running pipeline jobs were not checked."
    fi
fi

for path in "${TARGETS[@]}"; do
    rm -rf -- "$path"
done

echo
echo "Cleaned."
