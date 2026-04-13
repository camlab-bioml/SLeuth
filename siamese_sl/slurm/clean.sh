#!/usr/bin/env bash
#
# Clean training outputs from previous runs so the next pipeline run starts
# fresh. Downloaded data (data/, data/cache/, data/embeddings_cache/) is
# always preserved. Generated embeddings (data/all_genes_*.pt) are preserved
# unless --embeddings is passed, since they are expensive to regenerate.
#
# Usage:
#   ./slurm/clean.sh                 # clean outputs (results, logs, pycache)
#   ./slurm/clean.sh --embeddings    # also remove generated .pt embeddings
#   ./slurm/clean.sh --dry-run       # preview without deleting
#   ./slurm/clean.sh -h | --help     # show usage

set -euo pipefail

DRY_RUN=0
CLEAN_EMBEDDINGS=0

usage() {
    sed -n '3,12p' "$0" | sed 's/^# //; s/^#$//'
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

# Optional: generated embeddings in data/
if [[ "$CLEAN_EMBEDDINGS" -eq 1 ]]; then
    DATA_DIR="$SIAMESE_DIR/../data"
    if [[ -d "$DATA_DIR" ]]; then
        while IFS= read -r -d '' path; do
            TARGETS+=("$path")
        done < <(find "$DATA_DIR" -maxdepth 1 -name 'all_genes_*.pt' -print0)
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

for path in "${TARGETS[@]}"; do
    rm -rf -- "$path"
done

echo
echo "Cleaned."
