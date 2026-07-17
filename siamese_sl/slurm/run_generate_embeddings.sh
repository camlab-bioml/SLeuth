#!/bin/bash
#SBATCH --job-name=siamese_gen_emb
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3,gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
# 14 days: ESM-2 generation on the full proteome can take many hours per GPU,
# and the first run also downloads large pretrained models (~5 GB).
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Generate All Embeddings
# ============================================================================
# Downloads source data and generates all .pt embedding files:
#   0. SynLethDB experimental SL pairs
#   1. ESM-2 protein embeddings (GPU-heavy)
#   2. GO anc2vec embeddings
#   3. All other embedding types (auto-download + extraction)
#
# This is the first step in the pipeline. Training jobs (benchmark and
# best-per-category array jobs) depend on this job completing successfully.
#
# Usage:
#   cd siamese_sl
#
#   # Preferred: use the pipeline orchestrator (handles all dependencies)
#   ./slurm/submit_pipeline.sh
#
#   # Manual: submit directly and chain training jobs
#   GEN=$(sbatch --parsable slurm/run_generate_embeddings.sh)
#   ./slurm/submit_pipeline.sh --after $GEN
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_generate_embeddings.conf"

echo "SIAMESE SL - EMBEDDING GENERATION"
echo ""

# ============================================================================
# Clean previous generated .pt files (keep downloads in CACHE_DIR intact)
# ============================================================================
echo "--- Cleaning previous generated embeddings ---"
CLEANED=0
for entry in "${ALL_EMBEDDINGS[@]}"; do
    IFS=':' read -r _etype efile _ename <<< "$entry"
    if [ -f "$DATA_DIR/$efile" ]; then
        rm -f "$DATA_DIR/$efile"
        CLEANED=$((CLEANED + 1))
    fi
done
# Remove known-legacy files that were renamed/removed from the catalog.
# Safe to re-run: each rm is guarded by -f and the file list is fixed.
LEGACY_FILES=(
    "all_genes_esm.pt"          # renamed to all_genes_esm2.pt
    "all_genes_esm.genes.txt"   # sidecar of the renamed ESM file
)
LEGACY_CLEANED=0
for lf in "${LEGACY_FILES[@]}"; do
    if [ -f "$DATA_DIR/$lf" ]; then
        rm -f "$DATA_DIR/$lf"
        echo "  Removed legacy: $lf"
        LEGACY_CLEANED=$((LEGACY_CLEANED + 1))
    fi
done
# Remove GO embeddings and NCBI JSON cache (force re-fetch with new gene list)
rm -f "$GO_PATH"
rm -f "$CACHE_DIR/ncbi_gene_sequences.json"
rm -f "$CACHE_DIR/ncbi_protein_sequences.fasta"

# Remove computed per-embedding pickles so generate_embeddings.py regenerates
# them with the current code (e.g., the ComplEx real⊕imag fix needs a rerun).
# The ".pkl" extension is our convention for computed caches; downloaded
# sources use their own extensions (.gz, .obo, .tsv, .fasta, .txt, .pickle,
# .npz, .json) and are preserved.
PKL_CLEANED=$(find "$CACHE_DIR" -maxdepth 1 -name '*_gene_embeddings.pkl' -print -delete 2>/dev/null | wc -l | tr -d ' ')

echo "  Removed $CLEANED old .pt files, $LEGACY_CLEANED legacy files, $PKL_CLEANED cached .pkl embeddings, + NCBI sequence caches"
echo ""

# ============================================================================
# STEP 0: Download SynLethDB SL pairs
# ============================================================================
echo "--- STEP 0: SynLethDB Experimental SL Pairs ---"
sleep 10

# Skip only when BOTH the pairs .txt AND its .sources.tsv sidecar exist. The
# sidecar carries the per-cell-line provenance that USE_CELL_LINES=1 (now the
# default) needs; a .txt left without its sidecar would make every cell-mode
# train.py call crash at _setup_multilabel. Regenerate if either is missing.
SL_SIDECAR="${SL_PATH%.txt}.sources.tsv"
if [ -f "$SL_PATH" ] && [ -f "$SL_SIDECAR" ]; then
    echo "Already exist, skipping"
    wc -l "$SL_PATH" | awk '{print "  " $1 " SL pairs"}'
else
    echo "Downloading SynLethDB and filtering to experimental SL pairs..."
    $PYTHON_PATH download_synlethdb.py \
        --output "$SL_PATH" \
        --cache_dir "$CACHE_DIR"
    { [ -f "$SL_PATH" ] && [ -f "$SL_SIDECAR" ]; } \
        || { echo "FAILED (missing pairs .txt or .sources.tsv sidecar)"; exit 1; }
fi

# Experimentally screened NON-SL pairs + provenance sidecar. With cell-line
# conditioning ON by default (USE_CELL_LINES=1 -> NEG_PATH non-empty), these are
# REQUIRED and the download is FATAL on failure — cell mode has no random-
# negative fallback. Set USE_CELL_LINES=0 and NEG_PATH="" for the random-negative
# benchmark, where a missing negatives file is non-fatal. Skip only when BOTH the
# .txt and its sidecar exist; retry a transient (network/Drive) failure.
NEG_SIDECAR="${NEG_PATH_FILE%.txt}.sources.tsv"
if [ -f "$NEG_PATH_FILE" ] && [ -f "$NEG_SIDECAR" ]; then
    echo "Negatives already exist, skipping"
    wc -l "$NEG_PATH_FILE" | awk '{print "  " $1 " non-SL pairs"}'
else
    echo "Downloading SynLethDB experimental NON-SL pairs (negatives)..."
    for attempt in 1 2 3; do
        if $PYTHON_PATH download_synlethdb.py \
            --relation NONSL \
            --output "$NEG_PATH_FILE" \
            --cache_dir "$CACHE_DIR"; then
            break
        fi
        echo "WARNING: negatives download attempt $attempt/3 returned non-zero"
        sleep $((attempt * 15))
    done
    if [ ! -f "$NEG_PATH_FILE" ] || [ ! -f "$NEG_SIDECAR" ]; then
        if [ -n "${NEG_PATH:-}" ]; then
            echo "ERROR: NEG_PATH is set (cell-line / experimental-negative mode)" \
                 "but the negatives pairs+sidecar download failed after retries."
            exit 1
        fi
        echo "WARNING: could not download negatives; continuing (not needed" \
             "unless NEG_PATH is set)."
    fi
fi
echo ""

# ============================================================================
# STEP 1: Generate ESM Embeddings
# ============================================================================
echo "--- STEP 1: ESM Embeddings ---"
sleep 10

echo "Generating ESM embeddings with Pool PaRTI..."
$PYTHON_PATH generate_all_genes_esm.py \
    --output "$ESM_PATH" \
    --sl_path "$SL_PATH" \
    --cache_dir "$CACHE_DIR" \
    --pooling pool_parti \
    --batch_size 8 \
    --device cuda:0
[ -f "$ESM_PATH" ] || { echo "FAILED"; exit 1; }
echo ""

# ============================================================================
# STEP 2: Generate GO Embeddings
# ============================================================================
echo "--- STEP 2: GO Embeddings ---"
sleep 10

# Auto-download human GAF if missing (same URL used by
# generate_embeddings.py for kg_complex; we do it here so GO anc2vec
# doesn't depend on kg_complex having run first).
if [ ! -f "$GAF_PATH" ]; then
    echo "  GAF not cached, downloading to $GAF_PATH..."
    mkdir -p "$(dirname "$GAF_PATH")"
    set +e
    curl -fsSL --retry 3 -o "$GAF_PATH" \
        "http://geneontology.org/gene-associations/goa_human.gaf.gz"
    status=$?
    set -e
    if [ $status -ne 0 ]; then
        rm -f "$GAF_PATH"  # clean up partial download
        echo "WARNING: GAF download failed (exit $status), skipping GO"
    fi
fi

if [ -f "$GAF_PATH" ]; then
    # GO failure is tolerated (set +e) unlike ESM above, because ESM
    # provides the master gene list that all other embedding steps depend
    # on, whereas GO is just one optional modality.
    set +e
    $PYTHON_PATH generate_go_esm_embeddings.py \
        --go_only \
        --esm_embeddings "$ESM_PATH" \
        --gaf "$GAF_PATH" \
        --output "$GO_PATH"
    set -e
    [ -f "$GO_PATH" ] || echo "WARNING: GO generation failed, continuing"
else
    echo "WARNING: GAF not present at $GAF_PATH, skipping GO"
fi
echo ""

# ============================================================================
# STEP 3: Generate All Precomputed Embeddings
# ============================================================================
echo "--- STEP 3: Generate All Embeddings ---"
sleep 10

for etype in "${PRECOMPUTED_TYPES[@]}"; do
    OUTPUT="$DATA_DIR/all_genes_${etype}.pt"
    echo "[$etype] Generating..."
    set +e
    $PYTHON_PATH generate_embeddings.py \
        --type "$etype" \
        --gene_list "$ESM_PATH" \
        --output "$OUTPUT" \
        --cache_dir "$CACHE_DIR"
    set -e
    [ -f "$OUTPUT" ] && echo "[$etype] Done" || echo "[$etype] Skipped (not available)"
done
echo ""

# ============================================================================
# Summary: list available embeddings
# ============================================================================
echo "--- Available Embeddings ---"
for entry in "${ALL_EMBEDDINGS[@]}"; do
    IFS=':' read -r etype efile ename <<< "$entry"
    if [ -f "$DATA_DIR/$efile" ]; then
        echo "  [FOUND] $ename"
    else
        echo "  [SKIP]  $ename"
    fi
done

echo ""
echo "Embedding generation completed at: $(date)"
