#!/bin/bash
#SBATCH --job-name=siamese_gen_emb
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
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
source "$SLURM_SUBMIT_DIR/slurm/config.sh"
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
# Also remove GO embeddings and NCBI JSON cache (force re-fetch with new gene list)
rm -f "$GO_PATH"
rm -f "$CACHE_DIR/ncbi_gene_sequences.json"
rm -f "$CACHE_DIR/ncbi_protein_sequences.fasta"
echo "  Removed $CLEANED old .pt files + NCBI sequence caches"
echo ""

# ============================================================================
# STEP 0: Download SynLethDB SL pairs
# ============================================================================
echo "--- STEP 0: SynLethDB Experimental SL Pairs ---"
sleep 10

if [ -f "$SL_PATH" ]; then
    echo "Already exist, skipping"
    wc -l "$SL_PATH" | awk '{print "  " $1 " SL pairs"}'
else
    echo "Downloading SynLethDB 2.0 and filtering to experimental pairs..."
    $PYTHON_PATH download_synlethdb.py \
        --output "$SL_PATH" \
        --cache_dir "$CACHE_DIR"
    [ -f "$SL_PATH" ] || { echo "FAILED"; exit 1; }
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
    echo "WARNING: GAF not found at $GAF_PATH, skipping GO"
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
