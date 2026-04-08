#!/bin/bash
#SBATCH --job-name=siamese_gen_emb
#SBATCH --partition=gpu_Prosmn
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
# STEP 0: Download SynLethDB SL pairs
# ============================================================================
echo "--- STEP 0: SynLethDB Experimental SL Pairs ---"

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

if [ -f "$ESM_PATH" ]; then
    echo "Already exist, skipping"
    $PYTHON_PATH -c "
import torch
d = torch.load('$ESM_PATH', map_location='cpu', weights_only=False)
e = d.get('raw_embeddings', d.get('embeddings'))
print(f'  Shape: {e.shape}, Pooling: {d.get(\"pooling\", \"?\")}')" || true
else
    echo "Generating ESM embeddings with Pool PaRTI..."
    $PYTHON_PATH generate_all_genes_esm.py \
        --output "$ESM_PATH" \
        --pooling pool_parti \
        --batch_size 8 \
        --device cuda:0
    [ -f "$ESM_PATH" ] || { echo "FAILED"; exit 1; }
fi
echo ""

# ============================================================================
# STEP 2: Generate GO Embeddings
# ============================================================================
echo "--- STEP 2: GO Embeddings ---"

if [ -f "$GO_PATH" ]; then
    echo "Already exist, skipping"
else
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
fi
echo ""

# ============================================================================
# STEP 3: Generate All Precomputed Embeddings
# ============================================================================
echo "--- STEP 3: Generate All Embeddings ---"

for etype in "${PRECOMPUTED_TYPES[@]}"; do
    OUTPUT="$DATA_DIR/all_genes_${etype}.pt"
    if [ -f "$OUTPUT" ]; then
        echo "[$etype] Already exists"
    else
        echo "[$etype] Generating..."
        set +e
        $PYTHON_PATH generate_embeddings.py \
            --type "$etype" \
            --gene_list "$ESM_PATH" \
            --output "$OUTPUT" \
            --cache_dir "$CACHE_DIR"
        set -e
        [ -f "$OUTPUT" ] && echo "[$etype] Done" || echo "[$etype] Skipped (not available)"
    fi
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
