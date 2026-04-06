#!/bin/bash
#SBATCH --job-name=siamese_emb_bench
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=14-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================================
# Complete Siamese SL Pipeline + Embedding Benchmark
# ============================================================================
# Runs everything in one job:
#   1. Generate ESM + GO embeddings (if not exist)
#   2. Generate all embeddings (auto-download / extraction)
#   3. Train siamese model on each available embedding x 3 CV types
#   4. Print comparison table
#
# Up to 19 embedding types x 3 CVs = 57 training runs
#
# Usage:
#   cd siamese_sl
#   sbatch slurm/run_embedding_benchmark.sh
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

echo "SIAMESE SL - FULL PIPELINE + EMBEDDING BENCHMARK"
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
echo "Steps 1-3 completed at: $(date)"
echo ""

# ============================================================================
# STEP 4: Detect Available Embeddings & Train
# ============================================================================
echo "--- STEP 4: Embedding Benchmark ---"

AVAILABLE_EMBEDDINGS=()
echo "Checking available embeddings..."
for entry in "${ALL_EMBEDDINGS[@]}"; do
    IFS=':' read -r etype efile ename <<< "$entry"
    if [ -f "$DATA_DIR/$efile" ]; then
        echo "  [FOUND] $ename"
        AVAILABLE_EMBEDDINGS+=("$entry")
    else
        echo "  [SKIP]  $ename"
    fi
done
echo ""

[ ${#AVAILABLE_EMBEDDINGS[@]} -gt 0 ] || { echo "ERROR: No embeddings found"; exit 1; }

TOTAL_RUNS=$((${#AVAILABLE_EMBEDDINGS[@]} * ${#CV_TYPES[@]}))
echo "Training: ${#AVAILABLE_EMBEDDINGS[@]} embeddings x ${#CV_TYPES[@]} CVs = $TOTAL_RUNS runs"
echo "Config: encoder=[$ENCODER_DIMS], epochs=$EPOCHS, patience=$PATIENCE, folds=$NUM_FOLDS"
echo ""

RESULTS_FILE="results/embedding_benchmark_results.txt"
echo "EMBEDDING BENCHMARK RESULTS — $(date)" > $RESULTS_FILE

declare -A RESULTS_AUROC RESULTS_AUPR RESULTS_F1
FAILED_RUNS=0
SUCCESSFUL_RUNS=0

for entry in "${AVAILABLE_EMBEDDINGS[@]}"; do
    IFS=':' read -r ETYPE EFILE ENAME <<< "$entry"
    EMB_PATH="$DATA_DIR/$EFILE"

    echo "============================================================================"
    echo "Embedding: $ENAME"
    echo "============================================================================"

    for CV in "${CV_TYPES[@]}"; do
        echo "  --- $ETYPE / $CV --- $(date)"
        OUTPUT_DIR="results/${ETYPE}_${CV}"

        # Pre-create output dirs (server may not have mkdir in PATH)
        $PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"

        set +e
        $PYTHON_PATH train.py \
            --embeddings_paths "$EMB_PATH" \
            --sl_path "$SL_PATH" \
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
            --seed $SEED

        if [ -f "$OUTPUT_DIR/results.json" ]; then
            AUROC=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d['auroc_mean']:.4f} +/- {d['auroc_std']:.4f}\")" 2>/dev/null)
            AUPR=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d['aupr_mean']:.4f} +/- {d['aupr_std']:.4f}\")" 2>/dev/null)
            F1=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d['f1_mean']:.4f} +/- {d['f1_std']:.4f}\")" 2>/dev/null)
            PARAMS=$($PYTHON_PATH -c "import json; d=json.load(open('$OUTPUT_DIR/results.json'))['summary']; print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")" 2>/dev/null)
            echo "  AUROC=$AUROC  AUPR=$AUPR  F1=$F1  Params=$PARAMS"
            RESULTS_AUROC["${ETYPE}_${CV}"]="$AUROC"
            RESULTS_AUPR["${ETYPE}_${CV}"]="$AUPR"
            RESULTS_F1["${ETYPE}_${CV}"]="$F1"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        else
            echo "  FAILED"
            RESULTS_AUROC["${ETYPE}_${CV}"]="FAILED"
            RESULTS_AUPR["${ETYPE}_${CV}"]="FAILED"
            RESULTS_F1["${ETYPE}_${CV}"]="FAILED"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
        set -e

        # Brief pause between runs to let the disk flush
        sleep 10
    done
    echo ""
done

# ============================================================================
# STEP 5: Summary
# ============================================================================
EMB_TYPES_FOUND=()
for entry in "${AVAILABLE_EMBEDDINGS[@]}"; do
    IFS=':' read -r etype _ _ <<< "$entry"
    EMB_TYPES_FOUND+=("$etype")
done

SUMMARY_TMP=$(mktemp)
{
echo ""
echo "============================================================================"
echo "                    EMBEDDING BENCHMARK RESULTS"
echo "============================================================================"
echo ""
printf "%-20s | %-20s | %-20s | %-20s\n" "Embedding" "CV1 (Edge)" "CV2 (Gene)" "CV3 (Pair)"
echo "-------------------------------------------------------------------------------------"
for ETYPE in "${EMB_TYPES_FOUND[@]}"; do
    printf "%-20s | %-20s | %-20s | %-20s\n" \
        "$ETYPE" \
        "${RESULTS_AUROC[${ETYPE}_cv1]:-N/A}" \
        "${RESULTS_AUROC[${ETYPE}_cv2]:-N/A}" \
        "${RESULTS_AUROC[${ETYPE}_cv3]:-N/A}"
done
echo ""
echo "Runs: $SUCCESSFUL_RUNS successful, $FAILED_RUNS failed (of $TOTAL_RUNS)"
echo "============================================================================"
} > "$SUMMARY_TMP"

cat "$SUMMARY_TMP"
cat "$SUMMARY_TMP" >> $RESULTS_FILE
rm -f "$SUMMARY_TMP"

# JSON summary
$PYTHON_PATH << 'PYTHON_SCRIPT'
import json
from pathlib import Path
from datetime import datetime

cv_types = ["cv1", "cv2", "cv3"]
emb_types = sorted({d.name.rsplit("_cv", 1)[0] for d in Path("results").iterdir()
                     if d.is_dir() and "_cv" in d.name})

summary = {"timestamp": datetime.now().isoformat(), "results": {}}
for etype in emb_types:
    summary["results"][etype] = {}
    for cv in cv_types:
        f = Path(f"results/{etype}_{cv}/results.json")
        if f.exists():
            d = json.load(open(f))["summary"]
            summary["results"][etype][cv] = {
                "auroc": f"{d['auroc_mean']:.4f} +/- {d['auroc_std']:.4f}",
                "aupr": f"{d['aupr_mean']:.4f} +/- {d['aupr_std']:.4f}",
                "f1": f"{d['f1_mean']:.4f} +/- {d['f1_std']:.4f}",
                "total_params": d.get("total_params", 0),
                "nonzero_params": d.get("nonzero_params", 0),
                "weight_sparsity": round(d.get("weight_sparsity", 0), 2),
            }
        else:
            summary["results"][etype][cv] = {"error": "not found"}

json.dump(summary, open("results/embedding_benchmark_summary.json", "w"), indent=2)
print("JSON saved to: results/embedding_benchmark_summary.json")
PYTHON_SCRIPT

echo ""
echo "Completed at: $(date)"
