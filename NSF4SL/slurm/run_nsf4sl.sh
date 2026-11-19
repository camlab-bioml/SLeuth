#!/bin/bash
#SBATCH --job-name=NSF4SL
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). No
# orchestrator submits this script, so GPU_NODES does not reach it: pass
# --nodelist=<node> on the command line to run it elsewhere.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/nsf4sl_%j.out
#SBATCH --error=slurm/logs/nsf4sl_%j.err

# ============================================================================
# NSF4SL training (negative-sample-free contrastive SL prediction).
#
# Submit FROM the NSF4SL directory so relative paths (../data, results/) resolve:
#   cd NSF4SL && sbatch slurm/run_nsf4sl.sh              # runs cv1, cv2, cv3
#   cd NSF4SL && CV_TYPES="cv3" sbatch slurm/run_nsf4sl.sh   # one CV
#
# Features are the ORIGINAL NSF4SL ones by default: SynLethKG TransE entity
# embeddings, built once on a node with internet access (the login node):
#   cd NSF4SL && python build_transe_features.py --download
# EMB=<all_genes_*.pt> switches to --features embedding (z-scored), and
# LEGACY_LOOP=1 adds --legacy_loop plus the old --select_metric ndcg@50; both
# together restore the earlier kg_complex configuration (not its exact runs:
# validation NDCG@50 now ranks the grader's shifted matrix, so the selected
# epochs can differ):
#   cd NSF4SL && EMB=../data/all_genes_kg_complex.pt LEGACY_LOOP=1 \
#       sbatch slurm/run_nsf4sl.sh
# ============================================================================

set -e

echo "=================================================="
echo "NSF4SL Training"
echo "Start: $(date) | Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "=================================================="

# gnu15 (gnu14 was retired from Lmod, 2026-07) and NO gcc-toolset-13 SCL —
# the gnu15 toolchain supplies the compiler. Under `set -e` a load of the
# retired gnu14 modulefile aborts the job seconds after allocation, before
# any training, leaving a log with nothing but the banner.
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

PYTHON_PATH=<home>/pytorch/bin/python

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs

EMB="${EMB:-}"
TRANSE="${TRANSE:-../data/nsf4sl_transe_features.pt}"
CV_TYPES="${CV_TYPES:-cv1 cv2 cv3}"

if [ -n "$EMB" ]; then
    FEAT_ARGS=(--features embedding --embedding_path "$EMB")
    FEAT_DESC="$EMB"
else
    if [ ! -f "$TRANSE" ]; then
        echo "ERROR: $TRANSE not found. Build it first (needs internet):"
        echo "  cd NSF4SL && python build_transe_features.py --download"
        exit 1
    fi
    # Existence is not enough: the file records the sha256 of the HGNC table,
    # the correction table, gene_name_utils.py and the builder itself, and one
    # left over from before any of those changed attaches some KG genes to
    # OTHER Entrez rows (sl_data.load_transe_features only checks that
    # 'entity_mean' is present, so such a file trains silently). The benchmark
    # path gets this check from the orchestrator and from run_shared_models.sh;
    # a standalone submission has to run it itself. It reads the repo root from
    # its own path (cwd NSF4SL/ is fine) and always checks the default file, so
    # a custom TRANSE cannot be vouched for, hence the TRANSE test.
    CHECK=../sl_comparison/slurm/check_shared_inputs.py
    if [ "$TRANSE" = "../data/nsf4sl_transe_features.pt" ]; then
        if [ ! -f "$CHECK" ]; then
            # new files are untracked until `git add`ed, so a checkout synced
            # with `git commit -a` can lack this one. Without this test the
            # interpreter's "can't open file" would be reported below as a
            # stale TRANSE, sending the user to rebuild it over the internet.
            echo "ERROR: $CHECK not found in this checkout; sync it before" \
                 "submitting (it gates the TransE file's provenance)."
            exit 1
        fi
        if ! $PYTHON_PATH "$CHECK" --models NSF4SL; then
            echo "FATAL: $TRANSE is stale (see above); rebuild it first."
            exit 1
        fi
    else
        echo "NOTE: TRANSE=$TRANSE is not the default file; its provenance is" \
             "NOT checked (check_shared_inputs.py only knows" \
             "data/nsf4sl_transe_features.pt)."
    fi
    FEAT_ARGS=(--features transe --transe_path "$TRANSE")
    FEAT_DESC="SynLethKG TransE ($TRANSE)"
fi
if [ "${LEGACY_LOOP:-0}" = "1" ]; then
    # the earlier runs also selected on validation NDCG@50, not AUPRG
    FEAT_ARGS+=(--legacy_loop --select_metric ndcg@50)
fi

for CV in $CV_TYPES; do
    echo ""
    echo "--- Training NSF4SL / $CV (features: $FEAT_DESC) ---"
    $PYTHON_PATH train_nsf4sl.py \
        --cv_type "$CV" \
        --folds_dir ../sl_comparison/folds \
        "${FEAT_ARGS[@]}" \
        --output_dir "results/nsf4sl_standalone_${CV}" \
        --seed 123
done

echo "=================================================="
echo "Completed at: $(date)"
echo "Results in: results/nsf4sl_standalone_cv*/results.json"
echo "=================================================="
