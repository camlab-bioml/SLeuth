#!/bin/bash
#SBATCH --job-name=SLGNN
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
#SBATCH --output=slurm/logs/slgnn_%j.out
#SBATCH --error=slurm/logs/slgnn_%j.err

# ============================================================================
# SLGNN training. Default: the ORIGINAL SLGNN over SynLethKG (pure PyTorch; no
# dgl / torch_scatter). ARCH=port runs the earlier KG-free adaptation.
#
# The KG is built ONCE, on a node with internet (the login node), before
# submitting:
#   cd SLGNN && python build_synlethkg.py              # -> ../data/synlethkg
#
# Submit FROM the SLGNN directory so relative paths (../data, results/) resolve:
#   cd SLGNN && sbatch slurm/run_slgnn.sh                    # cv1, cv2, cv3
#   cd SLGNN && CV_TYPES="cv3" sbatch slurm/run_slgnn.sh     # one CV
#   cd SLGNN && ARCH=port EMB=../data/all_genes_esm2.pt sbatch slurm/run_slgnn.sh
# ============================================================================

set -e

ARCH="${ARCH:-original}"
KG_DIR="${KG_DIR:-../data/synlethkg}"
EMB="${EMB:-../data/all_genes_kg_complex.pt}"
CV_TYPES="${CV_TYPES:-cv1 cv2 cv3}"

echo "=================================================="
echo "SLGNN Training (arch: $ARCH, pure PyTorch)"
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

case "$ARCH" in
    original)
        if [ ! -f "$KG_DIR/meta.json" ]; then
            echo "ERROR: $KG_DIR/meta.json not found. Build SynLethKG first," \
                 "on a node with internet: cd SLGNN && python build_synlethkg.py"
            exit 1
        fi
        # Existence is not enough: a meta.json from an older build_synlethkg.py,
        # or mapped from another copy of the HGNC / NCBI / SynLethDB files,
        # trains on a wrong Entrez -> entity mapping and still writes a
        # results.json. The comparison array (run_shared_models.sh) and the root
        # orchestrator both gate on this check; so does this launcher. It reads
        # the repo root from its own path (cwd SLGNN/ is fine) and always checks
        # data/synlethkg, hence the KG_DIR test.
        CHECK=../sl_comparison/slurm/check_shared_inputs.py
        if [ "$KG_DIR" = "../data/synlethkg" ]; then
            if [ ! -f "$CHECK" ]; then
                # new files are untracked until `git add`ed, so a checkout
                # synced with `git commit -a` can lack this one
                echo "ERROR: $CHECK not found in this checkout; sync it before" \
                     "submitting (it gates the KG's provenance)."
                exit 1
            fi
            if ! $PYTHON_PATH "$CHECK" --models SLGNN; then
                echo "ERROR: $KG_DIR is stale or incomplete (see above)." \
                     "Rebuild on a node with internet:" \
                     "cd SLGNN && python build_synlethkg.py"
                exit 1
            fi
        else
            echo "NOTE: KG_DIR=$KG_DIR is not ../data/synlethkg;" \
                 "builder provenance was NOT checked."
        fi
        INPUT_ARGS=(--kg_dir "$KG_DIR")
        INPUT_DESC="SynLethKG: $KG_DIR"
        ;;
    port)
        INPUT_ARGS=(--arch port --embedding_path "$EMB")
        INPUT_DESC="features: $EMB"
        ;;
    *)
        echo "ERROR: ARCH must be 'original' or 'port' (got '$ARCH')"
        exit 1
        ;;
esac

for CV in $CV_TYPES; do
    echo ""
    echo "--- Training SLGNN / $CV ($INPUT_DESC) ---"
    $PYTHON_PATH train_slgnn.py \
        --cv_type "$CV" \
        --folds_dir ../sl_comparison/folds \
        "${INPUT_ARGS[@]}" \
        --output_dir "results/slgnn_standalone_${CV}" \
        --seed 123
done

echo "=================================================="
echo "Completed at: $(date)"
echo "Results in: results/slgnn_standalone_cv*/results.json"
echo "=================================================="
