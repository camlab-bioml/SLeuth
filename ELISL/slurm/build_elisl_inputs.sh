#!/bin/bash
#SBATCH --job-name=elisl_inputs
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request); --nodelist=<node> on the sbatch
# command line moves it.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=ELISL/slurm/logs/elisl_inputs_%j.out
#SBATCH --error=ELISL/slurm/logs/elisl_inputs_%j.err

# ============================================================================
# ELISL's context-free inputs (ELISL/build_elisl_inputs.py gpu):
#   fasta     accession-only UniProt FASTA + gene-name -> UniProt table
#   seqvec    the authors' `seqvec ... --id 0 --protein` on the GPU (SEQVEC_BIN)
#   node2vec  STRING v11.0 'ec' graph, the authors' node2vec options
#   features  data/elisl/gene_features.npz + meta.json for the shared genes
#
# BEFORE submitting, on the login node (internet):
#   1. python ELISL/build_elisl_inputs.py download
#      (UniProt FASTA, three STRING files, the SeqVec weights -> data/elisl/)
#   2. once: a Python 3.7 environment for seqvec==0.4.1 (the authors' pin;
#      allennlp 0.9 / torch < 2 / numpy < 2 cannot live in the cluster venv):
#      uv ships no Python 3.7, so point it at an installed 3.7 (e.g. a conda
#      or micromamba `python=3.7` env); overrides==3.1.0 is pinned because the
#      unpinned resolution (7.x) makes allennlp 0.9 fail on import with
#      "TypeError: Highway.forward: return type ..." (2026-11-13):
#        uv venv --python /path/to/python3.7 <home>/seqvec_env
#        uv pip install --python <home>/seqvec_env/bin/python \
#            seqvec==0.4.1 overrides==3.1.0
# The root orchestrator (since 2026-11-13) runs step 1 itself and submits
# this job whenever check_shared_inputs.py --models ELISL fails, after it has
# checked SEQVEC_BIN on the login node. By hand, from the REPO ROOT:
#   sbatch --export=ALL ELISL/slurm/build_elisl_inputs.sh
#   SEQVEC_BIN=/path/to/seqvec sbatch --export=ALL ELISL/slurm/build_elisl_inputs.sh
# The sl_comparison/folds must exist (features aligns to their genes.txt).
# ============================================================================

set -eu
module purge
module load gnu15 openmpi5 EasyBuild cmake openblas fftw

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f ELISL/build_elisl_inputs.py ]; then
    echo "FATAL: submit from the repo root"; exit 1
fi
. sl_comparison/slurm/python_path.sh
mkdir -p ELISL/slurm/logs
export SEQVEC_BIN="${SEQVEC_BIN:-<home>/seqvec_env/bin/seqvec}"
if [ ! -x "$SEQVEC_BIN" ]; then
    echo "FATAL: SEQVEC_BIN=$SEQVEC_BIN is not an executable seqvec==0.4.1 (see the header)"
    exit 1
fi
# It must also import (allennlp under an unpinned `overrides` fails there):
# caught here, not after the FASTA step.
if ! "$SEQVEC_BIN" --help >/dev/null 2>&1; then
    echo "FATAL: $SEQVEC_BIN --help fails: its environment is broken (pin"
    echo "  overrides==3.1.0, see the header); output:"
    "$SEQVEC_BIN" --help 2>&1 | tail -5
    exit 1
fi
if [ ! -f sl_comparison/folds/genes.txt ]; then
    echo "FATAL: sl_comparison/folds/genes.txt missing (run run_shared_folds.sh first)"
    exit 1
fi
# An orchestrated run may have frozen data/; the build writes data/elisl only.
mkdir -p data/elisl
nvidia-smi -L || true

$PYTHON_PATH ELISL/build_elisl_inputs.py gpu --seqvec_bin "$SEQVEC_BIN"
$PYTHON_PATH sl_comparison/slurm/check_shared_inputs.py --models ELISL
echo "Done: $(date)"
