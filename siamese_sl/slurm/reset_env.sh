#!/bin/bash
#SBATCH --job-name=reset_env
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# 4 h: the same install took ~3 min in four runs but 51m44s on 2026-08-27
# (shared-filesystem I/O). The venv is rm -rf'd first, so a TIMEOUT leaves no
# interpreter and cancels every afterok job downstream.
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm/logs/reset_env_%j.out
#SBATCH --error=slurm/logs/reset_env_%j.err

# ============================================================================
# Recreate the Python venv from scratch on a GPU node.
#
# Uses pip (not uv) to avoid NFS caching/hardlink issues.
# Must run on a GPU node (gpu3) — devhouse has a different Python version.
#
# Usage:
#   cd ~/SLMGAE-pytorch/siamese_sl
#   sbatch slurm/reset_env.sh
# ============================================================================

set -e

# The venv every job runs: config.conf's PYTHON_PATH is <venv>/bin/python3, so
# moving the venv stays a one-line edit there. Read in a subshell with
# SLURM_JOB_ID unset so config.conf's compute-node block (module purge, cd)
# stays inert, as sl_comparison/slurm/python_path.sh does.
PYTHON_PATH=$(bash -c 'unset SLURM_JOB_ID; source "$1" >/dev/null 2>&1; printf "%s" "${PYTHON_PATH:-}"' \
    _ "${SLURM_SUBMIT_DIR:-.}/slurm/config.conf")
case "$PYTHON_PATH" in
    /?*/bin/python*) ;;
    *) echo "ERROR: config.conf PYTHON_PATH='$PYTHON_PATH' is not <venv>/bin/python*"
       echo "  (submit from siamese_sl/: cd siamese_sl && sbatch slurm/reset_env.sh)"
       exit 1 ;;
esac
VENV_DIR="$(dirname "$(dirname "$PYTHON_PATH")")"
# Guard the rm -rf below against a PYTHON_PATH outside a venv (/usr/bin/python3
# would name /usr): an existing VENV_DIR must hold pyvenv.cfg or, after an
# interrupted reset, nothing but venv entries.
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    for _e in "$VENV_DIR"/* "$VENV_DIR"/.[!.]*; do
        { [ -e "$_e" ] || [ -L "$_e" ]; } || continue
        case "$(basename "$_e")" in
            bin|include|lib|lib64|share|etc|.gitignore) ;;   # .gitignore: Python >= 3.13 venvs
            *) echo "ERROR: $VENV_DIR (from config.conf PYTHON_PATH) has no pyvenv.cfg and"
               echo "  holds '$(basename "$_e")': not a venv, refusing to rm -rf it."
               exit 1 ;;
        esac
    done
fi

echo "============================================================================"
echo "Reset Python Environment"
echo "============================================================================"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date)"
echo "Venv: $VENV_DIR"
echo ""

# Step 1: Check system Python
echo "--- System Python ---"
/usr/bin/python3 --version
echo ""

# Step 2: Remove old venv
echo "--- Removing old venv ---"
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
    echo "Removed $VENV_DIR"
else
    echo "No existing venv found"
fi
echo ""

# Step 3: Create new venv
echo "--- Creating venv ---"
/usr/bin/python3 -m venv "$VENV_DIR"
echo "Created $VENV_DIR"
echo ""

# Step 4: Upgrade pip
echo "--- Upgrading pip ---"
"$VENV_DIR/bin/pip" install --upgrade pip
echo ""

# Step 4b: Install torch FIRST — it is the one driver-coupled dependency.
# torch ships ("built-in") its own bundled CUDA runtime; by default we take the
# latest PyPI build. If a floating build ever outruns the GPU node's NVIDIA
# driver, the check that would catch it here, Step 6's CUDA smoke test, is
# commented out (2026-09-23, by request). When generation runs, its ESM step
# refuses CPU and fails, which cancels the afterok DAG; when generation is
# skipped, every trainer falls back to CPU SILENTLY, so check the first
# bestcat / sl_compare log for 'Using CPU' / 'Device: cpu'. Pin a
# driver-matched build WITHOUT editing this file:
#   TORCH_SPEC="torch==2.5.1" \
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124" sbatch slurm/reset_env.sh
# (cuXXX must match the `nvidia-smi` driver on the GPU node). Installing torch separately
# also keeps the CPU-only PyPI deps below off any custom --index-url.
echo "--- Installing torch (${TORCH_SPEC:-torch}${TORCH_INDEX_URL:+ from $TORCH_INDEX_URL}) ---"
if [ -n "${TORCH_INDEX_URL:-}" ]; then
    "$VENV_DIR/bin/pip" install "${TORCH_SPEC:-torch}" --index-url "$TORCH_INDEX_URL"
else
    "$VENV_DIR/bin/pip" install "${TORCH_SPEC:-torch}"
fi
echo ""

# Step 5: Install all packages (using pip, not uv — more reliable on NFS)
# NOTE: setuptools is pinned <81 because setuptools 81 removed the bundled
# `pkg_resources` module, and node2vec 0.4.3 (the version the resolver picks to
# stay compatible with networkx/gensim/pykeen) still does `import pkg_resources`
# at load time. Without the pin, pip installs setuptools >=81 and node2vec fails
# to import ("No module named 'pkg_resources'"), which the verification below
# treats as fatal (exit 1) and blocks the whole afterok pipeline.
echo "--- Installing packages ---"
# torch is installed separately above (Step 4b) so it is intentionally NOT here.
"$VENV_DIR/bin/pip" install \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    tqdm \
    requests \
    networkx \
    fair-esm \
    transformers \
    huggingface-hub \
    safetensors \
    sentencepiece \
    sentence-transformers \
    gensim \
    obonet \
    node2vec \
    pykeen \
    regex \
    "setuptools<81" \
    charset-normalizer \
    protobuf \
    tiktoken \
    mygene \
    robpy \
    einops \
    openpyxl \
    lightgbm \
    scikit-optimize
echo ""

# Step 6: Verify
echo "--- Verification ---"
"$VENV_DIR/bin/python3" -c "
import sys
print(f'Python: {sys.version}')
print(f'Path: {sys.executable}')
print()

pkgs = [
    'numpy', 'torch', 'scipy', 'sklearn', 'pandas', 'tqdm', 'requests',
    'networkx', 'esm', 'transformers', 'huggingface_hub', 'safetensors',
    'sentencepiece', 'sentence_transformers', 'gensim', 'obonet',
    'node2vec', 'pykeen', 'tiktoken', 'google.protobuf', 'mygene', 'robpy',
    'einops', 'openpyxl',
    'lightgbm', 'skopt',   # for the ELISL baseline's trainer
]
ok = 0
fail = 0
for p in pkgs:
    try:
        m = __import__(p)
        v = getattr(m, '__version__', 'OK')
        print(f'  {p:<25} {v}')
        ok += 1
    except Exception as e:
        print(f'  {p:<25} FAILED: {e}')
        fail += 1

print()
print(f'{ok}/{ok+fail} packages OK')
if fail > 0:
    print(f'WARNING: {fail} packages failed')
    exit(1)

# Verify fair-esm (not EvolutionaryScale esm) is installed
import esm
assert hasattr(esm, 'pretrained'), (
    'esm.pretrained missing — wrong package installed. '
    'Need fair-esm, not esm (EvolutionaryScale).'
)
print(f'  esm.pretrained check:    OK (fair-esm)')

import torch
# CUDA smoke test — COMMENTED OUT 2026-09-23 (by request): it passes in almost
# every run, and a node-local CUDA failure shows up soon enough in the first
# GPU job. Restore this block to catch a torch-CUDA/driver mismatch (see the
# Step 4b TORCH_SPEC / TORCH_INDEX_URL note) at env-reset time instead. The
# torch VERSION line below does not touch the device, so it still runs.
# print()
# print('CUDA smoke test...')
# import time as _t
# _cuda_ok = False
# _last = None
# for _i in range(1, 4):
#     try:
#         torch.cuda.init()
#         assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
#         _x = torch.zeros(1, device='cuda'); _ = (_x + 1).item()
#         _cuda_ok = True
#         break
#     except Exception as _e:
#         _last = _e
#         print(f'  cuda check attempt {_i}/3 failed: {_e}')
#         try:
#             torch.cuda.empty_cache()
#         except Exception:
#             pass
#         _t.sleep(10 * _i)
# if not _cuda_ok:
#     print(f'  CUDA CHECK FAILED after 3 attempts: {_last}')
#     print(f'  torch={torch.__version__}  cuda={torch.version.cuda}')
#     print('  -> If a driver/build mismatch: pin torch via TORCH_SPEC /')
#     print('     TORCH_INDEX_URL (see Step 4b note). If node-local: resubmit.')
#     exit(1)
print()
print(f'  torch={torch.__version__}  cuda={torch.version.cuda}  '
      f'(CUDA smoke test disabled; device not touched)')

# Pre-cache HGNC gene name database
print()
print('Pre-caching HGNC gene name database...')
import sys
sys.path.insert(0, '$SLURM_SUBMIT_DIR')
from gene_name_utils import GeneNameMapper
mapper = GeneNameMapper(cache_dir='$SLURM_SUBMIT_DIR/../data/cache')
s = mapper.stats
print(f'  HGNC: {s[\"current_symbols\"]} genes, {s[\"aliases\"]} aliases')
print(f'  Test: SEPT1 -> {mapper.gene_name_normalize(\"SEPT1\")}')
print(f'  Test: MARCH1 -> {mapper.gene_name_normalize(\"MARCH1\")}')
"

echo ""
echo "============================================================================"
echo "Environment reset complete at: $(date)"
echo "Python: $VENV_DIR/bin/python3"
echo "============================================================================"
