#!/bin/bash
#SBATCH --job-name=reset_env
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-08-10, by request). gpu3 threw a transient "CUDA
# unknown error" at torch init in 2026-07 and the whole tree was moved to
# gpu2 for that; if it recurs, the symptom is a torch.cuda init failure in
# the very first seconds of the job, not a training-time error.
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
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

# Must match the parent directory of PYTHON_PATH in config.conf
VENV_DIR="/ddn_exa/campbell/kaiyang/pytorch"

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
# latest PyPI build. If a floating build ever outruns the gpu3 NVIDIA
# driver, the CUDA smoke test in Step 6 fails loudly HERE instead of deep inside
# every downstream GPU job. Pin a driver-matched build WITHOUT editing this file:
#   TORCH_SPEC="torch==2.5.1" \
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124" sbatch slurm/reset_env.sh
# (cuXXX must match `nvidia-smi` driver on gpu3). Installing torch separately
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
    openpyxl
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

# CUDA smoke test — this job holds a GPU (--gres=gpu:1), so exercise torch's
# built-in CUDA here to catch a torch-CUDA/driver mismatch (or a bad node) at
# env-reset time instead of deep inside every downstream training job. Retry the
# transient 'CUDA unknown error' a few times before failing loudly.
print()
print('CUDA smoke test...')
import torch
import time as _t
_cuda_ok = False
_last = None
for _i in range(1, 4):
    try:
        torch.cuda.init()
        assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
        _x = torch.zeros(1, device='cuda'); _ = (_x + 1).item()
        _cuda_ok = True
        break
    except Exception as _e:
        _last = _e
        print(f'  cuda check attempt {_i}/3 failed: {_e}')
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _t.sleep(10 * _i)
if not _cuda_ok:
    print(f'  CUDA CHECK FAILED after 3 attempts: {_last}')
    print(f'  torch={torch.__version__}  cuda={torch.version.cuda}')
    print('  -> If a driver/build mismatch: pin torch via TORCH_SPEC /')
    print('     TORCH_INDEX_URL (see Step 4b note). If node-local: resubmit.')
    exit(1)
print(f'  cuda check: OK  torch={torch.__version__}  '
      f'cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}')

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
