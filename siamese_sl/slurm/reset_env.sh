#!/bin/bash
#SBATCH --job-name=reset_env
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
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
# Must run on a GPU node (gpu2/gpu3) — devhouse has a different Python version.
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

# Step 5: Install all packages (using pip, not uv — more reliable on NFS)
echo "--- Installing packages ---"
"$VENV_DIR/bin/pip" install \
    numpy \
    torch \
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
    setuptools \
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
