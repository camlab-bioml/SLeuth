# Environment Setup Notes

## Python venv location

```
/ddn_exa/campbell/kaiyang/pytorch/
```

Created with `uv` on a GPU node (gpu2) using system Python 3.12.
**Must be created on a GPU node** — devhouse has Python 3.9 but GPU nodes have Python 3.12.
The venv symlinks to `/usr/bin/python3`, so it inherits whichever Python the node has.

## How to recreate the venv

Submit as a SLURM job on a GPU node (do NOT create on devhouse):

```bash
sbatch --job-name=setup_venv --partition=gpu_Prosmn --nodelist=gpu2 --gres=gpu:1 --mem=16G --time=0-01:00:00 --wrap='
set -e
rm -rf /ddn_exa/campbell/kaiyang/pytorch
/usr/bin/python3 -m venv /ddn_exa/campbell/kaiyang/pytorch
source /ddn_exa/campbell/kaiyang/pytorch/bin/activate
pip install uv
uv pip install numpy torch scipy scikit-learn pandas tqdm requests networkx fair-esm transformers huggingface_hub safetensors sentencepiece sentence-transformers gensim obonet node2vec pykeen regex setuptools charset-normalizer robpy
python -c "import numpy, torch, transformers, sentence_transformers, gensim, obonet, pykeen, robpy; print(\"ALL OK\")"
'
```

## Known issues

- **devhouse vs GPU nodes**: devhouse runs Python 3.9, GPU nodes (gpu1/gpu2) run Python 3.12. A venv created on devhouse will NOT work on GPU nodes.
- **NFS errors (os error 61)**: The NFS filesystem can be flaky. If `uv pip install` fails with "No data available (os error 61)", retry with `TMPDIR=/tmp uv pip install --no-cache ...`
- **TensorFlow conflict**: An old TensorFlow install causes `transformers` to crash on import. Fixed by `export TRANSFORMERS_NO_TF=1` (already set in `slurm/config.sh`).
- **setuptools / pkg_resources**: If `import pkg_resources` fails, run `pip install --force-reinstall setuptools` (use `pip`, not `uv` — `uv` sometimes installs to the wrong `lib`/`lib64` path).
- **SLURM scripts use `python3` not `python`**: The `PYTHON_PATH` in `slurm/config.sh` must point to `python3` (not `python` or `python3.9`), since the venv resolves the unversioned symlinks differently across nodes.

## Config

All SLURM scripts source `slurm/config.sh` for paths, hyperparameters, and environment setup. Edit that one file to change anything.
