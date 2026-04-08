# Environment Setup Notes

## Python venv location

```
/ddn_exa/campbell/kaiyang/pytorch/
```

Created with `uv` on a GPU node (gpu2) using system Python 3.12.
**Must be created on a GPU node** — devhouse has Python 3.9 but GPU nodes have Python 3.12.
The venv symlinks to `/usr/bin/python3`, so it inherits whichever Python the node has.

## How to recreate the venv

The canonical way to set up the environment is the SLURM script:

```bash
cd siamese_sl
sbatch slurm/reset_env.sh
```

This creates the venv on a GPU node (required -- devhouse has Python 3.9 but GPU nodes have 3.12), uses plain `pip` (not `uv`) to avoid NFS caching/hardlink issues, installs all dependencies, and verifies the installation.

### Manual / local setup

If you need to install manually (e.g., on a local machine), use plain `pip`:

```bash
pip install numpy torch scipy scikit-learn pandas tqdm requests networkx \
    fair-esm transformers huggingface-hub safetensors sentencepiece \
    sentence-transformers gensim obonet node2vec pykeen regex setuptools \
    charset-normalizer protobuf tiktoken mygene robpy
```

## Known issues

- **devhouse vs GPU nodes**: devhouse runs Python 3.9, GPU nodes (gpu1/gpu2) run Python 3.12. A venv created on devhouse will NOT work on GPU nodes.
- **NFS errors (os error 61)**: The NFS filesystem can be flaky. If `pip install` fails with "No data available (os error 61)", retry with `TMPDIR=/tmp pip install --no-cache-dir ...`
- **TensorFlow conflict**: An old TensorFlow install causes `transformers` to crash on import. Fixed by `export TRANSFORMERS_NO_TF=1` (already set in `slurm/config.sh`).
- **setuptools / pkg_resources**: If `import pkg_resources` fails, run `pip install --force-reinstall setuptools` (use `pip`, not `uv` — `uv` sometimes installs to the wrong `lib`/`lib64` path).
- **SLURM scripts use `python3` not `python`**: The `PYTHON_PATH` in `slurm/config.sh` must point to `python3` (not `python` or `python3.9`), since the venv resolves the unversioned symlinks differently across nodes.

## Config

All SLURM scripts source `slurm/config.sh` for paths, environment, training hyperparameters, and the embedding catalog. Per-job `.conf` files add job-specific settings (PCA variance target, embedding catalog with categories). See `slurm/README.md` for the full configuration structure.
