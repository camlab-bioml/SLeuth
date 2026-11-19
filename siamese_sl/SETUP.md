# Environment Setup Notes

## Python venv location

```
<home>/pytorch/
```

Created by `slurm/reset_env.sh` on a GPU node (gpu3) with `/usr/bin/python3 -m venv` + plain `pip` (not `uv`), using system Python 3.12.
**Must be created on a GPU node** — devhouse has Python 3.9 but GPU nodes have Python 3.12.
The venv symlinks to `/usr/bin/python3`, so it inherits whichever Python the node has.

## How to recreate the venv

The canonical way to set up the environment is the SLURM script:

```bash
cd siamese_sl
sbatch slurm/reset_env.sh
```

This creates the venv on a GPU node (required -- devhouse has Python 3.9 but GPU nodes have 3.12), uses plain `pip` (not `uv`) to avoid NFS caching/hardlink issues, installs all dependencies, and verifies the installation. Its CUDA smoke test is commented out (by request). When generation runs, its ESM step refuses CPU and fails, which cancels the `afterok` DAG; when generation is skipped (the repo-root orchestrator's auto-skip, `--skip-generate`, `--after`, the root `--skip-siamese`, `submit_pipeline.sh --ablation-only` / `--eval-only`) the trainers fall back to CPU silently, so check the first `bestcat` / `sl_compare` log for `Using CPU` / `Device: cpu`.

### Manual / local setup

If you need to install manually (e.g., on a local machine), use plain `pip`:

```bash
pip install numpy torch scipy scikit-learn pandas tqdm requests networkx \
    fair-esm transformers huggingface-hub safetensors sentencepiece \
    sentence-transformers gensim obonet node2vec pykeen regex "setuptools<81" \
    charset-normalizer protobuf tiktoken mygene robpy einops openpyxl
```

`einops` is needed by the ESM-C model code (`Synthyra/ESMplusplus_large`) and
`openpyxl` by `eval_finetuning.py` (the Gilbert `.xlsx` screen).

## Known issues

- **devhouse vs GPU nodes**: devhouse runs Python 3.9, the GPU nodes run Python 3.12. A venv created on devhouse will NOT work on GPU nodes. The pipeline's job scripts pin gpu3 (2026-10-01, by request; gpu2 from 2026-09-23 until then, after gpu3 failed an ECC test); `GPU_NODES` in `slurm/config.conf` moves an orchestrated submission (see `slurm/README.md`, "Change SLURM partition/node").
- **NFS errors (os error 61)**: The NFS filesystem can be flaky. If `pip install` fails with "No data available (os error 61)", retry with `TMPDIR=/tmp pip install --no-cache-dir ...`
- **TensorFlow conflict**: An old TensorFlow install causes `transformers` to crash on import. Fixed by `export TRANSFORMERS_NO_TF=1` (already set in `slurm/config.conf`).
- **setuptools / pkg_resources**: If `import pkg_resources` fails (node2vec then fails to import), run `pip install --force-reinstall "setuptools<81"`. setuptools 81 removed `pkg_resources`, which node2vec 0.4.x still imports at load time (see the pin in `slurm/reset_env.sh`), so an unpinned reinstall does not help. Use `pip`, not `uv` — `uv` sometimes installs to the wrong `lib`/`lib64` path.
- **SLURM scripts use `python3` not `python`**: The `PYTHON_PATH` in `slurm/config.conf` must point to `python3` (not `python` or `python3.9`), since the venv resolves the unversioned symlinks differently across nodes.

## Syncing code to the cluster

The cluster checkout (`<repo>`) is not kept
in sync by git: code reaches it by SFTP upload from the laptop, results come
back by SFTP download, and commits are made on the laptop. A new file reaches
the cluster only once it is uploaded, and the jobs import or source such files
only hours into the run (combo Step 2 imports `arch_signature.py`,
`SLGNN/train_slgnn.py` imports `slgnn_orig_model.py` and `build_synlethkg.py`,
and every `sl_comparison` job sources `sl_comparison/slurm/python_path.sh`).
The repo-root `run_benchmarking_siamese_vs_others.sh` refuses to submit when a
code file its stages need is missing from the checkout, but it checks only
those files. A new file must be committed on the laptop as well as uploaded,
so the repo holds every file a run used (`git add -A`, not `git commit -a`).
This check runs on the laptop and covers the commit only, not the upload:

```bash
# on the laptop, before committing: no file left untracked
git add -A && git status --short | grep '^??'   # must print nothing
```

Never upload `sl_comparison/folds/` (the repo-root orchestrator's step 0
regenerates it on the cluster and it stays read-only between runs), `data/`,
`*/results/` or `*/slurm/logs/`: these only come back by download (the
git-ignored inputs the orchestrator names when one is missing, e.g.
`data/Human_GOsim*.txt` or `data/embeddings_cache/scGPT_human/`, are the
exception: copy exactly those in, once, while no DAG runs). Never upload in a
mirror/delete sync mode. While a DAG is queued or running, upload
nothing if you can avoid it: jobs starting later read whatever is there. The
SLGNN / NSF4SL tasks re-fingerprint `SLGNN/build_synlethkg.py`,
`NSF4SL/build_transe_features.py`, `siamese_sl/gene_name_utils.py` and
`SLMGAE-in-pytorch/gene_corrections_config.py` and refuse any byte change.
Nothing refuses a changed `siamese_sl/cell_line_vocab.py`,
`sl_comparison/prepare_folds.py`, `data/SL_SynLethDB_experimental*` pair file
or `.sources.tsv` sidecar, or `siamese_sl/slurm/*.conf` mid-run (the folds job
only warns about a stale fold source, and later jobs read the new version), so
never upload those either: the run goes on and only `paper_tables.py` flags it,
PAPER INCOMPLETE (stale fold sources, or code drift). On
2026-09-23 a whole-tree upload made after the submission replaced
`sl_comparison/folds/meta.json` with an older copy, and the folds job refused
it and cancelled SLGNN / NSF4SL / SLMGAE and DegreePrior. Step 0 keeps
`sl_comparison/folds` read-only (`chmod -R a-w`) until the next step 0, so run
`prepare_folds.py` by hand only after `chmod -R u+w sl_comparison/folds`.
After a run, download `sl_comparison/folds` together with the results and
commit it on the laptop, so the repo's answer key is the one the run used.
The download can carry the read-only modes: run
`chmod -R u+w sl_comparison/folds` on the laptop before committing or any
other git operation there.

Git records the `.sh` scripts with their exec bit (100755), but an
SFTP-uploaded copy need not keep it: the same upload recreated
`cell_line_ablation/grade.sh` without it, and `./cell_line_ablation/grade.sh`
failed with `Permission denied`. Run scripts through `bash`
(`bash slurm/submit_pipeline.sh`, `bash cell_line_ablation/grade.sh`), never
as `./script.sh`.

## Config

All SLURM scripts source `slurm/config.conf` for paths, environment, training hyperparameters, and the embedding catalog. Per-job `.conf` files add job-specific settings (e.g. `run_best_per_category.conf`: the `pcavar<v>` results-directory tag, which is not a PCA setting, and the embedding catalog with categories). The PCA is set per PCA arm in `config.conf` (since 2026-10-20): the pipeline runs once per arm of `PCA_ARMS` (default `none joint per_modality both`: no PCA, PCA on the concatenation, PCA within each modality, both), each in its own results root (`results/`, `results/pca_<arm>/`), at the targets `PCA_MODALITY_VARIANCE` / `PCA_JOINT_VARIANCE` (0.8) or `PCA_JOINT_DIM` (one count for the arms `joint` and `both`, below the narrowest width their PCA of the concatenation meets: `train.py` refuses one at or above it since 2026-10-21). Arm `none` is the reported model, with no PCA; `POST_PCA_VARIANCE` / `POST_PCA_DIM`, the former post-concat PCA switch, are refused (`PCA_ARMS=joint` is that pipeline). See `slurm/README.md` for the full configuration structure.
