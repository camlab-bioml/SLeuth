# ELISL (context-free)

Early–late integrated synthetic-lethality prediction with forest ensembles:
Tepeli, Seale & Gonçalves, *Bioinformatics* 40(1):btad764 (2024), code at
[joanagoncalveslab/ELISL](https://github.com/joanagoncalveslab/ELISL) (vendored
from commit `eaf5f8c502a6`, 2024-04-22). Added as a published peer at a
reviewer's request.

**License.** The repository's `LICENSE` is the GNU GPL v3 (its README badge
says LGPL v3; the file is the GPL text). `ELISL/LICENSE` is that file,
unchanged. The vendored files in `elisl_orig/` stay under it, with their
origin and every changed line marked in each header. Distributing this
directory means distributing it under the GPL v3.

## What the authors' repository contains

- `src/models/ELRRF.py`: the model. One LightGBM classifier per data source
  (`boosting_type='rf'` for ELISL-RF, `'gbdt'` for ELISL-GB), late-integrated by
  a weighted average of their probabilities.
- `src/feature_generation/`: per-pair feature builders for its five sources.
  The two context-free ones are `sequence_features.py` (|SeqVec(g1) −
  SeqVec(g2)|, 1,024 dims) and `ppi_features.py` (|node2vec(g1) −
  node2vec(g2)|, 64 dims on STRING v11.0). The three context-specific ones are
  `cell_line_features.py` (CCLE / DepMap dependency + mutation or expression)
  and `tissue_features.py` (TCGA / GTEx).
- `src/embedding/`: node2vec on STRING (`ppi_embedding.py`), SeqVec loading and
  PCA (`sequence_embedding.py`), and the hand-crafted omics features.
- `src/experiments/` + `src/experiment.py`: single-cancer, double-holdout and
  cross-dataset protocols, the Bayesian hyperparameter search and the
  late-integration weights.
- `data/experiment_configs/`: three JSON configs. Everything else (labels,
  feature sets, embeddings, mappings) is in the authors' data release
  (SurfDrive; figshare 10.6084/m9.figshare.23607558), not in the repository.
- No pretrained ELISL models. Python 3.6/3.7 pins (LightGBM 2.3.0,
  scikit-learn 0.24, pandas 1.1, gensim 3.8, node2vec 0.3.3, seqvec 0.4.1).

## Why context-free, and why that is the authors' model

The benchmark is context-free SL over SynLethDB, so ELISL runs on its two
context-free sources only, by the user's choice. ELISL's design does not need
per-cancer data for them. A label's `cancer` column only filters which pairs a
run uses (`prepare_cancer_dataset`; with `cancer=None` it keeps all of them).
The sequence and PPI features never read it. A config listing only
`Sequence_1024` and `PPI_ec` is a valid input to the authors'
`single_cancer_experiment.py`. With it the model is three forests: one per
source, plus the early-integrated concatenation the script adds whenever a
config lists more than one source (`Sequence_1024&PPI_ec`). They are fused by
the late-integration weights. The paper reports these sources' models among
its six. What is dropped is the three cancer-specific sources, and with them
the six-model ensemble of the full ELISL.

## The model, as run here

Per shared fold (`train_elisl.py`):

1. The training pairs (`train_pos` 1, `train_neg` 0) are built into the
   authors' per-source frames: `pair_name`, `class`, then `seq0..1023` /
   `ppi_ec0..63`. Rows are sorted by gene names, as `prepare_cancer_dataset`
   sorts them, and the combined frame is an inner merge on `pair_name`.
2. **Undersampling** to 1:1 (`balance_by_index`, the authors' function), seed
   `seed + fold`.
3. **Bayesian search** (`skopt.gp_minimize`, 20 calls, `random_state = seed`)
   over `ELRRF.get_tree_search_grid()`: `n_estimators` 100–1200,
   `colsample_bytree` {0.5, 0.8, 1}, `max_depth` {10…110, −1},
   `min_child_samples` {1, 2, 4, 10}, `subsample` {0.632, 0.8, 0.99},
   `reg_lambda` {5, 10}. This is the README's `--grid_search true`;
   `--no_grid_search` keeps the constructor's defaults (400 trees,
   165 leaves, colsample 0.8, subsample 0.632, min_child_samples 10,
   reg_lambda 5).
   As shipped, `max_depth` is searched but never applied: the live branch of
   `ELRRF.fit_predict_report` builds `LGBMClassifier` without `max_depth=`
   (the branch that passes it is `if False:`), so `best_params.max_depth`
   in `results.json` has no effect. Kept as the authors ship it.
4. **Late-integration weights**: each forest's mean AUPRC over a 10-fold
   stratified, shuffled CV of the undersampled training rows
   (`fit_predict_cross_validation` + `calculate_data_weights`, the authors'
   functions).
5. **Final fit** of the three forests on the undersampled training rows, with
   `StandardScaler` + `VarianceThreshold(1e-4)` fitted on them (`process=True`).
   Each forest is `lgb.LGBMClassifier(boosting_type='rf', ...)`, fitted by the
   authors' `fit_predict_report`.
6. **Score** every gene pair: `Σ_d w_d · p_d(pair) / Σ_d w_d` over the three
   forests (`combine_preds`).

## Differences from the repository

Harness changes (the same kind as SLGNN's and NSF4SL's):

- **Folds.** The shared folds (`--folds_dir`, required) replace the authors'
  per-cancer 80/20 train/test files. The authors' 10 undersampling
  repetitions become one model per shared fold, with repetition seed
  `seed + fold` (the authors': `124 + repetition`).
- **Negatives.** The label-0 pairs are the shared folds' screened non-SL
  pairs (`train_neg`), as for every model in the comparison, undersampled to
  1:1 on the training side as the authors do. The authors' labels are their
  own curated per-cancer SL / non-SL sets.
- **Test balance.** The authors also undersample the test set
  (`undersample_train_test`). Here every test pair is graded, along with the
  full gene × gene matrix, by `sl_comparison/evaluate_model.py`, like every
  other row.
- **Hyperparameter search objective.** Same space, optimiser, call count and
  late integration (training-AP weights, as the authors' search fuses). The
  objective is **1 − validation AUPRG** of the fused score on the shared
  `val_pos` / `val_neg` (`sl_comparison/auprg.auprg`), not 1 − mean AUPRC over
  a 10-fold CV of the training set. This is the comparison's selection rule
  (SLGNN: val AUROC → val AUPRG). Validation pairs are never trained on, and
  test is never consulted. The late-integration weights (step 4) keep the
  authors' inner training-set CV, because they are part of the fitted model,
  not a selection.
- **Seed.** `--seed` stands in for the authors' fixed 124 at the harness
  level: the model's `random_state` (search, inner CV shuffling) and the
  undersampling seed. `--seed 124`, the default, is the authors' setting.
  `run_elisl.sh` passes the comparison's `SEED` (42). The fixed `124 + i`
  seeds inside the vendored `fit_predict_cross_validation` are unchanged.
  LightGBM's own seeds are not set by the authors and are not set here
  (LightGBM's defaults, deterministic).
- **Matrix export.** Added: every unordered pair, scored once and mirrored
  (the features are symmetric), diagonal 0, as an 8,844 × 8,844 float32
  `fold_<k>_<cv>_predictions.npy`. Each fold checks that the exported matrix
  reproduces the model's own validation probabilities (atol 1e-6) before it
  is saved.
- **Sources.** The three cancer-specific sources are not used (above).

Input construction (no authors' code exists for some of these steps):

- **Gene names.** The shared genes are Entrez ids, and ELISL looks genes up by
  name. Each id goes to its symbol through `data/gene_id_mapping.tsv` (8,840
  of 8,844 have one), then through the authors' name-based chains: sequence,
  gene name → UniProt (or UniProt entry name → UniProt); PPI, STRING display
  name → STRING id, then UniProt entry name → STRING id.
- **Hand-curated mappings not used.** The authors' three hand-curated mapping
  files (`PPI/STRING/hand_mapping.tsv` and two `uniprot_extra_mapping.tab`) are
  in their data release, not the repository. Without them the chains above
  are all there is. `--ppi_hand_mapping` / `--ppi_extra_mapping` /
  `--seq_extra_mapping` take them if you have them. The authors'
  `get_extra_mapping` raised on a missing file; it now returns an empty map.
- **FASTA and mapping table.** `uniprot_reviewed_9606.fasta` is the current
  UniProtKB reviewed human release (the authors' release date is not
  recorded), with headers rewritten to the accession alone. The authors'
  `seqvec ... --id 0` keys each embedding by the first `|` field, which on a
  raw UniProt header is `sp`. `uniprot_reviewed_9606_mappings.csv` (gene_name
  from `GN=`, accession, entry-name prefix) is generated from the same file.
  The authors ship neither the table nor code for it. When two entries share a
  gene name, the last one wins, as in their `set_index(...).to_dict()`.
- **STRING.** v11.0 `protein.links.full` (the code reads its
  `experiments` / `database` columns) and `human.uniprot_2_string.2018`, as
  in the README. `human.name_2_string` comes from the README's URL, which
  serves the current file.
- **Duplicate symbols.** A symbol shared by two shared genes is suffixed
  `#<entrez>` in `pair_name` only, so the authors' merge on `pair_name` cannot
  duplicate rows. The lookup uses the plain symbol.
- **Genes with no embedding** get the authors' all −1 features for every pair
  they are in, so their matrix rows can be near-constant.
  `evaluate_model.py` reports that as `degenerate_row_frac`. Coverage is in
  `data/elisl/meta.json`.

Code (`elisl_orig/`, each change marked `# HARNESS:`):

- `ELRRF.py`:
  - The `from src import config` import and the path hack are dropped.
  - `clf.fit(..., verbose=False)` → `clf.fit(...)` (twice). LightGBM ≥ 4
    removed that argument, which only silenced logging.
  - `fit_predict_report` re-raises a failed fit. Before, it returned all-zero
    probabilities, which on LightGBM 4 was every fit.
  - `DataFrame.append` → `pd.concat` (removed in pandas 2).
- `features.py`:
  - Paths are parameters instead of argparse / `config.DATA_DIR` globals.
  - A missing optional mapping file gives an empty map.
  - gensim 4 names: `index_to_key` / `vectors` for `index2word` / `syn0`.
  - The per-gene print in `find_mapping` is dropped.
  - Added: a vectorised `gene_vectors` + `pair_features`.
    `test_elisl.py` checks it against the authors' `create_seq_features` /
    `create_ppi_features`, kept verbatim beside it.

Library versions: the cluster venv's LightGBM 4.x, scikit-learn 1.x, pandas 3,
gensim 4, node2vec 0.4 and scikit-optimize 0.10, not the authors' 2020 pins.
The model's parameters are passed identically. Numerical results can still
differ from a Python 3.6 run. SeqVec runs at the authors' version (0.4.1) in
its own environment.

**Parameter count** (`trainable_params`): the forests' fitted values, one
threshold per split and one value per leaf (2·leaves − 1 per tree), over the
three forests, plus the three fusion weights. Preprocessing statistics are
not counted. This is a count of a different kind from the neural models'
weights. `model_params.json` also gives it per forest.

## Inputs (build once)

```bash
# 1. login node (internet): UniProt FASTA, three STRING files, SeqVec weights
#    -> data/elisl/raw/, data/elisl/seqvec_model/, data/elisl/downloads.json
python ELISL/build_elisl_inputs.py download

# 2. once: seqvec==0.4.1 in its own Python 3.7 env (allennlp 0.9, torch < 2,
#    numpy < 2: it cannot share the cluster venv)
#    uv ships no Python 3.7: point it at an installed one (conda/micromamba
#    python=3.7). overrides==3.1.0: the unpinned 7.x makes allennlp 0.9 fail
#    on import (TypeError: Highway.forward: return type ...).
uv venv --python /path/to/python3.7 <home>/seqvec_env
uv pip install --python <home>/seqvec_env/bin/python \
    seqvec==0.4.1 overrides==3.1.0

# 3. GPU job (gpu3): FASTA rewrite -> SeqVec -> node2vec -> gene_features.npz
#    + meta.json, then check_shared_inputs.py --models ELISL
sbatch --export=ALL ELISL/slurm/build_elisl_inputs.sh
#    (SEQVEC_BIN=/path/to/seqvec to use another seqvec)
```

The SeqVec step is the authors' command,
`seqvec -i <fasta> -o uniprot_1024_embeddings.npz --model data/elisl/seqvec_model --id 0 --protein`.
The node2vec step is `ppi_embedding.main()`: STRING edges with
`experiments > 0` or `database > 0`, 64 dims, walk length 30, 200 walks,
p = q = 1, 4 workers, window 10. `meta.json` records:

- coverage per source and the unmapped genes;
- the sha256 of every input and of `build_elisl_inputs.py` /
  `elisl_orig/features.py`, which `check_shared_inputs.py --models ELISL`
  compares with the current files;
- the sha256 of `gene_features.npz`.

`siamese_sl/slurm/reset_env.sh` now installs `lightgbm` and
`scikit-optimize` into the cluster venv. Until the venv is rebuilt, run
`$PYTHON_PATH -m pip install lightgbm scikit-optimize` once.

## Run

```bash
# repo root; one CV per array task (gpu3; LightGBM uses the 12 CPUs, the GPU idles)
sbatch --export=ALL ELISL/slurm/run_elisl.sh              # cv1, cv2, cv3
sbatch --export=ALL --array=2 ELISL/slurm/run_elisl.sh    # cv3 only
NO_GRID_SEARCH=1 sbatch --export=ALL ELISL/slurm/run_elisl.sh

# by hand, outside the comparison's directories
python ELISL/train_elisl.py --cv_type cv3 --folds_dir sl_comparison/folds
python -m pytest ELISL/test_elisl.py -q
```

`run_elisl.sh` does these steps in order:

1. Moves the previous `sl_comparison/results/ELISL_<cv>.json` to
   `stale_shared/` and deletes the previous matrices, before any check.
2. Gates on `check_shared_inputs.py --models ELISL` and on
   `import lightgbm, skopt`.
3. Trains into `ELISL/results/elisl_<cv>/`.
4. Grades with `evaluate_model.py --model_name ELISL`.

The root orchestrator (`run_benchmarking_siamese_vs_others.sh`, since
2026-11-13) runs all of this. When `check_shared_inputs.py --models ELISL`
fails it runs the download on the login node, checks `SEQVEC_BIN` there
(refusing before anything changes when it is not a working seqvec) and submits
`build_elisl_inputs.sh`. It then submits `run_elisl.sh` with the shared
models, after the folds job (and the inputs job), and every arm's compare
waits for it, so the first tables already have the row. `--skip-elisl` leaves
ELISL out and keeps its previous rows; `--skip-shared` reuses them. The
commands above are for a hand rerun; after one, rerun each arm's
`run_compare.sh` and `run_paper_tables.sh` in turn, then the cross-arm table.
The row is shared:

- `compare.py` (`MODEL_ORDER`: after NSF4SL) and `bootstrap_ci.py`
  (`ELISL/results/elisl_<cv>`) read it for every PCA arm.
- `paper_tables.py` lists it in `tab_tiers` as a published peer (`TIER_PEERS`,
  so `beats_peers_*` and the numbers.json margins include it). It refuses a
  row whose recorded `elisl_sources` is not `Sequence_1024|PPI_ec`.
- It has no external-screen rows: `external_transfer.py` does not score
  ELISL, and `tab_external*` keep the three original peers.

**Cost** (estimated, not measured):

- **Search.** Each fold runs 20 search calls of three forests (up to 1,200
  trees) on the undersampled training set, plus the 10-fold weight CV.
- **Scoring.** Then about 3.9 × 10⁷ gene pairs through three forests, in
  blocks of `--chunk` 200,000 pairs. Peak memory is a few GB per block.
- **Total.** Several hours per fold, so a CV can approach the 3-day limit
  with the search on.

## Model selection

- **Search.** The search objective (above) is the only selection: forests
  have no epochs and no checkpoints. Each fold's `val_auprg` is the selected
  model's late-integrated validation AUPRG.
- **Records.** `best_params` and the full `search_trace` are kept in
  `results.json`.
- **Validation required.** A fold without a two-class validation split is a
  hard error before anything trains. There is no test fallback.

## Output

- `<output_dir>/fold_<k>_<cv>_predictions.npy`: 8,844 × 8,844 float32, in the
  shared gene order.
- `<output_dir>/model_params.json` records:
  - the fold stamp, `selection_regime: "val"`, `selection_metric: "auprg"`
    and `seed`;
  - `elisl_sources` / `context_free`, `grid_search`, `input_meta_sha256`;
  - the size, total and per forest.
- `<output_dir>/results.json` holds per-fold test metrics (computed on the
  exported matrix, the way the grader computes them), `val_auprg`,
  `best_params`, `search_trace`, `fusion_weights` and mean ± std.

## Files

| File | Purpose |
|------|---------|
| `elisl_orig/ELRRF.py` | the authors' `ELRRF` class, `fit_predict_cross_validation`, `balance_by_index`, `combine_preds` (GPL v3; changes marked) |
| `elisl_orig/features.py` | the authors' STRING / UniProt mapping chains, node2vec, pair features, plus the vectorised form (GPL v3; changes marked) |
| `build_elisl_inputs.py` | `download` / `fasta` / `seqvec` / `node2vec` / `features` (`gpu` = the last four) → `data/elisl/` |
| `train_elisl.py` | per-fold undersampling, search, weights, fit, all-pair matrix, metrics |
| `slurm/run_elisl.sh` | train + grade, `--array=0-2` (gpu3) |
| `slurm/build_elisl_inputs.sh` | the GPU input build (gpu3) |
| `test_elisl.py` | unit tests on synthetic data (vendored-code diff, feature equivalence, fusion, end-to-end trainer with and without search, refusals, input gate, launchers) |
| `LICENSE` | GPL v3, the authors' |
