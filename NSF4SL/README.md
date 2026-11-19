# NSF4SL (PyTorch)

Negative-sample-free contrastive learning for ranking synthetic-lethal (SL)
gene pairs — Wang et al., *Bioinformatics* 2022. In the SL benchmark
(Nat. Commun. 2024) NSF4SL was among the top classifiers under the hard
cold-start split (CV3, F1 ≈ 0.685, third behind SLMGAE and SLGNN).

This runs the original model (`JieZheng-ShanghaiTech/NSF4SL` `model.py`, the
same code as `SL_benchmark/src/models/nsf4sl.py`) on its original input, the
SynLethKG TransE entity embeddings, wired into the shared folds and evaluator
(`../sl_comparison`) so results are directly comparable with the other models.

## What it does

BYOL-style dual encoder, **no negative samples**:

- an **online** encoder + an EMA **target** encoder (MLP: `d → 512 → 256 → latent`),
  with a linear **predictor** on the online branch;
- feature-masking **augmentation** (a random `aug_ratio` of feature dims reset
  to the column mean);
- symmetric **bootstrap loss** (`2 − 2·cos`) between the two genes of each SL pair.

Each fold trains only on that fold's **positive** pairs; the score matrix
`P(g)·O(g′)ᵀ + O(g)·P(g′)ᵀ` is evaluated against the fold's test positives +
negatives.

## Features: the original SynLethKG TransE embeddings

The original feeds each gene its row of `data/kg_embed/kg_TransE_l2_entity.npy`
(400-d TransE_l2 of SynLethKG, trained with DGL-KE after the `SL_GsG`,
`NONSL_GnsG` and `SR_GsrG` edges were removed, so no SL label is in the KG
embedding), **raw**, and augments by resetting a random 10% of dimensions to the
mean over **all** 54,012 KG entities. `--features transe` (the default) does
exactly that. `build_transe_features.py` downloads the pinned upstream files
(NSF4SL commit `de000de`, KG4SL commit `be3626a`, sha256-checked) into
`../data/nsf4sl_kg/` and writes `../data/nsf4sl_transe_features.pt` (both
git-ignored). The file records the sha256 of the HGNC table, the symbol
correction table, `gene_name_utils.py` and the builder itself;
`../sl_comparison/slurm/check_shared_inputs.py` (run by the root orchestrator
before submission, by each NSF4SL benchmark task, and by `slurm/run_nsf4sl.sh`
on the default file) refuses it once any of those changes, so rebuild after
editing them (the raw inputs stay cached). It links
KG entities to our Entrez ids with this chain:

- TransE row → entity id (`entities.tsv`)
- entity id → db id → name (KG4SL `entity2id.txt`, `dbid2name.csv`)
- gene entities: those at a Gene endpoint of a relation in `kg_triplet.csv`
- name → Entrez: SynLethDB 2.0's own id (`Human_SL.csv`) first, otherwise the
  repo's HGNC normaliser (`siamese_sl/gene_name_utils.py`)

The chain reproduces NSF4SL's own `id_mapping.csv` for all 9,258 of its SL
genes. It covers 8,820 of the 8,844 shared genes. The other 24 touch 22 of
22,166 positives. By HGNC locus type (the pinned `hgnc_complete_set.txt`) they
are 7 miRNAs, 5 pseudogenes, 3 lncRNAs, 2 read-through loci, 2 protein-coding
genes (GAGE2C, HNRNPCL4), 1 locus of unknown type (C8orf44) and 4 Entrez ids
the table does not list (7795, 284379, 284898, 102724560). 23 of them have no
TransE row: SynLethKG holds them only as endpoints of the removed SL edges, as
isolated nodes, or not at all. The 24th, LOC284379, has a TransE row, but
neither `Human_SL.csv` nor HGNC links its name to an Entrez id (it is not one
of the original's SL genes either). All 24 get the all-entity mean vector, the
same value the augmentation writes into a masked dimension, i.e. a fully
masked gene. That fill comes from the embedding alone, not from any fold, so
it cannot leak. Each run prints the count and `results.json` lists the genes
(`genes_without_features`); a `--features embedding` run lists the same way the
genes that file has no usable row for (absent, or an all-NaN row), which get
the column mean.

## Differences from the original, and why

| | original `main.py` | here |
|-|-|-|
| model, loss, EMA, predictor, init, score `P·Oᵀ + O·Pᵀ` | `model.py`, `utils.py` | identical |
| features | TransE, raw | identical (`--features transe`) |
| positives per epoch | both orientations | identical (the shared folds list each pair once; they are mirrored) |
| optimiser / schedule | Adam 1e-3, wd 5e-4, StepLR(10, 0.1), 100 epochs, batch 256, latent 256, momentum 0.995, aug 0.1 | identical |
| early stopping | every 10 epochs, stop at the 4th non-improving evaluation in total | identical counting; the metric differs (next row) |
| selection metric | validation P@100 | **validation AUPRG** (`../sl_comparison/auprg.py`), the one rule all benchmark models use |
| last partial batch | kept | kept, except a batch of one pair (BatchNorm cannot train on it; the original would crash) |
| folds / scoring universe | its own splits, its SL genes | the shared folds, all 8,844 shared genes |
| seed | 0, augmentation unseeded | `--seed`, augmentation seeded per fold |

`--features embedding --embedding_path <all_genes_*.pt> --legacy_loop
--select_metric ndcg@50` restores the configuration of this port's runs before
2026-09-21 (`kg_complex`, z-scored, each positive once per epoch, patience
reset on improvement, checkpoints selected on validation NDCG@50). It does not
reproduce those runs exactly: validation NDCG@50 is now computed on the
grader's shifted matrix (`score - min + 1`), while the earlier runs ranked the
raw one, where `cal_metrics`' literal-0 diagonal can outrank negative-scored
candidates, so the selected epochs can differ; and the trainer's own test
summary now masks every known positive except the fold's test positives (as
the grader does) instead of only `train_pos`. Without `--select_metric
ndcg@50` they select on AUPRG like every model now does.

## Run

```bash
# from NSF4SL/, once, on a machine with internet access
python build_transe_features.py --download

# writes results/nsf4sl_standalone_cv3/ (seed 123 unless --seed)
python train_nsf4sl.py --cv_type cv3 --folds_dir ../sl_comparison/folds

# cluster (gpu3) — runs cv1, cv2, cv3
cd NSF4SL && sbatch slurm/run_nsf4sl.sh
```

`results/nsf4sl_<cv>/` belongs to the benchmark pipeline
(`../sl_comparison/slurm/run_shared_models.sh`, `--seed $SEED`, 42 by
default): it is graded into `sl_comparison/results/NSF4SL_<cv>.json` and read
again by external transfer, so do not point a manual run at it.

## Files

| File | Purpose |
|------|---------|
| `nsf4sl_model.py` | `Net` (online/target/predictor) + BYOL loss + score matrix |
| `build_transe_features.py` | Downloads the original SynLethKG TransE embeddings and maps KG genes to Entrez |
| `sl_data.py` | shared-fold loader (`genes.txt` gene index, a copy of `../sl_comparison/shared_folds.py`) + TransE / `all_genes_*.pt` feature alignment |
| `train_nsf4sl.py` | CV training loop, benchmark `cal_metrics` evaluation, summary JSON |
| `evaluation.py`, `preprocess_benchmarking_paper.py` | vendored from SLMGAE-in-pytorch (`cal_metrics`) |
| `data_split.py` | vendored SLMGAE cv1/cv2/cv3 splitter; not used by the trainer (it has no validation split, so the shared folds replace it) |
| `slurm/run_nsf4sl.sh` | SLURM launcher (nodelist gpu3; no orchestrator submits it, so `GPU_NODES` does not reach it: `sbatch --nodelist=<node> slurm/run_nsf4sl.sh` moves it) |

## Output

`<output_dir>/results.json` (`results/nsf4sl_standalone_<cv>/` by default,
`results/nsf4sl_<cv>/` from the benchmark pipeline) — per-fold and mean±std of
AUROC, F1, AUPR and NDCG/Recall/Precision/MAP@{10,20,50} (the benchmark metric
set). Per-fold gene×gene score matrices saved as `fold_<k>_<cv>_predictions.npy`.
`model_params.json` beside them records the fold stamp, seed, selection rule
and which features / loop were used (`features`, `feature_file`,
`standardized`, `legacy_loop`); `sl_comparison/evaluate_model.py` copies it
into the graded row.

**Selection convention:** early stopping and checkpoint selection use **AUPRG on
the validation split** (`val_pos` = 1, `val_neg` = 0 from the shared folds,
carved from each fold's training pairs), computed with
`../sl_comparison/auprg.py` on the same pair scores that are exported for
grading — the criterion shared by all four models. Evaluation every 10 epochs,
stopping at the 4th non-improving evaluation in total (as the original counts
it); the best checkpoint is restored and test is scored exactly once.
`--select_metric ndcg@50` (or any other `cal_metrics` key) restores the earlier
ranking-based selection. A NaN validation score counts as −inf (logged; never
selected). `--folds_dir` is required and must point at a `prepare_folds.py`
set with a validation partition (as `slurm/run_nsf4sl.sh` does, writing
`results/nsf4sl_standalone_<cv>/`); a fold set without a two-class validation
split is refused rather than selected on test.
