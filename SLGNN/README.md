# SLGNN (PyTorch)

Factor-aware knowledge-graph GNN for synthetic-lethality (SL) prediction:
Zhu et al., *Bioinformatics* 2023, code at
[zy972014452/SLGNN](https://github.com/zy972014452/SLGNN). In the SL benchmark
(Nat. Commun. 2024) SLGNN was the runner-up classifier under the hard
cold-start split (CV3, F1 ≈ 0.737, a hair behind SLMGAE's 0.738).

Two architectures, chosen with `--arch`:

| `--arch` | What it is | Inputs |
|---|---|---|
| `original` (default) | the published model, ported line by line to pure PyTorch (`slgnn_orig_model.py`) | SynLethKG (`--kg_dir`, built by `build_synlethkg.py`), learned gene/entity embeddings |
| `port` | the earlier KG-free adaptation (`slgnn_model.py`), kept to reproduce old results | one `all_genes_*.pt` feature file (`--embedding_path`, default `kg_complex`) |

Both train on the shared folds (`--folds_dir`, required) and follow the same
selection and export rules (see Model selection).

## The original model

Every gene has a trainable gene-side embedding and, through its SynLethKG
entity, a trainable entity-side embedding. Per hop (3 by default):

- **KG aggregation**: each entity averages `entity_t * w_r` over its KG edges
  `(e, r, t)`, inverse relations included (48 relation embeddings);
- **factor prototypes** `l_f = softmax(disen_att_f) @ w`, 4 factors;
- **SL side**: each gene sums the entity embeddings of its training-SL
  neighbours and itself, plus, per factor, a 2-layer 8-head GAT over the
  training SL graph (input: the KG-aggregated entity embeddings) scaled by
  `l_f` and weighted by `softmax_f(gene · l_f)`;
- message dropout, L2 normalisation, residual sum over hops.

A pair `(a, b)` scores `gene_a · entity_b`. Loss: the original's
`BCEWithLogitsLoss(sigmoid(score), y)` (sigmoid applied twice, kept as
published) + L2 on the scored embeddings + distance-correlation independence
terms. Mini-batches of 2,048 training rows (directed pairs; each training
pair appears in both orientations), each running the full GNN; Adam, lr
3e-3, dim 64; node dropout (half the KG edges, SL-adjacency entries kept
with p = 0.5) and message dropout 0.1 during training. These are
`src/train.py`'s defaults. The authors' README run command passes
`--batch_size 1024` (its other flags equal the defaults), and SL_benchmark's
config uses 4,096. Each batch is one Adam step, so this choice sets the
steps per epoch under patience-5 early stopping.

**No dgl / torch_scatter.** `torch_scatter.scatter_mean` and
`dgl.nn.GATConv` (0.7.2) are re-implemented as the same computation
(`scatter_mean`, `edge_softmax`, `GATConv` in `slgnn_orig_model.py`, pinned to
the library formulas by `test_slgnn_orig.py`). Only float summation order can
differ, which is not fixed on a GPU in those libraries either.

**Differences from the repository:**

- Evaluation runs in `eval()` with no dropout (`score_matrix`, an added
  method). The original scores validation and test in train mode, with node,
  message and attention dropout active.
- Node dropout draws its randomness on the model's device, the one change
  inside the original's classes: `GraphConv._edge_sampling` uses
  `torch.randperm` instead of `np.random.choice`, and `_sparse_dropout` draws
  its mask on the device instead of on the CPU. Same distributions, different
  random streams, so a seeded run does not replay the original's dropout
  masks.
- Training rows are every shared-fold training pair in both orientations,
  `(a, b)` and `(b, a)`, with its label. The original's fold files list each
  of its pairs both ways, and the score is asymmetric. The SL graph built from
  these rows is the symmetric training adjacency plus self-loops. The
  difference is the split: the original splits its doubled rows at random,
  so about 20% of its training pairs have the reverse in val/test, and the
  reverse of most test pairs is in train. The shared folds split whole pairs,
  so neither happens here.
- Negatives: the original's label-0 rows are pre-sampled random pairs, 1:1.
  Here they are the shared folds' `train_neg` (screened non-SL), as for
  every model in the comparison.
- The pair order is shuffled once per fold. The original's fold files come
  pre-shuffled and its DataLoader does not shuffle.
- Early stopping and selection use validation AUPRG, not validation AUROC.
- The dense matrix mirrors the upper triangle. Validation and test pairs are
  scored once, as `a < b` (the shared folds' orientation); this is
  SL_benchmark's `final_test`. The original scores its val/test rows as
  listed, which is both orientations.

## SynLethKG

`build_synlethkg.py` downloads the original input, `data/SL/raw/kg2id.txt`,
together with the files that name its ids. It pins them to commit
`c56e23fbd20e` and checks their sha256.

- **Contents.** 2,231,921 triples, 54,012 entities, 24 relation types.
- **Relation vocabulary.** Every triple is joined back to the named
  `kg_triplet.csv`, which proves the vocabulary.
- **Label relations.** The build refuses if any triple is a label relation:
  `SL_GsG`, `SR_GsrG` (synthetic rescue) or `NONSL_GnsG`. `kg2id.txt`
  contains none of them.
- **Supervision.** The trainer's labels are each fold's `train_pos` and
  `train_neg` only. The KG itself is not label-free (next point).
- **Gene–gene edges between SL pairs are missing (upstream, kept on
  purpose).** The authors' shipped `kg2id.txt` is not `kg_triplet.csv` minus
  the three label relations. It also lacks every `INTERACTS`, `REGULATES` and
  `COVARIES` edge between two genes that SynLethKG labels SL: 1,251
  `kg_triplet.csv` rows (`INTERACTS_GiG` 741, `REGULATES_GrG` 489,
  `COVARIES_GcG` 21), all on SL pairs and none elsewhere.
  Of the 36,402 SL pairs, 1,216 have a gene–gene edge in `kg_triplet.csv` and
  0 in `kg2id.txt`; SR pairs go 29 → 25 and non-SL pairs 92 → 74, losing only
  edges on pairs that are also SL. The 100 gene–gene edges `kg2id.txt` keeps
  between labelled pairs are therefore all on SR or non-SL pairs.
  - **Effect on the shared folds.** A direct KG edge between two genes
    therefore correlates with non-SL. 83 of 22,166 shared positives (0.37%)
    have one, against 2,542 of 106,704 negatives (2.38%). Without the
    deletion the two would be 2.57% and 2.47%. This holds for val and test
    pairs too (cv1 fold 0: test positives 0.36%, test negatives 2.44%). For
    example, ATM–ATR is a cv1 fold-0 test positive; `kg_triplet.csv` has an
    `INTERACTS_GiG` edge between them and `kg2id.txt` does not.
  - **Why it stays.** SLGNN runs on its authors' input exactly as shipped, so
    the file is written unchanged. The build measures the deletion and records
    it in `meta.json` (`relation_checks.label_keyed_deletion`, `kg_source`).
  - **NSF4SL differs.** NSF4SL's TransE features (its `process_kg.ipynb`) were
    trained on `kg_triplet.csv` minus the three label relations only, so they
    keep these 1,251 edges. Of the two SynLethKG models, only SLGNN's input
    carries the deletion. Report SLGNN's numbers with this disclosed.

Gene entities map to Entrez IDs using local, tracked files only:

1. SynLethDB's own name/ID export (`data/cache/Human*.SL.detailed.tsv`), the
   KG's naming source;
2. HGNC approved symbols;
3. NCBI symbols;
4. HGNC previous symbols and aliases;
5. NCBI synonyms.

`meta.json` records the sha256 of the four files behind these lookups (the
two SynLethDB exports, `data/cache/hgnc_complete_set.txt` and
`data/embeddings_cache/Homo_sapiens.gene_info.gz`) and of
`build_synlethkg.py` itself as `mapping_sha256`, keyed by repo-relative path.
It is provenance: if any hash differs from the current file, the build was
mapped from another copy of that input, or by other mapping code (the
builder's own METHODS order and collision rules move genes with no input file
changing), so rebuild. `sl_comparison/slurm/check_shared_inputs.py` (run by
the root orchestrator before submission, by each SLGNN task and by
`slurm/run_slgnn.sh`) refuses such a build, and one without `mapping_sha256`.

Coverage of the shared 8,844 genes is **8,821**:

| Match | Genes |
|---|---|
| SynLethDB name | 8,750 |
| HGNC previous symbol | 69 |
| HGNC symbol | 1 |
| NCBI synonym | 1 |

The other **23** have no SynLethKG gene entity. By HGNC locus type (the pinned
`hgnc_complete_set.txt`) they are 7 miRNAs, 6 pseudogenes, 3 lncRNAs, 2
read-through loci, 2 protein-coding genes (GAGE2C, HNRNPCL4), 1 locus of
unknown type (C8orf44) and 2 Entrez ids the table does not list (7795,
284898). Each gets a fresh **isolated entity** with
its own trainable embedding and no KG edges, which is what the original does
for any entity without edges. The build reads no fold file (only `genes.txt`,
for this coverage count). The label-keyed deletion above comes from the
authors' `kg2id.txt`, not from the folds.

```bash
# once, on a node with internet (~100 MB); writes ../data/synlethkg (git-ignored).
# An orchestrated run leaves data/synlethkg read-only: unfreeze it first
chmod -R u+w data/synlethkg 2>/dev/null
cd SLGNN && python build_synlethkg.py
# or from an existing clone of the SLGNN repository
python build_synlethkg.py --src /path/to/SLGNN/data/SL/raw
```

## Run

```bash
# from SLGNN/ — shared folds with a validation split (see Model selection)
python train_slgnn.py --cv_type cv3 --folds_dir ../sl_comparison/folds \
    --output_dir results/slgnn_manual_cv3               # original, ../data/synlethkg

python train_slgnn.py --arch port --cv_type cv3 --folds_dir ../sl_comparison/folds \
    --embedding_path ../data/all_genes_kg_complex.pt --output_dir results/slgnn_port_manual_cv3

python -m pytest test_slgnn_orig.py -q                  # unit tests (see Files)
```

The comparison's `results/slgnn_<cv>/` is written only by
`sl_comparison/slurm/run_shared_models.sh` (its `SEED`, default 42), which
also re-grades `sl_comparison/results/SLGNN_<cv>.json`; retrain a comparison
row through that array task, not by hand. Without `--output_dir` the trainer writes
`results/slgnn_manual_<cv>/` (`--arch port`: `results/slgnn_port_manual_<cv>/`).
`slurm/run_slgnn.sh` (`cd SLGNN && sbatch slurm/run_slgnn.sh`; `ARCH=port` for
the adaptation) passes `--folds_dir ../sl_comparison/folds` and writes
`results/slgnn_standalone_<cv>/`, and stops if the KG has not been built, or
was built by older code / mapped from other gene tables (the same
`check_shared_inputs.py` gate as the comparison array; a non-default `KG_DIR`
is not provenance-checked, and says so).
Neither overwrites the comparison's directory.

## Files

| File | Purpose |
|------|---------|
| `slgnn_orig_model.py` | the original SLGNN in pure PyTorch (`GATConv`, `GAT`, `Aggregator`, `GraphConv`, `SLGNNOriginal`) |
| `build_synlethkg.py` | download + verify SynLethKG, label-relation checks, measurement of the upstream label-keyed edge deletion, Entrez mapping; `load_kg_for_genes` for the trainer |
| `test_slgnn_orig.py` | unit tests: scatter_mean / edge_softmax / GATConv against the library formulas, KG loader, score orientation, both-orientation training rows, the deletion measurement, the gene-mapping fingerprint, the standalone launcher's KG-provenance gate, the trainer's `--folds_dir` requirement, test metrics taken from the exported matrix |
| `slgnn_model.py` | `--arch port`: edge-list `GATLayer`, disentangled `SLGNN`, inner-product decoder |
| `sl_data.py` | shared-fold loader (`genes.txt` gene index, a copy of `../sl_comparison/shared_folds.py`) + `all_genes_*.pt` feature alignment (`--arch port`) |
| `train_slgnn.py` | per-fold training, validation-AUPRG early stopping, benchmark `cal_metrics` evaluation, summary JSON |
| `evaluation.py`, `preprocess_benchmarking_paper.py` | vendored from SLMGAE-in-pytorch (`cal_metrics`) |
| `data_split.py` | vendored SLMGAE cv1/cv2/cv3 splitter; not used by the trainer (it has no validation split, so the shared folds replace it) |
| `slurm/run_slgnn.sh` | SLURM launcher (nodelist gpu3; no orchestrator submits it, so `GPU_NODES` does not reach it: `sbatch --nodelist=<node> slurm/run_slgnn.sh` moves it) |

## Output

`<output_dir>/results.json` (the comparison's: `results/slgnn_<cv>/`) contains:

- per-fold and mean ± std of AUROC, F1, AUPR, AUPRG and
  NDCG / Recall / Precision / MAP@{10,20,50};
- each fold's `best_epoch` and `val_<select_metric>`;
- `arch`, plus the KG stamp or the feature file.

Each fold also writes its gene × gene score matrix as
`fold_<k>_<cv>_predictions.npy`: 8,844 × 8,844, float32, in shared order.
The fold's test metrics are computed on that same matrix (one forward pass
after the restore), so they describe the matrix `evaluate_model.py` grades.
`model_params.json` records the fold stamp, `arch`, `kg_stamp`,
`selection_regime: "val"`, `selection_metric` and the training `seed`.

## Model selection

Early stopping and checkpoint selection both use **validation AUPRG**
(`--select_metric auprg`, the default) on each fold's `val_pos` (label 1) and
`val_neg` (label 0). `prepare_folds.py --val_frac` carves both from the
fold's training pairs. AUPRG is `sl_comparison/auprg.auprg`, the function the
shared evaluator reports, applied to the float32-shifted matrix entries it
grades.

| `--arch` | Evaluated every | Patience | Epoch cap |
|---|---|---|---|
| `original` | epoch | 5 evaluations | 100 |
| `port` | 5 epochs | 10 evaluations | 500 |

The best checkpoint is restored and the test split is scored exactly once.
`--select_metric aupr` restores the previous criterion (vendored `cal_metrics`
AUPR, still on validation).

- **No test fallback.** `--folds_dir` is required, and a fold without a
  non-empty `val_pos` and `val_neg` (legacy fold sets) is a hard error before
  any fold trains.
- **NaN** validation scores count as −inf: never an improvement, they use up
  patience, and each one is logged. If a fold never records a finite score the
  trainer raises instead of exporting the last epoch's weights.
