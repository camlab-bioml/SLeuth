# Four-model SL comparison (SLDB 3.0)

Fairly compares four synthetic-lethality models — **siamese_sl, SLMGAE, SLGNN,
NSF4SL** — on the SLDB 3.0 *given* positives and negatives:

- positives = `data/SL_SynLethDB_experimental.txt` (~22.2k pairs)
- negatives = `data/SL_SynLethDB_experimental_negatives.txt` (**all** ~106.7k
  used after moving the 5.5k pos/neg conflicts to the positive side; no random
  sampling)

**Label rule (`conflict_policy=or_collapse`).** A pair is positive if any screen
called it SL in any cell line, negative only if every screen of it called it
non-SL. The 5,542 pairs screened both ways are therefore positive — a screen
that detected lethality is positive evidence, one that did not is a failure to
detect in one context. `folds/conflicts.tsv` lists all 5,542 with the cell lines
and heads asserted on each side (a side naming no usable line is attributed to
`OTHER`): 1,782 disagree within one line, 3,760 across lines, 107 name no line
at all. This is the same relation siamese's `*_any` view is graded on, so every
row in the table now shares one positive set.

## How fairness is enforced

- **One canonical fold set.** `prepare_folds.py` builds the CV folds once over
  the union gene universe (8,844 genes = every gene in a given pos **or** neg
  pair, so every negative is scorable), one seed (42), and writes them to
  `folds/`. The three graph/embedding models all read the **identical**
  train/test division — calling each model's own splitter with "the same seed"
  would NOT produce identical folds (they index genes differently).
- **Graph = positives only.** For the GNN-structure models the adjacency is
  built from train **positives**; the given negatives are label-0 supervision,
  never graph edges ("no space for negatives → the rest is 0").
- **One evaluator.** `evaluate_model.py` scores every model's saved score matrix
  with the same `cal_metrics` + AUPRG, on the same test edges.
- **siamese_sl runs its own pipeline** (its embedding selection + cell-line
  model on SLDB 3.0) and reports the **OR-collapse** ("SL in any cell line",
  `*_any`) view; `collect_siamese.py` folds those metrics into the table. It
  reads the **same `folds/`** as the other three: `siamese_sl/slurm/config.conf`
  sets `SHARED_FOLDS=../sl_comparison/folds` and passes `--folds_dir` whenever
  `USE_CELL_LINES=1` (the default), so the partition, the validation split used
  for checkpoint selection, and the negatives are shared, not merely similar.
  Set `SHARED_FOLDS=""` to fall back to siamese's own gene-disjoint split.

**Headline metrics = AUPRG, AP_norm and AUROC.** AUPRG and AP_norm are
*baseline-anchored skill scores*: a random ranker scores 0 at any prevalence, so
they remain meaningful across folds whose test-set base rate differs (cv3 ranges
0.0843–0.2640 on the current folds; read the `baseline` column of the run
itself rather than this figure, which moves with the label rule). They are **not** prevalence-invariant — holding the ranker fixed
and varying π moves both. AUROC is the only genuinely prevalence-invariant
column. Since every model here shares the same per-fold test set, no cross-model
comparison in the table is affected either way.

Prefer **AP** (average precision) over the `AUPR` column. `AUPR` is the vendored
benchmark's trapezoidal `auc(recall, precision)`, kept for comparability with
published SLMGAE/SLGNN/NSF4SL numbers; it interpolates between PR points and
rewards ties, so a *constant* score matrix scores exactly (1+π)/2 — about 0.584
on the current cv3 folds, above every real model. AP gives a constant predictor
exactly π. Take π from the run's own `baseline` column: it changed when the
2026-08-10 label rule moved the 5,542 conflicted pairs to the positive side.

**`DegreePrior` is a control row, not a competitor.** `degree_baseline.py` scores
`log1p(deg_train(i)) + log1p(deg_train(j))` — no features, no learning — through
the identical evaluator. On the **2026-08-06 run** (the previous fold set) it
tied siamese_sl on cv1 (AUROC 0.9059 vs 0.9078) and beat it 3.6× on NDCG@10,
because SynLethDB positives are hub-enriched relative to the experimental
negatives; it collapsed to 0.6402 on cv2 and to exactly 0.5000 on cv3 (its
scores are constant when both genes are unseen). Those figures are from the
OLD label rule and have not been re-measured — the cv3 0.5000 is structural and
will hold, the cv1/cv2 values will move. Read cv1 results against whatever this
row scores in the same table, never against these numbers.

## Files

| File | Purpose |
|------|---------|
| `prepare_folds.py` | Generate + verify the shared leak-free CV folds → `folds/` |
| `shared_folds.py` | Canonical loader (gene universe + per-fold split dicts) |
| `evaluate_model.py` | Score a model's saved score matrices (cal_metrics + AUPRG) |
| `auprg.py` | AUPRG + normalized AUPR (vendored from siamese_sl) |
| `collect_siamese.py` | Map siamese's `*_any` metrics into the unified schema |
| `compare.py` | Print the four-model table + tidy CSV |
| `slurm/run_shared_models.sh` | Cluster orchestrator (folds → 3 models → eval → table) |

Each model consumes the shared folds via:
- **SLGNN** `SLGNN/train_slgnn.py --folds_dir ../sl_comparison/folds`
- **NSF4SL** `NSF4SL/train_nsf4sl.py --folds_dir ../sl_comparison/folds`
- **SLMGAE** `SLMGAE-in-pytorch/train_slmgae_shared.py --folds_dir ../sl_comparison/folds`
  (its GO/PPI similarity views are remapped 6,372→8,844)

## Run it

**One command (whole pipeline)** — from the repo root, submits the full SLURM
DAG (reset_env → generate → {siamese best-cat+combo} ∥ {shared 3-model job} →
final compare):

```bash
./run_benchmarking_siamese_vs_others.sh                 # full pipeline
./run_benchmarking_siamese_vs_others.sh --skip-generate # embeddings already exist
./run_benchmarking_siamese_vs_others.sh --skip-siamese  # only the 3 shared models
```

**Preflight first (recommended)** — a fast GPU smoke test of all three
shared-fold models on a tiny fold set, to catch shape/device bugs before the
full run:

```bash
sbatch sl_comparison/slurm/preflight.sh                 # PASS/FAIL per model
```

**Manual pieces** (what the orchestrator chains):

```bash
sbatch sl_comparison/slurm/run_shared_models.sh         # 3 models + eval + interim table
cd siamese_sl && ./slurm/submit_pipeline.sh             # siamese's own pipeline
# then the final table (auto-discovers siamese's *_any results):
sbatch sl_comparison/slurm/run_compare.sh
# or locally:
python sl_comparison/collect_siamese.py --discover_dir siamese_sl/results \
    --out_dir sl_comparison/results
python sl_comparison/compare.py --results_dir sl_comparison/results \
    --csv sl_comparison/results/comparison.csv
```

## SLURM scripts

| Script | Role |
|--------|------|
| `../run_benchmarking_siamese_vs_others.sh` | **root orchestrator** — submits the whole DAG (gpu3) |
| `slurm/preflight.sh` | GPU smoke test on a tiny fold set |
| `slurm/run_shared_models.sh` | train + score the 3 shared-fold models |
| `slurm/run_compare.sh` | discover siamese + print the final table |

## Notes

- **Feature inputs** (each model as intended): SLGNN + NSF4SL use `kg_complex`;
  SLMGAE uses its native SL-graph + GO/PPI similarity views; siamese uses its own
  selected embedding. The comparison controls the *data split*, not the inputs —
  it compares methods, like the SL_benchmark paper.
- **SLMGAE memory:** at 8,844 nodes its `num_support × N²` attention parameter is
  ~3.75 GB (≈15 GB with Adam state) — heavy but expected to fit on the 64 GB
  GPUs. If it OOMs, restrict negatives to genes-in-positives (smaller universe,
  fewer negatives) or lighten the attention.
- `folds/` and the per-fold score matrices are regenerable artifacts (seed 42);
  large `.npy` matrices (~313 MB each at 8,844²) are not meant to be committed.
