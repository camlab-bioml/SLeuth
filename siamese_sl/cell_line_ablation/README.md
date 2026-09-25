# Cell-line ablation (post hoc)

How much of the published model's ranking comes from the cell-line embedding it
learned? This takes the **published checkpoints**, sets `cell_emb.weight` to
zero, and re-scores the same test edges. With a zero cell vector every head
encodes a gene pair to the same latent point, so the encoder can no longer tell
which cell line it is scoring.

Nothing is retrained. This measures how much the trained model *leans on* the
cell-line representation at inference, not what a model trained without cell
lines from scratch would achieve. Those are different questions and the second
one is not answered here.

## Run it

Local, CPU is fine (~4 min for all three tiers on a laptop):

```bash
cd siamese_sl
python3 cell_line_ablation/zero_cell_emb.py                      # all three CVs
python3 cell_line_ablation/zero_cell_emb.py --cv_type cv3         # one CV
python3 cell_line_ablation/zero_cell_emb.py --save_matrices       # + dense matrices
```

Needs `results/best_cat_geneformer_go2vec_kg_complex_ppi_raw_prot_t5_text_embed_<cv>/checkpoints/`
and the six `data/all_genes_*.pt` files, together with the matching shared folds.
The checkpoints and embeddings are not distributed with the source; existing
local embeddings may still be available. Restore any missing original inputs at
these paths before running the commands. Newly trained
checkpoints can be ablated in the same way but need not reproduce the archived
results below.

## Results

Mean (population sd) over the five folds, on the shared folds' test edges.
`Δ` is the mean **paired per-fold** change against as-trained — prevalence
varies a lot across folds and shifts every variant identically, so it cancels
in the difference but not in the sd column.

| Tier | variant | AUROC | AP | AUPRG |
|---|---|---|---|---|
| I (cv1) | as-trained | 0.8996 (0.007) | 0.7628 (0.011) | 0.9551 (0.004) |
| | `cell_emb=0` | 0.9051 (0.008) | 0.7471 (0.019) | 0.9521 (0.007) |
| | **Δ** | **+0.0055** | **−0.0158** | **−0.0031** |
| II (cv2) | as-trained | 0.7959 (0.047) | 0.5974 (0.061) | 0.8707 (0.076) |
| | `cell_emb=0` | 0.7410 (0.086) | 0.5221 (0.086) | 0.7826 (0.154) |
| | **Δ** | **−0.0548** | **−0.0754** | **−0.0881** |
| III (cv3) | as-trained | 0.6766 (0.031) | 0.4194 (0.089) | 0.7323 (0.136) |
| | `cell_emb=0` | 0.6624 (0.029) | 0.3923 (0.072) | 0.6490 (0.196) |
| | **Δ** | **−0.0142** | **−0.0271** | **−0.0833** |

Read with care: the effect is **not** uniform across folds, and at Tier I it is
not even signed consistently. Zeroing the embedding *improves* AUROC at Tier I
(+0.0055) and improves AUPRG on 2 of 5 folds at Tier III (+0.022, +0.015) while
costing 0.17–0.23 on two others. The Tier II and III AUPRG means (−0.088,
−0.083) sit well inside the fold spread. What the table supports is that the
cell-line embedding matters more once genes are held out than it does at Tier I,
not that it is worth a specific number of AUPRG points.

`results.json` holds the same numbers, plus every per-fold value.

## Self-check

The as-trained row is recomputed from the checkpoints, not copied, and the
script asserts it against `sl_comparison/results/comparison.csv`. All nine
cells (3 tiers × 3 metrics) reproduce the published values exactly:

```
auroc 0.8996 / 0.7959 / 0.6766      ap 0.7628 / 0.5974 / 0.4194
auprg 0.9551 / 0.8707 / 0.7323      -> match, match, match
```

If that check ever fails, the deltas are not trustworthy — a mistake in the
gene map or in the per-fold preprocessing reconstruction would show up there
first. The script says so and does not suppress the warning.

## The third row

The printed table carries a `cell_emb=0 +bias=0` row that is **identical** to
`cell_emb=0`. That is arithmetic, not a bug: once `cell_emb` is zero every head
scores a pair alike, so the max-over-heads collapse adds the single largest
per-head bias to every pair. A constant shift cannot change AUROC, AP or AUPRG,
all three being rank-based. The row is kept so that the per-head intercept is
visibly accounted for rather than silently ignored.

## Saved models

The script writes ablated checkpoints to
`models/no_cellline_<cv>/checkpoints/fold_<k>_best.pt` (15 when all three CVs and
five folds are available). These generated checkpoints are not included in the
source package.

Written in the same schema `train.py` uses, carrying `preprocessing_transform`,
`gene_order`, `embeddings_paths` and `input_dim` over verbatim, so they load
like any other checkpoint. Two things differ:

- `cell_emb.weight` is all zeros.
- `config["cell_line_ablation"]` records the variant and the source checkpoint,
  so an ablated model can never be mistaken for a published one.

`scoring_bias` is left **as trained** — it is an intercept, not a
representation, and the checkpoints are saved before the third variant zeroes
it.

`predict.py` refuses checkpoints with `cell_emb.*` keys, including these
ablated checkpoints. Score them through `zero_cell_emb.py`, use
`eval_finetuning.py --per_head` for external-screen evaluation, or call
`SiameseSLMultiCell` directly.

## Grading with the shared evaluator

The table above is computed in-process on the test edges, which is exactly what
`sl_comparison/evaluate_model.py` does for AUROC/AP/AUPRG. For the **ranking**
metrics (ndcg@k, recall@k) the evaluator needs the dense matrix:

```bash
cd siamese_sl && python3 cell_line_ablation/zero_cell_emb.py --save_matrices
cd .. && ./siamese_sl/cell_line_ablation/grade.sh
```

That writes `sl_comparison/results/siamese_sl_nocell_<cv>.json`, and
`compare.py` picks the new row up automatically (it globs `*.json` and keys on
`summary["model"]`). Budget ~313 MB per fold, ~4.7 GB for all fifteen.

**Do not** run `collect_siamese.py` to pick this up. It rediscovers the
`siamese_sl` row by scanning `siamese_sl/results` and ranking on
`val_auprg_mean`; this directory is deliberately outside that tree.

> Unrelated but live: that discovery currently ranks `ablate_no_text_cv1`
> (0.9928332) above the published combo (0.9927695), so re-running the
> comparison pipeline today would silently swap a leave-one-category-out
> ablation into the published `siamese_sl` row — the same class of error that
> once put a single-modality model into the paper's Tier II/III rows. Pin
> `SIAMESE_RESULTS_cv1/cv2/cv3` before the next comparison run.

## What this does not ablate

Only the cell-line conditioning of the **representation**, and only post hoc.
Not ablated: the masked multi-label training objective (the model was still
trained with per-head supervision), the per-head scoring bias in the saved
checkpoints, and the eight-head output shape.

A from-scratch ablation — retrain the same combo with no cell-line conditioning
— is a larger job than it looks. `--use_cell_lines` off is rejected together
with `--folds_dir` (`data_loader.py:1268`), because the shared-fold reader runs
entirely through the multi-label pair universe; and the all-pairs exporter that
writes the graded matrices is gated on `SiameseSLMultiCell`
(`train.py:1283`). Doing it properly means either editing those two paths or
training with `--cell_line_dim 0`, which keeps the multi-head plumbing and
costs exactly 576 trainable parameters (8×24 embedding + 24×16 entry columns).
