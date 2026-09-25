# NSF4SL (PyTorch)

Negative-sample-free contrastive learning for ranking synthetic-lethal (SL)
gene pairs — Wang et al., *Bioinformatics* 2022. In the SL benchmark
(Nat. Commun. 2024) NSF4SL was among the top classifiers under the hard
cold-start split (CV3, F1 ≈ 0.685, third behind SLMGAE and SLGNN).

This is a faithful port of the benchmark reference
(`SL_benchmark/src/models/nsf4sl.py`), wired into the same data/CV/evaluation
machinery as `../SLMGAE-in-pytorch` so results are directly comparable.

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

## Adaptation vs. the original

Only the **feature source** differs: instead of TransE knowledge-graph
embeddings from a fixed `.npy`, per-gene features come from any repo
`all_genes_*.pt` file. Default: `kg_complex` (the closest analog). Everything
else — architecture, loss, momentum update, augmentation — matches the reference.

## Run

```bash
# from NSF4SL/
python train_nsf4sl.py --cv_type cv3 \
    --embedding_path ../data/all_genes_kg_complex.pt \
    --output_dir results/nsf4sl_cv3

# cluster (gpu3) — runs cv1, cv2, cv3
cd NSF4SL && sbatch slurm/run_nsf4sl.sh
```

## Files

| File | Purpose |
|------|---------|
| `nsf4sl_model.py` | `Net` (online/target/predictor) + BYOL loss + score matrix |
| `sl_data.py` | SL edges + gene list (shared index with SLMGAE) + `all_genes_*.pt` feature alignment |
| `train_nsf4sl.py` | CV training loop, benchmark `cal_metrics` evaluation, summary JSON |
| `data_split.py`, `evaluation.py`, `preprocess_benchmarking_paper.py` | vendored from SLMGAE-in-pytorch (cv1/cv2/cv3 + `cal_metrics`) |
| `slurm/run_nsf4sl.sh` | SLURM launcher (nodelist gpu3) |

## Output

`results/nsf4sl_<cv>/results.json` — per-fold and mean±std of AUROC, F1, AUPR and
NDCG/Recall/Precision/MAP@{10,20,50} (the benchmark metric set). Per-fold
gene×gene score matrices saved as `fold_<k>_<cv>_predictions.npy`.

**Selection convention:** the best checkpoint per fold is chosen by test AUPR —
the same "select on test metric" convention as this repo's SLMGAE trainer.
