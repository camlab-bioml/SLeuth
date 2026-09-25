# SLGNN (adapted, PyTorch)

Factor-aware GNN for synthetic-lethality (SL) prediction — Zhu et al.,
*Bioinformatics* 2023. In the SL benchmark (Nat. Commun. 2024) SLGNN was the
runner-up classifier under the hard cold-start split (CV3, F1 ≈ 0.737, a hair
behind SLMGAE's 0.738).

This directory is an **adaptation** of SLGNN wired into the same
data/CV/evaluation machinery as `../SLMGAE-in-pytorch` so results are directly
comparable.

## What it does

A **disentangled, factor-aware** graph neural network over the SL graph:

- per-gene input features (any `all_genes_*.pt`; default `kg_complex`) projected
  to the model dim as the initial node embedding;
- `n_factors` latent factors, each with its **own GAT** over the SL graph, mixed
  per gene by a softmax **factor-attention** (`softmax(gene · latentᵀ)`);
- a **factor-independence** regularizer (distance correlation between factor
  prototypes) — SLGNN's `_cul_cor` term;
- an **inner-product decoder** on the final gene embeddings, trained full-graph
  with BCE on labeled pairs + an L2 embedding regularizer.

## Adaptation vs. the original — read this

The published SLGNN is a *knowledge-graph* model: it runs relational
aggregation over **SynLethKG** (h/r/t triples) using **DGL's GATConv** and
**torch_scatter**. This repo has neither the raw KG nor those dependencies, so
this port **drops the external KG** and computes attention on the sparse SL edge
list in pure PyTorch (no DGL / torch_scatter). It keeps SLGNN's signature ideas
(disentangled per-factor GATs + factor gating + independence regularizer) but is
**not** the exact KG version — treat it as "SLGNN-style", not a reproduction of
the paper's CV3 numbers.

To build the faithful KG version instead, you'd need the SynLethKG `kg2id.txt`
triples plus `dgl` + `torch_scatter` added to `reset_env.sh`.

## Run

```bash
# from SLGNN/
python train_slgnn.py --cv_type cv3 \
    --embedding_path ../data/all_genes_kg_complex.pt \
    --output_dir results/slgnn_cv3

# cluster (gpu3) — runs cv1, cv2, cv3
cd SLGNN && sbatch slurm/run_slgnn.sh
```

## Files

| File | Purpose |
|------|---------|
| `slgnn_model.py` | edge-list `GATLayer`, disentangled `SLGNN`, distance-correlation reg, inner-product decoder |
| `sl_data.py` | SL edges + gene list (shared index with SLMGAE) + `all_genes_*.pt` feature alignment |
| `train_slgnn.py` | full-graph CV training loop, benchmark `cal_metrics` evaluation, summary JSON |
| `data_split.py`, `evaluation.py`, `preprocess_benchmarking_paper.py` | vendored from SLMGAE-in-pytorch (cv1/cv2/cv3 + `cal_metrics`) |
| `slurm/run_slgnn.sh` | SLURM launcher (nodelist gpu3) |

## Output

`results/slgnn_<cv>/results.json` — per-fold and mean±std of AUROC, F1, AUPR and
NDCG/Recall/Precision/MAP@{10,20,50}. Per-fold gene×gene score matrices saved as
`fold_<k>_<cv>_predictions.npy`. Best checkpoint per fold selected by test AUPR
(same convention as this repo's SLMGAE trainer).
