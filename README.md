# SLeuth — ICLR submission code

Code and recorded results for **Simple Siamese Networks on Foundation Model Embeddings Outperform Complex Models on Unseen Synthetic Lethality Prediction**.

SLeuth is one model family: a shared residual encoder, positive-definite pair scoring, and eight cell-line heads. The paper varies the input embeddings and evaluation splits. CV1, CV2, and CV3 correspond to Tiers I, II, and III.

## Paper map

| Paper material | Where to find it |
|---|---|
| Architecture, preprocessing, masked objective | `siamese_sl/siamese_esm.py`, `data_loader.py`, `train.py`, `cell_line_vocab.py` |
| Embedding construction; appendix | `siamese_sl/generate_all_genes_esm.py`, `generate_go_esm_embeddings.py`, `generate_embeddings.py` |
| Shared splits and evaluation; Table 1 | `sl_comparison/folds/`, `prepare_folds.py`, `evaluate_model.py`, `results/comparison.csv` |
| SLMGAE, SLGNN, NSF4SL baselines | `SLMGAE-in-pytorch/train_slmgae_shared.py`, `SLGNN/train_slgnn.py`, `NSF4SL/train_nsf4sl.py` |
| Single-modality models; Table 2 | `siamese_sl/slurm/run_best_per_category.sh`; `siamese_sl/slurm/logs/bestcat_*.out` |
| Leave-one-category-out; Table 3 | `siamese_sl/slurm/run_ablation.sh`; `siamese_sl/results/ablation_summary.csv` |
| Cell-line embedding ablation; Table 4 | `siamese_sl/cell_line_ablation/zero_cell_emb.py`, `results.json` |
| SLeuth external transfer and fine-tuning; Tables 5–6 | `siamese_sl/eval_finetuning.py`; `siamese_sl/results/best_cat_geneformer_go2vec_kg_complex_ppi_raw_prot_t5_text_embed_cv3/eval_finetuning_cv*/` |
| Input pairs, feature sources, external screens | `data/`; `external data/` |

## Getting started

From the repository root, inspect the saved benchmark without training:

```bash
python3 sl_comparison/compare.py --results_dir sl_comparison/results
```

For training, use Python 3.12 and install the dependencies in [siamese_sl/SETUP.md](siamese_sl/SETUP.md). Configure interpreter paths, environment modules, and SLURM resources in `siamese_sl/slurm/config.conf` and the job scripts for your system. The pipeline generates embeddings, trains the shared-fold models, and runs category ablations and SLeuth external evaluation:

```bash
bash run_benchmarking_siamese_vs_others.sh --skip-reset-env
```

This command assumes an existing configured environment. Individual workflows are described in [siamese_sl/slurm/README.md](siamese_sl/slurm/README.md) and [sl_comparison/README.md](sl_comparison/README.md). Preserve the supplied folds and seed 42 when comparing with the recorded results. To check preprocessing:

```bash
python3 -m pytest siamese_sl/tests -q
```

