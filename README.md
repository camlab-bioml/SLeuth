# SLeuth

Code, results and run logs for the ICLR 2027 submission "Simple Siamese Networks on Foundation Model Embeddings Outperform Complex Models on Unseen Synthetic Lethality Prediction".

## Quick start (no training)

```bash
bash restore.sh                                                              # unzip the screen predictions
python3 sl_comparison/compare.py --results_dir sl_comparison/results         # model comparison (Table 1)
python3 sl_comparison/paper_tables.py --pca_arm none --out_dir /tmp/tables   # rebuild the paper tables
```

## Where the results are

| Paper | File |
|---|---|
| Table 1 | `sl_comparison/results/paper_tables/tab_tiers.csv` |
| Table 2 | `sl_comparison/results/paper_tables/tab_solo.csv`; multimodal rows: `analysis/pca_solo_loco_sel/s1_pca_paired.csv` |
| Table 3 | `sl_comparison/results/paper_tables/tab_loco.csv` |
| Table 4 | `sl_comparison/results/paper_tables/tab_cellline.csv` |
| Table 5 | `analysis/external_screens/out_none/table5_zero_shot.csv` |
| Table 6 | `analysis/external_screens/out_none/table6_finetune.csv` |
| Table 7 | `analysis/item11_sldb_agreement/agreement_split_points.csv` |
| Other numbers | `sl_comparison/results/paper_tables/numbers.json` |
| Figures 1 and 2 | `analysis/figures/make_figures.py` |

Tiers I, II and III are `cv1`, `cv2` and `cv3`. SLeuth is `pd_pooled` and SLeuth-CL is `pd_per_head`. Slurm logs are in `siamese_sl/slurm/logs/` and `sl_comparison/slurm/logs/`.

## Data

The external screens are not included. Download them into a folder named `external data/`:

| Save as | Source |
|---|---|
| `pair_sgRNA_phenotypes.txt` | Simpson et al. (2023), [doi:10.1101/2023.08.19.553986](https://doi.org/10.1101/2023.08.19.553986), sgRNA-pair phenotypes |
| `Simpson2023_TableS3_GI_scores.xlsx` | Simpson et al. (2023), [Table S3](https://www.biorxiv.org/content/biorxiv/early/2023/08/20/2023.08.19.553986/DC3/embed/media-3.xlsx?download=true) |
| `41586_2025_8815_MOESM5_ESM.csv` | Fielden et al. (2025), [doi:10.1038/s41586-025-08815-4](https://doi.org/10.1038/s41586-025-08815-4), supplementary file |
| `mmc5.xlsx` | Horlbeck et al. (2018), [doi:10.1016/j.cell.2018.06.010](https://doi.org/10.1016/j.cell.2018.06.010), supplementary table |

SynLethDB pairs: `siamese_sl/download_synlethdb.py` ([SynLethDB](https://www.synlethdb.com/download)). Gene embeddings: `siamese_sl/slurm/run_generate_embeddings.sh`. Baseline inputs: `SLGNN/build_synlethkg.py`, `NSF4SL/build_transe_features.py` and `ELISL/build_elisl_inputs.py`; SLMGAE's GO similarity files come from the original SLMGAE release.

## Run everything

`bash run_benchmarking_siamese_vs_others.sh` on a Slurm cluster with GPUs.

## License

MIT. `ELISL/` is under the GNU GPL v3.
