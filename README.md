# SLeuth: code, results and run logs

Code, results and cluster logs for "Simple Siamese Networks on Foundation Model Embeddings Outperform Complex Models on Unseen Synthetic Lethality Prediction" (ICLR 2027 submission). Every number in the paper can be traced to a file here.

## Layout

| Folder | Contents |
|---|---|
| `siamese_sl/` | SLeuth: training (`train.py`), model (`siamese_esm.py`), external-screen evaluation (`eval_finetuning.py`), Slurm scripts, results and logs |
| `sl_comparison/` | Shared folds, evaluator, baselines' grading, comparison, bootstrap CIs, paper tables |
| `SLMGAE-in-pytorch/`, `SLGNN/`, `NSF4SL/`, `ELISL/` | Baselines, run with the authors' code and inputs |
| `analysis/` | Scripts behind the significance tests, appendix tables and figures, with their outputs |
| `data/` | Inputs included in the repository (see [Data](#data)) |
| `run_benchmarking_siamese_vs_others.sh` | Runs the whole pipeline on Slurm |
| `*/slurm/logs/` | Slurm output of every job in the reported run |

Tiers I, II and III are `cv1`, `cv2` and `cv3` in file names. Model names: SLeuth = `pd_pooled`, SLeuth-CL = `pd_per_head` (no prefix), SLeuth-all = `allmod`. Results without a `pca_` folder are the reported model (no PCA). `pca_joint/` holds the PCA runs.

## Quick start (no training)

```bash
bash restore.sh        # rejoin split files, unzip screen predictions, check data present against DATA_SHA256SUMS
python3 sl_comparison/compare.py --results_dir sl_comparison/results         # model comparison (Table 1 numbers)
python3 sl_comparison/paper_tables.py --pca_arm none --out_dir /tmp/tables   # rebuild the paper-table CSVs
python3 analysis/make_tex.py                                                 # rebuild the LaTeX tables
python3 analysis/figures/make_figures.py                                     # rebuild Figures 1 and 2
```

`compare.py` rewrites `comparison.csv` byte for byte. `paper_tables.py` reproduces every table except the bilinear-probe rows of Table 1: they record a hash of the combo's `config.json`, which changed when paths were anonymized (see Integrity), so they are listed as missing.

## Where each result is

`PT` = `sl_comparison/results/paper_tables/`, `AN` = `analysis/`.

| Paper | File |
|---|---|
| Table 1 (tiers) | `PT/tab_tiers.csv`; tests: `AN/tiers_significance/out/table1_none.csv`, `sig_foldpaired.csv` |
| Table 2 (single embeddings) | `PT/tab_solo.csv`; tests: `AN/pca_solo_loco_sel/s2_solo_vs_multimodal_paired.csv` |
| Table 3 (category removed) | `PT/tab_loco.csv`; tests: `AN/pca_solo_loco_sel/s3_loco_paired.csv` |
| Table 4 (cell line) | `PT/tab_cellline.csv` |
| Table 5 (zero-shot) | `AN/external_screens/out_none/table5_zero_shot.csv`, `table5_diffs.csv` |
| Table 6 (fine-tuning) | `AN/external_screens/out_none/table6_finetune.csv`, `table6_tests.csv` |
| Table 7 (label agreement) | `AN/item11_sldb_agreement/agreement_split_points.csv`, `agreement_split_diffs.csv`, `extra_same_map_contrast.csv` |
| Table 8 (screens) | `siamese_sl/slurm/eval_finetuning.conf` (`DATASETS`) |
| Table 9 (paired comparisons) | `AN/tiers_significance/out/sig_foldpaired.csv` |
| Table 10 (selected embeddings) | `AN/pca_solo_loco_sel/s4_winners_per_fold.csv` |
| Table 11 (PCA) | `AN/pca_solo_loco_sel/s1_pca_paired.csv` |
| Table 12 (single embeddings, SLeuth-CL) | `PT/tab_solo.csv` (`siamese_sl` rows) |
| Tables 13–14 (zero-shot CIs, differences) | as Table 5 |
| Table 15 (fine-tuning, SLeuth-CL) | as Table 6 |
| Tables 16–19 | as Tables 2, 3, 4 and 6 |
| Other numbers in the text | `PT/numbers.json` (index: `PT/README.md`) |

Per-model metrics: `sl_comparison/results/<model>_cv<k>.json` (`siamese_sl_pd_pooled` = SLeuth, `siamese_sl` = SLeuth-CL, `siamese_sl_allmod*`, `siamese_sl_nocell` = cell line zeroed). All rows: `sl_comparison/results/comparison.csv`. `analysis/make_tex.py` rebuilds the LaTeX tables from these files.

## Integrity

- All results and logs are from one run, token `20261113T113339-1225758` (Slurm jobs 1265xxx). Every `run_token.txt` and `numbers.json` records it.
- `sl_comparison/slurm/logs/folds_*.sha256` was written when the run was submitted. The folds here match it: `cd sl_comparison/folds && shasum -a 256 -c ../slurm/logs/folds_*.sha256`.
- `sl_comparison/slurm/logs/code_*.sha256` hashes the code the jobs ran (`shasum -a 256 -c` from the repo root). Every file matches except six Slurm scripts whose cluster paths were replaced for anonymous review, and one macOS metadata file (`ELISL/._build_elisl_inputs.py`) that is not included.
- The reported model's checkpoints are included (`siamese_sl/results/**/checkpoints/`). `EXCLUDED_FILES.tsv` lists what is left out for size (PCA-arm checkpoints, per-pair probability files), with SHA-256. Files over 100 MB are split into `.part-*` and the screen predictions gzipped; `restore.sh` undoes both.
- For anonymity, cluster and local paths in logs, configs and results were replaced by `<repo>`, `<home>` and `<project>`. Hashes over files that contain paths (for example a combo's `config.json`) therefore differ from those recorded in the run. The same paths stored inside 330 checkpoints were rewritten the same way, weights unchanged; `ANONYMIZED_CHECKPOINTS.tsv` maps each checkpoint's original SHA-256 (the one the run recorded) to its new one.

## Data

`DATA_SHA256SUMS` lists the SHA-256 of every input file of the reported run, including those not in the repository; `restore.sh` checks the ones present.

### External screens (not included; download from the source)

These files belong to their authors and are not redistributed here. Download each one, save it under `external data/` (with the space) using the name in the last column, and check it with `shasum -a 256 -c DATA_SHA256SUMS --ignore-missing`.

| Screen | Source | File to download | Save as |
|---|---|---|---|
| Adamson γ | Simpson, Ling, Jing & Adamson (2023), *Mapping the Genetic Interaction Network of PARP inhibitor Response*, bioRxiv, [doi:10.1101/2023.08.19.553986](https://doi.org/10.1101/2023.08.19.553986) | sgRNA-pair phenotype table (columns `FirstGene`, `SecondGene`, `Gamma`, `Tau`, `Rho`) | `pair_sgRNA_phenotypes.txt` |
| Adamson GI | same preprint | Supplementary Table S3, [`media-3.xlsx`](https://www.biorxiv.org/content/biorxiv/early/2023/08/20/2023.08.19.553986/DC3/embed/media-3.xlsx?download=true) | `Simpson2023_TableS3_GI_scores.xlsx` |
| Corn | Fielden et al. (2025), *Comprehensive interrogation of synthetic lethality in the DNA damage response*, Nature 640:1093–1102, [doi:10.1038/s41586-025-08815-4](https://doi.org/10.1038/s41586-025-08815-4) | Supplementary file `41586_2025_8815_MOESM5_ESM.csv` | `41586_2025_8815_MOESM5_ESM.csv` |
| Gilbert | Horlbeck et al. (2018), *Mapping the Genetic Landscape of Human Cells*, Cell 174:953–967, [doi:10.1016/j.cell.2018.06.010](https://doi.org/10.1016/j.cell.2018.06.010) | Supplementary table `mmc5.xlsx` | `mmc5.xlsx` |

Columns used are in Appendix Table 8 of the paper and `siamese_sl/slurm/eval_finetuning.conf` (`DATASETS`).

### Other inputs

| Input | How to obtain |
|---|---|
| SynLethDB pairs | `cd siamese_sl && python download_synlethdb.py --output ../data/SL_SynLethDB_experimental.txt` and `--relation NONSL --output ../data/SL_SynLethDB_experimental_negatives.txt` (from [SynLethDB](https://www.synlethdb.com/download)); the `.sources.tsv` sidecars are included |
| Gene embeddings | `siamese_sl/slurm/run_generate_embeddings.sh` (sources and models in Appendix A of the paper); intermediate caches are in `data/embeddings_cache/` |
| SLMGAE GO similarity | `data/Human_GOsim.txt`, `data/Human_GOsim_CC.txt` from the original SLMGAE release (Hao et al., 2021) |
| SLGNN, NSF4SL, ELISL inputs | `SLGNN/build_synlethkg.py`, `NSF4SL/build_transe_features.py`, `ELISL/build_elisl_inputs.py` |
| Shared folds | Included: `sl_comparison/folds/` |

## Running

Python 3.12 (the cluster environment: `siamese_sl/slurm/config.conf`). Obtain the inputs above, then `bash run_benchmarking_siamese_vs_others.sh` on a Slurm cluster with GPUs. `siamese_sl/slurm/README.md` and `sl_comparison/README.md` describe each stage.
