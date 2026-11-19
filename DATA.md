# Data

All inputs of the reported run are in `data/` and `external data/`, except the files in `DATA_NOT_INCLUDED.tsv`. Run `bash restore.sh` first: it rejoins files split for GitHub and checks every file against `DATA_SHA256SUMS`.

| Input | Source | Path |
|---|---|---|
| SL and screened non-SL pairs | SynLethDB 3.0 (`siamese_sl/download_synlethdb.py`) | `data/SL_SynLethDB_experimental*` |
| Gene embeddings (16) | `siamese_sl/generate_embeddings.py`, from the models cited in Appendix A | `data/all_genes_<name>.pt`; raw inputs in `data/embeddings_cache/` |
| GO similarity (SLMGAE) | SLMGAE authors | `data/Human_GOsim*.txt` |
| SynLethKG, TransE features (SLGNN, NSF4SL) | The baselines' authors | `data/synlethkg/`, `data/nsf4sl_kg/`, `data/nsf4sl_transe_features.pt` |
| ELISL inputs | `ELISL/build_elisl_inputs.py` | `data/elisl/` |
| Adamson γ | Simpson et al. 2023, sgRNA-pair phenotypes | `external data/pair_sgRNA_phenotypes.txt` |
| Adamson GI | Simpson et al. 2023, Table S3 | `external data/Simpson2023_TableS3_GI_scores.xlsx` |
| Corn | Fielden et al. 2025, supplementary file | `external data/41586_2025_8815_MOESM5_ESM.csv` |
| Gilbert | Horlbeck et al. 2018, supplementary file | `external data/mmc5.xlsx` |

Not included (`DATA_NOT_INCLUDED.tsv`, with SHA-256): third-party model weights (scGPT, SeqVec), the BioConceptVec download (`concept_cbow.json`, 0.8 GB) and GenePT, which the paper does not use. These are needed only to regenerate the embeddings, which are included.
