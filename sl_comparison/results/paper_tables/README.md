# Paper tables and numbers: PCA arm none

PAPER COMPLETE

Generated 2026-11-16T20:50:02 by sl_comparison/paper_tables.py from <repo>.

## Tables

| Table | Status |
|---|---|
| tab_tiers.csv | ok |
| tab_solo.csv | ok |
| tab_loco.csv | ok |
| tab_cellline.csv | ok |
| tab_external.csv | ok |
| tab_external_ft.csv | ok |
| tab_sldb_agreement.csv | ok |

## Where each paper item is

In numbers.json each `items.<key>` is `{paper, value, status}`, so a key path `items.<key>.<rest>` in this README means `items.<key>.value.<rest>` (e.g. `items.fold_geometry.cv1.test_prevalence` is `items.fold_geometry.value.cv1.test_prevalence`).

| Paper item | File | Column / key |
|---|---|---|
| The PCA arm of every table and number here | every CSV / numbers.json / README.md | pca_arm (the first column of every CSV, on every row); run.pca_arm / pca (its label) / pca_targets / pca_settings (the PCA the combos trained with) / sia_results / cmp_results / step1_results; README.md line 1 and the first Run settings bullet. The arms side by side: sl_comparison/results/pca_arms/ (--pca_arms_table) |
| tab:tiers AUROC / AUPR / AUPRG mean (sd) | tab_tiers.csv | auroc_mean/_sd, ap_mean/_sd (the paper's AUPR is average precision), auprg_mean/_sd; bold = best_<metric> (among the SLeuth variants, the three peers and, in PCA arm none, the BilinearProbe baseline (kind baseline: siamese_sl/slurm/run_probe.sh); ties at the printed 3 decimals are all bold; empty for a tier with one of them missing), beats_peers_<metric> per variant (on the exact values, against the three published peers only, never the probe; empty unless all three peers have a row). With ALLMOD_COMBO=1 also kind ablation, rows siamese_sl_allmod[_<tag>] (siamese_sl/slurm/run_allmod_combo.sh: every catalog embedding, no modality selection), after the SLeuth rows: eligible for the bold, never beats_peers_*; params_fold_* from comparison.csv trainable_params_fold |
| tab:tiers trainable parameters | tab_tiers.csv | SLeuth: params_fold_mean / params_fold_sd over folds (combo results.json fold_metrics[].total_params); peers: trainable_params (empty on SLeuth rows: comparison.csv's copy is the last fold's count only) |
| tab:tiers caption pi (TT1) | tab_tiers.csv / numbers.json | baseline_mean (cv1); items.fold_geometry.cv1.test_prevalence |
| tab:tiers / tab:cellline row provenance | numbers.json | items.fold_provenance (every row's graded-on / trained-on fold fingerprints vs the current folds); a row not graded by evaluate_model.py or not on the current folds is left out and listed under Missing |
| tab:solo (+AB1) | tab_solo.csv | auprg_any_mean / auprg_any_sd per variant x embedding x tier; bold = best_in_category_paper16 (over the paper's 16 printed rows; ties at the printed 3 decimals are all bold; best_in_category is the exact maximum over every catalog embedding that ran; both empty for a single-candidate category such as kg, the paper's also for a category with one of its paper-16 members not fresh); in_paper marks the paper's 16; status |
| AB2 category means | numbers.json | items.solo.<variant>.per_cv.<cv>.category_means_paper16 / best_category_paper16 (over the paper's 16, the paper's definition; a category's mean is empty with one of its members not fresh, the best category with any of the 16 not fresh: paper16_absent names them); category_means / best_category over every catalog embedding that ran (unrounded) |
| AB3 / D4 single modalities above the multi-modal model | numbers.json | items.solo.<variant>.per_cv.<cv>.n_above_combo_shared_paper16 (the paper's 'of the sixteen', vs the tab:tiers AUPRG; empty unless all 16 are fresh) / _self (vs the combo's own auprg_any); n_above_combo_* over every catalog embedding that ran; above_combo_at_every_tier_* (empty while one of the 16 is undecided, i.e. without a fresh row at a tier; *_undecided) |
| tab:loco (+AB4) | tab_loco.csv | auprg_any_mean / auprg_any_sd; dropped='none' is the reference row (shared_eval_auprg = its tab:tiers value); delta_vs_reference (difference of means) |
| tab:loco caption: paired per-split spread | tab_loco.csv | delta_paired_mean / delta_paired_sd / n_paired (fold-by-fold ablation - reference auprg_any on the same splits, population sd) |
| AB5 LOCO deltas | numbers.json | items.loco.<variant>.<cv> (deltas; paired.<category>.mean / sd / n; largest_drop empty unless every category's ablation has a delta, categories_missing) |
| tab:solo / tab:loco on the tab:tiers evaluator (optional) | tab_solo.csv / tab_loco.csv | auprg_shared_mean / auprg_shared_sd (only with --regrade, for cells whose fold matrices were kept) |
| tab:cellline (+CL1) | tab_cellline.csv | condition included / removed (cell_emb=0) / retrained without cell lines (pooled); <metric>_mean/_sd, delta_<metric>_vs_included; bold = the sign of the removed row's delta (included vs removed only); selfcheck_as_trained. A removed row whose zero_cell_emb.py run did not pass, ablated another combo or does not reproduce the included row is 'stale: ...' (no delta, not in CL1) |
| tab:cellline: what cell_emb=0 leaves | tab_cellline.csv | residual_head_terms on the removed row (per-head terms that still tell the heads apart, e.g. per-head c_h); condition 'self-scored: ...' rows (zero_cell_emb.py, same test edges and metrics, forward-pass scores; delta vs its own as-trained row), e.g. +head_proj_bias=mean / =0 |
| tab:external (+X1, X4) | tab_external.csv | spearman_mean / spearman_sd per screen x split x model; SLeuth on row_set matched (paper_convention) and all_sldb_filtered, peers on matched (paper_convention); bold = best_matched (ties at the printed 3 decimals are all bold; empty for a screen / split without a matched row of every SLeuth variant and peer); split_label (the paper's 'one gene unseen' row is split cv2, >=1 gene unseen); scored_head (SLeuth). paper_convention: every model on row_set matched, the identical rows (best_matched and X5 use it too); SLeuth's all_sldb_filtered row (pairs with a gene outside genes.txt included) is kept for reference, as the submitted paper printed it |
| tab:external, in-distribution reading | tab_external.csv | screen_role (in_distribution_check: >= 50% of the screen's scored pairs in SynLethDB, e.g. Gilbert; else external); row_set matched_incl_sldb (all matched test pairs, SynLethDB ones kept) next to matched (held out), matched_in_sldb (the SynLethDB part), every model, and model SynLethDB_label (kind sldb_label: the label itself as a predictor on matched_in_sldb); bootstrap CIs in <cmp>/bootstrap/ci_external.csv; counts in external_transfer/sldb_overlap.csv |
| SynLethDB vs screen agreement | tab_sldb_agreement.csv | auroc / spearman (+ _ci_low / _ci_high) per screen x scope (all_pairs, all_pairs_genes_txt, test_genes_txt per split) |
| X5 ranks | numbers.json | items.external.<variant>.X5_rank_on_matched_rows (ranked on the exact values; rank empty unless all three peers have a matched row; peers_compared) |
| tab:external-ft (+F1) | tab_external_ft.csv | column Zero-shot (= zero_shot_raw) / LP_last / Full, row_set all_sldb_filtered (paper_convention) and matched; best_method = the best of the three columns (the paper prints no bold here; empty with one of them absent); method zero_shot_ols (column '(reference) zero-shot OLS') is not a paper column but the OLS-calibrated head LP_last / Full are warm-started from (it differs from zero_shot_raw by the sign of a, a < 0 on Adamson), never bolded; scored_head = the head that scored the screen (eval_finetuning effective_head: the screen's cell-line head for per_head, OTHER for Corn / RPE-1, 'ANY (pooled)' for a pooled variant) |
| F2 / F3 | numbers.json | items.external.<variant>.F2_F3_all_sldb_filtered (zero_shot_raw is the paper's printed Zero-shot column and the baseline of the *_minus_zero_shot_raw deltas; *_minus_zero_shot_ols measure the same gain against the calibrated head fine-tuning starts from) |
| I1 / E1 data statistics | numbers.json | items.data_stats |
| E2 / M8 fold metadata and conflicts | numbers.json | items.folds_meta |
| D1 / D7 / D8 / D9 / D10 fold geometry | numbers.json | items.fold_geometry.<cv> (test_share, discarded_share, test_prevalence, test_both_genes_in_train[_val]); D7_analytic_gene_split |
| R5 / M12 input width (after the arm's PCA) | numbers.json | items.pca_width.per_variant.<variant> (the encoder input width after the arm's PCA; min, max over the splits; empty unless every split of every tier has a width) |
| I2 parameter ratio | numbers.json | items.param_ratio.<variant> (min_ratio empty unless every tier has a ratio) |
| M7 / M9 heads | numbers.json | items.heads |
| M3-M6 / M10 / D3 architecture and training settings | numbers.json | items.arch.<variant>.arch_signature; items.arch.<variant>.pca_arm / pca_settings (the PCA its combo applied) |
| E3 / M11 selected modalities | numbers.json | items.selection |
| R4 / D2 prose margins | numbers.json | items.margins.<variant> (best_peer / margin_vs_best_peer / leads empty unless all three peers have a row; peers_compared) |
| X2 / X3 screen coverage | numbers.json | items.screens.per_variant.<variant>.<Screen>.X2_share_both_in_genes_txt / X3_sldb_overlap_share_among_both (shares of n_pairs, the rows of <Screen>_predictions.csv: the screen's pairs whose genes both map to Entrez and both carry an embedding, duplicate pairs averaged - not of the screen file as published; each screen is taken from the first eval_finetuning_<split>/ that holds it, eval_dir names which) |
| F4 additivity | numbers.json | items.screens.per_variant.<variant>.<Screen>.F4_additive_r2 |
| D11 SynLethDB agreement | numbers.json | items.screens.per_variant.<variant>.<Screen>.D11_sldb_agreement |
| R2 SynLethKG entity / relation types | numbers.json | items.inputs.synlethkg (n_entity_types from the relation names, n_relations, removed_label_relations; data/synlethkg/meta.json, the KG SLGNN ran on) |
| D3 single training seed | numbers.json | items.inputs.training_seeds (model_params.json seed per row, via comparison.csv), single_training_seed |
| AP1 / AP2 embedding widths | numbers.json | items.inputs.native_dims.<embedding> (real columns: ComplEx's 256 complex dims are 512) |

## Run settings

- PCA arm: none (no PCA); siamese results siamese_sl/results/, comparison results sl_comparison/results/ (the shared models' rows from sl_comparison/results/); single-embedding runs siamese_sl/results/ (single-embedding runs at native dimension (no PCA), shared by arms none and joint); its combos trained with PCA within each modality no, on the concatenation no (numbers.json run.pca_settings)
- SLeuth variants: pd:pooled, pd:per_head (rows siamese_sl_pd_pooled, siamese_sl)
- SELECTION_VARIANT: pd:pooled; single-modality runs for the other variants: on (BESTCAT_EXTRA_VARIANTS)
- All-modalities combo (no modality selection; tab_tiers kind ablation): on, rows siamese_sl_allmod_pd_pooled, siamese_sl_allmod (ALLMOD_COMBO)
- RUN_TOKEN: 20261113T113339-1225758; single-modality dirs checked against 20261113T113339-1225758, ablation dirs against 20261113T113339-1225758
- Code manifest (written at submission): <repo>/sl_comparison/slurm/logs/code_20261113T113339-1225758_20261113T113540.sha256; code changed, added or removed since: none
- ARCH_ARGS (environment, else config.conf): ''; the recorded flags of each variant are items.arch.<variant>
- Embedding catalog: 19 entries (16 in the paper; extra: esmc, scprint, seqvec)
- Winners: {"expression": "geneformer/scgpt", "go": "go/go2vec", "kg": "kg_complex", "ppi": "ppi_raw/node2vec_ppi/mashup/ppi_svd", "protein_seq": "esm2/esm1b/prot_t5", "text": "text_embed"}; differing from the paper: {"expression": {"run": "geneformer/scgpt", "paper": "geneformer"}, "go": {"run": "go/go2vec", "paper": "go2vec"}, "ppi": {"run": "ppi_raw/node2vec_ppi/mashup/ppi_svd", "paper": "ppi_raw"}, "protein_seq": {"run": "esm2/esm1b/prot_t5", "paper": "prot_t5"}}
- Modality selection: PER FOLD of every CV (SELECTION_MODE=per_fold): each category's argmax of sel_auprg_any on that fold's own selection split (a random 0.1 of its training pairs, never trained on; seed 42); each CV's combo trains every fold on its own winners. Picks per tier: numbers.json items.selection.per_fold / pick_counts, tab_solo n_folds_picked
- SynLethDB filter: a screen row is SynLethDB-filtered when its pair is in neither the SL positives nor the screened negatives (eval_finetuning.py --neg_path; external_transfer.py refuses a positives-only filter). 'matched' rows also have both genes in sl_comparison/folds/genes.txt.
- Every sd is the population sd (ddof=0).

## Missing inputs and items

- none

## Absent by design

Intended by the run's settings; not part of the verdict.

- data/all_genes_esmc.pt not generated (OPTIONAL_TYPES, siamese_sl/slurm/run_generate_embeddings.conf): tab_solo esmc rows absent (not among the paper's 16)
- data/all_genes_scprint.pt not generated (OPTIONAL_TYPES, siamese_sl/slurm/run_generate_embeddings.conf): tab_solo scprint rows absent (not among the paper's 16)
- data/all_genes_seqvec.pt not generated (OPTIONAL_TYPES, siamese_sl/slurm/run_generate_embeddings.conf): tab_solo seqvec rows absent (not among the paper's 16)

## Notes

- tab_external_ft: column Scratch = the from-scratch baseline (eval_finetuning --ft_modes ... Scratch): Full's schedule on the same architecture and fold feature space, freshly initialised, its head warm-started by OLS on its own logits; items.external F2_F3 full_minus_scratch = Full - Scratch
