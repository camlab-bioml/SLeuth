# Cell-line ablation (post hoc)

How much of a trained model's ranking comes from the cell-line embedding it
learned? This takes a per_head combo's **trained checkpoints**, sets
`cell_emb.weight` to zero, and re-scores the same test edges. With a zero cell
vector the encoder no longer sees which cell line it is scoring.

Nothing is retrained. This measures how much the trained model *leans on* the
cell-line representation at inference, not what a model trained without cell
lines from scratch would achieve. Those are different questions; the pooled
variants (below) answer the second one.

## In the pipeline

`run_cell_line_ablation.sh` is one SLURM job per PCA arm (since 2026-10-20;
`slurm/config.conf`, "PCA arms"), submitted from `siamese_sl/` by the root
orchestrator (`run_benchmarking_siamese_vs_others.sh`) after that arm's combo
job, as `PCA_ARM=<arm> sbatch ... --export=ALL`; the arm's compare and paper-tables
jobs wait for it. `slurm/submit_pipeline.sh` does not submit it: the block it
prints at its end to paste (since 2026-10-21) has it as each arm's
`_l_<arm>` line, afterok on that arm's combo, and that arm's compare and
paper-tables lines wait for it. Below, `<arm root>` is the arm's siamese results root
(`results/` for arm `none`, `results/pca_<arm>/` otherwise, from
`siamese_sl/`) and `<arm cmp>` its comparison root (`sl_comparison/results/`
or `sl_comparison/results/pca_<arm>/`, from the repo root). For every per_head
variant in `SIAMESE_VARIANTS` (`pd:per_head`, `bilinear:per_head`) and every
CV it:

1. reads the combo from `<arm root>/<prefix>latest_combo_<cv>.txt` (a missing
   marker means that combo did not train: its row is reported MISSING);
2. runs `zero_cell_emb.py --save_matrices` on it, into
   `<arm root>/<prefix>cellline_ablation_<cv>/`;
3. grades the zeroed model's matrices with `grade.sh` (the shared evaluator)
   into `<arm cmp>/<row>_nocell_<cv>.json`, `<row>` = `siamese_sl`
   (pd:per_head) or `siamese_sl_bilinear_per_head`. These rows, and the
   `cellline_ablation_<cv>/` dirs, name no arm in the file: their directory
   says which arm they are (the dir's `config.json` records the source
   combo's path).

The pooled variants have no cell-line embedding, so the ablation is not
applicable to them (logged as N/A); their combos are the retrained
no-cell-line counterparts. Before anything runs the job moves every
`*_nocell_<cv>.json` it could write to `<arm cmp>/stale_siamese/` and deletes
every `<arm root>/<prefix>cellline_ablation_<cv>/`, so a failure leaves a row
missing, never a previous run's; another arm's rows and dirs are never
touched. It exits 1 when any applicable row is missing.

## Run it by hand

From `siamese_sl/` (a GPU is used when present):

```bash
python3 cell_line_ablation/zero_cell_emb.py --combo 'best_cat_<winners>_%s'    # all three CVs
python3 cell_line_ablation/zero_cell_emb.py --cv_type cv3 --combo 'best_cat_<winners>_%s'
python3 cell_line_ablation/zero_cell_emb.py --combo 'pca_joint/best_cat_<winners>_%s'   # PCA arm joint's
python3 cell_line_ablation/zero_cell_emb.py --save_matrices \
    --model_dir cv1=results/best_cat_<winners>_cv1 --model_name siamese_sl_manual
GRADE_CVS=cv1 bash cell_line_ablation/grade.sh
```

`grade.sh` runs through `bash` (here and in the pipeline job): an
SFTP-uploaded copy can lack its exec bit, and `./cell_line_ablation/grade.sh`
then fails with `Permission denied`.

`--combo` (a name under `results/`, so another PCA arm's combo is
`pca_<arm>/<name>`) defaults to the published modality set's directory name, and the
outputs default to `models/no_cellline_<cv>/` (`--out_dir`, `--cv_dir_name`),
which is also `grade.sh`'s default `PRED_DIR_TEMPLATE`. `grade.sh`'s default
row, `siamese_sl_manual_nocell`, is deliberately not a pipeline row:
`paper_tables.py` vouches for the pipeline's `<row>_nocell` rows with
`<arm root>/<prefix>cellline_ablation_<cv>/results.json`, which a hand run into
`models/` does not write. To rebuild those rows, resubmit
`run_cell_line_ablation.sh` for the arm (from `siamese_sl/`,
`PCA_ARM=<arm> sbatch --export=ALL cell_line_ablation/run_cell_line_ablation.sh`,
with `PCA_ARM=none` for arm `none`: without `PCA_ARM` in front it is the arm
the submitting shell exports, arm `none` only where that shell exports none,
and an arm's hand recipe leaves one exported; `paper_tables.py`'s rerun hints
name the arm, arm `none`'s included, since 2026-10-21). A root-orchestrator run that retrains
siamese moves every `siamese_sl*_nocell_<cv>.json` of each arm it trains, this
one included, to that arm's `<arm cmp>/stale_siamese/` when it submits.

The checkpoints and the `data/all_genes_*.pt` files they list are git-ignored
local artifacts (`*.pt`). `slurm/run_best_per_category_combo.sh` moves
`<arm root>/best_cat_*` to `<arm root>/_superseded_combos/job_<id>/` when it starts
and deletes them there once Step 2 selection succeeds, `clean_previous_runs.sh`
deletes them, and `run_generate_embeddings.sh` regenerates `data/all_genes_*.pt`.
The script stops when the combo was not trained on the current
`sl_comparison/folds` (the fold stamp `train.py` wrote into the combo's
`model_params.json` must match them), since its test edges come from the
current folds, when the checkpoints' gene order differs from the current
embedding files', and when those files are not the ones the combo was trained
on (the content hashes its checkpoints record: a file regenerated in place).
One CV that stops this way, or with any other error, keeps no matrices and
does not end the run: the CVs after it are still ablated (exit status 1).

Any per_head `SiameseSLMultiCell` checkpoint is accepted, of either kernel and
under any `train.py` options: the model is rebuilt from its weights by
`eval_finetuning.load_multicell_model` (LayerNorm or not, raw or log τ, the
per-head projection bias c_h, `kernel_A`, the `--gene_term` / `--pair_features`
terms, ...). A pooled or single-output
checkpoint is skipped as not applicable.

Each run (except `--no_save`, which leaves the output dirs untouched) first
deletes the previous outputs of every CV it processes (matrices, checkpoints,
`config.json`, `model_params.json`, `results.json`, and any `eval_finetuning_*/`
or `pair_predictions/` scored from those checkpoints), so a skipped fold or CV,
a crash or a saving run without `--save_matrices` leaves `grade.sh` nothing
stale to grade. A CV with a missing fold checkpoint (`incomplete`) or a failed
self-check keeps no matrices at all. Exit status: 0 when every requested CV was
ablated or is not applicable, else 1.

## Rows

| row | what is zeroed |
|---|---|
| `as-trained` | nothing (the self-check row) |
| `cell_emb=0` | `cell_emb.weight` — **the exported / graded 'removed' row** |
| `cell_emb=0 +bias=0` | also the per-head scoring bias b_h |
| `cell_emb=0 +head_proj_bias=mean` | also c_h's head identity (every head gets the head-average c_h); only when c_h exists |
| `cell_emb=0 +head_proj_bias=0` | c_h removed outright (also drops its gene-level term c_hᵀW h_i); only when c_h exists |

Under the published architecture (`--no_per_head_last_layer_bias`, one shared
τ, no per-head offsets) all head identity lives in `cell_emb`, so once it is
zero every head scores a pair alike: the `+bias=0` row is then **identical**
to `cell_emb=0` by construction (max-over-heads adds the single largest bias to
every pair, a constant shift, and all three metrics are rank-based). Under
`train.py`'s defaults the per-head c_h still tells the heads apart after
`cell_emb` is zeroed; the script names the remaining per-head terms and prints
the `+head_proj_bias` rows.

## results.json (one per CV)

`<out_dir>/<cv_dir_name % cv>/results.json`:

| key | content |
|---|---|
| `status` | `ok`, `incomplete` (a fold checkpoint missing), `selfcheck_failed` |
| `model_name`, `graded_row` | the source row (`siamese_sl`, ...) and `<row>_nocell` |
| `source_model_dir`, `kernel`, `cell_lines`, `cv_type` | the ablated combo |
| `rows`, `exported_row` | row labels (table above); `cell_emb=0` |
| `residual_head_terms` | per-head terms that survive `cell_emb=0` (empty = published architecture) |
| `folds`, `folds_skipped`, `n_folds` | fold ids scored / skipped |
| `per_fold[row]` | `[{fold, auroc, ap, auprg}, ...]` |
| `agg[row][metric]` | `[mean, sd]` over folds, population sd (ddof=0) |
| `paired[row][metric]` | mean paired per-fold change vs as-trained |
| `self_check` | `{ok, tol, failed_folds, per_fold: [{fold, n, pi, reference, abs_diff, ok, matrix_vs_forward_max_abs}]}` |

`ap` is average precision (the paper's AUPR), exactly as `evaluate_model.py`
scores test edges. The graded `<row>_nocell_<cv>.json` in the arm's comparison
root is the shared evaluator's summary (`auroc_mean`, `ap_mean`, `auprg_mean`, ...,
population sd) of the `cell_emb=0` matrices; the 'included' rows are the
variant's own row in `comparison.csv`.

## Self-check

The as-trained row is recomputed from the checkpoints, not copied, and must
reproduce the source combo's own test metrics, `results.json`
`fold_metrics[k]` `auroc_any` / `aupr_any` / `auprg_any` (the same test edges
and OR-collapse, scored by `train.py`), within `--selfcheck_tol` (1e-3) in
every fold. A mismatch means the gene map or the per-fold preprocessing
reconstruction is wrong: the CV is marked `selfcheck_failed`, its matrices
are deleted and the script exits 1, so no row is graded from it. The exported
matrix is also checked entry by entry against the `cell_emb=0` forward pass on
the test edges before it is written. The cross-check against `comparison.csv`
(which only exists after the compare job) is done by
`sl_comparison/paper_tables.py`.

## Published run (for reference)

Mean (population sd) over the five folds, on the shared folds' test edges, for
the published pd:per_head checkpoints (`--layernorm --log_temperature
--no_per_head_last_layer_bias`). `Δ` is the mean **paired per-fold** change
against as-trained — prevalence varies a lot across folds and shifts every row
identically, so it cancels in the difference but not in the sd column.

| Tier | row | AUROC | AP | AUPRG |
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
not that it is worth a specific number of AUPRG points. For that run the
as-trained row reproduced all nine published cells (3 tiers × 3 metrics)
exactly. A rerun under `train.py`'s current defaults is a different model, so
its numbers replace these.

## Saved models

`<out_dir>/<cv_dir_name % cv>/checkpoints/fold_<k>_best.pt` (git-ignored),
written in the same schema `train.py` uses, carrying `preprocessing_transform`,
`gene_order`, `embeddings_paths` and `input_dim` over verbatim, next to a
`config.json` (the source combo's `config.json` plus the ablation provenance,
which `eval_finetuning.py` requires). Two things differ from a `train.py`
checkpoint:

- `cell_emb.weight` is all zeros.
- `config["cell_line_ablation"]` records the variant and the source checkpoint
  (the directory's `config.json`: the source combo dir), so an ablated model
  can never be mistaken for a trained one.

`scoring_bias` and any `head_proj_bias` are left **as trained**. The
directory's `results.json` is this ablation's own (no `fold_metrics`), not a
finished-training marker; run `eval_finetuning.py --per_head --model_dir
<that dir>` directly (plus the `--embeddings_paths`, `--datasets`,
`--sl_path` and `--base_dir` arguments `slurm/eval_finetuning.sh` builds from
`eval_finetuning.conf`) to evaluate an ablated model. `predict.py` refuses
every `SiameseSLMultiCell` checkpoint, these included.

## Grading with the shared evaluator

`grade.sh` (repo-root relative paths; it cds there itself) grades
`$PRED_DIR_TEMPLATE % cv` for each CV in `$GRADE_CVS` with
`sl_comparison/evaluate_model.py` into
`$GRADE_OUT_DIR/${MODEL_NAME}_<cv>.json` (default `sl_comparison/results`, arm
`none`'s comparison root; the pipeline job passes its arm's); it refuses a
partial fold set. Once a CV's matrix dir exists, a previous
`${MODEL_NAME}_<cv>.json` is moved to `$GRADE_OUT_DIR/stale_siamese/` before
grading, so a partial set or a
failed grading leaves the row missing rather than an earlier grading; a CV
whose dir does not exist is left untouched. It exits 0 only when every CV in
`$GRADE_CVS` was graded.
`compare.py` picks the new row up automatically (it globs `*.json` and keys on
`summary["model"]`). Budget ~313 MB per fold, ~4.7 GB per variant.

**Do not** run `collect_siamese.py` to pick this up. It rediscovers the
siamese rows under the arm's siamese root only through this run's
`latest_combo_<cv>.txt` markers (`--combo_markers`, in
`sl_comparison/slurm/collect_and_regrade_siamese.sh`), and ranks on
`val_<select_metric>_mean`; the ablation's `results.json` carries no
`summary`, so even the opt-in sweep (`SIAMESE_BASELINE_DISCOVER=all`) skips it.

## What this does not ablate

Only the cell-line conditioning of the **representation**, and only post hoc.
Not ablated: the masked multi-label training objective (the model was still
trained with per-head supervision), the per-head scoring bias in the saved
checkpoints, and the eight-head output shape.

A from-scratch counterpart — the same combo trained with no cell-line
conditioning — exists in the pipeline: the pooled variants
(`--cell_lines pooled`: one head on the OR label, no cell-line embedding) are
trained next to the per_head ones on the same winners, folds and selection,
and graded as the `siamese_sl_pd_pooled` / `siamese_sl_bilinear_pooled` rows.
