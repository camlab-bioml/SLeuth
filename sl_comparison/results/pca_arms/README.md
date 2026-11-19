# PCA arms: tab_pca and every arm's verdict

PAPER COMPLETE

Generated 2026-11-17T10:36:30 by sl_comparison/paper_tables.py --pca_arms_table from <repo>.

## PCA arms

Each arm is the whole pipeline under one PCA choice, in its own namespace; its paper tables are its own (their README.md's line 3 is its verdict, quoted here).

| Arm | PCA | Siamese results | Comparison results | Single-embedding runs | Paper tables | Verdict |
|---|---|---|---|---|---|---|
| none | no PCA | siamese_sl/results/ | sl_comparison/results/ | siamese_sl/results/ | sl_comparison/results/paper_tables/ | PAPER COMPLETE |
| joint | PCA on the concatenation of all modalities (variance 0.8) | siamese_sl/results/pca_joint/ | sl_comparison/results/pca_joint/ | siamese_sl/results/ | sl_comparison/results/pca_joint/paper_tables/ | PAPER COMPLETE |

## Tables

| Table | Status |
|---|---|
| tab_pca.csv | ok |

## Where each item is

In numbers.json `items.pca` is `{paper, value, status}`, so a key path `items.pca.<rest>` here means `items.pca.value.<rest>`.

| Item | File | Column / key |
|---|---|---|
| tab:pca (PCA arms; not yet in paper/main.tex) | tab_pca.csv | per variant x tier x arm (pca_arm; role reference = the first of --pca_arms): modality_pca / joint_pca (its combo's PCA), single_embedding, modalities / n_modalities / same_modalities_as_reference, input_dim_mean / _min / _max and params_fold_mean (its combo's folds), auprg / ap / auroc _mean / _sd (its graded comparison row, the shared evaluator, as tab:tiers), delta_<metric> (arm - reference), status, dir |
| tab:pca paired spread and reading | tab_pca.csv | delta_auprg_paired_sd (population sd) / delta_auprg_se (sample sd / sqrt(n)) / n_paired over the graded per-fold AUPRG paired by position; verdict (|delta| <= 0.005: no measurable difference; else better / worse than the reference beyond 2 SE, else within fold noise; n/a below two paired folds) |
| tab:pca best arm | tab_pca.csv / numbers.json | best_arm (exact values, every tied arm True; empty unless every requested arm's cell is ok); items.pca.<variant>.<cv>.best_arm (empty on a tie) / best_arms |
| PCA 2x2 effects | numbers.json | items.pca.<variant>.<cv>.effects.per_modality / joint / interaction (mean, sd, se = sample sd / sqrt(n), n over the folds all four arms' graded rows scored; empty unless all four arms are requested and ok); items.pca.<variant>.<cv>.arms.<arm> (means, deltas, paired, verdict, input_dim, params_fold_mean, modalities) |
| Each arm's modality selection | numbers.json | items.pca.selection (winners per arm, same_selection groups, step1_results, run_token, selection_cv) |
| PCA settings | numbers.json | items.pca.settings (this job's targets; trained: the PCA each arm's combos trained with) |
| Each arm's paper tables and verdict | README.md / numbers.json | ## PCA arms; arms[] (pca, sia_results, cmp_results, step1_results, paper_tables, verdict) |

## Run settings

- PCA arms compared: none, joint (--pca_arms / PCA_ARMS); the reference arm is none: every delta is arm minus none
- PCA targets (the environment, else siamese_sl/slurm/config.conf, else its defaults): PCA_MODALITY_VARIANCE=0.8, PCA_JOINT_VARIANCE=0.8, PCA_JOINT_DIM=(unset); what each arm's combos trained with: items.pca.settings.trained
- SLeuth variants: pd:pooled, pd:per_head (rows siamese_sl_pd_pooled, siamese_sl)
- RUN_TOKEN: 20261113T113339-1225758; each arm's selection run_token: none 20261113T113339-1225758; joint 20261113T113339-1225758 (an arm of another submission is stale)
- Selection CV (each arm's _selection_meta.selection_cv): none cv1; joint cv1 (an arm whose modalities were selected on another CV than the reference arm's is stale)
- Metrics: each arm's GRADED comparison row (the shared evaluator on the shared folds, as tab:tiers), not train.py's self-scored auprg_any; paired differences over the graded per-fold AUPRG (summary.folds of the row JSONs, by position)
- Code manifest (written at submission): <repo>/sl_comparison/slurm/logs/code_20261113T113339-1225758_20261113T113540.sha256; code changed, added or removed since: none
- Every sd is the population sd (ddof=0); se is the sample sd / sqrt(n).

## Missing inputs and items

- none

## Absent by design

Intended by the run's settings; not part of the verdict.

- numbers.json pca effects: the 2x2's per-modality / joint / interaction effects need all four PCA arms; this run compares none, joint (--pca_arms / PCA_ARMS), so they are left empty
