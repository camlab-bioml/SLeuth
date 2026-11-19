# SLeuth architecture: current vs previous

"Previous" is the published SLeuth model, the default until 2026-09-21.
"Current" is what a plain run trains now: `ARCH_ARGS` empty
(`slurm/config.conf`), in four variants (`SIAMESE_VARIANTS`), in PCA arm
`none`; since 2026-10-20 a plain run also trains PCA arms `joint`,
`per_modality` and `both` (last section).

Both are `SiameseSLMultiCell` (`siamese_esm.py`): a shared encoder applied to
each gene, then a symmetric score per cell-line head.

The previous model is reproduced by restoring every row of the table below.
First set `ENCODER_DIMS="16 16 16 16 16 16 16"` in `slurm/config.conf` (it has
no environment override: a value given on the command line is overwritten
when the file is sourced), then:

```bash
ARCH_ARGS="--layernorm --log_temperature --no_per_head_last_layer_bias --select_metric auprg --no_gene_term --no_pair_features" \
PCA_ARMS=joint \
SIAMESE_VARIANTS_LIST="pd:per_head" bash slurm/submit_pipeline.sh
```

`PCA_ARMS=joint` is the previous model's input (the PCA of the concatenation
at `PCA_JOINT_VARIANCE=0.8`, the default, on native-dimension single-modality
runs): its outputs are in `results/pca_joint/`, and, run alone, it trains its
own single-modality runs there. `POST_PCA_VARIANCE=0.8`, this recipe's knob
until 2026-10-20, is refused. As written the recipe regenerates the
embeddings, which removes the combo markers of every PCA arm (their combos
were trained on the old vectors); add `--skip-reset-env --skip-generate` to
keep the other arms' outputs.

## Differences

| Part | Previous | Current | How to restore the previous |
| --- | --- | --- | --- |
| Encoder width | 16 (7 Linear layers, latent 16) | 32 (7 Linear layers, latent 32), since 2026-10-05 | `ENCODER_DIMS="16 16 16 16 16 16 16"` in `config.conf` |
| Input to the encoder | Concatenated modalities, standardized per column, projected on the principal components reaching 80% of the variance (k = 1087-1534), renormalized | Concatenated modalities as they are; no PCA, no per-column standardization (k = about 4.4k-4.9k), since 2026-10-05 (PCA arm `none`) | `PCA_ARMS=joint` (PCA arm `joint`, `PCA_JOINT_VARIANCE=0.8`) |
| Normalisation in encoder blocks | LayerNorm in every block: `Linear → LayerNorm → LeakyReLU → Dropout` | None: `Linear → LeakyReLU → Dropout` | `--layernorm` |
| Temperature τ | `τ = exp(log τ)`, always positive | τ learned directly, sign unconstrained (initial value `1/√latent`) | `--log_temperature` |
| Projection offset | None: `z = W h` | One offset per head: `z_H = W h + c_H` (`head_proj_bias`, init 0) | `--no_per_head_last_layer_bias` |
| Kernel | PD only: `z_i·z_j + ε h_i·h_j` | PD, or bilinear `z_iᵀ A_sym z_j + ε h_i·h_j` with `A_sym = (A + Aᵀ)/2`, `A` learnable, init `I` | `--kernel pd` (the default) |
| Per-gene term | None | `g(h_i) + g(h_j)`, `g(h) = gene_w · h` on the hidden `h`, added to the logit; `gene_w` init 0 (since 2026-10-05) | `--no_gene_term` |
| Pair-feature term | None | `p(z_i, z_j) = v · LeakyReLU(U [\|z_i − z_j\| ; z_i ⊙ z_j] + u)`, added to the logit; hidden width = latent, `v` init 0 (since 2026-10-05) | `--no_pair_features` |
| Cell-line outputs | Per-head only: 8 heads, 24-d cell-line embedding concatenated to the input | Per-head, or pooled: 1 head on the "SL in any line" label, no cell-line embedding | `--cell_lines per_head` (the default) |
| Variants trained | 1 (`pd:per_head`) | 4: `pd:per_head`, `pd:pooled`, `bilinear:per_head`, `bilinear:pooled` | `SIAMESE_VARIANTS_LIST="pd:per_head"` |
| Checkpoint selection (not architecture, changed with it) | Validation stratified AUPRG, pooled over observed (pair, cell line) cells | Validation `auprg_any`: OR label, max over heads | `--select_metric auprg` |

## Unchanged

- Encoder shape: entry `Linear(D → w)`, five residual `w → w` blocks
  (`y = x + F(x)`), projection `Linear(w → w)`. All Linear layers are
  bias-free; LeakyReLU, dropout 0.2.
- Learnable per-feature input offset (one shared vector).
- Per-head scoring bias `b_H`.
- PD term `ε = 1e-3` on `h_i·h_j`.
- Cell-line embedding width 24 (per-head variants), concatenated at the input,
  so the entry layer is `Linear(k + 24 → w)` (`Linear(k → w)` for pooled).
- Objective: masked multi-label BCE, no `pos_weight`.
- Proximal L1 on the 7 encoder matrices only. Offsets, `cell_emb`, `kernel_A`,
  `gene_w`, the pair-term weights, τ and `b_H` are never penalised.
  `L1_LAMBDAS` is unchanged and not retuned for the new width.

## Score, side by side

Previous (one variant):

```
h   = Encoder([x ; e_H])               # LayerNorm in each block, width 16
z   = W h
s_H = exp(log τ) · (z_i·z_j + ε h_i·h_j) + b_H
```

Current (`pd` shown; `bilinear` replaces `z_i·z_j` with `z_iᵀ A_sym z_j`):

```
h   = Encoder([x ; e_H])               # no LayerNorm, width 32; pooled: Encoder(x)
z_H = W h + c_H
s_H = τ · (z_H,i·z_H,j + ε h_i·h_j) + g(h_i) + g(h_j) + p(z_H,i, z_H,j) + b_H
```

Both added terms are symmetric in the two genes and exactly 0 at
initialisation. The pair term does not factorise over the two genes, so the
dense all-pairs export evaluates it pair by pair and is slower.

## Parameter count

Trainable parameters. "Previous" is the mean over the five CV1 folds of the
2026-09-23 run (it still had the PCA; its four variants differ from the
published model only by the rows above that changed on 2026-09-21). "Current"
is counted from the model class at an input width of 4,864, the concatenated
width of the modality set that run selected; it moves with the selection.

| Variant | 2026-09-23 run (width 16, PCA) | Current (width 32, no PCA, both terms) |
| --- | --- | --- |
| `pd:per_head` | 23,870 | 170,049 |
| `pd:pooled` | 23,151 | 168,834 |
| `bilinear:per_head` | 24,126 | 171,073 |
| `bilinear:pooled` | 23,407 | 169,858 |

Where the growth comes from, for `pd:per_head`: the entry layer is 156,416
weights (4,888 × 32) against about 25k, the six 32 × 32 matrices are 6,144
against 1,536, and the two new terms add 2,144 (32 for `gene_w`, 2,112 for the
pair network). With width 16 and no PCA the same model has 85,521.

## 2026-10-20: PCA arms

No row above changed: PCA arm `none` is the current model's input and arm
`joint` the previous one's. What changed is that the pipeline now trains the
2x2 that the "Input to the encoder" row is one corner of, the whole pipeline
once per arm (`PCA_ARMS` in `slurm/config.conf`; default all four, any
subset; a `submit_pipeline.sh` rerun of a subset with `--skip-generate` leaves
the other arms' outputs in place, but for the table across the arms, when it
keeps pairs whole or lists second arms only: `none` without `joint`, or
`per_modality` without `both`, rebuilds the single-modality runs the left-out
partner ranked, with a WARNING, and that arm's `tab:solo` is another run's
until it is rerun):

| Arm | PCA within each modality | PCA on the concatenation | Single-modality runs | Results |
| --- | --- | --- | --- | --- |
| `none` (current) | no | no (`--no_post_pca`) | native dimension | `results/` |
| `joint` (previous, at 0.8) | no | yes (`--post_pca_variance $PCA_JOINT_VARIANCE`, or `--post_pca_dim $PCA_JOINT_DIM`) | native dimension (`none`'s runs) | `results/pca_joint/` |
| `per_modality` | yes (`--pca_variance $PCA_MODALITY_VARIANCE`) | no | PCA within the modality | `results/pca_per_modality/` |
| `both` | yes | yes | PCA within the modality (`per_modality`'s runs) | `results/pca_both/` |

Each arm runs its own modality selection, trains every variant's combo, the
eval and the leave-one-category-out and cell-line ablations, and gets its own
comparison, external-transfer and paper tables
(`../sl_comparison/results/` for `none`, `../sl_comparison/results/pca_<arm>/`
otherwise). Every table names its arm in the file, as do the comparison and
external-transfer CSVs, the collected SLeuth rows and the selection; the other
outputs (markers, `stale_siamese/`, the cell-line ablation's `*_nocell` rows,
as `zero_cell_emb.py` records no arm, and the combo and ablation directories,
whose `config.json` records their PCA) are told apart by their directory, arm
`none`'s being the unprefixed one. The single-modality runs get the
per-modality PCA only, so `joint` ranks `none`'s runs and `both`
`per_modality`'s (when both arms of the pair run), and the two arms of a pair
select the same modalities: within a pair they differ by the PCA of the
concatenation alone. Across the pairs the selections may differ, so
`per_modality` against `none` is the whole pipeline's difference, the
selection included, not the PCA's alone. The repo-root orchestrator sets the
arms side by side, each against the first (`none` when it runs), in
`../sl_comparison/results/pca_arms/tab_pca.csv` (not yet in `paper/main.tex`,
whose methods describe arm `joint`). The targets are `PCA_MODALITY_VARIANCE`
and `PCA_JOINT_VARIANCE` (0.8 each), or `PCA_JOINT_DIM` (an exact width; one
count for `joint` and `both`, kept below the narrowest concatenation their
joint PCA meets: about 1,951 columns for arm `both`'s 2026-08-11 cv1 combo at
0.8, about 1,140 in its ablation without `ppi`; `train.py` refuses a count at
or above it since 2026-10-21); `POST_PCA_VARIANCE` / `POST_PCA_DIM` are
refused (the recipe at the top uses `PCA_ARMS=joint`).

This replaces the fixed-selection PCA ablation added the day before
(2026-10-19: `slurm/run_pca_ablation.sh`, which retrained each reported combo
under the three other arms with its modalities held fixed and read them
against it, in `pcaabl_*` dirs, `results/pca_ablation_summary.csv` and a
`tab_pca.csv` among the paper tables; `--skip-pca-ablation` /
`--pca-ablation-only`, `PCA_ABLATION_*`). It was never run on the cluster, and
is removed with all of its flags, settings and outputs.

An arm changes more than the PCA. The input width, and with it the entry
layer, differs per arm (on the 2026-08-11 cv1 winner set re-fit on the
current embedding files and shared folds, cv1 fold 0 at 0.8, so `joint`
need not fall in the k = 1087-1534 above: `none` 4,864, `per_modality`
1,951, `joint` 1,570, `both` 1,291 inputs), and
`L1_LAMBDAS` is not retuned for any of them. Under `PCA_METHOD=plain` a PCA
also z-scores every column; in `both` the joint PCA z-scores the per-modality
components again, so every kept component weighs the same. Another width also
draws another random stream (initialisation, shuffles, dropout masks). The
four arms take up to 910 GB of fold matrices and exports in the repo-root
orchestrator's default run, against up to 272 GB for one (per arm 19 GB of
combos, 42 GB of pair export, 113 GB of ablation and 9 GB of cell-line
ablation matrices; per pair of arms up to 89 GB of single-modality runs, 75 GB
when the optional esmc, scprint and seqvec are absent), and queue 43 jobs and
491 tasks counting every array task, against 14 and 160 (Slurm's
`MaxSubmitJobs` limit counts each task).
