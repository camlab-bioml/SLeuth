# SLURM Pipeline for Siamese SL Training

## Quick Start

```bash
cd siamese_sl

# Full pipeline: env reset → generate → (benchmark ∥ best-per-category) → combo
#                → external-screen eval → leave-one-category-out ablation
#                → all-gene-pair export (last), once per PCA arm of PCA_ARMS
#                (default: none joint per_modality both; see "PCA arms" below)
bash slurm/submit_pipeline.sh

# One PCA arm, or any subset: its own results root (results/ for none,
# results/pca_<arm>/ otherwise). The arms left out keep their outputs only with
# --skip-generate: a submission that regenerates the embeddings removes every
# arm's combo markers (their combos were trained on the old vectors); every
# best-cat submission removes ../sl_comparison/results/pca_arms/. Keep pairs
# whole or list second arms only: none (or per_modality) without its partner
# joint (both) rebuilds the single-embedding runs the partner's selection ranked
# (a WARNING says so; the partner's tab:solo is another run's until it is rerun)
PCA_ARMS=joint bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate
PCA_ARMS="none joint" bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate

# Skip generation (embeddings already exist)
bash slurm/submit_pipeline.sh --skip-generate

# Skip the chained external-screen eval + fine-tuning
bash slurm/submit_pipeline.sh --skip-eval

# Run eval only against an existing combo (no training, and no env reset unless
# --reset-env is added: the reset deletes the venv every running job uses);
# submitted as the arm whose results root holds the combo
bash slurm/submit_pipeline.sh --eval-only results/best_cat_..._cv3
bash slurm/submit_pipeline.sh --eval-only results/pca_joint/best_cat_..._cv3
# ... or a direct sbatch of the eval, which is the arm the shell exports (else arm
# none) unless PCA_ARM is passed
# (eval_finetuning.sh's header has every form). It clears and rewrites the
# combos' eval_finetuning_<cv>/, the fine-tuned models included: rerun that
# arm's pair export, external transfer and paper tables after it
PCA_ARM=joint sbatch --export=ALL --array=0-11 slurm/eval_finetuning.sh
EVAL_VARIANT=bilinear:pooled PCA_ARM=joint sbatch --export=ALL slurm/eval_finetuning.sh

# Leave-one-category-out ablation against the already-trained combos of every
# arm of PCA_ARMS (no env reset unless --reset-env is added; the ablation then
# waits for it). Export the PCA targets the combos were trained with: an arm
# whose combos' config.json records another PCA than this shell gives it is
# refused before anything is removed, with what to submit instead (since
# 2026-10-21): ONE command when one set of targets trains every arm's combos,
# else one line per arm or group of variants, to submit one after the other
# (see --ablation-only under "File Reference")
bash slurm/submit_pipeline.sh --ablation-only
PCA_ARMS=per_modality bash slurm/submit_pipeline.sh --ablation-only
# ... after a run whose combos were trained with PCA_JOINT_DIM=150
PCA_JOINT_DIM=150 bash slurm/submit_pipeline.sh --ablation-only

# Everything except the ablation sweep
bash slurm/submit_pipeline.sh --skip-ablation

# Everything except the all-gene-pair export (42 GB per arm by default)
bash slurm/submit_pipeline.sh --skip-pair-export

# The all-gene-pair export alone: this run's combos of one arm, PCA_ARM=none
# for arm none (a direct sbatch without PCA_ARM in front is the arm the
# submitting shell exports, arm none only where it exports none), or one
# explicit combo
PCA_ARM=none sbatch --export=ALL slurm/export_pair_predictions.sh
PCA_ARM=joint sbatch --export=ALL slurm/export_pair_predictions.sh
PAIR_EXPORT_MODEL_DIR=results/best_cat_..._cv3 sbatch slurm/export_pair_predictions.sh
# An explicit combo the eval never ran on (cv1 / cv2, an ablation, --skip-eval):
# probabilities only (without PAIR_EXPORT_FT_SPLITS=none the job exits 1 after writing them)
PAIR_EXPORT_FT_SPLITS=none PAIR_EXPORT_MODEL_DIR=results/best_cat_..._cv1 sbatch slurm/export_pair_predictions.sh

# Only some of the 2x2 model variants (a scalar, so it reaches every job;
# SIAMESE_VARIANTS="..." on this command line is accepted as an alias)
SIAMESE_VARIANTS_LIST="pd:per_head bilinear:pooled" bash slurm/submit_pipeline.sh
# The published SLeuth architecture + selection rule for every training call,
# in PCA arm joint (its PCA of the concatenation at PCA_JOINT_VARIANCE=0.8, the
# default; results/pca_joint/; POST_PCA_VARIANCE=0.8 is refused since 2026-10-20;
# add --skip-reset-env --skip-generate to keep the other arms' outputs).
# FIRST set ENCODER_DIMS="16 16 16 16 16 16 16" in slurm/config.conf (it has no
# environment override). Every job reads that file when it STARTS, hours or
# days after the submission: restore 32 only after the LAST job of this run
# has finished. See ../ARCHITECTURE_CHANGES.md
ARCH_ARGS="--layernorm --log_temperature --no_per_head_last_layer_bias --select_metric auprg --no_gene_term --no_pair_features" \
PCA_ARMS=joint bash slurm/submit_pipeline.sh

# Monitor jobs
squeue -u $(whoami)
tail -f slurm/logs/bench_<JOB_ID>_<TASK_ID>.out
```

## Pipeline Overview

The pipeline has six stages. Stage 1 runs once; Stage 2 and Stage 3's
Step-1 array run once per OWNER of single-embedding runs, and Stage 3's combo
and Stages 4-6 once per PCA arm of `PCA_ARMS` (since 2026-10-20; see "PCA
arms" below). Stages 2 and 3 run **in parallel** (independent output
directories); each arm's combo waits for its owner's Step-1 array. Stage 4
chains after its arm's Stage 3 on success, and Stage 5 runs concurrently with
Stage 4 (both chain off Stage 3). Stage 6 runs last, after every other job of
every arm (`afterany`). Each training stage uses **SLURM array jobs** so every
(embedding, CV) combination runs as a separate task on its own GPU.

```
Stage 1: Embedding Generation (single job, once)
    │
    ├──► Stage 2: Embedding Benchmark (array job → summary job; per owner)
    │       Each task: one embedding × one CV, with the owner's single-embedding
    │       PCA (native dim for none / joint, PCA within the modality for
    │       per_modality / both)
    │       Output: <owner root>/<type>_<cv>/results.json
    │
    └──► Stage 3: Best-per-Category (array job per owner → combo job per arm)
            Step 1 (array, per owner): one embedding × one CV with the owner's
                            single-embedding PCA (SELECTION_VARIANT; + the other
                            variants, reporting only, with BESTCAT_EXTRA_VARIANTS=1)
            Step 2 (combo, per arm): select best per category from the owner's
                            runs — ONCE, for all variants
            Step 3 (combo, per arm): train multi-modal combinations with the
                            arm's PCA, one per SIAMESE_VARIANTS × CV (4 × 3 = 12
                            by default)
                            → writes <arm root>/<prefix>latest_combo_${CV}.txt on success
            Output: <owner root>/pcavar<variance>_<type>_<cv>/results.json
                    <arm root>/best_per_category.json
                    <arm root>/<prefix>best_cat_<combo>_<cv>/results.json
               │
               ├──► Stage 4: External-screen eval + fine-tuning (chained afterok)
               │     Zero-shot (closed-form OLS) + LP_last + Full FT against
               │     Adamson, Corn, Gilbert CRISPR screens on EVERY variant's
               │     CV3 combo: |EVAL_CV_TYPES| × |SIAMESE_VARIANTS| = 12 tasks.
               │     MODEL_DIR auto-resolved from <arm root>/<prefix>latest_combo_cv3.txt.
               │     Output: ${MODEL_DIR}/eval_finetuning_<cv>/*
               │     Skip with --skip-eval.
               │
               └──► Stage 5: Leave-one-category-out ablation (chained afterok,
                     concurrent with Stage 4; array → summary)
                     Each task: one variant's combo minus ONE biological
                     category, on one CV, with the arm's PCA. 6 categories
                     × 3 CVs × 4 variants = 72 tasks, each read against ITS
                     variant's combo of the arm. The full combos are NOT
                     retrained — they are the reference rows, so no reported
                     number can move.
                     Output: <arm root>/<prefix>ablate_no_<category>_<cv>/
                             <arm root>/ablation_summary.csv (the variants
                             of the submission that wrote it)
                     Keeps each run's 5 fold matrices (313 MB each) for
                     paper_tables.py --regrade: about 113 GB over an arm's 72
                     runs by default (ABLATION_PURGE_PREDICTIONS=1 removes them).
                     Skip with --skip-ablation.

Stage 6: All-gene-pair export (one job per arm, afterany on EVERY job above; since 2026-10-07)
    For each variant's CV3 combo (<arm root>/<prefix>latest_combo_cv3.txt):
    the predicted SL probability of every gene pair, one matrix per
    cell-line head (+ ANY; ANY only for a pooled variant), and the predicted
    Adamson value of every gene pair from the models Stage 4 fine-tuned.
    No retraining; nothing reads its output, which is why it goes last.
    Output: ${MODEL_DIR}/pair_predictions/ (42 GB over the four variants by default)
    Skip with --skip-pair-export.

<arm root> is results/ for PCA arm none and results/pca_<arm>/ for the others;
<owner root> is the root of the arm that owns the arm's single-embedding runs
(results/ for none / joint, results/pca_per_modality/ for per_modality / both
by default). <prefix> is empty for the baseline pd:per_head (historical names)
and var_<kernel>_<cell_lines>__ for the other variants — see "Model variants".
```

Array sizes are **computed dynamically** from the `.conf` files. Adding or
removing embeddings from `ALL_EMBEDDINGS[]` or `EMB_CATALOG[]` automatically
adjusts the number of SLURM tasks — no script change is needed *for the array
size*. A brand-new embedding **type** needs more than a catalog entry; see
"Customization → Add a new embedding".

The root `run_benchmarking_siamese_vs_others.sh` submits Stages 1, 3, 4, 5
and 6 from these same scripts (not Stage 2), for the same arms and owners,
plus stages `submit_pipeline.sh` does not have: per arm the post-hoc
cell-line ablation (`cell_line_ablation/run_cell_line_ablation.sh`) and the
`sl_comparison` jobs (compare, external transfer, paper tables), once the
shared folds and models (queued right after Stage 1, ahead of every siamese
job, since 2026-10-21), and with two arms or more the table across the arms
(`sl_comparison/slurm/run_paper_tables.sh --pca_arms_table`). Stage 6 is its
last job too, after the paper tables. Use it for the four-model benchmark.
`submit_pipeline.sh` prints those per-arm jobs instead, as a block to paste
(below).

## File Reference

### Orchestrator

| File | Description |
|------|-------------|
| `submit_pipeline.sh` | **Login-node script.** Parses `.conf` files to compute array sizes, then submits all SLURM jobs with correct dependencies: the reset and generation once, the benchmark array + summary and the Step-1 array once per owner of single-embedding runs, the combo, eval, ablation and export once per PCA arm of `PCA_ARMS` (28 jobs and 586 tasks, counting every array task, for the default four arms; 10 and 207 for one; each arm's jobs as `PCA_ARM=<arm> sbatch ... --export=ALL`, under the usual job names; see "PCA arms" below). Supports `--skip-reset-env`, `--skip-generate`, `--after <JOB_ID>` (the env reset, unless skipped, waits for that job: it deletes the venv the job runs from; with a variant or arm subset, the other variants' and arms' `*latest_combo_cv[123].txt` are kept, so remove them by hand when JOB regenerates), `--benchmark-only`, `--best-cat-only`, `--skip-eval`, `--eval-only <combo_dir>` (no env reset unless `--reset-env` is also given; submitted as the arm whose results root holds the combo, `results/pca_<arm>/<combo>`, else arm `none`; the all-gene-pair export of that combo follows its eval, and waits behind any export a queued run of yours still has pending), `--skip-ablation`, `--ablation-only` (the ablation of the existing combos of every arm of `PCA_ARMS`, each under its own selection's token; refused before anything changes when an arm has no `best_per_category.json` or no `latest_combo` marker at all, while a missing (variant, CV) marker is named in a NOTE and its tasks SKIP; also refused, since 2026-10-21, when a marker-named combo's `config.json` records another PCA than the one this shell gives its arm, compared on `arch_signature.PCA_SIGNATURE_KEYS` (0.80 is 0.8), since every task of it would stop at its signature check after the arm's previous ablation was removed: the ERROR lists the combos and prints what to submit instead, each command complete and alone on its line (the settings taken from the calling shell, `printf %q`-quoted, then the three PCA targets in full; since 2026-10-21, as the root's commands; then `USE_CELL_LINES`, the variants and the arms as this run resolved them, since 2026-10-22: pasted into a shell exporting `USE_CELL_LINES=0`, a line ablated the single-output model, its tasks stopping at their signature check after the arm's ablation dirs were removed; every orchestrator command either orchestrator prints pins these and the PCA targets (the `--benchmark-only` remedy: no variant list, it trains the selection variant alone), and so do the tab:loco hint and the `ablation_summary` rebuild; a refused export's sbatch line pins the variants and `USE_CELL_LINES` only, `--eval-only`'s export and the venv rebuilds none. Any other setting the pasting shell exports, `SEED`, `ARCH_ARGS`, `PCA_METHOD`, `PAIR_EXPORT_*` and the like, still reaches its jobs (a printed orchestrator command names each in a NOTE when it runs; a printed sbatch command names none): paste them into a shell that exports none of the pipeline's settings): ONE command, this submission again (a one-shot `PCA_ARMS` or `ALLOW_LIVE_JOBS` of it kept) with the targets that train the combos of every arm, when one set does (always so for a forgotten or stray target); else one line per arm, or per group of an arm's variants trained at different targets (with that group's `SIAMESE_VARIANTS_LIST`), then one for the arms whose combos this shell's targets match, to submit one after the other: job names carry no arm, so a line is refused while jobs an earlier line queued are queued or running, and once `squeue` lists only those (and, with `--reset-env`, the first line's `reset_env` job has finished) `ALLOW_LIVE_JOBS=1` in front of the next line submits it alongside them, since an `--ablation-only` line removes and writes only the ablation dirs of its own arm or variants (never printed on a line: pasted as a block, such lines would pass a real conflict); `--reset-env` stays on the first line only. An arm with combos no targets train (another arm's PCA, or CVs that disagree) gets, first, a `--best-cat-only --skip-generate` line that retrains it whole, at this shell's targets, and ablates it (a retrain of some variants alone would replace the selection under the combos and the ablation of its other variants): whole is this run's variants and every other one the arm holds a combo of, set on the line as `SIAMESE_VARIANTS_LIST` (a variant subset in the calling shell retrained that subset only, until 2026-10-21) after `USE_CELL_LINES` as this run resolved it (pasted into a shell exporting `USE_CELL_LINES=0`, the line trained the single-output model in place of the arm's cell-line ones, until 2026-10-22; under `USE_CELL_LINES=0` it trains that model alone, as `pd:per_head`, and the refusal names the arm's cell-line combos it would leave under the new selection: remove their markers first, or run the `--ablation-only` at `USE_CELL_LINES=1`, which, refused, prints what to submit instead and, accepted, ablates them as they are), and the line keeps the venv (`--skip-reset-env`, unless `--reset-env` was given: the other arms were trained in it), as does the line that trains an arm with no selection or no marker. Every such retrain line keeps a pair whole (since 2026-10-24, `config.conf` `pca_pair_closed`): an arm to retrain whose pair partner is in `PCA_ARMS` retrains with it (and so is not on the "ablate the others alone" line), since retrained alone `joint` (or `both`) became its own Step-1 owner, and `none` (or `per_modality`) rebuilt the runs its partner's kept selection ranked. An arm split into groups gets, after the lines, the `sbatch ... slurm/run_ablation_summary.sh` command (from `siamese_sl/`, under its selection's `RUN_TOKEN`) that rewrites its `ablation_summary.csv` over all its variants once every line's jobs have finished, since each line's summary job writes the file over that line's variants only (no table reads it; with no token recorded, no such command). So export the targets, and the `ARCH_ARGS`, the combos were trained with; no env reset unless `--reset-env` is also given, in either order; refused with `--after`) and `--skip-pair-export` (the one-day PCA-ablation stage's `--skip-pca-ablation` / `--pca-ablation-only` are unknown options since 2026-10-20). Stage order is generate → benchmark + Step 1 (per owner) → combo (per arm) → {external screens ∥ ablation} → all-gene-pair export; each arm's eval (12 tasks) and 72-task ablation array (both scale with `SIAMESE_VARIANTS`) chain off its combo and run concurrently, and the exports (one per arm, only when the run trains combos or with `--eval-only`) wait `afterany` for every job queued before them, of every arm, and, `singleton`, for any other `pair_export` job of yours (two exports of one combo would clear each other's files). A failure to submit an export is a WARNING, not an error: nothing depends on it, and the WARNING prints the `sbatch` line that keeps it last, the settings taken from the calling shell in front (since 2026-10-21: a one-shot `PAIR_EXPORT_UNIVERSE=shared PAIR_EXPORT_DTYPE=float16` or variant subset never reached the shell it is pasted into). A mistyped `PAIR_EXPORT_*` value is refused at submission, before anything is removed or queued (`check_pair_export` in `export_pair_predictions.conf`; the names in `PAIR_EXPORT_FT_DATASETS` / `PAIR_EXPORT_FT_MODES` are not checked there), since the job that reads it is the last one; so are a `PCA_ARMS` or PCA target `config.conf` refuses (an unknown or repeated arm; a `PCA_ARMS` set but empty or blank, which is never read as all four) and `POST_PCA_VARIANCE` / `POST_PCA_DIM` (`PCA_ARMS=joint` is that pipeline now), and an inherited `PCA_ARM` is ignored with a NOTE. Every setting it takes from the calling shell (the root orchestrator's list of names, plus `SHARED_FOLDS`) is named in a NOTE under the node check, and recorded with its value before `config.conf` for the commands it prints to run later (since 2026-10-21). `MAX_CONCURRENT` caps each benchmark, best-per-category and ablation array on its own and `EVAL_CONCURRENT` each eval array (every arm's ablation array runs beside its eval, so with four arms that tail stage runs up to 4 × (`MAX_CONCURRENT` + `EVAL_CONCURRENT`) tasks; it checks only these two: it has no shared array): empty (the default), `0`, `none`, `all` or `unlimited` means no cap (no `%N`, so SLURM runs as many tasks as the partition allows), a positive integer `N` (no leading zero) adds `%N` to `--array`, and anything else is refused before anything changes, as is `GPU_EXCLUDE` without `GPU_NODES` or sharing a node with it, or a node selection `sbatch --test-only` rejects (see "Change SLURM partition/node"). The banner lists the arms with their roots and PCA, the owners, the task counts per arm and per owner, the jobs and array tasks the submission queues (Slurm's `MaxSubmitJobs` counts every array task, so check `sacctmgr show assoc user=$USER format=MaxSubmit,GrpSubmit` and the QOS's `MaxSubmitPU` before a four-arm run) and the disk of the fold matrices and pair export it keeps (up to 1,062 GB by default: per arm 19 GB of combos, 42 GB of pair export and 113 GB of ablation fold matrices, per owner up to 94 GB of benchmark and 89 GB of Step-1 fold matrices, as a task whose embedding is absent keeps none; up to 357 GB for one arm), with what comes on top on a `not counted:` line under it (the eval's per-pair CSVs, about 5 GB per arm, when the eval runs, and the checkpoints; since 2026-10-21). Every `afterok` edge carries `--kill-on-invalid-dep=yes`, so a failed parent cancels its children (whose `afterany` summaries then run and report) instead of leaving them pending forever. Before any `sbatch` it refuses to start while jobs of an earlier submission that it would break are queued or running (`ALLOW_LIVE_JOBS=1` downgrades the refusal to a WARNING; job names carry no arm, so another arm's jobs count too): any pipeline job, other than `--after`'s, for the stage sets that reset the venv (every job runs from it) or regenerate the embeddings; for a best-per-category submission that does neither (`--skip-reset-env --skip-generate`, `--skip-reset-env --after <JOB>`), the jobs that read or write what it removes at submission, listed below (`bestcat_emb`, `bestcat_combo`, `eval_finetuning`, `ablation`, `ablation_summary`, `cellline_ablation`, `pair_export`, `sl_final_compare`, `sl_ext_transfer`, `sl_paper_tables`, `--after`'s job included; the manual DAG's `reset_env`, `siamese_gen_emb`, `sl_folds` and `sl_compare` jobs do not block it); for `--ablation-only`, a live `ablation`, `ablation_summary`, `bestcat_emb` or `bestcat_combo` job (it retrains the ablation dirs under the ablated runs' own tokens). `--eval-only` without `--reset-env` is never checked. `--benchmark-only` is refused outright when generation would run while an earlier run's `*latest_combo_cv[123].txt` or `best_per_category.json` exist in any arm's results root (all four, whatever `PCA_ARMS` says): it retrains no combo, and regenerating replaces the (not bit-for-bit reproducible) embeddings those combos were trained on — pass `--skip-generate` (the ERROR prints this submission again with it, the settings taken from the calling shell and the PCA targets in front). `--skip-generate` runs `embedding_versions.py` and WARNs when it cannot confirm the files as current. It then removes what this submission rebuilds, for every arm it trains and in that arm's own roots: with best-per-category, the replaced combos' cell-line ablation outputs (`<arm root>/<prefix>cellline_ablation_<cv>/`, `<arm cmp>/<row>_nocell_<cv>.json`, moved to that arm's `stale_siamese/`), since it does not rerun that stage (the block it ends with has that job: the `_l_<arm>` line), the variants' `<arm root>/<prefix>latest_combo_<cv>.txt` markers and `<arm root>/best_per_category.json` (the combo deletes them only once it starts), their `sl_comparison` rows and `*_selected_<cv>.txt` markers (moved to the arm's `stale_siamese/`) and the `sl_comparison` outputs built from them (`comparison.csv`, `comparison_notes.txt`, `external_transfer/`, `paper_tables/`, and the table across the arms, `sl_comparison/results/pca_arms/` — it queues no `sl_comparison` job: the block it ends with rebuilds them, `external_transfer/` only with the eval and the table across the arms only with two arms or more, as step 0a says), **and** the variants' `<arm root>/<prefix>ablate_no_*` dirs and `<arm root>/ablation_summary.csv` (also with `--skip-ablation`: the retrained combos are their reference row, so `tab:loco` reports them missing rather than pairing last run's ablations with this run's combos). Whenever it trains combos (on that state, not on what it removed, so a resubmission prints it too), it ends with a ready-to-paste block wired to this submission's job ids (since 2026-10-21; it replaces an unwired hint whose jobs, pasted as told, started before their inputs existed). The block's first line defines `_paste_env`, which runs every job with this run's `PCA_ARMS`, PCA targets, `SIAMESE_VARIANTS_LIST` and `USE_CELL_LINES`, and the other settings those jobs read as this run resolved them (since 2026-10-21; an empty one is still set, `NAME=`, so the pasting shell's never wins): `SELECTION_VARIANT`, `BEST_CAT_PCA_VARIANCE` (the Step-1 dir tag), `BESTCAT_EXTRA_VARIANTS`, `ARCH_ARGS` and `CVS` (the compare's CVs, which reach no job of this script: one that is not a subset of `cv1 cv2 cv3`, each once, is refused before anything changes, as the root refuses it, and so it is with `--ablation-only`, whose refusals print `--best-cat-only` lines that carry it); and without the pasting shell's `RUN_TOKEN`, `CODE_MANIFEST` or compare pins (`SIAMESE_RESULTS_*`, `SIAMESE_BASELINE_DISCOVER`); it stays defined in that shell. Then, per arm: `_l_<arm>` (the cell-line ablation, afterok on the arm's combo, kill on an invalid dependency; leave the line out to skip `tab:cellline`; the NOTE printed with each arm's combo points at this line and prints no command of its own, so the job is never submitted twice), `_c_<arm>` (compare, afterany on the combo and the `_l_<arm>` job when pasted), `_e_<arm>` (external transfer, afterany on the arm's eval; no such line with `--skip-eval`) and `_t_<arm>` (paper tables, afterany on those, the eval and the ablation summary), each with the arm's `PCA_ARM=<arm>` and tagged logs; with two arms or more, last, the table across the arms after every `_t_<arm>`, submitted as the first arm. With regeneration it also removes the combo markers of every variant in all four arm roots: those combos were trained on the embeddings it replaces. `--ablation-only` removes the same ablation outputs of its arms, and (since 2026-10-24) each one's `<arm cmp>/paper_tables/` and `sl_comparison/results/pca_arms/`, which were built on them (a PAPER COMPLETE `tab:loco` stayed published over removed ablation dirs, also when the new ablation failed); it ends with the ready-to-paste rebuild, `_paste_env` and one `_t_<arm>` line per arm after its ablation summary (with `--ablation_run_token <its token>` when the arm's selection records none), then, with two arms or more, the table across the arms. `<arm root>` is `results/` or `results/pca_<arm>/`, `<arm cmp>` `../sl_comparison/results/` or `../sl_comparison/results/pca_<arm>/`; an arm `PCA_ARMS` leaves out keeps everything else, so a one-arm rerun is `PCA_ARMS=<arm> bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate`; but a subset that lists `none` (or `per_modality`) without `joint` (or `both`) retrains, as the pair's owner, the single-embedding runs the left-out partner's selection ranked, and says so in a WARNING before anything changes (that arm's `tab:solo` is another run's until it is rerun). |

### Shared Configuration

| File | Description |
|------|-------------|
| `config.conf` | **Sourced by every script.** Sets Python path, project directories (`WORK_DIR`, `DATA_DIR`, etc.), key data file paths, shared training hyperparameters (architecture, optimizer, regularization), the PCA arms (below), CV types, the full embedding catalog (`ALL_EMBEDDINGS[]`), and compute-node environment (module loads, GPU info, and a `PCA arm: <arm> (<what it trains>) \| results: <arm root>/` banner line; the embedding generation, one job shared by every arm, sets `PCA_EVERY_ARM=1` and prints `PCA arms: <arms> (a job shared by every arm: no arm's results root)` instead, since 2026-10-21). Works from both login nodes and compute nodes — SLURM-specific setup is guarded by `$SLURM_JOB_ID`. |

#### Negatives & cell-line mode (in `config.conf`)

- `NEG_PATH` — the experimentally screened non-SL pairs (default `$NEG_PATH_FILE`). **Required** in cell-line mode (`USE_CELL_LINES=1`: they are the per-head negatives, and `""` makes every train.py call raise); **ignored** with `USE_CELL_LINES=0`, whose labels come from the shared folds. There is no random-negative mode: every run needs `SHARED_FOLDS`. `POS_NEG_RATIO` has no effect either — the shared folds fix the class balance (22,166 positives vs 106,704 negatives, about 1:4.8).
- `USE_CELL_LINES` (`0`/`1`, **default `1`**) + `CELL_LINE_DIM` (default `24`; train.py's own `--cell_line_dim` default is 8; `pooled` variants ignore it) — **masked multi-label cell-line conditioning**, on by default. Each gene pair gets a per-head (label, mask) over `[K562, JURKAT, A549, HELA, A375, 293T, PC9, OTHER]`; the model (`SiameseSLMultiCell`) predicts SL per head (early stopping / checkpoint selection on the validation split: `auprg_any` — AUPRG of the OR label, max over heads — by default, or the stratified `auprg` pooled over (pair, head) cells with `--select_metric auprg`, the published rule). Requires `NEG_PATH` and the `.sources.tsv` sidecars (produced by Stage 1). These resolve to `$CELL_LINE_ARGS`, threaded into all three training run-scripts. **The Stage 4 external-screen eval consumes the multi-head checkpoint PER cell-line head** — `PER_HEAD` is auto-enabled in `eval_finetuning.conf` when `USE_CELL_LINES=1`, scoring each screen against its cell-line head (Adamson/Gilbert → K562, Corn → OTHER); an exported `PER_HEAD` overrides it for a direct sbatch or `--eval-only`, and both orchestrators ignore one, with a NOTE, for the evals they chain (`submit_pipeline.sh` since 2026-10-22: it stopped every eval task at its first checkpoint). `USE_CELL_LINES=0` trains the single-output `SiameseSL` instead, on the shared folds' own pos/neg labels (`NEG_PATH` is then not used); `SIAMESE_VARIANTS` collapses to `pd:per_head`, since `--kernel`/`--cell_lines` act only on the cell-line model. Both modes need `SHARED_FOLDS`: train.py early-stops and selects only on a validation split, which only the shared folds carry.

#### Model variants (in `config.conf`)

- `SIAMESE_VARIANTS` — bash array of `KERNEL:CELL_LINES` entries, default the 2x2 grid `pd:per_head pd:pooled bilinear:per_head bilinear:pooled`. **Override it through the scalar `SIAMESE_VARIANTS_LIST`** (space-separated): bash cannot export arrays, so `config.conf` builds the array from `SIAMESE_VARIANTS_LIST` and re-exports the validated list, which is how the orchestrator and every `sbatch --export=ALL` job see the same variants (the array sizes and the task → variant mapping depend on it). `SIAMESE_VARIANTS="..."` on the login-node command line is read once as an alias; `SIAMESE_VARIANTS_LIST` wins when both are set. Each entry maps to train.py `--kernel KERNEL --cell_lines CELL_LINES` (`siamese_variant_args`). The combo, the external-screen eval and the leave-one-category-out ablation run once **per variant**, in every PCA arm (the root orchestrator's cell-line ablation once per `per_head` variant). Requires `USE_CELL_LINES=1` (otherwise the grid collapses to `pd:per_head`).
- **train.py defaults changed on 2026-09-21** and `ARCH_ARGS` is empty by default, so a default run is NOT the published model: no LayerNorm in the encoder blocks (`--layernorm` restores it), τ learned directly and free in sign (`--log_temperature` restores τ = exp(log τ) > 0), a per-head projection bias c_h (`--no_per_head_last_layer_bias` removes it), and selection on `auprg_any` (`--select_metric auprg` restores the stratified rule). Since 2026-10-05 the reported model applies no PCA either (PCA arm `none`; the former post-concat PCA is PCA arm `joint`, which a default run trains next to it, and `PCA_ARMS=joint` alone), the encoder is width 32 (`ENCODER_DIMS`, was 16) and the two extra score terms are on.
- **Modality selection runs once** per PCA arm, on `SELECTION_VARIANT` (default `pd:per_head`): the single-embedding runs (Stage 2, Stage 3 Step 1) train that variant only, and every variant's combo of the arm concatenates the same category winners. Selecting separately per variant would give each variant a different modality set and confound the 2x2 with feature selection.
- **Naming.** The baseline `pd:per_head` keeps the historical names (`results/best_cat_<names>_<cv>`, `results/ablate_no_<cat>_<cv>`, `results/latest_combo_<cv>.txt`). Every other variant is prefixed `var_<kernel>_<cell_lines>__` (`siamese_variant_prefix`), e.g. `results/var_bilinear_pooled__best_cat_<names>_<cv>`, which no existing `best_cat_*` / `ablate_no_*` / `latest_combo_*` glob can match. Every PCA arm but `none` repeats this layout under `results/pca_<arm>/` (see "PCA arms"), which none of those globs matches either (none of them recurses). `sl_comparison` gets one row per variant in each arm's comparison root: `siamese_sl` (baseline, pin `SIAMESE_RESULTS_<cv>`) and `siamese_sl_<kernel>_<cell_lines>` (pin `SIAMESE_RESULTS_<kernel>_<cell_lines>_<cv>`).
- `ARCH_ARGS` — extra train.py flags appended to every siamese training call and echoed in each log. The published model is `--layernorm --log_temperature --no_per_head_last_layer_bias --select_metric auprg --no_gene_term --no_pair_features` (with `MODEL_ARGS`' `--no-last-layer-bias`), at `ENCODER_DIMS` width 16 (edit `config.conf`: no environment override) in PCA arm `joint` (`PCA_ARMS=joint`, `PCA_JOINT_VARIANCE=0.8`, the default); see `../ARCHITECTURE_CHANGES.md`. `--kernel` / `--cell_lines` are refused there, and so are the PCA flags (`--pca_variance`, `--pca_dims`, `--post_pca_variance`, `--post_pca_dim`, `--no_post_pca`, an abbreviation or an `=value` form), wherever `config.conf` is sourced, at submission included, because one would come last on every command line: it would silently override the PCA of the job's arm (the logs echo only `ARCH_ARGS`), so every arm would train the same PCA under its own arm's name. The PCA is the arm's (`PCA_ARMS` / `PCA_ARM` and the `PCA_*` targets below). `--pca_method` (the preprocessing, not an arm) stays allowed. Two extra score terms are on by train.py's default since 2026-10-05 (a per-gene propensity and a symmetric pair-feature MLP added to each head's logit; cell-line model only; the pair term makes the per-fold all-pairs export slower); `ARCH_ARGS="--no_gene_term --no_pair_features"` drops them.
- **Category winners and the `sl_comparison` siamese rows are ranked on the metric each run's own checkpoint selection used**: `val_<select_metric>_mean`, `select_metric` read from the run's config (`auprg` for runs predating the flag), falling back to `val_auprg_mean`. The combo stops if the `SELECTION_CV` candidates were selected on different metrics.
- `sl_comparison` rows (`sl_comparison/slurm/collect_and_regrade_siamese.sh`, per arm): each row is THIS run's combo only, the directory named by the arm's `<arm root>/<prefix>latest_combo_<cv>.txt` (`collect_siamese.py --combo_markers`; a CV with no marker, or whose marker names a missing or foreign directory, gets no row, never an older `best_cat_*` directory), and a candidate whose recorded `kernel` / `cell_lines` differs from the row is dropped, so the 2x2 compares like with like. `SIAMESE_BASELINE_DISCOVER=all` restores the old sweep-wide baseline rglob of the arm's root (no freshness check; not comparable with the variant rows; it skips the other arms' roots `pca_<arm>/` and `pca_arms/` by name); `SIAMESE_RESULTS_<cv>` / `SIAMESE_RESULTS_<tag>_<cv>` pins still win. Before each collection the arm's existing 2x2 rows (`siamese_sl_cv[123].json`, `siamese_sl_<tag>_cv[123].json`) and their `*_selected_cv[123].txt` markers are MOVED, not deleted, into the arm's `sl_comparison/results[/pca_<arm>]/stale_siamese/`, as is any row whose shared re-grade is missing or fails (as `<row>.unregraded.json`). Other `siamese_sl_*` rows (e.g. `siamese_sl_nocell_*`) are left in place.
- Compare variants on `auprg_any` (the OR label), not on the stratified `auprg`: for `per_head` that pools (pair, head) cells, for `pooled` it pools pairs. The combo summary prints both.
- The combo job (one per PCA arm) trains all variants sequentially: 12 combos by default, about 4× the old wall time (the last 3-combo run took ~6 h; `--time` is 5 days); the arms' combo jobs run side by side. It exits non-zero (cancelling its arm's `afterok` eval, ablation and cell-line ablation) only when Step 2 selection fails or nothing trained at all (every combo failed, or all were skipped for having < 2 modalities). Any single failed combo, the `SELECTION_VARIANT` / baseline `pd:per_head` (`siamese_sl` row) ones included, is logged loudly and the job exits 0, so one failed cell does not cancel every variant's downstream tables; each failed (variant, CV) has no marker, so sl_comparison writes no row for it, that CV's ablation tasks skip (`run_ablation.sh` requires the marker to name the reference combo), the cell-line ablation names it MISSING, and the variant's eval (cv3 combo only) stops only if its cv3 combo failed; the paper tables (the arm's and the table across the arms) list every absent row.
- **Step 2 stops on a stale or incomplete `SELECTION_CV` candidate set.** A candidate with this run's token but another variant, architecture / selection signature or fold stamp always stops the job. A missing score (failed Step-1 task, or embedding present but no readable result), a candidate carrying another submission's `run_token.txt` (or none; whether or not its embedding exists), or a category with no candidate at all stops it unless `BEST_CAT_ALLOW_MISSING=1`, which selects on what is there instead. A never-generated embedding with no result only warns; the unranked entries are listed in `best_per_category.json` `_selection_meta` (`unranked_failed`, `unranked_foreign_token`, `unranked_not_generated`, `missing_categories`).
- **`RUN_TOKEN`** (`config.conf`, exported): one id per submission, set on the login node by each orchestrator before the first `sbatch` and carried to every job by `--export=ALL`. It is made fresh (an inherited `RUN_TOKEN` is replaced, with a NOTE) with two exceptions, both resolved per PCA arm: `submit_pipeline.sh --ablation-only` takes, for each arm it ablates, the token of the run whose combos it ablates (`<arm root>/best_per_category.json` `_selection_meta.run_token`, so `paper_tables.py` accepts that run's Step-1 dirs and the new ablations together; the arm's old ablation dirs are removed at submission), and the root script's `--skip-siamese` takes each arm's reused selection's token (`<arm root>/best_per_category.json` `_selection_meta.run_token`); only when that file records none does it read the `pcavar<PCA_VARIANCE>_*/run_token.txt` files of the arm's owner root that `paper_tables.py` reads, and only when those disagree does it keep an exported `RUN_TOKEN` that names one of them (`RUN_TOKEN=<token> bash run_benchmarking_siamese_vs_others.sh --skip-siamese`). When the arms' tokens differ, each arm's jobs get their own in sbatch's environment (`PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ... --export=ALL`, with a WARNING and the tokens in the banner), and the exported `RUN_TOKEN` (the shared jobs', the manifests') is the first arm's. Only a direct `sbatch` falls back to `config.conf`, which keeps an exported `RUN_TOKEN` or makes a new one, per job and per array task. `run_embedding_benchmark.sh`, `run_best_per_category.sh` and `run_ablation.sh` write it to `<output_dir>/run_token.txt` after a successful train. The combo's Step 2 ranks only `pcavar*` candidates carrying this run's token, `run_ablation_summary.sh` reports an ablation run with another token as a MISMATCH with no delta, and `run_embedding_benchmark_summary.sh` shows such a cell as STALE: a task that never started (cancelled while pending, dependency never satisfied) keeps its previous run's directory, which can agree with this run on variant, signature and fold stamp. So export one token before submitting stages by hand (see "Change SLURM partition/node"), or, to have a direct `sbatch` read an earlier submission's output on purpose, export that token (`cat results/pcavar0.8_<type>_cv1/run_token.txt`; `results/pca_per_modality/...` for the arms `per_modality` / `both` by default).

#### PCA arms (in `config.conf`, since 2026-10-20)

- `PCA_ARMS` — the arms a submission trains, a space-separated scalar (exported; default `none joint per_modality both`, any subset, normalised to that order; an unknown or repeated arm is refused before anything changes, and so are a value that is set but empty, `PCA_ARMS=` or an empty `"$ARMS"` (`ERROR: PCA_ARMS is set but empty; list at least one of none joint per_modality both, or unset it for all four`), and a whitespace-only one (`ERROR: PCA_ARMS is empty; ...`): neither is read as all four, only an unset `PCA_ARMS` is): the 2x2 of PCA within each modality (no / yes) × PCA on the concatenation of all modalities (no / yes). Each arm is the whole pipeline (Stages 3-6, Stage 2 and Stage 3's Step 1 per owner, below; in the root orchestrator also the cell-line ablation, compare, external transfer and paper tables) under its own results root:

  | Arm | Multi-modal trainings (`pca_multi_args`: combo, ablation) | Single-embedding runs (`pca_single_args`: Stage 2, Step 1) | Root |
  | --- | --- | --- | --- |
  | `none` (no PCA; the reported model since 2026-10-05) | `--no_post_pca` | `--no_post_pca` | `results/` |
  | `joint` (PCA on the concatenation; the published model's) | `--post_pca_variance $PCA_JOINT_VARIANCE`, or `--post_pca_dim $PCA_JOINT_DIM` | `--no_post_pca` | `results/pca_joint/` |
  | `per_modality` (PCA within each modality) | `--pca_variance $PCA_MODALITY_VARIANCE --no_post_pca` | `--pca_variance $PCA_MODALITY_VARIANCE --no_post_pca` | `results/pca_per_modality/` |
  | `both` (`per_modality`, then `joint`) | `--pca_variance $PCA_MODALITY_VARIANCE` + the joint flag | `--pca_variance $PCA_MODALITY_VARIANCE --no_post_pca` | `results/pca_both/` |

  Every file name inside an arm's root is the one this README gives for `results/`; the arm's `sl_comparison` outputs are in `../sl_comparison/results/` (arm `none`) or `../sl_comparison/results/pca_<arm>/`, next to the shared models' rows, which stay in `../sl_comparison/results/` for every arm.
- `PCA_MODALITY_VARIANCE` (0.8), `PCA_JOINT_VARIANCE` (0.8), `PCA_JOINT_DIM` (empty; an exact count, which wins over `PCA_JOINT_VARIANCE`) — the PCA targets, exported and validated (a fraction `0.<digits>`; a positive integer without a leading zero), the same for every arm of a submission. `PCA_JOINT_DIM` is ONE count for both arms with the joint PCA, `joint` AND `both`, which meet very different widths: keep it below the narrowest concatenation the joint PCA acts on in any run of either arm, the leave-one-category-out ablations included. In arm `both` that is the kept modalities' per-modality components (about 1,951 for the 2026-08-11 cv1 combo at 0.8, lower in every ablation: about 1,140 without `ppi`, whose `ppi_raw` keeps about 810; it moves per fold and per CV), in arm `joint` the native width without the widest category (about 3.1k or more). A count at or above that width would run no joint PCA under a label that says it ran, so `train.py` refuses it (since 2026-10-21; a `ValueError` from `data_loader`'s fit naming both numbers) at the fit of the first fold that narrow, in each combo or ablation run separately, after the Step-1 array: the orchestrators' `--print_arch` check loads no data and cannot see a width. In arm `joint` that is fold 0 (its native width is the same in every fold); in arm `both`, where the width moves per fold and CV, the run's earlier folds, and in the combo job whole earlier combos, may have trained by then (the refused run writes no marker; the combo job goes on with its other combos). A later `--ablation-only` (or the root's `--skip-siamese`) of those arms needs the same targets exported: both compare them with the combos' `config.json` first. `POST_PCA_VARIANCE` / `POST_PCA_DIM`, the post-concat PCA switch until 2026-10-20, are refused, not reinterpreted (an old recipe would otherwise train all four arms): `PCA_ARMS=joint` is that pipeline now.
- **Single-embedding runs are shared by pairs of arms.** The joint PCA acts on a concatenation (on one modality it would repeat the per-modality PCA, or, after it, cut an already flat spectrum at an arbitrary place), so `none` and `joint` rank the same native-dimension runs and `per_modality` and `both` the same per-modality-PCA runs. Each pair's benchmark and Step-1 arrays run once, as its OWNER (`pca_step1_arm`: the pair's first arm that `PCA_ARMS` lists), into the owner's root (`STEP1_ROOT`: `results/` and `results/pca_per_modality/` by default; `PCA_ARMS=joint` alone trains its own, in `results/pca_joint/`), and each arm's combo waits for its owner's Step-1 array. Step 2 builds its expected signature with the arm's single-embedding flags, so a candidate trained with another PCA stops the job, and `best_per_category.json` `_selection_meta` records `pca_arm`, `pca`, `pca_args`, `step1_root`, `step1_pca` and `step1_pca_args`. The two arms of a pair therefore select the same winners and differ only by the PCA of the concatenation.
- `PCA_ARM` — the arm of ONE job (exported, validated; default `none`). Both orchestrators submit each arm's jobs as `PCA_ARM=<arm> sbatch ... --export=ALL` (the arm in sbatch's own environment: an `--export` list such as `--export=ALL,PCA_ARM=<arm>` makes Slurm imply `--get-user-env`, a login-environment fetch at every task launch that holds the job when it fails; 2026-10-21) under the usual job names (so every live-job check sees every arm's jobs, and `singleton` runs the arms' exports one after another) and drop one inherited from their own shell, with a NOTE; arm X's jobs log to `slurm/logs/<stage>_pca_X_<job id>.out`. A job submitted by hand is arm `none` unless `PCA_ARM` is exported or passed (`PCA_ARM=<arm> sbatch --export=ALL ...`); export the run's `PCA_ARMS` (it decides the owner) and targets with it. From the job's own `PCA_ARM`, `config.conf` derives, not exported, `RESULTS_ROOT` (`<arm root>`), `STEP1_ROOT` (`<owner root>`), `CMP_ROOT` (`sl_comparison/results[/pca_<arm>]`, from the repo root), `PCA_MULTI_ARGS` and `PCA_SINGLE_ARGS`.
- **Subsets.** An arm `PCA_ARMS` leaves out keeps its siamese outputs and SLeuth rows, with four exceptions. (1) A submission that regenerates the embeddings (`submit_pipeline.sh` by default; the root orchestrator with `--force-generate` or when its auto-skip check fails) removes the combo markers of every arm in all four roots, since their combos were trained on the vectors it replaces: a one-arm rerun is `PCA_ARMS=<arm> bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate`. (2) Every best-per-category submission and every root run removes the table across the arms, `../sl_comparison/results/pca_arms/`. (3) A root run that retrains the shared models (no `--skip-shared`) removes the left-out arms' outputs built on the shared rows it replaces, their `comparison.csv` + `comparison_notes.txt`, `external_transfer/` and `paper_tables/` (their SLeuth rows and siamese outputs stay). Every such run names every left-out arm without tables, whichever submission removed them (so also after a submission that stopped part-way): the step-0a NOTE names the arms, and the final listing prints ready-to-paste jobs wired with `--dependency` to this run's, each left-out arm's compare and external transfer after this run's shared array, its paper tables after both, then the table across every arm after those and after this run's own table across the arms (with one arm, after that arm's paper tables); every pasted job goes through `_paste_env`, defined on the block's first line, which keeps the pasting shell's `RUN_TOKEN`, `CODE_MANIFEST` and compare pins out and gives the job, as `PCA_ARMS`, the arms of the table across the arms (this run's and the left-out arms the block rebuilds; not an arm given the retraining command), this run's PCA targets, a left-out arm its own where its combos were trained at others, and this run's variants (`SIAMESE_VARIANTS_LIST`, `USE_CELL_LINES`: those whose combos the block's PCA check compared; a left-out arm's compare then rebuilds their rows only) (since 2026-10-21). A left-out arm with a selection but no combo marker (every one, after a regeneration) is given `PCA_ARMS="<arms>" bash run_benchmarking_siamese_vs_others.sh --skip-generate --skip-shared` instead, for once this run has finished, printed with the settings this run took from the calling shell in front, values included (`SEED`, `ARCH_ARGS`, the PCA targets, ...; tables rebuilt now would have no SLeuth rows; that retraining run replaces this run's table across the arms with one over the arms it retrains, and a table across every arm, with its 2x2 effects, needs one submission's selection of every arm: `bash run_benchmarking_siamese_vs_others.sh --skip-generate --skip-shared` with `PCA_ARMS` unset); also once it has finished, `--skip-siamese` with every arm in `PCA_ARMS` rebuilds all of it in one submission, retraining the shared models once more. A table across the arms lists as stale (no values) an arm whose selection another submission made than its first arm's. (4) A subset that lists the first arm of a pair (`none` or `per_modality`) without its partner (`joint` or `both`) makes it the pair's owner, whose Step 1 rebuilds the single-embedding runs the partner's kept selection ranked (`_selection_meta.step1_root`): both orchestrators print a WARNING before anything changes, and the partner's `tab:solo` belongs to another run until the partner is rerun with `PCA_ARMS` listing it. A subset of second arms only (`joint`, `both`) trains its own runs in its own root. So a root subset run with `--skip-generate --skip-shared` (its `--help` recipe) that keeps pairs whole (`none joint`, `per_modality both`) or lists second arms only meets only the second exception.
- **Cost.** Each arm adds 12 combos, 12 eval and 72 ablation tasks and an export, each owner a benchmark and a Step-1 array; the banner prints them, the jobs and array tasks the submission queues and the disk of the fold matrices and pair export it keeps: `submit_pipeline.sh` 28 jobs, 586 tasks and up to 1,062 GB by default (10, 207 and up to 357 GB for one arm), the root orchestrator 43 jobs, 491 tasks and up to 910 GB (14, 160 and up to 272 GB). The eval's per-pair CSVs (about 5 GB per arm), in the root the peers' fold matrices (about 14 GB, once, with the shared array) and the checkpoints come on top: each banner names them on a `not counted:` line (since 2026-10-21). Slurm's `MaxSubmitJobs` limit counts every array task, and a four-arm run queues about three times what one arm did: check `sacctmgr show assoc user=$USER format=MaxSubmit,GrpSubmit` and the QOS's `MaxSubmitPU` first. `MAX_CONCURRENT` / `EVAL_CONCURRENT` cap each array on its own.
- Read a difference between arms with its confounds: the encoder input width, and with it the entry layer's capacity, differs per arm (on the 2026-08-11 cv1 winner set, re-fit on the current embedding files and shared folds, cv1 fold 0 at 0.8: `none` 4,864, `per_modality` 1,951, `joint` 1,570, `both` 1,291 inputs) and `L1_LAMBDAS` is retuned for none of them; under `PCA_METHOD=plain` a PCA also z-scores every column, where `none` rescales each modality by one scalar; `both`'s joint PCA z-scores the per-modality components again, so every kept component weighs the same; and another width draws another random stream (initialisation, shuffles, dropout masks). The root orchestrator's table across the arms (`../sl_comparison/results/pca_arms/tab_pca.csv`) reads each arm against the first with its fold-paired spread.
- These arms replace the fixed-selection PCA ablation of 2026-10-19 (`run_pca_ablation.sh`, Stage 5b: each combo retrained under the three other arms and read against the reported one), which was never run on the cluster; its flags, `PCA_ABLATION_*` settings and `pcaabl_*` dirs are gone.

#### Review-response additions (2026-11-12)

Five additions for the ICLR 2027 revision. Each script's header has the full contract; this is the map.

- **Per-fold modality selection** — `SELECTION_FRAC` (e.g. `0.1`), `SELECTION_SEED` (default `42`), `SELECTION_MODE` (`per_fold` by default when `SELECTION_FRAC > 0`, else `global`; pass `SELECTION_MODE=global` to rank on validation with a split held out) in `config.conf`, which produce `SELECTION_ARGS`. With `SELECTION_FRAC > 0`, every siamese training call (`train.py --selection_frac`) holds out a uniform random fraction of each fold's **training** pairs as a selection split. The split is disjoint from that fold's train, validation and test pairs. It is seeded by `(SELECTION_SEED, cv, fold)`, so it is identical in every run and variant on the same folds, and its sha256 is recorded as `sel_digest`. Genes that appear only in the selection split are excluded from that fold's preprocessing fit. Every run also scores its selected checkpoint on that split (`sel_auprg`, `sel_auprg_any`, `sel_auroc_any`, `sel_aupr_any`). `per_fold` picks each category's embedding per tier and per fold on that score (`fold_selection.py`) and trains the combo with `--fold_modalities`. Selection ranks only selection-split scores: validation labels are used only for early stopping (each candidate's selection-split score is that of its validation-chosen checkpoint), and test labels never reach it. The split is random training pairs at every tier, so at Tiers II and III about 97% of its pairs have both genes in that fold's training pairs: it ranks embeddings on a transductive (Tier I-like) task, not on unseen genes. SLeuth models then train on 90% of each fold's training pairs, while SLMGAE, SLGNN, NSF4SL and ELISL train on 100%. Since 2026-11-13 these settings are the `config.conf` / `run_best_per_category.conf` defaults (`PCA_ARMS="none joint"`, `SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head"`, `SELECTION_VARIANT=pd:pooled`, `BESTCAT_EXTRA_VARIANTS=1`, `SELECTION_FRAC=0.1`, `SELECTION_SEED=42`, `SELECTION_MODE=per_fold`, `ALLMOD_COMBO=1`), so `bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate` alone runs the paper's protocol. The explicit form, from `siamese_sl/`:
  ```
  PCA_ARMS="none joint" SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head" SELECTION_VARIANT=pd:pooled \
  BESTCAT_EXTRA_VARIANTS=1 SELECTION_FRAC=0.1 SELECTION_SEED=42 SELECTION_MODE=per_fold ALLMOD_COMBO=1 \
  bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate
  ```
- **From-scratch fine-tuning column** — `eval_finetuning.conf` `FT_MODES=("LP_last" "Full" "Scratch")`. Scratch trains Full's parameter set from a random initialisation (seed `split_seed + 7919 + fold`) on the same screen splits. It is the "does pretraining on SynLethDB help" control for Table 6.
- **Bilinear probe** (`run_probe.sh`) — `logit = τ·(W x_i)ᵀ(W x_j) + b` on arm `none` pd:pooled's per-fold features, selection split and training settings, read from that combo's `config.json`. The root orchestrator (`run_benchmarking_siamese_vs_others.sh`) submits it in arm `none` (afterok on that arm's combo job) and makes arm `none`'s compare wait for it (2026-11-13). After `submit_pipeline.sh` alone, submit it by hand and refresh the tables only AFTER the chained arm-`none` paper tables and the cross-arm table have finished (two compares of one arm must never overlap):
  ```
  sbatch --dependency=afterok:<arm none combo job> --export=ALL slurm/run_probe.sh      # from siamese_sl/
  # then FROM THE REPO ROOT, after the chained sl_paper_tables jobs (both arms and the cross-arm one):
  PCA_ARM=none sbatch --dependency=afterany:<probe job>:<last chained paper-tables job> --export=ALL sl_comparison/slurm/run_compare.sh
  PCA_ARM=none sbatch --dependency=afterany:<compare job> --export=ALL sl_comparison/slurm/run_paper_tables.sh
  PCA_ARMS="none joint" PCA_ARM=none sbatch --dependency=afterany:<that paper-tables job> --export=ALL sl_comparison/slurm/run_paper_tables.sh --pca_arms_table
  ```
  It appears as the `BilinearProbe` baseline row (arm `none` only). `paper_tables.py` tabulates it only when its `probe_source` names the `config.json` of the combo `var_pd_pooled__latest_combo_<cv>.txt` names now, so after a combo retrain it is listed missing until `run_probe.sh` runs again.
- **All-modalities combo** (`run_allmod_combo.sh`, `ALLMOD_COMBO=1` in `config.conf`, the default since 2026-11-13) — per PCA arm of `PCA_ARMS` and variant of `SIAMESE_VARIANTS`, the multi-modal model on EVERY `EMB_CATALOG` embedding in catalog order (an absent `OPTIONAL_TYPES` file is skipped and recorded; any other missing file refuses the task), with no modality selection: the arm's `PCA_MULTI_ARGS` (arm `none`: per-modality rescale + concat, ~9.4k columns for the 16 generated embeddings; arm `joint`: standardise + PCA of the concatenation at `PCA_JOINT_VARIANCE`), the variant's flags and the selected combo's training settings, `ARCH_ARGS`, L1, seed and `SELECTION_ARGS` (the selection split is held out, so it trains on the selected combo's pairs, and chooses nothing). `train.py --all_modalities` records it in `config.json` (an `arch_signature` key; refused in `ARCH_ARGS` and with `--fold_modalities`). One array task per CV; outputs `<arm root>/allmod__<prefix>all_modalities_<cv>/` (+ `all_modalities.json`) and marker `<arm root>/allmod__<prefix>latest_allmod_<cv>.txt`, names no selected-combo glob matches. Both orchestrators submit it with `ALLMOD_COMBO=1` after the generation only (no Step-1 / selection dependency) and chain the arm's compare on it; rows `siamese_sl_allmod` / `siamese_sl_allmod_<tag>` (collect_and_regrade_siamese.sh, `collect_siamese.py --allmod`) appear in `comparison.csv`, in `tab_tiers` as kind `ablation` after the selected SLeuth rows (eligible for the bold, never `beats_peers_*` / margins), and in `bootstrap_ci.py`, which adds the paired difference selected − all-modalities by default. `paper_tables.py` tabulates an all-modalities row only when its run's `arch_signature` equals the selected combo's outside the modality set. With `ALLMOD_COMBO=0` nothing is built, collected or listed missing. For a run that already exists:
  ```
  for a in none joint; do
    ALLMOD_COMBO=1 PCA_ARM=$a PCA_ARMS="none joint" SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head" \
    SELECTION_FRAC=0.1 SELECTION_SEED=42 sbatch --export=ALL \
      --output=slurm/logs/allmod_combo_${a}_%A_%a.out --error=slurm/logs/allmod_combo_${a}_%A_%a.err \
      slurm/run_allmod_combo.sh
  done
  # then per arm, FROM THE REPO ROOT (run_compare.sh works in its submit dir), after its job:
  #   ALLMOD_COMBO=1 PCA_ARM=<arm> PCA_ARMS="none joint" SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head" \
  #     sbatch --dependency=afterany:<allmod job> --export=ALL sl_comparison/slurm/run_compare.sh
  #   and sl_comparison/slurm/run_paper_tables.sh the same way after the compare
  ```
- **Bootstrap CIs** — `../sl_comparison/bootstrap_ci.py --pca_arm <arm>` resamples the saved test (score, label) pairs per fold and tier (1000 resamples by default; nothing is retrained). It covers AUROC, AUPR, AUPRG and the other threshold-free metrics, the paired model-vs-model differences, and the external-screen Spearman. Outputs go to `../sl_comparison/results[/pca_<arm>]/bootstrap/`: `ci_tiers.csv`, `ci_tiers_diffs.csv`, `ci_external.csv`, `ci_external_diffs.csv` and `manifest.json`. Order per arm: the tables first (bootstrap_ci reads their `tab_tiers.csv` and skips every row the tables refused, and skips the BilinearProbe outside arm `none`), then `bootstrap_ci.py`, then the tables again, which fill the CI columns (e.g. of the SLDB-agreement table) from `bootstrap/`. Since 2026-11-13 one `run_paper_tables.sh` job of an arm does all three (`bootstrap_ci.py --skip_missing`; `PT_BOOTSTRAP=0` builds the tables only), so every orchestrated or pasted tables job ends with the CIs.

### Stage 1: Embedding Generation

| File | Description |
|------|-------------|
| `run_generate_embeddings.sh` | **SLURM job.** First DELETES every `ALL_EMBEDDINGS` `.pt` in `DATA_DIR` (plus the legacy `all_genes_esm.pt`), the computed `*_gene_embeddings.pkl` caches and the NCBI sequence caches (downloaded sources in `CACHE_DIR` are kept). It then downloads the SynLethDB SL and non-SL pairs (each skipped only when its `.txt` and its `.sources.tsv` sidecar both exist; a failed download is fatal, and by then the old embeddings are already gone), always rebuilds `data/gene_universe.txt` (HGNC protein-coding ∪ every pair gene), and regenerates everything: ESM-2 (GPU-heavy, fatal on failure), GO anc2vec, and the `PRECOMPUTED_TYPES` via `generate_embeddings.py` (a failed type is retried once, except `OPTIONAL_TYPES`). It exits 1 at the end when any `EMB_CATALOG` type outside `OPTIONAL_TYPES` has no `.pt`: Step 1 would skip it with exit 0 and the combo would select over a smaller catalog than the paper's (with `--kill-on-invalid-dep` that cancels every `afterok` job behind generation, in the root DAG the shared models too). It also exits 1 when `embedding_versions.py` finds a catalog `.pt` that is not a current build: a builder that fails falls back to a cache file the cleanup keeps (an alt-named pkl such as GenePert's `gene_to_geneformer_embed.pkl`, or an `embeddings_cache/<type>/` copy) and stamps its output non-current. It does NOT reuse existing embeddings: pass `--skip-generate` to `submit_pipeline.sh` to keep them (the root `run_benchmarking_siamese_vs_others.sh` skips this job only when `data/all_genes_kg_complex.pt`'s `gene_order` matches `gene_universe.txt` AND `siamese_sl/embedding_versions.py --data_dir data` exits 0; `--force-generate` always runs it). |
| `run_generate_embeddings.conf` | **Config.** Lists embedding types for `generate_embeddings.py` (`PRECOMPUTED_TYPES[]`) and `OPTIONAL_TYPES[]` (default `esmc scprint seqvec genept`, none of them among the paper's 16 single-modality embeddings): the types whose absence does not fail the job. `esmc` is commented out of `PRECOMPUTED_TYPES`, so the combo ranks the paper's 16 candidates (a working ESM-C build would add a 17th `protein_seq` candidate); uncomment it to rank ESM-C. The full catalog (`ALL_EMBEDDINGS[]`) comes from `config.conf`. |

### Stage 2: Embedding Benchmark

| File | Description |
|------|-------------|
| `run_embedding_benchmark.sh` | **SLURM array worker** (one array per owner of single-embedding runs). Each array task trains the siamese model on one (embedding, CV) combination with the owner arm's `PCA_SINGLE_ARGS` (native dimensions, no PCA, for arms `none` / `joint`; PCA within the modality for `per_modality` / `both`), into `<owner root>/<type>_<cv>/`. Maps `SLURM_ARRAY_TASK_ID` to `(embedding_index, cv_index)` using `ALL_EMBEDDINGS[]` and `CV_TYPES[]`. Skips gracefully if the embedding file is missing — after deleting its output directory, so a modality that failed to regenerate produces no row (or a MISSING cell) rather than the previous run's row. Writes `run_token.txt` after a successful train. NOT resume-safe: the task deletes its output directory and retrains unconditionally. That is deliberate — skipping on an existing `results.json` would let a stale row from a previous fold set or config survive a re-run and be tabulated as if it were fresh — but it means re-submitting the array after a partial failure retrains every task, not just the failed ones. A task that never starts keeps its old directory; the summary shows it as STALE by its `run_token.txt`. |
| `run_embedding_benchmark_summary.sh` | **SLURM job** (runs after array). Collects `results.json` from the benchmark directories carrying this submission's `RUN_TOKEN` in `run_token.txt` (any other is a STALE cell, not tabulated; a direct `sbatch` must export the array's token), prints test stratified AUROC, test stratified AUPRG and test `auprg_any` (OR label) tables, and saves `<owner root>/embedding_benchmark_summary.json` (with `pca_arms`, `pca` and `pca_args`: the arms of this run that read those runs, one or both of the pair, and their PCA). None of these is the selection metric: checkpoints are selected on the VALIDATION `--select_metric` (default `auprg_any`), and nothing downstream reads this table. |
| `run_embedding_benchmark.conf` | **Config.** Empty by default (shared training hyperparameters, CV types, and `ALL_EMBEDDINGS[]` come from `config.conf`). Uncomment overrides here to tune the benchmark independently. |

### Stage 3: Best-per-Category

| File | Description |
|------|-------------|
| `run_best_per_category.sh` | **SLURM array worker** (one array per owner of single-embedding runs). Each array task trains the siamese model on one (embedding, CV) combination with the owner arm's `PCA_SINGLE_ARGS` (native dimensions, `--no_post_pca`, for arms `none` / `joint`; `--pca_variance $PCA_MODALITY_VARIANCE --no_post_pca` for `per_modality` / `both`; `PCA_VARIANCE` only names the `pcavar<variance>_*` directory), into `<owner root>/`, under `SELECTION_VARIANT` + `ARCH_ARGS`, with `PCA_METHOD` = plain (default), robust or svd selecting the impute/normalize pipeline. Preprocessing is fit per-fold on `PREPROCESSING_FIT_SCOPE` rows (default `train`, leak-free). Maps `SLURM_ARRAY_TASK_ID` to `(variant block, embedding_index, cv_index)` using `EMB_CATALOG[]`, `CV_TYPES[]` and `bestcat_task_variants` (see "How Array Task Mapping Works"); with `BESTCAT_EXTRA_VARIANTS=1` the blocks after the first train the other `SIAMESE_VARIANTS` for the single-modality table only, into `<owner root>/<solo prefix>pcavar<variance>_<type>_<cv>/`, and delete their checkpoints and prediction matrices once `results.json` exists (`BESTCAT_X_KEEP_PREDICTIONS=1` keeps them). Writes `run_token.txt` after a successful train. NOT resume-safe: the task deletes its output directory and retrains unconditionally. That is deliberate — skipping on an existing `results.json` would let a stale row from a previous fold set or config survive a re-run and be tabulated as if it were fresh — but it means re-submitting the array after a partial failure retrains every task, not just the failed ones. |
| `run_best_per_category_combo.sh` | **SLURM job** (one per PCA arm, after its owner's array). First deletes its variants' `latest_combo` markers in the arm's root and moves their previous `<arm root>/<prefix>best_cat_*` combos to `<arm root>/_superseded_combos/job_<id>/` (out of the markers and the `best_cat_*` / `var_*` globs the consumers use, though the opt-in `SIAMESE_BASELINE_DISCOVER=all` sweep, which has no freshness check, still rglobs them; deleted once Step 2 succeeds, kept there if it fails). Step 2: selects the best embedding per biological category once (on `SELECTION_CV`), from the owner's `<owner root>/pcavar*` runs, and writes `<arm root>/best_per_category.json` (its `_selection_meta` names the arm and the runs it ranked). Step 3: trains one multi-modal combo per `SIAMESE_VARIANTS` × `CV_TYPES`, all on the same winners, with the arm's `PCA_MULTI_ARGS`, and writes a per-variant `<arm root>/<prefix>latest_combo_<cv>.txt` marker (its text: the combo's path from `siamese_sl/`). Step 4: prints the summary table, one block per variant. Once Step 2 has succeeded it also removes, in the arm's namespaces, what was computed from the combos it replaces (as the orchestrators do at submission) and the table across the arms, `sl_comparison/results/pca_arms/`. |
| `run_best_per_category.conf` | **Config.** Job-specific settings: `PCA_VARIANCE` (overridable via `BEST_CAT_PCA_VARIANCE`; a legacy results-dir tag only — it changes no training), `PCA_METHOD` (plain, default ∣ robust ∣ svd), `PREPROCESSING_FIT_SCOPE` (train ∣ all), `EMB_CATALOG[]` (type:file:category, genept excluded for data leakage; bioconceptvec and the STRING-derived ppi / kg embeddings carry a smaller literature-derived risk, noted there), `BESTCAT_EXTRA_VARIANTS` (0 ∣ 1, default 1 since 2026-11-13) and the helpers `bestcat_task_variants`, `bestcat_n_tasks` (the Step-1 array size every launcher uses) and `bestcat_solo_prefix`. The PCA of the single-embedding runs and of the combo is the PCA arm's (`config.conf`, "PCA arms"), not set here; for the arms with the joint PCA note that a variance target of 0.8 resolves to k≈1087 under `plain` vs 145 under `svd`, and k drifts with the combo's modality selection (`PCA_JOINT_DIM` pins it). Shared training hyperparameters come from `config.conf`; uncomment overrides here to tune independently. |

### Stage 4: External-Screen Evaluation + Fine-tuning

| File | Description |
|------|-------------|
| `eval_finetuning.sh` | **SLURM array** (besides the metric CSVs each task writes, per screen, `<Screen>_sl_probabilities.csv` (zero-shot SL probability of every screen pair for every head), `<Screen>_finetuned_allpairs_<mode>.csv` (the fine-tuned predicted value of every pair, with its `split`) and `finetuned_models/`, which Stage 6 reads; chained after `run_best_per_category_combo.sh` via `afterok`; task = `cv_index × n_variants + variant_index`; with `EVAL_MODEL_DIR` / `EVAL_VARIANT` set, task = `cv_index`, and a task past the cv list FAILS, since only an override inherited into the chained array produces one; one array per PCA arm, each logging its arm). Runs `eval_finetuning.py` against each variant's CV3 combo of the job's arm (a pooled model's single head scores every screen, logged and recorded in `config_used.json` `effective_head` as `ANY (pooled)`; an `--inner_norm batch` model keeps its pretrained running statistics during fine-tuning): zero-shot (closed-form OLS of an `(a, c)` affine head) + `LP_last`, `Full` and `Scratch` (Full's parameters from a random initialisation) SGD fine-tuning modes against the Adamson (γ and γGI), Corn and Gilbert external CRISPR screens. Per-fold metrics and a 5-fold ensemble. Skippable via `--skip-eval` in the orchestrator. A direct `sbatch` is the arm the submitting shell exports (arm `none` where it exports none) unless `PCA_ARM` is passed (`PCA_ARM=<arm> sbatch --export=ALL --array=0-11 slurm/eval_finetuning.sh`; its header lists the `EVAL_VARIANT`, `EVAL_MODEL_DIR` and `submit_pipeline.sh --eval-only` forms), so from a shell that exports none the job evaluates, and rewrites, arm `none`'s combos: pass `PCA_ARM=none` for arm `none`, since an arm's hand recipe leaves `PCA_ARM` exported. A re-eval by hand clears and rewrites the combo's `eval_finetuning_<cv>/`, `finetuned_models/` included: rerun that arm's pair export (its manifest fingerprints those models), its external transfer and its paper tables after it. |
| `run_ablation.sh` | **SLURM array worker** (Stage 5, one array per PCA arm, chained after the arm's combo via `afterok`; writes `run_token.txt` after a successful train). Each array task retrains one variant's multi-modal combo with **one biological category removed**, on one CV — `\|ABLATE_CATEGORIES\|` × `\|CV_TYPES\|` × `\|SIAMESE_VARIANTS\|` = 72 tasks (variant-major; the baseline's tasks 0-17 keep the old layout). Reuses `<arm root>/best_per_category.json`, so no re-selection happens. Every hyperparameter, fold and preprocessing setting matches `run_best_per_category_combo.sh` Step 3, the arm's PCA (`PCA_MULTI_ARGS`, echoed once as `PCA: ...`) included; only the modality list differs. The full combos are **not** retrained — the summary reads the combo named by each variant's `<arm root>/<prefix>latest_combo_<cv>.txt` as its reference row (a (variant, CV) with no marker skips, and so does a task whose marker does not name the current winners' `<arm root>/<prefix>best_cat_<names>_<cv>`), so this stage cannot change any reported number. Writes `<arm root>/<prefix>ablate_no_<category>_<cv>/`, stamps `results.json["ablation"]`, and deletes checkpoints afterwards (`ABLATION_PURGE_CHECKPOINTS=1`) since no ablated model is ever reloaded. The fold prediction matrices are kept (`ABLATION_PURGE_PREDICTIONS=0`, for `paper_tables.py --regrade`): 5 × 313 MB per run, about 113 GB for an arm's 72 runs, about 450 GB for the four arms. |
| `run_ablation_summary.sh` | **SLURM job** (one per PCA arm; runs after the arm's ablation array, `afterany` so partial failures still produce a table). Writes `<arm root>/ablation_summary.csv` (with a leading `pca_arm` column and a `variant` column; the solo scores come from the single-embedding runs the selection's `_selection_meta.step1_root` names) and prints a per-(variant, CV) table, each variant against its own reference combo, pairing each category's **leave-one-out Δ** against that category's **standalone** score from the best-per-category pass. Reading the Δ alone is how ablation tables mislead — under redundancy a strong category costs nothing to remove — so both columns are reported side by side. A **positive** Δ means removing the category *helped*. The CSV has one `dropped=none` row per (variant, CV) for the reference, the fold mean and population sd of every metric (`<metric>`, `<metric>_std`, `ref_<metric>_mean` / `_std`), `solo_auprg` / `solo_auprg_std` with the `solo_variant` it was trained as (the row's own variant only where `BESTCAT_EXTRA_VARIANTS=1` trained it in the same Step-1 array, else `SELECTION_VARIANT`; blank when the selection run's `run_token.txt` is not the `_selection_meta.run_token` Step 2 recorded, e.g. a cv2/cv3 Step-1 task that never started), a MISMATCH verdict (no delta) for an ablation that is not reference-minus-one or carries another submission's `run_token.txt`, and the paired fold-by-fold spread of the AUPRG Δ (`delta_auprg_any_paired_sd`, population sd as in `paper_tables`' tab_loco; `delta_auprg_any_se`, sample sd / √n; `n_paired`). A row is called HARMFUL (Δ > +0.005) or contributes (Δ < −0.005) only when \|Δ\| also exceeds 2 SE; otherwise it is "within fold noise" (on 2026-09-23 a fixed ±0.005 cut labelled 12 rows HARMFUL, of which one clears 2 SE). |
| `run_ablation.conf` | **Config.** `ABLATE_CATEGORIES[]` (must match the category labels in `EMB_CATALOG`) and checkpoint/prediction retention flags; the PCA is the arm's (`config.conf`). Documents why the ablation drops a **category** rather than an embedding (dropping one embedding merely promotes its category's runner-up), and the confound that the encoder's input gets narrower as modalities are removed (in arm `none`, no PCA; in an arm with a variance-target PCA the kept k moves too). To control for capacity, pin one width for the combo **and** its ablation: an arm with the joint PCA at an exact count for the whole pipeline, e.g. `PCA_ARMS=joint PCA_JOINT_DIM=<k> bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate` (or the same exports for `--best-cat-only --skip-generate` and then `--ablation-only`, which refuses, before anything is removed, combos trained at other targets than its shell's; without `--skip-generate` the run regenerates the embeddings, which removes EVERY arm's combo markers). `<k>` is one count for `joint` and `both` and must stay below the narrowest width their joint PCA meets, the ablations' included (in `both` about 1,140 without `ppi` at 0.8): `train.py` refuses a count at or above it. `run_ablation.sh` refuses an ablation whose architecture signature — the PCA keys included — differs from its reference combo, so a width pinned for the ablation alone cannot run (the former `ABLATION_POST_PCA_DIM` is refused). |
| `eval_finetuning.conf` | **Config.** `MODEL_DIR` resolution order: `EVAL_MODEL_DIR` env var → `EVAL_VARIANT`'s `<arm root>/<prefix>latest_combo_cv3.txt` marker (written by the arm's combo job; the arm is the job's `PCA_ARM`). There is no fallback for any variant: with neither, `MODEL_DIR` is empty and `eval_finetuning.sh` stops with an error. Plus `EMBEDDINGS_PATHS[]` (must match combo's modalities), `DATASETS[]`, `FT_MODES[]`, and fine-tuning hyperparameters (`FT_EPOCHS`, `FT_LR`, `TRAIN_FRAC`, `SPLIT_SEED`). |

### Stage 6: All-Gene-Pair Export

| File | Description |
|------|-------------|
| `export_pair_predictions.sh` | **SLURM job** (job name `pair_export`; the last jobs each orchestrator queues, one per PCA arm, `afterany` on every job of every arm, so it can neither delay a job of that submission nor, should the disk fill, break one: in the root DAG that covers every table, the table across the arms included. `submit_pipeline.sh` queues neither the cell-line ablation nor the `sl_comparison` jobs; when you submit those by hand (its printed block included, whose jobs the exports it queued do not wait for), pass it `--skip-pair-export` and `sbatch` the export after them, as `sl_comparison/README.md` step 9 does. Its header carries `--dependency=singleton`: one `pair_export` job of a user runs at a time). One job per PCA arm over every `SIAMESE_VARIANTS` × `PAIR_EXPORT_CVS` combo named by the arm's `<arm root>/<prefix>latest_combo_<cv>.txt` (a direct `sbatch` is the arm of the `PCA_ARM` passed or exported, arm `none` only when neither is set: pass `PCA_ARM=none` for arm `none`, since an arm's hand recipe leaves `PCA_ARM` exported) (or the one combo in `PAIR_EXPORT_MODEL_DIR`): runs `export_pair_predictions.py` → `<combo>/pair_predictions/` (`sl_prob_<HEAD>.npy` per cell-line head plus `ANY`, `ANY` only for a pooled variant; `finetuned_<Screen>_<mode>_<split>.npy` for the cv3 combo, the only one Stage 4 fine-tunes; `genes.txt` / `genes.tsv`; `manifest.json`, written last). Refuses to run without CUDA. A combo without a marker, without `results.json`, or whose requested fine-tuned set is missing, partial, unreadable or fails while scored is named in the log and the job exits 1 after exporting the rest. It runs for minutes (one 19,397² matrix takes seconds; the job is bounded by writing 42 GB). 64G: the fold-mean accumulators take at most `PAIR_EXPORT_RAM_GB`. |
| `export_pair_predictions.conf` | **Config.** Every value can be set in the submitting shell. `PAIR_EXPORT_CVS` (default `cv3`: the combos Stage 4 evaluates), `PAIR_EXPORT_UNIVERSE` (`all` = every embedding gene, 19,397, 1.5 GB per matrix, the only choice that covers every Adamson pair; `shared` = the 8,844 benchmark genes in `sl_comparison/folds/genes.txt` order, 0.31 GB), `PAIR_EXPORT_DTYPE` (`float32` ∣ `float16`), `PAIR_EXPORT_PER_FOLD` and `PAIR_EXPORT_LOGITS` (0 ∣ 1), `PAIR_EXPORT_FT_SPLITS` (which screen split's fine-tuned models: default `cv1`; `none` = no fine-tuned matrices, which both orchestrators set when the eval is skipped), `PAIR_EXPORT_FT_DATASETS` (default `Adamson`), `PAIR_EXPORT_FT_MODES` (default `all`), `PAIR_EXPORT_RAM_GB` (default 40). Defaults: 2 per_head × (9 + 2) + 2 pooled × (1 + 2) = 28 matrices = 42 GB per PCA arm. Also defines `check_pair_export`, which the job and both orchestrators (at submission) run: any other value is refused, except a dataset or mode name (`PAIR_EXPORT_FT_DATASETS` / `PAIR_EXPORT_FT_MODES`; only `none` and a blank are refused there): the job alone matches those against the fine-tuned models, and a name that matches none (`adamson`, `full`) leaves that set out and makes the job exit 1 after writing the rest. |

### Utilities

| File | Description |
|------|-------------|
| `reset_env.sh` | **SLURM job.** Recreates the Python venv from scratch on a GPU node using pip. Installs all dependencies (PyTorch, ESM, robpy, etc.) and verifies imports. Its CUDA smoke test is commented out (by request). When generation runs, its ESM step refuses CPU and fails, which cancels the `afterok` DAG; when generation is skipped (the root orchestrator's auto-skip, `--skip-generate`, `--after`, the root `--skip-siamese`, `--ablation-only` / `--eval-only`) the trainers fall back to CPU silently, so check the first `bestcat` / `sl_compare` log for `Using CPU` / `Device: cpu`. Must run on a GPU node; submit from `siamese_sl/`. The venv it `rm -rf`'s and rebuilds is the one `config.conf`'s `PYTHON_PATH` lives in (`<venv>/bin/python3`), so moving the venv is that one edit; it refuses a directory that does not look like a venv. `--time` is 4 h: the install usually takes ~3 min but once took 52 min on the shared filesystem, and a TIMEOUT after the `rm -rf` cancels the whole DAG. |

### Output Directories

Paths are arm `none`'s: every other PCA arm writes the same names under `results/pca_<arm>/`, and the single-embedding runs (the benchmark and Step 1) are in the root of the arm that owns them (`results/` for `none` / `joint`, `results/pca_per_modality/` for `per_modality` / `both` by default).

| Directory | Source |
|-----------|--------|
| `slurm/logs/` | SLURM stdout/stderr. Array jobs: `bench_<JOB>_<TASK>.out`, `bestcat_<JOB>_<TASK>.out`. Summary/combo: `bench_summary_<JOB>.out`, `bestcat_combo_<JOB>.out`. Export: `pair_export_<JOB>.out`. Every PCA arm but `none` (and an owner other than `none`) puts its arm after the stage: `bestcat_combo_pca_joint_<JOB>.out`, `bestcat_pca_per_modality_<JOB>_<TASK>.out`. |
| `results/<type>_<cv>/` | Benchmark results (the owner's single-embedding PCA: native dims in `results/`), with `run_token.txt` |
| `results/pcavar<variance>_<type>_<cv>/` | Best-per-category Step 1 results (native dim, no PCA, in `results/`; PCA within the modality in `results/pca_per_modality/`; `pcavar<variance>` is only a name tag), with `run_token.txt` |
| `results/<solo prefix>pcavar<variance>_<type>_<cv>/` | Single-modality runs of the other variants (`BESTCAT_EXTRA_VARIANTS=1`; reporting only; `bestcat_solo_prefix`: `var_<kernel>_<cell_lines>__`, and `var_pd_per_head__` for `pd:per_head` when it is not `SELECTION_VARIANT`) |
| `results/best_cat_<combo>_<cv>/` | Multi-modal combo results (baseline `pd:per_head`) |
| `results/var_<kernel>_<cell_lines>__best_cat_<combo>_<cv>/` | Multi-modal combo results, other variants |
| `results/<prefix>ablate_no_<category>_<cv>/` | Leave-one-category-out runs, per variant |
| `results/ablation_summary.csv` | Leave-one-category-out table (the variants of the submission that wrote it; first column `pca_arm`) |
| `<cv3 combo>/eval_finetuning_<cv>/` | External-screen eval per screen split: metric CSVs, `<Screen>_predictions.csv`, `<Screen>_finetuned_predictions_<mode>.csv`, and (since 2026-10-07) `<Screen>_sl_probabilities.csv`, `<Screen>_finetuned_allpairs_<mode>.csv`, `finetuned_models/` |
| `<cv3 combo>/pair_predictions/` | All-gene-pair export (Stage 6): `sl_prob_<HEAD>.npy`, `finetuned_Adamson_<mode>_cv1.npy`, `genes.txt`, `genes.tsv`, `manifest.json`. Lives inside the combo dir, so it goes with it when the next combo job replaces that combo |
| `results/<prefix>latest_combo_<cv>.txt` | Marker: the variant's combo dir from the LAST combo job of the arm, as its path from `siamese_sl/` (read by the eval, the ablation and its summary, the cell-line ablation, the export and the sl_comparison collector) |
| `results/_superseded_combos/job_<id>/` | Previous combos moved aside by combo job `<id>`; deleted once its Step 2 succeeds, left there (unmarked; only the opt-in `SIAMESE_BASELINE_DISCOVER=all` collector sweep can still pick them up) if it fails |
| `results/embedding_benchmark_summary.json` | Benchmark summary JSON (owner roots only; `pca_arms`, `pca`, `pca_args`) |
| `results/best_per_category.json` | Category winner selections (`_selection_meta`: `pca_arm`, `pca`, `pca_args`, `step1_root`, `step1_pca`, `step1_pca_args`, `run_token`) |
| `results/pca_<arm>/` | PCA arm `joint`, `per_modality` or `both`: the same layout one directory down |

## How Array Task Mapping Works

Both array workers use the same formula to map `SLURM_ARRAY_TASK_ID` to an
(embedding, CV) pair:

```
embedding_index = SLURM_ARRAY_TASK_ID / num_cv_types
cv_index        = SLURM_ARRAY_TASK_ID % num_cv_types
```

`run_best_per_category.sh` puts that pair inside a variant block,
`task = (block × num_embeddings + embedding_index) × num_cv_types + cv_index`,
where block 0 is `SELECTION_VARIANT` and, with `BESTCAT_EXTRA_VARIANTS=1`,
blocks 1.. are the other `SIAMESE_VARIANTS` in order. With the default (0)
there is one block, so the mapping is the one above.

For `run_best_per_category.sh` (19 embeddings in `EMB_CATALOG`) and 3 CV types, task IDs 0-56 map to (the benchmark's 20-entry `ALL_EMBEDDINGS` gives 0-59):
- Task 0: embedding[0] / cv1
- Task 1: embedding[0] / cv2
- Task 2: embedding[0] / cv3
- Task 3: embedding[1] / cv1
- ...
- Task 56: embedding[18] / cv3

The submit script computes the array range dynamically:
`--array=0-$((NUM_EMB * NUM_CVS - 1))` for the benchmark and
`--array=0-$(( $(bestcat_n_tasks) - 1 ))` for best-per-category
(57 tasks by default, 228 with `BESTCAT_EXTRA_VARIANTS=1` and four variants),
each with a `%MAX_CONCURRENT` suffix only when `MAX_CONCURRENT` is a positive
integer.

## Customization

**Add a new embedding:** The array sizes adjust automatically on the next
`submit_pipeline.sh` run, but a catalog entry **alone** makes the generation job
fail *at the end*: Stage 1 builds only ESM-2, GO and `PRECOMPUTED_TYPES`, then
exits 1 on any `EMB_CATALOG` type outside `OPTIONAL_TYPES` with no `.pt`, which
— with `--kill-on-invalid-dep` — cancels every `afterok` job behind it, the root
DAG's shared models included. A new type needs all of:

1. `EMB_CATALOG[]` in `run_best_per_category.conf` (best-per-category) **and**
   `ALL_EMBEDDINGS[]` in `config.conf` (the benchmark, and the generation job's
   cleanup — a type missing there keeps a stale `.pt` across a regeneration).
2. A builder for the type in `generate_embeddings.py` **and** the type in
   `PRECOMPUTED_TYPES[]` (`run_generate_embeddings.conf`) — or, to accept its
   absence on purpose, in `OPTIONAL_TYPES[]`. Hand-placing a `.pt` does not
   work: `embedding_versions.py` reports any file without the current
   `generator_version` as stale, and the job exits 1 on that check too.
3. A **new category** also in `ABLATE_CATEGORIES[]` (`run_ablation.conf`), or
   the leave-one-category-out table silently omits it.

**PCA arms:** which arms a submission trains is `PCA_ARMS` (default all four;
any subset, run in the order `none joint per_modality both`), and their
targets are `PCA_MODALITY_VARIANCE` (the per-modality PCA of `per_modality` /
`both`), `PCA_JOINT_VARIANCE` (the PCA of the concatenation of `joint` /
`both`), both 0.8, and `PCA_JOINT_DIM` (an exact width for the latter, which
wins; empty by default; one count for `joint` and `both`, kept below the
narrowest width their joint PCA meets, as `train.py` refuses one at or above
it),
exported for the whole submission so every job of every arm reads the same
values (see "PCA arms" above). Both orchestrators refuse a bad value, and
`POST_PCA_VARIANCE` / `POST_PCA_DIM`, before anything changes. Never pass a
PCA flag through `ARCH_ARGS` (`config.conf` refuses it). Export the same
targets again for a later `--ablation-only` of those arms, or the root's
`--skip-siamese`: both refuse, before anything changes, combos trained at
other targets than the shell's, and print the command with theirs (when one
set of targets trains them all; else `--ablation-only` prints one line per
arm or group of variants, `--skip-siamese` a WARNING: see "File Reference").
E.g. the published model's PCA alone, or every arm at other targets:
```bash
PCA_ARMS=joint bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate
PCA_ARMS=joint PCA_JOINT_DIM=150 bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate --best-cat-only
PCA_MODALITY_VARIANCE=0.9 bash slurm/submit_pipeline.sh
```
An arm `PCA_ARMS` leaves out keeps its outputs only when the submission does
not regenerate the embeddings (`--skip-generate`): one that does removes the
combo markers of every arm, in all four roots, since those combos were trained
on the vectors it replaces. Every best-per-category submission also removes
the table across the arms, `../sl_comparison/results/pca_arms/`. And a subset
that lists the first arm of a pair (`none`, `per_modality`) without its
partner (`joint`, `both`) retrains, as the pair's owner, the single-embedding
runs the partner's kept selection ranked: a WARNING says so before anything
changes, and the partner's `tab:solo` is another run's until the partner is
rerun. Arms can so be trained by separate submissions, pairs whole or second
arms alone, but the root orchestrator's table across the arms then lists as
stale (no values) every arm whose selection another submission made than its
first arm's: its `PAPER COMPLETE` needs one submission's selection of every
arm.
`BEST_CAT_PCA_VARIANCE` only renames the `pcavar<value>_*` directories; it
changes no training.

**Limit GPU concurrency:** By default no array is capped: `MAX_CONCURRENT`
(each benchmark, best-per-category and ablation array, capped on its own),
`EVAL_CONCURRENT` (each eval array) and, in the root orchestrator,
`SHARED_CONCURRENT` (SLGNN / NSF4SL / SLMGAE array) are empty in
`config.conf`, so SLURM runs as many tasks as the partition allows. Every
owner submits its own benchmark and best-per-category arrays and every PCA arm
its own ablation and eval arrays, side by side: with caps set and all four
arms, the stage after the combos runs up to 4 × (`MAX_CONCURRENT` +
`EVAL_CONCURRENT`) tasks. Empty, `0`, `none`, `all` or `unlimited` means no cap; a
positive integer `N` (no leading zero) adds `%N` to that array's `--array`;
anything else is refused before anything changes (`submit_pipeline.sh` checks
only `MAX_CONCURRENT` and `EVAL_CONCURRENT`: it has no shared array):
```bash
MAX_CONCURRENT=8 bash slurm/submit_pipeline.sh
```

**Change SLURM partition/node:** Edit the `#SBATCH` headers in each `.sh`
file: every job script pins `--partition=gpu_Prosmn` and `--nodelist=gpu3`
(2026-10-01, by request; gpu2 from 2026-09-23 until then, after gpu3 failed an
ECC test). `GPU_NODES` (`config.conf`) moves a
whole orchestrated submission without editing them: both orchestrators pass it
as `--nodelist` on the command line, which replaces the directive. One node
(`GPU_NODES=gpu4`) works on every Slurm; a list (`GPU_NODES="gpu3,gpu4"`) lets
each one-node job take whichever listed node is free, on Slurm >= 23.02 only
(older Slurm rejects it: invalid number of nodes). `GPU_EXCLUDE` only adds
`--exclude` and cannot lift the directive's `--nodelist` (SLURM rejects a job
whose required and excluded nodes overlap), so both orchestrators refuse it
unless `GPU_NODES` is set and shares no node with it. Before anything changes,
both orchestrators always run `sbatch --test-only` (it submits nothing) on the
largest job script of each stage they submit (`reset_env.sh`,
`run_generate_embeddings.sh`, `run_best_per_category_combo.sh`; in
`submit_pipeline.sh` also `run_ablation.sh` and `eval_finetuning.sh` when it
queues them, so under `--ablation-only` alone `run_ablation.sh` is the one
stage script probed, with `--reset-env` also `reset_env.sh`; once per script,
whatever the number of arms; in the root orchestrator also
`sl_comparison/slurm/run_shared_models.sh`), with
`$GPU_NODE_ARGS`, or against each script's `#SBATCH --nodelist=gpu3` pin when
it is empty, and refuse when SLURM rejects one; `sinfo` prints the state of
the node(s) the jobs will run on, with a WARNING (not a refusal) when one is
down, drained or failed (`--test-only` cannot place a job on such a node, so
its failed probe is that WARNING too, and the remaining probes are skipped).
A direct `sbatch` of one script, and the scripts no
orchestrator submits (`sl_comparison/slurm/preflight.sh`,
`SLGNN/slurm/run_slgnn.sh`, `NSF4SL/slurm/run_nsf4sl.sh`), keep the pin unless
`--nodelist=<node>` is passed on the `sbatch` command line.
`submit_pipeline.sh` does not forward extra sbatch flags (unknown options are
rejected), so to override one worker without editing headers, submit it
directly from `siamese_sl/`; command-line flags override the `#SBATCH` headers,
and `--nodelist` must be overridden too:
```bash
cd siamese_sl
# One token for every job submitted from this shell (--export=ALL carries it):
# without it each array task mints its own, and the combo / summary refuse
# every directory as another submission's.
export RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"
# The PCA arm of these jobs (arm none without it; the Step-1 runs of joint are
# arm none's, those of both arm per_modality's: submit the owner's arm)
export PCA_ARM=none
# N = |ALL_EMBEDDINGS| x |CV_TYPES| = 20 x 3 = 60 for run_embedding_benchmark.sh
sbatch --export=ALL --partition=<partition> --nodelist=<node> --array=0-59 slurm/run_embedding_benchmark.sh
# best-per-category: size it as the launchers do (57 by default, 228 with
# BESTCAT_EXTRA_VARIANTS=1 and four variants)
N=$(bash -c 'unset SLURM_JOB_ID; source slurm/config.conf >/dev/null; source slurm/run_best_per_category.conf >/dev/null; bestcat_n_tasks')
sbatch --export=ALL --partition=<partition> --nodelist=<node> --array=0-$((N - 1)) slurm/run_best_per_category.sh
```
A direct submit does not queue the summary / combo jobs that
`submit_pipeline.sh` chains after the array; submit those afterwards from the
same shell (same exported `RUN_TOKEN` and `PCA_ARM`) with
`--export=ALL --dependency=afterany:<array job id>`, and the combo of the
pair's other arm as `PCA_ARM=<that arm> sbatch --export=ALL ...` after the same array.

A combo job submitted by hand, also one resubmitted alone with a finished
run's `RUN_TOKEN` (and its `PCA_ARM`, `PCA_ARMS` and `SELECTION_CV`, as the
combo job's own hints print them since 2026-10-21: `SELECTION_CV` is in no
signature, so a resubmission without the run's selects on the submitting
shell's, else cv1, and nothing in the job stops it) to retrain a failed combo,
removes (once
its selection has succeeded) what was computed from the combos it replaces in
its arm, as `submit_pipeline.sh` does at submission: the `sl_comparison` rows,
the cell-line ablation, the ablation, the external-transfer and paper tables,
and the table across the arms. It queues none of the jobs that rebuild them.
Rerun them all afterwards: `sl_comparison/README.md`, "After a combo job
submitted by hand".
