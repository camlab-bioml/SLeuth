# Four-model SL comparison (SLDB 3.0)

Fairly compares four synthetic-lethality models — **siamese_sl, SLMGAE, SLGNN,
NSF4SL** — on the SLDB 3.0 *given* positives and negatives:

- positives = `data/SL_SynLethDB_experimental.txt` (~22.2k pairs)
- negatives = `data/SL_SynLethDB_experimental_negatives.txt` (**all** ~106.7k
  used after moving the 5.5k pos/neg conflicts to the positive side; no random
  sampling)

**Label rule (`conflict_policy=or_collapse`).** A pair is positive if any screen
called it SL in any cell line, negative only if every screen of it called it
non-SL. The 5,542 pairs screened both ways are therefore positive — a screen
that detected lethality is positive evidence, one that did not is a failure to
detect in one context. `folds/conflicts.tsv` lists all 5,542 with the cell lines
and heads asserted on each side (a side naming no usable line is attributed to
`OTHER`): 1,783 disagree within one line, 3,759 across lines, 107 name no line
at all. This is the same relation siamese's `*_any` view is graded on, so every
row in the table now shares one positive set.

## How fairness is enforced

- **One canonical fold set.** `prepare_folds.py` builds the CV folds once over
  the union gene universe (8,844 genes = every gene in a given pos **or** neg
  pair, so every negative is scorable), one seed (42), and writes them to
  `folds/`. The three graph/embedding models all read the **identical**
  train/test division — calling each model's own splitter with "the same seed"
  would NOT produce identical folds (they index genes differently).
- **Graph = positives only.** For the GNN-structure models the adjacency is
  built from train **positives**; the given negatives are label-0 supervision,
  never graph edges ("no space for negatives → the rest is 0").
- **One evaluator.** `evaluate_model.py` scores every model's saved score matrix
  with the same `cal_metrics` + AUPRG, on the same test edges.
- **siamese_sl runs its own pipeline** (its embedding selection + cell-line
  model on SLDB 3.0) and is graded on the **OR-collapse** ("SL in any cell
  line") score. It reads the **same `folds/`** as the other three:
  `siamese_sl/slurm/config.conf` sets `SHARED_FOLDS=../sl_comparison/folds` and
  passes `--folds_dir` whenever `SHARED_FOLDS` is non-empty, in both
  `USE_CELL_LINES` modes, so the partition, the validation split used for
  checkpoint selection, and the negatives are shared, not merely similar.
  `SHARED_FOLDS` is required: train.py early-stops and selects checkpoints on
  that validation split only, so without `--folds_dir` every siamese training
  run exits. There is no own-split fallback.
- **siamese rows = this run's combos, re-graded, per PCA arm.** siamese runs
  once per PCA arm of `PCA_ARMS` (`siamese_sl/slurm/config.conf`, since
  2026-10-20: `none`, no PCA; `joint`, PCA on the concatenation of all
  modalities; `per_modality`, PCA within each modality; `both`), and each arm
  gets its own table: arm `none` reads `siamese_sl/results/` and writes
  `results/` (the paths of before 2026-10-20), arm X reads
  `siamese_sl/results/pca_X/` and writes `results/pca_X/`, under the same
  file names. `slurm/collect_and_regrade_siamese.sh` (sourced by
  `run_compare.sh`) takes, per variant and CV, only the combo named by the
  arm's `siamese_sl/results[/pca_<arm>]/<prefix>latest_combo_<cv>.txt` (the
  combo job deletes these when it starts and writes each on success; no
  marker, no row), then re-grades its exported score matrices through
  `evaluate_model.py`. A row whose re-grade is missing or fails is moved to
  the arm's `results[/pca_<arm>]/stale_siamese/`, so siamese's self-scored
  numbers (`collect_siamese.py` on its own) never reach the table. Each row
  also carries how its modalities were chosen (the arm's combo job's
  `best_per_category.json`: one per-category argmax on `SELECTION_CV`
  validation, reused for every CV and variant, not leakage-free), which
  `compare.py` prints under SELECTION BUDGET and writes into
  `comparison_notes.txt`. The three peers and DegreePrior are trained once:
  their rows stay in `results/`, and every arm's table reads them there.

**Headline metrics = AUPRG, AP_norm and AUROC.** AUPRG and AP_norm are
*baseline-anchored skill scores*: a random ranker scores 0 at any prevalence, so
they remain meaningful across folds whose test-set base rate differs (cv3 ranges
0.0843–0.2640 on the current folds; read the `baseline` column of the run
itself rather than this figure, which moves with the label rule). They are **not** prevalence-invariant — holding the ranker fixed
and varying π moves both. AUROC is the only genuinely prevalence-invariant
column. Since every model here shares the same per-fold test set, no cross-model
comparison in the table is affected either way.

Prefer **AP** (average precision) over the `AUPR` column. `AUPR` is the vendored
benchmark's trapezoidal `auc(recall, precision)`, kept for comparability with
published SLMGAE/SLGNN/NSF4SL numbers; it interpolates between PR points and
rewards ties, so a *constant* score matrix scores exactly (1+π)/2 — about 0.584
on the current cv3 folds, above every real model. AP gives a constant predictor
exactly π. Take π from the run's own `baseline` column: it changed when the
2026-08-10 label rule moved the 5,542 conflicted pairs to the positive side.

**`DegreePrior` is a control row, not a competitor.** `degree_baseline.py` scores
`log1p(deg_train(i)) + log1p(deg_train(j))` — no features, no learning — through
the identical evaluator. On the **2026-08-06 run** (the previous fold set) it
tied siamese_sl on cv1 (AUROC 0.9059 vs 0.9078) and beat it 3.6× on NDCG@10,
because SynLethDB positives are hub-enriched relative to the experimental
negatives; it collapsed to 0.6402 on cv2 and to exactly 0.5000 on cv3 (its
scores are constant when both genes are unseen). Those figures are from the
OLD label rule and have not been re-measured — the cv3 0.5000 is structural and
will hold, the cv1/cv2 values will move. Read cv1 results against whatever this
row scores in the same table, never against these numbers.

## Files

| File | Purpose |
|------|---------|
| `prepare_folds.py` | Generate + verify the shared leak-free CV folds → `folds/`. `meta.json` records `source_sha256` (the pair files and, when the conflict attribution ran, the `.sources.tsv` sidecars plus `siamese_sl/cell_line_vocab.py`, which they are parsed through); `--check_sources` compares them with the files now, read-only: exit 1 on a difference, a missing `source_sha256`, or a `source_sha256` recorded before the vocabulary was fingerprinted, 2 when a source cannot be read |
| `shared_folds.py` | Canonical loader (gene universe + per-fold split dicts) |
| `evaluate_model.py` | Score a model's saved score matrices (cal_metrics + AUPRG) |
| `auprg.py` | AUPRG (re-implementation of the Flach & Kull 2015 reference `prg.create_prg_curve` + `calc_auprg`) + normalized AUPR. The single definition: `siamese_sl/train.py` loads it by file path (no local copy), and `evaluate_model.py` and the SLGNN / NSF4SL / SLMGAE trainers import it |
| `collect_siamese.py` | Pick each siamese row's run (`--combo_markers`: this run's combo; `--discover_dir` is the arm's siamese root, `--out_dir` its comparison root) and map its `*_any` metrics into the unified schema. Self-scored until re-graded. Its opt-in sweep never ranks a run under another arm's root (`pca_none`, `pca_joint`, `pca_per_modality`, `pca_both`, `pca_arms` by name). `--pca_arm` (label only, since 2026-10-21; `slurm/collect_and_regrade_siamese.sh` passes the job's arm) records the arm in every row's `selection_audit.pca_arm`, pinned and sweep rows included, and a combo row's `modality_selection` also carries its selection's `pca_arm` / `pca`; the re-grade keeps both |
| `compare.py` | Print the four-model table + tidy CSV + `comparison_notes.txt`. Removes the previous CSV / notes first and exits 1 when no result row is usable, so an empty run never leaves an old table looking current. Per PCA arm (`run_compare.sh`): `--results_dir` the arm's directory, whose SLeuth rows (`siamese_sl`, `siamese_sl_*`) it reads, `--shared_results_dir sl_comparison/results`, whose other rows it reads (a shared-model row inside an arm's directory is named with a WARNING and not used), and `--pca_arm` / `--pca_label`, which put the arm under every table title, on the first line of `comparison_notes.txt` and in a trailing `pca_arm` column of the CSV |
| `external_transfer.py` | External-screen (Adamson / Corn / Gilbert) zero-shot table for every SLeuth variant, SLMGAE, SLGNN, NSF4SL and DegreePrior on identical rows, and SLeuth's fine-tuning table (population sd). Refuses an `eval_finetuning_<split>/` whose `config_used.json` is missing, matched `sldb_overlap` against positives only, or whose `checkpoint_sha256` / `fold_ids` do not match the current `checkpoints/fold_*_best.pt` (a retrain since the eval), unless `--allow_legacy_eval`. `--pca_arm` / `--pca_label` (label only, since 2026-10-21; `slurm/run_external_transfer.sh` passes the job's arm and its PCA in words) add a trailing `pca_arm` column to each of the five CSVs it writes and `PCA arm: <arm> (<label>)` as line 3 of `summary.md`, plus a Provenance bullet; without them the output is as before |
| `paper_tables.py` | Every table and number of `paper/main.tex` the run can compute, for all SLeuth variants, for ONE PCA arm (`--pca_arm`, default `$PCA_ARM`, else `none`) → its `results[/pca_<arm>]/paper_tables/` (`tab_*.csv`, each with a leading `pca_arm` column; `numbers.json`, whose `run.pca_arm` / `run.pca` / `run.pca_settings` and roots name the arm and the PCA its combos trained with; `README.md` titled `# Paper tables and numbers: PCA arm <arm>`, mapping each paper item to its file and column, and listing what is missing, apart from what is absent by design). The verdict, `PAPER COMPLETE` or `PAPER INCOMPLETE: <n> item(s)`, is README.md's line 3 (the first line under its title), numbers.json `verdict` and the last line it prints; it certifies the computed tables and numbers only (`paper/main.tex` is copied by hand and is not checked; its methods describe arm `joint`). Exits 0 only when complete, 1 when a table could not be built, 3 when every table was built but the verdict is INCOMPLETE (3, not 2: argparse and Python's can't-open-file already exit 2). `--pca_arms_table` builds the table across the arms of `--pca_arms` (default `$PCA_ARMS`; at least two) instead → `results/pca_arms/`: `tab_pca.csv` (per variant × tier × arm, against the first arm, `none` when listed: the arm's PCA, single-embedding runs, modalities, input width and parameters, graded AUPRG / AP / AUROC, the differences with their fold-paired spread and a verdict, `best_arm`; not yet in `paper/main.tex`), `numbers.json` (`items.pca`, with the 2x2 effects when all four arms are tabulated; `arms[]`) and `README.md` (each arm's paper tables and their verdict); its verdict is COMPLETE only when every cell is tabulated and every arm's paper tables are COMPLETE, and it exits 1 when no cell is tabulated (or fewer than two arms are asked for), 3 when incomplete |
| `test_paper_tables.py`, `test_comp_contracts.py`, `test_comparison_core.py`, `slurm/test_check_shared_inputs.py` | Deterministic tests on synthetic trees: `$PYTHON_PATH sl_comparison/test_<name>.py` from the repo root, with the interpreter of `siamese_sl/slurm/config.conf` (on the cluster `python3` is the system one, without numpy; `-m pytest <file> -q` also works where pytest is installed, and `reset_env.sh` does not install it). All four; the table used to name only the first two, so the evaluator/fold-provenance and shared-input checks were easy to miss. `siamese_sl/tests/test_external_transfer.py` covers `external_transfer.py` |
| `slurm/run_shared_folds.sh` | Verify (or regenerate) the shared folds + score the DegreePrior control; once, before the array |
| `slurm/run_shared_models.sh` | ARRAY worker: train + score ONE (model, CV) per task (`--array=0-8`, after `run_shared_folds.sh`) |
| `slurm/collect_and_regrade_siamese.sh` | Collect this run's siamese combos of one PCA arm and re-grade them with `evaluate_model.py` (sourced by `run_compare.sh`, after `slurm/pca_arm.sh`; it refuses a `PCA_ARM` that is not the arm `pca_arm.sh` resolved) |
| `slurm/pca_arm.sh` | Sourced by the per-arm jobs (`run_compare.sh`, `run_external_transfer.sh`, `run_paper_tables.sh`): resolves the job's `PCA_ARM` through `siamese_sl/slurm/config.conf` (read in a subshell) into `SLEUTH_ARM`, `SLEUTH_RESULTS` (the arm's siamese root), `SLEUTH_CMP` (its comparison root), `SLEUTH_PCA_ARMS` and `SLEUTH_PCA_LABEL`, and returns 1, before the job has changed anything, when `config.conf` refuses a setting, resolves another layout, or `PCA_ARM` is set but not exported |
| `slurm/python_path.sh` | Interpreter for every sl_comparison job: an exported `PYTHON_PATH`, else `siamese_sl/slurm/config.conf`'s (the siamese jobs' one) |
| `slurm/run_external_transfer.sh` | Runs `external_transfer.py`, per PCA arm, on every variant whose cv3 combo of that arm (directly in the arm's siamese root) and eval finished this run, plus the baselines that passed this run (their rows in `results/`); a model whose fold stamp is not the current `folds/` is left out (named in the log) instead of failing every table → `results[/pca_<arm>]/external_transfer/` |
| `slurm/run_paper_tables.sh` | Runs `paper_tables.py --pca_arm <arm>` for the job's PCA arm → `results[/pca_<arm>]/paper_tables/`, or, with `--pca_arms_table` as its FIRST argument, the table across the run's arms (`PCA_ARMS`, at least two) → `results/pca_arms/`, the last job a table waits for (only siamese's all-gene-pair exports, which no table reads, follow it); it exits with `paper_tables.py`'s status: `sacct` shows the `sl_paper_tables` job FAILED for 1 and 3. It refuses (exit 2, before anything is removed) a caller argument that would set the arm or a directory (`--pca_arm`, `--pca_arms`, `--out_dir`, `--comparison_csv`, `--external_dir`, `--root`, or an abbreviation) |

Each model consumes the shared folds via:
- **SLGNN** `SLGNN/train_slgnn.py --folds_dir ../sl_comparison/folds`
- **NSF4SL** `NSF4SL/train_nsf4sl.py --folds_dir ../sl_comparison/folds`
- **SLMGAE** `SLMGAE-in-pytorch/train_slmgae_shared.py --folds_dir ../sl_comparison/folds`
  (it keeps its native GO-BP / GO-CC / PPI similarity views, indexed by the
  6,375 genes of `data/List_Proteins_in_SL.txt`, and remaps them 6,375→8,844
  into the shared index. 4,587 of the 6,375 land in the shared universe on
  the current folds (the startup log prints the exact count); every other
  shared gene gets an all-zero view row. See "Upstream inputs kept as
  shipped" below.)
- **ELISL** `ELISL/train_elisl.py --folds_dir sl_comparison/folds` (context-free:
  SeqVec + STRING node2vec only; the authors' LightGBM forests, GPL v3, vendored
  in `ELISL/elisl_orig/`). Not in `run_shared_models.sh`'s array: its own
  launcher, `ELISL/slurm/run_elisl.sh`, which the root orchestrator submits
  with the shared models (and its input build when needed; `ELISL/README.md`), grades `results/ELISL_<cv>.json`, a shared
  row like the others (compare.py `MODEL_ORDER` after NSF4SL; paper_tables
  `tab_tiers` peer, `TIER_PEERS`; no external-screen rows). Deviations:
  `ELISL/README.md`.

## Run it

**One command (whole pipeline)** — from the repo root, submits the full SLURM
DAG (shared folds on the login node → reset_env → generate (always, since
2026-11-13; `--auto-generate` skips it when the embeddings on disk are
current, `--skip-generate` always) → {per
owner of single-embedding runs the siamese best-per-category array → per PCA
arm its combo → external-screen eval, leave-one-category-out ablation,
post-hoc cell-line ablation} ∥ {shared folds job → 3×3 model array}
→ per arm {final compare (after the arm's combo and cell-line ablation and the
3×3 array) ∥ external transfer (after the arm's eval array and the 3×3 array)}
→ the arm's paper tables → with two arms or more, the table across the arms
→ per arm the siamese all-gene-pair export
(`siamese_sl/slurm/export_pair_predictions.sh`, since 2026-10-07: the predicted
SL probability of every gene pair per cell-line head and the fine-tuned
Adamson predictions, for each SLeuth variant's cv3 combo; no table reads it;
`--skip-pair-export` omits it)). The owners are the first arm of each pair
(`none`, `joint`) / (`per_modality`, `both`) that `PCA_ARMS` lists: the two
arms of a pair rank the same single-embedding runs (native dimension, or PCA
within the modality) and so select the same modalities. With the default
`PCA_ARMS` (all four arms) the DAG is 43 jobs and 491 tasks counting every
array task (Slurm's `MaxSubmitJobs` limit counts each: the banner prints both
with the `sacctmgr` command that shows yours); `PCA_ARMS=none` gives the
14-job, 160-task DAG of before 2026-10-20. The shared folds job and the 3×3
array are submitted right after the generation, ahead of every siamese job
(since 2026-10-21; the DAG above is unchanged): under a scheduler that hands
a free GPU to the oldest eligible job their 9 tasks then run early, where
queued last they waited for every arm's ablation, and with them every arm's
compare and external transfer. Each arm's jobs are submitted as
`PCA_ARM=<arm> sbatch ... --export=ALL` (the arm in sbatch's own environment:
any `--export` list, `--export=ALL,PCA_ARM=<arm>` included, makes Slurm imply
`--get-user-env`, a login-environment fetch at every task launch that holds
the job when it fails) and log to `<stage>_pca_<arm>_<job id>.out`
(arm `none`: `<stage>_<job id>.out`). Every paper table and number it can
compute ends up in each arm's `paper_tables/`
(`sl_comparison/results/paper_tables/` for arm `none`,
`sl_comparison/results/pca_<arm>/paper_tables/` otherwise; `README.md` there
says which file and column holds each one; its line 3 is the verdict, see
`paper_tables.py` under "Files"), and the arms side by side in
`sl_comparison/results/pca_arms/` (`tab_pca.csv`; its `README.md` line 3 is
the verdict over every arm):

```bash
bash run_benchmarking_siamese_vs_others.sh --force-generate # the paper run: regenerate every embedding
bash run_benchmarking_siamese_vs_others.sh                  # full pipeline; skips generation only when
                                                            # data/all_genes_kg_complex.pt's gene_order matches
                                                            # data/gene_universe.txt and
                                                            # siamese_sl/embedding_versions.py exits 0
PCA_ARMS=joint bash run_benchmarking_siamese_vs_others.sh --skip-generate --skip-shared
                                                            # one PCA arm (or any subset), the others'
                                                            # outputs kept: see below for what still goes
bash run_benchmarking_siamese_vs_others.sh --skip-generate  # force-skip generation (only warns when the
                                                            # embeddings cannot be confirmed current)
bash run_benchmarking_siamese_vs_others.sh --skip-siamese   # only the 3 shared models
```

`--force-generate` cannot be combined with `--skip-generate`, `--after` or
`--skip-siamese`.

A run of a subset of the PCA arms (`PCA_ARMS=joint ...`) leaves the other
arms' siamese outputs and SLeuth rows in place, with four exceptions:
1. A run that regenerates the embeddings (`--force-generate`, or files the
   auto-skip check does not accept; `--skip-generate` prevents it) removes
   every arm's combo markers: their combos were trained on the vectors it
   replaces.
2. Every run removes `results/pca_arms/`.
3. A run that retrains the shared models (no `--skip-shared`) removes the
   left-out arms' outputs built on the shared rows it replaces, their
   `comparison.csv` + `comparison_notes.txt`, `external_transfer/` and
   `paper_tables/` (not their rows or siamese outputs). Every such run names
   every left-out arm that is without tables, whichever submission removed
   them, so also after a submission that stopped part-way. The step-0a NOTE
   names the arms; the final listing prints ready-to-paste jobs wired with
   `--dependency` to this run's: each left-out arm's `run_compare.sh` and
   `run_external_transfer.sh` after this run's shared array, its
   `run_paper_tables.sh` after both (all as
   `_paste_env PCA_ARM=<arm> sbatch ... --export=ALL`),
   then `run_paper_tables.sh --pca_arms_table` over every arm after those and
   after this run's own table across the arms (with one arm, after that arm's
   paper tables). The block's first line defines `_paste_env` (since
   2026-10-21), as
   `_paste_env() { env -u RUN_TOKEN -u CODE_MANIFEST -u SIAMESE_BASELINE_DISCOVER -u SIAMESE_RESULTS_cv1 ... PCA_ARMS="<arms>" PCA_MODALITY_VARIANCE=.. PCA_JOINT_VARIANCE=.. PCA_JOINT_DIM=.. SIAMESE_VARIANTS_LIST=.. USE_CELL_LINES=1 CVS=.. "$@"; }`
   (all 12 `SIAMESE_RESULTS_*` pin names `collect_and_regrade_siamese.sh`
   reads for the four variants and three CVs: a static list, in
   `siamese_sl/slurm/config.conf` `pasted_env_def`, which a new variant or CV
   must extend): no pasted job gets the pasting shell's run token, code manifest or
   compare pins, whether or not this run saw them, and every one gets, as
   `PCA_ARMS`, the arms of the table across the arms (this run's and the
   left-out arms the block rebuilds), this run's PCA targets and this run's
   variants (`SIAMESE_VARIANTS_LIST`, `USE_CELL_LINES`: those whose combos the
   block's PCA check compared, since 2026-10-21; a left-out arm's compare then
   rebuilds their rows only, the others' going to its `stale_siamese/`); an
   arm whose combos were trained at other
   targets gets its own on its lines (`_paste_env PCA_ARM=both PCA_JOINT_DIM= sbatch ...`),
   and an arm whose combos disagree among themselves, which no one set of
   targets fits, is named. A left-out arm with a selection but no combo
   marker (every one, after a regeneration) is given
   `PCA_ARMS="<arms>" bash run_benchmarking_siamese_vs_others.sh --skip-generate --skip-shared`
   instead, to run once this run has finished: its tables, rebuilt now, would
   have no SLeuth rows. That command (in the step-0a NOTE and the final
   listing) and the one-submission alternative below carry, with their
   values, the settings this run took from the calling shell (`SEED`,
   `ARCH_ARGS`, the PCA targets, ...; not `PCA_ARMS` or `ALLOW_LIVE_JOBS`), so
   a fresh shell reruns this run's settings (as does the command a refused
   export prints, since 2026-10-21). That retraining run replaces this run's table across
   the arms with one over the arms it retrains; a table across every arm,
   with its 2x2 effects, needs one submission's selection of every arm
   (`bash run_benchmarking_siamese_vs_others.sh --skip-generate --skip-shared`
   with `PCA_ARMS` unset). Once this run has finished, `--skip-siamese` with
   every arm in `PCA_ARMS` is the one-submission alternative; it retrains the
   shared models once more (`--skip-siamese --skip-shared` is refused). A
   table across the arms lists as stale (no values) an arm whose selection
   another submission made than its first arm's: its `PAPER COMPLETE` needs one
   submission's selection of every arm.
4. A subset that lists the first arm of a pair (`none` or `per_modality`)
   without its partner (`joint` or `both`) rebuilds the pair's
   single-modality runs, which the partner's kept selection ranked. Both
   orchestrators print a WARNING before anything changes, and the partner's
   `tab:solo` belongs to another run until the partner is rerun (with
   `PCA_ARMS` listing it).

So with `--skip-generate --skip-shared` (the subset recipe of the script's
`--help`) and a subset that keeps pairs whole (`none joint`,
`per_modality both`) or lists second arms only (`joint`, `both`), only the
second applies.

`--skip-siamese` leaves siamese's `latest_combo` markers alone, so every
arm's table reuses whichever combos its markers name (the previous siamese
run's), each arm under its reused selection's run token (`best_per_category.json`
`_selection_meta.run_token`, else its owner's `pcavar*/run_token.txt`); arms
whose tokens differ get one each in their sbatch's environment
(`PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ... --export=ALL`, with a WARNING), and the
table across the arms marks those arms' cells stale. Every job of the run
labels each arm, and `paper_tables.py` checks its combos, with the shell's
PCA targets, so the reused combos' targets must be exported as at their
submission. Before anything changes (since 2026-10-21) it compares each
marker-named combo's `config.json` with them: when one set of targets trains
every reused arm's combos but the shell's do not, it refuses and prints the
command with that set (e.g. `PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=150 bash ./run_benchmarking_siamese_vs_others.sh --skip-siamese`);
when the arms were trained at targets no one set reproduces, a WARNING names
the arms whose tables will be PAPER INCOMPLETE and each one's own targets,
and the run goes on. An arm with nothing on disk (no selection, `latest_combo`
marker or single-modality run in its roots) gets one NOTE instead of the
reused-tree WARNING: its tables get no SLeuth row (PAPER INCOMPLETE), the
table across the arms lists it as missing, and leaving it out of `PCA_ARMS`
avoids both. Every arm's comparison
rows, `comparison.csv`, `external_transfer/` and `paper_tables/`, and the
table across the arms, are rebuilt from them; the siamese outputs themselves
(combos, ablations, the cell-line ablation's rows) are kept. A normal run
removes the markers at submission and the combo job rewrites them on success.
In the same way the previous run's SLGNN / NSF4SL / SLMGAE / DegreePrior rows are
moved to `results/stale_shared/` at submission (unless `--skip-shared`), and
the three peers' previous `fold_<k>_<cv>_predictions.npy` are removed (again by
each shared task when it starts), so a task that never runs leaves its row and
its score matrices missing, not stale.

**Folds and uploads.** Step 0 regenerates `folds/` on the login node, writes
its manifest `slurm/logs/folds_<RUN_TOKEN>_<timestamp>.sha256` (one
`<sha256>  <path>` line per fold file, from `folds_manifest` in
`slurm/folds_manifest.sh`; exported to the jobs as `FOLDS_MANIFEST`) and makes
the set read-only (`chmod -R a-w`) until the next step 0; a standalone
`slurm/run_shared_folds.sh` regeneration also makes it writable first
(`chmod -R u+w`) and read-only again after. Under the orchestrator
(`SHARED_FOLDS_FROZEN=1`) the folds job never regenerates: it refuses (FATAL)
only when the fold files differ from `FOLDS_MANIFEST`; when only a fold source
or a `prepare_folds.py` default changed after step 0 but the fold files are
byte-identical to the manifest, it prints a WARNING and continues
(`paper_tables.py` still marks the fold sources stale, so the verdict is
INCOMPLETE). A hand `prepare_folds.py` run needs
`chmod -R u+w sl_comparison/folds` first (on a read-only set it prints that
hint instead of a traceback). The cluster checkout is updated by SFTP upload
from the laptop, not by git; results come back by SFTP download and commits
are made on the laptop. Never upload `sl_comparison/folds/` (step 0
regenerates it on the cluster and it stays read-only between runs), `data/`,
`*/results/` or `*/slurm/logs/`: these only come back by download (the
git-ignored inputs the orchestrator names when one is missing, e.g.
`data/Human_GOsim*.txt` or `data/embeddings_cache/scGPT_human/`, are the
exception: copy exactly those in, once, while no DAG runs). Never upload in a
mirror/delete sync mode. While a DAG is queued or running, upload
nothing if you can avoid it: jobs starting later read whatever is there. The
SLGNN / NSF4SL tasks re-fingerprint `SLGNN/build_synlethkg.py`,
`NSF4SL/build_transe_features.py`, `siamese_sl/gene_name_utils.py` and
`SLMGAE-in-pytorch/gene_corrections_config.py` and refuse any byte change.
Nothing refuses a changed `siamese_sl/cell_line_vocab.py`,
`sl_comparison/prepare_folds.py`, `data/SL_SynLethDB_experimental*` pair file
or `.sources.tsv` sidecar, or `siamese_sl/slurm/*.conf` mid-run (the folds job
only warns about a stale fold source, and later jobs read the new version), so
never upload those either: the run goes on and only `paper_tables.py` flags it,
PAPER INCOMPLETE (stale fold sources, or code drift). On
2026-09-23 a whole-tree upload after the submission replaced `folds/meta.json`
with an older copy, and the folds job refused it and cancelled SLGNN / NSF4SL
/ SLMGAE and DegreePrior. After the run, download `sl_comparison/folds`
together with the results and commit it on the laptop, so the repo's answer
key is the one the run used. The download can carry the read-only modes: run
`chmod -R u+w sl_comparison/folds` on the laptop before committing or any
other git operation there.

**Code and SynLethKG during a run.** At submission the orchestrator writes a
sha256 manifest of the code the jobs execute (`code_manifest.py`: the `*.py`,
`*.sh` and `*.conf` files of `siamese_sl`, `sl_comparison`, `SLGNN`, `NSF4SL`
and `SLMGAE-in-pytorch`, tests and login-node-only scripts excluded) to
`slurm/logs/code_<RUN_TOKEN>_<timestamp>.sha256`, exported as `CODE_MANIFEST`;
`run_paper_tables.sh` passes it to `paper_tables.py --code_manifest`, which
re-hashes those files and lists every one that changed, appeared or vanished
during the run (PAPER INCOMPLETE until checked; on 2026-09-23 an upload changed
code 32 minutes after the submission and nothing recorded it). The
orchestrator also makes `data/synlethkg/` read-only for the run (no job
writes it; it vanished from the cluster during the 2026-09-23 run): rebuild it
by hand only after `chmod -R u+w data/synlethkg`, as
`slurm/check_shared_inputs.py`'s hint says.

Before the first run, build SLGNN's and NSF4SL's inputs on the login node and
copy SLMGAE's two GO similarity files (`data/Human_GOsim.txt`,
`data/Human_GOsim_CC.txt`, ~108 MB each, from the original SLMGAE release;
git-ignored, and nothing in this repo builds them) into `data/`. The
orchestrator refuses to submit the shared models while any of them is missing
(see its `--help`).

**Preflight first (recommended)** — a fast GPU smoke test of all three
shared-fold models on a tiny fold set, to catch shape/device bugs before the
full run:

```bash
sbatch sl_comparison/slurm/preflight.sh                 # PASS/FAIL per model
```

It pins gpu3 like every job script, and no orchestrator submits it, so
`GPU_NODES` does not reach it: `sbatch --nodelist=<node>
sl_comparison/slurm/preflight.sh` runs it elsewhere.

**Manual pieces** (what the orchestrator chains), from the repo root.
`siamese_sl/slurm/reset_env.sh` `rm -rf`'s the venv that `PYTHON_PATH` points
into, which every job below runs from, so the folds are generated (and the
embeddings checked) before it and every job waits on it. A plain
`submit_pipeline.sh` would submit a second, unchained reset (and the embedding
benchmark, which the orchestrator does not run), hence its flags in step 5.
Every `afterok` edge carries `--kill-on-invalid-dep=yes`, as in the
orchestrator: a failed parent then cancels its children instead of leaving them
pending forever, so the `afterany` jobs of steps 6-8b still run and report what
is missing:

```bash
# 1. folds on the login node (PYTHON_PATH as in siamese_sl/slurm/config.conf),
#    BEFORE the reset: siamese reads them and has no ordering edge to the folds job.
#    An orchestrated run leaves folds/ read-only, so make it writable first, and
#    freeze it again for this run as the orchestrator does; after the run,
#    download the folds with the results, chmod -R u+w them on the laptop and
#    commit them there
chmod -R u+w sl_comparison/folds
$PYTHON_PATH sl_comparison/prepare_folds.py --seed 42 --verify
# the fold set this run uses, recorded as step 0 records it (same format)
. sl_comparison/slurm/folds_manifest.sh
export FOLDS_MANIFEST="$PWD/sl_comparison/slurm/logs/folds_manual_$(date +%Y%m%dT%H%M%S).sha256"
folds_manifest > "$FOLDS_MANIFEST"
chmod -R a-w sl_comparison/folds
# as the orchestrator: the folds job (step 4) then refuses a changed fold set
# instead of rewriting it under the siamese jobs (FATAL when the fold files
# differ from $FOLDS_MANIFEST, a WARNING when only a fold source or a
# prepare_folds.py default changed)
export SHARED_FOLDS_FROZEN=1
# 1b. also before the reset (it deletes $PYTHON_PATH): step 3 may be skipped
#     only when the orchestrator's auto-skip test holds, i.e.
#     data/all_genes_kg_complex.pt's gene_order matches data/gene_universe.txt
#     and this exits 0
$PYTHON_PATH siamese_sl/embedding_versions.py --data_dir data
# 2. rebuild the shared venv; everything below chains on it
ENV=$(cd siamese_sl && sbatch --parsable slurm/reset_env.sh)
# 3. embeddings, unless step 1b said skip (then GEN=$ENV)
GEN=$(cd siamese_sl && sbatch --parsable --dependency=afterok:$ENV \
    --kill-on-invalid-dep=yes slurm/run_generate_embeddings.sh)
# 4. verify the folds + DegreePrior control, then the 3 models x 3 CVs array
FOLDS=$(sbatch --parsable --dependency=afterok:$GEN --kill-on-invalid-dep=yes \
    sl_comparison/slurm/run_shared_folds.sh)
SHARED=$(sbatch --parsable --dependency=afterok:$FOLDS --kill-on-invalid-dep=yes \
    --array=0-8 sl_comparison/slurm/run_shared_models.sh)
# 5. siamese best-per-category + combo (+ eval, ablation) of every PCA arm of
#    PCA_ARMS (siamese_sl/slurm/config.conf; default all four), after $GEN,
#    with no reset of its own and no benchmark: one Step-1 array per owner of
#    single-embedding runs, then per arm its combo, eval and ablation. Each
#    arm's combo job writes siamese_sl/results[/pca_<arm>]/<prefix>latest_combo_<cv>.txt.
#    Note each arm's combo, eval and ablation summary job ids and the RUN_TOKEN
#    it prints, and export that token so step 8 checks the ablation dirs'
#    run_token.txt against it (unset, both kinds of dir are still checked
#    against the token the arm's best_per_category.json records; export it
#    only for an ablation rerun under a token of its own). Export the run's
#    PCA_ARMS and PCA_* targets here when they are not the defaults: every
#    step below reads them too.
#    --skip-pair-export: the all-gene-pair exports are submitted in step 9,
#    after the paper tables, instead of right after the siamese jobs.
#    It ends by printing a block to paste that submits 5b-8b for every arm:
#    do not paste it in this DAG, its compare and external-transfer lines
#    wait for none of the 3x3 array ($SHARED), which submit_pipeline.sh does
#    not know; use 5b-8b below
(cd siamese_sl && bash ./slurm/submit_pipeline.sh --skip-reset-env --best-cat-only \
    --skip-pair-export --after $GEN)
export RUN_TOKEN=<printed RUN_TOKEN>
# 5b-8 once per PCA arm A of PCA_ARMS, with that arm's job ids from step 5;
#     every job gets its arm in front of its sbatch (PCA_ARM=$A sbatch ...
#     --export=ALL as the orchestrators submit; without it a job is the arm
#     the shell exports, arm none only where the shell exports none)
A=none
# 5b. post-hoc cell-line ablation (tab:cellline): one job over every per_head
#     variant x CV, from siamese_sl/, after the arm's combo succeeded
CELL=$(cd siamese_sl && PCA_ARM=$A sbatch --parsable --dependency=afterok:<combo job id of A> \
    --kill-on-invalid-dep=yes --export=ALL cell_line_ablation/run_cell_line_ablation.sh)
# 6. collect + re-grade the arm's siamese combos (+ the _nocell rows), then its table
COMPARE=$(PCA_ARM=$A sbatch --parsable --dependency=afterany:$SHARED:<combo job id of A>:$CELL \
    --export=ALL sl_comparison/slurm/run_compare.sh)
# 7. the arm's external-screen tables, after its eval array and the 3x3 array
EXT=$(PCA_ARM=$A sbatch --parsable --dependency=afterany:$SHARED:<eval job id of A> \
    --export=ALL sl_comparison/slurm/run_external_transfer.sh)
# 8. every paper table / number of the arm
PT=$(PCA_ARM=$A sbatch --parsable --dependency=afterany:$COMPARE:$EXT:$CELL:<ablation summary job id of A>:<eval job id of A> \
    --export=ALL sl_comparison/slurm/run_paper_tables.sh)
# 8b. with two arms or more, once, after every arm's step 8: the table across the
#     arms -> sl_comparison/results/pca_arms/ (submitted as the first arm, as the
#     orchestrator does; in this mode the job's own arm plays no part)
PT_ARMS=$(PCA_ARM=none sbatch --parsable --dependency=afterany:<every arm's PT> \
    --export=ALL sl_comparison/slurm/run_paper_tables.sh --pca_arms_table)
# 9. optional, last, once per arm A: SLeuth's predictions for all gene pairs (no
#    table reads them); with one arm, afterany on its $PT
(cd siamese_sl && PCA_ARM=$A sbatch --dependency=afterany:$PT_ARMS,singleton --export=ALL slurm/export_pair_predictions.sh)
```

**After a combo job submitted by hand.** A direct `sbatch` of
`siamese_sl/slurm/run_best_per_category_combo.sh` is how a failed combo is
retrained without the Step-1 array, one PCA arm per job (from `siamese_sl/`:
`RUN_TOKEN=<the arm's token> PCA_ARM=<arm> PCA_ARMS="<the run's arms>" SELECTION_CV=<the run's> sbatch --export=ALL slurm/run_best_per_category_combo.sh`,
the form the combo job's own stop messages print (`SELECTION_CV` since
2026-10-21: it is in no signature, so a job without the run's selects on the
submitting shell's, else cv1, exits 0 under the run's token, and only the
table across the arms lists the arm as stale),
with `SIAMESE_VARIANTS_LIST="<variant> ..."` in front to limit it to those
variants; every CV of each is retrained, in that arm only; resubmit it per arm
whose combo failed). Submit it only once none of the
jobs that read or write what it removes is queued or running
(`squeue -u $USER -o '%i %j %T'`: `allmod_combo`, `bilinear_probe`,
`eval_finetuning`, `ablation`,
`ablation_summary`, `cellline_ablation`, `pair_export`, `sl_final_compare`,
`sl_ext_transfer`, `sl_paper_tables`; the job names carry no arm, so this
holds for every arm of the run; wait for the rest of the run or
`scancel` them). The job does not check, where both orchestrators refuse
these removals while such a job is alive: as it starts it removes the arm's
selection file (until its Step 2 has rewritten it, an ablation task of the
arm, of ANY variant, that starts fails) and its variants' markers and combos
in the arm (their running eval tasks fail), and after Step 2 the outputs
listed below, under the ablation tasks that are writing them. A job submitted
by hand takes its settings from the shell that submits it: if the run was
submitted with settings exported (each orchestrator's submission output names
each one in a NOTE line, `submit_pipeline.sh`'s since 2026-10-21), export the
same ones again before any hand submission of this section. `PCA_ARMS`, the
PCA targets and `SELECTION_CV` are among them: `PCA_ARMS` decides which root
holds the arm's single-embedding runs (the combo of arm `joint` ranks arm
`none`'s when both are listed, its own when `PCA_ARMS=joint`), and
`SELECTION_CV` the CV a combo job ranks them on, which no check of the arm
compares (only the table across the arms lists an arm selected on another CV
as stale).
The new combos take the SAME
directory names, so nothing downstream could tell them from the ones they
replace. Once its selection has succeeded, the job therefore removes, as
step 5 does at submission, what was computed from the replaced combos, in the
arm's namespaces only: their `eval_finetuning_<cv>/` and `pair_predictions/`
(with the old directories), the listed variants' comparison rows and
`*_selected_<cv>.txt` markers (moved to the arm's
`results[/pca_<arm>]/stale_siamese/`), their `*_nocell` rows, the arm's
`siamese_sl/results[/pca_<arm>]/<prefix>cellline_ablation_<cv>/` and
`<prefix>ablate_no_*` dirs, and the joint outputs (`ablation_summary.csv`,
the arm's `comparison.csv` and its notes, `external_transfer/` and
`paper_tables/`), and the table across the arms, `results/pca_arms/`, which
compares every arm's combos. A job
submitted by hand queues nothing that rebuilds them. So once it has FINISHED,
rerun all of this for its arm, from the repo root:

```bash
# the combo job's arm, and its token: <arm root>/best_per_category.json,
# _selection_meta.run_token (<arm root> = siamese_sl/results for arm none,
# siamese_sl/results/pca_<arm> otherwise); every sbatch below hands both to
# its job (--export=ALL), the table across the arms with PCA_ARM=none in front
export PCA_ARM=<arm>
export RUN_TOKEN=<the arm's token>
# optional: the code manifest written when the run was submitted (the oldest
# of its token; a submit_pipeline.sh run has none), so that paper_tables also
# lists the code changed since (PAPER INCOMPLETE until checked)
export CODE_MANIFEST="$(ls "$PWD"/sl_comparison/slurm/logs/code_"$RUN_TOKEN"_*.sha256 2>/dev/null | sort | head -n 1)"
# a. external-screen eval of every variant's new cv3 combo of the arm (3 splits x 4 variants)
EVAL=$(cd siamese_sl && sbatch --parsable --array=0-11 --export=ALL slurm/eval_finetuning.sh)
# b. the leave-one-category-out ablation against the arm's new combos (it joins
#    the arm's selection token and removes the arm's previous ablate_no_* dirs
#    at submission, after it has checked the combos' config.json against this
#    shell's PCA targets: a mismatch is refused, with the command to submit
#    instead, or one per group of the arm's variants where they were trained at
#    different targets, and then the ablation_summary.csv rebuild);
#    note the "Ablation summary" job id it prints
(cd siamese_sl && PCA_ARMS=$PCA_ARM bash ./slurm/submit_pipeline.sh --ablation-only)
# c. as steps 5b-9: the cell-line ablation, THEN compare (it tabulates the
#    *_nocell rows that job rewrites); the external transfer after the eval
CELL=$(cd siamese_sl && sbatch --parsable --export=ALL cell_line_ablation/run_cell_line_ablation.sh)
COMPARE=$(sbatch --parsable --dependency=afterany:$CELL --export=ALL sl_comparison/slurm/run_compare.sh)
EXT=$(sbatch --parsable --dependency=afterany:$EVAL --export=ALL sl_comparison/slurm/run_external_transfer.sh)
PT=$(sbatch --parsable --dependency=afterany:$COMPARE:$EXT:$CELL:<ablation summary job id>:$EVAL \
    --export=ALL sl_comparison/slurm/run_paper_tables.sh)
# d. the table across the arms, which the combo job removed (with two arms or
#    more in the run's PCA_ARMS; it reads every arm's paper tables), then the
#    arm's all-gene-pair export, last (with one arm: afterany:$PT)
PT_ARMS=$(PCA_ARM=none sbatch --parsable --dependency=afterany:$PT \
    --export=ALL sl_comparison/slurm/run_paper_tables.sh --pca_arms_table)
(cd siamese_sl && sbatch --dependency=afterany:$PT_ARMS,singleton --export=ALL slurm/export_pair_predictions.sh)
```

When one variant was retrained, a can be limited to it
(`EVAL_VARIANT=<variant>` in front of `sbatch`, with `--array=0-2`) and b to
the retrained variants (`SIAMESE_VARIANTS_LIST="<variant> ..."` in front of
`bash`; the arm's `ablation_summary.csv` then holds those only, and no table
reads it: once b's two jobs have finished,
`RUN_TOKEN=<the arm's token> PCA_ARM=<arm> sbatch --export=ALL slurm/run_ablation_summary.sh`
from `siamese_sl/` writes it again for every variant; without the token every
ablation row is a MISMATCH with no delta). Put them in front of
that one command and never `export` them: compare, the external transfer and
the tables read the variant list too and would leave the other variants out.
Three more hand reruns need a
follow-up, with the arm's `PCA_ARM`: after an eval task, rerun
`run_external_transfer.sh` and the tables, the arm's and then the table across
the arms (`paper_tables.py` lists a SLeuth row of the external tables that is
no longer what the eval on disk reports); after a `SELECTION_CV` Step-1 task,
rerun the combo job of each arm the owner's runs serve (both arms of the pair;
its Step 2 alone rewrites `best_per_category.json`), and then all of the above
for each (a selection the single-modality runs on disk no longer give is
listed); after one peer's cv3 task, rerun `run_external_transfer.sh` and the
tables of every arm, and the table across the arms (the peers' rows are
shared by every arm, and their external rows are tied to nothing, so this one
is NOT listed).

`paper_tables.py` tabulates a single-modality or ablation directory only
when it is this run's: `run_token.txt` equal to the expected token, the
cell's kernel / cell_lines / CV, a fold stamp matching the current folds,
the variant combo's `arch_signature` (minus the post-concat PCA setting for
single modalities, whose per-modality PCA must be the combo's: no arm applies
the PCA of a concatenation to one modality; the `gene_term` / `pair_features`
settings are part of it) and the same content of every embedding file it
shares with that combo (both `results.json` `embeddings_sha256`: a file
rebuilt in place between the two trainings differs). The expected token is,
for a single-modality dir, `--solo_run_token`, else `best_per_category.json`
`_selection_meta.run_token` (the Step-1 array the selection ranked; checked
even without `$RUN_TOKEN`), else `--run_token` / `$RUN_TOKEN`; for an
ablation dir, `--ablation_run_token` (CLI only; `$ABLATION_RUN_TOKEN` is
ignored), else `--run_token` / `$RUN_TOKEN`, else that same
`_selection_meta.run_token` (every orchestrator path ablates under the
selection's token, so a dir a cancelled task left from an earlier run is
marked stale rather than tabulated as this run's); `''` leaves that check off
(`paper_tables/README.md` states which token each kind was checked against).
Any other directory is reported as stale in `paper_tables/README.md`, never
tabulated. So a standalone run keeps both token checks even with no
`$RUN_TOKEN`; pass `--ablation_run_token` for an ablation rerun under a token
of its own, or `--ablation_run_token ''` to turn that check off. A comparison row
counts only when `evaluate_model.py` wrote it (a `results/<row>_<cv>.json`
without its `provenance`, e.g. a
`collect_siamese.py` row an interrupted compare never re-graded, is ignored)
and its recorded graded-on and trained-on fold fingerprints equal the current
`folds/`; every other row is left out and listed
(`numbers.json` `items.fold_provenance`). The orchestrator removes the
previous `comparison.csv` / `comparison_notes.txt` at submission, so a
compare job that dies early leaves no table rather than an old one.

Each arm's tables read that arm's files only: its selection, markers, combos,
ablation and cell-line ablation in its siamese root; its SLeuth and `_nocell`
rows, `*_selected_<cv>.txt` markers, `comparison.csv` and `external_transfer/`
in its comparison root; the shared models' rows in `results/`; and the
single-modality runs where its selection's `_selection_meta.step1_root` says
(its pair's owner root). A combo that is not a direct child of the arm's
siamese root, or whose `config.json` trains another arm's PCA
(`arch_signature.pca_arm`: another namespace, a hand copy), is listed and not
used, and so are a `comparison.csv` and an `external_transfer/` CSV whose
`pca_arm` column names another arm (an `external_transfer/` CSV without the
column, written before 2026-10-21, is read);
the cell-line ablation's recorded source combo is compared as a path, so arm
`none`'s and arm X's combos of one name are told apart. The table across the
arms (`--pca_arms_table`) tabulates a cell only when its combo is that arm's,
its `arch_signature` equals the reference arm's combo outside the four PCA
keys (`pca_variance`, `pca_dims`, `post_pca`, `post_pca_dim`), its
selection's `run_token` is the reference arm's (one submission), so is its
selection's `_selection_meta.selection_cv` (one selection protocol, since
2026-10-21: `SELECTION_CV` is in no signature; an unrecorded CV differs from a
recorded one), every embedding file it shares with the reference combo was trained
on the same content, and its graded row is usable (as `tab_tiers`'s);
otherwise the cell is missing or stale, listed, and gets no value (and its
variant and tier get no `best_arm` and no 2x2 effects).

To rebuild a PCA arm's table locally, run the same collect + re-grade path as
`run_compare.sh`. It needs on disk: `sl_comparison/folds/`, the shared models'
rows (in `results/`), and the arm's `siamese_sl/results[/pca_<arm>]/<prefix>latest_combo_<cv>.txt`
markers together with the combo directories they name (`results.json`,
`model_params.json`, whose `fold_stamp` the re-grade checks against
`sl_comparison/folds/` and refuses to grade without, and
`fold_*_<cv>_predictions.npy`; the checkpoints are not needed), plus the
arm's `best_per_category.json`, without which the rows carry no
modality-selection record. It first moves every existing 2x2 siamese row of
the arm to its `stale_siamese/` and then writes rows only for the combos those
markers name, so without the markers it publishes no siamese row at all
(other `siamese_sl_*` rows, e.g. the cell-line ablation's
`siamese_sl_nocell_<cv>.json`, are left in place). Arm `none` needs no arm
setting; any other arm takes its roots from `slurm/pca_arm.sh` (with
`PCA_ARM` exported, which `bash -c` sees as an environment variable):

```bash
# arm none: siamese_sl/results/ -> sl_comparison/results/
CVS="cv1 cv2 cv3" PYTHON_PATH=python3 \
    bash -c '. sl_comparison/slurm/collect_and_regrade_siamese.sh'
python3 sl_comparison/compare.py --results_dir sl_comparison/results \
    --csv sl_comparison/results/comparison.csv --pca_arm none
# arm joint: siamese_sl/results/pca_joint/ -> sl_comparison/results/pca_joint/,
# next to the shared models' rows in sl_comparison/results/
PCA_ARM=joint CVS="cv1 cv2 cv3" PYTHON_PATH=python3 \
    bash -c '. sl_comparison/slurm/pca_arm.sh && . sl_comparison/slurm/collect_and_regrade_siamese.sh'
python3 sl_comparison/compare.py --results_dir sl_comparison/results/pca_joint \
    --shared_results_dir sl_comparison/results \
    --csv sl_comparison/results/pca_joint/comparison.csv --pca_arm joint
```

Do not run `collect_siamese.py --discover_dir ...` on its own for the table:
it skips the shared re-grade (the row keeps siamese's self-scored numbers,
whose retrieval columns rank against its own embedding universe, not the
shared 8,844 genes), and without `--combo_markers`
it ranks every pd:per_head run under the `--discover_dir` root (single
embeddings, ablations, earlier runs' combos; never a run under another arm's
root, `pca_<arm>/` or `pca_arms/`) instead of this run's combo.

## SLURM scripts

| Script | Role |
|--------|------|
| `../run_benchmarking_siamese_vs_others.sh` | **root orchestrator** — submits the whole DAG (gpu3, or the node(s) `GPU_NODES` names) |
| `slurm/preflight.sh` | GPU smoke test on a tiny fold set (standalone: `GPU_NODES` does not reach it, pass `--nodelist`) |
| `slurm/run_shared_folds.sh` | verify/regenerate the shared folds + DegreePrior (CPU; before the array) |
| `slurm/run_shared_models.sh` | array worker: train + score one (model, CV) per task (`--array=0-8`, `%SHARED_CONCURRENT` only when that is a positive integer; afterok on the folds job) |
| `../ELISL/slurm/run_elisl.sh` | ELISL train + score, one CV per array task (`--array=0-2`; the root orchestrator submits it with the shared models since 2026-11-13, `--skip-elisl` to leave it out; gpu3, its GPU idle: LightGBM is CPU-only); `../ELISL/slurm/build_elisl_inputs.sh` builds its SeqVec / node2vec inputs (the root submits it when they are missing or stale) |
| `slurm/run_compare.sh` | per PCA arm (`PCA_ARM`): collect + re-grade this run's siamese combos of the arm, print the arm's final table (the shared models' rows from `results/`) and which `<row>_<cv>.json` rows are present / ABSENT (CPU only: no `--gres`) |
| `slurm/run_external_transfer.sh` | per PCA arm: cross-model external-screen tables (after the arm's eval array and the 3×3 array) |
| `slurm/run_paper_tables.sh` | per PCA arm: paper tables and numbers (afterany, so partial runs still report what is missing); `--pca_arms_table`: the table across the arms, once, after every arm's |

## Upstream inputs kept as shipped

SLGNN, NSF4SL and SLMGAE run on the inputs their authors ship, unchanged: the
comparison runs each method as published. Two of those inputs carry
information derived from SL labels. Both are kept on purpose and disclosed
here and in the `comparison_notes.txt` that `compare.py` writes next to the
table; neither is corrected.

- **SLGNN: SynLethKG `kg2id.txt`.** The upstream KG (SLGNN repository, pinned
  commit `c56e23fbd20e`) leaves out not only the SL / SR / non-SL label
  relations but also every ordinary gene-gene edge (INTERACTS, REGULATES,
  COVARIES) between two genes SynLethKG labels SL: all 1,251 non-label
  triples absent from `kg2id.txt` (`nonlabel_kg_triplet_rows_absent_from_kg2id`
  in `data/synlethkg/meta.json`) lie on SL-labelled pairs. On the shared folds
  a direct KG edge is therefore rarer between positives (83 of 22,166 pairs,
  0.37%) than between negatives (2,542 of 106,704, 2.38%); with those edges
  kept, positives would be at about 2.6%. In SLGNN's input, a direct KG edge
  between a pair correlates with the pair being non-SL, for validation and
  test pairs as well as training pairs.
- **SLMGAE: support-view coverage.** Its GO-BP, GO-CC and PPI views exist only
  for the 6,375 genes of `data/List_Proteins_in_SL.txt`, which is exactly the
  gene set of the SynLethDB v1 SL pairs (`data/SL_Human_Approved.txt`).
  4,587 of the 8,844 shared genes get a view row; the rest get an all-zero
  row. 73.1% of the genes that appear in any positive pair have a view row
  (3,845 of 5,257), against 20.7% of the genes that appear only in negatives
  (742 of 3,587), so view coverage alone is predictive: scoring each test
  pair by how many of its two genes have a view row gives AUROC 0.60-0.74 per
  cv3 fold and 0.67-0.72 on cv2, and the both-genes indicator gives 0.60 and
  0.69 on cv3 folds 0 and 1 (measured with `build_support_views` on the
  current folds, 2026-09-21). A gene with no training positive and no view
  row gets an exactly-zero embedding in every branch; that is where SLMGAE's
  constant score rows (`degen-rows`) come from.

## Notes

- **Feature inputs** (each model as intended): SLGNN runs the original model on
  SynLethKG (`SLGNN/build_synlethkg.py` -> `data/synlethkg/`: the upstream
  `kg2id.txt`, which ships without the SL, synthetic-rescue and non-SL
  relations and without the other gene-gene edges on SL pairs; see "Upstream
  inputs kept as shipped"), NSF4SL the original model on
  the SynLethKG TransE entity embeddings its repo ships
  (`NSF4SL/build_transe_features.py --download` ->
  `data/nsf4sl_transe_features.pt`); build both once on the login node. SLMGAE
  uses its native SL-graph + GO/PPI similarity views (List_Proteins-indexed,
  remapped into the shared index); siamese uses its own
  selected embedding. The earlier `kg_complex` substitute for SLGNN/NSF4SL is
  opt-in (`--arch port` / `--features embedding`). The comparison controls the *data split*, not the inputs —
  it compares methods, like the SL_benchmark paper.
- **SLMGAE memory:** at 8,844 nodes its `num_support × N²` attention parameter
  (3 × 8,844² = 234,649,008 float32 values) is ~0.94 GB (≈3.75 GB with its
  gradient and Adam m/v state; the N²-sized forward/backward activations come
  on top) — heavy but expected to fit on the 64 GB GPUs. If it OOMs, do not
  shrink SLMGAE alone: a smaller universe or fewer negatives means new shared
  folds for every model, and a lighter attention is no longer the authors'
  SLMGAE.
- `folds/` is tracked in git: it is the answer key, so commit the set a run
  used (see "Folds and uploads"). The per-fold score matrices are regenerable
  artifacts (seed 42); those large `.npy` matrices (~313 MB each at 8,844²)
  are not meant to be committed.
