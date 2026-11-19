#!/bin/bash
# ============================================================================
# run_benchmarking_siamese_vs_others.sh — ROOT pipeline orchestrator
# ============================================================================
# The paper run in ONE command: submits the whole SL benchmark on the SLDB 3.0
# standard as one SLURM DAG (SLeuth = siamese_sl, every SIAMESE_VARIANTS entry,
# vs SLMGAE / SLGNN / NSF4SL / ELISL / DegreePrior on the shared folds), once per PCA
# arm of PCA_ARMS (below), and ends with every paper table and number the
# pipeline can compute for each arm, in sl_comparison/results/paper_tables/
# (arm none) and sl_comparison/results/pca_<arm>/paper_tables/, and with one
# table across the arms in sl_comparison/results/pca_arms/. Run from the login
# node (repo root); it submits jobs with `sbatch --parsable` + `--dependency`,
# like siamese_sl/slurm/submit_pipeline.sh (whose conventions this follows).
#
#   login node: code / input / provenance checks, shared folds (step 0)
#   reset_env ── generate_embeddings ──┬─ per owner: siamese best-per-category array (bestcat_n_tasks)
#                                      │   └─ per arm: combo: select ONCE, then variants x cv1-3
#                                      │       ├─ eval_finetuning   (cv1-3 x variants)
#                                      │       ├─ ablation          (categories x CVs x variants)
#                                      │       │   └─ ablation summary
#                                      │       └─ cell-line ablation (per_head variants x CVs)
#                                      └─ shared folds job ─┬─ 3x3 array (SLGNN/NSF4SL/SLMGAE x cv1-3)
#   reset_env ── [elisl_inputs, only when data/elisl is stale] ─┴─ ELISL array (cv1-3)
#   per arm:
#   final compare      afterany: its combo, cell-line ablation, shared array, ELISL
#   external transfer  afterany: its every eval task, shared array
#   paper tables       afterany: its compare, external transfer, cell-line
#                      ablation, ablation summary, eval; ONE job: the tables,
#                      then the bootstrap CIs (sl_comparison/bootstrap_ci.py
#                      -> <cmp>/bootstrap/), then the tables again with the
#                      CI columns (PT_BOOTSTRAP=0: the tables only)
#   once, with 2 arms or more:
#   PCA arms table     afterany: every arm's paper tables (the last job a table
#                      waits for)
#   per arm:
#   pair export        afterany: every job of its arm and the PCA arms table
#                      (so every job above), and singleton: never beside
#                      another pair_export job of yours (the LAST jobs; no table
#                      or number reads them): predicted SL probability of ALL
#                      gene pairs, one matrix per cell-line head, and the
#                      fine-tuned models' predicted Adamson value, for each
#                      variant's cv3 combo
#
# PCA arms: PCA_ARMS (siamese_sl/slurm/config.conf; default "none joint"
# since 2026-11-13, any subset of none joint per_modality both, always run in that order) are the arms this
# DAG runs, each the WHOLE siamese pipeline and its own comparison, external
# transfer and paper tables, under its own roots:
#   none          no PCA                       siamese_sl/results/
#                                              sl_comparison/results/
#   joint         PCA on the concatenation of  siamese_sl/results/pca_joint/
#                 all modalities               sl_comparison/results/pca_joint/
#   per_modality  PCA within each modality     .../pca_per_modality/
#   both          per_modality, then joint     .../pca_both/
# Every file name inside an arm's roots is the one this help gives for arm
# none's. The fold set and the SLGNN / NSF4SL / SLMGAE / DegreePrior rows (in
# sl_comparison/results/, with their array) are shared: built once per run
# (--skip-shared reuses the previous run's) and graded and tabulated against
# every arm of the run; an arm the run leaves out loses its tables built on
# the rows the run retrains (see "Previous-run outputs"). none and joint rank
# the same single-embedding
# runs (native dimension), per_modality and both theirs (PCA within the
# modality): one Step-1 array per such pair, in the root of its OWNER, the
# pair's first arm that PCA_ARMS lists (siamese_sl/results/ and
# siamese_sl/results/pca_per_modality/ by default), and each arm's combo waits
# for its owner's array. Each arm's jobs are submitted as
# `PCA_ARM=<arm> sbatch ... --export=ALL` (the arm in sbatch's own
# environment, never in an --export list) under their usual job names, and
# log to <stage>_pca_<arm>_<job id>.out (arm none: <stage>_<job id>.out) in
# siamese_sl/slurm/logs/ and sl_comparison/slurm/logs/. The PCA arms table
# (sl_comparison/slurm/run_paper_tables.sh --pca_arms_table; its logs
# paper_tables_pca_arms_<job id>.out) sets each arm's combos and graded rows
# against the first arm's (none when it runs) and lists every arm's paper
# tables with its verdict.
#
# Every afterok edge carries --kill-on-invalid-dep=yes: when a parent fails,
# its afterok children are cancelled instead of pending forever, so every
# afterany consumer (compare, external transfer, paper tables) still runs and
# reports what is missing.
#
# All models use the ONE canonical fold set (sl_comparison/folds), generated
# on the login node in step 0, before reset_env is submitted. siamese reads
# the same folds (SHARED_FOLDS in siamese_sl/slurm/config.conf) and runs its
# own embedding selection + cell-line pipeline on them. run_compare.sh
# collects each siamese variant's combo from THIS run (its latest_combo
# markers) and re-grades it with the shared evaluator
# (sl_comparison/slurm/collect_and_regrade_siamese.sh); a row with no
# successful re-grade is moved aside, not tabulated. All models are graded by
# ONE evaluator on AUPRG / AP and AUROC.
#
# Outputs (repo-relative; for PCA arm X the same names under
# siamese_sl/results/pca_X/ and sl_comparison/results/pca_X/):
#   sl_comparison/results/comparison.csv       every model x CV (tab:tiers), one siamese
#                                              row per variant, the *_nocell rows
#   sl_comparison/results/external_transfer/   external screens, identical rows
#   siamese_sl/results/ablation_summary.csv    leave-one-category-out
#   sl_comparison/results/paper_tables/        one CSV per paper table, numbers.json,
#                                              README.md (table/number -> file/column,
#                                              and every missing input; its line 3 is
#                                              the arm's verdict)
#   siamese_sl/results/<cv3 combo>/eval_finetuning_<cv>/
#                                              per screen, next to the metric CSVs:
#                                              <Screen>_sl_probabilities.csv (zero-shot
#                                              SL probability of every screen pair for
#                                              every cell-line head),
#                                              <Screen>_finetuned_allpairs_<mode>.csv (the
#                                              fine-tuned predicted screen value of every
#                                              pair) and finetuned_models/
#   siamese_sl/results/<cv3 combo>/pair_predictions/
#                                              sl_prob_<HEAD>.npy: predicted SL probability
#                                              of ALL gene pairs, one matrix per cell-line
#                                              head plus ANY (pooled variants: ANY only);
#                                              finetuned_Adamson_<mode>_cv1.npy: predicted
#                                              Gamma of all gene pairs; manifest.json
# and once, with 2 arms or more:
#   sl_comparison/results/pca_arms/            tab_pca.csv (each arm against the
#                                              first), numbers.json, README.md (the
#                                              arms' paper tables and verdicts; its
#                                              line 3 the verdict over every arm)
#
# Prerequisites: run from the repo root, on a node with sbatch and squeue (it
# refuses, before changing anything, when they are missing or slurmctld does
# not answer). GPU jobs land on gpu3 (each script's `#SBATCH --nodelist=gpu3`;
# GPU_NODES moves them, see Env). The code every job imports must be in this
# checkout: a file never uploaded (or untracked, on a git-synced copy) is
# missing here; the script refuses to submit when one is missing. It refuses
# likewise when one of these git-ignored inputs is missing: the external
# screens `external data/pair_sgRNA_phenotypes.txt` (Adamson) and
# `external data/mmc5.xlsx` (Gilbert) unless --skip-eval, and, when this run
# generates, the scGPT checkpoint folder data/embeddings_cache/scGPT_human/
# (best_model.pt + vocab.json; never downloaded automatically, the refusal
# prints a manual gdown command). Geneformer is built from
# HuggingFace, so the generation node needs network access or an HF cache.
# It also refuses while jobs of an earlier submission (this script's,
# siamese_sl/slurm/submit_pipeline.sh's, or a standalone SLGNN / NSF4SL baseline
# job) are queued or running, other than --after's JOB: they run from the venv
# reset_env deletes and read the folds step 0 rewrites. Wait for them or scancel
# them (ALLOW_LIVE_JOBS=1 submits anyway). The job names carry no PCA arm, so
# another arm's jobs count too.
#
# Generation is FORCED by default (since 2026-11-13, by request): a run with
# neither --skip-generate nor --after (nor --skip-siamese) always regenerates
# the embeddings, as --force-generate does. --auto-generate restores the
# check below.
# With --auto-generate, generation is AUTO-SKIPPED only when data/all_genes_kg_complex.pt's
# gene_order matches data/gene_universe.txt AND
# `siamese_sl/embedding_versions.py --data_dir data` exits 0 (every catalog
# embedding outside run_generate_embeddings.conf OPTIONAL_TYPES is present,
# and every present one carries the current generator's provenance); anything
# else regenerates. So a plain resubmit after a cancel skips the flaky
# ESM/NCBI/CUDA step only when that is safe. reset_env still runs and gates
# the DAG. --force-generate always submits generation, which fails (and
# cancels the DAG) unless every catalog file it leaves is a current build by
# the same embedding_versions.py check. --skip-siamese never generates (the
# shared models do not read the embeddings).
#
# Usage:
#   bash run_benchmarking_siamese_vs_others.sh                 # the paper run, from scratch
#                                     # (regenerates the embeddings: the default since 2026-11-13)
#   bash run_benchmarking_siamese_vs_others.sh --auto-generate # full pipeline (auto-skips gen, see above)
#   PCA_ARMS=joint bash run_benchmarking_siamese_vs_others.sh --skip-generate --skip-shared
#                                     # one PCA arm (or any subset): its roots only, the other
#                                     # arms' outputs left as they are (alone, joint trains its
#                                     # own single-embedding runs). Both flags are part of the
#                                     # recipe: a run that regenerates the embeddings removes
#                                     # EVERY arm's combo markers (their combos were trained on
#                                     # the files it replaces), and one that retrains the shared
#                                     # models removes the other arms' comparison tables,
#                                     # external transfer and paper tables (the final listing
#                                     # prints the jobs that rebuild them, wired to its own)
#   bash run_benchmarking_siamese_vs_others.sh --skip-reset-env
#   bash run_benchmarking_siamese_vs_others.sh --skip-generate # force-skip generation
#   bash run_benchmarking_siamese_vs_others.sh --after <JOB>   # chain after a running gen job (the
#                                     # env reset waits for JOB, since it deletes the venv JOB
#                                     # runs from; add --skip-reset-env to keep that venv). With
#                                     # a variant or arm subset, first remove the OTHER variants'
#                                     # and arms' siamese_sl/results[/pca_<arm>]/
#                                     # *latest_combo_cv[123].txt by hand: JOB replaces the
#                                     # embeddings their combos were trained on, and --after
#                                     # keeps those markers
#   bash run_benchmarking_siamese_vs_others.sh --skip-siamese  # only the 3 shared models; every arm's
#                                     # tables reuse the siamese combos its existing latest_combo
#                                     # markers name (no embedding generation: nothing of this
#                                     # run reads it). Export the PCA targets the reused combos
#                                     # were trained with: when their config.json records others
#                                     # than this shell's, the run is refused before anything
#                                     # changes, with the targets that train every reused arm's
#                                     # combos, or, when no one set does, it WARNs naming the
#                                     # arms whose tables will be PAPER INCOMPLETE. An arm with
#                                     # nothing on disk gets a NOTE (its tables have no SLeuth row)
#   bash run_benchmarking_siamese_vs_others.sh --skip-shared   # only siamese
#                                     # (both: once the earlier submission's jobs have finished,
#                                     # see Prerequisites)
#   bash run_benchmarking_siamese_vs_others.sh --skip-eval     # no external-screen (Adamson/Corn/Gilbert) eval,
#                                     # so no external_transfer/ tables either (built from it)
#   bash run_benchmarking_siamese_vs_others.sh --skip-ablation # no leave-one-category-out ablation
#   bash run_benchmarking_siamese_vs_others.sh --skip-cellline-ablation  # no post-hoc cell_emb=0 rows
#   bash run_benchmarking_siamese_vs_others.sh --skip-external-transfer  # no external_transfer/ tables
#   bash run_benchmarking_siamese_vs_others.sh --skip-paper-tables       # no paper_tables/ jobs, nor
#                                     # the PCA arms table
#   bash run_benchmarking_siamese_vs_others.sh --skip-elisl              # no ELISL job (inputs or
#                                     # training); its previous rows are kept. ELISL's inputs
#                                     # are rebuilt only when check_shared_inputs.py --models
#                                     # ELISL fails: the downloads on this node, then a GPU job
#                                     # that needs SEQVEC_BIN (default <home>/
#                                     # seqvec_env/bin/seqvec; ELISL/README.md), checked here
#   bash run_benchmarking_siamese_vs_others.sh --skip-pair-export        # no all-gene-pair export
#                                     # (eval, ablation, cell-line ablation and the pair
#                                     # export hang off each arm's siamese combo; none
#                                     # changes a tab:tiers number)
# Every --skip-* flag acts on every arm.
#
# Previous-run outputs: whatever this DAG rebuilds is removed (rows moved into
# a stale_*/ beside them) at submission, so a stage that fails or never starts
# leaves NO output rather than an earlier run's. For every arm of PCA_ARMS, in
# its own roots: the siamese latest_combo markers, best_per_category.json, the
# ablation and its summary and the cell-line ablation (with siamese; a previous
# combo's eval_finetuning_<cv>/ and pair_predictions/ go with its directory
# when the combo job replaces it), the single-modality runs of every variant
# (in the owners' roots only), and the SLeuth comparison rows and their
# *_selected_* markers (into that arm's sl_comparison/results[/pca_<arm>]/
# stale_siamese/), comparison.csv (+ notes), external_transfer/ and
# paper_tables/ (always; each tables job also replaces the arm's bootstrap/);
# once: the shared rows, ELISL's included unless --skip-elisl (into
# sl_comparison/results/stale_shared/) and the four peers'
# fold_<k>_<cv>_predictions.npy (with the shared models), and sl_comparison/results/pca_arms/ (always). An arm
# PCA_ARMS leaves out keeps its siamese outputs and its SLeuth rows, but loses
# its combo markers when this run regenerates the embeddings (its combos were
# trained on the old ones), and its comparison.csv (+ notes), external_transfer/
# and paper_tables/ when this run retrains the shared models (they compare it
# with the rows this run replaces). Every such run names each left-out arm
# without tables, whichever submission removed them: the final listing prints
# the jobs that rebuild its tables, wired to this run's shared models and to
# its last table job (its table across the arms, which rewrites pca_arms/ over
# PCA_ARMS only), then the table across every arm (pasted jobs keep the
# pasting shell's RUN_TOKEN, CODE_MANIFEST and compare pins out, and get this
# run's PCA targets, a left-out arm its own where its combos were trained at
# others); or, for an arm without a combo marker (after a regeneration), the
# command that retrains it once this run has finished (that run replaces this
# run's table across the arms with one over the arms it retrains). That
# command, and the one-submission alternative of the pasted jobs, carry the
# settings this run took from the calling shell (SEED, ARCH_ARGS, the PCA
# targets, ...) and pin its arms, PCA targets, variants and USE_CELL_LINES; the
# pasted jobs get, as PCA_ARMS, the arms of the table across the arms (this
# run's and the left-out arms the block rebuilds, not an arm given the
# retraining command), this run's PCA targets (a left-out arm its own where its
# combos were trained at others), variants, USE_CELL_LINES and CVS
# (_paste_env), the left-out arms' other settings being another submission's.
# A table across the arms lists an
# arm another submission selected than its first arm's as stale, so one with
# every arm's 2x2 effects needs one submission of every arm. The owner of a
# pair's single-embedding runs rebuilds them whatever its partner arm's
# selection ranked: a partner left out that ranked them gets a WARNING (its
# tab:solo is another run's until it is rerun).
# Step 0 needs data/SL_SynLethDB_experimental{,_negatives}.{txt,sources.tsv}
# on this node (the negatives .txt is git-ignored); the script names the
# download command when one is missing.
#
# Env: MAX_CONCURRENT / EVAL_CONCURRENT / SHARED_CONCURRENT cap the best-cat +
# ablation, eval and shared-model arrays, each array on its own (every owner
# submits its own best-cat array and every arm its own ablation and eval
# arrays, so with all four arms up to 4 x (MAX_CONCURRENT + EVAL_CONCURRENT)
# tasks run beside each other); all EMPTY by default, so SLURM runs as many
# tasks at once as the partition allows (0, none, all and unlimited also mean
# no cap); a positive integer N adds %N; anything else is refused before
# anything changes.
# PCA_ARMS / PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE / PCA_JOINT_DIM
# (siamese_sl/slurm/config.conf): the arms and their PCA targets, the variance
# kept within each modality (per_modality, both; 0.8) and on the concatenation
# (joint, both; 0.8, or PCA_JOINT_DIM components when set: ONE count for joint
# and both, so keep it below the narrowest concatenation their joint PCA meets,
# the ablations' included; train.py refuses a count at or above it, since
# 2026-10-21, at the fit of the first fold that narrow in each combo or
# ablation run, which no check here can see: in arm both, earlier folds and
# earlier combos may have trained by then). A value config.conf
# refuses stops the run before anything is changed, and train.py's parser is
# asked about every distinct PCA flag set the arms train with (see "0" below).
# POST_PCA_VARIANCE / POST_PCA_DIM are refused (PCA_ARMS=joint is that
# pipeline now); PCA_ARM (the arm of ONE job) is ignored, with a NOTE.
# GPU_NODES (config.conf) moves every job of this submission: the --nodelist
# it builds replaces each script's `#SBATCH --nodelist=gpu3` (the pin since
# 2026-10-01, by request). GPU_NODES=gpu4 works on every Slurm; a list
# (GPU_NODES="gpu3,gpu4") lets each one-node job take whichever listed node is
# free on Slurm >= 23.02 only (older Slurm rejects it). GPU_EXCLUDE only adds
# --exclude, which cannot lift the pin (SLURM rejects a job whose required and
# excluded nodes overlap), so it is refused unless GPU_NODES is set and shares
# no node with it. Both empty keeps the pin. Before anything changes,
# `sbatch --test-only` checks the heaviest job script of each stage submitted
# on the selection (or the pin), and sinfo prints the node state (WARNING when
# down, drained or failing: the DAG would just pend; --test-only cannot place a
# job on such a node, so a failed probe is then that WARNING, not a refusal).
# sl_comparison/folds is read-only from step 0 to the next step 0 (see
# "0, freeze"): run prepare_folds.py by hand only after
# `chmod -R u+w sl_comparison/folds`. Never upload sl_comparison/folds/, data/
# (except the git-ignored inputs this script names when one is missing, copied
# in once while no DAG runs), */results/ or */slurm/logs/ to this checkout;
# while the DAG is queued or running upload nothing (jobs starting hours later
# read its code and config; the SLGNN / NSF4SL tasks refuse a changed builder
# or gene-name mapper, and any other change, fold sources included, is caught
# only by paper_tables: PAPER INCOMPLETE). At submission the code the jobs run is
# recorded (CODE_MANIFEST, see "0, code"; paper_tables lists what changed
# during the run) and data/synlethkg is made read-only (rebuild it by hand only
# after `chmod -R u+w data/synlethkg`). That record is this submission's only:
# a --skip-siamese rerun prints a WARNING naming the code that changed since
# the reused siamese run was submitted (paper_tables does not list it), or a
# NOTE that it cannot when that run left no code manifest here (one submitted
# by siamese_sl/slurm/submit_pipeline.sh writes none), and nothing compares
# the code behind rows reused with --skip-shared.
# PAIR_EXPORT_* (siamese_sl/slurm/export_pair_predictions.conf) set what the
# pair export writes. Defaults: each variant's cv3 combo, every embedding gene
# (19,397), the fold-mean matrices, and Adamson under the cv1 screen split in
# both fine-tuning modes: 28 matrices of 1.5 GB = 42 GB per arm under its
# siamese_sl results root. PAIR_EXPORT_UNIVERSE=shared (the 8,844 benchmark
# genes, which hold 398 of Adamson's 542) brings that to 9 GB;
# PAIR_EXPORT_DTYPE=float16 halves either. Not part of the export, on the same
# disk: the ablation keeps every run's 5 fold matrices (313 MB each on the
# 8,844 shared genes) for paper_tables.py --regrade, by default about 113 GB
# for an arm's 72 leave-one-category-out runs (a variant subset scales it
# down; ABLATION_PURGE_PREDICTIONS=1 removes them), the combos about 19 GB per
# arm, the cell-line ablation's zeroed models about 9 GB per arm, and each
# owner's Step-1 runs up to 89 GB (75 GB when the optional esmc, scprint and
# seqvec are absent: their 9 tasks skip); the banner adds them up for the
# submission (up to 910 GB by default; the eval's per-pair CSVs, the peers'
# fold matrices and the checkpoints come on top, named under it), with the
# jobs and array tasks it queues (491 by default: Slurm's MaxSubmitJobs counts
# every array task).
# PAIR_EXPORT_MODEL_DIR (a direct sbatch of
# export_pair_predictions.sh only) is ignored and cleared. A value that conf
# does not list is refused here, before anything is changed or queued (dataset
# and mode names are matched by the job).
# SEED (default 42). CVS (default "cv1 cv2 cv3") limits only the shared models
# and the compare/re-grade; siamese trains CV_TYPES from config.conf. A CV left
# out of CVS gets NO row from this run: its previous-run shared and DegreePrior
# rows are moved aside like every other tier's, so no table ever mixes two runs
# (--skip-shared is the one path that reuses the previous run's shared rows).
# BESTCAT_EXTRA_VARIANTS=1 (the default since 2026-11-13) also trains the
# single-modality runs of the non-selection variants in the best-per-category array
# (tab:solo for all four; reporting only, never selection; the combo waits for
# the whole array, about 4x the Step-1 wall time).
# RUN_TOKEN is minted here for every submission and exported to every job
# (one inherited from the shell is replaced; with --skip-siamese it is each
# arm's reused selection's, best_per_category.json _selection_meta.run_token,
# else read from the pcavar<PCA_VARIANCE>_*/run_token.txt files of its
# owner's root that paper_tables reads, and when those disagree an exported
# RUN_TOKEN naming one of them is kept; otherwise a WARNING says paper_tables
# will flag the reused dirs as stale). When the arms' reused runs carry
# different tokens, each arm's jobs get their own (PCA_ARM=<arm>
# RUN_TOKEN=<token> sbatch ... --export=ALL) with a WARNING, and RUN_TOKEN,
# which the shared jobs and the manifests' names carry, is the first arm's.
# EVAL_MODEL_DIR / EVAL_VARIANT / PER_HEAD (a direct sbatch of
# eval_finetuning.sh only) and SIAMESE_RESULTS_* / SIAMESE_BASELINE_DISCOVER
# (a hand-run run_compare.sh only) are ignored and cleared. ALLOW_LIVE_JOBS=1:
# see Prerequisites.
# SLGNN and NSF4SL need their original inputs built first on this
# (internet-connected) node, with config.conf's PYTHON_PATH interpreter
# (check_shared_inputs.py prints these with it filled in):
#   (cd SLGNN && $PYTHON_PATH build_synlethkg.py)
#   (cd NSF4SL && $PYTHON_PATH build_transe_features.py --download)
# SLMGAE needs its authors' support-view files in data/: Human_GOsim.txt and
# Human_GOsim_CC.txt (~108 MB each, from the original SLMGAE release; git-
# ignored and built by nothing in this repo, so copy them in) next to the
# tracked biogrid_ppi_sparse.txt and List_Proteins_in_SL.txt.
# The script refuses to submit the shared models until all of these exist
# and the two built inputs carry the current builders' provenance, every
# input file they recorded a hash of unchanged
# (sl_comparison/slurm/check_shared_inputs.py).
# SIAMESE_VARIANTS_LIST /
# ARCH_ARGS / SELECTION_VARIANT (siamese_sl/slurm/config.conf): the siamese
# combo, eval and ablation run once per variant of every arm (default since
# 2026-11-13 "pd:pooled pd:per_head"; any of {pd, bilinear} x {per_head, pooled}), and every arm's table
# gets one siamese row each. SIAMESE_VARIANTS_LIST is a space-separated scalar
# (bash cannot export arrays); config.conf, sourced below, exports the
# validated list so every submitted job sees the same variants this script
# sized its arrays for. SIAMESE_VARIANTS="..." on this script's command line
# works as an alias.
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SIAMESE_DIR="$ROOT/siamese_sl"
SLURM_DIR="$SIAMESE_DIR/slurm"

if [ ! -f "$SLURM_DIR/config.conf" ]; then
    echo "ERROR: $SLURM_DIR/config.conf not found — run from the repo root."
    exit 1
fi
# Settings this run honours from the calling shell, recorded BEFORE config.conf
# gives the unset ones their defaults. Every job is submitted with --export=ALL
# and re-sources the confs, so an export left over from an earlier experiment
# (export PCA_ARMS=joint, ARCH_ARGS=..., PCA_METHOD=robust) trains the paper
# run as another model while this script prints the same lines as ever, and
# paper_tables records the architecture as data without judging it. Each one
# is named after the banner (nothing is printed when none is set).
# Not listed: RUN_TOKEN, PCA_ARM, EVAL_MODEL_DIR / EVAL_VARIANT / PER_HEAD,
# PAIR_EXPORT_MODEL_DIR,
# SIAMESE_RESULTS_* and SIAMESE_BASELINE_DISCOVER are replaced or cleared
# below, each with its own NOTE; SHARED_FOLDS and ABLATION_POST_PCA_DIM are
# refused below, POST_PCA_DIM / POST_PCA_VARIANCE by config.conf itself.
# Their VALUES are recorded too, as NAME=<value, printf %q> words read back
# alike by bash and zsh (2026-10-21): every command this run prints to finish
# its work (the retrain recipe, its one-submission alternatives, a refused
# export's command) carries them, but for the lines of the left-out arms'
# pasted block, whose _paste_env sets only the arms, targets, variants,
# USE_CELL_LINES and CVS their jobs read (those arms' other settings are
# another submission's).
# siamese_sl/slurm/submit_pipeline.sh records the same names
# (test_slurm_contracts holds the two lists to each other). Carried since a
# one-shot SEED=7 or ARCH_ARGS=... in front of this script never
# reached the shell the command is pasted into, and a recipe without them
# reran step 0 at seed 42 under the shared rows it reuses, or retrained the
# arms without ARCH_ARGS. Recorded here, before config.conf normalises them
# and this script exports its own (PAIR_EXPORT_FT_SPLITS=none). _ENV_PREFIX
# leaves out PCA_ARMS (each printed command sets its own) and ALLOW_LIVE_JOBS
# (those commands are for once this run has finished: carried, it would lift
# the guard that keeps their reset_env and step 0 off this run's jobs).
# _ENV_SAME_NOPCA keeps both, for the same submission again, and leaves out the
# PCA targets, which the command that refuses one (--skip-siamese's PCA
# check) prints in full. _ENV_NOPCA leaves out all five, for this submission
# again once its own jobs are gone (the end NOTE's keep-one-token line, which
# sets its arms and targets in full: 2026-10-22).
_ENV_NOTES=""; _ENV_NAMES=""; _ENV_PREFIX=""; _ENV_SAME_NOPCA=""; _ENV_NOPCA=""
_ARGS_Q=$(printf ' %q' "$@")
[ $# -gt 0 ] || _ARGS_Q=""
for _v in ARCH_ARGS PCA_ARMS PCA_MODALITY_VARIANCE PCA_JOINT_VARIANCE \
          PCA_JOINT_DIM PCA_METHOD \
          PREPROCESSING_FIT_SCOPE SIAMESE_ENCODER_TYPE CELL_LINE_DIM \
          USE_CELL_LINES SIAMESE_VARIANTS_LIST SIAMESE_VARIANTS \
          SELECTION_VARIANT SELECTION_CV SELECTION_FRAC SELECTION_SEED \
          SELECTION_MODE ALLMOD_COMBO BESTCAT_EXTRA_VARIANTS \
          BESTCAT_X_KEEP_PREDICTIONS BEST_CAT_ALLOW_MISSING \
          BEST_CAT_PCA_VARIANCE ABLATION_PURGE_CHECKPOINTS \
          ABLATION_PURGE_PREDICTIONS SEED CVS MAX_CONCURRENT EVAL_CONCURRENT \
          SHARED_CONCURRENT GPU_NODES GPU_EXCLUDE TORCH_SPEC TORCH_INDEX_URL \
          ALLOW_LIVE_JOBS PAIR_EXPORT_CVS PAIR_EXPORT_UNIVERSE \
          PAIR_EXPORT_DTYPE PAIR_EXPORT_PER_FOLD PAIR_EXPORT_LOGITS \
          PAIR_EXPORT_FT_SPLITS PAIR_EXPORT_FT_DATASETS PAIR_EXPORT_FT_MODES \
          PAIR_EXPORT_RAM_GB; do
    # SELECTION_FRAC when set at all, even empty (2026-11-13: empty is the
    # way to turn the split off now that the default is 0.1).
    if [ -n "${!_v:-}" ] || { [ "$_v" = SELECTION_FRAC ] && [ -n "${SELECTION_FRAC+x}" ]; }; then
        # SIAMESE_VARIANTS is only an alias: config.conf reads it when
        # SIAMESE_VARIANTS_LIST is empty and ignores it otherwise, so beside a
        # list (named on the line before) it reaches no job.
        if [ "$_v" = SIAMESE_VARIANTS ] && [ -n "${SIAMESE_VARIANTS_LIST:-}" ]; then
            _ENV_NOTES="${_ENV_NOTES}NOTE: ignoring SIAMESE_VARIANTS='${!_v}' from the calling shell: SIAMESE_VARIANTS_LIST is set too, and wins."$'\n'
            continue
        fi
        _ENV_NOTES="${_ENV_NOTES}NOTE: $_v='${!_v}' comes from the calling shell: every job of this run gets it."$'\n'
        _ENV_NAMES="$_ENV_NAMES $_v"
        printf -v _q '%q' "${!_v}"
        case "$_v" in
            PCA_ARMS|ALLOW_LIVE_JOBS) ;;
            *) _ENV_PREFIX="${_ENV_PREFIX}$_v=$_q " ;;
        esac
        case "$_v" in
            PCA_MODALITY_VARIANCE|PCA_JOINT_VARIANCE|PCA_JOINT_DIM) ;;
            *) _ENV_SAME_NOPCA="${_ENV_SAME_NOPCA}$_v=$_q " ;;
        esac
        case "$_v" in
            PCA_ARMS|ALLOW_LIVE_JOBS|PCA_MODALITY_VARIANCE|PCA_JOINT_VARIANCE|PCA_JOINT_DIM) ;;
            *) _ENV_NOPCA="${_ENV_NOPCA}$_v=$_q " ;;
        esac
    fi
done
unset _v _q
# _ucl -> "USE_CELL_LINES=<as this run resolved it> " unless the calling
# shell's is on the line already (recorded); _vset LIST -> that, then
# SIAMESE_VARIANTS_LIST="LIST"; _arms_pin LIST -> 'PCA_ARMS="LIST" ' unless
# the recorded one is (_ENV_SAME_NOPCA). Every orchestrator command this
# script prints pins its arms, the PCA targets, the variants and
# USE_CELL_LINES as this run resolved them, as submit_pipeline.sh's refusal
# lines do (2026-10-22): from a shell exporting USE_CELL_LINES=0 or another
# variant list, a line moved every arm's SLeuth rows aside and rebuilt
# pd:per_head's (or that list's) alone. A refused export's sbatch line pins the
# variants and USE_CELL_LINES (the export takes its root from PCA_ARM), the
# venv rebuild none (reset_env.sh reads none of them). Any other setting the
# pasting shell exports still reaches its jobs: a printed orchestrator command
# names each in a NOTE when it runs, a printed sbatch command names none.
_ucl() {
    case " $_ENV_NOPCA" in
        *" USE_CELL_LINES="*) ;;
        *) printf 'USE_CELL_LINES=%s ' "${USE_CELL_LINES:-1}" ;;
    esac
}
_vset() { printf '%sSIAMESE_VARIANTS_LIST="%s"\n' "$(_ucl)" "$1"; }
_arms_pin() {
    case " $_ENV_SAME_NOPCA" in
        *" PCA_ARMS="*) ;;
        *) printf 'PCA_ARMS="%s" ' "$1" ;;
    esac
}
# One id for this whole submission, exported BEFORE config.conf is sourced so
# config.conf keeps it and every `sbatch --export=ALL` job carries it.
# Step-1 and ablation runs record it (run_token.txt); the combo refuses a
# candidate from another submission and the ablation summary flags one.
# Always fresh: a token inherited from the shell (exported to join an earlier
# run by a direct sbatch) would let that run's directories pass as this one's.
_INHERITED_RUN_TOKEN="${RUN_TOKEN:-}"
RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"
export RUN_TOKEN
# PCA_ARM is the arm of ONE job (config.conf). This script runs every arm of
# PCA_ARMS and hands each arm's jobs their own in sbatch's environment
# (_as_arm: PCA_ARM=<arm> sbatch ... --export=ALL), so one exported in the
# calling shell (meant for a direct sbatch) is dropped before config.conf
# reads it: it would only make this shell's per-job roots another arm's, or
# stop config.conf on a mistyped value. A NOTE below names it.
_INHERITED_PCA_ARM="${PCA_ARM:-}"
unset PCA_ARM
# Sourced for PYTHON_PATH / CV_TYPES and the SIAMESE_VARIANTS array (sizes the
# eval / ablation arrays; exports SIAMESE_VARIANTS_LIST), and the PCA arms
# (PCA_ARMS, validated, as PCA_ARM_LIST, and the pca_* helpers). On a login
# node the compute-node header inside config.conf is guarded by SLURM_JOB_ID
# (no-op here).
source "$SLURM_DIR/config.conf"
# config.conf sets PYTHON_PATH without exporting it. The siamese jobs re-source
# config.conf; the sl_comparison jobs do not, so export it for `sbatch
# --export=ALL` and every job in this DAG runs the same interpreter.
export PYTHON_PATH
# MAX_CONCURRENT / EVAL_CONCURRENT / SHARED_CONCURRENT default to EMPTY in
# config.conf = unlimited (no %N on --array).
export SEED="${SEED:-42}"          # same seed everywhere -> fair division
# (_targets_with, this shell's PCA targets with check_pca's exports in their
# place, is config.conf's pca_targets_with since 2026-10-21:
# submit_pipeline.sh --ablation-only's refusal prints the same.)
# _pca_refs ARM...: append to _CHK (_NREF counts the combos) config.conf
# pca_check_combos' arguments for each ARM: --arm ARM, the PCA flags this shell
# gives it (pca_multi_args), and <variant>=<config.json> of every combo its
# <prefix>latest_combo_<cv>.txt markers name (this run's variants, every CV).
# For --skip-siamese's PCA check and the rebuild of the arms a run leaves out.
_pca_refs() {
    local a v cv m d r
    for a in "$@"; do
        r="$SIAMESE_DIR/$(pca_arm_results "$a")"
        _CHK+=(--arm "$a" "$(pca_multi_args "$a")")
        for v in "${SIAMESE_VARIANTS[@]}"; do
            for cv in "${CV_TYPES[@]}"; do
                m="$r/$(siamese_variant_prefix "$v")latest_combo_${cv}.txt"
                [ -f "$m" ] || continue
                d=$(head -n 1 "$m" | tr -d '\r')
                # (a marker names its combo from siamese_sl/, the jobs' dir)
                case "$d" in ''|/*) ;; *) d="$SIAMESE_DIR/$d" ;; esac
                if [ -n "$d" ] && [ -f "$d/config.json" ]; then
                    _CHK+=("$v=$d/config.json")
                    _NREF=$((_NREF + 1))
                fi
            done
        done
    done
}

# One submission, several PCA arms (config.conf PCA_ARM_LIST). Job ids of one
# arm (or of one owner of single-embedding runs) are kept as <NAME>_<arm>, e.g.
# COMBO_JOB_per_modality (bash 3.2 has no associative arrays): each arm's
# stages run on the plain names (_arm_reset at its start, _arm_save at its
# end), and _arm_load brings them back for its later stages (compare, external
# transfer and paper tables after the shared array; the export last).
_ARM_VARS="COMBO_JOB EVAL_JOB ABL_JOB ABL_SUMM CELL_JOB COMPARE_JOB EXT_JOB PT_JOB PAIR_JOB ALLMOD_JOB PROBE_JOB"
_arm_reset() { local v; for v in $_ARM_VARS; do printf -v "$v" '%s' ""; done; }
_arm_save() { local v; for v in $_ARM_VARS; do printf -v "${v}_$1" '%s' "${!v}"; done; }
_arm_load() { local v n; for v in $_ARM_VARS; do n="${v}_$1"; printf -v "$v" '%s' "${!n}"; done; }
# _armtag ARM -> "" for arm none, "pca_<ARM>_" otherwise: the arm in the name of
# each of its jobs' log files.
_armtag() { local d; d=$(pca_arm_dir "$1"); echo "${d:+${d}_}"; }
# _logs ARM STAGE PATTERN -> the --output / --error of one of ARM's siamese jobs
# relative to siamese_sl/ (the submit dir), as submit_pipeline.sh passes them:
# nothing for arm none (the script's own #SBATCH --output, slurm/logs/
# <stage>_<pattern>.out), else the same names with the arm. For the commands
# printed to paste from siamese_sl/; the submissions below pass absolute ones.
_logs() {
    local t
    t=$(_armtag "$1")
    [ -n "$t" ] || return 0
    echo "--output=slurm/logs/${2}_${t}${3}.out --error=slurm/logs/${2}_${t}${3}.err"
}
# _as_arm ARM sbatch ARGS...: submit one of ARM's jobs, its PCA_ARM (plus its
# own RUN_TOKEN when the arms reuse runs of different tokens, --skip-siamese)
# in sbatch's OWN environment, as prefix assignments (never exported in this
# shell, so no later job inherits them), with a plain --export=ALL in ARGS:
# sbatch hands the job its environment. Without its own token the job gets
# this shell's exported RUN_TOKEN, as every other job. The jobs of every arm
# (reset_env, the generation, the shared folds and array) carry no arm of
# their own: they inherit this shell's PCA_ARM=none (config.conf exports its
# default), which a job without one resolves anyway. Since 2026-10-21; the
# list form --export=ALL,PCA_ARM=<arm>[,RUN_TOKEN=<token>] of 2026-10-20 is
# avoided on purpose: Slurm treats any --export value but ALL / NIL / NONE as
# an implied --get-user-env (slurmd runs `su - <user> -c env` at every task
# launch and lays the job's environment over the login one; from Slurm 25.05 a
# fetch that fails or takes over 120 s requeues the job HELD, and everything
# behind it pends), and the sbatch man page lets the caller's exported value
# take precedence over a listed one (config.conf exports PCA_ARM=none in this
# shell). Every earlier run of this pipeline used plain --export=ALL only.
_as_arm() {
    local a="$1" v="_ARM_TOKEN_$1"
    shift
    if [ -n "${!v}" ]; then
        PCA_ARM="$a" RUN_TOKEN="${!v}" "$@"
    else
        PCA_ARM="$a" "$@"
    fi
}
# _arm_env ARM -> the same assignments as words, "PCA_ARM=<arm>" plus
# " RUN_TOKEN=<token>" when _as_arm passes one: for the commands this script
# prints, in front of their sbatch (bash and zsh alike).
_arm_env() {
    local v="_ARM_TOKEN_$1"
    echo "PCA_ARM=$1${!v:+ RUN_TOKEN=${!v}}"
}
# Every job id this submission queued, in submission order (_on_submit_error).
_queued() { _QUEUED="$_QUEUED $1"; }
# The owners of single-embedding runs among the arms this DAG runs
# (config.conf pca_step1_arm), in PCA_ARMS order: none and per_modality by
# default; joint or both when it runs without the other arm of its pair.
_OWNERS=""
for _a in "${PCA_ARM_LIST[@]}"; do
    _o=$(pca_step1_arm "$_a")
    case " $_OWNERS " in *" $_o "*) ;; *) _OWNERS="${_OWNERS:+$_OWNERS }$_o" ;; esac
done
# _served OWNER -> the arms of this DAG whose single-embedding runs OWNER trains.
_served() {
    local a out=""
    for a in "${PCA_ARM_LIST[@]}"; do
        [ "$(pca_step1_arm "$a")" = "$1" ] && out="${out:+$out }$a"
    done
    echo "$out"
}
unset _a _o

# Job ids start empty: an exported variable of the same name in the calling
# shell must never become a dependency below, nor one of another arm.
ENV_JOB=""; GEN_JOB=""; BESTCAT_JOB=""; FOLDS_JOB=""; SHARED_JOB=""
ELISL_IN_JOB=""; ELISL_JOB=""; ELISL_BUILD=false
PT_ARMS_JOB=""; _QUEUED=""
_arm_reset
for _a in none joint per_modality both; do
    for _v in BESTCAT_JOB $_ARM_VARS; do printf -v "${_v}_$_a" '%s' ""; done
    printf -v "_ARM_TOKEN_$_a" '%s' ""
done
unset _a _v

# ---- flags -----------------------------------------------------------------
RESET_ENV=true; SKIP_GENERATE=false; FORCE_GENERATE=default; AFTER_JOB=""; RUN_SIAMESE=true; RUN_SHARED=true
# The ablation and the external-screen eval hang off the siamese combo. They
# lived only in siamese_sl/slurm/submit_pipeline.sh until 2026-08-24, which
# meant a run driven from THIS orchestrator silently produced neither.
RUN_ABLATION=true; RUN_EVAL=true
# The paper's cell-line, external-transfer and table stages, likewise.
RUN_CELL=true; RUN_EXT=true; RUN_PT=true
# The all-gene-pair export of the siamese combos (no table reads it).
RUN_PAIR=true
# ELISL (context-free), the fourth published peer: its inputs (built when
# check_shared_inputs.py --models ELISL fails) and its 3-CV array, with the
# shared models (since 2026-11-13; it was submitted by hand before).
RUN_ELISL=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-reset-env) RESET_ENV=false; shift ;;
        --skip-generate)  SKIP_GENERATE=true; shift ;;
        --force-generate) FORCE_GENERATE=true; shift ;;
        --auto-generate)  FORCE_GENERATE=false; shift ;;
        --after) case "${2:-}" in
                     ''|*[!0-9_]*|_*|*_|*_*_*)
                         echo "ERROR: --after needs a job ID (<id> or <id>_<task>), got '${2:-}'"; exit 1 ;;
                 esac
                 AFTER_JOB="$2"; shift 2 ;;
        --skip-siamese)   RUN_SIAMESE=false; shift ;;
        --skip-shared)    RUN_SHARED=false; shift ;;
        --skip-ablation)  RUN_ABLATION=false; shift ;;
        --skip-eval)      RUN_EVAL=false; shift ;;
        --skip-cellline-ablation) RUN_CELL=false; shift ;;
        --skip-external-transfer) RUN_EXT=false; shift ;;
        --skip-paper-tables)      RUN_PT=false; shift ;;
        --skip-pair-export)       RUN_PAIR=false; shift ;;
        --skip-elisl)             RUN_ELISL=false; shift ;;
        -h|--help) awk '/^#/{print} !/^#/{exit}' "$0" | tail -n +2 | sed -E 's/^# ?//'; exit 0 ;;
        *) echo "ERROR: unknown option: $1 (see --help)"; exit 1 ;;
    esac
done
if [ "$RUN_SIAMESE" = false ] && [ "$RUN_SHARED" = false ]; then
    echo "ERROR: --skip-siamese and --skip-shared together leave nothing to run."; exit 1
fi
# ELISL is a shared row: --skip-shared reuses it as it reuses the other peers'.
[ "$RUN_SHARED" = true ] || RUN_ELISL=false
# The external-transfer tables are built from the eval outputs of the combos
# this run trains (run_external_transfer.sh admits a variant only with its
# eval_finetuning_{cv1,cv2,cv3}/ complete), and the combo job replaces every
# combo directory: without the eval that job could only fail. --skip-siamese
# keeps it (the reused combos carry their eval outputs).
if [ "$RUN_SIAMESE" = true ] && [ "$RUN_EVAL" = false ] && [ "$RUN_EXT" = true ]; then
    echo "NOTE: --skip-eval: no external-transfer job either (its tables are built from"
    echo "      this run's eval outputs)."
    RUN_EXT=false
fi
if [ "$SKIP_GENERATE" = true ] && [ -n "$AFTER_JOB" ]; then
    echo "ERROR: --skip-generate and --after are mutually exclusive."; exit 1
fi
if [ "$SKIP_GENERATE" = true ] && [ "$FORCE_GENERATE" = true ]; then
    echo "ERROR: --skip-generate and --force-generate are mutually exclusive."; exit 1
fi
if [ "$FORCE_GENERATE" = true ] && [ -n "$AFTER_JOB" ]; then
    echo "ERROR: --force-generate and --after are mutually exclusive."; exit 1
fi
# CVS reaches the folds job, the shared-model array and the compare job as it
# is (--export=ALL), and only sizes the array here: a value that is not a
# list of cv1 / cv2 / cv3 (CVS=cv1,cv2, a typo, a name twice) would fail every
# shared task and leave the compare without a single SLeuth row, days later.
# Checked first (it changes nothing): every command a refusal below prints
# carries CVS, and pasted, it was refused here (2026-10-22).
_CVS_SEEN=" "
for _cv in ${CVS:-cv1 cv2 cv3}; do
    case "$_cv" in cv1|cv2|cv3) ;; *) _CVS_SEEN="" ; break ;; esac
    case "$_CVS_SEEN" in *" $_cv "*) _CVS_SEEN="" ; break ;; esac
    _CVS_SEEN="$_CVS_SEEN$_cv "
done
if [ -z "${_CVS_SEEN// /}" ]; then
    echo "ERROR: CVS='${CVS:-}': want a space-separated subset of cv1 cv2 cv3, each once"
    echo "  (e.g. CVS=\"cv1 cv3\"). Nothing was changed or submitted."
    exit 1
fi
unset _CVS_SEEN _cv
# Every check below that runs Python, step 0 and both manifests run on THIS node
# with config.conf's interpreter, before reset_env is submitted. While a
# reset_env job runs, and after one that failed, timed out or was cancelled (it
# rm -rf's the venv first, then reinstalls), there is none, or a half-installed
# one, and the first of those checks would report it as something else (the
# SLGNN / NSF4SL inputs). Say so here instead.
if ! _PY_ERR=$("$PYTHON_PATH" -c 'import numpy, torch' 2>&1); then
    echo "ERROR: config.conf's PYTHON_PATH cannot run the login-node checks and step 0:"
    echo "    $PYTHON_PATH -c 'import numpy, torch'"
    printf '%s\n' "$_PY_ERR" | tail -n 1 | sed 's/^/    /'
    echo "  The venv is missing or incomplete: a reset_env job is rebuilding it, or one"
    echo "  failed, timed out or was cancelled after it started (it deletes the venv first)."
    echo "  Wait for a live one (squeue -u ${USER:-$(whoami)} -n reset_env); with none, rebuild"
    echo "  the venv, wait for that job to end, then resubmit:"
    echo "    (cd siamese_sl && ${_ENV_PREFIX}sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} slurm/reset_env.sh)"
    echo "  Nothing was changed or submitted."
    exit 1
fi
# --skip-siamese generates nothing: SLGNN / NSF4SL / SLMGAE / DegreePrior train
# on their own inputs, not data/all_genes_*.pt. Generating would only rebuild
# the embeddings under the reused siamese combos, and a failed generation would
# cancel the folds job and every shared task with it.
if [ "$RUN_SIAMESE" = false ]; then
    if [ "$FORCE_GENERATE" = true ]; then
        echo "ERROR: --force-generate with --skip-siamese: no job of this run reads the"
        echo "  embeddings (the shared models train on their own inputs), and regenerating"
        echo "  them would replace the files the reused siamese combos were trained on."
        exit 1
    fi
    SKIP_GENERATE=true
fi
# Neither --force-generate nor --auto-generate: forced, unless this run
# generates nothing (--skip-generate, --skip-siamese) or chains generation
# after another job (--after).
if [ "$FORCE_GENERATE" = default ]; then
    if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
        FORCE_GENERATE=true
    else
        FORCE_GENERATE=false
    fi
fi
# ABLATION_POST_PCA_DIM (an ablation-only width pin) was removed, and
# run_ablation.sh refuses every task while it is set. Stop before step 0 and
# any submission, as submit_pipeline.sh does, rather than queue an ablation
# that can only fail once the multi-day combo has finished.
if [ "$RUN_SIAMESE" = true ] && [ "$RUN_ABLATION" = true ] && [ -n "${ABLATION_POST_PCA_DIM:-}" ]; then
    echo "ERROR: ABLATION_POST_PCA_DIM is no longer supported. To control the ablation"
    echo "  for capacity, train PCA arm joint at an exact width, which its combo AND its"
    echo "  ablation take (PCA_ARMS=joint PCA_JOINT_DIM=<k>; siamese_sl/slurm/config.conf),"
    echo "  or pass --skip-ablation."
    exit 1
fi
# eval_finetuning.sh decodes each task's (cv_type, variant) only when neither
# override is set; one inherited from the calling shell (meant for a direct
# sbatch) would evaluate a single model and no-op every task past the cv list
# with exit 0 (submit_pipeline.sh clears them the same way).
if [ -n "${EVAL_MODEL_DIR:-}" ] || [ -n "${EVAL_VARIANT:-}" ]; then
    echo "NOTE: ignoring EVAL_MODEL_DIR/EVAL_VARIANT from the environment: the chained"
    echo "      eval covers every SIAMESE_VARIANTS entry (direct-sbatch eval_finetuning.sh"
    echo "      for one model)."
fi
unset EVAL_MODEL_DIR EVAL_VARIANT
# eval_finetuning.conf honours an exported PER_HEAD (an override for a direct
# sbatch of one model). For the combos THIS run trains only the value it
# derives from USE_CELL_LINES can load them: every USE_CELL_LINES=1 combo
# (per_head or pooled) needs --per_head, the single-output one refuses it. One
# inherited from the calling shell could at best repeat that value; any other
# makes every eval task stop at its first checkpoint, after the combo job,
# and the external-transfer tables and the export's fine-tuned sets with it.
if [ -n "${PER_HEAD:-}" ]; then
    echo "NOTE: ignoring PER_HEAD=$PER_HEAD from the environment: the chained eval follows"
    echo "      USE_CELL_LINES (direct-sbatch eval_finetuning.sh to override it for one"
    echo "      model)."
fi
unset PER_HEAD
# export_pair_predictions.sh exports ONE combo when PAIR_EXPORT_MODEL_DIR is
# set (meant for a direct sbatch); inherited here, the chained export would
# skip every variant's combo of this run.
if [ -n "${PAIR_EXPORT_MODEL_DIR:-}" ]; then
    echo "NOTE: ignoring PAIR_EXPORT_MODEL_DIR from the environment: the chained pair"
    echo "      export covers every SIAMESE_VARIANTS entry (direct-sbatch"
    echo "      export_pair_predictions.sh for one combo)."
fi
unset PAIR_EXPORT_MODEL_DIR
# collect_and_regrade_siamese.sh lets per-CV pins (SIAMESE_RESULTS_<cv>,
# SIAMESE_RESULTS_<tag>_<cv>) and SIAMESE_BASELINE_DISCOVER=all replace this
# run's latest_combo combos; left over from a hand-run compare, they would
# grade another run's model as this run's SLeuth row in tab:tiers.
_PINS=$(compgen -v SIAMESE_RESULTS_ || true)
if [ -n "$_PINS" ] || [ -n "${SIAMESE_BASELINE_DISCOVER+x}" ]; then
    echo "NOTE: ignoring" $_PINS ${SIAMESE_BASELINE_DISCOVER+SIAMESE_BASELINE_DISCOVER} "from the"
    echo "      environment: the chained compare grades this run's combos only"
    echo "      (sbatch sl_comparison/slurm/run_compare.sh by hand to pin one)."
fi
for _v in $_PINS; do unset "$_v"; done
unset SIAMESE_BASELINE_DISCOVER
# (PCA_ARM was dropped before config.conf was sourced, above.)
if [ -n "$_INHERITED_PCA_ARM" ]; then
    echo "NOTE: ignoring PCA_ARM=$_INHERITED_PCA_ARM from the environment: this DAG runs the PCA"
    echo "      arms of PCA_ARMS ($PCA_ARMS), each job with its own (PCA_ARMS=<arm> runs"
    echo "      one)."
fi
# config.conf honours an EXPORTED SHARED_FOLDS (`${SHARED_FOLDS:-...}`) and turns
# it into --folds_dir for every siamese train.py, but step 0 rebuilds
# sl_comparison/folds and every peer and grader reads THAT set by a fixed path
# (run_shared_models.sh, evaluate_model.py's default, zero_cell_emb.py,
# run_external_transfer.sh, paper_tables.py). One left over from a siamese-only
# run (slurm/submit_pipeline.sh) trains SLeuth on another partition for days,
# and the fold stamp then refuses every SLeuth row, leaving the tables without
# one. Stop here, before step 0 and any submission, as the guards above do.
if [ "$RUN_SIAMESE" = true ]; then
    # Both sides normalised the same way, so a symlinked checkout (ROOT is the
    # LOGICAL path) can never make the default refuse itself; the leaf need not
    # exist yet, since step 0 writes it below.
    _norm_folds() {
        ( cd "$(dirname "$1")" 2>/dev/null \
            && printf '%s/%s' "$(pwd -P)" "$(basename "$1")" ) || printf '%s' "$1"
    }
    # The siamese jobs run from siamese_sl/ (config.conf's WORK_DIR).
    case "$SHARED_FOLDS" in
        /*) _SF="$SHARED_FOLDS" ;;
        *)  _SF="$SIAMESE_DIR/$SHARED_FOLDS" ;;
    esac
    if [ "$(_norm_folds "$_SF")" != "$(_norm_folds "$ROOT/sl_comparison/folds")" ]; then
        echo "ERROR: SHARED_FOLDS=$SHARED_FOLDS (siamese_sl/slurm/config.conf, overridable"
        echo "  from the environment) resolves to"
        echo "    $(_norm_folds "$_SF"),"
        echo "  not the $(_norm_folds "$ROOT/sl_comparison/folds") this DAG builds"
        echo "  (step 0) and grades every model on. Only the siamese jobs would follow the"
        echo "  override, and the re-grade, the cell-line ablation, the external transfer"
        echo "  and paper_tables would then refuse every SLeuth row on the fold stamp,"
        echo "  after the whole run. Unset SHARED_FOLDS (an override is for siamese-only"
        echo "  runs: siamese_sl/slurm/submit_pipeline.sh). Nothing was changed or submitted."
        exit 1
    fi
    unset _SF
fi
# --skip-siamese: no siamese job runs, and every arm's tables reuse that arm's
# siamese tree on disk. paper_tables.py checks the single-modality dirs against
# the token the selection recorded (best_per_category.json
# _selection_meta.run_token) and the ablation dirs against RUN_TOKEN, and every
# ablation of that selection carries its token (this script's run, or
# submit_pipeline.sh --ablation-only), so take it from there, per arm. A
# selection without one falls back to the single-modality runs of the pcavar
# tag paper_tables reads (other tags are other runs'), in the arm's owner's
# root, when they agree on one; when they disagree (e.g. a task re-sbatched by
# hand minted its own), an exported RUN_TOKEN naming one of them is kept. Arms
# whose reused runs carry different tokens each get their own (_as_arm), and
# RUN_TOKEN, which the shared jobs and the manifests carry, is the first arm's.
_TOKEN_NOTE=""; _PER_ARM_TOKENS=false; _REUSED_TOKENS="$RUN_TOKEN"
# _KEEP_TOKEN_ARMS: the arms whose reused runs carry several tokens, none kept
# (a run that queues its jobs ends with the submission that keeps one, and
# the PCA refusal below prints one); _TOKEN_KEPT: an exported RUN_TOKEN was
# kept for an arm (the refusal's resubmit line carries it).
_KEEP_TOKEN_ARMS=""; _TOKEN_KEPT=""
if [ "$RUN_SIAMESE" = false ]; then
    # Resolved as run_paper_tables.sh does (the conf, not only the environment).
    _PCA_TAG=$( ( source "$SLURM_DIR/run_best_per_category.conf" >/dev/null 2>&1 || true
                  echo "${PCA_VARIANCE:-}" ) 2>/dev/null )
    _PCA_TAG="${_PCA_TAG:-${BEST_CAT_PCA_VARIANCE:-0.8}}"
    _SEL_PY='import json,sys; print((json.load(open(sys.argv[1])).get("_selection_meta") or {}).get("run_token") or "")'
    _FRESH="$RUN_TOKEN"; _TOKS_ALL=""; _NOTES_FIRST=""; _NOTHING=""
    for _a in "${PCA_ARM_LIST[@]}"; do
        _SR_A="siamese_sl/$(pca_arm_results "$_a")"
        _SR_O="siamese_sl/$(pca_arm_results "$(pca_step1_arm "$_a")")"
        _SEL_JSON="$ROOT/$_SR_A/best_per_category.json"
        _SEL_TOKEN=""
        if [ -f "$_SEL_JSON" ]; then
            _SEL_TOKEN=$("$PYTHON_PATH" -c "$_SEL_PY" "$_SEL_JSON" 2>/dev/null \
                || python3 -c "$_SEL_PY" "$_SEL_JSON" 2>/dev/null || true)
        fi
        _TOK_FILES=("$ROOT/$_SR_O/pcavar${_PCA_TAG}_"*/run_token.txt)
        _TOKS=$(cat "${_TOK_FILES[@]}" 2>/dev/null | sort -u)
        _NTOK=0
        [ -n "$_TOKS" ] && _NTOK=$(printf '%s\n' "$_TOKS" | wc -l | tr -d ' ')
        _TOK=""; _WHY=""
        if [ -n "$_SEL_TOKEN" ]; then
            _TOK="$_SEL_TOKEN"
            _WHY=" (the reused selection's: best_per_category.json _selection_meta.run_token)"
        elif [ "$_NTOK" = 1 ]; then
            _TOK="$_TOKS"
            _WHY=" (the reused siamese run's)"
        elif [ "$_NTOK" -gt 1 ] && [ -n "$_INHERITED_RUN_TOKEN" ] \
                && printf '%s\n' "$_TOKS" | grep -qxF -- "$_INHERITED_RUN_TOKEN"; then
            _TOK="$_INHERITED_RUN_TOKEN"
            _WHY=" (from the environment: one of the reused siamese run's $_NTOK)"
            _TOKEN_KEPT=1
        else
            _TOK="$_FRESH"
            _NDIRS=$(ls -d "$ROOT/$_SR_O/pcavar${_PCA_TAG}_"*/ 2>/dev/null | wc -l | tr -d ' ')
            # Nothing of the arm on disk (never run in this checkout: no
            # selection, no combo marker in its root, no single-modality run
            # in its owner's): no dir for a token to judge. It gets one NOTE
            # below instead of this WARNING about "the siamese tree" it
            # reuses, and no token of its own, which the messages below would
            # read as a run of another submission (2026-10-21): the first
            # reused arm's, the fresh one when no arm has anything. Still
            # queued, as documented: its tables end PAPER INCOMPLETE, no
            # number wrong. (joint beside none's Step-1 runs is not such an
            # arm: it takes its owner's token, the branch above.)
            _EMPTY=false
            if [ ! -f "$_SEL_JSON" ] && [ "$_NDIRS" = 0 ]; then
                _EMPTY=true
                for _m in "$ROOT/$_SR_A"/latest_combo_cv[123].txt \
                          "$ROOT/$_SR_A"/var_*__latest_combo_cv[123].txt; do
                    [ ! -e "$_m" ] || _EMPTY=false
                done
            fi
            if [ "$_EMPTY" = true ]; then
                _TOK=""
                _NOTHING="${_NOTHING:+$_NOTHING }$_a"
            else
            echo "WARNING: --skip-siamese reuses the siamese tree of PCA arm $_a on disk, but its"
            echo "         selection ($_SR_A/best_per_category.json) records no run token and its"
            if [ "$_NTOK" = 0 ]; then
                echo "         single-modality runs ($_SR_O/pcavar${_PCA_TAG}_*/, $_NDIRS dir(s))"
                echo "         carry no run_token.txt."
            else
                echo "         single-modality runs ($_SR_O/pcavar${_PCA_TAG}_*/run_token.txt)"
                echo "         carry $_NTOK different run tokens (dirs per token):"
                cat "${_TOK_FILES[@]}" 2>/dev/null | sort | uniq -c | sed 's/^ */           /'
            fi
            echo "         With this submission's fresh token, paper_tables flags EVERY reused"
            echo "         single-modality and ablation dir (tab:solo, tab:loco) of that arm as"
            echo "         stale."
            # (the submission that keeps one of them is printed once this run
            # has queued its jobs, with their scancel line, or by the PCA
            # refusal below: printed here, it named a scancel line that a
            # refused run never printed, and carried targets the refusal
            # then refused; 2026-10-22)
            if [ "$_NTOK" -gt 1 ]; then
                echo "         (The submission that keeps one of them instead is printed once this run"
                echo "         is queued.)"
                _KEEP_TOKEN_ARMS="${_KEEP_TOKEN_ARMS:+$_KEEP_TOKEN_ARMS }$_a"
            fi
            fi
        fi
        # Handed to the arm's jobs as RUN_TOKEN=<token> in sbatch's
        # environment when the arms' tokens differ (_as_arm), and printed,
        # unquoted, in the commands this script prints to paste (_arm_env):
        # one plain shell word, as every token the orchestrators mint is.
        case "$_TOK" in
            *[!A-Za-z0-9._:+-]*)
                echo "ERROR: the run token '$_TOK' of PCA arm $_a is not one plain shell word"
                echo "  ([A-Za-z0-9._:+-] only), as the commands printed to paste need it."
                echo "  Nothing was changed or submitted."
                exit 1 ;;
        esac
        printf -v "_ARM_TOKEN_$_a" '%s' "$_TOK"
        [ -n "$_NOTES_FIRST" ] || _NOTES_FIRST="$_WHY"
        [ -z "$_TOK" ] || case " $_TOKS_ALL " in
            *" $_TOK "*) ;;
            *) _TOKS_ALL="${_TOKS_ALL:+$_TOKS_ALL }$_TOK" ;;
        esac
    done
    # An arm with nothing on disk takes the first reused arm's token (the
    # fresh one when no arm has anything), so it never reads as a run of
    # another submission.
    [ -n "$_TOKS_ALL" ] || _TOKS_ALL="$_FRESH"
    for _a in $_NOTHING; do printf -v "_ARM_TOKEN_$_a" '%s' "${_TOKS_ALL%% *}"; done
    if [ -n "$_NOTHING" ]; then
        echo "NOTE: PCA arm(s) $_NOTHING have no siamese outputs here (no best_per_category.json,"
        echo "  latest_combo marker or single-modality run in their roots): their comparison,"
        echo "  external-transfer and paper tables get no SLeuth row (PAPER INCOMPLETE), and the"
        echo "  table across the arms lists them as missing. Set PCA_ARMS to the other arms to"
        echo "  leave them out."
    fi
    # --skip-siamese labels every reused arm, and paper_tables checks its
    # combos, with THIS shell's PCA targets (config.conf exports them to every
    # job; pca_arm.sh, run_compare.sh --pca_label): combos another submission
    # trained at other targets were labelled with the wrong ones in
    # comparison_notes.txt and every table, and their paper tables (and the
    # table across the arms) came out PAPER INCOMPLETE where step 0a had just
    # removed COMPLETE ones. So each reused arm's marker-named combos'
    # config.json is compared first, before anything changes, with the PCA
    # this shell gives the arm (config.conf pca_check_combos, as
    # submit_pipeline.sh --ablation-only does): when one set of targets trains
    # every reused arm's combos, the run is refused with it; when no one set
    # does (the arms disagree among themselves), it is only a WARNING naming
    # the arms whose tables will be INCOMPLETE (2026-10-21).
    _CHK=(); _NREF=0
    _pca_refs "${PCA_ARM_LIST[@]}"
    _OUT=""
    if [ "$_NREF" -gt 0 ] && ! _OUT=$(pca_check_combos "${_CHK[@]}"); then
        echo "WARNING: could not compare the reused arms' PCA with their combos' config.json on"
        echo "  this node (arch_signature.py --check_pca): paper_tables still lists a combo of"
        echo "  other targets than this shell's."
        _OUT=""
    fi
    IFS='|' read -r _ST _EX <<< "$(printf '%s\n' "$_OUT" | sed -n 's/^all|//p')"
    if [ "$_ST" = fix ] || [ "$_ST" = other ]; then
        _OFF=""
        while IFS='|' read -r _k _x _s _rest; do
            [ "$_k" = arm ] && [ "$_s" != ok ] && _OFF="${_OFF:+$_OFF }$_x"
        done <<< "$_OUT"
        if [ "$_ST" = fix ]; then
            echo "ERROR: --skip-siamese reuses PCA arm(s) $_OFF, whose combos were trained at"
            echo "  other PCA targets than this shell's ($(pca_targets_env)):"
        else
            echo "WARNING: --skip-siamese reuses PCA arms trained at PCA targets that no one set of"
            echo "  exports reproduces for all of them (this shell's: $(pca_targets_env)):"
        fi
        while IFS='|' read -r _k _x _y _d _s _l _rest; do
            [ "$_k" = combo ] && [ "$_s" = bad ] && echo "    $_x: ${_d#$ROOT/}  ($_y: $_l)"
        done <<< "$_OUT"
        echo "  Every job of this run labels each arm, and paper_tables checks its combos, with"
        echo "  this shell's targets: the tables of PCA arm(s) $_OFF would be PAPER INCOMPLETE"
        echo "  (their combos are not of those targets), and so would the table across the arms."
        if [ "$_ST" = fix ]; then
            # (the same submission again: with the exported RUN_TOKEN it kept
            # for an arm, and, where an arm's reused runs carry several
            # tokens, the line that keeps one: the WARNING above prints none
            # since 2026-10-22, and the end NOTE that would is not reached)
            echo "  One set of targets trains every reused arm's combos: resubmit with it,"
            echo "    ${_TOKEN_KEPT:+RUN_TOKEN=$_INHERITED_RUN_TOKEN }${_ENV_SAME_NOPCA}$(_arms_pin "${PCA_ARM_LIST[*]}")$(pca_targets_with "$_EX") $(_vset "${SIAMESE_VARIANTS[*]}") bash $0$_ARGS_Q"
            if [ -n "$_KEEP_TOKEN_ARMS" ]; then
                echo "  or, to keep one of the run tokens of PCA arm(s) $_KEEP_TOKEN_ARMS (the WARNING above;"
                echo "  one RUN_TOKEN is kept only for the arms whose runs carry it), with that token in"
                echo "  place of <token>:"
                echo "    RUN_TOKEN=<token> ${_ENV_SAME_NOPCA}$(_arms_pin "${PCA_ARM_LIST[*]}")$(pca_targets_with "$_EX") $(_vset "${SIAMESE_VARIANTS[*]}") bash $0$_ARGS_Q"
            fi
            echo "  Nothing was changed or submitted."
            exit 1
        fi
        echo "  Their own targets (none: the arm's combos disagree among themselves, or are"
        echo "  not its PCA):"
        while IFS='|' read -r _k _x _s _e _rest; do
            [ "$_k" = arm ] && [ "$_s" != ok ] && echo "    $_x: ${_e/#-/none}"
        done <<< "$_OUT"
        echo "  Once this run has finished, rebuild such an arm's comparison, external transfer"
        echo "  and paper tables with its targets exported, or retrain it at this shell's."
    fi
    unset _CHK _NREF _OUT _ST _EX _OFF _k _x _y _d _s _l _e _rest
    _v="_ARM_TOKEN_${PCA_ARM_LIST[0]}"
    RUN_TOKEN="${!_v}"
    _REUSED_TOKENS="$_TOKS_ALL"
    if [ "${_TOKS_ALL// /}" = "$_TOKS_ALL" ]; then
        # one token for every arm (the reason of the first): no job needs one
        # of its own (_as_arm passes none; every job inherits RUN_TOKEN)
        _TOKEN_NOTE="$_NOTES_FIRST"
        for _a in none joint per_modality both; do printf -v "_ARM_TOKEN_$_a" '%s' ""; done
    else
        _PER_ARM_TOKENS=true
        _TOKEN_NOTE=" (PCA arm ${PCA_ARM_LIST[0]}'s; each arm's jobs get their own, below)"
        echo "WARNING: the PCA arms reused here were run by submissions of different tokens:"
        for _a in "${PCA_ARM_LIST[@]}"; do
            _v="_ARM_TOKEN_$_a"
            echo "    $_a: ${!_v}"
        done
        echo "  Each arm's jobs get their own (PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ... --export=ALL),"
        echo "  so its own tables check its own runs; the table across the arms"
        echo "  (sl_comparison/results/pca_arms/) flags the arms of another submission as stale."
    fi
    export RUN_TOKEN
    unset _PCA_TAG _SEL_PY _FRESH _TOKS_ALL _NOTES_FIRST _a _v _SR_A _SR_O _SEL_JSON \
          _SEL_TOKEN _TOK_FILES _TOKS _NTOK _TOK _WHY _NDIRS
fi
# The single-embedding runs this DAG removes (step 0a) and rebuilds (each
# owner's Step-1 array) may be the ones the owner's pair partner, an arm
# PCA_ARMS leaves out, ranked in the selection it keeps
# (_selection_meta.step1_root names that root): that arm's tables would then
# find this run's runs under its selection (its tab:solo flags them as another
# run's) until the arm is rerun. Said before anything changes; the arm's own
# outputs are not touched.
if [ "$RUN_SIAMESE" = true ]; then
    _S1_PY='import json,sys; print((json.load(open(sys.argv[1])).get("_selection_meta") or {}).get("step1_root") or "")'
    for _o in $_OWNERS; do
        # (the whole pair: config.conf's pca_step1_arms lists this run's arms only)
        case "$_o" in
            per_modality|both) _pair="per_modality both" ;;
            *)                 _pair="none joint" ;;
        esac
        for _p in $_pair; do
            case " $PCA_ARMS " in *" $_p "*) continue ;; esac
            _sel="siamese_sl/$(pca_arm_results "$_p")/best_per_category.json"
            [ -f "$ROOT/$_sel" ] || continue
            _s1=$("$PYTHON_PATH" -c "$_S1_PY" "$ROOT/$_sel" 2>/dev/null \
                  || python3 -c "$_S1_PY" "$ROOT/$_sel" 2>/dev/null || true)
            [ "${_s1%/}" = "$(pca_arm_results "$_o")" ] || continue
            echo "WARNING: PCA arm $_p, which PCA_ARMS leaves out, ranked the single-embedding runs in"
            echo "  siamese_sl/$(pca_arm_results "$_o")/ ($_sel, _selection_meta.step1_root), which"
            echo "  this run removes and rebuilds as PCA arm $_o's Step 1: until $_p is rerun"
            echo "  (PCA_ARMS listing it), its tab:solo will be another run's."
        done
    done
    unset _S1_PY _o _p _pair _sel _s1
fi
# EMB_CATALOG, PCA_VARIANCE and bestcat_n_tasks; it also validates
# BESTCAT_EXTRA_VARIANTS, so a bad value stops the run here. bestcat_n_tasks
# is the one Step-1 array size every launcher and run_best_per_category.sh's
# task decode share: |EMB_CATALOG| x |CV_TYPES| on SELECTION_VARIANT, plus one
# block per other variant with BESTCAT_EXTRA_VARIANTS=1.
if [ "$RUN_SIAMESE" = true ]; then
    source "$SLURM_DIR/run_best_per_category.conf"
    if declare -F bestcat_n_tasks >/dev/null; then
        BESTCAT_N=$(bestcat_n_tasks)
    elif [ "${BESTCAT_EXTRA_VARIANTS:-0}" != 0 ]; then
        echo "ERROR: BESTCAT_EXTRA_VARIANTS=$BESTCAT_EXTRA_VARIANTS but run_best_per_category.conf"
        echo "  defines no bestcat_n_tasks to size the array with."
        exit 1
    else
        BESTCAT_N=$(( ${#EMB_CATALOG[@]} * ${#CV_TYPES[@]} ))
    fi
fi

# ---- code the submitted jobs import or run, before anything else ----------
# A file that never reached this checkout (never uploaded, or never committed
# where the tree comes from) would make the DAG fail hours in (the combo's
# Step 2 imports arch_signature after the whole Step-1 array; every SLGNN task
# imports slgnn_orig_model / build_synlethkg; every sl_comparison job sources
# python_path.sh, and every arm's compare, external transfer and paper tables
# pca_arm.sh). Per stage, so a skipped stage needs nothing.
# folds_manifest.sh: step 0 below and the folds job source it;
# code_manifest.py: run below at submission, re-checked by paper_tables.
NEED_CODE="sl_comparison/slurm/python_path.sh sl_comparison/slurm/folds_manifest.sh \
    sl_comparison/code_manifest.py sl_comparison/slurm/pca_arm.sh"
[ "$RUN_SIAMESE" = true ] && NEED_CODE="$NEED_CODE siamese_sl/arch_signature.py"
# the generation job ends with it (and the auto-skip gate below runs it)
[ "$RUN_SIAMESE" = true ] && NEED_CODE="$NEED_CODE siamese_sl/embedding_versions.py"
[ "$RUN_SIAMESE" = true ] && [ "$RUN_CELL" = true ] \
    && NEED_CODE="$NEED_CODE siamese_sl/cell_line_ablation/run_cell_line_ablation.sh \
       siamese_sl/cell_line_ablation/zero_cell_emb.py siamese_sl/cell_line_ablation/grade.sh"
# the pair export is the LAST job: a file missing from the upload would
# otherwise surface only after everything else has run
[ "$RUN_SIAMESE" = true ] && [ "$RUN_PAIR" = true ] \
    && NEED_CODE="$NEED_CODE siamese_sl/export_pair_predictions.py \
       siamese_sl/slurm/export_pair_predictions.sh \
       siamese_sl/slurm/export_pair_predictions.conf"
# the ablation's array worker, summary and conf: a missing one would stop the
# run only at its submission, after step 0, with part of the DAG queued
[ "$RUN_SIAMESE" = true ] && [ "$RUN_ABLATION" = true ] \
    && NEED_CODE="$NEED_CODE siamese_sl/slurm/run_ablation.sh \
       siamese_sl/slurm/run_ablation.conf siamese_sl/slurm/run_ablation_summary.sh"
[ "$RUN_SHARED" = true ] && NEED_CODE="$NEED_CODE SLGNN/slgnn_orig_model.py \
    SLGNN/build_synlethkg.py sl_comparison/slurm/check_shared_inputs.py \
    NSF4SL/build_transe_features.py"
[ "$RUN_ELISL" = true ] && NEED_CODE="$NEED_CODE ELISL/train_elisl.py \
    ELISL/build_elisl_inputs.py ELISL/slurm/run_elisl.sh \
    ELISL/slurm/build_elisl_inputs.sh"
[ "$RUN_EXT" = true ] && NEED_CODE="$NEED_CODE sl_comparison/external_transfer.py \
    sl_comparison/slurm/run_external_transfer.sh"
# paper_tables.py also loads external_transfer.py (per-screen coverage) and
# arch_signature.py (tab:solo / tab:loco arch checks, each combo's PCA arm)
# even when their stages are skipped; without them those items go 'missing'.
[ "$RUN_PT" = true ] && NEED_CODE="$NEED_CODE sl_comparison/paper_tables.py \
    sl_comparison/slurm/run_paper_tables.sh sl_comparison/external_transfer.py \
    siamese_sl/arch_signature.py"
_MISSING_CODE=""
for _f in $NEED_CODE; do
    case " $_MISSING_CODE " in *" $_f "*) continue ;; esac
    [ -f "$ROOT/$_f" ] || _MISSING_CODE="$_MISSING_CODE $_f"
done
if [ -n "$_MISSING_CODE" ]; then
    echo "ERROR: files the submitted jobs need are missing from this checkout:"
    for _f in $_MISSING_CODE; do echo "  $_f"; done
    echo "  (never uploaded to this checkout, or untracked in git? copy them here, then resubmit)"
    exit 1
fi
# So would a mistyped PAIR_EXPORT_* value, seen only by that last job. In a
# subshell: the conf's defaults must reach neither this shell nor the jobs.
if [ "$RUN_SIAMESE" = true ] && [ "$RUN_PAIR" = true ] \
   && ! ( source "$SLURM_DIR/export_pair_predictions.conf" && check_pair_export ); then
    echo "  Fix it (siamese_sl/slurm/export_pair_predictions.conf lists the values) or"
    echo "  pass --skip-pair-export. Nothing was changed or submitted."
    exit 1
fi

# Auto-skip generation when the embeddings are already on disk AND current,
# so a resubmit after a cancel never re-triggers the flaky ESM/NCBI/CUDA
# generation step (its failure would cascade over the whole DAG).
# kg_complex (a siamese embedding, and the opt-in legacy SLGNN/NSF4SL input) is
# a reliable "generation done" marker; --force-generate overrides. The check
# runs where the DAG is submitted (cluster login node), which shares the
# compute nodes' filesystem.
EMB_MARKER="$ROOT/data/all_genes_kg_complex.pt"
EMB_VERSIONS="$SIAMESE_DIR/embedding_versions.py"
if [ "$SKIP_GENERATE" = false ] && [ "$FORCE_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    if [ -f "$EMB_MARKER" ]; then
        N_EMB=$(ls "$ROOT"/data/all_genes_*.pt 2>/dev/null | wc -l | tr -d ' ')
        # EXISTENCE IS NOT ENOUGH. This guard used to skip generation whenever
        # the file was present, so an embedding set built over a DIFFERENT gene
        # universe was silently reused — which is exactly how the benchmark ran
        # on 5,257 genes instead of the full universe, giving siamese a
        # different test-set prevalence from its peers with nothing in the log
        # saying so. Verify the stored gene_order matches gene_universe.txt.
        UNIV="$ROOT/data/gene_universe.txt"
        EMB_OK=$("$PYTHON_PATH" - "$EMB_MARKER" "$UNIV" <<'PYEOF'
import sys
try:
    import torch
    d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    go = d.get('gene_order') if isinstance(d, dict) else None
    if go is None:
        print("no gene_order stored"); raise SystemExit(0)
    want = {l.strip() for l in open(sys.argv[2]) if l.strip()}
    have = set(map(str, go))
    # SET comparison, not ordered. The .pt files store gene_order sorted
    # LEXICOGRAPHICALLY ('1','10','100') while gene_universe.txt is sorted
    # NUMERICALLY ('1','2','9'); data_loader maps the shared genes onto rows
    # by lookup, so gene_universe.txt's order is irrelevant here (the
    # identical cross-file order load_raw_multimodal needs is checked by
    # embedding_versions.py below). Comparing ordered lists here would report
    # every healthy embedding set as stale and burn a full ESM regeneration
    # on every run.
    if have != want:
        miss, extra = len(want - have), len(have - want)
        print(f"gene universe mismatch: {len(have)} genes stored vs "
              f"{len(want)} expected ({miss} missing, {extra} unexpected)")
        raise SystemExit(0)
    print("")           # empty == matches
except SystemExit:
    raise
except Exception as e:
    print(f"could not verify ({type(e).__name__}: {e})")
PYEOF
) || EMB_OK="verification failed"
        # A matching universe says nothing about HOW the vectors were built:
        # the canonical-protein rewrite (and the symbol-mapper / Mashup fixes)
        # kept the universe, so files from older generator code pass the check
        # above. embedding_versions.py checks every catalog file's recorded
        # generator provenance; any non-zero exit (stale, missing, crash) or a
        # missing checker regenerates.
        if [ -z "$EMB_OK" ]; then
            if [ ! -f "$EMB_VERSIONS" ]; then
                EMB_OK="siamese_sl/embedding_versions.py missing (cannot verify generator provenance)"
            elif ! "$PYTHON_PATH" "$EMB_VERSIONS" --data_dir "$ROOT/data"; then
                EMB_OK="built by older generation code or provenance missing (embedding_versions.py above)"
            fi
        fi
        if [ -n "$EMB_OK" ]; then
            echo "--- embeddings present but STALE: $EMB_OK ---"
            echo "    regenerating with the current generation code"
        else
            echo "--- embeddings already present ($N_EMB all_genes_*.pt, incl kg_complex;"
            echo "    gene_order matches gene_universe.txt and embedding_versions.py reports"
            echo "    every present file current, none but OPTIONAL_TYPES absent)"
            echo "    -> auto-skipping generation ---"
            echo "    (pass --force-generate to regenerate anyway)"
            SKIP_GENERATE=true
        fi
    else
        echo "--- no embeddings found ($EMB_MARKER missing) -> generation WILL run ---"
    fi
elif [ "$SKIP_GENERATE" = true ] && [ "$RUN_SIAMESE" = true ]; then
    # Honoured (the caller asked), but not silently.
    if [ ! -f "$EMB_VERSIONS" ] || ! "$PYTHON_PATH" "$EMB_VERSIONS" --data_dir "$ROOT/data"; then
        echo "WARNING: --skip-generate with embeddings embedding_versions.py cannot confirm"
        echo "         as current: siamese trains on them as they are."
    fi
fi

# Absolute log dirs so SLURM writes each job's stdout/stderr to the right place
# regardless of that job's submit dir. Passed to sbatch as --output/--error
# (a CLI value overrides the script's relative #SBATCH path), and created up
# front because SLURM opens the log file at job start — a missing directory
# makes it silently drop the job's output.
SIA_LOGS="$SIAMESE_DIR/slurm/logs"
CMP_LOGS="$ROOT/sl_comparison/slurm/logs"
ELISL_LOGS="$ROOT/ELISL/slurm/logs"
mkdir -p "$SIA_LOGS" "$CMP_LOGS"
[ "$RUN_ELISL" != true ] || mkdir -p "$ELISL_LOGS"
cd "$ROOT"

# Whether ELISL's inputs must be built (the banner counts the job): decided
# here, silently; reported, checked and downloaded under the banner (below,
# after the SLGNN / NSF4SL / SLMGAE input check).
_ELISL_CHECK=""
if [ "$RUN_ELISL" = true ]; then
    _ELISL_CHECK=$("$PYTHON_PATH" sl_comparison/slurm/check_shared_inputs.py --models ELISL 2>&1) \
        || ELISL_BUILD=true
fi


# What the DAG costs, per arm and per owner of single-embedding runs, and the
# fold matrices it keeps: NUM_FOLDS x 313 MB per run (8,844 x 8,844 float32 on
# the shared genes); the pair export 42 GB per arm by default. (BESTCAT_N and
# the ablation size are the arrays below; run_ablation.conf in a subshell, as
# its values must not reach this shell before step 3c.) Since 2026-10-21 it
# counts the cell-line ablation's kept fold matrices too (zero_cell_emb.py
# --save_matrices, 9 GB per arm by default), calls the single-embedding figure
# an upper bound (a Step-1 task of an absent optional embedding keeps none:
# 48 of the 57 trained in the 2026-09 run), and names what the submission
# queues in jobs and in array tasks: Slurm's MaxSubmitJobs (sacctmgr's
# MaxSubmit / GrpSubmit, a QOS's MaxSubmitPU) counts every task, and the four
# arms queue 491 by default where one arm queued 160.
_N_ARMS=${#PCA_ARM_LIST[@]}
_N_OWNERS=$(echo $_OWNERS | wc -w | tr -d ' ')
_ARM_COST=""; _OWNER_COST=""; _DISK=""; _ARM_GB=0; _OWNER_GB=0; _OWNER_MIN=0; _OPT=""
_NJOBS=0; _NTASKS=0
_count() { _NJOBS=$((_NJOBS + $1)); _NTASKS=$((_NTASKS + $2)); }   # N_JOBS N_TASKS
if [ "$RUN_SIAMESE" = true ]; then
    _gb() { echo $(( ($1 * NUM_FOLDS * 313 + 500) / 1000 )); }
    _COMBOS=$(( ${#SIAMESE_VARIANTS[@]} * ${#CV_TYPES[@]} ))
    _ARM_COST="$_COMBOS combos (${#SIAMESE_VARIANTS[@]} variants x ${#CV_TYPES[@]} CVs)"
    _ARM_GB=$(_gb "$_COMBOS"); _DISK="$_ARM_GB GB of combos"
    if [ "$ALLMOD_COMBO" = 1 ]; then
        _ARM_COST="$_ARM_COST, $_COMBOS all-modalities combos (ALLMOD_COMBO=1)"
        _ARM_GB=$(( _ARM_GB + $(_gb "$_COMBOS") ))
        _DISK="$_DISK, $(_gb "$_COMBOS") GB of all-modalities combos"
    fi
    if [ "$RUN_EVAL" = true ]; then
        _EVALS=$( ( set +u; source "$SLURM_DIR/eval_finetuning.conf" >/dev/null 2>&1; \
                    echo "${#EVAL_CV_TYPES[@]}" ) 2>/dev/null )
        case "$_EVALS" in ''|*[!0-9]*|0) _EVALS=3 ;; esac
        _ARM_COST="$_ARM_COST, $(( _EVALS * ${#SIAMESE_VARIANTS[@]} )) eval tasks"
    fi
    if [ "$RUN_PAIR" = true ]; then
        _ARM_GB=$(( _ARM_GB + 42 )); _DISK="$_DISK, 42 GB of pair export (PAIR_EXPORT_* defaults)"
    fi
    if [ "$RUN_ABLATION" = true ]; then
        _LOCO=$( ( source "$SLURM_DIR/run_ablation.conf" >/dev/null 2>&1 \
                   && echo $(( ${#ABLATE_CATEGORIES[@]} * ${#CV_TYPES[@]} * ${#SIAMESE_VARIANTS[@]} )) \
                   "${ABLATION_PURGE_PREDICTIONS:-0}" ) 2>/dev/null ) || _LOCO=""
        if [ -n "$_LOCO" ]; then
            _ARM_COST="$_ARM_COST, ${_LOCO%% *} leave-one-category-out tasks"
            if [ "${_LOCO#* }" = 1 ]; then
                _DISK="$_DISK, no ablation fold matrices (ABLATION_PURGE_PREDICTIONS=1)"
            else
                _ARM_GB=$(( _ARM_GB + $(_gb "${_LOCO%% *}") ))
                _DISK="$_DISK, $(_gb "${_LOCO%% *}") GB of ablation fold matrices"
            fi
        fi
    fi
    # The cell-line ablation (step 3d) keeps its zeroed models' fold matrices
    # (grade.sh deletes none): one set per per_head variant x CV, under the
    # condition of its sbatch.
    _NPH=0
    for _v in "${SIAMESE_VARIANTS[@]}"; do
        case "$_v" in *:per_head) _NPH=$((_NPH + 1)) ;; esac
    done
    if [ "$RUN_CELL" = true ] && [ "${USE_CELL_LINES:-1}" = 1 ] && [ "$_NPH" -gt 0 ]; then
        _ARM_GB=$(( _ARM_GB + $(_gb $(( _NPH * ${#CV_TYPES[@]} ))) ))
        _DISK="$_DISK, $(_gb $(( _NPH * ${#CV_TYPES[@]} ))) GB of cell-line ablation fold matrices"
    else
        _NPH=0
    fi
    _OWNER_COST="$BESTCAT_N Step-1 tasks"
    _OWNER_GB=$(_gb $(( ${#EMB_CATALOG[@]} * ${#CV_TYPES[@]} )))
    # "<tasks> <types>": the Step-1 tasks of the catalog types whose absence
    # generation accepts (run_generate_embeddings.conf OPTIONAL_TYPES), which
    # SKIP, keeping nothing, when their file is absent. (No `case` in here:
    # bash 3.2 cannot parse one inside $( ).)
    _OPT=$( ( source "$SLURM_DIR/run_generate_embeddings.conf" >/dev/null 2>&1 || exit 0
              _o=" ${OPTIONAL_TYPES[*]} "; _n=0; _t=""
              for _e in "${EMB_CATALOG[@]}"; do
                  _x=" ${_e%%:*} "
                  if [ "${_o/$_x/}" != "$_o" ]; then _n=$((_n + 1)); _t="$_t ${_e%%:*}"; fi
              done
              echo "$(( _n * ${#CV_TYPES[@]} ))$_t" ) 2>/dev/null ) || _OPT=""
    _n=${_OPT%% *}
    case "$_n" in ''|*[!0-9]*) _n=0; _OPT="" ;; esac
    [ "$_n" -gt 0 ] || _OPT=""
    _OWNER_MIN=$(_gb $(( ${#EMB_CATALOG[@]} * ${#CV_TYPES[@]} - _n )))
    # The figure is the fold matrices and the pair export the run keeps; what
    # else it writes is named under it (2026-10-21: "up to 910 GB" read as a
    # bound on the whole run, which every default run goes over): the eval's
    # per-pair CSVs (eval_finetuning.py, about 5 GB per arm at the default
    # variants and screens), the three peers' fold matrices (once, with the
    # shared array: 3 x |CVS| runs of NUM_FOLDS matrices) and the checkpoints
    # (combos, Step 1, the ablations' unless purged, the fine-tuned models).
    _UNCOUNTED=""
    [ "$RUN_EVAL" != true ] || _UNCOUNTED="the eval's per-pair CSVs (about 5 GB per arm by default)"
    if [ "$RUN_SHARED" = true ]; then
        _ncv=0
        for _c in ${CVS:-cv1 cv2 cv3}; do _ncv=$((_ncv + 1)); done
        _UNCOUNTED="${_UNCOUNTED:+$_UNCOUNTED, }the peers' fold matrices (about $(_gb $(( 3 * _ncv ))) GB, once)"
    fi
    _UNCOUNTED="${_UNCOUNTED:+$_UNCOUNTED and }the checkpoints"
    unset -f _gb
    unset _n _ncv _c
    # what the siamese side queues: one Step-1 array per owner; per arm the
    # combo, the eval array, the ablation array and its summary, the
    # cell-line ablation and the export
    _count "$_N_OWNERS" $(( BESTCAT_N * _N_OWNERS ))
    _count "$_N_ARMS" "$_N_ARMS"
    [ "$ALLMOD_COMBO" != 1 ] || _count "$_N_ARMS" $(( ${#CV_TYPES[@]} * _N_ARMS ))
    # the bilinear probe: arm none only, with pd:pooled among the variants
    if [[ " ${PCA_ARM_LIST[*]} " == *" none "* ]] && \
            [[ " ${SIAMESE_VARIANTS[*]} " == *" pd:pooled "* ]]; then
        _count 1 "${#CV_TYPES[@]}"
    fi
    [ "$RUN_EVAL" != true ] || _count "$_N_ARMS" $(( _EVALS * ${#SIAMESE_VARIANTS[@]} * _N_ARMS ))
    if [ "$RUN_ABLATION" = true ] && [ -n "${_LOCO:-}" ]; then
        _count $(( 2 * _N_ARMS )) $(( (${_LOCO%% *} + 1) * _N_ARMS ))
    fi
    [ "$_NPH" = 0 ] || _count "$_N_ARMS" "$_N_ARMS"
    [ "$RUN_PAIR" != true ] || _count "$_N_ARMS" "$_N_ARMS"
fi
# ... and the rest: the env reset and the generation once, the shared folds
# job and its 3 x |CVS| array once, per arm the compare, the external transfer
# and the paper tables, and the table across the arms
[ "$RESET_ENV" != true ] || _count 1 1
if [ "$RUN_SIAMESE" = true ] && [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    _count 1 1
fi
[ "$RUN_SHARED" != true ] || _count 2 $(( 1 + 3 * $(echo ${CVS:-cv1 cv2 cv3} | wc -w) ))
# ELISL's array (one task per CV of CVS), and its input build when needed
[ "$RUN_ELISL" != true ] || _count 1 "$(echo ${CVS:-cv1 cv2 cv3} | wc -w)"
[ "$ELISL_BUILD" != true ] || _count 1 1
_count "$_N_ARMS" "$_N_ARMS"
[ "$RUN_EXT" != true ] || _count "$_N_ARMS" "$_N_ARMS"
[ "$RUN_PT" != true ] || _count "$_N_ARMS" "$_N_ARMS"
if [ "$RUN_PT" = true ] && [ "$_N_ARMS" -ge 2 ]; then _count 1 1; fi
unset -f _count

echo "============================================================================"
echo "  SL benchmark: SLeuth (siamese_sl, ${#SIAMESE_VARIANTS[@]} variants: ${SIAMESE_VARIANTS[*]})"
echo "  vs SLMGAE / SLGNN / NSF4SL / DegreePrior, and the paper tables"
echo "  SLDB 3.0 given pos/neg | shared folds | seed $SEED | run token $RUN_TOKEN$_TOKEN_NOTE"
echo "  PCA arms ($_N_ARMS): ${PCA_ARM_LIST[*]}, each the whole siamese pipeline and its own tables:"
for _a in "${PCA_ARM_LIST[@]}"; do
    printf '    %-13s %s\n' "$_a" "$(pca_arm_label "$_a")"
    printf '    %-13s siamese_sl/%s/ | %s/\n' "" "$(pca_arm_results "$_a")" "$(pca_arm_cmp "$_a")"
done
if [ "$RUN_SIAMESE" = true ]; then
    for _o in $_OWNERS; do
        printf '    %-13s siamese_sl/%s/: %s\n' "(Step 1)" "$(pca_arm_results "$_o")" "$(pca_step1_label "$_o")"
    done
fi
if [ "$RUN_PT" = true ] && [ "$_N_ARMS" -ge 2 ]; then
    echo "    across them: sl_comparison/results/pca_arms/ (reference arm ${PCA_ARM_LIST[0]})"
fi
if [ "$_PER_ARM_TOKENS" = true ]; then
    for _a in "${PCA_ARM_LIST[@]}"; do
        _v="_ARM_TOKEN_$_a"
        echo "    PCA arm $_a: RUN_TOKEN=${!_v}"
    done
fi
[ -z "$_ARM_COST" ] || echo "  per PCA arm: $_ARM_COST"
[ -z "$_OWNER_COST" ] || [ "$RUN_SIAMESE" != true ] || echo "  per owner ($_N_OWNERS): $_OWNER_COST"
echo "  queues: $_NJOBS jobs, $_NTASKS tasks counting every array task (Slurm's MaxSubmitJobs counts"
echo "    each: sacctmgr show assoc user=\$USER format=MaxSubmit,GrpSubmit; the QOS's MaxSubmitPU)"
if [ "$RUN_SIAMESE" = true ]; then
    _KEPT="the fold matrices"
    [ "$RUN_PAIR" != true ] || _KEPT="$_KEPT and pair export"
    echo "  disk of $_KEPT it keeps, up to: $(( _N_ARMS * _ARM_GB + _N_OWNERS * _OWNER_GB )) GB in all; per arm $_DISK;"
    if [ -n "$_OPT" ]; then
        echo "    per owner up to $_OWNER_GB GB of single-embedding fold matrices ($_OWNER_MIN GB when the"
        echo "    optional ${_OPT#* } are absent: their ${_OPT%% *} Step-1 tasks skip; then"
        echo "    $(( _N_ARMS * _ARM_GB + _N_OWNERS * _OWNER_MIN )) GB in all)"
    else
        echo "    per owner up to $_OWNER_GB GB of single-embedding fold matrices"
    fi
    echo "    not counted: $_UNCOUNTED"
fi
echo "============================================================================"
unset _a _o _v _LOCO _EVALS _COMBOS _NPH _OPT _OWNER_MIN _KEPT _UNCOUNTED
if [ -n "$_INHERITED_RUN_TOKEN" ] && [ "$_INHERITED_RUN_TOKEN" != "$RUN_TOKEN" ]; then
    echo "NOTE: replaced RUN_TOKEN=$_INHERITED_RUN_TOKEN from the environment (each submission gets its own)."
fi
# One line per setting inherited from the calling shell (recorded at the top;
# the banner above names the variants, the arms, the seed and the token only).
[ -z "$_ENV_NOTES" ] || printf '%s' "$_ENV_NOTES"
# train.py's parser is the judge of ARCH_ARGS, PCA_METHOD,
# PREPROCESSING_FIT_SCOPE, SIAMESE_ENCODER_TYPE, CELL_LINE_DIM and the PCA arms'
# PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE / PCA_JOINT_DIM (config.conf takes
# each from the environment and checks only their form), and no job runs it
# before reset_env and generation have finished: a value it refuses fails
# every Step-1 task hours in (an arm's multi-modal PCA: every combo of the arm,
# after the whole Step-1 array), and the combo job's failure cancels eval,
# ablation and the cell-line ablation. Ask it here: --print_arch parses these
# settings as the jobs will pass them and exits without loading data or
# writing anything, once per DISTINCT PCA flag set of the arms this DAG trains
# (config.conf pca_single_args, the Step-1 runs', and pca_multi_args, the
# combos' and the ablation's: four sets for the four arms, one for arm none
# alone). Only argparse's own refusal (exit status 2 and "train.py: error:
# ...") stops the submission; any other failure (a module this node lacks) is
# only reported, once, so this check cannot stop a run the jobs would accept.
if [ "$RUN_SIAMESE" = true ]; then
    _PCA_SETS=""
    for _a in "${PCA_ARM_LIST[@]}"; do
        for _PP in "$(pca_single_args "$_a")" "$(pca_multi_args "$_a")"; do
            case "|$_PCA_SETS|" in
                *"|$_PP|"*) ;;
                *) _PCA_SETS="${_PCA_SETS:+$_PCA_SETS|}$_PP" ;;
            esac
        done
    done
    IFS='|' read -r -a _PCA_SET_LIST <<< "$_PCA_SETS"
    for _PP in "${_PCA_SET_LIST[@]}"; do
        _ARCH_RC=0
        _ARCH_ERR=$(cd "$SIAMESE_DIR" && PYTHONDONTWRITEBYTECODE=1 "$PYTHON_PATH" train.py \
            --embeddings_paths "$DATA_DIR/all_genes_esm2.pt" --sl_path "$SL_PATH" \
            ${NEG_PATH:+--neg_pairs_path "$NEG_PATH"} $CELL_LINE_ARGS \
            --output_dir results/_sigcheck --cv_type cv1 $MODEL_ARGS \
            --encoder_dims $ENCODER_DIMS --l1_lambdas $L1_LAMBDAS --seed "$SEED" \
            --preprocessing_fit_scope "$PREPROCESSING_FIT_SCOPE" \
            --pca_method "$PCA_METHOD" \
            --siamese_encoder_type "$SIAMESE_ENCODER_TYPE" $_PP \
            $(siamese_variant_args "$SELECTION_VARIANT") $SELECTION_ARGS $ARCH_ARGS \
            --print_arch 2>&1 >/dev/null) || _ARCH_RC=$?
        [ "$_ARCH_RC" != 0 ] || continue
        _ARCH_ERR=$(printf '%s\n' "$_ARCH_ERR" | tail -n 1)
        # the arms that train with this flag set
        _PP_ARMS=""
        for _a in "${PCA_ARM_LIST[@]}"; do
            if [ "$(pca_single_args "$_a")" = "$_PP" ] || [ "$(pca_multi_args "$_a")" = "$_PP" ]; then
                _PP_ARMS="${_PP_ARMS:+$_PP_ARMS }$_a"
            fi
        done
        case "$_ARCH_RC:$_ARCH_ERR" in
            "2:train.py: error:"*)
                echo "ERROR: train.py refuses the settings every siamese training of PCA arm(s)"
                echo "  $_PP_ARMS would be started with (ARCH_ARGS, PCA_METHOD, PREPROCESSING_FIT_SCOPE,"
                echo "  SIAMESE_ENCODER_TYPE, CELL_LINE_DIM, SEED, and the arm's PCA flags"
                echo "  '$_PP' from PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE / PCA_JOINT_DIM:"
                echo "  siamese_sl/slurm/config.conf, each one overridable from the environment):"
                echo "    $_ARCH_ERR"
                echo "  Unset or correct it, then resubmit. Nothing was changed or submitted."
                exit 1 ;;
            *)
                echo "WARNING: could not check the training settings on this node (train.py"
                echo "  --print_arch exited $_ARCH_RC: $_ARCH_ERR); the first training job will."
                break ;;
        esac
    done
    unset _PCA_SETS _PCA_SET_LIST _PP _PP_ARMS _ARCH_RC _ARCH_ERR _a
fi

# ---- original SLGNN / NSF4SL / SLMGAE inputs, before ANY submission --------
# Checked before anything is submitted (not only inside the array tasks), so a
# bad input stops the whole run instead of leaving the siamese tree queued
# and failing the SLGNN / NSF4SL / SLMGAE GPU tasks after the folds job.
# Existence AND provenance: SynLethKG's meta.json must carry
# relation_checks.label_keyed_deletion and a mapping_sha256 matching the
# current SynLethDB exports / HGNC table / NCBI gene_info and
# SLGNN/build_synlethkg.py itself, and the TransE file
# a meta.mapping_sha256 matching the current HGNC table / correction table /
# normaliser / builder (a file from an older build attaches some KG genes to
# other Entrez rows, and nothing downstream compares it; the check hashes both
# builders, hence NEED_CODE). SLMGAE's four files are
# its authors' support views (GO-BP, GO-CC, PPI) and the List_Proteins index
# they are keyed by: without
# one, SLMGAE is not the published model. Runs before reset_env is submitted
# (it needs this venv's torch).
if [ "$RUN_SHARED" = true ]; then
    echo "--- SLGNN / NSF4SL / SLMGAE inputs (existence + builder provenance) ---"
    if ! "$PYTHON_PATH" "$ROOT/sl_comparison/slurm/check_shared_inputs.py"; then
        echo "ERROR: rebuild or copy the inputs named above on this node, then resubmit"
        echo "  (see the Env note at the top of this script; --skip-shared runs siamese only)."
        exit 1
    fi
fi
# ELISL's inputs (data/elisl/gene_features.npz + meta.json, by
# ELISL/build_elisl_inputs.py): current ones are reused; otherwise this
# submission builds them in a GPU job (elisl_inputs) before ELISL trains. That
# job needs the downloads (fetched HERE, on the login node, which has the
# internet; present files are kept) and SEQVEC_BIN, a seqvec==0.4.1 in its own
# Python 3.7 environment (ELISL/README.md, Inputs), checked here so a missing
# one stops the run before anything changes, not days later.
if [ "$RUN_ELISL" = true ]; then
    echo "--- ELISL inputs (data/elisl) ---"
    [ -z "$_ELISL_CHECK" ] || printf '%s\n' "$_ELISL_CHECK" | sed 's/^/  /'
    if [ "$ELISL_BUILD" != true ]; then
        echo "  current: reused (no elisl_inputs job)"
    else
        export SEQVEC_BIN="${SEQVEC_BIN:-<home>/seqvec_env/bin/seqvec}"
        if [ ! -x "$SEQVEC_BIN" ] || ! "$SEQVEC_BIN" --help >/dev/null 2>&1; then
            echo "ERROR: ELISL's inputs must be built, and SEQVEC_BIN=$SEQVEC_BIN is not a working"
            echo "  seqvec==0.4.1. Once, with an installed Python 3.7 (conda/micromamba python=3.7;"
            echo "  uv ships none):"
            echo "    uv venv --python /path/to/python3.7 <home>/seqvec_env"
            echo "    uv pip install --python <home>/seqvec_env/bin/python seqvec==0.4.1 overrides==3.1.0"
            echo "  (or export SEQVEC_BIN=<another seqvec>), then resubmit; --skip-elisl runs without"
            echo "  ELISL (its previous rows are kept). Nothing was changed or submitted."
            exit 1
        fi
        echo "  to build: downloading what is missing (UniProt, STRING v11.0, SeqVec weights) ..."
        if ! (cd "$ROOT" && "$PYTHON_PATH" ELISL/build_elisl_inputs.py download); then
            echo "ERROR: the ELISL download failed (above); rerun it here, or resubmit with"
            echo "  --skip-elisl. Nothing else was changed or submitted."
            exit 1
        fi
        echo "  an elisl_inputs job (SeqVec, node2vec, features; $SEQVEC_BIN) runs before ELISL"
    fi
fi

# ---- 0. shared folds, BEFORE anything else ---------------------------------
# siamese takes its train/val/test partition from sl_comparison/folds
# (SHARED_FOLDS in siamese_sl/slurm/config.conf), so the fold set must exist
# before the siamese array job starts. The siamese and shared-model jobs are
# submitted as siblings with no ordering edge between them, so generating it
# inside run_shared_folds.sh alone would race, or fail every siamese task with
# "folds_dir has no genes.txt". folds/ is regenerated here on every run and
# then kept read-only while the DAG runs ("0, freeze"): it is git-tracked and
# travels with every upload of the tree. Doing it here on the login node is
# deterministic and costs seconds (pure numpy, no GPU).
# run_shared_folds.sh then finds it fresh (meta.json, contents and source
# fingerprints match) and reuses it. Under SHARED_FOLDS_FROZEN=1 it never
# rewrites the set, because the siamese jobs read these same files
# concurrently: fold files that differ from FOLDS_MANIFEST are FATAL, and a
# stale source or default with byte-identical fold files is only a WARNING
# (paper_tables then reports PAPER INCOMPLETE).
#
# THIS MUST COME BEFORE reset_env IS SUBMITTED. It runs on the login node with
# $PYTHON_PATH, which lives inside the very venv reset_env.sh does `rm -rf` on
# ($VENV_DIR is the venv ROOT, two dirnames up from <venv>/bin/python3 — the
# whole tree goes, not just bin/). Submitted first, reset_env can be
# allocated and start deleting the interpreter while prepare_folds.py is still
# running — killing it mid-write. prepare_folds.py removes the old fold dirs
# and meta.json first and writes meta.json last, so the half-written tree is
# detectably incomplete, but every job that reads it before a regeneration
# fails. Generating first closes the window entirely.
# prepare_folds.py reads the SynLethDB pairs and their .sources.tsv sidecars
# here, before any job exists; the negatives .txt is git-ignored (*.txt) and
# only the generation job downloads it, so a git-synced checkout can lack it.
_PAIRS_MISSING=0
for _pf in "$ROOT/data/SL_SynLethDB_experimental.txt" \
           "$ROOT/data/SL_SynLethDB_experimental.sources.tsv" \
           "$ROOT/data/SL_SynLethDB_experimental_negatives.txt" \
           "$ROOT/data/SL_SynLethDB_experimental_negatives.sources.tsv"; do
    [ -f "$_pf" ] || { echo "ERROR: ${_pf#$ROOT/} missing (step 0 builds the folds from it)"; _PAIRS_MISSING=1; }
done
if [ "$_PAIRS_MISSING" = 1 ]; then
    echo "  Copy it from the machine that has it, or rebuild pairs + sidecar:"
    echo "  (cd siamese_sl && $PYTHON_PATH download_synlethdb.py --output ../data/SL_SynLethDB_experimental.txt --cache_dir ../data/embeddings_cache)"
    echo "  (cd siamese_sl && $PYTHON_PATH download_synlethdb.py --relation NONSL --output ../data/SL_SynLethDB_experimental_negatives.txt --cache_dir ../data/embeddings_cache)"
    exit 1
fi
# Existence is not enough: a sidecar that lost rows (an upload or a download
# that stopped part-way) names no cell line for the pairs it lost, and every
# per_head training would supervise them at the OTHER head (train.py refuses
# it too, but only when the first training task starts, hours from now).
if ! "$PYTHON_PATH" - "$ROOT/data/SL_SynLethDB_experimental" \
        "$ROOT/data/SL_SynLethDB_experimental_negatives" <<'PYEOF'
import sys
bad = 0
for stem in sys.argv[1:]:
    with open(stem + ".txt") as f:
        txt = {frozenset(p[:2]) for p in (ln.split() for ln in f) if len(p) >= 2}
    with open(stem + ".sources.tsv") as f:
        next(f, None)  # header
        side = {frozenset(t[:2]) for t in (ln.rstrip("\r\n").split("\t") for ln in f)
                if len(t) >= 4}
    lost = len(txt - side)
    if lost:
        print(f"ERROR: {lost:,} pair(s) of {stem}.txt have no row in "
              f"{stem}.sources.tsv (out of sync: a partial copy?)")
        bad = 1
sys.exit(bad)
PYEOF
then
    echo "  Re-copy the pair file and its sidecar, or rebuild both with download_synlethdb.py."
    echo "  Nothing was changed or submitted."
    exit 1
fi
# The external screens eval_finetuning.py reads (DATASETS, relative to the repo
# root; git-ignored except Corn). eval_finetuning.py refuses a missing one
# (dataset_input_problems -> exit 1, its previous outputs untouched), but only
# once its task starts, days of generation and training later, and
# run_external_transfer.sh (afterany) then drops every variant. Check here so
# the DAG is never submitted without them.
if [ "$RUN_SIAMESE" = true ] && [ "$RUN_EVAL" = true ]; then
    _SCREENS_MISSING=0
    while IFS= read -r _sf; do
        [ -n "$_sf" ] || continue
        case "$_sf" in /*) _sp="$_sf" ;; *) _sp="$ROOT/$_sf" ;; esac
        [ -f "$_sp" ] || { echo "ERROR: external screen '$_sf' missing (eval_finetuning.conf DATASETS)"; _SCREENS_MISSING=1; }
    done < <( ( source "$SLURM_DIR/eval_finetuning.conf" >/dev/null 2>&1 || true
                for _d in "${DATASETS[@]}"; do IFS=: read -r _ _p _ <<< "$_d"; echo "$_p"; done ) 2>/dev/null )
    if [ "$_SCREENS_MISSING" = 1 ]; then
        echo "  Copy it into this checkout (git-ignored: a pull never brings it), or pass --skip-eval."
        exit 1
    fi
fi
# scGPT, a required catalog modality, is rebuilt from a hand-placed checkpoint
# folder (git-ignored, never downloaded by a job; generate_embeddings.py looks
# in these three places): the generation job deletes the cached pkl first, so
# without it the job fails its end-of-run catalog check hours in and cancels
# the DAG.
_SCGPT_REQUIRED=false
if [ "$RUN_SIAMESE" = true ] && [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    case " ${EMB_CATALOG[*]} " in *" scgpt:"*) _SCGPT_REQUIRED=true ;; esac
    _OPTIONAL=$( ( source "$SLURM_DIR/run_generate_embeddings.conf" >/dev/null 2>&1 || true
                   echo " ${OPTIONAL_TYPES[*]} " ) 2>/dev/null )
    case "$_OPTIONAL" in *" scgpt "*) _SCGPT_REQUIRED=false ;; esac
fi
if [ "$_SCGPT_REQUIRED" = true ]; then
    _SCGPT_OK=false
    for _d in "$CACHE_DIR/scGPT_human" "$CACHE_DIR/scgpt_human" "$CACHE_DIR"; do
        [ -f "$_d/best_model.pt" ] && [ -f "$_d/vocab.json" ] && _SCGPT_OK=true
    done
    if [ "$_SCGPT_OK" = false ]; then
        echo "ERROR: no scGPT checkpoint (best_model.pt + vocab.json) in ${CACHE_DIR#$ROOT/}/scGPT_human/;"
        echo "  the generation job rebuilds scgpt (a required catalog embedding) from it."
        echo "  Copy the folder into this checkout (git-ignored), or:"
        # -O without a trailing slash writes the folder's files into that dir
        # (without -O they would land in ./<Drive folder name>).
        echo "    pip install gdown && gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O \"$CACHE_DIR/scGPT_human\""
        exit 1
    fi
fi

# ---- SLURM reachable, and no earlier submission still live -----------------
# Before step 0 rewrites sl_comparison/folds and step 0a deletes the previous
# run's outputs, so a node without SLURM or an unreachable controller changes
# nothing. Every job of this pipeline (and of siamese_sl/slurm/
# submit_pipeline.sh) runs config.conf's venv, which reset_env.sh rm -rf's,
# and reads sl_comparison/folds, which step 0 deletes and rewrites: submitted
# while an earlier submission is queued or running (e.g. --skip-siamese after
# its shared array failed), this run breaks that one. --after's JOB is ordered.
# SLGNN / NSF4SL below are the standalone baseline launchers
# (SLGNN/slurm/run_slgnn.sh, NSF4SL/slurm/run_nsf4sl.sh): same venv, same
# --folds_dir ../sl_comparison/folds, and they start a FRESH interpreter per CV,
# so the venv delete aborts their remaining CVs under `set -e`. The jobs THIS
# script submits are named sl_compare / sl_folds / ..., never SLGNN or NSF4SL,
# so these two names cannot match its own submission. (GNN_Adamson/run.sh shares
# the venv but not the folds; not listed.)
if ! command -v sbatch >/dev/null 2>&1 || ! command -v squeue >/dev/null 2>&1; then
    echo "ERROR: sbatch / squeue not on PATH (run this on the SLURM login node);"
    echo "  nothing was changed or submitted."
    exit 1
fi
if ! _QUEUE=$(squeue -h -u "${USER:-$(whoami)}" -o '%i %100j' 2>&1); then
    echo "ERROR: squeue failed (slurmctld unreachable?); nothing was changed or submitted:"
    printf '%s\n' "$_QUEUE" | sed 's/^/  /'
    exit 1
fi
_PIPE_JOBS='^(reset_env|siamese_gen_emb|bench_emb|bench_summary|bestcat_emb|bestcat_combo|eval_finetuning|ablation|ablation_summary|cellline_ablation|bilinear_probe|allmod_combo|ELISL|elisl_inputs|sl_preflight|sl_folds|sl_compare|sl_final_compare|sl_ext_transfer|sl_paper_tables|pair_export|SLGNN|NSF4SL)$'
_LIVE=$(printf '%s\n' "$_QUEUE" | awk -v re="$_PIPE_JOBS" -v after="${AFTER_JOB%%_*}" \
    'NF >= 2 { split($1, id, "_"); if ($2 ~ re && id[1] != after) print "  " $1 " (" $2 ")" }')
if [ -n "$_LIVE" ]; then
    if [ "${ALLOW_LIVE_JOBS:-0}" = 1 ]; then _LVL=WARNING; else _LVL=ERROR; fi
    echo "$_LVL: jobs of an earlier submission (pipeline, or a standalone SLGNN /"
    echo "  NSF4SL baseline) are still queued or running:"
    printf '%s\n' "$_LIVE" | head -20
    echo "  They run from the venv reset_env deletes and read sl_comparison/folds,"
    echo "  which step 0 rewrites."
    if [ "$_LVL" = ERROR ]; then
        echo "  Wait for them or scancel them, then resubmit (ALLOW_LIVE_JOBS=1 submits"
        echo "  anyway; --skip-reset-env keeps their venv). Nothing was changed."
        exit 1
    fi
    echo "  ALLOW_LIVE_JOBS=1: submitting anyway."
fi

# ---- node selection and array caps (config.conf), before anything changes --
# Rejected at their sbatch, they would stop the run only after step 0 and 0a
# had rewritten the folds and removed the previous run's outputs, with part of
# the DAG already queued.
if ! check_throttles || ! check_node_selection; then
    echo "  Nothing was changed or submitted."
    exit 1
fi
# `sbatch --test-only` submits nothing: slurmctld answers whether it would
# accept the job on this node selection (GPU_NODE_ARGS, or each script's own
# #SBATCH --nodelist=gpu3 when empty). Probed with the heaviest request of each
# stage this run submits: shared models 96G / 12 CPUs / 3 days, generation 64G
# / 12 CPUs / 14 days, combo 64G / 12 CPUs / 5 days (reset_env is smaller).
_PROBES=""
[ "$RESET_ENV" = true ] && _PROBES="$_PROBES siamese_sl/slurm/reset_env.sh"
if [ "$RUN_SIAMESE" = true ]; then
    if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
        _PROBES="$_PROBES siamese_sl/slurm/run_generate_embeddings.sh"
    fi
    _PROBES="$_PROBES siamese_sl/slurm/run_best_per_category_combo.sh"
    [ "$ALLMOD_COMBO" != 1 ] || _PROBES="$_PROBES siamese_sl/slurm/run_allmod_combo.sh"
fi
[ "$RUN_SHARED" = true ] && _PROBES="$_PROBES sl_comparison/slurm/run_shared_models.sh"
[ "$RUN_ELISL" = true ] && _PROBES="$_PROBES ELISL/slurm/run_elisl.sh"
# (no quote characters inside ${VAR:-...}: bash 3.2 parses them there)
if [ -n "$GPU_NODE_ARGS" ]; then _SEL="$GPU_NODE_ARGS"
else _SEL="each script's #SBATCH --nodelist=gpu3"; fi
# A down or drained node is no submission error: the whole DAG would just pend.
# Its state is read first because sbatch --test-only cannot place a job on
# such a node and fails: that failure is then the WARNING below, not a refusal.
_STATE=""; _NODE_DOWN=""; _PROBED="sbatch --test-only ok:${_PROBES:- nothing to probe}"
if command -v sinfo >/dev/null 2>&1; then
    _STATE=$(sinfo -h -N -n "${GPU_NODES:-gpu3}" -o '%N %t' 2>/dev/null | sort -u) || _STATE=""
    if [ -n "$_STATE" ] && printf '%s\n' "$_STATE" | awk '{print $2}' \
         | grep -Eq '^(down|drain|drng|fail|maint|inval)|[*]$'; then
        _NODE_DOWN=1
    fi
fi
for _p in $_PROBES; do
    if ! _TEST=$(sbatch --test-only $GPU_NODE_ARGS "$ROOT/$_p" 2>&1); then
        if [ -n "$_NODE_DOWN" ]; then
            _PROBED="not probed: sbatch --test-only cannot place a job on the node state below"
            break
        fi
        echo "ERROR: SLURM would reject $_p on $_SEL:"
        printf '%s\n' "$_TEST" | sed 's/^/  /'
        echo "  Nothing was changed or submitted."
        exit 1
    fi
done
echo "  node selection: $_SEL ($_PROBED)"
if [ -n "$_STATE" ]; then
    printf '%s\n' "$_STATE" | sed 's/^/  node state: /'
    if [ -n "$_NODE_DOWN" ]; then
        echo "WARNING: a node this DAG is pinned to is down, drained, failing or not"
        echo "  responding: its jobs pend until it is back (GPU_NODES moves the whole DAG)."
    fi
fi

# The log keeps the per-fold counts prepare_folds prints (provenance).
_T0=$(date +%Y%m%dT%H%M%S)
FOLDS_LOG="$CMP_LOGS/prepare_folds_$_T0.log"
echo "--- shared folds (login node, seed $SEED; log $FOLDS_LOG) ---"
# folds_manifest: "<sha256>  <path>" per file under sl_comparison/folds.
. sl_comparison/slurm/folds_manifest.sh
_FOLDS_BEFORE=$(folds_manifest)
# The previous run left the fold set read-only (below); prepare_folds.py
# deletes and rewrites it.
if [ -d sl_comparison/folds ] && ! chmod -R u+w sl_comparison/folds; then
    echo "ERROR: cannot make sl_comparison/folds writable for step 0; nothing was"
    echo "  changed or submitted."
    exit 1
fi
if ! ( set -o pipefail
       $PYTHON_PATH sl_comparison/prepare_folds.py --seed "$SEED" --verify 2>&1 \
           | tee "$FOLDS_LOG" ); then
    echo "FATAL: fold generation failed — every model reads these folds; aborting"
    exit 1
fi
export SHARED_FOLDS_FROZEN=1

# ---- 0, freeze: the answer key is read-only while this DAG runs ------------
# On 2026-09-23 a whole-tree SFTP upload to the cluster checkout (12:10-12:18,
# after the 11:38 submission) replaced meta.json with the uploading machine's
# older copy; the folds job then refused it (SHARED_FOLDS_FROZEN) and SLGNN /
# NSF4SL / SLMGAE and DegreePrior were cancelled. No job writes under
# sl_comparison/folds, so it is made read-only (files and directories) until
# the next step 0, or a standalone run_shared_folds.sh regeneration, makes it
# writable again: an upload, checkout or copy onto it now fails loudly instead
# of changing the answer key under the running jobs. The manifest records the
# exact fold set this run used.
# Timestamped: a --skip-siamese rerun keeps the reused run's RUN_TOKEN, and
# must not overwrite that run's record. Exported: under SHARED_FOLDS_FROZEN the
# folds job refuses only a fold set that differs from it (run_shared_folds.sh).
FOLDS_MANIFEST="$CMP_LOGS/folds_${RUN_TOKEN}_$_T0.sha256"
folds_manifest > "$FOLDS_MANIFEST"
export FOLDS_MANIFEST
if chmod -R a-w sl_comparison/folds; then
    echo "  sl_comparison/folds: read-only until the next step 0 (manifest $FOLDS_MANIFEST)"
else
    echo "WARNING: could not make sl_comparison/folds read-only: nothing stops an"
    echo "  upload from replacing it while this DAG runs (manifest $FOLDS_MANIFEST)."
fi
_FOLDS_CHANGED=$(printf '%s\n' "$_FOLDS_BEFORE" | diff - "$FOLDS_MANIFEST" \
                     | sed -n 's/^[<>] [0-9a-f]*  //p' | sort -u)
if [ -n "$_FOLDS_CHANGED" ]; then
    echo "NOTE: step 0 changed $(printf '%s\n' "$_FOLDS_CHANGED" | wc -l | tr -d ' ') file(s) of sl_comparison/folds, e.g."
    printf '%s\n' "$_FOLDS_CHANGED" | head -5 | sed 's/^/    /'
    echo "  Download this fold set with the results and commit it on the laptop, so"
    echo "  the repo holds the answer key this run is graded on (the download can"
    echo "  carry the read-only modes: chmod -R u+w sl_comparison/folds there first)."
fi
echo "  Never upload sl_comparison/folds/, data/ (bar the inputs this script names"
echo "  when one is missing, copied in once while no DAG runs), */results/ or"
echo "  */slurm/logs/ to this checkout, and never in mirror/delete mode. While this"
echo "  DAG is queued or running, upload nothing if you can avoid it: jobs starting"
echo "  hours from now read its code and config. The SLGNN / NSF4SL tasks refuse a"
echo "  changed SLGNN/build_synlethkg.py, NSF4SL/build_transe_features.py,"
echo "  siamese_sl/gene_name_utils.py or SLMGAE-in-pytorch/gene_corrections_config.py;"
echo "  a changed siamese_sl/cell_line_vocab.py, sl_comparison/prepare_folds.py, data/"
echo "  pair file or sidecar, or slurm .conf is NOT refused (later jobs read the new"
echo "  version) and only makes paper_tables report PAPER INCOMPLETE."

# ---- 0, code: the code every job of this run executes ----------------------
# Jobs start over days and re-read config and code at start, while the batch
# scripts are the ones queued now: on 2026-09-23 an upload 32 min after the
# submission left the queued scripts one version and everything they sourced
# another, with nothing recording it (git does not run inside the jobs).
# paper_tables re-hashes these files when it runs (the pair export starts
# after it and is not covered) and lists every one that
# changed, appeared or vanished (PAPER INCOMPLETE until checked).
CODE_MANIFEST="$CMP_LOGS/code_${RUN_TOKEN}_$_T0.sha256"
if "$PYTHON_PATH" sl_comparison/code_manifest.py > "$CODE_MANIFEST"; then
    export CODE_MANIFEST
    echo "  code manifest: $CODE_MANIFEST ($(wc -l < "$CODE_MANIFEST" | tr -d ' ') files)"
else
    echo "WARNING: could not write the code manifest: paper_tables will not check"
    echo "  whether code changed during this run."
    rm -f "$CODE_MANIFEST"; CODE_MANIFEST=""
fi
# --skip-siamese reuses the siamese outputs of an earlier submission under its
# RUN_TOKEN, while the manifest above records the code as it is NOW, and
# paper_tables compares with that one alone: code uploaded between the two
# submissions is in no list. The oldest manifest carrying a token is that
# earlier submission's (each --skip-siamese rerun adds a newer one): say here
# what changed since, for the token of every reused arm (one, unless the arms
# were run by different submissions). A WARNING only; the verdict is not
# touched.
if [ "$RUN_SIAMESE" = false ] && [ -n "$CODE_MANIFEST" ]; then
    for _tok in $_REUSED_TOKENS; do
        # the arms of that token, named only when the arms' tokens differ
        _WHO=""
        if [ "$_PER_ARM_TOKENS" = true ]; then
            for _a in "${PCA_ARM_LIST[@]}"; do
                _v="_ARM_TOKEN_$_a"
                [ "${!_v}" = "$_tok" ] && _WHO="${_WHO:+$_WHO }$_a"
            done
            _WHO=" (PCA arm(s) $_WHO)"
        fi
        _OLD_CM=$(ls "$CMP_LOGS"/code_"${_tok}"_*.sha256 2>/dev/null | sort | head -n 1)
        if [ -n "$_OLD_CM" ] && [ "$_OLD_CM" != "$CODE_MANIFEST" ]; then
            _OLD_DRIFT=$("$PYTHON_PATH" sl_comparison/code_manifest.py --check "$_OLD_CM" || true)
            if [ -n "$_OLD_DRIFT" ]; then
                echo "WARNING: $(printf '%s\n' "$_OLD_DRIFT" | wc -l | tr -d ' ') code file(s) changed since run $_tok was first recorded$_WHO"
                echo "  (${_OLD_CM#$ROOT/}). The reused siamese combos, eval, ablations and"
                echo "  cell-line outputs were computed by the earlier versions; paper_tables checks"
                echo "  only against this submission's manifest and will NOT list these:"
                printf '%s\n' "$_OLD_DRIFT" | head -20 | sed 's/^/    /'
                _N_DRIFT=$(printf '%s\n' "$_OLD_DRIFT" | wc -l | tr -d ' ')
                if [ "$_N_DRIFT" -gt 20 ]; then
                    echo "    ... (the first 20 of $_N_DRIFT; all of them:"
                    echo "    \"$PYTHON_PATH\" sl_comparison/code_manifest.py --check ${_OLD_CM#$ROOT/})"
                fi
            fi
        else
            # No manifest but this submission's own: the reused run was not
            # submitted by this script (submit_pipeline.sh writes none) or its
            # logs are gone. Silence would read as "nothing changed", and this
            # manifest (named after RUN_TOKEN, the first arm's) is the oldest of
            # the token from now on.
            echo "NOTE: no earlier code manifest carries run token $_tok$_WHO under ${CMP_LOGS#$ROOT/}"
            echo "  (the reused siamese run was not submitted by this script, its token is unknown, or"
            echo "  its logs were removed): code changed since that run cannot be listed here, and a"
            echo "  later --skip-siamese rerun under this token lists only what changed after this one."
        fi
    done
    unset _tok _WHO _a _v _OLD_CM _OLD_DRIFT _N_DRIFT
fi

# ---- 0, SynLethKG: SLGNN's input is read-only while this DAG runs -----------
# Every SLGNN task, each shared task's input check and paper_tables (R2) read
# data/synlethkg; no job writes it. On 2026-09-23 it vanished from the cluster
# during the run (most likely the same mirror upload), which left R2 empty and
# the next submission refused. Read-only from here on, like the folds: an
# upload can no longer delete or replace it. Rebuilding it by hand needs
# `chmod -R u+w data/synlethkg` first (check_shared_inputs.py's hint says so).
# data/nsf4sl_transe_features.pt sits directly in data/, which the generation
# job writes, so no permission can keep an upload from deleting it: only the
# upload rule above protects it.
if [ "$RUN_SHARED" = true ] && [ -d data/synlethkg ]; then
    if chmod -R a-w data/synlethkg; then
        echo "  data/synlethkg: read-only from here on (rebuild by hand only after"
        echo "    chmod -R u+w data/synlethkg)"
    else
        echo "WARNING: could not make data/synlethkg read-only."
    fi
fi


# ---- 0a. previous-run outputs this DAG rebuilds -----------------------------
# Removed (or, for comparison rows, moved aside) only now, after every check
# passed, and before any submission: a stage that fails or never starts
# (cancelled, dependency never satisfied) must leave NO output, never an
# earlier run's, which every reader would take for this run's. The stages
# clean their own outputs too, but only once they start. Per PCA arm of
# PCA_ARMS, in that arm's own roots (siamese_sl/results[/pca_<arm>]/ and
# sl_comparison/results[/pca_<arm>]/, whose rows go into that root's own
# stale_siamese/: every arm's rows have the same names); the shared rows and
# matrices once. An arm this DAG leaves out keeps its siamese outputs and its
# SLeuth rows, but loses its combo markers when this run regenerates the
# embeddings, and its comparison.csv (+ notes), external_transfer/ and
# paper_tables/ when it retrains the shared models (built on the rows moved
# aside; the final listing prints the jobs that rebuild them, or, for an arm
# left without a combo marker, the command that retrains it). None of these
# globs recurses, so arm none's never reaches into another arm's root.
_ts=$(date +%Y%m%dT%H%M%S)
# The arms left out, when this run retrains the shared models (below): those
# whose tables to rebuild, and those whose combos to retrain (and why).
_LEFT_OUT=""; _LEFT_RETRAIN=""
if [ "$RUN_SIAMESE" = true ] && [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    _WHY_RETRAIN="their combos were trained on the embeddings this run replaces (markers removed above)"
else
    _WHY_RETRAIN="a selection without a combo marker (its combos never finished, or a regeneration removed the markers)"
fi
_has_marker() {   # DIR: does it hold a <prefix>latest_combo_cv<k>.txt?
    local f
    for f in "$1"/latest_combo_cv[123].txt "$1"/var_*__latest_combo_cv[123].txt; do
        [ -e "$f" ] && return 0
    done
    return 1
}
# _retrain_recipe INDENT: the command that retrains the arms of _LEFT_RETRAIN
# (step 0a's NOTE and the final listing), what that run does to the table
# across the arms (step 0a removes it, every run's), and what a table across
# every arm with its 2x2 effects takes instead: one submission's selection of
# every arm (a table across the arms lists the arms another submission
# selected as stale, without values), on this run's shared rows. Both
# commands carry the settings this run took from the calling shell
# (_ENV_NOPCA, recorded at the top: SEED, ARCH_ARGS, ...), which they must
# retrain and grade with (2026-10-21), then the PCA targets, USE_CELL_LINES and
# the variants in full (2026-10-22).
_retrain_recipe() {
    local i="$1" tab arms a kept="$PCA_ARMS"
    # with the pair partner this run trains, or whose selection it keeps
    # (pca_pair_closed, 2026-10-24): retrained alone, joint (or both) would
    # rank Step-1 runs of its own, and none (or per_modality) would rebuild
    # those its partner's selection ranked
    for a in none joint per_modality both; do
        [ ! -f "$SIAMESE_DIR/$(pca_arm_results "$a")/best_per_category.json" ] || kept="$kept $a"
    done
    arms=$(pca_pair_closed "$kept" $_LEFT_RETRAIN)
    case "$arms" in
        *" "*) tab="and builds one over $arms only." ;;
        *)     tab="and builds none (one arm)." ;;
    esac
    [ "$arms" = "$_LEFT_RETRAIN" ] || echo "${i}(PCA arm(s) $arms: a pair shares its single-embedding runs, so both arms of it retrain)"
    echo "$i  ${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") PCA_ARMS=\"$arms\" bash $0 --skip-generate --skip-shared"
    echo "${i}That run removes this run's table across the arms (sl_comparison/results/pca_arms/),"
    echo "${i}if any, $tab A table across every arm, with its 2x2 effects,"
    echo "${i}needs one submission's selection of every arm, which retrains this run's arms too"
    echo "${i}(and reuses its shared rows):"
    echo "$i  env -u PCA_ARMS ${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") bash $0 --skip-generate --skip-shared"
}
_rm_previous() {   # LABEL PATH... : rm -rf each existing PATH, report the count
    local label="$1" n=0 p
    shift
    for p in "$@"; do
        [ -e "$p" ] || continue
        rm -rf -- "$p"
        n=$((n + 1))
    done
    [ "$n" -gt 0 ] && echo "  removed $n previous-run $label"
    return 0
}
_move_rows_aside() {   # DIR FILE... : move each existing sl_comparison row / marker into DIR
    local dir="$1" p b rel
    shift
    for p in "$@"; do
        [ -f "$p" ] || continue
        mkdir -p "$dir"
        b=$(basename "$p")
        rel=${p#"$ROOT"/}
        echo "  moving previous-run file aside: $rel"
        # <name>.<ts>.<ext>, as collect_and_regrade_siamese.sh names its moves.
        mv -f "$p" "$dir/${b%.*}.$_ts.${b##*.}"
    done
    return 0
}
echo "--- previous-run outputs ---"
if [ "$RUN_SIAMESE" = true ]; then
    for _arm in "${PCA_ARM_LIST[@]}"; do
        SR="$SIAMESE_DIR/$(pca_arm_results "$_arm")"
        CR="$ROOT/$(pca_arm_cmp "$_arm")"
        # run_compare.sh publishes a siamese row only for a (variant, CV) whose
        # <arm root>/<prefix>latest_combo_<cv>.txt exists. The combo job
        # deletes them when it starts and writes each on success; removing the
        # previous run's here as well means a combo that never starts leaves no
        # row.
        for _v in "${SIAMESE_VARIANTS[@]}"; do
            rm -f "$SR/$(siamese_variant_prefix "$_v")latest_combo_"cv[123].txt
            # ALLMOD_COMBO=1: the variant's all-modalities models are
            # retrained too (run_allmod_combo.sh), and only their markers make
            # rows (collect_siamese.py --allmod).
            [ "$ALLMOD_COMBO" != 1 ] || \
                rm -f "$SR/$(allmod_prefix "$_v")latest_allmod_"cv[123].txt
        done
        # Everything else derived from the siamese tree this run retrains: the
        # ablation and its summary, and the cell-line ablation. A skipped
        # stage's outputs describe the combos this run replaces, so they go too.
        _rm_previous "ablation output(s) of PCA arm $_arm" \
            "$SR"/ablate_no_* "$SR"/var_*__ablate_no_* "$SR/ablation_summary.csv"
        _rm_previous "cell-line ablation dir(s) of PCA arm $_arm" \
            "$SR"/cellline_ablation_cv* "$SR"/var_*__cellline_ablation_cv*
        # (<combo>/pair_predictions/ and <combo>/eval_finetuning_<cv>/ need no
        # line here: they live inside the combo dir, which the combo job moves
        # aside when it starts and deletes once its selection has succeeded.)
        # The combo deletes it only once it starts; paper_tables reads the
        # winners (tab:loco, E3) from it.
        _rm_previous "modality selection of PCA arm $_arm" "$SR/best_per_category.json"
        # The post-hoc no-cell-line rows: collect_and_regrade_siamese.sh never
        # moves them (they are not its rows) and compare.py tabulates every
        # top-level results/*.json.
        _move_rows_aside "$CR/stale_siamese" "$CR"/siamese_sl*_nocell_cv[123].json
    done
    # With a subset of the variants (SIAMESE_VARIANTS_LIST, USE_CELL_LINES=0)
    # or of the arms (PCA_ARMS), the other combos are not retrained, but when
    # this run regenerates the embeddings they were trained on, their markers
    # go too, in EVERY arm's root: a later --skip-siamese rebuild or
    # submit_pipeline.sh --ablation-only would read them as current against
    # vectors they were not trained on.
    if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
        for _arm in none joint per_modality both; do
            SR="$SIAMESE_DIR/$(pca_arm_results "$_arm")"
            _rm_previous "combo marker(s) of PCA arm $_arm this run does not retrain (the embeddings are regenerated)" \
                "$SR"/latest_combo_cv[123].txt "$SR"/var_*__latest_combo_cv[123].txt \
                "$SR"/allmod__*latest_allmod_cv[123].txt
        done
    fi
    # Single-modality runs of EVERY variant (<solo prefix>pcavar*: an
    # extra-variant run from an earlier BESTCAT_EXTRA_VARIANTS=1 submission
    # would otherwise sit in tab:solo next to this run's), in the roots whose
    # Step-1 array this run queues: its owners' only (another root's are no
    # arm's of this run).
    for _o in $_OWNERS; do
        SR="$SIAMESE_DIR/$(pca_arm_results "$_o")"
        _rm_previous "single-modality run dir(s) of PCA arm(s) $(_served "$_o")" \
            "$SR"/pcavar"${PCA_VARIANCE}"_* "$SR"/var_*__pcavar"${PCA_VARIANCE}"_*
    done
fi
if [ "$RUN_SHARED" = true ]; then
    # run_shared_folds.sh / run_shared_models.sh move a row aside only once
    # that task runs, and run_compare.sh (afterany) reads every results/*.json.
    # EVERY cv, not only $CVS: this run retrains the shared tier, compare.py
    # globs every top-level results/*.json and paper_tables tabulates cv1-cv3
    # regardless of $CVS, and step 0 rebuilds the folds byte-identically (same
    # seed and sources), so an earlier run's row passes the fold-stamp gate and
    # is published as this run's. The SLeuth rows below are moved aside for all
    # three for the same reason. The cv3 rows matter twice: run_external_transfer
    # .sh gates each baseline on <NAME>_cv3.json. (--skip-shared, which reuses
    # the shared rows on purpose, never reaches this block.) Once, for every
    # arm: the rows live in sl_comparison/results/ only.
    # ELISL too (its array is submitted with the shared models since
    # 2026-11-13): a previous row passes the same fold-stamp gate and
    # paper_tables counts it as a tab_tiers peer. --skip-elisl keeps it, as
    # --skip-shared keeps every peer's.
    _ELISL_ROW=""; [ "$RUN_ELISL" != true ] || _ELISL_ROW=ELISL
    for _cv in cv1 cv2 cv3; do
        for _m in SLGNN NSF4SL SLMGAE DegreePrior $_ELISL_ROW; do
            _move_rows_aside "$ROOT/sl_comparison/results/stale_shared" \
                "$ROOT/sl_comparison/results/${_m}_$_cv.json"
        done
    done
    # Their score matrices too. A task clears its own only once it starts
    # (run_shared_models.sh), and an array cancelled with the folds job
    # (afterok) never does: the previous run's complete matrices would stay in
    # the graded directories with a fold stamp that still matches, and
    # evaluate_model.py run by hand there would grade them as this run's.
    _rm_previous "shared-model score matrix(es)" \
        "$ROOT"/SLGNN/results/slgnn_cv[123]/fold_*_predictions.npy \
        "$ROOT"/NSF4SL/results/nsf4sl_cv[123]/fold_*_predictions.npy \
        "$ROOT"/SLMGAE-in-pytorch/results/slmgae_shared_cv[123]/fold_*_predictions.npy
    [ "$RUN_ELISL" != true ] || _rm_previous "ELISL score matrix(es)" \
        "$ROOT"/ELISL/results/elisl_cv[123]/fold_*_predictions.npy
    unset _ELISL_ROW
    # Every arm's comparison table, external transfer and paper tables set it
    # against these rows. An arm this DAG leaves out (PCA_ARMS) keeps its
    # SLeuth rows and siamese outputs, but no job of this run rebuilds its
    # tables, which would go on publishing the peers this run replaces (each
    # still PAPER COMPLETE) beside this run's: they go too. The arms are
    # named by their state, not by what this run removed (a submission that
    # stopped part-way after this point left them without tables, and its
    # resubmission finds nothing left to remove): every left-out arm with
    # something to tabulate, all without a comparison.csv by now. With a combo
    # marker (_LEFT_OUT): its tables, rebuilt by the jobs the final listing
    # prints wired to this run's (their ids are known only there). With a
    # selection but no marker (_LEFT_RETRAIN: this run's regeneration removed
    # them above, or its combos never finished): a rebuild would have no
    # SLeuth rows, so the arm is retrained once this run has finished.
    for _arm in none joint per_modality both; do
        case " $PCA_ARMS " in *" $_arm "*) continue ;; esac
        CR="$ROOT/$(pca_arm_cmp "$_arm")"
        SR="$SIAMESE_DIR/$(pca_arm_results "$_arm")"
        _rm_previous "output(s) of PCA arm $_arm built on the shared rows (comparison table, external transfer, paper tables)" \
            "$CR/comparison.csv" "$CR/comparison_notes.txt" \
            "$CR/external_transfer" "$CR/paper_tables" "$CR/bootstrap"
        if _has_marker "$SR"; then
            _LEFT_OUT="${_LEFT_OUT:+$_LEFT_OUT }$_arm"
        elif [ -f "$SR/best_per_category.json" ]; then
            _LEFT_RETRAIN="${_LEFT_RETRAIN:+$_LEFT_RETRAIN }$_arm"
        fi
    done
    if [ -n "$_LEFT_OUT" ]; then
        echo "NOTE: PCA arm(s) $_LEFT_OUT, which PCA_ARMS leaves out, have no comparison, external-"
        echo "  transfer or paper tables now: they were built on the shared rows this run retrains"
        echo "  (their SLeuth rows and siamese outputs stay). The jobs that rebuild them, wired to"
        echo "  this run's, are printed at the end (--skip-shared would have kept the tables)."
    fi
    if [ -n "$_LEFT_RETRAIN" ]; then
        echo "NOTE: PCA arm(s) $_LEFT_RETRAIN, which PCA_ARMS leaves out, have no tables now and no combo"
        echo "  that counts: $_WHY_RETRAIN."
        echo "  Tables rebuilt now would have no SLeuth rows. Once this run has finished, retrain"
        echo "  them on its embeddings and shared rows (repeated at the end):"
        _retrain_recipe "  "
    fi
fi
# The 2x2 SLeuth rows and their selected-run markers, with or without
# --skip-siamese: the compare job (always submitted) rebuilds them from the
# latest_combo markers, but collect_and_regrade_siamese.sh moves the old ones
# aside only once that job starts, and paper_tables reads top-level rows when
# comparison.csv is absent, so a compare that dies first would publish the
# previous run's SLeuth numbers next to this run's peers. Built from both
# sides, so stale whichever side retrains, and their jobs clear them again at
# start: comparison.csv (compare.py deletes it only once it runs, and
# paper_tables prefers it over the top-level row JSONs), external_transfer/
# and paper_tables/ (the compare job is always submitted, so this never loses
# a table a job of this run would not rebuild). Per arm.
for _arm in "${PCA_ARM_LIST[@]}"; do
    CR="$ROOT/$(pca_arm_cmp "$_arm")"
    _move_rows_aside "$CR/stale_siamese" \
        "$CR"/siamese_sl{,_pd_pooled,_bilinear_per_head,_bilinear_pooled}_cv[123].json \
        "$CR"/siamese{,_sl_pd_pooled,_sl_bilinear_per_head,_sl_bilinear_pooled}_selected_cv[123].txt \
        "$CR"/siamese_sl_allmod{,_pd_pooled,_bilinear_per_head,_bilinear_pooled}_cv[123].json \
        "$CR"/siamese_sl_allmod{,_pd_pooled,_bilinear_per_head,_bilinear_pooled}_selected_cv[123].txt
    _rm_previous "external-transfer table dir of PCA arm $_arm" "$CR/external_transfer"
    _rm_previous "paper-tables and bootstrap dirs of PCA arm $_arm" "$CR/paper_tables" "$CR/bootstrap"
    _rm_previous "comparison table of PCA arm $_arm" "$CR/comparison.csv" \
        "$CR/comparison_notes.txt"
done
# The table across the arms reads every arm's (the job that rebuilds it is
# queued only with 2 arms or more: a previous one would compare this run's
# arms with an earlier run's).
_rm_previous "table across the PCA arms (sl_comparison/results/pca_arms/)" \
    "$ROOT/sl_comparison/results/pca_arms"
unset _arm _v _cv _m _o SR CR

# ---- 0b. environment reset (shared venv) -----------------------------------
# afterok edges below carry KILL: a failed parent cancels the child instead of
# leaving it pending (DependencyNeverSatisfied) forever, which would also hang
# every afterany consumer.
KILL="--kill-on-invalid-dep=yes"
# ":"-joined non-empty job ids (a skipped stage has none), each once; an
# argument may itself be a ":"-joined list.
_join_jobs() {
    local out="" a j
    for a in "$@"; do
        for j in ${a//:/ }; do
            case ":$out:" in *":$j:"*) ;; *) out="$out:$j" ;; esac
        done
    done
    echo "${out#:}"
}
# A submission that fails part-way (e.g. a QOS submit limit) exits under set -e
# with part of the DAG queued: name what to cancel before resubmitting, every
# arm's (_queued after each sbatch). Step 0a has run by then: the arms it left
# without tables are named too (the final listing that wires their rebuild is
# never reached; a resubmission names them again).
_on_submit_error() {
    if [ -n "$_QUEUED" ]; then
        # each command alone on its line (a trailing "# ..." reached scancel
        # as job ids in an interactive zsh, interactivecomments off), the
        # rebuild with the settings taken from the calling shell (TORCH_SPEC,
        # TORCH_INDEX_URL: reset_env.sh reads them)
        echo "ERROR: submission stopped part-way; already queued:$_QUEUED"
        echo "  Cancel them before resubmitting:"
        echo "    scancel$_QUEUED"
        if [ -n "$ENV_JOB" ]; then
            echo "  reset_env ($ENV_JOB) deletes the venv as soon as it starts, and this script"
            echo "  needs that venv on this node: once it has started (squeue -j $ENV_JOB),"
            echo "  let it finish, or after cancelling it rebuild the venv before resubmitting:"
            echo "    (cd siamese_sl && ${_ENV_PREFIX}sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} slurm/reset_env.sh)"
        fi
    fi
    local lo="$_LEFT_OUT${_LEFT_OUT:+${_LEFT_RETRAIN:+ }}$_LEFT_RETRAIN"
    [ -z "$lo" ] || \
        echo "  NOTE: left-out PCA arm(s) $lo have no tables now (step 0a); resubmitting names them again, with what rebuilds them."
    return 0
}
trap _on_submit_error ERR
ENV_DEP=""
if [ "$RESET_ENV" = true ]; then
    echo "--- reset_env ---"
    # --after <JOB>: that job may still be running from the venv reset_env.sh
    # rm -rf's, so the reset starts only once it has finished (afterany);
    # everything downstream still waits on both (afterok, step 1).
    ENV_AFTER=""
    if [ -n "$AFTER_JOB" ]; then ENV_AFTER="--dependency=afterany:$AFTER_JOB"; fi
    ENV_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable $GPU_NODE_ARGS --export=ALL $ENV_AFTER \
        --output="$SIA_LOGS/reset_env_%j.out" --error="$SIA_LOGS/reset_env_%j.err" \
        slurm/reset_env.sh)
    _queued "$ENV_JOB"
    echo "  reset_env job: $ENV_JOB"
    ENV_DEP="--dependency=afterok:$ENV_JOB"
fi

# ---- 1. embedding generation (siamese's all_genes_*.pt, incl kg_complex) ----
# (once: every arm reads the same files)
GEN_DEP="$ENV_DEP"
if [ "$RUN_SIAMESE" = false ]; then
    echo "--- no embedding generation (--skip-siamese: no job of this run reads it) ---"
    # --after still orders the shared side behind JOB, but afterany: its
    # outcome does not matter to them (with reset_env, the reset waits for it).
    if [ -n "$AFTER_JOB" ] && [ -z "$ENV_DEP" ]; then
        GEN_DEP="--dependency=afterany:$AFTER_JOB"
    fi
elif [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    echo "--- generate embeddings ---"
    GEN_JOB=$(cd "$SIAMESE_DIR" && sbatch --parsable $GPU_NODE_ARGS --export=ALL $ENV_DEP $KILL \
        --output="$SIA_LOGS/siamese_gen_emb_%j.out" --error="$SIA_LOGS/siamese_gen_emb_%j.err" \
        slurm/run_generate_embeddings.sh)
    _queued "$GEN_JOB"
    echo "  generation job: $GEN_JOB"
    GEN_DEP="--dependency=afterok:$GEN_JOB"
elif [ -n "$AFTER_JOB" ]; then
    echo "--- chaining after existing job $AFTER_JOB ---"
    if [ -n "$ENV_DEP" ]; then GEN_DEP="--dependency=afterok:$ENV_JOB:$AFTER_JOB"
    else GEN_DEP="--dependency=afterok:$AFTER_JOB"; fi
else
    echo "--- skipping generation (embeddings assumed present) ---"
fi

# (shared folds were generated in step 0, before reset_env could delete the
# interpreter out from under them.)

# ---- 2. shared models: folds job, then a 3x3 ARRAY (SLGNN/NSF4SL/SLMGAE) ----
# Submitted right after the generation, before any siamese job (2026-10-21;
# it came after every arm's, step 3 then): the DAG is the same, edges and
# options included, only the job ids change. Under a FIFO-like priority
# (priority/basic, or multifactor without age weight: a freed GPU goes to
# the lowest eligible id) the 9 shared tasks queued last waited for every
# arm's Step-1, eval, ablation and cell-line tasks, ~207 h in a default
# four-arm run, and every arm's compare and external transfer wait on
# them, so no arm had a table before the last arm's ablation was done;
# queued first they run ~8-11 h in. Under age-based priority it changes
# nothing.
# Was one job running nine trainings in a serial for-loop. Split so SLURM can
# overlap them: fold generation and the CPU-only degree prior run once in
# run_shared_folds.sh, then |MODELS| x |CVS| independent GPU tasks.
# The array is afterOK on folds — every task grades against that answer key, so
# there is nothing to salvage if fold generation failed. (Previous rows were
# moved aside in step 0a.) Once, for every arm: each arm's compare and external
# transfer read these rows.
if [ "$RUN_SHARED" = true ]; then
    echo "--- shared folds + degree prior (once, CPU-only) ---"
    FOLDS_JOB=$(sbatch --parsable $GPU_NODE_ARGS $GEN_DEP $KILL --export=ALL \
        --output="$CMP_LOGS/sl_folds_%j.out" --error="$CMP_LOGS/sl_folds_%j.err" \
        sl_comparison/slurm/run_shared_folds.sh)
    _queued "$FOLDS_JOB"
    echo "  shared-folds job:   $FOLDS_JOB"

    SHARED_TASKS=$(( 3 * $(echo "${CVS:-cv1 cv2 cv3}" | wc -w) ))
    echo "--- shared 3-model array ($SHARED_TASKS tasks: 3 models x CVs) ---"
    SHARED_JOB=$(sbatch --parsable $GPU_NODE_ARGS --dependency=afterok:$FOLDS_JOB $KILL --export=ALL \
        --array=0-$((SHARED_TASKS - 1))$(array_throttle "$SHARED_CONCURRENT") \
        --output="$CMP_LOGS/sl_compare_%A_%a.out" --error="$CMP_LOGS/sl_compare_%A_%a.err" \
        sl_comparison/slurm/run_shared_models.sh)
    _queued "$SHARED_JOB"
    echo "  shared-models array: $SHARED_JOB  (throttle $(throttle_label "$SHARED_CONCURRENT"))"
fi

# ---- 2'. ELISL (context-free): inputs when stale, then one task per CV -----
# The fourth published peer (ELISL/README.md), a shared row like the three
# above: its inputs job (when the login-node check above found them missing
# or stale; after the env reset, whose venv runs node2vec and the features),
# then ELISL/slurm/run_elisl.sh, task i = cv(i+1), afterok on the folds job
# and the inputs job (it grades against those folds and reads those inputs).
# Each arm's compare waits for it (afterany), so the first tables have it.
if [ "$RUN_ELISL" = true ]; then
    _ELISL_AFTER="$FOLDS_JOB"
    if [ "$ELISL_BUILD" = true ]; then
        echo "--- ELISL inputs (SeqVec on the GPU, node2vec, features -> data/elisl/) ---"
        ELISL_IN_JOB=$(sbatch --parsable $GPU_NODE_ARGS $ENV_DEP ${ENV_DEP:+$KILL} --export=ALL \
            --output="$ELISL_LOGS/elisl_inputs_%j.out" --error="$ELISL_LOGS/elisl_inputs_%j.err" \
            ELISL/slurm/build_elisl_inputs.sh)
        _queued "$ELISL_IN_JOB"
        echo "  ELISL inputs job:   $ELISL_IN_JOB  (SEQVEC_BIN=$SEQVEC_BIN)"
        _ELISL_AFTER="$_ELISL_AFTER:$ELISL_IN_JOB"
    fi
    _ELISL_IDX=""
    for _cv in ${CVS:-cv1 cv2 cv3}; do
        case "$_cv" in cv1) _i=0 ;; cv2) _i=1 ;; cv3) _i=2 ;; esac
        _ELISL_IDX="${_ELISL_IDX:+$_ELISL_IDX,}$_i"
    done
    echo "--- ELISL array (CVs ${CVS:-cv1 cv2 cv3}) ---"
    ELISL_JOB=$(sbatch --parsable $GPU_NODE_ARGS --dependency=afterok:$_ELISL_AFTER $KILL --export=ALL \
        --array=$_ELISL_IDX$(array_throttle "$SHARED_CONCURRENT") \
        --output="$ELISL_LOGS/elisl_%A_%a.out" --error="$ELISL_LOGS/elisl_%A_%a.err" \
        ELISL/slurm/run_elisl.sh)
    _queued "$ELISL_JOB"
    echo "  ELISL array:        $ELISL_JOB -> sl_comparison/results/ELISL_<cv>.json  (afterok:$_ELISL_AFTER)"
    unset _ELISL_AFTER _ELISL_IDX _cv _i
fi
# Every shared row's job: what an arm's compare waits for.
SHARED_ALL=$(_join_jobs "$SHARED_JOB" "$ELISL_JOB")

# ---- 3. siamese_sl: best-per-category (its own embedding, cell-line) + combo
if [ "$RUN_SIAMESE" = true ]; then
    # The single-embedding runs, once per owner (submitted as that arm: its
    # PCA_SINGLE_ARGS, its root), for every arm of its pair this DAG runs.
    # BESTCAT_N: bestcat_n_tasks (run_best_per_category.conf, sourced above).
    for _o in $_OWNERS; do
        _AT=$(_armtag "$_o")
        echo "--- siamese_sl best-per-category Step 1 of PCA arm(s) $(_served "$_o") ($BESTCAT_N tasks; BESTCAT_EXTRA_VARIANTS=${BESTCAT_EXTRA_VARIANTS:-0}) ---"
        BESTCAT_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_o" sbatch --parsable $GPU_NODE_ARGS $GEN_DEP $KILL --export=ALL \
            --output="$SIA_LOGS/bestcat_${_AT}%A_%a.out" --error="$SIA_LOGS/bestcat_${_AT}%A_%a.err" \
            --array=0-$((BESTCAT_N - 1))$(array_throttle "$MAX_CONCURRENT") slurm/run_best_per_category.sh)
        _queued "$BESTCAT_JOB"
        printf -v "BESTCAT_JOB_$_o" '%s' "$BESTCAT_JOB"
        echo "  best-cat array job: $BESTCAT_JOB  ($(pca_step1_label "$_o") -> siamese_sl/$(pca_arm_results "$_o")/)"
    done
    _HAS_PER_HEAD=false
    for _v in "${SIAMESE_VARIANTS[@]}"; do
        case "$_v" in *:per_head) _HAS_PER_HEAD=true ;; esac
    done
    # ABLATE_CATEGORIES (the ablation's array size, the same for every arm)
    if [ "$RUN_ABLATION" = true ]; then
        source "$SLURM_DIR/run_ablation.conf"
    fi
    if [ "$RUN_EVAL" = true ]; then
        EVAL_N=$( ( set +u; source "$SLURM_DIR/eval_finetuning.conf" >/dev/null 2>&1; \
                    echo "${#EVAL_CV_TYPES[@]}" ) 2>/dev/null )
        case "$EVAL_N" in ''|*[!0-9]*|0) EVAL_N=3 ;; esac
        EVAL_N=$((EVAL_N * ${#SIAMESE_VARIANTS[@]}))
    fi

    for _arm in "${PCA_ARM_LIST[@]}"; do
        # (every id of this arm starts empty, not the previous arm's)
        _arm_reset
        _AT=$(_armtag "$_arm")
        _v="BESTCAT_JOB_$(pca_step1_arm "$_arm")"
        echo "--- PCA arm $_arm: $(pca_arm_label "$_arm") -> siamese_sl/$(pca_arm_results "$_arm")/ ---"
        COMBO_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
            --dependency=afterany:${!_v} --export=ALL \
            --output="$SIA_LOGS/bestcat_combo_${_AT}%j.out" --error="$SIA_LOGS/bestcat_combo_${_AT}%j.err" \
            slurm/run_best_per_category_combo.sh)
        _queued "$COMBO_JOB"
        echo "  combo job:          $COMBO_JOB  (one combo per variant x CV, on the Step-1 array ${!_v}; re-graded by run_compare.sh)"

        # ---- 3a'. All-modalities combo (ALLMOD_COMBO=1) ----------------------
        # Every catalog embedding, no modality selection, one task per CV over
        # every variant (siamese_sl/slurm/run_allmod_combo.sh). It reads no
        # Step-1 run or selection, so it waits for the embeddings only
        # (GEN_DEP, as the Step-1 array); the arm's compare waits for it.
        if [ "$ALLMOD_COMBO" = 1 ]; then
            ALLMOD_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS $GEN_DEP $KILL --export=ALL \
                --array=0-$((${#CV_TYPES[@]} - 1))$(array_throttle "$MAX_CONCURRENT") \
                --output="$SIA_LOGS/allmod_combo_${_AT}%A_%a.out" --error="$SIA_LOGS/allmod_combo_${_AT}%A_%a.err" \
                slurm/run_allmod_combo.sh)
            _queued "$ALLMOD_JOB"
            echo "  all-modalities job: $ALLMOD_JOB  (${#CV_TYPES[@]} tasks, every catalog embedding, no selection; rows siamese_sl_allmod*)"
        fi

        # ---- 3a''. Bilinear probe (arm none, with pd:pooled; 2026-11-13) -----
        # Trains on arm none's pd:pooled combos' per-fold features and
        # settings (siamese_sl/slurm/run_probe.sh), so afterOK on that combo
        # job; the arm's compare waits for it (afterany), so its row is in the
        # first paper tables instead of a hand rerun's.
        if [ "$_arm" = none ] && [[ " ${SIAMESE_VARIANTS[*]} " == *" pd:pooled "* ]]; then
            PROBE_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterok:$COMBO_JOB $KILL --export=ALL \
                --array=0-$((${#CV_TYPES[@]} - 1))$(array_throttle "$MAX_CONCURRENT") \
                --output="$SIA_LOGS/bilinear_probe_%A_%a.out" --error="$SIA_LOGS/bilinear_probe_%A_%a.err" \
                slurm/run_probe.sh)
            _queued "$PROBE_JOB"
            echo "  bilinear probe job: $PROBE_JOB  (${#CV_TYPES[@]} tasks, on arm none's pd:pooled combos; rows BilinearProbe_<cv>)"
        fi

        # ---- 3b. External-screen transfer eval, one task per (cv_type, variant)
        # afterOK on the combo: it loads that combo's checkpoints, so a failed
        # combo leaves nothing to evaluate. Resolves MODEL_DIR at job start
        # from the variant's <arm root>/<prefix>latest_combo_cv3.txt, which the
        # combo writes on success. (EVAL_MODEL_DIR / EVAL_VARIANT were cleared
        # above.)
        if [ "$RUN_EVAL" = true ]; then
            echo "  external-screen eval + fine-tuning ($EVAL_N tasks: cv_types x ${#SIAMESE_VARIANTS[@]} variants)"
            EVAL_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterok:$COMBO_JOB $KILL --export=ALL \
                --array=0-$((EVAL_N - 1))$(array_throttle "$EVAL_CONCURRENT") \
                --output="$SIA_LOGS/eval_finetuning_${_AT}%A_%a.out" \
                --error="$SIA_LOGS/eval_finetuning_${_AT}%A_%a.err" \
                slurm/eval_finetuning.sh)
            _queued "$EVAL_JOB"
            echo "  eval array job:     $EVAL_JOB  (Adamson / Corn / Gilbert)"
        fi

        # ---- 3c. Leave-one-category-out ablation ----------------------------
        # Also afterOK on the combo, so it runs CONCURRENTLY with the eval
        # above rather than behind it; SLURM resolves the contention
        # (MAX_CONCURRENT caps the arrays only when set). The full combo is the
        # ablation's reference row and is never retrained here, so this stage
        # cannot change any number in the comparison table. The arm's PCA, as
        # its reference combos'.
        if [ "$RUN_ABLATION" = true ]; then
            ABL_N=$(( ${#ABLATE_CATEGORIES[@]} * ${#CV_TYPES[@]} * ${#SIAMESE_VARIANTS[@]} ))
            echo "  leave-one-category-out ablation ($ABL_N tasks: categories x CVs x variants)"
            ABL_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterok:$COMBO_JOB $KILL --export=ALL \
                --array=0-$((ABL_N - 1))$(array_throttle "$MAX_CONCURRENT") \
                --output="$SIA_LOGS/ablation_${_AT}%A_%a.out" \
                --error="$SIA_LOGS/ablation_${_AT}%A_%a.err" \
                slurm/run_ablation.sh)
            _queued "$ABL_JOB"
            echo "  ablation array job: $ABL_JOB"
            ABL_SUMM=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterany:$ABL_JOB --export=ALL \
                --output="$SIA_LOGS/ablation_summary_${_AT}%j.out" \
                --error="$SIA_LOGS/ablation_summary_${_AT}%j.err" \
                slurm/run_ablation_summary.sh)
            _queued "$ABL_SUMM"
            echo "  ablation summary:   $ABL_SUMM -> siamese_sl/$(pca_arm_results "$_arm")/ablation_summary.csv"
        fi

        # ---- 3d. Post-hoc cell-line ablation (tab:cellline) -----------------
        # ONE job over every per_head variant x CV of the arm: zeroes cell_emb
        # in that variant's combo checkpoints and grades the matrices with the
        # shared evaluator -> <arm's sl_comparison root>/<row>_nocell_<cv>.json.
        # Pooled variants have no cell embedding (they are the retrained
        # no-cell-line counterparts), so with no per_head variant, or
        # USE_CELL_LINES=0, there is nothing to ablate. afterOK on the combo
        # (it reads its checkpoints); the arm's final compare waits for it so
        # the rows reach its comparison.csv.
        if [ "$RUN_CELL" = true ] && [ "$_HAS_PER_HEAD" = true ] && [ "${USE_CELL_LINES:-1}" = 1 ]; then
            CELL_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterok:$COMBO_JOB $KILL --export=ALL \
                --output="$SIA_LOGS/cellline_ablation_${_AT}%j.out" \
                --error="$SIA_LOGS/cellline_ablation_${_AT}%j.err" \
                cell_line_ablation/run_cell_line_ablation.sh)
            _queued "$CELL_JOB"
            echo "  cell-line ablation: $CELL_JOB -> $(pca_arm_cmp "$_arm")/<row>_nocell_<cv>.json"
        elif [ "$RUN_CELL" = true ]; then
            echo "  post-hoc cell-line ablation: not applicable (no per_head variant with USE_CELL_LINES=1)"
        fi
        _arm_save "$_arm"
    done
fi

# ---- 4-6. per arm: final comparison, external transfer, paper tables -------
# Each arm's jobs read its own roots (the PCA_ARM they are submitted with:
# sl_comparison/slurm/pca_arm.sh) and the shared rows in sl_comparison/results/.
for _arm in "${PCA_ARM_LIST[@]}"; do
    _arm_load "$_arm"
    _AT=$(_armtag "$_arm")
    # (named after the stages queued below: --skip-eval queues no external
    # transfer, --skip-paper-tables no tables; 2026-10-22)
    _STAGES="comparison"
    [ "$RUN_EXT" != true ] || _STAGES="$_STAGES, external transfer"
    [ "$RUN_PT" != true ] || _STAGES="$_STAGES, paper tables"
    echo "--- PCA arm $_arm: $_STAGES -> $(pca_arm_cmp "$_arm")/ ---"

    # ---- 4. final comparison (afterANY on every terminal of the arm) --------
    # afterany (not afterok): the shared array runs 9 model×CV tasks and
    # individual tasks may fail; siamese's combo may also partially fail. We
    # still want the table built from whatever succeeded, so run compare
    # regardless. Depending on the array job id covers every task in it.
    TERMINALS=$(_join_jobs "$COMBO_JOB" "$ALLMOD_JOB" "$PROBE_JOB" "$CELL_JOB" "$SHARED_ALL")
    COMPARE_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS --dependency=afterany:$TERMINALS --export=ALL \
        --output="$CMP_LOGS/final_compare_${_AT}%j.out" --error="$CMP_LOGS/final_compare_${_AT}%j.err" \
        sl_comparison/slurm/run_compare.sh)
    _queued "$COMPARE_JOB"
    echo "  final compare job:  $COMPARE_JOB  (afterany:$TERMINALS)"

    # ---- 5. external transfer (tab:external, tab:external_ft) ---------------
    # afterany on EVERY eval task of the arm (the array id covers them) and the
    # shared array, so partial results still give tables, with each absent
    # model named in the log. Never with --skip-eval on a run that trains
    # siamese (RUN_EXT is cleared where the flags are read), so the
    # :-$COMBO_JOB fallback is no longer used.
    if [ "$RUN_EXT" = true ]; then
        EXT_DEPS=$(_join_jobs "${EVAL_JOB:-$COMBO_JOB}" "$SHARED_JOB")
        EXT_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS ${EXT_DEPS:+--dependency=afterany:$EXT_DEPS} --export=ALL \
            --output="$CMP_LOGS/external_transfer_${_AT}%j.out" --error="$CMP_LOGS/external_transfer_${_AT}%j.err" \
            sl_comparison/slurm/run_external_transfer.sh)
        _queued "$EXT_JOB"
        echo "  external transfer:  $EXT_JOB -> $(pca_arm_cmp "$_arm")/external_transfer/  (afterany:$EXT_DEPS)"
    fi

    # ---- 6. paper tables: the last job any table of the arm waits for -------
    if [ "$RUN_PT" = true ]; then
        PT_DEPS=$(_join_jobs "$COMPARE_JOB" "$EXT_JOB" "$CELL_JOB" "$ABL_SUMM" "$EVAL_JOB")
        PT_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS --dependency=afterany:$PT_DEPS --export=ALL \
            --output="$CMP_LOGS/paper_tables_${_AT}%j.out" --error="$CMP_LOGS/paper_tables_${_AT}%j.err" \
            sl_comparison/slurm/run_paper_tables.sh)
        _queued "$PT_JOB"
        echo "  paper tables job:   $PT_JOB -> $(pca_arm_cmp "$_arm")/paper_tables/  (afterany:$PT_DEPS)"
    fi
    _arm_save "$_arm"
done

# ---- 6'. the table across the PCA arms: the last job any table waits for ---
# With 2 arms or more: each arm's combos and graded rows against the first
# arm's (none when it runs), and every arm's paper tables with its verdict
# (sl_comparison/results/pca_arms/). Submitted as the reference arm.
if [ "$RUN_PT" = true ] && [ "${#PCA_ARM_LIST[@]}" -ge 2 ]; then
    _PT_ALL=""
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _v="PT_JOB_$_arm"
        _PT_ALL=$(_join_jobs "$_PT_ALL" "${!_v}")
    done
    echo "--- table across the PCA arms ${PCA_ARM_LIST[*]} (afterany:$_PT_ALL) ---"
    PT_ARMS_JOB=$(_as_arm "${PCA_ARM_LIST[0]}" sbatch --parsable $GPU_NODE_ARGS --dependency=afterany:$_PT_ALL --export=ALL \
        --output="$CMP_LOGS/paper_tables_pca_arms_%j.out" --error="$CMP_LOGS/paper_tables_pca_arms_%j.err" \
        sl_comparison/slurm/run_paper_tables.sh --pca_arms_table)
    _queued "$PT_ARMS_JOB"
    echo "  PCA arms table job: $PT_ARMS_JOB -> sl_comparison/results/pca_arms/"
    unset _PT_ALL _v
fi

# ---- 7. all-gene-pair export: the LAST jobs, one per arm --------------------
# For each variant's cv3 combo of the arm: the predicted SL probability of
# every gene pair, one matrix per cell-line head, and the fine-tuned models'
# predicted Adamson value for every gene pair (siamese_sl/slurm/
# export_pair_predictions.sh; what it writes: export_pair_predictions.conf,
# PAIR_EXPORT_*). Nothing reads it, so it runs after every other job
# (afterany): its arm's, and every other arm's through the table across the
# arms, or, without one (one arm, --skip-paper-tables), through each arm's
# last jobs: it can neither delay a table nor, should its 42 GB fill the disk,
# break one. singleton: never beside another pair_export job of yours (the
# arms' exports run one after another; a `submit_pipeline.sh --eval-only`
# export of the same combo would clear this one's files). Not with
# --skip-siamese (no combo of this run; sbatch it by hand for the reused
# ones). A submission problem here is a WARNING: the rest of the DAG is
# already queued and does not depend on it.
if [ "$RUN_SIAMESE" = true ] && [ "$RUN_PAIR" = true ]; then
    _ALL_LAST=""
    if [ -z "$PT_ARMS_JOB" ]; then
        for _a in "${PCA_ARM_LIST[@]}"; do
            for _v in PT_JOB COMPARE_JOB EXT_JOB CELL_JOB ABL_SUMM EVAL_JOB; do
                _n="${_v}_$_a"
                _ALL_LAST=$(_join_jobs "$_ALL_LAST" "${!_n}")
            done
        done
    fi
    if [ "$RUN_EVAL" != true ]; then
        # eval_finetuning.py writes the fine-tuned models; without it the
        # export would end FAILED on the missing sets.
        export PAIR_EXPORT_FT_SPLITS=none
        echo "  --skip-eval: probabilities only (no fine-tuned models to score)"
    fi
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _arm_load "$_arm"
        _AT=$(_armtag "$_arm")
        PAIR_DEPS=$(_join_jobs "$PT_JOB" "$COMPARE_JOB" "$EXT_JOB" "$CELL_JOB" "$ABL_SUMM" \
            "$EVAL_JOB" "$PT_ARMS_JOB" "$ALLMOD_JOB" "$PROBE_JOB" "$_ALL_LAST")
        # the options of the command printed below if the submission is
        # refused (to paste from siamese_sl/, where the logs are relative),
        # behind the arm's own assignments (_arm_env) in front of its sbatch.
        # That command starts with the settings this run took from the
        # calling shell (_ENV_PREFIX, 2026-10-21): the export reads
        # PAIR_EXPORT_*, and through config.conf the variants
        # (SIAMESE_VARIANTS_LIST, USE_CELL_LINES) and SHARED_FOLDS, and a
        # one-shot PAIR_EXPORT_UNIVERSE=shared PAIR_EXPORT_DTYPE=float16 in
        # front of this script never reached the shell it is pasted into (all
        # genes in float32 there: 42 GB per arm instead of about 4.5). The
        # words are printf %q-quoted (bash and zsh read them back). $_PAIR_ENV
        # and the arm come after them: the last assignment wins, so
        # --skip-eval's PAIR_EXPORT_FT_SPLITS=none beats a recorded one, as the
        # refused job got it.
        _PAIR_OPTS="--export=ALL"
        _L=$(_logs "$_arm" pair_export %j); [ -z "$_L" ] || _PAIR_OPTS="$_PAIR_OPTS $_L"
        echo "--- all-gene-pair export of PCA arm $_arm (afterany:$PAIR_DEPS) ---"
        if PAIR_JOB=$(cd "$SIAMESE_DIR" && _as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS --dependency=afterany:$PAIR_DEPS,singleton --export=ALL \
                --output="$SIA_LOGS/pair_export_${_AT}%j.out" --error="$SIA_LOGS/pair_export_${_AT}%j.err" \
                slurm/export_pair_predictions.sh); then
            _queued "$PAIR_JOB"
            echo "  pair export job:    $PAIR_JOB -> siamese_sl/$(pca_arm_results "$_arm")/<combo>/pair_predictions/"
        else
            PAIR_JOB=""
            _PAIR_ENV=""
            [ "$RUN_EVAL" = true ] || _PAIR_ENV="PAIR_EXPORT_FT_SPLITS=none "
            echo "WARNING: could not submit the all-gene-pair export of PCA arm $_arm; every other"
            echo "  job is queued. It has to stay the last job: from siamese_sl/, while they are queued,"
            echo "    ${_ENV_PREFIX}$(_vset "${SIAMESE_VARIANTS[*]}") PAIR_EXPORT_MODEL_DIR= ${_PAIR_ENV}$(_arm_env "$_arm") sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --dependency=afterany:$PAIR_DEPS,singleton $_PAIR_OPTS slurm/export_pair_predictions.sh"
            echo "  or the same without --dependency once every job above has finished."
        fi
        _arm_save "$_arm"
    done
    unset _ALL_LAST _a _v _n _L
fi
trap - ERR

echo "============================================================================"
echo "All jobs submitted (run token $RUN_TOKEN). Monitor:  squeue -u $(whoami)"
# The NOTE lines under the banner have scrolled away by now: name the
# inherited settings once more on the last screen.
[ -z "$_ENV_NAMES" ] || echo "NOTE: taken from the calling shell (values under the banner above):$_ENV_NAMES"
echo "Outputs, per PCA arm (each in its own roots; log files <stage>_pca_<arm>_<job id>.*"
echo "for every arm but none):"
for _arm in "${PCA_ARM_LIST[@]}"; do
_C=$(pca_arm_cmp "$_arm")
_v="PT_JOB_$_arm"; _PT=${!_v}
_v="COMPARE_JOB_$_arm"; _CMPJ=${!_v}
echo "  PCA arm $_arm: $(pca_arm_label "$_arm")"
echo "    $_C/comparison.csv        four-model table (tab:tiers source)"
[ "$RUN_EXT" = true ] && \
echo "    $_C/external_transfer/    external-screen tables"
[ "$RUN_SIAMESE" = true ] && [ "$RUN_ABLATION" = true ] && \
echo "    siamese_sl/$(pca_arm_results "$_arm")/ablation_summary.csv     leave-one-category-out"
if [ "$RUN_PT" = true ]; then
echo "    $_C/paper_tables/         every paper table (CSV), numbers.json, README.md"
[ "${PT_BOOTSTRAP:-1}" = 0 ] || \
echo "    $_C/bootstrap/            bootstrap CIs (the same job; the tables' CI columns)"
echo "    tail -f $CMP_LOGS/paper_tables_$(_armtag "$_arm")${_PT}.out"
echo "    verdict (when it ends): sed -n 3p $_C/paper_tables/README.md"
else
echo "    tail -f $CMP_LOGS/final_compare_$(_armtag "$_arm")${_CMPJ}.out"
fi
done
if [ -n "$PT_ARMS_JOB" ]; then
echo "  across the arms: sl_comparison/results/pca_arms/   tab_pca.csv, numbers.json, README.md"
echo "    tail -f $CMP_LOGS/paper_tables_pca_arms_${PT_ARMS_JOB}.out"
echo "    verdict (when it ends): sed -n 3p sl_comparison/results/pca_arms/README.md"
fi
if [ "$RUN_PT" = true ]; then
echo "  Each verdict: PAPER COMPLETE, or PAPER INCOMPLETE with the missing items; the jobs"
echo "    exit 0 / 3 (incomplete) / 1 (a table could not be built), so sacct shows them"
echo "    FAILED unless complete."
fi
unset _arm _C _v _PT _CMPJ
# The arms step 0a found without tables (its NOTEs have scrolled away by now).
# _LEFT_OUT: the jobs that rebuild their tables, wired to this run's now that
# the ids are known. Each arm's comparison and external transfer wait on this
# run's shared models, its paper tables on both. The table across the arms
# waits on those and on this run's own last table job: its table across the
# arms (which rewrites sl_comparison/results/pca_arms/ over PCA_ARMS only),
# else its arms' paper tables, else their comparison and external transfer.
if [ -n "$_LEFT_OUT" ]; then
    _OWN="$PT_ARMS_JOB"
    if [ -z "$_OWN" ]; then
        for _arm in "${PCA_ARM_LIST[@]}"; do
            _v="PT_JOB_$_arm"
            if [ -n "${!_v}" ]; then
                _OWN=$(_join_jobs "$_OWN" "${!_v}")
            else
                _v="COMPARE_JOB_$_arm"; _OWN=$(_join_jobs "$_OWN" "${!_v}")
                _v="EXT_JOB_$_arm"; _OWN=$(_join_jobs "$_OWN" "${!_v}")
            fi
        done
    fi
    _ALL_ARMS=""     # the arms the table across the arms then compares, in order
    for _arm in none joint per_modality both; do
        case " $PCA_ARMS $_LEFT_OUT " in
            *" $_arm "*) _ALL_ARMS="${_ALL_ARMS:+$_ALL_ARMS }$_arm" ;;
        esac
    done
    # Each pasted sbatch runs under _paste_env, which the block's first line
    # defines (config.conf pasted_env_def), so that no job gets the pasting
    # shell's RUN_TOKEN or CODE_MANIFEST (a hand recipe of
    # sl_comparison/README.md leaves both exported) or a compare pin
    # (SIAMESE_RESULTS_*, SIAMESE_BASELINE_DISCOVER: cleared above for this
    # run's own jobs; pasted into the same shell until 2026-10-21, a left-out
    # arm's compare graded the pinned combo, another run's or arm's, as that
    # arm's row; every pin name a compare reads is unset, whether this script
    # saw one or not), and the shell keeps them all: each arm's tables then
    # check its dirs against its own selection's token, as a standalone
    # run_paper_tables.sh does (paper_tables.py: RUN_TOKEN, else the
    # selection's; no code manifest is only noted). Every job gets the arms
    # of the table across the arms (PCA_ARMS), this run's PCA targets in
    # full (pca_targets_env: a one-shot target in front of this script never
    # reached the pasting shell) and this run's variants (SIAMESE_VARIANTS_LIST,
    # USE_CELL_LINES: the variants whose combos _pca_refs checks below, since
    # 2026-10-21; from the pasting shell, a one-shot subset was lost and the
    # jobs tabled config.conf's four, unchecked, under targets no line of
    # theirs set, or that shell's stray list), and this run's CVS (the
    # compare's CVs: this run's shared rows cover those only, a CV out of CVS
    # having its previous rows moved aside; empty, run_compare.sh's default,
    # all three; from the pasting shell until 2026-10-22), and a left-out arm whose combos
    # another submission trained at other targets gets those on its own lines
    # (its marker-named combos' config.json, config.conf pca_check_combos), so
    # its tables label and check them at the targets they were trained with.
    # A left-out arm's compare then rebuilds these variants' rows only (the
    # others' go to its stale_siamese/), as the one-submission alternative
    # below does, which pins the same list (_vset). The
    # table across the arms has one set of targets, this run's: an arm at
    # others is listed there as such. (Printed, not run: written so that no
    # line reads as a submission after the export's, the last one.) Each arm
    # rides in its sbatch's environment with a plain --export=ALL (as _as_arm
    # submits; never an --export list) and no RUN_TOKEN of its own: that is
    # the point of _paste_env.
    _CHK=(); _NREF=0
    _pca_refs $_LEFT_OUT
    _OUT=""; _UNCHECKED=""
    if [ "$_NREF" -gt 0 ] && ! _OUT=$(pca_check_combos "${_CHK[@]}"); then
        _OUT=""; _UNCHECKED=1
    fi
    _TS=""; _ODD=""
echo "  NOTE: PCA arm(s) $_LEFT_OUT, left out, have no comparison, external-transfer or paper"
echo "    tables (step 0a). To rebuild them on this run's shared models (job(s) $SHARED_ALL), paste"
echo "    these now (each job waits on this run's; \$_L is its log directory). _paste_env, defined"
echo "    on the first line, keeps the pasting shell's RUN_TOKEN, CODE_MANIFEST and compare pins"
echo "    (SIAMESE_RESULTS_*, SIAMESE_BASELINE_DISCOVER, ignored above) out of the jobs: each arm's"
echo "    tables check its dirs against its own selection's token, as a standalone"
echo "    run_paper_tables.sh does, and each arm's compare grades its own combos. Every job gets"
echo "    this run's PCA targets, an arm's own where its combos were trained at others, this run's"
echo "    variants (${SIAMESE_VARIANTS[*]}): the left-out arms are rebuilt over these, the"
echo "    variants their combos were checked for, and this run's CVS (its shared rows' CVs)."
echo "      $(pasted_env_def "PCA_ARMS=$_ALL_ARMS" $(pca_targets_env) \
          "SIAMESE_VARIANTS_LIST=${SIAMESE_VARIANTS[*]}" "USE_CELL_LINES=${USE_CELL_LINES:-1}" \
          "CVS=${CVS:-}" "ALLMOD_COMBO=$ALLMOD_COMBO")"
printf '      cd %q && _L=%q\n' "$ROOT" "$CMP_LOGS"
    for _arm in $_LEFT_OUT; do
        _AT=$(_armtag "$_arm")
        # the arm's own targets, where its combos' differ from this run's
        _EX=""
        while IFS='|' read -r _k _x _s _e _rest; do
            [ "$_k" = arm ] && [ "$_x" = "$_arm" ] || continue
            case "$_s" in
                fix)   _EX=" $_e" ;;
                other) _ODD="${_ODD:+$_ODD }$_arm" ;;
            esac
        done <<< "$_OUT"
        _SB="_paste_env PCA_ARM=$_arm$_EX sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --parsable"
        _X="--export=ALL"
echo "      _c_$_arm=\$($_SB --dependency=afterany:$SHARED_ALL $_X --output=\$_L/final_compare_${_AT}%j.out --error=\$_L/final_compare_${_AT}%j.err sl_comparison/slurm/run_compare.sh)"
echo "      _e_$_arm=\$($_SB --dependency=afterany:$SHARED_JOB $_X --output=\$_L/external_transfer_${_AT}%j.out --error=\$_L/external_transfer_${_AT}%j.err sl_comparison/slurm/run_external_transfer.sh)"
echo "      _t_$_arm=\$($_SB --dependency=afterany:\$_c_$_arm:\$_e_$_arm $_X --output=\$_L/paper_tables_${_AT}%j.out --error=\$_L/paper_tables_${_AT}%j.err sl_comparison/slurm/run_paper_tables.sh)"
        _TS="$_TS:\$_t_$_arm"
    done
echo "      _paste_env PCA_ARM=${_ALL_ARMS%% *} sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --dependency=afterany$_TS${_OWN:+:$_OWN} --export=ALL --output=\$_L/paper_tables_pca_arms_%j.out --error=\$_L/paper_tables_pca_arms_%j.err sl_comparison/slurm/run_paper_tables.sh --pca_arms_table"
echo "    The table across the arms lists an arm as stale (no values) when another submission than"
echo "    arm ${_ALL_ARMS%% *}'s made its selection: PAPER COMPLETE needs one submission's selection of every arm,"
echo "    at one set of PCA targets."
[ -z "$_ODD" ] || \
echo "    PCA arm(s) $_ODD: combos trained at PCA targets that no one set reproduces (they differ"
[ -z "$_ODD" ] || \
echo "    among themselves): their tables list them as missing whatever is exported."
[ -z "$_UNCHECKED" ] || \
echo "    (The left-out arms' combos could not be compared with this run's PCA targets on this"
[ -z "$_UNCHECKED" ] || \
echo "    node, arch_signature.py --check_pca: each line has this run's.)"
echo "    Or, once this run has finished, all of it in one submission (which retrains the shared"
echo "    models once more): ${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") PCA_ARMS=\"$_ALL_ARMS\" bash $0 --skip-siamese"
    unset _OWN _SB _TS _AT _X _arm _v _CHK _NREF _OUT _ODD _EX _k _x _s _e _rest _UNCHECKED
fi
# _LEFT_RETRAIN: no combo of theirs counts, so their tables would have no
# SLeuth rows: they are retrained once this run has finished.
if [ -n "$_LEFT_RETRAIN" ]; then
echo "  NOTE: PCA arm(s) $_LEFT_RETRAIN, left out, have no tables and no combo that counts:"
echo "    $_WHY_RETRAIN."
echo "    Once this run has finished, retrain them on its embeddings and shared rows:"
_retrain_recipe "    "
fi
_PAIR_JOBS=""
for _arm in "${PCA_ARM_LIST[@]}"; do
    _v="PAIR_JOB_$_arm"
    [ -z "${!_v}" ] || _PAIR_JOBS="${_PAIR_JOBS:+$_PAIR_JOBS }$_arm:${!_v}"
done
unset _arm _v
if [ -n "$_PAIR_JOBS" ]; then
# As the job will resolve them: the conf's defaults under this environment (in
# a subshell: they must reach neither this shell nor the jobs). Only a cv3
# combo is asked for fine-tuned sets. (No `case` in here: bash 3.2 cannot parse
# one inside $( ).) Each export's log names its arm as its root does
# (pair_export_<id>.out for arm none, pair_export_pca_<arm>_<id>.out else).
_PAIR_FT=$(source "$SLURM_DIR/export_pair_predictions.conf" >/dev/null 2>&1 || true
           _s=none
           for _c in ${PAIR_EXPORT_CVS:-cv3}; do
               [ "$_c" != cv3 ] || _s="${PAIR_EXPORT_FT_SPLITS:-}"
           done
           if [ "$_s" = none ]; then echo ", no fine-tuned predictions"
           else echo " and the fine-tuned ${PAIR_EXPORT_FT_DATASETS:-Adamson} predictions"; fi)
_PAIR_CVS=$(source "$SLURM_DIR/export_pair_predictions.conf" >/dev/null 2>&1 || true
            echo ${PAIR_EXPORT_CVS:-cv3})
echo "  siamese_sl/<arm root>/<$_PAIR_CVS combo>/pair_predictions/   SL probability of all gene pairs per"
echo "    cell line (sl_prob_<HEAD>.npy)$_PAIR_FT; the last jobs, one per arm:"
for _pj in $_PAIR_JOBS; do
    _t="${_pj%%:*}_"; [ "$_t" != none_ ] || _t=""
    echo "    ${_pj%%:*}: job ${_pj#*:} (tail -f $SIA_LOGS/pair_export_${_t:+pca_$_t}${_pj#*:}.out)"
done
fi
# The arms whose reused runs carry several tokens (--skip-siamese's WARNING)
# got this submission's fresh one: the submission that keeps one of theirs, to
# paste once these jobs are gone (the live-job guard refuses it before then:
# no ALLOW_LIVE_JOBS on it, which would lift that guard, 2026-10-22), this
# run's arms and targets in full. One RUN_TOKEN is kept only for the arms whose
# runs carry it (a token the run kept from the environment, too).
if [ -n "$_KEEP_TOKEN_ARMS" ] && [ -n "$_QUEUED" ]; then
_kept=""; [ -z "$_TOKEN_KEPT" ] || _kept=", as the exported $_INHERITED_RUN_TOKEN was"
echo "  NOTE: PCA arm(s) $_KEEP_TOKEN_ARMS reuse runs of several tokens (the WARNING above) and"
echo "    got this submission's fresh one. To keep one of those tokens instead, cancel this"
echo "    submission's jobs:"
echo "      scancel$_QUEUED"
echo "    and, once squeue lists none of them, submit it again with that token in place of"
echo "    <token> (one RUN_TOKEN is kept only for the arms whose runs carry it$_kept; the"
echo "    WARNING's other arms get a fresh one again):"
echo "      RUN_TOKEN=<token> ${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") PCA_ARMS=\"${PCA_ARM_LIST[*]}\" bash $0$_ARGS_Q"
fi
echo "============================================================================"
