#!/bin/bash
# ============================================================================
# Pipeline Orchestrator — run from the login node
# ============================================================================
# Submits the full Siamese SL training pipeline as SLURM jobs, the steps
# marked "per arm" once for every PCA arm of PCA_ARMS (below), the others once:
#
#   0. Environment reset          (default — recreates Python venv; skip with --skip-reset-env;
#                                 NOT with --eval-only / --ablation-only unless
#                                 --reset-env is passed; with --after it waits
#                                 for that job)
#   1. Embedding generation      (single job, sequential steps)
#   2. Embedding benchmark       (per owner, below; SLURM array job — one task
#                                 per embedding×CV)
#   3. Benchmark summary         (per owner; single job, collects results after array)
#   4. Best-per-category Step 1  (per owner; SLURM array job — one task per
#                                 embedding×CV; × variant with BESTCAT_EXTRA_VARIANTS=1)
#   5. Best-per-category combo   (per arm; single job — selects winners ONCE, then
#                                 trains one combo per SIAMESE_VARIANTS x CV_TYPES)
#   6. External-screen eval + fine-tuning
#                                (per arm; array — zero-shot + LP_last + Full FT
#                                 against Adamson/Corn/Gilbert on EVERY variant's
#                                 CV3 combo; |EVAL_CV_TYPES| x |SIAMESE_VARIANTS|
#                                 tasks; skip with --skip-eval)
#   7. Leave-one-category-out ablation
#                                (per arm; array — |ABLATE_CATEGORIES| x |CV_TYPES|
#                                 x |SIAMESE_VARIANTS| tasks, each against its
#                                 variant's own combo; keeps every run's 5 fold
#                                 matrices, 313 MB each, for paper_tables.py
#                                 --regrade: about 113 GB for an arm's 72 runs by
#                                 default unless ABLATION_PURGE_PREDICTIONS=1;
#                                 skip with --skip-ablation)
#   8. All-gene-pair export      (per arm, single job, the LAST ones: for every
#                                 variant's CV3 combo, the predicted SL
#                                 probability of every gene pair, one matrix per
#                                 cell-line head, and the fine-tuned models'
#                                 predicted Adamson value -> <combo>/pair_predictions/;
#                                 skip with --skip-pair-export)
#
# PCA arms: PCA_ARMS (slurm/config.conf; default "none joint" since
# 2026-11-13; any subset of none joint per_modality both, always run in that order) are the arms this submission
# trains, each the WHOLE pipeline above under its own results root: results/
# for none (no PCA), results/pca_<arm>/ for joint (PCA on the concatenation of
# all modalities), per_modality (PCA within each modality) and both
# (per_modality, then joint). Every name inside a root is the one this help
# gives for results/. Each arm's jobs are submitted as
# `PCA_ARM=<arm> sbatch ... --export=ALL` (the arm in sbatch's own
# environment, never in an --export list) under their usual job names and log to
# slurm/logs/<stage>_pca_<arm>_<job id>.out (arm none: the scripts' own
# slurm/logs/<stage>_<job id>.out). none and joint rank the same
# single-embedding runs (native dimension), per_modality and both theirs (PCA
# within the modality): steps 2-4 run once per such pair, as its OWNER, the
# pair's first arm that PCA_ARMS lists (results/ and results/pca_per_modality/
# by default), and each arm's combo waits for its owner's Step-1 array. Its
# _selection_meta records the arm, its PCA and the single-embedding root it
# ranked.
#
# Model variants: SIAMESE_VARIANTS (config.conf) is the 2x2 grid
# kernel {pd, bilinear} x cell_lines {per_head, pooled}. The single-embedding
# runs (steps 2 and 4) train SELECTION_VARIANT only, and the category winners
# are picked once per arm (step 5, from step 4); every variant's combo of that
# arm reuses them.
#
# Steps 2-3 and 4-5 run in parallel (independent output directories). Steps
# 6-7 chain after step 5 via afterok; step 6 reads each variant's CV3 combo
# path from the per-variant marker file written by step 5. Step 8 of every arm
# waits for every job above, of every arm (afterany), and for any other
# pair_export job of yours (singleton): no result depends on it, and it writes
# 42 GB per arm by default (slurm/export_pair_predictions.conf). The other
# outputs are larger, by default about 113 GB of ablation fold matrices and 19
# GB of combos per arm and, per owner, up to 89 GB of Step-1 and 94 GB of
# benchmark matrices (a task whose embedding is absent keeps none; a variant
# subset scales the per-arm ones down); the banner adds them up for the
# submission (1,062 GB at most by default; the eval's per-pair CSVs and the
# checkpoints come on top, named under it), with the jobs and the array tasks
# it queues (586 by default: Slurm's MaxSubmitJobs counts every task). Every
# afterok edge
# carries --kill-on-invalid-dep=yes (as in run_benchmarking_siamese_vs_others.sh):
# a failed parent cancels its children instead of leaving them, and every
# afterany summary / combo behind them, pending forever.
# Not here: the post-hoc cell-line ablation (tab:cellline). A best-per-category
# run removes, at submission, for every arm it trains, the replaced combos'
# cell-line ablation outputs, their sl_comparison rows (moved into that arm's
# own sl_comparison/results[/pca_<arm>]/stale_siamese/), their previous
# <arm root>/<prefix>ablate_no_* dirs and <arm root>/ablation_summary.csv (also
# with --skip-ablation: they describe the combos this run replaces), and the
# table across the arms, sl_comparison/results/pca_arms/; an arm PCA_ARMS leaves
# out is not touched, except that a run that regenerates the embeddings removes
# every arm's combo markers. A best-per-category run ends by printing, ready to
# paste and wired to its jobs, what rebuilds them: per arm the cell-line
# ablation (cell_line_ablation/run_cell_line_ablation.sh), the comparison, the
# external transfer (with the eval; without it nothing rebuilds
# external_transfer/) and the paper tables, then, with two arms or more, the
# table across the arms (with one arm nothing rebuilds it: one arm has nothing
# to compare), each with this run's arms, PCA targets and variants and the other
# settings those jobs read (the selection variant, the Step-1 dir tag
# BEST_CAT_PCA_VARIANCE, BESTCAT_EXTRA_VARIANTS, ARCH_ARGS, CVS), as this run
# resolved them, and without the pasting shell's RUN_TOKEN, CODE_MANIFEST and
# compare pins.
# Array sizes are computed dynamically from the .conf files so adding or
# removing embeddings requires no script changes. Adding a NEW embedding TYPE
# needs more than a catalog entry — see slurm/README.md "Add a new embedding".
#
# Prerequisites:
#   - Must be run from siamese_sl/ (the project working directory)
#   - SLURM cluster must be accessible (sbatch available)
#   - slurm/logs/ directory is created automatically
#   - No job of an earlier submission still queued or running that this one
#     would break: any pipeline job when it resets the venv or regenerates the
#     embeddings; otherwise the siamese-stage jobs that read or write what it
#     removes at submission (a best-per-category run: combo markers, selection,
#     ablation dirs, sl_comparison rows; --ablation-only: the ablation dirs).
#     Job names carry no PCA arm, so another arm's jobs count too. Refused
#     with a list; ALLOW_LIVE_JOBS=1 submits anyway (beside another arm's jobs
#     only when they read none of this submission's arms' roots: an owner's
#     single-embedding runs serve both arms of its pair)
#
# Usage:
#   cd siamese_sl
#
#   # (run as `bash slurm/...`: works whether or not the checkout kept the exec bit)
#
#   # Full pipeline: env reset → generate → (benchmark + best-per-category),
#   # every PCA arm
#   bash slurm/submit_pipeline.sh
#
#   # One PCA arm (or any subset): its own results root, the other arms'
#   # outputs left as they are (PCA_ARMS=joint alone trains its own
#   # single-embedding runs, in results/pca_joint/). --skip-generate is part
#   # of the recipe: without it the run regenerates the embeddings, which
#   # removes EVERY arm's combo markers (their combos were trained on the
#   # files it replaces), the arms it leaves out included.
#   PCA_ARMS=joint bash slurm/submit_pipeline.sh --skip-reset-env --skip-generate
#
#   # Skip the env reset (use existing venv as-is)
#   bash slurm/submit_pipeline.sh --skip-reset-env
#
#   # Skip generation (embeddings already exist; WARNs when embedding_versions.py
#   # cannot confirm them as builds of the current generator code)
#   bash slurm/submit_pipeline.sh --skip-generate
#
#   # Chain after an existing generation job. The env reset (unless
#   # --skip-reset-env) waits for that job to finish: it rm -rf's the venv the
#   # job runs from. Add --skip-reset-env to keep the venv it ran with.
#   # With a variant or arm subset (SIAMESE_VARIANTS_LIST, USE_CELL_LINES=0,
#   # PCA_ARMS) the other variants' and arms' <arm root>/*latest_combo_cv[123].txt
#   # are KEPT (this script cannot tell whether JOB regenerates): remove them by
#   # hand when it does, since their combos were trained on the vectors JOB
#   # replaces.
#   bash slurm/submit_pipeline.sh --after <JOB_ID>
#
#   # Run only one of the two training pipelines. --benchmark-only still
#   # REGENERATES the embeddings (it does not imply --skip-generate), and
#   # node2vec/go2vec are not bit-for-bit reproducible, so an earlier combo's
#   # checkpoints would no longer match the files on disk — pass
#   # --skip-generate to keep them. It retrains no combo, so it is REFUSED
#   # while an earlier run's *latest_combo_cv*.txt or best_per_category.json
#   # exist in any arm's results root and generation would run.
#   bash slurm/submit_pipeline.sh --benchmark-only
#   bash slurm/submit_pipeline.sh --best-cat-only
#
#   # Opt out of the chained external-screen eval (default: on when
#   # best-per-category runs, since the combo trained there is its target).
#   bash slurm/submit_pipeline.sh --skip-eval
#
#   # Everything except the all-gene-pair export (step 8).
#   bash slurm/submit_pipeline.sh --skip-pair-export
#
#   # Eval-only: run eval_finetuning.sh against an existing trained combo.
#   # Skips generation, benchmark, best-per-category AND the env reset: the
#   # reset rm -rf's the shared venv that any still-running job (ablation,
#   # shared models) executes from. Add --reset-env to force one anyway.
#   # It scores the combo's checkpoints against the .pt files on disk NOW: if
#   # a --benchmark-only run regenerated them since (that mode leaves
#   # generation ON), eval_finetuning.py refuses them on their recorded
#   # embeddings_sha256 and the task exits 1. The all-gene-pair export of that
#   # combo follows the eval (it scores the models the eval fine-tunes), unless
#   # --skip-pair-export is passed. Both write inside the combo and are
#   # submitted as the PCA arm whose results root holds it (none outside
#   # results/pca_<arm>/), for their logs.
#   bash slurm/submit_pipeline.sh --eval-only results/best_cat_..._cv3
#   bash slurm/submit_pipeline.sh --eval-only results/pca_joint/best_cat_..._cv3
#
#   # Ablation-only: retrain the ablations of the existing combos of every arm
#   # of PCA_ARMS. Joins the runs that selected them (each arm's
#   # best_per_category.json _selection_meta.run_token) and removes their
#   # previous ablate_no_* dirs at submission. Refused unless every arm has its
#   # selection and at least one combo marker; a (variant, CV) without a marker
#   # is named (its tasks SKIP). Export the PCA targets (PCA_MODALITY_VARIANCE /
#   # PCA_JOINT_VARIANCE / PCA_JOINT_DIM) and ARCH_ARGS the combos were trained
#   # with: an arm whose marker-named combos' config.json records another PCA
#   # than this shell gives it is refused before anything changes (each task
#   # would stop on it), with what to submit instead, each command complete
#   # (the settings taken from the calling shell, the targets in full): ONE
#   # when one set of targets trains the combos of every arm (always so for a
#   # forgotten or stray target); else one per arm, or per group of variants
#   # where those of an arm differ, to submit one after the other (a later one
#   # with ALLOW_LIVE_JOBS=1 once squeue lists only the earlier ones' jobs),
#   # then for a split arm the job that rebuilds its ablation_summary.csv over
#   # all its variants, and first a --best-cat-only retrain of an arm with
#   # combos no targets train in it (of this run's variants and every other
#   # one it holds a combo of; with USE_CELL_LINES other than 1 the
#   # single-output model alone, the refusal naming the cell-line combos it
#   # leaves; in the same venv unless --reset-env is given). Every such line
#   # sets USE_CELL_LINES and its variants as this run resolved them.
#   bash slurm/submit_pipeline.sh --ablation-only
#   PCA_ARMS=per_modality bash slurm/submit_pipeline.sh --ablation-only
#
# Environment variables:
#   ALLOW_LIVE_JOBS       1 downgrades the "earlier submission still live" refusal
#                         to a WARNING. Only for a queue whose pipeline jobs are
#                         known not to need the venv this run resets, the
#                         embeddings it regenerates or the outputs it removes
#   PCA_ARMS              The PCA arms this submission trains (config.conf;
#                         default "none joint per_modality both"; any subset,
#                         always run in that order): every per-arm step once per
#                         arm, steps 2-4 once per owner. A name config.conf does
#                         not know, or one listed twice, is refused before
#                         anything changes. A subset leaves the other arms'
#                         outputs alone only with --skip-generate: a run that
#                         regenerates removes every arm's combo markers. Its
#                         owner rebuilds a pair's single-embedding runs, which
#                         the pair's other arm, when left out, may have ranked
#                         (a WARNING says so: that arm's tab:solo is then
#                         another run's until it is rerun)
#   PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE / PCA_JOINT_DIM
#                         The arms' PCA targets (config.conf): the variance kept
#                         within each modality (per_modality, both; default 0.8)
#                         and on the concatenation (joint, both; default 0.8), or
#                         an exact count for the latter (PCA_JOINT_DIM, empty by
#                         default; it wins: ONE count for joint and both, so keep
#                         it below the narrowest concatenation their joint PCA
#                         meets, the ablations' included; train.py refuses a
#                         count at or above it, since 2026-10-21, at the fit of
#                         the first fold that narrow in each combo or ablation
#                         run, which no check here can see: in arm both,
#                         earlier folds and earlier combos may have trained by
#                         then). POST_PCA_VARIANCE / POST_PCA_DIM are
#                         refused: PCA_ARMS=joint is that pipeline now
#   PCA_ARM               The arm of ONE job (config.conf): ignored here, with a
#                         NOTE; every arm's jobs get their own in sbatch's
#                         environment (PCA_ARM=<arm> sbatch ... --export=ALL)
#   BEST_CAT_PCA_VARIANCE Results-dir tag of the single-embedding runs only
#                         (<owner root>/pcavar<value>_*; default 0.8). It sets NO
#                         PCA: the owner's arm does (none for results/,
#                         PCA_MODALITY_VARIANCE for results/pca_per_modality/)
#   EVAL_MODEL_DIR / EVAL_VARIANT
#                         Direct sbatch of eval_finetuning.sh only. --eval-only sets
#                         EVAL_MODEL_DIR itself; a full run clears both before the
#                         chained eval, since either would collapse its
#                         cv_type x variant array to one model
#   PAIR_EXPORT_*         What the all-gene-pair export writes (see
#                         slurm/export_pair_predictions.conf): PAIR_EXPORT_CVS
#                         (default cv3), PAIR_EXPORT_UNIVERSE (all = 19,397
#                         genes, 1.5 GB per matrix; shared = 8,844, 0.31 GB),
#                         PAIR_EXPORT_DTYPE, PAIR_EXPORT_PER_FOLD,
#                         PAIR_EXPORT_LOGITS, PAIR_EXPORT_FT_SPLITS / _DATASETS /
#                         _MODES (the fine-tuned sets; default Adamson, cv1
#                         screen split). PAIR_EXPORT_MODEL_DIR: a direct sbatch
#                         of export_pair_predictions.sh only; a full run clears
#                         it, --eval-only sets it itself. A value the conf
#                         does not list is refused here, before anything is
#                         removed or queued (dataset and mode names are
#                         matched by the job)
#   GPU_NODES             Node(s) for every job of this submission, e.g. "gpu4":
#                         passed as --nodelist, which replaces each script's
#                         `#SBATCH --nodelist=gpu3`. A list ("gpu3,gpu4") lets a
#                         one-node job take whichever listed node is free on
#                         Slurm >= 23.02 only (older Slurm rejects it).
#   GPU_EXCLUDE           Only together with GPU_NODES, sharing no node with it:
#                         --exclude cannot lift the scripts' --nodelist=gpu3, and
#                         the submission is refused otherwise. Both are read in
#                         config.conf and default to empty (each script's pin).
#                         Before anything changes, `sbatch --test-only` checks
#                         each stage's job script on the selection (or the pin),
#                         and sinfo prints the node state.
#   MAX_CONCURRENT        Optional cap on each benchmark, best-cat and ablation
#                         array on its own: every owner submits its own benchmark
#                         and best-cat arrays and every arm its own ablation and
#                         eval arrays, so with all four arms up to 4 x
#                         (MAX_CONCURRENT + EVAL_CONCURRENT) tasks run in the
#                         stage after the combos. A positive integer N adds `%N`
#                         to --array; empty (the default), 0, none, all or
#                         unlimited = no cap. EVAL_CONCURRENT: the same for the
#                         eval arrays. Anything else is refused before anything
#                         changes.
#   SELECTION_CV          Which CV to use when picking best-per-category winners (default cv1)
#   PREPROCESSING_FIT_SCOPE  "train" (default, leak-free) or "all" (legacy leaky)
#   PCA_METHOD            "plain" (default, classical SVD) or "robust" (ROBPCA);
#                         the preprocessing of every arm, not a PCA arm
#   SIAMESE_ENCODER_TYPE  "residual" (default) or "mlp"
#   SIAMESE_VARIANTS_LIST Space-separated KERNEL:CELL_LINES list (default since
#                         2026-11-13: "pd:pooled pd:per_head"; any of
#                         pd|bilinear x per_head|pooled). Scales the eval and
#                         ablation arrays and the combo job's run count. A scalar,
#                         because bash cannot export arrays: config.conf builds
#                         the SIAMESE_VARIANTS array from it and re-exports the
#                         validated list, so every job sees the list this script
#                         sized its arrays for. SIAMESE_VARIANTS="..." on this
#                         script's command line is accepted as an alias.
#   SELECTION_VARIANT     Variant the single-embedding runs train (default pd:pooled since 2026-11-13)
#   SELECTION_FRAC        train.py --selection_frac on every siamese training: a
#                         random fraction of each fold's training pairs held out
#                         as a modality-selection split (default 0.1 since
#                         2026-11-13; empty or 0: off). SELECTION_SEED its seed (42). Export the same
#                         values for a later --ablation-only: they are signature
#                         keys, so an ablation trained without them stops
#   SELECTION_MODE        global (winners ranked once on SELECTION_CV; default
#                         when SELECTION_FRAC is off)
#                         | per_fold (each CV and fold on its own selection
#                         split; needs SELECTION_FRAC > 0; the default when
#                         SELECTION_FRAC > 0). config.conf
#   ARCH_ARGS             Extra train.py flags appended to every siamese training
#                         call, e.g. "--layernorm --log_temperature
#                         --no_per_head_last_layer_bias --select_metric auprg
#                         --no_gene_term --no_pair_features" for the published
#                         model (which also needs ENCODER_DIMS at width 16,
#                         edited in config.conf, in PCA arm joint at
#                         PCA_JOINT_VARIANCE=0.8). A PCA flag is refused here:
#                         the arms set the PCA
#   BESTCAT_EXTRA_VARIANTS  1 also trains every single-modality run under the other
#                         SIAMESE_VARIANTS, in the Step-1 array (reporting only;
#                         never ranked). Default 1 (since 2026-11-13). Sized by bestcat_n_tasks
#                         (run_best_per_category.conf)
#   RUN_TOKEN             Id of this submission, made fresh here (an inherited one
#                         is replaced), except that --ablation-only reuses the
#                         token of the run whose combos it ablates (each arm's
#                         best_per_category.json _selection_meta.run_token), so
#                         paper_tables accepts its Step-1 and ablation dirs
#                         together; arms whose selections carry different
#                         tokens get one each (PCA_ARM=<arm> RUN_TOKEN=<token>
#                         sbatch ... --export=ALL; with a WARNING). Step-1 / benchmark /
#                         ablation runs record it; the combo and the summaries
#                         refuse another run's directories
#
# Every sbatch invocation below uses `--export=ALL` so any env var set above
# (or in the calling shell) propagates to the compute node. Without this,
# overrides are silently dropped on clusters configured with non-ALL
# SBATCH_EXPORT defaults. Plain ALL, never a list (--export=ALL,VAR=value):
# Slurm reads a list as an implied --get-user-env (see _as_arm below).
# ============================================================================

set -e

# MAX_CONCURRENT: optional cap on simultaneous array tasks (config.conf; empty
# = none). Applied to every benchmark, best-per-category and ablation array,
# each on its own. Each owner's benchmark and best-cat arrays run concurrently
# (2xN per owner), and later every arm's ablation array runs alongside its
# external-screen eval, so the standing ceiling is |PCA arms| x
# (MAX_CONCURRENT + EVAL_CONCURRENT) tasks in the tail stage, when either is set
# (both are EMPTY by default: no throttle, SLURM runs as many array tasks as
# the partition allows). EVAL_CONCURRENT caps the eval arrays separately because
# their task count is |EVAL_CV_TYPES| x |SIAMESE_VARIANTS| (12 by default), not
# set by the embedding catalog.
# Both default to EMPTY in config.conf = unlimited (no %N on --array).
# Every sbatch with an afterok dependency carries KILL: a failed parent cancels
# the child instead of leaving it pending (DependencyNeverSatisfied) forever,
# which would also hang every afterany consumer. No-op without a dependency.
KILL="--kill-on-invalid-dep=yes"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# This submission's flags, each quoted (printf %q: bash and zsh read it back),
# for the commands a refusal prints to rerun it (--ablation-only's PCA check),
# and the same without --reset-env, for the lines of that refusal after its
# first: each line is a submission of its own, and a later one's reset would
# delete the venv under the jobs an earlier line queued.
_ARGS_Q=$(printf ' %q' "$@")
[ $# -gt 0 ] || _ARGS_Q=""
_ARGS_Q_NORESET=""
for _x in "$@"; do
    [ "$_x" != --reset-env ] || continue
    printf -v _q '%q' "$_x"
    _ARGS_Q_NORESET="$_ARGS_Q_NORESET $_q"
done
# Settings this run honours from the calling shell, recorded BEFORE
# config.conf gives the unset ones their defaults, with their values as
# NAME=<value, printf %q> words read back alike by bash and zsh: the name list
# of run_benchmarking_siamese_vs_others.sh (test_slurm_contracts holds the two
# to it), plus SHARED_FOLDS, which this script honours (the root refuses one).
# Every job is submitted with --export=ALL, so each one named here is also
# named under the node check (nothing is printed when none is set). Since
# 2026-10-21 every command this script prints to run after it (the
# --ablation-only refusals' submissions, a refused export) carries them, but
# for the lines of the block it ends with, whose _paste_env sets the settings
# their jobs read (below): a one-shot
# SEED=7, ARCH_ARGS=..., SIAMESE_VARIANTS_LIST=... or PCA target in front of
# this script never reached the shell such a command is pasted into, and a
# refused --ablation-only's command, pasted there, was refused again (the
# other target printed, endlessly), or accepted at seed 42: it removed the
# arm's ablation and queued tasks that could only fail at their signature
# check. _ENV_PREFIX (a refused export's command) leaves out PCA_ARMS and
# ALLOW_LIVE_JOBS, as the root's does; _ENV_SAME_NOPCA keeps both, for the
# same submission again, and leaves out the PCA targets, which that command
# sets in full (pca_targets_with); _ENV_NOPCA leaves out all five, for the
# commands that set their own arms and targets and must never lift the
# live-job guard (pasted as a block, each would pass a real conflict).
# SIAMESE_VARIANTS is config.conf's alias for an empty list: beside
# SIAMESE_VARIANTS_LIST it reaches no job, and is named as ignored.
_ENV_NOTES=""; _ENV_PREFIX=""; _ENV_SAME_NOPCA=""; _ENV_NOPCA=""
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
          PAIR_EXPORT_RAM_GB SHARED_FOLDS; do
    # SELECTION_FRAC when set at all (2026-11-13): empty is the documented
    # way to turn the split off now that the default is 0.1, so it must
    # reach every job and printed command as SELECTION_FRAC= .
    if [ "$_v" = SELECTION_FRAC ]; then
        [ -n "${SELECTION_FRAC+x}" ] || continue
    else
        [ -n "${!_v:-}" ] || continue
    fi
    if [ "$_v" = SIAMESE_VARIANTS ] && [ -n "${SIAMESE_VARIANTS_LIST:-}" ]; then
        _ENV_NOTES="${_ENV_NOTES}NOTE: ignoring SIAMESE_VARIANTS='${!_v}' from the calling shell: SIAMESE_VARIANTS_LIST is set too, and wins."$'\n'
        continue
    fi
    _ENV_NOTES="${_ENV_NOTES}NOTE: $_v='${!_v}' comes from the calling shell: every job of this run gets it."$'\n'
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
done
unset _v _q _x

# One id for this whole submission, exported BEFORE config.conf is sourced so
# config.conf keeps it and every sbatch --export=ALL below carries it. Fresh
# (as in run_benchmarking_siamese_vs_others.sh): a token inherited from the
# shell (exported to join an earlier run by a direct sbatch) would let that
# run's never-restarted Step-1 / ablation directories pass as this one's.
# --ablation-only replaces it with the ablated runs' token further down.
_INHERITED_RUN_TOKEN="${RUN_TOKEN:-}"
RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"
export RUN_TOKEN
# PCA_ARM is the arm of ONE job (config.conf). This script trains every arm of
# PCA_ARMS and hands each arm's jobs their own in sbatch's environment
# (_as_arm: PCA_ARM=<arm> sbatch ... --export=ALL), so one exported in the
# calling shell (meant for a direct sbatch) is dropped before config.conf
# reads it: it would only make this shell's per-job roots another arm's, or
# stop config.conf on a mistyped value. Named under the banner.
_INHERITED_PCA_ARM="${PCA_ARM:-}"
unset PCA_ARM

# Source shared config (works from login node — SLURM-specific parts guarded)
source "$SCRIPT_DIR/config.conf"

# ============================================================================
# PWD sanity check
# ============================================================================
# Must be invoked from siamese_sl/ so the submit-time PWD equals WORK_DIR.
# sbatch sets SLURM_SUBMIT_DIR to the caller's PWD; on the compute node,
# config.conf uses that as WORK_DIR and `cd`s to it. If the caller's PWD
# isn't siamese_sl/, the job lands in the wrong directory and fails to
# locate the Python scripts. Catch this early, before any job is queued.
if [ "$PWD" != "$WORK_DIR" ]; then
    echo "ERROR: submit_pipeline.sh must be run from the siamese_sl/ directory."
    echo "  Current PWD:  $PWD"
    echo "  Required:     $WORK_DIR"
    echo "  Fix:          cd \"$WORK_DIR\" and re-run."
    exit 1
fi

# Code the submitted jobs import or read by path. A file never uploaded to this
# checkout (or never committed on the machine the tree comes from) would
# otherwise surface only hours in, e.g. as an ImportError in the combo's Step 2
# after the whole Step-1 array has run.
# embedding_versions.py: the generation job's closing provenance check.
for _f in "$WORK_DIR/arch_signature.py" "$BASE_DIR/sl_comparison/model_params.py" \
          "$BASE_DIR/sl_comparison/auprg.py" "$WORK_DIR/embedding_versions.py"; do
    [ -f "$_f" ] || { echo "ERROR: $_f missing (never uploaded to this checkout? copy it here, then resubmit)"; exit 1; }
done
unset _f

# ---------------------------------------------------------------------------
# How many cv_types the external-screen eval will run.
# EVAL_CV_TYPES lives in eval_finetuning.conf, which this script does not source
# (it resolves MODEL_DIR, which can legitimately fail on the login node). Read
# just that array in a subshell and fall back to 3 (the conf's default). The
# chained array exits 0 for a task id beyond the list; an --eval-only
# (EVAL_MODEL_DIR) task beyond it FAILS, as an override-collapsed array would.
eval_cv_count() {
    local n
    n=$( ( set +u; source "$SCRIPT_DIR/eval_finetuning.conf" >/dev/null 2>&1; \
           echo "${#EVAL_CV_TYPES[@]}" ) 2>/dev/null )
    case "$n" in ''|*[!0-9]*|0) echo 3 ;; *) echo "$n" ;; esac
}

# ---------------------------------------------------------------------------
# One submission, several PCA arms (config.conf PCA_ARM_LIST). Job ids of one
# arm (or one owner of single-embedding runs) are kept as <NAME>_<arm>, e.g.
# COMBO_JOB_per_modality (bash 3.2 has no associative arrays): an arm's steps
# run on the plain names (_arm_reset at its start, _arm_save at its end) and
# _arm_load brings them back for its export after every arm is queued.
_OWNER_VARS="BENCH_JOB BENCH_SUMM BESTCAT_JOB"
_ARM_VARS="COMBO_JOB EVAL_JOB ABL_JOB ABL_SUMM PAIR_JOB ALLMOD_JOB"
_arm_reset() { local v; for v in $_ARM_VARS; do printf -v "$v" '%s' ""; done; }
_arm_save() { local v; for v in $_ARM_VARS; do printf -v "${v}_$1" '%s' "${!v}"; done; }
_arm_load() { local v n; for v in $_ARM_VARS; do n="${v}_$1"; printf -v "$v" '%s' "${!n}"; done; }
# _armtag ARM -> "" for arm none, "pca_<ARM>_" otherwise: the arm in the name of
# each of its jobs' log files.
_armtag() { local d; d=$(pca_arm_dir "$1"); echo "${d:+${d}_}"; }
# _logs ARM STAGE PATTERN -> the --output / --error of one of ARM's jobs:
# nothing for arm none (each script's own #SBATCH --output, slurm/logs/
# <stage>_<pattern>.out), else the same names with the arm in them. Relative to
# siamese_sl/, the submit dir, as the scripts' own.
_logs() {
    local t
    t=$(_armtag "$1")
    [ -n "$t" ] || return 0
    echo "--output=slurm/logs/${2}_${t}${3}.out --error=slurm/logs/${2}_${t}${3}.err"
}
# _as_arm ARM sbatch ARGS...: submit one of ARM's jobs, its PCA_ARM (plus its
# own RUN_TOKEN when the arms joined runs of different tokens, --ablation-only)
# in sbatch's OWN environment, as prefix assignments (never exported in this
# shell, so no later job inherits them), with a plain --export=ALL in ARGS:
# sbatch hands the job its environment. Without its own token the job gets
# this shell's exported RUN_TOKEN, as every other job. The jobs of every arm
# (reset_env, the generation) carry no arm of their own: they inherit this
# shell's PCA_ARM=none (config.conf exports its default), which a job without
# one resolves anyway. Since 2026-10-21, as in the root orchestrator; the list
# form --export=ALL,PCA_ARM=<arm>[,RUN_TOKEN=<token>] of 2026-10-20 is avoided
# on purpose: Slurm treats any --export value but ALL / NIL / NONE as an
# implied --get-user-env (slurmd runs `su - <user> -c env` at every task
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
# The owners of single-embedding runs among the arms this submission trains
# (config.conf pca_step1_arm), in PCA_ARMS order: none and per_modality by
# default; joint or both when it runs without the other arm of its pair.
_OWNERS=""
for _a in "${PCA_ARM_LIST[@]}"; do
    _o=$(pca_step1_arm "$_a")
    case " $_OWNERS " in *" $_o "*) ;; *) _OWNERS="${_OWNERS:+$_OWNERS }$_o" ;; esac
done
# _served OWNER -> the arms of this submission whose single-embedding runs
# OWNER trains.
_served() {
    local a out=""
    for a in "${PCA_ARM_LIST[@]}"; do
        [ "$(pca_step1_arm "$a")" = "$1" ] && out="${out:+$out }$a"
    done
    echo "$out"
}
unset _a _o

# ============================================================================
# Parse command-line arguments
# ============================================================================

RESET_ENV=true
# Set only by an explicit --reset-env: --eval-only resets the venv only then.
RESET_ENV_EXPLICIT=false
SKIP_GENERATE=false
AFTER_JOB=""
RUN_BENCHMARK=true
RUN_BEST_CAT=true
RUN_EVAL=true
EVAL_ONLY_DIR=""
# Ablation runs as a normal stage. It retrains the combo once per (variant,
# dropped category, CV) and never touches the reported models — the reference row is
# the already-trained combo, which is not retrained. --skip-ablation opts out.
RUN_ABLATION=true
ABLATION_ONLY=false
# All-gene-pair export (slurm/export_pair_predictions.sh), the last job of each
# arm. Only when this run trains combos, or after --eval-only's eval;
# --skip-pair-export opts out.
RUN_PAIR=true
# Job ids start empty: an exported variable of the same name in the calling
# shell must never become a dependency below, nor one of another arm.
ENV_JOB=""; GEN_JOB=""; _QUEUED=""
BENCH_JOB=""; BENCH_SUMM=""; BESTCAT_JOB=""
_arm_reset
for _a in none joint per_modality both; do
    for _v in $_OWNER_VARS $_ARM_VARS; do printf -v "${_v}_$_a" '%s' ""; done
    printf -v "_ARM_TOKEN_$_a" '%s' ""
done
unset _a _v

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset-env)
            # The default for a full run; with --eval-only it opts INTO the reset.
            RESET_ENV=true; RESET_ENV_EXPLICIT=true; shift ;;
        --skip-reset-env)
            RESET_ENV=false; shift ;;
        --skip-generate)
            SKIP_GENERATE=true; shift ;;
        --after)
            case "${2:-}" in
                ''|*[!0-9_]*|_*|*_|*_*_*)
                    echo "ERROR: --after requires a job ID argument (<id> or <id>_<task>), got '${2:-}'"
                    exit 1 ;;
            esac
            AFTER_JOB="$2"; shift 2 ;;
        --benchmark-only)
            RUN_BEST_CAT=false; RUN_EVAL=false; RUN_ABLATION=false; shift ;;
        --best-cat-only)
            RUN_BENCHMARK=false; shift ;;
        --skip-eval)
            RUN_EVAL=false; shift ;;
        --ablation)
            RUN_ABLATION=true; shift ;;       # retained for back-compat (now the default)
        --skip-ablation)
            RUN_ABLATION=false; shift ;;
        --skip-pair-export)
            RUN_PAIR=false; shift ;;
        --ablation-only)
            # Reuses the existing best_per_category.json and the already-trained
            # reference combo; retrains nothing else. No env reset unless
            # --reset-env is also given (resolved after the loop, so the flag
            # order does not matter).
            RUN_ABLATION=true; ABLATION_ONLY=true
            RUN_BENCHMARK=false; RUN_BEST_CAT=false; RUN_EVAL=false
            SKIP_GENERATE=true; shift ;;
        --eval-only)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --eval-only requires a combo model directory argument"
                exit 1
            fi
            EVAL_ONLY_DIR="$2"; RUN_ABLATION=false; shift 2 ;;
        -h|--help)
            # Print the file-top doc block (all leading lines that start with '#').
            awk '/^#/{print} !/^#/{exit}' "$0" | tail -n +2 | sed -E 's/^# ?//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# --ablation-only skips the env reset (it rm -rf's the venv that still-running
# jobs execute from) unless --reset-env is passed, in either flag order.
if [ "$ABLATION_ONLY" = true ] && [ "$RESET_ENV_EXPLICIT" != true ]; then
    RESET_ENV=false
fi

# ABLATION_POST_PCA_DIM (an ablation-only width pin) was removed: the ablation
# must match its reference combo's architecture signature, its PCA included,
# so run_ablation.sh refused every task. Stop here, before any job is queued,
# rather than submit a pipeline whose ablation cannot run.
if [ "$RUN_ABLATION" = true ] && [ -n "${ABLATION_POST_PCA_DIM:-}" ]; then
    echo "ERROR: ABLATION_POST_PCA_DIM is no longer supported. To control the"
    echo "  ablation for capacity, train PCA arm joint at an exact width, which its"
    echo "  combo AND its ablation take (e.g. PCA_ARMS=joint PCA_JOINT_DIM=<k> bash"
    echo "  slurm/submit_pipeline.sh --skip-generate: without --skip-generate the run"
    echo "  regenerates the embeddings, which removes EVERY arm's combo markers); see"
    echo "  slurm/config.conf, PCA arms."
    exit 1
fi

# The all-gene-pair export is the last job of the run: a file that never
# reached this checkout would otherwise surface only after everything else.
if [ "$RUN_PAIR" = true ] && { [ "$RUN_BEST_CAT" = true ] || [ -n "$EVAL_ONLY_DIR" ]; }; then
    for _f in "$WORK_DIR/export_pair_predictions.py" "$SCRIPT_DIR/export_pair_predictions.sh" \
              "$SCRIPT_DIR/export_pair_predictions.conf"; do
        [ -f "$_f" ] || { echo "ERROR: $_f missing (never uploaded to this checkout? copy it here, or pass --skip-pair-export)"; exit 1; }
    done
    unset _f
    # So would a mistyped PAIR_EXPORT_* value. In a subshell: the conf's
    # defaults must reach neither this shell nor the jobs' environment.
    if ! ( source "$SCRIPT_DIR/export_pair_predictions.conf" && check_pair_export ); then
        echo "  Fix it (slurm/export_pair_predictions.conf lists the values) or pass"
        echo "  --skip-pair-export. Nothing was changed or submitted."
        exit 1
    fi
fi

# CVS (the CVs of the comparison) reaches no job of this script: only the
# block it ends with, which hands it to the pasted compare (2026-10-21). A
# value that is not a list of cv1 / cv2 / cv3 (CVS=cv1,cv2, a typo, a name
# twice) would leave that compare without a SLeuth row: refused here, as the
# root orchestrator refuses it, whenever that block is printed, and with
# --ablation-only, whose refusals print --best-cat-only lines that carry it
# (2026-10-22: such a line, run as printed, was refused here).
if { [ "$RUN_BEST_CAT" = true ] || [ "$ABLATION_ONLY" = true ]; } \
        && [ -z "$EVAL_ONLY_DIR" ] && [ -n "${CVS:-}" ]; then
    _CVS_SEEN=" "
    for _cv in $CVS; do
        case "$_cv" in cv1|cv2|cv3) ;; *) _CVS_SEEN=""; break ;; esac
        case "$_CVS_SEEN" in *" $_cv "*) _CVS_SEEN=""; break ;; esac
        _CVS_SEEN="$_CVS_SEEN$_cv "
    done
    if [ -z "${_CVS_SEEN// /}" ]; then
        echo "ERROR: CVS='$CVS': want a space-separated subset of cv1 cv2 cv3, each once"
        if [ "$ABLATION_ONLY" = true ]; then
            echo "  (the CVs of the comparison of the --best-cat-only lines an --ablation-only"
            echo "  refusal prints, which carry it). Nothing was changed or submitted."
        else
            echo "  (the CVs of the comparison the block this run ends with submits). Nothing"
            echo "  was changed or submitted."
        fi
        exit 1
    fi
    unset _CVS_SEEN _cv
fi
# Node selection and array caps (config.conf), before anything is removed or
# queued: SLURM would reject a bad one only at its sbatch. This script submits
# no shared-model array, so SHARED_CONCURRENT is not checked.
if ! check_throttles MAX_CONCURRENT EVAL_CONCURRENT || ! check_node_selection; then
    echo "  Nothing was changed or submitted."
    exit 1
fi
# run_best_per_category.conf refuses a BESTCAT_EXTRA_VARIANTS other than 0 or 1
# and a SELECTION_CV that is not one of CV_TYPES,
# but it is sourced only at Steps 4 and 7 below (after the previous outputs are
# removed and the first jobs are queued) and at the END of the generation job
# (--export=ALL carries the value there). Check it here whenever a job of this
# submission sources it. In a subshell: its values must not reach this shell
# before the benchmark is submitted.
if [ -z "$EVAL_ONLY_DIR" ] && { [ "$RUN_BEST_CAT" = true ] || [ "$RUN_ABLATION" = true ] \
       || { [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; }; }; then
    if ! ( source "$SCRIPT_DIR/run_best_per_category.conf" ); then
        echo "  Nothing was changed or submitted."
        exit 1
    fi
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not on PATH (run this on the SLURM login node); nothing was"
    echo "  changed or submitted."
    exit 1
fi
# `sbatch --test-only` submits nothing: slurmctld answers whether it would
# accept each stage's job script on this node selection (GPU_NODE_ARGS, or the
# scripts' own #SBATCH --nodelist=gpu3 when empty), resources included. Once
# per script: every arm submits the same scripts.
_PROBES=""
[ "$RESET_ENV" = true ] && _PROBES="$_PROBES reset_env.sh"
if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ] && [ -z "$EVAL_ONLY_DIR" ]; then
    _PROBES="$_PROBES run_generate_embeddings.sh"
fi
if [ "$RUN_BEST_CAT" = true ] && [ -z "$EVAL_ONLY_DIR" ]; then
    _PROBES="$_PROBES run_best_per_category_combo.sh"
    [ "$ALLMOD_COMBO" != 1 ] || _PROBES="$_PROBES run_allmod_combo.sh"
fi
[ "$RUN_ABLATION" = true ] && _PROBES="$_PROBES run_ablation.sh"
{ [ "$RUN_EVAL" = true ] || [ -n "$EVAL_ONLY_DIR" ]; } && _PROBES="$_PROBES eval_finetuning.sh"
# (no quote characters inside ${VAR:-...}: bash 3.2 parses them there)
if [ -n "$GPU_NODE_ARGS" ]; then _SEL="$GPU_NODE_ARGS"
else _SEL="each script's #SBATCH --nodelist=gpu3"; fi
# A down or drained node is no submission error: the jobs would just pend.
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
    if ! _TEST=$(sbatch --test-only $GPU_NODE_ARGS "$SCRIPT_DIR/$_p" 2>&1); then
        if [ -n "$_NODE_DOWN" ]; then
            _PROBED="not probed: sbatch --test-only cannot place a job on the node state below"
            break
        fi
        echo "ERROR: SLURM would reject slurm/$_p on $_SEL:"
        printf '%s\n' "$_TEST" | sed 's/^/  /'
        echo "  Nothing was changed or submitted."
        exit 1
    fi
done
echo "Node selection: $_SEL ($_PROBED)"
if [ -n "$_STATE" ]; then
    printf '%s\n' "$_STATE" | sed 's/^/  node state: /'
    if [ -n "$_NODE_DOWN" ]; then
        echo "WARNING: a node these jobs are pinned to is down, drained, failing or not"
        echo "  responding: they pend until it is back (GPU_NODES moves them)."
    fi
fi
# One line per setting inherited from the calling shell (recorded at the top,
# as the root names them under its banner; the banner below names some values,
# never where they came from), before anything is removed or queued.
[ -z "$_ENV_NOTES" ] || printf '%s' "$_ENV_NOTES"

# ============================================================================
# SLURM reachable, and no earlier submission still live
# ============================================================================
# The same check as run_benchmarking_siamese_vs_others.sh, whose job-name list
# already covers this script's jobs. With no argument, for the paths that reset
# the venv or regenerate the embeddings: reset_env.sh rm -rf's the venv
# (config.conf's PYTHON_PATH) that every queued or running job executes from,
# and the generation job deletes every ALL_EMBEDDINGS .pt before rebuilding it,
# so either one breaks a submission that is already in the queue. A path that
# does neither is checked only against the jobs whose inputs or outputs it
# removes (`outputs` / `ablation` below). That leaves alone --eval-only
# without --reset-env (the documented recovery step taken while the rest of a
# pipeline runs: never checked) and `--skip-reset-env --best-cat-only --after
# $GEN`, which is how sl_comparison/README.md's manual DAG invokes this script
# with reset_env, siamese_gen_emb, sl_folds and sl_compare queued. The job
# names carry no PCA arm (every arm submits the same scripts), so a live job
# of an arm this submission leaves out is listed like one of its own.
_check_no_live_jobs() {
    local _q _live _lvl _pipe _after
    if ! command -v sbatch >/dev/null 2>&1 || ! command -v squeue >/dev/null 2>&1; then
        echo "ERROR: sbatch / squeue not on PATH (run this on the SLURM login node);"
        echo "  nothing was changed or submitted."
        exit 1
    fi
    if ! _q=$(squeue -h -u "${USER:-$(whoami)}" -o '%i %100j' 2>&1); then
        echo "ERROR: squeue failed (slurmctld unreachable?); nothing was changed or submitted:"
        printf '%s\n' "$_q" | sed 's/^/  /'
        exit 1
    fi
    _pipe='^(reset_env|siamese_gen_emb|bench_emb|bench_summary|bestcat_emb|bestcat_combo|eval_finetuning|ablation|ablation_summary|cellline_ablation|bilinear_probe|allmod_combo|ELISL|elisl_inputs|pair_export|sl_preflight|sl_folds|sl_compare|sl_final_compare|sl_ext_transfer|sl_paper_tables)$'
    # --after's JOB is ordered before this submission's JOBS, so it is no
    # conflict for the venv or the embeddings.
    _after="${AFTER_JOB%%_*}"
    # A submission that neither resets the venv nor regenerates still removes,
    # further down and before any sbatch, what the siamese stages of a live run
    # read and write (slurm/clean.sh refuses the same removal). Scope:
    #   outputs   a best-per-category submission: the combo markers, the
    #             selection, the ablation dirs and the sl_comparison rows
    #   ablation  --ablation-only: the <arm root>/<prefix>ablate_no_* dirs,
    #             which it retrains under the ablated run's own token
    # Only the jobs that read or write those are listed: sl_comparison/README.md's
    # manual DAG calls this script with reset_env, siamese_gen_emb, sl_folds and
    # sl_compare queued. The removal happens now, at submission, so --after's
    # job is not exempt here (a generation or reset job is in neither list).
    case "${1:-}" in
        outputs)  _after=""; _pipe='^(bestcat_emb|bestcat_combo|allmod_combo|bilinear_probe|eval_finetuning|ablation|ablation_summary|cellline_ablation|pair_export|sl_final_compare|sl_ext_transfer|sl_paper_tables)$' ;;
        ablation) _after=""; _pipe='^(bestcat_emb|bestcat_combo|ablation|ablation_summary)$' ;;
    esac
    _live=$(printf '%s\n' "$_q" | awk -v re="$_pipe" -v after="$_after" \
        'NF >= 2 { split($1, id, "_"); if ($2 ~ re && id[1] != after) print "  " $1 " (" $2 ")" }')
    [ -n "$_live" ] || return 0
    if [ "${ALLOW_LIVE_JOBS:-0}" = 1 ]; then _lvl=WARNING; else _lvl=ERROR; fi
    echo "$_lvl: jobs of an earlier submission are still queued or running:"
    printf '%s\n' "$_live" | head -20
    case "${1:-}" in
        outputs)
            echo "  They read or write the combo markers, modality selection, ablation dirs"
            echo "  and sl_comparison rows this submission removes and rebuilds." ;;
        ablation)
            echo "  They write the <arm root>/<prefix>ablate_no_* dirs this submission removes"
            echo "  and retrains under the same run token, or replace the combos it ablates." ;;
        *)
            echo "  They run from the venv reset_env deletes, and read the embeddings the"
            echo "  generation job deletes and rebuilds." ;;
    esac
    if [ "$_lvl" = ERROR ]; then
        echo "  (Job names carry no PCA arm: those of another arm are listed too.)"
        echo "  Wait for them or scancel them, then resubmit (ALLOW_LIVE_JOBS=1 submits"
        echo "  anyway). Nothing was changed."
        exit 1
    fi
    echo "  ALLOW_LIVE_JOBS=1: submitting anyway."
}

# Cell-line mode (USE_CELL_LINES=1, from config.conf) trains multi-head
# SiameseSLMultiCell checkpoints. eval_finetuning.sh now consumes them via
# PER-HEAD evaluation (PER_HEAD auto-enabled in eval_finetuning.conf when
# USE_CELL_LINES=1): each external screen is scored against its cell-line head
# (Adamson/Gilbert -> K562, Corn -> OTHER). The eval is no longer force-skipped;
# pass --skip-eval to opt out.
if [ "${USE_CELL_LINES:-0}" = "1" ] && [ "$RUN_EVAL" = true ]; then
    echo "NOTE: USE_CELL_LINES=1 -> external-screen eval runs PER-HEAD"
    echo "      (each screen scored against its cell-line head; --skip-eval to opt out)."
fi
if [ -n "$_INHERITED_PCA_ARM" ]; then
    echo "NOTE: ignoring PCA_ARM=$_INHERITED_PCA_ARM from the environment: this submission"
    echo "      trains the PCA arms of PCA_ARMS ($PCA_ARMS), each job with its own"
    echo "      (PCA_ARMS=<arm> trains one; --eval-only takes the arm of its directory)."
fi

# Eval-only takes an independent short path: no training and, unless --reset-env
# is passed explicitly, NO env reset. reset_env.sh rm -rf's the shared venv
# (PYTHON_PATH) that every other job runs from, and an eval-only run is
# typically a recovery step while the rest of a pipeline (ablation array,
# shared models) is still executing from that venv.
if [ -n "$EVAL_ONLY_DIR" ]; then
    if [ "$SKIP_GENERATE" = true ] || [ -n "$AFTER_JOB" ] \
       || [ "$RUN_BENCHMARK" = false ] || [ "$RUN_BEST_CAT" = false ]; then
        echo "ERROR: --eval-only cannot be combined with training flags."
        echo "  --eval-only only submits eval_finetuning.sh against an existing combo."
        echo "  --skip-ablation omits the leave-one-category-out ablation stage."
        echo "  --ablation-only runs just the ablation against an existing best_per_category.json."
        exit 1
    fi
    if [ ! -d "$EVAL_ONLY_DIR" ]; then
        echo "ERROR: --eval-only directory not found: $EVAL_ONLY_DIR"
        exit 1
    fi
    if [ ! -f "$EVAL_ONLY_DIR/config.json" ]; then
        echo "ERROR: $EVAL_ONLY_DIR/config.json missing — not a trained combo?"
        exit 1
    fi
    if [ ! -f "$EVAL_ONLY_DIR/results.json" ]; then
        echo "ERROR: $EVAL_ONLY_DIR/results.json missing — the combo did not finish training."
        exit 1
    fi
    mkdir -p "$WORK_DIR/slurm/logs"
    # The PCA arm whose results root holds the combo (results/pca_<arm>/<combo>;
    # none anywhere else), for its jobs' logs and banners: what they write goes
    # inside the combo, whatever the arm.
    _EO_ARM=none
    _EO_PARENT=$(cd "$EVAL_ONLY_DIR/.." 2>/dev/null && pwd -P) || _EO_PARENT=""
    for _a in joint per_modality both; do
        _r=$(pca_arm_results "$_a")
        if [ -n "$_EO_PARENT" ] && [ -d "$_r" ] \
           && [ "$_EO_PARENT" = "$(cd "$_r" && pwd -P)" ]; then
            _EO_ARM="$_a"
        fi
    done
    unset _a _r _EO_PARENT

    EVAL_DEP=""
    if [ "$RESET_ENV_EXPLICIT" = true ] && [ "$RESET_ENV" = true ]; then
        # The reset deletes the venv every OTHER queued/running job runs from.
        _check_no_live_jobs
        echo "--- Submitting environment reset (--reset-env) ---"
        ENV_JOB=$(sbatch --parsable $GPU_NODE_ARGS --export=ALL "$SCRIPT_DIR/reset_env.sh")
        echo "  Reset env job: $ENV_JOB"
        EVAL_DEP="--dependency=afterok:$ENV_JOB"
    fi

    echo "--- Submitting eval_finetuning (target: $EVAL_ONLY_DIR; PCA arm $_EO_ARM) ---"
    # Export EVAL_MODEL_DIR normally; `--export=ALL` propagates it into the
    # job. This handles special characters in the path safely, unlike the
    # `--export=ALL,KEY=value` form which is bash-word-split on whitespace
    # (and which Slurm reads as an implied --get-user-env: _as_arm).
    export EVAL_MODEL_DIR="$EVAL_ONLY_DIR"
    EVAL_TASKS=$(eval_cv_count)
    EVAL_JOB=$(_as_arm "$_EO_ARM" sbatch --parsable $GPU_NODE_ARGS \
        $EVAL_DEP $KILL \
        --export=ALL $(_logs "$_EO_ARM" eval_finetuning %A_%a) \
        --array=0-$((EVAL_TASKS - 1))$(array_throttle "$EVAL_CONCURRENT") \
        "$SCRIPT_DIR/eval_finetuning.sh")
    echo "  Eval job: $EVAL_JOB  ($EVAL_TASKS cv_types, one per array task)"
    echo "  Tail:   tail -f $WORK_DIR/slurm/logs/eval_finetuning_$(_armtag "$_EO_ARM")${EVAL_JOB}_0.out"
    if [ "$RUN_PAIR" = true ]; then
        # The eval rewrites <combo>/eval_finetuning_<cv>/finetuned_models/, so
        # the all-gene-pair export of this combo follows it (afterany: a failed
        # eval task still leaves the SL probabilities to export). singleton:
        # --eval-only is also a recovery step beside a queued run, whose own
        # export of this combo must not overlap this one (each clears the
        # directory first) and may start before this eval has finished (it
        # then scores the models this eval replaces, or leaves out the sets
        # still being rewritten): SLURM starts one pair_export job of a user
        # at a time, in submission order, so this export runs after that one
        # and scores the models this eval wrote. The combo is passed like
        # EVAL_MODEL_DIR, through --export=ALL. A submission problem is a
        # WARNING: the eval is queued and does not depend on it. The command
        # it prints starts with the settings taken from the calling shell
        # (_ENV_PREFIX, 2026-10-21: the export reads PAIR_EXPORT_*, the
        # variants and SHARED_FOLDS; a one-shot one never reached the shell
        # it is pasted into), the combo and the arm after them.
        export PAIR_EXPORT_MODEL_DIR="$EVAL_ONLY_DIR"
        # (the arm rides in front of sbatch, _as_arm / _arm_env)
        _PAIR_OPTS="--export=ALL"
        _L=$(_logs "$_EO_ARM" pair_export %j); [ -z "$_L" ] || _PAIR_OPTS="$_PAIR_OPTS $_L"
        if PAIR_JOB=$(_as_arm "$_EO_ARM" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterany:$EVAL_JOB,singleton \
                $_PAIR_OPTS \
                "$SCRIPT_DIR/export_pair_predictions.sh"); then
            echo "  Pair-export job: $PAIR_JOB  -> $EVAL_ONLY_DIR/pair_predictions/ (--skip-pair-export to omit)"
        else
            echo "  WARNING: could not submit the all-gene-pair export. While the eval is queued:"
            echo "    ${_ENV_PREFIX}PAIR_EXPORT_MODEL_DIR=$EVAL_ONLY_DIR $(_arm_env "$_EO_ARM") sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --dependency=afterany:$EVAL_JOB,singleton $_PAIR_OPTS slurm/export_pair_predictions.sh"
            echo "  or the same without --dependency once it has finished."
        fi
    fi
    exit 0
fi

# Reject conflicting flags.
# --ablation-only legitimately disables both training stages (it reuses an
# existing best_per_category.json), so it is exempt — without this guard the
# "both false" state is indistinguishable from --benchmark-only --best-cat-only.
if [ "$RUN_BENCHMARK" = false ] && [ "$RUN_BEST_CAT" = false ] \
   && [ "$ABLATION_ONLY" != true ]; then
    echo "ERROR: --benchmark-only and --best-cat-only cannot be used together."
    exit 1
fi
# --ablation-only sets SKIP_GENERATE itself: named as the flag given.
if [ "$SKIP_GENERATE" = true ] && [ -n "$AFTER_JOB" ]; then
    if [ "$ABLATION_ONLY" = true ]; then
        echo "ERROR: --ablation-only and --after are mutually exclusive."
        echo "  --ablation-only retrains the ablation of the existing combos (no dependency)."
    else
        echo "ERROR: --skip-generate and --after are mutually exclusive."
        echo "  --skip-generate assumes embeddings already exist (no dependency)."
    fi
    echo "  --after chains training after an existing job (implies generation is running)."
    exit 1
fi
# Generation is DESTRUCTIVE and NOT reproducible: run_generate_embeddings.sh
# deletes every ALL_EMBEDDINGS .pt (and the *_gene_embeddings.pkl caches)
# before rebuilding, and generate_embeddings.py builds node2vec_ppi / go2vec /
# onto2vec with workers=4, which its own comments call not bit-for-bit
# reproducible. Every fold checkpoint and results.json now RECORD the training
# files' sha256 (train.py embeddings_sha256), and eval_finetuning.py /
# predict.py refuse a regenerated file at the same path — but nothing verifies
# it at submission time, and embedding_versions.py checks the generator
# version and gene order, not the vectors. So combos this submission does NOT
# retrain would be left naming features that no longer exist on disk, and a
# later --ablation-only or --eval-only would mix the two past every architecture,
# modality, fold and run-token check except the eval's content check (a
# confounded tab:loco delta: the ablation itself has no such check, and only
# paper_tables lists it afterwards, from the two results.json
# embeddings_sha256 records).
# The root orchestrator refuses the same shape (--force-generate with
# --skip-siamese). Only --benchmark-only reaches this state here: --ablation-only
# sets SKIP_GENERATE, --eval-only exits above, and every other mode retrains the
# listed variants' combos of the listed arms and removes EVERY variant's markers
# in EVERY arm's root below (a subset leaves the others' combos untrained on the
# new vectors). Every arm's root is looked at, whatever PCA_ARMS says: each
# one's combos were trained on the files this run would replace.
# _ucl -> "USE_CELL_LINES=<as this run resolved it> ", unless the calling
# shell's is on the line already (in _ENV_NOPCA / _ENV_SAME_NOPCA); _vset LIST
# -> that, then SIAMESE_VARIANTS_LIST="LIST"; _arms_pin LIST ->
# 'PCA_ARMS="LIST" ' unless the recorded one is (_ENV_SAME_NOPCA). Every
# submit_pipeline.sh command this script prints, and the tab:loco hint and the
# ablation_summary rebuild, pin the arms, the PCA targets, USE_CELL_LINES and
# (but the --benchmark-only remedy, which trains the selection variant alone)
# the variants as this run resolved them, as the end block's _paste_env does;
# a refused export's sbatch line pins the variants and USE_CELL_LINES (the
# export takes its root from PCA_ARM); --eval-only's export names its combo
# and the venv rebuild pins none (reset_env.sh reads none of them). Pasted into
# a shell exporting USE_CELL_LINES=0 (or another variant list, or PCA_ARMS), a
# line trained, ablated, exported or tabled another model, variants or arms
# than the refused run's (config.conf collapses the variants to pd:per_head
# under USE_CELL_LINES=0), and an --ablation-only one removed the arm's
# ablation dirs and queued tasks that could only stop at their signature check
# (2026-10-22). Any other setting the pasting shell exports (SEED, ARCH_ARGS,
# PCA_METHOD, PAIR_EXPORT_*, ...) still reaches its jobs: a printed
# orchestrator command names each in a NOTE when it runs, a printed sbatch
# command names none.
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
if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ] && [ "$RUN_BEST_CAT" = false ]; then
    _COMBO_OUT=$(for _a in none joint per_modality both; do
                     _r=$(pca_arm_results "$_a")
                     ls -d "$_r"/*latest_combo_cv[123].txt "$_r"/best_per_category.json 2>/dev/null
                 done || true)
    if [ -n "$_COMBO_OUT" ]; then
        echo "ERROR: this submission would regenerate the embeddings but retrains no combo,"
        echo "  and an earlier run's combo outputs are still on disk (every PCA arm's root):"
        printf '%s\n' "$_COMBO_OUT" | sed 's/^/    /'
        echo "  Regenerating replaces the .pt files those combos — and their ablations,"
        echo "  external-screen evals and paper_tables rows — were trained on, with"
        echo "  different vectors. A later --ablation-only or --eval-only would then mix"
        echo "  the two past every architecture, modality, fold and run-token check"
        echo "  (only the eval's own content check would catch it)."
        # (the same submission again, its settings from the calling shell and
        # its targets in full first, as the --ablation-only refusal prints it:
        # pasted bare, it benchmarked config.conf's arms at their defaults)
        echo "  Either keep the embeddings:"
        echo "    ${_ENV_SAME_NOPCA}$(_arms_pin "${PCA_ARM_LIST[*]}")$(pca_targets_env) $(_ucl)bash slurm/submit_pipeline.sh$_ARGS_Q --skip-generate"
        echo "  or retrain what depends on them (drop --benchmark-only), or delete those"
        echo "  files to regenerate on purpose. Nothing was changed or submitted."
        exit 1
    fi
    unset _COMBO_OUT _a _r
fi
# Checked here, before Step 0, so a missing selection file stops the run before
# an explicitly requested env reset is queued: every arm of PCA_ARMS ablates
# its own combos, from its own selection, against the combos its
# <prefix>latest_combo_<cv>.txt markers name. A task whose marker is missing
# SKIPs (exit 0), after the removal below has deleted that arm's previous
# ablation dirs: an arm with no marker at all (a regeneration removed them, say)
# would lose its whole tab:loco to tasks that all end COMPLETED, so it is
# refused; a marker missing for some (variant, CV) is named.
if [ "$ABLATION_ONLY" = true ] && [ "$RUN_ABLATION" = true ]; then
    # A selection's run token (best_per_category.json _selection_meta), for the
    # PCA refusal's rebuild below and the token block after it (which unsets it).
    _SEL_PY='import json,sys; print((json.load(open(sys.argv[1])).get("_selection_meta") or {}).get("run_token") or "")'
    # _retrain_variants ARM... -> the variants a printed --best-cat-only
    # retrain of these arms sets (SIAMESE_VARIANTS_LIST, in config.conf's
    # order): this submission's, and every other one with a combo marker in
    # an arm's root. The retrain replaces the arm's selection and its token,
    # and removes the markers and ablation dirs of the variants it lists only:
    # with a subset (one in the calling shell), the other variants' combos and
    # ablations were left under a selection of another run (2026-10-21).
    _retrain_variants() {
        local v a cv out=""
        for v in pd:per_head pd:pooled bilinear:per_head bilinear:pooled; do
            case " ${SIAMESE_VARIANTS[*]} " in
                *" $v "*) out="${out:+$out }$v"; continue ;;
            esac
            for a in "$@"; do
                for cv in cv1 cv2 cv3; do
                    if [ -f "$(pca_arm_results "$a")/$(siamese_variant_prefix "$v")latest_combo_${cv}.txt" ]; then
                        out="${out:+$out }$v"; continue 3
                    fi
                done
            done
        done
        echo "$out"
    }
    # _retrain_set ARM... -> the line's variants (_vset: after USE_CELL_LINES
    # as this run resolved it). With USE_CELL_LINES other than 1 the line
    # trains the single-output model alone, as pd:per_head (config.conf
    # collapses the list), and _retrain_left names the variants it leaves:
    # the cell-line models of the arm, under the selection it replaces.
    _retrain_set() {
        local vs
        vs=$(_retrain_variants "$@")
        [ "$USE_CELL_LINES" = 1 ] || vs=pd:per_head
        _vset "$vs"
    }
    _retrain_left() {
        local v out=""
        [ "$USE_CELL_LINES" != 1 ] || return 0
        for v in $(_retrain_variants "$@"); do
            [ "$v" = pd:per_head ] || out="${out:+$out }$v"
        done
        echo "$out"
    }
    _MISSING=""; _NO_MARKER=""; _HAVE=""
    for _a in "${PCA_ARM_LIST[@]}"; do
        _r=$(pca_arm_results "$_a")
        printf -v "_ABSENT_$_a" '%s' ""
        if [ ! -f "$_r/best_per_category.json" ]; then
            _MISSING="${_MISSING:+$_MISSING }$_a"
            continue
        fi
        _n=0; _absent=""
        for _v in "${SIAMESE_VARIANTS[@]}"; do
            for _cv in "${CV_TYPES[@]}"; do
                if [ -f "$_r/$(siamese_variant_prefix "$_v")latest_combo_${_cv}.txt" ]; then
                    _n=$((_n + 1))
                else
                    _absent="${_absent:+$_absent, }$_v $_cv"
                fi
            done
        done
        if [ "$_n" = 0 ]; then
            _NO_MARKER="${_NO_MARKER:+$_NO_MARKER }$_a"
        else
            _HAVE="${_HAVE:+$_HAVE }$_a"
            printf -v "_ABSENT_$_a" '%s' "$_absent"
        fi
    done
    if [ -n "$_MISSING" ] || [ -n "$_NO_MARKER" ]; then
        if [ -n "$_MISSING" ]; then
            echo "ERROR: --ablation-only needs the selection of every PCA arm it ablates; missing:"
            for _a in $_MISSING; do
                echo "    $(pca_arm_results "$_a")/best_per_category.json  (PCA arm $_a)"
            done
        fi
        if [ -n "$_NO_MARKER" ]; then
            echo "ERROR: --ablation-only needs a combo to ablate in every PCA arm it ablates; no"
            echo "  <prefix>latest_combo_<cv>.txt (variants ${SIAMESE_VARIANTS[*]}; ${CV_TYPES[*]}) in:"
            for _a in $_NO_MARKER; do
                echo "    $(pca_arm_results "$_a")/  (PCA arm $_a)"
            done
        fi
        # the arms to (re)train first, in PCA_ARMS order. Each command alone
        # on its line and complete (2026-10-21), as the PCA check's below:
        # the settings taken from the calling shell, this shell's targets in
        # full, its own variants (_retrain_variants) and arms; the others'
        # ablation is this submission again. The retrain keeps the venv
        # unless this submission asked for a reset (--reset-env): the arms it
        # leaves were trained in that venv.
        _RETRAIN=""
        for _a in "${PCA_ARM_LIST[@]}"; do
            case " $_MISSING $_NO_MARKER " in
                *" $_a "*) _RETRAIN="${_RETRAIN:+$_RETRAIN }$_a" ;;
            esac
        done
        # the pair partner too (pca_pair_closed): retrained alone, joint (or
        # both) would rank Step-1 runs of its own, not its pair's
        _rq="$_RETRAIN"; _RETRAIN=$(pca_pair_closed "$PCA_ARMS" $_RETRAIN)
        _hv=""
        for _a in $_HAVE; do
            case " $_RETRAIN " in *" $_a "*) ;; *) _hv="${_hv:+$_hv }$_a" ;; esac
        done
        _HAVE="$_hv"
        _rs=" --skip-reset-env"; [ "$RESET_ENV_EXPLICIT" != true ] || _rs=""
        if [ "$USE_CELL_LINES" = 1 ]; then
            echo "  Train them first (their selection, and the combos of the variants of this run and"
            echo "  of every other variant they hold a combo of; the embeddings kept):"
        else
            echo "  Train them first (their selection, and the single-output model's combos, as"
            echo "  pd:per_head; the embeddings kept):"
        fi
        [ "$_rq" = "$_RETRAIN" ] || echo "  (PCA arm(s) $_RETRAIN: a pair shares its single-embedding runs, so both arms of it retrain)"
        echo "    ${_ENV_NOPCA}$(pca_targets_env) $(_retrain_set $_RETRAIN) PCA_ARMS=\"$_RETRAIN\" bash slurm/submit_pipeline.sh --best-cat-only --skip-generate$_rs"
        _left=$(_retrain_left $_RETRAIN)
        if [ -n "$_left" ]; then
            echo "  The combos of $_left they hold (cell-line models) would stay, under a"
            echo "  selection they were not trained with: remove their markers (<arm root>/"
            echo "  var_*__latest_combo_<cv>.txt) before that line, or run this --ablation-only at"
            echo "  USE_CELL_LINES=1 instead (refused, it prints what to submit instead;"
            echo "  accepted, it ablates those combos as they are)."
        fi
        if [ -n "$_HAVE" ]; then
            echo "  or ablate the others alone:"
            echo "    ${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") PCA_ARMS=\"$_HAVE\" bash slurm/submit_pipeline.sh$_ARGS_Q"
        fi
        echo "  Nothing was changed or submitted."
        exit 1
    fi
    for _a in $_HAVE; do
        _v="_ABSENT_$_a"
        [ -n "${!_v}" ] || continue
        echo "NOTE: PCA arm $_a has no combo marker ($(pca_arm_results "$_a")/<prefix>latest_combo_<cv>.txt)"
        echo "      for: ${!_v}. Those ablation tasks will SKIP (no combo to ablate)."
    done
    # Every task of an arm trains with the PCA this shell gives the arm
    # (pca_multi_args of PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE /
    # PCA_JOINT_DIM as they are now) and stops at its signature check
    # (run_ablation.sh) on any other than its reference combo's. A target
    # exported for the combos' submission and not for this one (or a stray
    # one in this shell) would remove the arm's previous ablation below (72
    # runs, about 113 GB) and queue tasks that can only fail. So the PCA of
    # every marker-named combo (its config.json) is compared here first, on
    # the keys each task compares (config.conf pca_check_combos), and a
    # mismatch is refused with the submissions that reproduce the recorded
    # PCA (2026-10-21). Not _selection_meta.pca_args as text: 0.80 and 0.8
    # are one PCA, a variant-subset run rewrites it for some markers' combos
    # only, and a selection from before the arms records none.
    _CHK=(); _NREF=0
    for _a in $_HAVE; do
        _r=$(pca_arm_results "$_a")
        _CHK+=(--arm "$_a" "$(pca_multi_args "$_a")")
        for _v in "${SIAMESE_VARIANTS[@]}"; do
            for _cv in "${CV_TYPES[@]}"; do
                _m="$_r/$(siamese_variant_prefix "$_v")latest_combo_${_cv}.txt"
                [ -f "$_m" ] || continue
                _d=$(head -n 1 "$_m" | tr -d '\r')
                if [ -n "$_d" ] && [ -f "$_d/config.json" ]; then
                    _CHK+=("$_v=$_d/config.json")
                    _NREF=$((_NREF + 1))
                fi
            done
        done
    done
    # The selection split too (2026-11-13): selection_frac / selection_seed
    # are arch_signature keys, so a shell without the run's SELECTION_FRAC
    # would remove the previous ablation below and queue tasks that all stop
    # at run_ablation.sh's signature check. Refused here, before anything is
    # removed, against each marker-named combo's config.json.
    _WANT_FRAC=$(awk -v f="${SELECTION_FRAC:-0}" 'BEGIN { printf "%.6g", (f == "" ? 0 : f + 0) }')
    _WANT_SEED="${SELECTION_SEED:-42}"
    _SELBAD=""
    for _e in "${_CHK[@]}"; do
        case "$_e" in *=*/config.json) ;; *) continue ;; esac
        _cfg="${_e#*=}"
        _got=$($PYTHON_PATH -c 'import json,sys
c=json.load(open(sys.argv[1]))
f=float(c.get("selection_frac") or 0.0)
print("%.6g %s" % (f, int(c.get("selection_seed", 42)) if f > 0 else "-"))' "$_cfg" 2>/dev/null) || continue
        _gf="${_got% *}"; _gs="${_got#* }"
        if [ "$_gf" != "$_WANT_FRAC" ] || { [ "$_gs" != "-" ] && [ "$_gs" != "$_WANT_SEED" ]; }; then
            _SELBAD="${_SELBAD}    $_cfg  (selection_frac $_gf, seed $_gs)"$'\n'
        fi
    done
    if [ -n "$_SELBAD" ]; then
        echo "ERROR: --ablation-only would train the ablations with this shell's selection split"
        echo "    SELECTION_FRAC=${SELECTION_FRAC:-(empty)} SELECTION_SEED=$_WANT_SEED,"
        echo "  but combos it ablates were trained with another (every task would stop at its"
        echo "  signature check after the arm's previous ablation had been removed):"
        printf '%s' "$_SELBAD"
        echo "  Resubmit with the run's SELECTION_FRAC / SELECTION_SEED / SELECTION_MODE exported."
        exit 1
    fi
    if [ "$_NREF" -gt 0 ]; then
        if ! _OUT=$(pca_check_combos "${_CHK[@]}"); then
            echo "WARNING: could not compare each PCA arm's PCA with its combos' config.json on"
            echo "  this node (arch_signature.py --check_pca): each ablation task still checks"
            echo "  its own reference combo, and stops on another PCA."
            _OUT=""
        fi
        _BAD=""
        while IFS='|' read -r _k _x _s _rest; do
            [ "$_k" = arm ] && [ "$_s" != ok ] && _BAD="${_BAD:+$_BAD }$_x"
        done <<< "$_OUT"
        if [ -n "$_BAD" ]; then
            echo "ERROR: --ablation-only would train the ablations with this shell's PCA targets,"
            echo "    $(pca_targets_env)"
            echo "  but combos it ablates were trained with another PCA: every task of theirs would"
            echo "  stop at its signature check, after the arm's previous ablation had been removed."
            for _a in $_BAD; do
                echo "  PCA arm $_a ($(pca_multi_args "$_a") here):"
                while IFS='|' read -r _k _x _y _d _s _l _rest; do
                    [ "$_k" = combo ] && [ "$_x" = "$_a" ] && [ "$_s" = bad ] \
                        && echo "    $_d  ($_y: $_l)"
                done <<< "$_OUT"
            done
            # What to submit instead (2026-10-21). Every line printed below
            # is a command complete on its own and alone on its line, to run
            # as printed in bash or zsh from a shell that never held this
            # one's one-shot settings: the settings taken from the calling
            # shell first (recorded at the top), then the three targets in
            # full, then the line's own variants and arms (the last assignment
            # wins, in bash and zsh alike), the prose above the commands and
            # without an apostrophe. Until then each line carried only the
            # targets that differ and nothing of a one-shot prefix (pasted, it
            # was refused again, the other target printed, endlessly, or
            # accepted at seed 42 and every task failed after the ablation was
            # removed), one ended in "   (this shell's PCA)" (a syntax error
            # in bash, an open quote in zsh), and one line per arm was printed
            # even when one submission fits them all, each later line then
            # refused by the live-job check (job names carry no arm).
            # One set of targets that trains the combos of every arm
            # (check_pca's `all` line, which the root's --skip-siamese reads
            # too; always so for a forgotten or stray target): ONE command,
            # the same submission again with them.
            IFS='|' read -r _ST _EX <<< "$(printf '%s\n' "$_OUT" | sed -n 's/^all|//p')"
            if [ "$_ST" = fix ]; then
                echo "  One set of targets trains the combos of every arm. Submit the same again with it:"
                echo "    ${_ENV_SAME_NOPCA}$(_arms_pin "${PCA_ARM_LIST[*]}")$(pca_targets_with "$_EX") $(_vset "${SIAMESE_VARIANTS[*]}") bash slurm/submit_pipeline.sh$_ARGS_Q"
                echo "  Nothing was changed or submitted."
                exit 1
            fi
            # No one set does: one submission per arm, or per group of its
            # variants where those of an arm were trained at different
            # targets, and one for the arms that match this shell. An arm with
            # a combo no targets train in it (another arm's PCA, or CVs that
            # disagree) is retrained whole, which ablates it too: a retrain of
            # those variants alone (the line printed until 2026-10-21) replaced
            # the arm's selection, and its token, under the combos and the
            # ablation of its other variants. Whole is every variant with a
            # combo marker in the arm and this submission's (_retrain_variants:
            # a variant subset in the calling shell retrained that subset only),
            # in the venv the other arms were trained in (--skip-reset-env,
            # unless this submission asked for a reset). The retrain goes first
            # (it is refused while jobs that read or write the outputs it
            # removes are live); each later line is refused while an earlier
            # one's jobs are live, and
            # submitted alongside them with ALLOW_LIVE_JOBS=1 once the reader
            # has seen that only those are (never printed on a line: pasted as
            # a block, such lines would pass a real conflict). --reset-env
            # stays on the first line only: a later reset would delete the venv
            # under the jobs of the earlier lines.
            _RT=""; _SPLIT=""; _OKA=""
            for _a in $_BAD; do
                _n=0; _other=false
                while IFS='|' read -r _k _x _s _e _keys; do
                    [ "$_k" = group ] && [ "$_x" = "$_a" ] || continue
                    _n=$((_n + 1))
                    [ "$_s" = ok ] || [ "$_s" = fix ] || _other=true
                done <<< "$_OUT"
                if [ "$_other" = true ]; then
                    _RT="${_RT:+$_RT }$_a"
                elif [ "$_n" -gt 1 ]; then
                    _SPLIT="${_SPLIT:+$_SPLIT }$_a"
                fi
            done
            # the pair partner of an arm retrained whole retrains too
            # (pca_pair_closed), and is then left out of the lines below
            _rt0="$_RT"; [ -z "$_RT" ] || _RT=$(pca_pair_closed "$PCA_ARMS" $_RT)
            for _a in $_HAVE; do
                case " $_BAD $_RT " in *" $_a "*) ;; *) _OKA="${_OKA:+$_OKA }$_a" ;; esac
            done
            _CMDS=(); _args="$_ARGS_Q"
            if [ -n "$_RT" ]; then
                _rs=" --skip-reset-env"; [ "$RESET_ENV_EXPLICIT" != true ] || _rs=""
                _CMDS+=("${_ENV_NOPCA}$(pca_targets_env) $(_retrain_set $_RT) PCA_ARMS=\"$_RT\" bash slurm/submit_pipeline.sh --best-cat-only --skip-generate$_rs")
                _args="$_ARGS_Q_NORESET"
            fi
            for _a in $_BAD; do
                case " $_RT " in *" $_a "*) continue ;; esac
                while IFS='|' read -r _k _x _s _e _keys; do
                    [ "$_k" = group ] && [ "$_x" = "$_a" ] || continue
                    _vl="$(_vset "${SIAMESE_VARIANTS[*]}") "
                    case " $_SPLIT " in *" $_a "*) _vl="$(_vset "$_keys") " ;; esac
                    if [ "$_s" = fix ]; then _t=$(pca_targets_with "$_e"); else _t=$(pca_targets_env); fi
                    _CMDS+=("${_ENV_NOPCA}$_t ${_vl}PCA_ARMS=\"$_a\" bash slurm/submit_pipeline.sh$_args")
                    _args="$_ARGS_Q_NORESET"
                done <<< "$_OUT"
            done
            if [ -n "$_OKA" ]; then
                _CMDS+=("${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") PCA_ARMS=\"$_OKA\" bash slurm/submit_pipeline.sh$_args")
            fi
            # (the per-arm lines, when there is one besides the retrain)
            _nrt=0; [ -z "$_RT" ] || _nrt=1
            if [ "${#_CMDS[@]}" -gt "$_nrt" ]; then
                echo "  No one set of targets trains the combos of every arm. Submit these instead, each"
                echo "  with the targets its combos were trained with (those of this shell where they"
                echo "  match): one per arm or, where the variants of an arm were trained at different"
                echo "  targets, one per group of them."
            fi
            if [ -n "$_RT" ]; then
                echo "  No targets train some combos of PCA arm(s) $_rt0 (the PCA of another arm, or CVs"
                if [ "$USE_CELL_LINES" = 1 ]; then
                    echo "  that disagree): the --best-cat-only line retrains the arm(s) whole (the variants"
                    echo "  of this run and every other variant they hold a combo of), at the targets of this"
                    echo "  shell, and ablates them (a retrain of some variants alone would replace the"
                    echo "  selection under the combos and the ablation of the other variants)."
                else
                    echo "  that disagree): the --best-cat-only line retrains the arm(s)' selection and the"
                    echo "  single-output model (USE_CELL_LINES=$USE_CELL_LINES, as pd:per_head), at the targets"
                    echo "  of this shell, and ablates it."
                fi
                [ "$_rt0" = "$_RT" ] || echo "  (PCA arm(s) $_RT: a pair shares its single-embedding runs, so both arms of it retrain)"
                _left=$(_retrain_left $_RT)
                if [ -n "$_left" ]; then
                    echo "  The combos of $_left they hold (cell-line models) would stay, under a"
                    echo "  selection they were not trained with: remove their markers (<arm root>/"
                    echo "  var_*__latest_combo_<cv>.txt) before that line, or run this --ablation-only at"
                    echo "  USE_CELL_LINES=1 instead (refused, it prints what to submit instead;"
                    echo "  accepted, it ablates those combos as they are)."
                fi
            fi
            if [ "${#_CMDS[@]}" -gt 1 ]; then
                echo "  Submit them one after the other, in this order. Job names carry no PCA arm, so a"
                echo "  line is refused while jobs an earlier line queued are queued or running. Once squeue"
                if [ "$RESET_ENV_EXPLICIT" = true ]; then
                    echo "  lists only those, and the reset_env job of the first line has finished,"
                else
                    echo "  lists only those,"
                fi
                echo "  ALLOW_LIVE_JOBS=1 in front of the next line submits it alongside them: an"
                echo "  --ablation-only line removes and writes only the ablation dirs of its own arm or"
                echo "  variants."
            fi
            for _c in "${_CMDS[@]}"; do
                echo "    $_c"
            done
            # Each line of a split arm rewrites its ablation_summary.csv over
            # its own variants (the summary job of a submission covers that
            # submission's SIAMESE_VARIANTS): the job that writes it over all
            # of them, under the token every line joins (the arm's selection
            # token, read as the token block below reads it), for once every
            # line has finished. No table reads that file (tab:loco reads the
            # ablation dirs).
            for _a in $_SPLIT; do
                _r=$(pca_arm_results "$_a")
                _allk=""
                while IFS='|' read -r _k _x _s _e _keys; do
                    [ "$_k" = group ] && [ "$_x" = "$_a" ] || continue
                    _allk="$_allk $_keys"
                done <<< "$_OUT"
                _vs=""
                for _v in "${SIAMESE_VARIANTS[@]}"; do
                    case " $_allk " in *" $_v "*) _vs="${_vs:+$_vs }$_v" ;; esac
                done
                _stok=$("$PYTHON_PATH" -c "$_SEL_PY" "$_r/best_per_category.json" 2>/dev/null \
                        || python3 -c "$_SEL_PY" "$_r/best_per_category.json" 2>/dev/null || true)
                echo "  PCA arm $_a takes one line per group of its variants, and each rewrites"
                echo "  $_r/ablation_summary.csv over its own: the file then holds the variants of the"
                echo "  last line only (no table reads it: tab:loco reads the ablation dirs)."
                if [ -n "$_stok" ]; then
                    printf -v _q '%q' "$_stok"
                    echo "  Once the jobs of every line have finished, this writes it for all of them, from"
                    echo "  siamese_sl/:"
                    # (logged as the arm's own summary jobs are: _logs)
                    _lg=$(_logs "$_a" ablation_summary %j)
                    echo "    ${_ENV_NOPCA}$(pca_targets_env) RUN_TOKEN=$_q PCA_ARMS=\"$_a\" PCA_ARM=$_a $(_vset "$_vs") sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --export=ALL${_lg:+ $_lg} slurm/run_ablation_summary.sh"
                else
                    echo "  Its selection records no run token, so each line keeps a fresh one of its own,"
                    echo "  and no one rebuild of the file gives the rows of every line a delta."
                fi
            done
            echo "  Nothing was changed or submitted."
            exit 1
        fi
    fi
    unset _MISSING _NO_MARKER _HAVE _RETRAIN _a _r _n _v _cv _absent _CHK _NREF _m _d \
          _OUT _BAD _OKA _k _x _y _s _l _e _keys _allk _vl _rest _ST _EX _RT _SPLIT \
          _CMDS _args _c _t _other _vs _stok _q _nrt _rs _lg _left _rt0 _rq _hv
fi
# --ablation-only ablates the combos an earlier run trained, so it joins that
# run's token: paper_tables.py checks ablation dirs against --ablation_run_token
# (CLI only; an exported $ABLATION_RUN_TOKEN is ignored), else RUN_TOKEN, else
# best_per_category.json's _selection_meta.run_token — and the root
# --skip-siamese rebuild sets RUN_TOKEN to that same selection token (else the
# pcavar* dirs'), so a fresh token here would leave tab:loco stale unless
# run_paper_tables.sh is passed --ablation_run_token by hand (what the WARNING
# below asks for when the selection file records no token). Per arm: each arm's
# selection names its own; when they differ (arms trained by different
# submissions) each arm's jobs get theirs (_as_arm), and RUN_TOKEN, which the
# rest of the submission carries, is the first arm's.
# The previous ablation dirs are removed below, before any sbatch, so a task
# that never starts cannot pass as this submission's under the shared token.
_TOKEN_NOTE=""; _PER_ARM_TOKENS=false
if [ "$ABLATION_ONLY" = true ] && [ "$RUN_ABLATION" = true ]; then
    # (_SEL_PY: set with the selection check above)
    _FRESH="$RUN_TOKEN"; _TOKS=""
    for _a in "${PCA_ARM_LIST[@]}"; do
        _sel="$(pca_arm_results "$_a")/best_per_category.json"
        _tok=$("$PYTHON_PATH" -c "$_SEL_PY" "$_sel" 2>/dev/null || python3 -c "$_SEL_PY" "$_sel" 2>/dev/null || true)
        if [ -z "$_tok" ]; then
            _tok="$_FRESH"
            printf -v "_ABL_TOK_$_a" '%s' "$_tok"   # (the tables rebuild at the end)
            echo "WARNING: $_sel records no _selection_meta.run_token (PCA arm $_a);"
            echo "  its ablation keeps a fresh token ($_tok). For tab:loco pass it on the"
            echo "  command line (\$ABLATION_RUN_TOKEN is ignored), from the repo root:"
            # (with the settings taken from the calling shell, the targets in
            # full and the run's arms, which the tables job reads: pasted bare
            # in a fresh shell, the confs' defaults were graded instead)
            echo "    env -u RUN_TOKEN ${_ENV_NOPCA}$(pca_targets_env) $(_vset "${SIAMESE_VARIANTS[*]}") PCA_ARMS=\"$PCA_ARMS\" PCA_ARM=$_a sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --export=ALL sl_comparison/slurm/run_paper_tables.sh --ablation_run_token $_tok"
            echo "  Do not set RUN_TOKEN to it: with no selection token that would also check"
            echo "  tab:solo against it. The root --skip-siamese passes no ablation token and"
            echo "  flags tab:loco as another run's."
        fi
        # Handed to the arm's jobs as RUN_TOKEN=<token> in sbatch's
        # environment when the arms' tokens differ (_as_arm), and printed,
        # unquoted, in the commands this script prints to paste (above,
        # _arm_env): one plain shell word, as every token the orchestrators
        # mint is.
        case "$_tok" in
            *[!A-Za-z0-9._:+-]*)
                echo "ERROR: the run token '$_tok' of PCA arm $_a ($_sel) is not one plain"
                echo "  shell word ([A-Za-z0-9._:+-] only), as the commands printed to paste need"
                echo "  it. Nothing was changed or submitted."
                exit 1 ;;
        esac
        printf -v "_ARM_TOKEN_$_a" '%s' "$_tok"
        case " $_TOKS " in *" $_tok "*) ;; *) _TOKS="${_TOKS:+$_TOKS }$_tok" ;; esac
    done
    _v="_ARM_TOKEN_${PCA_ARM_LIST[0]}"
    RUN_TOKEN="${!_v}"
    export RUN_TOKEN
    if [ "$_TOKS" = "$_FRESH" ]; then
        _TOKEN_NOTE=""
    elif [ "${_TOKS// /}" = "$_TOKS" ]; then
        _TOKEN_NOTE=" (the ablated run's: best_per_category.json _selection_meta.run_token)"
    else
        _PER_ARM_TOKENS=true
        _TOKEN_NOTE=" (PCA arm ${PCA_ARM_LIST[0]}'s; each arm's jobs get their own, below)"
        echo "WARNING: the PCA arms to ablate were selected by runs of different tokens:"
        for _a in "${PCA_ARM_LIST[@]}"; do
            _v="_ARM_TOKEN_$_a"
            echo "    $_a: ${!_v}"
        done
        echo "  Each arm's ablation joins its own (PCA_ARM=<arm> RUN_TOKEN=<token> sbatch"
        echo "  ... --export=ALL); the table across the arms (sl_comparison/results/"
        echo "  pca_arms/) flags the arms of another submission as stale."
    fi
    # One token for every arm: no job needs one of its own (_as_arm passes
    # none; every job inherits RUN_TOKEN).
    if [ "$_PER_ARM_TOKENS" = false ]; then
        for _a in none joint per_modality both; do printf -v "_ARM_TOKEN_$_a" '%s' ""; done
    fi
    unset _SEL_PY _FRESH _TOKS _a _sel _tok _v
fi
# The single-embedding runs this submission rebuilds (each owner's Step 1) may
# be the ones the owner's pair partner, an arm PCA_ARMS leaves out, ranked in
# the selection it keeps (_selection_meta.step1_root names that root): that
# arm's tables would then find this submission's runs under its selection (its
# tab:solo flags them as another run's) until the arm is rerun. Said before
# anything changes; the arm's own outputs are not touched.
if [ "$RUN_BEST_CAT" = true ]; then
    _S1_PY='import json,sys; print((json.load(open(sys.argv[1])).get("_selection_meta") or {}).get("step1_root") or "")'
    for _o in $_OWNERS; do
        # (the whole pair: config.conf's pca_step1_arms lists this run's arms only)
        case "$_o" in
            per_modality|both) _pair="per_modality both" ;;
            *)                 _pair="none joint" ;;
        esac
        for _p in $_pair; do
            case " $PCA_ARMS " in *" $_p "*) continue ;; esac
            _sel="$(pca_arm_results "$_p")/best_per_category.json"
            [ -f "$_sel" ] || continue
            _s1=$("$PYTHON_PATH" -c "$_S1_PY" "$_sel" 2>/dev/null \
                  || python3 -c "$_S1_PY" "$_sel" 2>/dev/null || true)
            [ "${_s1%/}" = "$(pca_arm_results "$_o")" ] || continue
            echo "WARNING: PCA arm $_p, which PCA_ARMS leaves out, ranked the single-embedding runs in"
            echo "  $(pca_arm_results "$_o")/ ($_sel, _selection_meta.step1_root), which this"
            echo "  submission rebuilds as PCA arm $_o's Step 1: until $_p is rerun (PCA_ARMS listing"
            echo "  it), its tab:solo will be another run's."
        done
    done
    unset _S1_PY _o _p _pair _sel _s1
fi


# ============================================================================
# Ensure log directory exists
# ============================================================================
mkdir -p "$WORK_DIR/slurm/logs"

# What the submission costs, per arm and per owner (from the confs, in
# subshells: their values must reach neither this shell nor the jobs yet), and
# the fold matrices it keeps: NUM_FOLDS x 313 MB per run (8,844 x 8,844 float32
# on the shared genes); the export 42 GB per arm by default.
_N_ARMS=${#PCA_ARM_LIST[@]}
_N_OWNERS=$(echo $_OWNERS | wc -w | tr -d ' ')
_gb() { echo $(( ($1 * NUM_FOLDS * 313 + 500) / 1000 )); }
_COMBOS=$(( ${#SIAMESE_VARIANTS[@]} * ${#CV_TYPES[@]} ))
_EVALS=$(( $(eval_cv_count) * ${#SIAMESE_VARIANTS[@]} ))
_LOCO=$( ( source "$SCRIPT_DIR/run_ablation.conf" >/dev/null 2>&1 \
           && echo $(( ${#ABLATE_CATEGORIES[@]} * ${#CV_TYPES[@]} * ${#SIAMESE_VARIANTS[@]} )) \
           "${ABLATION_PURGE_PREDICTIONS:-0}" ) 2>/dev/null ) || _LOCO=""
_LOCO_PURGE="${_LOCO#* }"; _LOCO="${_LOCO%% *}"
_STEP1=$( ( source "$SCRIPT_DIR/run_best_per_category.conf" >/dev/null 2>&1 \
            && echo "$(bestcat_n_tasks) $(( ${#EMB_CATALOG[@]} * ${#CV_TYPES[@]} ))" ) 2>/dev/null ) || _STEP1=""
_STEP1_KEPT="${_STEP1#* }"; _STEP1="${_STEP1%% *}"
_BENCH=$(( ${#ALL_EMBEDDINGS[@]} * ${#CV_TYPES[@]} ))
_ARM_COST=""; _OWNER_COST=""; _ARM_GB=0; _OWNER_GB=0; _DISK=""
if [ "$RUN_BEST_CAT" = true ]; then
    _ARM_COST="$_COMBOS combos (${#SIAMESE_VARIANTS[@]} variants x ${#CV_TYPES[@]} CVs)"
    _ARM_GB=$(( _ARM_GB + $(_gb "$_COMBOS") ))
    _DISK="$(_gb "$_COMBOS") GB of combos"
    if [ "$ALLMOD_COMBO" = 1 ]; then
        _ARM_COST="$_ARM_COST, $_COMBOS all-modalities combos (ALLMOD_COMBO=1)"
        _ARM_GB=$(( _ARM_GB + $(_gb "$_COMBOS") ))
        _DISK="$_DISK, $(_gb "$_COMBOS") GB of all-modalities combos"
    fi
    if [ "$RUN_EVAL" = true ]; then _ARM_COST="$_ARM_COST, $_EVALS eval tasks"; fi
    if [ "$RUN_PAIR" = true ]; then
        _ARM_GB=$(( _ARM_GB + 42 )); _DISK="$_DISK, 42 GB of pair export (PAIR_EXPORT_* defaults)"
    fi
fi
if [ "$RUN_ABLATION" = true ] && [ -n "$_LOCO" ]; then
    _ARM_COST="${_ARM_COST:+$_ARM_COST, }$_LOCO leave-one-category-out tasks"
    if [ "$_LOCO_PURGE" = 1 ]; then
        _DISK="${_DISK:+$_DISK, }no ablation fold matrices (ABLATION_PURGE_PREDICTIONS=1)"
    else
        _ARM_GB=$(( _ARM_GB + $(_gb "$_LOCO") ))
        _DISK="${_DISK:+$_DISK, }$(_gb "$_LOCO") GB of ablation fold matrices"
    fi
fi
if [ "$RUN_BENCHMARK" = true ]; then
    _OWNER_COST="$_BENCH benchmark tasks"; _OWNER_GB=$(_gb "$_BENCH")
fi
if [ "$RUN_BEST_CAT" = true ] && [ -n "$_STEP1" ]; then
    _OWNER_COST="${_OWNER_COST:+$_OWNER_COST, }$_STEP1 Step-1 tasks"
    _OWNER_GB=$(( _OWNER_GB + $(_gb "$_STEP1_KEPT") ))
fi
# The single-embedding figure is an upper bound (2026-10-21; it read "about"
# before): a benchmark or Step-1 task whose embedding file is absent SKIPs and
# keeps nothing, and the types whose absence generation accepts
# (run_generate_embeddings.conf OPTIONAL_TYPES) are often absent: esmc is not
# generated by default, scprint and seqvec were missing in the 2026-09 run.
_OPT=$( ( source "$SCRIPT_DIR/run_generate_embeddings.conf" >/dev/null 2>&1 \
          && echo "${OPTIONAL_TYPES[*]}" ) 2>/dev/null ) || _OPT=""
unset -f _gb
# What the submission queues, in jobs and in array tasks: Slurm's
# MaxSubmitJobs (sacctmgr's MaxSubmit / GrpSubmit, a QOS's MaxSubmitPU) counts
# every array task, and the four arms queue 586 by default (one arm queued 160
# before 2026-10-20): a limit below that stops the submission part-way. In the
# order and under the conditions of the submissions below.
_NJOBS=0; _NTASKS=0
_count() { _NJOBS=$((_NJOBS + $1)); _NTASKS=$((_NTASKS + $2)); }   # N_JOBS N_TASKS
if [ "$RESET_ENV" = true ]; then _count 1 1; fi
if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then _count 1 1; fi
if [ "$RUN_BENCHMARK" = true ]; then _count $((2 * _N_OWNERS)) $(( (_BENCH + 1) * _N_OWNERS )); fi
if [ "$RUN_BEST_CAT" = true ]; then
    _count "$_N_OWNERS" $(( ${_STEP1:-0} * _N_OWNERS ))
    _count "$_N_ARMS" "$_N_ARMS"                                        # combos
    if [ "$ALLMOD_COMBO" = 1 ]; then _count "$_N_ARMS" $(( ${#CV_TYPES[@]} * _N_ARMS )); fi
    if [ "$RUN_EVAL" = true ]; then _count "$_N_ARMS" $(( _EVALS * _N_ARMS )); fi
    if [ "$RUN_PAIR" = true ]; then _count "$_N_ARMS" "$_N_ARMS"; fi        # exports
fi
if [ "$RUN_ABLATION" = true ] && { [ "$ABLATION_ONLY" = true ] || [ "$RUN_BEST_CAT" = true ]; }; then
    _count $((2 * _N_ARMS)) $(( (${_LOCO:-0} + 1) * _N_ARMS ))           # + summaries
fi
unset -f _count

echo "============================================================================"
echo "  Siamese SL Pipeline — Job Submission"
echo "============================================================================"
echo "  Working dir:    $WORK_DIR"
echo "  Data dir:       $DATA_DIR"
echo "  Variants (${#SIAMESE_VARIANTS[@]}):   ${SIAMESE_VARIANTS[*]}"
echo "  Selection on:   $SELECTION_VARIANT"
echo "  Selection mode: $SELECTION_MODE; selection split: ${SELECTION_ARGS:-off (SELECTION_FRAC empty)}"
echo "  ARCH_ARGS:      ${ARCH_ARGS:-(none; train.py defaults)}"
echo "  PCA arms ($_N_ARMS):   ${PCA_ARM_LIST[*]} (each the whole pipeline, in its own results root)"
for _a in "${PCA_ARM_LIST[@]}"; do
    printf '    %-13s %-26s %s\n' "$_a" "$(pca_arm_results "$_a")/" "$(pca_arm_label "$_a")"
done
if [ "$RUN_BENCHMARK" = true ] || [ "$RUN_BEST_CAT" = true ]; then
    for _o in $_OWNERS; do
        printf '    %-13s %-26s %s\n' "(Step 1)" "$(pca_arm_results "$_o")/" "$(pca_step1_label "$_o")"
    done
fi
echo "  RUN_TOKEN:      $RUN_TOKEN$_TOKEN_NOTE"
if [ "$_PER_ARM_TOKENS" = true ]; then
    for _a in "${PCA_ARM_LIST[@]}"; do
        _v="_ARM_TOKEN_$_a"
        echo "    PCA arm $_a: RUN_TOKEN=${!_v}"
    done
fi
if [ -n "$_INHERITED_RUN_TOKEN" ] && [ "$_INHERITED_RUN_TOKEN" != "$RUN_TOKEN" ]; then
    echo "  NOTE: replaced RUN_TOKEN=$_INHERITED_RUN_TOKEN from the environment"
fi
[ -z "$_ARM_COST" ] || echo "  Per PCA arm:    $_ARM_COST"
[ -z "$_OWNER_COST" ] || echo "  Per owner ($_N_OWNERS):  $_OWNER_COST"
echo "  Queues:         $_NJOBS jobs, $_NTASKS tasks counting every array task: Slurm's"
echo "                  MaxSubmitJobs counts each (sacctmgr show assoc user=\$USER"
echo "                  format=MaxSubmit,GrpSubmit; the QOS's MaxSubmitPU)"
if [ "$_ARM_GB" -gt 0 ] || [ "$_OWNER_GB" -gt 0 ] || [ -n "$_DISK" ]; then
    # The figure is the fold matrices and the pair export the submission
    # keeps; what else it writes is named under it (2026-10-21: "up to 1062
    # GB" read as a bound on the whole run, which every default run goes over
    # by the eval's per-pair CSVs, eval_finetuning.py, about 5 GB per arm at
    # the default variants and screens; and the checkpoints: the combos',
    # Step 1's, the ablation's unless purged, the fine-tuned models).
    _KEPT="the fold matrices"
    if [ "$RUN_BEST_CAT" = true ] && [ "$RUN_PAIR" = true ]; then
        _KEPT="$_KEPT and pair export"
    fi
    if [ "$_OWNER_GB" -gt 0 ]; then
        echo "  Disk, up to:    $(( _N_ARMS * _ARM_GB + _N_OWNERS * _OWNER_GB )) GB in all, of $_KEPT it keeps"
    else
        echo "  Disk, about:    $(( _N_ARMS * _ARM_GB )) GB in all, of $_KEPT it keeps"
    fi
    [ -z "$_DISK" ] || echo "                  per arm: $_DISK"
    if [ "$_OWNER_GB" -gt 0 ]; then
        echo "                  per owner: up to $_OWNER_GB GB of single-embedding fold matrices (a task"
        echo "                  whose embedding is absent keeps none${_OPT:+; optional ones: $_OPT})"
    fi
    _UNCOUNTED="the checkpoints"
    if [ "$RUN_BEST_CAT" = true ] && [ "$RUN_EVAL" = true ]; then
        _UNCOUNTED="the eval's per-pair CSVs (about 5 GB per arm by default) and $_UNCOUNTED"
    fi
    echo "                  not counted: $_UNCOUNTED"
fi
echo ""
unset _a _o _v _OPT _KEPT _UNCOUNTED

# Before anything is removed or queued: every pipeline job for the submissions
# that reset the venv or rebuild the embeddings, else the jobs that read or
# write what the block below removes (see _check_no_live_jobs above).
if [ "$RESET_ENV" = true ] \
   || { [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; }; then
    _check_no_live_jobs
elif [ "$RUN_BEST_CAT" = true ]; then
    _check_no_live_jobs outputs
elif [ "$ABLATION_ONLY" = true ] && [ "$RUN_ABLATION" = true ]; then
    _check_no_live_jobs ablation
fi

# ============================================================================
# Previous-run outputs this submission rebuilds (before any sbatch)
# ============================================================================
# As the root orchestrator's step 0a does for its stages: a job that fails or
# never starts must leave NO output rather than an earlier run's. Per PCA arm
# of PCA_ARMS, in that arm's own roots (results[/pca_<arm>]/ here,
# sl_comparison/results[/pca_<arm>]/ for its rows, moved into that root's own
# stale_siamese/: every arm's rows have the same names); an arm this
# submission leaves out keeps its outputs, but for the combo markers a
# regeneration removes.
_ts=$(date +%Y%m%dT%H%M%S)
if [ "$RUN_BEST_CAT" = true ]; then
    _CMP_GONE=""
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _R=$(pca_arm_results "$_arm")
        _C=$(pca_arm_cmp "$_arm")
        # The combo replaces these variants' combos, and this script does not
        # rerun the post-hoc cell-line ablation. Its dirs and sl_comparison
        # *_nocell rows (no token; collect_and_regrade_siamese.sh leaves them)
        # describe the replaced combos, and paper_tables / compare.py would
        # pair them with the new ones in tab:cellline (CL1).
        for _v in "${SIAMESE_VARIANTS[@]}"; do
            _p=$(siamese_variant_prefix "$_v")
            _t=$(siamese_variant_tag "$_v")
            for _cv in "${CV_TYPES[@]}"; do
                if [ -d "$_R/${_p}cellline_ablation_${_cv}" ]; then
                    rm -rf -- "$_R/${_p}cellline_ablation_${_cv}"
                    echo "  removed previous cell-line ablation dir $_R/${_p}cellline_ablation_${_cv}"
                fi
                _f="$BASE_DIR/$_C/siamese_sl${_t:+_$_t}_nocell_${_cv}.json"
                if [ -f "$_f" ]; then
                    mkdir -p "$BASE_DIR/$_C/stale_siamese"
                    mv -f "$_f" "$BASE_DIR/$_C/stale_siamese/$(basename "${_f%.json}").$_ts.json"
                    echo "  moved previous cell-line ablation row aside: $_C/$(basename "$_f")"
                fi
                # ...and this variant's comparison row and selected-run marker,
                # as the root orchestrator's step 0a does. They grade the combo
                # this submission replaces, and no job queued here rebuilds
                # them (run_compare.sh is the root's). The combo directory name
                # is fixed by the winners, so paper_tables' "row graded another
                # run" check still matches a retrained combo: left in place, the
                # previous combos' AUROC / AP / AUPRG would be published as
                # tab:tiers next to THIS run's parameter counts, PCA widths and
                # tab:loco reference.
                for _f in "$BASE_DIR/$_C/siamese_sl${_t:+_$_t}_${_cv}.json" \
                          "$BASE_DIR/$_C/siamese${_t:+_sl_$_t}_selected_${_cv}.txt"; do
                    [ -f "$_f" ] || continue
                    mkdir -p "$BASE_DIR/$_C/stale_siamese"
                    _b=$(basename "$_f")
                    mv -f "$_f" "$BASE_DIR/$_C/stale_siamese/${_b%.*}.$_ts.${_b##*.}"
                    echo "  moved previous comparison row aside: $_C/$_b"
                done
            done
            # The combo deletes its markers only once it starts (after
            # config.conf); a combo cancelled while pending would leave the
            # previous run's, which eval, collect_siamese --combo_markers,
            # external_transfer and paper_tables all take as this run's combos.
            rm -f "$_R/$_p"latest_combo_cv[123].txt
            # ALLMOD_COMBO=1: this run retrains the variant's all-modalities
            # models too (run_allmod_combo.sh). Their markers, rows and
            # selected-run markers of the previous run go now, as above: a
            # job that never starts must leave no row.
            if [ "$ALLMOD_COMBO" = 1 ]; then
                _ap=$(allmod_prefix "$_v")
                _ar=$(allmod_row "$_v")
                rm -f "$_R/$_ap"latest_allmod_cv[123].txt
                for _f in "$BASE_DIR/$_C/${_ar}"_cv[123].json \
                          "$BASE_DIR/$_C/${_ar}"_selected_cv[123].txt; do
                    [ -f "$_f" ] || continue
                    mkdir -p "$BASE_DIR/$_C/stale_siamese"
                    _b=$(basename "$_f")
                    mv -f "$_f" "$BASE_DIR/$_C/stale_siamese/${_b%.*}.$_ts.${_b##*.}"
                    echo "  moved previous all-modalities row aside: $_C/$_b"
                done
            fi
        done
        # Likewise the previous selection (paper_tables reads its winners and token).
        if [ -f "$_R/best_per_category.json" ]; then
            rm -f "$_R/best_per_category.json"
            echo "  removed previous modality selection $_R/best_per_category.json"
        fi
        # Built from BOTH sides, so stale the moment these combos are replaced,
        # and rebuilt by no job this script queues. paper_tables prefers
        # comparison.csv over the top-level row JSONs, so it goes with the rows
        # moved aside above: every SLeuth row is then reported missing instead
        # of being published with the previous combos' numbers.
        _n=0
        for _d in "$BASE_DIR/$_C/comparison.csv" "$BASE_DIR/$_C/comparison_notes.txt" \
                  "$BASE_DIR/$_C/external_transfer" "$BASE_DIR/$_C/paper_tables" \
                  "$BASE_DIR/$_C/bootstrap"; do
            [ -e "$_d" ] || continue
            rm -rf -- "$_d"
            _n=$((_n + 1))
        done
        if [ "$_n" -gt 0 ]; then
            echo "  removed $_n previous sl_comparison output(s) of PCA arm $_arm (comparison"
            echo "  table, external-transfer and paper-tables dirs under $_C/)"
            _CMP_GONE="${_CMP_GONE:+$_CMP_GONE }$_arm"
        fi
    done
    # A subset of the variants (SIAMESE_VARIANTS_LIST, or USE_CELL_LINES=0) or
    # of the PCA arms (PCA_ARMS) retrains only the listed combos, but the
    # generation job queued below rebuilds the embedding files under EVERY
    # variant's combo of EVERY arm (see the guard above). Their markers go too:
    # without its marker a combo gets no ablation, comparison row, eval or
    # export, instead of being read as current against vectors it was not
    # trained on. Kept with --skip-generate and --after: this submission then
    # rebuilds no file.
    if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
        _n=0
        for _a in none joint per_modality both; do
            _r=$(pca_arm_results "$_a")
            for _f in "$_r"/latest_combo_cv[123].txt "$_r"/var_*__latest_combo_cv[123].txt \
                      "$_r"/allmod__*latest_allmod_cv[123].txt; do
                [ -f "$_f" ] || continue
                rm -f -- "$_f"
                _n=$((_n + 1))
            done
        done
        if [ "$_n" -gt 0 ]; then
            echo "  removed $_n combo marker(s) of variants or PCA arms this run does not"
            echo "  retrain (it regenerates the embeddings their combos were trained on)"
        fi
    fi
    # The table across the arms compares the combos this submission replaces
    # with the others'. The block printed at the end rebuilds it, over this
    # run's arms, with two arms or more; with one, nothing does (one arm has
    # nothing to compare: run_paper_tables.sh --pca_arms_table refuses fewer
    # than two, and over every arm this one would be listed as another
    # submission's, stale), as the root says ("builds none (one arm)"). Said
    # so since 2026-10-21; until then this said the printed jobs rebuild it.
    _ARMS_TABLE_GONE=false
    if [ -e "$BASE_DIR/sl_comparison/results/pca_arms" ]; then
        rm -rf -- "$BASE_DIR/sl_comparison/results/pca_arms"
        echo "  removed the previous table across the PCA arms, sl_comparison/results/pca_arms/"
        _ARMS_TABLE_GONE=true
        [ "$_N_ARMS" -lt 2 ] || _CMP_GONE="${_CMP_GONE:-${PCA_ARM_LIST[*]}}"
    fi
    if [ -n "$_CMP_GONE" ]; then
        # The jobs that rebuild them are printed at the end, wired to this
        # submission's (their ids are known only there; until 2026-10-21 this
        # printed them here, unwired, "once the combos land": pasted, the
        # external transfer ran before the eval, the paper tables beside the
        # compare and the cross-arm table before every arm's). Without the
        # eval no job builds external_transfer/ (its tables come from it).
        echo "  This script queues no sl_comparison job: the jobs that rebuild them, wired"
        echo "  to this submission's, are printed at the end."
        [ "$RUN_EVAL" = true ] || \
        echo "  Not the external-transfer tables: with --skip-eval there is no eval to build them from."
    fi
    if [ "$_ARMS_TABLE_GONE" = true ] && [ "$_N_ARMS" -lt 2 ]; then
        echo "  No job rebuilds the table across the arms: one PCA arm has nothing to compare"
        echo "  (it needs one submission's selection of two arms or more)."
    fi
    unset _ARMS_TABLE_GONE
fi
# --ablation-only rebuilds these under (usually) the ablated run's token, which
# would also accept the previous ablation's dirs. A best-per-category run
# retrains the combos they are read against, so they describe the combos this
# submission replaces (the root orchestrator's step 0a drops them on every
# siamese run for the same reason). Without this, --skip-ablation — or an
# ablation task that SKIPs on a failed combo or a missing marker — would leave
# the previous run's dirs, and paper_tables, which checks an ablation dir
# against --ablation_run_token, else RUN_TOKEN, else best_per_category.json's
# _selection_meta.run_token, would tabulate them in tab:loco against THIS run's
# combos: with --ablation-only they carry the very token this submission joins,
# and nothing at all is checked when none of the three is set (a selection that
# recorded no token, read by a standalone run_paper_tables.sh). Per arm, in its
# own root.
if [ "$RUN_BEST_CAT" = true ] \
   || { [ "$ABLATION_ONLY" = true ] && [ "$RUN_ABLATION" = true ]; }; then
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _R=$(pca_arm_results "$_arm")
        _n=0
        for _v in "${SIAMESE_VARIANTS[@]}"; do
            for _d in "$_R/$(siamese_variant_prefix "$_v")"ablate_no_*; do
                [ -e "$_d" ] || continue
                rm -rf -- "$_d"
                _n=$((_n + 1))
            done
        done
        [ "$_n" -gt 0 ] && echo "  removed $_n previous ablation dir(s) of ${SIAMESE_VARIANTS[*]} in $_R/ (PCA arm $_arm)"
        rm -f "$_R/ablation_summary.csv"
        # --ablation-only: the arm's paper tables (tab:loco) were built from
        # the dirs just removed, and no job here rebuilds them, so a still
        # PAPER COMPLETE README published an ablation that no longer exists,
        # also when the new one fails. They go, and the end of this script
        # prints their rebuild, wired to the ablation summary (2026-10-24).
        # (A best-per-category run removed them with its combos, above.)
        _C=$(pca_arm_cmp "$_arm")
        if [ "$ABLATION_ONLY" = true ] && [ -e "$BASE_DIR/$_C/paper_tables" ]; then
            rm -rf -- "$BASE_DIR/$_C/paper_tables"
            echo "  removed previous paper tables $_C/paper_tables/ (PCA arm $_arm; built on that ablation)"
        fi
    done
    if [ "$ABLATION_ONLY" = true ] && [ -e "$BASE_DIR/sl_comparison/results/pca_arms" ]; then
        rm -rf -- "$BASE_DIR/sl_comparison/results/pca_arms"
        echo "  removed previous table across the arms sl_comparison/results/pca_arms/ (built on those)"
    fi
fi
unset _v _p _t _cv _f _d _n _a _r _R _C _arm _CMP_GONE
echo ""

# ============================================================================
# Step 0: Environment reset (optional)
# ============================================================================
# Recreates the Python venv from scratch on a GPU node.

# A submission that fails part-way (e.g. a QOS submit limit at the 72-task
# ablation array) exits under set -e with part of the DAG queued: name what to
# cancel before resubmitting, as run_benchmarking_siamese_vs_others.sh does.
# Every job this submission queued, of every arm (_queued after each sbatch).
_on_submit_error() {
    [ -n "$_QUEUED" ] || return 0
    # Each command alone on its line (a trailing "# ..." reached scancel as
    # job ids in an interactive zsh, which has interactivecomments off), the
    # rebuild with the settings taken from the calling shell (TORCH_SPEC,
    # TORCH_INDEX_URL: reset_env.sh reads them).
    echo "ERROR: submission stopped part-way; already queued:$_QUEUED"
    echo "  Cancel them before resubmitting:"
    echo "    scancel$_QUEUED"
    if [ -n "$ENV_JOB" ]; then
        echo "  reset_env ($ENV_JOB) deletes the venv as soon as it starts: once it has"
        echo "  started (squeue -j $ENV_JOB), let it finish, or after cancelling it rebuild"
        echo "  the venv before resubmitting:"
        echo "    ${_ENV_PREFIX}sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} slurm/reset_env.sh"
    fi
}
trap _on_submit_error ERR

ENV_DEP=""

# --skip-generate (also set by --ablation-only; refused with --after above) is
# honoured, since the caller asked, but not silently: the same check the root
# orchestrator prints for it. Embeddings from older generator code, or built
# over another gene universe, would otherwise train here with nothing in the
# log saying so. BEFORE the reset is queued, as in the root orchestrator: the
# check runs on this node from the venv reset_env.sh rm -rf's first thing, and
# a reset job that starts at once removes it under the check (no interpreter,
# a traceback, or every current file listed STALE "unreadable"). Inside `if !`,
# so a missing interpreter or checker warns instead of aborting the submission
# under `set -e`.
if [ "$SKIP_GENERATE" = true ]; then
    echo "--- Checking the embeddings on disk (--skip-generate) ---"
    if ! "$PYTHON_PATH" "$WORK_DIR/embedding_versions.py" \
            --data_dir "$DATA_DIR" --gene_universe "$GENE_UNIVERSE"; then
        echo "WARNING: --skip-generate with embeddings embedding_versions.py cannot confirm"
        echo "         as current: every training job below uses them as they are."
    fi
    echo ""
fi

if [ "$RESET_ENV" = true ]; then
    echo "--- Submitting environment reset ---"
    # --after <JOB>: that job may still be running from the venv reset_env.sh
    # rm -rf's, so the reset starts only once it has finished (afterany);
    # training still waits on both (afterok, Step 1).
    ENV_AFTER=""
    [ -n "$AFTER_JOB" ] && ENV_AFTER="--dependency=afterany:$AFTER_JOB"
    ENV_JOB=$(sbatch --parsable $GPU_NODE_ARGS --export=ALL $ENV_AFTER "$SCRIPT_DIR/reset_env.sh")
    _queued "$ENV_JOB"
    echo "  Reset env job: $ENV_JOB"
    ENV_DEP="--dependency=afterok:$ENV_JOB"
    echo ""
fi

# ============================================================================
# Step 1: Embedding generation (optional)
# ============================================================================
# Produces all .pt embedding files in DATA_DIR, which every arm reads. Must
# complete before any training jobs start.

GEN_DEP=""

if [ "$SKIP_GENERATE" = false ] && [ -z "$AFTER_JOB" ]; then
    echo "--- Submitting embedding generation ---"
    GEN_JOB=$(sbatch --parsable $GPU_NODE_ARGS --export=ALL $ENV_DEP $KILL "$SCRIPT_DIR/run_generate_embeddings.sh")
    _queued "$GEN_JOB"
    echo "  Generation job: $GEN_JOB"
    GEN_DEP="--dependency=afterok:$GEN_JOB"
elif [ -n "$AFTER_JOB" ]; then
    echo "--- Chaining after existing job $AFTER_JOB ---"
    # SLURM `afterok:A:B` waits on both A and B — chain env reset too if set.
    if [ -n "$ENV_DEP" ]; then
        GEN_DEP="--dependency=afterok:$ENV_JOB:$AFTER_JOB"
    else
        GEN_DEP="--dependency=afterok:$AFTER_JOB"
    fi
else
    echo "--- Skipping generation (--skip-generate) ---"
    # (embedding_versions.py already ran, before Step 0 queued the env reset.)
    # If env reset was requested, training jobs still depend on it
    GEN_DEP="$ENV_DEP"
fi
echo ""

# ============================================================================
# Steps 2-4: the single-embedding runs, once per owner
# ============================================================================
# Step 2: embedding benchmark (array job + summary): each embedding x all CV
# types, array size len(ALL_EMBEDDINGS) x len(CV_TYPES), computed from the .conf.
# Step 4 (3a in the job scripts): best-per-category Step 1, the runs the combo
# ranks: array size bestcat_n_tasks (run_best_per_category.conf):
# len(EMB_CATALOG) x len(CV_TYPES), times 1 + the other variants with
# BESTCAT_EXTRA_VARIANTS=1.
# Each owner's runs train with its arm's single-embedding PCA (config.conf
# PCA_SINGLE_ARGS: native dimension for none / joint, PCA within the modality
# for per_modality / both) into its results root, submitted as the owner
# (PCA_ARM=<owner>), and serve every arm of its pair this submission trains.

if [ "$RUN_BENCHMARK" = true ]; then
    # Source benchmark config for any overrides (shared defaults from config.conf)
    source "$SCRIPT_DIR/run_embedding_benchmark.conf"

    BENCH_NUM_EMB=${#ALL_EMBEDDINGS[@]}
    NUM_CVS=${#CV_TYPES[@]}
    BENCH_TASKS=$((BENCH_NUM_EMB * NUM_CVS))

    if [ "$BENCH_TASKS" -lt 1 ]; then
        echo "WARNING: No embeddings or CV types defined in config.conf"
        echo "  Skipping benchmark."
        RUN_BENCHMARK=false
    fi
fi

if [ "$RUN_BEST_CAT" = true ]; then
    # Source best-per-category config for EMB_CATALOG (and the PCA_VARIANCE dir tag).
    source "$SCRIPT_DIR/run_best_per_category.conf"

    BESTCAT_NUM_EMB=${#EMB_CATALOG[@]}
    NUM_CVS=${#CV_TYPES[@]}
    BESTCAT_TASKS=$(bestcat_n_tasks)
    bestcat_task_variants

    if [ "$BESTCAT_TASKS" -lt 1 ]; then
        echo "WARNING: No embeddings or CV types defined in run_best_per_category.conf"
        echo "  Skipping best-per-category."
        RUN_BEST_CAT=false
    fi
fi

for _owner in $_OWNERS; do
    { [ "$RUN_BENCHMARK" = true ] || [ "$RUN_BEST_CAT" = true ]; } || break
    _R=$(pca_arm_results "$_owner")
    echo "=== Single-embedding runs of PCA arm(s) $(_served "$_owner") -> $_R/ ==="
    echo "    $(pca_step1_label "$_owner")"
    if [ "$RUN_BENCHMARK" = true ]; then
        echo "--- Submitting embedding benchmark ---"
        echo "  $BENCH_NUM_EMB embeddings x $NUM_CVS CVs = $BENCH_TASKS tasks"
        echo "  Max concurrent: $(throttle_label "$MAX_CONCURRENT")"

        # Submit array job: each task trains one (embedding, CV) combination
        BENCH_JOB=$(_as_arm "$_owner" sbatch --parsable $GPU_NODE_ARGS \
            $GEN_DEP $KILL \
            --export=ALL $(_logs "$_owner" bench %A_%a) \
            --array=0-$((BENCH_TASKS - 1))$(array_throttle "$MAX_CONCURRENT") \
            "$SCRIPT_DIR/run_embedding_benchmark.sh")
        _queued "$BENCH_JOB"
        echo "  Benchmark array job: $BENCH_JOB"

        # Submit summary after all array tasks finish (afterany = run even if some failed)
        BENCH_SUMM=$(_as_arm "$_owner" sbatch --parsable $GPU_NODE_ARGS \
            --dependency=afterany:$BENCH_JOB \
            --export=ALL $(_logs "$_owner" bench_summary %j) \
            "$SCRIPT_DIR/run_embedding_benchmark_summary.sh")
        _queued "$BENCH_SUMM"
        echo "  Benchmark summary job: $BENCH_SUMM  -> $_R/embedding_benchmark_summary.json"
        printf -v "BENCH_JOB_$_owner" '%s' "$BENCH_JOB"
        printf -v "BENCH_SUMM_$_owner" '%s' "$BENCH_SUMM"
    fi
    if [ "$RUN_BEST_CAT" = true ]; then
        echo "--- Submitting best-per-category Step 1 ---"
        echo "  $BESTCAT_NUM_EMB embeddings x $NUM_CVS CVs x ${#BESTCAT_TASK_VARIANTS[@]} variant(s) = $BESTCAT_TASKS tasks"
        if [ "${#BESTCAT_TASK_VARIANTS[@]}" -gt 1 ]; then
            echo "  (BESTCAT_EXTRA_VARIANTS=1: ${BESTCAT_TASK_VARIANTS[*]:1} for reporting only;"
            echo "   the combos wait for them too)"
        fi
        echo "  Single-embedding runs: $_R/pcavar${PCA_VARIANCE}_* ($(pca_single_args "$_owner"))"
        echo "  Max concurrent: $(throttle_label "$MAX_CONCURRENT")"

        # Step 3a: array job for single-embedding training
        BESTCAT_JOB=$(_as_arm "$_owner" sbatch --parsable $GPU_NODE_ARGS \
            $GEN_DEP $KILL \
            --export=ALL $(_logs "$_owner" bestcat %A_%a) \
            --array=0-$((BESTCAT_TASKS - 1))$(array_throttle "$MAX_CONCURRENT") \
            "$SCRIPT_DIR/run_best_per_category.sh")
        _queued "$BESTCAT_JOB"
        echo "  Best-cat array job: $BESTCAT_JOB"
        printf -v "BESTCAT_JOB_$_owner" '%s' "$BESTCAT_JOB"
    fi
    echo ""
done

# ============================================================================
# Steps 5-7, per PCA arm: the combo, the external-screen eval, the ablation
# ============================================================================
# Step 5 (3b in the job scripts): select the winners on the arm's owner's
# single-embedding runs and train the arm's multi-modal combos (single job,
# afterany on the owner's Step-1 array), with the arm's PCA (config.conf
# PCA_MULTI_ARGS).
#
# Step 6, external-screen evaluation + fine-tuning, targets every variant's
# CV3 combo trained in Step 5. The combo script writes
# `<arm root>/<prefix>latest_combo_cv3.txt` per variant on success (prefix ""
# for the baseline), which eval_finetuning.conf reads to resolve MODEL_DIR
# automatically — no need to hard-code the combo's name (which depends on
# which embeddings win per category). One array task per (cv_type, variant).
# `afterok` (not afterany) so eval only runs if the combo succeeded.
# Skipped when RUN_BEST_CAT is false (nothing fresh to evaluate) or when
# the user passes --skip-eval.
#
# Step 7, the leave-one-category-out ablation, retrains the combo once per
# (variant, dropped category, CV), reusing the winners in the arm's
# best_per_category.json, with the arm's PCA like its reference. The full
# combo is NOT retrained — the summary reads the combo named by each variant's
# <arm root>/<prefix>latest_combo_<cv>.txt as its reference row, so this stage
# cannot change any reported number. Runs by default; --skip-ablation opts
# out. Chains off the COMBO, in parallel with the external-screen eval — both
# become eligible as soon as the combo lands and SLURM overlaps them (capped
# by %MAX_CONCURRENT only when set). With --ablation-only every arm's
# best_per_category.json must already exist, and the only dependency is the
# env reset when --reset-env is also given.

if [ "$RUN_ABLATION" = true ]; then
    # run_best_per_category.conf carries EMB_CATALOG (the source of the category
    # labels) and is NOT sourced above when --ablation-only skips the other
    # stages. NUM_CVS is likewise only assigned inside those stages, so recompute
    # it here from CV_TYPES (config.conf, sourced at the top) rather than
    # inheriting a variable that may never have been set.
    source "$SCRIPT_DIR/run_best_per_category.conf"
    source "$SCRIPT_DIR/run_ablation.conf"
    ABL_NUM_CVS=${#CV_TYPES[@]}
    N_ABL_CAT=${#ABLATE_CATEGORIES[@]}
    ABL_NUM_VAR=${#SIAMESE_VARIANTS[@]}
    ABL_TASKS=$((N_ABL_CAT * ABL_NUM_CVS * ABL_NUM_VAR))

    if [ "$ABL_TASKS" -lt 1 ]; then
        echo "ERROR: ablation resolved to $ABL_TASKS tasks (categories=$N_ABL_CAT, cvs=$ABL_NUM_CVS, variants=$ABL_NUM_VAR)."
        _on_submit_error
        exit 1
    fi

    if [ "$ABLATION_ONLY" != true ] && [ "$RUN_BEST_CAT" != true ]; then
        echo "--- Skipping ablation: no combo is being trained this run ---"
        echo "  Use --ablation-only to ablate an existing best_per_category.json."
        echo ""
        RUN_ABLATION=false
    fi
fi
if [ "$RUN_EVAL" = true ] && [ "$RUN_BEST_CAT" = true ]; then
    # eval_finetuning.sh decodes each task's variant only when neither override
    # is set; one exported from the calling shell (meant for a direct sbatch)
    # would evaluate a single model and no-op every task past the cv list.
    if [ -n "${EVAL_MODEL_DIR:-}" ] || [ -n "${EVAL_VARIANT:-}" ]; then
        echo "NOTE: ignoring EVAL_MODEL_DIR/EVAL_VARIANT from the environment for the"
        echo "      chained eval (use --eval-only or a direct sbatch for one model)."
    fi
    unset EVAL_MODEL_DIR EVAL_VARIANT
    # eval_finetuning.conf honours an exported PER_HEAD (an override for a
    # direct sbatch of one model). The combos this run trains load only with
    # the value it derives from USE_CELL_LINES: one inherited from the calling
    # shell, or from the shell a printed line is pasted into, could at best
    # repeat it, and any other stopped every eval task at its first
    # checkpoint. Ignored with a NOTE, as the root does (2026-10-22).
    # --eval-only, which submits above, keeps it.
    if [ -n "${PER_HEAD:-}" ]; then
        echo "NOTE: ignoring PER_HEAD=$PER_HEAD from the environment: the chained eval follows"
        echo "      USE_CELL_LINES (--eval-only or a direct sbatch to override it for one model)."
    fi
    unset PER_HEAD
fi
# Whether the post-hoc cell-line ablation applies to this run's combos (the
# root orchestrator's condition for queueing it): a per_head variant, whose
# cell embedding it zeroes, under USE_CELL_LINES=1. Not queued here: the
# block printed at the end holds it, wired, for every arm.
_HAS_CELL=false
if [ "${USE_CELL_LINES:-1}" = 1 ]; then
    for _v in "${SIAMESE_VARIANTS[@]}"; do
        case "$_v" in *:per_head) _HAS_CELL=true ;; esac
    done
fi

for _arm in "${PCA_ARM_LIST[@]}"; do
    { [ "$RUN_BEST_CAT" = true ] || [ "$RUN_ABLATION" = true ]; } || break
    # (every id of this arm starts empty, not the previous arm's)
    _arm_reset
    _owner=$(pca_step1_arm "$_arm")
    _R=$(pca_arm_results "$_arm")
    echo "=== PCA arm $_arm: $(pca_arm_label "$_arm") -> $_R/ ==="

    if [ "$RUN_BEST_CAT" = true ]; then
        echo "--- Submitting best-per-category combo ---"
        _v="BESTCAT_JOB_$_owner"
        # Step 3b: combo job after all of the owner's array tasks finish
        COMBO_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
            --dependency=afterany:${!_v} \
            --export=ALL $(_logs "$_arm" bestcat_combo %j) \
            "$SCRIPT_DIR/run_best_per_category_combo.sh")
        _queued "$COMBO_JOB"
        echo "  Best-cat combo job: $COMBO_JOB  (ranks the Step-1 array ${!_v}, $(pca_arm_results "$_owner")/pcavar${PCA_VARIANCE}_*)"
        echo "  PCA: $(pca_multi_args "$_arm")"
        # (one command for it, the block's: printed here as well, it was
        # pasted twice, two jobs over one arm's rows, 2026-10-21)
        if [ "$_HAS_CELL" = true ]; then
            echo "  NOTE: the post-hoc cell-line ablation (tab:cellline) is not submitted here: it is"
            echo "        the _l_$_arm line of the block printed at the end, after this combo and"
            echo "        before this arm's comparison and paper tables (left out, they run without it)."
        else
            echo "  (no post-hoc cell-line ablation: no per_head variant with USE_CELL_LINES=1)"
        fi
        echo ""
        # ALLMOD_COMBO=1: the all-modalities combo (every catalog embedding,
        # no modality selection), one task per CV over every variant. It reads
        # no Step-1 run or selection, so it waits only for the embeddings (the
        # generation, else the env reset; nothing with --skip-generate
        # --skip-reset-env) and runs beside the Step-1 array.
        if [ "$ALLMOD_COMBO" = 1 ]; then
            echo "--- Submitting all-modalities combo (ALLMOD_COMBO=1) ---"
            ALLMOD_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                $GEN_DEP $KILL \
                --export=ALL $(_logs "$_arm" allmod_combo %A_%a) \
                --array=0-$((${#CV_TYPES[@]} - 1))$(array_throttle "$MAX_CONCURRENT") \
                "$SCRIPT_DIR/run_allmod_combo.sh")
            _queued "$ALLMOD_JOB"
            echo "  All-modalities job: $ALLMOD_JOB  (${#CV_TYPES[@]} tasks: one per CV, ${#SIAMESE_VARIANTS[@]} variant(s) each; ${GEN_DEP:-no dependency})"
            echo "  Output: $_R/allmod__<prefix>all_modalities_<cv>/, rows $(for _v in "${SIAMESE_VARIANTS[@]}"; do printf '%s ' "$(allmod_row "$_v")"; done)"
            echo ""
        fi
    fi

    if [ "$RUN_EVAL" = true ] && [ "$RUN_BEST_CAT" = true ]; then
        echo "--- Submitting external-screen eval + fine-tuning ---"
        echo "  Target combo: each variant's cv3 combo, resolved at job-start from"
        echo "                $_R/<prefix>latest_combo_cv3.txt"
        echo "  Datasets:     Adamson, Corn, Gilbert (see eval_finetuning.conf)"
        # One array task per (cv_type, variant), so the evaluations overlap.
        EVAL_CVS=$(eval_cv_count)
        EVAL_TASKS=$((EVAL_CVS * ${#SIAMESE_VARIANTS[@]}))
        EVAL_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
            --dependency=afterok:$COMBO_JOB $KILL \
            --export=ALL $(_logs "$_arm" eval_finetuning %A_%a) \
            --array=0-$((EVAL_TASKS - 1))$(array_throttle "$EVAL_CONCURRENT") \
            "$SCRIPT_DIR/eval_finetuning.sh")
        _queued "$EVAL_JOB"
        echo "  Eval job:     $EVAL_JOB  ($EVAL_CVS cv_types x ${#SIAMESE_VARIANTS[@]} variants = $EVAL_TASKS tasks)"
        echo "  Tail:         tail -f slurm/logs/eval_finetuning_$(_armtag "$_arm")${EVAL_JOB}_0.out"
        echo ""
    fi

    if [ "$RUN_ABLATION" = true ]; then
        # (--ablation-only's best_per_category.json check runs before Step 0.)
        echo "--- Submitting leave-one-category-out ablation ---"
        echo "  $N_ABL_CAT categories x $ABL_NUM_CVS CVs x $ABL_NUM_VAR variants = $ABL_TASKS tasks"
        echo "  Max concurrent: $(throttle_label "$MAX_CONCURRENT")"
        # The arm's own PCA, which its reference combos were trained with (and
        # each task checks against the reference's config.json).
        echo "  PCA: $(pca_multi_args "$_arm") (the arm's, as its reference combos')"

        # Chains off the COMBO, not off the eval, so the ablation array and the
        # external-screen eval are both eligible to run as soon as the combo lands
        # and SLURM can overlap them. GPU contention is the scheduler's to resolve
        # (MAX_CONCURRENT caps the arrays only when set); serialising the two
        # stages here would only trade wall-clock for a guarantee SLURM provides.
        ABL_DEP=""
        if [ "$ABLATION_ONLY" != true ] && [ -n "${COMBO_JOB:-}" ]; then
            ABL_DEP="--dependency=afterok:$COMBO_JOB"
            if [ -n "$EVAL_JOB" ]; then
                echo "  Runs concurrently with the external-screen eval (both after job $COMBO_JOB)"
            else
                echo "  Runs after the combo (job $COMBO_JOB)"
            fi
        elif [ -n "$GEN_DEP" ]; then
            # --ablation-only --reset-env: SKIP_GENERATE=true (and --after is
            # rejected with it), so GEN_DEP is the env reset's afterok.
            ABL_DEP="$GEN_DEP"
        fi

        ABL_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
            $ABL_DEP $KILL \
            --export=ALL $(_logs "$_arm" ablation %A_%a) \
            --array=0-$((ABL_TASKS - 1))$(array_throttle "$MAX_CONCURRENT") \
            "$SCRIPT_DIR/run_ablation.sh")
        _queued "$ABL_JOB"
        echo "  Ablation array job: $ABL_JOB"

        # afterany, not afterok: a single failed category should still produce a
        # table for the ones that succeeded, with the failures shown as "not run".
        ABL_SUMM=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
            --dependency=afterany:$ABL_JOB \
            --export=ALL $(_logs "$_arm" ablation_summary %j) \
            "$SCRIPT_DIR/run_ablation_summary.sh")
        _queued "$ABL_SUMM"
        echo "  Ablation summary:   $ABL_SUMM"
        echo "  Output:             $_R/ablation_summary.csv"
        echo ""
    fi
    _arm_save "$_arm"
done

# ============================================================================
# Step 8: All-gene-pair export (the LAST jobs, one per arm)
# ============================================================================
# ONE job per arm over every variant's CV3 combo of that arm
# (<arm root>/<prefix>latest_combo_cv3.txt): the predicted SL probability of
# every gene pair, one matrix per cell-line head, and the fine-tuned models'
# predicted Adamson value for every gene pair -> <combo>/pair_predictions/
# (slurm/export_pair_predictions.conf sets what is written; 42 GB per arm by
# default). afterany on every job queued above, of EVERY arm, never afterok:
# nothing reads its output, a failed eval task or variant must not cancel the
# export of the rest, and it must not compete for disk with a job a result
# depends on, another arm's included. singleton: never beside another
# pair_export job of yours (the arms' exports run one after another; a
# --eval-only export of the same combo would clear this one's files). Only
# when this run trains combos, so not with --benchmark-only / --ablation-only.
# A submission problem is a WARNING: the rest of the pipeline is queued and
# does not depend on it.

if [ "$RUN_PAIR" = true ] && [ "$RUN_BEST_CAT" = true ]; then
    PAIR_DEPS=""
    for _a in "${PCA_ARM_LIST[@]}"; do
        for _v in COMBO_JOB EVAL_JOB ABL_SUMM ALLMOD_JOB; do
            _n="${_v}_$_a"
            [ -n "${!_n}" ] && PAIR_DEPS="$PAIR_DEPS:${!_n}"
        done
    done
    for _o in $_OWNERS; do
        _n="BENCH_SUMM_$_o"
        [ -n "${!_n}" ] && PAIR_DEPS="$PAIR_DEPS:${!_n}"
    done
    PAIR_DEPS="${PAIR_DEPS#:}"
    # export_pair_predictions.sh exports ONE combo when PAIR_EXPORT_MODEL_DIR
    # is set (a direct sbatch, or --eval-only above); inherited here it would
    # skip every variant's combo of this run.
    if [ -n "${PAIR_EXPORT_MODEL_DIR:-}" ]; then
        echo "NOTE: ignoring PAIR_EXPORT_MODEL_DIR from the environment for the chained export."
    fi
    unset PAIR_EXPORT_MODEL_DIR
    if [ "$RUN_EVAL" != true ]; then
        # eval_finetuning.py writes the fine-tuned models; without the eval
        # the export would end FAILED on the missing sets.
        export PAIR_EXPORT_FT_SPLITS=none
    fi
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _arm_load "$_arm"
        echo "--- Submitting all-gene-pair export of PCA arm $_arm (last) ---"
        [ "$RUN_EVAL" = true ] || echo "  No eval in this run: probabilities only (no fine-tuned models to score)"
        # (the arm rides in front of sbatch, _as_arm / _arm_env)
        _PAIR_OPTS="--export=ALL"
        _L=$(_logs "$_arm" pair_export %j); [ -z "$_L" ] || _PAIR_OPTS="$_PAIR_OPTS $_L"
        if PAIR_JOB=$(_as_arm "$_arm" sbatch --parsable $GPU_NODE_ARGS \
                --dependency=afterany:$PAIR_DEPS,singleton \
                $_PAIR_OPTS \
                "$SCRIPT_DIR/export_pair_predictions.sh"); then
            _queued "$PAIR_JOB"
            echo "  Pair-export job: $PAIR_JOB  (afterany:$PAIR_DEPS)"
        else
            PAIR_JOB=""
            _PAIR_ENV=""
            [ "$RUN_EVAL" = true ] || _PAIR_ENV="PAIR_EXPORT_FT_SPLITS=none "
            # Word for word the root's command: the settings taken from the
            # calling shell first (_ENV_PREFIX, 2026-10-21), then $_PAIR_ENV
            # and the arm, whose last assignment wins (--skip-eval's
            # PAIR_EXPORT_FT_SPLITS=none over a recorded one, as the refused
            # job got it).
            echo "  WARNING: could not submit the all-gene-pair export of PCA arm $_arm; every"
            echo "  other job is queued. It has to stay the last job: while they are queued,"
            echo "    ${_ENV_PREFIX}$(_vset "${SIAMESE_VARIANTS[*]}") PAIR_EXPORT_MODEL_DIR= ${_PAIR_ENV}$(_arm_env "$_arm") sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --dependency=afterany:$PAIR_DEPS,singleton $_PAIR_OPTS slurm/export_pair_predictions.sh"
            echo "  or the same without --dependency once every job above has finished."
        fi
        _arm_save "$_arm"
        echo ""
    done
    unset _a _v _n _o _L
fi

# ============================================================================
# Done
# ============================================================================
trap - ERR
echo "============================================================================"
echo "All jobs submitted. Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f slurm/logs/<job_file>.out   (<stage>_pca_<arm>_<id>.out for arm joint, ...)"
echo ""
echo "Results land in each PCA arm's results root (<prefix> is empty for pd:per_head,"
echo "var_<kernel>_<cell_lines>__ otherwise):"
for _arm in "${PCA_ARM_LIST[@]}"; do
echo "  $(pca_arm_results "$_arm")/  PCA arm $_arm: $(pca_arm_label "$_arm")"
done
if [ "$RUN_BEST_CAT" = true ]; then
echo "  <root>/<prefix>best_cat_*_<cv>/results.json  multimodal combo, per variant"
echo "  <root>/best_per_category.json               the arm's modality selection"
fi
if [ "$RUN_BEST_CAT" = true ] && [ "$RUN_EVAL" = true ]; then
echo "  <combo>/eval_finetuning_<cv>/                 external screens (Adamson/Corn/Gilbert)"
fi
if [ "$RUN_ABLATION" = true ]; then
# (the summary job covers this submission's SIAMESE_VARIANTS: a variant subset,
# or one line of a split --ablation-only refusal, leaves the others' rows out;
# "all variants" until 2026-10-21)
echo "  <root>/<prefix>ablate_no_<category>_<cv>/    leave-one-category-out runs"
echo "  <root>/ablation_summary.csv                  ablation table (this submission's variants)"
fi
for _o in $_OWNERS; do
if [ "$RUN_BEST_CAT" = true ]; then
echo "  $(pca_arm_results "$_o")/pcavar${PCA_VARIANCE}_<type>_<cv>/  Step-1 runs of PCA arm(s) $(_served "$_o")"
fi
if [ "$RUN_BENCHMARK" = true ]; then
echo "  $(pca_arm_results "$_o")/<type>_<cv>/, embedding_benchmark_summary.json  benchmark of $(_served "$_o")"
fi
done
_PAIR_JOBS=""
for _arm in "${PCA_ARM_LIST[@]}"; do
    _n="PAIR_JOB_$_arm"
    [ -z "${!_n}" ] || _PAIR_JOBS="${_PAIR_JOBS:+$_PAIR_JOBS }$_arm:${!_n}"
done
if [ -n "$_PAIR_JOBS" ]; then
# As the job will resolve them: the conf's defaults under this environment (in
# a subshell: they must reach neither this shell nor the jobs). Only a cv3
# combo is asked for fine-tuned sets. (No `case` in here: bash 3.2 cannot parse
# one inside $( ).)
_PAIR_FT=$(source "$SCRIPT_DIR/export_pair_predictions.conf" >/dev/null 2>&1 || true
           _s=none
           for _c in ${PAIR_EXPORT_CVS:-cv3}; do
               [ "$_c" != cv3 ] || _s="${PAIR_EXPORT_FT_SPLITS:-}"
           done
           if [ "$_s" = none ]; then echo "no fine-tuned predictions"
           else echo "+ fine-tuned ${PAIR_EXPORT_FT_DATASETS:-Adamson} predictions"; fi)
_PAIR_CVS=$(source "$SCRIPT_DIR/export_pair_predictions.conf" >/dev/null 2>&1 || true
            echo ${PAIR_EXPORT_CVS:-cv3})
echo "  <$_PAIR_CVS combo>/pair_predictions/                 SL probability of all gene pairs per cell line"
echo "                                                ($_PAIR_FT); the last jobs:"
for _pj in $_PAIR_JOBS; do
echo "                                                PCA arm ${_pj%%:*}: job ${_pj#*:}"
done
fi
# The sl_comparison side of this run, which no job here queues: ready to paste
# and wired to this submission's jobs as the root orchestrator chains its own
# (until 2026-10-21 step 0a printed them unwired, "once the combos land", and
# pasted so, the external transfer ran before the eval, the paper tables beside
# the compare and the table across the arms before any arm's). Per arm: the
# post-hoc cell-line ablation after its combo (afterok; the NOTE under each
# combo points here, so it has one command, never two), the compare after the
# combo and the cell-line ablation (it tabulates the *_nocell rows), the
# external transfer after the eval (none without one: its tables are built from
# this run's eval outputs, as in the root's --skip-eval), the paper tables
# after all of them and the ablation summary; then the table across the arms
# after every arm's paper tables, submitted as the first arm (the reference).
# Printed whenever this run trains combos, on state, not on what step 0a
# removed: a resubmission after a stop part-way finds nothing left to remove.
# Each job runs under _paste_env (config.conf pasted_env_def: none of the
# pasting shell's RUN_TOKEN, CODE_MANIFEST or compare pins; this script writes
# no code manifest, and each arm's tables then check its dirs against its own
# selection's token) with this run's arms, PCA targets and variants, which a
# one-shot prefix on this script's command line never left in that shell, and
# logs named after its arm (the root's names). Since 2026-10-21 also, by their
# values as this shell resolved them (an empty one still set, NAME=, so the
# pasting shell's never wins), every other setting those jobs read that this
# run's own jobs got from this shell or its defaults: SELECTION_VARIANT,
# BEST_CAT_PCA_VARIANCE (the Step-1 dir tag, run_best_per_category.conf's
# PCA_VARIANCE, sourced above whenever this block prints) and
# BESTCAT_EXTRA_VARIANTS (run_paper_tables.sh resolves all three from the
# confs: tab:solo read pcavar0.8_* dirs of no run, a non-default selection
# variant's every cell read as another variant's, both false PAPER
# INCOMPLETEs), ARCH_ARGS (paper_tables records it: without it the published
# model's run was recorded with the defaults) and CVS (run_compare.sh's CVs; empty:
# its default, all three). Checked and not needed: SEED, PCA_METHOD,
# PREPROCESSING_FIT_SCOPE, SIAMESE_ENCODER_TYPE, CELL_LINE_DIM, SHARED_FOLDS
# and SELECTION_CV (no pasted job reads their values: the graders read the
# fold set by its fixed path, and SELECTION_CV is only validated there).
if [ "$RUN_BEST_CAT" = true ]; then
    _TS=""; _LS=""
    if [ "$_HAS_CELL" = true ]; then
        for _arm in "${PCA_ARM_LIST[@]}"; do _LS="$_LS _l_$_arm="; done
    fi
    # (no external transfer under --skip-eval: no eval to build it from, so
    # no _e_<arm> line below)
    if [ "$RUN_EVAL" = true ]; then
echo "  sl_comparison, none of it queued here: to rebuild every arm's comparison, external"
echo "    transfer and paper tables, paste these where this ran (each job waits on this"
    else
echo "  sl_comparison, none of it queued here: to rebuild every arm's comparison and paper"
echo "    tables (not the external transfer: with --skip-eval there is no eval to build it"
echo "    from), paste these where this ran (each job waits on this"
    fi
echo "    run's; \$_L is their log directory). _paste_env, defined on the first line, gives"
echo "    every job this run's arms, PCA targets, variants, selection variant, Step-1 dir"
echo "    tag (BEST_CAT_PCA_VARIANCE), BESTCAT_EXTRA_VARIANTS, ARCH_ARGS and CVS, and none of"
echo "    the pasting shell's RUN_TOKEN, CODE_MANIFEST or compare pins (SIAMESE_RESULTS_*,"
echo "    SIAMESE_BASELINE_DISCOVER)."
[ "$_HAS_CELL" != true ] || \
echo "    The _l_<arm> lines are the post-hoc cell-line ablation (tab:cellline): left out, the"
[ "$_HAS_CELL" != true ] || \
echo "    comparison and paper tables run without it."
echo "      $(pasted_env_def "PCA_ARMS=$PCA_ARMS" $(pca_targets_env) \
          "SIAMESE_VARIANTS_LIST=${SIAMESE_VARIANTS[*]}" "USE_CELL_LINES=${USE_CELL_LINES:-1}" \
          "SELECTION_VARIANT=$SELECTION_VARIANT" "BEST_CAT_PCA_VARIANCE=$PCA_VARIANCE" \
          "BESTCAT_EXTRA_VARIANTS=${BESTCAT_EXTRA_VARIANTS:-1}" "ARCH_ARGS=${ARCH_ARGS:-}" \
          "CVS=${CVS:-}" "ALLMOD_COMBO=$ALLMOD_COMBO")"
printf '      cd %q && _L=%q%s\n' "$BASE_DIR" "$BASE_DIR/sl_comparison/slurm/logs" \
    "${_LS:+ &&$_LS}"
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _arm_load "$_arm"
        _AT=$(_armtag "$_arm")
        _J="_paste_env PCA_ARM=$_arm sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --parsable"
        # the cell-line job in the dependencies below, when its line was pasted
        _LD=""
        if [ "$_HAS_CELL" = true ]; then
            _L=$(_logs "$_arm" cellline_ablation %j)
echo "      _l_$_arm=\$(cd siamese_sl && $_J --dependency=afterok:$COMBO_JOB $KILL --export=ALL${_L:+ $_L} cell_line_ablation/run_cell_line_ablation.sh)"
            _LD="\${_l_$_arm:+:\$_l_$_arm}"
        fi
echo "      _c_$_arm=\$($_J --dependency=afterany:$COMBO_JOB${ALLMOD_JOB:+:$ALLMOD_JOB}$_LD --export=ALL --output=\$_L/final_compare_${_AT}%j.out --error=\$_L/final_compare_${_AT}%j.err sl_comparison/slurm/run_compare.sh)"
        _ED=""
        if [ -n "$EVAL_JOB" ]; then
echo "      _e_$_arm=\$($_J --dependency=afterany:$EVAL_JOB --export=ALL --output=\$_L/external_transfer_${_AT}%j.out --error=\$_L/external_transfer_${_AT}%j.err sl_comparison/slurm/run_external_transfer.sh)"
            _ED=":\$_e_$_arm:$EVAL_JOB"
        fi
echo "      _t_$_arm=\$($_J --dependency=afterany:\$_c_$_arm$_LD$_ED${ABL_SUMM:+:$ABL_SUMM} --export=ALL --output=\$_L/paper_tables_${_AT}%j.out --error=\$_L/paper_tables_${_AT}%j.err sl_comparison/slurm/run_paper_tables.sh)"
        _TS="$_TS:\$_t_$_arm"
    done
    if [ "$_N_ARMS" -ge 2 ]; then
echo "      _paste_env PCA_ARM=${PCA_ARM_LIST[0]} sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --dependency=afterany$_TS --export=ALL --output=\$_L/paper_tables_pca_arms_%j.out --error=\$_L/paper_tables_pca_arms_%j.err sl_comparison/slurm/run_paper_tables.sh --pca_arms_table"
    fi
[ -z "$_PAIR_JOBS" ] || \
echo "    (The pair exports queued above wait on none of these.)"
    unset _TS _LS _J _LD _ED _AT _L _arm
elif [ "$ABLATION_ONLY" = true ] && [ "$RUN_ABLATION" = true ]; then
    # the paper tables removed above (tab:loco), after each arm's ablation
    # summary; the arm's own token for tab:loco when its selection records
    # none (the WARNING above); then the table across the arms (2026-10-24)
    _pv=$( (source "$SCRIPT_DIR/run_best_per_category.conf" >/dev/null 2>&1 || true
           echo "${PCA_VARIANCE:-}") )
    _TS=""
echo "  sl_comparison, none of it queued here: to rebuild every arm's paper tables (their"
echo "    tab:loco is this ablation), paste these where this ran (each job waits on this run's"
echo "    ablation summary; \$_L is their log directory). _paste_env, defined on the first line,"
echo "    gives every job this run's arms, PCA targets, variants and settings, and none of the"
echo "    pasting shell's RUN_TOKEN, CODE_MANIFEST or compare pins."
echo "      $(pasted_env_def "PCA_ARMS=$PCA_ARMS" $(pca_targets_env) \
          "SIAMESE_VARIANTS_LIST=${SIAMESE_VARIANTS[*]}" "USE_CELL_LINES=${USE_CELL_LINES:-1}" \
          "SELECTION_VARIANT=$SELECTION_VARIANT" "BEST_CAT_PCA_VARIANCE=$_pv" \
          "BESTCAT_EXTRA_VARIANTS=${BESTCAT_EXTRA_VARIANTS:-1}" "ARCH_ARGS=${ARCH_ARGS:-}" \
          "CVS=${CVS:-}" "ALLMOD_COMBO=$ALLMOD_COMBO")"
printf '      cd %q && _L=%q\n' "$BASE_DIR" "$BASE_DIR/sl_comparison/slurm/logs"
    for _arm in "${PCA_ARM_LIST[@]}"; do
        _arm_load "$_arm"
        _AT=$(_armtag "$_arm")
        _v="_ABL_TOK_$_arm"
echo "      _t_$_arm=\$(_paste_env PCA_ARM=$_arm sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --parsable${ABL_SUMM:+ --dependency=afterany:$ABL_SUMM} --export=ALL --output=\$_L/paper_tables_${_AT}%j.out --error=\$_L/paper_tables_${_AT}%j.err sl_comparison/slurm/run_paper_tables.sh${!_v:+ --ablation_run_token ${!_v}})"
        _TS="$_TS:\$_t_$_arm"
    done
    if [ "$_N_ARMS" -ge 2 ]; then
echo "      _paste_env PCA_ARM=${PCA_ARM_LIST[0]} sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} --dependency=afterany$_TS --export=ALL --output=\$_L/paper_tables_pca_arms_%j.out --error=\$_L/paper_tables_pca_arms_%j.err sl_comparison/slurm/run_paper_tables.sh --pca_arms_table"
    fi
    unset _TS _AT _arm _v _pv
fi
echo "============================================================================"
