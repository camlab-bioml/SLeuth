#!/bin/bash
#SBATCH --job-name=bestcat_combo
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/logs/bestcat_combo_%j.out
#SBATCH --error=slurm/logs/bestcat_combo_%j.err

# ============================================================================
# Best-per-Category — Combo (Steps 2-4)
# ============================================================================
# Runs after all Step 1 array tasks complete (dependency: afterany).
#
# Step 2: Select the best embedding per biological category using
#         single-embedding VALIDATION scores on SELECTION_CV (default cv1):
#         val_<select_metric>_mean from each run's own config (the metric its
#         checkpoint selection used), else val_auprg_mean. No TEST score is
#         ever used. The SAME winners are reused for every CV combo. This is
#         NOT leakage-free: the five SELECTION_CV validation sets are pooled,
#         and each fold's training pool holds other folds' test pairs (see the
#         note at the selection below).
#         Reads $STEP1_ROOT/pcavar<variance>_<type>_<cv>/results.json (the
#         single-embedding runs of the job's PCA arm: config.conf
#         pca_step1_arm, shared by none / joint and by per_modality / both)
#         and saves the selection to $RESULTS_ROOT/best_per_category.json (the
#         arm's results root: results/, or results/pca_<arm>/). Ranks only
#         candidates whose run_token.txt is this submission's RUN_TOKEN
#         (config.conf). Stops when a ranked SELECTION_CV candidate is not
#         this run's (other variant / signature, the arm's single-embedding
#         PCA included / fold set), when a candidate
#         whose embedding exists has no score (failed or never-started Step-1
#         task, or another submission's token), or when a whole category has
#         none; BEST_CAT_ALLOW_MISSING=1 selects on what is there instead (the
#         missing ones are recorded in the JSON's _selection_meta).
#         SELECTION_MODE=per_fold (config.conf; needs SELECTION_FRAC > 0,
#         2026-11-12) ranks instead, for EACH CV and EACH fold k, the Step-1
#         runs of that CV on fold k's own selection split (a random
#         SELECTION_FRAC of fold k's training pairs, never trained on:
#         fold_metrics[k]["sel_<select_metric>"]); every check above applies
#         to every CV's candidates, and candidates whose selection splits
#         differ (sel_digest) or that lack the key stop the job
#         (fold_selection.py). best_per_category.json keeps its keys (per
#         category the most-picked "type"/"file") and adds each category's
#         "per_fold" picks, "types"/"files" and "n_folds_picked".
#
# Every training call of this job (and of the Step-1 / ablation jobs) takes
# SELECTION_ARGS (--selection_frac / --selection_seed when SELECTION_FRAC is
# set), so every SLeuth model of the run trains on the same pairs.
#
# Step 3: Train multi-modal combos for every (variant, CV) — SIAMESE_VARIANTS x
#         CV_TYPES — by concatenating the SAME winning embeddings (one per
#         category) for every variant; only --kernel/--cell_lines differ,
#         and every combo trains with the arm's PCA (PCA_MULTI_ARGS).
#         Per-fold mode: ONE train.py run per (variant, CV) over the UNION of
#         that CV's per-fold winners (categories sorted, catalog order within
#         one; the combo dir is named after the union) with --fold_modalities
#         mapping each fold to its own winners; every checkpoint records its
#         subset, so the readers of config.json embeddings_paths (eval, pair
#         export, cell-line ablation) apply each fold's own.
#         On each successful combo, writes
#         $RESULTS_ROOT/<prefix>latest_combo_${CV}.txt (prefix "" for the
#         baseline pd:per_head, "var_<kernel>_<cell_lines>__" otherwise)
#         pointing at the combo's output dir
#         $RESULTS_ROOT/<prefix>best_cat_<names>_<cv> — consumed by the chained
#         eval_finetuning.sh, run_ablation.sh and its summary, the cell-line
#         ablation, the pair export and the sl_comparison siamese collector.
#         The previous run's combos are moved to
#         $RESULTS_ROOT/_superseded_combos/ at job start and deleted once Step 2
#         succeeds (see there).
#
# Step 4: Print summary table with single-embedding and multi-modal results
#         (one block per variant).
#
# Submitted automatically by both orchestrators, once per PCA arm of
# PCA_ARMS, after the Step-1 array of the arm that owns its single-embedding
# runs (config.conf pca_step1_arm), with the arm in sbatch's environment:
#   PCA_ARM=<arm> sbatch --dependency=afterany:<that Step-1 array's job id> \
#       --export=ALL slurm/run_best_per_category_combo.sh
#
# Submitted BY HAND (a direct sbatch from siamese_sl/, e.g. to retrain a
# failed combo: RUN_TOKEN=<the run's token>, PCA_ARM=<its arm> (without it the
# arm the submitting shell exports, else none), PCA_ARMS=<the submission's,
# default all four: with PCA_ARM it decides whose single-embedding runs Step 2
# ranks, config.conf pca_step1_arm>, SELECTION_CV=<the submission's, default
# cv1: in no signature, so nothing in the job stops one that selects on
# another CV; 2026-10-21> and
# SIAMESE_VARIANTS_LIST="<variant> ..." to limit it, in front of sbatch;
# every CV of each listed variant is retrained): the new combos take the SAME
# directory names, so once Step 2 has succeeded the job removes what both
# orchestrators remove at submission (the block after "Deleted previous
# combos": in the arm's namespaces, the listed variants' sl_comparison rows and
# markers, *_nocell rows, <prefix>cellline_ablation_<cv>/ and <prefix>ablate_no_*
# dirs, ablation_summary.csv, comparison.csv, external_transfer/ and
# paper_tables/; and the cross-arm table sl_comparison/results/pca_arms/,
# which compares every arm's combos). Nothing is queued behind it: rerun what it removed and what reads
# the new combos, sl_comparison/README.md, "After a combo job submitted by
# hand". Submit it by hand only once none of the jobs that read or write what
# it removes is queued or running (allmod_combo, bilinear_probe,
# eval_finetuning, ablation, ablation_summary, cellline_ablation, pair_export, sl_final_compare,
# sl_ext_transfer, sl_paper_tables; wait for them or scancel them): the job
# does not check, and it removes the selection and its variants' markers and
# combos as it starts. Both orchestrators refuse these removals while such a
# job is alive.
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"
# OPTIONAL_TYPES only (it also sets PRECOMPUTED_TYPES, which nothing here
# reads): the catalog types whose absence generation accepts. Without them
# Step 2 cannot tell an accepted absence (esmc, scprint, seqvec) from a
# REQUIRED catalog embedding that was never built, and would log both the same
# way — on a --skip-generate path the generation job's own REQUIRED_MISSING
# check never runs.
source "$SLURM_SUBMIT_DIR/slurm/run_generate_embeddings.conf"

# FIRST, before anything below can fail: clean the previous selection JSON and
# the per-CV marker files. A marker is the only thing that says "this combo is
# THIS run's": the chained eval_finetuning job and the sl_comparison collector
# read them, so a stale one must never outlive a job that fails before Step 3
# rewrites it. Per variant, and only for the variants this job trains: the
# baseline's prefix is "" (<arm root>/latest_combo_*.txt, <arm root>/best_cat_*,
# exactly as before), and no other variant's var_* names match those globs. In
# the job's PCA arm only (RESULTS_ROOT, config.conf): the globs do not recurse,
# so arm none's never reach another arm's results/pca_<arm>/.
rm -f "$RESULTS_ROOT/best_per_category.json"
# An arm other than none writes one directory down (results/pca_<arm>/), which
# the first run of that arm has to create: Step 2 writes the selection there
# before any combo directory exists.
mkdir -p "$RESULTS_ROOT"
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    rm -f "$RESULTS_ROOT"/${VPREFIX}latest_combo_*.txt
done

# The previous combo DIRECTORIES ($RESULTS_ROOT/<prefix>best_cat_*, with their
# checkpoints and external evaluations) are MOVED aside now and DELETED only
# once Step 2 has produced a new selection (start of Step 3). A run whose
# selection fails therefore keeps the previous models (mv them back to reuse
# them), but nowhere a consumer looks: not at <arm root>/<prefix>best_cat_* (the
# ablation reference / summary, Step 4). The sl_comparison collector
# (collect_siamese.py --combo_markers, via collect_and_regrade_siamese.sh)
# reads only the combos named by this run's <arm root>/<prefix>latest_combo_<cv>.txt
# markers, deleted above, so it never reaches a previous run's combo, moved or
# not: the marker deletion, not the directory name, is what keeps a failed or
# cancelled combo job from publishing the previous run's models as this run's.
# The one exception is the opt-in SIAMESE_BASELINE_DISCOVER=all baseline sweep
# (no --combo_markers, no freshness check): its rglob still reaches the moved
# dirs, a moved pd:per_head combo can be selected, and a moved var_* combo is
# dropped only by the recorded kernel/cell_lines check. Kept under the arm's
# results root so the .gitignore rules for results/ (prediction matrices,
# checkpoints) still cover them; one directory per job, so a second failed run
# never collides with the first.
SUPERSEDED_DIR="$RESULTS_ROOT/_superseded_combos/job_${SLURM_JOB_ID:-local_$$}"
N_SUPERSEDED=0
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    for d in "$RESULTS_ROOT"/${VPREFIX}best_cat_*; do
        [ -d "$d" ] || continue
        mkdir -p "$SUPERSEDED_DIR"
        mv "$d" "$SUPERSEDED_DIR/superseded__$(basename "$d")"
        N_SUPERSEDED=$((N_SUPERSEDED + 1))
    done
done

sleep 10  # let disk settle after prior job

echo "BEST-PER-CATEGORY — COMBO (Steps 2-4)"
echo "  PCA arm: $PCA_ARM ($(pca_arm_label "$PCA_ARM"); $PCA_MULTI_ARGS) -> $RESULTS_ROOT/"
echo "  Single-embedding runs: $(pca_step1_label "$PCA_ARM") ($STEP1_ROOT/, dir tag pcavar$PCA_VARIANCE)"
echo "  Variants (${#SIAMESE_VARIANTS[@]}): ${SIAMESE_VARIANTS[*]}   selection: $SELECTION_VARIANT"
echo "  ARCH_ARGS: ${ARCH_ARGS:-(none; train.py defaults)}"
echo "  Modality selection: SELECTION_MODE=$SELECTION_MODE  split: ${SELECTION_ARGS:-(off: SELECTION_FRAC empty)}"
echo ""
echo "Cleaned previous selection + markers (${SIAMESE_VARIANTS[*]})"
if [ "$N_SUPERSEDED" -gt 0 ]; then
    echo "Moved $N_SUPERSEDED previous combo dir(s) to $SUPERSEDED_DIR"
    echo "  (deleted once Step 2 succeeds; kept there if it fails)"
fi
echo ""

# ============================================================================
# STEP 2: Select best embedding per category per CV
# ============================================================================
# Reads $STEP1_ROOT/pcavar<variance>_<type>_<cv>/results.json produced by the array job.
# Writes $RESULTS_ROOT/best_per_category.json.

# Build Python list literal from bash CV_TYPES array: ["cv1", "cv2", "cv3"]
# Used by all Python heredocs below so cv_types stays in sync with the .conf.
CV_TYPES_PY="[$(printf '"%s", ' "${CV_TYPES[@]}" | sed 's/, $//')]"

if [ "$SELECTION_MODE" = per_fold ]; then
    echo "--- STEP 2: Select best per category per (CV, fold) on each fold's selection split ---"
else
    echo "--- STEP 2: Select best per category (selection CV: $SELECTION_CV) ---"
fi

# The architecture / selection signature THIS run's Step-1 candidates must
# carry. Without it Step 2 only checks that the candidates agree with EACH
# OTHER, and a whole Step-1 array trained under other ARCH_ARGS or
# hyperparameters passes: this job re-sources config.conf days after the array
# started, and the hint below invites resubmitting it onto an earlier array's
# RUN_TOKEN. The modalities would then be chosen under one architecture and one
# validation metric and every combo trained under another.
# Built from run_best_per_category.sh's argument list, the arm's
# single-embedding PCA (PCA_SINGLE_ARGS) included, so a candidate trained with
# another PCA (another arm's runs) stops the job, and on SELECTION_VARIANT
# (the variant block 0 trains). train.py --print_arch loads no data and writes nothing, and neither
# embeddings_paths nor cv_type is a signature field, so any catalog entry
# stands for all of them. Keep the `| tail -1`: it also keeps a train.py
# failure from tripping set -e (an empty value is handled in PYSELECT).
_E="${EMB_CATALOG[0]#*:}"
EXPECT_STEP1_SIG=$($PYTHON_PATH train.py \
    --embeddings_paths "$DATA_DIR/${_E%%:*}" \
    --sl_path "$SL_PATH" \
    ${NEG_PATH:+--neg_pairs_path "$NEG_PATH"} \
    $CELL_LINE_ARGS \
    --output_dir results/_sigcheck \
    --cv_type "$SELECTION_CV" \
    $MODEL_ARGS \
    --encoder_dims $ENCODER_DIMS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --l1_lambdas $L1_LAMBDAS \
    --pd_epsilon $PD_EPSILON \
    --eval_interval $EVAL_INTERVAL \
    --patience $PATIENCE \
    --warmrestart_T0 $WARMRESTART_T0 \
    --warmrestart_Tmult $WARMRESTART_TMULT \
    --num_folds $NUM_FOLDS \
    --pos_neg_ratio $POS_NEG_RATIO \
    --seed $SEED \
    --preprocessing_fit_scope "${PREPROCESSING_FIT_SCOPE:-train}" \
    --pca_method "${PCA_METHOD:-plain}" \
    --siamese_encoder_type "${SIAMESE_ENCODER_TYPE:-residual}" \
    $PCA_SINGLE_ARGS \
    $(siamese_variant_args "$SELECTION_VARIANT") \
    $SELECTION_ARGS \
    $ARCH_ARGS \
    --print_arch | tail -1)
export EXPECT_STEP1_SIG

$PYTHON_PATH << PYSELECT
import json, os
from pathlib import Path

PCA_VARIANCE = "$PCA_VARIANCE"
SELECTION_CV = "$SELECTION_CV"
RUN_TOKEN = "$RUN_TOKEN"
# For the resubmission hints below (2026-10-21). A job submitted by hand is
# the arm its shell exports, else arm none's, unless PCA_ARM is passed
# (config.conf), and as it starts it
# removes that arm's selection and markers and moves its combos aside: a hint
# without the arm, followed for a failed arm, wiped arm none and left the
# failed arm as it was. So every hint names the arm, the run's PCA_ARMS (with
# PCA_ARM they decide STEP1_ROOT, whose runs Step 2 ranks) and this run's
# token, and a Step-1 rerun the arm that owns those runs, in the env form the
# orchestrators submit with. Filled in by bash once, here, so the messages
# themselves hold no dollar sign (the heredoc is unquoted).
PCA_ARM = "$PCA_ARM"
PCA_ARMS = "$PCA_ARMS"
STEP1_OWNER = "$(pca_step1_arm "$PCA_ARM")"     # whose array trains step1_dir
STEP1_READERS = "$(pca_step1_arms "$PCA_ARM")"  # the arms whose combos rank it
# A combo resubmission carries this run's SELECTION_CV as well (2026-10-21):
# it is in no signature (every Step-1 task trains every CV), so a hand job
# without it selected on run_best_per_category.conf's default, exited 0 under
# the run's token and sat beside arms selected on the run's CV; only the table
# across the arms notices (paper_tables build_pca_arms lists such an arm
# stale). A Step-1 rerun needs none: run_best_per_category.sh does not read
# it. READER_ENV is for the combo of each arm that ranks step1_dir.
# The selection split and mode are in the hints too (2026-11-13): a hand
# resubmission without SELECTION_MODE trained the combo on global selection
# under the run's token with nothing refusing it.
SELECTION_FRAC = "${SELECTION_FRAC:-}"
SELECTION_SEED = "$SELECTION_SEED"
SELECTION_MODE = "$SELECTION_MODE"
# Always, empty included (2026-11-13): with the 0.1 default an omitted
# SELECTION_FRAC is no longer "off".
_SEL_ENV = (f" SELECTION_FRAC={SELECTION_FRAC} SELECTION_SEED={SELECTION_SEED}"
            f" SELECTION_MODE={SELECTION_MODE}")
RUN_ENV = f"PCA_ARMS='{PCA_ARMS}' SELECTION_CV={SELECTION_CV}{_SEL_ENV}"
ARM_ENV = f"PCA_ARM={PCA_ARM} {RUN_ENV}"
READER_ENV = f"PCA_ARM=<that arm> {RUN_ENV}"
COMBO_CMD = "sbatch --export=ALL slurm/run_best_per_category_combo.sh"
# Nothing is chained behind a hand job: what the orchestrator queued after
# the combo (its afterok jobs) was cancelled with it.
AFTER_HAND = ('Then rerun what reads the new combos (sl_comparison/README.md, '
              '"After a combo job submitted by hand").')

# Parse catalog from shell array
catalog = []
for line in """$(printf '%s\n' "${EMB_CATALOG[@]}")""".strip().split('\n'):
    etype, efile, ecat = line.split(':')
    catalog.append((etype, efile, ecat))

cv_types = $CV_TYPES_PY  # Injected from CV_TYPES (config.conf; overridable in run_best_per_category.conf)
# The arm's results root (the selection is written there) and the root of its
# single-embedding runs (the candidates): config.conf RESULTS_ROOT / STEP1_ROOT.
results_dir = Path("$RESULTS_ROOT")
step1_dir = Path("$STEP1_ROOT")

if SELECTION_CV not in cv_types:
    print(f"ERROR: SELECTION_CV={SELECTION_CV!r} not in CV_TYPES={cv_types}.")
    raise SystemExit(1)

# The CVs whose Step-1 runs Step 2 ranks (2026-11-12): SELECTION_CV alone in
# the global mode (its winners are reused for every CV), every CV in the
# per_fold mode (each (CV, fold) picks its own winners on its own selection
# split, config.conf SELECTION_MODE). RANK_TXT names them in messages.
SELECTION_MODE = "$SELECTION_MODE"
PER_FOLD = SELECTION_MODE == "per_fold"
RANK_CVS = list(cv_types) if PER_FOLD else [SELECTION_CV]
RANK_TXT = (f"every CV ({', '.join(RANK_CVS)}; SELECTION_MODE=per_fold)"
            if PER_FOLD else "SELECTION_CV")
RANK_NAME = ", ".join(RANK_CVS)   # SELECTION_CV's name in the global mode


def step1_rerun(etypes, cvs=None):
    """The hand rerun of etypes' Step-1 tasks of cvs (default: the ranked
    CVs, RANK_CVS), as the arm that owns step1_dir, under this run's token.
    etypes may also be (etype, cv) pairs: those tasks only. Their array task
    ids are block 0's (SELECTION_VARIANT, the runs Step 2 ranks):
    run_best_per_category.sh maps
    task = (block * n_emb + emb_index) * n_cv + cv_index."""
    order = [e for e, _, _ in catalog]
    pairs = sorted({(x if isinstance(x, tuple) else (x, c))
                    for x in etypes for c in (cvs or RANK_CVS)},
                   key=lambda p: (order.index(p[0]), cv_types.index(p[1])))
    ids = ",".join(str(order.index(e) * len(cv_types) + cv_types.index(c))
                   for e, c in pairs)
    return (f"RUN_TOKEN={RUN_TOKEN} PCA_ARM={STEP1_OWNER} "
            f"PCA_ARMS='{PCA_ARMS}' sbatch --export=ALL "
            f"--array={ids or '<task ids>'} slurm/run_best_per_category.sh")

# Collect PCA benchmark results.
# Selection metric MUST match train.py's checkpoint-selection metric, run by
# run (the same rule collect_siamese.py uses for the siamese row):
#   val_<select_metric>_mean, select_metric read from the run's own config
#   (train.py --select_metric: auprg_any by default, auprg for the published
#   rule; runs predating the flag selected on auprg), falling back to
#   val_auprg_mean. Cell-line and single-output runs write the same keys (a
#   single-output run is one "ANY" column, where the two metrics coincide);
#   a candidate with none of them is never ranked (see below).
# PREFER THE VALIDATION KEY. This argmax picks which embedding modality wins its
# category, and the winners are concatenated into the final combo model — so
# ranking them on a TEST metric chooses the model's ARCHITECTURE on test, a
# winner's curse over |catalog| candidates that sits underneath the per-config
# one in collect_siamese.py. train.py writes the val_* means whenever it ran
# with --folds_dir; a candidate without one is an error, never ranked on test.
TEST_KEY = "auprg_mean"


def val_keys_for(full):
    """Validation keys to try, in order, for one run's results.json."""
    sm = (full.get("config") or {}).get("select_metric") or "auprg"
    return [f"val_{sm}_mean", "val_auprg_mean"]


def label_for(key):
    if "auprg_any" in key:
        return "AUPRG_any"
    return "AUPRG" if "auprg" in key else "AUPR"


scores = {}
bases = {}          # (etype, cv) -> "val" | "test", the basis of scores[...]
keys_used = {}      # (etype, cv) -> the summary key scores[...] was read from
configs = {}        # (etype, cv) -> the run's recorded train.py config
fulls = {}          # (etype, cv) -> the whole results.json (per_fold ranks
                    #                its fold_metrics)
foreign = {}        # (etype, cv) -> the run_token.txt found (None: missing)
for etype, efile, ecat in catalog:
    for cv in cv_types:
        rfile = step1_dir / f"pcavar{PCA_VARIANCE}_{etype}_{cv}" / "results.json"
        if rfile.exists():
            # Only THIS submission's Step-1 output is a candidate. A task that
            # never started (cancelled while pending, NODE_FAIL; this job runs
            # afterany) never reached its rm -rf, so its directory still holds
            # a previous run's results.json, which can agree with this run on
            # variant, signature and fold stamp and still be stale (other
            # embeddings, code or ARCH_ARGS). run_best_per_category.sh writes
            # run_token.txt only after a successful train.
            _tf = rfile.parent / "run_token.txt"
            _tok = _tf.read_text().strip() if _tf.is_file() else None
            if _tok != RUN_TOKEN:
                foreign[(etype, cv)] = _tok
                continue
            try:
                with open(rfile) as _f:
                    full = json.load(_f)
                d = full["summary"]
                aupr, basis, used = None, "val", None
                for vk in val_keys_for(full):
                    if d.get(vk) is not None:
                        aupr, used = d[vk], vk
                        break
                if aupr is None:
                    aupr, basis, used = d.get(TEST_KEY), "test", TEST_KEY
                if aupr is not None:
                    scores[(etype, cv)] = aupr
                    bases[(etype, cv)] = basis
                    keys_used[(etype, cv)] = used
                    configs[(etype, cv)] = full.get("config") or {}
                    fulls[(etype, cv)] = full
            except (KeyError, json.JSONDecodeError):
                pass

# Count on the population the argmax ACTUALLY consumes — only SELECTION_CV
# candidates decide the winners, so counting across every cv described the
# wrong set.
sel_bases = [b for (etype, cv), b in bases.items() if cv in RANK_CVS]
n_val = sel_bases.count("val")
n_test = sel_bases.count("test")
print(f"Selection basis on {RANK_NAME}: {n_val} val / {n_test} test")
if n_val and n_test:
    # HARD STOP. val_auprg and test auprg are different quantities; a max()
    # over a mix silently lets a test-selected candidate beat a val-selected
    # one on a number that was never comparable, and the winner is then
    # concatenated into the published combo model. Warning was not enough —
    # the run would carry on and produce a table that looks fine.
    print(f"ERROR: candidates for {RANK_TXT} are scored on a MIX of val and "
          "test — these are not comparable and must not share an argmax.")
    for (etype, cv), b in sorted(bases.items()):
        if cv in RANK_CVS and b == "test":
            print(f"  test-only (no validation score): {etype}")
    print("Re-run those embeddings with --folds_dir (SHARED_FOLDS) so every "
          "candidate carries a validation score, or delete their stale "
          "results.json so they are regenerated.")
    raise SystemExit(1)
sel_keys = sorted({k for (etype, cv), k in keys_used.items()
                   if cv in RANK_CVS})
if len(sel_keys) > 1:
    # HARD STOP, for the same reason as the val/test mix above: runs selected
    # on different metrics (e.g. val_auprg_mean vs val_auprg_any_mean, from a
    # changed ARCH_ARGS --select_metric) carry scores of different quantities.
    print(f"ERROR: candidates for {RANK_TXT} are scored on different keys "
          f"{sel_keys}; they must not share an argmax.")
    for (etype, cv), k in sorted(keys_used.items()):
        if cv in RANK_CVS:
            print(f"  {etype}: {k}")
    print("Re-run the single-embedding array with one --select_metric (the "
          "same ARCH_ARGS), or delete the stale results.json files.")
    raise SystemExit(1)
VAL_KEY = sel_keys[0] if sel_keys else TEST_KEY
SEL_LABEL = label_for(VAL_KEY)
if n_test:
    # HARD STOP: selection is val-only end to end. Choosing the modalities on
    # TEST would pick the model's architecture on the numbers it reports.
    print(f"ERROR: {n_test} candidate(s) for {RANK_NAME} carry no "
          f"validation score, so the category winners could only be chosen on "
          f"TEST {TEST_KEY}.")
    for (etype, cv), b in sorted(bases.items()):
        if cv in RANK_CVS and b == "test":
            print(f"  test-only (no validation score): {etype}")
    print("Re-run them with --folds_dir (SHARED_FOLDS), or delete their stale "
          "results.json files so they are regenerated.")
    raise SystemExit(1)
print(f"Selection key on {RANK_NAME}: {VAL_KEY}")

for (etype, cv), _tok in sorted(foreign.items()):
    if cv in RANK_CVS:
        print(f"WARNING: {etype}/{cv}: results.json is not this run's (run_token "
              f"{_tok!r} != {RUN_TOKEN!r}); not ranked")
_n_foreign_other = sum(1 for (_e, cv) in foreign if cv not in RANK_CVS)
if _n_foreign_other:
    print(f"NOTE: {_n_foreign_other} non-{RANK_NAME} results.json from another "
          f"run ignored (they only feed the per-CV diagnostic scores)")

if not scores:
    print("ERROR: No PCA benchmark results found.")
    print("Step 1 array tasks may have all failed. Check slurm/logs/bestcat_*.out")
    if foreign:
        print(f"  {len(foreign)} results.json carry another run's token. To rank "
              f"a finished Step-1 array from an earlier submission on purpose, "
              f"resubmit this job as PCA arm {PCA_ARM} with RUN_TOKEN set to "
              f"its token (cat {step1_dir}/pcavar{PCA_VARIANCE}_<type>_<cv>/"
              f"run_token.txt), from siamese_sl/: RUN_TOKEN=<its token> "
              f"{ARM_ENV} {COMBO_CMD}, with that run's ARCH_ARGS, "
              f"hyperparameters and other PCA_* targets exported — the "
              f"signature check below refuses candidates trained with other "
              f"settings. Without PCA_ARM a job submitted by hand is the arm "
              f"its shell exports, else arm none's, and it removes that arm's "
              f"selection and markers as it starts. {AFTER_HAND}")
    raise SystemExit(1)

# Build lookup
type_to_file = {t: f for t, f, _ in catalog}

# --- Single selection on SELECTION_CV (NOT leakage-free) ---
# Pick winners ONCE using SELECTION_CV validation scores (default cv1), averaged
# over its folds, and reuse them for every fold of every CV and every variant.
# Each fold's val split is carved from its own training pairs, but the mean
# pools five folds whose training pools hold the OTHER folds' test pairs, so a
# pair that is a test pair in one outer fold (in cv1, and through shared pairs
# also in cv2/cv3) can influence the modality choice through another fold's
# training or validation pairs. A leakage-free version selects per (CV, fold)
# on that fold's own validation split only (nested selection): that is
# SELECTION_MODE=per_fold (below, after the checks), on a selection split
# carved from each fold's training pairs (train.py --selection_frac).
ref_candidates = {}
for etype, efile, ecat in catalog:
    if (etype, SELECTION_CV) in scores:
        ref_candidates.setdefault(ecat, []).append(
            (etype, scores[(etype, SELECTION_CV)]))
# The categories each ranked CV has a candidate for (per_fold: every CV).
cats_by_cv = {cv: {ecat for etype, _f, ecat in catalog if (etype, cv) in scores}
              for cv in RANK_CVS}

for _cv in RANK_CVS:
    if not cats_by_cv[_cv]:
        print(f"ERROR: No single-embedding results for SELECTION_CV={_cv}."
              if not PER_FOLD else
              f"ERROR: No single-embedding results for {_cv} "
              f"(SELECTION_MODE=per_fold ranks every CV).")
        print(f"Check that Step 1 array tasks for {_cv} completed.")
        raise SystemExit(1)

# Every ranked candidate must be THIS run's Step-1 output. The run_token filter
# above already drops another submission's directories (a Step-1 task that
# never started, even when the whole array never ran); these checks catch a
# candidate carrying this run's token that still does not fit it. Stop, naming
# the offenders, when the SELECTION_CV candidates do not share one
# architecture / selection signature (arch_signature.py), when one was not
# trained in this USE_CELL_LINES mode or (cell-line mode) as
# SELECTION_VARIANT, or when its model_params.json fold_stamp is not that of
# SHARED_FOLDS.
import importlib.util as _ilu
from arch_signature import arch_signature
_mp_spec = _ilu.spec_from_file_location(
    "_sl_comparison_model_params", "$BASE_DIR/sl_comparison/model_params.py")
_mp = _ilu.module_from_spec(_mp_spec)
_mp_spec.loader.exec_module(_mp)
SHARED_FOLDS = "$SHARED_FOLDS"
USE_CELL_LINES = "$USE_CELL_LINES" == "1"
SELECTION_VARIANT = "$SELECTION_VARIANT"
# THIS run's expected Step-1 signature (computed above with --print_arch).
try:
    EXPECT_SIG = json.loads(os.environ.get("EXPECT_STEP1_SIG") or "null")
except ValueError:
    EXPECT_SIG = None
stale, sig_groups = [], {}
stale_types = set()   # the candidates named in stale (their tasks rerun)
if EXPECT_SIG is None:
    # (not a stale candidate: the Step-1 runs are fine and must not be deleted)
    print("ERROR: could not compute this run's Step-1 signature: train.py "
          "--print_arch failed in THIS job (its error is in this job's .err "
          "log), so the candidates cannot be checked against this run's "
          "settings. The Step-1 results are untouched: fix this job's "
          "ARCH_ARGS / environment and resubmit it with the same RUN_TOKEN, "
          f"as PCA arm {PCA_ARM}, from siamese_sl/: RUN_TOKEN={RUN_TOKEN} "
          f"{ARM_ENV} {COMBO_CMD}, with the run's other PCA_* targets "
          "exported (without PCA_ARM a job submitted by hand is the arm its "
          "shell exports, else arm none's, and it removes that arm's "
          f"selection and markers). {AFTER_HAND}")
    raise SystemExit(1)
def _lab(etype, cv):
    """A candidate's name in messages: its type (global: one CV ranked),
    type/cv in the per_fold mode."""
    return f"{etype}/{cv}" if PER_FOLD else etype


def _key(etype, cv):
    """What step1_rerun takes for a candidate: its type (global, rerun on
    SELECTION_CV), the (type, cv) task in the per_fold mode."""
    return (etype, cv) if PER_FOLD else etype


for (etype, cv), cfg in sorted(configs.items()):
    if cv not in RANK_CVS:
        continue
    sig = arch_signature(cfg)
    sig_groups.setdefault(json.dumps(sig, sort_keys=True), []).append(
        (etype, cv))
    if EXPECT_SIG is not None:
        # The candidate must fit THIS run, not merely the other candidates.
        # Choosing the modalities under one architecture / selection metric and
        # training the combo under another is not the model this run publishes.
        _d = sorted(k for k in sig if sig[k] != EXPECT_SIG.get(k))
        if _d:
            stale.append(f"{_lab(etype, cv)}: trained with other settings "
                         f"than this run ({', '.join(_d)})")
            stale_types.add(_key(etype, cv))
    if bool(sig["use_cell_lines"]) != USE_CELL_LINES:
        stale.append(f"{_lab(etype, cv)}: use_cell_lines="
                     f"{sig['use_cell_lines']}, but "
                     f"this run has USE_CELL_LINES=$USE_CELL_LINES")
        stale_types.add(_key(etype, cv))
    elif (USE_CELL_LINES
          and f"{sig['kernel']}:{sig['cell_lines']}" != SELECTION_VARIANT):
        stale.append(f"{_lab(etype, cv)}: trained as "
                     f"{sig['kernel']}:{sig['cell_lines']}, "
                     f"not SELECTION_VARIANT {SELECTION_VARIANT}")
        stale_types.add(_key(etype, cv))
    if SHARED_FOLDS:
        rec = _mp.read_model_params(
            step1_dir / f"pcavar{PCA_VARIANCE}_{etype}_{cv}").get("fold_stamp")
        diffs = (_mp.compare_fold_stamp(rec, SHARED_FOLDS, cv) if rec
                 else ["no fold_stamp in model_params.json"])
        if diffs:
            stale.append(f"{_lab(etype, cv)}: not trained on the folds in "
                         f"{SHARED_FOLDS} ({'; '.join(diffs)})")
            stale_types.add(_key(etype, cv))
if len(sig_groups) > 1:
    _groups = sorted(sig_groups.items(), key=lambda kv: -len(kv[1]))
    _ref_sig, _ref_types = json.loads(_groups[0][0]), _groups[0][1]
    for _s, _types in _groups[1:]:
        _s = json.loads(_s)
        _d = ", ".join(f"{k} {_ref_sig[k]!r} vs {_s.get(k)!r}"
                       for k in sorted(_ref_sig) if _ref_sig[k] != _s.get(k))
        stale.append(f"{', '.join(_lab(*p) for p in _types)}: "
                     f"architecture/selection signature differs from "
                     f"{', '.join(_lab(*p) for p in _ref_types)}'s ({_d})")
        stale_types.update(_key(*p) for p in _types)
if stale:
    print(f"ERROR: the {RANK_NAME} candidates are not all this run's Step-1 "
          f"output (stale results.json from an earlier run?):")
    for s in stale:
        print(f"  {s}")
    print(f"Delete the stale {step1_dir}/pcavar{PCA_VARIANCE}_<type>_<cv> dirs and "
          f"re-run those Step-1 tasks as the arm that owns them, "
          f"{STEP1_OWNER}, from siamese_sl/: {step1_rerun(stale_types)} "
          f"(with the run's ARCH_ARGS and other PCA_* targets exported), "
          f"then, once it has finished, resubmit the combo of each arm that "
          f"ranks them ({STEP1_READERS}): RUN_TOKEN={RUN_TOKEN} "
          f"{READER_ENV} {COMBO_CMD}. Or "
          f"resubmit this job with the Step-1 run's ARCH_ARGS / config: "
          f"RUN_TOKEN={RUN_TOKEN} {ARM_ENV} {COMBO_CMD}. Either way every "
          f"candidate is then this run's; without PCA_ARM a job submitted by "
          f"hand is the arm its shell exports, else arm none's. {AFTER_HAND}")
    raise SystemExit(1)

# Name every catalog entry the argmax could NOT rank. Step 1 SKIPs (exit 0) an
# embedding whose .pt was never generated, by design: that only warns. A
# Step-1 task that FAILED (exit 1; this job runs afterany) or wrote an
# unreadable results.json leaves no score although the embedding exists, and a
# category with no candidate left drops out of every combo, which changes the
# model's definition: both stop the job unless BEST_CAT_ALLOW_MISSING=1.
# A never-generated embedding still only warns (generation is gated before this
# job: the generation job stops on a REQUIRED catalog type and the root
# orchestrator regenerates whenever embedding_versions.py exits non-zero), but
# the warning says whether the absence is an accepted OPTIONAL_TYPES one, since
# only --skip-generate — an explicit override — can bring a required one here.
DATA_DIR = "$DATA_DIR"
# run_generate_embeddings.conf, sourced above.
OPTIONAL_TYPES = set("""$(printf '%s ' "${OPTIONAL_TYPES[@]}")""".split())
ALLOW_MISSING = os.environ.get("BEST_CAT_ALLOW_MISSING") == "1"
# (etype, category, cv) entries; one CV (SELECTION_CV) in the global mode.
not_generated, failed, foreign_sel = [], [], []
for etype, efile, ecat in catalog:
    for cv in RANK_CVS:
        if (etype, cv) not in scores:
            if (etype, cv) in foreign:
                foreign_sel.append((etype, ecat, cv))
            elif os.path.isfile(os.path.join(DATA_DIR, efile)):
                failed.append((etype, ecat, cv))
            elif cv == RANK_CVS[0]:   # a missing file is missing for every CV
                not_generated.append((etype, ecat, cv))
missing_cats = sorted({c for _, _, c in catalog}
                      - set.intersection(*(cats_by_cv[c] for c in RANK_CVS)))
req_not_generated = [(e, c, v) for e, c, v in not_generated
                     if e not in OPTIONAL_TYPES]


def _types_of(entries):
    """The distinct etypes of (etype, category, cv) entries, in order."""
    return list(dict.fromkeys(e for e, _c, _v in entries))


for etype, ecat, _cv in not_generated:
    print(f"WARNING: {etype} ({ecat}) not ranked: {DATA_DIR}/ has no embedding "
          f"file for it." + ("" if etype in OPTIONAL_TYPES else
          f"  It is NOT in OPTIONAL_TYPES (run_generate_embeddings.conf), so "
          f"generation should have built it and the argmax is running on a "
          f"smaller catalog than the paper's — did this run skip generation?"))
for etype, ecat, _cv in failed:
    print(f"WARNING: {etype} ({ecat}) not ranked: no readable {_cv} "
          f"results.json although the embedding exists (Step-1 task failed or "
          f"never started? see slurm/logs/bestcat_*.out).")
for etype, ecat, _cv in foreign_sel:
    print(f"WARNING: {etype} ({ecat}) not ranked: its {_cv} results.json "
          f"is another run's (Step-1 task never started this submission?).")
if missing_cats:
    print(f"WARNING: no {RANK_NAME} candidate for categor(ies) "
          f"{missing_cats}; every combo would lack them.")
if (failed or foreign_sel or missing_cats) and not ALLOW_MISSING:
    _rerun = [(e, v) for e, _c, v in failed + foreign_sel]
    print("ERROR: the category argmax would run on an incomplete candidate set. "
          + (f"Re-run the failed Step-1 task(s) as the arm that owns them, "
             f"{STEP1_OWNER}, from siamese_sl/: {step1_rerun(_rerun)} (with "
             f"the run's ARCH_ARGS and other PCA_* targets exported), then, "
             f"once it has finished, resubmit the combo of each arm that "
             f"ranks them ({STEP1_READERS}): RUN_TOKEN={RUN_TOKEN} "
             f"{READER_ENV} {COMBO_CMD}. Or "
             if _rerun else
             f"No Step-1 task to rerun: a category has no embedding file "
             f"(generate it, then rerun the Step-1 array of arm {STEP1_OWNER} "
             f"and the combos of {STEP1_READERS}). Or ")
          + f"select on what is there: BEST_CAT_ALLOW_MISSING=1 "
            f"RUN_TOKEN={RUN_TOKEN} {ARM_ENV} {COMBO_CMD} (without PCA_ARM a "
            f"job submitted by hand is the arm its shell exports, else arm "
            f"none's). {AFTER_HAND}")
    raise SystemExit(1)

ref_winners = {}
for cat, candidates in ref_candidates.items():
    best_type, best_score = max(candidates, key=lambda x: x[1])
    ref_winners[cat] = {
        "type": best_type,
        "file": type_to_file[best_type],
        "aupr": round(best_score, 4),
    }

def val_score(etype, cv):
    """etype's VALIDATION score on cv under VAL_KEY, or None if it lacks one."""
    if keys_used.get((etype, cv)) != VAL_KEY:
        return None  # no run, or scored on another key (e.g. a test fallback)
    return scores[(etype, cv)]


# Replicate the single SELECTION_CV selection across all CV keys (so the combo step,
# which reads selection["$CV"], gets the same winners regardless of CV).
# The "aupr" field shown per CV is the selected winner's VALIDATION score
# (VAL_KEY) on THAT CV (informational diagnostic; None if that run lacks
# VAL_KEY), while the chosen "type"/"file" are identical.
SELECTION_FRAC = float("$SELECTION_FRAC" or 0)
SELECTION_SEED = int("$SELECTION_SEED")
NUM_FOLDS = int("$NUM_FOLDS")
if not PER_FOLD:
    selection = {}
    for cv in cv_types:
        selection[cv] = {}
        for cat, info in ref_winners.items():
            cv_score = val_score(info["type"], cv)
            selection[cv][cat] = {
                "type": info["type"],
                "file": info["file"],
                "aupr": round(cv_score, 4) if cv_score is not None else None,
            }
else:
    # --- Per-fold, per-tier selection (SELECTION_MODE=per_fold, 2026-11-12) ---
    # For EACH CV and EACH fold k: per category, the argmax of
    # fold_metrics[k]["sel_<select_metric>"] of that CV's ranked Step-1 runs,
    # the val-selected checkpoint scored once on fold k's selection split (a
    # uniform random SELECTION_FRAC of fold k's TRAINING pairs, never trained
    # on). Fold k's test pairs never reach fold k's choice, and no other
    # fold's numbers do. fold_selection.rank_per_fold refuses a candidate
    # without the key (trained with SELECTION_FRAC empty) and candidates whose
    # selection splits differ for one (CV, fold) (sel_digest: another
    # SELECTION_SEED / fraction or fold set).
    import fold_selection as _fs
    selection, sel_digests, FOLD_KEY, _bad = {}, {}, None, []
    for cv in cv_types:
        _cands = {e: fulls[(e, cv)] for e, _f, _c in catalog
                  if (e, cv) in scores}
        try:
            _r = _fs.rank_per_fold(catalog, _cands, NUM_FOLDS, cv)
        except _fs.SelectionError as e:
            _bad.append(str(e))
            continue
        selection[cv] = _fs.selection_entries(catalog, _r["picks"])
        sel_digests[cv] = {str(k): d for k, d in sorted(_r["digests"].items())}
        FOLD_KEY = FOLD_KEY or _r["key"]
        if _r["key"] != FOLD_KEY:
            _bad.append(f"{cv} ranks on {_r['key']}, another CV on {FOLD_KEY}")
    if _bad:
        print("ERROR: SELECTION_MODE=per_fold cannot rank the Step-1 runs:")
        for _b in _bad:
            for _p in _b.split("; "):
                print(f"  {_p}")
        print(f"Every Step-1 run must be this run's, trained with the same "
              f"selection split (SELECTION_FRAC={SELECTION_FRAC:g}, "
              f"SELECTION_SEED={SELECTION_SEED}) on the same folds. Re-run the "
              f"Step-1 array of arm {STEP1_OWNER} with them exported (or "
              f"resubmit this job with the settings the Step-1 runs were "
              f"trained with): RUN_TOKEN={RUN_TOKEN} {ARM_ENV} "
              f"SELECTION_MODE=per_fold SELECTION_FRAC=<f> SELECTION_SEED=<s> "
              f"{COMBO_CMD}. {AFTER_HAND}")
        raise SystemExit(1)
    VAL_KEY, SEL_LABEL = FOLD_KEY, label_for(FOLD_KEY)

selection["_selection_meta"] = {
    "selection_cv": SELECTION_CV,
    # The single-embedding runs were trained under this variant only; every
    # variant's combo reuses these winners (config.conf, SIAMESE_VARIANTS).
    "selection_variant": "$SELECTION_VARIANT",
    "selection_metric": VAL_KEY,
    # Record the human label and the basis directly. The consumer used to
    # re-derive the label by testing selection_metric == "auprg_mean", which is
    # never true on the default path: with a validation split present the key is
    # "val_auprg_mean", so the winners table printed AUPRG numbers under an
    # "AUPR" header on every cell-line run.
    "selection_label": SEL_LABEL,
    # Always "val": the n_test hard stop above refuses any other basis.
    "selection_basis": "val",
    "n_scored_on_val": n_val,
    "n_scored_on_test": n_test,
    # Catalog entries the argmax did not rank (see the check above); the
    # failed ones and missing categories reach here only under
    # BEST_CAT_ALLOW_MISSING=1.
    "unranked_not_generated": _types_of(not_generated),
    # Of those, the ones OUTSIDE OPTIONAL_TYPES: generation should have built
    # them, so their absence is not an accepted one (reachable only when
    # generation was skipped).
    "unranked_required_missing": _types_of(req_not_generated),
    "unranked_failed": _types_of(failed),
    # Another submission's results.json (run_token.txt missing or different).
    "unranked_foreign_token": _types_of(foreign_sel),
    "run_token": RUN_TOKEN,
    # The PCA arm this selection belongs to (config.conf), the PCA its combos
    # train with, and where its candidates were read: the arms none / joint
    # and per_modality / both share single-embedding runs, hence winners.
    "pca_arm": "$PCA_ARM",
    "pca": "$(pca_arm_label "$PCA_ARM")",
    "pca_args": "$PCA_MULTI_ARGS",
    "step1_root": "$STEP1_ROOT",
    "step1_pca": "$(pca_step1_label "$PCA_ARM")",
    "step1_pca_args": "$PCA_SINGLE_ARGS",
    "missing_categories": missing_cats,
    "rationale": "same winners reused across CV combos; ranked on validation "
                 "val_<select_metric>_mean only (a candidate without one "
                 "stops the job)",
    # How Step 2 ranked (config.conf SELECTION_MODE; a selection without the
    # key is global) and the selection split every run of this job trained
    # with (SELECTION_FRAC / SELECTION_SEED; 0 = none).
    "selection_mode": SELECTION_MODE,
    "selection_frac": SELECTION_FRAC,
    "selection_seed": SELECTION_SEED if SELECTION_FRAC else None,
}

if PER_FOLD:
    selection["_selection_meta"].update({
        "selection_metric": FOLD_KEY,
        "selection_label": SEL_LABEL,
        # every (CV, fold) on its own selection split of its training pairs
        "selection_basis": "selection_split",
        "num_folds": NUM_FOLDS,
        # {cv: {fold: sha256 of the held-out pairs}}, the same in every run
        # ranked (and in every combo / ablation trained on these folds)
        "sel_digests": sel_digests,
        "rationale": "per (CV, fold): each category's argmax of "
                     "fold_metrics[k][sel_<select_metric>] on fold k's own "
                     "selection split (a random SELECTION_FRAC of its training "
                     "pairs, never trained on); each CV's combo trains on the "
                     "union of its folds' winners, fold k on its own "
                     "(--fold_modalities). selection_cv is not used.",
    })

if not PER_FOLD:
    # Print summary. NOTE: the JSON "aupr" fields above hold the VAL_KEY value
    # (val_<select_metric>_mean, val_auprg_any_mean by default, in both modes); the
    # field name is kept "aupr" for backward-compatible reads, while the printed
    # label (SEL_LABEL) reflects the real metric.
    print(f"\nSelected by {SELECTION_CV.upper()} VALIDATION {SEL_LABEL} "
          f"($(pca_step1_label "$PCA_ARM"), {step1_dir}/, dir tag pcavar{PCA_VARIANCE}):")
    print(f"  {'Category':15s} -> {'Type':15s}  " +
          "  ".join(f"val_{SEL_LABEL}_{cv}" for cv in cv_types))
    for cat in sorted(ref_winners):
        parts = [f"  {cat:15s} -> {ref_winners[cat]['type']:15s}"]
        for cv in cv_types:
            cv_score = val_score(ref_winners[cat]['type'], cv)
            parts.append(f"{cv_score:.4f}" if cv_score is not None else "  N/A ")
        print("  ".join(parts))

else:
    print(f"\nSelected per (CV, fold) on each fold's selection split, "
          f"{SEL_LABEL} ({FOLD_KEY}; $(pca_step1_label "$PCA_ARM"), "
          f"{step1_dir}/, dir tag pcavar{PCA_VARIANCE}; SELECTION_FRAC="
          f"{SELECTION_FRAC:g}, SELECTION_SEED={SELECTION_SEED}):")
    for cv in cv_types:
        print(f"  {cv}:  " + "  ".join(f"{'fold ' + str(k):22s}"
                                       for k in range(NUM_FOLDS)))
        for cat, info in sorted(selection[cv].items()):
            cells = [f"{w['type'] + ' (' + format(w['score'], '.4f') + ')':22s}"
                     for _k, w in sorted(info["per_fold"].items(),
                                         key=lambda kv: int(kv[0]))]
            print(f"    {cat:15s} " + "  ".join(cells))
        print(f"    picked: " + ", ".join(
            f"{e} x{n}" for e, n in _fs.pick_counts(selection, cv).items()))

out = results_dir / "best_per_category.json"
with open(out, "w") as _f:
    json.dump(selection, _f, indent=2)
print(f"\nSaved to {out}")
PYSELECT

[ -f "$RESULTS_ROOT/best_per_category.json" ] || { echo "ERROR: Selection failed"; exit 1; }
echo ""

# Selection succeeded: now delete the previous combos of the variants this job
# trains (moved aside at the top, see there), including any an earlier job left
# aside when its own Step 2 failed; those are superseded by this selection too.
# The baseline's prefix is "", and superseded__best_cat_* never matches a
# superseded__var_* entry, so other variants' leftovers are untouched.
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    for d in "$RESULTS_ROOT"/_superseded_combos/*/superseded__${VPREFIX}best_cat_*; do
        [ -d "$d" ] && rm -rf "$d"
    done
done
for d in "$RESULTS_ROOT"/_superseded_combos/*; do
    [ -d "$d" ] && rmdir "$d" 2>/dev/null || true
done
rmdir "$RESULTS_ROOT/_superseded_combos" 2>/dev/null || true
echo "Deleted previous combos (${SIAMESE_VARIANTS[*]})"
echo ""

# What was computed from the combos replaced above, outside their directories
# (eval_finetuning_<cv>/ and pair_predictions/ went with them): the comparison
# rows and their selected-run markers, the *_nocell rows and the cell-line
# ablation dirs, the ablations and their summaries, and the tables built from
# all of them. A retrained combo keeps its directory name (the winners fix
# it), so every path-based check downstream still matches and paper_tables
# would print the PREVIOUS combos' numbers under PAPER COMPLETE. Both
# orchestrators remove these at submission (then this finds nothing); a combo
# job submitted by hand has to do it here. Rows are moved to the arm's
# stale_siamese/, as everywhere else. All in the job's PCA arm (RESULTS_ROOT /
# CMP_ROOT), except the cross-arm table sl_comparison/results/pca_arms/: it
# compares every arm's combos, so any arm's replaced combos make it stale.
_CMP_RES="$BASE_DIR/$CMP_ROOT"
_STALE_TS=$(date +%Y%m%dT%H%M%S)
_N_STALE=0
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    VTAG=$(siamese_variant_tag "$VARIANT")
    for d in "$RESULTS_ROOT"/${VPREFIX}cellline_ablation_cv[123] \
             "$RESULTS_ROOT"/${VPREFIX}ablate_no_*; do
        [ -e "$d" ] || continue
        rm -rf -- "$d"
        _N_STALE=$((_N_STALE + 1))
    done
    for f in "$_CMP_RES"/siamese_sl${VTAG:+_$VTAG}_cv[123].json \
             "$_CMP_RES"/siamese_sl${VTAG:+_$VTAG}_nocell_cv[123].json \
             "$_CMP_RES"/siamese${VTAG:+_sl_$VTAG}_selected_cv[123].txt; do
        [ -f "$f" ] || continue
        mkdir -p "$_CMP_RES/stale_siamese"
        _b=$(basename "$f")
        mv -f "$f" "$_CMP_RES/stale_siamese/${_b%.*}.$_STALE_TS.${_b##*.}"
        _N_STALE=$((_N_STALE + 1))
    done
done
for d in "$RESULTS_ROOT/ablation_summary.csv" \
         "$_CMP_RES/comparison.csv" "$_CMP_RES/comparison_notes.txt" \
         "$_CMP_RES/external_transfer" "$_CMP_RES/paper_tables" \
         "$BASE_DIR/sl_comparison/results/pca_arms"; do
    [ -e "$d" ] || continue
    rm -rf -- "$d"
    _N_STALE=$((_N_STALE + 1))
done
if [ "$_N_STALE" -gt 0 ]; then
    echo "Removed $_N_STALE output(s) computed from the replaced combos (PCA arm"
    echo "  $PCA_ARM: comparison rows -> ${CMP_ROOT}/stale_siamese/, cell-line"
    echo "  ablation and ablation dirs, comparison.csv, external_transfer/,"
    echo "  paper_tables/; the cross-arm sl_comparison/results/pca_arms/). After"
    echo "  this job, rerun what the orchestrator chains, with PCA_ARM=$PCA_ARM exported"
    echo "  (the exact lines: sl_comparison/README.md, \"After a combo job submitted by"
    echo "  hand\"): from siamese_sl/ slurm/eval_finetuning.sh, the ablation"
    echo "  (PCA_ARMS=$PCA_ARM bash slurm/submit_pipeline.sh --ablation-only: without"
    echo "  PCA_ARMS it ablates every arm, and an exported PCA_ARM is ignored there) and"
    echo "  cell_line_ablation/run_cell_line_ablation.sh; from the repo root"
    echo "  sl_comparison/slurm/run_compare.sh, run_external_transfer.sh and"
    echo "  run_paper_tables.sh, then the cross-arm table, PCA_ARMS=\"$PCA_ARMS\" PCA_ARM=${PCA_ARMS%% *}"
    echo "  sbatch --export=ALL sl_comparison/slurm/run_paper_tables.sh --pca_arms_table"
    echo "  (PCA_ARMS = the arms to compare); last, from siamese_sl/ again,"
    echo "  slurm/export_pair_predictions.sh."
    echo ""
fi

# ============================================================================
# STEP 3: Train multi-modal combo for each (variant, CV)
# ============================================================================
# For each variant and CV type, concatenate the best embedding from every
# category in the selection. The modality list is identical across variants
# (one selection, Step 2); only the variant's --kernel/--cell_lines differ.

echo "--- STEP 3: Train multi-modal combo for each variant x CV ---"
echo ""

STEP3_SUCCESSFUL=0
STEP3_SKIPPED=0
STEP3_FAILED=0
# "VARIANT/CV" of every failed combo. Only a run where no combo trained fails
# the job (see the end; Step 2 failing already exited above). A failed
# SELECTION_VARIANT / baseline pd:per_head combo is also counted in
# STEP3_SELECTION_FAILED and logged as an ERROR, but the job still exits 0:
# like every other variant's, it is guarded downstream by its missing marker.
STEP3_FAILED_RUNS=()
STEP3_SELECTION_FAILED=0

# Variant-major, so the baseline's three CVs finish first. Flattened into one
# loop so the body below keeps its indentation.
COMBO_RUNS=()
for VARIANT in "${SIAMESE_VARIANTS[@]}"; do
    for CV in "${CV_TYPES[@]}"; do
        COMBO_RUNS+=("$VARIANT|$CV")
    done
done

for RUN in "${COMBO_RUNS[@]}"; do
    IFS='|' read -r VARIANT CV <<< "$RUN"
    VPREFIX=$(siamese_variant_prefix "$VARIANT")
    VARIANT_ARGS=$(siamese_variant_args "$VARIANT")
    echo "============================================================================"
    echo "Training best-per-category combo for $CV  [variant $VARIANT]"
    echo "============================================================================"

    # The combo's modalities, from the selection JSON (fold_selection.py
    # combo_spec): the categories sorted, one winner each (SELECTION_MODE
    # global), or the union of the CV's per-fold winners with FOLD_MAP
    # mapping each fold to its own (per_fold: train.py --fold_modalities).
    # Fields separated by | so spaces within EMB_PATHS are preserved.
    # We capture stdout AND exit code separately so a crash (corrupt
    # selection JSON, missing CV key) is treated as a hard failure instead of
    # silently producing empty fields that look like "0 modalities". Every
    # winner trained in this run's Step 1, so its file existed then: a combo
    # without one (--data_dir check) is not the selected model.
    set +e
    EXTRACT_OUT=$($PYTHON_PATH fold_selection.py combo_spec \
        --selection "$RESULTS_ROOT/best_per_category.json" --cv "$CV" \
        --data_dir "$DATA_DIR" 2>&1)
    EXTRACT_EXIT=$?
    set -e

    if [ $EXTRACT_EXIT -ne 0 ] || [ -z "$EXTRACT_OUT" ]; then
        echo "  ERROR: PYEXTRACT failed (exit=$EXTRACT_EXIT). Skipping this CV."
        echo "  Output captured: $EXTRACT_OUT"
        STEP3_FAILED=$((STEP3_FAILED + 1))
        STEP3_FAILED_RUNS+=("$VARIANT/$CV")
        if [ "$VARIANT" = "$SELECTION_VARIANT" ]; then
            STEP3_SELECTION_FAILED=$((STEP3_SELECTION_FAILED + 1))
        fi
        continue
    fi
    # NUM_MODALITIES: what each fold trains on; NUM_UNION: the files listed
    # (equal in the global mode); FOLD_MAP: empty in the global mode.
    IFS='|' read -r EMB_PATHS COMBO_NAME NUM_MODALITIES NUM_UNION FOLD_MAP <<< "$EXTRACT_OUT"

    if [ "${NUM_MODALITIES:-0}" -lt 2 ]; then
        echo "  SKIP: fewer than 2 modalities available"
        STEP3_SKIPPED=$((STEP3_SKIPPED + 1))
        continue
    fi

    SAFE_NAME=$(echo "$COMBO_NAME" | tr '+' '_')
    OUTPUT_DIR="$RESULTS_ROOT/${VPREFIX}best_cat_${SAFE_NAME}_${CV}"

    echo "  Variant: $VARIANT  ($VARIANT_ARGS)"
    echo "  ARCH_ARGS: ${ARCH_ARGS:-(none; train.py defaults)}"
    if [ -n "$FOLD_MAP" ]; then
        echo "  Modalities (union of the per-fold winners, $NUM_UNION; $NUM_MODALITIES per fold): $COMBO_NAME"
        echo "  Per-fold modalities (positions in that list): $FOLD_MAP"
    else
        echo "  Modalities ($NUM_MODALITIES): $COMBO_NAME"
    fi
    echo "  Selection split: ${SELECTION_ARGS:-(off)}"
    echo "  PCA arm: $PCA_ARM ($(pca_arm_label "$PCA_ARM")): $PCA_MULTI_ARGS"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # Create output directory. Under set -e a bare failure here (quota, EIO, a
    # file in the way) would end the job mid-loop with exit 1; count it as this
    # combo's failure instead, like every other Step-3 failure path.
    if ! $PYTHON_PATH -c "from pathlib import Path; Path('$OUTPUT_DIR/checkpoints').mkdir(parents=True, exist_ok=True)"; then
        echo "  FAILED: could not create $OUTPUT_DIR/checkpoints"
        STEP3_FAILED=$((STEP3_FAILED + 1))
        STEP3_FAILED_RUNS+=("$VARIANT/$CV")
        if [ "$VARIANT" = "$SELECTION_VARIANT" ]; then
            STEP3_SELECTION_FAILED=$((STEP3_SELECTION_FAILED + 1))
        fi
        echo ""
        continue
    fi

    # The arm's PCA (config.conf pca_multi_args): arm none passes --no_post_pca
    # and the encoder reads the concatenated width; a variance target resolves
    # k per fold (a different k under `plain` than under `robust`/`svd`).
    set +e
    $PYTHON_PATH train.py \
        --embeddings_paths $EMB_PATHS \
        --sl_path "$SL_PATH" \
        ${NEG_PATH:+--neg_pairs_path "$NEG_PATH"} \
        $CELL_LINE_ARGS \
        --output_dir "$OUTPUT_DIR" \
        --cv_type "$CV" \
        $MODEL_ARGS \
        --encoder_dims $ENCODER_DIMS \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --l1_lambdas $L1_LAMBDAS \
        --pd_epsilon $PD_EPSILON \
        --eval_interval $EVAL_INTERVAL \
        --patience $PATIENCE \
        --warmrestart_T0 $WARMRESTART_T0 \
        --warmrestart_Tmult $WARMRESTART_TMULT \
        --num_folds $NUM_FOLDS \
        --pos_neg_ratio $POS_NEG_RATIO \
        --seed $SEED \
        --preprocessing_fit_scope "${PREPROCESSING_FIT_SCOPE:-train}" \
        --pca_method "${PCA_METHOD:-plain}" \
        --siamese_encoder_type "${SIAMESE_ENCODER_TYPE:-residual}" \
        $PCA_MULTI_ARGS \
        $VARIANT_ARGS \
        ${FOLD_MAP:+--fold_modalities "$FOLD_MAP"} \
        $SELECTION_ARGS \
        $ARCH_ARGS
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  FAILED (exit code $TRAIN_EXIT)"
        STEP3_FAILED=$((STEP3_FAILED + 1))
        STEP3_FAILED_RUNS+=("$VARIANT/$CV")
        if [ "$VARIANT" = "$SELECTION_VARIANT" ]; then
            STEP3_SELECTION_FAILED=$((STEP3_SELECTION_FAILED + 1))
        fi
    elif [ -f "$OUTPUT_DIR/results.json" ]; then
        AUROC=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d['auroc_mean']; s=d['auroc_std']; print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        AUPR=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d['aupr_mean']; s=d['aupr_std']; print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        # auprg_mean (stratified; equals auprg_any with one output) is written
        # in every mode; .get() -> 'N/A' only for an older results.json.
        AUPRG=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); v=d.get('auprg_mean'); s=d.get('auprg_std'); print(f'{v:.4f} +/- {s:.4f}' if v is not None else 'N/A')" 2>/dev/null || echo "N/A")
        PARAMS=$($PYTHON_PATH -c "import json; f=open('$OUTPUT_DIR/results.json'); d=json.load(f)['summary']; f.close(); print(f\"{d.get('nonzero_params',0):,}/{d.get('total_params',0):,} ({d.get('weight_sparsity',0):.1f}% sparse)\")" 2>/dev/null || echo "N/A")
        echo "  AUROC=$AUROC  AUPR=$AUPR  AUPRG=$AUPRG  Params=$PARAMS"
        STEP3_SUCCESSFUL=$((STEP3_SUCCESSFUL + 1))
        # Drop a marker file so the chained eval_finetuning job can locate
        # this freshly-trained combo. Per-CV so each eval can target the
        # matching split, and per variant (prefix "" for the baseline, so its
        # marker keeps the historical name <arm root>/latest_combo_<cv>.txt).
        echo "$OUTPUT_DIR" > "$RESULTS_ROOT/${VPREFIX}latest_combo_${CV}.txt"
    else
        echo "  FAILED"
        STEP3_FAILED=$((STEP3_FAILED + 1))
        STEP3_FAILED_RUNS+=("$VARIANT/$CV")
        if [ "$VARIANT" = "$SELECTION_VARIANT" ]; then
            STEP3_SELECTION_FAILED=$((STEP3_SELECTION_FAILED + 1))
        fi
    fi
    set -e

    echo ""
done

# ============================================================================
# STEP 4: Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "                 BEST-PER-CATEGORY RESULTS"
echo "  PCA arm $PCA_ARM: $(pca_arm_label "$PCA_ARM")"
echo "============================================================================"

# [variant, results-dir prefix] pairs, so the summary lists every variant.
VARIANTS_PY="[$(for v in "${SIAMESE_VARIANTS[@]}"; do printf '["%s", "%s"], ' "$v" "$(siamese_variant_prefix "$v")"; done | sed 's/, $//')]"

# Informational only: never let it change the exit status (exit 1 here would
# cancel every variant's eval / ablation, see the end).
$PYTHON_PATH << PYSUMMARY || echo "WARNING: Step 4 summary failed (informational only; exit status unaffected)"
import json
from pathlib import Path

PCA_VARIANCE = "$PCA_VARIANCE"
VARIANTS = $VARIANTS_PY  # [variant, prefix] from SIAMESE_VARIANTS in config.conf
RESULTS = Path("$RESULTS_ROOT")  # the arm's results root (config.conf)
with open(RESULTS / "best_per_category.json") as _f:
    sel = json.load(_f)
cv_types = $CV_TYPES_PY  # Injected from CV_TYPES (config.conf; overridable in run_best_per_category.conf)
_meta = sel.get("_selection_meta", {})
selection_cv = _meta.get("selection_cv", "unknown")
# Label the winners table by the metric actually used (selection_label,
# AUPRG_any under the default --select_metric). The per-entry field is still
# named "aupr" for backward compat, but it holds the VAL_KEY value
# (val_<select_metric>_mean), so display the true metric name here.
# Read the label PYSELECT recorded; fall back to sniffing the metric key for
# selection JSONs written before selection_label existed. The fallback matches
# on a SUBSTRING because the key is "val_auprg_mean" whenever a validation
# split is present — testing equality against "auprg_mean" silently mislabels
# every default-path run.
SEL_LABEL = _meta.get("selection_label") or (
    "AUPRG" if "auprg" in str(_meta.get("selection_metric", "")) else "AUPR")
SEL_BASIS = _meta.get("selection_basis", "unknown")

# Step 1 results: selected winner's score across CVs.
# After the single SELECTION_CV selection, the same winner is chosen for every CV, so the
# table shows "winner_type (score_on_this_cv)" — column-to-column differences
# reveal how the SELECTION_CV-picked winner generalizes.
col_w = 15
header = " | ".join(f"{cv.upper():{col_w}s}" for cv in cv_types)
table_w = col_w + len(cv_types) * (3 + col_w)

print(f"\n--- Single-embedding {SEL_BASIS.upper()} {SEL_LABEL} of selected winners "
      f"(the selection score, not test; the combo table below is TEST) ---")
print(f"    (selection_cv={selection_cv}, {_meta.get('step1_pca', 'native dim')}, "
      f"dir tag pcavar{PCA_VARIANCE}, "
      f"selection_variant={_meta.get('selection_variant', 'pd:per_head')})")
print(f"{'Category':{col_w}s} | {header}")
print("-" * table_w)
cats = sorted({c for cv in cv_types if cv in sel for c in sel[cv]})
for cat in cats:
    row = []
    for cv in cv_types:
        if cv in sel and cat in sel[cv]:
            entry = sel[cv][cat]
            aupr_str = (f"{entry['aupr']:.3f}"
                        if entry['aupr'] is not None else "N/A")
            row.append(f"{entry['type']} ({aupr_str})")
        else:
            row.append("N/A")
    row_str = " | ".join(f"{cell:{col_w}s}" for cell in row)
    print(f"{cat:{col_w}s} | {row_str}")
if _meta.get("selection_mode") == "per_fold":
    # The table above shows each category's most-picked embedding and the
    # mean winning selection score; every fold trained on its own winners.
    print("  (SELECTION_MODE=per_fold: the most-picked embedding per category "
          "and the mean of the folds' winning selection scores; each fold's "
          "own winners:)")
    for cv in cv_types:
        for cat in sorted(sel.get(cv, {})):
            per = sel[cv][cat].get("per_fold") or {}
            print(f"    {cv} {cat:15s} " + "  ".join(
                f"fold {k}: {w['type']}" for k, w in
                sorted(per.items(), key=lambda kv: int(kv[0]))))

# Step 3 results: multi-modal combo (test metrics; AUPRG columns absent in an
# older results.json -> rendered N/A via .get()).
# One block per variant; a (variant, cv) row is the combo its
# <arm root>/<prefix>latest_combo_<cv>.txt marker names, as for every consumer: a
# failed combo keeps its directory (maybe with a results.json) but no marker.
# AUPRG (stratified; the checkpoint metric only under --select_metric auprg) is
# pooled over (pair, head) cells for per_head but over pairs for pooled, so it
# is NOT comparable across the cell_lines axis. AUPRG_any (OR label, max over
# heads; the default --select_metric) is the same quantity for every variant —
# compare variants on that column.
print(f"\n--- Multi-modal combo TEST results (same modalities for every variant) ---")
hdr = (f"{'CV':15s} | {'AUROC':25s} | {'AUPR':25s} | {'AUPRG':25s} | "
       f"{'AUPRG_any':25s} | {'Params (nonzero/total)':25s} | {'Sparsity':10s}")
for variant, prefix in VARIANTS:
    print(f"\n[{variant}]  {RESULTS}/{prefix}latest_combo_<cv>.txt")
    print(hdr)
    print("-" * len(hdr))
    for cv in cv_types:
        marker = RESULTS / f"{prefix}latest_combo_{cv}.txt"
        if not marker.is_file():
            print(f"{cv:15s} | {'NOT TRAINED (no marker)':25s} | {'N/A':25s} | "
                  f"{'N/A':25s} | {'N/A':25s} | {'N/A':25s} | {'N/A':10s}")
            continue
        rf = Path(marker.read_text().strip()) / "results.json"
        try:
            with open(rf) as _f:
                r = json.load(_f)["summary"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"{cv:15s} | unreadable {rf}: {type(e).__name__}: {e}")
            continue
        def ms(key):
            m, s = r.get(f"{key}_mean"), r.get(f"{key}_std")
            return f"{m:.4f} +/- {s:.4f}" if m is not None else "N/A"
        nz = r.get('nonzero_params') or 0
        tot = r.get('total_params') or 0
        sp = r.get('weight_sparsity') or 0
        params = f"{nz:,}/{tot:,}"
        sparsity = f"{sp:.1f}%"
        print(f"{cv:15s} | {ms('auroc'):25s} | {ms('aupr'):25s} | "
              f"{ms('auprg'):25s} | {ms('auprg_any'):25s} | "
              f"{params:25s} | {sparsity:10s}")
PYSUMMARY

echo ""
echo "============================================================================"
echo "Step 3 (multi-modal): $STEP3_SUCCESSFUL ok, $STEP3_SKIPPED skip, $STEP3_FAILED fail"
echo "Completed at: $(date)"
echo "============================================================================"

# Exit non-zero only when NOTHING trained (Step 2 selection failing already
# exited above), whether the combos failed or were all skipped for having < 2
# modalities (skips are all-or-nothing: every variant and CV reuses the same
# winners and files): there is nothing downstream to run.
#
# Any single failed combo, of ANY variant (the SELECTION_VARIANT and the
# baseline pd:per_head siamese_sl row included), is logged loudly and the job
# still exits 0. Exiting 1 would cancel, through afterok +
# --kill-on-invalid-dep, the eval, ablation and cell-line ablation of EVERY
# variant and CV, i.e. tab:external, tab:external-ft, tab:loco and
# tab:cellline, for one failed cell. Each consumer is guarded per (variant,
# CV) instead: the failed combo wrote no <arm root>/<prefix>latest_combo_<cv>.txt
# marker (cleaned at the top), so the sl_comparison collector writes no row for
# it (collect_siamese.py --combo_markers), that CV's ablation tasks skip for
# want of the marked reference combo (run_ablation.sh),
# the cell-line ablation names it MISSING, and, only when the failed CV is
# cv3, the variant's eval tasks stop with "MODEL_DIR not found ... did this
# variant's cv3 combo train?" (the eval reads only the cv3 marker,
# eval_finetuning.conf). paper_tables.py lists every absent row.
if [ "$SELECTION_VARIANT" != "pd:per_head" ]; then
    for _r in "${STEP3_FAILED_RUNS[@]}"; do
        case "$_r" in
            pd:per_head/*) STEP3_SELECTION_FAILED=$((STEP3_SELECTION_FAILED + 1)) ;;
        esac
    done
fi
if [ $STEP3_FAILED -gt 0 ]; then
    echo ""
    echo "Failed combos ($STEP3_FAILED): ${STEP3_FAILED_RUNS[*]}"
fi
if [ $STEP3_SUCCESSFUL -eq 0 ]; then
    echo "ERROR: no combo trained successfully ($STEP3_SKIPPED skipped, $STEP3_FAILED failed); nothing for eval / ablation."
    exit 1
fi
if [ $STEP3_SELECTION_FAILED -gt 0 ]; then
    echo "ERROR: $STEP3_SELECTION_FAILED combo(s) of the selection variant"
    echo "       $SELECTION_VARIANT / baseline pd:per_head (the siamese_sl row) FAILED."
    echo "       Exiting 0 anyway so every other (variant, CV) still gets its eval,"
    echo "       ablations and cell-line ablation; the failed cells have no marker, so"
    echo "       they have no row, eval, ablation or _nocell output, and the paper"
    echo "       tables list them as missing. Retrain them before publishing."
fi
if [ $STEP3_FAILED -gt 0 ]; then
    echo "WARNING: $STEP3_FAILED combo(s) failed; exiting 0 so the other"
    echo "         combos' eval and ablation still run. Each failed VARIANT/CV"
    echo "         listed above has no $RESULTS_ROOT/<prefix>latest_combo_<cv>.txt"
    echo "         marker: sl_comparison writes no row for it and that CV's"
    echo "         ablation tasks skip. The variant's external-screen eval reads"
    echo "         only its cv3 combo, so its eval tasks stop only if cv3 failed."
    echo "         Resubmitting this job retrains EVERY CV of its variants under the"
    echo "         same names. Resubmit it only once none of these jobs is queued or"
    echo "         running (squeue; wait for them or scancel them): allmod_combo,"
    echo "         bilinear_probe, eval_finetuning,"
    echo "         ablation, ablation_summary, cellline_ablation, pair_export,"
    echo "         sl_final_compare, sl_ext_transfer, sl_paper_tables (of any PCA arm"
    echo "         for the last one). Resubmit it from siamese_sl/ with"
    echo "         RUN_TOKEN=$RUN_TOKEN PCA_ARM=$PCA_ARM PCA_ARMS=\"$PCA_ARMS\" SELECTION_CV=$SELECTION_CV SELECTION_FRAC=$SELECTION_FRAC SELECTION_SEED=$SELECTION_SEED SELECTION_MODE=$SELECTION_MODE"
    echo "         (and the run's other PCA_* and ARCH_ARGS settings) exported:"
    echo "         PCA_ARMS with PCA_ARM decides whose single-embedding runs Step 2"
    echo "         ranks, SELECTION_CV the CV it ranks them on (in no signature:"
    echo "         given another, the job selects on that one and nothing stops it)."
    echo "         The job does not check: as it starts it removes"
    echo "         the selection and its variants' markers and combos, then their"
    echo "         ablation dirs and rows, under whichever of those jobs is running."
    echo "         Then rerun what was computed from the replaced combos"
    echo "         (sl_comparison/README.md, \"After a combo job submitted by hand\")."
fi
