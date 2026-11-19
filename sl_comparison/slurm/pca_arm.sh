#!/bin/bash
# Sourced (not executed), with cwd = repo root, by the sl_comparison jobs that
# read or write ONE PCA arm's outputs (run_compare.sh, run_external_transfer.sh,
# run_paper_tables.sh), right after python_path.sh. The arm is PCA_ARM:
# the orchestrators submit each arm's jobs as `PCA_ARM=<arm> sbatch ...
# --export=ALL` (the arm in sbatch's environment), and a job submitted by hand
# is arm none unless PCA_ARM is exported or passed that way. It is
# resolved by siamese_sl/slurm/config.conf, the file every siamese job of that
# arm sources, so the two sides cannot disagree on where an arm lives. Sets:
#   SLEUTH_ARM        none | joint | per_modality | both (config.conf PCA_ARM)
#   SLEUTH_RESULTS    the arm's siamese results root from the repo root:
#                     siamese_sl/results (arm none), else
#                     siamese_sl/results/pca_<arm> (siamese_sl/$RESULTS_ROOT)
#   SLEUTH_CMP        the arm's comparison root: sl_comparison/results (arm
#                     none), else sl_comparison/results/pca_<arm> (CMP_ROOT).
#                     The shared models' rows (SLGNN / NSF4SL / SLMGAE /
#                     DegreePrior) stay in sl_comparison/results for every arm.
#   SLEUTH_PCA_ARMS   the run's arms, in config.conf's order (PCA_ARMS)
#   SLEUTH_PCA_LABEL  what the arm trains, one line (pca_arm_label)
# Not exported: the jobs use them in the shell that sources this (and that
# shell's sourced collect_and_regrade_siamese.sh). Not SIAMESE_RESULTS_*: the
# root orchestrator unsets every variable with that prefix (the hand pins of
# collect_and_regrade_siamese.sh).
# config.conf is read in a bash subshell with SLURM_JOB_ID unset, so its
# compute-node block (module purge, cd, banner) stays inert. When it refuses
# (a bad PCA_* value, POST_PCA_*, or any other setting it rejects), its
# messages are printed and this returns 1, with none of the variables above
# set, before the caller has changed anything: the arm decides which
# directories a job clears and rewrites, so it is never guessed. The callers
# stop on that (run_compare.sh under set -e, the others with `|| exit 1`).
# Also sourced by hand, by bash (3.2 included) or zsh, before
# collect_and_regrade_siamese.sh: hence no `local`, no ${!x}, every expansion
# quoted (zsh does not word-split).

unset SLEUTH_ARM SLEUTH_RESULTS SLEUTH_CMP SLEUTH_PCA_ARMS SLEUTH_PCA_LABEL
if [ ! -f siamese_sl/slurm/config.conf ]; then
    echo "ERROR: no siamese_sl/slurm/config.conf under $(pwd): the PCA arm's"
    echo "       directories cannot be resolved (run from the repo root)."
    return 1
fi
# One line, PCA_ARM_OK|<arm>|<RESULTS_ROOT>|<CMP_ROOT>|<PCA_ARMS>|<label>,
# after whatever config.conf wrote to stderr (2>&1 >/dev/null: its stderr is
# kept, its stdout dropped). No field holds a '|'.
_sleuth_rc=0
_sleuth_out=$(bash -c '
    unset SLURM_JOB_ID
    source "$1" 2>&1 >/dev/null || exit 1
    printf "PCA_ARM_OK|%s|%s|%s|%s|%s\n" "$PCA_ARM" "$RESULTS_ROOT" "$CMP_ROOT" \
        "$PCA_ARMS" "$(pca_arm_label "$PCA_ARM")"
' _ siamese_sl/slurm/config.conf 2>&1) || _sleuth_rc=$?
_sleuth_line=""
if [ "$_sleuth_rc" = 0 ]; then
    _sleuth_line=$(printf '%s\n' "$_sleuth_out" | sed -n 's/^PCA_ARM_OK|//p' | tail -n 1)
fi
if [ -z "$_sleuth_line" ]; then
    echo "ERROR: siamese_sl/slurm/config.conf refused this job's settings, so the"
    echo "       job's PCA arm (PCA_ARM=${PCA_ARM:-unset, i.e. none}) and its"
    echo "       directories are unknown; nothing was changed. config.conf said:"
    printf '%s\n' "${_sleuth_out:-(nothing)}" | sed 's/^/         /'
    unset _sleuth_rc _sleuth_out _sleuth_line
    return 1
fi
IFS='|' read -r _sleuth_arm _sleuth_root _sleuth_cmp _sleuth_arms _sleuth_label \
    <<< "$_sleuth_line"
unset _sleuth_rc _sleuth_out _sleuth_line
# The layout every reader assumes (paper_tables.py, the e2e verifiers, the
# orchestrators' cleanup): arm none at the top, arm X one directory down. A
# config.conf that resolves anything else (an older copy, without the PCA
# arms, leaves both roots empty) is refused, not followed.
case "$_sleuth_arm" in
    none) _sleuth_want="results|sl_comparison/results" ;;
    joint|per_modality|both)
        _sleuth_want="results/pca_$_sleuth_arm|sl_comparison/results/pca_$_sleuth_arm" ;;
    *) _sleuth_want="an arm of none, joint, per_modality or both" ;;
esac
if [ "$_sleuth_root|$_sleuth_cmp" != "$_sleuth_want" ]; then
    echo "ERROR: siamese_sl/slurm/config.conf resolves PCA arm '$_sleuth_arm' to"
    echo "       RESULTS_ROOT='$_sleuth_root' CMP_ROOT='$_sleuth_cmp', not"
    echo "       $_sleuth_want (an older config.conf?); nothing was changed."
    unset _sleuth_arm _sleuth_root _sleuth_cmp _sleuth_arms _sleuth_label _sleuth_want
    return 1
fi
# A PCA_ARM this shell has but did not export (a hand session) never reached
# config.conf's subshell, which then resolved arm none: refuse rather than
# collect, grade or tabulate arm none under the arm that was asked for.
if [ -n "${PCA_ARM:-}" ] && [ "$PCA_ARM" != "$_sleuth_arm" ]; then
    echo "ERROR: PCA_ARM=$PCA_ARM in this shell, but config.conf (read in a bash"
    echo "       subshell) resolved arm $_sleuth_arm: export PCA_ARM. Nothing was"
    echo "       changed."
    unset _sleuth_arm _sleuth_root _sleuth_cmp _sleuth_arms _sleuth_label _sleuth_want
    return 1
fi
SLEUTH_ARM="$_sleuth_arm"
SLEUTH_RESULTS="siamese_sl/$_sleuth_root"
SLEUTH_CMP="$_sleuth_cmp"
# shellcheck disable=SC2034  # read by the sourcing job (run_paper_tables.sh)
SLEUTH_PCA_ARMS="$_sleuth_arms"
SLEUTH_PCA_LABEL="$_sleuth_label"
unset _sleuth_arm _sleuth_root _sleuth_cmp _sleuth_arms _sleuth_label _sleuth_want
echo "  PCA arm: $SLEUTH_ARM ($SLEUTH_PCA_LABEL): $SLEUTH_RESULTS/, $SLEUTH_CMP/"
