#!/bin/bash
# Sourced (not executed), with cwd = repo root, by every sl_comparison SLURM
# script to choose the interpreter, so these jobs run the SAME python as the
# siamese jobs:
#   1. PYTHON_PATH already in the environment (the root orchestrator exports
#      config.conf's value; a hand-run `PYTHON_PATH=... sbatch ...` also wins);
#   2. else siamese_sl/slurm/config.conf's PYTHON_PATH, the one every siamese
#      job sources, so moving the venv means editing that one line;
#   3. else the cluster venv, as before.
# config.conf is read in a bash subshell with SLURM_JOB_ID unset, so its
# compute-node block (module purge, cd, banner) stays inert.

if [ -z "${PYTHON_PATH:-}" ] && [ -f siamese_sl/slurm/config.conf ]; then
    PYTHON_PATH=$(bash -c '
        unset SLURM_JOB_ID
        source "$1" >/dev/null 2>&1
        printf "%s" "${PYTHON_PATH:-}"
    ' _ siamese_sl/slurm/config.conf 2>/dev/null) || PYTHON_PATH=""
fi
PYTHON_PATH="${PYTHON_PATH:-<home>/pytorch/bin/python3}"
echo "  python: $PYTHON_PATH"
