#!/usr/bin/env python3
"""sha256 manifest of the code a pipeline run's jobs execute.

The cluster checkout is updated by file upload, not by git, and the jobs of one
submission start over days: on 2026-09-23 a whole-tree upload landed 32 min
after the submission, so the queued batch scripts were one version while every
file they sourced or imported afterwards was another, and nothing recorded it
(the graded rows' git_sha is null, since git does not run in the jobs).
run_benchmarking_siamese_vs_others.sh writes this manifest at submission
(sl_comparison/slurm/logs/code_<RUN_TOKEN>_<timestamp>.sha256, exported as
CODE_MANIFEST); paper_tables.py re-hashes the same files when it runs, after
every job a table depends on, and lists every file that changed, appeared or
vanished in between (the all-gene-pair export starts after it: not covered).

Scope: the *.py / *.sh / *.conf files under the directories the DAG's jobs run
code from (CODE_DIRS), minus tests and the login-node-only scripts (a change
to those cannot reach a running job), with output, input and cache
directories pruned.

    python sl_comparison/code_manifest.py [--root .]            # print it
    python sl_comparison/code_manifest.py --check <manifest>    # exit 1 on drift
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

CODE_DIRS = ("siamese_sl", "sl_comparison", "SLGNN", "NSF4SL",
             "SLMGAE-in-pytorch", "ELISL")
SUFFIXES = (".py", ".sh", ".conf")
# never code the jobs run: outputs, inputs, caches, tests, vendored originals
PRUNE = {"results", "results_preflight", "logs", "checkpoints", "models",
         "folds", "folds_preflight", "data", "cache", "embeddings_cache",
         "__pycache__", "tests", "original_tensorflow_code", "kg_meta_only",
         ".git"}
# run on the login node only: a change after submission reaches no job
LOGIN_ONLY = {"siamese_sl/slurm/submit_pipeline.sh", "siamese_sl/slurm/clean.sh"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_files(root="."):
    """Repo-relative paths (posix) of the files in scope, sorted."""
    root = Path(root)
    out = []
    for top in CODE_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d not in PRUNE)
            for f in fn:
                if not f.endswith(SUFFIXES) or f.startswith("test_"):
                    continue
                rel = Path(dp, f).relative_to(root).as_posix()
                if rel not in LOGIN_ONLY:
                    out.append(rel)
    return sorted(out)


def manifest(root="."):
    """{repo-relative path: sha256} over code_files(root)."""
    root = Path(root)
    return {rel: _sha256(root / rel) for rel in code_files(root)}


def read_manifest(path):
    """Parse '<sha256>  <path>' lines back into {path: sha256}."""
    rec = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                digest, rel = line.split("  ", 1)
                rec[rel] = digest
    return rec


def drift(root, manifest_path):
    """Files that changed, appeared or vanished since the manifest was written:
    sorted '<path> (changed|new|removed)' strings; [] when none."""
    was, now = read_manifest(manifest_path), manifest(root)
    out = [f"{p} (changed)" for p in sorted(set(was) & set(now))
           if was[p] != now[p]]
    out += [f"{p} (new)" for p in sorted(set(now) - set(was))]
    out += [f"{p} (removed)" for p in sorted(set(was) - set(now))]
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", metavar="MANIFEST",
                    help="compare the tree with MANIFEST; exit 1 on drift")
    args = ap.parse_args(argv)
    if args.check:
        d = drift(args.root, args.check)
        for line in d:
            print(line)
        return 1 if d else 0
    for rel, digest in manifest(args.root).items():
        print(f"{digest}  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
