#!/usr/bin/env python3
"""Refuse SLGNN / NSF4SL / SLMGAE inputs that are missing or were built by older code.

Run from the repo root, with the siamese jobs' interpreter:
  * on the login node by run_benchmarking_siamese_vs_others.sh, before any
    submission (so a bad input stops the run instead of failing a GPU task
    hours later), and
  * by run_shared_models.sh for its own model, before training.

Checks (existence is not enough: every builder below records provenance, and a
file left over from an older build passes an existence test):

  SLGNN   data/synlethkg/meta.json carries relation_checks.label_keyed_deletion
          (written by the current SLGNN/build_synlethkg.py) and mapping_sha256,
          and every file it hashed (the two SynLethDB exports, the HGNC table,
          NCBI gene_info, and build_synlethkg.py itself; repo-relative keys)
          is unchanged: another copy of one, or other builder code, attaches
          some KG genes to other Entrez ids. The two files the
          trainer loads (kg_triples.npy, entity_entrez.tsv) are there and hold
          the triple / gene-entity counts meta.json records: meta.json is
          written last, so a short one is a copy or a rebuild that stopped.
  NSF4SL  data/nsf4sl_transe_features.pt carries meta.mapping_sha256, and every
          hash it recorded (HGNC table, gene_corrections_config.py,
          gene_name_utils.py, build_transe_features.py itself) equals the
          current file's: a different
          table or normaliser attaches some KG genes to other Entrez rows, and
          NSF4SL/sl_data only checks that 'entity_mean' exists.
  SLMGAE  the authors' support views and their index exist in data/ and are
          not empty (their shape against List_Proteins, and the PPI edge
          count, are the trainer's check: train_slmgae_shared.py).
  ELISL   (opt-in: --models ELISL; not in the default set, so the root
          orchestrator's login-node check, which runs every default model, is
          unchanged; ELISL/slurm/run_elisl.sh and build_elisl_inputs.sh call
          it) data/elisl/meta.json, written last by ELISL/build_elisl_inputs.py
          features, records the two context-free sources, the sha256 of
          gene_features.npz and of every input and code file it was built
          from; each must equal the current file's.

Prints each problem with the command that rebuilds the input (or, when that
builder is itself missing from this checkout, says so instead of printing a
command that cannot run); exit 1 if any.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ("SLGNN", "NSF4SL", "SLMGAE")
# Checked only when named (--models ELISL): ELISL is trained by its own
# launcher, ELISL/slurm/run_elisl.sh, not by run_shared_models.sh.
OPT_IN = ("ELISL", )

SYNLETHKG_META = "data/synlethkg/meta.json"
TRANSE_FEATURES = "data/nsf4sl_transe_features.pt"
SLMGAE_FILES = ("Human_GOsim.txt", "Human_GOsim_CC.txt",
                "biogrid_ppi_sparse.txt", "List_Proteins_in_SL.txt")
# Hashes the current build_transe_features.py records; a file lacking one
# predates it (gene_name_utils.py came with its prev_symbol > alias change, then
# the builder's own hash: its gene-entity / collision rules move KG genes that
# no input hash sees). Every other recorded hash is compared too.
REQUIRED_MAPPING_KEYS = ("hgnc_complete_set.txt", "gene_corrections_config.py",
                         "gene_name_utils.py", "build_transe_features.py")

BUILDERS = {"SLGNN": "SLGNN/build_synlethkg.py",
            "NSF4SL": "NSF4SL/build_transe_features.py",
            "ELISL": "ELISL/build_elisl_inputs.py"}
ELISL_META = "data/elisl/meta.json"
ELISL_FEATURES = "data/elisl/gene_features.npz"
ELISL_SOURCES = ["Sequence_1024", "PPI_ec"]
# Code files the current builder records (a meta.json lacking one predates it).
ELISL_CODE_KEYS = ("ELISL/build_elisl_inputs.py", "ELISL/elisl_orig/features.py")
# The SLGNN counterpart of REQUIRED_MAPPING_KEYS. data/synlethkg is git-ignored
# and built by hand on the login node, so a directory built before
# build_synlethkg.py started hashing itself would otherwise pass forever — and
# the builder's own rules (METHODS order, resolve_entrez collision precedence)
# move KG genes to other Entrez ids with no MAPPING_FILES byte changing.
REQUIRED_KG_KEYS = (BUILDERS["SLGNN"],)


def rebuild_hint(model, root=None):
    """The command to paste on the login node, with the interpreter spelled
    out: config.conf sets PYTHON_PATH without exporting it, so a literal
    `$PYTHON_PATH` would usually expand to nothing in the user's shell."""
    # A checkout the builder never reached (never uploaded, or never committed
    # where the tree comes from) lacks the very script this hint tells the user
    # to run (run_shared_models.sh calls this check directly, without the
    # orchestrator's NEED_CODE gate). Name that instead of a dead command.
    builder = BUILDERS.get(model)
    if root is not None and builder and not (Path(root) / builder).is_file():
        return (f"{builder} is not in this checkout — upload it here (and "
                f"commit it where you edit it), then rerun")
    py = os.environ.get("PYTHON_PATH") or sys.executable
    if os.sep in py and not os.path.isabs(py):
        py = os.path.abspath(py)          # the hint cd's into a subdirectory
    py = shlex.quote(py)
    return {
        # an orchestrated run leaves data/synlethkg read-only
        "SLGNN": f"(chmod -R u+w data/synlethkg 2>/dev/null; cd SLGNN && "
                 f"{py} build_synlethkg.py)",
        "NSF4SL": f"(cd NSF4SL && {py} build_transe_features.py --download)",
        "ELISL": f"{py} ELISL/build_elisl_inputs.py download (login node), "
                 f"then sbatch --export=ALL ELISL/slurm/build_elisl_inputs.sh",
        "SLMGAE": "copy the missing file(s) from the original SLMGAE release "
        "into data/ (the two Human_GOsim*.txt files are git-ignored)",
    }[model]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_slgnn(root):
    p = root / SYNLETHKG_META
    if not p.is_file():
        return [f"{SYNLETHKG_META} missing"]
    try:
        meta = json.loads(p.read_text())
    except Exception as e:
        return [f"{SYNLETHKG_META} unreadable ({type(e).__name__}: {e})"]
    checks = meta.get("relation_checks") if isinstance(meta, dict) else None
    if not isinstance(checks, dict) or "label_keyed_deletion" not in checks:
        return [f"{SYNLETHKG_META} has no relation_checks.label_keyed_deletion "
                f"(built by an older build_synlethkg.py)"]
    rec = meta.get("mapping_sha256")
    if not isinstance(rec, dict) or not rec:
        return [f"{SYNLETHKG_META} has no mapping_sha256 (built by an older "
                f"build_synlethkg.py)"]
    errs = [f"{SYNLETHKG_META} mapping_sha256 lacks '{k}' (built by an older "
            f"build_synlethkg.py)"
            for k in REQUIRED_KG_KEYS if k not in rec]
    for rel, want in sorted(rec.items()):
        f = root / rel      # keys are repo-relative (build_synlethkg.MAPPING_FILES)
        if not f.is_file():
            # Absent is not "mapped from another": these keys include the
            # builder itself, so the honest diagnosis is a file that never
            # reached this checkout, not a provenance mismatch (which would
            # send the operator into a multi-hour KG rebuild).
            errs.append(f"{SYNLETHKG_META} records '{rel}', which is not in "
                        f"this checkout ({f}) — never uploaded here? Copy it "
                        f"in, then rerun")
            continue
        have = sha256(f)
        if have != want:
            errs.append(f"{SYNLETHKG_META} was mapped from another '{rel}' "
                        f"(recorded {str(want)[:12]}, current "
                        f"{str(have)[:12]} = {f})")
    # meta.json is only this directory's manifest: the trainer also opens
    # kg_triples.npy and entity_entrez.tsv (build_synlethkg.load_kg_for_genes),
    # and build_synlethkg writes meta.json LAST. So a hand-copied or
    # part-deleted data/synlethkg (it is git-ignored, hence copied by hand),
    # or a REBUILD interrupted after the payload was rewritten, leaves this
    # meta.json beside a missing or short payload — and every SLGNN task then
    # dies after its GPU is allocated. The counts meta.json already records
    # catch the short copy, which otherwise raises nothing: a cut
    # entity_entrez.tsv silently turns its lost genes into isolated entities.
    kg_dir = p.parent
    errs += [f"data/synlethkg/{n} missing (incomplete build or copy)"
             for n in ("kg_triples.npy", "entity_entrez.tsv")
             if not (kg_dir / n).is_file()]
    if not errs:
        errs += synlethkg_payload_errors(kg_dir, meta)
    return errs


def synlethkg_payload_errors(kg_dir, meta):
    """Where kg_triples.npy / entity_entrez.tsv disagree with meta.json."""
    errs = []
    try:
        import numpy as np
        n_trip = int(len(np.load(kg_dir / "kg_triples.npy", mmap_mode="r")))
    except Exception as e:
        errs.append(f"data/synlethkg/kg_triples.npy unreadable "
                    f"({type(e).__name__}: {e})")
    else:
        if n_trip != meta.get("n_triples"):
            errs.append(f"data/synlethkg/kg_triples.npy has {n_trip:,} triples, "
                        f"{SYNLETHKG_META} records {meta.get('n_triples')} "
                        f"(interrupted rebuild or short copy)")
    try:
        with open(kg_dir / "entity_entrez.tsv") as f:
            n_rows = sum(1 for _ in f) - 1          # minus the header
    except Exception as e:
        errs.append(f"data/synlethkg/entity_entrez.tsv unreadable "
                    f"({type(e).__name__}: {e})")
    else:
        if n_rows != meta.get("gene_entities"):
            errs.append(f"data/synlethkg/entity_entrez.tsv has {n_rows:,} gene "
                        f"entities, {SYNLETHKG_META} records "
                        f"{meta.get('gene_entities')} (interrupted rebuild or "
                        f"short copy)")
    return errs


def mapping_files(root):
    """{recorded key: current file (or None)} for build_transe_features' keys.

    gene_corrections_config.py is resolved through gene_name_utils itself
    (the tracked SLMGAE-in-pytorch/ copy; a legacy code/ copy only when that
    is absent), so the check hashes the file the builder's normaliser
    actually imported.
    """
    sys.path.insert(0, str(root / "siamese_sl"))
    with contextlib.redirect_stdout(io.StringIO()):
        import gene_name_utils
    gcf = gene_name_utils.GENE_CORRECTIONS_FILE
    return {
        "hgnc_complete_set.txt": root / "data" / "cache" /
        "hgnc_complete_set.txt",
        "gene_corrections_config.py": Path(gcf) if gcf else None,
        "gene_name_utils.py": Path(gene_name_utils.__file__),
        "build_transe_features.py": root / "NSF4SL" /
        "build_transe_features.py",
    }


def check_nsf4sl(root):
    p = root / TRANSE_FEATURES
    if not p.is_file():
        return [f"{TRANSE_FEATURES} missing"]
    try:
        import torch
        data = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        return [f"{TRANSE_FEATURES} unreadable ({type(e).__name__}: {e})"]
    meta = data.get("meta") if isinstance(data, dict) else None
    rec = meta.get("mapping_sha256") if isinstance(meta, dict) else None
    if not isinstance(rec, dict) or not rec:
        return [f"{TRANSE_FEATURES} has no meta.mapping_sha256 (built by an "
                f"older build_transe_features.py)"]
    errs = [f"{TRANSE_FEATURES} meta.mapping_sha256 lacks '{k}' (built by an "
            f"older build_transe_features.py)"
            for k in REQUIRED_MAPPING_KEYS if k not in rec]
    try:
        files = mapping_files(root)
    except Exception as e:
        return errs + [f"could not resolve the gene-mapping inputs "
                       f"({type(e).__name__}: {e})"]
    for key, want in sorted(rec.items()):
        if key not in files:
            errs.append(f"{TRANSE_FEATURES} records a hash for '{key}', which "
                        f"this check cannot locate (update "
                        f"sl_comparison/slurm/check_shared_inputs.py)")
            continue
        f = files[key]
        if f is None or not f.is_file():
            # Absent is not "built from another": these keys include code
            # files, so the honest diagnosis is a file that never reached this
            # checkout, not a provenance mismatch to hunt.
            errs.append(f"{TRANSE_FEATURES} records '{key}', which is not in "
                        f"this checkout ({f}) — never uploaded here? Copy it "
                        f"in, then rerun")
            continue
        have = sha256(f)
        if have != want:
            # Name the file hashed: which gene_corrections_config.py is
            # current depends on gene_name_utils' resolution.
            errs.append(f"{TRANSE_FEATURES} was built from another '{key}' "
                        f"(recorded {str(want)[:12]}, current "
                        f"{str(have)[:12]} = {f})")
    return errs


def check_slmgae(root):
    return [f"data/{f} missing (SLMGAE's shipped input; without it this is "
            f"not the published model)" for f in SLMGAE_FILES
            if not (root / "data" / f).is_file()] + [
            f"data/{f} is empty (an interrupted copy? SLMGAE's shipped input)"
            for f in SLMGAE_FILES
            if (root / "data" / f).is_file()
            and (root / "data" / f).stat().st_size == 0]


def check_elisl(root):
    p = root / ELISL_META
    if not p.is_file():
        return [f"{ELISL_META} missing"]
    try:
        meta = json.loads(p.read_text())
    except Exception as e:
        return [f"{ELISL_META} unreadable ({type(e).__name__}: {e})"]
    if not isinstance(meta, dict):
        return [f"{ELISL_META} is not a JSON object"]
    errs = []
    if meta.get("sources") != ELISL_SOURCES or meta.get("context_free") is not True:
        errs.append(f"{ELISL_META} records sources {meta.get('sources')!r} "
                    f"(context_free {meta.get('context_free')!r}), not the "
                    f"context-free {ELISL_SOURCES}")
    feats = root / ELISL_FEATURES
    if not feats.is_file():
        errs.append(f"{ELISL_FEATURES} missing")
    elif sha256(feats) != meta.get("gene_features_sha256"):
        errs.append(f"{ELISL_FEATURES} is not the file {ELISL_META} was "
                    f"written with (a copy or an interrupted rebuild)")
    code = meta.get("code_sha256")
    if not isinstance(code, dict):
        return errs + [f"{ELISL_META} has no code_sha256 (built by an older "
                       f"ELISL/build_elisl_inputs.py)"]
    errs += [f"{ELISL_META} code_sha256 lacks '{k}' (built by an older "
             f"ELISL/build_elisl_inputs.py)" for k in ELISL_CODE_KEYS
             if k not in code]
    recorded = dict(meta.get("input_sha256") or {})
    recorded.update(code)
    for rel, want in sorted(recorded.items()):
        f = root / rel
        if not f.is_file():
            errs.append(f"{ELISL_META} records '{rel}', which is not in this "
                        f"checkout ({f}) — never uploaded here? Copy it in, "
                        f"then rerun")
            continue
        have = sha256(f)
        if have != want:
            errs.append(f"{ELISL_META} was built from another '{rel}' "
                        f"(recorded {str(want)[:12]}, current {have[:12]})")
    return errs


CHECKS = {"SLGNN": check_slgnn, "NSF4SL": check_nsf4sl, "SLMGAE": check_slmgae,
          "ELISL": check_elisl}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=MODELS + OPT_IN,
                    default=list(MODELS))
    ap.add_argument("--root", default=str(REPO),
                    help="repo root (default: this checkout)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    bad = 0
    for model in args.models:
        errs = CHECKS[model](root)
        if not errs:
            print(f"  inputs OK: {model}")
            continue
        bad += 1
        for e in errs:
            print(f"ERROR [{model}]: {e}")
        print(f"  rebuild on the login node (repo root): "
              f"{rebuild_hint(model, root)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
