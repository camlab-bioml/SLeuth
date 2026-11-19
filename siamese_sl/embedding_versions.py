#!/usr/bin/env python3
"""Are the catalog embedding .pt files the CURRENT generators' output?

Every generator (generate_all_genes_esm.py, generate_go_esm_embeddings.py,
generate_embeddings.py) stamps its .pt with `generator_version` =
generate_all_genes_esm.EMBEDDING_GENERATOR_VERSION, or with a marked-down
string when it reused an input it did not build (an older cached pkl, an ESM
file from older code). A file counts as current only when it carries exactly
the current version, its gene_order has exactly the gene universe's genes
(<data_dir>/gene_universe.txt, when that file exists) and it lists them in the
same order as the reference file (all_genes_esm2.pt when checked, else the
first current file): data_loader.load_raw_multimodal refuses to combine files
whose ordered gene lists differ. Where all_genes_esm2.pt records a `pooling`
(every real ESM output does), it must additionally match
generate_all_genes_esm.PIPELINE_ESM's pooling and model, since --pooling mean
or another --model changes the vectors without changing the code.
Bump EMBEDDING_GENERATOR_VERSION whenever a
generator change alters its output, so every older .pt reads as stale here
(a protein-resolver change also bumps SEQUENCE_CACHE_FORMAT, or the kept
protein_sequences_canonical.json feeds the old sequences to the new build).

The files checked are the embedding catalog the benchmark trains on:
slurm/run_best_per_category.conf EMB_CATALOG (esm2 and go included), read by
sourcing the conf files in bash; --all adds config.conf ALL_EMBEDDINGS
(genept etc.), --files names files explicitly instead. In catalog mode the
files of slurm/run_generate_embeddings.conf OPTIONAL_TYPES (types the
generation job may leave out, e.g. esmc, which it does not generate) may be
absent, as in that job's own end-of-run check; present, they must be current.

Usage (from the repo root; the root orchestrator's generation gate):
    python siamese_sl/embedding_versions.py --data_dir data

Exit status: 0 all current; 1 any file stale or missing (run generation);
2 the catalog could not be read (treat as stale). --allow_missing names files
(e.g. an embedding this environment cannot generate) whose absence is
reported but does not count; present, they must still be current.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from generate_all_genes_esm import (EMBEDDING_GENERATOR_VERSION,  # noqa: E402
                                    PIPELINE_ESM)

SLURM_DIR = HERE / "slurm"

# Prints EMB_CATALOG (and with $3 = 1 also ALL_EMBEDDINGS), one entry per
# line. SLURM_JOB_ID unset keeps config.conf's compute-node block inert.
_CATALOG_SH = r'''
unset SLURM_JOB_ID
source "$1" >/dev/null 2>&1 || exit 3
source "$2" >/dev/null 2>&1 || exit 3
printf '%s\n' "${EMB_CATALOG[@]}"
if [ "$3" = 1 ]; then printf '%s\n' "${ALL_EMBEDDINGS[@]}"; fi
'''


# Prints the catalog entries (type:file:...) of OPTIONAL_TYPES, sourced in
# the generation job's order: config.conf, run_generate_embeddings.conf,
# run_best_per_category.conf.
_OPTIONAL_SH = r'''
unset SLURM_JOB_ID
source "$1" >/dev/null 2>&1 || exit 3
source "$2" >/dev/null 2>&1 || exit 3
source "$3" >/dev/null 2>&1 || exit 3
for t in "${OPTIONAL_TYPES[@]}"; do
    for e in "${EMB_CATALOG[@]}" "${ALL_EMBEDDINGS[@]}"; do
        case "$e" in "$t":*) printf '%s\n' "$e" ;; esac
    done
done
'''


def optional_files(config_conf, gen_conf, catalog_conf):
    """.pt filenames of the generation job's OPTIONAL_TYPES."""
    out = subprocess.run(
        ["bash", "-c", _OPTIONAL_SH, "_", str(config_conf), str(gen_conf),
         str(catalog_conf)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sourcing {gen_conf} failed "
                           f"(exit {out.returncode}): {out.stderr.strip()}")
    return sorted({ln.split(":")[1] for ln in out.stdout.splitlines()
                   if len(ln.split(":")) >= 2})


def catalog_files(config_conf, catalog_conf, include_all=False):
    """Ordered unique .pt filenames named by the catalog conf arrays."""
    out = subprocess.run(
        ["bash", "-c", _CATALOG_SH, "_", str(config_conf), str(catalog_conf),
         "1" if include_all else "0"],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sourcing {config_conf} / {catalog_conf} failed "
                           f"(exit {out.returncode}): {out.stderr.strip()}")
    names = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(":")
        if len(parts) >= 2 and parts[1].endswith(".pt") \
                and parts[1] not in names:
            names.append(parts[1])
    if not names:
        raise RuntimeError(f"no .pt entries in EMB_CATALOG ({catalog_conf})")
    return names


def _load(path):
    """torch.load without reading tensor payloads when the format allows."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False,
                          mmap=True)
    except (RuntimeError, TypeError, ValueError):
        # legacy (non-zip) serialisation, or an older torch without mmap
        return torch.load(path, map_location="cpu", weights_only=False)


def _gene_order(path):
    """The file's gene_order as a list of str ([] if absent)."""
    d = _load(path)
    return list(map(str, (d.get("gene_order") or []) if isinstance(d, dict)
                    else []))


def check_file(path, universe=None, ref=None):
    """(status, detail): status is 'ok', 'stale' or 'missing'.

    universe: set of gene ids the file must hold exactly. ref: optional
    (name, ordered gene list) whose order the file must match.
    """
    path = Path(path)
    if not path.is_file():
        return "missing", "file not found"
    try:
        d = _load(path)
    except Exception as e:  # noqa: BLE001
        return "stale", f"unreadable ({type(e).__name__}: {e})"
    if not isinstance(d, dict):
        return "stale", f"not an embedding dict ({type(d).__name__})"
    ver = d.get("generator_version")
    if ver != EMBEDDING_GENERATOR_VERSION:
        return "stale", (f"generator_version {ver!r} != current "
                         f"{EMBEDDING_GENERATOR_VERSION!r}")
    # The version says which generator CODE wrote the file, not with which
    # options. generate_all_genes_esm.py now marks its stamp down for a
    # non-pipeline --pooling / --model, but a file written before that (with
    # the current version and, say, mean pooling) would still read as current,
    # and README/--help document a mean-pooling command. Only ESM outputs
    # carry `pooling`, so an absent key is not evidence of a deviant build.
    if path.name == "all_genes_esm2.pt" and "pooling" in d:
        built = (d.get("pooling"), d.get("model_name"))
        if built != PIPELINE_ESM:
            return "stale", (f"pooling/model {built!r} != the pipeline's "
                             f"{PIPELINE_ESM!r}")
    if universe is not None:
        have = set(map(str, d.get("gene_order") or []))
        if have != universe:
            return "stale", (f"gene_order has {len(have):,} genes vs "
                             f"{len(universe):,} in gene_universe.txt "
                             f"({len(universe - have):,} missing, "
                             f"{len(have - universe):,} unexpected)")
    if ref is not None and list(map(str, d.get("gene_order") or [])) != ref[1]:
        return "stale", (f"gene_order not in {ref[0]}'s order "
                         f"(data_loader.load_raw_multimodal needs identical "
                         f"ordered gene lists across files)")
    return "ok", f"generator_version {ver}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Report catalog embeddings not written by the current "
        "generator code (exit 1 if any is stale or missing).")
    ap.add_argument("--data_dir", required=True,
                    help="directory holding the all_genes_*.pt files")
    ap.add_argument("--gene_universe", default=None,
                    help="gene list every file must cover exactly (default "
                         "<data_dir>/gene_universe.txt; not checked when "
                         "absent)")
    ap.add_argument("--all", action="store_true",
                    help="also check config.conf ALL_EMBEDDINGS (genept etc.)")
    ap.add_argument("--files", nargs="+", default=None,
                    help="check these filenames under --data_dir instead of "
                         "the catalog")
    ap.add_argument("--allow_missing", nargs="+", default=[],
                    metavar="FILE",
                    help="filenames whose absence does not fail the check")
    ap.add_argument("--config_conf", default=str(SLURM_DIR / "config.conf"))
    ap.add_argument("--catalog_conf",
                    default=str(SLURM_DIR / "run_best_per_category.conf"))
    ap.add_argument("--gen_conf",
                    default=str(SLURM_DIR / "run_generate_embeddings.conf"),
                    help="its OPTIONAL_TYPES may be absent (catalog mode)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    allow_missing = set(args.allow_missing)
    if args.files:
        names = list(dict.fromkeys(args.files))
    else:
        try:
            names = catalog_files(args.config_conf, args.catalog_conf,
                                  args.all)
            allow_missing.update(optional_files(
                args.config_conf, args.gen_conf, args.catalog_conf))
        except (OSError, RuntimeError) as e:
            print(f"ERROR: cannot read the embedding catalog: {e}")
            return 2
    uni_path = Path(args.gene_universe or data_dir / "gene_universe.txt")
    universe = None
    if uni_path.is_file():
        universe = {ln.strip() for ln in open(uni_path) if ln.strip()}
    else:
        print(f"note: {uni_path} not found; gene_order is not checked")

    print(f"current generator_version: {EMBEDDING_GENERATOR_VERSION}")
    # Order reference: ESM-2's (GO and every generate_embeddings.py type take
    # their gene list from it), else the first file that is current.
    ref = None
    first = ["all_genes_esm2.pt"] if "all_genes_esm2.pt" in names else []
    for name in first + [n for n in names if n not in first]:
        if check_file(data_dir / name, universe)[0] == "ok":
            ref = (name, _gene_order(data_dir / name))
            break
    bad = {"stale": [], "missing": []}
    absent = []
    for name in names:
        status, detail = check_file(data_dir / name, universe, ref)
        if status == "missing" and name in allow_missing:
            print(f"  {'ABSENT':8s} {name}: {detail} (allowed)")
            absent.append(name)
            continue
        print(f"  {status.upper():8s} {name}: {detail}")
        if status != "ok":
            bad[status].append(name)
    n_bad = len(bad["stale"]) + len(bad["missing"])
    print(f"{len(names) - n_bad - len(absent)} of {len(names)} current"
          + (f"; absent (allowed): {' '.join(absent)}" if absent else "")
          + (f"; stale: {' '.join(bad['stale'])}" if bad["stale"] else "")
          + (f"; missing: {' '.join(bad['missing'])}" if bad["missing"]
             else ""))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
