#!/usr/bin/env python3
"""
Download SynLethDB human gene-pair tables and filter to experimental-only.

SynLethDB ships its human tables from the v3 portal (https://www.synlethdb.com)
as Google-Drive ``*.detailed.tsv`` files. The SL (positive) and non-SL
(negative) tables share an identical 15-column schema, so this one script
downloads either, applies the same experimental-source filter, and writes a
``gene1<TAB>gene2`` file (no header) plus a sidecar ``.sources.tsv`` that
preserves per-pair provenance.

NOTE on versions: the v3 portal's ``Human.SL.detailed.tsv`` is byte-identical
to the v2 export (same Drive file ID) — there is no new human-SL release. The
genuinely new v3 asset is ``Human.non.SL.detailed.tsv``: experimentally
screened non-SL pairs, used here as real negatives instead of random sampling.

Experimental sources (allowlist — pair kept if it has at least one):
  - GenomeRNAi
  - CRISPR/CRISPRi
  - High Throughput
  - Low Throughput
  - RNAi Screen
  - Drug Screen

Non-experimental sources (not counted): Computational Prediction,
Decipher, Synlethality, Daisy, Text Mining.

Output: gene1<TAB>gene2  (no header, no score — binary SL / non-SL status)
Provenance (rel_source, cell_line, pubmed_id) is preserved in a separate
``.sources.tsv`` file for reference.

Usage:
    # Positives (SL pairs) — default
    python download_synlethdb.py --output ../data/SL_SynLethDB_experimental.txt

    # Negatives (experimentally screened non-SL pairs)
    python download_synlethdb.py --relation NONSL \
        --output ../data/SL_SynLethDB_experimental_negatives.txt

    # Keep only specific source types
    python download_synlethdb.py --output ../data/SL_SynLethDB_experimental.txt \
        --keep_sources "CRISPR/CRISPRi" "GenomeRNAi" "RNAi Screen" "Low Throughput"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Set

# Google Drive file IDs from the SynLethDB v3 download page
# (https://www.synlethdb.com/download). "SL" is byte-identical to the v2 export.
DRIVE_IDS = {
    "SL": "1yDVv789aRbY3eBJz7qetrWQMn1zHoR6X",     # Human.SL.detailed.tsv
    "NONSL": "1SQRINp58iL5mN6EOYvQ5I9g9WvpD_F2T",  # Human.non.SL.detailed.tsv
}
RAW_NAMES = {
    "SL": "Human.SL.detailed.tsv",
    "NONSL": "Human.non.SL.detailed.tsv",
}

# Source types that count as experimental (wet-lab / screen-based).
# A pair is kept if it has at least one of these. Any unknown source
# type defaults to excluded — safer than a blocklist if SynLethDB adds
# new computational sources.
EXPERIMENTAL_SOURCES = {
    "GenomeRNAi",
    "CRISPR/CRISPRi",
    "High Throughput",
    "Low Throughput",
    "RNAi Screen",
    "Drug Screen",
}


def download_tsv(output_path: str, file_id: str, label: str) -> bool:
    """Download a SynLethDB ``*.detailed.tsv`` from Google Drive."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"Downloading SynLethDB {label} table (Drive id {file_id})...")

    # Try gdown first (handles Google Drive confirmations)
    try:
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  gdown failed: {e}")

    # Fallback to urllib
    try:
        import urllib.request
        print(f"  Downloading via urllib...")
        urllib.request.urlretrieve(url, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True
    except Exception as e:
        print(f"  urllib failed: {e}")

    # Fallback to subprocess wget/curl
    import subprocess
    import shutil
    for cmd in [
        ["wget", "-q", "-O", output_path, url],
        ["curl", "-sL", "-o", output_path, url],
    ]:
        if shutil.which(cmd[0]):
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode == 0 and os.path.exists(
                    output_path) and os.path.getsize(output_path) > 1000:
                return True

    print("ERROR: Could not download SynLethDB data")
    return False


def is_experimental(rel_source: str,
                    allow: Set[str] = EXPERIMENTAL_SOURCES) -> bool:
    """Check if a pair has at least one experimental source.

    rel_source can be semicolon-separated (e.g., "CRISPR/CRISPRi;Text Mining")
    or occasionally pipe-separated.
    """
    # Split on both ; and |
    sources = set()
    for part in rel_source.replace("|", ";").split(";"):
        sources.add(part.strip())

    # Keep if at least one source IS in the allowlist
    return any(s in allow for s in sources if s)


def _split_multi(value: str) -> Set[str]:
    """Split a ``;``/``|``-separated provenance cell into a clean set."""
    out = set()
    for part in value.replace("|", ";").split(";"):
        s = part.strip()
        if s:
            out.add(s)
    return out


def process_synlethdb(raw_tsv_path: str,
                      output_path: str,
                      keep_sources: list = None) -> dict:
    """Filter a SynLethDB detailed TSV to experimental pairs (Entrez IDs).

    Uses the x:START_ID and y:END_ID columns (NCBI Entrez Gene IDs)
    directly from the SynLethDB TSV. Gene symbols (x_name, y_name) are
    NOT used — Entrez IDs are the canonical identifier. Works for both the
    SL and non-SL tables (identical schema).

    Args:
        raw_tsv_path: Path to downloaded *.detailed.tsv
        output_path: Path for output file (entrez1 TAB entrez2)
        keep_sources: If set, only keep pairs with at least one of these
            source types (overrides default exclusion logic)

    Returns:
        Stats dict
    """
    # (entrez1, entrez2) -> {"sources": set, "cell_lines": set, "pubmed_ids": set}
    pairs = {}
    total = 0
    filtered_out = 0
    skipped_no_id = 0
    skipped_self_loop = 0

    with open(raw_tsv_path, "r") as f:
        header = f.readline().strip().split("\t")

        # Find column indices — use Entrez ID columns directly
        try:
            col_xid = header.index("x:START_ID")
            col_yid = header.index("y:END_ID")
            col_src = header.index("rel_source")
        except ValueError as e:
            print(f"ERROR: Missing column in TSV header: {e}")
            print(f"  Header: {header}")
            return {}

        # Optional provenance columns (graceful if a future export drops them)
        col_cell = header.index("cell_line") if "cell_line" in header else None
        col_pmid = header.index("pubmed_id") if "pubmed_id" in header else None
        # Keep a row if it has the ESSENTIAL columns; provenance columns are
        # captured opportunistically (guarded below), so a row missing only its
        # trailing cell_line/pubmed_id is still used. rstrip("\r\n") drops the
        # line terminator without collapsing trailing empty (tab-delimited)
        # fields the way .strip() would.
        essential_max = max(col_xid, col_yid, col_src)

        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) <= essential_max:
                continue
            total += 1

            rel_source = fields[col_src]

            # Filter by source type
            if keep_sources:
                if not any(s in keep_sources
                           for s in _split_multi(rel_source)):
                    filtered_out += 1
                    continue
            else:
                if not is_experimental(rel_source):
                    filtered_out += 1
                    continue

            eid1 = fields[col_xid].strip()
            eid2 = fields[col_yid].strip()

            if not eid1 or not eid2:
                skipped_no_id += 1
                continue

            # Skip self-loops
            if eid1 == eid2:
                skipped_self_loop += 1
                continue

            # Canonical ordering (numeric sort, keeping IDs as strings)
            if (len(eid1), eid1) > (len(eid2), eid2):
                eid1, eid2 = eid2, eid1

            key = (eid1, eid2)
            rec = pairs.setdefault(key, {
                "sources": set(),
                "cell_lines": set(),
                "pubmed_ids": set(),
            })
            rec["sources"].update(_split_multi(rel_source))
            if col_cell is not None and col_cell < len(fields):
                rec["cell_lines"].update(_split_multi(fields[col_cell]))
            if col_pmid is not None and col_pmid < len(fields):
                rec["pubmed_ids"].update(_split_multi(fields[col_pmid]))

    # Write output: entrez1 TAB entrez2 (no header)
    _numkey = lambda x: ((len(x[0]), x[0]), (len(x[1]), x[1]))
    with open(output_path, "w") as f:
        for (g1, g2) in sorted(pairs.keys(), key=_numkey):
            f.write(f"{g1}\t{g2}\n")

    # Write provenance file for reference
    if output_path.endswith(".txt"):
        sources_path = output_path.rsplit(".txt", 1)[0] + ".sources.tsv"
    else:
        sources_path = output_path + ".sources.tsv"
    with open(sources_path, "w") as f:
        f.write("entrez1\tentrez2\tsources\tcell_lines\tpubmed_ids\n")
        for (g1, g2) in sorted(pairs.keys(), key=_numkey):
            rec = pairs[(g1, g2)]
            src_str = ";".join(sorted(rec["sources"]))
            cell_str = ";".join(sorted(rec["cell_lines"]))
            pmid_str = ";".join(sorted(rec["pubmed_ids"]))
            f.write(f"{g1}\t{g2}\t{src_str}\t{cell_str}\t{pmid_str}\n")

    stats = {
        "total_in_synlethdb": total,
        "filtered_out_computational": filtered_out,
        "skipped_no_id": skipped_no_id,
        "skipped_self_loop": skipped_self_loop,
        "unique_pairs_kept": len(pairs),
        "output_path": output_path,
        "sources_path": sources_path,
    }

    print(f"\nSynLethDB filtering results:")
    print(f"  Total pairs in download: {total}")
    print(f"  Filtered out (computational): {filtered_out}")
    print(f"  Skipped (no Entrez ID): {skipped_no_id}")
    print(f"  Skipped self-loops: {skipped_self_loop}")
    print(f"  Unique experimental pairs: {len(pairs)}")
    print(f"  Saved to: {output_path}")
    print(f"  Provenance annotations: {sources_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download SynLethDB experimental SL / non-SL pairs")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output path (gene1 TAB gene2, no header)")
    parser.add_argument(
        "--relation",
        type=str,
        choices=sorted(DRIVE_IDS.keys()),
        default="SL",
        help="Which table to download: SL (positives, default) or "
        "NONSL (experimentally screened negatives)")
    parser.add_argument(
        "--file_id",
        type=str,
        default=None,
        help="Override the Google Drive file ID (default: per --relation)")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../data/cache",
        help="Directory for cached raw download (default: ../data/cache)")
    parser.add_argument("--keep_sources",
                        type=str,
                        nargs="+",
                        default=None,
                        help="Only keep pairs with these source types "
                        "(default: keep all except computational)")
    args = parser.parse_args()

    file_id = args.file_id or DRIVE_IDS[args.relation]
    raw_name = RAW_NAMES[args.relation]

    # Download raw TSV (cache it)
    os.makedirs(args.cache_dir, exist_ok=True)
    raw_path = os.path.join(args.cache_dir, raw_name)
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
        print(f"Using cached SynLethDB download: {raw_path}")
    else:
        if not download_tsv(raw_path, file_id, args.relation):
            sys.exit(1)

    # Verify it looks like a TSV
    with open(raw_path) as f:
        header = f.readline()
    if "x_name" not in header:
        print(f"ERROR: Downloaded file doesn't look like SynLethDB TSV")
        print(f"  First line: {header[:200]}")
        sys.exit(1)

    print(f"Raw TSV: {raw_path}")

    # Process
    keep = set(args.keep_sources) if args.keep_sources else None
    stats = process_synlethdb(raw_path, args.output, keep)

    if not stats:
        sys.exit(1)


if __name__ == "__main__":
    main()
