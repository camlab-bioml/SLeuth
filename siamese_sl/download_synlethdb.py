#!/usr/bin/env python3
"""
Download SynLethDB 2.0 human SL pairs and filter to experimental-only.

Downloads Human.SL.detailed.tsv from SynLethDB 2.0 (Google Drive),
filters out computationally derived pairs, uses Entrez Gene IDs directly
from SynLethDB (no gene name normalization is performed), deduplicates,
and saves in the same tab-separated format as
SL_Human_Approved.txt (gene1 TAB gene2).

Experimental sources (allowlist — pair kept if it has at least one):
  - GenomeRNAi
  - CRISPR/CRISPRi
  - High Throughput
  - Low Throughput
  - RNAi Screen
  - Drug Screen

Non-experimental sources (not counted): Computational Prediction,
Decipher, Synlethality, Daisy, Text Mining.

Output: gene1<TAB>gene2  (no header, no score — binary SL status)
The source types are preserved in a separate .sources.tsv file for reference.

Usage:
    python download_synlethdb.py --output ../data/SL_SynLethDB_experimental.txt

    # Keep only specific source types
    python download_synlethdb.py --output ../data/SL_SynLethDB_experimental.txt \
        --keep_sources "CRISPR/CRISPRi" "GenomeRNAi" "RNAi Screen" "Low Throughput"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Set

# Google Drive file ID for Human.SL.detailed.tsv
SYNLETHDB_FILE_ID = "1yDVv789aRbY3eBJz7qetrWQMn1zHoR6X"
SYNLETHDB_URL = f"https://drive.google.com/uc?export=download&id={SYNLETHDB_FILE_ID}"

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


def download_tsv(output_path: str) -> bool:
    """Download Human.SL.detailed.tsv from Google Drive."""
    print(f"Downloading SynLethDB 2.0 human SL pairs...")

    # Try gdown first (handles Google Drive confirmations)
    try:
        import gdown
        gdown.download(id=SYNLETHDB_FILE_ID, output=output_path, quiet=False)
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
        urllib.request.urlretrieve(SYNLETHDB_URL, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return True
    except Exception as e:
        print(f"  urllib failed: {e}")

    # Fallback to subprocess wget/curl
    import subprocess
    import shutil
    for cmd in [
        ["wget", "-q", "-O", output_path, SYNLETHDB_URL],
        ["curl", "-sL", "-o", output_path, SYNLETHDB_URL],
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


def process_synlethdb(raw_tsv_path: str,
                      output_path: str,
                      keep_sources: list = None) -> dict:
    """Filter SynLethDB TSV to experimental pairs using Entrez Gene IDs.

    Uses the x:START_ID and y:END_ID columns (NCBI Entrez Gene IDs)
    directly from the SynLethDB TSV. Gene symbols (x_name, y_name) are
    NOT used — Entrez IDs are the canonical identifier.

    Args:
        raw_tsv_path: Path to downloaded Human.SL.detailed.tsv
        output_path: Path for output file (entrez1 TAB entrez2)
        keep_sources: If set, only keep pairs with at least one of these
            source types (overrides default exclusion logic)

    Returns:
        Stats dict
    """
    pairs = {}  # (entrez1, entrez2) -> set of source types
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

        for line in f:
            fields = line.strip().split("\t")
            if len(fields) <= max(col_xid, col_yid, col_src):
                continue
            total += 1

            rel_source = fields[col_src]

            # Filter by source type
            if keep_sources:
                sources = set()
                for part in rel_source.replace("|", ";").split(";"):
                    sources.add(part.strip())
                if not any(s in keep_sources for s in sources):
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

            # Collect source types per pair
            if key not in pairs:
                pairs[key] = set()
            for part in rel_source.replace("|", ";").split(";"):
                s = part.strip()
                if s:
                    pairs[key].add(s)

    # Write output: entrez1 TAB entrez2 (no header)
    _numkey = lambda x: ((len(x[0]), x[0]), (len(x[1]), x[1]))
    with open(output_path, "w") as f:
        for (g1, g2) in sorted(pairs.keys(), key=_numkey):
            f.write(f"{g1}\t{g2}\n")

    # Write source annotations file for reference
    if output_path.endswith(".txt"):
        sources_path = output_path.rsplit(".txt", 1)[0] + ".sources.tsv"
    else:
        sources_path = output_path + ".sources.tsv"
    with open(sources_path, "w") as f:
        f.write("entrez1\tentrez2\tsources\n")
        for (g1, g2) in sorted(pairs.keys(), key=_numkey):
            src_str = ";".join(sorted(pairs[(g1, g2)]))
            f.write(f"{g1}\t{g2}\t{src_str}\n")

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
    print(f"  Source annotations: {sources_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download SynLethDB 2.0 experimental SL pairs")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output path (gene1 TAB gene2, no header)")
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

    # Download raw TSV (cache it)
    os.makedirs(args.cache_dir, exist_ok=True)
    raw_path = os.path.join(args.cache_dir, "Human.SL.detailed.tsv")
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
        print(f"Using cached SynLethDB download: {raw_path}")
    else:
        if not download_tsv(raw_path):
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
