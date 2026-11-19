#!/usr/bin/env python3
"""
One-time conversion of siamese data files from gene symbols to NCBI Entrez
Gene IDs. The migration has already been applied: download_synlethdb.py and
generate_embeddings.py now write Entrez IDs directly, so on the current data
every file below is detected as already Entrez-keyed and skipped. A file is
Entrez-keyed when at least half its ids are all-digit (the rule
data_loader.load_single_embedding uses); converting one would map every id
through the symbol table, find nothing and write the file back empty.

Converts (when still symbol-keyed):
  1. data/all_genes_esm2.genes.txt  (gene list)
  2. data/SL_SynLethDB_experimental.txt  (SL pairs)
  3. data/SL_SynLethDB_experimental.sources.tsv  (SL pairs with sources)
  4. data/all_genes_*.pt  (embedding files: gene_order + gene_to_idx)

Never converted: SLMGAE's native inputs List_Proteins_in_SL.txt,
SL_Human_Approved.txt, computational_pairs.txt and
embeddings_esm2_t33_650M_UR50D_meanpool.pt. They stay in symbol space.
List_Proteins_in_SL.txt is the 6,375-row index of the support views, so
dropping or merging rows would shift every similarity row;
SLMGAE-in-pytorch/train_slmgae_shared.load_lp_genes maps it to Entrez at load
time, one entry per row. The other three share its symbol space: the
SL_Human_Approved.txt pairs and the legacy ESM gene_order are looked up in it,
and original_tensorflow_code/utils.py compares computational_pairs.txt against
SL_Human_Approved.txt.

Pipeline per file:
  1. Read gene ids; skip the file if they are already Entrez IDs
  2. Apply gene name correction (custom + HGNC normalization)
  3. Map to Entrez Gene ID
  4. Drop genes that can't be mapped (with warning)
  5. Write back

For .pt files, also reindexes gene_to_idx and drops rows from embedding
tensors for unmapped genes.

Backs up originals to data/_backup_symbols/ before overwriting.

Usage:
    cd SLMGAE-pytorch
    python siamese_sl/convert_to_entrez.py
    python siamese_sl/convert_to_entrez.py --dry_run  # preview without writing
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gene_name_utils import get_mapper

DATA_DIR = Path(__file__).parent.parent / "data"
BACKUP_DIR = DATA_DIR / "_backup_symbols"


def _already_entrez(ids: List[str]) -> bool:
    """True when at least half the non-empty ids are all-digit Entrez IDs."""
    ids = [g for g in ids if g]
    return bool(ids) and sum(g.isdigit() for g in ids) >= 0.5 * len(ids)


def backup(path: Path, dry_run: bool) -> None:
    """Copy original file to backup dir."""
    if dry_run:
        return
    rel = path.relative_to(DATA_DIR)
    dst = BACKUP_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)


def convert_gene_list(path: Path, mapper, dry_run: bool) -> bool:
    """Convert a file with one gene symbol per line to Entrez IDs.

    Returns True when the file is (or, under dry_run, would be) rewritten.
    """
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    with open(path) as f:
        symbols = [line.strip() for line in f if line.strip()]
    print(f"  Input: {len(symbols)} genes")
    if _already_entrez(symbols):
        print("  already Entrez IDs, skipped")
        return False

    entrez_ids = []
    dropped = []
    for sym in symbols:
        eid = mapper.symbol_to_entrez(sym)
        if eid:
            entrez_ids.append(eid)
        else:
            dropped.append(sym)

    print(f"  Mapped: {len(entrez_ids)}, Dropped: {len(dropped)}")
    if dropped:
        print(f"  Dropped genes: {dropped[:20]}")

    # Check for duplicates after mapping
    seen = set()
    deduped = []
    for eid in entrez_ids:
        if eid not in seen:
            seen.add(eid)
            deduped.append(eid)
    if len(deduped) < len(entrez_ids):
        print(f"  Deduped: {len(entrez_ids)} -> {len(deduped)}")
    entrez_ids = deduped

    if dry_run:
        print(f"  [DRY RUN] Would write {len(entrez_ids)} Entrez IDs")
        return True

    backup(path, dry_run)
    with open(path, 'w') as f:
        for eid in entrez_ids:
            f.write(f"{eid}\n")
    print(f"  Written: {len(entrez_ids)} Entrez IDs")
    return True


def convert_pair_file(path: Path,
                      mapper,
                      dry_run: bool,
                      has_header: bool = False) -> bool:
    """Convert a tab-separated gene pair file to Entrez IDs.

    Handles formats:
      gene1 TAB gene2 [TAB score_or_extra...]
      With optional header line.
    Returns True when the file is (or, under dry_run, would be) rewritten.
    """
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    with open(path) as f:
        lines = f.readlines()

    header = None
    if has_header:
        header = lines[0]
        lines = lines[1:]

    print(f"  Input: {len(lines)} pairs")
    ids = [g for line in lines for g in line.strip().split('\t')[:2]]
    if _already_entrez(ids):
        print("  already Entrez IDs, skipped")
        return False

    output_lines = []
    dropped = 0
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        g1, g2 = parts[0], parts[1]
        eid1 = mapper.symbol_to_entrez(g1)
        eid2 = mapper.symbol_to_entrez(g2)
        if eid1 is None or eid2 is None:
            dropped += 1
            continue
        parts[0] = eid1
        parts[1] = eid2
        output_lines.append('\t'.join(parts) + '\n')

    print(f"  Mapped: {len(output_lines)}, Dropped: {dropped}")

    if dry_run:
        print(f"  [DRY RUN] Would write {len(output_lines)} pairs")
        return True

    backup(path, dry_run)
    with open(path, 'w') as f:
        if header:
            f.write(header)
        f.writelines(output_lines)
    print(f"  Written: {len(output_lines)} pairs")
    return True


def convert_pt_file(path: Path, mapper, dry_run: bool) -> bool:
    """Convert a .pt embedding file's gene_order from symbols to Entrez IDs.

    Drops genes that can't be mapped. Reindexes embeddings tensor and
    gene_to_idx accordingly. Returns True when the file is (or, under
    dry_run, would be) rewritten.
    """
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    data = torch.load(path, map_location='cpu', weights_only=False)

    if not isinstance(data, dict):
        print("  SKIP: not a dict (raw tensor)")
        return False

    # Find gene order
    if 'gene_order' in data:
        symbols = list(data['gene_order'])
    elif 'gene_to_idx' in data:
        g2i = data['gene_to_idx']
        n = len(g2i)
        symbols = [''] * n
        for gene, idx in g2i.items():
            symbols[idx] = gene
    else:
        print("  SKIP: no gene_order or gene_to_idx")
        return False

    print(f"  Input: {len(symbols)} genes")
    if _already_entrez([str(g) for g in symbols]):
        print("  already Entrez IDs, skipped")
        return False

    # Map symbols -> Entrez IDs, track which indices to keep
    keep_indices = []
    new_gene_order = []
    dropped = []
    seen_eids = set()

    for i, sym in enumerate(symbols):
        # Normalize first, then map
        normalized = mapper.gene_name_normalize(sym)
        eid = mapper.symbol_to_entrez(normalized)
        if eid is None:
            dropped.append(sym)
            continue
        if eid in seen_eids:
            continue  # deduplicate
        seen_eids.add(eid)
        keep_indices.append(i)
        new_gene_order.append(eid)

    n_kept = len(new_gene_order)
    print(f"  Mapped: {n_kept}, Dropped: {len(dropped)}")
    if dropped and len(dropped) <= 20:
        print(f"  Dropped: {dropped}")
    elif dropped:
        print(f"  Dropped: {dropped[:10]}... ({len(dropped)} total)")

    if dry_run:
        print(f"  [DRY RUN] Would reindex to {n_kept} genes")
        return True

    backup(path, dry_run)

    # Reindex tensors
    keep = torch.tensor(keep_indices, dtype=torch.long)
    for key in ('embeddings', 'raw_embeddings'):
        if key in data and isinstance(data[key], torch.Tensor):
            orig_shape = data[key].shape
            data[key] = data[key][keep]
            print(f"  {key}: {orig_shape} -> {data[key].shape}")

    # Update gene_order
    data['gene_order'] = new_gene_order

    # Update gene_to_idx
    if 'gene_to_idx' in data:
        data['gene_to_idx'] = {
            eid: idx
            for idx, eid in enumerate(new_gene_order)
        }

    # Update num_genes if present
    if 'num_genes' in data:
        data['num_genes'] = n_kept

    # Update failed_genes if present (also convert to Entrez)
    if 'failed_genes' in data:
        new_failed = []
        for g in data['failed_genes']:
            normalized = mapper.gene_name_normalize(g)
            eid = mapper.symbol_to_entrez(normalized)
            if eid:
                new_failed.append(eid)
        data['failed_genes'] = new_failed

    torch.save(data, path)
    print(f"  Saved: {n_kept} genes")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert siamese data files that are still keyed by gene "
        "symbols to Entrez IDs (files already keyed by Entrez IDs are skipped)")
    parser.add_argument('--dry_run',
                        action='store_true',
                        help="Preview changes without writing")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=str(DATA_DIR / "cache"),
                        help="HGNC cache directory")
    args = parser.parse_args()

    mapper = get_mapper(cache_dir=args.cache_dir)

    if not args.dry_run:
        # backup() creates the directory on the first file it rewrites
        print(f"Originals of rewritten files are backed up to: {BACKUP_DIR}")

    # SLMGAE's native List_Proteins_in_SL.txt, SL_Human_Approved.txt,
    # computational_pairs.txt and legacy ESM tensor stay symbol-keyed (see
    # the module docstring).

    # Number of files rewritten (or, under --dry_run, that would be).
    n_converted = 0

    # 1. Gene lists
    n_converted += convert_gene_list(DATA_DIR / "all_genes_esm2.genes.txt",
                                     mapper, args.dry_run)

    # 2. SL pair files (no header, tab-separated)
    n_converted += convert_pair_file(
        DATA_DIR / "SL_SynLethDB_experimental.txt", mapper, args.dry_run)

    # 3. SL sources file (has header)
    n_converted += convert_pair_file(
        DATA_DIR / "SL_SynLethDB_experimental.sources.tsv",
        mapper,
        args.dry_run,
        has_header=True)

    # 4. All .pt embedding files
    pt_files = sorted(DATA_DIR.glob("all_genes_*.pt"))
    for pt_path in pt_files:
        n_converted += convert_pt_file(pt_path, mapper, args.dry_run)

    print("\n" + "=" * 60)
    if n_converted == 0:
        print("Nothing to convert: no file needed rewriting (see the "
              "per-file lines above).")
    elif args.dry_run:
        print(f"[DRY RUN] {n_converted} file(s) would be converted; "
              f"nothing was written.")
    else:
        print(f"Converted {n_converted} file(s); originals backed up to: "
              f"{BACKUP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
