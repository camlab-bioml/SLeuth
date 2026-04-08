#!/usr/bin/env python3
"""
One-time conversion of all data files from gene symbols to NCBI Entrez Gene IDs.

Converts:
  1. data/List_Proteins_in_SL.txt  (gene list)
  2. data/SL_Human_Approved.txt    (SL pairs with scores)
  3. data/SL_SynLethDB_experimental.txt  (SL pairs)
  4. data/SL_SynLethDB_experimental.sources.tsv  (SL pairs with sources)
  5. data/computational_pairs.txt  (gene pairs with header)
  6. data/all_genes_esm.genes.txt  (gene list)
  7. data/all_genes_*.pt  (18 .pt embedding files: gene_order + gene_to_idx)
  8. data/embeddings_esm2_t33_650M_UR50D_meanpool.pt  (legacy ESM)

Pipeline per file:
  1. Read gene symbols
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


def backup(path: Path, dry_run: bool) -> None:
    """Copy original file to backup dir."""
    if dry_run:
        return
    rel = path.relative_to(DATA_DIR)
    dst = BACKUP_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)


def convert_gene_list(path: Path, mapper, dry_run: bool) -> None:
    """Convert a file with one gene symbol per line to Entrez IDs."""
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    with open(path) as f:
        symbols = [line.strip() for line in f if line.strip()]
    print(f"  Input: {len(symbols)} genes")

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
        return

    backup(path, dry_run)
    with open(path, 'w') as f:
        for eid in entrez_ids:
            f.write(f"{eid}\n")
    print(f"  Written: {len(entrez_ids)} Entrez IDs")


def convert_pair_file(path: Path,
                      mapper,
                      dry_run: bool,
                      has_header: bool = False) -> None:
    """Convert a tab-separated gene pair file to Entrez IDs.

    Handles formats:
      gene1 TAB gene2 [TAB score_or_extra...]
      With optional header line.
    """
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    with open(path) as f:
        lines = f.readlines()

    header = None
    if has_header:
        header = lines[0]
        lines = lines[1:]

    print(f"  Input: {len(lines)} pairs")

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
        return

    backup(path, dry_run)
    with open(path, 'w') as f:
        if header:
            f.write(header)
        f.writelines(output_lines)
    print(f"  Written: {len(output_lines)} pairs")


def convert_pt_file(path: Path, mapper, dry_run: bool) -> None:
    """Convert a .pt embedding file's gene_order from symbols to Entrez IDs.

    Drops genes that can't be mapped. Reindexes embeddings tensor and
    gene_to_idx accordingly.
    """
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    data = torch.load(path, map_location='cpu', weights_only=False)

    if not isinstance(data, dict):
        print("  SKIP: not a dict (raw tensor)")
        return

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
        return

    print(f"  Input: {len(symbols)} genes")

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
        return

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


def convert_raw_tensor(path: Path, gene_list_path: Path, mapper,
                       dry_run: bool) -> None:
    """Convert a raw tensor .pt file that is indexed by a gene list file.

    The gene list file should ALREADY be converted to Entrez IDs by the
    time this runs. We re-read it to get the Entrez IDs and the
    correspondence to original positions.

    Since the gene list was converted first (dropping unmapped genes),
    we need to use the BACKUP of the original gene list to find the
    original positions, then select the same rows from the tensor.
    """
    print(f"\n--- {path.relative_to(DATA_DIR)} ---")
    tensor = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        print("  SKIP: not a raw tensor")
        return

    # Read the ORIGINAL gene list from backup (symbols)
    backup_gene_list = BACKUP_DIR / gene_list_path.relative_to(DATA_DIR)
    if backup_gene_list.exists():
        with open(backup_gene_list) as f:
            original_symbols = [line.strip() for line in f if line.strip()]
    else:
        # Backup doesn't exist yet — read current (should still be symbols)
        with open(gene_list_path) as f:
            original_symbols = [line.strip() for line in f if line.strip()]

    print(f"  Tensor: {tensor.shape}, Gene list: {len(original_symbols)}")

    if tensor.shape[0] != len(original_symbols):
        print(f"  ERROR: tensor rows ({tensor.shape[0]}) != gene list "
              f"({len(original_symbols)})")
        return

    # Map symbols -> Entrez, track which rows to keep
    keep_indices = []
    new_gene_order = []
    seen = set()
    for i, sym in enumerate(original_symbols):
        eid = mapper.symbol_to_entrez(sym)
        if eid and eid not in seen:
            seen.add(eid)
            keep_indices.append(i)
            new_gene_order.append(eid)

    n_kept = len(new_gene_order)
    print(f"  Mapped: {n_kept}, Dropped: {len(original_symbols) - n_kept}")

    if dry_run:
        print(f"  [DRY RUN] Would reindex to {n_kept} rows")
        return

    backup(path, dry_run)
    keep = torch.tensor(keep_indices, dtype=torch.long)
    new_tensor = tensor[keep]

    # Save as dict with gene_order (upgrade from raw tensor)
    data = {
        'embeddings': new_tensor,
        'gene_order': new_gene_order,
        'num_genes': n_kept,
    }
    torch.save(data, path)
    print(f"  Saved: {new_tensor.shape} with gene_order")


def main():
    parser = argparse.ArgumentParser(
        description="Convert all data files from gene symbols to Entrez IDs")
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
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Backups will be saved to: {BACKUP_DIR}")

    # 1. Gene lists
    convert_gene_list(DATA_DIR / "List_Proteins_in_SL.txt", mapper,
                      args.dry_run)
    convert_gene_list(DATA_DIR / "all_genes_esm.genes.txt", mapper,
                      args.dry_run)

    # 2. SL pair files (no header, tab-separated)
    convert_pair_file(DATA_DIR / "SL_Human_Approved.txt", mapper, args.dry_run)
    convert_pair_file(DATA_DIR / "SL_SynLethDB_experimental.txt", mapper,
                      args.dry_run)

    # 3. SL sources file (has header)
    convert_pair_file(DATA_DIR / "SL_SynLethDB_experimental.sources.tsv",
                      mapper,
                      args.dry_run,
                      has_header=True)

    # 4. Computational pairs (has header)
    convert_pair_file(DATA_DIR / "computational_pairs.txt",
                      mapper,
                      args.dry_run,
                      has_header=True)

    # 5. All .pt embedding files
    pt_files = sorted(DATA_DIR.glob("all_genes_*.pt"))
    for pt_path in pt_files:
        convert_pt_file(pt_path, mapper, args.dry_run)

    # 6. Legacy ESM embedding (raw tensor indexed by List_Proteins_in_SL.txt)
    legacy_esm = DATA_DIR / "embeddings_esm2_t33_650M_UR50D_meanpool.pt"
    if legacy_esm.exists():
        convert_raw_tensor(legacy_esm, DATA_DIR / "List_Proteins_in_SL.txt",
                           mapper, args.dry_run)

    print("\n" + "=" * 60)
    print("Conversion complete.")
    if not args.dry_run:
        print(f"Originals backed up to: {BACKUP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
