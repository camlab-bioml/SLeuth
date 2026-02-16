#!/usr/bin/env python3
"""
Combine ESM-2 embeddings with anc2vec Gene Ontology embeddings.

Takes pre-generated ESM embeddings (.pt) and appends anc2vec GO embeddings
(200-dim, sum-pooled over annotated GO terms) to produce combined features.

anc2vec Reference:
  Edera et al. "anc2vec: embedding Gene Ontology terms by preserving
  ancestors relationships" Briefings in Bioinformatics, 2022.

Usage:
    python generate_go_esm_embeddings.py \
        --esm_embeddings ../data/all_genes_esm.pt \
        --gaf ../uniprot_GO/goa_human.gaf.gz \
        --output ../data/all_genes_esm_go.pt
"""

import gzip
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import requests
from tqdm import tqdm


ANC2VEC_URL = "https://github.com/aedera/anc2vec/raw/main/anc2vec/data/embeddings.npz"


def load_anc2vec(npz_path: str) -> dict:
    """Load anc2vec GO term embeddings from a local .npz file."""
    print(f"Loading anc2vec embeddings from {npz_path}...")
    npz = np.load(npz_path, allow_pickle=True)
    go_embeds = npz["embds"].item()  # dict: GO term ID -> 200-dim vector
    print(f"Loaded {len(go_embeds)} GO term embeddings (dim={next(iter(go_embeds.values())).shape[0]})")
    return go_embeds


def download_anc2vec(cache_path: str) -> dict:
    """Download anc2vec GO term embeddings, caching locally."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        print(f"Downloading anc2vec embeddings to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(ANC2VEC_URL, stream=True)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Using cached anc2vec: {cache_path}")
    return load_anc2vec(str(cache_path))


def parse_gaf(gaf_path: str) -> dict:
    """
    Parse GAF 2.2 annotation file to build gene -> GO term mapping.

    GAF format: tab-separated, comment lines start with '!'.
    Field[2] = gene symbol, Field[4] = GO term ID.
    """
    gene_to_go = defaultdict(set)
    not_excluded = 0

    opener = gzip.open if gaf_path.endswith(".gz") else open
    mode = "rt" if gaf_path.endswith(".gz") else "r"

    with opener(gaf_path, mode) as f:
        for line in f:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            # GAF 2.2: field[3] is Qualifier — skip NOT annotations
            qualifier = fields[3]
            if "NOT" in qualifier.upper():
                not_excluded += 1
                continue
            gene_symbol = fields[2]
            go_term = fields[4]
            gene_to_go[gene_symbol].add(go_term)

    print(f"Parsed GAF: {len(gene_to_go)} genes with GO annotations")
    if not_excluded:
        print(f"  Excluded {not_excluded} NOT-qualified annotations")
    return dict(gene_to_go)


def build_go_embeddings(gene_order: list, gene_to_go: dict, go_embeds: dict) -> tuple:
    """
    Build per-gene GO vectors via sum pooling of anc2vec embeddings.

    For each gene, sums the anc2vec vectors of all its annotated GO terms.
    Genes with no annotations or no anc2vec coverage get zero vectors.
    """
    go_dim = next(iter(go_embeds.values())).shape[0]
    go_matrix = np.zeros((len(gene_order), go_dim), dtype=np.float32)

    genes_with_go = 0
    total_terms_used = 0

    for i, gene in enumerate(tqdm(gene_order, desc="Building GO vectors")):
        go_terms = gene_to_go.get(gene, set())
        if not go_terms:
            continue

        # Sum pool anc2vec vectors for all annotated GO terms
        vec = np.zeros(go_dim, dtype=np.float32)
        terms_found = 0
        for term in go_terms:
            if term in go_embeds:
                vec += go_embeds[term].astype(np.float32)
                terms_found += 1

        if terms_found > 0:
            go_matrix[i] = vec
            genes_with_go += 1
            total_terms_used += terms_found

    print(f"GO coverage: {genes_with_go}/{len(gene_order)} genes "
          f"({100*genes_with_go/len(gene_order):.1f}%)")
    print(f"Total GO terms mapped: {total_terms_used}")
    return go_matrix, genes_with_go


def standardize(matrix: np.ndarray) -> np.ndarray:
    """
    Per-feature z-score standardization, excluding zero-vector rows from
    mean/std computation. Same pattern as generate_all_genes_esm.py L431-438.
    """
    non_zero_mask = np.any(matrix != 0, axis=1)
    if np.any(non_zero_mask):
        non_zero = matrix[non_zero_mask]
        mean = non_zero.mean(axis=0, keepdims=True)
        std = non_zero.std(axis=0, keepdims=True) + 1e-8
        matrix[non_zero_mask] = (non_zero - mean) / std
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Combine ESM embeddings with anc2vec GO embeddings"
    )
    parser.add_argument(
        "--esm_embeddings", type=str, required=True,
        help="Path to ESM embeddings .pt file (from generate_all_genes_esm.py)"
    )
    parser.add_argument(
        "--gaf", type=str, required=True,
        help="Path to GAF annotation file (e.g. goa_human.gaf.gz)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for combined embeddings (.pt file)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="../data/cache",
        help="Directory to cache anc2vec download"
    )
    parser.add_argument(
        "--anc2vec_path", type=str, default=None,
        help="Path to local anc2vec embeddings.npz (skips download)"
    )
    args = parser.parse_args()

    # 1. Load anc2vec GO embeddings (local path or download)
    if args.anc2vec_path:
        go_embeds = load_anc2vec(args.anc2vec_path)
    else:
        anc2vec_cache = Path(args.cache_dir) / "anc2vec_embeddings.npz"
        go_embeds = download_anc2vec(str(anc2vec_cache))

    # 2. Load existing ESM embeddings (already standardized)
    print(f"\nLoading ESM embeddings from {args.esm_embeddings}...")
    esm_data = torch.load(args.esm_embeddings, map_location="cpu", weights_only=False)
    esm_matrix = esm_data["embeddings"].numpy()
    gene_order = esm_data["gene_order"]
    esm_dim = esm_matrix.shape[1]
    print(f"ESM: {esm_matrix.shape[0]} genes x {esm_dim}-dim")

    # 3. Parse GAF annotations
    print(f"\nParsing GAF: {args.gaf}")
    gene_to_go = parse_gaf(args.gaf)

    # 4. Build per-gene GO vectors (sum pooling)
    print("\nBuilding GO embeddings (sum pooling)...")
    go_matrix, num_genes_with_go = build_go_embeddings(gene_order, gene_to_go, go_embeds)
    go_dim = go_matrix.shape[1]

    # 5. Standardize GO embeddings independently
    print("\nStandardizing GO embeddings (zero mean, unit variance per feature)...")
    go_matrix = standardize(go_matrix)

    # 6. Concatenate [ESM ; GO]
    combined = np.concatenate([esm_matrix, go_matrix], axis=1)
    combined_tensor = torch.from_numpy(combined).float()
    print(f"\nCombined shape: {combined_tensor.shape}")

    # 7. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "embeddings": combined_tensor,
        "gene_order": gene_order,
        "esm_dim": esm_dim,
        "go_dim": go_dim,
        "standardized": True,
        "go_source": "anc2vec",
        "go_pooling": "sum",
        "num_genes": len(gene_order),
        "num_genes_with_go": num_genes_with_go,
    }, output_path)

    print(f"\nSaved combined embeddings to {output_path}")
    print(f"  Shape: {combined_tensor.shape}")
    print(f"  ESM dim: {esm_dim}, GO dim: {go_dim}")
    print(f"  Genes with GO coverage: {num_genes_with_go}/{len(gene_order)}")


if __name__ == "__main__":
    main()
