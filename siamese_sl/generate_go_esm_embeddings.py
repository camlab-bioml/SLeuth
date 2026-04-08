#!/usr/bin/env python3
"""
Generate GO embeddings (GO-only or combined ESM+GO).

Builds anc2vec GO embeddings (200-dim, sum-pooled over annotated GO terms)
and optionally concatenates with ESM embeddings.

anc2vec Reference:
  Edera et al. "anc2vec: embedding Gene Ontology terms by preserving
  ancestors relationships" Briefings in Bioinformatics, 2022.

Usage:
    # GO-only (200-dim)
    python generate_go_esm_embeddings.py --go_only \
        --esm_embeddings ../data/all_genes_esm.pt \
        --gaf ../uniprot_GO/goa_human.gaf.gz \
        --output ../data/all_genes_go.pt

    # Combined ESM+GO (1480-dim)
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

from gene_name_utils import get_mapper

ANC2VEC_URL = "https://github.com/aedera/anc2vec/raw/main/anc2vec/data/embeddings.npz"


def load_anc2vec(npz_path: str) -> dict:
    """Load anc2vec GO term embeddings from a local .npz file."""
    print(f"Loading anc2vec embeddings from {npz_path}...")
    npz = np.load(npz_path, allow_pickle=True)
    go_embeds = npz["embds"].item()  # dict: GO term ID -> 200-dim vector
    print(
        f"Loaded {len(go_embeds)} GO term embeddings (dim={next(iter(go_embeds.values())).shape[0]})"
    )
    return go_embeds


def download_anc2vec(cache_path: str) -> dict:
    """Download anc2vec GO term embeddings, caching locally."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        print(f"Downloading anc2vec embeddings to {cache_path}...")
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
    not_qualified_excluded = 0

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
            if any(q.strip() == "NOT" for q in qualifier.upper().split("|")):
                not_qualified_excluded += 1
                continue
            gene_symbol = fields[2]
            go_term = fields[4]
            gene_to_go[gene_symbol].add(go_term)

    # Normalize gene symbols and convert to Entrez IDs
    mapper = get_mapper()
    entrez_go = defaultdict(set)
    n_unmapped = 0
    for gene, terms in gene_to_go.items():
        eid = mapper.symbol_to_entrez(gene)
        if eid:
            entrez_go[eid].update(terms)
        else:
            n_unmapped += 1

    print(f"Parsed GAF: {len(gene_to_go)} genes with GO annotations")
    print(f"  Mapped to Entrez IDs: {len(entrez_go)} genes")
    if n_unmapped:
        print(f"  Unmapped (no Entrez ID): {n_unmapped}")
    if not_qualified_excluded:
        print(f"  Excluded {not_qualified_excluded} NOT-qualified annotations")
    return dict(entrez_go)


def build_go_embeddings(gene_order: list, gene_to_go: dict,
                        go_embeds: dict) -> tuple:
    """
    Build per-gene GO vectors via sum pooling of anc2vec embeddings.

    For each gene, sums the anc2vec vectors of all its annotated GO terms.
    Genes with no annotations or no anc2vec coverage get NaN vectors
    (Huber-imputed at load time by data_loader.py).
    """
    go_dim = next(iter(go_embeds.values())).shape[0]
    go_matrix = np.full((len(gene_order), go_dim), np.nan, dtype=np.float32)

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
    Global per-feature z-score standardization, excluding NaN and zero-vector
    rows from mean/std computation. Applied at embedding generation time.
    Note: data_loader.py loads raw_embeddings and applies its own
    normalization pipeline (impute, optional PCA, MAD normalize). This
    function is retained for potential direct consumers that bypass
    data_loader.py.

    Returns a new array; does not mutate the input.
    """
    result = matrix.copy()
    # Exclude NaN rows (missing genes) first, then zero-vector rows
    valid_mask = ~np.isnan(result).any(axis=1) & np.any(result != 0, axis=1)
    if np.any(valid_mask):
        valid = result[valid_mask]
        mean = valid.mean(axis=0, keepdims=True)
        std = valid.std(axis=0, keepdims=True) + 1e-8
        result[valid_mask] = (valid - mean) / std
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Combine ESM embeddings with anc2vec GO embeddings")
    parser.add_argument(
        "--esm_embeddings",
        type=str,
        required=True,
        help="Path to ESM embeddings .pt file (from generate_all_genes_esm.py)"
    )
    parser.add_argument(
        "--gaf",
        type=str,
        required=True,
        help="Path to GAF annotation file (e.g. goa_human.gaf.gz)")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output path for combined embeddings (.pt file)")
    parser.add_argument("--cache_dir",
                        type=str,
                        default="../data/cache",
                        help="Directory to cache anc2vec download")
    parser.add_argument(
        "--anc2vec_path",
        type=str,
        default=None,
        help="Path to local anc2vec embeddings.npz (skips download)")
    parser.add_argument("--go_only",
                        action="store_true",
                        help="Output GO-only embeddings (200-dim) without ESM")
    args = parser.parse_args()

    # 1. Load anc2vec GO embeddings (local path or download)
    if args.anc2vec_path:
        go_embeds = load_anc2vec(args.anc2vec_path)
    else:
        anc2vec_cache = Path(args.cache_dir) / "anc2vec_embeddings.npz"
        go_embeds = download_anc2vec(str(anc2vec_cache))

    # 2. Load ESM file for gene_order (and ESM embeddings if not --go_only)
    print(f"\nLoading ESM data from {args.esm_embeddings}...")
    esm_data = torch.load(args.esm_embeddings,
                          map_location="cpu",
                          weights_only=False)
    gene_order = esm_data["gene_order"]
    print(f"Gene universe: {len(gene_order)} genes")

    # 3. Parse GAF annotations
    print(f"\nParsing GAF: {args.gaf}")
    gene_to_go = parse_gaf(args.gaf)

    # 4. Build per-gene GO vectors (sum pooling)
    print("\nBuilding GO embeddings (sum pooling)...")
    go_matrix, num_genes_with_go = build_go_embeddings(gene_order, gene_to_go,
                                                       go_embeds)
    go_dim = go_matrix.shape[1]

    # 6. Build output embeddings
    # No standardization in either branch: data_loader.py loads raw_embeddings
    # and applies its own normalization pipeline (impute, optional PCA, MAD normalize).
    output_path = Path(args.output)

    if args.go_only:
        # GO-only: 200-dim, no standardization (single-modal)
        out_tensor = torch.from_numpy(go_matrix).float()
        # In go_only mode, no standardization applied, so raw = embeddings
        # (torch.save deduplicates same-object references)
        print(f"\nGO-only shape: {out_tensor.shape}")

        torch.save(
            {
                "embeddings": out_tensor,
                "raw_embeddings":
                out_tensor,  # identical (no standardization in go_only mode)
                "gene_order": gene_order,
                "go_dim": go_dim,
                "standardized": False,
                "go_source": "anc2vec",
                "go_pooling": "sum",
                "num_genes": len(gene_order),
                "num_genes_with_go": num_genes_with_go,
            },
            output_path)

        print(f"\nSaved GO-only embeddings to {output_path}")
        print(f"  Shape: {out_tensor.shape}")
        print(
            f"  Genes with GO coverage: {num_genes_with_go}/{len(gene_order)}")
    else:
        # Combined [ESM ; GO]: 1480-dim
        # No standardization — data_loader.py loads raw_embeddings and
        # applies its own normalization pipeline (impute, optional PCA,
        # MAD normalize).
        esm_matrix = esm_data["embeddings"].numpy()
        esm_dim = esm_matrix.shape[1]
        combined = np.concatenate([esm_matrix, go_matrix], axis=1)

        combined_tensor = torch.from_numpy(combined).float()
        print(f"\nCombined shape: {combined_tensor.shape}")

        torch.save(
            {
                "embeddings": combined_tensor,
                "raw_embeddings":
                combined_tensor,  # identical (no standardization)
                "gene_order": gene_order,
                "component_dims": [esm_dim, go_dim],
                "esm_dim": esm_dim,
                "go_dim": go_dim,
                "embedding_type": "esm2+go",
                "standardized": False,
                "go_source": "anc2vec",
                "go_pooling": "sum",
                "num_genes": len(gene_order),
                "num_genes_with_go": num_genes_with_go,
            },
            output_path)

        print(f"\nSaved combined embeddings to {output_path}")
        print(f"  Shape: {combined_tensor.shape}")
        print(f"  ESM dim: {esm_dim}, GO dim: {go_dim}")
        print(
            f"  Genes with GO coverage: {num_genes_with_go}/{len(gene_order)}")


if __name__ == "__main__":
    main()
