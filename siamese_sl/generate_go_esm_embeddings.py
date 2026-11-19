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
        --esm_embeddings ../data/all_genes_esm2.pt \
        --gaf ../data/embeddings_cache/goa_human.gaf.gz \
        --output ../data/all_genes_go.pt

    # Combined ESM+GO (1480-dim)
    python generate_go_esm_embeddings.py \
        --esm_embeddings ../data/all_genes_esm2.pt \
        --gaf ../data/embeddings_cache/goa_human.gaf.gz \
        --output ../data/all_genes_esm_go.pt
"""

import gzip
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from gene_name_utils import get_mapper
from generate_all_genes_esm import EMBEDDING_GENERATOR_VERSION

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


def _is_npz(path: Path) -> bool:
    """True if *path* is a non-empty file with the zip (.npz) magic."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def download_anc2vec(cache_path: str) -> dict:
    """Download anc2vec GO term embeddings, caching locally.

    Downloads to ``<cache_path>.part`` and renames only a complete .npz, so
    an interrupted or HTML download is never reused as the cache.
    """
    import requests
    cache_path = Path(cache_path)
    if cache_path.exists() and not _is_npz(cache_path):
        print(f"Discarding invalid cached anc2vec: {cache_path}")
        cache_path.unlink()
    if not cache_path.exists():
        print(f"Downloading anc2vec embeddings to {cache_path}...")
        part = cache_path.with_name(cache_path.name + ".part")
        resp = requests.get(ANC2VEC_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        if not _is_npz(part):
            part.unlink()
            raise RuntimeError(f"anc2vec download from {ANC2VEC_URL} is not "
                               f"an .npz file")
        part.replace(cache_path)
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
    # gene_name_utils falls back to "custom corrections only" when the HGNC
    # table is missing and cannot be downloaded, or holds no parsable row.
    # Every GAF symbol then maps to None and the output would be an all-NaN
    # matrix stamped as a current build, so stop here instead.
    if not mapper.stats["entrez_ids"]:
        raise RuntimeError(
            f"HGNC table not loaded: hgnc_complete_set.txt in "
            f"{Path(mapper._cache_dir).resolve()} is missing and could not "
            f"be downloaded, or holds no Entrez ids, so no GAF symbol can be "
            f"mapped. Refusing to write an all-NaN GO embedding (the table "
            f"is tracked in git as data/cache/hgnc_complete_set.txt; run "
            f"from siamese_sl/).")
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
    (imputed per fold by data_loader.SLDataManager.get_fold_embeddings, fit
    on that fold's _fold_fit_mask rows; mean under the default pca_method
    plain, Huber under robust/svd).
    """
    go_dim = next(iter(go_embeds.values())).shape[0]
    go_matrix = np.full((len(gene_order), go_dim), np.nan, dtype=np.float32)

    genes_with_go = 0
    total_terms_used = 0

    for i, gene in enumerate(tqdm(gene_order, desc="Building GO vectors")):
        go_terms = gene_to_go.get(gene, set())
        if not go_terms:
            continue

        # Sum pool anc2vec vectors for all annotated GO terms, in sorted
        # order: set order follows the per-process string hash seed, and a
        # float32 sum depends on its order
        vec = np.zeros(go_dim, dtype=np.float32)
        terms_found = 0
        for term in sorted(go_terms):
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
        help="Path to GAF annotation file (the pipeline's, and the release "
        "go2vec/onto2vec/kg_complex read: ../data/embeddings_cache/"
        "goa_human.gaf.gz)")
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

    # 5. Build output embeddings
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
                "generator_version": EMBEDDING_GENERATOR_VERSION,
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

        # The ESM half is copied from --esm_embeddings, so the output is a
        # current build only if that file is one
        esm_version = esm_data.get("generator_version")
        generator_version = EMBEDDING_GENERATOR_VERSION
        if esm_version != EMBEDDING_GENERATOR_VERSION:
            generator_version = (f"{esm_version or 'unknown'} (ESM part from "
                                 f"{args.esm_embeddings})")
            print(f"WARNING: {args.esm_embeddings} was not built by this "
                  f"generator version ({EMBEDDING_GENERATOR_VERSION}); the "
                  f"output is stamped generator_version="
                  f"{generator_version!r}")

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
                "generator_version": generator_version,
            },
            output_path)

        print(f"\nSaved combined embeddings to {output_path}")
        print(f"  Shape: {combined_tensor.shape}")
        print(f"  ESM dim: {esm_dim}, GO dim: {go_dim}")
        print(
            f"  Genes with GO coverage: {num_genes_with_go}/{len(gene_order)}")


if __name__ == "__main__":
    main()
