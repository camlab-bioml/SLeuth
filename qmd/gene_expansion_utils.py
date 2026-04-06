"""
Utility functions for expanding gene matrices to include all human genes.
Uses pre-fetched UniProt data to avoid API issues during analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Set
import os


def load_human_genes_cached(
        cache_file: str = '../data/all_human_genes_uniprot.txt') -> Set[str]:
    """
    Load pre-fetched human genes from cache file.
    If cache doesn't exist, returns empty set.
    """
    if not os.path.exists(cache_file):
        print(f"  Warning: Cache file {cache_file} not found")
        print(
            "  Run 'python code/fetch_human_genes_uniprot.py' to generate it")
        return set()

    genes = set()
    with open(cache_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                genes.add(line)

    return genes


def expand_gene_matrix_cached(
        matrix: np.ndarray,
        gene_names: List[str],
        gene_corrections: dict = None) -> Tuple[np.ndarray, List[str]]:
    """
    Expand a gene matrix to include all human genes using cached UniProt data.
    
    Args:
        matrix: Original N x N matrix
        gene_names: List of N gene names
        gene_corrections: Optional dictionary of gene name corrections
        
    Returns:
        Tuple of (expanded_matrix, expanded_gene_names)
    """
    print("Expanding gene matrix to include all human genes...")

    # Apply gene corrections if provided
    if gene_corrections:
        corrected_names = [
            gene_corrections.get(gene, gene) for gene in gene_names
        ]
    else:
        corrected_names = gene_names

    print(f"  Starting with {len(corrected_names)} genes from SL network")

    # Load cached human genes
    all_human_genes = load_human_genes_cached()

    if not all_human_genes:
        print("  No cached human genes available, using original matrix")
        return matrix, corrected_names

    print(f"  Loaded {len(all_human_genes):,} human genes from cache")

    # Find genes not in the SL network
    current_genes_set = set(corrected_names)
    new_genes = sorted(list(all_human_genes - current_genes_set))

    print(
        f"  Found {len(new_genes):,} additional human genes not in SL network")

    if len(new_genes) == 0:
        print("  No new genes to add")
        return matrix, corrected_names

    # Create expanded gene list and matrix
    expanded_names = corrected_names + new_genes
    n_original = len(corrected_names)
    n_expanded = len(expanded_names)

    # Initialize expanded matrix with NaN for new entries
    expanded_matrix = np.full((n_expanded, n_expanded), np.nan)
    expanded_matrix[:n_original, :n_original] = matrix

    print(
        f"  Matrix expanded from {n_original}x{n_original} to {n_expanded}x{n_expanded}"
    )

    return expanded_matrix, expanded_names


def create_gene_mapping(original_genes: List[str],
                        expanded_genes: List[str]) -> dict:
    """
    Create a mapping from gene names to indices in the expanded matrix.
    """
    return {gene: idx for idx, gene in enumerate(expanded_genes)}


def save_expanded_gene_list(expanded_genes: List[str],
                            original_count: int,
                            output_file: str = 'expanded_gene_list.txt'):
    """
    Save the expanded gene list with annotations.
    """
    with open(output_file, 'w') as f:
        f.write("# Expanded gene list for SL analysis\n")
        f.write(f"# Total genes: {len(expanded_genes)}\n")
        f.write(f"# Original SL genes: {original_count}\n")
        f.write(
            f"# Additional human genes: {len(expanded_genes) - original_count}\n\n"
        )

        f.write("# ORIGINAL SL NETWORK GENES (1-{})\n".format(original_count))
        for i, gene in enumerate(expanded_genes[:original_count], 1):
            f.write(f"{i}\t{gene}\n")

        f.write(
            f"\n# ADDITIONAL HUMAN GENES ({original_count+1}-{len(expanded_genes)})\n"
        )
        for i, gene in enumerate(expanded_genes[original_count:],
                                 original_count + 1):
            f.write(f"{i}\t{gene}\n")

    print(f"  Saved expanded gene list to {output_file}")
