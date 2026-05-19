#!/usr/bin/env python3
"""
Load data exactly as in SLMGAE main study.
Uses SL_Human_Approved.txt for positive edges and generates negative edges.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import random
from sklearn.model_selection import train_test_split


def load_gene_list(
        use_esm_filtered: bool = True) -> Tuple[List[str], Dict[str, int]]:
    """
    Load gene list G = {g₁,...,g_n} with mapping φ: G → [n].
    
    Args:
        use_esm_filtered: If True, n=6298 (genes with ESM embeddings)
                         If False, n=6375 (all genes)
    
    Returns:
        gene_list: Ordered list of Entrez Gene ID strings
        gene_to_idx: Bijection φ: Entrez ID → index
    """
    if use_esm_filtered:
        # Load gene order from ESM embeddings file
        import torch
        data = torch.load('../data/embeddings_esm2_t33_650M_UR50D_meanpool.pt',
                          map_location='cpu')
        gene_list = data['gene_order']
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
        print(
            f"Loaded {len(gene_list)} genes from ESM embeddings (filtered for valid sequences)"
        )
    else:
        # Load all genes from List_Proteins_in_SL.txt
        gene_list = []
        gene_to_idx = {}

        with open('../data/List_Proteins_in_SL.txt', 'r') as f:
            idx = 0
            for line in f:
                gene = line.strip()
                if gene:
                    gene_list.append(gene)
                    gene_to_idx[gene] = idx
                    idx += 1

        print(f"Loaded {len(gene_list)} genes from List_Proteins_in_SL.txt")

    return gene_list, gene_to_idx


def load_sl_edges(gene_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Load positive edges E⁺ = {(i,j): genes i,j are SL}.
    
    Args:
        gene_to_idx: Mapping φ: gene → index
        
    Returns:
        E⁺ as list of canonical edges (i,j) where i<j
    """
    pos_edges = []
    missing_genes = set()

    with open('../data/SL_Human_Approved.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                gene1, gene2 = parts[0], parts[1]

                if gene1 in gene_to_idx and gene2 in gene_to_idx:
                    idx1 = gene_to_idx[gene1]
                    idx2 = gene_to_idx[gene2]
                    # Canonical form: (i,j) where i<j for undirected graph
                    if idx1 != idx2:
                        pos_edges.append((min(idx1, idx2), max(idx1, idx2)))
                else:
                    if gene1 not in gene_to_idx:
                        missing_genes.add(gene1)
                    if gene2 not in gene_to_idx:
                        missing_genes.add(gene2)

    # Remove duplicates
    pos_edges = list(set(pos_edges))

    print(f"Loaded {len(pos_edges)} positive SL edges")
    if missing_genes:
        print(
            f"Warning: {len(missing_genes)} genes from SL pairs not found in gene list"
        )

    return pos_edges


def generate_negative_edges(
        pos_edges: List[Tuple[int, int]],
        num_genes: int,
        neg_to_pos_ratio: int = 1) -> List[Tuple[int, int]]:
    """
    Sample E⁻ ~ Uniform({(i,j): (i,j) ∉ E⁺}) with |E⁻| = r·|E⁺|.
    
    Args:
        pos_edges: E⁺ (positive edges)
        num_genes: n (number of nodes)
        neg_to_pos_ratio: r (sampling ratio)
        
    Returns:
        E⁻ sampled from n(n-1)/2 - |E⁺| possible edges
    """
    pos_edge_set = set(pos_edges)  # O(1) membership check
    num_neg_edges = len(pos_edges) * neg_to_pos_ratio  # |E⁻| = r·|E⁺|

    neg_edges = []
    attempts = 0
    max_attempts = num_neg_edges * 10  # Prevent infinite loop

    while len(neg_edges) < num_neg_edges and attempts < max_attempts:
        # Sample (i,j) ~ Uniform([n] × [n])
        gene1 = random.randint(0, num_genes - 1)
        gene2 = random.randint(0, num_genes - 1)

        if gene1 != gene2:
            edge = (min(gene1, gene2), max(gene1, gene2))

            # Check if this edge is not in positive set and not already in negative set
            if edge not in pos_edge_set and edge not in neg_edges:
                neg_edges.append(edge)

        attempts += 1

    print(
        f"Generated {len(neg_edges)} negative edges (ratio: {neg_to_pos_ratio}:1)"
    )

    return neg_edges


def load_slmgae_data(val_ratio: float = 0.1,
                     test_ratio: float = 0.15,
                     neg_to_pos_ratio: int = 1,
                     seed: int = 42,
                     use_esm_filtered: bool = True) -> Dict:
    """
    Load data: split E = E⁺ ∪ E⁻ into train/val/test.
    
    Args:
        val_ratio: |E_val|/|E| fraction
        test_ratio: |E_test|/|E| fraction
        neg_to_pos_ratio: r where |E⁻| = r·|E⁺|
        seed: Random seed for reproducibility
        use_esm_filtered: n=6298 (ESM) vs n=6375 (all)
        
    Returns:
        Dictionary containing:
        - gene_list: List of gene names
        - gene_to_idx: Gene name to index mapping
        - train_edges: Training edge tensor (2, n_edges)
        - train_labels: Training labels tensor
        - val_edges: Validation edge tensor
        - val_labels: Validation labels
        - test_edges: Test edge tensor
        - test_labels: Test labels
        - num_genes: Total number of genes
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Load gene list
    gene_list, gene_to_idx = load_gene_list(use_esm_filtered=use_esm_filtered)
    num_genes = len(gene_list)

    # Load positive edges
    pos_edges = load_sl_edges(gene_to_idx)

    # Generate negative edges
    neg_edges = generate_negative_edges(pos_edges, num_genes, neg_to_pos_ratio)

    # Combine all edges and create labels
    all_edges = pos_edges + neg_edges
    all_labels = [1.0] * len(pos_edges) + [0.0] * len(neg_edges)

    # Convert to numpy arrays for splitting
    edges_array = np.array(all_edges)
    labels_array = np.array(all_labels)

    # Split into train/val/test
    # First split off test set
    train_val_edges, test_edges, train_val_labels, test_labels = train_test_split(
        edges_array,
        labels_array,
        test_size=test_ratio,
        stratify=labels_array,
        random_state=seed)

    # Then split train_val into train and val
    val_size_adjusted = val_ratio / (1 - test_ratio)
    train_edges, val_edges, train_labels, val_labels = train_test_split(
        train_val_edges,
        train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=seed)

    # Convert to tensors
    train_edges_tensor = torch.tensor(train_edges.T, dtype=torch.long)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)
    val_edges_tensor = torch.tensor(val_edges.T, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float)
    test_edges_tensor = torch.tensor(test_edges.T, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float)

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total genes: {num_genes}")
    print(
        f"Train: {len(train_labels)} edges ({train_labels.sum():.0f} positive)"
    )
    print(f"Val: {len(val_labels)} edges ({val_labels.sum():.0f} positive)")
    print(f"Test: {len(test_labels)} edges ({test_labels.sum():.0f} positive)")

    return {
        'gene_list': gene_list,
        'gene_to_idx': gene_to_idx,
        'train_edges': train_edges_tensor,
        'train_labels': train_labels_tensor,
        'val_edges': val_edges_tensor,
        'val_labels': val_labels_tensor,
        'test_edges': test_edges_tensor,
        'test_labels': test_labels_tensor,
        'num_genes': num_genes
    }


def build_adjacency_from_edges(edges: torch.Tensor,
                               num_nodes: int) -> np.ndarray:
    """
    Build adjacency A ∈ {0,1}ⁿˣⁿ from edge list E.
    
    Args:
        edges: E as tensor (2, |E|) with [src_nodes; dst_nodes]
        num_nodes: n (graph size)
        
    Returns:
        A where A_ij = 1 iff (i,j) ∈ E
    """
    adj = np.zeros((num_nodes, num_nodes))  # Initialize A = 0

    for i in range(edges.shape[1]):
        u, v = edges[0, i].item(), edges[1, i].item()
        adj[u, v] = 1  # A_uv = 1
        adj[v, u] = 1  # A_vu = 1 (symmetric for undirected)

    return adj


if __name__ == "__main__":
    # Test loading
    data = load_slmgae_data()

    print("\nData loaded successfully!")
    print(f"Gene universe: {data['num_genes']} genes")
    print(f"First 10 genes: {data['gene_list'][:10]}")
