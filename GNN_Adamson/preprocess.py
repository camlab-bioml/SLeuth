#!/usr/bin/env python3
"""
Preprocess Adamson dataset and relational features for GNN training.

Steps:
  1. Load raw data with NaN for missing entries
  2. Symmetrize: Q = (Q + Q.T)/2 handling NaN properly
  3. Standardize continuous matrices (0 mean, 1 sd) using non-NaN entries
  4. Impute NaN:
     - Continuous: impute with 0 (after standardization)
     - Binary: impute using naive edge probability per node

Outputs:
  - adamson_genes.txt: gene names (N rows)
  - adamson_gamma.txt: N x N gamma scores (continuous, standardized)
  - adamson_go_bp.txt: N x N GO BP similarity (continuous, standardized)
  - adamson_go_cc.txt: N x N GO CC similarity (continuous, standardized)
  - adamson_ppi.txt: N x N PPI adjacency (binary)
"""

import numpy as np
from collections import defaultdict

# =============================================================================
# Hyperparameters
# =============================================================================
KNN_SIZE = 45             # Number of nearest neighbors for GO similarity sparsification
BINARY_IMPUTE_THRESH = 0.5  # Threshold for binary imputation (if prob > thresh, impute 1)

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = '../data'
OUTPUT_DIR = '.'


def symmetrize_with_nan(Q):
    """
    Symmetrize matrix handling NaN:
    - If both Q[i,j] and Q[j,i] are non-NaN: average them
    - If one is NaN: use the non-NaN value
    - If both are NaN: keep NaN
    """
    n = Q.shape[0]
    result = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            a, b = Q[i, j], Q[j, i]
            if np.isnan(a) and np.isnan(b):
                val = np.nan
            elif np.isnan(a):
                val = b
            elif np.isnan(b):
                val = a
            else:
                val = (a + b) / 2
            result[i, j] = val
            result[j, i] = val

    return result


def standardize_matrix(Q):
    """
    Standardize continuous matrix to 0 mean, 1 sd using non-NaN entries.
    Diagonal is excluded from mean/std calculation.
    """
    # Get off-diagonal non-NaN values
    mask = ~np.isnan(Q)
    np.fill_diagonal(mask, False)  # Exclude diagonal
    vals = Q[mask]

    if len(vals) == 0:
        return Q

    mean = np.nanmean(vals)
    std = np.nanstd(vals)

    if std == 0 or np.isnan(std):
        std = 1.0

    result = (Q - mean) / std
    np.fill_diagonal(result, 0)  # Set diagonal to 0
    return result


def impute_continuous(Q):
    """Impute NaN with 0 for continuous matrices (after standardization)."""
    result = Q.copy()
    result[np.isnan(result)] = 0
    return result


def impute_binary_naive(Q):
    """
    Impute NaN in binary matrix using naive edge probability.
    For each node, calculate: p_i = (# non-NaN edges) / (N-1)
    Impute NaN entries: 1 if p_i > 0.5, else 0
    """
    n = Q.shape[0]
    result = Q.copy()

    # Calculate edge probability per node
    node_probs = np.zeros(n)
    for i in range(n):
        row = Q[i, :]
        non_nan_mask = ~np.isnan(row)
        non_nan_mask[i] = False  # Exclude self
        if np.sum(non_nan_mask) > 0:
            node_probs[i] = np.nansum(row[non_nan_mask]) / np.sum(non_nan_mask)

    # Impute NaN entries
    for i in range(n):
        for j in range(n):
            if i != j and np.isnan(result[i, j]):
                # Use average of both nodes' probabilities
                avg_prob = (node_probs[i] + node_probs[j]) / 2
                result[i, j] = 1 if avg_prob > BINARY_IMPUTE_THRESH else 0

    np.fill_diagonal(result, 0)
    return result


def load_adamson_pairs(filepath):
    """Load Adamson gene pairs, average duplicates, return as dict with NaN for missing."""
    pairs = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                g1, g2, score = parts[0].upper(), parts[1].upper(), float(parts[2])
                key = tuple(sorted([g1, g2]))
                pairs[key].append(score)

    # Average duplicates
    return {k: np.mean(v) for k, v in pairs.items()}


def build_matrix_from_pairs(pairs, genes, default=np.nan):
    """Build NxN matrix from gene pairs dict, fill missing with default."""
    n = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    matrix = np.full((n, n), default)
    np.fill_diagonal(matrix, 0)  # Self-loops = 0

    for (g1, g2), val in pairs.items():
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            matrix[i, j] = val
            matrix[j, i] = val

    return matrix


def load_full_gene_list(filepath):
    """Load gene list for index mapping."""
    with open(filepath, 'r') as f:
        return [line.strip().upper() for line in f]


def load_dense_feature_matrix(filepath, n_genes, nn_size=KNN_SIZE):
    """
    Load GO similarity (upper triangular format).
    Returns matrix with NaN for missing entries.
    """
    matrix = np.full((n_genes, n_genes), np.nan)
    np.fill_diagonal(matrix, 0)

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_genes:
                break
            parts = line.strip().split('\t')
            for j, val in enumerate(parts):
                col_idx = i + j + 1
                if val and col_idx < n_genes:
                    matrix[i, col_idx] = float(val)

    # Symmetrize
    matrix = symmetrize_with_nan(matrix)

    # Apply KNN sparsification (keep top-k per row)
    matrix = build_knn_matrix(matrix, nn_size)

    # Symmetrize again after KNN
    matrix = symmetrize_with_nan(matrix)

    return matrix


def build_knn_matrix(S, nn_size):
    """Keep top-k neighbors per node, set others to NaN."""
    n = S.shape[0]
    result = np.full((n, n), np.nan)
    np.fill_diagonal(result, 0)

    for i in range(n):
        row = S[i, :].copy()
        row[i] = -np.inf  # Exclude self
        # Get indices of non-NaN values
        valid_idx = np.where(~np.isnan(row))[0]
        if len(valid_idx) > 0:
            # Sort by value, keep top-k
            sorted_idx = valid_idx[np.argsort(row[valid_idx])[::-1]]
            topk_idx = sorted_idx[:min(nn_size, len(sorted_idx))]
            for j in topk_idx:
                result[i, j] = S[i, j]

    return result


def load_sparse_ppi_matrix(filepath, n_genes):
    """Load PPI as binary matrix with NaN for unknown pairs."""
    # Start with NaN (unknown)
    matrix = np.full((n_genes, n_genes), np.nan)
    np.fill_diagonal(matrix, 0)

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                r, c = int(parts[0]), int(parts[1])
                if r < n_genes and c < n_genes:
                    matrix[r, c] = 1
                    matrix[c, r] = 1

    # For PPI, observed edges are 1, but we don't know about non-edges
    # Keep NaN for truly unknown pairs
    return matrix


def filter_matrix_to_genes(full_matrix, full_genes, target_genes):
    """Extract submatrix for target genes."""
    full_gene_to_idx = {g: i for i, g in enumerate(full_genes)}
    n = len(target_genes)
    filtered = np.full((n, n), np.nan)

    for i, g1 in enumerate(target_genes):
        for j, g2 in enumerate(target_genes):
            if g1 in full_gene_to_idx and g2 in full_gene_to_idx:
                orig_i, orig_j = full_gene_to_idx[g1], full_gene_to_idx[g2]
                filtered[i, j] = full_matrix[orig_i, orig_j]

    return filtered


def save_matrix(matrix, filepath):
    np.savetxt(filepath, matrix, delimiter='\t', fmt='%.6f')


def save_gene_list(genes, filepath):
    with open(filepath, 'w') as f:
        for g in genes:
            f.write(f"{g}\n")


def print_matrix_stats(name, matrix, is_binary=False):
    """Print statistics for a matrix."""
    mask = ~np.isnan(matrix)
    np.fill_diagonal(mask, False)
    n_total = mask.size - matrix.shape[0]  # Exclude diagonal
    n_observed = np.sum(mask)
    n_missing = n_total - n_observed

    if is_binary:
        n_edges = int(np.nansum(matrix[mask]) / 2)
        print(f"  {name}: {n_edges} edges, {n_missing} missing entries")
    else:
        vals = matrix[mask]
        if len(vals) > 0:
            print(f"  {name}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}], {n_missing} missing")
        else:
            print(f"  {name}: all missing")


def main():
    print("=== Preprocessing Adamson data ===\n")

    # 1. Load Adamson pairs and extract unique genes
    print("Loading Adamson pairs...")
    adamson_pairs = load_adamson_pairs(f'{DATA_DIR}/SL_Adamson_gamma_thresholding.txt')
    print(f"  {len(adamson_pairs)} unique gene pairs")

    genes_set = set()
    for g1, g2 in adamson_pairs.keys():
        genes_set.add(g1)
        genes_set.add(g2)
    adamson_genes = sorted(list(genes_set))
    n = len(adamson_genes)
    print(f"  {n} unique genes")

    # 2. Build gamma matrix (continuous)
    print("\nBuilding gamma matrix...")
    gamma_raw = build_matrix_from_pairs(adamson_pairs, adamson_genes, default=np.nan)

    # 3. Load full gene list and feature matrices
    print("\nLoading feature matrices...")
    full_genes = load_full_gene_list(f'{DATA_DIR}/List_Proteins_in_SL.txt')
    n_full = len(full_genes)

    go_bp_full = load_dense_feature_matrix(f'{DATA_DIR}/Human_GOsim.txt', n_full)
    go_cc_full = load_dense_feature_matrix(f'{DATA_DIR}/Human_GOsim_CC.txt', n_full)
    ppi_full = load_sparse_ppi_matrix(f'{DATA_DIR}/biogrid_ppi_sparse.txt', n_full)

    # 4. Filter to Adamson genes
    print("\nFiltering to Adamson genes...")
    go_bp_raw = filter_matrix_to_genes(go_bp_full, full_genes, adamson_genes)
    go_cc_raw = filter_matrix_to_genes(go_cc_full, full_genes, adamson_genes)
    ppi_raw = filter_matrix_to_genes(ppi_full, full_genes, adamson_genes)

    # 5. Process matrices
    print("\n=== Raw data stats ===")
    print_matrix_stats("gamma", gamma_raw)
    print_matrix_stats("go_bp", go_bp_raw)
    print_matrix_stats("go_cc", go_cc_raw)
    print_matrix_stats("ppi", ppi_raw, is_binary=True)

    print("\nProcessing matrices...")

    # Symmetrize (already done during loading, but ensure)
    gamma_sym = symmetrize_with_nan(gamma_raw)
    go_bp_sym = symmetrize_with_nan(go_bp_raw)
    go_cc_sym = symmetrize_with_nan(go_cc_raw)
    ppi_sym = symmetrize_with_nan(ppi_raw)

    # Standardize continuous matrices
    print("  Standardizing continuous matrices...")
    gamma_std = standardize_matrix(gamma_sym)
    go_bp_std = standardize_matrix(go_bp_sym)
    go_cc_std = standardize_matrix(go_cc_sym)

    # Impute missing values
    print("  Imputing missing values...")
    gamma_final = impute_continuous(gamma_std)
    go_bp_final = impute_continuous(go_bp_std)
    go_cc_final = impute_continuous(go_cc_std)
    ppi_final = impute_binary_naive(ppi_sym)

    # 6. Save outputs
    print("\nSaving preprocessed data...")
    save_gene_list(adamson_genes, f'{OUTPUT_DIR}/adamson_genes.txt')
    save_matrix(gamma_final, f'{OUTPUT_DIR}/adamson_gamma.txt')
    save_matrix(go_bp_final, f'{OUTPUT_DIR}/adamson_go_bp.txt')
    save_matrix(go_cc_final, f'{OUTPUT_DIR}/adamson_go_cc.txt')
    save_matrix(ppi_final, f'{OUTPUT_DIR}/adamson_ppi.txt')

    # Save gamma observation mask (1 = observed, 0 = imputed)
    gamma_mask = (~np.isnan(gamma_sym)).astype(float)
    np.fill_diagonal(gamma_mask, 0)
    save_matrix(gamma_mask, f'{OUTPUT_DIR}/adamson_gamma_mask.txt')

    # Print final stats
    print("\n=== Final processed stats ===")
    print_matrix_stats("gamma (standardized)", gamma_final)
    print_matrix_stats("go_bp (standardized)", go_bp_final)
    print_matrix_stats("go_cc (standardized)", go_cc_final)
    print_matrix_stats("ppi (binary)", ppi_final, is_binary=True)

    print(f"\nOutput files saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
