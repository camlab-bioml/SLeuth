#!/usr/bin/env python3
"""
Simple two-layer GNN for edge prediction on Adamson data.

Architecture:
  - Node features: identity (one-hot)
  - Conv adjacency: user-specified matrix (with test edges masked to 0)
  - Target: user-specified matrix (with test edges masked to 0 during training)
  - Two GCN layers + bilinear decoder
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Hyperparameters
# =============================================================================
HIDDEN_DIM = 128      # Hidden layer dimension
EMBED_DIM = 64        # Embedding dimension (output of GCN)
DROPOUT = 0.2         # Dropout rate
LEAKY_RELU_SLOPE = 0.2  # LeakyReLU negative slope

# =============================================================================
# Data Configuration
# =============================================================================
ALL_MATRICES = ['gamma', 'go_bp', 'go_cc', 'ppi']


# =============================================================================
# Graph Operations
# =============================================================================

def normalized_laplacian(adj: np.ndarray, make_positive: bool = True) -> np.ndarray:
    """
    Normalized graph Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}

    Args:
        adj: Adjacency matrix (can have negative values if standardized)
        make_positive: If True, shift matrix to be non-negative before normalization
    """
    adj = np.array(adj, dtype=np.float64)
    adj[~np.isfinite(adj)] = 0.0

    # For standardized matrices with negative values, shift to positive
    if make_positive and adj.min() < 0:
        adj = adj - adj.min()

    # Set diagonal to 0 (no self-loops in Laplacian)
    np.fill_diagonal(adj, 0)

    # Compute degree matrix
    rowsum = adj.sum(1)
    d_inv_sqrt = np.zeros_like(rowsum)
    mask = rowsum > 0
    d_inv_sqrt[mask] = rowsum[mask] ** -0.5
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    norm_adj[~np.isfinite(norm_adj)] = 0.0

    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    n = adj.shape[0]
    L_norm = np.eye(n) - norm_adj

    return L_norm


# =============================================================================
# Model Components
# =============================================================================

class GCNLayer(nn.Module):
    """Single GCN layer: H' = LeakyReLU(A_norm @ H @ W)"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = adj @ x
        return F.leaky_relu(x, LEAKY_RELU_SLOPE)


class EdgeDecoder(nn.Module):
    """
    Bilinear decoder with edge features:
    R_ij = z_i^T W z_j + sum_k(alpha_k * E_k[i,j])

    Edge features are combined via learnable weights.
    """

    def __init__(self, embed_dim: int, n_edge_features: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Learnable weights for edge features
        self.n_edge_features = n_edge_features
        if n_edge_features > 0:
            self.edge_weights = nn.Parameter(torch.zeros(n_edge_features))

    def forward(self, z: torch.Tensor, edge_features: List[torch.Tensor] = None) -> torch.Tensor:
        # Bilinear term: z_i^T W z_j
        out = z @ self.weight @ z.t()

        # Add weighted edge features
        if edge_features is not None and self.n_edge_features > 0:
            for k, E in enumerate(edge_features):
                out = out + self.edge_weights[k] * E

        return out


class GNN(nn.Module):
    """Two-layer GNN for edge prediction with edge features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 embed_dim: int = 64, dropout: float = 0.2, n_edge_features: int = 0):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, embed_dim, dropout)
        self.decoder = EdgeDecoder(embed_dim, n_edge_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_features: List[torch.Tensor] = None) -> torch.Tensor:
        h = self.gcn1(x, adj)
        z = self.gcn2(h, adj)
        return self.decoder(z, edge_features)


# =============================================================================
# Data Loading
# =============================================================================

class AdamsonData:
    """Loader for Adamson preprocessed data."""

    def __init__(self, data_dir: str = '.', device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.data_dir = data_dir

        # Load gene list
        with open(f'{data_dir}/adamson_genes.txt') as f:
            self.genes = [line.strip() for line in f]
        self.num_nodes = len(self.genes)

        # Load all matrices (standardized, imputed)
        self.matrices = {
            'gamma': np.loadtxt(f'{data_dir}/adamson_gamma.txt', delimiter='\t'),
            'go_bp': np.loadtxt(f'{data_dir}/adamson_go_bp.txt', delimiter='\t'),
            'go_cc': np.loadtxt(f'{data_dir}/adamson_go_cc.txt', delimiter='\t'),
            'ppi': np.loadtxt(f'{data_dir}/adamson_ppi.txt', delimiter='\t'),
        }

        # Load gamma observation mask (1 = observed, 0 = imputed)
        try:
            self.gamma_mask = np.loadtxt(f'{data_dir}/adamson_gamma_mask.txt', delimiter='\t')
        except FileNotFoundError:
            self.gamma_mask = None

    def get_node_features(self) -> torch.Tensor:
        """Identity matrix as node features."""
        return torch.eye(self.num_nodes, dtype=torch.float32, device=self.device)

    def get_matrix(self, name: str) -> np.ndarray:
        """Get raw numpy matrix."""
        return self.matrices[name].copy()

    def mask_edges(self, matrix: np.ndarray, edge_indices: np.ndarray) -> np.ndarray:
        """Set specified edges to 0 in matrix (for train/test split)."""
        result = matrix.copy()
        for i, j in edge_indices:
            result[i, j] = 0
            result[j, i] = 0
        return result

    def get_adj_with_mask(self, name: str, mask_indices: np.ndarray = None) -> torch.Tensor:
        """Get normalized adjacency, optionally masking some edges to 0."""
        matrix = self.get_matrix(name)
        if mask_indices is not None and len(mask_indices) > 0:
            matrix = self.mask_edges(matrix, mask_indices)
        adj = normalized_laplacian(matrix)
        return torch.tensor(adj, dtype=torch.float32, device=self.device)

    def get_target_with_mask(self, name: str, mask_indices: np.ndarray = None) -> torch.Tensor:
        """Get target matrix, optionally masking some edges to 0."""
        matrix = self.get_matrix(name)
        if mask_indices is not None and len(mask_indices) > 0:
            matrix = self.mask_edges(matrix, mask_indices)
        return torch.tensor(matrix, dtype=torch.float32, device=self.device)

    def get_target(self, name: str) -> torch.Tensor:
        """Get full target matrix (for evaluation)."""
        return torch.tensor(self.matrices[name], dtype=torch.float32, device=self.device)

    def get_observed_edges(self, name: str) -> np.ndarray:
        """
        Get indices of observed edges (upper triangle).
        For gamma: uses mask file to identify observed entries
        For others: all upper triangle (fully observed after imputation)
        """
        n = self.num_nodes

        if name == 'gamma':
            # Use mask file if available, otherwise fallback to non-zero detection
            if self.gamma_mask is not None:
                indices = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if self.gamma_mask[i, j] == 1:
                            indices.append([i, j])
                return np.array(indices) if indices else np.array([]).reshape(0, 2)
            else:
                # Fallback: non-zero entries (fragile but works for current data)
                matrix = self.matrices[name]
                indices = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if matrix[i, j] != 0:
                            indices.append([i, j])
                return np.array(indices) if indices else np.array([]).reshape(0, 2)
        else:
            # All upper triangle pairs
            indices = []
            for i in range(n):
                for j in range(i + 1, n):
                    indices.append([i, j])
            return np.array(indices)

    def get_edge_feature_names(self, conv: str, target: str) -> List[str]:
        """Get matrix names excluding conv and target."""
        exclude = {conv, target}
        return [m for m in ALL_MATRICES if m not in exclude]

    def get_edge_features(self, conv: str, target: str) -> List[torch.Tensor]:
        """Get list of edge feature matrices (excluding conv and target)."""
        names = self.get_edge_feature_names(conv, target)
        return [torch.tensor(self.matrices[name], dtype=torch.float32, device=self.device)
                for name in names]
