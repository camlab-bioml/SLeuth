#!/usr/bin/env python3
"""
Load actual support views from the main study for Graph Transformer training.
Support views include:
1. GO-BP (Gene Ontology Biological Process) similarities
2. GO-CC (Gene Ontology Cellular Component) similarities  
3. PPI (Protein-Protein Interaction) network
"""

import numpy as np
import torch
import scipy.sparse
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path to import from code directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from utils import load_dense_feature_original, load_sparse_features_original


def create_edge_features_from_adjacency(adj_matrix: np.ndarray,
                                        edge_map: Dict[Tuple[int, int], int],
                                        feature_dim: int = 64) -> torch.Tensor:
    """
    Convert adjacency matrix values to edge features for graph transformer.
    
    Args:
        adj_matrix: Adjacency matrix (can be sparse or dense)
        edge_map: Dictionary mapping (i, j) pairs to edge indices
        feature_dim: Dimension of edge features to create
        
    Returns:
        Edge features tensor of shape (n_edges, feature_dim)
    """
    n_edges = len(edge_map)
    edge_features = np.zeros((n_edges, feature_dim))

    # Convert to dense if sparse
    if scipy.sparse.issparse(adj_matrix):
        adj_matrix = adj_matrix.toarray()

    # For each edge in edge_map, extract the adjacency value
    for (i, j), edge_idx in edge_map.items():
        # Get the edge weight/similarity value
        edge_value = adj_matrix[i, j]

        # Create feature vector for this edge
        # Option 1: Repeat the value across dimensions (simple)
        # edge_features[edge_idx, :] = edge_value

        # Option 2: Create a more informative feature vector
        # Use different transformations of the edge value
        if edge_value > 0:
            edge_features[edge_idx, 0] = edge_value  # Original value
            edge_features[edge_idx, 1] = np.log1p(edge_value)  # Log transform
            edge_features[edge_idx, 2] = edge_value**2  # Square
            edge_features[edge_idx, 3] = np.sqrt(edge_value)  # Square root

            # Add some statistical features
            edge_features[edge_idx, 4] = 1.0  # Edge exists indicator

            # Use remaining dimensions for a decaying pattern
            for k in range(5, min(20, feature_dim)):
                edge_features[edge_idx,
                              k] = edge_value * np.exp(-0.1 * (k - 5))

    return torch.FloatTensor(edge_features)


def load_biological_support_views(edge_map: Dict[Tuple[int, int], int],
                                  num_nodes: int,
                                  feature_dim: int = 64,
                                  knn: int = 5,
                                  nnSize: int = 30,
                                  data_dir: str = "../data/") -> torch.Tensor:
    """
    Load the three biological support views used in the main study.
    
    Args:
        edge_map: Dictionary mapping (i, j) pairs to edge indices
        num_nodes: Number of nodes in the graph
        feature_dim: Dimension of features for each support view
        knn: K-nearest neighbors for GO similarity matrices
        nnSize: Nearest neighbor size for GO similarity matrices
        data_dir: Directory containing the data files
        
    Returns:
        Support views tensor of shape (n_edges, 3, feature_dim)
    """
    print(f"\nLoading biological support views from main study...")

    # Load GO-BP similarities
    print("1. Loading GO-BP (Biological Process) similarities...")
    go_bp_path = os.path.join(data_dir, 'Human_GOsim.txt')
    go_bp = load_dense_feature_original(go_bp_path,
                                        knn=knn,
                                        nnSize=nnSize,
                                        num_nodes=num_nodes)
    go_bp_features = create_edge_features_from_adjacency(
        go_bp, edge_map, feature_dim)
    print(f"   GO-BP features shape: {go_bp_features.shape}")

    # Load GO-CC similarities
    print("2. Loading GO-CC (Cellular Component) similarities...")
    go_cc_path = os.path.join(data_dir, 'Human_GOsim_CC.txt')
    go_cc = load_dense_feature_original(go_cc_path,
                                        knn=knn,
                                        nnSize=nnSize,
                                        num_nodes=num_nodes)
    go_cc_features = create_edge_features_from_adjacency(
        go_cc, edge_map, feature_dim)
    print(f"   GO-CC features shape: {go_cc_features.shape}")

    # Load PPI network
    print("3. Loading PPI (Protein-Protein Interaction) network...")
    ppi_path = os.path.join(data_dir, 'biogrid_ppi_sparse.txt')
    ppi = load_sparse_features_original(ppi_path, num_nodes=num_nodes)
    ppi_features = create_edge_features_from_adjacency(ppi, edge_map,
                                                       feature_dim)
    print(f"   PPI features shape: {ppi_features.shape}")

    # Stack all support views
    support_views = torch.stack([go_bp_features, go_cc_features, ppi_features],
                                dim=1)
    print(f"\nCombined support views shape: {support_views.shape}")
    print(
        f"Format: (n_edges={support_views.shape[0]}, n_views={support_views.shape[1]}, feature_dim={support_views.shape[2]})"
    )

    # Print some statistics
    print("\nSupport view statistics:")
    for i, view_name in enumerate(['GO-BP', 'GO-CC', 'PPI']):
        view_data = support_views[:, i, :]
        non_zero_edges = (view_data.sum(dim=1) > 0).sum().item()
        print(
            f"   {view_name}: {non_zero_edges}/{support_views.shape[0]} edges with features ({non_zero_edges/support_views.shape[0]*100:.1f}%)"
        )

    return support_views


def prepare_support_views_from_study(
        adj_matrix: np.ndarray,
        num_nodes: int,
        support_view_dim: int = 64,
        knn: int = 5,
        nnSize: int = 30) -> Tuple[torch.Tensor, Dict]:
    """
    Prepare support views using the actual biological networks from the main study.
    This replaces the random initialization with real biological data.
    
    Args:
        adj_matrix: Graph adjacency matrix (for creating edge mapping)
        num_nodes: Number of nodes in the graph
        support_view_dim: Dimension of each support view
        knn: K-nearest neighbors for GO similarity matrices
        nnSize: Nearest neighbor size for GO similarity matrices
        
    Returns:
        support_views: Tensor of shape (n_edges, 3, support_view_dim)
        edge_map: Dictionary mapping (i, j) node pairs to edge indices
    """
    # First create edge mapping from adjacency matrix
    edge_map = {}
    edge_idx = 0

    # Find all edges in the adjacency matrix
    if scipy.sparse.issparse(adj_matrix):
        # For sparse matrix
        rows, cols = adj_matrix.nonzero()
        for i, j in zip(rows, cols):
            if i < j:  # Only store each edge once (undirected)
                edge_map[(i, j)] = edge_idx
                edge_idx += 1
    else:
        # For dense matrix
        for i in range(adj_matrix.shape[0]):
            for j in range(i + 1, adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    edge_map[(i, j)] = edge_idx
                    edge_idx += 1

    print(f"Created edge mapping with {len(edge_map)} edges")

    # Load biological support views
    support_views = load_biological_support_views(edge_map=edge_map,
                                                  num_nodes=num_nodes,
                                                  feature_dim=support_view_dim,
                                                  knn=knn,
                                                  nnSize=nnSize)

    return support_views, edge_map


if __name__ == "__main__":
    # Test loading support views
    print("Testing support view loading...")

    # Create a dummy adjacency matrix for testing
    num_nodes = 100
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix > 0.9).astype(float)
    adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix).T  # Make symmetric

    # Load support views
    support_views, edge_map = prepare_support_views_from_study(
        adj_matrix=adj_matrix, num_nodes=num_nodes, support_view_dim=64)

    print(f"\nTest complete!")
    print(f"Support views shape: {support_views.shape}")
    print(f"Number of edges: {len(edge_map)}")
