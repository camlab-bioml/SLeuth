#!/usr/bin/env python3
"""
Graph Transformer with weighted support views for synthetic lethality prediction.
Each graph linearization is treated as an independent training sample (data augmentation).
Support views are weighted by learnable non-negative parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class WeightedSupportViewAttention(nn.Module):
    """
    Multi-head attention that incorporates weighted support views as edge biases.
    Each support view has a learnable non-negative weight α_m, implemented as α_m = w_m².
    
    Math: Attention(Q,K,V) = softmax(QK^T/√d_k + B)V
    where B_{ij} = Σ_m α_m² · f_m(i,j) · W_proj, f_m is support view m
    """

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 n_support_views: int,
                 feature_dim_per_view: int,
                 dropout: float = 0.1,
                 undirected: bool = True):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.n_support_views = n_support_views
        self.undirected = undirected

        # Standard attention projections: Q=XW_q, K=XW_k, V=XW_v ∈ ℝ^{n×d}
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Learnable weights w_m ∈ ℝ^M where α_m = w_m² ensures non-negativity
        # Initialize w_m ~ N(0, 0.1²) so E[α_m] = E[w_m²] ≈ 0.01
        self.support_view_weights = nn.Parameter(
            torch.randn(n_support_views) * 0.1)

        # Project weighted support views to per-head scalar biases
        # Input: concatenated weighted support views, Output: bias per head
        self.support_view_proj = nn.Linear(feature_dim_per_view, n_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, support_views: Optional[torch.Tensor],
                edge_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Node representations (batch_size, seq_len, hidden_dim)
            support_views: Edge features (n_edges, n_support_views, view_dim_per_support)
            edge_indices: Edge positions in sequence (2, n_edges)
        Returns:
            Updated node representations (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values with shape (batch, n_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads,
                             self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads,
                             self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads,
                             self.d_k).transpose(1, 2)

        # Compute standard attention scores: S = QK^T/√d_k ∈ ℝ^{B×H×n×n}
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add weighted support view biases if available
        if support_views is not None and edge_indices is not None:
            # Non-negative weights: α_m = w_m² ∈ ℝ^M_≥0
            alpha_weights = self.support_view_weights**2

            # Compute α_m · f_m(e) for all edges e and views m
            # Shape: (E, M, D) where E=|edges|, M=|views|, D=feature_dim
            weighted_views = support_views * alpha_weights.view(1, -1, 1)

            # Aggregate: f_combined(e) = Σ_m α_m·f_m(e) ∈ ℝ^D
            combined_edge_features = weighted_views.sum(dim=1)

            # Project to per-head biases: (n_edges, n_heads)
            edge_biases = self.support_view_proj(combined_edge_features)

            # Create bias matrix: (n_heads, seq_len, seq_len)
            bias_matrix = torch.zeros(self.n_heads,
                                      seq_len,
                                      seq_len,
                                      device=scores.device)

            # Fill in biases for existing edges (vectorized)
            edge_i, edge_j = edge_indices
            bias_matrix[:, edge_i, edge_j] = edge_biases.t()

            # For undirected graphs, add symmetric biases
            if self.undirected:
                bias_matrix[:, edge_j, edge_i] = edge_biases.t()

            # Add biases to attention scores
            scores = scores + bias_matrix.unsqueeze(0)

        # Attention weights: A = softmax(S) where A_ij = exp(S_ij)/Σ_k exp(S_ik)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (batch, n_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)

        # Reshape and project output: (batch, seq_len, hidden_dim)
        context = context.transpose(1, 2).reshape(batch_size, seq_len,
                                                  self.hidden_dim)
        output = self.W_o(context)

        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with weighted support view attention."""

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 n_support_views: int,
                 feature_dim_per_view: int,
                 dropout: float = 0.1,
                 undirected: bool = True):
        super().__init__()

        # Multi-head attention with weighted support views
        self.attention = WeightedSupportViewAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_support_views=n_support_views,
            feature_dim_per_view=feature_dim_per_view,
            dropout=dropout,
            undirected=undirected)

        # FFN: x ↦ W_2·GELU(W_1·x) where W_1: d→4d, W_2: 4d→d
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, support_views: Optional[torch.Tensor],
                edge_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply attention and feed-forward with residual connections."""
        # x' = LayerNorm(x + Dropout(Attention(x)))
        attn_out = self.attention(x, support_views, edge_indices)
        x = self.norm1(x + self.dropout(attn_out))

        # x'' = LayerNorm(x' + Dropout(FFN(x')))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer for synthetic lethality prediction.
    Processes graph G=(V,E) via linearization σ: V→[n] with weighted support views.
    Final prediction: ŷ_ij = σ(MLP([h_i; h_j; f_combined(i,j)]))
    """

    def __init__(self,
                 node_feature_dim: int,
                 n_support_views: int,
                 feature_dim_per_view: int,
                 hidden_dim: int,
                 n_heads: int,
                 n_layers: int,
                 n_nodes: int,
                 dropout: float = 0.1,
                 undirected: bool = True):
        super().__init__()

        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_support_views = n_support_views
        self.feature_dim_per_view = feature_dim_per_view

        # Linear projection: ℝ^{d_ESM} → ℝ^{d_hidden}
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Positional encodings: PE ∈ ℝ^{n×d}, initialized ~ N(0, 0.02²)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_nodes, hidden_dim) * 0.02)

        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim=hidden_dim,
                             n_heads=n_heads,
                             n_support_views=n_support_views,
                             feature_dim_per_view=feature_dim_per_view,
                             dropout=dropout,
                             undirected=undirected) for _ in range(n_layers)
        ])

        # Edge classifier: [h_i; h_j; f(i,j)] ∈ ℝ^{2d+D} → ℝ
        # MLP: (2d+D) → d → d/2 → 1 with ReLU activations
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + feature_dim_per_view, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, linearization: List[int],
                support_views: torch.Tensor, edge_map: Dict[Tuple[int, int],
                                                            int],
                query_edges: torch.Tensor) -> torch.Tensor:
        """
        Process a single graph linearization and predict edge scores.
        
        Args:
            node_features: ESM embeddings for all nodes (n_nodes, node_feature_dim)
            linearization: Order of nodes in this linearization [node_idx_1, node_idx_2, ...]
            support_views: All edge features (n_edges, n_support_views, support_view_dim)
            edge_map: Maps (node_i, node_j) pairs to indices in support_views
            query_edges: Edges to predict scores for (2, n_query_edges)
            
        Returns:
            Predicted scores for query edges (n_query_edges,)
        """
        device = node_features.device
        seq_len = len(linearization)

        # Input validation
        assert seq_len <= self.n_nodes, f"Linearization length {seq_len} exceeds n_nodes {self.n_nodes}"
        assert node_features.shape[
            0] >= self.n_nodes, f"Not enough node features: {node_features.shape[0]} < {self.n_nodes}"
        assert all(
            0 <= idx < self.n_nodes
            for idx in linearization), "Invalid node indices in linearization"

        # Ensure linearization is on the same device as node_features
        if isinstance(linearization, (list, np.ndarray)):
            linearization = torch.tensor(linearization,
                                         device=node_features.device,
                                         dtype=torch.long)
        else:
            linearization = linearization.to(node_features.device)

        # Get node features in linearization order
        ordered_features = node_features[
            linearization]  # (seq_len, node_feature_dim)

        # Initial embeddings: h⁽⁰⁾ = W_proj·x + PE_σ
        h = self.node_proj(ordered_features)  # (seq_len, hidden_dim)
        h = h + self.pos_encoding[0, :seq_len, :]  # Add positional signal
        h = self.dropout(h)

        # Prepare edge information for this specific linearization
        seq_edge_indices, seq_support_views = self._prepare_sequence_edges(
            linearization, edge_map, support_views)

        # Stack L transformer layers: h⁽ˡ⁺¹⁾ = TransformerLayer(h⁽ˡ⁾)
        h = h.unsqueeze(0)  # Add batch dimension: (1, seq_len, hidden_dim)
        for layer in self.transformer_layers:
            h = layer(h, seq_support_views, seq_edge_indices)
        h = h.squeeze(0)  # Final representations h⁽ᴸ⁾ ∈ ℝ^{n×d}

        # Map representations back to original node indices
        node_representations = torch.zeros(self.n_nodes,
                                           self.hidden_dim,
                                           device=device)
        for seq_pos, node_idx in enumerate(linearization):
            node_representations[node_idx] = h[seq_pos]

        # Predict scores for query edges (vectorized for efficiency)
        assert query_edges.shape[
            0] == 2, f"Query edges should have shape (2, n_edges), got {query_edges.shape}"
        n_query_edges = query_edges.shape[1]

        # Validate all edges at once
        assert torch.all(query_edges >= 0) and torch.all(query_edges < self.n_nodes), \
            f"Invalid edge indices for n_nodes={self.n_nodes}"

        # Get all node representations at once
        u_indices = query_edges[0]
        v_indices = query_edges[1]
        u_reprs = node_representations[
            u_indices]  # (n_query_edges, hidden_dim)
        v_reprs = node_representations[
            v_indices]  # (n_query_edges, hidden_dim)

        # Prepare edge features
        edge_feats = torch.zeros(n_query_edges,
                                 self.feature_dim_per_view,
                                 device=device)

        # Get support view weights once
        alpha_weights = self.get_support_view_weights()

        # Process edges that exist in edge_map
        for i in range(n_query_edges):
            u, v = query_edges[0, i].item(), query_edges[1, i].item()
            edge_key = (min(u, v), max(u, v))
            if edge_key in edge_map:
                edge_idx = edge_map[edge_key]
                edge_feats[i] = (support_views[edge_idx] *
                                 alpha_weights.view(-1, 1)).sum(dim=0)

        # Edge prediction: logit(p_ij) = MLP([h_i⁽ᴸ⁾; h_j⁽ᴸ⁾; f(i,j)])
        combined = torch.cat([u_reprs, v_reprs, edge_feats],
                             dim=-1)  # ℝ^{E×(2d+D)}
        predictions = self.edge_predictor(combined).squeeze(-1)  # ℝ^E (logits)

        return predictions

    def _prepare_sequence_edges(
            self, linearization: List[int], edge_map: Dict[Tuple[int, int],
                                                           int],
            support_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare edge indices and support views for the current linearization.
        Maps global edge information to sequence-specific positions.
        """
        edge_indices_list = []
        edge_features_list = []

        # Check all pairs of nodes in the linearization
        for i, node_i in enumerate(linearization):
            for j, node_j in enumerate(linearization):
                if i != j:  # Skip self-loops
                    # Use canonical edge representation (smaller index first)
                    edge_key = (min(node_i, node_j), max(node_i, node_j))
                    if edge_key in edge_map:
                        # Record edge positions in the sequence
                        edge_indices_list.append([i, j])
                        # Get corresponding support views
                        edge_idx = edge_map[edge_key]
                        edge_features_list.append(support_views[edge_idx])

        if edge_indices_list:
            # Convert to tensors
            seq_edge_indices = torch.tensor(edge_indices_list,
                                            device=support_views.device).t()
            seq_support_views = torch.stack(edge_features_list)
        else:
            # No edges in this linearization (unlikely but handle gracefully)
            seq_edge_indices = None
            seq_support_views = None

        return seq_edge_indices, seq_support_views

    def get_support_view_weights(self) -> torch.Tensor:
        """Get non-negative support view weights: ᾱ_m = (1/L)Σ_l α_m⁽ˡ⁾ where α_m⁽ˡ⁾ = (w_m⁽ˡ⁾)²"""
        weights = []
        for layer in self.transformer_layers:
            weights.append(layer.attention.support_view_weights**2)
        return torch.stack(weights).mean(dim=0)  # Average α across L layers

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get all learnable support view weights for analysis."""
        weights = {}
        for i, layer in enumerate(self.transformer_layers):
            raw_weights = layer.attention.support_view_weights
            weights[f'layer_{i}_raw'] = raw_weights
            weights[f'layer_{i}_squared'] = raw_weights**2
        weights['average_squared'] = self.get_support_view_weights()
        return weights
