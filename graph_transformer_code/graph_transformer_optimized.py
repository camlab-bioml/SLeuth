#!/usr/bin/env python3
"""
Optimized Graph Transformer with memory-efficient operations.
Key optimizations:
1. Pre-allocate tensors instead of growing lists
2. Use in-place operations where possible
3. Avoid unnecessary intermediate tensors
4. Use sparse operations for edge biases
5. Batch operations to reduce loop overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class WeightedSupportViewAttention(nn.Module):
    """
    Memory-optimized multi-head attention with weighted support views.
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

        # Standard attention projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Learnable weights for support views
        # Note: Parameter will be moved to correct device when model.to(device) is called
        self.support_view_weights = nn.Parameter(
            torch.ones(n_support_views) + torch.randn(n_support_views) * 0.1)
        self.weight_clip_value = 5.0

        # Project support views to biases
        self.support_view_proj = nn.Linear(feature_dim_per_view, n_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, support_views: Optional[torch.Tensor],
                edge_indices: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V with single reshape operation
        qkv = x @ torch.cat(
            [self.W_q.weight.t(),
             self.W_k.weight.t(),
             self.W_v.weight.t()],
            dim=1)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.unbind(
            dim=2)  # More efficient than multiple view operations
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add support view biases efficiently
        if support_views is not None and edge_indices is not None:
            # Compute weights once
            clamped_weights = self.support_view_weights.clamp(
                -self.weight_clip_value, self.weight_clip_value)
            alpha_weights = clamped_weights.square()

            # Compute edge biases without creating intermediate tensors
            # Use einsum for efficient tensor operations
            edge_biases = torch.einsum('esf,s->ef', support_views,
                                       alpha_weights)
            edge_biases = self.support_view_proj(edge_biases)

            # Use sparse operations for bias matrix (more memory efficient)
            edge_i, edge_j = edge_indices
            n_edges = edge_i.size(0)

            # Add biases directly to scores using advanced indexing
            # This avoids creating the full bias matrix
            scores[:, :, edge_i, edge_j] += edge_biases.t().unsqueeze(0)

            if self.undirected:
                scores[:, :, edge_j, edge_i] += edge_biases.t().unsqueeze(0)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Efficient reshape without contiguous()
        context = context.transpose(1, 2).reshape(batch_size, seq_len,
                                                  self.hidden_dim)
        output = self.W_o(context)

        return output


class TransformerLayer(nn.Module):
    """Memory-optimized transformer layer."""

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 n_support_views: int,
                 feature_dim_per_view: int,
                 dropout: float = 0.1,
                 undirected: bool = True):
        super().__init__()

        self.attention = WeightedSupportViewAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_support_views=n_support_views,
            feature_dim_per_view=feature_dim_per_view,
            dropout=dropout,
            undirected=undirected)

        # More efficient feed-forward with fused operations
        self.ff_linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ff_linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, support_views: Optional[torch.Tensor],
                edge_indices: Optional[torch.Tensor]) -> torch.Tensor:
        # Use in-place operations for residual connections
        residual = x
        x = self.attention(x, support_views, edge_indices)
        x = self.dropout(x)
        x.add_(residual)  # In-place addition
        x = self.norm1(x)

        # Feed-forward with in-place operations
        residual = x
        x = self.ff_linear1(x)
        x = F.gelu(x, approximate='tanh')  # Faster GELU approximation
        x = self.dropout(x)
        x = self.ff_linear2(x)
        x = self.dropout(x)
        x.add_(residual)  # In-place addition
        x = self.norm2(x)

        return x


class GraphTransformer(nn.Module):
    """
    Memory-optimized Graph Transformer.
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

        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        # Note: Parameter will be moved to correct device when model.to(device) is called
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_nodes, hidden_dim) * 0.02)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim=hidden_dim,
                             n_heads=n_heads,
                             n_support_views=n_support_views,
                             feature_dim_per_view=feature_dim_per_view,
                             dropout=dropout,
                             undirected=undirected) for _ in range(n_layers)
        ])

        # Edge prediction MLP
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + feature_dim_per_view, hidden_dim),
            nn.ReLU(inplace=True),  # In-place ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),  # In-place ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, linearization: List[int],
                support_views: torch.Tensor, edge_map: Dict[Tuple[int, int],
                                                            int],
                query_edges: torch.Tensor) -> torch.Tensor:
        device = node_features.device
        seq_len = len(linearization)

        # Validation
        assert seq_len <= self.n_nodes
        assert node_features.shape[0] >= self.n_nodes

        # Get ordered features efficiently
        linearization_tensor = torch.tensor(linearization,
                                            device=device,
                                            dtype=torch.long)
        ordered_features = node_features.index_select(0, linearization_tensor)

        # Project and add positional encoding
        h = self.node_proj(ordered_features)
        h.add_(self.pos_encoding[0, :seq_len, :])  # In-place addition
        h = self.dropout(h)

        # Prepare edges efficiently
        seq_edge_indices, seq_support_views = self._prepare_sequence_edges_optimized(
            linearization, edge_map, support_views, device)

        # Apply transformer layers
        h = h.unsqueeze(0)
        for layer in self.transformer_layers:
            h = layer(h, seq_support_views, seq_edge_indices)
        h = h.squeeze(0)

        # Efficient node representation mapping
        # Only allocate for nodes we actually use
        node_representations = torch.zeros(self.n_nodes,
                                           self.hidden_dim,
                                           device=device)
        node_representations.index_copy_(0, linearization_tensor, h)

        # Batch process query edges for efficiency
        n_queries = query_edges.shape[1]
        predictions = torch.zeros(n_queries, device=device)

        # Get all edge representations at once
        u_indices = query_edges[0]
        v_indices = query_edges[1]
        u_reprs = node_representations[u_indices]
        v_reprs = node_representations[v_indices]

        # Get support view weights once
        alpha_weights = self.get_support_view_weights()

        # Process edges in batches
        batch_size = 1000  # Process 1000 edges at a time
        for i in range(0, n_queries, batch_size):
            end_idx = min(i + batch_size, n_queries)
            batch_u = u_reprs[i:end_idx]
            batch_v = v_reprs[i:end_idx]

            # Get edge features for batch
            batch_edge_feats = []
            for j in range(i, end_idx):
                u, v = query_edges[0, j].item(), query_edges[1, j].item()
                edge_key = (min(u, v), max(u, v))
                if edge_key in edge_map:
                    edge_idx = edge_map[edge_key]
                    edge_feat = torch.einsum('sf,s->f',
                                             support_views[edge_idx],
                                             alpha_weights)
                else:
                    edge_feat = torch.zeros(self.feature_dim_per_view,
                                            device=device)
                batch_edge_feats.append(edge_feat)

            batch_edge_feats = torch.stack(batch_edge_feats)

            # Concatenate and predict for batch
            combined = torch.cat([batch_u, batch_v, batch_edge_feats], dim=-1)
            batch_scores = self.edge_predictor(combined).squeeze(-1)
            predictions[i:end_idx] = batch_scores

        return predictions

    def _prepare_sequence_edges_optimized(
            self, linearization: List[int],
            edge_map: Dict[Tuple[int, int], int], support_views: torch.Tensor,
            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized edge preparation with pre-allocation and vectorized operations.
        """
        seq_len = len(linearization)

        # Pre-compute which edges exist to allocate exact size
        edge_count = 0
        edge_exists = {}
        for i, node_i in enumerate(linearization):
            for j, node_j in enumerate(linearization):
                if i != j:
                    edge_key = (min(node_i, node_j), max(node_i, node_j))
                    if edge_key in edge_map:
                        edge_exists[(i, j)] = edge_map[edge_key]
                        edge_count += 1

        if edge_count == 0:
            return None, None

        # Pre-allocate tensors with exact size
        seq_edge_indices = torch.zeros((2, edge_count),
                                       dtype=torch.long,
                                       device=device)
        seq_support_views = torch.zeros((edge_count, *support_views.shape[1:]),
                                        dtype=support_views.dtype,
                                        device=device)

        # Fill pre-allocated tensors
        idx = 0
        for (i, j), edge_idx in edge_exists.items():
            seq_edge_indices[0, idx] = i
            seq_edge_indices[1, idx] = j
            seq_support_views[idx] = support_views[edge_idx]
            idx += 1

        return seq_edge_indices, seq_support_views

    def get_support_view_weights(self) -> torch.Tensor:
        """Optimized weight computation using vectorized operations."""
        # Stack all weights at once
        all_weights = torch.stack([
            layer.attention.support_view_weights.clamp(
                -layer.attention.weight_clip_value,
                layer.attention.weight_clip_value)
            for layer in self.transformer_layers
        ])
        # Square and average in one operation
        return all_weights.square().mean(dim=0)

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get all learnable support view weights for analysis."""
        weights = {}
        for i, layer in enumerate(self.transformer_layers):
            raw_weights = layer.attention.support_view_weights
            weights[f'layer_{i}_raw'] = raw_weights
            weights[f'layer_{i}_squared'] = raw_weights.square()
        weights['average_squared'] = self.get_support_view_weights()
        return weights
