#!/usr/bin/env python3
"""
Adapted PyTorch implementation of SLGNN (factor-aware GNN for synthetic
lethality; Zhu et al., Bioinformatics 2023).

The original SLGNN is a *disentangled, factor-aware knowledge-graph* GNN built
on SynLethKG (h/r/t triples) using DGL's GATConv + torch_scatter. This repo has
neither the raw KG nor those dependencies, so this is a faithful **adaptation**
that keeps SLGNN's distinctive ideas while running on the plain PyTorch stack:

  * per-gene input features (any all_genes_*.pt; default kg_complex) projected
    to the model dimension as the initial node embedding;
  * ``n_factors`` DISENTANGLED latent factors, each with its own GAT over the
    SL graph, gated per-gene by a softmax factor-attention (mirrors the
    reference ``score = softmax(gene · latent^T)`` mixing);
  * a factor-independence regularizer (distance correlation between factor
    prototypes) — SLGNN's ``_cul_cor`` term;
  * an inner-product decoder on the final gene embeddings, trained with BCE on
    labeled pairs (the reference's classification objective).

What is dropped vs. the paper: the external knowledge graph and its relational
entity aggregation (no KG available here). Attention is computed on the sparse
SL edge list (no DGL / torch_scatter needed).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_graph_tensors(train_adj, device):
    """From a scipy SL adjacency (train positives) build the tensors the GNN needs.

    Returns:
        edge_index: LongTensor (2, E) directed edges (both directions +
            self-loops); message passes tail -> head (dst -> src).
        norm_adj: sparse FloatTensor, symmetric-normalized A+I for base
            propagation (D^-1/2 (A+I) D^-1/2).
    """
    adj = sp.coo_matrix(train_adj)
    adj = (adj + adj.T).tocoo()
    adj.sum_duplicates()  # merge repeats before binarizing
    adj.data[:] = 1.0

    n = adj.shape[0]
    # Symmetric-normalized adjacency with self-loops (for base propagation).
    # Self-loops guarantee degree >= 1, so d^-1/2 is always finite.
    a_i = adj + sp.eye(n)
    deg = np.asarray(a_i.sum(1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    norm = (D @ a_i @ D).tocoo()
    idx = torch.from_numpy(np.vstack((norm.row, norm.col)).astype(np.int64))
    val = torch.from_numpy(norm.data.astype(np.float32))
    norm_adj = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to(device)

    # Edge list for attention (undirected edges + self-loops).
    self_loops = np.arange(n)
    src = np.concatenate([adj.row, self_loops])
    dst = np.concatenate([adj.col, self_loops])
    edge_index = torch.from_numpy(np.vstack(
        (src, dst)).astype(np.int64)).to(device)
    return edge_index, norm_adj


class GATLayer(nn.Module):
    """Single-head GAT on a sparse edge list (no DGL / torch_scatter).

    Uses the factorized additive attention of Velickovic et al. (2018) with a
    numerically-stable exp (clamp instead of per-node max-subtraction) so the
    segment-softmax needs only ``index_add_``.
    """

    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.dropout = dropout
        self.alpha = alpha
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)

    def forward(self, h, edge_index, num_nodes):
        src, dst = edge_index[0], edge_index[1]  # message dst -> src
        Wh = self.W(h)  # (N, out)
        e = (Wh[src] * self.a_src).sum(-1) + (Wh[dst] * self.a_dst).sum(-1)
        e = F.leaky_relu(e, self.alpha).clamp(-10.0, 10.0).exp()  # (E,)
        denom = torch.zeros(num_nodes, device=h.device).index_add_(0, src, e)
        att = e / (denom[src] + 1e-16)
        att = F.dropout(att, self.dropout, self.training)
        out = torch.zeros(num_nodes, Wh.size(1), device=h.device)
        out.index_add_(0, src, att.unsqueeze(-1) * Wh[dst])
        return out


def distance_correlation(x, y):
    """Distance correlation between two 1-D tensors (SLGNN's independence term)."""
    n = x.shape[0]
    zeros = torch.zeros(n, n, device=x.device)
    zero = torch.zeros(1, device=x.device)
    x, y = x.unsqueeze(-1), y.unsqueeze(-1)
    a_ = torch.matmul(x, x.t()) * 2
    b_ = torch.matmul(y, y.t()) * 2
    xs, ys = x**2, y**2
    a = torch.sqrt(torch.max(xs - a_ + xs.t(), zeros) + 1e-8)
    b = torch.sqrt(torch.max(ys - b_ + ys.t(), zeros) + 1e-8)
    A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
    dcov_ab = torch.sqrt(torch.max((A * B).sum() / n**2, zero) + 1e-8)
    dcov_aa = torch.sqrt(torch.max((A * A).sum() / n**2, zero) + 1e-8)
    dcov_bb = torch.sqrt(torch.max((B * B).sum() / n**2, zero) + 1e-8)
    return dcov_ab / torch.sqrt(dcov_aa * dcov_bb + 1e-8)


class SLGNN(nn.Module):
    """Disentangled factor-aware GNN with an inner-product SL decoder."""

    def __init__(self,
                 num_nodes,
                 feat_dim,
                 gene_features,
                 train_adj,
                 dim=64,
                 n_factors=4,
                 n_hops=2,
                 dropout=0.1,
                 l2_weight=1e-4,
                 sim_regularity=1e-3,
                 device=torch.device("cpu")):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_factors = n_factors
        self.n_hops = n_hops
        self.dim = dim
        self.dropout = dropout
        self.l2_weight = l2_weight
        self.sim_regularity = sim_regularity
        self.device = device

        # Fixed input features -> initial node embedding via a learned projection.
        self.register_buffer("gene_features", gene_features.to(device))
        self.feat_proj = nn.Linear(feat_dim, dim)

        # Disentangled latent factor prototypes.
        self.latent_emb = nn.Parameter(torch.empty(n_factors, dim))
        nn.init.xavier_uniform_(self.latent_emb)

        # One GAT per (hop, factor).
        self.gats = nn.ModuleList([
            nn.ModuleList(
                [GATLayer(dim, dim, dropout) for _ in range(n_factors)])
            for _ in range(n_hops)
        ])
        self.mess_dropout = nn.Dropout(dropout)

        edge_index, norm_adj = build_graph_tensors(train_adj, device)
        self.edge_index = edge_index
        self.norm_adj = norm_adj

    def encode(self):
        """Run the GNN over the full SL graph -> (node_emb (N,dim), cor)."""
        h = self.feat_proj(self.gene_features)  # (N, dim)
        node_res = h
        for hop in range(self.n_hops):
            # Per-gene factor attention: softmax(node · latent^T).
            score = F.softmax(h @ self.latent_emb.t(), dim=1)  # (N, n_factors)
            base = torch.sparse.mm(self.norm_adj, h)  # (N, dim)
            factor_outs = []
            for f in range(self.n_factors):
                out_f = self.gats[hop][f](h, self.edge_index, self.num_nodes)
                factor_outs.append(out_f * self.latent_emb[f])
            stacked = torch.stack(factor_outs, dim=1)  # (N, F, dim)
            mixed = (stacked * score.unsqueeze(-1)).sum(dim=1)  # (N, dim)
            h = base + mixed
            h = self.mess_dropout(h)
            h = F.normalize(h)
            node_res = node_res + h

        cor = self._independence()
        return node_res, cor

    def _independence(self):
        """Sum of pairwise distance correlations between factor prototypes."""
        cor = torch.zeros(1, device=self.device)
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                cor = cor + distance_correlation(self.latent_emb[i],
                                                 self.latent_emb[j])
        return cor.squeeze()

    def pair_logits(self, node_emb, gene_a, gene_b):
        """Inner-product decoder + L2 embedding regularizer for a pair batch."""
        e_a = node_emb[gene_a]
        e_b = node_emb[gene_b]
        logits = (e_a * e_b).sum(dim=1)
        reg = (e_a.norm(dim=1).pow(2).sum() + e_b.norm(dim=1).pow(2).sum()) / 2
        emb_loss = self.l2_weight * reg / max(len(gene_a), 1)
        return logits, emb_loss

    @torch.no_grad()
    def score_matrix(self):
        """Full gene x gene SL score matrix (inner product), diagonal zeroed."""
        self.eval()
        node_emb, _ = self.encode()
        mat = (node_emb @ node_emb.t()).cpu().numpy()
        n = mat.shape[0]
        mat[np.arange(n), np.arange(n)] = 0
        return mat
