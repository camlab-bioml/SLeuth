#!/usr/bin/env python3
"""
Shared SL data loading for the SLGNN / NSF4SL PyTorch directories.

Uses the SAME gene index space and SL edge source as SLMGAE-in-pytorch so
results are directly comparable across models:
  - Gene list      : ../data/List_Proteins_in_SL.txt  (Entrez Gene ID per line)
  - SL positives   : ../data/SL_Human_Approved.txt    (Entrez pairs)

Per-gene features come from any of the repo's ``all_genes_*.pt`` embedding
files. These store a dict with ``embeddings`` (n_genes, dim) and ``gene_order``
(list of Entrez ID strings); we realign them to the SL gene index by Entrez ID.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def load_sl_edges(data_path):
    """Load SL positives + gene list, matching SLMGAE's DataLoader.load_sl_matrix.

    Returns:
        pos_edges: (P, 2) int array of upper-triangular positive gene-index pairs
        num_nodes: number of genes (rows in List_Proteins_in_SL.txt)
        gene_order: list of Entrez ID strings, index == gene index
    """
    gene_order = []
    sl_mapping = {}
    with open(f"{data_path}/List_Proteins_in_SL.txt", "r") as f:
        for idx, line in enumerate(f):
            gid = line.strip()
            gene_order.append(gid)
            sl_mapping[gid] = idx
    num_nodes = len(gene_order)

    row, col = [], []
    with open(f"{data_path}/SL_Human_Approved.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id1, id2 = parts[0], parts[1]
                if id1 in sl_mapping and id2 in sl_mapping:
                    row.append(sl_mapping[id1])
                    col.append(sl_mapping[id2])

    adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                        shape=(num_nodes, num_nodes))
    adj = (adj + adj.T).toarray()
    adj[adj != 0] = 1
    np.fill_diagonal(adj, 0)

    x, y = np.triu_indices(num_nodes, k=1)
    mask = adj[x, y] != 0
    pos_edges = np.stack([x[mask], y[mask]], axis=1).astype(np.int32)
    return pos_edges, num_nodes, gene_order


def load_gene_features(emb_path, gene_order, standardize=True):
    """Load an ``all_genes_*.pt`` embedding and align it to ``gene_order``.

    Genes absent from the embedding (or stored as NaN) are filled with the
    per-column mean over the available genes.

    Returns:
        feats: FloatTensor (num_nodes, dim)
        matched: number of genes matched to the embedding file
    """
    data = torch.load(emb_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        emb = (data["raw_embeddings"]
               if "raw_embeddings" in data else data["embeddings"])
        emb_genes = [str(g) for g in data["gene_order"]]
    else:
        raise ValueError(
            f"{emb_path} is a bare tensor without gene_order; expected a dict "
            f"with 'embeddings' and 'gene_order' (an all_genes_*.pt file).")

    emb = np.asarray(emb, dtype=np.float32)
    g2i = {g: i for i, g in enumerate(emb_genes)}
    dim = emb.shape[1]

    feats = np.full((len(gene_order), dim), np.nan, dtype=np.float32)
    matched = 0
    for idx, g in enumerate(gene_order):
        j = g2i.get(str(g))
        if j is not None:
            feats[idx] = emb[j]
            matched += 1

    # Fill missing genes / NaN cells with per-column mean over available rows.
    col_mean = np.nan_to_num(np.nanmean(feats, axis=0))
    nan_r, nan_c = np.where(np.isnan(feats))
    feats[nan_r, nan_c] = col_mean[nan_c]

    if standardize:
        mu = feats.mean(axis=0)
        sd = feats.std(axis=0)
        sd[sd == 0] = 1.0
        feats = (feats - mu) / sd

    return torch.from_numpy(feats.astype(np.float32)), matched


def load_shared_folds(folds_dir, cv_type):
    """Return (gene_order, num_nodes, splits) for one CV type."""
    root = Path(folds_dir)
    with open(root / "genes.txt") as f:
        gene_order = [line.strip() for line in f if line.strip()]
    num_nodes = len(gene_order)

    fold_dirs = sorted((root / cv_type).glob("fold_*"),
                       key=lambda d: int(d.name.split("_")[1]))
    splits = []
    for k, d in enumerate(fold_dirs):
        tp = np.load(d / "train_pos.npy")
        tn = np.load(d / "train_neg.npy")
        ep = np.load(d / "test_pos.npy")
        en = np.load(d / "test_neg.npy")
        # Validation (prepare_folds --val_frac). Absent in legacy fold sets,
        # in which case val_* come back empty and the caller must fall back to
        # test-based selection (and say so).
        vp = np.load(d / "val_pos.npy") if (d / "val_pos.npy").exists() \
            else ep[:0].copy()
        vn = np.load(d / "val_neg.npy") if (d / "val_neg.npy").exists() \
            else en[:0].copy()
        r = np.concatenate([tp[:, 0], tp[:, 1]])
        c = np.concatenate([tp[:, 1], tp[:, 0]])
        train_adj = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                                  shape=(num_nodes, num_nodes))
        train_adj.setdiag(0)
        train_adj.eliminate_zeros()
        splits.append({
            "fold":
            k,
            "train_adj":
            train_adj,
            "train_edges":
            np.vstack([tp, tn]).astype(np.int64),
            "train_labels":
            np.concatenate([np.ones(len(tp)),
                            np.zeros(len(tn))]),
            "test_edges":
            np.vstack([ep, en]).astype(np.int32),
            "test_labels":
            np.concatenate([np.ones(len(ep)),
                            np.zeros(len(en))]),
            "val_pos":
            vp.astype(np.int64),
            "val_neg":
            vn.astype(np.int64),
        })
    return gene_order, num_nodes, splits
