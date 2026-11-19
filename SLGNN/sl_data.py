#!/usr/bin/env python3
"""
SL data loading for the SLGNN PyTorch directory.

The gene index and the SL labels are the shared folds' (load_shared_folds, a
byte-identical copy of sl_comparison/shared_folds.py): genes.txt is the gene
universe (Entrez ID per line, index == gene index) and the fold files hold the
SLDB 3.0 positives, the screened negatives and the validation split, so
results are directly comparable across the four models. It is the only split
source train_slgnn.py trains on (--folds_dir is required).

--arch original learns its gene embeddings over SynLethKG
(build_synlethkg.load_kg_for_genes). --arch port instead reads per-gene
features from any of the repo's ``all_genes_*.pt`` embedding files
(load_gene_features). These store a dict with ``embeddings`` (n_genes, dim)
and ``gene_order`` (list of Entrez ID strings); we realign them to the gene
index by Entrez ID.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


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
    if not fold_dirs:
        # An empty CV directory must stop the run: every per-fold loop would
        # otherwise be skipped and the trainer would write an all-NaN summary.
        raise SystemExit(f"no fold_* directories under {root / cv_type}")
    splits = []
    for k, d in enumerate(fold_dirs):
        tp = np.load(d / "train_pos.npy")
        tn = np.load(d / "train_neg.npy")
        ep = np.load(d / "test_pos.npy")
        en = np.load(d / "test_neg.npy")
        # Validation (prepare_folds --val_frac). Absent in legacy fold sets,
        # in which case val_* come back empty and every trainer REFUSES the
        # fold set (selection is val-only; there is no test fallback).
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
