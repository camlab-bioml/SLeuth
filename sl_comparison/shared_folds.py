#!/usr/bin/env python3
"""
Canonical loader for the shared CV folds written by prepare_folds.py.

Model-agnostic: returns the gene universe and a list of per-fold split dicts
(train_adj built from positives only; negatives are label-0 supervision). Every
model that imports this reads the IDENTICAL division, which is what makes the
four-model comparison fair. (NSF4SL/ and SLGNN/ carry a byte-identical copy of
this function in their own sl_data.py to stay self-contained.)
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


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


def read_gene_order(folds_dir):
    """Just the shared gene universe (list of Entrez ID strings)."""
    with open(Path(folds_dir) / "genes.txt") as f:
        return [line.strip() for line in f if line.strip()]
