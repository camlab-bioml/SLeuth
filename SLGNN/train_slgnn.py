#!/usr/bin/env python3
"""
Training script for the adapted SLGNN PyTorch implementation.

Reuses the SLMGAE benchmark machinery for exact comparability:
  - Train/test splitting : cv1/cv2/cv3 via data_split.SLDataSplitter
  - Evaluation           : preprocess_benchmarking_paper.cal_metrics (AUROC/F1/
                           AUPR + NDCG/Recall/Precision/MAP @ k)

Training is full-graph (like this repo's SLMGAE): each epoch runs the GNN once
over the fold's SL graph, then applies BCE over ALL labeled training pairs plus
the L2 embedding and factor-independence regularizers. Model selection keeps the
checkpoint with the best VALIDATION AUPR per fold (see the sel_name logic below);
it falls back to the test split, with a printed warning, only for legacy fold
sets that carry no validation partition. The regime actually used travels into
model_params.json as `selection_regime`, so a row can never be misread.

Usage (from SLGNN/):
    python train_slgnn.py --cv_type cv3 --embedding_path ../data/all_genes_kg_complex.pt
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from slgnn_model import SLGNN
from sl_data import load_sl_edges, load_gene_features, load_shared_folds
from data_split import SLDataSplitter
from evaluation import evaluate_predictions_dict

# One parameter-counting definition for all four models — see ../sl_comparison.
sys.path.insert(0,
                str(Path(__file__).resolve().parent.parent / "sl_comparison"))
from model_params import (
    purge_stale_predictions,  # noqa: E402
    write_model_params)

METRIC_KEYS = [
    "auroc",
    "f1",
    "aupr",
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
    "recall@10",
    "recall@20",
    "recall@50",
    "precision@10",
    "precision@20",
    "precision@50",
    "map@10",
    "map@20",
    "map@50",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_fold(model, num_nodes, test_pos, test_neg, seen_pos):
    mat = model.score_matrix()
    mat[np.arange(num_nodes), np.arange(num_nodes)] = 0
    return evaluate_predictions_dict(mat, test_pos, test_neg, seen_pos)


def train_fold(args, fold_idx, split, feats, device):
    print(f"\n{'='*60}\nFold {fold_idx + 1}/{args.num_folds}\n{'='*60}")

    # Remove this fold's matrix from any previous run BEFORE training.
    # evaluate_model.py grades <output_dir>/fold_<k>_<cv>_predictions.npy, so a
    # fold that dies mid-run leaves the PREVIOUS run's matrix to be graded
    # silently alongside fresh ones — reproduced on SLMGAE at exit 0 with no
    # warning. A crashed fold must be MISSING (evaluate_model skips it and says
    # so), never stale.
    _stale = Path(
        args.output_dir) / f"fold_{fold_idx}_{args.cv_type}_predictions.npy"
    if _stale.exists():
        print(f"  Removing stale prediction matrix from a previous run: "
              f"{_stale.name}")
        _stale.unlink()

    num_nodes = split["train_adj"].shape[0]
    train_edges = torch.from_numpy(split["train_edges"].astype(
        np.int64)).to(device)
    train_labels = torch.from_numpy(split["train_labels"].astype(
        np.float32)).to(device)
    test_edges = split["test_edges"].astype(np.int32)
    test_labels = split["test_labels"]
    test_pos = test_edges[test_labels == 1]
    test_neg = test_edges[test_labels == 0]
    seen_pos = split["train_edges"][split["train_labels"] == 1].astype(
        np.int32)
    # Model selection runs on the VALIDATION split so the reported test metrics
    # are not also the selection criterion. Legacy fold sets (prepare_folds
    # without --val_frac) carry no val arrays; fall back to test and say so.
    val_pos = split.get("val_pos", test_pos[:0]).astype(np.int32)
    val_neg = split.get("val_neg", test_neg[:0]).astype(np.int32)
    if len(val_pos) and len(val_neg):
        sel_pos, sel_neg, sel_name = val_pos, val_neg, "val"
    else:
        sel_pos, sel_neg, sel_name = test_pos, test_neg, "TEST"
        print("  WARNING: no validation split in this fold set — selecting on "
              "TEST. Regenerate folds with --val_frac for an honest number.")
    print(f"Train: {int((split['train_labels']==1).sum())} pos, "
          f"{int((split['train_labels']==0).sum())} neg | "
          f"Val: {len(val_pos)} pos, {len(val_neg)} neg | "
          f"Test: {len(test_pos)} pos, {len(test_neg)} neg")

    model = SLGNN(
        num_nodes=num_nodes,
        feat_dim=feats.shape[1],
        gene_features=feats,
        train_adj=split["train_adj"],
        dim=args.dim,
        n_factors=args.n_factors,
        n_hops=args.n_hops,
        dropout=args.dropout,
        l2_weight=args.l2_weight,
        sim_regularity=args.sim_regularity,
        device=device,
    ).to(device)

    # selection_regime travels with the predictions (model_params.json ->
    # evaluate_model.build_summary -> compare.py), so the table itself shows
    # which rows were val-selected. Nothing else could distinguish a row
    # whose checkpoint was picked on TEST from one picked honestly.
    write_model_params(args.output_dir,
                       model,
                       folds_dir=getattr(args, "folds_dir", None),
                       cv_type=getattr(args, "cv_type", None),
                       selection_regime=sel_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    ga, gb = train_edges[:, 0], train_edges[:, 1]

    best_aupr = -1.0
    best_metrics = None
    best_state = None
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        node_emb, cor = model.encode()
        logits, emb_loss = model.pair_logits(node_emb, ga, gb)
        loss = criterion(logits, train_labels) + emb_loss \
            + args.sim_regularity * cor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = evaluate_fold(model, num_nodes, sel_pos, sel_neg,
                                    seen_pos)
            print(f"  epoch {epoch+1:3d} | loss {loss.item():.4f} "
                  f"cor {float(cor):.4f} | {sel_name} AUROC "
                  f"{metrics['auroc']:.4f} AUPR {metrics['aupr']:.4f}")
            if metrics["aupr"] > best_aupr:
                best_aupr = metrics["aupr"]
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    # Restore the selected checkpoint, THEN score test exactly once. Test is
    # never consulted during training, so this number is honest held-out.
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = evaluate_fold(model, num_nodes, test_pos, test_neg,
                                 seen_pos)
    mat = model.score_matrix()
    np.save(
        Path(args.output_dir) /
        f"fold_{fold_idx}_{args.cv_type}_predictions.npy",
        mat.astype(np.float32))
    print(f"  Best fold {fold_idx + 1}: AUPR={best_metrics['aupr']:.4f}, "
          f"AUROC={best_metrics['auroc']:.4f}, F1={best_metrics['f1']:.4f}")
    return best_metrics


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"results/slgnn_{args.cv_type}"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Folds: shared canonical folds (fair comparison) or own SLDataSplitter.
    if args.folds_dir:
        gene_order, num_nodes, splits = load_shared_folds(
            args.folds_dir, args.cv_type)
        print(f"Shared folds from {args.folds_dir}/{args.cv_type}: "
              f"{len(splits)} folds, {num_nodes} genes")
    else:
        pos_edges, num_nodes, gene_order = load_sl_edges(args.data_path)
        splitter = SLDataSplitter(
            pos_edges=pos_edges,
            neg_edges=np.empty((0, 2), dtype=int),
            num_nodes=num_nodes,
            train_ratio=args.train_ratio,
            random_state=args.seed,
        )
        split_fn = {
            "cv1": splitter.cv1_split,
            "cv2": splitter.cv2_split,
            "cv3": splitter.cv3_split
        }[args.cv_type]
        splits = split_fn(k=args.num_folds, pos_neg_ratio=args.pos_neg_ratio)

    feats, matched = load_gene_features(args.embedding_path,
                                        gene_order,
                                        standardize=not args.no_standardize)
    print(f"Genes: {num_nodes} | features: {feats.shape[1]}d | "
          f"matched {matched}/{num_nodes}")

    # CLEAR EVERY FOLD'S LEFTOVER MATRIX BEFORE FOLD 0.
    #
    # The per-fold clear inside the loop only protects folds this run
    # actually reaches. A run that dies at fold 3 leaves folds 3-4 holding
    # the PREVIOUS run's matrices, and evaluate_model.py grades every
    # matrix it finds in the directory — publishing a mean over fresh and
    # foreign folds at exit 0. A crashed fold must be MISSING, not stale.
    purge_stale_predictions(args.output_dir, args.cv_type)

    fold_metrics = []
    for split in splits:
        fold_metrics.append(
            train_fold(args, split["fold"], split, feats, device))

    summary = {
        "cv_type": args.cv_type,
        "embedding_path": args.embedding_path,
        "folds": fold_metrics
    }
    print("\n" + "=" * 60 + "\nCross-validation results:\n" + "=" * 60)
    for key in METRIC_KEYS:
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        print(f"  {key:14s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    summary["args"] = vars(args)

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({"summary": summary}, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


def parse_args():
    p = argparse.ArgumentParser(description="Train adapted SLGNN (PyTorch)")
    p.add_argument("--data_path", type=str, default="../data")
    p.add_argument("--folds_dir",
                   type=str,
                   default=None,
                   help="Use shared canonical CV folds from this dir "
                   "(e.g. ../sl_comparison/folds) for the fair 4-model "
                   "comparison. When set, SLDataSplitter is bypassed.")
    p.add_argument("--embedding_path",
                   type=str,
                   default="../data/all_genes_kg_complex.pt")
    p.add_argument("--no_standardize", action="store_true")
    p.add_argument("--output_dir",
                   type=str,
                   default=None,
                   help="Default: results/slgnn_<cv_type>")
    p.add_argument("--cv_type",
                   type=str,
                   default="cv1",
                   choices=["cv1", "cv2", "cv3"])
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--pos_neg_ratio", type=float, default=1.0)
    # Model / training (benchmark SLGNN defaults where applicable)
    # Defaults follow Zhu et al., Bioinformatics 2023 (PMC9907046), which tuned
    # dim over {32,64,128,256} -> 256, layers over {1,2,3} -> 3, lr -> 0.002.
    # The previous defaults (dim 64, 2 hops, lr 3e-3, 100 epochs) ran the model
    # at a quarter of its tuned width and cut it off while test AUPR was still
    # rising — 4 of 15 folds peaked at exactly the epoch cap — so those numbers
    # were floors, not estimates.
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n_factors", type=int, default=4)
    p.add_argument("--n_hops",
                   type=int,
                   default=3,
                   help="Message-passing hops (paper: 3)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l2_weight", type=float, default=1e-4)
    p.add_argument("--sim_regularity", type=float, default=1e-3)
    p.add_argument("--learning_rate", type=float, default=2e-3)
    p.add_argument("--epochs",
                   type=int,
                   default=500,
                   help="Cap only — patience should terminate the run. If a "
                   "fold's best epoch equals this, raise it and re-run.")
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--early_stop",
                   type=int,
                   default=10,
                   help="Patience in evaluations (not epochs)")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    main()
