#!/usr/bin/env python3
"""
Training script for the NSF4SL PyTorch implementation.

Reuses the SLMGAE benchmark machinery for exact comparability:
  - Train/test splitting : cv1/cv2/cv3 via data_split.SLDataSplitter
  - Evaluation           : preprocess_benchmarking_paper.cal_metrics (the
                           official benchmark evaluator — AUROC/F1/AUPR +
                           NDCG/Recall/Precision/MAP @ k)

NSF4SL is negative-sample-free: each fold trains ONLY on the fold's positive SL
pairs (BYOL bootstrap between the two genes + feature-masked augmentations).
Fold negatives are used solely to score the test set. Model selection keeps the
checkpoint with the best VALIDATION score per fold, on --select_metric (the
authors' own criterion, NDCG@50 by default — not AUPR); it falls back to the
test split, with a printed warning, only for legacy fold sets that carry no
validation partition. The regime used travels into model_params.json as
`selection_regime`.

Usage (from NSF4SL/):
    python train_nsf4sl.py --cv_type cv3 --embedding_path ../data/all_genes_kg_complex.pt
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from nsf4sl_model import Net, score_matrix
from sl_data import load_sl_edges, load_gene_features, load_shared_folds
from data_split import SLDataSplitter
from evaluation import evaluate_predictions_dict

# One parameter-counting definition for all four models — see ../sl_comparison.
sys.path.insert(0,
                str(Path(__file__).resolve().parent.parent / "sl_comparison"))
from model_params import (
    purge_stale_predictions,  # noqa: E402
    write_model_params)

# Metric keys returned by cal_metrics (via evaluate_predictions_dict).
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


class PosPairDataset(Dataset):
    """Positive SL pairs with feature-masking augmentation (negative-free).

    Each item yields the two genes' features and augmented copies where a random
    ``aug_ratio`` fraction of feature dimensions is reset to the column mean.
    """

    def __init__(self, pos_pairs, feats_np, col_mean, aug_ratio, rng):
        self.pairs = np.asarray(pos_pairs, dtype=np.int64)
        self.feats = feats_np
        self.col_mean = col_mean
        self.aug_ratio = aug_ratio
        self.dim = feats_np.shape[1]
        self.rng = rng

    def __len__(self):
        return len(self.pairs)

    def _augment(self, vec):
        aug = vec.copy()
        k = int(self.dim * self.aug_ratio)
        if k > 0:
            mask_ids = self.rng.sample(range(self.dim), k)
            aug[mask_ids] = self.col_mean[mask_ids]
        return aug

    def __getitem__(self, index):
        a, b = int(self.pairs[index][0]), int(self.pairs[index][1])
        f1, f2 = self.feats[a], self.feats[b]
        return (
            torch.from_numpy(f1),
            torch.from_numpy(self._augment(f1)),
            torch.from_numpy(f2),
            torch.from_numpy(self._augment(f2)),
        )


def evaluate_fold(model, feats_device, num_nodes, test_pos, test_neg,
                  seen_pos):
    """Score matrix -> benchmark metrics on the fold's test edges."""
    mat = score_matrix(model, feats_device)
    mat[np.arange(num_nodes), np.arange(num_nodes)] = 0
    return evaluate_predictions_dict(mat, test_pos, test_neg, seen_pos)


def train_fold(args, fold_idx, split, feats_np, col_mean, device):
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

    # The vendored splitter emits float64 train_edges (int positives vstacked
    # with float64 random negatives), so cast to int — train_pos is later handed
    # to cal_metrics as seen_index, which fancy-indexes and requires int dtype.
    train_edges = split["train_edges"].astype(np.int64)
    train_labels = split["train_labels"]
    test_edges = split["test_edges"].astype(np.int32)
    test_labels = split["test_labels"]

    # Negative-sample-free: train on positives only.
    train_pos = train_edges[train_labels == 1]
    test_pos = test_edges[test_labels == 1]
    test_neg = test_edges[test_labels == 0]
    # Selection runs on VALIDATION, and on a RANKING metric — NSF4SL is a
    # negative-sample-free contrastive ranker and its authors select on
    # validation P@100, not on discrimination AUPR. Selecting on test AUPR (the
    # previous behaviour) stopped every cv1 fold at epoch 18 and restored the
    # epoch-3 checkpoint, while NDCG@10 was still rising steeply.
    val_pos = split.get("val_pos", test_pos[:0]).astype(np.int32)
    val_neg = split.get("val_neg", test_neg[:0]).astype(np.int32)
    if len(val_pos) and len(val_neg):
        sel_pos, sel_neg, sel_name = val_pos, val_neg, "val"
    else:
        sel_pos, sel_neg, sel_name = test_pos, test_neg, "TEST"
        print("  WARNING: no validation split in this fold set — selecting on "
              "TEST. Regenerate folds with --val_frac for an honest number.")
    print(f"Train positives: {len(train_pos)} | "
          f"Val: {len(val_pos)} pos, {len(val_neg)} neg | "
          f"Test: {len(test_pos)} pos, {len(test_neg)} neg | "
          f"select on {sel_name}:{args.select_metric}")

    rng = random.Random(args.seed + fold_idx)
    dataset = PosPairDataset(train_pos, feats_np, col_mean, args.aug_ratio,
                             rng)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=len(dataset) > args.batch_size)

    feats_device = torch.from_numpy(feats_np).float().to(device)

    model = Net(feats_np.shape[1], args.latent_size, args.momentum).to(device)
    # selection_regime travels with the predictions (model_params.json ->
    # evaluate_model.build_summary -> compare.py), so the table itself shows
    # which rows were val-selected. Nothing else could distinguish a row
    # whose checkpoint was picked on TEST from one picked honestly.
    write_model_params(args.output_dir,
                       model,
                       folds_dir=getattr(args, "folds_dir", None),
                       cv_type=getattr(args, "cv_type", None),
                       selection_regime=sel_name)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_aupr = -1.0
    best_metrics = None
    best_state = None
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for f1, f1a, f2, f2a in loader:
            f1, f1a = f1.float().to(device), f1a.float().to(device)
            f2, f2a = f2.float().to(device), f2a.float().to(device)
            out = model([f1, f2, f1a, f2a])
            loss = model.get_loss(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._update_target()
            losses.append(loss.item())
        scheduler.step()

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = evaluate_fold(model, feats_device,
                                    split["train_adj"].shape[0], sel_pos,
                                    sel_neg, train_pos)
            score = metrics[args.select_metric]
            print(f"  epoch {epoch+1:3d} | loss {np.mean(losses):.4f} | "
                  f"{sel_name} {args.select_metric} {score:.4f} "
                  f"AUPR {metrics['aupr']:.4f}")
            if score > best_aupr:
                best_aupr = score
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
    best_metrics = evaluate_fold(model, feats_device,
                                 split["train_adj"].shape[0], test_pos,
                                 test_neg, train_pos)
    mat = score_matrix(model, feats_device)
    _save_predictions(args, fold_idx, mat)
    print(f"  Best fold {fold_idx + 1}: AUPR={best_metrics['aupr']:.4f}, "
          f"AUROC={best_metrics['auroc']:.4f}, F1={best_metrics['f1']:.4f}")
    return best_metrics


def _save_predictions(args, fold_idx, mat):
    out = Path(args.output_dir) / \
        f"fold_{fold_idx}_{args.cv_type}_predictions.npy"
    np.save(out, mat.astype(np.float32))


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"results/nsf4sl_{args.cv_type}"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Folds: either the shared canonical folds (fair 4-model comparison) or the
    # model's own SLDataSplitter (random negatives, standalone use).
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

    # Per-gene features aligned to the (shared or native) gene index.
    feats, matched = load_gene_features(args.embedding_path,
                                        gene_order,
                                        standardize=not args.no_standardize)
    feats_np = feats.numpy()
    col_mean = feats_np.mean(axis=0)
    print(f"Genes: {num_nodes} | features: {feats_np.shape[1]}d | "
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
        m = train_fold(args, split["fold"], split, feats_np, col_mean, device)
        fold_metrics.append(m)

    # Aggregate and save summary.
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
    p = argparse.ArgumentParser(description="Train NSF4SL (PyTorch)")
    # Data
    p.add_argument("--data_path", type=str, default="../data")
    p.add_argument("--folds_dir",
                   type=str,
                   default=None,
                   help="Use shared canonical CV folds from this dir "
                   "(e.g. ../sl_comparison/folds) for the fair 4-model "
                   "comparison. When set, SLDataSplitter is bypassed.")
    p.add_argument("--embedding_path",
                   type=str,
                   default="../data/all_genes_kg_complex.pt",
                   help="Per-gene feature file (default: kg_complex, the "
                   "closest analog to NSF4SL's original TransE-KG features)")
    p.add_argument("--no_standardize",
                   action="store_true",
                   help="Skip per-column z-scoring of input features")
    p.add_argument("--output_dir",
                   type=str,
                   default=None,
                   help="Default: results/nsf4sl_<cv_type>")
    # CV
    p.add_argument("--cv_type",
                   type=str,
                   default="cv1",
                   choices=["cv1", "cv2", "cv3"])
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--pos_neg_ratio", type=float, default=1.0)
    # Model / training (benchmark NSF4SL defaults)
    p.add_argument("--latent_size", type=int, default=256)
    p.add_argument("--momentum", type=float, default=0.995)
    p.add_argument("--aug_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--lr_step", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    # Upstream (JieZheng-ShanghaiTech/NSF4SL main.py) evaluates every 10 epochs
    # with patience 4 (=40 epochs) and selects on validation P@100. The previous
    # defaults (every 3 epochs, patience 5 = 15 epochs, on test AUPR) stopped
    # all five cv1 folds at epoch 18 and kept the epoch-3 checkpoint.
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--early_stop",
                   type=int,
                   default=4,
                   help="Patience in evaluations (not epochs)")
    p.add_argument("--select_metric",
                   type=str,
                   default="ndcg@50",
                   help="Checkpoint-selection metric, computed on the "
                   "validation split. Defaults to a RANKING metric "
                   "because NSF4SL is a negative-sample-free contrastive "
                   "ranker; 'aupr' restores discrimination-based "
                   "selection. Expect ranking selection to raise NDCG "
                   "and LOWER AUROC/AUPRG — no single choice wins both.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    main()
