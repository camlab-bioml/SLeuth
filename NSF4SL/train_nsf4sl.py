#!/usr/bin/env python3
"""
Training script for the NSF4SL PyTorch implementation.

Reuses the SLMGAE benchmark machinery for exact comparability:
  - Train/val/test split : the shared folds from --folds_dir (required;
                           built by sl_comparison/prepare_folds.py with the
                           validation partition that selection needs).
  - Evaluation           : preprocess_benchmarking_paper.cal_metrics (the
                           official benchmark evaluator — AUROC/F1/AUPR +
                           NDCG/Recall/Precision/MAP @ k)

NSF4SL is negative-sample-free: each fold trains ONLY on the fold's positive SL
pairs (BYOL bootstrap between the two genes + feature-masked augmentations).
Fold negatives are never trained on: the validation negatives drive early
stopping and checkpoint selection, and the test negatives are used only to
score the test set.

Gene features are the ORIGINAL ones by default (--features transe): the 400-d
SynLethKG TransE_l2 entity embeddings the NSF4SL repo ships, fed raw, with a
masked dimension reset to the mean over all KG entities; build them once with
build_transe_features.py. The training loop follows the original main.py: each
positive in both orientations, patience counted over all non-improving
evaluations, the last partial batch kept. --features embedding,
--legacy_loop and --select_metric ndcg@50 restore this port's earlier
configuration (an all_genes_*.pt file, z-scored, default kg_complex), not its
exact runs: validation NDCG@50 is now computed on the grader's shifted matrix,
whereas the earlier runs ranked the raw one (where cal_metrics' literal-0
diagonal can outrank negative-scored candidates), so the selected epochs can
differ. Early stopping and checkpoint selection use the VALIDATION split
(val_pos = 1, val_neg = 0, carved from the fold's training pairs by
prepare_folds) on --select_metric, which defaults to AUPRG — the same sl_comparison/auprg.py the shared evaluator grades
with, applied to the exported pair score. A fold set with no two-class
validation partition is REFUSED (it used to fall back to selecting on test).
The regime and metric travel into model_params.json as `selection_regime` and
`selection_metric`.

Usage (from NSF4SL/):
    python build_transe_features.py --download     # once
    python train_nsf4sl.py --cv_type cv3 --folds_dir ../sl_comparison/folds
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
from sl_data import load_gene_features, load_shared_folds, load_transe_features
from evaluation import evaluate_predictions_dict

# One parameter-counting definition for all four models — see ../sl_comparison.
sys.path.insert(0,
                str(Path(__file__).resolve().parent.parent / "sl_comparison"))
from model_params import (
    purge_stale_predictions,  # noqa: E402
    write_model_params)
# The SAME AUPRG the shared evaluator (sl_comparison/evaluate_model.py) grades
# with, so the selection criterion and the published column cannot diverge.
from auprg import auprg  # noqa: E402

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

# Valid --select_metric values: AUPRG (default) or any cal_metrics key (the
# pre-AUPRG behaviour, e.g. 'ndcg@50'). All are higher-is-better.
SELECT_METRICS = ["auprg"] + METRIC_KEYS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PosPairDataset(Dataset):
    """Positive SL pairs with feature-masking augmentation (negative-free).

    Each item yields the two genes' features and augmented copies where a random
    ``aug_ratio`` fraction of feature dimensions is reset to ``col_mean`` (for
    TransE features the mean over all KG entities, the original's
    kgemb_data_mean; for an all_genes_*.pt file the mean over the fold
    universe).
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


def ranked_for_cal_metrics(mat):
    """The exported matrix as evaluate_model.score_fold hands it to cal_metrics:
    shifted by (score - min + 1) in float32. cal_metrics writes a literal 0 on
    the diagonal and -999999 on seen pairs; on raw signed scores that 0 would
    outrank real candidates, so the ranking metrics here would not be the
    grader's."""
    mat = np.asarray(mat, dtype=np.float32)
    return (mat - mat.min() + np.float32(1.0)).astype(np.float32)


def graded_seen(folds_dir, test_pos, train_pos):
    """The known SL pairs the grader suppresses on TEST (evaluate_model.
    seen_for): every pair in <folds_dir>/all_pos.npy except this fold's test
    positives; train_pos when all_pos.npy is absent. TEST reporting only:
    validation selection masks train_pos alone, since all_pos contains the
    test positives."""
    p = Path(folds_dir) / "all_pos.npy" if folds_dir else None
    if p is None or not p.exists():
        return train_pos
    all_pos = np.load(p).astype(np.int64)
    keep = set(map(tuple, np.asarray(test_pos, dtype=np.int64).tolist()))
    mask = np.fromiter((tuple(e) not in keep for e in all_pos.tolist()),
                       dtype=bool, count=len(all_pos))
    return all_pos[mask]


def evaluate_fold(model, feats_device, num_nodes, test_pos, test_neg,
                  seen_pos):
    """Score matrix -> benchmark metrics on the given pairs, on the matrix
    _save_predictions exports, shifted as the grader shifts it (cal_metrics
    zeroes the diagonal itself)."""
    mat = score_matrix(model, feats_device)
    if not np.isfinite(mat).all():
        # Diverged model: metrics undefined -> NaN, not a crash in cal_metrics.
        return {k: float("nan") for k in METRIC_KEYS}
    return evaluate_predictions_dict(ranked_for_cal_metrics(mat), test_pos,
                                     test_neg, seen_pos)


def require_validation(split):
    """Return the fold's (val_pos, val_neg) as int32 (P, 2) / (Q, 2) arrays.

    Raises SystemExit when either is empty. Selection used to fall back to the
    TEST split here, which reports a maximum over every evaluation of the
    edges being graded rather than a held-out number. A fold set built
    without prepare_folds --val_frac (load_shared_folds then returns empty
    val_*) is refused too.
    """
    val_pos = np.asarray(split.get("val_pos", np.empty((0, 2)))).astype(
        np.int32).reshape(-1, 2)
    val_neg = np.asarray(split.get("val_neg", np.empty((0, 2)))).astype(
        np.int32).reshape(-1, 2)
    if len(val_pos) == 0 or len(val_neg) == 0:
        raise SystemExit(
            f"fold {split.get('fold', '?')}: validation split has "
            f"{len(val_pos)} positive / {len(val_neg)} negative pairs. "
            f"Early stopping and checkpoint selection need BOTH classes on "
            f"validation and never fall back to TEST. Point --folds_dir at "
            f"a fold set built by sl_comparison/prepare_folds.py --val_frac "
            f"> 0.")
    return val_pos, val_neg


def val_auprg(mat, val_pos, val_neg):
    """AUPRG of the GRADED pair score on the validation pairs.

    mat:     (N, N) score matrix from score_matrix() — the matrix
             _save_predictions exports. The score is taken after the exact
             float32 shift evaluate_model.py grades with (score - min + 1), so
             selection ranks the val pairs exactly as the grader would (the
             shift is order-preserving but can merge near-tied float32 values).
    val_pos: (P, 2) gene-index pairs, label 1.
    val_neg: (Q, 2) gene-index pairs, label 0.

    Returns NaN (never selectable) when ANY entry of the matrix is non-finite,
    not only a val entry: a checkpoint whose exported matrix holds NaN/inf
    elsewhere would be graded as NaN, so it must not win selection. Also NaN
    when the val set is single-class (auprg's own degenerate-prevalence guard).
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.size == 0 or not np.isfinite(mat).all():
        return float("nan")
    graded = (mat - mat.min() + np.float32(1.0)).astype(np.float32)
    s_pos = graded[val_pos[:, 0], val_pos[:, 1]]
    s_neg = graded[val_neg[:, 0], val_neg[:, 1]]
    scores = np.concatenate([s_pos, s_neg])
    if scores.size == 0:
        return float("nan")
    labels = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    return auprg(labels, scores)


def feature_provenance(args):
    """Which features and training loop made this run's matrices.

    Written into model_params.json, the one trainer file evaluate_model.py
    merges into the graded row (results.json never reaches it). Call after
    main() has resolved args.embedding_path.
    """
    return {
        "features": args.features,
        "feature_file": (args.transe_path if args.features == "transe" else
                         args.embedding_path),
        "standardized": (args.features == "embedding"
                         and not args.no_standardize),
        "legacy_loop": bool(args.legacy_loop),
    }


def update_selection(score, best, patience, reset=False):
    """One early-stopping step on a higher-is-better selection score.

    Returns (improved, best, patience, finite). A non-finite score (NaN from a
    single-class val set or from non-finite model scores) is treated as -inf:
    it can never become the selected checkpoint and it counts toward patience
    exactly like a non-improving evaluation. With best starting at -inf, the
    first FINITE evaluation is always taken; strict '>' keeps the earliest of
    tied checkpoints.

    patience counts non-improving evaluations. The original main.py never
    resets its early_stop_cnt, so it stops at the early_stop-th non-improving
    evaluation IN TOTAL; reset=True (--legacy_loop) zeroes the count on every
    improvement, which this port did before.
    """
    finite = bool(np.isfinite(score))
    effective = float(score) if finite else float("-inf")
    if effective > best:
        return True, effective, (0 if reset else patience), finite
    return False, best, patience + 1, finite


def train_fold(args, fold_idx, n_folds, split, feats_np, col_mean, device):
    print(f"\n{'='*60}\nFold {fold_idx + 1}/{n_folds}\n{'='*60}")

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

    # Int indices: train_pos is handed to cal_metrics as seen_index, which
    # fancy-indexes and requires an int dtype.
    train_edges = split["train_edges"].astype(np.int64)
    train_labels = split["train_labels"]
    test_edges = split["test_edges"].astype(np.int32)
    test_labels = split["test_labels"]

    # Negative-sample-free: train on positives only.
    train_pos = train_edges[train_labels == 1]
    test_pos = test_edges[test_labels == 1]
    test_neg = test_edges[test_labels == 0]
    # Selection runs on VALIDATION only — never on test. The metric is
    # --select_metric: AUPRG by default (one criterion for all four models,
    # the same function the shared evaluator publishes), or a cal_metrics key
    # such as 'ndcg@50' (the authors' ranking criterion, the previous
    # default). A fold without a two-class val split raises instead of
    # silently selecting on TEST.
    val_pos, val_neg = require_validation(split)
    sel_pos, sel_neg, sel_name = val_pos, val_neg, "val"
    print(f"Train positives: {len(train_pos)} | "
          f"Val: {len(val_pos)} pos, {len(val_neg)} neg | "
          f"Test: {len(test_pos)} pos, {len(test_neg)} neg | "
          f"select on {sel_name}:{args.select_metric}")

    # The original's train.npy lists every positive as (a, b) AND (b, a), so an
    # epoch passes over each pair twice, with independent augmentations and
    # twice the optimiser / EMA steps per StepLR epoch. The shared folds hold
    # each pair once (i < j); mirror them unless --legacy_loop.
    train_pairs = (train_pos if args.legacy_loop else np.vstack(
        [train_pos, train_pos[:, ::-1]]))
    rng = random.Random(args.seed + fold_idx)
    dataset = PosPairDataset(train_pairs, feats_np, col_mean, args.aug_ratio,
                             rng)
    # The original keeps the last partial batch. A batch of ONE pair cannot
    # pass BatchNorm in train mode (the original would crash there), so that
    # single case is dropped. --legacy_loop drops every partial batch.
    drop_last = (len(dataset) > args.batch_size if args.legacy_loop else
                 len(dataset) % args.batch_size == 1)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=drop_last)

    feats_device = torch.from_numpy(feats_np).float().to(device)

    model = Net(feats_np.shape[1], args.latent_size, args.momentum).to(device)
    # selection_regime travels with the predictions (model_params.json ->
    # evaluate_model.build_summary -> compare.py), so the table itself shows
    # which rows were val-selected. Nothing else could distinguish a row
    # whose checkpoint was picked on TEST from one picked honestly.
    # seed: compare.py's per-row training-seed record (measure-only).
    # feature_provenance: the graded row then says whether it is the
    # original-feature model or a --features embedding / --legacy_loop run.
    write_model_params(args.output_dir,
                       model,
                       folds_dir=getattr(args, "folds_dir", None),
                       cv_type=getattr(args, "cv_type", None),
                       selection_regime=sel_name,
                       selection_metric=args.select_metric,
                       seed=getattr(args, "seed", None),
                       **feature_provenance(args))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # -inf, not -1.0: AUPRG is unbounded below (precision gain -> -inf as
    # precision -> 0), so a -1.0 floor could leave no checkpoint selected.
    best_score = float("-inf")
    best_epoch = None
    best_metrics = None
    best_state = None
    patience = 0
    n_evals = 0

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
            n_evals += 1
            if args.select_metric == "auprg":
                # Scored from the same matrix _save_predictions exports.
                score = val_auprg(score_matrix(model, feats_device), sel_pos,
                                  sel_neg)
                extra = ""
            else:
                metrics = evaluate_fold(model, feats_device,
                                        split["train_adj"].shape[0], sel_pos,
                                        sel_neg, train_pos)
                score = metrics[args.select_metric]
                extra = f" AUPR {metrics['aupr']:.4f}"
            improved, best_score, patience, finite = update_selection(
                score, best_score, patience, reset=args.legacy_loop)
            print(f"  epoch {epoch+1:3d} | loss {np.mean(losses):.4f} | "
                  f"{sel_name} {args.select_metric} {score:.4f}{extra}"
                  f"{' *' if improved else ''}")
            if not finite:
                print(f"  WARNING: {sel_name} {args.select_metric} is "
                      f"{score} at epoch {epoch + 1} — treated as -inf "
                      f"(never selected; counts toward patience).")
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            elif patience >= args.early_stop:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Every evaluation non-finite (or none ran): there is no checkpoint the
    # selection rule chose, and scoring whatever the last epoch left would be
    # arbitrary. Fail the fold (its matrix was already cleared above, so it is
    # MISSING, not stale) rather than publish an unselected model.
    if best_state is None:
        raise RuntimeError(
            f"fold {fold_idx}: no checkpoint selected on {sel_name} "
            f"{args.select_metric} ({n_evals} evaluations, all non-finite or "
            f"none run) — refusing to export an unselected model.")
    if best_epoch is not None:
        print(f"  Selected epoch {best_epoch} "
              f"({sel_name} {args.select_metric} {best_score:.4f})")

    # Restore the selected checkpoint, THEN score test exactly once. Test is
    # never consulted during training, so this number is honest held-out.
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = evaluate_fold(
        model, feats_device, split["train_adj"].shape[0], test_pos, test_neg,
        graded_seen(getattr(args, "folds_dir", None), test_pos, train_pos))
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
    if args.features == "transe" and (args.embedding_path
                                      or args.no_standardize):
        # A launcher still passing the old kg_complex path would otherwise be
        # silently ignored and the run mislabelled.
        raise SystemExit(
            "--embedding_path / --no_standardize only apply with --features "
            "embedding; the default --features transe reads --transe_path.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Folds: the shared canonical folds (required; selection needs their
    # validation split). The vendored data_split.SLDataSplitter has none, so
    # the trainer no longer offers it.
    gene_order, num_nodes, splits = load_shared_folds(args.folds_dir,
                                                      args.cv_type)
    print(f"Shared folds from {args.folds_dir}/{args.cv_type}: "
          f"{len(splits)} folds, {num_nodes} genes")

    # Refuse a fold set with no two-class validation split BEFORE any fold
    # trains (train_fold re-checks per fold).
    for split in splits:
        require_validation(split)

    # Per-gene features aligned to the shared gene index (genes.txt), and the
    # value the augmentation writes into a masked dimension (col_mean). BOTH
    # branches report the genes they had to substitute (`missing`): an empty
    # genes_without_features must mean "none", never "not tracked".
    if args.features == "transe":
        # Original: raw TransE rows; masked dims -> mean over ALL KG entities.
        feats, col_mean, missing = load_transe_features(
            args.transe_path, gene_order)
        feats_np = feats.numpy()
        print(f"Genes: {num_nodes} | features: SynLethKG TransE "
              f"{feats_np.shape[1]}d (raw) from {args.transe_path} | "
              f"matched {num_nodes - len(missing)}/{num_nodes}; "
              f"{len(missing)} genes with no linked KG entity get the "
              f"all-entity mean" + (f" (e.g. {', '.join(missing[:10])})"
                         if missing else ""))
    else:
        args.embedding_path = (args.embedding_path
                               or "../data/all_genes_kg_complex.pt")
        feats, matched, missing = load_gene_features(
            args.embedding_path,
            gene_order,
            standardize=not args.no_standardize)
        feats_np = feats.numpy()
        col_mean = feats_np.mean(axis=0)
        # `missing` here is genes with no usable row (absent, or an all-NaN row
        # the file stores for a gene it could not embed): their whole vector is
        # the column mean. Reported like the TransE branch's, so
        # genes_without_features means the same thing on both.
        print(f"Genes: {num_nodes} | features: {args.embedding_path} "
              f"{feats_np.shape[1]}d | matched {matched}/{num_nodes}; "
              f"{len(missing)} genes with no features get the column mean" +
              (f" (e.g. {', '.join(missing[:10])})" if missing else ""))
    print(f"Training loop: "
          f"{'legacy port' if args.legacy_loop else 'original main.py'}")

    # CLEAR EVERY FOLD'S LEFTOVER MATRIX BEFORE FOLD 0.
    #
    # The per-fold clear inside the loop only protects folds this run
    # actually reaches. A run that dies at fold 3 leaves folds 3-4 holding
    # the PREVIOUS run's matrices, and evaluate_model.py grades every
    # matrix it finds in the directory — publishing a mean over fresh and
    # foreign folds at exit 0. A crashed fold must be MISSING, not stale.
    purge_stale_predictions(args.output_dir, args.cv_type)
    # results.json is written only after the last fold: a crashed rerun
    # must not leave the previous run's summary beside this run's matrices.
    (Path(args.output_dir) / "results.json").unlink(missing_ok=True)

    fold_metrics = []
    for split in splits:
        m = train_fold(args, split["fold"], len(splits), split, feats_np,
                       col_mean, device)
        fold_metrics.append(m)

    # Aggregate and save summary.
    # feature_provenance() is the one definition of which features made this
    # run (the same dict model_params.json carries). The old 'embedding_path'
    # key held the TRANSE path on a transe run while summary['args']
    # ['embedding_path'] was None, i.e. the file contradicted itself.
    summary = {
        "cv_type": args.cv_type,
        **feature_provenance(args),
        "genes_without_features": missing,
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
    p.add_argument("--folds_dir",
                   type=str,
                   required=True,
                   help="Shared canonical CV folds (e.g. "
                   "../sl_comparison/folds) built by "
                   "sl_comparison/prepare_folds.py with --val_frac > 0; "
                   "selection needs their validation split. The number of "
                   "folds is the number of fold_* directories.")
    p.add_argument("--features",
                   type=str,
                   default="transe",
                   choices=["transe", "embedding"],
                   help="'transe' (default): the original NSF4SL features, "
                   "SynLethKG TransE_l2 entity embeddings from --transe_path, "
                   "raw. 'embedding': an all_genes_*.pt file from "
                   "--embedding_path, z-scored (this port's earlier runs).")
    p.add_argument("--transe_path",
                   type=str,
                   default="../data/nsf4sl_transe_features.pt",
                   help="build_transe_features.py output (--features transe)")
    p.add_argument("--embedding_path",
                   type=str,
                   default=None,
                   help="Per-gene feature file for --features embedding "
                   "(default ../data/all_genes_kg_complex.pt). Refused "
                   "with --features transe.")
    p.add_argument("--no_standardize",
                   action="store_true",
                   help="Skip per-column z-scoring (--features embedding "
                   "only; TransE features are never standardised)")
    p.add_argument("--output_dir",
                   type=str,
                   default=None,
                   help="Default: results/nsf4sl_standalone_<cv_type>. "
                   "results/nsf4sl_<cv_type> is the benchmark pipeline's "
                   "graded dir (sl_comparison/slurm/run_shared_models.sh, "
                   "default seed 42); write there only from that launcher.")
    # CV
    p.add_argument("--cv_type",
                   type=str,
                   default="cv1",
                   choices=["cv1", "cv2", "cv3"])
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
    # and stops at the 4th non-improving evaluation in total (its counter is
    # never reset), selecting on validation P@100. The previous defaults
    # (every 3 epochs, patience 5 = 15 epochs, on test AUPR) stopped all five
    # cv1 folds at epoch 18 and kept the epoch-3 checkpoint.
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--early_stop",
                   type=int,
                   default=4,
                   help="Stop at this many non-improving evaluations (not "
                   "epochs); counted in total as in the original, or "
                   "consecutively with --legacy_loop")
    p.add_argument("--legacy_loop",
                   action="store_true",
                   help="This port's loop before 2026-09-21: each positive "
                   "once per epoch (the original uses both orientations), "
                   "patience reset on improvement, every partial last batch "
                   "dropped. With --features embedding and --select_metric "
                   "ndcg@50 it restores the earlier kg_complex "
                   "configuration, not its exact runs: validation NDCG@50 "
                   "now ranks the grader's shifted matrix (the earlier runs "
                   "ranked the raw one), so the selected epochs can differ.")
    p.add_argument("--select_metric",
                   type=str,
                   default="auprg",
                   choices=SELECT_METRICS,
                   help="Early-stopping and checkpoint-selection metric, "
                   "computed on the validation split only. Default "
                   "'auprg' (sl_comparison/auprg.py, the shared "
                   "evaluator's AUPRG) is the one criterion all four "
                   "models use. 'ndcg@50' restores the previous "
                   "ranking-based selection (NSF4SL's authors select on "
                   "a ranking metric); expect it to raise NDCG and "
                   "LOWER AUROC/AUPRG — no single choice wins both.")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    # range(<= 0) trains and evaluates nothing, so the random init would be
    # exported as a "val-selected" model; interval 0 divides by zero.
    if args.epochs < 1:
        p.error(f"--epochs must be >= 1 (got {args.epochs})")
    if args.eval_interval < 1:
        p.error(f"--eval_interval must be >= 1 (got {args.eval_interval})")
    if args.output_dir is None:
        # Not results/nsf4sl_<cv>: that is the benchmark's graded dir
        # (run_shared_models.sh, --seed $SEED, 42 by default), also read by
        # external transfer, and a manual run must not overwrite it.
        args.output_dir = f"results/nsf4sl_standalone_{args.cv_type}"
    return args


if __name__ == "__main__":
    main()
