#!/usr/bin/env python3
"""
Train SLGNN on the shared folds.

Two architectures (--arch):

  original (default) — the published SLGNN (Zhu et al., Bioinformatics 2023;
      github.com/zy972014452/SLGNN), ported to pure PyTorch in
      slgnn_orig_model.py: factor-aware message passing over SynLethKG
      (built by build_synlethkg.py into --kg_dir) plus one GAT per factor over
      the training SL graph, with learned gene and entity embeddings (no
      external gene features). Training is the original's: mini-batches of
      --batch_size labelled pairs, one full-graph forward per batch, loss
      BCEWithLogits(sigmoid(score)) + L2 + independence terms, Adam, one
      validation pass per epoch, patience 5, at most 100 epochs, node and
      message dropout on. The labelled pairs are the fold's train_pos (1) and
      train_neg (0): the original's pre-sampled label-0 rows are, here, the
      shared folds' screened negatives, like every model in the comparison.
      Each pair is trained in both orientations, as the original's fold files
      list it, and the rows are shuffled once per fold (the original's fold
      files come pre-shuffled and its DataLoader does not shuffle).
  port — the earlier adaptation (slgnn_model.py): no KG, per-gene features from
      --embedding_path (default kg_complex), full-graph training. Kept so
      results produced with it stay reproducible.

Both share the harness the comparison depends on: the shared folds
(--folds_dir, required), early stopping AND checkpoint selection on --select_metric
(default `auprg`) of the VALIDATION split (val_pos = 1, val_neg = 0), scored
with sl_comparison/auprg.auprg on the same float32-shifted matrix entries the
shared evaluator grades; no test fallback (a fold set without a non-empty
val_pos AND val_neg is a hard error); a non-finite matrix gives NaN metrics
(treated as -inf); the selected checkpoint scores TEST once; an 8,844 x 8,844
float32 fold_<k>_<cv>_predictions.npy in shared gene order; every stale
matrix purged before fold 0. model_params.json records the fold stamp,
`selection_regime` ("val"), `selection_metric`, `arch`, `seed` and (original)
the KG stamp, so a row can never be misread.

Usage (from SLGNN/; manual runs default to results/slgnn_manual_<cv>, never
the comparison's results/slgnn_<cv>, which run_shared_models.sh writes):
    python build_synlethkg.py                      # once: ../data/synlethkg
    python train_slgnn.py --cv_type cv3 --folds_dir ../sl_comparison/folds
    python train_slgnn.py --arch port --cv_type cv3 \\
        --folds_dir ../sl_comparison/folds \\
        --embedding_path ../data/all_genes_kg_complex.pt
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
from slgnn_orig_model import SLGNNOriginal
from build_synlethkg import load_kg_for_genes
from sl_data import load_gene_features, load_shared_folds
from evaluation import evaluate_predictions_dict

# One parameter-counting definition for all four models — see ../sl_comparison.
sys.path.insert(0,
                str(Path(__file__).resolve().parent.parent / "sl_comparison"))
from model_params import (
    purge_stale_predictions,  # noqa: E402
    write_model_params)
# The SAME AUPRG definition evaluate_model.score_fold grades with.
from auprg import auprg  # noqa: E402

METRIC_KEYS = [
    "auroc",
    "f1",
    "aupr",
    "auprg",
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

# Per-architecture defaults for the flags both share. original: the SLGNN
# repository's src/train.py (dim 64, lr 3e-3, 100 epochs, validation every
# epoch, patience 5). port: Zhu et al.'s tuned width/lr for the adaptation.
ARCH_DEFAULTS = {
    "original":
    dict(dim=64, learning_rate=3e-3, epochs=100, eval_interval=1,
         early_stop=5),
    "port":
    dict(dim=256, learning_rate=2e-3, epochs=500, eval_interval=5,
         early_stop=10),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graded_pair_scores(mat, pairs):
    """Scores of `pairs` (N, 2) exactly as evaluate_model.score_fold grades them.

    The grader loads the exported float32 matrix (N_genes, N_genes), shifts it
    by `score - score.min() + 1` in float32, and reads score[a, b] in the
    fold file's orientation. The shift is elementwise, so applying it to the
    gathered pairs is bit-identical and avoids a full-matrix copy. Returns (N,).
    """
    mat = np.asarray(mat, dtype=np.float32)
    s = mat[pairs[:, 0], pairs[:, 1]]
    return (s - mat.min() + np.float32(1.0)).astype(np.float32)


def pair_auprg(mat, pos, neg):
    """AUPRG of pos (label 1) vs neg (label 0) pairs on the graded scores.
    NaN when a class is empty (sl_comparison/auprg.auprg's contract)."""
    pairs = np.concatenate([pos, neg]).reshape(-1, 2)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return auprg(y, graded_pair_scores(mat, pairs))


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


def evaluate_fold(mat, pos, neg, seen_pos):
    """Metrics of `mat`, a score_matrix() output (diagonal already 0), scored
    the way the grader scores the exported matrix: AUPRG on the shifted pair
    scores, cal_metrics on the shifted matrix."""
    if not np.isfinite(mat).all():
        # A diverged model: every metric is undefined. NaN (not a crash in
        # cal_metrics) lets selection treat it as -inf and patience count it.
        return {k: float("nan") for k in METRIC_KEYS}
    metrics = evaluate_predictions_dict(ranked_for_cal_metrics(mat), pos, neg,
                                        seen_pos)
    metrics["auprg"] = pair_auprg(mat, pos, neg)
    return metrics


def require_val(split, fold_label):
    """Return (val_pos, val_neg) as int32 (N, 2), or raise.

    There is deliberately NO test fallback: selecting on TEST makes the
    reported number a maximum over the evaluation budget, not a held-out
    estimate (early-stopping audit). Both classes are required, which also
    keeps val AUPRG defined (a single-class set has no AUPRG).
    """
    vp, vn = split.get("val_pos"), split.get("val_neg")
    if vp is None or vn is None or len(vp) == 0 or len(vn) == 0:
        raise SystemExit(
            f"{fold_label}: no usable validation split "
            f"(val_pos={0 if vp is None else len(vp)}, "
            f"val_neg={0 if vn is None else len(vn)}). Early stopping and "
            f"checkpoint selection run on validation only and will not fall "
            f"back to TEST. Pass --folds_dir with a fold set built by "
            f"sl_comparison/prepare_folds.py with --val_frac > 0.")
    return np.asarray(vp).astype(np.int32), np.asarray(vn).astype(np.int32)


def selection_step(value, best, patience, early_stop):
    """One early-stopping / checkpoint decision.

    NaN is treated as -inf, consistently: it is never an improvement and it
    counts toward patience (the caller logs it). Improvement is strict `>`,
    as before. Returns (improved, best, patience, stop).
    """
    value = float(value)
    if np.isnan(value):
        value = -np.inf
    if value > best:
        return True, value, 0, False
    patience += 1
    return False, best, patience, patience >= early_stop


def build_port(args, split, feats, device):
    """The adaptation (slgnn_model.SLGNN) and its full-graph epoch."""
    num_nodes = split["train_adj"].shape[0]
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
    criterion = nn.BCEWithLogitsLoss()
    edges = torch.from_numpy(split["train_edges"].astype(np.int64)).to(device)
    labels = torch.from_numpy(split["train_labels"].astype(
        np.float32)).to(device)
    ga, gb = edges[:, 0], edges[:, 1]

    def run_epoch(optimizer):
        model.train()
        node_emb, cor = model.encode()
        logits, emb_loss = model.pair_logits(node_emb, ga, gb)
        loss = criterion(logits, labels) + emb_loss \
            + args.sim_regularity * cor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), float(cor)

    return model, run_epoch


def original_train_pairs(train_edges, train_labels, seed):
    """The original's training rows from the fold's training pairs.

    Every pair in both orientations, (a, b) and (b, a), with its label: the
    original's fold files list each of its 36,402 SL and 36,402 non-SL pairs
    both ways (72,804 rows per class), and its score gene_a . entity_b is not
    symmetric, so it trains S[a, b] and S[b, a]. The shared folds list each
    pair once, as a < b. Only training pairs are doubled, so no pair crosses
    a split (the original's row-level split puts the reverse of most of its
    test pairs in train; that is not reproduced). Then one fixed order, used
    every epoch (the original's DataLoader has no shuffle; its fold files are
    pre-shuffled, ours list pos then neg).

    train_edges (N, 2), train_labels (N,) -> edges (2N, 2) int64,
    labels (2N,) float32.
    """
    e = np.asarray(train_edges, dtype=np.int64)
    y = np.asarray(train_labels, dtype=np.float32)
    e, y = np.vstack([e, e[:, ::-1]]), np.concatenate([y, y])
    order = np.random.RandomState(seed).permutation(len(e))
    return e[order], y[order]


def build_original(args, split, kg, device, fold_idx):
    """The published model (slgnn_orig_model.SLGNNOriginal) and its epoch:
    the original src/train.py loop over mini-batches, without its per-epoch
    test pass."""
    num_nodes = split["train_adj"].shape[0]
    model = SLGNNOriginal(
        n_genes=num_nodes,
        n_entities=kg["n_entities"],
        n_relations=kg["n_relations"],
        gene_entity=kg["gene_entity"],
        edge_index=kg["edge_index"],
        edge_type=kg["edge_type"],
        sl_adj=split["train_adj"],
        dim=args.dim,
        context_hops=args.n_hops,
        n_factors=args.n_factors,
        ind=args.ind,
        l2=args.l2_weight,
        sim_regularity=args.sim_regularity,
        node_dropout=not args.no_node_dropout,
        node_dropout_rate=args.node_dropout_rate,
        mess_dropout=not args.no_mess_dropout,
        mess_dropout_rate=args.mess_dropout_rate,
        device=device,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # split["train_edges"] itself stays one row per pair (train_fold masks
    # seen positives with it).
    tr_edges, tr_labels = original_train_pairs(split["train_edges"],
                                               split["train_labels"],
                                               args.seed + fold_idx)
    print(f"  training rows: {len(tr_edges):,} (each training pair in both "
          f"orientations, as the original's fold files list them)")
    edges = torch.from_numpy(tr_edges).to(device)
    labels = torch.from_numpy(tr_labels).to(device)

    def run_epoch(optimizer):
        model.train()
        running_loss, running_cor, n_batches = 0.0, 0.0, 0
        for s in range(0, len(edges), args.batch_size):
            gene_a = edges[s:s + args.batch_size, 0]
            gene_b = edges[s:s + args.batch_size, 1]
            logits, emb_loss, cor_loss, cor = model(gene_a, gene_b)
            # The original applies BCEWithLogitsLoss to sigmoid(score).
            loss = criterion(logits, labels[s:s + args.batch_size])
            loss = loss + emb_loss + cor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_cor += float(cor)
            n_batches += 1
        return running_loss / n_batches, running_cor / n_batches

    return model, run_epoch


def train_fold(args, fold_idx, n_folds, split, device, feats=None, kg=None):
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

    test_edges = split["test_edges"].astype(np.int32)
    test_labels = split["test_labels"]
    test_pos = test_edges[test_labels == 1]
    test_neg = test_edges[test_labels == 0]
    seen_pos = split["train_edges"][split["train_labels"] == 1].astype(
        np.int32)
    # Early stopping and checkpoint selection run on the VALIDATION split
    # (carved from the fold's training pairs) so the reported test metrics are
    # never the selection criterion. No val -> hard error, never TEST.
    val_pos, val_neg = require_val(split, f"Fold {fold_idx + 1}")
    sel_name = "val"
    print(f"Train: {int((split['train_labels']==1).sum())} pos, "
          f"{int((split['train_labels']==0).sum())} neg | "
          f"Val: {len(val_pos)} pos, {len(val_neg)} neg | "
          f"Test: {len(test_pos)} pos, {len(test_neg)} neg | "
          f"select on {sel_name}:{args.select_metric}")

    if args.arch == "original":
        model, run_epoch = build_original(args, split, kg, device, fold_idx)
    else:
        model, run_epoch = build_port(args, split, feats, device)

    # selection_regime travels with the predictions (model_params.json ->
    # evaluate_model.build_summary -> compare.py), so the table itself shows
    # which rows were val-selected. Nothing else could distinguish a row
    # whose checkpoint was picked on TEST from one picked honestly.
    # seed: compare.py's per-row training-seed record (measure-only).
    provenance = {"arch": args.arch, "seed": getattr(args, "seed", None)}
    if kg is not None:
        provenance["kg_stamp"] = kg["stamp"]
    write_model_params(args.output_dir,
                       model,
                       folds_dir=getattr(args, "folds_dir", None),
                       cv_type=getattr(args, "cv_type", None),
                       selection_regime=sel_name,
                       selection_metric=args.select_metric,
                       **provenance)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_score = -np.inf
    best_metrics = None
    best_state = None
    best_epoch = None
    patience = 0
    n_nan = 0

    for epoch in range(args.epochs):
        loss_value, cor_value = run_epoch(optimizer)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = evaluate_fold(model.score_matrix(), val_pos, val_neg,
                                    seen_pos)
            value = float(metrics[args.select_metric])
            extra = {k: metrics[k] for k in ("auprg", "auroc", "aupr")
                     if k != args.select_metric}
            print(f"  epoch {epoch+1:3d} | loss {loss_value:.4f} "
                  f"cor {cor_value:.4f} | {sel_name} "
                  f"{args.select_metric.upper()} {value:.4f} (select) " +
                  " ".join(f"{k.upper()} {v:.4f}" for k, v in extra.items()))
            if np.isnan(value):
                n_nan += 1
                print(f"  WARNING: {sel_name} {args.select_metric} is NaN at "
                      f"epoch {epoch + 1} — treated as -inf (not an "
                      f"improvement; counts toward patience).")
            improved, best_score, patience, stop = selection_step(
                value, best_score, patience, args.early_stop)
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            elif stop:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Restore the selected checkpoint, THEN score test exactly once. Test is
    # never consulted during training, so this number is honest held-out.
    # No finite validation score ever -> no defensible checkpoint; refuse
    # rather than silently export whatever the last epoch left behind.
    if best_state is None:
        raise RuntimeError(
            f"Fold {fold_idx + 1}: no finite {sel_name} {args.select_metric} "
            f"was recorded ({n_nan} NaN evaluation(s)), so no checkpoint can "
            f"be selected. The fold's prediction matrix is left MISSING.")
    model.load_state_dict(best_state)
    print(f"  Restored epoch-{best_epoch} checkpoint ({sel_name} "
          f"{args.select_metric} {best_score:.4f}"
          f"{f'; {n_nan} NaN evaluation(s) skipped' if n_nan else ''})")
    # One forward: the test metrics describe exactly the exported matrix
    # (index_add_ is nondeterministic on CUDA, so a second pass can differ).
    mat = model.score_matrix().astype(np.float32, copy=False)
    best_metrics = evaluate_fold(
        mat, test_pos, test_neg,
        graded_seen(getattr(args, "folds_dir", None), test_pos, seen_pos))
    best_metrics["best_epoch"] = best_epoch
    best_metrics[f"{sel_name}_{args.select_metric}"] = best_score
    np.save(
        Path(args.output_dir) /
        f"fold_{fold_idx}_{args.cv_type}_predictions.npy", mat)
    print(f"  Best fold {fold_idx + 1}: AUPR={best_metrics['aupr']:.4f}, "
          f"AUROC={best_metrics['auroc']:.4f}, F1={best_metrics['f1']:.4f}")
    return best_metrics


def main():
    args = parse_args()
    for key, value in ARCH_DEFAULTS[args.arch].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.arch == "original":
        if args.embedding_path is not None:
            raise SystemExit(
                "--embedding_path is not an input of the original SLGNN (it "
                "learns gene and entity embeddings over SynLethKG, --kg_dir). "
                "Drop it, or pass --arch port to train the feature-based "
                "adaptation.")
    elif args.embedding_path is None:
        args.embedding_path = "../data/all_genes_kg_complex.pt"
    if args.output_dir is None:
        # Never the comparison's results/slgnn_<cv>: run_shared_models.sh
        # writes it (passing its $SEED, default 42, and that --output_dir
        # explicitly) and external_transfer.py reads its matrices, so a
        # manual run there would silently replace them under the graded
        # SLGNN_<cv>.json.
        args.output_dir = (f"results/slgnn_manual_{args.cv_type}"
                           if args.arch == "original" else
                           f"results/slgnn_port_manual_{args.cv_type}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arch: {args.arch}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Folds: the shared canonical folds (required; selection needs their
    # validation split). The vendored data_split.SLDataSplitter has none, so
    # the trainer no longer offers it.
    gene_order, num_nodes, splits = load_shared_folds(args.folds_dir,
                                                      args.cv_type)
    print(f"Shared folds from {args.folds_dir}/{args.cv_type}: "
          f"{len(splits)} folds, {num_nodes} genes")

    # Fail fast, before any fold trains or any old matrix is purged: every
    # fold must carry a two-class validation split (no TEST fallback).
    for split in splits:
        require_val(split, f"{args.cv_type} fold {split['fold'] + 1}")

    feats, kg, inputs = None, None, {}
    if args.arch == "original":
        kg = load_kg_for_genes(args.kg_dir,
                               gene_order,
                               inverse_r=not args.no_inverse_r)
        print(f"SynLethKG ({args.kg_dir}): {kg['n_entities']:,} entities, "
              f"{kg['n_relations']} relations, "
              f"{kg['edge_index'].shape[1]:,} edges | "
              f"{kg['n_in_kg']}/{num_nodes} genes in the KG, "
              f"{num_nodes - kg['n_in_kg']} as isolated entities")
        inputs = {
            "kg_dir": args.kg_dir,
            "kg_stamp": kg["stamp"],
            "genes_in_kg": kg["n_in_kg"]
        }
    else:
        feats, matched = load_gene_features(
            args.embedding_path,
            gene_order,
            standardize=not args.no_standardize)
        print(f"Genes: {num_nodes} | features: {feats.shape[1]}d | "
              f"matched {matched}/{num_nodes}")
        inputs = {"embedding_path": args.embedding_path}

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
        fold_metrics.append(
            train_fold(args, split["fold"], len(splits), split, device,
                       feats=feats, kg=kg))

    summary = {"cv_type": args.cv_type, "arch": args.arch, **inputs,
               "folds": fold_metrics}
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
    p = argparse.ArgumentParser(description="Train SLGNN (PyTorch)")
    p.add_argument("--arch",
                   type=str,
                   default="original",
                   choices=["original", "port"],
                   help="original: the published SLGNN over SynLethKG "
                   "(--kg_dir). port: the earlier KG-free adaptation on "
                   "--embedding_path features.")
    p.add_argument("--folds_dir",
                   type=str,
                   required=True,
                   help="Shared canonical CV folds (e.g. "
                   "../sl_comparison/folds) built by "
                   "sl_comparison/prepare_folds.py with --val_frac > 0; "
                   "selection needs their validation split. The number of "
                   "folds is the number of fold_* directories.")
    p.add_argument("--kg_dir",
                   type=str,
                   default="../data/synlethkg",
                   help="[original] SynLethKG built by build_synlethkg.py")
    p.add_argument("--embedding_path",
                   type=str,
                   default=None,
                   help="[port] gene features (default "
                   "../data/all_genes_kg_complex.pt); refused with "
                   "--arch original")
    p.add_argument("--no_standardize",
                   action="store_true",
                   help="[port] skip feature standardisation")
    p.add_argument("--output_dir",
                   type=str,
                   default=None,
                   help="Default: results/slgnn_manual_<cv_type> "
                   "(port: results/slgnn_port_manual_<cv_type>). The "
                   "comparison's results/slgnn_<cv_type> is written only by "
                   "sl_comparison/slurm/run_shared_models.sh, which passes "
                   "it explicitly.")
    p.add_argument("--cv_type",
                   type=str,
                   default="cv1",
                   choices=["cv1", "cv2", "cv3"])
    # Shared model / training flags. None = the --arch default (ARCH_DEFAULTS):
    # original follows the SLGNN repository's src/train.py; port follows Zhu
    # et al., Bioinformatics 2023 (PMC9907046), which tuned dim over
    # {32,64,128,256} -> 256, layers over {1,2,3} -> 3, lr -> 0.002 (the port's
    # earlier defaults, dim 64 / 2 hops / lr 3e-3 / 100 epochs, stopped it
    # while test AUPR was still rising).
    p.add_argument("--dim",
                   type=int,
                   default=None,
                   help="Embedding size (original 64, port 256)")
    p.add_argument("--n_factors", type=int, default=4)
    p.add_argument("--n_hops",
                   type=int,
                   default=3,
                   help="Message-passing hops (the original's "
                   "--context_hops; paper: 3)")
    p.add_argument("--l2_weight", type=float, default=1e-4)
    p.add_argument("--sim_regularity", type=float, default=1e-3)
    p.add_argument("--learning_rate",
                   type=float,
                   default=None,
                   help="original 3e-3, port 2e-3")
    p.add_argument("--epochs",
                   type=int,
                   default=None,
                   help="Cap only — patience should terminate the run "
                   "(original 100, port 500). If a fold's best epoch equals "
                   "this, raise it and re-run.")
    p.add_argument("--eval_interval",
                   type=int,
                   default=None,
                   help="Epochs between validation passes (original 1, "
                   "port 5)")
    p.add_argument("--early_stop",
                   type=int,
                   default=None,
                   help="Patience in evaluations, not epochs (original 5, "
                   "port 10)")
    # original only (src/train.py flags)
    p.add_argument("--batch_size",
                   type=int,
                   default=2048,
                   help="[original] training rows (directed pairs, both "
                   "orientations of each training pair) per step; each "
                   "step runs the full GNN. 2048 is src/train.py's default "
                   "(the authors' README example passes 1024)")
    p.add_argument("--ind",
                   type=str,
                   default="distance",
                   choices=["distance", "cosine", "mi"],
                   help="[original] factor-independence measure")
    p.add_argument("--node_dropout_rate", type=float, default=0.5)
    p.add_argument("--mess_dropout_rate", type=float, default=0.1)
    p.add_argument("--no_node_dropout",
                   action="store_true",
                   help="[original] disable KG-edge / SL-adjacency dropout")
    p.add_argument("--no_mess_dropout",
                   action="store_true",
                   help="[original] disable message dropout")
    p.add_argument("--no_inverse_r",
                   action="store_true",
                   help="[original] do not add inverse KG relations")
    # port only
    p.add_argument("--dropout",
                   type=float,
                   default=0.1,
                   help="[port] attention/message dropout")
    p.add_argument("--select_metric",
                   type=str,
                   default="auprg",
                   choices=METRIC_KEYS,
                   help="Early-stopping / checkpoint-selection metric, on "
                   "the VALIDATION split only. 'auprg' (default) is "
                   "sl_comparison/auprg.auprg on the graded pair scores; "
                   "'aupr' restores the previous (vendored cal_metrics) "
                   "criterion. NaN counts as -inf.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    main()
