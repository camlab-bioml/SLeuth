#!/usr/bin/env python3
"""
Zero-parameter degree-prior control row for the four-model comparison.

    score(i, j) = log1p(deg_train(i)) + log1p(deg_train(j))

No features, no embeddings, no learning — just how many training-set SL
partners each gene already has. Scored through sl_comparison/evaluate_model.py's
score_fold(), i.e. the identical affine shift, seen mask, cal_metrics call and
corrected-retrieval companions every model row gets.

WHY THIS ROW IS MANDATORY
-------------------------
SynLethDB positives are hub-enriched relative to the experimental negatives:
on cv1/fold_0 the endpoints of test positives average 197.8 training-graph
neighbours against 13.3 for test negatives. On the 2026-08-06 shared-fold run
this heuristic scored cv1 AUROC 0.9059 / AUPRG 0.9511 / NDCG@10 0.4081 — a tie
with siamese_sl (0.9078 / 0.9533 / 0.1129) on the classification metrics and a
3.6x win on retrieval, while beating SLGNN and NSF4SL outright. Any cv1 claim
made without this row is a claim that a model beat a control nobody ran.

It collapses exactly where the cold-start regime begins — cv2 AUROC 0.6402,
cv3 exactly 0.5000 (its scores are constant when both genes are unseen) — which
is what makes cv2 and cv3 the informative comparisons.

NOTE ON THE cv3 AUPR CELL: with a constant score vector the vendored
trapezoidal `aupr` returns exactly (1 + pi) / 2 — read pi off the run's own
`baseline` column rather than trusting a quoted constant, which goes stale the
moment the fold set is regenerated. This ranks the control FIRST in that
column. That is a defect of the metric, not a property of the control; the
`ap` column (average precision) correctly returns pi. Report `ap`.

Usage (from the repo root):
  python sl_comparison/degree_baseline.py --cv_type cv1 \
      --out sl_comparison/results/DegreePrior_cv1.json
"""

import argparse
from pathlib import Path

import numpy as np

from evaluate_model import (build_summary, fold_dirs_for, score_fold,
                            write_summary)

MODEL_NAME = "DegreePrior"


def degree_matrix(fold_dir, num_nodes):
    """log1p training-degree outer sum for one fold. Never sees test edges."""
    train_pos = np.load(Path(fold_dir) / "train_pos.npy").astype(np.int64)
    deg = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(deg, train_pos[:, 0], 1.0)
    np.add.at(deg, train_pos[:, 1], 1.0)
    s = np.log1p(deg)
    return (s[:, None] + s[None, :]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Degree-prior control baseline")
    ap.add_argument("--folds_dir", default="sl_comparison/folds")
    ap.add_argument("--cv_type", required=True, choices=["cv1", "cv2", "cv3"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    genes = Path(args.folds_dir) / "genes.txt"
    num_nodes = sum(1 for ln in open(genes) if ln.strip())

    fold_metrics = []
    for d in fold_dirs_for(args.folds_dir, args.cv_type):
        k = int(d.name.split("_")[1])
        # Built in memory and discarded: a persisted num_nodes^2 float32 matrix
        # is ~313 MB per fold, and nothing downstream needs it.
        m = score_fold(degree_matrix(d, num_nodes), d,
                       f"{MODEL_NAME}/{args.cv_type}/fold_{k}")
        if m is None:
            continue
        fold_metrics.append(m)
        print(f"  fold {k}: AUROC {m['auroc']:.4f} AP {m['ap']:.4f} "
              f"AUPRG {m['auprg']:.4f} NDCG@10 {m['ndcg@10']:.4f}")

    if not fold_metrics:
        raise SystemExit(f"No scorable folds for {args.cv_type}")

    summary = build_summary(MODEL_NAME, args.cv_type, args.folds_dir,
                            fold_metrics)
    # A control with no learned parameters. Stated explicitly so compare.py
    # renders "DegreePrior(0)" rather than silently omitting the count and
    # leaving a reader to wonder whether it just failed to record.
    summary["trainable_params"] = 0
    summary["total_params"] = 0
    summary["model_class"] = "log1p(deg_train[i]) + log1p(deg_train[j])"
    # Nothing is fitted, so no split is ever consulted for selection. Stated
    # explicitly rather than left blank, so compare.py's regime check does not
    # report this row as "unknown" alongside a row that genuinely is.
    summary["selection_regime"] = "n/a (no training, no selection)"
    write_summary(summary, args.out)


if __name__ == "__main__":
    main()
