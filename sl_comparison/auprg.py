#!/usr/bin/env python3
"""
AUPRG (Area Under the Precision-Recall-Gain curve; Flach & Kull, NeurIPS 2015)
and normalized AUPR — BASELINE-ANCHORED SKILL SCORES used as the common
yardstick across folds whose test sets differ in prevalence.

"Anchored", not "independent": both are constructed so a random ranker scores 0
and a perfect one 1 at ANY prevalence, which is what makes them poolable across
folds. Neither is prevalence-INVARIANT — hold the ranker fixed and vary pi and
both move (measured 2026-08-06 on siamese cv3/fold_1, AUROC held at 0.8105 and
pi varied synthetically: pi 0.2550 ->
aupr_norm 0.5509 / auprg 0.8473; pi 0.0167 -> 0.1286 / 0.8980). AUROC is the
only genuinely invariant one.

Vendored verbatim from siamese_sl/train.py so the four-model comparison scores
every model with the SAME AUPRG definition siamese reports as `*_any`.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def normalized_aupr(aupr, baseline):
    """Skill-score AUPR: (AUPR - pi) / (1 - pi); random -> 0, perfect -> 1."""
    if baseline is None or not np.isfinite(aupr) or not (0.0 < baseline < 1.0):
        return float("nan")
    return (aupr - baseline) / (1.0 - baseline)


def auprg(labels, scores):
    """Area Under the Precision-Recall-Gain curve.

    Uses the reference `prg` package when installed; otherwise a self-contained
    integrator. Returns NaN when prevalence is degenerate.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pi = float(np.mean(labels)) if labels.size else float("nan")
    if not (0.0 < pi < 1.0) or len(np.unique(labels)) < 2:
        return float("nan")

    try:
        import prg
        return float(prg.calc_auprg(prg.create_prg_curve(labels, scores)))
    except Exception:
        pass

    precision, recall, _ = precision_recall_curve(labels, scores)
    odds = pi / (1.0 - pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - odds * (1.0 - precision) / precision
        rec_gain = 1.0 - odds * (1.0 - recall) / recall
    valid = recall > 0
    rg_v, pg_v = rec_gain[valid], prec_gain[valid]
    rg = np.unique(rg_v)
    pg = np.array([pg_v[rg_v == u].max() for u in rg])

    # ANCHOR THE CURVE AT recall_gain = 0.
    #
    # AUPRG is the area over recall_gain in [0, 1]. Normally the curve contains
    # points with recall_gain < 0 (the first threshold retrieves one item, so
    # recall << pi) and the integration loop below clips the segment that
    # straddles 0. But precision_recall_curve COLLAPSES TIED SCORES into a
    # single point, so when the top tie block alone already pushes recall above
    # pi, every observed point has recall_gain > 0 and the strip [0, rg.min()]
    # is silently dropped — Flach & Kull's create_prg_curve inserts the
    # crossing point explicitly; this integrator did not.
    #
    # Within a tie block every item has the same score, so any sub-selection of
    # it has the same expected precision: precision_gain is flat across the
    # block, and the value at recall_gain = 0 is pg[0]. Prepending that point
    # restores the missing rectangle.
    #
    # Without this, a PERFECT ranking with 50 of 100 positives tied at the top
    # returns 0.200 instead of 1.0 — and because BCE keeps inflating logit
    # magnitude, float32 sigmoid saturation creates exactly such tie blocks as
    # training proceeds. AUPRG then FALLS while the ranking is unchanged, which
    # made it non-monotone in ranking quality: it is siamese's checkpoint
    # selection metric (train.py) and compare.py's first headline column.
    if len(rg) and rg[0] > 0.0:
        rg = np.concatenate(([0.0], rg))
        pg = np.concatenate(([pg[0]], pg))

    def _interp(r, r0, r1, p0, p1):
        if r1 == r0:
            return p1
        return p0 + (r - r0) / (r1 - r0) * (p1 - p0)

    area = 0.0
    for i in range(1, len(rg)):
        r0, r1, p0, p1 = rg[i - 1], rg[i], pg[i - 1], pg[i]
        lo, hi = max(0.0, min(r0, r1)), min(1.0, max(r0, r1))
        if hi <= lo:
            continue
        area += (hi - lo) * (_interp(lo, r0, r1, p0, p1) +
                             _interp(hi, r0, r1, p0, p1)) / 2.0
    return float(area)
