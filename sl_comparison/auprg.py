#!/usr/bin/env python3
"""
AUPRG (Area Under the Precision-Recall-Gain curve; Flach & Kull, NeurIPS 2015)
and normalized AUPR — BASELINE-ANCHORED SKILL SCORES used as the common
yardstick across folds whose test sets differ in prevalence.

"Anchored", not "independent": both are constructed so a random ranker scores 0
and a perfect one 1 at ANY prevalence, which is what makes them poolable across
folds. Neither is prevalence-INVARIANT — hold the ranker fixed and vary pi and
both move (measured 2026-08-06 on siamese cv3/fold_1 of the previous fold set,
with the superseded integrator described under auprg(); illustrative only.
AUROC held at 0.8105 and pi varied synthetically: pi 0.2550 ->
aupr_norm 0.5509 / auprg 0.8473; pi 0.0167 -> 0.1286 / 0.8980). AUROC is the
only genuinely invariant one.

This is the SINGLE AUPRG definition. siamese_sl/train.py loads this file by
path (_AUPRG_PY), and the NSF4SL / SLGNN / SLMGAE trainers, zero_cell_emb.py
and evaluate_model.py import it, so selection and grading score every model
with the same function siamese reports as `*_any`. It replaced train.py's
former local integrator, which disagreed with the reference prg.py; keep no
second copy.
"""

import numpy as np


def normalized_aupr(aupr, baseline):
    """Skill-score AUPR: (AUPR - pi) / (1 - pi); random -> 0, perfect -> 1.

    The random -> 0 anchor holds for average precision. Fed the trapezoidal
    auc(recall, precision) (cal_metrics' `aupr`), a constant (all-tied) scorer
    gets (1 + pi) / 2 and therefore 0.5 here, not 0.
    """
    if baseline is None or not np.isfinite(aupr) or not (0.0 < baseline < 1.0):
        return float("nan")
    return (aupr - baseline) / (1.0 - baseline)


def _gains(tp, fp, n_pos, n_neg):
    """Precision gain and recall gain at contingency counts (arrays or scalars),
    with prg.py's conventions: where TN + FN == 0 (everything predicted
    positive) precision gain is 0 and recall gain is 1."""
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = n_pos - tp
    tn = n_neg - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        pg = 1.0 - (n_pos / n_neg) * (fp / tp)
        rg = 1.0 - (n_pos / n_neg) * (fn / tp)
    full = (tn + fn) == 0
    pg = np.where(full, 0.0, pg)
    rg = np.where(full, 1.0, rg)
    return pg, rg


def prg_curve(labels, scores):
    """(recall_gain, precision_gain) points of the Precision-Recall-Gain curve.

    A faithful re-implementation of the reference create_prg_curve (Flach &
    Kull 2015; github.com/meeliskull/prg, Python_package/prg/prg.py):
    - one point per distinct score threshold, in threshold order, starting
      from the empty prediction (TP = FP = 0), with tied scores forming ONE
      segment; vertical drops (same recall gain, falling precision gain) are
      KEPT, because the next trapezoid starts from the dropped value;
    - a crossing point inserted where the curve crosses recall_gain = 0;
    - crossing points inserted where it crosses precision_gain = 0 at
      recall_gain >= 0.
    Crossing points interpolate the contingency counts linearly between the two
    neighbouring thresholds, exactly as the reference does, and after every
    insertion the points are re-sorted by (recall_gain asc, precision_gain
    desc).
    """
    y = (np.asarray(labels, dtype=float) == 1).astype(float)
    s = np.asarray(scores, dtype=float)
    n = float(y.size)
    n_pos = float(y.sum())
    n_neg = n - n_pos

    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # segment boundaries: last index of each run of equal scores
    ends = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    cum_pos = np.cumsum(y_sorted)[ends]
    cum_neg = (ends + 1) - cum_pos
    return prg_curve_from_counts(cum_pos, cum_neg, n_pos, n_neg)


def prg_curve_from_counts(cum_pos, cum_neg, n_pos, n_neg):
    """prg_curve from contingency counts instead of labels and scores.

    cum_pos / cum_neg: cumulative positives / negatives at the end of each
    distinct-score segment, in descending score order (one entry per segment,
    every segment non-empty, the last equal to n_pos / n_neg). prg_curve
    computes exactly these and calls this; sl_comparison/bootstrap_ci.py calls
    it with weighted (bootstrap-resampled) counts, so a resample is scored by
    the same curve construction without re-sorting.
    """
    n_pos = float(n_pos)
    n_neg = float(n_neg)
    n = n_pos + n_neg
    tp = np.r_[0.0, np.asarray(cum_pos, dtype=float)]
    fp = np.r_[0.0, np.asarray(cum_neg, dtype=float)]
    pg, rg = _gains(tp, fp, n_pos, n_neg)

    def _sort(tp, fp, pg, rg):
        o = np.lexsort((-pg, rg))
        return tp[o], fp[o], pg[o], rg[o]

    # crossing through the y-axis (recall_gain = 0)
    j = int(np.flatnonzero(rg >= 0).min())
    if rg[j] > 0:
        d_tp, d_fp = tp[j] - tp[j - 1], fp[j] - fp[j - 1]
        alpha = ((n_pos * n_pos / n - tp[j - 1]) / d_tp) if d_tp > 0 else 0.5
        ntp, nfp = tp[j - 1] + alpha * d_tp, fp[j - 1] + alpha * d_fp
        npg, _ = _gains(ntp, nfp, n_pos, n_neg)
        tp, fp = np.r_[ntp, tp], np.r_[nfp, fp]
        pg, rg = np.r_[float(npg), pg], np.r_[0.0, rg]
        tp, fp, pg, rg = _sort(tp, fp, pg, rg)

    # crossings through the x-axis (precision_gain = 0) at recall_gain >= 0.
    # The reference finds them once, then shifts every later index by one per
    # inserted point (its in-place `indices += 1`); k + inserted mirrors that.
    with np.errstate(invalid="ignore"):
        prod = np.r_[pg, 0.0] * np.r_[0.0, pg]
        cross = np.flatnonzero((prod < 0) & (np.r_[1.0, rg] >= 0))
    for inserted, k in enumerate(cross):
        i = int(k) + inserted
        x0, x1, y0, y1 = rg[i - 1], rg[i], pg[i - 1], pg[i]
        cross_x = x0 + (-y0) / (y1 - y0) * (x1 - x0)
        d_tp, d_fp = tp[i] - tp[i - 1], fp[i] - fp[i - 1]
        if d_tp > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - tp[i - 1]) / d_tp
        else:
            alpha = (n_neg / n_pos * tp[i - 1] - fp[i - 1]) / d_fp
        ntp, nfp = tp[i - 1] + alpha * d_tp, fp[i - 1] + alpha * d_fp
        _, nrg = _gains(ntp, nfp, n_pos, n_neg)
        tp, fp = np.r_[ntp, tp], np.r_[nfp, fp]
        pg, rg = np.r_[0.0, pg], np.r_[float(nrg), rg]
        tp, fp, pg, rg = _sort(tp, fp, pg, rg)
    return rg, pg


def auprg(labels, scores):
    """Area Under the Precision-Recall-Gain curve (Flach & Kull, NeurIPS 2015).

    Identical to the reference prg.calc_auprg(prg.create_prg_curve(...)):
    trapezoids between consecutive curve points (prg_curve), counted from
    every point whose recall gain is >= 0. Verified against the reference on
    thousands of random cases with and without ties (siamese_sl/tests/
    test_auprg.py pins reference-computed cases). The reference package is not
    imported: its current release calls np.alen, which numpy >= 1.23 removed.

    Returns NaN when prevalence is degenerate (no positives or no negatives).

    History: the integrator used until 2026-09-21 kept only the MAXIMUM
    precision gain at each recall gain, dropping vertical drops that change
    the next segment's starting height; labels [1, 0, 1] in descending score
    order gave +0.25 instead of the reference's -0.25.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pi = float(np.mean(labels)) if labels.size else float("nan")
    if not (0.0 < pi < 1.0) or len(np.unique(labels)) < 2:
        return float("nan")
    return _area(*prg_curve(labels, scores))


def auprg_from_counts(cum_pos, cum_neg):
    """auprg() from per-segment cumulative counts (see prg_curve_from_counts).

    Segments with no weight must be dropped by the caller: an empty segment
    would add a duplicate curve point that prg_curve never produces. NaN when
    either class has no weight.
    """
    cum_pos = np.asarray(cum_pos, dtype=float)
    cum_neg = np.asarray(cum_neg, dtype=float)
    if cum_pos.size == 0:
        return float("nan")
    n_pos, n_neg = float(cum_pos[-1]), float(cum_neg[-1])
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    return _area(*prg_curve_from_counts(cum_pos, cum_neg, n_pos, n_neg))


def _area(rg, pg):
    """Trapezoids between consecutive PRG points with recall gain >= 0."""
    ok = ~np.isnan(rg[:-1]) & (rg[:-1] >= 0)
    # Recall gain is -inf wherever TP = 0: when the top-scored segment holds
    # only negatives, np.diff meets -inf - -inf = NaN (a RuntimeWarning per
    # call); ok masks those segments out.
    with np.errstate(invalid="ignore"):
        area = (np.diff(rg) * (pg[1:] + pg[:-1]) / 2.0)[ok]
    return float(area.sum())
