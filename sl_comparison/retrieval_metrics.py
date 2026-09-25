#!/usr/bin/env python3
"""
Correctly-normalised NDCG@k / MAP@k / precision@k, computed alongside the
vendored benchmark versions rather than replacing them.

WHY THIS EXISTS
---------------
preprocess_benchmarking_paper.cal_metrics is kept byte-faithful to the
SL_benchmark reference implementation so our numbers stay comparable with
published SLMGAE / SLGNN / NSF4SL tables. That reference has three
normalisation defects, all of which inflate scores:

1. NDCG@k -- cal_metrics:1439-1441 calls
       sklearn.metrics.ndcg_score(y_bool_list, y_sorted_score_list, k=k)
   where `y_bool_list` carries relevance ONLY over each model's own top-100.
   sklearn therefore derives the ideal ranking from that truncated slice, so
   IDCG = min(k, hits-in-top-100) instead of min(k, n_relevant). A model that
   surfaces 1 of a gene's 30 true partners at rank 1 scores NDCG@10 = 1.0.

2. AP@k -- calculate_AP_at_k:1354-1361 returns
       precision_sum / relevant_docs
   dividing by hits-in-top-k rather than min(k, n_relevant). This is the larger
   of the two defects, and it is this repo's own code, not sklearn's.

3. precision@k -- cal_metrics:1443-1448 divides by min(n_relevant, k) rather
   than k, which makes it a second copy of recall@k whenever n_relevant <= k.
   On the 2026-08-06 run it agreed with recall@k to three decimals.

recall@k (cal_metrics:1449-1454) divides by the true n_relevant and is correct
as published; it is recomputed here only as a cross-check that this module
reproduces cal_metrics' ranking construction exactly.

The corrections did not reorder the models on any cell measured to date, and in
general they shrink scores -- but NOT unconditionally, and NDCG in particular is
not a pure renormalisation of the vendored value. cal_metrics obtains NDCG from
sklearn.metrics.ndcg_score, which AVERAGES gains across tied scores; `_dcg`
below is purely positional and does not. On a model with many tied scores the
corrected NDCG can therefore EXCEED the vendored one, because the two differ in
tie handling as well as in the normaliser. Read *_corrected as "a defensible
metric on the same ranking", not as "the vendored metric with a fixed
denominator".

The ranking CONSTRUCTION below (diagonal zeroing, -999999 seen sentinel,
test_gene_set, top-100 truncation, descending argsort tie-break) does mirror
cal_metrics exactly -- it is the DCG computed from that ranking that differs.
Any change to the construction must be mirrored here or the two columns stop
describing the same ranking.
"""

import numpy as np
import scipy.sparse as sp

SEEN_SCORE = -999999
TOP_N = 100  # cal_metrics truncates every gene's ranking at 100
KS = (10, 20, 50)


def _dcg(rel):
    """Discounted cumulative gain of a binary relevance vector, rank 1..len."""
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))


def _ideal_dcg(n_rel, k):
    """IDCG@k for n_rel relevant items: the best achievable DCG at cutoff k."""
    m = int(min(k, n_rel))
    if m <= 0:
        return 0.0
    return float(np.sum(1.0 / np.log2(np.arange(2, m + 2))))


def _ap_at_k(rel, k, n_rel):
    """Average precision at k, normalised by min(k, n_rel) as AP is defined."""
    denom = min(k, n_rel)
    if denom <= 0:
        return 0.0
    hits = 0
    acc = 0.0
    for i in range(min(k, len(rel))):
        if rel[i]:
            hits += 1
            acc += hits / (i + 1)
    return acc / denom


def ranking_lists(score_mat, pos_index, seen_index=None, top_n=TOP_N):
    """Rebuild cal_metrics' per-gene top-N relevance lists.

    Returns (y_bool_list, y_pos_num_list): for each test gene with at least one
    positive, the binary relevance of its top-`top_n` candidates and its true
    number of relevant partners.
    """
    # Keep the INPUT dtype for the ranking. cal_metrics argsorts a deepcopy of
    # score_mat (float32 for every model here); upcasting to float64 first would
    # hand numpy a different dtype-specialised argsort kernel, and the tie order
    # among equal keys is what decides the top-100 for SLMGAE's constant rows.
    # Measured: upcasting changed 0 of 21,200 relevance cells on the worst case
    # (SLMGAE cv3 fold_0, 31.6% constant rows) — so this is belt-and-braces, but
    # it makes the identity hold by construction instead of by measurement.
    score = np.array(score_mat, copy=True)
    n_gene = score.shape[0]
    score[range(n_gene), range(n_gene)] = 0

    pos_index = np.asarray(pos_index)
    pos_matrix = sp.csr_matrix(
        (np.ones(pos_index.shape[0]), (pos_index[:, 0], pos_index[:, 1])),
        shape=score.shape)
    pos_matrix = pos_matrix + pos_matrix.T

    if seen_index is not None and len(seen_index):
        seen_index = np.asarray(seen_index)
        score[seen_index[:, 0], seen_index[:, 1]] = SEEN_SCORE
        score[seen_index[:, 1], seen_index[:, 0]] = SEEN_SCORE

    y_bool_list, y_pos_num_list = [], []
    test_gene_set = list(set(pos_index[:, 0]) | set(pos_index[:, 1]))
    for i in test_gene_set:
        y_pos_index = pos_matrix[i, :].nonzero()[1]
        if len(y_pos_index) == 0:
            continue
        y_pos_num_list.append(len(y_pos_index))
        top = np.argsort(score[i, :])[::-1][:top_n]
        y_bool_list.append(pos_matrix[i, :].toarray()[0][top])
    return np.asarray(y_bool_list), np.asarray(y_pos_num_list)


def corrected_metrics(score_mat, pos_index, seen_index=None, ks=KS):
    """{'ndcg@10_corrected': ..., 'map@10_corrected': ..., 'precision@10_true':
    ..., 'recall@10_check': ...} with textbook normalisers.

    Suffixes are deliberate: these names never collide with the vendored keys,
    so a results.json carries both and no downstream consumer silently swaps
    one for the other.
    """
    y_bool, n_rel = ranking_lists(score_mat, pos_index, seen_index)
    out = {}
    if not len(y_bool):
        return {
            f"{m}@{k}{sfx}": float("nan")
            for k in ks
            for m, sfx in (("ndcg", "_corrected"), ("map", "_corrected"),
                           ("precision", "_true"), ("recall", "_check"))
        }
    for k in ks:
        ndcgs, aps, precs, recs = [], [], [], []
        for rel, nr in zip(y_bool, n_rel):
            rel_k = rel[:k]
            idcg = _ideal_dcg(nr, k)
            ndcgs.append(_dcg(rel_k) / idcg if idcg > 0 else 0.0)
            aps.append(_ap_at_k(rel, k, int(nr)))
            precs.append(float(rel_k.sum()) / k)
            recs.append(float(rel_k.sum()) / float(nr))
        out[f"ndcg@{k}_corrected"] = float(np.mean(ndcgs))
        out[f"map@{k}_corrected"] = float(np.mean(aps))
        out[f"precision@{k}_true"] = float(np.mean(precs))
        out[f"recall@{k}_check"] = float(np.mean(recs))
    return out
