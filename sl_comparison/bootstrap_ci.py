#!/usr/bin/env python3
"""
Bootstrap confidence intervals for every reported test metric, by resampling
the SAVED test predictions. Nothing is retrained and no model is re-run.

Two parts (--part):

tiers     (the cv1 / cv2 / cv3 benchmark table, compare.py / tab:tiers)
---------------------------------------------------------------------------
Each row of the PCA arm's table (compare.load_split_results: the arm's SLeuth
rows from its <cmp> directory, SLGNN / NSF4SL / SLMGAE / DegreePrior from
sl_comparison/results) is re-read from the SAME score matrices the shared
grader read, and turned into the SAME (score, label) vectors
evaluate_model.score_fold builds:

    score   = the fold's dense matrix M (fold_<k>_<cv>_predictions.npy), after
              score_fold's affine shift M - min(M) + 1 (float32);
    pairs   = <folds>/<cv>/fold_<k>/test_pos.npy then test_neg.npy, in file
              order; label 1 / 0. (The seen-mask only touches the @k ranking
              metrics, which are not bootstrapped here; it never enters the
              AUROC / AUPR / AP / AUPRG / F1 vectors.)

A cell-line SLeuth row needs nothing special: its matrix is already the
max-over-heads logit (siamese_sl retrieval_metrics.build_any_score_matrix),
graded against the shared folds' OR-collapsed ("any cell line") labels exactly
like every other row. DegreePrior's matrix is rebuilt in memory with
degree_baseline.degree_matrix, as degree_baseline.py does.

Prediction matrices are located the way the pipeline wrote them:
    SLGNN / NSF4SL / SLMGAE   SLGNN/results/slgnn_<cv>, NSF4SL/results/
                              nsf4sl_<cv>, SLMGAE-in-pytorch/results/
                              slmgae_shared_<cv> (run_shared_models.sh)
    BilinearProbe             siamese_sl/results/bilinear_probe_<cv>
                              (siamese_sl/slurm/run_probe.sh)
    ELISL                     ELISL/results/elisl_<cv> (ELISL/slurm/run_elisl.sh)
    siamese_sl[_<tag>]        the directory named by <cmp>/<stem>_selected_
                              <cv>.txt (collect_and_regrade_siamese.sh; stem
                              'siamese' for siamese_sl, else the row name;
                              siamese_sl_allmod[_<tag>] too, ALLMOD_COMBO=1)
    siamese_sl[_<tag>]_nocell siamese_sl/<arm results root>/[var_<tag>__]
                              cellline_ablation_<cv> (run_cell_line_ablation.sh)
    anything else / override  --pred_dir MODEL=DIR (DIR may contain {cv})
and every directory's fold_stamp is checked (model_params.verify_fold_stamp),
as evaluate_model.py does.

Per (cv, fold) one set of B resamples of the test pairs is drawn (with
replacement; --stratify resamples positives and negatives separately) and
applied to EVERY model, so model differences are paired. Resample b of
(cv, fold) uses np.random.default_rng([seed, cv, fold, b]) -- identical for
every model, every arm and every run with the same --seed.

Metrics per resample: auroc, aupr (cal_metrics' trapezoidal
auc(recall, precision), the table's 'aupr' column), ap (average precision),
auprg (sl_comparison/auprg.py, the single shared definition), f1 (cal_metrics'
max F1 over the PR curve), ap_norm, aupr_norm. They are computed from one
descending sort per (model, fold) plus per-resample integer weights, i.e.
weighted cumulative TP / FP per distinct-score segment; a resample never
re-sorts.

The tier pooled CI is the CI of the 5-fold MEAN: resample b's value averaged
over folds (independent resamples per fold, since the folds' test sets are
disjoint), percentile interval over b.

Paired differences: each SLeuth row minus each non-SLeuth row, each selected
combo siamese_sl[_<tag>] minus its all-modalities row siamese_sl_allmod[_<tag>]
when both are in the table (plus any --pair A:B), per fold and pooled, with p_boot = fraction of resamples in which
the difference is <= 0 (one-sided; small = A reliably above B).

Before anything is written, the point estimate (unresampled data) of every
(model, cv, fold, metric) is checked against (1) sklearn / auprg.py run on the
same vectors (the cal_metrics recipe) and (2) the published per-fold value in
the row's JSON, and every pooled point against its <metric>_mean. A mismatch
beyond --check_atol stops the run: the CI would not be a CI of the published
number.

external  (external_transfer.py's zero-shot and fine-tuning Spearman tables)
---------------------------------------------------------------------------
Reads the arm's <cmp>/external_transfer/summary.md provenance (the SLeuth
model dirs, baseline dirs, model CV the published run used; --ext_siamese /
--ext_baseline override) and rebuilds, through external_transfer.py's own
helpers, every per-fold score vector on the same rows:

    row sets   matched (all models), all_sldb_filtered (SLeuth only),
               matched_incl_sldb / matched_in_sldb (all models, zero-shot raw
               only; on matched_in_sldb also the SynLethDB_label predictor)
    methods    zero_shot_raw (SLeuth logit_fold<k>; baselines
               (M[i,j]+M[j,i])/2; DegreePrior), zero_shot_ols (SLeuth
               a_k*logit+c_k, OLS on the screen's train rows), LP_last / Full
               [/ Scratch, when run] (SLeuth pred_fold<k> of the fine-tuned
               CSVs)

(all_unfiltered, a summary.md-only diagnostic whose rows may carry
non-finite values, is not bootstrapped.)

Here the five fold models are scored on the SAME screen rows, so ONE resample
of the rows is applied to every fold and every model of a row set
(default_rng([seed, crc32(screen|split|row_set|row-key digest), b])); the
pooled CI is the CI of the fold-mean Spearman over those shared resamples.
Spearman on a resample is computed exactly (average ranks with ties, as
scipy.stats.spearmanr on the expanded resample) from per-row weights.
Paired differences: every SLeuth variant's zero_shot_raw minus every baseline
/ DegreePrior on the matched rows, and per SLeuth variant LP_last / Full minus
zero_shot_ols and (when eval_finetuning ran Scratch) Full minus Scratch on each
row set; plus --pair A:B (zero_shot_raw). Every point
estimate is checked against zero_shot_per_fold.csv / finetune_per_fold.csv.

Inputs that are missing (a matrix that lives only on the cluster, a missing
marker) stop the run with the full list, before any computation; --skip_missing
drops those rows instead and records them in manifest.json.

Outputs (--out_dir; default <cmp>/bootstrap/, <cmp> = sl_comparison/results
for arm none, sl_comparison/results/pca_<arm> otherwise)
    ci_tiers.csv            model, variant, pca_arm, cv, fold (0..k-1 or
                            'mean'), metric, point, published, boot_mean,
                            boot_se, ci_low, ci_high, ci_level, n_boot,
                            n_finite, n_test, n_pos, n_folds, resample, seed
    ci_tiers_diffs.csv      model, variant, peer, peer_variant, ... point
                            (A - B), ci_low, ci_high, p_boot_le0, ...
    ci_external.csv         model, kind, variant, method, pca_arm, screen,
                            split, row_set, fold, metric (spearman), ...
    ci_external_diffs.csv   model, method, peer, peer_method, ...
    ci_sldb_agreement.csv   screen, scope, split, metric (auroc_target_vs_label,
                            spearman_target_vs_label), point, published,
                            ci_low, ci_high, ... (external_transfer's
                            sldb_agreement.csv: target vs the SynLethDB label
                            on the pairs SynLethDB labels; pairs resampled)
    manifest.json           inputs, seeds, versions, skipped rows
    replicates_*.npz        (--save_replicates) every bootstrap replicate

Usage (from the repo root; CPU only, no GPU needed):
  python sl_comparison/bootstrap_ci.py                       # arm none, both parts
  python sl_comparison/bootstrap_ci.py --pca_arm joint --part tiers
  python sl_comparison/bootstrap_ci.py --part external --n_boot 2000 --seed 7

CAVEAT (stated in manifest.json too): pairs are resampled as independent
units. Test pairs share genes, so these intervals reflect test-pair sampling
only, not gene-level or training (fold-model) variability; they are narrower
than a gene-clustered bootstrap would give.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import auprg as A  # noqa: E402
import compare  # noqa: E402
import degree_baseline  # noqa: E402
import evaluate_model as em  # noqa: E402
import external_transfer as et  # noqa: E402

ARMS = ("none", "joint", "per_modality", "both")
CVS = ("cv1", "cv2", "cv3")
DEGREE_PRIOR = "DegreePrior"
# run_shared_models.sh PRED_DIR, relative to the repo root.
SHARED_PRED_DIRS = {
    "SLGNN": "SLGNN/results/slgnn_{cv}",
    "NSF4SL": "NSF4SL/results/nsf4sl_{cv}",
    "SLMGAE": "SLMGAE-in-pytorch/results/slmgae_shared_{cv}",
}
# Trained baselines outside run_shared_models.sh, whose rows also live in
# sl_comparison/results/ (compare.load_split_results reads them there):
# siamese_sl/slurm/run_probe.sh trains the bilinear probe into arm none's
# siamese results root, the directory its evaluate_model.py --pred_dir grades.
BASELINE_PRED_DIRS = {
    "BilinearProbe": "siamese_sl/results/bilinear_probe_{cv}",
    # ELISL/slurm/run_elisl.sh: train_elisl.py --output_dir, the directory its
    # evaluate_model.py --pred_dir grades (a published peer trained by its own
    # launcher, not by run_shared_models.sh).
    "ELISL": "ELISL/results/elisl_{cv}",
}
CLS_METRICS = ("auroc", "aupr", "ap", "auprg", "f1", "ap_norm", "aupr_norm")
EXT_ROW_SETS = (et.ROW_MATCHED, et.ROW_ALL_SLDB, et.ROW_MATCHED_INCL,
                et.ROW_MATCHED_IN_SLDB)
# Row sets with zero-shot rows only (external_transfer's fine-tuning table
# keeps ROW_SETS): SLeuth zero_shot_raw, the matrix models and, on
# matched_in_sldb, the SynLethDB label.
EXT_ZS_ONLY_ROW_SETS = (et.ROW_MATCHED_INCL, et.ROW_MATCHED_IN_SLDB)
FT_METHODS = ("LP_last", "Full", "Scratch")
# The modes warm-started from the pretrained logit's OLS head (differenced
# against zero_shot_ols). Scratch (eval_finetuning --ft_modes ... Scratch,
# opt-in) starts from a fresh init: it is bootstrapped when its CSV is there
# (absent: skipped, unless finetune_per_fold.csv publishes it) and
# differenced as Full - Scratch (what pretraining adds).
FT_OLS_WARM = ("LP_last", "Full")
FT_SCRATCH = "Scratch"


class MissingInput(Exception):
    """An input the bootstrap needs is not on this machine."""


# =============================================================================
# Small helpers
# =============================================================================


def repo_path(p) -> Path:
    """A path as the pipeline writes it (relative to the repo root)."""
    p = Path(str(p).strip())
    return p if p.is_absolute() else REPO_ROOT / p


def arm_cmp_dir(arm: str) -> Path:
    """sl_comparison/results for arm none, .../pca_<arm> otherwise
    (sl_comparison/slurm/pca_arm.sh SLEUTH_CMP)."""
    root = REPO_ROOT / "sl_comparison" / "results"
    return root if arm == "none" else root / f"pca_{arm}"


def arm_siamese_root(arm: str) -> Path:
    """siamese_sl/results for arm none, .../pca_<arm> otherwise."""
    root = REPO_ROOT / "siamese_sl" / "results"
    return root if arm == "none" else root / f"pca_{arm}"


def variant_of(model: str) -> str:
    """kernel_cell-lines tag of a SLeuth row ('' for a peer row)."""
    if not compare.is_siamese_row(model):
        return ""
    nocell = model.endswith("_nocell")
    base = model[:-len("_nocell")] if nocell else model
    if compare.is_allmod_row(base):
        rest = base[len("siamese_sl_allmod"):]
        return "allmod_" + (rest[1:] if rest else "pd_per_head") + \
            ("_nocell" if nocell else "")
    tag = "pd_per_head" if base == "siamese_sl" else base[len("siamese_sl_"):]
    return tag + ("_nocell" if nocell else "")


def percentile_ci(reps: np.ndarray, alpha: float) -> Tuple[float, float, int]:
    """(low, high, n_finite) percentile interval over finite replicates."""
    r = np.asarray(reps, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), int(r.size)


def boot_stats(point: float, reps: np.ndarray, alpha: float) -> dict:
    lo, hi, n_ok = percentile_ci(reps, alpha)
    r = np.asarray(reps, dtype=np.float64)
    r = r[np.isfinite(r)]
    return {
        "point": float(point),
        "boot_mean": float(r.mean()) if r.size else float("nan"),
        "boot_se": float(r.std(ddof=1)) if r.size > 1 else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "n_finite": n_ok,
    }


def diff_stats(point: float, reps: np.ndarray, alpha: float) -> dict:
    d = boot_stats(point, reps, alpha)
    r = np.asarray(reps, dtype=np.float64)
    r = r[np.isfinite(r)]
    d["p_boot_le0"] = float(np.mean(r <= 0)) if r.size else float("nan")
    return d


def write_csv_atomic(rows: List[dict], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


# =============================================================================
# Resampling
# =============================================================================


def resample_counts(n: int,
                    n_boot: int,
                    seed: int,
                    unit_key: Sequence[int],
                    strata: Optional[Sequence[Tuple[int, int]]] = None
                    ) -> np.ndarray:
    """(n_boot, n) uint16 multiplicities of n items under B bootstrap draws.

    Resample b uses default_rng([seed, *unit_key, b]) alone, so it does not
    depend on n_boot, chunking or which models are scored. strata: list of
    [lo, hi) index ranges resampled separately (sizes preserved).
    """
    out = np.empty((n_boot, n), dtype=np.uint16)
    for b in range(n_boot):
        rng = np.random.default_rng([int(seed), *map(int, unit_key), b])
        if strata is None:
            idx = rng.integers(0, n, size=n)
        else:
            idx = np.concatenate([rng.integers(lo, hi, size=hi - lo)
                                  for lo, hi in strata if hi > lo])
        c = np.bincount(idx, minlength=n)
        if c.max() >= np.iinfo(np.uint16).max:
            raise OverflowError("bootstrap multiplicity overflows uint16")
        out[b] = c
    return out


def unit_hash(*parts) -> int:
    """Stable non-negative int for a string key (seeds the external units)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# =============================================================================
# Classification metrics from (weighted) segment counts
# =============================================================================


class DescSegments:
    """One descending sort of a score vector: order and tie-segment starts."""

    def __init__(self, scores: np.ndarray):
        s = np.asarray(scores)
        # Same tie structure as sklearn's _binary_clf_curve / prg_curve: one
        # segment per distinct value (exact equality on the stored dtype).
        self.order = np.argsort(-s.astype(np.float64), kind="stable")
        ss = s[self.order]
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])


def segment_increments(W: np.ndarray, y: np.ndarray,
                       seg: DescSegments) -> Tuple[np.ndarray, np.ndarray]:
    """Per-resample positives / negatives in each segment: two (C, n_seg)."""
    Ws = W[:, seg.order].astype(np.int64)
    ys = y[seg.order].astype(np.int64)
    tot = np.add.reduceat(Ws, seg.starts, axis=1)
    pos = np.add.reduceat(Ws * ys, seg.starts, axis=1)
    return pos, tot - pos


def metrics_from_counts(inc_pos: np.ndarray, inc_neg: np.ndarray) -> dict:
    """Every classification metric of one (re)sample from its segment counts.

    Mirrors cal_metrics (roc_auc_score; auc(recall, precision) on
    precision_recall_curve; max F1 with its +1e-10), sklearn's
    average_precision_score and auprg.auprg, all of which see one curve point
    per distinct score. Empty segments (zero weight in this resample) are
    dropped first: they are thresholds the resample does not have.
    """
    keep = (inc_pos + inc_neg) > 0
    ip = inc_pos[keep].astype(np.float64)
    ineg = inc_neg[keep].astype(np.float64)
    cp = np.cumsum(ip)
    cn = np.cumsum(ineg)
    nan = {k: float("nan") for k in CLS_METRICS}
    if cp.size == 0 or cp[-1] <= 0 or cn[-1] <= 0:
        return nan
    P, N = cp[-1], cn[-1]
    tp0 = cp - ip
    # AUROC: trapezoids of the ROC staircase (ties -> diagonal), as
    # roc_auc_score; equals the Mann-Whitney U with half credit for ties.
    auroc = float(np.sum(ineg * (cp + tp0)) / (2.0 * P * N))
    prec = cp / (cp + cn)
    rec = cp / P
    ap = float(np.sum(ip / P * prec))
    # cal_metrics: auc(recall, precision) over precision_recall_curve's points
    # (descending recall, then the appended (recall 0, precision 1)).
    rec0 = np.r_[0.0, rec[:-1]]
    prec0 = np.r_[1.0, prec[:-1]]
    aupr = float(np.sum((rec - rec0) * (prec + prec0) / 2.0))
    f1 = float(max(0.0, np.max(2 * prec * rec / (prec + rec + 1e-10))))
    pi = P / (P + N)
    return {
        "auroc": auroc,
        "aupr": aupr,
        "ap": ap,
        "auprg": A.auprg_from_counts(cp, cn),
        "f1": f1,
        "ap_norm": A.normalized_aupr(ap, pi),
        "aupr_norm": A.normalized_aupr(aupr, pi),
    }


def sklearn_reference(y: np.ndarray, s: np.ndarray) -> dict:
    """The same metrics by the grader's own calls (cal_metrics recipe,
    sklearn average_precision_score, auprg.auprg) for the point check."""
    from sklearn.metrics import (auc, average_precision_score,
                                 precision_recall_curve, roc_auc_score)
    precision, recall, _ = precision_recall_curve(y, s)
    aupr = auc(recall, precision)
    ap = float(average_precision_score(y, s))
    pi = float(np.mean(y))
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(aupr),
        "ap": ap,
        "auprg": A.auprg(y, s),
        "f1": float(max(2 * precision * recall /
                        (precision + recall + 1e-10))),
        "ap_norm": A.normalized_aupr(ap, pi),
        "aupr_norm": A.normalized_aupr(float(aupr), pi),
    }


def bootstrap_classification(y: np.ndarray,
                             scores: Dict[str, np.ndarray],
                             W: np.ndarray,
                             chunk: int = 64
                             ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Point metrics and (B,) replicates per model on one unit.

    Returns (points[model][metric], reps[model][metric] -> (B,) array).
    """
    B = W.shape[0]
    segs = {m: DescSegments(s) for m, s in scores.items()}
    ones = np.ones((1, len(y)), dtype=np.uint16)
    points, reps = {}, {}
    for m, seg in segs.items():
        ip, ineg = segment_increments(ones, y, seg)
        points[m] = metrics_from_counts(ip[0], ineg[0])
        reps[m] = {k: np.full(B, np.nan) for k in CLS_METRICS}
    for c0 in range(0, B, chunk):
        Wc = W[c0:c0 + chunk]
        for m, seg in segs.items():
            ip, ineg = segment_increments(Wc, y, seg)
            for r in range(Wc.shape[0]):
                vals = metrics_from_counts(ip[r], ineg[r])
                for k in CLS_METRICS:
                    reps[m][k][c0 + r] = vals[k]
    return points, reps


# =============================================================================
# Spearman from weights
# =============================================================================


class RankPrep:
    """Ascending sort + tie groups of one vector, for weighted average ranks."""

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.order = np.argsort(x, kind="stable")
        xs = x[self.order]
        new = np.r_[True, xs[1:] != xs[:-1]]
        self.starts = np.flatnonzero(new)
        gid_sorted = np.cumsum(new) - 1
        self.gid = np.empty(len(x), dtype=np.int64)
        self.gid[self.order] = gid_sorted


def weighted_ranks(Wf: np.ndarray, prep: RankPrep) -> Tuple[np.ndarray, np.ndarray]:
    """(ranks (C, n), n_distinct_with_weight (C,)).

    An item of a tie group of total weight G that starts after `before`
    weighted items gets rank before + (G + 1) / 2: scipy's 'average' rank of
    the expanded resample.
    """
    G = np.add.reduceat(Wf[:, prep.order], prep.starts, axis=1)
    before = np.cumsum(G, axis=1) - G
    rg = before + (G + 1.0) / 2.0
    return rg[:, prep.gid], (G > 0).sum(axis=1)


class TargetChunk:
    """Target-side terms of one block of resamples, shared by every score."""

    def __init__(self, Wf: np.ndarray, prep_t: RankPrep):
        rt, self.nt = weighted_ranks(Wf, prep_t)
        self.W = Wf
        self.N = Wf.sum(axis=1)
        mt = (Wf * rt).sum(axis=1) / self.N
        self.WDt = Wf * (rt - mt[:, None])
        self.vt = (self.WDt * (rt - mt[:, None])).sum(axis=1)


def weighted_spearman(tc: TargetChunk, prep_s: RankPrep) -> np.ndarray:
    """Spearman of score vs target per resample row; NaN where either side
    is constant in the resample or it holds < 3 items (et.spearman's rule).
    Weighted Pearson of the average ranks; sum(W * (rt - mt)) = 0, so the
    covariance needs no centring of the score ranks."""
    rs, ns = weighted_ranks(tc.W, prep_s)
    Wrs = tc.W * rs
    ms = Wrs.sum(axis=1) / tc.N
    cov = (rs * tc.WDt).sum(axis=1)
    vs = (Wrs * rs).sum(axis=1) - tc.N * ms * ms
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov / np.sqrt(vs * tc.vt)
    bad = (ns < 2) | (tc.nt < 2) | (tc.N < 3)
    rho[bad] = np.nan
    return np.clip(rho, -1.0, 1.0)


def bootstrap_spearman(target: np.ndarray,
                       scores: Dict[object, np.ndarray],
                       W: np.ndarray,
                       chunk: int = 64
                       ) -> Tuple[Dict[object, float], Dict[object, np.ndarray]]:
    """Point Spearman and (B,) replicates per score vector on one row set."""
    B = W.shape[0]
    prep_t = RankPrep(target)
    preps = {k: RankPrep(s) for k, s in scores.items()}
    tc1 = TargetChunk(np.ones((1, len(target))), prep_t)
    points = {k: float(weighted_spearman(tc1, p)[0])
              for k, p in preps.items()}
    reps = {k: np.full(B, np.nan) for k in scores}
    for c0 in range(0, B, chunk):
        tc = TargetChunk(W[c0:c0 + chunk].astype(np.float64), prep_t)
        for k, p in preps.items():
            reps[k][c0:c0 + tc.W.shape[0]] = weighted_spearman(tc, p)
    return points, reps


# =============================================================================
# Tier part: inputs
# =============================================================================


def graded_vectors(score: np.ndarray, fold_dir: Path,
                   label: str) -> Tuple[np.ndarray, np.ndarray]:
    """(labels, scores) exactly as evaluate_model.score_fold feeds cal_metrics
    and average_precision_score / auprg: test_pos then test_neg, scores from
    the affinely shifted matrix (float32)."""
    d = Path(fold_dir)
    test_pos = np.load(d / "test_pos.npy").astype(np.int64)
    test_neg = np.load(d / "test_neg.npy").astype(np.int64)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise SystemExit(f"{label}: score matrix not square ({score.shape})")
    genes_txt = d.parent.parent / "genes.txt"
    n_genes = sum(1 for ln in open(genes_txt) if ln.strip())
    if score.shape[0] != n_genes:
        raise SystemExit(f"{label}: matrix is {score.shape[0]}^2, {genes_txt} "
                         f"defines {n_genes} genes")
    # cal_metrics zeroes the diagonal before reading the edge scores, while
    # score_fold's ap / auprg read the matrix without that; the two agree only
    # off the diagonal. No fold set has a self-pair; refuse rather than guess.
    for name, e in (("test_pos", test_pos), ("test_neg", test_neg)):
        if (e[:, 0] == e[:, 1]).any():
            raise SystemExit(f"{label}: {name} holds a self-pair")
    if len(test_pos) == 0 or len(test_neg) == 0:
        raise SystemExit(f"{label}: degenerate fold (score_fold skips it)")
    mn = score.min()
    if not np.isfinite(mn) or not np.isfinite(score.max()):
        raise SystemExit(f"{label}: non-finite entries in score matrix")
    pos = np.asarray(score[test_pos[:, 0], test_pos[:, 1]])
    neg = np.asarray(score[test_neg[:, 0], test_neg[:, 1]])
    # Element for element score_fold's (score - score.min() + 1).astype(f32).
    s = (np.concatenate([pos, neg]) - mn + np.float32(1.0)).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return y, s


def resolve_pred_dir(model: str, cv: str, arm: str, cmp_dir: Path,
                     overrides: Dict[str, str]) -> Optional[Path]:
    """Directory of a row's fold_<k>_<cv>_predictions.npy (None for
    DegreePrior, rebuilt in memory). Raises MissingInput."""
    if model in overrides:
        return repo_path(overrides[model].format(cv=cv))
    if model == DEGREE_PRIOR:
        return None
    if model in SHARED_PRED_DIRS:
        return repo_path(SHARED_PRED_DIRS[model].format(cv=cv))
    if model in BASELINE_PRED_DIRS:
        return repo_path(BASELINE_PRED_DIRS[model].format(cv=cv))
    if compare.is_siamese_row(model):
        if model.endswith("_nocell"):
            base = model[:-len("_nocell")]
            tag = "" if base == "siamese_sl" else base[len("siamese_sl_"):]
            prefix = f"var_{tag}__" if tag else ""
            return arm_siamese_root(arm) / f"{prefix}cellline_ablation_{cv}"
        stem = "siamese" if model == "siamese_sl" else model
        marker = cmp_dir / f"{stem}_selected_{cv}.txt"
        if not marker.exists():
            raise MissingInput(
                f"{model}/{cv}: marker {marker} missing (it names the combo "
                f"whose matrices were graded; collect_and_regrade_siamese.sh "
                f"writes it). Pass --pred_dir {model}=DIR")
        return repo_path(marker.read_text().strip())
    raise MissingInput(f"{model}/{cv}: no known prediction directory for this "
                       f"row; pass --pred_dir {model}=DIR (DIR may use {{cv}})")


def tier_jobs(arm: str, results_dir: Path, shared_dir: Path, folds_dir: Path,
              cvs: Sequence[str], models: Optional[Sequence[str]],
              overrides: Dict[str, str], allow_unstamped: bool
              ) -> Tuple[List[dict], List[str]]:
    """Every (model, cv) to bootstrap, with its inputs verified to exist.

    Returns (jobs, problems); a job is {model, cv, summary, pred_dir,
    fold_dirs}.
    """
    results, _, _ = compare.load_split_results(results_dir, shared_dir)
    jobs, problems = [], []
    # The rows paper_tables refused (2026-11-13): its tab_tiers.csv says
    # which rows it tabulates (status ok) after every usability gate (stale
    # probe, REDUCED / port / substitute baselines, foreign all-modalities
    # rows, ...). Bootstrapping a refused row published intervals and p-values
    # for a model the table does not show. Run paper tables first.
    refused = set()
    tt = results_dir / "paper_tables" / "tab_tiers.csv"
    if tt.exists():
        import csv as _csv
        with open(tt, newline="") as f:
            for r in _csv.DictReader(f):
                if r.get("status") != "ok":
                    refused.add((r.get("cv"), r.get("model")))
    else:
        print(f"WARNING: {tt} missing: the rows paper_tables would refuse "
              f"are not filtered out (run run_paper_tables.sh first)")
    for cv in cvs:
        per_model = results.get(cv, {})
        if not per_model:
            problems.append(f"no rows for {cv} in {results_dir} / {shared_dir}")
            continue
        fold_dirs = em.fold_dirs_for(folds_dir, cv)
        names = sorted(per_model)
        if models:
            unknown = sorted(set(models) - set(names))
            if unknown:
                problems.append(f"{cv}: --models {unknown} not in the table "
                                f"(rows: {names})")
            names = [m for m in names if m in models]
        for model in names:
            if model == "BilinearProbe" and arm != "none":
                # Trained on arm none's features only (paper_tables
                # BASELINE_ARMS): no difference against another arm's SLeuth.
                continue
            if (cv, model) in refused:
                print(f"  skipped {model}/{cv}: not tabulated by paper_tables "
                      f"(status != ok in {tt.name})")
                continue
            summ = per_model[model]
            where = f"{model}/{cv}"
            if summ.get("partial") or \
                    summ.get("n_folds_scored") != len(fold_dirs) or \
                    len(summ.get("folds", [])) != len(fold_dirs):
                problems.append(
                    f"{where}: JSON row is partial ({summ.get('n_folds_scored')}"
                    f" of {len(fold_dirs)} folds); its per-fold values cannot "
                    f"be matched to folds")
                continue
            try:
                pred_dir = resolve_pred_dir(model, cv, arm, results_dir,
                                            overrides)
            except MissingInput as e:
                problems.append(str(e))
                continue
            if pred_dir is not None:
                if not pred_dir.is_dir():
                    problems.append(f"{where}: prediction dir {pred_dir} does "
                                    f"not exist on this machine")
                    continue
                missing = [
                    str(pred_dir / f"fold_{int(d.name.split('_')[1])}_{cv}"
                        f"_predictions.npy") for d in fold_dirs
                    if not (pred_dir / f"fold_{int(d.name.split('_')[1])}_"
                            f"{cv}_predictions.npy").exists()
                ]
                if missing:
                    problems.append(
                        f"{where}: {len(missing)} of {len(fold_dirs)} score "
                        f"matrices missing in {pred_dir} (first: "
                        f"{Path(missing[0]).name})")
                    continue
                try:
                    em.verify_fold_stamp(str(pred_dir), str(folds_dir), cv,
                                         allow_unstamped=allow_unstamped)
                except SystemExit as e:
                    problems.append(f"{where}: {e}")
                    continue
            jobs.append({"model": model, "cv": cv, "summary": summ,
                         "pred_dir": pred_dir, "fold_dirs": fold_dirs})
    return jobs, problems


def load_fold_vectors(job: dict, k: int, fold_dir: Path, num_nodes: int):
    label = f"{job['model']}/{job['cv']}/fold_{k}"
    if job["pred_dir"] is None:
        mat = degree_baseline.degree_matrix(fold_dir, num_nodes)
    else:
        mat = np.load(job["pred_dir"] / f"fold_{k}_{job['cv']}_predictions.npy",
                      mmap_mode="r")
    return graded_vectors(mat, fold_dir, label)


def check_close(got: float, want, atol: float, what: str,
                problems: List[str]) -> None:
    if want is None:
        problems.append(f"{what}: no published value")
        return
    want = float(want)
    if np.isnan(got) and np.isnan(want):
        return
    if not np.isclose(got, want, rtol=0, atol=atol):
        problems.append(f"{what}: bootstrap point {got!r} != published "
                        f"{want!r} (|diff| {abs(got - want):.3g})")


def run_tiers(args, arm: str, out_dir: Path, manifest: dict) -> None:
    results_dir = Path(args.results_dir) if args.results_dir else arm_cmp_dir(arm)
    shared_dir = Path(args.shared_results_dir)
    folds_dir = Path(args.folds_dir)
    num_nodes = sum(1 for ln in open(folds_dir / "genes.txt") if ln.strip())
    tt = results_dir / "paper_tables" / "tab_tiers.csv"
    if not tt.exists() and not getattr(args, "allow_no_tables", False):
        # 2026-11-13: without the arm's paper tables the rows they refuse
        # (stale probe, foreign all-modalities run, REDUCED / port /
        # substitute baselines) cannot be left out; a paper-tables job that
        # crashed leaves none (run_paper_tables.sh clears its folder first).
        raise SystemExit(f"tiers: {tt} missing: run "
                         f"sl_comparison/slurm/run_paper_tables.sh for this "
                         f"arm first (or pass --allow_no_tables)")
    jobs, problems = tier_jobs(arm, results_dir, shared_dir, folds_dir,
                               args.cvs, args.models, args.pred_dir_map,
                               args.allow_unstamped)
    report_missing("tiers", problems, args.skip_missing, manifest)
    if not jobs:
        raise SystemExit("tiers: nothing to bootstrap")
    alpha = args.alpha
    rows, diff_rows = [], []
    reps_store: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    check_problems: List[str] = []
    resample = "stratified" if args.stratify else "pairs"
    for cv in args.cvs:
        cv_jobs = [j for j in jobs if j["cv"] == cv]
        if not cv_jobs:
            continue
        fold_dirs = cv_jobs[0]["fold_dirs"]
        fold_reps: Dict[str, List[Dict[str, np.ndarray]]] = {
            j["model"]: [] for j in cv_jobs}
        fold_points: Dict[str, List[dict]] = {j["model"]: [] for j in cv_jobs}
        for d in fold_dirs:
            k = int(d.name.split("_")[1])
            t0 = time.time()
            y_ref, scores = None, {}
            for j in cv_jobs:
                y, s = load_fold_vectors(j, k, d, num_nodes)
                if y_ref is None:
                    y_ref = y
                elif not np.array_equal(y, y_ref):
                    raise SystemExit(f"{cv}/fold_{k}: label vectors differ "
                                     f"between rows (cannot happen)")
                scores[j["model"]] = s
                if not args.no_sklearn_check:
                    ref = sklearn_reference(y, s)
                else:
                    ref = None
                # deferred: compared after the point pass below
                j.setdefault("_ref", {})[k] = ref
            n_pos = int(y_ref.sum())
            strata = [(0, n_pos), (n_pos, len(y_ref))] if args.stratify else None
            W = resample_counts(len(y_ref), args.n_boot, args.seed,
                                (int(cv[2:]), k), strata)
            points, reps = bootstrap_classification(y_ref, scores, W,
                                                    args.chunk)
            for j in cv_jobs:
                m = j["model"]
                pub = j["summary"]["folds"][k]
                for metric in CLS_METRICS:
                    what = f"{m}/{cv}/fold_{k}/{metric}"
                    if j["_ref"][k] is not None:
                        check_close(points[m][metric], j["_ref"][k][metric],
                                    args.check_atol,
                                    what + " (vs sklearn/auprg on the same "
                                    "vectors)", check_problems)
                    check_close(points[m][metric], pub.get(metric),
                                args.check_atol, what + " (vs row JSON)",
                                check_problems)
                    rows.append({
                        "model": m, "variant": variant_of(m), "pca_arm": arm,
                        "cv": cv, "fold": str(k), "metric": metric,
                        "published": pub.get(metric),
                        **boot_stats(points[m][metric], reps[m][metric],
                                     alpha),
                        "ci_level": 1 - alpha, "n_boot": args.n_boot,
                        "n_test": len(y_ref), "n_pos": n_pos, "n_folds": 1,
                        "resample": resample, "seed": args.seed,
                    })
                fold_reps[m].append(reps[m])
                fold_points[m].append(points[m])
            for (a, b) in tier_pairs([j["model"] for j in cv_jobs],
                                     args.pair):
                for metric in CLS_METRICS:
                    diff_rows.append({
                        "model": a, "variant": variant_of(a), "peer": b,
                        "peer_variant": variant_of(b), "pca_arm": arm,
                        "cv": cv, "fold": str(k), "metric": metric,
                        **diff_stats(points[a][metric] - points[b][metric],
                                     reps[a][metric] - reps[b][metric],
                                     alpha),
                        "ci_level": 1 - alpha, "n_boot": args.n_boot,
                        "n_folds": 1, "resample": resample, "seed": args.seed,
                    })
            print(f"  {cv}/fold_{k}: {len(cv_jobs)} rows, n={len(y_ref):,} "
                  f"({n_pos:,} pos), B={args.n_boot}  [{time.time() - t0:.1f}s]")
        # ---- pooled: CI of the fold mean ----
        n_f = len(fold_dirs)
        pooled = {}
        for j in cv_jobs:
            m = j["model"]
            pooled[m] = {}
            for metric in CLS_METRICS:
                R = np.vstack([fr[metric] for fr in fold_reps[m]])
                pt = float(np.mean([fp[metric] for fp in fold_points[m]]))
                rep = R.mean(axis=0)
                pooled[m][metric] = (pt, rep)
                reps_store[(m, cv)] = reps_store.get((m, cv), {})
                reps_store[(m, cv)][metric] = R
                pub = j["summary"].get(f"{metric}_mean")
                check_close(pt, pub, args.check_atol,
                            f"{m}/{cv}/mean/{metric} (vs row JSON "
                            f"{metric}_mean)", check_problems)
                rows.append({
                    "model": m, "variant": variant_of(m), "pca_arm": arm,
                    "cv": cv, "fold": "mean", "metric": metric,
                    "published": pub, **boot_stats(pt, rep, alpha),
                    "ci_level": 1 - alpha, "n_boot": args.n_boot,
                    "n_test": None, "n_pos": None, "n_folds": n_f,
                    "resample": resample, "seed": args.seed,
                })
        for (a, b) in tier_pairs([j["model"] for j in cv_jobs], args.pair):
            for metric in CLS_METRICS:
                pa, ra = pooled[a][metric]
                pb, rb = pooled[b][metric]
                diff_rows.append({
                    "model": a, "variant": variant_of(a), "peer": b,
                    "peer_variant": variant_of(b), "pca_arm": arm, "cv": cv,
                    "fold": "mean", "metric": metric,
                    **diff_stats(pa - pb, ra - rb, alpha),
                    "ci_level": 1 - alpha, "n_boot": args.n_boot,
                    "n_folds": n_f, "resample": resample, "seed": args.seed,
                })
    if check_problems:
        head = "\n  ".join(check_problems[:40])
        raise SystemExit(
            f"REFUSING TO WRITE: {len(check_problems)} point estimate(s) do "
            f"not reproduce the shared grader's values (atol "
            f"{args.check_atol}):\n  {head}")
    write_csv_atomic(rows, out_dir / "ci_tiers.csv")
    write_csv_atomic(diff_rows, out_dir / "ci_tiers_diffs.csv")
    if args.save_replicates:
        np.savez_compressed(
            out_dir / "replicates_tiers.npz",
            **{f"{m}|{cv}|{metric}": R
               for (m, cv), d in reps_store.items() for metric, R in d.items()})
    manifest["tiers"] = {
        "results_dir": str(results_dir),
        "shared_results_dir": str(shared_dir),
        "folds_dir": str(folds_dir),
        "rows": [{"model": j["model"], "cv": j["cv"],
                  "pred_dir": (str(j["pred_dir"]) if j["pred_dir"] is not None
                               else "in-memory degree_baseline.degree_matrix")}
                 for j in jobs],
        "point_check": ("every fold and mean point equals the row JSON within "
                        f"atol {args.check_atol}" +
                        ("" if args.no_sklearn_check else
                         " and sklearn/auprg on the same vectors")),
    }
    print(f"wrote {out_dir / 'ci_tiers.csv'} ({len(rows)} rows) and "
          f"{out_dir / 'ci_tiers_diffs.csv'} ({len(diff_rows)} rows)")


def allmod_partner(model: str) -> Optional[str]:
    """The selected combo's row an all-modalities row (compare.is_allmod_row:
    siamese_sl_allmod[_<tag>], siamese_sl/slurm/run_allmod_combo.sh) stands
    next to: siamese_sl[_<tag>]; None for any other row."""
    if not compare.is_allmod_row(model):
        return None
    return "siamese_sl" + model[len("siamese_sl_allmod"):]


def tier_pairs(models: Sequence[str], extra: Sequence[Tuple[str, str]]
               ) -> List[Tuple[str, str]]:
    """Each SLeuth row (not an all-modalities one) vs each non-SLeuth row,
    then each selected combo vs
    its all-modalities row (selected minus all modalities: what the modality
    selection buys; ALLMOD_COMBO=1), plus requested pairs present."""
    sia = [m for m in models if compare.is_siamese_row(m)]
    peers = [m for m in models if not compare.is_siamese_row(m)]
    # The all-modalities rows are an ablation of SLeuth, compared with their
    # selected combo only (below): "each SLeuth row vs each peer" on them
    # added p-values no table reports and inflated the comparison count.
    out = [(a, b) for a in sia if not compare.is_allmod_row(a)
           for b in peers]
    for b in sia:
        a = allmod_partner(b)
        if a is not None and a in models:
            out.append((a, b))
    for a, b in extra:
        if a in models and b in models and (a, b) not in out:
            out.append((a, b))
    return out


# =============================================================================
# External part
# =============================================================================


def parse_external_provenance(summary_md: Path) -> dict:
    """SLeuth dirs, baseline dirs, model CV and DegreePrior on/off from the
    Provenance block external_transfer.render_markdown writes."""
    if not summary_md.exists():
        raise MissingInput(f"{summary_md} missing: cannot tell which SLeuth / "
                           f"baseline directories the published run used; "
                           f"pass --ext_siamese/--ext_baseline")
    prov = {}
    for ln in summary_md.read_text().splitlines():
        m = re.match(r"^- \*\*(.+?)\*\*: (.*)$", ln.strip())
        if m:
            prov[m.group(1)] = m.group(2).strip()

    def pairs(text: str) -> List[Tuple[str, str]]:
        text = re.sub(r"\s*\(score = .*\)\s*$", "", text)
        if text.strip() in ("", "none"):
            return []
        out = []
        for part in text.split(";"):
            name, sep, path = part.partition(" = ")
            if not sep:
                raise ValueError(f"{summary_md}: cannot parse {part!r}")
            out.append((name.strip(), path.strip()))
        return out

    try:
        return {
            "siamese": pairs(prov["SLeuth variants"]),
            "baselines": pairs(prov.get("baselines", "none")),
            "model_cv": prov["model CV (baselines / DegreePrior folds)"],
            "degree_prior": not prov.get("DegreePrior", "off").startswith("off"),
            # the SynLethDB positives of the label row / agreement table
            # (2026-11-12; absent before, or "<path> not found: ...")
            "sl_path": (lambda v: v if v and not v.endswith("tables not written")
                         else None)(
                prov.get("SynLethDB positives (overlap / agreement)")),
        }
    except KeyError as e:
        raise MissingInput(f"{summary_md}: provenance line {e} missing")


class ExtEntry:
    """One score source on one row set: (model, method) with per-fold vectors."""

    def __init__(self, model: str, kind: str, method: str):
        self.model, self.kind, self.method = model, kind, method
        self.folds: Dict[int, np.ndarray] = {}


def run_external(args, arm: str, out_dir: Path, manifest: dict) -> None:
    import pandas as pd
    cmp_dir = Path(args.results_dir) if args.results_dir else arm_cmp_dir(arm)
    ext_dir = Path(args.external_dir) if args.external_dir else \
        cmp_dir / "external_transfer"
    folds_dir = Path(args.folds_dir)
    problems: List[str] = []
    try:
        prov = parse_external_provenance(ext_dir / "summary.md")
    except MissingInput as e:
        if not args.ext_siamese:
            report_missing("external", [str(e)], False, manifest)
        prov = {"siamese": [], "baselines": [], "model_cv": args.ext_model_cv,
                "degree_prior": True}
    siamese = ([tuple(s.split("=", 1)) for s in args.ext_siamese]
               if args.ext_siamese else prov["siamese"])
    baselines = ([tuple(s.split("=", 1)) for s in args.ext_baseline]
                 if args.ext_baseline else prov["baselines"])
    model_cv = args.ext_model_cv or prov["model_cv"]
    siamese = [(n, repo_path(p)) for n, p in siamese]
    baselines = [(n, repo_path(p)) for n, p in baselines]

    published = {}
    for name in ("zero_shot_per_fold.csv", "finetune_per_fold.csv"):
        p = ext_dir / name
        if not p.exists():
            problems.append(f"{p} missing: the point estimates could not be "
                            f"checked against the published table")
        else:
            published[name] = pd.read_csv(p)
    zs_pub = published.get("zero_shot_per_fold.csv")
    ft_pub = published.get("finetune_per_fold.csv")
    if zs_pub is not None:
        # The published run's screens and splits, in its order.
        screens = list(dict.fromkeys(zs_pub["screen"].astype(str)))
        splits = list(dict.fromkeys(zs_pub["split"].astype(str)))
    else:
        screens, splits = list(et.SCREENS), list(et.SPLITS)
    if args.screens:
        screens = [s for s in screens if s in args.screens]
    if args.splits:
        splits = [s for s in splits if s in args.splits]
    # A published run from before the matched_incl_sldb / matched_in_sldb row
    # sets (2026-11-12) has none to check against: bootstrap what it has.
    row_sets = [rs for rs in EXT_ROW_SETS if zs_pub is None or
                rs in set(zs_pub["row_set"].astype(str))]
    # The SynLethDB positives the published label row / agreement table read.
    sl_given = prov.get("sl_path") if isinstance(prov, dict) else None
    sl_path, _neg = et.default_sldb_paths(folds_dir, sl_given)
    pos_keys = (et.load_pair_keys(sl_path)
                if sl_path is not None and sl_path.is_file() else None)
    agree_pub = None
    if (ext_dir / "sldb_agreement.csv").exists():
        agree_pub = pd.read_csv(ext_dir / "sldb_agreement.csv")

    # ---- existence of every input, before any work ----
    gene_index = et.load_gene_index(folds_dir / "genes.txt")
    num_nodes = len(gene_index)
    for name, d in siamese:
        for split in splits:
            for screen in screens:
                p = d / f"eval_finetuning_{split}" / f"{screen}_predictions.csv"
                if not p.exists():
                    problems.append(f"SLeuth {name}: {p} missing")
    kept_baselines = []
    for name, d in baselines:
        try:
            et.baseline_fold_files(d, model_cv)
            et.check_fold_provenance(d, folds_dir, model_cv,
                                     args.allow_unstamped)
            kept_baselines.append((name, d))
        except (FileNotFoundError, SystemExit) as e:
            problems.append(f"baseline {name}: {e} (the dense score matrices "
                            f"are written on the cluster)")
    if not siamese:
        problems.append("no SLeuth variant: their CSVs define the screen rows")
    report_missing("external", problems, args.skip_missing, manifest)
    siamese = [(n, d) for n, d in siamese
               if all((d / f"eval_finetuning_{sp}" /
                       f"{sc}_predictions.csv").exists()
                      for sp in splits for sc in screens)]
    if not siamese:
        raise SystemExit("external: no SLeuth variant with every CSV present")

    matrix_models = [(n, "baseline",
                      et.matrix_fold_scorers(d, model_cv, num_nodes))
                     for n, d in kept_baselines]
    if prov.get("degree_prior", True) and args.ext_degree_prior:
        matrix_models.append((et.DEGREE_PRIOR_NAME, "degree_prior",
                              et.degree_fold_scorers(folds_dir, model_cv,
                                                     num_nodes)))

    alpha = args.alpha
    rows, diff_rows, check_problems = [], [], []
    reps_npz = {}
    for screen in screens:
        for split in splits:
            t0 = time.time()
            # unit key -> (keys, target, {entry_id: ExtEntry})
            units: Dict[Tuple[str, str], dict] = {}
            for name, d in siamese:
                path = d / f"eval_finetuning_{split}" / f"{screen}_predictions.csv"
                df = et.read_pair_csv(path)
                logit_cols = et.fold_columns(df, et._LOGIT_COL)
                target_all = df["target"].to_numpy(dtype=np.float64)
                train = (df["split"].astype(str) == "train").to_numpy()
                ols = {k: et.fit_ols_affine(df[c].to_numpy()[train],
                                            target_all[train])
                       for k, c in logit_cols}
                ft = {}
                for mode in FT_METHODS:
                    fp = d / f"eval_finetuning_{split}" / \
                        f"{screen}_finetuned_predictions_{mode}.csv"
                    if ft_pub is not None and not (
                            (ft_pub["model"] == name) &
                            (ft_pub["screen"] == screen) &
                            (ft_pub["split"] == split) &
                            (ft_pub["method"] == mode)).any():
                        continue  # not in the published table
                    if not fp.exists() and mode == FT_SCRATCH and \
                            ft_pub is None:
                        continue  # opt-in, not run (and not published)
                    if not fp.exists():
                        msg = f"SLeuth {name}: {fp} missing"
                        if not args.skip_missing:
                            raise SystemExit(msg)
                        manifest.setdefault("skipped", []).append(msg)
                        continue
                    ft[mode] = et.read_pair_csv(fp).set_index("key")
                for rs in row_sets:
                    mask = et.row_mask(df, gene_index, rs)
                    sub = df.loc[mask]
                    if sub["key"].duplicated().any():
                        raise SystemExit(f"{path}: duplicate pairs in {rs}")
                    sub = sub.sort_values("key").reset_index(drop=True)
                    keys = sub["key"].to_numpy()
                    tgt = sub["target"].to_numpy(dtype=np.float64)
                    et.require_finite(tgt, f"{path}: target", rs)
                    ukey = (rs, et.matched_digest(keys))
                    if ukey not in units:
                        units[ukey] = {"keys": keys, "target": tgt,
                                       "entries": {}, "ref": sub}
                    elif not np.allclose(units[ukey]["target"], tgt,
                                         rtol=1e-6, atol=1e-9):
                        raise SystemExit(f"{screen}/{split}/{rs}: {name} has "
                                         f"the same pairs but other targets")
                    ent = units[ukey]["entries"]
                    e_raw = ExtEntry(name, "siamese", "zero_shot_raw")
                    e_ols = ExtEntry(name, "siamese", "zero_shot_ols")
                    for k, c in logit_cols:
                        raw = sub[c].to_numpy(dtype=np.float64)
                        et.require_finite(raw, f"{path}: {c}", rs)
                        e_raw.folds[k] = raw
                        a, b0 = ols[k]
                        e_ols.folds[k] = a * raw + b0
                    ent[(name, "zero_shot_raw")] = e_raw
                    if rs in EXT_ZS_ONLY_ROW_SETS:
                        continue  # no fine-tuning / OLS rows published there
                    ent[(name, "zero_shot_ols")] = e_ols
                    for mode, fdf in ft.items():
                        e_ft = ExtEntry(name, "siamese", mode)
                        for k, c in et.fold_columns(fdf.reset_index(),
                                                    et._PRED_COL):
                            v = fdf[c].reindex(keys).to_numpy(dtype=np.float64)
                            et.require_finite(v, f"{name} {mode} {c}", rs)
                            e_ft.folds[k] = v
                        ent[(name, mode)] = e_ft
            # matrix models on the matched rows (each SLeuth variant's matched
            # set is identical; external_transfer refuses otherwise)
            matched = [u for (rs, _), u in units.items() if rs == et.ROW_MATCHED]
            if len(matched) != 1:
                raise SystemExit(f"{screen}/{split}: SLeuth variants disagree "
                                 f"on the matched row set ({len(matched)} "
                                 f"distinct sets)")
            mu = matched[0]
            # every matched-gene row set (matched_incl_sldb / matched_in_sldb
            # too, when bootstrapped): one unit each, as for matched
            m_units = [u for (rs, _), u in units.items()
                       if rs in et.MATCHED_ROW_SETS]
            for u in m_units if matrix_models else []:
                i, j = et._reference_indices(u["ref"], gene_index)
                for mname, kind, scorers in matrix_models:
                    e = ExtEntry(mname, kind, "zero_shot_raw")
                    for k, scorer in scorers:
                        v = np.asarray(scorer(i, j), dtype=np.float64)
                        et.require_finite(v, f"{mname} fold {k}")
                        e.folds[k] = v
                    u["entries"][(mname, "zero_shot_raw")] = e
            # the SynLethDB label as a predictor on matched_in_sldb
            in_units = [u for (rs, _), u in units.items()
                        if rs == et.ROW_MATCHED_IN_SLDB]
            if in_units and pos_keys is not None and zs_pub is not None and \
                    (zs_pub["model"] == et.SLDB_LABEL_NAME).any():
                u = in_units[0]
                lab = np.isin(u["keys"], list(pos_keys)).astype(np.float64)
                e = ExtEntry(et.SLDB_LABEL_NAME, et.SLDB_LABEL_KIND,
                             "zero_shot_raw")
                some = next(iter(u["entries"].values()))
                for k in some.folds:
                    e.folds[k] = lab
                u["entries"][(et.SLDB_LABEL_NAME, "zero_shot_raw")] = e

            for (rs, digest), u in units.items():
                W = resample_counts(len(u["keys"]), args.n_boot, args.seed,
                                    (unit_hash(screen, split, rs, digest),))
                vecs = {(eid, k): v for eid, e in u["entries"].items()
                        for k, v in e.folds.items()}
                pts, reps = bootstrap_spearman(u["target"], vecs, W,
                                               args.chunk)
                fold_ids = sorted({k for (_, k) in vecs})
                pooled = {}
                for eid, e in u["entries"].items():
                    if sorted(e.folds) != fold_ids:
                        raise SystemExit(f"{screen}/{split}/{rs}: {eid} has "
                                         f"folds {sorted(e.folds)}, others "
                                         f"{fold_ids}")
                    base = {"model": e.model, "kind": e.kind,
                            "variant": variant_of(e.model),
                            "method": e.method, "pca_arm": arm,
                            "screen": screen, "split": split, "row_set": rs,
                            "metric": "spearman"}
                    for k in fold_ids:
                        pub = published_spearman(zs_pub, ft_pub, e, screen,
                                                 split, rs, k)
                        check_close(pts[(eid, k)], pub, args.check_atol,
                                    f"{e.model}/{e.method}/{screen}/{split}/"
                                    f"{rs}/fold{k}", check_problems)
                        rows.append({**base, "fold": str(k), "published": pub,
                                     **boot_stats(pts[(eid, k)],
                                                  reps[(eid, k)], alpha),
                                     "ci_level": 1 - alpha,
                                     "n_boot": args.n_boot,
                                     "n_rows": len(u["keys"]), "n_folds": 1,
                                     "seed": args.seed})
                    # Over the folds with a finite point only, as the
                    # published mean (external_transfer.aggregate_folds drops
                    # a constant-score fold's NaN): a plain mean made the
                    # pooled CI NaN while the table printed a number.
                    fin = [k for k in fold_ids
                           if np.isfinite(pts[(eid, k)])]
                    pt = (float(np.mean([pts[(eid, k)] for k in fin]))
                          if fin else float("nan"))
                    rep = (np.mean([reps[(eid, k)] for k in fin], axis=0)
                           if fin else np.full(args.n_boot, np.nan))
                    pooled[eid] = (pt, rep, {k: (pts[(eid, k)], reps[(eid, k)])
                                             for k in fold_ids})
                    pubs = [published_spearman(zs_pub, ft_pub, e, screen,
                                               split, rs, k) for k in fold_ids]
                    pub_fin = [v for v in pubs
                               if v is not None and np.isfinite(v)]
                    pub_mean = (None if any(v is None for v in pubs) else
                                float(np.mean(pub_fin)) if pub_fin
                                else float("nan"))
                    check_close(pt, pub_mean, args.check_atol,
                                f"{e.model}/{e.method}/{screen}/{split}/{rs}/"
                                f"mean", check_problems)
                    rows.append({**base, "fold": "mean", "published": pub_mean,
                                 **boot_stats(pt, rep, alpha),
                                 "ci_level": 1 - alpha, "n_boot": args.n_boot,
                                 "n_rows": len(u["keys"]),
                                 "n_folds": len(fin), "seed": args.seed})
                    if args.save_replicates:
                        reps_npz[f"{e.model}|{e.method}|{screen}|{split}|"
                                 f"{rs}"] = np.vstack(
                                     [reps[(eid, k)] for k in fold_ids])
                for ea, eb in external_pairs(u["entries"], rs, args.pair):
                    A_, B_ = u["entries"][ea], u["entries"][eb]
                    base = {"model": A_.model, "method": A_.method,
                            "peer": B_.model, "peer_method": B_.method,
                            "pca_arm": arm, "screen": screen, "split": split,
                            "row_set": rs, "metric": "spearman"}
                    for k in fold_ids:
                        pa, ra = pooled[ea][2][k]
                        pb, rb = pooled[eb][2][k]
                        diff_rows.append({**base, "fold": str(k),
                                          **diff_stats(pa - pb, ra - rb, alpha),
                                          "ci_level": 1 - alpha,
                                          "n_boot": args.n_boot,
                                          "n_rows": len(u["keys"]),
                                          "n_folds": 1, "seed": args.seed})
                    # Paired over the folds finite for BOTH sources.
                    both = [k for k in fold_ids
                            if np.isfinite(pooled[ea][2][k][0])
                            and np.isfinite(pooled[eb][2][k][0])]
                    if both:
                        dpt = float(np.mean([pooled[ea][2][k][0] -
                                             pooled[eb][2][k][0]
                                             for k in both]))
                        drep = np.mean([pooled[ea][2][k][1] -
                                        pooled[eb][2][k][1] for k in both],
                                       axis=0)
                    else:
                        dpt, drep = float("nan"), np.full(args.n_boot, np.nan)
                    diff_rows.append({**base, "fold": "mean",
                                      **diff_stats(dpt, drep, alpha),
                                      "ci_level": 1 - alpha,
                                      "n_boot": args.n_boot,
                                      "n_rows": len(u["keys"]),
                                      "n_folds": len(both),
                                      "seed": args.seed})
            print(f"  {screen}/{split}: {sum(len(u['entries']) for u in units.values())}"
                  f" score sources over {len(units)} row set(s)  "
                  f"[{time.time() - t0:.1f}s]")
    if check_problems:
        head = "\n  ".join(check_problems[:40])
        raise SystemExit(
            f"REFUSING TO WRITE: {len(check_problems)} external point "
            f"estimate(s) do not reproduce {ext_dir}'s per-fold tables (atol "
            f"{args.check_atol}):\n  {head}")
    agree_rows = []
    if pos_keys is not None:
        agree_rows = bootstrap_sldb_agreement(
            args, arm, siamese, screens, splits, gene_index, pos_keys,
            agree_pub, check_problems)
    if check_problems:
        head = "\n  ".join(check_problems[:40])
        raise SystemExit(
            f"REFUSING TO WRITE: {len(check_problems)} SynLethDB agreement "
            f"point estimate(s) do not reproduce {ext_dir}/sldb_agreement.csv "
            f"(atol {args.check_atol}):\n  {head}")
    write_csv_atomic(rows, out_dir / "ci_external.csv")
    write_csv_atomic(diff_rows, out_dir / "ci_external_diffs.csv")
    if agree_rows:
        write_csv_atomic(agree_rows, out_dir / "ci_sldb_agreement.csv")
    else:
        manifest.setdefault("skipped", []).append(
            f"external: ci_sldb_agreement.csv not written (SynLethDB "
            f"positives {sl_path} not found)")
    if args.save_replicates:
        np.savez_compressed(out_dir / "replicates_external.npz", **reps_npz)
    manifest["external"] = {
        "external_dir": str(ext_dir),
        "model_cv": model_cv,
        "siamese": {n: str(d) for n, d in siamese},
        "baselines": {n: str(d) for n, d in kept_baselines},
        "degree_prior": any(n == et.DEGREE_PRIOR_NAME
                            for n, _, _ in matrix_models),
        "screens": screens, "splits": splits,
        "row_sets": list(row_sets),
        "sldb_agreement": {"sl_path": str(sl_path),
                           "rows": len(agree_rows),
                           "published": agree_pub is not None},
        "resample": "rows of each (screen, split, row_set), shared by every "
                    "fold model and every model on those rows",
    }
    print(f"wrote {out_dir / 'ci_external.csv'} ({len(rows)} rows) and "
          f"{out_dir / 'ci_external_diffs.csv'} ({len(diff_rows)} rows)")


AGREEMENT_METRICS = ("auroc_target_vs_label", "spearman_target_vs_label")


def bootstrap_sldb_agreement(args, arm: str, siamese, screens, splits,
                             gene_index, pos_keys: set, agree_pub,
                             check_problems: List[str]) -> List[dict]:
    """CIs of external_transfer's sldb_agreement.csv: per screen and scope
    (every scored screen pair, both genes in genes.txt, per split the
    matched_incl_sldb test rows), on the pairs SynLethDB labels (the eval's
    sldb_overlap; label 1 = SynLethDB SL), AUROC of the target for SL vs
    non-SL and Spearman(label, target), the labelled pairs resampled
    (default_rng([seed, crc32(screen|scope|split|sldb_agreement|digest), b])).
    Point estimates are checked against the published table when present."""
    import pandas as pd
    name, d = siamese[0]
    rows = []
    for screen in screens:
        pairs, reference = None, {}
        for split in splits:
            df = et.read_pair_csv(d / f"eval_finetuning_{split}" /
                                  f"{screen}_predictions.csv")
            if pairs is None:
                pairs = df
            m = et.row_mask(df, gene_index, et.ROW_MATCHED_INCL)
            reference[(screen, split)] = df.loc[m].sort_values(
                "key").reset_index(drop=True)
        for scope, sp, sub in et.screen_scopes(screen, pairs, reference,
                                               gene_index, splits):
            lab = sub["sldb_overlap"].to_numpy(dtype=bool)
            sub = sub.loc[lab].sort_values("key").reset_index(drop=True)
            keys = sub["key"].to_numpy()
            y = np.isin(keys, list(pos_keys))
            t = sub["target"].to_numpy(dtype=np.float64)
            base = {"screen": screen, "scope": scope, "split": sp,
                    "pca_arm": arm, "defined_by": name}
            if len(keys) < 3 or y.all() or not y.any():
                for met in AGREEMENT_METRICS:
                    rows.append({**base, "metric": met, "point": float("nan"),
                                 "n_labelled": int(len(keys)),
                                 "n_pos": int(y.sum())})
                continue
            W = resample_counts(len(keys), args.n_boot, args.seed,
                                (unit_hash(screen, scope, sp, "sldb_agreement",
                                           et.matched_digest(keys)),))
            cp, cr = bootstrap_classification(y.astype(np.int64),
                                              {"target": t}, W, args.chunk)
            sp_pts, sp_reps = bootstrap_spearman(t, {"label": y.astype(
                np.float64)}, W, args.chunk)
            got = {"auroc_target_vs_label": (cp["target"]["auroc"],
                                             cr["target"]["auroc"]),
                   "spearman_target_vs_label": (sp_pts["label"],
                                                sp_reps["label"])}
            for met, (pt, rep) in got.items():
                pub = None
                if agree_pub is not None:
                    hit = agree_pub[(agree_pub["screen"] == screen) &
                                    (agree_pub["scope"] == scope) &
                                    (agree_pub["split"].astype(str) == sp)]
                    if len(hit) == 1:
                        pub = float(hit[met].iloc[0])
                    check_close(pt, pub, args.check_atol,
                                f"sldb_agreement {screen}/{scope}/{sp}/{met}",
                                check_problems)
                rows.append({**base, "metric": met, "published": pub,
                             **boot_stats(pt, rep, args.alpha),
                             "ci_level": 1 - args.alpha, "n_boot": args.n_boot,
                             "n_labelled": int(len(keys)),
                             "n_pos": int(y.sum()), "seed": args.seed})
    return rows


def published_spearman(zs_pub, ft_pub, e: ExtEntry, screen, split, rs, k):
    """The published per-fold Spearman for one entry, or None."""
    if e.method == "zero_shot_raw" and zs_pub is not None:
        frame, extra = zs_pub, {}
    elif ft_pub is not None:
        frame, extra = ft_pub, {"method": e.method}
    else:
        return None
    sel = ((frame["model"] == e.model) & (frame["screen"] == screen) &
           (frame["split"] == split) & (frame["row_set"] == rs) &
           (frame["fold"] == k))
    for c, v in extra.items():
        sel &= frame[c] == v
    hit = frame.loc[sel, "spearman"]
    if len(hit) != 1:
        return None
    return float(hit.iloc[0])


def external_pairs(entries: Dict[Tuple[str, str], ExtEntry], rs: str,
                   extra: Sequence[Tuple[str, str]]
                   ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    out = []
    sia = sorted({m for (m, meth), e in entries.items() if e.kind == "siamese"})
    others = sorted({m for (m, meth), e in entries.items()
                     if e.kind != "siamese"})
    for a in sia:
        for b in others:
            out.append(((a, "zero_shot_raw"), (b, "zero_shot_raw")))
        for mode in FT_OLS_WARM:
            if (a, mode) in entries and (a, "zero_shot_ols") in entries:
                out.append(((a, mode), (a, "zero_shot_ols")))
        if (a, "Full") in entries and (a, FT_SCRATCH) in entries:
            out.append(((a, "Full"), (a, FT_SCRATCH)))
    for a, b in extra:
        ea, eb = (a, "zero_shot_raw"), (b, "zero_shot_raw")
        if ea in entries and eb in entries and (ea, eb) not in out:
            out.append((ea, eb))
    return out


# =============================================================================
# CLI
# =============================================================================


def report_missing(part: str, problems: List[str], skip: bool,
                   manifest: dict) -> None:
    if not problems:
        return
    msg = "\n  ".join(problems)
    if not skip:
        raise SystemExit(
            f"MISSING INPUTS ({part}): {len(problems)} problem(s); nothing "
            f"was computed or written.\n  {msg}\n"
            f"Per-pair score matrices (fold_<k>_<cv>_predictions.npy, "
            f"~300 MB each) are written on the cluster and are not part of "
            f"the synced results. Run this there, point --pred_dir at a "
            f"copy, or pass --skip_missing to bootstrap the rows that are "
            f"present (the skipped ones are listed in manifest.json).")
    print(f"WARNING ({part}, --skip_missing): skipping\n  {msg}")
    manifest.setdefault("skipped", []).extend(f"{part}: {p}" for p in problems)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Bootstrap CIs of the benchmark metrics from saved test "
        "predictions (no retraining).")
    ap.add_argument("--part", choices=["tiers", "external", "both"],
                    default="both")
    ap.add_argument("--pca_arm", choices=ARMS, default="none",
                    help="whose SLeuth rows / external tables to read "
                    "(sl_comparison/results[/pca_<arm>])")
    ap.add_argument("--results_dir", default=None,
                    help="the arm's comparison dir (default from --pca_arm)")
    ap.add_argument("--shared_results_dir",
                    default=str(REPO_ROOT / "sl_comparison" / "results"),
                    help="where SLGNN / NSF4SL / SLMGAE / DegreePrior rows live")
    ap.add_argument("--folds_dir",
                    default=str(REPO_ROOT / "sl_comparison" / "folds"))
    ap.add_argument("--cvs", nargs="+", default=list(CVS), choices=CVS)
    ap.add_argument("--models", nargs="+", default=None,
                    help="tier rows to bootstrap (default: every row of the "
                    "arm's table)")
    ap.add_argument("--pred_dir", action="append", default=[],
                    metavar="MODEL=DIR",
                    help="prediction dir of a tier row (DIR may contain {cv}); "
                    "overrides the pipeline's location")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B",
                    help="extra paired difference A - B (tiers: rows; "
                    "external: zero_shot_raw of two models)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="1 - CI level (default 0.05: 95%% percentile CI)")
    ap.add_argument("--stratify", action="store_true",
                    help="tiers: resample positives and negatives separately "
                    "(fixed prevalence); default resamples pairs")
    ap.add_argument("--chunk", type=int, default=64,
                    help="resamples per vectorised block (memory knob)")
    ap.add_argument("--check_atol", type=float, default=1e-9,
                    help="max |point - published| accepted")
    ap.add_argument("--no_sklearn_check", action="store_true",
                    help="skip the extra point check against sklearn/auprg "
                    "(the check against the published JSON always runs)")
    ap.add_argument("--allow_unstamped", action="store_true")
    ap.add_argument("--allow_no_tables", action="store_true",
                    help="bootstrap the tiers without the arm's "
                    "paper_tables/tab_tiers.csv (its refused rows are then "
                    "not left out)")
    ap.add_argument("--skip_missing", action="store_true",
                    help="bootstrap what is present; list the rest in "
                    "manifest.json instead of stopping")
    ap.add_argument("--external_dir", default=None,
                    help="external_transfer.py output dir (default "
                    "<cmp>/external_transfer)")
    ap.add_argument("--ext_siamese", action="append", default=[],
                    metavar="NAME=MODEL_DIR",
                    help="override the SLeuth dirs read from summary.md")
    ap.add_argument("--ext_baseline", action="append", default=[],
                    metavar="NAME=PRED_DIR",
                    help="override the baseline dirs read from summary.md")
    ap.add_argument("--ext_model_cv", default=None, choices=CVS)
    ap.add_argument("--no_ext_degree_prior", dest="ext_degree_prior",
                    action="store_false")
    ap.add_argument("--screens", nargs="+", default=None)
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--save_replicates", action="store_true")
    ap.add_argument("--out_dir", default=None,
                    help="default <cmp>/bootstrap/")
    args = ap.parse_args(argv)
    if args.n_boot < 2:
        ap.error("--n_boot must be >= 2")
    if not (0 < args.alpha < 1):
        ap.error("--alpha must be in (0, 1)")
    args.pred_dir_map = {}
    for spec in args.pred_dir:
        name, sep, path = spec.partition("=")
        if not sep:
            ap.error(f"--pred_dir expects MODEL=DIR, got {spec!r}")
        args.pred_dir_map[name.strip()] = path.strip()
    pairs = []
    for spec in args.pair:
        a, sep, b = spec.partition(":")
        if not sep:
            ap.error(f"--pair expects A:B, got {spec!r}")
        pairs.append((a.strip(), b.strip()))
    args.pair = pairs
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    arm = args.pca_arm
    cmp_dir = Path(args.results_dir) if args.results_dir else arm_cmp_dir(arm)
    out_dir = Path(args.out_dir) if args.out_dir else cmp_dir / "bootstrap"
    folds = Path(args.folds_dir).resolve()
    if out_dir.resolve() == folds or folds in out_dir.resolve().parents:
        raise SystemExit("--out_dir must not be inside the fold set")
    import sklearn
    manifest = {
        "script": "sl_comparison/bootstrap_ci.py",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "pca_arm": arm,
        "n_boot": args.n_boot, "seed": args.seed, "alpha": args.alpha,
        "ci": "percentile",
        "resample_tiers": ("stratified by class" if args.stratify else
                           "test pairs with replacement") +
        "; one resample set per (cv, fold), shared by every model",
        "caveat": "pairs are resampled as independent units; pairs share "
                  "genes, so these CIs cover test-pair sampling only (not "
                  "gene-level or training variability)",
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "sklearn": sklearn.__version__},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.part in ("tiers", "both"):
        print(f"=== tiers (PCA arm {arm}) ===")
        run_tiers(args, arm, out_dir, manifest)
    if args.part in ("external", "both"):
        print(f"=== external (PCA arm {arm}) ===")
        run_external(args, arm, out_dir, manifest)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json_atomic(manifest, out_dir / "manifest.json")
    print(f"wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
