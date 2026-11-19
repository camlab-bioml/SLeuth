#!/usr/bin/env python3
"""Tables 5 and 6 of the ICLR revision (external screens), with CIs and tests.

Inputs (read-only), per PCA arm (none = sl_comparison/results, joint =
sl_comparison/results/pca_joint):
  paper_tables/tab_external.csv, paper_tables/tab_external_ft.csv
  bootstrap/ci_external.csv, bootstrap/ci_external_diffs.csv,
  bootstrap/ci_sldb_agreement.csv, bootstrap/manifest.json
  external_transfer/zero_shot_per_fold.csv, finetune_per_fold.csv
  out_<arm>/replicates.npz  (repro_boot.py: the SAME bootstrap resamples,
                             rebuilt locally for SLeuth + DegreePrior and
                             checked against the published CSVs)

Statistics (stated in the report):
  * bootstrap: B=1000, seed 42, rows of each (screen, split, row set)
    resampled with replacement, one resample shared by every fold model and
    every model (paired); the CI is the percentile 95% CI of the 5-fold MEAN
    Spearman. Two-sided p from the one-sided p_boot_le0:
        p2 = min(1, 2 * max(min(p, 1 - p), 1 / (B + 1)))   (floor ~0.002)
  * Holm within each family (named in the output 'family' column).
  * fold-paired tests over the 5 fold models: paired t-test (two-sided, df 4)
    and the exact sign-flip permutation test (32 sign patterns; the smallest
    attainable two-sided p is 2/32 = 0.0625).
Usage: python -I analyze.py ARM
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCREENS = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
SPLITS = ("cv1", "cv2", "cv3")
SPLIT_LABEL = {"cv1": "random pairs", "cv2": ">=1 gene unseen",
               "cv3": "both genes unseen"}
SLEUTH = ("siamese_sl", "siamese_sl_pd_pooled")
SHOW = {"siamese_sl": "SLeuth (cell line)",
        "siamese_sl_pd_pooled": "SLeuth (pooled)", "SLMGAE": "SLMGAE",
        "SLGNN": "SLGNN", "NSF4SL": "NSF4SL", "DegreePrior": "DegreePrior",
        "SynLethDB_label": "SynLethDB label"}
PEERS = ("SLMGAE", "SLGNN", "NSF4SL", "DegreePrior")
LOCAL_SOURCES = SLEUTH + ("DegreePrior",)
B = 1000


def cmp_dir(arm):
    c = REPO / "sl_comparison" / "results"
    return c if arm == "none" else c / f"pca_{arm}"


def p2(p, b=B, p_ge0=None):
    """Two-sided bootstrap p from the one-sided p_boot_le0 (the rule in the
    docstring). With p_ge0 (local replicates) the upper tail is the fraction
    >= 0 instead of 1 - p_le0: identical without ties at 0, and 1 for a
    difference that is exactly 0 in every resample (e.g. Full returning the
    OLS warm start in every fold)."""
    if p is None or not np.isfinite(p):
        return float("nan")
    upper = (1 - p) if p_ge0 is None else p_ge0
    return float(min(1.0, 2 * max(min(p, upper), 1.0 / (b + 1))))


def p2_pub(row):
    """p2 of a published ci_external_diffs.csv row; a difference that is
    exactly 0 with a [0, 0] CI (all resamples 0) gets p = 1."""
    if row.point == 0 and row.ci_low == 0 and row.ci_high == 0:
        return 1.0
    return p2(row.p_boot_le0)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    out = np.full(len(ps), np.nan)
    ok = np.flatnonzero(np.isfinite(ps))
    m = len(ok)
    if not m:
        return out
    order = ok[np.argsort(ps[ok], kind="stable")]
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[idx]))
        out[idx] = run
    return out


def fold_tests(d):
    """(paired t p, exact sign-flip p, n, n_same_sign) for per-fold diffs."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n, 0
    same = int(max((d > 0).sum(), (d < 0).sum()))
    if np.all(d == 0):
        return float("nan"), 1.0, n, 0
    with np.errstate(all="ignore"):
        tp = float(stats.ttest_1samp(d, 0.0).pvalue)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    tstat = np.abs((signs * d).mean(axis=1))
    obs = abs(d.mean())
    pp = float(np.mean(tstat >= obs * (1 - 1e-12) - 1e-15))
    return tp, pp, n, same


def pct_ci(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if not r.size:
        return float("nan"), float("nan")
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


class Data:
    def __init__(self, arm):
        c = cmp_dir(arm)
        self.arm = arm
        self.c = c
        self.tab_ext = pd.read_csv(c / "paper_tables" / "tab_external.csv")
        self.tab_ft = pd.read_csv(c / "paper_tables" / "tab_external_ft.csv")
        self.ci = pd.read_csv(c / "bootstrap" / "ci_external.csv",
                              dtype={"fold": str})
        self.cid = pd.read_csv(c / "bootstrap" / "ci_external_diffs.csv",
                               dtype={"fold": str})
        self.agree = pd.read_csv(c / "bootstrap" / "ci_sldb_agreement.csv")
        self.man = json.load(open(c / "bootstrap" / "manifest.json"))
        self.zs = pd.read_csv(c / "external_transfer" /
                              "zero_shot_per_fold.csv")
        self.ft = pd.read_csv(c / "external_transfer" /
                              "finetune_per_fold.csv")
        self.row_sets = pd.read_csv(c / "external_transfer" / "row_sets.csv")
        rp = HERE / f"out_{arm}" / "replicates.npz"
        self.reps = dict(np.load(rp)) if rp.exists() else None
        chk = HERE / f"out_{arm}" / "repro_check.json"
        self.repro = json.load(open(chk)) if chk.exists() else None
        assert self.man["n_boot"] == B and self.man["seed"] == 42
        # indexes
        self.ci_mean = self.ci[self.ci.fold == "mean"].set_index(
            ["model", "method", "screen", "split", "row_set"])
        self.cid_mean = self.cid[self.cid.fold == "mean"].set_index(
            ["model", "method", "peer", "peer_method", "screen", "split",
             "row_set"])

    # ---- per-fold values ----
    def zs_folds(self, model, screen, split, rs):
        x = self.zs[(self.zs.model == model) & (self.zs.screen == screen) &
                    (self.zs.split == split) & (self.zs.row_set == rs)]
        x = x.sort_values("fold")
        return x.spearman.to_numpy(dtype=float), x.fold.to_numpy()

    def ft_folds(self, model, screen, split, rs, method):
        x = self.ft[(self.ft.model == model) & (self.ft.screen == screen) &
                    (self.ft.split == split) & (self.ft.row_set == rs) &
                    (self.ft.method == method)].sort_values("fold")
        return x.spearman.to_numpy(dtype=float), x.fold.to_numpy()

    def folds(self, model, method, screen, split, rs):
        if method == "zero_shot_raw" and model not in SLEUTH:
            return self.zs_folds(model, screen, split, rs)[0]
        if method == "zero_shot_raw" and rs in ("matched_in_sldb",
                                                "matched_incl_sldb"):
            return self.zs_folds(model, screen, split, rs)[0]
        return self.ft_folds(model, screen, split, rs, method)[0]

    def ci_of(self, model, method, screen, split, rs):
        return self.ci_mean.loc[(model, method, screen, split, rs)]

    def diff_of(self, a, ma, b, mb, screen, split, rs):
        return self.cid_mean.loc[(a, ma, b, mb, screen, split, rs)]

    # ---- local replicates (same resamples as the published run) ----
    def rep_mean(self, model, method, screen, split, rs):
        k = f"{model}|{method}|{screen}|{split}|{rs}"
        pts = self.reps[k + "|folds_point"]
        reps = self.reps[k + "|folds_reps"]
        fin = np.isfinite(pts)
        return pts, reps, fin

    def local_diff(self, a, ma, b, mb, screen, split_a, rs, split_b=None):
        """Fold-mean difference A - B with its bootstrap replicates. Same
        split: paired over the shared resamples and the folds finite for
        both. Different splits (split_b): independent resamples."""
        split_b = split_b or split_a
        pa, ra, fa = self.rep_mean(a, ma, screen, split_a, rs)
        pb, rb, fb = self.rep_mean(b, mb, screen, split_b, rs)
        if split_b == split_a:
            both = fa & fb
            point = float((pa[both] - pb[both]).mean())
            drep = (ra[both] - rb[both]).mean(axis=0)
        else:
            point = float(pa[fa].mean() - pb[fb].mean())
            drep = ra[fa].mean(axis=0) - rb[fb].mean(axis=0)
        lo, hi = pct_ci(drep)
        fin = drep[np.isfinite(drep)]
        p_le0 = float(np.mean(fin <= 0))
        p_ge0 = float(np.mean(fin >= 0))
        return point, lo, hi, p_le0, pa - pb, p_ge0

    def local_single(self, model, method, screen, split, rs):
        pts, reps, fin = self.rep_mean(model, method, screen, split, rs)
        r = reps[fin].mean(axis=0)
        lo, hi = pct_ci(r)
        rf = r[np.isfinite(r)]
        return (float(pts[fin].mean()), lo, hi, float(np.mean(rf <= 0)),
                float(np.mean(rf >= 0)))


def table5(D):
    """Zero-shot Spearman on the matched rows (Table 5) + tests vs 0."""
    rows = []
    for screen in SCREENS:
        for split in SPLITS:
            for model in SLEUTH + PEERS:
                te = D.tab_ext[(D.tab_ext.model == model) &
                               (D.tab_ext.screen == screen) &
                               (D.tab_ext.split == split) &
                               (D.tab_ext.row_set == "matched")].iloc[0]
                ci = D.ci_of(model, "zero_shot_raw", screen, split,
                             "matched")
                pf = D.folds(model, "zero_shot_raw", screen, split,
                             "matched")
                assert abs(pf.mean() - te.spearman_mean) < 1e-9
                assert abs(ci.point - te.spearman_mean) < 1e-9
                assert abs(pf.std(ddof=0) - te.spearman_sd) < 1e-9
                tp, pp, n, same = fold_tests(pf)
                rec = {"arm": D.arm, "screen": screen, "split": split,
                       "split_label": SPLIT_LABEL[split],
                       "screen_role": te.screen_role, "model": model,
                       "row_set": "matched", "n_rows": int(te.n_rows),
                       "scored_head": te.scored_head,
                       "mean": te.spearman_mean, "sd_pop": te.spearman_sd,
                       "sd_ddof1": pf.std(ddof=1),
                       "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                       "boot_se": ci.boot_se,
                       "ci_excludes_0": bool(ci.ci_low > 0 or
                                             ci.ci_high < 0),
                       "folds": " ".join(f"{v:+.4f}" for v in pf),
                       "n_folds_same_sign": same,
                       "t_p_vs0": tp, "signflip_p_vs0": pp,
                       "best_matched": te.best_matched,
                       "src": "tab_external.csv spearman_mean/_sd; "
                              "ci_external.csv fold=mean ci_low/ci_high"}
                if D.reps is not None and model in LOCAL_SOURCES:
                    pt, lo, hi, ple0, pge0 = D.local_single(
                        model, "zero_shot_raw", screen, split, "matched")
                    assert abs(lo - ci.ci_low) < 1e-9 and \
                        abs(hi - ci.ci_high) < 1e-9
                    rec["boot_p2_vs0"] = p2(ple0, p_ge0=pge0)
                rows.append(rec)
    df = pd.DataFrame(rows)
    # Holm over the 12 cells of each model for the bootstrap test vs 0
    df["boot_p2_vs0_holm12"] = np.nan
    df["t_p_vs0_holm12"] = np.nan
    for m, g in df.groupby("model"):
        if "boot_p2_vs0" in g and g["boot_p2_vs0"].notna().all():
            df.loc[g.index, "boot_p2_vs0_holm12"] = holm(g["boot_p2_vs0"])
        df.loc[g.index, "t_p_vs0_holm12"] = holm(g["t_p_vs0"])
    return df


def table5_diffs(D):
    """SLeuth - peer on the matched rows: published bootstrap + fold tests.
    Family: the 8 comparisons (2 SLeuth variants x 4 comparators) of one
    screen x split."""
    rows = []
    for screen in SCREENS:
        for split in SPLITS:
            fam = []
            for a in SLEUTH:
                for b in PEERS:
                    d = D.diff_of(a, "zero_shot_raw", b, "zero_shot_raw",
                                  screen, split, "matched")
                    fa = D.folds(a, "zero_shot_raw", screen, split,
                                 "matched")
                    fb = D.folds(b, "zero_shot_raw", screen, split,
                                 "matched")
                    dk = fa - fb
                    assert abs(dk.mean() - d.point) < 1e-9
                    tp, pp, n, same = fold_tests(dk)
                    fam.append({"arm": D.arm, "screen": screen,
                                "split": split, "row_set": "matched",
                                "A": a, "B": b, "delta": d.point,
                                "ci_low": d.ci_low, "ci_high": d.ci_high,
                                "p_boot_le0": d.p_boot_le0,
                                "p2": p2_pub(d),
                                "fold_deltas": " ".join(f"{v:+.4f}"
                                                        for v in dk),
                                "n_folds_same_sign": same,
                                "t_p": tp, "signflip_p": pp,
                                "family": f"{screen}x{split}: 2 SLeuth x 4 "
                                          f"comparators (8)",
                                "src": "ci_external_diffs.csv fold=mean"})
            ph = holm([r["p2"] for r in fam])
            th = holm([r["t_p"] for r in fam])
            for r, h, t in zip(fam, ph, th):
                r["p2_holm"] = h
                r["t_p_holm"] = t
            rows += fam
    return pd.DataFrame(rows)


def variant_diff(D):
    """siamese_sl - siamese_sl_pd_pooled (zero-shot, matched): local
    bootstrap on the published resamples; family = the 12 cells."""
    rows = []
    for screen in SCREENS:
        for split in SPLITS:
            pt, lo, hi, ple0, dk, pge0 = D.local_diff(
                "siamese_sl", "zero_shot_raw", "siamese_sl_pd_pooled",
                "zero_shot_raw", screen, split, "matched")
            tp, pp, n, same = fold_tests(dk)
            rows.append({"arm": D.arm, "screen": screen, "split": split,
                         "A": "siamese_sl", "B": "siamese_sl_pd_pooled",
                         "delta": pt, "ci_low": lo, "ci_high": hi,
                         "p_boot_le0": ple0, "p2": p2(ple0, p_ge0=pge0),
                         "fold_deltas": " ".join(f"{v:+.4f}" for v in dk),
                         "n_folds_same_sign": same, "t_p": tp,
                         "signflip_p": pp,
                         "family": "12 screen x split cells",
                         "src": "local replicates (repro_boot.py)"})
    df = pd.DataFrame(rows)
    df["p2_holm"] = holm(df.p2)
    df["t_p_holm"] = holm(df.t_p)
    return df


def table5_sldb(D):
    """matched_in_sldb rows (the SynLethDB part of the matched test pairs):
    the SynLethDB label as a predictor next to every model, and SLeuth -
    label (published diffs). Family for the diffs: the 2 SLeuth - label of
    one screen x split... reported with Holm over 24 (2 x 12)."""
    rows, drows = [], []
    for screen in SCREENS:
        for split in SPLITS:
            for model in SLEUTH + PEERS + ("SynLethDB_label",):
                ci = D.ci_of(model, "zero_shot_raw", screen, split,
                             "matched_in_sldb")
                pf = D.zs_folds(model, screen, split, "matched_in_sldb")[0]
                te = D.tab_ext[(D.tab_ext.model == model) &
                               (D.tab_ext.screen == screen) &
                               (D.tab_ext.split == split) &
                               (D.tab_ext.row_set == "matched_in_sldb")]
                te = te.iloc[0]
                assert abs(pf.mean() - te.spearman_mean) < 1e-9
                rows.append({"arm": D.arm, "screen": screen, "split": split,
                             "model": model, "row_set": "matched_in_sldb",
                             "n_rows": int(te.n_rows),
                             "mean": te.spearman_mean,
                             "sd_pop": te.spearman_sd,
                             "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                             "ci_excludes_0": bool(ci.ci_low > 0 or
                                                   ci.ci_high < 0)})
            for a in SLEUTH:
                d = D.diff_of(a, "zero_shot_raw", "SynLethDB_label",
                              "zero_shot_raw", screen, split,
                              "matched_in_sldb")
                drows.append({"arm": D.arm, "screen": screen,
                              "split": split, "A": a,
                              "B": "SynLethDB_label", "delta": d.point,
                              "ci_low": d.ci_low, "ci_high": d.ci_high,
                              "p2": p2_pub(d)})
    dd = pd.DataFrame(drows)
    dd["p2_holm24"] = holm(dd.p2)
    return pd.DataFrame(rows), dd


FT_METHODS = ("zero_shot_raw", "zero_shot_ols", "LP_last", "Full",
              "Scratch")


def table6(D):
    rows = []
    for v in SLEUTH:
        for rs in ("all_sldb_filtered", "matched"):
            for screen in SCREENS:
                for split in SPLITS:
                    for meth in FT_METHODS:
                        tf = D.tab_ft[(D.tab_ft.model == v) &
                                      (D.tab_ft.screen == screen) &
                                      (D.tab_ft.split == split) &
                                      (D.tab_ft.row_set == rs) &
                                      (D.tab_ft.method == meth)].iloc[0]
                        ci = D.ci_of(v, meth, screen, split, rs)
                        pf = D.ft_folds(v, screen, split, rs, meth)[0]
                        assert abs(pf.mean() - tf.spearman_mean) < 1e-9
                        assert abs(ci.point - tf.spearman_mean) < 1e-9
                        rows.append({"arm": D.arm, "model": v,
                                     "row_set": rs,
                                     "paper_convention":
                                     bool(tf.paper_convention),
                                     "screen": screen, "split": split,
                                     "method": meth,
                                     "n_rows": int(tf.n_rows),
                                     "mean": tf.spearman_mean,
                                     "sd_pop": tf.spearman_sd,
                                     "ci_low": ci.ci_low,
                                     "ci_high": ci.ci_high,
                                     "folds": " ".join(f"{x:+.4f}"
                                                       for x in pf),
                                     "scored_head": tf.scored_head})
    return pd.DataFrame(rows)


# (label, A method, B method, source)
FT_CONTRASTS = (
    ("Full - Scratch", "Full", "Scratch", "published"),
    ("LP_last - zero-shot OLS", "LP_last", "zero_shot_ols", "published"),
    ("Full - zero-shot OLS", "Full", "zero_shot_ols", "published"),
    ("LP_last - zero-shot raw", "LP_last", "zero_shot_raw", "local"),
    ("Full - zero-shot raw", "Full", "zero_shot_raw", "local"),
    ("Scratch - zero-shot raw", "Scratch", "zero_shot_raw", "local"),
    ("LP_last - Scratch", "LP_last", "Scratch", "local"),
)


def table6_tests(D):
    rows = []
    for v in SLEUTH:
        for rs in ("all_sldb_filtered", "matched"):
            for label, ma, mb, src in FT_CONTRASTS:
                fam = []
                for screen in SCREENS:
                    for split in SPLITS:
                        fa = D.ft_folds(v, screen, split, rs, ma)[0]
                        fb = D.ft_folds(v, screen, split, rs, mb)[0]
                        dk = fa - fb
                        if src == "published":
                            d = D.diff_of(v, ma, v, mb, screen, split, rs)
                            pt, lo, hi, ple0 = (d.point, d.ci_low,
                                                d.ci_high, d.p_boot_le0)
                            pv = p2_pub(d)
                            if D.reps is not None:  # cross-check
                                q = D.local_diff(v, ma, v, mb, screen,
                                                 split, rs)
                                assert abs(q[0] - pt) < 1e-9 and \
                                    abs(q[1] - lo) < 1e-9 and \
                                    abs(q[3] - ple0) < 1e-12
                        else:
                            pt, lo, hi, ple0, _, pge0 = D.local_diff(
                                v, ma, v, mb, screen, split, rs)
                            pv = p2(ple0, p_ge0=pge0)
                        assert abs(dk[np.isfinite(dk)].mean() - pt) < 1e-9
                        tp, pp, n, same = fold_tests(dk)
                        fam.append({"arm": D.arm, "model": v,
                                    "row_set": rs, "contrast": label,
                                    "screen": screen, "split": split,
                                    "delta": pt, "ci_low": lo,
                                    "ci_high": hi, "p_boot_le0": ple0,
                                    "p2": pv,
                                    "fold_deltas": " ".join(
                                        f"{x:+.4f}" for x in dk),
                                    "n_folds_same_sign": same,
                                    "t_p": tp, "signflip_p": pp,
                                    "source": src + (
                                        " (ci_external_diffs.csv)"
                                        if src == "published" else
                                        " (repro_boot.py replicates)"),
                                    "family": f"{v} x {rs} x {label}: 12 "
                                              f"screen x split cells"})
                ph = holm([r["p2"] for r in fam])
                th = holm([r["t_p"] for r in fam])
                for r, h, t in zip(fam, ph, th):
                    r["p2_holm"], r["t_p_holm"] = h, t
                rows += fam
    return pd.DataFrame(rows)


def split_drops(D, rs="all_sldb_filtered"):
    rows = []
    for v in SLEUTH:
        for meth in ("Full", "LP_last", "Scratch", "zero_shot_raw"):
            fam = []
            for screen in SCREENS:
                for s1, s2 in (("cv1", "cv2"), ("cv1", "cv3"),
                               ("cv2", "cv3")):
                    pt, lo, hi, ple0, _, pge0 = D.local_diff(
                        v, meth, v, meth, screen, s1, rs, split_b=s2)
                    f1 = D.ft_folds(v, screen, s1, rs, meth)[0]
                    f2 = D.ft_folds(v, screen, s2, rs, meth)[0]
                    dk = f1 - f2
                    tp, pp, n, same = fold_tests(dk)
                    m1 = float(np.nanmean(f1))
                    fam.append({"arm": D.arm, "model": v, "row_set": rs,
                                "method": meth, "screen": screen,
                                "from": s1, "to": s2,
                                "mean_from": m1,
                                "mean_to": float(np.nanmean(f2)),
                                "drop": pt, "ci_low": lo, "ci_high": hi,
                                "rel_drop": pt / m1 if m1 else np.nan,
                                "p2": p2(ple0, p_ge0=pge0),
                                "fold_deltas": " ".join(f"{x:+.4f}"
                                                        for x in dk),
                                "t_p": tp, "signflip_p": pp,
                                "family": f"{v} x {meth}: 4 screens x 3 "
                                          f"split contrasts (12)"})
            ph = holm([r["p2"] for r in fam])
            for r, h in zip(fam, ph):
                r["p2_holm"] = h
            rows += fam
    return pd.DataFrame(rows)


def agreement(D):
    a = D.agree.copy()
    a["arm"] = D.arm
    a["ci_excludes_null"] = np.where(
        a.metric.str.startswith("auroc"),
        (a.ci_low > 0.5) | (a.ci_high < 0.5),
        (a.ci_low > 0) | (a.ci_high < 0))
    return a[["arm", "screen", "scope", "split", "metric", "point",
              "ci_low", "ci_high", "ci_excludes_null", "n_labelled",
              "n_pos"]]


def main(arm):
    D = Data(arm)
    out = HERE / f"out_{arm}"
    out.mkdir(exist_ok=True)
    t5 = table5(D)
    t5d = table5_diffs(D)
    t5s, t5sd = table5_sldb(D)
    t6 = table6(D)
    t6t = table6_tests(D)
    sd = split_drops(D)
    sdm = split_drops(D, rs="matched")
    ag = agreement(D)
    vd = variant_diff(D) if D.reps is not None else None
    t5.to_csv(out / "table5_zero_shot.csv", index=False)
    t5d.to_csv(out / "table5_diffs.csv", index=False)
    t5s.to_csv(out / "table5_matched_in_sldb.csv", index=False)
    t5sd.to_csv(out / "table5_sleuth_minus_label.csv", index=False)
    t6.to_csv(out / "table6_finetune.csv", index=False)
    t6t.to_csv(out / "table6_tests.csv", index=False)
    sd.to_csv(out / "split_drops_all_sldb_filtered.csv", index=False)
    sdm.to_csv(out / "split_drops_matched.csv", index=False)
    ag.to_csv(out / "sldb_agreement_ci.csv", index=False)
    if vd is not None:
        vd.to_csv(out / "table5_cellline_minus_pooled.csv", index=False)
    print(f"arm {arm}: wrote {len(t5)} / {len(t5d)} / {len(t6)} / "
          f"{len(t6t)} / {len(sd)} rows to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
