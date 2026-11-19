#!/usr/bin/env python3
"""ICLR 2027 revision (Submission45596, SLeuth): tiers, significance, cell lines,
bilinear probe, baselines.  Read-only on the repo; every output goes to ./out/.

Run:  <venv312>/bin/python -I compute.py

Inputs (all read-only, repo = <repo>):
  sl_comparison/results[/pca_joint]/<row>_<cv>.json      summary.folds[k].{auroc,ap,auprg,baseline}
  sl_comparison/results[/pca_joint]/paper_tables/tab_tiers.csv, tab_cellline.csv  (cross-check)
  sl_comparison/results[/pca_joint]/bootstrap/ci_tiers.csv, ci_tiers_diffs.csv    (fold == 'mean')
  sl_comparison/results/paper_tables/numbers.json        items.fold_geometry (fold sizes)
  siamese_sl/results[/pca_joint]/cellline_ablation_<cv>/results.json  per_fold (self-scored rows)
  ELISL/results/elisl_<cv>/results.json                  summary.folds[k].trainable_params
  paper/main.tex                                         submitted Table 1 (tab:tiers) and Table 4

Statistics (one rule set, used everywhere):
  * main arm = none (no PCA); arm joint only for the appendix.
  * sd = population sd (ddof=0), as the paper tables and tab_tiers.csv.
  * bootstrap two-sided p from the one-sided p_boot_le0 with B = 1000:
        p2 = min(1, 2 * max(min(p, 1 - p), 1 / (B + 1)))   (floor 2/1001 = 0.002)
  * fold-paired tests over the 5 splits (folds paired by position; checked by the
    per-fold test prevalence): two-sided paired t-test (df = 4) and the exact
    sign-flip permutation test over all 2^5 sign vectors (min two-sided p 0.0625),
    plus, as a sensitivity check only, the Nadeau-Bengio corrected t
    (variance x (1/5 + n_test/n_train)).
  * Holm within explicitly defined families (see FAMILY_* below).
"""
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "sl_comparison/results"
CMP = {"none": SHARED, "joint": SHARED / "pca_joint"}
SIA = {"none": REPO / "siamese_sl/results", "joint": REPO / "siamese_sl/results/pca_joint"}
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

CVS = ["cv1", "cv2", "cv3"]
TIER = {"cv1": "I", "cv2": "II", "cv3": "III"}
METRICS = ["auroc", "ap", "auprg"]
MLABEL = {"auroc": "AUROC", "ap": "AUPR (AP)", "auprg": "AUPRG"}
B = 1000
P_FLOOR = 2.0 / (B + 1)

SLEUTH = ["siamese_sl_pd_pooled", "siamese_sl"]
ALLMOD = ["siamese_sl_allmod_pd_pooled", "siamese_sl_allmod"]
PEERS_NONE = ["SLMGAE", "SLGNN", "NSF4SL", "ELISL", "BilinearProbe", "DegreePrior"]
PEERS = {"none": PEERS_NONE, "joint": [p for p in PEERS_NONE if p != "BilinearProbe"]}
ROWS = {arm: SLEUTH + ALLMOD + PEERS[arm] for arm in CMP}
DISPLAY = {
    "siamese_sl_pd_pooled": "SLeuth (pooled)",
    "siamese_sl": "SLeuth-CL (cell-line heads)",
    "siamese_sl_allmod_pd_pooled": "SLeuth-all (pooled, 16 emb.)",
    "siamese_sl_allmod": "SLeuth-all-CL (cell-line, 16 emb.)",
    "siamese_sl_nocell": "SLeuth-CL, cell_emb=0",
    "SLMGAE": "SLMGAE", "SLGNN": "SLGNN", "NSF4SL": "NSF4SL", "ELISL": "ELISL",
    "BilinearProbe": "Bilinear probe", "DegreePrior": "Degree prior (control)",
}
ALLMOD_PARTNER = {"siamese_sl_pd_pooled": "siamese_sl_allmod_pd_pooled",
                  "siamese_sl": "siamese_sl_allmod"}

checks = []


def note(msg):
    checks.append(msg)
    print(msg)


def is_siamese(m):
    return m.startswith("siamese_sl")


# ----------------------------------------------------------------------------- loading
def load_row(arm, model, cv):
    base = CMP[arm] if is_siamese(model) else SHARED
    p = base / f"{model}_{cv}.json"
    s = json.load(open(p))["summary"]
    folds = s["folds"]
    assert len(folds) == 5 and s["n_folds_scored"] == 5 and not s["partial"], p
    return {
        "path": str(p.relative_to(REPO)), "summary": s,
        "f": {m: np.array([f[m] for f in folds], float) for m in METRICS + ["aupr", "baseline"]},
    }


DATA = {}
for arm in CMP:
    for cv in CVS:
        for m in ROWS[arm] + ["siamese_sl_nocell"]:
            DATA[(arm, m, cv)] = load_row(arm, m, cv)

# folds are paired by POSITION in summary.folds; the per-fold test prevalence
# (baseline) must be identical across every row of a (arm, cv)
for arm in CMP:
    for cv in CVS:
        ref = DATA[(arm, "SLMGAE", cv)]["f"]["baseline"]
        for m in ROWS[arm] + ["siamese_sl_nocell"]:
            b = DATA[(arm, m, cv)]["f"]["baseline"]
            assert np.allclose(b, ref, atol=1e-12, rtol=0), (arm, m, cv, b, ref)
note("CHECK fold alignment: per-fold test prevalence identical across all rows of every (arm, tier): OK")

geom = json.load(open(SHARED / "paper_tables/numbers.json"))["items"]["fold_geometry"]["value"]
RHO = {}
NTEST = {}
for cv in CVS:
    pf = geom[cv]["per_fold"]
    ntr = np.array([f["n_train_pos"] + f["n_train_neg"] + f["n_val_pos"] + f["n_val_neg"] for f in pf], float)
    nte = np.array([f["n_test"] for f in pf], float)
    RHO[cv] = float(np.mean(nte / ntr))
    NTEST[cv] = (int(nte.min()), int(nte.max()), [int(f["n_test_pos"]) for f in pf], [float(f["test_prevalence"]) for f in pf])
note("Nadeau-Bengio rho = mean n_test/(n_train+n_val) per tier (numbers.json items.fold_geometry): "
     + ", ".join(f"{TIER[cv]} {RHO[cv]:.3f}" for cv in CVS))


# ----------------------------------------------------------------------------- stats helpers
def p_two_sided_boot(p_le0):
    if p_le0 is None or (isinstance(p_le0, float) and math.isnan(p_le0)):
        return float("nan")
    return min(1.0, 2.0 * max(min(p_le0, 1.0 - p_le0), 1.0 / (B + 1)))


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx], kind="mergesort")]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out


SIGNS = np.array(list(itertools.product([1.0, -1.0], repeat=5)))


def fold_paired(a, b, cv):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    mean = float(d.mean())
    sd1 = float(d.std(ddof=1))
    if sd1 > 0:
        t = mean / (sd1 / math.sqrt(n))
        p_t = float(2 * stats.t.sf(abs(t), df=n - 1))
        t_nb = mean / math.sqrt((1.0 / n + RHO[cv]) * sd1 ** 2)
        p_nb = float(2 * stats.t.sf(abs(t_nb), df=n - 1))
    else:
        t = p_t = t_nb = p_nb = float("nan")
    perm = np.abs((SIGNS * d).mean(axis=1))
    p_sf = float(np.mean(perm >= abs(mean) - 1e-12))
    half = float(stats.t.ppf(0.975, df=n - 1) * sd1 / math.sqrt(n))
    return {"diff_mean": mean, "diff_sd_pop": float(d.std(ddof=0)), "diff_sd_sample": sd1,
            "diff_min": float(d.min()), "diff_max": float(d.max()),
            "ci_t_low": mean - half, "ci_t_high": mean + half,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n_folds": n,
            "t": t, "p_t": p_t, "p_signflip": p_sf, "t_nb": t_nb, "p_nb": p_nb,
            "per_fold_diff": ";".join(f"{x:+.4f}" for x in d)}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        return f"{0:.{nd}f}"
    return s.replace("-", "\u2212") if x < 0 else s


def fmt_p(p, m=1):
    """bootstrap p (two-sided); m = Holm family size when p is Holm-adjusted. A raw p at the
    floor 2/(B+1) is a bound, so its Holm value is printed as a bound too (<= m x floor)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p <= m * P_FLOOR + 1e-12:
        return f"\u2264 {m * P_FLOOR:.3f}"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def fmt_pt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if abs(p - 0.0625) < 1e-12:
        return "0.0625"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def fmt_int(x):
    return f"{int(round(x)):,}"


# ----------------------------------------------------------------------------- Table 1
def elisl_params(cv):
    s = json.load(open(REPO / f"ELISL/results/elisl_{cv}/results.json"))["summary"]
    v = np.array([f["trainable_params"] for f in s["folds"]], float)
    return float(v.mean()), float(v.std(ddof=0)), int(v.min()), int(v.max())


def params_of(arm, m, cv):
    s = DATA[(arm, m, cv)]["summary"]
    if is_siamese(m) or m == "BilinearProbe":
        return s["trainable_params_fold_mean"], s["trainable_params_fold_std"], "fold mean (sd)"
    if m == "ELISL":
        mu, sd, lo, hi = elisl_params(cv)
        return mu, sd, f"LightGBM fitted values, fold mean (sd), range {lo:,}-{hi:,}"
    return float(s["trainable_params"]), 0.0, "count (all folds)"


ci_rows, diff_rows = {}, {}
for arm in CMP:
    ci = pd.read_csv(CMP[arm] / "bootstrap/ci_tiers.csv", dtype={"fold": str})
    ci = ci[ci["fold"] == "mean"]
    ci_rows[arm] = ci.set_index(["model", "cv", "metric"])
    df = pd.read_csv(CMP[arm] / "bootstrap/ci_tiers_diffs.csv", dtype={"fold": str})
    df = df[df["fold"] == "mean"]
    diff_rows[arm] = df

T1 = {}
for arm in CMP:
    tt = pd.read_csv(CMP[arm] / "paper_tables/tab_tiers.csv").set_index(["model", "cv"])
    recs = []
    for cv in CVS:
        for m in ROWS[arm]:
            d = DATA[(arm, m, cv)]
            pm, psd, pkind = params_of(arm, m, cv)
            rec = {"pca_arm": arm, "tier": TIER[cv], "cv": cv, "model": m, "display": DISPLAY[m],
                   "params_mean": pm, "params_sd": psd, "params_kind": pkind, "row_json": d["path"]}
            for k in METRICS:
                v = d["f"][k]
                mu, sd = float(v.mean()), float(v.std(ddof=0))
                # cross-checks: tab_tiers.csv and the row JSON's own mean/std
                assert abs(mu - tt.loc[(m, cv), f"{k}_mean"]) < 1e-9, (arm, m, cv, k)
                assert abs(sd - tt.loc[(m, cv), f"{k}_sd"]) < 1e-9, (arm, m, cv, k)
                assert abs(mu - d["summary"][f"{k}_mean"]) < 1e-9
                assert abs(sd - d["summary"][f"{k}_std"]) < 1e-9
                c = ci_rows[arm].loc[(m, cv, k)]
                assert abs(c["point"] - mu) < 1e-9, (arm, m, cv, k, c["point"], mu)
                rec.update({f"{k}_mean": mu, f"{k}_sd": sd, f"{k}_sd_sample": float(v.std(ddof=1)),
                            f"{k}_ci_low": float(c["ci_low"]), f"{k}_ci_high": float(c["ci_high"]),
                            f"{k}_folds": ";".join(f"{x:.4f}" for x in v)})
            recs.append(rec)
    T1[arm] = pd.DataFrame(recs)
    T1[arm].to_csv(OUT / f"table1_{arm}.csv", index=False)
note("CHECK Table 1: every mean/sd equals tab_tiers.csv and the row JSON (1e-9); every CI point equals the mean")

# best per tier x metric among the rows a paper table would compare (all but the control)
def table1_md(arm):
    t = T1[arm]
    lines = []
    hdr = "| Tier | Model | Trainable params | AUROC | AUPR (AP) | AUPRG |"
    lines += [hdr, "|---|---|---:|---|---|---|"]
    for cv in CVS:
        sub = t[t.cv == cv]
        best = {}
        for k in METRICS:
            cand = sub[sub.model != "DegreePrior"]
            best[k] = round(cand[f"{k}_mean"].max(), 3)
        for _, r in sub.iterrows():
            cells = []
            for k in METRICS:
                s = f"{fmt(r[f'{k}_mean'])} ({fmt(r[f'{k}_sd'])}) [{fmt(r[f'{k}_ci_low'])}, {fmt(r[f'{k}_ci_high'])}]"
                if r.model != "DegreePrior" and round(r[f"{k}_mean"], 3) == best[k]:
                    s = "**" + s + "**"
                cells.append(s)
            if r.model == "DegreePrior":
                par = "0"
            elif r.params_sd and r.params_sd > 0:
                par = f"{fmt_int(r.params_mean)} ({fmt_int(r.params_sd)})"
            else:
                par = fmt_int(r.params_mean)
            lines.append(f"| {TIER[cv]} | {r.display} | {par} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


for arm in CMP:
    (OUT / f"table1_{arm}.md").write_text(table1_md(arm) + "\n")


# ----------------------------------------------------------------------------- bootstrap significance
FAMILY_BOOT = ("bootstrap families: (a) one SLeuth row (pooled, cell-line or cell_emb=0) x one tier x one "
               "metric over its non-SLeuth peers (6 in arm none: SLMGAE, SLGNN, NSF4SL, ELISL, "
               "BilinearProbe, DegreePrior; 5 in arm joint); (b) selected-vs-all-embedding: one tier x "
               "metric over the 2 variants")
boot_out = []
for arm in CMP:
    df = diff_rows[arm]
    df = df[df.metric.isin(METRICS)].copy()
    df["p2"] = df["p_boot_le0"].apply(p_two_sided_boot)
    df["p2_at_floor"] = df["p2"] <= P_FLOOR + 1e-12
    df["family"] = np.where(df.peer.isin(ALLMOD), "selected_vs_allmod", "vs_peers:" + df.model)
    df["p2_holm"] = np.nan
    for (fam, cv, k), g in df.groupby(["family", "cv", "metric"]):
        df.loc[g.index, "p2_holm"] = holm(g["p2"].values)
        df.loc[g.index, "family_m"] = len(g)
    df["family_m"] = df["family_m"].astype(int)
    df["pca_arm"] = arm
    df["tier"] = df.cv.map(TIER)
    boot_out.append(df)
BOOT = pd.concat(boot_out, ignore_index=True)
BOOT[["pca_arm", "family", "family_m", "model", "peer", "tier", "cv", "metric", "point", "ci_low", "ci_high",
      "boot_se", "p_boot_le0", "p2", "p2_at_floor", "p2_holm", "n_boot", "seed"]].to_csv(OUT / "sig_bootstrap.csv", index=False)


def bp(b):
    """'p (Holm)' for one bootstrap row, bounds at the floor."""
    raw = fmt_p(b.p2)
    adj = fmt_p(b.p2_holm, int(b.family_m)) if b.p2_at_floor else fmt_p(b.p2_holm)
    return f"{raw} ({adj})"
note("bootstrap diffs read from bootstrap/ci_tiers_diffs.csv fold=='mean'; " + FAMILY_BOOT)


# ----------------------------------------------------------------------------- fold-paired significance
FAMILY_FOLD = ("fold-paired families: (a) one SLeuth row (pooled, cell-line, all-emb pooled, all-emb "
               "cell-line) x tier x metric over its non-SLeuth peers (6 / 5); (b) pooled-vs-cell-line: "
               "tier x metric over {selected, all-embedding} (2); (c) selected-vs-all-embedding: tier x "
               "metric over {pooled, cell-line} (2); (d) Table 4: tier x metric over {cell_emb=0, "
               "retrained pooled} vs included (2); (e) Table 4 self-scored diagnostics: tier x metric over "
               "{+bias=0, +head_proj_bias=mean, +head_proj_bias=0} vs as-trained (3)")


def run_family(arm, fam, pairs):
    """pairs: list of (A_label, A_vec_getter, B_label, B_vec_getter); Holm within (cv, metric)."""
    out = []
    for cv in CVS:
        for k in METRICS:
            recs = []
            for a, fa, b, fb in pairs:
                r = fold_paired(fa(cv, k), fb(cv, k), cv)
                r.update({"pca_arm": arm, "family": fam, "tier": TIER[cv], "cv": cv, "metric": k,
                          "A": a, "B": b})
                recs.append(r)
            pt = holm([r["p_t"] for r in recs])
            ps = holm([r["p_signflip"] for r in recs])
            pn = holm([r["p_nb"] for r in recs])
            for r, x, y, z in zip(recs, pt, ps, pn):
                r["p_t_holm"], r["p_signflip_holm"], r["p_nb_holm"], r["family_m"] = x, y, z, len(recs)
            out += recs
    return out


def getter(arm, m):
    return lambda cv, k: DATA[(arm, m, cv)]["f"][k]


FOLD = []
for arm in CMP:
    for a in SLEUTH + ALLMOD:
        FOLD += run_family(arm, f"vs_peers:{a}", [(a, getter(arm, a), p, getter(arm, p)) for p in PEERS[arm]])
    FOLD += run_family(arm, "pooled_vs_cellline", [
        ("siamese_sl_pd_pooled", getter(arm, "siamese_sl_pd_pooled"), "siamese_sl", getter(arm, "siamese_sl")),
        ("siamese_sl_allmod_pd_pooled", getter(arm, "siamese_sl_allmod_pd_pooled"),
         "siamese_sl_allmod", getter(arm, "siamese_sl_allmod"))])
    FOLD += run_family(arm, "selected_vs_allmod", [
        ("siamese_sl_pd_pooled", getter(arm, "siamese_sl_pd_pooled"),
         "siamese_sl_allmod_pd_pooled", getter(arm, "siamese_sl_allmod_pd_pooled")),
        ("siamese_sl", getter(arm, "siamese_sl"), "siamese_sl_allmod", getter(arm, "siamese_sl_allmod"))])
    FOLD += run_family(arm, "table4_main", [
        ("siamese_sl_nocell", getter(arm, "siamese_sl_nocell"), "siamese_sl", getter(arm, "siamese_sl")),
        ("siamese_sl_pd_pooled", getter(arm, "siamese_sl_pd_pooled"), "siamese_sl", getter(arm, "siamese_sl"))])

# self-scored conditions (zero_cell_emb.py), per fold by the 'fold' key
SELF = {}
for arm in CMP:
    for cv in CVS:
        r = json.load(open(SIA[arm] / f"cellline_ablation_{cv}/results.json"))
        assert r["status"] == "ok" and r["self_check"]["ok"], (arm, cv)
        for cond, lst in r["per_fold"].items():
            lst = sorted(lst, key=lambda x: x["fold"])
            assert [x["fold"] for x in lst] == [0, 1, 2, 3, 4]
            SELF[(arm, cv, cond)] = {k: np.array([x[k] for x in lst], float) for k in METRICS}
        # self-scored as-trained must equal the graded siamese_sl row, cell_emb=0 the graded nocell row
        for k in METRICS:
            dg = np.abs(SELF[(arm, cv, "as-trained")][k] - DATA[(arm, "siamese_sl", cv)]["f"][k]).max()
            dn = np.abs(SELF[(arm, cv, "cell_emb=0")][k] - DATA[(arm, "siamese_sl_nocell", cv)]["f"][k]).max()
            assert dg < 1e-5 and dn < 1e-5, (arm, cv, k, dg, dn)
note("CHECK self-scored as-trained == graded siamese_sl and self-scored cell_emb=0 == graded "
     "siamese_sl_nocell, per fold, |diff| < 1e-5 (both arms, all tiers)")
SELF_CONDS = ["cell_emb=0 +bias=0", "cell_emb=0 +head_proj_bias=mean", "cell_emb=0 +head_proj_bias=0"]
for arm in CMP:
    FOLD += run_family(arm, "table4_selfscored", [
        (c, (lambda c: lambda cv, k: SELF[(arm, cv, c)][k])(c), "as-trained",
         lambda cv, k: SELF[(arm, cv, "as-trained")][k]) for c in SELF_CONDS])

FOLDDF = pd.DataFrame(FOLD)
cols = ["pca_arm", "family", "family_m", "A", "B", "tier", "cv", "metric", "diff_mean", "diff_sd_pop",
        "diff_sd_sample", "ci_t_low", "ci_t_high", "diff_min", "diff_max", "wins", "losses", "n_folds", "t", "p_t", "p_t_holm",
        "p_signflip", "p_signflip_holm", "t_nb", "p_nb", "p_nb_holm", "per_fold_diff"]
FOLDDF[cols].to_csv(OUT / "sig_foldpaired.csv", index=False)
note(FAMILY_FOLD)


# ----------------------------------------------------------------------------- markdown: significance tables
def sig_table(arm, a, metric_list=METRICS):
    """SLeuth row a vs every peer: bootstrap (if present) + fold-paired, one block per tier."""
    lines = ["| Tier | vs | Metric | Δ (A−B) | 95% boot CI of Δ | p_boot (Holm) | split-level 95% t-CI of Δ | folds A>B | paired t p (Holm) | sign-flip p |",
             "|---|---|---|---:|---|---|---|---|---|---|"]
    fd = FOLDDF[(FOLDDF.pca_arm == arm) & (FOLDDF.family == f"vs_peers:{a}")]
    bd = BOOT[(BOOT.pca_arm == arm) & (BOOT.model == a)]
    for cv in CVS:
        for p in PEERS[arm]:
            for k in metric_list:
                f = fd[(fd.cv == cv) & (fd.B == p) & (fd.metric == k)].iloc[0]
                b = bd[(bd.cv == cv) & (bd.peer == p) & (bd.metric == k)]
                if len(b):
                    b = b.iloc[0]
                    assert abs(b["point"] - f["diff_mean"]) < 1e-9
                    ci = f"[{fmt(b.ci_low)}, {fmt(b.ci_high)}]"
                    pb = bp(b)
                else:
                    ci, pb = "not bootstrapped", "n/a"
                lines.append(f"| {TIER[cv]} | {DISPLAY[p]} | {MLABEL[k]} | {fmt(f.diff_mean)} | {ci} | {pb} | "
                             f"[{fmt(f.ci_t_low)}, {fmt(f.ci_t_high)}] | "
                             f"{f.wins}/5 | {fmt_pt(f.p_t)} ({fmt_pt(f.p_t_holm)}) | {fmt_pt(f.p_signflip)} |")
    return "\n".join(lines)


for arm in CMP:
    for a in SLEUTH + ALLMOD:
        (OUT / f"sig_{arm}_{a}.md").write_text(sig_table(arm, a) + "\n")


def compact_matrix(arm, a):
    """One cell per tier x peer x metric: Δ, then markers. * = bootstrap Holm p<0.05 (or 'nb' if none);
    † = fold-paired t Holm p<0.05; wins."""
    fd = FOLDDF[(FOLDDF.pca_arm == arm) & (FOLDDF.family == f"vs_peers:{a}")]
    bd = BOOT[(BOOT.pca_arm == arm) & (BOOT.model == a)]
    hdr = "| Tier | Metric | " + " | ".join(DISPLAY[p] for p in PEERS[arm]) + " |"
    lines = [hdr, "|---|---|" + "---|" * len(PEERS[arm])]
    for cv in CVS:
        for k in METRICS:
            cells = []
            for p in PEERS[arm]:
                f = fd[(fd.cv == cv) & (fd.B == p) & (fd.metric == k)].iloc[0]
                b = bd[(bd.cv == cv) & (bd.peer == p) & (bd.metric == k)]
                mk = ""
                if len(b):
                    mk += "*" if b.iloc[0].p2_holm < 0.05 else ""
                mk += "†" if f.p_t_holm < 0.05 else ""
                cells.append(f"{fmt(f.diff_mean)}{mk} ({f.wins}/5)")
            lines.append(f"| {TIER[cv]} | {MLABEL[k]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


for arm in CMP:
    for a in SLEUTH + ALLMOD:
        (OUT / f"matrix_{arm}_{a}.md").write_text(compact_matrix(arm, a) + "\n")


def pair_table(arm, fam, boot_family=None):
    fd = FOLDDF[(FOLDDF.pca_arm == arm) & (FOLDDF.family == fam)]
    lines = ["| Tier | A − B | Metric | mean A | mean B | Δ | split-level 95% t-CI of Δ | folds A>B | paired t p (Holm) | sign-flip p | boot CI of Δ | p_boot (Holm) |",
             "|---|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for cv in CVS:
        for _, f in fd[fd.cv == cv].iterrows():
            ma = mean_of(arm, f.A, cv, f.metric)
            mb = mean_of(arm, f.B, cv, f.metric)
            ci, pb = "no bootstrap pair", "n/a"
            if boot_family is not None:
                b = BOOT[(BOOT.pca_arm == arm) & (BOOT.family == boot_family) & (BOOT.cv == cv) &
                         (BOOT.metric == f.metric) & (BOOT.model == f.A) & (BOOT.peer == f.B)]
                if len(b):
                    b = b.iloc[0]
                    ci = f"[{fmt(b.ci_low)}, {fmt(b.ci_high)}]"
                    pb = bp(b)
            lines.append(f"| {TIER[cv]} | {DISPLAY.get(f.A, f.A)} − {DISPLAY.get(f.B, f.B)} | {MLABEL[f.metric]} | "
                         f"{fmt(ma)} | {fmt(mb)} | {fmt(f.diff_mean)} | [{fmt(f.ci_t_low)}, {fmt(f.ci_t_high)}] | "
                         f"{f.wins}/5 | {fmt_pt(f.p_t)} ({fmt_pt(f.p_t_holm)}) | "
                         f"{fmt_pt(f.p_signflip)} | {ci} | {pb} |")
    return "\n".join(lines)


def mean_of(arm, m, cv, k):
    if (arm, m, cv) in DATA:
        return float(DATA[(arm, m, cv)]["f"][k].mean())
    return float(SELF[(arm, cv, m)][k].mean())


for arm in CMP:
    (OUT / f"pooled_vs_cellline_{arm}.md").write_text(pair_table(arm, "pooled_vs_cellline") + "\n")
    (OUT / f"selected_vs_allmod_{arm}.md").write_text(pair_table(arm, "selected_vs_allmod", "selected_vs_allmod") + "\n")
    (OUT / f"table4_tests_{arm}.md").write_text(pair_table(arm, "table4_main") + "\n")
    (OUT / f"table4_selfscored_{arm}.md").write_text(pair_table(arm, "table4_selfscored") + "\n")


# ----------------------------------------------------------------------------- Table 4 (cell lines)
def table4(arm):
    tc = pd.read_csv(CMP[arm] / "paper_tables/tab_cellline.csv")
    recs = []
    conds = [("included", "siamese_sl", "graded"), ("removed (cell_emb=0)", "siamese_sl_nocell", "graded"),
             ("retrained without cell lines (pooled)", "siamese_sl_pd_pooled", "graded")]
    conds += [(f"self-scored: {c}", c, "self") for c in SELF_CONDS]
    for cv in CVS:
        inc = {k: DATA[(arm, "siamese_sl", cv)]["f"][k] for k in METRICS}
        for label, key, kind in conds:
            v = ({k: DATA[(arm, key, cv)]["f"][k] for k in METRICS} if kind == "graded"
                 else {k: SELF[(arm, cv, key)][k] for k in METRICS})
            ref = inc if kind == "graded" else {k: SELF[(arm, cv, "as-trained")][k] for k in METRICS}
            rec = {"pca_arm": arm, "tier": TIER[cv], "cv": cv, "condition": label, "source": key, "kind": kind}
            for k in METRICS:
                rec[f"{k}_mean"] = float(v[k].mean())
                rec[f"{k}_sd"] = float(v[k].std(ddof=0))
                rec[f"delta_{k}"] = float(v[k].mean() - ref[k].mean())
            # cross-check against tab_cellline.csv
            mrow = tc[(tc.cv == cv) & (tc.condition.fillna("").str.startswith(
                {"included": "included", "removed (cell_emb=0)": "removed",
                 "retrained without cell lines (pooled)": "retrained"}.get(label, "self-scored: " + key)))]
            if kind == "self":
                mrow = tc[(tc.cv == cv) & (tc.condition == f"self-scored: {key}")]
            assert len(mrow) == 1, (arm, cv, label, len(mrow))
            for k in METRICS:
                assert abs(mrow.iloc[0][f"{k}_mean"] - rec[f"{k}_mean"]) < 1e-9, (arm, cv, label, k)
                assert abs(mrow.iloc[0][f"delta_{k}_vs_included"] - rec[f"delta_{k}"]) < 1e-9, (arm, cv, label, k)
            rec["residual_head_terms"] = mrow.iloc[0].get("residual_head_terms")
            recs.append(rec)
    return pd.DataFrame(recs)


T4 = {arm: table4(arm) for arm in CMP}
for arm in CMP:
    T4[arm].to_csv(OUT / f"table4_{arm}.csv", index=False)
note("CHECK Table 4: every mean and delta equals tab_cellline.csv (1e-9)")


def table4_md(arm):
    t = T4[arm]
    fd = FOLDDF[(FOLDDF.pca_arm == arm) & (FOLDDF.family.isin(["table4_main", "table4_selfscored"]))]
    lines = ["| Tier | Cell-line information | AUROC | AUPR (AP) | AUPRG | Δ AUROC | Δ AP | Δ AUPRG |",
             "|---|---|---|---|---|---|---|---|"]
    for cv in CVS:
        for _, r in t[t.cv == cv].iterrows():
            cells = [f"{fmt(r[f'{k}_mean'])} ({fmt(r[f'{k}_sd'])})" for k in METRICS]
            if r.condition == "included":
                dcells = ["—"] * 3
            else:
                dcells = []
                for k in METRICS:
                    f = fd[(fd.cv == cv) & (fd.A == r.source) & (fd.metric == k)].iloc[0]
                    star = "†" if f.p_t_holm < 0.05 else ""
                    dcells.append(f"{'+' if r[f'delta_{k}'] >= 0 else ''}{fmt(r[f'delta_{k}'])}{star} "
                                  f"({f.wins}/5 up; p_t {fmt_pt(f.p_t)})")
            lines.append(f"| {TIER[cv]} | {r.condition} | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")
    return "\n".join(lines)


for arm in CMP:
    (OUT / f"table4_{arm}.md").write_text(table4_md(arm) + "\n")


# nocell vs peers (the only bootstrap pairs that involve the ablated model)
def nocell_boot_md(arm):
    b = BOOT[(BOOT.pca_arm == arm) & (BOOT.model == "siamese_sl_nocell")]
    b2 = BOOT[(BOOT.pca_arm == arm) & (BOOT.model == "siamese_sl")]
    lines = ["| Tier | Peer | Metric | Δ cell_emb=0 − peer [CI] p_boot (Holm) | Δ included − peer [CI] p_boot (Holm) |",
             "|---|---|---|---|---|"]
    for cv in CVS:
        for p in PEERS[arm]:
            for k in METRICS:
                x = b[(b.cv == cv) & (b.peer == p) & (b.metric == k)].iloc[0]
                y = b2[(b2.cv == cv) & (b2.peer == p) & (b2.metric == k)].iloc[0]
                lines.append(f"| {TIER[cv]} | {DISPLAY[p]} | {MLABEL[k]} | {fmt(x.point)} [{fmt(x.ci_low)}, {fmt(x.ci_high)}] "
                             f"{bp(x)} | {fmt(y.point)} [{fmt(y.ci_low)}, {fmt(y.ci_high)}] {bp(y)} |")
    return "\n".join(lines)


for arm in CMP:
    (OUT / f"nocell_boot_{arm}.md").write_text(nocell_boot_md(arm) + "\n")


# ----------------------------------------------------------------------------- submitted paper Table 1 / Table 4
tex = (REPO / "paper/main.tex").read_text().splitlines()
start = next(i for i, l in enumerate(tex) if l.startswith("\\begin{table}") and i > 500)
sub = []
tier = None
num = r"\$?(\\textbf\{)?(\$-\$|−|-)?([0-9.]+)\}?\$? \(([0-9.]+)\)"
for i in range(start, start + 40):
    l = tex[i]
    if "\\label{tab:tiers}" in l:
        break
    m = re.match(r"\\multirow\{4\}\{\*\}\{(I+)\}", l.strip())
    if m:
        tier = m.group(1)
        continue
    m = re.match(r"\s*&\s*(SLeuth|SLMGAE|SLGNN|NSF4SL)\s*&\s*([^&]+)&(.*)\\\\", l)
    if m and tier:
        vals = re.findall(num, m.group(3))
        assert len(vals) == 3, (i, l, vals)
        rec = {"tier": tier, "model": m.group(1), "params_paper": m.group(2).strip(), "line": i + 1}
        for k, (_, sign, v, sd) in zip(METRICS, vals):
            rec[f"{k}_paper"] = (-1 if sign else 1) * float(v)
            rec[f"{k}_sd_paper"] = float(sd)
        sub.append(rec)
PAPER_T1 = pd.DataFrame(sub)
assert len(PAPER_T1) == 12, PAPER_T1
note(f"parsed submitted Table 1 (paper/main.tex lines {PAPER_T1.line.min()}-{PAPER_T1.line.max()}): 12 rows")

cmp_recs = []
# the submitted SLeuth = cell-line heads + joint PCA (0.8) on a globally selected 6-embedding set; its closest
# analog in this run is arm joint's siamese_sl (same heads and PCA, per-fold selection) -> listed too
MAP_NEW = {"SLeuth": [("none", "siamese_sl"), ("none", "siamese_sl_pd_pooled"), ("joint", "siamese_sl")],
           "SLMGAE": [("none", "SLMGAE")], "SLGNN": [("none", "SLGNN")], "NSF4SL": [("none", "NSF4SL")]}
for _, r in PAPER_T1.iterrows():
    cv = {v: k for k, v in TIER.items()}[r.tier]
    for arm_new, newm in MAP_NEW[r.model]:
        d = DATA[(arm_new, newm, cv)]
        rec = {"tier": r.tier, "paper_row": r.model, "paper_line": r.line, "paper_params": r.params_paper,
               "new_arm": arm_new, "new_row": newm, "new_params": params_of(arm_new, newm, cv)[0]}
        for k in METRICS:
            nm = float(d["f"][k].mean())
            rec[f"{k}_paper"] = r[f"{k}_paper"]
            rec[f"{k}_sd_paper"] = r[f"{k}_sd_paper"]
            rec[f"{k}_new"] = nm
            rec[f"{k}_sd_new"] = float(d["f"][k].std(ddof=0))
            rec[f"{k}_change"] = nm - r[f"{k}_paper"]
        cmp_recs.append(rec)
CMPDF = pd.DataFrame(cmp_recs)
CMPDF.to_csv(OUT / "baselines_vs_submitted.csv", index=False)


def cmp_md():
    lines = ["| Tier | Row (submitted → revision) | Params submitted → revision | AUROC submitted → revision (Δ) | AUPR submitted → AP revision (Δ) | AUPRG submitted → revision (Δ) |",
             "|---|---|---|---|---|---|"]
    for _, r in CMPDF.iterrows():
        cells = [f"{fmt(r[f'{k}_paper'])} ({fmt(r[f'{k}_sd_paper'])}) → {fmt(r[f'{k}_new'])} ({fmt(r[f'{k}_sd_new'])}) "
                 f"({'+' if r[f'{k}_change'] >= 0 else ''}{fmt(r[f'{k}_change'])})" for k in METRICS]
        armtag = "" if r.new_arm == "none" else f" [arm {r.new_arm}]"
        lines.append(f"| {r.tier} | {r.paper_row} → {DISPLAY[r.new_row]}{armtag} | {r.paper_params} → {fmt_int(r.new_params)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


(OUT / "baselines_vs_submitted.md").write_text(cmp_md() + "\n")

# baseline inputs, straight from the row JSONs
inp = []
for m in ["SLGNN", "NSF4SL", "SLMGAE", "ELISL", "BilinearProbe"]:
    for cv in CVS:
        s = DATA[("none", m, cv)]["summary"]
        inp.append({"model": m, "cv": cv, "model_class": s.get("model_class"), "arch": s.get("arch"),
                    "kg_stamp": json.dumps(s.get("kg_stamp")) if s.get("kg_stamp") else None,
                    "features": s.get("features"), "feature_file": s.get("feature_file"),
                    "standardized": s.get("standardized"), "support_views": json.dumps(s.get("support_views")) if s.get("support_views") else None,
                    "elisl_sources": s.get("elisl_sources"), "context_free": s.get("context_free"),
                    "probe_source_combo": (s.get("probe_source") or {}).get("combo_dir"),
                    "trainable_params": s.get("trainable_params"), "total_params": s.get("total_params")})
pd.DataFrame(inp).to_csv(OUT / "baseline_inputs.csv", index=False)

# ----------------------------------------------------------------------------- 84QL Tier III AUROC focus
foc = []
for a in SLEUTH + ALLMOD:
    f = FOLDDF[(FOLDDF.pca_arm == "none") & (FOLDDF.family == f"vs_peers:{a}") & (FOLDDF.cv == "cv3") &
               (FOLDDF.B == "SLMGAE")]
    for _, r in f.iterrows():
        b = BOOT[(BOOT.pca_arm == "none") & (BOOT.model == a) & (BOOT.peer == "SLMGAE") & (BOOT.cv == "cv3") &
                 (BOOT.metric == r.metric)]
        foc.append({"A": a, "metric": r.metric, "mean_A": mean_of("none", a, "cv3", r.metric),
                    "mean_SLMGAE": mean_of("none", "SLMGAE", "cv3", r.metric), "diff": r.diff_mean,
                    "per_fold_diff": r.per_fold_diff, "wins": r.wins, "p_t": r.p_t, "p_t_holm": r.p_t_holm,
                    "p_signflip": r.p_signflip, "p_nb": r.p_nb,
                    "boot_ci": (f"[{b.iloc[0].ci_low:.4f}, {b.iloc[0].ci_high:.4f}]" if len(b) else None),
                    "p_boot2": (b.iloc[0].p2 if len(b) else None), "p_boot2_holm": (b.iloc[0].p2_holm if len(b) else None)})
pd.DataFrame(foc).to_csv(OUT / "q84QL_tier3_vs_SLMGAE.csv", index=False)

# per-fold Tier III AUROC for the three rows 84QL is about
pf = {m: DATA[("none", m, "cv3")]["f"]["auroc"] for m in ["siamese_sl_pd_pooled", "siamese_sl", "SLMGAE"]}
pd.DataFrame({"fold": range(5), "test_prevalence": DATA[("none", "SLMGAE", "cv3")]["f"]["baseline"],
              **{f"auroc_{m}": v for m, v in pf.items()}}).to_csv(OUT / "q84QL_tier3_auroc_per_fold.csv", index=False)

# ----------------------------------------------------------------------------- summary counts per row x tier
cnt = []
for arm in CMP:
    for a in SLEUTH + ALLMOD:
        for cv in CVS:
            for peerset, peers in [("published peers (SLMGAE, SLGNN, NSF4SL, ELISL)", ["SLMGAE", "SLGNN", "NSF4SL", "ELISL"]),
                                   ("all non-SLeuth rows", PEERS[arm])]:
                f = FOLDDF[(FOLDDF.pca_arm == arm) & (FOLDDF.family == f"vs_peers:{a}") & (FOLDDF.cv == cv) & FOLDDF.B.isin(peers)]
                b = BOOT[(BOOT.pca_arm == arm) & (BOOT.model == a) & (BOOT.cv == cv) & BOOT.peer.isin(peers)]
                cnt.append({"pca_arm": arm, "A": a, "tier": TIER[cv], "peers": peerset, "n_comparisons": len(f),
                            "n_delta_pos": int((f.diff_mean > 0).sum()),
                            "n_win_5of5": int((f.wins == 5).sum()), "n_lose_5of5": int((f.losses == 5).sum()),
                            "n_t_holm_sig_pos": int(((f.p_t_holm < 0.05) & (f.diff_mean > 0)).sum()),
                            "n_t_holm_sig_neg": int(((f.p_t_holm < 0.05) & (f.diff_mean < 0)).sum()),
                            "n_boot": len(b),
                            "n_boot_holm_sig_pos": int(((b.p2_holm < 0.05) & (b.point > 0)).sum()) if len(b) else None,
                            "n_boot_holm_sig_neg": int(((b.p2_holm < 0.05) & (b.point < 0)).sum()) if len(b) else None,
                            "n_nb_holm_sig_pos": int(((f.p_nb_holm < 0.05) & (f.diff_mean > 0)).sum())})
CNT = pd.DataFrame(cnt)
CNT.to_csv(OUT / "summary_counts.csv", index=False)


# ----------------------------------------------------------------------------- LaTeX drafts (paste-ready)
def tex_num(x):
    s = fmt(x)
    return s.replace("−", "$-$")


def table1_tex(arm, with_ci):
    t = T1[arm]
    rows_per_tier = len(ROWS[arm])
    out = ["\\begin{tabular}{ll|cccc}", "\\hline",
           "Tier & Model & Trainable parameters & AUROC & AUPR (AP) & AUPRG \\\\", "\\hline"]
    for cv in CVS:
        sub = t[t.cv == cv]
        cand = sub[sub.model != "DegreePrior"]
        best = {k: round(cand[f"{k}_mean"].max(), 3) for k in METRICS}
        first = True
        for _, r in sub.iterrows():
            cells = []
            for k in METRICS:
                s = f"{tex_num(r[f'{k}_mean'])}"
                if r.model != "DegreePrior" and round(r[f"{k}_mean"], 3) == best[k]:
                    s = f"\\textbf{{{s}}}"
                s += f" ({tex_num(r[f'{k}_sd'])})"
                if with_ci:
                    s += f" {{\\scriptsize [{tex_num(r[f'{k}_ci_low'])}, {tex_num(r[f'{k}_ci_high'])}]}}"
                cells.append(s)
            if r.model == "DegreePrior":
                par = "0"
            elif r.params_sd and r.params_sd > 0:
                par = f"{fmt_int(r.params_mean).replace(',', '{,}')} ({fmt_int(r.params_sd).replace(',', '{,}')})"
            else:
                par = fmt_int(r.params_mean).replace(",", "{,}")
            tier_cell = f"\\multirow{{{rows_per_tier}}}{{*}}{{{TIER[cv]}}}" if first else ""
            first = False
            out.append(f"{tier_cell} & {r.display} & {par} & " + " & ".join(cells) + " \\\\")
        out.append("\\hline")
    out.append("\\end{tabular}")
    return "\n".join(out)


for arm in CMP:
    (OUT / f"table1_{arm}.tex").write_text(table1_tex(arm, with_ci=True) + "\n")
(OUT / "table1_none_noci.tex").write_text(table1_tex("none", with_ci=False) + "\n")


def table4_tex(arm):
    t = T4[arm]
    t = t[t.kind == "graded"]
    fd = FOLDDF[(FOLDDF.pca_arm == arm) & (FOLDDF.family == "table4_main")]
    out = ["\\begin{tabular}{ll|ccc}", "\\hline",
           "Tier & Cell-line information & AUROC & AUPR (AP) & AUPRG \\\\", "\\hline"]
    lab = {"included": "included (trained with)", "removed (cell_emb=0)": "zeroed at inference",
           "retrained without cell lines (pooled)": "retrained without (pooled)"}
    for cv in CVS:
        first = True
        for _, r in t[t.cv == cv].iterrows():
            cells = []
            for k in METRICS:
                s = f"{tex_num(r[f'{k}_mean'])} ({tex_num(r[f'{k}_sd'])})"
                if r.condition != "included":
                    f = fd[(fd.cv == cv) & (fd.A == r.source) & (fd.metric == k)].iloc[0]
                    sign = "+" if r[f"delta_{k}"] >= 0 else ""
                    s += f" {{\\scriptsize {sign}{tex_num(r[f'delta_{k}'])}{'$^\\dagger$' if f.p_t_holm < 0.05 else ''}}}"
                cells.append(s)
            tier_cell = f"\\multirow{{3}}{{*}}{{{TIER[cv]}}}" if first else ""
            first = False
            out.append(f"{tier_cell} & {lab[r.condition]} & " + " & ".join(cells) + " \\\\")
        out.append("\\hline")
    out.append("\\end{tabular}")
    return "\n".join(out)


for arm in CMP:
    (OUT / f"table4_{arm}.tex").write_text(table4_tex(arm) + "\n")

(OUT / "checks.txt").write_text("\n".join(checks) + "\n")
print("wrote", sorted(p.name for p in OUT.iterdir()))
