#!/usr/bin/env python3
"""Compact markdown tables for report.md, built only from ./out/*.csv written by compute.py.
Run after compute.py:  <venv312>/bin/python -I report_tables.py"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
F = pd.read_csv(OUT / "sig_foldpaired.csv")
Bt = pd.read_csv(OUT / "sig_bootstrap.csv")
T1 = pd.read_csv(OUT / "table1_none.csv")
TIERS = ["I", "II", "III"]
ML = {"auroc": "AUROC", "ap": "AP", "auprg": "AUPRG"}
FLOOR = 2.0 / 1001


def f3(x):
    s = f"{x:+.3f}"
    if float(s) == 0:
        s = "0.000"
    return s.replace("-", "−")


def pt(p):
    if abs(p - 0.0625) < 1e-12:
        return "0.0625"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def pb(p, m=1):
    if p <= m * FLOOR + 1e-12:
        return f"≤{m * FLOOR:.3f}"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def fold_cell(r):
    star = "†" if r.p_t_holm < 0.05 else ""
    return f"{f3(r.diff_mean)}{star} [{f3(r.ci_t_low)}, {f3(r.ci_t_high)}], {int(r.wins)}/5, p_t {pt(r.p_t)} ({pt(r.p_t_holm)})"


def boot_cell(arm, fam, a, b, tier, k):
    x = Bt[(Bt.pca_arm == arm) & (Bt.family == fam) & (Bt.model == a) & (Bt.peer == b) & (Bt.tier == tier) & (Bt.metric == k)]
    if not len(x):
        return "—"
    x = x.iloc[0]
    hol = pb(x.p2_holm, int(x.family_m)) if x.p2_at_floor else pb(x.p2_holm)
    return f"{f3(x.point)} [{f3(x.ci_low)}, {f3(x.ci_high)}], p {pb(x.p2)} ({hol})"


out = []

# 1. pooled vs cell-line, both pairs side by side (fold-paired only; no bootstrap pair exists)
for arm in ["none", "joint"]:
    out.append(f"### pooled_vs_cellline_compact_{arm}")
    out.append("| Tier | Metric | selected: pooled − cell-line: Δ [split 95% t-CI], folds up, p_t (Holm) | all-embedding: pooled − cell-line: Δ [t-CI], folds up, p_t (Holm) |")
    out.append("|---|---|---|---|")
    g = F[(F.pca_arm == arm) & (F.family == "pooled_vs_cellline")]
    for t in TIERS:
        for k in ML:
            a = g[(g.tier == t) & (g.metric == k) & (g.A == "siamese_sl_pd_pooled")].iloc[0]
            b = g[(g.tier == t) & (g.metric == k) & (g.A == "siamese_sl_allmod_pd_pooled")].iloc[0]
            out.append(f"| {t} | {ML[k]} | {fold_cell(a)} | {fold_cell(b)} |")
    out.append("")

# 2. selected vs all-embedding (modality selection effect), bootstrap + fold-paired
for arm in ["none", "joint"]:
    out.append(f"### selected_vs_allmod_compact_{arm}")
    out.append("| Tier | Metric | pooled: selected − all-16 (fold-paired) | pooled: pair bootstrap Δ [CI], p (Holm) | cell-line: selected − all-16 (fold-paired) | cell-line: pair bootstrap |")
    out.append("|---|---|---|---|---|---|")
    g = F[(F.pca_arm == arm) & (F.family == "selected_vs_allmod")]
    for t in TIERS:
        for k in ML:
            a = g[(g.tier == t) & (g.metric == k) & (g.A == "siamese_sl_pd_pooled")].iloc[0]
            b = g[(g.tier == t) & (g.metric == k) & (g.A == "siamese_sl")].iloc[0]
            out.append(f"| {t} | {ML[k]} | {fold_cell(a)} | {boot_cell(arm, 'selected_vs_allmod', 'siamese_sl_pd_pooled', 'siamese_sl_allmod_pd_pooled', t, k)} | "
                       f"{fold_cell(b)} | {boot_cell(arm, 'selected_vs_allmod', 'siamese_sl', 'siamese_sl_allmod', t, k)} |")
    out.append("")

# 3. bilinear probe (arm none only)
out.append("### probe_compact_none")
out.append("| Tier | Metric | SLeuth (pooled) − probe: boot Δ [CI], p (Holm) | fold-paired | SLeuth-CL − probe: boot | fold-paired | SLeuth-all (pooled) − probe: fold-paired |")
out.append("|---|---|---|---|---|---|---|")
for t in TIERS:
    for k in ML:
        cells = []
        for a in ["siamese_sl_pd_pooled", "siamese_sl"]:
            r = F[(F.pca_arm == "none") & (F.family == f"vs_peers:{a}") & (F.B == "BilinearProbe") & (F.tier == t) & (F.metric == k)].iloc[0]
            cells += [boot_cell("none", f"vs_peers:{a}", a, "BilinearProbe", t, k), fold_cell(r)]
        r = F[(F.pca_arm == "none") & (F.family == "vs_peers:siamese_sl_allmod_pd_pooled") & (F.B == "BilinearProbe") & (F.tier == t) & (F.metric == k)].iloc[0]
        cells.append(fold_cell(r))
        out.append(f"| {t} | {ML[k]} | " + " | ".join(cells) + " |")
out.append("")

# 4. 84QL: Tier III vs SLMGAE, every SLeuth row
out.append("### q84QL_compact")
out.append("| SLeuth row | Metric | mean (sd) | SLMGAE mean (sd) | per-fold Δ | pair bootstrap Δ [CI], p (Holm) | fold-paired: Δ [t-CI], folds up, p_t (Holm) | sign-flip p | NB-corrected p_t |")
out.append("|---|---|---|---|---|---|---|---|---|")
for a, lab in [("siamese_sl_pd_pooled", "SLeuth (pooled)"), ("siamese_sl", "SLeuth-CL"),
               ("siamese_sl_allmod_pd_pooled", "SLeuth-all (pooled)"), ("siamese_sl_allmod", "SLeuth-all-CL")]:
    for k in ML:
        r = F[(F.pca_arm == "none") & (F.family == f"vs_peers:{a}") & (F.B == "SLMGAE") & (F.tier == "III") & (F.metric == k)].iloc[0]
        ta = T1[(T1.model == a) & (T1.tier == "III")].iloc[0]
        ts = T1[(T1.model == "SLMGAE") & (T1.tier == "III")].iloc[0]
        out.append(f"| {lab} | {ML[k]} | {ta[f'{k}_mean']:.3f} ({ta[f'{k}_sd']:.3f}) | {ts[f'{k}_mean']:.3f} ({ts[f'{k}_sd']:.3f}) | "
                   f"{r.per_fold_diff.replace(';', ' ')} | {boot_cell('none', f'vs_peers:{a}', a, 'SLMGAE', 'III', k)} | "
                   f"{fold_cell(r)} | {pt(r.p_signflip)} | {pt(r.p_nb)} |")
out.append("")

# 5. main-text Table 1 (arm none): mean (sd); on a baseline cell, a dagger = SLeuth (pooled) above it
#    with fold-paired t-test Holm p < 0.05 (family: SLeuth (pooled) x tier x metric, 6 rows);
#    every baseline cell is also below SLeuth (pooled) in the pair bootstrap at Holm p <= 0.012.
out.append("### table1_main_none")
out.append("| Tier | Model | Params | AUROC | AUPR (AP) | AUPRG |")
out.append("|---|---|---:|---|---|---|")
peers = ["SLMGAE", "SLGNN", "NSF4SL", "ELISL", "BilinearProbe", "DegreePrior"]
for t in TIERS:
    sub = T1[T1.tier == t]
    cand = sub[sub.model != "DegreePrior"]
    best = {k: round(cand[f"{k}_mean"].max(), 3) for k in ML}
    for _, r in sub.iterrows():
        cells = []
        for k in ML:
            s = f"{r[f'{k}_mean']:.3f}"
            if r.model != "DegreePrior" and round(r[f"{k}_mean"], 3) == best[k]:
                s = f"**{s}**"
            s += f" ({r[f'{k}_sd']:.3f})"
            if r.model in peers:
                f = F[(F.pca_arm == "none") & (F.family == "vs_peers:siamese_sl_pd_pooled") & (F.B == r.model) &
                      (F.tier == t) & (F.metric == k)].iloc[0]
                assert f.diff_mean > 0 and f.wins == 5
                s += "†" if f.p_t_holm < 0.05 else ""
            cells.append(s.replace("-0.000", "0.000"))
        if r.model == "DegreePrior":
            par = "0"
        elif r.params_sd > 0:
            par = f"{r.params_mean:,.0f} ({r.params_sd:,.0f})"
        else:
            par = f"{r.params_mean:,.0f}"
        out.append(f"| {t} | {r.display} | {par} | " + " | ".join(cells) + " |")
out.append("")

# 6. merged Table 1 (arm none): mean (sd) [95% pair-bootstrap CI of the 5-split mean], dagger as in 5.
out.append("### table1_full_none")
out.append("| Tier | Model | Params | AUROC | AUPR (AP) | AUPRG |")
out.append("|---|---|---:|---|---|---|")


def c3(x):
    s = f"{x:.3f}"
    return "0.000" if float(s) == 0 else s.replace("-", "−")


for t in TIERS:
    sub = T1[T1.tier == t]
    cand = sub[sub.model != "DegreePrior"]
    best = {k: round(cand[f"{k}_mean"].max(), 3) for k in ML}
    for _, r in sub.iterrows():
        cells = []
        for k in ML:
            s = c3(r[f"{k}_mean"])
            if r.model != "DegreePrior" and round(r[f"{k}_mean"], 3) == best[k]:
                s = f"**{s}**"
            s += f" ({c3(r[f'{k}_sd'])}) [{c3(r[f'{k}_ci_low'])}, {c3(r[f'{k}_ci_high'])}]"
            if r.model in peers:
                f = F[(F.pca_arm == "none") & (F.family == "vs_peers:siamese_sl_pd_pooled") & (F.B == r.model) &
                      (F.tier == t) & (F.metric == k)].iloc[0]
                s += "†" if f.p_t_holm < 0.05 else ""
            cells.append(s)
        if r.model == "DegreePrior":
            par = "0"
        elif r.params_sd > 0:
            par = f"{r.params_mean:,.0f} ({r.params_sd:,.0f})"
        else:
            par = f"{r.params_mean:,.0f}"
        out.append(f"| {t} | {r.display} | {par} | " + " | ".join(cells) + " |")
out.append("")

(OUT / "report_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
