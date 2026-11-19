#!/usr/bin/env python3
"""Paper-ready tables (markdown + LaTeX) from agreement_split.py's CSVs.
<venv312>/bin/python -B make_tables.py  ->  tables.md, tab_sldb_agreement_split.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
P = pd.read_csv(OUT / "agreement_split_points.csv")
D = pd.read_csv(OUT / "agreement_split_diffs.csv")
SCREENS = ["Adamson", "AdamsonGI", "Corn", "Gilbert"]
LABEL = {"Adamson": "Adamson (growth $\\gamma$)", "AdamsonGI": "Adamson (GI)",
         "Corn": "Corn", "Gilbert": "Gilbert"}
MDLABEL = {"Adamson": "Adamson (Gamma, growth)", "AdamsonGI": "AdamsonGI (Gamma.GI)",
           "Corn": "Corn (sens.score, RPE-1)", "Gilbert": "Gilbert (K562_avg)"}
SUBLABEL = {"all": "all SL (published)", "own_line": "own-line SL",
            "other_line": "other-line SL", "unknown_line": "no-line SL",
            "not_own_line": "not own-line SL (other + no-line)",
            "other_line_uncontradicted": "other-line SL, not non-SL in own line",
            "other_line_contradicted": "other-line SL, non-SL in own line",
            "multi_line": "SL in >= 2 lines", "single_line": "SL in 1 line",
            "src_horlbeck2018": "SL from PMID 30033366 (Horlbeck 2018)",
            "src_gemini2019": "SL from PMID 31300006 (GEMINI)",
            "src_other": "SL from other sources",
            "ownline_label": "own-line label (SL vs non-SL in own line)"}
FLOOR = 2.0 / 1001


def f3(x):
    return "n/a" if not np.isfinite(x) else f"{x:.3f}"


def fs(x):
    return "n/a" if not np.isfinite(x) else f"{x:+.3f}"


def fp(p):
    if not np.isfinite(p):
        return "n/a"
    if p <= FLOOR + 1e-12:
        return "<= 0.002"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def row(screen, subset, scope="all_pairs"):
    r = P[(P.screen == screen) & (P.subset == subset) & (P.scope == scope)]
    return r.iloc[0] if len(r) else None


def diff(screen, contrast, metric, scope="all_pairs"):
    r = D[(D.screen == screen) & (D.contrast == contrast) & (D.metric == metric)
          & (D.scope == scope)]
    return r.iloc[0] if len(r) else None


def cell_auc(r):
    if r is None or not r["reportable"] or not np.isfinite(r["auroc"]):
        return "n/a"
    return f"{r['auroc']:.3f} [{r['auroc_ci_low']:.3f}, {r['auroc_ci_high']:.3f}]"


def cell_sp(r):
    if r is None or not r["reportable"] or not np.isfinite(r["spearman"]):
        return "n/a"
    return (f"{r['spearman']:+.3f} [{r['spearman_ci_low']:+.3f}, "
            f"{r['spearman_ci_high']:+.3f}]")


def cell_rrb(r):
    if r is None or not r["reportable"] or not np.isfinite(r["auroc"]):
        return "n/a"
    return f"{r['r_rankbiserial']:+.3f}"


def cell_d(r):
    if r is None or not r["reportable"] or not np.isfinite(r["auroc"]):
        return "n/a"
    return f"{r['d_binormal']:+.2f}"


def cell_diff(d):
    if d is None or not d["reportable"] or not np.isfinite(d.get("point", np.nan)):
        return "n/a", "n/a", "n/a"
    return (f"{d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
            fp(d["p_two_sided"]),
            fp(d["p_holm"]) if np.isfinite(d.get("p_holm", np.nan)) else "-")


lines = []
A = lines.append
A("## Table A (main, scope all_pairs): SynLethDB label vs screen phenotype, "
  "split by the SL pair's cell line\n")
A("AUROC = P(target of an SL pair > target of a screened non-SL pair) + ties/2 "
  "(target = -phenotype, higher = more SL-like); r_rb = 2 AUROC - 1; "
  "d = sqrt(2) Phi^-1(AUROC) (binormal); 95% percentile CI, pair bootstrap "
  "B = 1000, seed 42. Negatives = every screened non-SL pair of the screen "
  "(the published negatives).\n")
A("| Screen | SL subset | n SL / n non-SL | AUROC [95% CI] | r_rb | d | Spearman [95% CI] | p vs chance (Holm F5) |")
A("|---|---|---|---|---|---|---|---|")
for s in SCREENS:
    for sub in ("all", "own_line", "other_line", "unknown_line"):
        r = row(s, sub)
        n = f"{int(r['n_pos']):,} / {int(r['n_neg']):,}" if r is not None else "?"
        pv = (fp(r["p_holm_vs_chance"]) if r is not None and
              np.isfinite(r.get("p_holm_vs_chance", np.nan)) else "n/a")
        A(f"| {MDLABEL[s]} | {SUBLABEL[sub]} | {n} | {cell_auc(r)} | "
          f"{cell_rrb(r)} | {cell_d(r)} | {cell_sp(r)} | {pv} |")
A("")
A("## Table B (scope all_pairs): own-line minus other-line SL (paired bootstrap of the difference)\n")
A("| Screen | Delta AUROC [95% CI] | p (two-sided) | p Holm (F1) | Delta Spearman [95% CI] | p | p Holm (F2) | direct AUROC own vs other [95% CI] | p |")
A("|---|---|---|---|---|---|---|---|---|")
for s in SCREENS:
    da = cell_diff(diff(s, "own_line - other_line", "auroc"))
    dsp = cell_diff(diff(s, "own_line - other_line", "spearman"))
    dd = diff(s, "own_line vs other_line (direct)", "auroc_direct")
    if dd is not None and dd["reportable"] and np.isfinite(dd.get("point", np.nan)):
        ddc = f"{dd['point']:.3f} [{dd['ci_low']:.3f}, {dd['ci_high']:.3f}]"
        ddp = fp(dd["p_two_sided"])
    else:
        ddc, ddp = "n/a", "n/a"
    A(f"| {MDLABEL[s]} | {da[0]} | {da[1]} | {da[2]} | {dsp[0]} | {dsp[1]} | "
      f"{dsp[2]} | {ddc} | {ddp} |")
A("")
A("## Table C (scope all_pairs, secondary): breadth, SL in >= 2 cell lines vs exactly 1\n")
A("| Screen | multi-line n | AUROC [95% CI] | single-line n | AUROC [95% CI] | Delta AUROC [95% CI] | p | p Holm (F3) | Delta Spearman [95% CI] | p Holm (F4) |")
A("|---|---|---|---|---|---|---|---|---|---|")
for s in SCREENS:
    rm, rs = row(s, "multi_line"), row(s, "single_line")
    da = cell_diff(diff(s, "multi_line - single_line", "auroc"))
    dsp = cell_diff(diff(s, "multi_line - single_line", "spearman"))
    A(f"| {MDLABEL[s]} | {int(rm['n_pos'])} | {cell_auc(rm)} | {int(rs['n_pos'])} | "
      f"{cell_auc(rs)} | {da[0]} | {da[1]} | {da[2]} | {dsp[0]} | {dsp[2]} |")
A("")
A("## Table D (scope all_pairs, sensitivity / diagnostics)\n")
A("| Screen | subset | n SL / n non-SL | AUROC [95% CI] | Spearman [95% CI] |")
A("|---|---|---|---|---|")
for s in SCREENS:
    for sub in ("not_own_line", "other_line_uncontradicted",
                "other_line_contradicted", "ownline_label", "src_horlbeck2018",
                "src_gemini2019", "src_other"):
        r = row(s, sub)
        if r is None:
            continue
        n = f"{int(r['n_pos']):,} / {int(r['n_neg']):,}"
        A(f"| {s} | {SUBLABEL[sub]} | {n} | {cell_auc(r)} | {cell_sp(r)} |")
A("")
A("| Screen | sensitivity contrast | Delta AUROC [95% CI] | p | p Holm |")
A("|---|---|---|---|---|")
for s in SCREENS:
    for con in ("own_line - not_own_line", "own_line - other_line_uncontradicted",
                "ownline_label - all", "src_horlbeck2018 - src_gemini2019"):
        d = cell_diff(diff(s, con, "auroc"))
        A(f"| {s} | {con} | {d[0]} | {d[1]} | {d[2]} |")
A("")
A("## Table E (per screen split, scope test_genes_txt): AUROC [95% CI] by subset\n")
A("| Screen | split | all | own-line | other-line | no-line | Delta own-other [95% CI] (p) |")
A("|---|---|---|---|---|---|---|")
for s in SCREENS:
    for sp in ("cv1", "cv2", "cv3"):
        rr = {sub: P[(P.screen == s) & (P.subset == sub) & (P.scope == "test_genes_txt")
                     & (P.split == sp)] for sub in ("all", "own_line", "other_line", "unknown_line")}
        cells = []
        for sub in ("all", "own_line", "other_line", "unknown_line"):
            r = rr[sub].iloc[0]
            cells.append(f"{cell_auc(r)} (n={int(r['n_pos'])})")
        d = diff(s, "own_line - other_line", "auroc", "test_genes_txt")
        d = D[(D.screen == s) & (D.contrast == "own_line - other_line") &
              (D.metric == "auroc") & (D.scope == "test_genes_txt") & (D.split == sp)].iloc[0]
        dc = cell_diff(d)
        A(f"| {s} | {sp} | " + " | ".join(cells) + f" | {dc[0]} ({dc[1]}) |")
(OUT / "tables.md").write_text("\n".join(lines) + "\n")

# ---------------- LaTeX: the main table for the paper ----------------
tex = []
T = tex.append
T("% generated by make_tables.py from agreement_split_points.csv / _diffs.csv")
T("\\begin{table}[t]")
T("\\centering")
T("\\small")
T("\\begin{tabular}{ll|rcc}")
T("\\hline")
T("Screen & SynLethDB SL pairs & $n$ & AUROC [95\\% CI] & Spearman [95\\% CI] \\\\")
T("\\hline")
for s in SCREENS:
    first = True
    for sub, lab in (("all", "all (SL / non-SL)"), ("own_line", "in the screen's cell line"),
                     ("other_line", "in other cell lines only"),
                     ("unknown_line", "no cell line given")):
        r = row(s, sub)
        name = LABEL[s] if first else ""
        nn = (f"{int(r['n_pos']):,} / {int(r['n_neg']):,}" if sub == "all"
              else f"{int(r['n_pos']):,}")
        first = False
        if r is None or not r["reportable"] or not np.isfinite(r["auroc"]):
            T(f"{name} & {lab} & {nn} & -- & -- \\\\")
            continue
        T(f"{name} & {lab} & {nn} & "
          f"{r['auroc']:.3f} [{r['auroc_ci_low']:.3f}, {r['auroc_ci_high']:.3f}] & "
          f"{r['spearman']:.3f} [{r['spearman_ci_low']:.3f}, {r['spearman_ci_high']:.3f}] \\\\")
    da = diff(s, "own_line - other_line", "auroc")
    dsp = diff(s, "own_line - other_line", "spearman")
    if da is not None and da["reportable"] and np.isfinite(da.get("point", np.nan)):
        T(f" & $\\Delta$ cell line $-$ other & & {da['point']:+.3f} [{da['ci_low']:+.3f}, "
          f"{da['ci_high']:+.3f}] & {dsp['point']:+.3f} [{dsp['ci_low']:+.3f}, {dsp['ci_high']:+.3f}] \\\\")
    T("\\hline")
T("\\end{tabular}")
T("\\caption{\\textbf{Agreement between SynLethDB labels and the external "
  "screens.} On the screen pairs that SynLethDB labels, AUROC of the screen "
  "phenotype (sign-flipped so that higher is more SL-like; 0.5 = no agreement) "
  "for SynLethDB SL vs screened non-SL pairs, and Spearman correlation between "
  "the label and the phenotype. SL pairs are split by the cell line(s) "
  "SynLethDB reports for them (K562 for Adamson and Gilbert, RPE-1 for Corn); "
  "every row uses all screened non-SL pairs as negatives. 95\\% CIs: "
  "percentile bootstrap over pairs ($B=1000$); $\\Delta$: paired bootstrap of "
  "the difference, $p \\le 0.002$ for all three (Holm-adjusted $0.006$). The "
  "pair bootstrap treats pairs as independent although they share genes; a "
  "gene-level bootstrap gives wider intervals (Appendix), under which the "
  "Adamson-GI $\\Delta$ is not significant ($p=0.11$). SynLethDB lists no SL pair "
  "in RPE-1. Gilbert is an in-distribution check: SynLethDB's SL calls for its "
  "pairs come from the same CRISPRi map \\citep{Horlbeck2018}.}")
T("\\label{tab:sldb-agreement}")
T("\\end{table}")
(OUT / "tab_sldb_agreement_split.tex").write_text("\n".join(tex) + "\n")
print("\n".join(lines))
print("\n".join(tex))

# ---------------- follow-up tables (agreement_extra.py outputs) ----------------
X = []
B = X.append
SA = pd.read_csv(OUT / "extra_same_map_contrast.csv")
GP = pd.read_csv(OUT / "extra_gene_bootstrap_points.csv")
GD = pd.read_csv(OUT / "extra_gene_bootstrap_diffs.csv")
DS = pd.read_csv(OUT / "extra_descriptive.csv")


def ci(r, a="auroc", lo="auroc_ci_low", hi="auroc_ci_high", sign=False):
    if r is None or not np.isfinite(r.get(a, np.nan)):
        return "n/a"
    f = (lambda v: f"{v:+.3f}") if sign else (lambda v: f"{v:.3f}")
    return f"{f(r[a])} [{f(r[lo])}, {f(r[hi])}]"


B("## Table F (scope all_pairs): same-map cell-line contrast, K562 vs Jurkat-only SL, "
  "both from PMID 30033366 / 31300006 (same CRISPRi source)\n")
B("| Screen | K562 SL n | AUROC (pair CI) | Jurkat-only SL n | AUROC (pair CI) | Delta (pair CI), p, p Holm | Delta (gene CI), p, p Holm |")
B("|---|---|---|---|---|---|---|")
for s in SCREENS:
    k = SA[(SA.screen == s) & (SA.subset == "k562_same_map")].iloc[0]
    j = SA[(SA.screen == s) & (SA.subset == "jurkat_only_same_map")].iloc[0]
    d = SA[(SA.screen == s) & (SA.subset.str.startswith("DIFF"))]
    g = GD[(GD.screen == s) & (GD.contrast == "k562_same_map - jurkat_only_same_map") & (GD.metric == "auroc")]
    dc = (f"{ci(d.iloc[0], sign=True)}, {fp(d.iloc[0]['p_two_sided'])}, {fp(d.iloc[0]['p_holm'])}"
          if len(d) else "n/a")
    gc = (f"{g.iloc[0]['point']:+.3f} [{g.iloc[0]['ci_low']:+.3f}, {g.iloc[0]['ci_high']:+.3f}], "
          f"{fp(g.iloc[0]['p_two_sided'])}, {fp(g.iloc[0]['p_holm'])}" if len(g) else "n/a")
    B(f"| {MDLABEL[s]} | {int(k['n_pos'])} | {ci(k)} | {int(j['n_pos'])} | {ci(j)} | {dc} | {gc} |")
B("")
B("## Table G (scope all_pairs): pair bootstrap (stated rule) vs gene-level (vertex) bootstrap\n")
B("| Screen | subset | AUROC | pair-bootstrap 95% CI | gene-bootstrap 95% CI | gene-bootstrap p vs 0.5 |")
B("|---|---|---|---|---|---|")
for s in SCREENS:
    for sub in ("all", "own_line", "other_line", "unknown_line"):
        pr = row(s, sub)
        gr = GP[(GP.screen == s) & (GP.subset == sub)]
        if pr is None or not pr["reportable"] or not len(gr) or not np.isfinite(gr.iloc[0].get("auroc", np.nan)):
            B(f"| {s} | {SUBLABEL[sub]} | n/a | n/a | n/a | n/a |")
            continue
        gr = gr.iloc[0]
        B(f"| {s} | {SUBLABEL[sub]} | {pr['auroc']:.3f} | [{pr['auroc_ci_low']:.3f}, {pr['auroc_ci_high']:.3f}] | "
          f"[{gr['auroc_ci_low']:.3f}, {gr['auroc_ci_high']:.3f}] | {fp(gr['p_two_sided_vs_chance'])} |")
B("")
B("| Screen | contrast | Delta AUROC | pair CI, p (Holm) | gene CI, p (Holm) |")
B("|---|---|---|---|---|")
for s in SCREENS:
    pd_ = diff(s, "own_line - other_line", "auroc")
    g = GD[(GD.screen == s) & (GD.contrast == "own_line - other_line") & (GD.metric == "auroc")]
    if pd_ is None or not pd_["reportable"] or not len(g):
        B(f"| {s} | own - other | n/a | n/a | n/a |")
        continue
    g = g.iloc[0]
    B(f"| {s} | own - other | {pd_['point']:+.3f} | [{pd_['ci_low']:+.3f}, {pd_['ci_high']:+.3f}], "
      f"{fp(pd_['p_two_sided'])} ({fp(pd_['p_holm'])}) | [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}], "
      f"{fp(g['p_two_sided'])} ({fp(g['p_holm'])}) |")
B("")
B("## Table H (scope all_pairs): raw phenotype (outcome_raw; SL-like = negative) by subset, "
  "and share of pairs with both genes in the Gilbert (Horlbeck 2018) screen\n")
B("| Screen | subset | n | median outcome_raw [IQR] | both genes in Gilbert screen |")
B("|---|---|---|---|---|")
for _, r in DS.iterrows():
    if r["n"] == 0 or not np.isfinite(r.get("median_outcome_raw", np.nan)):
        continue
    B(f"| {r['screen']} | {r['subset']} | {int(r['n']):,} | {r['median_outcome_raw']:+.3f} "
      f"[{r['q25_outcome_raw']:+.3f}, {r['q75_outcome_raw']:+.3f}] | {r['frac_both_genes_in_gilbert_screen']:.2f} |")
(OUT / "tables.md").write_text("\n".join(lines) + "\n\n" + "\n".join(X) + "\n")
print("\n".join(X))


# ---------------- LaTeX: appendix sensitivity table ----------------
sx = []
U = sx.append
U("% generated by make_tables.py (appendix: sensitivity of the cell-line split)")
U("\\begin{table}[t]")
U("\\centering")
U("\\small")
U("\\begin{tabular}{ll|cc|cc}")
U("\\hline")
U(" & & \\multicolumn{2}{c|}{pair bootstrap} & \\multicolumn{2}{c}{gene bootstrap} \\\\")
U("Screen & contrast ($\\Delta$ AUROC) & $\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$ & 95\\% CI & $p_{\\mathrm{Holm}}$ \\\\")
U("\\hline")


def tex_p(p):
    t = fp(p)
    return "$\\le 0.002$" if t.startswith("<=") else t


for s_ in ("Adamson", "AdamsonGI", "Gilbert"):
    first = True
    items = [
        ("cell line $-$ other lines", diff(s_, "own_line - other_line", "auroc"),
         GD[(GD.screen == s_) & (GD.contrast == "own_line - other_line") & (GD.metric == "auroc")]),
        ("K562 $-$ Jurkat only, same map", SA[(SA.screen == s_) & SA.subset.str.startswith("DIFF")],
         GD[(GD.screen == s_) & (GD.contrast == "k562_same_map - jurkat_only_same_map") & (GD.metric == "auroc")]),
        ("K562 label $-$ any-line label", diff(s_, "ownline_label - all", "auroc"), None),
        ("$\\ge 2$ lines $-$ 1 line", diff(s_, "multi_line - single_line", "auroc"), None),
    ]
    for lab, pr, gr in items:
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0] if len(pr) else None
            pt, lo, hi = (pr["auroc"], pr["auroc_ci_low"], pr["auroc_ci_high"]) if pr is not None else (np.nan,) * 3
        else:
            pt, lo, hi = (pr["point"], pr["ci_low"], pr["ci_high"]) if pr is not None else (np.nan,) * 3
        ph = tex_p(pr["p_holm"]) if pr is not None else "n/a"
        if gr is not None and len(gr):
            g = gr.iloc[0]
            gcell = f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] & {tex_p(g['p_holm'])}"
        else:
            gcell = "-- & --"
        name = LABEL[s_] if first else ""
        first = False
        U(f"{name} & {lab} & {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] & {ph} & {gcell} \\\\")
    U("\\hline")
U("\\end{tabular}")
U("\\caption{\\textbf{Sensitivity of the cell-line split} (Table~\\ref{tab:sldb-agreement}). "
  "Same map: K562 vs Jurkat-only SynLethDB SL pairs whose evidence is the same CRISPRi "
  "genetic-interaction data (PubMed 30033366 / 31300006), so that technology and source are "
  "held fixed. K562 label: positives SL in K562, negatives screened non-SL in K562 "
  "(the K562 head's label), against the any-cell-line label. $\\ge 2$ lines: SL reported "
  "in at least two cell lines vs exactly one. Gene bootstrap: genes resampled with "
  "replacement, a pair weighted by the product of its genes' multiplicities ($B=1000$). "
  "Holm adjustment across the three screens within each contrast and bootstrap "
  "(4 screens for the breadth contrast, incl. Corn, not shown: $+0.038$ [$-0.120$, $+0.183$]).}")
U("\\label{tab:sldb-agreement-sens}")
U("\\end{table}")
(OUT / "tab_sldb_agreement_sens.tex").write_text("\n".join(sx) + "\n")
print("\n".join(sx))

# ---------------- typographic minus in the LaTeX outputs ----------------
import re  # noqa: E402
for fn in ("tab_sldb_agreement_split.tex", "tab_sldb_agreement_sens.tex"):
    f = OUT / fn
    f.write_text(re.sub(r"(?<![\$\w])-(?=\d)", "$-$", f.read_text()))
