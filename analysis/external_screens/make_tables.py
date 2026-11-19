#!/usr/bin/env python3
"""Markdown (and LaTeX) renderings of the analysis CSVs in out_<arm>/.

Usage: python -I make_tables.py ARM   -> out_<arm>/tables.md, table5.tex,
                                         table6.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCREENS = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
SPLITS = ("cv1", "cv2", "cv3")
SPL = {"cv1": "random pairs", "cv2": ">=1 gene unseen",
       "cv3": "both unseen"}
MODELS5 = ("siamese_sl", "siamese_sl_pd_pooled", "SLMGAE", "SLGNN",
           "NSF4SL", "DegreePrior")
NAME = {"siamese_sl": "SLeuth (cell line)",
        "siamese_sl_pd_pooled": "SLeuth (pooled)", "SLMGAE": "SLMGAE",
        "SLGNN": "SLGNN", "NSF4SL": "NSF4SL", "DegreePrior": "DegreePrior",
        "SynLethDB_label": "SynLethDB label"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:.{d}f}"
    return s.replace("-", "−") if s.startswith("-") else s


def fp(p):
    if not np.isfinite(p):
        return "n/a"
    if p <= 2.0 / 1001 + 1e-12:
        return "≤0.002"
    return f"{p:.3f}" if p < 0.995 else "1"


def ftp(p):
    """t-test p (no bootstrap floor)."""
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi):
    return f"[{f(lo)}, {f(hi)}]"


def n_pos(fold_deltas):
    v = [float(x) for x in str(fold_deltas).split()]
    return sum(x > 0 for x in v), len(v)


def table5_md(arm):
    o = HERE / f"out_{arm}"
    t = pd.read_csv(o / "table5_zero_shot.csv")
    out = ["| Screen | Split | n | " + " | ".join(NAME[m] for m in MODELS5)
           + " |", "|---|---|---|" + "---|" * len(MODELS5)]
    for s in SCREENS:
        for sp in SPLITS:
            x = t[(t.screen == s) & (t.split == sp)].set_index("model")
            cells = []
            for m in MODELS5:
                r = x.loc[m]
                mark = "†" if r.ci_excludes_0 else ""
                cells.append(f"{f(r['mean'])} ({f(r.sd_pop)}) "
                             f"{ci(r.ci_low, r.ci_high)}{mark}")
            out.append(f"| {s} | {SPL[sp]} | {int(x.n_rows.iloc[0]):,} | "
                       + " | ".join(cells) + " |")
    return "\n".join(out)


def table5_tex(arm):
    o = HERE / f"out_{arm}"
    t = pd.read_csv(o / "table5_zero_shot.csv")
    lines = [r"\begin{tabular}{llr|cccccc}", r"\hline",
             r"Screen & Split & $n$ & SLeuth (cell line) & SLeuth (pooled) & "
             r"SLMGAE & SLGNN & NSF4SL & DegreePrior \\", r"\hline"]

    def tx(v):
        s = f"{v:.3f}"
        return s.replace("-", "$-$")
    for s in SCREENS:
        for i, sp in enumerate(SPLITS):
            x = t[(t.screen == s) & (t.split == sp)].set_index("model")
            c1 = [f"{tx(x.loc[m]['mean'])} ({x.loc[m].sd_pop:.3f})"
                  + (r"$^\dagger$" if x.loc[m].ci_excludes_0 else "")
                  for m in MODELS5]
            c2 = [f"\\scriptsize[{tx(x.loc[m].ci_low)}, "
                  f"{tx(x.loc[m].ci_high)}]" for m in MODELS5]
            lab = s if i == 0 else ""
            lines.append(f"{lab} & {SPL[sp]} & {int(x.n_rows.iloc[0]):,} & "
                         + " & ".join(c1) + r" \\")
            lines.append(" & & & " + " & ".join(c2) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def diffs_md(arm, a):
    o = HERE / f"out_{arm}"
    d = pd.read_csv(o / "table5_diffs.csv")
    v = pd.read_csv(o / "table5_cellline_minus_pooled.csv")
    peers = ("SLMGAE", "SLGNN", "NSF4SL", "DegreePrior")
    hdr = ["| Screen | Split | " + " | ".join(f"{NAME[a]} − {p}"
                                             for p in peers)]
    if a == "siamese_sl":
        hdr[0] += " | cell line − pooled"
    hdr[0] += " |"
    hdr.append("|---|---|" + "---|" * (len(peers) + (a == "siamese_sl")))
    for s in SCREENS:
        for sp in SPLITS:
            cells = []
            for p in peers:
                r = d[(d.screen == s) & (d.split == sp) & (d.A == a) &
                      (d.B == p)].iloc[0]
                k, n = n_pos(r.fold_deltas)
                cells.append(f"{f(r.delta)} {ci(r.ci_low, r.ci_high)} "
                             f"pH={fp(r.p2_holm)}; {k}/{n}")
            if a == "siamese_sl":
                r = v[(v.screen == s) & (v.split == sp)].iloc[0]
                k, n = n_pos(r.fold_deltas)
                cells.append(f"{f(r.delta)} {ci(r.ci_low, r.ci_high)} "
                             f"pH={fp(r.p2_holm)}; {k}/{n}")
            hdr.append(f"| {s} | {SPL[sp]} | " + " | ".join(cells) + " |")
    return "\n".join(hdr)


def sldb_md(arm):
    o = HERE / f"out_{arm}"
    t = pd.read_csv(o / "table5_matched_in_sldb.csv")
    dd = pd.read_csv(o / "table5_sleuth_minus_label.csv")
    ms = ("SynLethDB_label", "siamese_sl", "siamese_sl_pd_pooled", "SLMGAE",
          "SLGNN", "NSF4SL", "DegreePrior")
    out = ["| Screen | Split | n (SynLethDB pairs) | " +
           " | ".join(NAME[m] for m in ms) +
           " | cell line − label (pH) |",
           "|---|---|---|" + "---|" * (len(ms) + 1)]
    for s in SCREENS:
        for sp in SPLITS:
            x = t[(t.screen == s) & (t.split == sp)].set_index("model")
            cells = [f"{f(x.loc[m]['mean'])} {ci(x.loc[m].ci_low, x.loc[m].ci_high)}"
                     + ("†" if x.loc[m].ci_excludes_0 else "") for m in ms]
            r = dd[(dd.screen == s) & (dd.split == sp) &
                   (dd.A == "siamese_sl")].iloc[0]
            cells.append(f"{f(r.delta)} {ci(r.ci_low, r.ci_high)} "
                         f"({fp(r.p2_holm24)})")
            out.append(f"| {s} | {SPL[sp]} | {int(x.n_rows.iloc[0]):,} | "
                       + " | ".join(cells) + " |")
    return "\n".join(out)


def table6_md(arm, model, rs):
    o = HERE / f"out_{arm}"
    t = pd.read_csv(o / "table6_finetune.csv")
    t = t[(t.model == model) & (t.row_set == rs)]
    meths = ("zero_shot_raw", "zero_shot_ols", "LP_last", "Full", "Scratch")
    lab = {"zero_shot_raw": "Zero-shot (raw)",
           "zero_shot_ols": "Zero-shot (OLS-calibrated)",
           "LP_last": "LP_last", "Full": "Full", "Scratch": "Scratch"}
    out = ["| Screen | Split | n | " + " | ".join(lab[m] for m in meths) +
           " |", "|---|---|---|" + "---|" * len(meths)]
    for s in SCREENS:
        for sp in SPLITS:
            x = t[(t.screen == s) & (t.split == sp)].set_index("method")
            cells = [f"{f(x.loc[m]['mean'])} ({f(x.loc[m].sd_pop)}) "
                     f"{ci(x.loc[m].ci_low, x.loc[m].ci_high)}"
                     for m in meths]
            out.append(f"| {s} | {SPL[sp]} | {int(x.n_rows.iloc[0]):,} | "
                       + " | ".join(cells) + " |")
    return "\n".join(out)


def table6_tex(arm, model="siamese_sl", rs="all_sldb_filtered"):
    o = HERE / f"out_{arm}"
    t = pd.read_csv(o / "table6_finetune.csv")
    t = t[(t.model == model) & (t.row_set == rs)]
    meths = ("zero_shot_raw", "zero_shot_ols", "LP_last", "Full", "Scratch")

    def tx(v):
        return f"{v:.3f}".replace("-", "$-$")
    lines = [r"\begin{tabular}{ll|ccccc}", r"\hline",
             r"Screen & Split & Zero-shot & Zero-shot (OLS) & "
             r"\texttt{LP\_last} & \texttt{Full} & \texttt{Scratch} \\",
             r"\hline"]
    for s in SCREENS:
        for i, sp in enumerate(SPLITS):
            x = t[(t.screen == s) & (t.split == sp)].set_index("method")
            c1 = [f"{tx(x.loc[m]['mean'])} ({x.loc[m].sd_pop:.3f})"
                  for m in meths]
            c2 = [f"\\scriptsize[{tx(x.loc[m].ci_low)}, "
                  f"{tx(x.loc[m].ci_high)}]" for m in meths]
            lab = s if i == 0 else ""
            lines.append(f"{lab} & {SPL[sp]} & " + " & ".join(c1) + r" \\")
            lines.append(" & & " + " & ".join(c2) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def tests6_md(arm, model, rs, contrasts):
    o = HERE / f"out_{arm}"
    d = pd.read_csv(o / "table6_tests.csv")
    d = d[(d.model == model) & (d.row_set == rs)]
    out = ["| Screen | Split | " + " | ".join(contrasts) + " |",
           "|---|---|" + "---|" * len(contrasts)]
    for s in SCREENS:
        for sp in SPLITS:
            cells = []
            for c in contrasts:
                r = d[(d.screen == s) & (d.split == sp) &
                      (d.contrast == c)].iloc[0]
                k, n = n_pos(r.fold_deltas)
                cells.append(f"{f(r.delta)} {ci(r.ci_low, r.ci_high)} "
                             f"pH={fp(r.p2_holm)}; {k}/{n}; t={ftp(r.t_p)}")
            out.append(f"| {s} | {SPL[sp]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def drops_md(arm, model, methods=("Full", "Scratch", "LP_last"),
             rs="all_sldb_filtered"):
    o = HERE / f"out_{arm}"
    fn = ("split_drops_all_sldb_filtered.csv" if rs == "all_sldb_filtered"
          else "split_drops_matched.csv")
    d = pd.read_csv(o / fn)
    d = d[d.model == model]
    out = ["| Screen | Mode | random pairs | >=1 unseen | both unseen | "
           "drop random→>=1 unseen | drop random→both unseen |",
           "|---|---|---|---|---|---|---|"]
    for s in SCREENS:
        for m in methods:
            x = d[(d.screen == s) & (d.method == m)]
            r12 = x[(x["from"] == "cv1") & (x["to"] == "cv2")].iloc[0]
            r13 = x[(x["from"] == "cv1") & (x["to"] == "cv3")].iloc[0]
            r23 = x[(x["from"] == "cv2") & (x["to"] == "cv3")].iloc[0]
            out.append(
                f"| {s} | {m} | {f(r12.mean_from)} | {f(r12.mean_to)} | "
                f"{f(r13.mean_to)} | {f(r12["drop"])} {ci(r12.ci_low, r12.ci_high)}"
                f" ({100 * r12.rel_drop:.0f}%; pH={fp(r12.p2_holm)}, "
                f"t={ftp(r12.t_p)}) | {f(r13["drop"])} "
                f"{ci(r13.ci_low, r13.ci_high)} ({100 * r13.rel_drop:.0f}%; "
                f"pH={fp(r13.p2_holm)}, t={ftp(r13.t_p)}) |")
    return "\n".join(out)


def main(arm):
    o = HERE / f"out_{arm}"
    parts = [f"# Tables, PCA arm {arm}", "",
             "## Table 5: zero-shot Spearman, matched rows", "",
             table5_md(arm), "",
             "## Table 5 differences: SLeuth (cell line) − comparator", "",
             diffs_md(arm, "siamese_sl"), "",
             "## Table 5 differences: SLeuth (pooled) − comparator", "",
             diffs_md(arm, "siamese_sl_pd_pooled"), "",
             "## Table 5 side table: SynLethDB pairs among the matched-gene "
             "test pairs (matched_in_sldb)", "", sldb_md(arm), ""]
    for model in ("siamese_sl", "siamese_sl_pd_pooled"):
        for rs in ("all_sldb_filtered", "matched"):
            parts += [f"## Table 6: {NAME[model]}, row set {rs}", "",
                      table6_md(arm, model, rs), "",
                      f"## Table 6 tests: {NAME[model]}, row set {rs}", "",
                      tests6_md(arm, model, rs,
                                ("Full - Scratch",
                                 "LP_last - zero-shot OLS",
                                 "Full - zero-shot OLS")), "",
                      tests6_md(arm, model, rs,
                                ("LP_last - zero-shot raw",
                                 "Full - zero-shot raw",
                                 "LP_last - Scratch")), ""]
        parts += [f"## Split drop: {NAME[model]}, all_sldb_filtered", "",
                  drops_md(arm, model), ""]
    (o / "tables.md").write_text("\n".join(parts))
    (o / "table5.tex").write_text(table5_tex(arm))
    (o / "table6_cellline.tex").write_text(table6_tex(arm))
    (o / "table6_pooled.tex").write_text(
        table6_tex(arm, model="siamese_sl_pd_pooled"))
    print(f"wrote {o / 'tables.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
