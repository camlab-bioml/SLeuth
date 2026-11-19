#!/usr/bin/env python3
"""Plan item 13 / 22DY W4, Q-c: what each external-screen variable is, its raw
sign convention, the sign applied, and the pair counts, checked against the
screen files in "external data/" and the eval_finetuning CSVs the tables were
computed from. Read-only on the repo.

Writes screen_variables.json and prints a summary.
Usage: python -I verify_screens.py
"""
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "external data"
OUT = Path(__file__).resolve().parent
SIA = REPO / "siamese_sl" / "results"
COMBO = {
    "siamese_sl": SIA / "best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3",
    "siamese_sl_pd_pooled": SIA / "var_pd_pooled__best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3",
}
SCREENS = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def col_index(ref):
    s = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def stream_xlsx_columns(path, sheet_xml, cols, skip_rows):
    """Yield tuples of the requested 0-based columns for every row after
    skip_rows (shared strings resolved)."""
    with zipfile.ZipFile(path) as z:
        sst = []
        with z.open("xl/sharedStrings.xml") as fh:
            for ev, el in ET.iterparse(fh, events=("end",)):
                if el.tag == NS + "si":
                    sst.append("".join(t.text or "" for t in el.iter(NS + "t")))
                    el.clear()
        with z.open(sheet_xml) as fh:
            i = 0
            for ev, el in ET.iterparse(fh, events=("end",)):
                if el.tag != NS + "row":
                    continue
                i += 1
                if i <= skip_rows:
                    el.clear()
                    continue
                cells = {}
                for c in el.iter(NS + "c"):
                    ci = col_index(c.get("r"))
                    if ci not in cols:
                        continue
                    v = c.find(NS + "v")
                    t = c.get("t")
                    if v is None:
                        val = None
                    elif t == "s":
                        val = sst[int(v.text)]
                    else:
                        val = v.text
                    cells[ci] = val
                el.clear()
                yield tuple(cells.get(ci) for ci in cols)


def main():
    rec = {}
    # ---------------- raw files ----------------
    # Adamson: sgRNA-pair phenotypes (Gamma / Tau / Rho)
    ad = pd.read_csv(EXT / "pair_sgRNA_phenotypes.txt", sep="\t",
                     usecols=["FirstGene", "SecondGene", "Gamma"])
    rec["Adamson_file"] = {
        "file": "external data/pair_sgRNA_phenotypes.txt",
        "columns": list(pd.read_csv(EXT / "pair_sgRNA_phenotypes.txt",
                                    sep="\t", nrows=1).columns),
        "n_rows": int(len(ad)),
        "n_self_rows": int((ad.FirstGene == ad.SecondGene).sum()),
        "gamma_quantiles": {q: float(ad.Gamma.quantile(q))
                            for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
        "share_gamma_negative": float((ad.Gamma < 0).mean()),
    }
    # Adamson: sgRNA level Gamma on self pairs (single-gene phenotype proxy)
    # Corn
    co = pd.read_csv(EXT / "41586_2025_8815_MOESM5_ESM.csv")
    co_sorted = co.sort_values("rank")
    rec["Corn_file"] = {
        "file": "external data/41586_2025_8815_MOESM5_ESM.csv",
        "columns": list(co.columns),
        "n_rows": int(len(co)),
        "share_exactly_zero": float((co["sens.score"] == 0).mean()),
        "share_negative": float((co["sens.score"] < 0).mean()),
        "share_positive": float((co["sens.score"] > 0).mean()),
        "rank1_rows": co_sorted.head(5)[["gene_combination", "sens.score",
                                          "rank"]].to_dict("records"),
        "last_rank_rows": co_sorted.tail(3)[["gene_combination",
                                             "sens.score",
                                             "rank"]].to_dict("records"),
        "spearman_rank_vs_score": float(co[["rank", "sens.score"]].corr(
            "spearman").iloc[0, 1]),
        "n_mis_suffix": int(co.gene_combination.str.contains("_mis").sum()),
    }
    # AdamsonGI: Simpson 2023 Table S3, sheet 1; header row 2
    p = EXT / "Simpson2023_TableS3_GI_scores.xlsx"
    rows = list(stream_xlsx_columns(p, "xl/worksheets/sheet1.xml",
                                    (0, 1, 2, 3, 4, 16), skip_rows=2))
    gi = pd.DataFrame(rows, columns=["GeneCombinationName", "Gamma.GI",
                                     "Gamma.Nsupp", "Gamma.Pvalue",
                                     "Gamma.Qvalue", "InteractionCategory"])
    gi["Gamma.GI"] = pd.to_numeric(gi["Gamma.GI"], errors="coerce")
    gi["Gamma.Qvalue"] = pd.to_numeric(gi["Gamma.Qvalue"], errors="coerce")
    cat = (gi.groupby("InteractionCategory", dropna=False)["Gamma.GI"]
           .agg(n="size", n_finite="count", mean="mean", median="median",
                share_negative=lambda s: float((s < 0).mean()))
           .reset_index())
    rec["AdamsonGI_file"] = {
        "file": "external data/Simpson2023_TableS3_GI_scores.xlsx",
        "n_rows": int(len(gi)),
        "n_gamma_gi_finite": int(gi["Gamma.GI"].notna().sum()),
        "n_same_gene": int(gi.GeneCombinationName.astype(str).apply(
            lambda s: len(set(s.split(":"))) == 1).sum()),
        "n_non_targeting": int(gi.GeneCombinationName.astype(str)
                               .str.upper().str.contains("NON-TARGETING")
                               .sum()),
        "interaction_category_vs_gamma_gi": cat.to_dict("records"),
        "share_gamma_gi_negative": float((gi["Gamma.GI"] < 0).mean()),
    }
    # Gilbert: Horlbeck 2018 mmc5 sheet 2 (gene GI scores), K562 Replicate
    # Average GI score = column G (index 6); rows 1-3 header, row 4 data
    p = EXT / "mmc5.xlsx"
    rows = list(stream_xlsx_columns(p, "xl/worksheets/sheet2.xml",
                                    (0, 1, 6), skip_rows=3))
    gb = pd.DataFrame(rows, columns=["gene1", "gene2", "K562_avg_GI"])
    gb["K562_avg_GI"] = pd.to_numeric(gb["K562_avg_GI"], errors="coerce")
    rec["Gilbert_file"] = {
        "file": "external data/mmc5.xlsx",
        "sheet": "gene GI scores and correlations (header rows: cell line / "
                 "replicate / metric)",
        "column": "K562 | Replicate Average | GI score (col G)",
        "n_rows": int(len(gb)),
        "n_finite": int(gb.K562_avg_GI.notna().sum()),
        "n_self": int((gb.gene1 == gb.gene2).sum()),
        "quantiles": {q: float(gb.K562_avg_GI.quantile(q))
                      for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
        "share_negative": float((gb.K562_avg_GI < 0).mean()),
    }

    # ---------------- what the eval scored (per split dir) ----------------
    ev = {}
    for name, d in COMBO.items():
        for split in ("cv1", "cv2", "cv3"):
            ed = d / f"eval_finetuning_{split}"
            cfg = json.load(open(ed / "config_used.json"))
            for s in SCREENS:
                df = pd.read_csv(ed / f"{s}_predictions.csv",
                                 dtype={"gene1_entrez": str,
                                        "gene2_entrez": str})
                ratio = df.target / df.outcome_raw
                nz = df.outcome_raw != 0
                ev[(name, split, s)] = {
                    "n_pairs": int(len(df)),
                    "n_genes": int(pd.unique(pd.concat(
                        [df.gene1_entrez, df.gene2_entrez])).size),
                    "target_equals_minus_outcome": bool(np.allclose(
                        df.target, -df.outcome_raw, rtol=0, atol=1e-12)),
                    "target_over_outcome_unique": sorted(set(np.round(
                        ratio[nz], 12)))[:3],
                    "split_counts": df.split.value_counts().to_dict(),
                    "n_test_sldb_overlap": int(
                        (df.split.eq("test") & df.sldb_overlap).sum()),
                    "n_test_not_sldb": int(
                        (df.split.eq("test") & ~df.sldb_overlap).sum()),
                    "share_outcome_zero": float((df.outcome_raw == 0).mean()),
                    "effective_head": cfg["effective_head"][s],
                    "ft_val_regime": cfg.get("ft_val_regime"),
                    "train_frac": cfg["train_frac"],
                    "val_frac": cfg["val_frac"],
                    "split_seed": cfg["split_seed"],
                    "ft_modes": cfg["ft_modes"],
                    "ft_epochs": cfg["ft_epochs"],
                    "ft_patience": cfg["ft_patience"],
                    "ft_lr": cfg["ft_lr"],
                    "ft_batch_size": cfg["ft_batch_size"],
                    "ft_weight_decay": cfg["ft_weight_decay"],
                    "spec": [x for x in cfg["datasets_spec"]
                             if x.startswith(s + ":")][0],
                    "sldb_overlap_sources": [Path(x).name for x in
                                             cfg["sldb_overlap_sources"]],
                }
    rec["eval"] = {"|".join(k): v for k, v in ev.items()}
    # same rows across variants / splits?
    for s in SCREENS:
        ns = {ev[(n, sp, s)]["n_pairs"] for n in COMBO for sp in
              ("cv1", "cv2", "cv3")}
        rec.setdefault("n_pairs_same_everywhere", {})[s] = sorted(ns)
    with open(OUT / "screen_variables.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    # print compact summary
    for k in ("Adamson_file", "Corn_file", "AdamsonGI_file", "Gilbert_file"):
        print("==", k)
        print(json.dumps(rec[k], indent=1, default=str)[:3000])
    for key, v in rec["eval"].items():
        if key.startswith("siamese_sl|"):
            print(key, {x: v[x] for x in ("n_pairs", "n_genes",
                                          "target_equals_minus_outcome",
                                          "split_counts",
                                          "n_test_sldb_overlap",
                                          "n_test_not_sldb",
                                          "share_outcome_zero",
                                          "effective_head",
                                          "ft_val_regime", "spec")})
    for key, v in rec["eval"].items():
        if key.startswith("siamese_sl_pd_pooled|"):
            print(key, v["effective_head"], v["n_pairs"],
                  v["target_equals_minus_outcome"])
    print(rec["n_pairs_same_everywhere"])


if __name__ == "__main__":
    sys.exit(main())
