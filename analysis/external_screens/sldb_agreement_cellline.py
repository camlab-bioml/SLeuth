#!/usr/bin/env python3
"""SynLethDB label vs screen phenotype, general (any cell line, the shared
folds' OR-collapse) and cell-line specific (plan item 11: 'with cell-line
specific SL and general SL types'; 84QL Q-b).

On every scored screen pair that SynLethDB labels (the eval's sldb_overlap
rows of <Screen>_predictions.csv), AUROC of the screen target (higher = more
SL, target = -1 x raw phenotype) for SynLethDB SL vs screened non-SL, and
Spearman(label, target), each with a pair-resampling bootstrap 95% CI
(B=1000, seed 42, bootstrap_ci.resample_counts).

  label 'any'   : 1 = in SL_SynLethDB_experimental.txt, 0 = screened non-SL
                  and not SL (OR-collapse). Reproduces the published
                  ci_sldb_agreement.csv all_pairs rows exactly (same unit
                  hash), which is checked.
  label 'K562'  : cell_line_vocab.pair_label_mask head K562 (the head the
                  K562 screens are scored with): 1 = SL in K562, 0 =
                  screened non-SL in K562 and not SL in K562; pairs not
                  screened in K562 are left out.
  label 'RPE1'  : (Corn's line) SynLethDB lists no SL pair in RPE1, so this
                  label is single-class wherever it exists; reported as
                  counts only.

Usage: python -I sldb_agreement_cellline.py
Writes sldb_agreement_cellline.csv (all arms: the screen rows and SynLethDB
labels do not depend on the PCA arm).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sl_comparison"))
# appended, not prepended: siamese_sl/ has its own retrieval_metrics.py
# (torch), which must not shadow sl_comparison's
sys.path.append(str(REPO / "siamese_sl"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bootstrap_ci as bc  # noqa: E402
import cell_line_vocab as clv  # noqa: E402
import external_transfer as et  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREENS = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
EVAL = (REPO / "siamese_sl" / "results" /
        "var_pd_pooled__best_cat_geneformer_scgpt_go_go2vec_kg_complex_"
        "ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3" /
        "eval_finetuning_cv1")
N_BOOT, SEED, ALPHA = 1000, 42, 0.05


def sidecar(path):
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    a, b = d.entrez1.str.strip(), d.entrez2.str.strip()
    key = np.where(a <= b, a + "|" + b, b + "|" + a)
    out = {}
    for k, cl in zip(key, d.cell_lines):
        out[k] = (out[k] + ";" + cl) if k in out else cl
    return out


def boot(keys, y, t, unit):
    W = bc.resample_counts(len(keys), N_BOOT, SEED, (unit,))
    cp, cr = bc.bootstrap_classification(y.astype(np.int64), {"t": t}, W)
    sp, sr = bc.bootstrap_spearman(t, {"lab": y.astype(np.float64)}, W)
    au = (cp["t"]["auroc"], *bc.percentile_ci(cr["t"]["auroc"], ALPHA)[:2],
          float(np.mean(cr["t"]["auroc"] <= 0.5)))
    rho = (sp["lab"], *bc.percentile_ci(sr["lab"], ALPHA)[:2],
           float(np.mean(sr["lab"] <= 0)))
    return au, rho


def main():
    pos = sidecar(REPO / "data" / "SL_SynLethDB_experimental.sources.tsv")
    neg = sidecar(REPO / "data" /
                  "SL_SynLethDB_experimental_negatives.sources.tsv")
    pub = pd.read_csv(REPO / "sl_comparison" / "results" / "bootstrap" /
                      "ci_sldb_agreement.csv")
    k562 = clv.HEADS.index("K562")
    rows = []
    for screen in SCREENS:
        df = et.read_pair_csv(EVAL / f"{screen}_predictions.csv")
        lab = df.loc[df.sldb_overlap].sort_values("key").reset_index(
            drop=True)
        keys = lab["key"].to_numpy()
        t = lab["target"].to_numpy(dtype=np.float64)
        is_sl = np.array([k in pos for k in keys])
        is_ns = np.array([k in neg for k in keys])
        assert (is_sl | is_ns).all(), "sldb_overlap row in neither file"
        # ---- general (any line) label: reproduce the published row ----
        y_any = is_sl.astype(int)
        unit = bc.unit_hash(screen, "all_pairs", "any", "sldb_agreement",
                            et.matched_digest(keys))
        au, rho = boot(keys, y_any, t, unit)
        p = pub[(pub.screen == screen) & (pub.scope == "all_pairs")]
        pa = p[p.metric == "auroc_target_vs_label"].iloc[0]
        ps = p[p.metric == "spearman_target_vs_label"].iloc[0]
        for got, want in ((au[0], pa.point), (au[1], pa.ci_low),
                          (au[2], pa.ci_high), (rho[0], ps.point),
                          (rho[1], ps.ci_low), (rho[2], ps.ci_high)):
            assert abs(got - want) < 1e-9, (screen, got, want)
        rows.append({"screen": screen, "label": "any cell line (OR)",
                     "n_labelled": len(keys), "n_pos": int(y_any.sum()),
                     "n_neg": int((1 - y_any).sum()),
                     "auroc": au[0], "auroc_ci_low": au[1],
                     "auroc_ci_high": au[2],
                     "auroc_p2_vs_0.5": bc_p2(au[3]),
                     "spearman": rho[0], "spearman_ci_low": rho[1],
                     "spearman_ci_high": rho[2],
                     "spearman_p2_vs_0": bc_p2(rho[3]),
                     "source": "reproduces bootstrap/ci_sldb_agreement.csv "
                               "all_pairs (checked)"})
        # ---- K562 head label ----
        lm = [clv.pair_label_mask(pos.get(k, ""), neg.get(k, ""), s, n)
              for k, s, n in zip(keys, is_sl, is_ns)]
        lab_h = np.array([l[k562] for l, m in lm])
        msk_h = np.array([m[k562] for l, m in lm]).astype(bool)
        kk, yy, tt = keys[msk_h], lab_h[msk_h].astype(int), t[msk_h]
        rec = {"screen": screen, "label": "K562 head (cell-line specific)",
               "n_labelled": int(len(kk)), "n_pos": int(yy.sum()),
               "n_neg": int((1 - yy).sum())}
        if len(kk) >= 3 and 0 < yy.sum() < len(yy):
            unit = bc.unit_hash(screen, "all_pairs", "any",
                                "sldb_agreement_K562", et.matched_digest(kk))
            au, rho = boot(kk, yy, tt, unit)
            rec.update({"auroc": au[0], "auroc_ci_low": au[1],
                        "auroc_ci_high": au[2],
                        "auroc_p2_vs_0.5": bc_p2(au[3]),
                        "spearman": rho[0], "spearman_ci_low": rho[1],
                        "spearman_ci_high": rho[2],
                        "spearman_p2_vs_0": bc_p2(rho[3])})
        rec["source"] = "new (cell_line_vocab.pair_label_mask, K562 head)"
        rows.append(rec)
        # ---- labels in other lines than K562 (the complement) ----
        other = ~msk_h
        rows.append({"screen": screen,
                     "label": "not screened in K562 (other lines only)",
                     "n_labelled": int(other.sum()),
                     "n_pos": int(is_sl[other].sum()),
                     "n_neg": int((~is_sl[other]).sum()),
                     "source": "counts only"})
        # ---- RPE1 (Corn's line): counts ----
        rpe_pos = sum("RPE1" in clv.parse_cell_field(pos.get(k, ""))
                      for k in keys if k in pos)
        rpe_neg = sum("RPE1" in clv.parse_cell_field(neg.get(k, ""))
                      for k in keys if k in neg)
        rows.append({"screen": screen, "label": "RPE1 (raw token)",
                     "n_pos": int(rpe_pos), "n_neg": int(rpe_neg),
                     "n_labelled": int(rpe_pos + rpe_neg),
                     "source": "counts only: SynLethDB has no SL pair in "
                               "RPE1"})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "sldb_agreement_cellline.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.drop(columns=["source"]).to_string())


def bc_p2(p, b=N_BOOT):
    return float(min(1.0, 2 * max(min(p, 1 - p), 1.0 / (b + 1))))


if __name__ == "__main__":
    main()
