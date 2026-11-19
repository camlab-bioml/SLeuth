#!/usr/bin/env python3
"""Diagnostic for the sign of the zero-shot correlations (8oBN: 'most
correlations are close to 0 or negative'): on each screen's cv1 matched rows,
per fold model, Spearman of the SLeuth logit with
  * the screen target (the Table 5 number, as a check),
  * DegreePrior (log1p deg_i + log1p deg_j over that fold's SynLethDB
    training positives: how much the score is a gene-degree prior),
  * the additive single-gene part of the target (least-squares a_i + a_j
    fitted on ALL scored screen pairs, as paper_tables F4; for Adamson this
    is essentially single-gene growth/essentiality),
and Spearman(DegreePrior, additive part).
Usage: python -I diag_zero_shot.py [ARM]
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sl_comparison"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.linalg import lsqr  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import external_transfer as et  # noqa: E402

HERE = Path(__file__).resolve().parent
COMBO = ("best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_"
         "mashup_esm1b_prot_t5_text_embed_cv3")


def additive_fit(df):
    y = df.target.to_numpy(float)
    g = np.concatenate([df.gene1_entrez.astype(str), df.gene2_entrez.astype(
        str)])
    uniq, inv = np.unique(g, return_inverse=True)
    n = len(y)
    X = coo_matrix((np.ones(2 * n), (np.tile(np.arange(n), 2), inv)),
                   shape=(n, len(uniq))).tocsr()
    a = lsqr(X, y, atol=1e-12, btol=1e-12, iter_lim=10000)[0]
    return pd.Series(X @ a, index=df["key"].to_numpy())


def main(arm="none"):
    root = REPO / "siamese_sl" / "results"
    if arm != "none":
        root = root / f"pca_{arm}"
    dirs = {"siamese_sl": root / COMBO,
            "siamese_sl_pd_pooled": root / f"var_pd_pooled__{COMBO}"}
    folds_dir = REPO / "sl_comparison" / "folds"
    gi = et.load_gene_index(folds_dir / "genes.txt")
    deg = dict(et.degree_fold_scorers(folds_dir, "cv3", len(gi)))
    rows = []
    for screen in ("Adamson", "AdamsonGI", "Corn", "Gilbert"):
        for name, d in dirs.items():
            df = et.read_pair_csv(d / "eval_finetuning_cv1" /
                                  f"{screen}_predictions.csv")
            add = additive_fit(df)
            m = et.row_mask(df, gi, "matched")
            sub = df.loc[m].sort_values("key").reset_index(drop=True)
            i, j = et._reference_indices(sub, gi)
            t = sub.target.to_numpy(float)
            a = add.reindex(sub.key).to_numpy(float)
            for k in range(5):
                s = sub[f"logit_fold{k}"].to_numpy(float)
                dg = np.asarray(deg[k](i, j), float)
                rows.append({"arm": arm, "screen": screen, "model": name,
                             "fold": k, "n": len(sub),
                             "rho_logit_target": spearmanr(s, t)[0],
                             "rho_logit_degree": spearmanr(s, dg)[0],
                             "rho_logit_additive": spearmanr(s, a)[0],
                             "rho_degree_additive": spearmanr(dg, a)[0],
                             "rho_degree_target": spearmanr(dg, t)[0],
                             "rho_additive_target": spearmanr(a, t)[0]})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"out_{arm}" / "diag_zero_shot_cv1_matched.csv",
               index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.groupby(["screen", "model"])[[c for c in out.columns
                                            if c.startswith("rho")]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
