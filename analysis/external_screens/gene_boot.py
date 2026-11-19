#!/usr/bin/env python3
"""How optimistic is the pair bootstrap? A gene-level (dyadic cluster)
bootstrap of the same statistics, for the score sources available locally.

Per (screen, split, row set) and resample b, the GENES of the rows are drawn
with replacement (multiplicities m_g); a pair (i, j) gets weight m_i * m_j,
so a pair whose genes were not drawn drops out and pairs sharing a resampled
gene move together (two-way / dyadic cluster bootstrap). The weighted
Spearman is bootstrap_ci.bootstrap_spearman's (average ranks of the expanded
resample). B = 1000, seed 42 (own streams: default_rng([42, crc32 key, b])).

Reported next to the published pair bootstrap SE / CI of the same quantity:
  zero-shot fold-mean Spearman on the matched rows (Table 5) of siamese_sl,
  siamese_sl_pd_pooled, DegreePrior, and siamese_sl - DegreePrior;
  Full, Scratch and Full - Scratch of siamese_sl on all_sldb_filtered
  (Table 6).
SLMGAE / SLGNN / NSF4SL cannot be re-scored here (cluster-only matrices).

Usage: python -I gene_boot.py [ARM]   -> out_<arm>/gene_bootstrap.csv
"""
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sl_comparison"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bootstrap_ci as bc  # noqa: E402
import external_transfer as et  # noqa: E402

HERE = Path(__file__).resolve().parent
COMBO = ("best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_"
         "mashup_esm1b_prot_t5_text_embed_cv3")
B, SEED = 1000, 42


def gene_weights(g1, g2, key):
    genes, inv = np.unique(np.concatenate([g1, g2]), return_inverse=True)
    i1, i2 = inv[:len(g1)], inv[len(g1):]
    W = np.empty((B, len(g1)), dtype=np.uint16)
    for b in range(B):
        rng = np.random.default_rng([SEED, key, b])
        m = np.bincount(rng.integers(0, len(genes), size=len(genes)),
                        minlength=len(genes))
        W[b] = (m[i1] * m[i2]).astype(np.uint16)
    return W, len(genes)


def summarise(pts, reps, ks):
    p = np.array([pts[k] for k in ks])
    r = np.vstack([reps[k] for k in ks]).mean(axis=0)
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(p.mean()), float(r.std(ddof=1)), float(lo), float(hi), r


def main(arm="none"):
    root = REPO / "siamese_sl" / "results"
    cmp = REPO / "sl_comparison" / "results"
    if arm != "none":
        root, cmp = root / f"pca_{arm}", cmp / f"pca_{arm}"
    dirs = {"siamese_sl": root / COMBO,
            "siamese_sl_pd_pooled": root / f"var_pd_pooled__{COMBO}"}
    pub = pd.read_csv(cmp / "bootstrap" / "ci_external.csv",
                      dtype={"fold": str})
    pubd = pd.read_csv(cmp / "bootstrap" / "ci_external_diffs.csv",
                       dtype={"fold": str})
    pub = pub[pub.fold == "mean"].set_index(["model", "method", "screen",
                                             "split", "row_set"])
    pubd = pubd[pubd.fold == "mean"].set_index(
        ["model", "method", "peer", "peer_method", "screen", "split",
         "row_set"])
    folds_dir = REPO / "sl_comparison" / "folds"
    gi = et.load_gene_index(folds_dir / "genes.txt")
    deg = dict(et.degree_fold_scorers(folds_dir, "cv3", len(gi)))
    rows = []
    for screen in ("Adamson", "AdamsonGI", "Corn", "Gilbert"):
        for split in ("cv1", "cv2", "cv3"):
            t0 = time.time()
            # ---- zero-shot, matched rows ----
            vec = {}
            for name, d in dirs.items():
                df = et.read_pair_csv(d / f"eval_finetuning_{split}" /
                                      f"{screen}_predictions.csv")
                m = et.row_mask(df, gi, "matched")
                sub = df.loc[m].sort_values("key").reset_index(drop=True)
                for k in range(5):
                    vec[(name, k)] = sub[f"logit_fold{k}"].to_numpy(float)
            i, j = et._reference_indices(sub, gi)
            for k in range(5):
                vec[("DegreePrior", k)] = np.asarray(deg[k](i, j), float)
            W, ng = gene_weights(sub.gene1_entrez.to_numpy(str),
                                 sub.gene2_entrez.to_numpy(str),
                                 bc.unit_hash("gene_boot", screen, split,
                                              "matched"))
            pts, reps = bc.bootstrap_spearman(sub.target.to_numpy(float),
                                              vec, W)
            res = {}
            for model in ("siamese_sl", "siamese_sl_pd_pooled",
                          "DegreePrior"):
                ks = [(model, k) for k in range(5)]
                pt, se, lo, hi, r = summarise(pts, reps, ks)
                res[model] = r
                pr = pub.loc[(model, "zero_shot_raw", screen, split,
                              "matched")]
                rows.append({"arm": arm, "table": "5", "screen": screen,
                             "split": split, "row_set": "matched",
                             "quantity": f"{model} zero-shot",
                             "n_rows": len(sub), "n_genes": ng,
                             "point": pt, "pair_se": pr.boot_se,
                             "pair_ci": f"[{pr.ci_low:.3f}, {pr.ci_high:.3f}]",
                             "gene_se": se,
                             "gene_ci": f"[{lo:.3f}, {hi:.3f}]",
                             "se_ratio": se / pr.boot_se,
                             "gene_ci_excludes_0": bool(lo > 0 or hi < 0)})
            dr = res["siamese_sl"] - res["DegreePrior"]
            pd_ = pubd.loc[("siamese_sl", "zero_shot_raw", "DegreePrior",
                            "zero_shot_raw", screen, split, "matched")]
            lo, hi = np.percentile(dr, [2.5, 97.5])
            rows.append({"arm": arm, "table": "5", "screen": screen,
                         "split": split, "row_set": "matched",
                         "quantity": "siamese_sl - DegreePrior",
                         "n_rows": len(sub), "n_genes": ng,
                         "point": pd_.point, "pair_se": pd_.boot_se,
                         "pair_ci": f"[{pd_.ci_low:.3f}, {pd_.ci_high:.3f}]",
                         "gene_se": float(dr.std(ddof=1)),
                         "gene_ci": f"[{lo:.3f}, {hi:.3f}]",
                         "se_ratio": float(dr.std(ddof=1)) / pd_.boot_se,
                         "gene_ci_excludes_0": bool(lo > 0 or hi < 0)})
            # ---- Table 6: siamese_sl Full / Scratch, all_sldb_filtered ----
            d = dirs["siamese_sl"] / f"eval_finetuning_{split}"
            vec = {}
            ref = None
            for mode in ("Full", "Scratch"):
                fdf = et.read_pair_csv(
                    d / f"{screen}_finetuned_predictions_{mode}.csv")
                fdf = fdf.loc[~fdf.sldb_overlap].sort_values(
                    "key").reset_index(drop=True)
                if ref is None:
                    ref = fdf
                assert (fdf.key.to_numpy() == ref.key.to_numpy()).all()
                for k in range(5):
                    vec[(mode, k)] = fdf[f"pred_fold{k}"].to_numpy(float)
            W, ng = gene_weights(ref.gene1_entrez.to_numpy(str),
                                 ref.gene2_entrez.to_numpy(str),
                                 bc.unit_hash("gene_boot", screen, split,
                                              "all_sldb_filtered"))
            pts, reps = bc.bootstrap_spearman(ref.target.to_numpy(float),
                                              vec, W)
            res = {}
            for mode in ("Full", "Scratch"):
                pt, se, lo, hi, r = summarise(pts, reps,
                                              [(mode, k) for k in range(5)])
                res[mode] = r
                pr = pub.loc[("siamese_sl", mode, screen, split,
                              "all_sldb_filtered")]
                assert abs(pt - pr.point) < 1e-9
                rows.append({"arm": arm, "table": "6", "screen": screen,
                             "split": split, "row_set": "all_sldb_filtered",
                             "quantity": f"siamese_sl {mode}",
                             "n_rows": len(ref), "n_genes": ng,
                             "point": pt, "pair_se": pr.boot_se,
                             "pair_ci": f"[{pr.ci_low:.3f}, {pr.ci_high:.3f}]",
                             "gene_se": se,
                             "gene_ci": f"[{lo:.3f}, {hi:.3f}]",
                             "se_ratio": se / pr.boot_se,
                             "gene_ci_excludes_0": bool(lo > 0 or hi < 0)})
            dr = res["Full"] - res["Scratch"]
            pd_ = pubd.loc[("siamese_sl", "Full", "siamese_sl", "Scratch",
                            screen, split, "all_sldb_filtered")]
            lo, hi = np.percentile(dr, [2.5, 97.5])
            rows.append({"arm": arm, "table": "6", "screen": screen,
                         "split": split, "row_set": "all_sldb_filtered",
                         "quantity": "siamese_sl Full - Scratch",
                         "n_rows": len(ref), "n_genes": ng,
                         "point": pd_.point, "pair_se": pd_.boot_se,
                         "pair_ci": f"[{pd_.ci_low:.3f}, {pd_.ci_high:.3f}]",
                         "gene_se": float(dr.std(ddof=1)),
                         "gene_ci": f"[{lo:.3f}, {hi:.3f}]",
                         "se_ratio": float(dr.std(ddof=1)) / pd_.boot_se,
                         "gene_ci_excludes_0": bool(lo > 0 or hi < 0)})
            print(f"{screen}/{split} [{time.time() - t0:.1f}s]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"out_{arm}" / "gene_bootstrap.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(out[["table", "screen", "split", "quantity", "n_rows", "n_genes",
               "point", "pair_ci", "gene_ci", "se_ratio",
               "gene_ci_excludes_0"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
