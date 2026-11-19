#!/usr/bin/env python3
"""
Plan item 11, follow-up analyses on scope all_pairs (same rows / target /
label as agreement_split.py, which must have run first):

 (a) same-map cell-line contrast: K562 SL vs Jurkat-only SL, both with
     SynLethDB evidence from PMID 30033366 (Horlbeck 2018) or 31300006
     (GEMINI): same CRISPRi technology and source, different cell line;
     pair bootstrap with the published resamples (as agreement_split.py).
 (b) descriptive: median raw phenotype (outcome_raw) per subset, and the
     share of pairs whose two genes are both genes of the Gilbert screen
     (the Horlbeck 2018 gene set, selected for K562 growth phenotypes).
 (c) gene-level (vertex) bootstrap of the main subsets and of the primary
     contrast: genes resampled with replacement, pair weight = m_i * m_j
     (m = a gene's multiplicity in the resample), B = 1000, seed 42 with its
     own stream; NOT part of the revision's stated rules (those use the
     pair bootstrap), a robustness check of the pair-independence caveat.
Read-only on the repo. <venv312>/bin/python -B agreement_extra.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import agreement_split as A  # same directory; its main() is not run

bc, et, clv = A.bc, A.et, A.clv
REPO, OUT = A.REPO, A.OUT


def labelled_scope(screen, sia_dir, splits, gene_index, scope_want="all_pairs"):
    pairs, reference = None, {}
    for split in splits:
        df = et.read_pair_csv(sia_dir / f"eval_finetuning_{split}" /
                              f"{screen}_predictions.csv")
        if pairs is None:
            pairs = df
        m = et.row_mask(df, gene_index, et.ROW_MATCHED_INCL)
        reference[(screen, split)] = df.loc[m].sort_values(
            "key").reset_index(drop=True)
    for scope, sp, sub in et.screen_scopes(screen, pairs, reference,
                                           gene_index, splits):
        if scope != scope_want:
            continue
        lab = sub.loc[sub["sldb_overlap"].to_numpy(dtype=bool)]
        lab = lab.sort_values("key").reset_index(drop=True)
        return scope, sp, lab, pairs
    raise KeyError(scope_want)


def main() -> int:
    t0 = time.time()
    prov = bc.parse_external_provenance(A.EXT / "summary.md")
    sia_dir = bc.repo_path(prov["siamese"][0][1])
    zs_pub = pd.read_csv(A.EXT / "zero_shot_per_fold.csv")
    screens = list(dict.fromkeys(zs_pub["screen"].astype(str)))
    splits = list(dict.fromkeys(zs_pub["split"].astype(str)))
    gene_index = et.load_gene_index(A.FOLDS / "genes.txt")
    pos_keys = et.load_pair_keys(REPO / "data/SL_SynLethDB_experimental.txt")
    pos_sc = A.read_sidecar(REPO / "data/SL_SynLethDB_experimental.sources.tsv")
    neg_sc = A.read_sidecar(
        REPO / "data/SL_SynLethDB_experimental_negatives.sources.tsv")
    sl_lines = {k: clv.parse_cell_field(v) for k, v in pos_sc["cell_lines"].items()}
    ns_lines = {k: clv.parse_cell_field(v) for k, v in neg_sc["cell_lines"].items()}
    sl_pmids = {k: {x.strip() for x in v.split(";") if x.strip()}
                for k, v in pos_sc["pubmed_ids"].items()}
    gil = et.read_pair_csv(sia_dir / "eval_finetuning_cv1" / "Gilbert_predictions.csv")
    gil_genes = set(gil["gene1_entrez"].astype(str)) | set(gil["gene2_entrez"].astype(str))

    rows_a, rows_b, rows_c, diffs_c = [], [], [], []
    for screen in screens:
        own = A.OWN_LINE[screen]
        scope, sp, lab, _pairs = labelled_scope(screen, sia_dir, splits, gene_index)
        keys = lab["key"].to_numpy()
        n = len(keys)
        y = np.fromiter((k in pos_keys for k in keys), bool, n)
        t = lab["target"].to_numpy(dtype=np.float64)
        raw = lab["outcome_raw"].to_numpy(dtype=np.float64)
        L = [sl_lines.get(k, set()) if yy else set() for k, yy in zip(keys, y)]
        NL = [ns_lines.get(k, set()) for k in keys]
        pm = [sl_pmids.get(k, set()) if yy else set() for k, yy in zip(keys, y)]
        nl = np.array([len(s) for s in L])
        own_sl = y & np.array([own in s for s in L])
        other_sl = y & (nl > 0) & ~own_sl
        unk_sl = y & (nl == 0)
        mapsrc = y & np.array([bool(s & {A.PMID_HORLBECK, A.PMID_GEMINI}) for s in pm])
        jurkat_only = y & np.array([s == {"JURKAT"} for s in L])
        neg = ~y
        neg_own = neg & np.array([own in s for s in NL])
        g1 = lab["gene1_entrez"].astype(str).to_numpy()
        g2 = lab["gene2_entrez"].astype(str).to_numpy()
        both_gil = np.array([a in gil_genes and b in gil_genes for a, b in zip(g1, g2)])
        masks = {"own_line": own_sl, "other_line": other_sl,
                 "unknown_line": unk_sl, "all": y,
                 "k562_same_map": own_sl & mapsrc,
                 "jurkat_only_same_map": jurkat_only & mapsrc,
                 "other_line_not_same_map": other_sl & ~mapsrc}

        # ---- (b) descriptive ----
        for name, m in list(masks.items()) + [("nonSL (all negatives)", neg),
                                              ("nonSL screened in own line", neg_own)]:
            if not m.any():
                rows_b.append({"screen": screen, "subset": name, "n": 0})
                continue
            rows_b.append({
                "screen": screen, "subset": name, "n": int(m.sum()),
                "median_outcome_raw": float(np.median(raw[m])),
                "q25_outcome_raw": float(np.quantile(raw[m], 0.25)),
                "q75_outcome_raw": float(np.quantile(raw[m], 0.75)),
                "frac_both_genes_in_gilbert_screen": float(both_gil[m].mean()),
            })

        # ---- (a) same-map contrast, published resamples ----
        W = bc.resample_counts(n, A.N_BOOT, A.SEED,
                               (bc.unit_hash(screen, scope, sp, "sldb_agreement",
                                             et.matched_digest(keys)),))
        reps = {}
        for name in ("k562_same_map", "jurkat_only_same_map",
                     "other_line_not_same_map"):
            cols = masks[name] | neg
            n_p = int(masks[name].sum())
            rec = {"screen": screen, "subset": name, "n_pos": n_p,
                   "n_neg": int(neg.sum())}
            if n_p == 0:
                rows_a.append(rec)
                continue
            a_pt, a_rep = A.auroc_boot(y[cols], t[cols], W[:, cols])
            s_pt, s_rep = A.spearman_boot(y[cols], t[cols], W[:, cols])
            sa, ss = bc.boot_stats(a_pt, a_rep, A.ALPHA), bc.boot_stats(s_pt, s_rep, A.ALPHA)
            reps[name] = (a_pt, a_rep)
            rows_a.append({**rec, "reportable": n_p >= A.MIN_POS,
                           "auroc": a_pt, "auroc_ci_low": sa["ci_low"],
                           "auroc_ci_high": sa["ci_high"], "spearman": s_pt,
                           "spearman_ci_low": ss["ci_low"],
                           "spearman_ci_high": ss["ci_high"]})
        if "k562_same_map" in reps and "jurkat_only_same_map" in reps:
            d = bc.diff_stats(reps["k562_same_map"][0] - reps["jurkat_only_same_map"][0],
                              reps["k562_same_map"][1] - reps["jurkat_only_same_map"][1],
                              A.ALPHA)
            rows_a.append({"screen": screen,
                           "subset": "DIFF k562_same_map - jurkat_only_same_map",
                           "n_pos": int(masks["k562_same_map"].sum()),
                           "n_neg": int(masks["jurkat_only_same_map"].sum()),
                           "auroc": d["point"], "auroc_ci_low": d["ci_low"],
                           "auroc_ci_high": d["ci_high"],
                           "p_boot_le0": d["p_boot_le0"],
                           "p_two_sided": A.p_two_sided(d["p_boot_le0"]),
                           "reportable": bool(masks["jurkat_only_same_map"].sum() >= A.MIN_POS)})

        # ---- (c) gene-level (vertex) bootstrap ----
        genes = np.unique(np.concatenate([g1, g2]))
        gidx = {g: i for i, g in enumerate(genes)}
        i1 = np.array([gidx[a] for a in g1])
        i2 = np.array([gidx[b] for b in g2])
        G = len(genes)
        Wg = np.empty((A.N_BOOT, n), dtype=np.uint16)
        h = bc.unit_hash(screen, scope, sp, "gene_bootstrap")
        for b in range(A.N_BOOT):
            rng = np.random.default_rng([A.SEED, h, b])
            mlt = np.bincount(rng.integers(0, G, size=G), minlength=G).astype(np.int64)
            w = mlt[i1] * mlt[i2]
            if w.max() >= np.iinfo(np.uint16).max:
                raise OverflowError("gene-bootstrap weight overflows uint16")
            Wg[b] = w
        greps = {}
        for name in ("all", "own_line", "other_line", "unknown_line",
                     "k562_same_map", "jurkat_only_same_map"):
            cols = masks[name] | neg
            n_p = int(masks[name].sum())
            rec = {"screen": screen, "subset": name, "n_pos": n_p,
                   "n_neg": int(neg.sum()), "n_genes": G}
            if n_p < A.MIN_POS:
                rows_c.append(rec)
                continue
            a_pt, a_rep = A.auroc_boot(y[cols], t[cols], Wg[:, cols])
            s_pt, s_rep = A.spearman_boot(y[cols], t[cols], Wg[:, cols])
            sa, ss = bc.boot_stats(a_pt, a_rep, A.ALPHA), bc.boot_stats(s_pt, s_rep, A.ALPHA)
            greps[name] = (a_pt, a_rep, s_pt, s_rep)
            pa = float(np.mean(a_rep[np.isfinite(a_rep)] <= 0.5))
            rows_c.append({**rec, "auroc": a_pt, "auroc_ci_low": sa["ci_low"],
                           "auroc_ci_high": sa["ci_high"], "auroc_boot_se": sa["boot_se"],
                           "auroc_n_finite": sa["n_finite"],
                           "p_two_sided_vs_chance": A.p_two_sided(pa),
                           "spearman": s_pt, "spearman_ci_low": ss["ci_low"],
                           "spearman_ci_high": ss["ci_high"],
                           "spearman_boot_se": ss["boot_se"]})
        for ca, cb in (("own_line", "other_line"),
                       ("k562_same_map", "jurkat_only_same_map")):
            if ca not in greps or cb not in greps:
                continue
            for met, ix in (("auroc", 0), ("spearman", 2)):
                d = bc.diff_stats(greps[ca][ix] - greps[cb][ix],
                                  greps[ca][ix + 1] - greps[cb][ix + 1],
                                  A.ALPHA)
                diffs_c.append({"screen": screen, "contrast": f"{ca} - {cb}",
                                "metric": met, **d,
                                "p_two_sided": A.p_two_sided(d["p_boot_le0"])})
        print(f"{screen}: n={n} genes={G} [{time.time() - t0:.0f}s]", flush=True)

    ra, rb, rc, dc = (pd.DataFrame(x) for x in (rows_a, rows_b, rows_c, diffs_c))
    # Holm across the screens for the gene-bootstrap primary contrast
    for con in dc.contrast.unique():
        for met in ("auroc", "spearman"):
            m = (dc.metric == met) & (dc.contrast == con)
            dc.loc[m, "p_holm"] = A.holm(dc.loc[m, "p_two_sided"].to_numpy())
    sel = ra["subset"].str.startswith("DIFF") & ra["reportable"].astype(bool)
    ra.loc[sel, "p_holm"] = A.holm(ra.loc[sel, "p_two_sided"].to_numpy())
    ra.to_csv(OUT / "extra_same_map_contrast.csv", index=False)
    rb.to_csv(OUT / "extra_descriptive.csv", index=False)
    rc.to_csv(OUT / "extra_gene_bootstrap_points.csv", index=False)
    dc.to_csv(OUT / "extra_gene_bootstrap_diffs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, df in (("(a) same-map contrast", ra), ("(b) descriptive", rb),
                     ("(c) gene bootstrap", rc), ("(c) gene bootstrap diffs", dc)):
        print(f"\n== {name}\n{df.to_string()}")
    print(json.dumps({"runtime_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
