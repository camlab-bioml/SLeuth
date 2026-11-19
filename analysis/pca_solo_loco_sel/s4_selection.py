"""Task 4: modality selection in run 20261113T113339-1225758.

(a) per tier x fold winners from best_per_category.json (both arms);
(b) verify each fold's selection split (selection_split_<cv>_fold<k>.npy,
    full-embedding gene indices) is inside that fold's TRAIN pairs and
    disjoint from its val and test pairs, and (cv2/cv3) that no test gene
    appears in it; mapping: embedding index -> Entrez via
    data/all_genes_esm2.genes.txt (the .pt gene order), fold index -> Entrez
    via sl_comparison/folds/genes.txt (read only);
(c) how well the selection-split score predicts the fold's test score,
    across the 16 single-embedding runs (Spearman per fold), vs the val split;
(d) selected combo - all-16 model: pair bootstrap (ci_tiers_diffs.csv,
    fold = mean) with two-sided p, and fold-paired tests on graded per-fold
    metrics. Holm family: one (arm, metric, tier): the 2 variants.
(e) all-16 models vs every non-SLeuth row: fold-paired graded tests,
    Holm family: one (arm, variant, metric, tier): the 6 non-SLeuth rows.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import *

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 200)

# ------------------------------------------------------------- (a) winners
sel = {arm: json.loads((SIA[arm] / "best_per_category.json").read_text()) for arm in ["none", "joint"]}
same = {cv: {c: sel["none"][cv][c]["per_fold"] == sel["joint"][cv][c]["per_fold"] for c in CATS}
        for cv in CVS}
print("arms none and joint picked identical per-fold winners:",
      all(all(v.values()) for v in same.values()))
print("_selection_meta (arm none):")
meta = sel["none"]["_selection_meta"]
for k in ["selection_mode", "selection_variant", "selection_metric", "selection_basis",
          "selection_frac", "selection_seed", "n_scored_on_val", "n_scored_on_test",
          "unranked_not_generated", "step1_pca", "run_token", "rationale"]:
    print(f"  {k}: {meta.get(k)}")
wrows = []
for cv in CVS:
    for c in CATS:
        e = sel["none"][cv][c]
        for k in range(5):
            w = e["per_fold"][str(k)]
            wrows.append({"tier": TIER[cv], "cv": cv, "category": c, "fold": k,
                          "winner": w["type"], "sel_score": w["score"]})
W = pd.DataFrame(wrows)
W.to_csv(OUT / "s4_winners_per_fold.csv", index=False)
piv = W.pivot_table(index=["tier", "category"], columns="fold", values="winner", aggfunc="first")
print("\n== per-fold winners (best_per_category.json per_fold; identical in both arms)")
print(piv.to_string())

# ------------------------------------------- (b) selection split disjointness
emb_genes = (REPO / "data/all_genes_esm2.genes.txt").read_text().split()
fold_genes = (REPO / "sl_comparison/folds/genes.txt").read_text().split()
f2e = {i: g for i, g in enumerate(fold_genes)}
e2g = {i: g for i, g in enumerate(emb_genes)}


def canon(pairs, m):
    return {tuple(sorted((m[int(a)], m[int(b)]))) for a, b in pairs}


drows = []
for cv in CVS:
    rd = combo_dirs("none")[("pd:pooled", "combo", cv)]
    for k in range(5):
        fd = REPO / f"sl_comparison/folds/{cv}/fold_{k}"
        part = {s: canon(np.concatenate([np.load(fd / f"{s}_pos.npy"), np.load(fd / f"{s}_neg.npy")]), f2e)
                for s in ["train", "val", "test"]}
        sp = canon(np.load(rd / f"selection_split_{cv}_fold{k}.npy"), e2g)
        test_genes = {g for p in part["test"] for g in p}
        train_genes = {g for p in part["train"] for g in p}
        sel_genes = {g for p in sp for g in p}
        n_train = len(part["train"])
        drows.append({"tier": TIER[cv], "fold": k, "n_sel_pairs": len(sp),
                      "n_train_pairs": n_train, "sel_frac": len(sp) / n_train,
                      "sel_in_train": len(sp & part["train"]), "sel_in_val": len(sp & part["val"]),
                      "sel_in_test": len(sp & part["test"]),
                      "sel_pairs_touching_test_gene": sum(1 for a, b in sp if a in test_genes or b in test_genes),
                      "sel_genes_not_in_train": len(sel_genes - train_genes),
                      "n_test_genes": len(test_genes)})
D = pd.DataFrame(drows)
D.to_csv(OUT / "s4_selection_split_disjointness.csv", index=False)
print("\n== selection split vs the fold's train / val / test pairs (counts of pairs)")
print(D.to_string(index=False))

# ---------------- (c) selection-split vs val vs test, across the 16 embeddings
crow = []
for v in ["pd:pooled", "pd:per_head"]:
    for cv in CVS:
        T = np.array([[self_folds(solo_dir(v, e, cv))[0].auprg_any.values[k] for e in PAPER16] for k in range(5)])
        S = np.array([[self_folds(solo_dir(v, e, cv))[0].sel_auprg_any.values[k] for e in PAPER16] for k in range(5)])
        V = np.array([[self_folds(solo_dir(v, e, cv))[0].val_auprg_any.values[k] for e in PAPER16] for k in range(5)])
        for k in range(5):
            rs = stats.spearmanr(S[k], T[k]).statistic
            rv = stats.spearmanr(V[k], T[k]).statistic
            crow.append({"variant": v, "tier": TIER[cv], "fold": k, "spearman_sel_vs_test": rs,
                         "spearman_val_vs_test": rv, "sel_range": S[k].max() - S[k].min(),
                         "test_range": T[k].max() - T[k].min()})
C = pd.DataFrame(crow)
C.to_csv(OUT / "s4_selection_score_vs_test.csv", index=False)
print("\n== Spearman (across the 16 single-embedding runs) of selection-split / val score with the fold's test AUPRG")
print(C.groupby(["variant", "tier"]).agg(sel_vs_test_mean=("spearman_sel_vs_test", "mean"),
                                         sel_vs_test_min=("spearman_sel_vs_test", "min"),
                                         sel_vs_test_max=("spearman_sel_vs_test", "max"),
                                         val_vs_test_mean=("spearman_val_vs_test", "mean"),
                                         val_vs_test_min=("spearman_val_vs_test", "min"),
                                         val_vs_test_max=("spearman_val_vs_test", "max"),
                                         sel_score_range=("sel_range", "mean"),
                                         test_score_range=("test_range", "mean")).round(3).to_string())

# per-category: what the selection picked vs the category's best on test, Tier III
print("\n== Tier III per category (pd:pooled): selection-split picks vs test-best / val-best member")
v, cv = "pd:pooled", "cv3"
for c in CATS:
    mem = [e for e in PAPER16 if CATEGORY[e] == c]
    Tm = {e: self_folds(solo_dir(v, e, cv))[0] for e in mem}
    picks = [sel["none"][cv][c]["per_fold"][str(k)]["type"] for k in range(5)]
    valp = [max(mem, key=lambda e: Tm[e].val_auprg_any.values[k]) for k in range(5)]
    testbest = max(mem, key=lambda e: Tm[e].auprg_any.mean())
    got = np.mean([Tm[picks[k]].auprg_any.values[k] for k in range(5)])
    gotv = np.mean([Tm[valp[k]].auprg_any.values[k] for k in range(5)])
    print(f"  {c:11s} picked {';'.join(picks):48s} -> test {got:.4f} | val-split picks {';'.join(valp):48s} -> {gotv:.4f}"
          f" | test-best member {testbest} {Tm[testbest].auprg_any.mean():.4f}")

# ---------------------------------- (d) combo - all-16 (bootstrap + fold-paired)
pairs = [("siamese_sl_pd_pooled", "siamese_sl_allmod_pd_pooled", "pd:pooled"),
         ("siamese_sl", "siamese_sl_allmod", "pd:per_head")]
brow = []
for arm in ["none", "joint"]:
    bd = pd.read_csv(CMP[arm] / "bootstrap/ci_tiers_diffs.csv", dtype={"fold": str})
    man = json.loads((CMP[arm] / "bootstrap/manifest.json").read_text())
    B = int(man["n_boot"])
    for a, b, v in pairs:
        for cv in CVS:
            ga, _ = graded_folds(arm, a, cv)
            gb, _ = graded_folds(arm, b, cv)
            for m in ["auprg", "ap", "auroc"]:
                x = bd[(bd.model == a) & (bd.peer == b) & (bd.cv == cv) & (bd.fold == "mean") & (bd.metric == m)]
                assert len(x) == 1, (arm, a, b, cv, m, len(x))
                x = x.iloc[0]
                fp = paired(ga[m].values - gb[m].values)
                brow.append({"arm": arm, "variant": v, "tier": TIER[cv], "metric": m,
                             "combo_mean": ga[m].mean(), "allmod_mean": gb[m].mean(),
                             "boot_point": x.point, "boot_ci_low": x.ci_low, "boot_ci_high": x.ci_high,
                             "p_boot_le0": x.p_boot_le0, "p2_boot": p2_boot(x.p_boot_le0, B), "B": B,
                             "fold_mean_diff": fp["mean_diff"], "fold_sd_diff": fp["sd_diff"],
                             "fold_ci95_low": fp["ci95_low"], "fold_ci95_high": fp["ci95_high"],
                             "p_t": fp["p_t"], "p_signflip": fp["p_signflip"], "n_pos": fp["n_pos"],
                             "src": rel(CMP[arm] / "bootstrap/ci_tiers_diffs.csv")})
BR = pd.DataFrame(brow)
BR["p2_boot_holm"] = np.nan
BR["p_t_holm"] = np.nan
for key, g in BR.groupby(["arm", "metric", "tier"]):
    BR.loc[g.index, "p2_boot_holm"] = holm(g.p2_boot.values)
    BR.loc[g.index, "p_t_holm"] = holm(g.p_t.values)
BR.to_csv(OUT / "s4_combo_minus_allmod.csv", index=False)
print("\n== selected combo - all-16 model (positive = selection better)")
print(BR[["arm", "variant", "tier", "metric", "combo_mean", "allmod_mean", "boot_point", "boot_ci_low",
          "boot_ci_high", "p_boot_le0", "p2_boot", "p2_boot_holm", "fold_sd_diff", "fold_ci95_low",
          "fold_ci95_high", "p_t", "p_t_holm", "p_signflip", "n_pos"]].round(4).to_string(index=False))
print("check: |boot point - fold mean diff| max =", float((BR.boot_point - BR.fold_mean_diff).abs().max()))

# --------------------------- (e) all-16 models vs non-SLeuth rows (fold-paired)
erow = []
for arm in ["none", "joint"]:
    for mrow, v in [("siamese_sl_allmod_pd_pooled", "pd:pooled"), ("siamese_sl_allmod", "pd:per_head")]:
        for cv in CVS:
            ga, _ = graded_folds(arm, mrow, cv)
            for m in ["auprg", "ap", "auroc"]:
                fam = []
                for peer in SHARED_ROWS:
                    gp, pp = graded_folds(arm, peer, cv)
                    r = paired(ga[m].values - gp[m].values)
                    r.update({"arm": arm, "variant": v, "tier": TIER[cv], "metric": m, "peer": peer,
                              "allmod_mean": ga[m].mean(), "peer_mean": gp[m].mean()})
                    fam.append(r)
                fam = pd.DataFrame(fam)
                fam["p_t_holm6"] = holm(fam.p_t.values)
                erow.append(fam)
E = pd.concat(erow, ignore_index=True)
E.to_csv(OUT / "s4_allmod_vs_peers_paired.csv", index=False)
print("\n== all-16 model - non-SLeuth rows (graded, fold-paired), AUPRG / AP / AUROC")
print(E[["arm", "variant", "tier", "metric", "peer", "allmod_mean", "peer_mean", "mean_diff", "ci95_low",
         "ci95_high", "p_t", "p_t_holm6", "p_signflip", "n_pos"]].round(4).to_string(index=False))

# per-row bootstrap CIs (fold mean) for the all-16 models, both arms
ci = []
for arm in ["none", "joint"]:
    c = pd.read_csv(CMP[arm] / "bootstrap/ci_tiers.csv", dtype={"fold": str})
    c = c[(c.fold == "mean") & c.metric.isin(["auprg", "ap", "auroc"])].assign(arm=arm)
    ci.append(c[["arm", "model", "cv", "metric", "point", "ci_low", "ci_high"]])
ci = pd.concat(ci)
ci.to_csv(OUT / "s4_row_bootstrap_cis.csv", index=False)
