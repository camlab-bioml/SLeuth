"""Task 1: PCA (arm joint, PCA of the concatenation at 0.8 variance) vs no PCA
(arm none). Fold-paired none - joint differences of the graded per-fold
AUPRG / AP / AUROC (row JSON summary.folds) for every SLeuth row of both arms.

Family for Holm: one (metric, tier): the 5 SLeuth rows (pooled combo,
cell-line combo, pooled all-16, cell-line all-16, cell-line cell_emb=0).
A second, wider family (one metric, all 15 row x tier tests) is reported too.
"""
import json
import numpy as np
import pandas as pd
from common import *

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)

# ---- (a) tab_pca.csv as written by paper_tables.py ------------------------
tp = pd.read_csv(CMP["none"] / "pca_arms/tab_pca.csv")
cols = ["pca_arm", "variant", "tier", "n_modalities", "same_modalities_as_reference",
        "input_dim_mean", "input_dim_min", "input_dim_max", "params_fold_mean",
        "auprg_mean", "auprg_sd", "ap_mean", "ap_sd", "auroc_mean", "auroc_sd",
        "delta_auprg", "delta_ap", "delta_auroc", "delta_auprg_paired_sd",
        "delta_auprg_se", "n_paired", "verdict", "best_arm"]
print("== tab_pca.csv (", rel(CMP["none"] / "pca_arms/tab_pca.csv"), ")")
print(tp[cols].to_string(index=False))

# ---- (b) all-embedding rows of both arms' tab_tiers -----------------------
for arm in ["none", "joint"]:
    tt = pd.read_csv(CMP[arm] / "paper_tables/tab_tiers.csv")
    sub = tt[tt.model.str.startswith("siamese_sl")]
    print(f"\n== {rel(CMP[arm] / 'paper_tables/tab_tiers.csv')} SLeuth rows")
    print(sub[["pca_arm", "tier", "model", "kind", "params_fold_mean", "auroc_mean",
               "auroc_sd", "ap_mean", "ap_sd", "auprg_mean", "auprg_sd"]].to_string(index=False))

# ---- (c) fold-paired none - joint ----------------------------------------
rows = []
for model in SLEUTH_ROWS:
    for cv in CVS:
        gn, pn = graded_folds("none", model, cv)
        gj, pj = graded_folds("joint", model, cv)
        for m in ["auprg", "ap", "auroc"]:
            d = gn[m].values - gj[m].values
            r = paired(d)
            r.update({"model": model, "label": ROW_LABEL[model], "cv": cv,
                      "tier": TIER[cv], "metric": m,
                      "none_mean": gn[m].mean(), "none_sd_pop": gn[m].std(ddof=0),
                      "joint_mean": gj[m].mean(), "joint_sd_pop": gj[m].std(ddof=0),
                      "none_folds": ";".join(f"{x:.4f}" for x in gn[m]),
                      "joint_folds": ";".join(f"{x:.4f}" for x in gj[m]),
                      "src_none": rel(pn), "src_joint": rel(pj)})
            rows.append(r)
res = pd.DataFrame(rows)
# Holm: family = (metric, tier) over the 5 SLeuth rows
res["p_t_holm_tier"] = np.nan
res["p_signflip_holm_tier"] = np.nan
for (m, cv), g in res.groupby(["metric", "cv"]):
    res.loc[g.index, "p_t_holm_tier"] = holm(g.p_t.values)
    res.loc[g.index, "p_signflip_holm_tier"] = holm(g.p_signflip.values)
# wider family: one metric, 15 tests
res["p_t_holm_metric15"] = np.nan
for m, g in res.groupby("metric"):
    res.loc[g.index, "p_t_holm_metric15"] = holm(g.p_t.values)
res.to_csv(OUT / "s1_pca_paired.csv", index=False)

show = ["label", "tier", "metric", "none_mean", "joint_mean", "mean_diff", "sd_diff",
        "se_diff", "ci95_low", "ci95_high", "t", "p_t", "p_t_holm_tier",
        "p_t_holm_metric15", "p_signflip", "n_pos"]
print("\n== fold-paired none - joint (graded per-fold), positive = no PCA better")
print(res[show].round(4).to_string(index=False))

# cross-check against tab_pca's delta / paired sd for AUPRG (arm - reference)
print("\n== cross-check vs tab_pca (delta_auprg = joint - none; paired sd = population)")
vmap = {"pd:pooled": "siamese_sl_pd_pooled", "pd:per_head": "siamese_sl"}
for _, r in tp[tp.pca_arm == "joint"].iterrows():
    mine = res[(res.model == vmap[r.variant]) & (res.cv == r.cv) & (res.metric == "auprg")].iloc[0]
    print(r.variant, r.cv, "tab_pca delta=%.6f mine=-%.6f | paired_sd tab=%.6f mine_pop=%.6f | se tab=%.6f mine=%.6f" % (
        r.delta_auprg, mine.mean_diff, r.delta_auprg_paired_sd, mine.sd_diff_pop,
        r.delta_auprg_se, mine.se_diff))

# ---- (d) per-arm bootstrap CIs of each SLeuth row (fold mean), both arms --
ci = []
for arm in ["none", "joint"]:
    c = pd.read_csv(CMP[arm] / "bootstrap/ci_tiers.csv", dtype={"fold": str})
    c = c[(c.fold == "mean") & c.model.str.startswith("siamese_sl") &
          c.metric.isin(["auprg", "ap", "auroc"])]
    c = c.assign(arm=arm)
    ci.append(c[["arm", "model", "cv", "metric", "point", "ci_low", "ci_high", "boot_se"]])
ci = pd.concat(ci)
ci.to_csv(OUT / "s1_pca_rowCIs.csv", index=False)
piv = ci[ci.metric == "auprg"].pivot_table(index=["model", "cv"], columns="arm",
                                           values=["point", "ci_low", "ci_high"])
print("\n== AUPRG fold-mean with pair-bootstrap 95% CI per arm (ci_tiers.csv, fold=mean)")
print(piv.round(4).to_string())

# ---- (e) widths / params of the four multimodal models per arm ----------
print("\n== encoder input width and trainable params (results.json fold_metrics)")
for arm in ["none", "joint"]:
    for key, rd in sorted(combo_dirs(arm).items()):
        r = json.loads((rd / "results.json").read_text())
        dims = [m["input_dim"] for m in r["fold_metrics"]]
        par = [m["total_params"] for m in r["fold_metrics"]]
        print(arm, key, "input_dim", min(dims), "-", max(dims), "mean %.0f" % np.mean(dims),
              "| params mean %.0f" % np.mean(par), "|", rel(rd))
