"""Task 3: leave-one-category-out (LOCO) ablations, both variants, both arms.

Per-fold self-scored auprg_any (and auroc_any, aupr_any) of each
ablate_no_<cat>_<cv> run (results.json) minus the reference combo's
(results.json of the marker-named combo; = graded to <= 1.4e-6, s0).
Holm family: one (arm, variant, tier): the 6 categories.
Also checks every ablation trained the combo's per-fold modalities minus the
category, with the arm's PCA, on the same selection split and run token.
"""
import json
import numpy as np
import pandas as pd
from common import *

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 500)
VARIANTS = ["pd:pooled", "pd:per_head"]


def fold_files(cfg):
    paths = [Path(p).name for p in cfg["embeddings_paths"]]
    fm = cfg.get("fold_modalities")
    if fm in (None, ""):
        return {k: paths for k in range(5)}
    if isinstance(fm, str):
        fm = json.loads(fm)
    return {int(k): [paths[i] for i in v] for k, v in fm.items()}


rows = []
checks = []
for arm in ["none", "joint"]:
    cd = combo_dirs(arm)
    tl = pd.read_csv(CMP[arm] / "paper_tables/tab_loco.csv")
    for v in VARIANTS:
        pre = "" if v == "pd:per_head" else "var_pd_pooled__"
        for cv in CVS:
            ref_dir = cd[(v, "combo", cv)]
            ref, rp, rr = self_folds(ref_dir)
            ref_ff = fold_files(rr["config"])
            fam = []
            for cat in CATS:
                ad = SIA[arm] / f"{pre}ablate_no_{cat}_{cv}"
                ab, ap_, ar = self_folds(ad)
                # provenance checks
                cfg = ar["config"]
                ab_ff = fold_files(cfg)
                exp_ff = {k: [f for f in ref_ff[k]
                              if CATEGORY[f.replace("all_genes_", "").replace(".pt", "")] != cat]
                          for k in range(5)}
                tok = (ad / "run_token.txt").read_text().strip()
                same_sel = all(a.get("sel_digest") == b.get("sel_digest")
                               for a, b in zip(sorted(ar["fold_metrics"], key=lambda m: m["fold"]),
                                               sorted(rr["fold_metrics"], key=lambda m: m["fold"])))
                checks.append({"arm": arm, "variant": v, "cv": cv, "cat": cat,
                               "modalities_ok": ab_ff == exp_ff,
                               "pca_same": (cfg.get("post_pca"), cfg.get("pca_variance"), cfg.get("no_post_pca")) ==
                                           (rr["config"].get("post_pca"), rr["config"].get("pca_variance"), rr["config"].get("no_post_pca")),
                               "sel_split_same": same_sel, "token": tok})
                r = {"arm": arm, "variant": v, "cv": cv, "tier": TIER[cv], "dropped": cat,
                     "removed_per_fold": ";".join(
                         [next((f for f in ref_ff[k] if CATEGORY[f.replace('all_genes_', '').replace('.pt', '')] == cat), "?")
                          .replace("all_genes_", "").replace(".pt", "") for k in range(5)]),
                     "ref_mean": ref.auprg_any.mean(), "abl_mean": ab.auprg_any.mean(),
                     "abl_sd_pop": ab.auprg_any.std(ddof=0)}
                p = paired(ab.auprg_any.values - ref.auprg_any.values)
                r.update(p)
                for m in ["auroc_any", "aupr_any"]:
                    q = paired(ab[m].values - ref[m].values)
                    r[f"d_{m}"] = q["mean_diff"]
                    r[f"p_t_{m}"] = q["p_t"]
                # tab_loco cross-check
                t = tl[(tl.variant == v) & (tl.cv == cv) & (tl.dropped == cat)].iloc[0]
                r["tab_delta_vs_reference"] = t.delta_vs_reference
                r["tab_delta_paired_sd"] = t.delta_paired_sd
                r["src_abl"] = rel(ap_)
                r["src_ref"] = rel(rp)
                fam.append(r)
            fam = pd.DataFrame(fam)
            fam["p_t_holm6"] = holm(fam.p_t.values)
            fam["p_signflip_holm6"] = holm(fam.p_signflip.values)
            rows.append(fam)
res = pd.concat(rows, ignore_index=True)
res.to_csv(OUT / "s3_loco_paired.csv", index=False)
ck = pd.DataFrame(checks)
ck.to_csv(OUT / "s3_loco_provenance.csv", index=False)
print("provenance: modalities_ok", ck.modalities_ok.all(), "| pca_same", ck.pca_same.all(),
      "| sel_split_same", ck.sel_split_same.all(), "| tokens", sorted(ck.token.unique()))
print("max |my delta - tab_loco delta_vs_reference| =",
      float((res.mean_diff - res.tab_delta_vs_reference).abs().max()),
      "| max |pop sd - tab delta_paired_sd| =", float((res.sd_diff_pop - res.tab_delta_paired_sd).abs().max()))

show = ["arm", "variant", "tier", "dropped", "removed_per_fold", "ref_mean", "abl_mean", "mean_diff",
        "sd_diff", "ci95_low", "ci95_high", "p_t", "p_t_holm6", "p_signflip", "n_neg", "d_auroc_any", "d_aupr_any"]
for arm in ["none", "joint"]:
    print(f"\n== LOCO, arm {arm}: ablated - reference (self-scored AUPRG), negative = the category helps")
    print(res[res.arm == arm][show].round(4).to_string(index=False))

# which category matters most: largest mean drop per (arm, variant, tier)
print("\n== largest drop (most negative mean_diff) per arm x variant x tier")
for (arm, v, cv), g in res.groupby(["arm", "variant", "cv"]):
    g = g.sort_values("mean_diff")
    w = g.iloc[0]
    print(f"  {arm:5s} {v:11s} Tier {TIER[cv]:3s}: {w.dropped:11s} {w.mean_diff:+.4f} "
          f"(paired sd {w.sd_diff:.4f}; p_t {w.p_t:.3f}, Holm {w.p_t_holm6:.3f}; sign-flip {w.p_signflip:.4f}); "
          f"order: " + ", ".join(f"{a}:{b:+.4f}" for a, b in zip(g.dropped, g.mean_diff)))
print("\n== cells significant at Holm-6 < 0.05")
print(res[res.p_t_holm6 < 0.05][show].round(4).to_string(index=False))
