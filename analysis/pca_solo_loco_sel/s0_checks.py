"""Sanity checks: (a) graded per-fold AUPRG (row JSON summary.folds, by
position) equals the self-scored per-fold auprg_any (results.json, by fold id)
for every selected combo and all-embedding model of both arms (so fold order
matches, and self-scored == shared-graded for these models); (b) tab_tiers
means/sds are the per-fold mean / population sd of the graded folds."""
import json
import numpy as np
import pandas as pd
from common import *

rows = []
for arm in ["none", "joint"]:
    dirs = combo_dirs(arm)
    for (variant, kind, cv), rd in sorted(dirs.items()):
        model = {("pd:per_head", "combo"): "siamese_sl",
                 ("pd:pooled", "combo"): "siamese_sl_pd_pooled",
                 ("pd:per_head", "allmod"): "siamese_sl_allmod",
                 ("pd:pooled", "allmod"): "siamese_sl_allmod_pd_pooled"}[(variant, kind)]
        g, gp = graded_folds(arm, model, cv)
        s, sp, _ = self_folds(rd)
        dif = (g["auprg"].values - s["auprg_any"].values)
        difroc = (g["auroc"].values - s["auroc_any"].values)
        difap = (g["ap"].values - s["aupr_any"].values)
        rows.append({"arm": arm, "model": model, "cv": cv,
                     "max_abs_auprg_graded_minus_self": float(np.abs(dif).max()),
                     "max_abs_auroc": float(np.abs(difroc).max()),
                     "max_abs_ap_vs_aupr_any": float(np.abs(difap).max()),
                     "graded": rel(gp), "self": rel(sp)})
chk = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
print(chk.drop(columns=["graded", "self"]).to_string(index=False))
print("overall max |graded - self| AUPRG:", chk.max_abs_auprg_graded_minus_self.max())
print("overall max |graded - self| AUROC:", chk.max_abs_auroc.max())
print("overall max |graded ap - self aupr_any|:", chk.max_abs_ap_vs_aupr_any.max())
chk.to_csv(OUT / "s0_graded_vs_self.csv", index=False)

# (b) tab_tiers consistency
bad = 0
for arm in ["none", "joint"]:
    tt = pd.read_csv(CMP[arm] / "paper_tables/tab_tiers.csv")
    for _, r in tt.iterrows():
        g, _ = graded_folds(arm, r["model"], r["cv"])
        for m in ["auprg", "ap", "auroc"]:
            if abs(g[m].mean() - r[f"{m}_mean"]) > 1e-9 or abs(g[m].std(ddof=0) - r[f"{m}_sd"]) > 1e-9:
                bad += 1
                print("MISMATCH", arm, r["model"], r["cv"], m, g[m].mean(), r[f"{m}_mean"], g[m].std(ddof=0), r[f"{m}_sd"])
print("tab_tiers mismatches (mean, population sd):", bad)
