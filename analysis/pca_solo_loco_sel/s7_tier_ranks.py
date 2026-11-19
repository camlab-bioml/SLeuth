"""Rank every row of each arm's tab_tiers.csv per tier x metric (descending)."""
import pandas as pd
from common import *
out = []
for arm in ["none", "joint"]:
    tt = pd.read_csv(CMP[arm] / "paper_tables/tab_tiers.csv")
    for (cv, m), _ in [((cv, m), None) for cv in CVS for m in ["auprg", "ap", "auroc"]]:
        g = tt[tt.cv == cv].sort_values(f"{m}_mean", ascending=False)
        out.append({"arm": arm, "tier": TIER[cv], "metric": m,
                    "order": " > ".join(f"{r.model} {r[m + '_mean']:.4f}" for _, r in g.iterrows())})
o = pd.DataFrame(out)
o.to_csv(OUT / "s7_tier_ranks.csv", index=False)
for _, r in o.iterrows():
    print(f"[{r.arm}] Tier {r.tier} {r.metric}: {r.order}")
