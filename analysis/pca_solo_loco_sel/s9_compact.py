"""Compact tables for the report's first section (formatting only)."""
import pandas as pd
from common import *
ci = pd.read_csv(OUT / "s4_row_bootstrap_cis.csv")
BR = pd.read_csv(OUT / "s4_combo_minus_allmod.csv")
L = []
tt = pd.read_csv(CMP["none"] / "paper_tables/tab_tiers.csv")
def c(model, cv, m, arm="none"):
    t = pd.read_csv(CMP[arm] / "paper_tables/tab_tiers.csv")
    r = t[(t.model == model) & (t.cv == cv)].iloc[0]
    b = ci[(ci.arm == arm) & (ci.model == model) & (ci.cv == cv) & (ci.metric == m)].iloc[0]
    return f"{r[m + '_mean']:.3f} [{b.ci_low:.3f}, {b.ci_high:.3f}]"
L.append("| Tier | Metric | All-16 pooled | All-16 cell-line | Selected pooled | Selected cell-line | Best non-SLeuth row |")
L.append("|---|---|---|---|---|---|---|")
for cv in CVS:
    for m in ["auprg", "ap", "auroc"]:
        g = tt[(tt.cv == cv) & ~tt.model.str.startswith("siamese_sl")].sort_values(f"{m}_mean", ascending=False).iloc[0]
        L.append(f"| {TIER[cv]} | {m.upper()} | {c('siamese_sl_allmod_pd_pooled', cv, m)} | {c('siamese_sl_allmod', cv, m)} | "
                 f"{c('siamese_sl_pd_pooled', cv, m)} | {c('siamese_sl', cv, m)} | {g.model} {c(g.model, cv, m)} |")
L.append("")
L.append("| Arm | Variant | Tier | Combo AUPRG | All-16 AUPRG | Combo - all-16 [boot 95% CI] | p2 boot | fold-paired 95% CI | p_t (Holm-2) | splits combo higher |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in BR[BR.metric == "auprg"].iterrows():
    p2 = "<= 0.002" if r.p2_boot <= 2 / 1001 + 1e-12 else f"{r.p2_boot:.3f}"
    L.append(f"| {r.arm} | {r.variant} | {r.tier} | {r.combo_mean:.3f} | {r.allmod_mean:.3f} | {r.boot_point:+.4f} [{r.boot_ci_low:+.4f}, {r.boot_ci_high:+.4f}] | {p2} | "
             f"[{r.fold_ci95_low:+.4f}, {r.fold_ci95_high:+.4f}] | {r.p_t:.3f} ({r.p_t_holm:.3f}) | {r.n_pos}/5 |")
(OUT / "tables_compact.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
