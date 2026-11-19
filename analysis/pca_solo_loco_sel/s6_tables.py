"""Builds the markdown tables of report.md from the s0-s5 outputs (no new
statistics here, only formatting), into tables.md."""
import json
import numpy as np
import pandas as pd
from common import *

L = []


def P(s=""):
    L.append(s)


def f3(x):
    return f"{x:.3f}"


def sgn(x, k=3):
    return f"{x:+.{k}f}"


def pt(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def pb(p2):
    return "<= 0.002" if p2 <= 2 / 1001 + 1e-12 else f"{p2:.3f}"


ci = pd.read_csv(OUT / "s4_row_bootstrap_cis.csv")
tt = {arm: pd.read_csv(CMP[arm] / "paper_tables/tab_tiers.csv") for arm in ["none", "joint"]}
BR = pd.read_csv(OUT / "s4_combo_minus_allmod.csv")
PCA = pd.read_csv(OUT / "s1_pca_paired.csv")
E = pd.read_csv(OUT / "s4_allmod_vs_peers_paired.csv")


def cell(arm, model, cv, m):
    r = tt[arm][(tt[arm].model == model) & (tt[arm].cv == cv)].iloc[0]
    c = ci[(ci.arm == arm) & (ci.model == model) & (ci.cv == cv) & (ci.metric == m)].iloc[0]
    return f"{r[m + '_mean']:.3f} ({r[m + '_sd']:.3f}) [{c.ci_low:.3f}, {c.ci_high:.3f}]"


# ---------------------------------------------------------------- T1
P("### T1. All-16-embedding models vs the selected combos (graded, shared evaluator)")
P()
P("Mean (population SD over the 5 splits) [pair-bootstrap 95% CI of the 5-split mean, B = 1000]. "
  "Source: `sl_comparison/results[/pca_joint]/paper_tables/tab_tiers.csv` (auprg/ap/auroc _mean/_sd) and "
  "`.../bootstrap/ci_tiers.csv` (fold = mean, ci_low/ci_high).")
P()
P("| Arm | Tier | Model | Params (fold mean) | AUPRG | AP | AUROC |")
P("|---|---|---|---|---|---|---|")
for arm in ["none", "joint"]:
    for cv in CVS:
        for model in ["siamese_sl_allmod_pd_pooled", "siamese_sl_pd_pooled", "siamese_sl_allmod", "siamese_sl"]:
            r = tt[arm][(tt[arm].model == model) & (tt[arm].cv == cv)].iloc[0]
            P(f"| {arm} | {TIER[cv]} | {ROW_LABEL[model]} | {r.params_fold_mean:,.0f} | "
              f"{cell(arm, model, cv, 'auprg')} | {cell(arm, model, cv, 'ap')} | {cell(arm, model, cv, 'auroc')} |")
P()

P("### T1b. Selected combo minus all-16 model, same variant and arm")
P()
P("Pair bootstrap: `ci_tiers_diffs.csv` (model = combo, peer = all-16, fold = mean): point [95% CI], two-sided "
  "p2 = min(1, 2 max(min(p, 1-p), 1/(B+1))) from p_boot_le0, B = 1000, floor 0.002; Holm over the 2 variants "
  "per (arm, metric, tier). Fold-paired: paired t over the 5 splits (graded per-fold values), exact sign-flip p "
  "(minimum 0.0625 with 5 splits), # = splits where the combo is higher.")
P()
P("| Arm | Variant | Tier | Metric | Combo | All-16 | Diff [boot 95% CI] | p2 boot (Holm) | fold-paired 95% CI | p_t (Holm) | p sign-flip | # combo > all-16 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in BR.iterrows():
    P(f"| {r.arm} | {r.variant} | {r.tier} | {r.metric} | {r.combo_mean:.3f} | {r.allmod_mean:.3f} | "
      f"{sgn(r.boot_point, 4)} [{sgn(r.boot_ci_low, 4)}, {sgn(r.boot_ci_high, 4)}] | {pb(r.p2_boot)} ({pb(r.p2_boot_holm)}) | "
      f"[{sgn(r.fold_ci95_low, 4)}, {sgn(r.fold_ci95_high, 4)}] | {pt(r.p_t)} ({pt(r.p_t_holm)}) | {r.p_signflip:.4f} | {r.n_pos}/5 |")
P()

P("### T1c. All-16 models minus the strongest other rows (graded, fold-paired; Holm over the 6 non-SLeuth rows per arm x variant x metric x tier)")
P()
P("| Arm | Variant | Tier | Metric | vs | All-16 | Other | Diff [95% CI] | p_t (Holm-6) | p sign-flip |")
P("|---|---|---|---|---|---|---|---|---|---|")
for arm in ["none", "joint"]:
    for v in ["pd:pooled", "pd:per_head"]:
        for cv in ["cv2", "cv3"]:
            for m in ["auprg", "auroc"]:
                g = E[(E.arm == arm) & (E.variant == v) & (E.tier == TIER[cv]) & (E.metric == m) &
                      E.peer.isin(["SLMGAE", "SLGNN", "ELISL", "BilinearProbe"])]
                for _, r in g.iterrows():
                    P(f"| {arm} | {v} | {r.tier} | {m} | {r.peer} | {r.allmod_mean:.3f} | {r.peer_mean:.3f} | "
                      f"{sgn(r.mean_diff)} [{sgn(r.ci95_low)}, {sgn(r.ci95_high)}] | {pt(r.p_t)} ({pt(r.p_t_holm6)}) | {r.p_signflip:.4f} |")
P()

# ---------------------------------------------------------------- T2 PCA
P("### T2. No PCA (arm none) minus PCA of the concatenation at 0.8 variance (arm joint), fold-paired, graded")
P()
P("Same modalities in both arms (tab_pca `same_modalities_as_reference` = True). Diff = none - joint "
  "[paired-t 95% CI]; Holm over the 5 SLeuth rows per (metric, tier); # = splits where no PCA is higher. "
  "Source: `s1_pca_paired.csv` from `sl_comparison/results[/pca_joint]/<row>_<cv>.json` summary.folds.")
P()
P("| Row | Tier | Metric | No PCA | PCA | Diff [95% CI] | p_t | Holm | p sign-flip | # |")
P("|---|---|---|---|---|---|---|---|---|---|")
for model in SLEUTH_ROWS:
    for cv in CVS:
        for m in ["auprg", "ap", "auroc"]:
            r = PCA[(PCA.model == model) & (PCA.cv == cv) & (PCA.metric == m)].iloc[0]
            P(f"| {ROW_LABEL[model]} | {TIER[cv]} | {m} | {r.none_mean:.3f} | {r.joint_mean:.3f} | "
              f"{sgn(r.mean_diff)} [{sgn(r.ci95_low)}, {sgn(r.ci95_high)}] | {pt(r.p_t)} | {pt(r.p_t_holm_tier)} | {r.p_signflip:.4f} | {r.n_pos}/5 |")
P()

P("### T2c. Compact (paper-sized) version of T2: AUPRG only, the four multimodal models")
P()
P("| Model | Tier | No PCA | PCA 0.8 | Diff [95% CI] | p_t (Holm-5) | p sign-flip | splits no PCA higher |")
P("|---|---|---|---|---|---|---|---|")
for model in SLEUTH_ROWS[:4]:
    for cv in CVS:
        r = PCA[(PCA.model == model) & (PCA.cv == cv) & (PCA.metric == "auprg")].iloc[0]
        P(f"| {ROW_LABEL[model]} | {TIER[cv]} | {r.none_mean:.3f} ({r.none_sd_pop:.3f}) | {r.joint_mean:.3f} ({r.joint_sd_pop:.3f}) | "
          f"{sgn(r.mean_diff)} [{sgn(r.ci95_low)}, {sgn(r.ci95_high)}] | {pt(r.p_t)} ({pt(r.p_t_holm_tier)}) | {r.p_signflip:.4f} | {r.n_pos}/5 |")
P()
P("Per-split Tier III AUPRG (no PCA | PCA), split 0-4:")
P()
for model in SLEUTH_ROWS[:4]:
    r = PCA[(PCA.model == model) & (PCA.cv == "cv3") & (PCA.metric == "auprg")].iloc[0]
    P(f"- {ROW_LABEL[model]}: {r.none_folds.replace(';', ', ')} | {r.joint_folds.replace(';', ', ')}")
P()

# ---------------------------------------------------------------- T3 solo
rank = pd.read_csv(OUT / "s2_solo_ranks.csv")
P("### T3. Single-embedding models (self-scored AUPRG_any, mean (population SD) [rank among the 16])")
P()
P("Native dimension, no PCA (shared by both arms). Source: `sl_comparison/results/paper_tables/tab_solo.csv` "
  "(auprg_any_mean/_sd; identical to the runs' results.json to 1e-16), pooled = `siamese_sl/results/pcavar0.8_<emb>_<cv>`, "
  "cell-line = `siamese_sl/results/var_pd_per_head__pcavar0.8_<emb>_<cv>`. Bold-worthy: rank 1.")
P()
P("| Category | Embedding | Pooled I | Pooled II | Pooled III | Cell-line I | Cell-line II | Cell-line III |")
P("|---|---|---|---|---|---|---|---|")
for e in PAPER16:
    cells = []
    for v in ["pd:pooled", "pd:per_head"]:
        for cv in CVS:
            r = rank[(rank.variant == v) & (rank.cv == cv) & (rank.embedding == e)].iloc[0]
            cells.append(f"{r.auprg_mean:.3f} ({r.auprg_sd_pop:.3f}) [{r['rank']}]")
    P(f"| {CATEGORY[e]} | {e} | " + " | ".join(cells) + " |")
solo_cmp = {}
for arm in ["none", "joint"]:
    for kind in ["combo", "allmod"]:
        cells = []
        for v in ["pd:pooled", "pd:per_head"]:
            for cv in CVS:
                df, _, _ = self_folds(combo_dirs(arm)[(v, kind, cv)])
                cells.append(f"{df.auprg_any.mean():.3f} ({df.auprg_any.std(ddof=0):.3f})")
        lab = {"combo": "selected combo", "allmod": "all 16 embeddings"}[kind]
        P(f"| multimodal | {lab}, {'no PCA' if arm == 'none' else 'PCA 0.8'} | " + " | ".join(cells) + " |")
P()
cm = pd.read_csv(OUT / "s2_category_means.csv")
P("Category means of the members (the paper's 'strongest category' sentence):")
P()
P("| Variant | Tier | expression | go | kg | ppi | protein_seq | text |")
P("|---|---|---|---|---|---|---|---|")
for _, r in cm.iterrows():
    P(f"| {r.variant} | {r.tier} | " + " | ".join(f"{r[c]:.4f}" for c in CATS) + " |")
P()

# ---------------------------------------------------------------- T4
S = pd.read_csv(OUT / "s2_solo_vs_multimodal_summary.csv")
P("### T4. How many of the 16 single embeddings beat each multimodal model (fold-paired self-scored AUPRG; Holm over the 16)")
P()
P("| Variant | Tier | Multimodal model | Its AUPRG | # above (point) | # above, Holm p<0.05 | # below, Holm p<0.05 | best single: diff [95% CI], p_t, sign-flip |")
P("|---|---|---|---|---|---|---|---|")
for v in ["pd:pooled", "pd:per_head"]:
    for t in ["I", "II", "III"]:
        for arm in ["none", "joint"]:
            for kind in ["combo", "allmod"]:
                r = S[(S.variant == v) & (S.tier == t) & (S.arm == arm) & (S.comparator == kind)].iloc[0]
                lab = {"combo": "selected combo", "allmod": "all-16"}[kind] + (", no PCA" if arm == "none" else ", PCA")
                P(f"| {v} | {t} | {lab} | {r.comp_mean:.3f} | {r.n_above_point} | {r['n_above_holm<0.05']} | {r['n_below_holm<0.05']} | "
                  f"{r.best_emb} {sgn(r.best_diff)} {r.best_ci}, {pt(r.best_p_t)}, {r.best_p_signflip:.3f} |")
P()

# ---------------------------------------------------------------- T5 pickers
H = pd.read_csv(OUT / "s2_honest_single_pickers.csv")
P("### T5. A single embedding picked WITHOUT the fold's test set, minus the no-PCA multimodal model (fold-paired AUPRG)")
P()
P("sel-best: best selection-split score (the split the combo's selection used); val-best: best score on the fold's "
  "validation split (tier-matched); LOSO-best: best mean test AUPRG over the other 4 splits; oracle: best mean test "
  "AUPRG over all 5 splits (uses the test set: optimistic).")
P()
P("| Variant | Tier | Picker | Picks (folds 0-4) | Picked AUPRG | vs | Multimodal | Diff [95% CI] | p_t | p sign-flip |")
P("|---|---|---|---|---|---|---|---|---|---|")
for _, r in H[H.tier.isin(["II", "III"])].iterrows():
    P(f"| {r.variant} | {r.tier} | {r.picker} | {r.picked.replace(';', ', ')} | {r.picked_mean:.3f} | {r.vs} | {r.vs_mean:.3f} | "
      f"{sgn(r.mean_diff)} [{sgn(r.ci95_low)}, {sgn(r.ci95_high)}] | {pt(r.p_t)} | {r.p_signflip:.4f} |")
P()

# ---------------------------------------------------------------- T6 LOCO
LO = pd.read_csv(OUT / "s3_loco_paired.csv")
P("### T6. Leave-one-category-out: ablated minus full combo (self-scored AUPRG, fold-paired; Holm over the 6 categories per arm x variant x tier)")
P()
P("Negative = the category helps. Source: `paper_tables/tab_loco.csv` (delta_vs_reference = delta_paired_mean, "
  "reproduced to 3e-16) and the `ablate_no_<cat>_<cv>` / `var_pd_pooled__ablate_no_<cat>_<cv>` results.json.")
P()
for arm in ["none", "joint"]:
    P(f"Arm {arm}:")
    P()
    P("| Variant | Category removed | Tier I diff (p_t) | Tier II diff (p_t) | Tier III diff [95% CI] | III p_t / Holm / sign-flip | III # worse |")
    P("|---|---|---|---|---|---|---|")
    for v in ["pd:pooled", "pd:per_head"]:
        for c in CATS:
            g = LO[(LO.arm == arm) & (LO.variant == v) & (LO.dropped == c)].set_index("tier")
            r3 = g.loc["III"]
            P(f"| {v} | {c} | {sgn(g.loc['I'].mean_diff, 4)} ({pt(g.loc['I'].p_t)}) | {sgn(g.loc['II'].mean_diff)} ({pt(g.loc['II'].p_t)}) | "
              f"{sgn(r3.mean_diff)} [{sgn(r3.ci95_low)}, {sgn(r3.ci95_high)}] | {pt(r3.p_t)} / {pt(r3.p_t_holm6)} / {r3.p_signflip:.4f} | {r3.n_neg}/5 |")
        ref = LO[(LO.arm == arm) & (LO.variant == v)].groupby("tier").ref_mean.first()
        P(f"| {v} | none (full combo) | {ref['I']:.4f} | {ref['II']:.3f} | {ref['III']:.3f} |  |  |")
    P()

# ---------------------------------------------------------------- T7 selection
W = pd.read_csv(OUT / "s4_winners_per_fold.csv")
P("### T7. Modalities picked per tier and split (identical in both arms)")
P()
P("| Tier | Category | split 0 | split 1 | split 2 | split 3 | split 4 |")
P("|---|---|---|---|---|---|---|")
for t in ["I", "II", "III"]:
    for c in CATS:
        g = W[(W.tier == t) & (W.category == c)].sort_values("fold")
        P(f"| {t} | {c} | " + " | ".join(g.winner) + " |")
P()
C = pd.read_csv(OUT / "s4_selection_score_vs_test.csv")
P("### T8. Does the selection score predict test performance? Spearman across the 16 single-embedding runs, per split")
P()
P("| Variant | Tier | selection split vs test: mean (min, max) | val split vs test: mean (min, max) | spread of selection scores | spread of test AUPRG |")
P("|---|---|---|---|---|---|")
for (v, t), g in C.groupby(["variant", "tier"]):
    P(f"| {v} | {t} | {g.spearman_sel_vs_test.mean():.2f} ({g.spearman_sel_vs_test.min():.2f}, {g.spearman_sel_vs_test.max():.2f}) | "
      f"{g.spearman_val_vs_test.mean():.2f} ({g.spearman_val_vs_test.min():.2f}, {g.spearman_val_vs_test.max():.2f}) | "
      f"{g.sel_range.mean():.3f} | {g.test_range.mean():.3f} |")
P()
(OUT / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
