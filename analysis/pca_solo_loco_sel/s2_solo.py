"""Task 2: single modality vs multimodal.

Metric on BOTH sides: train.py's self-scored per-fold auprg_any (results.json
fold_metrics), the metric tab_solo reports; s0_checks.py shows it equals the
shared-graded AUPRG of every combo / all-embedding model to <= 1.4e-6.

Comparators (arm none unless stated): the selected combo of the same variant,
the all-16-embedding model of the same variant; secondary: the same two in arm
joint (PCA on the concatenation), the setting of the submitted Table 2.

Holm family: one (variant, tier, comparator, arm): the 16 embeddings.
Also: honest single-embedding pickers that never look at the fold's test set:
  sel-best   per fold, the embedding with the best sel_auprg_any on that
             fold's selection split (random 10% of its training pairs; the
             split the multimodal selection used);
  val-best   per fold, the best val_auprg_any (that fold's validation split,
             tier-matched: cv2 pairs touching held-out genes, cv3 both genes
             held out; also used for early stopping);
  LOSO-best  per fold, the best mean TEST auprg_any over the other 4 folds.
"""
import json
import numpy as np
import pandas as pd
from common import *

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 500)
VARIANTS = ["pd:pooled", "pd:per_head"]

# ---------------------------------------------------------------- load solo
solo = {}      # (variant, cv, emb) -> DataFrame of per-fold metrics
src = {}
for v in VARIANTS:
    for cv in CVS:
        for e in PAPER16:
            df, p, r = self_folds(solo_dir(v, e, cv))
            cfg = r["config"]
            assert cfg.get("pca_variance") is None and cfg.get("no_post_pca") is True, p
            assert cfg.get("cell_lines") == ("pooled" if v == "pd:pooled" else "per_head"), p
            solo[(v, cv, e)] = df
            src[(v, cv, e)] = rel(p)

# check against tab_solo (both arms) -----------------------------------------
for arm in ["none", "joint"]:
    ts = pd.read_csv(CMP[arm] / "paper_tables/tab_solo.csv")
    ts = ts[ts.in_paper == True]
    worst = 0.0
    for _, r in ts.iterrows():
        df = solo[(r.variant, r.cv, r.embedding)]
        worst = max(worst, abs(df.auprg_any.mean() - r.auprg_any_mean),
                    abs(df.auprg_any.std(ddof=0) - r.auprg_any_sd))
    print(f"tab_solo ({arm}) vs results.json: max |mean or pop-sd diff| = {worst:.2e}")

# ---------------------------------------------------------- comparators
comp = {}
for arm in ["none", "joint"]:
    for (v, kind, cv), rd in combo_dirs(arm).items():
        df, p, _ = self_folds(rd)
        comp[(arm, v, kind, cv)] = (df, rel(p))

# ------------------------------------------------- ranking table (Table 2)
rank_rows = []
for v in VARIANTS:
    for cv in CVS:
        means = {e: solo[(v, cv, e)].auprg_any.mean() for e in PAPER16}
        order = sorted(PAPER16, key=lambda e: -means[e])
        # per-fold ranks
        F = np.array([[solo[(v, cv, e)].auprg_any.values[k] for e in PAPER16]
                      for k in range(5)])            # 5 x 16
        fold_rank = (-F).argsort(axis=1).argsort(axis=1) + 1
        for e in PAPER16:
            j = PAPER16.index(e)
            rank_rows.append({
                "variant": v, "cv": cv, "tier": TIER[cv], "category": CATEGORY[e],
                "embedding": e, "auprg_mean": means[e],
                "auprg_sd_pop": solo[(v, cv, e)].auprg_any.std(ddof=0),
                "rank": order.index(e) + 1,
                "fold_ranks": ";".join(map(str, fold_rank[:, j])),
                "n_folds_top1": int((fold_rank[:, j] == 1).sum()),
                "best_rank": int(fold_rank[:, j].min()), "worst_rank": int(fold_rank[:, j].max()),
                "sel_mean": solo[(v, cv, e)].sel_auprg_any.mean(),
                "val_mean": solo[(v, cv, e)].val_auprg_any.mean(),
                "src": src[(v, cv, e)]})
rank = pd.DataFrame(rank_rows)
rank.to_csv(OUT / "s2_solo_ranks.csv", index=False)
for v in VARIANTS:
    print(f"\n== {v}: 16 single embeddings, self-scored AUPRG mean (pop sd) [rank]; fold ranks")
    for cv in CVS:
        sub = rank[(rank.variant == v) & (rank.cv == cv)].sort_values("rank")
        c_df = comp[("none", v, "combo", cv)][0].auprg_any
        a_df = comp[("none", v, "allmod", cv)][0].auprg_any
        cj = comp[("joint", v, "combo", cv)][0].auprg_any
        aj = comp[("joint", v, "allmod", cv)][0].auprg_any
        print(f"  Tier {TIER[cv]}: combo(no PCA) {c_df.mean():.4f} ({c_df.std(ddof=0):.4f}) | "
              f"all-16(no PCA) {a_df.mean():.4f} ({a_df.std(ddof=0):.4f}) | "
              f"combo(PCA) {cj.mean():.4f} | all-16(PCA) {aj.mean():.4f}")
        for _, r in sub.iterrows():
            print(f"    {r['rank']:2d} {r.embedding:14s} {r.category:11s} {r.auprg_mean:.4f} ({r.auprg_sd_pop:.4f})"
                  f"  fold ranks {r.fold_ranks:16s} top1 in {r.n_folds_top1} folds")

# category means (over the paper's members)
cm = rank.groupby(["variant", "tier", "category"]).auprg_mean.mean().unstack("category")
print("\n== category means of single-embedding AUPRG (members averaged)")
print(cm.round(4).to_string())
cm.to_csv(OUT / "s2_category_means.csv")

# -------------------------------------- paired solo - comparator, Holm 16
pr = []
for v in VARIANTS:
    for cv in CVS:
        for arm in ["none", "joint"]:
            for kind in ["combo", "allmod"]:
                cdf, cp = comp[(arm, v, kind, cv)]
                fam = []
                for e in PAPER16:
                    d = solo[(v, cv, e)].auprg_any.values - cdf.auprg_any.values
                    r = paired(d)
                    r.update({"variant": v, "cv": cv, "tier": TIER[cv], "arm": arm,
                              "comparator": kind, "embedding": e,
                              "solo_mean": solo[(v, cv, e)].auprg_any.mean(),
                              "comp_mean": cdf.auprg_any.mean(), "comp_src": cp})
                    fam.append(r)
                fam = pd.DataFrame(fam)
                fam["p_t_holm16"] = holm(fam.p_t.values)
                fam["p_signflip_holm16"] = holm(fam.p_signflip.values)
                pr.append(fam)
pr = pd.concat(pr, ignore_index=True)
pr.to_csv(OUT / "s2_solo_vs_multimodal_paired.csv", index=False)

summ = []
for (v, cv, arm, kind), g in pr.groupby(["variant", "cv", "arm", "comparator"]):
    summ.append({"variant": v, "tier": TIER[cv], "arm": arm, "comparator": kind,
                 "comp_mean": g.comp_mean.iloc[0],
                 "n_above_point": int((g.mean_diff > 0).sum()),
                 "n_above_p_t_raw<0.05": int(((g.mean_diff > 0) & (g.p_t < 0.05)).sum()),
                 "n_below_p_t_raw<0.05": int(((g.mean_diff < 0) & (g.p_t < 0.05)).sum()),
                 "n_above_holm<0.05": int(((g.mean_diff > 0) & (g.p_t_holm16 < 0.05)).sum()),
                 "n_below_holm<0.05": int(((g.mean_diff < 0) & (g.p_t_holm16 < 0.05)).sum()),
                 "best_emb": g.loc[g.mean_diff.idxmax(), "embedding"],
                 "best_diff": g.mean_diff.max(),
                 "best_ci": f"[{g.loc[g.mean_diff.idxmax(), 'ci95_low']:.4f}, {g.loc[g.mean_diff.idxmax(), 'ci95_high']:.4f}]",
                 "best_p_t": g.loc[g.mean_diff.idxmax(), "p_t"],
                 "best_p_holm": g.loc[g.mean_diff.idxmax(), "p_t_holm16"],
                 "best_p_signflip": g.loc[g.mean_diff.idxmax(), "p_signflip"]})
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "s2_solo_vs_multimodal_summary.csv", index=False)
print("\n== solo - comparator (fold-paired self-scored AUPRG), Holm over the 16 embeddings")
print(summ.round(4).to_string(index=False))

print("\n== details: embeddings with positive point difference vs arm-none comparators")
for (v, cv, kind), g in pr[pr.arm == "none"].groupby(["variant", "cv", "comparator"]):
    g = g.sort_values("mean_diff", ascending=False)
    gg = g[g.mean_diff > 0]
    if len(gg):
        print(f"  {v} Tier {TIER[cv]} vs {kind} ({g.comp_mean.iloc[0]:.4f}):")
        for _, r in gg.iterrows():
            print(f"     {r.embedding:14s} diff {r.mean_diff:+.4f}  sd {r.sd_diff:.4f}  95%CI [{r.ci95_low:+.4f},{r.ci95_high:+.4f}]"
                  f"  p_t {r.p_t:.3f} (Holm {r.p_t_holm16:.3f})  p_sf {r.p_signflip:.4f}  folds+ {r.n_pos}/5")

# ------------------------------ top-1 vs others: "confidence set" for the best
cs_rows = []
for v in VARIANTS:
    for cv in CVS:
        means = {e: solo[(v, cv, e)].auprg_any.mean() for e in PAPER16}
        top = max(PAPER16, key=lambda e: means[e])
        fam = []
        for e in PAPER16:
            if e == top:
                continue
            d = solo[(v, cv, top)].auprg_any.values - solo[(v, cv, e)].auprg_any.values
            r = paired(d)
            r.update({"variant": v, "tier": TIER[cv], "top": top, "other": e})
            fam.append(r)
        fam = pd.DataFrame(fam)
        fam["p_t_holm15"] = holm(fam.p_t.values)
        not_worse = fam[fam.p_t >= 0.05].other.tolist()
        not_worse_holm = fam[fam.p_t_holm15 >= 0.05].other.tolist()
        cs_rows.append({"variant": v, "tier": TIER[cv], "top": top, "top_mean": means[top],
                        "n_not_distinguishable_raw": len(not_worse),
                        "n_not_distinguishable_holm": len(not_worse_holm),
                        "not_distinguishable_raw": ",".join(not_worse)})
        fam.to_csv(OUT / f"s2_top_vs_rest_{v.replace(':', '_')}_{cv}.csv", index=False)
cs = pd.DataFrame(cs_rows)
cs.to_csv(OUT / "s2_best_confidence_set.csv", index=False)
print("\n== embeddings not distinguishable from the top single embedding (paired t, p >= 0.05)")
print(cs.to_string(index=False))

# --------------------------------------------- honest single-embedding pickers
hp = []
for v in VARIANTS:
    for cv in CVS:
        T = np.array([[solo[(v, cv, e)].auprg_any.values[k] for e in PAPER16] for k in range(5)])
        S = np.array([[solo[(v, cv, e)].sel_auprg_any.values[k] for e in PAPER16] for k in range(5)])
        V = np.array([[solo[(v, cv, e)].val_auprg_any.values[k] for e in PAPER16] for k in range(5)])
        picks = {"sel-best": S.argmax(axis=1), "val-best": V.argmax(axis=1),
                 "LOSO-best": np.array([np.delete(T, k, axis=0).mean(axis=0).argmax() for k in range(5)]),
                 "oracle-best(test,biased)": np.repeat(T.mean(axis=0).argmax(), 5)}
        for name, idx in picks.items():
            got = T[np.arange(5), idx]
            for arm in ["none"]:
                for kind in ["combo", "allmod"]:
                    cdf = comp[(arm, v, kind, cv)][0].auprg_any.values
                    r = paired(got - cdf)
                    r.update({"variant": v, "tier": TIER[cv], "picker": name,
                              "picked": ";".join(PAPER16[i] for i in idx),
                              "picked_mean": got.mean(), "vs": kind, "vs_mean": cdf.mean()})
                    hp.append(r)
hp = pd.DataFrame(hp)
hp.to_csv(OUT / "s2_honest_single_pickers.csv", index=False)
print("\n== single embedding chosen without the fold's test set, minus multimodal (arm none), fold-paired")
print(hp[["variant", "tier", "picker", "picked", "picked_mean", "vs", "vs_mean", "mean_diff",
          "ci95_low", "ci95_high", "p_t", "p_signflip", "n_pos"]].round(4).to_string(index=False))

# --------------------- co-author check: Tier III differences vs fold-level error bars
print("\n== co-author check (Tier III): top single embeddings vs combo / all-16, both variants")
for v in VARIANTS:
    cv = "cv3"
    c = comp[("none", v, "combo", cv)][0].auprg_any
    a = comp[("none", v, "allmod", cv)][0].auprg_any
    sub = rank[(rank.variant == v) & (rank.cv == cv)].sort_values("rank").head(4)
    print(f"  {v}: combo {c.mean():.4f} (sd {c.std(ddof=0):.4f}, range {c.min():.4f}-{c.max():.4f}); "
          f"all-16 {a.mean():.4f} (sd {a.std(ddof=0):.4f})")
    for _, r in sub.iterrows():
        s = solo[(v, cv, r.embedding)].auprg_any
        dc = paired(s.values - c.values)
        da = paired(s.values - a.values)
        print(f"    #{r['rank']} {r.embedding:13s} {s.mean():.4f} (sd {s.std(ddof=0):.4f}) | "
              f"- combo {dc['mean_diff']:+.4f} [{dc['ci95_low']:+.4f},{dc['ci95_high']:+.4f}] p_t {dc['p_t']:.3f} sf {dc['p_signflip']:.3f} | "
              f"- all16 {da['mean_diff']:+.4f} [{da['ci95_low']:+.4f},{da['ci95_high']:+.4f}] p_t {da['p_t']:.3f} sf {da['p_signflip']:.3f}")
