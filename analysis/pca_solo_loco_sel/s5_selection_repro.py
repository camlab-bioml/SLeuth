"""Task 4 (cont.): reproduce the per-fold category winners from the Step-1
single-embedding runs (pd:pooled, the selection variant) and measure how
close each decision was (winner - runner-up selection score), plus what a
tier-matched rule (the fold's val split) and the test set would pick.
Catalog order (ties -> first): siamese_sl/slurm/run_best_per_category.conf
EMB_CATALOG (esmc, scprint, seqvec not generated -> not ranked)."""
import json
import numpy as np
import pandas as pd
from common import *

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 500)
CATALOG = ["geneformer", "scgpt", "gene2vec", "esm2", "esm1b", "prot_t5",
           "text_embed", "bioconceptvec", "node2vec_ppi", "ppi_svd", "ppi_raw",
           "mashup", "go", "go2vec", "onto2vec", "kg_complex"]
sel = json.loads((SIA["none"] / "best_per_category.json").read_text())
rows = []
mism = 0
for cv in CVS:
    F = {e: self_folds(solo_dir("pd:pooled", e, cv))[0] for e in CATALOG}
    for c in CATS:
        mem = [e for e in CATALOG if CATEGORY[e] == c]
        for k in range(5):
            s = {e: F[e].sel_auprg_any.values[k] for e in mem}
            win = max(mem, key=lambda e: (s[e], -mem.index(e)))
            got = sel[cv][c]["per_fold"][str(k)]["type"]
            mism += win != got
            srt = sorted(mem, key=lambda e: -s[e])
            margin = s[srt[0]] - s[srt[1]] if len(mem) > 1 else np.nan
            vbest = max(mem, key=lambda e: F[e].val_auprg_any.values[k])
            tbest = max(mem, key=lambda e: F[e].auprg_any.values[k])
            rows.append({"tier": TIER[cv], "category": c, "fold": k, "picked": got,
                         "recomputed": win, "sel_score": s[win],
                         "runner_up": srt[1] if len(mem) > 1 else "",
                         "margin_sel": margin, "val_pick": vbest, "test_best": tbest,
                         "picked_test": F[got].auprg_any.values[k],
                         "val_pick_test": F[vbest].auprg_any.values[k],
                         "test_best_test": F[tbest].auprg_any.values[k]})
R = pd.DataFrame(rows)
R.to_csv(OUT / "s5_selection_repro.csv", index=False)
print("recomputed winners that differ from best_per_category.json:", mism, "of", len(R))
multi = R[R.category != "kg"]
print("\n== selection margins (winner - runner-up sel_auprg_any), categories with >1 member")
print(multi.groupby("tier").margin_sel.describe().round(5).to_string())
print("\n== agreement of the selection-split pick with the val-split pick / the fold's test-best member")
agg = multi.groupby("tier").apply(lambda g: pd.Series({
    "n_decisions": len(g), "same_as_val_pick": int((g.picked == g.val_pick).sum()),
    "same_as_test_best": int((g.picked == g.test_best).sum()),
    "val_pick_same_as_test_best": int((g.val_pick == g.test_best).sum()),
    "mean_test_of_pick": g.picked_test.mean(), "mean_test_of_val_pick": g.val_pick_test.mean(),
    "mean_test_of_test_best": g.test_best_test.mean()}), include_groups=False)
print(agg.round(4).to_string())
print("\n== Tier III decisions in detail")
print(R[R.tier == "III"][["category", "fold", "picked", "sel_score", "runner_up", "margin_sel",
                          "val_pick", "test_best", "picked_test", "val_pick_test", "test_best_test"]]
      .round(4).to_string(index=False))
