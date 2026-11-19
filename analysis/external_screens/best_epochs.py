#!/usr/bin/env python3
"""How far fine-tuning moved: best_epoch (0 = the warm start won on val) per
variant x screen x split x mode, from eval_finetuning_<split>/finetuned_metrics.csv
(fold rows, filter sldb_filtered), plus the screen split sizes.
Usage: python -I best_epochs.py ARM"""
import json, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COMBO = ("best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_"
         "mashup_esm1b_prot_t5_text_embed_cv3")
arm = sys.argv[1] if len(sys.argv) > 1 else "none"
root = REPO / "siamese_sl" / "results"
if arm != "none":
    root = root / f"pca_{arm}"
rows = []
for name, d in {"siamese_sl": root / COMBO,
                "siamese_sl_pd_pooled": root / f"var_pd_pooled__{COMBO}"}.items():
    for sp in ("cv1", "cv2", "cv3"):
        m = pd.read_csv(d / f"eval_finetuning_{sp}" / "finetuned_metrics.csv")
        m = m[(m["filter"] == "sldb_filtered") & (m.fold != "ensemble")]
        for (ds, mode), g in m.groupby(["dataset", "mode"]):
            be = g.best_epoch.astype(int)
            rows.append({"arm": arm, "model": name, "split": sp, "screen": ds,
                         "mode": mode, "best_epochs": " ".join(map(str, be)),
                         "n_epoch0": int((be == 0).sum()),
                         "median_best_epoch": float(be.median())})
out = pd.DataFrame(rows)
out.to_csv(HERE / f"out_{arm}" / "best_epochs.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(out.pivot_table(index=["model", "screen", "mode"], columns="split",
                      values="best_epochs", aggfunc="first").to_string())
sv = json.load(open(HERE / "screen_variables.json"))
for k, v in sv["eval"].items():
    if k.startswith("siamese_sl|"):
        print(k, v["split_counts"])
