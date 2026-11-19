"""Each selected combo's per-fold training modalities (config.json
fold_modalities over embeddings_paths) == best_per_category.json per_fold
winners, for both variants and both arms; and results.json
fold_metrics[k].modalities agrees."""
import json
from pathlib import Path
from common import *
ok = True
for arm in ["none", "joint"]:
    sel = json.loads((SIA[arm] / "best_per_category.json").read_text())
    for (v, kind, cv), rd in sorted(combo_dirs(arm).items()):
        if kind != "combo":
            continue
        cfg = json.loads((rd / "config.json").read_text())
        res = json.loads((rd / "results.json").read_text())
        names = [Path(p).name.replace("all_genes_", "").replace(".pt", "") for p in cfg["embeddings_paths"]]
        fm = cfg["fold_modalities"]
        fm = json.loads(fm) if isinstance(fm, str) else fm
        for k in range(5):
            got = sorted(names[i] for i in fm[str(k)])
            want = sorted(sel[cv][c]["per_fold"][str(k)]["type"] for c in CATS)
            rm = next(m for m in res["fold_metrics"] if m["fold"] == k).get("modalities")
            rm = sorted(Path(str(x)).name.replace("all_genes_", "").replace(".pt", "") for x in rm) if rm else None
            if got != want or (rm is not None and rm != want):
                ok = False
                print("MISMATCH", arm, v, cv, k, got, want, rm)
        print(arm, v, cv, "union:", "+".join(names), "| fold sets match selection")
print("ALL OK" if ok else "MISMATCHES FOUND")
