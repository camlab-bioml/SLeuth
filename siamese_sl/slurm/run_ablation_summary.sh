#!/bin/bash
#SBATCH --job-name=ablation_summary
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu3
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm/logs/ablation_summary_%j.out
#SBATCH --error=slurm/logs/ablation_summary_%j.err
# ============================================================================
# Leave-One-Category-Out Ablation — summary
# ============================================================================
# Builds results/ablation_summary.csv and prints a per-CV table.
#
# Each row pairs the marginal ablation with the standalone score of the
# category that was dropped. Reading only one of the two is how ablation
# tables mislead: under redundancy a category can be individually strong and
# still cost nothing to remove, and the LOCO column alone would report it as
# worthless. The two columns together separate "uninformative" from "already
# covered by the others".
# ============================================================================

set -e
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_ablation.conf"

# CV_TYPES is a bash array; hand it to python as JSON rather than re-deriving
# the list there, so adding a CV type in config.conf reaches the summary too.
CV_TYPES_JSON=$(printf '%s\n' "${CV_TYPES[@]}" | $PYTHON_PATH -c "import sys,json;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")
export CV_TYPES_JSON
# The per-embedding runs live in results/pcavar<PCA_VARIANCE>_<type>_<cv>, and
# PCA_VARIANCE is overridable via BEST_CAT_PCA_VARIANCE. Hardcoding "0.8" here
# blanked the whole `solo` column the moment anyone used that documented
# override — silently, since a missing directory just yields None.
export PCA_VARIANCE

$PYTHON_PATH - <<'PYSUM'
import json, glob, os, csv, statistics as st

CVS = json.loads(os.environ.get("CV_TYPES_JSON", '["cv1","cv2","cv3"]'))
METRICS = [("auprg_any", "AUPRG"), ("aupr_any", "AP"), ("auroc_any", "AUROC")]

def foldmean(path, key):
    try:
        fm = json.load(open(path))["fold_metrics"]
        vals = [f[key] for f in fm if key in f]
        return (st.mean(vals), st.pstdev(vals)) if vals else (None, None)
    except Exception:
        return (None, None)

PCA_VARIANCE = os.environ.get("PCA_VARIANCE", "0.8")

def solo(cv, etype):
    p = f"results/pcavar{PCA_VARIANCE}_{etype}_{cv}/results.json"
    return foldmean(p, "auprg_any")[0] if os.path.exists(p) else None

try:
    winners = json.load(open("results/best_per_category.json"))
except Exception:
    print("ERROR: results/best_per_category.json missing — run the best-per-category pipeline first.")
    raise SystemExit(1)

rows = []
for cv in CVS:
    ref = glob.glob(f"results/best_cat_*_{cv}/results.json")
    if not ref:
        print(f"[{cv}] no reference combo found (results/best_cat_*_{cv}); skipping.\n")
        continue
    ref = ref[0]
    base = {k: foldmean(ref, k)[0] for k, _ in METRICS}
    nref = len(json.load(open(ref))["config"]["embeddings_paths"])

    print("=" * 96)
    print(f"{cv}   reference: full {nref}-category combo   " +
          "  ".join(f"{lab}={base[k]:.4f}" for k, lab in METRICS if base[k] is not None))
    print("=" * 96)
    print(f"  {'dropped':<13}{'n':>3}  " +
          "".join(f"{lab:>9}{'Δ':>9}" for _, lab in METRICS) +
          f"{'solo':>9}  read as")
    print("  " + "-" * 92)

    for cat in sorted(winners.get(cv, {})):
        p = f"results/ablate_no_{cat}_{cv}/results.json"
        if not os.path.exists(p):
            print(f"  {cat:<13}  -   (not run)")
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            # A torn or truncated results.json costs this row only. Letting the
            # exception escape would abort before the CSV is written, throwing
            # away every ablation that DID succeed.
            print(f"  {cat:<13}  -   (unreadable: {e})")
            continue
        n = d.get("ablation", {}).get("num_modalities", nref - 1)
        cells, deltas = "", {}
        for k, lab in METRICS:
            v = foldmean(p, k)[0]
            dv = (v - base[k]) if (v is not None and base[k] is not None) else None
            deltas[k] = dv
            cells += f"{v:>9.4f}" if v is not None else f"{'--':>9}"
            cells += f"{dv:>+9.4f}" if dv is not None else f"{'--':>9}"
        s = solo(cv, winners[cv][cat]["type"])
        socell = f"{s:>9.4f}" if s is not None else f"{'--':>9}"

        # Interpretation. `dg` is the AUPRG delta from removing the category.
        dg = deltas.get("auprg_any")
        if dg is None:
            verdict = ""
        elif dg > 0.005:
            verdict = "HARMFUL — removing it helps"
        elif dg < -0.005:
            verdict = "contributes"
        elif s is not None and base["auprg_any"] is not None and s > base["auprg_any"]:
            verdict = "redundant (strong alone, free to drop)"
        else:
            verdict = "no measurable effect"
        rows.append(dict(cv=cv, dropped=cat, winner=winners[cv][cat]["type"],
                         n_modalities=n, solo_auprg=s,
                         **{f"{k}": foldmean(p, k)[0] for k, _ in METRICS},
                         **{f"delta_{k}": deltas[k] for k, _ in METRICS},
                         verdict=verdict))
        print(f"  {cat:<13}{n:>3}  {cells}{socell}  {verdict}")
    print()

if rows:
    out = "results/ablation_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    print()
    print("Columns: Δ is the ablated score minus the full-combo reference, so a")
    print("POSITIVE Δ means the category was hurting the model. `solo` is that")
    print("category's winning embedding trained on its own, from the")
    print("best-per-category pass — no extra runs needed for it.")
else:
    print("No ablation results found. Submit with: ./slurm/submit_pipeline.sh --ablation-only")
PYSUM
