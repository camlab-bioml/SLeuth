#!/bin/bash
#SBATCH --job-name=bench_summary
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --nodelist=gpu2,gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm/logs/bench_summary_%j.out
#SBATCH --error=slurm/logs/bench_summary_%j.err

# ============================================================================
# Embedding Benchmark — Summary Collector
# ============================================================================
# Runs after all benchmark array tasks complete (dependency: afterany).
# Collects results.json from each results/<type>_<cv>/ directory, prints a
# summary table to stdout, and saves a JSON summary to
# results/embedding_benchmark_summary.json.
#
# Submitted automatically by submit_pipeline.sh with:
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> slurm/run_embedding_benchmark_summary.sh
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_embedding_benchmark.conf"

sleep 10  # let disk settle after benchmark array completes

# Clean previous summary
rm -f results/embedding_benchmark_summary.json

echo "EMBEDDING BENCHMARK — SUMMARY"
echo ""

# ============================================================================
# Collect results and print table
# ============================================================================

$PYTHON_PATH -c "from pathlib import Path; Path('results').mkdir(exist_ok=True)"

# Build Python list literal from bash CV_TYPES array: ["cv1", "cv2", "cv3"]
CV_TYPES_PY="[$(printf '"%s", ' "${CV_TYPES[@]}" | sed 's/, $//')]"

$PYTHON_PATH << PYTHON_SUMMARY
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Discover all benchmark result directories: results/<type>_<cv>/results.json
# ---------------------------------------------------------------------------
results_dir = Path("results")
cv_types = $CV_TYPES_PY  # Injected from CV_TYPES in run_embedding_benchmark.conf

# Find all embedding types that have at least one result
emb_types = sorted({
    d.name.rsplit("_cv", 1)[0]
    for d in results_dir.iterdir()
    if d.is_dir()
    and "_cv" in d.name
    # Exclude PCA and best-cat directories
    and not d.name.startswith("pca")
    and not d.name.startswith("best_cat_")
})

if not emb_types:
    print("ERROR: No benchmark results found. All array tasks may have failed.")
    print("Check slurm/logs/bench_*.out for details.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
data = {}  # data[etype][cv] = {auroc, aupr, f1, ...}
for etype in emb_types:
    data[etype] = {}
    for cv in cv_types:
        rfile = results_dir / f"{etype}_{cv}" / "results.json"
        if rfile.exists():
            try:
                with open(rfile) as _f:
                    s = json.load(_f)["summary"]
                data[etype][cv] = s
            except (KeyError, json.JSONDecodeError):
                data[etype][cv] = None
        else:
            data[etype][cv] = None

# ---------------------------------------------------------------------------
# Print one table per metric (AUPR is the selection metric — printed first).
# ---------------------------------------------------------------------------
def fmt(summary, key_mean, key_std):
    if summary is None:
        return "MISSING"
    v, s = summary.get(key_mean), summary.get(key_std)
    if v is None:
        return "N/A"
    return f"{v:.4f} +/- {s:.4f}"

cv_labels = {"cv1": "CV1 (Edge)", "cv2": "CV2 (Gene)", "cv3": "CV3 (Pair)"}
col_w = 20
header = " | ".join(f"{cv_labels.get(cv, cv.upper()):<{col_w}s}" for cv in cv_types)
table_w = 20 + len(cv_types) * (3 + col_w)

# Track success/fail once (AUPR — same as the selection metric).
successful = 0
failed = 0

def print_table(metric_name, key_mean, key_std, count=False):
    global successful, failed
    print()
    print("=" * table_w)
    print(f"EMBEDDING BENCHMARK RESULTS ({metric_name})".center(table_w))
    print("=" * table_w)
    print()
    print(f"{'Embedding':<20s} | {header}")
    print("-" * table_w)
    for etype in emb_types:
        row = []
        for cv in cv_types:
            s = data[etype].get(cv)
            cell = fmt(s, key_mean, key_std)
            row.append(cell)
            if count:
                if s is not None and s.get(key_mean) is not None:
                    successful += 1
                else:
                    failed += 1
        row_str = " | ".join(f"{cell:<{col_w}s}" for cell in row)
        print(f"{etype:<20s} | {row_str}")

# AUPR is the selection metric — print it first and use it for run counts.
print_table("AUPR — selection metric", "aupr_mean", "aupr_std", count=True)
print_table("AUROC", "auroc_mean", "auroc_std")
print_table("F1 (optimal threshold)", "f1_mean", "f1_std")

print()
print(f"Runs: {successful} successful, {failed} failed/missing (of {len(emb_types) * len(cv_types)})")
print("=" * table_w)

# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------
summary = {
    "timestamp": datetime.now().isoformat(),
    "results": {},
}
for etype in emb_types:
    summary["results"][etype] = {}
    for cv in cv_types:
        s = data[etype].get(cv)
        if s is not None and s.get("aupr_mean") is not None:
            summary["results"][etype][cv] = {
                "auroc": fmt(s, "auroc_mean", "auroc_std"),
                "aupr": fmt(s, "aupr_mean", "aupr_std"),
                "f1": fmt(s, "f1_mean", "f1_std"),
                "total_params": s.get("total_params", 0),
                "nonzero_params": s.get("nonzero_params", 0),
                "weight_sparsity": round(s.get("weight_sparsity") or 0, 2),
            }
        else:
            summary["results"][etype][cv] = {"error": "not found"}

out = results_dir / "embedding_benchmark_summary.json"
with open(out, "w") as _f:
    json.dump(summary, _f, indent=2)
print(f"\nJSON saved to: {out}")
PYTHON_SUMMARY

echo ""
echo "Completed at: $(date)"
