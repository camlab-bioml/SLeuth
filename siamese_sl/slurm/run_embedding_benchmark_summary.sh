#!/bin/bash
#SBATCH --job-name=bench_summary
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission:
# siamese_sl/slurm/submit_pipeline.sh, the only one that submits this script,
# passes it as --nodelist on the command line, which replaces this line
# (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
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
# Collects results.json from each $STEP1_ROOT/<type>_<cv>/ directory (the
# Step-1 root of the job's PCA arm, config.conf pca_step1_arm over PCA_ARMS:
# results/ and results/pca_per_modality/ with all four arms), prints a summary table
# to stdout, and saves a JSON summary to $STEP1_ROOT/embedding_benchmark_summary.json.
#
# Submitted automatically by submit_pipeline.sh with:
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> slurm/run_embedding_benchmark_summary.sh
#
# Only directories whose run_token.txt is this submission's RUN_TOKEN are
# tabulated; any other is shown as STALE: a task that never started
# (cancelled while pending, dependency never satisfied) keeps its previous
# run's results.json. A direct sbatch must export the array's token
# (cat <Step-1 root>/<type>_<cv>/run_token.txt), its PCA_ARM and the
# submission's PCA_ARMS.
# ============================================================================

set -e
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_embedding_benchmark.conf"

sleep 10  # let disk settle after benchmark array completes

# Clean previous summary
rm -f "$STEP1_ROOT/embedding_benchmark_summary.json"

echo "EMBEDDING BENCHMARK — SUMMARY ($(pca_step1_label "$PCA_ARM"); $STEP1_ROOT/)"
echo ""

# ============================================================================
# Collect results and print table
# ============================================================================

$PYTHON_PATH -c "from pathlib import Path; Path('$STEP1_ROOT').mkdir(parents=True, exist_ok=True)"

# Build Python list literal from bash CV_TYPES array: ["cv1", "cv2", "cv3"]
CV_TYPES_PY="[$(printf '"%s", ' "${CV_TYPES[@]}" | sed 's/, $//')]"
# Allow-list of embedding types, injected the same way CV_TYPES is. The
# discovery below used to DENY-list known non-embedding directory prefixes
# ("pca", "best_cat_"), which went stale the moment a new results/<x>_<cv>
# family appeared: the leave-one-category-out ablation writes
# results/ablate_no_<category>_<cv>, which matched the pattern and would have
# been tabulated as if "ablate_no_expression" were an embedding. An allow-list
# built from the catalog cannot go stale that way.
EMB_TYPES_PY="[$(printf '"%s", ' "${ALL_EMBEDDINGS[@]%%:*}" | sed 's/, $//')]"

$PYTHON_PATH << PYTHON_SUMMARY
import json
import os
from pathlib import Path
from datetime import datetime

# This submission's id (config.conf export); only directories carrying it in
# run_token.txt are tabulated.
RUN_TOKEN = os.environ.get("RUN_TOKEN", "")

# ---------------------------------------------------------------------------
# Discover all benchmark result directories: <Step-1 root>/<type>_<cv>/results.json
# ---------------------------------------------------------------------------
results_dir = Path("$STEP1_ROOT")
cv_types = $CV_TYPES_PY  # Injected from CV_TYPES (config.conf; overridable in run_embedding_benchmark.conf)

# Find all embedding types that have at least one result.
# ALLOW-list, not deny-list: only names present in ALL_EMBEDDINGS count. This
# is what keeps the best-per-category (pcavar*), combo (best_cat_*) and
# leave-one-category-out (ablate_no_*) directories out of the single-embedding
# table — they all match "<something>_<cv>" and a deny-list has to be extended
# every time a stage is added.
KNOWN_EMB = set($EMB_TYPES_PY)
emb_types = sorted({
    d.name.rsplit("_cv", 1)[0]
    for d in results_dir.iterdir()
    if d.is_dir()
    and "_cv" in d.name
    and d.name.rsplit("_cv", 1)[0] in KNOWN_EMB
})

if not emb_types:
    print("ERROR: No benchmark results found. All array tasks may have failed.")
    print("Check slurm/logs/bench_*.out for details.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
data = {}  # data[etype][cv] = {auroc, aupr, f1, ...}
stale = {}  # (etype, cv) -> run_token.txt found there (None = none)
for etype in emb_types:
    data[etype] = {}
    for cv in cv_types:
        rfile = results_dir / f"{etype}_{cv}" / "results.json"
        tfile = rfile.parent / "run_token.txt"
        tok = tfile.read_text().strip() if tfile.is_file() else None
        if rfile.exists() and tok != RUN_TOKEN:
            # Another submission's: a task that never started (this job runs
            # afterany) never reached its rm -rf of the previous directory.
            stale[(etype, cv)] = tok
            data[etype][cv] = None
        elif rfile.exists():
            try:
                with open(rfile) as _f:
                    s = json.load(_f)["summary"]
                data[etype][cv] = s
            except (KeyError, json.JSONDecodeError):
                data[etype][cv] = None
        else:
            data[etype][cv] = None

# ---------------------------------------------------------------------------
# Print AUROC table
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

print()
print("=" * table_w)
# Stratified test AUROC (train.py's "auroc": pooled over observed (pair,
# cell-line) cells; for a one-head model this equals the OR-label auroc_any).
print("EMBEDDING BENCHMARK RESULTS (test AUROC, stratified)".center(table_w))
print("$(pca_step1_label "$PCA_ARM")".center(table_w))
print("=" * table_w)
print()
print(f"{'Embedding':<20s} | {header}")
print("-" * table_w)

successful = 0
failed = 0
for etype in emb_types:
    row = []
    for cv in cv_types:
        s = data[etype].get(cv)
        cell = "STALE" if (etype, cv) in stale else fmt(s, "auroc_mean", "auroc_std")
        row.append(cell)
        if s is not None and s.get("auroc_mean") is not None:
            successful += 1
        else:
            failed += 1
    row_str = " | ".join(f"{cell:<{col_w}s}" for cell in row)
    print(f"{etype:<20s} | {row_str}")

print()
print(f"Runs: {successful} successful, {failed} failed/missing/stale (of {len(emb_types) * len(cv_types)})")
if stale:
    print(f"STALE (not tabulated; run_token.txt is not RUN_TOKEN={RUN_TOKEN}): "
          + ", ".join(f"{e}_{c} ({t or 'no token'})" for (e, c), t in sorted(stale.items(), key=str)))
    print("  To tabulate an older array, sbatch this job with that array's RUN_TOKEN, "
          "its PCA_ARM ($PCA_ARM) and the submission's PCA_ARMS ($PCA_ARMS) exported.")
print("=" * table_w)

# AUPRG tables, both on the TEST split (mean over folds):
#   auprg      stratified, pooled over observed (pair, cell-line) cells
#   auprg_any  the OR label (SL in any line; score = max over heads), the view
#              the four-model comparison reports
# NEITHER is what selected the checkpoints: train.py early-stops and selects on
# the VALIDATION --select_metric (auprg_any by default; stratified auprg only
# under --select_metric auprg; the per-line macro never selects), and category
# winners rank on val_<select_metric>_mean (run_best_per_category_combo.sh).
# Each table is shown only when present in a loaded summary; older runs without
# the key skip it silently. The AUROC table and its success sentinel above are
# left as the mode-agnostic liveness check.
for key, title in (("auprg", "test AUPRG, stratified"),
                   ("auprg_any", "test AUPRG_any, OR label")):
    if not any((data[etype].get(cv) or {}).get(f"{key}_mean") is not None
               for etype in emb_types for cv in cv_types):
        continue
    print()
    print("=" * table_w)
    print(f"EMBEDDING BENCHMARK RESULTS ({title})".center(table_w))
    print("=" * table_w)
    print()
    print(f"{'Embedding':<20s} | {header}")
    print("-" * table_w)
    for etype in emb_types:
        row = ["STALE" if (etype, cv) in stale else
               fmt(data[etype].get(cv), f"{key}_mean", f"{key}_std") for cv in cv_types]
        row_str = " | ".join(f"{cell:<{col_w}s}" for cell in row)
        print(f"{etype:<20s} | {row_str}")
    print("=" * table_w)

# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------
summary = {
    "timestamp": datetime.now().isoformat(),
    "run_token": RUN_TOKEN,
    # Which single-embedding PCA these runs trained with, and the arms of this
    # run that read them (one or both of the pair).
    "pca_arms": "$(pca_step1_arms "$PCA_ARM")",
    "pca": "$(pca_step1_label "$PCA_ARM")",
    "pca_args": "$PCA_SINGLE_ARGS",
    "results": {},
}
for etype in emb_types:
    summary["results"][etype] = {}
    for cv in cv_types:
        s = data[etype].get(cv)
        if s is not None and s.get("auroc_mean") is not None:
            summary["results"][etype][cv] = {
                "auroc": fmt(s, "auroc_mean", "auroc_std"),
                "aupr": fmt(s, "aupr_mean", "aupr_std"),
                # AUPRG (test; stratified, and the OR label) / normalized AUPR.
                # Absent in an older results.json -> fmt() returns "N/A".
                "auprg": fmt(s, "auprg_mean", "auprg_std"),
                "auprg_any": fmt(s, "auprg_any_mean", "auprg_any_std"),
                "aupr_norm": fmt(s, "aupr_norm_mean", "aupr_norm_std"),
                "f1": fmt(s, "f1_mean", "f1_std"),
                "total_params": s.get("total_params", 0),
                "nonzero_params": s.get("nonzero_params", 0),
                "weight_sparsity": round(s.get("weight_sparsity") or 0, 2),
            }
        elif (etype, cv) in stale:
            summary["results"][etype][cv] = {
                "error": f"stale: run_token {stale[(etype, cv)] or 'missing'} is not {RUN_TOKEN}"}
        else:
            summary["results"][etype][cv] = {"error": "not found"}

out = results_dir / "embedding_benchmark_summary.json"
with open(out, "w") as _f:
    json.dump(summary, _f, indent=2)
print(f"\nJSON saved to: {out}")
PYTHON_SUMMARY

echo ""
echo "Completed at: $(date)"
