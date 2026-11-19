# Sourced (never run) by run_benchmarking_siamese_vs_others.sh (step 0) and
# sl_comparison/slurm/run_shared_folds.sh; needs $PYTHON_PATH and the repo root
# as the working directory.
#
# folds_manifest: one "<sha256>  <path>" line per file under sl_comparison/folds,
# sorted by path (relative to sl_comparison/folds). Git-free on purpose: the
# cluster checkout is updated by file upload, not by git. Step 0 writes it to
# sl_comparison/slurm/logs/folds_<RUN_TOKEN>_<timestamp>.sha256 and exports
# that path as FOLDS_MANIFEST, so the folds job can tell a changed fold set
# (fatal) from a changed fold source with the fold files untouched (a warning).
folds_manifest() {
    "$PYTHON_PATH" -c '
import hashlib, os
root = "sl_comparison/folds"
rows = []
for dp, _, fn in os.walk(root):
    for f in fn:
        p = os.path.join(dp, f)
        with open(p, "rb") as fh:
            rows.append((os.path.relpath(p, root), hashlib.sha256(fh.read()).hexdigest()))
for rel, h in sorted(rows):
    print(h + "  " + rel)
'
}
