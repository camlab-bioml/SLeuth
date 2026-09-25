#!/usr/bin/env python3
"""
Score ONE model's per-fold predictions on the shared canonical folds, with the
single benchmark evaluator (cal_metrics) — so all four models are measured
identically on identical test sets.

A model writes, for each fold k of a CV type, a dense gene x gene score matrix
aligned to the shared gene index (sl_comparison/folds/genes.txt), named
    <pred_dir>/fold_<k>_<cv>_predictions.npy
This script loads each, scores it against that fold's test_pos / test_neg
(excluding train_pos edges from the ranking), aggregates mean/std across folds,
and writes results.json in the benchmark schema (auroc_mean, aupr_mean, ...),
which compare_sl_models.py already reads.

Usage:
  python sl_comparison/evaluate_model.py --model_name SLGNN --cv_type cv3 \
      --pred_dir SLGNN/results/slgnn_cv3 --out sl_comparison/results/SLGNN_cv3.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from evaluation import evaluate_predictions_dict
from auprg import auprg, normalized_aupr
from model_params import read_model_params, verify_fold_stamp
from retrieval_metrics import corrected_metrics

METRIC_KEYS = [
    "auroc",
    "f1",
    "aupr",
    "ap",
    "ap_norm",
    "auprg",
    "aupr_norm",
    "baseline",
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
    "recall@10",
    "recall@20",
    "recall@50",
    "precision@10",
    "precision@20",
    "precision@50",
    "map@10",
    "map@20",
    "map@50",
    # Correctly-normalised companions to the vendored ranking metrics — see
    # retrieval_metrics.py for what each defect is and which direction it bends.
    "ndcg@10_corrected",
    "ndcg@20_corrected",
    "ndcg@50_corrected",
    "map@10_corrected",
    "map@20_corrected",
    "map@50_corrected",
    "precision@10_true",
    "precision@20_true",
    "precision@50_true",
    # Diagnostic, not a score — see score_fold.
    "degenerate_row_frac",
]


def fold_provenance(folds_dir, cv_type):
    """Content fingerprint of the fold set that produced these numbers.

    Without this a results.json is untraceable: nothing in it says which gene
    universe, which conflict policy or which seed was used, so two runs whose
    tables disagree cannot be told apart. Hashes the files that define the
    answer key, so a regenerated fold set changes the digest even when
    meta.json still claims the same parameters.
    """
    root = Path(folds_dir)
    prov = {"folds_dir": str(root), "cv_type": cv_type}
    for name in ("genes.txt", "all_pos.npy", "meta.json"):
        p = root / name
        prov[f"{name}_sha256"] = (hashlib.sha256(
            p.read_bytes()).hexdigest()[:16] if p.exists() else None)
    meta = root / "meta.json"
    if meta.exists():
        try:
            prov["folds_meta"] = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    here = str(Path(__file__).resolve().parent)
    try:
        prov["git_sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=here).stdout.strip() or None
        # git_sha alone is misleading: it names the last COMMIT, not the code
        # that ran. Scoring with uncommitted edits records a SHA whose tree
        # never produced these numbers, and nothing downstream could tell.
        # Record the dirt so a reader knows the fingerprint is incomplete.
        # NO .strip() before splitlines: porcelain lines begin with a two-char
        # status field that is often " M", so stripping eats the leading space
        # of the FIRST line only and ln[3:] then over-slices it — recording
        # "gitignore" for ".gitignore" while every other path came out right.
        # Untracked files ARE included: a new module the pipeline imports but
        # nobody `git add`ed is exactly the state that makes git_sha a lie, and
        # --untracked-files=no hid it.
        dirty = [
            ln for ln in subprocess.run(["git", "status", "--porcelain"],
                                        capture_output=True,
                                        text=True,
                                        timeout=10,
                                        cwd=here).stdout.splitlines()
            if ln.strip()
        ]
        prov["git_dirty"] = bool(dirty)
        if dirty:
            paths = [ln[3:] for ln in dirty]
            prov["git_dirty_n"] = len(paths)
            # Code first, then everything else. A plain sorted()[:40] over 133
            # entries was filled entirely by deleted siamese_sl/slurm/logs/*
            # and named not one .py file — the truncation hid exactly what the
            # field exists to record.
            code = sorted(p for p in paths
                          if p.endswith((".py", ".sh", ".json", ".md")))
            rest = sorted(p for p in paths if p not in set(code))
            prov["git_dirty_files"] = (code + rest)[:40]
    except (OSError, subprocess.SubprocessError):
        prov["git_sha"] = None
        prov["git_dirty"] = None
    return prov


def seen_for(fold_dir, test_pos, train_pos):
    """Known SL pairs to suppress from the ranking: ALL of them except this
    fold's test positives.

    cal_metrics writes the -999999 sentinel over seen_index, so anything left
    out stays rankable and is scored as a NON-relevant item in a test gene's
    top-100. Passing train_pos alone therefore penalises a model for ranking a
    genuine SL pair highly — the validation positives, and on cv3 the pairs
    split_by_genes assigns to neither partition, are all real SL.

    Falls back to train_pos (with a warning) for fold sets built before
    all_pos.npy existed, so old fold directories still score rather than crash.
    """
    p = Path(fold_dir).parent.parent / "all_pos.npy"
    if not p.exists():
        print(
            f"  WARNING: {p} missing — masking only train_pos. Known SL "
            f"pairs outside this fold will count as false positives and "
            f"deflate every @k metric. Regenerate folds with prepare_folds.py."
        )
        return train_pos
    all_pos = np.load(p).astype(np.int64)
    keep = set(map(tuple, test_pos.tolist()))
    mask = np.fromiter((tuple(e) not in keep for e in all_pos.tolist()),
                       dtype=bool,
                       count=len(all_pos))
    return all_pos[mask]


def fold_dirs_for(folds_dir, cv_type):
    """Ordered fold_* directories of one CV type."""
    folds_root = Path(folds_dir) / cv_type
    dirs = sorted(folds_root.glob("fold_*"),
                  key=lambda d: int(d.name.split("_")[1]))
    if not dirs:
        raise SystemExit(f"No folds under {folds_root}")
    return dirs


def score_fold(score, fold_dir, label):
    """Metrics for ONE score matrix on ONE fold, or None if the fold is
    degenerate (no test positives or no test negatives).

    Shared by evaluate_model's CLI and degree_baseline.py, so a control row is
    scored by exactly the same code as a model row — same affine shift, same
    seen mask, same corrected retrieval companions. `label` appears only in
    messages.
    """
    d = Path(fold_dir)
    test_pos = np.load(d / "test_pos.npy").astype(np.int64)
    test_neg = np.load(d / "test_neg.npy").astype(np.int64)
    train_pos = np.load(d / "train_pos.npy").astype(np.int64)
    if score.shape[0] != score.shape[1]:
        raise SystemExit(f"{label}: score matrix not square ({score.shape})")

    # THE MATRIX MUST SPAN THE FOLD SET'S GENE UNIVERSE.
    #
    # Only squareness was checked before, so a matrix built over a different
    # gene indexing scored happily: the fold files index into genes.txt, and if
    # the matrix is a different width those indices address the wrong genes (or
    # raise, if the matrix is smaller). A 900x900 random matrix scored against
    # the 8,844-gene folds and produced plausible-looking numbers.
    genes_txt = Path(fold_dir).parent.parent / "genes.txt"
    if genes_txt.exists():
        n_genes = sum(1 for ln in open(genes_txt) if ln.strip())
        if score.shape[0] != n_genes:
            raise SystemExit(
                f"{label}: score matrix is {score.shape[0]}x{score.shape[0]} "
                f"but {genes_txt} defines {n_genes} genes. The fold arrays "
                f"index into genes.txt, so these scores address the wrong "
                f"genes. Re-export the matrix over the shared universe.")
    else:
        n_genes = int(score.shape[0])
        print(f"  WARNING: {genes_txt} missing — cannot confirm the score "
              f"matrix spans the fold set's gene universe.")

    # MAKE THE RANKING METRICS AFFINE-INVARIANT.
    #
    # cal_metrics writes a literal 0 onto the diagonal (:1400) and -999999
    # onto seen edges — absolute values on a scale it does not own. A model
    # whose scores are all negative (raw logits, unnormalised dot products)
    # therefore has the injected 0 outrank EVERY real candidate, losing slot #1
    # of every gene's top-100, while a model emitting sigmoids keeps it.
    # Measured on a synthetic matrix: the SAME ranking scores ndcg@10 0.0097
    # when negative and 0.0123 when shifted positive.
    #
    # Every metric here is rank- or threshold-based, so a strictly increasing
    # affine map is provably neutral — and applying it to all four models makes
    # the comparison independent of each model's output convention. siamese
    # already did this internally; doing it here means SLGNN / NSF4SL / SLMGAE
    # get the same treatment.
    if not np.isfinite(score).all():
        raise SystemExit(f"{label}: non-finite entries in score matrix")
    score = (score - score.min() + np.float32(1.0)).astype(np.float32)

    # cal_metrics' ranking loop assumes >=1 test positive (and both classes);
    # skip a degenerate fold rather than crash (can happen with tiny
    # --max_genes preflight subsets, not the full folds).
    if len(test_pos) == 0 or len(test_neg) == 0:
        print(f"  [skip] {label}: {len(test_pos)} pos / {len(test_neg)} neg "
              f"test edges (degenerate)")
        return None

    # DEGENERATE-ROW DIAGNOSTIC.
    #
    # Every @k metric is an argsort of one matrix row. If that row is CONSTANT
    # the ranking is an arbitrary tie-break, so the score is a property of the
    # sort implementation rather than of the model — and it is not reproducible
    # across numpy versions. Measured on SLMGAE (features are the training
    # adjacency, so a gene with no training edge gets a zero embedding):
    # 224/2212 scored cv1 test genes, 360/1875 on cv2, 67/212 on cv3. Re-scoring
    # the identical matrices on a different machine moved SLMGAE's cv3 recall@50
    # from 0.000882 to 0.000151 while every other model's retrieval column was
    # bit-identical. The other three models have zero constant rows.
    scored_genes = np.unique(np.concatenate([test_pos[:, 0], test_pos[:, 1]]))
    rows = score[scored_genes]
    m_degen = float(np.mean(rows.max(axis=1) == rows.min(axis=1)))

    seen = seen_for(d, test_pos, train_pos)
    m = evaluate_predictions_dict(score, test_pos, test_neg, seen)
    corrected = corrected_metrics(score, test_pos, seen)

    # SELF-VALIDATION, not decoration. retrieval_metrics rebuilds cal_metrics'
    # ranking from scratch; if the two constructions ever diverge, the
    # *_corrected columns describe a DIFFERENT ranking than the vendored ones
    # sitting beside them, which is worse than not publishing them at all.
    # recall@k is the one metric both compute with the same normaliser, so it
    # is the exact equality that proves the rankings match. Checked every fold.
    for k in (10, 20, 50):
        got, want = corrected.pop(f"recall@{k}_check"), m[f"recall@{k}"]
        if not (np.isnan(got) and np.isnan(want)) and \
                not np.isclose(got, want, rtol=0, atol=1e-12):
            raise SystemExit(
                f"{label}: retrieval_metrics diverged from cal_metrics at "
                f"recall@{k} ({got!r} vs {want!r}). The *_corrected columns "
                f"would describe a different ranking — refusing to write them."
            )
    m.update(corrected)
    m["degenerate_row_frac"] = m_degen
    m["n_genes"] = n_genes
    if m_degen > 0.05:
        print(
            f"  WARNING: {label}: {m_degen:.1%} of scored test genes have a "
            f"CONSTANT score row — their @k metrics are tie-break artifacts, "
            f"not model output, and will not reproduce across numpy versions.")

    # Baseline-independent metrics (fair across differing prevalences) —
    # computed from the same test-edge scores cal_metrics uses.
    pos_s = score[test_pos[:, 0], test_pos[:, 1]]
    neg_s = score[test_neg[:, 0], test_neg[:, 1]]
    y = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    s = np.concatenate([pos_s, neg_s])
    m["baseline"] = float(np.mean(y))
    m["auprg"] = auprg(y, s)
    m["aupr_norm"] = normalized_aupr(m["aupr"], m["baseline"])

    # AVERAGE PRECISION, alongside the vendored trapezoidal AUPR.
    #
    # cal_metrics:1396 uses auc(recall, precision), which linearly interpolates
    # between PR points. That interpolation is not achievable by any classifier
    # and it rewards ties: a CONSTANT score matrix scores exactly (1 + pi) / 2,
    # which on the 2026-08-06 cv3 fold set was 0.5786 — above EVERY model in
    # that table (siamese 0.5245, SLMGAE 0.4041). The constant is fold-set
    # specific: pi moved when the 2026-08-10 label rule made the 5,542
    # conflicted pairs positive. The (1 + pi) / 2 identity is not. Average
    # precision is the step-wise sum giving a constant predictor exactly pi.
    #
    # `aupr` is kept because published SLMGAE / SLGNN / NSF4SL tables use the
    # trapezoidal form and dropping it would break comparability with them;
    # `ap` is the one to report as ours.
    m["ap"] = float(average_precision_score(y, s))
    m["ap_norm"] = normalized_aupr(m["ap"], m["baseline"])

    nonfinite = sorted(k for k in METRIC_KEYS
                       if k in m and not np.isfinite(m[k]))
    if nonfinite:
        # F1 is NOT the risk here: the vendored cal_metrics already adds an
        # epsilon (`max(2*P*R/(P+R+1e-10))`), so an all-zero PR point yields 0,
        # not NaN. What can still be non-finite is the prevalence-anchored
        # family — auprg, aupr_norm, ap_norm — which is undefined at degenerate
        # prevalence (a fold with no positives or no negatives). Refuse to
        # publish a silent NaN either way: it would propagate into every
        # mean/std downstream.
        raise SystemExit(f"{label}: non-finite metric(s) {nonfinite}")
    return m


# Annotation keys that describe HOW a row was produced rather than what it
# scored. collect_siamese.py writes them and the shared re-grade overwrites the
# same file, so without carrying them forward they vanish — which is how
# compare.py came to reference a `note` field that existed in none of the three
# siamese JSONs.
#
# `note` is deliberately NOT carried. collect_siamese's note describes ITS
# numbers, which were scored against siamese's own 19,397-gene rank universe
# (rank_universe_n is 19397 in every run, never 8844). Once this script has
# re-graded the saved matrices the metrics are shared-universe numbers, and
# carrying that note forward would re-attach a retrieval caveat that is false
# for the values sitting next to it. write_summary attaches an accurate one.
# `rank_universe_n` is NOT carried either: it records the universe siamese
# ranked against in ITS OWN scoring (19,397), and next to a shared-evaluator
# note quoting the matrix's actual universe the two flatly contradict each
# other in the same object. It describes the discarded numbers, not these.
# `label_semantics` MUST be carried: it says what a positive MEANS for this row
# (siamese's is OR-collapsed over 8 cell-line heads; every other row uses the
# single unlabelled SL relation). Re-grading changes how the scores are
# measured, never what the labels mean, so this caveat outlives the re-grade —
# and it is the most important qualifier on the headline row.
CARRY_KEYS = ("selection_audit", "label_semantics")

# Templated from what is MEASURED, never asserted. An earlier version hardcoded
# "on the shared canonical folds ... as every other row", which a 300-gene
# throwaway --folds_dir reproduced verbatim. The fold directory and the ranking
# universe are facts this script has; canonicity is not.
SHARED_NOTE = (
    "Scored by sl_comparison/evaluate_model.py against folds_dir={folds} "
    "({n:,}-gene ranking universe, {k} folds). Every row graded through this "
    "script with the same folds_dir shares its test edges, ranking universe "
    "and cal_metrics call — check the `provenance` fingerprints to confirm two "
    "rows really are comparable. Ranking columns use the vendored normalisers; "
    "retrieval_metrics.py supplies corrected companions (*_corrected/*_true).")


def build_summary(model_name,
                  cv_type,
                  folds_dir,
                  fold_metrics,
                  pred_dir=None,
                  carry_from=None):
    """Aggregate per-fold dicts into the results.json schema."""
    # The ranking universe is the SCORE MATRIX's width, which score_fold has
    # already checked against genes.txt. Reading genes.txt again here would put
    # an unguarded open() after every fold has been scored — the worst place to
    # fail, since the expensive work is done and nothing has been written yet.
    n_genes = fold_metrics[0].get("n_genes")
    summary = {
        "model":
        model_name,
        "cv_type":
        cv_type,
        "n_folds_scored":
        len(fold_metrics),
        "note":
        SHARED_NOTE.format(folds=folds_dir,
                           n=n_genes or -1,
                           k=len(fold_metrics)),
        "provenance":
        fold_provenance(folds_dir, cv_type),
        "folds":
        fold_metrics,
    }
    # Trainable-parameter count, written next to the predictions by the trainer
    # (model_params.write_model_params). Absent for a model that has not been
    # updated yet — compare.py then just omits the count for that row.
    if pred_dir is not None:
        summary.update(read_model_params(pred_dir))
    if carry_from is not None and Path(carry_from).exists():
        try:
            prev = json.loads(Path(carry_from).read_text()).get("summary", {})
        except (OSError, json.JSONDecodeError):
            prev = {}
        for key in CARRY_KEYS:
            if key in prev:
                summary[key] = prev[key]
    for key in METRIC_KEYS:
        # Name the missing key and fold. A bare m[key] raised a KeyError with
        # no context whenever METRIC_KEYS gained an entry mid-regrade, which is
        # exactly when it happens — half the JSONs then silently lack the new
        # column and compare.py renders the gap as "-", indistinguishable from
        # a measured zero.
        missing = [i for i, m in enumerate(fold_metrics) if key not in m]
        if missing:
            raise SystemExit(
                f"{model_name}/{cv_type}: metric '{key}' missing from fold(s) "
                f"{missing}. The fold records were produced by a different "
                f"version of score_fold — re-grade every fold in one pass.")
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return summary


def write_summary(summary, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary}, f, indent=2)
    tp = summary.get("trainable_params")
    name = (f"{summary['model']}({tp:,})"
            if isinstance(tp, int) else summary["model"])
    print(f"\n{name}/{summary['cv_type']}: "
          f"AUROC {summary['auroc_mean']:.4f} | AP {summary['ap_mean']:.4f} "
          f"| AUPRG {summary['auprg_mean']:.4f} "
          f"| R@50 {summary['recall@50_mean']:.4f}  -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Score a model on shared folds")
    ap.add_argument("--folds_dir", default="sl_comparison/folds")
    ap.add_argument("--cv_type", required=True, choices=["cv1", "cv2", "cv3"])
    ap.add_argument(
        "--pred_dir",
        required=True,
        help="Dir with fold_<k>_<cv>_predictions.npy score matrices")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out", required=True, help="Output results.json path")
    ap.add_argument("--carry_from",
                    help="Existing results.json whose annotation keys "
                    f"({', '.join(CARRY_KEYS)}) should survive this "
                    "re-grade. Safe to point at --out itself.")
    ap.add_argument(
        "--allow_partial",
        action="store_true",
        help="Summarise even when fewer folds were scored than the "
        "fold set contains. Off by default: a partial mean "
        "reads identically to a complete one in the table.")
    ap.add_argument("--allow_unstamped",
                    action="store_true",
                    help="Grade matrices that carry no fold_stamp (produced "
                    "before stamping existed). A stamp MISMATCH is still "
                    "fatal — this only downgrades a MISSING stamp to a "
                    "warning.")
    args = ap.parse_args()

    # BEFORE anything is scored. The matrices are joined to the answer key by
    # directory path alone, so this is the only thing standing between a
    # leftover matrix from a previous fold set and a published mean. Checked
    # up front rather than per fold: a failure after four folds have been
    # scored wastes the work and reads like a partial success.
    verify_fold_stamp(args.pred_dir,
                      args.folds_dir,
                      args.cv_type,
                      allow_unstamped=args.allow_unstamped)

    fold_metrics = []
    for d in fold_dirs_for(args.folds_dir, args.cv_type):
        k = int(d.name.split("_")[1])
        pred_path = Path(args.pred_dir) / \
            f"fold_{k}_{args.cv_type}_predictions.npy"
        if not pred_path.exists():
            print(f"  [skip] missing prediction: {pred_path}")
            continue
        m = score_fold(np.load(pred_path), d, str(pred_path))
        if m is None:
            continue
        fold_metrics.append(m)
        print(f"  fold {k}: AUROC {m['auroc']:.4f} AP {m['ap']:.4f} "
              f"AUPRG {m['auprg']:.4f} NDCG@10 {m['ndcg@10']:.4f} "
              f"(corrected {m['ndcg@10_corrected']:.4f})")

    if not fold_metrics:
        raise SystemExit(f"No predictions found for {args.model_name}/"
                         f"{args.cv_type} in {args.pred_dir}")

    # A PARTIAL fold set must not publish a mean. n_folds_scored was recorded
    # in the summary but nothing ever read it, so a run that died after fold 2
    # produced a 3-fold mean that is indistinguishable in the table from a full
    # 5-fold one — and CV3's per-fold prevalence spans 0.084-0.264, so which
    # folds are missing moves the number materially. Refuse by default; the
    # opt-out is explicit and lands in the summary via the CLI record.
    n_expected = len(fold_dirs_for(args.folds_dir, args.cv_type))
    if len(fold_metrics) != n_expected and not args.allow_partial:
        raise SystemExit(
            f"REFUSING to summarise {args.model_name}/{args.cv_type}: scored "
            f"{len(fold_metrics)} of {n_expected} folds. A partial mean is not "
            f"comparable to a complete one. Re-run the missing folds, or pass "
            f"--allow_partial to publish it anyway (the count is recorded as "
            f"n_folds_scored either way).")

    summary = build_summary(args.model_name,
                            args.cv_type,
                            args.folds_dir,
                            fold_metrics,
                            pred_dir=args.pred_dir,
                            carry_from=args.carry_from)
    write_summary(summary, args.out)


if __name__ == "__main__":
    main()
