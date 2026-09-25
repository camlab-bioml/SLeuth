#!/usr/bin/env python3
"""
Collect siamese_sl's OR-collapse (`*_any`) metrics into the unified comparison
schema, so it sits in the same table as the shared-fold models.

siamese_sl runs its OWN pipeline (its embedding selection + cell-line model on
SLDB 3.0), and natively reports the OR / "SL in any cell line" view as
`auroc_any`, `aupr_any`, `auprg_any`, `aupr_any_norm` in its results.json. This
maps those onto the common metric names (auroc/aupr/auprg/aupr_norm) and writes
sl_comparison/results/siamese_sl_<cv>.json.

When siamese runs with train.py --folds_dir it uses the SAME canonical folds,
gene universe and test edges as SLMGAE / SLGNN / NSF4SL, and run_shared_models.sh
re-grades its saved matrices through evaluate_model.py so the numbers come from
the identical evaluator. The legacy path (no --folds_dir) uses siamese's own
gene-disjoint CV with its own negative set; convert() detects which happened via
`rank_universe_n` and attaches the appropriate caveat.

Usage:
  python sl_comparison/collect_siamese.py --siamese_results <path/results.json> \
      --cv_type cv3 --out sl_comparison/results/siamese_sl_cv3.json
"""

import argparse
import json
import re
from pathlib import Path

from model_params import read_model_params

# siamese *_any key  ->  common comparison key
KEY_MAP = {
    "auroc": "auroc_any",
    "aupr": "aupr_any",
    "auprg": "auprg_any",
    "aupr_norm": "aupr_any_norm",
    "baseline": "baseline_any",
    "f1": "f1_any",
    # Retrieval metrics (rank each test gene against the WHOLE universe, not
    # just the test negatives) - siamese emits these only when train.py runs
    # an all-pairs scoring pass; absent keys are simply skipped by convert().
    "ndcg@10": "ndcg@10_any",
    "ndcg@20": "ndcg@20_any",
    "ndcg@50": "ndcg@50_any",
    "recall@10": "recall@10_any",
    "recall@20": "recall@20_any",
    "recall@50": "recall@50_any",
    "precision@10": "precision@10_any",
    "precision@20": "precision@20_any",
    "precision@50": "precision@50_any",
    "map@10": "map@10_any",
    "map@20": "map@20_any",
    "map@50": "map@50_any",
}

SHARED_UNIVERSE = 8844  # wc -l sl_comparison/folds/genes.txt

# Filled in by convert() with the fold's measured universe size. Kept as a
# template so the number is never hard-coded.
RETRIEVAL_CAVEAT_HEAD = (
    " RETRIEVAL CAVEAT - these are siamese's SELF-SCORED numbers (this file is "
    "written before, and normally replaced by, the shared-evaluator re-grade in "
    "collect_and_regrade_siamese.sh; if you are reading this on a row in the "
    "comparison table, the re-grade did NOT run). ndcg@k / recall@k / "
    "precision@k / map@k are NOT comparable against the other models here: "
    "siamese ranked each test gene against its OWN {n:,}-gene candidate "
    "universe, while SLMGAE / SLGNN / NSF4SL rank against the shared "
    "{shared:,}-gene universe (sl_comparison/folds/genes.txt).")

# The direction of the bias depends on WHICH universe is bigger, and it is not
# a fixed sign. An earlier version assumed siamese's universe was the smaller
# one (it was ~5.3k when that text was written); with siamese now at 19,397
# against a shared 8,844 the arithmetic produced "The 0 genes ... are pure hard
# distractors" and stated the advantage backwards. Pick the branch from the
# measured sizes instead of asserting one.
RETRIEVAL_CAVEAT_SMALLER = (
    " Siamese's universe is the SMALLER one: the {extra:,} genes present in "
    "the shared universe but not siamese's appear in ZERO positive pairs, so "
    "they are pure hard distractors the other three must out-rank and siamese "
    "never sees. A random ranker scores recall@50 = 50/{n:,} here vs "
    "50/{shared:,} there ({ratio:.2f}x). Treat these as within-siamese "
    "diagnostics or as an UPPER bound on siamese's shared-universe value.")
RETRIEVAL_CAVEAT_LARGER = (
    " Siamese's universe is the LARGER one ({n:,} vs {shared:,}): it ranks "
    "against {extra:,} EXTRA candidates the other three never see, so a random "
    "ranker scores recall@50 = 50/{n:,} here vs 50/{shared:,} there "
    "({ratio:.2f}x) — the bias runs AGAINST siamese, and these values are a "
    "LOWER bound on its shared-universe value. Do not 'correct' for it by "
    "assuming siamese was flattered.")
RETRIEVAL_CAVEAT_TAIL = (
    " Note also that this evaluator's 'precision@k' divides by "
    "min(k, #relevant), not k (preprocess_benchmarking_paper.py:1443-1448), so "
    "it equals recall@k for any gene with <=k test positives - all four models "
    "get that same definition from the same function. For cross-model ranking "
    "use the baseline-anchored edge metrics: AUPRG, AP_norm and AUROC.")

# There is deliberately NO "shared universe" variant of the caveat here.
#
# An earlier attempt branched on `rank_universe_n == SHARED_UNIVERSE` to emit a
# reassuring note when siamese had used the shared folds. That branch was dead:
# siamese always ranks internally against its own gene set, so rank_universe_n
# is 19,397 in every run and never 8,844 — the condition could not fire. Worse,
# the reassurance belongs to the SHARED-EVALUATOR numbers, which this file never
# produces; evaluate_model.SHARED_NOTE attaches it there, and `note` is
# deliberately excluded from evaluate_model.CARRY_KEYS so the caveat below
# cannot survive a re-grade and end up describing values it is false for.


def convert(summary, cv_type):
    """Map a siamese summary's *_any keys onto the common schema (or None)."""
    out = {
        "model":
        "siamese_sl",
        "cv_type":
        cv_type,
        # SEPARATE from `note` on purpose. `note` describes how THESE numbers
        # were scored and is correctly replaced when evaluate_model.py re-grades
        # the matrices. Label semantics survive the re-grade untouched — the
        # positives are still OR-collapsed over cell lines however they are
        # scored — so this must be carried forward (evaluate_model.CARRY_KEYS).
        # It is the single most important caveat on the siamese row and it
        # briefly vanished from every shipped artifact when `note` stopped
        # being carried.
        "label_semantics":
        "OR-collapse: a pair is POSITIVE if it is SL in ANY of the 8 cell-line "
        "heads. Since 2026-08-10 the shared folds use the SAME rule "
        "(conflict_policy=or_collapse: positive if any screen called it SL in "
        "any line, negative only if every screen called it non-SL), so this row "
        "and the SLGNN / NSF4SL / SLMGAE / DegreePrior rows are graded on an "
        "identical positive set — including the 5,542 pairs screened both ways, "
        "which every row now counts as positive. What still differs is the "
        "SCORE: siamese emits 8 per-head logits collapsed by max, the others "
        "emit one score for the unlabelled relation.",
        "note":
        "AUPRG / AP_norm / AUROC are the cross-model metrics; AUPRG and "
        "AP_norm are baseline-anchored skill scores (random -> 0 at any "
        "prevalence), not prevalence-invariant."
    }
    found = []
    for common, src in KEY_MAP.items():
        mean = summary.get(f"{src}_mean")
        if mean is None:
            continue
        out[f"{common}_mean"] = mean
        out[f"{common}_std"] = summary.get(f"{src}_std")
        found.append(common)
    n = summary.get("rank_universe_n")
    if n and any(
            k.startswith(("ndcg@", "recall@", "precision@", "map@"))
            for k in found):
        n = int(n)
        out["rank_universe_n"] = n
        fmt = dict(n=n,
                   shared=SHARED_UNIVERSE,
                   extra=abs(SHARED_UNIVERSE - n),
                   ratio=SHARED_UNIVERSE / n)
        body = (RETRIEVAL_CAVEAT_SMALLER if n < SHARED_UNIVERSE else
                RETRIEVAL_CAVEAT_LARGER if n > SHARED_UNIVERSE else "")
        out["note"] += (RETRIEVAL_CAVEAT_HEAD.format(**fmt) +
                        body.format(**fmt) + RETRIEVAL_CAVEAT_TAIL)
    return (out, found) if found else (None, [])


def nested_loo_audit(cands):
    """Winner's-curse correction for picking 1-of-N configs on validation.

    Selecting the config with the best mean val_auprg and then reporting that
    config's mean TEST metric reuses the same folds for selection and for
    reporting. The unbiased version is nested: for each fold k, choose the
    config maximising mean val_auprg over the OTHER folds, and score it on
    fold k only.

    On the 2026-08-06 run this left cv1 unchanged and cv2 nearly so, but moved
    cv3 from 0.8797 to ~0.845 — the honest number to publish. The residual is
    NOT a constant offset: it is dominated by single folds where two configs
    are separated by a val margin far smaller than the fold-to-fold spread, so
    report it with the per-fold choices rather than as a flat correction.

    siamese_sl is the only model in this table selected over a config sweep;
    SLGNN and NSF4SL each get one hard-coded embedding and SLMGAE one config,
    so disclosing this is what keeps the comparison symmetric.
    """
    if len(cands) < 2:
        return None
    n_folds = min(len(pv) for _, pv, _ in cands)
    if n_folds < 2:
        return None
    per_fold, choices = [], []
    for k in range(n_folds):
        # argmax over configs of the val score on every fold EXCEPT k.
        best_i, best_v = None, None
        for i, (_p, pv, _pt) in enumerate(cands):
            v = sum(pv[j] for j in range(n_folds) if j != k) / (n_folds - 1)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        per_fold.append(cands[best_i][2][k])
        choices.append({
            "fold": k,
            "config": Path(cands[best_i][0]).parent.name,
            "held_out_val_mean": round(best_v, 6),
        })
    mean = sum(per_fold) / len(per_fold)
    # The config the TABLE reports: argmax of mean val over ALL folds. The
    # per-fold picks must be compared against THIS, not against their own mode
    # — the mode is a different quantity and on a split vote it can name a
    # config the table never used (and tie-breaks nondeterministically).
    all_fold_best = max(range(len(cands)),
                        key=lambda i: sum(cands[i][1][:n_folds]) / n_folds)
    return {
        "n_candidates":
        len(cands),
        "reported_config":
        Path(cands[all_fold_best][0]).parent.name,
        "nested_loo_auprg_mean":
        mean,
        "nested_loo_auprg_per_fold":
        per_fold,
        "per_fold_choice":
        choices,
        "method":
        "leave-one-fold-out config selection on val_auprg; the "
        "reported row selects on all folds at once and is optimistic "
        "by the difference",
    }


def write_out(out, found, cv_type, outp):
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump({"summary": out}, f, indent=2)
    msg = f"siamese_sl/{cv_type}: collected {found} -> {outp}"
    if "auprg" in found:
        msg += f"  [AUPRG(any)={out['auprg_mean']:.4f}]"
    print(msg)


def discover(discover_dir, out_dir):
    """Scan for siamese results.json files with *_any metrics.

    cv type is inferred from the path (cv1/cv2/cv3). If several match a CV, the
    best is kept — ranked on val_auprg_mean when the run used the shared folds
    (train.py --folds_dir), else on auprg_any_mean.

    Ranking ~17 candidate configs on a TEST metric is a winner's curse larger
    than the per-epoch one it sits on top of: the reported number is then a
    maximum over 17 noisy test estimates. Runs carrying val_auprg_mean are
    selected honestly; runs without it fall back to the legacy test ranking and
    say so, because refusing to rank at all would drop siamese from the table.
    """
    best = {}  # cv -> (score, summary, path)
    candidates = {}  # cv -> [(path, per_fold_val, per_fold_test)]
    n_val = n_test = 0
    for path in Path(discover_dir).rglob("results.json"):
        try:
            data = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary", {})
        if "auroc_any_mean" not in summary:
            continue
        # siamese writes cv_type at the TOP level (not inside `summary`); fall
        # back to the path only if neither carries it.
        m = re.search(r"cv[123]", str(path))
        cv = (summary.get("cv_type") or data.get("cv_type")
              or (m.group(0) if m else None))
        if cv not in ("cv1", "cv2", "cv3"):
            continue
        if summary.get("val_auprg_mean") is not None:
            score, basis = summary["val_auprg_mean"], "val"
            n_val += 1
        else:
            score, basis = (summary.get("auprg_any_mean") or -1), "TEST"
            n_test += 1
        # Per-fold arrays for the nested-selection audit below.
        fm = data.get("fold_metrics") or []
        pv = [f.get("val_auprg") for f in fm]
        pt = [f.get("auprg_any") for f in fm]
        if fm and all(v is not None for v in pv) and \
                all(t is not None for t in pt):
            candidates.setdefault(cv, []).append((path, pv, pt))
        # Rank on (has_val, score) so a val-selected run always outranks a
        # test-selected one — the two scores are different quantities and must
        # never be compared numerically against each other.
        key = (basis == "val", score)
        if cv not in best or key > (best[cv][3] == "val", best[cv][0]):
            best[cv] = (score, summary, path, basis)
    if not best:
        print(f"[discover] no siamese *_any results under {discover_dir}")
        return
    if n_test:
        print(f"[discover] WARNING: {n_test} candidate run(s) carry no "
              f"val_auprg_mean — ranked on TEST auprg_any (winner's curse "
              f"over {n_test} configs). Re-run with train.py --folds_dir.")
    for cv, (score, summary, path, basis) in sorted(best.items()):
        out, found = convert(summary, cv)
        audit = nested_loo_audit(candidates.get(cv, []))
        if out is not None and audit:
            out["selection_audit"] = audit
            rep = out.get("auprg_mean")
            if rep is not None:
                # `delta` is nested - reported: the ADDITIVE CORRECTION to
                # apply to the reported number, which is what compare.py wants
                # and what the parenthetical below prints.
                # `optimism` is the opposite sign by convention (Efron): how
                # much the reported score EXCEEDS the honest nested estimate,
                # so a positive optimism means the selection flattered us.
                # These were the same field, which made a positive `optimism`
                # read as "we over-reported" when it actually meant the reverse.
                delta = audit["nested_loo_auprg_mean"] - rep
                audit["reported_auprg_mean"] = rep
                audit["nested_minus_reported"] = delta
                audit["optimism"] = -delta
                print(
                    f"[selection] {cv}: reported AUPRG {rep:.4f} over "
                    f"{audit['n_candidates']} configs; nested leave-one-fold-"
                    f"out {audit['nested_loo_auprg_mean']:.4f} "
                    f"({delta:+.4f})")
        # Model size, so siamese carries a parameter count in the table even on
        # the legacy path where evaluate_model.py never re-grades it.
        if out is not None:
            out.update(read_model_params(path.parent))
        print(f"[discover] {cv} <- {path}  [selected on {basis}, "
              f"score={score:.4f}]")
        write_out(out, found, cv, Path(out_dir) / f"siamese_sl_{cv}.json")
        # Record which run won so run_shared_models.sh can re-grade its
        # prediction matrices through evaluate_model.py — the same evaluator,
        # test edges and gene universe as the other three models. Without this
        # the table mixes siamese's own scoring with the shared one.
        marker = Path(out_dir) / f"siamese_selected_{cv}.txt"
        marker.write_text(str(path.parent) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Collect siamese_sl *_any metrics")
    ap.add_argument("--siamese_results",
                    help="Path to one siamese results.json (single-file mode)")
    ap.add_argument("--cv_type", choices=["cv1", "cv2", "cv3"])
    ap.add_argument("--out", help="Output json (single-file mode)")
    ap.add_argument("--discover_dir",
                    help="Scan this dir for siamese results.json (auto mode)")
    ap.add_argument("--out_dir",
                    default="sl_comparison/results",
                    help="Where auto mode writes siamese_sl_<cv>.json")
    args = ap.parse_args()

    if args.discover_dir:
        discover(args.discover_dir, args.out_dir)
        return
    if not (args.siamese_results and args.cv_type and args.out):
        ap.error("provide --discover_dir, OR all of "
                 "--siamese_results/--cv_type/--out")
    summary = json.load(open(args.siamese_results)).get("summary", {})
    out, found = convert(summary, args.cv_type)
    if not found:
        raise SystemExit(
            f"No *_any metrics in {args.siamese_results} (expected "
            f"auroc_any_mean etc.). Was this a cell-line run?")
    write_out(out, found, args.cv_type, args.out)


if __name__ == "__main__":
    main()
