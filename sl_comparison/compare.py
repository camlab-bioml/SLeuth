#!/usr/bin/env python3
"""
Unified four-model comparison table.

Reads every sl_comparison/results/<Model>_<cv>.json (all written in one schema:
evaluate_model.py for SLMGAE/SLGNN/NSF4SL, collect_siamese.py for siamese_sl),
and prints one table per CV type plus a tidy CSV.

AUPRG, AP_norm and AUROC are the headline metrics. AUPRG and AP_norm are
BASELINE-ANCHORED SKILL SCORES: a random ranker scores 0 at any prevalence, so
they are comparable across folds whose test-set base rate differs. They are NOT
prevalence-invariant — hold a ranker fixed and vary pi and both move (measured
2026-08-06 on siamese cv3/fold_1, AUROC held at 0.8105 and pi varied
synthetically: pi 0.2550 -> AP_norm 0.5509 /
AUPRG 0.8473; pi 0.0167 -> 0.1286 / 0.8980). AUROC is the only genuinely
prevalence-invariant column. Because every model shares the same per-fold test
set here, no cross-model comparison in this table is affected.
"""

import argparse
import builtins
import csv
import json
from pathlib import Path

# DegreePrior last: it is a zero-parameter control, not a competitor.
MODEL_ORDER = ["siamese_sl", "SLMGAE", "SLGNN", "NSF4SL", "DegreePrior"]
CV_ORDER = ["cv1", "cv2", "cv3"]
CV_LABELS = {
    "cv1": "CV1 (edge)",
    "cv2": "CV2 (gene)",
    "cv3": "CV3 (pair/cold-start)"
}
# (label, key). Skill scores first; AP (average precision) before the vendored
# trapezoidal AUPR, which is kept only for comparability with published tables.
METRICS = [
    ("AUPRG", "auprg"),
    ("AP_norm", "ap_norm"),
    ("AUROC", "auroc"),
    ("AP", "ap"),
    ("AUPR[1]", "aupr"),
    ("baseline", "baseline"),
    ("maxF1[2]", "f1"),
]
MAIN_FOOTNOTE = (
    "AUPRG / AP_norm are baseline-anchored skill scores (random -> 0 at any\n"
    "prevalence), NOT prevalence-invariant; AUROC is the only invariant column.\n"
    "[1] AUPR is the vendored benchmark's trapezoidal auc(recall, precision)\n"
    "    (the auc(recall, precision) call in cal_metrics), kept for comparability with\n"
    "    published SLMGAE/SLGNN/NSF4SL tables. It linearly interpolates between\n"
    "    PR points, which no classifier achieves, and rewards ties: a CONSTANT\n"
    "    score matrix scores exactly (1+baseline)/2 — i.e. the `baseline`\n"
    "    column of this very table determines it, and on every CV here that\n"
    "    lands near 0.585, above most real models. NOT hard-coded: an earlier\n"
    "    footnote quoted 0.5786, which was the value under the pre-2026-08-10\n"
    "    fold set and silently went stale when the label rule changed.\n"
    "    Report AP (average precision), which gives a constant predictor exactly\n"
    "    `baseline`.\n"
    "[2] maxF1 is the MAXIMUM F1 over the PR curve, i.e. an oracle threshold\n"
    "    chosen against the test labels (the max-F1 expression in cal_metrics).\n"
    "    It is applied identically to every model, but it is not an operating\n"
    "    point any deployed model could pick.")
# Whole-universe ranking metrics, in a SEPARATE table because their normalisers
# differ from the classification block - see RETRIEVAL_FOOTNOTE. R@k first: it
# is the only vendored ranking metric with a correct denominator.
RETRIEVAL_METRICS = [
    ("R@10", "recall@10"),
    ("R@50", "recall@50"),
    ("NDCG@10", "ndcg@10"),
    ("NDCG@10c", "ndcg@10_corrected"),
    ("NDCG@50", "ndcg@50"),
    ("NDCG@50c", "ndcg@50_corrected"),
    ("MAP@10", "map@10"),
    ("MAP@10c", "map@10_corrected"),
    ("P@10", "precision@10"),
    ("P@10t", "precision@10_true"),
    ("degen-rows", "degenerate_row_frac"),
]
RETRIEVAL_FOOTNOTE = (
    "All models rank against the SAME shared gene universe\n"
    "(sl_comparison/folds/genes.txt) on the SAME folds, siamese_sl included\n"
    "(every siamese config.json carries folds_dir=../sl_comparison/folds).\n"
    "The columns differ only in how they are NORMALISED:\n"
    "  R@k       hits@k / #relevant. Correct as published - prefer these.\n"
    "  NDCG@k    vendored: IDCG taken over the model's OWN top-100, so IDCG =\n"
    "            min(k, hits-in-top-100) instead of min(k, #relevant)\n"
    "            (cal_metrics' ndcg_score call, handed y_bool over the model's\n"
    "            own top-100). Inflated.\n"
    "  NDCG@kc   same ranking, IDCG from the true #relevant. Corrected.\n"
    "  MAP@k     vendored: AP divided by hits-in-top-k rather than\n"
    "            min(k, #relevant) (calculate_AP_at_k). Inflated more\n"
    "            than NDCG.\n"
    "  MAP@kc    same ranking, divided by min(k, #relevant). Corrected.\n"
    "  P@10      vendored: hits/min(k, #relevant) - a second copy of R@k\n"
    "            whenever #relevant <= k (cal_metrics' precision_k block),\n"
    "            not precision.\n"
    "  P@10t     true hits/k.\n"
    "The corrections shrink every score and shrink the WEAKEST retriever most;\n"
    "on the 2026-08-06 run they reordered no cell. See retrieval_metrics.py.\n"
    "degen-rows = fraction of scored test genes whose score row is CONSTANT.\n"
    "  Those genes' @k values are an arbitrary tie-break, not model output, and\n"
    "  do not reproduce across numpy versions. Any row with a non-trivial\n"
    "  degen-rows figure has a retrieval column that should not be published.")


def load_results(results_dir):
    """{cv: {model: summary_dict}} from every *.json in results_dir."""
    out = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(path) as f:
                summary = json.load(f).get("summary", {})
        except (OSError, json.JSONDecodeError):
            continue
        model = summary.get("model")
        cv = summary.get("cv_type")
        if model and cv:
            out.setdefault(cv, {})[model] = summary
    return out


def cell(summary, key):
    mean = summary.get(f"{key}_mean")
    if mean is None:
        return "-"
    std = summary.get(f"{key}_std")
    return f"{mean:.4f}±{std:.4f}" if std is not None else f"{mean:.4f}"


def model_label(name, summary):
    """'SLMGAE(1,234,567)' — name with its trainable-parameter count.

    The count is written at model-construction time by
    model_params.write_model_params and carried through evaluate_model.py. A
    model whose trainer predates that (or whose count failed to record) renders
    as the bare name rather than a misleading zero.
    """
    n = summary.get("trainable_params")
    return f"{name}({n:,})" if isinstance(n, int) else name


def print_table(cv,
                per_model,
                metrics=METRICS,
                title="model comparison",
                footnote=None):
    label = CV_LABELS.get(cv, cv)
    models = [m for m in MODEL_ORDER if m in per_model] + \
        [m for m in per_model if m not in MODEL_ORDER]
    labels = {m: model_label(m, per_model[m]) for m in models}
    cw = 15
    mw = max([len("Model")] + [len(v) for v in labels.values()])
    header = f"{'Model':<{mw}s} | " + " | ".join(f"{lab:<{cw}s}"
                                                 for lab, _ in metrics)
    width = len(header)
    print("\n" + "=" * width)
    print(f"{label} - {title}".center(width))
    print("=" * width)
    print(header)
    print("-" * width)
    for m in models:
        cells = [cell(per_model[m], key) for _, key in metrics]
        print(f"{labels[m]:<{mw}s} | " + " | ".join(f"{c:<{cw}s}"
                                                    for c in cells))
    print("-" * width)
    print(footnote or MAIN_FOOTNOTE)
    print("=" * width)


PARAM_FOOTNOTE = """
PARAMETER COUNTS (the number in each model's label)
  Counted identically for all rows by model_params.count_parameters: every
  torch parameter with requires_grad=True at model-construction time. Two
  caveats a reader will otherwise draw the wrong conclusion from:
   - SLMGAE's count is dominated by a (3, n_genes, n_genes) attention block
     (234,649,008 of 253,547,952 = 92.6%). Gradient reaches it only at
     training-edge positions, so ~96% of the total never updates; its
     effective trained size is ~10.7M, not 253.5M.
   - siamese_sl's count EXCLUDES its frozen pretrained gene-embedding table
     (19,397 x 128 = 2,482,816 values), which is an input, not a parameter.
  DegreePrior(0) is exact: it has no parameters at all."""


def selection_regime_report(results):
    """Which rows picked their checkpoint on validation, and which on TEST.

    A row selected on the split it reports is a maximum over N test estimates,
    not a held-out measurement, and it is invisible in every metric. SLMGAE
    selected on TEST for the whole 2026-08-06 run while the other three used
    validation, and nothing in the table said so. Each trainer now records the
    regime via model_params.write_model_params(selection_regime=...), which
    travels with the predictions into the summary.
    """
    rows = {}
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        for model, summary in results[cv].items():
            rows.setdefault(summary.get("selection_regime", "UNRECORDED"),
                            set()).add(f"{model}/{cv}")
    if not rows:
        return
    print("\nMODEL SELECTION REGIME")
    for regime, cells in sorted(rows.items()):
        print(f"  {regime}: {', '.join(sorted(cells))}")
    bad = [r for r in rows if str(r).upper().startswith("TEST")]
    if bad:
        print("  ERROR: the rows above marked TEST chose their checkpoint on "
              "the SAME edges they report. Those numbers are a maximum over "
              "the evaluation budget, not held-out — re-run them before "
              "publishing anything that compares them to a val-selected row.")
    elif "UNRECORDED" in rows:
        print("  WARNING: UNRECORDED rows predate selection_regime tracking. "
              "Re-grade after a re-run to confirm how they were selected; do "
              "not assume validation.")


def label_semantics_report(results):
    """Surface any row whose POSITIVE label means something different.

    The tables share test edges and an evaluator, which makes them look fully
    like-for-like. They are not: siamese's positives are OR-collapsed over 8
    cell-line heads. That belongs next to the numbers, not in a JSON field.
    """
    seen = {}
    for cv in results:
        for model, summary in results[cv].items():
            ls = summary.get("label_semantics")
            if ls:
                seen.setdefault(ls, set()).add(model)
    if not seen:
        return
    print("\nLABEL SEMANTICS (rows do NOT share a definition of 'positive')")
    for ls, models in seen.items():
        print(f"  {', '.join(sorted(models))}: {ls}")


def selection_report(results):
    """Print the selection-budget disclosure for any model that swept configs.

    siamese_sl is chosen as 1-of-N embeddings per CV on validation while the
    other rows get one hard-coded configuration each. Selecting and reporting on
    the same folds is optimistic; collect_siamese.nested_loo_audit measures by
    how much. Printing it here means the asymmetry travels with the table
    instead of living in a JSON nobody opens.
    """
    rows = []
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        for model, summary in results[cv].items():
            a = summary.get("selection_audit")
            if a and a.get("nested_loo_auprg_mean") is not None:
                rows.append((cv, model, a, summary.get("auprg_mean")))
    if not rows:
        return
    print("\nSELECTION BUDGET (asymmetric across rows — disclose in methods)")
    for cv, model, a, table_auprg in rows:
        # The optimism is measured ENTIRELY inside the sweep's own scoring
        # (collect_siamese reads each candidate's per-fold auprg_any), so
        # `nested - self_reported` is an internally consistent difference. The
        # value in the table above comes from the SHARED evaluator and is a
        # different quantity — printing the sweep's own figure as "reported"
        # would show a number that appears nowhere in the table. Apply the
        # delta to the table's value instead.
        self_rep = a.get("reported_auprg_mean")
        nested = a["nested_loo_auprg_mean"]
        n = a["n_candidates"]
        if self_rep is None:
            print(f"  {model}/{cv}: nested leave-one-fold-out AUPRG "
                  f"{nested:.4f} over {n} configs (no paired value to "
                  f"difference against).")
            continue
        delta = nested - self_rep
        line = (f"  {model}/{cv}: best of {n} configs picked on validation. "
                f"Selection optimism measured on the sweep's own scoring: "
                f"{delta:+.4f} ({self_rep:.4f} -> {nested:.4f}).")
        if table_auprg is not None:
            line += (f" Applied to this table's AUPRG {table_auprg:.4f} that "
                     f"is {table_auprg + delta:.4f} — publish that.")
        print(line)
        # How CONCENTRATED is that correction? If the held-out selection picks
        # the same config on every fold but one, the whole delta rests on a
        # single fold's swap and must not be read as a stable offset.
        choices = a.get("per_fold_choice") or []
        picks = [c.get("config") for c in choices]
        # Compare against the config the TABLE reports, recorded by
        # nested_loo_audit. Using the mode of `picks` instead was both a
        # different quantity and nondeterministic on a tie (set iteration order
        # is hash-salted per process).
        base = a.get("reported_config")
        if picks and base:
            n_diff = sum(1 for p in picks if p != base)
            if n_diff == 0:
                # Zero swaps means nested selection reproduced the reported
                # config on every fold, so the delta is 0 BY CONSTRUCTION. It
                # is an identity, not evidence that selection cost nothing —
                # reporting "+0.0000 optimism" as a measurement overstates it.
                print(f"      held-out selection reproduced '{base}' on all "
                      f"{len(picks)} folds, so this delta is exactly 0 by "
                      f"construction — an identity, NOT a measurement that "
                      f"selection was free.")
            else:
                print(
                    f"      driven by {n_diff} of {len(picks)} folds swapping "
                    f"away from '{base}' — a {n_diff}-fold effect, not a "
                    f"uniform offset; report it with that caveat.")
    print("  Every other row is a single fixed configuration (no sweep).")


def provenance_report(results, emit=None):
    """Warn if the rows in one table did not all come from the same fold set.

    Each results.json records sha256 prefixes of genes.txt / all_pos.npy /
    meta.json (evaluate_model.fold_provenance). Rows built on different fold
    sets are not comparable, and nothing else in the pipeline would notice.

    `emit` collects the same lines for the notes sidecar. Printing alone put
    this warning only in the SLURM log, while comparison.csv — the artifact
    that outlives the log — said nothing at all.
    """
    lines = []

    def print(*a):  # noqa: A001
        lines.append(" ".join(str(x) for x in a))
        builtins.print(*a)

    seen = {}
    missing = []
    for cv, per_model in results.items():
        for model, summary in per_model.items():
            prov = summary.get("provenance")
            # A partial fingerprint is "unknown", not "different". Rows graded
            # before meta.json entered the key carry (g, a, None); bucketing
            # that alongside a complete key raised a false "NOT comparable"
            # alarm AND printed "meta.json None" as though it were a match.
            if not prov or prov.get("meta.json_sha256") is None:
                missing.append(f"{model}/{cv}")
                continue
            # meta.json MUST be in the key. A different --seed reshuffles every
            # fold assignment while leaving genes.txt and all_pos.npy
            # byte-identical (both are seed-independent), so a two-key check
            # cannot see the single most likely way two runs diverge. meta.json
            # records the seed, fold count, val_frac and conflict policy.
            key = (prov.get("genes.txt_sha256"),
                   prov.get("all_pos.npy_sha256"),
                   prov.get("meta.json_sha256"))
            seen.setdefault(key, []).append(f"{model}/{cv}")
    if missing:
        print(f"\nPROVENANCE: no complete fold fingerprint for "
              f"{', '.join(sorted(missing))} — comparability with the rows "
              f"below is UNVERIFIED, not confirmed. Re-grade with the current "
              f"evaluate_model.py to make them traceable.")
    if len(seen) > 1:
        print("\nPROVENANCE WARNING: rows come from DIFFERENT fold sets and "
              "are NOT comparable:")
        for key, rows in seen.items():
            print(f"  genes={key[0]} all_pos={key[1]} meta={key[2]}: "
                  f"{', '.join(sorted(rows))}")
    elif len(seen) == 1 and not missing:
        key = next(iter(seen))
        print(f"\nPROVENANCE: all rows scored on one fold set "
              f"(genes.txt {key[0]}, all_pos.npy {key[1]}, "
              f"meta.json {key[2]}).")
    # Same fold set is not the same CODE. Report the git state too, or a table
    # mixing an old and a new evaluator prints an unqualified all-clear.
    shas, dirty = set(), set()
    for cv in results:
        for summary in results[cv].values():
            p = summary.get("provenance") or {}
            if p.get("git_sha"):
                shas.add(p["git_sha"])
                dirty.add(bool(p.get("git_dirty")))
    if len(shas) > 1:
        print(f"  CODE WARNING: rows were graded by {len(shas)} different "
              f"commits ({', '.join(sorted(shas))}) — re-grade them all with "
              f"one version before publishing.")
    elif shas and any(dirty):
        print(f"  CODE: graded at {next(iter(shas))} with UNCOMMITTED changes "
              f"— that SHA does not identify the code that ran. Commit before "
              f"the run you intend to publish.")
    if emit is not None:
        emit.extend(lines)
    return lines


def write_csv(results, csv_path, provenance_lines=()):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cv", "model", "metric", "mean", "std"])
        for cv in sorted(results,
                         key=lambda c: CV_ORDER.index(c)
                         if c in CV_ORDER else 99):
            for model, summary in results[cv].items():
                # EVERY aggregated metric, not just the ones the printed tables
                # happen to show. Emitting only METRICS + RETRIEVAL_METRICS
                # silently dropped 12 keys the moment the table columns changed
                # — including aupr_norm, which the previously-published
                # comparison.csv contained. The CSV is the archival artifact;
                # it must not lose a metric because a column was reordered.
                keys = [
                    k[:-5] for k in summary if k.endswith("_mean")
                    and len(k) > 5 and isinstance(summary[k], (
                        int, float)) and not isinstance(summary[k], bool)
                ]
                for key in sorted(keys):
                    mean = summary.get(f"{key}_mean")
                    if mean is not None:
                        w.writerow([
                            cv, model, key, mean,
                            summary.get(f"{key}_std", "")
                        ])
                # Model size as an ordinary row so the CSV stays tidy/long.
                # Not a per-fold quantity, so std is blank.
                for key in ("trainable_params", "total_params"):
                    if isinstance(summary.get(key), int):
                        w.writerow([cv, model, key, summary[key], ""])
                # PROVENANCE AS DATA, not just as a stdout warning.
                #
                # provenance_report prints "rows come from DIFFERENT fold sets"
                # to the SLURM log, which nobody reads next to the CSV. The CSV
                # is the archival artifact, and sl_comparison/results is never
                # cleared — a model whose training failed keeps its previous
                # results.json, load_results picks it up, and it is archived
                # here indistinguishably from rows graded on the current folds.
                # Carrying the fingerprint per row makes that visible in the
                # artifact itself: one grep, one obvious odd row out.
                prov = summary.get("provenance") or {}
                for name in ("genes.txt", "all_pos.npy", "meta.json"):
                    v = prov.get(f"{name}_sha256")
                    if v:
                        w.writerow([cv, model, f"folds_{name}_sha256", v, ""])
                for key in ("git_sha", "git_dirty"):
                    if prov.get(key) is not None:
                        w.writerow([cv, model, key, prov[key], ""])
                # Produce-time stamp: which fold set the MATRICES were trained
                # against, as opposed to the one they were graded against.
                # evaluate_model refuses on a mismatch, so these agree by
                # construction — recording it is what makes that checkable.
                st = summary.get("fold_stamp") or {}
                for name in ("genes.txt", "all_pos.npy", "meta.json"):
                    if st.get(name):
                        w.writerow([
                            cv, model, f"trained_on_{name}_sha256", st[name],
                            ""
                        ])
    # The CSV is complete but UNANNOTATED, and completeness is what makes it
    # dangerous: it carries `aupr` and its derivative `aupr_norm`, on which a
    # constant predictor scores (1+pi)/2 — DegreePrior tops the cv3 aupr_norm
    # column at exactly 0.5000, above every trained model. Anyone who opens the CSV
    # without the printed tables sees that and draws the wrong conclusion.
    notes = Path(csv_path).with_name(Path(csv_path).stem + "_notes.txt")
    notes.write_text(
        "Notes for " + Path(csv_path).name + "\n" + "=" * 60 + "\n\n" +
        MAIN_FOOTNOTE + "\n\n" + RETRIEVAL_FOOTNOTE + "\n\n" +
        "aupr_norm is derived from `aupr` and inherits its defect: it is\n"
        "(aupr - baseline) / (1 - baseline), so a constant predictor scores\n"
        "0.5 rather than 0. Use ap_norm. Both are in this CSV for archival\n"
        "completeness; only ap / ap_norm / auprg / auroc are safe to report.\n\n"
        + PARAM_FOOTNOTE + "\n" +
        ("\n\nFOLD-SET PROVENANCE (as printed by compare.py; the\n"
         "per-row sha256 fingerprints are also rows in the CSV itself)\n" +
         "-" * 60 + "\n" + "\n".join(provenance_lines) +
         "\n" if provenance_lines else ""))
    print(f"\nTidy CSV -> {csv_path}")
    print(f"Caveats   -> {notes}")


def main():
    ap = argparse.ArgumentParser(description="Four-model comparison table")
    ap.add_argument("--results_dir", default="sl_comparison/results")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results in {args.results_dir}. Run the models + "
              f"evaluate_model.py / collect_siamese.py first.")
        return
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        print_table(cv, results[cv])
        if any(f"{k}_mean" in s for s in results[cv].values()
               for _, k in RETRIEVAL_METRICS):
            print_table(cv,
                        results[cv],
                        metrics=RETRIEVAL_METRICS,
                        title="retrieval (see normalisation footnote)",
                        footnote=RETRIEVAL_FOOTNOTE)
    print(PARAM_FOOTNOTE)
    selection_regime_report(results)
    label_semantics_report(results)
    selection_report(results)
    prov_lines = provenance_report(results)
    if args.csv:
        write_csv(results, args.csv, provenance_lines=prov_lines)


if __name__ == "__main__":
    main()
