#!/usr/bin/env python3
"""
Unified four-model comparison table.

Reads every <Model>_<cv>.json of --results_dir, sl_comparison/results by
default (all written in one schema by evaluate_model.py; siamese_sl's rows are
collected by collect_siamese.py and then re-graded through it by
slurm/collect_and_regrade_siamese.sh), and prints one table per CV type plus a
tidy CSV and its comparison_notes.txt.

One table per PCA arm (siamese_sl/slurm/config.conf, PCA arms): run_compare.sh
passes the arm's comparison root (sl_comparison/results, or
sl_comparison/results/pca_<arm>) as --results_dir and sl_comparison/results,
where the shared models' rows live for every arm, as --shared_results_dir. The
SLeuth rows (siamese_sl*) then come only from the first, every other row only
from the second (load_split_results), and --pca_arm names the arm under each
table's title, on the notes' first line and in the CSV's pca_arm column.

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
# BilinearProbe (2026-11-12, reviewer 22DY): tau * (W x_i)ᵀ(W x_j) + b, one
# bias-free Linear on PCA arm none's pd:pooled combo features
# (siamese_sl/slurm/run_probe.sh). A trained baseline, not a SLeuth row
# (is_siamese_row is False): like the shared models' rows it is read from
# sl_comparison/results/ only, so every arm's table shows the one probe, which
# was trained on arm none's features.
# ELISL (2026-11-12, reviewer request): Tepeli et al., Bioinformatics 2024,
# context-free (Sequence_1024 + PPI_ec), the authors' LightGBM forests
# (ELISL/slurm/run_elisl.sh). A published peer, shared row like the others.
MODEL_ORDER = ["siamese_sl", "SLMGAE", "SLGNN", "NSF4SL", "ELISL",
               "BilinearProbe", "DegreePrior"]
# siamese_sl/slurm/config.conf PCA_ARMS (copied, not parsed: compare.py stays
# import-light and needs no shell).
PCA_ARMS = ("none", "joint", "per_modality", "both")
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
    "    column of this very table determines it, and it can exceed real\n"
    "    models' AUPR. compare.py prints the value under each CV's table; no\n"
    "    number is quoted here: an earlier footnote quoted 0.5786, the value\n"
    "    under the pre-2026-08-10 fold set, which went stale when the label\n"
    "    rule changed.\n"
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
    "The corrections usually lower scores and lower the WEAKEST retriever\n"
    "most, but not always: NDCG@kc can EXCEED NDCG@k on a row with many tied\n"
    "scores (sklearn's ndcg_score averages tied gains; the corrected DCG is\n"
    "positional). On the 2026-08-06 run they reordered no cell. See\n"
    "retrieval_metrics.py.\n"
    "degen-rows = fraction of scored test genes whose score row is CONSTANT.\n"
    "  Those genes' @k values are an arbitrary tie-break, not model output, and\n"
    "  do not reproduce across numpy versions. Any row with a non-trivial\n"
    "  degen-rows figure has a retrieval column that should not be published.")


def load_results(results_dir, keep=None, dropped=None):
    """{cv: {model: summary_dict}} from every *.json in results_dir.

    keep: a predicate on summary["model"]; a row it rejects is left out and,
    when `dropped` is a list, appended there as (model, cv, file name). On the
    model, not the file name: the table is keyed on the summary's model.
    """
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
            if keep is not None and not keep(model):
                if dropped is not None:
                    dropped.append((model, cv, path.name))
                continue
            out.setdefault(cv, {})[model] = summary
    return out


def load_split_results(results_dir, shared_dir):
    """One PCA arm's table: its SLeuth rows from results_dir, the rest from
    shared_dir. Returns (results, dropped_arm, dropped_shared).

    Every arm's SLeuth rows (is_siamese_row: siamese_sl, siamese_sl_<tag>,
    the *_nocell rows) carry the same names, and sl_comparison/results holds
    arm none's beside the shared models' rows (SLGNN / NSF4SL / SLMGAE /
    DegreePrior, trained once for every arm). A union of the two directories
    would let file order pick the arm a cell shows, so each side is filtered
    on summary["model"]: dropped_arm lists the non-SLeuth rows in results_dir
    (never used: a hand copy, or a row written there by mistake) and
    dropped_shared the SLeuth rows in shared_dir (another table's, e.g. arm
    none's). With one directory for both (arm none) this is
    load_results(results_dir), nothing dropped.
    """
    if Path(results_dir).resolve() == Path(shared_dir).resolve():
        return load_results(results_dir), [], []
    dropped_arm, dropped_shared = [], []
    results = load_results(shared_dir, keep=lambda m: not is_siamese_row(m),
                           dropped=dropped_shared)
    for cv, per_model in load_results(results_dir, keep=is_siamese_row,
                                      dropped=dropped_arm).items():
        results.setdefault(cv, {}).update(per_model)
    return results, dropped_arm, dropped_shared


def _cv_rank(cv):
    return CV_ORDER.index(cv) if cv in CV_ORDER else 99


def _cells(dropped):
    """'SLGNN (cv3), siamese_sl (cv1 cv2)' from (model, cv, file) entries."""
    by_model = {}
    for model, cv, _ in dropped:
        by_model.setdefault(model, set()).add(cv)
    return ", ".join(f"{m} ({' '.join(sorted(cvs, key=_cv_rank))})"
                     for m, cvs in sorted(by_model.items()))


def split_report(results_dir, shared_dir, results, dropped_arm,
                 dropped_shared):
    """Print where the table's rows came from and what each side left out.

    Returns the source lines for comparison_notes.txt (the left-out rows are
    in no table, so they stay in the log)."""
    same = Path(results_dir).resolve() == Path(shared_dir).resolve()
    n_sleuth = sum(1 for per in results.values() for m in per
                   if is_siamese_row(m))
    n_shared = sum(1 for per in results.values() for m in per
                   if not is_siamese_row(m))
    if same:
        lines = [f"rows: {n_sleuth} SLeuth and {n_shared} shared-model "
                 f"row(s), all from {results_dir}"]
    else:
        lines = [f"rows: {n_sleuth} SLeuth row(s) (siamese_sl*) from "
                 f"{results_dir}; {n_shared} shared-model row(s) (SLGNN / "
                 f"NSF4SL / SLMGAE / DegreePrior, the same in every PCA "
                 f"arm's table) from {shared_dir}"]
    for ln in lines:
        print(ln)
    if dropped_shared:
        print(f"  not used: {len(dropped_shared)} SLeuth row(s) in "
              f"{shared_dir}, that directory's own table's (this table's "
              f"SLeuth rows are {results_dir}'s): {_cells(dropped_shared)}")
    if dropped_arm:
        names = ", ".join(f"{m}/{cv} ({f})"
                          for m, cv, f in sorted(dropped_arm))
        print(f"  WARNING: {len(dropped_arm)} non-SLeuth row(s) in "
              f"{results_dir} NOT used: the shared models' rows are read from "
              f"{shared_dir} only, where every arm's table reads them. Move "
              f"them out of {results_dir}: {names}")
    return lines


def pca_arm_line(arm, label=None):
    """'PCA arm: joint (PCA on the concatenation of all modalities (...))',
    as config.conf's job banner names an arm."""
    return f"PCA arm: {arm}" + (f" ({label})" if label else "")


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

    A siamese row's size can differ by fold (a PCA arm's variance target
    resolves the kept width k per fold; in arm none, no PCA, every fold has
    the same count) and its trainable_params is only the LAST fold's
    (model_params.json is rewritten at every fold start), so a row carrying
    collect_and_regrade_siamese.sh's trainable_params_fold_mean shows that.

    A row whose mean is over fewer folds than the fold set has (evaluate_model
    --allow_partial, or a control that skipped a degenerate fold) is marked
    PARTIAL: it otherwise reads exactly like a complete one in every column,
    and which folds are missing moves the number (per-fold prevalence varies).
    """
    mark = (f"[PARTIAL {summary.get('n_folds_scored')}/"
            f"{summary.get('n_folds_expected')}]"
            if summary.get("partial") else "")
    fm = summary.get("trainable_params_fold_mean")
    if isinstance(fm, (int, float)) and not isinstance(fm, bool):
        nf = summary.get("trainable_params_fold_n")
        return (f"{name}({fm:,.0f} mean"
                + (f" of {nf} folds" if isinstance(nf, int) else "") + ")"
                + mark)
    n = summary.get("trainable_params")
    return (f"{name}({n:,})" if isinstance(n, int) else name) + mark


def is_allmod_row(model):
    """The all-modalities combo's rows (siamese_sl/slurm/run_allmod_combo.sh,
    ALLMOD_COMBO=1): siamese_sl_allmod and siamese_sl_allmod_<tag>, SLeuth
    rows (is_siamese_row) trained on every catalog embedding with no
    modality selection."""
    return model == "siamese_sl_allmod" or \
        model.startswith("siamese_sl_allmod_")


def ordered_models(per_model):
    """MODEL_ORDER, with siamese_sl's variant rows (siamese_sl_<tag>, one per
    SIAMESE_VARIANTS entry other than the baseline) placed right after
    siamese_sl so the kernel x cell_lines grid reads as one block, then the
    all-modalities rows (siamese_sl_allmod[_<tag>]), the selection ablation
    next to it."""
    variants = sorted(m for m in per_model
                      if m.startswith("siamese_sl_") and m not in MODEL_ORDER
                      and not is_allmod_row(m))
    allmod = sorted(m for m in per_model if is_allmod_row(m))
    order = MODEL_ORDER[:1] + variants + allmod + MODEL_ORDER[1:]
    return [m for m in order if m in per_model] + \
        [m for m in per_model if m not in order]


def print_table(cv,
                per_model,
                metrics=METRICS,
                title="model comparison",
                footnote=None,
                subtitle=None):
    label = CV_LABELS.get(cv, cv)
    models = ordered_models(per_model)
    labels = {m: model_label(m, per_model[m]) for m in models}
    cw = 15
    mw = max([len("Model")] + [len(v) for v in labels.values()])
    header = f"{'Model':<{mw}s} | " + " | ".join(f"{lab:<{cw}s}"
                                                 for lab, _ in metrics)
    width = len(header)
    print("\n" + "=" * width)
    print(f"{label} - {title}".center(width))
    if subtitle:
        # The PCA arm: every arm's table has the same title and row names.
        print(subtitle.center(width))
    print("=" * width)
    print(header)
    print("-" * width)
    for m in models:
        cells = [cell(per_model[m], key) for _, key in metrics]
        print(f"{labels[m]:<{mw}s} | " + " | ".join(f"{c:<{cw}s}"
                                                    for c in cells))
    print("-" * width)
    print(footnote or MAIN_FOOTNOTE)
    if footnote is None:
        # The constant-score AUPR [1] describes, from this table's own
        # baseline column (every row shares the per-fold test sets).
        pis = sorted({round(s["baseline_mean"], 6)
                      for s in per_model.values()
                      if isinstance(s.get("baseline_mean"), (int, float))})
        if len(pis) == 1:
            print(f"    [1] here: a constant score matrix has AUPR "
                  f"(1+{pis[0]:.4f})/2 = {(1 + pis[0]) / 2:.4f}.")
        elif pis:
            vals = ", ".join(f"{p:.4f}" for p in pis)
            print(f"    [1] rows disagree on `baseline` ({vals}): they were "
                  f"not all graded on the same test pairs.")
    print("=" * width)


PARAM_FOOTNOTE = """
PARAMETER COUNTS (the number in each model's label)
  Counted identically for all rows by model_params.count_parameters: every
  torch parameter with requires_grad=True at model-construction time.
  Caveats a reader will otherwise draw the wrong conclusion from:
   - SLMGAE's count is dominated by a (3, n_genes, n_genes) attention block
     (234,649,008 of 253,547,952 = 92.6%). Gradient reaches it only at
     training-edge positions, so ~96% of the total never updates; its
     effective trained size is ~10.7M, not 253.5M.
   - siamese_sl's count EXCLUDES its frozen input feature matrix (n_genes x k,
     the pretrained embeddings after the PCA arm's PCA), which is an input,
     not a parameter. k is the encoder's input width: in PCA arm none (no
     PCA) the concatenated width of the selected modalities, which moves with
     the selection (4,424-4,864 on the sets selected so far, i.e. ~86-94M
     values at n_genes = 19,397); in arms joint, per_modality and both the
     width the PCA keeps, which a variance target resolves per fold (train.py
     logs it; an exact PCA_JOINT_DIM fixes the joint PCA's). This run's k is
     fold_metrics[].input_dim in each combo's results.json.
   - A siamese_sl label reads "<n> mean of <K> folds" (k, hence the entry
     layer, differs by fold under a variance-target PCA; in arm none every
     fold has the same count): the per-fold mean, also the CSV row
     trainable_params_fold (mean, population sd). Its CSV trainable_params /
     total_params rows are the LAST fold's count only (model_params.json is
     rewritten at every fold start), as is the label of a siamese row with no
     per-fold counts. Quote the mean.
   - BilinearProbe's count is W (latent x k, k the fold's input width, as
     for siamese_sl) plus tau and b; its label is the per-fold mean, like a
     siamese_sl row's (run_probe.sh writes trainable_params_fold_*).
  DegreePrior(0) is exact: it has no parameters at all."""

# Label-derived signals in inputs the authors ship. Kept on purpose (every
# external model runs as published) and stated next to the table rather than
# only in sl_comparison/README.md, which does not travel with the CSV.
UPSTREAM_INPUT_NOTES = {
    "SLGNN": (
        "SynLethKG kg2id.txt (SLGNN repo @ c56e23fbd20e) ships without the\n"
        "    SL / SR / non-SL relations AND without every other gene-gene edge\n"
        "    (INTERACTS, REGULATES, COVARIES) between genes SynLethKG labels SL:\n"
        "    all 1,251 non-label triples missing from it lie on SL pairs. On the\n"
        "    shared folds a direct KG edge joins 0.37% of positive pairs (83 of\n"
        "    22,166) and 2.38% of negatives (2,542 of 106,704), so in SLGNN's\n"
        "    input a direct edge correlates with non-SL, for val and test pairs\n"
        "    too."),
    "SLMGAE": (
        "its GO-BP / GO-CC / PPI support views cover only the 6,375 genes of\n"
        "    List_Proteins_in_SL.txt, the gene set of the SynLethDB v1 SL pairs\n"
        "    (4,587 of the 8,844 shared genes; the rest get all-zero view\n"
        "    rows). 73.1% of genes in any positive pair have a view row against\n"
        "    20.7% of negative-only genes, so coverage alone is predictive: the\n"
        "    number of a pair's genes with a view row scores cv3 test AUROC\n"
        "    0.60-0.74 per fold (cv2 0.67-0.72). A gene with no training\n"
        "    positive and no view row gets an exactly-zero embedding: SLMGAE's\n"
        "    constant score rows (degen-rows)."),
}
UPSTREAM_INPUT_HEAD = (
    "UPSTREAM INPUTS (as the authors ship them, kept on purpose and not\n"
    "corrected; each carries a label-derived signal - see sl_comparison/README.md)")
# Tail of train_slmgae_shared.LP_STATIC_FALLBACK_SPACE, the `lp_mapping`
# id_space an SLMGAE run records when the HGNC table could not be loaded and
# List_Proteins_in_SL.txt was mapped through the static table instead: 6,156 of
# 6,375 rows map, 4,470 of them into the shared universe rather than 4,587. The
# trainer calls that "Same policy as a missing view ... a REDUCED SLMGAE"
# (--allow_static_lp_mapping), so compare.py must name it exactly like a
# ZERO-FILLED view. Matched as a substring and copied rather than imported:
# compare.py stays import-light, and SLMGAE-in-pytorch is not importable by
# name (the hyphen).
LP_STATIC_FALLBACK_MARK = "static gene_id_mapping.tsv fallback"
# The other copied marker: the prefix train_slmgae_shared.read_support_views
# writes as the `support_views` provenance of an all-zero modality,
# f"ZERO-FILLED (missing {path})" (--allow_missing_views). Same why as above;
# test_comp_contracts.test_reduction_markers_match_the_slmgae_trainer asserts
# both copies still occur in the trainer's source, because a reworded marker
# would silently stop the REDUCED warning below from firing.
ZERO_FILLED_VIEW_MARK = "ZERO-FILLED"


def upstream_input_report(results):
    """Print, and return, UPSTREAM_INPUT_NOTES for the models in the table.

    Keyed on what each row ran, not only on its name (both fields come from
    the trainer's model_params.json via evaluate_model.build_summary):
      - SLGNN's note describes SynLethKG's kg2id.txt, the --arch original
        input. A --arch port row (`arch`) never read the KG and gets a
        one-line input statement instead. A row with NO `arch` record predates
        the SynLethKG path entirely: it is named as undisclosed rather than
        assumed original, exactly as model_config_report treats an NSF4SL row
        with no `features` record.
      - An SLMGAE row is named as a REDUCED model when `support_views` records
        a ZERO-FILLED view (train_slmgae_shared.py --allow_missing_views) OR
        `lp_mapping.id_space` is the static gene_id_mapping.tsv fallback
        (--allow_static_lp_mapping: 4,470 genes get a view row, not 4,587).
        Neither is the authors' SLMGAE, and the coverage note above describes
        neither of them.
    """
    lines = []
    for m, text in UPSTREAM_INPUT_NOTES.items():
        cells = sorted((cv, per_model[m]) for cv, per_model in results.items()
                       if m in per_model)
        if not cells:
            continue
        port = [f"{m}/{cv}" for cv, s in cells
                if m == "SLGNN" and s.get("arch") == "port"]
        # An SLGNN row with no `arch` at all was graded before the trainer
        # recorded one, i.e. before the --arch original SynLethKG path existed:
        # back then its only input was the per-gene --embedding_path features
        # (default kg_complex), so the kg2id note does not describe such a row
        # either. Named, not assumed original.
        unrec = [f"{m}/{cv}" for cv, s in cells
                 if m == "SLGNN" and "arch" not in s]
        if len(port) + len(unrec) < len(cells):
            lines.append(f"  {m}: {text}")
        if port:
            lines.append(f"  SLGNN --arch port ({', '.join(port)}): per-gene "
                         f"--embedding_path features, no SynLethKG input; the "
                         f"kg2id caveat does not apply to these rows.")
        if unrec:
            lines.append(f"  SLGNN rows with no `arch` record "
                         f"({', '.join(unrec)}): graded before the trainer "
                         f"recorded one, when the port's per-gene "
                         f"--embedding_path features were its only input. "
                         f"WHICH input produced them is NOT disclosed and the "
                         f"kg2id note does not describe them; re-grade a run "
                         f"of the current trainer before publishing them.")
        reduced = []
        for cv, s in cells:
            if m != "SLMGAE":
                continue
            # NOT an early `continue` on a missing support_views: a row can be
            # reduced through lp_mapping alone, and that one used to be read
            # by nothing at all.
            why = []
            views = s.get("support_views")
            if isinstance(views, dict):
                zero = sorted(k for k, v in views.items()
                              if str(v).startswith(ZERO_FILLED_VIEW_MARK))
                if zero:
                    why.append(f"ZERO-FILLED {', '.join(zero)} "
                               f"(--allow_missing_views)")
            lp = s.get("lp_mapping")
            if isinstance(lp, dict) and \
                    LP_STATIC_FALLBACK_MARK in str(lp.get("id_space", "")):
                why.append(f"static List_Proteins mapping "
                           f"(--allow_static_lp_mapping: "
                           f"{lp.get('in_universe')} of {lp.get('rows')} rows "
                           f"in the universe, not the 4,587 above)")
            if why:
                reduced.append(f"{m}/{cv} [{'; '.join(why)}]")
        if reduced:
            lines.append(f"  WARNING: REDUCED {m} rows, trained on LESS than "
                         f"the authors' shipped input: {'; '.join(reduced)}. "
                         f"These are not the authors' model and the coverage "
                         f"note above does not describe them; do not publish "
                         f"them as {m}.")
    if lines:
        lines.insert(0, UPSTREAM_INPUT_HEAD)
        print("\n" + "\n".join(lines))
    return lines


def _collecting_print(lines):
    """A print() that also appends each line to `lines` (for the notes file)."""

    def _print(*a):
        lines.append(" ".join(str(x) for x in a))
        builtins.print(*a)

    return _print


def selection_regime_report(results):
    """Which rows picked their checkpoint on validation, and which on TEST.

    A row selected on the split it reports is a maximum over N test estimates,
    not a held-out measurement, and it is invisible in every metric. SLMGAE
    selected on TEST for the whole 2026-08-06 run while the other three used
    validation, and nothing in the table said so. Each trainer now records the
    regime via model_params.write_model_params(selection_regime=...), which
    travels with the predictions into the summary. Returns the printed lines
    for comparison_notes.txt.
    """
    lines = []
    print = _collecting_print(lines)  # noqa: A001
    rows = {}
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        for model, summary in results[cv].items():
            rows.setdefault(summary.get("selection_regime", "UNRECORDED"),
                            set()).add(f"{model}/{cv}")
    if not rows:
        return lines
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
    return lines


def label_semantics_report(results):
    """Say what POSITIVE means for every row, and what each siamese row SCORES.

    Every row is graded on the SAME positives: the shared folds'
    conflict_policy=or_collapse label (folds/meta.json `label_rule`), which is
    the rule siamese's *_any view already used, so the tables are like-for-like
    on labels. What differs is what a siamese row SCORES — 8 per-head logits
    collapsed by max (--cell_lines per_head), one pooled head, or the
    single-output SiameseSL — against one score for the unlabelled SL relation
    from SLGNN / NSF4SL / SLMGAE / DegreePrior. That belongs next to the
    numbers, not in a JSON field. (This docstring used to say siamese's
    POSITIVES differ; they stopped differing when the folds adopted
    or_collapse, and the text this function prints says so.)
    Returns the printed lines for comparison_notes.txt.
    """
    lines = []
    print = _collecting_print(lines)  # noqa: A001
    seen = {}
    for cv in results:
        for model, summary in results[cv].items():
            ls = summary.get("label_semantics")
            if ls:
                seen.setdefault(ls, set()).add(model)
    if not seen:
        return lines
    print("\nLABEL SEMANTICS (what POSITIVE means, and what each siamese row scores)")
    for ls, models in seen.items():
        print(f"  {', '.join(sorted(models))}: {ls}")
    return lines


# What run_shared_models.sh runs NSF4SL as: the authors' SynLethKG TransE
# entity embeddings (--transe_path, the argparse default --features) and the
# current training loop.
NSF4SL_ORIGINAL_FEATURES = "transe"
# What ELISL/slurm/run_elisl.sh runs ELISL as: its two context-free sources
# (train_elisl.SOURCES; the cancer-specific sources are not used).
ELISL_SOURCES = "Sequence_1024|PPI_ec"


def model_config_report(results):
    """Name every row that did not run its model's BENCHMARK configuration.

    upstream_input_report names an SLGNN --arch port row and a REDUCED SLMGAE
    one. NSF4SL's opt-ins are the same kind of deviation — `--features
    embedding` is the kg_complex substitute, `--legacy_loop` (+ the old
    --select_metric ndcg@50) is the pre-2026-09-21 training loop, both
    documented in NSF4SL/slurm/run_nsf4sl.sh — and nothing read them, so such a
    row printed as a plain NSF4SL(n). train_nsf4sl.feature_provenance records
    them in model_params.json and evaluate_model.build_summary merges them into
    the graded row, so they are here to be read. A row with no `features` key
    predates that record: it is named separately rather than assumed original.
    Returns the printed lines for comparison_notes.txt.
    """
    lines = []
    print = _collecting_print(lines)  # noqa: A001
    off, unrecorded = [], []
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        s = results[cv].get("NSF4SL")
        if s is None:
            continue
        if "features" not in s:
            unrecorded.append(f"NSF4SL/{cv}")
            continue
        why = []
        if s.get("features") != NSF4SL_ORIGINAL_FEATURES:
            why.append(f"--features {s.get('features')} "
                       f"({s.get('feature_file')})")
        if s.get("legacy_loop"):
            why.append("--legacy_loop")
        if why:
            off.append(f"NSF4SL/{cv} ({', '.join(why)})")
    if off or unrecorded:
        print("\nMODEL CONFIGURATION (model_params.json, as graded)")
    if off:
        print(f"  WARNING: NSF4SL rows NOT on the authors' SynLethKG TransE "
              f"input / training loop: {'; '.join(off)}. These are not the "
              f"authors' model; do not publish them as NSF4SL.")
    if unrecorded:
        print(f"  NSF4SL rows with no `features` record (graded before "
              f"feature_provenance existed): {', '.join(unrecorded)} — which "
              f"features and which training loop produced them is NOT "
              f"disclosed. Re-grade a run of the current trainer before "
              f"publishing them.")
    # ELISL: train_elisl.py records its sources in model_params.json.
    el = [f"ELISL/{cv} ({results[cv]['ELISL'].get('elisl_sources')})"
          for cv in sorted(results) if "ELISL" in results[cv]
          and results[cv]["ELISL"].get("elisl_sources") != ELISL_SOURCES]
    if el:
        if not (off or unrecorded):
            print("\nMODEL CONFIGURATION (model_params.json, as graded)")
        print(f"  WARNING: ELISL rows not on the context-free sources "
              f"{ELISL_SOURCES}: {'; '.join(el)}. Do not publish them as "
              f"ELISL (context-free).")
    return lines


def partial_report(results):
    """Name every row whose mean is over FEWER folds than the fold set has.

    evaluate_model.build_summary records n_folds_scored / n_folds_expected /
    partial — set by --allow_partial, or by a control that skipped a degenerate
    fold — but a partial mean reads exactly like a complete one in the metric
    columns, and the per-fold test prevalence varies, so WHICH folds are
    missing moves the number. model_label marks such a row; this puts it in
    comparison_notes.txt too, which travels with the CSV. Returns the printed
    lines.
    """
    lines = []
    print = _collecting_print(lines)  # noqa: A001
    bad = []
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        for model, s in sorted(results[cv].items()):
            if s.get("partial"):
                bad.append((f"{model}/{cv}", s.get("n_folds_scored"),
                            s.get("n_folds_expected")))
    if not bad:
        return lines
    print("\nPARTIAL ROWS (mean over FEWER folds than the fold set has)")
    for name, scored, expected in bad:
        print(f"  {name}: {scored} of {expected} folds")
    print("  NOT comparable to the full-fold rows in the same column: the "
          "per-fold test prevalence varies, so which folds are missing moves "
          "the number. Re-run the missing folds before publishing anything "
          "that compares them.")
    return lines


def is_siamese_row(model):
    """siamese_sl and its siamese_sl_<tag> rows (variants, ablation rows)."""
    return model == "siamese_sl" or model.startswith("siamese_sl_")


def selection_report(results):
    """Print, and return, the selection-budget disclosure.

    Every row but siamese's is one hard-coded configuration. siamese's rows
    carry how theirs was chosen under `selection_audit` (the key the shared
    re-grade carries, evaluate_model.CARRY_KEYS), in one of two forms:
      modality_selection  the rerun path (collect_siamese.py --combo_markers):
          the combo's modalities are the per-category argmax of
          single-embedding VALIDATION scores on selection_cv
          (run_best_per_category_combo.sh), chosen once and reused for every
          CV and variant. With one combo per CV there is nothing to nest
          over, so the record states what was chosen, over how many runs, and
          why the choice is not leakage-free.
      nested_loo_*  the sweep-wide discovery (SIAMESE_BASELINE_DISCOVER=all):
          collect_siamese.nested_loo_audit's partial (lower-bound) estimate of
          the optimism of picking 1-of-N configs on the reported folds.
    A siamese row with neither is named as undisclosed. The returned lines go
    into comparison_notes.txt, so the asymmetry travels with the CSV instead
    of living in a JSON nobody opens.
    """
    lines = []
    print = _collecting_print(lines)  # noqa: A001
    rows, modality, unrecorded, allmod = [], {}, [], {}
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        for model, summary in results[cv].items():
            a = summary.get("selection_audit")
            a = a if isinstance(a, dict) else {}
            ms = a.get("modality_selection")
            nested = a.get("nested_loo_auprg_mean") is not None
            if nested:
                rows.append((cv, model, a, summary.get("auprg_mean")))
            am = a.get("all_modalities")
            if isinstance(am, dict):
                # ALLMOD_COMBO=1 (collect_siamese.allmod_record): no selection
                key = json.dumps({k: am.get(k) for k in
                                  ("modalities", "absent_optional")},
                                 sort_keys=True)
                allmod.setdefault(key, []).append((f"{model}/{cv}", am))
            elif isinstance(ms, dict):
                # One block per distinct selection (normally exactly one: the
                # combo job selects once for every variant and CV).
                key = json.dumps({k: v for k, v in ms.items()
                                  if k != "modalities"}, sort_keys=True)
                modality.setdefault(key, []).append((f"{model}/{cv}", ms))
            elif not nested and is_siamese_row(model):
                unrecorded.append(f"{model}/{cv}")
    if not (rows or modality or unrecorded or allmod):
        return lines
    print("\nSELECTION BUDGET (asymmetric across rows — disclose in methods)")
    for cells in allmod.values():
        am = cells[0][1]
        absent = am.get("absent_optional") or []
        print(f"  {', '.join(sorted(c for c, _ in cells))}: NO modality "
              f"selection, every catalog embedding in every fold "
              f"({am.get('n_modalities')}: "
              f"{', '.join(am.get('modalities') or [])}"
              + (f"; not generated, optional: {', '.join(absent)}"
                 if absent else "") +
              "). The same pairs as the selected combo (its selection split "
              "is held out of training too, and chooses nothing).")
    for cells in modality.values():
        ms = cells[0][1]
        scv = ms.get("selection_cv")
        winners = ms.get("winners") or {}
        if ms.get("selection_mode") == "per_fold":
            # SELECTION_MODE=per_fold (collect_siamese._per_fold_record): one
            # record per CV, each fold of it on its own winners.
            per = ms.get("per_fold") or {}
            print(f"  siamese modalities of {scv}, selected PER FOLD, shared "
                  f"by {', '.join(sorted(c for c, _ in cells))}: " +
                  "; ".join(f"{k}: " + ", ".join(
                      f"fold {f} {t}" for f, t in sorted(per.get(k, {}).items()))
                            for k in sorted(per)) +
                  f". Each fold's pick per category is the argmax of "
                  f"{ms.get('selection_label') or ms.get('selection_metric')} "
                  f"on that fold's own selection split (a random "
                  f"{ms.get('selection_frac')} of its TRAINING pairs, never "
                  f"trained on; seed {ms.get('selection_seed')}) over the "
                  f"single-embedding runs trained as "
                  f"{ms.get('selection_variant')}; the combo trains each fold "
                  f"on its own picks (embeddings picked per fold: "
                  f"{', '.join(f'{e} x{n}' for e, n in (ms.get('n_folds_picked') or {}).items())}). "
                  f"Leakage-free per fold: no fold's test pairs reached its "
                  f"own choice.")
        else:
            print(f"  siamese modalities, one selection shared by "
                  f"{', '.join(sorted(c for c, _ in cells))}: "
                  f"{', '.join(f'{k}={v}' for k, v in sorted(winners.items()))}. "
                  f"Each is the per-category argmax of {scv} VALIDATION "
                  f"{ms.get('selection_label') or ms.get('selection_metric')} "
                  f"over {ms.get('n_scored_on_val')} single-embedding runs trained "
                  f"as {ms.get('selection_variant')}, chosen ONCE and reused for "
                  f"every CV and variant. Not leakage-free (accepted): the mean "
                  f"pools the {scv} folds' validation scores, and each fold's "
                  f"validation split is carved from a training pool that holds "
                  f"other folds' test pairs. Those pools span the pair set every "
                  f"CV shares, so the test pairs of EVERY CV's rows (not only "
                  f"{scv}'s) reached the choice, a share of each test fold "
                  f"directly as {scv} validation pairs. One combo per CV, so no "
                  f"nested estimate of the optimism exists.")
        # The gaps Step 2 lets through only under BEST_CAT_ALLOW_MISSING=1.
        gaps = [f"{k} {', '.join(map(str, ms[k]))}"
                for k in ("missing_categories", "unranked_failed",
                          "unranked_foreign_token")
                if ms.get(k)]
        if gaps:
            print(f"      selected on an INCOMPLETE candidate set "
                  f"(BEST_CAT_ALLOW_MISSING=1): {'; '.join(gaps)}.")
        # An OPTIONAL_TYPES absence is by design (a warning); any other is not.
        if ms.get("unranked_not_generated"):
            print(f"      not ranked, embedding file never generated: "
                  f"{', '.join(map(str, ms['unranked_not_generated']))}.")
        if ms.get("unranked_required_missing"):
            print(f"      NOT ranked although generation should have built it "
                  f"(--skip-generate?): "
                  f"{', '.join(map(str, ms['unranked_required_missing']))} — "
                  f"the argmax ran on a smaller catalog than the paper's.")
        # collect_siamese attaches a record only to a combo trained on exactly
        # these winners (the combo refuses a missing winner file), so this
        # is a guard against a hand-made or legacy record.
        chosen = sorted(winners.values())
        if ms.get("selection_mode") == "per_fold":
            # "a/b": every embedding the CV's folds picked for a category
            chosen = sorted(x for v in winners.values()
                            for x in str(v).split("/"))
        for cell_name, m in cells:
            if sorted(m.get("modalities") or []) != chosen:
                print(f"      {cell_name} trained on "
                      f"{', '.join(m.get('modalities') or [])}, not this "
                      f"selection's winners (a legacy or foreign combo; "
                      f"retrain it).")
    if unrecorded:
        print(f"  {', '.join(unrecorded)}: no selection record (a pinned, "
              f"single-candidate or derived row, or a selection file that did "
              f"not match its combo; see the collect log) — how this "
              f"configuration was chosen is NOT disclosed here.")
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
                     f"is {table_auprg + delta:.4f}. A partial adjustment: the "
                     f"other folds' validation splits hold this fold's test "
                     f"pairs, so the optimism can be larger than shown.")
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
    return lines


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


class _TailWriter:
    """A csv.writer whose every row ends with the same extra cells."""

    def __init__(self, writer, tail):
        self._w = writer
        self._tail = list(tail)

    def writerow(self, row):
        self._w.writerow(list(row) + self._tail)


def write_csv(results,
              csv_path,
              provenance_lines=(),
              input_lines=(),
              report_lines=(),
              pca_arm=None,
              head_lines=()):
    """comparison.csv and its notes.

    pca_arm (compare.py --pca_arm): a trailing pca_arm column, the same value
    on every row, since every arm's CSV has the same model names; without it
    the header stays cv,model,metric,mean,std. head_lines open the notes
    (the arm line first)."""
    # An arm's directory (sl_comparison/results/pca_<arm>) need not exist yet.
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    tail = [] if pca_arm is None else [pca_arm]
    with open(csv_path, "w", newline="") as f:
        base = csv.writer(f)
        base.writerow(["cv", "model", "metric", "mean", "std"]
                      + ([] if pca_arm is None else ["pca_arm"]))
        w = _TailWriter(base, tail)
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
                # Model size (and the training seed, when the trainer passed
                # it to write_model_params) as ordinary rows so the CSV stays
                # tidy/long; std is blank. A siamese row's size is the LAST
                # fold's (PARAM_FOOTNOTE); its per-fold mean / sd is the
                # trainable_params_fold row written above.
                # n_folds_scored / n_folds_expected / partial: how many folds
                # the means above are over. A partial mean is invisible in the
                # metric columns themselves, and the CSV outlives the log.
                for key in ("trainable_params", "total_params", "seed",
                            "n_folds_scored", "n_folds_expected"):
                    if isinstance(summary.get(key), int) and \
                            not isinstance(summary.get(key), bool):
                        w.writerow([cv, model, key, summary[key], ""])
                if summary.get("partial") is not None:
                    w.writerow([cv, model, "partial",
                                bool(summary["partial"]), ""])
                # WHICH model ran, as data: the deviations upstream_input_report
                # and model_config_report name in the notes (SLGNN --arch port,
                # NSF4SL --features embedding / --legacy_loop). They are what
                # tells an author's model from a substitute, so the archival
                # CSV must carry them too, not only the notes sidecar.
                for key in ("arch", "features", "legacy_loop"):
                    if summary.get(key) is not None:
                        w.writerow([cv, model, key, summary[key], ""])
                # PROVENANCE AS DATA, not just as a stdout warning.
                #
                # provenance_report prints "rows come from DIFFERENT fold sets"
                # to the SLURM log, which nobody reads next to the CSV. The CSV
                # is the archival artifact, and the results directories are
                # never cleared — a model whose training failed keeps its
                # previous results.json, load_results picks it up, and it is
                # archived here indistinguishably from rows graded on the
                # current folds.
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
        ("\n".join(head_lines) + "\n\n" if head_lines else "") +
        "Notes for " + Path(csv_path).name + "\n" + "=" * 60 + "\n\n" +
        MAIN_FOOTNOTE + "\n\n" + RETRIEVAL_FOOTNOTE + "\n\n" +
        "aupr_norm is derived from `aupr` and inherits its defect: it is\n"
        "(aupr - baseline) / (1 - baseline), so a constant predictor scores\n"
        "0.5 rather than 0. Use ap_norm. Both are in this CSV for archival\n"
        "completeness; only ap / ap_norm / auprg / auroc are safe to report.\n\n"
        + PARAM_FOOTNOTE + "\n" +
        ("\n" + "\n".join(input_lines) + "\n" if input_lines else "") +
        # Selection regime, label semantics and the selection budget: what
        # each row's label means and how its configuration was chosen. As
        # with provenance, printing alone left them only in the SLURM log.
        ("\n" + "\n".join(report_lines).lstrip("\n") + "\n"
         if report_lines else "") +
        ("\n\nFOLD-SET PROVENANCE (as printed by compare.py; the\n"
         "per-row sha256 fingerprints are also rows in the CSV itself)\n" +
         "-" * 60 + "\n" + "\n".join(provenance_lines) +
         "\n" if provenance_lines else ""))
    print(f"\nTidy CSV -> {csv_path}")
    print(f"Caveats   -> {notes}")


def seed_report(results):
    """Training seed per row (model_params.json `seed`, when the trainer
    records it). The folds' seed in provenance is a different number, so an
    unrecorded training seed is named rather than assumed. Returns the
    printed lines for comparison_notes.txt."""
    lines = []
    print = _collecting_print(lines)  # noqa: A001
    seeds = {}
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        for model, summary in results[cv].items():
            s = summary.get("seed")
            seeds.setdefault("UNRECORDED" if s is None else str(s),
                             []).append(f"{model}/{cv}")
    if seeds:
        print("\nTRAINING SEED (model_params.json; not the fold seed)")
        for s, cells in sorted(seeds.items()):
            print(f"  {s}: {', '.join(sorted(cells))}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Four-model comparison table")
    ap.add_argument("--results_dir", default="sl_comparison/results",
                    help="the <model>_<cv>.json rows (default: "
                    "sl_comparison/results); with --shared_results_dir only "
                    "its SLeuth rows (siamese_sl*), e.g. a PCA arm's "
                    "sl_comparison/results/pca_<arm>")
    ap.add_argument("--shared_results_dir", default=None,
                    help="read every row that is not a SLeuth row (SLGNN / "
                    "NSF4SL / SLMGAE / DegreePrior) from this directory "
                    "only, and none of its SLeuth rows: sl_comparison/"
                    "results, which holds the shared models' rows for every "
                    "PCA arm (run_compare.sh). Default: one directory")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pca_arm", choices=PCA_ARMS, default=None,
                    help="the PCA arm of the SLeuth rows: named under each "
                    "table's title, on the first line of the notes and in "
                    "a trailing pca_arm column of the CSV (which has no "
                    "such column without it)")
    ap.add_argument("--pca_label", default=None,
                    help="what the arm trains, one line (config.conf "
                    "pca_arm_label), next to --pca_arm")
    args = ap.parse_args(argv)
    if args.pca_label is not None and args.pca_arm is None:
        ap.error("--pca_label names the arm --pca_arm gives; pass both")

    # Remove last run's CSV and notes before anything can fail: an empty or
    # crashed run must not leave them looking current (paper_tables.py reads
    # comparison.csv).
    if args.csv:
        csv_path = Path(args.csv)
        for p in (csv_path, csv_path.with_name(csv_path.stem + "_notes.txt")):
            p.unlink(missing_ok=True)

    arm_line = (None if args.pca_arm is None
                else pca_arm_line(args.pca_arm, args.pca_label))
    if arm_line:
        print(arm_line)
    head_lines = [arm_line] if arm_line else []
    if args.shared_results_dir is None:
        results = load_results(args.results_dir)
        where = str(args.results_dir)
    else:
        results, dropped_arm, dropped_shared = load_split_results(
            args.results_dir, args.shared_results_dir)
        where = (str(args.results_dir)
                 if Path(args.results_dir).resolve()
                 == Path(args.shared_results_dir).resolve() else
                 f"{args.results_dir} (SLeuth rows) or "
                 f"{args.shared_results_dir} (shared-model rows)")
        if results:
            head_lines += split_report(args.results_dir,
                                       args.shared_results_dir, results,
                                       dropped_arm, dropped_shared)
    if not results:
        print(f"No results in {where}. Run the models + "
              f"evaluate_model.py / collect_siamese.py first.")
        return 1
    for cv in sorted(results,
                     key=lambda c: CV_ORDER.index(c) if c in CV_ORDER else 99):
        print_table(cv, results[cv], subtitle=arm_line)
        if any(f"{k}_mean" in s for s in results[cv].values()
               for _, k in RETRIEVAL_METRICS):
            print_table(cv,
                        results[cv],
                        metrics=RETRIEVAL_METRICS,
                        title="retrieval (see normalisation footnote)",
                        footnote=RETRIEVAL_FOOTNOTE,
                        subtitle=arm_line)
    print(PARAM_FOOTNOTE)
    input_lines = upstream_input_report(results)
    report_lines = (selection_regime_report(results) +
                    seed_report(results) +
                    model_config_report(results) +
                    partial_report(results) +
                    label_semantics_report(results) +
                    selection_report(results))
    prov_lines = provenance_report(results)
    if args.csv:
        write_csv(results,
                  args.csv,
                  provenance_lines=prov_lines,
                  input_lines=input_lines,
                  report_lines=report_lines,
                  pca_arm=args.pca_arm,
                  head_lines=head_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
