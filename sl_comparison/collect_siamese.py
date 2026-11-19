#!/usr/bin/env python3
"""
Collect siamese_sl's OR-collapse (`*_any`) metrics into the unified comparison
schema, so it sits in the same table as the shared-fold models.

siamese_sl runs its OWN pipeline (its embedding selection + cell-line model on
SLDB 3.0), and natively reports the OR / "SL in any cell line" view as
`auroc_any`, `aupr_any`, `auprg_any`, `aupr_any_norm` in its results.json. This
maps those onto the common metric names (auroc/aupr/auprg/aupr_norm) and writes
<out_dir>/siamese_sl_<cv>.json, in the PCA arm's comparison root
(sl_comparison/results for arm none, sl_comparison/results/pca_<arm> for the
others; siamese_sl/slurm/config.conf, PCA arms), from the arm's siamese
results root (--discover_dir: siamese_sl/results, or
siamese_sl/results/pca_<arm>).

train.py --folds_dir (required: selection is val-only) makes siamese use the
SAME canonical folds, gene universe and test edges as SLMGAE / SLGNN / NSF4SL,
and sl_comparison/slurm/collect_and_regrade_siamese.sh (sourced by
run_compare.sh) re-grades its saved matrices through evaluate_model.py, so the
published numbers come from the identical evaluator. The JSON written here is
siamese's SELF-SCORED view and only an intermediate: never publish it without
that re-grade. convert() attaches a caveat saying so.

Which run becomes a row. --combo_markers (what collect_and_regrade uses)
takes, per CV, ONLY the combo named by this run's
<discover_dir>/<prefix>latest_combo_<cv>.txt. run_best_per_category_combo.sh
deletes those markers when it starts and writes each one only after that
(variant, CV) combo trained, while a previous run's combo directories can
outlive a failed or never-started combo job; so without a marker a CV gets NO
row, never an older model. Without --combo_markers, discovery ranks every
matching results.json under --discover_dir, with no freshness check at all,
except those under another PCA arm's root inside it (PCA_ARM_ROOTS).

Model variants (siamese_sl/slurm/config.conf SIAMESE_VARIANTS). The baseline
pd:per_head writes siamese_sl_<cv>.json as always. Every other variant trains
under results/var_<kernel>_<cell_lines>__* and gets its own row,
siamese_sl_<kernel>_<cell_lines>, via --model_name (+ --variant); its combo
markers carry the same var_<kernel>_<cell_lines>__ prefix. The baseline
discovery skips every var_* directory, so a variant can never be selected
into the siamese_sl row. On top of the directory filter, each
candidate's recorded train.py config must match the row's variant
(--variant, else parsed from --model_name): the baseline keeps only runs
whose kernel is pd (or unrecorded) and cell_lines per_head (or unrecorded);
a variant row keeps only runs recording exactly its kernel and cell_lines.

All-modalities combo (2026-11-12; siamese_sl/slurm/run_allmod_combo.sh,
ALLMOD_COMBO=1). --allmod --combo_markers builds row siamese_sl_allmod
(pd:per_head) or siamese_sl_allmod_<kernel>_<cell_lines> from the run named
by <discover_dir>/allmod__<variant prefix>latest_allmod_<cv>.txt, which must
record all_modalities in its config; the row discloses selection_audit
["all_modalities"] (allmod_record: no selection) instead of a modality
selection. Every other row's discovery skips allmod__* directories and any
run recording all_modalities.

Selection level. Candidates are ranked on the validation score of the metric
the run's OWN checkpoint selection used, val_<select_metric>_mean
(select_metric from its config; 'auprg' for runs predating the flag), falling
back to val_auprg_mean — the same rule run_best_per_category_combo.sh uses for
category winners.

Selection disclosure. With --combo_markers each CV has one candidate, so there
is no config sweep to audit; what was selected is the combo's MODALITIES, once,
by run_best_per_category_combo.sh on SELECTION_CV validation. Each row carries
that record (<discover_dir>/best_per_category.json, checked against the
combo's own embeddings_paths) as selection_audit["modality_selection"], which
the re-grade carries (evaluate_model.CARRY_KEYS) and compare.py prints under
SELECTION BUDGET and writes into comparison_notes.txt. The sweep-wide
discovery (no --combo_markers) attaches nested_loo_audit instead.

PCA arm (2026-10-21). The modality selection record carries the arm the
selection belongs to and its PCA (_selection_meta pca_arm / pca), and
--pca_arm (label only; collect_and_regrade_siamese.sh passes pca_arm.sh's
SLEUTH_ARM) writes selection_audit["pca_arm"] on every row, the pinned and
swept ones included, which have no selection record: every arm's rows have
the same names, and arm none's sit at the pre-arms path.

Usage (normally via collect_and_regrade_siamese.sh, which re-grades after):
  python sl_comparison/collect_siamese.py --siamese_results <path/results.json> \
      --cv_type cv3 --out sl_comparison/results/siamese_sl_cv3.json
  python sl_comparison/collect_siamese.py --discover_dir siamese_sl/results \
      --combo_markers                       # baseline row: this run's combos
  python sl_comparison/collect_siamese.py --discover_dir siamese_sl/results \
      --model_name siamese_sl_bilinear_pooled --variant bilinear:pooled \
      --combo_markers
"""

import argparse
import importlib.util
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

BASELINE_MODEL = "siamese_sl"
# Results directories of the non-baseline model variants start with this
# (siamese_sl/slurm/config.conf, siamese_variant_prefix). The baseline row's
# discovery must never rank them: they are different models.
VARIANT_DIR_PREFIX = "var_"
# The all-modalities combo (siamese_sl/slurm/run_allmod_combo.sh, config.conf
# allmod_prefix / allmod_row, 2026-11-12): every catalog embedding, no
# modality selection. Its runs are <root>/allmod__<variant prefix>
# all_modalities_<cv>, its success markers <root>/allmod__<variant prefix>
# latest_allmod_<cv>.txt, its rows siamese_sl_allmod[_<tag>]. Only --allmod
# collects them; no other row's discovery ranks one.
ALLMOD_DIR_PREFIX = "allmod__"
ALLMOD_DIR_STEM = "all_modalities_"
ALLMOD_MARKER_STEM = "latest_allmod_"
ALLMOD_MODEL = "siamese_sl_allmod"
ALLMOD_RECORD = "all_modalities.json"

# What a POSITIVE means is the same for every variant (the OR label); what the
# model SCORES differs with --cell_lines.
LABEL_SEMANTICS_PER_HEAD = (
    "OR-collapse: a pair is POSITIVE if it is SL in ANY of the 8 cell-line "
    "heads. Since 2026-08-10 the shared folds use the SAME rule "
    "(conflict_policy=or_collapse: positive if any screen called it SL in "
    "any line, negative only if every screen called it non-SL), so this row "
    "and the SLGNN / NSF4SL / SLMGAE / DegreePrior rows are graded on an "
    "identical positive set — including the 5,542 pairs screened both ways, "
    "which every row now counts as positive. What still differs is the "
    "SCORE: siamese emits 8 per-head logits collapsed by max, the others "
    "emit one score for the unlabelled relation.")
LABEL_SEMANTICS_POOLED = (
    "OR-collapse: a pair is POSITIVE if it is SL in ANY cell line — the shared "
    "folds' conflict_policy=or_collapse label, identical to the SLGNN / NSF4SL "
    "/ SLMGAE / DegreePrior rows. This siamese variant (--cell_lines pooled) "
    "is TRAINED on that label directly: one head, no cell-line embedding, one "
    "score per pair, like the other models.")
LABEL_SEMANTICS_SINGLE = (
    "OR-collapse: a pair is POSITIVE if it is SL in ANY cell line — the shared "
    "folds' conflict_policy=or_collapse label, identical to the SLGNN / NSF4SL "
    "/ SLMGAE / DegreePrior rows. This siamese row is the single-output "
    "SiameseSL (train.py without --use_cell_lines), TRAINED on the shared "
    "folds' own pos/neg labels: one output, no cell-line embedding, one score "
    "per pair, like the other models.")


def marker_name(model_name, cv_type):
    """Selected-run marker file. The baseline keeps its historical name
    (siamese_selected_<cv>.txt) so existing pins and re-grades still find it."""
    stem = "siamese" if model_name == BASELINE_MODEL else model_name
    return f"{stem}_selected_{cv_type}.txt"


def label_semantics_for(cfg):
    """The label_semantics text for a run's train.py config.

    A single-output run (use_cell_lines False) still records the argparse
    default cell_lines='per_head', so cell_lines alone would describe it as 8
    heads collapsed by max. `is False`, not falsiness: a run that never
    recorded the key keeps the per-head text, as before.
    """
    if cfg.get("use_cell_lines") is False:
        return LABEL_SEMANTICS_SINGLE
    if cfg.get("cell_lines") == "pooled":
        return LABEL_SEMANTICS_POOLED
    return LABEL_SEMANTICS_PER_HEAD


KERNELS = ("pd", "bilinear")
CELL_LINES = ("per_head", "pooled")
BASELINE_VARIANT = ("pd", "per_head")
CV_TYPES = ("cv1", "cv2", "cv3")
# run_best_per_category_combo.sh names a variant's combos
# results/<prefix>best_cat_<modalities>_<cv> and its success marker
# results/<prefix>latest_combo_<cv>.txt; <prefix> is variant_dir_prefix.
COMBO_DIR_STEM = "best_cat_"
# ...and writes its category winners (Step 2) to results/<this>, deleting the
# previous one at job start, before any marker can exist.
SELECTION_JSON = "best_per_category.json"
# The PCA arms (siamese_sl/slurm/config.conf): arm none's results root is
# siamese_sl/results itself and every other arm's is a child of it,
# results/pca_<arm>/ (pca_arms is the cross-arm table's name on the
# comparison side; pca_none is never made, listed so it can never be read).
# Another arm's runs are other models, so the sweep (no --combo_markers) of
# one arm's root never ranks a run under one of these. Exact names: the
# Step-1 runs' results/pcavar<v>_<type>_<cv> are an arm's own.
PCA_ARM_ROOTS = ("pca_none", "pca_joint", "pca_per_modality", "pca_both",
                 "pca_arms")


def variant_dir_prefix(variant):
    """'' for the baseline pd:per_head, else 'var_<kernel>_<cell_lines>__'
    (config.conf siamese_variant_prefix)."""
    if variant == BASELINE_VARIANT:
        return ""
    return f"{VARIANT_DIR_PREFIX}{variant[0]}_{variant[1]}__"


def allmod_dir_prefix(variant):
    """'allmod__' + variant_dir_prefix (config.conf allmod_prefix)."""
    return ALLMOD_DIR_PREFIX + variant_dir_prefix(variant)


def allmod_row_name(variant):
    """siamese_sl_allmod for pd:per_head, else siamese_sl_allmod_<k>_<c>
    (config.conf allmod_row)."""
    if variant == BASELINE_VARIANT:
        return ALLMOD_MODEL
    return f"{ALLMOD_MODEL}_{variant[0]}_{variant[1]}"


def is_allmod_row(model):
    """A row of the all-modalities combo (or a derived row of one)."""
    return model == ALLMOD_MODEL or str(model).startswith(ALLMOD_MODEL + "_")


def allmod_selected_row(model):
    """The selected combo's row an all-modalities row stands next to
    (siamese_sl_allmod -> siamese_sl, siamese_sl_allmod_<tag> ->
    siamese_sl_<tag>), or None for any other row."""
    if model == ALLMOD_MODEL:
        return BASELINE_MODEL
    if str(model).startswith(ALLMOD_MODEL + "_"):
        return BASELINE_MODEL + model[len(ALLMOD_MODEL):]
    return None


def run_config(data, path):
    """The run's train.py args: results.json `config`, else config.json."""
    cfg = data.get("config")
    if isinstance(cfg, dict) and cfg:
        return cfg
    try:
        with open(Path(path).parent / "config.json") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_variant(text):
    """'KERNEL:CELL_LINES' -> (kernel, cell_lines); ValueError otherwise."""
    k, sep, c = str(text).partition(":")
    if not sep or k not in KERNELS or c not in CELL_LINES:
        raise ValueError(f"variant must be KERNEL:CELL_LINES with KERNEL in "
                         f"{KERNELS} and CELL_LINES in {CELL_LINES}; got "
                         f"{text!r}")
    return k, c


def expected_variant(model_name, variant=None):
    """(kernel, cell_lines) a row must hold, or None when unknown.

    --variant wins; else siamese_sl -> the baseline pd:per_head and
    siamese_sl_<kernel>_<cell_lines> -> that variant (the row names
    collect_and_regrade_siamese.sh builds from config.conf).
    """
    if variant:
        return parse_variant(variant)
    sel = allmod_selected_row(model_name)
    if sel is not None:
        # the all-modalities row of a variant holds that variant
        model_name = sel
    if model_name == BASELINE_MODEL:
        return BASELINE_VARIANT
    prefix = BASELINE_MODEL + "_"
    if model_name.startswith(prefix):
        k, _, c = model_name[len(prefix):].partition("_")
        if k in KERNELS and c in CELL_LINES:
            return k, c
    return None


def variant_matches(cfg, want):
    """Does a run's config hold variant `want`?

    The baseline accepts a missing key (runs from before --kernel /
    --cell_lines existed were all pd:per_head); a variant needs both keys.
    """
    if want is None:
        return True
    k, c = cfg.get("kernel"), cfg.get("cell_lines")
    if want == BASELINE_VARIANT:
        return k in (None, "pd") and c in (None, "per_head")
    return (k, c) == want


def selection_val_key(summary, cfg):
    """(per-fold val key, value of its fold mean) the run was selected on.

    val_<select_metric>, select_metric from the run's config ('auprg' when
    the run predates --select_metric), falling back to val_auprg. (None, None)
    when the run carries no validation score.
    """
    sm = cfg.get("select_metric") or "auprg"
    for key in (f"val_{sm}", "val_auprg"):
        if summary.get(f"{key}_mean") is not None:
            return key, summary[f"{key}_mean"]
    return None, None

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
    "min(k, #relevant), not k (cal_metrics' precision_k block), so "
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


def convert(summary,
            cv_type,
            model_name=BASELINE_MODEL,
            label_semantics=LABEL_SEMANTICS_PER_HEAD):
    """Map a siamese summary's *_any keys onto the common schema (or None).

    label_semantics: label_semantics_for(the run's config)."""
    out = {
        "model":
        model_name,
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
        label_semantics,
        # Replaced by the re-grade (evaluate_model.SHARED_NOTE), so a row that
        # still carries this text in the table was never re-graded.
        "note":
        "SELF-SCORED by siamese's own evaluator (collect_siamese.py); not "
        "re-graded by sl_comparison/evaluate_model.py. AUPRG / AP_norm / "
        "AUROC are the cross-model metrics; AUPRG and AP_norm are "
        "baseline-anchored skill scores (random -> 0 at any prevalence), not "
        "prevalence-invariant."
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


def nested_loo_audit(cands, val_key="val_auprg"):
    """Leave-one-fold-out audit of picking 1-of-N configs on validation.

    Selecting the config with the best mean validation score (`val_key`, the
    candidates' own selection metric) and then reporting that config's mean
    TEST metric reuses the same folds for selection and for reporting. This
    audit removes the direct reuse: for each fold k, choose the config
    maximising the mean validation score over the OTHER folds, and score it on
    fold k only. It is NOT an unbiased estimate: each other fold's validation
    split is carved from that fold's training pool, which contains fold k's
    test pairs (always in cv1; in cv2/cv3 whenever the gene groups overlap),
    so fold k's test labels still reach the choice. Read the delta as a lower
    bound on the selection optimism, not as a correction to publish.

    On the 2026-08-06 run this left cv1 unchanged and cv2 nearly so, but moved
    cv3 from 0.8797 to ~0.845. The residual is
    NOT a constant offset: it is dominated by single folds where two configs
    are separated by a val margin far smaller than the fold-to-fold spread, so
    report it with the per-fold choices rather than as a flat correction.

    siamese_sl is the only model in this table selected over a config sweep;
    SLGNN and NSF4SL each get one hard-coded embedding and SLMGAE one config,
    so disclosing this is what keeps the comparison symmetric. It needs >= 2
    candidates, so it applies only to the sweep-wide discovery (no
    --combo_markers); the combo path discloses its modality selection instead
    (modality_selection_record).
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
        f"leave-one-fold-out config selection on {val_key}; the "
        "reported row selects on all folds at once and is optimistic "
        "by the difference",
    }


def load_modality_selection(discover_dir):
    """{"meta", "winners"} from <discover_dir>/best_per_category.json, or None.

    run_best_per_category_combo.sh deletes that file when it starts (with the
    latest_combo markers) and rewrites it in Step 2, before any marker is
    written, so while a marker exists the file describes the same job's
    selection. winners are the SELECTION_CV entries: the same types are
    replicated under every CV key.
    """
    path = Path(discover_dir) / SELECTION_JSON
    try:
        with open(path) as f:
            sel = json.load(f)
        meta = sel["_selection_meta"]
        winners = {
            cat: w
            for cat, w in sel[meta["selection_cv"]].items()
            if isinstance(w, dict) and w.get("type") and w.get("file")
        }
        # A per_fold selection (SELECTION_MODE=per_fold, 2026-11-12) picks
        # per (CV, fold): each CV's block holds that CV's picks, so the
        # record is read per combo CV (modality_selection_record).
        by_cv = {
            cv: {cat: w for cat, w in block.items()
                 if isinstance(w, dict) and w.get("type") and w.get("file")}
            for cv, block in sel.items()
            if not cv.startswith("_") and isinstance(block, dict)
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError,
            AttributeError) as e:
        print(f"[discover] WARNING: no readable selection record in {path} "
              f"({type(e).__name__}: {e}); the siamese modality selection "
              f"cannot be disclosed with the table.")
        return None
    return {"meta": meta, "winners": winners, "by_cv": by_cv, "raw": sel}


_FOLD_SELECTION = None


def _fold_selection():
    """siamese_sl/fold_selection.py (stdlib only), loaded once by path."""
    global _FOLD_SELECTION
    if _FOLD_SELECTION is None:
        p = Path(__file__).resolve().parents[1] / "siamese_sl" / \
            "fold_selection.py"
        spec = importlib.util.spec_from_file_location("_cs_fold_selection", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FOLD_SELECTION = mod
    return _FOLD_SELECTION


def _per_fold_record(selection, cfg, cfg_dir=None, why=None):
    """modality_selection_record of a per_fold selection: the combo of
    cfg's CV must be trained on the union of that CV's per-fold winners
    (its embeddings_paths, by file name) AND each fold on its own winners
    (config.json fold_modalities: fold_selection.run_matches_spec, as
    paper_tables checks it), or None (the reason appended to `why`)."""
    block = (selection.get("by_cv") or {}).get(cfg.get("cv_type"))
    if not block:
        if why is not None:
            why.append(f"has a CV ({cfg.get('cv_type')}) the selection has "
                       f"no block for")
        return None
    by_file = {}
    for w in block.values():
        for t, fl in zip(w.get("types") or [w["type"]],
                         w.get("files") or [w["file"]]):
            by_file[Path(fl).name] = t
    used = [Path(p).name for p in cfg.get("embeddings_paths") or []]
    if not used or sorted(used) != sorted(by_file):
        if why is not None:
            why.append("trained on embeddings that are not the union of its "
                       "CV's per-fold winners")
        return None
    fs = _fold_selection()
    try:
        bad = fs.run_matches_spec(
            cfg, fs.combo_spec(selection.get("raw") or {},
                               cfg.get("cv_type")), cfg_dir)
    except (fs.SelectionError, KeyError, TypeError, ValueError) as e:
        bad = f"{type(e).__name__}: {e}"
    if bad:
        if why is not None:
            why.append(f"is not the per_fold selection's combo ({bad})")
        return None
    meta = selection["meta"]
    arm = {k: meta[k] for k in ("pca_arm", "pca") if meta.get(k) is not None}
    counts = {}
    for w in block.values():
        for t, n in (w.get("n_folds_picked") or {}).items():
            counts[t] = counts.get(t, 0) + int(n)
    return {
        **arm,
        "selection_mode": "per_fold",
        "selection_cv": cfg.get("cv_type"),
        "selection_variant": meta.get("selection_variant"),
        "selection_metric": meta.get("selection_metric"),
        "selection_label": meta.get("selection_label"),
        "selection_frac": meta.get("selection_frac"),
        "selection_seed": meta.get("selection_seed"),
        "n_scored_on_val": meta.get("n_scored_on_val"),
        # per category: every embedding a fold of this CV picked
        "winners": {cat: "/".join(w.get("types") or [w["type"]])
                    for cat, w in sorted(block.items())},
        "per_fold": {cat: {k: p.get("type") for k, p in
                           sorted((w.get("per_fold") or {}).items(),
                                  key=lambda kv: int(kv[0]))}
                     for cat, w in sorted(block.items())},
        "n_folds_picked": dict(sorted(counts.items())),
        "modalities": sorted(by_file[u] for u in used),
        "missing_categories": meta.get("missing_categories") or [],
        "unranked_failed": meta.get("unranked_failed") or [],
        "unranked_foreign_token": meta.get("unranked_foreign_token") or [],
        "unranked_not_generated": meta.get("unranked_not_generated") or [],
        "unranked_required_missing":
            meta.get("unranked_required_missing") or [],
        "note": "modalities chosen per (CV, fold): each category's argmax of "
                "the single-embedding runs' score on that fold's own "
                "selection split (a random selection_frac of its TRAINING "
                "pairs, never trained on); the combo trains each fold on its "
                "own winners (their union is its embeddings_paths). No test "
                "pair of any fold reached its own fold's choice",
    }


def modality_selection_record(selection, cfg, cfg_dir=None, why=None):
    """How a combo's modalities were chosen, or None if they are not the
    selection's winners (the JSON is then not this combo's selection).

    The combo trains on every winner's embedding file or refuses
    (run_best_per_category_combo.sh PYEXTRACT), so its embeddings_paths must
    be exactly the winners' files; anything else is a legacy or foreign combo.
    A per_fold selection also needs each fold's own winners (cfg_dir: the
    run directory holding config.json; why: a list the per_fold reason is
    appended to).
    """
    if (selection["meta"].get("selection_mode") or "global") == "per_fold":
        return _per_fold_record(selection, cfg, cfg_dir, why)
    by_file = {Path(w["file"]).name: w["type"]
               for w in selection["winners"].values()}
    used = [Path(p).name for p in cfg.get("embeddings_paths") or []]
    if not used or sorted(used) != sorted(by_file):
        return None
    meta = selection["meta"]
    # The PCA arm the selection belongs to and its PCA in words, as the combo
    # job records them in _selection_meta (2026-10-20): the row JSONs of every
    # arm have the same names, and arm none's sit at the pre-arms path, so
    # the record (carried through the re-grade: evaluate_model.CARRY_KEYS)
    # says which arm's models a row is (2026-10-21). A selection from before
    # the arms records neither, and gets no None values.
    arm = {k: meta[k] for k in ("pca_arm", "pca") if meta.get(k) is not None}
    return {
        **arm,
        "selection_cv": meta.get("selection_cv"),
        "selection_variant": meta.get("selection_variant"),
        "selection_metric": meta.get("selection_metric"),
        "selection_label": meta.get("selection_label"),
        "n_scored_on_val": meta.get("n_scored_on_val"),
        "winners": {cat: w["type"]
                    for cat, w in sorted(selection["winners"].items())},
        "modalities": sorted(by_file[u] for u in used),
        "missing_categories": meta.get("missing_categories") or [],
        "unranked_failed": meta.get("unranked_failed") or [],
        # Every catalog entry the argmax skipped, as Step 2 records them.
        "unranked_foreign_token": meta.get("unranked_foreign_token") or [],
        "unranked_not_generated": meta.get("unranked_not_generated") or [],
        # Of those, the ones OUTSIDE OPTIONAL_TYPES: generation should have
        # built them, so their absence is NOT an accepted one (--skip-generate
        # with a catalog file missing) and the argmax ranked a smaller catalog
        # than the paper's. Dropping it here left compare.py printing only the
        # by-design wording for both kinds.
        "unranked_required_missing":
            meta.get("unranked_required_missing") or [],
        "note": "modalities chosen ONCE as the per-category argmax of "
                "single-embedding validation scores on selection_cv, reused "
                "for every CV and variant; the fold means pool validation "
                "splits whose training pools hold other folds' test pairs, "
                "and those pools span the pair set every CV shares, so test "
                "pairs of every CV (a share of each test fold directly as "
                "selection_cv validation pairs) reached the choice: it is "
                "not leakage-free",
    }


def allmod_record(run_dir, cfg):
    """What an all-modalities row discloses instead of a modality selection
    (selection_audit["all_modalities"]): every catalog embedding the run
    trained on, in its order, chosen by nothing. From the run's
    all_modalities.json (run_allmod_combo.sh: the types, the catalog size and
    the optional catalog types that were absent), else its config."""
    used = [Path(p).name for p in cfg.get("embeddings_paths") or []]
    rec = {}
    try:
        with open(Path(run_dir) / ALLMOD_RECORD) as f:
            rec = json.load(f)
        if not isinstance(rec, dict) or rec.get("files") != used:
            rec = {}
    except (OSError, json.JSONDecodeError):
        rec = {}
    types = rec.get("modalities") or [
        re.sub(r"^all_genes_|\.pt$", "", u) for u in used]
    return {
        "selection": "none",
        "modalities": list(types),
        "n_modalities": len(used),
        "files": used,
        "catalog_size": rec.get("catalog_size"),
        "absent_optional": rec.get("absent_optional") or [],
        "selection_frac": cfg.get("selection_frac"),
        "selection_seed": cfg.get("selection_seed"),
        "note": "no modality selection: every embedding of the catalog "
                "(catalog order), the same in every fold, CV and variant; the "
                "selection split (selection_frac) is held out of training as "
                "for the selected combo, but chooses nothing",
    }


def label_arm(out, pca_arm):
    """Record the PCA arm a row was collected for (--pca_arm, label only) in
    out["selection_audit"], the key the shared re-grade carries
    (evaluate_model.CARRY_KEYS): out alone says which arm's model it is.
    Nothing without an arm (a call from before the arms) or a row."""
    if out is not None and pca_arm is not None:
        out.setdefault("selection_audit", {})["pca_arm"] = pca_arm


def write_out(out, found, cv_type, outp):
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    # Through <name>.tmp and a rename, like evaluate_model.write_summary: a
    # write cut short must leave no file under the row's name.
    tmp = Path(str(outp) + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump({"summary": out}, f, indent=2)
        tmp.replace(outp)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    msg = f"{out['model']}/{cv_type}: collected {found} -> {outp}"
    if "auprg" in found:
        msg += f"  [AUPRG(any)={out['auprg_mean']:.4f}]"
    print(msg)


def combo_marker_paths(discover_dir, want, cv_types, model_name,
                       allmod=False):
    """{results.json path: cv} of THIS run's combos, one per CV at most.

    Read from <discover_dir>/<prefix>latest_combo_<cv>.txt, which
    run_best_per_category_combo.sh deletes when it starts and writes only after
    that (variant, CV) combo trained. The marker is the only proof a
    best_cat_* directory is this run's: a failed or never-started combo job
    can leave the previous run's directories on disk, and they pass every
    other check (variant, val key, fold stamp). The marker holds the combo's path relative to siamese_sl/
    (results/<prefix>best_cat_<modalities>_<cv>, or
    results/pca_<arm>/<prefix>best_cat_... in another PCA arm); its directory
    NAME is looked up in discover_dir, the arm's root, and must be one of this
    variant's combos for that CV. A CV without such a marker is reported and
    gets no row.

    allmod: the all-modalities combo's markers instead
    (<prefix>latest_allmod_<cv>.txt naming <prefix>all_modalities_<cv>, with
    <prefix> = allmod_dir_prefix; siamese_sl/slurm/run_allmod_combo.sh).
    """
    if want is None:
        raise SystemExit(f"--combo_markers: cannot tell which variant "
                         f"{model_name!r} stands for; pass --variant")
    if allmod:
        prefix, mstem, dstem = (allmod_dir_prefix(want), ALLMOD_MARKER_STEM,
                                ALLMOD_DIR_STEM)
    else:
        prefix, mstem, dstem = (variant_dir_prefix(want), "latest_combo_",
                                COMBO_DIR_STEM)
    found = {}
    for cv in cv_types:
        marker = Path(discover_dir) / f"{prefix}{mstem}{cv}.txt"
        if not marker.is_file():
            print(f"[discover] ERROR: {model_name}/{cv}: no {marker} (this "
                  f"run's {':'.join(want)} combo did not succeed for {cv}); "
                  f"no row is written.")
            continue
        name = Path(marker.read_text().strip()).name
        res = Path(discover_dir) / name / "results.json"
        ok = (name == f"{prefix}{dstem}{cv}" if allmod else
              name.startswith(prefix + dstem) and name.endswith(f"_{cv}"))
        if not ok:
            print(f"[discover] ERROR: {model_name}/{cv}: {marker} names "
                  f"{name!r}, not a {prefix}{dstem}"
                  f"{'' if allmod else '*_'}{cv} combo; no row is written.")
            continue
        if not res.is_file():
            print(f"[discover] ERROR: {model_name}/{cv}: {marker} names {res}, "
                  f"which does not exist; no row is written.")
            continue
        found[res] = cv
    return found


def discover(discover_dir,
             out_dir,
             model_name=BASELINE_MODEL,
             include_prefix=None,
             variant=None,
             combo_markers=False,
             cv_types=CV_TYPES,
             pca_arm=None,
             allmod=False):
    """Scan for siamese results.json files with *_any metrics.

    allmod: the row is the all-modalities combo (siamese_sl/slurm/
    run_allmod_combo.sh): only with combo_markers, from its own markers
    (combo_marker_paths allmod), a candidate must record all_modalities in its
    config, and the row carries allmod_record (no modality selection) as
    selection_audit["all_modalities"] instead of a selection record. Without
    allmod, no all-modalities run is ever a candidate.

    pca_arm: label only (label_arm), the arm of discover_dir's runs.

    combo_markers: consider ONLY the combos this run's latest_combo markers
    name (combo_marker_paths), for the CVs in cv_types. This is how
    collect_and_regrade_siamese.sh builds every row. Without it, every
    results.json under discover_dir competes, however old.

    cv_types: the CVs a row may be written for, with or without
    combo_markers (the caller re-grades only those; a row of any other CV
    would stay self-scored).

    include_prefix: consider only runs whose directory name starts with it
    (a variant row: its own combos, e.g. "var_bilinear_pooled__best_cat_";
    the baseline row: "best_cat_", its combos). The baseline row additionally
    skips every var_* directory, so another variant's combo or ablation can
    never be selected as siamese_sl.

    variant: "KERNEL:CELL_LINES" the row stands for (default: parsed from
    model_name, see expected_variant). A candidate whose recorded config
    holds another kernel / cell_lines is skipped (variant_matches).

    cv type is read from the run (else the path: cv1/cv2/cv3). If several match
    a CV, the best is kept — ranked on val_<select_metric>_mean (the metric the
    run's own checkpoint selection used; 'auprg' for runs predating
    --select_metric), falling back to val_auprg_mean.

    Ranking ~17 candidate configs on a TEST metric is a winner's curse larger
    than the per-epoch one it sits on top of: the reported number is then a
    maximum over 17 noisy test estimates. So a run with no validation score
    (trained without --folds_dir) is skipped, with a warning, never ranked on
    test.
    """
    best = {}  # cv -> (score, summary, path, basis, label_sem, basis_label)
    cfgs = {}  # results.json path -> its train.py config
    candidates = {}  # cv -> [(path, per_fold_val, per_fold_test, val_key)]
    keys_by_cv = {}  # cv -> {validation keys the candidates were ranked on}
    n_val = n_test = 0
    want = expected_variant(model_name, variant)
    n_wrong_variant = 0
    if allmod and not combo_markers:
        raise SystemExit("--allmod needs --combo_markers: an all-modalities "
                         "row is only ever this run's marked model")
    if combo_markers:
        marker_cv = combo_marker_paths(discover_dir, want, cv_types,
                                       model_name, allmod=allmod)
        paths = list(marker_cv)
    else:
        marker_cv = None
        paths = Path(discover_dir).rglob("results.json")
    for path in paths:
        rel_dirs = path.relative_to(discover_dir).parts[:-1]
        if (include_prefix is not None
                and not path.parent.name.startswith(include_prefix)):
            continue
        if want == BASELINE_VARIANT and any(
                p.startswith(VARIANT_DIR_PREFIX) for p in rel_dirs):
            continue
        if (include_prefix is None and want is None
                and any(p.startswith(VARIANT_DIR_PREFIX) for p in rel_dirs)):
            continue
        # Another PCA arm's results root inside this one (only arm none's has
        # them): its first part, never a deeper one (see PCA_ARM_ROOTS).
        if rel_dirs and rel_dirs[0] in PCA_ARM_ROOTS:
            continue
        # The all-modalities combo is its own row's only (allmod).
        if not allmod and any(p.startswith(ALLMOD_DIR_PREFIX)
                              for p in rel_dirs):
            continue
        try:
            data = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary", {})
        if "auroc_any_mean" not in summary:
            continue
        cfg = run_config(data, path)
        if not variant_matches(cfg, want):
            n_wrong_variant += 1
            continue
        if allmod and cfg.get("all_modalities") is not True:
            print(f"[discover] ERROR: {path.parent} does not record "
                  f"all_modalities in its config (not an all-modalities "
                  f"run); no {model_name} row from it.")
            continue
        if not allmod and cfg.get("all_modalities") is True:
            continue
        cfgs[path] = cfg
        # siamese writes cv_type at the TOP level (not inside `summary`); fall
        # back to the path only if neither carries it.
        m = re.search(r"cv[123]", str(path))
        cv = (summary.get("cv_type") or data.get("cv_type")
              or (m.group(0) if m else None))
        if cv not in CV_TYPES:
            continue
        if marker_cv is not None and cv != marker_cv[path]:
            print(f"[discover] ERROR: {path} records cv_type {cv} but was "
                  f"named by the {marker_cv[path]} marker; skipped.")
            continue
        if cv not in cv_types:
            # (after the marker check: its ERROR still names a marker whose
            # run records a CV outside cv_types)
            continue
        val_key, val_score = selection_val_key(summary, cfg)
        if val_key is None:
            # No validation score: the run can only be ranked on TEST, which
            # would make the published row a maximum over test results. Skip
            # it (selection is val-only end to end, like the trainers).
            n_test += 1
            continue
        score, basis = val_score, "val"
        n_val += 1
        keys_by_cv.setdefault(cv, set()).add(val_key)
        # Per-fold arrays for the nested-selection audit below, on the SAME
        # validation key the candidate is ranked on.
        fm = data.get("fold_metrics") or []
        pv = [f.get(val_key) for f in fm] if val_key else []
        pt = [f.get("auprg_any") for f in fm]
        if fm and pv and all(v is not None for v in pv) and \
                all(t is not None for t in pt):
            candidates.setdefault(cv, []).append((path, pv, pt, val_key))
        # Every candidate here carries a validation score; runs selected on
        # different validation metrics are refused below, before anything is
        # written, since their scores are different quantities.
        basis_label = f"{val_key}_mean"
        if cv not in best or score > best[cv][0]:
            best[cv] = (score, summary, path, basis, label_semantics_for(cfg),
                        basis_label)
    for cv, keys in sorted(keys_by_cv.items()):
        if len(keys) > 1 and cv in best:
            print(f"[discover] ERROR: {model_name}/{cv} candidates were "
                  f"selected on different validation metrics {sorted(keys)} "
                  f"(e.g. legacy stratified val_auprg vs pair-level "
                  f"val_auprg_any): their scores are not comparable, so no "
                  f"{model_name}/{cv} row is written. Collect runs trained "
                  f"with one --select_metric, or pin SIAMESE_RESULTS_*.")
            del best[cv]
    if n_wrong_variant:
        print(f"[discover] {model_name}: skipped {n_wrong_variant} run(s) "
              f"whose recorded kernel/cell_lines is not "
              f"{':'.join(want) if want else '?'}")
    # Before the early return: with --combo_markers the only candidate may be
    # the one skipped here, and the log must say why its row is missing.
    if n_test:
        print(f"[discover] WARNING: skipped {n_test} run(s) with no "
              f"validation score (val_<select_metric>_mean / val_auprg_mean): "
              f"selection is val-only, so they are never ranked on TEST. "
              f"Re-run them with train.py --folds_dir.")
    if not best:
        print(f"[discover] no {model_name} *_any results under {discover_dir}"
              + (" named by this run's latest_combo markers"
                 if combo_markers else "")
              + (f" matching {include_prefix}*" if include_prefix else ""))
        return
    # The combo path's only selection step is the modality argmax; record it
    # on every row (see the module docstring, "Selection disclosure").
    selection = load_modality_selection(discover_dir) \
        if combo_markers and not allmod else None
    for cv, (score, summary, path, basis, label_sem,
             basis_label) in sorted(best.items()):
        out, found = convert(summary, cv, model_name, label_sem)
        if out is not None and selection is not None:
            why = []
            rec = modality_selection_record(selection, cfgs.get(path, {}),
                                            path.parent, why)
            if rec is None:
                reason = (why[0] if why else "trained on embeddings that "
                          "are not the winners")
                print(f"[discover] WARNING: {model_name}/{cv}: {path.parent} "
                      f"{reason}: {Path(discover_dir) / SELECTION_JSON} is "
                      f"not this combo's selection; the row carries no "
                      f"selection record.")
            else:
                out["selection_audit"] = {"modality_selection": rec}
        if out is not None and allmod:
            out["selection_audit"] = {
                "all_modalities": allmod_record(path.parent,
                                                cfgs.get(path, {}))}
        cands = candidates.get(cv, [])
        val_keys = sorted({c[3] for c in cands})
        # Mixed validation keys were refused above (del best[cv]), and every
        # candidate's key is in keys_by_cv, so at most one is left here.
        assert len(val_keys) <= 1, (cv, val_keys)
        audit = nested_loo_audit([c[:3] for c in cands],
                                 val_keys[0] if val_keys else "val_auprg")
        if out is not None and audit:
            audit.update(out.get("selection_audit") or {})
            out["selection_audit"] = audit
            rep = out.get("auprg_mean")
            if rep is not None:
                # `delta` is nested - reported: the audit's shift of the
                # reported number, which compare.py applies to the table value
                # and the parenthetical below prints. A partial adjustment
                # only (see nested_loo_audit): the true optimism can be larger.
                # `optimism` is the opposite sign by convention (Efron): how
                # much the reported score EXCEEDS the nested estimate,
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
        label_arm(out, pca_arm)
        print(f"[discover] {model_name}/{cv} <- {path}  [selected on "
              f"{basis_label}, score={score:.4f}]")
        write_out(out, found, cv, Path(out_dir) / f"{model_name}_{cv}.json")
        # Record which run won so collect_and_regrade_siamese.sh can re-grade its
        # prediction matrices through evaluate_model.py — the same evaluator,
        # test edges and gene universe as the other three models. Without this
        # the table mixes siamese's own scoring with the shared one.
        marker = Path(out_dir) / marker_name(model_name, cv)
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
                    help="Where auto mode writes <model_name>_<cv>.json")
    ap.add_argument("--model_name",
                    default=BASELINE_MODEL,
                    help="Row name in the comparison table: siamese_sl for "
                    "the baseline variant, siamese_sl_<kernel>_<cell_lines> "
                    "for the others")
    ap.add_argument("--include_prefix",
                    help="Auto mode: only runs whose directory name starts "
                    "with this (a variant's combos, e.g. "
                    "var_bilinear_pooled__best_cat_; the baseline's, "
                    "best_cat_). The baseline row skips every var_* "
                    "directory either way")
    ap.add_argument("--variant",
                    help="KERNEL:CELL_LINES this row stands for; candidates "
                    "whose recorded config differs are skipped. Default: "
                    "parsed from --model_name (siamese_sl = pd:per_head, "
                    "siamese_sl_<kernel>_<cell_lines>)")
    ap.add_argument("--combo_markers",
                    action="store_true",
                    help="Auto mode: take, per CV, ONLY the combo named by "
                    "this run's <discover_dir>/<prefix>latest_combo_<cv>.txt "
                    "(the combo job deletes them at start, writes them on "
                    "success); a CV without one gets no row. Without it, "
                    "every matching results.json competes, however old")
    ap.add_argument("--allmod",
                    action="store_true",
                    help="Auto mode with --combo_markers: the all-modalities "
                    "combo's row (siamese_sl/slurm/run_allmod_combo.sh), from "
                    "<discover_dir>/allmod__<prefix>latest_allmod_<cv>.txt; "
                    "--model_name siamese_sl_allmod[_<kernel>_<cell_lines>]")
    ap.add_argument("--cv_types",
                    nargs="+",
                    choices=list(CV_TYPES),
                    default=list(CV_TYPES),
                    help="the CVs to write a row for (with --combo_markers: "
                         "the CVs whose markers to read)")
    ap.add_argument("--pca_arm",
                    choices=["none", "joint", "per_modality", "both"],
                    help="label only: the PCA arm the row is collected for "
                    "(siamese_sl/slurm/config.conf PCA arms), recorded as "
                    "selection_audit.pca_arm (the re-grade carries it)")
    args = ap.parse_args()
    if args.variant:
        try:
            parse_variant(args.variant)
        except ValueError as e:
            ap.error(str(e))

    if args.discover_dir:
        discover(args.discover_dir, args.out_dir, args.model_name,
                 args.include_prefix, args.variant,
                 combo_markers=args.combo_markers,
                 cv_types=args.cv_types, pca_arm=args.pca_arm,
                 allmod=args.allmod)
        return
    if not (args.siamese_results and args.cv_type and args.out):
        ap.error("provide --discover_dir, OR all of "
                 "--siamese_results/--cv_type/--out")
    data = json.load(open(args.siamese_results))
    summary = data.get("summary", {})
    want = expected_variant(args.model_name, args.variant)
    cfg = run_config(data, args.siamese_results)
    if not variant_matches(cfg, want):
        print(f"WARNING: pinned {args.siamese_results} records kernel="
              f"{cfg.get('kernel')!r} cell_lines={cfg.get('cell_lines')!r}, "
              f"not the {args.model_name} variant {':'.join(want)}; "
              f"collecting it anyway because it was pinned explicitly.")
    out, found = convert(summary, args.cv_type, args.model_name,
                         label_semantics_for(cfg))
    if not found:
        raise SystemExit(
            f"No *_any metrics in {args.siamese_results} (expected "
            f"auroc_any_mean etc.). Was this a cell-line run?")
    label_arm(out, args.pca_arm)
    write_out(out, found, args.cv_type, args.out)


if __name__ == "__main__":
    main()
