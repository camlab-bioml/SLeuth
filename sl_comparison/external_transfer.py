#!/usr/bin/env python3
"""
External-screen transfer table: every model scored on IDENTICAL screen rows.

Zero-shot table (all models)
----------------------------
For each external screen (Adamson, AdamsonGI, Corn, Gilbert) and each screen split
(cv1 random pairs / cv2 >=1 gene unseen / cv3 both genes unseen), the row set
is taken from SLeuth's own eval_finetuning output
    <model_dir>/eval_finetuning_<split>/<Screen>_predictions.csv
and restricted to

    split == 'test'  AND  not sldb_overlap  AND  both genes in genes.txt

("matched" rows). genes.txt is the shared 8,844-gene index the baselines'
dense score matrices are aligned to, so every model can score every matched
row and no model is evaluated on a row another model cannot see.

Per model and training fold k, the metric is Spearman(score_k, target) on the
matched rows, where target = -1 x screen phenotype (higher = more SL):

  SLeuth       score_k = logit_fold<k>, the RAW (uncalibrated) logit of
               whichever head eval_finetuning scored with (see config_used.json
               'effective_head'; older evals only record 'per_head' /
               'dataset_heads').
  baselines    score_k = (M_k[i, j] + M_k[j, i]) / 2,
               M_k = <pred_dir>/fold_<k>_<model_cv>_predictions.npy indexed
               through genes.txt (symmetrised: a screen pair is unordered).
  DegreePrior  score_k = log1p(deg_i) + log1p(deg_j), degrees counted over the
               <model_cv> fold-k TRAINING positives (train_pos.npy), summed in
               float64 and cast to float32 exactly as
               sl_comparison/degree_baseline.py builds its matrix (so ties, and
               hence Spearman, match that row).

Guards (each raises): a non-finite score or target on a REPORTED row (matched,
and the all_sldb_filtered rows the paper's SLeuth column is on; Spearman would
otherwise drop it silently, and the "identical rows" guarantee with it, while
the recorded n_rows still claims the full row set);
a baseline or DegreePrior fold set that differs from the SLeuth logit_fold<k>
set; a SLeuth model dir whose config.json cv_type is not --model_cv; a missing
default baseline (unless --allow_missing_baselines); an eval_finetuning_<split>
whose config_used.json is missing, matched sldb_overlap against fewer than two
sources (positives and screened negatives), records no checkpoint_sha256 /
fold_ids, or whose recorded checkpoint hashes differ from the current
<model_dir>/checkpoints/fold_*_best.pt (a retrain since the eval; unless
--allow_legacy_eval).

Reported: mean and POPULATION sd (ddof=0) over folds, the row count and every
per-fold value. (eval_finetuning's own summaries use the sample sd, ddof=1.)

Secondary figures for SLeuth only: the same raw-logit Spearman on ALL
SLDB-filtered test rows (no genes.txt restriction; the row set of the paper's
fine-tuning table, and of the SLeuth column the current draft prints in its
zero-shot table, which is therefore NOT on the baselines' rows: a like-for-like
cross-model column uses the matched rows) and on all test rows (unfiltered).

Every SLeuth variant must produce the same matched pair set for a given
(screen, split); otherwise the script stops. The screen split is seeded
identically, but eval_finetuning shuffles the model's embedding row indices
of the screen genes, so two variants trained on different gene universes (a
different modality set) can land on different splits.

Fine-tuning table (SLeuth variants only)
----------------------------------------
From <Screen>_finetuned_predictions_{LP_last,Full[,Scratch]}.csv (test rows only,
pred_fold0..4; its all-pair twin <Screen>_finetuned_allpairs_<mode>.csv,
<Screen>_sl_probabilities.csv and finetuned_models/ in the same directory are
not read here), the per-fold Spearman on the matched rows and on the model's
own full test rows (SLDB-filtered and unfiltered), next to two zero-shot
references on the same rows:

  zero_shot_raw  the raw logit (the number in the zero-shot table).
  zero_shot_ols  a_k * logit + c_k, (a_k, c_k) the closed-form OLS fit on the
                 screen's train split (all train rows) - eval_finetuning's
                 'zero-shot (calibrated)' row. Differs from raw only in sign
                 (sign of a_k).

LP_last and Full are warm-started from that OLS-calibrated head, so compare
them with zero_shot_ols. Scratch (opt-in, when eval_finetuning ran it) is the
from-scratch baseline: Full's schedule on a freshly initialised model, its
head warm-started on its own logits (starts_from_ols_head False); compare it
with Full. Where a_k < 0 (Adamson) calibration alone flips the
sign of the raw Spearman.

Outputs (--out_dir)
-------------------
  row_sets.csv                 per (screen, split): row counts + matched-set digest
  zero_shot_per_fold.csv       long: model x screen x split x row_set x fold
  zero_shot_summary.csv        mean, pop sd, n_rows, per-fold columns
  finetune_per_fold.csv        long: variant x screen x split x row_set x method x fold
  finetune_summary.csv         mean, pop sd, n_rows, per-fold columns
  sldb_overlap.csv             per screen x scope (all_pairs, all_pairs_genes_txt,
                               test_genes_txt per split): n / share of pairs that
                               are SynLethDB SL, screened non-SL, either, and in
                               each shared fold's TRAINING pairs; screen_role
  sldb_agreement.csv           same scopes: AUROC of target for SynLethDB SL vs
                               screened non-SL, Spearman(label, target)
  summary.md                   readable tables + provenance

In-distribution reframing (2026-11-12). Every model is also scored on
matched_incl_sldb (the matched genes' test rows WITH the SynLethDB pairs) and
matched_in_sldb (the SynLethDB pairs alone), next to the held-out `matched`;
on matched_in_sldb the SynLethDB label itself (1 = SL, 0 = screened non-SL) is
scored as model SynLethDB_label (kind sldb_label, one value under every fold
id). screen_role (row_sets.csv, zero_shot_summary.csv, the two sldb_*.csv) is
in_distribution_check for a screen >= IN_DISTRIBUTION_MIN_SLDB_SHARE of whose
scored pairs SynLethDB lists (Gilbert: Horlbeck et al. 2018 is a SynLethDB
source study), external otherwise. The fine-tuning table keeps its three row
sets. Bootstrap CIs: bootstrap_ci.py --part external (ci_external.csv for
every zero-shot row set, ci_sldb_agreement.csv for the agreement table).

With --pca_arm (label only, 2026-10-21: run_external_transfer.sh passes the
PCA arm of the SLeuth combos, siamese_sl/slurm/config.conf PCA arms) every CSV
gets a trailing pca_arm column and summary.md a 'PCA arm: <arm> (<label>)'
line under its title (and a Provenance bullet): arm none's tables sit at the
pre-arms path, and a copy taken out of its directory says nothing otherwise.
Without it the output is byte-identical to before.

Usage (from the repo root):
  python sl_comparison/external_transfer.py \\
      --siamese SLeuth=siamese_sl/results/<combo>_cv3 \\
      --out_dir /tmp/ext_run
  # (a missing default baseline stops the run; --allow_missing_baselines
  #  skips it instead; --no_baselines scores SLeuth + DegreePrior only)
  # several SLeuth variants, explicit baselines, no degree prior:
  python sl_comparison/external_transfer.py \\
      --siamese pd_per_head=DIR_A --siamese bilinear_pooled=DIR_B \\
      --baseline SLMGAE=SLMGAE-in-pytorch/results/slmgae_shared_cv3 \\
      --no-degree_prior --out_dir /tmp/ext_run
"""

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]

# AdamsonGI: the Adamson screen scored as a genetic interaction (Simpson et al.
# 2023 Table S3 Gamma.GI; siamese_sl/eval_finetuning._load_adamson_gi), next to
# its growth phenotype (Adamson, Gamma).
SCREENS: Tuple[str, ...] = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
SPLITS: Tuple[str, ...] = ("cv1", "cv2", "cv3")
SPLIT_LABELS: Dict[str, str] = {
    "cv1": "random pairs",
    "cv2": ">=1 gene unseen",
    "cv3": "both genes unseen",
}
FT_MODES: Tuple[str, ...] = ("LP_last", "Full", "Scratch")
# The modes warm-started from the pretrained logit's OLS head. Scratch
# (eval_finetuning's from-scratch baseline, opt-in) is not: it starts from a
# fresh init and an OLS fit on that model's own logits. A mode whose CSV is
# absent is skipped, so without Scratch every output is as before.
OLS_WARM_MODES: Tuple[str, ...] = ("LP_last", "Full")
SCRATCH_MODE = "Scratch"

ROW_MATCHED = "matched"
ROW_ALL_SLDB = "all_sldb_filtered"
ROW_ALL = "all_unfiltered"
# The two matched-gene row sets that keep the SynLethDB pairs (2026-11-12): a
# screen most of whose pairs SynLethDB lists (Gilbert, a SynLethDB source study)
# is an in-distribution check, so every model is also scored on ALL its matched
# test pairs and on the SynLethDB ones alone, next to the held-out `matched`.
ROW_MATCHED_INCL = "matched_incl_sldb"
ROW_MATCHED_IN_SLDB = "matched_in_sldb"
ROW_SETS: Tuple[str, ...] = (ROW_MATCHED, ROW_ALL_SLDB, ROW_ALL)
# Zero-shot rows: ROW_SETS plus the two above (the fine-tuning table keeps
# ROW_SETS). Every model, matrix models included, is scored on MATCHED_ROW_SETS.
ZS_ROW_SETS: Tuple[str, ...] = ROW_SETS + (ROW_MATCHED_INCL,
                                           ROW_MATCHED_IN_SLDB)
MATCHED_ROW_SETS: Tuple[str, ...] = (ROW_MATCHED, ROW_MATCHED_INCL,
                                     ROW_MATCHED_IN_SLDB)
ROW_SET_LABELS: Dict[str, str] = {
    ROW_MATCHED: "test, not in SynLethDB (SL or screened non-SL), both "
    "genes in genes.txt",
    ROW_ALL_SLDB: "test, not in SynLethDB (SL or screened non-SL; no "
    "genes.txt restriction)",
    ROW_ALL: "all test rows",
    ROW_MATCHED_INCL: "test, both genes in genes.txt, SynLethDB pairs kept",
    ROW_MATCHED_IN_SLDB: "test, both genes in genes.txt, in SynLethDB (SL or "
    "screened non-SL) only",
}
# The SynLethDB label scored as a predictor (1 = SynLethDB SL, 0 = screened
# non-SL and not SL: the shared folds' OR-collapse) on the matched_in_sldb rows,
# the only rows where it is defined; the same value under every fold id.
SLDB_LABEL_NAME = "SynLethDB_label"
SLDB_LABEL_KIND = "sldb_label"
# A screen whose pairs SynLethDB lists at least this share of (SL or screened
# non-SL, of every scored screen pair) is labelled an in-distribution check
# (screen_role), not an external test: its pairs largely ARE training data.
IN_DISTRIBUTION_MIN_SLDB_SHARE = 0.5
ROLE_IN_DISTRIBUTION = "in_distribution_check"
ROLE_EXTERNAL = "external"

# (name, path template relative to the repo root); {cv} = --model_cv.
DEFAULT_BASELINES: Tuple[Tuple[str, str], ...] = (
    ("SLMGAE", "SLMGAE-in-pytorch/results/slmgae_shared_{cv}"),
    ("SLGNN", "SLGNN/results/slgnn_{cv}"),
    ("NSF4SL", "NSF4SL/results/nsf4sl_{cv}"),
)
DEGREE_PRIOR_NAME = "DegreePrior"

# f(i, j) -> scores for gene-index arrays i, j (both shape (n,)); one fold.
PairScorer = Callable[[np.ndarray, np.ndarray], np.ndarray]

_LOGIT_COL = re.compile(r"^logit_fold(\d+)$")
_PRED_COL = re.compile(r"^pred_fold(\d+)$")
_BASELINE_FILE = re.compile(r"^fold_(\d+)_(cv\d)_predictions\.npy$")


# =============================================================================
# Parsing / loading
# =============================================================================


def parse_named_paths(specs: Sequence[str], flag: str) -> List[Tuple[str, Path]]:
    """Parse repeated NAME=PATH arguments; names must be unique."""
    out: List[Tuple[str, Path]] = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"{flag} expects NAME=PATH, got {spec!r}")
        out.append((name.strip(), Path(path.strip())))
    names = [n for n, _ in out]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{flag}: duplicate names {dupes}")
    return out


def load_gene_index(genes_path: Path) -> Dict[str, int]:
    """Entrez ID string -> row/column index of the shared score matrices."""
    genes = [ln.strip() for ln in open(genes_path) if ln.strip()]
    if len(set(genes)) != len(genes):
        raise ValueError(f"{genes_path} contains duplicate gene IDs")
    return {g: i for i, g in enumerate(genes)}


def _as_bool(s: pd.Series) -> np.ndarray:
    """sldb_overlap as a bool array whether pandas parsed it or not."""
    if s.dtype == bool:
        return s.to_numpy()
    mapped = s.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False,
    })
    if mapped.isna().any():
        bad = s[mapped.isna()].unique()[:5]
        raise ValueError(f"unparseable sldb_overlap values: {list(bad)}")
    return mapped.to_numpy(dtype=bool)


def pair_keys(g1: pd.Series, g2: pd.Series) -> np.ndarray:
    """Order-free pair key 'lo|hi' (string order) for Entrez ID columns."""
    a = g1.astype(str).str.strip().to_numpy()
    b = g2.astype(str).str.strip().to_numpy()
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    return np.char.add(np.char.add(lo.astype(str), "|"), hi.astype(str))


def fold_columns(df: pd.DataFrame, pattern: "re.Pattern") -> List[Tuple[int, str]]:
    """(fold index, column) for columns matching `pattern`, sorted by fold."""
    cols = [(int(m.group(1)), c) for c in df.columns
            for m in [pattern.match(c)] if m]
    if not cols:
        raise ValueError(f"no columns matching {pattern.pattern}")
    return sorted(cols)


def read_pair_csv(path: Path) -> pd.DataFrame:
    """Read an eval_finetuning pair CSV; adds `key` and a bool sldb_overlap.

    Entrez IDs are read as strings so they compare equal to genes.txt lines.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"gene1_entrez": str, "gene2_entrez": str})
    missing = {"gene1_entrez", "gene2_entrez", "target", "sldb_overlap"
               } - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["sldb_overlap"] = _as_bool(df["sldb_overlap"])
    df["key"] = pair_keys(df["gene1_entrez"], df["gene2_entrez"])
    return df


def row_mask(df: pd.DataFrame, gene_index: Dict[str, int],
             row_set: str) -> np.ndarray:
    """Boolean mask of the rows in `row_set`.

    A frame without a `split` column (the fine-tuned CSVs) holds test rows
    only, so every row counts as test.
    """
    if "split" in df.columns:
        m = (df["split"].astype(str) == "test").to_numpy()
    else:
        m = np.ones(len(df), dtype=bool)
    if row_set == ROW_ALL:
        return m
    ov = df["sldb_overlap"].to_numpy(dtype=bool)
    if row_set == ROW_ALL_SLDB:
        return m & ~ov
    if row_set in MATCHED_ROW_SETS:
        in1 = df["gene1_entrez"].astype(str).str.strip().isin(gene_index)
        in2 = df["gene2_entrez"].astype(str).str.strip().isin(gene_index)
        m = m & in1.to_numpy() & in2.to_numpy()
        if row_set == ROW_MATCHED:
            return m & ~ov
        if row_set == ROW_MATCHED_IN_SLDB:
            return m & ov
        return m
    raise ValueError(f"unknown row_set {row_set!r}")


def read_eval_config(model_dir: Path, split: str) -> dict:
    """eval_finetuning's config_used.json for one split ({} if absent)."""
    p = model_dir / f"eval_finetuning_{split}" / "config_used.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_hashes(model_dir: Path) -> Dict[str, str]:
    """{basename: sha256} of <model_dir>/checkpoints/fold_*_best.pt."""
    return {p.name: sha256_file(p)
            for p in sorted((model_dir / "checkpoints").glob("fold_*_best.pt"))}


def check_eval_fingerprint(model_dir: Path, split: str,
                           current: Dict[str, str],
                           allow_legacy: bool = False) -> Optional[str]:
    """Refuse an eval_finetuning_<split> that cannot be tied to THESE weights.

    A retrain into the same model dir rewrites config.json, model_params.json
    and checkpoints/ but leaves eval_finetuning_*/ in place, so the model-level
    checks pass while the CSVs still hold the previous weights' scores.
    eval_finetuning.py records checkpoint_sha256 (keyed by path, matched here
    by basename: it runs from siamese_sl/, this script from the repo root) and
    fold_ids; `current` is checkpoint_hashes(model_dir). Raises ValueError, or
    with allow_legacy prints a WARNING and returns it.
    """
    where = model_dir / f"eval_finetuning_{split}" / "config_used.json"
    cfg = read_eval_config(model_dir, split)
    problems: List[str] = []
    if not cfg:
        problems.append(
            f"{where} missing: cannot confirm that sldb_overlap also flagged "
            f"the screened negatives, or which checkpoints scored the CSVs; "
            f"rerun eval_finetuning.py")
    else:
        # The filtered row sets are only "leakage-filtered" if sldb_overlap
        # also flagged the screened negatives the fold models trained on
        # (eval_finetuning --neg_path).
        if len(cfg.get("sldb_overlap_sources") or []) < 2:
            problems.append(
                f"{model_dir}/eval_finetuning_{split}: sldb_overlap was "
                f"matched against SynLethDB positives only; rerun "
                f"eval_finetuning.py with --neg_path so the filtered rows "
                f"exclude the screened negatives too")
        rec = {Path(k).name: v
               for k, v in (cfg.get("checkpoint_sha256") or {}).items()}
        if not rec:
            problems.append(f"{where} records no checkpoint_sha256 (older "
                            f"eval); rerun eval_finetuning.py")
        elif set(rec) != set(current):
            problems.append(
                f"{where}: scored checkpoints {sorted(rec)} but "
                f"{model_dir}/checkpoints holds {sorted(current)}")
        else:
            changed = sorted(n for n in rec if rec[n] != current[n])
            if changed:
                problems.append(
                    f"{where}: sha256 of {changed} differs from the scored "
                    f"checkpoints: retrained since this eval; rerun "
                    f"eval_finetuning.py")
        fold_ids = cfg.get("fold_ids")
        if fold_ids is None:
            problems.append(f"{where} records no fold_ids (older eval)")
        elif list(fold_ids) != list(range(len(current))):
            problems.append(f"{where}: fold_ids {list(fold_ids)} are not "
                            f"0..{len(current) - 1}")
    if not problems:
        return None
    msg = "; ".join(problems)
    if not allow_legacy:
        raise ValueError(msg + " (or pass --allow_legacy_eval)")
    msg = f"WARNING (--allow_legacy_eval): {msg}"
    print(f"  {msg}")
    return msg


# =============================================================================
# Metrics
# =============================================================================


def spearman(score: np.ndarray, target: np.ndarray) -> float:
    """Spearman rho; NaN when either side is constant or n < 3."""
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if score.shape != target.shape:
        raise ValueError(f"shape mismatch {score.shape} vs {target.shape}")
    ok = np.isfinite(score) & np.isfinite(target)
    score, target = score[ok], target[ok]
    if len(score) < 3 or np.ptp(score) == 0 or np.ptp(target) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(score, target)[0])


# Row sets a table reports, hence the ones a non-finite value must not reach:
# `matched` is the cross-model column, `all_sldb_filtered` is the SLeuth column
# the paper prints (paper_tables tab_external / tab_external_ft
# paper_convention, numbers.json F2_F3_all_sldb_filtered). ROW_ALL is a
# summary.md-only diagnostic no table reads, so it stays tolerant.
GUARDED_ROW_SETS: Tuple[str, ...] = (ROW_MATCHED, ROW_ALL_SLDB,
                                     ROW_MATCHED_INCL, ROW_MATCHED_IN_SLDB)


def require_finite(values: np.ndarray, what: str,
                   rows: str = ROW_MATCHED) -> None:
    """Raise if any value is NaN/inf. Used on every REPORTED row set, where
    spearman()'s silent drop would score folds/models on different row subsets
    while the recorded n_rows still counts the whole set."""
    v = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} non-finite value(s) among "
                         f"{len(v):,} {rows} rows")


def aggregate_folds(values: Sequence[float]) -> Tuple[float, float, int]:
    """(mean, population sd with ddof=0, n finite folds) over per-fold values.

    NaN folds (constant score) are dropped and show up as n < #folds.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=0)), int(len(v))


def fit_ols_affine(logit: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Closed-form OLS y = a*logit + c (mirror of eval_finetuning.fit_ols_affine)."""
    l = np.asarray(logit, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    lm, ym = l.mean(), y.mean()
    denom = ((l - lm)**2).sum()
    if denom < 1e-12:
        return 1.0, float(ym - lm)
    a = float(((l - lm) * (y - ym)).sum() / denom)
    return a, float(ym - a * lm)


def summarize(long_rows: List[dict], key_cols: List[str]) -> pd.DataFrame:
    """Collapse long per-fold rows into mean / pop-sd / per-fold columns."""
    if not long_rows:
        return pd.DataFrame()
    df = pd.DataFrame(long_rows)
    max_fold = int(df["fold"].max())
    out = []
    for key, grp in df.groupby(key_cols, sort=False):
        key = key if isinstance(key, tuple) else (key, )
        grp = grp.sort_values("fold")
        mean, sd, n_ok = aggregate_folds(grp["spearman"].to_numpy())
        rec = dict(zip(key_cols, key))
        rec.update({
            "n_rows": int(grp["n_rows"].iloc[0]),
            "n_folds": int(len(grp)),
            "n_folds_finite": n_ok,
            "spearman_mean": mean,
            "spearman_sd_pop": sd,
        })
        per_fold = dict(zip(grp["fold"].astype(int), grp["spearman"]))
        for k in range(max_fold + 1):
            rec[f"spearman_fold{k}"] = per_fold.get(k, float("nan"))
        out.append(rec)
    return pd.DataFrame(out)


# =============================================================================
# Row sets (defined by the SLeuth CSVs)
# =============================================================================


def check_same_pairs(ref: pd.DataFrame, other: pd.DataFrame, ref_name: str,
                     other_name: str, where: str) -> None:
    """Raise unless `other` holds exactly ref's pair keys with equal targets."""
    a, b = set(ref["key"]), set(other["key"])
    if a != b:
        raise ValueError(
            f"{where}: SLeuth variants disagree on the matched row set — "
            f"{ref_name} has {len(a):,} pairs, {other_name} has {len(b):,} "
            f"({len(a - b):,} only in {ref_name}, {len(b - a):,} only in "
            f"{other_name}). The screen splits differ (the split shuffles each "
            f"model's own gene indices), so the variants cannot be compared "
            f"on identical rows.")
    t_ref = ref.set_index("key")["target"]
    t_oth = other.set_index("key")["target"].reindex(t_ref.index)
    if not np.allclose(t_ref.to_numpy(), t_oth.to_numpy(), rtol=1e-6,
                       atol=1e-9, equal_nan=True):
        raise ValueError(f"{where}: {ref_name} and {other_name} have the same "
                         f"pairs but different targets")


def matched_digest(keys: Sequence[str]) -> str:
    """Short content hash of a pair-key set (order-free)."""
    h = hashlib.sha1("\n".join(sorted(keys)).encode()).hexdigest()
    return h[:16]


def score_siamese_variant(
    name: str,
    model_dir: Path,
    screens: Sequence[str],
    splits: Sequence[str],
    gene_index: Dict[str, int],
    reference: Dict[Tuple[str, str], pd.DataFrame],
    ref_owner: Dict[Tuple[str, str], str],
    zs_rows: List[dict],
    ft_rows: List[dict],
    rowset_rows: List[dict],
    ft_modes: Sequence[str] = FT_MODES,
    fold_sets: Optional[Dict[str, Tuple[int, ...]]] = None,
    eval_fold_ids: Optional[Dict[str, Sequence[int]]] = None,
    screen_pairs: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Zero-shot + fine-tune per-fold Spearman for one SLeuth variant.

    The first variant to see a (screen, split) fills `reference` with its
    matched_incl_sldb rows (key, gene IDs, target, sldb_overlap: `matched` is
    its rows without sldb_overlap); later variants are checked against it.
    `screen_pairs` (screen -> every row of the first <Screen>_predictions.csv
    read, any split) feeds the SynLethDB overlap / agreement tables. Appends long-form records to zs_rows / ft_rows / rowset_rows, and
    records each CSV's logit_fold<k> set in `fold_sets` (run() requires one
    set everywhere). Raises on a non-finite target or logit on a GUARDED_ROW_SETS
    row (matched, and the all_sldb_filtered rows the paper's SLeuth column is
    on: spearman() would drop it silently and the fold's mean would be over
    fewer rows than the n_rows recorded beside it), and when a split's
    logit_fold<k> columns are not the fold_ids its config_used.json records
    (`eval_fold_ids`, keyed by split).
    """
    for screen in screens:
        for split in splits:
            where = f"{screen}/{split}"
            path = (model_dir / f"eval_finetuning_{split}" /
                    f"{screen}_predictions.csv")
            df = read_pair_csv(path)
            if "split" not in df.columns:
                raise ValueError(f"{path}: missing column 'split'")
            logit_cols = fold_columns(df, _LOGIT_COL)
            want = (eval_fold_ids or {}).get(split)
            if want is not None and \
                    tuple(k for k, _ in logit_cols) != tuple(want):
                raise ValueError(
                    f"{path}: logit_fold columns "
                    f"{[k for k, _ in logit_cols]} != config_used.json "
                    f"fold_ids {list(want)}")
            if fold_sets is not None:
                fold_sets[f"{name} {where}"] = tuple(k for k, _ in logit_cols)
            masks = {rs: row_mask(df, gene_index, rs) for rs in ZS_ROW_SETS}
            if screen_pairs is not None and screen not in screen_pairs:
                screen_pairs[screen] = df[["key", "gene1_entrez",
                                           "gene2_entrez", "target",
                                           "sldb_overlap"]].copy()
            for rs in GUARDED_ROW_SETS:
                mm = masks[rs]
                require_finite(df["target"].to_numpy(dtype=np.float64)[mm],
                               f"{path}: target", rs)
                for _k, col in logit_cols:
                    require_finite(df[col].to_numpy(dtype=np.float64)[mm],
                                   f"{path}: {col}", rs)

            matched = df.loc[masks[ROW_MATCHED_INCL],
                             ["key", "gene1_entrez", "gene2_entrez", "target",
                              "sldb_overlap"]]
            if matched["key"].duplicated().any():
                raise ValueError(f"{path}: duplicate pairs among test rows")
            matched = matched.sort_values("key").reset_index(drop=True)
            if (screen, split) not in reference:
                reference[(screen, split)] = matched
                ref_owner[(screen, split)] = name
                held_out = matched.loc[~matched["sldb_overlap"], "key"]
                rowset_rows.append({
                    "screen": screen,
                    "split": split,
                    "split_label": SPLIT_LABELS.get(split, split),
                    "n_screen_pairs": int(len(df)),
                    "n_test": int(masks[ROW_ALL].sum()),
                    "n_test_not_sldb": int(masks[ROW_ALL_SLDB].sum()),
                    "n_matched": int(len(held_out)),
                    "matched_digest": matched_digest(held_out),
                    "defined_by": name,
                    "n_matched_incl_sldb": int(len(matched)),
                    "n_matched_in_sldb": int(matched["sldb_overlap"].sum()),
                })
            else:
                check_same_pairs(reference[(screen, split)], matched,
                                 ref_owner[(screen, split)], name, where)

            target = df["target"].to_numpy(dtype=np.float64)
            train = (df["split"].astype(str) == "train").to_numpy()

            # ---- zero-shot: raw logits, every zero-shot row set ----
            for rs in ZS_ROW_SETS:
                m = masks[rs]
                for k, col in logit_cols:
                    zs_rows.append({
                        "model": name,
                        "kind": "siamese",
                        "screen": screen,
                        "split": split,
                        "row_set": rs,
                        "fold": k,
                        "n_rows": int(m.sum()),
                        "spearman": spearman(df[col].to_numpy()[m],
                                             target[m]),
                    })

            # ---- fine-tune table ----
            ols = {}
            for k, col in logit_cols:
                if train.sum() < 2:
                    raise ValueError(f"{path}: no train rows for OLS")
                ols[k] = fit_ols_affine(df[col].to_numpy()[train],
                                        target[train])
            test_keys = set(df.loc[masks[ROW_ALL], "key"])
            ft_frames = {}
            fdf_paths = {}
            # Only the modes the LAST eval_finetuning run in this directory
            # produced (its config_used.json): an older run's CSV for a mode
            # the last run skipped came from other checkpoints.
            cfg_used = path.parent / "config_used.json"
            ran_modes = (json.load(open(cfg_used)).get("ft_modes")
                         if cfg_used.exists() else None)
            for mode in ft_modes:
                fpath = (model_dir / f"eval_finetuning_{split}" /
                         f"{screen}_finetuned_predictions_{mode}.csv")
                if not fpath.exists():
                    # Scratch is opt-in: say nothing unless the eval ran it.
                    if mode != SCRATCH_MODE or (ran_modes is not None and
                                                mode in ran_modes):
                        print(f"  [{name}] {where}: no {fpath.name}; "
                              f"skipping {mode}")
                    continue
                if ran_modes is not None and mode not in ran_modes:
                    raise ValueError(
                        f"{fpath}: mode {mode} is not in {cfg_used}'s ft_modes "
                        f"{ran_modes}; the file is left over from an earlier "
                        f"run. Delete it or rerun eval_finetuning with it.")
                fdf = read_pair_csv(fpath)
                if set(fdf["key"]) != test_keys or len(fdf) != len(test_keys):
                    raise ValueError(
                        f"{fpath}: rows are not the test rows of {path.name}")
                # Same outcome: the target of every pair must equal the
                # zero-shot file's (a stale file with the other sign
                # convention, or another phenotype column, would pass the
                # key check and silently flip before/after comparisons).
                zt = df.loc[masks[ROW_ALL]].set_index("key")["target"]
                ft = fdf.set_index("key")["target"].reindex(zt.index)
                if not np.allclose(ft.to_numpy(dtype=np.float64),
                                   zt.to_numpy(dtype=np.float64),
                                   rtol=1e-6, atol=1e-9, equal_nan=False):
                    raise ValueError(
                        f"{fpath}: target differs from {path.name} on the "
                        f"same pairs (stale file or other phenotype/sign)")
                # Same folds: one pred_fold<k> per zero-shot logit_fold<k>.
                ft_folds = sorted(k for k, _ in fold_columns(fdf, _PRED_COL))
                zs_folds = sorted(k for k, _ in logit_cols)
                if ft_folds != zs_folds:
                    raise ValueError(
                        f"{fpath}: pred_fold columns {ft_folds} != "
                        f"{path.name} logit_fold columns {zs_folds}")
                ft_frames[mode] = fdf
                fdf_paths[mode] = fpath

            for rs in ROW_SETS:
                m = masks[rs]
                n = int(m.sum())
                sub_t = target[m]
                for k, col in logit_cols:
                    raw = df[col].to_numpy(dtype=np.float64)[m]
                    a, c = ols[k]
                    base = {
                        "model": name,
                        "screen": screen,
                        "split": split,
                        "row_set": rs,
                        "fold": k,
                        "n_rows": n,
                    }
                    ft_rows.append({**base, "method": "zero_shot_raw",
                                    "ols_a": float("nan"),
                                    "spearman": spearman(raw, sub_t)})
                    ft_rows.append({**base, "method": "zero_shot_ols",
                                    "ols_a": a,
                                    "spearman": spearman(a * raw + c, sub_t)})
                for mode, fdf in ft_frames.items():
                    fm = row_mask(fdf, gene_index, rs)
                    if int(fm.sum()) != n or set(
                            fdf.loc[fm, "key"]) != set(df.loc[m, "key"]):
                        raise ValueError(
                            f"{where} {mode}: fine-tuned rows for {rs} differ "
                            f"from the zero-shot rows")
                    ft_t = fdf["target"].to_numpy(dtype=np.float64)[fm]
                    if rs in GUARDED_ROW_SETS:
                        require_finite(ft_t, f"{fdf_paths[mode]}: target", rs)
                    for k, col in fold_columns(fdf, _PRED_COL):
                        if rs in GUARDED_ROW_SETS:
                            require_finite(
                                fdf[col].to_numpy(dtype=np.float64)[fm],
                                f"{fdf_paths[mode]}: {col}", rs)
                        ft_rows.append({
                            "model": name,
                            "screen": screen,
                            "split": split,
                            "row_set": rs,
                            "fold": k,
                            "n_rows": n,
                            "method": mode,
                            "ols_a": float("nan"),
                            "spearman": spearman(fdf[col].to_numpy()[fm],
                                                 ft_t),
                        })
            print(f"  [{name}] {where}: matched={int(masks[ROW_MATCHED].sum()):,} "
                  f"sldb_filtered={int(masks[ROW_ALL_SLDB].sum()):,} "
                  f"test={int(masks[ROW_ALL].sum()):,}")


# =============================================================================
# Matrix-scored models (baselines + DegreePrior)
# =============================================================================


def baseline_fold_files(pred_dir: Path, model_cv: str) -> List[Tuple[int, Path]]:
    """(fold, path) of <pred_dir>/fold_<k>_<model_cv>_predictions.npy."""
    out = []
    for p in sorted(pred_dir.glob(f"fold_*_{model_cv}_predictions.npy")):
        m = _BASELINE_FILE.match(p.name)
        if m and m.group(2) == model_cv:
            out.append((int(m.group(1)), p))
    if not out:
        raise FileNotFoundError(
            f"no fold_<k>_{model_cv}_predictions.npy under {pred_dir}")
    return sorted(out)


def degree_scores(fold_dir: Path, num_nodes: int) -> np.ndarray:
    """log1p training degree per gene (the DegreePrior's per-gene term)."""
    train_pos = np.load(fold_dir / "train_pos.npy").astype(np.int64)
    deg = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(deg, train_pos[:, 0], 1.0)
    np.add.at(deg, train_pos[:, 1], 1.0)
    return np.log1p(deg)


def _reference_indices(ref: pd.DataFrame,
                       gene_index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    i = ref["gene1_entrez"].astype(str).str.strip().map(gene_index)
    j = ref["gene2_entrez"].astype(str).str.strip().map(gene_index)
    if i.isna().any() or j.isna().any():
        raise AssertionError("matched rows contain a gene outside genes.txt")
    return i.to_numpy(dtype=np.int64), j.to_numpy(dtype=np.int64)


def score_matrix_model(
    name: str,
    kind: str,
    fold_scores: List[Tuple[int, PairScorer]],
    reference: Dict[Tuple[str, str], pd.DataFrame],
    gene_index: Dict[str, int],
    zs_rows: List[dict],
) -> None:
    """Score every reference row set with a per-fold pair scorer.

    `fold_scores` is [(k, f)] with f(i, j) -> scores; the fold's data is only
    loaded inside f, one fold at a time. Each reference holds the
    matched_incl_sldb rows; the score is read once and Spearman taken on each
    of MATCHED_ROW_SETS (matched = the rows without sldb_overlap).
    """
    idx = {key: _reference_indices(ref, gene_index)
           for key, ref in reference.items()}
    for k, scorer in fold_scores:
        for (screen, split), ref in reference.items():
            i, j = idx[(screen, split)]
            s = np.asarray(scorer(i, j))
            require_finite(s, f"{name} fold {k} on {screen}/{split}")
            t = ref["target"].to_numpy()
            for rs, m in reference_subsets(ref).items():
                zs_rows.append({
                    "model": name,
                    "kind": kind,
                    "screen": screen,
                    "split": split,
                    "row_set": rs,
                    "fold": k,
                    "n_rows": int(m.sum()),
                    "spearman": spearman(s[m], t[m]),
                })


def reference_subsets(ref: pd.DataFrame) -> Dict[str, np.ndarray]:
    """{row set: mask over a reference frame (matched_incl_sldb rows)} for
    MATCHED_ROW_SETS. A frame without sldb_overlap is the held-out rows only."""
    if "sldb_overlap" not in ref.columns:
        return {ROW_MATCHED: np.ones(len(ref), dtype=bool)}
    ov = ref["sldb_overlap"].to_numpy(dtype=bool)
    return {ROW_MATCHED: ~ov, ROW_MATCHED_INCL: np.ones(len(ref), dtype=bool),
            ROW_MATCHED_IN_SLDB: ov}


def score_sldb_label(reference: Dict[Tuple[str, str], pd.DataFrame],
                     pos_keys: set, folds: Sequence[int],
                     zs_rows: List[dict]) -> None:
    """The SynLethDB label as a predictor on the matched_in_sldb rows: 1 for a
    SynLethDB SL pair, 0 for a screened non-SL one (sldb_overlap and not SL).
    Not a trained model: the same value is recorded under every fold id, so
    it sits in the per-fold tables (sd 0) beside the models it is read with."""
    for (screen, split), ref in reference.items():
        m = reference_subsets(ref).get(ROW_MATCHED_IN_SLDB)
        if m is None or not m.any():
            continue
        lab = np.isin(ref["key"].to_numpy()[m], list(pos_keys)).astype(float)
        rho = spearman(lab, ref["target"].to_numpy()[m])
        for k in folds:
            zs_rows.append({
                "model": SLDB_LABEL_NAME,
                "kind": SLDB_LABEL_KIND,
                "screen": screen,
                "split": split,
                "row_set": ROW_MATCHED_IN_SLDB,
                "fold": k,
                "n_rows": int(m.sum()),
                "spearman": rho,
            })


def _model_params_module():
    """sl_comparison/model_params.py, loaded by file path (this module is also
    imported by tests from outside sl_comparison/)."""
    import importlib.util
    path = Path(__file__).resolve().parent / "model_params.py"
    spec = importlib.util.spec_from_file_location("_mp_for_external", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_fold_provenance(model_dir: Path, folds_dir: Path, model_cv: str,
                          allow_unstamped: bool) -> None:
    """Refuse matrices / models not trained on THESE folds (fold_stamp in
    model_params.json, written by every trainer). Same check as
    evaluate_model.py: a same-size matrix from an older gene ordering or a
    regenerated fold set would otherwise score the wrong pairs silently."""
    _model_params_module().verify_fold_stamp(
        str(model_dir), str(folds_dir), model_cv,
        allow_unstamped=allow_unstamped)


def matrix_fold_scorers(pred_dir: Path, model_cv: str,
                        num_nodes: int) -> List[Tuple[int, PairScorer]]:
    """Per-fold scorers reading (M_k[i, j] + M_k[j, i]) / 2 from memory-mapped
    .npy matrices. A screen pair is unordered and gene1/gene2 order in the
    screen CSV is arbitrary, so an asymmetric matrix must not be read on one
    triangle only."""
    out = []
    for k, path in baseline_fold_files(pred_dir, model_cv):

        def scorer(i, j, path=path):
            mat = np.load(path, mmap_mode="r")
            if mat.shape != (num_nodes, num_nodes):
                raise ValueError(f"{path}: shape {mat.shape}, expected "
                                 f"({num_nodes}, {num_nodes}) from genes.txt")
            return 0.5 * (np.asarray(mat[i, j], dtype=np.float64) +
                          np.asarray(mat[j, i], dtype=np.float64))

        out.append((k, scorer))
    return out


def degree_fold_scorers(folds_dir: Path, model_cv: str,
                        num_nodes: int) -> List[Tuple[int, PairScorer]]:
    """Per-fold DegreePrior scorers from <folds_dir>/<model_cv>/fold_k."""
    fold_dirs = sorted((folds_dir / model_cv).glob("fold_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(f"no fold_* dirs under {folds_dir / model_cv}")
    out = []
    for d in fold_dirs:

        def scorer(i, j, d=d):
            s = degree_scores(d, num_nodes)
            # float64 sum cast to float32, element-for-element what
            # degree_baseline.degree_matrix stores, so the same pairs tie.
            return (s[i] + s[j]).astype(np.float32)

        out.append((int(d.name.split("_")[1]), scorer))
    return out


# =============================================================================
# SynLethDB overlap and agreement (2026-11-12)
# =============================================================================


def load_pair_keys(path: Path) -> set:
    """'lo|hi' keys (pair_keys' format) of a whitespace-separated pair file;
    self pairs skipped."""
    out = set()
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if len(t) >= 2 and t[0] != t[1]:
                a, b = t[0].strip(), t[1].strip()
                out.add(f"{a}|{b}" if a <= b else f"{b}|{a}")
    return out


def fold_train_keys(folds_dir: Path, model_cv: str,
                    genes: Sequence[str]) -> Dict[int, set]:
    """{fold k: keys of its TRAINING pairs (train_pos + train_neg)} of the
    shared <model_cv> folds, the pairs every fold-k model was fitted on."""
    out: Dict[int, set] = {}
    fold_dirs = sorted((Path(folds_dir) / model_cv).glob("fold_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    g = np.asarray(list(genes), dtype=object)
    for d in fold_dirs:
        keys = set()
        for f in ("train_pos.npy", "train_neg.npy"):
            fp = d / f
            if not fp.exists():
                continue
            a = np.load(fp).astype(np.int64)
            if len(a):
                keys.update(pair_keys(pd.Series(g[a[:, 0]]),
                                      pd.Series(g[a[:, 1]])).tolist())
        out[int(d.name.split("_")[1])] = keys
    return out


def auroc(score: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC (average ranks for ties); NaN without both classes."""
    from scipy.stats import rankdata
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sldb_membership(keys: np.ndarray, flagged: np.ndarray, pos_keys: set,
                    neg_keys: Optional[set]) -> Tuple[np.ndarray, np.ndarray]:
    """(in_pos, in_neg) per pair. in_neg = screened non-SL and NOT SL (the
    shared folds' OR-collapse). Without the negatives file, the negatives are
    the eval's sldb_overlap pairs that are not SL (the eval matched
    sldb_overlap against both files, which check_eval_fingerprint verifies)."""
    k = np.asarray(keys, dtype=object)
    in_pos = np.fromiter((x in pos_keys for x in k), bool, len(k))
    if neg_keys is not None:
        in_neg = np.fromiter((x in neg_keys for x in k), bool, len(k))
    else:
        in_neg = np.asarray(flagged, dtype=bool).copy()
    return in_pos, in_neg & ~in_pos


def screen_scopes(screen: str, pairs: pd.DataFrame,
                  reference: Dict[Tuple[str, str], pd.DataFrame],
                  gene_index: Dict[str, int],
                  splits: Sequence[str]) -> List[Tuple[str, str, pd.DataFrame]]:
    """(scope, split, rows) the SynLethDB tables report for one screen:
    every scored screen pair (any split), those with both genes in genes.txt,
    and per split the matched_incl_sldb test rows."""
    both = (pairs["gene1_entrez"].astype(str).str.strip().isin(gene_index) &
            pairs["gene2_entrez"].astype(str).str.strip().isin(gene_index))
    out = [("all_pairs", "any", pairs),
           ("all_pairs_genes_txt", "any", pairs.loc[both.to_numpy()])]
    for sp in splits:
        if (screen, sp) in reference:
            out.append(("test_genes_txt", sp, reference[(screen, sp)]))
    return out


def screen_role(share_in_sldb: float) -> str:
    return (ROLE_IN_DISTRIBUTION if np.isfinite(share_in_sldb) and
            share_in_sldb >= IN_DISTRIBUTION_MIN_SLDB_SHARE else ROLE_EXTERNAL)


def sldb_tables(screen_pairs: Dict[str, pd.DataFrame],
                reference: Dict[Tuple[str, str], pd.DataFrame],
                gene_index: Dict[str, int], splits: Sequence[str],
                pos_keys: set, neg_keys: Optional[set],
                train_keys: Dict[int, set], strict: bool = True
                ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """The screen x scope SynLethDB overlap table, the target-vs-label
    agreement table, and {screen: screen_role}.

    overlap: n / share of the scope's pairs that are SynLethDB SL, screened
    non-SL (not SL), either, and in each fold's training pairs. agreement:
    on the pairs SynLethDB labels, AUROC of the target (higher = more SL)
    for SL vs non-SL and Spearman(label, target). screen_role is decided on
    the all_pairs share in SynLethDB (IN_DISTRIBUTION_MIN_SLDB_SHARE).
    strict: raise when the negatives file and the eval's sldb_overlap flag
    disagree (the eval matched other pair files)."""
    ov_rows, ag_rows, roles = [], [], {}
    folds = sorted(train_keys)
    for screen, pairs in screen_pairs.items():
        for scope, sp, rows in screen_scopes(screen, pairs, reference,
                                             gene_index, splits):
            keys = rows["key"].to_numpy()
            flagged = rows["sldb_overlap"].to_numpy(dtype=bool)
            in_pos, in_neg = sldb_membership(keys, flagged, pos_keys,
                                             neg_keys)
            labelled = in_pos | in_neg
            n_bad = int((labelled != flagged).sum())
            if n_bad and neg_keys is not None and strict and \
                    scope == "all_pairs":
                raise ValueError(
                    f"{screen}: {n_bad:,} pair(s) whose sldb_overlap flag "
                    f"disagrees with the SynLethDB positives / negatives "
                    f"files given here: the eval matched other pair files "
                    f"(pass the ones it used, or --allow_legacy_eval)")
            n = len(rows)

            def frac(x):
                return float(x) / n if n else float("nan")
            rec = {"screen": screen, "scope": scope, "split": sp,
                   "n_pairs": n,
                   "n_sldb_pos": int(in_pos.sum()),
                   "frac_sldb_pos": frac(in_pos.sum()),
                   "n_sldb_neg": int(in_neg.sum()),
                   "frac_sldb_neg": frac(in_neg.sum()),
                   "n_sldb_any": int(labelled.sum()),
                   "frac_sldb_any": frac(labelled.sum()),
                   "n_flag_mismatch": n_bad,
                   "neg_source": "file" if neg_keys is not None else
                   "sldb_overlap flag minus positives"}
            fr = []
            for k in folds:
                tk = train_keys[k]
                c = int(sum(1 for x in keys if x in tk))
                rec[f"n_train_fold{k}"] = c
                rec[f"frac_train_fold{k}"] = frac(c)
                fr.append(frac(c))
            rec["frac_train_mean"] = float(np.mean(fr)) if fr else float("nan")
            if scope == "all_pairs":
                roles[screen] = screen_role(rec["frac_sldb_any"])
            ov_rows.append(rec)
            t = rows["target"].to_numpy(dtype=np.float64)[labelled]
            y = in_pos[labelled]
            ag_rows.append({
                "screen": screen, "scope": scope, "split": sp,
                "n_labelled": int(labelled.sum()), "n_pos": int(y.sum()),
                "n_neg": int((~y).sum()),
                "auroc_target_vs_label": auroc(t, y),
                "spearman_target_vs_label": spearman(y.astype(float), t),
            })
    ov = pd.DataFrame(ov_rows)
    ag = pd.DataFrame(ag_rows)
    for df in (ov, ag):
        if not df.empty:
            df["screen_role"] = df["screen"].map(roles)
    return ov, ag, roles


# =============================================================================
# Markdown
# =============================================================================


def _cell(rec: Optional[pd.Series]) -> str:
    if rec is None or not np.isfinite(rec["spearman_mean"]):
        return "n/a"
    s = f"{rec['spearman_mean']:+.4f} ± {rec['spearman_sd_pop']:.4f}"
    if rec["n_folds_finite"] < rec["n_folds"]:
        s += f" ({rec['n_folds_finite']}/{rec['n_folds']} folds)"
    return s


def _lookup(summary: pd.DataFrame, **kw) -> Optional[pd.Series]:
    if summary.empty:
        return None
    m = np.ones(len(summary), dtype=bool)
    for c, v in kw.items():
        m &= (summary[c] == v).to_numpy()
    return summary[m].iloc[0] if m.any() else None


def _n_rows(summary: pd.DataFrame, screen: str, split: str,
            row_set: str) -> str:
    rec = _lookup(summary, screen=screen, split=split, row_set=row_set)
    return f"{int(rec['n_rows']):,}" if rec is not None else "?"


def render_markdown(
    zs_sum: pd.DataFrame,
    ft_sum: pd.DataFrame,
    rowsets: pd.DataFrame,
    siamese: List[Tuple[str, Path]],
    matrix_models: List[Tuple[str, str]],
    screens: Sequence[str],
    splits: Sequence[str],
    provenance: Dict[str, object],
    arm_line: Optional[str] = None,
    sldb_overlap: Optional[pd.DataFrame] = None,
    sldb_agreement: Optional[pd.DataFrame] = None,
    roles: Optional[Dict[str, str]] = None,
) -> str:
    """Readable summary: zero-shot table, SLeuth secondary rows, FT table.
    arm_line (run's --pca_arm): line 3, under the title. sldb_overlap /
    sldb_agreement / roles (sldb_tables): the in-distribution sections."""
    L: List[str] = []
    L.append("# External-screen transfer (zero-shot + fine-tuning)\n")
    if arm_line:
        L.append(arm_line + "\n")
    L.append("Metric: Spearman(score, target), target = -1 x screen phenotype "
             "(higher = more SL). Cells are mean ± population sd (ddof=0) over "
             "the training folds; per-fold values are in the CSVs and in the "
             "per-fold section below.\n")
    L.append("Screen splits: " + ", ".join(
        f"{s} = {SPLIT_LABELS.get(s, s)}" for s in splits) + ".\n")
    # Named as SPLIT_LABELS names them: cv2's screen test set is ~both_train
    # (eval_finetuning.split_pairs_cv2), so it CONTAINS the both-unseen pairs
    # rather than being the exactly-one-unseen complement of cv3.
    L.append("**The random / >=1-unseen / both-unseen columns only bear on "
             "fine-tuning.** They say how the SCREEN's pairs were divided into "
             "train / val / test (eval_finetuning --cv_type), i.e. which test "
             "genes the fine-tuned model saw in the screen's train split. A "
             "zero-shot score never uses the screen's train rows, so across "
             "the zero-shot columns only the test-row subset changes, not what "
             "the model was trained on.\n")
    L.append(f"Matched rows = {ROW_SET_LABELS[ROW_MATCHED]}. Every model below "
             "is scored on exactly these rows. SLeuth scores are RAW "
             "(uncalibrated) per-fold logits.\n")

    models = [(n, "siamese") for n, _ in siamese] + matrix_models

    L.append("## Zero-shot, matched rows\n")
    for screen in screens:
        L.append(f"### {screen}\n")
        hdr = ["Model"] + [
            f"{SPLIT_LABELS.get(s, s)} (n={_n_rows(zs_sum, screen, s, ROW_MATCHED)})"
            for s in splits
        ]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for name, _ in models:
            cells = [
                _cell(_lookup(zs_sum, model=name, screen=screen, split=s,
                              row_set=ROW_MATCHED)) for s in splits
            ]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")

    roles = roles or {}
    if roles:
        L.append("## Screen role (SynLethDB overlap)\n")
        L.append(f"A screen at least {IN_DISTRIBUTION_MIN_SLDB_SHARE:.0%} of "
                 "whose scored pairs SynLethDB lists (SL or screened non-SL) "
                 f"is `{ROLE_IN_DISTRIBUTION}`: its pairs largely are the "
                 "models' training data, so it checks in-distribution "
                 f"agreement, not transfer; otherwise `{ROLE_EXTERNAL}`. "
                 "`sldb_overlap.csv` has the counts.\n")
        if sldb_overlap is not None and not sldb_overlap.empty:
            hdr = ["Screen", "screen_role", "pairs", "SynLethDB SL",
                   "screened non-SL", "either", "in fold training (mean)"]
            L.append("| " + " | ".join(hdr) + " |")
            L.append("|" + "---|" * len(hdr))
            for _, r in sldb_overlap[sldb_overlap["scope"] ==
                                     "all_pairs"].iterrows():
                L.append(f"| {r['screen']} | {r['screen_role']} | "
                         f"{int(r['n_pairs']):,} | {r['frac_sldb_pos']:.3f} | "
                         f"{r['frac_sldb_neg']:.3f} | {r['frac_sldb_any']:.3f}"
                         f" | {r['frac_train_mean']:.3f} |")
            L.append("")
        L.append("## Zero-shot, all matched pairs vs held out\n")
        L.append(f"`{ROW_MATCHED_INCL}` = {ROW_SET_LABELS[ROW_MATCHED_INCL]}; "
                 f"`{ROW_MATCHED}` = the held-out part (not in SynLethDB); "
                 f"`{ROW_MATCHED_IN_SLDB}` = the SynLethDB part, where "
                 f"`{SLDB_LABEL_NAME}` (1 = SL, 0 = screened non-SL) is "
                 "scored as a predictor too.\n")
        hdr = ["Model", "Screen", "Row set"] + [SPLIT_LABELS.get(s, s)
                                                for s in splits]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for name, _ in models + [(SLDB_LABEL_NAME, SLDB_LABEL_KIND)]:
            for screen in screens:
                for rs in MATCHED_ROW_SETS:
                    recs = [_lookup(zs_sum, model=name, screen=screen,
                                    split=s, row_set=rs) for s in splits]
                    if all(r is None for r in recs):
                        continue
                    cells = [_cell(r) + (f" (n={int(r['n_rows']):,})"
                                         if r is not None else "")
                             for r in recs]
                    L.append(f"| {name} | {screen} | {rs} | " +
                             " | ".join(cells) + " |")
        L.append("")
    if sldb_agreement is not None and not sldb_agreement.empty:
        L.append("## SynLethDB label vs screen target\n")
        L.append("On the pairs SynLethDB labels: AUROC of the target (higher "
                 "= more SL) for SynLethDB SL vs screened non-SL, and "
                 "Spearman(label, target). CIs: bootstrap_ci.py --part "
                 "external (ci_sldb_agreement.csv).\n")
        hdr = ["Screen", "screen_role", "Scope", "Split", "n (SL / non-SL)",
               "AUROC", "Spearman"]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for _, r in sldb_agreement.iterrows():
            L.append(f"| {r['screen']} | {r['screen_role']} | {r['scope']} | "
                     f"{r['split']} | {int(r['n_labelled']):,} "
                     f"({int(r['n_pos']):,} / {int(r['n_neg']):,}) | "
                     f"{r['auroc_target_vs_label']:.4f} | "
                     f"{r['spearman_target_vs_label']:+.4f} |")
        L.append("")

    L.append("## Zero-shot, SLeuth on wider row sets (secondary)\n")
    L.append(f"`{ROW_ALL_SLDB}` = {ROW_SET_LABELS[ROW_ALL_SLDB]}: the row set "
             "of the paper's fine-tuning table. Baselines cannot score genes "
             "outside genes.txt, so these rows are SLeuth-only and NOT "
             "comparable across models: compare models on the matched rows "
             "above.\n")
    hdr = ["Variant", "Screen", "Row set"] + [SPLIT_LABELS.get(s, s)
                                              for s in splits]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for name, _ in siamese:
        for screen in screens:
            for rs in (ROW_ALL_SLDB, ROW_ALL):
                cells = []
                for s in splits:
                    rec = _lookup(zs_sum, model=name, screen=screen, split=s,
                                  row_set=rs)
                    n = f" (n={int(rec['n_rows']):,})" if rec is not None else ""
                    cells.append(_cell(rec) + n)
                L.append(f"| {name} | {screen} | {rs} | " + " | ".join(cells) +
                         " |")
    L.append("")

    if not ft_sum.empty:
        L.append("## Fine-tuning (SLeuth variants)\n")
        L.append(
            "**LP_last and Full start from an OLS-calibrated head**: each "
            "fold's (a, c) is warm-started from the closed-form OLS fit of "
            "target on the raw logit over the screen's train split. "
            "`zero_shot_raw` is the uncalibrated logit (the zero-shot table "
            "above); `zero_shot_ols` is that calibrated starting point, which "
            "differs from raw only by the sign of a (a < 0 flips it). Judge "
            "what fine-tuning adds against `zero_shot_ols`.\n")
        if (ft_sum["method"] == SCRATCH_MODE).any():
            L.append(
                "**Scratch** is the from-scratch baseline: the same "
                "architecture, fold feature space, split and fine-tuning "
                "schedule as Full, freshly initialised (no SynLethDB weight), "
                "its (a, c) warm-started from OLS on its own initial logits. "
                "Full minus Scratch is what pretraining adds.\n")
        methods = ["zero_shot_raw", "zero_shot_ols"] + list(FT_MODES)
        for name, _ in siamese:
            for rs in (ROW_MATCHED, ROW_ALL_SLDB, ROW_ALL):
                L.append(f"### {name} — {rs} ({ROW_SET_LABELS[rs]})\n")
                hdr = ["Screen", "Method"] + [SPLIT_LABELS.get(s, s)
                                              for s in splits]
                L.append("| " + " | ".join(hdr) + " |")
                L.append("|" + "---|" * len(hdr))
                for screen in screens:
                    for meth in methods:
                        recs = [
                            _lookup(ft_sum, model=name, screen=screen, split=s,
                                    row_set=rs, method=meth) for s in splits
                        ]
                        if all(r is None for r in recs):
                            continue
                        L.append(f"| {screen} | {meth} | " +
                                 " | ".join(_cell(r) for r in recs) + " |")
                L.append("")

    L.append("## Per-fold zero-shot values (matched rows)\n")
    fold_cols = sorted(
        [c for c in zs_sum.columns if re.match(r"^spearman_fold\d+$", c)],
        key=lambda c: int(c[len("spearman_fold"):]))
    hdr = ["Model", "Screen", "Split", "n"] + [c.replace("spearman_", "")
                                              for c in fold_cols]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for name, _ in models:
        for screen in screens:
            for s in splits:
                rec = _lookup(zs_sum, model=name, screen=screen, split=s,
                              row_set=ROW_MATCHED)
                if rec is None:
                    continue
                vals = [
                    "n/a" if not np.isfinite(rec[c]) else f"{rec[c]:+.4f}"
                    for c in fold_cols
                ]
                L.append(f"| {name} | {screen} | {s} | {int(rec['n_rows']):,} | "
                         + " | ".join(vals) + " |")
    L.append("")

    L.append("## Row sets\n")
    if not rowsets.empty:
        cols = ["screen", "split", "n_screen_pairs", "n_test",
                "n_test_not_sldb", "n_matched", "matched_digest", "defined_by"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for _, r in rowsets.iterrows():
            L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        L.append("")

    L.append("## Provenance\n")
    for k, v in provenance.items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================


def check_model_cv(model_dir: Path, model_cv: str) -> None:
    """Raise unless <model_dir>/config.json says it was trained on model_cv.

    The baselines and DegreePrior are read at --model_cv; a SLeuth combo
    trained on another CV would be compared against the wrong training
    folds.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found: needed to check the SLeuth model's "
            f"training CV against --model_cv {model_cv}")
    with open(cfg_path) as fh:
        cv = json.load(fh).get("cv_type")
    if cv != model_cv:
        raise ValueError(f"{model_dir}: trained on cv_type={cv!r} but "
                         f"--model_cv is {model_cv!r}; the baselines and "
                         f"DegreePrior would come from different folds")


def check_fold_set(name: str, got: Sequence[int],
                   want: Tuple[int, ...]) -> None:
    """Raise unless a matrix model covers exactly the SLeuth fold set."""
    if tuple(sorted(got)) != tuple(want):
        raise ValueError(
            f"{name}: folds {sorted(got)} != the SLeuth logit_fold<k> folds "
            f"{list(want)}; per-fold means would average different fold sets")


def heads_provenance(cfg: dict) -> str:
    """Scoring head(s) from eval_finetuning's config_used.json.

    Prefers `effective_head` (the head that actually scored each screen,
    'ANY (pooled)' for a pooled checkpoint) over `dataset_heads` (the head the
    DATASETS row names), which older evals record alone.
    """
    if not cfg:
        return "config_used.json not found"
    eff = cfg.get("effective_head")
    if eff is not None:
        return (f"effective_head={eff} (per_head={cfg.get('per_head')}, "
                f"model_num_heads={cfg.get('model_num_heads')})")
    return (f"per_head={cfg.get('per_head')}, "
            f"dataset_heads={cfg.get('dataset_heads')} (no effective_head "
            f"recorded: older eval; a pooled checkpoint scores every screen "
            f"with its single head whatever dataset_heads says)")


def run(
    siamese: List[Tuple[str, Path]],
    baselines: List[Tuple[str, Path]],
    degree_prior: bool,
    genes_path: Path,
    folds_dir: Path,
    out_dir: Path,
    model_cv: str = "cv3",
    screens: Sequence[str] = SCREENS,
    splits: Sequence[str] = SPLITS,
    allow_unstamped: bool = False,
    allow_legacy_eval: bool = False,
    pca_arm: Optional[str] = None,
    pca_label: Optional[str] = None,
    sl_path: Optional[Path] = None,
    neg_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Build every table and write them to `out_dir`. Returns the frames.

    sl_path / neg_path: the SynLethDB positives / screened negatives the
    overlap and agreement tables read (default: the pair files the folds were
    built from, <folds_dir>/meta.json pos_path / neg_path). Without the
    positives those tables and the SynLethDB_label row are not written (a
    provenance line says so); without the negatives they are derived from the
    eval's sldb_overlap flag.

    allow_unstamped: accept model dirs whose model_params.json carries no
    fold_stamp (test fixtures / pre-stamp runs); a stamp that is present
    but does not match `folds_dir` is always an error.
    allow_legacy_eval: downgrade check_eval_fingerprint's refusals (missing
    config_used.json, positives-only overlap, missing or mismatched
    checkpoint_sha256 / fold_ids) to warnings listed in the provenance.
    pca_arm / pca_label: label only (the PCA arm of the SLeuth combos and
    its PCA in words): a trailing pca_arm column on every CSV WRITTEN (the
    returned frames are unchanged) and summary.md's 'PCA arm:' line and
    Provenance bullet; None writes what it always wrote."""
    if pca_label is not None and pca_arm is None:
        raise ValueError("pca_label needs pca_arm")
    if not siamese:
        raise ValueError("at least one --siamese NAME=MODEL_DIR is required "
                         "(its CSVs define the screen rows)")
    gene_index = load_gene_index(genes_path)
    num_nodes = len(gene_index)
    # The matrices are indexed by genes_path, but every provenance check below
    # is against folds_dir. A reordered (same-size) gene list would pass those
    # checks and put every score on the wrong pair, so the two must be the
    # same list in the same order.
    fold_genes = Path(folds_dir) / "genes.txt"
    if not fold_genes.exists():
        raise ValueError(f"{fold_genes} missing: cannot confirm the gene "
                         f"order of the score matrices")
    if list(load_gene_index(fold_genes)) != list(gene_index):
        raise ValueError(
            f"--genes {genes_path} is not {fold_genes} (different genes or "
            f"order); the matrices are indexed by the fold set's genes.txt")
    legacy_evals: List[str] = []
    eval_fold_ids: Dict[str, Dict[str, List[int]]] = {}
    for _name, model_dir in siamese:
        check_model_cv(model_dir, model_cv)
        check_fold_provenance(model_dir, folds_dir, model_cv, allow_unstamped)
        current = checkpoint_hashes(model_dir)
        for split in splits:
            warn = check_eval_fingerprint(model_dir, split, current,
                                          allow_legacy_eval)
            if warn:
                legacy_evals.append(f"{_name}/{split}")
            fids = read_eval_config(model_dir, split).get("fold_ids")
            if fids is not None:
                eval_fold_ids.setdefault(_name, {})[split] = list(fids)

    reference: Dict[Tuple[str, str], pd.DataFrame] = {}
    ref_owner: Dict[Tuple[str, str], str] = {}
    zs_rows: List[dict] = []
    ft_rows: List[dict] = []
    rowset_rows: List[dict] = []
    fold_sets: Dict[str, Tuple[int, ...]] = {}
    screen_pairs: Dict[str, pd.DataFrame] = {}

    for name, model_dir in siamese:
        print(f"SLeuth variant {name}: {model_dir}")
        score_siamese_variant(name, model_dir, screens, splits, gene_index,
                              reference, ref_owner, zs_rows, ft_rows,
                              rowset_rows, fold_sets=fold_sets,
                              eval_fold_ids=eval_fold_ids.get(name),
                              screen_pairs=screen_pairs)
    distinct = sorted(set(fold_sets.values()))
    if len(distinct) != 1:
        raise ValueError(
            "SLeuth predictions carry different logit_fold<k> sets: " +
            "; ".join(f"{w}: {list(f)}" for w, f in sorted(fold_sets.items())))
    siamese_folds = distinct[0]

    matrix_models: List[Tuple[str, str]] = []
    for name, pred_dir in baselines:
        print(f"baseline {name}: {pred_dir}")
        check_fold_provenance(pred_dir, folds_dir, model_cv, allow_unstamped)
        scorers = matrix_fold_scorers(pred_dir, model_cv, num_nodes)
        check_fold_set(f"baseline {name} ({pred_dir})",
                       [k for k, _ in scorers], siamese_folds)
        score_matrix_model(name, "baseline", scorers, reference, gene_index,
                           zs_rows)
        matrix_models.append((name, "baseline"))
    if degree_prior:
        print(f"{DEGREE_PRIOR_NAME}: {folds_dir / model_cv}")
        scorers = degree_fold_scorers(folds_dir, model_cv, num_nodes)
        check_fold_set(f"{DEGREE_PRIOR_NAME} ({folds_dir / model_cv})",
                       [k for k, _ in scorers], siamese_folds)
        score_matrix_model(DEGREE_PRIOR_NAME, "degree_prior", scorers,
                           reference, gene_index, zs_rows)
        matrix_models.append((DEGREE_PRIOR_NAME, "degree_prior"))

    # ---- SynLethDB overlap / agreement / label row ----
    sl_path, neg_path = default_sldb_paths(folds_dir, sl_path, neg_path)
    sldb_prov: Dict[str, object] = {}
    ov_df = ag_df = pd.DataFrame()
    roles: Dict[str, str] = {}
    if sl_path is not None and sl_path.is_file():
        pos_keys = load_pair_keys(sl_path)
        neg_keys = (load_pair_keys(neg_path)
                    if neg_path is not None and neg_path.is_file() else None)
        if neg_keys is None:
            print(f"  WARNING: SynLethDB negatives {neg_path} not found: the "
                  f"screened-negative counts are the eval's sldb_overlap "
                  f"pairs that are not SL")
        train_keys = fold_train_keys(folds_dir, model_cv, list(gene_index))
        ov_df, ag_df, roles = sldb_tables(screen_pairs, reference, gene_index,
                                          splits, pos_keys, neg_keys,
                                          train_keys,
                                          strict=not allow_legacy_eval)
        score_sldb_label(reference, pos_keys, siamese_folds, zs_rows)
        sldb_prov = {
            "SynLethDB positives (overlap / agreement)": str(sl_path),
            "SynLethDB screened negatives (overlap / agreement)":
            str(neg_path) if neg_keys is not None else
            f"{neg_path} not found: sldb_overlap pairs that are not SL",
            "fold training pairs (overlap)":
            f"{folds_dir / model_cv}/fold_k/train_pos.npy + train_neg.npy",
            "screen_role": f"{ROLE_IN_DISTRIBUTION} when >= "
            f"{IN_DISTRIBUTION_MIN_SLDB_SHARE:g} of the screen's scored pairs "
            f"are in SynLethDB, else {ROLE_EXTERNAL}: " + ", ".join(
                f"{k}={v}" for k, v in roles.items()),
        }
    else:
        what = (f"{sl_path} not found" if sl_path is not None else
                f"none given (--sl_path) or recorded ({folds_dir}/meta.json "
                f"pos_path)")
        print(f"  WARNING: SynLethDB positives {what}: no sldb_overlap.csv / "
              f"sldb_agreement.csv / {SLDB_LABEL_NAME} row")
        sldb_prov = {"SynLethDB positives (overlap / agreement)":
                     f"{what}: tables not written"}

    zs_long = pd.DataFrame(zs_rows)
    ft_long = pd.DataFrame(ft_rows)
    rowsets = pd.DataFrame(rowset_rows)
    zs_sum = summarize(zs_rows, ["model", "kind", "screen", "split", "row_set"])
    if roles:
        for df_ in (zs_sum, rowsets):
            if not df_.empty:
                df_["screen_role"] = df_["screen"].map(roles)
    ft_sum = summarize(ft_rows,
                       ["model", "screen", "split", "row_set", "method"])
    if not ft_sum.empty:
        ft_sum["starts_from_ols_head"] = ft_sum["method"].isin(
            OLS_WARM_MODES)

    heads = {}
    for name, model_dir in siamese:
        heads[name] = heads_provenance(read_eval_config(model_dir, splits[0]))
    provenance: Dict[str, object] = {
        "genes.txt": f"{genes_path} ({num_nodes:,} genes)",
        "model CV (baselines / DegreePrior folds)": model_cv,
        "SLeuth variants": "; ".join(f"{n} = {p}" for n, p in siamese),
        "SLeuth scoring head (eval_finetuning config)": "; ".join(
            f"{n}: {h}" for n, h in heads.items()),
        "baselines": ("; ".join(f"{n} = {p}" for n, p in baselines) +
                      " (score = (M[i,j] + M[j,i]) / 2)") if baselines
        else "none",
        "folds": ", ".join(str(k) for k in siamese_folds) +
        " (identical for every model)",
        "DegreePrior": (f"log1p(deg_i) + log1p(deg_j) over "
                        f"{folds_dir / model_cv}/fold_k/train_pos.npy "
                        f"(float32, as degree_baseline.py)"
                        if degree_prior else "off"),
        "sd": "population (ddof=0); eval_finetuning's own CSVs use ddof=1",
        # Legacy evals are not verified leakage-filtered or tied to these
        # checkpoints, so the table must say so, not only the log.
        "eval fingerprint (checkpoint_sha256, fold_ids, 2 overlap sources)":
        ("NOT verified (--allow_legacy_eval) for " + ", ".join(legacy_evals))
        if legacy_evals else "verified for every SLeuth variant and split",
        **sldb_prov,
    }

    arm_line = None
    if pca_arm is not None:
        arm_line = (f"PCA arm: {pca_arm}" +
                    (f" ({pca_label})" if pca_label else ""))
        # (the baselines and DegreePrior are shared by every arm)
        provenance["PCA arm of the SLeuth combos"] = \
            arm_line[len("PCA arm: "):]

    def _lab(df: pd.DataFrame) -> pd.DataFrame:
        # The arm in a trailing column of the copy written, so the file says
        # it wherever it is taken; paper_tables reads these CSVs by column
        # name, so no table changes.
        return df if pca_arm is None else df.assign(pca_arm=pca_arm)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _replace(name: str, write: Callable[[Path], object]) -> None:
        # Through <name>.tmp and a rename, so a final name never holds a cut
        # file: the paper-tables job runs afterany and paper_tables.py reads
        # the two summaries with pandas, which takes a line cut short (a write
        # that failed part-way, e.g. a full quota) as a row, the cut number
        # included.
        tmp = out_dir / (name + ".tmp")
        try:
            write(tmp)
            tmp.replace(out_dir / name)
        finally:
            tmp.unlink(missing_ok=True)

    _replace("row_sets.csv", lambda p: _lab(rowsets).to_csv(p, index=False))
    _replace("zero_shot_per_fold.csv",
             lambda p: _lab(zs_long).to_csv(p, index=False))
    _replace("zero_shot_summary.csv",
             lambda p: _lab(zs_sum).to_csv(p, index=False))
    _replace("finetune_per_fold.csv",
             lambda p: _lab(ft_long).to_csv(p, index=False))
    _replace("finetune_summary.csv",
             lambda p: _lab(ft_sum).to_csv(p, index=False))
    if not ov_df.empty:
        _replace("sldb_overlap.csv",
                 lambda p: _lab(ov_df).to_csv(p, index=False))
        _replace("sldb_agreement.csv",
                 lambda p: _lab(ag_df).to_csv(p, index=False))
    md = render_markdown(zs_sum, ft_sum, rowsets, siamese, matrix_models,
                         screens, splits, provenance, arm_line,
                         sldb_overlap=ov_df, sldb_agreement=ag_df,
                         roles=roles)
    _replace("summary.md", lambda p: p.write_text(md))
    print(f"\nwrote {out_dir}/{{row_sets,zero_shot_per_fold,zero_shot_summary,"
          f"finetune_per_fold,finetune_summary" +
          (",sldb_overlap,sldb_agreement" if not ov_df.empty else "") +
          "}.csv + summary.md")
    return {
        "row_sets": rowsets,
        "zero_shot_per_fold": zs_long,
        "zero_shot_summary": zs_sum,
        "finetune_per_fold": ft_long,
        "finetune_summary": ft_sum,
        "sldb_overlap": ov_df,
        "sldb_agreement": ag_df,
        "screen_roles": roles,
        "markdown": md,
    }


def default_sldb_paths(folds_dir: Path, sl_path: Optional[Path] = None,
                       neg_path: Optional[Path] = None
                       ) -> Tuple[Optional[Path], Optional[Path]]:
    """sl_path / neg_path, each defaulting to the pair file the folds were
    built from (<folds_dir>/meta.json pos_path / neg_path, repo-relative;
    sl_comparison/prepare_folds.py records both). None when neither is given
    nor recorded (a fold set without meta.json: no SynLethDB tables)."""
    meta = {}
    mp = Path(folds_dir) / "meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except ValueError:
            meta = {}

    def _res(given, key):
        v = given or meta.get(key)
        if not v:
            return None
        p = Path(v)
        return p if p.is_absolute() else REPO_ROOT / p
    return _res(sl_path, "pos_path"), _res(neg_path, "neg_path")


def default_baselines(model_cv: str,
                      allow_missing: bool = False) -> List[Tuple[str, Path]]:
    """SLMGAE / SLGNN / NSF4SL prediction dirs at their standard paths.

    A missing one raises (the table would silently lose a row) unless
    allow_missing, which skips it with a note.
    """
    out, missing = [], []
    for name, tmpl in DEFAULT_BASELINES:
        p = REPO_ROOT / tmpl.format(cv=model_cv)
        if p.is_dir():
            out.append((name, p))
        else:
            missing.append((name, p))
    if missing and not allow_missing:
        raise FileNotFoundError(
            "default baseline(s) not found: " +
            "; ".join(f"{n} at {p}" for n, p in missing) +
            ". Pass --baseline NAME=DIR explicitly, or "
            "--allow_missing_baselines to skip them.")
    for name, p in missing:
        print(f"  default baseline {name} not found at {p}; skipped "
              f"(--allow_missing_baselines)")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Zero-shot external-transfer table on identical rows for "
        "SLeuth variants + baselines, and the SLeuth fine-tuning table.")
    ap.add_argument("--siamese", action="append", default=[],
                    metavar="NAME=MODEL_DIR", required=True,
                    help="SLeuth variant: its --model_cv combo dir (cv3 by "
                    "default) holding eval_finetuning_<split>/ for every "
                    "--splits entry (repeatable)")
    ap.add_argument("--baseline", action="append", default=None,
                    metavar="NAME=PRED_DIR",
                    help="dense-matrix baseline dir holding "
                    "fold_<k>_<model_cv>_predictions.npy (repeatable; default: "
                    "SLMGAE, SLGNN, NSF4SL at their standard paths if present)")
    ap.add_argument("--allow_missing_baselines", action="store_true",
                    help="with the default baselines (no --baseline), skip "
                    "one whose directory is missing instead of stopping")
    ap.add_argument("--no_baselines", action="store_true",
                    help="score no baseline (SLeuth variants and DegreePrior "
                    "only), instead of falling back to the default dirs; "
                    "cannot be combined with --baseline")
    ap.add_argument("--degree_prior", action=argparse.BooleanOptionalAction,
                    default=True, help="add the DegreePrior control row")
    ap.add_argument("--genes", default=None,
                    help="gene order of the score matrices; default and "
                    "required value: <folds_dir>/genes.txt")
    ap.add_argument("--folds_dir",
                    default=str(REPO_ROOT / "sl_comparison" / "folds"),
                    help="shared folds (DegreePrior training degrees)")
    ap.add_argument("--model_cv", default="cv3", choices=["cv1", "cv2", "cv3"],
                    help="benchmark CV the baselines/DegreePrior were trained "
                    "on (must match the SLeuth model dirs)")
    ap.add_argument("--allow_unstamped", action="store_true",
                    help="accept model dirs without a fold_stamp in "
                    "model_params.json (a stamp that does not match "
                    "--folds_dir is always refused)")
    ap.add_argument("--allow_legacy_eval", action="store_true",
                    help="accept eval_finetuning dirs without config_used.json, "
                    "with a positives-only sldb_overlap, or without matching "
                    "checkpoint_sha256 / fold_ids (older evals); their rows "
                    "are then NOT verified leakage-filtered or tied to the "
                    "current checkpoints (listed in summary.md). Off by "
                    "default")
    ap.add_argument("--screens", nargs="+", default=list(SCREENS))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--pca_arm", default=None,
                    choices=["none", "joint", "per_modality", "both"],
                    help="label only: the PCA arm of the --siamese combos "
                    "(siamese_sl/slurm/config.conf PCA arms), written as a "
                    "trailing pca_arm column of every CSV and a 'PCA arm:' "
                    "line in summary.md")
    ap.add_argument("--pca_label", default=None,
                    help="label only (needs --pca_arm): the arm's PCA in "
                    "words, as config.conf pca_arm_label says it")
    ap.add_argument("--sl_path", default=None,
                    help="SynLethDB SL pairs for sldb_overlap.csv / "
                    "sldb_agreement.csv / the SynLethDB_label row (default: "
                    "<folds_dir>/meta.json pos_path; neither: those are not "
                    "written)")
    ap.add_argument("--neg_path", default=None,
                    help="SynLethDB screened non-SL pairs (default: "
                    "<folds_dir>/meta.json neg_path; when absent, the eval's "
                    "sldb_overlap pairs that are not SL)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(argv)
    if args.pca_label is not None and args.pca_arm is None:
        ap.error("--pca_label needs --pca_arm")

    siamese = parse_named_paths(args.siamese, "--siamese")
    if args.no_baselines and args.baseline is not None:
        ap.error("--no_baselines and --baseline are mutually exclusive")
    baselines = ([] if args.no_baselines else
                 parse_named_paths(args.baseline, "--baseline")
                 if args.baseline is not None else default_baselines(
                     args.model_cv, args.allow_missing_baselines))
    names = ([n for n, _ in siamese] + [n for n, _ in baselines] +
             ([DEGREE_PRIOR_NAME] if args.degree_prior else []))
    clash = {n for n in names if names.count(n) > 1}
    if clash:
        raise SystemExit(f"model names used twice: {sorted(clash)}")

    res = run(siamese, baselines, args.degree_prior,
              Path(args.genes or Path(args.folds_dir) / "genes.txt"),
              Path(args.folds_dir), Path(args.out_dir), args.model_cv,
              args.screens, args.splits, args.allow_unstamped,
              args.allow_legacy_eval, args.pca_arm, args.pca_label,
              Path(args.sl_path) if args.sl_path else None,
              Path(args.neg_path) if args.neg_path else None)
    print()
    print(res["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
