#!/usr/bin/env python3
"""Deterministic tests for sl_comparison/external_transfer.py.

Builds tiny synthetic eval_finetuning CSVs, dense baseline matrices and a
fold dir in a temp directory, then checks:
  - row selection (split == test, not sldb_overlap, both genes in genes.txt)
  - the identical-rows assertion across SLeuth variants
  - Spearman per fold and the mean / POPULATION-sd (ddof=0) aggregation
  - baseline scoring through the genes.txt index (symmetrised, so an
    asymmetric matrix scores (M[i,j] + M[j,i]) / 2), DegreePrior degrees
    (float32, per fold) and the fine-tuning table (incl. the OLS sign flip)
  - the guards: non-finite scores/targets on a REPORTED row set (matched and
    all_sldb_filtered, the paper's SLeuth column; all_unfiltered stays
    tolerant), a fold set that differs from SLeuth's, a --model_cv mismatch,
    missing default baselines
  - the eval fingerprint (config_used.json checkpoint_sha256 + fold_ids +
    two overlap sources): a retrained or different checkpoint set, missing or
    renumbered fold ids and a missing config_used.json are refused unless
    allow_legacy_eval, which the provenance then discloses
  - a main() smoke run with explicit --baseline paths
  - the SLURM gate's fold-stamp pre-check (run_external_transfer.sh), which
    leaves a stale model ABSENT instead of letting run() refuse every table

Needs only numpy / pandas / scipy:

    python3 siamese_sl/tests/test_external_transfer.py
    # or: python -m pytest siamese_sl/tests/test_external_transfer.py -q
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# Loaded by file path so sl_comparison/ never lands on sys.path, where its
# modules could shadow same-named siamese_sl ones (retrieval_metrics.py).
_spec = importlib.util.spec_from_file_location(
    "external_transfer",
    os.path.join(_REPO, "sl_comparison", "external_transfer.py"))
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)

SCREEN = "Toy"
SPLIT = "cv1"
# Deliberately NOT sorted, so an index bug (numeric order vs genes.txt order)
# would show up in the baseline scores.
GENES = ["50", "30", "10", "60", "20", "40"]
GIDX = {g: i for i, g in enumerate(GENES)}

# (gene1, gene2, target, sldb_overlap, split)
ROWS = [
    ("10", "20", 0.1, False, "test"),  # matched
    ("10", "30", 0.2, False, "test"),  # matched
    ("20", "40", 0.3, False, "test"),  # matched
    ("30", "60", 0.4, False, "test"),  # matched
    ("40", "50", 0.5, False, "test"),  # matched
    ("10", "40", 0.9, True, "test"),  # SLDB positive -> only all_unfiltered
    ("10", "99", 0.7, False, "test"),  # 99 not in genes.txt -> not matched
    ("20", "30", 0.15, False, "train"),
    ("50", "60", 0.35, False, "train"),
    ("20", "50", 0.25, False, "train"),
    ("30", "40", 0.45, False, "val"),
]
MATCHED = [r for r in ROWS if r[4] == "test" and not r[3] and r[1] != "99"]
TEST = [r for r in ROWS if r[4] == "test"]
# fold1 logits: a fixed non-monotone map of target (negative slope overall,
# so its OLS a < 0 on the train rows).
FOLD1 = {0.1: -0.2, 0.2: -0.1, 0.3: -0.5, 0.4: -0.3, 0.5: -0.4, 0.9: -0.9,
         0.7: -0.6, 0.15: -0.15, 0.35: -0.35, 0.25: -0.25, 0.45: -0.45}
# Fine-tuned fold1 predictions on test rows (fold0 = target).
FT1 = {0.1: 0.3, 0.2: 0.1, 0.3: 0.2, 0.4: 0.5, 0.5: 0.4, 0.9: 0.0, 0.7: 0.6}


def _key(g1, g2):
    return "|".join(sorted([g1, g2]))


N_FOLDS = 2  # logit_fold0 / logit_fold1 below


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_variant(root: Path, rows, shuffle=False, swap=False,
                   model_cv="cv3", logit1=None,
                   modes=et.OLS_WARM_MODES) -> Path:
    """One SLeuth-style model dir with a single (screen, split).

    logit1: optional {target: value} overriding fold1's logit (e.g. NaN).
    modes: the fine-tuning modes the eval ran (LP_last / Full by default, as
    eval_finetuning.conf; Scratch is opt-in).
    Writes the eval fingerprint eval_finetuning.py records: one checkpoint
    per fold and a config_used.json holding their sha256 (keyed by a path
    relative to siamese_sl/, as the eval runs from there), fold_ids and both
    overlap sources.
    """
    d = root / "eval_finetuning_cv1"
    d.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"cv_type": model_cv}))
    ck = root / "checkpoints"
    ck.mkdir()
    hashes = {}
    for k in range(N_FOLDS):
        f = ck / f"fold_{k}_best.pt"
        f.write_bytes(f"{root.name} weights, fold {k}".encode())
        hashes[f"results/{root.name}/checkpoints/{f.name}"] = _sha256(f)
    (d / "config_used.json").write_text(json.dumps({
        "ft_modes": list(modes),
        "sldb_overlap_sources": ["sl.txt", "neg.txt"],
        "fold_ids": list(range(N_FOLDS)),
        "checkpoint_sha256": hashes}))
    recs = []
    for g1, g2, t, ov, sp in rows:
        if swap:
            g1, g2 = g2, g1
        l1 = FOLD1.get(t, -t)
        if logit1 and t in logit1:
            l1 = logit1[t]
        recs.append({"gene1_entrez": g1, "gene2_entrez": g2,
                     "outcome_raw": -t, "target": t, "sldb_overlap": ov,
                     "split": sp, "logit_fold0": t,
                     "logit_fold1": l1})
    df = pd.DataFrame(recs)
    df["logit_mean"] = df[["logit_fold0", "logit_fold1"]].mean(axis=1)
    if shuffle:
        df = df.iloc[::-1].reset_index(drop=True)
    df.to_csv(d / f"{SCREEN}_predictions.csv", index=False)
    test = df[df["split"] == "test"]
    for mode in modes:
        ft = test[["gene1_entrez", "gene2_entrez", "outcome_raw", "target",
                   "sldb_overlap"]].copy()
        ft["pred_fold0"] = test["target"].to_numpy()
        ft["pred_fold1"] = [FT1.get(t, t) for t in test["target"]]
        ft["pred_mean"] = ft[["pred_fold0", "pred_fold1"]].mean(axis=1)
        ft.to_csv(d / f"{SCREEN}_finetuned_predictions_{mode}.csv",
                  index=False)
    return root


# Training positives per DegreePrior fold (gene IDs).
#   fold 0: gene 10 degree 3; 20, 30, 40 degree 1; 50, 60 degree 0.
#   fold 1: gene 20 degree 4; 30, 40, 50, 60 degree 1; 10 degree 0.
DEG_TRAIN = {
    0: [("10", "20"), ("10", "30"), ("10", "40")],
    1: [("20", "30"), ("20", "40"), ("20", "50"), ("20", "60")],
}


def _write_common(root: Path, base_folds=(0, 1), deg_folds=(0, 1)):
    """genes.txt, a baseline (fold0 rho +1, fold1 rho -1) and a fold dir."""
    genes = root / "genes.txt"
    genes.write_text("\n".join(GENES) + "\n")
    bdir = root / "base"
    bdir.mkdir()
    for k in base_folds:
        sign = 1.0 if k == 0 else -1.0
        m = np.zeros((len(GENES), len(GENES)), dtype=np.float32)
        for g1, g2, t, _, _ in MATCHED:
            m[GIDX[g1], GIDX[g2]] = m[GIDX[g2], GIDX[g1]] = sign * t
        np.save(bdir / f"fold_{k}_cv3_predictions.npy", m)
    (root / "folds").mkdir()
    # run() requires --genes to be the fold set's own genes.txt
    (root / "folds" / "genes.txt").write_text("\n".join(GENES) + "\n")
    for k in deg_folds:
        fdir = root / "folds" / "cv3" / f"fold_{k}"
        fdir.mkdir(parents=True)
        np.save(fdir / "train_pos.npy",
                np.array([[GIDX[a], GIDX[b]] for a, b in DEG_TRAIN[k]],
                         dtype=np.int32))
    return genes, bdir, root / "folds"


def _run(root: Path, variants, baseline=True, degree=True,
         allow_legacy_eval=False, **common):
    genes, bdir, folds = _write_common(root, **common)
    return et.run(variants, [("Base", bdir)] if baseline else [], degree,
                  genes, folds, root / "out", "cv3", [SCREEN], [SPLIT],
                  allow_unstamped=True, allow_legacy_eval=allow_legacy_eval)


def _edit_eval_cfg(v: Path, **changes):
    """Update keys of v's config_used.json (a None value deletes the key)."""
    p = v / "eval_finetuning_cv1" / "config_used.json"
    cfg = json.loads(p.read_text())
    for k, val in changes.items():
        if val is None:
            cfg.pop(k, None)
        else:
            cfg[k] = val
    p.write_text(json.dumps(cfg))


def _raises(fn, exc, *needles):
    """Assert fn() raises exc whose message contains every needle."""
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"{exc.__name__} not raised")


def _get(summary, **kw):
    m = np.ones(len(summary), dtype=bool)
    for c, v in kw.items():
        m &= (summary[c] == v).to_numpy()
    assert m.sum() == 1, (kw, int(m.sum()))
    return summary[m].iloc[0]


# ---------------------------------------------------------------------------
# Unit pieces
# ---------------------------------------------------------------------------
def test_aggregate_folds_population_sd():
    mean, sd, n = et.aggregate_folds([1.0, -1.0])
    assert (mean, sd, n) == (0.0, 1.0, 2)  # ddof=1 would give sqrt(2)
    mean, sd, n = et.aggregate_folds([0.2, 0.4, float("nan")])
    assert n == 2 and abs(mean - 0.3) < 1e-12 and abs(sd - 0.1) < 1e-12
    mean, sd, n = et.aggregate_folds([float("nan")])
    assert n == 0 and np.isnan(mean) and np.isnan(sd)


def test_spearman_constant_is_nan():
    assert np.isnan(et.spearman(np.ones(5), np.arange(5.0)))
    assert abs(et.spearman(np.arange(5.0), np.arange(5.0)) - 1.0) < 1e-12


def test_parse_named_paths():
    out = et.parse_named_paths(["A=/x", "B = /y "], "--siamese")
    assert out == [("A", Path("/x")), ("B", Path("/y"))]
    for bad in (["noequals"], ["=/x"], ["A=/x", "A=/y"]):
        try:
            et.parse_named_paths(bad, "--siamese")
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------
def test_row_selection():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        df = et.read_pair_csv(v / "eval_finetuning_cv1" /
                              f"{SCREEN}_predictions.csv")
        got = {rs: set(df.loc[et.row_mask(df, GIDX, rs), "key"])
               for rs in et.ROW_SETS}
        assert got[et.ROW_MATCHED] == {_key(a, b) for a, b, *_ in MATCHED}
        assert got[et.ROW_ALL_SLDB] == got[et.ROW_MATCHED] | {"10|99"}
        assert got[et.ROW_ALL] == {_key(a, b) for a, b, *_ in TEST}

        res = _run(root, [("A", v)])
        rs = res["row_sets"].iloc[0]
        assert (rs["n_test"], rs["n_test_not_sldb"], rs["n_matched"]) == (7, 6, 5)
        zs = res["zero_shot_summary"]
        for model in ("A", "Base", et.DEGREE_PRIOR_NAME):
            assert _get(zs, model=model, row_set=et.ROW_MATCHED)["n_rows"] == 5
        assert _get(zs, model="A", row_set=et.ROW_ALL_SLDB)["n_rows"] == 6
        assert _get(zs, model="A", row_set=et.ROW_ALL)["n_rows"] == 7
        # Baselines/DegreePrior are scored on the matched-gene row sets only:
        # matched (held out), matched_incl_sldb (+ the SLDB test row 10|40)
        # and matched_in_sldb (10|40 alone).
        assert set(zs.loc[zs["model"] == "Base", "row_set"]) == \
            set(et.MATCHED_ROW_SETS)
        for model in ("A", "Base", et.DEGREE_PRIOR_NAME):
            assert _get(zs, model=model,
                        row_set=et.ROW_MATCHED_INCL)["n_rows"] == 6
            assert _get(zs, model=model,
                        row_set=et.ROW_MATCHED_IN_SLDB)["n_rows"] == 1
        assert (rs["n_matched_incl_sldb"], rs["n_matched_in_sldb"]) == (6, 1)


# ---------------------------------------------------------------------------
# Identical-rows assertion
# ---------------------------------------------------------------------------
def test_identical_rows_pass_with_reordered_and_swapped_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = _write_variant(root / "A", ROWS)
        b = _write_variant(root / "B", ROWS, shuffle=True, swap=True)
        zs = _run(root, [("A", a), ("B", b)])["zero_shot_summary"]
        for rs in et.ROW_SETS:
            ra = _get(zs, model="A", row_set=rs)
            rb = _get(zs, model="B", row_set=rs)
            assert ra["n_rows"] == rb["n_rows"]
            assert abs(ra["spearman_mean"] - rb["spearman_mean"]) < 1e-12


def test_identical_rows_assertion_raises_on_different_split():
    rows_b = [r if _key(r[0], r[1]) != "40|50" else r[:4] + ("train", )
              for r in ROWS]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = _write_variant(root / "A", ROWS)
        b = _write_variant(root / "B", rows_b)
        try:
            _run(root, [("A", a), ("B", b)])
        except ValueError as e:
            assert "disagree" in str(e) and "Toy/cv1" in str(e), e
            return
        raise AssertionError("different matched row sets were accepted")


def test_identical_rows_assertion_raises_on_different_target():
    rows_b = [r if _key(r[0], r[1]) != "10|20" else (r[0], r[1], 0.15) +
              r[3:] for r in ROWS]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = _write_variant(root / "A", ROWS)
        b = _write_variant(root / "B", rows_b)
        try:
            _run(root, [("A", a), ("B", b)])
        except ValueError as e:
            assert "different targets" in str(e), e
            return
        raise AssertionError("different targets were accepted")


# ---------------------------------------------------------------------------
# Spearman per fold + pop-sd aggregation, end to end
# ---------------------------------------------------------------------------
def test_zero_shot_values_and_aggregation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        res = _run(root, [("A", v)])
        zs = res["zero_shot_summary"]
        t = np.array([r[2] for r in MATCHED])

        # Baseline: fold0 = +target, fold1 = -target via the genes.txt index.
        base = _get(zs, model="Base", row_set=et.ROW_MATCHED)
        assert abs(base["spearman_fold0"] - 1.0) < 1e-12
        assert abs(base["spearman_fold1"] + 1.0) < 1e-12
        assert abs(base["spearman_mean"]) < 1e-12
        assert abs(base["spearman_sd_pop"] - 1.0) < 1e-12

        # SLeuth: raw logits, per fold.
        rho1 = spearmanr([FOLD1[x] for x in t], t)[0]
        sl = _get(zs, model="A", row_set=et.ROW_MATCHED)
        assert abs(sl["spearman_fold0"] - 1.0) < 1e-12
        assert abs(sl["spearman_fold1"] - rho1) < 1e-12
        assert abs(sl["spearman_mean"] - np.mean([1.0, rho1])) < 1e-12
        assert abs(sl["spearman_sd_pop"] - np.std([1.0, rho1])) < 1e-12

        # DegreePrior: log1p(deg) sums over each fold's training positives,
        # float64 sum cast to float32 as degree_baseline.py stores it.
        dp = _get(zs, model=et.DEGREE_PRIOR_NAME, row_set=et.ROW_MATCHED)
        rhos = []
        for k, pairs in sorted(DEG_TRAIN.items()):
            deg = {g: 0 for g in GENES}
            for a, b in pairs:
                deg[a] += 1
                deg[b] += 1
            s = np.array([np.log1p(deg[a]) + np.log1p(deg[b])
                          for a, b, *_ in MATCHED]).astype(np.float32)
            rhos.append(spearmanr(s, t)[0])
            assert abs(dp[f"spearman_fold{k}"] - rhos[-1]) < 1e-12, k
        assert dp["n_folds"] == 2
        assert abs(dp["spearman_sd_pop"] - np.std(rhos)) < 1e-12

        # Long form carries one row per fold.
        long = res["zero_shot_per_fold"]
        assert len(long[(long["model"] == "Base") &
                        (long["row_set"] == et.ROW_MATCHED)]) == 2
        for f in ("row_sets.csv", "zero_shot_per_fold.csv",
                  "zero_shot_summary.csv", "finetune_per_fold.csv",
                  "finetune_summary.csv", "summary.md"):
            assert (root / "out" / f).exists(), f
        md = (root / "out" / "summary.md").read_text()
        assert "OLS-calibrated head" in md
        assert "only bear on fine-tuning" in md


def test_finetune_table():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        ft = _run(root, [("A", v)], baseline=False,
                  degree=False)["finetune_summary"]
        t = np.array([r[2] for r in MATCHED])
        raw1 = spearmanr([FOLD1[x] for x in t], t)[0]

        # fold1's train logits are -target, so OLS a < 0 and calibration
        # flips the raw Spearman's sign; fold0 (a > 0) is unchanged.
        ols = _get(ft, model="A", row_set=et.ROW_MATCHED,
                   method="zero_shot_ols")
        assert abs(ols["spearman_fold0"] - 1.0) < 1e-12
        assert abs(ols["spearman_fold1"] + raw1) < 1e-12
        raw = _get(ft, model="A", row_set=et.ROW_MATCHED,
                   method="zero_shot_raw")
        assert abs(raw["spearman_fold1"] - raw1) < 1e-12

        ft1 = spearmanr([FT1[x] for x in t], t)[0]
        for mode in et.OLS_WARM_MODES:
            r = _get(ft, model="A", row_set=et.ROW_MATCHED, method=mode)
            assert r["n_rows"] == 5 and r["starts_from_ols_head"]
            assert abs(r["spearman_fold0"] - 1.0) < 1e-12
            assert abs(r["spearman_fold1"] - ft1) < 1e-12
            assert abs(r["spearman_sd_pop"] - np.std([1.0, ft1])) < 1e-12
            # Full test rows = the model's own: 6 SLDB-filtered, 7 in total.
            assert _get(ft, model="A", row_set=et.ROW_ALL_SLDB,
                        method=mode)["n_rows"] == 6
            assert _get(ft, model="A", row_set=et.ROW_ALL,
                        method=mode)["n_rows"] == 7


def test_scratch_mode_is_an_extra_column_and_changes_nothing_else():
    """eval_finetuning's opt-in Scratch mode (the from-scratch baseline) is
    one more method of the fine-tuning table, flagged as not starting from
    the pretrained OLS head; without its CSV every output file is
    byte-identical to a run that never knew the mode, and with it the other
    rows are unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v0 = _write_variant(root / "plain" / "A", ROWS)
        v1 = _write_variant(root / "scratch" / "A", ROWS,
                            modes=et.OLS_WARM_MODES + (et.SCRATCH_MODE, ))
        # the same eval fingerprint in both dirs (the checkpoint bytes name
        # the dir's basename, "A", in both)
        res0 = _run(root / "plain", [("A", v0)], baseline=False, degree=False)
        res1 = _run(root / "scratch", [("A", v1)], baseline=False,
                    degree=False)
        ft0, ft1 = res0["finetune_summary"], res1["finetune_summary"]
        assert et.SCRATCH_MODE not in set(ft0["method"])
        r = _get(ft1, model="A", row_set=et.ROW_MATCHED,
                 method=et.SCRATCH_MODE)
        assert r["n_rows"] == 5 and not r["starts_from_ols_head"]
        assert all(_get(ft1, model="A", row_set=et.ROW_MATCHED,
                        method=m)["starts_from_ols_head"]
                   for m in et.OLS_WARM_MODES)
        rest = ft1[ft1["method"] != et.SCRATCH_MODE].reset_index(drop=True)
        pd.testing.assert_frame_equal(rest, ft0.reset_index(drop=True))
        md0 = (root / "plain" / "out" / "summary.md").read_text()
        md1 = (root / "scratch" / "out" / "summary.md").read_text()
        assert "Scratch" not in md0
        assert "**Scratch** is the from-scratch baseline" in md1
        assert f"| {SCREEN} | {et.SCRATCH_MODE} |" in md1
        # zero-shot tables do not depend on the fine-tuning modes at all
        for f in ("row_sets.csv", "zero_shot_per_fold.csv",
                  "zero_shot_summary.csv"):
            assert (root / "plain" / "out" / f).read_bytes() == \
                (root / "scratch" / "out" / f).read_bytes(), f


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_nonfinite_matched_logit_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS, logit1={0.3: float("nan")})
        _raises(lambda: _run(root, [("A", v)]), ValueError, "logit_fold1",
                "non-finite")


def test_nonfinite_paper_convention_logit_raises():
    # 10|99 (gene 99 not in genes.txt) is NOT matched, but it IS an
    # all_sldb_filtered row -- the row set paper_tables publishes as SLeuth's
    # paper_convention column. Dropping it silently would average folds over
    # different rows under one n_rows.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS, logit1={0.7: float("nan")})
        _raises(lambda: _run(root, [("A", v)]), ValueError, "logit_fold1",
                "non-finite", et.ROW_ALL_SLDB)


def test_nonfinite_outside_reported_rows_is_tolerated():
    # 30|99 is an SLDB-overlap test row with a gene outside genes.txt: it
    # lives in all_unfiltered only, the one row set no table reads, so its NaN
    # may be dropped silently. (10|40, SLDB with both genes in genes.txt, is
    # in matched_incl_sldb / matched_in_sldb since 2026-11-12: reported.)
    rows = ROWS + [("30", "99", 0.95, True, "test")]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", rows, logit1={0.95: float("nan")})
        zs = _run(root, [("A", v)])["zero_shot_summary"]
        assert _get(zs, model="A", row_set=et.ROW_MATCHED)["n_rows"] == 5
        assert _get(zs, model="A", row_set=et.ROW_ALL_SLDB)["n_rows"] == 6


def test_nonfinite_on_reported_sldb_row_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS, logit1={0.9: float("nan")})
        _raises(lambda: _run(root, [("A", v)], baseline=False, degree=False),
                ValueError, "logit_fold1", "matched_incl_sldb")


def test_nonfinite_matched_target_raises():
    rows = [r if _key(r[0], r[1]) != "20|40" else
            (r[0], r[1], float("nan")) + r[3:] for r in ROWS]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", rows)
        _raises(lambda: _run(root, [("A", v)], baseline=False, degree=False),
                ValueError, "target", "non-finite")


def test_nonfinite_baseline_score_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        genes, bdir, folds = _write_common(root)
        m = np.load(bdir / "fold_1_cv3_predictions.npy")
        m[GIDX["30"], GIDX["60"]] = np.nan  # a matched pair (30|60)
        np.save(bdir / "fold_1_cv3_predictions.npy", m)
        _raises(lambda: et.run([("A", v)], [("Base", bdir)], False, genes,
                               folds, root / "out", "cv3", [SCREEN], [SPLIT],
                               allow_unstamped=True),
                ValueError, "Base fold 1", "non-finite")


def test_asymmetric_baseline_is_symmetrised():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS, swap=True)
        genes, _bdir, folds = _write_common(root)
        adir = root / "asym"
        adir.mkdir()
        rng = np.random.default_rng(0)
        mats = {}
        for k in (0, 1):
            m = rng.normal(size=(len(GENES), len(GENES))).astype(np.float32)
            mats[k] = m
            np.save(adir / f"fold_{k}_cv3_predictions.npy", m)
        assert not np.allclose(mats[0], mats[0].T)
        zs = et.run([("A", v)], [("Asym", adir)], False, genes, folds,
                    root / "out", "cv3", [SCREEN], [SPLIT],
                    allow_unstamped=True)["zero_shot_summary"]
        t = np.array([r[2] for r in MATCHED])
        rec = _get(zs, model="Asym", row_set=et.ROW_MATCHED)
        differs = []
        for k, m in mats.items():
            m64 = m.astype(np.float64)
            sym = [0.5 * (m64[GIDX[a], GIDX[b]] + m64[GIDX[b], GIDX[a]])
                   for a, b, *_ in MATCHED]
            one = [m64[GIDX[a], GIDX[b]] for a, b, *_ in MATCHED]
            assert abs(rec[f"spearman_fold{k}"] - spearmanr(sym, t)[0]) < 1e-12
            differs.append(abs(spearmanr(one, t)[0] - spearmanr(sym, t)[0]))
        # Guard the fixture: reading one triangle must give a different answer
        # on at least one fold, or this test could not tell the two apart.
        assert max(differs) > 1e-6, differs


def test_missing_baseline_fold_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _raises(lambda: _run(root, [("A", v)], degree=False, base_folds=(0, )),
                ValueError, "baseline Base", "[0]", "[0, 1]")


def test_missing_degree_fold_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _raises(lambda: _run(root, [("A", v)], baseline=False,
                             deg_folds=(1, )),
                ValueError, et.DEGREE_PRIOR_NAME, "[1]", "[0, 1]")


def test_model_cv_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS, model_cv="cv1")
        _raises(lambda: _run(root, [("A", v)]), ValueError, "cv1", "cv3")
        (v / "config.json").unlink()
        (root / "again").mkdir()
        _raises(lambda: _run(root / "again", [("A", v)]), FileNotFoundError,
                "config.json")


def test_missing_default_baselines_raise_unless_allowed():
    saved = et.REPO_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        try:
            et.REPO_ROOT = Path(tmp)
            (Path(tmp) / "SLGNN" / "results" / "slgnn_cv3").mkdir(parents=True)
            _raises(lambda: et.default_baselines("cv3"), FileNotFoundError,
                    "SLMGAE", "NSF4SL", "--allow_missing_baselines")
            got = et.default_baselines("cv3", allow_missing=True)
            assert [n for n, _ in got] == ["SLGNN"], got
        finally:
            et.REPO_ROOT = saved


def test_provenance_prefers_effective_head():
    cfg = {"per_head": True, "dataset_heads": {"Adamson": "K562"},
           "effective_head": {"Adamson": "ANY (pooled)"},
           "model_num_heads": 1}
    txt = et.heads_provenance(cfg)
    assert "effective_head" in txt and "ANY (pooled)" in txt
    assert "K562" not in txt
    old = et.heads_provenance({"per_head": True,
                               "dataset_heads": {"Adamson": "K562"}})
    assert "dataset_heads" in old and "older eval" in old
    assert et.heads_provenance({}) == "config_used.json not found"


# ---------------------------------------------------------------------------
# main() smoke test
# ---------------------------------------------------------------------------
def test_main_smoke_explicit_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        genes, bdir, folds = _write_common(root)
        out = root / "cli_out"
        rc = et.main(["--siamese", f"A={v}", "--baseline", f"Base={bdir}",
                      "--genes", str(genes), "--folds_dir", str(folds),
                      "--model_cv", "cv3", "--screens", SCREEN,
                      "--splits", SPLIT, "--out_dir", str(out),
                      "--allow_unstamped"])
        assert rc == 0
        for f in ("row_sets.csv", "zero_shot_per_fold.csv",
                  "zero_shot_summary.csv", "finetune_per_fold.csv",
                  "finetune_summary.csv", "summary.md"):
            assert (out / f).exists(), f
        zs = pd.read_csv(out / "zero_shot_summary.csv")
        assert set(zs["model"]) == {"A", "Base", et.DEGREE_PRIOR_NAME}
        base = _get(zs, model="Base", row_set=et.ROW_MATCHED)
        assert abs(base["spearman_fold0"] - 1.0) < 1e-12
        md = (out / "summary.md").read_text()
        assert "only bear on fine-tuning" in md
        assert "(M[i,j] + M[j,i]) / 2" in md


def test_pca_arm_labels_every_written_file():
    """--pca_arm / --pca_label (label only, 2026-10-21; run_external_transfer.sh
    passes the SLeuth combos' PCA arm): every CSV written gains a trailing
    pca_arm column, summary.md says 'PCA arm: <arm> (<label>)' on line 3,
    under its title, and in its Provenance; nothing else changes, and the
    returned frames are those of a run without the flags. --pca_label
    without --pca_arm, or an unknown arm, is refused."""
    import contextlib
    import io
    names = ("row_sets.csv", "zero_shot_per_fold.csv", "zero_shot_summary.csv",
             "finetune_per_fold.csv", "finetune_summary.csv")
    label = "PCA on the concatenation of all modalities (variance 0.8)"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        genes, bdir, folds = _write_common(root)
        argv = ["--siamese", f"A={v}", "--baseline", f"Base={bdir}",
                "--genes", str(genes), "--folds_dir", str(folds),
                "--model_cv", "cv3", "--screens", SCREEN, "--splits", SPLIT,
                "--allow_unstamped"]
        with contextlib.redirect_stdout(io.StringIO()):
            assert et.main(argv + ["--out_dir", str(root / "plain")]) == 0
            assert et.main(argv + ["--pca_arm", "joint", "--pca_label", label,
                                   "--out_dir", str(root / "arm")]) == 0
        for n in names:
            plain = (root / "plain" / n).read_text().splitlines()
            arm = (root / "arm" / n).read_text().splitlines()
            assert len(arm) == len(plain) > 1, n
            assert arm[0] == plain[0] + ",pca_arm", (n, arm[0])
            assert arm[1:] == [ln + ",joint" for ln in plain[1:]], n
            b = pd.read_csv(root / "arm" / n)
            assert b.drop(columns="pca_arm").equals(
                pd.read_csv(root / "plain" / n)), n
        plain = (root / "plain" / "summary.md").read_text().splitlines()
        arm = (root / "arm" / "summary.md").read_text().splitlines()
        assert arm[0] == plain[0] == \
            "# External-screen transfer (zero-shot + fine-tuning)"
        assert arm[2] == f"PCA arm: joint ({label})" and arm[3] == "", arm[:5]
        bullet = f"- **PCA arm of the SLeuth combos**: joint ({label})"
        assert bullet in arm and not any("PCA arm" in ln for ln in plain)
        assert arm[:2] + [ln for ln in arm[4:] if ln != bullet] == plain
        # the frames run() returns are never the labelled copies
        a = _run(root / "r1", [("A", _write_variant(root / "r1" / "A", ROWS))])
        b = et.run([("A", root / "r1" / "A")], [("Base", root / "r1" / "base")],
                   True, root / "r1" / "genes.txt", root / "r1" / "folds",
                   root / "r2", "cv3", [SCREEN], [SPLIT], allow_unstamped=True,
                   pca_arm="none", pca_label="no PCA")
        for k in ("row_sets", "zero_shot_per_fold", "zero_shot_summary",
                  "finetune_per_fold", "finetune_summary"):
            assert "pca_arm" not in b[k].columns and b[k].equals(a[k]), k
        assert b["markdown"].splitlines()[2] == "PCA arm: none (no PCA)"
        assert (root / "r2" / "zero_shot_summary.csv").read_text() \
            .splitlines()[0].endswith(",pca_arm")
        # a label alone is refused, as is an unknown arm
        _raises(lambda: et.run([("A", root / "r1" / "A")], [], True,
                               root / "r1" / "genes.txt",
                               root / "r1" / "folds", root / "r3", "cv3",
                               [SCREEN], [SPLIT], allow_unstamped=True,
                               pca_label="no PCA"),
                ValueError, "pca_label needs pca_arm")
        assert not (root / "r3").exists()
        for extra in (["--pca_label", label], ["--pca_arm", "pca_joint"]):
            with contextlib.redirect_stderr(io.StringIO()):
                _raises(lambda: et.main(argv + extra + [
                    "--out_dir", str(root / "r4")]), SystemExit)
            assert not (root / "r4").exists(), extra


def _rewrite_ft(v: Path, fn):
    """Apply fn(DataFrame) -> DataFrame to every fine-tuned CSV of variant v."""
    for f in sorted((v / "eval_finetuning_cv1").glob(
            f"{SCREEN}_finetuned_predictions_*.csv")):
        fn(pd.read_csv(f)).to_csv(f, index=False)


def test_finetune_target_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _rewrite_ft(v, lambda d: d.assign(target=-d["target"]))  # sign flip
        _raises(lambda: _run(root, [("A", v)]), ValueError, "target differs")


def test_finetune_fold_set_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _rewrite_ft(v, lambda d: d.drop(columns=["pred_fold1"]))  # 1 of 2 folds
        _raises(lambda: _run(root, [("A", v)]), ValueError,
                "pred_fold columns")


def test_fold_stamp_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        genes, bdir, folds = _write_common(root)
        (bdir / "model_params.json").write_text(json.dumps(
            {"fold_stamp": {"genes.txt": "0" * 16}}))
        _raises(lambda: et.run([("A", v)], [("Base", bdir)], False, genes,
                               folds, root / "out", "cv3", [SCREEN], [SPLIT],
                               allow_unstamped=True),
                SystemExit)


def test_shell_gate_leaves_a_stale_model_absent():
    # run() refuses the WHOLE run on one stale input (above), so the SLURM
    # gate (run_external_transfer.sh stamp_diffs) must drop such a model from
    # both loops first: empty output == on the current folds.
    import shutil
    import subprocess
    sh = Path(_REPO, "sl_comparison", "slurm", "run_external_transfer.sh")
    text = sh.read_text()
    start = text.index("stamp_diffs() {")
    func = text[start:text.index("\n}\n", start) + 3]
    for dvar, arg in (("$DIR", "--siamese"), ("$BD", "--baseline")):
        i_chk = text.index(f'SD=$(stamp_diffs "{dvar}")')
        assert i_chk < text.index(f'ARGS {arg} ', i_chk), (dvar, arg)
    spec = importlib.util.spec_from_file_location(
        "_mp_gate", os.path.join(_REPO, "sl_comparison", "model_params.py"))
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sl_comparison" / "folds").mkdir(parents=True)
        shutil.copy(os.path.join(_REPO, "sl_comparison", "model_params.py"),
                    root / "sl_comparison")
        folds = root / "sl_comparison" / "folds"
        (folds / "genes.txt").write_text("\n".join(GENES) + "\n")
        np.save(folds / "all_pos.npy", np.array([[0, 1]], dtype=np.int32))
        (folds / "meta.json").write_text('{"seed": 42}')
        cur = mp.fold_stamp(str(folds), "cv3")
        stale = dict(cur, **{"meta.json": "0" * 16})
        for name, st in (("cur", cur), ("stale", stale), ("none", None)):
            (root / name).mkdir()
            if st is not None:
                (root / name / "model_params.json").write_text(
                    json.dumps({"fold_stamp": st}))

        def diffs(d, py=sys.executable):
            r = subprocess.run(["bash", "-c", func + 'stamp_diffs "$1"', "_",
                                d], cwd=root, capture_output=True, text=True,
                               env=dict(os.environ, PYTHON_PATH=py,
                                        MODEL_CV="cv3"), timeout=60)
            assert r.returncode == 0, r.stderr
            return r.stdout

        assert diffs("cur") == ""
        assert "meta.json" in diffs("stale")
        assert "no fold_stamp" in diffs("none")
        # a check that cannot run counts as stale
        assert "fold-stamp check failed" in diffs("cur", py="/nonexistent/py")


def test_unstamped_refused_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        genes, bdir, folds = _write_common(root)
        _raises(lambda: et.run([("A", v)], [("Base", bdir)], False, genes,
                               folds, root / "out", "cv3", [SCREEN], [SPLIT]),
                SystemExit)


# ---------------------------------------------------------------------------
# Runner (works without pytest)
# ---------------------------------------------------------------------------
def test_ft_mode_not_in_last_eval_run_raises():
    """A fine-tuning CSV for a mode the last eval_finetuning run did not
    produce (config_used.json ft_modes) is left over and must be refused."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        (root / "ok").mkdir()
        (root / "stale").mkdir()
        _run(root / "ok", [("A", v)], baseline=False, degree=False)
        _edit_eval_cfg(v, ft_modes=list(et.OLS_WARM_MODES)[:1])
        _raises(lambda: _run(root / "stale", [("A", v)], baseline=False,
                             degree=False), ValueError, "left over")


def test_overlap_against_positives_only_raises():
    """sldb_filtered rows are leakage-filtered only if sldb_overlap also
    matched the screened negatives (eval_finetuning --neg_path)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _edit_eval_cfg(v, sldb_overlap_sources=["sl.txt"])
        _raises(lambda: _run(root, [("A", v)], baseline=False, degree=False),
                ValueError, "positives only", "--neg_path")


def test_genes_not_the_fold_sets_genes_txt_raises():
    """A same-size but reordered --genes would put scores on the wrong pairs
    while passing every provenance check; run() must refuse it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        genes, bdir, folds = _write_common(root)
        genes.write_text("\n".join(reversed(GENES)) + "\n")
        _raises(lambda: et.run([("A", v)], [("Base", bdir)], True, genes,
                               folds, root / "out", "cv3", [SCREEN], [SPLIT],
                               allow_unstamped=True), ValueError,
                "is not", "genes.txt")


# ---------------------------------------------------------------------------
# Eval fingerprint (config_used.json <-> checkpoints/fold_*_best.pt)
# ---------------------------------------------------------------------------
def test_fingerprint_verified_and_disclosed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _run(root, [("A", v)], baseline=False, degree=False)
        md = (root / "out" / "summary.md").read_text()
        assert "verified for every SLeuth variant and split" in md


def test_retrained_checkpoint_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        (v / "checkpoints" / "fold_1_best.pt").write_bytes(b"retrained")
        _raises(lambda: _run(root, [("A", v)], baseline=False, degree=False),
                ValueError, "fold_1_best.pt", "retrained since",
                "--allow_legacy_eval")


def test_different_checkpoint_set_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        (v / "checkpoints" / "fold_2_best.pt").write_bytes(b"extra fold")
        _raises(lambda: _run(root, [("A", v)], baseline=False, degree=False),
                ValueError, "scored checkpoints", "fold_2_best.pt")


def test_fold_ids_must_be_every_fold_unrenumbered():
    for fids, needle in (([1], "are not 0..1"), ([1, 0], "are not 0..1"),
                         (None, "records no fold_ids")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v = _write_variant(root / "A", ROWS)
            _edit_eval_cfg(v, fold_ids=fids)
            _raises(lambda: _run(root, [("A", v)], baseline=False,
                                 degree=False), ValueError, needle)


def test_logit_columns_must_match_fold_ids():
    # Three checkpoints and fold_ids 0..2, but the CSVs carry two folds.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        f = v / "checkpoints" / "fold_2_best.pt"
        f.write_bytes(b"third fold")
        cfg = json.loads((v / "eval_finetuning_cv1" /
                          "config_used.json").read_text())
        hashes = dict(cfg["checkpoint_sha256"])
        hashes["results/A/checkpoints/fold_2_best.pt"] = _sha256(f)
        _edit_eval_cfg(v, fold_ids=[0, 1, 2], checkpoint_sha256=hashes)
        _raises(lambda: _run(root, [("A", v)], baseline=False, degree=False),
                ValueError, "logit_fold columns", "fold_ids")


def test_missing_eval_config_raises_unless_legacy_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        (v / "eval_finetuning_cv1" / "config_used.json").unlink()
        (root / "strict").mkdir()
        (root / "legacy").mkdir()
        _raises(lambda: _run(root / "strict", [("A", v)], baseline=False,
                             degree=False), ValueError,
                "config_used.json missing", "--allow_legacy_eval")
        zs = _run(root / "legacy", [("A", v)], baseline=False, degree=False,
                  allow_legacy_eval=True)["zero_shot_summary"]
        assert _get(zs, model="A", row_set=et.ROW_MATCHED)["n_rows"] == 5
        md = (root / "legacy" / "out" / "summary.md").read_text()
        assert "NOT verified (--allow_legacy_eval) for A/cv1" in md


def test_main_allow_legacy_eval_flag():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        _edit_eval_cfg(v, checkpoint_sha256=None)
        genes, bdir, folds = _write_common(root)
        argv = ["--siamese", f"A={v}", "--baseline", f"Base={bdir}",
                "--genes", str(genes), "--folds_dir", str(folds),
                "--model_cv", "cv3", "--screens", SCREEN, "--splits", SPLIT,
                "--out_dir", str(root / "cli_out"), "--allow_unstamped"]
        _raises(lambda: et.main(argv), ValueError, "no checkpoint_sha256")
        assert et.main(argv + ["--allow_legacy_eval"]) == 0


def test_failed_write_leaves_no_cut_file_under_a_final_name():
    """A write that fails part-way (a full quota) must not leave a cut CSV
    under its final name: paper_tables.py takes a line cut short as a row."""
    real = pd.DataFrame.to_csv
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _write_variant(root / "A", ROWS)
        calls = []

        def failing(self, path, *a, **k):
            calls.append(path)
            if len(calls) == 3:                     # zero_shot_summary.csv
                Path(path).write_text("model,kind\nA,sia")
                raise OSError(122, "Disk quota exceeded")
            return real(self, path, *a, **k)

        pd.DataFrame.to_csv = failing
        try:
            _raises(lambda: _run(root, [("A", v)]), OSError, "quota")
        finally:
            pd.DataFrame.to_csv = real
        left = sorted(p.name for p in (root / "out").iterdir())
        assert left == ["row_sets.csv", "zero_shot_per_fold.csv"], left


def _main():
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n==== {passed} passed, {failed} failed ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
