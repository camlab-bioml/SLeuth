#!/usr/bin/env python3
"""Deterministic tests for sl_comparison/bootstrap_ci.py (synthetic trees only).

  metrics_from_counts reproduces cal_metrics / average_precision_score /
      auprg.auprg on tied data, and on every expanded bootstrap resample.
  auprg.auprg_from_counts equals auprg.auprg (the refactor behind it).
  the weighted Spearman equals scipy.stats.spearmanr on expanded resamples.
  resample_counts is seeded per resample (prefix-stable in n_boot) and
      --stratify keeps the class sizes.
  tiers end to end on a synthetic fold set graded by evaluate_model: every
      fold / mean point equals the row JSON; a row whose scores are a monotone
      transform of another's has a paired difference of exactly 0 in every
      resample (the resamples are shared); a tampered JSON value or a missing
      matrix stops the run before anything is written; --skip_missing records
      the skipped row.
  external end to end on a tree written by external_transfer.run: every
      per-fold point equals zero_shot_per_fold.csv / finetune_per_fold.csv.

    python3 sl_comparison/test_bootstrap_ci.py
    # or: python -m pytest sl_comparison/test_bootstrap_ci.py -q
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import auprg as A  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import degree_baseline as db  # noqa: E402
import evaluate_model as em  # noqa: E402
import external_transfer as et  # noqa: E402


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _raises_exit(fn, *needles):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except SystemExit as e:
        msg = str(e)
        for n in needles:
            assert n in msg, f"{n!r} not in {msg[:500]!r}"
        return msg
    raise AssertionError("no SystemExit")


# =============================================================================
# Metric kernels
# =============================================================================


def _rand_case(rng):
    n = int(rng.integers(5, 250))
    y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(float)
    s = np.round(rng.normal(size=n) + y * rng.uniform(0, 2),
                 int(rng.integers(0, 3))).astype(np.float32)
    return y, s


def test_metrics_match_grader_calls_with_ties():
    rng = np.random.default_rng(0)
    n_cases = 0
    for _ in range(200):
        y, s = _rand_case(rng)
        if y.sum() in (0, len(y)):
            continue
        seg = bc.DescSegments(s)
        ip, ineg = bc.segment_increments(np.ones((1, len(y)), np.uint16), y,
                                         seg)
        got = bc.metrics_from_counts(ip[0], ineg[0])
        ref = bc.sklearn_reference(y, s)
        for k in bc.CLS_METRICS:
            assert abs(got[k] - ref[k]) < 1e-12, (k, got[k], ref[k])
        n_cases += 1
    assert n_cases > 150


def test_weighted_resample_equals_expanded_resample():
    rng = np.random.default_rng(1)
    checked = 0
    for t in range(80):
        y, s = _rand_case(rng)
        W = bc.resample_counts(len(y), 4, 9, (t, ))
        seg = bc.DescSegments(s)
        ip, ineg = bc.segment_increments(W, y, seg)
        for b in range(4):
            idx = np.repeat(np.arange(len(y)), W[b])
            got = bc.metrics_from_counts(ip[b], ineg[b])
            if y[idx].sum() in (0, len(idx)):
                assert all(np.isnan(v) for v in got.values())
                continue
            ref = bc.sklearn_reference(y[idx], s[idx])
            for k in bc.CLS_METRICS:
                assert abs(got[k] - ref[k]) < 1e-12, (k, got[k], ref[k])
            checked += 1
    assert checked > 200


def test_auprg_from_counts_equals_auprg():
    rng = np.random.default_rng(2)
    for _ in range(300):
        y, s = _rand_case(rng)
        if y.sum() in (0, len(y)):
            continue
        o = np.argsort(-s.astype(float), kind="stable")
        ss, ys = s[o], y[o]
        ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
        cp = np.cumsum(ys)[ends]
        cn = (ends + 1) - cp
        assert A.auprg_from_counts(cp, cn) == A.auprg(y, s)
    # degenerate prevalence
    assert np.isnan(A.auprg_from_counts([0, 0], [1, 2]))


def test_weighted_spearman_equals_scipy():
    rng = np.random.default_rng(3)
    worst = 0.0
    for t in range(120):
        n = int(rng.integers(4, 150))
        s = np.round(rng.normal(size=n), 1)
        tg = np.round(rng.normal(size=n) + s * rng.uniform(-1, 1), 1)
        W = bc.resample_counts(n, 4, 3, (t, ))
        pts, reps = bc.bootstrap_spearman(tg, {"m": s}, W, chunk=3)
        worst = max(worst, abs(pts["m"] - spearmanr(s, tg)[0]))
        for b in range(4):
            idx = np.repeat(np.arange(n), W[b])
            r = spearmanr(s[idx], tg[idx])[0]
            if np.isnan(r):
                assert np.isnan(reps["m"][b])
                continue
            worst = max(worst, abs(reps["m"][b] - r))
    assert worst < 1e-12, worst
    # constant score -> NaN (et.spearman's rule)
    pts, reps = bc.bootstrap_spearman(np.arange(10.0), {"c": np.ones(10)},
                                      bc.resample_counts(10, 3, 0, (0, )))
    assert np.isnan(pts["c"]) and np.isnan(reps["c"]).all()


def test_resample_counts_seeded_per_resample_and_stratified():
    a = bc.resample_counts(500, 5, 11, (1, 2))
    b = bc.resample_counts(500, 12, 11, (1, 2))
    assert np.array_equal(a, b[:5])
    assert (a.sum(axis=1) == 500).all()
    assert not np.array_equal(a, bc.resample_counts(500, 5, 12, (1, 2)))
    assert not np.array_equal(a, bc.resample_counts(500, 5, 11, (1, 3)))
    st = bc.resample_counts(500, 20, 11, (0, ), strata=[(0, 70), (70, 500)])
    assert (st[:, :70].sum(axis=1) == 70).all()
    assert (st[:, 70:].sum(axis=1) == 430).all()


# =============================================================================
# Tiers end to end
# =============================================================================

N_GENES = 60
N_FOLDS = 3


def _pairs(rng, n, exclude):
    out = set()
    while len(out) < n:
        i, j = sorted(rng.choice(N_GENES, 2, replace=False).tolist())
        if (i, j) not in exclude:
            out.add((i, j))
    return np.array(sorted(out), dtype=np.int64)


def _write_tier_tree(root: Path):
    """folds/, three model prediction dirs, graded JSON rows, a marker."""
    rng = np.random.default_rng(7)
    folds = root / "folds"
    (folds / "cv1").mkdir(parents=True)
    (folds / "genes.txt").write_text(
        "\n".join(str(1000 + g) for g in range(N_GENES)) + "\n")
    all_pos = _pairs(rng, 150, set())
    np.save(folds / "all_pos.npy", all_pos)
    perm = rng.permutation(len(all_pos))
    used = set(map(tuple, all_pos.tolist()))
    negs = _pairs(rng, 450, used)
    for k in range(N_FOLDS):
        d = folds / "cv1" / f"fold_{k}"
        d.mkdir()
        te = all_pos[perm[k::N_FOLDS]]
        tr = all_pos[np.setdiff1d(perm, perm[k::N_FOLDS])]
        np.save(d / "test_pos.npy", te)
        np.save(d / "train_pos.npy", tr)
        np.save(d / "test_neg.npy", negs[k::N_FOLDS])
        np.save(d / "train_neg.npy", negs[:0])
    preds = {}
    truth = np.zeros((N_GENES, N_GENES))
    truth[all_pos[:, 0], all_pos[:, 1]] = 1
    truth = truth + truth.T
    for name, scale in (("peer", 1.0), ("sia", 2.0)):
        pd_ = root / f"pred_{name}"
        pd_.mkdir()
        for k in range(N_FOLDS):
            # negative raw scores (exercises the affine shift) with ties
            m = np.round(rng.normal(size=(N_GENES, N_GENES)) +
                         scale * truth - 5.0, 1).astype(np.float32)
            np.save(pd_ / f"fold_{k}_cv1_predictions.npy", m)
        preds[name] = pd_
    # a monotone transform of the peer: identical ranks on every resample
    mono = root / "pred_mono"
    mono.mkdir()
    for k in range(N_FOLDS):
        m = np.load(preds["peer"] / f"fold_{k}_cv1_predictions.npy")
        np.save(mono / f"fold_{k}_cv1_predictions.npy",
                (3.0 * m + 7.0).astype(np.float32))
    preds["mono"] = mono

    res = root / "results"
    res.mkdir()
    fdirs = em.fold_dirs_for(folds, "cv1")
    rows = {"SLGNN": preds["peer"], "siamese_sl": preds["sia"],
            "siamese_sl_pd_pooled": preds["mono"]}
    for model, pdir in rows.items():
        fm = [_quiet(em.score_fold, np.load(pdir / f"fold_{k}_cv1_"
                                            "predictions.npy"), d, model)
              for k, d in enumerate(fdirs)]
        summ = _quiet(em.build_summary, model, "cv1", str(folds), fm)
        _quiet(em.write_summary, summ, res / f"{model}_cv1.json")
    fm = [_quiet(em.score_fold, db.degree_matrix(d, N_GENES), d, "DP")
          for d in fdirs]
    _quiet(em.write_summary,
           _quiet(em.build_summary, "DegreePrior", "cv1", str(folds), fm),
           res / "DegreePrior_cv1.json")
    (res / "siamese_selected_cv1.txt").write_text(str(preds["sia"]) + "\n")
    (res / "siamese_sl_pd_pooled_selected_cv1.txt").write_text(
        str(preds["mono"]) + "\n")
    return folds, res, preds


def _tier_argv(folds, res, out, *extra):
    return ["--part", "tiers", "--cvs", "cv1", "--folds_dir", str(folds),
            "--results_dir", str(res), "--shared_results_dir", str(res),
            "--pred_dir", f"SLGNN={res.parent / 'pred_peer'}",
            "--allow_unstamped", "--allow_no_tables", "--n_boot", "60",
            "--seed", "5", "--out_dir", str(out), *extra]


def test_tiers_end_to_end_reproduces_and_pairs():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, res, _ = _write_tier_tree(root)
        out = root / "boot"
        assert _quiet(bc.main, _tier_argv(folds, res, out)) == 0
        ci = pd.read_csv(out / "ci_tiers.csv")
        assert set(ci["model"]) == {"SLGNN", "siamese_sl",
                                    "siamese_sl_pd_pooled", "DegreePrior"}
        need = {"model", "variant", "pca_arm", "cv", "fold", "metric",
                "point", "ci_low", "ci_high", "n_boot"}
        assert need <= set(ci.columns)
        assert (ci["point"] - ci["published"]).abs().max() < 1e-12
        assert set(ci["fold"].astype(str)) == {"0", "1", "2", "mean"}
        assert (ci["n_boot"] == 60).all()
        ok = ci.dropna(subset=["ci_low"])
        assert ((ok["ci_low"] <= ok["boot_mean"] + 1e-12) &
                (ok["boot_mean"] <= ok["ci_high"] + 1e-12)).all()
        # pooled point = mean of the per-fold points
        for (m, met), g in ci.groupby(["model", "metric"]):
            per = g[g["fold"] != "mean"]["point"].mean()
            assert abs(per - g[g["fold"] == "mean"]["point"].iloc[0]) < 1e-12
        d = pd.read_csv(out / "ci_tiers_diffs.csv")
        # each SLeuth row vs each peer (SLGNN, DegreePrior)
        assert set(zip(d["model"], d["peer"])) == {
            ("siamese_sl", "SLGNN"), ("siamese_sl", "DegreePrior"),
            ("siamese_sl_pd_pooled", "SLGNN"),
            ("siamese_sl_pd_pooled", "DegreePrior")}
        mono = d[(d["model"] == "siamese_sl_pd_pooled") &
                 (d["peer"] == "SLGNN")]
        assert len(mono) == len(bc.CLS_METRICS) * (N_FOLDS + 1)
        # identical ranks + shared resamples -> difference 0 in EVERY resample
        assert (mono[["point", "ci_low", "ci_high"]].abs() < 1e-12).all().all()
        assert (mono["p_boot_le0"] == 1.0).all()
        man = json.loads((out / "manifest.json").read_text())
        assert man["n_boot"] == 60 and man["seed"] == 5
        # rerun: identical numbers
        out2 = root / "boot2"
        _quiet(bc.main, _tier_argv(folds, res, out2))
        assert pd.read_csv(out2 / "ci_tiers.csv").equals(ci)


def test_tiers_published_mismatch_refuses_to_write():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, res, _ = _write_tier_tree(root)
        p = res / "SLGNN_cv1.json"
        j = json.loads(p.read_text())
        j["summary"]["folds"][1]["ap"] += 1e-4
        p.write_text(json.dumps(j))
        out = root / "boot"
        _raises_exit(lambda: bc.main(_tier_argv(folds, res, out)),
                     "REFUSING TO WRITE", "SLGNN/cv1/fold_1/ap")
        assert not out.exists() or not any(out.iterdir())


def test_tiers_missing_matrix_fails_loudly_or_is_skipped():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, res, preds = _write_tier_tree(root)
        (preds["sia"] / "fold_2_cv1_predictions.npy").unlink()
        out = root / "boot"
        _raises_exit(lambda: bc.main(_tier_argv(folds, res, out)),
                     "MISSING INPUTS", "siamese_sl/cv1",
                     "1 of 3 score matrices missing")
        assert not out.exists()
        # an unknown row without --pred_dir is reported, too
        argv = [a for a in _tier_argv(folds, res, out)]
        i = argv.index("--pred_dir")
        del argv[i:i + 2]
        # SLGNN's default dir (the repo's own) holds no matrices for this
        # 3-fold tree
        _raises_exit(lambda: bc.main(argv), "SLGNN/cv1: ")
        _quiet(bc.main, _tier_argv(folds, res, out, "--skip_missing"))
        ci = pd.read_csv(out / "ci_tiers.csv")
        assert "siamese_sl" not in set(ci["model"])
        man = json.loads((out / "manifest.json").read_text())
        assert any("siamese_sl/cv1" in s for s in man["skipped"])


def test_tiers_marker_missing_is_reported():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, res, _ = _write_tier_tree(root)
        (res / "siamese_selected_cv1.txt").unlink()
        _raises_exit(lambda: bc.main(_tier_argv(folds, res, root / "b")),
                     "MISSING INPUTS", "siamese_selected_cv1.txt")


def test_resolve_pred_dir_conventions():
    cmp_dir = Path("/x/cmp")
    assert bc.resolve_pred_dir("SLGNN", "cv2", "none", cmp_dir, {}) == \
        bc.REPO_ROOT / "SLGNN/results/slgnn_cv2"
    assert bc.resolve_pred_dir("DegreePrior", "cv2", "none", cmp_dir,
                               {}) is None
    # The bilinear probe (siamese_sl/slurm/run_probe.sh): arm none's siamese
    # root, whatever the arm (a shared row, trained once).
    for arm in ("none", "joint"):
        assert bc.resolve_pred_dir("BilinearProbe", "cv3", arm, cmp_dir,
                                   {}) == \
            bc.REPO_ROOT / "siamese_sl/results/bilinear_probe_cv3"
    # ELISL (ELISL/slurm/run_elisl.sh): its own launcher's output dir, for
    # every arm.
    for arm in ("none", "both"):
        assert bc.resolve_pred_dir("ELISL", "cv2", arm, cmp_dir, {}) == \
            bc.REPO_ROOT / "ELISL/results/elisl_cv2"
    assert bc.resolve_pred_dir(
        "siamese_sl_bilinear_pooled_nocell", "cv3", "joint", cmp_dir, {}) == \
        bc.REPO_ROOT / ("siamese_sl/results/pca_joint/var_bilinear_pooled__"
                        "cellline_ablation_cv3")
    assert bc.resolve_pred_dir("siamese_sl_nocell", "cv1", "none", cmp_dir,
                               {}) == \
        bc.REPO_ROOT / "siamese_sl/results/cellline_ablation_cv1"
    assert bc.resolve_pred_dir("X", "cv1", "none", cmp_dir,
                               {"X": "/p/{cv}"}) == Path("/p/cv1")
    assert bc.variant_of("siamese_sl") == "pd_per_head"
    assert bc.variant_of("siamese_sl_bilinear_pooled_nocell") == \
        "bilinear_pooled_nocell"
    assert bc.variant_of("SLGNN") == ""


# =============================================================================
# External end to end (tree written by external_transfer.run itself)
# =============================================================================

EXT_GENES = [str(5000 + 7 * g) for g in range(40)]


def _write_ext_tree(root: Path, sldb: bool = False, scratch: bool = False):
    """scratch: eval_finetuning also ran --ft_modes ... Scratch (its CSV and
    config_used.json ft_modes). sldb: also write SynLethDB positives / negatives consistent with the
    sldb_overlap flag (every 3rd flagged pair SL) and pass them to et.run,
    so the SynLethDB_label row and sldb_agreement.csv are written."""
    rng = np.random.default_rng(11)
    folds = root / "folds"
    (folds / "cv3").mkdir(parents=True)
    (folds / "genes.txt").write_text("\n".join(EXT_GENES) + "\n")
    gidx = {g: i for i, g in enumerate(EXT_GENES)}
    for k in range(2):
        d = folds / "cv3" / f"fold_{k}"
        d.mkdir()
        tp = np.array(sorted({tuple(sorted(rng.choice(40, 2, replace=False)))
                              for _ in range(60)}), dtype=np.int64)
        np.save(d / "train_pos.npy", tp)
    # screen pairs: some genes outside genes.txt, some SLDB-overlapping
    genes_all = EXT_GENES + ["99991", "99992", "99993"]
    pairs = set()
    while len(pairs) < 220:
        a, b = rng.choice(len(genes_all), 2, replace=False)
        pairs.add(tuple(sorted((genes_all[a], genes_all[b]))))
    pairs = sorted(pairs)
    target = np.round(rng.normal(size=len(pairs)), 2)
    split = rng.choice(["train", "val", "test", "test"], size=len(pairs))
    overlap = rng.random(len(pairs)) < 0.1

    def variant(name, noise):
        v = root / name
        e = v / "eval_finetuning_cv1"
        e.mkdir(parents=True)
        (v / "config.json").write_text(json.dumps({"cv_type": "cv3"}))
        (v / "checkpoints").mkdir()
        (e / "config_used.json").write_text(json.dumps(
            {"ft_modes": ["LP_last", "Full"] + (["Scratch"] if scratch
                                                else []),
             "sldb_overlap_sources": ["a", "b"], "fold_ids": [0, 1]}))
        df = pd.DataFrame({
            "gene1_entrez": [p[0] for p in pairs],
            "gene2_entrez": [p[1] for p in pairs],
            "outcome_raw": -target, "target": target,
            "sldb_overlap": overlap, "split": split})
        for k in range(2):
            df[f"logit_fold{k}"] = np.round(
                target * (0.5 - k) + noise * rng.normal(size=len(df)), 1)
        df.to_csv(e / "Toy_predictions.csv", index=False)
        test = df[df["split"] == "test"]
        for mode in ("LP_last", "Full") + (("Scratch", ) if scratch else ()):
            ft = test[["gene1_entrez", "gene2_entrez", "outcome_raw",
                       "target", "sldb_overlap"]].copy()
            for k in range(2):
                ft[f"pred_fold{k}"] = (test["target"].to_numpy() +
                                       rng.normal(size=len(test)))
            ft.to_csv(e / f"Toy_finetuned_predictions_{mode}.csv",
                      index=False)
        return v

    v1, v2 = variant("varA", 1.0), variant("varB", 0.5)
    base = root / "base"
    base.mkdir()
    for k in range(2):
        np.save(base / f"fold_{k}_cv3_predictions.npy",
                rng.normal(size=(40, 40)).astype(np.float32))
    ext = root / "ext"
    kw = {}
    if sldb:
        flagged = [p for p, o in zip(pairs, overlap) if o]
        sl, ng = root / "pos.txt", root / "neg.txt"
        sl.write_text("".join(f"{a}\t{b}\n" for a, b in flagged[::3]))
        ng.write_text("".join(f"{a}\t{b}\n" for i, (a, b) in
                              enumerate(flagged) if i % 3))
        kw = {"sl_path": sl, "neg_path": ng}
    _quiet(et.run, [("siamese_sl", v1), ("siamese_sl_pd_pooled", v2)],
           [("SLGNN", base)], True, folds / "genes.txt", folds, ext, "cv3",
           ["Toy"], ["cv1"], allow_unstamped=True, allow_legacy_eval=True,
           **kw)
    assert gidx  # (gene index used by et.run via genes.txt)
    return folds, ext


def test_external_end_to_end_reproduces_published():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, ext = _write_ext_tree(root)
        out = root / "boot"
        argv = ["--part", "external", "--folds_dir", str(folds),
                "--external_dir", str(ext), "--allow_unstamped",
                "--n_boot", "40", "--out_dir", str(out)]
        assert _quiet(bc.main, argv) == 0
        ci = pd.read_csv(out / "ci_external.csv")
        per = ci[ci["fold"] != "mean"]
        assert per["published"].notna().all()
        assert (per["point"] - per["published"]).abs().max() < 1e-12
        assert set(ci["model"]) == {"siamese_sl", "siamese_sl_pd_pooled",
                                    "SLGNN", "DegreePrior"}
        # no Scratch CSV (eval_finetuning's opt-in mode not run): no error,
        # no Scratch row
        assert set(ci[ci["kind"] == "siamese"]["method"]) == {
            "zero_shot_raw", "zero_shot_ols", "LP_last", "Full"}
        assert set(ci[ci["kind"] != "siamese"]["row_set"]) == \
            set(et.MATCHED_ROW_SETS)
        # the SynLethDB-keeping row sets carry zero_shot_raw only
        assert set(ci[ci["row_set"].isin(bc.EXT_ZS_ONLY_ROW_SETS)]
                   ["method"]) == {"zero_shot_raw"}
        # pooled = the published summary mean
        zs = pd.read_csv(ext / "zero_shot_summary.csv")
        for _, r in zs[zs["row_set"].isin(bc.EXT_ROW_SETS)].iterrows():
            got = ci[(ci["model"] == r["model"]) & (ci["fold"] == "mean") &
                     (ci["row_set"] == r["row_set"]) &
                     (ci["method"] == "zero_shot_raw")]["point"]
            assert len(got) == 1 and abs(got.iloc[0] -
                                         r["spearman_mean"]) < 1e-12
        d = pd.read_csv(out / "ci_external_diffs.csv")
        pairs = set(zip(d["model"], d["method"], d["peer"], d["peer_method"]))
        assert ("siamese_sl", "zero_shot_raw", "SLGNN",
                "zero_shot_raw") in pairs
        assert ("siamese_sl", "Full", "siamese_sl",
                "zero_shot_ols") in pairs
        # a tampered published value is refused
        p = ext / "zero_shot_per_fold.csv"
        z = pd.read_csv(p)
        z.loc[0, "spearman"] += 1e-3
        z.to_csv(p, index=False)
        _raises_exit(lambda: bc.main(argv[:-1] + [str(root / "b2")]),
                     "REFUSING TO WRITE")
        # a missing baseline matrix fails loudly
        z = pd.read_csv(p)
        z.loc[0, "spearman"] -= 1e-3
        z.to_csv(p, index=False)
        (root / "base" / "fold_1_cv3_predictions.npy").unlink()
        (root / "base" / "fold_0_cv3_predictions.npy").unlink()
        _raises_exit(lambda: bc.main(argv[:-1] + [str(root / "b3")]),
                     "MISSING INPUTS", "baseline SLGNN")


def test_external_scratch_bootstrapped_with_full_minus_scratch():
    """eval_finetuning --ft_modes ... Scratch: its rows are bootstrapped
    (reproducing finetune_per_fold.csv) and Full - Scratch is a default
    paired difference; Scratch is never differenced against zero_shot_ols.
    Without it (test_external_end_to_end_reproduces_published) nothing
    errors and no Scratch row is written."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, ext = _write_ext_tree(root, scratch=True)
        out = root / "boot"
        argv = ["--part", "external", "--folds_dir", str(folds),
                "--external_dir", str(ext), "--allow_unstamped",
                "--n_boot", "40", "--out_dir", str(out)]
        assert _quiet(bc.main, argv) == 0
        ci = pd.read_csv(out / "ci_external.csv")
        sc = ci[ci["method"] == "Scratch"]
        assert set(sc["model"]) == {"siamese_sl", "siamese_sl_pd_pooled"}
        per = sc[sc["fold"] != "mean"]
        assert len(per) and per["published"].notna().all()
        assert (per["point"] - per["published"]).abs().max() < 1e-12
        d = pd.read_csv(out / "ci_external_diffs.csv")
        pairs = set(zip(d["model"], d["method"], d["peer"], d["peer_method"]))
        for m in ("siamese_sl", "siamese_sl_pd_pooled"):
            assert (m, "Full", m, "Scratch") in pairs
            assert (m, "Scratch", m, "zero_shot_ols") not in pairs
        fs = d[(d["method"] == "Full") & (d["peer_method"] == "Scratch") &
               (d["fold"] == "mean")]
        full = ci[(ci["method"] == "Full") & (ci["fold"] == "mean")]
        scr = sc[sc["fold"] == "mean"]
        for _, r in fs.iterrows():
            key = ["model", "screen", "split", "row_set"]
            sel = (full[key] == r[key].to_numpy()).all(axis=1)
            sel2 = (scr[key] == r[key].to_numpy()).all(axis=1)
            assert abs(r["point"] - (full[sel]["point"].iloc[0] -
                                     scr[sel2]["point"].iloc[0])) < 1e-12


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
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


def test_external_sldb_label_and_agreement_cis():
    """The SynLethDB label row (matched_in_sldb) and ci_sldb_agreement.csv:
    every point reproduces external_transfer's published value, the CI
    brackets it, and the agreement resample covers the labelled pairs only."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, ext = _write_ext_tree(root, sldb=True)
        out = root / "boot"
        argv = ["--part", "external", "--folds_dir", str(folds),
                "--external_dir", str(ext), "--allow_unstamped",
                "--n_boot", "60", "--out_dir", str(out)]
        assert _quiet(bc.main, argv) == 0
        ci = pd.read_csv(out / "ci_external.csv")
        lab = ci[ci["model"] == et.SLDB_LABEL_NAME]
        assert set(lab["row_set"]) == {et.ROW_MATCHED_IN_SLDB}
        per = lab[lab["fold"] != "mean"]
        assert (per["point"] - per["published"]).abs().max() < 1e-12
        ag = pd.read_csv(out / "ci_sldb_agreement.csv")
        pub = pd.read_csv(ext / "sldb_agreement.csv")
        assert set(ag["metric"]) == set(bc.AGREEMENT_METRICS)
        m = ag.merge(pub, on=["screen", "scope", "split"])
        assert len(m) == len(ag) == 2 * len(pub)
        for met in bc.AGREEMENT_METRICS:
            x = m[m["metric"] == met]
            ok = x[met].notna()
            assert (x.loc[ok, "point"] - x.loc[ok, met]).abs().max() < 1e-12
            assert ((x.loc[ok, "ci_low"] <= x.loc[ok, "point"] + 1e-12) &
                    (x.loc[ok, "point"] <= x.loc[ok, "ci_high"] + 1e-12)).all()
        assert (m["n_labelled_x"] == m["n_labelled_y"]).all()


def test_tiers_refuse_without_paper_tables_and_skip_refused_rows():
    """Without the arm's paper_tables/tab_tiers.csv the tier part refuses
    (2026-11-13); with it, a row the tables refused (status != ok) is not
    bootstrapped."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        folds, res, _ = _write_tier_tree(root)
        argv = [a for a in _tier_argv(folds, res, root / "b0")
                if a != "--allow_no_tables"]
        _raises_exit(lambda: bc.main(argv), "tab_tiers.csv missing")
        tt = res / "paper_tables" / "tab_tiers.csv"
        tt.parent.mkdir(parents=True)
        tt.write_text("cv,model,status\ncv1,SLGNN,missing\n")
        jobs, _ = bc.tier_jobs("none", res, res, folds, ["cv1"], None,
                               {}, True)
        assert "SLGNN" not in {j["model"] for j in jobs}, jobs
