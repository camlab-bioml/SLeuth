#!/usr/bin/env python3
"""Deterministic tests for the shared evaluator and fold generator.

  evaluate_model.fold_provenance records git_sha / git_dirty as None (unknown)
      when git runs but exits non-zero, never as a clean tree.
  evaluate_model.build_summary / main record n_folds_expected and partial, so
      an --allow_partial mean identifies itself; without the flag it refuses.
  prepare_folds --check_sources exits 2, not 1 ("sources differ"), when a
      source cannot be read.

    python3 sl_comparison/test_comparison_core.py
    # or: python -m pytest sl_comparison/test_comparison_core.py -q
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_model as em  # noqa: E402
import prepare_folds as pf  # noqa: E402


@contextlib.contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _argv(argv):
    old, sys.argv = sys.argv, argv
    try:
        yield
    finally:
        sys.argv = old


def _pairs(*ps):
    return np.array(ps, dtype=np.int32).reshape(-1, 2)


def _folds(root: Path, n_folds=3) -> Path:
    """12-gene cv1 fold set; every fold has both test classes."""
    folds = root / "folds"
    (folds).mkdir(parents=True)
    (folds / "genes.txt").write_text("\n".join(str(g) for g in range(12)) +
                                     "\n")
    fold = {"train_pos": _pairs([0, 1], [1, 2], [2, 3], [3, 4]),
            "test_pos": _pairs([0, 2], [1, 6], [4, 7]),
            "test_neg": _pairs([1, 5], [7, 8], [2, 9], [3, 10])}
    np.save(folds / "all_pos.npy",
            np.vstack([fold["train_pos"], fold["test_pos"]]))
    (folds / "meta.json").write_text(json.dumps({"seed": 42,
                                                 "num_folds": n_folds}))
    for k in range(n_folds):
        d = folds / "cv1" / f"fold_{k}"
        d.mkdir(parents=True)
        for name, arr in fold.items():
            np.save(d / f"{name}.npy", arr)
    return folds


def test_fold_provenance_git_failure_is_unknown():
    """git exits 128 here (GIT_DIR names no repo). The old code read the
    empty stdout as a clean tree and recorded git_dirty=False."""
    with tempfile.TemporaryDirectory() as tmp:
        with _env(GIT_DIR=str(Path(tmp) / "no_repo"), GIT_WORK_TREE=None):
            prov = em.fold_provenance(tmp, "cv1")
        assert prov["git_sha"] is None, prov
        assert prov["git_dirty"] is None, prov
        assert "git_dirty_files" not in prov, prov


def test_fold_provenance_dirty_repo_still_recorded():
    """A repo whose status works still records its dirt, even when HEAD has
    no commit (rev-parse fails on its own)."""
    if shutil.which("git") is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        subprocess.run(["git", "init", "-q", str(repo)], check=True,
                       capture_output=True)
        (repo / "a.py").write_text("x = 1\n")
        old = em.__file__
        em.__file__ = str(repo / "evaluate_model.py")
        try:
            with _env(GIT_DIR=None, GIT_WORK_TREE=None):
                prov = em.fold_provenance(tmp, "cv1")
        finally:
            em.__file__ = old
        assert prov["git_sha"] is None, prov
        assert prov["git_dirty"] is True, prov
        assert "a.py" in prov["git_dirty_files"], prov


def test_build_summary_records_partial():
    with tempfile.TemporaryDirectory() as tmp:
        folds = _folds(Path(tmp))
        m = {k: 0.5 for k in em.METRIC_KEYS}
        m["n_genes"] = 12
        s = em.build_summary("X", "cv1", folds, [dict(m), dict(m)])
        assert (s["n_folds_scored"], s["n_folds_expected"], s["partial"]) \
            == (2, 3, True), s
        s = em.build_summary("X", "cv1", folds, [dict(m)] * 3)
        assert (s["n_folds_expected"], s["partial"]) == (3, False), s


def test_main_partial_refused_or_marked():
    """Two of three folds have a matrix: refused by default; --allow_partial
    writes a summary that says partial=true."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folds = _folds(root)
        pred = root / "pred"
        pred.mkdir()
        rng = np.random.RandomState(0)
        for k in range(2):
            np.save(pred / f"fold_{k}_cv1_predictions.npy",
                    rng.rand(12, 12).astype(np.float32))
        out = root / "X_cv1.json"
        argv = ["evaluate_model.py", "--folds_dir", str(folds), "--cv_type",
                "cv1", "--pred_dir", str(pred), "--model_name", "X", "--out",
                str(out), "--allow_unstamped"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), _argv(argv):
            try:
                em.main()
                raise AssertionError("partial fold set not refused")
            except SystemExit as e:
                assert "scored 2 of 3 folds" in str(e.code), e.code
        assert not out.exists()
        with contextlib.redirect_stdout(buf), \
                _argv(argv + ["--allow_partial"]):
            em.main()
        s = json.loads(out.read_text())["summary"]
        assert (s["n_folds_scored"], s["n_folds_expected"], s["partial"]) \
            == (2, 3, True), s


def test_write_summary_cut_write_leaves_no_row():
    """A write that fails part-way (a full quota) leaves nothing under the
    row's name: run_cell_line_ablation.sh, run_compare.sh and
    run_external_transfer.sh take a row file that EXISTS as a graded row, and
    the in-place write left the cut file there (the cell-line job then said
    '6 of 6 applicable row(s) graded' and exited 0)."""
    import errno
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folds = _folds(root)
        m = {k: 0.5 for k in em.METRIC_KEYS}
        m["n_genes"] = 12
        s = em.build_summary("X", "cv1", folds, [dict(m)] * 3)
        res = root / "results"
        out = res / "X_cv1.json"
        real = json.dump

        def cut(obj, f, **kw):
            f.write(json.dumps(obj, **kw)[:200])
            f.flush()
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

        json.dump = cut
        try:
            try:
                em.write_summary(s, out)
                raise AssertionError("the failed write raised nothing")
            except OSError as e:
                assert e.errno == errno.EDQUOT, e
        finally:
            json.dump = real
        assert [p.name for p in res.iterdir()] == [], list(res.iterdir())
        # a later write succeeds and leaves the row alone in the directory;
        # the re-grade's --carry_from <row> --out <row> still reads the old
        # row before replacing it
        buf = io.StringIO()
        s["selection_audit"] = {"kept": True}
        with contextlib.redirect_stdout(buf):
            em.write_summary(s, out)
        assert f"-> {out}" in buf.getvalue(), buf.getvalue()
        s2 = em.build_summary("X", "cv1", folds, [dict(m)] * 3,
                              carry_from=out)
        with contextlib.redirect_stdout(buf):
            em.write_summary(s2, out)
        assert [p.name for p in res.iterdir()] == ["X_cv1.json"]
        assert json.loads(out.read_text())["summary"]["selection_audit"] \
            == {"kept": True}


def _sources(root: Path):
    d = root / "data"
    d.mkdir()
    for name in ("pos", "neg"):
        (d / f"{name}.txt").write_text(f"1 2\n3 4 {name}\n")
    return d / "pos.txt", d / "neg.txt"


def test_check_sources_read_error_exits_2():
    """run_shared_folds.sh reads exit 1 as "sources differ" (FATAL under
    SHARED_FOLDS_FROZEN=1); an unreadable source is "not checked" instead.
    Monkeypatched rather than chmod so it holds when run as root."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pos, neg = _sources(root)
        out = root / "folds"
        out.mkdir()
        (out / "meta.json").write_text(json.dumps(
            {"source_sha256": pf.source_fingerprints(pos, neg, False)}))
        argv = ["prepare_folds.py", "--check_sources", "--out_dir", str(out),
                "--pos_path", str(pos), "--neg_path", str(neg)]

        def denied(path):
            raise PermissionError(13, "Permission denied", str(path))

        buf = io.StringIO()
        orig = pf._sha256
        with contextlib.redirect_stdout(buf), _argv(argv):
            pf.main()  # readable and matching: returns normally
            pf._sha256 = denied
            try:
                pf.main()
                raise AssertionError("no exit on an unreadable source")
            except SystemExit as e:
                assert e.code == 2, e.code
            finally:
                pf._sha256 = orig
        assert "could not fingerprint" in buf.getvalue(), buf.getvalue()
        assert sorted(p.name for p in out.iterdir()) == ["meta.json"]


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
