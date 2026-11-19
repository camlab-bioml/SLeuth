#!/usr/bin/env python3
"""The eval fingerprint both sides of the external-transfer table rely on.

eval_finetuning.py (run from siamese_sl/) scores fold_<k>_best.pt for every
k = 0..K-1 and records {checkpoint path: sha256} plus fold_ids in
config_used.json; sl_comparison/external_transfer.py (run from the repo root,
so the same checkpoint has another path) recomputes the hashes and refuses a
mismatch. Covers resolve_fold_checkpoints (numeric fold order, a gapped or
partial set refused, never renumbered) and that a fingerprint recorded the
way eval_finetuning.main records it passes check_eval_fingerprint, while a
retrain in place fails it.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_eval_fingerprint.py -q
    $PYTHON_PATH siamese_sl/tests/test_eval_fingerprint.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SIAMESE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

from eval_finetuning import resolve_fold_checkpoints

# By file path, so sl_comparison/ never lands on sys.path.
_spec = importlib.util.spec_from_file_location(
    "_external_transfer_fp",
    os.path.join(_REPO, "sl_comparison", "external_transfer.py"))
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)


def _ckpts(d: Path, folds):
    d.mkdir(parents=True, exist_ok=True)
    for k in folds:
        (d / f"fold_{k}_best.pt").write_bytes(f"weights {k}".encode())


def _raises_value_error(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError("ValueError not raised")


def test_resolve_fold_checkpoints_orders_and_refuses_gaps():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "checkpoints"
        _ckpts(d, range(11))
        got = resolve_fold_checkpoints(d, 11)
        assert [p.name for p in got] == [f"fold_{k}_best.pt"
                                         for k in range(11)]  # 10 after 9
        (d / "fold_3_best.pt").unlink()
        _raises_value_error(lambda: resolve_fold_checkpoints(d, 11),
                            "never renumbered")
        _raises_value_error(lambda: resolve_fold_checkpoints(d, 12))


def test_recorded_fingerprint_passes_external_transfer_until_retrained():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # the eval's view: cwd siamese_sl/, MODEL_DIR relative to it
        rel = Path("results") / "best_cat_x_cv3"
        model_dir = root / "siamese_sl" / rel
        _ckpts(model_dir / "checkpoints", range(5))
        paths = resolve_fold_checkpoints(model_dir / "checkpoints", 5)
        cfg = {
            "fold_ids": list(range(len(paths))),
            "checkpoint_sha256": {
                str(rel / "checkpoints" / p.name):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "sldb_overlap_sources": ["SL.txt", "SL_negatives.txt"],
        }
        ev = model_dir / "eval_finetuning_cv1"
        ev.mkdir()
        (ev / "config_used.json").write_text(json.dumps(cfg))
        current = et.checkpoint_hashes(model_dir)
        assert et.check_eval_fingerprint(model_dir, "cv1", current) is None
        (model_dir / "checkpoints" / "fold_4_best.pt").write_bytes(b"new")
        _raises_value_error(
            lambda: et.check_eval_fingerprint(
                model_dir, "cv1", et.checkpoint_hashes(model_dir)),
            "fold_4_best.pt", "--allow_legacy_eval")


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
