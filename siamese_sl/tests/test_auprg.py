#!/usr/bin/env python3
"""Deterministic tests for the shared AUPRG (sl_comparison/auprg.py).

Every expected value below was computed with the authors' reference
implementation (github.com/meeliskull/prg, Python_package/prg/prg.py:
calc_auprg(create_prg_curve(labels, scores))). The first case is the one a
code review used to show that the integrator used until 2026-09-21 was wrong
(it returned +0.25; the reference gives -0.25). train.py's _auprg must be the
same function, since checkpoint selection and the shared evaluator have to
agree.
"""
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_spec = importlib.util.spec_from_file_location(
    "shared_auprg", os.path.join(_REPO, "sl_comparison", "auprg.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

# (labels, scores, reference AUPRG)
REFERENCE_CASES = [
    ([1, 0, 1], [3, 2, 1], -0.25),
    ([1, 1, 0, 0], [4, 3, 2, 1], 1.0),
    ([1, 0, 1, 0], [1, 1, 1, 1], 0.0),
    ([1, 0], [1, 2], -0.5),
    ([0, 1, 1, 0, 1, 0, 0, 1], [0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5, 0.1],
     0.0555555555555556),
    ([1, 0, 0, 1, 0, 0, 0, 1, 0, 1], [5, 4, 4, 3, 3, 2, 2, 1, 1, 0],
     -0.27314814814814803),
    ([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
     -0.2743055555555556),
]


def test_matches_reference_values():
    for y, s, want in REFERENCE_CASES:
        got = A.auprg(y, s)
        assert abs(got - want) < 1e-12, (y, s, got, want)


def test_degenerate_prevalence_is_nan():
    assert np.isnan(A.auprg([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(A.auprg([0, 0], [0.1, 0.2]))
    assert np.isnan(A.auprg([], []))


def test_invariant_to_monotone_transforms():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(float)
    s = rng.normal(size=300) + y
    base = A.auprg(y, s)
    for f in (lambda x: 3 * x + 7, np.exp, lambda x: x ** 3):
        assert abs(A.auprg(y, f(s)) - base) < 1e-12


def test_train_uses_the_shared_function():
    """siamese's checkpoint selection must score exactly what the shared
    evaluator reports (one definition, no local fallback)."""
    src = open(os.path.join(_REPO, "siamese_sl", "train.py")).read()
    assert '"auprg.py"' in src and "return _amod.auprg(labels, scores)" in src
    assert "import prg" not in src


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
