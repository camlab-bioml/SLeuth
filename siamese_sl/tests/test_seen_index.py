#!/usr/bin/env python3
"""train.py's retrieval suppression set equals the shared grader's.

sl_comparison/evaluate_model.seen_for masks every known SL pair except the
fold's test positives; train.py's self-scored retrieval pass must mask the
same pairs (known_sl_seen_index), mapped from the shared int-sorted gene
index into siamese's string-sorted embedding rows. Covers the unit
behaviour (re-canonicalisation, genes without an embedding, the answer key
kept out, the train-positives fallback) and, on the tiny shared-fold tree of
test_single_output plus an all_pos.npy, that the set Trainer._retrieval_metrics
builds is exactly seen_for's, val positives included.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_seen_index.py -q
    $PYTHON_PATH siamese_sl/tests/test_seen_index.py
"""
import argparse
import ast
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SIAMESE)
for _p in (_SIAMESE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train as T
from data_loader import SLDataManager
from siamese_esm import SiameseSL
from test_single_output import make_shared_folds


def _grader_seen_for():
    """evaluate_model.seen_for's real body, without importing the module:
    its top-level imports would cache sl_comparison's retrieval_metrics under
    the name siamese_sl's own module uses."""
    src = open(os.path.join(_REPO, "sl_comparison", "evaluate_model.py")).read()
    fn = [n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "seen_for"][0]
    g = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "evaluate_model.py",
                 "exec"), g)
    return g["seen_for"]


def test_known_sl_seen_index_unit():
    s2f = np.array([5, 2, -1, 7, 0])  # shared -> full; the order flips
    all_pos = np.array([[0, 1], [0, 3], [1, 2], [3, 4], [1, 4]])
    train_pos = np.array([[2, 5]])
    answer = np.array([[5, 7]])  # shared (0, 3) in full space
    got = T.known_sl_seen_index(all_pos, s2f, train_pos, answer)
    # (0,1)->(2,5) duplicates train; (1,2) has no embedding; (3,4)->(0,7);
    # (1,4)->(0,2); (0,3) is the answer key.
    assert set(map(tuple, got.tolist())) == {(2, 5), (0, 7), (0, 2)}
    assert (got[:, 0] < got[:, 1]).all() and got.dtype == np.int64
    fb = T.known_sl_seen_index(None, None, train_pos, answer)
    assert set(map(tuple, fb.tolist())) == {(2, 5)}
    assert T.known_sl_seen_index(None, None, np.zeros((0, 2)),
                                 answer).shape == (0, 2)


class _Captured(Exception):
    pass


def _seen_from_trainer(dm, fold, emb, out_dir):
    """Run _retrieval_metrics up to its known_sl_seen_index call and return
    the set it builds (the rest of the pass is not needed here)."""
    tr = T.Trainer.__new__(T.Trainer)
    tr.args = argparse.Namespace(cv_type="cv1")
    tr.data_manager = dm
    tr.device = torch.device("cpu")
    tr.output_dir = out_dir
    torch.manual_seed(0)
    model = SiameseSL(input_dim=emb.shape[1], encoder_dims=[8, 4]).eval()
    real = T.known_sl_seen_index
    box = {}

    def spy(*a):
        box["seen"] = real(*a)
        raise _Captured

    T.known_sl_seen_index = spy
    try:
        tr._retrieval_metrics(model, fold, emb)
    except _Captured:
        pass
    finally:
        T.known_sl_seen_index = real
    assert "seen" in box, "_retrieval_metrics never built its seen set"
    return set(map(tuple, box["seen"].tolist()))


def test_trainer_seen_set_equals_grader_seen_for():
    seen_for = _grader_seen_for()
    with tempfile.TemporaryDirectory() as root:
        emb, sl, fd, shared, labels = make_shared_folds(root)
        pos = np.array(sorted(p for p, y in labels.items() if y),
                       dtype=np.int32)
        np.save(os.path.join(fd, "all_pos.npy"), pos)
        dm = SLDataManager(embeddings_paths=[emb], sl_pairs_path=sl,
                           folds_dir=fd, seed=0)
        _, s2f = dm._shared_gene_map()

        def to_full(a):
            f = s2f[np.asarray(a, dtype=np.int64).reshape(-1, 2)]
            return set(map(tuple, np.sort(f, axis=1).tolist()))

        folds = dm.get_cv1_splits(num_folds=5)
        out = Path(root) / "out"
        out.mkdir()
        for k in (0, 1):
            E, _ = dm.get_fold_embeddings(folds[k])
            got = _seen_from_trainer(dm, folds[k], E, out)
            d = Path(fd) / "cv1" / f"fold_{k}"
            tp = np.load(d / "test_pos.npy").astype(np.int64)
            trp = np.load(d / "train_pos.npy").astype(np.int64)
            vp = np.load(d / "val_pos.npy").astype(np.int64)
            want = to_full(seen_for(d, tp, trp))
            assert got == want, (k, len(got), len(want))
            assert len(vp) and to_full(vp) <= got, "val positives unmasked"
            assert not to_full(tp) & got

        # Without all_pos.npy both fall back to the train positives.
        os.remove(os.path.join(fd, "all_pos.npy"))
        E, _ = dm.get_fold_embeddings(folds[1])
        got = _seen_from_trainer(dm, folds[1], E, out)
        d = Path(fd) / "cv1" / "fold_1"
        trp = np.load(d / "train_pos.npy").astype(np.int64)
        assert got == to_full(trp)


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
