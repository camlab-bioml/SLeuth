#!/usr/bin/env python3
"""train.py never leaves an export it has just declared wrong on disk.

_retrieval_metrics writes fold_<k>_<cv>_predictions.npy (the matrix
sl_comparison/evaluate_model.py grades and re-grades every ranking column
from) BEFORE the A15 AUROC identity check on the very same array. A failed
identity therefore condemns the file, so the branch must remove it: a fold
that cannot be trusted has to be MISSING, never stale — otherwise
collect_and_regrade_siamese.sh sees a complete fold set and publishes it.

Also covers the embedding CONTENT fingerprint each run records
(Trainer.emb_sha256 -> every checkpoint's "embeddings_sha256" and
results.json), which is the only thing that can tell a same-universe
regeneration from the files a checkpoint was trained on.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_export_guard.py -q
    $PYTHON_PATH siamese_sl/tests/test_export_guard.py
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
for _p in (_SIAMESE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train as T
from data_loader import SLDataManager
from siamese_esm import SiameseSL
from test_single_output import make_shared_folds


def _stub_trainer(dm, out_dir):
    """A Trainer with just the attributes _retrieval_metrics reads."""
    tr = T.Trainer.__new__(T.Trainer)
    tr.args = argparse.Namespace(cv_type="cv1")
    tr.data_manager = dm
    tr.device = torch.device("cpu")
    tr.output_dir = out_dir
    return tr


def _run(tr, model, fold, emb, auroc_any_ref):
    """(raised RuntimeError or None, matrix exists?) for one retrieval pass."""
    mat = tr.output_dir / f"fold_{fold['fold']}_cv1_predictions.npy"
    mat.unlink(missing_ok=True)
    err = None
    try:
        tr._retrieval_metrics(model, fold, emb, auroc_any_ref=auroc_any_ref)
    except RuntimeError as e:
        err = e
    return err, mat.exists()


def test_a15_failure_removes_the_exported_matrix():
    with tempfile.TemporaryDirectory() as root:
        emb_path, sl, fd, _shared, labels = make_shared_folds(root)
        np.save(
            os.path.join(fd, "all_pos.npy"),
            np.array(sorted(p for p, y in labels.items() if y),
                     dtype=np.int32))
        dm = SLDataManager(embeddings_paths=[emb_path], sl_pairs_path=sl,
                           folds_dir=fd, seed=0)
        fold = dm.get_cv1_splits(num_folds=5)[0]
        E, _ = dm.get_fold_embeddings(fold)
        out = Path(root) / "out"
        out.mkdir()
        tr = _stub_trainer(dm, out)
        torch.manual_seed(0)
        model = SiameseSL(input_dim=E.shape[1], encoder_dims=[8, 4]).eval()

        # A passing pass (no reference to check against) keeps its export.
        err, exists = _run(tr, model, fold, E, None)
        assert err is None, f"clean retrieval pass raised {err}"
        assert exists, "the shared-space export was not written"

        # A15 fires: the identity is broken by construction, so the matrix
        # the check just condemned must be gone.
        err, exists = _run(tr, model, fold, E, 0.0)
        assert err is not None, "A15 did not fire on a broken AUROC identity"
        assert "REMOVED" in str(err), str(err)
        assert not exists, ("A15 failed but fold_0_cv1_predictions.npy is "
                            "still on disk; evaluate_model.py would grade it")


def test_failed_matrix_write_leaves_nothing_under_the_graded_name():
    """fold_<k>_<cv>_predictions.npy is what the shared grader reads and what
    collect_and_regrade_siamese.sh counts. A write that dies part-way (a full
    quota) leaves nothing under that name and no piece beside it, and the
    error still reaches train_fold (its banner for a missing export keys on
    the file). The next clean pass exports a whole matrix."""
    with tempfile.TemporaryDirectory() as root:
        emb_path, sl, fd, _shared, labels = make_shared_folds(root)
        np.save(
            os.path.join(fd, "all_pos.npy"),
            np.array(sorted(p for p, y in labels.items() if y),
                     dtype=np.int32))
        dm = SLDataManager(embeddings_paths=[emb_path], sl_pairs_path=sl,
                           folds_dir=fd, seed=0)
        fold = dm.get_cv1_splits(num_folds=5)[0]
        E, _ = dm.get_fold_embeddings(fold)
        out = Path(root) / "out"
        out.mkdir()
        tr = _stub_trainer(dm, out)
        torch.manual_seed(0)
        model = SiameseSL(input_dim=E.shape[1], encoder_dims=[8, 4]).eval()
        mat = out / "fold_0_cv1_predictions.npy"

        def half(file, arr, *a, **kw):
            piece = b"\x93NUMPY half a matrix"
            if hasattr(file, "write"):
                file.write(piece)
            else:
                Path(file).write_bytes(piece)
            raise OSError(28, "No space left on device")

        with mock.patch.object(T.np, "save", half):
            try:
                tr._retrieval_metrics(model, fold, E, auroc_any_ref=None)
            except OSError as e:
                assert "No space left" in str(e), e
            else:
                raise AssertionError("the failed write was swallowed")
        assert sorted(p.name for p in out.iterdir()) == [], \
            sorted(p.name for p in out.iterdir())
        err, exists = _run(tr, model, fold, E, None)
        assert err is None and exists, err
        a = np.load(mat)
        assert a.ndim == 2 and a.shape[0] == a.shape[1] \
            and a.dtype == np.float32, (a.shape, a.dtype)
        assert sorted(p.name for p in out.iterdir()) == [mat.name]


def test_matrix_cut_in_its_last_bytes_is_not_exported():
    """Handed the open file itself, np.save calls ndarray.tofile, which writes
    through a C stream of its own and never checks that stream's fclose(): a
    write refused in its last buffer raised nothing, and a matrix a few bytes
    short took the graded name with "Saved" in the log. Here the kernel
    refuses the last 100 bytes (a file-size limit, as a full quota would);
    nothing may be left under the graded name, and the next clean pass
    exports a whole matrix."""
    import io
    import resource
    with tempfile.TemporaryDirectory() as root:
        # 150 genes: 90,000 data bytes, a multiple of no stdio buffer size, so
        # the refused bytes sit in the stream's last buffer under glibc too.
        emb_path, sl, fd, shared, labels = make_shared_folds(root,
                                                             n_genes=150)
        np.save(
            os.path.join(fd, "all_pos.npy"),
            np.array(sorted(p for p, y in labels.items() if y),
                     dtype=np.int32))
        dm = SLDataManager(embeddings_paths=[emb_path], sl_pairs_path=sl,
                           folds_dir=fd, seed=0)
        fold = dm.get_cv1_splits(num_folds=5)[0]
        E, _ = dm.get_fold_embeddings(fold)
        out = Path(root) / "out"
        out.mkdir()
        tr = _stub_trainer(dm, out)
        torch.manual_seed(0)
        model = SiameseSL(input_dim=E.shape[1], encoder_dims=[8, 4]).eval()
        mat = out / "fold_0_cv1_predictions.npy"
        buf = io.BytesIO()
        np.save(buf, np.zeros((len(shared), len(shared)), dtype=np.float32))
        whole = buf.tell()
        real = T.Trainer._save_shared_prediction_matrix

        def limited(self, S, fold_idx):
            soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            resource.setrlimit(resource.RLIMIT_FSIZE, (whole - 100, hard))
            try:
                return real(self, S, fold_idx)
            finally:
                resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))

        with mock.patch.object(T.Trainer, "_save_shared_prediction_matrix",
                               limited):
            try:
                tr._retrieval_metrics(model, fold, E, auroc_any_ref=None)
            except OSError:
                raised = True
            else:
                raised = False
        left = {p.name: p.stat().st_size for p in out.iterdir()}
        assert left == {}, (f"a matrix cut short was left behind: {left} "
                            f"(whole = {whole} bytes)")
        assert raised, "the refused bytes raised nothing"
        err, exists = _run(tr, model, fold, E, None)
        assert err is None and exists, err
        assert mat.stat().st_size == whole, (mat.stat().st_size, whole)
        assert sorted(p.name for p in out.iterdir()) == [mat.name]


def _args(emb_paths, sl, fd, out):
    """A single-output run of 2 epochs on the tiny synthetic tree."""
    return argparse.Namespace(
        embeddings_paths=list(emb_paths), sl_path=sl, neg_pairs_path=None,
        gene_list_path=None, folds_dir=fd, output_dir=str(out), seed=0,
        pca_dims=None, post_pca=None, preprocessing_fit_scope="train",
        pca_method="plain", use_cell_lines=False, kernel="pd", gene_term=False, pair_features=False,
        cell_lines="per_head", input_dim=None, cpu=True, model_type="siamese",
        encoder_dims=[8, 4], hidden_dim=8, latent_dim=4, dropout=0.1,
        last_layer_bias=True, pd_epsilon=0.001, l1_lambdas=None,
        siamese_encoder_type="residual", learning_rate=1e-3,
        weight_decay=1e-5, batch_size=256, epochs=2, eval_interval=1,
        patience=5, warmrestart_T0=50, warmrestart_Tmult=2,
        select_metric="auprg_any", cv_type="cv1", num_folds=5,
        pos_neg_ratio=1.0)


def test_run_records_the_embedding_content_hash():
    with tempfile.TemporaryDirectory() as root:
        emb_path, sl, fd, _shared, _labels = make_shared_folds(root)
        out = Path(root) / "run"
        (out / "checkpoints").mkdir(parents=True)

        want = hashlib.sha256(Path(emb_path).read_bytes()).hexdigest()
        assert T.file_sha256(emb_path) == want
        tr = T.Trainer(_args([emb_path], sl, fd, out))
        assert tr.emb_sha256 == [want]

        # Every fold checkpoint and results.json carry it, so a later
        # eval_finetuning / zero_cell_emb / predict run can check the files it
        # loads against the ones these weights were trained on.
        tr.train()
        for k in range(5):
            ck = torch.load(out / "checkpoints" / f"fold_{k}_best.pt",
                            map_location="cpu", weights_only=False)
            assert ck["embeddings_sha256"] == [want], k
            assert ck["embeddings_paths"] == [emb_path], k
        res = json.loads((out / "results.json").read_text())
        assert res["embeddings_sha256"] == [want]
        assert res["config"]["embeddings_paths"] == [emb_path]

        # A same-universe regeneration: same path, same gene_order, different
        # vectors. The basename and gene_order guards downstream cannot see
        # it; the content hash must.
        data = torch.load(emb_path, map_location="cpu", weights_only=False)
        order_before = list(data["gene_order"])
        torch.manual_seed(1)
        data["raw_embeddings"] = torch.randn_like(data["raw_embeddings"])
        torch.save(data, emb_path)
        tr2 = T.Trainer(_args([emb_path], sl, fd, out))
        after = torch.load(emb_path, map_location="cpu", weights_only=False)
        assert list(after["gene_order"]) == order_before
        assert tr2.emb_sha256 != tr.emb_sha256, (
            "a regenerated embedding file kept the same fingerprint")


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
