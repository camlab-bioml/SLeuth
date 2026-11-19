#!/usr/bin/env python3
"""Deterministic unit tests for the NSF4SL harness (no training, no network).

Covers the trainer's CLI (the shared folds are required; the removed native
split flags and non-positive --epochs / --eval_interval are refused; the
default --output_dir is not the pipeline's graded dir), the feature / loop
provenance written to model_params.json, build_transe_features.fetch_sources
(a truncated or interrupted download is never left at the final path and
--download repairs one), build_transe_features.build on a tiny synthetic KG
(gene linking, collision precedence, the mapping_sha256 keys the input check
needs), sl_data.load_transe_features (raw rows, all-entity-mean fill) and
sl_data.load_gene_features (which genes it had to substitute).

    python -m pytest NSF4SL/test_nsf4sl.py -q     (from the repo root)
    python NSF4SL/test_nsf4sl.py
"""
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_transe_features  # noqa: E402
import train_nsf4sl  # noqa: E402
from sl_data import load_gene_features, load_transe_features  # noqa: E402

_PAYLOAD = bytes(range(256)) * 20  # 5,120 deterministic bytes


def _parse(argv):
    old = sys.argv
    sys.argv = ["train_nsf4sl.py"] + argv
    try:
        return train_nsf4sl.parse_args()
    finally:
        sys.argv = old


def _raises(exc, fn, *a):
    try:
        fn(*a)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


class _pinned_source:
    """Point build_transe_features.SOURCES at one local file:// input."""

    def __init__(self, tmp, sha=None):
        self.src = Path(tmp) / "upstream.bin"
        self.src.write_bytes(_PAYLOAD)
        self.sha = sha or hashlib.sha256(_PAYLOAD).hexdigest()

    def __enter__(self):
        self._old = build_transe_features.SOURCES
        build_transe_features.SOURCES = {
            "x.bin": (self.src.as_uri(), self.sha)
        }
        return self

    def __exit__(self, *exc):
        build_transe_features.SOURCES = self._old


def test_folds_dir_is_required():
    e = _raises(SystemExit, _parse, ["--cv_type", "cv1"])
    assert e.code == 2
    args = _parse(["--folds_dir", "folds", "--cv_type", "cv2"])
    assert args.folds_dir == "folds" and args.cv_type == "cv2"


def test_native_split_flags_are_gone():
    # --num_folds used to label the fold banner without setting the fold
    # count; --data_path / --train_ratio / --pos_neg_ratio fed only the
    # removed SLDataSplitter path.
    for flag, val in (("--num_folds", "2"), ("--data_path", "../data"),
                      ("--train_ratio", "0.8"), ("--pos_neg_ratio", "1")):
        e = _raises(SystemExit, _parse, ["--folds_dir", "f", flag, val])
        assert e.code == 2, flag


def test_nonpositive_epochs_refused():
    # --epochs <= 0 would skip every evaluation and export the random init.
    for bad in (["--epochs", "0"], ["--epochs", "-3"],
                ["--eval_interval", "0"]):
        e = _raises(SystemExit, _parse, ["--folds_dir", "f"] + bad)
        assert e.code == 2, bad
    args = _parse(["--folds_dir", "f", "--epochs", "1", "--eval_interval", "1"])
    assert args.epochs == 1 and args.eval_interval == 1


def test_fetch_missing_without_download_raises():
    with tempfile.TemporaryDirectory() as tmp, _pinned_source(tmp):
        raw = Path(tmp) / "raw"
        _raises(FileNotFoundError, build_transe_features.fetch_sources, raw,
                False)


def test_fetch_repairs_truncated_input_with_download():
    with tempfile.TemporaryDirectory() as tmp, _pinned_source(tmp):
        raw = Path(tmp) / "raw"
        raw.mkdir()
        (raw / "x.bin").write_bytes(_PAYLOAD[:100])
        e = _raises(RuntimeError, build_transe_features.fetch_sources, raw,
                    False)
        assert "--download" in str(e) and "truncated" in str(e)
        assert (raw / "x.bin").read_bytes() == _PAYLOAD[:100]
        build_transe_features.fetch_sources(raw, True)
        assert (raw / "x.bin").read_bytes() == _PAYLOAD
        assert not (raw / "x.bin.part").exists()
        build_transe_features.fetch_sources(raw, False)  # now verified


def test_fetch_never_leaves_an_unverified_download():
    with tempfile.TemporaryDirectory() as tmp, _pinned_source(tmp, "0" * 64):
        raw = Path(tmp) / "raw"
        _raises(RuntimeError, build_transe_features.fetch_sources, raw, True)
        assert not (raw / "x.bin").exists()
        assert not (raw / "x.bin.part").exists()


def test_fetch_interrupted_download_leaves_no_file():
    real = urllib.request.urlretrieve

    def short_read(url, filename):
        Path(filename).write_bytes(_PAYLOAD[:1000])
        raise urllib.error.ContentTooShortError("retrieval incomplete", None)

    with tempfile.TemporaryDirectory() as tmp, _pinned_source(tmp):
        raw = Path(tmp) / "raw"
        urllib.request.urlretrieve = short_read
        try:
            _raises(urllib.error.ContentTooShortError,
                    build_transe_features.fetch_sources, raw, True)
        finally:
            urllib.request.urlretrieve = real
        assert not (raw / "x.bin").exists()
        assert not (raw / "x.bin.part").exists()
        build_transe_features.fetch_sources(raw, True)  # a retry succeeds
        assert (raw / "x.bin").read_bytes() == _PAYLOAD


def test_load_transe_features_raw_rows_and_entity_mean_fill():
    emb = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    entity_mean = np.array([10.0, -4.0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "feats.pt"
        torch.save({"embeddings": torch.from_numpy(emb),
                    "gene_order": ["20", "10"],
                    "entity_mean": torch.from_numpy(entity_mean)}, p)
        feats, fill, missing = load_transe_features(p, ["10", "30", "20"])
    np.testing.assert_array_equal(
        feats.numpy(), [[3.0, 6.0], [10.0, -4.0], [1.0, 2.0]])
    np.testing.assert_array_equal(fill, entity_mean)
    assert missing == ["30"]


def test_load_gene_features_reports_substituted_genes():
    # results.json's genes_without_features used to be [] on every --features
    # embedding run, asserting full coverage while all-NaN rows (what
    # generate_embeddings.py writes for a gene it could not embed; 95 of the
    # 8,844 shared genes in all_genes_kg_complex.pt) were silently replaced by
    # the column mean.
    emb = np.array([[1.0, 3.0], [np.nan, np.nan], [5.0, np.nan]],
                   dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "emb.pt"
        torch.save({"embeddings": torch.from_numpy(emb),
                    "gene_order": ["10", "20", "30"]}, p)
        feats, matched, missing = load_gene_features(
            p, ["10", "20", "30", "40"], standardize=False)
    assert matched == 3              # 40 has no row at all
    assert missing == ["20", "40"]   # no usable row; 30 keeps its real dim 0
    np.testing.assert_array_equal(
        feats.numpy(),
        [[1.0, 3.0], [3.0, 3.0], [5.0, 3.0], [3.0, 3.0]])  # column means 3, 3


def test_default_output_dir_is_not_the_pipeline_dir():
    # results/nsf4sl_<cv> is run_shared_models.sh's graded dir (seed 42 by
    # default); a manual run with the seed-123 default must not overwrite it.
    args = _parse(["--folds_dir", "f", "--cv_type", "cv3"])
    assert args.output_dir == "results/nsf4sl_standalone_cv3"
    args = _parse(["--folds_dir", "f", "--output_dir", "results/nsf4sl_cv3"])
    assert args.output_dir == "results/nsf4sl_cv3"


def test_feature_provenance_names_features_and_loop():
    args = _parse(["--folds_dir", "f", "--transe_path", "t.pt"])
    assert train_nsf4sl.feature_provenance(args) == {
        "features": "transe", "feature_file": "t.pt",
        "standardized": False, "legacy_loop": False}
    # main() resolves --embedding_path before train_fold records it.
    args = _parse(["--folds_dir", "f", "--features", "embedding",
                   "--embedding_path", "e.pt", "--legacy_loop"])
    assert train_nsf4sl.feature_provenance(args) == {
        "features": "embedding", "feature_file": "e.pt",
        "standardized": True, "legacy_loop": True}
    args = _parse(["--folds_dir", "f", "--features", "embedding",
                   "--embedding_path", "e.pt", "--no_standardize"])
    assert train_nsf4sl.feature_provenance(args)["standardized"] is False


def _tiny_synlethkg(raw, cache):
    """Five KG entities (ids 10-14, db ids 100-104); 13 is a GO term. Gene
    entities 11 (ZZB, the current HGNC symbol) and 12 (ZZBALIAS, its alias)
    both map to Entrez 9002; 12 is the original's SL gene and must win."""
    raw.mkdir()
    cache.mkdir()
    emb = np.arange(15, dtype=np.float32).reshape(5, 3)
    np.save(raw / "kg_TransE_l2_entity.npy", emb)
    (raw / "entities.tsv").write_text(
        "".join(f"{r}\t{10 + r}\n" for r in range(5)))
    (raw / "entity2id.txt").write_text(
        "a\tb\n" + "".join(f"{100 + r}\t{10 + r}\n" for r in range(5)))
    names = ["ZZA", "ZZB", "ZZBALIAS", "GOTERM", "ZZC"]
    (raw / "dbid2name.csv").write_text(
        ",_id,name\n" + "".join(f"{r},{100 + r},{n}\n"
                                for r, n in enumerate(names)))
    (raw / "id_mapping.csv").write_text("id,name\n10,ZZA\n12,ZZBALIAS\n")
    (raw / "kg_triplet.csv").write_text(
        "h,r,t\n100,PARTICIPATES_GpBP,103\n101,INTERACTS_GiG,102\n"
        "103,EXPRESSES_AeG,104\n")
    (raw / "Human_SL.csv").write_text(
        "gene_a.name\tgene_a.identifier\tgene_b.name\tgene_b.identifier\n"
        "ZZA\t9001\tZZC\t9003\n")
    (cache / "hgnc_complete_set.txt").write_text(
        "symbol\tprev_symbol\talias_symbol\tentrez_id\n"
        "ZZB\t\tZZBALIAS\t9002\n")
    return emb


def test_build_on_tiny_kg_links_genes_and_hashes_every_input():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "sl_comparison" / "slurm"))
    import check_shared_inputs
    with tempfile.TemporaryDirectory() as tmp:
        raw, cache = Path(tmp) / "raw", Path(tmp) / "cache"
        emb = _tiny_synlethkg(raw, cache)
        out = build_transe_features.build(raw, cache)
        hgnc_sha = hashlib.sha256(
            (cache / "hgnc_complete_set.txt").read_bytes()).hexdigest()
    assert out["gene_order"] == ["9001", "9002", "9003"]
    assert out["kg_entity"] == [10, 12, 14]
    assert out["entrez_source"] == ["synlethdb_id", "hgnc", "synlethdb_id"]
    np.testing.assert_array_equal(out["embeddings"].numpy(), emb[[0, 2, 4]])
    np.testing.assert_allclose(out["entity_mean"].numpy(), emb.mean(axis=0))
    meta = out["meta"]
    assert meta["n_kg_genes"] == 4 and meta["n_entrez_collisions"] == 1
    # The real builder's mapping_sha256 layout, against the check that gates
    # the benchmark (check_shared_inputs): it holds every required key and
    # nothing the check cannot locate, and hashes the builder itself.
    rec = meta["mapping_sha256"]
    assert set(check_shared_inputs.REQUIRED_MAPPING_KEYS) <= set(rec), rec
    assert set(rec) <= set(check_shared_inputs.mapping_files(
        Path(__file__).resolve().parent.parent)), rec
    assert rec["hgnc_complete_set.txt"] == hgnc_sha
    assert rec["build_transe_features.py"] == hashlib.sha256(
        Path(build_transe_features.__file__).read_bytes()).hexdigest()


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n==== {len(tests) - failed} passed, {failed} failed ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
