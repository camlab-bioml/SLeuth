#!/usr/bin/env python3
"""Deterministic unit tests for the SLMGAE port and its shared-fold harness.

Pins the port to original_tensorflow_code/ where it once deviated (GO views
binarized by array2coo; decoders without dropout), the harness's
List_Proteins mapping (incl. the offline static-table fallback, which the
trainer refuses unless opted in) / measure-only coverage diagnostic /
support-view input checks / fold count / val selection on the graded,
float32-shifted matrix / provenance in model_params.json / crash purge, and the
auxiliary scripts' fixed pieces (train_BC's per-fold main view, case_study's
hold-out / main view / best-state restore / i < j scoring, case-study gene
keys, ESM batching, gene corrections vs HGNC, data_split's docstrings vs the
local logic it actually adds). Tiny synthetic
inputs only (plus the tracked data/List_Proteins_in_SL.txt and
data/cache/hgnc_complete_set.txt); the only training is a few CPU epochs on
12-14-gene toy graphs (seconds).

    python -m pytest SLMGAE-in-pytorch/test_slmgae_port.py -q  (from repo root)
    python SLMGAE-in-pytorch/test_slmgae_port.py
"""
import argparse
import csv
import os
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from slmgae_pytorch import SLMGAE, DataLoader, array2coo  # noqa: E402
from train_slmgae_shared import (LP_STATIC_FALLBACK_SPACE,  # noqa: E402
                                 PPI_EDGES_SHIPPED, SharedFoldTrainer,
                                 build_support_views, check_view_shape,
                                 find_row_collisions, load_lp_genes,
                                 remap_support, view_coverage)


# --- verbatim copies of original_tensorflow_code/utils.py (6375 -> n) -------
def _orig_build_KNN_mateix(S, nn_size):
    m, n = S.shape
    X = np.zeros((m, n))
    for i in range(m):
        ii = np.argsort(S[i, :])[::-1][:min(nn_size, n)]
        X[i, ii] = S[i, ii]
    return X


def _orig_array2coo(S, t=0):
    m, n = np.shape(S)
    row, col = [], []
    for i in range(len(S)):
        for j in range(i, len(S[i])):
            if S[i][j] > t:
                row.append(i)
                col.append(j)
    coo_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(m, n))
    coo_matrix = coo_matrix + coo_matrix.T
    return coo_matrix


def _orig_load_dense_feature(fileName, n, knn=False, nnSize=0):
    line = open(fileName, 'r').readlines()
    featureMatrix = np.zeros((n, n))
    for i in range(len(line)):
        s = line[i].replace('\n', '').split('\t')
        for j in range(i, len(s)):
            if s[j] == '':
                break
            featureMatrix[i][j + 1] = float(s[j])
    featureMatrix = featureMatrix + featureMatrix.T
    if knn:
        featureMatrix = _orig_build_KNN_mateix(featureMatrix, nn_size=nnSize)
    featureMatrix = featureMatrix + featureMatrix.T
    return _orig_array2coo(featureMatrix)


def _toy_triangular_file(path, n, seed=0):
    # the shipped Human_GOsim.txt format: n-1 rows, row i holds n-1-i values
    # and a trailing tab (exercises the loaders' parts[j] == '' break)
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for i in range(n - 1):
            vals = rng.uniform(0.01, 1.0, size=n - 1 - i).round(4)
            f.write("\t".join(str(v) for v in vals) + "\t\n")


def test_dense_view_matches_original_array2coo():
    n, k = 12, 3
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "gosim.txt")
        _toy_triangular_file(path, n)
        dl = DataLoader(data_path=td, nn_size=k)
        dl.num_nodes = n
        got = dl.load_dense_feature(path, knn=True).toarray()
        want = _orig_load_dense_feature(path, n, knn=True, nnSize=k).toarray()
    assert np.array_equal(got, want)
    assert set(np.unique(got)) <= {0.0, 1.0}  # unweighted, no doubling
    assert got.any()


def test_array2coo_is_verbatim_including_diagonal():
    rng = np.random.default_rng(3)
    S = rng.uniform(-1, 1, size=(7, 7))
    S[2, 2] = 0.5  # a positive diagonal entry becomes 2, as in the original
    assert np.array_equal(array2coo(S).toarray(), _orig_array2coo(S).toarray())


def test_binarize_false_keeps_weights():
    n = 6
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "gosim.txt")
        _toy_triangular_file(path, n, seed=1)
        dl = DataLoader(data_path=td, nn_size=n)
        dl.num_nodes = n
        w = dl.load_dense_feature(path, knn=False, binarize=False).toarray()
        b = dl.load_dense_feature(path, knn=False).toarray()
    assert np.array_equal(w > 0, b > 0)
    assert not set(np.unique(w)) <= {0.0, 1.0}


def test_decoders_have_no_dropout_gcn_layers_do():
    torch.manual_seed(0)
    m = SLMGAE(num_nodes=6, num_features=6, hidden1=4, hidden2=3,
               dropout=0.2, device=torch.device("cpu"))
    assert [d.dropout for d in m.decoders] == [0.0] * 4
    assert [g.dropout for g in m.gcn_sparse_layers] == [0.2] * 4
    assert [g.dropout for g in m.gcn_dense_layers] == [0.2] * 4
    m.train()
    z = torch.randn(6, 3)
    d = m.decoders[0]
    assert torch.equal(d(z), d(z))
    assert torch.allclose(d(z), z @ d.weight @ z.t())


def test_view_coverage_measures_both_genes():
    n = 6
    # genes 0-3 have a view row; 4 has only a self-loop; 5 has nothing
    v1 = sp.coo_matrix(([1., 1., 1.], ([0, 1, 4], [1, 0, 4])), shape=(n, n))
    v2 = sp.coo_matrix(([1., 1.], ([2, 3], [3, 2])), shape=(n, n))
    v3 = sp.coo_matrix((n, n))
    split = {
        "fold": 0,
        "test_edges": np.array([[0, 1], [2, 4], [0, 3], [4, 5]]),
        "test_labels": np.array([1., 1., 0., 0.]),
        "val_pos": np.array([[1, 2]]),
        "val_neg": np.array([[3, 5], [0, 2]]),
    }
    before = [v.copy() for v in (v1, v2, v3)]
    out = view_coverage([v1, v2, v3], ["a", "b", "c"], [split])
    assert out["genes_with_view_row"] == 4
    assert out["genes_with_row_per_view"] == {"a": 2, "b": 2, "c": 0}
    f = out["folds"][0]
    assert f["test_pos"] == {"n": 2, "both_have_view": 0.5}
    assert f["test_neg"] == {"n": 2, "both_have_view": 0.5}
    assert f["val_pos"] == {"n": 1, "both_have_view": 1.0}
    assert f["val_neg"] == {"n": 2, "both_have_view": 0.5}
    for a, b in zip(before, (v1, v2, v3)):  # measure-only: inputs untouched
        assert (a != b).nnz == 0


_HGNC_HEADER = "hgnc_id\tsymbol\tprev_symbol\talias_symbol\tentrez_id\n"
# Three of the four symbols LP_SYMBOL_OVERRIDES pins, as the real table spells
# them (data/cache/hgnc_complete_set.txt): QARS is a previous symbol of two
# genes (the last HGNC row wins in GeneNameMapper); C18orf21 likewise; TAZ is
# TAFAZZIN's previous symbol and WWTR1's alias (the previous symbol wins). Two
# of the four pins are load-bearing with the current mapper — unpinned, QARS
# lands on EPRS1 and C18orf21 on the pseudogene RMP24P1 — while TAZ and the
# fourth pin, SARS, already resolve right and only guard against a mapper or
# HGNC change. Only the QARS pin can change a support view today: neither
# C18orf21 candidate (83608 / 100287427) nor either SARS candidate is in the
# shared universe. Six List_Proteins symbols are ambiguous this way (claimed
# by more than one gene and not current symbols themselves); GIF (three genes)
# and MUM1 carry no pin.
_HGNC_ROWS = [
    ("HGNC:9751", "QARS1", "QARS", "", "5859"),
    ("HGNC:11577", "TAFAZZIN", '"CMD3A|EFE2|EFE|TAZ"', '"BTHS|G4.5|TAZ1"',
     "6901"),
    ("HGNC:24042", "WWTR1", "", '"TAZ|DKFZp586I1419"', "25937"),
    ("HGNC:3418", "EPRS1", '"QPRS|QARS|EPRS"', '"EARS|PARS|GLUPRORS"',
     "2058"),
    # the pseudogene row comes LAST in HGNC, so "last row wins" sends an
    # unpinned C18orf21 to RMP24P1 instead of RMP24
    ("HGNC:28802", "RMP24", "C18orf21", '"PNAS-131|PNAS-124|HsT3108"',
     "83608"),
    ("HGNC:57042", "RMP24P1", "C18orf21", "", "100287427"),
]


def test_lp_symbols_keep_one_gene_per_row():
    import train_slmgae_shared as tss
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "cache"))
        with open(os.path.join(td, "cache", "hgnc_complete_set.txt"),
                  "w") as f:
            f.write(_HGNC_HEADER)
            for r in _HGNC_ROWS:
                f.write("\t".join(r) + "\n")
        with open(os.path.join(td, "List_Proteins_in_SL.txt"), "w") as f:
            f.write("EPRS\nQARS\nTAZ\nWWTR1\nC18orf21\n")
        genes, space, n_unmapped = load_lp_genes(td)
        # Without the pins QARS lands on EPRS1 (a collision: two of the
        # authors' rows would be summed onto one gene) and C18orf21 on the
        # pseudogene RMP24P1; TAZ is already right, so the QARS and C18orf21
        # halves are the ones that exercise LP_SYMBOL_OVERRIDES.
        saved, tss.LP_SYMBOL_OVERRIDES = tss.LP_SYMBOL_OVERRIDES, {}
        try:
            unpinned, _, _ = load_lp_genes(td)
        finally:
            tss.LP_SYMBOL_OVERRIDES = saved
    assert space == "symbol->entrez" and n_unmapped == 0
    assert genes == ["2058", "5859", "6901", "25937", "83608"]
    assert unpinned == ["2058", "2058", "6901", "25937", "100287427"]
    g2i = {g: i
           for i, g in enumerate(["2058", "5859", "6901", "25937", "83608"])}
    assert find_row_collisions(genes, g2i) == {}
    assert find_row_collisions(unpinned, g2i) == {"2058": [0, 1]}


def test_lp_offline_hgnc_falls_back_to_static_table():
    # No cached HGNC table and no network: the mapper swallows the download
    # error, so load_lp_genes must fall back to data/gene_id_mapping.tsv
    # rather than return an all-empty mapping, and say so in id_space (the
    # static table maps fewer rows: a reduced SLMGAE).
    import urllib.request

    def offline(*a, **kw):
        raise OSError("offline")

    real = urllib.request.urlretrieve
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "cache"))
        with open(os.path.join(td, "List_Proteins_in_SL.txt"), "w") as f:
            f.write("A2M\nBRCA1\nTP53\nQARS\n")
        with open(os.path.join(td, "gene_id_mapping.tsv"), "w") as f:
            f.write("entrez_id\tsymbol\n2\tA2M\n672\tBRCA1\n7157\tTP53\n")
        urllib.request.urlretrieve = offline
        try:
            genes, space, n_unmapped = load_lp_genes(td)
        finally:
            urllib.request.urlretrieve = real
        assert not os.listdir(os.path.join(td, "cache"))
    assert space == LP_STATIC_FALLBACK_SPACE and n_unmapped == 0
    assert genes == ["2", "672", "7157", "5859"]  # QARS via its pin


def test_row_collisions_detected_and_would_sum():
    lp = ["2058", "2058", "25937"]
    g2i = {"2058": 0, "25937": 1, "5859": 2}
    assert find_row_collisions(lp, g2i) == {"2058": [0, 1]}
    assert find_row_collisions(["7", "7"], g2i) == {}  # outside universe
    # why it matters: remap_support sums the two rows into one gene
    view = sp.coo_matrix(([1., 1.], ([0, 1], [2, 2])), shape=(3, 3))
    out = remap_support(view, lp, g2i, 3).tocsr()
    assert out[0, 1] == 2.0


def _write_views(d, n, go=True, ppi="0\t1\n2\t3\n"):
    if go:
        _toy_triangular_file(os.path.join(d, "Human_GOsim.txt"), n, seed=4)
        _toy_triangular_file(os.path.join(d, "Human_GOsim_CC.txt"), n, seed=5)
    with open(os.path.join(d, "biogrid_ppi_sparse.txt"), "w") as f:
        f.write(ppi)


def _raises_system_exit(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise SystemExit")


def test_missing_view_is_fatal_unless_allowed():
    lp, g2i = ["10", "20", "30", "40"], {"10": 0, "20": 1, "30": 2, "40": 3}
    with tempfile.TemporaryDirectory() as td:
        _write_views(td, 4, go=False)
        msg = _raises_system_exit(build_support_views, td, 2, lp, g2i, 4)
        assert "Human_GOsim.txt" in msg and "Human_GOsim_CC.txt" in msg
        views, prov = build_support_views(td, 2, lp, g2i, 4,
                                          allow_missing=True)
    assert [v.nnz for v in views[:2]] == [0, 0] and views[2].nnz == 4
    assert prov["GO BP similarity"].startswith("ZERO-FILLED")
    assert prov["GO CC similarity"].startswith("ZERO-FILLED")
    assert prov["BioGRID PPI"].endswith("biogrid_ppi_sparse.txt")


def test_view_file_must_match_list_proteins():
    g2i = {str(g): i for i, g in enumerate(range(6))}
    with tempfile.TemporaryDirectory() as td:
        _write_views(td, 6, ppi="0\t1\n4\t5\n")
        views, prov = build_support_views(td, 2, list(g2i), g2i, 6)
        assert all(v.nnz > 0 for v in views)
        assert not any(p.startswith("ZERO") for p in prov.values())
        # stale 4-entry list vs 6-gene files: silently truncated before
        msg = _raises_system_exit(build_support_views, td, 2, list(g2i)[:4],
                                  g2i, 6)
        assert "Human_GOsim.txt" in msg and "got 5" in msg
        # dense files fine, PPI id beyond the list
        _write_views(td, 4, ppi="0\t1\n2\t5\n")
        msg = _raises_system_exit(build_support_views, td, 2, list(g2i)[:4],
                                  g2i, 6)
        assert "biogrid_ppi_sparse.txt" in msg and "got 5" in msg
        # a PPI file with no "i<TAB>j" line (empty or cut-short copy, or not
        # tab-separated) used to load as an all-zero view under the ordinary
        # provenance: the --allow_missing_views model, named by no table
        for empty in ("", "\n", "0 1\n2 3\n"):
            _write_views(td, 4, ppi=empty)
            msg = _raises_system_exit(build_support_views, td, 2,
                                      list(g2i)[:4], g2i, 6)
            assert "biogrid_ppi_sparse.txt" in msg and "got -1" in msg


def test_shipped_ppi_edge_count_is_pinned():
    # The ids in biogrid_ppi_sparse.txt are in no order (a prefix of ~1,600
    # lines already reaches the top id), so only the line count tells a
    # cut-short copy (an interrupted upload) from the shipped file. Pinned for
    # the 6,375-row list; other list sizes (the toy fixtures) are not.
    assert PPI_EDGES_SHIPPED == {6375: 98581}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "biogrid_ppi_sparse.txt")
        with open(p, "w") as f:
            f.write("0\t1\n6374\t2\n")
        msg = _raises_system_exit(check_view_shape, p, "sparse", 6375)
        assert "holds 2 edge lines" in msg and "98,581" in msg, msg
        check_view_shape(p, "sparse", 6376)    # another list size: no pin
    real = os.path.join(_HERE, "..", "data", "biogrid_ppi_sparse.txt")
    if not os.path.isfile(real):
        return
    check_view_shape(real, "sparse", 6375)     # the tracked file is whole
    with open(real) as f:
        lines = f.readlines()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "biogrid_ppi_sparse.txt")
        for keep in (len(lines) // 2, len(lines) - 1):
            with open(p, "w") as f:
                f.writelines(lines[:keep])
            msg = _raises_system_exit(check_view_shape, p, "sparse", 6375)
            assert f"holds {keep:,} edge lines" in msg and "cut-short" in msg
        # CRLF endings and trailing blank lines: still the whole file
        with open(p, "w", newline="") as f:
            f.write("".join(lines).replace("\n", "\r\n") + "\r\n\r\n")
        check_view_shape(p, "sparse", 6375)


def _pair(*p):
    return np.array(p, dtype=np.int64).reshape(-1, 2)


def _toy_shared_tree(td, symbols, n_universe, arrays, n_folds, hgnc=True,
                     ppi="0\t1\n2\t3\n"):
    """A data/ + folds/ tree for SharedFoldTrainer; returns its Namespace.

    List_Proteins row i = symbols[i] = Entrez str(i + 1) through a toy HGNC
    cache (hgnc=False: an empty cache dir, so the mapper falls back to
    gene_id_mapping.tsv, which knows all symbols but the last). The universe
    is "1".."n_universe"; every cv1 fold gets the same `arrays`.
    """
    data, folds, out = (os.path.join(td, x) for x in ("d", "f", "o"))
    os.makedirs(os.path.join(data, "cache"))
    if hgnc:
        with open(os.path.join(data, "cache", "hgnc_complete_set.txt"),
                  "w") as f:
            f.write(_HGNC_HEADER)
            for i, g in enumerate(symbols, 1):
                f.write(f"HGNC:{i}\t{g}\t\t\t{i}\n")
    with open(os.path.join(data, "gene_id_mapping.tsv"), "w") as f:
        f.write("entrez_id\tsymbol\n")
        for i, g in enumerate(symbols[:-1], 1):
            f.write(f"{i}\t{g}\n")
    with open(os.path.join(data, "List_Proteins_in_SL.txt"), "w") as f:
        f.write("\n".join(symbols) + "\n")
    _write_views(data, len(symbols), ppi=ppi)
    os.makedirs(folds)
    with open(os.path.join(folds, "genes.txt"), "w") as f:
        f.write("\n".join(str(i) for i in range(1, n_universe + 1)) + "\n")
    for k in range(n_folds):
        fd = os.path.join(folds, "cv1", f"fold_{k}")
        os.makedirs(fd)
        for name, v in arrays.items():
            np.save(os.path.join(fd, f"{name}.npy"), v)
    return argparse.Namespace(folds_dir=folds, cv_type="cv1", data_path=data,
                              output_dir=out, nn_size=2, seed=0, num_folds=5,
                              allow_missing_views=False)


_TOY4_ARRAYS = {
    "train_pos": _pair(0, 1),
    "train_neg": _pair(2, 3),
    "test_pos": _pair(0, 2),
    "test_neg": _pair(1, 3),
    "val_pos": _pair(1, 2),
    "val_neg": _pair(0, 3),
}


def test_shared_trainer_fold_count_and_view_provenance():
    with tempfile.TemporaryDirectory() as td:
        args = _toy_shared_tree(td, ["AAA", "BBB", "CCC", "DDD"], 4,
                                _TOY4_ARRAYS, n_folds=2)
        data, folds, out = args.data_path, args.folds_dir, args.output_dir
        t = SharedFoldTrainer(args)
        assert t.args.num_folds == 2 == len(t.splits)
        prov = t.model_params_extra["support_views"]
        assert set(prov) == {"GO BP similarity", "GO CC similarity",
                             "BioGRID PPI"}
        assert not any(p.startswith("ZERO") for p in prov.values())
        assert t.model_params_extra["lp_mapping"] == {
            "id_space": "symbol->entrez", "rows": 4, "mapped": 4,
            "in_universe": 4}
        # model_params_extra round-trips through model_params.json (the real
        # hand-off in Trainer.train_fold_from_split is pinned by
        # test_shared_train_selection_matches_graded_matrix)
        from model_params import read_model_params, write_model_params
        write_model_params(out, torch.nn.Linear(2, 1), folds_dir=folds,
                           cv_type="cv1", **t.model_params_extra)
        assert read_model_params(out)["lp_mapping"]["id_space"] == \
            "symbol->entrez"
        os.remove(os.path.join(data, "Human_GOsim_CC.txt"))
        _raises_system_exit(SharedFoldTrainer, args)


def test_shared_trainer_refuses_static_lp_mapping():
    # HGNC unavailable (empty cache, no network): the static-table fallback
    # maps fewer rows, a REDUCED SLMGAE, so the trainer aborts unless
    # --allow_static_lp_mapping, and an opted-in run says so in lp_mapping.
    import urllib.request

    def offline(*a, **kw):
        raise OSError("offline")

    real = urllib.request.urlretrieve
    urllib.request.urlretrieve = offline
    try:
        with tempfile.TemporaryDirectory() as td:
            args = _toy_shared_tree(td, ["AAA", "BBB", "CCC", "DDD"], 4,
                                    _TOY4_ARRAYS, n_folds=1, hgnc=False)
            msg = _raises_system_exit(SharedFoldTrainer, args)
            args.allow_static_lp_mapping = True
            t = SharedFoldTrainer(args)
    finally:
        urllib.request.urlretrieve = real
    assert "--allow_static_lp_mapping" in msg and "3/4" in msg
    assert t.model_params_extra["lp_mapping"] == {
        "id_space": LP_STATIC_FALLBACK_SPACE, "rows": 4, "mapped": 3,
        "in_universe": 3}


_TOY14_ARRAYS = {
    "train_pos": _pair(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8),
    "train_neg": _pair(0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 9, 6, 10, 7, 11),
    "val_pos": _pair(8, 9, 0, 2, 9, 10),
    "val_neg": _pair(0, 12, 1, 13, 3, 11),
    "test_pos": _pair(10, 11, 11, 12, 1, 3),
    "test_neg": _pair(0, 13, 2, 12, 4, 10),
}


def test_shared_train_selection_matches_graded_matrix():
    # The real train() -> train_fold_from_split path. Val selection must score
    # what evaluate_model.py grades: R[i, j] of the saved matrix after its
    # float32 shift (score - min + 1). Run 2 scales R by 1e-8 so that shift
    # merges val scores into ties, which selecting on unshifted scores would
    # miss. Run 3 dies at fold 0: neither the earlier run's summary nor its
    # fold matrix or checkpoint may be left (model_params.json is rewritten at
    # each fold start, and nothing is graded after a failed training task).
    import json

    from auprg import auprg
    from model_params import read_model_params

    class Tiny(SLMGAE):
        def forward(self, *a, **kw):
            r, m, att, s = super().forward(*a, **kw)
            return r * 1e-8, m, att, s

    e = np.vstack([_TOY14_ARRAYS["val_pos"], _TOY14_ARRAYS["val_neg"]])
    y = np.r_[np.ones(3), np.zeros(3)]

    def run(args, model_cls=None):
        t = SharedFoldTrainer(args)
        if model_cls is not None:
            t.create_model = lambda: model_cls(
                num_nodes=t.num_nodes, num_features=t.num_features,
                hidden1=args.hidden1, hidden2=args.hidden2,
                dropout=args.dropout, device=t.device).to(t.device)
        t.train()
        M = np.load(os.path.join(args.output_dir,
                                 "fold_0_cv1_predictions.npy"))
        S = (M - M.min() + np.float32(1.0)).astype(np.float32)
        with open(os.path.join(args.output_dir,
                               "training_summary.json")) as f:
            best = json.load(f)["selection"][0]["best_value"]
        return (best, auprg(y, S[e[:, 0], e[:, 1]].astype(float)),
                auprg(y, M[e[:, 0], e[:, 1]].astype(float)),
                read_model_params(args.output_dir))

    with tempfile.TemporaryDirectory() as td:
        args = _toy_shared_tree(td, [f"G{i}" for i in range(1, 13)], 14,
                                _TOY14_ARRAYS, n_folds=1,
                                ppi="0\t1\n2\t3\n4\t5\n6\t7\n8\t9\n")
        vars(args).update(nn_size=3, hidden1=8, hidden2=4, dropout=0.2,
                          alpha=2.0, beta=4.0, coe=2.0, learning_rate=0.01,
                          epochs=10, eva_epochs=5, early_stopping=20,
                          select_metric="auprg")
        out = args.output_dir
        best, graded, _, mp = run(args)
        assert best == graded, (best, graded)
        assert mp["selection_regime"] == "val" and mp["seed"] == 0
        assert set(mp["support_views"]) == {"GO BP similarity",
                                            "GO CC similarity", "BioGRID PPI"}
        assert mp["lp_mapping"]["id_space"] == "symbol->entrez"

        args.output_dir = os.path.join(td, "o_tiny")
        best, graded, raw, _ = run(args, Tiny)
        assert best == graded and raw != graded, (best, graded, raw)

        args.output_dir = out
        ck = os.path.join(out, "checkpoints", "fold_0_best.pt")
        assert os.path.exists(ck)  # run 1's checkpoint, so the check below bites

        def boom(self, *a, **kw):
            raise RuntimeError("fold crashed")

        t = SharedFoldTrainer(args)
        t.train_fold_from_split = boom.__get__(t)
        try:
            t.train()
        except RuntimeError:
            pass
        else:
            raise AssertionError("train() did not propagate the crash")
        assert not os.path.exists(os.path.join(out, "training_summary.json"))
        assert not os.path.exists(
            os.path.join(out, "fold_0_cv1_predictions.npy"))
        assert not os.path.exists(ck)


def test_esm_generator_flushes_last_partial_batch():
    import generate_esm_embeddings_gpu as gen

    class Stub(gen.ESMEmbeddingGenerator):
        def __init__(self):  # no ESM model: only the batching is tested
            self.batch_size, self.model_name, self.device = 8, "stub", "cpu"
            self.gene_corrections = {}
            self.batches = []

        def generate_embeddings_batch(self, batch_data):
            self.batches.append([g for g, _ in batch_data])
            return {g: np.ones(1280, dtype=np.float32) for g, _ in batch_data}

    noncoding = sorted(gen.NON_CODING_GENES)[0]
    with tempfile.TemporaryDirectory() as td:
        fa = os.path.join(td, "seq.fasta")
        with open(fa, "w") as f:
            f.write(">AAA1\nMKV\n>BBB2\nMKL\n>CCC3\nMKT\n")
        for last in (noncoding, "ZMISSING", "CCC3"):
            order = os.path.join(td, "order.txt")
            genes = ["AAA1", "BBB2"] + (["CCC3"] if last != "CCC3" else []) \
                + [last]
            with open(order, "w") as f:
                f.write("\n".join(genes) + "\n")
            g = Stub()
            emb = g.process_dataset(fa, os.path.join(td, "o", "e.pt"), order)
            assert g.batches == [["AAA1", "BBB2", "CCC3"]], (last, g.batches)
            assert all(emb[x].any() for x in ("AAA1", "BBB2", "CCC3"))


def test_bc_build_main_view_adds_no_edges():
    from train_BC import build_main_view
    m = build_main_view(np.array([[0, 1], [1, 2]]), 4)
    assert m.shape == (4, 4) and np.allclose(m, m.T)
    assert m[0, 1] > 0 and m[1, 2] > 0
    assert m[0, 2] == 0 and m[2, 3] == 0  # pairs not passed in stay out
    assert m[3, 3] == 1.0  # isolated gene: self-loop only


def _toy_support_adjs(n, seed):
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(3):
        a = (rng.uniform(size=(n, n)) > 0.7).astype(np.float64)
        a = np.triu(a, 1)
        views.append(a + a.T)
    return views


def _off_diag_pairs(m):
    m = np.array(m, copy=True)
    np.fill_diagonal(m, 0)
    return {tuple(sorted(p)) for p in np.argwhere(m != 0).tolist()}


def test_bc_run_cv_main_view_is_each_folds_training_positives():
    # Spy on what the model propagates over (adjs[3]) in every fold of the
    # real run_cv -> train_fold path: a main view from all SL edges would put
    # the fold's held-out positives there.
    import train_BC
    n = 12
    pos = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                    [7, 8], [8, 9], [9, 10], [10, 11], [0, 11]])
    neg = np.array([[i, j] for i in range(n) for j in range(i + 2, n)
                    if (i, j) != (0, 11)][:24])
    seen, folds = [], []

    class Spy(train_BC.SLMGAE):
        def forward(self, features, adjs, coe=2.0):
            seen.append(adjs[3].detach().cpu().numpy())
            return super().forward(features, adjs, coe=coe)

    orig_fold, orig_model = train_BC.BCTrainer.train_fold, train_BC.SLMGAE

    def fold_spy(self, tr_e, tr_l, va_e, va_l, support_adjs, args):
        folds.append((tr_e[tr_l == 1], va_e[va_l == 1], len(seen)))
        return orig_fold(self, tr_e, tr_l, va_e, va_l, support_adjs, args)

    args = argparse.Namespace(epochs=2, eva_epochs=1, cv_folds=3, hidden1=8,
                              hidden2=4, dropout=0.0, lr=0.01, alpha=0.5,
                              beta=2.0, coe=1.0)
    np.random.seed(0)
    torch.manual_seed(0)
    train_BC.SLMGAE, train_BC.BCTrainer.train_fold = Spy, fold_spy
    try:
        t = train_BC.BCTrainer([f"G{i}" for i in range(n)], device="cpu")
        aucs, _ = t.run_cv(pos, neg,
                           train_BC.prepare_adjacencies(
                               _toy_support_adjs(n, 1)), args)
    finally:
        train_BC.SLMGAE, train_BC.BCTrainer.train_fold = orig_model, orig_fold
    assert len(folds) == 3 and len(aucs) == 3
    for tr_pos, va_pos, first in folds:
        assert len(va_pos) > 0
        want = {tuple(sorted(p)) for p in tr_pos.tolist()}
        for m in seen[first:first + args.epochs + 1]:  # epochs + final score
            assert _off_diag_pairs(m) == want
            assert all(m[a, b] == 0 for a, b in va_pos)


def test_case_study_holds_out_pairs_and_restores_best_state():
    import case_study as cs
    n = 14
    pos = [[i, i + 1] for i in range(n - 1)] + [[0, 13], [2, 9]]
    edges = np.array(pos + [[b, a] for a, b in pos])
    holdout = [(1, 0), (3, 12)]  # one known SL pair, one non-SL pair
    seen = {"state": [], "main": []}

    class Spy(cs.SLMGAE):
        def forward(self, features, adjs, coe=2.0):
            if not self.training:
                seen["state"].append(
                    {k: v.detach().clone() for k, v in
                     self.state_dict().items()})
            seen["main"].append(adjs[3].detach().cpu().numpy())
            return super().forward(features, adjs, coe=coe)

    # Scripted val AUCs for the evaluations at epochs 0, 10, 20: the FIRST is
    # best, so the restored model must be the epoch-0 one, not the last.
    aucs = iter([0.9, 0.1, 0.2])
    orig = cs.SLMGAE, cs.roc_auc_score
    args = argparse.Namespace(use_esm=False, epochs=21, hidden1=8, hidden2=4,
                              dropout=0.0, learning_rate=0.01, alpha=2.0,
                              beta=4.0, coe=2.0)
    np.random.seed(0)
    torch.manual_seed(0)
    cs.SLMGAE, cs.roc_auc_score = Spy, lambda y, p: next(aucs)
    try:
        ev = cs.CaseStudyEvaluator(edges, [f"G{i}" for i in range(n)], {},
                                   args, device="cpu")
        ev.train_model(cs.prepare_adjacencies(_toy_support_adjs(n, 2)), args,
                       holdout_pairs=holdout)
    finally:
        cs.SLMGAE, cs.roc_auc_score = orig
    s0, s_last, s_final = seen["state"][0], seen["state"][2], seen["state"][3]
    assert all(torch.equal(s0[k], s_final[k]) for k in s0)
    assert not all(torch.equal(s_last[k], s_final[k]) for k in s0)
    held = {tuple(sorted(p)) for p in holdout}
    split = {k: {tuple(sorted(p)) for p in v.tolist()}
             for k, v in ev.split_edges.items()}
    assert not held & (split["train"] | split["val"])
    pos_set = {tuple(sorted(p)) for p in pos}
    train_pos = split["train"] & pos_set
    assert split["val"] & pos_set  # the check below is not vacuous
    for m in seen["main"]:
        assert _off_diag_pairs(m) == train_pos


def test_case_study_scores_upper_triangle():
    # Training, val selection and the threshold use i < j and R != R^T, so a
    # pair listed as (j, i) must still be scored at R[i, j].
    import contextlib
    import io

    import case_study as cs
    n = 6
    R = torch.arange(n * n, dtype=torch.float32).reshape(n, n)

    class Stub(torch.nn.Module):
        def forward(self, features, adjs, coe=2.0):
            return R, R, R, []

    args = argparse.Namespace(use_esm=False, coe=2.0)
    edges = np.array([[1, 4], [4, 1]])
    with contextlib.redirect_stdout(io.StringIO()):
        ev = cs.CaseStudyEvaluator(edges, [f"G{i}" for i in range(n)], {},
                                   args, device="cpu")
        pairs = [{"gene1": "G4", "gene2": "G1", "idx1": 4, "idx2": 1},
                 {"gene1": "G1", "gene2": "G4", "idx1": 1, "idx2": 4}]
        res = ev.evaluate_case_study(Stub(), None, None, pairs, edges,
                                     R[1, 4].item() - 0.5)
    assert [r["predicted_score"] for r in res] == [R[1, 4].item()] * 2
    assert [r["prediction"] for r in res] == ["SL", "SL"]
    assert [r["gene1"] for r in res] == ["G4", "G1"]  # listed order kept
    assert all(r["is_known_sl"] for r in res)


def test_case_study_pairs_are_symbols():
    from case_study import get_case_study_pairs
    lp = os.path.join(_HERE, "..", "data", "List_Proteins_in_SL.txt")
    with open(lp) as f:
        g2i = {g.strip(): i for i, g in enumerate(f) if g.strip()}
    pairs = get_case_study_pairs(g2i)
    assert len(pairs) == 10
    assert pairs[0]["gene1"] == "BRCA1" and pairs[0]["idx1"] == g2i["BRCA1"]


def test_gene_corrections_agree_with_hgnc_prev_symbols():
    # A correction for a retired symbol that HGNC assigns to exactly one gene
    # (as prev/alias) must land on that gene. Caught HIST4H4 -> H4C1 (=
    # HIST1H4A, 8359) where HGNC says H4C16 (121504).
    hgnc = os.path.join(_HERE, "..", "data", "cache", "hgnc_complete_set.txt")
    if not os.path.exists(hgnc):
        print("  (skipped: no HGNC cache)")
        return
    sys.path.insert(0, os.path.join(_HERE, "..", "siamese_sl"))
    from gene_corrections_config import GENE_CORRECTIONS
    from gene_name_utils import GeneNameMapper
    current, claims, own = set(), {}, {}
    with open(hgnc, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            current.add(row["symbol"].upper())
            own[row["symbol"].upper()] = row["entrez_id"] or None
            for col in ("prev_symbol", "alias_symbol"):
                for a in filter(None, row[col].split("|")):
                    claims.setdefault(a.upper(), set()).add(row["entrez_id"])
    mapper = GeneNameMapper(cache_dir=os.path.dirname(hgnc))
    bad = {
        k: (mapper.symbol_to_entrez(k), claims[k.upper()])
        for k in GENE_CORRECTIONS if k.upper() not in current
        and len(claims.get(k.upper(), ())) == 1
        and {mapper.symbol_to_entrez(k)} != claims[k.upper()]
    }
    assert not bad, bad
    assert mapper.symbol_to_entrez("HIST4H4") == "121504"
    # A key that is itself a CURRENT symbol keeps its own gene (TCL6 ->
    # TCL6f4 once made TCL6 unmappable). The one deliberate override is the
    # curated SETP10 -> SEPT10 "typo correction", which moves List_Proteins
    # row 4876 off the pseudogene SETP10 (100287822) onto SEPTIN10 (151011).
    # That row is not unused: SETP10's own similarities are all zero in the
    # shipped files, but the loader's preserved column shift (see
    # slmgae_pytorch.load_dense_feature: rows i >= (n-1)/2 read nothing of
    # their own line) still leaves it with 45 GO-BP / 47 GO-CC edges at
    # nn_size=45 (0 PPI). The override is inert only because NEITHER Entrez
    # id is in the shared universe, so remap_support drops the row whichever
    # way it resolves — pinned by test_setp10_correction_is_inert_for_slmgae.
    moved = {
        k: (mapper.symbol_to_entrez(k), own[k.upper()])
        for k in GENE_CORRECTIONS if k.upper() in own
        and k.upper() != "SETP10"
        and mapper.symbol_to_entrez(k) != own[k.upper()]
    }
    assert not moved, moved
    assert mapper.symbol_to_entrez("TCL6") == "27004"


def test_setp10_correction_is_inert_for_slmgae():
    # The SETP10 exemption in the test above is safe only while NEITHER of the
    # two candidate Entrez ids is in the shared universe: remap_support then
    # drops List_Proteins row 4876 whichever gene the correction picks. The
    # row itself carries view edges (45 GO-BP / 47 GO-CC at nn_size=45), so if
    # either id enters the universe the correction starts deciding which gene
    # receives them. Cheap file check; no views are loaded.
    genes_txt = os.path.join(_HERE, "..", "sl_comparison", "folds",
                             "genes.txt")
    if not os.path.exists(genes_txt):
        print("  (skipped: no shared folds)")
        return
    with open(genes_txt) as f:
        universe = {ln.strip() for ln in f if ln.strip()}
    hit = {"151011", "100287822"} & universe  # SEPTIN10, SETP10
    assert not hit, (
        f"{sorted(hit)} entered the shared universe: the SETP10 -> SEPT10 "
        f"correction now decides which gene gets List_Proteins row 4876's "
        f"support-view edges, so the SETP10 exemption in "
        f"test_gene_corrections_agree_with_hgnc_prev_symbols needs a "
        f"re-audit against gene_corrections_config.")


def test_data_split_docstrings_disclose_its_local_logic():
    # data_split.py is a wrapper around the vendored SL_benchmark splitter, but
    # not a pure one: it pre-trims the cv1 negative pool itself and retries the
    # split under a bumped seed. Its header once claimed "No custom logic",
    # which hid both from anyone auditing that (retired) path for fidelity.
    # Parsed, not imported: importing data_split pulls in
    # preprocess_benchmarking_paper, which reseeds random/numpy at import.
    import ast
    with open(os.path.join(_HERE, "data_split.py"), encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    tree = ast.parse(src)
    mod_doc = ast.get_docstring(tree) or ""
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "SLDataSplitter")
    cls_doc = ast.get_docstring(cls) or ""
    body = {n.name: "\n".join(lines[n.lineno - 1:n.end_lineno])
            for n in cls.body if isinstance(n, ast.FunctionDef)}
    # the two local steps still exist ...
    assert "random.Random" in body["_run_cv"], body["_run_cv"]
    assert "self.random_state + attempt" in body["_split_with_retry"]
    # ... so both docstrings must say so, and neither may deny it.
    for doc in (mod_doc, cls_doc):
        low = doc.lower()
        assert "no custom logic" not in low, doc
        assert "only does two things" not in low, doc
        assert "pre-trim" in low, doc
        assert "retr" in low, doc  # retry / retries
    assert "_run_cv" in mod_doc and "_split_with_retry" in mod_doc


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
