#!/usr/bin/env python3
"""Deterministic unit tests for the pure-PyTorch original SLGNN.

The model replaces torch_scatter.scatter_mean and dgl.nn.GATConv with plain
torch code; these tests pin each replacement to the formula the library
computes (written out densely here), plus the KG loader's inverse edges and
isolated-entity rule, the score matrix's orientation, the training rows (both
orientations of each pair, as the original's fold files), the build's
label-keyed-deletion measurement, the gene-mapping fingerprint, the standalone
launcher's KG-provenance gate, the trainer's --folds_dir requirement and that
train_fold's test metrics are those of the matrix it exports. No training
beyond one toy epoch.

    python -m pytest SLGNN/test_slgnn_orig.py -q     (from the repo root)
    python SLGNN/test_slgnn_orig.py
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slgnn_orig_model import (GATConv, SLGNNOriginal, edge_softmax,  # noqa
                              scatter_mean)
import build_synlethkg  # noqa: E402
from build_synlethkg import (BUILDER_REL, MAPPING_FILES,  # noqa: E402
                             load_kg_for_genes, mapping_sha256,
                             measure_label_keyed_deletion)
import train_slgnn  # noqa: E402


def _toy_graph(n=6):
    # symmetric edges + self-loops, as SLGNNOriginal builds them
    und = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 5)]
    src = [a for a, b in und] + [b for a, b in und] + list(range(n))
    dst = [b for a, b in und] + [a for a, b in und] + list(range(n))
    return torch.tensor(src), torch.tensor(dst), n


def test_scatter_mean_matches_torch_scatter_semantics():
    g = torch.Generator().manual_seed(0)
    src = torch.randn(9, 3, generator=g)
    index = torch.tensor([0, 2, 2, 5, 0, 0, 5, 2, 2])
    got = scatter_mean(src, index, 7)
    want = torch.zeros(7, 3)
    for r in range(7):
        rows = src[index == r]
        if len(rows):
            want[r] = rows.mean(0)  # rows without entries stay 0
    assert torch.allclose(got, want, atol=1e-6)


def test_edge_softmax_is_softmax_over_incoming_edges():
    src, dst, n = _toy_graph()
    g = torch.Generator().manual_seed(1)
    e = torch.randn(len(src), 8, 1, generator=g) * 5
    got = edge_softmax(e, dst, n)
    want = torch.empty_like(e)
    for v in range(n):
        m = dst == v
        want[m] = torch.softmax(e[m], dim=0)
    assert torch.allclose(got, want, atol=1e-6)


def test_edge_softmax_gradient_is_dgl_backward():
    src, dst, n = _toy_graph()
    g = torch.Generator().manual_seed(2)
    e = torch.randn(len(src), 2, 1, generator=g).requires_grad_(True)
    grad_out = torch.randn(len(src), 2, 1, generator=g)
    out = edge_softmax(e, dst, n)
    out.backward(grad_out)
    # DGL EdgeSoftmax.backward: sds - out * dst_sum(sds), sds = out * grad
    o = out.detach()
    sds = o * grad_out
    acc = torch.zeros(n, 2, 1).index_add_(0, dst, sds)
    want = sds - o * acc[dst]
    assert torch.allclose(e.grad, want, atol=1e-6)


def test_gatconv_matches_dense_gat_formula():
    torch.manual_seed(3)
    src, dst, n = _toy_graph()
    conv = GATConv(4, 5, 3, 0.2, 0.2, 0.2, residual=True,
                   activation=torch.nn.functional.leaky_relu).eval()
    with torch.no_grad():
        conv.bias.normal_()  # DGL starts it at 0; exercise it anyway
    x = torch.randn(n, 4)
    got = conv(src, dst, n, x)
    # dense DGL 0.7.2 formula
    ft = (x @ conv.fc.weight.t()).view(n, 3, 5)
    el = (ft * conv.attn_l).sum(-1)  # (n, 3)
    er = (ft * conv.attn_r).sum(-1)
    adj = torch.zeros(n, n, dtype=torch.bool)
    adj[src, dst] = True  # adj[u, v]: edge u -> v
    want = torch.empty(n, 3, 5)
    for v in range(n):
        us = torch.where(adj[:, v])[0]
        logits = torch.nn.functional.leaky_relu(el[us] + er[v], 0.2)
        a = torch.softmax(logits, dim=0)  # (deg, 3)
        want[v] = (a.unsqueeze(-1) * ft[us]).sum(0)
    want = want + (conv.res_fc(x) if not isinstance(
        conv.res_fc, torch.nn.Identity) else x).view(n, -1, 5)
    want = want + conv.bias.view(1, 3, 5)
    want = torch.nn.functional.leaky_relu(want)
    assert torch.allclose(got, want, atol=1e-5)


def _toy_kg_dir(tmp):
    # 4 KG entities, 2 relations; entity 3 is gene 10's, entity 1 gene 20's
    kg = np.array([[0, 0, 1], [1, 1, 2], [3, 0, 1], [0, 0, 1]], np.int64)
    np.save(os.path.join(tmp, "kg_triples.npy"), kg)
    with open(os.path.join(tmp, "entity_entrez.tsv"), "w") as f:
        f.write("entity\tname\tentrez\tmethod\tdegree\tselected\n"
                "1\tB\t20\tsynlethdb\t3\t1\n"
                "3\tD\t10\tsynlethdb\t1\t1\n"
                "2\tC\t30\thgnc_prev\t1\t0\n")
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"n_entities": 4, "n_relations": 2, "source_commit": "x"},
                  f)


def test_kg_loader_inverse_edges_and_isolated_genes():
    with tempfile.TemporaryDirectory() as tmp:
        _toy_kg_dir(tmp)
        kg = load_kg_for_genes(tmp, ["10", "99", "20", "30"])
    # genes 10 / 20 -> their entities; 99 and 30 (not selected) -> new ids
    assert kg["gene_entity"].tolist() == [3, 4, 1, 5]
    assert kg["n_entities"] == 6 and kg["n_in_kg"] == 2
    assert kg["n_relations"] == 4
    # 6 rows, not 8: the duplicated [0, 0, 1] collapses (the set below
    # cannot see that); every (h, r, t) gains (t, h, r + 2)
    assert kg["edge_index"].shape == (2, 6) and kg["edge_type"].shape == (6, )
    trip = set(map(tuple, np.vstack([kg["edge_index"], kg["edge_type"]]).T))
    assert trip == {(0, 1, 0), (1, 0, 2), (1, 2, 1), (2, 1, 3), (3, 1, 0),
                    (1, 3, 2)}


def test_score_matrix_orientation_and_symmetry():
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as tmp:
        _toy_kg_dir(tmp)
        genes = ["10", "99", "20", "30"]
        kg = load_kg_for_genes(tmp, genes)
    adj = sp.csr_matrix((np.ones(2), ([0, 2], [2, 0])), shape=(4, 4))
    model = SLGNNOriginal(4, kg["n_entities"], kg["n_relations"],
                          kg["gene_entity"], kg["edge_index"], kg["edge_type"],
                          adj, dim=8, context_hops=2, n_factors=2)
    S = model.score_matrix()
    assert not model.training
    ent, gen, _ = model.generate()
    full = (gen @ ent[model.gene_entity].t()).numpy()
    iu = np.triu_indices(4, 1)
    assert np.allclose(S[iu], full[iu]) and np.allclose(S, S.T)
    assert np.all(np.diag(S) == 0)
    # training forward: (sigmoid scores, emb_loss, cor_loss, cor)
    model.train()
    p, emb_loss, cor_loss, cor = model(torch.tensor([0, 1]),
                                       torch.tensor([2, 3]))
    assert p.shape == (2, ) and bool(((p > 0) & (p < 1)).all())
    (p.sum() + emb_loss + cor_loss).backward()
    assert model.all_embed.grad is not None


def test_original_train_pairs_both_orientations():
    e = np.array([[0, 2], [1, 3], [0, 1]], np.int64)
    y = np.array([1.0, 0.0, 0.0])
    got_e, got_y = train_slgnn.original_train_pairs(e, y, seed=7)
    assert got_e.dtype == np.int64 and got_y.dtype == np.float32
    rows = sorted(zip(map(tuple, got_e.tolist()), got_y.tolist()))
    assert rows == sorted([((0, 2), 1.0), ((2, 0), 1.0), ((1, 3), 0.0),
                           ((3, 1), 0.0), ((0, 1), 0.0), ((1, 0), 0.0)])
    again_e, again_y = train_slgnn.original_train_pairs(e, y, seed=7)
    assert np.array_equal(got_e, again_e) and np.array_equal(got_y, again_y)


def test_build_original_trains_both_orientations():
    torch.manual_seed(5)
    with tempfile.TemporaryDirectory() as tmp:
        _toy_kg_dir(tmp)
        kg = load_kg_for_genes(tmp, ["10", "99", "20", "30"])
    pairs = np.array([[0, 2], [1, 3], [0, 1]], np.int64)
    split = {
        "train_adj": sp.csr_matrix((np.ones(2), ([0, 2], [2, 0])),
                                   shape=(4, 4)),
        "train_edges": pairs.copy(),
        "train_labels": np.array([1.0, 0.0, 0.0]),
    }
    args = argparse.Namespace(dim=8, n_hops=2, n_factors=2, ind="distance",
                              l2_weight=1e-4, sim_regularity=1e-3,
                              no_node_dropout=False, node_dropout_rate=0.5,
                              no_mess_dropout=False, mess_dropout_rate=0.1,
                              seed=0, batch_size=4)
    model, run_epoch = train_slgnn.build_original(args, split, kg,
                                                  torch.device("cpu"), 0)
    fed, forward = [], model.forward

    def spy(a, b):
        fed.extend(zip(a.tolist(), b.tolist()))
        return forward(a, b)

    model.forward = spy
    run_epoch(torch.optim.Adam(model.parameters(), lr=1e-3))
    want = [tuple(p) for p in pairs.tolist()]
    assert sorted(fed) == sorted(want + [(b, a) for a, b in want])
    # the fold's own pair list is untouched (train_fold masks seen pos with it)
    assert np.array_equal(split["train_edges"], pairs)


def test_label_keyed_deletion_measurement():
    # SL on {0,1} and {2,3}; {0,1} is also non-SL; SR on {1,2}
    lab = pd.DataFrame({"h": [0, 3, 1, 1],
                        "type": ["SL_GsG", "SL_GsG", "NONSL_GnsG", "SR_GsrG"],
                        "t": [1, 2, 0, 2]})
    kept = pd.DataFrame({
        "h": [1, 2, 1, 0, 5],
        "type": ["INTERACTS_GiG", "REGULATES_GrG", "INTERACTS_GiG",
                 "PARTICIPATES_GpBP", "INTERACTS_GiG"],
        "t": [0, 3, 2, 7, 6]})
    # kg2id keeps only the SR pair's edge (and the non-gene-gene row)
    kg_gg = pd.DataFrame({"h": [1], "r": [4], "t": [2],
                          "type": ["INTERACTS_GiG"]})
    absent = {(1, "INTERACTS_GiG", 0), (2, "REGULATES_GrG", 3),
              (5, "INTERACTS_GiG", 6)}
    got = measure_label_keyed_deletion(lab, absent, kept, kg_gg)
    assert got["absent_rows_by_relation"] == {"INTERACTS_GiG": 2,
                                              "REGULATES_GrG": 1}
    assert got["absent_rows_by_pair_label"] == {"NONSL_GnsG": 1, "SL_GsG": 2,
                                                "unlabelled": 1}
    assert got["absent_rows_not_on_SL_pairs"] == 1
    lp = got["labelled_pairs"]
    assert lp["SL_GsG"] == {"pairs": 2, "with_gene_gene_edge_in_kg_triplet": 2,
                            "with_gene_gene_edge_in_kg2id": 0}
    assert lp["SR_GsrG"] == {"pairs": 1,
                             "with_gene_gene_edge_in_kg_triplet": 1,
                             "with_gene_gene_edge_in_kg2id": 1}
    assert lp["NONSL_GnsG"] == {"pairs": 1,
                                "with_gene_gene_edge_in_kg_triplet": 1,
                                "with_gene_gene_edge_in_kg2id": 0}


def test_mapping_sha256_covers_what_map_gene_entities_reads():
    # meta.json's mapping_sha256 must fingerprint exactly the files the
    # Entrez mapping reads, keyed by repo-relative path, plus this builder
    # itself (its METHODS / resolve_entrez rules move genes with no input
    # file changing), as NSF4SL/build_transe_features.py records its own
    read, saved = [], {}
    fakes = {
        "_read_synlethdb": lambda paths: read.extend(paths) or {},
        "_read_hgnc": lambda path: read.append(path) or ({}, {}, {}),
        "_read_ncbi": lambda path: read.append(path) or ({}, {}),
    }
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        (raw / "entity2id.txt").write_text("a\tb\n100\t0\n101\t1\n")
        (raw / "dbid2name.csv").write_text(",_id,name\n0,100,A\n1,101,B\n")
        kg = np.array([[0, 3, 1]], np.int64)  # REGULATES_GrG, gene-gene
        try:
            for k, v in fakes.items():
                saved[k] = getattr(build_synlethkg, k)
                setattr(build_synlethkg, k, v)
            build_synlethkg.map_gene_entities(raw, kg)
        finally:
            for k, v in saved.items():
                setattr(build_synlethkg, k, v)
    root = build_synlethkg.REPO_DIR
    assert sorted(read) == sorted(root / r for r in MAPPING_FILES.values())
    with tempfile.TemporaryDirectory() as tmp:
        for i, rel in enumerate(MAPPING_FILES.values()):
            f = Path(tmp) / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"v%d" % i)
        before = mapping_sha256(tmp)
        (Path(tmp) / MAPPING_FILES["hgnc"]).write_bytes(b"new")
        after = mapping_sha256(tmp)
    assert set(before) == set(MAPPING_FILES.values()) | {BUILDER_REL}
    # hashed from __file__, so it is the same in `before` and `after`
    assert before[BUILDER_REL] == hashlib.sha256(
        Path(build_synlethkg.__file__).read_bytes()).hexdigest()
    assert {k for k in before if before[k] != after[k]} == {
        MAPPING_FILES["hgnc"]}
    # check_shared_inputs.py resolves every recorded key as root / key
    assert (build_synlethkg.REPO_DIR / BUILDER_REL).resolve() == Path(
        build_synlethkg.__file__).resolve()


def test_standalone_launcher_gates_the_default_kg_on_provenance():
    # slurm/run_slgnn.sh is a documented standalone path (README, "Run"), so a
    # data/synlethkg left over from an older build_synlethkg.py, or mapped from
    # another copy of the HGNC / NCBI / SynLethDB tables, must stop it BEFORE
    # training rather than produce results/slgnn_standalone_<cv>/ on a wrong
    # Entrez -> entity mapping. An existence test alone cannot see that.
    sh = (Path(__file__).resolve().parent / "slurm" /
          "run_slgnn.sh").read_text()
    gate = sh.index("--models SLGNN")
    assert "check_shared_inputs.py" in sh[:gate]
    assert gate < sh.index("train_slgnn.py")          # before any training
    # refuses, not merely warns, and only for the file the checker knows
    assert "exit 1" in sh[gate:sh.index("INPUT_ARGS=(--kg_dir")]
    assert '[ "$KG_DIR" = "../data/synlethkg" ]' in sh
    assert "NOT checked" in sh


def test_train_fold_reports_the_matrix_it_exports():
    # A stub whose score_matrix differs on every call (as index_add_ can on
    # CUDA): results.json's test metrics must be the SAVED matrix's, so the
    # restored checkpoint is scored exactly once.
    n, rng = 40, np.random.RandomState(0)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def score_matrix(self):
            self.calls += 1
            m = np.triu(np.random.RandomState(self.calls).randn(n, n), 1)
            return m + m.T  # float64: the export casts to float32 first

    stub = Stub()
    iu = np.array(np.triu_indices(n, 1)).T
    iu = iu[rng.permutation(len(iu))]
    lab = np.triu(rng.rand(n, n) < 0.5, 1)
    lab = lab | lab.T

    def y(e):
        return lab[e[:, 0], e[:, 1]].astype(np.int64)

    tr, va, te = iu[:300], iu[300:500], iu[500:780]
    split = {"train_adj": sp.csr_matrix((n, n)), "train_edges": tr,
             "train_labels": y(tr), "val_pos": va[y(va) == 1],
             "val_neg": va[y(va) == 0], "test_edges": te,
             "test_labels": y(te), "fold": 0}
    fakes = {"build_original": lambda *a: (stub, lambda opt: (0.0, 0.0)),
             "write_model_params": lambda *a, **k: None}
    saved = {k: getattr(train_slgnn, k) for k in fakes}
    try:
        for k, v in fakes.items():
            setattr(train_slgnn, k, v)
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(arch="original", output_dir=tmp,
                                      cv_type="cv1", epochs=2,
                                      eval_interval=1, early_stop=5,
                                      select_metric="auprg",
                                      learning_rate=1e-3, seed=0,
                                      folds_dir=None)
            with open(os.devnull, "w") as null, \
                    contextlib.redirect_stdout(null):
                got = train_slgnn.train_fold(args, 0, 1, split,
                                             torch.device("cpu"))
            mat = np.load(os.path.join(tmp, "fold_0_cv1_predictions.npy"))
    finally:
        for k, v in saved.items():
            setattr(train_slgnn, k, v)
    assert stub.calls == 3  # 2 validation passes + 1 after the restore
    assert mat.dtype == np.float32
    te32 = te.astype(np.int32)
    want = train_slgnn.evaluate_fold(mat, te32[y(te) == 1], te32[y(te) == 0],
                                     tr[y(tr) == 1].astype(np.int32))
    assert all(got[k] == want[k] for k in train_slgnn.METRIC_KEYS)


def test_trainer_requires_shared_folds():
    # the shared folds are the only split source with a validation split, so
    # --folds_dir is required and the legacy SLDataSplitter flags are gone
    argv = sys.argv
    try:
        sys.argv = ["train_slgnn.py", "--cv_type", "cv1"]
        with open(os.devnull, "w") as null:
            stderr, sys.stderr = sys.stderr, null
            try:
                train_slgnn.parse_args()
                raised = False
            except SystemExit:
                raised = True
            finally:
                sys.stderr = stderr
        assert raised
        sys.argv = ["train_slgnn.py", "--folds_dir", "x"]
        args = train_slgnn.parse_args()
    finally:
        sys.argv = argv
    assert args.folds_dir == "x"
    assert not any(hasattr(args, k) for k in ("num_folds", "train_ratio",
                                              "pos_neg_ratio", "data_path"))


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
