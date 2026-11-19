#!/usr/bin/env python3
"""Single-output mode on the shared folds (train.py without --use_cell_lines).

Covers the path that makes it trainable: labels read from the shared fold
files (one column, mask 1), the --sl_path agreement check, the train/val/test
partition kept exactly, and SiameseSL's dense all-pairs matrix (the export
sl_comparison grades) matching model.forward.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_single_output.py -q
    $PYTHON_PATH siamese_sl/tests/test_single_output.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

from data_loader import SLDataManager, create_val_dataloader
from retrieval_metrics import build_any_score_matrix
from siamese_esm import SiameseSL

N_GENES = 160
PARTS = ("train_pos", "train_neg", "val_pos", "val_neg", "test_pos",
         "test_neg")


def make_shared_folds(root, n_genes=N_GENES, n_pairs=1500, num_folds=5,
                      seed=0):
    """Write a tiny embedding .pt, SL file and cv1 shared-fold tree.

    Genes are Entrez-like digit strings. The embedding holds every shared gene
    plus 10 extras, in a different order from genes.txt (int-sorted, as
    prepare_folds.py writes it). A pair is SL when its latent dot product is
    positive. Returns (emb_path, sl_path, folds_dir, shared_genes, labels)
    with labels[(a, b)] in shared-index space.
    """
    rng = np.random.RandomState(seed)
    shared = sorted(rng.choice(np.arange(100, 100000), n_genes,
                               replace=False).tolist())
    shared = [str(g) for g in shared]
    extras = [str(200000 + k) for k in range(10)]
    emb_genes = list(reversed(shared)) + extras
    lat = rng.randn(len(emb_genes), 6).astype(np.float32)
    emb_path = os.path.join(root, "all_genes_toy.pt")
    torch.save({"raw_embeddings": torch.from_numpy(lat),
                "gene_order": emb_genes}, emb_path)
    row = {g: i for i, g in enumerate(emb_genes)}

    labels = {}
    while len(labels) < n_pairs:
        a, b = sorted(rng.choice(n_genes, 2, replace=False).tolist())
        s = float(lat[row[shared[a]]] @ lat[row[shared[b]]])
        labels[(a, b)] = int(s > 0.0)
    with open(os.path.join(root, "sl.txt"), "w") as f:
        for (a, b), y in sorted(labels.items()):
            if y:
                f.write(f"{shared[a]}\t{shared[b]}\n")

    folds_dir = os.path.join(root, "folds")
    os.makedirs(os.path.join(folds_dir, "cv1"))
    with open(os.path.join(folds_dir, "genes.txt"), "w") as f:
        f.write("\n".join(shared) + "\n")
    keys = sorted(labels)
    order = rng.permutation(len(keys))
    groups = np.array_split(order, num_folds)
    for k in range(num_folds):
        test = [keys[i] for i in groups[k]]
        rest = [keys[i] for g in range(num_folds) if g != k
                for i in groups[g]]
        n_val = len(rest) // 10
        split = {"val": rest[:n_val], "train": rest[n_val:], "test": test}
        d = os.path.join(folds_dir, "cv1", f"fold_{k}")
        os.makedirs(d)
        for part, prs in split.items():
            for y, tag in ((1, "pos"), (0, "neg")):
                arr = np.array([p for p in prs if labels[p] == y],
                               dtype=np.int32).reshape(-1, 2)
                np.save(os.path.join(d, f"{part}_{tag}.npy"), arr)
    return (emb_path, os.path.join(root, "sl.txt"), folds_dir, shared,
            labels)


def _manager(emb_path, sl_path, folds_dir):
    return SLDataManager(embeddings_paths=[emb_path], sl_pairs_path=sl_path,
                         folds_dir=folds_dir, seed=0)


def test_fold_labels_match_the_fold_files_exactly():
    with tempfile.TemporaryDirectory() as root:
        emb, sl, fd, shared, labels = make_shared_folds(root)
        dm = _manager(emb, sl, fd)
        assert dm.fold_labels and not dm.use_cell_lines
        folds = dm.get_cv1_splits(num_folds=5)
        assert len(folds) == 5
        for k, fold in enumerate(folds):
            d = os.path.join(fd, "cv1", f"fold_{k}")
            for part in ("train", "val", "test"):
                got = {}
                for (i, j), y, m in zip(fold[part]["pairs"].tolist(),
                                        fold[part]["labels"][:, 0],
                                        fold[part]["mask"][:, 0]):
                    assert m == 1.0
                    gi, gj = dm.idx_to_gene[i], dm.idx_to_gene[j]
                    a, b = sorted((shared.index(gi), shared.index(gj)))
                    got[(a, b)] = int(y)
                want = {}
                for tag, y in (("pos", 1), ("neg", 0)):
                    for a, b in np.load(os.path.join(d, f"{part}_{tag}.npy")):
                        want[(int(a), int(b))] = y
                assert got == want, (k, part)
            # canonical, full-space, disjoint
            for part in ("train", "val", "test"):
                p = fold[part]["pairs"]
                assert (p[:, 0] < p[:, 1]).all()
            tr = set(map(tuple, fold["train"]["pairs"].tolist()))
            assert not tr & set(map(tuple, fold["val"]["pairs"].tolist()))
            assert not tr & set(map(tuple, fold["test"]["pairs"].tolist()))
            assert create_val_dataloader(dm.get_fold_embeddings(fold)[0],
                                         fold) is not None


def test_sl_file_disagreeing_with_the_folds_raises():
    with tempfile.TemporaryDirectory() as root:
        emb, sl, fd, shared, labels = make_shared_folds(root)
        # drop one fold positive from the SL file
        lines = open(sl).read().splitlines()
        with open(sl, "w") as f:
            f.write("\n".join(lines[1:]) + "\n")
        dm = _manager(emb, sl, fd)
        try:
            dm.get_cv1_splits(num_folds=5)
        except RuntimeError as e:
            assert "disagree" in str(e)
        else:
            raise AssertionError("a fold positive missing from --sl_path "
                                 "must raise")


def test_siamese_sl_dense_matrix_matches_forward():
    torch.manual_seed(0)
    m = SiameseSL(input_dim=6, encoder_dims=[8, 8, 4], dropout=0.2,
                  pd_epsilon=0.01)
    with torch.no_grad():
        m.scoring_bias.fill_(-0.7)
        m.log_temperature.fill_(0.4)
    m.eval()
    x = torch.randn(40, 6)
    S = build_any_score_matrix(m, x, row_block=7)
    i, j = torch.triu_indices(40, 40, offset=1)
    ref = m(x[i], x[j])[:, 0]
    assert torch.allclose(S[i, j], ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(S, S.T)


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
