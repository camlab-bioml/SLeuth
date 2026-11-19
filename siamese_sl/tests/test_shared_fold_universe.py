#!/usr/bin/env python3
"""The shared folds and siamese's labelled pair universe must agree.

Cell-line mode takes the train/val/test PARTITION from the shared folds but
its own labels from --sl_path / --neg_pairs_path. A fold pair that is absent
from those files used to be dropped silently: SLeuth then trained on fewer
pairs than SLMGAE / SLGNN / NSF4SL on the same folds, and a dropped TEST
pair's genes stayed inside the per-fold preprocessing fit (_fold_fit_mask sees
only the pairs that survived) while evaluate_model still graded them off the
dense matrix. _shared_ml_folds now raises instead. A gene with no embedding row
is a different, unavoidable miss and stays a counted drop (except in val).

    $PYTHON_PATH -m pytest siamese_sl/tests/test_shared_fold_universe.py -q
    $PYTHON_PATH siamese_sl/tests/test_shared_fold_universe.py
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

from data_loader import SLDataManager

N_GENES = 24
# Pairs the siamese SL/non-SL files always carry.
CORE_POS = [(10, 11), (12, 13), (14, 15)]
CORE_NEG = [(10, 12), (11, 13), (16, 17)]
# Pairs the shared folds always carry; `extra_in_files` decides whether the
# siamese files know them too. (18, 19) is a train pair, (20, 21) a test pair.
EXTRA_POS = [(18, 19), (20, 21)]


def make_case(root, extra_in_files=True, drop_embedding_for=None):
    """Write a toy embedding, SL/non-SL files + sidecars and a cv3 fold tree.

    Returns (emb_path, sl_path, neg_path, folds_dir, shared_genes).
    `drop_embedding_for` is a shared-gene position whose row is left out of the
    embedding file, to exercise the "gene has no embedding row" miss.
    """
    genes = [str(100 + k) for k in range(N_GENES)]
    lat = np.random.RandomState(0).randn(N_GENES, 5).astype(np.float32)
    emb_genes, emb_rows = list(genes), lat
    if drop_embedding_for is not None:
        keep = [k for k in range(N_GENES) if k != drop_embedding_for]
        emb_genes = [genes[k] for k in keep]
        emb_rows = lat[keep]
    emb_path = os.path.join(root, "all_genes_toy.pt")
    torch.save({"raw_embeddings": torch.from_numpy(emb_rows),
                "gene_order": emb_genes}, emb_path)

    pos = CORE_POS + (EXTRA_POS if extra_in_files else [])
    for name, prs in (("sl", pos), ("neg", CORE_NEG)):
        txt = os.path.join(root, f"{name}.txt")
        with open(txt, "w") as f, open(txt[:-4] + ".sources.tsv", "w") as s:
            s.write("entrez1\tentrez2\tsources\tcell_lines\tpubmed_ids\n")
            for a, b in prs:
                f.write(f"{genes[a]}\t{genes[b]}\n")
                s.write(f"{genes[a]}\t{genes[b]}\tScreen\tK562\t1\n")

    folds_dir = os.path.join(root, "folds")
    d = os.path.join(folds_dir, "cv3", "fold_0")
    os.makedirs(d)
    with open(os.path.join(folds_dir, "genes.txt"), "w") as f:
        f.write("\n".join(genes) + "\n")
    split = {
        "train_pos": [(10, 11), (18, 19)],
        "train_neg": [(10, 12), (11, 13)],
        "val_pos": [(12, 13)],
        "val_neg": [(16, 17)],
        "test_pos": [(14, 15), (20, 21)],
        "test_neg": [],
    }
    for nm, prs in split.items():
        np.save(os.path.join(d, f"{nm}.npy"),
                np.array(prs, dtype=np.int32).reshape(-1, 2))
    return (emb_path, os.path.join(root, "sl.txt"),
            os.path.join(root, "neg.txt"), folds_dir, genes)


def _manager(emb, sl, neg, folds_dir):
    return SLDataManager(embeddings_paths=[emb], sl_pairs_path=sl,
                         neg_pairs_path=neg, use_cell_lines=True,
                         folds_dir=folds_dir, seed=0)


def test_fold_pair_outside_the_labelled_universe_raises():
    with tempfile.TemporaryDirectory() as root:
        emb, sl, neg, fd, _ = make_case(root, extra_in_files=False)
        dm = _manager(emb, sl, neg, fd)
        try:
            dm.get_cv3_splits(num_folds=1)
        except RuntimeError as e:
            assert "not in the labelled universe" in str(e), str(e)
        else:
            raise AssertionError(
                "a fold pair missing from --sl_path/--neg_pairs_path must "
                "raise, not be dropped silently")


def test_matching_files_keep_every_fold_pair_and_exclude_test_genes():
    with tempfile.TemporaryDirectory() as root:
        emb, sl, neg, fd, genes = make_case(root, extra_in_files=True)
        dm = _manager(emb, sl, neg, fd)
        fold = dm.get_cv3_splits(num_folds=1)[0]
        got = {tuple(sorted((dm.idx_to_gene[i], dm.idx_to_gene[j])))
               for part in ("train", "val", "test")
               for i, j in fold[part]["pairs"].tolist()}
        want = {tuple(sorted((genes[a], genes[b])))
                for a, b in CORE_POS + CORE_NEG + EXTRA_POS}
        assert got == want, sorted(got ^ want)
        # Regression guard for the leak the silent drop caused: no gene of a
        # test pair may survive in the preprocessing fit.
        mask = dm._fold_fit_mask(fold)
        for i, j in fold["test"]["pairs"].tolist():
            assert not bool(mask[i]) and not bool(mask[j]), (i, j)


def test_gene_without_an_embedding_row_is_counted_not_fatal():
    # Gene 18 has no embedding row, so the train pair (18, 19) is unusable —
    # a different miss from the one above, and one the run survives.
    with tempfile.TemporaryDirectory() as root:
        emb, sl, neg, fd, genes = make_case(root, extra_in_files=True,
                                            drop_embedding_for=18)
        dm = _manager(emb, sl, neg, fd)
        fold = dm.get_cv3_splits(num_folds=1)[0]
        kept = {tuple(sorted((dm.idx_to_gene[i], dm.idx_to_gene[j])))
                for i, j in fold["train"]["pairs"].tolist()}
        assert tuple(sorted((genes[18], genes[19]))) not in kept
        assert len(kept) == 3, sorted(kept)


def test_a_sidecar_that_lost_rows_is_refused():
    """A pair of --sl_path / --neg_pairs_path with no row in its
    .sources.tsv sidecar has no cell line: it used to be supervised at the
    OTHER head with one WARNING line. A row with an EMPTY cell_lines field is
    a different thing (no line recorded: OTHER by design) and still loads."""
    with tempfile.TemporaryDirectory() as root:
        emb, sl, neg, fd, _ = make_case(root, extra_in_files=True)
        side = neg[:-4] + ".sources.tsv"
        rows = open(side).read().splitlines(keepends=True)
        # an empty cell_lines field on one row: still a row
        cells = rows[1].split("\t")
        cells[3] = ""
        with open(side, "w") as f:
            f.writelines([rows[0], "\t".join(cells)] + rows[2:])
        _manager(emb, sl, neg, fd)
        # the last row lost (a copy that stopped part-way)
        with open(side, "w") as f:
            f.writelines(rows[:-1])
        try:
            _manager(emb, sl, neg, fd)
        except RuntimeError as e:
            assert "1 pair(s)" in str(e) and "out of sync" in str(e), str(e)
        else:
            raise AssertionError("a pair without a sidecar row must be "
                                 "refused, not trained at the OTHER head")


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
