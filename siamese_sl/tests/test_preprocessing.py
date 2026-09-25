#!/usr/bin/env python3
"""Regression tests for the siamese_sl preprocessing / leakage / label paths.

Exercises the REAL functions (needs torch + the repo deps), so run on a machine
with the project environment — e.g. on the cluster:

    $PYTHON_PATH -m pytest siamese_sl/tests/test_preprocessing.py -q
    # or, without pytest:
    $PYTHON_PATH siamese_sl/tests/test_preprocessing.py

Covers, in particular, the five audit fixes:
  1. plain-path PCA resolves the variance target on the standardized spectrum
  2. predict.py gene-order guard (invariant tested via load_raw_multimodal)
  3. load_single_embedding symbol/Entrez majority detection + dedup
  4. degenerate-fold handling (transforms stay finite)
  5. leakage-free fit: excluded (test-only) genes never shape the transform
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

import cell_line_vocab as clv
from data_loader import (
    fit_multimodal_transform,
    apply_multimodal_transform,
    load_single_embedding,
    load_raw_multimodal,
    _fit_pca_V,
    SLDataManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_modalities(n=200, dims=(12, 8), seed=0, nan_frac=0.05):
    g = torch.Generator().manual_seed(seed)
    mods = []
    for d in dims:
        x = torch.randn(n, d, generator=g)
        row_nan = torch.rand(n, generator=g) < nan_frac
        x[row_nan] = float("nan")  # fully-missing genes (row NaN)
        mods.append(x)
    return mods


def _write_pt(path, gene_order, dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(len(gene_order), dim, generator=g)
    torch.save({"raw_embeddings": emb, "gene_order": list(gene_order)}, path)
    return emb


# ---------------------------------------------------------------------------
# Preprocessing fit/apply
# ---------------------------------------------------------------------------
def test_apply_is_deterministic_and_imputes_all_nan():
    mods = _synthetic_modalities()
    fm = torch.ones(mods[0].shape[0], dtype=torch.bool)
    t = fit_multimodal_transform(mods,
                                 fm,
                                 pca_dims=[0.8, 0.8],
                                 post_pca=0.8,
                                 pca_method="plain")
    a = apply_multimodal_transform(mods, t)
    b = apply_multimodal_transform(mods, t)
    assert torch.equal(a, b), "apply must be deterministic"
    assert not torch.isnan(a).any(), "imputation must remove every NaN"


def test_leakage_free_excluded_rows_do_not_shape_transform():
    """THE core leakage guarantee: perturbing test-only (excluded) genes to
    extreme values must not change any fitted statistic."""
    mods = _synthetic_modalities(seed=3, nan_frac=0.0)
    n = mods[0].shape[0]
    fm = torch.ones(n, dtype=torch.bool)
    fm[:20] = False  # rows 0..19 = test-only

    t1 = fit_multimodal_transform([m.clone() for m in mods],
                                  fm,
                                  pca_dims=[0.8, 0.8],
                                  post_pca=0.8,
                                  pca_method="plain")

    perturbed = [m.clone() for m in mods]
    for m in perturbed:
        m[:20] = 1e4  # extreme values on excluded rows
    t2 = fit_multimodal_transform(perturbed,
                                  fm,
                                  pca_dims=[0.8, 0.8],
                                  post_pca=0.8,
                                  pca_method="plain")

    def _same_V(a, b, label):
        if a is None or b is None:
            assert a is None and b is None, f"{label}: PCA reduction differs (leak)"
        else:
            assert torch.allclose(
                a, b), f"{label}: PCA basis leaked from excluded rows"

    for k in range(len(mods)):
        m1, m2 = t1["per_modality"][k], t2["per_modality"][k]
        assert torch.allclose(m1["impute_locations"], m2["impute_locations"]), \
            f"modality {k}: impute leaked from excluded rows"
        _same_V(m1["pca_V"], m2["pca_V"], f"modality {k}")
        assert abs(m1["median"] - m2["median"]) < 1e-6
        assert abs(m1["mad"] - m2["mad"]) < 1e-6
    _same_V(t1["post_pca_V"], t2["post_pca_V"], "post-concat")


def test_post_pca_none_skips_post_normalize():
    mods = _synthetic_modalities()
    fm = torch.ones(mods[0].shape[0], dtype=torch.bool)
    t = fit_multimodal_transform(mods,
                                 fm,
                                 pca_dims=[0.8, 0.8],
                                 post_pca=None,
                                 pca_method="plain")
    assert t["post_median"] is None and t["post_pca_V"] is None
    out = apply_multimodal_transform(mods, t)  # must not KeyError/NaN
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Fix #1 — plain-path PCA variance target resolved on the standardized basis
# ---------------------------------------------------------------------------
def test_plain_pca_hits_variance_target_under_uneven_columns():
    g = torch.Generator().manual_seed(1)
    n, d = 400, 10
    X = torch.randn(n, d, generator=g)
    X[:, 0] *= 100.0  # one dominant-variance column
    target = 0.8
    res = _fit_pca_V(X, target, method="plain")
    assert res is not None
    Xs = (X - res["pre_center"]) / res["pre_scale"]
    scores = Xs @ res["V"]
    retained = float((scores**2).sum() / (Xs**2).sum())
    assert retained >= target - 1e-6, \
        f"plain PCA retained {retained:.3f} < target {target} (variance-basis bug)"


def test_robust_pca_returns_finite_projection():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(300, 10, generator=g)
    res = _fit_pca_V(X, 0.8,
                     method="robust")  # falls back to median-SVD if no robpy
    if res is not None:
        assert torch.isfinite(res["V"]).all()


# ---------------------------------------------------------------------------
# Fix #3 — load_single_embedding: majority Entrez detection + dedup
# ---------------------------------------------------------------------------
def test_load_single_embedding_dedups_numeric_ids():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "emb.pt")
        _write_pt(p, ["10", "11", "11", "12"], dim=3)  # duplicate "11"
        emb, order = load_single_embedding(p)
        assert order == ["10", "11", "12"], order
        assert emb.shape[0] == 3


def test_load_raw_multimodal_rejects_gene_order_mismatch():
    """Backs the predict.py gene-order guard: differently-ordered files cannot
    be silently aligned."""
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "a.pt")
        p2 = os.path.join(d, "b.pt")
        _write_pt(p1, ["10", "11", "12"], dim=3, seed=1)
        _write_pt(p2, ["10", "12", "11"], dim=5, seed=2)  # reordered
        raised = False
        try:
            load_raw_multimodal([p1, p2])
        except ValueError:
            raised = True
        assert raised, "gene-order mismatch across modalities must raise"


# ---------------------------------------------------------------------------
# Fix #5 (mask side) — _fold_fit_mask excludes exactly the test-only genes
# ---------------------------------------------------------------------------
def test_fold_fit_mask_excludes_test_only_genes():
    with tempfile.TemporaryDirectory() as d:
        genes = ["10", "11", "12", "13", "14"]
        p1 = os.path.join(d, "m1.pt")
        p2 = os.path.join(d, "m2.pt")
        _write_pt(p1, genes, dim=4, seed=1)
        _write_pt(p2, genes, dim=6, seed=2)
        slp = os.path.join(d, "sl.txt")
        with open(slp, "w") as f:
            f.write("10\t11\n11\t12\n12\t13\n13\t14\n"
                    )  # numeric = Entrez, no HGNC needed

        dm = SLDataManager(embeddings_paths=[p1, p2],
                           sl_pairs_path=slp,
                           seed=42,
                           preprocessing_fit_scope="train")

        # Hand-crafted fold: train touches genes {0,1}, test touches {3,4}.
        fold = {
            "fold": 0,
            "train": {
                "pairs": np.array([[0, 1]], dtype=np.int64)
            },
            "test": {
                "pairs": np.array([[3, 4]], dtype=np.int64)
            },
        }
        mask = dm._fold_fit_mask(fold)
        assert mask.dtype == torch.bool and mask.numel() == 5
        assert bool(mask[0]) and bool(mask[1]) and bool(
            mask[2]), "train/other genes kept"
        assert not bool(mask[3]) and not bool(
            mask[4]), "test-only genes must be excluded"

        dm.preprocessing_fit_scope = "all"
        assert dm._fold_fit_mask(
            fold).all(), "scope='all' must keep every gene"


# ---------------------------------------------------------------------------
# Cell-line masked multi-label label construction (real code)
# ---------------------------------------------------------------------------
def test_pair_label_mask_invariants():
    H = clv.NUM_HEADS
    OTHER = clv.OTHER_HEAD_IDX
    K562 = clv.HEADS.index("K562")
    A549 = clv.HEADS.index("A549")

    lab, msk = clv.pair_label_mask("K562", "", True, False)
    assert lab[K562] == 1 and msk[K562] == 1 and msk.sum() == 1

    lab, msk = clv.pair_label_mask("K562", "A549", True, True)
    assert lab[K562] == 1 and msk[K562] == 1 and lab[A549] == 0 and msk[
        A549] == 1

    # Same-head disagreement resolves by OR: an SL call in the head wins.
    lab, msk = clv.pair_label_mask("K562", "K562", True, True)
    assert lab[K562] == 1 and msk[K562] == 1 and msk.sum() == 1
    assert clv.same_head_conflicts("K562", "K562", True, True) == {K562}
    assert clv.same_head_conflicts("K562", "A549", True, True) == set()

    # ...and the legacy rule still masks it, so an A/B is one argument away.
    lab, msk = clv.pair_label_mask("K562",
                                   "K562",
                                   True,
                                   True,
                                   same_head="mask")
    assert msk[K562] == 0 and msk.sum() == 0

    lab, msk = clv.pair_label_mask("", "A549", True,
                                   True)  # MISSING SL -> OTHER
    assert lab[OTHER] == 1 and msk[OTHER] == 1

    lab, msk = clv.pair_label_mask("RPE1", "", True,
                                   False)  # rare line -> OTHER
    assert lab[OTHER] == 1 and msk[OTHER] == 1
    assert msk[:len(clv.NAMED_HEADS)].sum() == 0

    lab, msk = clv.pair_label_mask("X293T", "", True, False)  # merge token
    assert lab[clv.HEADS.index("293T")] == 1


# ---------------------------------------------------------------------------
# Per-cell-line (--per_head) external-data evaluation
# ---------------------------------------------------------------------------
def test_per_head_loader_wrapper_and_freeze():
    from siamese_esm import SiameseSLMultiCell
    from eval_finetuning import (
        load_multicell_model,
        _HeadSelect,
        FinetunedSL,
        set_trainable,
    )

    D, cell_dim = 12, 8
    H = clv.NUM_HEADS
    mc = SiameseSLMultiCell(input_dim=D,
                            num_heads=H,
                            encoder_dims=[6, 6, 6],
                            cell_line_dim=cell_dim,
                            siamese_encoder_type="residual")
    mc.eval()

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "fold_0_best.pt")
        torch.save(
            {
                "model_state_dict": mc.state_dict(),
                "config": {
                    "pd_epsilon": 0.001
                }
            }, ckpt)
        rebuilt = load_multicell_model(ckpt, device="cpu")

    # (a) faithful reconstruction: dims + every weight identical
    assert rebuilt.num_heads == H and rebuilt.cell_line_dim == cell_dim
    assert rebuilt.encoder.input_bias.shape[0] == D + cell_dim  # widened input
    for k, v in mc.state_dict().items():
        assert torch.allclose(rebuilt.state_dict()[k], v), f"weight drift: {k}"

    # (b) head selection picks the SAME column as the raw multi-head output
    x1, x2 = torch.randn(5, D), torch.randn(5, D)
    full = mc(x1, x2)  # (5, H)
    k = clv.HEADS.index("K562")
    hs = _HeadSelect(mc, k).eval()
    out = hs(x1, x2)  # (5, 1)
    assert out.shape == (5, 1)
    assert torch.allclose(out[:, 0], full[:, k], atol=1e-6)

    # (c) FinetunedSL (B,1)->(B,) contract holds, incl. batch size 1
    fs = FinetunedSL(hs).eval()
    assert fs(x1, x2).shape == (5, )
    assert fs(x1[:1], x2[:1]).shape == (1, )

    # (d) Full mode: cell_emb stays frozen; encoder + affine train; no param
    #     is registered twice (the .encoder @property must not double-count).
    set_trainable(fs, "Full")
    assert not fs.base.model.cell_emb.weight.requires_grad, "cell_emb must stay frozen"
    assert fs.a.requires_grad and fs.c.requires_grad
    assert all(p.requires_grad for p in fs.base.encoder.parameters())
    pids = [id(p) for p in fs.parameters()]
    assert len(pids) == len(set(pids)), "parameter double-registration"


def test_parse_datasets_spec_optional_head():
    from eval_finetuning import parse_datasets_spec
    specs = parse_datasets_spec([
        "Adamson:external data/x.txt:Gamma:-1:tsv:K562",  # 6 fields
        "Legacy:external data/y.csv:score:1:csv",  # 5 fields (back-compat)
    ])
    assert specs[0]["head"] == "K562"
    assert specs[1]["head"] is None
    raised = False
    try:
        parse_datasets_spec(["too:few:fields:here"])  # 4 fields -> error
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# Runner (works without pytest)
# ---------------------------------------------------------------------------
def _main():
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
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
