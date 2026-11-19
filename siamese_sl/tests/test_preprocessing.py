#!/usr/bin/env python3
"""Regression tests for the siamese_sl preprocessing / leakage / label paths.

Exercises the REAL functions (needs torch + the repo deps), so run on a machine
with the project environment — e.g. on the cluster:

    $PYTHON_PATH -m pytest siamese_sl/tests/test_preprocessing.py -q
    # or, without pytest:
    $PYTHON_PATH siamese_sl/tests/test_preprocessing.py

Covers, in particular, the five audit fixes:
  1. plain-path PCA resolves the variance target on the standardized spectrum
  2. gene-order guards: predict.py refuses an embedding file whose order is not
     the checkpoint's stored gene_order (run through predict.main), and
     load_raw_multimodal refuses modalities in different orders
  3. load_single_embedding symbol/Entrez majority detection + dedup
  4. degenerate-fold handling (a constant column, a column all-NaN on the fit
     rows, fewer fit rows than dims: transforms stay finite)
  5. leakage-free fit: excluded (val-/test-only) genes never shape any fitted
     statistic, and _fold_fit_mask excludes exactly those genes
"""
import contextlib
import io
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

    # The post-concat branch must actually run, or its stats go unchecked.
    assert t1["post_pca_V"] is not None and t1["post_median"] is not None

    def _same(a, b, label):
        if a is None or b is None:
            assert a is None and b is None, f"{label}: fitted/skipped differs (leak)"
        elif torch.is_tensor(a):
            assert a.shape == b.shape and torch.allclose(a, b), \
                f"{label}: leaked from excluded rows"
        elif isinstance(a, (int, float)):
            assert abs(a - b) < 1e-6, f"{label}: leaked from excluded rows"
        else:
            assert a == b, f"{label}: differs"

    # EVERY fitted field, per modality (impute locations, pre-PCA column
    # standardization, V, normalize center/scale) and post-concat (the same
    # plus the re-normalization), so a new or moved statistic is covered too.
    for k in range(len(mods)):
        m1, m2 = t1["per_modality"][k], t2["per_modality"][k]
        assert m1.keys() == m2.keys()
        for key in m1:
            _same(m1[key], m2[key], f"modality {k} {key}")
    assert t1.keys() == t2.keys()
    for key in t1:
        if key != "per_modality":
            _same(t1[key], t2[key], f"post-concat {key}")
    # End to end: the same unperturbed input maps to the same features.
    assert torch.allclose(apply_multimodal_transform(mods, t1),
                          apply_multimodal_transform(mods, t2)), \
        "features differ although only excluded rows were perturbed"


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


def test_post_pca_count_at_or_above_the_width_is_refused():
    """An exact post-concat count (--post_pca_dim: PCA_JOINT_DIM of PCA arms
    joint and both) at or above the width it would reduce ran NO joint PCA
    (_fit_pca_V's no-op) while config.json still recorded it, so the run was
    labelled with a PCA it never had: refused per fold since 2026-10-21,
    naming the count and the width, k == width included. One below the width
    projects; a variance target is never refused, nor an int per-modality
    entry at or above its modality's dim (a documented no-op)."""
    g = torch.Generator().manual_seed(5)
    mods = [torch.randn(300, 6, generator=g) @ torch.randn(6, d, generator=g)
            + 0.1 * torch.randn(300, d, generator=g) for d in (20, 30)]
    fm = torch.ones(300, dtype=torch.bool)

    def fit(**kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_multimodal_transform(mods, fm, **kw)

    def refused(k, w, **kw):
        try:
            fit(post_pca=k, **kw)
        except ValueError as e:
            msg = str(e)
            assert f"post-concat PCA to {k} components" in msg \
                and f"has only {w} columns" in msg \
                and "PCA_JOINT_DIM" in msg, msg
            return True
        return False

    for method in ("plain", "robust", "svd"):
        # the width the joint stage meets: the per-modality components
        t = fit(pca_dims=[0.8, 0.8], post_pca=None, pca_method=method)
        w = sum(m["pca_V"].shape[1] for m in t["per_modality"])
        assert 2 < w < 50, (method, w)
        t = fit(pca_dims=[0.8, 0.8], post_pca=w - 1, pca_method=method)
        assert t["post_pca_V"] is not None and \
            tuple(t["post_pca_V"].shape) == (w, w - 1), method
        assert tuple(apply_multimodal_transform(mods, t).shape) == \
            (300, w - 1), method
        for k in (w, w + 5):
            assert refused(k, w, pca_dims=[0.8, 0.8], pca_method=method), \
                (method, k, w)
    # without a per-modality PCA it is the native concatenation's width
    assert refused(50, 50, pca_dims=None)
    t = fit(pca_dims=None, post_pca=49)
    assert tuple(t["post_pca_V"].shape) == (50, 49)
    # a variance target is never refused, even one keeping every component
    for post in (0.8, 0.999999):
        fit(pca_dims=[0.8, 0.8], post_pca=post)
        fit(pca_dims=None, post_pca=post)
    # nor an int per-modality entry at or above its modality's dim
    t = fit(pca_dims=[20, 30], post_pca=None)
    assert all(m["pca_V"] is None for m in t["per_modality"])


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


def test_load_single_embedding_majority_entrez_keeps_stray_non_digit():
    """A mostly-Entrez list with one non-digit id up front must take the
    Entrez branch (whole-list majority vote), not the symbol branch, which
    would drop every numeric id (the old first-10-rows rule did exactly
    that)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "emb.pt")
        order_in = ["NOT_A_GENE"] + [str(10 + i) for i in range(12)]
        emb_in = _write_pt(p, order_in, dim=3)
        emb, order = load_single_embedding(p)
        assert order == order_in, order
        assert torch.equal(emb, emb_in)


def test_load_raw_multimodal_rejects_gene_order_mismatch():
    """Differently-ordered modality files cannot be silently aligned
    (load_raw_multimodal raises)."""
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


def test_predict_refuses_checkpoint_gene_order_mismatch():
    """predict.py's own guard: an embedding file whose row order differs from
    the checkpoint's stored gene_order is not the file the model was trained
    on and must be refused. Run through predict.main on a tiny single-output
    checkpoint; the matching order is the control (same files, same genes),
    so a ValueError can only come from the order check."""
    import predict
    from siamese_esm import SiameseSL

    with tempfile.TemporaryDirectory() as d:
        genes = ["10", "11", "12", "13", "14", "15"]
        p = os.path.join(d, "emb.pt")
        _write_pt(p, genes, dim=5, seed=3)
        raw, _, idx_to_gene = load_raw_multimodal([p])
        t = fit_multimodal_transform(raw,
                                     torch.ones(len(genes), dtype=torch.bool),
                                     pca_dims=None,
                                     post_pca=None,
                                     pca_method="plain")
        model = SiameseSL(input_dim=5, encoder_dims=[4, 3])
        ckpt = os.path.join(d, "fold_0_best.pt")

        def _predict(gene_order):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "pd_epsilon": 0.001,
                        "dropout": 0.2
                    },
                    "preprocessing_transform": t,
                    "gene_order": gene_order,
                }, ckpt)
            argv = sys.argv
            sys.argv = [
                "predict.py", "--model", ckpt, "--embeddings_paths", p,
                "--gene1", "10", "--gene2", "11"
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    predict.main()
            finally:
                sys.argv = argv

        _predict([idx_to_gene[i] for i in range(len(idx_to_gene))])  # control
        raised = False
        try:
            _predict(list(reversed(genes)))
        except ValueError:
            raised = True
        assert raised, "a checkpoint/file gene-order mismatch must raise"


# ---------------------------------------------------------------------------
# Fix #4 — degenerate folds keep the transform finite
# ---------------------------------------------------------------------------
def test_degenerate_fold_transform_stays_finite():
    """A constant column (std 0), a column all-NaN on every fit row, and a
    fold with fewer fit rows than dims must all give a finite transform."""
    g = torch.Generator().manual_seed(4)
    n = 40
    a = torch.randn(n, 12, generator=g)
    a[:, 0] = 3.0  # constant column
    b = torch.randn(n, 8, generator=g)
    fm = torch.ones(n, dtype=torch.bool)
    fm[:10] = False
    b[10:, 1] = float("nan")  # observed only on excluded rows
    for fit_mask in (fm, torch.arange(n) < 5):  # the second: n_fit=5 < d
        t = fit_multimodal_transform([a.clone(), b.clone()],
                                     fit_mask,
                                     pca_dims=[0.8, 0.8],
                                     post_pca=0.8,
                                     pca_method="plain")
        out = apply_multimodal_transform([a.clone(), b.clone()], t)
        assert torch.isfinite(out).all(), \
            f"degenerate fold (n_fit={int(fit_mask.sum())}) gave non-finite output"


# ---------------------------------------------------------------------------
# Fix #5 (mask side) — _fold_fit_mask excludes exactly the val-/test-only genes
# ---------------------------------------------------------------------------
def test_fold_fit_mask_excludes_val_and_test_only_genes():
    with tempfile.TemporaryDirectory() as d:
        genes = ["10", "11", "12", "13", "14", "15"]
        p1 = os.path.join(d, "m1.pt")
        p2 = os.path.join(d, "m2.pt")
        _write_pt(p1, genes, dim=4, seed=1)
        _write_pt(p2, genes, dim=6, seed=2)
        slp = os.path.join(d, "sl.txt")
        with open(slp, "w") as f:
            f.write("10\t11\n11\t12\n12\t13\n13\t14\n14\t15\n"
                    )  # numeric = Entrez, no HGNC needed

        dm = SLDataManager(embeddings_paths=[p1, p2],
                           sl_pairs_path=slp,
                           seed=42,
                           preprocessing_fit_scope="train")

        # Hand-crafted fold, shaped like the shared folds (train/val/test):
        # train touches {0,1}, val {1,2} (gene 1 is also a train gene), test
        # {3,4}; gene 5 is in no pair.
        fold = {
            "fold": 0,
            "train": {
                "pairs": np.array([[0, 1]], dtype=np.int64)
            },
            "val": {
                "pairs": np.array([[1, 2]], dtype=np.int64)
            },
            "test": {
                "pairs": np.array([[3, 4]], dtype=np.int64)
            },
        }
        mask = dm._fold_fit_mask(fold)
        assert mask.dtype == torch.bool and mask.numel() == 6
        assert bool(mask[0]) and bool(mask[1]), \
            "train genes kept (incl. one shared with val)"
        assert bool(mask[5]), "genes in no pair stay in the fit"
        assert not bool(mask[2]), "val-only genes must be excluded"
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
