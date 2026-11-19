#!/usr/bin/env python3
"""retrieval_metrics' dense-matrix checks under the rerun's model options.

Covers check_matrix_matches_forward on the train.py defaults the rerun trains
(no LayerNorm, raw tau, per-head c_h) for every {pd, bilinear} x {per_head,
pooled} variant; that a real matrix/forward mismatch still fails it; and that
the float32 shift check tolerates merged ties (its order assertion guards a
monotone shift it computes itself, so the public API cannot trigger it).

    $PYTHON_PATH -m pytest siamese_sl/tests/test_retrieval_metrics.py -q
    $PYTHON_PATH siamese_sl/tests/test_retrieval_metrics.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

from retrieval_metrics import (build_any_score_matrix,
                               check_matrix_matches_forward, shift_rank_stats)
from siamese_esm import SiameseSLMultiCell

D = 9
DIMS = [8, 8, 8]


def _rerun_model(kernel, pooled, seed=0, **extra):
    """A SiameseSLMultiCell with the rerun's defaults, every term moved off
    its initial value so a bug in any of them would show."""
    torch.manual_seed(seed)
    shape = (dict(num_heads=1, cell_line_dim=0) if pooled else
             dict(num_heads=8, cell_line_dim=4))
    m = SiameseSLMultiCell(input_dim=D, encoder_dims=DIMS, dropout=0.2,
                           pd_epsilon=0.01, siamese_encoder_type="residual",
                           use_layernorm=False, raw_temperature=True,
                           per_head_last_layer_bias=True, kernel=kernel,
                           **shape, **extra)
    with torch.no_grad():
        m.scoring_bias.normal_(0, 1.0)
        m.tau.fill_(-0.4)
        m.encoder.input_bias.normal_(0, 0.1)
        m.head_proj_bias.normal_(0, 0.5)
        if m.kernel_A is not None:
            m.kernel_A.add_(0.3 * torch.randn_like(m.kernel_A))
        if m.gene_w is not None:
            m.gene_w.normal_(0, 0.3)
        if m.pair_fc2 is not None:
            m.pair_fc2.weight.normal_(0, 0.5)
    return m.eval()


def test_matrix_matches_forward_on_rerun_variants():
    x = torch.randn(60, D, generator=torch.Generator().manual_seed(1))
    for kernel in ("pd", "bilinear"):
        for pooled in (False, True):
            stats = check_matrix_matches_forward(
                _rerun_model(kernel, pooled), x, n_sub=48, verbose=False)
            assert stats["max_abs_err_determinism"] == 0.0


EXTRAS = (dict(gene_term=True), dict(pair_features=True),
          dict(gene_term=True, pair_features=True))


def test_matrix_matches_forward_with_extra_score_terms():
    """--gene_term / --pair_features: the dense matrix adds both terms exactly
    as forward does, per head."""
    x = torch.randn(60, D, generator=torch.Generator().manual_seed(1))
    for extra in EXTRAS:
        for kernel in ("pd", "bilinear"):
            for pooled in (False, True):
                stats = check_matrix_matches_forward(
                    _rerun_model(kernel, pooled, **extra), x, n_sub=48,
                    verbose=False)
                assert stats["max_abs_err_determinism"] == 0.0


def test_pair_term_sub_blocks_do_not_change_the_matrix():
    x = torch.randn(40, D, generator=torch.Generator().manual_seed(4))
    m = _rerun_model("bilinear", False, gene_term=True, pair_features=True)
    whole = build_any_score_matrix(m, x)
    split = build_any_score_matrix(m, x, row_block=17, pair_rows=5)
    assert torch.allclose(whole, split, atol=1e-5)


def test_extra_score_terms_start_at_zero_and_are_symmetric():
    x1 = torch.randn(20, D, generator=torch.Generator().manual_seed(2))
    x2 = torch.randn(20, D, generator=torch.Generator().manual_seed(3))
    kw = dict(input_dim=D, encoder_dims=DIMS, num_heads=8, cell_line_dim=4,
              use_layernorm=False, raw_temperature=True,
              per_head_last_layer_bias=True, kernel="bilinear")
    torch.manual_seed(0)
    base = SiameseSLMultiCell(**kw).eval()
    both = SiameseSLMultiCell(gene_term=True, pair_features=True, **kw).eval()
    # same encoder / scorer weights: at init both terms contribute exactly 0
    both.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.equal(base(x1, x2), both(x1, x2))
    moved = _rerun_model("bilinear", False, gene_term=True, pair_features=True)
    with torch.no_grad():
        assert torch.allclose(moved(x1, x2), moved(x2, x1), atol=1e-6)
        assert not torch.allclose(
            moved(x1, x2),
            _rerun_model("bilinear", False)(x1, x2), atol=1e-4)


def test_extra_score_terms_get_gradients_from_init():
    """Zero init must not be a dead start: gene_w and pair_fc2 get a gradient
    at step 0, pair_fc1 once pair_fc2 has moved."""
    torch.manual_seed(0)
    m = SiameseSLMultiCell(input_dim=D, encoder_dims=DIMS, num_heads=1,
                           cell_line_dim=0, gene_term=True,
                           pair_features=True)
    x1, x2 = torch.randn(16, D), torch.randn(16, D)
    m(x1, x2).sum().backward()
    assert m.gene_w.grad.abs().sum() > 0
    assert m.pair_fc2.weight.grad.abs().sum() > 0
    assert m.pair_fc1.weight.grad.abs().sum() == 0
    with torch.no_grad():
        m.pair_fc2.weight.normal_(0, 0.5)
    m.zero_grad()
    m(x1, x2).sum().backward()
    assert m.pair_fc1.weight.grad.abs().sum() > 0


def test_matrix_forward_mismatch_still_fails():
    # encode_head drifts from forward by 1e-2 in z: an O(1e-2) logit error,
    # far below a permutation's but far above rounding. It must be caught.
    class Drifted(SiameseSLMultiCell):
        def encode_head(self, x, head):
            z, h = super().encode_head(x, head)
            return z + 1e-2, h

    good = _rerun_model("pd", pooled=False)
    bad = Drifted(input_dim=D, encoder_dims=DIMS, dropout=0.2,
                  pd_epsilon=0.01, use_layernorm=False, raw_temperature=True,
                  per_head_last_layer_bias=True, num_heads=8,
                  cell_line_dim=4)
    bad.load_state_dict(good.state_dict())
    bad.eval()
    x = torch.randn(60, D, generator=torch.Generator().manual_seed(1))
    try:
        check_matrix_matches_forward(bad, x, n_sub=48, verbose=False)
    except AssertionError:
        return
    raise AssertionError("a 1e-2 matrix/forward drift must fail the check")


def test_shift_check_tolerates_merged_ties_only():
    # Distinct by one float32 ULP near 25.27; after - min + 1 with min = -1000
    # (spacing ~1.2e-4 near 1026) the two become one value.
    S = torch.tensor([[-1000.0, 25.26714515686035, 25.267147064208984, 3.0],
                      [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    assert S[0, 1] != S[0, 2]
    stats = shift_rank_stats(S)
    assert stats["shift_merged_ties"] == 1.0
    assert stats["shift_min"] == 1.0


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
