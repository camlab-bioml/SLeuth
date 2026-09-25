#!/usr/bin/env python3
"""Dense all-pairs scoring for siamese_sl's OR-collapsed ("_any") view.

The other three benchmark models get ndcg@k / recall@k / precision@k / map@k
from sl_comparison/evaluate_model.py:91, which hands a dense gene x gene score
matrix to cal_metrics (sl_comparison/preprocess_benchmarking_paper.py:1371).
Its ranking half (:1399-1455) argsorts each test gene's WHOLE row over every
gene in the universe. siamese only ever scored candidate pairs, which is why
those columns rendered as "-".

The multi-cell scorer FACTORIZES (siamese_esm.py:610-616), so the whole matrix
is two matmuls per head:

    z_h, hid_h = encoder.forward_with_hidden(cat([feats, cell_emb[h]], dim=1))
    L_h        = tau * (z_h @ z_h.T + pd_epsilon * hid_h @ hid_h.T) + bias[h]
    S          = max_h L_h                # == train.py:469's OR-collapse

Three details that are easy to get wrong and are pinned by the equivalence
test below:
  * the per-head bias is added AFTER the temperature scale (siamese_esm.py:616);
  * scoring_bias is PER-HEAD (:585), so it cannot be factored out of the max;
  * the max is over ALL num_heads (train.py:469 uses P[agg_keep].max(axis=1);
    the mask builds only the LABEL, never the score).

Run `python retrieval_metrics.py` on a GPU node to execute _self_test().
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


@torch.no_grad()
def _encode_head(model, feats: torch.Tensor, h: int, enc_block: int = 65536):
    """(Z_h, Hid_h) for every row of `feats` under head h.

    Mirrors siamese_esm.py:603-608 exactly: concat the head's cell embedding
    onto each gene, then encoder.forward_with_hidden. The encoder stacks are
    LayerNorm + LeakyReLU + Dropout only -- NO BatchNorm (siamese_esm.py:262-267,
    :354-360) -- so row-chunking is exact.
    """
    cell = model.cell_emb.weight[h]
    zs, hs = [], []
    for s in range(0, int(feats.shape[0]), enc_block):
        xb = feats[s:s + enc_block]
        ce = cell.unsqueeze(0).expand(xb.shape[0], -1).to(xb.dtype)
        z, hid = model.encoder.forward_with_hidden(torch.cat([xb, ce], dim=1))
        zs.append(z.float())
        hs.append(hid.float())
    return torch.cat(zs).contiguous(), torch.cat(hs).contiguous()


@torch.no_grad()
def build_any_score_matrix(model,
                           feats: torch.Tensor,
                           *,
                           only_head: Optional[int] = None,
                           row_block: int = 2048) -> torch.Tensor:
    """Dense (n, n) float32 max-over-heads LOGIT matrix on feats.device.

    Returns RAW logits. The affine shift that makes cal_metrics' injected
    0-diagonal safe is applied by the CALLER, after the forward-equivalence
    check has compared these entries against model(x1, x2).

    only_head : score one head instead of max-over-heads (test hook only).
    """
    if model.training:
        raise RuntimeError(
            "build_any_score_matrix requires model.eval(). In train mode the "
            "encoder's Dropout (siamese_esm.py:265, :359) draws a fresh mask "
            "per call, so the z used for row i differs from the z used for "
            "column i -- the matrix is not even symmetric -- and inverted "
            "dropout rescales every layer by 1/(1-p). The result would look "
            "plausible and be meaningless.")
    mdev = next(model.parameters()).device
    if feats.device != mdev:
        raise RuntimeError(f"feats on {feats.device} but model on {mdev}")
    if not torch.isfinite(feats).all():
        raise RuntimeError("feats contain NaN/Inf")

    n = int(feats.shape[0])
    heads: List[int] = (list(range(int(model.num_heads)))
                        if only_head is None else [int(only_head)])
    tau = torch.exp(model.log_temperature).float()
    bias = model.scoring_bias.reshape(-1).float()
    eps = float(model.pd_epsilon)

    # TF32 truncates the matmul mantissa to 10 bits (~5e-4 relative), enough to
    # reorder near-ties inside an argsort-top-100 and to break the equivalence
    # check. Force full fp32 for these (tiny, k<=32) matmuls.
    try:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:  # CPU-only build / API rename
        prev_tf32 = None
    try:
        ZH = [_encode_head(model, feats, h) for h in heads]
        S = torch.empty((n, n), dtype=torch.float32, device=feats.device)
        rb = max(1, min(int(row_block), n))
        for s in range(0, n, rb):
            e = min(s + rb, n)
            acc = None
            for k, h in enumerate(heads):
                Z, Hd = ZH[k]
                cur = Z[s:e] @ Z.T  # z_i . z_j
                if eps:
                    cur.addmm_(Hd[s:e], Hd.T, alpha=eps)  # + eps hid_i . hid_j
                cur.mul_(tau).add_(bias[h])  # tau*inner + b_h
                # Running max seeded INSIDE the block -- never carried across
                # block boundaries.
                acc = cur if acc is None else torch.maximum(acc, cur, out=acc)
            S[s:e].copy_(acc)
    finally:
        if prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return 0.5 * (S + S.T)  # kill fp32 GEMM asymmetry


@torch.no_grad()
def check_matrix_matches_forward(model,
                                 feats: torch.Tensor,
                                 *,
                                 n_sub: int = 128,
                                 seed: int = 0,
                                 pair_batch: int = 8192,
                                 rtol: float = 1e-4,
                                 atol: float = 1e-5,
                                 verbose: bool = True) -> Dict[str, float]:
    """Assert the dense matrix reproduces model(x1, x2), entry by entry.

    build_any_score_matrix is index-space agnostic, so feeding it an n_sub-row
    slice returns exactly the corresponding submatrix. That lets us compare
    EVERY entry of an n_sub x n_sub block against a real forward over the full
    cartesian product -- cheap, and exhaustive PER HEAD. Testing only the
    collapsed max would hide a bug in a head that never wins.
    """
    if model.training:
        raise RuntimeError("run the check with model.eval()")
    # TF32 must be off around the WHOLE function: the encoder's nn.Linear
    # layers are matmuls too, so if it were on, the reference forward and the
    # matrix path would encode at different precision and the assert would
    # fire spuriously.
    try:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        prev_tf32 = None
    try:
        n = int(feats.shape[0])
        g = torch.Generator().manual_seed(seed)
        sub = torch.randperm(n, generator=g)[:min(n_sub, n)]
        F = feats[sub.to(feats.device)].contiguous()
        ns = int(F.shape[0])
        H = int(model.num_heads)

        ii = torch.arange(ns, device=F.device).repeat_interleave(ns)
        jj = torch.arange(ns, device=F.device).repeat(ns)
        parts = []
        for s in range(0, int(ii.numel()), pair_batch):
            sl = slice(s, s + pair_batch)
            parts.append(model(F[ii[sl]], F[jj[sl]]).float())
        ref = torch.cat(parts).reshape(ns, ns, -1)  # (ns, ns, H)
        assert int(ref.shape[2]) == H, (tuple(ref.shape), H)

        stats: Dict[str, float] = {}

        # 1. PER-HEAD logits -- not just the argmax.
        worst = 0.0
        for h in range(H):
            Lh = build_any_score_matrix(model, F, only_head=h, row_block=ns)
            d = float((Lh - ref[:, :, h]).abs().max())
            worst = max(worst, d)
            assert torch.allclose(Lh, ref[:, :, h], rtol=rtol, atol=atol), \
                f"head {h}: max |matrix - forward| = {d:g}"
        stats["max_abs_err_per_head_logit"] = worst

        # 2. collapsed "_any" == max_h forward  (train.py:469)
        S = build_any_score_matrix(model, F, row_block=ns)
        ref_any = ref.max(dim=2).values
        d_any = float((S - ref_any).abs().max())
        assert torch.allclose(S, ref_any, rtol=rtol, atol=atol), \
            f"_any collapse: max |matrix - max_h forward| = {d_any:g}"
        stats["max_abs_err_any_logit"] = d_any

        # 3. row_block invariance: block boundaries + running-max reset.
        d_blk = float(
            (S - build_any_score_matrix(model, F, row_block=37)).abs().max())
        assert d_blk < 1e-6, f"row_block changed the result by {d_blk:g}"
        stats["max_abs_err_row_block"] = d_blk

        # 4. determinism -> no submodule left in train() (build already raises
        #    on a top-level model.training, so this catches e.g. a stray
        #    model.encoder.train()).
        d_det = float(
            (S - build_any_score_matrix(model, F, row_block=ns)).abs().max())
        assert d_det == 0.0, (f"two builds differ by {d_det:g} -- dropout is "
                              f"still active in a submodule")
        stats["max_abs_err_determinism"] = d_det

        # 5. symmetry (the scorer is symmetric in (x1, x2) by construction)
        stats["max_abs_asymmetry"] = float((S - S.T).abs().max())

        # 6. the affine shift used by the caller is rank-neutral
        Sh = S - S.min() + 1.0
        assert float(Sh.min()) > 0.0
        assert torch.equal(S.argsort(dim=1), Sh.argsort(dim=1)), \
            "affine shift reordered a row -- it must be strictly increasing"
        stats["shift_min"] = float(Sh.min())

        if verbose:
            print(f"[equivalence] n_sub={ns} heads={H} pairs={ns * ns:,}")
            for k, v in stats.items():
                print(f"  {k:<32} {v:.3e}")
        return stats
    finally:
        if prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _self_test() -> None:
    """No data files needed. Run on gpu3 -- never forces CPU."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from siamese_esm import SiameseSLMultiCell, set_seed

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type != "cuda":
        print("WARNING: no CUDA -- the cluster runs on GPU; rerun on gpu3.")
    set_seed(0)
    D, n = 61, 400
    dims = [16, 16, 16, 16, 16, 16, 16]  # slurm/config.conf:118
    feats = torch.randn(n, D, device=dev)

    for enc in ("residual", "mlp"):
        mc = SiameseSLMultiCell(input_dim=D,
                                num_heads=8,
                                encoder_dims=dims,
                                dropout=0.2,
                                pd_epsilon=0.001,
                                siamese_encoder_type=enc,
                                cell_line_dim=24).to(dev)  # config.conf:94
        with torch.no_grad():
            # Fresh models have scoring_bias == 0 (siamese_esm.py:585) and
            # input_bias == 0, so a bug in either term would be invisible.
            mc.scoring_bias.normal_(0, 1.0)
            mc.log_temperature.fill_(0.3)
            mc.encoder.input_bias.normal_(0, 0.1)
        mc.eval()
        print(f"\n--- SiameseSLMultiCell / {enc}")
        check_matrix_matches_forward(mc, feats, n_sub=128)

    n_real = 5257  # data/all_genes_esm2.genes.txt
    S = build_any_score_matrix(mc, torch.randn(n_real, D, device=dev))
    print(f"\nfull matrix {tuple(S.shape)} {S.dtype} "
          f"{S.numel() * 4 / 2 ** 20:.0f} MiB "
          f"min={S.min():.4f} max={S.max():.4f}")
    assert S.shape == (n_real, n_real) and S.dtype == torch.float32
    if dev.type == "cuda":
        print(f"peak GPU alloc: "
              f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
    print("OK")


if __name__ == "__main__":
    _self_test()
