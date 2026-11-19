#!/usr/bin/env python3
"""Dense all-pairs scoring for siamese_sl's OR-collapsed ("_any") view.

The other three benchmark models get ndcg@k / recall@k / precision@k / map@k
from sl_comparison/evaluate_model.score_fold, which hands a dense gene x gene
score matrix to cal_metrics (sl_comparison/preprocess_benchmarking_paper.py).
Its ranking half (from the score_matrix_ndcg copy on) argsorts each test
gene's WHOLE row over every gene in the universe. siamese only ever scored
candidate pairs, which is why those columns rendered as "-".

The multi-cell scorer (SiameseSLMultiCell.forward) FACTORIZES, so the whole
matrix is two matmuls per head:

    z_h, hid_h = model.encode_head(feats, h)
                 # = encoder.forward_with_hidden(cat([feats, cell_emb[h]]) + a_h)
                 #   + c_h; no cell_emb under --cell_lines pooled (H = 1),
                 #   a_h only with --per_head_input_bias, c_h unless
                 #   --no_per_head_last_layer_bias (train.py default: on)
    K_h        = z_h @ A_sym @ z_h.T + pd_epsilon * hid_h @ hid_h.T
                 # A_sym = (A + A.T)/2 for --kernel bilinear, I for pd
    K_h        = (K_h - mu_h) / sd_h   # --inner_norm batch only (running stats)
    L_h        = tau_h * K_h + bias[h] # tau = model.temperature(): exp(log_tau)
                                       # or the raw tau; one per head with
                                       # --per_head_temperature, else shared
    S          = max_h L_h             # == train.py's OR-collapse

Three details that are easy to get wrong and are pinned by the equivalence
test below:
  * the per-head bias is added AFTER the temperature scale (forward:
    temperature() * inner + scoring_bias);
  * scoring_bias is PER-HEAD, so it cannot be factored out of the max;
  * the max is over ALL num_heads (train.py _evaluate_multilabel uses
    L[agg_keep].max(axis=1), the max LOGIT, i.e. exactly this matrix's
    entries; the mask builds only the LABEL, never the score).

Run `python retrieval_metrics.py` on a GPU node to execute _self_test().
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


@torch.no_grad()
def _encode_head(model, feats: torch.Tensor, h: int, enc_block: int = 65536):
    """(Z_h, Hid_h) for every row of `feats` under head h.

    Mirrors SiameseSLMultiCell.forward exactly via model.encode_head: concat
    the head's cell embedding onto each gene (plus the per-head input offset
    and per-head projection bias when those options are on). The encoder
    stacks are LayerNorm (optional) + LeakyReLU + Dropout -- NO BatchNorm --
    so row-chunking is exact.
    """
    zs, hs = [], []
    for s in range(0, int(feats.shape[0]), enc_block):
        xb = feats[s:s + enc_block]
        z, hid = model.encode_head(xb, h)
        zs.append(z.float())
        hs.append(hid.float())
    return torch.cat(zs).contiguous(), torch.cat(hs).contiguous()


@torch.no_grad()
def build_any_score_matrix(model,
                           feats: torch.Tensor,
                           *,
                           only_head: Optional[int] = None,
                           row_block: int = 2048,
                           pair_rows: Optional[int] = None) -> torch.Tensor:
    """Dense (n, n) float32 max-over-heads LOGIT matrix on feats.device.

    Returns RAW logits. The affine shift that makes cal_metrics' injected
    0-diagonal safe is applied by the CALLER, after the forward-equivalence
    check has compared these entries against model(x1, x2).

    only_head : score one head instead of max-over-heads: that head's own
                logit matrix (export_pair_predictions.py writes one per cell
                line; check_matrix_matches_forward tests each head with it).
    pair_rows : rows per sub-block of the --pair_features term (default: sized
                to ~256 MB of pair features; test hook otherwise).
    """
    if model.training:
        raise RuntimeError(
            "build_any_score_matrix requires model.eval(). In train mode the "
            "encoder's Dropout (every hidden block) draws a fresh mask "
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
    # tau is (H,) under per_head_temperature, else a scalar: broadcast to (H,).
    tau = model.temperature().float().reshape(-1).expand(int(model.num_heads))
    bias = model.scoring_bias.reshape(-1).float()
    eps = float(model.pd_epsilon)
    # Per-head kernel standardisation at eval uses the running estimates.
    if getattr(model, "inner_norm", "none") == "batch":
        in_mu, in_sd = (t.float() for t in model.inner_stats())
        in_sd = in_sd.clamp_min(1e-6)
    else:
        in_mu = in_sd = None

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
        # kernel="bilinear": row side is Z @ A_sym, so cur = Z A_sym Zᵀ, which
        # is the symmetrised (S + Sᵀ)/2 of Z A Zᵀ.
        A_sym = (model.kernel_matrix()
                 if hasattr(model, "kernel_matrix") else None)
        ZL = [Z if A_sym is None else Z @ A_sym.float() for Z, _ in ZH]
        # Extra score terms, both added after tau like b_h (see
        # SiameseSLMultiCell): the per-gene propensity g(hid) broadcasts as
        # g_i + g_j; the pair-feature MLP does not factorise over the two
        # genes, so it is evaluated pair by pair in row sub-blocks sized to
        # keep its (rows, n, 2*latent) feature tensor near 256 MB.
        has_gene = getattr(model, "gene_w", None) is not None
        has_pair = getattr(model, "pair_fc1", None) is not None
        G = [model.gene_score(Hd).float() for _, Hd in ZH] if has_gene else None
        if has_pair:
            pb = (max(1, int(pair_rows)) if pair_rows else max(
                1, (1 << 26) // max(1, n * 2 * int(ZH[0][0].shape[1]))))
        S = torch.empty((n, n), dtype=torch.float32, device=feats.device)
        rb = max(1, min(int(row_block), n))
        for s in range(0, n, rb):
            e = min(s + rb, n)
            acc = None
            for k, h in enumerate(heads):
                Z, Hd = ZH[k]
                cur = ZL[k][s:e] @ Z.T  # z_i . z_j  (or z_iᵀ A_sym z_j)
                if eps:
                    cur.addmm_(Hd[s:e], Hd.T, alpha=eps)  # + eps hid_i . hid_j
                if in_mu is not None:
                    cur.sub_(in_mu[h]).div_(in_sd[h])
                cur.mul_(tau[h]).add_(bias[h])  # tau_h*inner + b_h
                if has_gene:
                    cur.add_(G[k][s:e, None]).add_(G[k][None, :])
                if has_pair:
                    for ps in range(s, e, pb):
                        pe = min(ps + pb, e)
                        cur[ps - s:pe - s].add_(model.pair_score(
                            Z[ps:pe, None, :], Z[None, :, :]).float())
                # Running max seeded INSIDE the block -- never carried across
                # block boundaries.
                acc = cur if acc is None else torch.maximum(acc, cur, out=acc)
            S[s:e].copy_(acc)
    finally:
        if prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return 0.5 * (S + S.T)  # kill fp32 GEMM asymmetry


@torch.no_grad()
def shift_rank_stats(S: torch.Tensor) -> Dict[str, float]:
    """Assert the caller's shift S - S.min() + 1 is rank-neutral on S (n, m).

    Checks that the shifted values are > 0 and non-decreasing along each row's
    own (stable) order. float32 rounding can MERGE adjacent distinct values
    into a tie -- an unstable argsort then orders the tie by index, so
    comparing argsorts fails on a correct shift -- but can never invert two
    entries. Merged ties are counted in `shift_merged_ties`, not failed.
    """
    Sh = S - S.min() + 1.0
    assert float(Sh.min()) > 0.0, "shifted scores must be > 0"
    S_sorted, order = S.sort(dim=1, stable=True)
    Sh_along = Sh.gather(1, order)
    assert bool((Sh_along[:, 1:] >= Sh_along[:, :-1]).all()), \
        "affine shift inverted two entries of a row"
    merged = ((Sh_along[:, 1:] == Sh_along[:, :-1])
              & (S_sorted[:, 1:] != S_sorted[:, :-1]))
    return {"shift_min": float(Sh.min()),
            "shift_merged_ties": float(merged.sum())}


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

        # 2. collapsed "_any" == max_h forward  (train.py
        #    _evaluate_multilabel's agg_s)
        S = build_any_score_matrix(model, F, row_block=ns)
        ref_any = ref.max(dim=2).values
        d_any = float((S - ref_any).abs().max())
        assert torch.allclose(S, ref_any, rtol=rtol, atol=atol), \
            f"_any collapse: max |matrix - max_h forward| = {d_any:g}"
        stats["max_abs_err_any_logit"] = d_any

        # 3. row_block invariance: block boundaries + running-max reset. A
        #    different block height changes the GEMM's rounding by an ULP or
        #    two of the matrix's scale (measured 3.05e-5 at |S| ~ 246 on CPU),
        #    so the bound is relative; a boundary / running-max bug is O(1)
        #    relative and still fails it.
        d_blk = float(
            (S - build_any_score_matrix(model, F, row_block=37)).abs().max())
        tol_blk = 1e-6 * max(1.0, float(S.abs().max()))
        assert d_blk <= tol_blk, (f"row_block changed the result by "
                                  f"{d_blk:g} (tol {tol_blk:g})")
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
        stats.update(shift_rank_stats(S))

        if verbose:
            print(f"[equivalence] n_sub={ns} heads={H} pairs={ns * ns:,}")
            for k, v in stats.items():
                print(f"  {k:<32} {v:.3e}")
        return stats
    finally:
        if prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _self_test() -> None:
    """No data files needed. Run on gpu3 -- never forces CPU.

    Covers the published architecture (the SiameseSLMultiCell defaults) and
    the rerun's train.py defaults (no LayerNorm, raw tau, per-head c_h) under
    every 2x2 variant {pd, bilinear} x {per_head, pooled}, for both encoders.
    """
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
    dims = [32, 32, 32, 32, 32, 32, 32]  # ENCODER_DIMS, slurm/config.conf
    feats = torch.randn(n, D, device=dev)

    # (label, constructor overrides). per_head uses CELL_LINE_DIM=24
    # (slurm/config.conf); pooled is one head with no cell-line embedding.
    per_head = dict(num_heads=8, cell_line_dim=24)
    pooled = dict(num_heads=1, cell_line_dim=0)
    rerun = dict(use_layernorm=False, raw_temperature=True,
                 per_head_last_layer_bias=True)
    configs = [("published pd:per_head", dict(per_head))]
    for kernel in ("pd", "bilinear"):
        for cl, shape in (("per_head", per_head), ("pooled", pooled)):
            configs.append((f"rerun {kernel}:{cl}",
                            dict(shape, kernel=kernel, **rerun)))
    # --gene_term / --pair_features, alone and together
    for cl, shape in (("per_head", per_head), ("pooled", pooled)):
        for extra in (dict(gene_term=True), dict(pair_features=True),
                      dict(gene_term=True, pair_features=True)):
            configs.append((f"rerun bilinear:{cl} + {'+'.join(extra)}",
                            dict(shape, kernel="bilinear", **rerun, **extra)))

    for enc in ("residual", "mlp"):
        for label, kw in configs:
            mc = SiameseSLMultiCell(input_dim=D,
                                    encoder_dims=dims,
                                    dropout=0.2,
                                    pd_epsilon=0.001,
                                    siamese_encoder_type=enc,
                                    **kw).to(dev)
            with torch.no_grad():
                # Fresh models have scoring_bias == 0, input_bias == 0,
                # c_h == 0 and A == I, so a bug in any of those terms would
                # be invisible; move each off its initial value.
                mc.scoring_bias.normal_(0, 1.0)
                if mc.raw_temperature:
                    mc.tau.fill_(-0.4)  # raw tau may be negative
                else:
                    mc.log_temperature.fill_(0.3)
                mc.encoder.input_bias.normal_(0, 0.1)
                if mc.head_proj_bias is not None:
                    mc.head_proj_bias.normal_(0, 0.5)
                if mc.kernel_A is not None:
                    mc.kernel_A.add_(0.3 * torch.randn_like(mc.kernel_A))
                # both extra terms are 0 at init: move them off it
                if mc.gene_w is not None:
                    mc.gene_w.normal_(0, 0.3)
                if mc.pair_fc2 is not None:
                    mc.pair_fc2.weight.normal_(0, 0.5)
            mc.eval()
            print(f"\n--- SiameseSLMultiCell / {enc} / {label}")
            check_matrix_matches_forward(mc, feats, n_sub=128)

    # Full-size build: the rerun universe (data/gene_universe.txt, 19,397
    # genes; ~1.4 GiB per matrix, ~4 GiB peak) on GPU, a smaller one on CPU.
    n_real = 19397 if dev.type == "cuda" else 5000
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
