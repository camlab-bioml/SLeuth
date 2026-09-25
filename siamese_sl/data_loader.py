#!/usr/bin/env python3
"""
Data loading utilities for Siamese SL prediction.

Preprocessing pipeline (applied per modality, then concatenated):
  1. Load .pt file, normalize gene names (custom corrections + HGNC), deduplicate
  2. Impute NaN — per-dimension
  3. PCA (optional) — reduces per-modality dimension
  4. Normalize — global scalar (center + scale)
  5. Concatenate along feature axis
  6. Post-concat PCA (optional) — applied once to the concatenated tensor
  7. Re-normalize after post-PCA (global scalar) to restore ~unit scale

Each stage has two variants selected by `pca_method`:
  "plain"  (DEFAULT) — column-mean impute → per-column mean+std standardize →
             SVD → global mean/std normalize (classical PCA end-to-end)
  "robust" — Huber M-estimate impute → ROBPCA (Hubert et al. 2005; falls back
             to median-centered SVD) → global median/MAD normalize
All robust measures (Huber / median / MAD) stay in the robust path; all
classical measures (mean / std) stay in the plain path. "plain" is the default
as of 2026-08-10 — one classical pipeline for every run, rather than three
regimes whose selection varied by job.

Fit/apply split — every preprocessing statistic (impute locations, PCA
standardization + projection, normalization center/scale) is fit on a
subset of genes (training rows) and applied to the full gene matrix. This
prevents test-fold gene embeddings from leaking into the preprocessing
basis under CV2/CV3.

`load_multimodal_embeddings` is a legacy all-gene-fit wrapper preserved for
predict.py and notebooks. Training-time per-fold preprocessing lives in
SLDataManager.get_fold_embeddings.

Cross-validation uses SLMGAE benchmark's splitting code for consistency:
  - CV1: Edge-based split (random split of SL pairs)
  - CV2: Gene-based split (hold out genes, test pairs have ≥1 unseen gene)
  - CV3: Pair-based split (both genes in test pairs are unseen)
"""

import sys
import warnings
from pathlib import Path

# Add SLMGAE code directory to path for importing benchmark code
SLMGAE_CODE_DIR = Path(__file__).parent.parent / "SLMGAE-in-pytorch"
if str(SLMGAE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SLMGAE_CODE_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from gene_name_utils import get_mapper

# =============================================================================
# Huber M-estimator (helper — used for imputation fit only, never inside PCA)
# =============================================================================


def _huber_location(x: torch.Tensor,
                    c: float = 1.345,
                    max_iter: int = 50,
                    tol: float = 1e-6) -> torch.Tensor:
    """Huber M-estimator for location (robust mean), vectorized over columns.

    Solves: mu = argmin_mu  sum_i rho_H(x_i - mu)
    where rho_H is the Huber loss with tuning constant c.
    c = 1.345 gives 95% efficiency at the normal distribution.

    Args:
        x: (n, d) tensor with no NaN. Each column is estimated independently.
        c: Huber tuning constant. Larger = closer to mean; smaller = closer to median.

    Returns:
        (d,) tensor of robust location estimates.
    """
    mu = torch.median(x, dim=0).values  # (d,)

    for _ in range(max_iter):
        r = x - mu.unsqueeze(0)  # (n, d) residuals
        scale = 1.4826 * torch.median(torch.abs(r), dim=0).values  # (d,)
        scale = torch.clamp(scale, min=tol)
        threshold = c * scale.unsqueeze(0)  # (1, d)
        w = torch.clamp(threshold / (torch.abs(r) + tol), max=1.0)  # (n, d)
        mu_new = (w * x).sum(dim=0) / w.sum(dim=0)  # (d,)
        if (mu_new - mu).abs().max() < tol:
            return mu_new
        mu = mu_new

    return mu


# =============================================================================
# PCA internals — all return projection matrices so fit and apply are decoupled.
#   robust path: ROBPCA (Hubert et al. 2005) → median-centered-SVD fallback.
#   plain  path: per-column mean+std standardize → classical SVD.
# =============================================================================


def _remove_singular_dims(embeddings: torch.Tensor,
                          n_components: int) -> tuple:
    """Remove rank-deficient embedding dimensions before ROBPCA.

    If some feature dimensions are linearly dependent, ROBPCA's h-subset
    covariance is singular and step-4's rank check would silently reduce
    k. This uses torch.linalg.svd to find the effective rank of the feature
    space and projects to the full-rank subspace, so ROBPCA receives a
    well-conditioned input at the requested component count.

    Args:
        embeddings: (n_genes, dim) tensor.
        n_components: Desired number of PCA components.

    Returns:
        (reduced_embeddings, n_components, V_pre):
        - reduced_embeddings: (n_genes, rank) in full-rank feature subspace
        - n_components: unchanged (guaranteed < rank)
        - V_pre: (dim, rank) projection matrix, or None if already full rank

    Raises:
        ValueError: If effective rank <= n_components (ROBPCA can't reduce).
    """
    n_genes, dim = embeddings.shape

    # Center before SVD: covariance is computed from centered data, so the
    # rank of (X - mean) — not raw X — determines covariance singularity.
    center = embeddings.mean(dim=0)
    centered = embeddings - center

    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    tol = S[0] * max(n_genes, dim) * torch.finfo(embeddings.dtype).eps
    rank = int((S > tol).sum().item())

    if rank >= dim:
        return embeddings, n_components, None

    V_pre = Vh[:rank].T  # (dim, rank) — keep only full-rank directions
    reduced = embeddings @ V_pre  # (n_genes, rank)

    print(f"    Rank-deficient: {dim}d has effective rank {rank}, "
          f"projecting to {rank}d full-rank subspace")

    if rank <= n_components:
        raise ValueError(
            f"Effective rank ({rank}) <= requested components "
            f"({n_components}); cannot reduce further with ROBPCA")

    return reduced, n_components, V_pre


def _fit_robpca_V(embeddings: torch.Tensor,
                  n_components: int,
                  alpha: float = 0.75) -> torch.Tensor:
    """Fit ROBPCA on the given rows, return the projection matrix V.

    Caller must ensure `embeddings` contains only the fit subset (no test
    rows) and no NaN. The returned V has shape (orig_dim, k_actual).
    Project with `X @ V` — no centering, same as `_robpca` did inline.

    Runs ROBPCA with `final_MCD_step=False` (step-5 FastMCD is disabled
    because it crashes on near-rank-limit score matrices).
    Reference: Hubert, Rousseeuw & Vanden Branden (2005); robpy (Leyder 2024).
    """
    from robpy.pca import ROBPCA

    orig_dim = embeddings.shape[1]

    reduced_emb, n_components, V_pre = _remove_singular_dims(
        embeddings, n_components)

    X_np = reduced_emb.numpy().astype(np.float64)

    pca = ROBPCA(n_components=n_components,
                 alpha=alpha,
                 final_MCD_step=False,
                 random_seed=42)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*_validate_data.*",
            category=FutureWarning,
        )
        pca.fit(X_np)

    V_rob = torch.from_numpy(pca.components_).float()
    k_actual = V_rob.shape[1]
    if k_actual == 0:
        raise ValueError("ROBPCA returned 0 components")
    if torch.isnan(V_rob).any():
        raise ValueError("ROBPCA produced NaN components (near-singular "
                         "covariance)")

    V_final = V_pre @ V_rob if V_pre is not None else V_rob  # (orig_dim, k)

    # Sanity: projecting the fit data must stay finite.
    projected = embeddings @ V_final
    if torch.isnan(projected).any():
        raise ValueError("ROBPCA projection contains NaN")

    # Variance explained against the fit data.
    # Numerator and denominator must come from the SAME matrix and the same
    # estimator. pca.explained_variance_ is ROBPCA's robust h-subset variance
    # of the SCORES; dividing it by a classical ddof=1 variance of the raw
    # embeddings compares two different quantities on two different matrices,
    # so the printed percentage could exceed 100 or understate badly. Use the
    # projection already computed above, centred the same way on both sides.
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    scores = projected - projected.mean(dim=0, keepdim=True)
    total_var = float((centered**2).sum())
    retained_var = float((scores**2).sum())
    pct = (retained_var / total_var * 100) if total_var > 0 else 0.0

    print(f"    ROBPCA {orig_dim}d -> {k_actual}d "
          f"({pct:.1f}% variance, alpha={alpha})")

    return V_final


def _fit_pca_fallback_V(embeddings: torch.Tensor,
                        n_components: int) -> torch.Tensor:
    """Plain PCA fallback: return the (orig_dim, n_components) projection V.

    Median-centered SVD (still a robust centering — used only inside the
    robust path when ROBPCA is unavailable or all alpha retries raise).
    Project with `X @ V` (location-preserving; matches _fit_robpca_V's
    convention).

    Uses deterministic torch.linalg.svd so per-fold projections are
    reproducible across runs.
    """
    orig_dim = embeddings.shape[1]
    center = torch.median(embeddings, dim=0).values
    centered = embeddings - center.unsqueeze(0)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    V = Vh[:n_components].T.contiguous()

    # Variance explained against the fit data.
    total_ss = (centered**2).sum()
    projected_scores = centered @ V
    retained_ss = (projected_scores**2).sum()
    pct = (retained_ss / total_ss * 100).item() if total_ss > 0 else 0.0

    print(f"    PCA (fallback) {orig_dim}d -> {n_components}d "
          f"({pct:.1f}% variance)")

    return V


def _resolve_n_components(embeddings: torch.Tensor,
                          target,
                          basis="raw") -> int:
    """Resolve PCA target to an integer number of components (on fit rows).

    Args:
        embeddings: (n_fit, dim) tensor with no NaN.
        target: int (exact component count) or float in (0, 1) for
            variance fraction.
        basis: which spectrum this is being resolved on, for the log line only.
            "correlation" under pca_method=plain (columns standardized first),
            "raw" under robust/svd.

    THE BASIS MATTERS AND IS NOT COSMETIC. A correlation spectrum is far flatter
    than the covariance spectrum of the same matrix, because standardizing gives
    every dimension equal variance. The SAME 0.8 target therefore retains far
    more components under `plain` than under `robust`/`svd` — measured on the
    6-modality combo (4,424d concat): plain k=1074, svd/robust k=144, a 7.5x
    wider encoder input. Neither is wrong; they answer the same question about
    different matrices. But the encoder width, the hand-tuned proximal-L1
    schedule and the reported parameter count all follow from k, so the basis is
    printed next to it rather than left for someone to infer from a jump in the
    numbers.
    """
    if isinstance(target, int):
        return target

    centered = embeddings - embeddings.mean(dim=0)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var_explained = (S**2).numpy()
    cumulative = np.cumsum(var_explained)
    total = cumulative[-1]
    if total == 0:
        return embeddings.shape[1]

    ratio = cumulative / total
    k = int(np.searchsorted(ratio, target) + 1)
    k = min(k, embeddings.shape[1])
    k = max(k, 1)

    print(f"    Variance target {target:.0%} on the {basis} spectrum: "
          f"{k} components (explains {ratio[k-1]:.1%} of "
          f"{embeddings.shape[1]}d)")
    return k


def _try_fit_robpca_V(embeddings: torch.Tensor,
                      n_components: int,
                      alpha: float = 0.75) -> Optional[torch.Tensor]:
    """Attempt ROBPCA with alpha retries. Returns V or None on total failure."""
    _robpca_errors = (ValueError, np.linalg.LinAlgError)
    try:
        return _fit_robpca_V(embeddings, n_components, alpha=alpha)
    except ImportError:
        return None
    except _robpca_errors as e:
        print(f"    [ROBPCA failed with alpha={alpha}: {e}]")

    retry_alphas = [a for a in [0.85, 0.90, 0.95] if a > alpha]
    for retry_alpha in retry_alphas:
        try:
            print(f"    Retrying ROBPCA with alpha={retry_alpha}...")
            return _fit_robpca_V(embeddings, n_components, alpha=retry_alpha)
        except _robpca_errors as e:
            print(f"    [ROBPCA failed with alpha={retry_alpha}: {e}]")

    return None


def _fit_plain_pca_V(embeddings_standardized: torch.Tensor,
                     n_components: int) -> torch.Tensor:
    """Classical PCA: SVD on already-standardized fit rows. Returns V.

    Caller is responsible for per-column mean/std standardization BEFORE
    calling this. That split lets the caller store (mean, std) alongside
    V so the same standardization is reapplied at inference time.

    Uses torch.linalg.svd (deterministic, no RNG use) rather than
    torch.pca_lowrank (randomized init) so per-fold results are bitwise
    reproducible across runs.
    """
    orig_dim = embeddings_standardized.shape[1]
    # Full (thin) SVD — X = U diag(S) Vh, where Vh is (min(n,d), d).
    # V's rows are the principal directions (sklearn convention); transpose
    # so columns match our `X @ V` projection convention.
    _, _, Vh = torch.linalg.svd(embeddings_standardized, full_matrices=False)
    V = Vh[:n_components].T.contiguous()  # (orig_dim, n_components)

    total_ss = (embeddings_standardized**2).sum()
    retained_scores = embeddings_standardized @ V
    retained_ss = (retained_scores**2).sum()
    pct = (retained_ss / total_ss * 100).item() if total_ss > 0 else 0.0
    print(f"    Plain PCA {orig_dim}d -> {n_components}d "
          f"({pct:.1f}% variance, standardized)")
    return V


def _fit_pca_V(emb_fit: torch.Tensor,
               target,
               alpha: float = 0.75,
               method: str = "plain") -> Optional[Dict]:
    """Fit PCA on emb_fit (fit rows only). Returns a dict describing the
    projection (or None when the target is a no-op, target >= orig_dim).

    Return schema:
        robust:  {"V": Tensor (orig_dim, k),
                  "pre_center": None, "pre_scale": None}
        plain:   {"V": Tensor (orig_dim, k),
                  "pre_center": Tensor (orig_dim,) — per-column mean,
                  "pre_scale":  Tensor (orig_dim,) — per-column std}

    Args:
        emb_fit: (n_fit, orig_dim) tensor — fit rows, no NaN.
        target: int (exact component count) or float in (0, 1) for variance
            fraction.
        alpha: ROBPCA coverage parameter (ignored when method="plain").
        method: "plain" (default) — classical PCA: per-column mean centering
            + per-column std scaling + SVD. No robust measures.
            "robust" — ROBPCA with alpha retries, plain-PCA fallback
            (median-centered SVD) on failure. Keeps all robust machinery
            even in the fallback.
    """
    if method not in ("robust", "plain", "svd"):
        raise ValueError(
            f"pca_method must be 'robust', 'plain' or 'svd'; got {method!r}")

    orig_dim = emb_fit.shape[1]

    if method == "svd":
        # Option B (matrix-wise): median-centered SVD with NO per-column
        # standardization, so high-variance / more-informative dims keep their
        # weight (unlike 'plain'), and NO ROBPCA, so it stays fast at high
        # concat dims (unlike 'robust'). The matrix-wise (global scalar)
        # normalization is handled separately by _fit_normalize_scalars.
        n_components = _resolve_n_components(emb_fit, target)
        if n_components >= orig_dim:
            return None
        V = _fit_pca_fallback_V(emb_fit, n_components)
        return {"V": V, "pre_center": None, "pre_scale": None}

    if method == "plain":
        # Classical PCA runs on the per-column standardized (correlation)
        # basis, so the variance-target k must be resolved on that SAME
        # standardized spectrum. Resolving it on the raw covariance spectrum
        # (as the robust path does) would retain a different fraction than
        # requested whenever column variances are uneven.
        pre_center, pre_scale = _fit_column_standardize(emb_fit)
        standardized = _apply_column_standardize(emb_fit, pre_center,
                                                 pre_scale)
        n_components = _resolve_n_components(standardized,
                                             target,
                                             basis="correlation")
        if n_components >= orig_dim:
            return None
        V = _fit_plain_pca_V(standardized, n_components)
        return {"V": V, "pre_center": pre_center, "pre_scale": pre_scale}

    # Robust path: resolve k on the (mean-centered) fit spectrum — ROBPCA does
    # not standardize columns — then ROBPCA first, median-centered SVD fallback.
    n_components = _resolve_n_components(emb_fit, target)
    if n_components >= orig_dim:
        return None
    V = _try_fit_robpca_V(emb_fit, n_components, alpha=alpha)
    if V is not None:
        return {"V": V, "pre_center": None, "pre_scale": None}

    warnings.warn(
        "ROBPCA unavailable or all retries failed. "
        "Falling back to median-centered SVD (still robust centering). "
        "Robust OUTLIER filtering is NOT active for this modality.",
        stacklevel=2,
    )
    V = _fit_pca_fallback_V(emb_fit, n_components)
    return {"V": V, "pre_center": None, "pre_scale": None}


# =============================================================================
# Imputation / PCA / normalize — fit on a subset, apply to the full matrix.
# =============================================================================


def _fit_huber_locations(embeddings: torch.Tensor,
                         fit_mask: torch.Tensor,
                         c: float = 1.345) -> torch.Tensor:
    """Column-wise Huber location fit on the fit subset.

    Args:
        embeddings: (n_genes, dim) tensor, may contain NaN.
        fit_mask: (n_genes,) bool — rows to use for the fit.
        c: Huber tuning constant (1.345 ≈ 95% normal efficiency).

    Returns:
        (dim,) tensor of per-column locations. Columns with no valid fit
        rows are filled with 0 (same policy as the legacy all-row fit).
    """
    fit_rows = embeddings[fit_mask]
    dim = embeddings.shape[1]
    col_locations = torch.zeros(dim)

    if fit_rows.shape[0] == 0:
        return col_locations

    nan_mask_fit = torch.isnan(fit_rows)
    row_all_nan = nan_mask_fit.all(dim=1)
    row_all_valid = ~nan_mask_fit.any(dim=1)

    if row_all_nan.sum() + row_all_valid.sum() == fit_rows.shape[0]:
        # Clean case: row-level NaN (entire-row missing genes only).
        valid_rows = fit_rows[row_all_valid]
        if valid_rows.shape[0] > 0:
            col_locations = _huber_location(valid_rows, c=c)
        return col_locations

    # Mixed case: per-column Huber from valid entries in the fit subset.
    for d in range(dim):
        col = fit_rows[:, d]
        valid = col[~torch.isnan(col)]
        if len(valid) > 0:
            col_locations[d] = _huber_location(valid.unsqueeze(1),
                                               c=c).squeeze()
        else:
            col_locations[d] = 0.0
    return col_locations


def _fit_mean_locations(embeddings: torch.Tensor,
                        fit_mask: torch.Tensor) -> torch.Tensor:
    """Column-wise MEAN over non-NaN values on the fit subset (plain-PCA path).

    Classical counterpart to `_fit_huber_locations`. Columns with no valid
    fit rows are filled with 0 (same policy).
    """
    fit_rows = embeddings[fit_mask]
    dim = embeddings.shape[1]
    col_means = torch.zeros(dim)
    if fit_rows.shape[0] == 0:
        return col_means
    for d in range(dim):
        col = fit_rows[:, d]
        valid = col[~torch.isnan(col)]
        if len(valid) > 0:
            col_means[d] = valid.mean()
    return col_means


def _apply_huber_imputation(embeddings: torch.Tensor,
                            col_locations: torch.Tensor) -> torch.Tensor:
    """Impute NaN cells with fitted per-column locations.

    Name kept for historical reasons; the function is method-agnostic —
    col_locations can come from the Huber M-estimator (robust path) or the
    column mean (plain path). It just fills NaN cells.
    """
    nan_mask = torch.isnan(embeddings)
    if not nan_mask.any():
        return embeddings
    return torch.where(nan_mask, col_locations.unsqueeze(0), embeddings)


def _fit_column_standardize(
        embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-column mean + (unbiased) std on the fit matrix (no NaN).

    Returns (mean, std), both shape (dim,). std is clamped to 1e-8 to
    avoid divide-by-zero on constant columns. Used by the plain-PCA path
    to z-score each embedding dimension before SVD — the "standardize"
    step of classical PCA that the robust path skips.
    """
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0, unbiased=True)
    # Constant-only guard, NOT an absolute floor. `clamp(std, min=1e-8)` cannot
    # tell a constant column from one whose real std is below 1e-8, and mashup's
    # diffusion tail has 43 genuinely non-constant columns in that range. Under
    # an absolute floor those columns come out with variance 0.44-0.74 instead
    # of 1, so the z-score stops being scale-invariant: multiplying the same
    # matrix by 1000 moved the retained k from 330 to 346 at the same 0.8
    # variance target. k is the encoder input width, so the model's size would
    # depend on the units a modality happened to be stored in.
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def _apply_column_standardize(embeddings: torch.Tensor, mean: torch.Tensor,
                              std: torch.Tensor) -> torch.Tensor:
    """Per-column z-score using pre-fit mean/std."""
    return (embeddings - mean.unsqueeze(0)) / std.unsqueeze(0)


def _fit_normalize_scalars(
    embeddings: torch.Tensor,
    method: str = "plain",
) -> Tuple[float, float]:
    """Global scalar normalization stats on the fit matrix (no NaN).

    Returns (center, scale):
      method="plain" (default) — global mean, global (unbiased) std.
      method="robust"          — global median, MAD (× 1.4826 consistency).
    Scale is clamped to 1e-8 to avoid divide-by-zero on constant matrices.
    """
    if method == "plain":
        center = float(embeddings.mean().item())
        scale = float(embeddings.std(unbiased=True).item())
    else:
        center = float(torch.median(embeddings).item())
        centered = embeddings - center
        scale = 1.4826 * float(torch.median(torch.abs(centered)).item())
        # DEGENERATE-MAD GUARD.
        #
        # MAD measures the middle of the distribution. When more than half the
        # entries sit at (near) the center, it collapses toward 0 while the
        # actual signal lives in the tail — MAD is then not robust, it is
        # meaningless. data/all_genes_mashup.pt is exactly this: 50.9% of its
        # values are below 1e-7, giving MAD 9.9e-08 against std 5.0e-03, a
        # MAD/std ratio of 2e-05. Dividing by that MAD multiplies the modality
        # by ~1e7, so after concatenation it carries ~5e4x the variance of its
        # co-modalities and the post-concat SVD basis becomes ~100% mashup —
        # every other modality is annihilated in float32 and the "multi-modal"
        # model is arithmetically single-modality, with nothing in the log
        # saying so.
        #
        # The pre-existing `scale < 1e-8` clamp does not catch this: 9.9e-08
        # slips past it. Trigger on the RATIO instead, which is scale-free.
        # Measured over all 17 catalog embeddings the next-lowest ratio is
        # 0.145 (ppi_svd), so 1e-3 separates the degenerate case by four
        # orders of magnitude and cannot misfire on the others.
        sd = float(embeddings.std(unbiased=True).item())
        if sd > 0 and scale < 1e-3 * sd:
            print(f"    WARNING: degenerate MAD {scale:.3e} vs std {sd:.3e} "
                  f"(ratio {scale / sd:.1e}) — over half this modality's mass "
                  f"sits at the center. Falling back to std so it cannot "
                  f"dominate the concatenation.")
            scale = sd
    if scale < 1e-8:
        scale = 1e-8
    return center, scale


def _apply_normalize(embeddings: torch.Tensor, median: float,
                     mad: float) -> torch.Tensor:
    """Apply a (center, scale) scalar transform. The keyword names are kept
    as `median`/`mad` for backward-compat of the transform-dict schema; they
    carry the fitted center/scale regardless of whether those came from the
    robust (median/MAD) or plain (mean/std) path."""
    return (embeddings - median) / mad


# =============================================================================
# Legacy fit-and-apply convenience wrappers (all-gene fit).
# Kept so external callers (predict.py, notebooks) continue to work. All four
# now delegate to the new primitives so there is one set of fit/apply logic.
# =============================================================================


def impute_nan_with_huber(embeddings: torch.Tensor,
                          c: float = 1.345) -> torch.Tensor:
    """Fit + apply Huber imputation on ALL genes. Legacy convenience."""
    if not torch.isnan(embeddings).any():
        return embeddings

    fit_mask = torch.ones(embeddings.shape[0], dtype=torch.bool)
    col_locations = _fit_huber_locations(embeddings, fit_mask, c=c)

    valid_per_col = (~torch.isnan(embeddings)).sum(dim=0)
    if (valid_per_col == 0).any():
        all_nan_cols = valid_per_col == 0
        print(f"  Warning: {all_nan_cols.sum().item()} dimensions have no "
              f"valid values, filling with 0")

    imputed = _apply_huber_imputation(embeddings, col_locations)
    nan_mask = torch.isnan(embeddings)
    n_imputed = nan_mask.any(dim=1).sum().item()
    n_total = embeddings.shape[0]
    print(f"  Huber-imputed {n_imputed}/{n_total} genes "
          f"({nan_mask.sum().item()} NaN values, c={c})")
    return imputed


def apply_pca(embeddings: torch.Tensor,
              n_components,
              alpha: float = 0.75,
              method: str = "plain") -> torch.Tensor:
    """Fit + apply PCA on ALL genes. Legacy convenience.

    method="plain" (default): classical PCA — per-column mean + std
      standardization, then SVD. No robust measures.
    method="robust": ROBPCA first (Hubert et al. 2005); median-centered-SVD
      fallback when robpy is unavailable or all alpha retries raise.
    """
    pca_t = _fit_pca_V(embeddings, n_components, alpha=alpha, method=method)
    if pca_t is None:
        return embeddings
    emb = embeddings
    if pca_t["pre_center"] is not None:
        emb = _apply_column_standardize(emb, pca_t["pre_center"],
                                        pca_t["pre_scale"])
    return emb @ pca_t["V"]


def normalize_modality(embeddings: torch.Tensor,
                       method: str = "plain") -> torch.Tensor:
    """Fit + apply global scalar normalization on ALL rows. Legacy convenience.

    method="plain" (default): global mean  / std.
    method="robust":          global median / MAD.
    """
    center, scale = _fit_normalize_scalars(embeddings, method=method)
    return _apply_normalize(embeddings, center, scale)


# =============================================================================
# Single-file loader (gene-name normalization — no preprocessing).
# =============================================================================


def load_single_embedding(
    path: str,
    gene_list_path: Optional[str] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Load a single .pt embedding file and return (embeddings, gene_order).

    Gene identifiers in gene_order are NCBI Entrez Gene ID strings.
    If the file still contains gene symbols (e.g., freshly generated from
    an external source), they are normalized and converted to Entrez IDs.

    Deduplicates any collisions after conversion. Drops genes that cannot
    be mapped to an Entrez ID.

    Returns:
        (embeddings, gene_order) where gene_order is a list of Entrez ID
        strings. Embeddings may contain NaN for missing genes.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        embeddings = (data["raw_embeddings"]
                      if "raw_embeddings" in data else data["embeddings"])
        if "gene_order" in data:
            gene_order = list(data["gene_order"])
        elif "gene_to_idx" in data:
            gene_to_idx = data["gene_to_idx"]
            n = len(gene_to_idx)
            gene_order = [""] * n
            for gene, idx in gene_to_idx.items():
                if not (0 <= idx < n):
                    raise ValueError(
                        f"gene_to_idx has out-of-range index {idx} for "
                        f"{gene} (expected 0..{n-1})")
                gene_order[idx] = gene
            if "" in gene_order:
                raise ValueError(
                    "gene_to_idx has non-contiguous indices (gaps detected)")
        elif gene_list_path:
            with open(gene_list_path, "r") as f:
                gene_order = [line.strip() for line in f]
        else:
            raise ValueError(
                f"No gene order in {path} and no gene_list_path provided")
    else:
        embeddings = data
        if gene_list_path:
            with open(gene_list_path, "r") as f:
                gene_order = [line.strip() for line in f]
        else:
            raise ValueError("Raw tensor embeddings require gene_list_path")

    # Check if gene_order is already Entrez IDs (all-digit strings) or still
    # gene symbols (needs conversion). Decide from a majority vote over the
    # WHOLE list rather than the first 10 rows (which can be unrepresentative
    # if a file is sorted or has a non-digit header-like entry up front). A
    # mostly-digit list routes to the Entrez dedupe branch, which keeps any
    # stray non-digit id as-is (non-destructive); the symbol branch would drop
    # numeric ids, so majority-Entrez must never land there.
    mapper = get_mapper()
    n_digit = sum(1 for g in gene_order if g.isdigit())
    looks_like_entrez = bool(gene_order) and n_digit >= 0.5 * len(gene_order)
    if gene_order and not looks_like_entrez:
        new_order = []
        keep = []
        seen = set()
        dropped = []
        for i, sym in enumerate(gene_order):
            eid = mapper.symbol_to_entrez(sym)
            if eid is None:
                dropped.append(sym)
                continue
            if eid in seen:
                continue
            seen.add(eid)
            keep.append(i)
            new_order.append(eid)

        if dropped:
            print(f"  Warning: {len(dropped)} genes unmapped to Entrez ID "
                  f"(dropped): {dropped[:10]}")
        if len(keep) < len(gene_order):
            embeddings = embeddings[keep]
        gene_order = new_order
    else:
        seen = {}
        dups = []
        keep = []
        for i, gid in enumerate(gene_order):
            if gid in seen:
                dups.append((gid, seen[gid], i))
            else:
                seen[gid] = i
                keep.append(i)
        if dups:
            examples = "; ".join(f"{g} at rows {a},{b}"
                                 for g, a, b in dups[:5])
            print(f"  Warning: {len(dups)} duplicate Entrez ID(s), "
                  f"keeping first: {examples}")
            embeddings = embeddings[keep]
            gene_order = [gene_order[i] for i in keep]

    return embeddings, gene_order


# =============================================================================
# Multi-modal: raw loader + fit/apply transform.
# =============================================================================


def load_raw_multimodal(
    paths: List[str],
    gene_list_path: Optional[str] = None,
) -> Tuple[List[torch.Tensor], Dict[str, int], Dict[int, str]]:
    """Load embedding files and align gene order — NO preprocessing.

    Returns:
        (raw_per_modality, gene_to_idx, idx_to_gene)
        raw_per_modality: list of (n_genes, dim_i) tensors, one per file.
            Tensors may contain NaN for missing genes.
    """
    raw_per_modality: List[torch.Tensor] = []
    canonical_gene_order: Optional[List[str]] = None

    for i, path in enumerate(paths):
        emb, gene_order = load_single_embedding(path, gene_list_path)
        if canonical_gene_order is None:
            canonical_gene_order = gene_order
        else:
            if gene_order != canonical_gene_order:
                mismatches = []
                for j, (a,
                        b) in enumerate(zip(canonical_gene_order, gene_order)):
                    if a != b:
                        mismatches.append(f"  idx {j}: {a} vs {b}")
                    if len(mismatches) >= 5:
                        break
                if len(canonical_gene_order) != len(gene_order):
                    mismatches.append(
                        f"  lengths differ: {len(canonical_gene_order)} "
                        f"vs {len(gene_order)}")
                raise ValueError(
                    f"Gene order mismatch between {paths[0]} and {path}. "
                    f"All embeddings must share the same gene list.\n" +
                    "\n".join(mismatches))
        raw_per_modality.append(emb)

    gene_to_idx = {gene: idx for idx, gene in enumerate(canonical_gene_order)}
    idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
    return raw_per_modality, gene_to_idx, idx_to_gene


def fit_multimodal_transform(
    raw_per_modality: List[torch.Tensor],
    fit_mask: torch.Tensor,
    pca_dims: Optional[List[float]] = None,
    post_pca: Optional[float] = None,
    labels: Optional[List[str]] = None,
    pca_method: str = "plain",
) -> Dict:
    """Fit imputation + PCA + normalization stats on the fit subset of rows.

    Args:
        raw_per_modality: list of (n_genes, dim_i) tensors (possibly with NaN).
        fit_mask: (n_genes,) bool — rows used to fit every statistic.
        pca_dims: per-modality PCA targets (int or float in (0,1)). None to skip.
        post_pca: post-concat PCA target. None to skip.
        labels: optional per-modality display labels for logs.
        pca_method: "plain" (default) — mean-impute + per-column mean/std
            standardize + SVD + global mean/std normalize. All classical
            measures.
            "robust" — Huber-impute + ROBPCA (+ alpha retries +
            median-centered-SVD fallback) + global median/MAD normalize.
            Selects the whole pipeline, applied uniformly to per-modality
            and post-concat stages.

    Returns:
        transform dict (torch.save-serializable):
          {
            "per_modality": [
              {"impute_locations": tensor,
               "pca_pre_center":  tensor or None,  # per-column, plain only
               "pca_pre_scale":   tensor or None,  # per-column, plain only
               "pca_V":           tensor or None,
               "median":          float,           # center (median or mean)
               "mad":             float},          # scale  (MAD    or std)
              ...
            ],
            "post_pca_pre_center": tensor or None,
            "post_pca_pre_scale":  tensor or None,
            "post_pca_V":          tensor or None,
            "post_median":         float  or None,
            "post_mad":            float  or None,
            "pca_method":          "robust" | "plain",
          }
        Fields named `median`/`mad` carry the fitted scalar center/scale
        regardless of whether they came from the robust (median/MAD) or
        plain (mean/std) path — kept for backward-compat schema.
    """
    if pca_dims is not None and len(pca_dims) != len(raw_per_modality):
        raise ValueError(f"pca_dims has {len(pca_dims)} values but "
                         f"{len(raw_per_modality)} modalities provided")

    if len(raw_per_modality) == 0:
        raise ValueError("raw_per_modality must be a non-empty list")
    n_total = raw_per_modality[0].shape[0]
    if fit_mask.dim() != 1 or fit_mask.shape[0] != n_total:
        raise ValueError(
            f"fit_mask must be a 1D bool tensor of length {n_total}; "
            f"got shape {tuple(fit_mask.shape)}")
    if fit_mask.dtype != torch.bool:
        raise ValueError(
            f"fit_mask must be dtype torch.bool; got {fit_mask.dtype}")
    for i, raw in enumerate(raw_per_modality):
        if raw.shape[0] != n_total:
            raise ValueError(
                f"modality {i} has {raw.shape[0]} rows but modality 0 has "
                f"{n_total} — all modalities must share gene order")
    n_fit = int(fit_mask.sum().item())
    if n_fit == 0:
        raise ValueError("fit_mask selects zero rows — cannot fit")
    print(f"  Fit subset: {n_fit}/{n_total} rows "
          f"({100.0 * n_fit / max(1, n_total):.1f}%)")

    per_modality_t: List[Dict] = []
    imputed_fit_pieces: List[torch.Tensor] = []  # for post-concat fit

    # Pick the fit primitives that match the requested method.
    # robust: Huber-impute, ROBPCA (or median-SVD fallback), median/MAD normalize.
    # plain : mean-impute,  classical PCA (mean+std),           mean/std normalize.
    impute_fitter = (_fit_mean_locations
                     if pca_method == "plain" else _fit_huber_locations)
    impute_label = "mean" if pca_method == "plain" else "Huber"

    for i, raw in enumerate(raw_per_modality):
        label = labels[i] if labels else f"mod{i}"
        dim = raw.shape[1]
        nan_mask = torch.isnan(raw)
        n_nan_genes = nan_mask.any(dim=1).sum().item()
        print(f"  [{label}] {dim}d, {raw.shape[0]} genes, "
              f"{n_nan_genes} missing")

        # Fit impute on fit rows, apply to all.
        impute_locations = impute_fitter(raw, fit_mask)
        imputed_all = _apply_huber_imputation(raw, impute_locations)
        if n_nan_genes > 0:
            print(f"    {impute_label}-imputed {n_nan_genes}/{raw.shape[0]} "
                  f"genes ({int(nan_mask.sum().item())} NaN cells; "
                  f"locations fit on {int(fit_mask.sum().item())} fit rows)")

        # Fit PCA on fit rows (after imputation), apply to all.
        # Plain PCA additionally fits per-column mean/std (pre-standardize)
        # and stores them in the transform so apply-time reproduces the
        # same standardization. Robust PCA has pre_center/pre_scale = None.
        fit_imputed = imputed_all[fit_mask]
        pca_t: Optional[Dict] = None
        if pca_dims is not None:
            pca_t = _fit_pca_V(fit_imputed, pca_dims[i], method=pca_method)

        if pca_t is None:
            emb_all = imputed_all
            pca_V = None
            pre_center = None
            pre_scale = None
        else:
            pca_V = pca_t["V"]
            pre_center = pca_t["pre_center"]
            pre_scale = pca_t["pre_scale"]
            if pre_center is not None:
                emb_all = _apply_column_standardize(imputed_all, pre_center,
                                                    pre_scale)
            else:
                emb_all = imputed_all
            emb_all = emb_all @ pca_V

        # Fit normalize on fit rows (after PCA), apply to all.
        fit_emb = emb_all[fit_mask]
        center, scale = _fit_normalize_scalars(fit_emb, method=pca_method)
        emb_all = _apply_normalize(emb_all, center, scale)

        per_modality_t.append({
            "impute_locations": impute_locations,
            "pca_pre_center": pre_center,
            "pca_pre_scale": pre_scale,
            "pca_V": pca_V,
            # Kept as "median"/"mad" for transform-dict backward compat —
            # these carry the fitted scalar center/scale regardless of whether
            # they came from the robust (median/MAD) or plain (mean/std) path.
            "median": center,
            "mad": scale,
        })
        imputed_fit_pieces.append(emb_all[fit_mask])

    post_pca_V: Optional[torch.Tensor] = None
    post_pca_pre_center: Optional[torch.Tensor] = None
    post_pca_pre_scale: Optional[torch.Tensor] = None
    post_median: Optional[float] = None
    post_mad: Optional[float] = None
    if post_pca is not None:
        concatenated_fit = torch.cat(imputed_fit_pieces, dim=1)
        post_pca_t = _fit_pca_V(concatenated_fit, post_pca, method=pca_method)
        if post_pca_t is not None:
            post_pca_V = post_pca_t["V"]
            post_pca_pre_center = post_pca_t["pre_center"]
            post_pca_pre_scale = post_pca_t["pre_scale"]
            if post_pca_pre_center is not None:
                concatenated_fit = _apply_column_standardize(
                    concatenated_fit, post_pca_pre_center, post_pca_pre_scale)
            concatenated_fit = concatenated_fit @ post_pca_V
        # Always re-normalize after the post-concat PCA (even if PCA was a
        # no-op) so the output scale matches the pre-refactor pipeline.
        post_median, post_mad = _fit_normalize_scalars(concatenated_fit,
                                                       method=pca_method)

    return {
        "per_modality": per_modality_t,
        "post_pca_pre_center": post_pca_pre_center,
        "post_pca_pre_scale": post_pca_pre_scale,
        "post_pca_V": post_pca_V,
        "post_median": post_median,
        "post_mad": post_mad,
        "pca_method": pca_method,
    }


def apply_multimodal_transform(
    raw_per_modality: List[torch.Tensor],
    transform: Dict,
) -> torch.Tensor:
    """Apply a fitted multimodal transform to all rows — returns concatenated.

    Produces the full (n_genes, post_dim) embedding matrix used by the model.
    """
    if len(raw_per_modality) != len(transform["per_modality"]):
        raise ValueError(
            f"Transform has {len(transform['per_modality'])} modalities but "
            f"got {len(raw_per_modality)} raw tensors")

    pieces: List[torch.Tensor] = []
    for i, (raw, mod_t) in enumerate(
            zip(raw_per_modality, transform["per_modality"])):
        if raw.shape[1] != mod_t["impute_locations"].shape[0]:
            raise ValueError(
                f"modality {i}: raw dim {raw.shape[1]} != transform's "
                f"impute_locations dim {mod_t['impute_locations'].shape[0]}. "
                f"Embedding files must match the ones used to fit the "
                f"transform (order matters).")
        emb = _apply_huber_imputation(raw, mod_t["impute_locations"])
        # Pre-PCA per-column standardization (plain method only; both
        # fields are None in the robust path).
        pre_center = mod_t.get("pca_pre_center")
        pre_scale = mod_t.get("pca_pre_scale")
        if pre_center is not None:
            if emb.shape[1] != pre_center.shape[0]:
                raise ValueError(
                    f"modality {i}: post-impute dim {emb.shape[1]} != "
                    f"pca_pre_center dim {pre_center.shape[0]}.")
            emb = _apply_column_standardize(emb, pre_center, pre_scale)
        if mod_t["pca_V"] is not None:
            if emb.shape[1] != mod_t["pca_V"].shape[0]:
                raise ValueError(
                    f"modality {i}: post-impute dim {emb.shape[1]} != "
                    f"pca_V input dim {mod_t['pca_V'].shape[0]}.")
            emb = emb @ mod_t["pca_V"]
        emb = _apply_normalize(emb, mod_t["median"], mod_t["mad"])
        pieces.append(emb)

    concatenated = torch.cat(pieces, dim=1)

    # Post-concat pre-PCA standardization (plain method only).
    post_pre_center = transform.get("post_pca_pre_center")
    post_pre_scale = transform.get("post_pca_pre_scale")
    if post_pre_center is not None:
        concatenated = _apply_column_standardize(concatenated, post_pre_center,
                                                 post_pre_scale)
    post_V = transform.get("post_pca_V")
    if post_V is not None:
        concatenated = concatenated @ post_V
    # Post-normalize was fit whenever post_pca was requested (see
    # fit_multimodal_transform) regardless of whether post_V is None.
    if transform.get("post_median") is not None:
        concatenated = _apply_normalize(concatenated, transform["post_median"],
                                        transform["post_mad"])
    return concatenated


def load_multimodal_embeddings(
    paths: List[str],
    gene_list_path: Optional[str] = None,
    pca_dims: Optional[List[float]] = None,
    post_pca: Optional[float] = None,
    pca_method: str = "plain",
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """Legacy entry: load, fit preprocessing on ALL genes, apply, concatenate.

    This function preserves the old all-gene-fit behavior for predict.py and
    notebooks. Training-time per-fold preprocessing should call
    load_raw_multimodal + fit_multimodal_transform with a fold-specific mask
    (see SLDataManager.get_fold_embeddings).

    pca_method:
      "plain" (DEFAULT)  — mean-impute + per-column mean/std standardize +
                           SVD + global mean/std normalize (classical
                           end-to-end). This is the default everywhere since
                           2026-08-10; the signature above defaults to it.
      "robust"           — Huber-impute + ROBPCA + median/MAD normalize.
      "svd"              — Huber-impute + matrix-wise median/MAD rescale +
                           median-centered SVD, no per-column standardize.
    """
    n = len(paths)
    desc = "modality" if n == 1 else f"{n} modalities"
    print(f"\nLoading {desc} (impute → PCA → normalize → concat):")

    raw_per_modality, gene_to_idx, idx_to_gene = load_raw_multimodal(
        paths, gene_list_path)

    labels = [Path(p).stem.replace("all_genes_", "") for p in paths]
    fit_mask = torch.ones(raw_per_modality[0].shape[0], dtype=torch.bool)

    transform = fit_multimodal_transform(
        raw_per_modality,
        fit_mask,
        pca_dims=pca_dims,
        post_pca=post_pca,
        labels=labels,
        pca_method=pca_method,
    )
    concatenated = apply_multimodal_transform(raw_per_modality, transform)

    total_dim_after_mod = sum(
        (t["pca_V"].
         shape[1] if t["pca_V"] is not None else raw_per_modality[i].shape[1])
        for i, t in enumerate(transform["per_modality"]))
    print(f"  -> Concatenated: {total_dim_after_mod}d "
          f"({concatenated.shape[0]} genes)")
    if transform["post_pca_V"] is not None:
        print(f"  -> Post-concat ROBPCA + renorm: {total_dim_after_mod}d -> "
              f"{concatenated.shape[1]}d (target={post_pca})")

    return concatenated, gene_to_idx, idx_to_gene


# =============================================================================
# CV + Dataset
# =============================================================================

# Import SLMGAE benchmark's data splitting code
from data_split import SLDataSplitter

# Single set_seed definition lives in siamese_esm.py
from siamese_esm import set_seed

# Cell-line heads for masked multi-label conditioning (optional).
import cell_line_vocab


class SLDataset(Dataset):
    """
    Dataset for Synthetic Lethality prediction.

    Each sample is a gene pair with:
      - Embedding for gene1 (any supported type or multi-modal concatenation)
      - Embedding for gene2
      - Label (1 = SL, 0 = non-SL)
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        pairs: np.ndarray,
        labels: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        """
        Args:
            embeddings: Gene embeddings tensor (num_genes, embed_dim)
            pairs: Gene index pairs (N, 2)
            labels: Single-label (N,) in {0,1}; OR multi-label (N, num_heads)
                when ``mask`` is given (cell-line mode).
            mask: Optional multi-label supervision mask (N, num_heads), 1 where
                the head is known/supervised. When provided, __getitem__ yields
                a 4-tuple (x1, x2, label_vec, mask_vec) for masked multi-label
                BCE; otherwise the original 3-tuple (x1, x2, label).
        """
        self.embeddings = embeddings
        self.pairs = pairs
        self.labels = labels.astype(np.float32)
        self.mask = mask.astype(np.float32) if mask is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        i, j = self.pairs[idx]
        if self.mask is not None:
            return (
                self.embeddings[i],
                self.embeddings[j],
                torch.tensor(self.labels[idx]),  # (num_heads,)
                torch.tensor(self.mask[idx]),  # (num_heads,)
            )
        return (
            self.embeddings[i],
            self.embeddings[j],
            torch.tensor(self.labels[idx]),
        )


class SLDataManager:
    """
    Manages data loading and CV splits for SL prediction.

    Uses SLMGAE benchmark's SLDataSplitter for CV1/CV2/CV3 splits
    to ensure consistency with published results.

    Preprocessing (impute → PCA → normalize) is fit PER FOLD on genes that
    appear in training pairs (plus non-SL genes that appear in no pair at
    all). Genes that appear only in a test pair are excluded from the fit
    so their embedding values cannot shape the PCA basis — closing the
    CV2/CV3 leakage path. CV1 is unaffected because its test pairs reuse
    training genes.

    NOTE: The --seed argument controls model init AND CV splitting.
    """

    def __init__(
        self,
        embeddings_paths: List[str],
        sl_pairs_path: str = "",
        neg_pairs_path: Optional[str] = None,
        gene_list_path: Optional[str] = None,
        seed: int = 42,
        pca_dims: Optional[List[float]] = None,
        post_pca: Optional[float] = None,
        preprocessing_fit_scope: str = "train",
        pca_method: str = "plain",
        use_cell_lines: bool = False,
        folds_dir: Optional[str] = None,
    ):
        """
        Args:
            embeddings_paths: List of .pt embedding files.
            sl_pairs_path: Path to SL pairs file (gene1 TAB gene2 [TAB weight])
            neg_pairs_path: Optional path to a curated NON-SL pairs file
                (gene1 TAB gene2). When given, these experimentally screened
                negatives replace the benchmark's random negative sampling.
                Only pairs whose BOTH genes also appear in a positive pair are
                usable (the splitter works in SL-only index space); pairs that
                coincide with a positive are dropped. When None, the benchmark
                samples random non-edges as before.
            gene_list_path: Optional path to gene list (for ordering)
            seed: Random seed for reproducibility
            pca_dims: Per-modality PCA targets (one per embedding file).
                Int for exact count, float in (0,1) for variance fraction.
            post_pca: Optional post-concatenation ROBPCA target.
            preprocessing_fit_scope: Which gene rows to fit preprocessing
                stats (Huber locations, ROBPCA basis, median/MAD) on.
                - "train" (default): per-fold fit on training-pair genes
                  plus non-SL genes; test-only genes excluded — leak-free.
                - "all": fit on every gene regardless of fold. Matches the
                  pre-refactor behavior and reintroduces CV2/CV3 leakage
                  of test-gene embeddings into the basis. Kept for A/B
                  comparison so the effect of the fix can be quantified.
            pca_method: "plain" (DEFAULT) — per-column mean/std standardize
                then classical SVD, with a global mean/std normalize.
                "robust" — ROBPCA with alpha retries and a median-centered-SVD
                fallback on failure. "svd" — median-centered SVD directly, no
                per-column standardize, no outlier filtering. Applies
                uniformly to per-modality and post-concat PCA.
            use_cell_lines: Enable masked multi-label cell-line conditioning.
                Requires neg_pairs_path. The unit becomes a gene pair carrying
                a per-head (label, mask) vector over cell_line_vocab.HEADS
                (read from the ".sources.tsv" sidecars next to sl_pairs_path /
                neg_pairs_path). The unified set of pairs that are SL and/or
                non-SL in some head is split gene-disjointly (the splitter's
                random negatives are discarded — supervision comes entirely
                from the label/mask), and the model (SiameseSLMultiCell) emits
                one logit per head. neg_pairs_path's pairs are NOT restricted
                to the SL gene universe here; the gene universe is all pairs
                that carry any known label.
            folds_dir: Use the shared canonical CV folds in this directory
                (e.g. ../sl_comparison/folds) instead of siamese's own
                gene-disjoint partition. This is what makes the four-model
                comparison fair: identical train/val/test division and
                identical gene index space. Per-head labels stay siamese's,
                so cell-line conditioning is unaffected — only the PARTITION
                is shared. Requires use_cell_lines=True.
        """
        if not embeddings_paths:
            raise ValueError("embeddings_paths must be a non-empty list")
        if preprocessing_fit_scope not in ("train", "all"):
            raise ValueError(
                f"preprocessing_fit_scope must be 'train' or 'all'; "
                f"got {preprocessing_fit_scope!r}")
        if pca_method not in ("robust", "plain", "svd"):
            raise ValueError(f"pca_method must be 'robust', 'plain' or 'svd'; "
                             f"got {pca_method!r}")

        self.seed = seed
        set_seed(seed)
        self.use_cell_lines = bool(use_cell_lines)
        if self.use_cell_lines and not neg_pairs_path:
            raise ValueError(
                "use_cell_lines=True requires neg_pairs_path (the screened "
                "non-SL pairs provide the negative labels per cell line).")
        self.folds_dir = folds_dir or None
        if self.folds_dir and not self.use_cell_lines:
            raise ValueError(
                "folds_dir requires use_cell_lines=True; the shared-fold "
                "reader maps onto the multi-label pair universe.")
        self.embeddings_paths = list(embeddings_paths)
        self.pca_dims = pca_dims
        self.post_pca = post_pca
        self.preprocessing_fit_scope = preprocessing_fit_scope
        self.pca_method = pca_method
        self._modality_labels = [
            Path(p).stem.replace("all_genes_", "") for p in embeddings_paths
        ]

        # Load RAW per-modality tensors (no preprocessing — preprocessing is
        # fit per-fold via get_fold_embeddings).
        print(f"\nLoading {len(embeddings_paths)} modality file(s) [raw, "
              f"preprocessing deferred to per-fold fit]:")
        self.raw_per_modality, self.gene_to_idx, self.idx_to_gene = \
            load_raw_multimodal(embeddings_paths, gene_list_path)
        self.num_genes = len(self.gene_to_idx)

        # Load SL pairs (positive pairs) - indices into full embedding matrix
        self.pos_pairs_full_idx, self.sl_genes = self._load_sl_pairs(
            sl_pairs_path)
        self.num_sl_genes = len(self.sl_genes)

        # Create mapping between full embedding indices and SL-only indices
        # This is needed because SLDataSplitter expects indices in [0, num_sl_genes)
        self._create_sl_index_mapping()

        # Convert pos_pairs to SL-only indices for the splitter
        self.pos_pairs_sl_idx = self._full_to_sl_indices(
            self.pos_pairs_full_idx)

        # Negative pairs for the benchmark splitter.
        #   "random"       — leave the pool empty; the benchmark samples random
        #                    non-edges among SL genes during splitting.
        #   "experimental" — inject curated non-SL pairs (e.g. SynLethDB
        #                    Human.non.SL) as the negative pool, restricted to
        #                    the splitter's SL-only index space and disjoint
        #                    from the positives.
        if self.use_cell_lines:
            # Masked multi-label cell-line mode: no single negative pool — each
            # pair carries a per-head (label, mask). Build that universe.
            self.negative_strategy = "multilabel"
            self.neg_pairs = np.zeros((0, 2), dtype=np.int64)
            neg_stats = None
            self._setup_multilabel(sl_pairs_path, neg_pairs_path)
        elif neg_pairs_path:
            self.negative_strategy = "experimental"
            self.neg_pairs, neg_stats = self._load_neg_pairs(neg_pairs_path)
            if len(self.neg_pairs) == 0:
                raise ValueError(
                    f"No usable experimental negatives loaded from "
                    f"{neg_pairs_path!r} (all dropped: missing embedding, "
                    f"outside the positive-SL gene universe, or coinciding "
                    f"with a positive pair). Omit --neg_pairs_path to use "
                    f"random negatives.")
        else:
            self.negative_strategy = "random"
            self.neg_pairs = np.zeros((0, 2), dtype=np.int64)
            neg_stats = None

        print(f"Loaded raw tensors for {self.num_genes} genes")
        print(f"Loaded {len(self.pos_pairs_full_idx)} positive SL pairs")
        print(f"SL pairs involve {self.num_sl_genes} unique genes")
        if self.negative_strategy == "multilabel":
            pass  # _setup_multilabel already printed a per-head summary
        elif self.negative_strategy == "experimental":
            print(f"Negative strategy: EXPERIMENTAL "
                  f"({len(self.neg_pairs):,} curated non-SL pairs in pool)")
            for key, val in neg_stats.items():
                print(f"    {key}: {val:,}")
        else:
            print("Negative strategy: RANDOM "
                  "(benchmark samples non-edges during splitting)")

    def _load_sl_pairs(self,
                       sl_pairs_path: str) -> Tuple[np.ndarray, Set[str]]:
        """
        Load positive SL pairs from file.

        The file contains Entrez Gene ID pairs (tab-separated).
        If the file still contains gene symbols, they are normalized
        and converted to Entrez IDs.

        Returns pairs as indices into the embedding matrix.
        Only includes pairs where BOTH genes have embeddings.
        """
        pairs = []
        sl_genes = set()
        seen = set()

        mapper = get_mapper()

        with open(sl_pairs_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    g1, g2 = parts[0], parts[1]
                    if not g1.isdigit():
                        g1 = mapper.symbol_to_entrez(g1) or ""
                    if not g2.isdigit():
                        g2 = mapper.symbol_to_entrez(g2) or ""

                    if g1 not in self.gene_to_idx or g2 not in self.gene_to_idx:
                        continue

                    idx1, idx2 = self.gene_to_idx[g1], self.gene_to_idx[g2]
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1

                    pair_key = (idx1, idx2)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append([idx1, idx2])
                        sl_genes.add(g1)
                        sl_genes.add(g2)

        return np.array(pairs, dtype=np.int64), sl_genes

    def _load_neg_pairs(self, neg_pairs_path: str) -> Tuple[np.ndarray, Dict]:
        """Load curated NON-SL pairs as a negative pool (SL-only indices).

        Mirrors _load_sl_pairs' gene handling, but a pair is kept only if:
          * both genes have an embedding,
          * both genes appear in at least one positive pair (so they live in
            the splitter's SL-only index space), and
          * the pair is not itself a positive (drops SL/non-SL conflicts).

        Self-loops and duplicates are removed and the pair is canonicalized
        (idx1 < idx2). Returns negatives in SL-only indices plus a stats dict.
        """
        mapper = get_mapper()
        # Positive pairs are already canonical (idx1 < idx2, full indices).
        pos_set = set(map(tuple, self.pos_pairs_full_idx.tolist()))

        neg_sl = []
        seen = set()
        n_total = 0
        n_no_embedding = 0
        n_outside_universe = 0
        n_conflict_with_pos = 0
        n_self_loop = 0

        with open(neg_pairs_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                n_total += 1
                g1, g2 = parts[0], parts[1]
                if not g1.isdigit():
                    g1 = mapper.symbol_to_entrez(g1) or ""
                if not g2.isdigit():
                    g2 = mapper.symbol_to_entrez(g2) or ""

                if g1 not in self.gene_to_idx or g2 not in self.gene_to_idx:
                    n_no_embedding += 1
                    continue

                idx1, idx2 = self.gene_to_idx[g1], self.gene_to_idx[g2]
                if idx1 == idx2:
                    n_self_loop += 1
                    continue
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                # Both genes must live in the SL-only index space.
                if idx1 not in self.full_to_sl or idx2 not in self.full_to_sl:
                    n_outside_universe += 1
                    continue

                if (idx1, idx2) in pos_set:
                    n_conflict_with_pos += 1
                    continue

                key = (idx1, idx2)
                if key in seen:
                    continue
                seen.add(key)
                neg_sl.append([self.full_to_sl[idx1], self.full_to_sl[idx2]])

        stats = {
            "neg_pairs_in_file": n_total,
            "kept": len(neg_sl),
            "dropped_no_embedding": n_no_embedding,
            "dropped_outside_sl_universe": n_outside_universe,
            "dropped_conflict_with_positive": n_conflict_with_pos,
            "dropped_self_loop": n_self_loop,
        }
        return np.array(neg_sl, dtype=np.int64).reshape(-1, 2), stats

    # ----- Masked multi-label cell-line mode ------------------------------
    def _sidecar_path(self, pairs_path: str) -> str:
        """Path to the ".sources.tsv" provenance sidecar for a pairs file."""
        if pairs_path.endswith(".txt"):
            return pairs_path[:-len(".txt")] + ".sources.tsv"
        return pairs_path + ".sources.tsv"

    def _read_sidecar_cells(self, path: str) -> Dict[Tuple[int, int], str]:
        """Map canonical full-idx pair -> raw cell_lines field from a sidecar
        (header: entrez1, entrez2, sources, cell_lines, pubmed_ids). Genes
        without an embedding are skipped; last write wins on duplicate pairs."""
        mapper = get_mapper()
        out: Dict[Tuple[int, int], str] = {}
        with open(path, "r") as f:
            f.readline()  # header
            for line in f:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 4:
                    continue
                g1, g2, cell_field = fields[0], fields[1], fields[3]
                if not g1.isdigit():
                    g1 = mapper.symbol_to_entrez(g1) or ""
                if not g2.isdigit():
                    g2 = mapper.symbol_to_entrez(g2) or ""
                if g1 not in self.gene_to_idx or g2 not in self.gene_to_idx:
                    continue
                idx1, idx2 = self.gene_to_idx[g1], self.gene_to_idx[g2]
                if idx1 == idx2:
                    continue
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                out[(idx1, idx2)] = cell_field
        return out

    def _load_neg_pairs_full(self,
                             neg_pairs_path: str) -> Set[Tuple[int, int]]:
        """Load NON-SL pairs as canonical full-idx tuples (both genes embedded).

        Unlike _load_neg_pairs (single-label), this does NOT restrict to the
        SL gene universe and does NOT drop SL/non-SL conflicts — in multi-label
        mode a pair can be SL in one head and non-SL in another."""
        mapper = get_mapper()
        out: Set[Tuple[int, int]] = set()
        with open(neg_pairs_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                g1, g2 = parts[0], parts[1]
                if not g1.isdigit():
                    g1 = mapper.symbol_to_entrez(g1) or ""
                if not g2.isdigit():
                    g2 = mapper.symbol_to_entrez(g2) or ""
                if g1 not in self.gene_to_idx or g2 not in self.gene_to_idx:
                    continue
                idx1, idx2 = self.gene_to_idx[g1], self.gene_to_idx[g2]
                if idx1 == idx2:
                    continue
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                out.add((idx1, idx2))
        return out

    def _setup_multilabel(self, sl_pairs_path: str,
                          neg_pairs_path: str) -> None:
        """Build the labeled-pair universe with per-head (label, mask) vectors.

        Each pair that is SL and/or non-SL in some head becomes one example;
        its label/mask vectors over cell_line_vocab.HEADS come from the SL and
        non-SL ".sources.tsv" sidecars. A P-gene index space (genes appearing
        in any labeled pair) is built for the splitter.
        """
        H = cell_line_vocab.NUM_HEADS
        pos_sidecar = self._sidecar_path(sl_pairs_path)
        neg_sidecar = self._sidecar_path(neg_pairs_path)
        for p in (pos_sidecar, neg_sidecar):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"use_cell_lines=True needs the provenance sidecar {p!r} "
                    f"(produced by download_synlethdb.py). Not found.")
        pos_raw = self._read_sidecar_cells(pos_sidecar)
        neg_raw = self._read_sidecar_cells(neg_sidecar)

        pos_set = set(map(tuple, self.pos_pairs_full_idx.tolist()))
        neg_set = self._load_neg_pairs_full(neg_pairs_path)
        universe = sorted(pos_set | neg_set)

        pairs: List[List[int]] = []
        labels: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        n_dropped_unlabeled = 0
        n_sidecar_missing = 0  # pair in a .txt set but absent from its sidecar
        n_or_pairs = 0  # pairs with >=1 head asserted both ways
        or_per_head = np.zeros(H, dtype=np.int64)
        for (i, j) in universe:
            is_sl = (i, j) in pos_set
            is_ns = (i, j) in neg_set
            # The .txt pairs file and its .sources.tsv sidecar are written in
            # lockstep by download_synlethdb.py; a membership without a sidecar
            # row means they drifted. Count it (a missing field falls back to
            # "" -> the pair's known side becomes OTHER, which we surface).
            if (is_sl and (i, j) not in pos_raw) or \
               (is_ns and (i, j) not in neg_raw):
                n_sidecar_missing += 1
            sl_field = pos_raw.get((i, j), "")
            ns_field = neg_raw.get((i, j), "")
            both = cell_line_vocab.same_head_conflicts(sl_field, ns_field,
                                                       is_sl, is_ns)
            if both:
                n_or_pairs += 1
                for h in both:
                    or_per_head[h] += 1
            lab, msk = cell_line_vocab.pair_label_mask(sl_field, ns_field,
                                                       is_sl, is_ns)
            if msk.sum() == 0:  # no known head (never screened in any head)
                n_dropped_unlabeled += 1
                continue
            pairs.append([i, j])
            labels.append(lab)
            masks.append(msk)

        self.ml_pairs_full = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        self.ml_label = np.array(labels, dtype=np.float32).reshape(-1, H)
        self.ml_mask = np.array(masks, dtype=np.float32).reshape(-1, H)

        # P-gene index space: genes that appear in any labeled pair, used ONLY
        # to assign gene-disjoint CV folds (cv2/cv3). The dataset itself uses
        # ml_pairs_full (full embedding indices) directly.
        genes = sorted({int(g) for pr in self.ml_pairs_full for g in pr})
        full_to_p = {g: p for p, g in enumerate(genes)}
        self.num_p_genes = len(genes)
        self.ml_pairs_p = np.array([[full_to_p[int(i)], full_to_p[int(j)]]
                                    for i, j in self.ml_pairs_full],
                                   dtype=np.int64)

        if n_sidecar_missing:
            print(f"    WARNING: {n_sidecar_missing:,} pair(s) present in a "
                  f".txt set but missing from its .sources.tsv sidecar "
                  f"(treated as OTHER-head). Check the files are in sync.")

        # Summary.
        known = self.ml_mask.sum(axis=0)  # per head, #supervised
        pos_per = (self.ml_label * self.ml_mask).sum(axis=0)
        print(f"Cell-line conditioning: ENABLED (masked multi-label, "
              f"{H} heads)")
        print(f"    labeled pairs: {len(self.ml_pairs_full):,} over "
              f"{self.num_p_genes:,} genes "
              f"(dropped {n_dropped_unlabeled:,} with no known head)")
        if n_or_pairs:
            # These are the pairs screened BOTH ways within one head. Under the
            # legacy same_head="mask" rule they were unsupervised there; the OR
            # rule labels them SL. Printed per head because the count is very
            # unevenly distributed (JURKAT carries most of it) and a head whose
            # positives are largely recovered conflicts should be read as such.
            recovered = ", ".join(
                f"{cell_line_vocab.HEADS[h]} {int(or_per_head[h]):,}"
                for h in np.argsort(-or_per_head) if or_per_head[h])
            print(f"    same-head SL/non-SL conflicts resolved by OR "
                  f"(label=SL): {n_or_pairs:,} pairs -> "
                  f"{int(or_per_head.sum()):,} head-cells [{recovered}]")
        for h, name in enumerate(cell_line_vocab.HEADS):
            print(
                f"    {name:<8} known {int(known[h]):>7,}  "
                f"pos {int(pos_per[h]):>6,}  neg {int(known[h]-pos_per[h]):>7,}"
                f"  (or-resolved {int(or_per_head[h]):>5,})")

    def _make_ml_folds(self, cv_type: str, num_folds: int) -> List[Tuple]:
        """Gene-disjoint CV folds for the multi-label universe (no benchmark
        splitter, no negative sampling — every pair is one example with a
        label/mask vector, so we just partition the pairs).

        Returns a list of (train_idx, test_idx) arrays into the ml_* arrays.
          cv1: random edge KFold (genes may recur — easiest).
          cv2: KFold the P-genes; test = pairs touching the held-out gene group
               (≥1 unseen gene), train = pairs with neither gene held out.
          cv3: same gene KFold but test = pairs with BOTH genes held out and
               train = pairs with neither held out; pairs with exactly one
               held-out gene are dropped for that fold (both genes must be
               unseen). Train never sees a held-out gene -> leak-free.
        """
        rng = np.random.RandomState(self.seed)
        n = len(self.ml_pairs_p)
        folds: List[Tuple] = []
        if cv_type == "cv1":
            groups = np.array_split(rng.permutation(n), num_folds)
            for f in range(num_folds):
                test_idx = groups[f]
                train_idx = np.concatenate(
                    [groups[g] for g in range(num_folds) if g != f])
                folds.append((train_idx, test_idx))
            return folds
        # cv2 / cv3: partition genes into folds, classify pairs by gene group.
        gene_fold = np.empty(self.num_p_genes, dtype=np.int64)
        for f, grp in enumerate(
                np.array_split(rng.permutation(self.num_p_genes), num_folds)):
            gene_fold[grp] = f
        g1 = gene_fold[self.ml_pairs_p[:, 0]]
        g2 = gene_fold[self.ml_pairs_p[:, 1]]
        for f in range(num_folds):
            u1, u2 = (g1 == f), (g2 == f)
            neither = ~u1 & ~u2
            train_idx = np.where(neither)[0]
            if cv_type == "cv2":
                test_idx = np.where(u1 | u2)[0]
            else:  # cv3
                test_idx = np.where(u1 & u2)[0]
            folds.append((train_idx, test_idx))
        return folds

    def _shared_gene_map(self) -> Tuple[List[str], np.ndarray]:
        """Load the shared gene index space and map it onto embedding rows.

        Returns (shared_genes, shared_to_full) where shared_to_full[s] is the
        row of this run's embedding matrix holding shared gene s, or -1 when
        that gene has no embedding. With the genome-wide gene universe
        (run_generate_embeddings.sh STEP 0b) nothing should be -1; any misses
        are reported rather than silently dropped.
        """
        genes_file = Path(self.folds_dir) / "genes.txt"
        if not genes_file.exists():
            raise FileNotFoundError(
                f"folds_dir={self.folds_dir!r} has no genes.txt — run "
                f"sl_comparison/prepare_folds.py first.")
        with open(genes_file) as f:
            shared_genes = [ln.strip() for ln in f if ln.strip()]
        s2f = np.full(len(shared_genes), -1, dtype=np.int64)
        for s, g in enumerate(shared_genes):
            fi = self.gene_to_idx.get(g)
            if fi is not None:
                s2f[s] = fi
        return shared_genes, s2f

    def _shared_ml_folds(self, cv_type: str, num_folds: int) -> List[Tuple]:
        """Read sl_comparison/folds/<cv>/fold_k/*.npy -> ml_* row indices.

        This is the fairness path: the train/val/test PARTITION comes from the
        shared canonical folds (identical to SLMGAE / SLGNN / NSF4SL), while the
        per-head labels stay siamese's own, so cell-line conditioning survives.
        Because train rows are built ONLY from the shared train arrays, no
        shared val/test pair can leak into training regardless of how the two
        label universes differ.

        Returns [(train_idx, val_idx, test_idx), ...].
        """
        shared_genes, s2f = self._shared_gene_map()
        n_unmapped = int((s2f < 0).sum())
        if n_unmapped:
            print(f"    WARNING: {n_unmapped:,}/{len(shared_genes):,} shared "
                  f"genes have no embedding row; their pairs are unusable. "
                  f"Regenerate embeddings over the full gene universe.")

        row_of = {
            (int(a), int(b)): r
            for r, (a, b) in enumerate(self.ml_pairs_full)
        }

        def rows(arr: np.ndarray) -> Tuple[np.ndarray, int]:
            """Shared-index pairs -> ml row indices; count unrepresentable."""
            out, miss = [], 0
            for a, b in np.asarray(arr, dtype=np.int64):
                fa, fb = s2f[a], s2f[b]
                if fa < 0 or fb < 0:
                    miss += 1
                    continue
                key = (int(fa), int(fb)) if fa < fb else (int(fb), int(fa))
                r = row_of.get(key)
                if r is None:
                    miss += 1
                    continue
                out.append(r)
            return np.array(sorted(set(out)), dtype=np.int64), miss

        base = Path(self.folds_dir) / cv_type
        folds: List[Tuple] = []
        total_miss = 0
        for f in range(num_folds):
            d = base / f"fold_{f}"
            if not d.exists():
                raise FileNotFoundError(
                    f"{d} missing — prepare_folds.py was run with fewer than "
                    f"{num_folds} folds, or a different cv_types list.")

            def part(*names):
                nonlocal total_miss
                idxs = []
                for nm in names:
                    p = d / f"{nm}.npy"
                    if not p.exists():
                        continue
                    r, m = rows(np.load(p))
                    total_miss += m
                    idxs.append(r)
                if not idxs:
                    return np.array([], dtype=np.int64)
                return np.unique(np.concatenate(idxs))

            tr = part("train_pos", "train_neg")
            va = part("val_pos", "val_neg")
            te = part("test_pos", "test_neg")

            # Guard the whole point of this path: training must not touch any
            # pair the shared folds reserved for selection or scoring.
            for other, nm in ((va, "val"), (te, "test")):
                overlap = np.intersect1d(tr, other)
                if len(overlap):
                    raise RuntimeError(
                        f"{cv_type} fold {f}: {len(overlap)} pairs appear in "
                        f"both train and {nm} of the shared folds")
            folds.append((tr, va, te))

        if total_miss:
            print(f"    {total_miss:,} shared fold pair-slots carry no "
                  f"cell-line head in this run and are not trainable examples "
                  f"(they are still scored — grading uses the dense matrix).")
        return folds

    def _ml_fold_data(self,
                      fold: int,
                      train_idx: np.ndarray,
                      test_idx: np.ndarray,
                      val_idx: Optional[np.ndarray] = None) -> Dict:
        """Assemble a multi-label fold dict by slicing the ml_* arrays."""

        def part(idx):
            return {
                "pairs": self.ml_pairs_full[idx],
                "labels": self.ml_label[idx],
                "mask": self.ml_mask[idx],
            }

        out = {"fold": fold, "train": part(train_idx), "test": part(test_idx)}
        if val_idx is not None and len(val_idx):
            out["val"] = part(val_idx)
        return out

    def _get_ml_splits(self, cv_type: str, num_folds: int) -> List[Dict]:
        """Build + report multi-label folds for one CV type.

        Shared canonical folds when folds_dir is set (fair four-model
        comparison), otherwise siamese's own gene-disjoint partition.
        """
        if self.folds_dir:
            folds = self._shared_ml_folds(cv_type, num_folds)
        else:
            folds = [(tr, None, te)
                     for tr, te in self._make_ml_folds(cv_type, num_folds)]
        converted = []
        for f, (tr, va, te) in enumerate(folds):
            fd = self._ml_fold_data(f, tr, te, va)
            converted.append(fd)
            vtxt = (f"  val {len(va):,} pairs/"
                    f"{int(fd['val']['mask'].sum()):,} labels"
                    if "val" in fd else "")
            print(
                f"  Fold {f + 1}: "
                f"train {len(tr):,} pairs/{int(fd['train']['mask'].sum()):,} "
                f"labels{vtxt}  test {len(te):,} pairs/"
                f"{int(fd['test']['mask'].sum()):,} labels")
        return converted

    def _create_sl_index_mapping(self) -> None:
        """
        Create bidirectional mapping between full embedding indices and SL-only indices.
        """
        unique_full_indices = set()
        for g1, g2 in self.pos_pairs_full_idx:
            unique_full_indices.add(g1)
            unique_full_indices.add(g2)

        unique_full_indices = sorted(list(unique_full_indices))

        self.full_to_sl = {
            full_idx: sl_idx
            for sl_idx, full_idx in enumerate(unique_full_indices)
        }
        self.sl_to_full = {
            sl_idx: full_idx
            for full_idx, sl_idx in self.full_to_sl.items()
        }

    def _full_to_sl_indices(self, pairs: np.ndarray) -> np.ndarray:
        """Convert pairs from full embedding indices to SL-only indices."""
        converted = []
        for g1, g2 in pairs:
            converted.append([self.full_to_sl[g1], self.full_to_sl[g2]])
        return np.array(converted, dtype=np.int64)

    def _sl_to_full_indices(self, pairs: np.ndarray) -> np.ndarray:
        """Convert pairs from SL-only indices back to full embedding indices."""
        converted = []
        for g1, g2 in pairs:
            converted.append([self.sl_to_full[g1], self.sl_to_full[g2]])
        return np.array(converted, dtype=np.int64)

    def _create_splitter(self) -> SLDataSplitter:
        """Create SLMGAE benchmark's data splitter using SL-only indices."""
        return SLDataSplitter(
            pos_edges=self.pos_pairs_sl_idx,
            neg_edges=self.neg_pairs,
            num_nodes=self.num_sl_genes,
            train_ratio=0.8,
            random_state=self.seed,
            negative_strategy=self.negative_strategy,
        )

    def _convert_split_to_dataset_format(
        self,
        split: Dict,
    ) -> Dict:
        """
        Convert SLMGAE benchmark split format to our dataset format.
        """
        train_pairs_full = self._sl_to_full_indices(split["train_edges"])
        test_pairs_full = self._sl_to_full_indices(split["test_edges"])

        return {
            "fold": split["fold"],
            "train": {
                "pairs": train_pairs_full,
                "labels": split["train_labels"],
            },
            "test": {
                "pairs": test_pairs_full,
                "labels": split["test_labels"],
            },
            "num_train_pos": split["num_train_pos"],
            "num_train_neg": split["num_train_neg"],
            "num_test_pos": split["num_test_pos"],
            "num_test_neg": split["num_test_neg"],
        }

    def _convert_and_report(self, splits: List[Dict]) -> List[Dict]:
        """Convert raw splitter folds to dataset folds and print a per-fold
        summary (single-label path; cell-line mode uses _get_ml_splits)."""
        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold'] + 1}: "
                  f"train={split['num_train_pos']}+/"
                  f"{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/"
                  f"{split['num_test_neg']}-")
        return converted

    def get_cv1_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV1: Edge-based cross-validation (from SLMGAE benchmark).

        Randomly splits SL pairs into folds.
        Tests ability to predict held-out interactions between known genes.
        """
        if self.use_cell_lines:
            print("\nGenerating CV1 splits (edge-based, multi-label)...")
            return self._get_ml_splits("cv1", num_folds)
        print(
            f"\nGenerating CV1 splits (edge-based) using SLMGAE benchmark code..."
        )
        splitter = self._create_splitter()
        splits = splitter.cv1_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        return self._convert_and_report(splits)

    def get_cv2_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV2: Gene-based cross-validation (from SLMGAE benchmark).
        """
        if self.use_cell_lines:
            print("\nGenerating CV2 splits (gene-based, multi-label)...")
            return self._get_ml_splits("cv2", num_folds)
        print(
            f"\nGenerating CV2 splits (gene-based) using SLMGAE benchmark code..."
        )
        splitter = self._create_splitter()
        splits = splitter.cv2_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        return self._convert_and_report(splits)

    def get_cv3_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV3: Pair-based cross-validation (from SLMGAE benchmark).

        Both genes in test pairs are unseen during training.
        """
        if self.use_cell_lines:
            print("\nGenerating CV3 splits (pair-based, multi-label)...")
            return self._get_ml_splits("cv3", num_folds)
        print(
            f"\nGenerating CV3 splits (pair-based) using SLMGAE benchmark code..."
        )
        splitter = self._create_splitter()
        splits = splitter.cv3_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        return self._convert_and_report(splits)

    # ------------------------------------------------------------------
    # Per-fold preprocessing
    # ------------------------------------------------------------------

    def _fold_fit_mask(self, fold_data: Dict) -> torch.Tensor:
        """Build the fit mask for preprocessing.

        scope = "all":
          Every gene is a fit row. Preprocessing stats include test-only
          gene embeddings — reintroduces CV2/CV3 leakage. Use only for
          A/B comparison.

        scope = "train" (default, leak-free):
          - Genes that appear in any training pair (positive or negative)
            stay in the fit — their labels are the supervision signal.
          - Non-SL genes (absent from both training and test pairs) stay
            in the fit — they carry no label information to leak and keep
            the ROBPCA sample size high.
          - Genes that appear in test pairs but NOT in training pairs are
            excluded. In CV1 this set is small but NOT empty — an edge split
            guarantees nothing about gene reuse for low-degree genes, and
            2,629 genes have exactly one SL pair, so roughly a fifth of them
            land test-only on any given 5-fold split; in CV3 it is
            the full test-pair gene set.
        """
        if self.preprocessing_fit_scope == "all":
            return torch.ones(self.num_genes, dtype=torch.bool)

        train_genes: Set[int] = set()
        for g1, g2 in fold_data["train"]["pairs"]:
            train_genes.add(int(g1))
            train_genes.add(int(g2))
        test_genes: Set[int] = set()
        for g1, g2 in fold_data["test"]["pairs"]:
            test_genes.add(int(g1))
            test_genes.add(int(g2))
        # VAL is held out exactly like test and must be excluded on the same
        # terms. The shared cv2/cv3 folds carve val by holding out a GENE group
        # (prepare_folds.py:228-240), so val-only genes appear in no train pair
        # and no test pair — without this they survive `test_genes -
        # train_genes` and their raw embeddings shape the Huber locations, the
        # ROBPCA basis and the median/MAD. That makes val transductive and
        # optimistically biases the very number used both for per-epoch
        # checkpoint selection and for cross-config ranking in
        # collect_siamese.discover. Measured on the shared folds (fold 0):
        # 602 such genes in cv2, 154 in cv3.
        for g1, g2 in fold_data.get("val", {}).get("pairs", []):
            test_genes.add(int(g1))
            test_genes.add(int(g2))
        exclude = test_genes - train_genes

        mask = torch.ones(self.num_genes, dtype=torch.bool)
        if exclude:
            idx = torch.tensor(sorted(exclude), dtype=torch.long)
            mask[idx] = False
        return mask

    def get_fold_embeddings(
        self,
        fold_data: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Fit preprocessing on this fold's training-gene subset; return
        (fold_embeddings, transform_dict).

        transform_dict is the serializable form of MultiModalTransform (see
        fit_multimodal_transform) and should be saved alongside the fold's
        model checkpoint so inference time can reproduce the exact feature
        space via apply_multimodal_transform.
        """
        fit_mask = self._fold_fit_mask(fold_data)
        method_desc = {
            "robust": "Huber → ROBPCA → median/MAD",
            "plain": "mean → col-standardize+SVD → mean/std",
            "svd": "Huber → median-SVD (no col-standardize) → median/MAD",
        }.get(self.pca_method, self.pca_method)
        print(f"\n[fold {fold_data['fold']}] fitting preprocessing "
              f"[method={self.pca_method}: {method_desc}; "
              f"scope={self.preprocessing_fit_scope}]" +
              (" (+ post-concat PCA + renorm)" if self.
               post_pca is not None else ""))
        transform = fit_multimodal_transform(
            self.raw_per_modality,
            fit_mask,
            pca_dims=self.pca_dims,
            post_pca=self.post_pca,
            labels=self._modality_labels,
            pca_method=self.pca_method,
        )
        embeddings = apply_multimodal_transform(self.raw_per_modality,
                                                transform)
        print(f"[fold {fold_data['fold']}] fold embeddings: "
              f"{tuple(embeddings.shape)}")
        return embeddings, transform


def create_fold_dataloaders(
    embeddings: torch.Tensor,
    fold_data: Dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for a CV fold.

    Args:
        embeddings: Gene embeddings tensor (num_genes, dim). Should already
            be imputed (no NaN) — imputation happens at load time.
        fold_data: Fold dict from get_cv*_splits()
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = SLDataset(
        embeddings=embeddings,
        pairs=fold_data["train"]["pairs"],
        labels=fold_data["train"]["labels"],
        mask=fold_data["train"].get("mask"),
    )
    test_dataset = SLDataset(
        embeddings=embeddings,
        pairs=fold_data["test"]["pairs"],
        labels=fold_data["test"]["labels"],
        mask=fold_data["test"].get("mask"),
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, test_loader


def create_val_dataloader(
    embeddings: torch.Tensor,
    fold_data: Dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Optional[DataLoader]:
    """Loader over the shared VAL split, or None when the fold has no val.

    Only shared canonical folds carry a val partition (prepare_folds.py
    --val_frac); siamese's own folds do not, and those runs keep the legacy
    select-on-test behaviour.
    """
    if "val" not in fold_data or len(fold_data["val"]["pairs"]) == 0:
        return None
    return DataLoader(
        SLDataset(
            embeddings=embeddings,
            pairs=fold_data["val"]["pairs"],
            labels=fold_data["val"]["labels"],
            mask=fold_data["val"].get("mask"),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
