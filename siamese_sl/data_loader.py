#!/usr/bin/env python3
"""
Data loading utilities for Siamese SL prediction.

Symbol-based version (no Entrez ID conversion). Use this with the old
gene_name_utils.py that lacks symbol_to_entrez().

Unified embedding pipeline (applied per modality, then concatenated):
  1. Load .pt file, normalize gene symbols (custom corrections + HGNC), deduplicate
  2. Impute NaN via Huber M-estimates (per-dimension, global)
  3. Robust PCA (optional): robpy ROBPCA (Hubert et al., 2005) with
     fallback to iterative Huber M-estimator of scatter
  4. Normalize: center per-column median, scale by global MAD
  5. Concatenate along feature axis

A single embedding file goes through the same pipeline as multiple files —
there is no separate code path for single vs. multi-modal.

Cross-validation uses SLMGAE benchmark's splitting code for consistency:
  - CV1: Edge-based split (random split of SL pairs)
  - CV2: Gene-based split (hold out genes, test pairs have ≥1 unseen gene)
  - CV3: Pair-based split (both genes in test pairs are unseen)
"""

import sys
from pathlib import Path

# Add SLMGAE code directory to path for importing benchmark code
SLMGAE_CODE_DIR = Path(__file__).parent.parent / "code"
if str(SLMGAE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SLMGAE_CODE_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from gene_name_utils import get_mapper


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
    # Initial estimate: median (robust)
    mu = torch.median(x, dim=0).values  # (d,)

    for _ in range(max_iter):
        r = x - mu.unsqueeze(0)  # (n, d) residuals
        # MAD-based scale estimate per dimension
        scale = 1.4826 * torch.median(torch.abs(r), dim=0).values  # (d,)
        # Avoid division by zero for constant columns
        scale = torch.clamp(scale, min=tol)
        # Huber weights: 1 for |r| <= c*scale, c*scale/|r| for |r| > c*scale
        threshold = c * scale.unsqueeze(0)  # (1, d)
        w = torch.clamp(threshold / (torch.abs(r) + tol), max=1.0)  # (n, d)
        # Weighted mean
        mu_new = (w * x).sum(dim=0) / w.sum(dim=0)  # (d,)
        if (mu_new - mu).abs().max() < tol:
            return mu_new
        mu = mu_new

    return mu


def _robpca(embeddings: torch.Tensor,
            n_components: int,
            alpha: float = 0.75) -> torch.Tensor:
    """Robust PCA via robpy's ROBPCA (Hubert, Rousseeuw & Vanden Branden, 2005).

    Uses projection pursuit + MCD on scores to find robust PC directions.
    This is the gold-standard robust PCA implementation from the original
    authors (University of Antwerp).

    The algorithm:
      1. Classical PCA to reduce to rank-n subspace
      2. Stahel-Donoho outlyingness to identify h least-outlying genes
      3. Eigendecompose h-subset covariance → robust PC directions
      4. Orthogonal distance filter → keep genes with small OD
      5. FastMCD on scores for final robust eigenvectors
      6. Project: Z = X @ V (V computed from centered data; projection is linear)

    Args:
        embeddings: (n_genes, dim) tensor with no NaN.
        n_components: Number of components to keep.
        alpha: Coverage parameter in [0.5, 1.0]. Fraction of genes assumed
               clean. Lower = more robust but less efficient. Default 0.75
               tolerates up to 25% outlier genes.

    Returns:
        (n_genes, n_components) projected tensor with location preserved.
    """
    from robpy.pca import ROBPCA

    orig_dim = embeddings.shape[1]
    X_np = embeddings.numpy().astype(np.float64)

    pca = ROBPCA(n_components=n_components, alpha=alpha, random_seed=42)
    pca.fit(X_np)

    # robpy's ROBPCA stores components_ as (features, n_components) — columns are
    # eigenvectors (unlike sklearn's (n_components, features) convention).
    V = torch.from_numpy(pca.components_).float()  # (p, n_components)
    projected = embeddings @ V  # (n, k)

    # Variance explained
    evals = pca.explained_variance_
    total_var = float(np.var(X_np, axis=0, ddof=1).sum())
    retained_var = float(np.sum(evals))
    pct = (retained_var / total_var * 100) if total_var > 0 else 0.0

    print(f"    ROBPCA {orig_dim}d -> {n_components}d "
          f"({pct:.1f}% variance, alpha={alpha})")

    return projected


def _robpca_fallback(embeddings: torch.Tensor,
                     n_components: int,
                     c: float = 1.345,
                     max_iter: int = 20,
                     tol: float = 1e-4) -> torch.Tensor:
    """Fallback robust PCA via iterative Huber M-estimator (low-rank).

    Used when robpy is not available (Python < 3.10). Implements a low-rank
    approximation of the Huber M-estimator of scatter with iterative
    Mahalanobis reweighting, following the same spirit as ROBPCA: compute
    Mahalanobis distances in the current PC subspace rather than inverting
    the full p×p covariance.

    At each iteration:
      1. Project centered data into current PC subspace (n_components dims)
      2. Compute Mahalanobis distance per gene using PC eigenvalues
      3. Assign Huber weights: w_i = min(1, c·MAD(d) / d_i)
      4. Re-estimate PCs via weighted SVD: SVD(diag(sqrt(w)) @ centered)

    Args:
        embeddings: (n_genes, dim) tensor with no NaN.
        n_components: Number of components to keep. No-op if >= current dim.
        c: Huber tuning constant (1.345 = 95% efficiency at normal).
        max_iter: Maximum iterations for the M-estimator loop.
        tol: Convergence tolerance on relative change in singular values.

    Returns:
        (n_genes, n_components) projected tensor with location preserved.
    """
    orig_dim = embeddings.shape[1]
    n_genes = embeddings.shape[0]

    # Robust center via Huber M-estimate
    center = _huber_location(embeddings)
    centered = embeddings - center.unsqueeze(0)

    # Initial SVD (unweighted) to bootstrap Mahalanobis distances
    U, S, V = torch.pca_lowrank(centered, q=n_components, center=False)
    S_prev = S.clone()
    weighted = centered  # fallback if max_iter=0
    n_down = 0
    iteration = -1  # will be incremented in loop

    # Iterative Huber M-estimator of scatter (low-rank Mahalanobis)
    for iteration in range(max_iter):
        scores = centered @ V
        eig_vals = torch.clamp((S**2) / n_genes, min=1e-12)
        mahal = torch.sqrt((scores**2 / eig_vals.unsqueeze(0)).sum(dim=1))

        mad_mahal = torch.clamp(1.4826 * torch.median(mahal), min=1e-8)
        threshold = c * mad_mahal
        weights = torch.where(mahal <= threshold, torch.ones_like(mahal),
                              threshold / mahal)
        n_down = (weights < 1.0).sum().item()

        weighted = centered * weights.unsqueeze(1).sqrt()
        U, S, V = torch.pca_lowrank(weighted, q=n_components, center=False)

        rel_change = (S - S_prev).abs().sum() / (S_prev.abs().sum() + 1e-12)
        S_prev = S.clone()
        if rel_change < tol:
            break

    # V was computed from centered data (correct). Project original X @ V.
    projected = embeddings @ V

    total_ss = (weighted**2).sum()
    retained_ss = (S**2).sum()
    pct = (retained_ss / total_ss * 100).item()

    print(f"    Robust PCA (fallback) {orig_dim}d -> {n_components}d "
          f"({pct:.1f}% variance, {n_down} genes downweighted, "
          f"{iteration + 1} iterations)")

    return projected


def apply_pca(embeddings: torch.Tensor,
              n_components: int,
              alpha: float = 0.75) -> torch.Tensor:
    """Robust PCA with location preservation.

    Applied per modality AFTER imputation and BEFORE normalization so that
    PCA sees the natural variance structure of complete data.

    Primary: robpy's ROBPCA (Hubert, Rousseeuw & Vanden Branden, 2005) —
    projection pursuit + MCD on scores. This is the gold-standard robust
    PCA from the original authors.

    Fallback: iterative Huber M-estimator of scatter with low-rank
    Mahalanobis reweighting (used when robpy is not installed, e.g.,
    Python < 3.10).

    Both methods compute V from centered data (correct for PCA), then
    project as Z = X @ V (a linear rotation). Feature scales
    are NOT equalized (no per-column standardization) — within-modality
    scale differences are preserved, consistent with the normalization
    design.

    Args:
        embeddings: (n_genes, dim) tensor with no NaN.
        n_components: Number of components to keep. No-op if >= current dim.
        alpha: ROBPCA coverage parameter in [0.5, 1.0]. Fraction of genes
               assumed clean. Lower = more robust. Default 0.75 tolerates
               up to 25% outlier genes.

    Returns:
        (n_genes, n_components) projected tensor with location preserved.
    """
    orig_dim = embeddings.shape[1]
    if n_components >= orig_dim:
        return embeddings

    try:
        return _robpca(embeddings, n_components, alpha=alpha)
    except ImportError:
        print("    [robpy not available, using fallback Huber M-estimator]")
        return _robpca_fallback(embeddings, n_components)


def normalize_modality(embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize a single modality's embedding matrix.

    1. Center each feature (column) by its median — removes DC offset.
       The learnable input bias in the encoder recovers per-feature location.
    2. Scale entire matrix by its global MAD — robust scale equalization
       across modalities when concatenated.

    Within-modality feature scale differences are preserved (no per-column
    variance scaling). The L1 penalty on the first encoder layer handles
    feature importance.

    MUST be called AFTER imputation (no NaN allowed).

    Args:
        embeddings: (num_genes, dim) tensor with no NaN.

    Returns:
        Normalized tensor (centered per column, globally MAD-scaled).
    """
    # Center per feature (column median)
    col_medians = torch.median(embeddings, dim=0).values  # (dim,)
    centered = embeddings - col_medians.unsqueeze(0)

    # Scale by global MAD (robust Frobenius-like normalization)
    # 1.4826 makes MAD a consistent estimator of std at the normal distribution
    global_mad = 1.4826 * torch.median(torch.abs(centered))
    global_mad = torch.clamp(global_mad, min=1e-8)

    return centered / global_mad


def impute_nan_with_huber(embeddings: torch.Tensor,
                          c: float = 1.345) -> torch.Tensor:
    """Replace NaN values with per-dimension Huber M-estimate across non-NaN genes.

    Huber's M-estimator is a robust alternative to the mean that downweights
    outlier genes when computing the imputation value. With c=1.345, it
    achieves 95% efficiency at the normal distribution while resisting
    outlier contamination.

    Args:
        embeddings: (num_genes, dim) tensor, may contain NaN for missing genes.
        c: Huber tuning constant. 1.345 = 95% normal efficiency.

    Returns:
        Imputed tensor with no NaN values.
    """
    if not torch.isnan(embeddings).any():
        return embeddings

    nan_mask = torch.isnan(embeddings)

    # Compute Huber location per dimension using non-NaN values
    # For columns with mixed NaN/valid, extract valid values per column
    dim = embeddings.shape[1]
    col_locations = torch.zeros(dim)

    # Check if all genes are missing for any dimension
    valid_per_col = (~nan_mask).sum(dim=0)  # (dim,)

    if (valid_per_col == 0).any():
        # Some columns are entirely NaN — fill with 0
        all_nan_cols = valid_per_col == 0
        print(f"  Warning: {all_nan_cols.sum().item()} dimensions have no "
              f"valid values, filling with 0")

    # For efficiency, compute Huber on all valid rows if missingness is row-level
    # (i.e., entire rows are NaN, not individual cells)
    row_all_nan = nan_mask.all(dim=1)
    row_all_valid = ~nan_mask.any(dim=1)

    if row_all_nan.sum() + row_all_valid.sum() == embeddings.shape[0]:
        # Clean case: NaN is row-level (missing genes have ALL dims NaN)
        valid_rows = embeddings[row_all_valid]  # (n_valid, dim)
        if valid_rows.shape[0] > 0:
            col_locations = _huber_location(valid_rows, c=c)
        else:
            col_locations = torch.zeros(dim)
    else:
        # Mixed case: some genes have partial NaN (e.g., concatenated embeddings)
        for d in range(dim):
            col = embeddings[:, d]
            valid = col[~torch.isnan(col)]
            if len(valid) > 0:
                col_locations[d] = _huber_location(valid.unsqueeze(1),
                                                   c=c).squeeze()
            else:
                col_locations[d] = 0.0

    embeddings = torch.where(nan_mask, col_locations.unsqueeze(0), embeddings)

    n_imputed = nan_mask.any(dim=1).sum().item()
    n_total = embeddings.shape[0]
    print(f"  Huber-imputed {n_imputed}/{n_total} genes "
          f"({nan_mask.sum().item()} NaN values, c={c})")

    return embeddings


def load_single_embedding(
    path: str,
    gene_list_path: Optional[str] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Load a single .pt embedding file and return (embeddings, gene_order).

    Gene identifiers in gene_order are normalized gene symbols (custom
    corrections + HGNC current symbol resolution). Deduplicates any
    collisions after normalization.

    Returns:
        (embeddings, gene_order) where gene_order is a list of normalized
        gene symbol strings. Embeddings may contain NaN for missing genes.
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

    # Normalize gene names (custom corrections + HGNC)
    mapper = get_mapper()
    gene_order = mapper.gene_name_normalize_list(gene_order)

    # Deduplicate after normalization
    seen = {}
    dups = []
    keep = []
    for i, g in enumerate(gene_order):
        if g in seen:
            dups.append((g, seen[g], i))
        else:
            seen[g] = i
            keep.append(i)
    if dups:
        examples = "; ".join(f"{g} at rows {a},{b}" for g, a, b in dups[:5])
        print(f"  Warning: {len(dups)} duplicate gene(s) after name "
              f"normalization, keeping first: {examples}")
        embeddings = embeddings[keep]
        gene_order = [gene_order[i] for i in keep]

    return embeddings, gene_order


def load_multimodal_embeddings(
    paths: List[str],
    gene_list_path: Optional[str] = None,
    pca_dims: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """Load one or more embedding files, impute, normalize, and concatenate.

    This is the sole entry point for all embedding loading — single and
    multi-modal alike. A single file is treated as a one-element list.

    Pipeline per modality:
      1. Load .pt file, normalize gene names, deduplicate
      2. Verify gene order matches across all modalities
      3. Impute NaN via Huber M-estimates (per modality, on raw values)
      4. Robust PCA (optional): reduce to per-modality target dimensions
      5. Normalize: center per column median, scale by global MAD
    Then concatenate along feature axis.

    Args:
        paths: List of .pt embedding file paths (one or more).
        gene_list_path: Optional gene list for raw tensor files.
        pca_dims: Per-modality PCA target dimensions (one per path).
            Each modality is reduced to its corresponding value.
            Values >= original dim are no-ops. None to skip PCA.

    Returns:
        (concatenated_embeddings, gene_to_idx, idx_to_gene)
    """
    if pca_dims is not None and len(pca_dims) != len(paths):
        raise ValueError(f"--pca_dims has {len(pca_dims)} values but "
                         f"{len(paths)} embedding files provided")
    n = len(paths)
    desc = "modality" if n == 1 else f"{n} modalities"
    print(f"\nLoading {desc} (impute → PCA → normalize → concat):")

    all_normalized = []
    canonical_gene_order = None
    total_dim = 0

    for i, path in enumerate(paths):
        emb, gene_order = load_single_embedding(path, gene_list_path)
        dim = emb.shape[1]
        label = Path(path).stem.replace("all_genes_", "")

        # Verify gene order matches across modalities
        if canonical_gene_order is None:
            canonical_gene_order = gene_order
        else:
            if gene_order != canonical_gene_order:
                # Find first mismatches for error message
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

        # Pipeline: impute → robust PCA (optional) → normalize
        n_nan = torch.isnan(emb).any(dim=1).sum().item()
        print(f"  [{label}] {dim}d, "
              f"{emb.shape[0]} genes, {n_nan} missing")
        emb = impute_nan_with_huber(emb)
        if pca_dims is not None:
            emb = apply_pca(emb, pca_dims[i])
        emb = normalize_modality(emb)

        all_normalized.append(emb)
        total_dim += emb.shape[1]

    concatenated = torch.cat(all_normalized, dim=1)
    print(f"  -> Concatenated: {total_dim}d "
          f"({concatenated.shape[0]} genes)")

    gene_to_idx = {gene: idx for idx, gene in enumerate(canonical_gene_order)}
    idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}

    return concatenated, gene_to_idx, idx_to_gene


# Import SLMGAE benchmark's data splitting code
from data_split import SLDataSplitter

# Single set_seed definition lives in siamese_esm.py
from siamese_esm import set_seed


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
    ):
        """
        Args:
            embeddings: Gene embeddings tensor (num_genes, embed_dim)
            pairs: Gene index pairs (N, 2)
            labels: Labels for each pair (N,)
        """
        self.embeddings = embeddings
        self.pairs = pairs
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
            self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j = self.pairs[idx]
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

    NOTE: The --seed argument controls both model weight initialization
    AND CV splitting (passed as random_state to SLDataSplitter).

    Gene universe:
      - All genes with ESM embeddings (~20k)
      - SL labels only for ~6k genes from SLMGAE dataset
      - Negatives sampled from non-SL pairs by benchmark code
    """

    def __init__(
        self,
        embeddings_paths: List[str],
        sl_pairs_path: str = "",
        gene_list_path: Optional[str] = None,
        seed: int = 42,
        pca_dims: Optional[List[int]] = None,
    ):
        """
        Args:
            embeddings_paths: List of .pt embedding files. Even a single file
                goes through the full pipeline (impute → PCA → normalize) for
                consistency.
            sl_pairs_path: Path to SL pairs file (gene1 TAB gene2 [TAB weight])
            gene_list_path: Optional path to gene list (for ordering)
            seed: Random seed for reproducibility
            pca_dims: Per-modality PCA target dimensions (one per embedding
                file). Applied after imputation, before normalization.
        """
        if not embeddings_paths:
            raise ValueError("embeddings_paths must be a non-empty list")

        self.seed = seed
        set_seed(seed)

        # Unified pipeline: impute → robust PCA (optional) → normalize → concat
        self.embeddings, self.gene_to_idx, self.idx_to_gene = \
            load_multimodal_embeddings(embeddings_paths, gene_list_path,
                                       pca_dims)
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

        # Create negative pairs for benchmark splitter
        # Benchmark code samples negatives during splitting
        self.neg_pairs = np.zeros((0, 2), dtype=np.int64)

        print(f"Loaded {len(self.embeddings)} gene embeddings")
        print(f"Loaded {len(self.pos_pairs_full_idx)} positive SL pairs")
        print(f"SL pairs involve {self.num_sl_genes} unique genes")

    def _load_sl_pairs(self,
                       sl_pairs_path: str) -> Tuple[np.ndarray, Set[str]]:
        """
        Load positive SL pairs from file.

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
                    g1, g2 = mapper.gene_name_normalize(
                        parts[0]), mapper.gene_name_normalize(parts[1])

                    # Skip if gene not in embeddings
                    if g1 not in self.gene_to_idx or g2 not in self.gene_to_idx:
                        continue

                    # Consistent ordering (smaller index first)
                    idx1, idx2 = self.gene_to_idx[g1], self.gene_to_idx[g2]
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                        g1, g2 = g2, g1

                    pair_key = (idx1, idx2)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append([idx1, idx2])
                        sl_genes.add(g1)
                        sl_genes.add(g2)

        return np.array(pairs, dtype=np.int64), sl_genes

    def _create_sl_index_mapping(self) -> None:
        """
        Create bidirectional mapping between full embedding indices and SL-only indices.

        Full indices: 0 to num_embeddings-1 (e.g., 0-20000)
        SL indices: 0 to num_sl_genes-1 (e.g., 0-6000)

        This is needed because SLDataSplitter expects indices in [0, num_sl_genes).
        """
        # Get unique gene indices involved in SL pairs
        unique_full_indices = set()
        for g1, g2 in self.pos_pairs_full_idx:
            unique_full_indices.add(g1)
            unique_full_indices.add(g2)

        # Sort for deterministic mapping
        unique_full_indices = sorted(list(unique_full_indices))

        # Create mappings
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
            pos_edges=self.
            pos_pairs_sl_idx,  # Use SL-only indices [0, num_sl_genes)
            neg_edges=self.neg_pairs,
            num_nodes=self.num_sl_genes,
            train_ratio=0.8,
            random_state=self.seed,
        )

    def _convert_split_to_dataset_format(
        self,
        split: Dict,
    ) -> Dict:
        """
        Convert SLMGAE benchmark split format to our dataset format.

        Benchmark format:
            train_edges, train_labels, test_edges, test_labels (using SL-only indices)

        Our format:
            train/test dicts with pairs using FULL embedding indices
        """
        # Convert SL-only indices back to full embedding indices
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
        print(
            f"\nGenerating CV1 splits (edge-based) using SLMGAE benchmark code..."
        )
        splitter = self._create_splitter()
        splits = splitter.cv1_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold'] + 1}: "
                  f"train={split['num_train_pos']}+/{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/{split['num_test_neg']}-")

        return converted

    def get_cv2_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV2: Gene-based cross-validation (from SLMGAE benchmark).

        Holds out entire genes. Test pairs have at least one unseen gene.
        Tests ability to generalize to partially new genes.
        """
        print(
            f"\nGenerating CV2 splits (gene-based) using SLMGAE benchmark code..."
        )
        splitter = self._create_splitter()
        splits = splitter.cv2_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold'] + 1}: "
                  f"train={split['num_train_pos']}+/{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/{split['num_test_neg']}-")

        return converted

    def get_cv3_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV3: Pair-based cross-validation (from SLMGAE benchmark).

        Both genes in test pairs are unseen during training.
        Tests ability to generalize to completely novel gene pairs.
        This is the hardest setting.
        """
        print(
            f"\nGenerating CV3 splits (pair-based) using SLMGAE benchmark code..."
        )
        splitter = self._create_splitter()
        splits = splitter.cv3_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold'] + 1}: "
                  f"train={split['num_train_pos']}+/{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/{split['num_test_neg']}-")

        return converted


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
    )
    test_dataset = SLDataset(
        embeddings=embeddings,
        pairs=fold_data["test"]["pairs"],
        labels=fold_data["test"]["labels"],
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
