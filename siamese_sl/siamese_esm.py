#!/usr/bin/env python3
"""
Siamese network for Synthetic Lethality prediction using gene embeddings.

Architecture:
  - Input: Two gene embeddings (e.g. 200-dim GO-only, 1280-dim ESM, or 1480-dim ESM+GO)
  - Encoder: Shared MLP projecting embeddings to latent space
  - Predictor: Combines encoded representations to predict SL probability

This implementation:
  - Uses gene embeddings as the ONLY node features (no graph structure needed)
  - Works with any gene that has embeddings (not limited to SL dataset genes)
  - Is fully reproducible with fixed random seeds
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Efficient MLP Encoder Variants
# =============================================================================


class LowRankLinear(nn.Module):
    """
    Low-rank factorized linear layer: W ≈ UV where U ∈ ℝ^{out×r}, V ∈ ℝ^{r×in}

    Parameters: O(r(m+n)) instead of O(mn)

    Reference: "Low-Rank Matrix Approximation for Neural Network Compression"
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int,
                 bias: bool = True):
        super().__init__()
        self.rank = rank
        # Factorize: W = U @ V where W is (out, in)
        self.U = nn.Linear(rank, out_features, bias=False)
        self.V = nn.Linear(in_features, rank, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize for stable training
        nn.init.kaiming_normal_(self.V.weight)
        nn.init.kaiming_normal_(self.U.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ Vᵀ @ Uᵀ + b = x @ (UV)ᵀ + b
        out = self.U(self.V(x))
        if self.bias is not None:
            out = out + self.bias
        return out

    @property
    def num_params(self) -> tuple:
        """Number of parameters (low_rank, full) for comparison."""
        in_f = self.V.in_features
        out_f = self.U.out_features
        full = in_f * out_f
        low_rank = self.rank * (in_f + out_f)
        return low_rank, full


class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP: in_dim → bottleneck → out_dim

    Forces information compression through narrow bottleneck.
    Equivalent to learning a low-rank transformation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU): split input, one half gates the other.

    out = (Wx + b) ⊙ σ(Vx + c)

    Used in gMLP and efficient transformers.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class EfficientEncoder(nn.Module):
    """
    Efficient MLP encoder with configurable architecture.

    Supports:
    - 'standard': Regular MLP
    - 'lowrank': Low-rank factorized layers
    - 'bottleneck': Bottleneck architecture
    - 'gated': Gated Linear Units (GLU)

    Args:
        input_dim: Input dimension (e.g., 200 for GO-only, 1280 for ESM, or 1480 for ESM+GO)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        encoder_type: Architecture type
        rank: Rank for low-rank factorization
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        output_dim: int = 256,
        encoder_type: str = 'standard',
        rank: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'standard':
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

        elif encoder_type == 'lowrank':
            # Low-rank factorized layers
            self.net = nn.Sequential(
                LowRankLinear(input_dim, hidden_dim, rank=rank),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                LowRankLinear(hidden_dim, output_dim, rank=rank),
                nn.LayerNorm(output_dim),
            )

        elif encoder_type == 'bottleneck':
            # Two-stage bottleneck
            bottleneck1 = max(rank, input_dim // 8)
            bottleneck2 = max(rank // 2, output_dim // 4)
            self.net = nn.Sequential(
                BottleneckMLP(input_dim, hidden_dim, bottleneck1, dropout),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                BottleneckMLP(hidden_dim, output_dim, bottleneck2, dropout),
                nn.LayerNorm(output_dim),
            )

        elif encoder_type == 'gated':
            # Gated Linear Units with reduced intermediate dim
            gated_hidden = hidden_dim // 2  # Compensate for GLU doubling
            self.net = nn.Sequential(
                GatedLinearUnit(input_dim, gated_hidden),
                nn.LayerNorm(gated_hidden),
                nn.Dropout(dropout),
                GatedLinearUnit(gated_hidden, output_dim),
                nn.LayerNorm(output_dim),
            )

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SiameseEncoder(nn.Module):
    """
    Shared encoder that projects gene embeddings to a latent space.

    Architecture: input_dim -> hidden -> latent
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode gene embedding to latent representation."""
        return self.encoder(x)


class SiameseSL(nn.Module):
    """
    Siamese network for Synthetic Lethality prediction.

    Takes two gene embeddings and predicts their SL probability.

    Architecture:
      gene1_esm -+-> SharedEncoder -> z1 -+
                 |                         +-> Predictor -> P(SL)
      gene2_esm -+-> SharedEncoder -> z2 -+

    The predictor combines z1 and z2 using SYMMETRIC features only:
      - Element-wise sum: z1 + z2
      - Element-wise product: z1 * z2
      - Absolute difference: |z1 - z2|

    NOTE: We avoid concat([z1, z2]) because it's NOT symmetric.
    SL is a symmetric relationship: SL(A,B) = SL(B,A).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        predictor_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared encoder for both genes
        self.encoder = SiameseEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Predictor MLP: takes SYMMETRIC features only
        # Input: z1+z2, z1*z2, |z1-z2| = 3 * latent_dim
        # NOTE: concat([z1,z2]) is NOT symmetric, so we don't use it
        predictor_input_dim = latent_dim * 3

        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, predictor_hidden),
            nn.LayerNorm(predictor_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of gene embeddings."""
        return self.encoder(x)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict SL probability for gene pairs.

        Args:
            x1: Gene embeddings for first genes (batch_size, input_dim)
            x2: Gene embeddings for second genes (batch_size, input_dim)

        Returns:
            SL probability logits (batch_size, 1)
        """
        # Encode both genes with shared weights
        z1 = self.encoder(x1)  # (batch, latent_dim)
        z2 = self.encoder(x2)  # (batch, latent_dim)

        # Combine representations using ONLY symmetric features
        # These are order-invariant: f(z1, z2) = f(z2, z1)
        sum_emb = z1 + z2  # (batch, latent) - symmetric
        product = z1 * z2  # (batch, latent) - symmetric
        diff = torch.abs(z1 - z2)  # (batch, latent) - symmetric

        combined = torch.cat([sum_emb, product, diff],
                             dim=1)  # (batch, 3*latent)

        # Predict SL probability
        logits = self.predictor(combined)

        return logits

    def predict_proba(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Return SL probability (sigmoid applied)."""
        logits = self.forward(x1, x2)
        return torch.sigmoid(logits)


class SiameseSLWithAttention(nn.Module):
    """
    Enhanced Siamese network with cross-attention between gene pairs.

    Allows the model to learn which features of one gene are most relevant
    for predicting SL with the other gene.

    The predictor combines z1 and z2 using SYMMETRIC features only:
      - Element-wise sum: z1 + z2
      - Element-wise product: z1 * z2
      - Absolute difference: |z1 - z2|

    NOTE: We avoid concat([z1, z2]) because it's NOT symmetric.
    SL is a symmetric relationship: SL(A,B) = SL(B,A).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention projection
        self.post_attn = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
        )

        # Predictor: takes SYMMETRIC features only
        # Input: z1+z2, z1*z2, |z1-z2| = 3 * latent_dim
        # NOTE: concat([z1,z2]) is NOT symmetric, so we don't use it
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Predict SL with cross-attention."""
        # Project to hidden dim
        h1 = self.proj(x1).unsqueeze(1)  # (batch, 1, hidden)
        h2 = self.proj(x2).unsqueeze(1)  # (batch, 1, hidden)

        # Cross-attention: each gene attends to the other
        h1_attn, _ = self.cross_attn(h1, h2, h2)  # gene1 attends to gene2
        h2_attn, _ = self.cross_attn(h2, h1, h1)  # gene2 attends to gene1

        # Squeeze and project
        z1 = self.post_attn(h1_attn.squeeze(1))
        z2 = self.post_attn(h2_attn.squeeze(1))

        # Combine representations using ONLY symmetric features
        # These are order-invariant: f(z1, z2) = f(z2, z1)
        sum_emb = z1 + z2  # (batch, latent) - symmetric
        product = z1 * z2  # (batch, latent) - symmetric
        diff = torch.abs(z1 - z2)  # (batch, latent) - symmetric

        combined = torch.cat([sum_emb, product, diff],
                             dim=1)  # (batch, 3*latent)

        return self.predictor(combined)


class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for Gaussian kernel approximation.

    Approximates k(x,y) = exp(-||x-y||²/2σ²) using:
        k(x,y) ≈ z(x)ᵀz(y)
    where z(x) = √(2/D) * [cos(ω₁ᵀx + b₁), ..., cos(ω_Dᵀx + b_D)]

    Based on Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines"

    Args:
        input_dim: Dimension of input features
        num_features: Number of random Fourier features (D)
        sigma: Kernel bandwidth (learnable if learn_sigma=True)
        learn_sigma: Whether to learn the bandwidth parameter
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int = 256,
        sigma: float = 1.0,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features

        # Random frequencies ω ~ N(0, 1/σ²)
        # We sample from N(0, 1) and scale by 1/σ at forward time
        self.register_buffer(
            "omega", torch.randn(input_dim, num_features, dtype=torch.float32))
        # Random phase shifts b ~ Uniform(0, 2π)
        self.register_buffer(
            "bias",
            torch.rand(num_features, dtype=torch.float32) * 2 * np.pi)

        # Learnable bandwidth (log-scale for numerical stability)
        if learn_sigma:
            self.log_sigma = nn.Parameter(
                torch.tensor(np.log(sigma), dtype=torch.float32))
        else:
            self.register_buffer(
                "log_sigma", torch.tensor(np.log(sigma), dtype=torch.float32))

        # Scaling factor √(2/D)
        self.scale = np.sqrt(2.0 / num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input to random Fourier feature space.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            z: Random Fourier features (batch_size, num_features)
        """
        sigma = torch.exp(self.log_sigma)
        # Project: x @ (ω / σ) + b
        projection = x @ (self.omega / sigma) + self.bias
        # Apply cosine and scale
        return self.scale * torch.cos(projection)


class HilbertSpaceMap(nn.Module):
    """
    Learnable mapping to a finite-dimensional Hilbert space.

    Combines:
    1. Linear projection (learned basis in H)
    2. Random Fourier Features (Gaussian RKHS approximation)

    The output φ(x) lives in H where inner products ⟨φ(x), φ(y)⟩
    approximate kernel similarity.

    Args:
        input_dim: Input feature dimension
        hilbert_dim: Dimension of the Hilbert space approximation
        rff_dim: Dimension of RFF component (set to 0 to disable)
        sigma: RBF kernel bandwidth for RFF
    """

    def __init__(
        self,
        input_dim: int,
        hilbert_dim: int = 256,
        rff_dim: int = 128,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.hilbert_dim = hilbert_dim
        self.rff_dim = rff_dim
        self.total_dim = hilbert_dim + rff_dim

        # Learnable linear map to Hilbert space (Mahalanobis-like)
        # This learns a basis where inner products are meaningful
        self.linear_map = nn.Linear(input_dim, hilbert_dim, bias=False)
        nn.init.orthogonal_(
            self.linear_map.weight)  # Start with orthonormal basis

        # RFF for Gaussian RKHS component
        if rff_dim > 0:
            self.rff = RandomFourierFeatures(
                input_dim=input_dim,
                num_features=rff_dim,
                sigma=sigma,
                learn_sigma=True,
            )
        else:
            self.rff = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input to Hilbert space.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            φ(x): Hilbert space representation (batch, hilbert_dim + rff_dim)
        """
        # Linear component: learned basis
        h_linear = self.linear_map(x)  # (batch, hilbert_dim)

        if self.rff is not None:
            # RFF component: Gaussian RKHS approximation
            h_rff = self.rff(x)  # (batch, rff_dim)
            return torch.cat([h_linear, h_rff], dim=-1)
        else:
            return h_linear


class SiameseSLKernel(nn.Module):
    """
    RKHS-based Siamese network for Synthetic Lethality prediction.

    This architecture explicitly maps embeddings to a Hilbert space H,
    then computes similarity as inner products in H.

    Pipeline:
        x ──> Encoder ──> z ──> φ(z) ∈ H ──> ⟨φ(z1), φ(z2)⟩ ──> P(SL)

    The Hilbert space mapping φ combines:
    1. Learned linear projection (Mahalanobis-like metric)
    2. Random Fourier Features (Gaussian kernel approximation)

    The kernel between two genes is: k(z1, z2) = ⟨φ(z1), φ(z2)⟩_H

    References:
    - Rahimi & Recht (2007): Random Features for Large-Scale Kernel Machines
    - arXiv:2508.04476: Metric Learning in an RKHS

    Args:
        input_dim: Gene embedding dimension (e.g. 200 GO-only, 1280 ESM, or 1480 ESM+GO)
        hidden_dim: Encoder hidden dimension
        latent_dim: Latent space dimension (before Hilbert mapping)
        rff_features: RFF dimension (Gaussian RKHS component)
        bilinear_rank: Hilbert space linear projection dimension
        predictor_hidden: Predictor hidden dimension
        dropout: Dropout rate
        sigma: Initial RBF kernel bandwidth
        encoder_type: Encoder architecture ('standard', 'lowrank', 'bottleneck', 'gated')
        encoder_rank: Rank for lowrank encoder factorization
    """

    def __init__(
            self,
            input_dim: int = 1280,
            hidden_dim: int = 512,
            latent_dim: int = 256,
            rff_features: int = 128,  # RFF dimension
            bilinear_rank: int = 128,  # Hilbert linear projection dim
            predictor_hidden: int = 128,
            dropout: float = 0.2,
            sigma: float = 1.0,
            encoder_type:
        str = 'standard',  # 'standard', 'lowrank', 'bottleneck', 'gated'
            encoder_rank: int = 64,  # Rank for lowrank encoder
    ):
        super().__init__()

        self.latent_dim = latent_dim
        hilbert_dim = bilinear_rank
        rff_dim = rff_features

        # Step 1: Shared encoder (input → latent) - configurable efficiency
        self.encoder = EfficientEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            encoder_type=encoder_type,
            rank=encoder_rank,
            dropout=dropout,
        )

        # Step 2: Map to Hilbert space (latent → H)
        self.hilbert_map = HilbertSpaceMap(
            input_dim=latent_dim,
            hilbert_dim=hilbert_dim,
            rff_dim=rff_dim,
            sigma=sigma,
        )

        # Step 3: Predictor from SYMMETRIC Hilbert space features
        # Same symmetric aggregation as SiameseSL: [sum, product, abs_diff]
        total_hilbert_dim = hilbert_dim + rff_dim
        self.predictor = nn.Sequential(
            nn.Linear(total_hilbert_dim * 3, predictor_hidden),
            nn.LayerNorm(predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode gene embedding to latent space."""
        return self.encoder(x)

    def map_to_hilbert(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent representation to Hilbert space."""
        return self.hilbert_map(z)

    def kernel(self, phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel as inner product in Hilbert space.

        k(x, y) = ⟨φ(x), φ(y)⟩_H

        Args:
            phi1: First Hilbert space vectors (batch, H_dim)
            phi2: Second Hilbert space vectors (batch, H_dim)

        Returns:
            Kernel values (batch,)
        """
        return (phi1 * phi2).sum(dim=-1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict SL probability for gene pairs.

        Args:
            x1: Gene embeddings for first genes (batch, input_dim)
            x2: Gene embeddings for second genes (batch, input_dim)

        Returns:
            SL probability logits (batch, 1)
        """
        # Step 1: Encode to latent space
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # Step 2: Map to Hilbert space
        phi1 = self.hilbert_map(z1)  # (batch, H_dim)
        phi2 = self.hilbert_map(z2)  # (batch, H_dim)

        # Step 3: Symmetric features in Hilbert space
        sum_emb = phi1 + phi2
        product = phi1 * phi2
        diff = torch.abs(phi1 - phi2)
        combined = torch.cat([sum_emb, product, diff], dim=1)

        # Step 4: Predict
        return self.predictor(combined)

    def predict_proba(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Return SL probability (sigmoid applied)."""
        return torch.sigmoid(self.forward(x1, x2))

    def get_kernel_bandwidth(self) -> float:
        """Return current learned kernel bandwidth."""
        if self.hilbert_map.rff is not None:
            return torch.exp(self.hilbert_map.rff.log_sigma).item()
        return float('nan')


def get_model(model_type: str = "siamese", **kwargs) -> nn.Module:
    """
    Factory function to create SL prediction models.

    Args:
        model_type: "siamese", "attention", or "kernel"
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    if model_type == "siamese":
        return SiameseSL(**kwargs)
    elif model_type == "attention":
        return SiameseSLWithAttention(**kwargs)
    elif model_type == "kernel":
        return SiameseSLKernel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
