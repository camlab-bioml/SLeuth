#!/usr/bin/env python3
"""
Siamese network for Synthetic Lethality prediction using gene embeddings.

Architecture:
  - Input: Two gene embeddings (any supported type or multi-modal concatenation)
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


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

    A learnable per-feature input bias is added before the first layer to
    compensate for location information lost during normalization (global
    median centering).

    Supports:
    - 'standard': Regular MLP
    - 'lowrank': Low-rank factorized layers
    - 'bottleneck': Bottleneck architecture
    - 'gated': Gated Linear Units (GLU)

    Args:
        input_dim: Input dimension (auto-detected from loaded embeddings)
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

        # Learnable per-feature input bias.
        self.input_bias = nn.Parameter(torch.zeros(input_dim))

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
        return self.net(x + self.input_bias)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SiameseEncoder(nn.Module):
    """
    Shared encoder that projects gene embeddings to a latent space.

    Architecture is defined by a list of layer dimensions. The last entry is
    the projection (latent) dim; everything before it is a hidden layer.
    Example: encoder_dims=[256, 128, 64] builds:
        input + bias
          -> Linear(256)->LN->LReLU->Drop
          -> Linear(128)->LN->LReLU->Drop
          -> Linear(64)                  # projection, bare (latent dim 64)
    (The current SLURM default ENCODER_DIMS targets the residual variant —
    see `SiameseEncoderResidual` below.)

    A learnable per-feature input bias is added before the first layer to
    compensate for location information lost during normalization (global
    median centering). The first Linear has bias=False and LayerNorm
    re-centers activations, so without this input bias the per-feature
    DC offset is irrecoverable.

    Hidden layers: Linear -> LayerNorm -> LeakyReLU -> Dropout.
    Last layer: bare Linear projection (Xavier init, no activation).

    The encoder is split into hidden layers and projection so that
    both h (hidden output) and z (projection output) are accessible.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list,
        dropout: float = 0.2,
        last_layer_bias: bool = True,
    ):
        super().__init__()

        if not encoder_dims:
            raise ValueError("encoder_dims must be non-empty")

        # Learnable per-feature input bias.
        # Initialized to zero (no shift); the model learns the optimal
        # per-feature offset during training.
        self.input_bias = nn.Parameter(torch.zeros(input_dim))

        # Hidden layers (all except last)
        # bias=False because LayerNorm has its own learnable shift (beta)
        hidden_layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(encoder_dims[:-1]):
            hidden_layers.append(nn.Linear(in_dim, out_dim, bias=False))
            hidden_layers.append(nn.LayerNorm(out_dim))
            hidden_layers.append(nn.LeakyReLU(0.2))
            hidden_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.hidden = nn.Sequential(*hidden_layers)

        # Projection layer (bare linear, Xavier init)
        proj_in = encoder_dims[-2] if len(encoder_dims) > 1 else input_dim
        self.projection = nn.Linear(proj_in,
                                    encoder_dims[-1],
                                    bias=last_layer_bias)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode gene embedding to projection output z."""
        h = self.hidden(x + self.input_bias)
        return self.projection(h)

    def forward_with_hidden(self, x: torch.Tensor):
        """Return both projection output z and hidden output h."""
        h = self.hidden(x + self.input_bias)
        z = self.projection(h)
        return z, h


class SiameseEncoderResidual(nn.Module):
    """Residual-MLP variant of SiameseEncoder.

    Same interface as SiameseEncoder (input_dim, encoder_dims, dropout,
    last_layer_bias). Each hidden block is
        F(x) = Dropout(LeakyReLU(LayerNorm(Linear(x))))
    wrapped with a residual connection y = x + F(x) WHEN the block's input
    and output widths match. Blocks where widths differ (typically the first
    "entry lift" from input_dim and any dim-change in the middle) are plain
    F-applies with no skip.

    The projection (last entry of encoder_dims) is always a bare Linear
    with no skip — the existing `pd_epsilon · h1ᵀh2` term in SiameseSL's
    scoring already acts as a score-level residual around the projection.

    Example (encoder_dims=[16, 16, 16, 16, 16, 16, 16], the live SLURM default;
    D is the post-preprocessing input width plus CELL_LINE_DIM):
        input(Dd) + input_bias
          -> Linear(D→16) + LN + LReLU + Drop               # entry, no skip
          -> x + (Linear(16→16) + LN + LReLU + Drop)        # residual block 1
          -> x + (Linear(16→16) + LN + LReLU + Drop)        # residual block 2
          -> x + (Linear(16→16) + LN + LReLU + Drop)        # residual block 3
          -> x + (Linear(16→16) + LN + LReLU + Drop)        # residual block 4
          -> x + (Linear(16→16) + LN + LReLU + Drop)        # residual block 5
          -> Linear(16→16)                                   # projection, bare

    Rationale: depth alone (Telgarsky 2016; Raghu et al. 2017) gives
    exponential expressivity gains over width, but plain deep narrow ReLU
    MLPs suffer rank-collapse / gradient-path pathologies (Pennington et al.
    2017; Hanin & Rolnick 2019). Residual connections preserve a
    gradient-free path from loss to input and keep the activation's rank
    from collapsing layer-by-layer. LayerNorm inside F(x) keeps the block
    output's magnitude comparable to the skip branch at init.

    Implementation notes:
      - `residual_skips` is a Python list of bools derived from encoder_dims
        at __init__. It's NOT a Parameter/Buffer — the structure is fully
        determined by encoder_dims and is reconstructed in load_model at
        inference.
      - The skip uses pre-activation `x` and post-activation `F(x)` — a
        simplified single-Linear ResNet block. For 5–6 blocks this variant
        trains cleanly without explicit scaling; deeper stacks (>10 blocks)
        would benefit from additional tricks (layer-scale init, stochastic
        depth) that aren't implemented here.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list,
        dropout: float = 0.2,
        last_layer_bias: bool = True,
    ):
        super().__init__()

        if not encoder_dims:
            raise ValueError("encoder_dims must be non-empty")

        self.input_bias = nn.Parameter(torch.zeros(input_dim))

        hidden_widths = encoder_dims[:-1]
        blocks = []
        skips = []
        in_dim = input_dim
        for out_dim in hidden_widths:
            blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=False),
                    nn.LayerNorm(out_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ))
            # Skip only when dims match; the entry lift (input_dim != first
            # hidden width) and any mid-stack dim changes get plain blocks.
            skips.append(in_dim == out_dim)
            in_dim = out_dim
        self.hidden_blocks = nn.ModuleList(blocks)
        self.residual_skips = skips  # structural info; derived from encoder_dims

        proj_in = hidden_widths[-1] if hidden_widths else input_dim
        self.projection = nn.Linear(proj_in,
                                    encoder_dims[-1],
                                    bias=last_layer_bias)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.input_bias
        for block, skip in zip(self.hidden_blocks, self.residual_skips):
            out = block(h)
            h = out + h if skip else out
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._forward_hidden(x)
        return self.projection(h)

    def forward_with_hidden(self, x: torch.Tensor):
        h = self._forward_hidden(x)
        return self.projection(h), h


class SiameseSL(nn.Module):
    """
    Siamese network for Synthetic Lethality prediction.

    Takes two gene embeddings and predicts their SL probability
    using inner product with a learnable temperature.

    Architecture example (encoder_dims=[16, 16, 16, 16, 16, 16, 16] — the live
    SLURM default, slurm/config.conf; note every width is EQUAL, which is what
    lets SiameseEncoderResidual fire a skip on each hidden block. A narrowing
    stack such as [96, 80, 64, 48, 32, 32] gets zero skips under that encoder):
      gene1 -+-> (entry lift + 5x residual Linear->LN->LReLU->Drop) -> Linear -> z1
             |                                                            (projection) |
             |                                                                       |
             |          logit = τ · (z1ᵀz2 + ε · h1ᵀh2) + b -> sigmoid -> P(SL)    |
             |                                                                       |
      gene2 -+-> (same shared encoder) -------------------------------------------> z2

    With last_layer_bias=False the score is h1ᵀ(WᵀW + εI)h2, where W is the
    projection matrix; WᵀW + εI is strictly positive definite (ε > 0 prevents
    degeneracy when L1 pushes columns of W to zero). With a projection bias b
    the identity picks up per-gene terms — the score becomes
    h1ᵀ(WᵀW + εI)h2 + bᵀW(h1 + h2) + ‖b‖² — which is exactly the gene-specific
    baseline described below, and is no longer a pure kernel in (h1, h2).

    Last layer is a bare linear projection (Xavier init, no activation).
    Use last_layer_bias=False to remove the gene-specific baseline
    (node degree prior: some genes are SL with many partners).
    Symmetric by construction.
    """

    def __init__(
            self,
            input_dim: int = 1280,
            encoder_dims: list = None,
            dropout: float = 0.2,
            last_layer_bias: bool = True,
            pd_epsilon: float = 0.001,
            siamese_encoder_type: str = "residual",
            **kwargs,  # ignore hidden_dim/latent_dim etc. for backward compat
    ):
        super().__init__()
        if kwargs:
            import warnings
            warnings.warn(
                f"SiameseSL: ignoring unknown kwargs: {list(kwargs.keys())}")

        if encoder_dims is None:
            encoder_dims = [256, 128, 64]

        if siamese_encoder_type not in ("mlp", "residual"):
            raise ValueError(
                f"siamese_encoder_type must be 'mlp' or 'residual'; "
                f"got {siamese_encoder_type!r}")

        self.pd_epsilon = pd_epsilon
        self.siamese_encoder_type = siamese_encoder_type

        # Shared encoder for both genes. Residual is the default — same
        # interface as SiameseEncoder but wraps same-width hidden blocks
        # with y = x + F(x) skips, keeping gradient flow intact through
        # deeper stacks. See SiameseEncoderResidual's docstring.
        encoder_cls = (SiameseEncoderResidual if siamese_encoder_type
                       == "residual" else SiameseEncoder)
        self.encoder = encoder_cls(
            input_dim=input_dim,
            encoder_dims=encoder_dims,
            dropout=dropout,
            last_layer_bias=last_layer_bias,
        )

        # Learnable temperature (log-space for positivity and numerical stability)
        # τ = 1/√d is the standard scaled-dot-product init, derived for a bare
        # last layer where z1ᵀz2 ~ N(0, d) so initial logits have std ≈ 1.
        # NOTE that derivation does NOT hold for siamese_encoder_type=residual,
        # the live default: `h = out + h` is never re-normalised, so ‖h‖² grows
        # with the number of skips and the initial logits are correspondingly
        # larger than unit-scale. The temperature is trainable and recovers,
        # but do not read this init as calibrating the residual encoder.
        latent_dim = encoder_dims[-1]
        self.log_temperature = nn.Parameter(
            torch.tensor(-0.5 * np.log(float(latent_dim)),
                         dtype=torch.float32))

        # Scoring bias: shifts the decision boundary.
        # Accounts for SL base rate (SL is rare, so optimal boundary ≠ 0).
        self.scoring_bias = nn.Parameter(torch.tensor(0.0))

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
        z1, h1 = self.encoder.forward_with_hidden(x1)
        z2, h2 = self.encoder.forward_with_hidden(x2)

        # Kernel: h1ᵀ(WᵀW + εI)h2
        #   z1ᵀz2   = h1ᵀ(WᵀW)h2  — main term (projection inner product)
        #   ε·h1ᵀh2 = h1ᵀ(εI)h2   — PD regularizer (hidden inner product)
        inner = (z1 * z2).sum(dim=1) + self.pd_epsilon * (h1 * h2).sum(dim=1)

        # Scale by learnable temperature + scoring bias
        temperature = torch.exp(self.log_temperature)
        logits = temperature * inner + self.scoring_bias

        return logits.unsqueeze(1)  # (batch, 1)

    def predict_proba(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Return SL probability (sigmoid applied)."""
        logits = self.forward(x1, x2)
        return torch.sigmoid(logits)


class SiameseSLMultiCell(nn.Module):
    """Cell-line-conditioned SL scorer — masked multi-label, concat-at-input.

    Same PD-kernel scorer as SiameseSL, but the gene features are concatenated
    with a per-cell-line trainable embedding BEFORE the shared encoder, so the
    cell context interacts with the biological embedding inside the network.
    For a gene pair the model emits one logit PER cell-line head (see
    cell_line_vocab.HEADS); training supervises only the heads with a known
    label via a mask (handled in the loss), inference reads all heads.

    forward(x1, x2) -> (batch, num_heads):
        for head h: in = [x ‖ cell_emb[h]] -> z_h, score_h = τ·(z1ᵀz2 +
        ε·h1ᵀh2) + b_h. Symmetric in (x1, x2); per-head independent.

    A per-head scoring bias (b_h) absorbs each head's SL base rate (which range
    from ~2% to ~43% across lines). Temperature is shared. The cell embedding
    (cell_emb) is categorical context and is NOT L1-penalised (train.py excludes
    names starting with "cell_"), so --l1_lambdas keeps matching the encoder's
    weight matrices one-to-one.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        encoder_dims: list = None,
        dropout: float = 0.2,
        last_layer_bias: bool = True,
        pd_epsilon: float = 0.001,
        siamese_encoder_type: str = "residual",
        cell_line_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            import warnings
            warnings.warn(f"SiameseSLMultiCell: ignoring unknown kwargs: "
                          f"{list(kwargs.keys())}")
        if encoder_dims is None:
            encoder_dims = [256, 128, 64]
        if siamese_encoder_type not in ("mlp", "residual"):
            raise ValueError(
                f"siamese_encoder_type must be 'mlp' or 'residual'; "
                f"got {siamese_encoder_type!r}")

        self.pd_epsilon = pd_epsilon
        self.siamese_encoder_type = siamese_encoder_type
        self.num_heads = num_heads
        self.cell_line_dim = cell_line_dim

        # One trainable vector per cell-line head, concatenated onto each gene.
        self.cell_emb = nn.Embedding(num_heads, cell_line_dim)
        nn.init.normal_(self.cell_emb.weight, std=0.1)

        # Shared encoder sees [gene_emb ‖ cell_emb], so its input is widened.
        encoder_cls = (SiameseEncoderResidual if siamese_encoder_type
                       == "residual" else SiameseEncoder)
        self.encoder = encoder_cls(
            input_dim=input_dim + cell_line_dim,
            encoder_dims=encoder_dims,
            dropout=dropout,
            last_layer_bias=last_layer_bias,
        )

        latent_dim = encoder_dims[-1]
        self.log_temperature = nn.Parameter(
            torch.tensor(-0.5 * np.log(float(latent_dim)),
                         dtype=torch.float32))
        # Per-head scoring bias (one intercept per cell line).
        self.scoring_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Per-head SL logits.

        Args:
            x1, x2: gene embeddings (batch, input_dim).
        Returns:
            logits (batch, num_heads).
        """
        B = x1.shape[0]
        H = self.num_heads
        cell = self.cell_emb.weight  # (H, cell_dim)

        def encode_all_heads(x):
            # (B, D) -> (B, H, D+cell_dim) -> (B*H, D+cell_dim) -> encode
            xe = x.unsqueeze(1).expand(B, H, x.shape[1])
            ce = cell.unsqueeze(0).expand(B, H, self.cell_line_dim)
            xin = torch.cat([xe, ce], dim=-1).reshape(B * H, -1)
            return self.encoder.forward_with_hidden(xin)  # (B*H, latent/proj)

        z1, h1 = encode_all_heads(x1)
        z2, h2 = encode_all_heads(x2)

        # PD-kernel score per (pair, head), same form as SiameseSL.
        inner = (z1 * z2).sum(dim=1) + self.pd_epsilon * (h1 * h2).sum(dim=1)
        temperature = torch.exp(self.log_temperature)
        logits = (temperature * inner).reshape(B, H) + self.scoring_bias
        return logits  # (batch, num_heads)

    def predict_proba(self, x1: torch.Tensor,
                      x2: torch.Tensor) -> torch.Tensor:
        """Per-head SL probabilities (sigmoid applied)."""
        return torch.sigmoid(self.forward(x1, x2))


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

    # Note: With single-gene embeddings (sequence length 1), cross-attention
    # degenerates to a linear transformation. This class is retained for
    # experimentation but SiameseSL is the primary model.

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Learnable per-feature input bias.
        self.input_bias = nn.Parameter(torch.zeros(input_dim))

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
        # Apply input bias before projection
        x1 = x1 + self.input_bias
        x2 = x2 + self.input_bias

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
    then computes symmetric features from the Hilbert-space representations
    and feeds them through a predictor MLP.

    Pipeline:
        x ──> Encoder ──> z ──> φ(z) ∈ H ──> [sum, product, |diff|] ──> MLP ──> P(SL)

    The Hilbert space mapping φ combines:
    1. Learned linear projection (Mahalanobis-like metric)
    2. Random Fourier Features (Gaussian kernel approximation)

    Symmetric features [φ(z1)+φ(z2), φ(z1)*φ(z2), |φ(z1)-φ(z2)|] are
    concatenated and passed through a predictor MLP to produce logits.

    References:
    - Rahimi & Recht (2007): Random Features for Large-Scale Kernel Machines
    - arXiv:2508.04476: Metric Learning in an RKHS

    Args:
        input_dim: Gene embedding dimension (auto-detected from loaded embeddings)
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
        # Same symmetric aggregation as SiameseSLWithAttention:
        # [sum, product, abs_diff]. (SiameseSL itself does NOT aggregate this
        # way — it scores with a temperature-scaled inner product and no
        # predictor MLP.)
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
