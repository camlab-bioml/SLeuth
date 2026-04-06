#!/usr/bin/env python3
"""
Objective (loss) functions for SLMGAE.

This module provides different loss function implementations for experimentation.
Contains SLMGAELoss and can be extended with alternative objective functions.

KEY PRINCIPLE:
--------------
Loss is computed ONLY on selected edges (training edges), not on all possible edges.
This prevents computing loss on O(n²) pairs and focuses only on relevant edges.

MATHEMATICAL FORMULATION:
-------------------------
Let:
  - V = set of nodes/genes, |V| = n
  - S = set of training edges (both positive and negative)
  - y: S → {0,1} = ground truth labels
  - R⁽ᵐ⁾: V×V → ℝ = support view reconstruction (m = 1,2,3)
  - R_main: V×V → ℝ = main branch reconstruction
  - R: V×V → ℝ = final combined reconstruction

SLMGAE Loss Function:
  L_total = α·L_supp + β·L_pred + L_main

Where:
  L_supp = Σ_{m=1}^3 MSE(R⁽ᵐ⁾|_S, y)  [Sum over 3 support views]
  L_pred = MSE(R|_S, y)                [Final prediction]
  L_main = MSE(R_main|_S, y)           [Main branch]

Notation:
  - R|_S means R restricted to edges in S
  - MSE(R|_S, y) = (1/|S|) Σ_{(i,j)∈S} (R[i,j] - y[i,j])²

CRITICAL: All MSE terms are computed on training edges S only, NOT on all V×V pairs.

Reference: Matches TensorFlow implementation in original_tensorflow_code/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SLMGAELoss(nn.Module):
    """Loss: L = α·L_supp + β·L_pred + L_main where all use MSE"""

    def __init__(self, alpha=2.0, beta=4.0):
        super(SLMGAELoss, self).__init__()
        self.alpha = alpha  # Weight for support loss
        self.beta = beta  # Weight for prediction loss

    def forward(self, reconstructions, main_rec, att, support_recs,
                train_edges, train_labels):
        """L_total = α·Σ_m MSE(R^(m),y) + β·MSE(R,y) + MSE(R_main,y)"""

        # Sample predictions at training edges: R[E_train]
        labels_sub = train_labels.float()
        main_sub = main_rec[train_edges[:, 0], train_edges[:,
                                                           1]]  # R_main[edges]
        preds_sub = reconstructions[train_edges[:, 0],
                                    train_edges[:, 1]]  # R[edges]

        # Loss components (all using MSE = (1/n)Σ(y_pred-y_true)²):
        # L_supp: Sum of MSE for each individual support view reconstruction
        # Matching TF: for viewRec in supp: loss_supp += MSE(viewRec, labels)
        loss_supp = 0
        for support_rec in support_recs:  # Loop over 3 support views
            support_sub = support_rec[train_edges[:, 0], train_edges[:, 1]]
            loss_supp += F.mse_loss(support_sub, labels_sub)

        # L_main: MSE(R_main, y) - main branch reconstruction
        loss_main = F.mse_loss(main_sub, labels_sub)

        # L_pred: MSE(R, y) - final combined prediction
        loss_preds = F.mse_loss(preds_sub, labels_sub)

        # Total: L = α·L_supp + β·L_pred + 1.0·L_main
        total_loss = self.alpha * loss_supp + self.beta * loss_preds + 1.0 * loss_main

        return total_loss, loss_preds, loss_supp, loss_main


# Placeholder for alternative objective functions
# Users can add custom loss functions here for experimentation
class AlternativeObjective(nn.Module):
    """
    Placeholder for alternative objective functions.

    Example: weighted loss, focal loss, contrastive loss, etc.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super(AlternativeObjective, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # Add custom objective logic here

    def forward(
        self,
        reconstructions: torch.Tensor,
        main_rec: torch.Tensor,
        att: torch.Tensor,
        support_recs: list,
        train_edges: torch.Tensor,
        train_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Custom objective function.

        Args:
            reconstructions: Final reconstruction matrix R ∈ ℝ^(n×n)
            main_rec: Main branch reconstruction R_main ∈ ℝ^(n×n)
            att: Attention-combined support reconstruction R_att ∈ ℝ^(n×n)
            support_recs: List of 3 support view reconstructions [R⁽¹⁾, R⁽²⁾, R⁽³⁾]
            train_edges: Training edge indices, shape (|S|, 2)
            train_labels: Ground truth labels y ∈ {0,1}^|S|

        Returns:
            total_loss: Scalar loss value
            loss_dict: Dictionary with loss components
        """
        # Placeholder - implement custom loss here
        raise NotImplementedError("Implement custom objective function")


# Export default loss
__all__ = ["SLMGAELoss", "AlternativeObjective"]
