#!/usr/bin/env python3
"""
Loss functions for imbalanced synthetic lethality prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    where p_t = p if y=1, else 1-p; α_t balances classes, γ focuses on hard examples.
    
    Reference: Lin et al., ICCV 2017.
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        Args:
            alpha: Weighting factor for rare class (positive class)
            gamma: Focusing parameter. Higher gamma reduces loss for well-classified examples
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0 to 0.2)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (N,)
            targets: Binary targets (N,)
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (
                1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # BCE: -[y·log(p) + (1-y)·log(1-p)]
        ce_loss = F.binary_cross_entropy_with_logits(logits,
                                                     targets,
                                                     reduction='none')

        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # α_t = α if y=1, else 1-α (class balance)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulation: (1-p_t)^γ down-weights easy examples
        focal_weight = (1 - p_t)**self.gamma

        # FL = -α_t(1-p_t)^γ log(p_t)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE: L = -w⁺[y·log(p)] - w⁻[(1-y)·log(1-p)]
    where w⁺ = |neg|/|pos| for class balance.
    """

    def __init__(self,
                 pos_weight: Optional[float] = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        Args:
            pos_weight: Weight for positive class. If None, computed automatically
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (N,)
            targets: Binary targets (N,)
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (
                1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Auto-compute w⁺ = |D⁻|/|D⁺| if not provided
        pos_weight = self.pos_weight
        if pos_weight is None:
            n_pos = targets.sum()
            n_neg = (1 - targets).sum()
            if n_pos > 0:
                pos_weight = n_neg / n_pos  # Inverse frequency
            else:
                pos_weight = 1.0

        # Convert to tensor if needed
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight,
                                      device=logits.device,
                                      dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(logits,
                                                  targets,
                                                  pos_weight=pos_weight,
                                                  reduction=self.reduction)

        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss: L = -[y(1-p)^γ⁺ log(p) + (1-y)p^γ⁻ log(1-p)]
    with γ⁻ > γ⁺ to focus more on negatives.
    
    Reference: Ridnik et al., ICCV 2021.
    """

    def __init__(self,
                 gamma_neg: float = 4.0,
                 gamma_pos: float = 1.0,
                 clip: float = 0.05,
                 eps: float = 1e-8,
                 reduction: str = 'mean'):
        """
        Args:
            gamma_neg: Focusing parameter for negative class
            gamma_pos: Focusing parameter for positive class  
            clip: Probability clipping threshold
            eps: Small epsilon for numerical stability
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (N,)
            targets: Binary targets (N,)
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            probs = torch.clamp(probs, self.clip, 1 - self.clip)

        # Basic CE calculation
        targets_pos = targets
        targets_neg = 1 - targets

        # Positive and negative log probabilities
        log_pos = torch.log(probs + self.eps)
        log_neg = torch.log(1 - probs + self.eps)

        # Asymmetric modulation: γ⁻ > γ⁺ focuses on hard negatives
        pos_loss = targets_pos * (1 - probs)**self.gamma_pos * log_pos
        neg_loss = targets_neg * probs**self.gamma_neg * log_neg

        # L_asym = -[L⁺ + L⁻]
        loss = -(pos_loss + neg_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss: L = 1 - 2|X∩Y|/(|X|+|Y|)
    where X=predictions, Y=targets. Robust to class imbalance.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (N,)
            targets: Binary targets (N,)
        """
        probs = torch.sigmoid(logits)

        # Dice coefficient: DSC = 2Σ(p·y)/(Σp + Σy)
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth)

        # L_dice = 1 - DSC
        dice_loss = 1 - dice_coeff

        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for robust training on imbalanced data.
    """

    def __init__(self,
                 focal_weight: float = 0.7,
                 bce_weight: float = 0.3,
                 dice_weight: float = 0.0,
                 focal_params: dict = None,
                 bce_params: dict = None):
        """
        Args:
            focal_weight: Weight for focal loss component
            bce_weight: Weight for BCE loss component  
            dice_weight: Weight for dice loss component
            focal_params: Parameters for FocalLoss
            bce_params: Parameters for WeightedBCELoss
        """
        super().__init__()

        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # Initialize loss functions
        focal_params = focal_params or {}
        bce_params = bce_params or {}

        self.focal_loss = FocalLoss(
            **focal_params) if focal_weight > 0 else None
        self.bce_loss = WeightedBCELoss(
            **bce_params) if bce_weight > 0 else None
        self.dice_loss = DiceLoss() if dice_weight > 0 else None

        # Normalize weights
        total_weight = focal_weight + bce_weight + dice_weight
        if total_weight > 0:
            self.focal_weight /= total_weight
            self.bce_weight /= total_weight
            self.dice_weight /= total_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (N,)
            targets: Binary targets (N,)
        """
        total_loss = 0.0

        if self.focal_loss is not None and self.focal_weight > 0:
            focal_loss = self.focal_loss(logits, targets)
            total_loss += self.focal_weight * focal_loss

        if self.bce_loss is not None and self.bce_weight > 0:
            bce_loss = self.bce_loss(logits, targets)
            total_loss += self.bce_weight * bce_loss

        if self.dice_loss is not None and self.dice_weight > 0:
            dice_loss = self.dice_loss(logits, targets)
            total_loss += self.dice_weight * dice_loss

        return total_loss


def get_loss_function(loss_name: str,
                      pos_weight: Optional[float] = None,
                      **kwargs) -> nn.Module:
    """
    Factory function to create loss functions for imbalanced SL prediction.
    
    Args:
        loss_name: Name of loss function ('focal', 'weighted_bce', 'asymmetric', 'dice', 'combined')
        pos_weight: Positive class weight (computed automatically if None)
        **kwargs: Additional parameters for the loss function
        
    Returns:
        Configured loss function
    """
    loss_name = loss_name.lower()

    if loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'weighted_bce':
        return WeightedBCELoss(pos_weight=pos_weight, **kwargs)
    elif loss_name == 'asymmetric':
        return AsymmetricLoss(**kwargs)
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def compute_class_weights(targets: torch.Tensor) -> dict:
    """
    Compute class weights: w⁺ = |D⁻|/|D⁺| for class balance.
    
    Args:
        targets: Binary labels y ∈ {0,1}ⁿ
        
    Returns:
        Dictionary with weights and imbalance statistics
    """
    n_pos = targets.sum().item()
    n_neg = len(targets) - n_pos
    total = len(targets)

    if n_pos == 0:
        pos_weight = 1.0
        imbalance_ratio = float('inf')
    else:
        pos_weight = n_neg / n_pos
        imbalance_ratio = n_neg / n_pos

    return {
        'pos_weight': pos_weight,
        'imbalance_ratio': imbalance_ratio,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'total': total,
        'positive_ratio': n_pos / total,
        'negative_ratio': n_neg / total
    }
