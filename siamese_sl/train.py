#!/usr/bin/env python3
"""
Training script for Siamese SL prediction model.

Usage:
    python train.py --embeddings_path ../data/all_genes_esm.pt \
                    --sl_path ../data/SL_Human_Approved.txt \
                    --output_dir results/siamese_esm

Features:
    - Full reproducibility via seeding
    - Cross-validation with multiple strategies
    - Early stopping and model checkpointing
    - Comprehensive metrics logging
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from tqdm import tqdm

from siamese_esm import SiameseSL, SiameseSLWithAttention, SiameseSLKernel, set_seed
from data_loader import SLDataManager, create_fold_dataloaders

# ── L1 proximal regularization ──────────────────────────────────────────
# Default L1 lambda for soft-thresholding (overridden by --l1_lambda CLI arg).
# Set to 0.0 to disable.
L1_LAMBDA_DEFAULT = 0.1


def calculate_optimal_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate optimal F1 from precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return float(np.max(f1_scores))


class Trainer:
    """Trainer for Siamese SL model."""

    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        # Device setup
        if torch.cuda.is_available() and not args.cpu:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and not args.cpu:
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        # Load data
        self.data_manager = SLDataManager(
            embeddings_path=args.embeddings_path,
            sl_pairs_path=args.sl_path,
            gene_list_path=args.gene_list_path,
            seed=args.seed,
        )

    def create_model(self) -> nn.Module:
        """Create the model."""
        if self.args.model_type == "attention":
            model = SiameseSLWithAttention(
                input_dim=self.args.input_dim,
                hidden_dim=self.args.hidden_dim,
                latent_dim=self.args.latent_dim,
                num_heads=self.args.num_heads,
                dropout=self.args.dropout,
            )
        elif self.args.model_type == "kernel":
            model = SiameseSLKernel(
                input_dim=self.args.input_dim,
                hidden_dim=self.args.hidden_dim,
                latent_dim=self.args.latent_dim,
                rff_features=self.args.rff_features,
                bilinear_rank=self.args.bilinear_rank,
                dropout=self.args.dropout,
                sigma=self.args.kernel_sigma,
                encoder_type=self.args.encoder_type,
                encoder_rank=self.args.encoder_rank,
            )
        else:
            model = SiameseSL(
                input_dim=self.args.input_dim,
                hidden_dim=self.args.hidden_dim,
                latent_dim=self.args.latent_dim,
                predictor_hidden=self.args.predictor_hidden,
                dropout=self.args.dropout,
            )
        return model.to(self.device)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x1, x2, labels in train_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Proximal L1: soft-thresholding outside autograd
            # Skip biases and LayerNorm params (only regularize weight matrices)
            # Use current LR from scheduler (not fixed CLI arg)
            if self.args.l1_lambda > 0:
                current_lr = optimizer.param_groups[0]["lr"]
                thresh = self.args.l1_lambda * current_lr
                with torch.no_grad():
                    for _, param in model.named_parameters():
                        if param.dim() < 2:
                            continue
                        param.data = torch.sign(param.data) * torch.clamp(
                            param.data.abs() - thresh, min=0
                        )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader,
    ) -> dict:
        """Evaluate model on test set."""
        model.eval()
        all_labels = []
        all_scores = []

        for x1, x2, labels in test_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)

            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_labels.extend(labels.numpy())
            all_scores.extend(probs)

        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        # Guard against single-class test set (can happen in edge cases)
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Single-class test set (only class {unique_labels[0]}), metrics undefined")
            return {"auroc": float("nan"), "aupr": float("nan"), "f1": float("nan")}

        # Compute metrics
        auroc = roc_auc_score(all_labels, all_scores)
        aupr = average_precision_score(all_labels, all_scores)
        f1 = calculate_optimal_f1(all_labels, all_scores)

        return {
            "auroc": auroc,
            "aupr": aupr,
            "f1": f1,
        }

    def train_fold(self, fold_data: dict) -> dict:
        """Train a single CV fold."""
        fold_idx = fold_data["fold"]
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}")
        print(f"{'='*60}")

        # Create dataloaders
        train_loader, test_loader = create_fold_dataloaders(
            embeddings=self.data_manager.embeddings,
            fold_data=fold_data,
            batch_size=self.args.batch_size,
        )

        # Create model and optimizer
        model = self.create_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        # Cosine annealing with warm restarts (AdamWR)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.args.warmrestart_T0, T_mult=self.args.warmrestart_Tmult,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_auroc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_metrics = {}

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold_idx}")
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

            # Evaluate periodically
            if (epoch + 1) % self.args.eval_interval == 0:
                metrics = self.evaluate(model, test_loader)

                pbar.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "auroc": f"{metrics['auroc']:.4f}",
                })

                # Save best model (based on AUROC)
                if metrics["auroc"] > best_auroc:
                    best_auroc = metrics["auroc"]
                    best_epoch = epoch + 1
                    best_metrics = metrics.copy()
                    patience_counter = 0

                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                        "config": vars(self.args),
                    }, self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.args.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

        # Handle case where no evaluation occurred (epochs < eval_interval)
        if not best_metrics:
            print(f"\nWarning: No evaluation performed (epochs={self.args.epochs} < eval_interval={self.args.eval_interval})")
            best_metrics = self.evaluate(model, test_loader)
            best_epoch = self.args.epochs
            # Save checkpoint so nonzero counting and downstream loading work
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": best_metrics,
                "config": vars(self.args),
            }, self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt")

        # Count nonzero parameters from best checkpoint (exclude buffers)
        ckpt = torch.load(
            self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt",
            map_location="cpu", weights_only=False,
        )
        sd = ckpt["model_state_dict"]
        param_names = {n for n, _ in model.named_parameters()}
        params = [sd[n] for n in sd if n in param_names]
        total_params = sum(p.numel() for p in params)
        nonzero_params = sum(int(p.ne(0).sum()) for p in params)
        # Weight matrices only (L1 targets)
        wt_total = sum(p.numel() for p in params if p.dim() >= 2)
        wt_nonzero = sum(int(p.ne(0).sum()) for p in params if p.dim() >= 2)
        wt_sparsity = 100 * (1 - wt_nonzero / wt_total) if wt_total > 0 else 0

        print(f"\nFold {fold_idx} Best (epoch {best_epoch}): "
              f"AUROC={best_metrics['auroc']:.4f}, AUPR={best_metrics['aupr']:.4f}, "
              f"F1={best_metrics['f1']:.4f}")
        print(f"  Params: {nonzero_params:,}/{total_params:,} nonzero "
              f"| Weights: {wt_nonzero:,}/{wt_total:,} nonzero ({wt_sparsity:.2f}% sparse)")

        best_metrics["total_params"] = total_params
        best_metrics["nonzero_params"] = nonzero_params
        best_metrics["weight_sparsity"] = wt_sparsity

        return best_metrics

    def train(self):
        """Run full cross-validation training."""
        print("="*60)
        print(f"Siamese SL Training")
        print(f"Model: {self.args.model_type}")
        print(f"CV Type: {self.args.cv_type}")
        print(f"Folds: {self.args.num_folds}")
        print("="*60)

        # Get CV splits
        if self.args.cv_type == "cv1":
            splits = self.data_manager.get_cv1_splits(
                num_folds=self.args.num_folds,
                pos_neg_ratio=self.args.pos_neg_ratio,
            )
        elif self.args.cv_type == "cv2":
            splits = self.data_manager.get_cv2_splits(
                num_folds=self.args.num_folds,
                pos_neg_ratio=self.args.pos_neg_ratio,
            )
        else:  # cv3
            splits = self.data_manager.get_cv3_splits(
                num_folds=self.args.num_folds,
                pos_neg_ratio=self.args.pos_neg_ratio,
            )

        # Train each fold
        all_metrics = []
        for split in splits:
            metrics = self.train_fold(split)
            all_metrics.append(metrics)

        # Summarize results
        auroc_scores = np.array([m["auroc"] for m in all_metrics], dtype=float)
        aupr_scores = np.array([m["aupr"] for m in all_metrics], dtype=float)
        f1_scores = np.array([m["f1"] for m in all_metrics], dtype=float)

        # Print results table
        cv_desc = {
            "cv1": "CV1 (edge-based)",
            "cv2": "CV2 (gene-based)",
            "cv3": "CV3 (pair-based)",
        }

        print("\n" + "="*60)
        print(f"Cross-Validation Results: {cv_desc.get(self.args.cv_type, self.args.cv_type)}")
        print("="*60)

        # Per-fold results
        print(f"\n{'Fold':<6} {'AUROC':>10} {'AUPR':>10} {'F1':>10}")
        print("-" * 40)
        for i, m in enumerate(all_metrics):
            print(f"{i+1:<6} {m['auroc']:>10.4f} {m['aupr']:>10.4f} {m['f1']:>10.4f}")
        print("-" * 40)
        print(f"{'Mean':<6} {np.nanmean(auroc_scores):>10.4f} {np.nanmean(aupr_scores):>10.4f} {np.nanmean(f1_scores):>10.4f}")
        print(f"{'Std':<6} {np.nanstd(auroc_scores):>10.4f} {np.nanstd(aupr_scores):>10.4f} {np.nanstd(f1_scores):>10.4f}")

        # Sparsity summary
        sparsity_scores = np.array([m.get("weight_sparsity", 0) for m in all_metrics])
        avg_nonzero = int(np.mean([m.get("nonzero_params", 0) for m in all_metrics]))
        avg_total = int(np.mean([m.get("total_params", 0) for m in all_metrics]))

        print("\n" + "="*60)
        print("Summary:")
        print(f"  AUROC: {np.nanmean(auroc_scores):.4f} ± {np.nanstd(auroc_scores):.4f}")
        print(f"  AUPR:  {np.nanmean(aupr_scores):.4f} ± {np.nanstd(aupr_scores):.4f}")
        print(f"  F1:    {np.nanmean(f1_scores):.4f} ± {np.nanstd(f1_scores):.4f}")
        print(f"  Params: {avg_nonzero:,}/{avg_total:,} nonzero "
              f"(weight sparsity: {np.mean(sparsity_scores):.2f}%)")
        print("="*60)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "cv_type": self.args.cv_type,
            "cv_description": cv_desc.get(self.args.cv_type, self.args.cv_type),
            "config": vars(self.args),
            "fold_metrics": all_metrics,
            "summary": {
                "auroc_mean": float(np.nanmean(auroc_scores)),
                "auroc_std": float(np.nanstd(auroc_scores)),
                "aupr_mean": float(np.nanmean(aupr_scores)),
                "aupr_std": float(np.nanstd(aupr_scores)),
                "f1_mean": float(np.nanmean(f1_scores)),
                "f1_std": float(np.nanstd(f1_scores)),
            },
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese network for SL prediction"
    )

    # Data paths
    parser.add_argument(
        "--embeddings_path", type=str, required=True,
        help="Path to gene embeddings file (.pt)"
    )
    parser.add_argument(
        "--sl_path", type=str, default="../data/SL_Human_Approved.txt",
        help="Path to SL pairs file"
    )
    parser.add_argument(
        "--gene_list_path", type=str, default=None,
        help="Path to gene list file (for ordering)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/siamese_esm",
        help="Output directory"
    )

    # Model architecture
    parser.add_argument(
        "--model_type", type=str, default="kernel",
        choices=["siamese", "attention", "kernel"],
        help="Model: siamese (MLP), attention (cross-attn), kernel (RKHS-based, default)"
    )
    parser.add_argument("--input_dim", type=int, default=1280, help="Input embedding dim")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dim")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dim")
    parser.add_argument("--predictor_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.2)

    # RKHS/Kernel model specific
    parser.add_argument("--rff_features", type=int, default=128,
                        help="Random Fourier features for kernel approximation")
    parser.add_argument("--bilinear_rank", type=int, default=128,
                        help="Hilbert space linear projection dimension")
    parser.add_argument("--kernel_sigma", type=float, default=1.0,
                        help="Initial RBF kernel bandwidth")

    # Efficient encoder options
    parser.add_argument("--encoder_type", type=str, default="standard",
                        choices=["standard", "lowrank", "bottleneck", "gated"],
                        help="Encoder architecture for efficiency")
    parser.add_argument("--encoder_rank", type=int, default=64,
                        help="Rank for lowrank encoder factorization")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--l1_lambda", type=float, default=L1_LAMBDA_DEFAULT,
                        help="L1 proximal regularization strength (0 to disable)")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping")
    parser.add_argument("--warmrestart_T0", type=int, default=50,
                        help="CosineAnnealingWarmRestarts: initial cycle length (epochs)")
    parser.add_argument("--warmrestart_Tmult", type=int, default=2,
                        help="CosineAnnealingWarmRestarts: cycle length multiplier")

    # Cross-validation
    parser.add_argument(
        "--cv_type", type=str, default="cv1",
        choices=["cv1", "cv2", "cv3"],
        help="CV type: cv1=edge-based, cv2=gene-based, cv3=pair-based (both genes unseen)"
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--pos_neg_ratio", type=float, default=1.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
