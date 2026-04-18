#!/usr/bin/env python3
"""
Training script for Siamese SL prediction model.

All embeddings go through the same pipeline per modality:
  impute (Huber) → robust PCA (optional) → normalize (median center, MAD scale)
then concatenate along the feature axis, followed by an optional post-concat
robust PCA + re-normalize (on by default, --no_post_pca to disable).

Usage:
    # Single embedding
    python train.py --embeddings_paths ../data/all_genes_go.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --output_dir results/siamese_go

    # Multi-modal (concatenated)
    python train.py --embeddings_paths ../data/all_genes_bioconceptvec.pt \
                        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --output_dir results/multi_bio_go_ppi
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

        # Output directory (must already exist — server does not allow mkdir)
        self.output_dir = Path(args.output_dir)
        ckpt_dir = self.output_dir / "checkpoints"
        if not self.output_dir.is_dir() or not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Output dirs must be pre-created:\n"
                                    f"  {self.output_dir}\n  {ckpt_dir}")

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        # Load data (unified pipeline: impute → PCA → normalize → concat)
        self.data_manager = SLDataManager(
            embeddings_paths=args.embeddings_paths,
            sl_pairs_path=args.sl_path,
            gene_list_path=args.gene_list_path,
            seed=args.seed,
            pca_dims=args.pca_dims,
            post_pca=args.post_pca,
        )

        # Auto-detect input_dim from loaded embeddings
        actual_dim = self.data_manager.embeddings.shape[1]
        if args.input_dim is not None and args.input_dim != actual_dim:
            print(f"Warning: --input_dim={args.input_dim} overrides "
                  f"detected dim={actual_dim}")
        else:
            args.input_dim = actual_dim
        print(f"Input dim: {args.input_dim}")

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
                predictor_hidden=self.args.predictor_hidden,
                dropout=self.args.dropout,
                sigma=self.args.kernel_sigma,
                encoder_type=self.args.encoder_type,
                encoder_rank=self.args.encoder_rank,
            )
        else:
            encoder_dims = self.args.encoder_dims or [
                self.args.hidden_dim, self.args.latent_dim
            ]
            model = SiameseSL(
                input_dim=self.args.input_dim,
                encoder_dims=encoder_dims,
                dropout=self.args.dropout,
                last_layer_bias=self.args.last_layer_bias,
                pd_epsilon=self.args.pd_epsilon,
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

            # Proximal L1: per-layer soft-thresholding outside autograd
            # Each weight matrix (dim >= 2) gets its own lambda from --l1_lambdas
            # Biases and LayerNorm params (dim < 2) are never penalized
            if self._l1_map:
                current_lr = optimizer.param_groups[0]["lr"]
                eps = optimizer.defaults.get("eps", 1e-8)
                beta2 = optimizer.defaults["betas"][1]
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name not in self._l1_map:
                            continue
                        lam = self._l1_map[name]
                        if lam <= 0:
                            continue
                        state = optimizer.state.get(param, {})
                        if "exp_avg_sq" in state:
                            step = state["step"]
                            if isinstance(step, torch.Tensor):
                                step = step.item()
                            v_hat = state["exp_avg_sq"] / (1 - beta2**step)
                            denom = torch.clamp(v_hat.sqrt(), min=1e-1) + eps
                            thresh = lam * current_lr / denom
                        else:
                            thresh = lam * current_lr
                        param.data = torch.sign(param.data) * torch.clamp(
                            param.data.abs() - thresh, min=0)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

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
            print(
                f"Warning: Single-class test set (only class {unique_labels[0]}), metrics undefined"
            )
            return {
                "auroc": float("nan"),
                "aupr": float("nan"),
                "f1": float("nan"),
            }

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

        # Build per-layer L1 lambda map: param_name -> lambda
        # Only weight matrices (dim >= 2) are penalized
        self._l1_map = {}
        if self.args.l1_lambdas:
            weight_params = [(n, p) for n, p in model.named_parameters()
                             if p.dim() >= 2]
            if len(self.args.l1_lambdas) != len(weight_params):
                raise ValueError(
                    f"--l1_lambdas has {len(self.args.l1_lambdas)} values but "
                    f"model has {len(weight_params)} weight matrices: "
                    f"{[n for n, _ in weight_params]}")
            for (name, _), lam in zip(weight_params, self.args.l1_lambdas):
                self._l1_map[name] = lam
                print(f"  L1 λ={lam} for {name}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        # Cosine annealing with warm restarts (AdamWR)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.args.warmrestart_T0,
            T_mult=self.args.warmrestart_Tmult,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_auroc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_metrics = {}

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold_idx + 1}")
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer,
                                          criterion)
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

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": metrics,
                            "config": vars(self.args),
                        }, self.output_dir / "checkpoints" /
                        f"fold_{fold_idx}_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.args.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

        # Handle case where no evaluation occurred (epochs < eval_interval)
        if not best_metrics:
            print(
                f"\nWarning: No evaluation performed (epochs={self.args.epochs} < eval_interval={self.args.eval_interval})"
            )
            best_metrics = self.evaluate(model, test_loader)
            best_epoch = self.args.epochs
            # Save checkpoint so nonzero counting and downstream loading work
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": best_metrics,
                    "config": vars(self.args),
                },
                self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt")

        # Count nonzero parameters from best checkpoint (exclude buffers)
        ckpt = torch.load(
            self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt",
            map_location="cpu",
            weights_only=False,
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

        print(
            f"\nFold {fold_idx + 1} Best (epoch {best_epoch}): "
            f"AUROC={best_metrics['auroc']:.4f}, AUPR={best_metrics['aupr']:.4f}, "
            f"F1={best_metrics['f1']:.4f}")
        print(
            f"  Params: {nonzero_params:,}/{total_params:,} nonzero "
            f"| Weights: {wt_nonzero:,}/{wt_total:,} nonzero ({wt_sparsity:.2f}% sparse)"
        )

        # Per-layer sparsity breakdown
        layer_sparsities = {}
        for name in sd:
            if name in param_names and sd[name].dim() >= 2:
                t = sd[name]
                total = t.numel()
                nz = int(t.ne(0).sum())
                sp = 100 * (1 - nz / total)
                layer_sparsities[name] = sp
                print(
                    f"    {name}: {nz:,}/{total:,} nonzero ({sp:.1f}% sparse)")

        best_metrics["total_params"] = total_params
        best_metrics["nonzero_params"] = nonzero_params
        best_metrics["weight_sparsity"] = wt_sparsity
        best_metrics["layer_sparsities"] = layer_sparsities

        return best_metrics

    def train(self):
        """Run full cross-validation training."""
        print("=" * 60)
        print(f"Siamese SL Training")
        print(f"Model: {self.args.model_type}")
        print(f"CV Type: {self.args.cv_type}")
        print(f"Folds: {self.args.num_folds}")
        print("=" * 60)

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

        print("\n" + "=" * 60)
        print(
            f"Cross-Validation Results: {cv_desc.get(self.args.cv_type, self.args.cv_type)}"
        )
        print("=" * 60)

        # Per-fold results
        print(f"\n{'Fold':<6} {'AUROC':>10} {'AUPR':>10} {'F1':>10}")
        print("-" * 40)
        for i, m in enumerate(all_metrics):
            print(
                f"{i+1:<6} {m['auroc']:>10.4f} {m['aupr']:>10.4f} {m['f1']:>10.4f}"
            )
        print("-" * 40)
        print(
            f"{'Mean':<6} {np.nanmean(auroc_scores):>10.4f} {np.nanmean(aupr_scores):>10.4f} {np.nanmean(f1_scores):>10.4f}"
        )
        print(
            f"{'Std':<6} {np.nanstd(auroc_scores):>10.4f} {np.nanstd(aupr_scores):>10.4f} {np.nanstd(f1_scores):>10.4f}"
        )

        # Sparsity summary
        sparsity_scores = np.array(
            [m.get("weight_sparsity", 0) for m in all_metrics])
        avg_nonzero = int(
            np.mean([m.get("nonzero_params", 0) for m in all_metrics]))
        avg_total = int(
            np.mean([m.get("total_params", 0) for m in all_metrics]))

        print("\n" + "=" * 60)
        print("Summary:")
        print(
            f"  AUROC: {np.nanmean(auroc_scores):.4f} ± {np.nanstd(auroc_scores):.4f}"
        )
        print(
            f"  AUPR:  {np.nanmean(aupr_scores):.4f} ± {np.nanstd(aupr_scores):.4f}"
        )
        print(
            f"  F1:    {np.nanmean(f1_scores):.4f} ± {np.nanstd(f1_scores):.4f}"
        )
        print(f"  Params: {avg_nonzero:,}/{avg_total:,} nonzero "
              f"(weight sparsity: {np.mean(sparsity_scores):.2f}%)")
        print("=" * 60)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "cv_type": self.args.cv_type,
            "cv_description": cv_desc.get(self.args.cv_type,
                                          self.args.cv_type),
            "config": vars(self.args),
            "fold_metrics": all_metrics,
            "summary": {
                "auroc_mean": float(np.nanmean(auroc_scores)),
                "auroc_std": float(np.nanstd(auroc_scores)),
                "aupr_mean": float(np.nanmean(aupr_scores)),
                "aupr_std": float(np.nanstd(aupr_scores)),
                "f1_mean": float(np.nanmean(f1_scores)),
                "f1_std": float(np.nanstd(f1_scores)),
                "total_params": avg_total,
                "nonzero_params": avg_nonzero,
                "weight_sparsity": float(np.mean(sparsity_scores)),
            },
        }

        def _nan_to_none(obj):
            """Replace NaN with None for valid JSON serialization."""
            if isinstance(obj, (float, np.floating)) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _nan_to_none(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_nan_to_none(v) for v in obj]
            return obj

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(_nan_to_none(summary), f, indent=2)

        print(f"\nResults saved to {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese network for SL prediction")

    # Data paths
    parser.add_argument("--embeddings_path",
                        type=str,
                        default=None,
                        help="(Deprecated: use --embeddings_paths) "
                        "Single embedding file, equivalent to "
                        "--embeddings_paths with one file.")
    parser.add_argument("--embeddings_paths",
                        type=str,
                        nargs='+',
                        default=None,
                        help="One or more .pt embedding files. Each modality "
                        "is imputed, optionally PCA-reduced, and "
                        "MAD-normalized before concatenation.")
    parser.add_argument("--sl_path",
                        type=str,
                        default="../data/SL_SynLethDB_experimental.txt",
                        help="Path to SL pairs file")
    parser.add_argument("--gene_list_path",
                        type=str,
                        default=None,
                        help="Path to gene list file (for ordering)")
    parser.add_argument("--output_dir",
                        type=str,
                        default="results/siamese_esm",
                        help="Output directory")

    # Model architecture
    parser.add_argument(
        "--model_type",
        type=str,
        default="siamese",
        choices=["siamese", "attention", "kernel"],
        help=
        "Model: siamese (inner product, default), attention (cross-attn), kernel (RKHS-based)"
    )
    parser.add_argument("--input_dim",
                        type=int,
                        default=None,
                        help="Input embedding dim (auto-detected if omitted)")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=512,
                        help="Hidden dim")
    parser.add_argument("--latent_dim",
                        type=int,
                        default=256,
                        help="Latent dim")
    parser.add_argument("--predictor_hidden", type=int, default=128)
    parser.add_argument("--num_heads",
                        type=int,
                        default=4,
                        help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-last-layer-bias",
                        dest="last_layer_bias",
                        action="store_false",
                        default=True,
                        help="Remove last layer bias (gene node degree prior)")
    parser.add_argument(
        "--pd_epsilon",
        type=float,
        default=0.001,
        help="PD regularizer: ε in kernel K = WᵀW + εI (0 to disable)")
    parser.add_argument(
        "--encoder_dims",
        type=int,
        nargs='+',
        default=None,
        help="Encoder layer dims for siamese model (e.g., 256 128 64). "
        "Falls back to --hidden_dim/--latent_dim if not set.")

    # RKHS/Kernel model specific
    parser.add_argument(
        "--rff_features",
        type=int,
        default=128,
        help="Random Fourier features for kernel approximation")
    parser.add_argument("--bilinear_rank",
                        type=int,
                        default=128,
                        help="Hilbert space linear projection dimension")
    parser.add_argument("--kernel_sigma",
                        type=float,
                        default=1.0,
                        help="Initial RBF kernel bandwidth")

    # Efficient encoder options
    parser.add_argument("--encoder_type",
                        type=str,
                        default="standard",
                        choices=["standard", "lowrank", "bottleneck", "gated"],
                        help="Encoder architecture for efficiency")
    parser.add_argument("--encoder_rank",
                        type=int,
                        default=64,
                        help="Rank for lowrank encoder factorization")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--pca_dims",
        type=int,
        nargs='+',
        default=None,
        help="Per-modality PCA target dimensions (one per embedding file, "
        "e.g., 50 64 64). Robust PCA after imputation, before normalization. "
        "Values >= original dim are no-ops. Mutually exclusive with "
        "--pca_variance.")
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=None,
        help="Target variance fraction for PCA (0-1, e.g., 0.8 for 80%%). "
        "Applied to all modalities — each gets as many components as needed "
        "to explain this fraction. Mutually exclusive with --pca_dims.")
    parser.add_argument(
        "--post_pca_variance",
        type=float,
        default=0.8,
        help="Target variance fraction for a second ROBPCA applied to the "
        "concatenated matrix AFTER per-modality normalization. "
        "Default 0.8; pass --no_post_pca to disable, "
        "or --post_pca_dim for an exact component count.")
    parser.add_argument(
        "--post_pca_dim",
        type=int,
        default=None,
        help="Exact component count for the post-concat ROBPCA. "
        "Overrides --post_pca_variance when set.")
    parser.add_argument(
        "--no_post_pca",
        dest="no_post_pca",
        action="store_true",
        default=False,
        help="Disable the post-concat ROBPCA step (on by default).")
    parser.add_argument(
        "--l1_lambdas",
        type=float,
        nargs='+',
        default=None,
        help="Per-layer L1 lambdas (one per weight matrix, e.g., 0.1 0.05)")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (in eval intervals, not epochs)")
    parser.add_argument(
        "--warmrestart_T0",
        type=int,
        default=50,
        help="CosineAnnealingWarmRestarts: initial cycle length (epochs)")
    parser.add_argument(
        "--warmrestart_Tmult",
        type=int,
        default=2,
        help="CosineAnnealingWarmRestarts: cycle length multiplier")

    # Cross-validation
    parser.add_argument(
        "--cv_type",
        type=str,
        default="cv1",
        choices=["cv1", "cv2", "cv3"],
        help=
        "CV type: cv1=edge-based, cv2=gene-based, cv3=pair-based (both genes unseen)"
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--pos_neg_ratio", type=float, default=1.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    # Unify: --embeddings_path is a convenience alias for a single-element list
    if args.embeddings_paths and args.embeddings_path:
        parser.error("Use --embeddings_path OR --embeddings_paths, not both")
    if args.embeddings_path:
        args.embeddings_paths = [args.embeddings_path]
    if not args.embeddings_paths:
        parser.error("--embeddings_paths is required (or --embeddings_path)")

    # PCA: --pca_variance is a convenience flag that sets pca_dims to a
    # single float applied to all modalities.
    if args.pca_variance is not None and args.pca_dims is not None:
        parser.error("Use --pca_dims OR --pca_variance, not both")
    if args.pca_variance is not None:
        if not 0 < args.pca_variance < 1:
            parser.error("--pca_variance must be in (0, 1)")
        args.pca_dims = [args.pca_variance] * len(args.embeddings_paths)

    # Post-concat PCA: on by default (variance=0.8). Resolution order:
    #   --no_post_pca  -> disabled (None)
    #   --post_pca_dim -> exact component count
    #   otherwise      -> --post_pca_variance (default 0.8)
    if args.no_post_pca:
        args.post_pca = None
    elif args.post_pca_dim is not None:
        if args.post_pca_dim <= 0:
            parser.error("--post_pca_dim must be a positive integer")
        args.post_pca = args.post_pca_dim
    else:
        if not 0 < args.post_pca_variance < 1:
            parser.error("--post_pca_variance must be in (0, 1)")
        args.post_pca = args.post_pca_variance

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
