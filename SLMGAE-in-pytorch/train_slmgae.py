#!/usr/bin/env python3
"""
Training script for SLMGAE PyTorch implementation.

Uses external SL_benchmark functions for:
  - Train/test splitting: cv1/cv2/cv3 from preprocess_benchmarking_paper.py
  - Loss function: SLMGAELoss (custom PyTorch implementation)
  - Evaluation: sklearn AUC/AUPR + optimal F1 from precision-recall curve

Source: https://github.com/JieZheng-ShanghaiTech/SL_benchmark
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import argparse
from tqdm import tqdm

from slmgae_pytorch import (
    SLMGAE,
    DataLoader,
    sparse_to_torch_sparse,
    normalize_adj,
    set_random_seed,
)
from objective import SLMGAELoss

# One parameter-counting definition for all four models — see ../sl_comparison.
import sys as _sys

_sys.path.insert(0,
                 str(Path(__file__).resolve().parent.parent / "sl_comparison"))
from model_params import write_model_params  # noqa: E402
from data_split import SLDataSplitter  # Wrapper for SL_benchmark cv1/cv2/cv3
from evaluation import calculate_optimal_f1


class Trainer:
    """Trainer for SLMGAE model matching TensorFlow training."""

    def __init__(self, args):
        self.args = args
        # Set up device (respects CUDA_VISIBLE_DEVICES set by SLURM)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Don't hardcode GPU - use the one assigned by SLURM via CUDA_VISIBLE_DEVICES
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            self.device = torch.device("cpu")
            print("Using CPU (GPU not available)")

        # Set random seed
        set_random_seed(args.seed)

        # Create output directories
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{args.output_dir}/checkpoints").mkdir(parents=True,
                                                     exist_ok=True)
        Path(f"{args.output_dir}/predictions").mkdir(parents=True,
                                                     exist_ok=True)

        # Load data
        self.data_loader = DataLoader(data_path=args.data_path,
                                      nn_size=args.nn_size)
        self.pos_edges, self.neg_edges, self.support_adjs, self.num_nodes = (
            self.data_loader.load_data())

        # NOTE: Do NOT shuffle edges here! The SL_benchmark cv1/cv2/cv3 functions
        # handle their own shuffling internally with seed=123. Pre-shuffling
        # would interfere with their deterministic split logic.

        self.num_features = self.num_nodes  # Number of features is the number of nodes

        print(f"Loaded data:")
        print(f"  Positive edges: {len(self.pos_edges)}")
        print(f"  Negative edges: {len(self.neg_edges)}")
        print(f"  Support views: {len(self.support_adjs)}")

    def create_model(self):
        """Create model matching TF architecture."""
        model = SLMGAE(
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            hidden1=self.args.hidden1,
            hidden2=self.args.hidden2,
            dropout=self.args.dropout,
            num_support_views=len(self.support_adjs),
            device=self.device,
        ).to(self.device)
        return model

    def save_predictions(self, fold_idx, x_idx, y_idx, predictions):
        """Save predictions as matrix with Entrez Gene IDs."""
        # Load gene identifiers (Entrez IDs)
        with open(f"{self.args.data_path}/List_Proteins_in_SL.txt", "r") as f:
            genes = [line.strip() for line in f]

        n_genes = len(genes)

        # Create matrix
        matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
        for i in range(len(x_idx)):
            matrix[x_idx[i], y_idx[i]] = predictions[i]
            matrix[y_idx[i], x_idx[i]] = predictions[i]  # Symmetric

        # Save as DataFrame with gene names
        df_matrix = pd.DataFrame(matrix, index=genes, columns=genes)
        matrix_file = f"{self.args.output_dir}/fold_{fold_idx}_{self.args.cv_type}_predictions.csv"
        df_matrix.to_csv(matrix_file, float_format="%.6f")

        print(
            f"  Saved {n_genes}x{n_genes} prediction matrix to fold_{fold_idx}_{self.args.cv_type}_predictions.csv"
        )

    def get_features(self, train_adj):
        """
        Get node features for training.

        By default, uses the training adjacency matrix as features (matching TensorFlow).
        Subclasses can override this to use different feature types (e.g., ESM embeddings).

        Args:
            train_adj: Training adjacency matrix (scipy sparse)

        Returns:
            torch.Tensor: Node features (num_nodes x num_features)
        """
        return sparse_to_torch_sparse(train_adj).to(self.device)

    def train_fold_from_split(self, fold_idx, split):
        """Train a single CV fold using SLDataSplitter split."""
        print(f"\nTraining fold {fold_idx + 1}/{self.args.num_folds}...")

        # Extract split data
        train_adj = split["train_adj"].tocsr()  # Training adjacency
        train_edges = split["train_edges"]  # Training edge pairs
        train_labels = split["train_labels"]  # Training labels
        test_edges = split["test_edges"]  # Test edge pairs
        test_labels = split["test_labels"]  # Test labels

        # CHECKPOINT SELECTION RUNS ON VALIDATION, NEVER ON TEST.
        #
        # This loop previously scored split["test_edges"] every eva_epochs and
        # kept the argmax-AUROC checkpoint — i.e. it reported a MAXIMUM over ~12
        # test estimates per fold, while SLGNN (train_slgnn.py:90-96), NSF4SL
        # (train_nsf4sl.py:139) and siamese_sl all selected on the shared
        # validation split. On the 2026-08-06 shared-fold run that showed up as
        # a cv1 AUPRG std of 0.0006 across five DIFFERENT edge splits (siamese:
        # 0.0058) and as 9 of 15 selected epochs being interior maxima.
        #
        # The shared folds already carry val arrays (sl_comparison/shared_folds.py
        # :36-38, :61-63); the legacy SLDataSplitter path does not, so fall back
        # to test and say so loudly rather than crash.
        val_pos = np.asarray(split.get("val_pos", test_edges[:0]))
        val_neg = np.asarray(split.get("val_neg", test_edges[:0]))
        if len(val_pos) and len(val_neg):
            sel_edges = np.vstack([val_pos, val_neg]).astype(np.int64)
            sel_labels = np.concatenate(
                [np.ones(len(val_pos)),
                 np.zeros(len(val_neg))]).astype(np.float32)
            sel_name = "val"
        else:
            sel_edges = np.asarray(test_edges).astype(np.int64)
            sel_labels = np.asarray(test_labels).astype(np.float32)
            sel_name = "TEST"
            print(
                "  WARNING: no validation split in this fold set — selecting "
                "the checkpoint on TEST, the same edges this fold reports. "
                "Regenerate folds with prepare_folds.py --val_frac for an "
                "honest number.")

        # Normalize adjacency (normalize_adj adds self-loops internally)
        norm_adj = normalize_adj(train_adj)
        norm_adj_torch = sparse_to_torch_sparse(norm_adj).to(self.device)

        # Get features (can be overridden by subclasses, e.g., for ESM embeddings)
        features = self.get_features(train_adj)

        # Prepare support adjacencies (normalize_adj adds self-loops internally)
        support_adjs_torch = [
            sparse_to_torch_sparse(normalize_adj(adj)).to(self.device)
            for adj in self.support_adjs
        ]

        # Convert edges to torch
        train_edges_torch = torch.LongTensor(train_edges).to(self.device)
        train_labels_torch = torch.FloatTensor(train_labels).to(self.device)
        test_edges_torch = torch.LongTensor(test_edges).to(self.device)
        test_labels_torch = torch.FloatTensor(test_labels).to(self.device)
        sel_edges_torch = torch.LongTensor(sel_edges).to(self.device)
        sel_labels_np = sel_labels

        # Create model
        model = self.create_model()
        # selection_regime travels with the predictions (model_params.json ->
        # evaluate_model.build_summary -> compare.py), so the table itself shows
        # which rows were val-selected. Nothing else could distinguish a row
        # whose checkpoint was picked on TEST from one picked honestly.
        write_model_params(self.args.output_dir,
                           model,
                           folds_dir=getattr(self.args, "folds_dir", None),
                           cv_type=getattr(self.args, "cv_type", None),
                           selection_regime=sel_name)

        # Create optimizer and loss
        optimizer = optim.Adam(model.parameters(),
                               lr=self.args.learning_rate,
                               weight_decay=0)
        loss_fn = SLMGAELoss(alpha=self.args.alpha, beta=self.args.beta)

        # Training loop
        best_auc = 0
        best_epoch = 0
        patience_counter = 0

        # DELETE ANY CHECKPOINT LEFT BY A PREVIOUS RUN OF THIS FOLD.
        #
        # The post-loop torch.load treats a missing checkpoint as a hard error,
        # but it cannot tell a checkpoint THIS run wrote from one left behind by
        # an earlier run under different code. Re-running these folds after the
        # selection fix would otherwise silently reload a TEST-selected
        # checkpoint if training died before the first evaluation at
        # eva_epochs — publishing the old number under the new code.
        # The saved score matrix must go too. evaluate_model.py grades
        # <output_dir>/fold_<k>_<cv>_predictions.npy, NOT the checkpoint, so a
        # fold that fails after the matrix from a previous run was written
        # leaves that stale matrix to be graded silently alongside fresh ones.
        # Reproduced: four real cv3 matrices plus one foreign cv1 matrix scored
        # at exit 0 with no warning (AUROC 0.5767 where the true value is
        # 0.4961). Delete both, so a crashed fold is missing rather than wrong.
        cv = getattr(self.args, "cv_type", "cv")
        for stale in (Path(f"{self.args.output_dir}/checkpoints/"
                           f"fold_{fold_idx}_best.pt"),
                      Path(f"{self.args.output_dir}/"
                           f"fold_{fold_idx}_{cv}_predictions.npy")):
            if stale.exists():
                print(f"  Removing stale artifact from a previous run: "
                      f"{stale.name}")
                stale.unlink()

        epoch_pbar = tqdm(
            range(self.args.epochs),
            desc=f"Fold {fold_idx + 1} training",
            unit="epoch",
            leave=True,
        )

        for epoch in epoch_pbar:
            # Train
            model.train()
            optimizer.zero_grad()

            # Forward pass
            reconstructions, main_rec, att, support_recs = model(
                features, support_adjs_torch + [norm_adj_torch], self.args.coe)

            # Compute loss
            loss, loss_preds, loss_supp, loss_main = loss_fn(
                reconstructions,
                main_rec,
                att,
                support_recs,
                train_edges_torch,
                train_labels_torch,
            )

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Evaluate periodically
            if (epoch + 1) % self.args.eva_epochs == 0:
                model.eval()
                with torch.no_grad():
                    reconstructions, main_rec, att, support_recs = model(
                        features, support_adjs_torch + [norm_adj_torch],
                        self.args.coe)
                    # Use final reconstruction (already combined in model output)
                    final_rec = reconstructions

                    # Score the SELECTION split (validation), not test.
                    sel_preds = final_rec[sel_edges_torch[:, 0],
                                          sel_edges_torch[:, 1]]
                    sel_preds_np = sel_preds.cpu().numpy()

                    # Compute metrics
                    auc = roc_auc_score(sel_labels_np, sel_preds_np)
                    ap = average_precision_score(sel_labels_np, sel_preds_np)
                    f1 = calculate_optimal_f1(sel_labels_np, sel_preds_np)

                    epoch_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        f"{sel_name}_auc": f"{auc:.4f}"
                    })

                    # Save best model
                    if auc > best_auc:
                        best_auc = auc
                        best_epoch = epoch + 1
                        patience_counter = 0
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                # These are SELECTION-split metrics. They are
                                # NOT the reported numbers — test is scored
                                # exactly once below, after this checkpoint is
                                # restored. `selected_on` makes that auditable
                                # from the checkpoint alone.
                                "selected_on": sel_name,
                                "auc": auc,
                                "ap": ap,
                                "f1": f1,
                            },
                            f"{self.args.output_dir}/checkpoints/fold_{fold_idx}_best.pt",
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= self.args.early_stopping:
                            epoch_pbar.close()
                            print(f"\n  Early stopping at epoch {epoch + 1}")
                            break

        # Load best model and evaluate
        try:
            checkpoint = torch.load(
                f"{self.args.output_dir}/checkpoints/fold_{fold_idx}_best.pt",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            raise RuntimeError(
                f"No checkpoint found for fold {fold_idx}. Training may have failed "
                f"before the first evaluation at epoch {self.args.eva_epochs}."
            )

        model.eval()
        with torch.no_grad():
            reconstructions, main_rec, att, support_recs = model(
                features, support_adjs_torch + [norm_adj_torch], self.args.coe)
            final_rec = reconstructions  # Use combined reconstruction from model

            # Get test predictions
            test_preds = final_rec[test_edges_torch[:, 0], test_edges_torch[:,
                                                                            1]]
            test_preds_np = test_preds.cpu().numpy()
            test_labels_np = test_labels_torch.cpu().numpy()

            # Compute final metrics using sklearn (simple AUC/AUPR/F1)
            # Note: SL_benchmark's cal_metrics provides 15 metrics including NDCG, Recall@K
            # For compatibility with TensorFlow original, we use simple sklearn metrics here
            auc = roc_auc_score(test_labels_np, test_preds_np)
            ap = average_precision_score(test_labels_np, test_preds_np)
            f1 = calculate_optimal_f1(test_labels_np, test_preds_np)

            # Save full prediction matrix
            # Extract upper triangular indices and predictions
            x_idx, y_idx = np.triu_indices(self.num_nodes, k=1)
            all_preds_matrix = final_rec.cpu().numpy()
            all_preds = all_preds_matrix[x_idx, y_idx]
            self.save_predictions(fold_idx, x_idx, y_idx, all_preds)

        print(f"  Best fold {fold_idx + 1}: epoch {best_epoch} "
              f"(selected on {sel_name}, {sel_name} AUROC {best_auc:.4f}) | "
              f"TEST AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

        return auc, ap, f1

    def train(self):
        """
        Main training loop using SL_benchmark's cross-validation strategies.

        Pipeline:
          1. Call SL_benchmark cv1/cv2/cv3 via data_split.py wrapper
          2. SL_benchmark splits edges and builds adjacency per fold
          3. Train model using pre-built adjacency from training edges only
          4. Evaluate on test edges using sklearn metrics
        """
        print("=" * 60)
        print(
            f"Starting {self.args.num_folds}-fold cross-validation training ({self.args.cv_type.upper()})"
        )
        print(
            f"Train ratio: {self.args.train_ratio:.2f}, Test ratio: {1-self.args.train_ratio:.2f}"
        )
        print("=" * 60)

        # Step 1: Get splits from SL_benchmark
        # Wrapper injects our edges into SL_benchmark's cv1/cv2/cv3 functions
        splitter = SLDataSplitter(
            pos_edges=self.pos_edges,
            neg_edges=self.neg_edges,
            num_nodes=self.num_nodes,
            train_ratio=self.args.train_ratio,
            random_state=self.args.seed,
        )

        # Call appropriate SL_benchmark function
        if self.args.cv_type == "cv1":
            splits = splitter.cv1_split(k=self.args.num_folds,
                                        pos_neg_ratio=self.args.pos_neg_ratio)
        elif self.args.cv_type == "cv2":
            splits = splitter.cv2_split(k=self.args.num_folds,
                                        pos_neg_ratio=self.args.pos_neg_ratio)
        elif self.args.cv_type == "cv3":
            splits = splitter.cv3_split(k=self.args.num_folds,
                                        pos_neg_ratio=self.args.pos_neg_ratio)
        else:
            raise ValueError(f"Unknown CV type: {self.args.cv_type}")

        # Each split contains:
        #   - train_adj: Pre-built from training positive edges only
        #   - train_edges, train_labels: Training supervision
        #   - test_edges, test_labels: Test supervision

        auc_scores = []
        ap_scores = []
        f1_scores = []

        print("\nStarting cross-validation...")

        for split in splits:
            fold_idx = split["fold"]

            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{self.args.num_folds}")
            print(f"{'='*60}")
            print(
                f"Train: {split['num_train_pos']} positive, {split['num_train_neg']} negative"
            )
            print(
                f"Test: {split['num_test_pos']} positive, {split['num_test_neg']} negative"
            )

            # Train fold using split data
            auc, ap, f1 = self.train_fold_from_split(fold_idx, split)

            auc_scores.append(auc)
            ap_scores.append(ap)
            f1_scores.append(f1)

            print(
                f"\nFold {fold_idx + 1} Results: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}"
            )

        # Print summary
        print("\n" + "=" * 60)
        print("Cross-validation results:")
        print("=" * 60)
        print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        print(f"AP:  {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
        print(f"F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

        # Save summary
        summary = {
            "cv_type": self.args.cv_type,
            "auc_scores": auc_scores,
            "ap_scores": ap_scores,
            "f1_scores": f1_scores,
            "auc_mean": float(np.mean(auc_scores)),
            "auc_std": float(np.std(auc_scores)),
            "ap_mean": float(np.mean(ap_scores)),
            "ap_std": float(np.std(ap_scores)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "args": vars(self.args),
        }

        with open(f"{self.args.output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {self.args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train SLMGAE PyTorch model")

    # Model parameters matching TF defaults
    parser.add_argument("--hidden1",
                        type=int,
                        default=512,
                        help="Number of units in hidden layer 1")
    parser.add_argument("--hidden2",
                        type=int,
                        default=256,
                        help="Number of units in hidden layer 2")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.20,
                        help="Dropout rate")
    parser.add_argument("--nn_size",
                        type=int,
                        default=45,
                        help="Number of K for the KNN")

    # Loss coefficients matching TF
    parser.add_argument("--alpha",
                        type=float,
                        default=2.0,
                        help="Coefficient of support view loss")
    parser.add_argument(
        "--coe",
        type=float,
        default=2.0,
        help=
        "Coefficient for attention combination (lambda in R = R_main + lambda*R_att)",
    )
    parser.add_argument("--beta",
                        type=float,
                        default=4.0,
                        help="Coefficient of final loss")

    # Training parameters matching TF
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--epochs",
                        type=int,
                        default=300,
                        help="Number of epochs to train")
    parser.add_argument("--eva_epochs",
                        type=int,
                        default=25,
                        help="Number of epochs to evaluate")
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=20,
        help="Tolerance for early stopping (# of evaluations)",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--data_path",
                        type=str,
                        default="../data",
                        help="Path to data directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/slmgae_pytorch",
        help="Output directory",
    )
    parser.add_argument("--use_esm",
                        action="store_true",
                        help="Use ESM embeddings as features")
    parser.add_argument(
        "--cv_type",
        type=str,
        default="cv1",
        choices=["cv1", "cv2", "cv3"],
        help=
        "Cross-validation type: cv1 (edge-based), cv2 (gene-based), cv3 (pair-based)",
    )
    parser.add_argument(
        "--pos_neg_ratio",
        type=float,
        default=1.0,
        help="Positive to negative ratio for training (default: 1.0 for 1:1)",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help=
        "Fraction of data for training (default: 0.8, test will be 1-train_ratio)",
    )

    args = parser.parse_args()

    # Train model
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
