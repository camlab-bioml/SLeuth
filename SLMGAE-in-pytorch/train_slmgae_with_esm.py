#!/usr/bin/env python3
"""
Training script for SLMGAE with optional ESM embeddings.
Extension of base training script to support ESM features.
"""

import os
import torch

from train_slmgae import Trainer as BaseTrainer
from slmgae_pytorch import SLMGAE


class ESMTrainer(BaseTrainer):
    """
    Extended trainer with ESM embedding support.

    Architecture:
    - Inherits the complete training loop from BaseTrainer (no code duplication)
    - Overrides get_features() to use ESM protein embeddings instead of adjacency
    - Overrides create_model() to handle different feature dimensions (1280 for ESM)
    - All training, evaluation, checkpointing logic is shared with BaseTrainer

    This design eliminates ~160 lines of duplicated code and ensures consistency
    between standard and ESM-based training.
    """

    def __init__(self, args):
        super().__init__(args)

        if args.use_esm:
            # Load ESM embeddings
            self.load_esm_embeddings()
            self.num_features = 1280  # ESM embedding dimension
        else:
            self.num_features = self.num_nodes  # Use adjacency as features
            self.esm_embeddings = None

    def load_esm_embeddings(self):
        """Load pre-computed ESM embeddings and reorder to match List_Proteins_in_SL.txt.

        Both the ESM file's gene_order and List_Proteins_in_SL.txt use
        Entrez Gene IDs as identifiers.
        """
        esm_path = f"{self.args.data_path}/embeddings_esm2_t33_650M_UR50D_meanpool.pt"

        if not os.path.exists(esm_path):
            esm_path = "../ESM_embedding/main_esm_embeddings.pt"

        if not os.path.exists(esm_path):
            raise FileNotFoundError(
                f"ESM embeddings not found. Please run generate_esm_embeddings.py first."
            )

        print(f"Loading ESM embeddings from {esm_path}")

        # Load ESM embeddings
        if esm_path.endswith(".pt"):
            data = torch.load(esm_path, map_location="cpu", weights_only=False)

            if isinstance(data, dict):
                esm_embeddings_raw = data["embeddings"]
                esm_gene_order = data.get("gene_order", None)
            else:
                esm_embeddings_raw = data
                esm_gene_order = None
        else:
            raise ValueError(f"Unsupported file format: {esm_path}")

        print(f"ESM embeddings shape (raw): {esm_embeddings_raw.shape}")

        # Load correct gene order from List_Proteins_in_SL.txt
        genes_path = f"{self.args.data_path}/List_Proteins_in_SL.txt"
        with open(genes_path, "r") as f:
            correct_genes = [line.strip() for line in f]

        print(
            f"Expected gene order from {genes_path}: {len(correct_genes)} genes"
        )

        # If gene_order is available, reorder embeddings
        if esm_gene_order is not None:
            print(f"ESM gene order available: {len(esm_gene_order)} genes")

            # Create mapping: gene_name -> index in ESM embeddings
            esm_gene_to_idx = {
                gene: idx
                for idx, gene in enumerate(esm_gene_order)
            }

            # Reorder embeddings to match correct_genes order
            embedding_dim = esm_embeddings_raw.shape[1]
            reordered_embeddings = torch.zeros(len(correct_genes),
                                               embedding_dim)

            matched_count = 0
            for target_idx, gene in enumerate(correct_genes):
                if gene in esm_gene_to_idx:
                    source_idx = esm_gene_to_idx[gene]
                    reordered_embeddings[target_idx] = esm_embeddings_raw[
                        source_idx]
                    matched_count += 1
                # else: keep zeros for missing genes

            self.esm_embeddings = reordered_embeddings
            print(
                f"Reordered embeddings: {matched_count}/{len(correct_genes)} genes matched"
            )

            if matched_count < len(correct_genes):
                print(
                    f"Warning: {len(correct_genes) - matched_count} genes missing from ESM embeddings (using zeros)"
                )
        else:
            # No gene_order available - assume ESM order matches correct order
            print(
                "Warning: No gene_order in ESM file. Assuming ESM order matches List_Proteins_in_SL.txt"
            )

            if esm_embeddings_raw.shape[0] != len(correct_genes):
                print(
                    f"Warning: ESM has {esm_embeddings_raw.shape[0]} embeddings, "
                    f"expected {len(correct_genes)}. Padding/truncating.")

                embedding_dim = esm_embeddings_raw.shape[1]
                reordered_embeddings = torch.zeros(len(correct_genes),
                                                   embedding_dim)
                min_len = min(esm_embeddings_raw.shape[0], len(correct_genes))
                reordered_embeddings[:min_len] = esm_embeddings_raw[:min_len]
                self.esm_embeddings = reordered_embeddings
            else:
                self.esm_embeddings = esm_embeddings_raw

        print(f"Final ESM embeddings shape: {self.esm_embeddings.shape}")
        self.esm_embeddings = self.esm_embeddings.to(self.device)

    def get_features(self, train_adj):
        """
        Override to use ESM embeddings when available.

        Uses pre-computed ESM protein embeddings if use_esm flag is set,
        otherwise falls back to using training adjacency as features.

        Args:
            train_adj: Training adjacency matrix (scipy sparse) - used as fallback

        Returns:
            torch.Tensor: Node features
                - If use_esm=True: ESM embeddings (num_nodes x 1280)
                - Otherwise: Sparse adjacency features (num_nodes x num_nodes)
        """
        if self.args.use_esm and self.esm_embeddings is not None:
            # Use pre-loaded ESM embeddings (already on device from __init__)
            return self.esm_embeddings
        else:
            # Fall back to adjacency features (matching TensorFlow)
            return super().get_features(train_adj)

    def create_model(self):
        """Create model with correct feature dimension."""
        model = SLMGAE(
            num_nodes=self.num_nodes,
            num_features=self.num_features,  # Will be 1280 if using ESM
            hidden1=self.args.hidden1,
            hidden2=self.args.hidden2,
            dropout=self.args.dropout,
            num_support_views=len(self.support_adjs),
        ).to(self.device)
        return model


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SLMGAE with optional ESM embeddings")

    # Model parameters
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

    # Loss coefficients
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

    # Training parameters
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
    parser.add_argument("--early_stopping",
                        type=int,
                        default=20,
                        help="Tolerance for early stopping")

    # Other parameters
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--data_path",
                        type=str,
                        default="../data",
                        help="Path to data directory")
    parser.add_argument("--output_dir",
                        type=str,
                        default="results",
                        help="Output directory")
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

    # Keep output directory as specified (no subdirectories)

    print("=" * 60)
    print("SLMGAE PyTorch Training")
    print("=" * 60)
    print(f"Using ESM embeddings: {args.use_esm}")
    print(f"Output directory: {args.output_dir}")

    # Train model
    trainer = ESMTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
