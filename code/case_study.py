#!/usr/bin/env python3
"""
Case study for SLMGAE - FIXED to use TensorFlow-matching architecture
Tests on specific gene pairs of interest using the EXACT same model as TensorFlow
"""

import os
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import scipy.sparse as sp
import random
import warnings

warnings.filterwarnings("ignore")

# Import the TensorFlow-matching SLMGAE model
from slmgae_pytorch import SLMGAE
from objective import SLMGAELoss


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_main_data():
    """Load main dataset"""
    print("Loading main dataset...")

    # Load gene list
    genes_file = Path("../data/List_Proteins_in_SL.txt")
    with open(genes_file, "r") as f:
        genes = [line.strip() for line in f if line.strip()]

    print(f"Total genes: {len(genes)}")

    # Create gene to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    # Load SL edges
    sl_file = Path("../data/SL_Human_Approved.txt")
    edges = []
    with open(sl_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    gene1, gene2 = parts[0], parts[1]
                    if gene1 in gene_to_idx and gene2 in gene_to_idx:
                        edges.append([gene_to_idx[gene1], gene_to_idx[gene2]])
                        edges.append([gene_to_idx[gene2],
                                      gene_to_idx[gene1]])  # Undirected

    edges = np.array(edges)
    print(f"Loaded {len(edges)//2} SL pairs")

    return edges, genes, gene_to_idx


def load_support_views():
    """Load support view adjacency matrices using DataLoader methods (matching main training)"""
    print("Loading support views...")

    # Use DataLoader to load support views with proper preprocessing (KNN, symmetrization)
    from slmgae_pytorch import DataLoader as SLDataLoader

    loader = SLDataLoader(data_path="../data/", nn_size=45)
    loader.num_nodes = 6375  # Full dataset size

    support_views = []

    # Load GO BP with proper preprocessing (triangular format + KNN + double symmetrization)
    gosim_bp = loader.load_dense_feature("../data/Human_GOsim.txt", knn=True)
    gosim_bp_dense = gosim_bp.toarray()
    support_views.append(gosim_bp_dense)
    print(f"  Loaded GO BP similarity: {gosim_bp_dense.shape}")

    # Load GO CC with proper preprocessing
    gosim_cc = loader.load_dense_feature("../data/Human_GOsim_CC.txt",
                                         knn=True)
    gosim_cc_dense = gosim_cc.toarray()
    support_views.append(gosim_cc_dense)
    print(f"  Loaded GO CC similarity: {gosim_cc_dense.shape}")

    # Load PPI with proper preprocessing (edge list + symmetrization)
    ppi = loader.load_sparse_feature("../data/biogrid_ppi_sparse.txt")
    ppi_dense = ppi.toarray()
    support_views.append(ppi_dense)
    print(f"  Loaded PPI network: {ppi_dense.shape}")

    print(f"Total support views: {len(support_views)}")
    return support_views


def prepare_adjacencies(edges, n_genes, support_views):
    """Prepare all adjacency matrices matching TensorFlow order: [support1, support2, support3, main]"""

    # Main adjacency from SL edges
    main_adj = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_genes, n_genes),
        dtype=np.float32,
    )
    main_adj = main_adj.tocsr()
    main_adj.setdiag(0)
    main_adj.eliminate_zeros()
    main_adj = main_adj.toarray()

    # Normalize adjacencies
    def normalize_adj(adj):
        """Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)"""
        adj = adj + np.eye(adj.shape[0])  # Add self-loops
        degree = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Create list matching TensorFlow: [support1, support2, support3, main]
    adjs_normalized = []

    # Add support views (sparse for first layer)
    for support_adj in support_views:
        adj_norm = normalize_adj(support_adj)
        adj_sparse = sp.coo_matrix(adj_norm, dtype=np.float32)
        adjs_normalized.append(adj_sparse)

    # Add main view (dense for second layer)
    main_norm = normalize_adj(main_adj)
    adjs_normalized.append(main_norm.astype(np.float32))

    return adjs_normalized


def get_case_study_pairs(gene_to_idx):
    """Define case study gene pairs"""
    case_study_pairs = [
        # Well-known synthetic lethal pairs
        ("BRCA1", "PARP1"),
        ("BRCA2", "PARP1"),
        ("TP53", "MDM2"),
        ("KRAS", "STK33"),
        ("VHL", "HIF1A"),
        ("ARID1A", "ARID1B"),
        ("SMARCA4", "SMARCA2"),
        ("RB1", "E2F1"),
        ("PTEN", "PIK3CA"),
        ("MYC", "CDK9"),
    ]

    # Convert to indices
    valid_pairs = []
    for gene1, gene2 in case_study_pairs:
        if gene1 in gene_to_idx and gene2 in gene_to_idx:
            valid_pairs.append({
                "gene1": gene1,
                "gene2": gene2,
                "idx1": gene_to_idx[gene1],
                "idx2": gene_to_idx[gene2],
            })
        else:
            missing = []
            if gene1 not in gene_to_idx:
                missing.append(gene1)
            if gene2 not in gene_to_idx:
                missing.append(gene2)
            print(
                f"Warning: Gene pair ({gene1}, {gene2}) - missing genes: {missing}"
            )

    return valid_pairs


class CaseStudyEvaluator:

    def __init__(self, edges, genes, gene_to_idx, args, device="cuda"):
        self.edges = edges
        self.genes = genes
        self.gene_to_idx = gene_to_idx
        self.n_genes = len(genes)
        self.args = args
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Enable GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load and reorder ESM embeddings if requested (BEFORE model creation)
        if args.use_esm:
            self.num_features = 1280  # ESM embedding dimension
            self.esm_embeddings = self._load_and_reorder_esm()
        else:
            self.num_features = self.n_genes  # Use adjacency as features
            self.esm_embeddings = None

    def _load_and_reorder_esm(self):
        """
        Load ESM embeddings and reorder to match gene order.

        Reuses the logic from ESMTrainer.load_esm_embeddings to ensure
        embeddings align with List_Proteins_in_SL.txt gene order.
        """
        esm_path = "../data/embeddings_esm2_t33_650M_UR50D_meanpool.pt"
        if not os.path.exists(esm_path):
            raise FileNotFoundError(
                f"ESM embeddings not found at {esm_path}. "
                f"Please run generate_esm_embeddings.py first.")

        print(f"Loading ESM embeddings from {esm_path}")
        data = torch.load(esm_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            esm_embeddings_raw = data["embeddings"]
            esm_gene_order = data.get("gene_order", None)
        else:
            esm_embeddings_raw = data
            esm_gene_order = None

        print(f"ESM embeddings shape (raw): {esm_embeddings_raw.shape}")

        # Reorder embeddings to match self.genes order
        if esm_gene_order is not None:
            print(f"ESM gene order available: {len(esm_gene_order)} genes")
            esm_gene_to_idx = {
                gene: idx
                for idx, gene in enumerate(esm_gene_order)
            }

            embedding_dim = esm_embeddings_raw.shape[1]
            reordered_embeddings = torch.zeros(len(self.genes), embedding_dim)

            matched_count = 0
            for target_idx, gene in enumerate(self.genes):
                if gene in esm_gene_to_idx:
                    source_idx = esm_gene_to_idx[gene]
                    reordered_embeddings[target_idx] = esm_embeddings_raw[
                        source_idx]
                    matched_count += 1

            print(
                f"Reordered embeddings: {matched_count}/{len(self.genes)} genes matched"
            )
            if matched_count < len(self.genes):
                print(
                    f"Warning: {len(self.genes) - matched_count} genes missing (using zeros)"
                )

            esm_embeddings = reordered_embeddings
        else:
            print(
                "Warning: No gene_order in ESM file. Assuming order matches List_Proteins_in_SL.txt"
            )
            if esm_embeddings_raw.shape[0] != len(self.genes):
                print(
                    f"Warning: ESM has {esm_embeddings_raw.shape[0]} embeddings, expected {len(self.genes)}"
                )
                embedding_dim = esm_embeddings_raw.shape[1]
                reordered_embeddings = torch.zeros(len(self.genes),
                                                   embedding_dim)
                min_len = min(esm_embeddings_raw.shape[0], len(self.genes))
                reordered_embeddings[:min_len] = esm_embeddings_raw[:min_len]
                esm_embeddings = reordered_embeddings
            else:
                esm_embeddings = esm_embeddings_raw

        print(f"Final ESM embeddings shape: {esm_embeddings.shape}")
        return esm_embeddings.to(self.device)

    def train_model(self, adjs_normalized, args):
        """Train SLMGAE model matching TensorFlow exactly"""
        print("\nTraining SLMGAE model (TensorFlow-matching architecture)...")
        print("Architecture:")
        print("  - 4 parallel branches (3 support + 1 main)")
        print("  - GraphConvolutionSparse → GraphConvolution")
        print("  - LeakyReLU activation (not ReLU)")
        print("  - Attention layer for support views")
        print("  - InnerProductDecoder with Z*W*Z^T")
        print("  - MSE loss (not BCE)")
        print("  - No sigmoid on predictions")

        # Get positive edges
        pos_edges = np.unique(self.edges.reshape(-1, 2), axis=0)
        pos_edges = pos_edges[pos_edges[:, 0]
                              < pos_edges[:, 1]]  # Upper triangle only

        # Sample negative edges
        all_possible = set()
        for i in range(self.n_genes):
            for j in range(i + 1, self.n_genes):
                all_possible.add((i, j))

        pos_set = set(map(tuple, pos_edges))
        neg_edges = np.array(list(all_possible - pos_set))

        # Sample negatives (1:10 ratio)
        n_neg_sample = min(len(neg_edges), len(pos_edges) * 10)
        neg_edges = neg_edges[np.random.choice(len(neg_edges),
                                               n_neg_sample,
                                               replace=False)]

        # Combine and create labels
        all_edges = np.vstack([pos_edges, neg_edges])
        all_labels = np.hstack(
            [np.ones(len(pos_edges)),
             np.zeros(len(neg_edges))])

        # Shuffle and split train/val
        idx = np.random.permutation(len(all_edges))
        all_edges = all_edges[idx]
        all_labels = all_labels[idx]

        split_idx = int(0.8 * len(all_edges))

        # Extract TRAINING positive edges only (avoid data leakage)
        train_pos_edges = all_edges[:split_idx][all_labels[:split_idx] == 1]

        # Build adjacency from TRAINING positive edges only
        train_adj = sp.coo_matrix(
            (
                np.ones(len(train_pos_edges)),
                (train_pos_edges[:, 0], train_pos_edges[:, 1]),
            ),
            shape=(self.n_genes, self.n_genes),
            dtype=np.float32,
        )
        # Symmetrize: add transpose (matching TensorFlow train.py:74-75)
        train_adj = train_adj + train_adj.T

        # Convert train/val edges to tensors
        train_edges = torch.tensor(all_edges[:split_idx],
                                   dtype=torch.long).to(self.device)
        train_labels = torch.tensor(all_labels[:split_idx],
                                    dtype=torch.float32).to(self.device)
        val_edges = torch.tensor(all_edges[split_idx:],
                                 dtype=torch.long).to(self.device)
        val_labels = torch.tensor(all_labels[split_idx:],
                                  dtype=torch.float32).to(self.device)

        # Create SLMGAE model with correct feature dimension
        model = SLMGAE(
            num_nodes=self.n_genes,
            num_features=self.num_features,  # 1280 if ESM, else n_genes
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            num_support_views=3,  # 3 support views
            device=self.device,
        ).to(self.device)

        # Use ESM embeddings if available, otherwise use training adjacency
        if self.esm_embeddings is not None:
            features = self.esm_embeddings  # Already on device from __init__
        else:
            # Use TRAINING adjacency as features (matching TensorFlow)
            from slmgae_pytorch import sparse_to_torch_sparse

            features = sparse_to_torch_sparse(train_adj).to(self.device)

        # Convert adjacencies to tensors
        adjs_torch = []
        for i, adj in enumerate(adjs_normalized):
            if i < 3:  # Support views (sparse)
                if sp.issparse(adj):
                    indices = torch.tensor(np.vstack(adj.nonzero()),
                                           dtype=torch.long)
                    values = torch.tensor(adj.data, dtype=torch.float32)
                    shape = adj.shape
                else:
                    coo = sp.coo_matrix(adj)
                    indices = torch.tensor(np.vstack(coo.nonzero()),
                                           dtype=torch.long)
                    values = torch.tensor(coo.data, dtype=torch.float32)
                    shape = coo.shape
                adj_tensor = torch.sparse_coo_tensor(indices,
                                                     values,
                                                     shape,
                                                     dtype=torch.float32).to(
                                                         self.device)
            else:  # Main view (dense)
                adj_tensor = torch.tensor(adj,
                                          dtype=torch.float32).to(self.device)
            adjs_torch.append(adj_tensor)

        # Optimizer and loss (matching TensorFlow)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=0)
        loss_fn = SLMGAELoss(alpha=args.alpha, beta=args.beta)

        # Training loop
        best_val_auc = 0
        best_model_state = None

        for epoch in range(args.epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            # Forward pass through SLMGAE
            reconstructions, main_rec, att, support_recs = model(features,
                                                                 adjs_torch,
                                                                 coe=args.coe)

            # Compute loss (MSE, not BCE) - pass support_recs for individual view losses
            loss, loss_preds, loss_supp, loss_main = loss_fn(
                reconstructions, main_rec, att, support_recs, train_edges,
                train_labels)

            loss.backward()
            optimizer.step()

            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    reconstructions, _, _, _ = model(features,
                                                     adjs_torch,
                                                     coe=args.coe)

                    # Get predictions (NO sigmoid)
                    val_pred = reconstructions[val_edges[:, 0], val_edges[:,
                                                                          1]]
                    val_pred_np = val_pred.cpu().numpy()
                    val_labels_np = val_labels.cpu().numpy()

                    val_auc = roc_auc_score(val_labels_np, val_pred_np)
                    val_ap = average_precision_score(val_labels_np,
                                                     val_pred_np)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()

                if epoch % 50 == 0:
                    print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, "
                          f"Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")

        # Load best model
        model.load_state_dict(best_model_state)

        # Compute optimal threshold from validation set
        model.eval()
        with torch.no_grad():
            reconstructions, _, _, _ = model(features,
                                             adjs_torch,
                                             coe=args.coe)
            val_pred = reconstructions[val_edges[:, 0], val_edges[:, 1]]
            val_pred_np = val_pred.cpu().numpy()
            val_labels_np = val_labels.cpu().numpy()

            # Find optimal threshold from precision-recall curve
            precision, recall, thresholds = precision_recall_curve(
                val_labels_np, val_pred_np)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = (thresholds[optimal_idx]
                                 if optimal_idx < len(thresholds) else 0.5)

        print(f"Optimal F1 threshold from validation: {optimal_threshold:.4f}")

        return model, features, adjs_torch, optimal_threshold

    def evaluate_case_study(
        self,
        model,
        features,
        adjs_torch,
        case_pairs,
        edges_array,
        optimal_threshold=0.5,
    ):
        """Evaluate on case study pairs using optimal threshold from training"""
        print("\n" + "=" * 60)
        print("Case Study Evaluation (TensorFlow-Matching Model)")
        print(f"Using optimal threshold: {optimal_threshold:.4f}")
        print("=" * 60)

        # Create adjacency for checking ground truth
        adj = sp.coo_matrix(
            (np.ones(len(edges_array)),
             (edges_array[:, 0], edges_array[:, 1])),
            shape=(self.n_genes, self.n_genes),
            dtype=np.float32,
        )
        adj = adj.tocsr()

        model.eval()
        results = []

        with torch.no_grad():
            # Get full reconstruction matrix (use configured coe, not hardcoded)
            reconstructions, _, _, _ = model(features,
                                             adjs_torch,
                                             coe=self.args.coe)

            for pair in case_pairs:
                # Get prediction (NO sigmoid)
                pred_score = reconstructions[pair["idx1"], pair["idx2"]].item()

                # Check if this is actually an SL pair
                is_sl = adj[pair["idx1"], pair["idx2"]] > 0

                # Use optimal threshold for binary prediction
                pred_binary = "SL" if pred_score > optimal_threshold else "Non-SL"

                result = {
                    "gene1": pair["gene1"],
                    "gene2": pair["gene2"],
                    "predicted_score": pred_score,
                    "is_known_sl": is_sl,
                    "prediction": pred_binary,
                }
                results.append(result)

                status = ("✓" if (is_sl and pred_score > optimal_threshold) or
                          (not is_sl and pred_score <= optimal_threshold) else
                          "✗")
                print(
                    f"{status} {pair['gene1']:10} - {pair['gene2']:10}: "
                    f"Score={pred_score:.3f}, Actual={'SL' if is_sl else 'Non-SL'}, "
                    f"Pred={pred_binary}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Case study for SLMGAE (TensorFlow-matching)")
    parser.add_argument("--use_esm",
                        action="store_true",
                        help="Use ESM embeddings")
    parser.add_argument("--epochs",
                        type=int,
                        default=300,
                        help="Number of epochs")
    parser.add_argument("--hidden1",
                        type=int,
                        default=512,
                        help="Hidden layer 1 size")
    parser.add_argument("--hidden2",
                        type=int,
                        default=256,
                        help="Hidden layer 2 size")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="Dropout rate")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("--alpha",
                        type=float,
                        default=2.0,
                        help="Alpha for loss")
    parser.add_argument("--beta",
                        type=float,
                        default=4.0,
                        help="Beta for loss")
    parser.add_argument(
        "--coe",
        type=float,
        default=2.0,
        help=
        "Coefficient for attention combination (lambda in R = R_main + lambda*R_att)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../outputs/case_study_tensorflow_matching",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.use_esm:
        print(
            "ESM embeddings mode - will use ESM features instead of adjacency")
        args.output_dir = args.output_dir.replace("tensorflow_matching",
                                                  "tensorflow_matching_esm")

    # Set random seed
    set_seed(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    edges, genes, gene_to_idx = load_main_data()

    # Load support views
    support_views = load_support_views()

    # Prepare adjacencies (3 support + 1 main)
    adjs_normalized = prepare_adjacencies(edges, len(genes), support_views)

    # Get case study pairs
    case_pairs = get_case_study_pairs(gene_to_idx)
    print(f"\nEvaluating {len(case_pairs)} gene pairs")

    # Create evaluator (ESM loading happens in __init__ if use_esm=True)
    evaluator = CaseStudyEvaluator(edges, genes, gene_to_idx, args)

    # Train model (uses ESM features if available)
    model, features, adjs_torch, optimal_threshold = evaluator.train_model(
        adjs_normalized, args)

    # Evaluate on case study pairs
    results = evaluator.evaluate_case_study(model, features, adjs_torch,
                                            case_pairs, edges,
                                            optimal_threshold)

    # Calculate accuracy
    correct = sum(1 for r in results
                  if (r["is_known_sl"] and r["prediction"] == "SL") or (
                      not r["is_known_sl"] and r["prediction"] == "Non-SL"))
    accuracy = correct / len(results) if results else 0

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "SLMGAE_TensorFlow_Matching",
        "use_esm": args.use_esm,
        "n_genes": len(genes),
        "n_support_views": len(support_views),
        "n_case_pairs": len(case_pairs),
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "case_study_results": results,
        "architecture": {
            "branches": "4 (3 support + 1 main)",
            "layers": "GraphConvolutionSparse → GraphConvolution",
            "activation": "LeakyReLU",
            "attention": "Element-wise for support views",
            "decoder": "InnerProductDecoder with Z*W*Z^T",
            "loss": "MSE",
            "output": "No sigmoid",
        },
        "args": vars(args),
    }

    results_file = output_dir / "case_study_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("Case Study Complete (TensorFlow-Matching SLMGAE)!")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
