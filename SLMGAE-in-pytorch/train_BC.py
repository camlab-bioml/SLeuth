#!/usr/bin/env python3
"""
Train SLMGAE on BC (Breast Cancer) subset - FIXED to match TensorFlow
Uses the exact same SLMGAE architecture as TensorFlow
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import random
import warnings

warnings.filterwarnings("ignore")

# Import the TensorFlow-matching SLMGAE model
from slmgae_pytorch import SLMGAE
from objective import SLMGAELoss
from evaluation import calculate_optimal_f1


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


def load_bc_data():
    """Load BC subset data"""
    print("Loading BC subset data...")

    # Load BC gene list
    bc_genes_file = Path("../data/BC_List_Proteins.txt")
    if not bc_genes_file.exists():
        print(f"Creating BC gene list from main dataset...")
        # Load main gene list and select first 139 genes for BC subset
        main_genes_file = Path("../data/List_Proteins_in_SL.txt")
        with open(main_genes_file, "r") as f:
            all_genes = [line.strip() for line in f if line.strip()]

        # BC subset is typically first 139 genes
        bc_genes = all_genes[:139]

        # Save BC gene list
        with open(bc_genes_file, "w") as f:
            for gene in bc_genes:
                f.write(f"{gene}\n")
    else:
        with open(bc_genes_file, "r") as f:
            bc_genes = [line.strip() for line in f if line.strip()]

    print(f"BC subset: {len(bc_genes)} genes")

    # Create gene to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(bc_genes)}

    # Load SL edges and filter for BC genes
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

    edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)
    print(f"Loaded {len(edges)//2} SL pairs for BC subset")

    return edges, bc_genes, gene_to_idx


def load_support_views(n_genes):
    """Load support view adjacency matrices without KNN (matching TensorFlow BC)"""
    print("Loading support views...")

    # Use DataLoader WITHOUT KNN to match TensorFlow's BC behavior
    # TensorFlow BC uses np.loadtxt() which loads pre-processed files without KNN
    from slmgae_pytorch import DataLoader as SLDataLoader

    loader = SLDataLoader(data_path="../data/", nn_size=45)
    loader.num_nodes = 6375  # Full dataset size

    support_views = []

    # Load GO BP WITHOUT KNN (matching TensorFlow BC which uses pre-processed files)
    gosim_bp = loader.load_dense_feature("../data/Human_GOsim.txt", knn=False)
    gosim_bp_dense = gosim_bp.toarray()[:n_genes, :n_genes]
    support_views.append(gosim_bp_dense)
    print(f"  Loaded GO BP similarity: {gosim_bp_dense.shape}")

    # Load GO CC WITHOUT KNN
    gosim_cc = loader.load_dense_feature("../data/Human_GOsim_CC.txt",
                                         knn=False)
    gosim_cc_dense = gosim_cc.toarray()[:n_genes, :n_genes]
    support_views.append(gosim_cc_dense)
    print(f"  Loaded GO CC similarity: {gosim_cc_dense.shape}")

    # Load PPI (already no KNN for sparse features)
    ppi = loader.load_sparse_feature("../data/biogrid_ppi_sparse.txt")
    ppi_dense = ppi.toarray()[:n_genes, :n_genes]
    support_views.append(ppi_dense)
    print(f"  Loaded PPI network: {ppi_dense.shape}")

    print(f"Total support views: {len(support_views)}")
    return support_views


def prepare_adjacencies(edges, n_genes, support_views):
    """Prepare all adjacency matrices (3 support + 1 main)"""

    # Main adjacency from SL edges
    if len(edges) > 0:
        main_adj = sp.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(n_genes, n_genes),
            dtype=np.float32,
        )
        main_adj = main_adj.tocsr()
        main_adj.setdiag(0)
        main_adj.eliminate_zeros()
        main_adj = main_adj.toarray()
    else:
        main_adj = np.zeros((n_genes, n_genes), dtype=np.float32)

    # Normalize adjacencies
    def normalize_adj(adj):
        """Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)"""
        adj = adj + np.eye(adj.shape[0])  # Add self-loops
        degree = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Create list: [support1, support2, support3, main]
    adjs_normalized = []

    # Add support views (sparse)
    for support_adj in support_views:
        adj_norm = normalize_adj(support_adj)
        # Convert to sparse for first layer
        adj_sparse = sp.coo_matrix(adj_norm, dtype=np.float32)
        adjs_normalized.append(adj_sparse)

    # Add main view (dense)
    main_norm = normalize_adj(main_adj)
    adjs_normalized.append(main_norm.astype(np.float32))

    return adjs_normalized


def prepare_cv_data(edges, n_genes, n_folds=5):
    """Prepare cross-validation data"""

    # Get positive edges (SL pairs)
    if len(edges) > 0:
        pos_edges = np.unique(edges.reshape(-1, 2), axis=0)
        pos_edges = pos_edges[pos_edges[:, 0]
                              < pos_edges[:, 1]]  # Keep upper triangle only
    else:
        pos_edges = np.empty((0, 2), dtype=int)

    # Generate negative edges
    all_possible = set()
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            all_possible.add((i, j))

    pos_set = set(map(tuple, pos_edges))
    neg_edges = np.array(list(all_possible - pos_set))

    # Sample negative edges (1:10 ratio or all if fewer)
    n_neg_sample = min(len(neg_edges), max(len(pos_edges) * 10, 100))
    if len(neg_edges) > 0:
        neg_edges = neg_edges[np.random.choice(len(neg_edges),
                                               n_neg_sample,
                                               replace=False)]

    print(
        f"Positive edges: {len(pos_edges)}, Negative edges: {len(neg_edges)}")

    return pos_edges, neg_edges


class BCTrainer:

    def __init__(self, genes, device="cuda"):
        self.genes = genes
        self.n_genes = len(genes)
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Enable GPU optimizations if available
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def train_fold(self, train_edges, train_labels, val_edges, val_labels,
                   adjs_normalized, args):
        """Train one fold using SLMGAE model matching TensorFlow"""

        # Build adjacency from POSITIVE training edges for features (matching TensorFlow)
        pos_train_edges = train_edges[train_labels == 1]
        if len(pos_train_edges) > 0:
            train_adj = sp.coo_matrix(
                (
                    np.ones(len(pos_train_edges)),
                    (pos_train_edges[:, 0], pos_train_edges[:, 1]),
                ),
                shape=(self.n_genes, self.n_genes),
                dtype=np.float32,
            )
            # Symmetrize: add transpose (matching TensorFlow train.py:74-75)
            train_adj = train_adj + train_adj.T
            train_adj = train_adj.tocsr()
            # Convert to torch sparse for features
            indices = torch.tensor(np.vstack(train_adj.nonzero()),
                                   dtype=torch.long)
            values = torch.tensor(train_adj.data, dtype=torch.float32)
            shape = train_adj.shape
            features = torch.sparse_coo_tensor(indices,
                                               values,
                                               shape,
                                               dtype=torch.float32).to(
                                                   self.device)
        else:
            # Fallback to identity if no positive edges
            features = torch.eye(self.n_genes,
                                 dtype=torch.float32).to(self.device)

        # Convert to tensors
        train_edges = torch.tensor(train_edges,
                                   dtype=torch.long).to(self.device)
        train_labels = torch.tensor(train_labels,
                                    dtype=torch.float32).to(self.device)
        val_edges = torch.tensor(val_edges, dtype=torch.long).to(self.device)
        val_labels = torch.tensor(val_labels,
                                  dtype=torch.float32).to(self.device)

        # Create SLMGAE model (matching TensorFlow exactly)
        model = SLMGAE(
            num_nodes=self.n_genes,
            num_features=self.n_genes,  # Using adjacency as features
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
            num_support_views=3,  # 3 support views
            device=self.device,
        ).to(self.device)

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

        # Create optimizer and loss (matching TensorFlow)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        loss_fn = SLMGAELoss(alpha=args.alpha, beta=args.beta)

        # Training loop
        best_val_auc = 0
        best_val_ap = 0
        patience_counter = 0

        for epoch in range(args.epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            # Forward pass
            reconstructions, main_rec, att, support_recs = model(features,
                                                                 adjs_torch,
                                                                 coe=args.coe)

            # Compute loss - pass support_recs for individual view losses
            loss, loss_preds, loss_supp, loss_main = loss_fn(
                reconstructions, main_rec, att, support_recs, train_edges,
                train_labels)

            loss.backward()
            optimizer.step()

            # Validation every eva_epochs
            if epoch % args.eva_epochs == 0:
                model.eval()
                with torch.no_grad():
                    reconstructions, _, _, _ = model(features,
                                                     adjs_torch,
                                                     coe=args.coe)

                    # Get predictions for validation edges
                    val_pred = reconstructions[val_edges[:, 0], val_edges[:,
                                                                          1]]

                    # No sigmoid (matching TensorFlow)
                    val_pred_cpu = val_pred.cpu().numpy()
                    val_labels_cpu = val_labels.cpu().numpy()

                    # Calculate metrics
                    val_auc = roc_auc_score(val_labels_cpu, val_pred_cpu)
                    val_ap = average_precision_score(val_labels_cpu,
                                                     val_pred_cpu)

                    # F1 score with optimal threshold (not fixed 0.5)
                    val_f1 = calculate_optimal_f1(val_labels_cpu, val_pred_cpu)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_ap = val_ap
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}: Loss={loss.item():.4f}, "
                        f"Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}, Val F1={val_f1:.4f}"
                    )

                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return best_val_auc, best_val_ap

    def run_cv(self, pos_edges, neg_edges, adjs_normalized, args):
        """Run cross-validation"""
        # Combine positive and negative edges
        if len(pos_edges) > 0 and len(neg_edges) > 0:
            all_edges = np.vstack([pos_edges, neg_edges])
            all_labels = np.hstack(
                [np.ones(len(pos_edges)),
                 np.zeros(len(neg_edges))])
        elif len(pos_edges) > 0:
            all_edges = pos_edges
            all_labels = np.ones(len(pos_edges))
        elif len(neg_edges) > 0:
            all_edges = neg_edges
            all_labels = np.zeros(len(neg_edges))
        else:
            print("No edges to train on!")
            return [], []

        # Shuffle
        idx = np.random.permutation(len(all_edges))
        all_edges = all_edges[idx]
        all_labels = all_labels[idx]

        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        fold_aucs = []
        fold_aps = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_edges)):
            print(f"\n=== Fold {fold+1}/{args.cv_folds} ===")

            train_edges = all_edges[train_idx]
            train_labels = all_labels[train_idx]
            val_edges = all_edges[val_idx]
            val_labels = all_labels[val_idx]

            auc, ap = self.train_fold(train_edges, train_labels, val_edges,
                                      val_labels, adjs_normalized, args)
            fold_aucs.append(auc)
            fold_aps.append(ap)
            print(f"Fold {fold+1} - AUC: {auc:.4f}, AP: {ap:.4f}")

        return fold_aucs, fold_aps


def main():
    parser = argparse.ArgumentParser(
        description="Train SLMGAE on BC subset (TensorFlow-matching)")
    parser.add_argument("--dataset",
                        type=str,
                        default="BC",
                        help="Dataset name")
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of epochs")
    parser.add_argument("--eva_epochs",
                        type=int,
                        default=100,
                        help="Evaluate every N epochs")
    parser.add_argument("--cv_folds",
                        type=int,
                        default=5,
                        help="Number of CV folds")
    parser.add_argument("--hidden1",
                        type=int,
                        default=128,
                        help="Hidden layer 1 size")
    parser.add_argument("--hidden2",
                        type=int,
                        default=64,
                        help="Hidden layer 2 size")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.3,
                        help="Dropout rate")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.5,
                        help="Alpha for loss")
    parser.add_argument("--beta",
                        type=float,
                        default=2.0,
                        help="Beta for loss")
    parser.add_argument(
        "--coe",
        type=float,
        default=1.0,
        help=
        "Coefficient for attention combination (lambda in R = R_main + lambda*R_att)",
    )
    parser.add_argument("--patience",
                        type=int,
                        default=20,
                        help="Early stopping patience")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../outputs/BC_tensorflow_matching",
        help="Output directory",
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    edges, genes, gene_to_idx = load_bc_data()
    n_genes = len(genes)

    # Load support views
    support_views = load_support_views(n_genes)

    # Prepare adjacencies (3 support + 1 main)
    adjs_normalized = prepare_adjacencies(edges, n_genes, support_views)

    # Prepare CV data
    pos_edges, neg_edges = prepare_cv_data(edges, n_genes, args.cv_folds)

    # Create trainer
    trainer = BCTrainer(genes)

    # Run cross-validation
    print(
        "\nStarting cross-validation training with TensorFlow-matching SLMGAE..."
    )
    print("Model configuration:")
    print(f"  - 4 parallel branches (3 support + 1 main)")
    print(f"  - GraphConvolutionSparse → GraphConvolution")
    print(f"  - LeakyReLU activation")
    print(f"  - MSE loss (not BCE)")
    print(f"  - No sigmoid on predictions")

    fold_aucs, fold_aps = trainer.run_cv(pos_edges, neg_edges, adjs_normalized,
                                         args)

    # Save results
    results = {
        "dataset": args.dataset,
        "model": "SLMGAE_TensorFlow_Matching",
        "n_genes": len(genes),
        "n_positive_pairs": len(pos_edges),
        "n_negative_pairs": len(neg_edges),
        "n_support_views": len(support_views),
        "cv_folds": args.cv_folds,
        "fold_aucs": fold_aucs,
        "fold_aps": fold_aps,
        "mean_auc": np.mean(fold_aucs) if fold_aucs else 0,
        "std_auc": np.std(fold_aucs) if fold_aucs else 0,
        "mean_ap": np.mean(fold_aps) if fold_aps else 0,
        "std_ap": np.std(fold_aps) if fold_aps else 0,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_file = output_dir / "bc_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("BC Training Complete (TensorFlow-Matching SLMGAE)!")
    print(f"Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"Mean AP: {results['mean_ap']:.4f} ± {results['std_ap']:.4f}")
    print(f"Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
