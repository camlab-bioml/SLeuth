#!/usr/bin/env python3
"""
Legacy ad-hoc SLMGAE run on a TOY gene subset. NOT the TensorFlow BC experiment.

The original BC run (original_tensorflow_code/train_BC.py, utils.load_BC_data,
models.SLMGAE_BC) uses 332 breast-cancer genes, 5 pre-built support views
(BC_data/F1_F2_*_for_train), the given BC_data/pos|neg_edge_binary.npy and a
6-branch model. This script instead takes the first 139 (alphabetical) genes
of List_Proteins_in_SL.txt, 3 support views sliced from the main dataset
(weighted GO BP / GO CC similarity without kNN, BioGRID PPI), the 4-branch
SLMGAE and random 1:10 negatives. What it does follow from the TF BC: the main (SL) view of each fold
is built from that fold's training positives only, and the held-out fold is
scored once, with the final model. Not used by the benchmark.
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

# The 4-branch SLMGAE (the main-dataset model, not SLMGAE_BC)
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
    """Load the toy subset: the first 139 lines of List_Proteins_in_SL.txt.

    Built in memory on every run; nothing is written under ../data/. The
    support views are sliced [:139, :139] from the List_Proteins-indexed
    views, so the subset must be exactly these leading rows.
    """
    print("Loading toy gene subset...")

    main_genes_file = Path("../data/List_Proteins_in_SL.txt")
    with open(main_genes_file, "r") as f:
        all_genes = [line.strip() for line in f if line.strip()]

    # Arbitrary toy subset: the first 139 genes of the alphabetical list
    # (A2M, A2ML1, ...), not a breast-cancer gene set
    bc_genes = all_genes[:139]

    print(f"Toy subset: {len(bc_genes)} genes")

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
    print(f"Loaded {len(edges)//2} SL pairs for the toy subset")

    return edges, bc_genes, gene_to_idx


def load_support_views(n_genes):
    """Load 3 support views (GO similarity weighted, no kNN), sliced to the toy subset.

    The TF BC instead np.loadtxt()s 5 pre-built BC_data matrices; these are the
    main dataset's GO BP / GO CC / PPI views.
    """
    print("Loading support views...")

    from slmgae_pytorch import DataLoader as SLDataLoader

    loader = SLDataLoader(data_path="../data/", nn_size=45)
    loader.num_nodes = 6375  # Full dataset size

    support_views = []

    # Load GO BP WITHOUT KNN
    # binarize=False keeps the weighted similarity this BC path has always
    # used; without kNN, array2coo would turn it into a near-complete graph.
    gosim_bp = loader.load_dense_feature("../data/Human_GOsim.txt",
                                         knn=False,
                                         binarize=False)
    gosim_bp_dense = gosim_bp.toarray()[:n_genes, :n_genes]
    support_views.append(gosim_bp_dense)
    print(f"  Loaded GO BP similarity: {gosim_bp_dense.shape}")

    # Load GO CC WITHOUT KNN
    gosim_cc = loader.load_dense_feature("../data/Human_GOsim_CC.txt",
                                         knn=False,
                                         binarize=False)
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


def normalize_adj(adj):
    """Normalize adjacency matrix: D^(-1/2) * (A + I) * D^(-1/2)"""
    adj = adj + np.eye(adj.shape[0])  # Add self-loops
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def prepare_adjacencies(support_views):
    """Normalise the 3 support views (sparse, for the first layer).

    The main (SL) view is NOT built here: it depends on the fold (see
    build_main_view), and a view from all SL edges would put every fold's
    held-out positives into the graph the main branch propagates over.
    """
    return [
        sp.coo_matrix(normalize_adj(support_adj), dtype=np.float32)
        for support_adj in support_views
    ]


def build_main_view(pos_train_edges, n_genes):
    """Normalised main view of ONE fold, from its TRAINING positives only.

    Mirrors original_tensorflow_code/train_BC.py:70-76 (adj from
    pos_edge[train_pos], + adj.T). Binary, symmetric, zero diagonal; dense,
    for the main branch.
    """
    main_adj = np.zeros((n_genes, n_genes), dtype=np.float32)
    if len(pos_train_edges) > 0:
        main_adj[pos_train_edges[:, 0], pos_train_edges[:, 1]] = 1
        main_adj[pos_train_edges[:, 1], pos_train_edges[:, 0]] = 1
    np.fill_diagonal(main_adj, 0)
    return normalize_adj(main_adj).astype(np.float32)


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
                   support_adjs, args):
        """Train one fold; score its held-out split (val_*) once, at the end.

        val_edges / val_labels are the KFold HELD-OUT split. They never drive
        training or selection: the model after the last epoch is scored once,
        as the TF BC does (train_BC.py:168-172).
        """

        # Build adjacency from POSITIVE training edges for features (matching TensorFlow)
        pos_train_edges = train_edges[train_labels == 1]
        # [support1, support2, support3, main]; main from this fold's training
        # positives only
        adjs_normalized = list(support_adjs) + [
            build_main_view(pos_train_edges, self.n_genes)
        ]
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

        # Create the 4-branch SLMGAE model
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

        # Training loop (no selection: nothing is scored until it ends)
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

            if epoch % args.eva_epochs == 0:
                print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}")

        # Score the held-out split ONCE, with the final model. Keeping the best
        # of several held-out evaluations would report a maximum chosen on the
        # split being reported.
        model.eval()
        with torch.no_grad():
            reconstructions, _, _, _ = model(features, adjs_torch, coe=args.coe)
            # No sigmoid (matching TensorFlow)
            val_pred = reconstructions[val_edges[:, 0],
                                       val_edges[:, 1]].cpu().numpy()
        val_labels_cpu = val_labels.cpu().numpy()
        val_auc = roc_auc_score(val_labels_cpu, val_pred)
        val_ap = average_precision_score(val_labels_cpu, val_pred)
        # F1 score with optimal threshold (not fixed 0.5)
        val_f1 = calculate_optimal_f1(val_labels_cpu, val_pred)
        print(f"Held-out (final model): AUC={val_auc:.4f}, AP={val_ap:.4f}, "
              f"F1={val_f1:.4f}")

        return val_auc, val_ap

    def run_cv(self, pos_edges, neg_edges, support_adjs, args):
        """Run cross-validation (val_* below = each fold's held-out split)"""
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
                                      val_labels, support_adjs, args)
            fold_aucs.append(auc)
            fold_aps.append(ap)
            print(f"Fold {fold+1} - AUC: {auc:.4f}, AP: {ap:.4f}")

        return fold_aucs, fold_aps


def main():
    parser = argparse.ArgumentParser(
        description="Legacy toy SLMGAE run on the first 139 List_Proteins "
        "genes (NOT the TensorFlow BC experiment)")
    parser.add_argument("--dataset",
                        type=str,
                        default="toy_first139_List_Proteins",
                        help="Label recorded in the results file")
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of epochs")
    parser.add_argument("--eva_epochs",
                        type=int,
                        default=100,
                        help="Print the training loss every N epochs")
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../outputs/toy_first139",
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

    # Normalise the 3 support views (the main view is built per fold)
    support_adjs = prepare_adjacencies(support_views)

    # Prepare CV data
    pos_edges, neg_edges = prepare_cv_data(edges, n_genes, args.cv_folds)

    # Create trainer
    trainer = BCTrainer(genes)

    # Run cross-validation
    print("\nStarting cross-validation training (toy subset, 4-branch "
          "SLMGAE)...")
    print("Model configuration:")
    print(f"  - 4 parallel branches (3 support + 1 main)")
    print(f"  - GraphConvolutionSparse → GraphConvolution")
    print(f"  - LeakyReLU activation")
    print(f"  - MSE loss (not BCE)")
    print(f"  - No sigmoid on predictions")

    fold_aucs, fold_aps = trainer.run_cv(pos_edges, neg_edges, support_adjs,
                                         args)

    # Save results
    results = {
        "dataset": args.dataset,
        "model": "SLMGAE_4branch_toy",
        "note": ("Toy subset (first 139 List_Proteins genes, 3 views, random "
                 "1:10 negatives); NOT the TF BC experiment. Fold metrics = "
                 "final model on the held-out fold."),
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
    print("Toy-subset training complete (4-branch SLMGAE)!")
    print(f"Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"Mean AP: {results['mean_ap']:.4f} ± {results['std_ap']:.4f}")
    print(f"Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
