#!/usr/bin/env python3
"""
Training script for Adamson GNN with edge features.

- Conv matrix: used for graph convolution (e.g., PPI)
- Target matrix: what we predict (e.g., gamma)
- Edge features: all other matrices, combined via learnable weights

Usage:
    python train.py --conv ppi --target gamma
    # Uses go_bp + go_cc as edge features
"""

import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import pearsonr, spearmanr

from model import GNN, AdamsonData

# =============================================================================
# Hyperparameters
# =============================================================================
CONV_MATRIX = 'ppi'       # Matrix for graph convolution
TARGET_MATRIX = 'gamma'   # Matrix to predict
HIDDEN_DIM = 128          # Hidden layer dimension
EMBED_DIM = 64            # Embedding dimension
DROPOUT = 0.2             # Dropout rate
LEARNING_RATE = 0.001     # Adam learning rate
EPOCHS = 200              # Training epochs
FOLDS = 5                 # K-fold cross-validation
TEST_RATIO = 0.1          # Hold-out test set ratio
SEED = 42                 # Random seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--conv', default=CONV_MATRIX, choices=['gamma', 'go_bp', 'go_cc', 'ppi'],
                   help='Matrix for graph convolution')
    p.add_argument('--target', default=TARGET_MATRIX, choices=['gamma', 'go_bp', 'go_cc', 'ppi'],
                   help='Matrix to predict')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--lr', type=float, default=LEARNING_RATE)
    p.add_argument('--hidden', type=int, default=HIDDEN_DIM)
    p.add_argument('--embed', type=int, default=EMBED_DIM)
    p.add_argument('--dropout', type=float, default=DROPOUT)
    p.add_argument('--folds', type=int, default=FOLDS)
    p.add_argument('--test_ratio', type=float, default=TEST_RATIO,
                   help='Hold-out test set ratio (0 to disable)')
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, opt, x, adj, edge_features, target, edge_indices):
    """Train on specified edges only."""
    model.train()
    opt.zero_grad()
    pred = model(x, adj, edge_features)

    # Extract predictions and targets for training edges
    pred_vals = pred[edge_indices[:, 0], edge_indices[:, 1]]
    true_vals = target[edge_indices[:, 0], edge_indices[:, 1]]

    loss = torch.nn.functional.mse_loss(pred_vals, true_vals)
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, adj, edge_features, target_full, edge_indices):
    """Evaluate on specified edges using FULL (unmasked) target values."""
    model.eval()
    pred = model(x, adj, edge_features)

    pred_vals = pred[edge_indices[:, 0], edge_indices[:, 1]].cpu().numpy()
    true_vals = target_full[edge_indices[:, 0], edge_indices[:, 1]].cpu().numpy()

    return {
        'mse': float(np.mean((pred_vals - true_vals) ** 2)),
        'pearson': float(pearsonr(pred_vals, true_vals)[0]),
        'spearman': float(spearmanr(pred_vals, true_vals)[0]),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data
    data = AdamsonData()
    device = data.device

    # Node features: identity (one-hot)
    x = data.get_node_features()
    input_dim = x.shape[1]

    # Edge features (all matrices except conv and target)
    edge_feature_names = data.get_edge_feature_names(args.conv, args.target)
    edge_features = data.get_edge_features(args.conv, args.target)
    n_edge_features = len(edge_features)

    # Get observed edges for target matrix (upper triangle only)
    all_edges = data.get_observed_edges(args.target)
    n_edges = len(all_edges)

    # Full target for evaluation (original values)
    target_full = data.get_target(args.target)

    # Print configuration
    print(f"Device: {device}")
    print(f"Nodes: {data.num_nodes}, Observed edges: {n_edges}")
    print(f"Node features: identity (dim={input_dim})")
    print(f"Conv matrix: {args.conv}")
    print(f"Target matrix: {args.target}")
    print(f"Edge features: {edge_feature_names} (n={n_edge_features})")

    # Train/test split on edges
    if args.test_ratio > 0:
        train_val_idx, test_idx = train_test_split(
            np.arange(n_edges), test_size=args.test_ratio, random_state=args.seed
        )
        train_val_edges = all_edges[train_val_idx]
        test_edges = all_edges[test_idx]
        print(f"Train/Val edges: {len(train_val_edges)}, Test edges: {len(test_edges)}")
    else:
        train_val_edges = all_edges
        test_edges = np.array([]).reshape(0, 2)
        print(f"All edges used for training (no test set)")

    print(f"{'='*50}")

    # K-fold CV
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv_results = []
    best_model_state = None
    best_val_pearson = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_edges)):
        train_edges = train_val_edges[train_idx]
        val_edges = train_val_edges[val_idx]

        # Edges to mask from target (test + val)
        target_mask_edges = np.vstack([test_edges, val_edges]) if len(test_edges) > 0 else val_edges

        # For adjacency: only mask if conv == target (to prevent information leakage)
        if args.conv == args.target:
            adj = data.get_adj_with_mask(args.conv, target_mask_edges)
        else:
            adj = data.get_adj_with_mask(args.conv, None)

        # Target always needs masking during training
        target_masked = data.get_target_with_mask(args.target, target_mask_edges)

        # Convert train edges to tensor for indexing
        train_edges_t = torch.tensor(train_edges, dtype=torch.long, device=device)
        val_edges_t = torch.tensor(val_edges, dtype=torch.long, device=device)

        model = GNN(input_dim, args.hidden, args.embed, args.dropout, n_edge_features).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            loss = train_one_epoch(model, opt, x, adj, edge_features, target_masked, train_edges_t)
            if (epoch + 1) % 50 == 0:
                m = evaluate(model, x, adj, edge_features, target_full, val_edges_t)
                print(f"Fold {fold+1} Epoch {epoch+1}: loss={loss:.6f}, val_r={m['pearson']:.4f}")

        m = evaluate(model, x, adj, edge_features, target_full, val_edges_t)
        cv_results.append(m)
        print(f"Fold {fold+1} Final: MSE={m['mse']:.6f}, r={m['pearson']:.4f}, rho={m['spearman']:.4f}")

        # Print learned edge feature weights (if any)
        if n_edge_features > 0:
            weights = model.decoder.edge_weights.detach().cpu().numpy()
            weight_str = ", ".join([f"{name}={w:.4f}" for name, w in zip(edge_feature_names, weights)])
            print(f"  Edge weights: {weight_str}")
        print()

        if m['pearson'] > best_val_pearson:
            best_val_pearson = m['pearson']
            best_model_state = model.state_dict().copy()

    # CV Summary
    print(f"{'='*50}")
    print("CV Results:")
    for k in ['mse', 'pearson', 'spearman']:
        vals = [r[k] for r in cv_results]
        print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Test evaluation (using best model)
    if len(test_edges) > 0 and best_model_state is not None:
        print(f"\n{'='*50}")
        print("Test Set Evaluation (best model):")

        if args.conv == args.target:
            adj_test = data.get_adj_with_mask(args.conv, test_edges)
        else:
            adj_test = data.get_adj_with_mask(args.conv, None)

        test_edges_t = torch.tensor(test_edges, dtype=torch.long, device=device)

        model = GNN(input_dim, args.hidden, args.embed, args.dropout, n_edge_features).to(device)
        model.load_state_dict(best_model_state)

        test_m = evaluate(model, x, adj_test, edge_features, target_full, test_edges_t)
        print(f"  MSE: {test_m['mse']:.6f}")
        print(f"  Pearson: {test_m['pearson']:.4f}")
        print(f"  Spearman: {test_m['spearman']:.4f}")

        # Print final edge feature weights (if any)
        if n_edge_features > 0:
            weights = model.decoder.edge_weights.detach().cpu().numpy()
            print(f"\nLearned edge feature weights:")
            for name, w in zip(edge_feature_names, weights):
                print(f"  {name}: {w:.4f}")

        # Final summary
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Configuration:")
        print(f"  Conv matrix: {args.conv}")
        print(f"  Target matrix: {args.target}")
        print(f"  Edge features: {edge_feature_names}")
        print(f"\nTest Set Metrics:")
        print(f"  MSE:      {test_m['mse']:.6f}")
        print(f"  Pearson:  {test_m['pearson']:.4f}")
        print(f"  Spearman: {test_m['spearman']:.4f}")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()
