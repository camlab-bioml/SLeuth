#!/usr/bin/env python3
"""
Training script for Graph Transformer with weighted support views.
Implements independent linearization approach (data augmentation).
"""

import os
import argparse
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from graph_transformer import GraphTransformer
from data_utils import (load_and_split_sldb_data, build_graph_from_train_data,
                        GraphLinearizer, prepare_support_views,
                        create_edge_mapping, load_esm_embeddings,
                        load_esm_gene_order, filter_training_data_by_genes)
from load_slmgae_data import load_slmgae_data, build_adjacency_from_edges
from loss_functions import get_loss_function, compute_class_weights


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def convert_to_sl_labels(df: pd.DataFrame,
                         sl_percentage: float = 0.01) -> pd.DataFrame:
    """
    Convert sensitivity scores to binary SL labels.
    Top sl_percentage most negative scores are labeled as SL.
    """
    df = df.copy()

    # Ensure numeric scores
    df['sens.score'] = pd.to_numeric(df['sens.score'], errors='coerce')
    df = df.dropna(subset=['sens.score'])

    if len(df) == 0:
        raise ValueError("No valid sensitivity scores found in the data")

    # Sort by sensitivity score (most negative first)
    df = df.sort_values('sens.score')
    df['rank'] = range(1, len(df) + 1)

    # Label top percentage as SL
    sl_threshold = int(len(df) * sl_percentage)
    df['sl_label'] = (df['rank'] <= sl_threshold).astype(float)

    print(
        f"SL pairs: {df['sl_label'].sum():.0f}/{len(df)} ({df['sl_label'].mean()*100:.2f}%)"
    )

    return df


def prepare_edge_tensors(
        df: pd.DataFrame,
        gene_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dataframe edges to tensor format."""
    edges = []
    labels = []

    for _, row in df.iterrows():
        if row['gene1'] in gene_to_idx and row['gene2'] in gene_to_idx:
            i = gene_to_idx[row['gene1']]
            j = gene_to_idx[row['gene2']]
            edges.append([i, j])
            labels.append(row['sl_label'])

    edge_tensor = torch.tensor(edges, dtype=torch.long).t()  # (2, n_edges)
    label_tensor = torch.tensor(labels, dtype=torch.float)  # (n_edges,)

    return edge_tensor, label_tensor


def train_epoch(model: GraphTransformer, node_features: torch.Tensor,
                linearizations: List[List[int]], support_views: torch.Tensor,
                edge_map: Dict, train_edges: torch.Tensor,
                train_labels: torch.Tensor, optimizer: optim.Optimizer,
                criterion: nn.Module, batch_size: int,
                device: torch.device) -> float:
    """Train one epoch via SGD: θ←θ-η∇L where L=𝔼_σ[ℒ(f_θ(·|σ),y)]."""
    model.train()

    # Shuffle training edges
    n_edges = train_edges.shape[1]
    indices = torch.randperm(n_edges, device=device)

    total_loss = 0.0
    n_batches = 0

    # Process in batches
    pbar = tqdm(range(0, n_edges, batch_size), desc="Training")
    for i in pbar:
        # Get batch
        batch_indices = indices[i:i + batch_size]
        batch_edges = train_edges[:, batch_indices]
        batch_labels = train_labels[batch_indices]

        # Randomly sample a linearization for this batch
        linearization = random.choice(linearizations)

        # Forward pass - use model's registered node features if learnable
        model_node_features = getattr(model, 'node_features', node_features)
        predictions = model(node_features=model_node_features,
                            linearization=linearization,
                            support_views=support_views,
                            edge_map=edge_map,
                            query_edges=batch_edges)

        # Loss: L = ℒ(logits, y) where ℒ is BCE/focal loss
        loss = criterion(predictions, batch_labels)

        # Backprop: compute ∇_θ L
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: ||∇||_2 ≤ 1.0 for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update: θ ← θ - η∇L (or Adam variant)
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / n_batches


def evaluate(model: GraphTransformer,
             node_features: torch.Tensor,
             linearizations: List[List[int]],
             support_views: torch.Tensor,
             edge_map: Dict,
             eval_edges: torch.Tensor,
             eval_labels: torch.Tensor,
             n_samples: int = 10,
             device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate via Monte Carlo: p(y=1|e) ≈ (1/K)Σ_k σ(f(e|σ_k)) for K sampled σ.
    """
    model.eval()

    with torch.no_grad():
        # Sample linearizations for evaluation
        if len(linearizations) <= n_samples:
            sampled_lins = linearizations
        else:
            sampled_lins = random.sample(linearizations, n_samples)

        # Collect predictions from each linearization
        all_predictions = []
        # Use model's registered node features if learnable
        model_node_features = getattr(model, 'node_features', node_features)
        for linearization in sampled_lins:
            preds = model(node_features=model_node_features,
                          linearization=linearization,
                          support_views=support_views,
                          edge_map=edge_map,
                          query_edges=eval_edges)
            all_predictions.append(torch.sigmoid(preds))

        # MC estimate: ŷ = (1/K)Σ_k p_k where p_k = σ(f(e|σ_k))
        avg_predictions = torch.stack(all_predictions).mean(dim=0)

        # Move to CPU for metrics calculation
        preds_cpu = avg_predictions.cpu().numpy()
        labels_cpu = eval_labels.cpu().numpy()

        # Calculate metrics
        metrics = {}

        # ROC-AUC
        try:
            metrics['auc'] = roc_auc_score(labels_cpu, preds_cpu)
        except ValueError as e:
            # This happens when only one class is present
            print(f"Warning: ROC-AUC calculation failed: {e}")
            metrics['auc'] = 0.5

        # Average Precision
        try:
            metrics['avg_precision'] = average_precision_score(
                labels_cpu, preds_cpu)
        except ValueError as e:
            # This happens when only one class is present
            print(f"Warning: Average Precision calculation failed: {e}")
            metrics['avg_precision'] = labels_cpu.mean()

        # Accuracy: acc = 𝔼[𝟙(ŷ>0.5) = y]
        binary_preds = (preds_cpu > 0.5).astype(float)
        metrics['accuracy'] = (binary_preds == labels_cpu).mean()

        # Precision and recall at 0.5 threshold
        true_positives = ((binary_preds == 1) & (labels_cpu == 1)).sum()
        false_positives = ((binary_preds == 1) & (labels_cpu == 0)).sum()
        false_negatives = ((binary_preds == 0) & (labels_cpu == 1)).sum()

        # Precision: P = TP/(TP+FP), Recall: R = TP/(TP+FN)
        metrics['precision'] = true_positives / (true_positives +
                                                 false_positives + 1e-8)
        metrics['recall'] = true_positives / (true_positives +
                                              false_negatives + 1e-8)
        # F1: harmonic mean = 2PR/(P+R)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall'] + 1e-8)

        return metrics


def plot_training_history(history: Dict, output_dir: str):
    """Plot and save training curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # ROC-AUC
    val_aucs = [m['auc'] for m in history['val_metrics']]
    axes[0, 1].plot(epochs, val_aucs, 'g-', label='Val AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('Validation ROC-AUC')
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    val_f1s = [m['f1'] for m in history['val_metrics']]
    axes[1, 0].plot(epochs, val_f1s, 'r-', label='Val F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(True, alpha=0.3)

    # Support view weights over time
    if 'support_view_weights' in history:
        weights_over_time = np.array(history['support_view_weights'])
        for i in range(weights_over_time.shape[1]):
            axes[1, 1].plot(epochs,
                            weights_over_time[:, i],
                            label=f'View {i+1}',
                            alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight (α²)')
        axes[1, 1].set_title('Support View Weights')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()


def main(args):
    """Main training function."""
    # Setup
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    set_random_seeds(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load and prepare data using SLMGAE approach
    print("\n=== Data Preparation (SLMGAE Method) ===")
    print("1. Loading data from SL_Human_Approved.txt...")

    data = load_slmgae_data(val_ratio=args.val_ratio,
                            test_ratio=args.test_ratio,
                            neg_to_pos_ratio=args.neg_to_pos_ratio,
                            seed=args.seed)

    # Extract data components
    all_genes = data['gene_list']
    gene_to_idx = data['gene_to_idx']
    n_genes = data['num_genes']
    train_edges = data['train_edges']
    train_labels = data['train_labels']
    val_edges = data['val_edges']
    val_labels = data['val_labels']
    test_edges = data['test_edges']
    test_labels = data['test_labels']

    # Build adjacency matrix from training edges
    print("\n2. Building graph from training data...")
    adj_matrix = build_adjacency_from_edges(train_edges, n_genes)
    print(f"Graph density: {(adj_matrix > 0).sum() / (n_genes * n_genes):.4f}")

    # Generate K diverse linearizations: {σ_k}_{k=1}^K with d(σ_i,σ_j) ≥ 0.3
    print(
        f"\n4. Generating {args.num_linearizations} diverse graph linearizations..."
    )
    linearizer = GraphLinearizer(adj_matrix)
    linearizations = linearizer.generate_multiple_linearizations(
        num_linearizations=args.num_linearizations,
        min_diversity_threshold=0.3,  # min d(σ_i,σ_j) = 0.3
        max_attempts_per_lin=20)

    if len(linearizations) < args.num_linearizations:
        print(
            f"Warning: Could only generate {len(linearizations)}/{args.num_linearizations} diverse linearizations"
        )

    print(
        f"Average linearization length: {np.mean([len(l) for l in linearizations]):.1f}"
    )

    # Prepare features
    print("\n5. Preparing features...")

    # Node features: X ∈ ℝ^{n×d} from ESM or learnable params
    if args.learnable_node_features:
        # Initialize: X ~ N(0, 0.01I)
        node_features = torch.randn(
            n_genes, args.node_feature_dim, device=device) * 0.1
        # Will register as nn.Parameter after model creation
    else:
        # ESM embeddings: pre-trained protein representations
        node_features = load_esm_embeddings(
            gene_list=all_genes,
            embedding_path=args.esm_embedding_path,
            embedding_dim=args.node_feature_dim,
            model_name=args.esm_model_name).to(device)

    # Support views and edge mapping
    support_views, edge_map = prepare_support_views(
        adj_matrix=adj_matrix,
        n_support_views=args.n_support_views,
        support_view_dim=args.support_view_dim,
        support_view_paths=args.
        support_view_paths,  # Can be None for random init
        use_biological_views=args.use_biological_views,
        num_nodes=n_genes)
    support_views = support_views.to(device)
    print(f"Support views shape: {support_views.shape}")
    print(f"Number of edges in graph: {len(edge_map)}")

    # Edge tensors are already prepared from load_slmgae_data
    # No need to prepare them again

    # Move to device
    train_edges, train_labels = train_edges.to(device), train_labels.to(device)
    val_edges, val_labels = val_edges.to(device), val_labels.to(device)
    test_edges, test_labels = test_edges.to(device), test_labels.to(device)

    print(f"\nDataset sizes:")
    print(
        f"  Train: {train_edges.shape[1]} edges ({train_labels.sum():.0f} positive)"
    )
    print(
        f"  Val: {val_edges.shape[1]} edges ({val_labels.sum():.0f} positive)")
    print(
        f"  Test: {test_edges.shape[1]} edges ({test_labels.sum():.0f} positive)"
    )

    # Create model
    print("\n=== Model Creation ===")
    model = GraphTransformer(
        node_feature_dim=args.node_feature_dim,
        n_support_views=args.n_support_views,
        feature_dim_per_view=args.support_view_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_nodes=n_genes,
        dropout=args.dropout,
        undirected=True  # SL prediction graphs are undirected
    ).to(device)

    # Add learnable node features to model if specified
    if args.learnable_node_features:
        model.register_parameter('node_features',
                                 torch.nn.Parameter(node_features.to(device)))
        # Node features are now part of model.parameters() and will be optimized
        optimizer_params = model.parameters()
    else:
        # Ensure node_features are on correct device for non-learnable case
        node_features = node_features.to(device)
        optimizer_params = model.parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup training
    optimizer = optim.AdamW(optimizer_params,
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=args.epochs,
                                  eta_min=args.min_lr)

    # Setup loss function for imbalanced data
    print(f"\n6. Setting up loss function: {args.loss_function}")

    # Compute class weights from training data
    train_class_weights = compute_class_weights(train_labels)
    print(f"Training set class distribution:")
    print(
        f"  Positive: {train_class_weights['n_positive']} ({train_class_weights['positive_ratio']:.1%})"
    )
    print(
        f"  Negative: {train_class_weights['n_negative']} ({train_class_weights['negative_ratio']:.1%})"
    )
    print(f"  Imbalance ratio: {train_class_weights['imbalance_ratio']:.2f}")
    print(f"  Recommended pos_weight: {train_class_weights['pos_weight']:.2f}")

    # Create loss function
    if args.loss_function == 'combined':
        criterion = get_loss_function(args.loss_function,
                                      focal_weight=0.6,
                                      bce_weight=0.4,
                                      focal_params={
                                          'alpha': 0.25,
                                          'gamma': 2.0,
                                          'label_smoothing':
                                          args.label_smoothing
                                      },
                                      bce_params={
                                          'pos_weight':
                                          train_class_weights['pos_weight'],
                                          'label_smoothing':
                                          args.label_smoothing
                                      })
    else:
        criterion = get_loss_function(
            args.loss_function,
            pos_weight=train_class_weights['pos_weight'],
            alpha=0.25 if 'focal' in args.loss_function else None,
            gamma=2.0 if 'focal' in args.loss_function else None,
            label_smoothing=args.label_smoothing)

    # Training history
    history = {'train_loss': [], 'val_metrics': [], 'support_view_weights': []}

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Training loop
    print("\n=== Training ===")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model=model,
                                 node_features=node_features,
                                 linearizations=linearizations,
                                 support_views=support_views,
                                 edge_map=edge_map,
                                 train_edges=train_edges,
                                 train_labels=train_labels,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 batch_size=args.batch_size,
                                 device=device)
        history['train_loss'].append(train_loss)

        # Validate
        val_metrics = evaluate(model=model,
                               node_features=node_features,
                               linearizations=linearizations,
                               support_views=support_views,
                               edge_map=edge_map,
                               eval_edges=val_edges,
                               eval_labels=val_labels,
                               n_samples=min(10, len(linearizations)),
                               device=device)
        history['val_metrics'].append(val_metrics)

        # Track support view weights
        support_weights = model.get_support_view_weights().detach().cpu(
        ).numpy()
        history['support_view_weights'].append(support_weights)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f}")
        print(
            f"  Val AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}"
        )
        print(f"  Support View Weights (α²): {support_weights}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'config': vars(args)
                }, os.path.join(args.output_dir, 'best_model.pt'))

            print(f"  → New best model saved (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Load best model for final evaluation
    print(f"\n=== Final Evaluation ===")
    print(f"Loading best model from epoch {best_epoch+1}")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set with more samples
    test_metrics = evaluate(model=model,
                            node_features=node_features,
                            linearizations=linearizations,
                            support_views=support_views,
                            edge_map=edge_map,
                            eval_edges=test_edges,
                            eval_labels=test_labels,
                            n_samples=min(20, len(linearizations)),
                            device=device)

    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save final results
    history['test_metrics'] = test_metrics
    history['best_epoch'] = best_epoch

    with open(os.path.join(args.output_dir, 'training_history.json'),
              'w') as f:
        json.dump(history, f, indent=2)

    # Plot training history
    plot_training_history(history, args.output_dir)

    # Generate predictions for all gene pairs
    print("\n=== Generating Predictions ===")
    predictions_df = generate_all_predictions(model=model,
                                              node_features=node_features,
                                              linearizations=linearizations,
                                              support_views=support_views,
                                              edge_map=edge_map,
                                              all_genes=all_genes,
                                              gene_to_idx=gene_to_idx,
                                              device=device,
                                              n_samples=20,
                                              batch_size=1000)

    # Save predictions
    predictions_file = os.path.join(args.output_dir, 'all_predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Saved predictions to: {predictions_file}")

    # Save gene order for reference
    gene_order_file = os.path.join(args.output_dir, 'gene_order.txt')
    with open(gene_order_file, 'w') as f:
        for gene in all_genes:
            f.write(f"{gene}\n")
    print(f"Saved gene order to: {gene_order_file}")

    # Save final support view weights
    final_weights = model.get_attention_weights()
    torch.save(final_weights,
               os.path.join(args.output_dir, 'support_view_weights.pt'))

    print(f"\nTraining complete! Results saved to: {args.output_dir}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Train Graph Transformer with weighted support views for SL prediction"
    )

    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data/SL_Human_Approved.txt',
        help='Path to SL data file (default: SL_Human_Approved.txt)')
    parser.add_argument('--support_view_paths',
                        type=str,
                        nargs='*',
                        default=None,
                        help='Paths to support view files (optional)')
    parser.add_argument(
        '--use_biological_views',
        action='store_true',
        help='Use biological support views from main study (GO-BP, GO-CC, PPI)'
    )
    parser.add_argument('--esm_embedding_path',
                        type=str,
                        default=None,
                        help='Path to precomputed ESM embeddings (NPZ/PT/H5)')
    parser.add_argument('--esm_model_name',
                        type=str,
                        default='esm2_t33_650M_UR50D',
                        help='ESM model name for on-the-fly generation')
    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.15,
                        help='Test set ratio')
    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--neg_to_pos_ratio',
                        type=int,
                        default=1,
                        help='Ratio of negative to positive samples')

    # Model architecture
    parser.add_argument('--node_feature_dim',
                        type=int,
                        default=1280,
                        help='Dimension of node features (ESM embeddings)')
    parser.add_argument('--learnable_node_features',
                        action='store_true',
                        help='Make node features learnable parameters')
    parser.add_argument('--n_support_views',
                        type=int,
                        default=3,
                        help='Number of support views')
    parser.add_argument('--support_view_dim',
                        type=int,
                        default=64,
                        help='Dimension of each support view')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=256,
                        help='Hidden dimension of transformer')
    parser.add_argument('--n_heads',
                        type=int,
                        default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers',
                        type=int,
                        default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate')

    # Graph linearization
    parser.add_argument('--num_linearizations',
                        type=int,
                        default=100,
                        help='Number of graph linearizations to generate')

    # Training arguments
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size for edge predictions')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--min_lr',
                        type=float,
                        default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-5,
                        help='Weight decay for AdamW')
    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='Early stopping patience')

    # Loss function arguments
    parser.add_argument(
        '--loss_function',
        type=str,
        default='focal',
        choices=['focal', 'weighted_bce', 'asymmetric', 'dice', 'combined'],
        help='Loss function for imbalanced data')
    parser.add_argument('--label_smoothing',
                        type=float,
                        default=0.0,
                        help='Label smoothing factor (0.0 to 0.2)')

    # Other arguments
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")

    # Run training
    main(args)

# === Prediction Generation ===


def generate_all_predictions(model,
                             node_features,
                             linearizations,
                             support_views,
                             edge_map,
                             all_genes,
                             gene_to_idx,
                             device,
                             n_samples=20,
                             batch_size=1000):
    """
    Generate predictions for all possible gene pairs.
    
    Returns:
        predictions_df: DataFrame with gene1, gene2, and sl_probability columns
    """
    import pandas as pd
    import torch
    from tqdm import tqdm

    print(
        f"\nGenerating predictions for all {len(all_genes)}×{len(all_genes)} gene pairs..."
    )

    model.eval()
    n_genes = len(all_genes)

    # Generate all possible gene pairs (upper triangle for undirected graph)
    all_pairs = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            all_pairs.append((i, j))

    print(f"Total gene pairs to predict: {len(all_pairs)}")

    # Process in batches
    all_predictions = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(all_pairs), batch_size)):
            batch_end = min(batch_start + batch_size, len(all_pairs))
            batch_pairs = all_pairs[batch_start:batch_end]

            # Convert to edge tensor format
            batch_edges = torch.tensor(batch_pairs,
                                       dtype=torch.long).t().to(device)

            # Collect predictions from multiple linearizations
            batch_preds = []
            n_samples_actual = min(n_samples, len(linearizations))

            for i in range(n_samples_actual):
                linearization = linearizations[i]

                # Get node features (use learnable features if available)
                if hasattr(model, 'node_features'):
                    current_node_features = model.node_features
                else:
                    current_node_features = node_features

                # Forward pass
                preds = model(node_features=current_node_features,
                              linearization=linearization,
                              support_views=support_views,
                              edge_map=edge_map,
                              eval_edges=batch_edges)

                batch_preds.append(torch.sigmoid(preds))

            # Average predictions across linearizations
            avg_preds = torch.stack(batch_preds).mean(dim=0)
            all_predictions.append(avg_preds.cpu().numpy())

    # Combine all predictions
    import numpy as np
    all_predictions = np.concatenate(all_predictions)

    # Create DataFrame with gene names
    results = []
    for i, (gene_i, gene_j) in enumerate(all_pairs):
        gene1_name = all_genes[gene_i]
        gene2_name = all_genes[gene_j]
        sl_probability = all_predictions[i]

        results.append({
            'gene1': gene1_name,
            'gene2': gene2_name,
            'sl_probability': sl_probability
        })

    predictions_df = pd.DataFrame(results)

    print(f"\n✅ Generated predictions for {len(predictions_df)} gene pairs")
    print(
        f"   Mean SL probability: {predictions_df['sl_probability'].mean():.4f}"
    )
    print(
        f"   Top 1% threshold: {predictions_df['sl_probability'].quantile(0.99):.4f}"
    )

    return predictions_df
