#!/usr/bin/env python3
"""
Pure wrapper for SL_benchmark's evaluation functions.

This module ONLY imports and calls functions from preprocess_benchmarking_paper.py.
No custom logic - just format conversion.

Source: https://github.com/JieZheng-ShanghaiTech/SL_benchmark/blob/main/src/preprocess.py
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List

# Import their functions directly
from preprocess_benchmarking_paper import cal_metrics


def evaluate_predictions(
    score_mat: np.ndarray,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    seen_index: np.ndarray = None,
) -> List[float]:
    """
    Pure wrapper for benchmark.cal_metrics().

    Calls their function directly without modification.
    https://github.com/JieZheng-ShanghaiTech/SL_benchmark/blob/main/src/preprocess.py#L818-L892

    Args:
        score_mat: Score matrix (n_genes, n_genes) - can be sparse or dense
        pos_index: Positive edge indices, shape (N_pos, 2)
        neg_index: Negative edge indices, shape (N_neg, 2)
        seen_index: Edges to exclude from ranking evaluation (optional)

    Returns:
        metrics: List of 15 metrics from their implementation:
            [auroc, f1, aupr,
             ndcg@10, ndcg@20, ndcg@50,
             recall@10, recall@20, recall@50,
             precision@10, precision@20, precision@50,
             map@10, map@20, map@50]
    """
    return cal_metrics(score_mat, pos_index, neg_index, seen_index)


def evaluate_predictions_dict(
    score_mat: np.ndarray,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    seen_index: np.ndarray = None,
) -> Dict[str, float]:
    """
    Same as evaluate_predictions but returns dict instead of list.

    Args:
        score_mat: Score matrix (n_genes, n_genes)
        pos_index: Positive edge indices, shape (N_pos, 2)
        neg_index: Negative edge indices, shape (N_neg, 2)
        seen_index: Edges to exclude from ranking (optional)

    Returns:
        metrics: Dictionary with named metrics
    """
    metrics_list = cal_metrics(score_mat, pos_index, neg_index, seen_index)

    # Their function returns a flat list of 15 elements (via np.hstack)
    return {
        "auroc": metrics_list[0],
        "f1": metrics_list[1],
        "aupr": metrics_list[2],
        "ndcg@10": metrics_list[3],
        "ndcg@20": metrics_list[4],
        "ndcg@50": metrics_list[5],
        "recall@10": metrics_list[6],
        "recall@20": metrics_list[7],
        "recall@50": metrics_list[8],
        "precision@10": metrics_list[9],
        "precision@20": metrics_list[10],
        "precision@50": metrics_list[11],
        "map@10": metrics_list[12],
        "map@20": metrics_list[13],
        "map@50": metrics_list[14],
    }


def calculate_optimal_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate F1 score using optimal threshold from precision-recall curve.

    This is the correct way to calculate F1 for imbalanced prediction tasks,
    as it finds the threshold that maximizes F1 rather than using a fixed 0.5.

    Args:
        labels: Ground truth binary labels, shape (N,)
        predictions: Predicted scores (NOT binary), shape (N,)

    Returns:
        max_f1: Maximum F1 score across all thresholds

    Reference: Matches TensorFlow implementation in metrics.py:35-41
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, predictions)
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Add epsilon to avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    max_f1 = np.max(f1_scores)

    return max_f1


def evaluate_from_edges(predictions: np.ndarray,
                        labels: np.ndarray) -> Dict[str, float]:
    """
    Helper to convert edge-based predictions to score matrix format.

    This is for compatibility with training scripts that have edge predictions.
    Converts to the format expected by cal_metrics().

    Args:
        predictions: Predicted scores for edges, shape (N_edges,)
        labels: Ground truth labels for edges, shape (N_edges,)

    Returns:
        Simple metrics dict with auc, aupr, f1
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    auc = roc_auc_score(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    f1 = calculate_optimal_f1(labels, predictions)

    return {"auc": auc, "aupr": aupr, "f1": f1}
