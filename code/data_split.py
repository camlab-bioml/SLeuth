#!/usr/bin/env python3
"""
Pure wrapper for SL_benchmark's data splitting functions.

This module ONLY imports and calls functions from preprocess_benchmarking_paper.py.
No custom logic - just format conversion.

Source: https://github.com/JieZheng-ShanghaiTech/SL_benchmark/blob/main/src/preprocess.py
Source saved as preprocess_benchmarking_paper.py
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import List, Dict

# Import their functions directly
from preprocess_benchmarking_paper import cv1, cv2, cv3
import preprocess_benchmarking_paper


class SLDataSplitter:
    """
    Pure wrapper for SL_benchmark's cv1/cv2/cv3 functions.

    Only does two things:
    1. Set global variables to inject our data into their functions
    2. Convert their output format to our expected format
    """

    def __init__(
        self,
        pos_edges: np.ndarray,
        neg_edges: np.ndarray,
        num_nodes: int,
        train_ratio: float = 0.8,
        random_state: int = 123,
    ):
        """Initialize and inject our data into benchmark module.

        Args:
            pos_edges: Positive edge pairs (N, 2)
            neg_edges: Negative edge pairs (M, 2)
            num_nodes: Total number of nodes/genes
            train_ratio: Fraction of data for training (default: 0.8)
            random_state: Random seed (default: 123)
        """
        self.num_nodes = num_nodes
        self.train_ratio = train_ratio
        self.valid_ratio = 0.0  # We don't use validation set
        self.test_ratio = 1.0 - train_ratio  # Dynamic: test_ratio = 1 - train_ratio

        # Validate input
        if len(pos_edges) == 0:
            raise ValueError(
                "pos_edges is empty - need at least some positive examples")

        if pos_edges.shape[1] != 2:
            raise ValueError(
                f"pos_edges should have shape (N, 2), got {pos_edges.shape}")

        if not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"train_ratio must be between 0 and 1, got {train_ratio}")

        # Inject our positive edges into benchmark's global variable
        # SL_benchmark's cv1/cv2/cv3 will split these and build adjacency per fold
        preprocess_benchmarking_paper.human_sl_pairs_df = pd.DataFrame({
            "unified_id_A":
            pos_edges[:, 0],
            "unified_id_B":
            pos_edges[:, 1]
        })

    def cv1_split(self, k: int = 5, pos_neg_ratio: float = 1.0) -> List[Dict]:
        """Call their cv1() function and convert output."""
        result = cv1(
            kfold=k,
            num_node=self.num_nodes,
            train_rat=self.train_ratio,
            valid_rat=self.valid_ratio,
            test_rat=self.test_ratio,
            training_rat=1.0,
            xtimes=pos_neg_ratio,
            negative_strategy="Random",
            exp_data_path=None,
            score_data_path=None,
            ex_compt=None,
        )

        # Check if benchmark function failed (returns None instead of tuple)
        if result is None:
            raise RuntimeError(
                "SL_benchmark cv1() failed - likely due to invalid train/valid/test ratios. "
                "Check that train_rat + valid_rat + test_rat == 1.0")

        pos_samples, neg_samples = result
        return self._convert_output(pos_samples, neg_samples, k)

    def cv2_split(self, k: int = 5, pos_neg_ratio: float = 1.0) -> List[Dict]:
        """Call their cv2() function and convert output."""
        result = cv2(
            kfold=k,
            num_node=self.num_nodes,
            train_rat=self.train_ratio,
            valid_rat=self.valid_ratio,
            test_rat=self.test_ratio,
            training_rat=1.0,
            xtimes=pos_neg_ratio,
            negative_strategy="Random",
            exp_data_path=None,
            score_data_path=None,
            ex_compt=None,
        )

        # Check if benchmark function failed (returns None instead of tuple)
        if result is None:
            raise RuntimeError(
                "SL_benchmark cv2() failed - likely due to invalid train/valid/test ratios. "
                "Check that train_rat + valid_rat + test_rat == 1.0")

        pos_samples, neg_samples = result
        return self._convert_output(pos_samples, neg_samples, k)

    def cv3_split(self, k: int = 5, pos_neg_ratio: float = 1.0) -> List[Dict]:
        """Call their cv3() function and convert output."""
        result = cv3(
            kfold=k,
            num_node=self.num_nodes,
            train_rat=self.train_ratio,
            valid_rat=self.valid_ratio,
            test_rat=self.test_ratio,
            training_rat=1.0,
            xtimes=pos_neg_ratio,
            negative_strategy="Random",
            exp_data_path=None,
            score_data_path=None,
            ex_compt=None,
        )

        # Check if benchmark function failed (returns None instead of tuple)
        if result is None:
            raise RuntimeError(
                "SL_benchmark cv3() failed - likely due to invalid train/valid/test ratios. "
                "Check that train_rat + valid_rat + test_rat == 1.0")

        pos_samples, neg_samples = result
        return self._convert_output(pos_samples, neg_samples, k)

    def _convert_output(self, pos_samples: List, neg_samples: List,
                        k: int) -> List[Dict]:
        """
        Convert SL_benchmark output to training format.

        SL_benchmark already did the splitting and built adjacency matrices.
        We just reformat their output for our training loop.

        Input (SL_benchmark format):
            pos_samples = [graph_train_pos, graph_test_pos, train_pos_edges, test_pos_edges]
            neg_samples = [graph_train_neg, graph_test_neg, train_neg_edges, test_neg_edges]

        Output (training format) - one dict per fold:
            {
                'train_adj': Pre-built adjacency from training positive edges only
                'train_edges': Combined positive + negative training edge pairs
                'train_labels': 1 for positive, 0 for negative
                'test_edges': Combined positive + negative test edge pairs
                'test_labels': 1 for positive, 0 for negative
            }
        """
        # Validate output format from SL_benchmark
        if not isinstance(pos_samples, list) or len(pos_samples) != 4:
            raise ValueError(
                f"SL_benchmark returned unexpected pos_samples format. "
                f"Expected list of 4 elements, got {type(pos_samples)} with {len(pos_samples) if isinstance(pos_samples, list) else 'N/A'} elements"
            )

        if not isinstance(neg_samples, list) or len(neg_samples) != 4:
            raise ValueError(
                f"SL_benchmark returned unexpected neg_samples format. "
                f"Expected list of 4 elements, got {type(neg_samples)} with {len(neg_samples) if isinstance(neg_samples, list) else 'N/A'} elements"
            )

        # Validate that each element has k folds
        for i, sample_list in enumerate(pos_samples):
            if not isinstance(sample_list, list) or len(sample_list) != k:
                raise ValueError(
                    f"pos_samples[{i}] should contain {k} folds, got {len(sample_list) if isinstance(sample_list, list) else 'not a list'}"
                )

        for i, sample_list in enumerate(neg_samples):
            if not isinstance(sample_list, list) or len(sample_list) != k:
                raise ValueError(
                    f"neg_samples[{i}] should contain {k} folds, got {len(sample_list) if isinstance(sample_list, list) else 'not a list'}"
                )

        splits = []

        for fold_idx in range(k):
            # Extract edges for this fold
            train_pos = pos_samples[2][
                fold_idx]  # Training positive edge pairs
            test_pos = pos_samples[3][fold_idx]  # Test positive edge pairs
            train_neg = neg_samples[2][
                fold_idx]  # Training negative edge pairs
            test_neg = neg_samples[3][fold_idx]  # Test negative edge pairs

            # Validate that fold has non-empty data
            if len(train_pos) == 0:
                raise ValueError(
                    f"Fold {fold_idx}: train_pos is empty - need training examples"
                )
            if len(test_pos) == 0:
                raise ValueError(
                    f"Fold {fold_idx}: test_pos is empty - need test examples")
            if len(train_neg) == 0:
                raise ValueError(
                    f"Fold {fold_idx}: train_neg is empty - need negative training examples"
                )
            if len(test_neg) == 0:
                raise ValueError(
                    f"Fold {fold_idx}: test_neg is empty - need negative test examples"
                )

            # Validate adjacency matrix
            train_adj = pos_samples[0][fold_idx]
            if not sp.issparse(train_adj):
                raise ValueError(
                    f"Fold {fold_idx}: train_adj should be a scipy sparse matrix, got {type(train_adj)}"
                )

            splits.append({
                "fold":
                fold_idx,
                # Pre-built adjacency from training positive edges (excludes test edges)
                "train_adj":
                train_adj,
                # Training supervision: edges + labels
                "train_edges":
                np.vstack([train_pos, train_neg]),
                "train_labels":
                np.hstack([np.ones(len(train_pos)),
                           np.zeros(len(train_neg))]),
                # Test supervision: edges + labels
                "test_edges":
                np.vstack([test_pos, test_neg]),
                "test_labels":
                np.hstack([np.ones(len(test_pos)),
                           np.zeros(len(test_neg))]),
                # Metadata
                "num_train_pos":
                len(train_pos),
                "num_train_neg":
                len(train_neg),
                "num_test_pos":
                len(test_pos),
                "num_test_neg":
                len(test_neg),
            })

        return splits
