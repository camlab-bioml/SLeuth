#!/usr/bin/env python3
"""
Pure wrapper for SL_benchmark's data splitting functions.

This module ONLY imports and calls functions from preprocess_benchmarking_paper.py.
No custom logic - just format conversion.

Source: https://github.com/JieZheng-ShanghaiTech/SL_benchmark/blob/main/src/preprocess.py
Source saved as preprocess_benchmarking_paper.py
"""

import os
import random
import tempfile

import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import List, Dict

# Import their functions directly
from preprocess_benchmarking_paper import cv1, cv2, cv3
import preprocess_benchmarking_paper

# With experimental negatives a CV fold can come out empty (most often a CV3
# test fold whose held-out-gene region contains no screened negative). We retry
# the split with incremented seeds — different seed -> different gene partition
# — up to this many attempts before failing.
DEFAULT_MAX_SPLIT_RETRIES = 10


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
        negative_strategy: str = "random",
        max_split_retries: int = DEFAULT_MAX_SPLIT_RETRIES,
    ):
        """Initialize and inject our data into benchmark module.

        Args:
            pos_edges: Positive edge pairs (N, 2)
            neg_edges: Negative edge pairs (M, 2). Only used when
                ``negative_strategy == "experimental"``; ignored (and may be
                empty) for the random strategy.
            num_nodes: Total number of nodes/genes
            train_ratio: Fraction of data for training (default: 0.8)
            random_state: Random seed (default: 123)
            negative_strategy:
                "random"       — the benchmark samples random non-edges among
                                 the nodes during splitting (original behavior).
                "experimental" — use the supplied ``neg_edges`` as the negative
                                 pool. The benchmark's own "Exp" code path then
                                 partitions them with the SAME gene-disjoint
                                 logic used for positives (so CV2/CV3 stay
                                 leak-free), drawing per fold from this pool.
            max_split_retries: For the experimental strategy only, how many
                seeds to try (base, base+1, ...) if a fold comes out empty.
                The random strategy ignores this (it splits on the base seed).
        """
        if negative_strategy not in ("random", "experimental"):
            raise ValueError(
                f"negative_strategy must be 'random' or 'experimental'; "
                f"got {negative_strategy!r}")

        self.num_nodes = num_nodes
        self.train_ratio = train_ratio
        self.valid_ratio = 0.0  # We don't use validation set
        self.test_ratio = 1.0 - train_ratio  # Dynamic: test_ratio = 1 - train_ratio
        self.random_state = random_state
        self.negative_strategy = negative_strategy
        self.max_split_retries = max(1, int(max_split_retries))
        self.num_pos = len(pos_edges)

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

        if negative_strategy == "experimental":
            if neg_edges is None or len(neg_edges) == 0:
                raise ValueError(
                    "negative_strategy='experimental' requires a non-empty "
                    "neg_edges pool")
            self.neg_edges = np.asarray(neg_edges, dtype=int)
            if self.neg_edges.ndim != 2 or self.neg_edges.shape[1] != 2:
                raise ValueError(f"neg_edges should have shape (M, 2), got "
                                 f"{self.neg_edges.shape}")
        else:
            self.neg_edges = None

        # Inject our positive edges into benchmark's global variable
        # SL_benchmark's cv1/cv2/cv3 will split these and build adjacency per fold
        preprocess_benchmarking_paper.human_sl_pairs_df = pd.DataFrame({
            "unified_id_A":
            pos_edges[:, 0],
            "unified_id_B":
            pos_edges[:, 1]
        })
        preprocess_benchmarking_paper.setup_all_seeds = random_state

    def _materialize_exp_npy(self, neg_array: np.ndarray, tmpdir: str):
        """Write the two .npy files the benchmark's "Exp" path expects.

        The benchmark loads negatives via::

            neg_id_scores = np.load(score_data_path)   # (M, >=2)
            neg_index     = np.load(exp_data_path)     # row indices
            neg_position  = neg_id_scores[neg_index, :2]

        so a score file of shape (M, 2) plus an identity index of arange(M)
        reproduces exactly our negative pool, with no change to the vendored
        benchmark code (ex_compt is left None so no id remapping is applied).
        """
        score_path = os.path.join(tmpdir, "neg_scores.npy")
        index_path = os.path.join(tmpdir, "neg_index.npy")
        np.save(score_path, neg_array.astype(int))
        np.save(index_path, np.arange(len(neg_array), dtype=int))
        return score_path, index_path

    def _run_cv(self, cv_func, k, pos_neg_ratio, cv1_balance, seed):
        """Dispatch one CV function under the configured negative strategy.

        The caller sets ``preprocess_benchmarking_paper.setup_all_seeds = seed``
        before invoking this, so the benchmark's gene KFold / negative sampling
        use that seed; the same seed drives the cv1 pre-trim below.
        """
        if self.negative_strategy == "random":
            return cv_func(
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

        # experimental: feed our curated pool through the benchmark's "Exp"
        # path. cv1 splits the negative pool with an independent KFold and does
        # NOT subsample it, so to honor pos_neg_ratio we pre-trim the pool here.
        # cv2/cv3 subsample per fold inside their *_division step, so they get
        # the full pool (a larger pool = better gene-disjoint coverage).
        neg = self.neg_edges
        if cv1_balance:
            target = int(self.num_pos * pos_neg_ratio)
            if 0 < target < len(neg):
                rng = random.Random(seed)
                sel = sorted(rng.sample(range(len(neg)), target))
                neg = neg[np.asarray(sel)]
        with tempfile.TemporaryDirectory() as tmpdir:
            score_path, index_path = self._materialize_exp_npy(neg, tmpdir)
            return cv_func(
                kfold=k,
                num_node=self.num_nodes,
                train_rat=self.train_ratio,
                valid_rat=self.valid_ratio,
                test_rat=self.test_ratio,
                training_rat=1.0,
                xtimes=pos_neg_ratio,
                negative_strategy="Exp",
                exp_data_path=index_path,
                score_data_path=score_path,
                ex_compt=None,
            )

    def _split_with_retry(self, cv_func, name, k, pos_neg_ratio, cv1_balance):
        """Run a CV split, retrying with incremented seeds on an empty fold.

        With experimental negatives a fold can come out empty — most often a
        CV3 test fold whose held-out-gene region contains no screened negative
        (the pool is sparse there). A different seed yields a different gene
        partition, which usually resolves it. Seeds are deterministic
        (base, base+1, ...) and the effective seed is logged so runs stay
        reproducible. The random strategy is never retried (it splits once on
        the base seed), so its behavior is unchanged.
        """
        max_attempts = (self.max_split_retries
                        if self.negative_strategy == "experimental" else 1)
        last_err = None
        for attempt in range(max_attempts):
            seed = self.random_state + attempt
            preprocess_benchmarking_paper.setup_all_seeds = seed
            result = self._run_cv(cv_func, k, pos_neg_ratio, cv1_balance, seed)
            if result is None:
                raise RuntimeError(
                    f"SL_benchmark {name}() failed - likely due to invalid "
                    f"train/valid/test ratios. Check that "
                    f"train_rat + valid_rat + test_rat == 1.0")
            pos_samples, neg_samples = result
            try:
                splits = self._convert_output(pos_samples, neg_samples, k)
            except ValueError as e:
                # Only empty-fold errors are retryable. Structural errors, and
                # any empty fold under the random strategy (max_attempts == 1),
                # re-raise the original error so that path is unchanged.
                if self.negative_strategy != "experimental" \
                        or "is empty" not in str(e):
                    raise
                last_err = e
                if attempt + 1 < max_attempts:
                    print(f"  [{name}] empty fold at seed={seed} ({e}); "
                          f"retrying at seed={seed + 1} "
                          f"(attempt {attempt + 2}/{max_attempts})")
                    continue
                break  # retries exhausted -> clean error after the loop
            if attempt > 0:
                print(
                    f"  [{name}] produced non-empty folds at seed={seed} "
                    f"(after {attempt} retr{'y' if attempt == 1 else 'ies'}; "
                    f"base seed was {self.random_state})")
            return splits

        raise ValueError(
            f"{name}: could not produce non-empty folds after "
            f"{max_attempts} seeds (base={self.random_state}). The "
            f"experimental-negative pool is likely too sparse for k={k} folds "
            f"in this CV mode (try fewer folds, a larger pool, or a higher "
            f"pos_neg_ratio cap). Last error: {last_err}")

    def cv1_split(self, k: int = 5, pos_neg_ratio: float = 1.0) -> List[Dict]:
        """Call their cv1() function and convert output."""
        return self._split_with_retry(cv1, "cv1", k, pos_neg_ratio, True)

    def cv2_split(self, k: int = 5, pos_neg_ratio: float = 1.0) -> List[Dict]:
        """Call their cv2() function and convert output."""
        return self._split_with_retry(cv2, "cv2", k, pos_neg_ratio, False)

    def cv3_split(self, k: int = 5, pos_neg_ratio: float = 1.0) -> List[Dict]:
        """Call their cv3() function and convert output."""
        return self._split_with_retry(cv3, "cv3", k, pos_neg_ratio, False)

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
