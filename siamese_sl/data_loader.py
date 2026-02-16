#!/usr/bin/env python3
"""
Data loading utilities for Siamese SL prediction.

Uses SLMGAE benchmark's CV splitting code for consistency:
  - CV1: Edge-based split (random split of SL pairs)
  - CV2: Gene-based split (hold out genes, test pairs have ≥1 unseen gene)
  - CV3: Pair-based split (both genes in test pairs are unseen)

Handles:
  - Loading gene embeddings for all genes (~20k)
  - Loading SL interaction pairs from SLMGAE (~6k genes, ~10k pairs)
  - Creating train/test splits via SLMGAE benchmark code
  - Batching for training
"""

import sys
from pathlib import Path

# Add SLMGAE code directory to path for importing benchmark code
SLMGAE_CODE_DIR = Path(__file__).parent.parent / "code"
if str(SLMGAE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SLMGAE_CODE_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import random

# Import SLMGAE benchmark's data splitting code
from data_split import SLDataSplitter


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SLDataset(Dataset):
    """
    Dataset for Synthetic Lethality prediction.

    Each sample is a gene pair with:
      - Embedding for gene1 (ESM-only or combined ESM+GO)
      - Embedding for gene2
      - Label (1 = SL, 0 = non-SL)
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        pairs: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            embeddings: Gene embeddings tensor (num_genes, embed_dim)
            pairs: Gene index pairs (N, 2)
            labels: Labels for each pair (N,)
        """
        self.embeddings = embeddings
        self.pairs = pairs
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j = self.pairs[idx]
        return (
            self.embeddings[i],
            self.embeddings[j],
            torch.tensor(self.labels[idx]),
        )


class SLDataManager:
    """
    Manages data loading and CV splits for SL prediction.

    Uses SLMGAE benchmark's SLDataSplitter for CV1/CV2/CV3 splits
    to ensure consistency with published results.

    NOTE: The --seed argument controls both model weight initialization
    AND CV splitting (passed as random_state to SLDataSplitter).

    Gene universe:
      - All genes with ESM embeddings (~20k)
      - SL labels only for ~6k genes from SLMGAE dataset
      - Negatives sampled from non-SL pairs by benchmark code
    """

    def __init__(
        self,
        embeddings_path: str,
        sl_pairs_path: str,
        gene_list_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            embeddings_path: Path to gene embeddings (.pt file)
            sl_pairs_path: Path to SL pairs file (gene1 TAB gene2 [TAB weight])
            gene_list_path: Optional path to gene list (for ordering)
            seed: Random seed for reproducibility
        """
        self.seed = seed
        set_seed(seed)

        # Load embeddings
        self.embeddings, self.gene_to_idx, self.idx_to_gene = self._load_embeddings(
            embeddings_path, gene_list_path
        )
        self.num_genes = len(self.gene_to_idx)

        # Load SL pairs (positive pairs) - indices into full embedding matrix
        self.pos_pairs_full_idx, self.sl_genes = self._load_sl_pairs(sl_pairs_path)
        self.num_sl_genes = len(self.sl_genes)

        # Create mapping between full embedding indices and SL-only indices
        # This is needed because SLDataSplitter expects indices in [0, num_sl_genes)
        self._create_sl_index_mapping()

        # Convert pos_pairs to SL-only indices for the splitter
        self.pos_pairs_sl_idx = self._full_to_sl_indices(self.pos_pairs_full_idx)

        # Create negative pairs for benchmark splitter
        # Benchmark code samples negatives during splitting
        self.neg_pairs = np.zeros((0, 2), dtype=np.int64)

        print(f"Loaded {len(self.embeddings)} gene embeddings")
        print(f"Loaded {len(self.pos_pairs_full_idx)} positive SL pairs")
        print(f"SL pairs involve {self.num_sl_genes} unique genes")

    def _load_embeddings(
        self,
        embeddings_path: str,
        gene_list_path: Optional[str],
    ) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
        """Load gene embeddings and create gene index mappings."""
        data = torch.load(embeddings_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            embeddings = data["embeddings"]
            if "gene_order" in data:
                gene_order = data["gene_order"]
            elif "gene_to_idx" in data:
                # Reconstruct gene_order from gene_to_idx (must be contiguous 0..N-1)
                gene_to_idx = data["gene_to_idx"]
                n = len(gene_to_idx)
                gene_order = [""] * n
                for gene, idx in gene_to_idx.items():
                    if not (0 <= idx < n):
                        raise ValueError(f"gene_to_idx has out-of-range index {idx} for {gene} (expected 0..{n-1})")
                    gene_order[idx] = gene
                if "" in gene_order:
                    raise ValueError("gene_to_idx has non-contiguous indices (gaps detected)")
            elif gene_list_path:
                with open(gene_list_path, "r") as f:
                    gene_order = [line.strip() for line in f]
            else:
                raise ValueError("No gene order in embeddings file and no gene_list_path")
        else:
            embeddings = data
            if gene_list_path:
                with open(gene_list_path, "r") as f:
                    gene_order = [line.strip() for line in f]
            else:
                raise ValueError("Raw tensor embeddings require gene_list_path")

        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_order)}
        idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}

        return embeddings, gene_to_idx, idx_to_gene

    def _load_sl_pairs(self, sl_pairs_path: str) -> Tuple[np.ndarray, Set[str]]:
        """
        Load positive SL pairs from file.

        Returns pairs as indices into the embedding matrix.
        Only includes pairs where BOTH genes have embeddings.
        """
        pairs = []
        sl_genes = set()
        seen = set()

        with open(sl_pairs_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    g1, g2 = parts[0], parts[1]

                    # Skip if gene not in embeddings
                    if g1 not in self.gene_to_idx or g2 not in self.gene_to_idx:
                        continue

                    # Consistent ordering (smaller index first)
                    idx1, idx2 = self.gene_to_idx[g1], self.gene_to_idx[g2]
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                        g1, g2 = g2, g1

                    pair_key = (idx1, idx2)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append([idx1, idx2])
                        sl_genes.add(g1)
                        sl_genes.add(g2)

        return np.array(pairs, dtype=np.int64), sl_genes

    def _create_sl_index_mapping(self) -> None:
        """
        Create bidirectional mapping between full embedding indices and SL-only indices.

        Full indices: 0 to num_embeddings-1 (e.g., 0-20000)
        SL indices: 0 to num_sl_genes-1 (e.g., 0-6000)

        This is needed because SLDataSplitter expects indices in [0, num_sl_genes).
        """
        # Get unique gene indices involved in SL pairs
        unique_full_indices = set()
        for g1, g2 in self.pos_pairs_full_idx:
            unique_full_indices.add(g1)
            unique_full_indices.add(g2)

        # Sort for deterministic mapping
        unique_full_indices = sorted(list(unique_full_indices))

        # Create mappings
        self.full_to_sl = {full_idx: sl_idx for sl_idx, full_idx in enumerate(unique_full_indices)}
        self.sl_to_full = {sl_idx: full_idx for full_idx, sl_idx in self.full_to_sl.items()}

    def _full_to_sl_indices(self, pairs: np.ndarray) -> np.ndarray:
        """Convert pairs from full embedding indices to SL-only indices."""
        converted = []
        for g1, g2 in pairs:
            converted.append([self.full_to_sl[g1], self.full_to_sl[g2]])
        return np.array(converted, dtype=np.int64)

    def _sl_to_full_indices(self, pairs: np.ndarray) -> np.ndarray:
        """Convert pairs from SL-only indices back to full embedding indices."""
        converted = []
        for g1, g2 in pairs:
            converted.append([self.sl_to_full[g1], self.sl_to_full[g2]])
        return np.array(converted, dtype=np.int64)

    def _create_splitter(self) -> SLDataSplitter:
        """Create SLMGAE benchmark's data splitter using SL-only indices."""
        return SLDataSplitter(
            pos_edges=self.pos_pairs_sl_idx,  # Use SL-only indices [0, num_sl_genes)
            neg_edges=self.neg_pairs,
            num_nodes=self.num_sl_genes,
            train_ratio=0.8,
            random_state=self.seed,
        )

    def _convert_split_to_dataset_format(
        self,
        split: Dict,
    ) -> Dict:
        """
        Convert SLMGAE benchmark split format to our dataset format.

        Benchmark format:
            train_edges, train_labels, test_edges, test_labels (using SL-only indices)

        Our format:
            train/test dicts with pairs using FULL embedding indices
        """
        # Convert SL-only indices back to full embedding indices
        train_pairs_full = self._sl_to_full_indices(split["train_edges"])
        test_pairs_full = self._sl_to_full_indices(split["test_edges"])

        return {
            "fold": split["fold"],
            "train": {
                "pairs": train_pairs_full,
                "labels": split["train_labels"],
            },
            "test": {
                "pairs": test_pairs_full,
                "labels": split["test_labels"],
            },
            "num_train_pos": split["num_train_pos"],
            "num_train_neg": split["num_train_neg"],
            "num_test_pos": split["num_test_pos"],
            "num_test_neg": split["num_test_neg"],
        }

    def get_cv1_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV1: Edge-based cross-validation (from SLMGAE benchmark).

        Randomly splits SL pairs into folds.
        Tests ability to predict held-out interactions between known genes.
        """
        print(f"\nGenerating CV1 splits (edge-based) using SLMGAE benchmark code...")
        splitter = self._create_splitter()
        splits = splitter.cv1_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold']}: "
                  f"train={split['num_train_pos']}+/{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/{split['num_test_neg']}-")

        return converted

    def get_cv2_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV2: Gene-based cross-validation (from SLMGAE benchmark).

        Holds out entire genes. Test pairs have at least one unseen gene.
        Tests ability to generalize to partially new genes.
        """
        print(f"\nGenerating CV2 splits (gene-based) using SLMGAE benchmark code...")
        splitter = self._create_splitter()
        splits = splitter.cv2_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold']}: "
                  f"train={split['num_train_pos']}+/{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/{split['num_test_neg']}-")

        return converted

    def get_cv3_splits(
        self,
        num_folds: int = 5,
        pos_neg_ratio: float = 1.0,
    ) -> List[Dict]:
        """
        CV3: Pair-based cross-validation (from SLMGAE benchmark).

        Both genes in test pairs are unseen during training.
        Tests ability to generalize to completely novel gene pairs.
        This is the hardest setting.
        """
        print(f"\nGenerating CV3 splits (pair-based) using SLMGAE benchmark code...")
        splitter = self._create_splitter()
        splits = splitter.cv3_split(k=num_folds, pos_neg_ratio=pos_neg_ratio)

        converted = []
        for split in splits:
            converted.append(self._convert_split_to_dataset_format(split))
            print(f"  Fold {split['fold']}: "
                  f"train={split['num_train_pos']}+/{split['num_train_neg']}- "
                  f"test={split['num_test_pos']}+/{split['num_test_neg']}-")

        return converted


def create_dataloaders(
    data: Dict,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from split data.

    Args:
        data: Dict with 'pairs' and 'labels' keys
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers

    Returns:
        DataLoader
    """
    # Note: This function expects embeddings to be passed separately
    # In practice, you'll use SLDataset directly with embeddings
    raise NotImplementedError(
        "Use SLDataset directly with embeddings tensor. "
        "Example: dataset = SLDataset(embeddings, pairs, labels)"
    )


def create_fold_dataloaders(
    embeddings: torch.Tensor,
    fold_data: Dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for a CV fold.

    Args:
        embeddings: Gene embeddings tensor (num_genes, dim)
        fold_data: Fold dict from get_cv*_splits()
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = SLDataset(
        embeddings=embeddings,
        pairs=fold_data["train"]["pairs"],
        labels=fold_data["train"]["labels"],
    )
    test_dataset = SLDataset(
        embeddings=embeddings,
        pairs=fold_data["test"]["pairs"],
        labels=fold_data["test"]["labels"],
    )

    # pin_memory only benefits CUDA, causes warnings on MPS
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, test_loader
