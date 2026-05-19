#!/usr/bin/env python3
"""
Data utilities for Graph Transformer.
Handles data loading, graph construction, and linearization generation.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import random
from sklearn.model_selection import train_test_split


def load_and_split_sldb_data(
        file_path: str,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load SLDB data and split into train/val/test sets.
    Split is done BEFORE graph construction to prevent data leakage.
    
    Args:
        file_path: Path to SLDB CSV file
        test_ratio: Fraction of data for test set
        val_ratio: Fraction of data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df: DataFrames with gene pairs and scores
    """
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} gene pairs from {file_path}")

    # Extract gene names from combination column
    df[['gene1', 'gene2']] = df['gene_combination'].str.split(';', expand=True)

    # Create stratification based on score ranks for balanced splits
    df['sens.score'] = pd.to_numeric(df['sens.score'], errors='coerce')
    df = df.dropna(subset=['sens.score'])
    df = df.sort_values('sens.score')
    df['rank'] = range(1, len(df) + 1)

    # Use top 5% for stratification to ensure SL pairs are distributed
    stratify_col = (df['rank'] <= len(df) * 0.05).astype(int)

    # First split: separate test set
    train_val_df, test_df = train_test_split(df,
                                             test_size=test_ratio,
                                             random_state=random_state,
                                             stratify=stratify_col)

    # Second split: separate validation from training
    stratify_col_train = (train_val_df['rank']
                          <= len(train_val_df) * 0.05).astype(int)
    train_df, val_df = train_test_split(train_val_df,
                                        test_size=val_ratio / (1 - test_ratio),
                                        random_state=random_state,
                                        stratify=stratify_col_train)

    print(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def build_graph_from_train_data(
        train_df: pd.DataFrame,
        all_genes: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build adjacency matrix using ONLY training data edges.
    This prevents data leakage from val/test sets.
    
    Args:
        train_df: Training dataframe with gene pairs
        all_genes: List of all genes in the dataset
        
    Returns:
        adj_matrix: Weighted adjacency matrix (n_genes x n_genes)
        gene_to_idx: Mapping from gene names to indices
    """
    # Create gene index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    n_nodes = len(all_genes)

    # Initialize adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    # Fill adjacency matrix with edge weights from training data
    edge_count = 0
    for _, row in train_df.iterrows():
        if row['gene1'] in gene_to_idx and row['gene2'] in gene_to_idx:
            i = gene_to_idx[row['gene1']]
            j = gene_to_idx[row['gene2']]

            # Use absolute sensitivity score as edge weight
            weight = abs(row['sens.score'])

            # Undirected graph - set both directions
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
            edge_count += 1

    print(
        f"Added {edge_count} edges to graph (density: {edge_count/(n_nodes**2):.4f})"
    )

    return adj_matrix, gene_to_idx


class GraphLinearizer:
    """
    Generate multiple diverse linearizations of a graph.
    Each linearization σ: V→[n] is a bijective ordering of all nodes.
    Diversity metric: d(σ₁,σ₂) = αd_pos + (1-α)d_rank where α=0.6
    """

    def __init__(self, adj_matrix: np.ndarray):
        """
        Args:
            adj_matrix: Adjacency matrix of the graph
        """
        self.adj_matrix = adj_matrix
        self.n_nodes = adj_matrix.shape[0]

        # Build adjacency list for efficient traversal
        self.adj_list = self._build_adjacency_list()

        # Compute node degrees for degree-based strategies
        self.degrees = np.sum(adj_matrix > 0, axis=1)

    def _build_adjacency_list(self) -> Dict[int, List[int]]:
        """Convert adjacency matrix to adjacency list format."""
        adj_list = defaultdict(list)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adj_matrix[i, j] > 0 and i != j:
                    adj_list[i].append(j)

        return dict(adj_list)

    def generate_multiple_linearizations(
            self,
            num_linearizations: int = 100,
            min_diversity_threshold: float = 0.3,
            max_attempts_per_lin: int = 20) -> List[List[int]]:
        """
        Generate multiple diverse linearizations using different strategies.
        
        Args:
            num_linearizations: Number of linearizations to generate
            min_diversity_threshold: Minimum required diversity score (0-1)
            max_attempts_per_lin: Maximum attempts to generate each unique linearization
            
        Returns:
            List of linearizations, each is a list of node indices
        """
        linearizations = []
        diversity_scores = []

        # Use different strategies to ensure diversity
        strategies = [
            self._bfs_linearization, self._dfs_linearization,
            self._random_walk_linearization, self._degree_based_linearization,
            self._hybrid_linearization
        ]

        print(f"Generating {num_linearizations} diverse linearizations...")

        # Generate linearizations with diversity checking
        attempt_count = 0
        max_total_attempts = num_linearizations * max_attempts_per_lin

        while len(
                linearizations
        ) < num_linearizations and attempt_count < max_total_attempts:
            strategy_idx = len(linearizations) % len(strategies)
            strategy = strategies[strategy_idx]

            # Generate candidate linearization
            linearization = strategy(seed=attempt_count)
            attempt_count += 1

            # Skip if incomplete
            if len(linearization) != self.n_nodes:
                continue

            # Check diversity against existing linearizations
            if linearizations:
                diversity_score = self._compute_linearization_diversity(
                    linearization, linearizations)

                if diversity_score < min_diversity_threshold:
                    continue  # Too similar, try again

                diversity_scores.append(diversity_score)

            linearizations.append(linearization)

            if len(linearizations) % 20 == 0:
                print(
                    f"Generated {len(linearizations)}/{num_linearizations} linearizations"
                )

        print(f"Generated {len(linearizations)} complete linearizations")

        # Analyze diversity
        self._analyze_linearization_diversity(linearizations, diversity_scores)

        return linearizations

    def _compute_linearization_diversity(self, candidate: List[int],
                                         existing: List[List[int]]) -> float:
        """
        Compute diversity score of candidate linearization against existing ones.
        d(σ,τ) = 0.6·(|{i: σ(i)≠τ(i)}|/n) + 0.4·mean(|rank_σ(v)-rank_τ(v)|/(n-1))
        
        Args:
            candidate: Candidate linearization σ
            existing: List of existing linearizations {τᵢ}
            
        Returns:
            Diversity score ∈ [0,1] (0 = identical, 1 = maximally different)
        """
        if not existing:
            return 1.0

        # Compute average position-wise difference
        total_diversity = 0.0

        for existing_lin in existing:
            # Position diversity: d_pos = |{i: σ(i)≠τ(i)}|/n
            position_diff = sum(
                1 for i, (a, b) in enumerate(zip(candidate, existing_lin))
                if a != b)
            position_diversity = position_diff / len(candidate)

            # Rank diversity: d_rank = mean(|rank_σ(v)-rank_τ(v)|/(n-1))
            candidate_ranks = {
                node: i
                for i, node in enumerate(candidate)
            }  # σ⁻¹
            existing_ranks = {
                node: i
                for i, node in enumerate(existing_lin)
            }  # τ⁻¹

            rank_differences = []
            for node in candidate:
                rank_diff = abs(candidate_ranks[node] - existing_ranks[node])
                normalized_diff = rank_diff / (len(candidate) - 1)
                rank_differences.append(normalized_diff)

            rank_diversity = np.mean(rank_differences)

            # Combined: d(σ,τ) = 0.6·d_pos + 0.4·d_rank
            combined_diversity = 0.6 * position_diversity + 0.4 * rank_diversity
            total_diversity += combined_diversity

        # Return average diversity against all existing linearizations
        return total_diversity / len(existing)

    def _analyze_linearization_diversity(
            self, linearizations: List[List[int]],
            diversity_scores: List[float]) -> None:
        """Analyze and report diversity statistics."""
        if not linearizations:
            return

        print("\n=== Linearization Diversity Analysis ===")

        # Starting node diversity
        unique_starts = len(set(lin[0] for lin in linearizations))
        print(f"Unique starting nodes: {unique_starts}/{len(linearizations)} "
              f"({unique_starts/len(linearizations)*100:.1f}%)")

        # Ending node diversity
        unique_ends = len(set(lin[-1] for lin in linearizations))
        print(f"Unique ending nodes: {unique_ends}/{len(linearizations)} "
              f"({unique_ends/len(linearizations)*100:.1f}%)")

        # Diversity score statistics
        if diversity_scores:
            print(f"Diversity scores - Mean: {np.mean(diversity_scores):.3f}, "
                  f"Std: {np.std(diversity_scores):.3f}, "
                  f"Min: {np.min(diversity_scores):.3f}, "
                  f"Max: {np.max(diversity_scores):.3f}")

        # Sample a few linearizations to show variety
        if len(linearizations) >= 3:
            print("\nSample linearization prefixes (first 10 nodes):")
            sample_indices = [
                0, len(linearizations) // 2,
                len(linearizations) - 1
            ]
            for i, idx in enumerate(sample_indices):
                prefix = linearizations[idx][:10]
                print(f"  Lin {idx+1}: {prefix}")

        # Check for exact duplicates (shouldn't happen with diversity checking)
        unique_lins = set(tuple(lin) for lin in linearizations)
        if len(unique_lins) < len(linearizations):
            print(
                f"Warning: {len(linearizations) - len(unique_lins)} duplicate linearizations found!"
            )

        print("=" * 45)

    def _bfs_linearization(self, seed: int) -> List[int]:
        """BFS linearization: σ_BFS with P(start=v) ∝ deg(v) for high-degree nodes."""
        random.seed(seed)
        np.random.seed(seed)  # Also set numpy seed for consistency

        # Choose random start node (prefer high-degree nodes)
        if random.random() < 0.7:  # 70% chance to start from high-degree node
            start_node = np.random.choice(
                np.argsort(self.degrees)[-max(1, self.n_nodes // 10):])
        else:
            start_node = random.randint(0, self.n_nodes - 1)

        visited = set()
        queue = deque([start_node])
        linearization = []

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            linearization.append(node)

            # Add neighbors in random order for diversity
            neighbors = self.adj_list.get(node, [])
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Add any disconnected nodes
        for node in range(self.n_nodes):
            if node not in visited:
                linearization.append(node)

        return linearization

    def _dfs_linearization(self, seed: int) -> List[int]:
        """Depth-first search linearization with randomization."""
        random.seed(seed)
        np.random.seed(seed)

        # Random start node
        start_node = random.randint(0, self.n_nodes - 1)

        visited = set()
        stack = [start_node]
        linearization = []

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            linearization.append(node)

            # Add neighbors in random order
            neighbors = self.adj_list.get(node, [])
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

        # Add disconnected nodes
        remaining = list(set(range(self.n_nodes)) - visited)
        random.shuffle(remaining)
        linearization.extend(remaining)

        return linearization

    def _random_walk_linearization(self, seed: int) -> List[int]:
        """Random walk with restart: P(restart)=0.1, P(next|v)=Uniform(N(v)∩unvisited)."""
        random.seed(seed)
        np.random.seed(seed)

        visited = set()
        linearization = []

        # Start from random node
        current = random.randint(0, self.n_nodes - 1)

        # Random walk with restart
        restart_prob = 0.1

        while len(visited) < self.n_nodes:
            if current not in visited:
                visited.add(current)
                linearization.append(current)

            # Get unvisited neighbors
            neighbors = [
                n for n in self.adj_list.get(current, []) if n not in visited
            ]

            if neighbors and random.random() > restart_prob:
                # Continue walk to random neighbor
                current = random.choice(neighbors)
            else:
                # Restart from unvisited node
                unvisited = list(set(range(self.n_nodes)) - visited)
                if unvisited:
                    current = random.choice(unvisited)
                else:
                    break

        return linearization

    def _degree_based_linearization(self, seed: int) -> List[int]:
        """Degree-based: σ_deg where σ⁻¹(v) ∝ deg(v) + ε, ε~U(0,1) for randomness."""
        random.seed(seed)
        np.random.seed(seed)

        # Create degree-based ordering with randomization
        nodes_degrees = list(enumerate(self.degrees))

        if seed % 3 == 0:
            # High degree first
            nodes_degrees.sort(key=lambda x: x[1], reverse=True)
        elif seed % 3 == 1:
            # Low degree first
            nodes_degrees.sort(key=lambda x: x[1])
        else:
            # Random with degree bias
            nodes_degrees.sort(key=lambda x: x[1] + random.random(),
                               reverse=True)

        # Extract node indices
        linearization = [node for node, _ in nodes_degrees]

        return linearization

    def _hybrid_linearization(self, seed: int) -> List[int]:
        """Hybrid strategy combining multiple approaches."""
        random.seed(seed)
        np.random.seed(seed)

        visited = set()
        linearization = []

        # Phase 1: Start with high-degree nodes (hubs)
        hub_threshold = np.percentile(self.degrees, 90)
        hubs = [
            i for i in range(self.n_nodes) if self.degrees[i] >= hub_threshold
        ]
        random.shuffle(hubs)

        for hub in hubs[:max(1, len(hubs) // 3)]:
            if hub not in visited:
                # BFS from hub
                queue = deque([hub])
                bfs_count = 0
                max_bfs = self.n_nodes // 10  # Limit BFS extent

                while queue and bfs_count < max_bfs:
                    node = queue.popleft()
                    if node not in visited:
                        visited.add(node)
                        linearization.append(node)
                        bfs_count += 1

                        neighbors = self.adj_list.get(node, [])
                        random.shuffle(neighbors)
                        queue.extend(
                            [n for n in neighbors if n not in visited])

        # Phase 2: Random walk for remaining nodes
        while len(visited) < self.n_nodes:
            # Pick random unvisited node
            unvisited = list(set(range(self.n_nodes)) - visited)
            start = random.choice(unvisited)

            # Short random walk
            current = start
            walk_length = min(10, len(unvisited))

            for _ in range(walk_length):
                if current not in visited:
                    visited.add(current)
                    linearization.append(current)

                neighbors = [
                    n for n in self.adj_list.get(current, [])
                    if n not in visited
                ]
                if neighbors:
                    current = random.choice(neighbors)
                else:
                    break

        return linearization


def prepare_support_views(adj_matrix: np.ndarray,
                          n_support_views: int,
                          support_view_dim: int,
                          support_view_paths: Optional[List[str]] = None,
                          use_biological_views: bool = True,
                          num_nodes: int = None) -> Tuple[torch.Tensor, Dict]:
    """
    Prepare support view features S = {f_m: E→ℝᴰ}ᴹ_m=1 for edges.
    
    Args:
        adj_matrix: Graph adjacency matrix
        n_support_views: Number of support views
        support_view_dim: Dimension of each support view
        support_view_paths: Optional paths to load support views from
        use_biological_views: Whether to use biological views from main study
        num_nodes: Number of nodes (required if use_biological_views=True)
        
    Returns:
        support_views: Tensor of shape (n_edges, n_support_views, support_view_dim)
        edge_map: Dictionary mapping (i, j) node pairs to edge indices
    """
    # If using biological views from main study
    if use_biological_views and n_support_views == 3:
        print(
            "Loading biological support views from main study (GO-BP, GO-CC, PPI)..."
        )
        from load_support_views import prepare_support_views_from_study
        return prepare_support_views_from_study(
            adj_matrix=adj_matrix,
            num_nodes=num_nodes,
            support_view_dim=support_view_dim)

    # Create edge mapping
    edge_map = create_edge_mapping(adj_matrix)
    n_edges = len(edge_map)

    if support_view_paths and len(support_view_paths) == n_support_views:
        # Load support views from files
        print(f"Loading {n_support_views} support views from files...")
        support_views_list = []

        for i, path in enumerate(support_view_paths):
            try:
                # Placeholder - adjust based on actual file format
                view_data = np.load(path)

                # Validate shape
                if len(view_data.shape) == 3:
                    # Assume shape is (n_nodes, n_nodes, feature_dim)
                    n_nodes_data = view_data.shape[0]
                    assert view_data.shape[0] == view_data.shape[
                        1], f"Expected square matrix, got {view_data.shape}"
                    assert n_nodes_data >= adj_matrix.shape[
                        0], f"Not enough nodes in data: {n_nodes_data} < {adj_matrix.shape[0]}"

                    # Extract features for edges in edge_map
                    view_features = np.zeros((n_edges, support_view_dim))

                    for (node_i, node_j), edge_idx in edge_map.items():
                        # Extract feature for this edge
                        if node_i < n_nodes_data and node_j < n_nodes_data:
                            view_features[edge_idx] = view_data[
                                node_i, node_j, :support_view_dim]
                else:
                    raise ValueError(
                        f"Unexpected shape for support view data: {view_data.shape}"
                    )

                support_views_list.append(view_features)
                print(f"  Loaded support view {i+1} from {path}")

            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
                # Use random initialization as fallback
                support_views_list.append(
                    np.random.randn(n_edges, support_view_dim) * 0.1)

        # Stack all views
        support_views = np.stack(support_views_list, axis=1)

    else:
        # Random initialization for placeholder
        print(f"Initializing {n_support_views} random support views...")
        support_views = np.random.randn(n_edges, n_support_views,
                                        support_view_dim) * 0.1

    # Return tensor on CPU - caller should move to appropriate device
    return torch.FloatTensor(support_views), edge_map


def create_edge_mapping(adj_matrix: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Create bijection φ: {(i,j): A_ij>0, i<j} → [|E|].
    Uses canonical edge representation (i,j) where i<j for undirected graphs.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Dictionary mapping (i, j) tuples to edge indices
    """
    edge_map = {}
    edge_idx = 0

    n_nodes = adj_matrix.shape[0]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only upper triangle for undirected
            if adj_matrix[i, j] > 0:
                edge_map[(i, j)] = edge_idx
                edge_idx += 1

    return edge_map


def load_esm_embeddings(
        gene_list: List[str],
        embedding_path: Optional[str] = None,
        embedding_dim: int = 1280,
        model_name: str = "esm2_t33_650M_UR50D") -> torch.Tensor:
    """
    Load ESM embeddings X ∈ ℝⁿˣᵈ where d=1280 (ESM-2 650M model).
    
    Args:
        gene_list: List of gene names
        embedding_path: Optional path to precomputed embedding file (NPZ, PT, or H5)
        embedding_dim: Dimension of embeddings
        model_name: ESM model name for generating embeddings
        
    Returns:
        Tensor of shape (n_genes, embedding_dim)
    """
    n_genes = len(gene_list)

    if embedding_path and os.path.exists(embedding_path):
        try:
            print(
                f"Loading precomputed ESM embeddings from {embedding_path}...")

            # Support multiple formats
            if embedding_path.endswith('.npz'):
                data = np.load(embedding_path)
                embeddings = data['embeddings']
                gene_names = data.get('gene_names', None)
            elif embedding_path.endswith('.pt') or embedding_path.endswith(
                    '.pth'):
                data = torch.load(embedding_path, map_location='cpu')
                if isinstance(data, dict) and 'embeddings' in data:
                    embeddings = data['embeddings'].numpy()
                    gene_names = data.get('gene_order',
                                          data.get('gene_names', None))
                else:
                    embeddings = data.numpy() if hasattr(data,
                                                         'numpy') else data
                    gene_names = None
            elif embedding_path.endswith('.h5'):
                import h5py
                with h5py.File(embedding_path, 'r') as f:
                    embeddings = f['embeddings'][:]
                    gene_names = f.get('gene_names', None)
                    if gene_names is not None:
                        gene_names = [
                            g.decode() if isinstance(g, bytes) else g
                            for g in gene_names
                        ]
            else:
                raise ValueError(f"Unsupported file format: {embedding_path}")

            # Map embeddings to gene list order
            if gene_names is not None:
                gene_to_embedding = dict(zip(gene_names, embeddings))
                ordered_embeddings = []
                missing_genes = []

                for gene in gene_list:
                    if gene in gene_to_embedding:
                        ordered_embeddings.append(gene_to_embedding[gene])
                    else:
                        missing_genes.append(gene)
                        # Use average embedding for missing genes
                        ordered_embeddings.append(np.mean(embeddings, axis=0))

                if missing_genes:
                    print(
                        f"Warning: {len(missing_genes)} genes not found in embeddings, using average embedding"
                    )

                embeddings = np.array(ordered_embeddings)
            else:
                # Assume embeddings are in same order as gene_list
                if embeddings.shape[0] != n_genes:
                    print(
                        f"Warning: Embedding count mismatch ({embeddings.shape[0]} vs {n_genes})"
                    )
                    if embeddings.shape[0] > n_genes:
                        embeddings = embeddings[:n_genes]
                    else:
                        # Pad with average embeddings
                        avg_embedding = np.mean(embeddings, axis=0)
                        padding = np.tile(avg_embedding,
                                          (n_genes - embeddings.shape[0], 1))
                        embeddings = np.vstack([embeddings, padding])

            print(f"Loaded embeddings shape: {embeddings.shape}")

        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("Falling back to on-the-fly generation...")
            embeddings = _generate_esm_embeddings(gene_list, model_name,
                                                  embedding_dim)
    else:
        print(
            f"No embedding file provided, generating ESM embeddings on-the-fly..."
        )
        embeddings = _generate_esm_embeddings(gene_list, model_name,
                                              embedding_dim)

    # Return tensor on CPU - caller should move to appropriate device
    return torch.FloatTensor(embeddings)


def _generate_esm_embeddings(gene_list: List[str],
                             model_name: str = "esm2_t33_650M_UR50D",
                             embedding_dim: int = 1280) -> np.ndarray:
    """
    Generate ESM embeddings on-the-fly using transformers library.
    """
    try:
        from transformers import EsmModel, EsmTokenizer
        import torch

        print(f"Loading ESM model: {model_name}")
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{model_name}")
        model = EsmModel.from_pretrained(f"facebook/{model_name}")
        model.eval()

        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        embeddings = []
        batch_size = 8  # Adjust based on GPU memory

        print(f"Generating embeddings for {len(gene_list)} genes...")

        # Note: This assumes gene names correspond to protein sequences
        # In practice, you'd need a mapping from gene names to sequences
        for i in range(0, len(gene_list), batch_size):
            batch_genes = gene_list[i:i + batch_size]

            # Placeholder: generate dummy sequences based on gene names
            # In practice, load actual protein sequences from FASTA or database
            sequences = [_gene_to_dummy_sequence(gene) for gene in batch_genes]

            # Tokenize sequences
            inputs = tokenizer(sequences,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling of last hidden states
                last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_embeddings = last_hidden_states * attention_mask
                batch_embeddings = masked_embeddings.sum(
                    dim=1) / attention_mask.sum(dim=1)

                embeddings.extend(batch_embeddings.cpu().numpy())

        embeddings = np.array(embeddings)
        print(f"Generated embeddings shape: {embeddings.shape}")

        # Save for future use
        save_path = f"generated_esm_embeddings_{len(gene_list)}_genes.npz"
        np.savez(save_path, embeddings=embeddings, gene_names=gene_list)
        print(f"Saved embeddings to {save_path}")

        return embeddings

    except ImportError:
        print("transformers library not available, using random embeddings")
        return np.random.randn(len(gene_list), embedding_dim) * 0.1
    except Exception as e:
        print(f"Error generating ESM embeddings: {e}")
        print("Using random embeddings as fallback")
        return np.random.randn(len(gene_list), embedding_dim) * 0.1


def _gene_to_dummy_sequence(gene_name: str, length: int = 200) -> str:
    """
    Generate a dummy protein sequence based on gene name.
    In practice, this should load actual sequences from a database.
    """
    # Use gene name to seed random sequence generation for consistency
    import hashlib
    seed = int(hashlib.md5(gene_name.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    sequence = ''.join(np.random.choice(list(amino_acids), length))

    return sequence


# === Gene Order Compatibility Functions ===


def load_esm_gene_order():
    """Load the correct gene order to match ESM embeddings."""
    import os
    from pathlib import Path

    try:
        # Load from gene mapping file (genes with ESM embeddings)
        gene_mapping_file = Path(
            '../data/main_gene_mapping_with_sequences.txt')
        if gene_mapping_file.exists():
            esm_genes = []
            with open(gene_mapping_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            esm_genes.append(
                                parts[1])  # Gene name in second column
            print(f"✅ Using ESM gene order: {len(esm_genes)} genes")
            return esm_genes

        # Fallback to original gene list
        protein_list_file = Path('../data/List_Proteins_in_SL.txt')
        if protein_list_file.exists():
            with open(protein_list_file, 'r') as f:
                genes = [
                    line.strip() for line in f
                    if line.strip() and not line.startswith('#')
                ]
            print(f"⚠️  Using fallback gene order: {len(genes)} genes")
            return genes

        raise FileNotFoundError("No gene order files found")

    except Exception as e:
        print(f"❌ Error loading gene order: {e}")
        raise


def filter_training_data_by_genes(train_df, val_df, test_df, gene_list):
    """Filter training data to only include genes that exist in our gene list."""
    gene_set = set(gene_list)

    def filter_df(df, name):
        # Filter to only include gene pairs where both genes are in our gene list
        original_size = len(df)
        df_filtered = df[df['gene1'].isin(gene_set)
                         & df['gene2'].isin(gene_set)].copy()
        filtered_size = len(df_filtered)
        print(f"   {name}: {original_size} → {filtered_size} pairs "
              f"({filtered_size/original_size*100:.1f}% retained)")
        return df_filtered

    print("Filtering training data to match ESM gene set:")
    train_df_filtered = filter_df(train_df, "Train")
    val_df_filtered = filter_df(val_df, "Val")
    test_df_filtered = filter_df(test_df, "Test")

    return train_df_filtered, val_df_filtered, test_df_filtered
