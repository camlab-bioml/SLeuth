#!/usr/bin/env python3
"""
PyTorch implementation of SLMGAE matching TensorFlow version exactly.
Uses sparse tensors and kNN caching for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict, Optional, Callable, Union


# Auto-detect CUDA availability
def get_device() -> torch.device:
    """Get the best available device with detailed info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device


def sparse_to_torch_sparse(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def normalize_adj(adj: Union[sp.spmatrix, np.ndarray]) -> sp.coo_matrix:
    """Symmetric normalization: Ã = D^{-1/2}(A+I)D^{-1/2} where D_ii = Σ_j(A+I)_ij"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # Self-loops: A+I
    rowsum = np.array(adj_.sum(1)).flatten()  # Degree vector
    # Handle isolated nodes (degree=0 leads to inf)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    degree_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^{-1/2}
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(
        degree_mat_inv_sqrt).tocoo()
    return adj_normalized


class GraphConvolution(nn.Module):
    """GCN layer: H' = σ(ÃXW) where Ã is normalized adjacency, W ∈ ℝ^{d×d'}"""

    def __init__(
            self,
            in_features,
            out_features,
            dropout=0.0,
            act=lambda x: F.leaky_relu(x, negative_slope=0.2),
    ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        # Weight W ~ Xavier/Glorot uniform
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights matching TensorFlow's Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """H' = σ(Ã·(Dropout(X)·W)) - transformation then aggregation"""
        # Dropout on input features
        x = F.dropout(x, self.dropout, training=self.training)

        # Linear transform: XW
        x = torch.mm(x, self.weight)

        # Graph convolution: Ã(XW); torch.sparse.mm is faster than torch.mm for sparse matrices and only takes sparse @ dense
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x)
        else:
            x = torch.mm(adj, x)

        # Activation: σ = LeakyReLU(α=0.2)
        if self.act is not None:
            x = self.act(x)

        return x


class GraphConvolutionSparse(nn.Module):
    """Sparse graph convolution layer for sparse inputs matching TensorFlow."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        act: Optional[Callable[
            [torch.Tensor],
            torch.Tensor]] = lambda x: F.leaky_relu(x, negative_slope=0.2),
    ):
        super(GraphConvolutionSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        # Weight initialization matching TF (no bias)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights matching TensorFlow's Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse input matching TF."""
        # Handle sparse input
        if x.is_sparse:
            # Convert to dense for dropout (simplified sparse dropout)
            x = x.to_dense()

        # 1. Apply dropout during training only, matching TF, not on weights but on input features
        x = F.dropout(x, self.dropout, training=self.training)

        # 2. Linear transformation
        x = torch.mm(x, self.weight)

        # 3. Graph convolution
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x)
        else:
            x = torch.mm(adj, x)

        # 4. Apply activation
        if self.act is not None:
            x = self.act(x)

        return x


class AttentionLayer(nn.Module):
    """Attention: R_att = Σ_m softmax(α_m)·R_m where α_m ∈ ℝ^{n×n}, m ∈ {1..M}"""

    def __init__(self, num_nodes: int, num_support: int):
        super(AttentionLayer, self).__init__()
        self.num_nodes = num_nodes
        self.num_support = num_support

        # Attention weights α ~ U(0.9, 1.1) initially; yes, this is a huge softmax weighting matrix but it's what SLMGAE did
        self.attweights = nn.Parameter(
            torch.empty(num_support, num_nodes, num_nodes).uniform_(0.9, 1.1))

    def forward(self, support_recs: List[torch.Tensor]) -> torch.Tensor:
        """R_att = Σ_m softmax(α_m)_ij · R^(m)_ij (element-wise attention)"""
        # Softmax over M support views: softmax(α)_m = exp(α_m)/Σ_k exp(α_k)
        attention = F.softmax(self.attweights, dim=0)

        # Weighted combination: Σ_m α'_m ⊙ R_m
        weighted_recs = []
        for i in range(self.num_support):
            weighted_recs.append(attention[i] * support_recs[i])

        return sum(weighted_recs)


class InnerProductDecoder(nn.Module):
    """Decoder: R = ZWZ^T where Z ∈ ℝ^{n×d}, W ∈ ℝ^{d×d} learnable"""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        # Weight matrix W for bilinear decoding
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction: R = Dropout(Z)·W·Dropout(Z)^T ∈ ℝ^{n×n}"""
        z = F.dropout(z, self.dropout, training=self.training)
        # Bilinear form: (ZW)Z^T
        x = torch.mm(z, self.weight)  # ZW: n×d
        adj_reconstructed = torch.mm(x, z.t())  # (ZW)Z^T: n×n
        # Identity activation (no sigmoid)
        return adj_reconstructed


class SLMGAE(nn.Module):
    """SLMGAE model matching TensorFlow implementation exactly."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.2,
        num_support_views: int = 3,
        device: Optional[torch.device] = None,
    ):
        super(SLMGAE, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.num_support_views = num_support_views
        self.device = device if device else get_device()

        # Validate num_support_views
        if num_support_views != 3:
            raise ValueError(
                f"SLMGAE requires exactly 3 support views, got {num_support_views}. "
                f"The architecture is designed for 3 support views + 1 main view = 4 branches total."
            )

        # First layer: 4 parallel SPARSE GCN layers (3 support + 1 main)
        # Matching TF GraphConvolutionSparse layers
        self.gcn_sparse_layers = nn.ModuleList()
        for i in range(4):  # 3 support + 1 main
            layer = GraphConvolutionSparse(
                num_features,
                hidden1,
                dropout=dropout,
                act=lambda x: F.leaky_relu(x, negative_slope=0.2),
            )
            self.gcn_sparse_layers.append(layer)

        # Second layer: 4 parallel DENSE GCN layers
        # Matching TF GraphConvolution layers
        self.gcn_dense_layers = nn.ModuleList()
        for i in range(4):
            layer = GraphConvolution(
                hidden1,
                hidden2,
                dropout=dropout,
                act=lambda x: F.leaky_relu(x, negative_slope=0.2),
            )
            self.gcn_dense_layers.append(layer)

        # Decoders: one for each branch
        self.decoders = nn.ModuleList()
        for i in range(4):  # 3 support + 1 main decoder
            decoder = InnerProductDecoder(hidden_dim=hidden2, dropout=dropout)
            self.decoders.append(decoder)

        # Attention layer for combining support views (matching TF)
        self.attention_layer = AttentionLayer(num_nodes=num_nodes,
                                              num_support=num_support_views)

    def forward(
        self,
        features: torch.Tensor,
        adjs: List[torch.Tensor],
        coe: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward: R = R_main + coe·R_att where R_main from main branch, R_att from attention.
        Args:
            features: X ∈ ℝ^{n×f} node features
            adjs: [Ã₁,Ã₂,Ã₃,Ã_main] normalized adjacencies
            coe: Weight λ for attention (default=2.0)
        Returns:
            Tuple of (reconstructions, main_rec, att, support_recs)
        """
        # Layer 1: H₁^(m) = GCN_sparse(X, Ã_m) for m ∈ {1,2,3,main}
        hidden1_outputs = []
        for i in range(4):
            hidden = self.gcn_sparse_layers[i](features, adjs[i])
            hidden1_outputs.append(hidden)

        # Layer 2: Z^(m) = GCN_dense(H₁^(m), Ã_m) ∈ ℝ^{n×d₂}
        hidden2_outputs = []
        for i in range(4):
            hidden = self.gcn_dense_layers[i](hidden1_outputs[i], adjs[i])
            hidden2_outputs.append(hidden)

        # Decode support views: R^(m) = Z^(m)W_mZ^(m)^T for m ∈ {1,2,3}
        support_recs = []
        for i in range(3):  # Support branches
            rec = self.decoders[i](hidden2_outputs[i])
            support_recs.append(rec)

        # Attention combination: R_att = Σ_m softmax(α_m)⊙R^(m)
        att = self.attention_layer(support_recs)

        # Decode main view: R_main = Z^(main)W_mainZ^(main)^T
        main_rec = self.decoders[3](hidden2_outputs[3])

        # Final: R = R_main + λ·R_att where λ=coe (default=2.0)
        reconstructions = main_rec + coe * att

        return reconstructions, main_rec, att, support_recs


class DataLoader:
    """Data loader matching TensorFlow data loading exactly."""

    def __init__(self, data_path="../data/", nn_size=45):
        self.data_path = data_path
        self.nn_size = nn_size

    def load_sl_matrix(self):
        """Load SL adjacency: A_ij = 1 if (i,j) is SL pair, 0 otherwise"""
        # Gene index mapping: gene_name → index ∈ [0, n-1]
        sl_mapping = {}
        with open(f"{self.data_path}/List_Proteins_in_SL.txt", "r") as f:
            for idx, line in enumerate(f):
                gene = line.strip()
                sl_mapping[gene] = idx

        # Infer num_nodes from gene list
        num_nodes = len(sl_mapping)

        # Load SL edges
        row, col = [], []
        with open(f"{self.data_path}/SL_Human_Approved.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    gene1, gene2 = parts[0], parts[1]
                    if gene1 in sl_mapping and gene2 in sl_mapping:
                        row.append(sl_mapping[gene1])
                        col.append(sl_mapping[gene2])

        # Create adjacency matrix
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                            shape=(num_nodes, num_nodes))
        adj = adj + adj.T  # Make symmetric
        adj = adj.toarray()
        adj[adj != 0] = (
            1  # Convert non-zero to 1; this is just to ensure the the adjacency matrix is binary
        )
        np.fill_diagonal(
            adj, 0)  # Set diagonal to 0; this is to ensure the diagonal is 0

        # Get positive and negative edges
        x, y = np.triu_indices(num_nodes, k=1)
        pos_edges, neg_edges = [], []

        for e in zip(x, y):
            if adj[e[0], e[1]] == 0:
                neg_edges.append(e)
            else:
                pos_edges.append(e)

        pos_edges = np.array(pos_edges, dtype=np.int32)
        neg_edges = np.array(neg_edges, dtype=np.int32)

        return pos_edges, neg_edges, num_nodes

    def build_knn_matrix(self, S, nn_size):
        """Build kNN graph: keep top-k neighbors per node based on similarity S"""
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(nn_size, n)]
            X[i, ii] = S[i, ii]
        return X

    def load_dense_feature(self, filename, knn=True):
        """Load dense feature matrix matching TensorFlow implementation."""
        print(f"Loading {filename}")

        # Load using TensorFlow's approach: upper triangular format with j+1 offset
        # File format: row i contains columns from i onwards (triangular)
        # TensorFlow places them at featureMatrix[i][j+1] (1-indexed columns)
        with open(filename, "r") as f:
            lines = f.readlines()

        feature_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(len(lines)):
            if i >= self.num_nodes:
                break
            parts = lines[i].replace("\n", "").split("\t")
            for j in range(i, len(parts)):
                if j >= self.num_nodes - 1:  # j+1 must be < num_nodes
                    break
                if parts[j] == "":
                    break
                feature_matrix[i][j + 1] = float(parts[j])

        # First symmetrization (before KNN)
        feature_matrix = feature_matrix + feature_matrix.T

        if knn:
            feature_matrix = self.build_knn_matrix(feature_matrix,
                                                   self.nn_size)

        # Second symmetrization (after KNN) - matching TensorFlow
        # NOTE: This doubles edge weights if both nodes are in each other's top-k.
        # While mathematically questionable, this matches the original TensorFlow
        # implementation for exact reproducibility.
        feature_matrix = feature_matrix + feature_matrix.T

        # Convert to a sparse format for consistency
        coo_matrix = sp.coo_matrix(feature_matrix)

        return coo_matrix

    def load_sparse_feature(self, filename):
        """Load sparse feature matrix matching TF implementation."""
        print(f"Loading {filename}")
        row, col = [], []

        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    r, c = int(parts[0]), int(parts[1])
                    if r < self.num_nodes and c < self.num_nodes:
                        row.append(r)
                        col.append(c)

        features = sp.coo_matrix((np.ones(len(row)), (row, col)),
                                 shape=(self.num_nodes, self.num_nodes))
        features = features + features.T

        return features

    def load_data(self):
        """Load all data matching TF implementation."""
        # Load SL matrix and infer num_nodes from gene list
        pos_edges, neg_edges, num_nodes = self.load_sl_matrix()
        self.num_nodes = num_nodes  # Store for use in load_dense/sparse_feature

        # Load support views
        support_adjs = []
        support_adjs.append(
            self.load_dense_feature(f"{self.data_path}/Human_GOsim.txt",
                                    knn=True))
        support_adjs.append(
            self.load_dense_feature(f"{self.data_path}/Human_GOsim_CC.txt",
                                    knn=True))
        support_adjs.append(
            self.load_sparse_feature(
                f"{self.data_path}/biogrid_ppi_sparse.txt"))

        return pos_edges, neg_edges, support_adjs, num_nodes


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
