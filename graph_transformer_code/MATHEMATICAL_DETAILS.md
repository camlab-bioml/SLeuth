# Mathematical Details of Graph Transformer with Weighted Support Views

## Overview

This document provides a comprehensive mathematical description of the Graph Transformer with weighted support views for synthetic lethality (SL) prediction. The implementation combines graph linearization with transformer architecture and learnable support view weighting to predict SL interactions between gene pairs.

## Code Architecture Overview

The implementation consists of several key components:

- **GraphLinearizer**: Generates diverse graph traversal orders
- **WeightedSupportViewAttention**: Custom attention with edge biases
- **GraphTransformer**: Main model combining linearization and attention
- **Training Loop**: Stochastic sampling of linearizations for data augmentation

## 1. Problem Formulation

Given a graph $G = (V, E)$ with:

- Nodes $V = {v_1, v_2, ..., v_n}$ representing genes
- Edges $E \\subseteq V \\times V$ representing gene interactions
- Edge weights $w\_{ij}$ from sensitivity scores

We aim to predict synthetic lethality (SL) for gene pairs by:

1. Creating multiple complete linearizations of $G$
1. Processing each linearization as an independent sequence
1. Aggregating information across linearizations

## 2. Graph Linearization

### 2.1 Definition

A linearization $\\pi: V \\rightarrow {1, 2, ..., n}$ is a bijective mapping that assigns each node a position in a sequence.

### 2.2 Multiple Linearization Strategies

We generate $K$ different linearizations ${\\pi_1, \\pi_2, ..., \\pi_K}$ using:

**1. BFS-based Linearization:**

```
π_BFS(s) = BFS traversal starting from node s
```

**2. DFS-based Linearization:**

```
π_DFS(s) = Randomized DFS traversal from node s
```

**3. Degree-based Linearization:**

```
π_degree = Nodes ordered by degree d(v) = Σ_u 1[w_uv > 0]
```

**4. Random Priority Linearization:**

```
π_random = Nodes ordered by random priorities p(v) ~ U(0,1)
```

### 2.3 Properties

Each linearization $\\pi_k$ satisfies:

- **Completeness**: $|\\pi_k(V)| = |V| = n$
- **Uniqueness**: $\\pi_k(v_i) \\neq \\pi_k(v_j)$ for $i \\neq j$
- **Independence**: Different $\\pi_k$ provide different orderings

## 3. Sequence Representation

### 3.1 Node Features

For each node $v_i$, we have ESM protein embeddings:
$$x_i = e_i \\in \\mathbb{R}^{d_e}$$

where $e_i$ is the ESM embedding for gene $i$ (dimension 1280 for ESM2-650M).

### 3.2 Edge Features

For each edge $(i,j) \\in E$, we have support view features:
$$s\_{ij} = [s\_{ij}^{(1)} || s\_{ij}^{(2)} || s\_{ij}^{(3)}] \\in \\mathbb{R}^{d_s}$$

where:

- $s\_{ij}^{(1)}$ is GO-BP (Biological Process) similarity
- $s\_{ij}^{(2)}$ is GO-CC (Cellular Component) similarity
- $s\_{ij}^{(3)}$ is PPI (Protein-Protein Interaction) network edge

### 3.3 Linearized Sequences

For linearization $\\pi_k$, create sequence:
$$S_k = [x\_{\\pi_k^{-1}(1)}, x\_{\\pi_k^{-1}(2)}, ..., x\_{\\pi_k^{-1}(n)}]$$

where $\\pi_k^{-1}(j)$ gives the node at position $j$.

## 4. Transformer Processing

### 4.1 Input Projection

Project features to hidden dimension:
$$h_i^{(0)} = W\_{proj} \\cdot x_i + b\_{proj}$$

### 4.2 Positional Encoding

Add learnable positional embeddings:
$$h_i^{(0)} = h_i^{(0)} + PE(i)$$

where $PE(i) \\in \\mathbb{R}^{d\_{hidden}}$ is the positional encoding for position $i$.

### 4.3 Edge-Aware Multi-Head Self-Attention

For each linearization $S_k$, apply $L$ transformer layers with edge-aware attention:

#### Standard Attention:

$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

#### Edge-Aware Attention with Weighted Support Views:

For attention head $h$, the score between positions $i$ and $j$ is:
$$\\alpha\_{ij}^{(h)} = \\frac{\\langle q_i^{(h)}, k_j^{(h)} \\rangle}{\\sqrt{d_k}} + b\_{ij}^{(h)}$$

**Weighted Support View Implementation:**

Each support view $m$ has a learnable weight $\\alpha_m$ (squared for non-negativity):
$$w_m = \\alpha_m^2$$

The combined edge features are:
$$\\tilde{s}_{ij} = \\sum_{m=1}^{M} w_m \\cdot s\_{ij}^{(m)} = \\sum\_{m=1}^{M} \\alpha_m^2 \\cdot s\_{ij}^{(m)}$$

where:

- $M$ is the number of support views
- $s\_{ij}^{(m)} \\in \\mathbb{R}^{d_s}$ is support view $m$ for edge $(i,j)$
- $\\alpha_m$ are learnable parameters (initialized near 1.0)

The edge bias for head $h$ is:
$$b\_{ij}^{(h)} = W_h^T \\cdot \\tilde{s}\_{ij}$$

where $W_h \\in \\mathbb{R}^{d_s}$ projects the weighted features to a scalar bias for head $h$.

The edge-aware attention becomes:
$$\\text{EdgeAttention}(Q,K,V,S,\\alpha) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + B\\right)V$$

where $B\_{h,i,j} = b\_{ij}^{(h)}$ if edge $(i,j)$ exists in the linearization.

### 4.4 Node Representations

After transformer processing, for linearization $\\pi_k$, node $v_i$ at position $j = \\pi_k(v_i)$ has representation:
$$r_i^{(k)} = H_k^{(L)}[j, :]$$

## 5. Independent Linearization Training

Each linearization is treated as an independent training sample:

### 5.1 Training Process

For each batch:

1. Randomly sample a linearization: $\\pi_k \\sim {\\pi_1, ..., \\pi_K}$
1. Process the graph using this ordering
1. Compute loss and update shared parameters

### 5.2 Key Properties

- **Parameter Sharing**: All linearizations share the same $W_Q, W_K, W_V$, FFN, and edge projection weights
- **Data Augmentation**: Each edge $(u,v)$ is seen in $K$ different sequential contexts
- **Stochastic Training**: Different linearizations provide different views of the same graph structure

### 5.3 Inference Options

**Option 1 - Single Linearization:**
$$\\hat{y}_{uv} = f_\\theta(G, \\pi_k)$$

**Option 2 - Ensemble Average:**
$$\\hat{y}_{uv} = \\frac{1}{K'} \\sum_{k=1}^{K'} f\_\\theta(G, \\pi_k)$$

where $K' \\leq K$ is the number of linearizations used at inference.

### 5.4 Advantages

This approach is analogous to data augmentation in computer vision:

- Random crops → Random linearizations
- Different views of same image → Different orderings of same graph
- Improved generalization through stochastic regularization
- Simpler architecture without aggregation mechanisms

## 6. Edge Prediction

### 6.1 Edge Feature Construction

For edge $(u,v)$:
$$f\_{uv} = [r_u^{final} || r_v^{final} || s\_{uv}]$$

where $s\_{uv}$ is the support view feature for edge $(u,v)$.

### 6.2 SL Prediction

$$\\hat{y}_{uv} = \\sigma(\\text{MLP}_{edge}(f\_{uv}))$$

where $\\text{MLP}\_{edge}$ is a multi-layer perceptron that processes both node representations and edge features.

## 7. Training

### 7.1 Loss Function

Binary cross-entropy for SL prediction:
$$\\mathcal{L} = -\\frac{1}{|E\_{train}|} \\sum\_{(u,v) \\in E\_{train}} \\left\[ y\_{uv} \\log \\hat{y}_{uv} + (1-y_{uv}) \\log(1-\\hat{y}\_{uv}) \\right\]$$

### 7.2 Optimization

- Use AdamW optimizer with learning rate $\\eta$
- Apply gradient clipping with max norm $\\gamma$
- Use dropout rate $p$ for regularization

## 8. Complexity Analysis

### 8.1 Time Complexity

- Linearization generation: $O(K \\cdot n \\cdot \\bar{d})$ where $\\bar{d}$ is average degree
- Transformer processing: $O(K \\cdot n^2 \\cdot d)$ for each linearization
- Total: $O(K \\cdot n^2 \\cdot d)$

### 8.2 Space Complexity

- Storing linearizations: $O(K \\cdot n)$
- Model parameters: $O(L \\cdot d^2 + H \\cdot d_s)$ (including edge projection)
- Node representations: $O(K \\cdot n \\cdot d)$
- Edge features: $O(|E| \\cdot d_s)$

## 9. Advantages of Multiple Linearizations

### 9.1 Complete Coverage

Each linearization covers all nodes, ensuring no information loss.

### 9.2 Diverse Perspectives

Different linearizations capture different structural patterns:

- BFS: Breadth-first neighborhoods
- DFS: Depth-first paths
- Degree-based: Hub-centric views
- Random: Unbiased sampling

### 9.3 Robust Representations

Aggregating across multiple linearizations:

- Reduces dependence on any single traversal order
- Captures both local and global graph structure
- Provides multiple contexts for each node

### 9.4 Sufficient Data

With $K$ linearizations of length $n$:

- Total sequences: $K$
- Total tokens: $K \\times n$
- Each node appears $K$ times in different contexts

## 10. Implementation Considerations

### 10.1 Linearization Diversity

Ensure linearizations are sufficiently different:
$$\\text{diversity}(\\pi_i, \\pi_j) = \\frac{1}{n} \\sum\_{v \\in V} |\\pi_i(v) - \\pi_j(v)|$$

### 10.2 Batch Processing

Process multiple linearizations in parallel:

- Stack sequences: $(K, n, d)$
- Use batch dimension for efficient computation

### 10.3 Memory Management

For large graphs:

- Generate linearizations on-the-fly
- Use gradient checkpointing in transformer
- Process in mini-batches of linearizations

## 11. Implementation Details

### 11.1 Data Processing Pipeline (`data_utils.py`)

#### Graph Construction

```python
# Build adjacency matrix from training data only
adj_matrix = build_graph_from_train_data(train_df, all_genes)
```

- Uses absolute sensitivity scores as edge weights: $w\_{ij} = |sens_score\_{ij}|$
- Ensures undirected graph: $w\_{ij} = w\_{ji}$
- Prevents data leakage by using only training edges

#### Linearization Generation

```python
linearizer = GraphLinearizer(adj_matrix)
linearizations = linearizer.generate_multiple_linearizations(
    num_linearizations=100,
    min_diversity_threshold=0.3
)
```

**Diversity Metric**:
$$diversity(\\pi_1, \\pi_2) = 0.6 \\cdot \\frac{|{i: \\pi_1(i) \\neq \\pi_2(i)}|}{n} + 0.4 \\cdot \\frac{1}{n}\\sum\_{v} \\frac{|\\pi_1(v) - \\pi_2(v)|}{n-1}$$

#### Support View Preparation

```python
support_views, edge_map = prepare_support_views(
    adj_matrix=adj_matrix,
    n_support_views=3,
    support_view_dim=64
)
```

- Creates edge mapping: $(i,j) \\mapsto edge_index$ for canonical ordering $(i < j)$
- Initializes random support views if no pre-computed views provided
- Shape: $(n_edges, n_support_views, view_dim)$

### 11.2 Model Architecture (`graph_transformer.py`)

#### Weighted Support View Attention

```python
class WeightedSupportViewAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_support_views, feature_dim_per_view):
        # Learnable weights initialized near 0 for softplus ≈ 1
        self.support_view_weights = nn.Parameter(torch.randn(n_support_views) * 0.1)
        self.support_view_proj = nn.Linear(feature_dim_per_view, n_heads)
```

**Weight Computation**:
$$\\alpha_m = \\text{softplus}(w_m) + \\epsilon \\quad \\text{where } \\epsilon = 10^{-6}$$

**Weighted Feature Combination**:
$$\\tilde{s}_{ij} = \\sum_{m=1}^M \\alpha_m \\cdot s\_{ij}^{(m)}$$

**Edge Bias Injection**:

```python
bias_matrix = torch.zeros(n_heads, seq_len, seq_len, device=scores.device)
bias_matrix[:, edge_i, edge_j] = edge_biases.t()
if undirected:
    bias_matrix[:, edge_j, edge_i] = edge_biases.t()
```

#### Main GraphTransformer Model

```python
class GraphTransformer(nn.Module):
    def forward(self, node_features, linearization, support_views, 
                edge_map, query_edges):
        # 1. Create linearized sequence
        seq_features = node_features[linearization]
        
        # 2. Add positional encoding
        seq_features = seq_features + self.pos_encoding[:len(linearization)]
        
        # 3. Apply transformer layers with edge-aware attention
        for layer in self.transformer_layers:
            seq_features = layer(seq_features, support_views, edge_indices)
        
        # 4. Extract query edge predictions
        predictions = self.edge_predictor(query_edge_features)
```

### 11.3 Training Process (`train.py`)

#### Stochastic Linearization Sampling

```python
def train_epoch(model, linearizations, ...):
    for batch in data_loader:
        # Randomly sample linearization for this batch
        linearization = random.choice(linearizations)
        
        # Forward pass with this specific ordering
        predictions = model(node_features, linearization, ...)
        
        # Standard loss computation
        loss = criterion(predictions, labels)
```

#### Loss Functions (`loss_functions.py`)

The implementation supports multiple loss functions for imbalanced data:

**Focal Loss**:
$$L\_{focal} = -\\alpha_t (1-p_t)^\\gamma \\log(p_t)$$

**Weighted BCE**:
$$L\_{BCE} = -w\_+ y \\log(p) - w\_- (1-y) \\log(1-p)$$

**Combined Loss**:
$$L\_{combined} = \\lambda_1 L\_{focal} + \\lambda_2 L\_{BCE}$$

#### Multi-Linearization Evaluation

```python
def evaluate(model, linearizations, ...):
    predictions = []
    for linearization in sample(linearizations, n_samples=10):
        pred = model(node_features, linearization, ...)
        predictions.append(torch.sigmoid(pred))
    
    # Average predictions across linearizations
    avg_predictions = torch.stack(predictions).mean(dim=0)
```

### 11.4 Key Implementation Features

#### Memory Optimization

- Uses gradient clipping with `max_norm=1.0` for stability
- Processes edges in batches to manage memory
- Supports both CPU and GPU computation

#### Reproducibility

```python
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

#### Data Splitting Strategy

- Stratified split based on sensitivity score ranks
- Ensures SL pairs are distributed across train/val/test
- Prevents data leakage by building graph only from training data

### 11.5 Hyperparameter Configuration

**Model Architecture**:

- `node_feature_dim`: 1280 (ESM embedding dimension)
- `hidden_dim`: 128 (reduced for efficiency)
- `n_heads`: 4 (reduced for efficiency)
- `n_layers`: 2 (reduced for efficiency)
- `n_support_views`: 3 (GO-BP, GO-CC, PPI)
- `support_view_dim`: 64

**Training Configuration**:

- `num_linearizations`: 50 (reduced for efficiency)
- `batch_size`: 512
- `lr`: 1e-3 with cosine annealing to 1e-6
- `weight_decay`: 1e-5
- `patience`: 10 (early stopping)

**Data Configuration**:

- Gene universe: 6,298 genes with valid protein sequences
- Positive edges: ~19,464 SL pairs from `SL_Human_Approved.txt`
- `neg_to_pos_ratio`: 1 (balanced sampling)
- `test_ratio`: 0.15
- `val_ratio`: 0.1

## 12. Computational Complexity

### 12.1 Time Complexity Analysis

**Linearization Generation**: $O(K \\cdot n \\cdot \\log n)$ for $K$ linearizations
**Per Forward Pass**: $O(L \\cdot n^2 \\cdot d + |E| \\cdot d_s)$

- $L$ transformer layers
- $n^2$ attention computation
- $|E|$ edge bias computations

**Training Epoch**: $O(\\frac{|E\_{train}|}{B} \\cdot L \\cdot n^2 \\cdot d)$ for batch size $B$

### 12.2 Memory Requirements

**Model Parameters**: $O(L \\cdot d^2 + M \\cdot d_s \\cdot H)$
**Activations**: $O(B \\cdot n \\cdot d + B \\cdot H \\cdot n^2)$
**Support Views**: $O(|E| \\cdot M \\cdot d_s)$

### 12.3 Scalability Considerations

The implementation handles large graphs through:

- Batch processing of edges rather than full graph attention
- On-demand linearization generation
- Support for both learnable and pre-computed node features
- Efficient edge mapping using canonical ordering

## 13. Theoretical Justification

### 13.1 Why Multiple Linearizations Work

**Coverage**: Each linearization provides a complete traversal, ensuring all nodes and edges are considered.

**Diversity**: Different traversal strategies (BFS, DFS, degree-based, random) capture complementary structural patterns.

**Regularization**: Stochastic linearization sampling acts as data augmentation, improving generalization.

### 13.2 Weighted Support Views

**Adaptivity**: Learnable weights $\\alpha_m^2$ allow the model to emphasize informative support views.

**Non-negativity**: Using softplus ensures positive weights, maintaining interpretation as importance scores.

**Gradient Flow**: Softplus provides better gradients than ReLU near zero, enabling effective learning of small weights.

### 13.3 Edge-Aware Attention

**Structural Bias**: Edge features provide inductive bias about graph structure within sequences.

**Local Context**: Edge biases help attention focus on graph-relevant connections rather than just positional proximity.

**Undirected Handling**: Symmetric bias application ensures consistent treatment of undirected edges.
