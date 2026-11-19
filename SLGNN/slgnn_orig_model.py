#!/usr/bin/env python3
"""
The ORIGINAL SLGNN (Zhu et al., Bioinformatics 2023), in pure PyTorch.

A line-by-line port of src/model.py of github.com/zy972014452/SLGNN at commit
c56e23fbd20e (the one build_synlethkg.py pins), minus its two compiled
dependencies, each replaced by the SAME computation:

  * torch_scatter.scatter_mean (2.0.9)  -> scatter_mean below: index_add_ sum,
    divided by the per-row count clamped to >= 1 (torch_scatter clamps the
    same way, so rows without edges come out 0);
  * dgl.nn.GATConv (0.7.2)              -> GATConv below: the same parameters,
    the same initialisation (xavier_normal_ with the ReLU gain, bias 0), the
    same forward (feat_drop -> fc -> el/er -> leaky_relu(el_src + er_dst) ->
    edge softmax over each node's incoming edges, max-shifted -> attn_drop ->
    sum of a * ft_src at dst -> residual -> bias -> activation).

Only the floating-point summation order can differ, and on a GPU it is not
fixed in torch_scatter / DGL either (atomic adds).

The model: every gene has a trainable gene-side embedding and, through its
SynLethKG entity, a trainable entity-side embedding (all_embed). Per hop,
  entity_agg_e = mean_{(e, r, t)} entity_t * w_r            (KG aggregation)
  gene_i       = sum_{j in N_SL(i) + i} entity_j
                 + sum_f softmax_f(gene_i . l_f) * GAT_f(entity_agg)_i * l_f
with factor prototypes l_f = softmax(disen_att_f) @ w (one 2-layer, 8-head GAT
per factor over the training SL graph), then message dropout, L2
normalisation, and a residual sum over hops. A pair (a, b) scores
gene_a . entity_b: gene side for a, KG-entity side for b.

Kept exactly as the original, including its quirks:
  * forward() returns sigmoid(score) and train_slgnn.py feeds that to
    BCEWithLogitsLoss, i.e. the loss is BCE(sigmoid(sigmoid(score)), y);
  * the "latent_emb" parameter is never used (Aggregator recomputes it);
  * the second independence term uses for_reg, which the hop loop overwrites
    and then adds to itself, so with >1 hop it is 2x the LAST hop's mean
    factor output;
  * node dropout keeps int(E * rate) KG edges and keeps each SL-adjacency
    entry with probability `rate` (scaled by 1/(1 - rate)); both are 0.5 by
    default, where the two readings agree;
  * all_embed, latent_emb, the relation weights and disen_weight_att use
    xavier_uniform_, created in the original's order.

Deliberate differences: evaluation goes through score_matrix(), an added
method that runs eval() and generate() (no dropout of any kind; the original
scored validation/test in train mode, i.e. with node, message and attention
dropout still active). Inside GraphConv, node dropout draws its randomness on
the model's device: _edge_sampling takes the KG-edge subset with
torch.randperm instead of np.random.choice, and _sparse_dropout draws its
mask with torch.rand on the device instead of on the CPU and then moving it.
The distributions are the same (a uniform subset without replacement; each
entry kept with probability `rate`), but the random streams are not, so a
seeded run does not replay the original's dropout masks. Training pairs come
in both orientations, as the original's fold files list them, so the SL graph
is the symmetric training adjacency (plus self-loops); the original's graph
comes from its row-split fold files, where about 20% of the training pairs
have their reverse in val/test instead.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src, index, dim_size):
    """torch_scatter.scatter_mean(src, index, dim=0, dim_size=dim_size)."""
    out = src.new_zeros((dim_size, ) + src.shape[1:]).index_add_(0, index, src)
    count = torch.bincount(index, minlength=dim_size).clamp_(min=1)
    return out / count.to(src.dtype).view(-1, *([1] * (src.dim() - 1)))


def edge_softmax(e, dst, num_nodes):
    """dgl.ops.edge_softmax(graph, e) (norm_by='dst'): softmax of e over the
    edges entering each destination node. e: (E, H, 1)."""
    idx = dst.view(-1, *([1] * (e.dim() - 1))).expand_as(e)
    emax = e.new_full((num_nodes, ) + e.shape[1:], float("-inf"))
    # The max shift is a constant, as in DGL's forward; the softmax Jacobian
    # autograd then computes is DGL's custom backward exactly.
    emax = emax.scatter_reduce(0, idx, e.detach(), reduce="amax")
    ex = torch.exp(e - emax[dst])
    den = torch.zeros_like(emax).index_add_(0, dst, ex)
    return ex / den[dst]


class GATConv(nn.Module):
    """dgl.nn.pytorch.GATConv of DGL 0.7.2, homogeneous graph, on an edge list.

    forward(src, dst, num_nodes, feat) with messages src -> dst. DGL
    (allow_zero_in_degree=False) refuses a node without incoming edges;
    SLGNNOriginal checks that once, at construction (self-loops guarantee it).
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 bias=True):
        super().__init__()
        self._num_heads = num_heads
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = (nn.Parameter(torch.empty(num_heads * out_feats))
                     if bias else None)
        if residual:
            self.res_fc = (nn.Linear(in_feats, num_heads * out_feats,
                                     bias=False)
                           if in_feats != out_feats * num_heads else
                           nn.Identity())
        else:
            self.res_fc = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, src, dst, num_nodes, feat):
        h = self.feat_drop(feat)
        ft = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (N, H, 1)
        er = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1)
        e = self.leaky_relu(el[src] + er[dst])  # (E, H, 1)
        a = self.attn_drop(edge_softmax(e, dst, num_nodes))
        rst = ft.new_zeros(num_nodes, self._num_heads, self._out_feats)
        rst = rst.index_add_(0, dst, ft[src] * a)
        if self.res_fc is not None:
            rst = rst + self.res_fc(h).view(h.shape[0], -1, self._out_feats)
        if self.bias is not None:
            rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
        if self.activation:
            rst = self.activation(rst)
        return rst


class GAT(nn.Module):
    """model.py GAT: num_layers hidden GATConv layers (heads flattened) and an
    output GATConv whose heads are averaged."""

    def __init__(self, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop,
                    negative_slope, False, activation))
        for l in range(1, num_layers):
            self.gat_layers.append(
                GATConv(num_hidden * heads[l - 1], num_hidden, heads[l],
                        feat_drop, attn_drop, negative_slope, residual,
                        activation))
        self.gat_layers.append(
            GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop,
                    attn_drop, negative_slope, residual, None))

    def forward(self, graph, h):
        src, dst, n = graph
        for l in range(self.num_layers):
            h = self.gat_layers[l](src, dst, n, h).flatten(1)
        return self.gat_layers[-1](src, dst, n, h).mean(1)


def distance_correlation(tensor_1, tensor_2):
    """utils.DistanceCorrelation, verbatim."""
    channel = tensor_1.shape[0]
    zeros = torch.zeros(channel, channel).to(tensor_1.device)
    zero = torch.zeros(1).to(tensor_1.device)
    tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
    a_, b_ = (torch.matmul(tensor_1, tensor_1.t()) * 2,
              torch.matmul(tensor_2, tensor_2.t()) * 2)
    tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
    a = torch.sqrt(
        torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8)
    b = torch.sqrt(
        torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)
    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
    dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel**2, zero) + 1e-8)
    dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel**2, zero) + 1e-8)
    dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel**2, zero) + 1e-8)
    return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)


class Aggregator(nn.Module):
    """model.py Aggregator (one hop; parameter-free)."""

    def __init__(self, n_genes, n_factors):
        super().__init__()
        self.n_genes = n_genes
        self.n_factors = n_factors

    def forward(self, gats, sl_graph, entity_emb, gene_sl_emb, edge_index,
                edge_type, interact_mat, weight, disen_weight_att,
                gene_entity):
        n_entities = entity_emb.shape[0]
        # KG aggregate
        head, tail = edge_index
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        entity_agg = scatter_mean(neigh_relation_emb, head, n_entities)

        latent_emb = torch.mm(F.softmax(disen_weight_att, dim=-1), weight)
        score = F.softmax(torch.mm(gene_sl_emb, latent_emb.t()),
                          dim=1).unsqueeze(-1)  # (n_genes, n_factors, 1)

        gene_agg = torch.sparse.mm(interact_mat, entity_emb[gene_entity])
        input_emb = entity_agg[gene_entity]
        output_sl = torch.stack([
            gats[i](sl_graph, input_emb) * latent_emb[i]
            for i in range(self.n_factors)
        ], dim=1)  # (n_genes, n_factors, channel)
        gene_return = gene_agg + (output_sl * score).sum(dim=1)
        return entity_agg, gene_return, output_sl


class GraphConv(nn.Module):
    """model.py GraphConv: n_hops Aggregators sharing one GAT per factor."""

    def __init__(self, channel, n_hops, n_genes, n_factors, n_relations, ind,
                 node_dropout_rate, mess_dropout_rate):
        super().__init__()
        self.n_relations = n_relations
        self.n_genes = n_genes
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.GATConv = nn.ModuleList([
            GAT(1, channel, channel, channel, [8, 8], F.leaky_relu, 0.2, 0.2,
                0.2, True) for _ in range(n_factors)
        ])
        self.temperature = 0.2
        initializer = nn.init.xavier_uniform_
        self.weight = nn.Parameter(initializer(torch.empty(n_relations,
                                                           channel)))
        self.disen_weight_att = nn.Parameter(
            initializer(torch.empty(n_factors, n_relations)))
        self.convs = nn.ModuleList(
            [Aggregator(n_genes, n_factors) for _ in range(n_hops)])
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        keep = torch.randperm(n_edges,
                              device=edge_index.device)[:int(n_edges * rate)]
        return edge_index[:, keep], edge_type[keep]

    def _sparse_dropout(self, x, rate=0.5):
        noise = rate + torch.rand(x._nnz(), device=x.device)
        mask = torch.floor(noise).bool()
        i = x._indices()[:, mask]
        v = x._values()[mask]
        return torch.sparse_coo_tensor(i, v, x.shape) * (1. / (1 - rate))

    def _cul_cor(self):
        if self.ind == "mi":
            disen_T = self.disen_weight_att.t()
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T,
                                   dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att),
                                   dim=1)
            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)
            return -torch.sum(torch.log(pos_scores / ttl_scores))
        cor = 0
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                t1, t2 = self.disen_weight_att[i], self.disen_weight_att[j]
                if self.ind == "distance":
                    cor += distance_correlation(t1, t2)
                else:  # cosine
                    cor += ((t1 / t1.norm(dim=0, keepdim=True)) *
                            (t2 / t2.norm(dim=0, keepdim=True))).sum(dim=0)**2
        return cor

    def forward(self,
                gene_emb,
                entity_emb,
                edge_index,
                edge_type,
                interact_mat,
                gene_entity,
                sl_graph,
                mess_dropout=True,
                node_dropout=False):
        """sl_graph = (src, dst, n_genes) of the full training SL graph; the
        GATs always see it undropped (the original builds its DGL graph once,
        from the undropped matrix)."""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(
                edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat,
                                                self.node_dropout_rate)

        entity_res_emb = entity_emb
        gene_res_emb = gene_emb
        cor = self._cul_cor()
        for_reg = None
        for i in range(len(self.convs)):
            entity_emb, gene_emb, output_gene = self.convs[i](
                self.GATConv, sl_graph, entity_emb, gene_emb, edge_index,
                edge_type, interact_mat, self.weight, self.disen_weight_att,
                gene_entity)
            for_reg = output_gene.mean(dim=0)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                gene_emb = self.dropout(gene_emb)
            entity_emb = F.normalize(entity_emb)
            gene_emb = F.normalize(gene_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            gene_res_emb = torch.add(gene_res_emb, gene_emb)
            # As in the original: for_reg was just overwritten, so from the
            # second hop on this is twice the current hop's mean.
            if i == 0:
                for_reg = output_gene.mean(dim=0)
            else:
                for_reg = torch.add(for_reg, output_gene.mean(dim=0))

        cor_2 = 0
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                cor_2 += distance_correlation(for_reg[i], for_reg[j])
        return entity_res_emb, gene_res_emb, cor, cor_2


class SLGNNOriginal(nn.Module):
    """model.py SLModel on the shared gene universe.

    n_genes     genes of the shared universe (rows of the score matrix)
    n_entities  SynLethKG entities + one isolated entity per gene the KG
                lacks (build_synlethkg.load_kg_for_genes)
    gene_entity (n_genes,) LongTensor, the original's reindex_dict
    edge_index  (2, E) KG edges (head, tail), inverse edges included
    edge_type   (E,) relation ids in [0, n_relations)
    sl_adj      scipy (n_genes, n_genes) training SL adjacency, symmetric,
                no diagonal; self-loops are added here, as SLDataset does
    """

    def __init__(self,
                 n_genes,
                 n_entities,
                 n_relations,
                 gene_entity,
                 edge_index,
                 edge_type,
                 sl_adj,
                 dim=64,
                 context_hops=3,
                 n_factors=4,
                 ind="distance",
                 l2=1e-4,
                 sim_regularity=1e-3,
                 node_dropout=True,
                 node_dropout_rate=0.5,
                 mess_dropout=True,
                 mess_dropout_rate=0.1,
                 device=torch.device("cpu")):
        super().__init__()
        self.n_genes = n_genes
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.decay = l2
        self.sim_decay = sim_regularity
        self.emb_size = dim
        self.n_factors = n_factors
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        initializer = nn.init.xavier_uniform_
        self.all_embed = nn.Parameter(
            initializer(torch.empty(n_genes + n_entities, dim)))
        self.latent_emb = nn.Parameter(initializer(torch.empty(n_factors,
                                                               dim)))

        # Data, not state: kept out of state_dict so checkpoint copies stay
        # small (the KG edge list is ~70 MB).
        self.register_buffer("gene_entity",
                             torch.as_tensor(gene_entity,
                                             dtype=torch.long).to(device),
                             persistent=False)
        self.register_buffer("edge_index",
                             torch.as_tensor(edge_index,
                                             dtype=torch.long).to(device),
                             persistent=False)
        self.register_buffer("edge_type",
                             torch.as_tensor(edge_type,
                                             dtype=torch.long).to(device),
                             persistent=False)

        adj = (sl_adj.tocsr().astype(np.float32) +
               sp.eye(n_genes, format="csr", dtype=np.float32)).tocoo()
        idx = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
        self.register_buffer("sl_index", idx.to(device), persistent=False)
        self.register_buffer("sl_value",
                             torch.from_numpy(adj.data.astype(
                                 np.float32)).to(device),
                             persistent=False)
        indeg = np.bincount(adj.col, minlength=n_genes)
        if (indeg == 0).any():  # DGL refuses 0-in-degree nodes
            raise ValueError("SL graph has a node without incoming edges")

        self.gcn = GraphConv(channel=dim,
                             n_hops=context_hops,
                             n_genes=n_genes,
                             n_factors=n_factors,
                             n_relations=n_relations,
                             ind=ind,
                             node_dropout_rate=node_dropout_rate,
                             mess_dropout_rate=mess_dropout_rate)

    def _interact_mat(self):
        return torch.sparse_coo_tensor(self.sl_index, self.sl_value,
                                       (self.n_genes, self.n_genes))

    def _run_gcn(self, mess_dropout, node_dropout):
        gene_emb = self.all_embed[:self.n_genes, :]
        item_emb = self.all_embed[self.n_genes:, :]
        return self.gcn(gene_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self._interact_mat(),
                        self.gene_entity,
                        (self.sl_index[0], self.sl_index[1], self.n_genes),
                        mess_dropout=mess_dropout,
                        node_dropout=node_dropout)

    def forward(self, gene_a, gene_b):
        entity_gcn_emb, gene_gcn_emb, cor, cor_2 = self._run_gcn(
            self.mess_dropout, self.node_dropout)
        e_u = gene_gcn_emb[gene_a]
        e_e = entity_gcn_emb[self.gene_entity[gene_b]]
        scores = (e_u * e_e).sum(dim=1)
        regularizer = (torch.norm(e_u)**2 + torch.norm(e_e)**2) / 2
        emb_loss = self.decay * regularizer / len(gene_a)
        cor_loss = self.sim_decay * (cor + cor_2)
        return torch.sigmoid(scores), emb_loss, cor_loss, cor

    @torch.no_grad()
    def generate(self):
        """(entity_emb, gene_emb, cor) without node or message dropout."""
        return self._run_gcn(mess_dropout=False, node_dropout=False)[:-1]

    @torch.no_grad()
    def score_matrix(self):
        """(n_genes, n_genes) raw pair scores, symmetric, diagonal 0.

        S[a, b] = gene_a . entity_b is not symmetric. The original scores a
        pair in the orientation its fold file lists it; the shared folds list
        every pair as a < b, so the upper triangle holds exactly those scores,
        and it is mirrored (as SL_benchmark's final_test does).
        """
        self.eval()
        entity_emb, gene_emb, _ = self.generate()
        S = gene_emb @ entity_emb[self.gene_entity].t()
        S = torch.triu(S, diagonal=1)
        return (S + S.t()).cpu().numpy()
