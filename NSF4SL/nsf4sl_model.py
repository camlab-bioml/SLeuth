#!/usr/bin/env python3
"""
PyTorch implementation of NSF4SL (negative-sample-free contrastive learning
for ranking synthetic-lethal gene pairs; Wang et al., Bioinformatics 2022).

Faithful port of the benchmark reference (SL_benchmark/src/models/nsf4sl.py):
  - BYOL-style dual encoder: an online encoder + an EMA target encoder, with a
    predictor head on the online branch (no negative samples).
  - Feature-masking augmentation (masked dims -> dataset column mean).
  - Symmetric bootstrap loss between the two genes of an SL pair.

The only adaptation vs. the original is the feature source: instead of TransE
knowledge-graph embeddings loaded from a fixed .npy, per-gene features come from
any repo ``all_genes_*.pt`` file (default: kg_complex, the closest analog).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


def xavier_init(module):
    """Xavier-normal init for Linear/Embedding; zero bias (matches reference)."""
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class MLP(nn.Module):
    """Encoder: Linear -> BN -> LeakyReLU -> Linear -> BN -> LeakyReLU -> Linear."""

    def __init__(self,
                 input_size,
                 projection_size,
                 hid_size1=512,
                 hid_size2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hid_size1),
            nn.BatchNorm1d(hid_size1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size1, hid_size2),
            nn.BatchNorm1d(hid_size2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size2, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    """BYOL-style contrastive module (online + momentum target + predictor)."""

    def __init__(self,
                 input_size,
                 latent_size,
                 momentum,
                 hid_size1=512,
                 hid_size2=256):
        super(Net, self).__init__()
        self.latent_size = latent_size
        self.momentum = momentum
        self.input_size = input_size

        self.online_encoder = MLP(input_size, latent_size, hid_size1,
                                  hid_size2)
        self.target_encoder = MLP(input_size, latent_size, hid_size1,
                                  hid_size2)
        self.predictor = nn.Linear(latent_size, latent_size)

        self.online_encoder.apply(xavier_init)
        self.predictor.apply(xavier_init)
        self._init_target()

    def _init_target(self):
        """Copy online weights into the (frozen) target encoder."""
        for p_o, p_t in zip(self.online_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def _update_target(self):
        """EMA update of the target encoder from the online encoder."""
        for p_o, p_t in zip(self.online_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data = p_t.data * self.momentum + p_o.data * (1. -
                                                              self.momentum)

    def forward(self, inputs):
        g1, g2, g1_aug, g2_aug = inputs[0], inputs[1], inputs[2], inputs[3]
        g1_online = self.predictor(self.online_encoder(g1_aug))
        g1_target = self.target_encoder(g1)
        g2_online = self.predictor(self.online_encoder(g2_aug))
        g2_target = self.target_encoder(g2)
        return g1_online, g1_target, g2_online, g2_target

    @torch.no_grad()
    def get_embedding(self, inputs):
        """Return (predicted online, online) embeddings for scoring."""
        g_online = self.online_encoder(inputs.float())
        return self.predictor(g_online), g_online

    def get_loss(self, output):
        """Symmetric bootstrap loss (BYOL): 2 - 2*cos, both directions."""
        u_online, u_target, i_online, i_target = output
        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)
        return (loss_ui + loss_iu).mean()


@torch.no_grad()
def score_matrix(model, all_gene_feature):
    """Full gene x gene SL score matrix from the trained encoders.

    score = P(g)·O(g')^T + O(g)·P(g')^T  (symmetric), diagonal zeroed, where
    P is the predicted-online embedding and O the online embedding. Mirrors the
    reference ``cal_score_mat``.
    """
    model.eval()
    predicted, online = model.get_embedding(all_gene_feature)
    mat1 = torch.matmul(predicted, online.t())
    mat2 = torch.matmul(online, predicted.t())
    mat = (mat1 + mat2).cpu().numpy()
    n = mat.shape[0]
    mat[np.arange(n), np.arange(n)] = 0
    return mat
