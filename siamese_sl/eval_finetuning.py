#!/usr/bin/env python3
"""
Evaluate a pretrained multi-modal siamese SL model against external CRISPR /
Perturb-seq screens (Adamson, Corn, Gilbert), with optional fine-tuning.

Every row of a dataset is evaluated on the SAME held-out test split, chosen
by --cv_type (split_pairs / split_pairs_cv2 / split_pairs_cv3):

  cv1   random pair split: train/val/test = 72/8/20% of the pairs at the
        default --train_frac 0.8 --val_frac 0.1.
  cv2   gene-held-out: --train_frac is a fraction of the screen's GENES; test
        = every pair with >= 1 held-out gene (~36% of pairs at 0.8).
  cv3   both genes held out: test = pairs with both genes held out (~4%);
        mixed pairs (~32%) are discarded.

For cv2/cv3 the VAL split is gene-held-out as well: --val_frac is a slice of
the non-test genes (not a random slice of the train+val pairs), so the
fine-tuning checkpoint is selected under the same unseen-gene regime it is
scored in. sl_comparison/prepare_folds.py carves the shared SLDB cv2/cv3 val
the same way, and its verify_fold asserts it. cv3 therefore discards the extra
pairs with exactly one val gene (a screen-wide ~44% discarded at 0.8/0.1
instead of ~32%).

The exact sizes depend on the screen; they are printed and stored as n_test.

  Zero-shot OLS (calibrated)  closed-form OLS of y = a*logit + c on the TRAIN
                              split only (val is kept for the SGD modes'
                              early stopping).
  Zero-shot raw               the uncalibrated logit itself (R² is NaN: the
                              logit is not in the target's units).
  LP_last                     SGD MSE on (a, c) + the final projection layer
                              (+ a multi-cell checkpoint's per-head
                              projection bias head_proj_bias, when present).
  Full                        SGD MSE on (a, c) + the entire encoder (+ the
                              multi-cell head_proj_bias / head_input_bias /
                              kernel_A / gene_w / pair_fc1 + pair_fc2, when
                              present). See set_trainable.
  Scratch                     (opt-in, --ft_modes ... Scratch) the from-
                              scratch baseline: Full's trainable set,
                              optimiser, loss, epochs and early stopping on
                              the same architecture, fold feature space and
                              screen split, but every weight and buffer at
                              the constructor's fresh init (torch seed
                              split_seed + SCRATCH_INIT_SEED_OFFSET + fold),
                              none from SynLethDB. Parts Full keeps frozen
                              (temperature, scoring bias, cell embedding,
                              inner-norm statistics) stay frozen at their
                              init values. Its (a, c) is warm-started from the
                              OLS fit on the FRESH model's own train-split
                              logits, never on the pretrained logit.

The (a, c) affine head is ALWAYS present and trainable in every mode — R²
only becomes meaningful once the arbitrary pretrained-logit scale is
calibrated to the external-screen target's units. Rank metrics (Spearman,
Pearson, AUROC@1%, AUPR@1%) are invariant to the affine transform only up to
sign(a): OLS does not constrain a, and a fold with a < 0 reports the raw
logit's Spearman / Pearson negated and AUROC as 1 - AUROC (AUPR changes too).
So the 'Zero-shot OLS' rank metrics are the raw logit's with a direction
chosen on the screen's train split; the 'Zero-shot raw' rows (and a_ols in
the zero-shot CSVs) show the uncalibrated direction. The (a, c) head alone
(mode LP) cannot move them beyond that sign. LP_last / Full are warm-started
from the OLS (a, c) and inherit sign(a), so compare them with 'Zero-shot OLS':
their rank metrics are the diagnostic for whether the last-layer / full
unfreezes learned new structure beyond pure calibration. Scratch shares no
start with them (its own OLS sign); Full minus Scratch is what the SynLethDB
pretraining adds under the same fine-tuning budget.

Target sign convention
----------------------
Pretrained logit grows with SL-likeness. All three external scores (Corn
sens.score, Adamson Gamma, Gilbert K562_avg) grow MORE NEGATIVE with
SL-likeness. We therefore fine-tune against `target = sign * raw_score`
with sign = -1 (configured per dataset) so the pretrained direction is
preserved and (a, c) starts near identity.

Per-fold vs ensemble
--------------------
Each row reports two aggregations of its metrics:
  - Per-fold: compute the metric on each of 5 folds separately, then
    mean ± std across folds.
  - Ensemble: average the 5 folds' logits first (for each mode, AFTER that
    mode's (a, c) head has been fit per fold; the raw logits for
    'Zero-shot raw'), then compute the metric once on the ensembled
    prediction.

Outputs (under {MODEL_DIR}/eval_finetuning_{cv_type}/, e.g. eval_finetuning_cv3/)
----------------------------------------------------------------------------
  config_used.json               resolved configuration snapshot, incl.
                                 fold_ids (always 0..K-1) and each
                                 scored checkpoint's sha256
  gene_coverage.csv              per-dataset coverage + SynLethDB-overlap counts
  zero_shot_metrics.csv          long: dataset x calibration (ols/raw) x fold x metric
  zero_shot_summary.csv          wide: dataset x calibration x metric (per-fold
                                 mean±std, ensemble; + the OLS slope a_ols)
  finetuned_metrics.csv          long: dataset x mode x fold x metric
  finetuned_summary.csv          wide: dataset x mode x metric (per-fold, ensemble)
  {dataset}_predictions.csv      per-pair zero-shot logits from all 5 folds
  {dataset}_finetuned_predictions_{mode}.csv  per-pair fine-tuned preds, test split only
  {dataset}_sl_probabilities.csv per-pair zero-shot SL PROBABILITY, sigmoid of
                                 the logit, for EVERY screen pair and EVERY
                                 model head: prob_<HEAD>_fold<k> and
                                 prob_<HEAD>_mean (mean over folds), 12
                                 significant digits. A cell-line model has one
                                 head per cell line (cell_line_vocab.HEADS)
                                 plus prob_ANY_* (the largest of its heads,
                                 the score the benchmark ranks by; no head is
                                 trained on the any-line label); a pooled or
                                 single-output model has prob_ANY_* only (its
                                 one head, trained on that label).
  {dataset}_finetuned_allpairs_{mode}.csv  the fine-tuned model's predicted
                                 screen value for EVERY screen pair, with a
                                 `split` column (train / val / test /
                                 discarded): pred_fold<k>, pred_mean on the
                                 TARGET scale (sign * raw score) and
                                 pred_outcome_fold<k>, pred_outcome_mean in
                                 the screen's own sign and units (e.g. the
                                 predicted Adamson Gamma). Train and val rows
                                 are in-sample; only test rows are held out.
  finetuned_models/{dataset}_{mode}_fold{k}.pt  the fine-tuned weights with
                                 their affine head (a, c) and the head they
                                 score through; export_pair_predictions.py
                                 turns them into predicted screen values for
                                 all gene pairs (--no_save_finetuned skips
                                 them)
These three are optional extras: a failure in one is reported (exit 1) after
every table and metric above has been written, and never costs a screen its
{dataset}_predictions.csv.

The screen is predicted through the cell line it was run in: with --per_head
a cell-line model is scored, fine-tuned and saved through the head named in
the DATASETS row (K562 for Adamson), i.e. with that line's embedding
concatenated onto each gene's features. A pooled or single-output model has no
cell-line input; its one head is used.

Each *_metrics.csv carries a column `filter` taking values `unfiltered` (all
test-split pairs) and `sldb_filtered` (the same test pairs with
SynLethDB-overlap pairs removed) — provided REPORT_SLDB_FILTERED=true in the
config.

Exit status 1, with the previous outputs left untouched (nothing cleared, no
GPU time spent), when the model dir has no config.json / num_folds, its
checkpoint set is partial, a DATASETS file or loader is missing, a DATASETS row
names no model head under --per_head against a multi-head checkpoint, or
--embeddings_paths differ from the files the checkpoints record (basenames in
order, gene_order, and the files' sha256 — a same-path regeneration keeps the
first two). Also exit 1, after writing the rest, when a screen yields
no {dataset}_predictions.csv — fewer than 100 evaluable pairs, or a split that
leaves under 10 train or val pairs to fit and select on:
sl_comparison/external_transfer.py needs every screen.
"""

import argparse
import hashlib
import io
import json
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Project-local imports
from data_loader import (
    apply_multimodal_transform,
    fit_multimodal_transform,
    load_raw_multimodal,
)
from gene_name_utils import get_mapper
import cell_line_vocab
from predict import load_model  # handles checkpoint loading + arch inference
from siamese_esm import SiameseSL, SiameseSLMultiCell, set_seed

# =============================================================================
# FinetunedSL container
# =============================================================================


class FinetunedSL(nn.Module):
    """Wraps a pretrained SiameseSL with a trainable affine head y = a*logit + c.

    The base model is kept intact; (a, c) live on this wrapper so there is no
    risk of mutating the pretrained class across code paths.
    """

    def __init__(self, base: SiameseSL, a: float = 1.0, c: float = 0.0):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(float(c), dtype=torch.float32))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        logit = self.base(x1, x2).squeeze(-1)  # (batch,)
        return self.a * logit + self.c

    def raw_logit(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Pre-affine logit; used for zero-shot OLS fitting."""
        return self.base(x1, x2).squeeze(-1)


# =============================================================================
# Fine-tuning modes
# =============================================================================
# Scratch: the from-scratch baseline. Same architecture (rebuilt from the fold
# checkpoint's weight shapes), same fold feature space (that checkpoint's
# preprocessing_transform), same screen split, optimiser, loss, epochs, early
# stopping and the SAME trainable set as Full; but every parameter and buffer
# of the scorer is the constructor's fresh initialisation (seeded per fold,
# scratch_init_seed) instead of the SynLethDB weights. What Full keeps frozen
# (temperature, scoring bias, cell embedding, an --inner_norm batch model's
# running statistics) is frozen here too, at its INIT value: a fixed random
# cell vector, τ0 = 1/sqrt(latent), b_h = 0, running mean 0 / var 1. The
# affine head is warm-started from the OLS fit of the train-split target on
# the FRESH model's own logits (Full's rule applied to Scratch's own starting
# model), never on a pretrained logit.
SCRATCH_MODE = "Scratch"
FT_MODE_CHOICES = ("LP", "LP_last", "Full", SCRATCH_MODE)
# The trainable set / train-eval regime each mode shares with another.
_TRAINS_LIKE = {SCRATCH_MODE: "Full"}
# Scratch fold k is initialised under torch seed split_seed + this + k: a
# stream of its own (fine-tuning itself still seeds split_seed + k, as Full,
# so both see the same batch order).
SCRATCH_INIT_SEED_OFFSET = 7919


def scratch_init_seed(split_seed: int, fold: int) -> int:
    """The torch seed fold `fold`'s Scratch model is initialised under."""
    return int(split_seed) + SCRATCH_INIT_SEED_OFFSET + int(fold)


def _multicell(model: FinetunedSL) -> Optional[SiameseSLMultiCell]:
    """The SiameseSLMultiCell behind a _HeadSelect base, else None."""
    mc = getattr(model.base, "model", None)
    return mc if isinstance(mc, SiameseSLMultiCell) else None


def _unfreeze_attrs(module: Optional[nn.Module], names) -> None:
    """requires_grad_(True) on each named Parameter the module actually has."""
    if module is None:
        return
    for name in names:
        p = getattr(module, name, None)
        if isinstance(p, nn.Parameter):
            p.requires_grad_(True)


def set_trainable(model: FinetunedSL, mode: str) -> None:
    """Freeze everything, then unfreeze the parameters this mode trains.

    Modes:
      LP       — (a, c) only; zero-shot closed-form uses the same freeze.
      LP_last  — (a, c) + encoder.projection (final Linear layer) + the
                 per-head projection bias head_proj_bias when the checkpoint
                 has one (train.py's default per-head c_h, turned off by
                 --no_per_head_last_layer_bias, replaces the projection's
                 bias, so it IS the last layer's bias).
      Full     — (a, c) + entire encoder (input_bias, hidden layers,
                 projection) + the multi-cell parameters that stand in for
                 encoder pieces or sit between encoder and score, when present:
                 head_proj_bias (projection bias), head_input_bias (input bias,
                 --per_head_input_bias), kernel_A (--kernel bilinear), gene_w
                 (--gene_term) and pair_fc1 / pair_fc2 (--pair_features).
      Scratch  — exactly Full's set, on a freshly initialised model
                 (SCRATCH_MODE above).

    The base model's temperature (log_temperature / tau), scoring_bias and
    cell_emb stay frozen in every mode — SGD can absorb calibration through
    (a, c) directly without needing to perturb those pretrained scalars, and
    the learned cell representation is not adapted to one screen. Only the
    selected head's rows of head_proj_bias / head_input_bias receive a
    gradient (_HeadSelect exposes one head's logit).
    """
    mode = _TRAINS_LIKE.get(mode, mode)
    for p in model.parameters():
        p.requires_grad_(False)
    model.a.requires_grad_(True)
    model.c.requires_grad_(True)
    mc = _multicell(model)

    if mode == "LP":
        return
    if mode == "LP_last":
        for p in model.base.encoder.projection.parameters():
            p.requires_grad_(True)
        _unfreeze_attrs(mc, ("head_proj_bias", ))
        return
    if mode == "Full":
        for p in model.base.encoder.parameters():
            p.requires_grad_(True)
        _unfreeze_attrs(mc, ("head_proj_bias", "head_input_bias", "kernel_A",
                             "gene_w"))
        # --pair_features: the pair-term MLP sits between encoder and score
        for name in ("pair_fc1", "pair_fc2"):
            mod = getattr(mc, name, None) if mc is not None else None
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
        return
    raise ValueError(f"Unknown fine-tune mode: {mode}")


def set_frozen_to_eval(model: FinetunedSL, mode: str) -> None:
    """Put frozen submodules in eval() mode to disable their dropout.

    Must be called AFTER every `model.train()` call in the training loop.
    `model.train()` recursively sets `training=True` on ALL submodules
    (including the frozen ones). Without this helper, dropout would stay
    active on the frozen backbone, injecting noise into the forward pass
    that the trainable parameters don't expect — a departure from the
    standard linear-probing convention (frozen backbone is deterministic).

    Train/eval mode affects Dropout and, for a multi-cell checkpoint trained
    with --inner_norm batch, the per-head kernel standardisation
    (SiameseSLMultiCell.standardize_inner: batch statistics + a running-
    estimate update in train mode). LayerNorm has no running stats and behaves
    identically either way.

    The inner normalisation is a pretrained, frozen part of the scorer in
    EVERY mode, so for such a checkpoint this sets
    `model.base.model.freeze_inner_stats = True`: the running estimates are
    applied and never updated, even in Full where the multi-cell model is in
    train mode. Otherwise fine-tuning would standardise each screen batch by
    its own mean/std and overwrite the pretrained statistics.

    Modes:
      LP       — entire base → eval (backbone + log_temperature/scoring_bias
                 parents); only the (a, c) head stays "stochastic-free"
                 alongside the eval'd backbone.
      LP_last  — the encoder's hidden stack → eval (the only dropout-bearing
                 frozen region). base.encoder.projection stays in train
                 mode but has no dropout anyway. Covers both the standard
                 `SiameseEncoder` (`encoder.hidden` nn.Sequential) and the
                 `SiameseEncoderResidual` variant (`encoder.hidden_blocks`
                 nn.ModuleList) — whichever attribute exists on the loaded
                 checkpoint. Inner-norm frozen (above).
      Full     — dropout fires (entire encoder is trainable); inner-norm
                 frozen (above).
      Scratch  — as Full (its inner-norm statistics are frozen at their
                 init values, mean 0 / var 1).
    """
    mode = _TRAINS_LIKE.get(mode, mode)
    mc = _multicell(model)
    if mc is not None and getattr(mc, "inner_norm", "none") == "batch":
        mc.freeze_inner_stats = True
    if mode == "LP":
        model.base.eval()
    elif mode == "LP_last":
        enc = model.base.encoder
        # Handle both encoder variants. Each has exactly one of these.
        if hasattr(enc, "hidden_blocks"):
            enc.hidden_blocks.eval()  # residual encoder
        elif hasattr(enc, "hidden"):
            enc.hidden.eval()  # mlp / standard encoder
        # input_bias isn't a Module (just a Parameter), no need to eval()
    elif mode == "Full":
        pass
    else:
        raise ValueError(f"Unknown fine-tune mode: {mode}")


# =============================================================================
# Cell-line-conditioned (per-head) evaluation
# =============================================================================
# A SiameseSLMultiCell checkpoint emits one logit (PD or bilinear kernel) PER
# cell-line head (cell_line_vocab.HEADS; a --cell_lines pooled checkpoint has
# a single head, used for every screen). To evaluate it against an external
# screen we pick the
# head whose cell line matches that screen (e.g. K562 for Adamson / Gilbert) and
# expose ONLY that head's logit as a single-output model, so the existing
# FinetunedSL / finetune_one / compute_logits machinery is reused verbatim.


# What the log and config_used.json name as the scoring head when a pooled
# (--cell_lines pooled, num_heads 1) checkpoint scores a screen: its single
# head is trained on the OR label, whatever cell line DATASETS names.
POOLED_HEAD_LABEL = "ANY (pooled)"


def _multicell_heads(sd: dict, ckpt_path) -> Tuple[int, int]:
    """(num_heads, cell_line_dim) of a SiameseSLMultiCell state dict."""
    sb = sd.get("scoring_bias")
    if "cell_emb.weight" in sd:
        return (int(sd["cell_emb.weight"].shape[0]),
                int(sd["cell_emb.weight"].shape[1]))
    if sb is not None and sb.dim() == 1:
        # --cell_lines pooled: one head, no cell-line embedding. (The
        # single-output SiameseSL has a SCALAR scoring_bias, so it never
        # lands here.)
        return int(sb.shape[0]), 0
    raise ValueError(f"{ckpt_path} is not a SiameseSLMultiCell checkpoint "
                     f"(no cell_emb.weight, scalar scoring_bias). Run "
                     f"without --per_head.")


def checkpoint_num_heads(ckpt_path) -> int:
    """Output heads of a SiameseSLMultiCell checkpoint (1 = pooled)."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return _multicell_heads(ckpt["model_state_dict"], ckpt_path)[0]


def effective_head(dataset_head: Optional[str], per_head: bool,
                   model_num_heads: Optional[int]) -> Optional[str]:
    """The head that actually scores a dataset, as logged / recorded.

    None on the single-output path (no heads); POOLED_HEAD_LABEL for a pooled
    checkpoint, whose one head scores every screen; else the DATASETS head.
    """
    if not per_head:
        return None
    if model_num_heads == 1:
        return POOLED_HEAD_LABEL
    return dataset_head


def load_multicell_model(ckpt_path: str,
                         device: str = "cpu",
                         load_weights: bool = True) -> SiameseSLMultiCell:
    """Reconstruct a SiameseSLMultiCell from a checkpoint's state dict.

    Mirrors predict.load_model's encoder-shape inference but rebuilds the
    multi-head cell model (which predict.load_model deliberately refuses). The
    biological input_dim is the encoder's input WIDTH minus cell_line_dim,
    because the shared encoder sees [gene_emb || cell_emb].

    load_weights=False returns the same architecture at the constructor's own
    initialisation (drawn from the caller's torch RNG state) and loads nothing
    from the checkpoint: the Scratch fine-tuning mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    num_heads, cell_line_dim = _multicell_heads(sd, ckpt_path)

    if any("hidden_blocks" in k for k in sd):  # residual encoder
        block_keys = sorted(
            [
                k for k in sd if k.startswith("encoder.hidden_blocks.")
                and k.endswith(".0.weight") and sd[k].dim() == 2
            ],
            key=lambda k: int(
                k.split("encoder.hidden_blocks.")[1].split(".")[0]))
        enc_type = "residual"
    else:  # mlp encoder
        block_keys = sorted(
            [
                k for k in sd if k.startswith("encoder.hidden.")
                and k.endswith(".weight") and sd[k].dim() == 2
            ],
            key=lambda k: int(k.split("encoder.hidden.")[1].split(".")[0]))
        enc_type = "mlp"
    proj_key = "encoder.projection.weight"

    enc_in = int(sd[block_keys[0]].shape[1] if block_keys else
                 sd[proj_key].shape[1])  # = input_dim + cell_line_dim
    input_dim = enc_in - cell_line_dim
    encoder_dims = [int(sd[k].shape[0]) for k in block_keys]
    encoder_dims.append(int(sd[proj_key].shape[0]))
    has_bias = "encoder.projection.bias" in sd
    pd_eps = ckpt.get("config", {}).get("pd_epsilon", 0.001)
    # Dropout is not in the weights: restore the value the checkpoint was
    # trained with, so Full / LP_last fine-tune under the same regularisation
    # (the class default 0.2 was used before, whatever the run had set).
    dropout = float(ckpt.get("config", {}).get("dropout", 0.2))

    # Per-head options, read from the weights (absent in older checkpoints).
    model = SiameseSLMultiCell(
        input_dim=input_dim,
        num_heads=num_heads,
        encoder_dims=encoder_dims,
        dropout=dropout,
        last_layer_bias=has_bias,
        pd_epsilon=pd_eps,
        siamese_encoder_type=enc_type,
        cell_line_dim=cell_line_dim,
        per_head_input_bias="head_input_bias" in sd,
        per_head_temperature=sd.get("log_temperature", sd.get("tau")).dim() > 0,
        raw_temperature="tau" in sd,
        per_head_last_layer_bias="head_proj_bias" in sd,
        kernel="bilinear" if "kernel_A" in sd else "pd",
        gene_term="gene_w" in sd,
        pair_features="pair_fc1.weight" in sd,
        inner_norm="batch" if "inner_running_mean" in sd else "none",
        use_layernorm=any(k.startswith(("encoder.hidden_blocks.0.1.",
                                        "encoder.hidden.1.")) for k in sd),
    )
    if not load_weights:
        _check_same_architecture(model, sd, ckpt_path)
        model.to(device).eval()
        return model
    # Parity with predict.load_model: inject any missing input_bias as zeros.
    for key, param in model.named_parameters():
        if key not in sd and "input_bias" in key:
            sd[key] = torch.zeros_like(param)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def _check_same_architecture(model: nn.Module, sd: dict, ckpt_path) -> None:
    """Raise unless a freshly built model has the checkpoint's tensors: every
    state-dict key with the same shape (a legacy checkpoint may lack an
    input_bias, which the weight-loading path injects as zeros)."""
    mine = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in sd.items()}
    extra = sorted(k for k in set(mine) - set(theirs)
                   if "input_bias" not in k)
    gone = sorted(set(theirs) - set(mine))
    shape = sorted(k for k in set(mine) & set(theirs) if mine[k] != theirs[k])
    if extra or gone or shape:
        raise ValueError(
            f"fresh model rebuilt from {ckpt_path} is not its architecture "
            f"(only in the new model: {extra}; only in the checkpoint: "
            f"{gone}; other shape: {shape})")


class _HeadSelect(nn.Module):
    """Expose ONE cell-line head of a SiameseSLMultiCell as a single-output
    model. forward(x1, x2) -> (batch, 1) = that head's logit (PD or bilinear
    kernel). A pooled checkpoint (num_heads 1) has only head 0.

    `.encoder` proxies the shared encoder so set_trainable / set_frozen_to_eval
    (which reach through `.base.encoder`) operate on it unchanged; they reach
    the multi-cell model itself through `.model`. The cell embedding,
    temperature and per-head scoring bias live OUTSIDE `.encoder` and stay
    frozen in every mode, so LP / LP_last / Full fine-tuning never perturbs the
    learned cell representation. What adapts: the affine head (every mode);
    the projection plus head_proj_bias when present (LP_last); the whole
    shared biological encoder plus head_proj_bias, head_input_bias, kernel_A,
    gene_w and pair_fc1 / pair_fc2 when present (Full). An --inner_norm batch
    standardisation keeps its pretrained running statistics throughout
    (freeze_inner_stats).
    """

    def __init__(self, multicell: SiameseSLMultiCell, head_idx: int):
        super().__init__()
        self.model = multicell
        self.head_idx = int(head_idx)
        self.cell_line_dim = int(multicell.cell_line_dim)

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        logits = self.model(x1, x2)  # (batch, num_heads)
        return logits[:, self.head_idx:self.head_idx + 1]  # (batch, 1)


def _load_eval_base(ckpt, args, head_idx: Optional[int], device: str,
                    init_seed: Optional[int] = None):
    """Per-fold base model for evaluation.

    --per_head : rebuild the multi-head cell model, expose `head_idx`.
    otherwise  : the single-output SiameseSL path (predict.load_model), which
                 still refuses cell checkpoints with a clear NotImplementedError.

    init_seed (the Scratch mode): the same architecture, freshly initialised
    under torch.manual_seed(init_seed) on the CPU (so the draw does not depend
    on the device), no checkpoint weight loaded. The global RNG state is
    restored afterwards, so building it perturbs nothing else.
    """
    fresh = init_seed is not None
    if fresh:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            if getattr(args, "per_head", False):
                mc = load_multicell_model(str(ckpt), device="cpu",
                                          load_weights=False)
            else:
                single = load_model(str(ckpt), model_type="siamese",
                                    device="cpu", load_weights=False)
                _check_same_architecture(
                    single, torch.load(str(ckpt), map_location="cpu",
                                       weights_only=False)[
                                           "model_state_dict"], ckpt)
                return single.to(device).eval()
    elif getattr(args, "per_head", False):
        mc = load_multicell_model(str(ckpt), device=device)
    else:
        return load_model(str(ckpt), model_type="siamese", device=device)
    if mc.num_heads == 1:
        # --cell_lines pooled checkpoint: its single head scores every
        # screen, whatever cell-line head the dataset names.
        head_idx = 0
    return _HeadSelect(mc, head_idx).to(device)


# =============================================================================
# Every head's output (per-cell-line probabilities, fine-tuned weights)
# =============================================================================
# The metrics above read ONE head per screen. The per-pair probability CSV and
# export_pair_predictions.py report every head, labelled as below.

# Label of a one-output model's head: a pooled checkpoint's head is trained on
# the OR label ("SL in any line"), like the single-output SiameseSL. On a
# cell-line model the same label names the max over its heads, the score the
# benchmark grades it on.
ANY_HEAD = "ANY"

# Where main() saves the fine-tuned weights, under its output dir.
FINETUNED_DIR = "finetuned_models"
FINETUNED_FORMAT = "siamese_sl.finetuned.v1"


def head_labels(num_heads: int) -> List[str]:
    """One label per output head, in head order: the cell lines of
    cell_line_vocab.HEADS, ANY_HEAD for a one-head model."""
    if int(num_heads) == 1:
        return [ANY_HEAD]
    if int(num_heads) == len(cell_line_vocab.HEADS):
        return list(cell_line_vocab.HEADS)
    return [f"head{h}" for h in range(int(num_heads))]


def scoring_model(base: nn.Module) -> nn.Module:
    """The module that emits every head's logit: the SiameseSLMultiCell
    behind a _HeadSelect, else the single-output base itself."""
    mc = getattr(base, "model", None)
    return mc if isinstance(mc, SiameseSLMultiCell) else base


def checkpoint_model_config(ckpt_path) -> Dict[str, float]:
    """The two constructor values a state dict cannot give back, with the
    defaults load_multicell_model / predict.load_model fall back to."""
    cfg = torch.load(str(ckpt_path), map_location="cpu",
                     weights_only=False).get("config") or {}
    return {"pd_epsilon": cfg.get("pd_epsilon", 0.001),
            "dropout": float(cfg.get("dropout", 0.2))}


def save_finetuned_model(path: Path, model: FinetunedSL, model_config: Dict,
                         **meta) -> None:
    """Write one fine-tuned (fold, mode) model so it can be scored again.

    `model_state_dict` + `config` use the keys of a train.py checkpoint, so
    load_multicell_model (cell-line / pooled) and predict.load_model
    (single-output) rebuild the scorer from this file unchanged. The affine
    head and `meta` (dataset, head, fold, source checkpoint ...) ride along;
    the prediction is affine.a * logit[head_idx] + affine.c on the TARGET
    scale (sign * raw screen score). The fold's preprocessing transform is NOT
    copied: it stays in the source checkpoint, which `meta` fingerprints.

    Written to <name>.tmp and renamed, and only when the archive is whole: a
    save cut short by a full quota or a killed task (torch.save can return
    normally from the former) must not leave an unreadable file under a
    model's name for export_pair_predictions.py to find.
    """
    scorer = scoring_model(model.base)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save({
            "format": FINETUNED_FORMAT,
            "model_class": type(scorer).__name__,
            "model_state_dict": {k: v.detach().cpu().clone()
                                 for k, v in scorer.state_dict().items()},
            "config": dict(model_config),
            "affine": {"a": float(model.a.detach().cpu()),
                       "c": float(model.c.detach().cpu())},
            **meta,
        }, tmp)
        # torch.save writes a zip archive, whose directory is its last record.
        if not zipfile.is_zipfile(tmp):
            raise OSError(f"{tmp} was not written in full (disk or quota "
                          f"full?)")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


@torch.no_grad()
def compute_all_head_logits(
    base: nn.Module,
    embeddings: torch.Tensor,
    idx1: np.ndarray,
    idx2: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    """Logits of EVERY head for each pair, (n_pairs, num_heads) float32.

    Same batches and eval mode as compute_logits, which reads only the head
    the screen is scored with; that head's column here is the same forward.
    """
    scorer = scoring_model(base)
    scorer.eval()
    out: Optional[np.ndarray] = None
    for start in range(0, len(idx1), batch_size):
        end = min(start + batch_size, len(idx1))
        b1 = torch.as_tensor(idx1[start:end], dtype=torch.long)
        b2 = torch.as_tensor(idx2[start:end], dtype=torch.long)
        lg = scorer(embeddings[b1].to(device),
                    embeddings[b2].to(device)).detach().float().cpu().numpy()
        if out is None:
            out = np.empty((len(idx1), lg.shape[1]), dtype=np.float32)
        out[start:end] = lg
    if out is None:
        out = np.empty((0, int(getattr(scorer, "num_heads", 1))),
                       dtype=np.float32)
    return out


def sigmoid_probability(logits: np.ndarray) -> np.ndarray:
    """sigmoid in float64 (float32 rounds to exactly 1 above a logit of ~17)."""
    return expit(np.asarray(logits, dtype=np.float64))


def head_probability_columns(logits_folds: np.ndarray,
                             labels: List[str]) -> Dict[str, np.ndarray]:
    """prob_<HEAD>_fold<k> / prob_<HEAD>_mean columns from per-fold logits.

    logits_folds: (n_folds, n_pairs, num_heads). A multi-head model also gets
    prob_ANY_*: the max over its heads within each fold, then the fold mean.
    The mean is over the folds' PROBABILITIES, not the sigmoid of a mean logit.
    """
    n_folds = int(logits_folds.shape[0])
    probs = sigmoid_probability(logits_folds)
    by_head = [(lab, probs[:, :, h]) for h, lab in enumerate(labels)]
    if len(labels) > 1:
        by_head.append((ANY_HEAD, probs.max(axis=2)))
    cols: Dict[str, np.ndarray] = {}
    for lab, p in by_head:
        for k in range(n_folds):
            cols[f"prob_{lab}_fold{k}"] = p[k]
        cols[f"prob_{lab}_mean"] = p.mean(axis=0)
    return cols


def pair_symbols(entrez: pd.Series) -> pd.Series:
    """HGNC symbol of each Entrez ID ("" when HGNC has none)."""
    mapper = get_mapper()
    sym = {e: (mapper.entrez_to_symbol(e) or "")
           for e in sorted(set(entrez.astype(str)))}
    return entrez.astype(str).map(sym)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """frame.to_csv(path, index=False) through <name>.tmp and a rename: a
    write cut short (a full quota, a killed task) leaves nothing under the
    name sl_comparison/external_transfer.py and paper_tables.py read."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# =============================================================================
# External dataset loaders
# =============================================================================
# Each loader returns a DataFrame with columns: gene1_symbol, gene2_symbol,
# outcome_raw (the selected column, NaN-dropped). Normalization to Entrez IDs
# and outcome-sign application happens in the common pipeline downstream.


def _load_adamson(path: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    keep = ["FirstGene", "SecondGene", column]
    df = df[keep].dropna()
    df = df.rename(
        columns={
            "FirstGene": "gene1_symbol",
            "SecondGene": "gene2_symbol",
            column: "outcome_raw"
        })
    df["gene1_symbol"] = df["gene1_symbol"].astype(str).str.upper()
    df["gene2_symbol"] = df["gene2_symbol"].astype(str).str.upper()
    # Drop self-pairs (same gene with itself) — present in raw Adamson.
    df = df[df["gene1_symbol"] != df["gene2_symbol"]].copy()
    return df


def _load_corn(path: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["gene_combination", column]].dropna()
    # gene_combination is "GENE1;GENE2" with an optional "_mis" suffix.
    splits = df["gene_combination"].str.split(";", expand=True)
    if splits.shape[1] < 2:
        raise ValueError(f"Corn gene_combination failed to split on ';': "
                         f"got {splits.shape[1]} columns in {path}")
    df["gene1_symbol"] = (splits[0].str.replace("_mis", "",
                                                regex=False).str.upper())
    df["gene2_symbol"] = (splits[1].str.replace("_mis", "",
                                                regex=False).str.upper())
    df = df.rename(columns={column: "outcome_raw"})
    df = df[["gene1_symbol", "gene2_symbol", "outcome_raw"]].dropna()
    df = df[df["gene1_symbol"] != df["gene2_symbol"]].copy()
    return df


def _load_gilbert(path: str, column: str) -> pd.DataFrame:
    """Load Horlbeck et al. 2018 (Cell) mmc5.xlsx, the K562/Jurkat GI map.

    The xlsx has a 3-row MultiIndex header on the 'gene GI scores and
    correlations' sheet:
      Level 0: Cell line (K562 / Jurkat), first two cols unnamed (genes)
      Level 1: Replicate (Rep1, Rep2, ..., Replicate Average, ...)
      Level 2: Metric (GI score, Correlation, ...)
    plus an empty row at position 3 that must be skipped.

    The `column` argument is a semantic alias; we currently support
    'K562_avg' → ('K562', 'Replicate Average', 'GI score') as in the
    reference qmd pipeline. Matches what gene_pairs_df['Gilbert_score']
    contains in the CIHR figure.
    """
    # Match the qmd's read parameters exactly so values align row-for-row.
    gdf = pd.read_excel(
        path,
        sheet_name='gene GI scores and correlations',
        header=[0, 1, 2],
        skiprows=[3],
    )

    if column == "K562_avg":
        target = ("K562", "Replicate Average", "GI score")
    else:
        # Also accept a pipe-delimited triple in the config, e.g.
        # 'Jurkat|Replicate Average|GI score' — kept open for future use.
        parts = column.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"Gilbert column '{column}' is not a known alias and not a "
                f"'Level0|Level1|Level2' triple. Supported: K562_avg")
        target = tuple(parts)

    target_col = None
    for col in gdf.columns:
        if tuple(col) == target:
            target_col = col
            break
    if target_col is None:
        raise ValueError(f"Gilbert column {target} not found in {path}. "
                         f"Available (first 10): {list(gdf.columns)[:10]}")

    # First two columns are gene names (unnamed in the header rows).
    # Drop NaN rows BEFORE stringifying so "nan" doesn't leak in as a fake
    # gene symbol (astype(str) would otherwise convert NaN to the literal
    # string "nan", inflating `pairs_raw` in the coverage report).
    out = pd.DataFrame({
        "gene1_symbol":
        gdf.iloc[:, 0],
        "gene2_symbol":
        gdf.iloc[:, 1],
        "outcome_raw":
        pd.to_numeric(gdf[target_col], errors="coerce"),
    }).dropna()
    out["gene1_symbol"] = out["gene1_symbol"].astype(str).str.upper()
    out["gene2_symbol"] = out["gene2_symbol"].astype(str).str.upper()
    out = out[out["gene1_symbol"] != out["gene2_symbol"]].copy()
    return out


# The published gene-level genetic-interaction scores of the "Adamson" screen:
# Simpson, Ling, Jing & Adamson, "Mapping the Genetic Interaction Network of
# PARP inhibitor Response", bioRxiv 2023, doi:10.1101/2023.08.19.553986
# (PMID 37645833), Table S3 ("Genetic interaction scores for all gene pairs",
# media-3.xlsx of the bioRxiv supplement). The same K562 CRISPRi dual-sgRNA
# screen as external data/pair_sgRNA_phenotypes.txt: 543 genes, 147,153 gene
# pairs, 1,264,760 analysed constructs (that file's row count). GI = observed
# minus expected dual-sgRNA phenotype, the expectation a per-query-sgRNA
# (mostly quadratic) fit of the single-sgRNA phenotypes, averaged to the gene
# pair; the paper: "positive and negative scores representing buffering and
# synergistic interactions, respectively" (its InteractionCategory column
# calls the significant negative unmodified ones "synthetic lethal"). So the
# SL direction is NEGATIVE GI and the DATASETS sign is -1, as for Gamma.
# Columns: Gamma.GI (untreated growth), Tau.GI (niraparib-treated), Nu.GI
# (Tau.GI - Gamma.GI, drug-modified); each with .Nsupp/.Pvalue/.Qvalue.
ADAMSON_GI_SHEET_TITLE = "Genetic interaction scores for all gene pairs."


def _load_adamson_gi(path: str, column: str) -> pd.DataFrame:
    """Simpson et al. 2023 Table S3 (gene-level GI scores; see above).

    Row 1 is the sheet title, row 2 the header; GeneCombinationName is
    'GENE1:GENE2'. Self pairs ('Same gene') and the non-targeting control
    rows are dropped, as are rows whose `column` is empty (no score for that
    phenotype). A .csv / .tsv copy with the same header row is read too.
    """
    p = str(path)
    if p.endswith((".csv", ".tsv", ".txt")):
        df = pd.read_csv(p, sep="," if p.endswith(".csv") else "\t")
        if "GeneCombinationName" not in df.columns:
            df = pd.read_csv(p, sep="," if p.endswith(".csv") else "\t",
                             skiprows=1)
    else:
        df = pd.read_excel(p, sheet_name=0, skiprows=1)
    for need in ("GeneCombinationName", column):
        if need not in df.columns:
            raise ValueError(
                f"AdamsonGI: column {need!r} not in {path} (expected "
                f"Simpson et al. 2023 Table S3; columns: "
                f"{[c for c in df.columns if not str(c).startswith('Unnamed')][:20]})")
    df = df[["GeneCombinationName", column]].copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna()
    parts = df["GeneCombinationName"].astype(str).str.split(":", expand=True)
    if parts.shape[1] != 2:
        raise ValueError(f"AdamsonGI: GeneCombinationName in {path} is not "
                         f"'GENE1:GENE2' ({parts.shape[1]} fields)")
    # A name without ':' (or with an empty side) splits into a null / empty
    # gene: not a pair (a NaN gene2 would pass the self-pair filter below).
    ok = (parts[0].notna() & parts[1].notna()
          & (parts[0].fillna("").str.strip() != "")
          & (parts[1].fillna("").str.strip() != ""))
    parts, df = parts[ok], df[ok]
    out = pd.DataFrame({
        "gene1_symbol": parts[0].str.strip().str.upper(),
        "gene2_symbol": parts[1].str.strip().str.upper(),
        "outcome_raw": df[column].astype(float),
    })
    ctrl = (out["gene1_symbol"].str.startswith("NON-TARGETING")
            | out["gene2_symbol"].str.startswith("NON-TARGETING"))
    out = out[~ctrl & (out["gene1_symbol"] != out["gene2_symbol"])]
    return out.reset_index(drop=True)


DATASET_LOADERS: Dict[str, Callable[[str, str], pd.DataFrame]] = {
    "Adamson": _load_adamson,
    "AdamsonGI": _load_adamson_gi,
    "Corn": _load_corn,
    "Gilbert": _load_gilbert,
}

# =============================================================================
# Pair preparation (symbol → Entrez, dedupe, SynLethDB overlap)
# =============================================================================


# Gene names of the screens that HGNC lists as a PREVIOUS symbol of two genes,
# pinned to the gene the screen means. GeneNameMapper keeps the last HGNC row
# within one field, so QARS (Gilbert's name for glutaminyl-tRNA synthetase,
# QARS1 since 2019) went to EPRS1, whose row also lists QARS: all 446 Gilbert
# pairs of that gene were scored with another gene's features, 6-21% of the
# rows the external-transfer tables compare the models on. MUM1 (Corn: the
# DNA repair factor PWWP3A; also a former name of IRF4) resolves to the right
# gene only through that row order. The same pin as LP_SYMBOL_OVERRIDES in
# SLMGAE-in-pytorch/train_slmgae_shared.py, and local for the same reason:
# gene_corrections_config.py and gene_name_utils.py also feed embedding
# generation and the mapping hashes NSF4SL's features record.
SCREEN_SYMBOL_OVERRIDES: Dict[str, str] = {
    "QARS": "5859",    # QARS1 (EPRS1 also lists QARS as a previous symbol)
    "MUM1": "84939",   # PWWP3A (IRF4 also lists MUM1 as a previous symbol)
    # Corn screens BIVM, ERCC5 and the BIVM-ERCC5 readthrough as three
    # targets; gene_corrections_config.py maps "BIVM.ERCC5" to BIVM, which
    # averaged 546 X;BIVM.ERCC5 rows into X;BIVM (Spearman 0.13 between the
    # two profiles) and made BIVM;BIVM.ERCC5 a self pair. Pinned to the
    # readthrough's own Entrez ID (HGNC BIVM-ERCC5).
    "BIVM.ERCC5": "100533467",
}


def symbol_entrez(mapper, symbol: str) -> Optional[str]:
    """Entrez ID of a gene name a screen or a user gives: a pinned name
    (SCREEN_SYMBOL_OVERRIDES), else what the mapper resolves it to (None
    without a mapper)."""
    pinned = SCREEN_SYMBOL_OVERRIDES.get(str(symbol).upper())
    if pinned is not None or mapper is None:
        return pinned
    return mapper.symbol_to_entrez(symbol)


def _symbols_to_entrez(symbols: pd.Series, mapper) -> pd.Series:
    """Map each symbol to an Entrez ID; unmapped -> NaN."""
    uniq = sorted(set(symbols.dropna().astype(str)))
    out_map: Dict[str, Optional[str]] = {}
    for s in uniq:
        out_map[s] = symbol_entrez(mapper, s)
    return symbols.map(out_map)


def _load_synlethdb_pairs(sl_path: str) -> set:
    """Return the set of unordered (entrez_a, entrez_b) frozensets in SynLethDB."""
    pairs: set = set()
    with open(sl_path) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            pairs.add(frozenset([parts[0], parts[1]]))
    return pairs


def prepare_dataset(
    df_raw: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    sldb_pairs: set,
    sign: float,
    sldb_pos: Optional[set] = None,
    sldb_neg: Optional[set] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Map symbols -> Entrez IDs, filter to covered genes, dedupe, flag overlap.

    Args:
        df_raw: DataFrame with columns gene1_symbol, gene2_symbol, outcome_raw.
        gene_to_idx: Entrez ID -> embedding row index.
        sldb_pairs: set of frozenset({entrez1, entrez2}) from SynLethDB.
        sign: +1 or -1. target = sign * outcome_raw.

    Returns:
        prepared DataFrame with columns:
          gene1_entrez, gene2_entrez, outcome_raw, target,
          idx1, idx2, pair_key (frozenset), sldb_overlap (bool)
        coverage dict:
          pairs_raw, pairs_mapped_both, pairs_kept, sldb_overlap_count
          (+ with sldb_pos / sldb_neg, the two SynLethDB files apart:
          sldb_pos_count, sldb_neg_count (screened non-SL and not SL, the
          shared folds' OR-collapse), and their *_frac shares of pairs_kept)
    """
    mapper = get_mapper()
    cov: Dict[str, int] = {"pairs_raw": int(len(df_raw))}

    df = df_raw.copy()
    df["gene1_entrez"] = _symbols_to_entrez(df["gene1_symbol"], mapper)
    df["gene2_entrez"] = _symbols_to_entrez(df["gene2_symbol"], mapper)
    df = df.dropna(subset=["gene1_entrez", "gene2_entrez"])
    cov["pairs_both_mapped"] = int(len(df))

    df = df[df["gene1_entrez"].isin(gene_to_idx)
            & df["gene2_entrez"].isin(gene_to_idx)].copy()
    cov["pairs_both_in_model"] = int(len(df))

    # Drop self-pairs after Entrez mapping (possible after alias collapse).
    df = df[df["gene1_entrez"] != df["gene2_entrez"]].copy()

    # Symmetrize pair_key so frozenset({A, B}) == frozenset({B, A}),
    # then average duplicate measurements (matches the qmd's policy).
    df["pair_key"] = df.apply(
        lambda r: frozenset([r["gene1_entrez"], r["gene2_entrez"]]),
        axis=1,
    )
    agg = (df.groupby("pair_key",
                      sort=False).agg(outcome_raw=("outcome_raw",
                                                   "mean")).reset_index())
    # Restore two Entrez IDs as sorted tuple for determinism
    agg["gene1_entrez"] = agg["pair_key"].apply(lambda s: sorted(list(s))[0])
    agg["gene2_entrez"] = agg["pair_key"].apply(lambda s: sorted(list(s))[1])
    agg["idx1"] = agg["gene1_entrez"].map(gene_to_idx)
    agg["idx2"] = agg["gene2_entrez"].map(gene_to_idx)
    agg["target"] = sign * agg["outcome_raw"].astype(float)
    agg["sldb_overlap"] = agg["pair_key"].isin(sldb_pairs)

    cov["pairs_kept"] = int(len(agg))
    cov["sldb_overlap_count"] = int(agg["sldb_overlap"].sum())
    if sldb_pos is not None:
        in_pos = agg["pair_key"].isin(sldb_pos)
        cov["sldb_pos_count"] = int(in_pos.sum())
        cov["sldb_pos_frac"] = float(in_pos.mean()) if len(agg) else 0.0
        if sldb_neg is not None:
            in_neg = agg["pair_key"].isin(sldb_neg) & ~in_pos
            cov["sldb_neg_count"] = int(in_neg.sum())
            cov["sldb_neg_frac"] = float(in_neg.mean()) if len(agg) else 0.0

    return agg, cov


# =============================================================================
# Model prediction (forward pass over a pair DataFrame)
# =============================================================================


@torch.no_grad()
def compute_logits(
    model: FinetunedSL,
    embeddings: torch.Tensor,
    idx1: np.ndarray,
    idx2: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    """Forward the pretrained model (ignoring the affine head) in batches."""
    model.eval()
    out = np.empty(len(idx1), dtype=np.float32)
    for start in range(0, len(idx1), batch_size):
        end = min(start + batch_size, len(idx1))
        b1 = torch.as_tensor(idx1[start:end], dtype=torch.long)
        b2 = torch.as_tensor(idx2[start:end], dtype=torch.long)
        x1 = embeddings[b1].to(device)
        x2 = embeddings[b2].to(device)
        out[start:end] = (model.raw_logit(
            x1, x2).detach().cpu().numpy().astype(np.float32))
    return out


# =============================================================================
# Metrics
# =============================================================================


def _safe_metric(fn, *args, **kwargs) -> float:
    """Compute a metric; return NaN on failure (e.g., all-zero labels)."""
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


# compute_metrics: a "1%" tail holding more than this share of the pairs is
# ties at the threshold, not a tail, and its metrics are NaN.
TAIL_MAX_FRAC = 0.02


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Spearman, Pearson, R² + AUROC/AUPR at top-1% and bot-1% of target.

    `pred` is the model's final output (after (a, c)); `target` is the
    signed raw score (sign * outcome_raw). R² is computed as
    1 - SS_res(pred, target)/SS_tot(target) using sklearn.

    AUROC / AUPR @ top-1%: label = 1 where target >= 99th percentile.
    AUROC / AUPR @ bot-1%: label = 1 where target <=  1st percentile.
    Either is NaN when ties at the threshold put more than TAIL_MAX_FRAC of
    the pairs in that tail (the "1%" is then not a tail).
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    m: Dict[str, float] = {}
    m["spearman"] = _safe_metric(lambda: spearmanr(pred, target)[0])
    m["pearson"] = _safe_metric(lambda: pearsonr(pred, target)[0])
    m["r2"] = _safe_metric(r2_score, target, pred)

    top_thr = np.percentile(target, 99)
    bot_thr = np.percentile(target, 1)
    top_lab = (target >= top_thr).astype(int)
    bot_lab = (target <= bot_thr).astype(int)
    # A tail is ill-defined when ties at its threshold pull in far more than
    # 1% of the pairs: Corn's sens.score is exactly 0 for 49% of pairs, so its
    # "bottom 1%" covered half the screen and AUPR@bot1% read as prevalence.
    # Such a tail gets NaN, not a number.
    if top_lab.mean() > TAIL_MAX_FRAC:
        top_lab = np.zeros_like(top_lab)
    if bot_lab.mean() > TAIL_MAX_FRAC:
        bot_lab = np.zeros_like(bot_lab)

    if top_lab.sum() > 0 and top_lab.sum() < len(top_lab):
        m["auroc_top1pct"] = _safe_metric(roc_auc_score, top_lab, pred)
        m["aupr_top1pct"] = _safe_metric(average_precision_score, top_lab,
                                         pred)
    else:
        m["auroc_top1pct"] = float("nan")
        m["aupr_top1pct"] = float("nan")

    if bot_lab.sum() > 0 and bot_lab.sum() < len(bot_lab):
        m["auroc_bot1pct"] = _safe_metric(roc_auc_score, bot_lab, pred)
        m["aupr_bot1pct"] = _safe_metric(average_precision_score, bot_lab,
                                         pred)
    else:
        m["auroc_bot1pct"] = float("nan")
        m["aupr_bot1pct"] = float("nan")
    return m


# Zero-shot rows come in two calibrations: "ols" (a*logit + c, the baseline
# LP_last / Full are warm-started from) and "raw" (the logit itself). Their
# rank metrics differ only through sign(a) — see the module docstring.
ZERO_SHOT_CALIBRATIONS = ("ols", "raw")


def zero_shot_metrics(pred: np.ndarray, target: np.ndarray,
                      calibration: str) -> Dict[str, float]:
    """compute_metrics for one zero-shot row; R² is NaN for the raw logit,
    which is not in the target's units (its R² would only measure scale)."""
    m = compute_metrics(pred, target)
    if calibration == "raw":
        m["r2"] = float("nan")
    return m


# =============================================================================
# Zero-shot closed-form (a, c)
# =============================================================================


def fit_ols_affine(logit_train: np.ndarray,
                   target_train: np.ndarray) -> Tuple[float, float]:
    """OLS closed form: y = a*logit + c minimizing sum (a*l+c - y)^2.

    Uses sample variance guard: if var(logit) is effectively zero, fall back
    to identity (a=1, c=mean target - mean logit) — still a valid scalar
    calibration, and preserves downstream rank metrics.
    """
    l = logit_train.astype(np.float64)
    y = target_train.astype(np.float64)
    lm = l.mean()
    ym = y.mean()
    denom = ((l - lm)**2).sum()
    if denom < 1e-12:
        return 1.0, float(ym - lm)
    a = float(((l - lm) * (y - ym)).sum() / denom)
    c = float(ym - a * lm)
    return a, c


# =============================================================================
# SGD fine-tuning
# =============================================================================


def finetune_one(
    model: FinetunedSL,
    embeddings: torch.Tensor,
    idx1: np.ndarray,
    idx2: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    mode: str,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: str,
    warm_start: Optional[Tuple[float, float]] = None,
    seed: int = 42,
) -> Tuple[FinetunedSL, int]:
    """Fine-tune one (fold, mode), early-stopping on val R². Returns (best_model, best_epoch).

    - Warm-starts (a, c) from OLS if provided (standard trick to skip calibration epochs).
    - Trains with MSE loss on (a * logit + c, target).
    - Saves best-val checkpoint; returns it after training. The warm start is
      scored on val first and competes as epoch 0, so best_epoch == 0 means
      fine-tuning never beat the un-adapted (OLS-calibrated) model on val.
    """
    set_seed(seed)
    if warm_start is not None:
        with torch.no_grad():
            model.a.copy_(torch.tensor(float(warm_start[0])))
            model.c.copy_(torch.tensor(float(warm_start[1])))
    set_trainable(model, mode)
    model.to(device)

    train_i1 = torch.as_tensor(idx1[train_mask], dtype=torch.long)
    train_i2 = torch.as_tensor(idx2[train_mask], dtype=torch.long)
    train_t = torch.as_tensor(target[train_mask], dtype=torch.float32)

    val_i1 = torch.as_tensor(idx1[val_mask], dtype=torch.long)
    val_i2 = torch.as_tensor(idx2[val_mask], dtype=torch.long)
    val_t = target[val_mask]
    # r2_score on fewer than two points raises, _safe_metric turns that into
    # NaN, and `NaN > -inf` is False: best_state would stay the pre-training
    # warm start and this would return an un-adapted model as "fine-tuned".
    if len(val_i1) < 2:
        raise ValueError(
            f"finetune_one: {len(val_i1)} validation pair(s) — too few to "
            f"select a checkpoint on (val R² would be NaN and the returned "
            f"model would be the un-adapted warm start).")

    ds = TensorDataset(train_i1, train_i2, train_t)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=True,
                        generator=g,
                        drop_last=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    # The warm start is a candidate like any epoch: scored on val (eval mode)
    # before training, so a fine-tuned epoch must strictly beat it. Until
    # 2026-10-02 it was never scored and epoch 1 always replaced it, even when
    # worse on val; on the 2026-09-23 run about 30% of the Adamson / Corn
    # LP_last / Full fold models ended below their own warm start on test.
    model.eval()
    with torch.no_grad():
        pv0 = model(embeddings[val_i1].to(device),
                    embeddings[val_i2].to(device)).detach().cpu().numpy()
    best_val_r2 = _safe_metric(r2_score, val_t, pv0)
    if not np.isfinite(best_val_r2):   # unscoreable: let epoch 1 replace it
        best_val_r2 = -float("inf")
    best_epoch = 0
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        # model.train() recursively set training=True on ALL submodules,
        # including the frozen ones. Put their dropout back to eval so the
        # frozen backbone is deterministic (standard linear-probing convention).
        set_frozen_to_eval(model, mode)
        for b1, b2, yt in loader:
            x1 = embeddings[b1].to(device)
            x2 = embeddings[b2].to(device)
            yt = yt.to(device)
            opt.zero_grad()
            pred = model(x1, x2)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()

        # Val R²
        model.eval()
        with torch.no_grad():
            x1v = embeddings[val_i1].to(device)
            x2v = embeddings[val_i2].to(device)
            pv = model(x1v, x2v).detach().cpu().numpy()
        val_r2 = _safe_metric(r2_score, val_t, pv)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch


@torch.no_grad()
def predict_finetuned(
    model: FinetunedSL,
    embeddings: torch.Tensor,
    idx1: np.ndarray,
    idx2: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx1), dtype=np.float32)
    for start in range(0, len(idx1), batch_size):
        end = min(start + batch_size, len(idx1))
        b1 = torch.as_tensor(idx1[start:end], dtype=torch.long)
        b2 = torch.as_tensor(idx2[start:end], dtype=torch.long)
        x1 = embeddings[b1].to(device)
        x2 = embeddings[b2].to(device)
        out[start:end] = model(x1, x2).detach().cpu().numpy()
    return out


# =============================================================================
# Orchestration
# =============================================================================


def split_pairs(n: int, train_frac: float, val_frac: float,
                seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV1 (edge-based): deterministic 80/10-of-80/20 random pair split.

    Returns boolean masks (train, val, test) of length n. val_frac is the
    fraction of the train+val portion held out as val.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_trval = int(round(train_frac * n))
    n_val = int(round(val_frac * n_trval))
    n_train = n_trval - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_trval:]

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def split_pairs_cv2(
    idx1: np.ndarray,
    idx2: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV2 (gene-based): test pairs have >= 1 unseen gene.

    1. Partition unique genes into trainval_genes (train_frac) / test_genes.
    2. Hold out a val_frac slice of trainval_genes as the VAL genes.
    3. Train: both genes among the remaining (fit) genes.
    4. Val: both genes in trainval_genes, at least one of them a val gene.
    5. Test: at least one gene in test_genes. Nothing is discarded.

    Val is a GENE slice, not a random slice of the train+val pairs: every test
    pair carries a gene the fine-tuning never saw, so a val set whose genes
    were all seen would pick the stopping epoch in a regime the test split
    never shares. sl_comparison/prepare_folds.generate carves the shared SLDB
    cv2/cv3 val by the same rule, and verify_fold asserts it.
    """
    rng = np.random.default_rng(seed)

    all_genes = np.unique(np.concatenate([idx1, idx2]))
    rng.shuffle(all_genes)
    n_trainval_genes = int(round(train_frac * len(all_genes)))
    trainval_genes = all_genes[:n_trainval_genes]
    # The genes are already shuffled, so the leading slice is a random one.
    n_val_genes = max(1, int(round(val_frac * len(trainval_genes))))
    fit_genes = trainval_genes[n_val_genes:]

    both_trainval = (np.isin(idx1, trainval_genes)
                     & np.isin(idx2, trainval_genes))
    test_mask = ~both_trainval
    train_mask = np.isin(idx1, fit_genes) & np.isin(idx2, fit_genes)
    val_mask = both_trainval & ~train_mask  # >= 1 val gene, no test gene
    return train_mask, val_mask, test_mask


def split_pairs_cv3(
    idx1: np.ndarray,
    idx2: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV3 (pair-based): both genes in each test pair are unseen.

    1. Partition unique genes into trainval_genes (train_frac) / test_genes.
    2. Hold out a val_frac slice of trainval_genes as the VAL genes.
    3. Train: both genes among the remaining (fit) genes.
    4. Val: BOTH genes in the val slice — cold-start in the same sense as
       test, which is what the stopping epoch has to reflect (see
       split_pairs_cv2; sl_comparison/prepare_folds.generate does the same).
    5. Test: both genes in test_genes.
    6. Mixed pairs are excluded from every mask: one train and one test gene,
       and now also one fit and one val gene (prepare_folds discards those on
       cv3 too).
    """
    rng = np.random.default_rng(seed)

    all_genes = np.unique(np.concatenate([idx1, idx2]))
    rng.shuffle(all_genes)
    n_trainval_genes = int(round(train_frac * len(all_genes)))
    trainval_genes = all_genes[:n_trainval_genes]
    test_genes = all_genes[n_trainval_genes:]
    n_val_genes = max(1, int(round(val_frac * len(trainval_genes))))
    val_genes = trainval_genes[:n_val_genes]
    fit_genes = trainval_genes[n_val_genes:]

    train_mask = np.isin(idx1, fit_genes) & np.isin(idx2, fit_genes)
    val_mask = np.isin(idx1, val_genes) & np.isin(idx2, val_genes)
    test_mask = np.isin(idx1, test_genes) & np.isin(idx2, test_genes)
    return train_mask, val_mask, test_mask


def parse_datasets_spec(specs: List[str]) -> List[Dict]:
    # The 5th field ("format": tsv/csv/xlsx) is documentary — each loader in
    # DATASET_LOADERS knows its file's format implicitly. We still require it
    # in the spec so `.conf` rows remain self-describing.
    out = []
    for s in specs:
        parts = s.split(":")
        if len(parts) not in (5, 6):
            raise ValueError(
                f"DATASETS row must have 5 or 6 colon-separated fields "
                f"(name:file:column:sign:format[:head]); got: {s}")
        name, filepath, column, sign_str, _fmt = parts[:5]
        # Optional 6th field = cell-line head for --per_head eval (ignored
        # otherwise). Must name one of cell_line_vocab.HEADS.
        head = parts[5] if len(parts) == 6 else None
        out.append({
            "name": name,
            "file": filepath,
            "column": column,
            "sign": float(sign_str),
            "head": head,
        })
    return out


def dataset_input_problems(ds_specs: List[Dict],
                           base_dir: Path,
                           require_head: bool = False) -> List[str]:
    """Why DATASETS rows cannot be scored, one line each ([] = all can).

    sl_comparison/external_transfer.py needs every screen's
    <name>_predictions.csv, and two of the screen files are git-ignored, so a
    missing one must stop the task up front (before the previous outputs are
    cleared and any GPU time is spent), not be skipped with exit 0.

    require_head: also check each row's 6th ':HEAD' field against
    cell_line_vocab.HEADS. run_for_dataset raises on a bad head too, but only
    once THAT dataset's turn comes — after the clear and after the earlier
    screens have spent their GPU time. A pooled (1-head) checkpoint ignores
    the field, so main() passes require_head only for a real multi-head one.
    """
    out = []
    for s in ds_specs:
        if s["name"] not in DATASET_LOADERS:
            out.append(f"{s['name']}: no loader registered (supported: "
                       f"{list(DATASET_LOADERS)})")
        p = Path(base_dir) / s["file"]
        if not p.is_file():
            out.append(f"{s['name']}: file not found: {p}")
        if require_head and s.get("head") not in cell_line_vocab.HEADS:
            out.append(f"{s['name']}: --per_head needs a 6th ':HEAD' field "
                       f"naming a model head, got {s.get('head')!r} "
                       f"(valid: {cell_line_vocab.HEADS})")
    return out


def _file_sha256(path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes, chunked (the .pt files are ~200MB each).

    Same digest train.py records as `embeddings_sha256` (train.file_sha256).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_input_problems(ckpt_inputs: List[Tuple[Path, Optional[list],
                                                      Optional[list],
                                                      Optional[list]]],
                              embeddings_paths: List[str],
                              gene_order: List[str]) -> List[str]:
    """Why --embeddings_paths are not the files the fold checkpoints were
    trained on, one line each ([] = no evidence they differ).

    ckpt_inputs: (checkpoint, its stored gene_order, its stored
    embeddings_paths, its stored embeddings_sha256) per fold; None where a
    legacy checkpoint lacks the key. The same guard predict.py and
    cell_line_ablation/zero_cell_emb.py apply: the per-feature transform and
    the width check cannot tell other (e.g. regenerated) embeddings from the
    trained ones. The basename order also catches two same-width modalities
    passed in swapped order, which would otherwise get each other's transform.
    Neither of those catches a same-universe regeneration of the same files
    (run_generate_embeddings.sh rebuilds every .pt at the same path, and
    node2vec / go2vec / onto2vec are not bit-for-bit reproducible), which is
    what train.py's embeddings_sha256 records and the CONTENT check below
    compares.
    """
    names = [Path(p).name for p in embeddings_paths]
    # Hashed once for every fold (they all name the same files), and not at
    # all when every checkpoint is a legacy one — these files are ~200MB each.
    got_shas = ([_file_sha256(p) for p in embeddings_paths]
                if any(c[3] is not None for c in ckpt_inputs) else None)
    out = []
    for ckpt, order, paths, shas in ckpt_inputs:
        if paths is not None and [Path(p).name for p in paths] != names:
            out.append(f"{ckpt}: trained on {[Path(p).name for p in paths]}, "
                       f"got {names} (want the same files in the same order)")
        if order is not None and list(map(str, order)) != gene_order:
            out.append(f"{ckpt}: its stored gene_order differs from the one "
                       f"these embedding files load in, so they are not the "
                       f"files it was trained on (regenerated since?)")
        if shas is not None and list(map(str, shas)) != got_shas:
            out.append(f"{ckpt}: the embedding files' CONTENTS differ from "
                       f"the ones it was trained on (regenerated in place?); "
                       f"its stored per-fold PCA basis and these weights "
                       f"would score different vectors")
    return out


def _load_fold_transform_and_embeddings(
    ckpt_path: Path,
    raw_per_modality: List[torch.Tensor],
    fallback_pca_variance: Optional[float],
    fallback_post_pca_variance: Optional[float],
    modality_labels: List[str],
) -> torch.Tensor:
    """Return the fold's preprocessed embedding matrix.

    Prefers the `preprocessing_transform` stored inside the checkpoint
    (new-format, leakage-free). Falls back to fitting on ALL genes via
    --pca_variance / --post_pca_variance (legacy, leaky) if the checkpoint
    pre-dates the fix, emitting a loud warning.
    """
    ckpt_raw = torch.load(str(ckpt_path),
                          map_location="cpu",
                          weights_only=False)
    transform = ckpt_raw.get("preprocessing_transform")
    if transform is not None:
        return apply_multimodal_transform(raw_per_modality, transform)

    print(
        f"  WARNING: checkpoint {ckpt_path.name} has no "
        f"`preprocessing_transform` — falling back to all-gene PCA fit "
        f"using --pca_variance/--post_pca_variance. This REINTRODUCES "
        f"the CV2/CV3 leakage the new pipeline fixes; retrain to remove "
        f"the warning.",
        file=sys.stderr)
    if fallback_pca_variance is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} lacks preprocessing_transform and no "
            f"--pca_variance was provided for fallback.")
    n_genes = raw_per_modality[0].shape[0]
    fit_mask = torch.ones(n_genes, dtype=torch.bool)
    pca_dims = [fallback_pca_variance] * len(raw_per_modality)
    legacy_transform = fit_multimodal_transform(
        raw_per_modality,
        fit_mask,
        pca_dims=pca_dims,
        post_pca=fallback_post_pca_variance,
        labels=modality_labels)
    return apply_multimodal_transform(raw_per_modality, legacy_transform)


def resolve_fold_checkpoints(ckpt_dir: Path, num_folds: int) -> List[Path]:
    """fold_<k>_best.pt for k = 0..num_folds-1, in fold order.

    Refuses a partial or gapped set: run_for_dataset names folds by list
    position (logit_fold<k> / pred_fold<k>, the "fold" column, the
    fine-tune seed), so the survivors {1, 3} of a damaged set would be
    published as folds 0 and 1, and every summary would average a subset.
    """
    by_fold: Dict[int, Path] = {}
    for p in Path(ckpt_dir).glob("fold_*_best.pt"):
        try:
            by_fold[int(p.stem.split("_")[1])] = p
        except (IndexError, ValueError):
            continue
    want = list(range(int(num_folds)))
    if sorted(by_fold) != want:
        raise ValueError(
            f"{ckpt_dir} holds fold checkpoints {sorted(by_fold)} but the "
            f"model's num_folds={num_folds} needs folds {want}; refusing a "
            f"partial checkpoint set (folds are never renumbered)")
    return [by_fold[k] for k in want]


def run_for_dataset(
    ds_spec: Dict,
    base_dir: Path,
    raw_per_modality: List[torch.Tensor],
    gene_to_idx: Dict[str, int],
    checkpoint_paths: List[Path],
    sldb_pairs: set,
    device: str,
    args,
    out_dir: Path,
    modality_labels: List[str],
    model_num_heads: Optional[int] = None,
    checkpoint_sha256: Optional[Dict[str, str]] = None,
    aux_problems: Optional[List[str]] = None,
    sldb_pos: Optional[set] = None,
    sldb_neg: Optional[set] = None,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """Evaluate and fine-tune for one external dataset. Returns:
      zero_shot_rows, finetuned_rows, coverage_dict

    model_num_heads: heads of the --per_head checkpoint (checkpoint_num_heads);
    1 = pooled, whose single head scores the dataset whatever its DATASETS
    head field says.
    checkpoint_sha256: {str(checkpoint): sha256}, stamped into each saved
    fine-tuned model as its source.
    aux_problems: one line per optional all-pair output that could not be
    written ({name}_sl_probabilities.csv, {name}_finetuned_allpairs_{mode}.csv,
    finetuned_models/). They never cost the screen its tables: every failure
    there is caught and listed here, and main() turns a non-empty list into
    exit 1 after everything else is on disk.
    """
    name = ds_spec["name"]
    col = ds_spec["column"]
    sign = ds_spec["sign"]
    path = base_dir / ds_spec["file"]

    print(f"\n{'=' * 78}\nDataset: {name}\n{'=' * 78}")
    print(f"  file    : {path}")
    print(f"  column  : {col}")
    print(f"  sign    : {sign:+g} (target = sign * raw_score)")

    # Per-head (cell-line-conditioned) eval: resolve which model head scores
    # this dataset. head_idx stays None on the default single-output path.
    head_idx: Optional[int] = None
    if getattr(args, "per_head", False) and model_num_heads == 1:
        # --cell_lines pooled checkpoint: its ONE head (trained on the OR
        # label) scores every screen, whatever cell line DATASETS names. Log
        # the head actually used, not the dataset's.
        head_idx = 0
        print(f"  head    : {POOLED_HEAD_LABEL} (idx 0) [pooled checkpoint; "
              f"DATASETS names {ds_spec.get('head') or 'no head'}, unused]")
    elif getattr(args, "per_head", False):
        head_name = ds_spec.get("head")
        if not head_name:
            raise ValueError(
                f"--per_head is set but dataset '{name}' has no cell-line head. "
                f"Add a 6th ':HEAD' field to its DATASETS row (e.g. "
                f"'...:tsv:K562'). Valid heads: {cell_line_vocab.HEADS}")
        if head_name not in cell_line_vocab.HEADS:
            raise ValueError(
                f"dataset '{name}': head '{head_name}' is not a model head. "
                f"Valid heads: {cell_line_vocab.HEADS}")
        head_idx = cell_line_vocab.HEADS.index(head_name)
        print(
            f"  head    : {head_name} (idx {head_idx}) [per-head cell-line eval]"
        )

    if not path.exists():
        print(f"  SKIP: file not found. Set the dataset file under "
              f"{base_dir}/external\\ data/ to include it.")
        return [], [], {"name": name, "status": "file_not_found"}

    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"No loader registered for dataset '{name}'. "
                         f"Supported: {list(DATASET_LOADERS.keys())}")

    df_raw = loader(str(path), col)
    df, cov = prepare_dataset(df_raw, gene_to_idx, sldb_pairs, sign,
                              sldb_pos=sldb_pos, sldb_neg=sldb_neg)
    cov["name"] = name
    print(
        f"  coverage: raw={cov['pairs_raw']:,}  both_mapped="
        f"{cov['pairs_both_mapped']:,}  both_in_model="
        f"{cov['pairs_both_in_model']:,}  kept={cov['pairs_kept']:,}  "
        f"sldb_overlap={cov['sldb_overlap_count']:,} "
        f"({100.0 * cov['sldb_overlap_count'] / max(1, cov['pairs_kept']):.2f}%)"
    )
    if "sldb_pos_count" in cov:
        print(f"  SynLethDB: positives={cov['sldb_pos_count']:,} "
              f"({100.0 * cov['sldb_pos_frac']:.2f}%)" +
              (f"  screened negatives={cov['sldb_neg_count']:,} "
               f"({100.0 * cov['sldb_neg_frac']:.2f}%)"
               if "sldb_neg_count" in cov else "") +
              "  (shares of kept pairs; per-fold training overlap: "
              "sl_comparison/external_transfer.py sldb_overlap.csv)")

    if len(df) < 100:
        print(f"  SKIP: fewer than 100 evaluable pairs.")
        return [], [], cov

    # copy=True guarantees writable numpy buffers so downstream
    # `torch.as_tensor(arr[slice])` calls don't emit the non-writable warning.
    idx1 = df["idx1"].to_numpy(copy=True)
    idx2 = df["idx2"].to_numpy(copy=True)
    target = df["target"].to_numpy(copy=True)
    outcome_raw = df["outcome_raw"].to_numpy(copy=True)
    overlap_mask = df["sldb_overlap"].to_numpy(copy=True)

    cv = getattr(args, "cv_type", "cv1")
    if cv == "cv2":
        train_mask, val_mask, test_mask = split_pairs_cv2(
            idx1, idx2, args.train_frac, args.val_frac, args.split_seed)
    elif cv == "cv3":
        train_mask, val_mask, test_mask = split_pairs_cv3(
            idx1, idx2, args.train_frac, args.val_frac, args.split_seed)
    else:
        train_mask, val_mask, test_mask = split_pairs(len(df), args.train_frac,
                                                      args.val_frac,
                                                      args.split_seed)

    n_used = int(train_mask.sum() + val_mask.sum() + test_mask.sum())
    n_discarded = len(df) - n_used
    print(
        f"  split({cv}): train={train_mask.sum():,}  val={val_mask.sum():,}  "
        f"test={test_mask.sum():,}" +
        (f"  discarded(mixed)={n_discarded:,}" if n_discarded else ""))

    # A screen with too few genes can leave the cv2/cv3 gene-sliced val (or the
    # train side) nearly empty. finetune_one would then early-stop on a NaN
    # val R² — `NaN > -inf` is False, so best_epoch stays 0 and it returns the
    # UN-adapted warm start, indistinguishable from a warm start that won on
    # val. Fail the screen instead; main() turns a missing predictions CSV
    # into exit 1.
    if int(train_mask.sum()) < 10 or int(val_mask.sum()) < 10:
        print(f"  SKIP: the {cv} split leaves train={int(train_mask.sum())} "
              f"val={int(val_mask.sum())} pairs — too few to fit and to "
              f"select a fine-tuning checkpoint on.")
        return [], [], cov

    n_folds = len(checkpoint_paths)

    # Define test-filter variants once (masks are fold-independent).
    test_filters: List[Tuple[str, np.ndarray]] = [("unfiltered", test_mask)]
    if args.report_sldb_filtered:
        test_filters.append(("sldb_filtered", test_mask & ~overlap_mask))

    # ---- Zero-shot: compute raw logit per fold, OLS on TRAIN, eval on TEST ----
    zero_shot_rows: List[dict] = []
    finetuned_rows: List[dict] = []

    # Collect per-fold raw logits on ALL pairs (used for ensemble later too).
    raw_logits_folds = np.zeros((n_folds, len(df)), dtype=np.float32)
    # Per-fold zero-shot predictions (a_k * l_k + c_k) on ALL pairs.
    zshot_preds_folds = np.zeros_like(raw_logits_folds)
    # Per-fold fine-tuned predictions on ALL pairs, keyed by mode.
    ft_preds_folds: Dict[str, np.ndarray] = {
        m: np.zeros((n_folds, len(df)), dtype=np.float32)
        for m in args.ft_modes
    }
    # Per-(fold, mode) best-epoch for metadata.
    best_epoch_tracker: Dict[Tuple[int, str], int] = {}
    # ---- the optional all-pair outputs --------------------------------------
    # Nothing a table reads depends on them, so nothing in them may stop the
    # screen: each step is guarded and reports through _aux instead.
    def _aux(what: str, err) -> None:
        msg = f"{name}: {what} NOT written ({err})"
        print(f"  ERROR: {msg}", file=sys.stderr)
        if aux_problems is not None:
            aux_problems.append(msg)

    # Every head's zero-shot logit on ALL pairs (the per-cell-line probability
    # CSV); allocated at the first fold, when the head count is known.
    head_logits_folds: Optional[np.ndarray] = None
    head_logit_gap = 0.0  # worst |scored head's column - logit_fold<k>| / tol
    head_error: Optional[str] = None
    # The column of that array the screen is scored with (0 on the one-head
    # paths), and the label it is recorded under.
    head_col = int(head_idx) if head_idx is not None else 0
    scored_head = effective_head(ds_spec.get("head"),
                                 bool(getattr(args, "per_head", False)),
                                 model_num_heads)
    save_ft = bool(getattr(args, "save_finetuned", True))
    ft_dir = out_dir / FINETUNED_DIR

    # main() passes fold_0 .. fold_{K-1} in order (resolve_fold_checkpoints),
    # so the position k IS the fold id in every column, row and seed below.
    for k, ckpt in enumerate(checkpoint_paths):
        print(f"  [fold {k}] loading checkpoint {ckpt.name}")
        # Rebuild this fold's preprocessed embedding matrix from its own
        # saved transform. Each fold uses a DIFFERENT feature space
        # (preprocessing was fit on that fold's training-pair genes), so we
        # cannot share a single `embeddings` tensor across folds anymore.
        fold_embeddings = _load_fold_transform_and_embeddings(
            ckpt,
            raw_per_modality,
            fallback_pca_variance=args.pca_variance,
            fallback_post_pca_variance=args.post_pca_variance,
            modality_labels=modality_labels,
        )
        base = _load_eval_base(ckpt, args, head_idx, device)
        # Sanity: the encoder's first Linear takes input_width features. For
        # a per-head cell model input_width = biological_dim + cell_line_dim,
        # so subtract cell_line_dim before comparing to the fold's biological
        # embedding dim. (input_bias is None under --per_head_input_bias.)
        enc = base.encoder
        first = (enc.hidden_blocks[0][0] if hasattr(enc, "hidden_blocks")
                 and len(enc.hidden_blocks) else
                 enc.hidden[0] if hasattr(enc, "hidden") and len(enc.hidden)
                 else getattr(enc, "projection", None))
        # A legacy single-output encoder with no projection Linear
        # (SiameseSLKernel's EfficientEncoder) still carries input_bias.
        input_width = int(first.in_features if first is not None else
                          enc.input_bias.shape[0])
        expected_dim = (input_width - base.cell_line_dim if getattr(
            args, "per_head", False) else input_width)
        if expected_dim != fold_embeddings.shape[1]:
            raise ValueError(
                f"fold {k}: embedding dim {fold_embeddings.shape[1]} does "
                f"not match checkpoint's expected input_dim {expected_dim}. "
                f"The saved preprocessing_transform is out of sync with the "
                f"model weights.")

        model = FinetunedSL(base).to(device)

        raw = compute_logits(model, fold_embeddings, idx1, idx2, device)
        raw_logits_folds[k] = raw

        # Every head's logit for the same pairs. The scored head's column is
        # the forward compute_logits just ran, so the two must agree; a gap
        # means the head mapping is wrong and the CSV is withheld below.
        try:
            all_logits = compute_all_head_logits(base, fold_embeddings, idx1,
                                                 idx2, device)
            if head_logits_folds is None:
                head_logits_folds = np.zeros(
                    (n_folds, len(df), all_logits.shape[1]), dtype=np.float32)
            head_logits_folds[k] = all_logits
            _tol = 1e-4 * max(1.0, float(np.abs(raw).max()))
            _gap = float(np.abs(all_logits[:, head_col] - raw).max()) / _tol
            head_logit_gap = max(head_logit_gap,
                                 _gap if np.isfinite(_gap) else float("inf"))
        except Exception as e:  # noqa: BLE001  (optional output, see _aux)
            head_error = head_error or f"fold {k}: {type(e).__name__}: {e}"

        a, c = fit_ols_affine(raw[train_mask], target[train_mask])
        zshot_preds_folds[k] = a * raw + c
        zshot_by_calib = {"ols": zshot_preds_folds[k], "raw": raw}

        for flt_name, flt_mask in test_filters:
            if flt_mask.sum() < 10:
                continue
            tgts = target[flt_mask]
            # Both rows carry this fold's (a, c): on the "raw" row a_ols is
            # the sign that separates it from its "ols" twin.
            for calib in ZERO_SHOT_CALIBRATIONS:
                m = zero_shot_metrics(zshot_by_calib[calib][flt_mask], tgts,
                                      calib)
                zero_shot_rows.append({
                    "dataset": name,
                    "calibration": calib,
                    "fold": k,
                    "filter": flt_name,
                    "n_test": int(flt_mask.sum()),
                    "a_ols": float(a),
                    "c_ols": float(c),
                    **m,
                })

        print(f"    zero-shot OLS: a={a:+.4f}  c={c:+.4f}" +
              ("  (a < 0: the calibrated row reverses the raw logit's "
               "ranking)" if a < 0 else ""))

        # ---- Fine-tune (per mode, per fold) — warm-start (a, c) from OLS ----
        for mode in args.ft_modes:
            # Reload fresh base for each mode so folds/modes don't contaminate.
            init_seed = (scratch_init_seed(args.split_seed, k)
                         if mode == SCRATCH_MODE else None)
            base_ft = _load_eval_base(ckpt, args, head_idx, device,
                                      init_seed=init_seed)
            model_ft = FinetunedSL(base_ft).to(device)
            warm = (a, c)
            if init_seed is not None:
                # Scratch: OLS on the fresh model's OWN logits (train split),
                # the rule Full applies to its starting model; the pretrained
                # logit `raw` above is not used.
                raw_s = compute_logits(model_ft, fold_embeddings, idx1, idx2,
                                       device)
                warm = fit_ols_affine(raw_s[train_mask], target[train_mask])
                print(f"    {mode}: fresh init (torch seed {init_seed}); "
                      f"OLS on its own logits a={warm[0]:+.4f} "
                      f"c={warm[1]:+.4f}")
            model_ft, best_epoch = finetune_one(
                model_ft,
                fold_embeddings,
                idx1,
                idx2,
                target,
                train_mask=train_mask,
                val_mask=val_mask,
                mode=mode,
                epochs=args.ft_epochs,
                patience=args.ft_patience,
                lr=args.ft_lr,
                batch_size=args.ft_batch_size,
                weight_decay=args.ft_weight_decay,
                device=device,
                warm_start=warm,
                # Per-fold seed so the SGD trajectories of the 5 folds are
                # independent draws — important for the ensemble path, which
                # averages logits across folds. Using a constant split_seed
                # made every fold's SGD state identical at init, defeating
                # ensemble diversity. The split itself still uses split_seed.
                seed=args.split_seed + k,
            )
            preds_all = predict_finetuned(model_ft, fold_embeddings, idx1,
                                          idx2, device)
            ft_preds_folds[mode][k] = preds_all
            best_epoch_tracker[(k, mode)] = int(best_epoch)
            if save_ft:
                # The weights behind pred_fold<k>: export_pair_predictions.py
                # scores all gene pairs with them. `head` is the label,
                # `head_idx` the column of the scorer's output it reads.
                try:
                    ft_dir.mkdir(exist_ok=True)
                    save_finetuned_model(
                        ft_dir / f"{name}_{mode}_fold{k}.pt",
                        model_ft,
                        checkpoint_model_config(ckpt),
                        dataset=name,
                        column=col,
                        sign=float(sign),
                        mode=mode,
                        fold=int(k),
                        best_epoch=int(best_epoch),
                        head_idx=head_col,
                        head=(scored_head
                              if scored_head is not None else ANY_HEAD),
                        cv_type=cv,
                        split_seed=int(args.split_seed),
                        train_frac=float(args.train_frac),
                        val_frac=float(args.val_frac),
                        n_pairs=int(len(df)),
                        n_genes=int(len(np.unique(
                            np.concatenate([idx1, idx2])))),
                        n_train=int(train_mask.sum()),
                        n_val=int(val_mask.sum()),
                        n_test=int(test_mask.sum()),
                        source_checkpoint=str(ckpt),
                        source_checkpoint_sha256=(checkpoint_sha256
                                                  or {}).get(str(ckpt)),
                        # Scratch: no weight of the source checkpoint is in
                        # this model, only its feature space (transform).
                        **({"init": "scratch", "init_seed": int(init_seed)}
                           if init_seed is not None else {}),
                    )
                except Exception as e:  # noqa: BLE001  (optional, see _aux)
                    _aux(f"{FINETUNED_DIR}/{name}_{mode}_fold{k}.pt",
                         f"{type(e).__name__}: {e}")

            for flt_name, flt_mask in test_filters:
                if flt_mask.sum() < 10:
                    continue
                preds = preds_all[flt_mask]
                tgts = target[flt_mask]
                m = compute_metrics(preds, tgts)
                finetuned_rows.append({
                    "dataset": name,
                    "mode": mode,
                    "fold": k,
                    "filter": flt_name,
                    "n_test": int(flt_mask.sum()),
                    "best_epoch": int(best_epoch),
                    **m,
                })

    # ---- Ensemble rows: mean of per-fold predictions ----
    # Zero-shot ensemble: mean of the calibrated predictions ("ols") and of
    # the raw logits ("raw").
    zshot_ensemble_pred = {
        "ols": zshot_preds_folds.mean(axis=0),
        "raw": raw_logits_folds.mean(axis=0),
    }
    for flt_name, flt_mask in test_filters:
        if flt_mask.sum() < 10:
            continue
        for calib in ZERO_SHOT_CALIBRATIONS:
            m = zero_shot_metrics(zshot_ensemble_pred[calib][flt_mask],
                                  target[flt_mask], calib)
            zero_shot_rows.append({
                "dataset": name,
                "calibration": calib,
                "fold": "ensemble",
                "filter": flt_name,
                "n_test": int(flt_mask.sum()),
                "a_ols": float("nan"),
                "c_ols": float("nan"),
                **m,
            })

    # Fine-tuned ensemble, per mode
    for mode in args.ft_modes:
        ensemble_pred = ft_preds_folds[mode].mean(axis=0)
        for flt_name, flt_mask in test_filters:
            if flt_mask.sum() < 10:
                continue
            m = compute_metrics(ensemble_pred[flt_mask], target[flt_mask])
            finetuned_rows.append({
                "dataset": name,
                "mode": mode,
                "fold": "ensemble",
                "filter": flt_name,
                "n_test": int(flt_mask.sum()),
                "best_epoch": -1,  # not meaningful for the ensemble row
                **m,
            })

    # ---- Save per-dataset zero-shot predictions (all pairs) ----
    pred_cols = {f"logit_fold{k}": raw_logits_folds[k] for k in range(n_folds)}
    pred_cols["logit_mean"] = raw_logits_folds.mean(axis=0)
    zshot_df = pd.DataFrame({
        "gene1_entrez":
        df["gene1_entrez"],
        "gene2_entrez":
        df["gene2_entrez"],
        "outcome_raw":
        outcome_raw,
        "target":
        target,
        "sldb_overlap":
        overlap_mask,
        "split":
        np.where(
            train_mask, "train",
            np.where(val_mask, "val", np.where(test_mask, "test",
                                               "discarded"))),
        **pred_cols,
    })
    write_csv_atomic(zshot_df, out_dir / f"{name}_predictions.csv")

    # ---- Save per-dataset fine-tuned predictions (test split only, one CSV per mode) ----
    for mode in args.ft_modes:
        ft_pred_cols = {
            f"pred_fold{k}": ft_preds_folds[mode][k][test_mask]
            for k in range(n_folds)
        }
        ft_pred_cols["pred_mean"] = ft_preds_folds[mode].mean(
            axis=0)[test_mask]
        ft_df = pd.DataFrame({
            "gene1_entrez":
            df["gene1_entrez"].to_numpy()[test_mask],
            "gene2_entrez":
            df["gene2_entrez"].to_numpy()[test_mask],
            "outcome_raw":
            outcome_raw[test_mask],
            "target":
            target[test_mask],
            "sldb_overlap":
            overlap_mask[test_mask],
            **ft_pred_cols,
        })
        write_csv_atomic(
            ft_df, out_dir / f"{name}_finetuned_predictions_{mode}.csv")

    # ---- Every pair, every head: the two all-pair tables ----
    # Separate files, so the two CSVs above keep their columns and row sets:
    # sl_comparison/external_transfer.py refuses a fine-tuned CSV whose rows
    # are not the test rows, and paper_tables.py counts the rows of the first.
    try:
        pair_cols = {
            "gene1_entrez": df["gene1_entrez"].to_numpy(),
            "gene2_entrez": df["gene2_entrez"].to_numpy(),
            "gene1_symbol": pair_symbols(df["gene1_entrez"]).to_numpy(),
            "gene2_symbol": pair_symbols(df["gene2_entrez"]).to_numpy(),
            "outcome_raw": outcome_raw,
            "target": target,
            "sldb_overlap": overlap_mask,
            "split": zshot_df["split"].to_numpy(),
        }
    except Exception as e:  # noqa: BLE001  (optional output, see _aux)
        pair_cols = None
        _aux("the all-pair tables", f"{type(e).__name__}: {e}")

    # Zero-shot SL probability of every head (cell line), float64, written
    # with 12 significant digits (distinct up to a logit of about 27; the
    # measured columns keep their full precision).
    if pair_cols is None:
        pass
    elif head_error is not None or head_logits_folds is None:
        _aux(f"{name}_sl_probabilities.csv", head_error or "no fold scored")
    elif head_logit_gap > 1.0:
        _aux(f"{name}_sl_probabilities.csv",
             f"head column {head_col} of the all-head forward differs from "
             f"logit_fold<k> by {head_logit_gap:.3g}x the tolerance")
    else:
        try:
            labels = head_labels(head_logits_folds.shape[2])
            prob_cols = head_probability_columns(head_logits_folds, labels)
            pd.DataFrame({
                **pair_cols,
                **{c: np.char.mod("%.12g", v) for c, v in prob_cols.items()},
            }).to_csv(out_dir / f"{name}_sl_probabilities.csv", index=False)
            print(f"  SL probabilities: {len(df):,} pairs x heads "
                  f"{labels + ([ANY_HEAD] if len(labels) > 1 else [])} -> "
                  f"{name}_sl_probabilities.csv (largest logit "
                  f"{float(head_logits_folds.max()):.1f})")
        except Exception as e:  # noqa: BLE001  (optional output, see _aux)
            (out_dir / f"{name}_sl_probabilities.csv").unlink(missing_ok=True)
            _aux(f"{name}_sl_probabilities.csv", f"{type(e).__name__}: {e}")

    # Fine-tuned predicted screen value for EVERY pair (the CSV above holds
    # the test rows only). pred_* is on the target scale the model was
    # trained on (sign * raw score); pred_outcome_* undoes the sign, i.e. it
    # is in the units of outcome_raw. Only split == 'test' rows are held out.
    for mode in (args.ft_modes if pair_cols is not None else []):
        all_csv = out_dir / f"{name}_finetuned_allpairs_{mode}.csv"
        try:
            preds = ft_preds_folds[mode]
            pred_mean = preds.mean(axis=0)
            all_cols = {f"pred_fold{k}": preds[k] for k in range(n_folds)}
            all_cols["pred_mean"] = pred_mean
            for k in range(n_folds):
                all_cols[f"pred_outcome_fold{k}"] = preds[k] / np.float32(sign)
            all_cols["pred_outcome_mean"] = pred_mean / np.float32(sign)
            pd.DataFrame({**pair_cols, **all_cols}).to_csv(all_csv,
                                                          index=False)
        except Exception as e:  # noqa: BLE001  (optional output, see _aux)
            all_csv.unlink(missing_ok=True)
            _aux(all_csv.name, f"{type(e).__name__}: {e}")

    # Print per-dataset results table to stdout (visible in slurm .out)
    _print_dataset_results(
        name=name,
        zero_shot_rows=zero_shot_rows,
        finetuned_rows=finetuned_rows,
        ft_modes=list(args.ft_modes),
        metric_cols=METRIC_COLS,
        report_sldb_filtered=bool(args.report_sldb_filtered),
    )

    return zero_shot_rows, finetuned_rows, cov


# =============================================================================
# Summary helpers
# =============================================================================

# Metric label table used by both the CSV output and the stdout tables.
_METRIC_LABELS = {
    "spearman": "Spearman",
    "pearson": "Pearson",
    "r2": "R²",
    "auroc_top1pct": "AUROC@top1%",
    "aupr_top1pct": "AUPR@top1%",
    "auroc_bot1pct": "AUROC@bot1%",
    "aupr_bot1pct": "AUPR@bot1%",
}


def _print_dataset_results(
    name: str,
    zero_shot_rows: List[dict],
    finetuned_rows: List[dict],
    ft_modes: List[str],
    metric_cols: List[str],
    report_sldb_filtered: bool,
) -> None:
    """Print per-dataset metrics table to stdout (train.py-style).

    For each filter (unfiltered, sldb_filtered), emits one table with a row
    per (mode, aggregation). Aggregations: mean±std across per-fold rows,
    and the single ensemble row. Zero-shot is always shown, OLS-calibrated
    and raw; fine-tuned modes appear in `ft_modes` order. Columns are the 7
    metrics.
    """
    col_w = 15
    mode_w = 13  # fits "Zero-shot OLS" / "Zero-shot raw"
    filters = ["unfiltered"]
    if report_sldb_filtered:
        filters.append("sldb_filtered")

    print()
    print("=" * 80)
    print(f"RESULTS: {name}")
    print("=" * 80)

    for flt in filters:
        zs = [r for r in zero_shot_rows if r["filter"] == flt]
        ft_by_mode = {
            m: [
                r for r in finetuned_rows
                if r["filter"] == flt and r["mode"] == m
            ]
            for m in ft_modes
        }

        if not zs and not any(ft_by_mode.values()):
            continue  # nothing to show (e.g., too few pairs after filter)

        # N_test is fold-independent and filter-specific.
        sample_row = zs[0] if zs else next(r for rs in ft_by_mode.values()
                                           if rs for r in rs)
        n_test = sample_row["n_test"]

        print(f"\nFilter: {flt}  (N = {n_test:,} test pairs)")

        # Header
        label_hdr = f"  {'Mode':<{mode_w}} {'Agg':<9}  "
        metric_hdr = "  ".join(f"{_METRIC_LABELS.get(m, m):>{col_w}s}"
                               for m in metric_cols)
        print(label_hdr + metric_hdr)
        print("  " + "-" * (len(label_hdr) - 2 + len(metric_hdr)))

        blocks: List[Tuple[str, List[dict]]] = [
            ("Zero-shot OLS", [r for r in zs if r["calibration"] == "ols"]),
            ("Zero-shot raw", [r for r in zs if r["calibration"] == "raw"]),
        ]
        for m in ft_modes:
            blocks.append((m, ft_by_mode[m]))

        for block_name, rows in blocks:
            if not rows:
                continue
            perfold = [r for r in rows if r["fold"] != "ensemble"]
            ensemble = [r for r in rows if r["fold"] == "ensemble"]

            if perfold:
                # An all-NaN column (the raw row's R², a degenerate @1% tail)
                # prints nan without numpy's empty-slice warnings.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    means = [
                        float(np.nanmean([r[m] for r in perfold]))
                        for m in metric_cols
                    ]
                    # ddof=1 (sample std) matches pandas groupby().agg("std")
                    # default, so stdout numbers match the summary CSV exactly.
                    stds = [
                        float(np.nanstd([r[m] for r in perfold], ddof=1))
                        for m in metric_cols
                    ]
                cells = "  ".join(f"{mn:+.4f}±{sd:.3f}".rjust(col_w)
                                  for mn, sd in zip(means, stds))
                print(f"  {block_name:<{mode_w}} {'mean±std':<9}  {cells}")
            if ensemble:
                cells = "  ".join(f"{ensemble[0][m]:+.4f}".rjust(col_w)
                                  for m in metric_cols)
                print(f"  {block_name:<{mode_w}} {'ensemble':<9}  {cells}")

    # Footnote: explain the @bot1% convention. Target = sign * raw_score with
    # sign = -1 puts SL-like pairs at HIGH target. @top1% (high-target tail)
    # is the SL-aligned headline metric; @bot1% (low-target tail) treats the
    # NON-SL tail as positive class, so a model that ranks with the target
    # will score sub-0.5 AUROC there. Kept for diagnostic symmetry with the
    # CIHR figure, not as a second headline.
    print()
    print("  Note: @top1% metrics are the SL-aligned headline "
          "(higher = better ranking of SL-like pairs).")
    print("        @bot1% metrics are diagnostic — a model that ranks with "
          "the target typically scores < 0.5 there.")
    print("        Zero-shot OLS = raw up to sign(a_ols) on the rank "
          "metrics: a fold with a < 0 reverses the raw ranking.")


def summarize_metrics(
    rows: List[dict],
    key_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Collapse long-form per-fold metrics into wide mean±std (per filter).

    Ensemble rows (fold == 'ensemble') are preserved as separate rows.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ensemble = df[df["fold"] == "ensemble"].copy()
    perfold = df[df["fold"] != "ensemble"].copy()

    out_parts = []
    if not perfold.empty:
        agg = (perfold.groupby(key_cols)[metric_cols].agg(["mean", "std"
                                                           ]).reset_index())
        # Flatten MultiIndex columns
        agg.columns = (
            list(agg.columns[:len(key_cols)].get_level_values(0)) +
            [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")])
        agg["aggregation"] = "per_fold"
        out_parts.append(agg)
    if not ensemble.empty:
        keep = key_cols + metric_cols
        ens = ensemble[keep].copy()
        # Rename to match mean_std schema — ensemble has no std.
        for m in metric_cols:
            ens[f"{m}_mean"] = ens[m]
            ens[f"{m}_std"] = np.nan
            ens = ens.drop(columns=[m])
        ens["aggregation"] = "ensemble"
        out_parts.append(ens)
    return pd.concat(out_parts, ignore_index=True)


# =============================================================================
# Main
# =============================================================================

METRIC_COLS = [
    "spearman",
    "pearson",
    "r2",
    "auroc_top1pct",
    "aupr_top1pct",
    "auroc_bot1pct",
    "aupr_bot1pct",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate / fine-tune a pretrained siamese SL model on "
        "external CRISPR screens (Adamson / Corn / Gilbert).")
    p.add_argument("--model_dir",
                   required=True,
                   help="Pretrained combo results dir (must contain "
                   "config.json and checkpoints/fold_*_best.pt).")
    p.add_argument("--embeddings_paths",
                   nargs="+",
                   required=True,
                   help="Embedding .pt files matching the pretrained combo.")
    p.add_argument("--pca_variance",
                   type=float,
                   default=None,
                   help="Legacy fallback only: per-modality PCA variance. "
                   "Used ONLY when a checkpoint predates per-fold "
                   "`preprocessing_transform` storage. New checkpoints "
                   "carry their own transform and ignore this flag.")
    p.add_argument("--post_pca_variance",
                   type=float,
                   default=None,
                   help="Legacy fallback only: post-concat PCA variance. "
                   "See --pca_variance note.")
    p.add_argument("--datasets",
                   nargs="+",
                   required=True,
                   help="Spec rows 'name:file:column:sign:format[:head]' "
                   "(the optional :head is the cell-line head used by "
                   "--per_head).")
    p.add_argument("--sl_path",
                   required=True,
                   help="SynLethDB SL .txt for overlap detection.")
    p.add_argument("--neg_path",
                   default=None,
                   help="SynLethDB screened non-SL .txt (the negatives the "
                   "fold models trained on). Its pairs count as overlap too, "
                   "so sldb_overlap / the sldb_filtered rows mean 'not in "
                   "SynLethDB at all', not just 'not a SynLethDB positive'. "
                   "Pass it whenever the models trained on NEG_PATH.")
    p.add_argument("--base_dir",
                   required=True,
                   help="Project base dir; dataset 'file' entries are "
                   "resolved relative to this.")
    p.add_argument("--out_dir",
                   default=None,
                   help="Output dir (default: <model_dir>/eval_finetuning_"
                   "<cv_type>, the only location sl_comparison/"
                   "external_transfer.py reads).")
    p.add_argument("--ft_modes",
                   nargs="+",
                   default=["LP_last", "Full"],
                   choices=list(FT_MODE_CHOICES),
                   help="SGD modes after the closed-form zero-shot rows. "
                   "Scratch = Full's trainable set, optimiser and schedule "
                   "on the same architecture and fold feature space, "
                   "freshly initialised (no SynLethDB weight).")
    p.add_argument("--ft_epochs", type=int, default=100)
    p.add_argument("--ft_patience", type=int, default=10)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--ft_batch_size", type=int, default=256)
    p.add_argument("--ft_weight_decay", type=float, default=0.0)
    p.add_argument("--cv_type",
                   default="cv1",
                   choices=["cv1", "cv2", "cv3"],
                   help="CV split strategy for external data: "
                   "cv1=edge-based (random), "
                   "cv2=gene-based (>=1 unseen gene), "
                   "cv3=pair-based (both genes unseen).")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="train+val share: of the screen's PAIRS under cv1, of "
                   "its GENES under cv2/cv3.")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="val share of that train+val pool: of its PAIRS under "
                   "cv1, of its GENES under cv2/cv3 (val is gene-held-out "
                   "there, like the test split it selects for).")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--report_sldb_filtered",
                   action="store_true",
                   help="Also report metrics with SynLethDB-overlap pairs "
                   "removed from the test split.")
    p.add_argument("--per_head",
                   action="store_true",
                   help="Evaluate a cell-line-conditioned SiameseSLMultiCell "
                   "checkpoint: score each dataset against the model head "
                   "named in its 6th DATASETS field "
                   "(name:file:column:sign:format:HEAD, e.g. K562). A "
                   "--cell_lines pooled checkpoint has one head, which "
                   "scores every dataset (recorded as 'ANY (pooled)' in "
                   "config_used.json effective_head). "
                   "Without it, cell checkpoints are refused as before.")
    p.add_argument("--no_save_finetuned",
                   dest="save_finetuned",
                   action="store_false",
                   help="Do not write finetuned_models/{dataset}_{mode}_"
                   "fold{k}.pt (the fine-tuned weights export_pair_"
                   "predictions.py scores all gene pairs with; under 1 MB "
                   "each at the SLURM default width).")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.split_seed)

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model_dir not found: {model_dir}", file=sys.stderr)
        return 1
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"ERROR: missing {config_file}", file=sys.stderr)
        return 1
    with open(config_file) as f:
        train_cfg = json.load(f)

    ckpt_dir = model_dir / "checkpoints"
    num_folds = train_cfg.get("num_folds")
    if num_folds is None:
        print(f"ERROR: {config_file} records no num_folds, so a complete "
              f"checkpoint set cannot be checked", file=sys.stderr)
        return 1
    # A finished train.py run lists one fold_metrics entry per fold; an ablated
    # dir's results.json (cell_line_ablation) has no fold_metrics.
    _res = model_dir / "results.json"
    if _res.exists():
        try:
            _fm = json.loads(_res.read_text()).get("fold_metrics")
        except (OSError, ValueError):
            _fm = None
        if _fm is not None and len(_fm) != int(num_folds):
            print(f"ERROR: {_res} has {len(_fm)} fold_metrics but config.json "
                  f"num_folds={num_folds}", file=sys.stderr)
            return 1
    # Before the old outputs are cleared: a refused run leaves them intact.
    try:
        checkpoint_paths = resolve_fold_checkpoints(ckpt_dir, num_folds)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    fold_ids = list(range(len(checkpoint_paths)))
    # Fingerprint the exact weights scored here, so a reader of these CSVs
    # (sl_comparison/external_transfer.py) can refuse them once a checkpoint
    # is retrained in place; plus each fold's encoder input width, which only
    # the checkpoints record (config.json's input_dim is null by design).
    checkpoint_sha256: Dict[str, str] = {}
    input_dim_per_fold: List[Optional[int]] = []
    ckpt_inputs: List[Tuple[Path, Optional[list], Optional[list],
                            Optional[list]]] = []
    for p in checkpoint_paths:
        _bytes = p.read_bytes()
        checkpoint_sha256[str(p)] = hashlib.sha256(_bytes).hexdigest()
        _ck = torch.load(io.BytesIO(_bytes), map_location="cpu",
                         weights_only=False)
        input_dim_per_fold.append(
            int(_ck["input_dim"]) if _ck.get("input_dim") is not None else None)
        ckpt_inputs.append((p, _ck.get("gene_order"),
                            _ck.get("embeddings_paths"),
                            _ck.get("embeddings_sha256")))
        del _ck, _bytes
    print(f"Loaded {len(checkpoint_paths)} fold checkpoints (folds "
          f"{fold_ids}) from {ckpt_dir}")
    # Heads of the --per_head checkpoint: 1 = --cell_lines pooled, whose one
    # head scores every screen (logged and recorded as POOLED_HEAD_LABEL).
    model_num_heads: Optional[int] = None
    if args.per_head:
        model_num_heads = checkpoint_num_heads(checkpoint_paths[0])
        print(f"  per-head checkpoint: {model_num_heads} head(s)" +
              (f" -> every dataset scored by {POOLED_HEAD_LABEL}"
               if model_num_heads == 1 else ""))

    # The screen files, their loaders, the DATASETS heads and (below) the
    # embedding inputs are checked here, while a refused run still leaves the
    # previous outputs intact and before any GPU time is spent. A row's
    # `column` is the one input still validated by its loader, when it reads
    # the file.
    ds_specs = parse_datasets_spec(args.datasets)
    problems = dataset_input_problems(
        ds_specs, Path(args.base_dir),
        require_head=bool(args.per_head) and model_num_heads != 1)
    if problems:
        print("ERROR: external screen input(s) cannot be scored (two of the "
              "screen files are git-ignored: copy them into this "
              "checkout):\n  " +
              "\n  ".join(problems), file=sys.stderr)
        return 1

    # --- Load RAW embeddings once; each fold applies its own transform ---
    n_mod = len(args.embeddings_paths)
    print(f"\nLoading RAW embeddings ({n_mod} modalities; per-fold "
          f"preprocessing transforms will be pulled from each checkpoint).")
    raw_per_modality, gene_to_idx, idx_to_gene = load_raw_multimodal(
        args.embeddings_paths)
    modality_labels = [
        Path(p).stem.replace("all_genes_", "") for p in args.embeddings_paths
    ]
    print(f"  raw shapes: "
          f"{[tuple(t.shape) for t in raw_per_modality]}  "
          f"genes: {len(gene_to_idx):,}")
    problems = checkpoint_input_problems(
        ckpt_inputs, args.embeddings_paths,
        [idx_to_gene[i] for i in range(len(idx_to_gene))])
    if problems:
        print("ERROR: --embeddings_paths are not the embedding files the "
              "checkpoints were trained on:\n  " + "\n  ".join(problems) +
              "\nPass the training files in the training order (the combo's "
              "config.json embeddings_paths), or retrain.", file=sys.stderr)
        return 1
    del ckpt_inputs

    # --- Load SynLethDB pairs (positives, plus the screened negatives the
    # models trained on) for overlap detection ---
    sldb_pairs = _load_synlethdb_pairs(args.sl_path)
    sldb_pos = set(sldb_pairs)
    sldb_neg: Optional[set] = None
    print(f"SynLethDB positives loaded: {len(sldb_pairs):,} pairs")
    if args.neg_path:
        _neg = _load_synlethdb_pairs(args.neg_path)
        sldb_neg = _neg
        sldb_pairs = sldb_pairs | _neg
        print(f"SynLethDB screened negatives loaded: {len(_neg):,} pairs "
              f"(overlap = SL or non-SL: {len(sldb_pairs):,} pairs)")
    else:
        print("  WARNING: no --neg_path: sldb_overlap flags SynLethDB "
              "positives only, so the sldb_filtered rows can still hold "
              "screened negatives the models trained on")

    out_dir = Path(
        args.out_dir) if args.out_dir else (model_dir /
                                            f"eval_finetuning_{args.cv_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear a previous run's outputs from this directory first. Every file is
    # rewritten below only for THIS run's datasets and --ft_modes, so e.g. a
    # rerun with --ft_modes LP_last would otherwise leave the old Full CSVs
    # (from older checkpoints) where sl_comparison/external_transfer.py finds
    # them. A crashed run leaves its files missing, never stale.
    _old = sorted({
        p
        for pat in ("*_predictions.csv", "*_finetuned_predictions_*.csv",
                    "*.csv.tmp",
                    "*_sl_probabilities.csv", "*_finetuned_allpairs_*.csv",
                    f"{FINETUNED_DIR}/*.pt", f"{FINETUNED_DIR}/*.pt.tmp",
                    "zero_shot_*.csv", "finetuned_*.csv", "gene_coverage.csv",
                    "config_used.json") for p in out_dir.glob(pat)
    })
    for p in _old:
        p.unlink()
    if _old:
        print(f"Removed {len(_old)} output file(s) of a previous run from "
              f"{out_dir}")

    # --- Resolved config snapshot ---
    resolved = {
        "model_dir": str(model_dir),
        "checkpoints": [str(p) for p in checkpoint_paths],
        # checkpoints[i] is fold_<fold_ids[i]>_best.pt; always 0..K-1, so
        # logit_fold<k> / pred_fold<k> and the "fold" column name fold k.
        "num_folds": len(checkpoint_paths),
        "fold_ids": fold_ids,
        "checkpoint_sha256": checkpoint_sha256,
        "embeddings_paths": args.embeddings_paths,
        "cv_type": args.cv_type,
        "pca_variance_fallback": args.pca_variance,
        "post_pca_variance_fallback": args.post_pca_variance,
        # Per-fold encoder input widths from the checkpoints (config.json's
        # input_dim is written before any fold resolves one, so it is null).
        "input_dim_per_fold": input_dim_per_fold,
        "datasets_spec": args.datasets,
        "ft_modes": args.ft_modes,
        # Scratch only (absent otherwise): fold k's init seed.
        **({"scratch_init_seeds": [scratch_init_seed(args.split_seed, k)
                                   for k in fold_ids],
            "scratch_warm_start": "ols_on_fresh_model_logits"}
           if SCRATCH_MODE in args.ft_modes else {}),
        "ft_epochs": args.ft_epochs,
        "ft_patience": args.ft_patience,
        "ft_lr": args.ft_lr,
        "ft_batch_size": args.ft_batch_size,
        "ft_weight_decay": args.ft_weight_decay,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        # How the fine-tuning checkpoint was selected: cv2/cv3 early-stop on a
        # gene-held-out val slice (same unseen-gene regime as their test
        # split), cv1 on a random pair slice (same regime as its test split).
        "ft_val_regime": ("gene_holdout"
                          if args.cv_type in ("cv2", "cv3") else
                          "random_pairs"),
        "split_seed": args.split_seed,
        "report_sldb_filtered": bool(args.report_sldb_filtered),
        # What sldb_overlap was matched against (external_transfer.py checks
        # it before calling a row set leakage-filtered).
        "sldb_overlap_sources": [args.sl_path] + ([args.neg_path]
                                                  if args.neg_path else []),
        "per_head": bool(args.per_head),
        # dataset_heads: the head each DATASETS row NAMES. effective_head: the
        # head that actually scored it — POOLED_HEAD_LABEL for a pooled
        # (1-head) checkpoint, None on the single-output path.
        "dataset_heads": {s["name"]: s.get("head")
                          for s in ds_specs},
        "model_num_heads": model_num_heads,
        "effective_head": {
            s["name"]: effective_head(s.get("head"), bool(args.per_head),
                                      model_num_heads)
            for s in ds_specs
        },
        # Head labels of {dataset}_sl_probabilities.csv (prob_<label>_*), in
        # head order; a multi-head model also gets prob_ANY_* (max over heads).
        "sl_probability_heads": head_labels(model_num_heads or 1),
        # Sub-directory of the fine-tuned weights, None with
        # --no_save_finetuned.
        "finetuned_models_dir": FINETUNED_DIR if args.save_finetuned else None,
        "device": args.device,
        "n_genes_in_model": len(gene_to_idx),
        "preprocessing": "per-fold (leakage-free) from checkpoint transforms",
    }
    with open(out_dir / "config_used.json", "w") as f:
        json.dump(resolved, f, indent=2)

    # --- Run per dataset ---
    zero_shot_rows: List[dict] = []
    finetuned_rows: List[dict] = []
    coverage_rows: List[Dict[str, int]] = []
    aux_problems: List[str] = []
    for ds_spec in ds_specs:
        zs, ft, cov = run_for_dataset(
            ds_spec=ds_spec,
            base_dir=Path(args.base_dir),
            raw_per_modality=raw_per_modality,
            gene_to_idx=gene_to_idx,
            checkpoint_paths=checkpoint_paths,
            sldb_pairs=sldb_pairs,
            device=args.device,
            args=args,
            out_dir=out_dir,
            modality_labels=modality_labels,
            model_num_heads=model_num_heads,
            checkpoint_sha256=checkpoint_sha256,
            aux_problems=aux_problems,
            sldb_pos=sldb_pos,
            sldb_neg=sldb_neg,
        )
        zero_shot_rows.extend(zs)
        finetuned_rows.extend(ft)
        coverage_rows.append(cov)

    # --- Persist ---
    pd.DataFrame(coverage_rows).to_csv(out_dir / "gene_coverage.csv",
                                       index=False)

    pd.DataFrame(zero_shot_rows).to_csv(out_dir / "zero_shot_metrics.csv",
                                        index=False)
    pd.DataFrame(finetuned_rows).to_csv(out_dir / "finetuned_metrics.csv",
                                        index=False)

    # a_ols rides along so the summary shows the OLS slope (its sign is what
    # separates each "ols" row from its "raw" twin); NaN on ensemble rows.
    zs_summary = summarize_metrics(zero_shot_rows,
                                   key_cols=["dataset", "calibration", "filter"],
                                   metric_cols=METRIC_COLS + ["a_ols"])
    ft_summary = summarize_metrics(finetuned_rows,
                                   key_cols=["dataset", "mode", "filter"],
                                   metric_cols=METRIC_COLS)
    zs_summary.to_csv(out_dir / "zero_shot_summary.csv", index=False)
    ft_summary.to_csv(out_dir / "finetuned_summary.csv", index=False)

    print(f"\n{'=' * 78}\nDone. Outputs under: {out_dir}\n{'=' * 78}")
    print("  config_used.json")
    print("  gene_coverage.csv")
    print("  zero_shot_metrics.csv  /  zero_shot_summary.csv")
    print("  finetuned_metrics.csv  /  finetuned_summary.csv")
    print(
        "  {dataset}_predictions.csv  /  {dataset}_finetuned_predictions_{mode}.csv"
    )
    print("  {dataset}_sl_probabilities.csv  (every pair x every head)")
    print("  {dataset}_finetuned_allpairs_{mode}.csv  (every pair, with split)")
    if args.save_finetuned:
        print(f"  {FINETUNED_DIR}/{{dataset}}_{{mode}}_fold{{k}}.pt")
    # A skipped screen (e.g. < 100 evaluable pairs) leaves no predictions CSV,
    # and sl_comparison/external_transfer.py needs every one: fail the task.
    skipped = [s["name"] for s in ds_specs
               if not (out_dir / f"{s['name']}_predictions.csv").exists()]
    if skipped:
        print(f"ERROR: no predictions written for {skipped} (see the SKIP "
              f"lines above); every screen is needed downstream.",
              file=sys.stderr)
        return 1
    if aux_problems:
        print("ERROR: optional all-pair output(s) incomplete (every table and "
              "metric above is complete):\n  " + "\n  ".join(aux_problems),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
