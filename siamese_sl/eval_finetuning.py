#!/usr/bin/env python3
"""
Evaluate a pretrained multi-modal siamese SL model against external CRISPR /
Perturb-seq screens (Adamson, Corn, Gilbert), with optional fine-tuning.

Three rows per dataset, all evaluated on the SAME 20% pair-held-out split:

  Zero-shot (calibrated)   closed-form OLS of y = a*logit + c on the 80% train.
  LP_last                  SGD MSE on (a, c) + the final projection layer.
  Full                     SGD MSE on (a, c) + the entire encoder.

The (a, c) affine head is ALWAYS present and trainable in every mode — R²
only becomes meaningful once the arbitrary pretrained-logit scale is
calibrated to the external-screen target's units. Rank metrics (Spearman,
Pearson, AUROC@1%, AUPR@1%) are invariant to the affine transform, so LP
cannot move them by construction — they act as a diagnostic for whether the
last-layer / full unfreezes learned new structure beyond pure calibration.

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
    mode's (a, c) head has been fit per fold), then compute the metric once
    on the ensembled prediction.

Outputs (under {MODEL_DIR}/eval_finetuning/)
--------------------------------------------
  config_used.json               resolved configuration snapshot
  gene_coverage.csv              per-dataset coverage + SynLethDB-overlap counts
  zero_shot_metrics.csv          long: dataset x fold x metric
  zero_shot_summary.csv          wide: dataset x metric (per-fold mean±std, ensemble)
  finetuned_metrics.csv          long: dataset x mode x fold x metric
  finetuned_summary.csv          wide: dataset x mode x metric (per-fold, ensemble)
  {dataset}_predictions.csv      per-pair zero-shot logits from all 5 folds
  {dataset}_finetuned_predictions_{mode}.csv  per-pair fine-tuned preds, 20% holdout

Each *_metrics.csv carries a column `filter` taking values `unfiltered` (all
20% test pairs) and `sldb_filtered` (the same 20% with SynLethDB-overlap
pairs removed) — provided REPORT_SLDB_FILTERED=true in the config.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def set_trainable(model: FinetunedSL, mode: str) -> None:
    """Freeze everything, then unfreeze the parameters this mode trains.

    Modes:
      LP       — (a, c) only; zero-shot closed-form uses the same freeze.
      LP_last  — (a, c) + encoder.projection (final Linear layer).
      Full     — (a, c) + entire encoder (input_bias, hidden layers, projection).

    The base model's log_temperature and scoring_bias stay frozen even in Full
    mode — SGD can absorb calibration through (a, c) directly without needing
    to perturb those pretrained scalars.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.a.requires_grad_(True)
    model.c.requires_grad_(True)

    if mode == "LP":
        return
    if mode == "LP_last":
        for p in model.base.encoder.projection.parameters():
            p.requires_grad_(True)
        return
    if mode == "Full":
        for p in model.base.encoder.parameters():
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

    Only Dropout layers are affected by train/eval mode in this model;
    LayerNorm has no running stats and behaves identically either way.

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
                 checkpoint.
      Full     — no-op; entire encoder is trainable and dropout should fire.
    """
    if mode == "LP":
        model.base.eval()
    elif mode == "LP_last":
        enc = model.base.encoder
        # Handle both encoder variants. Each has exactly one of these.
        if hasattr(enc, "hidden_blocks"):
            enc.hidden_blocks.eval()          # residual encoder
        elif hasattr(enc, "hidden"):
            enc.hidden.eval()                 # mlp / standard encoder
        # input_bias isn't a Module (just a Parameter), no need to eval()
    elif mode == "Full":
        pass
    else:
        raise ValueError(f"Unknown fine-tune mode: {mode}")


# =============================================================================
# Cell-line-conditioned (per-head) evaluation
# =============================================================================
# A SiameseSLMultiCell checkpoint emits one PD-kernel logit PER cell-line head
# (cell_line_vocab.HEADS). To evaluate it against an external screen we pick the
# head whose cell line matches that screen (e.g. K562 for Adamson / Gilbert) and
# expose ONLY that head's logit as a single-output model, so the existing
# FinetunedSL / finetune_one / compute_logits machinery is reused verbatim.


def load_multicell_model(ckpt_path: str,
                         device: str = "cpu") -> SiameseSLMultiCell:
    """Reconstruct a SiameseSLMultiCell from a checkpoint's state dict.

    Mirrors predict.load_model's encoder-shape inference but rebuilds the
    multi-head cell model (which predict.load_model deliberately refuses). The
    biological input_dim is the encoder's input WIDTH minus cell_line_dim,
    because the shared encoder sees [gene_emb || cell_emb].
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    if "cell_emb.weight" not in sd:
        raise ValueError(
            f"{ckpt_path} is not a SiameseSLMultiCell checkpoint "
            f"(no cell_emb.weight). Run without --per_head.")

    num_heads = int(sd["cell_emb.weight"].shape[0])
    cell_line_dim = int(sd["cell_emb.weight"].shape[1])

    if any("hidden_blocks" in k for k in sd):            # residual encoder
        block_keys = sorted(
            [k for k in sd if k.startswith("encoder.hidden_blocks.")
             and k.endswith(".0.weight") and sd[k].dim() == 2],
            key=lambda k: int(
                k.split("encoder.hidden_blocks.")[1].split(".")[0]))
        enc_type = "residual"
    else:                                                # mlp encoder
        block_keys = sorted(
            [k for k in sd if k.startswith("encoder.hidden.")
             and k.endswith(".weight") and sd[k].dim() == 2],
            key=lambda k: int(k.split("encoder.hidden.")[1].split(".")[0]))
        enc_type = "mlp"
    proj_key = "encoder.projection.weight"

    enc_in = int(sd[block_keys[0]].shape[1] if block_keys
                 else sd[proj_key].shape[1])             # = input_dim + cell_line_dim
    input_dim = enc_in - cell_line_dim
    encoder_dims = [int(sd[k].shape[0]) for k in block_keys]
    encoder_dims.append(int(sd[proj_key].shape[0]))
    has_bias = "encoder.projection.bias" in sd
    pd_eps = ckpt.get("config", {}).get("pd_epsilon", 0.001)

    model = SiameseSLMultiCell(
        input_dim=input_dim,
        num_heads=num_heads,
        encoder_dims=encoder_dims,
        last_layer_bias=has_bias,
        pd_epsilon=pd_eps,
        siamese_encoder_type=enc_type,
        cell_line_dim=cell_line_dim,
    )
    # Parity with predict.load_model: inject any missing input_bias as zeros.
    for key, param in model.named_parameters():
        if key not in sd and "input_bias" in key:
            sd[key] = torch.zeros_like(param)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


class _HeadSelect(nn.Module):
    """Expose ONE cell-line head of a SiameseSLMultiCell as a single-output
    model. forward(x1, x2) -> (batch, 1) = that head's PD-kernel logit.

    `.encoder` proxies the shared encoder so set_trainable / set_frozen_to_eval
    (which reach through `.base.encoder`) operate on it unchanged. The cell
    embedding, temperature and per-head bias live OUTSIDE `.encoder`, so LP /
    LP_last / Full fine-tuning never perturbs the learned cell representation —
    only the shared biological encoder (Full) and the affine head adapt.
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
        logits = self.model(x1, x2)                       # (batch, num_heads)
        return logits[:, self.head_idx:self.head_idx + 1]  # (batch, 1)


def _load_eval_base(ckpt, args, head_idx: Optional[int], device: str):
    """Per-fold base model for evaluation.

    --per_head : rebuild the multi-head cell model, expose `head_idx`.
    otherwise  : the single-output SiameseSL path (predict.load_model), which
                 still refuses cell checkpoints with a clear NotImplementedError.
    """
    if getattr(args, "per_head", False):
        mc = load_multicell_model(str(ckpt), device=device)
        return _HeadSelect(mc, head_idx).to(device)
    return load_model(str(ckpt), model_type="siamese", device=device)


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
    df = df.rename(columns={"FirstGene": "gene1_symbol",
                             "SecondGene": "gene2_symbol",
                             column: "outcome_raw"})
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
    df["gene1_symbol"] = (splits[0].str.replace("_mis", "", regex=False)
                          .str.upper())
    df["gene2_symbol"] = (splits[1].str.replace("_mis", "", regex=False)
                          .str.upper())
    df = df.rename(columns={column: "outcome_raw"})
    df = df[["gene1_symbol", "gene2_symbol", "outcome_raw"]].dropna()
    df = df[df["gene1_symbol"] != df["gene2_symbol"]].copy()
    return df


def _load_gilbert(path: str, column: str) -> pd.DataFrame:
    """Load Gilbert et al. mmc5.xlsx (Kampmann 2015 Cell paper).

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
        raise ValueError(
            f"Gilbert column {target} not found in {path}. "
            f"Available (first 10): {list(gdf.columns)[:10]}")

    # First two columns are gene names (unnamed in the header rows).
    # Drop NaN rows BEFORE stringifying so "nan" doesn't leak in as a fake
    # gene symbol (astype(str) would otherwise convert NaN to the literal
    # string "nan", inflating `pairs_raw` in the coverage report).
    out = pd.DataFrame({
        "gene1_symbol": gdf.iloc[:, 0],
        "gene2_symbol": gdf.iloc[:, 1],
        "outcome_raw": pd.to_numeric(gdf[target_col], errors="coerce"),
    }).dropna()
    out["gene1_symbol"] = out["gene1_symbol"].astype(str).str.upper()
    out["gene2_symbol"] = out["gene2_symbol"].astype(str).str.upper()
    out = out[out["gene1_symbol"] != out["gene2_symbol"]].copy()
    return out


DATASET_LOADERS: Dict[str, Callable[[str, str], pd.DataFrame]] = {
    "Adamson": _load_adamson,
    "Corn": _load_corn,
    "Gilbert": _load_gilbert,
}


# =============================================================================
# Pair preparation (symbol → Entrez, dedupe, SynLethDB overlap)
# =============================================================================


def _symbols_to_entrez(symbols: pd.Series, mapper) -> pd.Series:
    """Map each symbol to an Entrez ID; unmapped -> NaN."""
    uniq = sorted(set(symbols.dropna().astype(str)))
    out_map: Dict[str, Optional[str]] = {}
    for s in uniq:
        out_map[s] = mapper.symbol_to_entrez(s)
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
    agg = (df.groupby("pair_key", sort=False)
           .agg(outcome_raw=("outcome_raw", "mean"))
           .reset_index())
    # Restore two Entrez IDs as sorted tuple for determinism
    agg["gene1_entrez"] = agg["pair_key"].apply(lambda s: sorted(list(s))[0])
    agg["gene2_entrez"] = agg["pair_key"].apply(lambda s: sorted(list(s))[1])
    agg["idx1"] = agg["gene1_entrez"].map(gene_to_idx)
    agg["idx2"] = agg["gene2_entrez"].map(gene_to_idx)
    agg["target"] = sign * agg["outcome_raw"].astype(float)
    agg["sldb_overlap"] = agg["pair_key"].isin(sldb_pairs)

    cov["pairs_kept"] = int(len(agg))
    cov["sldb_overlap_count"] = int(agg["sldb_overlap"].sum())

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
        out[start:end] = (model.raw_logit(x1, x2)
                          .detach().cpu().numpy().astype(np.float32))
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


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Spearman, Pearson, R² + AUROC/AUPR at top-1% and bot-1% of target.

    `pred` is the model's final output (after (a, c)); `target` is the
    signed raw score (sign * outcome_raw). R² is computed as
    1 - SS_res(pred, target)/SS_tot(target) using sklearn.

    AUROC / AUPR @ top-1%: label = 1 where target >= 99th percentile.
    AUROC / AUPR @ bot-1%: label = 1 where target <=  1st percentile.
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

    if top_lab.sum() > 0 and top_lab.sum() < len(top_lab):
        m["auroc_top1pct"] = _safe_metric(roc_auc_score, top_lab, pred)
        m["aupr_top1pct"] = _safe_metric(average_precision_score, top_lab, pred)
    else:
        m["auroc_top1pct"] = float("nan")
        m["aupr_top1pct"] = float("nan")

    if bot_lab.sum() > 0 and bot_lab.sum() < len(bot_lab):
        m["auroc_bot1pct"] = _safe_metric(roc_auc_score, bot_lab, pred)
        m["aupr_bot1pct"] = _safe_metric(average_precision_score, bot_lab, pred)
    else:
        m["auroc_bot1pct"] = float("nan")
        m["aupr_bot1pct"] = float("nan")
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
    denom = ((l - lm) ** 2).sum()
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
    - Saves best-val checkpoint; returns it after training.
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

    ds = TensorDataset(train_i1, train_i2, train_t)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=g, drop_last=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
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

    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def split_pairs_cv2(
    idx1: np.ndarray, idx2: np.ndarray,
    train_frac: float, val_frac: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV2 (gene-based): test pairs have >= 1 unseen gene.

    1. Partition unique genes into train_genes (train_frac) / test_genes.
    2. Train+val: both genes in train_genes.
    3. Test: at least one gene in test_genes.
    4. val_frac of the train+val portion held out as validation.
    """
    rng = np.random.default_rng(seed)
    n = len(idx1)

    all_genes = np.unique(np.concatenate([idx1, idx2]))
    rng.shuffle(all_genes)
    n_train_genes = int(round(train_frac * len(all_genes)))
    train_genes = all_genes[:n_train_genes]

    both_train = np.isin(idx1, train_genes) & np.isin(idx2, train_genes)
    test_mask = ~both_train

    trainval_idx = np.where(both_train)[0]
    rng.shuffle(trainval_idx)
    n_val = int(round(val_frac * len(trainval_idx)))

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[trainval_idx[n_val:]] = True
    val_mask[trainval_idx[:n_val]] = True
    return train_mask, val_mask, test_mask


def split_pairs_cv3(
    idx1: np.ndarray, idx2: np.ndarray,
    train_frac: float, val_frac: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV3 (pair-based): both genes in each test pair are unseen.

    1. Partition unique genes into train_genes (train_frac) / test_genes.
    2. Train+val: both genes in train_genes.
    3. Test: both genes in test_genes.
    4. Mixed pairs (one gene in each set) are excluded from all masks.
    5. val_frac of the train+val portion held out as validation.
    """
    rng = np.random.default_rng(seed)
    n = len(idx1)

    all_genes = np.unique(np.concatenate([idx1, idx2]))
    rng.shuffle(all_genes)
    n_train_genes = int(round(train_frac * len(all_genes)))
    train_genes = all_genes[:n_train_genes]
    test_genes = all_genes[n_train_genes:]

    g1_train = np.isin(idx1, train_genes)
    g2_train = np.isin(idx2, train_genes)
    both_train = g1_train & g2_train
    both_test = np.isin(idx1, test_genes) & np.isin(idx2, test_genes)

    test_mask = both_test

    trainval_idx = np.where(both_train)[0]
    rng.shuffle(trainval_idx)
    n_val = int(round(val_frac * len(trainval_idx)))

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[trainval_idx[n_val:]] = True
    val_mask[trainval_idx[:n_val]] = True
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
    ckpt_raw = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
    transform = ckpt_raw.get("preprocessing_transform")
    if transform is not None:
        return apply_multimodal_transform(raw_per_modality, transform)

    print(f"  WARNING: checkpoint {ckpt_path.name} has no "
          f"`preprocessing_transform` — falling back to all-gene PCA fit "
          f"using --pca_variance/--post_pca_variance. This REINTRODUCES "
          f"the CV2/CV3 leakage the new pipeline fixes; retrain to remove "
          f"the warning.", file=sys.stderr)
    if fallback_pca_variance is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} lacks preprocessing_transform and no "
            f"--pca_variance was provided for fallback.")
    n_genes = raw_per_modality[0].shape[0]
    fit_mask = torch.ones(n_genes, dtype=torch.bool)
    pca_dims = [fallback_pca_variance] * len(raw_per_modality)
    legacy_transform = fit_multimodal_transform(
        raw_per_modality, fit_mask, pca_dims=pca_dims,
        post_pca=fallback_post_pca_variance, labels=modality_labels)
    return apply_multimodal_transform(raw_per_modality, legacy_transform)


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
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """Evaluate and fine-tune for one external dataset. Returns:
      zero_shot_rows, finetuned_rows, coverage_dict
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
    if getattr(args, "per_head", False):
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
        print(f"  head    : {head_name} (idx {head_idx}) [per-head cell-line eval]")

    if not path.exists():
        print(f"  SKIP: file not found. Set the dataset file under "
              f"{base_dir}/external\\ data/ to include it.")
        return [], [], {"name": name, "status": "file_not_found"}

    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"No loader registered for dataset '{name}'. "
                         f"Supported: {list(DATASET_LOADERS.keys())}")

    df_raw = loader(str(path), col)
    df, cov = prepare_dataset(df_raw, gene_to_idx, sldb_pairs, sign)
    cov["name"] = name
    print(f"  coverage: raw={cov['pairs_raw']:,}  both_mapped="
          f"{cov['pairs_both_mapped']:,}  both_in_model="
          f"{cov['pairs_both_in_model']:,}  kept={cov['pairs_kept']:,}  "
          f"sldb_overlap={cov['sldb_overlap_count']:,} "
          f"({100.0 * cov['sldb_overlap_count'] / max(1, cov['pairs_kept']):.2f}%)")

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
        train_mask, val_mask, test_mask = split_pairs(
            len(df), args.train_frac, args.val_frac, args.split_seed)

    n_used = int(train_mask.sum() + val_mask.sum() + test_mask.sum())
    n_discarded = len(df) - n_used
    print(f"  split({cv}): train={train_mask.sum():,}  val={val_mask.sum():,}  "
          f"test={test_mask.sum():,}"
          + (f"  discarded(mixed)={n_discarded:,}" if n_discarded else ""))

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

    for k, ckpt in enumerate(checkpoint_paths):
        print(f"  [fold {k}] loading checkpoint {ckpt.name}")
        # Rebuild this fold's preprocessed embedding matrix from its own
        # saved transform. Each fold uses a DIFFERENT feature space
        # (preprocessing was fit on that fold's training-pair genes), so we
        # cannot share a single `embeddings` tensor across folds anymore.
        fold_embeddings = _load_fold_transform_and_embeddings(
            ckpt, raw_per_modality,
            fallback_pca_variance=args.pca_variance,
            fallback_post_pca_variance=args.post_pca_variance,
            modality_labels=modality_labels,
        )
        base = _load_eval_base(ckpt, args, head_idx, device)
        # Sanity: encoder.input_bias is (input_width,). For a per-head cell
        # model input_width = biological_dim + cell_line_dim, so subtract
        # cell_line_dim before comparing to the fold's biological embedding dim.
        input_width = int(base.encoder.input_bias.shape[0])
        expected_dim = (input_width - base.cell_line_dim
                        if getattr(args, "per_head", False) else input_width)
        if expected_dim != fold_embeddings.shape[1]:
            raise ValueError(
                f"fold {k}: embedding dim {fold_embeddings.shape[1]} does "
                f"not match checkpoint's expected input_dim {expected_dim}. "
                f"The saved preprocessing_transform is out of sync with the "
                f"model weights.")

        model = FinetunedSL(base).to(device)

        raw = compute_logits(model, fold_embeddings, idx1, idx2, device)
        raw_logits_folds[k] = raw

        a, c = fit_ols_affine(raw[train_mask], target[train_mask])
        zshot_preds_folds[k] = a * raw + c

        for flt_name, flt_mask in test_filters:
            if flt_mask.sum() < 10:
                continue
            preds = zshot_preds_folds[k][flt_mask]
            tgts = target[flt_mask]
            m = compute_metrics(preds, tgts)
            zero_shot_rows.append({
                "dataset": name, "fold": k, "filter": flt_name,
                "n_test": int(flt_mask.sum()),
                "a_ols": float(a), "c_ols": float(c),
                **m,
            })

        print(f"    zero-shot OLS: a={a:+.4f}  c={c:+.4f}")

        # ---- Fine-tune (per mode, per fold) — warm-start (a, c) from OLS ----
        for mode in args.ft_modes:
            # Reload fresh base for each mode so folds/modes don't contaminate.
            base_ft = _load_eval_base(ckpt, args, head_idx, device)
            model_ft = FinetunedSL(base_ft).to(device)
            model_ft, best_epoch = finetune_one(
                model_ft, fold_embeddings, idx1, idx2, target,
                train_mask=train_mask, val_mask=val_mask,
                mode=mode, epochs=args.ft_epochs,
                patience=args.ft_patience, lr=args.ft_lr,
                batch_size=args.ft_batch_size,
                weight_decay=args.ft_weight_decay,
                device=device,
                warm_start=(a, c),
                # Per-fold seed so the SGD trajectories of the 5 folds are
                # independent draws — important for the ensemble path, which
                # averages logits across folds. Using a constant split_seed
                # made every fold's SGD state identical at init, defeating
                # ensemble diversity. The split itself still uses split_seed.
                seed=args.split_seed + k,
            )
            preds_all = predict_finetuned(
                model_ft, fold_embeddings, idx1, idx2, device)
            ft_preds_folds[mode][k] = preds_all
            best_epoch_tracker[(k, mode)] = int(best_epoch)

            for flt_name, flt_mask in test_filters:
                if flt_mask.sum() < 10:
                    continue
                preds = preds_all[flt_mask]
                tgts = target[flt_mask]
                m = compute_metrics(preds, tgts)
                finetuned_rows.append({
                    "dataset": name, "mode": mode, "fold": k,
                    "filter": flt_name, "n_test": int(flt_mask.sum()),
                    "best_epoch": int(best_epoch),
                    **m,
                })

    # ---- Ensemble rows: mean of per-fold predictions ----
    # Zero-shot ensemble
    zshot_ensemble_pred = zshot_preds_folds.mean(axis=0)
    for flt_name, flt_mask in test_filters:
        if flt_mask.sum() < 10:
            continue
        m = compute_metrics(zshot_ensemble_pred[flt_mask], target[flt_mask])
        zero_shot_rows.append({
            "dataset": name, "fold": "ensemble", "filter": flt_name,
            "n_test": int(flt_mask.sum()),
            "a_ols": float("nan"), "c_ols": float("nan"),
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
                "dataset": name, "mode": mode, "fold": "ensemble",
                "filter": flt_name, "n_test": int(flt_mask.sum()),
                "best_epoch": -1,  # not meaningful for the ensemble row
                **m,
            })

    # ---- Save per-dataset zero-shot predictions (all pairs) ----
    pred_cols = {f"logit_fold{k}": raw_logits_folds[k]
                 for k in range(n_folds)}
    pred_cols["logit_mean"] = raw_logits_folds.mean(axis=0)
    zshot_df = pd.DataFrame({
        "gene1_entrez": df["gene1_entrez"],
        "gene2_entrez": df["gene2_entrez"],
        "outcome_raw": outcome_raw,
        "target": target,
        "sldb_overlap": overlap_mask,
        "split": np.where(train_mask, "train",
                          np.where(val_mask, "val",
                                   np.where(test_mask, "test", "discarded"))),
        **pred_cols,
    })
    zshot_df.to_csv(out_dir / f"{name}_predictions.csv", index=False)

    # ---- Save per-dataset fine-tuned predictions (20% test only, one CSV per mode) ----
    for mode in args.ft_modes:
        ft_pred_cols = {f"pred_fold{k}": ft_preds_folds[mode][k][test_mask]
                        for k in range(n_folds)}
        ft_pred_cols["pred_mean"] = ft_preds_folds[mode].mean(axis=0)[test_mask]
        ft_df = pd.DataFrame({
            "gene1_entrez": df["gene1_entrez"].to_numpy()[test_mask],
            "gene2_entrez": df["gene2_entrez"].to_numpy()[test_mask],
            "outcome_raw": outcome_raw[test_mask],
            "target": target[test_mask],
            "sldb_overlap": overlap_mask[test_mask],
            **ft_pred_cols,
        })
        ft_df.to_csv(out_dir / f"{name}_finetuned_predictions_{mode}.csv",
                     index=False)

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
    "spearman":      "Spearman",
    "pearson":       "Pearson",
    "r2":            "R²",
    "auroc_top1pct": "AUROC@top1%",
    "aupr_top1pct":  "AUPR@top1%",
    "auroc_bot1pct": "AUROC@bot1%",
    "aupr_bot1pct":  "AUPR@bot1%",
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
    and the single ensemble row. Zero-shot is always shown; fine-tuned
    modes appear in `ft_modes` order. Columns are the 7 metrics.
    """
    col_w = 15
    filters = ["unfiltered"]
    if report_sldb_filtered:
        filters.append("sldb_filtered")

    print()
    print("=" * 80)
    print(f"RESULTS: {name}")
    print("=" * 80)

    for flt in filters:
        zs = [r for r in zero_shot_rows if r["filter"] == flt]
        ft_by_mode = {m: [r for r in finetuned_rows
                          if r["filter"] == flt and r["mode"] == m]
                      for m in ft_modes}

        if not zs and not any(ft_by_mode.values()):
            continue  # nothing to show (e.g., too few pairs after filter)

        # N_test is fold-independent and filter-specific.
        sample_row = zs[0] if zs else next(r for rs in ft_by_mode.values()
                                           if rs for r in rs)
        n_test = sample_row["n_test"]

        print(f"\nFilter: {flt}  (N = {n_test:,} test pairs)")

        # Header
        label_hdr = f"  {'Mode':<11} {'Agg':<9}  "
        metric_hdr = "  ".join(
            f"{_METRIC_LABELS.get(m, m):>{col_w}s}" for m in metric_cols)
        print(label_hdr + metric_hdr)
        print("  " + "-" * (len(label_hdr) - 2 + len(metric_hdr)))

        blocks: List[Tuple[str, List[dict]]] = [("Zero-shot", zs)]
        for m in ft_modes:
            blocks.append((m, ft_by_mode[m]))

        for block_name, rows in blocks:
            if not rows:
                continue
            perfold = [r for r in rows if r["fold"] != "ensemble"]
            ensemble = [r for r in rows if r["fold"] == "ensemble"]

            if perfold:
                means = [float(np.nanmean([r[m] for r in perfold]))
                         for m in metric_cols]
                # ddof=1 (sample std) matches pandas groupby().agg("std")
                # default, so stdout numbers match the summary CSV exactly.
                stds = [float(np.nanstd([r[m] for r in perfold], ddof=1))
                        for m in metric_cols]
                cells = "  ".join(
                    f"{mn:+.4f}±{sd:.3f}".rjust(col_w)
                    for mn, sd in zip(means, stds))
                print(f"  {block_name:<11} {'mean±std':<9}  {cells}")
            if ensemble:
                cells = "  ".join(
                    f"{ensemble[0][m]:+.4f}".rjust(col_w)
                    for m in metric_cols)
                print(f"  {block_name:<11} {'ensemble':<9}  {cells}")

    # Footnote: explain the @bot1% convention. Target = sign * raw_score with
    # sign = -1 puts SL-like pairs at HIGH target. @top1% (high-target tail)
    # is the SL-aligned headline metric; @bot1% (low-target tail) treats the
    # NON-SL tail as positive class, so a well-calibrated model will score
    # sub-0.5 AUROC there. Kept for diagnostic symmetry with the CIHR figure,
    # not as a second headline.
    print()
    print("  Note: @top1% metrics are the SL-aligned headline "
          "(higher = better ranking of SL-like pairs).")
    print("        @bot1% metrics are diagnostic — well-calibrated models "
          "typically score < 0.5 there.")


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
        agg = (perfold.groupby(key_cols)[metric_cols]
               .agg(["mean", "std"])
               .reset_index())
        # Flatten MultiIndex columns
        agg.columns = (list(agg.columns[:len(key_cols)].get_level_values(0))
                       + [f"{m}_{s}"
                          for m in metric_cols
                          for s in ("mean", "std")])
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
    "spearman", "pearson", "r2",
    "auroc_top1pct", "aupr_top1pct",
    "auroc_bot1pct", "aupr_bot1pct",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate / fine-tune a pretrained siamese SL model on "
                    "external CRISPR screens (Adamson / Corn / Gilbert).")
    p.add_argument("--model_dir", required=True,
                   help="Pretrained combo results dir (must contain "
                        "config.json and checkpoints/fold_*_best.pt).")
    p.add_argument("--embeddings_paths", nargs="+", required=True,
                   help="Embedding .pt files matching the pretrained combo.")
    p.add_argument("--pca_variance", type=float, default=None,
                   help="Legacy fallback only: per-modality ROBPCA variance. "
                        "Used ONLY when a checkpoint predates per-fold "
                        "`preprocessing_transform` storage. New checkpoints "
                        "carry their own transform and ignore this flag.")
    p.add_argument("--post_pca_variance", type=float, default=None,
                   help="Legacy fallback only: post-concat ROBPCA variance. "
                        "See --pca_variance note.")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Spec rows 'name:file:column:sign:format[:head]' "
                        "(the optional :head is the cell-line head used by "
                        "--per_head).")
    p.add_argument("--sl_path", required=True,
                   help="SynLethDB .txt for overlap detection.")
    p.add_argument("--base_dir", required=True,
                   help="Project base dir; dataset 'file' entries are "
                        "resolved relative to this.")
    p.add_argument("--out_dir", default=None,
                   help="Output dir (default: <model_dir>/eval_finetuning).")
    p.add_argument("--ft_modes", nargs="+", default=["LP_last", "Full"],
                   choices=["LP", "LP_last", "Full"])
    p.add_argument("--ft_epochs", type=int, default=100)
    p.add_argument("--ft_patience", type=int, default=10)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--ft_batch_size", type=int, default=256)
    p.add_argument("--ft_weight_decay", type=float, default=0.0)
    p.add_argument("--cv_type", default="cv1",
                   choices=["cv1", "cv2", "cv3"],
                   help="CV split strategy for external data: "
                        "cv1=edge-based (random), "
                        "cv2=gene-based (>=1 unseen gene), "
                        "cv3=pair-based (both genes unseen).")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--report_sldb_filtered", action="store_true",
                   help="Also report metrics with SynLethDB-overlap pairs "
                        "removed from the test split.")
    p.add_argument("--per_head", action="store_true",
                   help="Evaluate a cell-line-conditioned SiameseSLMultiCell "
                        "checkpoint: score each dataset against the model head "
                        "named in its 6th DATASETS field "
                        "(name:file:column:sign:format:HEAD, e.g. K562). "
                        "Without it, cell checkpoints are refused as before.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                                else "cpu")
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
    checkpoint_paths = sorted(
        [p for p in ckpt_dir.glob("fold_*_best.pt")],
        key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoint_paths:
        print(f"ERROR: no fold checkpoints found under {ckpt_dir}",
              file=sys.stderr)
        return 1
    print(f"Loaded {len(checkpoint_paths)} fold checkpoints from {ckpt_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        model_dir / f"eval_finetuning_{args.cv_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load RAW embeddings once; each fold applies its own transform ---
    n_mod = len(args.embeddings_paths)
    print(f"\nLoading RAW embeddings ({n_mod} modalities; per-fold "
          f"preprocessing transforms will be pulled from each checkpoint).")
    raw_per_modality, gene_to_idx, idx_to_gene = load_raw_multimodal(
        args.embeddings_paths)
    modality_labels = [Path(p).stem.replace("all_genes_", "")
                       for p in args.embeddings_paths]
    print(f"  raw shapes: "
          f"{[tuple(t.shape) for t in raw_per_modality]}  "
          f"genes: {len(gene_to_idx):,}")

    # --- Load SynLethDB positives for overlap detection ---
    sldb_pairs = _load_synlethdb_pairs(args.sl_path)
    print(f"SynLethDB positives loaded: {len(sldb_pairs):,} pairs")

    # --- Resolved config snapshot ---
    resolved = {
        "model_dir": str(model_dir),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "embeddings_paths": args.embeddings_paths,
        "cv_type": args.cv_type,
        "pca_variance_fallback": args.pca_variance,
        "post_pca_variance_fallback": args.post_pca_variance,
        "input_dim_pretrained_config": train_cfg.get("input_dim"),
        "datasets_spec": args.datasets,
        "ft_modes": args.ft_modes,
        "ft_epochs": args.ft_epochs,
        "ft_patience": args.ft_patience,
        "ft_lr": args.ft_lr,
        "ft_batch_size": args.ft_batch_size,
        "ft_weight_decay": args.ft_weight_decay,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "split_seed": args.split_seed,
        "report_sldb_filtered": bool(args.report_sldb_filtered),
        "per_head": bool(args.per_head),
        "dataset_heads": {s["name"]: s.get("head")
                          for s in parse_datasets_spec(args.datasets)},
        "device": args.device,
        "n_genes_in_model": len(gene_to_idx),
        "preprocessing": "per-fold (leakage-free) from checkpoint transforms",
    }
    with open(out_dir / "config_used.json", "w") as f:
        json.dump(resolved, f, indent=2)

    # --- Run per dataset ---
    ds_specs = parse_datasets_spec(args.datasets)
    zero_shot_rows: List[dict] = []
    finetuned_rows: List[dict] = []
    coverage_rows: List[Dict[str, int]] = []
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
        )
        zero_shot_rows.extend(zs)
        finetuned_rows.extend(ft)
        coverage_rows.append(cov)

    # --- Persist ---
    pd.DataFrame(coverage_rows).to_csv(out_dir / "gene_coverage.csv", index=False)

    pd.DataFrame(zero_shot_rows).to_csv(
        out_dir / "zero_shot_metrics.csv", index=False)
    pd.DataFrame(finetuned_rows).to_csv(
        out_dir / "finetuned_metrics.csv", index=False)

    zs_summary = summarize_metrics(
        zero_shot_rows, key_cols=["dataset", "filter"],
        metric_cols=METRIC_COLS)
    ft_summary = summarize_metrics(
        finetuned_rows, key_cols=["dataset", "mode", "filter"],
        metric_cols=METRIC_COLS)
    zs_summary.to_csv(out_dir / "zero_shot_summary.csv", index=False)
    ft_summary.to_csv(out_dir / "finetuned_summary.csv", index=False)

    print(f"\n{'=' * 78}\nDone. Outputs under: {out_dir}\n{'=' * 78}")
    print("  config_used.json")
    print("  gene_coverage.csv")
    print("  zero_shot_metrics.csv  /  zero_shot_summary.csv")
    print("  finetuned_metrics.csv  /  finetuned_summary.csv")
    print("  {dataset}_predictions.csv  /  {dataset}_finetuned_predictions_{mode}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
