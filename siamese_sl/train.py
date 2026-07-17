#!/usr/bin/env python3
"""
Training script for Siamese SL prediction model.

All embeddings go through the same pipeline per modality:
  impute (Huber) → robust PCA (optional) → normalize (median center, MAD scale)
then concatenate along the feature axis, followed by an optional post-concat
robust PCA + re-normalize (on by default, --no_post_pca to disable).

Usage:
    # Single embedding
    python train.py --embeddings_paths ../data/all_genes_go.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --output_dir results/siamese_go

    # Multi-modal (concatenated)
    python train.py --embeddings_paths ../data/all_genes_bioconceptvec.pt \
                        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --output_dir results/multi_bio_go_ppi
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from tqdm import tqdm

from siamese_esm import (
    SiameseSL,
    SiameseSLMultiCell,
    SiameseSLWithAttention,
    SiameseSLKernel,
    set_seed,
)
from data_loader import SLDataManager, create_fold_dataloaders
import cell_line_vocab


def calculate_optimal_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate optimal F1 from precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return float(np.max(f1_scores))


def _normalized_aupr(aupr: float, baseline: float) -> float:
    """Baseline-normalized AUPR (a.k.a. AUPR skill score).

    AUPR's random-classifier baseline equals the positive prevalence pi, so raw
    AUPR is not comparable across heads / runs with different pi. This rescales
    it to [random -> 0, perfect -> 1]:

        aupr_norm = (aupr - pi) / (1 - pi)

    A value < 0 means worse-than-random (kept unclipped, since that is
    informative). Returns NaN when pi is undefined (pi >= 1, i.e. no negatives)
    or when aupr itself is NaN. NOTE: this min-max skill rescaling is a common
    and transparent convention but is not as canonical as AUROC; the formally
    established baseline-free PR metric is AUPRG (Flach & Kull, 2015).
    """
    if (aupr is None or np.isnan(aupr) or baseline is None
            or np.isnan(baseline) or baseline >= 1.0):
        return float("nan")
    return float((aupr - baseline) / (1.0 - baseline))


def _auprg(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the Precision-Recall-Gain curve (Flach & Kull, NeurIPS 2015).

    The principled baseline-free counterpart of AUPR: precision and recall are
    replaced by their *gain* versions relative to the always-positive baseline
    (prevalence pi), so a random classifier scores ~0 and the metric is
    comparable across heads / runs with different pi. Unlike AUPR it is not
    dominated by pi.

    Uses the reference `prg` package when installed; otherwise falls back to a
    self-contained implementation that integrates precision-gain over
    recall-gain on [0, 1] (interpolating the recall-gain = 0 crossing).
    Returns NaN when pi is degenerate (no positives or no negatives).
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pi = float(np.mean(labels)) if labels.size else float("nan")
    if not (0.0 < pi < 1.0) or len(np.unique(labels)) < 2:
        return float("nan")

    try:                                   # canonical reference implementation
        import prg
        return float(prg.calc_auprg(prg.create_prg_curve(labels, scores)))
    except Exception:
        pass

    precision, recall, _ = precision_recall_curve(labels, scores)
    odds = pi / (1.0 - pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - odds * (1.0 - precision) / precision
        rec_gain = 1.0 - odds * (1.0 - recall) / recall
    valid = recall > 0                     # drop the terminal recall=0 point
    rg_v, pg_v = rec_gain[valid], prec_gain[valid]
    # Collapse duplicate recall-gain values (vertical PR-gain segments) to their
    # upper envelope: at each recall-gain keep the max precision-gain reached.
    # sklearn emits many points at recall=1.0 (adding FPs at full recall), all
    # of which map to recall_gain=1.0; a plain argsort orders those ties
    # arbitrarily, so the trapezoid up to recall_gain=1.0 can descend to a
    # low-precision tie and a perfect ranker returns 0.875 instead of 1.0 (and
    # the result becomes numpy/platform tie-order dependent). np.unique also
    # returns the recall-gains sorted ascending, giving a clean L->R trace.
    rg = np.unique(rg_v)
    pg = np.array([pg_v[rg_v == u].max() for u in rg])

    def _interp(r, r0, r1, p0, p1):
        if r1 == r0:
            return p1
        return p0 + (r - r0) / (r1 - r0) * (p1 - p0)

    area = 0.0
    for i in range(1, len(rg)):
        r0, r1, p0, p1 = rg[i - 1], rg[i], pg[i - 1], pg[i]
        lo, hi = max(0.0, min(r0, r1)), min(1.0, max(r0, r1))
        if hi <= lo:                       # segment outside [0, 1] recall-gain
            continue
        area += (hi - lo) * (_interp(lo, r0, r1, p0, p1)
                             + _interp(hi, r0, r1, p0, p1)) / 2.0
    return float(area)


class Trainer:
    """Trainer for Siamese SL model."""

    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        # Device setup
        if torch.cuda.is_available() and not args.cpu:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and not args.cpu:
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Output directory (must already exist — server does not allow mkdir)
        self.output_dir = Path(args.output_dir)
        ckpt_dir = self.output_dir / "checkpoints"
        if not self.output_dir.is_dir() or not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Output dirs must be pre-created:\n"
                                    f"  {self.output_dir}\n  {ckpt_dir}")

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        # Load RAW embeddings (no preprocessing). Preprocessing is fit
        # PER FOLD on training-pair genes (excluding test-only genes) to
        # prevent CV2/CV3 leakage of test-gene embeddings into the PCA /
        # normalization basis. See SLDataManager.get_fold_embeddings.
        # --preprocessing_fit_scope flips this to all-gene fit for A/B
        # comparison with the legacy leaky pipeline.
        self.data_manager = SLDataManager(
            embeddings_paths=args.embeddings_paths,
            sl_pairs_path=args.sl_path,
            neg_pairs_path=args.neg_pairs_path,
            gene_list_path=args.gene_list_path,
            seed=args.seed,
            pca_dims=args.pca_dims,
            post_pca=args.post_pca,
            preprocessing_fit_scope=args.preprocessing_fit_scope,
            pca_method=args.pca_method,
            use_cell_lines=args.use_cell_lines,
        )

        # input_dim is resolved per-fold once that fold's preprocessing is
        # fit (variance-target PCA may pick a different k per fold).
        if args.input_dim is not None:
            print(f"Warning: --input_dim={args.input_dim} is ignored; "
                  f"per-fold input_dim is detected from the fold's "
                  f"preprocessed embedding matrix.")
        args.input_dim = None

    def create_model(self) -> nn.Module:
        """Create the model."""
        if self.args.use_cell_lines:
            encoder_dims = self.args.encoder_dims or [
                self.args.hidden_dim, self.args.latent_dim
            ]
            model = SiameseSLMultiCell(
                input_dim=self.args.input_dim,
                num_heads=cell_line_vocab.NUM_HEADS,
                encoder_dims=encoder_dims,
                dropout=self.args.dropout,
                last_layer_bias=self.args.last_layer_bias,
                pd_epsilon=self.args.pd_epsilon,
                siamese_encoder_type=self.args.siamese_encoder_type,
                cell_line_dim=self.args.cell_line_dim,
            )
            return model.to(self.device)
        if self.args.model_type == "attention":
            model = SiameseSLWithAttention(
                input_dim=self.args.input_dim,
                hidden_dim=self.args.hidden_dim,
                latent_dim=self.args.latent_dim,
                num_heads=self.args.num_heads,
                dropout=self.args.dropout,
            )
        elif self.args.model_type == "kernel":
            model = SiameseSLKernel(
                input_dim=self.args.input_dim,
                hidden_dim=self.args.hidden_dim,
                latent_dim=self.args.latent_dim,
                rff_features=self.args.rff_features,
                bilinear_rank=self.args.bilinear_rank,
                predictor_hidden=self.args.predictor_hidden,
                dropout=self.args.dropout,
                sigma=self.args.kernel_sigma,
                encoder_type=self.args.encoder_type,
                encoder_rank=self.args.encoder_rank,
            )
        else:
            encoder_dims = self.args.encoder_dims or [
                self.args.hidden_dim, self.args.latent_dim
            ]
            model = SiameseSL(
                input_dim=self.args.input_dim,
                encoder_dims=encoder_dims,
                dropout=self.args.dropout,
                last_layer_bias=self.args.last_layer_bias,
                pd_epsilon=self.args.pd_epsilon,
                siamese_encoder_type=self.args.siamese_encoder_type,
            )
        return model.to(self.device)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if self.args.use_cell_lines:
                # batch = (x1, x2, label_vec(H), mask_vec(H))
                x1, x2, labels, mask = batch
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.to(self.device)   # (B, H)
                mask = mask.to(self.device)        # (B, H)
                logits = model(x1, x2)             # (B, H)
                per = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none")  # (B, H); no pos_weight
                # Masked mean over supervised (pair, head) cells — the only
                # weighting (each observed cell weighted equally).
                loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                x1, x2, labels = batch
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                logits = model(x1, x2)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Proximal L1: per-layer soft-thresholding outside autograd
            # Each weight matrix (dim >= 2) gets its own lambda from --l1_lambdas
            # Biases and LayerNorm params (dim < 2) are never penalized
            if self._l1_map:
                current_lr = optimizer.param_groups[0]["lr"]
                eps = optimizer.defaults.get("eps", 1e-8)
                beta2 = optimizer.defaults["betas"][1]
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name not in self._l1_map:
                            continue
                        lam = self._l1_map[name]
                        if lam <= 0:
                            continue
                        state = optimizer.state.get(param, {})
                        if "exp_avg_sq" in state:
                            step = state["step"]
                            if isinstance(step, torch.Tensor):
                                step = step.item()
                            v_hat = state["exp_avg_sq"] / (1 - beta2**step)
                            denom = torch.clamp(v_hat.sqrt(), min=1e-1) + eps
                            thresh = lam * current_lr / denom
                        else:
                            thresh = lam * current_lr
                        param.data = torch.sign(param.data) * torch.clamp(
                            param.data.abs() - thresh, min=0)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader,
    ) -> dict:
        """Evaluate model on test set."""
        if self.args.use_cell_lines:
            return self._evaluate_multilabel(model, test_loader)
        model.eval()
        all_labels = []
        all_scores = []

        for x1, x2, labels in test_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)

            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_labels.extend(labels.numpy())
            all_scores.extend(probs)

        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        # Guard against single-class test set (can happen in edge cases)
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            print(
                f"Warning: Single-class test set (only class {unique_labels[0]}), metrics undefined"
            )
            return {
                "auroc": float("nan"),
                "aupr": float("nan"),
                "aupr_baseline": float("nan"),
                "aupr_norm": float("nan"),
                "auprg": float("nan"),
                "f1": float("nan"),
            }

        # Compute metrics
        auroc = roc_auc_score(all_labels, all_scores)
        aupr = average_precision_score(all_labels, all_scores)
        baseline = float(np.mean(all_labels))          # positive prevalence (pi)
        aupr_norm = _normalized_aupr(aupr, baseline)   # skill score: random -> 0
        auprg = _auprg(all_labels, all_scores)         # baseline-free (Flach&Kull)
        f1 = calculate_optimal_f1(all_labels, all_scores)

        return {
            "auroc": auroc,
            "aupr": aupr,
            "aupr_baseline": baseline,
            "aupr_norm": aupr_norm,
            "auprg": auprg,
            "f1": f1,
        }

    @torch.no_grad()
    def _evaluate_multilabel(self, model: nn.Module, test_loader) -> dict:
        """Cell-line-mode evaluation, three views.

        PRIMARY = STRATIFIED: pool every observed (pair, cell_line) cell into
        one long set and compute AUROC / AUPR / AUPRG / F1 on it. Stratified
        AUPRG is the model-selection metric.
        GLOBAL (`*_any`): one-time headline — collapse the head axis per pair
        (label = SL in any line; score = max over heads).
        PER-CELL-LINE (`*_macro`, `per_head_*`): equal-weight-per-line
        diagnostic. Heads with <2 classes this fold are skipped.
        """
        model.eval()
        logits_all, labels_all, masks_all = [], [], []
        for x1, x2, labels, mask in test_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            logits = model(x1, x2)  # (B, H)
            logits_all.append(logits.cpu())
            labels_all.append(labels)
            masks_all.append(mask)
        L = torch.cat(logits_all).numpy()
        Y = torch.cat(labels_all).numpy()
        M = torch.cat(masks_all).numpy()
        P = 1.0 / (1.0 + np.exp(-L))  # sigmoid -> per-head probabilities

        per_head_aupr, per_head_auroc, per_head_f1 = {}, [], []
        per_head_baseline, per_head_aupr_norm, per_head_auprg = {}, {}, {}
        aupr_list = []            # raw per-head AUPRs (macro-averaged for report)
        aupr_norm_list = []       # normalized per-head AUPRs (reported)
        auprg_list = []           # per-head AUPRG (macro diagnostic)
        baseline_list = []        # per-head prevalence of scored heads
        pooled_y, pooled_s = [], []
        for h, name in enumerate(cell_line_vocab.HEADS):
            sel = M[:, h] == 1
            if sel.sum() == 0:
                per_head_aupr[name] = float("nan")
                per_head_baseline[name] = float("nan")
                per_head_aupr_norm[name] = float("nan")
                per_head_auprg[name] = float("nan")
                continue
            yh, sh = Y[sel, h], P[sel, h]
            pooled_y.extend(yh.tolist())
            pooled_s.extend(sh.tolist())
            pi = float(np.mean(yh))            # this head's positive prevalence
            per_head_baseline[name] = pi
            if len(np.unique(yh)) < 2:
                per_head_aupr[name] = float("nan")
                per_head_aupr_norm[name] = float("nan")
                per_head_auprg[name] = float("nan")
                continue
            ap = average_precision_score(yh, sh)
            ap_norm = _normalized_aupr(ap, pi)
            ap_g = _auprg(yh, sh)
            per_head_aupr[name] = float(ap)
            per_head_aupr_norm[name] = ap_norm
            per_head_auprg[name] = ap_g
            aupr_list.append(ap)
            baseline_list.append(pi)
            if not np.isnan(ap_norm):
                aupr_norm_list.append(ap_norm)
            if not np.isnan(ap_g):
                auprg_list.append(ap_g)
            per_head_auroc.append(roc_auc_score(yh, sh))
            per_head_f1.append(calculate_optimal_f1(yh, sh))

        pooled_y = np.array(pooled_y)
        pooled_s = np.array(pooled_s)

        # --- STRATIFIED (PRIMARY): pool every observed (pair, cell_line) cell -
        # The "cell_line x N observations" long form — each supervised
        # (pair, head) cell is one row. Selection AND the headline run on this
        # pooled set. Prevalence-weighted (dense lines like K562 dominate); the
        # per-cell-line macro below is the equal-weight-per-line complement.
        if len(np.unique(pooled_y)) >= 2:
            strat_aupr = float(average_precision_score(pooled_y, pooled_s))
            strat_auroc = float(roc_auc_score(pooled_y, pooled_s))
            strat_f1 = float(calculate_optimal_f1(pooled_y, pooled_s))
        else:
            strat_aupr = strat_auroc = strat_f1 = float("nan")
        strat_baseline = (float(np.mean(pooled_y)) if pooled_y.size
                          else float("nan"))
        strat_aupr_norm = _normalized_aupr(strat_aupr, strat_baseline)
        strat_auprg = _auprg(pooled_y, pooled_s)   # <- selection metric

        # --- GLOBAL (one-time headline): collapse the head axis per pair ------
        # "SL in ANY cell line": one row per pair.
        #   label = 1 if the pair is SL in any supervised line, else 0 (a pair
        #           both SL and non-SL across lines counts positive).
        #   score = max over ALL heads of the SL probability.
        sup = M == 1
        has_pos = (sup & (Y == 1)).any(axis=1)
        has_neg = (sup & (Y == 0)).any(axis=1)
        agg_keep = has_pos | has_neg          # every pair has >=1 supervised head
        agg_y = has_pos[agg_keep].astype(np.float32)
        agg_s = P[agg_keep].max(axis=1)       # max over all heads -> one score/pair
        baseline_any = float(np.mean(agg_y)) if agg_y.size else float("nan")
        if len(np.unique(agg_y)) >= 2:
            aupr_any = float(average_precision_score(agg_y, agg_s))
            auroc_any = float(roc_auc_score(agg_y, agg_s))
        else:
            aupr_any = auroc_any = float("nan")
        aupr_any_norm = _normalized_aupr(aupr_any, baseline_any)
        auprg_any = _auprg(agg_y, agg_s)

        # --- PER-CELL-LINE macro (diagnostic; equal weight per line) ---------
        macro = lambda xs: float(np.mean(xs)) if xs else float("nan")
        return {
            # PRIMARY = stratified pooled (drives selection + headline).
            "auroc": strat_auroc,
            "aupr": strat_aupr,
            "auprg": strat_auprg,             # stratified AUPRG -> selection
            "aupr_norm": strat_aupr_norm,
            "aupr_baseline": strat_baseline,
            "f1": strat_f1,
            # GLOBAL one-time headline (SL in any cell line, max over heads).
            "auroc_any": auroc_any,
            "aupr_any": aupr_any,
            "auprg_any": auprg_any,
            "aupr_any_norm": aupr_any_norm,
            "baseline_any": baseline_any,
            # PER-CELL-LINE macro + breakdown (diagnostic, equal weight/line).
            "auroc_macro": macro(per_head_auroc),
            "aupr_macro": macro(aupr_list),
            "auprg_macro": macro(auprg_list),
            "aupr_norm_macro": macro(aupr_norm_list),
            "aupr_baseline_macro": macro(baseline_list),
            "f1_macro": macro(per_head_f1),
            "per_head_aupr": per_head_aupr,
            "per_head_aupr_norm": per_head_aupr_norm,
            "per_head_auprg": per_head_auprg,
            "per_head_baseline": per_head_baseline,
            "n_heads_scored": len(aupr_list),
        }

    def train_fold(self, fold_data: dict) -> dict:
        """Train a single CV fold."""
        fold_idx = fold_data["fold"]
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}")
        print(f"{'='*60}")

        # Fit preprocessing on THIS fold's training-pair genes, apply to all.
        # Returns the per-fold embedding matrix + the transform we must save
        # alongside the checkpoint so inference can rebuild the same feature
        # space from the raw .pt files.
        fold_embeddings, fold_transform = \
            self.data_manager.get_fold_embeddings(fold_data)
        self.args.input_dim = fold_embeddings.shape[1]
        print(f"Fold {fold_idx + 1} input_dim: {self.args.input_dim}")

        # Create dataloaders
        train_loader, test_loader = create_fold_dataloaders(
            embeddings=fold_embeddings,
            fold_data=fold_data,
            batch_size=self.args.batch_size,
        )

        # Create model and optimizer
        model = self.create_model()

        # No pos_weight anywhere (by design): class imbalance is left to the
        # masked-mean BCE and the rank metrics (AUPR/AUPRG). The masked mean is
        # the only weighting.

        # Build per-layer L1 lambda map: param_name -> lambda
        # Only weight matrices (dim >= 2) are penalized. Cell-line embedding
        # params (cell_emb) are categorical context, not feature-selection
        # weights — excluded so --l1_lambdas keeps matching the encoder.
        self._l1_map = {}
        if self.args.l1_lambdas:
            weight_params = [(n, p) for n, p in model.named_parameters()
                             if p.dim() >= 2 and not n.startswith("cell_")]
            if len(self.args.l1_lambdas) != len(weight_params):
                raise ValueError(
                    f"--l1_lambdas has {len(self.args.l1_lambdas)} values but "
                    f"model has {len(weight_params)} weight matrices: "
                    f"{[n for n, _ in weight_params]}")
            for (name, _), lam in zip(weight_params, self.args.l1_lambdas):
                self._l1_map[name] = lam
                print(f"  L1 λ={lam} for {name}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        # Cosine annealing with warm restarts (AdamWR)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.args.warmrestart_T0,
            T_mult=self.args.warmrestart_Tmult,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training loop. Best-checkpoint metric:
        #   cell-line mode  -> STRATIFIED AUPRG (pooled over every observed
        #                      (pair, cell_line) cell; baseline-free).
        #   single-output   -> raw AUPR (unchanged project convention, so the
        #                      benchmark stays comparable to prior runs).
        # Early stopping shares the same metric — patience increments only when
        # it fails to improve. A NaN selection value (degenerate split) never
        # beats best_sel because NaN > x is False, so the save is skipped.
        sel_metric = "auprg" if self.args.use_cell_lines else "aupr"
        best_sel = float("-inf")
        best_epoch = 0
        patience_counter = 0
        best_metrics = {}
        n_evals = 0  # how many periodic evaluations actually ran

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold_idx + 1}")
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer,
                                          criterion)
            scheduler.step()

            # Evaluate periodically
            if (epoch + 1) % self.args.eval_interval == 0:
                metrics = self.evaluate(model, test_loader)
                n_evals += 1

                postfix = {
                    "loss": f"{train_loss:.4f}",
                    "aupr": f"{metrics['aupr']:.4f}",
                    "auprg": f"{metrics['auprg']:.4f}",
                    "auroc": f"{metrics['auroc']:.4f}",
                }
                pbar.set_postfix(postfix)

                if metrics[sel_metric] > best_sel:
                    best_sel = metrics[sel_metric]
                    best_epoch = epoch + 1
                    best_metrics = metrics.copy()
                    patience_counter = 0
                    if self.args.use_cell_lines:
                        ph = metrics.get("per_head_aupr", {})
                        phb = metrics.get("per_head_baseline", {})
                        phn = metrics.get("per_head_aupr_norm", {})
                        phg = metrics.get("per_head_auprg", {})
                        nan = float("nan")
                        print("\n  cell-line metrics  [baseline | raw AUPR | "
                              "norm AUPR | AUPRG]")
                        print(f"    strat : {metrics['aupr_baseline']:.3f} | "
                              f"{metrics['aupr']:.3f} | "
                              f"{metrics['aupr_norm']:.3f} | "
                              f"{metrics['auprg']:.3f}   <- AUPRG selects")
                        print(f"    any   : {metrics['baseline_any']:.3f} | "
                              f"{metrics['aupr_any']:.3f} | "
                              f"{metrics['aupr_any_norm']:.3f} | "
                              f"{metrics['auprg_any']:.3f}   "
                              f"(AUROC {metrics['auroc_any']:.3f})")
                        print(f"    macro : {metrics['aupr_baseline_macro']:.3f}"
                              f" | {metrics['aupr_macro']:.3f} | "
                              f"{metrics['aupr_norm_macro']:.3f} | "
                              f"{metrics['auprg_macro']:.3f}   (per-line avg)")
                        for n in cell_line_vocab.HEADS:
                            print(f"    {n:<7}: {phb.get(n, nan):.3f} | "
                                  f"{ph.get(n, nan):.3f} | "
                                  f"{phn.get(n, nan):.3f} | "
                                  f"{phg.get(n, nan):.3f}")

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": metrics,
                            "config": vars(self.args),
                            "input_dim": int(self.args.input_dim),
                            "preprocessing_transform": fold_transform,
                            "gene_order": [
                                self.data_manager.idx_to_gene[i]
                                for i in range(self.data_manager.num_genes)
                            ],
                            "embeddings_paths": list(
                                self.args.embeddings_paths),
                        }, self.output_dir / "checkpoints" /
                        f"fold_{fold_idx}_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.args.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

        # No checkpoint was saved during the loop. Two distinct causes:
        #   n_evals == 0 — epochs < eval_interval, so evaluate() never ran.
        #   n_evals  > 0 — every eval returned a NaN selection value (degenerate
        #                  fold), so `NaN > best_sel` was always False -> no save.
        if not best_metrics:
            if n_evals == 0:
                print(
                    f"\nWarning: No evaluation performed "
                    f"(epochs={self.args.epochs} < "
                    f"eval_interval={self.args.eval_interval})")
            else:
                print(
                    f"\nWarning: all {n_evals} evaluation(s) returned undefined "
                    f"AUPR (degenerate fold); saving the final-epoch checkpoint")
            best_metrics = self.evaluate(model, test_loader)
            best_epoch = self.args.epochs
            # Save checkpoint so nonzero counting and downstream loading work
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": best_metrics,
                    "config": vars(self.args),
                    "input_dim": int(self.args.input_dim),
                    "preprocessing_transform": fold_transform,
                    "gene_order": [
                        self.data_manager.idx_to_gene[i]
                        for i in range(self.data_manager.num_genes)
                    ],
                    "embeddings_paths": list(self.args.embeddings_paths),
                },
                self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt")

        # Count nonzero parameters from best checkpoint (exclude buffers)
        ckpt = torch.load(
            self.output_dir / "checkpoints" / f"fold_{fold_idx}_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        sd = ckpt["model_state_dict"]
        param_names = {n for n, _ in model.named_parameters()}
        params = [sd[n] for n in sd if n in param_names]
        total_params = sum(p.numel() for p in params)
        nonzero_params = sum(int(p.ne(0).sum()) for p in params)
        # Weight matrices only (L1 targets)
        wt_total = sum(p.numel() for p in params if p.dim() >= 2)
        wt_nonzero = sum(int(p.ne(0).sum()) for p in params if p.dim() >= 2)
        wt_sparsity = 100 * (1 - wt_nonzero / wt_total) if wt_total > 0 else 0

        print(
            f"\nFold {fold_idx + 1} Best (epoch {best_epoch}): "
            f"AUROC={best_metrics['auroc']:.4f}, AUPR={best_metrics['aupr']:.4f}, "
            f"F1={best_metrics['f1']:.4f}")
        print(
            f"  Params: {nonzero_params:,}/{total_params:,} nonzero "
            f"| Weights: {wt_nonzero:,}/{wt_total:,} nonzero ({wt_sparsity:.2f}% sparse)"
        )

        # Per-layer sparsity breakdown
        layer_sparsities = {}
        for name in sd:
            if name in param_names and sd[name].dim() >= 2:
                t = sd[name]
                total = t.numel()
                nz = int(t.ne(0).sum())
                sp = 100 * (1 - nz / total)
                layer_sparsities[name] = sp
                print(
                    f"    {name}: {nz:,}/{total:,} nonzero ({sp:.1f}% sparse)")

        best_metrics["total_params"] = total_params
        best_metrics["nonzero_params"] = nonzero_params
        best_metrics["weight_sparsity"] = wt_sparsity
        best_metrics["layer_sparsities"] = layer_sparsities

        return best_metrics

    def train(self):
        """Run full cross-validation training."""
        print("=" * 60)
        print(f"Siamese SL Training")
        print(f"Model: {self.args.model_type}")
        print(f"CV Type: {self.args.cv_type}")
        print(f"Folds: {self.args.num_folds}")
        print("=" * 60)

        # Get CV splits
        if self.args.cv_type == "cv1":
            splits = self.data_manager.get_cv1_splits(
                num_folds=self.args.num_folds,
                pos_neg_ratio=self.args.pos_neg_ratio,
            )
        elif self.args.cv_type == "cv2":
            splits = self.data_manager.get_cv2_splits(
                num_folds=self.args.num_folds,
                pos_neg_ratio=self.args.pos_neg_ratio,
            )
        else:  # cv3
            splits = self.data_manager.get_cv3_splits(
                num_folds=self.args.num_folds,
                pos_neg_ratio=self.args.pos_neg_ratio,
            )

        # Train each fold
        all_metrics = []
        for split in splits:
            metrics = self.train_fold(split)
            all_metrics.append(metrics)

        # Summarize results
        auroc_scores = np.array([m["auroc"] for m in all_metrics], dtype=float)
        aupr_scores = np.array([m["aupr"] for m in all_metrics], dtype=float)
        f1_scores = np.array([m["f1"] for m in all_metrics], dtype=float)

        # Print results table
        cv_desc = {
            "cv1": "CV1 (edge-based)",
            "cv2": "CV2 (gene-based)",
            "cv3": "CV3 (pair-based)",
        }

        print("\n" + "=" * 60)
        print(
            f"Cross-Validation Results: {cv_desc.get(self.args.cv_type, self.args.cv_type)}"
        )
        print("=" * 60)

        # Per-fold results
        print(f"\n{'Fold':<6} {'AUROC':>10} {'AUPR':>10} {'F1':>10}")
        print("-" * 40)
        for i, m in enumerate(all_metrics):
            print(
                f"{i+1:<6} {m['auroc']:>10.4f} {m['aupr']:>10.4f} {m['f1']:>10.4f}"
            )
        print("-" * 40)
        print(
            f"{'Mean':<6} {np.nanmean(auroc_scores):>10.4f} {np.nanmean(aupr_scores):>10.4f} {np.nanmean(f1_scores):>10.4f}"
        )
        print(
            f"{'Std':<6} {np.nanstd(auroc_scores):>10.4f} {np.nanstd(aupr_scores):>10.4f} {np.nanstd(f1_scores):>10.4f}"
        )

        # Sparsity summary
        sparsity_scores = np.array(
            [m.get("weight_sparsity", 0) for m in all_metrics])
        avg_nonzero = int(
            np.mean([m.get("nonzero_params", 0) for m in all_metrics]))
        avg_total = int(
            np.mean([m.get("total_params", 0) for m in all_metrics]))

        print("\n" + "=" * 60)
        print("Summary:")
        print(
            f"  AUROC: {np.nanmean(auroc_scores):.4f} ± {np.nanstd(auroc_scores):.4f}"
        )
        print(
            f"  AUPR:  {np.nanmean(aupr_scores):.4f} ± {np.nanstd(aupr_scores):.4f}"
        )
        print(
            f"  F1:    {np.nanmean(f1_scores):.4f} ± {np.nanstd(f1_scores):.4f}"
        )
        print(f"  Params: {avg_nonzero:,}/{avg_total:,} nonzero "
              f"(weight sparsity: {np.mean(sparsity_scores):.2f}%)")
        print("=" * 60)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "cv_type": self.args.cv_type,
            "cv_description": cv_desc.get(self.args.cv_type,
                                          self.args.cv_type),
            "config": vars(self.args),
            "fold_metrics": all_metrics,
            "summary": {
                "auroc_mean": float(np.nanmean(auroc_scores)),
                "auroc_std": float(np.nanstd(auroc_scores)),
                "aupr_mean": float(np.nanmean(aupr_scores)),
                "aupr_std": float(np.nanstd(aupr_scores)),
                "f1_mean": float(np.nanmean(f1_scores)),
                "f1_std": float(np.nanstd(f1_scores)),
                "total_params": avg_total,
                "nonzero_params": avg_nonzero,
                "weight_sparsity": float(np.mean(sparsity_scores)),
            },
        }

        # Cell-line mode: summarize the three views across folds. The primary
        # `aupr`/`auroc`/`auprg` above are STRATIFIED (pooled over observed
        # cells); these add the per-cell-line macro diagnostic and the global
        # "SL in any line" one-time headline.
        if self.args.use_cell_lines:
            def _fold_mean_std(key):
                arr = np.array([m.get(key, float("nan")) for m in all_metrics],
                               dtype=float)
                return float(np.nanmean(arr)), float(np.nanstd(arr))
            for key in ("auprg", "aupr_norm", "aupr_baseline",
                        "auprg_macro", "aupr_macro", "aupr_norm_macro",
                        "aupr_baseline_macro",
                        "aupr_any", "aupr_any_norm", "auprg_any", "auroc_any"):
                mean, std = _fold_mean_std(key)
                summary["summary"][f"{key}_mean"] = mean
                summary["summary"][f"{key}_std"] = std
            s = summary["summary"]
            print("\nCell-line metrics (mean over folds):")
            print("  STRATIFIED (pooled per-cell-line obs; SELECTION basis):")
            print(f"    baseline(pi) {s['aupr_baseline_mean']:.4f}  "
                  f"AUPR {s['aupr_mean']:.4f}  normAUPR {s['aupr_norm_mean']:.4f}"
                  f"  AUPRG {s['auprg_mean']:.4f}  AUROC {s['auroc_mean']:.4f}")
            print("  GLOBAL (SL in any cell line; one-time headline):")
            print(f"    AUPRG {s['auprg_any_mean']:.4f} "
                  f"± {s['auprg_any_std']:.4f}   AUPR {s['aupr_any_mean']:.4f}"
                  f"   AUROC {s['auroc_any_mean']:.4f}")
            print("  PER-LINE macro (equal weight/line; diagnostic):")
            print(f"    AUPRG {s['auprg_macro_mean']:.4f}  "
                  f"AUPR {s['aupr_macro_mean']:.4f}")

        def _nan_to_none(obj):
            """Replace NaN with None for valid JSON serialization."""
            if isinstance(obj, (float, np.floating)) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _nan_to_none(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_nan_to_none(v) for v in obj]
            return obj

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(_nan_to_none(summary), f, indent=2)

        print(f"\nResults saved to {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese network for SL prediction")

    # Data paths
    parser.add_argument("--embeddings_path",
                        type=str,
                        default=None,
                        help="(Deprecated: use --embeddings_paths) "
                        "Single embedding file, equivalent to "
                        "--embeddings_paths with one file.")
    parser.add_argument("--embeddings_paths",
                        type=str,
                        nargs='+',
                        default=None,
                        help="One or more .pt embedding files. Each modality "
                        "is imputed, optionally PCA-reduced, and "
                        "normalized before concatenation (see --pca_method).")
    parser.add_argument("--sl_path",
                        type=str,
                        default="../data/SL_SynLethDB_experimental.txt",
                        help="Path to SL pairs file")
    parser.add_argument(
        "--neg_pairs_path",
        type=str,
        default=None,
        help="Optional path to a curated NON-SL pairs file (gene1 TAB "
        "gene2), e.g. ../data/SL_SynLethDB_experimental_negatives.txt. When "
        "set, these experimentally screened negatives replace the benchmark's "
        "random negative sampling. Only pairs whose both genes also appear in "
        "a positive pair are usable; --pos_neg_ratio still controls balance.")
    parser.add_argument(
        "--use_cell_lines",
        action="store_true",
        help="Enable masked multi-label cell-line conditioning. The unit "
        "becomes a gene pair with a per-head (label, mask) over "
        "[K562, JURKAT, A549, HELA, A375, 293T, PC9, OTHER]; cell embeddings "
        "are concatenated onto the gene features (SiameseSLMultiCell), and the "
        "model predicts SL per head. Reads cell context from the "
        "'.sources.tsv' sidecars next to --sl_path / --neg_pairs_path; "
        "requires --neg_pairs_path. Selection metric is STRATIFIED AUPRG "
        "(pooled over observed cells; global 'any-line' + per-line reported).")
    parser.add_argument(
        "--cell_line_dim",
        type=int,
        default=8,
        help="Cell-line embedding dimension (only used with --use_cell_lines).")
    parser.add_argument("--gene_list_path",
                        type=str,
                        default=None,
                        help="Path to gene list file (for ordering)")
    parser.add_argument("--output_dir",
                        type=str,
                        default="results/siamese_esm",
                        help="Output directory")

    # Model architecture
    parser.add_argument(
        "--model_type",
        type=str,
        default="siamese",
        choices=["siamese", "attention", "kernel"],
        help=
        "Model: siamese (inner product, default), attention (cross-attn), kernel (RKHS-based)"
    )
    parser.add_argument("--input_dim",
                        type=int,
                        default=None,
                        help="Input embedding dim (auto-detected if omitted)")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=512,
                        help="Hidden dim")
    parser.add_argument("--latent_dim",
                        type=int,
                        default=256,
                        help="Latent dim")
    parser.add_argument("--predictor_hidden", type=int, default=128)
    parser.add_argument("--num_heads",
                        type=int,
                        default=4,
                        help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-last-layer-bias",
                        dest="last_layer_bias",
                        action="store_false",
                        default=True,
                        help="Remove last layer bias (gene node degree prior)")
    parser.add_argument(
        "--pd_epsilon",
        type=float,
        default=0.001,
        help="PD regularizer: ε in kernel K = WᵀW + εI (0 to disable)")
    parser.add_argument(
        "--encoder_dims",
        type=int,
        nargs='+',
        default=None,
        help="Encoder layer dims for siamese model (e.g., 8 8 8 8 8 8 8 — "
        "the SLURM default for the residual encoder, 6 hidden + 1 "
        "projection). The last entry is the projection (latent) dim. Falls "
        "back to --hidden_dim/--latent_dim if not set.")
    parser.add_argument(
        "--siamese_encoder_type",
        type=str,
        default="residual",
        choices=["mlp", "residual"],
        help="Architecture of the shared siamese encoder. "
        "'residual' (default): same-width hidden blocks get y = x + F(x) "
        "skip connections — gradient-friendly, enables deeper stacks. "
        "'mlp': plain stack of Linear→LN→LReLU→Dropout blocks (no skips). "
        "Kernel and attention siamese variants are unaffected by this flag.")

    # RKHS/Kernel model specific
    parser.add_argument(
        "--rff_features",
        type=int,
        default=128,
        help="Random Fourier features for kernel approximation")
    parser.add_argument("--bilinear_rank",
                        type=int,
                        default=128,
                        help="Hilbert space linear projection dimension")
    parser.add_argument("--kernel_sigma",
                        type=float,
                        default=1.0,
                        help="Initial RBF kernel bandwidth")

    # Efficient encoder options
    parser.add_argument("--encoder_type",
                        type=str,
                        default="standard",
                        choices=["standard", "lowrank", "bottleneck", "gated"],
                        help="Encoder architecture for efficiency")
    parser.add_argument("--encoder_rank",
                        type=int,
                        default=64,
                        help="Rank for lowrank encoder factorization")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--pca_dims",
        type=int,
        nargs='+',
        default=None,
        help="Per-modality PCA target dimensions (one per embedding file, "
        "e.g., 50 64 64). Robust PCA after imputation, before normalization. "
        "Values >= original dim are no-ops. Mutually exclusive with "
        "--pca_variance.")
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=None,
        help="Target variance fraction for PCA (0-1, e.g., 0.8 for 80%%). "
        "Applied to all modalities — each gets as many components as needed "
        "to explain this fraction. Mutually exclusive with --pca_dims.")
    parser.add_argument(
        "--post_pca_variance",
        type=float,
        default=0.8,
        help="Target variance fraction for a second ROBPCA applied to the "
        "concatenated matrix AFTER per-modality normalization. "
        "Default 0.8; pass --no_post_pca to disable, "
        "or --post_pca_dim for an exact component count.")
    parser.add_argument(
        "--post_pca_dim",
        type=int,
        default=None,
        help="Exact component count for the post-concat ROBPCA. "
        "Overrides --post_pca_variance when set.")
    parser.add_argument(
        "--no_post_pca",
        dest="no_post_pca",
        action="store_true",
        default=False,
        help="Disable the post-concat ROBPCA step (on by default).")
    parser.add_argument(
        "--preprocessing_fit_scope",
        type=str,
        default="train",
        choices=["train", "all"],
        help="Which gene rows to fit preprocessing (Huber-impute, ROBPCA, "
        "median/MAD normalize) on, PER FOLD.  "
        "'train' (default): only training-pair genes + non-SL genes — "
        "leak-free.  "
        "'all': every gene — matches pre-refactor legacy behavior and "
        "reintroduces CV2/CV3 leakage of test-gene embeddings into the "
        "basis. Use ONLY for A/B comparison.")
    parser.add_argument(
        "--pca_method",
        type=str,
        default="robust",
        choices=["robust", "plain", "svd"],
        help="PCA algorithm for per-modality and post-concat reduction. "
        "'robust': ROBPCA (Hubert et al. 2005) with alpha retries and a "
        "median-centered-SVD fallback; global median/MAD normalize. "
        "'plain': per-column mean/std standardize + SVD (each dim equal "
        "weight). 'svd': matrix-wise global-scalar normalize + median-"
        "centered SVD, NO per-column standardize (keeps per-dim variance "
        "importance) and NO ROBPCA (fast at high concat dims).")
    parser.add_argument(
        "--l1_lambdas",
        type=float,
        nargs='+',
        default=None,
        help="Per-layer L1 lambdas (one per weight matrix, e.g., 0.1 0.05)")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (in eval intervals, not epochs)")
    parser.add_argument(
        "--warmrestart_T0",
        type=int,
        default=50,
        help="CosineAnnealingWarmRestarts: initial cycle length (epochs)")
    parser.add_argument(
        "--warmrestart_Tmult",
        type=int,
        default=2,
        help="CosineAnnealingWarmRestarts: cycle length multiplier")

    # Cross-validation
    parser.add_argument(
        "--cv_type",
        type=str,
        default="cv1",
        choices=["cv1", "cv2", "cv3"],
        help=
        "CV type: cv1=edge-based, cv2=gene-based, cv3=pair-based (both genes unseen)"
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--pos_neg_ratio", type=float, default=1.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    # Unify: --embeddings_path is a convenience alias for a single-element list
    if args.embeddings_paths and args.embeddings_path:
        parser.error("Use --embeddings_path OR --embeddings_paths, not both")
    if args.embeddings_path:
        args.embeddings_paths = [args.embeddings_path]
    if not args.embeddings_paths:
        parser.error("--embeddings_paths is required (or --embeddings_path)")

    # PCA: --pca_variance is a convenience flag that sets pca_dims to a
    # single float applied to all modalities.
    if args.pca_variance is not None and args.pca_dims is not None:
        parser.error("Use --pca_dims OR --pca_variance, not both")
    if args.pca_variance is not None:
        if not 0 < args.pca_variance < 1:
            parser.error("--pca_variance must be in (0, 1)")
        args.pca_dims = [args.pca_variance] * len(args.embeddings_paths)

    # Post-concat PCA: on by default (variance=0.8). Resolution order:
    #   --no_post_pca  -> disabled (None)
    #   --post_pca_dim -> exact component count
    #   otherwise      -> --post_pca_variance (default 0.8)
    if args.no_post_pca:
        args.post_pca = None
    elif args.post_pca_dim is not None:
        if args.post_pca_dim <= 0:
            parser.error("--post_pca_dim must be a positive integer")
        args.post_pca = args.post_pca_dim
    else:
        if not 0 < args.post_pca_variance < 1:
            parser.error("--post_pca_variance must be in (0, 1)")
        args.post_pca = args.post_pca_variance

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
