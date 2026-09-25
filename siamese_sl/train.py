#!/usr/bin/env python3
"""
Training script for Siamese SL prediction model.

All embeddings go through the same pipeline per modality (--pca_method,
default "plain"):
  impute (mean) → PCA (optional) → normalize (mean center, std scale)
then concatenate along the feature axis, followed by an optional post-concat
PCA + re-normalize (on by default, --no_post_pca to disable). --pca_method
robust swaps in Huber/ROBPCA/median-MAD throughout.

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
from data_loader import (SLDataManager, create_fold_dataloaders,
                         create_val_dataloader)
import cell_line_vocab

import warnings
import importlib.util as _ilu

from retrieval_metrics import build_any_score_matrix

# --- shared benchmark evaluator (retrieval metrics) -------------------------
# The other three models get ndcg@k / recall@k / precision@k / map@k from
# sl_comparison/evaluate_model.py:91 -> evaluation.evaluate_predictions_dict ->
# preprocess_benchmarking_paper.cal_metrics (:1371). Load THE SAME function so
# siamese's numbers come out of identical code.
#
# Loaded BY FILE PATH, not via sys.path: ../SLMGAE-in-pytorch is already on
# sys.path (data_loader.py:44-45, which ran at line 47 above) and holds its own
# BYTE-IDENTICAL evaluation.py (verified with `diff -q`), so a bare
# `import evaluation` would resolve there with ambiguous provenance; and
# inserting sl_comparison/ at sys.path[0] would shadow siamese_sl/ for every
# later import in the process.
#
# MODULE SCOPE, not lazy: evaluation.py:16's `from preprocess_benchmarking_paper
# import cal_metrics` reaches a module that runs random.seed(123) /
# np.random.seed(123) at import (preprocess_benchmarking_paper.py:24-25). Doing
# it HERE -- before Trainer.__init__ calls set_seed(args.seed) at train.py:144
# -- keeps the run's own seed authoritative. (It is already a sys.modules hit
# via data_loader.py:1045 -> SLMGAE-in-pytorch/data_split.py, so nothing is
# reseeded either way; module scope keeps it that way.)
_EVAL_PY = (Path(__file__).resolve().parent.parent / "sl_comparison" /
            "evaluation.py")
try:
    _spec = _ilu.spec_from_file_location("_sl_comparison_evaluation", _EVAL_PY)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    evaluate_predictions_dict = _mod.evaluate_predictions_dict
except Exception as _e:  # noqa: BLE001
    evaluate_predictions_dict = None
    # MUST stay inside THIS handler. _retrieval_metrics is gated on
    # evaluate_predictions_dict being non-None (train.py:~1190), and it is the
    # sole caller of _save_shared_prediction_matrix — so a silent failure here
    # means the run finishes cleanly having written NO prediction matrices, and
    # collect_and_regrade_siamese.sh then publishes siamese from its own folds
    # and prevalence with nothing in the log naming the cause.
    print(f"Warning: retrieval metrics unavailable - could not load "
          f"{_EVAL_PY} ({type(_e).__name__}: {_e})")

# Parameter counting: one definition for all four models. Loaded BY FILE PATH
# for the same reason as evaluation.py above — putting sl_comparison/ on
# sys.path would shadow siamese_sl/ for every later import in the process.
_PARAMS_PY = (Path(__file__).resolve().parent.parent / "sl_comparison" /
              "model_params.py")
try:
    _pspec = _ilu.spec_from_file_location("_sl_comparison_model_params",
                                          _PARAMS_PY)
    _pmod = _ilu.module_from_spec(_pspec)
    _pspec.loader.exec_module(_pmod)
    write_model_params = _pmod.write_model_params
    purge_stale_predictions = _pmod.purge_stale_predictions
except Exception as _pe:  # noqa: BLE001
    # Bind the message NOW via a default argument: Python unbinds the `as`
    # name at the end of the except clause, so a closure over it would raise
    # NameError at call time — turning "never fatal" parameter counting into
    # the thing that kills the run.
    def write_model_params(output_dir,
                           model,
                           _msg=f"{type(_pe).__name__}: {_pe}",
                           **extra):
        print(f"  Warning: parameter counting unavailable ({_msg})")
        return None, None

    def purge_stale_predictions(output_dir, cv_type, keep=()):
        """Fallback: clear the previous run's matrices without the helper.

        This one is NOT optional the way parameter counting is — leaving a
        previous run's fold_*_predictions.npy in place lets evaluate_model.py
        grade it — so reimplement it rather than degrading to a warning.
        """
        out = Path(output_dir)
        n = 0
        for p in sorted(out.glob(f"fold_*_{cv_type}_predictions.npy")):
            p.unlink()
            n += 1
        if n:
            print(f"  Cleared {n} stale score matrix/matrices in {out}")
        return n

    print(f"Warning: could not load {_PARAMS_PY} "
          f"({type(_pe).__name__}: {_pe}); models will report no "
          f"parameter count")

# cal_metrics' 15 outputs, in evaluation.py:70-85 order. We keep ONLY these 12.
# auroc / f1 / aupr (indices 0-2) are DISCARDED: they are the same OR-collapsed
# edge-set quantity train.py already emits as auroc_any / f1_any / aupr_any,
# but computed by DIFFERENT formulas -- cal_metrics uses auc(recall, precision)
# (preprocess:1395, trapezoidal) vs average_precision_score, and
# max(2PR/(P+R)+1e-10) in cal_metrics vs calculate_optimal_f1's 1e-8
# (train.py:49-53). Emitting them under the same keys would silently replace
# siamese's headline numbers with different ones.
RETRIEVAL_KEYS = (
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
    "recall@10",
    "recall@20",
    "recall@50",
    "precision@10",
    "precision@20",
    "precision@50",
    "map@10",
    "map@20",
    "map@50",
)


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

    try:  # canonical reference implementation
        import prg
        return float(prg.calc_auprg(prg.create_prg_curve(labels, scores)))
    except Exception:
        pass

    precision, recall, _ = precision_recall_curve(labels, scores)
    odds = pi / (1.0 - pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - odds * (1.0 - precision) / precision
        rec_gain = 1.0 - odds * (1.0 - recall) / recall
    valid = recall > 0  # drop the terminal recall=0 point
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

    # ANCHOR AT recall_gain = 0 — keep in sync with sl_comparison/auprg.py.
    # precision_recall_curve collapses tied scores, so when the top tie block
    # alone lifts recall above pi there is no point with recall_gain <= 0 and
    # the strip [0, rg.min()] is never integrated. A PERFECT ranking with half
    # its positives tied at the top then scores 0.2 instead of 1.0.
    #
    # This matters most HERE: BCE keeps inflating logit magnitude, so float32
    # sigmoid saturation manufactures exactly those tie blocks as training
    # proceeds. Uncorrected, AUPRG falls while the ranking is unchanged — and
    # this function is the checkpoint-selection and early-stopping metric in
    # cell-line mode, so it was systematically preferring EARLIER, less
    # confident epochs for reasons unrelated to ranking quality.
    #
    # Precision-gain is flat within a tie block (all items share a score), so
    # the value at recall_gain = 0 is pg[0].
    if len(rg) and rg[0] > 0.0:
        rg = np.concatenate(([0.0], rg))
        pg = np.concatenate(([pg[0]], pg))

    def _interp(r, r0, r1, p0, p1):
        if r1 == r0:
            return p1
        return p0 + (r - r0) / (r1 - r0) * (p1 - p0)

    area = 0.0
    for i in range(1, len(rg)):
        r0, r1, p0, p1 = rg[i - 1], rg[i], pg[i - 1], pg[i]
        lo, hi = max(0.0, min(r0, r1)), min(1.0, max(r0, r1))
        if hi <= lo:  # segment outside [0, 1] recall-gain
            continue
        area += (hi - lo) * (_interp(lo, r0, r1, p0, p1) +
                             _interp(hi, r0, r1, p0, p1)) / 2.0
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

        # The shared-space prediction matrix is written from inside the
        # retrieval pass (the only place the dense matrix is built), so
        # disabling that would silently produce no gradeable output.
        if getattr(args, "folds_dir", None) and not args.retrieval_metrics:
            raise ValueError("--folds_dir needs the all-pairs pass to export "
                             "fold_<k>_<cv>_predictions.npy for sl_comparison/"
                             "evaluate_model.py; drop --no_retrieval_metrics.")

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
            folds_dir=getattr(args, "folds_dir", None),
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
                labels = labels.to(self.device)  # (B, H)
                mask = mask.to(self.device)  # (B, H)
                logits = model(x1, x2)  # (B, H)
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
        baseline = float(np.mean(all_labels))  # positive prevalence (pi)
        aupr_norm = _normalized_aupr(aupr,
                                     baseline)  # skill score: random -> 0
        auprg = _auprg(all_labels, all_scores)  # baseline-free (Flach&Kull)
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
        aupr_list = []  # raw per-head AUPRs (macro-averaged for report)
        aupr_norm_list = []  # normalized per-head AUPRs (reported)
        auprg_list = []  # per-head AUPRG (macro diagnostic)
        baseline_list = []  # per-head prevalence of scored heads
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
            pi = float(np.mean(yh))  # this head's positive prevalence
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
        strat_baseline = (float(np.mean(pooled_y))
                          if pooled_y.size else float("nan"))
        strat_aupr_norm = _normalized_aupr(strat_aupr, strat_baseline)
        strat_auprg = _auprg(pooled_y, pooled_s)  # <- selection metric

        # --- GLOBAL (one-time headline): collapse the head axis per pair ------
        # "SL in ANY cell line": one row per pair.
        #   label = 1 if the pair is SL in any supervised line, else 0 (a pair
        #           both SL and non-SL across lines counts positive).
        #   score = max over ALL heads of the SL probability.
        sup = M == 1
        has_pos = (sup & (Y == 1)).any(axis=1)
        has_neg = (sup & (Y == 0)).any(axis=1)
        agg_keep = has_pos | has_neg  # every pair has >=1 supervised head
        agg_y = has_pos[agg_keep].astype(np.float32)
        agg_s = P[agg_keep].max(axis=1)  # max over all heads -> one score/pair
        baseline_any = float(np.mean(agg_y)) if agg_y.size else float("nan")
        if len(np.unique(agg_y)) >= 2:
            aupr_any = float(average_precision_score(agg_y, agg_s))
            auroc_any = float(roc_auc_score(agg_y, agg_s))
        else:
            aupr_any = auroc_any = float("nan")
        aupr_any_norm = _normalized_aupr(aupr_any, baseline_any)
        auprg_any = _auprg(agg_y, agg_s)
        # F1 on the SAME OR-collapsed labels/scores, so the four-model table's
        # F1 column is populated for siamese the way it is for the shared-fold
        # models (which get it from cal_metrics on their test edges). A
        # single-class fold mirrors aupr_any/auroc_any and yields nan.
        f1_any = (float(calculate_optimal_f1(agg_y, agg_s)) if len(
            np.unique(agg_y)) >= 2 else float("nan"))

        # --- PER-CELL-LINE macro (diagnostic; equal weight per line) ---------
        macro = lambda xs: float(np.mean(xs)) if xs else float("nan")
        return {
            # PRIMARY = stratified pooled (drives selection + headline).
            "auroc": strat_auroc,
            "aupr": strat_aupr,
            "auprg": strat_auprg,  # stratified AUPRG -> selection
            "aupr_norm": strat_aupr_norm,
            "aupr_baseline": strat_baseline,
            "f1": strat_f1,
            # GLOBAL one-time headline (SL in any cell line, max over heads).
            "auroc_any": auroc_any,
            "aupr_any": aupr_any,
            "auprg_any": auprg_any,
            "aupr_any_norm": aupr_any_norm,
            "baseline_any": baseline_any,
            "f1_any": f1_any,
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

    def _save_shared_prediction_matrix(self, S: np.ndarray,
                                       fold_idx: int) -> None:
        """Write fold_<k>_<cv>_predictions.npy in SHARED gene index order.

        siamese scores against its own (genome-wide) embedding universe, but
        the four-model comparison is graded on sl_comparison/folds/genes.txt.
        Subsetting the dense matrix to that index space lets
        sl_comparison/evaluate_model.py run the SAME cal_metrics pass over the
        SAME test edges for all four models — which is what makes `baseline`
        identical rather than merely similar, and makes ndcg@k / recall@k
        comparable (every model now ranks against the same 8,844 candidates
        instead of siamese's ~19.4k).

        Genes in the shared universe with no embedding row get the matrix
        minimum, i.e. they are never retrieved. With the genome-wide universe
        that set is empty; it is filled defensively, not as a fallback.
        """
        shared_genes, s2f = self.data_manager._shared_gene_map()
        n_shared = len(shared_genes)
        have = np.where(s2f >= 0)[0]
        rows = s2f[have]
        out = np.full((n_shared, n_shared),
                      np.float32(S.min()),
                      dtype=np.float32)
        out[np.ix_(have, have)] = S[np.ix_(rows, rows)]
        path = (self.output_dir /
                f"fold_{fold_idx}_{self.args.cv_type}_predictions.npy")
        np.save(path, out)
        print(f"  Saved {n_shared}x{n_shared} shared-space score matrix -> "
              f"{path.name}")

    # MUST stay directly above _retrieval_metrics. The forward-equivalence
    # check below calls .numpy() on a model forward; with autograd enabled that
    # raises "Can't call numpy() on Tensor that requires grad", which the
    # never-fatal try/except in train_fold would swallow — silently skipping
    # retrieval AND the shared-matrix export for every fold.
    @torch.no_grad()
    def _retrieval_metrics(self,
                           model: nn.Module,
                           fold_data: dict,
                           fold_embeddings: torch.Tensor,
                           auroc_any_ref=None) -> dict:
        """All-pairs retrieval metrics for the OR-collapsed ("_any") view.

        Scored by the SAME evaluator the other three benchmark models use:
        cal_metrics' ranking half (preprocess_benchmarking_paper.py:1399-1455)
        argsorts each test gene's WHOLE row over every gene in the universe.

        UNIVERSE: siamese's own full embedding gene list (n = num_genes =
        5,257 today), NOT the shared 8,844 of sl_comparison/folds/genes.txt.
        Gathering into 8,844 is a numerical NO-OP -- the 3,587 extra genes
        appear in ZERO positive pairs (verified by set difference), so they are
        never a scored row and their columns can never enter a top-100 while
        5,256 real scores exist -- while the two gene lists use DIFFERENT sort
        orders (siamese string-sorted '10','100','10000'; shared int-sorted
        '2','9','10'), so a remap is a live silent-wrong risk for zero gain.
        The asymmetry is footnoted instead (see collect_siamese.py's note).

        Returns {} on a degenerate fold: no keys -> nanmean skips the fold ->
        collect_siamese.convert() omits the key -> compare.py renders "-".
        A blank cell is correct; a plausible wrong number is not.
        """
        dm = self.data_manager
        n = int(fold_embeddings.shape[0])

        # ================= INDEX-SPACE CONTRACT (assert, never assume) =====
        # A1: matrix rows are the FULL embedding space.
        if n != dm.num_genes:
            raise RuntimeError(
                f"fold_embeddings has {n} rows but num_genes={dm.num_genes}")
        if n < 100:
            # cal_metrics takes argsort(...)[:100]; with n < 100 the row IS the
            # top-k and recall@50 is trivially ~1.0 -- plausible and wrong.
            print(f"  [retrieval] universe too small ({n} < 100) -> skipped")
            return {}

        for part in ("train", "test"):
            pr = np.asarray(fold_data[part]["pairs"], dtype=np.int64)
            # A2: shape.
            if pr.ndim != 2 or pr.shape[1] != 2:
                raise RuntimeError(f"{part} pairs shape {pr.shape}")
            # A3: range. A -1 would silently WRAP to the last row of S.
            if pr.size and (int(pr.min()) < 0 or int(pr.max()) >= n):
                raise RuntimeError(
                    f"{part} pairs index range "
                    f"[{int(pr.min())}, {int(pr.max())}] outside 0..{n - 1}")
            # A4: canonical i<j. cal_metrics builds pos_matrix + pos_matrix.T
            # (preprocess:1383) and DUPLICATE COORDINATES SUM, so a mirrored
            # pair yields y_bool entries of 2: recall/precision inflate and
            # map@k collapses (measured: recall@50 0.1429 -> 0.2857,
            # map@50 -> 0.0).
            if pr.size and not bool((pr[:, 0] < pr[:, 1]).all()):
                raise RuntimeError(f"{part} pairs are not canonical i<j")
            # A5: uniqueness -- same csr double-count hazard.
            if len({tuple(p) for p in pr.tolist()}) != len(pr):
                raise RuntimeError(f"{part} pairs contain duplicates")

        te_pairs = np.asarray(fold_data["test"]["pairs"], dtype=np.int64)
        # A6: fold pairs MUST be rows of ml_pairs_full (FULL embedding index
        # space), never ml_pairs_p (data_loader.py:1504). TODAY full_to_p is
        # provably the IDENTITY (num_p_genes == num_genes == 5257, because
        # data/all_genes_esm2.genes.txt is set-identical to the positive-pair
        # gene set), so a P-space/full-space mix-up produces NO error and NO
        # wrong number -- it detonates silently the first time the gene list
        # changes. This guard checks the contract, not the coincidence.
        _full_rows = set(map(tuple, dm.ml_pairs_full.tolist()))
        if not all(tuple(p) in _full_rows for p in te_pairs[:2000].tolist()):
            raise RuntimeError("fold pairs are not rows of ml_pairs_full -- "
                               "wrong gene index space")
        _np_genes = int(getattr(dm, "num_p_genes", -1))
        _space = ("P-space == full space, so a P/full mix-up would be SILENT"
                  if _np_genes == n else "P-space differs from full space")
        print(f"  [retrieval] index space: n={n} num_genes={dm.num_genes} "
              f"num_p_genes={_np_genes} ({_space})")

        # ================= OR-collapse, byte-identical to train.py:464-469 ==
        def _any(part, val):
            Y = np.asarray(fold_data[part]["labels"])
            M = np.asarray(fold_data[part]["mask"])
            return ((M == 1) & (Y == val)).any(axis=1)

        te_pos, te_neg = _any("test", 1), _any("test", 0)
        # train.py:467 keeps has_pos | has_neg; mirror that literally rather
        # than relying on ~has_pos (equivalent only because data_loader.py:
        # 1487-1489 drops mask-empty pairs).
        if te_pairs.size and not bool((te_pos | te_neg).all()):
            raise RuntimeError(
                "test pair with no supervised head -- the *_any "
                "view (train.py:467) would exclude it")
        pos_index = te_pairs[te_pos]
        neg_index = te_pairs[te_neg & ~te_pos]

        # cv3 assigns pairs with EXACTLY ONE held-out gene to neither train nor
        # test (data_loader.py:1557-1565). Such a positive is neither rewarded
        # (absent from pos_index) nor suppressed (absent from seen_index).
        # sl_comparison/prepare_folds.py does the IDENTICAL thing for the other
        # three models, so the deflation is SHARED. Do NOT "fix" it here --
        # that would make siamese optimistically biased.
        tr_pairs = np.asarray(fold_data["train"]["pairs"], dtype=np.int64)
        seen_index = tr_pairs[_any("train", 1)]  # train POSITIVES only

        if len(pos_index) == 0:
            print("  [retrieval] no OR-collapsed test positives -> skipped")
            return {}
        # A9: seen_index must be train POSITIVES only. cal_metrics sets seen
        # entries to -999999, so masking train NEGATIVES too would delete hard
        # distractors that the peers keep (evaluate_model.py:91 passes
        # train_pos, never train_neg) -- silently inflating siamese's @k.
        if len(seen_index) >= len(tr_pairs):
            raise RuntimeError("seen_index must be train POSITIVES only")
        # A8: a test positive suppressed to -999999 can never be retrieved,
        # silently deflating recall.
        _ov = (set(map(tuple, pos_index.tolist()))
               & set(map(tuple, seen_index.tolist())))
        if _ov:
            raise RuntimeError(f"{len(_ov)} test positives also in seen_index")
        if len(neg_index) == 0:
            # VERIFIED by execution: an empty neg_index does NOT crash
            # cal_metrics -- precision_recall_curve accepts the all-positive
            # y_true and roc_auc_score merely warns and returns nan. We discard
            # auroc/f1/aupr anyway, so feed a sentinel and keep the retrieval
            # half rather than throwing computable numbers away.
            neg_index = pos_index[:1]
            auroc_any_ref = None  # AUROC identity undefined here
            print("  [retrieval] no OR-collapsed test negatives; edge metrics "
                  "undefined (discarded anyway), retrieval kept")

        # ================= dense n x n max-over-heads LOGIT matrix ==========
        model.eval()
        E = fold_embeddings.to(self.device, dtype=torch.float32)
        S_t = build_any_score_matrix(model, E)

        # ---- A14: FORWARD-EQUIVALENCE PROOF (primary alignment guard) -----
        # Re-score real test pairs through the ordinary batched forward -- the
        # exact path _evaluate_multilabel uses -- and require the matrix to
        # agree. This ties pairs -> fold_embeddings rows -> matrix indices in
        # ONE check. Any row/col permutation, P-vs-full mix-up, stale weights
        # or live dropout is an O(1) discrepancy here, not O(1e-6). Uses a
        # LOCAL RandomState so the run's global RNG is untouched.
        k = int(min(1024, len(te_pairs)))
        sel = np.random.RandomState(0).choice(len(te_pairs), k, replace=False)
        pi = torch.as_tensor(te_pairs[sel, 0],
                             device=self.device,
                             dtype=torch.long)
        pj = torch.as_tensor(te_pairs[sel, 1],
                             device=self.device,
                             dtype=torch.long)
        ref_logit = model(E[pi], E[pj]).max(dim=1).values.float().cpu().numpy()
        got = S_t[pi, pj].float().cpu().numpy()
        err = float(np.abs(ref_logit - got).max())
        if err > 1e-3:
            print("!" * 70)
            print(f"!! RETRIEVAL ALIGNMENT CHECK FAILED: the all-pairs matrix "
                  f"disagrees with model.forward on {k} real test pairs "
                  f"(max |diff| = {err:.3e}). Gene index space or weights are "
                  f"wrong. Suppressing retrieval metrics.")
            print("!" * 70)
            raise RuntimeError(f"matrix vs forward mismatch {err:.3e}")

        S = S_t.float().cpu().numpy()
        del S_t
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # A11: NaN sorts LAST ascending, so [::-1] puts it FIRST -- the top-100
        # would fill with NaN genes.
        if not np.isfinite(S).all():
            raise RuntimeError("non-finite entries in the all-pairs matrix")
        # A13: degenerate / collapsed matrix.
        span = float(S.max() - S.min())
        if span < 1e-3:
            raise RuntimeError(f"degenerate score matrix (range {span:.3e})")

        # ---- THE AFFINE SHIFT (do NOT sigmoid, do NOT leave raw) ----------
        # cal_metrics hard-writes score_matrix_ndcg[i, i] = 0 (preprocess:1400)
        # and the sentinel -999999 into seen edges (:1401-1406) -- values on a
        # scale it does not own. This model's max-over-heads logit is
        # tau*inner + max_h b_h; with the real trained tau = 0.229 and
        # max_h b_h = -0.915 most of the range is NEGATIVE, so the injected 0
        # would outrank real candidates and steal a top-100 slot in every row.
        # Every cal_metrics output is rank- or threshold-based, so a strictly
        # increasing affine map is provably neutral.
        #
        # MEASURED (600-gene synthetic in the real trained-parameter regime,
        # scratchpad/probe_retrieval.py): SHIFTED == sigmoid on all 12 metrics
        # to 1e-12; RAW logits differ (ndcg@20 0.029924 vs 0.029653, and the
        # 0-diagonal reached the top-10 of 1/398 scored rows); and sigmoid
        # DRIFTS once float32 saturates -- at 2.9% saturated entries ndcg@10
        # inflated 0.01724 -> 0.02241 (+30%) because argsort breaks exact ties
        # by gene index. log_temperature is an unclamped learned parameter
        # (siamese_esm.py:581-583), so saturation is not hypothetical.
        S = (S - S.min() + np.float32(1.0)).astype(np.float32)

        # A10: cal_metrics line 1399 deepcopies the ORIGINAL score_mat (not the
        # np.asarray'd copy), so a sparse / np.matrix input leaves
        # score_matrix_ndcg[i, :] 2-D and argsort(...)[::-1] reverses the WRONG
        # axis -- silent garbage. float16 would overflow -999999 to -inf.
        if not (isinstance(S, np.ndarray) and S.ndim == 2
                and S.shape[0] == S.shape[1] == n and S.dtype == np.float32):
            raise RuntimeError(f"score matrix must be a dense square float32 "
                               f"ndarray; got {type(S)} {S.shape} {S.dtype}")
        # A12: the injected sentinels must sit strictly below every real score.
        if not float(S.min()) > 0.0:
            raise RuntimeError("affine shift failed: min(S) must exceed the "
                               "0 cal_metrics writes on the diagonal")
        # ...and closes the loop rigorously: the 0 diagonal can only reach a
        # top-100 if a row has fewer than 100 unseen non-diagonal entries.
        _deg = np.bincount(seen_index.ravel(), minlength=n)
        if int(_deg.max()) > n - 101:
            raise RuntimeError(
                f"gene with seen-degree {int(_deg.max())} > {n - 101}: the "
                f"zeroed diagonal could enter its top-100")

        # Export the same matrix in SHARED gene order so the peers' evaluator
        # grades siamese on the shared answer key (identical baseline).
        if self.data_manager.folds_dir:
            self._save_shared_prediction_matrix(S, fold_data["fold"])

        # ================= score ===========================================
        with warnings.catch_warnings():
            # preprocess:1397 divides 0/0 on the PR curve's endpoint on EVERY
            # call (that f1 is discarded); left unsuppressed it spams the log.
            warnings.simplefilter("ignore")
            m = evaluate_predictions_dict(S, pos_index, neg_index, seen_index)

        # ---- A15: AUROC identity (secondary, end-to-end) ------------------
        # cal_metrics' AUROC comes from S[pos]/S[neg]; max-of-logits is rank-
        # equivalent to max-of-sigmoids, so it MUST equal auroc_any. Tolerance
        # is 1e-2, NOT 1e-6: cal_metrics' side comes from a cuBLAS z @ z.T
        # (K=16, M=N=5257) while auroc_any comes from (z1*z2).sum(dim=1) over a
        # (B*H, 16) batch (siamese_esm.py:614) -- different reduction shapes,
        # ~1e-6-scale logit differences, and with ~2.7e8 comparisons a few
        # hundred near-tie flips already breach 1e-6.
        if (auroc_any_ref is not None and np.isfinite(auroc_any_ref)
                and np.isfinite(m["auroc"])):
            d = abs(float(m["auroc"]) - float(auroc_any_ref))
            if d > 1e-2:
                raise RuntimeError(
                    f"cal_metrics AUROC {m['auroc']:.6f} vs auroc_any "
                    f"{auroc_any_ref:.6f} (delta {d:.2e})")
            if d > 1e-4:
                print(f"  [retrieval] WARNING: AUROC delta {d:.2e} "
                      f"vs auroc_any")

        # ================= keep ONLY the retrieval half =====================
        out = {f"{key}_any": float(m[key]) for key in RETRIEVAL_KEYS}
        # A16: cal_metrics also returns auroc/f1/aupr, which would become
        # auroc_any/f1_any/aupr_any and clobber siamese's headline metrics.
        if {"auroc_any", "f1_any", "aupr_any"} & set(out):
            raise RuntimeError("cal_metrics' edge metrics must not overwrite "
                               "siamese's own")
        out["rank_universe_n"] = n
        n_rows = len(set(pos_index.ravel().tolist()))
        print(f"  [retrieval] universe {n:,} genes (peers rank vs 8,844), "
              f"{n_rows:,} scored rows, {len(pos_index):,} test-pos, "
              f"{len(seen_index):,} train-pos suppressed | "
              f"fwd-check {err:.2e} | "
              f"NDCG@10 {out['ndcg@10_any']:.4f} "
              f"NDCG@50 {out['ndcg@50_any']:.4f} "
              f"R@50 {out['recall@50_any']:.4f} "
              f"MAP@50 {out['map@50_any']:.4f}")
        return out

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
        # Shared canonical folds carry a VAL split; select on it so the
        # reported test numbers are not also the selection criterion. Siamese's
        # own folds have none — those runs keep the legacy select-on-test path.
        val_loader = create_val_dataloader(
            embeddings=fold_embeddings,
            fold_data=fold_data,
            batch_size=self.args.batch_size,
        )
        sel_loader = val_loader if val_loader is not None else test_loader
        sel_split = "val" if val_loader is not None else "TEST"
        if val_loader is None:
            print("  NOTE: no validation split in this fold — selecting on "
                  "TEST. Use --folds_dir with folds built by prepare_folds.py "
                  "--val_frac for an honest number.")

        # Create model and optimizer
        model = self.create_model()
        # selection_regime travels with the predictions (model_params.json ->
        # evaluate_model.build_summary -> compare.py), so the table itself shows
        # which rows were val-selected. Nothing else could distinguish a row
        # whose checkpoint was picked on TEST from one picked honestly.
        write_model_params(self.output_dir,
                           model,
                           folds_dir=getattr(self.args, "folds_dir", None),
                           cv_type=getattr(self.args, "cv_type", None),
                           selection_regime=sel_split)

        # Remove this fold's matrix from any previous run BEFORE training.
        # evaluate_model.py grades fold_<k>_<cv>_predictions.npy, so a fold that
        # dies before _save_shared_prediction_matrix leaves the PREVIOUS run's
        # matrix to be graded silently alongside fresh ones — reproduced on
        # SLMGAE at exit 0 with no warning. A crashed fold must be MISSING
        # (evaluate_model skips it and says so), never stale.
        _stale = (self.output_dir /
                  f"fold_{fold_idx}_{self.args.cv_type}_predictions.npy")
        if _stale.exists():
            print(f"  Removing stale prediction matrix from a previous run: "
                  f"{_stale.name}")
            _stale.unlink()

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
        degenerate_fold = False  # set if selection never fired (see below)

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold_idx + 1}")
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer,
                                          criterion)
            scheduler.step()

            # Evaluate periodically
            if (epoch + 1) % self.args.eval_interval == 0:
                metrics = self.evaluate(model, sel_loader)
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
                        print(
                            f"    macro : {metrics['aupr_baseline_macro']:.3f}"
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
                            "epoch":
                            epoch + 1,
                            "model_state_dict":
                            model.state_dict(),
                            "optimizer_state_dict":
                            optimizer.state_dict(),
                            "metrics":
                            metrics,
                            "config":
                            vars(self.args),
                            "input_dim":
                            int(self.args.input_dim),
                            "preprocessing_transform":
                            fold_transform,
                            "gene_order": [
                                self.data_manager.idx_to_gene[i]
                                for i in range(self.data_manager.num_genes)
                            ],
                            "embeddings_paths":
                            list(self.args.embeddings_paths),
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
                print(f"\nWarning: No evaluation performed "
                      f"(epochs={self.args.epochs} < "
                      f"eval_interval={self.args.eval_interval})")
            else:
                print(
                    f"\nWarning: all {n_evals} evaluation(s) returned undefined "
                    f"AUPR (degenerate fold); saving the final-epoch checkpoint"
                )
            best_metrics = self.evaluate(model, test_loader)
            best_epoch = self.args.epochs
            # This path already scored TEST, and nothing was ever selected on
            # val. Flag it so the rescore block below does not relabel a test
            # number as `val_<metric>` — collect_siamese.py ranks configs on
            # that key and would silently be selecting on test again.
            degenerate_fold = True
            # Save checkpoint so nonzero counting and downstream loading work
            torch.save(
                {
                    "epoch":
                    best_epoch,
                    "model_state_dict":
                    model.state_dict(),
                    "optimizer_state_dict":
                    optimizer.state_dict(),
                    "metrics":
                    best_metrics,
                    "config":
                    vars(self.args),
                    "input_dim":
                    int(self.args.input_dim),
                    "preprocessing_transform":
                    fold_transform,
                    "gene_order": [
                        self.data_manager.idx_to_gene[i]
                        for i in range(self.data_manager.num_genes)
                    ],
                    "embeddings_paths":
                    list(self.args.embeddings_paths),
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
        # All 2-D parameters. NOTE this is NOT the L1 target set: in
        # cell-line mode cell_emb.weight is 2-D and is counted here, but it is
        # deliberately excluded from the proximal L1 (names starting with
        # "cell_", see the _l1_map construction above). So this sparsity figure
        # is diluted by one dense tensor that L1 was never applied to — read it
        # as "weight-matrix sparsity", not "sparsity of what L1 targeted".
        wt_total = sum(p.numel() for p in params if p.dim() >= 2)
        wt_nonzero = sum(int(p.ne(0).sum()) for p in params if p.dim() >= 2)
        wt_sparsity = 100 * (1 - wt_nonzero / wt_total) if wt_total > 0 else 0

        # With a shared val split, best_metrics so far are VAL metrics — the
        # selection criterion, not a result. Restore the selected checkpoint and
        # score TEST exactly once. Test then never influenced training, matching
        # SLGNN/train_slgnn.py:149-152 and NSF4SL/train_nsf4sl.py.
        if val_loader is not None and not degenerate_fold:
            sel_value = best_metrics.get(sel_metric, float("nan"))
            model.load_state_dict(sd)
            best_metrics = self.evaluate(model, test_loader)
            best_metrics[f"val_{sel_metric}"] = float(sel_value)
            print(f"\n  selected on {sel_split} ({sel_metric}="
                  f"{sel_value:.4f} @ epoch {best_epoch}); test scored once")
        elif val_loader is not None:
            # Degenerate fold: best_metrics are already TEST metrics from the
            # final epoch and no val selection ever happened. Record NaN so the
            # fold contributes nothing to val_<metric>_mean rather than
            # contributing a test value disguised as a val one.
            best_metrics[f"val_{sel_metric}"] = float("nan")
            print(f"\n  WARNING: no val selection occurred in this fold; "
                  f"val_{sel_metric} recorded as NaN")

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

        # --- Retrieval metrics on the BEST checkpoint (ONCE per fold) -------
        # `model` may hold EITHER the last epoch's weights (no-val path, where
        # nothing is ever restored) OR the val-selected weights (the val path
        # calls load_state_dict a few dozen lines above). Reload unconditionally
        # rather than reasoning about which: the point is that
        # ndcg/recall/precision/map must describe the same checkpoint as
        # best_metrics / auroc_any, whichever path got us here. Mirrors the peer
        # implementation in SLGNN's trainer.
        #
        # NEVER FATAL: train() writes results.json only after ALL folds finish
        # (train.py:898-899), so an exception on fold 3 would throw away folds
        # 0-2 including the aupr_any/auroc_any that already work today. A
        # failure must degrade to "-" (the status quo), never to a lost run.
        if (self.args.use_cell_lines and self.args.retrieval_metrics
                and evaluate_predictions_dict is not None
                and isinstance(model, SiameseSLMultiCell)):
            try:
                model.load_state_dict(sd)
                model.eval()
                best_metrics.update(
                    self._retrieval_metrics(
                        model,
                        fold_data,
                        fold_embeddings,
                        auroc_any_ref=best_metrics.get("auroc_any")))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  Warning: retrieval metrics skipped for fold "
                      f"{fold_idx + 1} ({type(e).__name__}: {e}); those "
                      f"columns render as '-'")

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

        # CLEAR EVERY FOLD'S LEFTOVER MATRIX BEFORE FOLD 0.
        #
        # The per-fold clear inside the loop only protects folds this run
        # actually reaches. A run that dies at fold 3 leaves folds 3-4 holding
        # the PREVIOUS run's matrices, and evaluate_model.py grades every
        # matrix it finds in the directory — publishing a mean over fresh and
        # foreign folds at exit 0. A crashed fold must be MISSING, not stale.
        purge_stale_predictions(self.output_dir, self.args.cv_type)

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

            # Every *_any key the four-model table consumes MUST be listed here:
            # collect_siamese.py reads <key>_mean, so a metric computed per fold
            # but missing from this tuple never gets a _mean and silently renders
            # as "-" in the comparison table. baseline_any did exactly that.
            agg_keys = [
                "auprg", "aupr_norm", "aupr_baseline", "auprg_macro",
                "aupr_macro", "aupr_norm_macro", "aupr_baseline_macro",
                "aupr_any", "aupr_any_norm", "auprg_any", "auroc_any",
                "baseline_any", "f1_any"
            ]
            # The VAL selection score. With --folds_dir this is what config-level
            # selection (collect_siamese.py --discover_dir) must rank on: picking
            # the best of ~17 candidate configs by a TEST metric is a second,
            # larger winner's curse on top of the per-epoch one.
            if any("val_auprg" in m for m in all_metrics):
                agg_keys.append("val_auprg")
            # Retrieval metrics (rank each test gene against the WHOLE gene
            # universe, not just the test negatives). Appended only when at
            # least one fold actually produced them -- otherwise every value is
            # absent and np.nanmean warns on an empty slice. Folds that failed
            # individually contribute NaN -> nanmean skips them; if EVERY fold
            # failed the key is absent entirely -> compare.py renders "-".
            _has_retr = any("ndcg@10_any" in m for m in all_metrics)
            if _has_retr:
                agg_keys += [
                    "ndcg@10_any",
                    "ndcg@20_any",
                    "ndcg@50_any",
                    "recall@10_any",
                    "recall@20_any",
                    "recall@50_any",
                    "precision@10_any",
                    "precision@20_any",
                    "precision@50_any",
                    "map@10_any",
                    "map@20_any",
                    "map@50_any",
                ]
            for key in agg_keys:
                mean, std = _fold_mean_std(key)
                summary["summary"][f"{key}_mean"] = mean
                summary["summary"][f"{key}_std"] = std
            # Ranking-universe size: an exact integer, NOT a nanmean (a mean of
            # a gene count is meaningless). collect_siamese reads it to build
            # the comparability footnote.
            if _has_retr:
                _u = [
                    int(m["rank_universe_n"]) for m in all_metrics
                    if m.get("rank_universe_n")
                ]
                if _u:
                    if len(set(_u)) != 1:
                        print(f"  WARNING: ranking universe varied across "
                              f"folds: {sorted(set(_u))}")
                    summary["summary"]["rank_universe_n"] = _u[0]
            s = summary["summary"]
            print("\nCell-line metrics (mean over folds):")
            print("  STRATIFIED (pooled per-cell-line obs; SELECTION basis):")
            print(
                f"    baseline(pi) {s['aupr_baseline_mean']:.4f}  "
                f"AUPR {s['aupr_mean']:.4f}  normAUPR {s['aupr_norm_mean']:.4f}"
                f"  AUPRG {s['auprg_mean']:.4f}  AUROC {s['auroc_mean']:.4f}")
            print("  GLOBAL (SL in any cell line; one-time headline):")
            print(f"    AUPRG {s['auprg_any_mean']:.4f} "
                  f"± {s['auprg_any_std']:.4f}   AUPR {s['aupr_any_mean']:.4f}"
                  f"   AUROC {s['auroc_any_mean']:.4f}")
            print("  PER-LINE macro (equal weight/line; diagnostic):")
            print(f"    AUPRG {s['auprg_macro_mean']:.4f}  "
                  f"AUPR {s['aupr_macro_mean']:.4f}")
            if _has_retr and "ndcg@10_any_mean" in s:
                print(f"  RETRIEVAL (rank vs the whole "
                      f"{s.get('rank_universe_n', '?'):,}-gene universe; "
                      f"peers rank vs 8,844 -- NOT strictly comparable):")
                print(f"    NDCG@10 {s['ndcg@10_any_mean']:.4f}  "
                      f"NDCG@50 {s['ndcg@50_any_mean']:.4f}  "
                      f"R@50 {s['recall@50_any_mean']:.4f}  "
                      f"MAP@50 {s['map@50_any_mean']:.4f}")

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
        "--folds_dir",
        type=str,
        default=None,
        help="Use the shared canonical CV folds from this directory "
        "(e.g. ../sl_comparison/folds) instead of siamese's own gene-disjoint "
        "partition — the fair four-model comparison path. Train/val/test "
        "division and gene index space become identical to SLMGAE / SLGNN / "
        "NSF4SL; per-head labels stay siamese's, so cell-line conditioning is "
        "unaffected. Checkpoint selection moves onto the shared VAL split, and "
        "each fold writes fold_<k>_<cv>_predictions.npy in shared gene order "
        "so sl_comparison/evaluate_model.py grades it identically to the "
        "peers. Requires --use_cell_lines.")
    parser.add_argument(
        "--cell_line_dim",
        type=int,
        default=8,
        help="Cell-line embedding dimension (only used with --use_cell_lines)."
    )
    parser.add_argument(
        "--no_retrieval_metrics",
        dest="retrieval_metrics",
        action="store_false",
        default=True,
        help="Skip the once-per-fold all-pairs retrieval pass (ndcg@k / "
        "recall@k / precision@k / map@k for the OR-collapsed 'any' view, "
        "computed with sl_comparison's cal_metrics so the four-model table "
        "uses identical code). Cell-line mode only - ignored without "
        "--use_cell_lines. ON by default; the pass is wrapped in try/except "
        "so a failure only drops those columns, never the run.")
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
        help="Encoder layer dims for siamese model. The SLURM default is "
        "16 16 16 16 16 16 16 (slurm/config.conf): 6 hidden + 1 projection. "
        "The last entry is the projection (latent) dim. Equal widths matter "
        "under --siamese_encoder_type residual — a skip fires only between "
        "same-width blocks, so a narrowing stack gets none. Falls back to "
        "--hidden_dim/--latent_dim if not set.")
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
        default="plain",
        choices=["plain", "robust", "svd"],
        help="Selects the WHOLE preprocessing pipeline, not just the PCA "
        "algorithm, for per-modality and post-concat reduction. "
        "'plain' (default): mean-impute + per-column mean/std standardize + "
        "SVD + global mean/std normalize — classical measures throughout. "
        "'robust': Huber-impute + ROBPCA (Hubert et al. 2005) with alpha "
        "retries and a median-centered-SVD fallback; global median/MAD "
        "normalize. 'svd': matrix-wise global-scalar normalize + median-"
        "centered SVD, NO per-column standardize and NO ROBPCA. 'robust' and "
        "'svd' are kept so earlier runs stay reproducible.")
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
