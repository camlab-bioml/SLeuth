#!/usr/bin/env python3
"""
Training script for Siamese SL prediction model.

All embeddings go through the same pipeline per modality (--pca_method,
default "plain"):
  impute (mean) → PCA (optional) → normalize (mean center, std scale)
then concatenate along the feature axis. No PCA runs by default; an optional
post-concat PCA + re-normalize is available with --post_pca_variance or
--post_pca_dim. --pca_method robust swaps in Huber/median-MAD (and ROBPCA,
where a PCA is requested) throughout.

Usage (early stopping and checkpoint selection are validation-only, and only
the shared folds carry a val split, so every run needs --folds_dir; without it
train.py exits. Cell-line mode also needs --use_cell_lines and
--neg_pairs_path; single-output mode reads its labels from the folds):
    # Single embedding
    python train.py --embeddings_paths ../data/all_genes_go.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --neg_pairs_path ../data/SL_SynLethDB_experimental_negatives.txt \
                    --use_cell_lines --folds_dir ../sl_comparison/folds \
                    --output_dir results/siamese_go

    # Multi-modal (concatenated)
    python train.py --embeddings_paths ../data/all_genes_bioconceptvec.pt \
                        ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --neg_pairs_path ../data/SL_SynLethDB_experimental_negatives.txt \
                    --use_cell_lines --folds_dir ../sl_comparison/folds \
                    --output_dir results/multi_bio_go_ppi

    # Single output (SiameseSL, one logit per pair, the folds' own labels)
    python train.py --embeddings_paths ../data/all_genes_go.pt \
                    --sl_path ../data/SL_SynLethDB_experimental.txt \
                    --folds_dir ../sl_comparison/folds \
                    --output_dir results/siamese_go_single
"""

import json
import types
import hashlib
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
# sl_comparison/evaluate_model.score_fold -> evaluation.evaluate_predictions_dict
# -> preprocess_benchmarking_paper.cal_metrics (:1371). Load THE SAME function
# so siamese's numbers come out of identical code.
#
# Loaded BY FILE PATH, not via sys.path: ../SLMGAE-in-pytorch is already on
# sys.path (data_loader.py inserts it at import, i.e. during the
# `from data_loader import ...` above) and holds its own evaluation.py (same
# code; only the module docstrings differ), so a bare `import evaluation`
# would resolve there with ambiguous provenance; and inserting sl_comparison/
# at sys.path[0] would shadow siamese_sl/ for every later import in the
# process.
#
# MODULE SCOPE, not lazy: evaluation.py's `from preprocess_benchmarking_paper
# import cal_metrics` reaches a module that runs random.seed(123) /
# np.random.seed(123) at import (preprocess_benchmarking_paper.py:24-25). Doing
# it HERE -- before Trainer.__init__ calls set_seed(args.seed) -- keeps the
# run's own seed authoritative. (It is already a sys.modules hit via
# data_loader.py's `from data_split import SLDataSplitter` ->
# SLMGAE-in-pytorch/data_split.py, so nothing is reseeded either way; module
# scope keeps it that way.)
#
# NO FALLBACK (like auprg.py below, and like the peer trainers' plain
# imports): every run needs --folds_dir, and the retrieval pass that calls
# this evaluator is also the only place the graded fold_<k>_<cv>_predictions.npy
# is written, so a run without it would train for hours and leave nothing
# evaluate_model.py can grade. Fail here instead.
_EVAL_PY = (Path(__file__).resolve().parent.parent / "sl_comparison" /
            "evaluation.py")
_spec = _ilu.spec_from_file_location("_sl_comparison_evaluation", _EVAL_PY)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evaluate_predictions_dict = _mod.evaluate_predictions_dict

# Parameter counting, the fold stamp and the stale-matrix purge: one
# definition for all four models. Loaded BY FILE PATH for the same reason as
# evaluation.py above — putting sl_comparison/ on sys.path would shadow
# siamese_sl/ for every later import in the process. No fallback either:
# write_model_params also writes the fold_stamp that evaluate_model.py
# requires (verify_fold_stamp refuses unstamped matrices), so without this
# module every fold would be ungradeable.
_PARAMS_PY = (Path(__file__).resolve().parent.parent / "sl_comparison" /
              "model_params.py")
_pspec = _ilu.spec_from_file_location("_sl_comparison_model_params",
                                      _PARAMS_PY)
_pmod = _ilu.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
write_model_params = _pmod.write_model_params
purge_stale_predictions = _pmod.purge_stale_predictions

# cal_metrics' 15 outputs, in evaluate_predictions_dict's key order. We keep
# ONLY these 12. auroc / f1 / aupr (indices 0-2) are DISCARDED: they are the
# same OR-collapsed edge-set quantity train.py already emits as auroc_any /
# f1_any / aupr_any, but computed by DIFFERENT formulas -- cal_metrics uses
# auc(recall, precision) (preprocess:1395, trapezoidal) vs
# average_precision_score, and max(2PR/(P+R)) in cal_metrics (unguarded in the
# SLMGAE-in-pytorch copy this process imports; the sl_comparison copy adds
# 1e-10 to the denominator) vs calculate_optimal_f1's 1e-8 (below). Emitting
# them under the same keys would silently replace siamese's headline numbers
# with different ones.
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


# AUPRG: ONE definition for siamese and the shared evaluator. Loaded BY FILE
# PATH, like evaluation.py / model_params.py above, so sl_comparison/ never
# shadows siamese_sl/ on sys.path. sl_comparison/auprg.py re-implements the
# reference prg.create_prg_curve + calc_auprg exactly; the integrator that
# lived here until 2026-09-21 kept only the maximum precision gain per recall
# gain and disagreed with the reference (labels [1, 0, 1] by descending score:
# +0.25 vs the reference's -0.25). There is deliberately NO local fallback: a
# second copy is how the two drifted apart.
_AUPRG_PY = (Path(__file__).resolve().parent.parent / "sl_comparison" /
             "auprg.py")
_aspec = _ilu.spec_from_file_location("_sl_comparison_auprg", _AUPRG_PY)
_amod = _ilu.module_from_spec(_aspec)
_aspec.loader.exec_module(_amod)


def _auprg(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the Precision-Recall-Gain curve (Flach & Kull, NeurIPS 2015).

    Baseline-anchored counterpart of AUPR (random ~0, perfect 1 at any
    prevalence). Delegates to sl_comparison/auprg.py, the same function the
    shared evaluator and the three baseline trainers use. NaN when prevalence
    is degenerate (no positives or no negatives).
    """
    return _amod.auprg(labels, scores)


def known_sl_seen_index(all_pos_shared, shared_to_full, train_pos,
                        answer_key) -> np.ndarray:
    """Pairs the self-scored retrieval pass suppresses (FULL index space).

    Mirrors sl_comparison/evaluate_model.seen_for: EVERY known SL pair except
    the answer key. cal_metrics writes -999999 over seen_index, and anything
    left out stays rankable as a NON-relevant item in a test gene's top-100,
    so leaving the val positives (and on cv3 the pairs in neither partition)
    unmasked would count genuine SL partners as misses, which the grader does
    for no model.

    all_pos_shared: (P, 2) shared-index pairs (<folds_dir>/all_pos.npy), or
                    None for a fold set built before that file existed (then
                    only train_pos is masked, as seen_for falls back).
    shared_to_full: (n_shared,) shared index -> full embedding row, -1 = none
                    (SLDataManager._shared_gene_map). The two orders differ
                    (int- vs string-sorted), so pairs are re-canonicalised.
    train_pos:      (T, 2) canonical full-index train positives (always kept).
    answer_key:     (K, 2) canonical full-index test positives being scored.

    Returns (S, 2) int64, canonical i<j, unique, disjoint from answer_key.
    """
    parts = [np.asarray(train_pos, dtype=np.int64).reshape(-1, 2)]
    if all_pos_shared is not None:
        a = np.asarray(all_pos_shared, dtype=np.int64).reshape(-1, 2)
        fa, fb = shared_to_full[a[:, 0]], shared_to_full[a[:, 1]]
        ok = (fa >= 0) & (fb >= 0)
        parts.append(
            np.stack([np.minimum(fa[ok], fb[ok]),
                      np.maximum(fa[ok], fb[ok])], axis=1))
    cand = np.concatenate(parts)
    cand = cand[cand[:, 0] < cand[:, 1]]
    cand = np.unique(cand, axis=0).reshape(-1, 2)
    keep = set(map(tuple, np.asarray(answer_key).reshape(-1, 2).tolist()))
    m = np.fromiter((tuple(p) not in keep for p in cand.tolist()),
                    dtype=bool,
                    count=len(cand))
    return cand[m]


def file_sha256(path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes (chunked: the .pt files are tens of MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


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

        # Selection is VAL-only and only the shared folds (--folds_dir) carry
        # a val split, so any other run is certain to stop in train_fold. Stop
        # HERE, before config.json is overwritten and train() purges this
        # directory's fold_*_predictions.npy, so a misconfigured rerun cannot
        # damage a finished run's outputs.
        if not getattr(args, "folds_dir", None):
            raise SystemExit(
                "no validation split: early stopping and checkpoint selection "
                "run on validation AUPRG only, which needs the shared folds "
                "(--folds_dir <folds built by prepare_folds.py --val_frac>).")
        if args.use_cell_lines and not args.neg_pairs_path:
            # SLDataManager raises the same, but only after config.json has
            # been overwritten; stop before it for the reason above.
            raise SystemExit(
                "--use_cell_lines requires --neg_pairs_path: the screened "
                "non-SL pairs (and their .sources.tsv sidecar) are the "
                "per-head negative labels.")
        if not args.use_cell_lines:
            # --kernel / --cell_lines define the 2x2 grid but act only on the
            # cell-line model; ignoring them here would train one model under
            # several names.
            if args.kernel != "pd" or args.cell_lines != "per_head":
                raise SystemExit(
                    f"--kernel {args.kernel} --cell_lines {args.cell_lines} "
                    f"need --use_cell_lines; the single-output SiameseSL has "
                    f"the pd kernel and one output (use --use_cell_lines "
                    f"--cell_lines pooled for a one-output model with "
                    f"--kernel bilinear).")
            # The two extra score terms are on by default but exist only in
            # the cell-line model: record them as off for a single-output run.
            args.gene_term = args.pair_features = False
            if not getattr(args, "input_bias", True):
                raise SystemExit(
                    "--no_input_bias needs --use_cell_lines: the "
                    "single-output SiameseSL always has its input offset.")
        if (getattr(args, "per_head_input_bias", False)
                and not getattr(args, "input_bias", True)):
            raise SystemExit(
                "--no_input_bias removes the input offset that "
                "--per_head_input_bias adds back per head: pass one of them.")

        # input_dim is resolved per-fold once that fold's preprocessing is
        # fit (variance-target PCA may pick a different k per fold). Reset it
        # BEFORE config.json is written, so the saved config never records a
        # width no model was built with.
        if args.input_dim is not None:
            print(f"Warning: --input_dim={args.input_dim} is ignored; "
                  f"per-fold input_dim is detected from the fold's "
                  f"preprocessed embedding matrix.")
        args.input_dim = None

        # A bad --fold_modalities (no such file, a fold outside the run's)
        # stops HERE, before config.json is overwritten, for the reason above.
        fold_modalities = _parse_fold_modalities(
            getattr(args, "fold_modalities", None), args.num_folds)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        # CONTENT fingerprint of the files this run trains on, in
        # --embeddings_paths order. The basenames and gene_order each
        # checkpoint stores cannot tell a REGENERATED file from the trained
        # one: run_generate_embeddings.sh deletes and rebuilds every .pt, and
        # the Word2Vec / node2vec types are not bit-for-bit reproducible
        # (generate_embeddings.py), so a same-universe rebuild keeps the name,
        # the generator_version and the gene order while changing every
        # vector — after which a stored per-fold PCA basis and these weights
        # would score different vectors with every existing check green.
        # Saved in each fold checkpoint and in results.json.
        self.emb_sha256 = [file_sha256(p) for p in args.embeddings_paths]

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
            pool_cell_lines=(args.use_cell_lines
                             and args.cell_lines == "pooled"),
            selection_frac=getattr(args, "selection_frac", 0.0),
            selection_seed=getattr(args, "selection_seed", 42),
            fold_modalities=fold_modalities,
        )
        # Head names the evaluator reports: the 8 cell-line heads, or one
        # "ANY" head (label = SL in any line) under --cell_lines pooled and in
        # single-output mode (the shared folds' own label).
        self.head_names = (("ANY", ) if (self.data_manager.pool_cell_lines
                                         or self.data_manager.fold_labels)
                           else tuple(cell_line_vocab.HEADS))

    def create_model(self) -> nn.Module:
        """Create the model."""
        if self.args.use_cell_lines:
            encoder_dims = self.args.encoder_dims or [
                self.args.hidden_dim, self.args.latent_dim
            ]
            pooled = self.data_manager.pool_cell_lines
            model = SiameseSLMultiCell(
                input_dim=self.args.input_dim,
                num_heads=1 if pooled else cell_line_vocab.NUM_HEADS,
                encoder_dims=encoder_dims,
                dropout=self.args.dropout,
                last_layer_bias=self.args.last_layer_bias,
                pd_epsilon=self.args.pd_epsilon,
                siamese_encoder_type=self.args.siamese_encoder_type,
                cell_line_dim=0 if pooled else self.args.cell_line_dim,
                per_head_input_bias=self.args.per_head_input_bias,
                per_head_temperature=self.args.per_head_temperature,
                inner_norm=self.args.inner_norm,
                use_layernorm=self.args.layernorm,
                raw_temperature=not self.args.log_temperature,
                per_head_last_layer_bias=self.args.per_head_last_layer_bias,
                kernel=self.args.kernel,
                gene_term=self.args.gene_term,
                pair_features=self.args.pair_features,
                input_bias=getattr(self.args, "input_bias", True),
            )
            return model.to(self.device)
        # Single-output mode. Only the factorising inner-product scorer can
        # build the dense all-pairs matrix the shared folds are graded on, so
        # it is the only one (--model_type accepts nothing else).
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
    ) -> float:
        """Train for one epoch.

        Every run reads the shared folds, whose examples carry a (label, mask)
        pair: 8 cell-line columns, or one (--cell_lines pooled, single-output
        mode) with mask 1.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
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
            # weighting (each observed cell weighted equally). With one
            # column and mask 1 this is the plain mean BCE.
            loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
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
        """Evaluate on a (label, mask) loader: every run's data format (see
        train_epoch); single-output mode is one column with mask 1."""
        return self._evaluate_multilabel(model, test_loader)

    @torch.no_grad()
    def _evaluate_multilabel(self, model: nn.Module, test_loader) -> dict:
        """Evaluation on (label, mask) data, three views.

        PRIMARY = STRATIFIED: pool every observed (pair, cell_line) cell into
        one long set and compute AUROC / AUPR / AUPRG / F1 on it. Stratified
        AUPRG (`auprg`) selects checkpoints only under --select_metric auprg
        (the published SLeuth rule).
        GLOBAL (`*_any`): collapse the head axis per pair (label = SL in any
        line; score = max over heads). `auprg_any` is the DEFAULT selection
        metric (--select_metric auprg_any), the outcome every benchmark model
        is selected and graded on. All views rank the LOGITS, so with
        --cell_lines pooled (one head) the stratified and global views are
        identical.
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
        # Every ranking metric below (AUROC / AUPR / AUPRG / max-F1) scores the
        # LOGITS: the sigmoid is monotone, but in float32 it saturates to 1.0
        # above |logit| ~ 16.6 and ties pairs the logits (and, up to the
        # float32 rounding of its shift, the graded dense export) keep apart.
        # Scoring logits everywhere also makes the pooled stratified view and
        # the *_any view identical under --cell_lines pooled.

        per_head_aupr, per_head_auroc, per_head_f1 = {}, [], []
        per_head_baseline, per_head_aupr_norm, per_head_auprg = {}, {}, {}
        aupr_list = []  # raw per-head AUPRs (macro-averaged for report)
        aupr_norm_list = []  # normalized per-head AUPRs (reported)
        auprg_list = []  # per-head AUPRG (macro diagnostic)
        baseline_list = []  # per-head prevalence of scored heads
        pooled_y, pooled_s = [], []
        for h, name in enumerate(self.head_names):
            sel = M[:, h] == 1
            if sel.sum() == 0:
                per_head_aupr[name] = float("nan")
                per_head_baseline[name] = float("nan")
                per_head_aupr_norm[name] = float("nan")
                per_head_auprg[name] = float("nan")
                continue
            yh, sh = Y[sel, h], L[sel, h]
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
        # (pair, head) cell is one row. Checkpoint selection runs on this
        # pooled set only under --select_metric auprg (the default selects on
        # auprg_any below). Prevalence-weighted (dense lines like K562
        # dominate); the per-cell-line macro below is the equal-weight-per-line
        # complement.
        if len(np.unique(pooled_y)) >= 2:
            strat_aupr = float(average_precision_score(pooled_y, pooled_s))
            strat_auroc = float(roc_auc_score(pooled_y, pooled_s))
            strat_f1 = float(calculate_optimal_f1(pooled_y, pooled_s))
        else:
            strat_aupr = strat_auroc = strat_f1 = float("nan")
        strat_baseline = (float(np.mean(pooled_y))
                          if pooled_y.size else float("nan"))
        strat_aupr_norm = _normalized_aupr(strat_aupr, strat_baseline)
        strat_auprg = _auprg(pooled_y, pooled_s)  # selects iff --select_metric auprg

        # --- GLOBAL (one-time headline): collapse the head axis per pair ------
        # "SL in ANY cell line": one row per pair.
        #   label = 1 if the pair is SL in any supervised line, else 0 (a pair
        #           both SL and non-SL across lines counts positive).
        #   score = max over ALL heads of the SL logit (see below).
        sup = M == 1
        has_pos = (sup & (Y == 1)).any(axis=1)
        has_neg = (sup & (Y == 0)).any(axis=1)
        agg_keep = has_pos | has_neg  # every pair has >=1 supervised head
        agg_y = has_pos[agg_keep].astype(np.float32)
        # max over all heads of the LOGIT from model.forward (sigmoid is
        # monotone, but float32 sigmoid saturates above |logit| ~ 16.6 and
        # would tie pairs the exported logits keep apart). This is the score
        # the dense export grades up to float32 rounding, NOT bit for bit: the
        # export comes from build_any_score_matrix (a blocked GEMM, ~1e-6
        # apart from this forward in the logit) and is then shifted twice,
        # S - S.min() + 1 over the whole num_genes matrix (_retrieval_metrics)
        # and again over the shared 8,844 subset (evaluate_model.py). Both
        # shifts are monotone but can merge near-tied values into ties. See
        # the selection comment in train_fold for why siamese still selects
        # on this value.
        agg_s = L[agg_keep].max(axis=1)
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
            # PRIMARY = stratified pooled (selects under --select_metric auprg).
            "auroc": strat_auroc,
            "aupr": strat_aupr,
            "auprg": strat_auprg,  # stratified AUPRG
            "aupr_norm": strat_aupr_norm,
            "aupr_baseline": strat_baseline,
            "f1": strat_f1,
            # GLOBAL (SL in any cell line, max over heads); auprg_any selects
            # under the default --select_metric auprg_any.
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
        # Through <name>.tmp and a rename: a write that dies part-way (a full
        # quota) left a truncated file under the name the shared grader reads
        # and collect_and_regrade_siamese.sh counts, and train_fold's banner
        # for a missing export stayed silent. (An open file: np.save would
        # append ".npy" to the temporary name.)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as f:
                # np.save gets an object with nothing but .write(), not f
                # itself: handed a real file it calls ndarray.tofile, which
                # writes through a C stream of its own and never checks that
                # stream's fclose(). A write error in the last buffer, or one
                # the filesystem reports at that close, then raised nothing
                # and a short file took the graded name ("Saved", no banner).
                # This way every byte goes through f, and a failed write,
                # flush or close raises here.
                np.save(types.SimpleNamespace(write=f.write), out)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
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
        cal_metrics' ranking half (preprocess_benchmarking_paper.py:1399-1455
        in the SLMGAE-in-pytorch copy this process imports, 1407-1463 in the
        sl_comparison copy) argsorts each test gene's WHOLE row over every
        gene in the universe. The preprocess:<line> references below are to
        the SLMGAE-in-pytorch copy too.

        UNIVERSE: siamese's own full embedding gene list (n = num_genes;
        19,397 with the protein-coding + benchmark universe that
        run_generate_embeddings.sh STEP 0b builds), a SUPERSET of the shared
        8,844 of sl_comparison/folds/genes.txt. The ~10.5k extra genes are in
        no benchmark pair, so they are never a scored row, but their columns
        carry real scores and CAN enter a row's top-k. The self-scored @k
        values computed HERE therefore rank against a larger candidate set
        than the peers' and are not comparable to them (collect_siamese.py
        attaches that caveat from rank_universe_n). The comparable numbers come
        from the 8,844 x 8,844 matrix _save_shared_prediction_matrix gathers
        from S by Entrez id (the two gene lists use DIFFERENT sort orders:
        siamese string-sorted '10','100','10000', shared int-sorted
        '2','9','10'), which sl_comparison/evaluate_model.py grades. The
        suppressed (seen) set follows evaluate_model.seen_for (every known SL
        pair except the answer key; known_sl_seen_index), so the universe is
        the one convention that differs.

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
        # space), never ml_pairs_p. full_to_p is NOT the identity with the
        # genome-wide universe (num_p_genes, the genes in labelled pairs, is at
        # most 8,844 while num_genes is 19,397), so a P-space/full-space mix-up
        # would address the wrong rows; this guard checks the contract, and the
        # print below reports the live sizes (A14 would catch it too).
        _full_rows = set(map(tuple, dm.ml_pairs_full.tolist()))
        if not all(tuple(p) in _full_rows for p in te_pairs[:2000].tolist()):
            raise RuntimeError("fold pairs are not rows of ml_pairs_full -- "
                               "wrong gene index space")
        _np_genes = int(getattr(dm, "num_p_genes", -1))
        _space = ("P-space == full space, so a P/full mix-up would be SILENT"
                  if _np_genes == n else "P-space differs from full space")
        print(f"  [retrieval] index space: n={n} num_genes={dm.num_genes} "
              f"num_p_genes={_np_genes} ({_space})")

        # ====== OR-collapse, byte-identical to _evaluate_multilabel's GLOBAL
        # view (has_pos / has_neg / agg_keep) ================================
        def _any(part, val):
            Y = np.asarray(fold_data[part]["labels"])
            M = np.asarray(fold_data[part]["mask"])
            return ((M == 1) & (Y == val)).any(axis=1)

        te_pos, te_neg = _any("test", 1), _any("test", 0)
        # _evaluate_multilabel keeps agg_keep = has_pos | has_neg; mirror that
        # literally rather than relying on ~has_pos (equivalent only because
        # SLDataManager._setup_multilabel drops mask-empty pairs, its
        # `if msk.sum() == 0`).
        if te_pairs.size and not bool((te_pos | te_neg).all()):
            raise RuntimeError(
                "test pair with no supervised head -- the *_any "
                "view (_evaluate_multilabel's agg_keep) would exclude it")
        pos_index = te_pairs[te_pos]
        neg_index = te_pairs[te_neg & ~te_pos]

        if len(pos_index) == 0:
            print("  [retrieval] no OR-collapsed test positives -> skipped")
            return {}

        # Suppression set = evaluate_model.seen_for's: every known SL pair
        # (<folds_dir>/all_pos.npy) except this pass's answer key. The val
        # positives, and on cv3 the pairs with exactly one held-out gene
        # (in neither train nor test), are genuine SL; left rankable they
        # would count as misses in a test gene's top-100, which the grader
        # does for no model. Train positives are always included.
        tr_pairs = np.asarray(fold_data["train"]["pairs"], dtype=np.int64)
        _all_pos, _s2f = None, None
        _ap = Path(dm.folds_dir) / "all_pos.npy" if dm.folds_dir else None
        if _ap is not None and _ap.exists():
            _all_pos = np.load(_ap)
            _, _s2f = dm._shared_gene_map()
        else:
            print(f"  [retrieval] WARNING: <folds_dir>/all_pos.npy missing "
                  f"({dm.folds_dir}) -- masking only "
                  f"train positives (as evaluate_model.seen_for falls back); "
                  f"known SL pairs outside this fold deflate every @k.")
        seen_index = known_sl_seen_index(_all_pos, _s2f,
                                         tr_pairs[_any("train", 1)],
                                         pos_index)
        # A9: seen_index must hold POSITIVES only. cal_metrics sets seen
        # entries to -999999, so masking a labelled NEGATIVE would delete a
        # hard distractor the grader keeps for every model (seen_for masks
        # positives only) -- silently inflating siamese's @k.
        _negs = set(map(tuple, neg_index.tolist()))
        for _part in ("train", "val"):
            if _part in fold_data:
                _pp = np.asarray(fold_data[_part]["pairs"], dtype=np.int64)
                _negs |= set(map(tuple, _pp[_any(_part, 0)
                                            & ~_any(_part, 1)].tolist()))
        _bad = _negs & set(map(tuple, seen_index.tolist()))
        if _bad:
            raise RuntimeError(f"seen_index contains {len(_bad)} labelled "
                               f"negatives")
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
        # 1e-3 absolute, widened to 1e-5 of the logit scale above |logit| 100:
        # without LayerNorm and with a raw tau the logits are unbounded, and
        # at |logit| ~ 1e4 one float32 ULP is already ~1e-3, so a fixed bound
        # would fail a correct matrix (and lose this fold's export). A
        # misalignment is O(spread of the logits), far above either bound.
        tol = max(1e-3, 1e-5 * float(np.abs(ref_logit).max()))
        if err > tol:
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
        # increasing affine map preserves the ranking (in float32 it can merge
        # values closer than ~ulp(1 + range) into ties; it never inverts them).
        #
        # MEASURED (600-gene synthetic in the real trained-parameter regime,
        # scratchpad/probe_retrieval.py): SHIFTED == sigmoid on all 12 metrics
        # to 1e-12; RAW logits differ (ndcg@20 0.029924 vs 0.029653, and the
        # 0-diagonal reached the top-10 of 1/398 scored rows); and sigmoid
        # DRIFTS once float32 saturates -- at 2.9% saturated entries ndcg@10
        # inflated 0.01724 -> 0.02241 (+30%) because argsort breaks exact ties
        # by gene index. The temperature is an unclamped learned parameter
        # (SiameseSLMultiCell.__init__: `tau` by default, `log_temperature`
        # with --log_temperature), so saturation is not hypothetical.
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
        # (K=16, M=N=num_genes) while auroc_any comes from zz.sum(dim=1) over
        # a (B*H, 16) batch (SiameseSLMultiCell.forward) -- different
        # reduction shapes, ~1e-6-scale logit differences, and with ~2.7e8
        # comparisons a few hundred near-tie flips already breach 1e-6.
        if (auroc_any_ref is not None and np.isfinite(auroc_any_ref)
                and np.isfinite(m["auroc"])):
            d = abs(float(m["auroc"]) - float(auroc_any_ref))
            if d > 1e-2:
                # The exported matrix IS this S (the affine shift and the
                # export above both precede cal_metrics), so a broken identity
                # condemns the FILE, not just these columns: evaluate_model.py
                # recomputes every ranking column from the .npy and drops
                # siamese's own, so leaving it on disk would publish numbers
                # the model disagrees with, with nothing but a traceback in the
                # training log. Remove it, as train_fold removes a previous
                # run's matrix before training — a fold that cannot be trusted
                # must be MISSING, never stale. The handler in train_fold then
                # prints the "not on disk" banner and
                # collect_and_regrade_siamese.sh refuses this CV's fold set.
                (self.output_dir / f"fold_{fold_data['fold']}_"
                 f"{self.args.cv_type}_predictions.npy").unlink(
                     missing_ok=True)
                raise RuntimeError(
                    f"cal_metrics AUROC {m['auroc']:.6f} vs auroc_any "
                    f"{auroc_any_ref:.6f} (delta {d:.2e}); this fold's "
                    f"prediction matrix was REMOVED")
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
              f"{len(seen_index):,} known SL pairs suppressed | "
              f"fwd-check {err:.2e} | "
              f"NDCG@10 {out['ndcg@10_any']:.4f} "
              f"NDCG@50 {out['ndcg@50_any']:.4f} "
              f"R@50 {out['recall@50_any']:.4f} "
              f"MAP@50 {out['map@50_any']:.4f}")
        return out

    def train_fold(self, fold_data: dict) -> dict:
        """Train a single CV fold."""
        fold_idx = fold_data["fold"]
        # Seed per fold (2026-11-13): seeded once per run, fold k's init,
        # shuffles and dropout masks depended on how many epochs folds
        # 0..k-1 ran (early stopping), so no fold could be reproduced alone
        # and a change in one fold moved every later fold's random stream.
        set_seed(int(self.args.seed) + int(fold_idx))
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
        # own folds have none, and such a run stops just below (there is no
        # select-on-test fallback any more).
        val_loader = create_val_dataloader(
            embeddings=fold_embeddings,
            fold_data=fold_data,
            batch_size=self.args.batch_size,
            select_metric=self.args.select_metric,
        )
        # Early stopping and checkpoint selection are VAL-only (as for SLMGAE,
        # SLGNN and NSF4SL): a fold without a val split is an error, never a
        # silent fallback to selecting on TEST.
        if val_loader is None:
            raise SystemExit(
                f"fold {fold_idx}: no validation split. Early stopping and "
                f"checkpoint selection run on validation AUPRG only, so a "
                f"run needs the shared folds: --folds_dir <folds built by "
                f"prepare_folds.py --val_frac>. Siamese's own folds carry no "
                f"val split.")
        sel_loader = val_loader
        sel_split = "val"

        # Create model and optimizer
        model = self.create_model()
        # selection_regime travels with the predictions (model_params.json ->
        # evaluate_model.build_summary -> compare.py), so the table itself shows
        # which rows were val-selected. Nothing else could distinguish a row
        # whose checkpoint was picked on TEST from one picked honestly.
        # seed: compare.py's per-row training-seed record (paper D3).
        write_model_params(self.output_dir,
                           model,
                           folds_dir=getattr(self.args, "folds_dir", None),
                           cv_type=getattr(self.args, "cv_type", None),
                           selection_regime=sel_split,
                           selection_metric=self.args.select_metric,
                           seed=getattr(self.args, "seed", None))

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
        # weights — excluded so --l1_lambdas keeps matching the encoder. The
        # per-head offsets (head_input_bias, head_proj_bias; 2-D, one row per
        # head) are biases and are never penalised, like input_bias. The
        # --kernel bilinear matrix (kernel_A) and the --pair_features MLP
        # (pair_fc1 / pair_fc2) are part of the scorer, not the encoder, and
        # are not penalised either (--gene_term's gene_w is 1-D).
        self._l1_map = {}
        if self.args.l1_lambdas:
            weight_params = [(n, p) for n, p in model.named_parameters()
                             if p.dim() >= 2
                             and not n.startswith(("cell_", "kernel_",
                                                   "pair_"))
                             and not n.endswith("bias")]
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

        # Training loop. Best-checkpoint metric (VALIDATION split only):
        #   cell-line mode  -> --select_metric: "auprg_any" (default) = AUPRG of
        #                      the pair-level OR label with the max-over-heads
        #                      logit, the outcome every benchmark model is
        #                      selected and graded on; "auprg" = the stratified
        #                      AUPRG pooled over (pair, head) cells (the
        #                      published SLeuth rule).
        #   single-output   -> --select_metric too; with one column the two
        #                      metrics are the same number.
        # Early stopping shares the same metric — patience increments only when
        # it fails to improve. A NaN selection value (degenerate split) never
        # beats best_sel because NaN > x is False, so the save is skipped.
        # With --cell_lines pooled (and in single-output mode) the two metrics
        # are identical (one head, the same labels, both ranking logits).
        #
        # ONE DEVIATION FROM THE PEERS (known, not fixed). SLMGAE, SLGNN and
        # NSF4SL score their val pairs on the exported matrix after the
        # grader's float32 shift (score - min + 1), i.e. on the exact values
        # evaluate_model.py would read. Siamese scores them on the forward
        # logits (_evaluate_multilabel), which differ from the graded values
        # by the GEMM-vs-forward reduction order (~1e-6 in the logit; A14 in
        # _retrieval_metrics bounds it) and by near-ties the two float32
        # shifts merge. Measured on synthetic val sets of the real size
        # (1,773 pos / 8,536 neg): the val AUPRG moves by a few 1e-6 at most
        # (<= 2.5e-6), so two evaluations could only swap order when their
        # AUPRGs are that close (evaluations are EVAL_INTERVAL epochs apart).
        # Reproducing the graded values exactly needs the full num_genes x
        # num_genes build at EVERY evaluation (the first shift's minimum and
        # each entry's GEMM rounding both depend on it: ~4 GiB transient at
        # 19,397 genes), and it would break the one-output identity
        # auprg == auprg_any that arch_signature.py relies on, so the
        # difference is documented instead.
        sel_metric = self.args.select_metric
        best_sel = float("-inf")
        best_epoch = 0
        patience_counter = 0
        best_metrics = {}
        n_evals = 0  # how many periodic evaluations actually ran

        pbar = tqdm(range(self.args.epochs), desc=f"Fold {fold_idx + 1}")
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer)
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
                # Show the metric that selects (auprg_any by default) too: the
                # cell-line block below prints only when it improves.
                postfix.setdefault(sel_metric, f"{metrics[sel_metric]:.4f}")
                pbar.set_postfix(postfix)

                if not np.isfinite(metrics[sel_metric]):
                    # A NaN val AUPRG can never win (NaN > x is False) and
                    # counts toward patience; log it so it is not silent.
                    print(f"\n  WARNING: epoch {epoch + 1}: val {sel_metric} "
                          f"is {metrics[sel_metric]} — not selectable")
                if metrics[sel_metric] > best_sel:
                    best_sel = metrics[sel_metric]
                    best_epoch = epoch + 1
                    best_metrics = metrics.copy()
                    patience_counter = 0
                    if len(self.head_names) > 1:
                        ph = metrics.get("per_head_aupr", {})
                        phb = metrics.get("per_head_baseline", {})
                        phn = metrics.get("per_head_aupr_norm", {})
                        phg = metrics.get("per_head_auprg", {})
                        nan = float("nan")
                        # Mark the row whose AUPRG --select_metric selects on.
                        sel_mark = {k: ("   <- AUPRG selects"
                                        if sel_metric == k else "")
                                    for k in ("auprg", "auprg_any")}
                        print("\n  cell-line metrics  [baseline | raw AUPR | "
                              "norm AUPR | AUPRG]")
                        print(f"    strat : {metrics['aupr_baseline']:.3f} | "
                              f"{metrics['aupr']:.3f} | "
                              f"{metrics['aupr_norm']:.3f} | "
                              f"{metrics['auprg']:.3f}{sel_mark['auprg']}")
                        print(f"    any   : {metrics['baseline_any']:.3f} | "
                              f"{metrics['aupr_any']:.3f} | "
                              f"{metrics['aupr_any_norm']:.3f} | "
                              f"{metrics['auprg_any']:.3f}   "
                              f"(AUROC {metrics['auroc_any']:.3f})"
                              f"{sel_mark['auprg_any']}")
                        print(
                            f"    macro : {metrics['aupr_baseline_macro']:.3f}"
                            f" | {metrics['aupr_macro']:.3f} | "
                            f"{metrics['aupr_norm_macro']:.3f} | "
                            f"{metrics['auprg_macro']:.3f}   (per-line avg)")
                        for n in self.head_names:
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
                            # content hash of those files (see __init__)
                            "embeddings_sha256":
                            list(self.emb_sha256),
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
        # Either way no checkpoint was chosen on val, and exporting an arbitrary
        # final-epoch model would report numbers nothing selected — stop, as the
        # SLMGAE / SLGNN / NSF4SL trainers do.
        if not best_metrics:
            if n_evals == 0:
                raise RuntimeError(
                    f"fold {fold_idx}: no validation evaluation ran "
                    f"(epochs={self.args.epochs} < "
                    f"eval_interval={self.args.eval_interval}); no checkpoint "
                    f"could be selected")
            raise RuntimeError(
                f"fold {fold_idx}: all {n_evals} validation evaluation(s) "
                f"returned an undefined {sel_metric}; no checkpoint could be "
                f"selected on val")

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
        # cell-line mode cell_emb.weight (absent under --cell_lines pooled),
        # the per-head offsets head_input_bias / head_proj_bias and the
        # --kernel bilinear kernel_A and the --pair_features pair_fc weights
        # are 2-D and counted here, but they are
        # deliberately excluded from the proximal L1 (see the _l1_map
        # construction above). So this sparsity figure is diluted by up to six
        # dense tensors that L1 was never applied to — read it as "2-D
        # parameter sparsity", not "sparsity of what L1 targeted".
        wt_total = sum(p.numel() for p in params if p.dim() >= 2)
        wt_nonzero = sum(int(p.ne(0).sum()) for p in params if p.dim() >= 2)
        wt_sparsity = 100 * (1 - wt_nonzero / wt_total) if wt_total > 0 else 0

        # With a shared val split, best_metrics so far are VAL metrics — the
        # selection criterion, not a result. Restore the selected checkpoint and
        # score TEST exactly once. Test then never influenced training, matching
        # SLGNN/train_slgnn.py and NSF4SL/train_nsf4sl.py (train_fold).
        # Both val AUPRG views are kept at the selected checkpoint whatever
        # --select_metric is, so val_auprg_mean and val_auprg_any_mean both
        # exist (run_best_per_category_combo.sh and collect_siamese.py rank on
        # val_<select_metric>_mean and fall back to val_auprg_mean).
        # Always the val path: a fold with no val split, or in which no val
        # evaluation was ever selectable, has already raised above.
        val_keys = ("auprg", "auprg_any")
        sel_value = best_metrics.get(sel_metric, float("nan"))
        val_snap = {k: best_metrics.get(k, float("nan")) for k in val_keys}
        model.load_state_dict(sd)
        best_metrics = self.evaluate(model, test_loader)
        for k, v in val_snap.items():
            best_metrics[f"val_{k}"] = float(v)
        best_metrics[f"val_{sel_metric}"] = float(sel_value)
        print(f"\n  selected on {sel_split} ({sel_metric}="
              f"{sel_value:.4f} @ epoch {best_epoch}); test scored once")
        # Modality-selection split (--selection_frac): carved from TRAIN, never
        # trained on, scored once here with the same selected checkpoint. Its
        # sel_* keys are what per-fold modality selection ranks; test stays
        # untouched by selection.
        if "sel" in fold_data:
            sel_loader_ms = create_val_dataloader(
                embeddings=fold_embeddings,
                fold_data={"val": fold_data["sel"]},
                batch_size=self.args.batch_size,
                select_metric=self.args.select_metric,
            )
            sel_m = self.evaluate(model, sel_loader_ms)
            for k in ("auprg", "auprg_any", "auroc_any", "aupr_any"):
                if k in sel_m:
                    best_metrics[f"sel_{k}"] = float(sel_m[k])
            best_metrics["sel_n_pairs"] = int(len(fold_data["sel"]["pairs"]))
            best_metrics["sel_digest"] = fold_data["sel_digest"]
            np.save(self.output_dir /
                    f"selection_split_{self.args.cv_type}_fold{fold_idx}.npy",
                    np.asarray(fold_data["sel"]["pairs"], dtype=np.int64))
            print(f"  selection split ({best_metrics['sel_n_pairs']:,} pairs, "
                  f"{fold_data['sel_digest'][:12]}): "
                  f"{sel_metric}={best_metrics.get('sel_' + sel_metric, float('nan')):.4f}")

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

        # The fold id and this fold's encoder input width (the post-concat PCA
        # k, resolved per fold) travel with the fold's metrics, so
        # results.json alone gives the per-split widths and parameter counts
        # (model_params.json keeps only the last fold's).
        best_metrics["fold"] = int(fold_idx)
        best_metrics["input_dim"] = int(self.args.input_dim)
        # Per-fold modality selection: the embeddings this fold trained on.
        _midx = fold_transform.get("modality_idx")
        best_metrics["modalities"] = [
            Path(self.args.embeddings_paths[i]).stem.replace("all_genes_", "")
            for i in (range(len(self.args.embeddings_paths))
                      if _midx is None else _midx)]
        best_metrics["total_params"] = total_params
        best_metrics["nonzero_params"] = nonzero_params
        best_metrics["weight_sparsity"] = wt_sparsity
        best_metrics["layer_sparsities"] = layer_sparsities

        # --- Retrieval metrics on the BEST checkpoint (ONCE per fold) -------
        # `model` already holds the val-selected weights (load_state_dict a few
        # dozen lines above; selection is val-only, so there is no other path).
        # Reload anyway rather than rely on that: the point is that
        # ndcg/recall/precision/map must describe the same checkpoint as
        # best_metrics / auroc_any, whichever path got us here. Mirrors the peer
        # implementation in SLGNN's trainer.
        #
        # NEVER FATAL: train() writes results.json only after ALL folds finish,
        # so an exception on fold 3 would throw away folds 0-2 including the
        # aupr_any/auroc_any that already work. Any check up to
        # _save_shared_prediction_matrix, AND the A15 AUROC identity after it
        # (which unlinks the export before raising, because that check
        # condemns the exported values themselves), leaves this fold's graded
        # matrix missing, so collect_and_regrade_siamese.sh refuses the partial
        # fold set and the CV's row is not published. Only a failure that is
        # not about the matrix keeps the export and degrades to "-" in
        # siamese's self-scored retrieval columns; the banner below says which
        # of the two happened.
        # Both inner-product scorers (SiameseSLMultiCell, and SiameseSL in
        # single-output mode) expose encode_head/temperature for the builder.
        if isinstance(model, (SiameseSLMultiCell, SiameseSL)):
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
                _mat = (self.output_dir / f"fold_{fold_idx}_"
                        f"{self.args.cv_type}_predictions.npy")
                if self.data_manager.folds_dir and not _mat.exists():
                    print("!" * 70)
                    print(f"!! {_mat.name} is not on disk: the failure came "
                          f"before the shared-space export, or the export was "
                          f"removed because it failed the AUROC-identity "
                          f"check. Either way "
                          f"collect_and_regrade_siamese.sh will refuse this "
                          f"CV's partial fold set and its row will not be "
                          f"published. Fix the cause and rerun this CV.")
                    print("!" * 70)

        return best_metrics

    def train(self):
        """Run full cross-validation training."""
        print("=" * 60)
        print(f"Siamese SL Training")
        # The class create_model builds (--model_type only names the family).
        print("Model: " + (
            f"SiameseSLMultiCell (kernel {self.args.kernel}, cell_lines "
            f"{self.args.cell_lines}"
            f"{', gene_term' if self.args.gene_term else ''}"
            f"{', pair_features' if self.args.pair_features else ''})"
            if self.args.use_cell_lines else
            "SiameseSL (single output)"))
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
        # Fold means, not truncated (int() turned 23869.8 into 23869): the
        # post-concat PCA width, and with it the parameter count, differs by
        # fold. fold_metrics[k] keeps each fold's exact count.
        avg_nonzero = round(float(
            np.mean([m.get("nonzero_params", 0) for m in all_metrics])), 1)
        avg_total = round(float(
            np.mean([m.get("total_params", 0) for m in all_metrics])), 1)

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
            # content hash of config["embeddings_paths"], same order, so the
            # inputs can be checked without loading a checkpoint (see __init__)
            "embeddings_sha256": list(self.emb_sha256),
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

        # Summarize the three views across folds (every run has them; with one
        # column they coincide). The primary `aupr`/`auroc`/`auprg` above are
        # STRATIFIED (pooled over observed cells); these add the per-cell-line
        # macro diagnostic and the global "SL in any line" one-time headline.
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
        if any("val_auprg_any" in m for m in all_metrics):
            agg_keys.append("val_auprg_any")
        # --selection_frac: the modality-selection split's scores.
        for _k in ("sel_auprg", "sel_auprg_any", "sel_auroc_any",
                   "sel_aupr_any"):
            if any(_k in m for m in all_metrics):
                agg_keys.append(_k)
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
        # Label the view --select_metric selected checkpoints on.
        _sel = getattr(self.args, "select_metric", "auprg_any")
        _basis = {k: ("; SELECTION basis" if _sel == k else "")
                  for k in ("auprg", "auprg_any")}
        print("\nCell-line metrics (mean over folds):")
        print(f"  STRATIFIED (pooled per-cell-line obs"
              f"{_basis['auprg']}):")
        print(
            f"    baseline(pi) {s['aupr_baseline_mean']:.4f}  "
            f"AUPR {s['aupr_mean']:.4f}  normAUPR {s['aupr_norm_mean']:.4f}"
            f"  AUPRG {s['auprg_mean']:.4f}  AUROC {s['auroc_mean']:.4f}")
        print(f"  GLOBAL (SL in any cell line; headline"
              f"{_basis['auprg_any']}):")
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


def _parse_fold_modalities(value, num_folds):
    """--fold_modalities: JSON text or a JSON file -> {fold: [positions]}.
    Exactly the folds 0..num_folds-1 (positions are checked against
    embeddings_paths by SLDataManager, which does not know num_folds)."""
    if value in (None, ""):
        return None
    import json as _json
    txt = value
    if not value.lstrip().startswith("{"):
        # Not a JSON object: a file path, which must exist (json.loads of
        # the path would fail with an opaque JSONDecodeError).
        if not Path(value).is_file():
            raise SystemExit(f"--fold_modalities: {value!r} is neither a "
                             f"JSON object nor an existing file (file not "
                             f"found)")
        txt = Path(value).read_text()
    try:
        raw = _json.loads(txt)
        out = {int(k): [int(i) for i in v] for k, v in raw.items()}
    except (ValueError, TypeError, AttributeError) as e:
        raise SystemExit(f"--fold_modalities: not a JSON object "
                         f"{{fold: [positions]}} ({type(e).__name__}: {e})")
    missing = [k for k in range(num_folds) if k not in out]
    if missing:
        raise SystemExit(f"--fold_modalities has no entry for fold(s) "
                         f"{missing} (need 0..{num_folds - 1})")
    extra = sorted(k for k in out if not 0 <= k < num_folds)
    if extra:
        raise SystemExit(f"--fold_modalities has fold(s) {extra} outside "
                         f"0..{num_folds - 1} (--num_folds {num_folds})")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese network for SL prediction",
        # An abbreviation such as `--kern bilinear` would resolve to --kernel
        # and, being LAST on the command line, silently override the --kernel
        # the variant sets — training all four 2x2 variants identically.
        # config.conf scans $ARCH_ARGS for exactly that (its comment concedes
        # the gap); this closes it for MODEL_ARGS and hand-run calls too.
        # Every SLURM caller passes full option names, so nothing changes.
        allow_abbrev=False)

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
        help="Screened NON-SL pairs file (gene1 TAB gene2), e.g. "
        "../data/SL_SynLethDB_experimental_negatives.txt. REQUIRED with "
        "--use_cell_lines: with its '.sources.tsv' sidecar it supplies the "
        "per-head negative labels (the pairs are not restricted to the SL "
        "gene universe). Not used in single-output mode, whose labels come "
        "from --folds_dir.")
    parser.add_argument(
        "--use_cell_lines",
        action="store_true",
        help="Enable masked multi-label cell-line conditioning. The unit "
        "becomes a gene pair with a per-head (label, mask) over "
        "[K562, JURKAT, A549, HELA, A375, 293T, PC9, OTHER]; cell embeddings "
        "are concatenated onto the gene features (SiameseSLMultiCell), and the "
        "model predicts SL per head. Reads cell context from the "
        "'.sources.tsv' sidecars next to --sl_path / --neg_pairs_path; "
        "requires --neg_pairs_path. Checkpoint selection uses --select_metric "
        "on the validation split (default auprg_any, the pair-level OR label; "
        "stratified, global 'any-line' and per-line views are all reported). "
        "Without it train.py trains the single-output SiameseSL on the "
        "shared folds' own labels (needs --folds_dir).")
    parser.add_argument(
        "--folds_dir",
        type=str,
        default=None,
        help="Use the shared canonical CV folds from this directory "
        "(e.g. ../sl_comparison/folds) — the fair four-model comparison path "
        "and the only one train.py runs (only these folds carry a val "
        "split). Train/val/test "
        "division and gene index space become identical to SLMGAE / SLGNN / "
        "NSF4SL; with --use_cell_lines the per-head labels stay siamese's, so "
        "cell-line conditioning is unaffected; without it (single-output "
        "mode) the labels are the folds' own pos/neg and --neg_pairs_path is "
        "not used. Checkpoint selection runs on the shared VAL split only "
        "(REQUIRED: train.py refuses to run without it), and "
        "each fold writes fold_<k>_<cv>_predictions.npy in shared gene order "
        "so sl_comparison/evaluate_model.py grades it identically to the "
        "peers.")
    parser.add_argument(
        "--cell_line_dim",
        type=int,
        default=8,
        help="Cell-line embedding dimension (only used with --use_cell_lines; "
        "ignored under --cell_lines pooled, which has no cell-line embedding)."
    )
    parser.add_argument(
        "--per_head_input_bias",
        action="store_true",
        help="One input offset per cell-line head, shape (heads, input_dim + "
        "cell_line_dim), replacing the encoder's shared input_bias. Only used "
        "with --use_cell_lines.")
    parser.add_argument(
        "--no_input_bias",
        dest="input_bias",
        action="store_false",
        default=True,
        help="Drop the encoder's shared learnable input offset a (in = x + a; "
        "kept by default). With no hidden layer (one --encoder_dims entry) "
        "z = W (x + a) carries the per-gene term <W a, W x>, a degree prior; "
        "the bilinear probe (slurm/run_probe.sh) passes this flag so its "
        "logit is exactly tau * (W x_i)^T (W x_j) + b. Refused with "
        "--per_head_input_bias. Only used with --use_cell_lines.")
    parser.add_argument(
        "--per_head_temperature",
        action="store_true",
        help="One temperature per cell-line head (shape (heads,)) instead of "
        "a shared one. Only used with --use_cell_lines.")
    parser.add_argument(
        "--selection_frac",
        type=float,
        default=0.0,
        help="Hold out this fraction of each fold's TRAINING pairs (uniform "
        "random pairs, at every CV type) as a modality-selection split: never "
        "trained on, scored once with the val-selected checkpoint (sel_* keys "
        "in results.json; sel_<select_metric>_mean is what per-fold modality "
        "selection ranks). 0 (default) = off. Every run on the same folds with "
        "the same --selection_seed holds out the same pairs (digest recorded "
        "per fold as sel_digest).")
    parser.add_argument(
        "--fold_modalities",
        default=None,
        help="Per-fold modality sets (per-fold modality selection): a JSON "
        "object, or a path to a JSON file, mapping every fold id to the "
        "positions in --embeddings_paths that fold trains on, e.g. "
        "'{\"0\": [0, 2, 3], \"1\": [0, 1, 3], ...}'. Each checkpoint's "
        "preprocessing_transform records its fold's positions, so inference "
        "(eval_finetuning, export, cell-line ablation) loads the run's full "
        "embeddings_paths and applies each fold's own subset. Default: every "
        "fold uses every modality.")
    parser.add_argument(
        "--all_modalities",
        action="store_true",
        help="Record that --embeddings_paths is the WHOLE embedding catalog, "
        "chosen by no modality selection (slurm/run_allmod_combo.sh, the "
        "all-modalities combo). A label in config.json and an arch_signature "
        "key: it changes no computation, but keeps the model from ever being "
        "taken for a selected combo. Refused with --fold_modalities.")
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Seed of the --selection_frac split (independent of --seed).")
    parser.add_argument(
        "--select_metric",
        choices=["auprg_any", "auprg"],
        default="auprg_any",
        help="Validation metric for early stopping and checkpoint selection. "
        "With one output (--cell_lines pooled, or single-output mode) the two "
        "choices are the same number. 'auprg_any' (default): AUPRG of the pair-level OR "
        "label with the max-over-heads score, the same outcome the other "
        "benchmark models are selected on. 'auprg': stratified AUPRG pooled "
        "over (pair, head) cells (the published SLeuth rule).")
    parser.add_argument(
        "--cell_lines",
        choices=["per_head", "pooled"],
        default="per_head",
        help="'per_head' (default): 8 cell-line heads with a cell-line "
        "embedding, masked multi-label loss. 'pooled': no cell-line embedding, "
        "one head trained on the OR-collapsed label (SL in any line), the "
        "outcome the other benchmark models use. Same pair universe, folds "
        "and selection. Only used with --use_cell_lines.")
    parser.add_argument(
        "--kernel",
        choices=["pd", "bilinear"],
        default="pd",
        help="Pair score. 'pd' (default): z_iᵀ z_j, the positive-definite "
        "SLeuth kernel. 'bilinear': the symmetrised z_iᵀ A z_j = "
        "(s_ij + s_ji)/2, A learnable and initialised to I, so the score "
        "need not be positive definite. Both add eps * h_iᵀ h_j. Only used "
        "with --use_cell_lines.")
    parser.add_argument(
        "--gene_term",
        dest="gene_term",
        action="store_true",
        default=True,
        help="Add a per-gene propensity g(h_i) + g(h_j) to each head's logit, "
        "g(h) = <gene_w, h> on the hidden h (the projection's input), after "
        "the temperature. Zero-initialised, never L1-penalised. ON by default "
        "since 2026-10-05 (--no_gene_term removes it). Only used with "
        "--use_cell_lines.")
    parser.add_argument(
        "--no_gene_term",
        dest="gene_term",
        action="store_false",
        help="Drop the per-gene propensity term (see --gene_term).")
    parser.add_argument(
        "--pair_features",
        dest="pair_features",
        action="store_true",
        default=True,
        help="Add a symmetric pair term to each head's logit: a "
        "one-hidden-layer MLP (width = latent dim) of [|z_i - z_j|, "
        "z_i * z_j], after the temperature. Its output layer is "
        "zero-initialised; never L1-penalised. It does not factorise over the "
        "two genes, so the dense all-pairs export evaluates it pair by pair "
        "(slower). ON by default since 2026-10-05 (--no_pair_features removes "
        "it). Only used with --use_cell_lines.")
    parser.add_argument(
        "--no_pair_features",
        dest="pair_features",
        action="store_false",
        help="Drop the pair-feature term (see --pair_features).")
    parser.add_argument(
        "--no_per_head_last_layer_bias",
        dest="per_head_last_layer_bias",
        action="store_false",
        default=True,
        help="Drop the per-head projection bias c_h (z_h = W h + c_h, one "
        "(latent_dim,) row per cell-line head). On by default; when on it "
        "replaces the encoder's shared projection bias, whatever "
        "--no-last-layer-bias says. The published model has neither; its "
        "recipe is --layernorm --log_temperature --no_per_head_last_layer_bias "
        "--select_metric auprg --no_gene_term --no_pair_features (plus "
        "--no-last-layer-bias, seven --encoder_dims of 16 and "
        "--post_pca_variance 0.8 on the combo). Only used with "
        "--use_cell_lines.")
    parser.add_argument(
        "--log_temperature",
        action="store_true",
        help="Learn log(tau) and use tau = exp(log_tau) > 0 (the published "
        "SLeuth model). Off by default: tau is learned directly and may go "
        "negative. Only used with --use_cell_lines.")
    parser.add_argument(
        "--inner_norm",
        choices=["none", "batch"],
        default="none",
        help="'batch': standardise each head's kernel value over the pairs in "
        "the batch before the temperature, (k - mean) / std; eval uses running "
        "per-head estimates. Only used with --use_cell_lines.")
    parser.add_argument(
        "--layernorm",
        action="store_true",
        help="Keep the LayerNorm inside each encoder block (the published "
        "SLeuth model). Off by default: blocks are Linear (bias-free) -> "
        "LeakyReLU -> Dropout, so past the first layer's input offset the "
        "hidden layers carry no additive shift "
        "(LayerNorm's beta was their only one). Only used with "
        "--use_cell_lines.")
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
        choices=["siamese"],
        help="Model family: siamese (inner-product scorer), the only one "
        "with the dense all-pairs scorer the shared-fold export needs. The "
        "kernel / attention classes in siamese_esm.py are kept for loading "
        "legacy checkpoints (predict.py) and cannot be trained here. "
        "--use_cell_lines always builds SiameseSLMultiCell.")
    parser.add_argument("--input_dim",
                        type=int,
                        default=None,
                        help="Ignored (kept for old command lines): the "
                        "encoder input width is taken from each fold's "
                        "preprocessed embedding matrix and recorded in each "
                        "fold checkpoint ('input_dim').")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=512,
                        help="Hidden dim (only when --encoder_dims is unset)")
    parser.add_argument("--latent_dim",
                        type=int,
                        default=256,
                        help="Latent dim (only when --encoder_dims is unset)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--no-last-layer-bias",
        dest="last_layer_bias",
        action="store_false",
        default=True,
        help="Remove the encoder's shared projection bias (z = W h + b; its "
        "per-gene terms act as a gene degree prior). SiameseSL (single "
        "output): removes it. SiameseSLMultiCell (--use_cell_lines): has an "
        "effect only together with --no_per_head_last_layer_bias; by default "
        "the per-head bias c_h (z_h = W h + c_h) replaces the shared bias and "
        "keeps a per-gene prior whatever this flag says.")
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
        "32 32 32 32 32 32 32 (slurm/config.conf): 6 hidden + 1 projection. "
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
        "'mlp': plain stack of Linear→[LN]→LReLU→Dropout blocks (no skips). "
        "The LayerNorm is present only with --layernorm under "
        "--use_cell_lines (off by default there) and always in single-output "
        "mode.")

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
        "e.g., 50 64 64). PCA (method set by --pca_method) after imputation, "
        "before normalization. "
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
        default=None,
        help="Target variance fraction for a PCA (method set by "
        "--pca_method; classical SVD under the default 'plain') applied to "
        "the concatenated matrix AFTER per-modality normalization. "
        "Default: none (no post-concat PCA). "
        "Use --post_pca_dim for an exact component count.")
    parser.add_argument(
        "--post_pca_dim",
        type=int,
        default=None,
        help="Exact component count for the post-concat PCA. "
        "Overrides --post_pca_variance when set. A count at or above the "
        "width of the concatenation it would reduce is refused at that "
        "fold's fit (no post-concat PCA would run).")
    parser.add_argument(
        "--no_post_pca",
        dest="no_post_pca",
        action="store_true",
        default=False,
        help="Disable the post-concat PCA step. It is off by default; "
        "this flag also overrides --post_pca_variance / --post_pca_dim.")
    parser.add_argument(
        "--preprocessing_fit_scope",
        type=str,
        default="train",
        choices=["train", "all"],
        help="Which gene rows to fit preprocessing (impute, PCA and "
        "normalize statistics; the estimators are set by --pca_method) on, "
        "PER FOLD.  "
        "'train' (default): only training-pair genes + genes in no pair — "
        "leak-free; genes that appear only in val or test pairs are "
        "excluded, in every CV type (CV1 included).  "
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
    parser.add_argument(
        "--pos_neg_ratio",
        type=float,
        default=1.0,
        help="No effect: the shared folds (--folds_dir, required) fix the "
        "class balance. Kept so existing SLURM command lines parse.")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--print_arch",
        action="store_true",
        help="Print this run's architecture + selection signature "
        "(arch_signature.py) as JSON and exit without loading any data.")
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

    # Post-concat PCA: off by default. Resolution order:
    #   --no_post_pca        -> disabled (None)
    #   --post_pca_dim       -> exact component count
    #   --post_pca_variance  -> variance target
    #   otherwise            -> disabled (None)
    if args.no_post_pca:
        args.post_pca = None
    elif args.post_pca_dim is not None:
        if args.post_pca_dim <= 0:
            parser.error("--post_pca_dim must be a positive integer")
        args.post_pca = args.post_pca_dim
    elif args.post_pca_variance is not None:
        if not 0 < args.post_pca_variance < 1:
            parser.error("--post_pca_variance must be in (0, 1)")
        args.post_pca = args.post_pca_variance
    else:
        args.post_pca = None

    if args.all_modalities and args.fold_modalities:
        parser.error("--all_modalities trains every modality in every fold: "
                     "it cannot go with --fold_modalities (a per-fold "
                     "selection)")

    if args.print_arch:
        # The run's architecture + selection signature, from exactly the args
        # a real run would save to config.json; nothing is loaded or written.
        # slurm/run_ablation.sh compares it with the reference combo's.
        import json as _json
        from arch_signature import arch_signature
        print(_json.dumps(arch_signature(vars(args)), sort_keys=True))
        return

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
