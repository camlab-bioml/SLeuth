#!/usr/bin/env python3
"""
Predict Synthetic Lethality for gene pairs using trained Siamese model.

Accepts both gene symbols (BRCA1) and NCBI Entrez Gene IDs (672).
Internally uses Entrez IDs; symbols are resolved via gene_name_utils.

Usage:
    # Predict for a single pair (by symbol or Entrez ID)
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --gene1 BRCA1 --gene2 PARP1

    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --gene1 672 --gene2 142

    # Predict for multiple pairs from file
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --pairs_file pairs.txt --output predictions.csv

    # Predict all pairs for a gene (find SL partners)
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --gene1 BRCA1 --top_k 100
"""

import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import warnings

import torch
import pandas as pd
from tqdm import tqdm

from siamese_esm import SiameseSL, SiameseSLWithAttention, SiameseSLKernel, set_seed
from data_loader import (
    apply_multimodal_transform,
    load_multimodal_embeddings,
    load_raw_multimodal,
)
from gene_name_utils import get_mapper


def _file_sha256(path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes, chunked (the .pt files are ~200MB each).

    The same digest train.py records as `embeddings_sha256`.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _find_key(state_dict: dict, contains: list) -> str:
    for key in state_dict.keys():
        if all(token in key for token in contains):
            return key
    raise KeyError(f"Missing key containing: {contains}")


def _infer_kernel_encoder_type(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    if any("encoder.net.0.U.weight" in k for k in keys):
        return "lowrank"
    if any("encoder.net.0.net.0.weight" in k for k in keys):
        return "bottleneck"
    if any("encoder.net.0.linear.weight" in k for k in keys):
        return "gated"
    return "standard"


def _infer_kernel_dims(state_dict: dict) -> dict:
    encoder_type = _infer_kernel_encoder_type(state_dict)
    if encoder_type == "lowrank":
        v_key = _find_key(state_dict, ["encoder.net.0.V.weight"])
        u_key = _find_key(state_dict, ["encoder.net.0.U.weight"])
        input_dim = state_dict[v_key].shape[1]
        hidden_dim = state_dict[u_key].shape[0]
        encoder_rank = state_dict[v_key].shape[0]

        latent_u_key = _find_key(state_dict, ["encoder.net.4.U.weight"])
        latent_dim = state_dict[latent_u_key].shape[0]
    elif encoder_type == "bottleneck":
        first_key = _find_key(state_dict, ["encoder.net.0.net.0.weight"])
        input_dim = state_dict[first_key].shape[1]
        bottleneck1 = state_dict[first_key].shape[0]

        hidden_key = _find_key(state_dict, ["encoder.net.0.net.4.weight"])
        hidden_dim = state_dict[hidden_key].shape[0]

        bottleneck2_key = _find_key(state_dict, ["encoder.net.3.net.0.weight"])
        bottleneck2 = state_dict[bottleneck2_key].shape[0]

        latent_key = _find_key(state_dict, ["encoder.net.3.net.4.weight"])
        latent_dim = state_dict[latent_key].shape[0]

        # Recover encoder_rank from bottleneck dims.
        # EfficientEncoder computes: bottleneck1 = max(rank, input_dim // 8)
        #                            bottleneck2 = max(rank // 2, output_dim // 4)
        # Try candidates and verify both equations hold.
        encoder_rank = bottleneck1  # fallback
        for candidate in [bottleneck1, bottleneck2 * 2]:
            if (max(candidate, input_dim // 8) == bottleneck1
                    and max(candidate // 2, latent_dim // 4) == bottleneck2):
                encoder_rank = candidate
                break
    elif encoder_type == "gated":
        first_key = _find_key(state_dict, ["encoder.net.0.linear.weight"])
        input_dim = state_dict[first_key].shape[1]
        hidden_dim = state_dict[first_key].shape[0]

        latent_key = _find_key(state_dict, ["encoder.net.3.linear.weight"])
        latent_dim = state_dict[latent_key].shape[0] // 2
        encoder_rank = 64
        warnings.warn(
            f"_infer_kernel_dims: cannot infer encoder_rank for '{encoder_type}' "
            f"encoder from the weights; assuming {encoder_rank}. The checkpoint "
            f"config's encoder_rank overrides this when present (there is no "
            f"CLI override); a wrong value fails the state_dict load with a "
            f"shape mismatch."
        )
    else:
        first_key = _find_key(state_dict, ["encoder.net.0.weight"])
        input_dim = state_dict[first_key].shape[1]
        hidden_dim = state_dict[first_key].shape[0]

        latent_key = _find_key(state_dict, ["encoder.net.4.weight"])
        latent_dim = state_dict[latent_key].shape[0]
        encoder_rank = 64
        warnings.warn(
            f"_infer_kernel_dims: cannot infer encoder_rank for '{encoder_type}' "
            f"encoder from the weights; assuming {encoder_rank}. The checkpoint "
            f"config's encoder_rank overrides this when present (there is no "
            f"CLI override); a wrong value fails the state_dict load with a "
            f"shape mismatch."
        )

    return {
        "encoder_type": encoder_type,
        "encoder_rank": encoder_rank,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
    }


def load_model(
    checkpoint_path: str,
    model_type: str = "siamese",
    device: str = "cpu",
    num_heads: Optional[int] = None,
    load_weights: bool = True,
) -> torch.nn.Module:
    """Load trained model from checkpoint.

    load_weights=False: the checkpoint's architecture at the constructor's
    own initialisation (the caller's torch RNG state), no weight loaded
    (eval_finetuning's Scratch mode)."""
    checkpoint = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=False)

    # Infer model params from state dict
    state_dict = checkpoint["model_state_dict"]

    # Cell-line (SiameseSLMultiCell) checkpoints are not supported here — they
    # emit one logit PER head, not a single SL score. Detect every form, not
    # just cell_emb: a --cell_lines pooled checkpoint has NO cell_emb (one
    # head, no cell-line embedding) but a (1,)-shaped scoring_bias, and the
    # per-head / bilinear / raw-tau / inner-norm options add parameters that
    # SiameseSL does not have. Loaded as SiameseSL, such a checkpoint would
    # mostly fail with an opaque state-dict error; a pooled checkpoint with
    # otherwise published options WOULD load (load_state_dict squeezes its
    # (1,) scoring_bias into SiameseSL's scalar) and score the same function,
    # but it is refused too so every multi-cell checkpoint takes one path.
    _MULTICELL_KEYS = ("tau", "head_proj_bias", "head_input_bias", "kernel_A",
                       "inner_running_mean", "gene_w", "pair_fc1.weight")
    _sb = state_dict.get("scoring_bias")
    _mc_found = sorted(
        [k for k in state_dict if k.startswith("cell_emb")] +
        [k for k in _MULTICELL_KEYS if k in state_dict] +
        (["scoring_bias (per-head, dim 1)"]
         if _sb is not None and _sb.dim() == 1 else []))
    if _mc_found:
        raise NotImplementedError(
            f"{checkpoint_path} is a SiameseSLMultiCell checkpoint (trained "
            f"with --use_cell_lines; found {', '.join(_mc_found)}): it emits "
            f"one logit per head, per_head or pooled, which predict.py's "
            f"single-output SiameseSL cannot represent. Load it with "
            f"eval_finetuning.load_multicell_model (e.g. eval_finetuning.py "
            f"--per_head) and read the head column you need, or run "
            f"export_pair_predictions.py for every gene pair and head.")

    # Auto-detect model type from state dict keys (overrides --model_type)
    if "hilbert_map" in str(state_dict.keys()):
        model_type = "kernel"
    elif "cross_attn" in str(state_dict.keys()):
        model_type = "attention"

    # The checkpoint's own dropout: eval_finetuning's Full / LP_last modes
    # fine-tune this model in train mode, where the rate matters (a model
    # trained with dropout 0.0 must not be adapted at the constructor's 0.2).
    ckpt_dropout = float((checkpoint.get("config") or {}).get("dropout", 0.2))

    if model_type == "siamese":
        # Detect encoder variant from state-dict key layout:
        #   residual → encoder.hidden_blocks.{i}.0.weight  (ModuleList of Sequential)
        #   mlp      → encoder.hidden.{idx}.weight         (flat Sequential)
        has_residual_encoder = any("hidden_blocks" in k for k in state_dict)

        if has_residual_encoder:
            # Residual: the 2D weight in each block sits at index 0 of that
            # block's Sequential. Keys look like encoder.hidden_blocks.2.0.weight.
            block_keys = sorted(
                [
                    k for k in state_dict
                    if k.startswith("encoder.hidden_blocks.")
                    and k.endswith(".0.weight") and state_dict[k].dim() == 2
                ],
                key=lambda k: int(
                    k.split("encoder.hidden_blocks.")[1].split(".")[0]))
            proj_key = "encoder.projection.weight"

            input_dim = (state_dict[block_keys[0]].shape[1]
                         if block_keys else state_dict[proj_key].shape[1])
            encoder_dims = [state_dict[k].shape[0] for k in block_keys]
            encoder_dims.append(state_dict[proj_key].shape[0])
            siamese_encoder_type = "residual"
        else:
            hidden_keys = sorted(
                [
                    k for k in state_dict if k.startswith("encoder.hidden.")
                    and k.endswith(".weight") and state_dict[k].dim() == 2
                ],
                key=lambda k: int(k.split("encoder.hidden.")[1].split(".")[0]))
            proj_key = "encoder.projection.weight"

            input_dim = (state_dict[hidden_keys[0]].shape[1]
                         if hidden_keys else state_dict[proj_key].shape[1])
            encoder_dims = [state_dict[k].shape[0] for k in hidden_keys]
            encoder_dims.append(state_dict[proj_key].shape[0])
            siamese_encoder_type = "mlp"

        # Infer last_layer_bias
        has_bias = "encoder.projection.bias" in state_dict

        # Infer pd_epsilon from checkpoint config (default 0.001)
        config = checkpoint.get("config", {})
        pd_eps = config.get("pd_epsilon", 0.001)

        model = SiameseSL(
            input_dim=input_dim,
            encoder_dims=encoder_dims,
            last_layer_bias=has_bias,
            pd_epsilon=pd_eps,
            siamese_encoder_type=siamese_encoder_type,
            dropout=ckpt_dropout,
        )
    elif model_type == "kernel":
        kernel_dims = _infer_kernel_dims(state_dict)

        # Override encoder_rank from checkpoint config if available
        config = checkpoint.get("config", {})
        if "encoder_rank" in config:
            kernel_dims["encoder_rank"] = config["encoder_rank"]

        hilbert_key = _find_key(state_dict, ["hilbert_map.linear_map.weight"])
        bilinear_rank = state_dict[hilbert_key].shape[0]

        rff_keys = [
            k for k in state_dict.keys() if "hilbert_map.rff.omega" in k
        ]
        rff_features = state_dict[rff_keys[0]].shape[1] if rff_keys else 0

        predictor_key = _find_key(state_dict, ["predictor.0.weight"])
        predictor_hidden = state_dict[predictor_key].shape[0]

        model = SiameseSLKernel(
            input_dim=kernel_dims["input_dim"],
            hidden_dim=kernel_dims["hidden_dim"],
            latent_dim=kernel_dims["latent_dim"],
            bilinear_rank=bilinear_rank,
            rff_features=rff_features,
            predictor_hidden=predictor_hidden,
            encoder_type=kernel_dims["encoder_type"],
            encoder_rank=kernel_dims["encoder_rank"],
            dropout=ckpt_dropout,
        )
    else:  # attention
        first_layer_key = _find_key(state_dict, ["proj.0.weight"])
        input_dim = state_dict[first_layer_key].shape[1]
        hidden_dim = state_dict[first_layer_key].shape[0]

        latent_key = _find_key(state_dict, ["post_attn.0.weight"])
        latent_dim = state_dict[latent_key].shape[0]

        # num_heads can't be inferred from weight shapes; use CLI override,
        # checkpoint config, config.json, or default 4 (in priority order)
        if num_heads is None:
            if "config" in checkpoint:
                num_heads = checkpoint["config"].get("num_heads", 4)
            else:
                config_path = Path(
                    checkpoint_path).parent.parent / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    num_heads = config.get("num_heads", 4)
                else:
                    num_heads = 4
                    print(
                        f"Warning: no config in checkpoint or {config_path}, using num_heads={num_heads}"
                    )

        model = SiameseSLWithAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=ckpt_dropout,
        )

    if not load_weights:
        model.to(device)
        model.eval()
        return model

    # Backward compatibility: old checkpoints lack input_bias (added later).
    # Inject missing keys as zero tensors so load_state_dict doesn't fail.
    for key, param in model.named_parameters():
        if key not in state_dict and "input_bias" in key:
            state_dict[key] = torch.zeros_like(param)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


class SLPredictor:
    """Predictor for Synthetic Lethality.

    Embeddings are already imputed and normalized by load_multimodal_embeddings().
    """

    def __init__(
        self,
        model: torch.nn.Module,
        embeddings: torch.Tensor,
        gene_to_idx: dict,
        idx_to_gene: dict,
        device: str = "cpu",
    ):
        self.model = model
        self.embeddings = embeddings.to(device)
        self.gene_to_idx = gene_to_idx
        self.idx_to_gene = idx_to_gene
        self.device = device

    def _resolve_gene(self, gene: str) -> Optional[str]:
        """Resolve a gene identifier (symbol or Entrez ID) to Entrez ID."""
        # Already an Entrez ID?
        if gene in self.gene_to_idx:
            return gene
        # Try as symbol -> Entrez ID; a name the screens pin
        # (eval_finetuning.SCREEN_SYMBOL_OVERRIDES) is the pinned gene, as in
        # the eval and the pair export. Imported here: eval_finetuning
        # imports this module.
        from eval_finetuning import symbol_entrez
        eid = symbol_entrez(get_mapper(), gene)
        if eid and eid in self.gene_to_idx:
            return eid
        return None

    def predict_pair(self, gene1: str, gene2: str) -> Optional[float]:
        """Predict SL probability for a single gene pair.

        Accepts either gene symbols (BRCA1) or Entrez IDs (672).
        """
        orig1, orig2 = gene1, gene2
        gene1 = self._resolve_gene(gene1)
        gene2 = self._resolve_gene(gene2)
        if gene1 is None:
            print(f"Warning: {orig1} not found in embeddings")
            return None
        if gene2 is None:
            print(f"Warning: {orig2} not found in embeddings")
            return None

        idx1 = self.gene_to_idx[gene1]
        idx2 = self.gene_to_idx[gene2]

        x1 = self.embeddings[idx1].unsqueeze(0)
        x2 = self.embeddings[idx2].unsqueeze(0)

        with torch.no_grad():
            logit = self.model(x1, x2)
            prob = torch.sigmoid(logit).item()

        return prob

    def predict_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 256,
    ) -> List[dict]:
        """Predict SL for multiple gene pairs (order-preserving).

        Accepts gene symbols or Entrez IDs.
        """
        results = [None] * len(pairs)

        # Resolve gene identifiers and partition into valid/invalid
        valid_indices = []  # indices into original pairs list
        valid_pairs = []
        for i, (g1, g2) in enumerate(pairs):
            eid1 = self._resolve_gene(g1)
            eid2 = self._resolve_gene(g2)
            if eid1 is not None and eid2 is not None:
                valid_indices.append(i)
                valid_pairs.append((eid1, eid2))
            else:
                results[i] = {
                    "gene1": g1,
                    "gene2": g2,
                    "probability": None,
                    "error": "Gene not found",
                }

        # Batch predict
        for batch_start in tqdm(range(0, len(valid_pairs), batch_size),
                                desc="Predicting"):
            batch = valid_pairs[batch_start:batch_start + batch_size]
            batch_orig_idx = valid_indices[batch_start:batch_start +
                                           batch_size]

            idx1 = [self.gene_to_idx[g1] for g1, g2 in batch]
            idx2 = [self.gene_to_idx[g2] for g1, g2 in batch]

            x1 = self.embeddings[idx1]
            x2 = self.embeddings[idx2]

            with torch.no_grad():
                logits = self.model(x1, x2)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for j, orig_i in enumerate(batch_orig_idx):
                g1, g2 = pairs[orig_i]
                results[orig_i] = {
                    "gene1": g1,
                    "gene2": g2,
                    "probability": float(probs[j]),
                    "error": None,
                }

        return results

    def find_sl_partners(
        self,
        gene: str,
        top_k: int = 100,
        batch_size: int = 512,
    ) -> List[dict]:
        """Find top SL partners for a given gene.

        Accepts gene symbol or Entrez ID.
        """
        resolved = self._resolve_gene(gene)
        if resolved is None:
            print(f"Error: {gene} not found in embeddings")
            return []

        gene_idx = self.gene_to_idx[resolved]
        gene_emb = self.embeddings[gene_idx].unsqueeze(0)

        all_scores = []

        # Score all other genes
        all_idx = list(range(len(self.embeddings)))
        all_idx.remove(gene_idx)

        for i in tqdm(range(0, len(all_idx), batch_size),
                      desc=f"Scoring partners for {gene}"):
            batch_idx = all_idx[i:i + batch_size]
            batch_emb = self.embeddings[batch_idx]

            # Expand gene embedding to match batch
            gene_batch = gene_emb.expand(len(batch_idx), -1)

            with torch.no_grad():
                logits = self.model(gene_batch, batch_emb)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for j, idx in enumerate(batch_idx):
                all_scores.append({
                    "gene": self.idx_to_gene[idx],
                    "probability": float(probs[j]),
                })

        # Sort by probability and return top_k
        all_scores.sort(key=lambda x: x["probability"], reverse=True)
        return all_scores[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Predict SL for gene pairs")

    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--embeddings",
                        type=str,
                        default=None,
                        help="(Deprecated: use --embeddings_paths) "
                        "Single embedding file.")
    parser.add_argument("--embeddings_paths",
                        type=str,
                        nargs='+',
                        default=None,
                        help="One or more .pt embedding files "
                        "(must match training modalities).")
    parser.add_argument("--model_type",
                        type=str,
                        default="siamese",
                        choices=["siamese", "attention", "kernel"],
                        help="Model architecture")

    # Prediction modes
    parser.add_argument("--gene1", type=str, help="First gene")
    parser.add_argument("--gene2", type=str, help="Second gene")
    parser.add_argument("--pairs_file",
                        type=str,
                        help="File with gene pairs (gene1 TAB gene2 per line)")
    parser.add_argument("--top_k",
                        type=int,
                        default=None,
                        help="Find top K SL partners for gene1")
    parser.add_argument("--output",
                        type=str,
                        default=None,
                        help="Output file for results")

    # Model overrides (for old checkpoints without embedded config)
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help=
        "Attention heads override (auto-detected from checkpoint if available)"
    )

    # Embedding pipeline options
    parser.add_argument(
        "--pca_dims",
        type=int,
        nargs='+',
        default=None,
        help="Per-modality PCA target dimensions (must match training). "
        "Mutually exclusive with --pca_variance.")
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=None,
        help="Target variance fraction for PCA (0-1, must match training). "
        "Mutually exclusive with --pca_dims.")

    # Runtime
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Unify: --embeddings is a convenience alias for a single-element list
    if args.embeddings_paths and args.embeddings:
        parser.error("Use --embeddings OR --embeddings_paths, not both")
    if args.embeddings:
        args.embeddings_paths = [args.embeddings]
    if not args.embeddings_paths:
        parser.error("--embeddings_paths is required (or --embeddings)")

    # PCA: --pca_variance is a convenience flag
    if args.pca_variance is not None and args.pca_dims is not None:
        parser.error("Use --pca_dims OR --pca_variance, not both")
    if args.pca_variance is not None:
        if not 0 < args.pca_variance < 1:
            parser.error("--pca_variance must be in (0, 1)")
        args.pca_dims = [args.pca_variance] * len(args.embeddings_paths)

    set_seed(args.seed)

    print("Loading model...")
    model = load_model(args.model, args.model_type, args.device,
                       args.num_heads)

    # Prefer the checkpoint's stored preprocessing transform (new-format,
    # leakage-free fold-specific fit). Fall back to legacy all-gene fit
    # driven by --pca_variance/--pca_dims when the checkpoint predates the
    # fit/apply split.
    ckpt_raw = torch.load(args.model, map_location="cpu", weights_only=False)
    # A fine-tuned model saved by eval_finetuning.save_finetuned_model holds
    # the weights and the affine head only: the fold's transform and the keys
    # the three guards below read stay in its source checkpoint, so it would
    # be taken for a legacy checkpoint (all-gene fit, no guard), and its
    # output is a screen value a * logit + c, not an SL probability.
    if str(ckpt_raw.get("format", "")).startswith("siamese_sl.finetuned"):
        raise SystemExit(
            f"{args.model} is a fine-tuned model saved by eval_finetuning.py "
            f"(format {ckpt_raw['format']}): it carries no preprocessing "
            f"transform and predicts a screen value, not an SL probability. "
            f"Score it with export_pair_predictions.py --finetuned_eval_dirs.")
    fold_transform = ckpt_raw.get("preprocessing_transform")

    # Same-files guard at the basename level, as
    # eval_finetuning.checkpoint_input_problems applies it. Neither the
    # gene-order check below (every catalog file is generated over one gene
    # universe, so all modalities share an order) nor
    # apply_multimodal_transform's per-modality width check can tell two
    # same-width modalities passed in swapped order from the trained order —
    # each would then be pushed through the other's fitted impute / PCA /
    # normalise statistics. Legacy checkpoints lack the key and are not
    # refused.
    ckpt_emb_paths = ckpt_raw.get("embeddings_paths")
    if ckpt_emb_paths is not None:
        want = [Path(p).name for p in ckpt_emb_paths]
        got = [Path(p).name for p in args.embeddings_paths]
        if want != got:
            raise ValueError(
                f"The checkpoint was trained on {want} (in that order); got "
                f"{got}. Pass the same embedding file(s), in the same order, "
                f"used for training.")

    # CONTENT guard (eval_finetuning.checkpoint_input_problems applies the
    # same one). The basename and gene_order checks both survive a rebuild of
    # the SAME files at the SAME paths over the SAME gene universe, which
    # run_generate_embeddings.sh does routinely and node2vec / go2vec /
    # onto2vec do not reproduce bit for bit — leaving the checkpoint's fitted
    # transform and weights scoring different vectors. train.py records the
    # trained files' sha256; legacy checkpoints lack the key and are not
    # refused.
    ckpt_emb_sha = ckpt_raw.get("embeddings_sha256")
    if ckpt_emb_sha is not None:
        got_sha = [_file_sha256(p) for p in args.embeddings_paths]
        if list(map(str, ckpt_emb_sha)) != got_sha:
            raise ValueError(
                "The embedding file(s) named have DIFFERENT CONTENTS from "
                "the ones the checkpoint was trained on (same paths, "
                "regenerated in place?). Its stored per-fold PCA basis and "
                "these weights would score different vectors; pass the files "
                "it was trained on, or retrain.")

    if fold_transform is not None:
        print("Loading embeddings (applying checkpoint's transform)...")
        raw_per_modality, gene_to_idx, idx_to_gene = load_raw_multimodal(
            args.embeddings_paths)
        # A same-files guard, not an indexing one. The fitted transform is
        # row-independent (every stored statistic is per feature — impute
        # locations, PCA pre-center/scale, V — or a scalar center/scale), the
        # model has no per-gene parameters, and gene_to_idx is rebuilt from
        # these same files, so row order alone cannot misindex a prediction
        # (feature dims are checked by apply_multimodal_transform, and the
        # modality list and its order against the checkpoint's
        # embeddings_paths above). But the training files load in exactly the
        # stored gene_order, so a different order proves these are other (e.g.
        # regenerated) embeddings, whose features the model was never trained
        # on.
        ckpt_gene_order = ckpt_raw.get("gene_order")
        if ckpt_gene_order is not None:
            current_order = [idx_to_gene[i] for i in range(len(idx_to_gene))]
            if list(ckpt_gene_order) != current_order:
                raise ValueError(
                    "Gene order of --embeddings_paths does not match the "
                    "checkpoint's stored gene_order, so these are not the "
                    "embedding file(s) the model was trained on (gene "
                    "indexing itself is order-independent; their features "
                    "may not be the trained ones). Pass the same embedding "
                    "file(s), in the same order, used for training.")
        embeddings = apply_multimodal_transform(raw_per_modality,
                                                fold_transform)
        print(f"  embeddings: {tuple(embeddings.shape)}")
    else:
        print("Loading embeddings (legacy all-gene PCA fit — checkpoint "
              "has no preprocessing_transform)...")
        embeddings, gene_to_idx, idx_to_gene = load_multimodal_embeddings(
            args.embeddings_paths, pca_dims=args.pca_dims)

    predictor = SLPredictor(model, embeddings, gene_to_idx, idx_to_gene,
                            args.device)

    # Run predictions
    if args.top_k and args.gene1:
        # Find SL partners mode
        if args.gene2:
            print(
                f"Warning: --gene2 '{args.gene2}' is ignored when --top_k is used"
            )
        results = predictor.find_sl_partners(args.gene1, args.top_k,
                                             args.batch_size)
        df = pd.DataFrame(results)
        df["query_gene"] = args.gene1

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved {len(results)} results to {args.output}")
        else:
            print(f"\nTop {args.top_k} SL partners for {args.gene1}:")
            print(df.head(20).to_string(index=False))

    elif args.pairs_file:
        # Batch prediction mode
        pairs = []
        with open(args.pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))

        results = predictor.predict_pairs(pairs, args.batch_size)
        df = pd.DataFrame(results)

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved {len(results)} predictions to {args.output}")
        else:
            print(df.to_string(index=False))

    elif args.gene1 and args.gene2:
        # Single pair mode
        prob = predictor.predict_pair(args.gene1, args.gene2)
        if prob is not None:
            print(
                f"\nSL probability for {args.gene1} - {args.gene2}: {prob:.4f}"
            )
        else:
            print("Prediction failed - check gene names")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
