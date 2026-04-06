#!/usr/bin/env python3
"""
Predict Synthetic Lethality for gene pairs using trained Siamese model.

Usage:
    # Predict for a single pair
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --gene1 BRCA1 --gene2 PARP1

    # Predict for multiple pairs from file
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --pairs_file pairs.txt --output predictions.csv

    # Predict all pairs for a gene (find SL partners)
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_go.pt \
                      --gene1 BRCA1 --top_k 100

    # Multi-modal prediction (must match training modalities)
    python predict.py --model checkpoints/fold_0_best.pt \
                      --embeddings_paths ../data/all_genes_bioconceptvec.pt \
                          ../data/all_genes_go.pt ../data/all_genes_node2vec_ppi.pt \
                      --gene1 BRCA1 --gene2 PARP1
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm

from siamese_esm import SiameseSL, SiameseSLWithAttention, SiameseSLKernel, set_seed
from data_loader import load_multimodal_embeddings
from gene_name_utils import get_mapper


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
    else:
        first_key = _find_key(state_dict, ["encoder.net.0.weight"])
        input_dim = state_dict[first_key].shape[1]
        hidden_dim = state_dict[first_key].shape[0]

        latent_key = _find_key(state_dict, ["encoder.net.4.weight"])
        latent_dim = state_dict[latent_key].shape[0]
        encoder_rank = 64

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
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=False)

    # Infer model params from state dict
    state_dict = checkpoint["model_state_dict"]

    # Auto-detect model type from state dict keys (overrides --model_type)
    if "hilbert_map" in str(state_dict.keys()):
        model_type = "kernel"
    elif "cross_attn" in str(state_dict.keys()):
        model_type = "attention"

    if model_type == "siamese":
        # Infer encoder_dims from state dict
        # Hidden layers: encoder.hidden.{idx}.weight (2D)
        # Projection: encoder.projection.weight (2D)
        hidden_keys = sorted(
            [
                k for k in state_dict if k.startswith("encoder.hidden.")
                and k.endswith(".weight") and state_dict[k].dim() == 2
            ],
            key=lambda k: int(k.split("encoder.hidden.")[1].split(".")[0]))
        proj_key = "encoder.projection.weight"

        input_dim = state_dict[hidden_keys[0]].shape[1] if hidden_keys else \
            state_dict[proj_key].shape[1]
        encoder_dims = [state_dict[k].shape[0] for k in hidden_keys]
        encoder_dims.append(state_dict[proj_key].shape[0])

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
        )
    elif model_type == "kernel":
        kernel_dims = _infer_kernel_dims(state_dict)

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
        )

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

    def predict_pair(self, gene1: str, gene2: str) -> Optional[float]:
        """Predict SL probability for a single gene pair."""
        mapper = get_mapper()
        gene1 = mapper.gene_name_normalize(gene1)
        gene2 = mapper.gene_name_normalize(gene2)
        if gene1 not in self.gene_to_idx:
            print(f"Warning: {gene1} not found in embeddings")
            return None
        if gene2 not in self.gene_to_idx:
            print(f"Warning: {gene2} not found in embeddings")
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
        """Predict SL for multiple gene pairs (order-preserving)."""
        mapper = get_mapper()
        results = [None] * len(pairs)

        # Normalize gene names and partition into valid/invalid
        valid_indices = []  # indices into original pairs list
        valid_pairs = []
        for i, (g1, g2) in enumerate(pairs):
            g1_norm = mapper.gene_name_normalize(g1)
            g2_norm = mapper.gene_name_normalize(g2)
            if g1_norm in self.gene_to_idx and g2_norm in self.gene_to_idx:
                valid_indices.append(i)
                valid_pairs.append((g1_norm, g2_norm))
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
        """Find top SL partners for a given gene."""
        mapper = get_mapper()
        gene = mapper.gene_name_normalize(gene)
        if gene not in self.gene_to_idx:
            print(f"Error: {gene} not found in embeddings")
            return []

        gene_idx = self.gene_to_idx[gene]
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
        help=
        "Per-modality PCA target dimensions (must match training --pca_dims)")

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

    set_seed(args.seed)

    print("Loading model...")
    model = load_model(args.model, args.model_type, args.device,
                       args.num_heads)

    print("Loading embeddings...")
    embeddings, gene_to_idx, idx_to_gene = load_multimodal_embeddings(
        args.embeddings_paths, pca_dims=args.pca_dims)

    predictor = SLPredictor(model, embeddings, gene_to_idx, idx_to_gene,
                            args.device)

    # Run predictions
    if args.top_k and args.gene1:
        # Find SL partners mode
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
