#!/usr/bin/env python3
"""
Generate ESM embeddings for ALL human protein-coding genes.

This script:
  1. Downloads the UniProt human proteome (reference proteome UP000005640)
  2. Extracts gene names and protein sequences
  3. Generates ESM-2 embeddings using Pool PaRTI (PageRank-based pooling)
  4. Saves embeddings with gene order for reproducibility

Pool PaRTI Reference:
  Tartici et al. "Pool PaRTI: a PageRank-based pooling method for identifying
  critical residues and enhancing protein sequence representations"
  Bioinformatics, 2025. https://github.com/Helix-Research-Lab/Pool_PaRTI

Requirements:
  pip install fair-esm biopython requests tqdm networkx

Usage:
    # Basic usage with Pool PaRTI (default)
    python generate_all_genes_esm.py --output ../data/all_genes_esm.pt

    # Use mean pooling instead
    python generate_all_genes_esm.py --pooling mean --output ../data/all_genes_esm.pt

    # GPU with larger batch size
    python generate_all_genes_esm.py --device cuda:0 --batch_size 16 --output out.pt
"""

import gzip
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from gene_name_utils import get_mapper

try:
    import networkx as nx
except ImportError:
    nx = None  # Will be checked if pool_parti is used

# UniProt human reference proteome
UNIPROT_HUMAN_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "format=fasta&"
    "query=reviewed:true+AND+organism_id:9606"  # Reviewed human proteins
)

# Alternative: full proteome (including unreviewed)
UNIPROT_FULL_URL = ("https://rest.uniprot.org/uniprotkb/stream?"
                    "format=fasta&"
                    "query=organism_id:9606+AND+proteome:UP000005640")


def download_uniprot_fasta(output_path: str,
                           reviewed_only: bool = True) -> str:
    """Download human proteome from UniProt."""
    print("Downloading human proteome from UniProt...")

    url = UNIPROT_HUMAN_URL if reviewed_only else UNIPROT_FULL_URL

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(total=total_size,
                  unit="B",
                  unit_scale=True,
                  desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded to {output_path}")
    return output_path


def parse_fasta(fasta_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse FASTA file to extract gene names and sequences.

    UniProt FASTA headers have format:
    >sp|P12345|GENE_HUMAN Description GN=GeneName ...

    Returns:
        Dict mapping gene name -> (uniprot_id, sequence)
    """
    gene_seqs = {}  # gene_name -> (uniprot_id, sequence)
    duplicates = defaultdict(list)

    current_header = None
    current_seq = []

    def parse_header(header: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract gene name and UniProt ID from header."""
        parts = header.split("|")
        uniprot_id = parts[1] if len(parts) > 1 else None

        # Try to find gene name (GN=...)
        gene_name = None
        if "GN=" in header:
            gn_part = header.split("GN=")[1]
            gene_name = gn_part.split()[0]
        elif len(parts) > 2:
            # Use entry name (e.g., GENE_HUMAN)
            entry_name = parts[2].split()[0]
            if "_HUMAN" in entry_name:
                gene_name = entry_name.replace("_HUMAN", "")

        return gene_name, uniprot_id

    def save_entry():
        if current_header and current_seq:
            gene_name, uniprot_id = parse_header(current_header)
            seq = "".join(current_seq)

            if gene_name:
                if gene_name in gene_seqs:
                    # Keep longest sequence for duplicates
                    existing_seq = gene_seqs[gene_name][1]
                    if len(seq) > len(existing_seq):
                        duplicates[gene_name].append(gene_seqs[gene_name])
                        gene_seqs[gene_name] = (uniprot_id, seq)
                    else:
                        duplicates[gene_name].append((uniprot_id, seq))
                else:
                    gene_seqs[gene_name] = (uniprot_id, seq)

    # Handle gzipped files
    if fasta_path.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt")
    else:
        opener = lambda p: open(p, "r")

    with opener(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                save_entry()
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)

        save_entry()  # Don't forget last entry

    print(f"Parsed {len(gene_seqs)} unique genes")
    if duplicates:
        print(
            f"  (resolved {len(duplicates)} duplicates by keeping longest sequence)"
        )

    # Normalize gene names (custom corrections + HGNC alias/previous symbol)
    mapper = get_mapper()
    normalized = {}
    renamed = 0
    for gene, val in gene_seqs.items():
        new_name = mapper.gene_name_normalize(gene)
        if new_name != gene.upper():
            renamed += 1
        # If two old names map to the same current symbol, keep longest sequence
        if new_name in normalized:
            if len(val[1]) > len(normalized[new_name][1]):
                normalized[new_name] = val
        else:
            normalized[new_name] = val
    if renamed:
        print(f"  Gene name normalization: {renamed} renamed")
    print(f"  Final: {len(normalized)} unique genes after normalization")

    return normalized


class PoolPaRTI:
    """
    Pool PaRTI: PageRank-based Token Importance pooling.

    Converts attention matrices to a directed graph and uses PageRank
    to compute importance weights for each residue position.

    Reference: Tartici et al., Bioinformatics 2025
    """

    def __init__(self, alpha: float = 0.85):
        """
        Args:
            alpha: PageRank damping factor (default 0.85)
        """
        if nx is None:
            raise ImportError(
                "networkx required for Pool PaRTI. Install: pip install networkx"
            )
        self.alpha = alpha

    def _aggregate_attention(self, attentions: torch.Tensor) -> np.ndarray:
        """
        Aggregate attention matrices across all layers and heads.

        Uses position-wise max pooling as in the original Pool PaRTI paper.

        Args:
            attentions: Attention tensor (num_layers, num_heads, seq_len, seq_len)

        Returns:
            Aggregated attention matrix (seq_len, seq_len)
        """
        # attentions shape: (layers, heads, seq_len, seq_len)
        # Max pool across layers and heads
        agg_attn = attentions.max(dim=0).values  # (heads, seq_len, seq_len)
        agg_attn = agg_attn.max(dim=0).values  # (seq_len, seq_len)
        return agg_attn.cpu().numpy()

    def _compute_pagerank(self, attention_matrix: np.ndarray) -> np.ndarray:
        """
        Compute PageRank scores from attention matrix.

        Args:
            attention_matrix: Aggregated attention (seq_len, seq_len)

        Returns:
            Importance weights for each position (seq_len,)
        """
        # Create directed graph from attention matrix
        G = nx.from_numpy_array(attention_matrix, create_using=nx.DiGraph)

        # Handle edge case of disconnected graph
        if G.number_of_edges() == 0:
            # Fall back to uniform weights
            n = attention_matrix.shape[0]
            return np.ones(n) / n

        # Compute PageRank
        try:
            pr = nx.pagerank(G,
                             alpha=self.alpha,
                             tol=1e-06,
                             weight='weight',
                             max_iter=100)
            weights = np.array([pr[i] for i in range(len(pr))])
        except nx.PowerIterationFailedConvergence:
            # Fall back to uniform weights on convergence failure
            n = attention_matrix.shape[0]
            weights = np.ones(n) / n

        return weights

    def pool(
        self,
        token_embeddings: torch.Tensor,
        attentions: torch.Tensor,
        seq_len: int,
    ) -> np.ndarray:
        """
        Apply Pool PaRTI to get sequence embedding.

        Args:
            token_embeddings: Per-token embeddings (1, full_seq_len, embed_dim)
            attentions: Attention matrices (num_layers, num_heads, full_seq_len, full_seq_len)
            seq_len: Actual sequence length (excluding special tokens)

        Returns:
            Sequence embedding (embed_dim,)
        """
        # Extract embeddings for actual sequence (exclude BOS/EOS tokens)
        # ESM adds BOS at position 0, sequence is 1:seq_len+1
        embeddings = token_embeddings[
            0, 1:seq_len + 1, :].cpu().numpy()  # (seq_len, embed_dim)

        # Extract attention for actual sequence positions
        attn = attentions[:, :, 1:seq_len + 1,
                          1:seq_len + 1]  # (layers, heads, seq_len, seq_len)

        # Aggregate attention matrices
        agg_attn = self._aggregate_attention(attn)

        # Compute PageRank importance weights
        weights = self._compute_pagerank(agg_attn)

        # Normalize weights to sum to 1
        weights = weights / (weights.sum() + 1e-8)

        # Weighted average of embeddings
        pooled = np.sum(embeddings * weights[:, np.newaxis], axis=0)

        return pooled


class ESMEmbeddingGenerator:
    """Generate ESM-2 embeddings for protein sequences."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
        batch_size: int = 8,
        pooling: str = "pool_parti",
    ):
        """
        Args:
            model_name: ESM model name
            device: Device to run on
            batch_size: Batch size for processing
            pooling: Pooling method - "pool_parti" (default) or "mean"
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling = pooling

        # Detect device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        print(f"Loading ESM model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Pooling method: {pooling}")

        # Initialize Pool PaRTI if needed
        if pooling == "pool_parti":
            self.pooler = PoolPaRTI(alpha=0.85)
        else:
            self.pooler = None

        # Load ESM
        try:
            import esm
        except ImportError:
            raise ImportError(
                "fair-esm not installed. Run: uv pip install fair-esm")

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Get model info
        self.embed_dim = self.model.embed_dim
        self.num_layers = self.model.num_layers
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Number of layers: {self.num_layers}")

    def _generate_batch(self,
                        batch: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for a batch of sequences."""
        labels, strs, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        # Request attention if using Pool PaRTI
        need_attn = self.pooling == "pool_parti"

        with torch.no_grad():
            results = self.model(
                tokens,
                repr_layers=[self.num_layers],
                need_head_weights=need_attn,  # Returns attention weights
                return_contacts=False,
            )
            representations = results["representations"][self.num_layers]

            # Get attention if needed
            # ESM returns attentions with shape: (batch, layers, heads, seq_len, seq_len)
            if need_attn and "attentions" in results:
                attentions = results["attentions"]
            else:
                attentions = None

        embeddings = {}
        for i, (gene, seq) in enumerate(batch):
            seq_len = min(len(seq), 1022)  # ESM max length

            if self.pooling == "pool_parti" and attentions is not None:
                # Pool PaRTI: PageRank-weighted pooling
                # Attention shape: (batch, layers, heads, seq, seq)
                attn_i = attentions[i]  # (layers, heads, seq, seq)
                emb_i = representations[i:i + 1]  # (1, seq, dim)
                embedding = self.pooler.pool(emb_i, attn_i, seq_len)
            else:
                # Mean pooling fallback
                embedding = representations[i, 1:seq_len +
                                            1].mean(0).cpu().numpy()

            embeddings[gene] = embedding

        return embeddings

    def generate_embeddings(
        self,
        gene_seqs: Dict[str, Tuple[str, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        """
        Generate embeddings for all genes.

        Args:
            gene_seqs: Dict mapping gene name -> (uniprot_id, sequence)

        Returns:
            (embeddings tensor, raw_embeddings tensor, gene_order list, failed_genes list)
        """
        gene_order = sorted(gene_seqs.keys())
        all_embeddings = {}

        # Prepare batches
        batch = []
        failed_genes = []

        print(f"\nGenerating embeddings for {len(gene_order)} genes...")
        pbar = tqdm(gene_order, desc="Embedding")

        for gene in pbar:
            _, seq = gene_seqs[gene]

            # Skip invalid sequences — mark as NaN for Huber imputation
            if not seq or len(seq) < 10:
                failed_genes.append(gene)
                all_embeddings[gene] = np.full(self.embed_dim,
                                               np.nan,
                                               dtype=np.float32)
                continue

            # Truncate long sequences
            if len(seq) > 1022:
                seq = seq[:1022]

            batch.append((gene, seq))

            # Process batch
            if len(batch) >= self.batch_size:
                try:
                    batch_embs = self._generate_batch(batch)
                    all_embeddings.update(batch_embs)
                except Exception as e:
                    print(f"\nBatch failed: {e}")
                    for g, _ in batch:
                        failed_genes.append(g)
                        all_embeddings[g] = np.full(self.embed_dim,
                                                    np.nan,
                                                    dtype=np.float32)
                batch = []

        # Process remaining
        if batch:
            try:
                batch_embs = self._generate_batch(batch)
                all_embeddings.update(batch_embs)
            except Exception as e:
                print(f"\nFinal batch failed: {e}")
                for g, _ in batch:
                    failed_genes.append(g)
                    all_embeddings[g] = np.full(self.embed_dim,
                                                np.nan,
                                                dtype=np.float32)

        # Stack into tensor (failed genes have NaN, imputed at load time)
        embedding_matrix = np.stack([all_embeddings[g] for g in gene_order])

        n_failed = len(failed_genes)
        n_ok = len(gene_order) - n_failed
        print(f"\nGenerated {len(gene_order)} embeddings")
        print(f"  - Successful: {n_ok}")
        print(f"  - Failed (NaN, Huber-imputed at load time): {n_failed}")

        embeddings = torch.from_numpy(embedding_matrix).float()
        return (
            embeddings,
            embeddings,  # raw_embeddings = embeddings (no standardization)
            gene_order,
            failed_genes,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM embeddings for all human genes")
    parser.add_argument(
        "--fasta",
        type=str,
        default=None,
        help=
        "Path to FASTA file (optional; will download from UniProt if not provided)"
    )
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output path for embeddings (.pt file)")
    parser.add_argument("--cache_dir",
                        type=str,
                        default="../data/cache",
                        help="Directory to cache downloaded files")
    parser.add_argument("--model",
                        type=str,
                        default="esm2_t33_650M_UR50D",
                        help="ESM model name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--pooling",
        type=str,
        default="pool_parti",
        choices=["pool_parti", "mean"],
        help="Pooling method: pool_parti (PageRank-based, default) or mean")
    parser.add_argument(
        "--include_unreviewed",
        action="store_true",
        help=
        "Include unreviewed (TrEMBL) proteins (default: reviewed/Swiss-Prot only)"
    )

    args = parser.parse_args()

    # Default: reviewed (Swiss-Prot) only (~20k proteins)
    reviewed_only = not args.include_unreviewed

    cache_dir = Path(args.cache_dir)

    # Get FASTA file (distinct cache per mode to avoid reviewed/unreviewed mismatch)
    if args.fasta:
        fasta_path = args.fasta
    else:
        cache_name = "human_proteome_reviewed.fasta" if reviewed_only else "human_proteome_full.fasta"
        fasta_path = cache_dir / cache_name
        if not fasta_path.exists():
            download_uniprot_fasta(str(fasta_path), reviewed_only)
        else:
            print(f"Using cached FASTA: {fasta_path}")

    # Parse sequences
    gene_seqs = parse_fasta(str(fasta_path))

    # Generate embeddings
    generator = ESMEmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )

    embeddings, raw_embeddings, gene_order, failed = generator.generate_embeddings(
        gene_seqs)

    # Save
    output_path = Path(args.output)

    torch.save(
        {
            "embeddings": embeddings,
            "raw_embeddings": raw_embeddings,
            "gene_order": gene_order,
            "model_name": args.model,
            "dimension": embeddings.shape[1],
            "pooling": args.pooling,
            "standardized":
            False,  # Raw embeddings, Huber-imputed at load time
            "failed_genes": failed,
            "source": str(fasta_path),
            "num_genes": len(gene_order),
        },
        output_path)

    # Also save gene list as text file
    gene_list_path = output_path.with_suffix(".genes.txt")
    with open(gene_list_path, "w") as f:
        for gene in gene_order:
            f.write(f"{gene}\n")

    print(f"\nSaved embeddings to {output_path}")
    print(f"Saved gene list to {gene_list_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
