#!/usr/bin/env python3
"""
ESM-2 embedding generation script optimized for GPU cluster.
Generates 1280-dimensional mean-pooled embeddings using ESM-2 model.

Usage:
    python generate_esm_embeddings_gpu.py [--batch_size 8] [--device cuda:0]
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import argparse
import time
from gene_corrections_config import GENE_CORRECTIONS, NON_CODING_GENES


class ESMEmbeddingGenerator:
    """Generate ESM-2 embeddings for protein sequences on GPU."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = None,
        batch_size: int = 8,
    ):
        """Initialize ESM model.

        Args:
            model_name: ESM model to use
            device: Device to run on (cuda:0, cuda:1, etc)
            batch_size: Batch size for processing sequences
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect GPU if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"=" * 60)
        print(f"ESM Embedding Generator (GPU Optimized)")
        print(f"=" * 60)
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")

        if "cuda" in str(self.device):
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        print(f"Loading ESM model...")

        # Import ESM
        try:
            import esm
        except ImportError:
            print("ESM not installed. Installing...")
            os.system("pip install fair-esm")
            import esm

        # Load model
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        print(f"Model loaded. Embedding dimension: 1280")
        print(f"Max sequence length: 1022 (will truncate longer sequences)")

        # Use centralized gene corrections
        self.gene_corrections = GENE_CORRECTIONS

    def is_noncoding_gene(self, gene: str) -> bool:
        """Check if a gene is non-coding."""
        return gene in NON_CODING_GENES

    def correct_gene_name(self, gene: str) -> str:
        """Apply gene name corrections."""
        return self.gene_corrections.get(gene, gene)

    def read_fasta(self, fasta_file: str) -> Dict[str, str]:
        """Read sequences from FASTA file."""
        sequences = {}
        current_gene = None
        current_seq = []

        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_gene:
                        sequences[current_gene] = "".join(current_seq)
                    current_gene = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_gene:
                sequences[current_gene] = "".join(current_seq)

        return sequences

    def generate_embeddings_batch(
            self, batch_data: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for a batch of sequences."""
        if not batch_data:
            return {}

        # Convert batch
        batch_labels, batch_strs, batch_tokens = self.batch_converter(
            batch_data)
        batch_tokens = batch_tokens.to(self.device)

        # Generate embeddings
        with torch.no_grad():
            results = self.model(batch_tokens,
                                 repr_layers=[33],
                                 return_contacts=False)
            token_embeddings = results["representations"][33]

        # Mean pooling for each sequence
        embeddings = {}
        for i, (gene, seq) in enumerate(batch_data):
            sequence_length = min(len(seq), 1022)  # Account for truncation
            # Mean pooling (excluding special tokens)
            embedding = token_embeddings[i, 1:sequence_length + 1].mean(0)
            embeddings[gene] = embedding.cpu().numpy()

        return embeddings

    def process_dataset(self,
                        fasta_file: str,
                        output_file: str,
                        gene_order_file: str = None) -> Dict[str, np.ndarray]:
        """Process a dataset and generate embeddings with batching."""
        print(f"\nProcessing {fasta_file}...")

        # Read sequences
        sequences = self.read_fasta(fasta_file)
        print(f"Loaded {len(sequences)} sequences from FASTA")

        # Get gene order if specified
        if gene_order_file and os.path.exists(gene_order_file):
            with open(gene_order_file, "r") as f:
                gene_order = [line.strip() for line in f if line.strip()]
            print(f"Using gene order from {gene_order_file}")
        else:
            gene_order = sorted(sequences.keys())
            print("Using alphabetical gene order")

        print(f"Total genes to process: {len(gene_order)}")

        # Initialize storage
        all_embeddings = {}
        failed_genes = []
        noncoding_genes = []
        corrected_genes = []
        placeholder_genes = []

        # Process genes in batches
        batch_data = []
        batch_genes = []

        start_time = time.time()

        with tqdm(total=len(gene_order), desc="Generating embeddings") as pbar:
            for gene_idx, gene in enumerate(gene_order):
                original_gene = gene

                # Step 1: Check if it's a non-coding gene
                if self.is_noncoding_gene(gene):
                    # Use ZERO embedding for non-coding genes
                    all_embeddings[gene] = np.zeros(1280, dtype=np.float32)
                    noncoding_genes.append(gene)
                    pbar.update(1)
                    continue

                # Step 2: Try to correct gene name if needed
                corrected_gene = self.correct_gene_name(gene)
                if corrected_gene != gene:
                    corrected_genes.append((gene, corrected_gene))
                    if corrected_gene in sequences:
                        gene = corrected_gene
                    elif gene not in sequences:
                        gene = corrected_gene

                # Step 3: Get sequence
                if gene not in sequences and original_gene not in sequences:
                    # Gene not found - use zero embedding
                    all_embeddings[original_gene] = np.zeros(1280,
                                                             dtype=np.float32)
                    failed_genes.append(original_gene)
                    pbar.update(1)
                    continue

                seq = sequences.get(gene, sequences.get(original_gene, ""))

                # Check if this is a placeholder sequence
                if (seq in ["Mplaceholder", "MPLACEHOLDER", "X", "XXXX"]
                        or seq.upper() == "MPLACEHOLDER"):
                    all_embeddings[original_gene] = np.zeros(1280,
                                                             dtype=np.float32)
                    placeholder_genes.append(original_gene)
                    pbar.update(1)
                    continue

                # Check sequence validity
                if len(seq) == 0:
                    all_embeddings[original_gene] = np.zeros(1280,
                                                             dtype=np.float32)
                    failed_genes.append(original_gene)
                    pbar.update(1)
                    continue

                # Truncate if needed
                if len(seq) > 1022:
                    seq = seq[:1022]

                # Add to batch
                batch_data.append((original_gene, seq))
                batch_genes.append(original_gene)

                # Process batch when full or at end
                if (len(batch_data) >= self.batch_size
                        or gene_idx == len(gene_order) - 1):
                    try:
                        batch_embeddings = self.generate_embeddings_batch(
                            batch_data)
                        all_embeddings.update(batch_embeddings)
                    except Exception as e:
                        print(f"\nError processing batch: {e}")
                        # Fallback to zero embeddings for this batch
                        for g in batch_genes:
                            all_embeddings[g] = np.zeros(1280,
                                                         dtype=np.float32)
                            failed_genes.append(g)

                    # Clear batch
                    batch_data = []
                    batch_genes = []

                    # Update progress
                    pbar.update(
                        len(batch_embeddings) if "batch_embeddings" in
                        locals() else 0)

                    # Print stats periodically
                    if gene_idx % 500 == 0 and gene_idx > 0:
                        elapsed = time.time() - start_time
                        rate = gene_idx / elapsed
                        remaining = (len(gene_order) - gene_idx) / rate
                        print(
                            f"\n[{gene_idx}/{len(gene_order)}] Rate: {rate:.1f} genes/sec, "
                            f"ETA: {remaining/60:.1f} min")

        # Convert to matrix in correct order
        embedding_matrix = np.stack(
            [all_embeddings[gene] for gene in gene_order])

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Normalize embeddings: zero mean, then normalize to unit norm
        print(f"\nNormalizing embeddings...")
        # Only normalize non-zero embeddings (skip placeholders/non-coding)
        non_zero_mask = np.any(embedding_matrix != 0, axis=1)

        if np.any(non_zero_mask):
            # Step 1: Center to zero mean (per feature across all genes)
            non_zero_embeddings = embedding_matrix[non_zero_mask]
            mean = np.mean(non_zero_embeddings, axis=0, keepdims=True)
            embedding_matrix[non_zero_mask] = non_zero_embeddings - mean

            # Step 2: Normalize each embedding to unit norm
            for i in range(len(embedding_matrix)):
                if non_zero_mask[i]:
                    norm = np.linalg.norm(embedding_matrix[i])
                    if norm > 0:
                        embedding_matrix[i] = embedding_matrix[i] / norm

            print(
                f"  - Centered {np.sum(non_zero_mask)} embeddings to zero mean"
            )
            print(f"  - Normalized to unit norm")

            # Verify normalization
            normalized_embeddings = embedding_matrix[non_zero_mask]
            norms = np.linalg.norm(normalized_embeddings, axis=1)
            print(
                f"  - Norm check: mean={np.mean(norms):.6f}, std={np.std(norms):.6f}"
            )
            print(f"  - Mean check: mean={np.mean(normalized_embeddings):.6f}")

        # Save as PyTorch tensor with metadata
        print(f"\nSaving normalized embeddings to {output_file}...")
        torch.save(
            {
                "embeddings": torch.from_numpy(embedding_matrix).float(),
                "gene_order": gene_order,
                "model_name": self.model_name,
                "dimension": 1280,
                "pooling": "mean",
                "normalized": True,
                "normalization": "zero_mean_unit_norm",
                "noncoding_genes": noncoding_genes,
                "placeholder_genes": placeholder_genes,
                "corrected_genes": corrected_genes,
                "failed_genes": failed_genes,
                "embedding_type": "esm_with_zero_for_noncoding_normalized",
                "device_used": str(self.device),
                "batch_size": self.batch_size,
            },
            output_file,
        )

        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Embedding Generation Complete!")
        print(f"{'='*60}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Processing rate: {len(gene_order)/elapsed:.1f} genes/sec")
        print(f"Output saved to: {output_file}")
        print(f"Shape: {embedding_matrix.shape}")

        print(f"\nStatistics:")
        print(f"  Total genes: {len(gene_order)}")
        print(
            f"  Protein sequences processed: {len(gene_order) - len(noncoding_genes) - len(placeholder_genes) - len(failed_genes)}"
        )
        print(f"  Non-coding genes (zero): {len(noncoding_genes)}")
        print(f"  Placeholder genes (zero): {len(placeholder_genes)}")
        print(f"  Failed genes (zero): {len(failed_genes)}")
        print(f"  Gene names corrected: {len(corrected_genes)}")

        # Count non-zero embeddings
        non_zero = np.sum(np.any(embedding_matrix != 0, axis=1))
        print(
            f"  Non-zero embeddings: {non_zero}/{len(gene_order)} ({non_zero/len(gene_order)*100:.1f}%)"
        )

        return all_embeddings


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate ESM embeddings on GPU")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        help="Device to use (e.g., cuda:0, cuda:1)")
    parser.add_argument("--model",
                        type=str,
                        default="esm2_t33_650M_UR50D",
                        help="ESM model to use")
    args = parser.parse_args()

    # Initialize generator
    generator = ESMEmbeddingGenerator(model_name=args.model,
                                      device=args.device,
                                      batch_size=args.batch_size)

    # Step 1: Generate embeddings for main dataset
    print("\n" + "=" * 60)
    print("Step 1: Generating ESM embeddings for main SL dataset")
    print("=" * 60)

    main_embeddings = generator.process_dataset(
        fasta_file="../protein_seq/main_protein_seq.fasta",
        output_file="../ESM_embedding/main_esm_embeddings.pt",
        gene_order_file="../data/List_Proteins_in_SL.txt",
    )

    # Step 2: Create compatibility file for graph transformer
    print("\n" + "=" * 60)
    print("Step 2: Creating compatibility file for graph transformer")
    print("=" * 60)

    main_data = torch.load(
        "../ESM_embedding/main_esm_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.save(main_data["embeddings"],
               "../data/embeddings_esm2_t33_650M_UR50D_meanpool.pt")

    with open("../data/esm_gene_order.txt", "w") as f:
        for gene in main_data["gene_order"]:
            f.write(f"{gene}\n")

    print("Created ../data/embeddings_esm2_t33_650M_UR50D_meanpool.pt")

    # Final summary
    num_genes = len(main_data["gene_order"])
    print("\n" + "=" * 60)
    print("ESM EMBEDDINGS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  1. ../ESM_embedding/main_esm_embeddings.pt ({num_genes} genes)")
    print(
        "  2. ../data/embeddings_esm2_t33_650M_UR50D_meanpool.pt (compatibility)"
    )
    print("\nYou can now run:")
    print("  python train_slmgae_with_esm.py --use_esm")


if __name__ == "__main__":
    main()
