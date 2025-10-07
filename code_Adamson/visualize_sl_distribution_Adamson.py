#!/usr/bin/env python3
"""
Visualize the distribution of how many SL partners each gene has in SL_Adamson_gamma_thresholding.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def main():
    # Load Adamson SL data
    sl_file = Path("../data/SL_Adamson_gamma_thresholding.txt")
    gene_sl_counts = defaultdict(int)
    gamma_scores = []

    with open(sl_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:  # gene1, gene2, gamma_score
                gene_sl_counts[parts[0]] += 1
                gene_sl_counts[parts[1]] += 1
                gamma_scores.append(float(parts[2]))

    # Get counts
    sl_degrees = list(gene_sl_counts.values())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: SL degree distribution
    ax1.hist(sl_degrees,
             bins="auto",
             edgecolor="black",
             alpha=0.7,
             color="darkgreen")
    ax1.set_xlabel("Number of SL Partners per Gene", fontsize=12)
    ax1.set_ylabel("Number of Genes", fontsize=12)
    ax1.set_title(
        f"Adamson SL Partner Distribution\n{len(gene_sl_counts):,} genes with at least 1 SL partner",
        fontsize=14,
    )
    ax1.grid(True, alpha=0.3)

    # Add mean and median lines to first plot
    ax1.axvline(
        np.mean(sl_degrees),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(sl_degrees):.1f}",
    )
    ax1.axvline(
        np.median(sl_degrees),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(sl_degrees):.0f}",
    )
    ax1.legend(fontsize=11)

    # Plot 2: Gamma score distribution
    ax2.hist(gamma_scores,
             bins=50,
             edgecolor="black",
             alpha=0.7,
             color="darkred")
    ax2.set_xlabel("Adamson Gamma Score", fontsize=12)
    ax2.set_ylabel("Number of SL Pairs", fontsize=12)
    ax2.set_title(
        f"Adamson Gamma Score Distribution\n{len(gamma_scores):,} SL pairs",
        fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add mean and median lines to second plot
    ax2.axvline(
        np.mean(gamma_scores),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(gamma_scores):.3f}",
    )
    ax2.axvline(
        np.median(gamma_scores),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(gamma_scores):.3f}",
    )
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("adamson_sl_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    print("=" * 60)
    print("Adamson SL Network Summary")
    print("=" * 60)
    print(f"Total SL pairs: {len(gamma_scores):,}")
    print(f"Total genes with SL partners: {len(gene_sl_counts):,}")
    print(f"\nSL Degree Statistics:")
    print(f"  Mean: {np.mean(sl_degrees):.2f} partners per gene")
    print(f"  Median: {np.median(sl_degrees):.0f} partners per gene")
    print(f"  Max: {max(sl_degrees)} partners")
    print(f"  Min: {min(sl_degrees)} partners")
    print(f"\nGamma Score Statistics:")
    print(f"  Mean: {np.mean(gamma_scores):.4f}")
    print(f"  Median: {np.median(gamma_scores):.4f}")
    print(f"  Max: {max(gamma_scores):.4f}")
    print(f"  Min: {min(gamma_scores):.4f}")

    # Top hub genes
    hub_genes = sorted(gene_sl_counts.items(),
                       key=lambda x: x[1],
                       reverse=True)[:5]
    print("\nTop 5 hub genes:")
    for gene, count in hub_genes:
        print(f"  {gene}: {count} partners")


if __name__ == "__main__":
    main()
