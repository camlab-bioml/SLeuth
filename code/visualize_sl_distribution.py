#!/usr/bin/env python3
"""
Visualize the distribution of how many SL partners each gene has in SL_Human_Approved.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def main():
    # Load SL data
    sl_file = Path("../data/SL_Human_Approved.txt")
    gene_sl_counts = defaultdict(int)

    with open(sl_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if (len(parts) >= 3 and float(parts[2])
                    != 0):  # Only count non-zero scores as SL
                gene_sl_counts[parts[0]] += 1
                gene_sl_counts[parts[1]] += 1

    # Get counts
    sl_degrees = list(gene_sl_counts.values())

    # Plot histogram
    plt.figure(figsize=(12, 7))
    plt.hist(sl_degrees,
             bins="auto",
             edgecolor="black",
             alpha=0.7,
             color="steelblue")
    plt.xlabel("Number of SL Partners per Gene", fontsize=12)
    plt.ylabel("Number of Genes", fontsize=12)
    plt.title(
        f"Distribution of SL Partners\n{len(gene_sl_counts):,} genes with at least 1 SL partner",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)

    # Add mean and median lines
    plt.axvline(
        np.mean(sl_degrees),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(sl_degrees):.1f}",
    )
    plt.axvline(
        np.median(sl_degrees),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(sl_degrees):.0f}",
    )
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("sl_partner_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    print(f"Total genes with SL partners: {len(gene_sl_counts):,}")
    print(f"Mean: {np.mean(sl_degrees):.2f} partners")
    print(f"Median: {np.median(sl_degrees):.0f} partners")
    print(f"Max: {max(sl_degrees)} partners")

    # Top hub genes
    hub_genes = sorted(gene_sl_counts.items(),
                       key=lambda x: x[1],
                       reverse=True)[:5]
    print("\nTop 5 hub genes:")
    for gene, count in hub_genes:
        print(f"  {gene}: {count} partners")


if __name__ == "__main__":
    main()
