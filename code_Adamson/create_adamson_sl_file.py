#!/usr/bin/env python3
"""
Create SL_Adamson_gamma_thresholding.txt from QMD analysis output.
Extract gene pairs where SL_score is not NaN AND Adamson_SL_status is 1.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count


def process_chunk(chunk_data):
    """Process a chunk of dataframe rows and return filtered results"""
    chunk_df, chunk_idx, total_chunks = chunk_data
    results = []
    genes = set()

    # Create progress bar for this chunk
    desc = f"Core {chunk_idx // (len(chunk_df) + 1) + 1}/{total_chunks}"

    for _, row in tqdm(
            chunk_df.iterrows(),
            total=len(chunk_df),
            desc=desc,
            position=chunk_idx // (len(chunk_df) + 1),
            leave=False,
    ):
        # Only write if SL_score is not NaN AND Adamson_Gamma_SL_status is 1
        if pd.notna(row["SL_score"]) and row.get("Adamson_Gamma_SL_status",
                                                 0) == 1:
            results.append(
                f"{row['gene1']}\t{row['gene2']}\t{row['Adamson_Gamma']}")
            genes.add(row["gene1"])
            genes.add(row["gene2"])

    return results, genes


def main():
    # Read the QMD output pickle file
    qmd_output = Path("../qmd/SL_predictions_merged_analysis.pkl")
    if not qmd_output.exists():
        print(f"Error: {qmd_output} not found! Run QMD analysis first.")
        return

    print("Reading QMD output from pickle file...")
    df = pd.read_pickle(qmd_output)
    print(f"Loaded {len(df):,} gene pairs")

    # Determine number of cores to use (n_cores - 1)
    n_cores = max(1, cpu_count() - 1)
    print(f"\nUsing {n_cores} cores for parallel processing")

    # Split dataframe into chunks for parallel processing
    chunk_size = len(df) // n_cores + 1
    chunks = [(df.iloc[i:i + chunk_size], i, n_cores)
              for i in range(0, len(df), chunk_size)]

    print("\nFiltering and writing Adamson SL pairs...")
    print(f"Each core processing ~{chunk_size:,} rows\n")

    # Process chunks in parallel
    all_results = []
    adamson_genes = set()

    with Pool(n_cores) as pool:
        # Process all chunks and collect results
        for results, genes in pool.imap(process_chunk, chunks):
            all_results.extend(results)
            adamson_genes.update(genes)

    # Clear progress bars
    print("\n" * n_cores)

    # Write all results to file
    output_file = Path("../data/SL_Adamson_gamma_thresholding.txt")
    print("Writing results to file...")
    with open(output_file, "w") as f:
        for line in all_results:
            f.write(line + "\n")

    print(f"\nFound {len(all_results):,} SL pairs from Adamson thresholding")
    print(f"Wrote to {output_file}")

    # Check for genes not in List_Proteins_in_SL.txt
    with open("../data/List_Proteins_in_SL.txt", "r") as f:
        sl_genes = set(line.strip() for line in f)

    missing = adamson_genes - sl_genes

    if missing:
        print(f"\n⚠️  {len(missing)} genes NOT in List_Proteins_in_SL.txt:")
        for gene in sorted(missing):
            print(f"  - {gene}")

        # Save missing genes to file for correction
        missing_genes_file = Path(
            "../data/Adamson_SL_genes_to_be_incorporated.txt")
        with open(missing_genes_file, "w") as f:
            for gene in sorted(missing):
                f.write(f"{gene}\n")
        print(f"\nSaved missing genes to: {missing_genes_file}")
    else:
        print("✓ All genes present in List_Proteins_in_SL.txt")


if __name__ == "__main__":
    main()
