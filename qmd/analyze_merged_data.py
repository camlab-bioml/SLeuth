#!/usr/bin/env python3
import pandas as pd

df = pd.read_pickle('SL_predictions_merged_analysis.pkl')
print(f"Dataframe: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

datasets = {
    'SLDB': 'main_without_ESM',
    'Adamson': 'Adamson_Gamma',
    'Corn': 'Corn_score',
    'Gilbert': 'Gilbert_score'
}

for name, col in datasets.items():
    if col not in df.columns:
        print(f"{name}: Column '{col}' not found")
        continue

    # Get non-NA indices for this dataset
    non_na_idx = df[col].notna()

    # Combine gene1 and gene2 to get total unique genes
    all_genes = pd.concat(
        [df.loc[non_na_idx, 'gene1'], df.loc[non_na_idx, 'gene2']]).unique()
    unique_genes_total = len(all_genes)
    num_pairs = non_na_idx.sum()

    print(f"{name}:")
    print(f"  Unique genes: {unique_genes_total:,}")
    print(f"  Gene pairs:   {num_pairs:,}")
