#!/usr/bin/env python3
"""
Fetch all human gene names from UniProt for expanding the SL gene matrix.
This script handles the UniProt API properly and saves the results.
"""

import urllib.request
import urllib.parse
import json
import time
import os
from typing import Set, List
import sys


def fetch_uniprot_human_genes(max_retries: int = 3) -> Set[str]:
    """
    Fetch all reviewed human gene names from UniProt.
    
    Returns:
        Set of human gene names
    """
    all_genes = set()

    # UniProt REST API endpoint for streaming results
    base_url = "https://rest.uniprot.org/uniprotkb/stream"

    # Query parameters
    query_params = {
        'query':
        '(organism_id:9606) AND (reviewed:true)',  # Human, reviewed entries only
        'fields': 'gene_primary',  # Only get primary gene names
        'format': 'tsv',  # Tab-separated values
        'size': '500'  # Batch size for pagination
    }

    # Construct URL with proper encoding
    url = base_url + '?' + urllib.parse.urlencode(query_params)

    print(f"Fetching human genes from UniProt...")
    print(f"URL: {url}")

    for attempt in range(max_retries):
        try:
            # Create request with headers
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Python SLMGAE/1.0')
            req.add_header('Accept', 'text/tab-separated-values')

            # Make request with timeout
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read().decode('utf-8')

                # Process TSV data
                lines = data.strip().split('\n')
                print(f"Retrieved {len(lines)} lines from UniProt")

                # Skip header line
                for i, line in enumerate(lines[1:], 1):
                    line = line.strip()
                    if line and line != '-' and line != '':
                        # Handle multiple gene names (separated by spaces)
                        # Take only the primary (first) gene name
                        parts = line.split()
                        if parts and parts[0]:
                            all_genes.add(parts[0])

                    # Progress indicator
                    if i % 1000 == 0:
                        print(
                            f"  Processed {i:,} entries, found {len(all_genes):,} unique genes"
                        )

                print(f"Successfully retrieved {len(all_genes):,} human genes")
                return all_genes

        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)  # Wait before retry
            else:
                print("Failed to fetch from UniProt after all retries")
                raise

        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
            else:
                raise

    return all_genes


def fetch_uniprot_alternative() -> Set[str]:
    """
    Alternative method using the search endpoint with pagination.
    """
    all_genes = set()

    print("Using alternative UniProt API method...")

    # Use search endpoint instead of stream
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    # Start with first batch
    size = 500
    offset = 0
    total_retrieved = 0

    while True:
        query_params = {
            'query': 'organism_id:9606 AND reviewed:true',
            'fields': 'gene_primary',
            'format': 'tsv',
            'size': str(size),
            'offset': str(offset)
        }

        url = base_url + '?' + urllib.parse.urlencode(query_params)

        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Python SLMGAE/1.0')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode('utf-8')
                lines = data.strip().split('\n')

                # If we get only header or no data, we're done
                if len(lines) <= 1:
                    break

                # Process entries (skip header)
                batch_genes = 0
                for line in lines[1:]:
                    line = line.strip()
                    if line and line != '-':
                        gene = line.split()[0] if ' ' in line else line
                        if gene:
                            all_genes.add(gene)
                            batch_genes += 1

                total_retrieved += batch_genes
                print(
                    f"  Retrieved batch at offset {offset}: {batch_genes} genes (total: {len(all_genes):,} unique)"
                )

                # If we got fewer than requested, we've reached the end
                if batch_genes < size:
                    break

                offset += size

                # Safety limit to prevent infinite loops
                if offset > 50000:
                    print("  Reached safety limit of 50,000 entries")
                    break

        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    print(f"Total unique human genes retrieved: {len(all_genes):,}")
    return all_genes


def load_sl_genes(filepath: str = None) -> List[str]:
    """
    Load genes from the SL network file.
    """
    if filepath is None:
        # Try to find the file in common locations
        possible_paths = [
            'data/List_Proteins_in_SL.txt',  # From project root
            '../data/List_Proteins_in_SL.txt',  # From qmd directory
            './List_Proteins_in_SL.txt'  # Current directory
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                gene_ids = [line.strip() for line in f if line.strip()]
            # File contains Entrez Gene IDs; convert to symbols for UniProt
            import csv
            mapping_path = os.path.join(os.path.dirname(filepath),
                                        'gene_id_mapping.tsv')
            if os.path.exists(mapping_path):
                entrez_to_sym = {}
                with open(mapping_path) as mf:
                    reader = csv.DictReader(mf, delimiter='\t')
                    for row in reader:
                        entrez_to_sym[row['entrez_id']] = row['symbol']
                genes = [entrez_to_sym.get(eid, eid) for eid in gene_ids]
            else:
                genes = gene_ids
            print(f"Loaded {len(genes)} genes from SL network")
            return genes
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []
    else:
        print(f"Warning: Could not find SL gene file")
        return []


def main():
    """
    Main function to fetch genes and save results.
    """
    print("=" * 60)
    print("Fetching Human Genes from UniProt")
    print("=" * 60)

    # Try primary method first
    try:
        all_human_genes = fetch_uniprot_human_genes()
    except Exception as e:
        print(f"\nPrimary method failed: {e}")
        print("Trying alternative method...")
        all_human_genes = fetch_uniprot_alternative()

    if not all_human_genes:
        print("Failed to retrieve genes from UniProt")
        sys.exit(1)

    # Load SL genes if available
    sl_genes = load_sl_genes()

    if sl_genes:
        sl_genes_set = set(sl_genes)
        new_genes = all_human_genes - sl_genes_set
        print(f"\nAnalysis:")
        print(f"  Total human genes from UniProt: {len(all_human_genes):,}")
        print(f"  Genes in SL network: {len(sl_genes_set):,}")
        print(f"  Additional human genes not in SL: {len(new_genes):,}")
        print(f"  Overlap: {len(sl_genes_set & all_human_genes):,}")

    # Save results - determine output path based on current directory
    if os.path.exists('data'):
        output_file = 'data/all_human_genes_uniprot.txt'  # From project root
    else:
        output_file = '../data/all_human_genes_uniprot.txt'  # From qmd directory

    with open(output_file, 'w') as f:
        f.write("# All reviewed human genes from UniProt\n")
        f.write(f"# Total: {len(all_human_genes)} genes\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for gene in sorted(all_human_genes):
            f.write(f"{gene}\n")

    print(f"\nSaved all human genes to: {output_file}")

    # Also save the new genes separately if SL genes were loaded
    if sl_genes and new_genes:
        if os.path.exists('data'):
            new_genes_file = 'data/human_genes_not_in_sl.txt'  # From project root
        else:
            new_genes_file = '../data/human_genes_not_in_sl.txt'  # From qmd directory

        with open(new_genes_file, 'w') as f:
            f.write("# Human genes not in SL network\n")
            f.write(f"# Total: {len(new_genes)} genes\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for gene in sorted(new_genes):
                f.write(f"{gene}\n")
        print(f"Saved non-SL genes to: {new_genes_file}")

    print("\nDone!")
    return all_human_genes


if __name__ == "__main__":
    main()
