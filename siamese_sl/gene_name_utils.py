#!/usr/bin/env python3
"""
Gene name normalization using custom corrections + HGNC database.

Applies gene name corrections in order:
  1. Custom corrections (from ../code/gene_corrections_config.py, manually verified)
  2. HGNC current symbol check (already correct, return as-is)
  3. HGNC alias/previous symbol lookup (old name -> current name)
  4. Return as-is (uppercased) if no match found

The HGNC complete set is downloaded once from EBI and cached locally.
Custom corrections from gene_corrections_config.py take priority over HGNC
because they are manually verified against UniProt.

Usage:
    from gene_name_utils import get_mapper

    mapper = get_mapper(cache_dir="../data/cache")
    mapper.gene_name_normalize("SEPT1")   # -> "SEPTIN1"  (HGNC rename 2020)
    mapper.gene_name_normalize("P53")     # -> "TP53"     (alias)
    mapper.gene_name_normalize("MARCH1")  # -> "MARCHF1"  (HGNC rename 2020)
    mapper.gene_name_normalize("BRCA1")   # -> "BRCA1"    (already current)
    mapper.gene_name_normalize("MAR-02")  # -> "MARCHF2"  (custom correction)

    # Normalize all keys in an embedding dict
    fixed = mapper.gene_name_normalize_dict({"SEPT1": vec1, "BRCA1": vec2})
    # -> {"SEPTIN1": vec1, "BRCA1": vec2}

    # Normalize a gene list
    fixed = mapper.gene_name_normalize_list(["SEPT1", "BRCA1"])
    # -> ["SEPTIN1", "BRCA1"]

    # Check if a gene is non-coding (no protein product)
    mapper.is_non_coding("LINC00029")  # -> True
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set

HGNC_URL = ("https://storage.googleapis.com/public-download-files/"
            "hgnc/tsv/tsv/hgnc_complete_set.txt")

# Import custom gene corrections from the SLMGAE code directory.
# On the server: siamese_sl/../code/ = SLMGAE-pytorch/code/
_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

try:
    from gene_corrections_config import GENE_CORRECTIONS, NON_CODING_GENES
except ImportError:
    # Graceful fallback if gene_corrections_config.py is not found.
    # HGNC-based normalization still works; only custom corrections are skipped.
    GENE_CORRECTIONS: Dict[str, str] = {}
    NON_CODING_GENES: set = set()


class GeneNameMapper:
    """Map old/alias gene symbols to current HGNC-approved symbols.

    Correction priority:
      1. Custom corrections (gene_corrections_config.py) — manually verified
         against UniProt, handles edge cases like MAR-02 -> MARCHF2
      2. HGNC current symbols — gene is already correct, return as-is
      3. HGNC previous/alias symbols — old name mapped to current name
      4. Unknown — return uppercased input unchanged

    All lookups are case-insensitive (uppercased internally).
    The HGNC TSV (~10 MB) is downloaded once and cached locally.
    """

    def __init__(self, cache_dir: str = "../data/cache"):
        self._current_symbols: Set[str] = set()
        self._alias_to_current: Dict[str, str] = {}
        self._custom_corrections: Dict[str, str] = {}
        self._non_coding: Set[str] = set()
        self._loaded = False
        self._cache_dir = cache_dir

    def _ensure_loaded(self):
        """Lazy-load all data sources on first use."""
        if self._loaded:
            return

        # 1. Load custom corrections (from gene_corrections_config.py)
        for old_name, new_name in GENE_CORRECTIONS.items():
            self._custom_corrections[old_name.upper()] = new_name.upper()
        self._non_coding = {g.upper() for g in NON_CODING_GENES}
        if self._custom_corrections:
            print(f"  Custom gene corrections: "
                  f"{len(self._custom_corrections)} mappings loaded")
        if self._non_coding:
            print(f"  Non-coding genes: "
                  f"{len(self._non_coding)} known (will get NaN embeddings)")

        # 2. Load HGNC database (download if not cached)
        hgnc_path = os.path.join(self._cache_dir, "hgnc_complete_set.txt")
        if not os.path.exists(hgnc_path):
            self._download(hgnc_path)
        if os.path.exists(hgnc_path):
            self._parse_hgnc(hgnc_path)

        self._loaded = True

    def _download(self, output_path: str):
        """Download HGNC complete set TSV from EBI FTP."""
        print("  Downloading HGNC gene name database...")
        try:
            import urllib.request
            urllib.request.urlretrieve(HGNC_URL, output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Gene name normalization will use custom corrections "
                  "only (no HGNC)")

    def _parse_hgnc(self, hgnc_path: str):
        """Parse HGNC complete set TSV.

        Builds two data structures:
          - _current_symbols: set of all current approved symbols (uppercased)
          - _alias_to_current: dict mapping old/alias symbols to current symbols

        HGNC TSV columns used:
          - symbol: current approved gene symbol
          - prev_symbol: pipe-separated previous symbols (e.g., "FOO|BAR")
          - alias_symbol: pipe-separated alias symbols (e.g., "BAZ|QUX")

        Both prev_symbol and alias_symbol may be quoted (e.g., "FOO|BAR").
        """
        n_genes = 0
        n_aliases = 0

        with open(hgnc_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            # Validate required columns exist
            if reader.fieldnames is None:
                print("  WARNING: HGNC file is empty")
                return
            required = {"symbol", "prev_symbol", "alias_symbol"}
            missing_cols = required - set(reader.fieldnames)
            if missing_cols:
                print(f"  WARNING: HGNC file missing columns: {missing_cols}")
                return

            for row in reader:
                symbol = row.get("symbol", "").strip()
                if not symbol:
                    continue

                symbol_upper = symbol.upper()
                self._current_symbols.add(symbol_upper)
                n_genes += 1

                # Parse pipe-separated, possibly quoted previous symbols
                for field in ("prev_symbol", "alias_symbol"):
                    raw = row.get(field, "")
                    if not raw:
                        continue
                    raw = raw.strip('"')
                    for alias in raw.split("|"):
                        alias = alias.strip().strip('"')
                        if not alias:
                            continue
                        alias_upper = alias.upper()
                        # Don't overwrite if this alias is itself a
                        # current approved symbol for a different gene
                        if alias_upper not in self._current_symbols:
                            self._alias_to_current[alias_upper] = symbol_upper
                            n_aliases += 1

        print(f"  HGNC loaded: {n_genes} current symbols, "
              f"{n_aliases} aliases/previous symbols")

    def gene_name_normalize(self, gene_name: str) -> str:
        """Normalize a single gene name to its current HGNC symbol.

        Applies corrections in order:
          1. Custom corrections (gene_corrections_config.py)
          2. HGNC current symbol (already correct)
          3. HGNC alias/previous symbol
          4. Return as-is (uppercased)

        Args:
            gene_name: Gene symbol to normalize (case-insensitive).

        Returns:
            Current HGNC symbol (uppercased), or input uppercased if unknown.
        """
        self._ensure_loaded()
        upper = gene_name.upper()
        # 1. Custom corrections take priority (manually verified),
        #    then re-normalize through HGNC in case the correction
        #    target is itself an outdated symbol (e.g. SEPT1 -> SEPTIN1)
        if upper in self._custom_corrections:
            corrected = self._custom_corrections[upper]
            if corrected in self._current_symbols:
                return corrected
            if corrected in self._alias_to_current:
                return self._alias_to_current[corrected]
            return corrected
        # 2. Already a current HGNC symbol — no change needed
        if upper in self._current_symbols:
            return upper
        # 3. Known HGNC alias or previous symbol
        if upper in self._alias_to_current:
            return self._alias_to_current[upper]
        # 4. Unknown — return as-is (uppercased)
        return upper

    def is_non_coding(self, gene_name: str) -> bool:
        """Check if a gene is known to be non-coding (no protein product).

        Non-coding genes include pseudogenes, lncRNAs, snoRNA host genes,
        antisense RNAs, etc. These typically don't have UniProt entries and
        will receive NaN embeddings for protein-based embedding types.
        """
        self._ensure_loaded()
        return gene_name.upper() in self._non_coding

    def gene_name_normalize_dict(self, d: dict, report: bool = True) -> dict:
        """Normalize all gene name keys in a dict.

        Args:
            d: Dict mapping gene_name -> value (e.g., gene -> embedding vector).
            report: If True, print how many genes were renamed.

        Returns:
            New dict with normalized keys. If two old names map to the same
            current symbol, the first occurrence is kept and later ones are
            dropped (logged as collisions).
        """
        self._ensure_loaded()
        normalized = {}
        renamed = 0
        collisions = 0

        for gene, value in d.items():
            new_name = self.gene_name_normalize(gene)
            if new_name != gene.upper():
                renamed += 1
            if new_name in normalized:
                collisions += 1
                continue  # keep first occurrence
            normalized[new_name] = value

        if report and (renamed > 0 or collisions > 0):
            msg = f"  Gene name normalization: {renamed} renamed"
            if collisions:
                msg += f", {collisions} collisions dropped"
            print(msg)

        return normalized

    def gene_name_normalize_list(self, genes: List[str]) -> List[str]:
        """Normalize a list of gene names to current HGNC symbols.

        Args:
            genes: List of gene symbols.

        Returns:
            List of normalized gene symbols (same length, same order).
        """
        self._ensure_loaded()
        return [self.gene_name_normalize(g) for g in genes]

    @property
    def stats(self) -> dict:
        """Return summary statistics about loaded data sources."""
        self._ensure_loaded()
        return {
            "current_symbols": len(self._current_symbols),
            "aliases": len(self._alias_to_current),
            "custom_corrections": len(self._custom_corrections),
            "non_coding": len(self._non_coding),
        }


# Module-level singleton for convenience.
# Created on first call to get_mapper(). The cache_dir from the first
# call is used; subsequent calls with different cache_dir are ignored.
_default_mapper: Optional[GeneNameMapper] = None


def get_mapper(cache_dir: str = "../data/cache") -> GeneNameMapper:
    """Get or create the default GeneNameMapper singleton.

    Args:
        cache_dir: Directory to cache the HGNC TSV download.
                   Only used on first call (when creating the singleton).
    """
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = GeneNameMapper(cache_dir=cache_dir)
    return _default_mapper
