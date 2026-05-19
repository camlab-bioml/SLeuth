#!/usr/bin/env python3
"""
Gene identifier utilities: normalization, symbol ↔ NCBI Entrez ID mapping.

All gene identifiers in this project use NCBI Entrez Gene IDs (strings)
as the canonical internal representation. Gene symbols are used only at
I/O boundaries (user-facing CLI, external data files keyed by symbol).

Gene name correction pipeline (applied when reading symbols from any source):
  1. Custom corrections (from ../code/gene_corrections_config.py, manually verified)
  2. Re-normalize correction through HGNC (in case correction target is stale)
  3. HGNC current symbol check (already correct, return as-is)
  4. HGNC alias/previous symbol lookup (old name -> current name)
  5. Return as-is (uppercased) if no match found
  6. Map corrected symbol -> Entrez Gene ID

The HGNC complete set is downloaded once from EBI and cached locally.
It contains both gene symbols and Entrez Gene IDs.

Usage:
    from gene_name_utils import get_mapper

    mapper = get_mapper(cache_dir="../data/cache")

    # Symbol normalization (step 1-5)
    mapper.gene_name_normalize("SEPT1")   # -> "SEPTIN1"  (HGNC rename 2020)
    mapper.gene_name_normalize("MAR-02")  # -> "MARCHF2"  (custom correction)

    # Symbol -> Entrez ID (normalize first, then map)
    mapper.symbol_to_entrez("BRCA1")      # -> "672"
    mapper.symbol_to_entrez("SEPT1")      # -> "55832"  (normalizes to SEPTIN1 first)

    # Entrez ID -> Symbol (reverse lookup)
    mapper.entrez_to_symbol("672")        # -> "BRCA1"

    # Batch: normalize symbols and convert to Entrez IDs
    mapper.symbols_to_entrez_list(["BRCA1", "SEPT1"])  # -> ["672", "55832"]

    # Convert a dict keyed by symbols to Entrez IDs
    mapper.symbols_to_entrez_dict({"BRCA1": vec1, "SEPT1": vec2})
    # -> {"672": vec1, "55832": vec2}
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    """Map gene symbols to current HGNC symbols and NCBI Entrez Gene IDs.

    Correction priority for symbol normalization:
      1. Custom corrections (gene_corrections_config.py) — manually verified
         against UniProt, handles edge cases like MAR-02 -> MARCHF2
      2. Re-normalize correction through HGNC (SEPT1 -> SEPTIN1)
      3. HGNC current symbols — gene is already correct, return as-is
      4. HGNC previous/alias symbols — old name mapped to current name
      5. Unknown — return uppercased input unchanged

    Entrez ID mapping:
      - symbol_to_entrez: normalized symbol -> Entrez ID string
      - entrez_to_symbol: Entrez ID string -> current HGNC symbol

    All lookups are case-insensitive (uppercased internally).
    The HGNC TSV (~10 MB) is downloaded once and cached locally.
    """

    def __init__(self, cache_dir: str = "../data/cache"):
        self._current_symbols: Set[str] = set()
        self._alias_to_current: Dict[str, str] = {}
        self._custom_corrections: Dict[str, str] = {}
        self._non_coding: Set[str] = set()

        # Entrez ID mapping (populated from HGNC)
        self._symbol_to_entrez: Dict[str,
                                     str] = {}  # UPPER symbol -> entrez_id
        self._entrez_to_symbol: Dict[str,
                                     str] = {}  # entrez_id -> UPPER symbol

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("  Downloading HGNC gene name database...")
        try:
            import urllib.request
            urllib.request.urlretrieve(HGNC_URL, output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"  Download failed: {e}")
            print("  Gene name normalization will use custom corrections "
                  "only (no HGNC)")

    def _parse_hgnc(self, hgnc_path: str):
        """Parse HGNC complete set TSV.

        Builds:
          - _current_symbols: set of all current approved symbols (uppercased)
          - _alias_to_current: dict mapping old/alias symbols to current symbols
          - _symbol_to_entrez: dict mapping current symbol to Entrez Gene ID
          - _entrez_to_symbol: dict mapping Entrez Gene ID to current symbol

        HGNC TSV columns used:
          - symbol: current approved gene symbol
          - prev_symbol: pipe-separated previous symbols
          - alias_symbol: pipe-separated alias symbols
          - entrez_id: NCBI Entrez Gene ID
        """
        n_genes = 0
        n_aliases = 0
        n_entrez = 0

        with open(hgnc_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            # Validate required columns exist
            if reader.fieldnames is None:
                print("  WARNING: HGNC file is empty")
                return
            required = {"symbol", "prev_symbol", "alias_symbol", "entrez_id"}
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

                # Entrez ID mapping
                entrez_id = row.get("entrez_id", "").strip()
                if entrez_id:
                    self._symbol_to_entrez[symbol_upper] = entrez_id
                    self._entrez_to_symbol[entrez_id] = symbol_upper
                    n_entrez += 1

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
              f"{n_aliases} aliases/previous symbols, "
              f"{n_entrez} Entrez IDs")

    # =========================================================================
    # Symbol normalization (symbol -> current HGNC symbol)
    # =========================================================================

    def gene_name_normalize(self, gene_name: str) -> str:
        """Normalize a single gene name to its current HGNC symbol.

        Applies corrections in order:
          1. Custom corrections (gene_corrections_config.py)
          2. Re-normalize correction through HGNC
          3. HGNC current symbol (already correct)
          4. HGNC alias/previous symbol
          5. Return as-is (uppercased)

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
        # 3. Already a current HGNC symbol — no change needed
        if upper in self._current_symbols:
            return upper
        # 4. Known HGNC alias or previous symbol
        if upper in self._alias_to_current:
            return self._alias_to_current[upper]
        # 5. Unknown — return as-is (uppercased)
        return upper

    # =========================================================================
    # Entrez ID mapping
    # =========================================================================

    def symbol_to_entrez(self, gene_name: str) -> Optional[str]:
        """Convert a gene symbol to its NCBI Entrez Gene ID.

        The symbol is normalized first (custom corrections + HGNC), then
        mapped to its Entrez ID.

        Args:
            gene_name: Gene symbol (case-insensitive). Aliases and old names
                are accepted — they are normalized first.

        Returns:
            Entrez Gene ID as a string (e.g., "672" for BRCA1), or None if
            the gene has no Entrez ID in HGNC.
        """
        self._ensure_loaded()
        normalized = self.gene_name_normalize(gene_name)
        return self._symbol_to_entrez.get(normalized)

    def entrez_to_symbol(self, entrez_id: str) -> Optional[str]:
        """Convert an NCBI Entrez Gene ID to its current HGNC symbol.

        Args:
            entrez_id: Entrez Gene ID as a string (e.g., "672").

        Returns:
            Current HGNC symbol (uppercased), or None if not found.
        """
        self._ensure_loaded()
        return self._entrez_to_symbol.get(str(entrez_id))

    # =========================================================================
    # Batch operations (symbol -> Entrez)
    # =========================================================================

    def symbols_to_entrez_list(
            self,
            genes: List[str],
            drop_unmapped: bool = False) -> Tuple[List[str], List[str]]:
        """Convert a list of gene symbols to Entrez IDs.

        Each symbol is normalized first, then mapped to Entrez ID.

        Args:
            genes: List of gene symbols.
            drop_unmapped: If True, unmapped genes are omitted (returns
                shorter list). If False, unmapped genes get None.

        Returns:
            (entrez_ids, unmapped_symbols): entrez_ids is a list of Entrez
            ID strings (same order as input unless drop_unmapped=True).
            unmapped_symbols lists genes that could not be mapped.
        """
        self._ensure_loaded()
        entrez_ids = []
        unmapped = []
        for gene in genes:
            eid = self.symbol_to_entrez(gene)
            if eid is not None:
                entrez_ids.append(eid)
            else:
                unmapped.append(gene)
                if not drop_unmapped:
                    entrez_ids.append(None)
        if unmapped:
            print(f"  Warning: {len(unmapped)} genes could not be mapped to "
                  f"Entrez IDs: {unmapped[:10]}"
                  f"{'...' if len(unmapped) > 10 else ''}")
        return entrez_ids, unmapped

    def symbols_to_entrez_dict(self, d: dict, report: bool = True) -> dict:
        """Convert dict keys from gene symbols to Entrez IDs.

        Each symbol key is normalized first, then mapped to Entrez ID.
        Keys that cannot be mapped are dropped with a warning.

        Args:
            d: Dict mapping gene_symbol -> value.
            report: If True, print mapping statistics.

        Returns:
            New dict mapping entrez_id (str) -> value.
        """
        self._ensure_loaded()
        result = {}
        mapped = 0
        unmapped = 0
        collisions = 0

        for gene, value in d.items():
            eid = self.symbol_to_entrez(gene)
            if eid is None:
                unmapped += 1
                continue
            if eid in result:
                collisions += 1
                continue
            result[eid] = value
            mapped += 1

        if report and (unmapped > 0 or collisions > 0):
            msg = f"  Symbol->Entrez mapping: {mapped} mapped"
            if unmapped:
                msg += f", {unmapped} unmapped (dropped)"
            if collisions:
                msg += f", {collisions} collisions (dropped)"
            print(msg)

        return result

    # =========================================================================
    # Batch operations (symbol normalization only)
    # =========================================================================

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

    # =========================================================================
    # Utilities
    # =========================================================================

    def is_non_coding(self, gene_name: str) -> bool:
        """Check if a gene is known to be non-coding (no protein product)."""
        self._ensure_loaded()
        return gene_name.upper() in self._non_coding

    @property
    def stats(self) -> dict:
        """Return summary statistics about loaded data sources."""
        self._ensure_loaded()
        return {
            "current_symbols": len(self._current_symbols),
            "aliases": len(self._alias_to_current),
            "custom_corrections": len(self._custom_corrections),
            "non_coding": len(self._non_coding),
            "entrez_ids": len(self._symbol_to_entrez),
        }


# Module-level singleton for convenience.
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
    elif cache_dir != _default_mapper._cache_dir:
        import warnings
        warnings.warn(
            f"get_mapper() called with cache_dir={cache_dir!r} but singleton "
            f"already exists with cache_dir={_default_mapper._cache_dir!r}. "
            f"Returning existing mapper; new cache_dir is ignored.",
            stacklevel=2,
        )
    return _default_mapper
