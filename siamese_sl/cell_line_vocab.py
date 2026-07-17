"""Cell-line heads for masked multi-label SL prediction (concat-at-input).

The model predicts SL for a gene pair across a fixed set of cell-line
*categories* at once. Each category is one output head AND one trainable
embedding that is concatenated onto the gene features at the encoder input, so
cell context interacts with the biological embedding inside the network.

Heads (decided from the experimental-set statistics, June 2026):
  * Seven named lines that have enough samples in BOTH classes to be learnable
    targets: K562, JURKAT, A549, HELA, A375, 293T, PC9.
  * One catch-all ``OTHER`` head that absorbs (a) every rare / single-class
    recognised line (RPE1, MeWo, DLD-1, the CRISPR panel, …) and (b) pairs with
    no usable cell line at all ("MISSING"). MISSING cannot be its own head — it
    has zero negatives (every screened non-SL pair names a line), so a MISSING
    head would be single-class and degenerate; folding it into OTHER keeps that
    data while leaving OTHER with both classes.

Labels are partial: for a pair we only know the lines it was actually screened
in. ``pair_label_mask`` returns, per head, a label in {0,1} and a mask in {0,1}
(1 = known/supervised, 0 = unknown). A pair labelled SL in one line and non-SL
in another simply sets the two heads independently — no contradiction. A pair
labelled BOTH SL and non-SL in the SAME head (genuine screen disagreement) is
masked out for that head.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# --- Heads: 7 named lines + OTHER catch-all (absorbs rare lines + MISSING) ---
# Order is fixed; do not reorder (head indices are persisted into trained
# weights and reported per-head).
NAMED_HEADS: List[str] = [
    "K562",
    "JURKAT",
    "A549",
    "HELA",
    "A375",
    "293T",
    "PC9",
]
OTHER_HEAD = "OTHER"
HEADS: List[str] = NAMED_HEADS + [OTHER_HEAD]
NUM_HEADS: int = len(HEADS)
OTHER_HEAD_IDX: int = HEADS.index(OTHER_HEAD)

_LINE_TO_HEAD: Dict[str, int] = {name: i for i, name in enumerate(NAMED_HEADS)}

# --- Raw-field normalisation (free-text cell_line column) --------------------
_MERGE: Dict[str, str] = {
    "X293T": "293T",
}
# Tokens that are NOT cell lines (cancer types, phenotypes, placeholders).
_JUNK: Set[str] = {
    "", '""', "TBD", "NA", "-",
    "BRCA", "LAML", "KIRC",
    "HR-DEFECTS", "AND OVCAR8", "HUMAN HELA CELLS", "BRCA-DEFICIENT",
    "C538231", "PKC_C537180", "GINGIVAL C", "CAS", "EMBRYONIC LETHALITY",
    "NEUROB", "BREAST AND OVA", "HEPATOCELLULAR CA", "HUMAN HA1E CELLS",
}


def normalize_token(raw: str) -> Optional[str]:
    """Normalise one raw cell-line token to a canonical name, or None if junk."""
    t = raw.strip().upper()
    if t in _JUNK:
        return None
    return _MERGE.get(t, t)


def parse_cell_field(raw_field: str) -> Set[str]:
    """Parse a ``;``-joined raw cell_line field into canonical line names.

    Drops junk/placeholder tokens. Empty set means "no usable cell line".
    """
    out: Set[str] = set()
    if not raw_field:
        return out
    for tok in raw_field.split(";"):
        norm = normalize_token(tok)
        if norm is not None:
            out.add(norm)
    return out


def head_for_line(canonical_line: str) -> int:
    """Map a canonical line name to its head index (rare lines -> OTHER)."""
    return _LINE_TO_HEAD.get(canonical_line, OTHER_HEAD_IDX)


def heads_for_field(raw_field: str, is_member: bool) -> Set[int]:
    """Heads implied by one side's (SL or non-SL) raw cell field.

    ``is_member`` is whether the pair is in that relation at all. A member with
    no usable line ("MISSING") contributes the OTHER head; a non-member
    contributes nothing.
    """
    if not is_member:
        return set()
    lines = parse_cell_field(raw_field)
    if not lines:
        return {OTHER_HEAD_IDX}  # MISSING -> OTHER
    return {head_for_line(l) for l in lines}


def pair_label_mask(
    sl_field: str,
    nonsl_field: str,
    is_sl: bool,
    is_nonsl: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-head (label, mask) vectors for one gene pair.

    Args:
        sl_field: raw cell_line field from the SL sidecar ("" if not SL).
        nonsl_field: raw cell_line field from the non-SL sidecar.
        is_sl: pair appears in the SL set.
        is_nonsl: pair appears in the non-SL set.

    Returns:
        (label, mask), each shape (NUM_HEADS,) float32.
        mask[h]=1 means head h is supervised; label[h] in {0,1}.
        A head known SL AND non-SL (same-head disagreement) is masked out.
    """
    sl_heads = heads_for_field(sl_field, is_sl)
    ns_heads = heads_for_field(nonsl_field, is_nonsl)
    label = np.zeros(NUM_HEADS, dtype=np.float32)
    mask = np.zeros(NUM_HEADS, dtype=np.float32)
    for h in range(NUM_HEADS):
        in_sl = h in sl_heads
        in_ns = h in ns_heads
        if in_sl and in_ns:
            continue  # genuine disagreement in this head -> unknown
        if in_sl:
            label[h] = 1.0
            mask[h] = 1.0
        elif in_ns:
            label[h] = 0.0
            mask[h] = 1.0
    return label, mask
