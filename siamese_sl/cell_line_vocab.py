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
in another simply sets the two heads independently — no contradiction.

SAME-HEAD DISAGREEMENT (SL *and* non-SL in the same head) resolves by OR:
label 1, supervised. Two distinct situations land here and OR is right for both.

  * At a named head it is a genuine screen disagreement between two labs — but
    an SL call is a positive assay readout while a non-SL call is a failure to
    detect, so the evidence is not symmetric; "SL in at least one screen" is the
    label the field uses and the one the ``*_any`` headline is graded on.
  * At ``OTHER`` it is usually not a disagreement at all, just head collapse:
    SL in RPE1 and non-SL in DLD-1 are different lines that share a catch-all.

Masking these out instead (``same_head="mask"``, the pre-August-2026 default)
un-supervised 1,876 of the 5,542 pairs screened both ways on at least one head
— 1,461 of them at JURKAT alone, which alone moved that head from 2,617
positives to 1,156 — and left 316 pairs with no supervised head at all, i.e.
dropped from the dataset. (316, not the 107 in meta.json: that is
``num_conflicts_missing_cell_line``, a different quantity — pairs naming no
usable line on one side.) Keep "mask" only to reproduce the older behaviour.
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
    "",
    '""',
    "TBD",
    "NA",
    "-",
    "BRCA",
    "LAML",
    "KIRC",
    "HR-DEFECTS",
    "AND OVCAR8",
    "HUMAN HELA CELLS",
    "BRCA-DEFICIENT",
    "C538231",
    "PKC_C537180",
    "GINGIVAL C",
    "CAS",
    "EMBRYONIC LETHALITY",
    "NEUROB",
    "BREAST AND OVA",
    "HEPATOCELLULAR CA",
    "HUMAN HA1E CELLS",
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


SAME_HEAD_POLICIES = ("positive", "mask")


def pair_label_mask(
    sl_field: str,
    nonsl_field: str,
    is_sl: bool,
    is_nonsl: bool,
    same_head: str = "positive",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-head (label, mask) vectors for one gene pair.

    Args:
        sl_field: raw cell_line field from the SL sidecar ("" if not SL).
        nonsl_field: raw cell_line field from the non-SL sidecar.
        is_sl: pair appears in the SL set.
        is_nonsl: pair appears in the non-SL set.
        same_head: how to resolve a head that is asserted BOTH ways.
            "positive" (default) — OR: label 1, supervised. See the module
            docstring for why the evidence is asymmetric.
            "mask" — legacy: leave the head unsupervised.

    Returns:
        (label, mask), each shape (NUM_HEADS,) float32.
        mask[h]=1 means head h is supervised; label[h] in {0,1}.
    """
    if same_head not in SAME_HEAD_POLICIES:
        raise ValueError(
            f"same_head must be one of {SAME_HEAD_POLICIES}, got {same_head!r}"
        )
    sl_heads = heads_for_field(sl_field, is_sl)
    ns_heads = heads_for_field(nonsl_field, is_nonsl)
    label = np.zeros(NUM_HEADS, dtype=np.float32)
    mask = np.zeros(NUM_HEADS, dtype=np.float32)
    for h in range(NUM_HEADS):
        in_sl = h in sl_heads
        in_ns = h in ns_heads
        if in_sl and in_ns and same_head == "mask":
            continue  # legacy: disagreement in this head -> unknown
        if in_sl:  # OR: an SL call anywhere in the head wins
            label[h] = 1.0
            mask[h] = 1.0
        elif in_ns:
            label[h] = 0.0
            mask[h] = 1.0
    return label, mask


def same_head_conflicts(sl_field: str, nonsl_field: str, is_sl: bool,
                        is_nonsl: bool) -> Set[int]:
    """Heads asserted BOTH ways for this pair — what `same_head` resolves.

    Reported (never used to label) so a run can state how many head-cells the
    OR rule recovered rather than leaving it to be re-derived from the sidecars.
    """
    return (heads_for_field(sl_field, is_sl)
            & heads_for_field(nonsl_field, is_nonsl))
