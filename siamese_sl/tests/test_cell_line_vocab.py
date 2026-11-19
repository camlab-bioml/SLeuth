#!/usr/bin/env python3
"""Deterministic tests for the cell-line vocabulary (cell_line_vocab).

    python3 tests/test_cell_line_vocab.py      # from siamese_sl/
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

import cell_line_vocab as clv  # noqa: E402


def test_free_text_line_names_are_lines_not_junk():
    assert clv.normalize_token(" Human HeLa cells ") == "HELA"
    assert clv.normalize_token("and OVCAR8") == "OVCAR8"
    assert clv.normalize_token("Human HA1E cells") == "HA1E"
    assert clv.parse_cell_field("293T;Human HeLa cells") == {"293T", "HELA"}
    # placeholders and cancer types are still dropped
    assert clv.parse_cell_field("TBD;BRCA;NA") == set()


def test_hela_sl_call_or_resolves_its_head():
    """BRCA2-TP53 on the shipped sidecars: SL "293T;Human HeLa cells",
    non-SL "A549;HELA". HELA is asserted both ways, so OR labels it SL; the
    old _JUNK entry supervised it as non-SL."""
    hela = clv.HEADS.index("HELA")
    lab, msk = clv.pair_label_mask("293T;Human HeLa cells", "A549;HELA", True,
                                   True)
    assert lab[hela] == 1 and msk[hela] == 1
    assert lab[clv.HEADS.index("293T")] == 1
    assert lab[clv.HEADS.index("A549")] == 0 and msk[clv.HEADS.index(
        "A549")] == 1
    assert clv.same_head_conflicts("293T;Human HeLa cells", "A549;HELA", True,
                                   True) == {hela}


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n==== {passed} passed, {failed} failed ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
