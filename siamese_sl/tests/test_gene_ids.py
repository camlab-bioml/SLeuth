#!/usr/bin/env python3
"""Deterministic tests for gene-id mapping (gene_name_utils) and the
symbol -> Entrez migration script (convert_to_entrez).

Each test builds a tiny HGNC table in a temp dir, so nothing here reads the
tracked data/cache/hgnc_complete_set.txt or writes under data/.

    python3 tests/test_gene_ids.py      # from siamese_sl/
"""
import ast
import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

import convert_to_entrez  # noqa: E402
import eval_finetuning as ef  # noqa: E402
from gene_name_utils import GeneNameMapper  # noqa: E402

_HEADER = "symbol\tprev_symbol\talias_symbol\tentrez_id\n"
# SARS / TAZ are each one gene's previous approved symbol and another gene's
# alias; the tracked HGNC file used to send them to the alias owner.
_ROWS = [
    "SARS1\tSARS\t\t6301\n",
    "SARS2\t\tSARS|SARSM|SERRS\t54938\n",
    "WWTR1\t\tTAZ\t25937\n",
    "TAFAZZIN\tTAZ\t\t6901\n",
    "ATM\t\tATD\t472\n",
    "BRCA1\t\t\t672\n",
]


def _mapper(tmp, rows):
    with open(os.path.join(tmp, "hgnc_complete_set.txt"), "w") as f:
        f.write(_HEADER)
        f.writelines(rows)
    m = GeneNameMapper(cache_dir=tmp)
    with contextlib.redirect_stdout(io.StringIO()):
        m._ensure_loaded()
    return m


def test_prev_symbol_outranks_alias_in_either_row_order():
    for rows in (_ROWS, list(reversed(_ROWS))):
        with tempfile.TemporaryDirectory() as tmp:
            m = _mapper(tmp, rows)
            assert m.symbol_to_entrez("SARS") == "6301"  # SARS1, not SARS2
            assert m.symbol_to_entrez("TAZ") == "6901"  # TAFAZZIN, not WWTR1
            assert m.symbol_to_entrez("SARSM") == "54938"  # plain alias
            assert m.symbol_to_entrez("sars2") == "54938"  # approved symbol


# QARS and MUM1 are each a PREVIOUS symbol of two genes: the mapper keeps the
# last row, so the tracked HGNC file sends QARS to EPRS1.
_TIED = [
    "QARS1\tQARS\t\t5859\n",
    "EPRS1\tQPRS|QARS|EPRS\t\t2058\n",
    "IRF4\tMUM1\t\t3662\n",
    "PWWP3A\tMUM1\t\t84939\n",
    "BRCA1\t\t\t672\n",
]


def test_screen_symbols_two_genes_claim_are_pinned_in_either_row_order():
    for rows in (_TIED, list(reversed(_TIED))):
        with tempfile.TemporaryDirectory() as tmp:
            m = _mapper(tmp, rows)
            # the mapper alone follows the row order ...
            assert m.symbol_to_entrez("QARS") == (
                "2058" if rows is _TIED else "5859")
            assert m.symbol_to_entrez("MUM1") == (
                "84939" if rows is _TIED else "3662")
            # ... the screens' lookup does not
            names = ["QARS", "qars", "EPRS", "QARS1", "MUM1", "BRCA1", "NOPE"]
            got = ef._symbols_to_entrez(pd.Series(names), m)
            assert got.tolist()[:6] == ["5859", "5859", "2058", "5859",
                                        "84939", "672"], got.tolist()
            assert got.isna().tolist() == [False] * 6 + [True]
            assert [ef.symbol_entrez(m, n) for n in names[:6]] == \
                got.tolist()[:6]
    # without a mapper only a pinned name is known
    assert ef.symbol_entrez(None, "QARS") == "5859"
    assert ef.symbol_entrez(None, "BRCA1") is None


def test_prepare_dataset_scores_a_pinned_symbol_as_its_own_gene():
    """The rows of a screen that names QARS carry QARS1's Entrez ID and
    embedding row, and their SynLethDB-overlap flag is QARS1's (they carried
    EPRS1's: 446 Gilbert pairs)."""
    raw = pd.DataFrame({
        "gene1_symbol": ["QARS", "QARS", "BRCA1", "EPRS"],
        "gene2_symbol": ["BRCA1", "EPRS", "EPRS", "QARS"],
        "outcome_raw": [1.0, 2.0, 3.0, 4.0]})
    g2i = {"5859": 7, "2058": 3, "672": 5}
    sldb = {frozenset(["5859", "672"])}
    with tempfile.TemporaryDirectory() as tmp:
        m = _mapper(tmp, _TIED)
        with mock.patch.object(ef, "get_mapper", return_value=m):
            df, cov = ef.prepare_dataset(raw, g2i, sldb, -1.0)
    got = {tuple(sorted(k)): (i1, i2, ov, y) for k, i1, i2, ov, y in zip(
        df["pair_key"], df["idx1"], df["idx2"], df["sldb_overlap"],
        df["outcome_raw"])}
    assert got == {
        ("5859", "672"): (7, 5, True, 1.0),
        # QARS-EPRS, listed in both orders: one pair, its two values averaged
        ("2058", "5859"): (3, 7, False, 3.0),
        ("2058", "672"): (3, 5, False, 3.0)}, got
    assert cov["pairs_kept"] == 3 and cov["sldb_overlap_count"] == 1, cov


def test_predict_resolves_a_typed_symbol_as_the_screens_do():
    """predict.py's own lookup of a typed gene name follows the screens' pins
    in either HGNC row order (a typed QARS was scored with EPRS1's row); an
    Entrez ID of the embedding files passes through without any lookup."""
    import predict
    g2i = {"5859": 0, "2058": 1, "672": 2, "84939": 3, "3662": 4}
    p = predict.SLPredictor(None, torch.zeros(5, 2), g2i,
                            {i: g for g, i in g2i.items()})
    names = ["QARS", "qars", "QARS1", "EPRS", "MUM1", "BRCA1", "NOPE"]
    for rows in (_TIED, list(reversed(_TIED))):
        with tempfile.TemporaryDirectory() as tmp:
            m = _mapper(tmp, rows)
            with mock.patch.object(predict, "get_mapper", return_value=m):
                got = [p._resolve_gene(n) for n in names]
        assert got == ["5859", "5859", "5859", "2058", "84939", "672",
                       None], got
    with mock.patch.object(predict, "get_mapper",
                           side_effect=AssertionError("no lookup")):
        assert p._resolve_gene("2058") == "2058"


def test_screen_pins_name_a_gene_that_lists_the_symbol():
    """On the tracked HGNC table (skipped where it is absent): each pin sends
    its symbol to a gene whose own row lists that symbol, so a pin can never
    point at an unrelated gene; and the SLMGAE loader's pins agree. A
    screen file written through R's make.names spells a hyphen as '.'
    (Corn's "BIVM.ERCC5" for HGNC's BIVM-ERCC5 readthrough), so each listed
    name also claims that spelling, and only that one."""
    table = Path(_SIAMESE).parent / "data" / "cache" / "hgnc_complete_set.txt"
    slm = (Path(_SIAMESE).parent / "SLMGAE-in-pytorch"
           / "train_slmgae_shared.py")
    lp = None
    for node in ast.walk(ast.parse(slm.read_text())):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "LP_SYMBOL_OVERRIDES"
                for t in node.targets):
            lp = ast.literal_eval(node.value)
    assert lp, "LP_SYMBOL_OVERRIDES not found in train_slmgae_shared.py"
    for sym in set(lp) & set(ef.SCREEN_SYMBOL_OVERRIDES):
        assert lp[sym] == ef.SCREEN_SYMBOL_OVERRIDES[sym], sym
    assert all(s == s.upper() and e.isdigit()
               for s, e in ef.SCREEN_SYMBOL_OVERRIDES.items())
    if not table.is_file():
        print("    (no tracked HGNC table here: claimants not checked)")
        return
    claim = {s: set() for s in ef.SCREEN_SYMBOL_OVERRIDES}
    with open(table, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            names = {r["symbol"].upper()}
            for field in ("prev_symbol", "alias_symbol"):
                names |= {a.strip().strip('"').upper() for a in
                          (r.get(field) or "").strip('"').split("|")}
            names |= {n.replace("-", ".") for n in names if "-" in n}
            for s in claim:
                if s in names:
                    claim[s].add(r["entrez_id"])
    for s, e in ef.SCREEN_SYMBOL_OVERRIDES.items():
        assert e in claim[s], (s, e, sorted(claim[s]))


def test_source_pins_agree_with_the_screen_and_slmgae_pins():
    """generate_embeddings.SOURCE_SYMBOL_OVERRIDES (read as text: importing
    the generator pulls in every model library) pins a name to the same gene
    as the screens' and SLMGAE's pins do."""
    src = Path(_SIAMESE) / "generate_embeddings.py"
    slm = (Path(_SIAMESE).parent / "SLMGAE-in-pytorch"
           / "train_slmgae_shared.py")

    def _dict(path, name):
        for node in ast.walk(ast.parse(path.read_text())):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign)
                       else [])
            if any(getattr(t, "id", None) == name for t in targets):
                return ast.literal_eval(node.value)
        return None
    pins = _dict(src, "SOURCE_SYMBOL_OVERRIDES")
    assert pins and pins.get("QARS") == "5859", pins
    assert all(s == s.upper() and e.isdigit() for s, e in pins.items())
    for other in (ef.SCREEN_SYMBOL_OVERRIDES,
                  _dict(slm, "LP_SYMBOL_OVERRIDES")):
        for sym in set(pins) & set(other):
            assert pins[sym] == other[sym], sym


def test_dict_collision_keeps_the_current_symbol_key():
    with tempfile.TemporaryDirectory() as tmp:
        m = _mapper(tmp, _ROWS)
        with contextlib.redirect_stdout(io.StringIO()):
            # "ATD" (an alias) comes first, but ATM's own row must win.
            assert m.symbols_to_entrez_dict({"ATD": 1, "ATM": 2}) == {"472": 2}
            assert m.symbols_to_entrez_dict({"ATM": 2, "ATD": 1}) == {"472": 2}
            assert m.symbols_to_entrez_dict({"ATD": 1}) == {"472": 1}
            assert m.symbols_to_entrez_dict({"SARSM": 1, "SARS2": 2,
                                             "SARS": 3}) == {"54938": 2,
                                                             "6301": 3}
            # two aliases, neither the current symbol: the first still wins
            assert m.symbols_to_entrez_dict({"SERRS": 1, "SARSM": 2}) == {
                "54938": 1}


def test_correction_to_unknown_target_keeps_hgnc_resolution():
    """A custom correction whose target HGNC does not know (the tracked
    TCL6 -> TCL6f4) used to turn the approved symbol TCL6 into an unmappable
    name; corrections to known targets, and every correction when HGNC is not
    loaded, must still apply."""
    corrections = {"TCL6": "TCL6F4", "MAR-02": "MARCHF2", "SEP-01": "SEPT1",
                   "OLDX": "NOSUCHGENE", "JUNK-1": "NOSUCHGENE"}
    rows = ["TCL6\t\t\t27004\n", "MARCHF2\t\t\t51257\n",
            "SEPTIN1\tSEPT1\t\t1731\n", "GENEX\t\tOLDX\t111\n"]
    with tempfile.TemporaryDirectory() as tmp:
        m = _mapper(tmp, rows)
        m._custom_corrections = dict(corrections)
        assert m.gene_name_normalize("TCL6") == "TCL6"
        assert m.symbol_to_entrez("TCL6") == "27004"
        # pre-normalizing callers (convert_to_entrez, GO/STRING maps)
        assert m.symbol_to_entrez(m.gene_name_normalize("tcl6")) == "27004"
        assert m.symbol_to_entrez("OLDX") == "111"  # HGNC alias of GENEX
        assert m.symbol_to_entrez("MAR-02") == "51257"  # known target
        assert m.symbol_to_entrez("SEP-01") == "1731"  # target is an alias
        # unknown to HGNC on both sides: the correction still applies
        assert m.gene_name_normalize("JUNK-1") == "NOSUCHGENE"
    with tempfile.TemporaryDirectory() as tmp:
        m = _mapper(tmp, [])  # header only: HGNC knows nothing
        m._custom_corrections = dict(corrections)
        assert m.gene_name_normalize("TCL6") == "TCL6F4"
        assert m.gene_name_normalize("MAR-02") == "MARCHF2"


def test_convert_to_entrez_leaves_entrez_files_untouched():
    """Re-running the migration on Entrez-keyed data must be a no-op (it used
    to map every id through the symbol table and write the files back empty);
    a still-symbol-keyed .pt must still convert."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _mapper(tmp, _ROWS)
        data = Path(tmp) / "data"
        data.mkdir()
        old = (convert_to_entrez.DATA_DIR, convert_to_entrez.BACKUP_DIR)
        convert_to_entrez.DATA_DIR = data
        convert_to_entrez.BACKUP_DIR = data / "_backup_symbols"
        try:
            (data / "genes.txt").write_text("672\n6301\n472\n")
            (data / "pairs.txt").write_text("672\t6301\n472\t672\n")
            (data / "pairs.sources.tsv").write_text(
                "entrez1\tentrez2\tsources\n672\t6301\tx\n")
            torch.save({"embeddings": torch.ones(3, 2),
                        "gene_order": ["672", "6301", "472"]},
                       data / "all_genes_entrez.pt")
            torch.save({"embeddings": torch.arange(6.).reshape(3, 2),
                        "gene_order": ["BRCA1", "SARS", "NOTAGENE"]},
                       data / "all_genes_sym.pt")
            before = {p.name: p.read_bytes() for p in data.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                rewrote = [
                    convert_to_entrez.convert_gene_list(
                        data / "genes.txt", m, False),
                    convert_to_entrez.convert_pair_file(
                        data / "pairs.txt", m, False),
                    convert_to_entrez.convert_pair_file(
                        data / "pairs.sources.tsv", m, False, has_header=True),
                    convert_to_entrez.convert_pt_file(
                        data / "all_genes_entrez.pt", m, False),
                    convert_to_entrez.convert_pt_file(
                        data / "all_genes_sym.pt", m, False),
                ]
            # main() counts these to report what it rewrote.
            assert rewrote == [False, False, False, False, True], rewrote
            for name in ("genes.txt", "pairs.txt", "pairs.sources.tsv",
                         "all_genes_entrez.pt"):
                assert (data / name).read_bytes() == before[name], name
            sym = torch.load(data / "all_genes_sym.pt", weights_only=False)
            assert sym["gene_order"] == ["672", "6301"]
            assert torch.equal(sym["embeddings"],
                               torch.tensor([[0., 1.], [2., 3.]]))
        finally:
            convert_to_entrez.DATA_DIR, convert_to_entrez.BACKUP_DIR = old


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
