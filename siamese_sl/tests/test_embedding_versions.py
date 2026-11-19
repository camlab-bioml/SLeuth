#!/usr/bin/env python3
"""embedding_versions.py: the gate that decides whether generation reruns.

A .pt counts as current only with exactly the current generator_version,
when <data_dir>/gene_universe.txt exists exactly that gene set, and the same
gene order as all_genes_esm2.pt; in catalog mode an absent
run_generate_embeddings.conf OPTIONAL_TYPES file is allowed (present, it must
be current). Covers each status (ok / stale / missing, unreadable, order), the
exit codes the root orchestrator and the generation job branch on (0 current,
1 any stale or missing), the OPTIONAL_TYPES allowance, and that the default
file list is exactly the EMB_CATALOG array of the real .conf file (--all:
EMB_CATALOG then ALL_EMBEDDINGS, so genept only there).

    $PYTHON_PATH -m pytest siamese_sl/tests/test_embedding_versions.py -q
    $PYTHON_PATH siamese_sl/tests/test_embedding_versions.py
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# A pipeline switch still exported by the calling shell (a mistyped
# SELECTION_CV of a submission the orchestrator has just refused, the
# USE_CELL_LINES=0 of an earlier experiment) must not decide these tests:
# run_best_per_category.conf and config.conf honour them wherever they are
# sourced. A test that means one passes it to its subprocess explicitly.
for _v in ("SELECTION_CV",):
    os.environ.pop(_v, None)
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

import embedding_versions as ev

CUR = ev.EMBEDDING_GENERATOR_VERSION
GENES = ["10", "11", "12"]


def _pt(path, version=CUR, genes=GENES, extra=None):
    d = {"embeddings": torch.zeros(len(genes), 2), "gene_order": genes}
    if version is not None:
        d["generator_version"] = version
    d.update(extra or {})
    torch.save(d, path)


def test_check_file_statuses():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        _pt(t / "ok.pt")
        _pt(t / "old.pt", version="0-legacy")
        _pt(t / "none.pt", version=None)
        (t / "junk.pt").write_bytes(b"not a torch file")
        uni = set(GENES)
        assert ev.check_file(t / "ok.pt", uni)[0] == "ok"
        assert ev.check_file(t / "ok.pt")[0] == "ok"
        assert ev.check_file(t / "old.pt", uni)[0] == "stale"
        assert ev.check_file(t / "none.pt", uni)[0] == "stale"
        assert ev.check_file(t / "junk.pt", uni)[0] == "stale"
        assert ev.check_file(t / "absent.pt", uni)[0] == "missing"
        status, detail = ev.check_file(t / "ok.pt", uni | {"13"})
        assert status == "stale" and "gene_universe" in detail


def test_esm2_pooling_and_model_are_pinned():
    """The only check keeping the paper's Pool PaRTI / esm2_t33_650M claim
    honest for a file that carries the current generator_version and the right
    genes — exactly what a documented `--pooling mean` build produces.
    """
    pool, model = ev.PIPELINE_ESM
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        uni = set(GENES)
        (t / "gene_universe.txt").write_text("\n".join(GENES) + "\n")
        pinned = {"pooling": pool, "model_name": model}

        _pt(t / "all_genes_esm2.pt", extra=pinned)
        assert ev.check_file(t / "all_genes_esm2.pt", uni)[0] == "ok"
        # older ESM stamps carry no `pooling` at all: not evidence of a
        # deviant build, so still ok
        _pt(t / "all_genes_esm2.pt")
        assert ev.check_file(t / "all_genes_esm2.pt", uni)[0] == "ok"

        base = ["--data_dir", str(t), "--files", "all_genes_esm2.pt"]
        for deviant in ({"pooling": "mean", "model_name": model},
                        {"pooling": pool, "model_name": "esm2_t36_3B_UR50D"}):
            _pt(t / "all_genes_esm2.pt", extra=deviant)
            status, detail = ev.check_file(t / "all_genes_esm2.pt", uni)
            assert status == "stale" and "pooling/model" in detail, detail
            assert ev.main(base) == 1

        # only esm2 is pinned: another generator's file may carry `pooling`
        _pt(t / "all_genes_go.pt", extra={"pooling": "mean"})
        assert ev.check_file(t / "all_genes_go.pt", uni)[0] == "ok"


def test_main_exit_codes():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        _pt(t / "a.pt")
        _pt(t / "b.pt")
        (t / "gene_universe.txt").write_text("\n".join(GENES) + "\n")
        base = ["--data_dir", str(t), "--files"]
        assert ev.main(base + ["a.pt", "b.pt"]) == 0
        assert ev.main(base + ["a.pt", "c.pt"]) == 1  # missing
        _pt(t / "b.pt", genes=GENES[:2])  # universe changed since
        assert ev.main(base + ["a.pt", "b.pt"]) == 1


def _conf_array(conf, var):
    """Field 2 (the .pt name) of each entry of a bash array literal in conf.

    A plain text parse, independent of the sourcing catalog_files does.
    """
    m = re.search(rf"^{var}=\((.*?)^\)", Path(conf).read_text(), re.S | re.M)
    assert m, f"no {var}=( ... ) in {conf}"
    out = []
    for ln in m.group(1).splitlines():
        ln = ln.split("#", 1)[0].strip().strip('"').strip("'")
        if ln:
            out.append(ln.split(":")[1])
    return list(dict.fromkeys(out))


def test_default_list_is_the_catalog():
    cfg = ev.SLURM_DIR / "config.conf"
    cat = ev.SLURM_DIR / "run_best_per_category.conf"
    names = ev.catalog_files(cfg, cat)
    assert names == _conf_array(cat, "EMB_CATALOG"), names
    assert "all_genes_esm2.pt" in names and "all_genes_go.pt" in names
    # genept (leakage) is ALL_EMBEDDINGS-only: never in the default gate
    assert "all_genes_genept.pt" not in names
    full = ev.catalog_files(cfg, cat, include_all=True)
    assert full == list(dict.fromkeys(
        names + _conf_array(cfg, "ALL_EMBEDDINGS"))), full
    assert "all_genes_genept.pt" in full
    with tempfile.TemporaryDirectory() as tmp:
        # an empty data dir: every catalog file is missing -> regenerate
        assert ev.main(["--data_dir", tmp]) == 1


def test_same_set_other_order_is_stale():
    lex = ["1", "10", "100", "11", "2", "9"]
    num = sorted(lex, key=int)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        _pt(t / "all_genes_esm2.pt", genes=lex)
        _pt(t / "same.pt", genes=list(lex))
        _pt(t / "numeric.pt", genes=num)
        (t / "gene_universe.txt").write_text("\n".join(num) + "\n")
        base = ["--data_dir", tmp, "--files"]
        assert ev.main(base + ["all_genes_esm2.pt", "same.pt"]) == 0
        # esm2 is the reference even when listed after the other file
        assert ev.main(base + ["numeric.pt", "all_genes_esm2.pt"]) == 1
        status, detail = ev.check_file(t / "numeric.pt", set(lex),
                                       ("all_genes_esm2.pt", lex))
        assert status == "stale" and "order" in detail, detail


def test_catalog_mode_allows_absent_optional_types():
    """The generation job does not build esmc (and may miss the other
    OPTIONAL_TYPES); their absence must not fail the gate, or no resubmit
    could ever auto-skip generation. Present, they must still be current."""
    cfg = ev.SLURM_DIR / "config.conf"
    cat = ev.SLURM_DIR / "run_best_per_category.conf"
    gen = ev.SLURM_DIR / "run_generate_embeddings.conf"
    optional = ev.optional_files(cfg, gen, cat)
    assert "all_genes_esmc.pt" in optional and "all_genes_go.pt" not in optional
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for name in ev.catalog_files(cfg, cat):
            if name not in optional:
                _pt(t / name)
        assert ev.main(["--data_dir", tmp]) == 0
        _pt(t / "all_genes_esmc.pt", version="0-legacy")
        assert ev.main(["--data_dir", tmp]) == 1
        (t / "all_genes_esmc.pt").unlink()
        (t / "all_genes_go.pt").unlink()
        assert ev.main(["--data_dir", tmp]) == 1


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
