#!/usr/bin/env python3
"""Deterministic tests for check_shared_inputs.py on a synthetic repo root.

Run from the repo root:  python sl_comparison/slurm/test_check_shared_inputs.py
(or with pytest). Each case runs the checker in a fresh interpreter, because it
imports the root's siamese_sl/gene_name_utils.py.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

CHECKER = Path(__file__).resolve().parent / "check_shared_inputs.py"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# build_synlethkg.MAPPING_FILES (repo-relative); the HGNC table is shared with
# the TransE build.
KG_MAPPING = ("data/cache/Human.SL.detailed.tsv",
              "data/cache/Human.non.SL.detailed.tsv",
              "data/cache/hgnc_complete_set.txt",
              "data/embeddings_cache/Homo_sapiens.gene_info.gz")


def build_root(tmp, label_keyed=True, mapping=True, extra_key=None,
               kg_mapping=True):
    root = Path(tmp)
    hgnc = _write(root / "data/cache/hgnc_complete_set.txt", "hgnc_id\tsymbol\n")
    for rel in KG_MAPPING[:2] + KG_MAPPING[3:]:
        _write(root / rel, f"{rel}\n")
    # Untracked builders: present in a complete checkout, so the rebuild hints
    # below are real commands. Written BEFORE kg_meta, because build_synlethkg
    # records its own sha256 in mapping_sha256 (BUILDER_REL) and a fixture that
    # omitted that key left the builder self-hash untested.
    _write(root / "SLGNN/build_synlethkg.py", "# kg builder\n")
    kg_meta = ({"relation_checks": {"label_keyed_deletion": {"n": 0}}}
               if label_keyed else {"relation_checks": {}})
    if kg_mapping:
        kg_meta["mapping_sha256"] = {rel: _sha(root / rel) for rel in KG_MAPPING}
        kg_meta["mapping_sha256"]["SLGNN/build_synlethkg.py"] = _sha(
            root / "SLGNN/build_synlethkg.py")
    # The payload build_synlethkg writes BEFORE meta.json, and the counts
    # meta.json records for it (load_kg_for_genes reads both files).
    (root / "data/synlethkg").mkdir(parents=True, exist_ok=True)
    np.save(root / "data/synlethkg/kg_triples.npy",
            np.arange(12, dtype=np.int64).reshape(4, 3))
    _write(root / "data/synlethkg/entity_entrez.tsv",
           "entity\tname\tentrez\tmethod\tdegree\tselected\n"
           "0\tA\t10\tsymbol\t3\t1\n1\tB\t20\tsymbol\t1\t1\n")
    kg_meta["n_triples"] = 4
    kg_meta["gene_entities"] = 2
    _write(root / "data/synlethkg/meta.json", json.dumps(kg_meta))
    gcf = _write(root / "SLMGAE-in-pytorch/gene_corrections_config.py",
                 "GENE_CORRECTIONS = {'HIST4H4': 'H4C16'}\n")
    gnu = _write(root / "siamese_sl/gene_name_utils.py",
                 f"GENE_CORRECTIONS_FILE = {str(gcf)!r}\n")
    btf = _write(root / "NSF4SL/build_transe_features.py", "# builder\n")
    rec = {"hgnc_complete_set.txt": _sha(hgnc),
           "gene_corrections_config.py": _sha(gcf),
           "gene_name_utils.py": _sha(gnu),
           "build_transe_features.py": _sha(btf)}
    if extra_key:
        rec[extra_key] = "0" * 64
    meta = {"mapping_sha256": rec} if mapping else {"source": "old"}
    (root / "data").mkdir(parents=True, exist_ok=True)
    torch.save({"entity_mean": torch.zeros(2), "meta": meta},
               root / "data/nsf4sl_transe_features.pt")
    for f in ("Human_GOsim.txt", "Human_GOsim_CC.txt",
              "biogrid_ppi_sparse.txt", "List_Proteins_in_SL.txt"):
        _write(root / "data" / f, "x\n")
    return root


def run(root, *models, python_path=None):
    cmd = [sys.executable, str(CHECKER), "--root", str(root)]
    if models:
        cmd += ["--models", *models]
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_PATH"}
    if python_path is not None:
        env["PYTHON_PATH"] = python_path
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def test_all_current_inputs_pass():
    with tempfile.TemporaryDirectory() as t:
        rc, out = run(build_root(t))
        assert rc == 0, out
        assert out.count("inputs OK") == 3, out


def test_synlethkg_without_label_keyed_deletion_fails():
    with tempfile.TemporaryDirectory() as t:
        rc, out = run(build_root(t, label_keyed=False), "SLGNN")
        assert rc == 1 and "label_keyed_deletion" in out, out
        assert "build_synlethkg.py" in out, out


def test_synlethkg_without_mapping_sha256_fails():
    with tempfile.TemporaryDirectory() as t:
        rc, out = run(build_root(t, kg_mapping=False), "SLGNN")
        assert rc == 1 and "has no mapping_sha256" in out, out
        assert "build_synlethkg.py" in out, out


def test_synlethkg_mapped_from_other_table_fails():
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        # e.g. a pull that updates the tracked NCBI gene_info after the build
        _write(root / "data/embeddings_cache/Homo_sapiens.gene_info.gz", "new\n")
        rc, out = run(root, "SLGNN")
        assert rc == 1, out
        assert "mapped from another 'data/embeddings_cache/Homo_sapiens" in out, out
        # NSF4SL / SLMGAE do not hash that file.
        rc, out = run(root, "NSF4SL", "SLMGAE")
        assert rc == 0, out
        # A recorded mapping file that is not in the checkout fails too
        # (named as unsynced, not blamed on provenance).
        (root / "data/cache/Human.SL.detailed.tsv").unlink()
        rc, out = run(root, "SLGNN")
        assert rc == 1 and "Human.SL.detailed.tsv" in out, out
        assert "not in this checkout" in out, out


def test_synlethkg_built_by_other_builder_code_fails():
    # The builder's own METHODS order / resolve_entrez collision precedence
    # moves KG genes that no INPUT hash sees, so its self-hash must be
    # compared, not merely required.
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        _write(root / "SLGNN/build_synlethkg.py",
               "# METHODS reordered: other Entrez ids\n")
        rc, out = run(root, "SLGNN")
        assert rc == 1 and "another 'SLGNN/build_synlethkg.py'" in out, out
        # NSF4SL / SLMGAE do not depend on it.
        rc, out = run(root, "NSF4SL", "SLMGAE")
        assert rc == 0, out


def test_synlethkg_predating_builder_hash_fails():
    # data/synlethkg is git-ignored and built by hand, so a directory from
    # before the builder hashed itself would otherwise pass forever.
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        p = root / "data/synlethkg/meta.json"
        meta = json.loads(p.read_text())
        del meta["mapping_sha256"]["SLGNN/build_synlethkg.py"]
        p.write_text(json.dumps(meta))
        rc, out = run(root, "SLGNN")
        assert rc == 1 and "lacks 'SLGNN/build_synlethkg.py'" in out, out


def test_synlethkg_without_payload_files_fails():
    # meta.json alone is not a complete build: the trainer also opens
    # kg_triples.npy / entity_entrez.tsv (build_synlethkg.load_kg_for_genes),
    # and without them every SLGNN task dies after its GPU is allocated.
    for name in ("kg_triples.npy", "entity_entrez.tsv"):
        with tempfile.TemporaryDirectory() as t:
            root = build_root(t)
            (root / "data/synlethkg" / name).unlink()
            rc, out = run(root, "SLGNN")
            assert rc == 1 and f"{name} missing" in out, out
            assert "build_synlethkg.py" in out, out


def test_synlethkg_short_payload_fails():
    # An interrupted rebuild or a copy cut short: meta.json is written last, so
    # it is the older/complete one. A short entity_entrez.tsv raises nothing on
    # its own — its lost genes silently become isolated entities.
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        np.save(root / "data/synlethkg/kg_triples.npy",
                np.arange(9, dtype=np.int64).reshape(3, 3))
        rc, out = run(root, "SLGNN")
        assert rc == 1 and "records 4" in out, out
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        p = root / "data/synlethkg/entity_entrez.tsv"
        p.write_text("".join(p.read_text().splitlines(True)[:-1]))
        rc, out = run(root, "SLGNN")
        assert rc == 1 and "entity_entrez.tsv has 1 gene" in out, out


def test_transe_without_mapping_sha256_fails():
    with tempfile.TemporaryDirectory() as t:
        rc, out = run(build_root(t, mapping=False), "NSF4SL")
        assert rc == 1 and "no meta.mapping_sha256" in out, out
        assert "build_transe_features.py --download" in out, out


def test_transe_built_from_other_corrections_fails():
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        # The table changed after the build (e.g. HIST4H4 -> H4C1 -> H4C16).
        _write(root / "SLMGAE-in-pytorch/gene_corrections_config.py",
               "GENE_CORRECTIONS = {'HIST4H4': 'H4C1'}\n")
        rc, out = run(root, "NSF4SL")
        assert rc == 1 and "gene_corrections_config.py" in out, out
        # The refusal names the file it hashed (gene_name_utils' pick).
        assert "SLMGAE-in-pytorch/gene_corrections_config.py" in out, out
        # SLGNN / SLMGAE are unaffected.
        rc, out = run(root, "SLGNN", "SLMGAE")
        assert rc == 0, out


def test_transe_built_by_other_mapper_code_fails():
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        p = root / "siamese_sl/gene_name_utils.py"
        p.write_text(p.read_text() + "# prev_symbol now outranks alias\n")
        rc, out = run(root, "NSF4SL")
        assert rc == 1 and "gene_name_utils.py" in out, out


def test_transe_built_by_other_builder_fails():
    # The builder's own gene-entity / collision rules move KG genes that no
    # INPUT hash sees, so its self-hash must be compared, not merely required.
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        _write(root / "NSF4SL/build_transe_features.py",
               "# builder: prefer the longer gene entity on a collision\n")
        rc, out = run(root, "NSF4SL")
        assert rc == 1 and "another 'build_transe_features.py'" in out, out
        # SLGNN / SLMGAE do not depend on it.
        rc, out = run(root, "SLGNN", "SLMGAE")
        assert rc == 0, out


def test_unsynced_builder_is_named_not_blamed_on_provenance():
    # Both builders are untracked until `git add`ed. A checkout that lacks one
    # must be told so — not sent hunting a provenance mismatch, nor handed a
    # rebuild command whose script is not there.
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        (root / "NSF4SL/build_transe_features.py").unlink()
        rc, out = run(root, "NSF4SL")
        assert rc == 1 and "not in this checkout" in out, out
        assert "was built from another" not in out, out
        assert "build_transe_features.py --download" not in out, out
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        # The builder alone: meta.json records it, so this is the unsynced
        # case the check must name (not "was mapped from another").
        (root / "SLGNN/build_synlethkg.py").unlink()
        rc, out = run(root, "SLGNN")
        assert rc == 1 and "not in this checkout" in out, out
        assert "was mapped from another" not in out, out
        assert "(cd SLGNN &&" not in out, out


def test_transe_predating_mapper_hash_fails():
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        p = root / "data/nsf4sl_transe_features.pt"
        d = torch.load(p, weights_only=False)
        del d["meta"]["mapping_sha256"]["gene_name_utils.py"]
        torch.save(d, p)
        rc, out = run(root, "NSF4SL")
        assert rc == 1 and "lacks 'gene_name_utils.py'" in out, out


def test_transe_predating_builder_hash_fails():
    # A file from the builder before it hashed itself must not pass after a
    # change to its gene-entity / collision rules.
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        p = root / "data/nsf4sl_transe_features.pt"
        d = torch.load(p, weights_only=False)
        del d["meta"]["mapping_sha256"]["build_transe_features.py"]
        torch.save(d, p)
        rc, out = run(root, "NSF4SL")
        assert rc == 1 and "lacks 'build_transe_features.py'" in out, out


def test_unlocatable_recorded_key_fails():
    with tempfile.TemporaryDirectory() as t:
        rc, out = run(build_root(t, extra_key="new_input.tsv"), "NSF4SL")
        assert rc == 1 and "new_input.tsv" in out, out


def test_rebuild_hint_names_the_interpreter():
    # Pasted as printed, the hint must run: a literal $PYTHON_PATH is empty
    # in a login shell (config.conf does not export it).
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t, label_keyed=False, mapping=False)
        rc, out = run(root, "SLGNN", "NSF4SL",
                      python_path="/opt/sl venv/bin/python3")
        assert rc == 1 and "$PYTHON_PATH" not in out, out
        py = shlex.quote("/opt/sl venv/bin/python3")
        # an orchestrated run leaves data/synlethkg read-only: unfreeze first
        assert (f"(chmod -R u+w data/synlethkg 2>/dev/null; cd SLGNN && "
                f"{py} build_synlethkg.py)") in out, out
        assert (f"(cd NSF4SL && {py} build_transe_features.py --download)"
                in out), out
        # Unset: the interpreter running the check (the orchestrator's).
        rc, out = run(root, "SLGNN")
        assert rc == 1, out
        assert (f"cd SLGNN && {shlex.quote(sys.executable)} "
                f"build_synlethkg.py)") in out, out


def test_missing_transe_and_slmgae_view_fail():
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        (root / "data/nsf4sl_transe_features.pt").unlink()
        (root / "data/Human_GOsim_CC.txt").unlink()
        rc, out = run(root)
        assert rc == 1, out
        assert "nsf4sl_transe_features.pt missing" in out, out
        assert "data/Human_GOsim_CC.txt missing" in out, out
        assert "inputs OK: SLGNN" in out, out


def test_an_empty_slmgae_input_is_named_at_the_login_check():
    # A 0-byte copy (an interrupted upload) exists, so the existence check
    # passed and three GPU tasks failed later (or, for the PPI edge list,
    # trained without the view).
    with tempfile.TemporaryDirectory() as t:
        root = build_root(t)
        rc, out = run(root, "SLMGAE")
        assert rc == 0 and "inputs OK: SLMGAE" in out, out
        (root / "data/biogrid_ppi_sparse.txt").write_text("")
        rc, out = run(root, "SLMGAE")
        assert rc == 1 and "data/biogrid_ppi_sparse.txt is empty" in out, out
        assert "biogrid_ppi_sparse.txt missing" not in out, out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
