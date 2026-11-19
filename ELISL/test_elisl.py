#!/usr/bin/env python3
"""Unit tests for the ELISL (context-free) integration, on tiny synthetic data.

Pins: the vendored model differs from the authors' only on its `# HARNESS:`
lines; the vectorised pair features equal the authors' create_seq_features /
create_ppi_features; the late integration equals evaluate_all_models'
groupby(combine_preds); a failed forest fit raises instead of scoring zeros;
the UniProt / Entrez input steps; the trainer, end to end on a 24-gene fold
set (matrix shape, symmetry, finiteness, the exported matrix reproducing the
model's own validation scores, fold stamp + sources in model_params.json, the
shared evaluator's acceptance), with and without the Bayesian search; the
gene-universe and validation-split refusals; check_shared_inputs --models
ELISL; and the launchers' GPU pin. LightGBM runs on the CPU (the authors'
model has no GPU path), so nothing here forces or skips CUDA.

    python -m pytest ELISL/test_elisl.py -q      (from the repo root)
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO / "sl_comparison"))
from elisl_orig import ELRRF as E  # noqa: E402
from elisl_orig import features as F  # noqa: E402
import build_elisl_inputs as B  # noqa: E402
import train_elisl as T  # noqa: E402


# ---------------------------------------------------------------------------
# vendored code
# ---------------------------------------------------------------------------

def test_vendored_model_changes_only_marked_lines():
    text = (HERE / "elisl_orig" / "ELRRF.py").read_text()
    src = text.split('"""', 2)[2]                 # the code, past the docstring
    assert "from src" not in src and "config." not in src
    # every LightGBM fit without fit(verbose=...), which LightGBM 4 refuses
    assert "verbose=False)" not in src and "verbose=False, init" not in src
    assert src.count("# HARNESS:") == 5          # the import block + 4 lines
    assert ".append(model_pred_df" not in src
    # the authors' defaults, unchanged
    m = E.ELRRF()
    assert (m.boosting_type, m.num_leaves, m.n_estimators, m.subsample,
            m.colsample_bytree, m.min_child_weight, m.min_child_samples,
            m.reg_lambda, m.random_state) == ("rf", 165, 400, 0.632, 0.8, 5,
                                              10, 5.0, 124)
    assert (HERE / "LICENSE").read_text().lstrip().startswith(
        "GNU GENERAL PUBLIC LICENSE")


def test_failed_fit_raises_instead_of_scoring_zero():
    m = E.ELRRF(n_estimators=5)
    x = np.random.RandomState(0).rand(20, 3)
    with pytest.raises(Exception):
        # one class only: LightGBM's classifier cannot fit it
        m.fit_predict_report(x, np.zeros(20, dtype=int), x)


def _toy_embeddings(rng, n_prot=12, dim=6):
    emb = {f"P{i:05d}": rng.randn(dim).astype(np.float32)
           for i in range(n_prot)}
    return emb


def test_pair_features_equal_the_authors_loop():
    rng = np.random.RandomState(1)
    emb = _toy_embeddings(rng)
    name2uid = {f"G{i}": f"P{i:05d}" for i in range(10)}   # G10, G11 unmapped
    mappings = [name2uid, {}, {}, {}]
    genes = [f"G{i}" for i in range(12)]
    pairs = [(genes[a], genes[b]) for a in range(12) for b in range(12)
             if a < b]
    df = pd.DataFrame(pairs, columns=["gene1", "gene2"])
    want = F.create_seq_features(df, emb, mappings, emb_name="seq")
    want_ppi = F.create_ppi_features(df, emb, mappings, emb_name="ppi_ec")
    vecs, miss, nf = F.gene_vectors(genes, mappings, emb)
    assert nf == ["G10", "G11"] and miss.tolist() == [False] * 10 + [True] * 2
    idx = {g: i for i, g in enumerate(genes)}
    a = np.array([idx[x] for x, _ in pairs])
    b = np.array([idx[y] for _, y in pairs])
    got = F.pair_features(vecs, miss, a, b)
    cols = [f"seq{i}" for i in range(6)]
    assert np.array_equal(got, want[cols].to_numpy())
    assert np.array_equal(got, want_ppi[[f"ppi_ec{i}" for i in range(6)]]
                          .to_numpy())
    assert (got[miss[a] | miss[b]] == -1).all()


def test_fuse_equals_evaluate_all_models_grouping():
    rng = np.random.RandomState(2)
    names = [f"a|b{i}" for i in range(7)]
    preds = {d: {"probabilities": rng.rand(7), "tr_auc": w, "test_names":
                 names} for d, w in zip(T.DATASETS, (0.6, 0.3, 0.8))}
    rows = pd.concat([pd.DataFrame({"pair_name": names, "probability":
                                    p["probabilities"], "tr_auc": p["tr_auc"]})
                      for p in preds.values()])
    want = rows.groupby("pair_name").apply(E.combine_preds)["probability"]
    got, w = T.fuse(preds)
    assert np.allclose(got, want.loc[names].to_numpy(), atol=1e-12)
    assert abs(sum(w.values()) - 1) < 1e-12


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def test_uniprot_fasta_parsing_and_accession_headers(tmp_path):
    fa = tmp_path / "u.fasta"
    fa.write_text(">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo "
                  "sapiens OX=9606 GN=TP53 PE=1 SV=4\nMEEPQ\nSDPSV\n"
                  ">sp|Q9XYZ1|ABC_HUMAN Uncharacterized OS=Homo sapiens "
                  "OX=9606 PE=4 SV=1\nMK\n")
    recs = B.parse_uniprot_fasta(fa)
    assert recs == [("P04637", "P53", "TP53", "MEEPQSDPSV"),
                    ("Q9XYZ1", "ABC", None, "MK")]


def test_entrez_symbols_from_the_repo_table(tmp_path):
    t = tmp_path / "m.tsv"
    t.write_text("entrez_id\tsymbol\n672\tBRCA1\n675\tBRCA2\n")
    assert B.entrez_symbols(["675", "999", "672"], t) == ["BRCA2", None,
                                                         "BRCA1"]


# ---------------------------------------------------------------------------
# trainer, end to end on synthetic folds
# ---------------------------------------------------------------------------

N = 24


def _write_folds(root, n_folds=2, val=True, seed=0):
    rng = np.random.RandomState(seed)
    genes = [str(1000 + i) for i in range(N)]
    (root / "genes.txt").write_text("\n".join(genes) + "\n")
    allp = [(a, b) for a in range(N) for b in range(a + 1, N)]
    rng.shuffle(allp)
    allp = np.array(allp, dtype=np.int64)
    # labels from a hidden gene score so the forests have something to learn
    s = rng.rand(N)
    lab = (s[allp[:, 0]] + s[allp[:, 1]] > 1.15).astype(int)
    np.save(root / "all_pos.npy", allp[lab == 1])
    (root / "meta.json").write_text(json.dumps({"num_nodes": N}))
    for k in range(n_folds):
        d = root / "cv1" / f"fold_{k}"
        d.mkdir(parents=True)
        perm = rng.permutation(len(allp))
        te, va, tr = perm[:40], perm[40:80], perm[80:]
        for name, sel in (("train", tr), ("test", te), ("val", va)):
            if name == "val" and not val:
                continue
            np.save(d / f"{name}_pos.npy", allp[sel][lab[sel] == 1])
            np.save(d / f"{name}_neg.npy", allp[sel][lab[sel] == 0])
    return genes, s


def _write_features(path, genes, s, seq_dim=8, ppi_dim=4):
    rng = np.random.RandomState(3)
    seq = (rng.randn(N, seq_dim) + 3 * s[:, None]).astype(np.float32)
    ppi = (rng.randn(N, ppi_dim) - 2 * s[:, None]).astype(np.float32)
    seq_m = np.zeros(N, bool)
    seq_m[[2, 5]] = True
    seq[seq_m] = -1
    ppi_m = np.zeros(N, bool)
    ppi_m[[5, 7]] = True
    ppi[ppi_m] = -1
    names = np.asarray([f"G{i}" for i in range(N - 1)] + ["G0"])  # one dup
    np.savez(path, genes=np.asarray(genes), seq=seq, seq_missing=seq_m,
             ppi=ppi, ppi_missing=ppi_m, names=names)


def _run(tmp, *extra):
    folds = tmp / "folds"
    folds.mkdir()
    genes, s = _write_folds(folds)
    feats = tmp / "gene_features.npz"
    _write_features(feats, genes, s)
    out = tmp / "out"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.main(["--folds_dir", str(folds), "--cv_type", "cv1", "--features",
                str(feats), "--output_dir", str(out), "--chunk", "50",
                *extra])
    return folds, out, buf.getvalue()


def _small_trees(monkeypatch):
    # keep the forests tiny: the defaults are the authors' (400 trees)
    orig = T.make_model

    def make(seed, gs):
        m = orig(seed, gs)
        m.n_estimators, m.num_leaves, m.min_child_samples = 20, 8, 2
        return m
    monkeypatch.setattr(T, "make_model", make)


def test_trainer_end_to_end_without_search(tmp_path, monkeypatch):
    _small_trees(monkeypatch)
    folds, out, log = _run(tmp_path, "--no_grid_search")
    for k in range(2):
        mat = np.load(out / f"fold_{k}_cv1_predictions.npy")
        assert mat.shape == (N, N) and mat.dtype == np.float32
        assert np.isfinite(mat).all() and np.array_equal(mat, mat.T)
        assert (np.diag(mat) == 0).all()
        off = mat[~np.eye(N, dtype=bool)]
        assert (off >= 0).all() and (off <= 1).all()
    res = json.loads((out / "results.json").read_text())["summary"]
    assert res["elisl_sources"] == "Sequence_1024|PPI_ec"
    assert res["context_free"] is True
    f0 = res["folds"][0]
    assert set(f0["fusion_weights"]) == set(T.DATASETS)
    assert f0["best_params"] == {} and f0["n_train_undersampled"] % 2 == 0
    mp = json.loads((out / "model_params.json").read_text())
    assert mp["elisl_sources"] == "Sequence_1024|PPI_ec"
    assert mp["selection_regime"] == "val" and mp["fold_stamp"]
    assert mp["trainable_params"] == sum(mp["params_per_dataset"].values()) + 3
    # the shared evaluator grades it: fold stamp + matrix width accepted
    import evaluate_model as em
    from model_params import verify_fold_stamp
    verify_fold_stamp(out, folds, "cv1")
    mat = np.load(out / "fold_0_cv1_predictions.npy")
    m = em.score_fold(mat, folds / "cv1" / "fold_0", "ELISL/cv1/fold_0")
    assert abs(m["auprg"] - f0["auprg"]) < 1e-9


def test_trainer_with_bayesian_search(tmp_path, monkeypatch):
    _small_trees(monkeypatch)
    pytest.importorskip("skopt")
    _, out, log = _run(tmp_path, "--n_calls", "10")
    res = json.loads((out / "results.json").read_text())["summary"]
    grid = E.ELRRF().get_tree_search_grid()
    for f in res["folds"]:
        assert set(f["best_params"]) == set(grid)
        assert all(f["best_params"][k] in grid[k] for k in grid)
        assert len(f["search_trace"]) == 10


def test_trainer_refuses_other_universe_and_no_val(tmp_path):
    folds = tmp_path / "folds"
    folds.mkdir()
    genes, s = _write_folds(folds, val=False)
    feats = tmp_path / "f.npz"
    _write_features(feats, genes, s)
    with pytest.raises(SystemExit, match="no usable validation split"):
        T.main(["--folds_dir", str(folds), "--features", str(feats),
                "--output_dir", str(tmp_path / "o")])
    folds2 = tmp_path / "folds2"
    folds2.mkdir()
    genes2, _ = _write_folds(folds2)
    _write_features(feats, list(reversed(genes2)), s)
    with pytest.raises(SystemExit, match="another gene universe"):
        T.main(["--folds_dir", str(folds2), "--features", str(feats),
                "--output_dir", str(tmp_path / "o")])


# ---------------------------------------------------------------------------
# input gate and launchers
# ---------------------------------------------------------------------------

def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_check_shared_inputs_elisl(tmp_path):
    sys.path.insert(0, str(REPO / "sl_comparison" / "slurm"))
    import check_shared_inputs as C
    root = tmp_path
    assert C.check_elisl(root) == ["data/elisl/meta.json missing"]
    (root / "data" / "elisl").mkdir(parents=True)
    feats = root / C.ELISL_FEATURES
    feats.write_bytes(b"npz")
    for c in C.ELISL_CODE_KEYS:
        (root / c).parent.mkdir(parents=True, exist_ok=True)
        (root / c).write_text(f"# {c}\n")
    src = root / "data" / "elisl" / "uniprot_1024_embeddings.npz"
    src.write_bytes(b"emb")
    meta = {"sources": ["Sequence_1024", "PPI_ec"], "context_free": True,
            "gene_features_sha256": _sha(feats),
            "code_sha256": {c: _sha(root / c) for c in C.ELISL_CODE_KEYS},
            "input_sha256": {"data/elisl/uniprot_1024_embeddings.npz":
                             _sha(src)}}
    (root / C.ELISL_META).write_text(json.dumps(meta))
    assert C.check_elisl(root) == []
    src.write_bytes(b"other")
    errs = C.check_elisl(root)
    assert len(errs) == 1 and "built from another" in errs[0]
    src.write_bytes(b"emb")
    meta["sources"] = ["Sequence_1024", "PPI_ec", "Tissue"]
    (root / C.ELISL_META).write_text(json.dumps(meta))
    assert "not the context-free" in C.check_elisl(root)[0]
    # opt-in: the default set (the root orchestrator's) is unchanged
    assert C.MODELS == ("SLGNN", "NSF4SL", "SLMGAE") and "ELISL" in C.OPT_IN


@pytest.mark.parametrize("script", ["run_elisl.sh", "build_elisl_inputs.sh"])
def test_launchers_are_gpu3_jobs(script):
    txt = (HERE / "slurm" / script).read_text()
    assert "#SBATCH --nodelist=gpu3" in txt and "#SBATCH --gres=gpu:1" in txt
    import re
    # nothing hides the GPU or forces a CPU path
    assert not re.search(r"CUDA_VISIBLE_DEVICES\s*=", txt)
    assert not re.search(r"--cpu(?![-\w])", txt)
    assert "check_shared_inputs.py --models ELISL" in txt
    r = subprocess.run(["bash", "-n", str(HERE / "slurm" / script)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_run_elisl_moves_the_row_aside_before_any_check():
    txt = (HERE / "slurm" / "run_elisl.sh").read_text()
    i = txt.index('mv -f "$OUT_JSON" "$STALE_TO"')
    assert i < txt.index(
        "$PYTHON_PATH sl_comparison/slurm/check_shared_inputs.py --models ELISL")
    assert i < txt.index("$PYTHON_PATH ELISL/train_elisl.py --cv_type")
    assert "--model_name \"$ROW\"" in txt and 'ROW="ELISL"' in txt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
