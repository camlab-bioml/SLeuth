#!/usr/bin/env python3
"""All-pair outputs: export_pair_predictions.py and eval_finetuning's
per-head / all-pair tables (synthetic data, no HGNC table; on CUDA when
the node has one).

  export_pair_predictions.py
      one matrix per cell-line head, equal to the fold mean of
      sigmoid(model.forward) on every gene pair, plus ANY (max over heads per
      fold); a pooled and a single-output model write ANY only; the gene
      universe follows the list it is given; --per_fold / --also_logits /
      --dtype / a small --ram_gb change the files, not the values; a refused
      run clears nothing; a matrix that is not the forward stops the export;
      a fine-tuned set that is missing, partial, unreadable, from other
      checkpoints or fails while scored is reported, not exported, and costs
      nothing else; the reader opens each name's own file and refuses an
      export whose models have changed.
  eval_finetuning.py
      {dataset}_sl_probabilities.csv holds sigmoid(logit) of every head for
      every screen pair, its scored-head column being the logit of
      {dataset}_predictions.csv; {dataset}_finetuned_allpairs_{mode}.csv holds
      every pair with its split, its test rows being
      {dataset}_finetuned_predictions_{mode}.csv; the saved fine-tuned models
      reproduce those predictions through the screen's own head on all gene
      pairs; no failure of an optional output costs a screen its tables or
      leaves a truncated file behind; the opt-in Scratch mode writes the same
      outputs from a freshly initialised model and leaves LP_last / Full's
      byte-identical.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_pair_predictions.py -q
    $PYTHON_PATH siamese_sl/tests/test_pair_predictions.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

import cell_line_vocab  # noqa: E402
import eval_finetuning as ef  # noqa: E402
import export_pair_predictions as xp  # noqa: E402
import gene_name_utils  # noqa: E402
from data_loader import (apply_multimodal_transform,  # noqa: E402
                         fit_multimodal_transform, load_raw_multimodal)
from siamese_esm import SiameseSL, SiameseSLMultiCell  # noqa: E402

N_GENES = 30
GENES = [str(101 + i) for i in range(N_GENES)]
DIMS = (5, 4)
N_FOLDS = 2
HEADS = list(cell_line_vocab.HEADS)
SCREEN_HEAD = "HELA"  # not head 0, so a wrong head index cannot pass
# The dense matrix (a GEMM) and the pair-by-pair forward round differently in
# float32; a wrong head, fold or gene order is O(0.1) in probability.
PTOL = 2e-5
# Both scripts run on CUDA wherever there is one (the cluster's GPU nodes), so
# device placement is exercised there; the reference forwards stay on CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _FakeMapper:
    """Symbols are 'G<entrez>' ('OLD<entrez>' is a previous symbol of the
    same gene, which entrez_to_symbol never returns); nothing is downloaded."""

    def symbol_to_entrez(self, s):
        s = str(s).upper()
        for prefix in ("G", "OLD"):
            if s.startswith(prefix) and s[len(prefix):].isdigit():
                return s[len(prefix):]
        return None

    def entrez_to_symbol(self, e):
        return f"G{e}"


def _mapper_patches():
    fake = _FakeMapper()
    return (mock.patch.object(ef, "get_mapper", return_value=fake),
            mock.patch.object(gene_name_utils, "get_mapper",
                              return_value=fake))


def _quiet(fn, *a, **kw):
    """Run fn with stdout/stderr captured; returns (result, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def _make_model(kind: str, seed: int):
    """A model of each checkpoint family, every head made different."""
    torch.manual_seed(seed)
    d = sum(DIMS)
    if kind == "single":
        m = SiameseSL(input_dim=d, encoder_dims=[6, 6, 4], dropout=0.1,
                      pd_epsilon=0.01)
        with torch.no_grad():
            m.scoring_bias.fill_(-0.3)
        return m.eval()
    shape = (dict(num_heads=1, cell_line_dim=0) if kind == "pooled" else
             dict(num_heads=len(HEADS), cell_line_dim=3))
    m = SiameseSLMultiCell(input_dim=d, encoder_dims=[6, 6, 4], dropout=0.1,
                           pd_epsilon=0.01, use_layernorm=False,
                           raw_temperature=True,
                           per_head_last_layer_bias=True, kernel="bilinear",
                           gene_term=True, pair_features=True, **shape)
    with torch.no_grad():
        m.scoring_bias.normal_(0, 1.0)
        m.head_proj_bias.normal_(0, 0.5)
        m.kernel_A.add_(0.3 * torch.randn_like(m.kernel_A))
        m.gene_w.normal_(0, 0.3)
        m.pair_fc2.weight.normal_(0, 0.5)
        if m.cell_emb is not None:
            m.cell_emb.weight.normal_(0, 1.0)
    return m.eval()


def _model_dir(root: Path, kind: str = "per_head", seed: int = 0):
    """A finished combo dir as train.py leaves it, on random embeddings.
    Returns (model_dir, [embedding paths])."""
    g = torch.Generator().manual_seed(100 + seed)
    paths = []
    for name, dim in zip(("all_genes_a.pt", "all_genes_b.pt"), DIMS):
        p = root / name
        torch.save({"embeddings": torch.randn(N_GENES, dim, generator=g),
                    "gene_order": list(GENES)}, p)
        paths.append(str(p))
    raw, _, idx_to_gene = load_raw_multimodal(paths)
    md = root / "combo_cv3"
    (md / "checkpoints").mkdir(parents=True)
    (md / "config.json").write_text(json.dumps({
        "num_folds": N_FOLDS, "cv_type": "cv3", "kernel": "bilinear",
        "cell_lines": "pooled" if kind == "pooled" else "per_head"}))
    shas = [ef._file_sha256(p) for p in paths]
    for k in range(N_FOLDS):
        # each fold fits its transform on its own genes, as train.py does
        fit = torch.ones(N_GENES, dtype=torch.bool)
        fit[k * 5:k * 5 + 5] = False
        transform = fit_multimodal_transform(raw, fit, pca_dims=None,
                                             post_pca=None, pca_method="plain")
        torch.save({"model_state_dict":
                    _make_model(kind, 10 * seed + k).state_dict(),
                    "config": {"pd_epsilon": 0.01, "dropout": 0.1},
                    "input_dim": sum(DIMS),
                    "preprocessing_transform": transform,
                    "gene_order": [idx_to_gene[i]
                                   for i in range(len(idx_to_gene))],
                    "embeddings_paths": [f"../data/{Path(p).name}"
                                         for p in paths],
                    "embeddings_sha256": shas},
                   md / "checkpoints" / f"fold_{k}_best.pt")
    return md, paths


def _fold_logits(md: Path, paths, ckpts=None):
    """(n_folds, n, n, H) logits from the ordinary forward, pair by pair."""
    raw, _, _ = load_raw_multimodal(paths)
    ii = torch.arange(N_GENES).repeat_interleave(N_GENES)
    jj = torch.arange(N_GENES).repeat(N_GENES)
    out = []
    for k in range(N_FOLDS):
        src = md / "checkpoints" / f"fold_{k}_best.pt"
        ck = torch.load(src, map_location="cpu", weights_only=False)
        E = apply_multimodal_transform(raw, ck["preprocessing_transform"])
        model = xp.load_scorer(ckpts[k] if ckpts else src, "cpu")
        with torch.no_grad():
            out.append(model(E[ii], E[jj]).reshape(N_GENES, N_GENES,
                                                   -1).double().numpy())
    return np.stack(out)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _export(md, paths, *extra):
    argv = ["--model_dir", str(md), "--embeddings_paths", *paths,
            "--device", DEVICE, *extra]
    (rc, text) = _quiet(xp.main, argv)
    return rc, text


def _off_diag(a):
    return a[~np.eye(a.shape[0], dtype=bool)]


# =============================================================================
# export_pair_predictions.py
# =============================================================================


def test_export_writes_one_probability_matrix_per_cell_line():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
        md, paths = _model_dir(Path(tmp))
        rc, text = _export(md, paths)
        assert rc == 0, text
        out = md / "pair_predictions"
        probs = _sigmoid(_fold_logits(md, paths))       # (K, n, n, H)
        mf = json.loads((out / "manifest.json").read_text())
        assert mf["heads"] == HEADS and mf["n_genes"] == N_GENES
        assert list(mf["sl_probability"]["files"]) == HEADS + ["ANY"]
        assert mf["diagonal"] == "nan" and mf["finetuned"] == []
        assert (out / "genes.txt").read_text().split() == GENES
        tsv = (out / "genes.tsv").read_text().splitlines()
        assert tsv[0] == "index\tentrez_id\tsymbol" and tsv[1] == "0\t101\tG101"
        for h, lab in enumerate(HEADS):
            got = np.load(out / f"sl_prob_{lab}.npy")
            assert got.shape == (N_GENES, N_GENES) and got.dtype == np.float32
            assert np.isnan(np.diag(got)).all()
            want = probs[:, :, :, h].mean(axis=0)
            assert np.allclose(_off_diag(got), _off_diag(want), atol=PTOL), lab
            # Symmetric to float32 rounding, not bit for bit: the logit
            # matrix is exactly symmetric, but torch's CPU sigmoid rounds
            # the last numel % 8 elements of a buffer up to 2 ulp (1.2e-7)
            # apart from the vectorised ones. An asymmetric scorer is O(0.01).
            assert np.allclose(_off_diag(got), _off_diag(got.T), rtol=0,
                               atol=1e-6), lab
        # heads must really differ, or a head mix-up would go unseen
        assert not np.allclose(probs[..., 0], probs[..., 3], atol=1e-3)
        any_want = probs.max(axis=3).mean(axis=0)
        got = np.load(out / "sl_prob_ANY.npy")
        assert np.allclose(_off_diag(got), _off_diag(any_want), atol=PTOL)
        st = mf["sl_probability"]["stats"]["ANY"]
        assert abs(st["max"] - np.nanmax(got)) < 1e-7 and 0 <= st["min"]
        assert sorted(p.name for p in out.glob("*.npy")) == sorted(
            f"sl_prob_{lab}.npy" for lab in HEADS + ["ANY"])


def test_export_one_head_models_write_any_only():
    for kind in ("pooled", "single"):
        with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
            md, paths = _model_dir(Path(tmp), kind)
            rc, text = _export(md, paths)
            assert rc == 0, (kind, text)
            out = md / "pair_predictions"
            assert [p.name for p in out.glob("*.npy")] == ["sl_prob_ANY.npy"]
            mf = json.loads((out / "manifest.json").read_text())
            assert mf["heads"] == ["ANY"], kind
            assert mf["model_class"] == ("SiameseSL" if kind == "single" else
                                         "SiameseSLMultiCell")
            want = _sigmoid(_fold_logits(md, paths))[..., 0].mean(axis=0)
            got = np.load(out / "sl_prob_ANY.npy")
            assert np.allclose(_off_diag(got), _off_diag(want), atol=PTOL), kind


def test_export_follows_the_gene_list_it_is_given():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        rc, text = _export(md, paths, "--out_dir", str(root / "all"))
        assert rc == 0, text
        full = np.load(root / "all" / "sl_prob_K562.npy")
        # another order, by Entrez ID and by symbol
        order = [7, 2, 19, 11, 0]
        (root / "mine.txt").write_text(
            "\n".join(["108", "G103", "120", "g112", "101"]) + "\n")
        rc, text = _export(md, paths, "--genes_file", str(root / "mine.txt"),
                           "--out_dir", str(root / "mine"))
        assert rc == 0, text
        assert (root / "mine" / "genes.txt").read_text().split() == [
            GENES[i] for i in order]
        sub = np.load(root / "mine" / "sl_prob_K562.npy")
        assert np.allclose(_off_diag(sub),
                           _off_diag(full[np.ix_(order, order)]), atol=PTOL)
        # --universe shared reads --shared_genes in its own order
        (root / "shared.txt").write_text("\n".join(GENES[::-1][:6]) + "\n")
        rc, text = _export(md, paths, "--universe", "shared",
                           "--shared_genes", str(root / "shared.txt"),
                           "--out_dir", str(root / "shared"))
        assert rc == 0, text
        mf = json.loads((root / "shared" / "manifest.json").read_text())
        assert mf["universe"] == "shared" and mf["n_genes"] == 6
        want = GENES[::-1][:6]
        assert (root / "shared" / "genes.txt").read_text().split() == want
        rev = list(range(N_GENES))[::-1][:6]
        assert np.allclose(
            _off_diag(np.load(root / "shared" / "sl_prob_K562.npy")),
            _off_diag(full[np.ix_(rev, rev)]), atol=PTOL)
        # no HGNC table on the node: the Entrez IDs still stand
        with mock.patch.object(gene_name_utils, "get_mapper",
                               side_effect=RuntimeError("no HGNC table")):
            rc, text = _export(md, paths, "--out_dir", str(root / "nosym"))
        assert rc == 0 and "gene symbols unavailable" in text, text
        assert (root / "nosym" / "genes.tsv").read_text().splitlines()[1] == \
            "0\t101\t"
        # a listed gene without an embedding row is refused, nothing written
        (root / "bad.txt").write_text("101\n999999\n")
        rc, text = _export(md, paths, "--genes_file", str(root / "bad.txt"),
                           "--out_dir", str(root / "bad"))
        assert rc == 1 and "no embedding row" in text, text
        assert not (root / "bad").exists()


def test_export_options_change_the_files_not_the_values():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        rc, text = _export(md, paths, "--out_dir", str(root / "plain"))
        assert rc == 0, text
        rc, text = _export(md, paths, "--per_fold", "--also_logits",
                           "--ram_gb", "1e-9", "--out_dir", str(root / "all"))
        assert rc == 0 and "passes over the folds" in text, text
        mf = json.loads((root / "all" / "manifest.json").read_text())
        logits = _fold_logits(md, paths)
        for h, lab in enumerate(HEADS + ["ANY"]):
            a = np.load(root / "plain" / f"sl_prob_{lab}.npy")
            b = np.load(root / "all" / f"sl_prob_{lab}.npy")
            assert np.allclose(a, b, rtol=0, atol=PTOL, equal_nan=True), lab
            names = mf["sl_probability"]["per_fold_files"][lab]
            assert names == [f"fold_{k}_sl_prob_{lab}.npy"
                             for k in range(N_FOLDS)], names
            folds = [np.load(root / "all" / f) for f in names]
            assert np.allclose(_off_diag(np.mean(folds, axis=0)),
                               _off_diag(b), atol=1e-6), lab
            want = (logits.max(axis=3) if lab == "ANY" else logits[..., h])
            lnames = mf["sl_probability"]["per_fold_logit_files"][lab]
            assert lnames == [f"fold_{k}_sl_logit_{lab}.npy"
                              for k in range(N_FOLDS)], lnames
            for k in range(N_FOLDS):
                # fold_<k>_* is fold k's own model, NaN diagonal like the mean
                fl = np.load(root / "all" / lnames[k])
                assert np.isnan(np.diag(folds[k])).all(), (lab, k)
                assert np.isnan(np.diag(fl)).all(), (lab, k)
                assert np.allclose(_off_diag(folds[k]),
                                   _off_diag(_sigmoid(want[k])),
                                   atol=PTOL), (lab, k)
                assert np.allclose(_off_diag(fl), _off_diag(want[k]),
                                   rtol=1e-4, atol=1e-3), (lab, k)
            got = np.load(root / "all" /
                          mf["sl_probability"]["logit_files"][lab])
            assert np.allclose(_off_diag(got), _off_diag(want.mean(axis=0)),
                               rtol=1e-4, atol=1e-3), lab
            # the logits are symmetric bit for bit, on every device
            assert np.array_equal(_off_diag(got), _off_diag(got.T)), lab
        # the manifest names every matrix on disk
        sp = mf["sl_probability"]
        named = set(sp["files"].values()) | set(sp["logit_files"].values())
        for key in ("per_fold_files", "per_fold_logit_files"):
            named |= {f for fs in sp[key].values() for f in fs}
        assert sorted(q.name for q in (root / "all").glob("*.npy")) == sorted(
            named)
        # ... and a default export carries none of the optional keys
        plain = json.loads((root / "plain" / "manifest.json").read_text())
        assert sorted(plain["sl_probability"]) == ["files", "stats", "value"]
        rc, text = _export(md, paths, "--dtype", "float16",
                           "--out_dir", str(root / "half"))
        assert rc == 0, text
        half = np.load(root / "half" / "sl_prob_K562.npy")
        assert half.dtype == np.float16
        assert np.allclose(_off_diag(half.astype(np.float32)), _off_diag(
            np.load(root / "plain" / "sl_prob_K562.npy")), atol=1e-3)


def test_reader_resolves_older_names_from_the_repositorys_own_table():
    """gene_row's older-name lookup reads the repository's HGNC table by its
    own path: the same answer from any working directory, nothing downloaded
    or created, and the process-wide gene_name_utils mapper left alone."""
    # the file the pipeline reads: get_mapper()'s ../data/cache from siamese_sl/
    assert xp.HGNC_TABLE == (Path(_SIAMESE).resolve().parent / "data" / "cache"
                             / "hgnc_complete_set.txt"), xp.HGNC_TABLE
    import urllib.request

    downloads = []

    def no_download(*a, **kw):
        # recorded as well: GeneNameMapper._download and gene_row both catch
        # what this raises, so the raise alone fails nothing at lookup time
        downloads.append(a)
        raise AssertionError("reading an export back must not download")

    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
        root = Path(tmp).resolve()
        md, paths = _model_dir(root)
        rc, text = _export(md, paths)
        assert rc == 0, text
        pp = xp.open_pair_predictions(md / "pair_predictions")
        # a table in the real format: an older name of gene 103, an alias of
        # gene 120, an all-digit alias (HGNC has three), a gene not exported
        table = root / "repo" / "data" / "cache" / "hgnc_complete_set.txt"
        table.parent.mkdir(parents=True)
        table.write_text("symbol\tprev_symbol\talias_symbol\tentrez_id\n"
                         "G103\tOLD103\t52\t103\n"
                         "G120\tQARS\tTWENTY|XX20\t120\n"
                         "G999\tOLD999\t\t999\n")
        gone = root / "nowhere" / "hgnc_complete_set.txt"
        elsewhere = root / "elsewhere" / "deeper"
        elsewhere.mkdir(parents=True)
        process_mapper = gene_name_utils._default_mapper
        cwd = os.getcwd()
        os.chdir(elsewhere)
        try:
            with mock.patch.object(xp, "HGNC_TABLE", table), \
                    mock.patch.object(xp, "_alias_mapper", []), \
                    mock.patch.object(urllib.request, "urlretrieve",
                                      no_download):
                m = xp.alias_mapper()
                assert isinstance(m, gene_name_utils.GeneNameMapper)
                assert m._cache_dir == str(table.parent), m._cache_dir
                assert xp.alias_mapper() is m      # built once
                # (_quiet: the mapper prints what it loaded, once)
                rows, _ = _quiet(lambda: [pp.gene_row(g) for g in (
                    "OLD103", "old103", "twenty", "XX20", "G120")])
                assert rows == [2, 2, 19, 19, 19], rows
                # never to a gene outside the export, and never for digits:
                # "52" is an alias of gene 103 in this table. A name the
                # screens pin (eval_finetuning.SCREEN_SYMBOL_OVERRIDES) is
                # the pinned gene, whatever the table says: here the table
                # would send QARS to gene 120, which is exported
                assert m.symbol_to_entrez("52") == "103"
                assert m.symbol_to_entrez("QARS") == "120"
                assert ef.SCREEN_SYMBOL_OVERRIDES["QARS"] == "5859"
                for unknown in ("OLD999", "G999", "52", "999", "NOPE", "QARS"):
                    try:
                        pp.gene_row(unknown)
                    except KeyError as e:
                        assert "not in this export" in str(e), e
                    else:
                        raise AssertionError(f"{unknown} is not in the export")
            # no table at that path, in a directory that is missing and in
            # one that exists (a checkout's data/cache/ always does): the
            # name stays unknown; nothing is fetched and nothing is made,
            # there or beside the cwd
            bare = root / "cache_only" / gone.name
            bare.parent.mkdir()
            for absent in (gone, bare):
                with mock.patch.object(xp, "HGNC_TABLE", absent), \
                        mock.patch.object(xp, "_alias_mapper", []), \
                        mock.patch.object(urllib.request, "urlretrieve",
                                          no_download):
                    assert xp.alias_mapper() is None, absent
                    assert pp.gene_row("103") == pp.gene_row("G103") == 2
                    try:
                        pp.gene_row("OLD103")
                    except KeyError as e:
                        assert "not in this export" in str(e), e
                    else:
                        raise AssertionError("an older name needs the table")
            # (urlretrieve was never called, with the table or without it)
            assert not downloads, downloads
            assert not gone.parent.exists() and not list(bare.parent.iterdir())
            assert sorted(q.name for q in (root / "elsewhere").iterdir()) == [
                "deeper"] and not list(elsewhere.iterdir())
        finally:
            os.chdir(cwd)
        # the reader built its own mapper: every later get_mapper() caller of
        # this process still gets the table of its own first call
        assert gene_name_utils._default_mapper is process_mapper


def test_export_refusals_leave_a_previous_export_alone():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        rc, text = _export(md, paths)
        assert rc == 0, text
        out = md / "pair_predictions"
        before = (out / "manifest.json").read_text()
        # the two modalities swapped: not the files the checkpoints record
        rc, text = _export(md, paths[::-1])
        assert rc == 1 and "not the embedding files" in text, text
        # an output dir that is the model dir
        for bad in (md, md / "checkpoints"):
            rc, text = _export(md, paths, "--out_dir", str(bad))
            assert rc == 1 and "model dir itself" in text, (bad, text)
        # a device no machine has ("gpu" is not a torch device type): the
        # device is first used after the previous export is cleared
        rc, text = _export(md, paths, "--device", "gpu")
        assert rc == 1 and "not usable" in text, text
        # the two filters choose among the models of --finetuned_eval_dirs
        for flag, value in (("--finetuned_datasets", "Adamson"),
                            ("--finetuned_modes", "Full")):
            rc, text = _export(md, paths, flag, value)
            assert rc == 1 and "which was not given" in text, (flag, text)
        # a partial checkpoint set
        ck = md / "checkpoints" / "fold_1_best.pt"
        ck.rename(ck.with_suffix(".bak"))
        rc, text = _export(md, paths)
        assert rc == 1 and "partial checkpoint set" in text, text
        ck.with_suffix(".bak").rename(ck)
        # a checkpoint from before the per-fold fit stores no transform
        kept = ck.read_bytes()
        d = torch.load(ck, map_location="cpu", weights_only=False)
        d["preprocessing_transform"] = None
        torch.save(d, ck)
        rc, text = _export(md, paths)
        assert rc == 1 and "stores no preprocessing_transform" in text, text
        ck.write_bytes(kept)
        # no room for an export larger than the one it would replace
        with mock.patch.object(xp.shutil, "disk_usage",
                               return_value=mock.Mock(free=0)):
            rc, text = _export(md, paths, "--per_fold")
        assert rc == 1 and "GB free" in text, text
        assert (out / "manifest.json").read_text() == before
        assert (out / "sl_prob_K562.npy").exists()
        # a rerun replaces its own files and nothing else: one left-over of
        # each kind an export with other options wrote
        (out / "notes.txt").write_text("mine")
        stale = ["fold_0_sl_prob_K562.npy", "sl_logit_K562.npy",
                 "fold_0_sl_logit_K562.npy", "finetuned_Corn_Full_cv1.npy",
                 "fold_0_finetuned_Corn_Full_cv1.npy", "sl_prob_GONE.npy"]
        for f in stale:
            (out / f).write_bytes(b"stale")
        rc, text = _export(md, paths)
        assert rc == 0 and "previous export" in text, text
        assert (out / "notes.txt").read_text() == "mine"
        assert not [f for f in stale if (out / f).exists()]
        assert not (out / xp.IN_PROGRESS).exists()
        # a directory holding ANOTHER tool's genes.txt (the shared folds dir)
        # is never cleared: the rerun clear deletes genes.txt by name
        folds = root / "folds"
        folds.mkdir()
        (folds / "genes.txt").write_text("2\n9\n10\n")
        rc, text = _export(md, paths, "--out_dir", str(folds))
        assert rc == 1 and "no export of this script wrote" in text, text
        assert (folds / "genes.txt").read_text() == "2\n9\n10\n"
        # ... nor is one whose manifest.json is another tool's
        (folds / "manifest.json").write_text('{"format": "other.v1"}')
        rc, text = _export(md, paths, "--out_dir", str(folds))
        assert rc == 1 and "no export of this script wrote" in text, text
        assert (folds / "genes.txt").read_text() == "2\n9\n10\n"
        # an export that died half-way (marker, no manifest) can be redone
        (out / "manifest.json").unlink()
        (out / xp.IN_PROGRESS).write_text("x")
        assert xp.foreign_files(out) == []
        rc, text = _export(md, paths)
        assert rc == 0 and not (out / xp.IN_PROGRESS).exists(), text
        # ... but genes.txt with neither marker nor manifest is not ours
        (out / "manifest.json").unlink()
        assert xp.foreign_files(out) == ["genes.txt", "genes.tsv"]
        rc, text = _export(md, paths)
        assert rc == 1 and "no export of this script wrote" in text, text


def test_export_stops_when_a_matrix_is_not_the_forward():
    """The run-time self-check, the only value check a real-size export has:
    a dense matrix that is off on one head, or holds a NaN, stops the export
    with no manifest (an unfinished export)."""
    real = xp.build_any_score_matrix

    def shifted(model, feats, **kw):
        L = real(model, feats, **kw)
        return L + 0.01 if kw.get("only_head") == 5 else L

    def holed(model, feats, **kw):
        L = real(model, feats, **kw)
        L[1, 2] = float("nan")
        return L

    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1]:
        md, paths = _model_dir(Path(tmp))
        for fake, msg in ((shifted, f"fold 0 head {HEADS[5]}: the all-pairs "
                           f"matrix disagrees with model.forward"),
                          (holed, "non-finite entries")):
            with mock.patch.object(xp, "build_any_score_matrix", fake):
                try:
                    _export(md, paths)
                except RuntimeError as e:
                    assert msg in str(e), e
                else:
                    raise AssertionError(f"exported a wrong matrix ({msg})")
            assert not (md / "pair_predictions" / "manifest.json").exists()


def test_export_helpers():
    i, j = xp.sample_pairs(7, 500)
    assert (i != j).all() and i.min() >= 0 and max(i.max(), j.max()) < 7
    i, j = xp.sample_pairs(2, 10)
    assert (i != j).all()
    units = [("head", 0), ("head", 1), ("any", ), ("ft", 0)]
    per = {u: (1 if u[0] == "ft" else 2) for u in units}
    assert xp.plan_passes(units, per, 10, 1e9) == [units]
    assert xp.plan_passes(units, per, 10, 45) == [units[:2], units[2:]]
    assert xp.plan_passes(units, per, 10, 1) == [[u] for u in units]
    g2i = {g: k for k, g in enumerate(GENES)}
    i2g = dict(enumerate(GENES))
    genes, rows, label = xp.resolve_universe("all", None, "x", g2i, i2g)
    assert genes == GENES and rows.tolist() == list(range(N_GENES))
    assert label == "all"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "g.txt"
        f.write_text("101\n101\n")
        try:
            xp.resolve_universe("all", str(f), "x", g2i, i2g)
        except ValueError as e:
            assert "duplicate" in str(e)
        else:
            raise AssertionError("duplicate genes must be refused")
        # a symbol in --genes_file is resolved as the screens resolve it: a
        # pinned name is the pinned gene, whatever the mapper would say
        class Wrong:
            def symbol_to_entrez(self, s):
                return GENES[0]

        f.write_text(f"QARS\n{GENES[1]}\n")
        g2 = {**g2i, "5859": N_GENES}
        with mock.patch.object(gene_name_utils, "get_mapper",
                               return_value=Wrong()):
            genes, rows, _ = xp.resolve_universe(
                "all", str(f), "x", g2, {**i2g, N_GENES: "5859"})
        assert genes == ["5859", GENES[1]] and rows.tolist() == [N_GENES, 1]
        # write_matrix: the off-diagonal statistics of what it stored
        a = np.array([[7, 1, 0], [1, 7, 0.5], [0, 0.5, 7]], dtype=np.float32)
        st = xp.write_matrix(Path(tmp) / "m.npy", a, "float32")
        assert st == {"min": 0.0, "max": 1.0, "mean": 0.5,
                      "n_exactly_0": 2, "n_exactly_1": 2}, st
        got = np.load(Path(tmp) / "m.npy")
        assert np.isnan(np.diag(got)).all() and got[1, 2] == 0.5
        # ... and a save that dies part-way (a full disk) leaves no piece
        # behind and the file already under that name whole
        real_save = np.save

        def half(file, arr, *a_, **kw):
            Path(file).write_bytes(b"half")
            raise OSError(28, "No space left on device")

        with mock.patch.object(xp.np, "save", half):
            try:
                xp.write_matrix(Path(tmp) / "m.npy", a + 1, "float32")
            except OSError as e:
                assert "No space left" in str(e), e
            else:
                raise AssertionError("the failed save must be raised")
        assert np.save is real_save
        assert sorted(q.name for q in Path(tmp).glob("m*")) == ["m.npy"]
        assert np.array_equal(np.load(Path(tmp) / "m.npy"), got,
                              equal_nan=True)

        # ... whatever the save raised, and a cleanup that fails itself does
        # not replace the save's error (last: it leaves m.tmp.npy behind)
        def half_mem(file, arr, *a_, **kw):
            Path(file).write_bytes(b"half")
            raise MemoryError("part-way")

        with mock.patch.object(xp.np, "save", half_mem):
            try:
                xp.write_matrix(Path(tmp) / "m.npy", a + 1, "float32")
            except MemoryError:
                pass
            else:
                raise AssertionError("the failed save must be raised")
        assert sorted(q.name for q in Path(tmp).glob("m*")) == ["m.npy"]
        real_unlink = Path.unlink
        refused = []

        def no_unlink(self, *a_, **kw):
            if self.name.endswith(".tmp.npy"):
                refused.append(self.name)
                raise PermissionError(13, "unlink refused")
            return real_unlink(self, *a_, **kw)

        with mock.patch.object(xp.np, "save", half), \
                mock.patch.object(Path, "unlink", no_unlink):
            try:
                xp.write_matrix(Path(tmp) / "m.npy", a + 1, "float32")
            except OSError as e:
                assert "No space left" in str(e), e
            else:
                raise AssertionError("the failed save must be raised")
        # (the cleanup did go through the fake: else this block tests nothing)
        assert refused == ["m.tmp.npy"], refused


def test_open_pair_predictions_reads_an_export_back():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[1], \
            mock.patch.object(xp, "alias_mapper", return_value=_FakeMapper()):
        md, paths = _model_dir(Path(tmp))
        rc, text = _export(md, paths)
        assert rc == 0, text
        pp = xp.open_pair_predictions(md / "pair_predictions")
        assert pp.names() == HEADS + ["ANY"]
        k562 = np.load(md / "pair_predictions" / "sl_prob_K562.npy")
        assert pp.matrix("K562")[pp.index["103"], pp.index["120"]] == k562[2, 19]
        t = pp.pair_table([("G103", "120"), ("101", "g102")])
        assert list(t.columns[:4]) == ["gene1", "gene2", "gene1_entrez",
                                       "gene2_entrez"]
        assert t["gene1_entrez"].tolist() == ["103", "101"]
        assert t["sl_prob_K562"].tolist() == [k562[2, 19], k562[0, 1]]
        assert len(t.columns) == 4 + len(HEADS) + 1
        # every name reads ITS OWN file: K562 is the first one, so a reader
        # (or a manifest) that hands every head the first file passes above
        own = {lab: np.load(md / "pair_predictions" / f"sl_prob_{lab}.npy")
               for lab in HEADS + ["ANY"]}
        assert len({m.tobytes() for m in own.values()}) == len(own)
        for lab, m in own.items():
            assert np.asarray(pp.matrix(lab)).tobytes() == m.tobytes(), lab
            assert t[f"sl_prob_{lab}"].tolist() == [m[2, 19], m[0, 1]], lab
        # A screen may label a gene by an older name than genes.tsv holds
        # (Adamson: MRE11A for MRE11): resolved as the pipeline resolves it.
        assert pp.gene_row("OLD103") == pp.gene_row("old103") == 2
        t2 = pp.pair_table([("OLD103", "G120")], names=["K562"])
        assert t2["gene1_entrez"].tolist() == ["103"]
        assert t2["sl_prob_K562"].tolist() == [k562[2, 19]]
        # ... but never to a gene outside the export, and never for digits
        # (an Entrez ID that is absent stays absent)
        for unknown in ("OLD999", "999", "NOPE"):
            try:
                pp.gene_row(unknown)
            except KeyError as e:
                assert "not in this export" in str(e), e
            else:
                raise AssertionError(f"{unknown} is not in the export")
        # ... even where the table lists the digits as another gene's alias
        # (HGNC does, three times; the mapper above knows no such name, so it
        # could not tell): a mapper that resolves every name to gene 103
        every = _FakeMapper()
        every.symbol_to_entrez = lambda s: "103"
        with mock.patch.object(xp, "alias_mapper", return_value=every):
            assert pp.gene_row("NOPE") == 2      # this mapper is the one asked
            try:
                pp.gene_row("999")
            except KeyError as e:
                assert "not in this export" in str(e), e
            else:
                raise AssertionError("an absent Entrez ID read as an alias")
        # without an HGNC table (none, or one that cannot be read): IDs and
        # genes.tsv symbols still resolve
        for no_table in (dict(return_value=None),
                         dict(side_effect=RuntimeError("no table"))):
            with mock.patch.object(xp, "alias_mapper", **no_table):
                assert pp.gene_row("103") == pp.gene_row("G103") == 2
                try:
                    pp.gene_row("OLD103")
                except KeyError as e:
                    assert "not in this export" in str(e), e
                else:
                    raise AssertionError(
                        "an alias cannot resolve without a table")
        # the combo is retrained in place: the export describes other weights
        ck = md / "checkpoints" / "fold_0_best.pt"
        d = torch.load(ck, map_location="cpu", weights_only=False)
        d["model_state_dict"] = _make_model("per_head", 99).state_dict()
        torch.save(d, ck)
        try:
            xp.open_pair_predictions(md / "pair_predictions")
        except RuntimeError as e:
            assert "checkpoints/fold_0_best.pt was retrained" in str(e), e
        else:
            raise AssertionError("an export of other weights must be refused")
        old = xp.open_pair_predictions(md / "pair_predictions",
                                       check_model=False)
        assert old.stale_reasons() == [
            "checkpoints/fold_0_best.pt was retrained"]
        # copied away from its model dir there is nothing to compare with
        import shutil
        shutil.copytree(md / "pair_predictions", Path(tmp) / "copy")
        assert xp.open_pair_predictions(Path(tmp) / "copy").names()[-1] == "ANY"
        (md / "pair_predictions" / "manifest.json").unlink()
        try:
            xp.open_pair_predictions(md / "pair_predictions")
        except FileNotFoundError as e:
            assert "not a finished export" in str(e)
        else:
            raise AssertionError("a dir without manifest.json is unfinished")


# =============================================================================
# eval_finetuning.py: per-head probabilities, all-pair fine-tuned predictions
# =============================================================================


def _screen(root: Path, paths, seed: int = 0) -> Path:
    """An Adamson-format screen over every gene pair (symbols, both orders,
    one duplicated measurement). Gamma is a smooth function of the genes'
    features (another model's logit) plus noise, so fine-tuning has something
    to learn and really moves the weights."""
    rng = np.random.default_rng(seed)
    raw, _, _ = load_raw_multimodal(paths)
    X = torch.cat(raw, dim=1)
    pairs = [(i, j) for i in range(N_GENES) for j in range(i + 1, N_GENES)]
    ii = torch.as_tensor([a for a, _ in pairs])
    jj = torch.as_tensor([b for _, b in pairs])
    with torch.no_grad():
        t = _make_model("per_head", 77)(X[ii], X[jj])[:, 5].numpy()
    gamma = -(t - t.mean()) / t.std() + 0.05 * rng.normal(size=len(pairs))
    rows = ["FirstGene\tSecondGene\tGamma"]
    for (i, j), y in zip(pairs, gamma):
        a, b = (f"G{GENES[i]}", f"G{GENES[j]}")
        if (i + j) % 2:
            a, b = b, a
        rows.append(f"{a}\t{b}\t{y:.6f}")
    rows.append(f"G{GENES[0]}\tG{GENES[1]}\t0.25")
    p = root / "screen.tsv"
    p.write_text("\n".join(rows) + "\n")
    (root / "sl.txt").write_text(f"{GENES[0]}\t{GENES[1]}\n")
    return p


def _eval(md, paths, root, *extra, per_head=True, modes=("LP_last", "Full")):
    """eval_finetuning.main() on the synthetic screen; returns (rc, output)."""
    argv = ["--model_dir", str(md), "--embeddings_paths", *paths,
            "--datasets", f"Adamson:screen.tsv:Gamma:-1:tsv:{SCREEN_HEAD}",
            "--sl_path", str(root / "sl.txt"), "--base_dir", str(root),
            "--cv_type", "cv1", "--ft_modes", *modes, "--ft_epochs", "8",
            "--ft_patience", "8", "--ft_lr", "0.01", "--ft_batch_size", "32",
            "--device", DEVICE, *extra]
    if per_head:
        argv.append("--per_head")
    with mock.patch.object(sys, "argv", ["eval_finetuning.py"] + argv):
        return _quiet(ef.main)


def _read(path):
    return pd.read_csv(path, dtype={"gene1_entrez": str, "gene2_entrez": str})


def test_eval_writes_every_head_probability_for_every_pair():
    n_pairs = N_GENES * (N_GENES - 1) // 2
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        rc, text = _eval(md, paths, root)
        assert rc == 0, text
        ed = md / "eval_finetuning_cv1"
        zs = _read(ed / "Adamson_predictions.csv")
        pr = _read(ed / "Adamson_sl_probabilities.csv")
        assert len(zs) == len(pr) == n_pairs
        for c in ("gene1_entrez", "gene2_entrez", "split", "sldb_overlap"):
            assert pr[c].tolist() == zs[c].tolist(), c
        assert np.array_equal(pr["outcome_raw"], zs["outcome_raw"])
        assert pr["gene1_symbol"].tolist() == ["G" + g
                                               for g in pr["gene1_entrez"]]
        want_cols = [f"prob_{lab}_{s}" for lab in HEADS + ["ANY"]
                     for s in ("fold0", "fold1", "mean")]
        assert [c for c in pr.columns if c.startswith("prob_")] == want_cols
        # the scored head's column is the logit of the existing table
        for k in range(N_FOLDS):
            assert np.allclose(pr[f"prob_{SCREEN_HEAD}_fold{k}"],
                               _sigmoid(zs[f"logit_fold{k}"].to_numpy()),
                               rtol=1e-6, atol=1e-7), k
        # every head is the ordinary forward of that head
        logits = _fold_logits(md, paths)
        g2i = {g: i for i, g in enumerate(GENES)}
        i = pr["gene1_entrez"].map(g2i).to_numpy()
        j = pr["gene2_entrez"].map(g2i).to_numpy()
        for h, lab in enumerate(HEADS):
            for k in range(N_FOLDS):
                assert np.allclose(pr[f"prob_{lab}_fold{k}"],
                                   _sigmoid(logits[k, i, j, h]),
                                   rtol=1e-4, atol=PTOL), (lab, k)
            assert np.allclose(
                pr[f"prob_{lab}_mean"],
                pr[[f"prob_{lab}_fold{k}" for k in range(N_FOLDS)]].mean(
                    axis=1), rtol=1e-6, atol=1e-8), lab
        for k in range(N_FOLDS):
            assert np.allclose(
                pr[f"prob_ANY_fold{k}"],
                pr[[f"prob_{lab}_fold{k}" for lab in HEADS]].max(axis=1),
                rtol=1e-7), k
        cfg = json.loads((ed / "config_used.json").read_text())
        assert cfg["sl_probability_heads"] == HEADS
        assert cfg["finetuned_models_dir"] == "finetuned_models"
        assert cfg["effective_head"] == {"Adamson": SCREEN_HEAD}


def test_eval_writes_finetuned_predictions_for_every_pair():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        rc, text = _eval(md, paths, root)
        assert rc == 0, text
        ed = md / "eval_finetuning_cv1"
        zs = _read(ed / "Adamson_predictions.csv")
        # The two CSVs that predate the all-pair outputs keep exactly their
        # columns, in order (external_transfer.py and paper_tables.py read
        # them): nothing added, dropped or reordered, no index column.
        pair = ["gene1_entrez", "gene2_entrez", "outcome_raw", "target",
                "sldb_overlap"]
        assert list(zs.columns) == pair + ["split"] + [
            f"logit_fold{k}" for k in range(N_FOLDS)] + ["logit_mean"], \
            list(zs.columns)
        for mode in ("LP_last", "Full"):
            allp = _read(ed / f"Adamson_finetuned_allpairs_{mode}.csv")
            test = _read(ed / f"Adamson_finetuned_predictions_{mode}.csv")
            assert len(allp) == len(zs)
            assert allp["split"].tolist() == zs["split"].tolist()
            assert set(allp["split"]) == {"train", "val", "test"}
            # the existing table is unchanged: test rows only, no split column
            assert list(test.columns) == pair + [
                f"pred_fold{k}" for k in range(N_FOLDS)] + ["pred_mean"], \
                (mode, list(test.columns))
            sub = allp[allp["split"] == "test"].reset_index(drop=True)
            assert len(test) == len(sub) < len(allp)
            for c in ("gene1_entrez", "gene2_entrez"):
                assert sub[c].tolist() == test[c].tolist()
            for c in ["pred_fold0", "pred_fold1", "pred_mean"]:
                assert np.array_equal(sub[c], test[c]), (mode, c)
                # sign -1: the screen's own units are the negated target scale
                assert np.array_equal(allp[c.replace("pred_", "pred_outcome_")],
                                      -allp[c]), (mode, c)
            assert np.array_equal(allp["target"], -allp["outcome_raw"])
        saved = sorted(p.name for p in (ed / "finetuned_models").glob("*.pt"))
        assert saved == sorted(f"Adamson_{m}_fold{k}.pt"
                               for m in ("LP_last", "Full")
                               for k in range(N_FOLDS)), saved
        ck = torch.load(ed / "finetuned_models" / "Adamson_Full_fold1.pt",
                        map_location="cpu", weights_only=False)
        assert ck["format"] == ef.FINETUNED_FORMAT
        assert ck["head"] == SCREEN_HEAD
        assert ck["head_idx"] == HEADS.index(SCREEN_HEAD)
        assert ck["dataset"] == "Adamson" and ck["column"] == "Gamma"
        assert ck["sign"] == -1.0 and ck["fold"] == 1 and ck["cv_type"] == "cv1"
        assert ck["source_checkpoint_sha256"] == ef._file_sha256(
            md / "checkpoints" / "fold_1_best.pt")
        assert ck["config"] == {"pd_epsilon": 0.01, "dropout": 0.1}
        # the split it was fine-tuned on (eval_finetuning.py's defaults), as
        # counted in the per-pair table
        assert (ck["split_seed"], ck["train_frac"], ck["val_frac"]) == (
            42, 0.8, 0.1)
        n_split = zs["split"].value_counts()
        assert (ck["n_pairs"], ck["n_genes"]) == (len(zs), N_GENES)
        assert (ck["n_train"], ck["n_val"], ck["n_test"]) == (
            n_split["train"], n_split["val"], n_split["test"])
        assert not list((ed / "finetuned_models").glob("*.tmp"))
        # The saved weights are the SELECTED ones: they differ from the
        # pretrained fold exactly when a fine-tuning epoch beat the warm start
        # (best_epoch > 0); the cell embedding stays frozen in every mode.
        epochs = []
        for f in sorted((ed / "finetuned_models").glob("*.pt")):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            src = torch.load(md / "checkpoints" / f"fold_{ck['fold']}_best.pt",
                             map_location="cpu",
                             weights_only=False)["model_state_dict"]
            ft = ck["model_state_dict"]
            assert set(ft) == set(src), f.name
            assert torch.equal(ft["cell_emb.weight"], src["cell_emb.weight"])
            moved = not torch.equal(ft["encoder.projection.weight"],
                                    src["encoder.projection.weight"])
            assert moved == (ck["best_epoch"] > 0), (f.name, ck["best_epoch"])
            epochs.append(ck["best_epoch"])
        assert max(epochs) > 0, \
            f"no fold fine-tuned past its warm start ({epochs}): the " \
            f"all-pair checks below would not see a wrong weight set"


def test_eval_scratch_mode_trains_a_fresh_model_through_the_same_outputs():
    """--ft_modes ... Scratch (the from-scratch baseline): its CSVs, rows and
    summaries are Full's in shape; no tensor of its saved models is the
    checkpoint's; what Full keeps frozen is frozen at the fresh init; it
    trains; and adding it leaves the LP_last / Full outputs byte-identical."""
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        rc, text = _eval(md, paths, root, "--report_sldb_filtered")
        assert rc == 0, text
        ed = md / "eval_finetuning_cv1"
        before = {f.name: f.read_bytes() for f in ed.glob("*.csv")
                  if "Scratch" not in f.name and "summary" not in f.name
                  and "metrics" not in f.name}
        rc, text = _eval(md, paths, root, "--report_sldb_filtered",
                         modes=("LP_last", "Full", ef.SCRATCH_MODE))
        assert rc == 0, text
        assert "OLS on its own logits" in text
        for name, data in before.items():
            assert (ed / name).read_bytes() == data, \
                f"{name} changed when Scratch was added"
        # the long and summary tables: Full's rows unchanged, Scratch's added
        for f, key in (("finetuned_metrics.csv", "mode"),
                       ("finetuned_summary.csv", "mode")):
            df = pd.read_csv(ed / f)
            assert set(df[key]) == {"LP_last", "Full", ef.SCRATCH_MODE}, f
            full = df[df[key] == "Full"].drop(columns=[key]).reset_index(
                drop=True)
            scr = df[df[key] == ef.SCRATCH_MODE].drop(
                columns=[key]).reset_index(drop=True)
            assert list(full.columns) == list(scr.columns)
            assert len(full) == len(scr) > 0, f
        for kind in ("finetuned_predictions", "finetuned_allpairs"):
            a = _read(ed / f"Adamson_{kind}_Full.csv")
            b = _read(ed / f"Adamson_{kind}_{ef.SCRATCH_MODE}.csv")
            assert list(a.columns) == list(b.columns), kind
            assert a["gene1_entrez"].tolist() == b["gene1_entrez"].tolist()
            assert not np.allclose(a["pred_mean"], b["pred_mean"]), kind
        cfg = json.loads((ed / "config_used.json").read_text())
        assert cfg["ft_modes"] == ["LP_last", "Full", ef.SCRATCH_MODE]
        assert cfg["scratch_init_seeds"] == [
            ef.scratch_init_seed(42, k) for k in range(N_FOLDS)]
        epochs = []
        for k in range(N_FOLDS):
            ck = torch.load(ed / "finetuned_models" /
                            f"Adamson_{ef.SCRATCH_MODE}_fold{k}.pt",
                            map_location="cpu", weights_only=False)
            assert ck["mode"] == ef.SCRATCH_MODE and ck["init"] == "scratch"
            assert ck["init_seed"] == ef.scratch_init_seed(42, k)
            src_path = md / "checkpoints" / f"fold_{k}_best.pt"
            assert ck["source_checkpoint_sha256"] == ef._file_sha256(src_path)
            src = torch.load(src_path, map_location="cpu",
                             weights_only=False)["model_state_dict"]
            ft = ck["model_state_dict"]
            assert set(ft) == set(src)
            # the fresh model it started from (rebuilt the same way)
            fresh = ef.scoring_model(ef._load_eval_base(
                src_path, type("A", (), {"per_head": True})(),
                HEADS.index(SCREEN_HEAD), "cpu",
                init_seed=ck["init_seed"])).state_dict()
            # Every tensor the checkpoint moved off its init is not the
            # checkpoint's here (tau and the zero input bias, constant inits
            # _make_model never moves, are the same number either way).
            for key in src:
                if not torch.equal(src[key], fresh[key]):
                    assert not torch.equal(ft[key], src[key]), \
                        f"fold {k} {key}: the pretrained value survived"
            same = {x for x in src if torch.equal(src[x], fresh[x])}
            # only constant inits the synthetic checkpoint never moved
            assert same <= {"tau", "encoder.input_bias"}, same
            # frozen in Full -> frozen at the init value here
            for key in ("cell_emb.weight", "tau", "scoring_bias"):
                assert torch.equal(ft[key], fresh[key]), (k, key)
            moved = not torch.equal(ft["encoder.projection.weight"],
                                    fresh["encoder.projection.weight"])
            assert moved == (ck["best_epoch"] > 0), (k, ck["best_epoch"])
            epochs.append(ck["best_epoch"])
        assert max(epochs) > 0, f"Scratch never trained past its start {epochs}"


def test_export_scores_all_gene_pairs_with_the_finetuned_models():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        rc, text = _eval(md, paths, root)
        assert rc == 0, text
        ed = md / "eval_finetuning_cv1"
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Adamson")
        assert rc == 0, text
        out = md / "pair_predictions"
        mf = json.loads((out / "manifest.json").read_text())
        assert [f["name"] for f in mf["finetuned"]] == [
            "finetuned_Adamson_Full_cv1", "finetuned_Adamson_LP_last_cv1"]
        assert mf["finetuned_not_exported"] == []
        g2i = {g: i for i, g in enumerate(GENES)}
        pr = _read(ed / "Adamson_sl_probabilities.csv")
        i = pr["gene1_entrez"].map(g2i).to_numpy()
        j = pr["gene2_entrez"].map(g2i).to_numpy()
        # the genome-wide matrices agree with the per-screen tables, both ways
        for lab in HEADS + ["ANY"]:
            m = np.load(out / f"sl_prob_{lab}.npy")
            assert np.allclose(m[i, j], pr[f"prob_{lab}_mean"], atol=PTOL), lab
            assert np.allclose(m[j, i], pr[f"prob_{lab}_mean"], atol=PTOL), lab
        for entry in mf["finetuned"]:
            assert entry["head"] == SCREEN_HEAD and entry["sign"] == -1.0
            assert entry["head_idx"] == HEADS.index(SCREEN_HEAD)
            assert entry["column"] == "Gamma" and entry["eval_cv"] == "cv1"
            assert [f["fold"] for f in entry["folds"]] == [0, 1]
            allp = _read(ed / f"Adamson_finetuned_allpairs_{entry['mode']}.csv")
            m = np.load(out / entry["file"])
            assert np.isnan(np.diag(m)).all()
            assert np.allclose(m[i, j], allp["pred_outcome_mean"],
                               rtol=1e-4, atol=1e-4), entry["name"]
            assert np.allclose(m[i, j], -allp["pred_mean"],
                               rtol=1e-4, atol=1e-4), entry["name"]
            # which entries are held out travels with the matrix: the screen
            # split of the models, and the per-pair table that carries it
            ss = entry["screen_split"]
            assert (ss["cv_type"], ss["split_seed"]) == ("cv1", 42)
            assert ss["n_pairs"] == len(allp) and ss["n_genes"] == N_GENES
            assert [ss[f"n_{x}"] for x in ("train", "val", "test")] == [
                int((allp["split"] == x).sum())
                for x in ("train", "val", "test")]
            assert Path(entry["screen_pairs"]) == ed / (
                f"Adamson_finetuned_allpairs_{entry['mode']}.csv")
            assert "split == test" in entry["held_out"]
            assert "growth phenotype" in entry["note"]
            assert entry["eval_dir"] == "eval_finetuning_cv1"
        pp = xp.open_pair_predictions(out)
        assert pp.names()[-2:] == ["finetuned_Adamson_Full_cv1",
                                   "finetuned_Adamson_LP_last_cv1"]
        ft_names = [e["name"] for e in mf["finetuned"]]
        t = pp.pair_table([("101", "102")], names=ft_names)
        assert list(t.columns[4:]) == ft_names
        own = {n: np.load(out / f"{n}.npy") for n in ft_names}
        # the two modes must differ, or a name that opens the other mode's
        # file (or carries its statistics) would go unseen
        assert own[ft_names[0]].tobytes() != own[ft_names[1]].tobytes()
        for e in mf["finetuned"]:
            m = own[e["name"]]
            assert np.asarray(pp.matrix(e["name"])).tobytes() == m.tobytes()
            assert t[e["name"]].tolist() == [m[0, 1]], e["name"]
            assert abs(e["stats"]["mean"]
                       - np.nanmean(m, dtype=np.float64)) < 1e-9, e["name"]

        # The options reach the fine-tuned sets too: a small --ram_gb gives
        # each a pass of its own (no head in it), --dtype and --per_fold
        # apply to its files. Same values as the one-pass float32 run above,
        # and fold_<k>_* is fold k's own fine-tuned model.
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Adamson", "--ram_gb", "1e-9", "--per_fold",
                           "--dtype", "float16", "--out_dir",
                           str(root / "small"))
        assert rc == 0 and "11 passes over the folds" in text, text
        small = json.loads((root / "small" / "manifest.json").read_text())
        for f in sorted(out.glob("*.npy")):
            for name in [f.name] + [f"fold_{k}_{f.name}"
                                    for k in range(N_FOLDS)]:
                assert np.load(root / "small" / name).dtype == np.float16, name
            assert np.allclose(np.load(root / "small" / f.name), np.load(f),
                               rtol=2e-3, atol=2e-3, equal_nan=True), f.name
        assert [e["per_fold_files"] for e in small["finetuned"]] == [
            [f"fold_{k}_{e['name']}.npy" for k in range(N_FOLDS)]
            for e in mf["finetuned"]]
        for e in small["finetuned"]:
            allp = _read(ed / f"Adamson_finetuned_allpairs_{e['mode']}.csv")
            for k, name in enumerate(e["per_fold_files"]):
                mk = np.load(root / "small" / name)
                assert np.isnan(np.diag(mk)).all(), name
                assert np.allclose(mk[i, j], allp[f"pred_outcome_fold{k}"],
                                   rtol=2e-3, atol=2e-3), name

        # an eval rerun rewrites finetuned_models/ under unchanged fold
        # checkpoints: the matrices scored with the old ones are stale
        ft = ed / "finetuned_models" / "Adamson_Full_fold1.pt"
        kept = ft.read_bytes()
        d = torch.load(ft, map_location="cpu", weights_only=False)
        d["affine"]["c"] += 1.0
        torch.save(d, ft)
        try:
            xp.open_pair_predictions(out)
        except RuntimeError as e:
            assert ("eval_finetuning_cv1/finetuned_models/Adamson_Full_"
                    "fold1.pt was fine-tuned again") in str(e), e
            assert "finetuned_Adamson_Full_cv1.npy" in str(e), e
        else:
            raise AssertionError("matrices of replaced fine-tuned models "
                                 "must be refused")
        ft.write_bytes(kept)
        assert xp.open_pair_predictions(out).stale_reasons() == []

        # a fold missing from one set (an eval task that timed out): that set
        # is reported, the other set and the probabilities are exported
        ft.unlink()
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1")
        assert rc == 1 and "Adamson / Full: folds [0], need 0..1" in text, text
        part = json.loads((out / "manifest.json").read_text())
        assert [f["name"] for f in part["finetuned"]] == [
            "finetuned_Adamson_LP_last_cv1"]
        assert len(part["finetuned_not_exported"]) == 1
        assert (out / "sl_prob_K562.npy").exists()
        # a save cut short (a full quota, a killed task) leaves a file torch
        # cannot read: reported like the missing fold it is, the rest exported
        ft.write_bytes(kept[:len(kept) // 2])
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Adamson")
        assert rc == 1 and "Adamson_Full_fold1.pt: unreadable" in text, text
        # one set, two WARNING lines (the file, and the fold gap it leaves):
        # the closing ERROR line counts the lines
        assert text.count("WARNING: fine-tuned set NOT exported") == 2 \
            and "ERROR: 2 problem(s) with the requested" in text, text
        part = json.loads((out / "manifest.json").read_text())
        assert [f["name"] for f in part["finetuned"]] == [
            "finetuned_Adamson_LP_last_cv1"]
        assert any("unreadable" in m for m in part["finetuned_not_exported"])
        assert (out / "sl_prob_K562.npy").exists()
        ft.write_bytes(kept)
        # ... one of a screen that was NOT asked for costs nothing
        bad = ed / "finetuned_models" / "Corn_Full_fold0.pt"
        bad.write_bytes(kept[:len(kept) // 2])
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Adamson")
        assert rc == 0 and "ignoring" in text and bad.name in text, text
        # ... and with no dataset filter it is asked for
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1")
        assert rc == 1 and "Corn_Full_fold0.pt: unreadable" in text, text
        part = json.loads((out / "manifest.json").read_text())
        assert len(part["finetuned"]) == 2, part["finetuned_not_exported"]
        bad.unlink()
        # ... nor does one of a MODE that was not asked for, with or without
        # a dataset filter
        lp = ed / "finetuned_models" / "Adamson_LP_last_fold0.pt"
        kept_lp = lp.read_bytes()
        lp.write_bytes(kept_lp[:len(kept_lp) // 2])
        for flt in ([], ["--finetuned_datasets", "Adamson"]):
            rc, text = _export(md, paths, "--finetuned_eval_dirs",
                               "eval_finetuning_cv1", *flt,
                               "--finetuned_modes", "Full")
            assert rc == 0 and "ignoring" in text and lp.name in text, text
            part = json.loads((out / "manifest.json").read_text())
            assert [f["name"] for f in part["finetuned"]] == [
                "finetuned_Adamson_Full_cv1"]
            assert part["finetuned_not_exported"] == []
        # ... while its own mode asks for it (the whole name: LP_last is not
        # LP, which has no model here)
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_modes",
                           "LP_last")
        assert rc == 1 and "Adamson_LP_last_fold0.pt: unreadable" in text \
            and "ignoring" not in text, text
        # (one lost set, two WARNING lines: the unreadable file and the fold
        # it leaves missing; the closing ERROR counts the lines, not sets)
        assert text.count("WARNING: fine-tuned set NOT exported") == 2 \
            and "ERROR: 2 problem(s) with the requested fine-tuned sets" \
            in text, text
        # ... also beside another requested mode and another requested screen
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_modes",
                           "Full", "LP_last", "--finetuned_datasets",
                           "Adamson", "Corn")
        assert rc == 1 and "Adamson_LP_last_fold0.pt: unreadable" in text \
            and "ignoring" not in text, text
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_modes",
                           "Full", "LP")
        assert rc == 1 and "ignoring" in text and lp.name in text \
            and "no model of the requested LP" in text, text
        lp.write_bytes(kept_lp)
        # models saved before the pair counts were recorded export all the
        # same; the manifest holds null for what they do not say
        counts = ("n_pairs", "n_genes", "n_train", "n_val", "n_test")
        olds = {}
        for k in range(N_FOLDS):
            f = ed / "finetuned_models" / f"Adamson_LP_last_fold{k}.pt"
            olds[f] = f.read_bytes()
            d = torch.load(f, map_location="cpu", weights_only=False)
            for c in counts:
                del d[c]
            torch.save(d, f)
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1")
        assert rc == 0, text
        by_mode = {e["mode"]: e["screen_split"] for e in json.loads(
            (out / "manifest.json").read_text())["finetuned"]}
        assert [by_mode["LP_last"][c] for c in counts] == [None] * 5, by_mode
        assert by_mode["LP_last"]["cv_type"] == "cv1"
        assert None not in by_mode["Full"].values(), by_mode
        for f, b in olds.items():
            f.write_bytes(b)

        # a dataset / mode filter that matches nothing is reported
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Corn")
        assert rc == 1 and "no model matches" in text, text
        # ... and so is a requested name with no model beside one that has
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Adamson", "Corn")
        assert rc == 1 and "no model of the requested Corn" in text, text
        part = json.loads((out / "manifest.json").read_text())
        assert len(part["finetuned"]) == 2, part["finetuned"]
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_datasets",
                           "Adamson", "--finetuned_modes", "LP_last", "LP")
        assert rc == 1 and "no model of the requested Adamson / LP" in text \
            and "Adamson / LP_last" not in text, text
        # a mode filter that every name satisfies selects, and reports nothing
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "--finetuned_modes", "Full")
        assert rc == 0, text
        assert [q.name for q in out.glob("finetuned_*.npy")] == [
            "finetuned_Adamson_Full_cv1.npy"]

        # An eval kept OUTSIDE the model dir (its --out_dir), beside the
        # pipeline's own eval_finetuning_cv1 with same-named, other models:
        # the reader compares the files that were scored, not their namesakes.
        kept_dir = root / "kept" / "eval_finetuning_cv1"
        rc, text = _eval(md, paths, root, "--out_dir", str(kept_dir),
                         "--ft_lr", "0.02")
        assert rc == 0, text
        rc, text = _export(md, paths, "--finetuned_eval_dirs", str(kept_dir))
        assert rc == 0, text
        pp = xp.open_pair_predictions(out)
        assert pp.stale_reasons() == []
        assert {e["eval_dir"] for e in pp.manifest["finetuned"]} == {
            os.path.join("..", "kept", "eval_finetuning_cv1")}
        other = kept_dir / "finetuned_models" / "Adamson_Full_fold1.pt"
        d = torch.load(other, map_location="cpu", weights_only=False)
        d["affine"]["c"] += 1.0
        torch.save(d, other)
        stale = xp.open_pair_predictions(out, check_model=False).stale_reasons()
        assert len(stale) == 1 and "kept/eval_finetuning_cv1/finetuned_models/" \
            "Adamson_Full_fold1.pt was fine-tuned again" in stale[0], stale
        # an eval dir that never ran
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv2")
        assert rc == 1 and "no finetuned_models" in text, text

        # sets discover_finetuned must report instead of exporting: a file
        # that is not a fine-tuned model, folds fine-tuned through different
        # heads, two eval dirs that would be written under one name
        import shutil
        fdir = ed / "finetuned_models"
        for stray in ({"model_state_dict": {}}, [1.0]):   # a dict or not
            torch.save(stray, fdir / "stray.pt")
            rc, text = _export(md, paths, "--finetuned_eval_dirs",
                               "eval_finetuning_cv1")
            assert rc == 1 and f"stray.pt: not a {ef.FINETUNED_FORMAT} " \
                "file" in text, text
        (fdir / "stray.pt").unlink()
        d = torch.load(ft, map_location="cpu", weights_only=False)
        d["head_idx"] = 0
        torch.save(d, ft)
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1")
        assert rc == 1 and "Adamson / Full: folds disagree on head_idx" \
            in text, text
        ft.write_bytes(kept)
        shutil.copytree(ed, md / "eval_again")
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1", "eval_again")
        assert rc == 1 and text.count("would both be written as") == 2, text
        assert not list(out.glob("finetuned_*.npy"))
        shutil.rmtree(md / "eval_again")
        # ... while one eval dir named twice is one eval dir: the same name
        # again, with a trailing slash, by its absolute path
        for again in ("eval_finetuning_cv1", "eval_finetuning_cv1/", str(ed)):
            rc, text = _export(md, paths, "--finetuned_eval_dirs",
                               "eval_finetuning_cv1", again)
            assert rc == 0 and "would both be written" not in text, text
            assert len(list(out.glob("finetuned_*.npy"))) == 2

        # the combo is retrained in place: the old fine-tuned models belong
        # to other weights and must not be paired with the new folds
        ck = md / "checkpoints" / "fold_1_best.pt"
        d = torch.load(ck, map_location="cpu", weights_only=False)
        d["model_state_dict"] = _make_model("per_head", 99).state_dict()
        torch.save(d, ck)
        rc, text = _export(md, paths, "--finetuned_eval_dirs",
                           "eval_finetuning_cv1")
        assert rc == 1 and "other weights" in text, text
        mf = json.loads((out / "manifest.json").read_text())
        assert mf["finetuned"] == [] and len(mf["finetuned_not_exported"]) == 2
        assert not list(out.glob("finetuned_*.npy"))
        assert (out / "sl_prob_K562.npy").exists()


def test_export_a_finetuned_set_that_cannot_be_scored_costs_nothing_else():
    """A fine-tuned set that fails while it is scored (its forward check on
    the last fold or on the first, a model that cannot be loaded, a write
    that dies part-way or after the rename) is dropped and reported; the
    per-cell-line matrices and the other fine-tuned set are written, exit
    1, and no file of the dropped set is left."""
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        rc, text = _eval(md, paths, root)
        assert rc == 0, text
        real = xp.check_against_forward

        def check(L, ref, pi, pj, what):
            if what == f"fold {N_FOLDS - 1} finetuned_Adamson_Full_cv1":
                raise RuntimeError(f"{what}: injected")
            return real(L, ref, pi, pj, what)

        with mock.patch.object(xp, "check_against_forward", check):
            rc, text = _export(md, paths, "--finetuned_eval_dirs",
                               "eval_finetuning_cv1", "--per_fold")
        assert rc == 1 and "everything else is complete" in text, text
        # the closing line counts what is on disk: the heads, ANY, one set
        n_mean = len(HEADS) + 2
        assert f"{n_mean} fold-mean matrices + {n_mean * N_FOLDS} per-fold " \
            in text, text
        out = md / "pair_predictions"
        mf = json.loads((out / "manifest.json").read_text())
        assert [f["name"] for f in mf["finetuned"]] == [
            "finetuned_Adamson_LP_last_cv1"]
        assert "injected" in mf["finetuned_not_exported"][0]
        # the dropped set is gone completely: no mean, no earlier fold's file
        assert not list(out.glob("*Adamson_Full*"))
        assert sorted(q.name for q in out.glob("sl_prob_*.npy")) == sorted(
            f"sl_prob_{lab}.npy" for lab in HEADS + ["ANY"])
        assert xp.open_pair_predictions(out).names()[-1] == \
            "finetuned_Adamson_LP_last_cv1"
        kept_files = {q.name: np.load(q) for q in out.glob("*.npy")}
        assert len(kept_files) == n_mean * (1 + N_FOLDS), sorted(kept_files)

        def dropped(rc, text, why):
            """The Full set is gone without a trace; all else is as above
            (another run of the export: compared with a tolerance)."""
            assert rc == 1 and why in text \
                and "everything else is complete" in text, text
            assert not list(out.glob("*Adamson_Full*")), sorted(
                q.name for q in out.glob("*Adamson_Full*"))
            assert not list(out.glob("*.tmp.npy"))
            assert sorted(q.name for q in out.glob("*.npy")) == sorted(
                kept_files)
            for name, m in kept_files.items():
                assert np.allclose(np.load(out / name), m, rtol=0, atol=PTOL,
                                   equal_nan=True), name
            now = json.loads((out / "manifest.json").read_text())
            assert [f["name"] for f in now["finetuned"]] == [
                "finetuned_Adamson_LP_last_cv1"]
            assert len(now["finetuned_not_exported"]) == 1, now

        # ... on the FIRST fold as well: the later folds must not score the
        # set again (their sum / n_folds would be written under its name)
        def first(L, ref, pi, pj, what):
            if what == "fold 0 finetuned_Adamson_Full_cv1":
                raise RuntimeError(f"{what}: injected")
            return real(L, ref, pi, pj, what)

        with mock.patch.object(xp, "check_against_forward", first):
            dropped(*_export(md, paths, "--finetuned_eval_dirs",
                             "eval_finetuning_cv1", "--per_fold"),
                    "Adamson / Full: fold 0 could not be scored")

        # ... when its model cannot even be loaded (a state dict that no
        # longer fits)
        real_load = xp.load_scorer

        def load(path, device):
            if Path(path).name == "Adamson_Full_fold0.pt":
                raise RuntimeError("injected: state dict does not fit")
            return real_load(path, device)

        with mock.patch.object(xp, "load_scorer", load):
            dropped(*_export(md, paths, "--finetuned_eval_dirs",
                             "eval_finetuning_cv1", "--per_fold"),
                    "Adamson / Full: fold 0 could not be scored")

        # ... and when the fold file it was writing fails: emit() lists a
        # fold file only once write_matrix has returned, so neither a save
        # that dies part-way (<name>.tmp.npy) nor a failure after the rename
        # may leave that file behind
        real_save, real_replace = np.save, os.replace
        last = f"fold_{N_FOLDS - 1}_finetuned_Adamson_Full"

        def save(file, arr, *a, **kw):
            if last in str(file):
                Path(file).write_bytes(b"partial")
                raise OSError(28, "No space left on device")
            return real_save(file, arr, *a, **kw)

        def replace(src, dst, *a, **kw):
            real_replace(src, dst, *a, **kw)
            if last in str(dst):
                raise MemoryError("after the rename")

        for target, name, fake, why in (
                (xp.np, "save", save, "No space left on device"),
                (xp.os, "replace", replace, "after the rename")):
            with mock.patch.object(target, name, fake):
                dropped(*_export(md, paths, "--finetuned_eval_dirs",
                                 "eval_finetuning_cv1", "--per_fold"), why)

        # ... and when the set scored SECOND in a fold is the one dropped
        # (after the rename, so its own fold file is there to remove), the
        # first set's file of that fold, already written, stays
        def second(src, dst, *a, **kw):
            real_replace(src, dst, *a, **kw)
            if f"fold_{N_FOLDS - 1}_finetuned_Adamson_LP_last" in str(dst):
                raise MemoryError("after the rename")

        with mock.patch.object(xp.os, "replace", second):
            rc, text = _export(md, paths, "--finetuned_eval_dirs",
                               "eval_finetuning_cv1", "--per_fold")
        assert rc == 1 and "ERROR: 1 problem(s) with the requested" in text \
            and not list(out.glob("*Adamson_LP_last*")), text
        now = json.loads((out / "manifest.json").read_text())
        assert [f["name"] for f in now["finetuned"]] == [
            "finetuned_Adamson_Full_cv1"]
        assert sorted(q.name for q in out.glob("*Adamson_Full*")) == sorted(
            [now["finetuned"][0]["file"]]
            + now["finetuned"][0]["per_fold_files"]), text

        # a head that fails its check is still fatal: no manifest
        def head_check(L, ref, pi, pj, what):
            if what == "fold 0 head K562":
                raise RuntimeError(f"{what}: injected")
            return real(L, ref, pi, pj, what)

        with mock.patch.object(xp, "check_against_forward", head_check):
            try:
                _export(md, paths)
            except RuntimeError as e:
                assert "fold 0 head K562" in str(e)
            else:
                raise AssertionError("a failed head check must stop the export")
        assert not (out / "manifest.json").exists()


def test_eval_one_head_models_use_their_single_head():
    for kind, per_head, head in (("pooled", True, ef.POOLED_HEAD_LABEL),
                                 ("single", False, "ANY")):
        with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
                _mapper_patches()[1]:
            root = Path(tmp)
            md, paths = _model_dir(root, kind)
            _screen(root, paths)
            rc, text = _eval(md, paths, root, per_head=per_head,
                             modes=("Full", ))
            assert rc == 0, (kind, text)
            ed = md / "eval_finetuning_cv1"
            pr = _read(ed / "Adamson_sl_probabilities.csv")
            zs = _read(ed / "Adamson_predictions.csv")
            assert [c for c in pr.columns if c.startswith("prob_")] == [
                "prob_ANY_fold0", "prob_ANY_fold1", "prob_ANY_mean"], kind
            assert np.allclose(pr["prob_ANY_fold1"],
                               _sigmoid(zs["logit_fold1"].to_numpy()),
                               rtol=1e-6, atol=1e-7), kind
            ck = torch.load(ed / "finetuned_models" / "Adamson_Full_fold0.pt",
                            map_location="cpu", weights_only=False)
            assert ck["head"] == head and ck["head_idx"] == 0, kind
            cfg = json.loads((ed / "config_used.json").read_text())
            assert cfg["sl_probability_heads"] == ["ANY"], kind
            rc, text = _export(md, paths, "--finetuned_eval_dirs",
                               "eval_finetuning_cv1")
            assert rc == 0, (kind, text)
            allp = _read(ed / "Adamson_finetuned_allpairs_Full.csv")
            g2i = {g: i for i, g in enumerate(GENES)}
            i = allp["gene1_entrez"].map(g2i).to_numpy()
            j = allp["gene2_entrez"].map(g2i).to_numpy()
            m = np.load(md / "pair_predictions" /
                        "finetuned_Adamson_Full_cv1.npy")
            assert np.allclose(m[i, j], allp["pred_outcome_mean"],
                               rtol=1e-4, atol=1e-4), kind


def test_predict_refuses_a_finetuned_model_file():
    """predict.py prints SL probabilities from a train.py checkpoint. A
    fine-tuned file of the single-output model loads (export_pair_predictions
    needs that), but it holds no transform and none of the keys predict.py's
    input guards read, and its output is a screen value: main() must refuse
    it, not read it as a legacy checkpoint. Its source checkpoint is the
    control, so the refusal can only come from the file kind."""
    import predict
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root, "single")
        _screen(root, paths)
        rc, text = _eval(md, paths, root, per_head=False, modes=("Full", ))
        assert rc == 0, text
        ft = (md / "eval_finetuning_cv1" / "finetuned_models" /
              "Adamson_Full_fold0.pt")

        def _predict(model):
            argv = ["predict.py", "--model", str(model), "--embeddings_paths",
                    *paths, "--gene1", GENES[0], "--gene2", GENES[1]]
            with mock.patch.object(sys, "argv", argv):
                return _quiet(predict.main)[1]

        assert "SL probability" in _predict(
            md / "checkpoints" / "fold_0_best.pt")      # control
        try:
            out = _predict(ft)
        except SystemExit as e:
            assert "fine-tuned model" in str(e), e
        else:
            raise AssertionError("predict.py scored a fine-tuned model file "
                                 "as an SL probability: " + out[-120:])
        # the loader itself must keep reading it (load_scorer -> load_model)
        assert isinstance(xp.load_scorer(ft, "cpu"), SiameseSL)


def test_eval_tables_the_readers_take_are_never_left_cut_short():
    """<Screen>_predictions.csv and <Screen>_finetuned_predictions_<mode>.csv
    are what external_transfer.py and paper_tables.py read, and neither
    reader knows how many rows a file should hold. A write that dies
    part-way (a full quota) fails the run and leaves nothing under the name
    they read; a piece a killed task left goes with the next run."""
    real_to_csv = pd.DataFrame.to_csv

    def quota_on(marker):
        def to_csv(self, path_or_buf=None, *a, **kw):
            if marker in Path(str(path_or_buf)).name:
                Path(path_or_buf).write_text("gene1_entrez,gene2_ent")
                raise OSError(28, "No space left on device")
            return real_to_csv(self, path_or_buf, *a, **kw)

        return mock.patch.object(pd.DataFrame, "to_csv", to_csv)

    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        ed = md / "eval_finetuning_cv1"
        rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 0, text
        names = ("Adamson_predictions.csv",
                 "Adamson_finetuned_predictions_LP_last.csv")
        rows = {n: _read(ed / n)[["gene1_entrez", "gene2_entrez"]]
                for n in names}
        for name in names:
            with quota_on(name):
                try:
                    rc, text = _eval(md, paths, root, modes=("LP_last", ))
                except OSError as e:
                    assert "No space left" in str(e), e
                else:
                    assert rc != 0, (name, text[-300:])
            assert not (ed / name).exists(), name
            assert not list(ed.glob("*.tmp")), sorted(
                q.name for q in ed.glob("*.tmp"))
        # (also of a table this run does not write again)
        for piece in ("Adamson_predictions.csv.tmp",
                      "Adamson_finetuned_predictions_Full.csv.tmp"):
            (ed / piece).write_text("gene1_entrez,gene2")
        rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 0, text
        assert not list(ed.glob("*.tmp"))
        for name in names:
            assert _read(ed / name)[["gene1_entrez", "gene2_entrez"]].equals(
                rows[name]), name


def test_eval_rerun_clears_the_new_outputs_too():
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        rc, text = _eval(md, paths, root)
        assert rc == 0, text
        ed = md / "eval_finetuning_cv1"
        assert (ed / "Adamson_finetuned_allpairs_Full.csv").exists()
        assert (ed / "finetuned_models" / "Adamson_Full_fold0.pt").exists()
        # what a save killed part-way leaves behind
        (ed / "finetuned_models" / "Adamson_Full_fold0.pt.tmp").write_bytes(
            b"cut short")
        rc, text = _eval(md, paths, root, "--no_save_finetuned",
                         modes=("LP_last", ))
        assert rc == 0, text
        assert not (ed / "Adamson_finetuned_allpairs_Full.csv").exists()
        assert (ed / "Adamson_finetuned_allpairs_LP_last.csv").exists()
        assert (ed / "Adamson_sl_probabilities.csv").exists()
        assert not list((ed / "finetuned_models").iterdir())
        cfg = json.loads((ed / "config_used.json").read_text())
        assert cfg["finetuned_models_dir"] is None


def test_eval_withholds_the_probability_table_on_a_head_mismatch():
    real = ef.compute_all_head_logits

    def rolled(*a, **kw):   # every head's column shifted by one
        return np.roll(real(*a, **kw), 1, axis=1)

    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        with mock.patch.object(ef, "compute_all_head_logits", rolled):
            rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 1 and "NOT written" in text, text
        ed = md / "eval_finetuning_cv1"
        assert not (ed / "Adamson_sl_probabilities.csv").exists()
        # everything the tables and the paper read is complete
        for f in ("Adamson_predictions.csv", "zero_shot_summary.csv",
                  "Adamson_finetuned_predictions_LP_last.csv",
                  "Adamson_finetuned_allpairs_LP_last.csv",
                  "finetuned_summary.csv", "config_used.json"):
            assert (ed / f).exists(), f

        def not_finite(*a, **kw):   # a gap that cannot be measured
            return np.full_like(real(*a, **kw), np.nan)

        with mock.patch.object(ef, "compute_all_head_logits", not_finite):
            rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 1 and "NOT written" in text, text
        assert not (ed / "Adamson_sl_probabilities.csv").exists()
        assert (ed / "Adamson_predictions.csv").exists()


def test_eval_optional_outputs_never_cost_a_screen_its_tables():
    """The fine-tuned weights are saved inside the fold loop, before any
    table of the screen is on disk: a write error there (a full quota) must
    be reported, not raised."""
    mandatory = ("Adamson_predictions.csv", "zero_shot_metrics.csv",
                 "Adamson_finetuned_predictions_LP_last.csv",
                 "zero_shot_summary.csv", "finetuned_metrics.csv",
                 "finetuned_summary.csv", "gene_coverage.csv",
                 "config_used.json")

    def no_space(*a, **kw):
        raise OSError(28, "No space left on device")

    def no_table(*a, **kw):
        raise RuntimeError("HGNC table unavailable")

    real_to_csv = pd.DataFrame.to_csv

    def quota_on(marker):
        """to_csv failing as a full quota does, on the one file whose name
        holds `marker`: the file exists, truncated, when the write raises.
        Every other CSV is written as usual."""
        def to_csv(self, path_or_buf=None, *a, **kw):
            if marker in Path(str(path_or_buf)).name:
                Path(path_or_buf).write_text("gene1_entrez,gene2_ent")
                raise OSError(28, "No space left on device")
            return real_to_csv(self, path_or_buf, *a, **kw)

        return mock.patch.object(pd.DataFrame, "to_csv", to_csv)

    real_save = torch.save

    def save_dies(obj, f, *a, **kw):    # part of the archive, then the error
        Path(f).write_bytes(b"PK\x03\x04 half an archive")
        raise OSError(28, "No space left on device")

    def save_truncates(obj, f, *a, **kw):   # returns as if all were well
        real_save(obj, f, *a, **kw)
        data = Path(f).read_bytes()
        Path(f).write_bytes(data[:len(data) // 2])

    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        ed = md / "eval_finetuning_cv1"
        with mock.patch.object(ef, "save_finetuned_model", no_space):
            rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 1 and "No space left" in text, text
        assert text.count("NOT written") >= N_FOLDS, text
        for f in mandatory + ("Adamson_sl_probabilities.csv",
                              "Adamson_finetuned_allpairs_LP_last.csv"):
            assert (ed / f).exists(), f
        assert not list((ed / "finetuned_models").glob("*.pt"))
        with mock.patch.object(ef, "pair_symbols", no_table):
            rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 1 and "the all-pair tables NOT written" in text, text
        for f in mandatory:
            assert (ed / f).exists(), f
        assert not (ed / "Adamson_sl_probabilities.csv").exists()
        assert not (ed / "Adamson_finetuned_allpairs_LP_last.csv").exists()
        assert len(list((ed / "finetuned_models").glob("*.pt"))) == N_FOLDS
        with mock.patch.object(ef, "compute_all_head_logits", no_table):
            rc, text = _eval(md, paths, root, modes=("LP_last", ))
        assert rc == 1 and "Adamson_sl_probabilities.csv NOT written" in text
        for f in mandatory + ("Adamson_finetuned_allpairs_LP_last.csv", ):
            assert (ed / f).exists(), f
        assert not (ed / "Adamson_sl_probabilities.csv").exists()

        # The two all-pair CSVs are the large writes. A quota error half-way
        # through either one is reported, costs no table and not the other
        # CSV, and leaves no truncated file behind.
        for marker, lost, kept in (
                ("_sl_probabilities", "Adamson_sl_probabilities.csv",
                 "Adamson_finetuned_allpairs_LP_last.csv"),
                ("_finetuned_allpairs_",
                 "Adamson_finetuned_allpairs_LP_last.csv",
                 "Adamson_sl_probabilities.csv")):
            with quota_on(marker):
                rc, text = _eval(md, paths, root, modes=("LP_last", ))
            assert rc == 1 and f"{lost} NOT written" in text, text
            for f in mandatory + (kept, ):
                assert (ed / f).exists(), f
            assert not (ed / lost).exists(), lost

        # The model file itself: a save that dies part-way, or returns after
        # writing half the archive, leaves NOTHING under a model's name (the
        # export would read it), and costs no table.
        for fake, why in ((save_dies, "No space left"),
                          (save_truncates, "not written in full")):
            with mock.patch.object(torch, "save", fake):
                rc, text = _eval(md, paths, root, modes=("LP_last", ))
            assert rc == 1 and why in text, text
            assert text.count("NOT written") >= N_FOLDS, text
            assert not list((ed / "finetuned_models").iterdir()), why
            for f in mandatory + ("Adamson_sl_probabilities.csv",
                                  "Adamson_finetuned_allpairs_LP_last.csv"):
                assert (ed / f).exists(), (why, f)


def test_external_transfer_reads_an_eval_dir_with_the_new_outputs_unchanged():
    """sl_comparison/external_transfer.py builds the same tables from an eval
    dir with and without the all-pair outputs (it reads its three files by
    exact name, and refuses a fine-tuned CSV that is not the test rows)."""
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location(
        "_et_pair_predictions",
        os.path.join(os.path.dirname(_SIAMESE), "sl_comparison",
                     "external_transfer.py"))
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    with tempfile.TemporaryDirectory() as tmp, _mapper_patches()[0], \
            _mapper_patches()[1]:
        root = Path(tmp)
        md, paths = _model_dir(root)
        _screen(root, paths)
        (root / "neg.txt").write_text(f"{GENES[2]}\t{GENES[3]}\n")
        rc, text = _eval(md, paths, root, "--neg_path", str(root / "neg.txt"),
                         "--report_sldb_filtered")
        assert rc == 0, text
        folds = root / "folds"
        folds.mkdir()
        (folds / "genes.txt").write_text("\n".join(GENES[:25]) + "\n")

        def tables(out):
            _quiet(et.run, [("SLeuth", md)], [], False, folds / "genes.txt",
                   folds, out, "cv3", ["Adamson"], ["cv1"],
                   allow_unstamped=True)
            return {f.name: f.read_text() for f in sorted(out.glob("*.csv"))}

        with_new = tables(root / "out_new")
        assert {"zero_shot_per_fold.csv", "finetune_per_fold.csv",
                "row_sets.csv"} <= set(with_new), sorted(with_new)
        ed = md / "eval_finetuning_cv1"
        shutil.rmtree(ed / "finetuned_models")
        for f in list(ed.glob("*_sl_probabilities.csv")) + list(
                ed.glob("*_finetuned_allpairs_*.csv")):
            f.unlink()
        cfg = json.loads((ed / "config_used.json").read_text())
        for k in ("sl_probability_heads", "finetuned_models_dir"):
            cfg.pop(k)
        (ed / "config_used.json").write_text(json.dumps(cfg))
        assert tables(root / "out_old") == with_new


def test_compute_all_head_logits_spans_batches():
    """A real screen is many batches; the fixture's 435 pairs are one."""
    model = _make_model("per_head", 5).to(DEVICE)
    E = torch.randn(N_GENES, sum(DIMS),
                    generator=torch.Generator().manual_seed(5))
    i = np.repeat(np.arange(N_GENES), N_GENES)
    j = np.tile(np.arange(N_GENES), N_GENES)
    # 900 pairs in batches of 64: 14 full ones and a last one of 4
    got = ef.compute_all_head_logits(model, E, i, j, DEVICE, batch_size=64)
    with torch.no_grad():
        want = model(E[i].to(DEVICE), E[j].to(DEVICE)).cpu().numpy()
    assert got.shape == (N_GENES * N_GENES, len(HEADS))
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_head_probability_columns():
    logits = np.array([[[0.0, 2.0], [-1.0, -3.0]],
                       [[4.0, 0.0], [-1.0, 30.0]]])    # (folds, pairs, heads)
    cols = ef.head_probability_columns(logits, ["A", "B"])
    assert list(cols) == ["prob_A_fold0", "prob_A_fold1", "prob_A_mean",
                          "prob_B_fold0", "prob_B_fold1", "prob_B_mean",
                          "prob_ANY_fold0", "prob_ANY_fold1", "prob_ANY_mean"]
    s = _sigmoid
    assert np.allclose(cols["prob_A_mean"], [(s(0) + s(4)) / 2, s(-1)])
    assert np.allclose(cols["prob_ANY_fold0"], [s(2.0), s(-1.0)])
    # the fold mean of the max, not the max of the fold means
    assert np.allclose(cols["prob_ANY_mean"],
                       [(s(2.0) + s(4.0)) / 2, (s(-1.0) + s(30.0)) / 2])
    assert cols["prob_B_fold1"][1] < 1.0     # float64: not saturated at 30
    one = ef.head_probability_columns(logits[:, :, :1], ["ANY"])
    assert list(one) == ["prob_ANY_fold0", "prob_ANY_fold1", "prob_ANY_mean"]
    assert ef.head_labels(1) == ["ANY"] and ef.head_labels(8) == HEADS
    assert ef.head_labels(3) == ["head0", "head1", "head2"]


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
