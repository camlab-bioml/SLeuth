#!/usr/bin/env python3
"""Input guards of the siamese downstream stages (no data files, no GPU).

  eval_finetuning.py  refuses (exit 1, previous outputs untouched) a missing
                      DATASETS screen file, a DATASETS row whose --per_head
                      cell-line head is not a model head, and embedding files
                      other than the ones the checkpoints record (basenames in
                      order, gene_order), instead of skipping / scoring them;
                      its cv2/cv3 screen splits early-stop on a gene-held-out
                      val slice, cold-start in the same sense as their test;
                      its Scratch mode rebuilds the checkpoint's architecture
                      at the constructor's fresh init (nothing pretrained),
                      trains Full's set and is warm-started on its own logits.
  predict.py          refuses embedding files that are not the ones the
                      checkpoint records, including two same-width modalities
                      passed in the other order.
  zero_cell_emb.py    refuses an output dir that is the combo it ablates
                      (its up-front clear would delete the checkpoints), and
                      carries the source combo's training seed / selection
                      metric onto the ablated row it grades.
  grade.sh            once a CV's matrix dir exists, a previous row is moved
                      to stale_siamese/ before grading, so a partial set or a
                      failed grading leaves the row missing; exit 0 only when
                      every requested CV graded (stub evaluator).
  run_ablation_summary.sh
                      reads a leave-one-category-out AUPRG delta beyond 0.005
                      as HARMFUL / contributes only when it also exceeds 2 SE
                      of the paired (ablation fold k - reference fold k)
                      deltas, else as within fold noise, n/a below 2 paired
                      folds; the CSV carries the paired sd, SE and n_paired
                      (synthetic results tree).

    $PYTHON_PATH -m pytest siamese_sl/tests/test_downstream_guards.py -q
    $PYTHON_PATH siamese_sl/tests/test_downstream_guards.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

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

import eval_finetuning as ef  # noqa: E402

GENES = ["101", "102", "103", "104"]


def _write_emb(path: Path, genes, dim=3, offset=0.0):
    torch.save({"embeddings": offset + torch.arange(len(genes) * dim,
                                                    dtype=torch.float32).reshape(
                                                        len(genes), dim),
                "gene_order": list(genes)}, path)


def _model_dir(root: Path, gene_order, emb_names, num_heads=None):
    """A combo dir with just what main() reads before its first refusal.

    num_heads: also give each checkpoint a state dict main() can read the head
    count off (checkpoint_num_heads), so the --per_head path can be driven.
    """
    md = root / "combo_cv3"
    (md / "checkpoints").mkdir(parents=True)
    (md / "config.json").write_text(json.dumps({"num_folds": 2}))
    for k in range(2):
        ck = {"input_dim": 3, "gene_order": list(gene_order),
              "embeddings_paths": [f"../data/{n}" for n in emb_names]}
        if num_heads is not None:
            ck["model_state_dict"] = {
                "cell_emb.weight": torch.zeros(num_heads, 24)}
        torch.save(ck, md / "checkpoints" / f"fold_{k}_best.pt")
    old = md / "eval_finetuning_cv3"
    old.mkdir()
    (old / "Adamson_predictions.csv").write_text("OLD")
    return md


def _load_zero_cell_emb():
    """cell_line_ablation/zero_cell_emb.py by path (it prepends sl_comparison/
    to sys.path, which must not survive into the rest of the suite)."""
    path_before = list(sys.path)
    spec = importlib.util.spec_from_file_location(
        "_zero_cell_emb_guard",
        os.path.join(_SIAMESE, "cell_line_ablation", "zero_cell_emb.py"))
    zce = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(zce)
    finally:
        sys.path[:] = path_before
    return zce


def _run_main(argv):
    with mock.patch.object(sys, "argv", ["eval_finetuning.py"] + argv):
        return ef.main()


def test_dataset_input_problems():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "a.tsv").write_text("x")
        specs = ef.parse_datasets_spec(["Adamson:a.tsv:Gamma:-1:tsv:K562",
                                        "Corn:missing.csv:s:-1:csv:OTHER",
                                        "Nope:a.tsv:s:-1:csv"])
        got = ef.dataset_input_problems(specs, base)
        assert len(got) == 2, got
        assert got[0].startswith("Corn: file not found"), got
        assert got[1].startswith("Nope: no loader"), got
        assert ef.dataset_input_problems(specs[:1], base) == []


def test_checkpoint_input_problems():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = root / "all_genes_a.pt", root / "all_genes_b.pt"
        _write_emb(a, GENES, dim=3)
        _write_emb(b, GENES, dim=3, offset=7.0)
        paths = [str(a), str(b)]
        shas = [ef._file_sha256(p) for p in paths]
        ok = [(Path("f0"), GENES, ["../data/all_genes_a.pt",
                                   "../data/all_genes_b.pt"], shas)]
        assert ef.checkpoint_input_problems(ok, paths, GENES) == []
        # legacy checkpoints without the keys are not refused
        assert ef.checkpoint_input_problems([(Path("f0"), None, None, None)],
                                            paths, GENES) == []
        swapped = [(Path("f0"), GENES,
                    ["x/all_genes_b.pt", "x/all_genes_a.pt"], None)]
        got = ef.checkpoint_input_problems(swapped, paths, GENES)
        assert len(got) == 1 and "in the same order" in got[0], got
        other = [(Path("f1"), GENES[:3] + ["999"], None, None)]
        got = ef.checkpoint_input_problems(other, paths, GENES)
        assert len(got) == 1 and "gene_order" in got[0], got


def test_checkpoint_input_problems_sees_a_regenerated_file():
    """A same-path rebuild keeps the basenames AND the gene order.

    run_generate_embeddings.sh deletes and rebuilds every .pt at the same path
    over the same gene universe, and node2vec / go2vec / onto2vec are not
    bit-for-bit reproducible, so only the CONTENT hash train.py records can
    tell the rebuilt vectors from the trained ones.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first, second = root / "v1" / "all_genes_a.pt", root / "v2" / "all_genes_a.pt"
        first.parent.mkdir()
        second.parent.mkdir()
        _write_emb(first, GENES, dim=3)
        _write_emb(second, GENES, dim=3, offset=1.0)  # same name + gene_order
        trained = [ef._file_sha256(first)]
        paths = [str(second)]
        # legacy (no hash recorded) stays loadable
        assert ef.checkpoint_input_problems(
            [(Path("f0"), GENES, ["../data/all_genes_a.pt"], None)],
            paths, GENES) == []
        got = ef.checkpoint_input_problems(
            [(Path("f0"), GENES, ["../data/all_genes_a.pt"], trained)],
            paths, GENES)
        assert len(got) == 1 and "CONTENTS" in got[0], got
        # the files it WAS trained on pass
        assert ef.checkpoint_input_problems(
            [(Path("f0"), GENES, ["../data/all_genes_a.pt"], trained)],
            [str(first)], GENES) == []


def test_dataset_input_problems_require_head():
    """The --per_head head check runs in the SAME pre-clear gate as the files.

    run_for_dataset raises on a bad head too, but only once that dataset's turn
    comes: after the output clear and after the earlier screens have spent
    their GPU time.
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "a.tsv").write_text("x")
        good = ef.parse_datasets_spec(["Adamson:a.tsv:Gamma:-1:tsv:K562",
                                       "Corn:a.tsv:s:-1:csv:OTHER"])
        assert ef.dataset_input_problems(good, base, require_head=True) == []
        # a lower-cased head and a 5-field row (no head at all)
        bad = ef.parse_datasets_spec(["Adamson:a.tsv:Gamma:-1:tsv:k562",
                                      "Corn:a.tsv:s:-1:csv"])
        assert ef.dataset_input_problems(bad, base) == []  # off by default
        got = ef.dataset_input_problems(bad, base, require_head=True)
        assert len(got) == 2 and all("6th ':HEAD'" in g for g in got), got


def test_cv2_cv3_screen_val_is_gene_held_out():
    """cv2/cv3 pick the fine-tuning checkpoint in the regime they are scored in.

    A val slice drawn at random from the train+val PAIRS would have both genes
    seen in training while every test pair carries an unseen gene, so the
    stopping epoch would be chosen under a regime the test split never shares.
    sl_comparison/prepare_folds carves the shared SLDB cv2/cv3 val by the same
    rule and its verify_fold asserts it.
    """
    import numpy as np
    pairs = np.array([(i, j) for i in range(60) for j in range(i + 1, 60)])
    i1, i2 = pairs[:, 0], pairs[:, 1]
    for fn in (ef.split_pairs_cv2, ef.split_pairs_cv3):
        tr, va, te = fn(i1, i2, 0.8, 0.1, 42)
        assert tr.sum() and va.sum() and te.sum(), fn.__name__
        assert not (tr & va).any() and not (tr & te).any() \
            and not (va & te).any(), fn.__name__
        fit = set(i1[tr].tolist()) | set(i2[tr].tolist())
        for mask, what in ((va, "val"), (te, "test")):
            assert not any(int(a) in fit and int(b) in fit
                           for a, b in zip(i1[mask], i2[mask])), \
                f"{fn.__name__}: a {what} pair sits entirely inside the " \
                f"genes the fine-tuning saw"
    # cv3 is stricter still: BOTH genes of a val pair are held out, as they
    # are in its test pairs (pairs with exactly one val gene are discarded).
    tr, va, te = ef.split_pairs_cv3(i1, i2, 0.8, 0.1, 42)
    fit = set(i1[tr].tolist()) | set(i2[tr].tolist())
    for mask in (va, te):
        assert not (set(i1[mask].tolist()) | set(i2[mask].tolist())) & fit
    assert int(tr.sum()) + int(va.sum()) + int(te.sum()) < len(i1)
    # cv1 is a pair split in both halves and discards nothing
    tr, va, te = ef.split_pairs(len(i1), 0.8, 0.1, 42)
    assert int(tr.sum()) + int(va.sum()) + int(te.sum()) == len(i1)


def test_main_refuses_missing_screen_and_keeps_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        md = _model_dir(root, GENES, ["all_genes_a.pt"])
        _write_emb(root / "all_genes_a.pt", GENES)
        rc = _run_main([
            "--model_dir", str(md), "--embeddings_paths",
            str(root / "all_genes_a.pt"), "--datasets",
            "Adamson:external data/missing.txt:Gamma:-1:tsv:K562",
            "--sl_path", str(root / "sl.txt"), "--base_dir", str(root),
            "--cv_type", "cv3", "--device", "cpu"])
        assert rc == 1
        assert (md / "eval_finetuning_cv3" /
                "Adamson_predictions.csv").read_text() == "OLD"
        assert not (md / "eval_finetuning_cv3" / "config_used.json").exists()


def test_main_refuses_other_embedding_files_and_keeps_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "screen.txt").write_text("x")
        spec = "Adamson:screen.txt:Gamma:-1:tsv:K562"
        common = ["--datasets", spec, "--sl_path", str(root / "sl.txt"),
                  "--base_dir", str(root), "--cv_type", "cv3",
                  "--device", "cpu"]
        # same files, but regenerated over another gene universe
        md = _model_dir(root, GENES, ["all_genes_a.pt"])
        _write_emb(root / "all_genes_a.pt", GENES[:3] + ["999"])
        rc = _run_main(["--model_dir", str(md), "--embeddings_paths",
                        str(root / "all_genes_a.pt")] + common)
        assert rc == 1
        assert (md / "eval_finetuning_cv3" /
                "Adamson_predictions.csv").read_text() == "OLD"
        # right universe, but two modalities passed in swapped order
        shutil.rmtree(md)
        md = _model_dir(root, GENES, ["all_genes_a.pt", "all_genes_b.pt"])
        _write_emb(root / "all_genes_a.pt", GENES)
        _write_emb(root / "all_genes_b.pt", GENES)
        rc = _run_main(["--model_dir", str(md), "--embeddings_paths",
                        str(root / "all_genes_b.pt"),
                        str(root / "all_genes_a.pt")] + common)
        assert rc == 1
        assert (md / "eval_finetuning_cv3" /
                "Adamson_predictions.csv").read_text() == "OLD"


def test_main_refuses_a_bad_per_head_dataset_head_and_keeps_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "screen.txt").write_text("x")
        md = _model_dir(root, GENES, ["all_genes_a.pt"], num_heads=8)
        _write_emb(root / "all_genes_a.pt", GENES)
        argv = ["--model_dir", str(md), "--embeddings_paths",
                str(root / "all_genes_a.pt"), "--sl_path", str(root / "sl.txt"),
                "--base_dir", str(root), "--cv_type", "cv3", "--device", "cpu",
                "--per_head"]
        for spec in ("Adamson:screen.txt:Gamma:-1:tsv:k562",   # wrong case
                     "Adamson:screen.txt:Gamma:-1:tsv"):       # no head field
            rc = _run_main(argv + ["--datasets", spec])
            assert rc == 1, spec
            assert (md / "eval_finetuning_cv3" /
                    "Adamson_predictions.csv").read_text() == "OLD", spec
            assert not (md / "eval_finetuning_cv3" /
                        "config_used.json").exists(), spec


def test_predict_refuses_embedding_files_in_the_wrong_order():
    """Two same-width modalities passed in the other order would each be pushed
    through the OTHER's fitted impute / PCA / normalise statistics. Neither the
    gene-order check (every catalog file is generated over one gene universe,
    so all modalities share an order) nor apply_multimodal_transform's
    per-modality width check can see it. The trained order is the control, so
    the ValueError can only come from the basename-order guard.
    """
    import predict
    from data_loader import fit_multimodal_transform, load_raw_multimodal
    from siamese_esm import SiameseSL

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = root / "all_genes_a.pt", root / "all_genes_b.pt"
        _write_emb(a, GENES, dim=5)
        _write_emb(b, GENES, dim=5, offset=7.0)  # same width, other values
        raw, _, idx_to_gene = load_raw_multimodal([str(a), str(b)])
        transform = fit_multimodal_transform(
            raw, torch.ones(len(GENES), dtype=torch.bool),
            pca_dims=None, post_pca=None, pca_method="plain")
        model = SiameseSL(input_dim=10, encoder_dims=[4, 3])
        ckpt = root / "fold_0_best.pt"
        torch.save({"model_state_dict": model.state_dict(),
                    "config": {"pd_epsilon": 0.001, "dropout": 0.2},
                    "preprocessing_transform": transform,
                    "gene_order": [idx_to_gene[i]
                                   for i in range(len(idx_to_gene))],
                    "embeddings_paths": [f"../data/{p.name}" for p in (a, b)]},
                   ckpt)

        def _predict(paths):
            argv = ["predict.py", "--model", str(ckpt), "--embeddings_paths"]
            argv += [str(p) for p in paths]
            argv += ["--gene1", GENES[0], "--gene2", GENES[1]]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    predict.main()

        _predict([a, b])  # control: the trained order
        raised = ""
        try:
            _predict([b, a])
        except ValueError as e:
            raised = str(e)
        assert "in that order" in raised, \
            f"swapped same-width modalities must be refused; got {raised!r}"


def test_zero_cell_emb_carries_the_combo_training_params():
    """The ablated row is not retrained, so its seed / selection metric ARE
    the source combo's; without them compare.py's seed report and
    paper_tables' item_inputs (D3) file every <row>_nocell row as UNRECORDED.
    """
    zce = _load_zero_cell_emb()
    with tempfile.TemporaryDirectory() as tmp:
        combo = Path(tmp) / "best_cat_X_cv3"
        combo.mkdir()
        (combo / "model_params.json").write_text(json.dumps({
            "trainable_params": 5, "fold_stamp": {"genes.txt": "a"},
            "selection_regime": "val", "selection_metric": "auprg_any",
            "seed": 42}))
        assert zce.carried_training_params(combo) == {
            "selection_regime": "val", "selection_metric": "auprg_any",
            "seed": 42}
        # A combo that records neither contributes no null keys, so the
        # ablated row's model_params.json looks exactly as it used to.
        (combo / "model_params.json").write_text("{}")
        assert zce.carried_training_params(combo) == {"selection_regime": "val"}


def test_zero_cell_emb_refuses_to_clear_the_source_combo():
    zce = _load_zero_cell_emb()
    with tempfile.TemporaryDirectory() as tmp:
        combo = Path(tmp) / "results" / "best_cat_X_cv3"
        (combo / "checkpoints").mkdir(parents=True)
        ck = combo / "checkpoints" / "fold_0_best.pt"
        ck.write_bytes(b"weights")
        (combo / "fold_0_cv3_predictions.npy").write_bytes(b"m")
        try:
            zce.run_cv("cv3", combo, {}, [], out_dir=Path(tmp) / "results",
                       cv_dir_name="best_cat_X_%s")
        except SystemExit as e:
            assert "source combo" in str(e), e
        else:
            raise AssertionError("run_cv cleared its own source combo")
        assert ck.read_bytes() == b"weights"
        assert (combo / "fold_0_cv3_predictions.npy").exists()


def test_zero_cell_emb_clears_what_a_previous_run_left():
    zce = _load_zero_cell_emb()
    with tempfile.TemporaryDirectory() as tmp:
        mdir = Path(tmp) / "no_cellline_cv3"
        for d in ("checkpoints", "eval_finetuning_cv1", "pair_predictions"):
            (mdir / d).mkdir(parents=True)
        gone = [mdir / "checkpoints" / "fold_0_best.pt",
                mdir / "fold_0_cv3_predictions.npy", mdir / "config.json",
                mdir / "results.json", mdir / "model_params.json",
                mdir / "eval_finetuning_cv1" / "Adamson_predictions.csv",
                mdir / "pair_predictions" / "manifest.json"]
        kept = [mdir / "fold_0_cv2_predictions.npy", mdir / "notes.txt"]
        for p in gone + kept:
            p.write_bytes(b"x")
        zce.clear_previous_outputs(mdir, "cv3")
        assert not [p for p in gone if p.exists()]
        assert not (mdir / "pair_predictions").exists()
        assert not (mdir / "eval_finetuning_cv1").exists()
        assert [p for p in kept if p.exists()] == kept


def test_zero_cell_emb_one_cvs_hard_stop_costs_no_other_cv():
    """main() clears EVERY requested CV's previous outputs before any CV
    runs, and goes on after a CV that stops hard (a stale fold stamp, an
    unreadable checkpoint, an OOM): the CVs after it are still ablated,
    nothing gradeable of the failed one is left, and the exit status is 1.
    Before, the failing CV ended the invocation, and the CVs it never
    reached kept a previous run's matrices and fold stamp for grade.sh."""
    zce = _load_zero_cell_emb()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "models"
        for cv in ("cv1", "cv2", "cv3"):
            d = out / f"no_cellline_{cv}"
            d.mkdir(parents=True)
            for n in (f"fold_0_{cv}_predictions.npy", "model_params.json",
                      "results.json"):
                (d / n).write_bytes(b"previous run")

        def files():
            return sorted(q.relative_to(out).as_posix()
                          for q in out.rglob("*") if q.is_file())

        seen = []

        def run_cv(cv, model_dir, raw_cache, shared_genes, out_dir=None,
                   cv_dir_name="no_cellline_%s", **kw):
            seen.append((cv, files()))
            d = Path(out_dir) / (cv_dir_name % cv)
            d.mkdir(parents=True, exist_ok=True)
            if cv == "cv1":    # stops after it began to write
                (d / "fold_0_cv1_predictions.npy").write_bytes(b"half")
                (d / "model_params.json").write_text("{}")
                raise SystemExit("ERROR: the combo's fold stamp differs")
            if cv == "cv2":
                raise RuntimeError("CUDA out of memory")
            (d / "fold_0_cv3_predictions.npy").write_bytes(b"new")
            return {"cv_type": cv, "status": "ok"}

        def main(*extra):
            argv = ["zero_cell_emb.py", "--out_dir", str(out),
                    "--save_matrices", "--device", "cpu", "--json_out",
                    str(Path(tmp) / "r.json"), *extra]
            with mock.patch.object(zce, "run_cv", run_cv), \
                    mock.patch.object(zce, "load_shared_genes", lambda: []), \
                    mock.patch.object(zce, "print_report", lambda r: None), \
                    mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                return zce.main()

        assert main() == 1
        assert [cv for cv, _ in seen] == ["cv1", "cv2", "cv3"], seen
        assert seen[0][1] == [], seen[0]     # all cleared before the first
        assert files() == ["no_cellline_cv3/fold_0_cv3_predictions.npy"]
        res = json.loads((Path(tmp) / "r.json").read_text())
        assert {cv: r["status"] for cv, r in res.items()} == {
            "cv1": "error", "cv2": "error", "cv3": "ok"}, res
        assert "fold stamp differs" in res["cv1"]["error"] \
            and "out of memory" in res["cv2"]["error"], res
        # the report takes a failed CV without its numbers
        with contextlib.redirect_stdout(io.StringIO()):
            zce.print_report({"cv1": res["cv1"]})
        # an out dir that IS a requested CV's source combo: refused before
        # anything of ANY CV is cleared
        before = files()
        try:
            main("--model_dir", f"cv3={out / 'no_cellline_cv3'}")
        except SystemExit as e:
            assert "source combo" in str(e), e
        else:
            raise AssertionError("cleared a source combo")
        assert files() == before


def test_zero_cell_emb_refuses_embedding_files_rebuilt_in_place():
    """The ablation loads data/<basename> of the files the combo trained on.
    Rebuilt at the same path over the same genes, they pass the basename and
    gene-order checks; only the recorded content hash tells. Every fold
    checkpoint is compared, and a checkpoint that predates the record is not
    refused."""
    import eval_finetuning as ef
    zce = _load_zero_cell_emb()
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        data.mkdir()
        _write_emb(data / "all_genes_a.pt", GENES)
        _write_emb(data / "all_genes_b.pt", GENES, offset=3.0)
        paths = ["../data/all_genes_a.pt", "../data/all_genes_b.pt"]
        shas = [ef._file_sha256(data / Path(p).name) for p in paths]

        def ckpt(k, sha):
            p = Path(tmp) / f"fold_{k}_best.pt"
            d = {"gene_order": list(GENES), "embeddings_paths": paths}
            if sha is not None:
                d["embeddings_sha256"] = sha
            torch.save(d, p)
            return p

        ckpts = [ckpt(0, shas), ckpt(1, shas)]

        def problems():
            first = torch.load(ckpts[0], map_location="cpu",
                               weights_only=False)
            return zce.embedding_content_problems(ckpts, first, list(GENES),
                                                  data)

        assert problems() == []
        _write_emb(data / "all_genes_b.pt", GENES, offset=3.5)   # rebuilt
        got = problems()
        assert len(got) == 2 and all("CONTENTS differ" in g for g in got), got
        # only the SECOND fold's checkpoint carries the record
        ckpts = [ckpt(0, None), ckpt(1, shas)]
        got = problems()
        assert len(got) == 1 and "fold_1_best.pt" in got[0], got
        # legacy checkpoints (no record): nothing to compare, not refused
        ckpts = [ckpt(0, None), ckpt(1, None)]
        assert problems() == []


def test_zero_cell_emb_stops_a_cv_whose_embedding_files_were_rebuilt():
    """run_cv ACTS on embedding_content_problems (the helper alone is tested
    above): a CV whose embedding files are not the ones its combo trained on
    stops before anything is scored; with matching files it goes on."""
    zce = _load_zero_cell_emb()

    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached()

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "combo_cv3"
        (model_dir / "checkpoints").mkdir(parents=True)
        torch.save({"model_state_dict": {}, "gene_order": list(GENES),
                    "embeddings_paths": ["../data/all_genes_a.pt"],
                    "config": {}},
                   model_dir / "checkpoints" / "fold_0_best.pt")

        def run(problems):
            with mock.patch.object(zce, "not_applicable_reason",
                                   lambda sd: None), \
                    mock.patch.object(zce, "require_combo_fold_stamp",
                                      lambda *a, **k: None), \
                    mock.patch.object(zce, "resolve_raw",
                                      lambda *a, **k: (None, {},
                                                       list(GENES))), \
                    mock.patch.object(zce, "embedding_content_problems",
                                      lambda *a, **k: problems), \
                    mock.patch.object(zce, "combo_fold_reference", reached), \
                    contextlib.redirect_stdout(io.StringIO()):
                zce.run_cv("cv3", model_dir, {}, list(GENES))

        try:
            run(["all_genes_a.pt: its CONTENTS differ from the checkpoint's"])
        except SystemExit as e:
            assert "are not the ones" in str(e) \
                and "CONTENTS differ" in str(e), e
        else:
            raise AssertionError("scored a combo on rebuilt embedding files")
        try:
            run([])
        except Reached:
            pass
        else:
            raise AssertionError("did not go on past the content check")


_STUB_EVAL = """#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in --out) OUT="$2"; shift;; --cv_type) CV="$2"; shift;; esac
  shift
done
case " ${FAIL_CVS:-} " in *" $CV "*) exit 1;; esac
echo NEW > "$OUT"
"""


def test_grade_sh_leaves_no_earlier_grading_next_to_new_matrices():
    if shutil.which("bash") is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        abl = root / "siamese_sl" / "cell_line_ablation"
        abl.mkdir(parents=True)
        shutil.copy(os.path.join(_SIAMESE, "cell_line_ablation", "grade.sh"),
                    abl / "grade.sh")
        for cv in ("cv1", "cv2", "cv3", "cv4"):
            for k in range(3):
                (root / "sl_comparison" / "folds" / cv /
                 f"fold_{k}").mkdir(parents=True)
        res = root / "sl_comparison" / "results"
        res.mkdir()
        stub = Path(tmp) / "stub_eval"
        stub.write_text(_STUB_EVAL)
        stub.chmod(0o755)
        name = "siamese_sl_manual_nocell"
        for cv in ("cv1", "cv2", "cv3", "cv4"):
            (res / f"{name}_{cv}.json").write_text("OLD")
        (res / "siamese_sl_nocell_cv1.json").write_text("PIPELINE")
        models = abl / "models"
        for cv in ("cv1", "cv3"):            # complete sets
            d = models / f"no_cellline_{cv}"
            d.mkdir(parents=True)
            for k in range(3):
                (d / f"fold_{k}_{cv}_predictions.npy").write_bytes(b"")
        (models / "no_cellline_cv2").mkdir()  # cleared by zero_cell_emb.py
        env = dict(os.environ, PYTHON_PATH=str(stub), FAIL_CVS="cv3",
                   GRADE_CVS="cv1 cv2 cv3 cv4")
        env.pop("MODEL_NAME", None)
        env.pop("PRED_DIR_TEMPLATE", None)
        r = subprocess.run(["bash", str(abl / "grade.sh")], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 1, r.stdout + r.stderr
        assert (res / f"{name}_cv1.json").read_text().strip() == "NEW"
        assert not (res / f"{name}_cv2.json").exists()   # partial set
        assert not (res / f"{name}_cv3.json").exists()   # failed grading
        assert (res / f"{name}_cv4.json").read_text() == "OLD"  # no dir
        assert (res / "siamese_sl_nocell_cv1.json").read_text() == "PIPELINE"
        stale = sorted(p.name.split(".")[0]
                       for p in (res / "stale_siamese").iterdir())
        assert stale == [f"{name}_cv1", f"{name}_cv2", f"{name}_cv3"], stale
        env["GRADE_CVS"] = "cv1"
        r = subprocess.run(["bash", str(abl / "grade.sh")], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


_REF_AUPRG = [0.70, 0.60, 0.80, 0.65, 0.75]
# category -> (winner type, auprg_any of ablation fold k minus reference fold k)
_LOCO = {
    "expression": ("geneformer", [0.036, -0.024, 0.046, -0.034, 0.006]),
    "go": ("go2vec", [0.019, 0.021, 0.020, 0.018, 0.022]),
    "kg": ("kg_complex", [-0.019, -0.021, -0.020, -0.018, -0.022]),
    "ppi": ("mashup", [0.02] * 5),          # results.json without fold ids
    "protein_seq": ("esm1b", [0.02] * 5),   # another submission's run_token
    "text": ("text_embed", [0.001, -0.001, 0.002, -0.002, 0.0]),
}


def _summary_run(d: Path, emb_files, auprg, fold_ids=True, token=None):
    """results.json / model_params.json (/ run_token.txt) of one train.py run,
    as far as run_ablation_summary.sh reads them."""
    d.mkdir(parents=True)
    fm = [dict({"fold": k} if fold_ids else {}, auprg_any=v, aupr_any=v,
               auroc_any=v) for k, v in enumerate(auprg)]
    (d / "results.json").write_text(json.dumps({
        "config": {"embeddings_paths": [f"../data/{f}" for f in emb_files]},
        "fold_metrics": fm}))
    (d / "model_params.json").write_text(json.dumps(
        {"fold_stamp": {"folds_dir": "../sl_comparison/folds", "a": "1"}}))
    if token is not None:
        (d / "run_token.txt").write_text(token + "\n")


def test_ablation_summary_calls_a_direction_only_beyond_2_se():
    """The 2026-09-23 run labelled 12 rows HARMFUL on a mean delta above
    +0.005 alone (e.g. +0.0058) that the fold-to-fold spread did not support.
    Both runs train on the same shared folds, so the SE is taken over the
    paired deltas (ablation fold k minus reference fold k)."""
    if shutil.which("bash") is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        sia = Path(tmp) / "siamese_sl"
        (sia / "slurm").mkdir(parents=True)
        for f in ("run_ablation_summary.sh", "run_best_per_category.conf",
                  "run_ablation.conf"):
            shutil.copy(os.path.join(_SIAMESE, "slurm", f), sia / "slurm")
        shutil.copy(os.path.join(_SIAMESE, "arch_signature.py"), sia)
        shutil.copy(os.path.join(_SIAMESE, "fold_selection.py"), sia)
        # the real config.conf; its PYTHON_PATH is the cluster venv
        (sia / "slurm" / "config.conf").write_text(
            f'source "{os.path.join(_SIAMESE, "slurm", "config.conf")}"\n'
            f'PYTHON_PATH="{sys.executable}"\n')
        res = sia / "results"
        winners = {cat: {"type": t, "file": f"all_genes_{t}.pt"}
                   for cat, (t, _) in _LOCO.items()}
        files = [w["file"] for w in winners.values()]
        _summary_run(res / "best_cat_all_cv1", files, _REF_AUPRG)
        (res / "best_per_category.json").write_text(json.dumps(
            {"cv1": winners}))
        (res / "latest_combo_cv1.txt").write_text("results/best_cat_all_cv1\n")
        for cat, (_, d) in _LOCO.items():
            _summary_run(res / f"ablate_no_{cat}_cv1",
                         [f for f in files if f != winners[cat]["file"]],
                         [a + b for a, b in zip(_REF_AUPRG, d)],
                         fold_ids=cat != "ppi",
                         token="other" if cat == "protein_seq" else "tok-1")
        env = {k: v for k, v in os.environ.items()
               if k not in ("SLURM_JOB_ID", "SIAMESE_VARIANTS",
                            "SELECTION_VARIANT", "BESTCAT_EXTRA_VARIANTS",
                            "BEST_CAT_PCA_VARIANCE")}
        env.update(SLURM_SUBMIT_DIR=str(sia), RUN_TOKEN="tok-1",
                   SIAMESE_VARIANTS_LIST="pd:per_head")
        r = subprocess.run(["bash", str(sia / "slurm" /
                                        "run_ablation_summary.sh")],
                           env=env, capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        with open(res / "ablation_summary.csv") as f:
            rd = csv.DictReader(f)
            rows = {row["dropped"]: row for row in rd}
            cols = rd.fieldnames
        new = ["delta_auprg_any_paired_sd", "delta_auprg_any_se", "n_paired"]
        i = cols.index("delta_auroc_any")
        assert cols[i + 1:i + 4] == new, cols
        want = {"expression": "within fold noise (|Δ| <= 2 SE)",
                "go": "HARMFUL — removing it helps",
                "kg": "contributes",
                "ppi": "n/a (fewer than 2 paired folds)",
                "text": "no measurable effect"}
        for cat, v in want.items():
            assert rows[cat]["verdict"] == v, (cat, rows[cat]["verdict"])
        for cat in ("expression", "go", "kg", "text"):
            d, row = _LOCO[cat][1], rows[cat]
            assert row["n_paired"] == "5", (cat, row)
            for col, x in (("delta_auprg_any", statistics.mean(d)),
                           ("delta_auprg_any_paired_sd", statistics.pstdev(d)),
                           ("delta_auprg_any_se",
                            statistics.stdev(d) / math.sqrt(len(d)))):
                assert abs(float(row[col]) - x) < 1e-9, (cat, col, row[col], x)
        # +0.006 clears the 0.005 cut the old rule used alone, not 2 SE
        e = rows["expression"]
        assert 0.005 < float(e["delta_auprg_any"]) \
            < 2 * float(e["delta_auprg_any_se"]), e
        # a delta with no fold ids to pair it on
        p = rows["ppi"]
        assert float(p["delta_auprg_any"]) > 0.005 and p["n_paired"] == "0" \
            and p["delta_auprg_any_se"] == "", p
        # no delta (another submission's run, the reference), no spread
        m = rows["protein_seq"]
        assert m["verdict"].startswith("MISMATCH"), m["verdict"]
        for row in (m, rows["none"]):
            assert row["delta_auprg_any"] == "" and \
                all(row[c] == "" for c in new), row
        # the printed row shows the SE next to the AUPRG delta
        g = _LOCO["go"][1]
        line = next(ln for ln in r.stdout.splitlines()
                    if ln.startswith("  go "))
        se = statistics.stdev(g) / math.sqrt(len(g))
        assert f"{statistics.mean(g):>+9.4f}{se:>9.4f}" in line, line


def test_tail_metrics_are_nan_when_ties_fill_the_tail():
    # Corn's sens.score is exactly 0 for about half the pairs, so the "bottom
    # 1%" of its negated phenotype was half the screen (AUPR@bot1% = the
    # prevalence). Such a tail is NaN now; a continuous target keeps its tails.
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2000
    target = np.where(rng.random(n) < 0.49, 0.0, rng.exponential(1.0, n))
    pred = target + rng.normal(0, 0.5, n)
    m = ef.compute_metrics(pred, target)
    assert np.isnan(m["auroc_bot1pct"]) and np.isnan(m["aupr_bot1pct"]), m
    assert np.isfinite(m["auroc_top1pct"]) and np.isfinite(m["aupr_top1pct"]), m
    cont = rng.normal(0, 1, n)
    m = ef.compute_metrics(cont + rng.normal(0, 0.5, n), cont)
    assert all(np.isfinite(m[k]) for k in ("auroc_bot1pct", "aupr_bot1pct",
                                           "auroc_top1pct", "aupr_top1pct")), m


def test_finetune_keeps_the_warm_start_when_training_only_hurts():
    # The warm start (OLS-calibrated, un-adapted) competes as epoch 0: when
    # every fine-tuning epoch is worse on val, it is what comes back
    # (best_epoch 0). Until 2026-10-02 epoch 1 always replaced it.
    import numpy as np
    from siamese_esm import SiameseSL
    torch.manual_seed(0)
    n_genes, dim, n_pairs = 40, 6, 400
    emb = torch.randn(n_genes, dim)
    base = SiameseSL(input_dim=dim, encoder_dims=[8, 8], dropout=0.0)
    model = ef.FinetunedSL(base)
    rng = np.random.default_rng(1)
    idx1 = rng.integers(0, n_genes, n_pairs)
    idx2 = rng.integers(0, n_genes, n_pairs)
    with torch.no_grad():
        logit = base(emb[torch.as_tensor(idx1)],
                     emb[torch.as_tensor(idx2)]).squeeze(-1).numpy()
    a, c = 2.0, 0.5
    target = a * logit + c          # the warm start fits val exactly (R² = 1)
    train_mask = np.zeros(n_pairs, bool)
    train_mask[:300] = True
    val_mask = ~train_mask
    with torch.no_grad():
        before = (a * logit + c)[val_mask]
    m, best_epoch = ef.finetune_one(
        model, emb, idx1, idx2, target + rng.normal(0, 5.0, n_pairs) * train_mask,
        train_mask, val_mask, mode="Full", epochs=3, patience=3, lr=0.5,
        batch_size=64, weight_decay=0.0, device="cpu", warm_start=(a, c),
        seed=0)
    assert best_epoch == 0, best_epoch
    m.eval()
    with torch.no_grad():
        after = m(emb[torch.as_tensor(idx1[val_mask])],
                  emb[torch.as_tensor(idx2[val_mask])]).numpy()
    assert np.allclose(after, before, atol=1e-5), np.abs(after - before).max()


def test_scratch_mode_rebuilds_the_architecture_at_its_fresh_init():
    """Scratch: _load_eval_base(init_seed=s) is the checkpoint's architecture
    (every tensor, same shape) at exactly the constructor's initialisation
    under torch.manual_seed(s), with nothing of the checkpoint in it, for a
    cell-line and a single-output checkpoint alike; deterministic per seed;
    the caller's RNG untouched; and it trains Full's parameter set."""
    import argparse
    from siamese_esm import SiameseSL, SiameseSLMultiCell

    mc_kw = dict(input_dim=6, num_heads=8, encoder_dims=[4, 4, 3],
                 cell_line_dim=2, use_layernorm=True, raw_temperature=True,
                 per_head_temperature=True, per_head_last_layer_bias=True,
                 per_head_input_bias=True, kernel="bilinear", gene_term=True,
                 pair_features=True, inner_norm="batch", dropout=0.2,
                 pd_epsilon=0.001)
    single_kw = dict(input_dim=6, encoder_dims=[4, 4, 3], dropout=0.2,
                     pd_epsilon=0.001)
    for per_head, cls, kw in ((True, SiameseSLMultiCell, mc_kw),
                              (False, SiameseSL, single_kw)):
        torch.manual_seed(0)
        src = cls(**kw)
        with torch.no_grad():   # "pretrained": every tensor off its init
            for t in list(src.parameters()) + list(src.buffers()):
                t.add_(0.5 + torch.rand_like(t))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "fold_0_best.pt"
            torch.save({"model_state_dict": src.state_dict(),
                        "config": {"pd_epsilon": 0.001, "dropout": 0.2}},
                       ckpt)
            args = argparse.Namespace(per_head=per_head)
            rng = torch.get_rng_state()
            got = ef._load_eval_base(ckpt, args, 2, "cpu", init_seed=123)
            assert torch.equal(torch.get_rng_state(), rng), "RNG perturbed"
            again = ef._load_eval_base(ckpt, args, 2, "cpu", init_seed=123)
            other = ef._load_eval_base(ckpt, args, 2, "cpu", init_seed=124)
            pre = ef._load_eval_base(ckpt, args, 2, "cpu")
        torch.manual_seed(123)
        want = cls(**kw).state_dict()
        sc = ef.scoring_model(got).state_dict()
        ck = src.state_dict()
        assert set(sc) == set(ck) == set(want), (set(sc) ^ set(ck))
        for k in ck:
            assert sc[k].shape == ck[k].shape, k
            assert torch.equal(sc[k], want[k]), f"{k}: not the fresh init"
            assert not torch.equal(sc[k], ck[k]), f"{k}: pretrained survived"
            assert torch.equal(sc[k],
                               ef.scoring_model(again).state_dict()[k]), k
        assert any(not torch.equal(sc[k],
                                   ef.scoring_model(other).state_dict()[k])
                   for k in sc), "another seed drew the same init"
        # the default path still loads the checkpoint
        assert all(torch.equal(v, ck[k]) for k, v in
                   ef.scoring_model(pre).state_dict().items())
        # Scratch trains exactly what Full trains; the rest stays frozen
        on = {}
        for mode in ("Full", ef.SCRATCH_MODE):
            ft = ef.FinetunedSL(got)
            ef.set_trainable(ft, mode)
            on[mode] = {n for n, q in ft.named_parameters() if q.requires_grad}
        assert on["Full"] == on[ef.SCRATCH_MODE], on
        if per_head:
            frozen = {"base.model.cell_emb.weight", "base.model.tau",
                      "base.model.scoring_bias"}
            assert not (frozen & on["Full"]), on["Full"]
            ft = ef.FinetunedSL(got)
            ft.train()
            ef.set_frozen_to_eval(ft, ef.SCRATCH_MODE)
            assert got.model.freeze_inner_stats and got.model.training
    assert ef.scratch_init_seed(42, 3) == 42 + ef.SCRATCH_INIT_SEED_OFFSET + 3


def test_scratch_mode_is_not_warm_started_from_the_pretrained_logit():
    """End to end on a toy single-output fold: the Scratch (a, c) warm start
    is the OLS fit on the FRESH model's logits, and fine-tuning moves the
    fresh weights (it trains)."""
    import argparse
    import numpy as np
    from siamese_esm import SiameseSL
    torch.manual_seed(0)
    n_genes, dim, n_pairs = 40, 6, 400
    emb = torch.randn(n_genes, dim)
    rng = np.random.default_rng(1)
    idx1 = rng.integers(0, n_genes, n_pairs)
    idx2 = rng.integers(0, n_genes, n_pairs)
    # A target the toy encoder can learn: a smooth function of both genes.
    w = torch.randn(dim)
    target = ((emb[idx1] @ w) * (emb[idx2] @ w)).numpy().astype(np.float32)
    train_mask = np.zeros(n_pairs, bool)
    train_mask[:300] = True
    val_mask = ~train_mask
    src = SiameseSL(input_dim=dim, encoder_dims=[8, 8], dropout=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "fold_0_best.pt"
        torch.save({"model_state_dict": src.state_dict(),
                    "config": {"pd_epsilon": 0.001, "dropout": 0.0}}, ckpt)
        args = argparse.Namespace(per_head=False)
        base = ef._load_eval_base(ckpt, args, None, "cpu", init_seed=7)
    init = {k: v.clone() for k, v in base.state_dict().items()}
    model = ef.FinetunedSL(base)
    raw_fresh = ef.compute_logits(model, emb, idx1, idx2, "cpu")
    raw_pre = ef.compute_logits(ef.FinetunedSL(src), emb, idx1, idx2, "cpu")
    assert not np.allclose(raw_fresh, raw_pre)
    warm = ef.fit_ols_affine(raw_fresh[train_mask], target[train_mask])
    m, best_epoch = ef.finetune_one(
        model, emb, idx1, idx2, target, train_mask, val_mask,
        mode=ef.SCRATCH_MODE, epochs=30, patience=30, lr=0.01, batch_size=64,
        weight_decay=0.0, device="cpu", warm_start=warm, seed=0)
    assert best_epoch > 0, "Scratch never beat its own warm start"
    moved = [k for k, v in m.base.state_dict().items()
             if not torch.equal(v, init[k])]
    assert "encoder.projection.weight" in moved, moved
    # temperature and scoring bias are frozen (Full's rule) at init values
    assert "log_temperature" not in moved and "scoring_bias" not in moved


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



def test_extra_score_terms_survive_the_checkpoint_loader():
    """--gene_term / --pair_features: load_multicell_model rebuilds both from
    the weights, scores identically, and Full fine-tuning (only) trains them."""
    import eval_finetuning as ef
    from siamese_esm import SiameseSLMultiCell

    torch.manual_seed(0)
    src = SiameseSLMultiCell(input_dim=6, num_heads=8, encoder_dims=[4, 4, 3],
                             cell_line_dim=2, use_layernorm=False,
                             raw_temperature=True,
                             per_head_last_layer_bias=True, kernel="bilinear",
                             gene_term=True, pair_features=True)
    with torch.no_grad():
        src.gene_w.normal_(0, 0.3)
        src.pair_fc2.weight.normal_(0, 0.5)
    src.eval()
    x1, x2 = torch.randn(5, 6), torch.randn(5, 6)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "fold_0_best.pt"
        torch.save({"model_state_dict": src.state_dict(),
                    "config": {"pd_epsilon": 0.001, "dropout": 0.2}}, ckpt)
        got = ef.load_multicell_model(str(ckpt))
        plain = SiameseSLMultiCell(input_dim=6, num_heads=8,
                                   encoder_dims=[4, 4, 3], cell_line_dim=2,
                                   use_layernorm=False, raw_temperature=True,
                                   per_head_last_layer_bias=True)
        torch.save({"model_state_dict": plain.state_dict(),
                    "config": {"pd_epsilon": 0.001, "dropout": 0.2}}, ckpt)
        old = ef.load_multicell_model(str(ckpt))
    assert got.gene_w is not None and got.pair_fc1 is not None
    assert old.gene_w is None and old.pair_fc1 is None
    with torch.no_grad():
        assert torch.equal(got(x1, x2), src(x1, x2))

    ft = ef.FinetunedSL(ef._HeadSelect(got, 0))
    extra = {"base.model.gene_w", "base.model.pair_fc1.weight",
             "base.model.pair_fc1.bias", "base.model.pair_fc2.weight"}
    for mode, want in (("LP", False), ("LP_last", False), ("Full", True)):
        ef.set_trainable(ft, mode)
        on = {n for n, q in ft.named_parameters() if q.requires_grad}
        assert (extra <= on) == want and (want or not (extra & on)), (mode, on)


if __name__ == "__main__":
    sys.exit(_main())
