#!/usr/bin/env python3
"""sl_comparison/collect_siamese.py --combo_markers on a synthetic results tree.

A siamese row must be THIS run's combo, the one results/<prefix>latest_combo_
<cv>.txt names, never an older best_cat_* directory that happens to score
higher, and a CV without a marker gets no row. The baseline row never takes
another variant's var_* directory, by marker or by sweep. The row carries the modality
selection record from best_per_category.json, including Step 2's unranked
candidate lists, and compare.selection_report prints it as the
VALIDATION-selection disclosure. Per PCA arm: each arm's marker names its own
root's combo, and the sweep of arm none's root never ranks another arm's runs
(results/pca_<arm>/), while its own pcavar* runs still compete; every row
names its arm (the selection record's pca_arm / pca, and --pca_arm's
selection_audit.pca_arm on marked, swept and pinned rows alike).

    $PYTHON_PATH -m pytest siamese_sl/tests/test_collect_markers.py -q
    $PYTHON_PATH siamese_sl/tests/test_collect_markers.py
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_CMP = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "sl_comparison")


def _load(name, alias=None):
    # By file path, so sl_comparison/ never lands on sys.path, where its
    # modules could shadow same-named siamese_sl ones (retrieval_metrics.py).
    spec = importlib.util.spec_from_file_location(
        alias or name, os.path.join(_CMP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# collect_siamese does `from model_params import ...`; siamese_sl has no
# module of that name, so registering sl_comparison's shadows nothing.
sys.modules.setdefault("model_params", _load("model_params"))
cs = _load("collect_siamese", "_collect_siamese")
cmp = _load("compare", "_compare")

WINNERS = {"go": "all_genes_go2vec.pt", "ppi": "all_genes_ppi_raw.pt"}


def _run(d: Path, name, auprg, val, cv="cv1", embs=None):
    run = d / name
    run.mkdir(parents=True)
    cfg = {"kernel": "pd", "cell_lines": "per_head", "use_cell_lines": True,
           "select_metric": "auprg_any",
           "embeddings_paths": [f"../data/{f}" for f in
                                (embs or WINNERS.values())]}
    (run / "results.json").write_text(json.dumps({
        "cv_type": cv, "config": cfg,
        "summary": {"auroc_any_mean": 0.8, "auroc_any_std": 0.01,
                    "auprg_any_mean": auprg, "auprg_any_std": 0.02,
                    "val_auprg_any_mean": val}}))
    return run


def _tree(d: Path):
    _run(d, "best_cat_go2vec_ppi_raw_cv1", auprg=0.70, val=0.60)
    # An older combo left on disk: higher validation score, no marker.
    _run(d, "best_cat_esm2_go_cv1", auprg=0.99, val=0.99,
         embs=["all_genes_esm2.pt", "all_genes_go.pt"])
    # cv2 trained but its marker is absent (the combo failed for cv2).
    _run(d, "best_cat_go2vec_ppi_raw_cv2", auprg=0.50, val=0.50, cv="cv2")
    (d / "latest_combo_cv1.txt").write_text(
        "results/best_cat_go2vec_ppi_raw_cv1\n")
    sel = {"_selection_meta": {"selection_cv": "cv1",
                               "selection_variant": "pd:per_head",
                               "selection_metric": "val_auprg_any_mean",
                               "n_scored_on_val": 19}}
    for cv in ("cv1", "cv2", "cv3"):
        sel[cv] = {cat: {"type": Path(f).stem.replace("all_genes_", ""),
                         "file": f} for cat, f in WINNERS.items()}
    (d / "best_per_category.json").write_text(json.dumps(sel))


def test_marker_names_the_row_and_selection_is_disclosed():
    with tempfile.TemporaryDirectory() as tmp:
        disc, out = Path(tmp) / "results", Path(tmp) / "out"
        disc.mkdir()
        _tree(disc)
        cs.discover(disc, out, combo_markers=True)
        rows = sorted(p.name for p in out.glob("*.json"))
        assert rows == ["siamese_sl_cv1.json"], rows
        s = json.loads((out / "siamese_sl_cv1.json").read_text())["summary"]
        assert s["auprg_mean"] == 0.70  # the marked combo, not the 0.99 one
        ms = s["selection_audit"]["modality_selection"]
        assert ms["selection_cv"] == "cv1"
        assert ms["modalities"] == ["go2vec", "ppi_raw"]
        marker = (out / "siamese_selected_cv1.txt").read_text().strip()
        assert marker.endswith("best_cat_go2vec_ppi_raw_cv1")
        lines = cmp.selection_report(cmp.load_results(out))
        assert any("VALIDATION" in ln for ln in lines), lines


def test_marker_naming_another_variants_combo_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        disc, out = Path(tmp) / "results", Path(tmp) / "out"
        disc.mkdir()
        _tree(disc)
        (disc / "latest_combo_cv1.txt").write_text(
            "results/var_pd_pooled__best_cat_go2vec_ppi_raw_cv1\n")
        # The named combo exists and its config records pd:per_head, so only
        # the var_* directory-name guards can refuse it (not a missing file,
        # not variant_matches).
        _run(disc, "var_pd_pooled__best_cat_go2vec_ppi_raw_cv1",
             auprg=0.9, val=0.9)
        # the marker itself is refused (its name is not a best_cat_* combo) ...
        want = cs.expected_variant(cs.BASELINE_MODEL, None)
        assert cs.combo_marker_paths(disc, want, ["cv1"],
                                     cs.BASELINE_MODEL) == {}
        # ... and so is the row
        cs.discover(disc, out, combo_markers=True)
        assert not list(out.glob("*.json"))
        assert not (out / "siamese_selected_cv1.txt").exists()


def test_baseline_sweep_skips_other_variants_dirs():
    # Without markers every results.json competes; a var_* combo whose config
    # records pd:per_head and outscores the rest must still never become the
    # baseline siamese_sl row (only the var_* directory skip refuses it).
    with tempfile.TemporaryDirectory() as tmp:
        disc, out = Path(tmp) / "results", Path(tmp) / "out"
        disc.mkdir()
        _tree(disc)
        _run(disc, "var_pd_pooled__best_cat_go2vec_ppi_raw_cv1",
             auprg=0.9, val=0.999)
        cs.discover(disc, out)
        s = json.loads((out / "siamese_sl_cv1.json").read_text())["summary"]
        assert s["auprg_mean"] == 0.99  # best_cat_esm2_go_cv1, not the var_


def test_sweep_writes_rows_only_for_the_requested_cvs():
    # The caller (collect_and_regrade_siamese.sh) pins and re-grades $CVS
    # only: a row the sweep wrote for another CV would stay self-scored at
    # top level and reach comparison.csv beside shared-evaluator rows.
    def rows(**kw):
        with tempfile.TemporaryDirectory() as tmp:
            disc, out = Path(tmp) / "results", Path(tmp) / "out"
            disc.mkdir()
            _tree(disc)               # cv1 combos + one cv2 run, pd:per_head
            cs.discover(disc, out, **kw)
            return sorted(p.name for p in out.glob("*"))
    both = ["siamese_selected_cv1.txt", "siamese_selected_cv2.txt",
            "siamese_sl_cv1.json", "siamese_sl_cv2.json"]
    assert rows() == both, rows()
    assert rows(cv_types=["cv1"]) == [both[0], both[2]], rows(cv_types=["cv1"])
    assert rows(cv_types=["cv2", "cv3"]) == [both[1], both[3]]
    sh = (Path(cs.__file__).parent / "slurm"
          / "collect_and_regrade_siamese.sh").read_text()
    assert 'SIAMESE_BASELINE_ARGS=""' not in sh \
        and 'SIAMESE_BASELINE_ARGS="--cv_types $CVS"' in sh


def test_sweep_never_ranks_another_pca_arms_runs():
    # SIAMESE_BASELINE_DISCOVER=all: every results.json under arm none's root
    # siamese_sl/results competes for the row, and the other PCA arms' roots
    # are its children (results/pca_<arm>/). Their runs are other models: they
    # never become arm none's row, however well they score, for the baseline
    # or a variant. Arm none's own Step-1 runs (pcavar*) still compete: the
    # skip is on exact arm-root names, not a pca* prefix.
    with tempfile.TemporaryDirectory() as tmp:
        disc, out = Path(tmp) / "results", Path(tmp) / "out"
        disc.mkdir()
        _tree(disc)                       # cv1: 0.60 marked, 0.99 unmarked
        _run(disc, "pcavar0.8_go2vec_cv1", auprg=0.5, val=0.995,
             embs=["all_genes_go2vec.pt"])
        for root in cs.PCA_ARM_ROOTS:
            _run(disc / root, "best_cat_go2vec_ppi_raw_cv1", auprg=0.1,
                 val=0.999)
        _run(disc / "pca_per_modality", "pcavar0.8_go2vec_cv1", auprg=0.1,
             val=0.9999, embs=["all_genes_go2vec.pt"])
        assert set(cs.PCA_ARM_ROOTS) == {
            "pca_none", "pca_joint", "pca_per_modality", "pca_both",
            "pca_arms"}
        cs.discover(disc, out, cv_types=["cv1"])
        marker = (out / "siamese_selected_cv1.txt").read_text().strip()
        assert marker == str(disc / "pcavar0.8_go2vec_cv1"), marker
        # a variant row likewise (its runs are var_<k>_<c>__*)
        cfg = {"cell_lines": "pooled"}
        for d, val in ((disc, 0.5), (disc / "pca_joint", 0.999)):
            run = _run(d, "var_pd_pooled__best_cat_go2vec_ppi_raw_cv1",
                       auprg=val, val=val)
            res = json.loads((run / "results.json").read_text())
            res["config"].update(cfg)
            (run / "results.json").write_text(json.dumps(res))
        cs.discover(disc, out, model_name="siamese_sl_pd_pooled",
                    cv_types=["cv1"])
        marker = (out / "siamese_sl_pd_pooled_selected_cv1.txt").read_text()
        assert marker.strip() == str(
            disc / "var_pd_pooled__best_cat_go2vec_ppi_raw_cv1"), marker
        # ... while arm joint's own sweep ranks its own runs
        cs.discover(disc / "pca_joint", Path(tmp) / "out_joint",
                    cv_types=["cv1"])
        marker = (Path(tmp) / "out_joint" / "siamese_selected_cv1.txt")
        assert marker.read_text().strip() == str(
            disc / "pca_joint" / "best_cat_go2vec_ppi_raw_cv1")


def test_markers_name_each_arms_own_combo():
    # Arm X's combo has arm none's combo's name (the winners fix it) one
    # directory down, and its marker says results/pca_<arm>/<combo>. Each
    # arm's row is its own root's combo: the marker's directory name is
    # looked up in the arm's root (--discover_dir), never resolved elsewhere.
    with tempfile.TemporaryDirectory() as tmp:
        disc = Path(tmp) / "results"
        disc.mkdir()
        _tree(disc)                       # arm none: 0.70 marked
        joint = disc / "pca_joint"
        joint.mkdir()
        _tree(joint)
        res = json.loads((joint / "best_cat_go2vec_ppi_raw_cv1" /
                          "results.json").read_text())
        res["summary"]["auprg_any_mean"] = 0.42
        (joint / "best_cat_go2vec_ppi_raw_cv1" / "results.json").write_text(
            json.dumps(res))
        (joint / "latest_combo_cv1.txt").write_text(
            "results/pca_joint/best_cat_go2vec_ppi_raw_cv1\n")
        for d, want in ((disc, 0.70), (joint, 0.42)):
            out = Path(tmp) / f"out_{d.name}"
            cs.discover(d, out, combo_markers=True, cv_types=["cv1"])
            s = json.loads((out / "siamese_sl_cv1.json").read_text())[
                "summary"]
            assert s["auprg_mean"] == want, (d, s["auprg_mean"])
            assert (out / "siamese_selected_cv1.txt").read_text().strip() \
                == str(d / "best_cat_go2vec_ppi_raw_cv1")
            assert s["selection_audit"]["modality_selection"][
                "modalities"] == ["go2vec", "ppi_raw"]


def test_rows_name_their_pca_arm():
    # Every arm's rows have the same names, and arm none's sit at the
    # pre-arms path (2026-10-21): a combo row's selection record carries the
    # arm and PCA its selection recorded (_selection_meta pca_arm / pca),
    # and --pca_arm (label only, collect_and_regrade_siamese.sh's SLEUTH_ARM)
    # writes selection_audit.pca_arm on every row: the marked combo's, the
    # SIAMESE_BASELINE_DISCOVER=all sweep's and a pinned one's, all of which
    # the re-grade carries (evaluate_model.CARRY_KEYS). A selection from
    # before the arms gets no None values, and the arms' records of one
    # selection still print as one disclosure.
    label = "PCA on the concatenation of all modalities (variance 0.8)"
    with tempfile.TemporaryDirectory() as tmp:
        disc = Path(tmp) / "results" / "pca_joint"
        disc.mkdir(parents=True)
        _tree(disc)
        (disc / "latest_combo_cv1.txt").write_text(
            "results/pca_joint/best_cat_go2vec_ppi_raw_cv1\n")
        p = disc / "best_per_category.json"
        sel = json.loads(p.read_text())
        sel["_selection_meta"].update(pca_arm="joint", pca=label)
        p.write_text(json.dumps(sel))
        out = Path(tmp) / "out"
        cs.discover(disc, out, combo_markers=True, cv_types=["cv1"],
                    pca_arm="joint")
        s = json.loads((out / "siamese_sl_cv1.json").read_text())["summary"]
        ms = s["selection_audit"]["modality_selection"]
        assert (ms["pca_arm"], ms["pca"]) == ("joint", label), ms
        assert s["selection_audit"]["pca_arm"] == "joint", s
        lines = cmp.selection_report(cmp.load_results(out))
        assert sum("one selection shared by" in ln for ln in lines) == 1, lines
        # the sweep (no --combo_markers, no selection record) and no label
        cs.discover(disc, Path(tmp) / "sweep", cv_types=["cv1"],
                    pca_arm="joint")
        s = json.loads((Path(tmp) / "sweep" / "siamese_sl_cv1.json")
                       .read_text())["summary"]
        assert s["selection_audit"]["pca_arm"] == "joint" and \
            "modality_selection" not in s["selection_audit"], s
        cs.discover(disc, Path(tmp) / "nolabel", cv_types=["cv1"])
        s = json.loads((Path(tmp) / "nolabel" / "siamese_sl_cv1.json")
                       .read_text())["summary"]
        assert "pca_arm" not in (s.get("selection_audit") or {}), s
        # a pinned run (single-file mode), as the pin loop calls it
        pin = Path(tmp) / "pin.json"
        argv = sys.argv
        try:
            sys.argv = ["collect_siamese.py", "--siamese_results",
                        str(disc / "best_cat_go2vec_ppi_raw_cv1" /
                            "results.json"), "--cv_type", "cv1",
                        "--pca_arm", "joint", "--out", str(pin)]
            cs.main()
        finally:
            sys.argv = argv
        s = json.loads(pin.read_text())["summary"]
        assert s["selection_audit"] == {"pca_arm": "joint"}, s
        # a selection from before the arms: no arm keys, no None values
        del sel["_selection_meta"]["pca_arm"], sel["_selection_meta"]["pca"]
        p.write_text(json.dumps(sel))
        rec = cs.modality_selection_record(
            cs.load_modality_selection(disc),
            {"embeddings_paths": [f"../data/{f}" for f in WINNERS.values()]})
        assert rec is not None and "pca_arm" not in rec and "pca" not in rec, \
            rec


def test_combo_on_a_subset_of_the_winners_gets_no_selection_record():
    # The combo job refuses a missing winner file, so a marked combo trained
    # on fewer modalities is a legacy / foreign one: its row is kept but the
    # selection is not attached to it (compare then says "no selection record").
    with tempfile.TemporaryDirectory() as tmp:
        disc, out = Path(tmp) / "results", Path(tmp) / "out"
        disc.mkdir()
        _tree(disc)
        _run(disc, "best_cat_go2vec_cv1", auprg=0.6, val=0.6,
             embs=["all_genes_go2vec.pt"])
        (disc / "latest_combo_cv1.txt").write_text(
            "results/best_cat_go2vec_cv1\n")
        cs.discover(disc, out, combo_markers=True)
        s = json.loads((out / "siamese_sl_cv1.json").read_text())["summary"]
        assert s["auprg_mean"] == 0.6
        assert "modality_selection" not in (s.get("selection_audit") or {})
    sel = {"meta": {}, "winners": {c: {"type": Path(f).stem, "file": f}
                                   for c, f in WINNERS.items()}}
    assert cs.modality_selection_record(
        sel, {"embeddings_paths": ["../data/all_genes_go2vec.pt"]}) is None
    rec = cs.modality_selection_record(
        sel, {"embeddings_paths": [f"../data/{f}" for f in WINNERS.values()]})
    assert rec is not None and len(rec["modalities"]) == 2


def test_write_out_cut_write_leaves_no_row():
    """write_out, like evaluate_model.write_summary: a write that fails
    part-way leaves nothing under the row's name (run_compare.sh's closing
    summary counts a row file that exists as present)."""
    import errno
    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "results"
        outp = res / "siamese_sl_cv1.json"
        row = {"model": "siamese_sl", "auprg_mean": 0.5}
        real = json.dump

        def cut(obj, f, **kw):
            f.write(json.dumps(obj, **kw)[:20])
            f.flush()
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

        json.dump = cut
        try:
            try:
                cs.write_out(row, ["auprg"], "cv1", outp)
                raise AssertionError("the failed write raised nothing")
            except OSError as e:
                assert e.errno == errno.EDQUOT, e
        finally:
            json.dump = real
        assert [p.name for p in res.iterdir()] == [], list(res.iterdir())
        cs.write_out(row, ["auprg"], "cv1", str(outp))
        assert [p.name for p in res.iterdir()] == ["siamese_sl_cv1.json"]
        assert json.loads(outp.read_text()) == {"summary": row}


def test_unranked_candidates_reach_the_disclosure():
    # Step 2's unranked lists (the BEST_CAT_ALLOW_MISSING=1 gaps, and every
    # catalog entry whose .pt was never generated) must travel with the row
    # and be printed: the default run always has at least one of the latter
    # (esmc is in EMB_CATALOG but commented out of PRECOMPUTED_TYPES).
    with tempfile.TemporaryDirectory() as tmp:
        disc, out = Path(tmp) / "results", Path(tmp) / "out"
        disc.mkdir()
        _tree(disc)
        p = disc / "best_per_category.json"
        sel = json.loads(p.read_text())
        # mashup is in EMB_CATALOG and NOT in OPTIONAL_TYPES, so Step 2 also
        # records it as required-missing: an absence that is NOT accepted
        # (--skip-generate with a catalog file gone), and the argmax then
        # ranked a smaller catalog than the paper's.
        sel["_selection_meta"].update(unranked_foreign_token=["scgpt"],
                                      unranked_not_generated=["esmc",
                                                              "mashup"],
                                      unranked_required_missing=["mashup"])
        p.write_text(json.dumps(sel))
        cs.discover(disc, out, combo_markers=True)
        ms = json.loads((out / "siamese_sl_cv1.json").read_text())[
            "summary"]["selection_audit"]["modality_selection"]
        assert ms["unranked_foreign_token"] == ["scgpt"]
        assert ms["unranked_not_generated"] == ["esmc", "mashup"]
        assert ms["unranked_required_missing"] == ["mashup"]
        lines = cmp.selection_report(cmp.load_results(out))
        assert any("INCOMPLETE" in ln and "scgpt" in ln for ln in lines), lines
        assert any("never generated" in ln and "esmc" in ln
                   for ln in lines), lines
        assert any("should have built it" in ln and "mashup" in ln
                   for ln in lines), lines


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
