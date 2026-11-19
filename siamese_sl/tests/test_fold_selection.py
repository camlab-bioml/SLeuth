#!/usr/bin/env python3
"""Per-fold, per-tier modality selection (SELECTION_MODE=per_fold, 2026-11-12).

  - fold_selection.rank_per_fold: per (cv, fold) each category's argmax of
    fold_metrics[k]["sel_<select_metric>"] (first maximum in catalog order);
    refused when a candidate lacks the key or a fold, when the candidates
    rank on different keys, or when two candidates held out different
    selection splits for one fold (sel_digest).
  - selection_entries / combo_spec: the cv's combo is ONE run over the union
    of its folds' winners (categories sorted, catalog order within one) with
    --fold_modalities mapping each fold to its own winners; a fold that picked
    what a global selection would have trains the same columns in the same
    order. The leave-one-category-out ablation drops the category's pick
    from every fold's set and recomputes the union. A global selection
    gives exactly what the scripts built before (no fold map).
  - the command line the SLURM scripts parse, and run_matches_spec (the
    check run_ablation_summary.sh / paper_tables apply to a combo or
    ablation's config.json).
  - arch_signature: selection_frac / selection_seed are signature keys (an
    ablation and its combo, a Step-1 run and the combo's expected signature
    must agree), off (0 / absent) whatever the seed says.
  - config.conf: SELECTION_FRAC / SELECTION_SEED / SELECTION_MODE validated
    and turned into SELECTION_ARGS; per_fold needs SELECTION_FRAC > 0;
    ARCH_ARGS may not carry the selection flags.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_fold_selection.py -q
    $PYTHON_PATH siamese_sl/tests/test_fold_selection.py
"""
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
sys.path.insert(0, _SIAMESE)

import fold_selection as fs  # noqa: E402
from arch_signature import arch_signature  # noqa: E402

CATALOG = [("esm2", "all_genes_esm2.pt", "protein"),
           ("esm1b", "all_genes_esm1b.pt", "protein"),
           ("go", "all_genes_go.pt", "function"),
           ("bioconceptvec", "all_genes_bioconceptvec.pt", "function"),
           ("ppi_raw", "all_genes_ppi_raw.pt", "network")]
DIGEST = ["d%d" % k + "0" * 62 for k in range(3)]


def _res(scores, key="sel_auprg_any", digests=DIGEST, metric="auprg_any"):
    """A results.json with one sel score per fold."""
    return {"config": {"select_metric": metric},
            "fold_metrics": [
                {"fold": k, key: s, "sel_digest": digests[k],
                 "sel_n_pairs": 100}
                for k, s in enumerate(scores)]}


def _results():
    return {"esm2": _res([0.50, 0.40, 0.60]),
            "esm1b": _res([0.45, 0.40, 0.70]),   # fold 1 ties esm2
            "go": _res([0.30, 0.35, 0.20]),
            "bioconceptvec": _res([0.31, 0.10, 0.10]),
            "ppi_raw": _res([0.20, 0.20, 0.20])}


def _selection(results=None):
    sel = {}
    for cv in ("cv1", "cv2"):
        r = fs.rank_per_fold(CATALOG, results or _results(), 3, cv)
        sel[cv] = fs.selection_entries(CATALOG, r["picks"])
    sel["_selection_meta"] = {"selection_mode": "per_fold",
                              "selection_metric": "sel_auprg_any"}
    return sel


def _raises(fn, *needles):
    try:
        fn()
    except fs.SelectionError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return str(e)
    raise AssertionError("no SelectionError")


def test_per_fold_argmax_first_maximum_in_catalog_order():
    r = fs.rank_per_fold(CATALOG, _results(), 3, "cv1")
    assert r["key"] == "sel_auprg_any"
    p = {k: {c: w["type"] for c, w in v.items()} for k, v in r["picks"].items()}
    assert p == {0: {"protein": "esm2", "function": "bioconceptvec",
                     "network": "ppi_raw"},
                 # 0.40 == 0.40: esm2 comes first in the catalog
                 1: {"protein": "esm2", "function": "go", "network": "ppi_raw"},
                 2: {"protein": "esm1b", "function": "go",
                     "network": "ppi_raw"}}, p
    assert r["digests"] == {0: DIGEST[0], 1: DIGEST[1], 2: DIGEST[2]}
    # the key follows the runs' own checkpoint metric
    r = fs.rank_per_fold(CATALOG[:1], {"esm2": _res([1, 2, 3], "sel_auprg",
                                                    metric="auprg")}, 3)
    assert r["key"] == "sel_auprg"


def test_digest_mismatch_and_missing_key_are_refused():
    res = _results()
    res["go"] = _res([0.3, 0.3, 0.3], digests=[DIGEST[0], "e" * 64, DIGEST[2]])
    msg = _raises(lambda: fs.rank_per_fold(CATALOG, res, 3, "cv2"),
                  "cv2 fold 1", "different selection splits")
    assert "cv2 fold 0" not in msg and "go" in msg, msg
    res = _results()
    del res["esm1b"]["fold_metrics"][2]["sel_auprg_any"]
    _raises(lambda: fs.rank_per_fold(CATALOG, res, 3, "cv1"),
            "fold 2: esm1b has no finite sel_auprg_any", "--selection_frac")
    res = _results()
    del res["ppi_raw"]["fold_metrics"][0]["sel_digest"]
    _raises(lambda: fs.rank_per_fold(CATALOG, res, 3), "records no sel_digest")
    res = _results()
    res["go"]["fold_metrics"] = res["go"]["fold_metrics"][:2]
    _raises(lambda: fs.rank_per_fold(CATALOG, res, 3), "go has no fold_metrics")
    res = _results()
    res["go"] = _res([0.3] * 3, key="sel_auprg", metric="auprg")
    _raises(lambda: fs.rank_per_fold(CATALOG, res, 3), "different keys")
    _raises(lambda: fs.rank_per_fold(CATALOG, {}, 3, "cv3"), "no candidate")


def test_selection_entries_keep_the_global_keys_readable():
    sel = _selection()
    prot = sel["cv1"]["protein"]
    assert prot["type"] == "esm2" and prot["file"] == "all_genes_esm2.pt"
    assert prot["types"] == ["esm2", "esm1b"]
    assert prot["files"] == ["all_genes_esm2.pt", "all_genes_esm1b.pt"]
    assert prot["n_folds_picked"] == {"esm2": 2, "esm1b": 1}
    assert prot["per_fold"]["2"] == {"type": "esm1b",
                                     "file": "all_genes_esm1b.pt",
                                     "score": 0.7}
    assert prot["aupr"] == round((0.5 + 0.4 + 0.7) / 3, 4)
    # a modal tie goes to catalog order: function is bioconceptvec x1, go x2
    assert sel["cv1"]["function"]["type"] == "go"
    assert fs.pick_counts(sel, "cv1") == {"esm2": 2, "esm1b": 1, "go": 2,
                                          "bioconceptvec": 1, "ppi_raw": 3}
    assert fs.selection_mode(sel) == "per_fold"
    assert fs.selection_mode({"cv1": {}}) == "global"
    assert fs.pick_counts({"cv1": {"x": {"type": "go"}}}, "cv1") == {}


def test_union_and_fold_map():
    spec = fs.combo_spec(_selection(), "cv1")
    # categories sorted (function, network, protein), catalog order inside
    assert spec["names"] == ["go", "bioconceptvec", "ppi_raw", "esm2",
                             "esm1b"], spec
    assert spec["files"][0] == "all_genes_go.pt"
    assert spec["fold_modalities"] == {0: [1, 2, 3], 1: [0, 2, 3],
                                       2: [0, 2, 4]}, spec
    assert spec["per_fold"][2] == ["go", "ppi_raw", "esm1b"]
    assert (spec["n_per_fold"], spec["n_union"]) == (3, 5)
    assert fs.spec_fold_files(spec)[0] == [
        "all_genes_bioconceptvec.pt", "all_genes_ppi_raw.pt",
        "all_genes_esm2.pt"]


def test_ablation_drops_the_category_from_every_fold():
    spec = fs.combo_spec(_selection(), "cv1", drop="protein")
    assert spec["names"] == ["go", "bioconceptvec", "ppi_raw"]
    assert spec["fold_modalities"] == {0: [1, 2], 1: [0, 2], 2: [0, 2]}
    assert spec["n_per_fold"] == 2
    spec = fs.combo_spec(_selection(), "cv1", drop="function")
    assert spec["names"] == ["ppi_raw", "esm2", "esm1b"]
    assert spec["fold_modalities"] == {0: [0, 1], 1: [0, 1], 2: [0, 2]}
    _raises(lambda: fs.combo_spec(_selection(), "cv1", drop="text"),
            "nothing to ablate")


def test_global_selection_spec_is_the_old_combo():
    sel = {"cv1": {"protein": {"type": "esm2", "file": "all_genes_esm2.pt",
                               "aupr": 0.5},
                   "function": {"type": "go", "file": "all_genes_go.pt",
                                "aupr": 0.4}},
           "_selection_meta": {"selection_cv": "cv1"}}
    spec = fs.combo_spec(sel, "cv1")
    assert spec["names"] == ["go", "esm2"] and spec["fold_modalities"] is None
    assert fs.combo_spec(sel, "cv1", drop="function")["names"] == ["esm2"]
    assert fs.run_matches_spec(
        {"embeddings_paths": ["../data/all_genes_go.pt",
                              "../data/all_genes_esm2.pt"]}, spec) is None


def test_command_line_prints_what_the_scripts_parse():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "best_per_category.json")
        with open(p, "w") as f:
            json.dump(_selection(), f)
        r = subprocess.run([sys.executable, os.path.join(
            _SIAMESE, "fold_selection.py"), "combo_spec", "--selection", p,
            "--cv", "cv1", "--drop", "network"], capture_output=True,
            text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        paths, name, n, nu, fmap = r.stdout.strip().split("|")
        assert name == "go+bioconceptvec+esm2+esm1b" and (n, nu) == ("2", "4")
        assert json.loads(fmap) == {"0": [1, 2], "1": [0, 2], "2": [0, 3]}
        assert " " not in fmap
        # --data_dir: every file must exist
        r = subprocess.run([sys.executable, os.path.join(
            _SIAMESE, "fold_selection.py"), "combo_spec", "--selection", p,
            "--cv", "cv1", "--data_dir", tmp], capture_output=True,
            text=True, timeout=60)
        assert r.returncode == 1 and "missing" in r.stderr, r.stderr
        r = subprocess.run([sys.executable, os.path.join(
            _SIAMESE, "fold_selection.py"), "combo_spec", "--selection", p,
            "--cv", "cv3"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 1 and "cv3" in r.stderr, r.stderr


def test_run_matches_spec_checks_the_union_and_each_fold():
    spec = fs.combo_spec(_selection(), "cv1")
    paths = [f"../data/{f}" for f in spec["files"]]
    fm = json.dumps({str(k): v for k, v in spec["fold_modalities"].items()})
    assert fs.run_matches_spec({"embeddings_paths": paths,
                                "fold_modalities": fm}, spec) is None
    assert "every fold on the union" in fs.run_matches_spec(
        {"embeddings_paths": paths}, spec)
    bad = dict(spec["fold_modalities"])
    bad[2] = [0, 2, 3]
    assert "fold(s) 2" in fs.run_matches_spec(
        {"embeddings_paths": paths,
         "fold_modalities": json.dumps({str(k): v for k, v in bad.items()})},
        spec)
    assert "not " in fs.run_matches_spec(
        {"embeddings_paths": paths[::-1], "fold_modalities": fm}, spec)


def test_relative_fold_modalities_file_resolves_against_the_run():
    """config.json keeps --fold_modalities as train.py received it: a
    relative file path is relative to the directory train.py ran in
    (output_dir's base), not to the reader's cwd nor config.json's dir."""
    spec = fs.combo_spec(_selection(), "cv1")
    paths = [f"../data/{f}" for f in spec["files"]]
    fm = {str(k): v for k, v in spec["fold_modalities"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.path.join(tmp, "siamese_sl")
        run = os.path.join(cwd, "results", "best_cat_x_cv1")
        os.makedirs(os.path.join(cwd, "maps"))
        os.makedirs(run)
        with open(os.path.join(cwd, "maps", "fm.json"), "w") as f:
            json.dump(fm, f)
        cfg = {"embeddings_paths": paths, "fold_modalities": "maps/fm.json",
               "output_dir": "results/best_cat_x_cv1"}
        assert fs.run_matches_spec(cfg, spec, run) is None
        assert fs.config_fold_files(cfg, run)[0] == \
            [os.path.basename(paths[i]) for i in fm["0"]]
        # not under this process's cwd: refused with a clear reason
        why = fs.run_matches_spec(cfg, spec)
        assert why and "not found" in why, why
        # a copied config whose output_dir is not its directory's tail
        why = fs.run_matches_spec(dict(cfg, output_dir="results/other"),
                                  spec, run)
        assert why and "not recoverable" in why, why


def test_train_parse_fold_modalities_refuses_clearly():
    """train.py --fold_modalities: a value that is neither a JSON object nor
    an existing file is 'file not found' (not an opaque JSONDecodeError);
    folds outside 0..num_folds-1 are refused like missing ones."""
    import train

    def _raises(fn, *needles):
        try:
            fn()
        except SystemExit as e:
            for n in needles:
                assert n in str(e), (n, str(e))
            return
        raise AssertionError("no SystemExit")

    ok = train._parse_fold_modalities('{"0": [0], "1": [1]}', 2)
    assert ok == {0: [0], 1: [1]}
    _raises(lambda: train._parse_fold_modalities("no/such/map.json", 2),
            "file not found", "no/such/map.json")
    _raises(lambda: train._parse_fold_modalities(
        '{"0": [0], "1": [1], "5": [0]}', 2), "fold(s) [5]", "0..1")
    _raises(lambda: train._parse_fold_modalities(
        '{"-1": [0], "0": [0], "1": [1]}', 2), "fold(s) [-1]")
    _raises(lambda: train._parse_fold_modalities('{"0": [0]}', 2),
            "no entry for fold(s) [1]")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "fm.json")
        with open(p, "w") as f:
            json.dump({"0": [1], "1": [0]}, f)
        assert train._parse_fold_modalities(p, 2) == {0: [1], 1: [0]}


def test_selection_keys_are_in_the_signature():
    base = {"use_cell_lines": True}
    off = arch_signature(base)
    assert off["selection_frac"] == 0.0 and off["selection_seed"] is None
    # off whatever the seed says: train.py records its default 42 either way
    assert arch_signature(dict(base, selection_frac=0.0,
                               selection_seed=42)) == off
    on = arch_signature(dict(base, selection_frac=0.1, selection_seed=42))
    assert (on["selection_frac"], on["selection_seed"]) == (0.1, 42)
    assert on != off
    assert arch_signature(dict(base, selection_frac=0.1,
                               selection_seed=7)) != on
    # --fold_modalities is the modality set, never a signature key (an
    # ablation and its combo differ there by design)
    assert "fold_modalities" not in on
    out = subprocess.run(
        [sys.executable, os.path.join(_SIAMESE, "train.py"),
         "--embeddings_paths", "x.pt", "--use_cell_lines",
         "--selection_frac", "0.1", "--selection_seed", "7",
         "--print_arch"], capture_output=True, text=True, cwd=_SIAMESE,
        timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    sig = json.loads(out.stdout.strip().splitlines()[-1])
    assert (sig["selection_frac"], sig["selection_seed"]) == (0.1, 7)


def _source(**env):
    e = {k: v for k, v in os.environ.items()
         if k != "SLURM_JOB_ID" and not k.startswith(("SELECTION_",
                                                      "PCA_", "POST_PCA_"))}
    e.update(env)
    return subprocess.run(
        ["bash", "-c", 'source slurm/config.conf >/dev/null || exit 3; '
         'printf "%s|%s|%s|%s" "$SELECTION_FRAC" "$SELECTION_SEED" '
         '"$SELECTION_MODE" "$SELECTION_ARGS"'], cwd=_SIAMESE, env=e,
        capture_output=True, text=True, timeout=60)


def test_config_conf_selection_settings():
    # Defaults = the revision run (2026-11-13): a 0.1 split, per-fold mode.
    r = _source()
    assert r.returncode == 0 and r.stdout == (
        "0.1|42|per_fold|--selection_frac 0.1 --selection_seed 42"), (
            r.stdout, r.stderr)
    # Set but empty turns the split off; the mode then defaults to global.
    r = _source(SELECTION_FRAC="")
    assert r.returncode == 0 and r.stdout == "|42|global|", r.stdout
    # A split without a mode is per_fold (a hand resubmission with the split
    # but no SELECTION_MODE used to fall back to global silently); global
    # only when asked for.
    r = _source(SELECTION_FRAC="0.2")
    assert r.returncode == 0 and r.stdout == (
        "0.2|42|per_fold|--selection_frac 0.2 --selection_seed 42"), r.stdout
    r = _source(SELECTION_FRAC="0.1", SELECTION_MODE="global")
    assert r.returncode == 0 and r.stdout == (
        "0.1|42|global|--selection_frac 0.1 --selection_seed 42"), r.stdout
    r = _source(SELECTION_FRAC="0")
    assert r.returncode == 0 and r.stdout.endswith("|global|"), r.stdout
    r = _source(SELECTION_FRAC="0.1", SELECTION_MODE="per_fold")
    assert r.returncode == 0 and r.stdout == (
        "0.1|42|per_fold|--selection_frac 0.1 --selection_seed 42"), r.stdout
    r = _source(SELECTION_FRAC=".25", SELECTION_SEED="7")
    assert r.returncode == 0 and r.stdout.endswith(
        "--selection_frac .25 --selection_seed 7"), r.stdout
    for bad in ({"SELECTION_FRAC": "1"}, {"SELECTION_FRAC": "0.1x"},
                {"SELECTION_FRAC": "-0.1"}, {"SELECTION_SEED": "x"},
                {"SELECTION_MODE": "fold"},
                {"SELECTION_MODE": "per_fold", "SELECTION_FRAC": ""},
                {"SELECTION_MODE": "per_fold", "SELECTION_FRAC": "0"}):
        r = _source(**bad)
        assert r.returncode != 0 and "SELECTION_" in r.stderr, (bad, r.stderr)
    for arch in ("--selection_frac 0.1", "--layernorm --selection_seed 3",
                 "--fold_modalities {}"):
        r = _source(ARCH_ARGS=arch)
        assert r.returncode != 0 and "SELECTION_FRAC" in r.stderr, \
            (arch, r.stderr)
    r = _source(ARCH_ARGS="--select_metric auprg")
    assert r.returncode == 0, r.stderr


# ---- the readers: collect_siamese / compare, paper_tables -------------------
_CMP = os.path.join(os.path.dirname(_SIAMESE), "sl_comparison")


def _load_cmp(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(_CMP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_and_compare_disclose_a_per_fold_selection():
    sys.modules.setdefault("model_params",
                           _load_cmp("model_params", "model_params"))
    cs = _load_cmp("collect_siamese", "_fs_collect_siamese")
    cmp = _load_cmp("compare", "_fs_compare")
    sel = _selection()
    sel["_selection_meta"].update({"selection_cv": "cv1",
                                   "selection_variant": "pd:per_head",
                                   "selection_label": "AUPRG_any",
                                   "selection_frac": 0.1,
                                   "selection_seed": 42})
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "best_per_category.json"), "w") as f:
            json.dump(sel, f)
        loaded = cs.load_modality_selection(tmp)
    spec = fs.combo_spec(sel, "cv2")
    fm = json.dumps({str(k): v for k, v in spec["fold_modalities"].items()})
    cfg = {"cv_type": "cv2",
           "embeddings_paths": [f"../data/{f}" for f in spec["files"]],
           "fold_modalities": fm}
    rec = cs.modality_selection_record(loaded, cfg)
    assert rec and rec["selection_mode"] == "per_fold", rec
    assert rec["winners"]["protein"] == "esm2/esm1b"
    assert rec["per_fold"]["function"] == {"0": "bioconceptvec", "1": "go",
                                           "2": "go"}
    assert rec["n_folds_picked"]["ppi_raw"] == 3
    # another CV's union, or one fold's set, is not this combo's
    assert cs.modality_selection_record(
        loaded, dict(cfg, embeddings_paths=cfg["embeddings_paths"][:3])) is None
    # the right union but no per-fold map (every fold on the union), or a
    # fold mapped to other winners: no record, and the reason says why (as
    # paper_tables' run_matches_spec check does)
    why = []
    assert cs.modality_selection_record(
        loaded, {k: v for k, v in cfg.items() if k != "fold_modalities"},
        why=why) is None
    assert why and "every fold on the union" in why[0], why
    bad = {str(k): v for k, v in spec["fold_modalities"].items()}
    bad["1"] = bad["0"]  # fold 1 trained on fold 0's winners
    assert bad["1"] != spec["fold_modalities"][1]
    why = []
    assert cs.modality_selection_record(
        loaded, dict(cfg, fold_modalities=json.dumps(bad)), why=why) is None
    assert why and "fold(s) 1" in why[0], why
    lines = cmp.selection_report({"cv2": {"siamese_sl": {
        "selection_audit": {"modality_selection": rec}}}})
    text = "\n".join(lines)
    assert "selected PER FOLD" in text and "fold 0 bioconceptvec" in text, text
    assert "not this selection's winners" not in text, text
    assert "chosen ONCE" not in text, text


def test_paper_tables_per_fold_selection():
    """tab_solo gains how often each embedding was picked per tier; tab_loco
    names what each fold removed and checks every ablation against the
    per-fold spec; items.selection lists the per-fold picks and counts;
    items.arch matches each combo to its CV's union and fold map. A global
    selection's tables keep their columns (test_paper_tables)."""
    tpt = _load_cmp("test_paper_tables", "_fs_test_paper_tables")
    pt = tpt.pt
    with tempfile.TemporaryDirectory() as tmp:
        root = __import__("pathlib").Path(tmp)
        tpt._build(root)
        res = root / "siamese_sl" / "results"
        cat_of = dict(tpt.CATALOG)
        cat_list = [(e, f"all_genes_{e}.pt", c) for e, c in tpt.CATALOG
                    if e != "kb"]
        # fold 0 of cv1 prefers ea, fold 1 eb (SOLO cv1: ea .90 < eb .93)
        def score(e, cv, k):
            return tpt.SOLO[cv][e] + (.05 if (e, cv, k) == ("ea", "cv1", 0)
                                      else 0)
        sel = {}
        for cv in pt.CVS:
            for e, _f, _c in cat_list:
                p = res / f"pcavar0.8_{e}_{cv}" / "results.json"
                r = json.loads(p.read_text())
                for k, m in enumerate(r["fold_metrics"]):
                    m.update({"sel_auprg_any": score(e, cv, k),
                              "sel_digest": f"d{cv}{k}"})
                p.write_text(json.dumps(r))
            picks = {}
            for k in range(2):
                picks[k] = {}
                for e, f, c in cat_list:
                    s = score(e, cv, k)
                    if c not in picks[k] or s > picks[k][c]["score"]:
                        picks[k][c] = {"type": e, "file": f, "score": s}
            sel[cv] = fs.selection_entries(cat_list, picks)
        old = json.loads((res / "best_per_category.json").read_text())
        sel["_selection_meta"] = dict(
            old["_selection_meta"], selection_mode="per_fold",
            selection_metric="sel_auprg_any", selection_frac=0.1,
            selection_seed=42)
        (res / "best_per_category.json").write_text(json.dumps(sel))
        assert sel["cv1"]["expression"]["types"] == ["ea", "eb"]

        def respec(d, spec):
            for name in ("config.json", "results.json"):
                p = d / name
                j = json.loads(p.read_text())
                c = j if name == "config.json" else j["config"]
                c["embeddings_paths"] = [f"../data/{f}" for f in spec["files"]]
                c["fold_modalities"] = json.dumps(
                    {str(k): v for k, v in spec["fold_modalities"].items()})
                p.write_text(json.dumps(j))

        for v in tpt.VARIANTS:
            for cv in pt.CVS:
                m = res / f"{pt.variant_prefix(v)}latest_combo_{cv}.txt"
                respec(root / "siamese_sl" / m.read_text().strip(),
                       fs.combo_spec(sel, cv))
        for cv in pt.CVS:
            for cat in tpt.LOCO[cv]:
                respec(res / f"ablate_no_{cat}_{cv}",
                       fs.combo_spec(sel, cv, drop=cat))
        respec(res / "var_bilinear_pooled__ablate_no_expression_cv3",
               fs.combo_spec(sel, "cv3", drop="expression"))
        # one ablation left on the union for every fold: stale
        p = res / "ablate_no_kg_cv1" / "config.json"
        j = json.loads(p.read_text())
        j["fold_modalities"] = None
        p.write_text(json.dumps(j))

        tpt._run(root)
        out = root / "sl_comparison" / "results" / "paper_tables"
        solo = tpt._rows(out / "tab_solo.csv")
        cols = list(solo[0])
        i = cols.index("selection_variant")
        assert cols[i + 1:i + 4] == list(pt.SOLO_PER_FOLD_COLS), cols
        ea = tpt._one(solo, variant="pd:per_head", cv="cv1", embedding="ea")
        eb = tpt._one(solo, variant="pd:per_head", cv="cv1", embedding="eb")
        pa = tpt._one(solo, variant="pd:per_head", cv="cv1", embedding="pa")
        assert (ea["n_folds_picked"], ea["picked_folds"]) == ("1", "0"), ea
        assert (eb["n_folds_picked"], eb["picked_folds"]) == ("1", "1"), eb
        assert (pa["n_folds_picked"], pa["picked_folds"]) == ("2", "0;1"), pa
        tpt._close(ea["sel_score_mean"], (.95 + .90) / 2)
        other = tpt._one(solo, variant="pd:pooled", cv="cv1", embedding="ea")
        assert other["n_folds_picked"] == "", other

        loco = tpt._rows(out / "tab_loco.csv")
        ex = tpt._one(loco, variant="pd:per_head", cv="cv1",
                      dropped="expression")
        assert (ex["removed_embedding"], ex["removed_per_fold"]) == (
            "ea/eb", "0:ea;1:eb"), ex
        assert ex["status"] == "ok", ex["status"]
        assert tpt._one(loco, variant="pd:per_head", cv="cv3",
                        dropped="ppi")["status"] == "ok"
        kg = tpt._one(loco, variant="pd:per_head", cv="cv1", dropped="kg")
        assert kg["status"].startswith("stale") and \
            "every fold on the union" in kg["status"], kg["status"]

        nums = json.loads((out / "numbers.json").read_text())
        s = nums["items"]["selection"]["value"]
        assert s["selection_mode"] == "per_fold"
        assert s["per_fold"]["cv1"]["expression"] == {"0": "ea", "1": "eb"}
        assert s["pick_counts"]["cv1"] == {"ea": 1, "eb": 1, "ka": 2,
                                           "pa": 2}, s["pick_counts"]
        a = nums["items"]["arch"]["value"]["pd:per_head"]
        assert a["embeddings_match_winners"] == {
            "cv1": True, "cv2": True, "cv3": True}, a
        missing = "\n".join(nums["missing"])
        for bad in ("per_fold selection's", "selection is not the one",
                    "not loadable"):
            assert bad not in missing, (bad, missing)
        readme = (out / "README.md").read_text()
        assert "Modality selection: PER FOLD" in readme


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
