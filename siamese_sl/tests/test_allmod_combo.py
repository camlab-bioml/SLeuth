#!/usr/bin/env python3
"""The all-modalities combo (2026-11-12, review response; ALLMOD_COMBO=1).

  * fold_selection.allmod_spec / `fold_selection.py allmod_spec`: every
    EMB_CATALOG embedding in catalog order, an absent OPTIONAL_TYPES file
    skipped and named, any other missing file refused.
  * config.conf: ALLMOD_COMBO 0 (default) | 1, anything else refused;
    allmod_prefix / allmod_row; ARCH_ARGS may not carry --all_modalities.
  * train.py --all_modalities: recorded by --print_arch (an arch_signature key
    that tells the model from a selected combo), refused with
    --fold_modalities.
  * slurm/run_allmod_combo.sh: its train.py call is the selected combo's
    Step-3 call minus --fold_modalities plus --all_modalities; run end to end
    in a scratch tree (real confs, stub train.py) it trains every variant of
    its CV on the catalog with the arm's PCA and the selection split, moves
    the previous row aside first, writes all_modalities.json and the marker,
    and refuses a missing required embedding.
  * collect_siamese.py --allmod: the marked run only, with the
    all_modalities record; no other row's discovery reaches it.
  * compare.py / bootstrap_ci.py: the rows sort after the variants, disclose
    "no modality selection", and selected - allmod is a default pair.
  * both orchestrators with ALLMOD_COMBO=1: one array per arm after the
    generation only, the arm's compare after it; with 0 nothing changes.

    python3 siamese_sl/tests/test_allmod_combo.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SIAMESE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SIAMESE)
_CMP = os.path.join(_ROOT, "sl_comparison")
for p in (_SIAMESE, _CMP, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import fold_selection as fs  # noqa: E402
from arch_signature import arch_signature, signature_diff  # noqa: E402
import collect_siamese as cs  # noqa: E402
import compare  # noqa: E402

_SLURM = os.path.join(_SIAMESE, "slurm")
_SH = os.path.join(_SLURM, "run_allmod_combo.sh")


def _bash(script, **env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "ALLMOD_", "PCA_", "SIAMESE_"))
         and k not in ("ARCH_ARGS", "SELECTION_FRAC", "SELECTION_MODE")}
    e.update(env)
    return subprocess.run(["bash", "-c", script], cwd=_SIAMESE, env=e,
                          capture_output=True, text=True, timeout=120)


def _catalog():
    r = _bash('source slurm/config.conf >/dev/null && '
              'source slurm/run_best_per_category.conf && '
              'source slurm/run_generate_embeddings.conf && '
              'printf "%s\\n" "${EMB_CATALOG[@]}"; echo "--"; '
              'printf "%s\\n" "${OPTIONAL_TYPES[@]}"')
    assert r.returncode == 0, r.stderr
    cat, opt = r.stdout.split("--\n")
    return ([tuple(e.split(":")) for e in cat.split()], opt.split())


# ---------------------------------------------------------------------------
# the modality set
# ---------------------------------------------------------------------------

def test_allmod_spec_is_the_catalog_in_order():
    cat, opt = _catalog()
    assert len(cat) == 19 and len(cat) - len(set(opt) & {t for t, _, _ in cat}) \
        == 16, (cat, opt)
    with tempfile.TemporaryDirectory() as d:
        for t, f, _ in cat:
            if t not in opt:
                open(os.path.join(d, f), "w").close()
        spec = fs.allmod_spec(cat, d, opt)
        want = [t for t, _, _ in cat if t not in opt]
        assert spec["names"] == want and len(want) == 16, spec
        assert spec["absent_optional"] == [t for t, _, _ in cat if t in opt]
        # an optional one that IS generated is part of the combo, in place
        open(os.path.join(d, "all_genes_esmc.pt"), "w").close()
        names = fs.allmod_spec(cat, d, opt)["names"]
        assert names == [t for t, _, _ in cat
                         if t not in opt or t == "esmc"], names
        # the command line the job reads: paths|types|n|absent optional
        r = subprocess.run(
            [sys.executable, os.path.join(_SIAMESE, "fold_selection.py"),
             "allmod_spec", "--data_dir", d, "--optional", *opt,
             "--catalog", *(":".join(e) for e in cat)],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        paths, types, n, absent = r.stdout.strip().split("|")
        assert paths.split() == [os.path.join(d, f) for t, f, _ in cat
                                 if t in names]
        assert types.split("+") == names and int(n) == 17
        assert absent.split(",") == [t for t in ("scprint", "seqvec")]
        # a required file missing: refused, naming it
        os.remove(os.path.join(d, "all_genes_go.pt"))
        r = subprocess.run(
            [sys.executable, os.path.join(_SIAMESE, "fold_selection.py"),
             "allmod_spec", "--data_dir", d, "--optional", *opt,
             "--catalog", *(":".join(e) for e in cat)],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 1 and "go" in r.stderr \
            and r.stdout == "", (r.stdout, r.stderr)
    try:
        fs.allmod_spec([("a", "x.pt", "c"), ("a", "y.pt", "c")])
        raise AssertionError("a repeated type was accepted")
    except fs.SelectionError:
        pass


def test_entry_layer_width_of_the_catalog():
    # The input width of arm none is the sum of the 16 native widths
    # (paper_tables items.inputs.native_dims): 9,448 against the selected
    # six-modality combo's 4,668, so the entry Linear(k [+24] -> 32) about
    # doubles; the 64G / 5-day header of the combo still holds (header).
    native = {"geneformer": 1152, "scgpt": 512, "gene2vec": 200,
              "esm2": 1280, "esm1b": 1280, "prot_t5": 1024,
              "text_embed": 1024, "bioconceptvec": 100, "node2vec_ppi": 128,
              "ppi_svd": 256, "ppi_raw": 1024, "mashup": 500, "go": 200,
              "go2vec": 128, "onto2vec": 128, "kg_complex": 512}
    cat, opt = _catalog()
    assert sorted(native) == sorted(t for t, _, _ in cat if t not in opt)
    k = sum(native.values())
    assert k == 9448
    assert 32 * k + 32 == 302368          # pooled entry weights + bias
    head = open(_SH).read().split("\nset -e\n", 1)[0]
    for line in ("#SBATCH --nodelist=gpu3", "#SBATCH --gres=gpu:1",
                 "#SBATCH --mem=64G", "#SBATCH --time=5-00:00:00",
                 "#SBATCH --job-name=allmod_combo", "#SBATCH --array=0-2"):
        assert line in head, line
    assert not re.search(r"--cpu\b(?!s-per-task)", open(_SH).read())  # GPU


# ---------------------------------------------------------------------------
# config.conf, train.py, arch_signature
# ---------------------------------------------------------------------------

def test_config_conf_switch_and_names():
    r = _bash('source slurm/config.conf >/dev/null && echo "[$ALLMOD_COMBO]" '
              '&& for v in pd:per_head pd:pooled bilinear:pooled; do '
              'echo "$(allmod_prefix $v) $(allmod_row $v)"; done')
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "[1]", "allmod__ siamese_sl_allmod",    # on by default since 2026-11-13
        "allmod__var_pd_pooled__ siamese_sl_allmod_pd_pooled",
        "allmod__var_bilinear_pooled__ siamese_sl_allmod_bilinear_pooled"], \
        r.stdout
    r = _bash('source slurm/config.conf >/dev/null && echo "$ALLMOD_COMBO"',
              ALLMOD_COMBO="1")
    assert r.returncode == 0 and r.stdout.strip() == "1", r.stderr
    for bad_env in ({"ALLMOD_COMBO": "yes"}, {"ARCH_ARGS": "--all_modalities"},
                    {"ARCH_ARGS": "--layernorm --all"}):
        r = _bash("source slurm/config.conf", **bad_env)
        assert r.returncode != 0, (bad_env, r.stdout, r.stderr)
    # the printed blocks' _paste_env never passes the pasting shell's switch
    r = _bash('source slurm/config.conf >/dev/null && pasted_env_def X=1')
    assert "-u ALLMOD_COMBO" in r.stdout, r.stdout


def test_train_py_records_all_modalities_and_signature_tells_it_apart():
    def arch(*extra):
        r = subprocess.run(
            [sys.executable, "train.py", "--embeddings_paths", "x.pt",
             "--sl_path", "y.txt", "--output_dir", "results/_x",
             "--use_cell_lines", "--print_arch", *extra],
            cwd=_SIAMESE, capture_output=True, text=True, timeout=300)
        return r
    a, b = arch(), arch("--all_modalities")
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    sa, sb = json.loads(a.stdout.splitlines()[-1]), \
        json.loads(b.stdout.splitlines()[-1])
    assert sa["all_modalities"] is False and sb["all_modalities"] is True
    assert {k for k in sa if sa[k] != sb[k]} == {"all_modalities"}
    r = arch("--all_modalities", "--fold_modalities", '{"0": [0]}')
    assert r.returncode != 0 and "--fold_modalities" in r.stderr, r.stderr
    # a config that predates the flag is a selected / single run
    base = {"use_cell_lines": True, "encoder_dims": [32]}
    assert arch_signature(base)["all_modalities"] is False
    assert signature_diff(base, {**base, "all_modalities": True}) == \
        {"all_modalities": (False, True)}


def _logical(text):
    return text.replace("\\\n", " ")


def test_train_call_is_the_combo_call_without_selection():
    combo = open(os.path.join(_SLURM, "run_best_per_category_combo.sh")).read()
    i = combo.index("    $PYTHON_PATH train.py \\\n"
                    "        --embeddings_paths $EMB_PATHS")
    want = [ln.strip().rstrip("\\").rstrip() for ln in
            combo[i:combo.index("\n    TRAIN_EXIT=$?", i)].splitlines()[1:]]
    text = open(_SH).read()
    i = text.index("        $PYTHON_PATH train.py \\\n")
    got = [ln.strip().rstrip("\\").rstrip() for ln in
           text[i:text.index("\n        TRAIN_EXIT=$?", i)].splitlines()[1:]]
    assert '${FOLD_MAP:+--fold_modalities "$FOLD_MAP"}' in want
    assert got == [a for a in want if "FOLD_MAP" not in a] + \
        ["--all_modalities"], (got, want)
    # no Step-1 run, no selection is read
    body = text.split("\nset -e\n", 1)[1]
    assert "best_per_category.json" not in body and "pcavar" not in body \
        and "combo_spec" not in body, body


# ---------------------------------------------------------------------------
# run_allmod_combo.sh end to end
# ---------------------------------------------------------------------------

_STUB_TRAIN = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--output_dir") + 1]
cv = argv[argv.index("--cv_type") + 1]
with open(os.environ["ALLMOD_TEST_LOG"], "a") as f:
    f.write(json.dumps(argv) + "\n")
i = argv.index("--embeddings_paths") + 1
paths = []
while not argv[i].startswith("--"):
    paths.append(argv[i]); i += 1
json.dump({"embeddings_paths": paths, "cv_type": cv,
           "all_modalities": "--all_modalities" in argv,
           "fold_modalities": None}, open(os.path.join(out, "config.json"), "w"))
if os.environ.get("ALLMOD_TEST_FAIL") == cv:
    sys.exit(1)
json.dump({"summary": {"auprg_mean": 0.5}}, open(os.path.join(out, "results.json"), "w"))
'''


def _tree(tmp):
    tmp = os.path.realpath(tmp)       # config.conf resolves DATA_DIR by pwd
    sia = Path(tmp) / "siamese_sl"
    shutil.copytree(_SLURM, sia / "slurm",
                    ignore=shutil.ignore_patterns("logs"))
    conf = sia / "slurm" / "config.conf"
    text, n = re.subn(r"(?m)^PYTHON_PATH=.*$",
                      f'PYTHON_PATH="{sys.executable}"', conf.read_text())
    assert n == 1
    conf.write_text(text)
    shutil.copy(os.path.join(_SIAMESE, "fold_selection.py"), sia)
    (sia / "train.py").write_text(_STUB_TRAIN)
    (Path(tmp) / "sl_comparison" / "folds").mkdir(parents=True)
    data = Path(tmp) / "data"
    data.mkdir()
    cat, opt = _catalog()
    for t, f, _ in cat:
        if t not in opt:
            (data / f).write_text("")
    return sia, data, cat, opt


def _run(sia, log, **env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "ALLMOD_", "PCA_", "SIAMESE_",
                              "SELECTION_"))
         and k not in ("ARCH_ARGS", "RUN_TOKEN", "USE_CELL_LINES")}
    e.update(ALLMOD_TEST_LOG=str(log), **env)
    return subprocess.run(["bash", "slurm/run_allmod_combo.sh"], cwd=sia,
                          env=e, capture_output=True, text=True, timeout=300)


def _argval(argv, flag):
    return argv[argv.index(flag) + 1]


def test_run_allmod_script_trains_every_variant_on_the_catalog():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)
        sia, data, cat, opt = _tree(tmp)
        log = Path(tmp) / "calls.jsonl"
        res, cmp_ = sia / "results" / "pca_joint", \
            Path(tmp) / "sl_comparison" / "results" / "pca_joint"
        cmp_.mkdir(parents=True)
        for n in ("siamese_sl_allmod_pd_pooled_cv2.json",
                  "siamese_sl_allmod_pd_pooled_selected_cv2.txt",
                  "siamese_sl_pd_pooled_cv2.json"):
            (cmp_ / n).write_text("old\n")
        old = res / "allmod__var_pd_pooled__all_modalities_cv2"
        old.mkdir(parents=True)
        (old / "stale.npy").write_text("x")
        r = _run(sia, log, PCA_ARM="joint", PCA_ARMS="none joint",
                 SIAMESE_VARIANTS_LIST="pd:pooled pd:per_head",
                 SELECTION_FRAC="0.1", SELECTION_SEED="42",
                 SELECTION_MODE="per_fold", ALLMOD_CVS="cv2",
                 ARCH_ARGS="--layernorm")
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        calls = [json.loads(ln) for ln in log.read_text().splitlines()]
        assert len(calls) == 2, calls
        want_files = [str(data / f) for t, f, _ in cat if t not in opt]
        for argv, v in zip(calls, ("pd:pooled", "pd:per_head")):
            i = argv.index("--embeddings_paths") + 1
            assert argv[i:i + 16] == want_files, argv
            assert argv[i + 16].startswith("--")
            assert _argval(argv, "--cv_type") == "cv2"
            assert (_argval(argv, "--kernel"), _argval(argv, "--cell_lines")) \
                == tuple(v.split(":"))
            assert _argval(argv, "--post_pca_variance") == "0.8" \
                and "--no_post_pca" not in argv
            assert (_argval(argv, "--selection_frac"),
                    _argval(argv, "--selection_seed")) == ("0.1", "42")
            assert "--fold_modalities" not in argv and \
                argv[-2:] == ["--layernorm", "--all_modalities"], argv
            assert "--use_cell_lines" in argv and "--folds_dir" in argv
        for pre in ("allmod__var_pd_pooled__", "allmod__"):
            d = res / f"{pre}all_modalities_cv2"
            assert (res / f"{pre}latest_allmod_cv2.txt").read_text() \
                .strip() == f"results/pca_joint/{pre}all_modalities_cv2"
            rec = json.loads((d / "all_modalities.json").read_text())
            assert rec["all_modalities"] is True and rec["pca_arm"] == "joint" \
                and rec["n_modalities"] == 16 and rec["catalog_size"] == 19 \
                and rec["absent_optional"] == [t for t, _, _ in cat
                                               if t in opt], rec
        assert not (old / "stale.npy").exists()     # the previous model went
        # the previous allmod row went aside first; the selected row stayed
        assert not (cmp_ / "siamese_sl_allmod_pd_pooled_cv2.json").exists()
        assert (cmp_ / "siamese_sl_pd_pooled_cv2.json").exists()
        stale = sorted(p.name for p in (cmp_ / "stale_siamese").iterdir())
        assert len(stale) == 2 and all(n.startswith(
            "siamese_sl_allmod_pd_pooled_") for n in stale), stale
        # no selected-combo name is ever written
        assert not list(res.glob("*latest_combo*")) and \
            not list(res.glob("*best_cat_*"))

        # arm none: the PCA is --no_post_pca; a failed train writes no marker
        log.unlink()
        r = _run(sia, log, PCA_ARM="none", PCA_ARMS="none joint",
                 SIAMESE_VARIANTS_LIST="pd:pooled", ALLMOD_CVS="cv1 cv3",
                 ALLMOD_TEST_FAIL="cv3", SELECTION_FRAC="")
        assert r.returncode == 1 and "pd:pooled/cv3" in r.stdout, r.stdout
        calls = [json.loads(ln) for ln in log.read_text().splitlines()]
        assert [_argval(a, "--cv_type") for a in calls] == ["cv1", "cv3"]
        assert all("--no_post_pca" in a and "--selection_frac" not in a
                   for a in calls)
        r0 = sia / "results"
        assert (r0 / "allmod__var_pd_pooled__latest_allmod_cv1.txt").exists()
        assert not (r0 / "allmod__var_pd_pooled__latest_allmod_cv3.txt").exists()

        # a required embedding missing: refused, nothing trained, no marker
        (data / "all_genes_go.pt").unlink()
        log.unlink()
        r = _run(sia, log, PCA_ARM="none", SIAMESE_VARIANTS_LIST="pd:pooled",
                 ALLMOD_CVS="cv1")
        assert r.returncode == 1 and "REFUSED" in r.stdout, r.stdout
        assert not log.exists()
        assert not (r0 / "allmod__var_pd_pooled__latest_allmod_cv1.txt").exists()


# ---------------------------------------------------------------------------
# collection, compare, bootstrap pairs
# ---------------------------------------------------------------------------

def _run_dir(d: Path, cv, auprg, allmod, cell_lines="pooled"):
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"kernel": "pd", "cell_lines": cell_lines, "cv_type": cv,
           "use_cell_lines": True, "select_metric": "auprg_any",
           "embeddings_paths": ["../data/all_genes_a.pt",
                                "../data/all_genes_b.pt"],
           "selection_frac": 0.1, "selection_seed": 42}
    if allmod:
        cfg["all_modalities"] = True
    (d / "config.json").write_text(json.dumps(cfg))
    summ = {k: 0.5 for k in ("auroc_any_mean", "aupr_any_mean",
                             "auprg_any_mean")}
    summ.update(val_auprg_any_mean=auprg, auprg_any_mean=auprg)
    (d / "results.json").write_text(json.dumps(
        {"summary": summ, "cv_type": cv, "config": cfg}))


def test_collect_allmod_takes_its_marked_run_only():
    with tempfile.TemporaryDirectory() as tmp:
        sia, out = Path(tmp) / "results", Path(tmp) / "cmp"
        a = sia / "allmod__var_pd_pooled__all_modalities_cv1"
        _run_dir(a, "cv1", 0.61, allmod=True)
        (a / "all_modalities.json").write_text(json.dumps(
            {"modalities": ["a", "b"], "files": ["all_genes_a.pt",
                                                  "all_genes_b.pt"],
             "catalog_size": 3, "absent_optional": ["c"]}))
        (sia / "allmod__var_pd_pooled__latest_allmod_cv1.txt").write_text(
            "results/allmod__var_pd_pooled__all_modalities_cv1\n")
        # a selected combo that scores higher, and an unmarked allmod cv2
        _run_dir(sia / "var_pd_pooled__best_cat_a_b_cv1", "cv1", 0.9, False)
        (sia / "var_pd_pooled__latest_combo_cv1.txt").write_text(
            "results/var_pd_pooled__best_cat_a_b_cv1\n")
        _run_dir(sia / "allmod__var_pd_pooled__all_modalities_cv2", "cv2",
                 0.7, allmod=True)
        cs.discover(str(sia), str(out), "siamese_sl_allmod_pd_pooled",
                    variant="pd:pooled", combo_markers=True,
                    cv_types=["cv1", "cv2"], pca_arm="none", allmod=True)
        rows = sorted(p.name for p in out.iterdir())
        assert rows == ["siamese_sl_allmod_pd_pooled_cv1.json",
                        "siamese_sl_allmod_pd_pooled_selected_cv1.txt"], rows
        s = json.loads((out / rows[0]).read_text())["summary"]
        assert s["auprg_mean"] == 0.61, s
        rec = s["selection_audit"]["all_modalities"]
        assert rec["selection"] == "none" and rec["modalities"] == ["a", "b"] \
            and rec["absent_optional"] == ["c"] and \
            s["selection_audit"]["pca_arm"] == "none", s["selection_audit"]
        assert (out / rows[1]).read_text().strip().endswith(
            "allmod__var_pd_pooled__all_modalities_cv1")
        # the selected row never takes the allmod run, nor the reverse
        out2 = Path(tmp) / "cmp2"
        cs.discover(str(sia), str(out2), "siamese_sl_pd_pooled",
                    variant="pd:pooled", combo_markers=True,
                    cv_types=["cv1"], pca_arm="none")
        s2 = json.loads((out2 / "siamese_sl_pd_pooled_cv1.json")
                        .read_text())["summary"]
        assert s2["auprg_mean"] == 0.9
        # a marker naming a run that does not record all_modalities: no row
        (sia / "allmod__var_pd_pooled__latest_allmod_cv1.txt").write_text(
            "results/allmod__var_pd_pooled__all_modalities_cv1\n")
        cfgp = a / "config.json"
        cfg = json.loads(cfgp.read_text())
        cfg.pop("all_modalities")
        cfgp.write_text(json.dumps(cfg))
        data = json.loads((a / "results.json").read_text())
        data["config"] = cfg
        (a / "results.json").write_text(json.dumps(data))
        out3 = Path(tmp) / "cmp3"
        cs.discover(str(sia), str(out3), "siamese_sl_allmod_pd_pooled",
                    variant="pd:pooled", combo_markers=True,
                    cv_types=["cv1"], allmod=True)
        assert not out3.exists() or not list(out3.iterdir())
        # the pd:per_head sweep skips allmod__ dirs
        _run_dir(sia / "allmod__all_modalities_cv1", "cv1", 0.99, True,
                 cell_lines="per_head")
        _run_dir(sia / "best_cat_a_b_cv1", "cv1", 0.4, False,
                 cell_lines="per_head")
        out4 = Path(tmp) / "cmp4"
        cs.discover(str(sia), str(out4), "siamese_sl", cv_types=["cv1"])
        s4 = json.loads((out4 / "siamese_sl_cv1.json").read_text())["summary"]
        assert s4["auprg_mean"] == 0.4, s4
    assert cs.expected_variant("siamese_sl_allmod") == ("pd", "per_head")
    assert cs.expected_variant("siamese_sl_allmod_bilinear_pooled") == \
        ("bilinear", "pooled")


def test_collect_and_regrade_builds_allmod_rows_only_when_switched_on():
    text = open(os.path.join(_CMP, "slurm",
                             "collect_and_regrade_siamese.sh")).read()
    for v in ("", "_pd_pooled", "_bilinear_per_head", "_bilinear_pooled"):
        assert f"-name 'siamese_sl_allmod{v}_cv[123].json'" in text
        assert f"-name 'siamese_sl_allmod{v}_selected_cv[123].txt'" in text
    i = text.index("SIAMESE_ALLMOD_TAGS=\"\"")
    block = text[i:text.index("# Rows as MODEL|MARKER_STEM|PIN_STEM", i)]
    assert 'if [ "$SIAMESE_ALLMOD" = 1 ]; then' in block \
        and "--allmod" in block and "--combo_markers" in block \
        and '--pca_arm "${SLEUTH_ARM:-none}"' in block, block
    assert '$AMODEL|$AMODEL|-' in text and '[ "$SPIN" != - ] || continue' \
        in text


def test_compare_order_disclosure_and_bootstrap_pairs():
    per = {m: {} for m in ("siamese_sl", "SLMGAE", "siamese_sl_allmod",
                           "siamese_sl_pd_pooled",
                           "siamese_sl_allmod_pd_pooled", "DegreePrior")}
    assert compare.ordered_models(per) == [
        "siamese_sl", "siamese_sl_pd_pooled", "siamese_sl_allmod",
        "siamese_sl_allmod_pd_pooled", "SLMGAE", "DegreePrior"]
    rec = {"selection": "none", "modalities": ["a", "b"], "n_modalities": 2,
           "absent_optional": ["c"]}
    res = {"cv1": {"siamese_sl_allmod": {"selection_audit":
                                         {"all_modalities": rec}}}}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        lines = compare.selection_report(res)
    text = "\n".join(lines)
    assert "NO modality selection" in text and "no selection record" \
        not in text, text
    import bootstrap_ci as bc
    pairs = bc.tier_pairs(list(per), [])
    assert ("siamese_sl", "siamese_sl_allmod") in pairs and \
        ("siamese_sl_pd_pooled", "siamese_sl_allmod_pd_pooled") in pairs \
        and ("siamese_sl_allmod", "SLMGAE") not in pairs, pairs
    # an all-modalities row is compared with its selected combo only
    assert bc.tier_pairs(["siamese_sl", "SLMGAE"], []) == \
        [("siamese_sl", "SLMGAE")]
    assert bc.variant_of("siamese_sl_allmod") == "allmod_pd_per_head" and \
        bc.variant_of("siamese_sl_allmod_pd_pooled") == "allmod_pd_pooled"


# ---------------------------------------------------------------------------
# the orchestrators
# ---------------------------------------------------------------------------

def test_submit_pipeline_submits_it_per_arm_after_the_generation_only():
    import test_slurm_contracts as tsc
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = tsc._stub_tree(tmp)
        base = ("--skip-reset-env", "--best-cat-only")
        jobs0, out0 = tsc._submit(tmp, *base, "--skip-generate",
                                  PCA_ARMS="none joint", ALLMOD_COMBO="0")
        assert not [a for _, _, a in jobs0 if "run_allmod_combo.sh" in a]
        assert "ALLMOD_COMBO=1" not in out0
        jobs, out = tsc._submit(tmp, *base, PCA_ARMS="none joint",
                                ALLMOD_COMBO="1")
        gen = [j for j, _, a in jobs if a.endswith("run_generate_embeddings.sh")]
        assert len(gen) == 1, jobs
        am = [(job[0], job[2], job.arm) for job in jobs
              if job[2].endswith("run_allmod_combo.sh")]
        assert [arm for _, _, arm in am] == ["none", "joint"], am
        for j, a, arm in am:
            assert f"--dependency=afterok:{gen[0]}" in a \
                and "--kill-on-invalid-dep=yes" in a and "--array=0-2" in a, a
        # the arm's printed compare waits for it; the block passes the switch
        for j, _, arm in am:
            assert re.search(rf"_c_{arm}=.*--dependency=afterany:\d+:{j}\b",
                             out), out[-3000:]
        assert "ALLMOD_COMBO=1" in tsc._sp_block(out)[0]
        # the export waits for it too
        pair = [a for _, _, a in jobs if a.endswith("export_pair_predictions.sh")]
        assert pair and all(any(j in a for a in pair) for j, _, _ in am)


def test_root_orchestrator_chains_the_compare_on_it():
    import test_slurm_contracts as tsc
    with tempfile.TemporaryDirectory() as tmp:
        tsc._root_tree(tmp)
        # (tsc._JOBS names run_allmod_combo.sh "allmod" since 2026-11-13)
        assert tsc._JOBS.get("run_allmod_combo.sh") == "allmod"
        rc, out, jobs = tsc._root_run(tmp, "--skip-generate",
                                      ALLMOD_COMBO="1")
        assert rc == 0, out[-3000:]
        assert "allmod/none" in jobs, sorted(jobs)
        # the Step-1 array's dependency (the generation / env reset),
        # never the Step-1 array or the combo
        kind, parents, kill, array, _ = jobs["allmod/none"]
        assert (kind, parents) == jobs["bestcat/none"][:2] and kill \
            and array == "0-2", (jobs["allmod/none"], jobs["bestcat/none"])
        assert "allmod/none" in jobs["compare/none"][1], jobs["compare/none"]
        rc, out, jobs = tsc._root_run(tmp, "--skip-generate",
                                      ALLMOD_COMBO="0")
        assert rc == 0 and "allmod/none" not in jobs, sorted(jobs)


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n==== {len(tests) - failed} passed, {failed} failed ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
