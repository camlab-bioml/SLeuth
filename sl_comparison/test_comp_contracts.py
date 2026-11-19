#!/usr/bin/env python3
"""Deterministic tests for sl_comparison guards.

  C7  external_transfer.py refuses an eval_finetuning dir that cannot be tied
      to the current checkpoints (missing config_used.json, positives-only
      overlap, missing / changed checkpoint_sha256, fold_ids not 0..K-1,
      logit_fold columns other than fold_ids) unless allow_legacy_eval;
      --no_baselines scores SLeuth + DegreePrior only.
  C8  prepare_folds.check_sources / --check_sources compare meta.json's
      source_sha256 with the pair files on disk (and, when the conflict
      attribution ran, the sidecars and the cell-line vocabulary they are
      parsed with), read-only.
  prepare_folds.clear_stale_outputs removes only its own fold-set files.
  p14 compare.py with no usable rows removes the previous comparison.csv and
      its notes and exits 1.
  PCA arms  slurm/pca_arm.sh resolves a job's arm and its two roots from
      siamese_sl/slurm/config.conf (or refuses, setting none); compare.py
      --shared_results_dir takes SLeuth rows only from the arm's directory
      and the rest only from the shared one, --pca_arm labels it; the collect
      + re-grade, the compare, external-transfer and paper-tables jobs read
      and write only their arm's directories (and the shared rows), under
      bash and, where sourced by hand, zsh.

    python3 sl_comparison/test_comp_contracts.py
    # or: python -m pytest sl_comparison/test_comp_contracts.py -q
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_cc_{name}",
                                                  HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


et = _load("external_transfer")
pf = _load("prepare_folds")
cmp = _load("compare")

GENES = ["10", "20", "30", "40", "50"]
ROWS = [("10", "20", .1, "test"), ("10", "30", .3, "test"),
        ("20", "40", .2, "test"), ("30", "50", .5, "test"),
        ("20", "30", .4, "train"), ("40", "50", .6, "train")]


def _raises(fn, exc, *needles):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"{exc.__name__} not raised")


def _variant(root: Path, n_logit=2, cfg_extra=None, write_cfg=True) -> Path:
    """A SLeuth cv3 model dir with one screen/split, two checkpoints and an
    eval_finetuning_cv1 whose config_used.json fingerprints them."""
    (root / "checkpoints").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"cv_type": "cv3"}))
    hashes = {}
    for k in range(2):
        p = root / "checkpoints" / f"fold_{k}_best.pt"
        p.write_bytes(f"weights {k}".encode())
        hashes[f"results/x_cv3/checkpoints/{p.name}"] = hashlib.sha256(
            p.read_bytes()).hexdigest()
    ed = root / "eval_finetuning_cv1"
    ed.mkdir()
    recs = []
    for g1, g2, t, sp in ROWS:
        r = {"gene1_entrez": g1, "gene2_entrez": g2, "target": t,
             "sldb_overlap": False, "split": sp}
        for k in range(n_logit):
            r[f"logit_fold{k}"] = t * (1 if k == 0 else -1) + 0.01 * k
        recs.append(r)
    pd.DataFrame(recs).to_csv(ed / "Toy_predictions.csv", index=False)
    if write_cfg:
        cfg = {"ft_modes": [], "checkpoint_sha256": hashes, "fold_ids": [0, 1],
               "sldb_overlap_sources": ["pos.txt", "neg.txt"]}
        cfg.update(cfg_extra or {})
        (ed / "config_used.json").write_text(json.dumps(cfg))
    return root


def _run(root: Path, v: Path, **kw):
    folds = root / "folds"
    folds.mkdir(exist_ok=True)
    (folds / "genes.txt").write_text("\n".join(GENES) + "\n")
    return et.run([("siamese_sl", v)], [], False, folds / "genes.txt", folds,
                  root / "out", "cv3", ["Toy"], ["cv1"], allow_unstamped=True,
                  **kw)


def test_c7_fingerprint_ok():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        res = _run(root, _variant(root / "v"))
        assert "verified for every" in res["markdown"]


def test_c7_retrained_checkpoint_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _variant(root / "v")
        (v / "checkpoints" / "fold_1_best.pt").write_bytes(b"retrained")
        _raises(lambda: _run(root, v), ValueError, "sha256", "fold_1_best.pt",
                "--allow_legacy_eval")


def test_c7_missing_config_refused_unless_legacy():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _variant(root / "v", write_cfg=False)
        _raises(lambda: _run(root, v), ValueError, "config_used.json missing")
        res = _run(root, v, allow_legacy_eval=True)
        assert "NOT verified (--allow_legacy_eval) for siamese_sl/cv1" in \
            res["markdown"]


def test_c7_positives_only_and_no_hashes_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _variant(root / "v", cfg_extra={"sldb_overlap_sources": ["pos"],
                                            "checkpoint_sha256": {}})
        _raises(lambda: _run(root, v), ValueError, "positives only",
                "--neg_path", "no checkpoint_sha256")


def test_c7_fold_ids_not_contiguous_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _variant(root / "v", cfg_extra={"fold_ids": [0, 2]})
        _raises(lambda: _run(root, v), ValueError, "fold_ids [0, 2]")


def test_c7_logit_columns_must_match_fold_ids():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _variant(root / "v", n_logit=3)
        _raises(lambda: _run(root, v), ValueError, "logit_fold columns")


def test_c7_cli_flag_present():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            et.main(["--help"])
        except SystemExit:
            pass
    assert "--allow_legacy_eval" in buf.getvalue()


def test_no_baselines_flag():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v = _variant(root / "v")
        folds = root / "folds"
        (folds / "cv3" / "fold_0").mkdir(parents=True)
        (folds / "cv3" / "fold_1").mkdir(parents=True)
        (folds / "genes.txt").write_text("\n".join(GENES) + "\n")
        for k in range(2):
            np.save(folds / "cv3" / f"fold_{k}" / "train_pos.npy",
                    np.array([[0, 1]], dtype=np.int32))
        rc = et.main(["--siamese", f"siamese_sl={v}", "--no_baselines",
                      "--folds_dir", str(folds), "--screens", "Toy",
                      "--splits", "cv1", "--out_dir", str(root / "o"),
                      "--allow_unstamped"])
        assert rc == 0
        zs = pd.read_csv(root / "o" / "zero_shot_summary.csv")
        # + the SynLethDB label row, scored where matched_in_sldb has rows
        assert set(zs["model"]) - {et.SLDB_LABEL_NAME} == \
            {"siamese_sl", et.DEGREE_PRIOR_NAME}
        try:
            et.main(["--siamese", f"siamese_sl={v}", "--no_baselines",
                     "--baseline", f"B={root}", "--out_dir", str(root / "p")])
            raise AssertionError("--no_baselines with --baseline accepted")
        except SystemExit as e:
            assert e.code == 2


def _sources(root: Path):
    d = root / "data"
    d.mkdir()
    for name in ("pos", "neg"):
        (d / f"{name}.txt").write_text(f"1 2\n3 4 {name}\n")
        (d / f"{name}.sources.tsv").write_text("h\n1\t2\ts\tK562\t1\n")
    return d / "pos.txt", d / "neg.txt"


def test_c8_check_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pos, neg = _sources(root)
        out = root / "folds"
        out.mkdir()
        assert pf.check_sources(out, pos, neg) == [f"{out / 'meta.json'} "
                                                   f"missing"]
        (out / "meta.json").write_text(json.dumps({"seed": 42}))
        assert "no source_sha256" in pf.check_sources(out, pos, neg)[0]
        fp = pf.source_fingerprints(pos, neg, sidecars=True)
        assert set(fp) == {"pos", "neg", "pos_sources", "neg_sources",
                           "cell_line_vocab"}
        assert set(pf.source_fingerprints(pos, neg, sidecars=False)) == {
            "pos", "neg"}
        (out / "meta.json").write_text(json.dumps({"source_sha256": fp}))
        assert pf.check_sources(out, pos, neg) == []
        # meta.json's conflict counts and conflicts.tsv are a function of the
        # cell-line vocabulary the sidecars are parsed with: a fold set that
        # did not fingerprint it cannot be shown to be fresh, and a changed
        # vocabulary is a difference. (The real vocabulary file lives outside
        # this tmp tree, so tamper with the RECORDED hash, not the file.)
        (out / "meta.json").write_text(json.dumps({"source_sha256": {
            k: v for k, v in fp.items() if k != "cell_line_vocab"}}))
        probs = pf.check_sources(out, pos, neg)
        assert len(probs) == 1 and "lacks 'cell_line_vocab'" in probs[0], probs
        (out / "meta.json").write_text(json.dumps(
            {"source_sha256": dict(fp, cell_line_vocab="0" * 64)}))
        probs = pf.check_sources(out, pos, neg)
        assert len(probs) == 1 and probs[0].startswith("cell_line_vocab"), probs
        (out / "meta.json").write_text(json.dumps({"source_sha256": fp}))
        before = (out / "meta.json").read_bytes()
        pf.sidecar_path(neg).write_text("h\n1\t2\ts\tA549\t1\n")
        probs = pf.check_sources(out, pos, neg)
        assert len(probs) == 1 and probs[0].startswith("neg_sources"), probs
        assert (out / "meta.json").read_bytes() == before  # read-only


def test_c8_cli_exit_codes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pos, neg = _sources(root)
        out = root / "folds"
        out.mkdir()
        (out / "meta.json").write_text(json.dumps(
            {"source_sha256": pf.source_fingerprints(pos, neg, False)}))
        argv = ["prepare_folds.py", "--check_sources", "--out_dir", str(out),
                "--pos_path", str(pos), "--neg_path", str(neg)]
        old, sys.argv = sys.argv, argv
        try:
            pf.main()  # match: returns normally
            pos.write_text("1 2\n")
            try:
                pf.main()
                raise AssertionError("no exit on a changed source")
            except SystemExit as e:
                assert e.code == 1
        finally:
            sys.argv = old
        assert sorted(p.name for p in out.iterdir()) == ["meta.json"]


def _tiny_checkout(root: Path):
    """What prepare_folds.py and the folds job read, with made-up pairs: 120
    genes, 3,000 pairs (a quarter SL, 20 of them also listed as screened
    negatives, so the sidecars and cell_line_vocab.py are fold sources as in
    the real data), cell lines in the sidecars. Returns the environment the
    folds job runs with (the interpreter of this test, a stub `module`)."""
    import shutil
    rng = np.random.default_rng(3)
    (root / "sl_comparison" / "slurm").mkdir(parents=True)
    (root / "siamese_sl").mkdir()
    (root / "data").mkdir()
    (root / "bin").mkdir()
    for f in HERE.glob("*.py"):
        if not f.name.startswith("test_"):
            shutil.copy(f, root / "sl_comparison" / f.name)
    for f in ("run_shared_folds.sh", "folds_manifest.sh", "python_path.sh"):
        shutil.copy(HERE / "slurm" / f, root / "sl_comparison" / "slurm" / f)
    shutil.copy(HERE.parent / "siamese_sl" / "cell_line_vocab.py",
                root / "siamese_sl" / "cell_line_vocab.py")
    (root / "bin" / "module").write_text("#!/bin/bash\nexit 0\n")
    (root / "bin" / "module").chmod(0o755)
    genes = [str(1000 + k) for k in range(120)]
    iu = np.triu_indices(len(genes), 1)
    pick = rng.choice(len(iu[0]), size=3000, replace=False)
    lines = ["K562", "JURKAT", "HELA", "A549", "RPE1"]
    for stem, sel in (("SL_SynLethDB_experimental", pick[:750]),
                      ("SL_SynLethDB_experimental_negatives", pick[730:])):
        pairs = [(genes[iu[0][k]], genes[iu[1][k]]) for k in sel]
        (root / "data" / f"{stem}.txt").write_text(
            "".join(f"{a}\t{b}\n" for a, b in pairs))
        (root / "data" / f"{stem}.sources.tsv").write_text(
            "entrez1\tentrez2\tsources\tcell_lines\tpubmed_ids\n" + "".join(
                f"{a}\t{b}\tHigh Throughput\t{lines[k % 5]}\t1\n"
                for k, (a, b) in enumerate(pairs)))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SLURM_")
           and k not in ("SHARED_FOLDS_FROZEN", "FOLDS_MANIFEST", "SEED")}
    env.update(PATH=str(root / "bin") + os.pathsep + env["PATH"],
               PYTHON_PATH=sys.executable, PYTHONDONTWRITEBYTECODE="1",
               CVS="cv1")
    r = subprocess.run([sys.executable, "sl_comparison/prepare_folds.py",
                        "--seed", "42"], cwd=root, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return env


def _folds_job(root: Path, env, **extra):
    r = subprocess.run(["bash", "sl_comparison/slurm/run_shared_folds.sh"],
                       cwd=root, env=dict(env, **extra), capture_output=True,
                       text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


def test_frozen_folds_job_compares_the_fold_files_with_step0s_manifest():
    """Under the orchestrator (SHARED_FOLDS_FROZEN=1) the folds job reuses the
    fold set only when every file is the one step 0 recorded in
    FOLDS_MANIFEST. The meta.json checks alone compare genes.txt and
    all_pos.npy by COUNT and the arrays by partition digest, so a set that
    differs in gene order (or a whole other self-consistent set) was reused
    and DegreePrior and the shared models ran on it."""
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        env = _tiny_checkout(base)
        mani = base / "sl_comparison" / "slurm" / "logs"
        mani.mkdir()
        mani = mani / "folds_T.sha256"
        r = subprocess.run(
            ["bash", "-c", ". sl_comparison/slurm/folds_manifest.sh && "
             "folds_manifest"], cwd=base, env=env, capture_output=True,
            text=True)
        assert r.returncode == 0 and r.stdout.count("\n") > 90, r.stderr
        mani.write_text(r.stdout)

        def fresh(name):
            root = Path(tmp) / name
            shutil.copytree(base, root)
            return root, str(root / "sl_comparison/slurm/logs/folds_T.sha256")

        def swap_two_genes(root):
            g = root / "sl_comparison" / "folds" / "genes.txt"
            rows = g.read_text().split("\n")
            rows[0], rows[1] = rows[1], rows[0]
            g.write_text("\n".join(rows))

        # the set step 0 built: reused, the control row is written
        root, m = fresh("same")
        rc, out = _folds_job(root, env, SHARED_FOLDS_FROZEN="1",
                             FOLDS_MANIFEST=m)
        assert rc == 0 and "reusing existing shared folds" in out, out[-1500:]
        assert (root / "sl_comparison/results/DegreePrior_cv1.json").exists()
        # two genes swapped (the count meta.json records is unchanged)
        root, m = fresh("swapped")
        swap_two_genes(root)
        rc, out = _folds_job(root, env, SHARED_FOLDS_FROZEN="1",
                             FOLDS_MANIFEST=m)
        assert rc == 1 and "FATAL: shared folds are STALE (fold files differ " \
            "from the set step 0 built" in out, out[-1500:]
        assert not (root / "sl_comparison/results/DegreePrior_cv1.json").exists()
        # ... a file added to the set after step 0
        root, m = fresh("stray")
        (root / "sl_comparison" / "folds" / "notes.txt").write_text("x\n")
        rc, out = _folds_job(root, env, SHARED_FOLDS_FROZEN="1",
                             FOLDS_MANIFEST=m)
        assert rc == 1 and "fold files differ from the set step 0 built" \
            in out, out[-1500:]
        # no record (an empty manifest, or none) is no refusal, as before
        root, m = fresh("empty")
        Path(m).write_text("")
        rc, out = _folds_job(root, env, SHARED_FOLDS_FROZEN="1",
                             FOLDS_MANIFEST=m)
        assert rc == 0 and "reusing existing shared folds" in out, out[-1500:]
        # a standalone run never consults a manifest left in the environment
        root, m = fresh("standalone")
        swap_two_genes(root)
        rc, out = _folds_job(root, env, FOLDS_MANIFEST=m)
        assert rc == 0 and "reusing existing shared folds" in out, out[-1500:]
        # a fold SOURCE changed after step 0, the fold files untouched: the
        # shared models still go ahead, with the WARNING
        root, m = fresh("source")
        with open(root / "siamese_sl" / "cell_line_vocab.py", "a") as f:
            f.write("# edited after step 0\n")
        rc, out = _folds_job(root, env, SHARED_FOLDS_FROZEN="1",
                             FOLDS_MANIFEST=m)
        assert rc == 0 and "WARNING: shared folds are STALE" in out \
            and "byte-identical" in out, out[-1500:]


def test_prepare_folds_replaces_read_only_files_in_writable_dirs():
    """An archive of a frozen fold set unpacked over a checkout leaves
    read-only FILES in writable directories. A regeneration then deleted the
    fold dirs and meta.json and died on open(genes.txt, "w"): it must rebuild
    the same set instead."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "t"
        env = _tiny_checkout(root)
        folds = root / "sl_comparison" / "folds"
        before = {q.relative_to(folds).as_posix(): q.read_bytes()
                  for q in folds.rglob("*") if q.is_file()}
        for q in folds.rglob("*"):
            if q.is_file():
                q.chmod(0o444)
        r = subprocess.run([sys.executable, "sl_comparison/prepare_folds.py",
                            "--seed", "42"], cwd=root, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
        after = {q.relative_to(folds).as_posix(): q.read_bytes()
                 for q in folds.rglob("*") if q.is_file()}
        assert after == before, sorted(set(after) ^ set(before))
        # a fully frozen set (directories too) is still refused untouched
        for q in sorted(folds.rglob("*"), reverse=True):
            q.chmod(0o555 if q.is_dir() else 0o444)
        folds.chmod(0o555)
        try:
            r = subprocess.run(
                [sys.executable, "sl_comparison/prepare_folds.py", "--seed",
                 "42"], cwd=root, env=env, capture_output=True, text=True)
            assert r.returncode != 0 and "chmod -R u+w" in r.stdout + r.stderr
            still = {q.relative_to(folds).as_posix(): q.read_bytes()
                     for q in folds.rglob("*") if q.is_file()}
            assert still == before
        finally:
            for q in [folds] + list(folds.rglob("*")):
                q.chmod(0o755 if q.is_dir() else 0o644)


def test_a_pin_that_is_not_a_file_is_named_not_dropped():
    # collect_and_regrade_siamese.sh: SIAMESE_RESULTS_<cv> pins a run's
    # results.json. A value that is no file (a typo, the run DIRECTORY, a path
    # relative to siamese_sl/) used to be dropped without a word, and the row
    # in the table was the discovered combo. Runs the script's own pin loop,
    # under bash and (the file is also sourced by hand) zsh, for arm none
    # (SLEUTH_CMP unset: the inline default) and arm joint, whose pinned row
    # and marker go into its own sl_comparison/results/pca_joint/.
    import shutil
    text = (HERE / "slurm" / "collect_and_regrade_siamese.sh").read_text()
    a = text.index("# An explicit per-CV pin overrides discovery.")
    loop = text[a:text.index("# Re-grade with the SAME evaluator", a)]
    assert loop.count("\ndone\n") == 1 and "exit" not in loop, loop
    shells = [["bash", "-e", "-c"]]
    if shutil.which("zsh"):
        shells.append(["zsh", "-e", "-o", "SH_WORD_SPLIT", "-c"])
    for sh in shells:
        for cmp_dir in ("sl_comparison/results",
                        "sl_comparison/results/pca_joint"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / cmp_dir).mkdir(parents=True)
                run = root / "siamese_sl" / "results" / "best_cat_x_cv2"
                run.mkdir(parents=True)
                (run / "results.json").write_text("{}")
                py = root / "py"
                py.write_text('#!/bin/bash\necho "$*" >> "$PYLOG"\n')
                py.chmod(0o755)
                env = dict(
                    {k: v for k, v in os.environ.items()
                     if not k.startswith(("SIAMESE_RESULTS", "SLEUTH_"))},
                    PYTHON_PATH=str(py), PYLOG=str(root / "pylog"),
                    CVS="cv1 cv2 cv3",
                    SIAMESE_ROWS="siamese_sl|siamese|SIAMESE_RESULTS "
                                 "siamese_sl_pd_pooled|siamese_sl_pd_pooled|"
                                 "SIAMESE_RESULTS_pd_pooled",
                    SIAMESE_RESULTS_cv1=str(run),               # the directory
                    SIAMESE_RESULTS_cv2=str(run / "results.json"),  # a file
                    SIAMESE_RESULTS_pd_pooled_cv3="results/nope/results.json")
                arm = "none"
                if cmp_dir != "sl_comparison/results":
                    env["SLEUTH_CMP"] = cmp_dir
                    env["SLEUTH_ARM"] = arm = "joint"
                r = subprocess.run(sh + [loop + 'echo "LOOP DONE"\n'],
                                   cwd=root, env=env, capture_output=True,
                                   text=True, timeout=60)
                assert r.returncode == 0 and "LOOP DONE" in r.stdout, \
                    (sh[0], r.returncode, r.stdout, r.stderr)
                errs = [ln for ln in r.stdout.splitlines() if "ERROR" in ln]
                assert errs == [
                    f"ERROR: pin SIAMESE_RESULTS_cv1={run} is not a file "
                    f"(give the path of a",
                    "ERROR: pin SIAMESE_RESULTS_pd_pooled_cv3=results/nope/"
                    "results.json is not a file (give the path of a",
                ], (sh[0], r.stdout)
                assert "siamese_sl/cv1 is the discovered run" in r.stdout \
                    and "siamese_sl_pd_pooled/cv3 is the discovered run" \
                    in r.stdout, r.stdout
                # the valid pin still wins and refreshes its marker, both in
                # the arm's directory; no other call
                calls = (root / "pylog").read_text().splitlines()
                assert len(calls) == 1 \
                    and f"--siamese_results {run}/results.json" in calls[0] \
                    and "--cv_type cv2" in calls[0] \
                    and f"--pca_arm {arm} " in calls[0] \
                    and f"--out {cmp_dir}/siamese_sl_cv2.json" in calls[0], \
                    calls
                marker = root / cmp_dir / "siamese_selected_cv2.txt"
                assert marker.read_text().strip() == str(run)
                assert sorted(p.name for p in (root / cmp_dir).iterdir()) \
                    == ["siamese_selected_cv2.txt"], cmp_dir
                assert sorted(p.name for p in
                              (root / "sl_comparison/results").iterdir()) \
                    == ([Path(cmp_dir).name] if "pca_" in cmp_dir else
                        ["siamese_selected_cv2.txt"])


def test_compare_job_summarises_the_rows_of_the_variants_it_collected():
    # run_compare.sh's closing "rows: X of Y present / ABSENT" lines. A
    # hand-submitted compare exports no SIAMESE_VARIANTS_LIST, and the summary
    # then knew only pd:per_head ("15 of 18" on a 24-row table, the other
    # variants' absent rows never named). Runs the script's own block.
    text = (HERE / "slurm" / "run_compare.sh").read_text()
    a = text.index('ROW_VARIANTS=""\n')
    block = text[a:text.index("\nROWS\n", a) + len("\nROWS\n")]
    four = "BASELINE pd_pooled bilinear_per_head bilinear_pooled"
    have = ["siamese_sl_cv1", "siamese_sl_bilinear_per_head_cv1",
            "siamese_sl_bilinear_per_head_nocell_cv1", "SLMGAE_cv1",
            "siamese_sl_pd_pooled_cv3"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sl_comparison" / "results").mkdir(parents=True)
        for n in have:
            (root / "sl_comparison" / "results" / f"{n}.json").write_text("{}")

        def summary(tags, cvs="cv1 cv2 cv3", **env):
            e = {k: v for k, v in os.environ.items()
                 if k != "SIAMESE_VARIANTS_LIST"
                 and not k.startswith("SLEUTH_")}
            e.update(PYTHON_PATH=sys.executable, CVS=cvs,
                     SIAMESE_ALL_TAGS=tags, **env)
            r = subprocess.run(["bash", "-e", "-c", block + 'echo "DONE"\n'],
                               cwd=root, env=e, capture_output=True,
                               text=True, timeout=60)
            assert r.returncode == 0 and "DONE" in r.stdout, \
                (r.stdout, r.stderr)
            return r.stdout

        out = summary(four)                      # by hand: config.conf's four
        assert "--- rows: 5 of 30 present " \
            "(sl_comparison/results/<row>_<cv>.json) ---" in out, out
        for absent in ("siamese_sl_pd_pooled/cv1", "siamese_sl_nocell/cv1",
                       "siamese_sl_bilinear_per_head_nocell/cv2",
                       "siamese_sl_bilinear_pooled/cv3", "NSF4SL/cv2"):
            assert absent in out, (absent, out)
        assert "siamese_sl_pd_pooled_nocell" not in out \
            and "siamese_sl_bilinear_pooled_nocell" not in out, out
        assert "--- rows: 2 of 18 present" in summary("BASELINE")
        assert "--- rows: 5 of 20 present" in summary(four, cvs="cv1 cv3")
        # the orchestrated run's exported list decides, as before
        out = summary(four, SIAMESE_VARIANTS_LIST="pd:per_head bilinear:pooled")
        assert "--- rows: 2 of 21 present" in out, out
        out = summary("BASELINE", SIAMESE_VARIANTS_LIST="bilinear:pooled")
        assert "--- rows: 1 of 15 present" in out, out


def test_compare_job_counts_an_arms_rows_in_its_own_directory():
    # Per PCA arm the closing summary looks for the arm's SLeuth rows (and
    # *_nocell) in its directory, SLEUTH_CMP, and for the shared models' rows
    # in sl_comparison/results: arm none's SLeuth rows there are not arm
    # joint's, and a shared-model name in the arm's directory is no shared row.
    text = (HERE / "slurm" / "run_compare.sh").read_text()
    a = text.index('ROW_VARIANTS=""\n')
    block = text[a:text.index("\nROWS\n", a) + len("\nROWS\n")]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        res = root / "sl_comparison" / "results"
        (res / "pca_joint").mkdir(parents=True)
        for n in ("siamese_sl_cv1", "siamese_sl_nocell_cv1",   # arm none's
                  "siamese_sl_cv2", "SLGNN_cv1", "DegreePrior_cv1"):
            (res / f"{n}.json").write_text("{}")
        for n in ("siamese_sl_cv1", "SLMGAE_cv1"):              # arm joint's
            (res / "pca_joint" / f"{n}.json").write_text("{}")

        def summary(**env):
            e = {k: v for k, v in os.environ.items()
                 if k != "SIAMESE_VARIANTS_LIST"
                 and not k.startswith("SLEUTH_")}
            e.update(PYTHON_PATH=sys.executable, CVS="cv1 cv2",
                     SIAMESE_ALL_TAGS="BASELINE", **env)
            r = subprocess.run(["bash", "-e", "-c", block + 'echo "DONE"\n'],
                               cwd=root, env=e, capture_output=True,
                               text=True, timeout=60)
            assert r.returncode == 0 and "DONE" in r.stdout, \
                (r.stdout, r.stderr)
            return r.stdout

        out = summary(SLEUTH_CMP="sl_comparison/results/pca_joint")
        assert "--- rows: 3 of 12 present (SLeuth rows sl_comparison/" \
            "results/pca_joint/<row>_<cv>.json, shared sl_comparison/" \
            "results/<row>_<cv>.json) ---" in out, out
        for absent in ("siamese_sl/cv2", "siamese_sl_nocell/cv1",
                       "SLMGAE/cv1", "NSF4SL/cv2"):
            assert absent in out, (absent, out)
        assert "siamese_sl/cv1 " not in out and "SLGNN/cv1" not in out, out
        out = summary()                                  # arm none
        assert "--- rows: 5 of 12 present " \
            "(sl_comparison/results/<row>_<cv>.json) ---" in out, out


def _minimal_env(bin_dir=None, **extra):
    """PATH (bin_dir first), HOME, TMPDIR and `extra`: no SLURM_*, PCA_*,
    SIAMESE_* or other setting of the calling shell reaches config.conf."""
    env = {"PATH": (f"{bin_dir}{os.pathsep}" if bin_dir else "")
           + os.environ["PATH"],
           "HOME": os.environ.get("HOME", "/tmp"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    if os.environ.get("TMPDIR"):
        env["TMPDIR"] = os.environ["TMPDIR"]
    env.update(extra)
    return env


def _result_literals(script):
    """The non-comment lines of slurm/<script> that spell a results root."""
    text = (HERE / "slurm" / script).read_text()
    return [ln.strip() for ln in text.splitlines()
            if not ln.lstrip().startswith("#")
            and ("sl_comparison/results" in ln or "siamese_sl/results" in ln)]


def _shells():
    """bash (as on PATH: 3.2 on the laptop, 5.2 when its dir is prepended)
    and, when installed, zsh: the files a hand session sources."""
    import shutil
    return ["bash"] + (["zsh"] if shutil.which("zsh") else [])


def test_pca_arm_sh_resolves_the_arm_from_config_conf():
    # sl_comparison/slurm/pca_arm.sh: the arm of a comparison job and its two
    # roots, from siamese_sl/slurm/config.conf (read-only; sourced here from
    # this checkout's root, under bash and zsh). A setting config.conf refuses
    # leaves no root set and returns 1 with config.conf's message, which stops
    # a set -e caller before it moves anything.
    root = HERE.parent
    show = ('. sl_comparison/slurm/pca_arm.sh; rc=$?; echo "RC=$rc|'
            '${SLEUTH_ARM-UNSET}|${SLEUTH_RESULTS-UNSET}|${SLEUTH_CMP-UNSET}|'
            '${SLEUTH_PCA_ARMS-UNSET}|${SLEUTH_PCA_LABEL-UNSET}"')
    for sh in _shells():
        def run(script=show, **env):
            r = subprocess.run([sh, "-c", script], cwd=root,
                               env=_minimal_env(**env), capture_output=True,
                               text=True, timeout=60)
            return r.returncode, r.stdout + r.stderr

        rc, out = run()
        assert rc == 0 and out.splitlines() == [
            "  PCA arm: none (no PCA): siamese_sl/results/, "
            "sl_comparison/results/",
            "RC=0|none|siamese_sl/results|sl_comparison/results|"
            "none joint|no PCA"], (sh, out)
        for arm, label in (
                ("joint", "PCA on the concatenation of all modalities "
                          "(variance 0.8)"),
                ("per_modality", "PCA within each modality (variance 0.8)"),
                ("both", "PCA within each modality (variance 0.8), then on "
                         "their concatenation (variance 0.8)")):
            listed = ["both", "none"] + ([arm] if arm != "both" else [])
            rc, out = run(PCA_ARM=arm, PCA_ARMS=" ".join(listed))
            arms = " ".join(a for a in ("none", "joint", "per_modality",
                                        "both") if a in listed)
            assert rc == 0 and out.splitlines() == [
                f"  PCA arm: {arm} ({label}): siamese_sl/results/pca_{arm}/, "
                f"sl_comparison/results/pca_{arm}/",
                f"RC=0|{arm}|siamese_sl/results/pca_{arm}|"
                f"sl_comparison/results/pca_{arm}|{arms}|{label}"], (sh, out)
        rc, out = run(PCA_ARM="joint", PCA_JOINT_DIM="300")
        assert "(PCA on the concatenation of all modalities (300 " \
            "components))" in out, (sh, out)
        unset = "RC=1|UNSET|UNSET|UNSET|UNSET|UNSET"
        for env, said in (
                ({"PCA_ARM": "bogus"}, "ERROR: PCA_ARM='bogus' (want none, "
                                       "joint, per_modality or both)"),
                ({"POST_PCA_VARIANCE": "0.8"},
                 "ERROR: POST_PCA_VARIANCE / POST_PCA_DIM are replaced by "
                 "the PCA arms"),
                ({"PCA_ARMS": "joint joint"},
                 "ERROR: PCA_ARMS='joint joint' lists joint twice")):
            rc, out = run(**env)
            assert unset in out and "refused this job's settings" in out \
                and said in out and "PCA arm: " not in out, (sh, env, out)
        # roots left from an earlier arm in this shell are cleared, too
        rc, out = run("SLEUTH_CMP=sl_comparison/results/pca_both; " + show,
                      PCA_ARM="bogus")
        assert unset in out, (sh, out)
        # a PCA_ARM this shell did not export never reached config.conf
        rc, out = run("PCA_ARM=joint; " + show)
        assert unset in out and "PCA_ARM=joint in this shell" in out \
            and "export PCA_ARM" in out, (sh, out)
        # set -e: the caller stops at the refusal
        rc, out = run("set -e; . sl_comparison/slurm/pca_arm.sh; "
                      "echo NOT-REACHED", PCA_ARM="bogus")
        assert rc != 0 and "NOT-REACHED" not in out, (sh, out)
    # not from the repo root: no config.conf there
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["bash", "-c", f". {root}/sl_comparison/slurm/"
                            f"pca_arm.sh; echo RC=$?"], cwd=tmp,
                           env=_minimal_env(), capture_output=True,
                           text=True, timeout=60)
        assert "RC=1" in r.stdout and "no siamese_sl/slurm/config.conf" \
            in r.stdout, r.stdout


def test_compare_lists_the_bilinear_probe_as_a_shared_baseline():
    # The bilinear probe (siamese_sl/slurm/run_probe.sh, reviewer 22DY) is a
    # shared row, like DegreePrior: not is_siamese_row (so
    # collect_and_regrade_siamese.sh neither moves nor regrades it), read from
    # the shared directory for every arm, ordered between the peers and the
    # control, and its per-fold size labels it.
    import contextlib
    import io
    assert not cmp.is_siamese_row("BilinearProbe")
    assert cmp.MODEL_ORDER.index("NSF4SL") < \
        cmp.MODEL_ORDER.index("BilinearProbe") < \
        cmp.MODEL_ORDER.index("DegreePrior")
    order = cmp.ordered_models({m: {} for m in (
        "DegreePrior", "BilinearProbe", "siamese_sl_pd_pooled", "SLGNN",
        "siamese_sl")})
    assert order == ["siamese_sl", "siamese_sl_pd_pooled", "SLGNN",
                     "BilinearProbe", "DegreePrior"], order
    crs = (HERE / "slurm" / "collect_and_regrade_siamese.sh").read_text()
    assert "BilinearProbe" not in crs and "-name 'siamese_sl_cv[123].json'" \
        in crs

    def row(d, model, cv, auprg, **kw):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{model}_{cv}.json").write_text(json.dumps({"summary": dict(
            model=model, cv_type=cv, auprg_mean=auprg, auprg_std=0.01,
            **kw)}))

    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results"
        pj = res / "pca_joint"
        row(res, "BilinearProbe", "cv1", 0.4, trainable_params=149378,
            trainable_params_fold_mean=149378.0,
            trainable_params_fold_std=0.0, trainable_params_fold_n=5)
        row(res, "DegreePrior", "cv1", 0.2, trainable_params=0)
        row(res, "siamese_sl_pd_pooled", "cv1", 0.6)
        row(pj, "siamese_sl_pd_pooled", "cv1", 0.7)
        row(pj, "BilinearProbe", "cv1", 0.0)                # a stray copy
        for d in (res, pj):
            results, dropped_arm, _ = cmp.load_split_results(d, res)
            assert results["cv1"]["BilinearProbe"]["auprg_mean"] == 0.4
            assert ("BilinearProbe" in {m for m, _, _ in dropped_arm}) == \
                (d == pj)
        lab = cmp.model_label("BilinearProbe",
                              results["cv1"]["BilinearProbe"])
        assert lab == "BilinearProbe(149,378 mean of 5 folds)", lab
        csv_p = res / "comparison.csv"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmp.main(["--results_dir", str(res), "--csv", str(csv_p)])
        assert rc == 0, buf.getvalue()
        rows = pd.read_csv(csv_p)
        got = rows[(rows["model"] == "BilinearProbe")]
        assert float(got[got["metric"] == "auprg"]["mean"].iloc[0]) == 0.4
        assert float(got[got["metric"] == "trainable_params_fold"][
            "mean"].iloc[0]) == 149378.0
        table = buf.getvalue()
        i, j = (table.index("BilinearProbe(149,378"),
                table.index("DegreePrior(0)"))
        assert table.index("siamese_sl_pd_pooled") < i < j, table


def test_compare_reads_an_arms_sleuth_rows_and_the_shared_rows_beside():
    # compare.py --shared_results_dir: SLeuth rows (siamese_sl*) only from
    # --results_dir (the arm's), every other row only from the shared
    # directory. Arm none's SLeuth rows sit in that shared directory under the
    # same names, and a stray copy of a shared row in the arm's directory is
    # never the table's. --pca_arm labels the table, the notes' first line and
    # a trailing CSV column; without it the CSV header is unchanged.
    import contextlib
    import io
    label = "PCA on the concatenation of all modalities (variance 0.8)"

    def row(d, model, cv, auprg, **kw):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{model}_{cv}.json").write_text(json.dumps({"summary": dict(
            model=model, cv_type=cv, auprg_mean=auprg, auprg_std=0.01,
            **kw)}))

    def main(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmp.main([str(a) for a in argv])
        return rc, buf.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results"
        pj = res / "pca_joint"
        for cv in ("cv1", "cv3"):
            row(res, "SLGNN", cv, 0.5, trainable_params=10)
        row(res, "DegreePrior", "cv1", 0.2, trainable_params=0)
        row(res, "siamese_sl", "cv1", 0.6)                  # arm none's
        row(res, "siamese_sl_nocell", "cv1", 0.55)          # arm none's
        row(pj, "siamese_sl", "cv1", 0.9)
        row(pj, "SLGNN", "cv1", 0.0)                        # a stray copy
        csv_p = pj / "tables" / "comparison.csv"            # no such dir yet
        rc, out = main("--results_dir", pj, "--shared_results_dir", res,
                       "--csv", csv_p, "--pca_arm", "joint",
                       "--pca_label", label)
        assert rc == 0, out
        rows = pd.read_csv(csv_p)
        assert list(rows.columns) == ["cv", "model", "metric", "mean", "std",
                                      "pca_arm"], list(rows.columns)
        assert set(rows["pca_arm"]) == {"joint"}
        got = {(r.cv, r.model): float(r["mean"]) for _, r in
               rows[rows["metric"] == "auprg"].iterrows()}
        assert got == {("cv1", "siamese_sl"): 0.9, ("cv1", "SLGNN"): 0.5,
                       ("cv3", "SLGNN"): 0.5, ("cv1", "DegreePrior"): 0.2}, \
            got
        lines = out.splitlines()
        i = next(k for k, ln in enumerate(lines)
                 if "CV1 (edge) - model comparison" in ln)
        assert lines[i + 1].strip() == f"PCA arm: joint ({label})", \
            lines[i:i + 2]
        assert any("WARNING: 1 non-SLeuth row(s)" in ln
                   and "SLGNN/cv1 (SLGNN_cv1.json)" in ln for ln in lines), out
        assert any("not used: 2 SLeuth row(s)" in ln
                   and "siamese_sl (cv1), siamese_sl_nocell (cv1)" in ln
                   for ln in lines), out
        notes = (csv_p.parent / "comparison_notes.txt").read_text()
        head = notes.splitlines()
        assert head[0] == f"PCA arm: joint ({label})", head[:3]
        assert head[1].startswith(f"rows: 1 SLeuth row(s) (siamese_sl*) "
                                  f"from {pj}; 3 shared-model row(s)"), head
        assert head[3] == "Notes for comparison.csv", head[:5]

        # arm none: one directory for both is exactly the plain table
        a, b = res / "a" / "comparison.csv", res / "b" / "comparison.csv"
        assert main("--results_dir", res, "--shared_results_dir", res,
                    "--csv", a)[0] == 0
        assert main("--results_dir", res, "--csv", b)[0] == 0
        assert a.read_text() == b.read_text()
        assert a.read_text().splitlines()[0] == "cv,model,metric,mean,std"
        assert b.with_name("comparison_notes.txt").read_text().startswith(
            "Notes for comparison.csv\n")
        rc, out = main("--results_dir", res, "--shared_results_dir", res,
                       "--csv", a, "--pca_arm", "none", "--pca_label",
                       "no PCA")
        rows = pd.read_csv(a)
        assert list(rows.columns)[-1] == "pca_arm" \
            and set(rows["pca_arm"]) == {"none"}
        assert {(r.cv, r.model) for _, r in rows.iterrows()} == {
            ("cv1", "SLGNN"), ("cv3", "SLGNN"), ("cv1", "DegreePrior"),
            ("cv1", "siamese_sl"), ("cv1", "siamese_sl_nocell")}
        assert "non-SLeuth row(s)" not in out and "  not used:" not in out, \
            out
        assert a.with_name("comparison_notes.txt").read_text().startswith(
            "PCA arm: none (no PCA)\nrows: 2 SLeuth and 3 shared-model "
            f"row(s), all from {res}\n\nNotes for comparison.csv\n")

        # nothing in either directory: both named, exit 1
        empty = Path(tmp) / "empty"
        empty.mkdir()
        rc, out = main("--results_dir", empty / "pca_both",
                       "--shared_results_dir", empty, "--csv",
                       empty / "pca_both" / "comparison.csv")
        assert rc == 1 and f"No results in {empty / 'pca_both'} (SLeuth " \
            f"rows) or {empty} (shared-model rows)" in out, out
        # a label without its arm is a usage error
        with contextlib.redirect_stderr(io.StringIO()):
            _raises(lambda: cmp.main(["--results_dir", str(res),
                                      "--pca_label", "x"]), SystemExit)
            _raises(lambda: cmp.main(["--results_dir", str(res),
                                      "--pca_arm", "pca_joint"]), SystemExit)


def _sh_tree(root: Path, *files):
    """A checkout copy for the job scripts: config.conf (+ the given files,
    paths from the repo root) and stubs bin/module and bin/py, the logging
    PYTHON_PATH (each call's arguments to $PYLOG; `py --help` names
    --no_baselines; `paper_tables.py ... --out_dir D` writes D/README.md with
    $PT_VERDICT as line 3, and D/tab_tiers.csv when $PT_TIERS is set, and
    exits $PT_RC; evaluate_model.py exits 0, bootstrap_ci.py $BOOT_RC;
    everything else runs this interpreter)."""
    import shutil
    for rel in ("siamese_sl/slurm/config.conf",
                "sl_comparison/slurm/pca_arm.sh",
                "sl_comparison/slurm/python_path.sh") + files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE.parent / rel, root / rel)
    b = root / "bin"
    b.mkdir()
    (b / "module").write_text("#!/bin/bash\nexit 0\n")
    (b / "py").write_text(
        '#!/bin/bash\n'
        'echo "$*" >> "$PYLOG"\n'
        'case " $* " in *" --help "*)\n'
        '  echo "  --no_baselines"; exit 0 ;;\n'
        'esac\n'
        'case "$1" in\n'
        '  sl_comparison/evaluate_model.py|sl_comparison/external_transfer.py)'
        ' exit 0 ;;\n'
        '  sl_comparison/paper_tables.py)\n'
        '    out=""; prev=""\n'
        '    for a in "$@"; do [ "$prev" = --out_dir ] && out="$a"; '
        'prev="$a"; done\n'
        '    mkdir -p "$out"\n'
        '    printf "# t\\n\\n%s\\n" "${PT_VERDICT:-PAPER COMPLETE}" '
        '> "$out/README.md"\n'
        '    [ -z "${PT_TIERS:-}" ] || echo x > "$out/tab_tiers.csv"\n'
        '    exit "${PT_RC:-0}" ;;\n'
        '  sl_comparison/bootstrap_ci.py) exit "${BOOT_RC:-0}" ;;\n'
        '  -c) exit 0 ;;\n'
        'esac\n'
        f'exec "{sys.executable}" "$@"\n')
    for f in b.iterdir():
        f.chmod(0o755)
    return b


def _combo(d: Path, cv, auprg, with_preds=True, cell_lines="per_head"):
    """A pd:<cell_lines> combo run as collect_siamese.py reads it (2 folds;
    a combo of another variant than the row's is dropped there)."""
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "cv_type": cv,
        "config": {"kernel": "pd", "cell_lines": cell_lines,
                   "use_cell_lines": True, "select_metric": "auprg_any",
                   "embeddings_paths": ["../data/all_genes_go2vec.pt"]},
        "summary": {"auroc_any_mean": 0.8, "auroc_any_std": 0.01,
                    "auprg_any_mean": auprg, "auprg_any_std": 0.02,
                    "val_auprg_any_mean": auprg},
        "fold_metrics": [{"total_params": 100}, {"total_params": 102}]}))
    if with_preds:
        for k in (0, 1):
            (d / f"fold_{k}_{cv}_predictions.npy").write_text("x")


def test_collect_and_regrade_keeps_to_its_pca_arm():
    # The sourced collect + re-grade of ONE arm: rows, markers and stale
    # copies only in the arm's sl_comparison directory, combos only from its
    # siamese root. Arm none's directory is the parent of arm joint's: its
    # rows must never move when arm joint's are rebuilt, nor the reverse; the
    # shared models' rows are never touched. Under bash and zsh (sourced by
    # hand there), and with PCA_ARM exported but no pca_arm.sh, refused
    # before anything moves. A second variant (pd:pooled) goes through the
    # per-variant collect, the loop that writes 9 of a default run's 12
    # SLeuth rows per arm: its row is its own arm's MARKED combo (an older
    # unmarked one scores higher) and names the arm (2026-10-21: no test ran
    # that loop, so dropping its --pca_arm or --combo_markers passed).
    text = (HERE / "slurm" / "collect_and_regrade_siamese.sh").read_text()
    code = [ln for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    for ln in code:     # every comparison / siamese path is the arm's
        assert "sl_comparison/results" not in ln.replace(
            "${SLEUTH_CMP:-sl_comparison/results}", ""), ln
        assert "siamese_sl/results" not in ln or "SLEUTH_RESULTS:-" in ln, ln
    assert "SIAMESE_RESULTS_DIR=\"${SLEUTH_RESULTS:-${SIAMESE_DIR:-" \
        "siamese_sl/results}}\"" in text
    for sh in _shells():
        for arm in ("joint", "none", "refused"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                b = _sh_tree(root, "sl_comparison/slurm/"
                             "collect_and_regrade_siamese.sh",
                             "sl_comparison/collect_siamese.py",
                             "sl_comparison/model_params.py")
                sia, res = root / "siamese_sl/results", \
                    root / "sl_comparison/results"
                _combo(sia / "best_cat_ea_cv1", "cv1", 0.70)
                _combo(sia / "pca_joint" / "best_cat_ea_cv1", "cv1", 0.42)
                (sia / "latest_combo_cv1.txt").write_text(
                    "results/best_cat_ea_cv1\n")
                (sia / "pca_joint" / "latest_combo_cv1.txt").write_text(
                    "results/pca_joint/best_cat_ea_cv1\n")
                # pd:pooled in both roots: the marked combo, and an older
                # unmarked one that scores higher
                for d, auprg in ((sia, 0.71), (sia / "pca_joint", 0.43)):
                    v = d / "var_pd_pooled__best_cat_ea_cv1"
                    _combo(v, "cv1", auprg, cell_lines="pooled")
                    (d / "var_pd_pooled__latest_combo_cv1.txt").write_text(
                        str(v.relative_to(root / "siamese_sl")) + "\n")
                    _combo(d / "var_pd_pooled__best_cat_old_cv1", "cv1", 0.99,
                           cell_lines="pooled")
                for k in (0, 1):
                    (root / f"sl_comparison/folds/cv1/fold_{k}").mkdir(
                        parents=True)
                before = {}
                for d in (res, res / "pca_joint"):
                    d.mkdir(parents=True, exist_ok=True)
                    for name in ("siamese_sl_cv1.json",
                                 "siamese_selected_cv1.txt"):
                        (d / name).write_text(f"previous {d.name}\n")
                (res / "SLGNN_cv1.json").write_text("shared\n")
                for p in res.rglob("*"):
                    if p.is_file():
                        before[p] = p.read_text()
                env = _minimal_env(b, PYTHON_PATH=str(b / "py"),
                                   PYLOG=str(root / "pylog"), CVS="cv1",
                                   SIAMESE_VARIANTS_LIST="pd:per_head "
                                                         "pd:pooled")
                if arm == "joint":
                    env["PCA_ARM"] = "joint"
                    script = (". sl_comparison/slurm/pca_arm.sh && . "
                              "sl_comparison/slurm/"
                              "collect_and_regrade_siamese.sh")
                elif arm == "none":
                    script = (". sl_comparison/slurm/"
                              "collect_and_regrade_siamese.sh")
                else:
                    env["PCA_ARM"] = "joint"
                    script = (". sl_comparison/slurm/"
                              "collect_and_regrade_siamese.sh")
                r = subprocess.run([sh, "-c", script + '; echo "RC=$?"'],
                                   cwd=root, env=env, capture_output=True,
                                   text=True, timeout=120)
                out = r.stdout + r.stderr
                calls = ((root / "pylog").read_text().splitlines()
                         if (root / "pylog").exists() else [])
                after = {p: p.read_text() for p in res.rglob("*")
                         if p.is_file()}
                if arm == "refused":
                    assert "RC=1" in out and "ERROR: PCA_ARM=joint, but " \
                        "this shell's roots are PCA arm none's:" in out, \
                        (sh, out)
                    assert after == before and calls == [], (sh, out)
                    continue
                assert "RC=0" in out, (sh, out)
                mine = res / "pca_joint" if arm == "joint" else res
                other = res if arm == "joint" else res / "pca_joint"
                combo = ("siamese_sl/results/pca_joint/best_cat_ea_cv1"
                         if arm == "joint" else
                         "siamese_sl/results/best_cat_ea_cv1")
                # the other arm's rows, markers and the shared row: untouched
                for p, t in before.items():
                    if p.parent == other or p.name == "SLGNN_cv1.json":
                        assert after.get(p) == t, (sh, arm, p)
                assert not (other / "stale_siamese").exists(), (sh, arm)
                stale = sorted(p.name.split(".")[0] for p in
                               (mine / "stale_siamese").iterdir())
                assert stale == ["siamese_selected_cv1", "siamese_sl_cv1"], \
                    (sh, arm, stale)
                s = json.loads((mine / "siamese_sl_cv1.json").read_text())[
                    "summary"]
                assert s["auprg_mean"] == (0.42 if arm == "joint" else 0.70)
                assert s["trainable_params_fold_mean"] == 101.0, s
                assert (mine / "siamese_selected_cv1.txt").read_text() \
                    .strip() == combo, (sh, arm)
                rel = mine.relative_to(root)
                assert any("collect_siamese.py --discover_dir "
                           f"{Path(combo).parent} --variant pd:per_head" in c
                           and f"--pca_arm {arm} --out_dir {rel}" in c
                           for c in calls), calls
                # the row says its arm (label only; the re-grade carries
                # selection_audit, evaluate_model.CARRY_KEYS)
                assert s["selection_audit"]["pca_arm"] == arm, s
                # the variant's row too: its own arm's marked combo, named
                v = json.loads((mine / "siamese_sl_pd_pooled_cv1.json")
                               .read_text())["summary"]
                assert v["auprg_mean"] == (0.43 if arm == "joint" else 0.71), \
                    (sh, arm, v)
                assert (v.get("selection_audit") or {}).get("pca_arm") \
                    == arm, (sh, arm, v)
                assert any("--model_name siamese_sl_pd_pooled "
                           "--variant pd:pooled" in c
                           and f"--pca_arm {arm} --out_dir {rel}" in c
                           for c in calls), calls
                assert not (other / "siamese_sl_pd_pooled_cv1.json").exists()
                assert any(c.startswith("sl_comparison/evaluate_model.py")
                           and f"--pred_dir {combo}" in c
                           and f"--out {rel}/siamese_sl_cv1.json" in c
                           for c in calls), calls
                assert f"--- PCA arm {arm}: combos from " \
                    f"{Path(combo).parent}/, rows into {rel}/ ---" in out, out


def test_compare_job_tables_one_arm_beside_the_shared_rows():
    # The whole run_compare.sh (set -e) for arm joint: pca_arm.sh, the
    # collect + re-grade, compare.py on the arm's SLeuth rows and the shared
    # rows of sl_comparison/results, the closing rows summary. A setting
    # config.conf refuses stops it before any row or table moves.
    assert _result_literals("run_compare.sh") == [
        "--shared_results_dir sl_comparison/results \\",
        '"${SLEUTH_CMP:-sl_comparison/results}" sl_comparison/results '
        "<<'ROWS'"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = _sh_tree(root, "sl_comparison/slurm/run_compare.sh",
                     "sl_comparison/slurm/collect_and_regrade_siamese.sh",
                     "sl_comparison/collect_siamese.py",
                     "sl_comparison/model_params.py",
                     "sl_comparison/compare.py")
        sia, res = root / "siamese_sl/results", root / "sl_comparison/results"
        _combo(sia / "pca_joint" / "best_cat_ea_cv1", "cv1", 0.42)
        (sia / "pca_joint" / "latest_combo_cv1.txt").write_text(
            "results/pca_joint/best_cat_ea_cv1\n")
        for k in (0, 1):
            (root / f"sl_comparison/folds/cv1/fold_{k}").mkdir(parents=True)
        res.mkdir(parents=True)
        for model, auprg in (("SLGNN", 0.5), ("siamese_sl", 0.6)):
            (res / f"{model}_cv1.json").write_text(json.dumps({"summary": {
                "model": model, "cv_type": "cv1", "auprg_mean": auprg,
                "auprg_std": 0.1}}))
        (res / "comparison.csv").write_text("arm none's\n")

        def job(**env):
            r = subprocess.run(
                ["bash", "sl_comparison/slurm/run_compare.sh"], cwd=root,
                env=_minimal_env(b, PYTHON_PATH=str(b / "py"),
                                 PYLOG=str(root / "pylog"), CVS="cv1",
                                 SIAMESE_VARIANTS_LIST="pd:per_head", **env),
                capture_output=True, text=True, timeout=180)
            return r.returncode, r.stdout + r.stderr

        rc, out = job(PCA_ARM="joint")
        assert rc == 0 and "--- four-model comparison table (PCA arm " \
            "joint) ---" in out, out
        rows = pd.read_csv(res / "pca_joint" / "comparison.csv")
        got = {r.model: float(r["mean"]) for _, r in
               rows[rows["metric"] == "auprg"].iterrows()}
        assert got == {"siamese_sl": 0.42, "SLGNN": 0.5}, got
        assert set(rows["pca_arm"]) == {"joint"}
        assert "--- rows: 2 of 6 present (SLeuth rows sl_comparison/results/" \
            "pca_joint/<row>_<cv>.json, shared sl_comparison/results/" \
            "<row>_<cv>.json) ---" in out, out
        assert (res / "comparison.csv").read_text() == "arm none's\n"
        assert json.loads((res / "siamese_sl_cv1.json").read_text())[
            "summary"]["auprg_mean"] == 0.6
        before = {p: p.read_bytes() for p in res.rglob("*") if p.is_file()}
        rc, out = job(PCA_ARM="joint", POST_PCA_DIM="64")
        assert rc == 1 and "refused this job's settings" in out \
            and "--- four-model" not in out, out
        assert {p: p.read_bytes() for p in res.rglob("*")
                if p.is_file()} == before


def test_external_transfer_job_reads_only_its_arms_combos():
    # run_external_transfer.sh for arm joint: the markers in its siamese root,
    # a marker naming a combo outside that root (arm none's spelling,
    # results/<combo>) refused with a message, output and rm -rf only in its
    # own sl_comparison/results/pca_joint/external_transfer/. The one literal
    # results path left is the shared models' rows' gate.
    assert _result_literals("run_external_transfer.sh") == [
        'ROW="sl_comparison/results/${BN}_$MODEL_CV.json"']
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = _sh_tree(root, "sl_comparison/slurm/run_external_transfer.sh")
        sia = root / "siamese_sl/results"
        for d in (sia / "pca_joint" / "best_cat_ea_cv3",
                  sia / "var_pd_pooled__best_cat_ea_cv3"):
            for s in ("cv1", "cv2", "cv3"):
                e = d / f"eval_finetuning_{s}"
                e.mkdir(parents=True)
                (e / "config_used.json").write_text("{}")
                for sc in et.SCREENS:
                    (e / f"{sc}_predictions.csv").write_text("x\n")
        (sia / "pca_joint" / "latest_combo_cv3.txt").write_text(
            "results/pca_joint/best_cat_ea_cv3\n")
        (sia / "pca_joint" / "var_pd_pooled__latest_combo_cv3.txt") \
            .write_text("results/var_pd_pooled__best_cat_ea_cv3\n")
        res = root / "sl_comparison/results"
        (res / "external_transfer").mkdir(parents=True)
        (res / "external_transfer" / "arm_none.csv").write_text("x")
        (res / "pca_joint" / "external_transfer").mkdir(parents=True)
        (res / "pca_joint" / "external_transfer" / "old.csv").write_text("x")

        def job(**env):
            r = subprocess.run(
                ["bash", "sl_comparison/slurm/run_external_transfer.sh"],
                cwd=root, env=_minimal_env(
                    b, PYTHON_PATH=str(b / "py"), PYLOG=str(root / "pylog"),
                    SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled", **env),
                capture_output=True, text=True, timeout=120)
            return r.returncode, r.stdout + r.stderr

        rc, out = job(PCA_ARM="joint")
        assert rc == 0, out
        last = (root / "pylog").read_text().splitlines()[-1]
        assert last.startswith("sl_comparison/external_transfer.py --siamese "
                               "siamese_sl=siamese_sl/results/pca_joint/"
                               "best_cat_ea_cv3 --no_baselines") \
            and last.endswith("--out_dir sl_comparison/results/pca_joint/"
                              "external_transfer"), last
        # every output names its arm (label only, 2026-10-21): the arm and
        # its label (one argument; the log joins words) before --out_dir
        assert " --pca_arm joint --pca_label PCA on the concatenation of all " \
            "modalities (variance 0.8) --out_dir " in last, last
        assert "siamese_sl_pd_pooled" not in last, last
        assert "WARNING: siamese_sl_pd_pooled (pd:pooled) ABSENT" in out \
            and "names siamese_sl/results/var_pd_pooled__best_cat_ea_cv3, " \
            "which is not a var_pd_pooled__best_cat_*_cv3 combo" in out \
            and "directly in PCA arm joint's siamese_sl/results/pca_joint/" \
            in out, out
        assert "External-screen transfer tables: PCA arm joint (PCA on the " \
            "concatenation of all modalities (variance 0.8))" in out, out
        assert (res / "external_transfer" / "arm_none.csv").exists()
        assert not (res / "pca_joint" / "external_transfer").exists()
        # arm none: its own markers (none here), never arm joint's combos
        (root / "pylog").unlink()
        rc, out = job()
        assert rc == 1 and "no SLeuth variant of PCA arm none" in out, out
        assert not (root / "pylog").exists() \
            or "--siamese" not in (root / "pylog").read_text()
        assert not (res / "external_transfer").exists()
        # a refused setting: nothing is removed
        (res / "pca_joint" / "external_transfer").mkdir()
        rc, out = job(PCA_ARM="joint", POST_PCA_DIM="64")
        assert rc == 1 and "refused this job's settings" in out, out
        assert (res / "pca_joint" / "external_transfer").exists()


def test_paper_tables_job_passes_the_arms_directories():
    # run_paper_tables.sh: an arm's tables from its own comparison root
    # (--pca_arm / --pca_arms / --out_dir / --comparison_csv / --external_dir
    # all the arm's), --pca_arms_table (first argument only) the cross-arm
    # table into sl_comparison/results/pca_arms/; the verdict line from that
    # mode's README and paper_tables.py's exit status. What would let a
    # caller's flag move the arm's directories is refused before anything is
    # removed.
    assert _result_literals("run_paper_tables.sh") == [
        "OUT=sl_comparison/results/pca_arms"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = _sh_tree(root, "sl_comparison/slurm/run_paper_tables.sh",
                     "siamese_sl/slurm/run_best_per_category.conf")
        res = root / "sl_comparison/results"
        for d in ("paper_tables", "pca_joint/paper_tables", "pca_arms"):
            (res / d).mkdir(parents=True)
            (res / d / "keep").write_text("x")

        def job(*args, **env):
            if (root / "pylog").exists():
                (root / "pylog").unlink()
            r = subprocess.run(
                ["bash", "sl_comparison/slurm/run_paper_tables.sh", *args],
                cwd=root, env=_minimal_env(
                    b, PYTHON_PATH=str(b / "py"), PYLOG=str(root / "pylog"),
                    SIAMESE_VARIANTS_LIST="pd:per_head", **env),
                capture_output=True, text=True, timeout=120)
            calls = ((root / "pylog").read_text().splitlines()
                     if (root / "pylog").exists() else [])
            return r.returncode, r.stdout + r.stderr, calls

        rc, out, calls = job("--ablation_run_token", "T1", PCA_ARM="joint")
        assert rc == 0 and calls == [
            "sl_comparison/paper_tables.py --pca_arm joint --pca_arms none "
            "joint --out_dir sl_comparison/results/"
            "pca_joint/paper_tables --comparison_csv sl_comparison/results/"
            "pca_joint/comparison.csv --external_dir sl_comparison/results/"
            "pca_joint/external_transfer --variants pd:per_head "
            "--selection_variant pd:pooled --pca_variance 0.8 "
            "--bestcat_extra_variants 1 --ablation_run_token T1"], calls
        assert "verdict: PAPER COMPLETE" in out \
            and "Paper tables: PCA arm joint (PCA on the concatenation" in out
        assert not (res / "pca_joint/paper_tables/keep").exists()
        assert (res / "paper_tables/keep").exists() \
            and (res / "pca_arms/keep").exists()
        rc, out, calls = job("--pca_arms_table", PCA_ARMS="both none",
                             PT_RC="3", PT_VERDICT="PAPER INCOMPLETE: 1 "
                                                   "item(s) (x)")
        assert rc == 3 and calls == [
            "sl_comparison/paper_tables.py --pca_arms_table --pca_arms none "
            "both --out_dir sl_comparison/results/pca_arms --variants "
            "pd:per_head --selection_variant pd:pooled --pca_variance 0.8 "
            "--bestcat_extra_variants 1"], (out, calls)
        assert "verdict: PAPER INCOMPLETE: 1 item(s) (x)" in out, out
        assert not (res / "pca_arms/keep").exists()
        rc, out, calls = job("--regrade", "--pca_variance", "0.9",
                             "--code_manifest=")                # arm none
        assert rc == 0 and calls[0].startswith(
            "sl_comparison/paper_tables.py --pca_arm none --pca_arms none "
            "joint --out_dir sl_comparison/results/"
            "paper_tables --comparison_csv sl_comparison/results/"
            "comparison.csv --external_dir sl_comparison/results/"
            "external_transfer ") and calls[0].endswith(
            " --regrade --pca_variance 0.9 --code_manifest="), calls
        # refused, nothing removed and paper_tables.py never run
        (res / "pca_arms").mkdir(exist_ok=True)
        (res / "pca_arms/keep").write_text("x")
        for args, env, why in (
                (("--pca_arm", "both"), {"PCA_ARM": "joint"},
                 "--pca_arm: the PCA arm and its directories are this job's"),
                (("--out", "x"), {}, "--out: the PCA arm"),
                (("--e", "x"), {}, "--e: the PCA arm"),
                (("--pca_arms=none",), {}, "--pca_arms=none: the PCA arm"),
                (("--root=/elsewhere",), {}, "--root=/elsewhere: the PCA arm"),
                (("--regrade", "--pca_arms_table"), {},
                 "--pca_arms_table must be the FIRST argument"),
                (("--pca_arms_table",), {"PCA_ARMS": "joint"},
                 "compares at least two PCA arms; PCA_ARMS\n       resolves "
                 "to 'joint'"),
                (("--pca_arms_table",), {"POST_PCA_VARIANCE": "0.8"},
                 "refused this job's settings")):
            rc, out, calls = job(*args, **env)
            assert rc in (1, 2) and why in out and calls == [], \
                (args, rc, out, calls)
            assert (res / "pca_arms/keep").exists(), args


def test_paper_tables_job_runs_the_bootstrap_between_two_passes():
    # Since 2026-11-13 an arm's run_paper_tables.sh job builds the tables,
    # then bootstrap_ci.py (it reads their tab_tiers.csv) into the arm's
    # bootstrap/ (the previous one removed first), then the tables again
    # (their CI columns); the cross-arm job and PT_BOOTSTRAP=0 do not, and
    # without a tab_tiers.csv there is nothing to bootstrap. A failed
    # bootstrap turns a complete verdict's 0 into 3.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = _sh_tree(root, "sl_comparison/slurm/run_paper_tables.sh",
                     "siamese_sl/slurm/run_best_per_category.conf")
        res = root / "sl_comparison/results"
        boot = res / "pca_joint/bootstrap"

        def job(*args, **env):
            (root / "pylog").unlink(missing_ok=True)
            boot.mkdir(parents=True, exist_ok=True)
            (boot / "old").write_text("x")
            r = subprocess.run(
                ["bash", "sl_comparison/slurm/run_paper_tables.sh", *args],
                cwd=root, env=_minimal_env(
                    b, PYTHON_PATH=str(b / "py"), PYLOG=str(root / "pylog"),
                    SIAMESE_VARIANTS_LIST="pd:per_head", PCA_ARM="joint",
                    **env),
                capture_output=True, text=True, timeout=120)
            calls = [c.split()[0] for c in
                     (root / "pylog").read_text().splitlines()]
            return r.returncode, r.stdout + r.stderr, calls

        PT, BS = "sl_comparison/paper_tables.py", "sl_comparison/bootstrap_ci.py"
        rc, out, calls = job(PT_TIERS="1")
        assert rc == 0 and calls == [PT, BS, PT], (calls, out)
        assert not (boot / "old").exists()
        assert (root / "pylog").read_text().splitlines()[1] == (
            f"{BS} --pca_arm joint --out_dir "
            "sl_comparison/results/pca_joint/bootstrap --skip_missing")
        rc, out, calls = job(PT_TIERS="1", BOOT_RC="1")
        assert rc == 3 and calls == [PT, BS, PT] \
            and "bootstrap CIs FAILED" in out, out
        rc, out, calls = job(PT_TIERS="1", BOOT_RC="1", PT_RC="1")
        assert rc == 1, out
        rc, out, calls = job()                         # no tab_tiers.csv
        assert rc == 0 and calls == [PT] and "no bootstrap CIs" in out
        rc, out, calls = job(PT_TIERS="1", PT_BOOTSTRAP="0")
        assert rc == 0 and calls == [PT] and (boot / "old").exists()
        rc, out, calls = job("--pca_arms_table", PT_TIERS="1",
                             PCA_ARMS="none joint")
        assert rc == 0 and calls == [PT] and (boot / "old").exists()


def test_preflight_does_not_pass_without_cuda():
    # preflight.sh exists to exercise CUDA device placement; the three
    # trainers fall back to the CPU without it and exit 0, and the job then
    # printed "PREFLIGHT PASSED - all three models ran + scored on GPU". Runs
    # the script's own probe and verdict.
    text = (HERE / "slurm" / "preflight.sh").read_text()
    a = text.index("FAIL=0\n")
    b = text.index('echo "--- building tiny fold set')
    assert text.count("FAIL=0") == 1, "nothing may reset the verdict"
    assert text.index(". sl_comparison/slurm/python_path.sh") < a < b
    probe = text[a:b]
    assert "torch.cuda.is_available()" in probe and "exit" not in \
        probe.replace("sys.exit(", ""), probe   # the models still run after it
    verdict = text[text.rindex("\nif [ $FAIL -eq 0 ]; then"):]
    with tempfile.TemporaryDirectory() as tmp:
        py = Path(tmp) / "py"
        py.write_text('#!/bin/bash\nexit "$CUDA_RC"\n')
        py.chmod(0o755)
        for rc, want in (("0", "PREFLIGHT PASSED"), ("1", "PREFLIGHT FAILED")):
            r = subprocess.run(
                ["bash", "-u", "-c", probe + verdict], cwd=tmp,
                env=dict(os.environ, PYTHON_PATH=str(py), CUDA_RC=rc),
                capture_output=True, text=True, timeout=60)
            assert r.returncode == int(rc) and want in r.stdout, \
                (rc, r.stdout, r.stderr)
            assert ("FAIL: CUDA is not available" in r.stdout) == (rc == "1")


def test_prepare_folds_clear_stale_outputs():
    """Only this script's own layout goes: cv*/fold_<k> dirs holding
    train_pos.npy, conflicts.tsv and meta.json."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for cv in ("cv1", "cv2", "cv3"):
            for k in range(5):
                d = out / cv / f"fold_{k}"
                d.mkdir(parents=True)
                np.save(d / "train_pos.npy", np.zeros((1, 2), dtype=np.int32))
        (out / "cv1" / "fold_x").mkdir()  # no train_pos.npy: not ours
        for name in ("conflicts.tsv", "meta.json", "genes.txt", "keep.txt"):
            (out / name).write_text("x")
        pf.clear_stale_outputs(out)
        left = sorted(str(p.relative_to(out)) for p in out.rglob("*"))
        assert left == ["cv1", "cv1/fold_x", "cv2", "cv3", "genes.txt",
                        "keep.txt"], left


def test_p14_empty_results_clear_csv_and_exit_1():
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results"
        res.mkdir()
        csv_p = res / "comparison.csv"
        notes = res / "comparison_notes.txt"
        csv_p.write_text("cv,model,metric,mean,std\ncv3,SLGNN,auprg,0.9,0.01\n")
        notes.write_text("OLD NOTES\n")
        (res / "stale_shared").mkdir()
        (res / "stale_shared" / "SLGNN_cv3.json").write_text("{}")
        rc = cmp.main(["--results_dir", str(res), "--csv", str(csv_p)])
        assert rc == 1
        assert not csv_p.exists() and not notes.exists()


def test_p14_seed_row_written():
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results"
        res.mkdir()
        s = {"model": "SLGNN", "cv_type": "cv3", "auprg_mean": .5,
             "auprg_std": .1, "seed": 42, "trainable_params": 10}
        (res / "SLGNN_cv3.json").write_text(json.dumps({"summary": s}))
        csv_p = res / "comparison.csv"
        assert cmp.main(["--results_dir", str(res), "--csv", str(csv_p)]) == 0
        rows = pd.read_csv(csv_p)
        assert int(rows[rows["metric"] == "seed"]["mean"].iloc[0]) == 42
        assert "TRAINING SEED" in (res / "comparison_notes.txt").read_text()


def _report(fn, results):
    """Run one compare.py report function on a synthetic {cv: {model: summary}}
    and return its lines, with stdout swallowed."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return "\n".join(fn(results))


def test_upstream_report_flags_both_slmgae_reductions():
    """--allow_static_lp_mapping is a REDUCED SLMGAE exactly like a ZERO-FILLED
    view; only the latter used to be named."""
    views = {"GO BP": "data/Human_GOsim.txt", "PPI": "data/ppi.txt"}
    static = {"id_space": "symbol->entrez (static gene_id_mapping.tsv "
                          "fallback)", "rows": 6375, "mapped": 6156,
              "in_universe": 4470}
    hgnc = {"id_space": "symbol->entrez", "rows": 6375, "mapped": 6375,
            "in_universe": 4587}

    def rep(summary):
        return _report(cmp.upstream_input_report, {"cv3": {"SLMGAE": summary}})

    out = rep({"support_views": views, "lp_mapping": static})
    assert "REDUCED" in out and "--allow_static_lp_mapping" in out, out
    zf = dict(views, **{"PPI": "ZERO-FILLED (missing)"})
    out = rep({"support_views": zf, "lp_mapping": hgnc})
    assert "REDUCED" in out and "--allow_missing_views" in out, out
    # A row reduced through lp_mapping alone (no support_views key at all).
    assert "REDUCED" in rep({"lp_mapping": static})
    assert "REDUCED" not in rep({"support_views": views, "lp_mapping": hgnc})


def test_upstream_report_does_not_attach_kg2id_to_an_unrecorded_slgnn_row():
    """A row with no `arch` predates the SynLethKG path, so the kg2id caveat
    describes an input it never had."""

    def rep(summary):
        return _report(cmp.upstream_input_report, {"cv3": {"SLGNN": summary}})

    out = rep({})
    assert "no `arch` record" in out and "SLGNN/cv3" in out, out
    assert "kg2id.txt" not in out, out
    # A row that DID record the authors' architecture still gets the caveat.
    assert "kg2id.txt" in rep({"arch": "original"})
    # Mixed: the caveat still prints (it describes the original row).
    mixed = _report(cmp.upstream_input_report,
                    {"cv2": {"SLGNN": {"arch": "original"}},
                     "cv3": {"SLGNN": {}}})
    assert "kg2id.txt" in mixed and "no `arch` record" in mixed, mixed


def test_reduction_markers_match_the_slmgae_trainer():
    """compare.py copies these two strings out of train_slmgae_shared.py (it
    stays import-light, and SLMGAE-in-pytorch is not importable by name). If
    the trainer rewords either, the REDUCED warning silently stops firing, so
    pin them by text."""
    src = (HERE.parent / "SLMGAE-in-pytorch" /
           "train_slmgae_shared.py").read_text()
    assert cmp.LP_STATIC_FALLBACK_MARK in src, cmp.LP_STATIC_FALLBACK_MARK
    assert f'f"{cmp.ZERO_FILLED_VIEW_MARK} (missing ' in src, \
        cmp.ZERO_FILLED_VIEW_MARK


def test_model_config_report_flags_nsf4sl_deviations():
    """--features embedding / --legacy_loop are not the authors' NSF4SL."""
    out = _report(cmp.model_config_report,
                  {"cv3": {"NSF4SL": {"features": "embedding",
                                      "feature_file": "kg_complex.pt",
                                      "legacy_loop": True}}})
    assert "WARNING" in out and "NSF4SL/cv3" in out, out
    assert "--features embedding" in out and "--legacy_loop" in out, out
    ok = _report(cmp.model_config_report,
                 {"cv3": {"NSF4SL": {"features": "transe",
                                     "legacy_loop": False}}})
    assert ok == "", ok
    old = _report(cmp.model_config_report, {"cv3": {"NSF4SL": {}}})
    assert "no `features` record" in old, old


def test_elisl_is_a_peer_between_nsf4sl_and_the_probe():
    """ELISL (ELISL/slurm/run_elisl.sh): a shared row, ordered with the
    published peers, and named by model_config_report when it ran on other
    than its two context-free sources."""
    assert not cmp.is_siamese_row("ELISL")
    assert cmp.MODEL_ORDER.index("NSF4SL") < cmp.MODEL_ORDER.index("ELISL") \
        < cmp.MODEL_ORDER.index("BilinearProbe")
    ok = _report(cmp.model_config_report,
                 {"cv3": {"ELISL": {"elisl_sources": "Sequence_1024|PPI_ec"}}})
    assert ok == "", ok
    bad = _report(cmp.model_config_report,
                  {"cv3": {"ELISL": {"elisl_sources": "Sequence_1024"}}})
    assert "WARNING" in bad and "ELISL/cv3 (Sequence_1024)" in bad, bad


def test_model_configuration_rows_reach_the_csv():
    """arch / features / legacy_loop are what tells an author's model from a
    substitute, so the archival CSV must carry them, not only the notes."""
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results"
        res.mkdir()
        (res / "NSF4SL_cv3.json").write_text(json.dumps({"summary": {
            "model": "NSF4SL", "cv_type": "cv3", "auprg_mean": .5,
            "features": "embedding", "legacy_loop": True}}))
        (res / "SLGNN_cv3.json").write_text(json.dumps({"summary": {
            "model": "SLGNN", "cv_type": "cv3", "auprg_mean": .4,
            "arch": "port"}}))
        csv_p = res / "comparison.csv"
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            assert cmp.main(["--results_dir", str(res),
                             "--csv", str(csv_p)]) == 0
        rows = pd.read_csv(csv_p)

        def got(model, metric):
            m = rows[(rows["model"] == model) & (rows["metric"] == metric)]
            return str(m["mean"].iloc[0]) if len(m) else None

        assert got("NSF4SL", "features") == "embedding"
        assert got("NSF4SL", "legacy_loop") == "True"
        assert got("SLGNN", "arch") == "port"
        notes = (res / "comparison_notes.txt").read_text()
        assert "do not publish them as NSF4SL" in notes
        assert "--arch port" in notes


def test_partial_row_is_marked_everywhere():
    """A partial mean must not read like a complete one in the table, the CSV
    or the notes."""
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results"
        res.mkdir()
        base = {"cv_type": "cv3", "auprg_mean": .5, "auprg_std": .1}
        (res / "SLGNN_cv3.json").write_text(json.dumps({"summary": dict(
            base, model="SLGNN", n_folds_scored=5, n_folds_expected=5,
            partial=False)}))
        (res / "DegreePrior_cv3.json").write_text(json.dumps({"summary": dict(
            base, model="DegreePrior", trainable_params=0, n_folds_scored=4,
            n_folds_expected=5, partial=True)}))
        csv_p = res / "comparison.csv"
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cmp.main(["--results_dir", str(res),
                             "--csv", str(csv_p)]) == 0
        assert "DegreePrior(0)[PARTIAL 4/5]" in buf.getvalue()
        assert "SLGNN[PARTIAL" not in buf.getvalue()
        rows = pd.read_csv(csv_p)
        got = rows[(rows["model"] == "DegreePrior")
                   & (rows["metric"] == "partial")]["mean"].iloc[0]
        assert str(got) == "True", got
        notes = (res / "comparison_notes.txt").read_text()
        assert "PARTIAL ROWS" in notes and "DegreePrior/cv3: 4 of 5" in notes


def test_render_markdown_names_cv2_as_split_labels_does():
    """summary.md's prose must use the >=1-unseen label, not 'one-unseen':
    the cv2 screen split contains the both-unseen pairs."""
    md = et.render_markdown(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                            [("A", Path("a"))], [], ["Toy"],
                            list(et.SPLITS), {})
    assert "one-unseen" not in md, md
    for s in et.SPLITS:
        assert et.SPLIT_LABELS[s] in md, s


def test_shared_task_stopped_by_its_checks_leaves_no_previous_matrices():
    """run_shared_models.sh moves the task's previous row aside before its
    checks; the previous score matrices must go at the same point. The
    trainers purge them only after loading folds and inputs, so a task
    stopped earlier left the previous run's complete matrices (fold stamp
    still matching an unchanged fold set) for evaluate_model.py to grade."""
    import shutil
    dirs = {"SLGNN": "SLGNN/results/slgnn_{cv}",
            "NSF4SL": "NSF4SL/results/nsf4sl_{cv}",
            "SLMGAE": "SLMGAE-in-pytorch/results/slmgae_shared_{cv}"}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sl_comparison" / "slurm").mkdir(parents=True)
        (root / "bin").mkdir()
        for f in ("run_shared_models.sh", "python_path.sh"):
            shutil.copy(HERE / "slurm" / f, root / "sl_comparison" / "slurm" / f)
        for name in ("module", "py"):       # py: every check and trainer fails
            (root / "bin" / name).write_text(
                "#!/bin/bash\nexit %d\n" % (name == "py"))
            (root / "bin" / name).chmod(0o755)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SLURM_") and k not in ("CVS", "SEED")}
        env.update(PATH=str(root / "bin") + os.pathsep + env["PATH"],
                   PYTHON_PATH=str(root / "bin" / "py"))
        res = root / "sl_comparison" / "results"
        for task, (name, tmpl) in enumerate(dirs.items()):      # cv1 tasks
            d1, d2 = root / tmpl.format(cv="cv1"), root / tmpl.format(cv="cv2")
            for d, cv in ((d1, "cv1"), (d2, "cv2")):
                (d / "checkpoints").mkdir(parents=True)
                for k in (0, 1):
                    (d / f"fold_{k}_{cv}_predictions.npy").write_text("x")
                (d / "model_params.json").write_text("{}")
            res.mkdir(parents=True, exist_ok=True)
            (res / f"{name}_cv1.json").write_text("{}")
            r = subprocess.run(
                ["bash", "sl_comparison/slurm/run_shared_models.sh"], cwd=root,
                env=dict(env, SLURM_ARRAY_TASK_ID=str(task)),
                capture_output=True, text=True, timeout=120)
            out = r.stdout + r.stderr
            assert r.returncode == 1 and f"FATAL: {name} inputs" in out, out
            assert not (res / f"{name}_cv1.json").exists()
            assert len(list((res / "stale_shared").glob(f"{name}_cv1.*.json"))) == 1
            assert sorted(p.name for p in d1.iterdir()) == \
                ["checkpoints", "model_params.json"], (name, list(d1.iterdir()))
            # another CV's directory is another task's
            assert len(list(d2.glob("fold_*_cv2_predictions.npy"))) == 2


def test_a_failed_per_fold_param_rewrite_keeps_the_regraded_row():
    # collect_and_regrade_siamese.sh adds trainable_params_fold_* to the row
    # evaluate_model.py has just written. It is an optional annotation
    # (`|| echo WARNING`): a write that fails part-way (a full quota) must
    # cost the counts, never the re-graded row, which compare.py would then
    # skip without a word while run_compare.sh counts the file as present.
    # Runs the script's own heredoc; a kernel file-size limit fails the write.
    import resource
    text = (HERE / "slurm" / "collect_and_regrade_siamese.sh").read_text()
    a = text.index("<<'PYEOF'")
    a = text.index("\n", text.index("|| echo", a)) + 1
    code = text[a:text.index("\nPYEOF\n", a) + 1]
    row0 = json.dumps({"summary": {"model": "siamese_sl", "cv_type": "cv1",
                                   "auprg_mean": 0.9,
                                   "folds": [{"auprg": 0.9}] * 5}}, indent=2)
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "results.json"
        row = Path(tmp) / "siamese_sl_cv1.json"
        res.write_text(json.dumps(
            {"fold_metrics": [{"total_params": 100 + k} for k in range(5)]}))

        def run(limit=None):
            row.write_text(row0)

            def cap():
                hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
                resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
            return subprocess.run(
                [sys.executable, "-", str(res), str(row)], input=code,
                capture_output=True, text=True, timeout=120,
                preexec_fn=None if limit is None else cap)

        r = run()
        s = json.loads(row.read_text())["summary"]
        assert r.returncode == 0 and s["auprg_mean"] == 0.9 \
            and s["trainable_params_fold_mean"] == 102.0 \
            and s["trainable_params_fold_n"] == 5, (r.stdout, r.stderr)
        for limit in (0, 40):       # no byte / only the first 40 reach the disk
            r = run(limit)
            assert r.returncode != 0 and "OSError" in r.stderr, \
                (limit, r.stdout, r.stderr)
            assert row.read_text() == row0, (limit, row.read_text()[:60])
            assert sorted(p.name for p in Path(tmp).iterdir()) == \
                ["results.json", "siamese_sl_cv1.json"], limit


def test_model_params_failed_rewrite_keeps_previous_copy():
    # write_model_params is never fatal and every fold start rewrites the
    # file: a rewrite that fails part-way (a full quota) must leave the
    # previous fold's copy, whose fold stamp is the same, not a cut file that
    # evaluate_model.py and the combo's Step 2 refuse as "no fold_stamp". A
    # run that never manages to write must still end without a stamp, also in
    # a directory that holds a previous run's copy (the peers keep theirs).
    mp = _load("model_params")
    real_json = mp.json

    class Model:
        def parameters(self):
            return []

    class CutJson:      # some bytes reach the file, then the write is refused
        @staticmethod
        def dump(payload, f, **kw):
            f.write('{"trainable_params"')
            raise OSError(122, "Disk quota exceeded")

    def write(out, folds, fail=False, **extra):
        mp.json = CutJson if fail else real_json
        try:
            return mp.write_model_params(out, Model(), folds_dir=folds,
                                         cv_type="cv1", **extra)
        finally:
            mp.json = real_json

    with tempfile.TemporaryDirectory() as tmp:
        folds, out = Path(tmp) / "folds", Path(tmp) / "out"
        folds.mkdir()
        for n in mp.STAMP_FILES:
            (folds / n).write_text(n)
        assert write(out, folds, seed=1) == (0, 0)
        good = (out / mp.PARAMS_FILENAME).read_bytes()
        assert write(out, folds, fail=True, seed=1) == (None, None)
        assert (out / mp.PARAMS_FILENAME).read_bytes() == good
        assert [q.name for q in out.iterdir()] == [mp.PARAMS_FILENAME]
        mp.verify_fold_stamp(out, folds, "cv1")          # still gradeable

        # a new run starts without the previous run's copy ...
        (out / "fold_0_cv1_predictions.npy").write_text("x")
        assert mp.purge_stale_predictions(out, "cv1") == 1
        assert list(out.iterdir()) == []
        assert write(out, folds, fail=True, seed=2) == (None, None)
        assert list(out.iterdir()) == []                 # ... so: no stamp
        _raises(lambda: mp.verify_fold_stamp(out, folds, "cv1"), SystemExit,
                "no fold_stamp")
        # ... while a deliberate resume (keep) keeps the stamp of its folds
        assert write(out, folds, seed=2) == (0, 0)
        (out / "fold_0_cv1_predictions.npy").write_text("x")
        assert mp.purge_stale_predictions(out, "cv1", keep=(0,)) == 0
        assert mp.read_model_params(out)["seed"] == 2


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
