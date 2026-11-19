#!/usr/bin/env python3
"""The PCA arms in the siamese stage jobs (no data files, no GPU, no training).

Since 2026-10-20 the whole pipeline runs once per PCA arm (none, joint,
per_modality, both: slurm/config.conf "PCA arms"), and each job finds its arm
in PCA_ARM. Covered here:

  config.conf    the arm settings and helpers: every arm's results roots and
                 train.py flags, the owner of a pair's single-embedding runs
                 (the first arm of none / joint, or of per_modality / both,
                 that PCA_ARMS lists), the refused values (an unknown or
                 repeated arm, POST_PCA_*, a bad fraction or count), the
                 canonical order of PCA_ARMS, what is and is not exported.
  stage scripts  no literal results/ path left in their code: every path goes
                 through the arm's root (RESULTS_ROOT / STEP1_ROOT / CMP_ROOT).
  the jobs       run in a temp tree whose $PYTHON_PATH sends `train.py
                 --print_arch` to the real train.py and a training call to a
                 fake trainer (train.py's own parser, the real
                 fit_multimodal_transform on tiny random embedding files, no
                 model): the single-embedding array of each owner arm, the
                 combo job of each arm (Step 2 ranks the owner's runs, Step 3
                 trains with the arm's PCA, everything in the arm's root; a
                 rerun sets the previous combos aside in the arm's own root),
                 a combo refusing single-embedding runs of another PCA, the
                 resubmission each Step-2 stop prints, followed as printed
                 (arm none untouched; under a cv2 SELECTION_CV too, which
                 every arm keeps) and the one a failed combo's closing
                 WARNING prints, a joint count at or above arm both's
                 width failing its combos, the leave-one-category-out worker
                 and its summary in arm joint (read against the arm's own
                 combo), the embedding benchmark worker and summary in arm
                 both and owner none (the default PCA_ARMS too), the eval
                 tasks and the pair export of every arm (stand-ins for
                 eval_finetuning.py / export_pair_predictions.py), the
                 cell-line ablation job and grade.sh in arms joint and none
                 (stand-ins for zero_cell_emb.py and evaluate_model.py). No
                 job of one arm writes into another arm's root.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_pca_arms.py -q
    $PYTHON_PATH siamese_sl/tests/test_pca_arms.py
"""
import concurrent.futures
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SIAMESE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

from arch_signature import PCA_ARMS, pca_arm, pca_settings  # noqa: E402

_SLURM = os.path.join(_SIAMESE, "slurm")

# Settings the confs read from the environment: one still exported by the
# calling shell (an earlier experiment's PCA_ARM or POST_PCA_VARIANCE,
# SLURM_JOB_ID, which makes config.conf load modules) must not decide these
# tests. A test that means one passes it explicitly.
_ENV_PREFIXES = ("SLURM_", "SIAMESE_", "PCA_", "POST_PCA_", "BESTCAT_",
                 "BEST_CAT_", "ABLATION_", "FAKE_")
_ENV_NAMES = ("RUN_TOKEN", "ARCH_ARGS", "USE_CELL_LINES", "CELL_LINE_DIM",
              "SHARED_FOLDS", "SEED", "PCA_METHOD", "PREPROCESSING_FIT_SCOPE",
              "SELECTION_VARIANT", "SELECTION_CV", "SELECTION_FRAC",
              "SELECTION_SEED", "SELECTION_MODE", "PYTHON_PATH",
              "MAX_CONCURRENT", "EVAL_CONCURRENT", "SHARED_CONCURRENT",
              "GPU_NODES", "GPU_EXCLUDE")


# The pipeline settings these tests were written under, pinned: on
# 2026-11-13 config.conf / run_best_per_category.conf took the paper's
# protocol as their defaults (PCA_ARMS "none joint", the variants
# "pd:pooled pd:per_head" with pd:pooled selecting, BESTCAT_EXTRA_VARIANTS=1,
# SELECTION_FRAC=0.1 with per-fold selection, ALLMOD_COMBO=1). What the arms
# tests check (each arm's roots and flags, the owners of the single-embedding
# runs, the jobs' namespaces) does not depend on those defaults, so they keep
# running the whole 2x2 of arms and of variants, selected globally on
# pd:per_head without a selection split, as before; test_config_conf_defaults
# checks the defaults themselves. A test passes its own value of any of them
# to override it, None to leave it unset (config.conf's default).
_PINNED = {"PCA_ARMS": " ".join(PCA_ARMS),
           "SIAMESE_VARIANTS_LIST": "pd:per_head pd:pooled bilinear:per_head "
                                    "bilinear:pooled",
           "SELECTION_VARIANT": "pd:per_head", "BESTCAT_EXTRA_VARIANTS": "0",
           "SELECTION_FRAC": "", "ALLMOD_COMBO": "0"}


def _env(_defaults=False, **extra):
    """This process's environment without any pipeline setting, then the
    _PINNED settings (none with _defaults=True: config.conf's own
    defaults), then `extra` (a value of None unsets that name)."""
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(_ENV_PREFIXES) and k not in _ENV_NAMES
         and k != "ALLMOD_COMBO"}
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    if not _defaults:
        e.update(_PINNED)
    e.update(extra)
    return {k: v for k, v in e.items() if v is not None}


def _conf(script, _defaults=False, **env):
    """(returncode, stdout lines, stderr) of `script` after config.conf."""
    r = subprocess.run(
        ["bash", "-c", 'source slurm/config.conf >/dev/null || exit 9; '
         + script], cwd=_SIAMESE, env=_env(_defaults, **env),
        capture_output=True,
        text=True, timeout=120)
    return r.returncode, r.stdout.splitlines(), r.stderr


# --- config.conf ---------------------------------------------------------------
def test_config_conf_arm_roots_and_flags():
    show = ('echo "$PCA_ARM|$RESULTS_ROOT|$STEP1_ROOT|$CMP_ROOT|'
            '$PCA_MULTI_ARGS|$PCA_SINGLE_ARGS"')
    want = {
        "none": "none|results|results|sl_comparison/results|--no_post_pca|"
                "--no_post_pca",
        "joint": "joint|results/pca_joint|results|"
                 "sl_comparison/results/pca_joint|--post_pca_variance 0.8|"
                 "--no_post_pca",
        "per_modality": "per_modality|results/pca_per_modality|"
                        "results/pca_per_modality|"
                        "sl_comparison/results/pca_per_modality|"
                        "--pca_variance 0.8 --no_post_pca|"
                        "--pca_variance 0.8 --no_post_pca",
        "both": "both|results/pca_both|results/pca_per_modality|"
                "sl_comparison/results/pca_both|"
                "--pca_variance 0.8 --post_pca_variance 0.8|"
                "--pca_variance 0.8 --no_post_pca"}
    for arm, line in want.items():
        rc, out, err = _conf(show, PCA_ARM=arm)
        assert rc == 0 and out == [line], (arm, rc, out, err)
    # an unset PCA_ARM is none; the targets and an exact count move the flags
    rc, out, _ = _conf(show)
    assert out == [want["none"]], out
    rc, out, _ = _conf(show, PCA_ARM="both", PCA_MODALITY_VARIANCE="0.9",
                       PCA_JOINT_DIM="24")
    assert out == ["both|results/pca_both|results/pca_per_modality|"
                   "sl_comparison/results/pca_both|"
                   "--pca_variance 0.9 --post_pca_dim 24|"
                   "--pca_variance 0.9 --no_post_pca"], out
    rc, out, _ = _conf('pca_arm_label joint', PCA_JOINT_VARIANCE="0.7")
    assert out == ["PCA on the concatenation of all modalities "
                   "(variance 0.7)"], out


def test_config_conf_single_embedding_owner_follows_pca_arms():
    # The first arm of a pair that the run includes keeps the pair's
    # single-embedding runs; a run without none never rewrites results/.
    for arms, arm, step1 in (
            ("none joint per_modality both", "joint", "results"),
            ("joint both", "joint", "results/pca_joint"),
            ("joint both", "both", "results/pca_both"),
            ("none both", "both", "results/pca_both"),
            ("per_modality", "both", "results/pca_per_modality"),
            ("joint", "none", "results/pca_joint")):
        rc, out, err = _conf('echo "$STEP1_ROOT"', PCA_ARMS=arms,
                             PCA_ARM=arm)
        assert rc == 0 and out == [step1], (arms, arm, out, err)
    # and its label names the arms of this run that read them
    for arms, arm, who in (("none joint per_modality both", "joint",
                            "none joint"),
                           ("joint both", "joint", "joint"),
                           ("joint both", "none", "none joint"),
                           ("none both", "both", "both"),
                           ("per_modality", "both", "per_modality both")):
        rc, out, err = _conf('pca_step1_arms "$PCA_ARM"; '
                             'pca_step1_label "$PCA_ARM"', PCA_ARMS=arms,
                             PCA_ARM=arm)
        what = ("at native dimension (no PCA)" if arm in ("none", "joint")
                else "with PCA within the modality (variance 0.8)")
        w = who.split()
        tail = (f"shared by arms {w[0]} and {w[1]}" if len(w) == 2
                else f"used by arm {w[0]} only")
        assert rc == 0 and out == [
            who, f"single-embedding runs {what}, {tail}"], (arms, arm, out,
                                                            err)


def test_config_conf_refuses_bad_arm_settings():
    for env, msg in (({"PCA_ARM": "half"}, "PCA_ARM='half'"),
                     ({"PCA_ARMS": "none sideways"}, "'sideways' is not an arm"),
                     ({"PCA_ARMS": "joint none joint"}, "lists joint twice"),
                     ({"PCA_ARMS": " "}, "PCA_ARMS is empty"),
                     # set but empty is a mistake, not all four arms
                     ({"PCA_ARMS": ""}, "PCA_ARMS is set but empty"),
                     ({"PCA_ARMS": "*"}, "'*' is not an arm"),
                     ({"POST_PCA_VARIANCE": "0.8"},
                      "POST_PCA_VARIANCE / POST_PCA_DIM are replaced"),
                     ({"POST_PCA_DIM": "64"},
                      "POST_PCA_VARIANCE / POST_PCA_DIM are replaced"),
                     ({"PCA_MODALITY_VARIANCE": "1"}, "PCA_MODALITY_VARIANCE='1'"),
                     ({"PCA_MODALITY_VARIANCE": "0.0"},
                      "PCA_MODALITY_VARIANCE='0.0'"),
                     ({"PCA_JOINT_VARIANCE": "80"}, "PCA_JOINT_VARIANCE='80'"),
                     ({"PCA_JOINT_DIM": "0"}, "PCA_JOINT_DIM='0'"),
                     ({"PCA_JOINT_DIM": "08"}, "PCA_JOINT_DIM='08'"),
                     ({"PCA_JOINT_DIM": "1.5"}, "PCA_JOINT_DIM='1.5'"),
                     ({"ARCH_ARGS": "--pca_variance 0.8"},
                      "ARCH_ARGS must not set a PCA")):
        rc, out, err = _conf("echo reached", **env)
        assert rc == 9 and "reached" not in out and msg in err, (env, rc, err)


def test_config_conf_defaults_are_the_papers_protocol():
    # 2026-11-13: a plain submission reproduces the paper. With no pipeline
    # setting in the environment config.conf (+ run_best_per_category.conf)
    # resolve to: PCA arms none and joint (none owning both arms' Step-1
    # runs), the variants pd:pooled and pd:per_head, pd:pooled selecting,
    # the extra Step-1 block of the other variant, a 10% selection split
    # (seed 42, every training's train.py flags) with per-fold selection,
    # and the all-modalities combo; each exported, so every job sees them.
    show = ('source slurm/run_best_per_category.conf >/dev/null || exit 8; '
            'bash -c \'echo "$PCA_ARMS|$SIAMESE_VARIANTS_LIST|$SELECTION_FRAC|'
            '$SELECTION_SEED|$SELECTION_MODE|$ALLMOD_COMBO"\'; '
            'echo "${PCA_ARM_LIST[*]}|${SIAMESE_VARIANTS[*]}|'
            '$SELECTION_VARIANT|$BESTCAT_EXTRA_VARIANTS|$SELECTION_ARGS|'
            '$(pca_step1_arm joint)"')
    rc, out, err = _conf(show, _defaults=True)
    assert rc == 0 and out == [
        "none joint|pd:pooled pd:per_head|0.1|42|per_fold|1",
        "none joint|pd:pooled pd:per_head|pd:pooled|1|"
        "--selection_frac 0.1 --selection_seed 42|none"], (out, err)
    # SELECTION_FRAC set but empty (or 0) turns the split off, and the mode
    # falls back to global; per_fold without a split is refused
    for frac in ("", "0"):
        rc, out, err = _conf('echo "[$SELECTION_ARGS]|$SELECTION_MODE"',
                             _defaults=True, SELECTION_FRAC=frac)
        assert rc == 0 and out == ["[]|global"], (frac, out, err)
    rc, out, err = _conf("echo reached", _defaults=True, SELECTION_FRAC="",
                         SELECTION_MODE="per_fold")
    assert rc == 9 and "per_fold needs a selection split" in err, (rc, err)
    # ... and the pinned settings the other tests of this file run under
    # are honoured over those defaults
    rc, out, err = _conf(show)
    assert rc == 0 and out == [
        "none joint per_modality both|pd:per_head pd:pooled bilinear:per_head"
        " bilinear:pooled||42|global|0",
        "none joint per_modality both|pd:per_head pd:pooled bilinear:per_head"
        " bilinear:pooled|pd:per_head|0||none"], (out, err)


def test_config_conf_exports_the_arms_not_the_roots():
    # PCA_ARMS reaches every job (normalised to the canonical order); the
    # derived roots never do: an sl_comparison job, which does not source
    # config.conf, must derive its own from PCA_ARM, never inherit the
    # orchestrator's arm-none roots.
    rc, out, err = _conf(
        "bash -c 'echo \"[$PCA_ARMS][$PCA_ARM][${RESULTS_ROOT-unset}]"
        "[${CMP_ROOT-unset}][${PCA_MULTI_ARGS-unset}]\"'",
        PCA_ARMS="both none", PCA_ARM="joint")
    assert rc == 0 and out == ["[none both][joint][unset][unset][unset]"], (
        out, err)
    rc, out, _ = _conf('echo "${PCA_ARM_LIST[*]}"')
    assert out == [" ".join(PCA_ARMS)], out


# --- no literal results/ path in the stage scripts -----------------------------
_STAGE_SCRIPTS = (
    "slurm/run_best_per_category.sh", "slurm/run_best_per_category_combo.sh",
    "slurm/run_embedding_benchmark.sh",
    "slurm/run_embedding_benchmark_summary.sh", "slurm/run_ablation.sh",
    "slurm/run_ablation_summary.sh", "slurm/eval_finetuning.conf",
    "slurm/eval_finetuning.sh", "slurm/export_pair_predictions.sh",
    "cell_line_ablation/run_cell_line_ablation.sh",
    "cell_line_ablation/grade.sh")


def test_stage_scripts_reach_results_only_through_the_arm_roots():
    # A literal results/ (or sl_comparison/results) in a stage job's code is
    # arm none's directory whatever PCA_ARM says: another arm's job would read
    # or overwrite arm none's files. Flagged: results/..., Path("results"),
    # and a bare root as a value (results, sl_comparison/results or
    # ../sl_comparison/results as an argument or an assignment; in a message
    # it is a word). Comments, and the defaults of a quoted heredoc's
    # os.environ reads, are fine.
    lit = re.compile(r"(?<![A-Za-z_$])results/|Path\([\"']results[\"']\)")
    bare = re.compile(r"(?<![\w$.{-])(?:\.\./)?(?:sl_comparison/|siamese_sl/)?"
                      r"results(?=[\"'\s\\;)]|$)")
    message = re.compile(r"^(echo|printf|print\(|f?[\"'])")
    allowed = (
        'os.environ.get("RESULTS_ROOT", "results")',
        'os.environ.get("STEP1_ROOT", "results")',
        'GRADE_OUT_DIR="${GRADE_OUT_DIR:-sl_comparison/results}"',
        # the benchmark summary's JSON key
        'summary["results"]',
        # the cell-line rows' hint names the arm's directory and the shared one
        'echo "  sl_comparison/results."',
        # train.py --print_arch's required --output_dir, never created
        "--output_dir results/_sigcheck",
        # the cross-arm table compares every arm: one directory for all
        "sl_comparison/results/pca_arms",
    )
    bad = []
    for rel in _STAGE_SCRIPTS:
        with open(os.path.join(_SIAMESE, rel)) as f:
            for n, ln in enumerate(f, 1):
                # the line's code, without a trailing comment
                code = re.sub(r"\s#\s.*$", "", ln).strip()
                if not code or code.startswith("#"):
                    continue
                if any(a in code for a in allowed):
                    continue
                if lit.search(code) or (bare.search(code)
                                        and not message.search(code)):
                    bad.append(f"{rel}:{n}: {code}")
    assert not bad, "\n".join(bad)


# --- the jobs, with a fake trainer ---------------------------------------------
_PY_STUB = """#!/bin/bash
# $PYTHON_PATH of the job tests: `train.py ... --print_arch` is the real
# train.py, a training call the fake trainer, anything else the interpreter.
if [ "$1" = "train.py" ]; then
    shift
    case " $* " in
        *" --print_arch "*) exec @PY@ @TRAIN@ "$@" ;;
    esac
    exec @PY@ @FAKE@ "$@"
fi
exec @PY@ "$@"
"""

_FAKE_TRAIN = '''"""train.py for the job tests: train.py's own parser and argument
resolution (train.main), so config.json is what train.py would save, and the
real fit_multimodal_transform on the run's tiny embedding files; no model.
model_params.json carries the real fold_stamp of --folds_dir."""
import importlib.util
import json
import os
import sys

sys.path.insert(0, @SIAMESE@)
import torch  # noqa: E402

import train  # noqa: E402
from data_loader import (apply_multimodal_transform,  # noqa: E402
                         fit_multimodal_transform)

_spec = importlib.util.spec_from_file_location("_mp", @MODEL_PARAMS@)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


class FakeTrainer:
    def __init__(self, args):
        self.args = args
        if not os.path.isdir(os.path.join(args.output_dir, "checkpoints")):
            raise FileNotFoundError(args.output_dir)  # train.py never mkdirs
        args.input_dim = None
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        self.sha = [train.file_sha256(p) for p in args.embeddings_paths]

    def train(self):
        a, out = self.args, self.args.output_dir
        raw_all = [torch.load(p)["embeddings"] for p in a.embeddings_paths]
        gen = torch.Generator().manual_seed(a.seed)
        fmods = train._parse_fold_modalities(a.fold_modalities, a.num_folds)
        fm = []
        for k in range(a.num_folds):
            fit = torch.rand(raw_all[0].shape[0], generator=gen) < 0.7
            idx = (list(range(len(raw_all))) if fmods is None
                   else list(fmods[k]))
            raw = [raw_all[i] for i in idx]
            t = fit_multimodal_transform(
                raw, fit, pca_dims=(None if a.pca_dims is None else
                                    [a.pca_dims[i] for i in idx]),
                post_pca=a.post_pca, pca_method=a.pca_method)
            if fmods is not None:
                t["modality_idx"] = idx
            a.input_dim = int(apply_multimodal_transform(raw_all, t).shape[1])
            torch.save({"input_dim": a.input_dim,
                        "preprocessing_transform": t, "config": vars(a),
                        "embeddings_paths": list(a.embeddings_paths),
                        "embeddings_sha256": list(self.sha)},
                       os.path.join(out, "checkpoints", f"fold_{k}_best.pt"))
            with open(os.path.join(
                    out, f"fold_{k}_{a.cv_type}_predictions.npy"), "wb"):
                pass
            # a score that depends on the files, so each category has a winner
            v = round(0.3 + 0.01 * k + a.input_dim / 1000
                      + 0.001 * sum(len(p) for p in a.embeddings_paths), 6)
            fm.append({"fold": k, "input_dim": a.input_dim,
                       "total_params": 33 * a.input_dim,
                       "nonzero_params": 30 * a.input_dim,
                       "auprg_any": v, "aupr_any": v / 2,
                       "auroc_any": 0.5 + v / 2, "val_auprg_any": v,
                       "modalities": [os.path.basename(a.embeddings_paths[i])
                                      for i in idx]})
            if a.selection_frac:
                # --selection_frac: a score on the fold's selection split
                # whose order moves with the fold (so per-fold picks differ),
                # and the split's digest (the same in every run of a seed)
                h = sum(ord(ch) for p in a.embeddings_paths
                        for ch in os.path.basename(p))
                fm[-1].update({
                    "sel_auprg_any": round(0.2 + ((h * (k + 3)) % 17) / 100,
                                           6),
                    "sel_digest": f"{a.selection_seed}-{a.selection_frac}-"
                                  f"{a.cv_type}-{k}",
                    "sel_n_pairs": 10})
        with open(os.path.join(out, "model_params.json"), "w") as f:
            json.dump({"fold_stamp": mp.fold_stamp(a.folds_dir, a.cv_type)}, f)
        n = len(fm)
        mean = lambda k: sum(m[k] for m in fm) / n  # noqa: E731
        res = {"cv_type": a.cv_type, "config": vars(a),
               "embeddings_sha256": list(self.sha), "fold_metrics": fm,
               "summary": {"auprg_any_mean": mean("auprg_any"),
                           "auprg_any_std": 0.01,
                           "val_auprg_any_mean": mean("val_auprg_any"),
                           "auroc_mean": mean("auroc_any"), "auroc_std": 0.0,
                           "aupr_mean": mean("aupr_any"), "aupr_std": 0.0,
                           "total_params": mean("total_params"),
                           "nonzero_params": mean("nonzero_params")}}
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(res, f, indent=2)


train.Trainer = FakeTrainer
sys.argv = ["train.py"] + sys.argv[1:]
train.main()
'''

# The tiny catalog: category -> embedding types (native width); every type
# gets a file. The ablation categories are these three.
_CATALOG = (("geneformer", "expression", 16), ("scgpt", "expression", 12),
            ("go2vec", "go", 10), ("onto2vec", "go", 9),
            ("kg_complex", "kg", 6))
_ALL_EMB = (("go", 8),)        # the embedding benchmark's catalog


def _tree(tmp):
    """<tmp>/siamese_sl: copies of the stage jobs and confs (their 10 s
    disk-settle sleep removed; a tiny catalog and ablation list), the real
    config.conf behind the stub interpreter, arch_signature.py; <tmp>/data
    with the catalog's files (60 genes); <tmp>/sl_comparison/model_params.py
    (the combo's fold-stamp check)."""
    root = Path(tmp)
    sia = root / "siamese_sl"
    (sia / "slurm").mkdir(parents=True)
    for f in ("run_best_per_category.sh", "run_best_per_category_combo.sh",
              "run_embedding_benchmark.sh",
              "run_embedding_benchmark_summary.sh", "run_ablation.sh",
              "run_ablation_summary.sh", "run_best_per_category.conf",
              "run_ablation.conf", "run_embedding_benchmark.conf",
              "run_generate_embeddings.conf"):
        text = Path(_SLURM, f).read_text()
        if f.endswith(".sh") and "sleep 10" in text:
            text, n = re.subn(r"(?m)^sleep 10\b.*$", "true", text)
            assert n == 1, (f, n)
        (sia / "slurm" / f).write_text(text)
    conf = sia / "slurm" / "run_best_per_category.conf"
    text, n = re.subn(r"(?ms)^EMB_CATALOG=\(\n.*?^\)\n", "EMB_CATALOG=(\n" + "".join(
        f'    "{t}:all_genes_{t}.pt:{c}"\n' for t, c, _ in _CATALOG) + ")\n",
        conf.read_text())
    assert n == 1
    conf.write_text(text)
    abl = sia / "slurm" / "run_ablation.conf"
    text, n = re.subn(r"(?m)^ABLATE_CATEGORIES=\(.*\)$",
                      "ABLATE_CATEGORIES=(expression go kg)", abl.read_text())
    assert n == 1
    abl.write_text(text)
    shutil.copy(os.path.join(_SIAMESE, "arch_signature.py"), sia)
    shutil.copy(os.path.join(_SIAMESE, "fold_selection.py"), sia)
    (root / "sl_comparison").mkdir()
    shutil.copy(os.path.join(_REPO, "sl_comparison", "model_params.py"),
                root / "sl_comparison")
    b = root / "bin"
    b.mkdir()
    (b / "fake_train.py").write_text(
        _FAKE_TRAIN.replace("@SIAMESE@", repr(_SIAMESE)).replace(
            "@MODEL_PARAMS@",
            repr(os.path.join(_REPO, "sl_comparison", "model_params.py"))))
    (b / "python").write_text(
        _PY_STUB.replace("@PY@", shlex.quote(sys.executable))
        .replace("@TRAIN@", shlex.quote(os.path.join(_SIAMESE, "train.py")))
        .replace("@FAKE@", shlex.quote(str(b / "fake_train.py"))))
    (b / "python").chmod(0o755)
    # the real config.conf, then the stub interpreter and a one-entry
    # ALL_EMBEDDINGS (after config.conf's own checks ran)
    (sia / "slurm" / "config.conf").write_text(
        f'source "{os.path.join(_SLURM, "config.conf")}" || return 1\n'
        f'PYTHON_PATH="{b / "python"}"\n'
        "ALL_EMBEDDINGS=(" + " ".join(
            f'"{t}:all_genes_{t}.pt:{t}"' for t, _ in _ALL_EMB) + ")\n")
    (root / "data").mkdir()
    gen = torch.Generator().manual_seed(0)
    genes = [str(1000 + i) for i in range(60)]
    for t, dim in [(t, d) for t, _, d in _CATALOG] + list(_ALL_EMB):
        torch.save({"embeddings": torch.randn(len(genes), dim, generator=gen),
                    "gene_order": genes}, root / "data" / f"all_genes_{t}.pt")
    return sia


def _job(sia, script, task=None, **env):
    # (env may pass another RUN_TOKEN / variant list: a later submission's)
    e = _env(**{"SLURM_SUBMIT_DIR": str(sia), "RUN_TOKEN": "tok-1",
                "SIAMESE_VARIANTS_LIST": "pd:per_head", **env})
    if task is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(task)
    # cwd: SLURM starts a job in its submit dir (config.conf cds there too,
    # inside a job)
    return subprocess.run(["bash", str(sia / "slurm" / script)], env=e,
                          cwd=str(sia), capture_output=True, text=True,
                          timeout=900)


def _array(sia, script, n, **env):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_job, sia, script, t, **env) for t in range(n)]
        runs = [f.result() for f in futs]
    for t, r in enumerate(runs):
        assert r.returncode == 0, (script, t, r.stdout[-3000:] + r.stderr[-3000:])
    return runs


def _snapshot(d):
    """{relative path: mtime_ns} of every file under d (missing: {})."""
    d = Path(d)
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): p.stat().st_mtime_ns
            for p in d.rglob("*") if p.is_file()}


def _cfg(d):
    return json.loads((Path(d) / "config.json").read_text())


def test_each_arm_trains_and_selects_in_its_own_root():
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        res = sia / "results"
        n1 = len(_CATALOG) * 3
        # the two owners' single-embedding arrays
        _array(sia, "run_best_per_category.sh", n1, PCA_ARM="none")
        _array(sia, "run_best_per_category.sh", n1, PCA_ARM="per_modality")
        for t, _, _ in _CATALOG:
            for cv in ("cv1", "cv2", "cv3"):
                assert pca_arm(_cfg(res / f"pcavar0.8_{t}_{cv}")) == "none"
                pm = res / "pca_per_modality" / f"pcavar0.8_{t}_{cv}"
                assert pca_arm(_cfg(pm)) == "per_modality", pm
                assert (pm / "run_token.txt").read_text().strip() == "tok-1"
        step1_none = _snapshot(res)
        step1_none = {k: v for k, v in step1_none.items()
                      if not k.startswith("pca_")}
        step1_pm = _snapshot(res / "pca_per_modality")
        # every arm's combo job; none and joint share results/'s runs,
        # per_modality and both share results/pca_per_modality/'s
        for arm in PCA_ARMS:
            r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM=arm)
            assert r.returncode == 0, (arm, r.stdout[-4000:] + r.stderr[-3000:])
            assert f"PCA arm: {arm} (" in r.stdout, r.stdout[:2000]
        roots = {a: res / (f"pca_{a}" if a != "none" else "") for a in PCA_ARMS}
        roots["none"] = res
        sels = {}
        for arm, root in roots.items():
            sel = json.loads((root / "best_per_category.json").read_text())
            meta = sel["_selection_meta"]
            want_step1 = ("results" if arm in ("none", "joint")
                          else "results/pca_per_modality")
            assert (meta["pca_arm"], meta["step1_root"]) == (arm, want_step1), (
                arm, meta)
            assert meta["pca_args"] == {
                "none": "--no_post_pca", "joint": "--post_pca_variance 0.8",
                "per_modality": "--pca_variance 0.8 --no_post_pca",
                "both": "--pca_variance 0.8 --post_pca_variance 0.8"}[arm]
            sels[arm] = {c: w["type"] for c, w in sel["cv1"].items()}
            for cv in ("cv1", "cv2", "cv3"):
                text = (root / f"latest_combo_{cv}.txt").read_text().strip()
                d = sia / text
                # the marker names the combo inside the arm's own root, in the
                # spelling run_ablation.sh compares it with
                assert d.parent == root and text == (
                    f"{want_step1 if arm in ('none', 'per_modality') else 'results/pca_' + arm}"
                    f"/{d.name}"), (arm, text)
                assert pca_arm(_cfg(d)) == arm, (arm, cv)
                s = pca_settings(_cfg(d))
                assert (s["modality"] is not None) == (
                    arm in ("per_modality", "both")), (arm, s)
                assert (s["joint"] is not None) == (arm in ("joint", "both"))
        # a pair of arms ranks one set of runs, hence one selection
        assert sels["none"] == sels["joint"], sels
        assert sels["per_modality"] == sels["both"], sels
        # no combo job rewrote a single-embedding run, and arm none's root
        # holds no other arm's selection, combo or marker
        assert {k: v for k, v in _snapshot(res).items()
                if k in step1_none} == step1_none
        assert {k: v for k, v in _snapshot(res / "pca_per_modality").items()
                if k in step1_pm} == step1_pm
        top = {p.name for p in res.iterdir()}
        assert not any(n.startswith("pca_") and n not in (
            "pca_joint", "pca_per_modality", "pca_both") for n in top), top
        for arm in ("joint", "both"):
            assert not list(roots[arm].glob("pcavar*")), arm
        # A rerun, as every repeat of the pipeline is (both orchestrators
        # leave the previous combos to the combo job): the two arms that do
        # not own their single-embedding runs (STEP1_ROOT != RESULTS_ROOT)
        # move their previous combos, with whatever eval_finetuning_*/ and
        # pair_predictions/ they hold, into THEIR OWN <arm root>/
        # _superseded_combos/job_<id>/, never into the owner's root, and
        # delete them there once Step 2 has succeeded (2026-10-21).

        def outside_arm_roots(root):
            return {k: v for k, v in _snapshot(root).items()
                    if not k.startswith("pca_")}

        for arm, owner in (("joint", "none"), ("both", "per_modality")):
            before = outside_arm_roots(roots[owner])
            r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM=arm)
            assert r.returncode == 0, (arm, r.stdout[-4000:] + r.stderr[-3000:])
            moved = [ln for ln in r.stdout.splitlines()
                     if ln.startswith("Moved ")]
            assert len(moved) == 1 and moved[0].startswith(
                f"Moved 3 previous combo dir(s) to results/pca_{arm}/"
                "_superseded_combos/job_"), (arm, moved)
            assert outside_arm_roots(roots[owner]) == before, arm
        assert not list(res.rglob("*superseded*")), \
            list(res.rglob("*superseded*"))
        # A failed Step 2 (another submission's token: every candidate is
        # foreign) keeps them in the arm, and the owner's next combo job
        # leaves them alone.
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="joint",
                 RUN_TOKEN="tok-2")
        assert r.returncode == 1, r.stdout[-3000:]
        kept = sorted(res.rglob("superseded__*"))
        assert len(kept) == 3 and all(
            p.parent.parent == roots["joint"] / "_superseded_combos"
            and pca_arm(_cfg(p)) == "joint" for p in kept), kept
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="none")
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-3000:]
        assert sorted(res.rglob("superseded__*")) == kept


def test_a_combo_refuses_single_embedding_runs_of_another_pca():
    # per_modality's combo finds native runs in its owner's root (copied there
    # by hand): its Step-2 signature (built with the arm's single-embedding
    # PCA) refuses every candidate before anything trains.
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        res = sia / "results"
        _array(sia, "run_best_per_category.sh", len(_CATALOG) * 3,
               PCA_ARM="none")
        (res / "pca_per_modality").mkdir()
        for d in res.glob("pcavar*"):
            shutil.copytree(d, res / "pca_per_modality" / d.name)
        r = _job(sia, "run_best_per_category_combo.sh",
                 PCA_ARM="per_modality")
        assert r.returncode == 1, r.stdout[-3000:]
        assert "trained with other settings than this run" in r.stdout \
            and "pca_variance" in r.stdout, r.stdout[-3000:]
        assert not (res / "pca_per_modality" / "best_per_category.json").exists()
        assert not list((res / "pca_per_modality").glob("best_cat_*"))


# A command a Step-2 stop prints to paste (from siamese_sl/): the run's token,
# the arm, the run's PCA_ARMS, for a combo the run's SELECTION_CV (2026-10-21:
# in no signature, so a combo resubmitted without it selected on cv1 and
# exited 0), then sbatch, an --array for a Step-1 rerun.
_HINT_CMD = re.compile(r"RUN_TOKEN=(\S+) PCA_ARM=(\S+) PCA_ARMS='([^']*)' "
                       r"(?:SELECTION_CV=(\S+) )?"
                       # the selection split settings, always carried
                       # since 2026-11-13 (an omitted SELECTION_FRAC is
                       # no longer "off")
                       r"(?:SELECTION_FRAC=\S* SELECTION_SEED=\S+ "
                       r"SELECTION_MODE=\S+ )?"
                       r"sbatch --export=ALL (?:--array=(\S+) )?"
                       r"slurm/(run_best_per_category(?:_combo)?\.sh)")


def _hints(out):
    """[(token, arm, arms, selection CV or None, array ids or None, script)]
    in the order printed."""
    return [m.groups() for m in _HINT_CMD.finditer(out)]


def _words(cmd):
    """A printed command taken literally, as a shell would: the NAME=VALUE
    words before sbatch ({name: value}) and its --array ids (or None)."""
    w = shlex.split(cmd)
    i = w.index("sbatch")
    assert all("=" in x for x in w[:i]), cmd
    ids = [x.split("=", 1)[1] for x in w[i:] if x.startswith("--array=")]
    return dict(x.split("=", 1) for x in w[:i]), (ids[0] if ids else None)


def test_combo_step2_hints_name_the_arm_and_spare_arm_none():
    # Every Step-2 stop of a combo job prints the resubmission to paste. A
    # job submitted by hand is arm none's unless PCA_ARM is passed or
    # exported (each stop says so, pinned below), and it
    # removes that arm's selection and markers as it starts: a hint without
    # the arm, followed for a failed arm, wiped arm none (2026-10-21). Each
    # stop of a non-none arm is followed here exactly as printed (only the
    # command's NAME=VALUE words reach the job), from a shell exporting
    # nothing else (the test's single variant aside): the failed arm is
    # repaired and arm none is untouched throughout:
    #   --print_arch fails (a mistyped ARCH_ARGS): resubmit, same token;
    #   no candidate of this run's token: resubmit on the earlier one;
    #   candidates of other settings: rerun their tasks as the owner, or
    #     resubmit with the Step-1 run's settings;
    #   a SELECTION_CV task missing: rerun it as the owner (its exact id),
    #     then the combo of each arm that ranks its runs.
    # The last stop again under SELECTION_CV=cv2 (run_best_per_category.conf's
    # documented override, which every job of the run gets): the rerun is the
    # cv2 task's own id, and the combo commands carry the run's SELECTION_CV
    # (in no signature: without it the recovered arms selected on cv1 and
    # exited 0), so all four arms end selected on cv2 under one token.
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        res = sia / "results"
        n1 = len(_CATALOG) * 3
        cat = [t for t, _, _ in _CATALOG]
        arms4 = " ".join(PCA_ARMS)
        _array(sia, "run_best_per_category.sh", n1, PCA_ARM="none")
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="none")
        assert r.returncode == 0, r.stdout[-3000:]

        def arm_none():
            return {k: v for k, v in _snapshot(res).items()
                    if not k.startswith("pca_")}

        def selection(arm):
            root = res / ("" if arm == "none" else f"pca_{arm}")
            return json.loads((root / "best_per_category.json").read_text())[
                "_selection_meta"]

        def follow(cmd, arm=None):
            """Submit a printed combo command (<that arm> filled in): the
            arm's _selection_meta."""
            env, _ = _words(cmd if arm is None else
                            cmd.replace("<that arm>", arm))
            r = _job(sia, "run_best_per_category_combo.sh", **env)
            assert r.returncode == 0, (cmd, arm, r.stdout[-3000:])
            meta = selection(env["PCA_ARM"])
            assert (meta["pca_arm"], meta["run_token"]) == (
                env["PCA_ARM"], env["RUN_TOKEN"]), meta
            return meta

        def recover(out):
            """An incomplete-candidate stop followed as printed: the owner's
            Step-1 rerun, then the combo of each arm that ranks its runs;
            {reader: its _selection_meta}."""
            env, ids = _words(_HINT_CMD.search(out).group(0))
            for t in ids.split(","):
                x = _job(sia, "run_best_per_category.sh", int(t), **env)
                assert x.returncode == 0, x.stdout[-2000:]
            m = re.search(r"each arm that ranks them \(([^)]*)\): "
                          r"(RUN_TOKEN=.*? sbatch --export=ALL "
                          r"slurm/run_best_per_category_combo\.sh)", out)
            return {a: follow(m.group(2), a) for a in m.group(1).split()}

        def remark(out):
            """Each stop says what a job submitted by hand without PCA_ARM
            is: the arm its shell exports, else arm none's (2026-10-21:
            "arm none's", wrong in a shell an arm's recipe left PCA_ARM
            exported in)."""
            flat = " ".join(out.split())
            assert "submitted by hand is the arm its shell exports, else " \
                "arm none's" in flat and "is arm none's" not in flat, \
                out[-2000:]

        none_before = arm_none()
        assert any(k.startswith("best_cat_") for k in none_before)
        # --print_arch fails, arm joint: this job's command, with the arm
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="joint",
                 ARCH_ARGS="--layernrom")
        assert r.returncode == 1 and "could not compute this run's Step-1 " \
            "signature" in r.stdout, r.stdout[-3000:]
        remark(r.stdout)
        assert _hints(r.stdout) == [("tok-1", "joint", arms4, "cv1", None,
                                     "run_best_per_category_combo.sh")], \
            r.stdout[-2000:]
        follow(_HINT_CMD.search(r.stdout).group(0))   # ARCH_ARGS fixed
        assert arm_none() == none_before
        # arm joint under a later token whose arrays never ran: the
        # earlier array's token from the file the hint names
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="joint",
                 RUN_TOKEN="tok-2")
        assert r.returncode == 1 and "No PCA benchmark results" in r.stdout
        remark(r.stdout)
        m = re.search(r"\(cat (\S+)/pcavar0\.8_<type>_<cv>/run_token\.txt\)"
                      r", from siamese_sl/: (RUN_TOKEN=<its token> "
                      r"PCA_ARM=(\S+) PCA_ARMS='([^']*)' SELECTION_CV=(\S+) "
                      r"(?:SELECTION_FRAC=\S* SELECTION_SEED=\S+ "
                      r"SELECTION_MODE=\S+ )?"
                      r"sbatch --export=ALL "
                      r"slurm/run_best_per_category_combo\.sh)", r.stdout)
        assert m and m.group(1) == "results" and m.groups()[2:] == (
            "joint", arms4, "cv1"), r.stdout[-2000:]
        tok = (sia / m.group(1) / "pcavar0.8_go2vec_cv1" / "run_token.txt") \
            .read_text().strip()
        follow(m.group(2).replace("<its token>", tok))
        assert arm_none() == none_before
        # arm joint under other ARCH_ARGS than its Step-1 runs: the
        # owner's tasks of every stale candidate, or this job as the arm
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="joint",
                 ARCH_ARGS="--layernorm")
        assert r.returncode == 1 and "trained with other settings" in \
            r.stdout, r.stdout[-3000:]
        remark(r.stdout)
        # (the readers' line, PCA_ARM=<that arm>, is checked as text below)
        ids = ",".join(str(i * 3) for i in range(len(cat)))
        assert _hints(r.stdout) == [
            ("tok-1", "none", arms4, None, ids, "run_best_per_category.sh"),
            ("tok-1", "joint", arms4, "cv1", None,
             "run_best_per_category_combo.sh")], r.stdout[-2000:]
        assert "resubmit the combo of each arm that ranks them (none " \
            "joint): RUN_TOKEN=tok-1 PCA_ARM=<that arm> PCA_ARMS='" + arms4 \
            + "' SELECTION_CV=cv1 SELECTION_FRAC= SELECTION_SEED=42 " \
            "SELECTION_MODE=global sbatch --export=ALL " \
            "slurm/run_best_per_category_combo.sh" in r.stdout, \
            r.stdout[-2000:]
        assert arm_none() == none_before
        # arm both: owner per_modality's array lost its go2vec/cv1 task
        miss = cat.index("go2vec") * 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            runs = list(ex.map(
                lambda t: _job(sia, "run_best_per_category.sh", t,
                               PCA_ARM="per_modality"),
                [t for t in range(n1) if t != miss]))
        assert all(x.returncode == 0 for x in runs)
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="both")
        assert r.returncode == 1 and "incomplete candidate set" in r.stdout
        remark(r.stdout)
        assert _hints(r.stdout) == [
            ("tok-1", "per_modality", arms4, None, str(miss),
             "run_best_per_category.sh"),
            ("tok-1", "both", arms4, "cv1", None,
             "run_best_per_category_combo.sh")], r.stdout[-2000:]
        metas = recover(r.stdout)
        assert list(metas) == ["per_modality", "both"], r.stdout[-2000:]
        for meta in metas.values():
            assert meta["step1_root"] == "results/pca_per_modality", meta
            assert not meta["unranked_failed"], meta
        assert arm_none() == none_before
        # SELECTION_CV=cv2: arms none and joint select on cv2 as the run
        # queued them, then per_modality's array loses go2vec/cv2
        for a in ("none", "joint"):
            r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM=a,
                     SELECTION_CV="cv2")
            assert r.returncode == 0, (a, r.stdout[-3000:])
        none_before = arm_none()
        shutil.rmtree(res / "pca_per_modality" / "pcavar0.8_go2vec_cv2")
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="both",
                 SELECTION_CV="cv2")
        assert r.returncode == 1 and "incomplete candidate set" in r.stdout
        remark(r.stdout)
        assert _hints(r.stdout) == [
            ("tok-1", "per_modality", arms4, None,
             str(cat.index("go2vec") * 3 + 1), "run_best_per_category.sh"),
            ("tok-1", "both", arms4, "cv2", None,
             "run_best_per_category_combo.sh")], r.stdout[-2000:]
        assert list(recover(r.stdout)) == ["per_modality", "both"]
        for a in PCA_ARMS:
            meta = selection(a)
            assert (meta["selection_cv"], meta["run_token"]) == (
                "cv2", "tok-1") and not meta["unranked_failed"], (a, meta)
        assert arm_none() == none_before


def test_combo_step3_warning_names_the_run_s_selection_cv():
    # A failed combo next to trained ones: the closing WARNING says how to
    # resubmit the job, with the run's SELECTION_CV beside the token and the
    # arms (2026-10-21: in no signature, so a resubmission without it
    # selected on cv1 and nothing stopped it). The job's closing block, run
    # as it ends a job of arm both under SELECTION_CV=cv2.
    with open(os.path.join(_SLURM, "run_best_per_category_combo.sh")) as f:
        text = f.read()
    tail = text[text.index("# Exit non-zero only when NOTHING trained"):]
    pre = ("set -e\nSELECTION_VARIANT=pd:per_head\nSTEP3_SKIPPED=0\n"
           "STEP3_SELECTION_FAILED=0\nSTEP3_SUCCESSFUL=11\nSTEP3_FAILED=1\n"
           "STEP3_FAILED_RUNS=(pd:pooled/cv2)\n"
           "RESULTS_ROOT=results/pca_both\n"
           # config.conf's, which the job sourced (the split off here)
           "SELECTION_FRAC=\nSELECTION_SEED=42\nSELECTION_MODE=global\n")
    r = subprocess.run(["bash", "-c", pre + tail], capture_output=True,
                       text=True, timeout=60,
                       env=_env(RUN_TOKEN="tok-1", PCA_ARM="both",
                                PCA_ARMS=" ".join(PCA_ARMS),
                                SELECTION_CV="cv2"))
    flat = " ".join(r.stdout.split())
    assert r.returncode == 0 and "Resubmit it from siamese_sl/ with " \
        "RUN_TOKEN=tok-1 PCA_ARM=both PCA_ARMS=\"none joint per_modality " \
        "both\" SELECTION_CV=cv2 SELECTION_FRAC= SELECTION_SEED=42 " \
        "SELECTION_MODE=global (and the run's other PCA_* and ARCH_ARGS " \
        "settings) exported" in flat, r.stdout + r.stderr


def test_a_joint_count_at_or_above_the_width_fails_the_combo():
    # PCA_JOINT_DIM is ONE count for arms joint and both (config.conf), and
    # arm both's joint stage meets only its per-modality components. A count
    # at or above that width ran no joint PCA while every label said both
    # (2026-10-21): train.py now refuses it at the fold's fit
    # (data_loader.fit_multimodal_transform), so each both combo fails, no
    # marker is written and the job exits 1 (nothing trained). A count below
    # the width trains, as an exact count end to end.
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        both = sia / "results" / "pca_both"
        _array(sia, "run_best_per_category.sh", len(_CATALOG) * 3,
               PCA_ARM="per_modality")
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="both",
                 PCA_JOINT_DIM="1000")
        assert r.returncode == 1 and "Step 3 (multi-modal): 0 ok, 0 skip, " \
            "3 fail" in r.stdout and "no combo trained successfully" in \
            r.stdout, r.stdout[-3000:]
        assert r.stderr.count("ValueError: post-concat PCA to 1000 "
                              "components") == 3 and "PCA_JOINT_DIM" in \
            r.stderr, r.stderr[-3000:]
        assert not list(both.glob("*latest_combo_*")), list(both.iterdir())
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="both",
                 PCA_JOINT_DIM="5")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        for cv in ("cv1", "cv2", "cv3"):
            d = sia / (both / f"latest_combo_{cv}.txt").read_text().strip()
            s = pca_settings(_cfg(d))
            assert s["joint"] == ("count", 5) and s["modality"] is not None, s
            fm = json.loads((d / "results.json").read_text())["fold_metrics"]
            assert {m["input_dim"] for m in fm} == {5}, fm


def test_per_fold_selection_end_to_end():
    """SELECTION_MODE=per_fold (2026-11-12): every training of the run gets
    the selection split (SELECTION_FRAC); Step 2 picks per (CV, fold) the
    argmax of fold_metrics[k].sel_auprg_any; each CV's combo is one run over
    the union of its folds' winners with --fold_modalities (each fold's
    checkpoint transform records its own subset); each ablation drops the
    category's pick from every fold; the summary reads them against their
    combo; candidates whose selection splits differ, or trained without the
    split, are refused."""
    sys.path.insert(0, _SIAMESE)
    import fold_selection as fs
    sel_env = {"SELECTION_FRAC": "0.1", "SELECTION_MODE": "per_fold"}
    cvs, cats = ("cv1", "cv2", "cv3"), sorted({c for _, c, _ in _CATALOG})
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        res = sia / "results"
        _array(sia, "run_best_per_category.sh", len(_CATALOG) * 3,
               PCA_ARM="none", **sel_env)
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="none",
                 **sel_env)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-3000:]
        sel = json.loads((res / "best_per_category.json").read_text())
        meta = sel["_selection_meta"]
        assert (meta["selection_mode"], meta["selection_metric"],
                meta["selection_frac"], meta["selection_seed"]) == (
                    "per_fold", "sel_auprg_any", 0.1, 42), meta
        mixed = False
        for cv in cvs:
            runs = {t: json.loads((res / f"pcavar0.8_{t}_{cv}" /
                                   "results.json").read_text())
                    for t, _, _ in _CATALOG}
            for t, run in runs.items():
                assert run["config"]["selection_frac"] == 0.1, t
            for cat in cats:
                info = sel[cv][cat]
                mixed |= len(info["types"]) > 1
                for k in range(5):
                    cands = [t for t, c, _ in _CATALOG if c == cat]
                    best = max(cands, key=lambda t: (
                        runs[t]["fold_metrics"][k]["sel_auprg_any"],
                        -cands.index(t)))
                    assert info["per_fold"][str(k)]["type"] == best, (
                        cv, cat, k, info)
            d = sia / (res / f"latest_combo_{cv}.txt").read_text().strip()
            spec = fs.combo_spec(sel, cv)
            assert d.name == f"best_cat_{'_'.join(spec['names'])}_{cv}", d
            cfg = _cfg(d)
            assert fs.run_matches_spec(cfg, spec) is None, (cv, cfg)
            assert cfg["selection_frac"] == 0.1
            for k in range(5):
                ck = torch.load(d / "checkpoints" / f"fold_{k}_best.pt",
                                weights_only=False)
                assert ck["preprocessing_transform"]["modality_idx"] == \
                    spec["fold_modalities"][k], (cv, k)
        # the fake scores make some category pick differently across folds,
        # so the union and the fold map are exercised
        assert mixed, sel
        assert "SELECTION_MODE=per_fold" in r.stdout
        # every ablation: each fold's own winners minus the category
        runs = _array(sia, "run_ablation.sh", 9, PCA_ARM="none", **sel_env)
        for cat in cats:
            for cv in cvs:
                d = res / f"ablate_no_{cat}_{cv}"
                assert fs.run_matches_spec(
                    _cfg(d), fs.combo_spec(sel, cv, drop=cat)) is None, d
                stamp = json.loads((d / "results.json").read_text())[
                    "ablation"]
                assert stamp["num_modalities"] == len(cats) - 1, stamp
        r = _job(sia, "run_ablation_summary.sh", PCA_ARM="none", **sel_env)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        with open(res / "ablation_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert sorted((x["cv"], x["dropped"]) for x in rows) == sorted(
            (cv, c) for cv in cvs for c in ["none"] + cats), rows
        assert not [x for x in rows if x["verdict"].startswith("MISMATCH")], \
            [x["verdict"] for x in rows]
        # an ablation trained without the split is not its combo's
        r = _job(sia, "run_ablation.sh", 0, PCA_ARM="none",
                 SELECTION_FRAC="0.1", SELECTION_SEED="7",
                 SELECTION_MODE="per_fold")
        assert r.returncode == 1 and "selection_seed" in r.stdout, \
            r.stdout[-3000:]
        # candidates whose selection splits differ for one (CV, fold)
        p = res / "pcavar0.8_go2vec_cv2" / "results.json"
        run = json.loads(p.read_text())
        run["fold_metrics"][3]["sel_digest"] = "another-split"
        p.write_text(json.dumps(run))
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="none",
                 **sel_env)
        assert r.returncode == 1 and "cv2 fold 3" in r.stdout and \
            "different selection splits" in r.stdout, r.stdout[-3000:]
        assert not (res / "best_per_category.json").exists()
        # Step-1 runs without the split: refused on their signature
        _array(sia, "run_best_per_category.sh", len(_CATALOG) * 3,
               PCA_ARM="none")
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="none",
                 **sel_env)
        assert r.returncode == 1 and "selection_frac" in r.stdout, \
            r.stdout[-3000:]
        # ...and the global mode on them is the selection it always was
        r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM="none")
        assert r.returncode == 0, r.stdout[-3000:]
        sel = json.loads((res / "best_per_category.json").read_text())
        assert sel["_selection_meta"]["selection_mode"] == "global"
        assert all(set(w) == {"type", "file", "aupr"}
                   for cv in cvs for w in sel[cv].values()), sel
        cfg = _cfg(sia / (res / "latest_combo_cv1.txt").read_text().strip())
        assert cfg["fold_modalities"] is None and cfg["selection_frac"] == 0


def test_ablation_worker_and_summary_stay_in_their_arm():
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        res = sia / "results"
        _array(sia, "run_best_per_category.sh", len(_CATALOG) * 3,
               PCA_ARM="none")
        for arm in ("none", "joint"):
            r = _job(sia, "run_best_per_category_combo.sh", PCA_ARM=arm)
            assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        before = _snapshot(res)
        before = {k: v for k, v in before.items() if not k.startswith("pca_")}
        # 3 categories x 3 CVs x 1 variant
        runs = _array(sia, "run_ablation.sh", 9, PCA_ARM="joint")
        assert all("PCA arm: joint (" in r.stdout for r in runs)
        joint = res / "pca_joint"
        for cat in ("expression", "go", "kg"):
            for cv in ("cv1", "cv2", "cv3"):
                d = joint / f"ablate_no_{cat}_{cv}"
                cfg = _cfg(d)
                assert pca_arm(cfg) == "joint" and cfg["post_pca"] == 0.8, d
                assert (d / "run_token.txt").read_text().strip() == "tok-1"
                assert not (res / f"ablate_no_{cat}_{cv}").exists()
        r = _job(sia, "run_ablation_summary.sh", PCA_ARM="joint")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        assert "PCA arm: joint (" in r.stdout and \
            "single-embedding runs in results/" in r.stdout, r.stdout[:1500]
        with open(joint / "ablation_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows and {x["pca_arm"] for x in rows} == {"joint"}, rows[:2]
        # the dropped category's solo score is read from the shared runs
        assert all(x["solo_auprg"] for x in rows if x["dropped"] != "none")
        # ...read against arm joint's OWN combo (its marker in
        # results/pca_joint/, not the single-embedding owner's in results/,
        # whose combo has the same name), so every ablation row is that combo
        # minus one category and carries a delta (2026-10-21)
        for x in rows:
            ref = (joint / f"latest_combo_{x['cv']}.txt").read_text().strip()
            assert x["reference"] == ref, (x["cv"], x["reference"], ref)
            assert not x["verdict"].startswith("MISMATCH"), x
            if x["dropped"] == "none":
                s = json.loads((sia / ref / "results.json").read_text())[
                    "summary"]
                assert abs(float(x["auprg_any"]) - s["auprg_any_mean"]) \
                    < 1e-9, (x, s)
        # every (cv, category) row is there: otherwise the checks above and
        # the solo check pass vacuously when the ablation rows go missing
        assert sorted((x["cv"], x["dropped"]) for x in rows) == sorted(
            (cv, c) for cv in ("cv1", "cv2", "cv3")
            for c in ("none", "expression", "go", "kg")), rows
        assert all(x["delta_auprg_any"] for x in rows if x["dropped"] != "none")
        assert not (res / "ablation_summary.csv").exists()
        # nothing of arm none's changed
        after = {k: v for k, v in _snapshot(res).items()
                 if not k.startswith("pca_")}
        assert after == before


def test_benchmark_worker_and_summary_in_arm_both():
    # the embedding benchmark is a single-embedding run too: arm both trains
    # it with the per-modality PCA, in its pair's root. Its summary names the
    # arms of the run that read those runs and what they train: the pair,
    # never every arm, also under the default four-arm run (PCA_ARMS unset),
    # whose other owner, arm none, is read by none and joint (2026-10-21).
    pm_label = ("single-embedding runs with PCA within the modality "
                "(variance 0.8), shared by arms per_modality and both")
    for arms in ("per_modality both", None):
        env = {} if arms is None else {"PCA_ARMS": arms}
        with tempfile.TemporaryDirectory() as tmp:
            sia = _tree(tmp)
            res = sia / "results"
            _array(sia, "run_embedding_benchmark.sh", 3, PCA_ARM="both",
                   **env)
            for cv in ("cv1", "cv2", "cv3"):
                d = res / "pca_per_modality" / f"go_{cv}"
                assert pca_arm(_cfg(d)) == "per_modality", d
            r = _job(sia, "run_embedding_benchmark_summary.sh",
                     PCA_ARM="both", **env)
            assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
            s = json.loads((res / "pca_per_modality" /
                            "embedding_benchmark_summary.json").read_text())
            assert s["pca_arms"] == "per_modality both" and \
                s["pca"] == pm_label and \
                s["pca_args"] == "--pca_variance 0.8 --no_post_pca", (arms, s)
            assert set(s["results"]["go"]) == {"cv1", "cv2", "cv3"}, s
            assert not (res / "embedding_benchmark_summary.json").exists()
            assert not list(res.glob("go_cv*"))
    # the default run's other owner
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        res = sia / "results"
        _array(sia, "run_embedding_benchmark.sh", 3, PCA_ARM="none")
        r = _job(sia, "run_embedding_benchmark_summary.sh", PCA_ARM="none")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        s = json.loads((res / "embedding_benchmark_summary.json").read_text())
        assert s["pca_arms"] == "none joint" and s["pca"] == (
            "single-embedding runs at native dimension (no PCA), shared by "
            "arms none and joint") and s["pca_args"] == "--no_post_pca", s
        assert set(s["results"]["go"]) == {"cv1", "cv2", "cv3"}, s
        assert not [p for p in res.iterdir() if p.name.startswith("pca_")]


# --- the eval tasks and the pair export ------------------------------------------
_EVAL_EXPORT_PY = r'''#!/bin/bash
# $PYTHON_PATH of the eval / pair-export job test: a -c snippet (the
# config.json reader) runs for real and the CUDA probe passes; a call of
# eval_finetuning.py or export_pair_predictions.py is logged (script,
# --model_dir, --cv_type) and leaves in its combo what the real one writes.
if [ "$1" = "-c" ]; then
    case "$2" in *cuda.is_available*) exit 0 ;; esac
    exec "$PY_REAL" "$@"
fi
d=""; cv=""; prev=""
for a in "$@"; do
    [ "$prev" = "--model_dir" ] && d="$a"
    [ "$prev" = "--cv_type" ] && cv="$a"
    prev="$a"
done
printf '%s\t%s\t%s\n' "$1" "$d" "$cv" >> "$PY_LOG"
case "$1" in
    eval_finetuning.py)
        mkdir -p "$d/eval_finetuning_$cv" &&
            echo '{}' > "$d/eval_finetuning_$cv/config_used.json" ;;
    export_pair_predictions.py)
        mkdir -p "$d/pair_predictions" &&
            echo '{}' > "$d/pair_predictions/manifest.json" ;;
    *) exit 9 ;;
esac
'''

# bash 3.2 (macOS) has no mapfile builtin, the compute nodes' bash has
_MAPFILE_SHIM = ('type mapfile >/dev/null 2>&1 || mapfile() { '
                 '[ "$1" = "-t" ] && shift; local __l; eval "$1=()"; '
                 'while IFS= read -r __l; do eval "$1+=(\\"\\$__l\\")"; done; }; ')


def test_eval_and_pair_export_take_their_arms_own_combos():
    # Each arm's chained eval tasks and its pair export take the combos ITS
    # markers name (RESULTS_ROOT), never those of the arm that owns its
    # single-embedding runs (STEP1_ROOT: arm none's root for joint,
    # per_modality's for both), and write inside them only. Both arms of a
    # pair select the same winners, hence the same combo names: every root
    # holds same-named combos and markers of every variant here, so a lookup
    # in another root would find a combo and exit 0 (2026-10-21). Under the
    # default PCA_ARMS: with "joint both" or one arm STEP1_ROOT is the arm's
    # own root and the two cannot be told apart. A marker missing from the
    # arm's root stops its task / leaves that combo out, whatever the other
    # roots hold.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sia = root / "siamese_sl"
        shutil.copytree(_SLURM, sia / "slurm",
                        ignore=shutil.ignore_patterns("logs"))
        b = root / "bin"
        b.mkdir()
        (b / "python").write_text(_EVAL_EXPORT_PY)
        (b / "python").chmod(0o755)
        with open(sia / "slurm" / "config.conf", "a") as f:
            f.write(f'\nPYTHON_PATH="{b / "python"}"\n')
        (root / "data").mkdir()
        (root / "data" / "all_genes_a.pt").write_text("")
        rc, prefixes, err = _conf('for v in "${SIAMESE_VARIANTS[@]}"; do '
                                  'echo "[$(siamese_variant_prefix "$v")]"; '
                                  'done')
        prefixes = [p[1:-1] for p in prefixes]
        assert rc == 0 and len(prefixes) == 4 and prefixes[0] == "", (
            prefixes, err)
        rc, cvs, err = _conf('source slurm/eval_finetuning.conf >/dev/null; '
                             'printf "%s\\n" "${EVAL_CV_TYPES[@]}"')
        assert rc == 0 and len(cvs) >= 2, (cvs, err)
        roots = {a: "results" + (f"/pca_{a}" if a != "none" else "")
                 for a in PCA_ARMS}
        for arm, res in roots.items():
            for p in prefixes:
                for cv in ("cv1", "cv3"):
                    combo = f"{res}/{p}best_cat_a_{cv}"
                    (sia / combo).mkdir(parents=True)
                    (sia / combo / "config.json").write_text(json.dumps(
                        {"embeddings_paths": ["/elsewhere/all_genes_a.pt"]}))
                    (sia / combo / "results.json").write_text("{}")
                    (sia / res / f"{p}latest_combo_{cv}.txt").write_text(
                        combo + "\n")

        def run(script, arm, task=None):
            log = root / f"py_{arm}_{task}.log"
            if log.exists():
                log.unlink()
            e = {k: v for k, v in _env().items()
                 if not k.startswith(("EVAL_", "PAIR_EXPORT_", "PY_"))
                 and k != "PER_HEAD"}
            e.update(SLURM_SUBMIT_DIR=str(sia), PCA_ARM=arm, PY_LOG=str(log),
                     PY_REAL=sys.executable)
            if task is not None:
                e["SLURM_ARRAY_TASK_ID"] = str(task)
            # the job's `sleep 5` and (bash 3) mapfile, neither under test
            r = subprocess.run(["bash", "-c", "sleep() { :; }; "
                                + _MAPFILE_SHIM + f". slurm/{script}"],
                               cwd=str(sia), env=e, capture_output=True,
                               text=True, timeout=120)
            calls = ([tuple(ln.split("\t")) for ln in
                      log.read_text().splitlines()] if log.exists() else [])
            return r.returncode, r.stdout + r.stderr, calls

        n = len(prefixes)
        for arm, res in roots.items():
            before = _snapshot(sia / "results")
            # the chained array: task = cv_index * n_variants + variant_index
            for task in range(len(cvs) * n):
                p = prefixes[task % n]
                rc, out, calls = run("eval_finetuning.sh", arm, task)
                assert rc == 0 and calls == [(
                    "eval_finetuning.py", f"{res}/{p}best_cat_a_cv3",
                    cvs[task // n])], (arm, task, calls, out[-1500:])
                assert re.search(rf"(?m)^PCA arm:\s+{arm} \(", out) and \
                    f"(marker {res}/{p}latest_combo_cv3.txt)" in out, \
                    (arm, task, out[:1500])
            rc, out, calls = run("export_pair_predictions.sh", arm)
            assert rc == 0 and calls == [
                ("export_pair_predictions.py", f"{res}/{p}best_cat_a_cv3",
                 "") for p in prefixes], (arm, calls, out[-1500:])
            assert f"PCA arm:   {arm} (" in out and \
                f"combos of {res}/" in out, out[:1500]
            # what changed is in this arm's cv3 combos only
            after = _snapshot(sia / "results")
            changed = sorted(k for k in set(before) | set(after)
                             if before.get(k) != after.get(k))
            own = [f"{res}/{p}best_cat_a_cv3/" for p in prefixes]
            assert changed and all(
                any(f"results/{k}".startswith(o) for o in own)
                for k in changed), (arm, changed)
        # no marker of the arm's own: no fallback to another root's
        (sia / roots["joint"] / "latest_combo_cv3.txt").unlink()
        rc, out, calls = run("eval_finetuning.sh", "joint", 0)
        assert rc == 1 and calls == [] and \
            "Marker results/pca_joint/latest_combo_cv3.txt missing" in out, \
            out[-1500:]
        rc, out, calls = run("export_pair_predictions.sh", "joint")
        assert rc == 1 and calls == [
            ("export_pair_predictions.py",
             f"{roots['joint']}/{p}best_cat_a_cv3", "")
            for p in prefixes[1:]], (calls, out[-1500:])
        assert "no marker results/pca_joint/latest_combo_cv3.txt" in out, \
            out[-1500:]


# --- the cell-line ablation job and grade.sh ------------------------------------
_CELL_STUB = '''"""$PYTHON_PATH of the cell-line job test: zero_cell_emb.py and
evaluate_model.py stand-ins. Each call is logged (cwd, script, arguments) and
writes what the job reads next: two folds' matrices per CV, one row per CV."""
import json
import os
import sys

script, args = sys.argv[1], sys.argv[2:]
with open(os.environ["CELL_LOG"], "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "script": script,
                        "args": args}) + "\\n")


def opt(name):
    return [args[i + 1] for i, a in enumerate(args) if a == name]


if script == "cell_line_ablation/zero_cell_emb.py":
    (out,), (name,) = opt("--out_dir"), opt("--cv_dir_name")
    for spec in opt("--model_dir"):
        cv, d = spec.split("=", 1)
        assert os.path.isdir(d), d
        os.makedirs(os.path.join(out, name % cv))
        for k in (0, 1):
            open(os.path.join(out, name % cv,
                              f"fold_{k}_{cv}_predictions.npy"), "wb").close()
elif script == "sl_comparison/evaluate_model.py":
    (out,), (pred,) = opt("--out"), opt("--pred_dir")
    assert os.path.isdir(pred), pred
    with open(out, "w") as f:
        json.dump({"model": opt("--model_name")[0], "pred_dir": pred}, f)
else:
    sys.exit(f"unexpected {script}")
'''

_CELL_VARIANTS = "pd:per_head bilinear:per_head pd:pooled"
_BILINEAR = "var_bilinear_per_head__"


def _cell_tree(tmp):
    """<tmp>/siamese_sl with copies of the cell-line job and grade.sh behind the
    real config.conf and the stub interpreter, two folds per CV, and in arms
    none and joint the same layout: pd:per_head combos (and markers) for every
    CV, bilinear:per_head ones for cv1 and cv2 only, a previous grading's rows
    (pd cv1, bilinear cv3) with their matrix dirs, and a row the job never
    writes."""
    root = Path(tmp)
    sia = root / "siamese_sl"
    (sia / "slurm").mkdir(parents=True)
    shutil.copytree(os.path.join(_SIAMESE, "cell_line_ablation"),
                    sia / "cell_line_ablation",
                    ignore=lambda d, names: [n for n in names
                                             if not n.endswith(".sh")])
    b = root / "bin"
    b.mkdir()
    (b / "cell_stub.py").write_text(_CELL_STUB)
    (b / "python").write_text(
        f'#!/bin/bash\nexec {shlex.quote(sys.executable)} '
        f'{shlex.quote(str(b / "cell_stub.py"))} "$@"\n')
    (b / "python").chmod(0o755)
    (sia / "slurm" / "config.conf").write_text(
        f'source "{os.path.join(_SLURM, "config.conf")}" || return 1\n'
        f'PYTHON_PATH="{b / "python"}"\n')
    for cv in ("cv1", "cv2", "cv3"):
        for k in (0, 1):
            (root / "sl_comparison" / "folds" / cv / f"fold_{k}").mkdir(
                parents=True)
    for arm in ("none", "joint"):
        res = "results" + ("/pca_joint" if arm == "joint" else "")
        cmpr = root / "sl_comparison" / res
        cmpr.mkdir(parents=True, exist_ok=True)
        for prefix, cvs in (("", ("cv1", "cv2", "cv3")),
                            (_BILINEAR, ("cv1", "cv2"))):
            for cv in cvs:
                combo = f"{res}/{prefix}best_cat_x_{cv}"
                (sia / combo).mkdir(parents=True)
                (sia / res / f"{prefix}latest_combo_{cv}.txt").write_text(
                    combo + "\n")
        for prefix, row, cv in (("", "siamese_sl", "cv1"),
                                (_BILINEAR, "siamese_sl_bilinear_per_head",
                                 "cv3")):
            old = sia / res / f"{prefix}cellline_ablation_{cv}"
            old.mkdir()
            (old / "fold_0_old.npy").write_text(arm)
            (cmpr / f"{row}_nocell_{cv}.json").write_text(
                json.dumps({"previous": arm}))
        (cmpr / "siamese_sl_cv1.json").write_text(json.dumps({"row": arm}))
    return sia


def _arm_files(sia, arm):
    """{path: mtime_ns} of everything arm `arm` owns, in both namespaces."""
    out = {}
    for top in (sia / "results", sia.parent / "sl_comparison" / "results"):
        d = top / "pca_joint" if arm == "joint" else top
        out.update({f"{top.parent.name}:{k}": v
                    for k, v in _snapshot(d).items()
                    if arm == "joint" or not k.startswith("pca_")})
    return out


def test_cell_line_job_and_grading_stay_in_their_arm():
    with tempfile.TemporaryDirectory() as tmp:
        sia = _cell_tree(tmp)
        root = sia.parent
        log = root / "cell_calls.jsonl"
        for arm, other in (("joint", "none"), ("none", "joint")):
            res = "results" + ("/pca_joint" if arm == "joint" else "")
            cmpr = root / "sl_comparison" / res
            before = _arm_files(sia, other)
            if log.exists():
                log.unlink()
            r = subprocess.run(
                ["bash", str(sia / "cell_line_ablation" /
                             "run_cell_line_ablation.sh")],
                cwd=str(sia), capture_output=True, text=True, timeout=300,
                env=_env(SLURM_SUBMIT_DIR=str(sia), PCA_ARM=arm,
                         SIAMESE_VARIANTS_LIST=_CELL_VARIANTS,
                         CELL_LOG=str(log)))
            out = r.stdout + r.stderr
            # bilinear has no cv3 combo: its row is MISSING, the job exits 1
            assert r.returncode == 1, (arm, out[-3000:])
            assert f"PCA arm: {arm} (" in out and \
                "MISSING rows (the paper tables flag them): " \
                "siamese_sl_bilinear_per_head_nocell/cv3" in out and \
                "N/A: pd:pooled" in out, (arm, out[-3000:])
            # every call reads and writes the arm's namespaces only
            calls = [json.loads(x) for x in log.read_text().splitlines()]
            zero = [c for c in calls if c["script"].startswith("cell_line")]
            ev = [c for c in calls if c["script"].startswith("sl_comparison")]
            assert len(zero) == 2 and len(ev) == 5, calls
            for c in zero:
                a = c["args"]
                assert Path(c["cwd"]).resolve() == sia.resolve() and \
                    a[a.index("--out_dir") + 1] == res, c
                assert all(Path(x.split("=", 1)[1]).parent == Path(res)
                           for i, x in enumerate(a)
                           if i and a[i - 1] == "--model_dir"), c
            for c in ev:
                a = c["args"]
                assert Path(c["cwd"]).resolve() == root.resolve() and \
                    a[a.index("--out") + 1] \
                    .startswith(f"sl_comparison/{res}/siamese_sl"), c
            # the new rows, graded from the arm's own matrices
            for row, prefix, cvs in (("siamese_sl", "", ("cv1", "cv2", "cv3")),
                                     ("siamese_sl_bilinear_per_head",
                                      _BILINEAR, ("cv1", "cv2"))):
                for cv in cvs:
                    got = json.loads(
                        (cmpr / f"{row}_nocell_{cv}.json").read_text())
                    assert got == {
                        "model": f"{row}_nocell",
                        "pred_dir": f"siamese_sl/{res}/"
                                    f"{prefix}cellline_ablation_{cv}"}, got
            # the previous grading is set aside, never left standing
            assert not (cmpr / "siamese_sl_bilinear_per_head_nocell_cv3.json"
                        ).exists()
            stale = sorted(p.name.split(".")[0] for p in
                           (cmpr / "stale_siamese").iterdir())
            assert stale == ["siamese_sl_bilinear_per_head_nocell_cv3",
                             "siamese_sl_nocell_cv1"], stale
            assert {json.loads(p.read_text())["previous"] for p in
                    (cmpr / "stale_siamese").iterdir()} == {arm}
            assert not (sia / res / "cellline_ablation_cv1" /
                        "fold_0_old.npy").exists()
            assert not (sia / res / f"{_BILINEAR}cellline_ablation_cv3").exists()
            assert json.loads((cmpr / "siamese_sl_cv1.json").read_text()) == {
                "row": arm}
            # and nothing of the other arm's changed
            assert _arm_files(sia, other) == before, other

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
            import traceback
            traceback.print_exc()
    print(f"\n==== {passed} passed, {failed} failed ====")
    sys.exit(1 if failed else 0)


def test_ablation_without_selection_keeps_the_pairs_step1_runs():
    # (2026-10-24) An ablation task of an arm whose selection is missing
    # names how to retrain it. Arm joint beside none ranks none's
    # single-embedding runs: the printed command must keep that pairing
    # (the combo resubmission, or every arm of the run), never
    # PCA_ARMS=joint alone, which makes joint its own Step-1 owner. A lone
    # joint owns its runs and keeps the one-arm command.
    with tempfile.TemporaryDirectory() as tmp:
        sia = _tree(tmp)
        arms4 = " ".join(PCA_ARMS)
        r = _job(sia, "run_ablation.sh", 0, PCA_ARM="joint", PCA_ARMS=arms4)
        out = r.stdout + r.stderr
        assert r.returncode == 1 and "best_per_category.json missing" in out, out[-2000:]
        assert "ranks the single-embedding runs in results/ (its pair's)" in out \
            and re.search(r"RUN_TOKEN=<that token> PCA_ARM=joint PCA_ARMS='"
                          + re.escape(arms4) + r"' (SELECTION_CV=cv\d )?PCA_MODALITY_VARIANCE=\S+ "
                          r"PCA_JOINT_VARIANCE=\S+ PCA_JOINT_DIM=\S* USE_CELL_LINES=\d "
                          r'SIAMESE_VARIANTS_LIST="[^"]+" sbatch '
                          r"--export=ALL slurm/run_best_per_category_combo\.sh",
                          out) \
            and (f'PCA_ARMS="{arms4}" bash slurm/submit_pipeline.sh '
                 "--best-cat-only --skip-generate --skip-reset-env") in out \
            and 'PCA_ARMS="joint" bash' not in out, out[-2000:]
        # the combo line ranks on the owner's selection CV, not this job's
        os.makedirs(os.path.join(sia, "results"), exist_ok=True)
        with open(os.path.join(sia, "results", "best_per_category.json"), "w") as f:
            json.dump({"_selection_meta": {"selection_cv": "cv3"}}, f, indent=2)
        r = _job(sia, "run_ablation.sh", 0, PCA_ARM="joint", PCA_ARMS=arms4)
        assert "SELECTION_CV=cv3 " in r.stdout + r.stderr, (r.stdout + r.stderr)[-2000:]
        os.remove(os.path.join(sia, "results", "best_per_category.json"))
        # a lone joint owns its runs and keeps the one-arm command; arm none
        # beside joint retrains with it (the pair shares none's runs)
        for arm, arms, want in (("joint", "joint", "joint"),
                                ("none", arms4, "none joint")):
            r = _job(sia, "run_ablation.sh", 0, PCA_ARM=arm, PCA_ARMS=arms)
            out = r.stdout + r.stderr
            assert r.returncode == 1 and (f'PCA_ARMS="{want}" bash slurm/submit_pipeline.sh '
                                          "--best-cat-only --skip-generate --skip-reset-env") in out \
                and "its pair's" not in out and "PCA_MODALITY_VARIANCE=" in out, (arm, out[-2000:])


if __name__ == "__main__":
    _main()
