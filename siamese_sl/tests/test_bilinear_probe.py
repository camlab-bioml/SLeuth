#!/usr/bin/env python3
"""The bilinear probe (reviewer 22DY): slurm/run_probe.sh + probe_args.py.

  * train.py's probe flag set builds exactly logit = tau * (W x_i)ᵀ(W x_j) + b
    (one bias-free Linear, nothing else trainable but tau and b), with one
    L1 weight matrix; without --no_input_bias the encoder's input offset
    would add a per-gene term, which is why the probe drops it.
  * probe_args.py builds the probe's train.py command from a pd:pooled
    combo's config.json (embeddings, fold_modalities, selection split,
    schedule) and refuses any other combo; the probe's arch signature differs
    from the combo's in the probe's own keys only.
  * run_probe.sh (stub train.py / evaluate_model.py in a scratch tree) trains
    with exactly those arguments, grades into BilinearProbe_<cv>.json, records
    per-fold sizes and the source combo, moves a previous row aside first and
    refuses a CV without its marker.

    python3 siamese_sl/tests/test_bilinear_probe.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

import probe_args as pa  # noqa: E402
import train as train_mod  # noqa: E402
from arch_signature import arch_signature, signature_diff  # noqa: E402
from siamese_esm import SiameseSLMultiCell  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 4668    # arm none's pd:pooled input width on the 2026-11-10 selection

# A pd:pooled combo's config.json as train.py saves it (arm none, defaults).
COMBO = {
    "embeddings_paths": ["/d/all_genes_geneformer.pt", "/d/all_genes_go.pt",
                         "/d/all_genes_text_embed.pt"],
    "sl_path": "/d/SL.txt", "neg_pairs_path": "/d/neg.txt",
    "use_cell_lines": True, "folds_dir": "../sl_comparison/folds",
    "cell_line_dim": 24, "per_head_input_bias": False,
    "per_head_temperature": False, "select_metric": "auprg_any",
    "cell_lines": "pooled", "kernel": "pd", "gene_term": True,
    "pair_features": True, "per_head_last_layer_bias": True,
    "log_temperature": False, "inner_norm": "none", "layernorm": False,
    "output_dir": "results/var_pd_pooled__best_cat_x_cv2",
    "model_type": "siamese", "input_dim": None, "hidden_dim": 512,
    "latent_dim": 256, "dropout": 0.2, "last_layer_bias": False,
    "pd_epsilon": 0.001, "encoder_dims": [32] * 7,
    "siamese_encoder_type": "residual", "epochs": 2000, "batch_size": 256,
    "learning_rate": 0.001, "weight_decay": 1e-05, "pca_dims": None,
    "pca_variance": None, "post_pca_variance": None, "post_pca_dim": None,
    "no_post_pca": True, "preprocessing_fit_scope": "train",
    "pca_method": "plain",
    "l1_lambdas": [0.01, 0.0075, 0.005, 0.004, 0.0025, 0.0015, 0.001],
    "eval_interval": 30, "patience": 10, "warmrestart_T0": 50,
    "warmrestart_Tmult": 2, "cv_type": "cv2", "num_folds": 5,
    "pos_neg_ratio": 1.0, "seed": 42, "post_pca": None,
    "selection_frac": 0.1, "selection_seed": 7,
    "fold_modalities": '{"0": [0, 1], "1": [0, 2], "2": [1, 2], '
                       '"3": [0, 1, 2], "4": [0, 1]}',
}
# The keys in which the probe is meant to differ from its combo.
PROBE_KEYS = {"encoder_dims", "pd_epsilon", "gene_term", "pair_features",
              "per_head_last_layer_bias", "input_bias", "l1_lambdas"}


def _parsed_args(argv):
    """train.py main()'s args namespace for argv (Trainer replaced)."""
    got = {}

    class _Stub:
        def __init__(self, args):
            got["args"] = args

        def train(self):
            pass

    real, old_argv = train_mod.Trainer, sys.argv
    train_mod.Trainer, sys.argv = _Stub, ["train.py"] + list(argv)
    try:
        train_mod.main()
    finally:
        train_mod.Trainer, sys.argv = real, old_argv
    return got["args"]


def _build(argv, k=K):
    """The model train.py's create_model builds for argv at input width k."""
    args = _parsed_args(argv)
    args.input_dim = k
    fake = types.SimpleNamespace(
        args=args, device=DEV,
        data_manager=types.SimpleNamespace(pool_cell_lines=True))
    return args, train_mod.Trainer.create_model(fake)


def _l1_weights(model):
    """train.py's --l1_lambdas weight list (Trainer.train's filter)."""
    return [n for n, p in model.named_parameters()
            if p.dim() >= 2 and not n.startswith(("cell_", "kernel_", "pair_"))
            and not n.endswith("bias")]


def _probe_argv(cfg=COMBO, cv="cv2"):
    return pa.probe_train_args(cfg, cv, "results/bilinear_probe_" + cv)


def test_flag_set_builds_the_pure_bilinear_model():
    args, m = _build(_probe_argv())
    assert isinstance(m, SiameseSLMultiCell) and m.num_heads == 1
    shapes = {n: tuple(p.shape) for n, p in m.named_parameters()}
    assert shapes == {"encoder.projection.weight": (32, K), "tau": (),
                      "scoring_bias": (1, )}, shapes
    assert m.encoder.input_bias is None and m.encoder.projection.bias is None
    assert len(m.encoder.hidden_blocks) == 0 and m.cell_emb is None
    assert m.head_proj_bias is None and m.head_input_bias is None
    assert m.gene_w is None and m.pair_fc1 is None and m.kernel_A is None
    assert m.pd_epsilon == 0.0 and m.inner_norm == "none"
    # parameter count: 32 * k + tau + b
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 32 * K + 2 == 149378, n
    # --l1_lambdas needs exactly one value, the combo's first
    assert _l1_weights(m) == ["encoder.projection.weight"]
    assert args.l1_lambdas == [0.01]
    # the logit IS tau * (W x_i)ᵀ(W x_j) + b, for any parameter values
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        m.tau.fill_(-0.37)
        m.scoring_bias.fill_(0.81)
        m.encoder.projection.weight.copy_(torch.randn(32, K, generator=g))
    m.eval()
    x1 = torch.randn(64, K, generator=g).to(DEV)
    x2 = torch.randn(64, K, generator=g).to(DEV)
    W = m.encoder.projection.weight
    with torch.no_grad():
        want = -0.37 * ((x1 @ W.T) * (x2 @ W.T)).sum(1) + 0.81
        got = m(x1, x2)[:, 0]
        # symmetric, and train mode (no dropout anywhere) gives the same
        m.train()
        got_train = m(x2, x1)[:, 0]
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-3), \
        (got - want).abs().max()
    assert torch.allclose(got_train, got, rtol=1e-5, atol=1e-4)


def test_input_offset_would_add_a_per_gene_term():
    """Without --no_input_bias the hidden-free encoder computes W (x + a): a
    projection bias W a in disguise, i.e. per-gene terms <W a, W x>."""
    argv = [w for w in _probe_argv() if w != "--no_input_bias"]
    _, m = _build(argv, k=40)
    assert tuple(m.encoder.input_bias.shape) == (40, )
    n = sum(p.numel() for p in m.parameters())
    assert n == 32 * 40 + 40 + 2, n
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        m.encoder.input_bias.copy_(torch.randn(40, generator=g))
    m.eval()
    x1 = torch.randn(16, 40, generator=g).to(DEV)
    x2 = torch.randn(16, 40, generator=g).to(DEV)
    W, a = m.encoder.projection.weight, m.encoder.input_bias
    with torch.no_grad():
        tau = m.temperature()
        pure = tau * ((x1 @ W.T) * (x2 @ W.T)).sum(1)
        c = W @ a
        gene = tau * ((x1 @ W.T) @ c + (x2 @ W.T) @ c + c @ c)
        got = m(x1, x2)[:, 0] - m.scoring_bias
    assert torch.allclose(got, pure + gene, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(got, pure, rtol=1e-3, atol=1e-3)


def test_no_input_bias_refusals():
    try:
        SiameseSLMultiCell(input_dim=8, num_heads=1, encoder_dims=[4],
                           cell_line_dim=0, per_head_input_bias=True,
                           input_bias=False)
        raise AssertionError("per_head_input_bias + input_bias=False built")
    except ValueError as e:
        assert "per_head_input_bias" in str(e)
    # train.py: refused before config.json is written (no data is read)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "o"
        (out / "checkpoints").mkdir(parents=True)
        for extra, use_cl in ((["--per_head_input_bias"], True), ([], False)):
            argv = (["--embeddings_paths", "x.pt", "--folds_dir", tmp,
                     "--neg_pairs_path", "n.txt", "--output_dir", str(out),
                     "--no_input_bias"] + extra +
                    (["--use_cell_lines"] if use_cl else []))
            args = _parsed_args(argv)
            try:
                train_mod.Trainer(args)
                raise AssertionError(f"train.py accepted {argv}")
            except SystemExit as e:
                assert "--no_input_bias" in str(e), e
            assert not (out / "config.json").exists()
    # default: every existing run keeps its input offset, signature included
    assert arch_signature({"use_cell_lines": True})["input_bias"] is True
    assert signature_diff(COMBO, dict(COMBO, input_bias=True)) == {}


def test_probe_args_from_a_combo_config():
    a = _probe_argv()
    i = a.index("--embeddings_paths")
    assert a[i + 1:i + 4] == COMBO["embeddings_paths"] and \
        a[i + 4].startswith("--")
    # fold_modalities stays one word, exactly as the combo received it
    assert a[a.index("--fold_modalities") + 1] == COMBO["fold_modalities"]
    for flag, val in (("--sl_path", "/d/SL.txt"),
                      ("--neg_pairs_path", "/d/neg.txt"),
                      ("--folds_dir", "../sl_comparison/folds"),
                      ("--cv_type", "cv2"),
                      ("--output_dir", "results/bilinear_probe_cv2"),
                      ("--selection_frac", "0.1"), ("--selection_seed", "7"),
                      ("--epochs", "2000"), ("--learning_rate", "0.001"),
                      ("--eval_interval", "30"), ("--patience", "10"),
                      ("--seed", "42"), ("--select_metric", "auprg_any"),
                      ("--encoder_dims", "32"), ("--pd_epsilon", "0"),
                      ("--l1_lambdas", "0.01"), ("--cell_lines", "pooled"),
                      ("--kernel", "pd"), ("--pca_method", "plain")):
        assert a[a.index(flag) + 1] == val, (flag, a)
    for flag in ("--use_cell_lines", "--no_gene_term", "--no_pair_features",
                 "--no_per_head_last_layer_bias", "--no-last-layer-bias",
                 "--no_input_bias", "--no_post_pca"):
        assert a.count(flag) == 1, flag
    assert "--log_temperature" not in a and "--pca_variance" not in a
    # every parsed training / data setting is the combo's
    args = _parsed_args(a)
    for k in ("embeddings_paths", "sl_path", "neg_pairs_path", "folds_dir",
              "fold_modalities", "selection_frac", "selection_seed",
              "epochs", "batch_size", "learning_rate", "weight_decay",
              "eval_interval", "patience", "warmrestart_T0",
              "warmrestart_Tmult", "num_folds", "seed", "select_metric",
              "preprocessing_fit_scope", "pca_method", "post_pca",
              "pca_dims", "dropout", "cv_type"):
        assert getattr(args, k) == COMBO[k], (k, getattr(args, k), COMBO[k])
    # ... so the signatures differ in the probe's own keys only
    assert set(signature_diff(vars(args), COMBO)) == PROBE_KEYS, \
        signature_diff(vars(args), COMBO)
    # global selection: no fold_modalities, no split
    glob = {k: v for k, v in COMBO.items()
            if k not in ("fold_modalities", "selection_frac",
                         "selection_seed")}
    g = _probe_argv(glob)
    assert "--fold_modalities" not in g and "--selection_frac" not in g
    assert _parsed_args(g).fold_modalities is None
    # a combo trained with exp(log tau) / LayerNorm: the probe follows
    lt = _probe_argv(dict(COMBO, log_temperature=True, layernorm=True))
    assert "--log_temperature" in lt and "--layernorm" in lt
    assert set(signature_diff(vars(_parsed_args(lt)), dict(
        COMBO, log_temperature=True, layernorm=True))) == PROBE_KEYS


def test_probe_args_refuses_another_combo():
    for bad, why in ((dict(COMBO, cell_lines="per_head"), "cell_lines"),
                     (dict(COMBO, kernel="bilinear"), "kernel"),
                     (dict(COMBO, cv_type="cv1"), "cv_type"),
                     (dict(COMBO, use_cell_lines=False), "use_cell_lines"),
                     (dict(COMBO, post_pca=0.8, post_pca_variance=0.8,
                           no_post_pca=False), "PCA arm joint"),
                     (dict(COMBO, pca_variance=0.8, pca_dims=[0.8] * 3),
                      "PCA arm per_modality"),
                     (dict(COMBO, folds_dir=None), "folds_dir")):
        try:
            _probe_argv(bad)
            raise AssertionError(f"accepted {why}")
        except pa.ProbeRefused as e:
            assert why in str(e), (why, str(e))
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, os.path.join(_SIAMESE,
                                                         "probe_args.py"),
                            "--cv", "cv1", "--results_root", tmp],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 2 and r.stdout == "" and \
            "var_pd_pooled__latest_combo_cv1.txt" in r.stderr, r


def test_real_config_prefix_and_script_header():
    """The marker the script reads is config.conf's own pd:pooled prefix, and
    the script is a GPU job on gpu3 that sources config.conf."""
    r = subprocess.run(
        ["bash", "-c", 'unset SLURM_JOB_ID; source "$1" >/dev/null 2>&1 || '
         'exit 9; siamese_variant_prefix pd:pooled; echo "${CV_TYPES[*]}"; '
         'echo "$(pca_arm_results none)"', "_",
         os.path.join(_SIAMESE, "slurm", "config.conf")],
        capture_output=True, text=True, timeout=120, cwd=_SIAMESE)
    assert r.returncode == 0, r
    prefix, cvs, root = r.stdout.splitlines()[-3:]
    assert prefix == pa.MARKER_PREFIX and cvs == "cv1 cv2 cv3" and \
        root == "results", r.stdout
    sh = Path(_SIAMESE, "slurm", "run_probe.sh").read_text()
    for line in ("#SBATCH --gres=gpu:1", "#SBATCH --nodelist=gpu3",
                 "#SBATCH --array=0-2",
                 'source "$SUBMIT_DIR/slurm/config.conf"',
                 'export PCA_ARM=none', '--model_name "$PROBE_ROW"',
                 'PROBE_ROW="BilinearProbe"', 'VARIANT="pd:pooled"'):
        assert line in sh, line
    assert "--cpu " not in sh and "--cpu\n" not in sh   # never forces CPU


# ---------------------------------------------------------------------------
# run_probe.sh end to end, in a scratch tree with stub train.py / grader
# ---------------------------------------------------------------------------

_STUB_CONFIG = r'''
PYTHON_PATH="__PY__"
CV_TYPES=("cv1" "cv2" "cv3")
RESULTS_ROOT=results
siamese_variant_prefix() {
    local t; if [ "$1" = "pd:per_head" ]; then t=""; else t="${1%%:*}_${1#*:}"; fi
    echo "${t:+var_${t}__}"
}
'''

_STUB_TRAIN = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--output_dir") + 1]
cv = argv[argv.index("--cv_type") + 1]
with open(os.environ["PROBE_TEST_LOG"], "a") as f:
    f.write(json.dumps(argv) + "\n")
json.dump({"fold_metrics": [{"total_params": 149378 + k} for k in range(5)]},
          open(os.path.join(out, "results.json"), "w"))
for k in range(5):
    open(os.path.join(out, f"fold_{k}_{cv}_predictions.npy"), "w").write("x")
'''

_STUB_GRADER = r'''
import argparse, json
ap = argparse.ArgumentParser()
for a in ("--model_name", "--cv_type", "--pred_dir", "--out"):
    ap.add_argument(a)
a = ap.parse_args()
json.dump({"summary": {"model": a.model_name, "cv_type": a.cv_type,
                       "pred_dir": a.pred_dir, "auprg_mean": 0.5}},
          open(a.out, "w"))
'''


def _script_tree(tmp: Path):
    sia, cmp_ = tmp / "siamese_sl", tmp / "sl_comparison"
    (sia / "slurm").mkdir(parents=True)
    shutil.copy(os.path.join(_SIAMESE, "slurm", "run_probe.sh"),
                sia / "slurm" / "run_probe.sh")
    (sia / "slurm" / "config.conf").write_text(
        _STUB_CONFIG.replace("__PY__", sys.executable))
    for f in ("probe_args.py", "arch_signature.py"):
        shutil.copy(os.path.join(_SIAMESE, f), sia / f)
    (sia / "train.py").write_text(_STUB_TRAIN)
    (cmp_ / "results").mkdir(parents=True)
    (cmp_ / "evaluate_model.py").write_text(_STUB_GRADER)
    for cv in ("cv1", "cv2", "cv3"):
        for k in range(5):
            (cmp_ / "folds" / cv / f"fold_{k}").mkdir(parents=True)
    # cv2's combo (a marker and config); cv1 has neither
    combo = sia / "results" / "var_pd_pooled__best_cat_x_cv2"
    combo.mkdir(parents=True)
    (combo / "config.json").write_text(json.dumps(COMBO))
    # every trained combo has one; probe_source records its sha256
    (combo / "results.json").write_text(json.dumps({"fold_metrics": []}))
    (sia / "results" / "var_pd_pooled__latest_combo_cv2.txt").write_text(
        "results/var_pd_pooled__best_cat_x_cv2\n")
    # previous rows, to be moved aside
    for cv in ("cv1", "cv2"):
        (cmp_ / "results" / f"BilinearProbe_{cv}.json").write_text("{}")
    return sia, cmp_


def _run_script(sia, cvs, log):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "PCA_"))}
    env.update(PROBE_CVS=cvs, PROBE_TEST_LOG=str(log), PCA_ARM="joint")
    return subprocess.run(["bash", "slurm/run_probe.sh"], cwd=sia, env=env,
                          capture_output=True, text=True, timeout=300)


def test_run_probe_script_trains_grades_and_refuses():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sia, cmp_ = _script_tree(tmp)
        log = tmp / "train_calls.jsonl"
        r = _run_script(sia, "cv2", log)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        assert "PCA_ARM=joint ignored" in out, out
        calls = [json.loads(ln) for ln in log.read_text().splitlines()]
        assert calls == [_probe_argv()], calls          # exactly probe_args'
        row = json.loads((cmp_ / "results" / "BilinearProbe_cv2.json")
                         .read_text())["summary"]
        assert row["model"] == "BilinearProbe" and row["cv_type"] == "cv2"
        assert row["pred_dir"] == "siamese_sl/results/bilinear_probe_cv2"
        assert row["trainable_params_fold_n"] == 5 and \
            row["trainable_params_fold_min"] == 149378 and \
            row["trainable_params_fold_max"] == 149382
        src = row["probe_source"]
        assert src["combo_dir"] == "results/var_pd_pooled__best_cat_x_cv2"
        assert src["marker"] == "results/var_pd_pooled__latest_combo_cv2.txt"
        assert len(src["combo_config_sha256"]) == 64
        assert len(src["combo_results_sha256"]) == 64
        stale = sorted(p.name for p in
                       (cmp_ / "results" / "stale_shared").iterdir())
        assert len(stale) == 1 and stale[0].startswith("BilinearProbe_cv2."), \
            stale
        # cv1: no marker -> refused, nothing trained, its old row moved aside
        log.unlink()
        r = _run_script(sia, "cv1", log)
        out = r.stdout + r.stderr
        assert r.returncode == 1 and "REFUSED" in out, out
        assert not log.exists()
        assert not (cmp_ / "results" / "BilinearProbe_cv1.json").exists()
        assert not (sia / "results" / "bilinear_probe_cv1").exists()
        assert (cmp_ / "results" / "BilinearProbe_cv2.json").exists()
        # a marker naming a per_head combo: probe_args refuses, nothing runs
        (sia / "results" / "var_pd_pooled__latest_combo_cv1.txt").write_text(
            "results/var_pd_pooled__best_cat_x_cv2\n")
        r = _run_script(sia, "cv1", log)
        out = r.stdout + r.stderr
        assert r.returncode == 1 and "REFUSED" in out and "cv_type" in out, out
        assert not log.exists()


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
