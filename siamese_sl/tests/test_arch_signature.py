#!/usr/bin/env python3
"""arch_signature: which config differences make two runs incomparable.

run_ablation.sh refuses an ablation whose signature differs from its
reference combo, and run_ablation_summary.sh / combo Step 2 use the same
check. Covers: a variance-target PCA is compared as the target, not as the
modality-count-long list train.py expands it to; an unset --encoder_dims
falls back to [hidden_dim, latent_dim]; configs predating a key compare as
the value those runs used (published scorer) or never match (pre-0767425
preprocessing); options a mode ignores are ignored; and `train.py
--print_arch` (no data loaded) prints the same signature, with the rerun
defaults (no LayerNorm, raw tau, per-head c_h, auprg_any selection).
The PCA arms (slurm/config.conf: the whole pipeline once per arm) are read
from a run with pca_settings / pca_arm, which say the same of a config, of its
signature and of train.py --print_arch, for every arm; config.conf's
pca_multi_args / pca_single_args resolve, through --print_arch, to their own
arm; paper_tables' cross-arm table compares the arms' combos with the four
PCA_SIGNATURE_KEYS left out (signature_diff(..., ignore=)). The orchestrators'
PCA check of the combos they reuse (--check_pca, 2026-10-21): pca_flags_config
resolves every arm's flags as train.py --print_arch does (any target, any
spelling of a fraction) and refuses what its parser refuses;
pca_target_needs names the config.conf targets that train a combo's PCA,
which config.conf turns back into that PCA; the command line's verdicts.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_arch_signature.py -q
    $PYTHON_PATH siamese_sl/tests/test_arch_signature.py
"""
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

from arch_signature import (PCA_ARMS, PCA_SIGNATURE_KEYS, arch_signature,
                            pca_arm, pca_flags_config, pca_matches,
                            pca_multi_flags, pca_settings, pca_target_needs,
                            signature_diff)

# A cell-line config as train.py saves it with the rerun defaults.
NOW = dict(use_cell_lines=True, cell_lines="per_head", kernel="pd",
           layernorm=False, log_temperature=False,
           per_head_last_layer_bias=True, per_head_input_bias=False,
           per_head_temperature=False, inner_norm="none",
           select_metric="auprg_any", model_type="siamese", cell_line_dim=8,
           siamese_encoder_type="residual", encoder_dims=[16] * 7,
           last_layer_bias=False, dropout=0.2, pd_epsilon=0.001,
           l1_lambdas=None, pca_method="plain",
           preprocessing_fit_scope="train", pca_variance=None,
           pca_dims=None, post_pca=0.8, post_pca_dim=None, hidden_dim=512,
           latent_dim=256, learning_rate=1e-3, weight_decay=1e-5,
           batch_size=256, epochs=200, eval_interval=10, patience=20,
           warmrestart_T0=50, warmrestart_Tmult=2, seed=42)


def test_identical_configs_match():
    assert signature_diff(NOW, dict(NOW)) == {}


def test_extra_score_terms_are_compared():
    """--gene_term / --pair_features: absent in an older config means off."""
    assert signature_diff(NOW, dict(NOW, gene_term=False,
                                    pair_features=False)) == {}
    assert set(signature_diff(NOW, dict(NOW, gene_term=True))) == {
        "gene_term"}
    assert set(signature_diff(NOW, dict(NOW, pair_features=True))) == {
        "pair_features"}
    # without --use_cell_lines neither option exists
    single = dict(NOW, use_cell_lines=False)
    assert signature_diff(single, dict(single, gene_term=True)) == {}


def test_pca_variance_is_compared_as_the_target():
    a = dict(NOW, pca_variance=0.5, pca_dims=[0.5] * 5)
    assert set(signature_diff(NOW, a)) == {"pca_variance"}
    # An ablation drops a modality, so the expanded list is one shorter.
    assert signature_diff(a, dict(a, pca_dims=[0.5] * 4)) == {}


def test_explicit_pca_dims_are_compared():
    assert "pca_dims" in signature_diff(NOW, dict(NOW, pca_dims=[50, 64]))


def test_unset_encoder_dims_fall_back_to_hidden_latent():
    e = dict(NOW, encoder_dims=None)
    assert arch_signature(e)["encoder_dims"] == [512.0, 256.0]
    assert "encoder_dims" in signature_diff(
        e, dict(e, hidden_dim=64, latent_dim=16))


def test_pre_0767425_config_never_matches():
    legacy = {k: v for k, v in NOW.items()
              if k not in ("pca_method", "preprocessing_fit_scope",
                           "siamese_encoder_type")}
    assert set(signature_diff(legacy, NOW)) == {
        "pca_method", "preprocessing_fit_scope", "siamese_encoder_type"}


def test_pre_2026_09_21_config_is_the_published_scorer():
    new_keys = ("cell_lines", "kernel", "layernorm", "log_temperature",
                "per_head_last_layer_bias", "per_head_input_bias",
                "per_head_temperature", "inner_norm", "select_metric")
    old = {k: v for k, v in NOW.items() if k not in new_keys}
    published = dict(NOW, layernorm=True, log_temperature=True,
                     per_head_last_layer_bias=False, select_metric="auprg")
    assert signature_diff(old, published) == {}
    assert set(signature_diff(old, NOW)) == {
        "layernorm", "log_temperature", "per_head_last_layer_bias",
        "select_metric"}


def test_ignored_options_are_ignored():
    pooled = dict(NOW, cell_lines="pooled")
    assert signature_diff(pooled, dict(pooled, cell_line_dim=24)) == {}
    assert "cell_line_dim" in signature_diff(NOW, dict(NOW, cell_line_dim=24))
    single = dict(NOW, use_cell_lines=False)
    assert signature_diff(single, dict(single, layernorm=True,
                                       select_metric="auprg",
                                       kernel="bilinear")) == {}
    # the cell-line model ignores --model_type
    assert signature_diff(NOW, dict(NOW, model_type="other")) == {}


def test_print_arch_matches_and_keeps_the_rerun_defaults():
    # --print_arch loads no data (x.pt need not exist) and exits.
    out = subprocess.run(
        [sys.executable, os.path.join(_SIAMESE, "train.py"),
         "--embeddings_paths", "x.pt", "y.pt", "--use_cell_lines",
         "--encoder_dims", "16", "16", "--pca_variance", "0.5",
         "--print_arch"], capture_output=True, text=True, cwd=_SIAMESE,
        timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    sig = json.loads(out.stdout.strip().splitlines()[-1])
    assert sig["pca_variance"] == 0.5 and sig["pca_dims"] is None
    assert sig["encoder_dims"] == [16.0, 16.0]
    assert (sig["layernorm"], sig["log_temperature"],
            sig["per_head_last_layer_bias"], sig["select_metric"]) == (
                False, False, True, "auprg_any")
    assert sig["kernel"] == "pd" and sig["cell_lines"] == "per_head"
    # both extra score terms are on by default since 2026-10-05
    assert sig["gene_term"] is True and sig["pair_features"] is True
    assert set(sig) == set(arch_signature({}))


def test_extra_score_terms_can_be_switched_off_and_skip_single_output():
    def sig(*flags):
        out = subprocess.run(
            [sys.executable, os.path.join(_SIAMESE, "train.py"),
             "--embeddings_paths", "x.pt", "--print_arch", *flags],
            capture_output=True, text=True, cwd=_SIAMESE, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        return json.loads(out.stdout.strip().splitlines()[-1])

    off = sig("--use_cell_lines", "--no_gene_term", "--no_pair_features")
    assert off["gene_term"] is False and off["pair_features"] is False
    one = sig("--use_cell_lines", "--no_pair_features")
    assert one["gene_term"] is True and one["pair_features"] is False
    # single-output mode has neither term: not part of its signature
    single = sig()
    assert single["gene_term"] is None and single["pair_features"] is None


# NOW under each PCA arm, the keys as train.py resolves them (NOW: joint 0.8)
_NONE = dict(NOW, post_pca=None)
_ARM_CFGS = {"none": _NONE, "joint": NOW,
             "per_modality": dict(_NONE, pca_variance=0.8, pca_dims=[0.8] * 3),
             "both": dict(NOW, pca_variance=0.8, pca_dims=[0.8] * 3)}


def test_pca_keys_and_arms():
    assert PCA_SIGNATURE_KEYS == ("pca_variance", "pca_dims", "post_pca",
                                  "post_pca_dim")
    assert set(PCA_SIGNATURE_KEYS) < set(arch_signature({}))
    # the estimators and the fit rows stay compared: an arm must match its
    # reference there
    assert not {"pca_method", "preprocessing_fit_scope"} & set(
        PCA_SIGNATURE_KEYS)
    # the 2x2, in the order config.conf runs the arms in (the order it
    # normalises any list to, all four included)
    assert PCA_ARMS == ("none", "joint", "per_modality", "both")
    assert _conf("echo \"$PCA_ARMS\"",
                 PCA_ARMS="both per_modality joint none") == [
        "none joint per_modality both"]
    assert _conf("echo \"$PCA_ARMS\"", PCA_ARMS="both none joint") == [
        "none joint both"]
    # config.conf's default PCA_ARMS is the paper's protocol (2026-11-13):
    # no PCA and the joint PCA, not the whole 2x2
    assert _conf("echo \"$PCA_ARMS\"") == ["none joint"]


def test_signature_diff_can_leave_out_the_pca_keys():
    for a in _ARM_CFGS.values():
        for b in _ARM_CFGS.values():
            assert signature_diff(a, b, ignore=PCA_SIGNATURE_KEYS) == {}
    # by default they are compared, as before (run_ablation.sh, the LOCO
    # summary)
    assert signature_diff(_ARM_CFGS["none"], _ARM_CFGS["both"]) == {
        "pca_variance": (None, 0.8), "post_pca": (None, 0.8)}
    assert set(signature_diff(NOW, dict(NOW, post_pca=512,
                                         post_pca_dim=512))) == {
        "post_pca", "post_pca_dim"}
    # ... and only the named keys are left out
    other = dict(_ARM_CFGS["both"], dropout=0.3, pca_method="svd",
                 preprocessing_fit_scope="all")
    assert set(signature_diff(_ARM_CFGS["none"], other,
                              ignore=PCA_SIGNATURE_KEYS)) == {
        "dropout", "pca_method", "preprocessing_fit_scope"}
    assert set(signature_diff(NOW, dict(NOW, post_pca=None, seed=1),
                              ignore=("post_pca",))) == {"seed"}


def test_pca_settings_and_arm_of_a_config():
    for cfg, mod, joint, arm in (
            (_ARM_CFGS["none"], None, None, "none"),
            (_ARM_CFGS["joint"], None, ("variance", 0.8), "joint"),
            (dict(NOW, post_pca=512, post_pca_dim=512), None, ("count", 512),
             "joint"),
            (_ARM_CFGS["per_modality"], ("variance", 0.8), None,
             "per_modality"),
            (dict(_NONE, pca_dims=[50, 64, 64]), ("dims", [50, 64, 64]), None,
             "per_modality"),
            # an expanded --pca_variance without its pca_variance key
            (dict(_NONE, pca_dims=[0.8, 0.8]), ("variance", 0.8), None,
             "per_modality"),
            (_ARM_CFGS["both"], ("variance", 0.8), ("variance", 0.8), "both"),
            # --no_post_pca next to a count: post_pca is what train.py ran
            (dict(_NONE, post_pca_dim=512), None, None, "none"),
            # a config predating every PCA key ran none
            ({}, None, None, "none")):
        want = {"modality": mod, "joint": joint}
        assert pca_settings(cfg) == want, (cfg, pca_settings(cfg))
        # the signature (what train.py --print_arch prints) says the same
        assert pca_settings(arch_signature(cfg)) == want, cfg
        assert pca_arm(cfg) == pca_arm(arch_signature(cfg)) == arm, cfg


def _conf(script, **env):
    """stdout lines of `script` run after sourcing slurm/config.conf (as a
    job does, minus its compute-node block), with env on top."""
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "PCA_", "POST_PCA_"))}
    e.update(env)
    out = subprocess.run(
        ["bash", "-c", 'source slurm/config.conf >/dev/null || exit 9; '
         + script], cwd=_SIAMESE, env=e, capture_output=True, text=True,
        timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.splitlines()


def test_config_conf_pca_flags_resolve_to_their_arm():
    # every arm's multi-modal flags (combo, leave-one-category-out) and
    # single-embedding flags (Step 1, the benchmark), as config.conf builds
    # them, read back through train.py --print_arch: the multi-modal flags are
    # the arm itself; a single-embedding run gets the per-modality stage only,
    # so none / joint train natively and per_modality / both with the
    # per-modality PCA
    single_arm = {"none": "none", "joint": "none",
                  "per_modality": "per_modality", "both": "per_modality"}
    for extra, mod, joint in (
            ({}, ("variance", 0.8), ("variance", 0.8)),
            ({"PCA_MODALITY_VARIANCE": "0.7", "PCA_JOINT_DIM": "24"},
             ("variance", 0.7), ("count", 24))):
        for arm in PCA_ARMS:
            multi, single = _conf('echo "$PCA_MULTI_ARGS"; '
                                  'echo "$PCA_SINGLE_ARGS"',
                                  PCA_ARM=arm, **extra)
            s = _print_arch(*multi.split())
            assert pca_arm(s) == arm, (arm, multi, s)
            want = {"modality": mod if arm in ("per_modality", "both")
                    else None,
                    "joint": joint if arm in ("joint", "both") else None}
            assert pca_settings(s) == want, (arm, multi, pca_settings(s))
            s1 = _print_arch(*single.split())
            assert pca_arm(s1) == single_arm[arm], (arm, single, s1)
            assert pca_settings(s1)["joint"] is None, (arm, single)


def _print_arch(*flags):
    out = subprocess.run(
        [sys.executable, os.path.join(_SIAMESE, "train.py"),
         "--embeddings_paths", "x.pt", "y.pt", "--use_cell_lines",
         "--print_arch", *flags], capture_output=True, text=True,
        cwd=_SIAMESE, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_pca_arm_agrees_with_print_arch():
    # paper_tables and the combo selection read a run's arm from its config;
    # for every way of spelling a PCA on train.py's command line, the arm and
    # settings --print_arch resolves are the flags' own
    def sig(*flags):
        out = subprocess.run(
            [sys.executable, os.path.join(_SIAMESE, "train.py"),
             "--embeddings_paths", "x.pt", "y.pt", "--use_cell_lines",
             "--print_arch", *flags], capture_output=True, text=True,
            cwd=_SIAMESE, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        return json.loads(out.stdout.strip().splitlines()[-1])

    sigs = []
    for flags, (arm, mod, joint) in {
            ("--no_post_pca",): ("none", None, None),
            ("--post_pca_variance", "0.8"): ("joint", None, ("variance", 0.8)),
            ("--post_pca_dim", "512"): ("joint", None, ("count", 512)),
            ("--pca_variance", "0.8", "--no_post_pca"): (
                "per_modality", ("variance", 0.8), None),
            ("--pca_dims", "50", "64"): ("per_modality", ("dims", [50, 64]),
                                         None),
            ("--pca_variance", "0.7", "--post_pca_dim", "24"): (
                "both", ("variance", 0.7), ("count", 24)),
            # train.py ignores the count, the signature keeps it: config.conf
            # never passes one next to --no_post_pca
            ("--no_post_pca", "--post_pca_dim", "512"): ("none", None, None),
    }.items():
        s = sig(*flags)
        assert pca_arm(s) == arm and pca_settings(s) == {
            "modality": mod, "joint": joint}, (flags, s)
        sigs.append(s)
    assert sigs[-1]["post_pca_dim"] == 512 and sigs[-1]["post_pca"] is None
    # outside the PCA keys every arm has one signature
    for s in sigs:
        assert signature_diff(sigs[0], s, ignore=PCA_SIGNATURE_KEYS) == {}


def test_config_conf_reads_every_line_of_a_list():
    # PCA_ARMS and SIAMESE_VARIANTS_LIST with one entry per line ($(cat
    # list.txt)) are read whole, as with spaces: a plain `read` took the first
    # line only and dropped the rest without a refusal until 2026-10-21 (a
    # name after the newline that config.conf refuses: test_slurm_contracts)
    assert _conf('echo "$PCA_ARMS|${PCA_ARM_LIST[*]}"',
                 PCA_ARMS="both\nnone") == ["none both|none both"]
    assert _conf('echo "${SIAMESE_VARIANTS[*]}|$SIAMESE_VARIANTS_LIST"',
                 SIAMESE_VARIANTS_LIST="pd:pooled\nbilinear:pooled\n",
                 USE_CELL_LINES="1") == [
        "pd:pooled bilinear:pooled|pd:pooled bilinear:pooled"]


# Targets the PCA arms take (config.conf PCA_MODALITY_VARIANCE /
# PCA_JOINT_VARIANCE / PCA_JOINT_DIM): the defaults, an exact count with
# another modality fraction, another joint fraction, and fractions spelled with
# trailing zeros (0.80 is 0.8).
_TARGETS = ({}, {"PCA_MODALITY_VARIANCE": "0.7", "PCA_JOINT_DIM": "24"},
            {"PCA_JOINT_VARIANCE": "0.65"},
            {"PCA_MODALITY_VARIANCE": "0.850", "PCA_JOINT_VARIANCE": "0.80"})


def _conf_flags(targets):
    """{arm: (PCA_MULTI_ARGS, PCA_SINGLE_ARGS)} as config.conf builds them
    under these targets."""
    out = {}
    for arm in PCA_ARMS:
        multi, single = _conf('echo "$PCA_MULTI_ARGS"; echo "$PCA_SINGLE_ARGS"',
                              PCA_ARM=arm, **targets)
        out[arm] = (multi, single)
    return out


def _pca_keys(sig):
    return {k: sig[k] for k in PCA_SIGNATURE_KEYS}


def test_pca_flags_resolve_as_train_py_does():
    # The orchestrators compare each combo they reuse with the PCA flags they
    # would train it with (pca_matches), resolving the flags here, without
    # train.py and its torch: for every arm's multi-modal and single-embedding
    # flags under any targets (an exact count, other fractions, 0.80 for 0.8)
    # the PCA keys are train.py --print_arch's own; pca_multi_flags is
    # config.conf's pca_multi_args, word for word.
    seen = {}
    for targets in _TARGETS:
        for arm, (multi, single) in _conf_flags(targets).items():
            assert pca_multi_flags(arm, targets) == multi, (arm, targets)
            for flags in (multi, single):
                if flags not in seen:
                    seen[flags] = _print_arch(*flags.split())
                got = arch_signature(pca_flags_config(flags))
                assert _pca_keys(got) == _pca_keys(seen[flags]), (
                    arm, targets, flags, _pca_keys(got))
                assert pca_matches(seen[flags], flags.split()), flags


def test_pca_flags_refuse_what_train_py_refuses():
    # A flag set train.py's parser stops on raises (the orchestrators then
    # only warn: every task still checks), never resolves to some PCA.
    for flags in (["--pca_variance", "1.5"], ["--pca_variance", "0"],
                  ["--pca_variance", "0.8", "--pca_dims", "50", "64"],
                  ["--post_pca_dim", "0"], ["--post_pca_variance", "1"],
                  ["--post_pca_dim", "2.5"], ["--pca_variance", "x"],
                  ["--pca_dims"]):
        try:
            pca_flags_config(flags)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {flags}")
        r = subprocess.run(
            [sys.executable, os.path.join(_SIAMESE, "train.py"),
             "--embeddings_paths", "x.pt", "y.pt", "--use_cell_lines",
             "--print_arch", *flags], capture_output=True, text=True,
            cwd=_SIAMESE, timeout=300)
        assert r.returncode == 2 and "error:" in r.stderr, (flags, r.stderr)
    # and anything that is no PCA flag (train.py has others: not for here)
    for flags in (["--bogus"], ["--pca_var", "0.8"]):
        try:
            pca_flags_config(flags)
        except ValueError:
            continue
        raise AssertionError(f"accepted {flags}")


def test_pca_target_needs_train_the_recorded_pca():
    # pca_target_needs names, for a combo's config and its arm, the
    # config.conf targets that train its PCA, the exports an orchestrator's
    # refusal prints: given back to config.conf they make flags that train
    # exactly that PCA. Another arm's PCA, or one config.conf never passes
    # (an explicit --pca_dims list, an expanded --pca_variance without its
    # key), has none.
    reads = {"none": set(), "joint": {"PCA_JOINT_VARIANCE", "PCA_JOINT_DIM"},
             "per_modality": {"PCA_MODALITY_VARIANCE"},
             "both": {"PCA_MODALITY_VARIANCE", "PCA_JOINT_VARIANCE",
                      "PCA_JOINT_DIM"}}
    for targets in _TARGETS:
        for arm, (multi, _) in _conf_flags(targets).items():
            cfg = pca_flags_config(multi)
            needs = pca_target_needs(cfg, arm)
            names = set(reads[arm])
            if arm in ("joint", "both") and targets.get("PCA_JOINT_DIM"):
                names.discard("PCA_JOINT_VARIANCE")   # the count wins
            assert needs is not None and set(needs) == names, (arm, needs)
            back, = _conf('echo "$PCA_MULTI_ARGS"', PCA_ARM=arm, **needs)
            assert pca_matches(cfg, back), (arm, targets, needs, back)
            for other in PCA_ARMS:
                if other != arm:
                    assert pca_target_needs(cfg, other) is None, (arm, other)
    assert pca_target_needs({}, "none") == {}
    assert pca_target_needs({"post_pca": 0.80}, "joint") == {
        "PCA_JOINT_VARIANCE": "0.8", "PCA_JOINT_DIM": ""}
    for cfg in ({"pca_dims": [50, 64]}, {"pca_dims": [0.8, 0.8]}):
        assert pca_target_needs(cfg, "per_modality") is None, cfg


def test_check_pca_command_line():
    # The orchestrators' --check_pca: per combo its PCA and needs, per arm and
    # per group of its variants a verdict (ok: the flags' PCA; fix: one other
    # set of targets, the exports; other: none), and one over every arm; then
    # "end", which the callers require.
    script = os.path.join(_SIAMESE, "arch_signature.py")
    shell = "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM="
    with tempfile.TemporaryDirectory() as tmp:
        def cfg(name, **keys):
            os.makedirs(os.path.join(tmp, name))
            path = os.path.join(tmp, name, "config.json")
            with open(path, "w") as f:
                json.dump(keys, f)
            return path

        def run(*args, rc=0):
            r = subprocess.run([sys.executable, script, "--check_pca",
                                "--targets", shell, *args],
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == rc, (args, r.stdout, r.stderr)
            if rc:
                assert "end" not in r.stdout.split(), r.stdout
                return r.stderr
            lines = r.stdout.splitlines()
            assert lines[-1] == "end", r.stdout
            return [ln.split("|") for ln in lines[:-1]]

        none = cfg("none")
        j12 = cfg("j12", post_pca=12, post_pca_dim=12)
        j08 = cfg("j08", post_pca=0.8)
        b12 = cfg("b12", pca_variance=0.8, pca_dims=[0.8, 0.8], post_pca=12,
                  post_pca_dim=12)
        b08 = cfg("b08", pca_variance=0.8, pca_dims=[0.8, 0.8], post_pca=0.8)
        joint = ("--arm", "joint", "--post_pca_variance 0.80")
        both = ("--arm", "both", "--pca_variance 0.8 --post_pca_variance 0.8")
        # every arm at this shell's PCA (0.80 is 0.8)
        lines = run("--arm", "none", "--no_post_pca", f"pd:per_head={none}",
                    *joint, f"pd:per_head={j08}")
        assert ["combo", "joint", "pd:per_head", os.path.dirname(j08), "ok",
                "PCA on the concatenation of all modalities (variance 0.8)",
                "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM="] in lines, lines
        assert [ln[2] for ln in lines if ln[0] == "arm"] == ["ok", "ok"] \
            and lines[-1] == ["all", "ok", ""], lines
        # joint and both at 12 components: one export trains both
        lines = run(*joint, f"pd:per_head={j12}", f"pd:pooled={j12}",
                    *both, f"pd:per_head={b12}")
        assert ["group", "joint", "fix", "PCA_JOINT_DIM=12",
                "pd:per_head pd:pooled"] in lines \
            and ["arm", "joint", "fix", "PCA_JOINT_DIM=12"] in lines \
            and ["arm", "both", "fix", "PCA_JOINT_DIM=12"] in lines \
            and lines[-1] == ["all", "fix", "PCA_JOINT_DIM=12"], lines
        # joint at 12, both at this shell's: each arm right alone, no one
        # set for both together
        lines = run(*joint, f"pd:per_head={j12}", *both, f"pd:per_head={b08}")
        assert ["arm", "joint", "fix", "PCA_JOINT_DIM=12"] in lines \
            and ["arm", "both", "ok", ""] in lines \
            and lines[-1] == ["all", "other", "-"], lines
        # two variants of one arm at different targets: one group each
        lines = run(*joint, f"pd:per_head={j12}", f"pd:pooled={j08}")
        assert ["group", "joint", "fix", "PCA_JOINT_DIM=12",
                "pd:per_head"] in lines \
            and ["group", "joint", "ok", "", "pd:pooled"] in lines \
            and ["arm", "joint", "other", "-"] in lines, lines
        # a combo of another arm's PCA: no target trains it in this arm
        lines = run(*joint, f"pd:per_head={b08}")
        assert lines[0][4:] == ["bad", "PCA within each modality (variance "
                                "0.8), then on their concatenation "
                                "(variance 0.8)", "-"] \
            and ["arm", "joint", "other", "-"] in lines, lines
        # an unreadable config is named, and compared with nothing
        nope = os.path.join(tmp, "nope", "config.json")
        assert run(*joint, f"pd:per_head={nope}") == [
            ["skip", "joint", "pd:per_head", nope]]
        assert run() == []
        # a bad call: exit 2, no "end"
        assert "unknown PCA arm" in run("--arm", "bogus", "--no_post_pca",
                                        rc=2)
        assert "must be in (0, 1)" in run("--arm", "joint",
                                          "--post_pca_variance 1.5",
                                          f"x={j08}", rc=2)
        assert "want KEY=CONFIG.JSON" in run(*joint, "nokey", rc=2)
        assert "want --arm ARM FLAGS" in run("--arm", "joint", rc=2)
        r = subprocess.run([sys.executable, script], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "usage" in r.stderr, r.stderr


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
