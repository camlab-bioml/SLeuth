#!/usr/bin/env python3
"""Architecture + selection signature of a siamese train.py run.

Two runs can only be compared as "the same model minus X" when everything
outside X matches: the encoder, the scorer options added on 2026-09-21, the
preprocessing that shapes the encoder input, the checkpoint-selection metric
and the optimisation schedule. arch_signature() reduces a run's config
(train.py's vars(args), as saved in config.json, or as printed by
`train.py ... --print_arch`) to exactly those fields. Keys that older configs
lack are filled with the value those runs actually used where a current value
reproduces it, so a pre-2026-09-21 checkpoint (LayerNorm, exp(log_tau), no
per-head bias, stratified AUPRG selection) compares correctly against a new
one. Where no current value does (configs predating commit 0767425, which
fit ROBPCA preprocessing on every gene with a plain-MLP encoder), the filler
is None, so such a config never matches a present value.

Used by slurm/run_ablation.sh (refuses to train an ablation whose signature
differs from its reference combo) and slurm/run_ablation_summary.sh. The PCA
arms (slurm/config.conf: the whole pipeline once per arm) use pca_settings() /
pca_arm(): the combo selection and paper_tables check that a run trained the
PCA of the arm whose results it sits in, and paper_tables tab:pca compares the
arms' combos, which differ in PCA_SIGNATURE_KEYS only, with
signature_diff(..., ignore=PCA_SIGNATURE_KEYS).

The orchestrators that reuse an earlier submission's combos
(slurm/submit_pipeline.sh --ablation-only, the root's --skip-siamese) compare,
before anything changes, the PCA each combo trained with with the PCA the
submitting shell gives its arm, through the command line at the end of this
file (check_pca: pca_flags_config resolves train.py's PCA flags as its main()
does, pca_target_needs names the config.conf targets that reproduce a combo's).

    python arch_signature.py --check_pca [--targets "NAME=VALUE ..."] \\
        --arm ARM "FLAGS" KEY=CONFIG.JSON ... [--arm ARM "FLAGS" ...]
"""
import argparse
import json
import os
import sys

# key -> value runs predating the key used (None: no default; must be present)
_SIGNATURE_DEFAULTS = {
    # options added 2026-09-21 (absent in older configs = published model)
    "cell_lines": "per_head",
    "kernel": "pd",
    "layernorm": True,
    "log_temperature": True,
    "per_head_last_layer_bias": False,
    "per_head_input_bias": False,
    "per_head_temperature": False,
    "inner_norm": "none",
    "select_metric": "auprg",
    # options added 2026-10-05 (absent in older configs = off)
    "gene_term": False,
    "pair_features": False,
    # train.py --no_input_bias (2026-11-12, the bilinear probe; absent = the
    # encoder's input offset, which every earlier run had)
    "input_bias": True,
    # encoder / scorer
    "use_cell_lines": False,
    "model_type": "siamese",  # single-output mode only
    "cell_line_dim": None,
    "siamese_encoder_type": None,  # absent = pre-0767425 plain MLP
    "encoder_dims": None,
    "last_layer_bias": True,
    "dropout": None,
    "pd_epsilon": None,
    "l1_lambdas": None,
    # preprocessing that shapes the encoder input (absent = pre-0767425:
    # ROBPCA pipeline fit on every gene, which no current value reproduces)
    "pca_method": None,
    "preprocessing_fit_scope": None,
    "pca_variance": None,
    "pca_dims": None,
    "post_pca": None,
    "post_pca_dim": None,
    # optimisation and selection schedule
    "learning_rate": None,
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "eval_interval": None,
    "patience": None,
    "warmrestart_T0": None,
    "warmrestart_Tmult": None,
    "seed": None,
    # the modality-selection split (train.py --selection_frac /
    # --selection_seed, 2026-11-12; absent = off): it removes training
    # pairs, so a run with another split trained on other pairs. Off
    # whatever the seed says (normalised below).
    "selection_frac": 0.0,
    "selection_seed": None,
    # train.py --all_modalities (2026-11-12, slurm/run_allmod_combo.sh): the
    # whole catalog with no modality selection. A label that changes no
    # computation, but a signature key so the all-modalities model is never
    # matched with a selected combo (absent = a selected or single run).
    "all_modalities": False,
}


def arch_signature(cfg: dict) -> dict:
    """The comparable fields of one run's config (see module docstring)."""
    sig = {k: cfg.get(k, default) for k, default in _SIGNATURE_DEFAULTS.items()}
    if not sig["use_cell_lines"]:
        # the cell-line options do nothing without --use_cell_lines, and with
        # one output the two --select_metric choices are the same number
        for k in ("cell_lines", "kernel", "layernorm", "log_temperature",
                  "per_head_last_layer_bias", "per_head_input_bias",
                  "per_head_temperature", "inner_norm", "select_metric",
                  "cell_line_dim", "gene_term", "pair_features",
                  "input_bias"):
            sig[k] = None
    else:
        sig["model_type"] = None  # the cell-line model ignores --model_type
    if sig["use_cell_lines"] and sig["cell_lines"] == "pooled":
        sig["cell_line_dim"] = None  # no cell-line embedding when pooled
    if not sig["encoder_dims"]:
        # train.py builds [hidden_dim, latent_dim] when --encoder_dims is unset
        sig["encoder_dims"] = [cfg.get("hidden_dim", 512),
                               cfg.get("latent_dim", 256)]
    if sig["pca_variance"] is not None:
        # train.py expands --pca_variance to pca_dims = [v] * n_modalities;
        # that list's length follows the modality set (an ablation drops one),
        # so only an explicit --pca_dims list is compared.
        sig["pca_dims"] = None
    # --selection_frac 0 (train.py's default) holds nothing out: the seed it
    # records then means nothing, and a config without the keys is the same
    # run. --fold_modalities is the modality set, like embeddings_paths, and
    # never a signature key (an ablation differs from its combo there).
    frac = float(sig["selection_frac"] or 0.0)
    sig["selection_frac"] = frac
    sig["selection_seed"] = (None if frac == 0.0 or sig["selection_seed"] is None
                             else int(sig["selection_seed"]))
    sig["all_modalities"] = bool(sig["all_modalities"])
    for k in ("encoder_dims", "l1_lambdas", "pca_dims"):
        if sig[k] is not None:
            sig[k] = [float(x) for x in sig[k]]
    return sig


def signature_diff(a: dict, b: dict, ignore=()) -> dict:
    """{key: (a_value, b_value)} for every signature field that differs,
    leaving out the keys in `ignore`."""
    sa, sb = arch_signature(a), arch_signature(b)
    return {k: (sa[k], sb[k]) for k in sa
            if k not in ignore and sa[k] != sb[k]}


# The signature keys that say which PCA a run applied: train.py resolves
# --pca_variance / --pca_dims (within each modality) and --no_post_pca /
# --post_pca_dim / --post_pca_variance (on the concatenation) into them.
PCA_SIGNATURE_KEYS = ("pca_variance", "pca_dims", "post_pca", "post_pca_dim")

# The PCA arms' 2x2, in their fixed order (slurm/config.conf PCA_ARMS): PCA
# within each modality (no / yes) x PCA on the concatenation of all modalities
# (no / yes).
PCA_ARMS = ("none", "joint", "per_modality", "both")


def pca_settings(cfg: dict) -> dict:
    """The two PCA stages of one run's config (train.py's vars(args)):
    {"modality": None | ("variance", v) | ("dims", [k, ...]),
     "joint":    None | ("variance", v) | ("count", k)}.

    The per-modality stage is read from pca_variance first (train.py expands
    it to pca_dims = [v] * n_modalities), else from an explicit pca_dims
    list. The joint stage is read from post_pca, the value train.py resolved
    (None under --no_post_pca, whatever --post_pca_* say): an exact count
    when post_pca_dim is set, else a variance fraction."""
    v = cfg.get("pca_variance")
    dims = [float(x) for x in cfg.get("pca_dims") or []]
    if v is not None:
        modality = ("variance", float(v))
    elif dims and all(0 < x < 1 for x in dims) and len(set(dims)) == 1:
        # an expanded --pca_variance without its pca_variance key (--pca_dims
        # itself only takes integers)
        modality = ("variance", dims[0])
    elif dims:
        modality = ("dims", [int(x) for x in dims])
    else:
        modality = None
    post = cfg.get("post_pca")
    if post is None:
        joint = None
    elif cfg.get("post_pca_dim") is not None:
        joint = ("count", int(cfg["post_pca_dim"]))
    else:
        joint = ("variance", float(post))
    return {"modality": modality, "joint": joint}


def pca_arm(cfg: dict) -> str:
    """Which of PCA_ARMS one run's config is."""
    s = pca_settings(cfg)
    return PCA_ARMS[2 * (s["modality"] is not None) + (s["joint"] is not None)]


# ---------------------------------------------------------------------------
# A submission's PCA against the PCA of the combos it reuses (2026-10-21).
# submit_pipeline.sh --ablation-only retrains an arm's ablations against the
# combos an earlier submission trained, with the PCA flags THIS shell gives the
# arm (config.conf pca_multi_args of PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE
# / PCA_JOINT_DIM): every task then stops at run_ablation.sh's signature check
# when they are not its reference combo's, after the arm's previous ablation
# was removed at submission. The root's --skip-siamese labels and tabulates
# the reused combos under those targets, and paper_tables lists a combo trained
# at others as missing (PAPER INCOMPLETE). Both orchestrators compare first,
# before anything changes, through check_pca (the command line below), on the
# PCA keys every task compares; train.py's own flags are resolved here as its
# main() resolves them (test_arch_signature checks it against --print_arch).

# config.conf's PCA targets, in the order its pca_multi_args reads them.
PCA_TARGETS = ("PCA_MODALITY_VARIANCE", "PCA_JOINT_VARIANCE", "PCA_JOINT_DIM")


class _FlagParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _pca_flag_parser():
    """train.py's PCA arguments as its parser defines them (types, nargs,
    defaults). No abbreviations: train.py's longer flag list would resolve
    one otherwise, and config.conf passes none."""
    p = _FlagParser(prog="train.py", add_help=False, allow_abbrev=False)
    p.add_argument("--pca_dims", type=int, nargs="+", default=None)
    p.add_argument("--pca_variance", type=float, default=None)
    p.add_argument("--post_pca_variance", type=float, default=None)
    p.add_argument("--post_pca_dim", type=int, default=None)
    p.add_argument("--no_post_pca", dest="no_post_pca", action="store_true",
                   default=False)
    return p


def pca_flags_config(flags, n_modalities=1) -> dict:
    """The PCA keys of vars(args) that train.py main() resolves from these
    PCA flags (a list of words, or one string such as config.conf's
    pca_multi_args / pca_single_args): pca_dims becomes [pca_variance] *
    n_modalities under --pca_variance; post_pca is None under --no_post_pca,
    else the --post_pca_dim count, else the --post_pca_variance fraction,
    else None; the flags' own values are kept. Raises ValueError where
    train.py's parser stops (an unknown flag, a bad value, --pca_dims next to
    --pca_variance, a fraction outside (0, 1), a count below 1)."""
    words = flags.split() if isinstance(flags, str) else list(flags)
    a = _pca_flag_parser().parse_args(words)
    if a.pca_variance is not None and a.pca_dims is not None:
        raise ValueError("Use --pca_dims OR --pca_variance, not both")
    if a.pca_variance is not None:
        if not 0 < a.pca_variance < 1:
            raise ValueError("--pca_variance must be in (0, 1)")
        a.pca_dims = [a.pca_variance] * n_modalities
    if a.no_post_pca:
        a.post_pca = None
    elif a.post_pca_dim is not None:
        if a.post_pca_dim <= 0:
            raise ValueError("--post_pca_dim must be a positive integer")
        a.post_pca = a.post_pca_dim
    elif a.post_pca_variance is not None:
        if not 0 < a.post_pca_variance < 1:
            raise ValueError("--post_pca_variance must be in (0, 1)")
        a.post_pca = a.post_pca_variance
    else:
        a.post_pca = None
    return vars(a)


def pca_matches(cfg: dict, flags) -> bool:
    """Does a run's config (its config.json) carry the PCA these train.py
    flags train? On PCA_SIGNATURE_KEYS, as run_ablation.sh compares a task
    with its reference combo: --pca_variance 0.80 and 0.8 are one PCA."""
    want = arch_signature(pca_flags_config(flags))
    got = arch_signature(cfg)
    return all(got[k] == want[k] for k in PCA_SIGNATURE_KEYS)


def pca_multi_flags(arm: str, targets: dict) -> str:
    """config.conf's pca_multi_args ARM under these targets ({name: value}
    of PCA_TARGETS; a missing one takes config.conf's default). A mirror,
    used to check pca_target_needs' answers; test_arch_signature holds it to
    config.conf's own."""
    t = dict({"PCA_MODALITY_VARIANCE": "0.8", "PCA_JOINT_VARIANCE": "0.8",
              "PCA_JOINT_DIM": ""}, **targets)
    joint = ("--post_pca_dim " + t["PCA_JOINT_DIM"] if t["PCA_JOINT_DIM"]
             else "--post_pca_variance " + t["PCA_JOINT_VARIANCE"])
    modality = "--pca_variance " + t["PCA_MODALITY_VARIANCE"]
    return {"joint": joint, "per_modality": modality + " --no_post_pca",
            "both": modality + " " + joint}.get(arm, "--no_post_pca")


def _fraction(v) -> str:
    """A fraction written as config.conf takes one (0.<digits>, not all of
    them 0), or "" when it is none."""
    s = repr(float(v))
    if "e" in s:
        s = ("%.15f" % float(v)).rstrip("0")
    return s if (s.startswith("0.") and s[2:].isdigit()
                 and s[2:].strip("0")) else ""


def pca_target_needs(cfg: dict, arm: str):
    """The config.conf targets under which ARM's multi-modal flags
    (pca_multi_args) train the PCA of this config: {name: value} for the
    targets that arm reads, PCA_MODALITY_VARIANCE (per_modality, both) and
    PCA_JOINT_DIM (joint, both: the count, or "" for a fraction, then with
    PCA_JOINT_VARIANCE); {} for arm none. None when no targets do: the
    config is another arm's PCA, an explicit --pca_dims list, or anything
    else config.conf never passes (each answer is checked by the flags it
    gives, through pca_matches)."""
    if pca_arm(cfg) != arm:
        return None
    s = pca_settings(cfg)
    needs = {}
    if s["modality"] is not None:
        kind, v = s["modality"]
        needs["PCA_MODALITY_VARIANCE"] = _fraction(v) if kind == "variance" \
            else ""
        if not needs["PCA_MODALITY_VARIANCE"]:
            return None
    if s["joint"] is not None:
        kind, v = s["joint"]
        if kind == "count":
            needs["PCA_JOINT_DIM"] = str(v)
        else:
            needs["PCA_JOINT_VARIANCE"] = _fraction(v)
            needs["PCA_JOINT_DIM"] = ""
            if not needs["PCA_JOINT_VARIANCE"]:
                return None
    try:
        if not pca_matches(cfg, pca_multi_flags(arm, needs)):
            return None
    except ValueError:
        return None
    return needs


def pca_label(cfg: dict) -> str:
    """One line saying which PCA a run's config trained, in the words of
    config.conf's pca_arm_label."""
    def stage(st):
        kind, v = st
        if kind == "count":
            return f"{v} components"
        if kind == "dims":
            return "dims " + " ".join(str(x) for x in v)
        return "variance " + (_fraction(v) or str(v))
    s = pca_settings(cfg)
    m, j = s["modality"], s["joint"]
    if m is None and j is None:
        return "no PCA"
    if m is None:
        return f"PCA on the concatenation of all modalities ({stage(j)})"
    if j is None:
        return f"PCA within each modality ({stage(m)})"
    return (f"PCA within each modality ({stage(m)}), then on their "
            f"concatenation ({stage(j)})")


def _needs_text(needs) -> str:
    """{name: value} as NAME=VALUE words in PCA_TARGETS order; "-" for None
    (no targets do)."""
    if needs is None:
        return "-"
    return " ".join(f"{k}={needs[k]}" for k in PCA_TARGETS if k in needs)


def _merge(needs_list):
    """One set of targets meeting every one of needs_list, or None."""
    out = {}
    for n in needs_list:
        if n is None:
            return None
        for k, v in n.items():
            if out.setdefault(k, v) != v:
                return None
    return out


def check_pca(entries, targets=None):
    """Compare each arm's combos with the PCA its flags train.

    entries: [(arm, flags, [(key, config.json path), ...]), ...]: the flags
    the submission trains the arm with (pca_multi_args under its targets),
    each combo labelled by a key (its variant). targets: the submission's own
    {name: value} (PCA_TARGETS), so that the exports name only what has to
    change. Yields the fields of each output line (the command line prints
    them joined by "|"):

      combo|<arm>|<key>|<dir>|ok or bad|<its PCA, pca_label>|<its needs>
      skip|<arm>|<key>|<path>               a config that cannot be read
      group|<arm>|<status>|<exports>|<keys> per set of the arm's keys
      arm|<arm>|<status>|<exports>          per arm
      all|<status>|<exports>                every arm in one submission

    status ok: every combo there trains the flags' PCA. fix: they agree on
    one other set of targets, set by <exports> (NAME=VALUE words; an empty
    VALUE empties it). other: no set of targets trains all of them (they
    disagree, or one is not its arm's PCA at all), <exports> "-". A group
    is the keys (variants) of one arm with one status and one set of
    exports: one submission per group (SIAMESE_VARIANTS_LIST=<keys>)
    reproduces each. `all` holds every combo of every arm to one set of
    targets: the arms already right keep theirs."""
    targets = dict(targets or {})

    def verdict(needs_of, oks):
        if all(oks):
            return "ok", ""
        merged = _merge(needs_of)
        if merged is None:
            return "other", "-"
        exports = {k: v for k, v in merged.items() if targets.get(k) != v}
        # nothing to change, yet another PCA: these flags are not what the
        # targets give (config.conf moved on); no export is known to help
        return ("fix", _needs_text(exports)) if exports else ("other", "-")

    every_needs, every_ok = [], []
    for arm, flags, items in entries:
        if arm not in PCA_ARMS:
            raise ValueError(f"unknown PCA arm {arm!r}")
        per_key = {}            # key -> ([needs], [ok]) of its combos
        for key, path in items:
            try:
                with open(path) as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    raise ValueError(path)
            except (OSError, ValueError):
                yield ("skip", arm, key, path)
                continue
            ok = pca_matches(cfg, flags)
            needs = pca_target_needs(cfg, arm)
            n, o = per_key.setdefault(key, ([], []))
            n.append(needs)
            o.append(ok)
            yield ("combo", arm, key, os.path.dirname(path) or ".",
                   "ok" if ok else "bad", pca_label(cfg), _needs_text(needs))
        if not per_key:
            continue
        groups = {}             # (status, exports) -> [key], in key order
        for key, (n, o) in per_key.items():
            groups.setdefault(verdict(n, o), []).append(key)
        for (status, exports), keys in groups.items():
            yield ("group", arm, status, exports, " ".join(keys))
        arm_needs = [x for n, _ in per_key.values() for x in n]
        arm_ok = [x for _, o in per_key.values() for x in o]
        yield ("arm", arm) + verdict(arm_needs, arm_ok)
        every_needs += arm_needs
        every_ok += arm_ok
    if every_ok:
        yield ("all",) + verdict(every_needs, every_ok)


def _main(argv):
    """--check_pca [--targets "NAME=VALUE ..."] --arm ARM FLAGS KEY=PATH ...
    [--arm ...]: check_pca's lines, then "end", which a caller requires (an
    older copy of this file prints nothing and must not read as a match)."""
    if argv[:1] != ["--check_pca"]:
        print("usage: arch_signature.py --check_pca [--targets \"NAME=VALUE"
              " ...\"] --arm ARM FLAGS KEY=CONFIG.JSON ... [--arm ...]",
              file=sys.stderr)
        return 2
    args, targets, entries = list(argv[1:]), {}, []
    if args[:1] == ["--targets"] and len(args) > 1:
        for w in args[1].split():
            k, _, v = w.partition("=")
            targets[k] = v
        args = args[2:]
    while args:
        if args[0] != "--arm" or len(args) < 3:
            print(f"arch_signature.py --check_pca: want --arm ARM FLAGS, got "
                  f"{args[:3]}", file=sys.stderr)
            return 2
        arm, flags, args, items = args[1], args[2], args[3:], []
        while args and args[0] != "--arm":
            key, sep, path = args.pop(0).partition("=")
            if not sep:
                print(f"arch_signature.py --check_pca: want KEY=CONFIG.JSON, "
                      f"got {key!r}", file=sys.stderr)
                return 2
            items.append((key, path))
        entries.append((arm, flags, items))
    try:
        lines = list(check_pca(entries, targets))
    except ValueError as e:
        print(f"arch_signature.py --check_pca: {e}", file=sys.stderr)
        return 2
    for fields in lines:
        print("|".join(fields))
    print("end")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
