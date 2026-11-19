#!/usr/bin/env python3
"""train.py arguments of the bilinear probe (reviewer 22DY), from a combo.

The probe asks whether SLeuth's encoder / head add anything over "a simple
bilinear probe on the same features":

    logit(i, j) = tau * (W x_i)ᵀ (W x_j) + b

W one Linear map (no bias) from the preprocessed input features x to the
combo's latent width (32 under the current ENCODER_DIMS), no hidden layer,
no input offset, no projection bias, no per-gene or pair term, eps = 0, one
pooled head on the OR label. tau and b are scalars; tau is raw (any sign)
unless the combo trained exp(log tau) (--log_temperature), as every SLeuth
model of the run does. Parameters: latent * k + 2 (k = the fold's input
width; 32 * 4,668 + 2 = 149,378 on the 2026-11-10 arm-none modality set).

"The same features" is PCA arm none's pd:pooled combo of the same CV (the
pooled SLeuth model, the one with the probe's outcome and head): its
config.json supplies everything that decides WHICH pairs and WHICH features
the probe trains on, and every training setting, so the probe differs from
it in the model only:

    reused       embeddings_paths (same order), fold_modalities (per-fold
                 modality selection; absent in the global mode), sl_path,
                 neg_pairs_path, folds_dir, selection_frac / selection_seed,
                 preprocessing_fit_scope, pca_method (arm none: no PCA,
                 --no_post_pca), epochs, batch_size, learning_rate,
                 weight_decay, eval_interval, patience, warmrestart_T0 /
                 _Tmult, num_folds, seed, select_metric, log_temperature
    probe's own  --use_cell_lines --cell_lines pooled --kernel pd
                 --encoder_dims <latent> --pd_epsilon 0 --no_gene_term
                 --no_pair_features --no_per_head_last_layer_bias
                 --no-last-layer-bias --no_input_bias --inner_norm none
                 --l1_lambdas <the combo's FIRST lambda>: the probe's only
                 weight matrix is, like the combo's entry layer, the map
                 from the input features (0.01 under L1_LAMBDAS)

The combo must be arm none's pd:pooled model of the requested CV, trained
with the cell-line model (refused otherwise). slurm/run_probe.sh calls this
once per CV:

    python probe_args.py --cv cv1 --output_dir results/bilinear_probe_cv1 \\
        [--marker results/var_pd_pooled__latest_combo_cv1.txt]

and reads the argument list from stdout, NUL-separated (--sep newline for a
readable listing). Run from siamese_sl/ (the marker's combo path and the
combo's relative folds_dir are siamese_sl-relative). Exit 2 with the reason
on stderr when the marker, the combo or its config is missing or not the
headline pd:pooled combo of arm none.
"""
import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from arch_signature import pca_arm, pca_label  # noqa: E402

PROBE_ROW = "BilinearProbe"            # sl_comparison row name
PROBE_VARIANT = "pd:pooled"            # the combo whose features it reuses
# config.conf siamese_variant_prefix pd:pooled (arm none: results/)
MARKER_PREFIX = "var_pd_pooled__"


def marker_path(results_root: str, cv: str) -> Path:
    """<results_root>/var_pd_pooled__latest_combo_<cv>.txt"""
    return Path(results_root) / f"{MARKER_PREFIX}latest_combo_{cv}.txt"


def probe_output_dir(results_root: str, cv: str) -> Path:
    return Path(results_root) / f"bilinear_probe_{cv}"


class ProbeRefused(Exception):
    """The combo cannot stand for the probe's features (the message says why)."""


def read_combo(marker) -> tuple:
    """(combo_dir, config dict) named by a latest_combo marker."""
    marker = Path(marker)
    if not marker.is_file():
        raise ProbeRefused(
            f"no {marker}: arm none's pd:pooled combo of this CV has not "
            f"trained (run_best_per_category_combo.sh writes the marker only "
            f"once that combo finished)")
    text = marker.read_text().strip().splitlines()
    combo = Path(text[0].strip()) if text else None
    if combo is None or not combo.is_dir():
        raise ProbeRefused(f"{marker} names {combo}, which is not a directory")
    cfg_path = combo / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as e:
        raise ProbeRefused(f"cannot read {cfg_path}: {e}")
    if not isinstance(cfg, dict):
        raise ProbeRefused(f"{cfg_path} is not a JSON object")
    return combo, cfg


def check_combo(cfg: dict, cv: str) -> None:
    """Raise ProbeRefused unless cfg is arm none's pd:pooled combo of cv."""
    problems = []
    if not cfg.get("use_cell_lines"):
        problems.append("not a --use_cell_lines run")
    if cfg.get("cell_lines", "per_head") != "pooled":
        problems.append(f"cell_lines {cfg.get('cell_lines', 'per_head')!r}, "
                        f"not 'pooled'")
    if cfg.get("kernel", "pd") != "pd":
        problems.append(f"kernel {cfg.get('kernel')!r}, not 'pd'")
    if cfg.get("cv_type") != cv:
        problems.append(f"cv_type {cfg.get('cv_type')!r}, not {cv!r}")
    if pca_arm(cfg) != "none":
        problems.append(f"trained with {pca_label(cfg)} (PCA arm "
                        f"{pca_arm(cfg)}), not arm none's no-PCA features")
    for key in ("embeddings_paths", "sl_path", "neg_pairs_path", "folds_dir"):
        if not cfg.get(key):
            problems.append(f"no {key}")
    if problems:
        raise ProbeRefused("the combo is not arm none's pd:pooled combo of "
                           f"{cv}: " + "; ".join(problems))


def latent_dim(cfg: dict) -> int:
    dims = cfg.get("encoder_dims") or [cfg.get("hidden_dim", 512),
                                       cfg.get("latent_dim", 256)]
    return int(dims[-1])


def probe_train_args(cfg: dict, cv: str, output_dir) -> list:
    """train.py's argument list (without the interpreter and script) for the
    probe on cfg's pairs and features. cfg is the combo's config.json."""
    check_combo(cfg, cv)
    a = ["--embeddings_paths", *[str(p) for p in cfg["embeddings_paths"]],
         "--sl_path", str(cfg["sl_path"]),
         "--neg_pairs_path", str(cfg["neg_pairs_path"]),
         "--folds_dir", str(cfg["folds_dir"]),
         "--output_dir", str(output_dir),
         "--cv_type", cv]
    fm = cfg.get("fold_modalities")
    if fm is not None and fm != "":
        # As train.py received it: a JSON object string or a file path
        # (config.json keeps the raw argument); a dict only from a hand edit.
        a += ["--fold_modalities",
              fm if isinstance(fm, str) else json.dumps(fm, sort_keys=True)]
    frac = float(cfg.get("selection_frac") or 0.0)
    if frac > 0:
        a += ["--selection_frac", repr(frac),
              "--selection_seed", str(int(cfg.get("selection_seed", 42)))]
    # The data pipeline exactly as the combo's (arm none: no PCA at all).
    a += ["--preprocessing_fit_scope",
          str(cfg.get("preprocessing_fit_scope") or "train"),
          "--pca_method", str(cfg.get("pca_method") or "plain"),
          "--no_post_pca"]
    # Optimisation and selection schedule.
    for key, flag in (("epochs", "--epochs"), ("batch_size", "--batch_size"),
                      ("learning_rate", "--learning_rate"),
                      ("weight_decay", "--weight_decay"),
                      ("eval_interval", "--eval_interval"),
                      ("patience", "--patience"),
                      ("warmrestart_T0", "--warmrestart_T0"),
                      ("warmrestart_Tmult", "--warmrestart_Tmult"),
                      ("num_folds", "--num_folds"), ("seed", "--seed"),
                      ("dropout", "--dropout")):
        if cfg.get(key) is not None:
            a += [flag, str(cfg[key])]
    a += ["--select_metric", str(cfg.get("select_metric") or "auprg_any")]
    if cfg.get("log_temperature"):
        a.append("--log_temperature")
    if cfg.get("layernorm"):
        # Inert with no hidden block (the LayerNorm sits inside the blocks);
        # passed so the probe's arch_signature differs from the combo's in
        # the probe's own keys only.
        a.append("--layernorm")
    # The probe itself: one bias-free Linear, pd kernel at eps 0, one head.
    a += ["--use_cell_lines", "--cell_lines", "pooled", "--kernel", "pd",
          "--model_type", "siamese",
          "--siamese_encoder_type",
          str(cfg.get("siamese_encoder_type") or "residual"),
          "--encoder_dims", str(latent_dim(cfg)),
          "--pd_epsilon", "0",
          "--no_gene_term", "--no_pair_features",
          "--no_per_head_last_layer_bias", "--no-last-layer-bias",
          "--no_input_bias", "--inner_norm", "none"]
    lams = cfg.get("l1_lambdas") or []
    if lams:
        a += ["--l1_lambdas", str(lams[0])]
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cv", required=True, choices=["cv1", "cv2", "cv3"])
    ap.add_argument("--results_root", default="results",
                    help="arm none's siamese results root (siamese_sl-"
                    "relative; default results)")
    ap.add_argument("--marker", default=None,
                    help="the pd:pooled latest_combo marker (default "
                    "<results_root>/var_pd_pooled__latest_combo_<cv>.txt)")
    ap.add_argument("--output_dir", default=None,
                    help="the probe's train.py --output_dir (default "
                    "<results_root>/bilinear_probe_<cv>)")
    ap.add_argument("--sep", choices=["nul", "newline"], default="nul")
    ap.add_argument("--print_combo", action="store_true",
                    help="print the combo directory instead of the arguments")
    args = ap.parse_args(argv)
    marker = args.marker or marker_path(args.results_root, args.cv)
    out = args.output_dir or probe_output_dir(args.results_root, args.cv)
    try:
        combo, cfg = read_combo(marker)
        words = probe_train_args(cfg, args.cv, out)
    except ProbeRefused as e:
        print(f"probe_args.py: REFUSED ({args.cv}): {e}", file=sys.stderr)
        return 2
    if args.print_combo:
        print(combo)
        return 0
    sep = "\0" if args.sep == "nul" else "\n"
    sys.stdout.write("".join(w + sep for w in words))
    return 0


if __name__ == "__main__":
    sys.exit(main())
