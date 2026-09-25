#!/usr/bin/env python3
"""Produce-time run artifacts, shared by all four benchmarked models.

Three things every trainer must record or clear so that what evaluate_model.py
later grades is provably the output of the run that just finished:

  * write_model_params  — trainable-parameter counts (below) AND the fold stamp
  * fold_stamp          — content hash of the fold set the model trained on
  * purge_stale_predictions — delete the PREVIOUS run's matrices up front

They live in one module because all four trainers already import exactly one
file from sl_comparison/ (siamese loads it by path, deliberately, to avoid
putting sl_comparison/ on sys.path), and a second module would have to
replicate that plumbing.

One definition of "how big is this model" for every row of the comparison
table, so the counts are actually comparable:

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

Each trainer calls write_model_params(output_dir, model) at model-construction
time. That directory is the same --pred_dir evaluate_model.py reads, so the
count travels with the predictions and lands in the summary JSON without any
extra plumbing; compare.py then renders the model column as SLMGAE(1,234,567).

Counted at construction rather than after training on purpose: L1 proximal
soft-thresholding (siamese) drives many weights to exactly 0 without freeing
them, so a post-training nonzero count answers a different question. siamese
still reports that separately as nonzero_params / weight_sparsity.
"""

import hashlib
import json
from pathlib import Path

PARAMS_FILENAME = "model_params.json"

# Files whose contents DEFINE the answer key. Hashing them at produce time is
# what lets the evaluator prove a score matrix was trained against the same
# fold set it is about to be graded on.
STAMP_FILES = ("genes.txt", "all_pos.npy", "meta.json")


def count_parameters(model):
    """(trainable, total) parameter counts for a torch Module."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def fold_stamp(folds_dir, cv_type=None):
    """Content fingerprint of the fold set a model is about to train on.

    WHY THIS EXISTS. evaluate_model.py joins score matrices to an answer key by
    DIRECTORY PATH: it globs fold_<k>_<cv>_predictions.npy out of --pred_dir and
    grades them against whatever sl_comparison/folds contains at that moment.
    Nothing in the matrix records which folds produced it. So when the fold set
    is regenerated — as it was on 2026-08-10, when the conflict policy moved
    5,542 pairs and the positive count went 20,384 -> 22,166 — a leftover matrix
    from the previous run is graded against the NEW test edges without a murmur.
    That is not a small error: 2,970 of the 4,434 new cv1/fold_0 test positives
    (67%) were TRAINING positives under the old fold set, because changing the
    pair count reshuffles rng.permutation for every fold. The universe width is
    unchanged at 8,844, so score_fold's shape check passes and the run exits 0.

    Written by write_model_params into model_params.json beside the matrices,
    and checked by verify_fold_stamp before any scoring happens.
    """
    root = Path(folds_dir)
    stamp = {"folds_dir": str(root)}
    if cv_type:
        stamp["cv_type"] = cv_type
    for name in STAMP_FILES:
        p = root / name
        stamp[name] = (hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                       if p.exists() else None)
    return stamp


def compare_fold_stamp(recorded, folds_dir, cv_type=None):
    """Return a list of human-readable mismatches (empty == the folds agree).

    Only the content hashes are compared. folds_dir is deliberately NOT
    compared: the same fold set is legitimately reached by different relative
    paths (siamese runs from siamese_sl/ as ../sl_comparison/folds, the shared
    models from the repo root), and a path difference is not a data difference.
    """
    current = fold_stamp(folds_dir, cv_type)
    diffs = []
    for name in STAMP_FILES:
        was, now = recorded.get(name), current.get(name)
        if was != now:
            diffs.append(
                f"{name}: trained against {was}, grading against {now}")
    if cv_type and recorded.get("cv_type") not in (None, cv_type):
        diffs.append(f"cv_type: matrices are {recorded['cv_type']}, "
                     f"grading as {cv_type}")
    return diffs


def verify_fold_stamp(pred_dir, folds_dir, cv_type, allow_unstamped=False):
    """Refuse to grade matrices that were not produced on THESE folds.

    Raises SystemExit on a mismatch, and on a missing stamp unless
    allow_unstamped. Failing loudly is the whole point: the alternative is a
    published mean over four fresh matrices and one foreign one.
    """
    recorded = read_model_params(pred_dir).get("fold_stamp")
    if not recorded:
        msg = (
            f"{pred_dir} has no fold_stamp in {PARAMS_FILENAME}, so there is "
            f"no way to tell which fold set produced these matrices. They "
            f"may predate the fold set in {folds_dir}. Retrain, or pass "
            f"--allow_unstamped if you have verified them by other means.")
        if allow_unstamped:
            print(f"  WARNING: {msg}")
            return
        raise SystemExit(f"REFUSING TO GRADE: {msg}")
    diffs = compare_fold_stamp(recorded, folds_dir, cv_type)
    if diffs:
        raise SystemExit(
            f"REFUSING TO GRADE: the matrices in {pred_dir} were trained "
            f"against a DIFFERENT fold set than {folds_dir}:\n  " +
            "\n  ".join(diffs) +
            "\nGrading them anyway would score a model on pairs it trained on. "
            f"Delete {pred_dir} and retrain against the current folds.")


def write_model_params(output_dir,
                       model,
                       folds_dir=None,
                       cv_type=None,
                       **extra):
    """Write <output_dir>/model_params.json. Never fatal.

    Parameter counting must not be able to kill a training run that is
    otherwise fine, so any failure degrades to a warning and the comparison
    table simply shows no count for that model.

    Pass folds_dir (and cv_type) to stamp the fold set into the same file —
    see fold_stamp. A trainer that omits it produces matrices the evaluator
    will refuse to grade, which is the intended failure direction.
    """
    try:
        trainable, total = count_parameters(model)
        payload = {
            "trainable_params": trainable,
            "total_params": total,
            "model_class": type(model).__name__,
        }
        if folds_dir:
            payload["fold_stamp"] = fold_stamp(folds_dir, cv_type)
        payload.update(extra)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / PARAMS_FILENAME, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Trainable parameters: {trainable:,} "
              f"(total {total:,}) [{type(model).__name__}]")
        if folds_dir:
            st = payload["fold_stamp"]
            print(f"  Fold stamp: genes {st['genes.txt']} "
                  f"all_pos {st['all_pos.npy']} meta {st['meta.json']}")
        else:
            print(
                "  Warning: no folds_dir given — these predictions carry no "
                "fold stamp and evaluate_model.py will refuse to grade them.")
        return trainable, total
    except Exception as e:  # noqa: BLE001
        print(f"  Warning: could not record parameter count ({e})")
        return None, None


def purge_stale_predictions(output_dir, cv_type, keep=()):
    """Delete EVERY fold's leftover score matrix + checkpoint, once, up front.

    Each trainer also clears its own fold's artifacts at fold start, but that
    only protects folds the run actually reaches. A run that dies at fold 3
    leaves folds 3 and 4 holding the PREVIOUS run's matrices, and
    evaluate_model.py iterates the fold directories and grades every matrix it
    finds — so the published mean is three fresh folds plus two foreign ones,
    at exit 0. Clearing everything before fold 0 makes a crashed fold MISSING
    (skipped, and said so) rather than stale.

    `keep` names fold indices to leave alone, for a deliberate resume.
    """
    out = Path(output_dir)
    if not out.exists():
        return 0
    keep = {int(k) for k in keep}
    removed = []
    for p in sorted(out.glob(f"fold_*_{cv_type}_predictions.npy")):
        try:
            k = int(p.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if k in keep:
            continue
        p.unlink()
        removed.append(p.name)
        ck = out / "checkpoints" / f"fold_{k}_best.pt"
        if ck.exists():
            ck.unlink()
            removed.append(ck.name)
    if removed:
        print(
            f"  Cleared {len(removed)} stale artifact(s) from a previous run "
            f"in {out}: {', '.join(removed[:6])}"
            f"{' ...' if len(removed) > 6 else ''}")
    return len(removed)


def read_model_params(pred_dir):
    """Read back the counts written by write_model_params, or {} if absent."""
    p = Path(pred_dir) / PARAMS_FILENAME
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
