#!/usr/bin/env python3
"""Per-fold, per-tier modality selection (2026-11-12; SELECTION_MODE=per_fold).

The global selection (SELECTION_MODE=global, the default) ranks each
category's single-embedding runs ONCE, on the fold-mean validation score of
SELECTION_CV, and every fold of every CV reuses those winners. That is not
leakage-free: the fold means pool validation splits whose training pools hold
other folds' test pairs.

Per-fold selection ranks every (cv, fold) on its OWN modality-selection split:
train.py --selection_frac F holds a uniform random F of that fold's TRAINING
pairs out of training (identical in every run on the same folds and
--selection_seed; fold_metrics[k]["sel_digest"]), and scores the val-selected
checkpoint on it once (fold_metrics[k]["sel_<select_metric>"]). The test split
of fold k never touches fold k's choice, and no other fold's numbers do either.

Each category's winner is then the argmax of sel_<select_metric> per (cv, fold)
(first maximum in catalog order). A cv's combo is ONE train.py run over the
UNION of its folds' winners (--embeddings_paths, fixed order: categories
sorted, then catalog order within a category) with --fold_modalities mapping
each fold to the positions of its own winners; every checkpoint records its
subset (preprocessing_transform["modality_idx"]), so every reader that loads
config.json's embeddings_paths applies the right one.

best_per_category.json in per_fold mode keeps the global layout's keys, so a
reader that only wants "the" type of a category still gets one:
    <cv>: {<category>: {
        "type", "file":    the category's most-picked embedding over the cv's
                           folds (ties: catalog order),
        "aupr":            mean over the folds of the winning selection score,
        "types", "files":  every embedding picked in some fold (catalog order),
        "n_folds_picked":  {type: number of folds that picked it},
        "per_fold":        {"<k>": {"type", "file", "score"}}}}
    _selection_meta: {"selection_mode": "per_fold", "selection_metric":
                      "sel_<select_metric>", "selection_frac",
                      "selection_seed", "sel_digests": {cv: {"<k>": sha256}},
                      ...}
A selection without _selection_meta.selection_mode is global (every
selection written before 2026-11-12).

Command line (the SLURM scripts):
    python fold_selection.py combo_spec --selection best_per_category.json \\
        --cv cv1 [--drop CATEGORY] [--data_dir ../data]
prints "<paths>|<name+name...>|<n per fold>|<n union>|<fold map JSON or empty>"
and exits 1 (message on stderr) on a selection it cannot use or, with
--data_dir, an embedding file that does not exist.
    python fold_selection.py allmod_spec --data_dir ../data \\
        --optional esmc scprint --catalog type:file:category ...
prints "<paths>|<name+name...>|<n>|<absent optional types, comma-separated>"
for the all-modalities combo (slurm/run_allmod_combo.sh: every catalog
embedding, catalog order, no selection; exit 1 on a missing required file).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MODES = ("global", "per_fold")


class SelectionError(RuntimeError):
    """A per-fold selection that cannot be made or used (message: why)."""


def selection_mode(selection: Optional[dict]) -> str:
    """'per_fold' or 'global' (a selection that records no mode)."""
    meta = (selection or {}).get("_selection_meta") or {}
    mode = meta.get("selection_mode") or "global"
    if mode not in MODES:
        raise SelectionError(f"unknown _selection_meta.selection_mode "
                             f"{mode!r} (want one of {MODES})")
    return mode


def sel_key(cfg: Optional[dict]) -> str:
    """The fold_metrics key per-fold selection ranks a run on: its own
    checkpoint metric (train.py --select_metric; auprg for runs predating
    the flag) on the selection split."""
    return f"sel_{(cfg or {}).get('select_metric') or 'auprg'}"


def _finite(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fold_entries(res: dict, num_folds: int) -> Dict[int, dict]:
    """{fold: fold_metrics entry} of one results.json, keyed by the entry's
    'fold' (train.py records it), else by position."""
    fms = [m for m in (res or {}).get("fold_metrics") or []
           if isinstance(m, dict)]
    if fms and all(isinstance(m.get("fold"), int) for m in fms):
        out = {m["fold"]: m for m in fms}
    else:
        out = dict(enumerate(fms))
    return {k: v for k, v in out.items() if 0 <= k < num_folds}


def rank_per_fold(catalog: Sequence[Tuple[str, str, str]],
                  results: Dict[str, dict], num_folds: int,
                  cv: str = "?") -> dict:
    """The per-fold winners of one cv.

    catalog: [(etype, efile, category)] (EMB_CATALOG order: ties go to the
    first). results: {etype: results.json} of the cv's ranked candidates.
    Returns {"key": the fold key, "picks": {fold: {category: {"type",
    "file", "score"}}}, "digests": {fold: sel_digest}, "scores": {fold:
    {etype: score}}}. Raises SelectionError naming every problem: the
    candidates rank on different keys, a candidate lacks a fold or the key
    (trained without --selection_frac), or two candidates held out different
    selection splits for one fold (another --selection_seed / fraction /
    fold set)."""
    order = [e for e, _, _ in catalog]
    by_type = {e: (f, c) for e, f, c in catalog}
    cands = [e for e in order if e in results]
    if not cands:
        raise SelectionError(f"{cv}: no candidate to rank")
    problems = []
    keys = {e: sel_key((results[e] or {}).get("config")) for e in cands}
    if len(set(keys.values())) > 1:
        problems.append(f"{cv}: candidates rank on different keys "
                        + ", ".join(f"{e}={k}" for e, k in keys.items()))
    key = keys[cands[0]]
    scores: Dict[int, Dict[str, float]] = {k: {} for k in range(num_folds)}
    digests: Dict[int, Dict[str, list]] = {k: {} for k in range(num_folds)}
    for e in cands:
        fe = fold_entries(results[e], num_folds)
        for k in range(num_folds):
            m = fe.get(k)
            if m is None:
                problems.append(f"{cv} fold {k}: {e} has no fold_metrics "
                                f"entry")
                continue
            s = _finite(m.get(keys[e]))
            if s is None:
                problems.append(
                    f"{cv} fold {k}: {e} has no finite {keys[e]} (trained "
                    f"without --selection_frac, i.e. SELECTION_FRAC empty?)")
            else:
                scores[k][e] = s
            d = m.get("sel_digest")
            if not d:
                problems.append(f"{cv} fold {k}: {e} records no sel_digest")
            else:
                digests[k].setdefault(str(d), []).append(e)
    for k in range(num_folds):
        if len(digests[k]) > 1:
            problems.append(
                f"{cv} fold {k}: the candidates held out different selection "
                f"splits (sel_digest " + "; ".join(
                    f"{d[:12]}: {', '.join(es)}"
                    for d, es in sorted(digests[k].items())) +
                "): trained with another --selection_frac / --selection_seed "
                "or on another fold set")
    if problems:
        raise SelectionError("; ".join(problems))
    picks: Dict[int, Dict[str, dict]] = {}
    for k in range(num_folds):
        picks[k] = {}
        for e in cands:   # catalog order: the first maximum wins a tie
            f, c = by_type[e]
            s = scores[k][e]
            if c not in picks[k] or s > picks[k][c]["score"]:
                picks[k][c] = {"type": e, "file": f, "score": s}
    return {"key": key, "picks": picks,
            "digests": {k: next(iter(digests[k])) for k in range(num_folds)},
            "scores": scores}


def selection_entries(catalog: Sequence[Tuple[str, str, str]],
                      picks: Dict[int, Dict[str, dict]]) -> Dict[str, dict]:
    """best_per_category.json's <cv> block from one cv's per-fold picks."""
    order = [e for e, _, _ in catalog]
    files = {e: f for e, f, _ in catalog}
    cats = sorted({c for p in picks.values() for c in p})
    out = {}
    for c in cats:
        per = {k: p[c] for k, p in sorted(picks.items()) if c in p}
        counts: Dict[str, int] = {}
        for w in per.values():
            counts[w["type"]] = counts.get(w["type"], 0) + 1
        types = [e for e in order if e in counts]
        modal = max(types, key=lambda e: (counts[e], -order.index(e)))
        out[c] = {
            "type": modal,
            "file": files[modal],
            "aupr": round(sum(w["score"] for w in per.values()) / len(per), 4),
            "types": types,
            "files": [files[e] for e in types],
            "n_folds_picked": {e: counts[e] for e in types},
            "per_fold": {str(k): {"type": w["type"], "file": w["file"],
                                  "score": round(w["score"], 6)}
                         for k, w in per.items()},
        }
    return out


def fold_picks(selection: dict, cv: str) -> Dict[int, Dict[str, dict]]:
    """{fold: {category: {"type", "file"}}} of a per-fold selection's cv."""
    block = (selection or {}).get(cv)
    if not isinstance(block, dict):
        raise SelectionError(f"best_per_category.json has no {cv!r} block")
    out: Dict[int, Dict[str, dict]] = {}
    for c, info in block.items():
        per = info.get("per_fold") if isinstance(info, dict) else None
        if not isinstance(per, dict) or not per:
            raise SelectionError(f"{cv}/{c}: no per_fold picks (a per_fold "
                                 f"selection records one per fold)")
        for k, w in per.items():
            out.setdefault(int(k), {})[c] = w
    folds = sorted(out)
    if folds != list(range(len(folds))):
        raise SelectionError(f"{cv}: per_fold picks for folds {folds}, not "
                             f"0..{len(folds) - 1}")
    lack = sorted({c for p in out.values() for c in p}
                  - set.intersection(*(set(p) for p in out.values())))
    if lack:
        raise SelectionError(f"{cv}: categor(ies) {lack} not picked in every "
                             f"fold")
    return out


def combo_spec(selection: dict, cv: str, drop: Optional[str] = None) -> dict:
    """The combo (drop=None) or its leave-one-category-out ablation (drop =
    that category) of one cv, as train.py takes it:
    {"names": types in --embeddings_paths order, "files": their files,
     "fold_modalities": {fold: [positions]} (None in global mode),
     "per_fold": {fold: [types]} (None in global mode),
     "n_per_fold": modalities each fold trains on, "n_union": len(files),
     "mode"}.
    Global: the categories sorted, one winner each (as always). Per-fold: the
    union of the cv's per-fold winners, categories sorted, then catalog order
    within a category ("types"); each fold maps to its own winners, in the
    same category order, so a fold that picked what a global run would have
    concatenates the same columns in the same order. drop removes the
    category from the combo (global) or from every fold's set (per-fold)."""
    mode = selection_mode(selection)
    block = (selection or {}).get(cv)
    if not isinstance(block, dict):
        raise SelectionError(f"best_per_category.json has no {cv!r} block")
    cats = sorted(c for c, i in block.items() if isinstance(i, dict))
    if drop is not None:
        if drop not in cats:
            raise SelectionError(f"category {drop!r} not present for {cv} "
                                 f"(have: {cats}): nothing to ablate")
        cats = [c for c in cats if c != drop]
    if mode == "global":
        names = [block[c]["type"] for c in cats]
        files = [block[c]["file"] for c in cats]
        return {"mode": mode, "names": names, "files": files,
                "fold_modalities": None, "per_fold": None,
                "n_per_fold": len(files), "n_union": len(files)}
    picks = fold_picks(selection, cv)
    names, files = [], []
    for c in cats:
        info = block[c]
        types = list(info.get("types") or [])
        tfiles = list(info.get("files") or [])
        picked = {w["type"] for p in picks.values() for w in [p[c]]}
        if len(types) != len(tfiles) or set(types) != picked:
            raise SelectionError(f"{cv}/{c}: types {types} are not the "
                                 f"embeddings its per_fold entries pick "
                                 f"{sorted(picked)}")
        names += types
        files += tfiles
    pos = {t: i for i, t in enumerate(names)}
    fmap = {k: [pos[p[c]["type"]] for c in cats]
            for k, p in sorted(picks.items())}
    return {"mode": mode, "names": names, "files": files,
            "fold_modalities": fmap,
            "per_fold": {k: [names[i] for i in v] for k, v in fmap.items()},
            "n_per_fold": len(cats), "n_union": len(files)}


def _train_cwd(cfg: dict, cfg_dir) -> Optional[Path]:
    """The directory train.py ran in, from where its config.json is
    (cfg_dir = the run's output directory) and the --output_dir it recorded:
    cfg_dir itself less output_dir's trailing parts (None when output_dir is
    absent or is not cfg_dir's tail, e.g. a copied config)."""
    out = (cfg or {}).get("output_dir")
    if not out:
        return None
    d = Path(cfg_dir).resolve()
    out = Path(str(out))
    if out.is_absolute():
        return None  # says nothing of the directory it ran in
    parts = [p for p in out.parts if p not in ("", ".")]
    if not parts or ".." in parts or list(d.parts[-len(parts):]) != parts:
        return None
    return d.parents[len(parts) - 1]


def config_fold_files(cfg: dict, cfg_dir=None
                      ) -> Optional[Dict[int, List[str]]]:
    """{fold: [embedding file names]} a run's config.json trained each fold
    on: its --fold_modalities (JSON text, or a JSON file) over its
    embeddings_paths; None when it trained every fold on every file.
    config.json keeps the argument as train.py received it, so a RELATIVE
    file path is relative to the directory train.py ran in (siamese_sl/ for
    the SLURM jobs), not to this process's: with cfg_dir (the directory
    holding config.json) it is resolved against that run directory
    (_train_cwd), and refused (FileNotFoundError) when that is unknown or the
    file is not there. Without cfg_dir: against the current directory."""
    paths = [Path(str(p)).name for p in (cfg or {}).get("embeddings_paths")
             or []]
    raw = (cfg or {}).get("fold_modalities")
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        txt = raw
        if not raw.lstrip().startswith("{"):
            f = Path(raw)
            if cfg_dir is not None and not f.is_absolute():
                base = _train_cwd(cfg, cfg_dir)
                if base is None:
                    raise FileNotFoundError(
                        f"relative --fold_modalities file {raw!r}: the "
                        f"directory train.py ran in is not recoverable from "
                        f"{cfg_dir} and its output_dir "
                        f"{(cfg or {}).get('output_dir')!r}")
                f = base / f
            if not f.is_file():
                raise FileNotFoundError(f"--fold_modalities file {str(f)!r} "
                                        f"not found")
            txt = f.read_text()
        raw = json.loads(txt)
    return {int(k): [paths[int(i)] for i in v] for k, v in raw.items()}


def spec_fold_files(spec: dict) -> Optional[Dict[int, List[str]]]:
    """combo_spec's per-fold sets as file names (None in global mode)."""
    if spec.get("fold_modalities") is None:
        return None
    names = [Path(f).name for f in spec["files"]]
    return {int(k): [names[i] for i in v]
            for k, v in spec["fold_modalities"].items()}


def run_matches_spec(cfg: dict, spec: dict, cfg_dir=None) -> Optional[str]:
    """Why a run's config.json is not combo_spec's run (None: it is): its
    embeddings_paths (by file name, in order) and, per fold, the files it
    trained on. cfg_dir: the directory holding config.json
    (config_fold_files)."""
    used = [Path(str(p)).name for p in (cfg or {}).get("embeddings_paths")
            or []]
    want = [Path(f).name for f in spec["files"]]
    if used != want:
        return (f"trained on {', '.join(used) or 'nothing'}, not "
                f"{', '.join(want)}")
    try:
        got = config_fold_files(cfg, cfg_dir)
    except (OSError, ValueError, TypeError, IndexError) as e:
        return f"unreadable fold_modalities ({type(e).__name__}: {e})"
    exp = spec_fold_files(spec)
    if exp is None:
        if got is not None and any(v != want for v in got.values()):
            return "trained per-fold modality sets, the selection is global"
        return None
    if got is None:
        return "trained every fold on the union, not each fold's own winners"
    bad = sorted(k for k in set(exp) | set(got) if exp.get(k) != got.get(k))
    if bad:
        return ("per-fold modalities differ from the selection's in fold(s) "
                + ", ".join(map(str, bad)))
    return None


def pick_counts(selection: dict, cv: str) -> Dict[str, int]:
    """{etype: number of the cv's folds that picked it} of a per-fold
    selection ({} for a global one, or when the cv has no block)."""
    block = (selection or {}).get(cv)
    if selection_mode(selection) != "per_fold" or not isinstance(block, dict):
        return {}
    out: Dict[str, int] = {}
    for c, info in sorted(block.items()):
        if isinstance(info, dict) and isinstance(info.get("n_folds_picked"),
                                                 dict):
            for e, n in info["n_folds_picked"].items():
                out[e] = out.get(e, 0) + int(n)
    return out


def allmod_spec(catalog: Sequence[Tuple[str, str, str]],
                data_dir: Optional[str] = None,
                optional: Sequence[str] = ()) -> dict:
    """The all-modalities combo (slurm/run_allmod_combo.sh, 2026-11-12): EVERY
    catalog embedding, in catalog order, no selection.
    {"names": types, "files": their files, "absent_optional": the catalog
    types skipped because their file was not generated (data_dir given) and
    generation accepts their absence (`optional`, OPTIONAL_TYPES)}.
    Raises SelectionError for a repeated type or file, for a missing file of
    a type outside `optional` (the paper's catalog would silently shrink), or
    for fewer than two modalities."""
    names, files, absent, required = [], [], [], []
    seen_t, seen_f = set(), set()
    for etype, efile, _cat in catalog:
        if etype in seen_t or efile in seen_f:
            raise SelectionError(f"catalog lists {etype} / {efile} twice")
        seen_t.add(etype)
        seen_f.add(efile)
        if data_dir is not None and \
                not os.path.isfile(os.path.join(data_dir, efile)):
            (absent if etype in set(optional) else required).append(etype)
            continue
        names.append(etype)
        files.append(efile)
    if required:
        raise SelectionError(
            "embedding file(s) of required catalog type(s) missing in "
            f"{data_dir}: {', '.join(required)} (not in OPTIONAL_TYPES: "
            "generate them; the all-modalities combo trains on the whole "
            "catalog)")
    if len(files) < 2:
        raise SelectionError(f"only {len(files)} catalog embedding(s) "
                             f"available: nothing to combine")
    return {"names": names, "files": files, "absent_optional": absent}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    am = sub.add_parser("allmod_spec")
    am.add_argument("--catalog", nargs="+", required=True,
                    help="EMB_CATALOG entries type:file:category")
    am.add_argument("--optional", nargs="*", default=[],
                    help="OPTIONAL_TYPES: types whose absence is accepted")
    am.add_argument("--data_dir", default=None,
                    help="prefix every file with it; check which exist")
    cs = sub.add_parser("combo_spec")
    cs.add_argument("--selection", required=True)
    cs.add_argument("--cv", required=True)
    cs.add_argument("--drop", default=None)
    cs.add_argument("--data_dir", default=None,
                    help="prefix every file with it and require it to exist")
    a = ap.parse_args(argv)
    if a.cmd == "allmod_spec":
        try:
            cat = []
            for e in a.catalog:
                parts = e.split(":")
                if len(parts) != 3 or not all(parts):
                    raise SelectionError(f"catalog entry {e!r} is not "
                                         f"type:file:category")
                cat.append(tuple(parts))
            spec = allmod_spec(cat, a.data_dir, a.optional)
        except SelectionError as e:
            print(f"fold_selection: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        paths = spec["files"]
        if a.data_dir:
            paths = [os.path.join(a.data_dir, f) for f in paths]
        print(f"{' '.join(paths)}|{'+'.join(spec['names'])}|{len(paths)}|"
              f"{','.join(spec['absent_optional'])}")
        return 0
    try:
        with open(a.selection) as f:
            sel = json.load(f)
        spec = combo_spec(sel, a.cv, a.drop)
    except (OSError, ValueError, KeyError, TypeError, SelectionError) as e:
        print(f"fold_selection: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    paths = spec["files"]
    if a.data_dir:
        paths = [os.path.join(a.data_dir, f) for f in paths]
        gone = [p for p in paths if not os.path.isfile(p)]
        if gone:
            # Every winner trained in this run's Step 1, so its file existed
            # then: a combo without it is not the selected model.
            print("fold_selection: embedding file(s) missing: "
                  + "; ".join(gone), file=sys.stderr)
            return 1
    fmap = ("" if spec["fold_modalities"] is None else
            json.dumps({str(k): v for k, v in spec["fold_modalities"].items()},
                       separators=(",", ":")))
    print(f"{' '.join(paths)}|{'+'.join(spec['names'])}|"
          f"{spec['n_per_fold']}|{spec['n_union']}|{fmap}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
