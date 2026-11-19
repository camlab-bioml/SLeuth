#!/usr/bin/env python3
"""
Predicted SL probability of ALL gene pairs, one matrix per cell line, from a
trained SLeuth combo; optionally the predicted screen value (e.g. Adamson
Gamma) of all gene pairs from the models eval_finetuning.py fine-tuned.

Post-hoc and read-only on the model: it loads <model_dir>/checkpoints/
fold_<k>_best.pt, rebuilds each fold's features from the transform stored in
that checkpoint, and scores every gene pair with retrieval_metrics.
build_any_score_matrix(only_head=h), the dense scorer train.py grades with
(checked here against model.forward on sampled pairs, per fold and head).
Nothing is retrained and no reported number depends on its output.

What is written (--out_dir, default <model_dir>/pair_predictions/)
-------------------------------------------------------------------
  manifest.json            what every file is; written LAST, so a directory
                           without it is an unfinished export
  genes.txt                Entrez ID of row/column i, one per line
  genes.tsv                index, entrez_id, symbol (HGNC)
  sl_prob_<HEAD>.npy       (n, n) symmetric: sigmoid(logit of that head),
                           averaged over the fold models. One file per head:
                             cell-line model   K562, JURKAT, A549, HELA, A375,
                                               293T, PC9, OTHER (cell_line_
                                               vocab.HEADS), each read with
                                               that line's embedding, plus ANY
                                               = the largest of the eight per
                                               fold (the score the benchmark
                                               ranks by; no head is trained on
                                               the any-line label, so it is
                                               not a calibrated probability)
                             pooled / single-output model   ANY only (one
                                               head, trained on the any-line
                                               label; no cell-line input)
  fold_<k>_sl_prob_<HEAD>.npy   the same per fold model      (--per_fold)
  sl_logit_<HEAD>.npy           the fold-mean logit          (--also_logits)
  fold_<k>_sl_logit_<HEAD>.npy  that logit per fold model    (both flags)
  finetuned_<Screen>_<mode>_<cv>.npy
                           (--finetuned_eval_dirs) predicted screen value of
                           every gene pair, in the screen's own sign and units
                           (Adamson: the predicted Gamma), from the models
                           eval_finetuning.py fine-tuned on that screen under
                           its <cv> split with <mode> (LP_last / Full / Scratch),
                           averaged over the fold models. Each is scored
                           through the head it was fine-tuned with: the
                           screen's cell line (K562 for Adamson) on a
                           cell-line model, the one head otherwise.
  fold_<k>_finetuned_<Screen>_<mode>_<cv>.npy
                           that value per fold model        (--per_fold)

The diagonal (a gene with itself) is NaN in every matrix. The fold mean is a
mean of the folds' probabilities (or predicted values), not of logits. It is
NOT a held-out prediction for a labelled SynLethDB pair: each fold model
trained on 72% of the labelled pairs under cv1 and on about half under cv2 /
cv3 (the SLURM job exports the cv3 combos by default). With --per_fold, fold
k's matrix is held out for the pairs outside fold k's train and val splits
(sl_comparison/folds/<cv>/fold_<k>/); under cv3 the fold's test split is only
the pairs with both genes unseen, one labelled pair in five over the five
folds. float32 stores a probability above 1 - 6e-8 as exactly 1 (logit above
~17); manifest.json counts those entries and --also_logits keeps the
unsaturated ranking.

A cell-line head is fitted only on the pairs screened in that line, so its
probabilities sit on the scale of that line's screened pairs, whose SL
fraction runs from 3.5% (A549) to 44% (A375): rank pairs within a head, do
not compare levels (or counts above a cut-off) across heads. OTHER is a
catch-all of rare lines and pairs with no recorded line, not one cell line.

A finetuned_* matrix is IN-SAMPLE at the screen's own pairs: every fold model
was fine-tuned on the same screen split (cv1: 72% of the pairs fitted, 8% used
to select the checkpoint), so --per_fold does not help there. Only the pairs
marked split == test in <eval dir>/<Screen>_finetuned_allpairs_<mode>.csv are
held out; a screen pair's entry is that file's pred_outcome_mean. Every pair
the screen did not measure (99.9% of an Adamson matrix at 19,397 genes) is an
extrapolation that no metric of the run grades.

Gene universe (--universe)
--------------------------
  all      every gene of the embedding files (19,397 on the current build):
           the only universe that covers every pair of the external screens
           (398 of Adamson's 542 genes are in the shared 8,844). Every
           labelled pair the models were trained on lies within those 8,844
           genes: a pair with a gene outside them (79% of this universe's
           pairs) is an extrapolation from the gene embeddings.
  shared   the genes of --shared_genes, in that order
           (sl_comparison/folds/genes.txt, the order of every model's
           fold_<k>_<cv>_predictions.npy).
  --genes_file PATH   your own list (Entrez IDs or symbols, one per line).

Size: one matrix is n*n*4 bytes (1.5 GB at 19,397 genes, 0.31 GB at 8,844); a
cell-line model writes 9 of them by default, a pooled model 1.

Usage (from siamese_sl/):
  python export_pair_predictions.py \\
      --model_dir results/<combo>_cv3 \\
      --embeddings_paths ../data/all_genes_a.pt ../data/all_genes_b.pt \\
      --finetuned_eval_dirs eval_finetuning_cv1 --finetuned_datasets Adamson

Read it back:
  from export_pair_predictions import open_pair_predictions
  pp = open_pair_predictions("results/<combo>_cv3/pair_predictions")
  pp.matrix("K562")[pp.index["672"], pp.index["142"]]      # BRCA1 - PARP1
  pp.pair_table([("BRCA1", "PARP1")])                      # every column
A gene is named by its Entrez ID, its symbol in genes.tsv, or an older name
that the repository's data/cache/hgnc_complete_set.txt resolves (read by its
own path, from any working directory; nothing is downloaded).

Exit status 1, with nothing cleared, when the model dir, its checkpoint set,
the embedding files or --device are not usable; exit 1 after writing the rest
when a requested fine-tuned set is missing, partial, unreadable, from other
checkpoints, or fails while it is scored.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data_loader import load_raw_multimodal
from eval_finetuning import (
    ANY_HEAD,
    FINETUNED_DIR,
    FINETUNED_FORMAT,
    _load_fold_transform_and_embeddings,
    _multicell_heads,
    checkpoint_input_problems,
    head_labels,
    load_multicell_model,
    resolve_fold_checkpoints,
    symbol_entrez,
)
from predict import load_model
from retrieval_metrics import build_any_score_matrix
from siamese_esm import SiameseSL, SiameseSLMultiCell

MANIFEST = "manifest.json"
MANIFEST_FORMAT = "siamese_sl.pair_predictions.v1"
DEFAULT_SUBDIR = "pair_predictions"
# Present from the first write until the manifest is: an export that died.
IN_PROGRESS = ".export_in_progress"
# What a screen column measures, where its name does not say (README.md,
# "Predictions for all gene pairs").
SCREEN_VALUE_NOTES = {
    ("Adamson", "Gamma"): "Gamma is the growth phenotype of the double "
                          "knockdown, not a genetic-interaction score: about "
                          "90% of its variance over the screen's pairs is the "
                          "two genes' own (additive) growth effects",
}
# Everything this script writes; a rerun removes exactly these, manifest first.
OWN_PATTERNS = (MANIFEST, "genes.txt", "genes.tsv", "*.tmp.npy",
                "sl_prob_*.npy", "sl_logit_*.npy", "finetuned_*.npy",
                "fold_*_sl_prob_*.npy", "fold_*_sl_logit_*.npy",
                "fold_*_finetuned_*.npy")
# Names other tools use too (sl_comparison/folds/genes.txt indexes every
# model's graded matrix): replaced only in a directory an export of this
# script wrote.
SHARED_NAMES = (MANIFEST, "genes.txt", "genes.tsv")


def foreign_files(out_dir: Path) -> List[str]:
    """SHARED_NAMES files in out_dir that no export of this script wrote
    ([] when the directory is new, an export, or one that died half-way)."""
    if (out_dir / IN_PROGRESS).exists():
        return []
    mf = out_dir / MANIFEST
    if mf.is_file():
        try:
            if json.loads(mf.read_text()).get("format") == MANIFEST_FORMAT:
                return []
        except (OSError, ValueError):
            pass
    return [n for n in SHARED_NAMES if (out_dir / n).exists()]


# =============================================================================
# Loading
# =============================================================================


def load_scorer(ckpt_path, device: str) -> torch.nn.Module:
    """The model a checkpoint holds, in eval mode: SiameseSLMultiCell
    (cell-line or pooled) or the single-output SiameseSL.

    Works on a train.py fold checkpoint and on a fine-tuned model saved by
    eval_finetuning.save_finetuned_model (same `model_state_dict` / `config`
    keys).
    """
    sd = torch.load(str(ckpt_path), map_location="cpu",
                    weights_only=False)["model_state_dict"]
    try:
        _multicell_heads(sd, ckpt_path)
        model = load_multicell_model(str(ckpt_path), device=device)
    except ValueError:
        model = load_model(str(ckpt_path), model_type="siamese", device=device)
    if not isinstance(model, (SiameseSL, SiameseSLMultiCell)):
        raise ValueError(
            f"{ckpt_path}: {type(model).__name__} has no dense all-pairs "
            f"scorer (only SiameseSL / SiameseSLMultiCell do)")
    return model.eval()


def sha256_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_universe(universe: str, genes_file: Optional[str],
                     shared_genes: str,
                     gene_to_idx: Dict[str, int],
                     idx_to_gene: Dict[int, str]) -> Tuple[List[str],
                                                           np.ndarray, str]:
    """(Entrez IDs in output order, their embedding rows, a label).

    A listed gene with no embedding row is an error, never a silent drop: the
    matrix would otherwise be shorter than the list it claims to follow.
    """
    if genes_file is None and universe == "all":
        genes = [idx_to_gene[i] for i in range(len(idx_to_gene))]
        return genes, np.arange(len(genes), dtype=np.int64), "all"

    path = Path(genes_file if genes_file is not None else shared_genes)
    if not path.is_file():
        raise ValueError(f"gene list not found: {path}")
    tokens = [ln.strip() for ln in open(path) if ln.strip()]
    genes: List[str] = []
    unknown: List[str] = []
    mapper = None
    for t in tokens:
        if t in gene_to_idx:
            genes.append(t)
            continue
        # Not an Entrez ID of the embedding files: try it as a symbol.
        if mapper is None:
            from gene_name_utils import get_mapper
            mapper = get_mapper()
        e = symbol_entrez(mapper, t)
        if e is not None and e in gene_to_idx:
            genes.append(e)
        else:
            unknown.append(t)
    if unknown:
        raise ValueError(
            f"{path}: {len(unknown)} of {len(tokens)} genes have no embedding "
            f"row (first: {unknown[:5]})")
    if len(set(genes)) != len(genes):
        dup = sorted({g for g in genes if genes.count(g) > 1})[:5]
        raise ValueError(f"{path}: duplicate genes (first: {dup})")
    if len(genes) < 2:
        raise ValueError(f"{path}: fewer than 2 genes")
    rows = np.asarray([gene_to_idx[g] for g in genes], dtype=np.int64)
    label = f"file:{path}" if genes_file is not None else "shared"
    return genes, rows, label


def discover_finetuned(model_dir: Path, eval_dirs: Sequence[str],
                       datasets: Optional[Sequence[str]],
                       modes: Optional[Sequence[str]],
                       checkpoint_paths: List[Path],
                       fold_sha: List[str]) -> Tuple[List[dict], List[str]]:
    """Fine-tuned model sets to export, and why any requested one cannot be.

    One group per (eval dir, dataset, mode): a complete set of
    finetuned_models/<dataset>_<mode>_fold<k>.pt for k = 0..K-1, each
    fine-tuned from THIS model dir's fold-k checkpoint (sha256 recorded by
    eval_finetuning.py), all through one head and one sign. Anything else is
    reported, not exported: a retrain into the same dir leaves the old eval
    dirs behind, and their models would be paired with the new folds'
    features.
    """
    groups: List[dict] = []
    problems: List[str] = []
    n_folds = len(checkpoint_paths)
    for name in eval_dirs:
        edir = Path(name) if os.path.isabs(name) else model_dir / name
        fdir = edir / FINETUNED_DIR
        files = sorted(fdir.glob("*.pt")) if fdir.is_dir() else []
        if not files:
            problems.append(f"{edir}: no {FINETUNED_DIR}/*.pt (was "
                            f"eval_finetuning.py run there, without "
                            f"--no_save_finetuned?)")
            continue
        by_key: Dict[Tuple[str, str], Dict[int, Tuple[Path, dict]]] = {}
        for f in files:
            # A save that failed part-way (a full quota, a killed task) leaves
            # a file torch cannot read: report it, export everything else.
            try:
                ck = torch.load(str(f), map_location="cpu", weights_only=False)
            except Exception as e:  # noqa: BLE001  (OSError, EOFError, ...)
                why = f"{f}: unreadable ({type(e).__name__}: {e})"
                # eval_finetuning.py names it <dataset>_<mode>_fold<k>.pt: a
                # file of a screen or a mode that was not asked for is not a
                # missing set (the whole _<mode>: LP is not LP_last).
                stem = f.name.rsplit("_fold", 1)[0]
                if ((datasets and not any(stem.startswith(f"{d}_")
                                          for d in datasets))
                        or (modes and not any(stem.endswith(f"_{m}")
                                              for m in modes))):
                    print(f"NOTE: ignoring {why}; not a requested dataset "
                          f"or mode", file=sys.stderr)
                else:
                    problems.append(f"{why}; rerun eval_finetuning.py for "
                                    f"that split")
                continue
            if not isinstance(ck, dict) or ck.get("format") != FINETUNED_FORMAT:
                problems.append(f"{f}: not a {FINETUNED_FORMAT} file")
                continue
            if datasets and ck["dataset"] not in datasets:
                continue
            if modes and ck["mode"] not in modes:
                continue
            meta = {k: v for k, v in ck.items() if k != "model_state_dict"}
            by_key.setdefault((ck["dataset"], ck["mode"]), {})[int(
                ck["fold"])] = (f, meta)
        if not by_key:
            problems.append(f"{fdir}: no model matches datasets="
                            f"{list(datasets or [])} modes={list(modes or [])}")
        else:
            # A requested name with no model at all, beside one that has
            # some, is as missing as a set whose folds are incomplete.
            for d in dict.fromkeys(datasets or [None]):
                for m in dict.fromkeys(modes or [None]):
                    if not any(d in (None, kd) and m in (None, km)
                               for kd, km in by_key):
                        problems.append(
                            f"{fdir}: no model of the requested "
                            + " / ".join(x for x in (d, m) if x))
        for (dataset, mode), folds in sorted(by_key.items()):
            what = f"{fdir}: {dataset} / {mode}"
            if sorted(folds) != list(range(n_folds)):
                problems.append(f"{what}: folds {sorted(folds)}, need "
                                f"0..{n_folds - 1}")
                continue
            metas = [folds[k][1] for k in range(n_folds)]
            stale = [k for k in range(n_folds)
                     if metas[k].get("source_checkpoint_sha256") != fold_sha[k]]
            if stale:
                problems.append(
                    f"{what}: fold(s) {stale} were fine-tuned from other "
                    f"weights than {model_dir}/checkpoints holds now "
                    f"(retrained since that eval?); rerun eval_finetuning.py")
                continue
            for field in ("head_idx", "head", "sign", "column", "cv_type"):
                if len({json.dumps(m.get(field)) for m in metas}) != 1:
                    problems.append(f"{what}: folds disagree on {field}")
                    break
            else:
                m0 = metas[0]
                if not float(m0["sign"]):
                    problems.append(f"{what}: sign 0")
                    continue
                groups.append({
                    "dataset": dataset,
                    "mode": mode,
                    "eval_cv": m0["cv_type"],
                    "column": m0["column"],
                    "sign": float(m0["sign"]),
                    "head": m0["head"],
                    "head_idx": int(m0["head_idx"]),
                    "source": str(fdir),
                    # Relative to the model dir, which is where the reader
                    # looks for it again (PairPredictions.stale_reasons).
                    "eval_dir": os.path.relpath(edir.resolve(),
                                                model_dir.resolve()),
                    # .get: models saved before the counts were recorded.
                    "screen_split": {k: m0.get(k) for k in (
                        "cv_type", "split_seed", "train_frac", "val_frac",
                        "n_pairs", "n_genes", "n_train", "n_val", "n_test")},
                    "files": [folds[k][0] for k in range(n_folds)],
                    "model_sha256": {folds[k][0].name:
                                     sha256_bytes(folds[k][0])
                                     for k in range(n_folds)},
                    "folds": [{
                        "fold": k,
                        "a": metas[k]["affine"]["a"],
                        "c": metas[k]["affine"]["c"],
                        "best_epoch": metas[k].get("best_epoch"),
                    } for k in range(n_folds)],
                    "stem": f"finetuned_{dataset}_{mode}_{m0['cv_type']}",
                })
    stems = [g["stem"] for g in groups]
    for s in sorted({s for s in stems if stems.count(s) > 1}):
        problems.append(f"two fine-tuned sets would both be written as "
                        f"{s}.npy (two eval dirs of the same split?)")
        groups = [g for g in groups if g["stem"] != s]
    return groups, problems


# =============================================================================
# Scoring
# =============================================================================


def sample_pairs(n: int, k: int, seed: int = 0) -> Tuple[np.ndarray,
                                                         np.ndarray]:
    """Up to k distinct-gene pairs (i != j) of range(n), reproducibly."""
    rs = np.random.RandomState(seed)
    k = int(max(1, min(k, n * (n - 1))))
    i = rs.randint(0, n, size=k)
    j = (i + rs.randint(1, n, size=k)) % n
    return i.astype(np.int64), j.astype(np.int64)


@torch.no_grad()
def forward_reference(model, E: torch.Tensor, pi: torch.Tensor,
                      pj: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    """model(E[pi], E[pj]) for the sampled pairs, (k, num_heads) float32."""
    parts = [model(E[pi[s:s + batch]], E[pj[s:s + batch]]).float()
             for s in range(0, int(pi.numel()), batch)]
    return torch.cat(parts)


def check_against_forward(L: torch.Tensor, ref: torch.Tensor,
                          pi: torch.Tensor, pj: torch.Tensor,
                          what: str) -> float:
    """Raise unless the dense matrix reproduces the ordinary forward on the
    sampled pairs. Same bound as train.py's export guard: 1e-3 absolute,
    widened to 1e-5 of the logit scale (the logits are unbounded without
    LayerNorm). A wrong gene order, head or weight set is O(spread)."""
    if not torch.isfinite(L).all():
        raise RuntimeError(f"{what}: non-finite entries in the all-pairs "
                           f"matrix")
    err = float((L[pi, pj] - ref).abs().max())
    tol = max(1e-3, 1e-5 * float(ref.abs().max()))
    if not err <= tol:
        raise RuntimeError(
            f"{what}: the all-pairs matrix disagrees with model.forward on "
            f"{int(pi.numel())} sampled pairs (max |diff| {err:.3e} > "
            f"{tol:.3e})")
    return err


def write_matrix(path: Path, arr: np.ndarray, dtype: str) -> Dict[str, float]:
    """Save `arr` (n, n) float32 with a NaN diagonal as `dtype`, atomically.
    Mutates arr's diagonal. Returns the off-diagonal statistics as stored."""
    np.fill_diagonal(arr, np.nan)
    out = arr if arr.dtype == np.dtype(dtype) else arr.astype(dtype)
    tmp = path.with_name(path.name[:-len(".npy")] + ".tmp.npy")
    try:
        np.save(tmp, out)
        os.replace(tmp, path)
    finally:
        # A save that failed part-way leaves no piece behind (after the
        # rename there is nothing under this name).
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "min": float(np.nanmin(out)),
        "max": float(np.nanmax(out)),
        "mean": float(np.nanmean(out, dtype=np.float64)),
        "n_exactly_0": int((out == 0).sum()),
        "n_exactly_1": int((out == 1).sum()),
    }


def plan_passes(units: List[tuple], arrays_per_unit: Dict[tuple, int],
                bytes_per_array: int, ram_bytes: float) -> List[List[tuple]]:
    """Split the units into passes whose fold-mean accumulators fit in RAM.
    Every pass re-reads the folds, so one pass is the fast path."""
    passes: List[List[tuple]] = [[]]
    used = 0
    for u in units:
        need = arrays_per_unit[u] * bytes_per_array
        if passes[-1] and used + need > ram_bytes:
            passes.append([])
            used = 0
        passes[-1].append(u)
        used += need
    return [p for p in passes if p]


# =============================================================================
# Reading an export back
# =============================================================================


# The gene-name table the pipeline reads (gene_name_utils.get_mapper()'s
# ../data/cache as seen from siamese_sl/), by its own path.
HGNC_TABLE = (Path(__file__).resolve().parents[1] / "data" / "cache"
              / "hgnc_complete_set.txt")
_alias_mapper: list = []    # [mapper or None], once it has been looked for


def alias_mapper():
    """The mapper PairPredictions.gene_row resolves older gene names with, or
    None where HGNC_TABLE is absent.

    An instance of its own, not gene_name_utils.get_mapper(): that one reads
    ../data/cache relative to the working directory (HGNC_TABLE only from one
    level below the repository root), downloads the table where it is missing,
    and is one per process, so a reader would also choose the table of every
    later caller. Reading an export back downloads and creates nothing.
    """
    if not _alias_mapper:
        m = None
        if HGNC_TABLE.is_file():
            from gene_name_utils import GeneNameMapper
            m = GeneNameMapper(cache_dir=str(HGNC_TABLE.parent))
        _alias_mapper.append(m)
    return _alias_mapper[0]


class PairPredictions:
    """A finished export: manifest, gene index and lazily mapped matrices.

    check_model: refuse an export whose model dir was retrained, or whose
    fine-tuned models were redone, since (see stale_reasons); pass False to
    read it anyway.
    """

    def __init__(self, pred_dir, check_model: bool = True):
        self.dir = Path(pred_dir)
        mf = self.dir / MANIFEST
        if not mf.is_file():
            raise FileNotFoundError(
                f"{mf} missing: {self.dir} is not a finished export")
        self.manifest = json.loads(mf.read_text())
        if check_model:
            stale = self.stale_reasons()
            if stale:
                raise RuntimeError(
                    f"{self.dir} no longer matches the models under "
                    f"{self.dir.parent} ({'; '.join(stale)}): export it "
                    f"again, or pass check_model=False to read the old one")
        self.genes = [ln.strip() for ln in open(self.dir / "genes.txt")
                      if ln.strip()]
        self.index = {g: i for i, g in enumerate(self.genes)}
        self.symbols: Dict[str, str] = {}
        tsv = self.dir / "genes.tsv"
        if tsv.is_file():
            for ln in tsv.read_text().splitlines()[1:]:
                parts = ln.split("\t")
                if len(parts) >= 3 and parts[2]:
                    self.symbols[parts[2].upper()] = parts[1]

    def stale_reasons(self) -> List[str]:
        """Why this export no longer describes the models beside it ([] if it
        does, or if there is nothing to compare with).

        Compared by sha256, where the export still sits in its model dir
        (<model_dir>/<this dir>, the default layout): the fold checkpoints it
        scored (a retrain into the same dir rewrites checkpoints/ and leaves
        this directory behind) and, per fine-tuned set, the
        eval_finetuning_<cv>/finetuned_models/ files (an eval rerun rewrites
        them). A file that is gone, or an export copied elsewhere, proves
        nothing and is not reported.
        """
        md = self.dir.resolve().parent
        if Path(str(self.manifest.get("model_dir", ""))).name != md.name:
            return []

        def changed(folder: Path, recorded: Dict[str, str]) -> List[str]:
            return [n for n, sha in sorted((recorded or {}).items())
                    if (folder / n).is_file()
                    and sha256_bytes(folder / n) != sha]

        out = [f"checkpoints/{n} was retrained" for n in changed(
            md / "checkpoints", self.manifest.get("checkpoint_sha256"))]
        for ft in self.manifest.get("finetuned", []):
            sub = f"{ft.get('eval_dir', '')}/{FINETUNED_DIR}"
            out += [f"{sub}/{n} was fine-tuned again (stale: {ft['file']})"
                    for n in changed(md / sub, ft.get("model_sha256"))]
        return out

    def names(self) -> List[str]:
        """Every matrix name: the head labels, then the fine-tuned sets."""
        return (list(self.manifest["sl_probability"]["files"]) +
                [f["name"] for f in self.manifest.get("finetuned", [])])

    def _file(self, name: str) -> str:
        files = dict(self.manifest["sl_probability"]["files"])
        files.update({f["name"]: f["file"]
                      for f in self.manifest.get("finetuned", [])})
        if name not in files:
            raise KeyError(f"{name!r} not in {sorted(files)}")
        return files[name]

    def matrix(self, name: str, mmap: bool = True) -> np.ndarray:
        """The (n, n) matrix of a head label (e.g. 'K562', 'ANY') or of a
        fine-tuned set (e.g. 'finetuned_Adamson_Full_cv1')."""
        return np.load(self.dir / self._file(name),
                       mmap_mode="r" if mmap else None)

    def gene_row(self, gene: str) -> int:
        """Row of an Entrez ID or a gene symbol: the HGNC symbol of genes.tsv,
        else any name the repository's HGNC table (alias_mapper) resolves to
        a gene of this export (a previous symbol or alias, e.g. Adamson's
        MRE11A for MRE11)."""
        g = str(gene).strip()
        if g in self.index:
            return self.index[g]
        e = self.symbols.get(g.upper())
        if e is None and not g.isdigit():
            # Not a genes.tsv symbol: resolve it as the pipeline resolves a
            # screen's symbols (older names included, the pinned ones as
            # eval_finetuning pins them). Never for digits (an Entrez ID;
            # HGNC lists 225, 3635 and 52 as aliases of other genes). Without
            # the table only a pinned name is known.
            try:
                e = symbol_entrez(alias_mapper(), g)
            except Exception:  # noqa: BLE001
                e = None
        if e is None or e not in self.index:
            raise KeyError(f"gene {gene!r} is not in this export")
        return self.index[e]

    def pair_table(self, pairs: Iterable[Tuple[str, str]],
                   names: Optional[Sequence[str]] = None):
        """One row per pair, one column per matrix (pandas DataFrame)."""
        import pandas as pd
        pairs = [(str(a), str(b)) for a, b in pairs]
        i = np.asarray([self.gene_row(a) for a, _ in pairs], dtype=np.int64)
        j = np.asarray([self.gene_row(b) for _, b in pairs], dtype=np.int64)
        out = {"gene1": [a for a, _ in pairs], "gene2": [b for _, b in pairs],
               "gene1_entrez": [self.genes[x] for x in i],
               "gene2_entrez": [self.genes[x] for x in j]}
        for name in (names if names is not None else self.names()):
            col = name if name.startswith("finetuned_") else f"sl_prob_{name}"
            out[col] = np.asarray(self.matrix(name)[i, j])
        return pd.DataFrame(out)


def open_pair_predictions(pred_dir, check_model: bool = True) -> PairPredictions:
    return PairPredictions(pred_dir, check_model=check_model)


# =============================================================================
# Main
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="SL probability of all gene pairs per cell line from a "
        "trained SLeuth combo (and, optionally, the fine-tuned models' "
        "predicted screen value for all gene pairs).")
    p.add_argument("--model_dir", required=True,
                   help="Trained combo dir (config.json + "
                   "checkpoints/fold_*_best.pt).")
    p.add_argument("--embeddings_paths", nargs="+", required=True,
                   help="The embedding .pt files the combo was trained on, "
                   "in the training order.")
    p.add_argument("--out_dir", default=None,
                   help=f"Default: <model_dir>/{DEFAULT_SUBDIR}.")
    p.add_argument("--universe", default="all", choices=["all", "shared"],
                   help="all = every gene of the embedding files; shared = "
                   "the genes of --shared_genes, in that order.")
    p.add_argument("--shared_genes",
                   default="../sl_comparison/folds/genes.txt",
                   help="Gene list of --universe shared (read only).")
    p.add_argument("--genes_file", default=None,
                   help="Your own gene list (Entrez IDs or symbols, one per "
                   "line); overrides --universe.")
    p.add_argument("--per_fold", action="store_true",
                   help="Also write each fold model's matrices "
                   "(fold_<k>_*.npy): num_folds times the disk.")
    p.add_argument("--also_logits", action="store_true",
                   help="Also write sl_logit_<HEAD>.npy (twice the disk).")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16"],
                   help="Stored dtype. float16 halves the disk and keeps "
                   "about 3 significant digits.")
    p.add_argument("--finetuned_eval_dirs", nargs="*", default=[],
                   help="eval_finetuning output dirs (absolute, or relative "
                   "to --model_dir, e.g. eval_finetuning_cv1) whose "
                   "finetuned_models/ are scored on all gene pairs.")
    p.add_argument("--finetuned_datasets", nargs="*", default=None,
                   help="Screens to export from those dirs (default: all "
                   "found), e.g. Adamson.")
    p.add_argument("--finetuned_modes", nargs="*", default=None,
                   help="Fine-tuning modes to export (default: all found).")
    p.add_argument("--ram_gb", type=float, default=32.0,
                   help="RAM for the fold-mean accumulators; a larger "
                   "request is split into passes over the folds.")
    p.add_argument("--check_pairs", type=int, default=2048,
                   help="Pairs sampled per fold to check the matrix against "
                   "model.forward.")
    p.add_argument("--no_space_check", action="store_true",
                   help="Do not refuse when the free space looks too small.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    def fail(msg: str) -> int:
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    # The device is first used after a previous export is cleared: refuse one
    # that cannot hold a tensor here (no CUDA, a bad ordinal, a typo) now.
    try:
        torch.zeros(1, device=args.device)
    except Exception as e:  # noqa: BLE001  (AssertionError or RuntimeError)
        return fail(f"--device {args.device} is not usable on this machine "
                    f"({type(e).__name__}: {e}); nothing was cleared")
    if not args.finetuned_eval_dirs and (args.finetuned_datasets
                                         or args.finetuned_modes):
        return fail("--finetuned_datasets / --finetuned_modes choose among "
                    "the models of --finetuned_eval_dirs, which was not given")

    # ---- the model dir and its complete checkpoint set (as eval_finetuning) -
    model_dir = Path(args.model_dir)
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        return fail(f"missing {config_file}")
    train_cfg = json.loads(config_file.read_text())
    num_folds = train_cfg.get("num_folds")
    if num_folds is None:
        return fail(f"{config_file} records no num_folds")
    res = model_dir / "results.json"
    if res.exists():
        try:
            fm = json.loads(res.read_text()).get("fold_metrics")
        except (OSError, ValueError):
            fm = None
        if fm is not None and len(fm) != int(num_folds):
            return fail(f"{res} has {len(fm)} fold_metrics but config.json "
                        f"num_folds={num_folds}")
    try:
        checkpoint_paths = resolve_fold_checkpoints(model_dir / "checkpoints",
                                                    num_folds)
    except ValueError as e:
        return fail(str(e))
    n_folds = len(checkpoint_paths)
    fold_sha: List[str] = []
    ckpt_inputs = []
    for cp in checkpoint_paths:
        raw = cp.read_bytes()
        fold_sha.append(hashlib.sha256(raw).hexdigest())
        ck = torch.load(io.BytesIO(raw), map_location="cpu",
                        weights_only=False)
        if ck.get("preprocessing_transform") is None:
            # eval_finetuning's all-gene refit for such a checkpoint leaks
            # test genes into the features; not offered here.
            return fail(f"{cp} stores no preprocessing_transform (a "
                        f"checkpoint from before the per-fold fit); retrain")
        ckpt_inputs.append((cp, ck.get("gene_order"),
                            ck.get("embeddings_paths"),
                            ck.get("embeddings_sha256")))
        del ck, raw

    out_dir = Path(args.out_dir) if args.out_dir else model_dir / DEFAULT_SUBDIR
    if out_dir.resolve() in (model_dir.resolve(),
                             (model_dir / "checkpoints").resolve()):
        return fail(f"--out_dir {out_dir} is the model dir itself; give it "
                    f"its own directory")
    # A rerun deletes its own files by name, genes.txt among them: never in a
    # directory that holds another tool's (the shared folds dir, say).
    not_ours = foreign_files(out_dir)
    if not_ours:
        return fail(f"--out_dir {out_dir} already holds {not_ours}, which no "
                    f"export of this script wrote; give it its own directory")

    # ---- embeddings: the files the checkpoints were trained on -------------
    print(f"Loading RAW embeddings ({len(args.embeddings_paths)} modalities)")
    raw_per_modality, gene_to_idx, idx_to_gene = load_raw_multimodal(
        args.embeddings_paths)
    problems = checkpoint_input_problems(
        ckpt_inputs, args.embeddings_paths,
        [idx_to_gene[i] for i in range(len(idx_to_gene))])
    if problems:
        return fail("--embeddings_paths are not the embedding files the "
                    "checkpoints were trained on:\n  " + "\n  ".join(problems))
    modality_labels = [Path(x).stem.replace("all_genes_", "")
                       for x in args.embeddings_paths]

    try:
        genes, rows, universe = resolve_universe(
            args.universe, args.genes_file, args.shared_genes, gene_to_idx,
            idx_to_gene)
    except ValueError as e:
        return fail(str(e))
    n = len(genes)

    # ---- what to write ------------------------------------------------------
    try:
        first = load_scorer(checkpoint_paths[0], "cpu")
    except (ValueError, NotImplementedError, KeyError) as e:
        return fail(str(e))
    H = int(first.num_heads)
    labels = head_labels(H)
    model_class = type(first).__name__
    del first
    ft_groups, ft_problems = ([], [])
    if args.finetuned_eval_dirs:
        # (two names of one directory are one dir: the name repeated, a
        # trailing slash, ./, its absolute path, a symlink. Its sets would
        # otherwise be refused as two that are written under one name.)
        eval_dirs: Dict[str, str] = {}
        for d in args.finetuned_eval_dirs:
            eval_dirs.setdefault(os.path.realpath(model_dir / d), d)
        ft_groups, ft_problems = discover_finetuned(
            model_dir, list(eval_dirs.values()),
            args.finetuned_datasets, args.finetuned_modes, checkpoint_paths,
            fold_sha)
    for msg in ft_problems:
        print(f"WARNING: fine-tuned set NOT exported: {msg}", file=sys.stderr)

    units: List[tuple] = [("head", h) for h in range(H)]
    if H > 1:
        units.append(("any", ))
    units += [("ft", gi) for gi in range(len(ft_groups))]
    per_unit = {u: (1 if u[0] == "ft" else 1 + int(args.also_logits))
                for u in units}
    n_mean_files = sum(per_unit.values())
    n_files = n_mean_files * (1 + (n_folds if args.per_fold else 0))
    item = np.dtype(args.dtype).itemsize
    need = n_files * n * n * item
    print(f"Model: {model_class}, heads {labels}"
          f"{' + ' + ANY_HEAD if H > 1 else ''}; {n_folds} folds; universe "
          f"{universe} ({n:,} genes); device {args.device}")
    print(f"Writing {n_files} matrices of {n:,} x {n:,} {args.dtype} = "
          f"{need / 1e9:.1f} GB -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # A previous export's files are removed below, so their space counts.
    old = sorted({q for pat in OWN_PATTERNS for q in out_dir.glob(pat)},
                 key=lambda q: q.name != MANIFEST)
    free = shutil.disk_usage(out_dir).free + sum(q.stat().st_size for q in old)
    if need * 1.02 > free and not args.no_space_check:
        return fail(f"{need / 1e9:.1f} GB needed but {free / 1e9:.1f} GB free "
                    f"under {out_dir}. Use --universe shared, --dtype "
                    f"float16, fewer --finetuned_* sets, or "
                    f"--no_space_check.")

    # ---- clear a previous export, manifest first ---------------------------
    (out_dir / IN_PROGRESS).write_text("export_pair_predictions.py\n")
    for q in old:
        q.unlink()
    if old:
        print(f"Removed {len(old)} file(s) of a previous export")

    (out_dir / "genes.txt").write_text("\n".join(genes) + "\n")
    try:
        from gene_name_utils import get_mapper
        mapper = get_mapper()
        symbols = [mapper.entrez_to_symbol(g) or "" for g in genes]
    except Exception as e:  # noqa: BLE001  (no HGNC table: IDs still stand)
        print(f"WARNING: gene symbols unavailable ({e}); genes.tsv has an "
              f"empty symbol column", file=sys.stderr)
        symbols = [""] * n
    with open(out_dir / "genes.tsv", "w") as fh:
        fh.write("index\tentrez_id\tsymbol\n")
        for i, (g, s) in enumerate(zip(genes, symbols)):
            fh.write(f"{i}\t{g}\t{s}\n")

    device = args.device
    rows_t = torch.as_tensor(rows, dtype=torch.long)
    pi_np, pj_np = sample_pairs(n, args.check_pairs)
    pi = torch.as_tensor(pi_np, device=device)
    pj = torch.as_tensor(pj_np, device=device)

    def stem_of(unit: tuple, kind: str) -> str:
        if unit[0] == "ft":
            return ft_groups[unit[1]]["stem"]
        lab = ANY_HEAD if unit[0] == "any" else labels[unit[1]]
        return f"sl_{kind}_{lab}"

    stats: Dict[str, Dict[str, float]] = {}
    per_fold_files: Dict[str, List[str]] = {}
    worst_err = 0.0
    ft_failed: set = set()   # ft_groups indices dropped while scoring
    passes = plan_passes(units, per_unit, n * n * 4, args.ram_gb * 1e9)
    if len(passes) > 1:
        print(f"{n_mean_files} accumulators exceed --ram_gb {args.ram_gb:g}: "
              f"{len(passes)} passes over the folds")
    t0 = time.time()
    for pno, units_now in enumerate(passes):
        acc: Dict[str, np.ndarray] = {}

        def emit(stem: str, k: int, values: torch.Tensor) -> None:
            """Add one fold's (n, n) values to the fold mean of `stem`."""
            v = values.float().cpu().numpy()
            if stem not in acc:
                acc[stem] = np.zeros((n, n), dtype=np.float32)
            acc[stem] += v
            if args.per_fold:
                name = f"fold_{k}_{stem}.npy"
                write_matrix(out_dir / name, v, args.dtype)
                per_fold_files.setdefault(stem, []).append(name)

        def emit_head(unit: tuple, k: int, L: torch.Tensor) -> None:
            """Logit matrix of one head (or the max over heads) -> its
            probability (and logit) outputs. Consumes L."""
            if args.also_logits:
                emit(stem_of(unit, "logit"), k, L)
            emit(stem_of(unit, "prob"), k, L.sigmoid_())

        want_any = ("any", ) in units_now
        heads_now = sorted({u[1] for u in units_now if u[0] == "head"}
                           | (set(range(H)) if want_any else set()))
        for k, cp in enumerate(checkpoint_paths):
            E = _load_fold_transform_and_embeddings(
                cp, raw_per_modality, None, None,
                modality_labels)[rows_t].to(device, dtype=torch.float32)
            if heads_now:
                model = load_scorer(cp, device)
                ref = forward_reference(model, E, pi, pj)
                run_max = None
                for h in heads_now:
                    L = build_any_score_matrix(model, E, only_head=h)
                    worst_err = max(worst_err, check_against_forward(
                        L, ref[:, h], pi, pj, f"fold {k} head {labels[h]}"))
                    if want_any:
                        run_max = (L.clone() if run_max is None else
                                   torch.maximum(run_max, L, out=run_max))
                    if ("head", h) in units_now:
                        emit_head(("head", h), k, L)
                    del L
                if want_any:
                    emit_head(("any", ), k, run_max)
                del model, ref, run_max
            for u in units_now:
                if u[0] != "ft" or u[1] in ft_failed:
                    continue
                g = ft_groups[u[1]]
                scorer = L = None
                try:
                    scorer = load_scorer(g["files"][k], device)
                    L = build_any_score_matrix(scorer, E,
                                               only_head=g["head_idx"])
                    err = check_against_forward(
                        L, forward_reference(scorer, E, pi,
                                             pj)[:, g["head_idx"]],
                        pi, pj, f"fold {k} {g['stem']}")
                    # a * logit + c is on the target scale (sign * raw
                    # score); dividing by the sign gives the screen's own
                    # units.
                    fo = g["folds"][k]
                    emit(g["stem"], k,
                         L.mul_(fo["a"]).add_(fo["c"]).div_(g["sign"]))
                    worst_err = max(worst_err, err)
                except Exception as e:  # noqa: BLE001
                    # A fine-tuned set is an extra: it must not cost this
                    # combo its per-cell-line matrices. Drop the set (all
                    # folds) and report it like one discover_finetuned
                    # refused: exit 1 after writing the rest. This fold's
                    # own file goes too: emit() lists a file only once
                    # write_matrix has returned, which can fail after the
                    # rename (the statistics).
                    ft_failed.add(u[1])
                    acc.pop(g["stem"], None)
                    for name in per_fold_files.pop(g["stem"], []) + [
                            f"fold_{k}_{g['stem']}{ext}"
                            for ext in (".npy", ".tmp.npy")]:
                        (out_dir / name).unlink(missing_ok=True)
                    msg = (f"{g['source']}: {g['dataset']} / {g['mode']}: "
                           f"fold {k} could not be scored "
                           f"({type(e).__name__}: {e})")
                    ft_problems.append(msg)
                    print(f"WARNING: fine-tuned set NOT exported: {msg}",
                          file=sys.stderr)
                del scorer, L
            del E
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  pass {pno + 1}/{len(passes)} fold {k}: done "
                  f"({time.time() - t0:.0f}s)")
        for stem in sorted(acc):
            mean = acc.pop(stem)
            mean /= np.float32(n_folds)
            stats[stem] = write_matrix(out_dir / f"{stem}.npy", mean,
                                       args.dtype)
            del mean

    # ---- manifest, last ----------------------------------------------------
    n_files -= len(ft_failed) * (1 + (n_folds if args.per_fold else 0))
    n_mean_files -= len(ft_failed)
    prob_labels = labels + ([ANY_HEAD] if H > 1 else [])
    sl = {
        "value": "sigmoid(logit) of the head. "
                 + ("A cell-line head is trained only on the pairs screened "
                    "in that line, so its value is on the scale of that "
                    "line's screened pairs, whose SL fraction differs widely "
                    "between lines: rank pairs within a head, do not compare "
                    "levels (or counts above a cut-off) across heads. OTHER "
                    "is a catch-all of rare lines and pairs with no recorded "
                    "line, not one cell line. "
                    f"{ANY_HEAD} is the largest of the {H} head probabilities "
                    "within each fold: the score the benchmark ranks this "
                    "model by, NOT a head trained on the any-line label and "
                    "not calibrated as P(SL in any line)."
                    if H > 1 else
                    f"{ANY_HEAD} is this model's one head, trained on the "
                    "any-line (OR) label; it has no cell-line input."),
        "files": {lab: f"sl_prob_{lab}.npy" for lab in prob_labels},
        "stats": {lab: stats[f"sl_prob_{lab}"] for lab in prob_labels},
    }
    if args.also_logits:
        sl["logit_files"] = {lab: f"sl_logit_{lab}.npy" for lab in prob_labels}
    if args.per_fold:
        sl["per_fold_files"] = {lab: per_fold_files[f"sl_prob_{lab}"]
                                for lab in prob_labels}
        if args.also_logits:
            sl["per_fold_logit_files"] = {
                lab: per_fold_files[f"sl_logit_{lab}"] for lab in prob_labels}
    finetuned = []
    for gi, g in enumerate(ft_groups):
        if gi in ft_failed:
            continue
        finetuned.append({
            "name": g["stem"],
            "file": f"{g['stem']}.npy",
            "dataset": g["dataset"],
            "column": g["column"],
            "sign": g["sign"],
            "mode": g["mode"],
            "eval_cv": g["eval_cv"],
            "head": g["head"],
            "head_idx": g["head_idx"],
            "value": f"predicted {g['dataset']} {g['column']} in the "
                     f"screen's own sign and units, (a * logit[head] + c) / "
                     f"sign; the model was fitted to sign * {g['column']}, "
                     f"the mean of the screen's rows for a gene pair",
            "note": SCREEN_VALUE_NOTES.get((g["dataset"], g["column"]), ""),
            # Which entries are held out: the screen split every fold model
            # was fine-tuned on, and the per-pair table that carries it.
            "screen_split": g["screen_split"],
            "screen_pairs": str(Path(g["source"]).parent /
                                f"{g['dataset']}_finetuned_allpairs_"
                                f"{g['mode']}.csv"),
            "held_out": "only the pairs with split == test in screen_pairs "
                        f"(or in {g['dataset']}_predictions.csv beside it); "
                        "its train and val pairs are in-sample in this "
                        "matrix, and a pair the screen did not measure is an "
                        "extrapolation that no metric of the run grades",
            "folds": g["folds"],
            "source": g["source"],
            "eval_dir": g["eval_dir"],
            "model_sha256": g["model_sha256"],
            "stats": stats[g["stem"]],
            **({"per_fold_files": per_fold_files[g["stem"]]}
               if args.per_fold else {}),
        })
    manifest = {
        "format": MANIFEST_FORMAT,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "model_class": model_class,
        "cv_type": train_cfg.get("cv_type"),
        "kernel": train_cfg.get("kernel"),
        "cell_lines": train_cfg.get("cell_lines"),
        "num_folds": n_folds,
        "fold_ids": list(range(n_folds)),
        "checkpoint_sha256": {cp.name: s
                              for cp, s in zip(checkpoint_paths, fold_sha)},
        "embeddings_paths": list(args.embeddings_paths),
        "universe": universe,
        "n_genes": n,
        "genes": "genes.txt",
        "shape": [n, n],
        "dtype": args.dtype,
        "symmetric": True,
        "diagonal": "nan",
        "aggregation": "mean over the fold models' values (probabilities / "
                       "predicted values, not logits)",
        "heads": labels,
        "sl_probability": sl,
        "finetuned": finetuned,
        "finetuned_not_exported": ft_problems,
        "forward_check": {"pairs_per_fold": int(pi.numel()),
                          "max_abs_logit_err": worst_err},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / IN_PROGRESS).unlink(missing_ok=True)
    print(f"Done in {time.time() - t0:.0f}s: {n_mean_files} fold-mean "
          f"matrices"
          f"{f' + {n_files - n_mean_files} per-fold' if args.per_fold else ''}"
          f" + {MANIFEST} under {out_dir} (forward check max |diff| "
          f"{worst_err:.2e})")
    if ft_problems:
        # (the WARNING lines are counted, not the sets: one set can give two)
        return fail(f"{len(ft_problems)} problem(s) with the requested "
                    f"fine-tuned sets (the WARNING lines above): what they "
                    f"name was not exported; everything else is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
