#!/usr/bin/env python3
"""Deterministic tests for sl_comparison/paper_tables.py.

Builds a small synthetic repo tree (folds, data pair files + sidecars, siamese
combos / single-modality / ablation dirs, comparison.csv, external-transfer
summaries, cell-line ablation rows; arm none's namespace, and on request any
other PCA arm's one directory down), runs paper_tables.main
on it and checks every output file and the values the paper cites: parameter
mean (sd) over folds, best flags, category means (all-catalog and the paper's
16) and 'beats the combo' counts, single-modality dir prefixes under a
non-default SELECTION_VARIANT, LOCO deltas (difference of means and
fold-paired), the
cell-line deltas (incl. the JSON fallback and the self-check), external bests
(DegreePrior excluded), fold geometry, M9, PCA width, parameter ratio, screen
coverage / additivity / SynLethDB agreement, stale-cell marking, empty
numbers.json items marked missing, and exit 1 plus a cleared file when a
table's source is missing. Also: rows not graded by evaluate_model.py or not
on the current folds are left out (fold provenance), every absent external
cell is listed, the paper-16 bold, the every-tier claim needing all three
references, the cell-line residual-head rows, the scored head, and the
R2 / D3 / AP inputs item. And: zero_cell_emb.py rows kept out of tab:cellline
/ CL1 unless its run is clean, ablated this combo and reproduces the included
row; combos off the current folds ignored; a comparison row that is not the
latest_combo combo (pin / parameter counts) gets no combo params; every bold
/ rank / best (tab:tiers, beats_peers, margins, tab:external, X5,
tab:external-ft, the paper-16 bold, AB2 / AB3, AB5, R5, I2) unknown with a
row it is decided over missing; the single-modality and ablation run tokens
checked separately, and an exported $ABLATION_RUN_TOKEN ignored; the
config.conf selection variant;
the README catalog line without a selection; missing means and NaN folds of
tab:solo / tab:loco listed; combo modalities vs the winners and a category
without a winner. And: every screen taken from the first split that holds it
with an absent one named, tab:solo covering SELECTION_VARIANT outside the
variant list, an ablation dir of another modality set marked stale, a partial
fold set / a REDUCED SLMGAE / a row re-graded after comparison.csv refused, an
external row of a variant outside the run kept out of the bold, a _nocell row
graded from another ablation marked stale, the rows of an ablation whose combo
was retrained in place since (same directory name) marked stale, and a
single-output
(USE_CELL_LINES=0) run reporting no cell-line or head input as missing (nor a
cell-line embedding width). And: a variant without a single screen CSV leaving
the X2 / X3 'equal across variants' claim unknown, and a comparison.csv row
whose evaluate_model JSON a rerun moved aside refused (while
--comparison_csv still tabulates it). And: a missing DegreePrior row listed
as the control; a peer whose model-CV tab:tiers row is refused dropped from
tab:external / X5; OPTIONAL_TYPES absences and not-requested tab:solo cells
listed under 'Absent by design'; the verdict (PAPER COMPLETE -> exit 0,
PAPER INCOMPLETE -> 3, a table without a source -> 1); stale fold sources
marking items.folds_meta stale. And: an unloadable arch_signature.py, no
torch, an unread paper-16 width, a peer without a parameter count, a mean
without an sd and a trained paper row without a seed listed under Missing;
a dropped peer named as dropped, not as absent from its CSV. And: code
changed, added or removed since the submission-time CODE_MANIFEST listed
under Missing (unchanged: nothing; no manifest: noted; an unreadable manifest
or code_manifest.py: missing); the paper's bold (tab:tiers, tab:solo paper-16,
tab:external) on every row tied at the printed 3 decimals, the other flags and
ranks exact. And the PCA arms (2026-10-20): the arm helpers equal
siamese_sl/slurm/config.conf's and the CLI / environment resolution refuses
what config.conf refuses; an arm X reads only its own namespace
(siamese_sl/results/pca_X/, sl_comparison/results/pca_X/, its owner's
single-embedding runs) plus the shared models' rows, writes only its own
paper_tables/, and arm none's files never change its tables (nor X's arm
none's); a SLeuth row of the shared directory never stands in for an arm's
and a shared model's row in an arm's directory is not read; the "JSON beside
it" rule names the row's own directory (and holds for the arm's paths as
run_paper_tables.sh spells them); the single-embedding root from the
selection's step1_root (one of the pair other than the owner rule's only
noted), else config.conf's owner rule; a combo of another arm, outside the
namespace (a combo linked back from other storage is inside), a selection of
another arm and a comparison.csv of another arm listed and not used; the
cell-line source compared as a path, its rerun hint naming the arm on both
commands; the labels (README line 1, the arm bullet, numbers.json run.pca_*
and step1_pca, the leading pca_arm column of every CSV), the PCA targets and
arms from the environment else config.conf, and combos trained off the
targets listed. And the cross-arm tab_pca (--pca_arms_table): values on
every arm's graded rows, paired deltas, their se over three folds and their
reading, best_arm (exact ties, not ties at 3 decimals), the 2x2 effects with
four arms and absent by design with three, cells of another run's token,
signature, embedding content, another graded run or without a combo or a
selection marked stale / missing, each arm judged with its own caches, the
modalities and selection checks, each arm's own verdict read from its
README, the verdict and exit status, and fewer than two arms refused. Every
run pins the PCA arm, so an exported PCA_ARM cannot change a test.

    python3 sl_comparison/test_paper_tables.py
    # or: python -m pytest sl_comparison/test_paper_tables.py -q
"""
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # the native-width item is then only checked as absent
    torch = None

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("paper_tables",
                                               HERE / "paper_tables.py")
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)
_mspec = importlib.util.spec_from_file_location("_mp", HERE / "model_params.py")
mp = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(mp)
# What the fixture's configs are checked against (pca_arm, pca_settings).
arch = pt._load_module("_t_arch_signature",
                       HERE.parent / "siamese_sl" / "arch_signature.py")

VARIANTS = list(pt.ALL_VARIANTS)
TOKEN = "TOK"
SEED = 42
GENES = [str(101 + i) for i in range(12)]
CATALOG = [("ea", "expression"), ("eb", "expression"), ("pa", "ppi"),
           ("ka", "kg"), ("kb", "kg")]  # kb: never generated
# Native width of each generated data/all_genes_<e>.pt (AP1 / AP2).
WIDTH = {"ea": 7, "eb": 5, "pa": 6, "ka": 8}
WINNERS = {"expression": "ea", "ppi": "pa", "kg": "ka"}
POS = [("101", "102"), ("102", "103"), ("103", "104"), ("104", "105"),
       ("105", "106"), ("101", "103"), ("102", "107"), ("109", "110")]
NEG = [("101", "104"), ("101", "105"), ("102", "104"), ("102", "105"),
       ("103", "105"), ("103", "106"), ("104", "106"), ("101", "106"),
       ("102", "106"), ("107", "108"), ("103", "108"), ("109", "111"),
       ("101", "102")]  # conflict: also SL
POS_LINES = {("101", "102"): "K562;A549", ("104", "105"): "TBD",
             ("109", "110"): "K562;JURKAT"}
NEG_LINES = {("101", "102"): "JURKAT;HeLa;A375", ("101", "104"): "A549;293T"}

# (auroc, ap, auprg) per model per cv; sd 0.02 unless overridden.
CMP = {
    "siamese_sl": {"cv1": (.96, .80, .951), "cv2": (.90, .70, .871),
                   "cv3": (.677, .40, .732)},
    "siamese_sl_pd_pooled": {"cv1": (.95, .79, .94), "cv2": (.89, .69, .86),
                             "cv3": (.67, .39, .72)},
    "siamese_sl_bilinear_per_head": {"cv1": (.955, .81, .95),
                                     "cv2": (.905, .71, .88),
                                     "cv3": (.68, .41, .74)},
    "siamese_sl_bilinear_pooled": {"cv1": (.94, .78, .93),
                                   "cv2": (.88, .68, .85),
                                   "cv3": (.66, .38, .71)},
    "SLMGAE": {"cv1": (.98, .85, .97), "cv2": (.85, .60, .80),
               "cv3": (.705, .35, .70)},
    "SLGNN": {"cv1": (.90, .70, .90), "cv2": (.80, .55, .75),
              "cv3": (.60, .30, .60)},
    "NSF4SL": {"cv1": (.85, .65, .85), "cv2": (.75, .50, .70),
               "cv3": (.55, .25, .55)},
    # ELISL (ELISL/slurm/run_elisl.sh): a published peer of tab_tiers
    # (pt.TIER_PEERS), below the best peer in every column.
    "ELISL": {"cv1": (.84, .64, .84), "cv2": (.74, .49, .69),
              "cv3": (.54, .24, .54)},
    "DegreePrior": {"cv1": (.99, .90, .99), "cv2": (.60, .30, .40),
                    "cv3": (.50, .20, .0)},
    # The bilinear probe (siamese_sl/slurm/run_probe.sh): a shared trained
    # baseline, below every SLeuth variant and the best peer in each column.
    "BilinearProbe": {"cv1": (.93, .74, .92), "cv2": (.83, .57, .78),
                      "cv3": (.62, .31, .64)},
    "siamese_sl_nocell": {"cv1": (.95, .78, .94), "cv2": (.85, .60, .80),
                          "cv3": (.60, .30, .65)},
    "siamese_sl_bilinear_per_head_nocell": {"cv1": (.95, .80, .945),
                                            "cv2": (.86, .62, .81)},
}
NOCELL_BILINEAR_CV3 = (.61, .31, .66)  # JSON only (not in comparison.csv)
PARAMS = {"SLMGAE": 253547952, "SLGNN": 924928, "NSF4SL": 527104,
          "ELISL": 1981203, "BilinearProbe": 149378}
FOLD_PARAMS = {"pd:per_head": {"cv1": [29000, 29044], "cv2": [28000, 28340],
                               "cv3": [29100, 29100]},
               "pd:pooled": [28000, 28000],
               "bilinear:per_head": [29256, 29300],
               "bilinear:pooled": [28256, 28256]}
INPUT_DIM = {"cv1": [1505, 1510], "cv2": [1520, 1585], "cv3": [1530, 1540]}
COMBO_SELF = {"cv1": .95, "cv2": .87, "cv3": .73}
SOLO = {"cv1": {"ea": .90, "eb": .93, "pa": .96, "ka": .80},
        "cv2": {"ea": .80, "eb": .88, "pa": .90, "ka": .70},
        "cv3": {"ea": .70, "eb": .75, "pa": .74, "ka": .60}}
LOCO = {"cv1": {"expression": .94, "ppi": .90, "kg": .95},
        "cv2": {"expression": .86, "ppi": .80, "kg": .87},
        "cv3": {"expression": .738, "ppi": .60, "kg": .736}}
# The PCA arms (siamese_sl/slurm/config.conf), in their fixed order.
ARMS = ("none", "joint", "per_modality", "both")
# What train.py records in config.json per arm at the default targets (0.8 /
# 0.8): a multi-modal run's (the combo, its ablations; config.conf
# pca_multi_args) and a single-embedding run's (pca_single_args: native for
# none / joint, --pca_variance for per_modality / both). train.py expands
# --pca_variance into pca_dims, one per file (_cfg).
ARM_CFG = {"none": {"post_pca": None, "post_pca_variance": None,
                    "no_post_pca": True},
           "joint": {"post_pca": 0.8, "post_pca_variance": 0.8},
           "per_modality": {"pca_variance": 0.8, "post_pca": None,
                            "post_pca_variance": None, "no_post_pca": True},
           "both": {"pca_variance": 0.8, "post_pca": 0.8,
                    "post_pca_variance": 0.8}}
SOLO_CFG = {"none": ARM_CFG["none"], "joint": ARM_CFG["none"],
            "per_modality": ARM_CFG["per_modality"],
            "both": ARM_CFG["per_modality"]}
# Each arm's SLeuth rows: CMP's values plus ARM_SHIFT (so a value read from
# another arm's namespace is told apart), graded per fold at mean -/+
# FOLD_SPREAD (fold 0 / fold 1). Against arm none, joint is then no
# measurable difference (|-.003| <= .005), per_modality better (.02 > 2 SE,
# SE = |.013 - .01|) and both within fold noise (.01 <= 2 SE, SE = .01).
ARM_SHIFT = {"none": 0.0, "joint": -.003, "per_modality": .02, "both": .01}
FOLD_SPREAD = {"none": .01, "joint": .012, "per_modality": .013, "both": .02}
# Each arm's combos: the encoder input width and the trainable parameters
# per fold are arm none's (INPUT_DIM, FOLD_PARAMS) minus these.
ARM_WIDTH_CUT = {"none": 0, "joint": 1000, "per_modality": 500, "both": 1200}
ARM_PARAMS_CUT = {"none": 0, "joint": 8000, "per_modality": 5000,
                  "both": 9000}
# The combos' modality list (WINNERS order).
WINNER_FILES = [f"../data/all_genes_{t}.pt" for t in WINNERS.values()]
# What paper_tables reads from the environment for the arms: cleared (or set)
# around every run of a test, so an exported value cannot reach one.
PCA_ENV = ("PCA_ARM", "PCA_ARMS", "PCA_MODALITY_VARIANCE",
           "PCA_JOINT_VARIANCE", "PCA_JOINT_DIM")
# A run's arm pinned as the pipeline's default run passes it (arm none of all
# four), for the tests that call pt.main directly.
PIN_ARM_NONE = ("--pca_arm", "none", "--pca_arms", *ARMS)
SCREEN_A = {"101": 1, "102": 2, "103": 3, "107": 4, "108": 5, "111": 6,
            "901": 7, "902": 8}
# What zero_cell_emb.py records for a clean CV (the fixture folds are two).
CELLLINE_OK = {"status": "ok", "n_folds": 2, "self_check": {"ok": True}}
# WHICH model the peers ran, as the trainers record it in model_params.json
# and compare.py copies it into comparison.csv: the AUTHORS' configuration
# (run_shared_models.sh passes no --arch, so SLGNN defaults to "original";
# NSF4SL defaults to --features transe with the current loop). row_usable
# drops any other, so a substitute never sits in tab:tiers under their name.
MODEL_CONFIG = {"SLGNN": {"arch": "original"},
                "NSF4SL": {"features": "transe", "legacy_loop": False},
                "ELISL": {"elisl_sources": "Sequence_1024|PPI_ec"}}
# paper_tables' exit status when every table is built but something is listed
# under Missing (PAPER INCOMPLETE): 3, not 2, which argparse and Python's
# can't-open-file already use. The fixture tree always is: its catalog holds
# none of the paper's 16, kb (a required type) is never generated and some
# dirs are stale on purpose. _complete_tree builds the complete case.
INCOMPLETE = 3
# What the complete tree still lists without torch (AP1 / AP2 unread).
NO_TORCH = "torch not importable: items.inputs.native_dims not read"
NOTORCH = [] if torch is not None else [NO_TORCH]
# run_generate_embeddings.conf's OPTIONAL_TYPES in the fixture: kb is not
# among them, so its absence is missing, not absent by design.
OPTIONAL = ("esmc", "genept")


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _wj(path: Path, obj) -> None:
    _w(path, json.dumps(obj, indent=1))


def _pairs(*ps):
    return np.array(ps, dtype=np.int32).reshape(-1, 2)


def _folds(root: Path) -> Path:
    folds = root / "sl_comparison" / "folds"
    _w(folds / "genes.txt", "\n".join(GENES) + "\n")
    np.save(folds / "all_pos.npy", _pairs([0, 1]))
    _wj(folds / "meta.json", {
        "seed": 42, "num_folds": 2, "num_nodes": 12, "num_positives": 8,
        "num_negatives": 12, "val_frac": 0.1, "conflict_policy": "or_collapse",
        "cv_types": list(pt.CVS), "pos_path": "data/pos.txt",
        "neg_path": "data/neg.txt", "num_conflicts": 1,
        "split_digest": {"cv1": "x"}, "source_sha256": {"pos": "p",
                                                        "neg": "n"}})
    cv1 = {"train_pos": _pairs([0, 1], [1, 2], [2, 3], [3, 4], [4, 5]),
           "val_pos": _pairs([5, 6]), "test_pos": _pairs([0, 2], [1, 6]),
           "train_neg": _pairs([0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [2, 5],
                               [3, 5], [0, 5]),
           "val_neg": _pairs([6, 7]),
           "test_neg": _pairs([1, 5], [7, 8], [2, 7])}
    cv3 = [{"train_pos": _pairs([0, 1], [1, 2], [2, 3]),
            "val_pos": _pairs([4, 5]), "test_pos": _pairs([8, 9]),
            "train_neg": _pairs([0, 2], [0, 3], [1, 3], [3, 1]),
            "val_neg": _pairs([4, 6]), "test_neg": _pairs([8, 10], [9, 10])},
           {"train_pos": _pairs([0, 1], [1, 2], [2, 3]),
            "val_pos": _pairs([4, 5]), "test_pos": _pairs([8, 9], [10, 11]),
            "train_neg": _pairs([0, 2], [0, 3], [1, 3], [3, 1]),
            "val_neg": _pairs([4, 6]), "test_neg": _pairs([8, 10], [9, 11])}]
    for cv in pt.CVS:
        for k in range(2):
            d = folds / cv / f"fold_{k}"
            d.mkdir(parents=True)
            for name, arr in (cv3[k] if cv == "cv3" else cv1).items():
                np.save(d / f"{name}.npy", arr)
    return folds


def _data(root: Path) -> None:
    hdr = "entrez1\tentrez2\tsources\tcell_lines\tpubmed_ids\n"
    for name, pairs, lines, default in (("pos", POS, POS_LINES, "K562"),
                                        ("neg", NEG, NEG_LINES, "K562")):
        _w(root / "data" / f"{name}.txt",
           "".join(f"{a}\t{b}\n" for a, b in pairs))
        _w(root / "data" / f"{name}.sources.tsv", hdr + "".join(
            f"{a}\t{b}\tsrc\t{lines.get((a, b), default)}\t1\n"
            for a, b in pairs))
    # A readable width (AP1 / AP2) for every generated embedding: a paper-16
    # one without is listed as missing (0-byte placeholders without torch,
    # whose absence is then listed instead).
    for etype, _ in CATALOG:
        if etype == "kb":
            continue
        p = root / "data" / f"all_genes_{etype}.pt"
        if torch is None:
            _w(p, "")
        else:
            torch.save({"embeddings": torch.zeros(3, WIDTH[etype]),
                        "gene_order": GENES[:3]}, p)
    _wj(root / "data" / "synlethkg" / "meta.json", {
        "n_entities": 50, "n_relations": 4, "n_triples": 99,
        "relations": ["PARTICIPATES_GpCC", "CAUSES_CcSE", "INCLUDES_PCiC",
                      "BINDS_CbG"],
        "removed_label_relations": ["SL_GsG", "SR_GsrG", "NONSL_GnsG"]})


def _confs(root: Path) -> None:
    slurm = root / "siamese_sl" / "slurm"
    body = "\n".join(f'    "{e}:all_genes_{e}.pt:{c}"' for e, c in CATALOG)
    _w(slurm / "run_best_per_category.conf",
       "# catalog (comments hold parentheses: (like this))\n"
       f"EMB_CATALOG=(\n    # Expression (two)\n{body}\n)\n")
    _w(slurm / "run_ablation.conf",
       "ABLATE_CATEGORIES=(expression ppi kg)  # (three)\n")
    _optional_types(root, *OPTIONAL)


def _optional_types(root: Path, *types) -> None:
    """run_generate_embeddings.conf with OPTIONAL_TYPES = types (the array
    line as the real conf writes it, after an array with comments)."""
    _w(root / "siamese_sl" / "slurm" / "run_generate_embeddings.conf",
       'PRECOMPUTED_TYPES=(\n    "ea"  # (expression)\n    # "esmc"  # (off)\n'
       ')\n# OPTIONAL_TYPES (absence accepted)\nOPTIONAL_TYPES=(' +
       " ".join(f'"{t}"' for t in types) + ")\n")


def _arm_rel(arm: str) -> str:
    """The arm's siamese results root relative to siamese_sl/ (config.conf
    pca_arm_results, written out here rather than taken from paper_tables)."""
    return "results" if arm == "none" else f"results/pca_{arm}"


def _sia(root: Path, arm: str = "none") -> Path:
    return root / "siamese_sl" / _arm_rel(arm)


def _cmp(root: Path, arm: str = "none") -> Path:
    """The arm's sl_comparison root (config.conf pca_arm_cmp)."""
    d = root / "sl_comparison" / "results"
    return d if arm == "none" else d / f"pca_{arm}"


def _owner(arm: str, arms=ARMS) -> str:
    """config.conf pca_step1_arm: the arm whose root holds arm's
    single-embedding runs (the first of its pair that arms lists)."""
    pair = (("per_modality", "both") if arm in ("per_modality", "both")
            else ("none", "joint"))
    return next((a for a in pair if a in arms), arm)


def _step1_label(arm: str, arms=ARMS) -> str:
    """config.conf pca_step1_label at the default targets, as the combo job
    records it in _selection_meta.step1_pca: which arms of the run (arms)
    read arm's single-embedding runs."""
    pair = (("per_modality", "both") if arm in ("per_modality", "both")
            else ("none", "joint"))
    who = [a for a in pair if a in arms or a == arm]
    who = (f"shared by arms {who[0]} and {who[1]}" if len(who) == 2 else
           f"used by arm {who[0]} only")
    return ("single-embedding runs with PCA within the modality (variance "
            f"0.8), {who}" if arm in ("per_modality", "both") else
            f"single-embedding runs at native dimension (no PCA), {who}")


def _cfg(v: str, cv: str, solo: bool = False, arm: str = "none",
         **over) -> dict:
    """A run's config.json under PCA arm `arm` (ARM_CFG; SOLO_CFG for a
    single-embedding run of the arm's pair)."""
    k, c = v.split(":")
    cfg = {"use_cell_lines": True, "kernel": k, "cell_lines": c,
           "cv_type": cv, "cell_line_dim": 24, "encoder_dims": [16] * 7,
           "siamese_encoder_type": "residual", "dropout": 0.2,
           "pd_epsilon": 0.001, "seed": 42, "layernorm": False,
           "log_temperature": False, "per_head_last_layer_bias": True,
           "select_metric": "auprg_any",
           "embeddings_paths": ["../data/all_genes_ea.pt"]}
    cfg.update((SOLO_CFG if solo else ARM_CFG)[arm])
    cfg.update(over)
    if cfg.get("pca_variance") is not None and "pca_dims" not in over:
        cfg["pca_dims"] = [cfg["pca_variance"]] * len(cfg["embeddings_paths"])
    return cfg


def _siamese_dir(d: Path, cfg: dict, cv: str, folds: Path, auprg: float,
                 fold_metrics, token=TOKEN, stamp=True) -> None:
    _wj(d / "config.json", cfg)
    _wj(d / "results.json", {"cv_type": cv, "config": cfg,
                             "fold_metrics": fold_metrics,
                             "summary": {"auprg_any_mean": auprg,
                                         "auprg_any_std": 0.01}})
    if stamp:
        _wj(d / "model_params.json",
            {"trainable_params": 1, "fold_stamp": mp.fold_stamp(folds, cv)})
    if token is not None:
        _w(d / "run_token.txt", token + "\n")


def _selected_marker(root: Path, v: str, cv: str, text: str,
                     arm: str = "none") -> None:
    """collect_siamese.py's selected-run marker (repo-root relative), in the
    arm's sl_comparison root."""
    stem = "siamese" if v == "pd:per_head" else pt.variant_row(v)
    _w(_cmp(root, arm) / f"{stem}_selected_{cv}.txt", text + "\n")


def _minus(winner_files, cat: str):
    """The combo's modality list without one category's winner."""
    return [p for p in winner_files
            if Path(p).name != f"all_genes_{WINNERS[cat]}.pt"]


def _siamese(root: Path, folds: Path, arm: str = "none", arms=ARMS,
             token: str = TOKEN) -> None:
    """PCA arm `arm`'s siamese namespace (siamese_sl/results[/pca_<arm>]/):
    markers whose text is that root, its combos, ablations, selection
    (_selection_meta pca_arm / step1_root / run_token=token), cell-line
    ablations and cv3 evals; the single-embedding runs only when the arm owns
    its pair's (_owner over `arms`), in its root."""
    res = _sia(root, arm)
    rel = _arm_rel(arm)
    winner_files = WINNER_FILES
    for v in VARIANTS:
        pre = pt.variant_prefix(v)
        for cv in pt.CVS:
            name = f"{rel}/{pre}best_cat_ea_ka_pa_{cv}"
            _w(res / f"{pre}latest_combo_{cv}.txt", name + "\n")
            _selected_marker(root, v, cv, f"siamese_sl/{name}", arm)
            fp = FOLD_PARAMS[v]
            fp = fp[cv] if isinstance(fp, dict) else fp
            fm = [{"fold": k, "total_params": p - ARM_PARAMS_CUT[arm],
                   "input_dim": d - ARM_WIDTH_CUT[arm],
                   "auprg_any": COMBO_SELF[cv] + dx}
                  for k, (p, d, dx) in enumerate(zip(fp, INPUT_DIM[cv],
                                                     (-.02, .02)))]
            # Stamped on these folds, as train.py stamps its output dir.
            _siamese_dir(root / "siamese_sl" / name,
                         _cfg(v, cv, arm=arm, embeddings_paths=winner_files),
                         cv, folds, COMBO_SELF[cv], fm, token=None)
    sel = "pd:per_head"
    owner = _owner(arm, arms)
    for cv in pt.CVS:
        for etype, _ in CATALOG:
            if etype == "kb" or owner != arm:
                continue
            over = {"dropout": 0.5} if (etype, cv) == ("pa", "cv3") else {}
            tok = "OLD" if (etype, cv) == ("eb", "cv2") else TOKEN
            _siamese_dir(res / f"pcavar0.8_{etype}_{cv}",
                         _cfg(sel, cv, solo=True, arm=arm, **over), cv, folds,
                         SOLO[cv][etype],
                         [{"fold": k, "auprg_any": SOLO[cv][etype]}
                          for k in range(2)], token=tok)
        for cat, val in LOCO[cv].items():
            # Listed fold 1 first: pairing must go by 'fold', not position.
            # The combo's modalities minus this category's winner, as
            # run_ablation.sh trains them.
            _siamese_dir(res / f"ablate_no_{cat}_{cv}",
                         _cfg(sel, cv, arm=arm,
                              embeddings_paths=_minus(winner_files, cat)),
                         cv, folds, val, [{"fold": 1, "auprg_any": val + .01},
                                          {"fold": 0, "auprg_any": val - .01}],
                         token=token)
    _siamese_dir(res / "var_bilinear_pooled__ablate_no_expression_cv3",
                 _cfg("bilinear:pooled", "cv3", arm=arm,
                      embeddings_paths=_minus(winner_files, "expression")),
                 "cv3", folds, .70, [{}, {}], token=token)
    _wj(res / "best_per_category.json", {
        **{cv: {cat: {"type": t, "file": f"all_genes_{t}.pt", "aupr": .9}
                for cat, t in WINNERS.items()} for cv in pt.CVS},
        "_selection_meta": {"selection_cv": "cv1",
                            "selection_variant": sel,
                            "selection_metric": "val_auprg_any_mean",
                            "run_token": token, "missing_categories": [],
                            "pca_arm": arm,
                            "step1_root": _arm_rel(owner),
                            "step1_pca": _step1_label(arm, arms)}})
    shift = ARM_SHIFT[arm]
    agg = {m: [x + shift, 0.01]
           for m, x in zip(pt.METRICS, CMP["siamese_sl"]["cv1"])}
    hpb = {m: [x + shift - .1, 0.02]
           for m, x in zip(pt.METRICS, CMP["siamese_sl"]["cv1"])}
    # zero_cell_emb.py records the --model_dir it was given: the marker text.
    _wj(res / "cellline_ablation_cv1" / "results.json",
        {**CELLLINE_OK, "source_model_dir": f"{rel}/best_cat_ea_ka_pa_cv1",
         "rows": ["as-trained", "cell_emb=0", "cell_emb=0 +head_proj_bias=0"],
         "residual_head_terms": ["head_proj_bias (per-head c_h)"],
         "agg": {"as-trained": agg, "cell_emb=0": agg,
                 "cell_emb=0 +head_proj_bias=0": hpb}})
    # --json_out's layout (keyed by the CV); its as-trained row is not the
    # included row (another combo was ablated).
    _wj(res / "cellline_ablation_cv2" / "results.json",
        {"cv2": {**CELLLINE_OK,
                 "agg": {"as-trained": {"auroc": [.90, .01], "ap": [.70, .01],
                                        "auprg": [.80, .01]}}}})
    # External screens, identical rows for two variants' cv3 combos.
    rng = np.random.RandomState(0)
    genes = sorted(SCREEN_A)
    pairs = [(a, b) for i, a in enumerate(genes) for b in genes[i + 1:]]
    known = {pt.canon(*p) for p in POS + NEG}
    corn = rng.randn(len(pairs))
    for v in VARIANTS:
        heads = ({"Adamson": "K562", "AdamsonGI": "K562", "Corn": "OTHER",
                  "Gilbert": "K562"}
                 if v.endswith(":per_head") else
                 dict.fromkeys(pt.SCREENS, "ANY (pooled)"))
        for split in pt.SPLITS:
            _wj(res / f"{pt.variant_prefix(v)}best_cat_ea_ka_pa_cv3" /
                f"eval_finetuning_{split}" / "config_used.json",
                {"sldb_overlap_sources": ["data/pos.txt", "data/neg.txt"],
                 "effective_head": heads})
    # One eval task per variant, as eval_finetuning.sh submits them.
    for v in VARIANTS:
        ed = (res / f"{pt.variant_prefix(v)}best_cat_ea_ka_pa_cv3" /
              "eval_finetuning_cv1")
        for screen, target in (
                ("Adamson", [SCREEN_A[a] + SCREEN_A[b] for a, b in pairs]),
                ("AdamsonGI", [SCREEN_A[a] - SCREEN_A[b] for a, b in pairs]),
                ("Corn", list(corn)),
                ("Gilbert", [-(SCREEN_A[a] + SCREEN_A[b]) for a, b in pairs])):
            lines = ["gene1_entrez,gene2_entrez,outcome_raw,target,"
                     "sldb_overlap,split,logit_fold0,logit_fold1"]
            for (a, b), t in zip(pairs, target):
                lines.append(f"{a},{b},{-t},{t},{pt.canon(a, b) in known},"
                             f"test,0.1,0.2")
            _w(ed / f"{screen}_predictions.csv", "\n".join(lines) + "\n")


def _provenance(folds: Path, cv: str) -> dict:
    """evaluate_model.fold_provenance's fingerprint keys for these folds."""
    st = mp.fold_stamp(folds, cv)
    return {f"{n}_sha256": st[n] for n in mp.STAMP_FILES}


def _row_json(path: Path, model: str, cv: str, vals, folds: Path,
              trained=True, spread: float = .01) -> None:
    """A build_summary-shaped row (provenance, fold_stamp unless trained is
    False, the per-fold records it averages: fold 0 / 1 at mean -/+ spread),
    carrying the peers' model-config record."""
    s = {"model": model, "cv_type": cv, "provenance": _provenance(folds, cv)}
    s.update(MODEL_CONFIG.get(model, {}))
    if trained:
        s["fold_stamp"] = mp.fold_stamp(folds, cv)
    for m, x in zip(pt.METRICS, vals):
        s[f"{m}_mean"], s[f"{m}_std"] = x, .02
    s["folds"] = [{m: x + d for m, x in zip(pt.METRICS, vals)}
                  for d in (-spread, spread)]
    _wj(path, {"summary": s})


def _comparison(root: Path, arm: str = "none") -> None:
    """The arm's comparison.csv (in its sl_comparison root; arm X's with
    compare.py --pca_arm's trailing pca_arm column), its SLeuth rows' JSONs
    beside it (values + ARM_SHIFT[arm]) and, for arm none only, the shared
    models' rows in sl_comparison/results/."""
    cmp_dir = _cmp(root, arm)
    shared = _cmp(root)
    folds = root / "sl_comparison" / "folds"
    tail = [] if arm == "none" else [arm]
    rows = [["cv", "model", "metric", "mean", "std"] + (
        ["pca_arm"] if tail else [])]

    def row(*cells):
        rows.append(list(cells) + tail)

    for model, per in CMP.items():
        sleuth = model.startswith("siamese_sl")
        for cv, vals in per.items():
            if sleuth:
                vals = tuple(x + ARM_SHIFT[arm] for x in vals)
            for m, x in zip(pt.METRICS, vals):
                sd = .03 if (model, cv, m) == ("SLMGAE", "cv3", "auroc") \
                    else .02
                row(cv, model, m, x, sd)
            row(cv, model, "baseline", .172 if cv == "cv1" else .2, 0)
            if model in PARAMS:
                row(cv, model, "trainable_params", PARAMS[model], "")
            elif sleuth and not model.endswith("_nocell"):
                # compare.py's siamese copy: the last fold's count only
                row(cv, model, "trainable_params", 29044, "")
            # As compare.py writes them: graded-on for every row, trained-on
            # wherever model_params.json has a fold_stamp (not DegreePrior).
            st = mp.fold_stamp(folds, cv)
            for n in mp.STAMP_FILES:
                row(cv, model, f"folds_{n}_sha256", st[n], "")
                if model != "DegreePrior":
                    row(cv, model, f"trained_on_{n}_sha256", st[n], "")
            if model in PARAMS or (sleuth and not model.endswith("_nocell")):
                row(cv, model, "seed", SEED, "")
            # WHICH model ran, as compare.py archives it.
            for key, val in MODEL_CONFIG.get(model, {}).items():
                row(cv, model, key, val, "")
            # compare.py writes the table FROM these, so a real run always has
            # one beside every row (a rerun that moved its row aside and then
            # failed is the state paper_tables must refuse). The shared
            # models' rows exist once, in sl_comparison/results/.
            if sleuth:
                _row_json(cmp_dir / f"{model}_{cv}.json", model, cv, vals,
                          folds, spread=FOLD_SPREAD[arm])
            elif arm == "none":
                _row_json(shared / f"{model}_{cv}.json", model, cv, vals,
                          folds, trained=(model != "DegreePrior"))
    cmp_dir.mkdir(parents=True, exist_ok=True)
    with open(cmp_dir / "comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    _row_json(cmp_dir / "siamese_sl_bilinear_per_head_nocell_cv3.json",
              "siamese_sl_bilinear_per_head_nocell", "cv3",
              tuple(x + ARM_SHIFT[arm] for x in NOCELL_BILINEAR_CV3), folds)


def _external(root: Path, arm: str = "none") -> None:
    ext = _cmp(root, arm) / "external_transfer"
    ext.mkdir(parents=True, exist_ok=True)
    zs = [["model", "kind", "screen", "split", "row_set", "n_rows", "n_folds",
           "n_folds_finite", "spearman_mean", "spearman_sd_pop",
           "screen_role"]]
    ft = [["model", "screen", "split", "row_set", "method", "n_rows",
           "n_folds", "n_folds_finite", "spearman_mean", "spearman_sd_pop"]]
    sleuth = {"siamese_sl": .10, "siamese_sl_pd_pooled": .05,
              "siamese_sl_bilinear_per_head": .02,
              "siamese_sl_bilinear_pooled": .01}
    for s in pt.SCREENS:
        role = ROLE.get(s, "external")
        for sp in pt.SPLITS:
            for m, x in sleuth.items():
                if s == "Gilbert" and m == "siamese_sl":
                    x = .5
                x += ARM_SHIFT[arm]
                for rs, dx in (("matched", 0), ("all_sldb_filtered", .02),
                               ("all_unfiltered", .5),
                               ("matched_incl_sldb", .04),
                               ("matched_in_sldb", .06)):
                    zs.append([m, "siamese", s, sp, rs, 10, 2, 2, x + dx, .01,
                               role])
                for rs in ("matched", "all_sldb_filtered", "all_unfiltered"):
                    for meth, y in (("zero_shot_raw", -.2),
                                    ("zero_shot_ols", .2), ("LP_last", .3),
                                    ("Full", .97)):
                        ft.append([m, s, sp, rs, meth, 10, 2, 2, y, .01])
            for m, x in (("SLMGAE", .2), ("SLGNN", .15), ("NSF4SL", -.1)):
                for rs, dx in (("matched", 0), ("matched_incl_sldb", .04),
                               ("matched_in_sldb", .06)):
                    zs.append([m, "baseline", s, sp, rs, 10, 2, 2, x + dx,
                               .01, role])
            for rs in ("matched", "matched_incl_sldb", "matched_in_sldb"):
                zs.append(["DegreePrior", "degree_prior", s, sp, rs, 10, 2,
                           2, .9, .01, role])
            zs.append(["SynLethDB_label", "sldb_label", s, sp,
                       "matched_in_sldb", 4, 2, 2, .2, 0.0, role])
    with open(ext / "zero_shot_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(zs)
    with open(ext / "finetune_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(ft)
    ag = [["screen", "scope", "split", "n_labelled", "n_pos", "n_neg",
           "auroc_target_vs_label", "spearman_target_vs_label",
           "screen_role"]]
    for s in pt.SCREENS:
        for scope, sp in [("all_pairs", "any")] + [("test_genes_txt", x)
                                                   for x in pt.SPLITS]:
            ag.append([s, scope, sp, 10, 4, 6, AGREE_AUROC, .3,
                       ROLE.get(s, "external")])
    with open(ext / "sldb_agreement.csv", "w", newline="") as f:
        csv.writer(f).writerows(ag)


# external_transfer.py's screen_role in the fixture, and the agreement value.
ROLE = {"Gilbert": "in_distribution_check"}
AGREE_AUROC = .7


def _stamp_fold_sources(root: Path, mangle=None) -> None:
    """Record the REAL source_sha256 of the pair files _data just wrote.

    Ctx now runs prepare_folds.check_sources (a standalone run_paper_tables.sh
    must not publish meta.json's conflict attribution beside head statistics
    recomputed with a newer cell_line_vocab.py), so a placeholder would make
    every test tree read as stale folds. It must run BEFORE the rows are
    written: fold_stamp hashes meta.json, so a later edit invalidates them.
    `mangle` rewrites the dict for the stale-folds cases.
    """
    pf = pt._load_module("_t_prepare_folds", HERE / "prepare_folds.py")
    meta_p = root / "sl_comparison" / "folds" / "meta.json"
    meta = json.loads(meta_p.read_text())
    rec = pf.source_fingerprints(root / meta["pos_path"],
                                 root / meta["neg_path"], sidecars=True)
    meta["source_sha256"] = mangle(rec) if mangle else rec
    _wj(meta_p, meta)


def _build(root: Path, mangle_sources=None) -> None:
    """The shared inputs (folds, data, confs, the shared models' rows) and
    PCA arm none's namespace."""
    folds = _folds(root)
    _data(root)
    _stamp_fold_sources(root, mangle_sources)
    _confs(root)
    _siamese(root, folds)
    _comparison(root)
    _stamp_probe(root)
    _external(root)


def _stamp_probe(root: Path) -> None:
    """run_probe.sh's probe_source on each BilinearProbe row: the arm-none
    pd:pooled combo it was built on (paper_tables refuses a row built on
    another one)."""
    import hashlib
    sia = root / "siamese_sl"
    for cv in pt.CVS:
        q = root / "sl_comparison" / "results" / f"BilinearProbe_{cv}.json"
        m = sia / "results" / f"var_pd_pooled__latest_combo_{cv}.txt"
        if not (q.exists() and m.exists()):
            continue
        combo = m.read_text().strip()
        o = json.loads(q.read_text())
        o["summary"]["probe_source"] = {
            "variant": "pd:pooled", "pca_arm": "none", "marker": str(m),
            "combo_dir": combo,
            "combo_config_sha256": hashlib.sha256(
                (sia / combo / "config.json").read_bytes()).hexdigest(),
            "combo_results_sha256": hashlib.sha256(
                (sia / combo / "results.json").read_bytes()).hexdigest()}
        q.write_text(json.dumps(o))


def _build_arm(root: Path, arm: str, arms=ARMS, token: str = TOKEN) -> None:
    """PCA arm `arm`'s namespaces on a _build tree (its single-embedding runs
    too when it owns its pair's among `arms`)."""
    _siamese(root, root / "sl_comparison" / "folds", arm, arms, token)
    _comparison(root, arm)
    _external(root, arm)


def _arm_readme(root: Path, arm: str, verdict: str = None,
                token: str = TOKEN) -> Path:
    """An arm's own paper-tables README.md as run_paper_tables.sh leaves it:
    the arm on line 1, its verdict on line 3, and the numbers.json written
    with it (its run's arm and token)."""
    p = _cmp(root, arm) / "paper_tables" / "README.md"
    _w(p, f"# Paper tables and numbers: PCA arm {arm}\n\n"
          f"{verdict or pt.VERDICT_COMPLETE}\n\nGenerated ...\n")
    _w(p.parent / "numbers.json", json.dumps(
        {"verdict": verdict or pt.VERDICT_COMPLETE,
         "run": {"pca_arm": arm, "run_token": token}}))
    return p


def _build_arms(root: Path, arms=ARMS) -> None:
    """A tree with every arm of `arms` (arm none's namespace always: _build),
    each arm's own paper tables saying PAPER COMPLETE."""
    _build(root)
    for a in arms:
        if a != "none":
            _build_arm(root, a, arms)
        _arm_readme(root, a)


class _EnvCleared:
    """Temporarily drop environment variables Ctx would otherwise read."""

    def __init__(self, *keys):
        self.keys = keys

    def __enter__(self):
        import os
        self.saved = {k: os.environ.pop(k) for k in self.keys
                      if k in os.environ}

    def __exit__(self, *exc):
        import os
        os.environ.update(self.saved)


class _PcaEnv:
    """The PCA arms' environment (PCA_ENV) cleared for the block, then set
    to `env`; restored after."""

    def __init__(self, **env):
        self.env = env

    def __enter__(self):
        import os
        self.saved = {k: os.environ.pop(k) for k in PCA_ENV
                      if k in os.environ}
        os.environ.update(self.env)

    def __exit__(self, *exc):
        import os
        for k in PCA_ENV:
            os.environ.pop(k, None)
        os.environ.update(self.saved)


def _run(root: Path, *extra, arm: str = "none", env=None) -> int:
    # Every setting Ctx would otherwise take from the environment is pinned
    # (an exported CODE_MANIFEST names the real tree's code, not the
    # fixture's; the PCA arm and targets are this test's, `env` aside), and
    # the run's arms are the pipeline's default, all four.
    with _PcaEnv(**(env or {})):
        return pt.main(["--root", str(root), "--variants", *VARIANTS,
                        "--selection_variant", "pd:per_head", "--run_token",
                        TOKEN, "--bestcat_extra_variants", "0",
                        "--pca_variance", "0.8", "--code_manifest", "",
                        "--pca_arm", arm, "--pca_arms", *ARMS, *extra])


def _run_arms(root: Path, *extra, arms=ARMS, env=None) -> int:
    """--pca_arms_table over `arms`, settings pinned as _run pins them."""
    with _PcaEnv(**(env or {})):
        return pt.main(["--root", str(root), "--variants", *VARIANTS,
                        "--selection_variant", "pd:per_head", "--run_token",
                        TOKEN, "--bestcat_extra_variants", "0",
                        "--pca_variance", "0.8", "--code_manifest", "",
                        "--pca_arms_table", "--pca_arms", *arms, *extra])


def _arms_out(root: Path) -> Path:
    return root / "sl_comparison" / "results" / "pca_arms"


def _rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _one(rows, **kw):
    hit = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(hit) == 1, (kw, len(hit))
    return hit[0]


def _close(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) <= tol, (a, b)


def _setup():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _build(root)
    rc = _run(root)
    out = root / "sl_comparison" / "results" / "paper_tables"
    return tmp, root, out, rc


def test_all_outputs_written():
    tmp, root, out, rc = _setup()
    with tmp:
        assert rc == INCOMPLETE
        for f in pt.OUTPUT_FILES:
            assert (out / f).exists(), f
        nums = json.loads((out / "numbers.json").read_text())
        assert all(t["status"] == "ok" for t in nums["tables"].values())
        readme = (out / "README.md").read_text()
        for t in pt.TABLES:
            assert f"{t}.csv" in readme
        assert "Missing inputs" in readme


def test_tiers():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_tiers.csv")
        r = _one(rows, cv="cv1", model="siamese_sl")
        _close(r["params_fold_mean"], 29022)
        _close(r["params_fold_sd"], 22)
        r2 = _one(rows, cv="cv2", model="siamese_sl")
        _close(r2["params_fold_mean"], 28170)
        _close(r2["params_fold_sd"], 170)
        _close(r["ap_mean"], .80)
        # DegreePrior has the top AUROC but is a control: never bold.
        assert _one(rows, cv="cv1", model="SLMGAE")["best_auroc"] == "True"
        assert _one(rows, cv="cv1", model="DegreePrior")["best_auroc"] == ""
        # Only the printed SLeuth variant (pd:pooled), the peers and the
        # probe decide the bold (2026-11-13): an unprinted variant leading
        # the tier is not bold and does not take the bold from the printed
        # rows.
        assert _one(rows, cv="cv2", model="siamese_sl_bilinear_per_head"
                    )["best_auroc"] == "", "unprinted variant bolded"
        assert _one(rows, cv="cv2", model="siamese_sl_bilinear_per_head"
                    )["in_paper_table"] == "False"
        assert _one(rows, cv="cv2", model="siamese_sl_pd_pooled"
                    )["in_paper_table"] == "True"
        assert any(r["best_auroc"] == "True" for r in rows
                   if r["cv"] == "cv2" and r["in_paper_table"] == "True")
        assert _one(rows, cv="cv2", model="siamese_sl")["beats_peers_auroc"] \
            == "True"
        assert _one(rows, cv="cv1", model="SLMGAE")["trainable_params"] == \
            str(PARAMS["SLMGAE"])
        # a SLeuth row carries its size only as params_fold_* (the column's
        # last-fold copy would sit next to the peers' exact counts)
        assert r["trainable_params"] == "", r["trainable_params"]
        # 4 SLeuth variants + 4 peers (ELISL among them) + BilinearProbe +
        # DegreePrior
        assert len({r["model"] for r in rows}) == 10
        # ELISL (ELISL/slurm/run_elisl.sh): a published peer of tab_tiers,
        # after NSF4SL, counted in beats_peers_* (pt.TIER_PEERS) but not in
        # the external tables' pt.PEERS (it has no external-screen rows).
        order1 = [r["model"] for r in rows if r["cv"] == "cv1"]
        assert order1.index("NSF4SL") < order1.index("ELISL") < \
            order1.index("BilinearProbe"), order1
        for cv in pt.CVS:
            e = _one(rows, cv=cv, model="ELISL")
            assert e["kind"] == "peer" and e["status"] == "ok", e
            _close(e["auprg_mean"], CMP["ELISL"][cv][2])
            assert e["trainable_params"] == str(PARAMS["ELISL"])
        assert "ELISL" in pt.TIER_PEERS and "ELISL" not in pt.PEERS
        # The bilinear probe (reviewer 22DY): a printed trained baseline, in
        # arm none, between the peers and the control; it competes for the
        # bold but is not a peer (beats_peers_* is against the three only).
        order = [r["model"] for r in rows if r["cv"] == "cv1"]
        assert order.index("NSF4SL") < order.index("BilinearProbe") < \
            order.index("DegreePrior"), order
        for cv in pt.CVS:
            p = _one(rows, cv=cv, model="BilinearProbe")
            assert p["kind"] == "baseline" and p["variant"] == "", p
            assert p["in_paper_table"] == "True" and p["status"] == "ok", p
            assert p["display"] == "BilinearProbe", p
            _close(p["auprg_mean"], CMP["BilinearProbe"][cv][2])
            assert p["trainable_params"] == str(PARAMS["BilinearProbe"])
            assert p["best_auprg"] == "False", p
            assert all(p[f"beats_peers_{m}"] == "" for m in pt.METRICS), p
        # a probe above every SLeuth variant would not change beats_peers_*
        assert _one(rows, cv="cv2", model="siamese_sl")["beats_peers_auroc"] \
            == "True"
        assert "BilinearProbe" not in pt.PEERS and \
            not pt.is_sleuth_row("BilinearProbe")


def test_solo():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_solo.csv")
        sel = [r for r in rows if r["variant"] == "pd:per_head"]
        assert _one(sel, cv="cv1", embedding="eb")["best_in_category"] == "True"
        assert _one(sel, cv="cv1", embedding="ea")["best_in_category"] == "False"
        st = _one(sel, cv="cv2", embedding="eb")["status"]
        assert st.startswith("stale") and "run_token" in st, st
        st = _one(sel, cv="cv3", embedding="pa")["status"]
        assert st.startswith("stale") and "dropout" in st, st
        assert _one(sel, cv="cv3", embedding="pa")["auprg_any_mean"] == ""
        assert _one(sel, cv="cv1", embedding="kb")["status"] == "not_generated"
        other = _one(rows, variant="pd:pooled", cv="cv1", embedding="ea")
        assert other["status"].startswith("not_run"), other["status"]
        nums = json.loads((out / "numbers.json").read_text())
        s = nums["items"]["solo"]["value"]["pd:per_head"]
        _close(s["per_cv"]["cv1"]["category_means"]["expression"], .915)
        assert s["per_cv"]["cv1"]["best_category"] == "ppi"
        assert s["per_cv"]["cv3"]["best_category"] == "expression"
        assert s["per_cv"]["cv1"]["above_combo_shared"] == ["pa"]
        assert s["per_cv"]["cv3"]["above_combo_shared"] == ["eb"]
        assert s["per_cv"]["cv3"]["n_above_combo_self"] == 1
        assert s["above_combo_at_every_tier_shared"] == []
        # None of the synthetic embeddings is among the paper's 16.
        assert s["per_cv"]["cv1"]["category_means_paper16"] == {}
        assert s["per_cv"]["cv1"]["best_category_paper16"] is None
        readme = (out / "README.md").read_text()
        assert "pcavar0.8_eb_cv2 is stale" in readme


def test_category_means_paper16():
    """AB2 averages over the paper's 16; extra catalog members (esmc) only
    enter the all-catalog means."""
    import types
    ctx = types.SimpleNamespace(variants=["pd:per_head"],
                                  selection_variant="pd:per_head")
    rows = [{"variant": "pd:per_head", "status": "ok", "cv": "cv1",
             "category": c, "embedding": e, "in_paper": e in pt.PAPER_SOLO,
             "auprg_any_mean": x}
            for e, c, x in (("esm2", "protein_seq", .9),
                            ("prot_t5", "protein_seq", .8),
                            ("esmc", "protein_seq", .1),
                            ("ppi_raw", "ppi", .7))]
    s = pt.solo_summaries(ctx, rows)["pd:per_head"]["per_cv"]["cv1"]
    _close(s["category_means"]["protein_seq"], .6)
    assert s["best_category"] == "ppi"
    _close(s["category_means_paper16"]["protein_seq"], .85)
    assert s["best_category_paper16"] == "protein_seq"
    assert s["n_embeddings"] == 4 and s["n_embeddings_in_paper16"] == 3


def test_solo_prefix_follows_selection_variant():
    """bestcat_solo_prefix: SELECTION_VARIANT's single-modality runs are the
    unprefixed dirs, a non-selection pd:per_head's are var_pd_per_head__."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"
        folds = root / "sl_comparison" / "folds"
        for d in sorted(res.glob("pcavar0.8_*")):
            d.rename(res / f"var_pd_per_head__{d.name}")
        for cv in pt.CVS:
            for etype in ("ea", "pa"):
                _siamese_dir(res / f"pcavar0.8_{etype}_{cv}",
                             _cfg("bilinear:pooled", cv, solo=True), cv, folds,
                             SOLO[cv][etype] - .05, [{}, {}])
        # A repeated option's last value wins.
        rc = _run(root, "--selection_variant", "bilinear:pooled",
                  "--bestcat_extra_variants", "1")
        out = root / "sl_comparison" / "results" / "paper_tables"
        assert rc == INCOMPLETE
        rows = _rows(out / "tab_solo.csv")
        bp = _one(rows, variant="bilinear:pooled", cv="cv1", embedding="ea")
        assert bp["status"] == "ok" and bp["selection_variant"] == "True"
        assert bp["dir"] == "siamese_sl/results/pcavar0.8_ea_cv1"
        _close(bp["auprg_any_mean"], SOLO["cv1"]["ea"] - .05)
        ph = _one(rows, variant="pd:per_head", cv="cv1", embedding="ea")
        assert ph["status"] == "ok", ph["status"]
        assert ph["dir"] == ("siamese_sl/results/"
                             "var_pd_per_head__pcavar0.8_ea_cv1")
        _close(ph["auprg_any_mean"], SOLO["cv1"]["ea"])
        other = _one(rows, variant="pd:pooled", cv="cv1", embedding="ea")
        assert other["status"] == "missing"  # requested (=1) but absent
        assert other["dir"] == "siamese_sl/results/var_pd_pooled__pcavar0.8_ea_cv1"


def test_empty_item_is_missing():
    assert pt.is_empty(None) and pt.is_empty("") and pt.is_empty({})
    assert pt.is_empty({"a": {"x": {}, "y": []}, "b": None})
    assert not pt.is_empty({"a": {"x": 0}}) and not pt.is_empty([False])
    tmp, root, out, rc = _setup()
    with tmp:
        import shutil
        shutil.rmtree(root / "sl_comparison" / "results" / "external_transfer")
        assert _run(root) == 1
        nums = json.loads((out / "numbers.json").read_text())
        assert nums["items"]["external"]["status"] == "missing: no input"
        assert "numbers.json external: empty (no input)" in nums["missing"]
        assert nums["items"]["data_stats"]["status"] == "ok"


def test_loco():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_loco.csv")
        sel = [r for r in rows if r["variant"] == "pd:per_head"]
        ref = _one(sel, cv="cv3", dropped="none")
        _close(ref["auprg_any_mean"], .73)
        _close(ref["shared_eval_auprg"], .732)
        go = _one(sel, cv="cv3", dropped="ppi")
        _close(go["delta_vs_reference"], .60 - .73)
        assert go["removed_embedding"] == "pa"
        # Paired by 'fold' (listed in reverse): per-fold deltas -.12 / -.14,
        # so sd .01; pairing by position would give sd .03.
        _close(go["delta_paired_mean"], .60 - .73)
        _close(go["delta_paired_sd"], .01)
        assert go["n_paired"] == "2"
        assert ref["delta_paired_mean"] == ""
        bp = _one(rows, variant="bilinear:pooled", cv="cv3",
                  dropped="expression")
        assert bp["status"] == "ok"
        assert bp["delta_paired_mean"] == ""  # its folds carry no auprg_any
        assert "var_bilinear_pooled__ablate_no_expression_cv3/results.json " \
            "fold_metrics[].auprg_any pairing" in (out / "README.md").read_text()
        assert _one(rows, variant="bilinear:pooled", cv="cv3",
                    dropped="ppi")["status"] == "missing"
        nums = json.loads((out / "numbers.json").read_text())
        lo = nums["items"]["loco"]["value"]["pd:per_head"]["cv3"]
        assert lo["largest_drop"] == "ppi"
        assert lo["categories_missing"] == []
        assert lo["no_drop"] == ["expression", "kg"]
        # Only the expression ablation of bilinear:pooled cv3 exists: its
        # largest drop is unknown, not "expression".
        bl = nums["items"]["loco"]["value"]["bilinear:pooled"]["cv3"]
        assert bl["largest_drop"] is None
        assert bl["categories_missing"] == ["ppi", "kg"]
        assert "numbers.json loco bilinear:pooled/cv3: no ppi, kg ablation " \
            "delta; largest_drop left empty" in nums["missing"]
        assert lo["paired"]["ppi"]["n"] == 2
        _close(lo["paired"]["ppi"]["sd"], .01)


def test_paired_deltas_alignment():
    fm = [{"fold": 1, "auprg_any": .5}, {"fold": 0, "auprg_any": .4}]
    ref = [{"fold": 0, "auprg_any": .3}, {"fold": 1, "auprg_any": .3}]
    p = pt.paired_deltas(fm, ref)
    _close(p["delta_paired_mean"], .15)
    _close(p["delta_paired_sd"], .05)
    assert p["n_paired"] == 2
    # No fold ids: by position, only when the lengths agree.
    p = pt.paired_deltas([{"auprg_any": .5}, {"auprg_any": .4}],
                         [{"auprg_any": .3}, {"auprg_any": .2}])
    _close(p["delta_paired_sd"], 0)
    assert pt.paired_deltas([{"auprg_any": .5}], [{}, {}]) is None
    # Only the folds both sides scored; NaN is not a score.
    p = pt.paired_deltas([{"fold": 0, "auprg_any": .5},
                          {"fold": 2, "auprg_any": float("nan")}],
                         [{"fold": 0, "auprg_any": .1},
                          {"fold": 1, "auprg_any": .1},
                          {"fold": 2, "auprg_any": .1}])
    assert p["n_paired"] == 1
    assert pt.paired_deltas([], []) is None


def test_cellline():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_cellline.csv")
        rem = _one(rows, variant="pd:per_head", cv="cv1",
                   condition="removed (cell_emb=0)")
        _close(rem["delta_auprg_vs_included"], .94 - .951)
        inc = _one(rows, variant="pd:per_head", cv="cv1", condition="included")
        assert inc["selfcheck_as_trained"] == "match"
        assert rem["status"] == "ok"
        assert _one(rows, variant="pd:per_head", cv="cv2",
                    condition="included")["selfcheck_as_trained"] == "MISMATCH"
        # A MISMATCH means the ablated model is not the included one: its
        # removed row differences two models, so no delta and not in CL1.
        rem2 = _one(rows, variant="pd:per_head", cv="cv2",
                    condition="removed (cell_emb=0)")
        assert rem2["status"].startswith("stale: zero_cell_emb.py's "
                                         "as-trained row differs"), rem2
        assert rem2["delta_auprg_vs_included"] == ""
        _close(rem2["auprg_mean"], CMP["siamese_sl_nocell"]["cv2"][2])
        nums = json.loads((out / "numbers.json").read_text())
        cl1 = nums["items"]["cellline"]["value"]["pd:per_head"]
        assert set(cl1) == {"cv1"}, cl1  # cv3: no results.json to check
        assert any(m.startswith("tab_cellline siamese_sl/cv2: ")
                   for m in nums["missing"]), nums["missing"]
        jb = _one(rows, variant="bilinear:per_head", cv="cv3",
                  condition="removed (cell_emb=0)")
        _close(jb["auprg_mean"], NOCELL_BILINEAR_CV3[2])  # JSON fallback
        assert jb["status"].startswith("stale: no zero_cell_emb.py")
        ret = _one(rows, variant="pd:per_head", cv="cv3",
                   condition="retrained without cell lines (pd:pooled)")
        _close(ret["auprg_mean"], .72)
        na = _one(rows, variant="pd:pooled", cv="cv1")
        assert na["status"].startswith("not applicable")


def test_external():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_external.csv")
        assert {r["row_set"] for r in rows if r["kind"] == "siamese"} == {
            "matched", "all_sldb_filtered", "matched_incl_sldb",
            "matched_in_sldb"}
        assert {r["row_set"] for r in rows if r["kind"] != "siamese"} == {
            "matched", "matched_incl_sldb", "matched_in_sldb"}
        # the SynLethDB label as a predictor: matched_in_sldb only, never bold
        lab = [r for r in rows if r["kind"] == "sldb_label"]
        assert lab and {r["row_set"] for r in lab} == {"matched_in_sldb"}
        assert all(r["best_matched"] == "" for r in lab)
        # screen_role: Gilbert is the in-distribution check
        assert {r["screen_role"] for r in rows if r["screen"] == "Gilbert"} \
            == {"in_distribution_check"}
        assert {r["screen_role"] for r in rows if r["screen"] == "Corn"} \
            == {"external"}
        assert all(r["best_matched"] == "" for r in rows
                   if r["row_set"] != "matched")
        c = [r for r in rows if r["screen"] == "Adamson" and r["split"] == "cv1"]
        assert _one(c, model="SLMGAE", row_set="matched"
                    )["best_matched"] == "True"
        assert _one(c, model="DegreePrior", row_set="matched"
                    )["best_matched"] == ""
        g = [r for r in rows if r["screen"] == "Gilbert" and r["split"] == "cv2"]
        assert _one(g, model="siamese_sl", row_set="matched"
                    )["best_matched"] == "True"
        ft = _rows(out / "tab_external_ft.csv")
        full = _one(ft, model="siamese_sl", screen="Adamson", split="cv1",
                    row_set="all_sldb_filtered", method="Full")
        assert full["best_method"] == "True" and full["column"] == "Full"
        zs = _one(ft, model="siamese_sl", screen="Adamson", split="cv1",
                  row_set="all_sldb_filtered", method="zero_shot_raw")
        assert zs["column"] == "Zero-shot"
        nums = json.loads((out / "numbers.json").read_text())
        e = nums["items"]["external"]["value"]["pd:per_head"]
        rk = e["X5_rank_on_matched_rows"]["Adamson/cv1"]
        assert rk["rank"] == 3 and rk["of"] == 4
        _close(e["F2_F3_all_sldb_filtered"]["Adamson/cv1"]
               ["full_minus_zero_shot_raw"], 1.17)


def test_numbers():
    tmp, root, out, rc = _setup()
    with tmp:
        it = json.loads((out / "numbers.json").read_text())["items"]
        assert all(v["status"] == "ok" for v in it.values()), {
            k: v["status"] for k, v in it.items() if v["status"] != "ok"}
        ds = it["data_stats"]["value"]
        assert ds["all_pairs"] == 66
        _close(ds["positive_pct_of_all_pairs"], 100 * 8 / 66)
        fg = it["fold_geometry"]["value"]
        _close(fg["cv1"]["test_share"]["mean"], .25)
        _close(fg["cv1"]["discarded_share"]["mean"], 0)
        _close(fg["cv1"]["test_both_genes_in_train"]["mean"], .4)
        _close(fg["cv1"]["test_both_genes_in_train_val"]["mean"], .8)
        _close(fg["cv1"]["test_prevalence"]["min"], .4)
        _close(fg["cv3"]["test_share"]["mean"], .175)
        _close(fg["cv3"]["test_share"]["min"], .15)
        _close(fg["cv3"]["discarded_share"]["mean"], .375)
        _close(fg["cv3"]["test_prevalence"]["min"], 1 / 3)
        _close(fg["cv3"]["test_prevalence"]["max"], .5)
        _close(fg["D7_analytic_gene_split"]["p_mixed"], .5)
        hd = it["heads"]["value"]
        assert hd["max_heads_per_pair"] == 5
        assert hd["pairs_by_n_heads"] == {"1": 17, "2": 2, "5": 1}
        assert hd["num_heads"] == 8
        pw = it["pca_width"]["value"]
        assert pw["per_variant"]["pd:per_head"]["min"] == 1505
        assert pw["per_variant"]["pd:per_head"]["max"] == 1585
        assert pw["equal_across_variants"] is True
        pr = it["param_ratio"]["value"]["pd:per_head"]
        _close(pr["per_cv"]["cv1"]["ratio"], PARAMS["SLMGAE"] / 29022)
        _close(pr["min_ratio"], PARAMS["SLMGAE"] / 29100)
        assert pw["per_variant"]["pd:per_head"]["n_splits"] == 6
        sel = it["selection"]["value"]
        assert sel["winners"] == WINNERS
        assert sel["catalog_size"] == 5
        assert set(sel["differs_from_paper"]) == {"expression", "ppi", "kg"}
        mg = it["margins"]["value"]["pd:per_head"]
        _close(mg["tier3_auroc_gap_slmgae_minus_sleuth"], .705 - .677)
        assert mg["tier3_auroc_gap_inside_spread"] is True
        assert mg["tier3_gap_not_smaller_than_tier1_margin"] is True
        assert mg["cv2"]["auprg"]["leads"] is True
        sc = it["screens"]["value"]
        assert sc["equal_across_variants"] is True
        a = sc["per_variant"]["pd:per_head"]["Adamson"]
        _close(a["X2_share_both_in_genes_txt"], 15 / 28)
        _close(a["X3_sldb_overlap_share_among_both"], 6 / 15)
        _close(a["recomputed_share_in_pos_or_neg_among_both"], 6 / 15)
        _close(a["recomputed_share_in_pos_only_among_both"], 4 / 15)
        _close(a["F4_additive_r2"]["r2"], 1.0, 1e-6)
        _close(a["D11_sldb_agreement"]["auroc_target_vs_sldb_label"], 0.0)
        g = sc["per_variant"]["pd:per_head"]["Gilbert"]
        _close(g["D11_sldb_agreement"]["auroc_target_vs_sldb_label"], 1.0)
        assert 0 <= sc["per_variant"]["pd:per_head"]["Corn"][
            "F4_additive_r2"]["r2"] < 0.999
        arch = it["arch"]["value"]["bilinear:pooled"]
        assert arch["arch_signature"]["kernel"] == "bilinear"
        assert arch["arch_signature"]["cell_line_dim"] is None  # pooled


def test_missing_source_exits_1_and_clears_stale_output():
    tmp, root, out, rc = _setup()
    with tmp:
        assert rc == INCOMPLETE and (out / "tab_external.csv").exists()
        (root / "sl_comparison" / "results" / "external_transfer" /
         "zero_shot_summary.csv").unlink()
        # The superseded fixed-selection PCA ablation's table (2026-10-19)
        # goes too: it must not sit next to this run's as one of them.
        _w(out / "tab_pca.csv", "arm,role\nnone,reference\n")
        assert _run(root) == 1
        assert not (out / "tab_external.csv").exists()
        assert not (out / "tab_pca.csv").exists()
        assert (out / "tab_tiers.csv").exists()
        nums = json.loads((out / "numbers.json").read_text())
        assert nums["tables"]["tab_external"]["status"] == "missing"
        assert "tab_external.csv (no usable source)" in (
            out / "README.md").read_text()


def test_regrade_attaches_shared_auprg():
    """--regrade grades cells whose matrices were kept (evaluate_model.py is
    stubbed) and names the rest as skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        d = root / "siamese_sl" / "results" / "pcavar0.8_ea_cv1"
        for k in range(2):
            np.save(d / f"fold_{k}_cv1_predictions.npy", np.zeros((2, 2)))
        calls = []

        def fake(pred_dir, folds, cv, name, out):
            calls.append((pred_dir.name, cv))
            _wj(out, {"summary": {"auprg_mean": .5, "auprg_std": .1}})
            return True

        orig, pt._evaluate = pt._evaluate, fake
        try:
            # Settings pinned by _run: an exported SELECTION_VARIANT /
            # BEST_CAT_PCA_VARIANCE must not move the dirs looked up.
            rc = _run(root, "--regrade")
        finally:
            pt._evaluate = orig
        assert rc == INCOMPLETE and calls == [("pcavar0.8_ea_cv1", "cv1")], calls
        rows = _rows(root / "sl_comparison" / "results" / "paper_tables" /
                     "tab_solo.csv")
        r = _one(rows, variant="pd:per_head", cv="cv1", embedding="ea")
        assert r["regrade"] == "ok"
        _close(r["auprg_shared_mean"], .5)
        assert _one(rows, variant="pd:per_head", cv="cv1", embedding="eb"
                    )["regrade"].startswith("skipped")


def test_config_conf_fallback():
    """Without --variants / the environment, the variant list, the selection
    variant and ARCH_ARGS come from siamese_sl/slurm/config.conf."""
    import os
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Defaults other than Ctx's own fallbacks, so a lost config.conf
        # lookup is caught.
        _w(root / "siamese_sl" / "slurm" / "config.conf",
           'SIAMESE_VARIANTS=("pd:per_head" "bilinear:pooled")\n'
           'SELECTION_VARIANT="${SELECTION_VARIANT:-bilinear:pooled}"\n'
           'ARCH_ARGS="${ARCH_ARGS:---layernorm}"\n'
           'echo "noise that must be ignored"\n')
        # The pipeline's own overrides would win over the fake conf (and
        # reach the sourcing subshell), so clear them for the whole check.
        saved = {k: os.environ.pop(k) for k in (
            "SIAMESE_VARIANTS_LIST", "SELECTION_VARIANT", "ARCH_ARGS",
            "RUN_TOKEN", "BESTCAT_EXTRA_VARIANTS",
            "BEST_CAT_PCA_VARIANCE", "CODE_MANIFEST") + PCA_ENV
            if k in os.environ}
        try:
            assert pt.read_config_conf(root) == {
                "SIAMESE_VARIANTS_LIST": "pd:per_head bilinear:pooled",
                "SELECTION_VARIANT": "bilinear:pooled",
                "ARCH_ARGS": "--layernorm"}
            args = pt.argparse.Namespace(
                root=str(root), out_dir=None, external_dir=None,
                comparison_csv=None, variants=None, selection_variant=None,
                run_token=None, bestcat_extra_variants=None,
                pca_variance=None, regrade=False)
            ctx = pt.Ctx(args)
        finally:
            os.environ.update(saved)
        assert ctx.variants == ["pd:per_head", "bilinear:pooled"]
        assert ctx.selection_variant == "bilinear:pooled"
        assert ctx.arch_args_env == "--layernorm"
        # No PCA arm at all (a bare Namespace): arm none's paths, as before.
        r = root.resolve()
        assert (ctx.pca_arm, ctx.pca_arms) == ("none", ["none"])
        assert ctx.sia_results == ctx.step1_results == \
            r / "siamese_sl" / "results"
        assert ctx.cmp_results == ctx.shared_results == \
            r / "sl_comparison" / "results"
        assert ctx.out_dir == r / "sl_comparison" / "results" / "paper_tables"
        # ...and it picks the single-modality dirs.
        assert pt.solo_prefix(ctx, "bilinear:pooled") == ""
        assert pt.solo_prefix(ctx, "pd:per_head") == "var_pd_per_head__"


def test_bash_array_parser_strips_comments():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.conf"
        p.write_text('A=(\n  # (x) y\n  "a:b:c"  # (z)\n  "d:e:f"\n)\n'
                     'B=(one two)\n')
        assert pt.parse_bash_array(p, "A") == ["a:b:c", "d:e:f"]
        assert pt.parse_bash_array(p, "B") == ["one", "two"]


def _missing(out: Path):
    return json.loads((out / "numbers.json").read_text())["missing"]


def test_unregraded_or_foreign_json_row_ignored():
    """A top-level row without evaluate_model provenance (a collect_siamese.py
    row an interrupted compare never re-graded), or graded on other folds,
    never fills a table cell."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        cmp_dir = root / "sl_comparison" / "results"
        p = cmp_dir / "siamese_sl_bilinear_per_head_nocell_cv3.json"
        row = json.loads(p.read_text())
        del row["summary"]["provenance"]
        _wj(p, row)
        # JSON-only row, graded on another fold set.
        folds = root / "sl_comparison" / "folds"
        q = cmp_dir / "siamese_sl_nocell_cv3.json"
        _row_json(q, "siamese_sl_nocell", "cv3", (.5, .5, .5), folds)
        row = json.loads(q.read_text())
        row["summary"]["provenance"]["meta.json_sha256"] = "0" * 16
        _wj(q, row)
        c = [r for r in _rows(cmp_dir / "comparison.csv")
             if not (r["model"] == "siamese_sl_nocell" and r["cv"] == "cv3")]
        with open(cmp_dir / "comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(c[0]))
            w.writeheader()
            w.writerows(c)
        assert _run(root) == INCOMPLETE
        out = cmp_dir / "paper_tables"
        rows = _rows(out / "tab_cellline.csv")
        for v in ("bilinear:per_head", "pd:per_head"):
            rem = _one(rows, variant=v, cv="cv3",
                       condition="removed (cell_emb=0)")
            assert rem["status"] == "missing" and rem["auprg_mean"] == "", v
        miss = _missing(out)
        assert any("siamese_sl_bilinear_per_head_nocell_cv3.json (not graded "
                   "by evaluate_model.py; ignored)" in m for m in miss), miss
        assert any(m.startswith("siamese_sl_nocell/cv3 (json) is not on the "
                                "current folds") for m in miss), miss


def test_fold_provenance():
    tmp, root, out, rc = _setup()
    with tmp:
        fp = json.loads((out / "numbers.json").read_text())["items"][
            "fold_provenance"]["value"]
        assert fp["all_rows_same_folds"] is True
        assert fp["rows_without_training_stamp"] == [
            f"DegreePrior/{cv}" for cv in pt.CVS]
        assert fp["rows"]["SLGNN/cv2"]["trained_on"]["meta.json"] == \
            fp["current"]["meta.json"]
        # One comparison.csv row graded on another fold set: left out.
        p = root / "sl_comparison" / "results" / "comparison.csv"
        c = _rows(p)
        for r in c:
            if (r["cv"], r["model"], r["metric"]) == (
                    "cv2", "SLGNN", "folds_meta.json_sha256"):
                r["mean"] = "1234567890123456"  # all digits: still a hash
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(c[0]))
            w.writeheader()
            w.writerows(c)
        assert _run(root) == INCOMPLETE
        tiers = _rows(out / "tab_tiers.csv")
        assert _one(tiers, cv="cv2", model="SLGNN")["status"] == "missing"
        assert _one(tiers, cv="cv1", model="SLGNN")["status"] == "ok"
        nums = json.loads((out / "numbers.json").read_text())
        fp = nums["items"]["fold_provenance"]["value"]
        assert fp["all_rows_same_folds"] is False
        assert fp["rows"]["SLGNN/cv2"]["problems"]
        assert any(m.startswith("SLGNN/cv2 (comparison.csv) is not on the "
                                "current folds") for m in nums["missing"])


def test_external_grid_cells_listed():
    """audit4 a7: an absent screen / split / row set / method cell, or a mean
    over fewer finite folds, is named even though the table is built."""
    tmp, root, out, rc = _setup()
    with tmp:
        ext = root / "sl_comparison" / "results" / "external_transfer"
        zs = [r for r in _rows(ext / "zero_shot_summary.csv")
              if not (r["model"] == "SLGNN" and r["screen"] == "Gilbert")]
        for r in zs:
            if (r["model"], r["screen"], r["split"], r["row_set"]) == (
                    "NSF4SL", "Corn", "cv2", "matched"):
                r["n_folds_finite"] = "1"
        ft = [r for r in _rows(ext / "finetune_summary.csv")
              if not (r["model"] == "siamese_sl_pd_pooled" and
                      r["method"] == "Full")]
        for name, rows in (("zero_shot_summary.csv", zs),
                           ("finetune_summary.csv", ft)):
            with open(ext / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        assert _run(root) == INCOMPLETE
        miss = _missing(out)
        for want in (
                "tab_external SLGNN Gilbert/cv1/matched not in",
                "tab_external SLGNN Gilbert/cv3/matched not in",
                "tab_external NSF4SL Corn/cv2/matched: 1 of 2 folds finite",
                "tab_external_ft siamese_sl_pd_pooled "
                "Adamson/cv1/all_sldb_filtered/Full not in",
                "tab_external_ft siamese_sl_pd_pooled Gilbert/cv3/matched/Full "
                "not in"):
            assert any(m.startswith(want) for m in miss), (want, miss)
        assert not any("siamese_sl_pd_pooled" in m and "LP_last" in m
                       for m in miss)
        # The default tree names no external cell.
        tmp2, _, out2, _ = _setup()
        with tmp2:
            assert not any(m.startswith("tab_external")
                           for m in _missing(out2)), _missing(out2)


def test_not_generated_embedding_listed():
    tmp, root, out, rc = _setup()
    with tmp:
        assert "data/all_genes_kb.pt not generated: tab_solo kb rows absent " \
            "(not among the paper's 16)" in _missing(out)
        # The synthetic catalog holds none of the paper's 16.
        assert any(m.startswith("EMB_CATALOG lacks 16 of tab:solo's 16 "
                                "embeddings (bioconceptvec, esm1b")
                   for m in _missing(out)), _missing(out)


def test_every_tier_needs_every_reference():
    """audit4 a6: with one tier's combo reference unknown, 'above the combo at
    every tier' is unknown, not the intersection of the other two."""
    import types
    ctx = types.SimpleNamespace(variants=["pd:per_head"],
                                  selection_variant="pd:per_head")

    def rows(ref_cv2):
        out = []
        for cv in pt.CVS:
            for e, x in (("go2vec", .9), ("esm2", .1)):
                ref = ref_cv2 if cv == "cv2" else .5
                out.append({"variant": "pd:per_head", "status": "ok",
                            "cv": cv, "category": "go", "embedding": e,
                            "in_paper": True, "auprg_any_mean": x,
                            "above_combo_shared": None if ref is None
                            else x > ref, "above_combo_self": x > .5,
                            "combo_auprg_shared": ref,
                            "combo_auprg_any_self": .5})
        return out

    s = pt.solo_summaries(ctx, rows(None))["pd:per_head"]
    assert s["above_combo_at_every_tier_shared"] is None
    assert s["above_combo_at_every_tier_self"] == ["go2vec"]
    s = pt.solo_summaries(ctx, rows(.5))["pd:per_head"]
    assert s["above_combo_at_every_tier_shared"] == ["go2vec"]


def test_solo_bold_paper16():
    """The paper's bold is over its 16 printed rows; a single-candidate
    category (kg) gets none."""
    rows = [{"variant": "pd:per_head", "cv": "cv1", "status": "ok",
             "category": c, "embedding": e, "in_paper": e in pt.PAPER_SOLO,
             "auprg_any_mean": x}
            for e, c, x in (("esm2", "protein_seq", .8),
                            ("prot_t5", "protein_seq", .7),
                            ("esmc", "protein_seq", .9),
                            ("kg_complex", "kg", .5))]
    assert pt.flag_solo_best(rows) == []
    by = {r["embedding"]: r for r in rows}
    assert by["esmc"]["best_in_category"] is True
    assert by["esmc"]["best_in_category_paper16"] is None
    assert by["esm2"]["best_in_category_paper16"] is True
    assert by["prot_t5"]["best_in_category_paper16"] is False
    assert by["kg_complex"]["best_in_category"] is None
    assert by["kg_complex"]["best_in_category_paper16"] is None
    # esm1b's row exists but is stale: the paper's bold of protein_seq is
    # unknown (not prot_t5 by default); best_in_category still ranks those
    # that ran.
    rows.append({"variant": "pd:per_head", "cv": "cv1", "status": "stale: x",
                 "category": "protein_seq", "embedding": "esm1b",
                 "in_paper": True, "auprg_any_mean": None})
    rows = [{k: x for k, x in r.items() if not k.startswith("best")}
            for r in rows if r["embedding"] != "esm2"]
    assert pt.flag_solo_best(rows) == [("pd:per_head", "cv1", "protein_seq",
                                        ["esm1b"])]
    by = {r["embedding"]: r for r in rows}
    assert by["prot_t5"]["best_in_category_paper16"] is None
    assert by["esmc"]["best_in_category"] is True


def test_cellline_residual_heads_and_self_scored_rows():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_cellline.csv")
        rem = _one(rows, variant="pd:per_head", cv="cv1",
                   condition="removed (cell_emb=0)")
        assert rem["residual_head_terms"] == "head_proj_bias (per-head c_h)"
        hpb = _one(rows, variant="pd:per_head", cv="cv1",
                   condition="self-scored: cell_emb=0 +head_proj_bias=0")
        _close(hpb["auprg_mean"], CMP["siamese_sl"]["cv1"][2] - .1)
        _close(hpb["auprg_sd"], .02)
        _close(hpb["delta_auprg_vs_included"], -.1)
        # zero_cell_emb.py's own as-trained / cell_emb=0 rows are not repeated.
        assert not [r for r in rows if r["condition"] in (
            "self-scored: as-trained", "self-scored: cell_emb=0")]


def test_external_scored_head_and_split_label():
    tmp, root, out, rc = _setup()
    with tmp:
        rows = _rows(out / "tab_external.csv")
        c = [r for r in rows if r["screen"] == "Corn" and r["split"] == "cv2"]
        assert _one(c, model="siamese_sl", row_set="matched"
                    )["scored_head"] == "OTHER"
        assert _one(c, model="siamese_sl_bilinear_pooled", row_set="matched"
                    )["scored_head"] == "ANY (pooled)"
        assert _one(c, model="SLMGAE", row_set="matched"
                    )["scored_head"] == ""
        assert _one(c, model="SLMGAE", row_set="matched"
                    )["split_label"] == ">=1 gene unseen"
        ft = _rows(out / "tab_external_ft.csv")
        r = _one(ft, model="siamese_sl", screen="Adamson", split="cv3",
                 row_set="all_sldb_filtered", method="Full")
        assert r["scored_head"] == "K562"
        assert r["split_label"] == "both genes unseen"


def test_split_labels_match_external_transfer():
    ext = pt._load_module("_t_ext", HERE / "external_transfer.py")
    assert pt.SPLIT_LABELS == ext.SPLIT_LABELS


def test_inputs_item():
    tmp, root, out, rc = _setup()
    with tmp:
        it = json.loads((out / "numbers.json").read_text())["items"]["inputs"]
        assert it["status"] == "ok"
        v = it["value"]
        assert v["synlethkg"]["n_relations"] == 4
        assert v["synlethkg"]["entity_types_from_relations"] == [
            "C", "CC", "G", "PC", "SE"]
        assert v["training_seeds"]["SLGNN/cv1"] == SEED
        assert v["training_seeds"]["siamese_sl_bilinear_pooled/cv3"] == SEED
        assert v["single_training_seed"] is True
        assert "DegreePrior/cv1" in v["rows_without_seed"]
        if torch is not None:  # every generated file's own width (kb: none)
            assert v["native_dims"] == WIDTH, v["native_dims"]
    # On the relation list SLGNN actually ships: the paper's 11 / 24.
    kg = pt._load_module("_t_kg", HERE.parent / "SLGNN" / "build_synlethkg.py")
    rels = [name for name, _, _ in kg.RELATIONS]
    assert len(rels) == 24 and len(pt.kg_entity_types(rels)) == 11


def _drop_rows(root: Path, *cells) -> None:
    """Remove (cv, model) rows from comparison.csv AND their row JSONs, as a
    failed task leaves them (its JSON is moved aside, and compare.py then
    writes no row for it)."""
    _rewrite_cmp(root, lambda r: (r["cv"], r["model"]) not in cells)
    for cv, model in cells:
        (root / "sl_comparison" / "results" / f"{model}_{cv}.json").unlink(
            missing_ok=True)


def _rewrite_cmp(root: Path, keep) -> None:
    """Keep only the comparison.csv rows for which keep(row) is true."""
    p = root / "sl_comparison" / "results" / "comparison.csv"
    c = [r for r in _rows(p) if keep(r)]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(c[0]))
        w.writeheader()
        w.writerows(c)


def _out(root: Path, arm: str = "none") -> Path:
    return _cmp(root, arm) / "paper_tables"


def _numbers(root: Path, arm: str = "none") -> dict:
    return json.loads((_out(root, arm) / "numbers.json").read_text())


def test_cellline_source_status_and_ablated_combo():
    """zero_cell_emb.py still writes results.json (and exits 1) for a failed
    self-check or a partial fold set, and it may have ablated another combo:
    its removed and self-scored rows are then not tabulated."""
    res = Path("siamese_sl") / "results"
    for over, want in (
            ({"status": "selfcheck_failed", "self_check": {"ok": False}},
             "status selfcheck_failed"),
            ({"status": "incomplete", "n_folds": 1}, "status incomplete"),
            ({"n_folds": 1}, "scored 1 of 2 folds"),
            ({"self_check": {"ok": None}}, "self-check did not pass"),
            ({"source_model_dir": "results/best_cat_ea_pa_cv1"},
             "ablated results/best_cat_ea_pa_cv1")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root)
            # cv2 made clean, so the table keeps one tabulated removed row.
            agg2 = {m: [x, .01] for m, x in zip(pt.METRICS,
                                                CMP["siamese_sl"]["cv2"])}
            _wj(root / res / "cellline_ablation_cv2" / "results.json",
                {**CELLLINE_OK, "agg": {"as-trained": agg2}})
            p = root / res / "cellline_ablation_cv1" / "results.json"
            _wj(p, {**json.loads(p.read_text()), **over})
            assert _run(root) == INCOMPLETE, over
            rows = _rows(_out(root) / "tab_cellline.csv")
            rem = _one(rows, variant="pd:per_head", cv="cv1",
                       condition="removed (cell_emb=0)")
            assert rem["status"].startswith("stale") and \
                want in rem["status"], (over, rem["status"])
            assert rem["delta_auprg_vs_included"] == ""
            assert not [r for r in rows if r["variant"] == "pd:per_head" and
                        r["cv"] == "cv1" and
                        r["condition"].startswith("self-scored")], over
            assert _one(rows, variant="pd:per_head", cv="cv2",
                        condition="removed (cell_emb=0)")["status"] == "ok"
            nums = _numbers(root)
            assert set(nums["items"]["cellline"]["value"]["pd:per_head"]) \
                == {"cv2"}
            assert any(m.startswith("tab_cellline siamese_sl/cv1: ") and
                       want in m for m in nums["missing"]), nums["missing"]


def test_combo_off_current_folds_ignored():
    """A combo trained on another fold set (--skip-siamese after the folds
    changed) feeds no width, parameter, arch or tab_loco reference value, and
    the selection its run made is not reported."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        folds = root / "sl_comparison" / "folds"
        for cv in pt.CVS:
            d = (root / "siamese_sl" / "results" /
                 f"var_bilinear_pooled__best_cat_ea_ka_pa_{cv}")
            st = mp.fold_stamp(folds, cv)
            st["meta.json"] = "0" * 16
            _wj(d / "model_params.json", {"fold_stamp": st})
        _run(root)
        nums = _numbers(root)
        it = nums["items"]
        assert "bilinear:pooled" not in it["pca_width"]["value"]["per_variant"]
        assert "pd:per_head" in it["pca_width"]["value"]["per_variant"]
        assert "bilinear:pooled" not in it["arch"]["value"]
        assert "bilinear:pooled" not in it["param_ratio"]["value"]
        assert it["selection"]["status"].startswith("missing"), it["selection"]
        assert any("var_bilinear_pooled__best_cat_ea_ka_pa_cv1 "
                   "(bilinear:pooled cv1 combo) is not on the current folds"
                   in m for m in nums["missing"]), nums["missing"]
        loco = _rows(_out(root) / "tab_loco.csv")
        assert _one(loco, variant="bilinear:pooled", cv="cv3",
                    dropped="none")["status"] == "missing"
        tiers = _rows(_out(root) / "tab_tiers.csv")
        assert _one(tiers, cv="cv1", model="siamese_sl_bilinear_pooled"
                    )["params_fold_mean"] == ""


def test_tiers_row_not_its_combo():
    """A comparison row graded from another run than the latest_combo combo
    (a SIAMESE_RESULTS_<cv> pin), or whose per-fold parameter counts are not
    the combo's, gets no combo parameter counts and is listed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _selected_marker(root, "pd:per_head", "cv1",
                         "siamese_sl/results/pinned_run_cv1")
        # An absolute marker naming the combo itself is fine.
        _selected_marker(root, "pd:per_head", "cv2", str(
            root / "siamese_sl" / "results" / "best_cat_ea_ka_pa_cv2"))
        p = root / "sl_comparison" / "results" / "comparison.csv"
        with open(p, "a", newline="") as f:
            csv.writer(f).writerow(["cv3", "siamese_sl",
                                    "trainable_params_fold", 900000, 0])
        assert _run(root) == INCOMPLETE
        tiers = _rows(_out(root) / "tab_tiers.csv")
        r1 = _one(tiers, cv="cv1", model="siamese_sl")
        assert r1["params_fold_mean"] == "" and r1["status"] == "ok"
        _close(r1["auprg_mean"], CMP["siamese_sl"]["cv1"][2])
        _close(_one(tiers, cv="cv2", model="siamese_sl")["params_fold_mean"],
               28170)
        assert _one(tiers, cv="cv3", model="siamese_sl"
                    )["params_fold_mean"] == ""
        nums = _numbers(root)
        miss = nums["missing"]
        assert any(m.startswith("siamese_sl/cv1: comparison row graded ") and
                   m.endswith("; the combo's parameter counts are not put on "
                              "its tab_tiers row, and its PCA widths / arch / "
                              "tab_loco reference row describe the combo, "
                              "not that run")
                   for m in miss), miss
        assert not any(m.startswith("siamese_sl/cv2: comparison row")
                       for m in miss), miss
        assert any(m.startswith("tab_tiers siamese_sl/cv3: "
                                "trainable_params_fold 900000") for m in miss)
        pr = nums["items"]["param_ratio"]["value"]["pd:per_head"]
        assert set(pr["per_cv"]) == {"cv2"}
        # The minimum over one tier is not the paper's '>8,600 times'.
        assert pr["min_ratio"] is None
        assert "numbers.json param_ratio pd:per_head: no ratio at cv1, cv3; " \
            "min_ratio left empty" in miss, miss


def test_tier_flags_unknown_with_a_row_missing():
    """With a peer row absent (an SLMGAE task that failed), beats_peers /
    margins 'leads' / the tier's bold are unknown, not decided against the
    peers that survived; an absent SLeuth variant leaves only the bold
    unknown."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _drop_rows(root, ("cv3", "SLMGAE"), ("cv1", "siamese_sl_pd_pooled"))
        assert _run(root) == INCOMPLETE
        tiers = _rows(_out(root) / "tab_tiers.csv")
        for r in tiers:
            if r["cv"] in ("cv1", "cv3"):
                assert r["best_auroc"] == "" and r["best_auprg"] == "", r
        for v in VARIANTS:
            row = pt.variant_row(v)
            assert _one(tiers, cv="cv3", model=row)["beats_peers_auroc"] == ""
        # All three peers at cv1: beats_peers still decided there.
        assert _one(tiers, cv="cv1", model="siamese_sl"
                    )["beats_peers_auroc"] == "False"
        assert _one(tiers, cv="cv2", model="siamese_sl"
                    )["beats_peers_auroc"] == "True"
        assert any(r["best_auroc"] == "True" for r in tiers
                   if r["cv"] == "cv2" and r["in_paper_table"] == "True")
        nums = _numbers(root)
        mg = nums["items"]["margins"]["value"]["pd:per_head"]
        a3 = mg["cv3"]["auroc"]
        assert a3["leads"] is None and a3["margin_vs_best_peer"] is None
        assert a3["best_peer"] is None
        # the margins compare against tab_tiers' peers (pt.TIER_PEERS)
        assert a3["peers_compared"] == ["ELISL", "NSF4SL", "SLGNN"]
        assert mg["cv2"]["auprg"]["leads"] is True
        assert mg["cv1"]["auroc"]["leads"] is False
        miss = nums["missing"]
        assert "tab_tiers cv3: best_* left empty (no SLMGAE row); " \
            "beats_peers_* too" in miss, miss
        assert "tab_tiers cv1: best_* left empty (no siamese_sl_pd_pooled " \
            "row)" in miss, miss
        assert any(m.startswith("numbers.json margins cv3/auroc: no SLMGAE")
                   for m in miss), miss


def test_elisl_is_an_optional_peer():
    """ELISL is run by hand (ELISL/slurm/run_elisl.sh, no orchestrator
    submits it): with no ELISL row at all, tab_tiers' beats_peers_* and the
    numbers.json margins compare SLeuth with the three required peers and
    ELISL is listed under 'Absent by design', never under Missing; with its
    row present it counts as a peer (test_tiers / the margins test)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _drop_rows(root, *[(cv, "ELISL") for cv in pt.CVS])
        _run(root)
        tiers = _rows(_out(root) / "tab_tiers.csv")
        assert not [r for r in tiers if r["model"] == "ELISL"]
        for cv in pt.CVS:
            r = _one(tiers, cv=cv, model="siamese_sl")
            if r["auprg_mean"] == "":
                continue
            decided = [r[f"beats_peers_{m}"] for m in pt.METRICS]
            assert all(x in ("True", "False") for x in decided), (cv, r)
        nums = _numbers(root)
        assert not [m for m in nums["missing"] if "ELISL" in m], \
            nums["missing"]
        for cv in pt.CVS:
            assert any(m.startswith(f"ELISL/{cv}: absent (optional peer, "
                                    f"run by hand")
                       for m in nums["absent_by_design"]), \
                nums["absent_by_design"]
        mg = nums["items"]["margins"]["value"]["pd:per_head"]
        a2 = mg["cv2"]["auprg"]
        assert a2["peers_compared"] == ["NSF4SL", "SLGNN", "SLMGAE"], a2
        assert a2["leads"] is not None and a2["best_peer"] is not None, a2


def test_tokens_per_table():
    """audit4 a8: single-modality dirs are checked against the Step-1 array
    the selection recorded, ablation dirs against the ablation run's token
    (--ablation_run_token), so an ablation rerun under a token of its own
    keeps both tables; an exported $ABLATION_RUN_TOKEN (e.g. left over from
    such a rerun) is ignored, so it cannot mark a fresh run's dirs stale.
    Without any RUN_TOKEN both fall back to the selection's token, and only
    --ablation_run_token '' turns the ablation check off."""
    import os
    res = Path("siamese_sl") / "results"
    for extra in (("--run_token", "T2"),
                  ("--run_token", TOKEN, "--ablation_run_token", "T2")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root)
            for d in (root / res).glob("ablate_no_*"):
                _w(d / "run_token.txt", "T2\n")
            # An ablation task of the rerun that never started keeps the
            # earlier run's dir (and token).
            _w(root / res / "ablate_no_kg_cv1" / "run_token.txt", TOKEN + "\n")
            with _PcaEnv():
                rc = pt.main(["--root", str(root), "--variants", *VARIANTS,
                              "--selection_variant", "pd:per_head",
                              "--bestcat_extra_variants", "0",
                              "--pca_variance", "0.8", *PIN_ARM_NONE, *extra])
            assert rc == INCOMPLETE, extra
            solo = _rows(_out(root) / "tab_solo.csv")
            assert _one(solo, variant="pd:per_head", cv="cv1", embedding="ea"
                        )["status"] == "ok"
            assert "run_token" in _one(solo, variant="pd:per_head", cv="cv2",
                                       embedding="eb")["status"]
            loco = _rows(_out(root) / "tab_loco.csv")
            assert _one(loco, variant="pd:per_head", cv="cv2",
                        dropped="ppi")["status"] == "ok", extra
            assert "run_token" in _one(loco, variant="pd:per_head", cv="cv1",
                                       dropped="kg")["status"]
            run = _numbers(root)["run"]
            assert (run["solo_run_token"], run["ablation_run_token"]) == (
                TOKEN, "T2"), extra
            assert "pca_ablation_run_token" not in run
    # A leftover export: the default (one-token) tree stays fresh, and it is
    # noted as ignored.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        with _EnvCleared("ABLATION_RUN_TOKEN"):
            os.environ["ABLATION_RUN_TOKEN"] = "LEFTOVER"
            try:
                assert _run(root) == INCOMPLETE
            finally:
                os.environ.pop("ABLATION_RUN_TOKEN", None)
        loco = _rows(_out(root) / "tab_loco.csv")
        assert _one(loco, variant="pd:per_head", cv="cv1",
                    dropped="kg")["status"] == "ok"
        nums = _numbers(root)
        assert nums["run"]["ablation_run_token"] == TOKEN
        assert any("$ABLATION_RUN_TOKEN is ignored" in n
                   for n in nums["notes"]), nums["notes"]
    # No RUN_TOKEN (a standalone run_paper_tables.sh): BOTH tables fall back
    # to the token the selection recorded, so an ablate_no_* dir an earlier
    # run left is marked stale rather than tabulated as this run's; only an
    # explicit --ablation_run_token '' leaves them unchecked.
    for extra, want_stale in (([], True), (["--ablation_run_token", ""],
                                           False)):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root)
            _w(root / res / "ablate_no_kg_cv1" / "run_token.txt", "OTHER\n")
            with _EnvCleared("RUN_TOKEN", "ABLATION_RUN_TOKEN"), _PcaEnv():
                assert pt.main(["--root", str(root), "--variants", *VARIANTS,
                                "--selection_variant", "pd:per_head",
                                "--bestcat_extra_variants", "0",
                                "--pca_variance", "0.8", *PIN_ARM_NONE,
                                *extra]) == INCOMPLETE
            solo = _rows(_out(root) / "tab_solo.csv")
            assert "run_token" in _one(solo, variant="pd:per_head", cv="cv2",
                                       embedding="eb")["status"]
            loco = _rows(_out(root) / "tab_loco.csv")
            st = _one(loco, variant="pd:per_head", cv="cv1",
                      dropped="kg")["status"]
            assert ("run_token" in st) is want_stale, (extra, st)
            run = _numbers(root)["run"]
            assert run["ablation_run_token"] == (TOKEN if want_stale
                                                 else None), extra


def test_readme_catalog_extra_without_selection():
    """The catalog's entries beyond the paper's 16 come from EMB_CATALOG, not
    from the selection item (None without best_per_category.json)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        (root / "siamese_sl" / "results" / "best_per_category.json").unlink()
        _run(root)
        readme = (_out(root) / "README.md").read_text()
        assert "(16 in the paper; extra: ea, eb, ka, kb, pa)" in readme


def test_solo_loco_missing_mean_and_nan_folds_listed():
    """A fresh results.json without a mean, or with a mean over fewer finite
    folds than the fold set has, is named (as tab_external's are)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"

        def edit(name, fm=None, mean=None):
            p = res / name / "results.json"
            r = json.loads(p.read_text())
            if fm is not None:
                r["fold_metrics"] = fm
            if mean is not None:
                r["summary"]["auprg_any_mean"] = mean
            _wj(p, r)

        edit("pcavar0.8_ea_cv1", mean=float("nan"))
        edit("pcavar0.8_eb_cv1", fm=[{"fold": 0, "auprg_any": .93},
                                     {"fold": 1, "auprg_any": float("nan")}])
        edit("ablate_no_ppi_cv2", mean=float("nan"))
        assert _run(root) == INCOMPLETE
        miss = _numbers(root)["missing"]
        for want in (
                "siamese_sl/results/pcavar0.8_ea_cv1/results.json "
                "summary.auprg_any_mean",
                "tab_solo siamese_sl cv1/eb: 1 of 2 folds finite",
                "siamese_sl/results/ablate_no_ppi_cv2/results.json "
                "summary.auprg_any_mean"):
            assert want in miss, (want, miss)
        solo = _rows(_out(root) / "tab_solo.csv")
        eb = _one(solo, variant="pd:per_head", cv="cv1", embedding="eb")
        assert eb["status"] == "ok" and eb["n_folds_finite"] == "1"
        assert _one(solo, variant="pd:per_head", cv="cv1", embedding="ea"
                    )["status"] == "missing: no auprg_any_mean"
        assert _one(_rows(_out(root) / "tab_loco.csv"), variant="pd:per_head",
                    cv="cv2", dropped="ppi")["status"] == "missing"


def test_arch_embeddings_and_lacking_category():
    """E3: a combo trained without one of the selection's winners, or a
    selection without a category (BEST_CAT_ALLOW_MISSING=1), is named."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        p = (root / "siamese_sl" / "results" /
             "var_pd_pooled__best_cat_ea_ka_pa_cv2" / "config.json")
        cfg = json.loads(p.read_text())
        cfg["embeddings_paths"] = ["../data/all_genes_ea.pt",
                                   "../data/all_genes_ka.pt"]
        _wj(p, cfg)
        _run(root)
        nums = _numbers(root)
        a = nums["items"]["arch"]["value"]
        assert a["pd:pooled"]["embeddings_match_winners"] == {
            "cv1": True, "cv2": False, "cv3": True}
        assert a["pd:per_head"]["embeddings_per_cv"]["cv1"] == [
            "all_genes_ea.pt", "all_genes_ka.pt", "all_genes_pa.pt"]
        assert any(m.startswith("pd:pooled/cv2 combo trained on "
                                "all_genes_ea.pt, all_genes_ka.pt, not")
                   for m in nums["missing"]), nums["missing"]
        assert not any(m.startswith("pd:per_head/") and "combo trained on" in m
                       for m in nums["missing"])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        p = root / "siamese_sl" / "results" / "best_per_category.json"
        sel = json.loads(p.read_text())
        for cv in pt.CVS:
            del sel[cv]["kg"]
        sel["_selection_meta"]["missing_categories"] = ["kg"]
        _wj(p, sel)
        _run(root)
        nums = _numbers(root)
        s = nums["items"]["selection"]["value"]
        assert s["missing_categories"] == ["kg"]
        assert s["differs_from_paper"]["kg"] == {"run": None,
                                                 "paper": "kg_complex"}
        assert "selection has no kg winner: every combo lacks kg" in \
            nums["missing"]


def _paper16_rows(values, stale=()):
    """tab_solo rows of every paper-16 embedding (+ esmc) at every tier:
    values(e, cv) -> auprg_any_mean; (e, cv) in stale -> not fresh. The combo
    reference is .5 at every tier."""
    out = []
    for cv in pt.CVS:
        for e, c in list(pt.PAPER_SOLO.items()) + [("esmc", "protein_seq")]:
            bad = (e, cv) in stale
            x = None if bad else values(e, cv)
            out.append({"variant": "pd:per_head", "cv": cv, "category": c,
                        "embedding": e, "in_paper": e in pt.PAPER_SOLO,
                        "status": "stale: x" if bad else "ok",
                        "auprg_any_mean": x,
                        "above_combo_shared": None if bad else x > .5,
                        "above_combo_self": None if bad else x > .5,
                        "combo_auprg_shared": .5,
                        "combo_auprg_any_self": .5})
    return out


def test_solo_ab2_ab3_unknown_with_a_paper16_member_missing():
    """AB2 / AB3 are over the paper's 16: with one of them not fresh at a
    tier, that category's paper-16 mean, the best category and the paper-16
    counts are unknown (not over the rest), and an embedding above the combo
    wherever it has a row is undecided in 'at every tier'."""
    import types
    hi = {"go2vec", "text_embed", "esm1b"}

    def values(e, cv):
        return .9 if e in hi else .1 if e != "ppi_raw" else .8

    miss = []
    ctx = types.SimpleNamespace(variants=["pd:per_head"],
                                  selection_variant="pd:per_head",
                                  miss=miss.append)
    s = pt.solo_summaries(ctx, _paper16_rows(values))["pd:per_head"]
    c1 = s["per_cv"]["cv1"]
    assert c1["paper16_absent"] == [] and not miss, miss
    assert c1["best_category_paper16"] == "text"
    assert c1["n_above_combo_shared_paper16"] == 4  # + ppi_raw
    assert s["above_combo_at_every_tier_shared"] == ["esm1b", "go2vec",
                                                     "ppi_raw", "text_embed"]
    # esm1b stale at Tier II.
    s = pt.solo_summaries(ctx, _paper16_rows(values, stale={
        ("esm1b", "cv2")}))["pd:per_head"]
    c1, c2 = s["per_cv"]["cv1"], s["per_cv"]["cv2"]
    assert c1["best_category_paper16"] == "text"
    assert c2["paper16_absent"] == ["esm1b"]
    assert c2["category_means_paper16"]["protein_seq"] is None
    _close(c2["category_means_paper16"]["text"], .5)
    assert c2["best_category_paper16"] is None
    assert c2["n_above_combo_shared_paper16"] is None
    assert c2["n_above_combo_shared"] == 3  # of those that ran
    assert s["above_combo_at_every_tier_shared"] is None
    assert s["above_combo_at_every_tier_shared_undecided"] == ["esm1b"]
    assert any(m.startswith("numbers.json solo pd:per_head/cv2: no fresh "
                            "esm1b row") for m in miss), miss
    assert any(m.startswith("numbers.json solo pd:per_head: "
                            "above_combo_at_every_tier_shared left empty")
               for m in miss), miss
    # A member stale where it is below the combo elsewhere is decided.
    miss.clear()
    s = pt.solo_summaries(ctx, _paper16_rows(values, stale={
        ("esm2", "cv2")}))["pd:per_head"]
    assert s["above_combo_at_every_tier_shared"] == ["esm1b", "go2vec",
                                                     "ppi_raw", "text_embed"]


def _rewrite(path: Path, keep) -> None:
    rows = [r for r in _rows(path) if keep(r)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_external_bold_rank_and_best_method_unknown_with_a_row_missing():
    """tab:external's bold and X5 ranks need every model's matched row (a
    failed SLMGAE task must not hand the bold to SLGNN), tab:external-ft's
    best_method all three columns."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        ext = root / "sl_comparison" / "results" / "external_transfer"
        _rewrite(ext / "zero_shot_summary.csv", lambda r: (
            r["model"], r["screen"], r["split"]) not in (
                ("SLMGAE", "Adamson", "cv1"),
                ("siamese_sl_pd_pooled", "Corn", "cv2")) or
            r["row_set"] != "matched")
        _rewrite(ext / "finetune_summary.csv", lambda r: (
            r["model"], r["screen"], r["split"], r["row_set"],
            r["method"]) != ("siamese_sl", "Adamson", "cv1",
                             "all_sldb_filtered", "Full"))
        assert _run(root) == INCOMPLETE
        rows = _rows(_out(root) / "tab_external.csv")
        for s, sp in (("Adamson", "cv1"), ("Corn", "cv2")):
            assert all(r["best_matched"] == "" for r in rows
                       if r["screen"] == s and r["split"] == sp), (s, sp)
        a2 = [r for r in rows if r["screen"] == "Adamson" and r["split"] == "cv2"]
        assert _one(a2, model="SLMGAE", row_set="matched"
                    )["best_matched"] == "True"
        miss = _numbers(root)["missing"]
        assert "tab_external Adamson/cv1: best_matched left empty (no matched " \
            "SLMGAE row)" in miss, miss
        assert "tab_external Corn/cv2: best_matched left empty (no matched " \
            "siamese_sl_pd_pooled row)" in miss, miss
        x5 = _numbers(root)["items"]["external"]["value"]["pd:per_head"][
            "X5_rank_on_matched_rows"]
        assert x5["Adamson/cv1"]["rank"] is None
        assert x5["Adamson/cv1"]["peers_compared"] == ["NSF4SL", "SLGNN"]
        assert x5["Adamson/cv2"]["rank"] == 3
        assert x5["Corn/cv2"]["rank"] == 3  # another variant absent: no matter
        assert "numbers.json external X5 Adamson/cv1: no matched SLMGAE row; " \
            "rank left empty" in miss
        ft = _rows(_out(root) / "tab_external_ft.csv")
        cell = [r for r in ft if (r["model"], r["screen"], r["split"]) == (
            "siamese_sl", "Adamson", "cv1")]
        assert all(r["best_method"] == "" for r in cell
                   if r["row_set"] == "all_sldb_filtered")
        assert _one(cell, row_set="matched", method="Full"
                    )["best_method"] == "True"
        assert "tab_external_ft siamese_sl: best_method left empty in 1 " \
            "screen/split/row_set cell(s) (no Full value)" in miss, miss


def test_external_ft_scratch_column_is_opt_in():
    """eval_finetuning's from-scratch baseline (mode Scratch): a fourth
    tab:external-ft column for a model whose finetune_summary.csv holds
    Scratch rows, required in every cell of that model only, competing for
    best_method, with numbers.json F2_F3 full_minus_scratch; a model without
    Scratch rows is tabulated exactly as before."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        ext = root / "sl_comparison" / "results" / "external_transfer"
        p = ext / "finetune_summary.csv"
        rows = _rows(p)
        extra = [dict(r, method="Scratch", spearman_mean="0.5")
                 for r in rows if r["model"] == "siamese_sl" and
                 r["method"] == "Full" and (r["screen"], r["split"],
                                            r["row_set"]) !=
                 ("Corn", "cv3", "matched")]
        # Adamson/cv2: Scratch above Full, so it takes the bold
        for r in extra:
            if (r["screen"], r["split"]) == ("Adamson", "cv2"):
                r["spearman_mean"] = "0.99"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows + extra)
        assert _run(root) == INCOMPLETE  # the one Scratch cell left out
        ft = _rows(_out(root) / "tab_external_ft.csv")
        sc = _one(ft, model="siamese_sl", screen="Adamson", split="cv1",
                  row_set="all_sldb_filtered", method="Scratch")
        assert sc["column"] == "Scratch" and sc["best_method"] == "False"
        assert _one(ft, model="siamese_sl", screen="Adamson", split="cv1",
                    row_set="all_sldb_filtered", method="Full"
                    )["best_method"] == "True"
        assert _one(ft, model="siamese_sl", screen="Adamson", split="cv2",
                    row_set="all_sldb_filtered", method="Scratch"
                    )["best_method"] == "True"
        assert not any(r["method"] == "Scratch" for r in ft
                       if r["model"] != "siamese_sl")
        nums = _numbers(root)
        miss = nums["missing"]
        assert any("tab_external_ft siamese_sl Corn/cv3/matched/Scratch "
                   "not in" in m for m in miss), miss
        assert "tab_external_ft siamese_sl: best_method left empty in 1 " \
            "screen/split/row_set cell(s) (no Scratch value)" in miss, miss
        # no other model is asked for a Scratch column
        assert not any("Scratch" in m for m in miss
                       if "siamese_sl " not in m and "siamese_sl:" not in m)
        assert any("column Scratch = the from-scratch baseline" in n
                   for n in nums["notes"]), nums["notes"]
        f2 = nums["items"]["external"]["value"]
        _close(f2["pd:per_head"]["F2_F3_all_sldb_filtered"]["Adamson/cv1"]
               ["full_minus_scratch"], .97 - .5)
        assert "full_minus_scratch" not in f2["pd:pooled"][
            "F2_F3_all_sldb_filtered"]["Adamson/cv1"]


def test_pca_width_range_needs_every_split():
    """R5's range is over all fifteen splits: a variant with a tier's combo
    absent gets no min / max, and 'equal across variants' is unknown."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        (root / "siamese_sl" / "results" /
         "var_pd_pooled__latest_combo_cv2.txt").unlink()
        _run(root)
        nums = _numbers(root)
        pw = nums["items"]["pca_width"]["value"]
        assert pw["per_variant"]["pd:pooled"]["min"] is None
        assert pw["per_variant"]["pd:pooled"]["n_splits"] == 4
        assert pw["per_variant"]["pd:per_head"]["min"] == 1505
        assert pw["equal_across_variants"] is None
        assert pw["variants_incomplete"] == ["pd:pooled"]
        assert "numbers.json pca_width pd:pooled: widths of 4 split(s) " \
            "only; min / max left empty" in nums["missing"]


def test_screens_span_splits_and_list_an_absent_screen():
    """X2 / X3 / F4 / D11: each screen is read from the first split that holds
    it (an eval task that died after the first screen leaves the others'
    splits complete), and a screen no split holds is named instead of
    silently dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"
        for v in VARIANTS:                          # cv1 died after Adamson
            c = res / f"{pt.variant_prefix(v)}best_cat_ea_ka_pa_cv3"
            for s in ("AdamsonGI", "Corn", "Gilbert"):
                (c / "eval_finetuning_cv1" / f"{s}_predictions.csv").rename(
                    c / "eval_finetuning_cv2" / f"{s}_predictions.csv")
        assert _run(root) == INCOMPLETE
        sc = _numbers(root)["items"]["screens"]["value"]
        rec = sc["per_variant"]["pd:per_head"]
        assert all(s in rec for s in pt.SCREENS), sorted(rec)
        _close(rec["Adamson"]["X2_share_both_in_genes_txt"], 15 / 28)
        assert rec["eval_dir"]["Corn"].endswith("eval_finetuning_cv2")
        assert rec["eval_dir"]["Adamson"].endswith("eval_finetuning_cv1")
        assert sc["equal_across_variants"] is True
        assert sc["variants_incomplete"] == []
        assert not [m for m in _missing(_out(root))
                    if m.startswith("numbers.json screens")]
        for v in VARIANTS:                          # now Gilbert is nowhere
            c = res / f"{pt.variant_prefix(v)}best_cat_ea_ka_pa_cv3"
            for sp in pt.SPLITS:
                p = c / f"eval_finetuning_{sp}" / "Gilbert_predictions.csv"
                if p.exists():
                    p.unlink()
        assert _run(root) == INCOMPLETE
        sc = _numbers(root)["items"]["screens"]["value"]
        assert "Gilbert" not in sc["per_variant"]["pd:per_head"]
        assert sc["equal_across_variants"] is None
        assert sc["variants_incomplete"] == VARIANTS
        assert any(m.startswith("numbers.json screens siamese_sl: no Gilbert")
                   for m in _missing(_out(root))), _missing(_out(root))
        assert "no Gilbert _predictions.csv" in (
            _out(root) / "README.md").read_text()


def test_screens_whose_rows_differ_between_variants_are_listed():
    """Every variant's eval writes the same rows of a screen, so a variant
    whose <Screen>_predictions.csv is not the others' (cut short by a partial
    copy of the results, or another run's) makes the verdict incomplete
    instead of publishing X2 / X3 / F4 / D11 computed from it - also when the
    cut keeps both shares, and also for the selection (reported) variant."""
    def screens(root):
        return ([m for m in _missing(_out(root))
                 if m.startswith("numbers.json screens")],
                _numbers(root)["items"]["screens"]["value"])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"

        def csv_of(v, screen="Adamson"):
            return (res / f"{pt.variant_prefix(v)}best_cat_ea_ka_pa_cv3" /
                    "eval_finetuning_cv1" / f"{screen}_predictions.csv")
        whole = csv_of("pd:per_head").read_text()
        lines = whole.splitlines()
        assert _run(root) == INCOMPLETE          # the fixture always is
        listed, sc = screens(root)
        assert listed == [] and sc["equal_across_variants"] is True
        for v in ("pd:pooled", "pd:per_head"):   # a line-boundary cut
            csv_of(v).write_text("\n".join(lines[:15]) + "\n")
            assert _run(root) == INCOMPLETE
            listed, sc = screens(root)
            assert len(listed) == 1, listed
            assert listed[0].startswith(
                "numbers.json screens Adamson: the variants' "
                "Adamson_predictions.csv are not the same rows"), listed
            assert f"{pt.variant_row(v)} 14 / " in listed[0], listed
            assert sc["equal_across_variants"] is False
            assert sc["per_variant"][v]["Adamson"]["n_pairs"] == 14
            csv_of(v).write_text(whole)
        # A cut that keeps X2 and X3 exactly (6 of 12 and 12 of 24 rows with
        # both genes in genes.txt, 2 of 6 and 4 of 12 of those in SynLethDB).
        genes = set(GENES)
        known = {pt.canon(*p) for p in POS + NEG}
        by = {"ov": [], "both": [], "other": []}
        for r in lines[1:]:
            a, b = r.split(",")[:2]
            both = a in genes and b in genes
            by["ov" if both and pt.canon(a, b) in known else
               "both" if both else "other"].append(r)
        half = by["ov"][:2] + by["both"][:4] + by["other"][:6]
        rest = by["ov"][2:4] + by["both"][4:8] + by["other"][6:12]
        for v in VARIANTS:
            csv_of(v).write_text("\n".join([lines[0]] + half + rest) + "\n")
        assert _run(root) == INCOMPLETE
        assert screens(root)[0] == []
        csv_of("bilinear:pooled").write_text(
            "\n".join([lines[0]] + half) + "\n")
        assert _run(root) == INCOMPLETE
        listed, sc = screens(root)
        assert len(listed) == 1 and "siamese_sl_bilinear_pooled 12 / 0.5 / " \
            in listed[0] and "siamese_sl 24 / 0.5 / " in listed[0], listed
        assert sc["equal_across_variants"] is False  # n_pairs is compared too


def test_solo_covers_the_selection_variant_outside_the_variant_list():
    """Step 1 always trains SELECTION_VARIANT's single-modality runs (block 0
    of bestcat_task_variants), so a variant list that omits it still has a
    tab:solo."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        # rc is not asserted: such a run has no per_head variant, so
        # tab:cellline is legitimately absent.
        with _PcaEnv():
            pt.main(["--root", str(root), "--variants", "bilinear:pooled",
                     "--selection_variant", "pd:per_head", "--run_token",
                     TOKEN, "--bestcat_extra_variants", "0", "--pca_variance",
                     "0.8", *PIN_ARM_NONE])
        nums = _numbers(root)
        assert nums["tables"]["tab_solo"]["status"] == "ok"
        r = _one(_rows(_out(root) / "tab_solo.csv"), variant="pd:per_head",
                 cv="cv1", embedding="ea")
        assert r["status"] == "ok" and r["selection_variant"] == "True"
        assert r["dir"] == "siamese_sl/results/pcavar0.8_ea_cv1"
        _close(r["auprg_any_mean"], SOLO["cv1"]["ea"])
        # No combo of that variant in this run: no reference is invented.
        assert r["combo_auprg_shared"] == "" and r["above_combo_shared"] == ""
        assert "pd:per_head" in nums["items"]["solo"]["value"]
        assert nums["items"]["solo"]["status"] == "ok"


def test_loco_dir_of_another_modality_set_is_stale():
    """An ablate_no_<cat> dir must be the combo MINUS that category's winner:
    its name records no modality set and the arch signature carries no
    embeddings_paths, so an ablation of an earlier selection would otherwise
    be tabulated under this run's winner."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"
        full = json.loads((res / "best_cat_ea_ka_pa_cv1" /
                           "config.json").read_text())["embeddings_paths"]
        for name in ("config.json", "results.json"):
            p = res / "ablate_no_ppi_cv1" / name
            o = json.loads(p.read_text())
            (o if name == "config.json" else o["config"])[
                "embeddings_paths"] = list(full)   # nothing removed
            _wj(p, o)
        assert _run(root) == INCOMPLETE
        r = _one(_rows(_out(root) / "tab_loco.csv"), variant="pd:per_head",
                 cv="cv1", dropped="ppi")
        assert r["status"].startswith("stale: "), r["status"]
        assert "not the combo minus ppi's winner all_genes_pa.pt" in \
            r["status"], r["status"]
        assert r["auprg_any_mean"] == "" and r["delta_vs_reference"] == ""
        lo = _numbers(root)["items"]["loco"]["value"]["pd:per_head"]["cv1"]
        assert lo["categories_missing"] == ["ppi"]
        assert lo["largest_drop"] is None


def _record_content(d: Path, shas, paths=None) -> None:
    """results.json embeddings_sha256 as train.py writes it: one digest per
    config.embeddings_paths entry, same order (paths: also rewrite that list,
    in config.json and in results.json's config)."""
    for name in ("config.json", "results.json"):
        q = d / name
        o = json.loads(q.read_text())
        cfg = o if name == "config.json" else o["config"]
        if paths is not None:
            cfg["embeddings_paths"] = list(paths)
        if name == "results.json":
            if shas is not None:
                assert len(shas) == len(cfg["embeddings_paths"]), d
                o["embeddings_sha256"] = list(shas)
            else:
                o.pop("embeddings_sha256", None)
        _wj(q, o)


def test_run_on_another_content_of_an_embedding_file_is_stale():
    """A single-modality or ablation dir trained on another CONTENT of an
    embedding file than its variant's combo is stale: the file was rebuilt in
    place between the two trainings (same name, generator stamp and gene
    order, so the token / arch / fold-stamp / modality-name checks all pass).
    Compared through results.json embeddings_sha256, by file name, on the
    files both runs list; a run that records none is not judged."""
    A, B, C, D, E = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
    f = pt.embedding_content_problem
    combo = {"dir": Path("x/best_cat_ea_ka_pa_cv1"), "results": {
        "config": {"embeddings_paths": ["/run/data/all_genes_ea.pt",
                                        "/run/data/all_genes_pa.pt"]},
        "embeddings_sha256": [A, B]}}

    def res(paths, shas):
        return {"config": {"embeddings_paths": paths},
                "embeddings_sha256": shas}

    assert f(res(["../data/all_genes_ea.pt"], [A]), combo) is None
    assert f(res(["../data/all_genes_pa.pt", "../data/all_genes_ea.pt"],
                 [B, A]), combo) is None            # by name, not position
    assert f(res(["../data/all_genes_eb.pt"], [E]), combo) is None  # unshared
    why = f(res(["../data/all_genes_ea.pt", "../data/all_genes_pa.pt"],
                [A, D]), combo)
    assert why == ("trained on another content of all_genes_pa.pt than "
                   "best_cat_ea_ka_pa_cv1 (embeddings_sha256: rebuilt in "
                   "place between the two trainings)"), why
    # nothing recorded, or a record that cannot be aligned: not judged
    assert f(res(["../data/all_genes_pa.pt"], None), combo) is None
    assert f({"config": {"embeddings_paths": ["../data/all_genes_pa.pt"]}},
             combo) is None
    assert f(res(["../data/all_genes_pa.pt", "x.pt"], [D]), combo) is None
    assert f(res("../data/all_genes_pa.pt", D), combo) is None
    assert f(res(["../data/all_genes_pa.pt"], [D]), None) is None
    assert f(res(["../data/all_genes_pa.pt"], [D]),
             {"dir": Path("c"), "results": {"config": {}}}) is None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res_dir = root / "siamese_sl" / "results"
        data = "../data/all_genes_{}.pt".format
        # the pd:per_head cv1 combo: ea, pa, ka (WINNERS order)
        _record_content(res_dir / "best_cat_ea_ka_pa_cv1", [A, B, C])
        _record_content(res_dir / "pcavar0.8_ea_cv1", [A], [data("ea")])
        _record_content(res_dir / "pcavar0.8_pa_cv1", [D], [data("pa")])
        _record_content(res_dir / "pcavar0.8_eb_cv1", [E], [data("eb")])
        _record_content(res_dir / "ablate_no_ppi_cv1", [A, C])      # ea, ka
        _record_content(res_dir / "ablate_no_kg_cv1", [A, D])       # ea, pa
        _record_content(res_dir / "ablate_no_expression_cv1", None)  # none
        # cv2: the dir records a content, its combo none: not judged
        _record_content(res_dir / "pcavar0.8_pa_cv2", [D], [data("pa")])
        assert _run(root) == INCOMPLETE
        solo = _rows(_out(root) / "tab_solo.csv")
        loco = _rows(_out(root) / "tab_loco.csv")
        want = ("trained on another content of all_genes_pa.pt than "
                "best_cat_ea_ka_pa_cv1 (embeddings_sha256")
        r = _one(solo, variant="pd:per_head", cv="cv1", embedding="pa")
        assert r["status"].startswith("stale: ") and want in r["status"], \
            r["status"]
        assert r["auprg_any_mean"] == "" and r["above_combo_self"] == ""
        for emb in ("ea", "eb"):
            assert _one(solo, variant="pd:per_head", cv="cv1",
                        embedding=emb)["status"] == "ok", emb
        assert _one(solo, variant="pd:per_head", cv="cv2",
                    embedding="pa")["status"] == "ok"
        r = _one(loco, variant="pd:per_head", cv="cv1", dropped="kg")
        assert r["status"].startswith("stale: ") and want in r["status"], \
            r["status"]
        assert r["auprg_any_mean"] == "" and r["delta_vs_reference"] == ""
        for cat in ("ppi", "expression"):
            assert _one(loco, variant="pd:per_head", cv="cv1",
                        dropped=cat)["status"] == "ok", cat
        miss = _missing(_out(root))
        for d in ("pcavar0.8_pa_cv1", "ablate_no_kg_cv1"):
            assert any(m.startswith(f"siamese_sl/results/{d} is stale (") and
                       want in m for m in miss), (d, miss)
        lo = _numbers(root)["items"]["loco"]["value"]["pd:per_head"]["cv1"]
        assert lo["categories_missing"] == ["kg"] and \
            lo["largest_drop"] is None, lo

    if torch is None:
        return
    # The verdict: a complete run that records its contents stays complete;
    # one ablation on another content of a shared file makes it incomplete.
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        res_dir = root / "siamese_sl" / "results"
        for cv in pt.CVS:
            _record_content(res_dir / f"best_cat_ea_ka_pa_{cv}", [A, B, C])
            _record_content(res_dir / f"ablate_no_ppi_{cv}", [A, C])
            _record_content(res_dir / f"ablate_no_kg_{cv}", [A, B])
            _record_content(res_dir / f"ablate_no_expression_{cv}", [B, C])
            for e, h in (("ea", A), ("pa", B), ("ka", C), ("eb", E)):
                _record_content(res_dir / f"pcavar0.8_{e}_{cv}", [h],
                                [f"../data/all_genes_{e}.pt"])
        rc, _ = _run_complete(root)
        assert rc == 0 and _missing(_out(root)) == [], _missing(_out(root))
        _record_content(res_dir / "ablate_no_kg_cv3", [A, D])
        rc, lines = _run_complete(root)
        miss = _missing(_out(root))
        assert rc == INCOMPLETE and lines[-1].startswith("PAPER INCOMPLETE")
        assert miss[0] == (
            "siamese_sl/results/ablate_no_kg_cv3 is stale (trained on another "
            "content of all_genes_pa.pt than best_cat_ea_ka_pa_cv3 "
            "(embeddings_sha256: rebuilt in place between the two "
            "trainings))"), miss
        assert all("loco pd:per_head/cv3" in m for m in miss[1:]), miss


def test_partial_fold_set_row_is_not_tabulated():
    """evaluate_model.py --allow_partial (and a control that dropped a
    degenerate fold) record summary.partial, which compare.py copies into
    comparison.csv: a partial mean must not be published as a complete one,
    nor take a bold."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        with open(root / "sl_comparison" / "results" / "comparison.csv", "a",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["cv1", "SLMGAE", "n_folds_scored", 1, ""])
            w.writerow(["cv1", "SLMGAE", "n_folds_expected", 2, ""])
            w.writerow(["cv1", "SLMGAE", "partial", "True", ""])
        assert _run(root) == INCOMPLETE
        rows = _rows(_out(root) / "tab_tiers.csv")
        r = _one(rows, cv="cv1", model="SLMGAE")
        assert r["status"] == "missing" and r["auroc_mean"] == "", r
        assert all(x["best_auroc"] == "" for x in rows if x["cv"] == "cv1")
        assert any(m.startswith("SLMGAE/cv1 is not tabulated: partial fold "
                                "set (1 of 2 folds)")
                   for m in _missing(_out(root))), _missing(_out(root))


def test_reduced_slmgae_row_is_not_tabulated():
    """train_slmgae_shared.py --allow_missing_views / --allow_static_lp_mapping
    run a REDUCED SLMGAE and record it; it must not sit in the four-model
    table under SLMGAE's name."""
    for extra, want in (
            ({"support_views": {"GO BP similarity": "ZERO-FILLED (missing x)"}},
             "ZERO-FILLED support view(s) GO BP similarity"),
            ({"lp_mapping": {"id_space": "symbol->entrez (static "
                             "gene_id_mapping.tsv fallback)"}},
             "static table")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root)
            cmp_dir = root / "sl_comparison" / "results"
            folds = root / "sl_comparison" / "folds"
            p = cmp_dir / "SLMGAE_cv3.json"
            _row_json(p, "SLMGAE", "cv3", CMP["SLMGAE"]["cv3"], folds)
            row = json.loads(p.read_text())
            row["summary"].update(extra)
            _wj(p, row)
            assert _run(root) == INCOMPLETE, extra
            r = _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv3",
                     model="SLMGAE")
            assert r["status"] == "missing" and r["auprg_mean"] == "", extra
            assert any(m.startswith("SLMGAE/cv3 is not tabulated: REDUCED "
                                    "SLMGAE") and want in m
                       for m in _missing(_out(root))), _missing(_out(root))


def test_substitute_input_peer_rows_are_not_tabulated():
    """A substitute-input peer must not sit in tab:tiers under the authors'
    name. compare.py only WARNS about these in comparison_notes.txt, which
    paper_tables does not read, so the gate has to live here."""
    cases = [
        ("SLGNN", {"arch": "port"}, "--arch port"),
        ("SLGNN", {"arch": None}, "architecture not disclosed"),
        ("NSF4SL", {"features": "embedding"}, "--features embedding"),
        ("NSF4SL", {"legacy_loop": True}, "--legacy_loop"),
        # ELISL: only its two context-free sources are the reported model
        ("ELISL", {"elisl_sources": "Sequence_1024|PPI_ec|Tissue"},
         "not the context-free"),
        ("ELISL", {"elisl_sources": None}, "sources not recorded"),
    ]
    for model, override, want in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root)
            cmp_dir = root / "sl_comparison" / "results"
            # Both copies compare.py writes: the row JSON and the CSV cell.
            p = cmp_dir / f"{model}_cv3.json"
            row = json.loads(p.read_text())
            for k, v in override.items():
                if v is None:
                    row["summary"].pop(k, None)
                else:
                    row["summary"][k] = v
            _wj(p, row)
            csv_p = cmp_dir / "comparison.csv"
            kept = []
            for r in csv.reader(csv_p.read_text().splitlines()):
                if r[:2] == ["cv3", model] and r[2] in override:
                    if override[r[2]] is None:
                        continue
                    r = [r[0], r[1], r[2], override[r[2]], ""]
                kept.append(r)
            with open(csv_p, "w", newline="") as f:
                csv.writer(f).writerows(kept)

            assert _run(root) == INCOMPLETE, (model, override)
            r = _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv3",
                     model=model)
            assert r["status"] == "missing" and r["auprg_mean"] == "", \
                (model, override, r)
            miss = _missing(_out(root))
            assert any(m.startswith(f"{model}/cv3 is not tabulated:")
                       and want in m for m in miss), (model, override, miss)
            # the other CVs, which were not touched, still publish
            assert _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1",
                        model=model)["status"] == "ok", (model, override)


def test_reduction_markers_match_the_slmgae_trainer():
    """paper_tables keeps its OWN copies of the two REDUCED-SLMGAE markers
    (SLMGAE-in-pytorch is not importable by name), and reduced_slmgae is the
    only thing keeping a --allow_missing_views / --allow_static_lp_mapping run
    out of tab:tiers. Only compare.py's copies were pinned to the trainer, so
    a reword there could leave these silently not matching."""
    src = (HERE.parent / "SLMGAE-in-pytorch" /
           "train_slmgae_shared.py").read_text()
    assert pt.LP_STATIC_FALLBACK_MARK in src, pt.LP_STATIC_FALLBACK_MARK
    assert f'f"{pt.ZERO_FILLED_VIEW} (missing ' in src, pt.ZERO_FILLED_VIEW


def test_stale_fold_sources_are_flagged():
    """A standalone run_paper_tables.sh after a cell_line_vocab.py edit (or a
    pair-file change) would otherwise publish meta.json's conflict attribution
    beside head statistics recomputed with the live vocabulary. One entry
    names every difference and the remedies (the finished run's own folds
    downloaded with its results, else a new orchestrated run; never an
    in-place regeneration, which would break every row's fold stamp), and
    items.folds_meta is marked stale."""
    def _edited_vocab(rec):
        return {**rec, "cell_line_vocab": "0" * 64}

    def _predating_vocab(rec):
        return {k: v for k, v in rec.items() if k != "cell_line_vocab"}

    for mangle, want in ((_edited_vocab, "cell_line_vocab: "),
                         (_predating_vocab, "lacks 'cell_line_vocab'")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root, mangle_sources=mangle)
            assert _run(root) == INCOMPLETE, want
            miss = _missing(_out(root))
            stale = [m for m in miss
                     if m.startswith("shared folds are STALE: ")]
            assert len(stale) == 1 and want in stale[0], (want, miss)
            assert stale[0].startswith(
                "shared folds are STALE: the fold set on disk cannot be "
                "shown to be built from the current pair files / sidecars / "
                "cell-line vocabulary ("), stale
            for say in ("these numbers are not publishable",
                        "if this tree is a copy of a finished run's results",
                        "download that run's sl_comparison/folds (and the "
                        "data/ pair files and sidecars it fingerprints) with "
                        "them instead of rerunning",
                        "otherwise rerun the orchestrator",
                        "read-only for the run",
                        "do not upload sl_comparison/folds, the data/ pair "
                        "files or siamese_sl/cell_line_vocab.py while a run "
                        "is live"):
                assert say in stale[0], (say, stale[0])
            assert "run_shared_folds.sh" not in stale[0], stale[0]
            nums = _numbers(root)
            st = nums["items"]["folds_meta"]["status"]
            assert st.startswith("stale: ") and want in st, st
            assert nums["items"]["data_stats"]["status"] == "ok"
            assert nums["verdict"].startswith("PAPER INCOMPLETE: "), \
                nums["verdict"]
            # The tables are still built: this is a disclosure, not a refusal.
            assert _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1",
                        model="SLGNN")["status"] == "ok", want
    # Fresh sources: folds_meta stays ok.
    tmp, root, out, rc = _setup()
    with tmp:
        nums = json.loads((out / "numbers.json").read_text())
        assert nums["items"]["folds_meta"]["status"] == "ok"
        assert not any("STALE" in m for m in nums["missing"])


def test_row_regraded_after_the_comparison_table_is_refused():
    """compare.py writes comparison.csv FROM the row JSONs; only
    run_compare.sh rewrites it, while a standalone rerun (run_shared_models.sh)
    rewrites the JSON alone. Two copies that disagree are the pre-rerun run's
    and this run's: neither is published."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _row_json(root / "sl_comparison" / "results" / "SLGNN_cv1.json",
                  "SLGNN", "cv1", (.10, .11, .44),
                  root / "sl_comparison" / "folds")
        assert _run(root) == INCOMPLETE
        r = _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1", model="SLGNN")
        assert r["status"] == "missing" and r["auprg_mean"] == "", r
        assert any(m.startswith("SLGNN/cv1 auroc: comparison.csv has")
                   for m in _missing(_out(root))), _missing(_out(root))
        # An agreeing JSON changes nothing.
        _row_json(root / "sl_comparison" / "results" / "SLGNN_cv1.json",
                  "SLGNN", "cv1", CMP["SLGNN"]["cv1"],
                  root / "sl_comparison" / "folds")
        assert _run(root) == INCOMPLETE
        assert _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1",
                    model="SLGNN")["status"] == "ok"


def test_external_row_of_a_variant_outside_the_run_is_not_bold():
    """A re-render with a narrower --variants tabulates the other variants'
    rows (empty variant cell) but must not decide best_matched with one."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        zs = (root / "sl_comparison" / "results" / "external_transfer" /
              "zero_shot_summary.csv")
        rows = _rows(zs)
        for r in rows:      # the unrequested variant is the best here
            if (r["model"], r["screen"], r["split"], r["row_set"]) == (
                    "siamese_sl_bilinear_pooled", "Adamson", "cv1",
                    "matched"):
                r["spearman_mean"] = "0.6"
        with open(zs, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with _PcaEnv():
            pt.main(["--root", str(root), "--variants", "pd:per_head",
                     "--selection_variant", "pd:per_head", "--run_token",
                     TOKEN, "--bestcat_extra_variants", "0", "--pca_variance",
                     "0.8", *PIN_ARM_NONE])
        c = [r for r in _rows(_out(root) / "tab_external.csv")
             if r["screen"] == "Adamson" and r["split"] == "cv1"]
        assert _one(c, model="SLMGAE", row_set="matched"
                    )["best_matched"] == "True"
        assert _one(c, model="siamese_sl_bilinear_pooled",
                    row_set="matched")["best_matched"] == ""
        assert any("siamese_sl_bilinear_pooled is not among this run's SLeuth"
                   in n for n in _numbers(root)["notes"])


def test_cellline_row_graded_from_another_ablation_is_stale():
    """zero_cell_emb.py stamps the combo it ablated into the graded row
    (source_combo): a row from another ablation must not be differenced
    against this combo's included row."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        cmp_dir = root / "sl_comparison" / "results"
        p = cmp_dir / "siamese_sl_nocell_cv1.json"
        _row_json(p, "siamese_sl_nocell", "cv1", CMP["siamese_sl_nocell"]["cv1"],
                  root / "sl_comparison" / "folds")
        row = json.loads(p.read_text())
        row["summary"]["source_combo"] = "results/best_cat_OTHER_cv1"
        _wj(p, row)
        # cv2 made clean, so the table keeps one tabulated removed row.
        agg2 = {m: [x, .01] for m, x in zip(pt.METRICS,
                                            CMP["siamese_sl"]["cv2"])}
        _wj(root / "siamese_sl" / "results" / "cellline_ablation_cv2" /
            "results.json", {**CELLLINE_OK, "agg": {"as-trained": agg2}})
        assert _run(root) == INCOMPLETE
        rem = _one(_rows(_out(root) / "tab_cellline.csv"),
                   variant="pd:per_head", cv="cv1",
                   condition="removed (cell_emb=0)")
        assert rem["status"].startswith(
            "stale: siamese_sl_nocell_cv1.json was graded from an ablation"), \
            rem["status"]
        assert "cv1" not in _numbers(root)["items"]["cellline"]["value"].get(
            "pd:per_head", {})


def _cellline_refs(fold_metrics) -> list:
    """self_check.per_fold as zero_cell_emb.py writes it: per fold, the
    combo's own results.json values under zero_cell_emb's metric names."""
    return [{"fold": m["fold"], "ok": True,
             "reference": {ours: m[key]
                           for ours, key in pt.CELLLINE_REF_KEYS.items()}}
            for m in fold_metrics]


def test_cellline_of_a_combo_retrained_in_place_is_stale():
    """The combo job replaces a combo under the SAME directory name, so
    source_model_dir still matches a retrained combo, and the 5e-3
    cross-check passes a retrain that moves the fold means by less.
    zero_cell_emb.py's results.json holds the combo's per-fold test metrics
    as it read them (self_check.per_fold[k].reference): once they are not
    the combo's results.json any more, its rows are another model's. Paired
    by 'fold'; a results.json without references is not judged."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"
        combo = res / "best_cat_ea_ka_pa_cv1" / "results.json"
        c = json.loads(combo.read_text())
        for m in c["fold_metrics"]:
            m.update(auroc_any=.90 + m["fold"] / 100,
                     aupr_any=.80 + m["fold"] / 100)
        _wj(combo, c)
        p = res / "cellline_ablation_cv1" / "results.json"
        clean = {**json.loads(p.read_text()), "self_check": {
            "ok": True,
            # listed fold 1 first: pairing must go by 'fold', not position
            "per_fold": _cellline_refs(c["fold_metrics"])[::-1]}}
        _wj(p, clean)
        # cv2 made clean (no references recorded: not judged), so the table
        # keeps one tabulated removed row.
        agg2 = {m: [x, .01] for m, x in zip(pt.METRICS,
                                            CMP["siamese_sl"]["cv2"])}
        _wj(res / "cellline_ablation_cv2" / "results.json",
            {**CELLLINE_OK, "agg": {"as-trained": agg2}})

        def cv1():
            rows = _rows(_out(root) / "tab_cellline.csv")
            return (_one(rows, variant="pd:per_head", cv="cv1",
                         condition="removed (cell_emb=0)"),
                    _one(rows, variant="pd:per_head", cv="cv1",
                         condition="included"),
                    [r for r in rows if r["variant"] == "pd:per_head" and
                     r["cv"] == "cv1" and
                     r["condition"].startswith("self-scored")])

        # The ablation of the combo on disk: tabulated.
        assert _run(root) == INCOMPLETE
        rem, inc, own = cv1()
        assert rem["status"] == "ok" and own, rem["status"]
        _close(rem["delta_auprg_vs_included"], .94 - .951)
        # The combo retrained in place: one fold's test AP moved in the 9th
        # decimal, every mean (and so the 5e-3 cross-check) as before.
        c["fold_metrics"][1]["aupr_any"] += 1e-9
        _wj(combo, c)
        assert _run(root) == INCOMPLETE
        rem, inc, own = cv1()
        assert inc["selfcheck_as_trained"] == "match", inc
        assert rem["status"].startswith(
            "stale: zero_cell_emb.py ablated siamese_sl/results/"
            "best_cat_ea_ka_pa_cv1 when its fold 1 aupr_any was "), \
            rem["status"]
        # Arm none's reruns name the arm too (2026-10-21): without PCA_ARM
        # they redo the arm the submitting shell exports.
        assert rem["status"].endswith(
            "retrained in place since: from siamese_sl/ resubmit "
            "cell_line_ablation/run_cell_line_ablation.sh with PCA_ARM=none, "
            "then from the repo root sl_comparison/slurm/run_compare.sh with "
            "PCA_ARM=none)"), rem["status"]
        assert rem["delta_auprg_vs_included"] == "" and not own
        nums = _numbers(root)
        assert "cv1" not in nums["items"]["cellline"]["value"].get(
            "pd:per_head", {})
        assert any(m.startswith("tab_cellline siamese_sl/cv1: zero_cell_emb"
                                ".py ablated siamese_sl/results/"
                                "best_cat_ea_ka_pa_cv1 when ")
                   for m in nums["missing"]), nums["missing"]
        # An earlier reason keeps its place and wording: another directory.
        _wj(p, {**clean, "source_model_dir": "results/best_cat_ea_pa_cv1"})
        assert _run(root) == INCOMPLETE
        assert "ablated results/best_cat_ea_pa_cv1, not the latest_combo " \
            "combo" in cv1()[0]["status"], cv1()[0]["status"]
    # The function alone: what is and is not judged.
    c = {"dir": Path("results/best_cat_x_cv1"), "results": {"fold_metrics": [
        {"fold": 0, "auroc_any": .9, "aupr_any": .8, "auprg_any": .7},
        {"fold": 1, "auroc_any": .91, "aupr_any": .81, "auprg_any": .71}]}}
    refs = _cellline_refs(c["results"]["fold_metrics"])
    ok = {"self_check": {"ok": True, "per_fold": refs}}
    prob = pt.cellline_reference_problem
    assert prob(ok, c) is None
    # the combo's own list in another order: its 'fold' decides, not position
    assert prob(ok, {**c, "results": {
        "fold_metrics": c["results"]["fold_metrics"][::-1]}}) is None
    assert prob(None, c) is None and prob(ok, None) is None
    assert prob({"self_check": {"ok": True}}, c) is None       # no per_fold
    assert prob({"self_check": {"ok": True, "per_fold": [
        {"fold": 0, "ok": None}]}}, c) is None                 # no reference
    # fold_metrics without 'fold' (an older train.py) pair by position
    old = {"dir": c["dir"], "results": {"fold_metrics": [
        {k: v for k, v in m.items() if k != "fold"}
        for m in c["results"]["fold_metrics"]]}}
    assert prob(ok, old) is None
    assert "fold 0 auroc_any was 0.9; its results.json now has 0.91" in prob(
        ok, {**old, "results": {
            "fold_metrics": old["results"]["fold_metrics"][::-1]}})
    # a fold or a key the combo no longer records
    assert "fold 1 auroc_any" in prob(ok, {"dir": c["dir"], "results": {
        "fold_metrics": c["results"]["fold_metrics"][:1]}})
    assert "fold 0 auprg_any was 0.7; its results.json now has None" in prob(
        ok, {"dir": c["dir"], "results": {"fold_metrics": [
            {k: v for k, v in m.items() if k != "auprg_any"}
            for m in c["results"]["fold_metrics"]]}})


def test_cellline_reference_keys_are_zero_cell_embs():
    """paper_tables keeps its OWN copy of zero_cell_emb.SELFCHECK_KEYS
    (importing that script needs torch and the siamese tree), and
    cellline_reference_problem compares what zero_cell_emb.py copies under
    them: a key renamed there would turn every clean ablation stale, a
    reference rounded or dropped there would switch the check off."""
    import ast
    src = (HERE.parent / "siamese_sl" / "cell_line_ablation" /
           "zero_cell_emb.py").read_text()
    keys = [ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Assign) and
            [getattr(t, "id", None) for t in n.targets] == ["SELFCHECK_KEYS"]]
    assert keys == [pt.CELLLINE_REF_KEYS], keys
    # copied as read (no rounding), under the name read here
    assert 'fc["reference"] = ref[k]' in src
    assert "ref[k] = {ours: m.get(key) for ours, key in " \
        "SELFCHECK_KEYS.items()}" in src


def test_single_output_run_lists_no_cellline_or_head_inputs():
    """USE_CELL_LINES=0 collapses the list to pd:per_head, but that combo is
    the single-output SiameseSL: there is no cell-line embedding to remove and
    the eval records no head, so neither is reported as a missing input."""
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        res = root / "siamese_sl" / "results"
        for p in res.glob("*best_cat_ea_ka_pa_cv*/config.json"):
            cfg = json.loads(p.read_text())
            cfg["use_cell_lines"] = False
            _wj(p, cfg)
        for p in res.glob("*best_cat_ea_ka_pa_cv3/eval_finetuning_*/"
                          "config_used.json"):
            c = json.loads(p.read_text())
            c["per_head"] = False
            c["effective_head"] = dict.fromkeys(pt.SCREENS)
            _wj(p, c)
        for cv in pt.CVS:
            shutil.rmtree(res / f"cellline_ablation_{cv}", ignore_errors=True)
        # 1: tab:cellline / CL1 cannot be produced by such a run at all.
        with _PcaEnv():
            assert pt.main(["--root", str(root), "--variants", "pd:per_head",
                            "--selection_variant", "pd:per_head",
                            "--run_token", TOKEN, "--bestcat_extra_variants",
                            "0", "--pca_variance", "0.8",
                            *PIN_ARM_NONE]) == 1
        miss = _missing(_out(root))
        assert not [m for m in miss if "cellline_ablation" in m or
                    "_nocell" in m or "effective_head" in m], miss
        assert any("no variant of this run has a cell-line embedding" in n
                   for n in _numbers(root)["notes"])
        # ...and M7 / M9 report no cell-line width for it either: config.json
        # keeps --cell_line_dim's unused default (train.py dumps vars(args)).
        assert _numbers(root)["items"]["heads"]["value"][
            "cell_line_dim_per_variant"] == {}
        r = _one([x for x in _rows(_out(root) / "tab_external.csv")
                  if x["screen"] == "Corn" and x["split"] == "cv2"],
                 model="siamese_sl", row_set="matched")
        assert r["scored_head"] == "n/a (single output)"


def test_a_variant_without_any_screen_is_incomplete_not_equal():
    """A variant whose three eval tasks all died writes no screen CSV at all,
    so it has no record here: the completeness gate must be taken from the
    run's variants, not from the records that survived (X2 / X3 'equal across
    variants' would otherwise assert a comparison never made)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        c = (root / "siamese_sl" / "results" /
             f"{pt.variant_prefix('bilinear:per_head')}best_cat_ea_ka_pa_cv3")
        for p in sorted(c.glob("eval_finetuning_*/*_predictions.csv")):
            p.unlink()
        assert _run(root) == INCOMPLETE
        sc = _numbers(root)["items"]["screens"]["value"]
        assert "bilinear:per_head" not in sc["per_variant"]
        assert sc["variants_incomplete"] == ["bilinear:per_head"]
        assert sc["equal_across_variants"] is None
        assert any(m.startswith("numbers.json screens "
                                "siamese_sl_bilinear_per_head: no Adamson, "
                                "AdamsonGI, Corn, Gilbert")
                   for m in _missing(_out(root))), _missing(_out(root))


def test_comparison_row_without_its_json_is_not_tabulated():
    """run_shared_models.sh / run_shared_folds.sh / grade.sh move this task's
    previous row aside BEFORE anything can fail ('a failed task must leave its
    row MISSING') and none of them rewrites comparison.csv: a row of the
    default table with no JSON beside it is the pre-rerun number. The
    documented --comparison_csv override, whose rows need not have a JSON
    beside them, is unaffected."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        cmp_dir = root / "sl_comparison" / "results"
        (cmp_dir / "SLMGAE_cv1.json").unlink()      # as the aside-move leaves it
        assert _run(root) == INCOMPLETE
        rows = _rows(_out(root) / "tab_tiers.csv")
        r = _one(rows, cv="cv1", model="SLMGAE")
        assert r["status"] == "missing" and r["auroc_mean"] == "", r
        # ...so the tier's bold is unknown, not handed to the next model.
        assert all(x["best_auroc"] == "" for x in rows if x["cv"] == "cv1")
        assert any(m.startswith("SLMGAE/cv1: comparison.csv has a row but no "
                                "evaluate_model JSON beside it")
                   for m in _missing(_out(root))), _missing(_out(root))
        # The same table passed with --comparison_csv still tabulates it.
        other = root / "comparison_copy.csv"
        other.write_text((cmp_dir / "comparison.csv").read_text())
        assert _run(root, "--comparison_csv", str(other)) == INCOMPLETE
        assert _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1",
                    model="SLMGAE")["status"] == "ok"


KB_MISSING = ("data/all_genes_kb.pt not generated: tab_solo kb rows absent "
              "(not among the paper's 16)")
KB_BY_DESIGN = ("data/all_genes_kb.pt not generated (OPTIONAL_TYPES, "
                "siamese_sl/slurm/run_generate_embeddings.conf): tab_solo kb "
                "rows absent (not among the paper's 16)")


def _section(readme: str, title: str) -> str:
    """The body of README.md's '## <title>' section."""
    return readme.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def test_missing_control_row_is_listed():
    """A DegreePrior task that failed leaves the control row missing: it is
    listed (as the control, which decides no bold), not silently dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _drop_rows(root, ("cv1", "DegreePrior"), ("cv2", "SLGNN"))
        assert _run(root) == INCOMPLETE
        rows = _rows(_out(root) / "tab_tiers.csv")
        assert _one(rows, cv="cv1", model="DegreePrior")["status"] == "missing"
        # The bold is over the paper's rows only: still decided at cv1.
        assert _one(rows, cv="cv1", model="SLMGAE")["best_auroc"] == "True"
        miss = _missing(_out(root))
        assert "tab_tiers DegreePrior/cv1 (no usable comparison row; " \
            "control, never bolded)" in miss, miss
        assert "tab_tiers SLGNN/cv2 (no usable comparison row)" in miss, miss
        assert not any(m.startswith("tab_tiers cv1: best_*") for m in miss)
        assert not any(m.startswith(("tab_tiers DegreePrior/cv2",
                                     "tab_tiers DegreePrior/cv3"))
                       for m in miss), miss


def test_external_peer_without_a_tabulated_tiers_row_is_dropped():
    """A peer whose model-CV (cv3) comparison row tab:tiers refuses (its
    evaluate_model JSON moved aside by a rerun, or graded on another fold
    set) is not bolded or ranked in tab:external / X5 either, although its
    row_usable record (arch, features) is clean."""
    def no_json(root):
        (root / "sl_comparison" / "results" / "SLGNN_cv3.json").unlink()

    def foreign_folds(root):
        p = root / "sl_comparison" / "results" / "comparison.csv"
        c = _rows(p)
        for r in c:
            if (r["cv"], r["model"], r["metric"]) == (
                    "cv3", "SLGNN", "folds_meta.json_sha256"):
                r["mean"] = "0" * 16
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(c[0]))
            w.writeheader()
            w.writerows(c)

    for edit, why in (
            (no_json, "SLGNN/cv3: comparison.csv has a row but no "
                      "evaluate_model JSON beside it"),
            (foreign_folds, "SLGNN/cv3 (comparison.csv) is not on the "
                            "current folds")):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build(root)
            edit(root)
            assert _run(root) == INCOMPLETE, why
            assert _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv3",
                        model="SLGNN")["status"] == "missing", why
            ext = _rows(_out(root) / "tab_external.csv")
            assert not [r for r in ext if r["model"] == "SLGNN"], why
            assert [r for r in ext if r["model"] == "SLMGAE"], why
            assert all(r["best_matched"] == "" for r in ext), why
            miss = _missing(_out(root))
            assert any(m.startswith(why) for m in miss), (why, miss)
            dropped = ("tab_external: SLGNN rows dropped (its cv3 row is not "
                       "tabulated; see the reason above)")
            assert dropped in miss, miss
            # Its rows ARE in the CSV: named as dropped, after the reasons.
            assert "tab_external SLGNN: dropped (see above)" in miss, miss
            assert not any(m.startswith("tab_external SLGNN not in")
                           for m in miss), miss
            reason = next(i for i, m in enumerate(miss) if m.startswith(why))
            assert reason < miss.index(dropped) < miss.index(
                "tab_external SLGNN: dropped (see above)"), miss
            assert "tab_external Adamson/cv1: best_matched left empty (no " \
                "matched SLGNN row)" in miss, miss
            x5 = _numbers(root)["items"]["external"]["value"]["pd:per_head"][
                "X5_rank_on_matched_rows"]
            assert all(c["rank"] is None for c in x5.values()), x5
            assert x5["Adamson/cv1"]["peers_compared"] == ["NSF4SL",
                                                           "SLMGAE"], x5
            assert "numbers.json external X5 Adamson/cv1: no matched SLGNN " \
                "row; rank left empty" in miss, miss


def test_absent_by_design_is_listed_apart():
    """A catalog embedding outside the paper's 16 that OPTIONAL_TYPES accepts
    as not generated, and the non-selection variants' tab_solo cells under
    BESTCAT_EXTRA_VARIANTS=0 (listed nowhere before), are intended: they go
    to 'Absent by design' (numbers.json absent_by_design), not Missing. A
    required type, a paper-16 one whatever OPTIONAL_TYPES says, and every
    type when the conf is unreadable stay missing."""
    not_run = [f"tab_solo {v}: 12 cell(s) not_run (the single-modality runs "
               f"of variants other than SELECTION_VARIANT pd:per_head are "
               f"off: BESTCAT_EXTRA_VARIANTS=0)"
               for v in VARIANTS if v != "pd:per_head"]
    tmp, root, out, rc = _setup()          # kb is required here
    with tmp:
        nums = json.loads((out / "numbers.json").read_text())
        assert KB_MISSING in nums["missing"], nums["missing"]
        assert nums["absent_by_design"] == not_run, nums["absent_by_design"]
        assert not any("not_run" in m for m in nums["missing"])
        sec = _section((out / "README.md").read_text(), "Absent by design")
        assert all(f"- {m}\n" in sec for m in not_run), sec
        assert "kb" not in sec, sec
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _optional_types(root, *OPTIONAL, "kb")
        assert _run(root) == INCOMPLETE
        nums = _numbers(root)
        assert nums["absent_by_design"] == [KB_BY_DESIGN] + not_run, \
            nums["absent_by_design"]
        assert not any("all_genes_kb.pt" in m for m in nums["missing"])
        readme = (_out(root) / "README.md").read_text()
        assert f"- {KB_BY_DESIGN}\n" in _section(readme, "Absent by design")
        assert "all_genes_kb.pt" not in _section(readme,
                                                 "Missing inputs and items")
        # The cell itself is unchanged.
        assert _one(_rows(_out(root) / "tab_solo.csv"), variant="pd:per_head",
                    cv="cv1", embedding="kb")["status"] == "not_generated"
        # A paper-16 embedding stays missing whatever OPTIONAL_TYPES says.
        cat = CATALOG + [("esm2", "protein_seq")]
        _w(root / "siamese_sl" / "slurm" / "run_best_per_category.conf",
           "EMB_CATALOG=(\n" + "\n".join(f'    "{e}:all_genes_{e}.pt:{c}"'
                                          for e, c in cat) + "\n)\n")
        _optional_types(root, *OPTIONAL, "kb", "esm2")
        _run(root)
        nums = _numbers(root)
        assert "data/all_genes_esm2.pt not generated: tab_solo esm2 rows " \
            "absent" in nums["missing"], nums["missing"]
        assert not any("esm2" in m for m in nums["absent_by_design"])
        # Unreadable conf: no type is optional (noted).
        (root / "siamese_sl" / "slurm" / "run_generate_embeddings.conf"
         ).unlink()
        _run(root)
        nums = _numbers(root)
        assert KB_MISSING in nums["missing"], nums["missing"]
        assert nums["absent_by_design"] == not_run, nums["absent_by_design"]
        assert any(n.startswith("OPTIONAL_TYPES not read") and
                   "run_generate_embeddings.conf" in n and
                   str(root) not in n for n in nums["notes"]), nums["notes"]


def _complete_tree(root: Path, mangle_sources=None) -> None:
    """The fixture made complete for a pd:per_head-only run (_run_complete,
    under _FixturePaper16): its two stale single-modality dirs refreshed, a
    clean cell-line ablation at every tier, kb an OPTIONAL_TYPES absence."""
    _build(root, mangle_sources)
    folds = root / "sl_comparison" / "folds"
    res = root / "siamese_sl" / "results"
    _w(res / "pcavar0.8_eb_cv2" / "run_token.txt", TOKEN + "\n")
    _siamese_dir(res / "pcavar0.8_pa_cv3", _cfg("pd:per_head", "cv3",
                                                 solo=True),
                 "cv3", folds, SOLO["cv3"]["pa"],
                 [{"fold": k, "auprg_any": SOLO["cv3"]["pa"]}
                  for k in range(2)])
    for cv in ("cv2", "cv3"):
        agg = {m: [x, .01] for m, x in zip(pt.METRICS, CMP["siamese_sl"][cv])}
        _wj(res / f"cellline_ablation_{cv}" / "results.json",
            {**CELLLINE_OK, "agg": {"as-trained": agg}})
    _optional_types(root, *OPTIONAL, "kb")


class _FixturePaper16:
    """pt.PAPER_SOLO set to the fixture's four generated embeddings for the
    block: the fixture catalog holds none of the paper's 16, so no fixture
    run is complete otherwise."""

    def __enter__(self):
        self.saved = pt.PAPER_SOLO
        pt.PAPER_SOLO = {e: c for e, c in CATALOG if e != "kb"}

    def __exit__(self, *exc):
        pt.PAPER_SOLO = self.saved


def _run_complete(root: Path, *extra, pin_manifest: bool = True):
    """(exit status, lines printed) of a pd:per_head-only run; no code
    manifest unless extra passes one (pin_manifest=False: $CODE_MANIFEST)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), _PcaEnv():
        rc = pt.main(["--root", str(root), "--variants", "pd:per_head",
                      "--selection_variant", "pd:per_head", "--run_token",
                      TOKEN, "--bestcat_extra_variants", "0",
                      "--pca_variance", "0.8", "--pca_arm", "none",
                      "--pca_arms", *ARMS] +
                     (["--code_manifest", ""] if pin_manifest else []) +
                     list(extra))
    return rc, buf.getvalue().splitlines()


def test_verdict_and_exit_status():
    """PAPER COMPLETE (exit 0) only with nothing under Missing; a run whose
    tables are all built but that lists something exits 3 (not argparse's
    2), one with a table that has no usable source 1. The verdict is the
    last line printed, README.md's first line under the title (line 3) and
    numbers.json 'verdict'."""
    if torch is None:  # AP1 / AP2 unread: no run is complete without torch
        with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
            _complete_tree(Path(tmp))
            rc, lines = _run_complete(Path(tmp))
            assert rc == INCOMPLETE and _missing(_out(Path(tmp))) == \
                [NO_TORCH], _missing(_out(Path(tmp)))
        return

    def check(root, lines, want):
        nums = _numbers(root)
        assert nums["verdict"] == want, (nums["verdict"], want)
        readme = (_out(root) / "README.md").read_text()
        assert readme.splitlines()[:3] == [
            "# Paper tables and numbers: PCA arm none", "", want], \
            readme.splitlines()[:3]
        assert lines[-1] == want, lines[-3:]
        return nums, readme

    with _FixturePaper16():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_tree(root)
            rc, lines = _run_complete(root)
            nums, readme = check(root, lines, pt.VERDICT_COMPLETE)
            assert rc == 0 and nums["missing"] == [], nums["missing"]
            assert _section(readme, "Missing inputs and items") == \
                "\n- none\n", _section(readme, "Missing inputs and items")
            # ...with the intended absence still listed, apart.
            assert nums["absent_by_design"] == [KB_BY_DESIGN]
            assert nums["items"]["folds_meta"]["status"] == "ok"
            # One item: the control row a failed DegreePrior task leaves.
            _drop_rows(root, ("cv1", "DegreePrior"))
            rc, lines = _run_complete(root)
            item = ("tab_tiers DegreePrior/cv1 (no usable comparison row; "
                    "control, never bolded)")
            nums, _ = check(root, lines,
                            f"PAPER INCOMPLETE: 1 item(s) ({item})")
            assert rc == INCOMPLETE and nums["missing"] == [item]
            # A table without a usable source: 1, as before.
            (root / "sl_comparison" / "results" / "external_transfer" /
             "zero_shot_summary.csv").unlink()
            rc, lines = _run_complete(root)
            nums = _numbers(root)
            assert rc == 1
            assert "tab_external.csv (no usable source)" in nums["missing"]
            check(root, lines, nums["verdict"])
            assert nums["verdict"].startswith(
                f"PAPER INCOMPLETE: {len(nums['missing'])} item(s) ("), \
                nums["verdict"]
            assert lines[-2] == "ERROR: table(s) not built: tab_external", \
                lines[-3:]
        # Stale fold sources alone: incomplete, items.folds_meta stale.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_tree(root, mangle_sources=lambda rec: {
                **rec, "cell_line_vocab": "0" * 64})
            rc, lines = _run_complete(root)
            nums = _numbers(root)
            assert rc == INCOMPLETE and len(nums["missing"]) == 1, \
                nums["missing"]
            assert nums["missing"][0].startswith("shared folds are STALE: ")
            check(root, lines,
                  f"PAPER INCOMPLETE: 1 item(s) ({nums['missing'][0]})")
            assert nums["items"]["folds_meta"]["status"].startswith(
                "stale: cell_line_vocab: "), nums["items"]["folds_meta"]
    # The fixture tree: more than three items, the first three named.
    tmp, root, out, rc = _setup()
    with tmp:
        miss = _missing(out)
        assert rc == INCOMPLETE and len(miss) > 3, (rc, miss)
        want = (f"PAPER INCOMPLETE: {len(miss)} item(s) "
                f"({'; '.join(miss[:3])}; ...)")
        assert _numbers(root)["verdict"] == want
        assert (out / "README.md").read_text().splitlines()[2] == want


def test_unloadable_arch_signature_is_missing():
    """Without siamese_sl/arch_signature.py no single-modality or ablation
    dir can be shown to be its combo's architecture, nor a combo to be the
    arm's PCA: listed under Missing (the verdict is incomplete), not only
    noted, with nothing listed beyond that and every table still built. The
    cross-arm table compares the arms' PCA (pca_arm / pca_settings live
    there): not built at all (exit 1)."""
    load = pt._optional_module

    def no_arch(name, path):
        return None if Path(path).name == "arch_signature.py" else \
            load(name, path)

    want = ("siamese_sl/arch_signature.py not loadable: tab_solo / tab_loco "
            "arch signatures and the combos' PCA arm unchecked")
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        pt._optional_module = no_arch
        try:
            rc, lines = _run_complete(root)
        finally:
            pt._optional_module = load
        nums = _numbers(root)
        miss = nums["missing"]
        assert rc == INCOMPLETE and miss == [want] + NOTORCH, miss
        assert want not in nums["notes"], nums["notes"]
        assert all(t["status"] == "ok" for t in nums["tables"].values()), \
            nums["tables"]
        assert nums["run"]["pca_settings"] == {"modality": None,
                                               "joint": None}
        assert lines[-1] == (f"PAPER INCOMPLETE: {len(miss)} item(s) "
                             f"({'; '.join(miss[:3])}"
                             f"{'; ...' if len(miss) > 3 else ''})"), lines[-1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root, ("none", "joint"))
        pt._optional_module = no_arch
        try:
            rc = _run_arms(root, arms=("none", "joint"))
        finally:
            pt._optional_module = load
        nums = json.loads((_arms_out(root) / "numbers.json").read_text())
        assert rc == 1 and nums["tables"]["tab_pca"] == {
            "status": "missing", "rows": 0}, (rc, nums["tables"])
        assert nums["missing"] == [want, "tab_pca.csv (no usable source)",
                                   "numbers.json pca: empty (no input)"], \
            nums["missing"]
        assert not (_arms_out(root) / "tab_pca.csv").exists()


def test_inputs_without_torch_or_a_paper16_width_are_missing():
    """AP1 / AP2 quote the paper's 16 native widths: torch not importable, or
    a paper-16 file whose width is not read (no 2-D matrix, unreadable), is
    listed under Missing; outside the 16 an unreadable file is only noted."""
    inputs = pt.item_inputs

    def no_torch(ctx):
        absent = object()   # restore exactly: absent is not a None entry
        saved = sys.modules.get("torch", absent)
        sys.modules["torch"] = None  # `import torch` raises ImportError
        try:
            return inputs(ctx)
        finally:
            if saved is absent:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = saved

    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        pt.item_inputs = no_torch
        try:
            rc, _ = _run_complete(root)
        finally:
            pt.item_inputs = inputs
        nums = _numbers(root)
        assert rc == INCOMPLETE and nums["missing"] == [NO_TORCH], \
            nums["missing"]
        assert nums["items"]["inputs"]["value"]["native_dims"] is None
        if torch is None:
            return
        torch.save({"gene_order": GENES[:3]},          # no matrix
                   root / "data" / "all_genes_pa.pt")
        _w(root / "data" / "all_genes_eb.pt", "")      # unreadable
        rc, _ = _run_complete(root)
        nums = _numbers(root)
        assert rc == INCOMPLETE and nums["missing"] == [
            f"native width of data/all_genes_{e}.pt not read (AP1/AP2)"
            for e in ("eb", "pa")], nums["missing"]
        assert nums["items"]["inputs"]["value"]["native_dims"] == {
            **WIDTH, "eb": None, "pa": None}
        assert any(n.startswith("native width of data/all_genes_eb.pt "
                                "unreadable: ") for n in nums["notes"])
    with tempfile.TemporaryDirectory() as tmp:     # none of the paper's 16
        root = Path(tmp)
        _build(root)
        _w(root / "data" / "all_genes_eb.pt", "")
        assert _run(root) == INCOMPLETE
        nums = _numbers(root)
        assert not any(m.startswith("native width")
                       for m in nums["missing"]), nums["missing"]
        assert any(n.startswith("native width of data/all_genes_eb.pt "
                                "unreadable: ") for n in nums["notes"])


def test_paper_row_without_params_sd_or_seed_is_missing():
    """The paper prints each tab:tiers cell as mean (sd), a parameter count
    per peer and one training seed (D3): a tabulated paper row lacking one is
    listed (its values are still written). The control's sd, DegreePrior's
    and the _nocell rows' seeds (neither is a training run) and a variant
    outside the run are not asked for."""
    no_sd = {("cv3", "siamese_sl", "auroc"), ("cv3", "siamese_sl", "auprg"),
             ("cv2", "NSF4SL", "ap"), ("cv1", "DegreePrior", "auroc")}
    gone = {("cv1", "SLGNN", "trainable_params"), ("cv2", "SLMGAE", "seed"),
            ("cv3", "siamese_sl", "seed"),
            ("cv1", "siamese_sl_pd_pooled", "seed")}   # not in this run
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        p = root / "sl_comparison" / "results" / "comparison.csv"
        rows = [r for r in _rows(p)
                if (r["cv"], r["model"], r["metric"]) not in gone]
        for r in rows:
            if (r["cv"], r["model"], r["metric"]) in no_sd:
                r["std"] = ""
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        rc, _ = _run_complete(root)
        nums = _numbers(root)
        assert rc == INCOMPLETE and nums["missing"] == [
            "tab_tiers SLGNN/cv1: no trainable_params (the paper's parameter "
            "column)",
            "tab_tiers NSF4SL/cv2: no sd for ap (the paper prints mean (sd))",
            "tab_tiers siamese_sl/cv3: no sd for auroc, auprg (the paper "
            "prints mean (sd))",
            "numbers.json inputs SLMGAE/cv2: no training seed recorded (D3)",
            "numbers.json inputs siamese_sl/cv3: no training seed recorded "
            "(D3)"] + NOTORCH, nums["missing"]
        tiers = _rows(_out(root) / "tab_tiers.csv")
        r = _one(tiers, cv="cv3", model="siamese_sl")
        assert r["status"] == "ok" and r["auroc_sd"] == "", r
        _close(r["auroc_mean"], CMP["siamese_sl"]["cv3"][0])
        assert _one(tiers, cv="cv1", model="SLGNN")["trainable_params"] == ""
        assert {"SLMGAE/cv2", "siamese_sl/cv3", "siamese_sl_pd_pooled/cv1",
                "DegreePrior/cv1", "siamese_sl_nocell/cv1"} <= set(
            nums["items"]["inputs"]["value"]["rows_without_seed"])


def test_check_grid_names_a_dropped_model_apart():
    """A model drop_unusable_peers dropped is in the CSV: named as dropped
    (its reason is listed above), not as absent from the file; a model that
    is absent, and an absent cell, still are."""
    import types
    miss = []
    pt.check_grid(types.SimpleNamespace(miss=miss.append), "tab_x",
                  [{"model": "A", "k": "a"}], ("k", ),
                  {"A": [("a", ), ("b", )], "B": [("a", )], "C": [("a", )]},
                  Path("x.csv"), {"C"})
    assert miss == ["tab_x A b not in x.csv", "tab_x B not in x.csv",
                    "tab_x C: dropped (see above)"], miss


def _code_files(root: Path) -> None:
    """Code in code_manifest.CODE_DIRS' layout (the fixture's
    siamese_sl/slurm/*.conf are in scope too)."""
    for rel, text in (("siamese_sl/train.py", "print('train')\n"),
                      ("siamese_sl/slurm/run_step.sh", "#!/bin/bash\n"),
                      ("sl_comparison/compare.py", "print('compare')\n"),
                      ("sl_comparison/slurm/run_compare.sh", "#!/bin/bash\n")):
        _w(root / rel, text)


def _code_manifest(root: Path) -> Path:
    """The manifest the orchestrator writes at submission: code_manifest.py's
    '<sha256>  <path>' lines, in sl_comparison/slurm/logs/."""
    cm = pt._load_module("_t_code_manifest", HERE / "code_manifest.py")
    p = root / "sl_comparison" / "slurm" / "logs" / f"code_{TOKEN}_0.sha256"
    _w(p, "".join(f"{d}  {rel}\n" for rel, d in cm.manifest(root).items()))
    return p


def test_code_drift_since_submission():
    """The orchestrator hashes the code its jobs run into CODE_MANIFEST at
    submission and the jobs start over days: code changed, added or removed
    by the end of the DAG is one Missing entry (the first five files named,
    all in numbers.json run.code_drift). No change adds nothing, a run
    without a manifest (standalone) only notes it, and an unreadable
    manifest or code_manifest.py is missing."""
    import os
    ok = 0 if torch is not None else INCOMPLETE     # AP1 / AP2 need torch
    tail = ("jobs of this run may have run different versions; check them "
            "before publishing")
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        _code_files(root)
        man = _code_manifest(root)
        assert "siamese_sl/train.py" in man.read_text()
        rc, _ = _run_complete(root, "--code_manifest", str(man))
        nums = _numbers(root)
        assert rc == ok and nums["missing"] == NOTORCH, nums["missing"]
        assert nums["run"]["code_manifest"] == str(man.resolve())
        assert nums["run"]["code_drift"] == []
        assert not any("code manifest" in n for n in nums["notes"])
        # The orchestrator's export is the default (and this run's own
        # outputs, written above, are not code).
        with _EnvCleared("CODE_MANIFEST"):
            os.environ["CODE_MANIFEST"] = str(man)
            try:
                rc, _ = _run_complete(root, pin_manifest=False)
            finally:
                os.environ.pop("CODE_MANIFEST", None)
        assert rc == ok and _numbers(root)["run"]["code_drift"] == []
        # One file uploaded after the submission.
        _w(root / "siamese_sl" / "train.py", "print('train v2')\n")
        rc, lines = _run_complete(root, "--code_manifest", str(man))
        nums = _numbers(root)
        want = (f"code changed after submission (1 file(s): siamese_sl/"
                f"train.py (changed)): {tail}")
        assert rc == INCOMPLETE and nums["missing"] == [want] + NOTORCH, \
            nums["missing"]
        assert nums["run"]["code_drift"] == ["siamese_sl/train.py (changed)"]
        if torch is not None:
            assert lines[-1] == f"PAPER INCOMPLETE: 1 item(s) ({want})", \
                lines[-1]
        assert "; code changed, added or removed since: 1 file(s) " \
            "(numbers.json run.code_drift)" in (
                _out(root) / "README.md").read_text()
        # More than five: the first five named, every one recorded.
        (root / "sl_comparison" / "compare.py").unlink()
        for k in range(4):
            _w(root / "sl_comparison" / "slurm" / f"new_{k}.sh", "")
        _run_complete(root, "--code_manifest", str(man))
        nums = _numbers(root)
        drift = ["siamese_sl/train.py (changed)",
                 "sl_comparison/compare.py (removed)"] + [
                     f"sl_comparison/slurm/new_{k}.sh (new)" for k in range(4)]
        assert nums["run"]["code_drift"] == drift, nums["run"]["code_drift"]
        assert nums["missing"] == [
            f"code changed after submission (6 file(s): "
            f"{', '.join(drift[:5])}, ...): {tail}"] + NOTORCH, \
            nums["missing"]
        # No manifest (a standalone run): noted, the verdict unaffected.
        rc, _ = _run_complete(root)
        nums = _numbers(root)
        assert rc == ok and nums["missing"] == NOTORCH, nums["missing"]
        assert "no code manifest (not an orchestrated run): code drift not " \
            "checked" in nums["notes"], nums["notes"]
        assert (nums["run"]["code_manifest"], nums["run"]["code_drift"]) == \
            (None, None)
        # A manifest that cannot be read: missing, drift not checked.
        _w(root / "garbage.sha256", "not a manifest\n")
        _w(root / "empty.sha256", "")
        for bad, why in (("nope.sha256", "No such file or directory"),
                         ("sl_comparison", "Is a directory"),
                         ("garbage.sha256", "not '<sha256>  <path>' lines"),
                         ("empty.sha256", "no entries")):
            rc, _ = _run_complete(root, "--code_manifest", str(root / bad))
            nums = _numbers(root)
            assert rc == INCOMPLETE and nums["missing"] == [
                f"code manifest {bad} unreadable ({why}): code drift during "
                f"the run not checked"] + NOTORCH, nums["missing"]
            assert nums["run"]["code_drift"] is None
        # ...and so is code_manifest.py.
        load = pt._optional_module

        def no_manifest_module(name, path):
            return None if Path(path).name == "code_manifest.py" else \
                load(name, path)

        pt._optional_module = no_manifest_module
        try:
            rc, _ = _run_complete(root, "--code_manifest", str(man))
        finally:
            pt._optional_module = load
        nums = _numbers(root)
        assert rc == INCOMPLETE and nums["missing"] == [
            "sl_comparison/code_manifest.py not loadable: code drift during "
            "the run not checked"] + NOTORCH, nums["missing"]
        assert nums["run"]["code_drift"] is None


def test_paper_bold_ties_at_the_printed_precision():
    """The paper prints 3 decimals: its bold (tab_tiers best_<metric>,
    tab_solo best_in_category_paper16, tab_external best_matched) marks every
    row that prints the best value; best_in_category, beats_peers_* and the
    X5 ranks stay on the exact values."""
    rows = [{"g": 1, "v": .940432}, {"g": 1, "v": .940003},
            {"g": 1, "v": .9394}, {"g": 2, "v": .0004},
            {"g": 2, "v": -.0004}, {"g": 2, "v": -.0006}]
    pt._flag_best(rows, ["g"], "v", "exact")
    pt._flag_best(rows, ["g"], "v", "printed", decimals=3)
    assert [r["exact"] for r in rows] == [True, False, False, True, False,
                                          False]
    # -0.000 is 0.000's equal; -0.001 is not.
    assert [r["printed"] for r in rows] == [True, True, False, True, True,
                                            False]
    # tab_solo (the 2026-09-23 run's Tier I go: go2vec 0.940432, go 0.940003).
    solo = [{"variant": "pd:per_head", "cv": "cv1", "status": "ok",
             "category": "go", "embedding": e, "in_paper": True,
             "auprg_any_mean": x}
            for e, x in (("go2vec", .940432), ("go", .940003),
                         ("onto2vec", .909))]
    assert pt.flag_solo_best(solo) == []
    by = {r["embedding"]: r for r in solo}
    assert [by[e]["best_in_category"] for e in ("go2vec", "go", "onto2vec")
            ] == [True, False, False]
    assert [by[e]["best_in_category_paper16"]
            for e in ("go2vec", "go", "onto2vec")] == [True, True, False]
    # tab_tiers and tab_external on the fixture: the printed SLeuth variant
    # (pd:pooled) 0.9704 vs
    # SLMGAE 0.97 (Tier I AUPRG), SLGNN 0.2004 vs SLMGAE 0.2 (Adamson cv1).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = CMP["siamese_sl_pd_pooled"]["cv1"]
        CMP["siamese_sl_pd_pooled"]["cv1"] = (.955, .81, .9704)
        try:
            _build(root)
        finally:
            CMP["siamese_sl_pd_pooled"]["cv1"] = saved
        zs = (root / "sl_comparison" / "results" / "external_transfer" /
              "zero_shot_summary.csv")
        ext = _rows(zs)
        for r in ext:
            if (r["model"], r["screen"], r["split"]) == ("SLGNN", "Adamson",
                                                         "cv1"):
                r["spearman_mean"] = "0.2004"
        with open(zs, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ext[0]))
            w.writeheader()
            w.writerows(ext)
        assert _run(root) == INCOMPLETE
        t1 = [r for r in _rows(_out(root) / "tab_tiers.csv")
              if r["cv"] == "cv1"]
        assert [_one(t1, model=m)["best_auprg"] for m in (
            "siamese_sl_pd_pooled", "SLMGAE", "siamese_sl")] == [
                "True", "True", ""]     # siamese_sl (pd:per_head): unprinted
        assert _one(t1, model="siamese_sl_pd_pooled"
                    )["beats_peers_auprg"] == "True"       # exact
        assert _one(t1, model="SLMGAE")["best_auroc"] == "True"
        assert _one(t1, model="siamese_sl_pd_pooled"
                    )["best_auroc"] == "False"
        a1 = [r for r in _rows(_out(root) / "tab_external.csv")
              if (r["screen"], r["split"]) == ("Adamson", "cv1")]
        assert [_one(a1, model=m, row_set="matched")["best_matched"]
                for m in ("SLGNN", "SLMGAE", "NSF4SL")] == \
            ["True", "True", "False"]
        assert _one(a1, model="siamese_sl", row_set="matched"
                    )["best_matched"] == "False"
        x5 = _numbers(root)["items"]["external"]["value"]["pd:per_head"][
            "X5_rank_on_matched_rows"]["Adamson/cv1"]
        assert x5["order"][:2] == ["SLGNN", "SLMGAE"] and x5["rank"] == 3


def test_fold_set_short_a_fold_dir_is_listed():
    """items.fold_geometry's ranges are over the fold_<k> dirs found, and
    nothing else reads the fold arrays (the fold stamp hashes genes.txt /
    all_pos.npy / meta.json only): a fold set that is not exactly fold_0 ..
    fold_<num_folds - 1> (a partial copy short a directory, or one with a
    leftover extra) is listed, so the verdict is not PAPER COMPLETE. The
    whole set lists nothing."""
    import shutil

    def item(cv, have):
        return (f"numbers.json fold_geometry {cv}: fold dirs {have} under "
                f"sl_comparison/folds/{cv}, meta.json num_folds is 2; its "
                f"ranges are over those only")

    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        folds = root / "sl_comparison" / "folds"
        rc, _ = _run_complete(root)
        assert rc == (INCOMPLETE if NOTORCH else 0), rc
        assert _missing(_out(root)) == NOTORCH, _missing(_out(root))
        whole = _numbers(root)["items"]["fold_geometry"]["value"]["cv3"]
        # Short a directory.
        shutil.move(str(folds / "cv3" / "fold_0"), str(root / "aside"))
        rc, lines = _run_complete(root)
        assert rc == INCOMPLETE, rc
        assert _missing(_out(root)) == NOTORCH + [item("cv3", [1])], \
            _missing(_out(root))
        if not NOTORCH:
            assert lines[-1] == \
                f"PAPER INCOMPLETE: 1 item(s) ({item('cv3', [1])})", lines[-1]
        part = _numbers(root)["items"]["fold_geometry"]["value"]["cv3"]
        assert [r["fold"] for r in part["per_fold"]] == [1]
        assert part["test_prevalence"] != whole["test_prevalence"]
        # Whole again, plus a leftover extra directory.
        shutil.move(str(root / "aside"), str(folds / "cv3" / "fold_0"))
        shutil.copytree(folds / "cv1" / "fold_0", folds / "cv1" / "fold_2")
        rc, _ = _run_complete(root)
        assert rc == INCOMPLETE and _missing(_out(root)) == \
            NOTORCH + [item("cv1", [0, 1, 2])], _missing(_out(root))
        # The right number of directories, one under another index.
        shutil.rmtree(folds / "cv1" / "fold_2")
        shutil.move(str(folds / "cv2" / "fold_1"), str(folds / "cv2" / "fold_5"))
        rc, _ = _run_complete(root)
        assert rc == INCOMPLETE and _missing(_out(root)) == \
            NOTORCH + [item("cv2", [0, 5])], _missing(_out(root))
        # The whole set: nothing listed.
        shutil.move(str(folds / "cv2" / "fold_5"), str(folds / "cv2" / "fold_1"))
        rc, _ = _run_complete(root)
        assert _missing(_out(root)) == NOTORCH, _missing(_out(root))
        assert rc == (INCOMPLETE if NOTORCH else 0), rc
        # A meta.json that records no num_folds has nothing to compare with.
        meta = json.loads((folds / "meta.json").read_text())
        del meta["num_folds"]
        (folds / "meta.json").write_text(json.dumps(meta))
        _run_complete(root)
        assert not [m for m in _missing(_out(root))
                    if m.startswith("numbers.json fold_geometry")], \
            _missing(_out(root))


def test_comparison_table_cut_short_is_listed():
    """A comparison.csv whose last line was cut (a full disk while compare.py
    wrote it, a copy cut short) is listed under Missing, not a traceback:
    csv.DictReader fills the fields a short row lacks with None. The rows
    after the cut fall back to their JSON, so every table is still written.
    A whole last line that only lost its line end is not a cut."""
    item = ("sl_comparison/results/comparison.csv has a line cut short "
            "(compare.py did not finish writing it); rerun "
            "sl_comparison/slurm/run_compare.sh with PCA_ARM=none")
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        p = root / "sl_comparison" / "results" / "comparison.csv"
        whole = p.read_bytes()
        assert whole.endswith(b",\r\n"), whole[-40:]
        last = whole[:-2].rsplit(b"\n", 1)[-1]
        start = len(whole) - 2 - len(last)
        second = last.index(b",", last.index(b",") + 1)
        third = last.index(b",", second + 1)
        # inside field 1, before / after the second comma, inside the value
        for cut in (1, second, second + 1, third + 2):
            p.write_bytes(whole[:start + cut])
            rc, _ = _run_complete(root)
            assert rc == INCOMPLETE, (cut, rc)
            assert item in _missing(_out(root)), (cut, _missing(_out(root)))
            assert (_out(root) / "tab_tiers.csv").exists(), cut
        for keep in (len(last), len(last) + 1):  # no line end / CR only
            p.write_bytes(whole[:start + keep])
            _run_complete(root)
            assert _missing(_out(root)) == NOTORCH, \
                (keep, _missing(_out(root)))


def _set_val(root: Path, etype: str, cv: str, value, key="val_auprg_any_mean",
             res: Path = None):
    """Record a validation mean in a single-modality run's results.json, as
    train.py does on the shared folds (Step 2 ranks on it); in arm none's
    root unless res names another single-embedding root."""
    p = ((res or root / "siamese_sl" / "results") /
         f"pcavar0.8_{etype}_{cv}" / "results.json")
    res = json.loads(p.read_text())
    if value is None:
        res["summary"].pop(key, None)
    else:
        res["summary"][key] = value
    _wj(p, res)


def test_selection_recomputed_from_single_modality_runs():
    """E3 / M11: best_per_category.json is written by the combo job's Step 2
    alone. A SELECTION_CV single-modality run trained after it (a failed task
    left unranked under BEST_CAT_ALLOW_MISSING=1 and rerun by hand, or a rerun
    whose validation score changed) is fresh, so tab:solo tabulates it: a
    category whose recorded winner is no longer the first maximum of the
    recorded selection metric over the fresh SELECTION_CV dirs is listed.
    Ties go to the first in catalog order (Step 2's max()); a stale or absent
    candidate, a stale winner, another CV and a selection that records no
    metric are not judged. The entry gives the combo job's command with the
    token those runs carry (without it Step 2 ranks none of them), the arm,
    arm none's too, and the run's PCA_ARMS (with PCA_ARM they decide whose
    runs Step 2 ranks: test_selection_rerun_hint_ranks_the_recorded_root),
    and the CV the selection records (2026-10-21: SELECTION_CV is in no
    signature, so a job without it selected on cv1 and exited 0)."""
    if torch is None:
        return
    start = "selection is not the one the single-modality runs on disk give ("

    def sel_items(root):
        return [m for m in _missing(_out(root)) if m.startswith(start)]

    with _FixturePaper16(), _EnvCleared("ARCH_ARGS"), \
            tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        res = root / "siamese_sl" / "results"
        # The scores Step 2 ranked: ea wins expression, as recorded.
        for etype, val in (("ea", .80), ("eb", .70), ("pa", .75), ("ka", .60)):
            _set_val(root, etype, "cv1", val)
        # Another CV never decides (eb ahead there).
        _set_val(root, "ea", "cv2", .10)
        _set_val(root, "eb", "cv2", .90)
        rc, _ = _run_complete(root)
        assert rc == 0 and _missing(_out(root)) == [], _missing(_out(root))
        # eb/cv1 trained after the selection, ahead of the recorded winner.
        _set_val(root, "eb", "cv1", .85)
        rc, lines = _run_complete(root)
        want = (start + "expression: best_per_category.json selected ea, "
                "whose run on disk has cv1 val_auprg_any_mean 0.800000, but "
                "eb has 0.850000): a selection-CV single-modality run was "
                "trained after the combo job's Step 2 ranked them; rerun "
                "siamese_sl/slurm/run_best_per_category_combo.sh under the "
                f"run's token (from siamese_sl/: RUN_TOKEN={TOKEN} "
                'PCA_ARM=none PCA_ARMS="none joint per_modality both" '
                "SELECTION_CV=cv1 sbatch --export=ALL "
                "slurm/run_best_per_category_combo.sh, with the run's other "
                "PCA_* targets, ARCH_ARGS and SIAMESE_VARIANTS_LIST exported; "
                "without the token the job ranks nothing and stops) and every "
                "job after it")
        assert rc == INCOMPLETE and _missing(_out(root)) == [want], \
            _missing(_out(root))
        assert lines[-1] == f"PAPER INCOMPLETE: 1 item(s) ({want})", lines[-1]
        s = _numbers(root)["items"]["selection"]
        assert s["status"] == "ok" and s["value"]["winners"] == WINNERS, s
        # The command carries the token the ranked runs carry (the
        # selection's), not the tables job's own $RUN_TOKEN: a combo job
        # submitted without it takes a fresh token from config.conf, removes
        # the selection and the combo markers, ranks no run and stops.
        rc, _ = _run_complete(root, "--run_token", "T2",
                              "--ablation_run_token", TOKEN)
        assert rc == INCOMPLETE and _missing(_out(root)) == [want], \
            _missing(_out(root))
        # No expected token (--solo_run_token ''): where to read it, never
        # "RUN_TOKEN=None".
        rc, _ = _run_complete(root, "--solo_run_token", "")
        assert rc == INCOMPLETE and _missing(_out(root)) == [want.replace(
            f"RUN_TOKEN={TOKEN} PCA_ARM=none", "RUN_TOKEN=<the token in the "
            "single-modality runs' run_token.txt> PCA_ARM=none")], \
            _missing(_out(root))
        # A tie keeps the first in catalog order, as Step 2's max() does.
        _set_val(root, "eb", "cv1", .80)
        rc, _ = _run_complete(root)
        assert rc == 0 and _missing(_out(root)) == [], _missing(_out(root))
        # A candidate that is not this run's is not ranked here either (Step
        # 2 refuses it): only build_solo's entries for the stale cell.
        _set_val(root, "eb", "cv1", .85)
        _w(res / "pcavar0.8_eb_cv1" / "run_token.txt", "OLD\n")
        rc, _ = _run_complete(root)
        assert rc == INCOMPLETE and not sel_items(root), _missing(_out(root))
        assert any(m.endswith("pcavar0.8_eb_cv1 is stale (run_token.txt is "
                              "another run's)")
                   for m in _missing(_out(root))), _missing(_out(root))
        _w(res / "pcavar0.8_eb_cv1" / "run_token.txt", TOKEN + "\n")
        rc, _ = _run_complete(root)
        assert rc == INCOMPLETE and _missing(_out(root)) == [want], \
            _missing(_out(root))
        # The recorded winner's own run stale: nothing to compare with (the
        # item itself stays whole).
        _w(res / "pcavar0.8_ea_cv1" / "run_token.txt", "OLD\n")
        rc, _ = _run_complete(root)
        assert rc == INCOMPLETE and not sel_items(root), _missing(_out(root))
        assert _numbers(root)["items"]["selection"]["status"] == "ok"
        assert not any("numbers.json selection" in m
                       for m in _missing(_out(root))), _missing(_out(root))
        _w(res / "pcavar0.8_ea_cv1" / "run_token.txt", TOKEN + "\n")
        # A candidate without the metric is not ranked (Step 2 skips it too).
        _set_val(root, "eb", "cv1", None)
        rc, _ = _run_complete(root)
        assert rc == 0 and _missing(_out(root)) == [], _missing(_out(root))
        # A selection made on another CV (a SELECTION_CV=cv2 run; eb is
        # ahead there) is judged on that CV's runs, and its command carries
        # that CV: pasted without it, the job selects on cv1 and exits 0
        # (2026-10-21).
        p = res / "best_per_category.json"
        sel = json.loads(p.read_text())
        sel["_selection_meta"]["selection_cv"] = "cv2"
        _wj(p, sel)
        rc, _ = _run_complete(root)
        tail = want[want.index("): a selection-CV single-modality run"):]
        want2 = (start + "expression: best_per_category.json selected ea, "
                 "whose run on disk has cv2 val_auprg_any_mean 0.100000, but "
                 "eb has 0.900000" + tail.replace("SELECTION_CV=cv1 ",
                                                  "SELECTION_CV=cv2 "))
        assert "SELECTION_CV=cv2 sbatch" in want2, want2
        assert rc == INCOMPLETE and _missing(_out(root)) == [want2], \
            _missing(_out(root))
        sel["_selection_meta"]["selection_cv"] = "cv1"
        _wj(p, sel)
        # A selection that records no metric (an older file) is not judged.
        _set_val(root, "eb", "cv1", .85)
        sel = json.loads(p.read_text())
        del sel["_selection_meta"]["selection_metric"]
        _wj(p, sel)
        rc, _ = _run_complete(root)
        assert rc == 0 and _missing(_out(root)) == [], _missing(_out(root))


def test_selection_rerun_hint_ranks_the_recorded_root():
    """The selection entry's combo-job command, pasted from siamese_sl/ into
    bash or zsh, from a shell that exports nothing of the run's and from one
    that exports another arm's PCA_ARM and PCA_ARMS and another SELECTION_CV
    (an arm's "After a combo job submitted by hand" recipe leaves PCA_ARM
    exported: arm none's command, printed without it, removed and retrained
    arm joint; 2026-10-21), makes config.conf and run_best_per_category.conf
    (the real ones, sourced as every job sources them) put RESULTS_ROOT at
    the hint's arm's root, STEP1_ROOT at the root the selection records
    (_selection_meta.step1_root) and SELECTION_CV at the CV it records:
    without PCA_ARMS a shell means all four arms, where joint ranks none's
    runs and both per_modality's, after the job has removed the arm's
    selection and markers (2026-10-21). Its PCA_ARMS is the tables run's
    arms, adjusted only where they would place another root: arm joint
    owning its runs (tables run with joint alone, joint and both, no
    --pca_arms at all, all four), arm joint reading none's runs (a
    joint-alone tables run after a four-arm run), arm both owning its runs
    beside none, arm none. ARCH_ARGS verbatim (quoted) when the tables job
    has it."""
    if torch is None:
        return
    import os
    import shlex
    import shutil
    import subprocess
    siamese = HERE.parent / "siamese_sl"
    start = "selection is not the one the single-modality runs on disk give ("
    shells = ["bash"] + (["zsh"] if shutil.which("zsh") else [])
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("PCA_", "POST_PCA_", "SLURM_"))
            and k not in ("ARCH_ARGS", "SELECTION_CV")}
    # the shell another arm's hand recipe leaves (export PCA_ARM=<arm>), with
    # another run's arms and selection CV exported too
    other = dict(base, PCA_ARM="joint", PCA_ARMS="joint", SELECTION_CV="cv3")

    def hint(root, arm, pca_arms, arch=None):
        """The printed assignments before sbatch, and their dict."""
        if arch:
            os.environ["ARCH_ARGS"] = arch
        try:
            with _PcaEnv():
                pt.main(["--root", str(root), "--variants", *VARIANTS,
                         "--selection_variant", "pd:per_head", "--run_token",
                         TOKEN, "--bestcat_extra_variants", "0",
                         "--pca_variance", "0.8", "--code_manifest", "",
                         "--pca_arm", arm] +
                        (["--pca_arms", *pca_arms] if pca_arms else []))
        finally:
            os.environ.pop("ARCH_ARGS", None)
        got = [m for m in _missing(_out(root, arm)) if m.startswith(start)]
        assert len(got) == 1, (arm, pca_arms, _missing(_out(root, arm)))
        text = got[0].split("(from siamese_sl/: ", 1)[1].split(
            " sbatch --export=ALL slurm/run_best_per_category_combo.sh", 1)[0]
        words = shlex.split(text)
        assert words[0] == f"RUN_TOKEN={TOKEN}" and all(
            "=" in w for w in words), text
        return text, dict(w.split("=", 1) for w in words), got[0]

    def job(text):
        """(RESULTS_ROOT, STEP1_ROOT, SELECTION_CV) of a combo job submitted
        with these assignments, pasted into each shell (sbatch passes the
        environment on), from the clean shell and from `other`: one answer,
        the command's own."""
        got = set()
        for env in (base, other):
            for sh in shells:
                r = subprocess.run(
                    [sh, "-c", text + " bash -c 'source slurm/config.conf "
                     ">/dev/null 2>&1 && source slurm/run_best_per_category"
                     ".conf >/dev/null 2>&1 || exit 7; echo \"$RESULTS_ROOT "
                     "$STEP1_ROOT $SELECTION_CV\"'"],
                    cwd=str(siamese), env=env, capture_output=True,
                    text=True, timeout=120)
                assert r.returncode == 0, (sh, text, r.stdout,
                                           r.stderr[-600:])
                got.add(tuple(r.stdout.split()))
        assert len(got) == 1, (text, got)
        return got.pop()

    def scores(root, res):
        # cv1 as Step 2 ranked it (ea wins expression), then eb trained
        # after the selection, ahead of the recorded winner
        for etype, val in (("ea", .80), ("eb", .85), ("pa", .75),
                           ("ka", .60)):
            _set_val(root, etype, "cv1", val, res=res)

    with _EnvCleared("ARCH_ARGS"), tempfile.TemporaryDirectory() as tmp:
        # (a) joint owns its runs: a run of joint alone recorded pca_joint
        root = Path(tmp) / "a"
        _build(root)
        _build_arm(root, "joint", arms=("joint",))
        assert json.loads((_sia(root, "joint") / "best_per_category.json")
                          .read_text())["_selection_meta"]["step1_root"] \
            == "results/pca_joint"
        scores(root, _sia(root, "joint"))
        for pca_arms, want in ((["joint"], "joint"),
                               (["joint", "both"], "joint both"),
                               (None, "joint"),
                               (list(ARMS), "joint per_modality both")):
            text, env, entry = hint(root, "joint", pca_arms)
            assert (env.get("PCA_ARM"), env.get("PCA_ARMS"),
                    env.get("SELECTION_CV")) == ("joint", want, "cv1"), (
                pca_arms, entry)
            assert job(text) == ("results/pca_joint", "results/pca_joint",
                                 "cv1"), (pca_arms, text)
            assert "with the run's other PCA_* targets, ARCH_ARGS and " \
                "SIAMESE_VARIANTS_LIST exported" in entry, entry
        # ARCH_ARGS as the tables job has it, quoted for the shell
        text, env, entry = hint(root, "joint", ["joint"],
                                arch="--layernorm --dropout 0.3")
        assert env["ARCH_ARGS"] == "--layernorm --dropout 0.3" and \
            "ARCH_ARGS='--layernorm --dropout 0.3' sbatch" in entry and \
            "with the run's other PCA_* targets and SIAMESE_VARIANTS_LIST " \
            "exported" in entry, entry
        assert job(text) == ("results/pca_joint", "results/pca_joint",
                             "cv1"), text
        # (b) joint reads arm none's runs (a four-arm run's selection), the
        # tables run under joint alone: none is put back in front
        root = Path(tmp) / "b"
        _build(root)
        _build_arm(root, "joint")
        scores(root, _sia(root))
        for arm, pca_arms, want in (
                ("joint", ["joint"], "none joint"),
                ("joint", list(ARMS), "none joint per_modality both"),
                ("none", list(ARMS), "none joint per_modality both"),
                ("none", ["joint"], "none joint")):
            text, env, entry = hint(root, arm, pca_arms)
            assert (env.get("PCA_ARM"), env.get("PCA_ARMS"),
                    env.get("SELECTION_CV")) == (arm, want, "cv1"), (
                arm, pca_arms, entry)
            # RESULTS_ROOT tells none from joint, whose STEP1_ROOT is
            # results/ too
            assert job(text) == ("results" if arm == "none" else
                                 "results/pca_" + arm, "results", "cv1"), (
                arm, pca_arms, text)
        # (c) both owns its runs beside none (PCA_ARMS="none both")
        root = Path(tmp) / "c"
        _build(root)
        _build_arm(root, "both", arms=("none", "both"))
        scores(root, _sia(root, "both"))
        for pca_arms, want in ((["none", "both"], "none both"),
                               (["both"], "both"),
                               (list(ARMS), "none joint both")):
            text, env, entry = hint(root, "both", pca_arms)
            assert (env.get("PCA_ARM"), env.get("PCA_ARMS")) == (
                "both", want), (pca_arms, entry)
            assert job(text) == ("results/pca_both", "results/pca_both",
                                 "cv1"), (pca_arms, text)


def test_external_summary_cut_short_is_not_read_as_rows():
    """A copy of an external-transfer summary that ends inside a line (an
    interrupted download of the results tree) is not tabulated: pandas would
    take the cut line as a row, a number cut inside its digits included. The
    whole file is read as before."""
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        ext = root / "sl_comparison" / "results" / "external_transfer"
        rc, _ = _run_complete(root)
        assert _missing(_out(root)) == NOTORCH, _missing(_out(root))
        for name, table in (("zero_shot_summary.csv", "tab_external.csv"),
                            ("finetune_summary.csv", "tab_external_ft.csv")):
            p = ext / name
            whole = p.read_bytes()
            assert whole.endswith(b"\n")
            p.write_bytes(whole[:-7])            # inside the last line
            _run_complete(root)
            miss = _missing(_out(root))
            assert any(m.startswith(f"sl_comparison/results/external_transfer/"
                                    f"{name}") for m in miss), (name, miss)
            assert any(table in m for m in miss), (name, miss)
            p.write_bytes(whole)
        _run_complete(root)
        assert _missing(_out(root)) == NOTORCH, _missing(_out(root))


def test_external_tables_of_another_arm_are_not_read():
    """external_transfer.py --pca_arm (2026-10-21) writes the SLeuth combos'
    arm in a trailing pca_arm column of every CSV. A summary naming this arm
    is read exactly as one without the column (a table from before); one
    naming another arm is that arm's tables in this arm's namespace (a hand
    copy): not read, listed, as a comparison.csv of another arm is not."""
    import csv as _csv
    with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _complete_tree(root)
        ext = root / "sl_comparison" / "results" / "external_transfer"
        _run_complete(root)
        before = {t: (_out(root) / t).read_text()
                  for t in ("tab_external.csv", "tab_external_ft.csv")}
        orig = {n: (ext / n).read_text() for n in ("zero_shot_summary.csv",
                                                   "finetune_summary.csv")}

        def label(arm):
            for n, text in orig.items():
                rows = list(_csv.reader(text.splitlines()))
                with open(ext / n, "w", newline="") as f:
                    _csv.writer(f).writerows(
                        [rows[0] + ["pca_arm"]] + [r + [arm] for r in rows[1:]])

        label("none")
        rc, _ = _run_complete(root)
        assert _missing(_out(root)) == NOTORCH, _missing(_out(root))
        for t, text in before.items():
            assert (_out(root) / t).read_text() == text, t
        label("joint")
        rc, _ = _run_complete(root)
        miss = _missing(_out(root))
        for n in orig:
            assert (f"sl_comparison/results/external_transfer/{n} names PCA "
                    f"arm(s) joint in its pca_arm column, not none: another "
                    f"arm's external-transfer table, not read (rerun "
                    f"sl_comparison/slurm/run_external_transfer.sh with "
                    f"PCA_ARM=none)") in \
                [m.replace(str(root) + "/", "") for m in miss], (n, miss)
        tabs = _numbers(root)["tables"]
        assert tabs["tab_external"]["status"] == "missing" and \
            tabs["tab_external_ft"]["status"] == "missing", tabs
        assert rc == 1, rc


def _eval_own_summaries(root: Path, raw: dict, modes: dict,
                        splits=pt.SPLITS, flt: str = "sldb_filtered") -> None:
    """eval_finetuning.py's own zero_shot_summary.csv / finetuned_summary.csv
    under the pd:per_head cv3 combo's eval_finetuning_<split>/: per screen
    the per-fold mean Spearman on the `flt` rows (raw: {screen: value},
    modes: {mode: value}), next to the rows paper_tables must not read (the
    OLS-calibrated row, the unfiltered rows, the ensemble rows)."""
    combo = root / "siamese_sl" / "results" / "best_cat_ea_ka_pa_cv3"
    other = "unfiltered" if flt == "sldb_filtered" else "other"
    for split in splits:
        zs = ["dataset,calibration,filter,spearman_mean,spearman_std,"
              "aggregation"]
        ft = ["dataset,mode,filter,spearman_mean,spearman_std,aggregation"]
        for s in pt.SCREENS:
            zs += [f"{s},raw,{flt},{raw[s]!r},0.01,per_fold",
                   f"{s},ols,{flt},0.9,0.01,per_fold",
                   f"{s},raw,{other},0.9,0.01,per_fold",
                   f"{s},raw,{flt},0.9,,ensemble"]
            for mode, x in modes.items():
                ft += [f"{s},{mode},{flt},{x!r},0.01,per_fold",
                       f"{s},{mode},{other},0.9,0.01,per_fold",
                       f"{s},{mode},{flt},0.9,,ensemble"]
        ed = combo / f"eval_finetuning_{split}"
        _w(ed / "zero_shot_summary.csv", "\n".join(zs) + "\n")
        _w(ed / "finetuned_summary.csv", "\n".join(ft) + "\n")
def test_external_rows_checked_against_the_eval_on_disk():
    """external_transfer.py's CSVs record no fingerprint of the eval they were
    computed from, a retrained cv3 combo keeps its directory name, and a
    rerun eval rewrites its outputs in place. A SLeuth row whose
    all_sldb_filtered Spearman is not what the eval dir NOW under the cv3
    combo reports is listed (the verdict was PAPER COMPLETE with the replaced
    model's numbers): one entry per table and model, within
    EXTERNAL_EVAL_TOL, and nothing for a cell without a counterpart."""
    def listed(root):
        return [m for m in _missing(_out(root)) if m not in NOTORCH]

    with tempfile.TemporaryDirectory() as tmp, _FixturePaper16():
        root = Path(tmp)
        _complete_tree(root)
        ext = root / "sl_comparison" / "results" / "external_transfer"
        raw = {"Adamson": .10 + .02, "AdamsonGI": .10 + .02,
               "Corn": .10 + .02, "Gilbert": .5 + .02}
        modes = {"LP_last": .3, "Full": .97}
        # As external_transfer.py writes them: the fine-tuning table's
        # zero_shot_raw is the zero-shot table's value.
        rows = _rows(ext / "finetune_summary.csv")
        for r in rows:
            if (r["model"], r["row_set"], r["method"]) == (
                    "siamese_sl", "all_sldb_filtered", "zero_shot_raw"):
                r["spearman_mean"] = repr(raw[r["screen"]])
        with open(ext / "finetune_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        # The eval these tables were built from: nothing listed.
        _eval_own_summaries(root, raw, modes)
        rc, _ = _run_complete(root)
        assert listed(root) == [] and rc == (0 if torch is not None else
                                             INCOMPLETE), listed(root)
        # Round-off between the two programs is not a difference.
        _eval_own_summaries(root, {s: x + 1e-9 for s, x in raw.items()},
                            {m: x - 1e-9 for m, x in modes.items()})
        _run_complete(root)
        assert listed(root) == [], listed(root)
        # The cv2 eval of a retrained combo: another zero-shot value. Listed
        # for both tables (the fine-tuning table repeats the column), and the
        # rows stay in the tables.
        _eval_own_summaries(root, {**raw, "Corn": .2}, modes, splits=("cv2", ))
        rc, _ = _run_complete(root)
        miss = listed(root)
        assert rc == INCOMPLETE and len(miss) == 2, miss
        for table, m in zip(("tab_external", "tab_external_ft"), miss):
            assert m.startswith(f"{table} siamese_sl: 1 value(s) in "), m
            assert ("Corn/cv2 zero_shot_raw: 0.120000 there, 0.200000 in "
                    "the eval") in m, m
            assert m.endswith("rerun sl_comparison/slurm/"
                              "run_external_transfer.sh with PCA_ARM=none"), m
        assert _one(_rows(_out(root) / "tab_external.csv"), screen="Corn",
                    split="cv2", model="siamese_sl",
                    row_set="all_sldb_filtered")["spearman_mean"] != ""
        # Only the fine-tuning was rerun (same checkpoints, same zero-shot):
        # tab_external_ft alone, one entry however many cells.
        _eval_own_summaries(root, raw, {"LP_last": .31, "Full": .5})
        rc, _ = _run_complete(root)
        miss = listed(root)
        assert rc == INCOMPLETE and len(miss) == 1, miss
        assert miss[0].startswith(
            "tab_external_ft siamese_sl: 24 value(s) in "), miss
        # No SynLethDB-filtered rows in the eval's summaries: no counterpart,
        # not judged (as with no summary at all, the fixture's default).
        _eval_own_summaries(root, {s: .9 for s in raw},
                            {"LP_last": .9, "Full": .9}, flt="unfiltered")
        _run_complete(root)
        assert listed(root) == [], listed(root)


def test_pca_effects_alignment():
    """The 2x2 effects pair the four arms fold by fold as paired_deltas pairs
    two runs: by 'fold' in any listed order, else by position when every
    list has one length; only the folds all four scored (NaN is not a
    score); no se from one fold; None when no fold pairs."""
    def runs(n, j, p, b, ids=True):
        return {a: [{**({"fold": k} if ids else {}), "auprg_any": x}
                    for k, x in enumerate(vals)]
                for a, vals in (("none", n), ("joint", j),
                                ("per_modality", p), ("both", b))}

    arms = ((.5, .7), (.6, .7), (.55, .65), (.8, .9))

    def check(e):
        # fold 0: per_modality .125, joint .175, interaction .15;
        # fold 1: per_modality .075, joint .125, interaction .25
        for name, mean, sd in (("per_modality", .1, .025),
                               ("joint", .15, .025), ("interaction", .2, .05)):
            _close(e[name]["mean"], mean)
            _close(e[name]["sd"], sd)
            _close(e[name]["se"], sd)
            assert e[name]["n"] == 2, name

    check(pt.pca_effects(runs(*arms)))
    r = runs(*arms)
    r["both"] = r["both"][::-1]               # 'fold' decides, not position
    check(pt.pca_effects(r))
    check(pt.pca_effects(runs(*arms, ids=False)))
    r = runs(*arms, ids=False)
    r["joint"] = r["joint"][:1]               # no ids, another length
    assert pt.pca_effects(r) is None
    r = runs((.5, float("nan")), *arms[1:])
    e = pt.pca_effects(r)                      # fold 1 has no 'none' score
    assert e["interaction"]["n"] == 1 and e["interaction"]["se"] is None
    _close(e["interaction"]["mean"], .15)
    _close(e["interaction"]["sd"], 0)
    r["none"] = [{"fold": 2, "auprg_any": .5}]  # no fold all four scored
    assert pt.pca_effects(r) is None
    assert pt.pca_effects({a: [] for a in r}) is None


# =============================================================================
# PCA arms (2026-10-20): per-arm mode
# =============================================================================


def _no_miss_outside(miss, arm: str) -> list:
    """The Missing entries naming a siamese_sl/results/ or
    sl_comparison/results/ path that is not arm `arm`'s own, its owner's
    single-embedding runs or a shared model's row: what another arm's files
    reaching its tables would look like."""
    import re
    own = "" if arm == "none" else f"pca_{arm}/"
    s1 = "" if _owner(arm) == "none" else f"pca_{_owner(arm)}/"
    shared = ("SLMGAE_", "SLGNN_", "NSF4SL_", "DegreePrior_",
              "BilinearProbe_", "ELISL_")

    def under(p, prefix):
        return p.startswith(prefix) if prefix else not p.startswith("pca_")

    return [m for m in miss
            if any(not (under(p, own) or (under(p, s1) and
                                          p[len(s1):].startswith("pcavar")))
                   for p in re.findall(r"siamese_sl/results/(\S+)", m)) or
            any(not (under(p, own) or p.startswith(shared))
                for p in re.findall(r"sl_comparison/results/(\S+)", m))]


def test_pca_arm_helpers_match_config_conf():
    """paper_tables keeps its OWN copies of siamese_sl/slurm/config.conf's
    PCA-arm helpers (pca_arm_results / pca_arm_cmp / pca_step1_arm /
    pca_step1_arms / pca_arm_label / pca_step1_label) and of the arms' order
    (as arch_signature.PCA_ARMS): a drift there would read another arm's
    namespace or mislabel one. Sourced as every job sources it, under
    several PCA_ARMS and targets; --pca_arms normalises as config.conf
    normalises PCA_ARMS."""
    import os
    import subprocess
    assert pt.PCA_ARM_NAMES == ARMS == tuple(arch.PCA_ARMS)
    conf = HERE.parent / "siamese_sl" / "slurm" / "config.conf"
    script = ('unset SLURM_JOB_ID; source "$1" >/dev/null 2>&1 || exit 1; '
              'echo "ARMS|$PCA_ARMS"; for a in none joint per_modality both; '
              'do printf "%s|%s|%s|%s|%s|%s|%s|%s\\n" "$a" '
              '"$(pca_arm_dir $a)" "$(pca_arm_results $a)" '
              '"$(pca_arm_cmp $a)" "$(pca_step1_arm $a)" '
              '"$(pca_step1_arms $a)" "$(pca_arm_label $a)" '
              '"$(pca_step1_label $a)"; done')
    base = {k: v for k, v in os.environ.items()
            if k not in PCA_ENV and not k.startswith("POST_PCA")}
    for env in ({}, {"PCA_ARMS": "both joint"}, {"PCA_ARMS": "per_modality"},
                {"PCA_ARMS": "joint", "PCA_JOINT_DIM": "300"},
                {"PCA_ARMS": "both none", "PCA_MODALITY_VARIANCE": "0.9",
                 "PCA_JOINT_VARIANCE": "0.75"}):
        r = subprocess.run(["bash", "-c", script, "_", str(conf)],
                           capture_output=True, text=True, timeout=120,
                           cwd=str(conf.parent.parent), env={**base, **env})
        assert r.returncode == 0, (env, r.stdout[-800:], r.stderr[-800:])
        lines = r.stdout.strip().splitlines()
        conf_arms = next(x for x in lines if x.startswith("ARMS|")).split(
            "|", 1)[1].split()
        if "PCA_ARMS" in env:
            got = pt.resolve_pca_arms(pt.argparse.Namespace(),
                                      {"PCA_ARMS": env["PCA_ARMS"]})[1]
            assert got == conf_arms, (env, got, conf_arms)
        t = pt.pca_targets(env)
        rows = [x.split("|") for x in lines if x.count("|") == 7]
        assert [x[0] for x in rows] == list(ARMS), lines
        for a, d, res, cmpr, s1, s1s, label, s1label in rows:
            assert (pt.pca_arm_dir(a), pt.pca_arm_results(a),
                    pt.pca_arm_cmp(a)) == (d, res, cmpr), (env, a)
            assert pt.pca_step1_arm(a, conf_arms) == s1, (env, a)
            assert " ".join(pt.pca_step1_arms(a, conf_arms)) == s1s, (env, a)
            assert pt.pca_arm_label(a, t) == label, (env, a, label)
            assert pt.pca_step1_label(a, t, conf_arms) == s1label, \
                (env, a, s1label)
            assert s1label == _step1_label(a, conf_arms) or \
                env.get("PCA_MODALITY_VARIANCE"), (env, a)


def test_pca_arm_cli_and_environment():
    """--pca_arm: CLI, else $PCA_ARM, else none; --pca_arms: CLI, else
    $PCA_ARMS, else the arm alone, read in the fixed order (a quoted list
    too). An unknown or repeated arm, an empty list and a PCA target
    config.conf refuses stop the run (exit 1) before anything is read or
    written; the targets default to config.conf's 0.8 / 0.8 / unset."""
    NS = pt.argparse.Namespace
    f = pt.resolve_pca_arms
    assert f(NS(), {}) == ("none", ["none"])
    assert f(NS(), {"PCA_ARM": "both"}) == ("both", ["both"])
    assert f(NS(pca_arm="joint"), {"PCA_ARM": "both"}) == ("joint", ["joint"])
    assert f(NS(), {"PCA_ARMS": "both none joint"}) == (
        "none", ["none", "joint", "both"])
    assert f(NS(pca_arms=["both", "per_modality"]), {"PCA_ARMS": "none"}) == (
        "none", ["per_modality", "both"])
    assert f(NS(pca_arms=["both joint"]), {}) == ("none", ["joint", "both"])
    for ns, env, why in ((NS(pca_arm="x"), {}, "PCA arm 'x'"),
                         (NS(), {"PCA_ARM": "joint2"}, "PCA arm 'joint2'"),
                         (NS(pca_arms=["none", "x"]), {}, "'x' is not an arm"),
                         (NS(pca_arms=["joint", "joint"]), {},
                          "lists joint twice"),
                         (NS(), {"PCA_ARMS": "  "}, "lists no arm")):
        try:
            f(ns, env)
        except SystemExit as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(why)
    assert pt.pca_targets({}) == {"PCA_MODALITY_VARIANCE": "0.8",
                                  "PCA_JOINT_VARIANCE": "0.8",
                                  "PCA_JOINT_DIM": None}
    for env, why in (({"PCA_MODALITY_VARIANCE": "1"},
                      "$PCA_MODALITY_VARIANCE='1'"),
                     ({"PCA_JOINT_VARIANCE": "0.0"},
                      "$PCA_JOINT_VARIANCE='0.0'"),
                     ({"PCA_JOINT_DIM": "012"}, "$PCA_JOINT_DIM='012'"),
                     ({"PCA_JOINT_DIM": "3x"}, "$PCA_JOINT_DIM='3x'")):
        try:
            pt.pca_targets(env)
        except SystemExit as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(why)
    # From the command line and the environment: nothing written.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        for kw in ({"arm": "sideways"}, {"env": {"PCA_JOINT_DIM": "0"}}):
            try:
                _run(root, **kw)
            except SystemExit as e:
                assert str(e).startswith("ERROR: "), (kw, e)
            else:
                raise AssertionError(kw)
            assert not _out(root).exists(), kw
        # $PCA_ARM is the default arm.
        with _PcaEnv(PCA_ARM="joint"):
            _build_arm(root, "joint")
            pt.main(["--root", str(root), "--variants", *VARIANTS,
                     "--selection_variant", "pd:per_head", "--run_token",
                     TOKEN, "--bestcat_extra_variants", "0",
                     "--pca_variance", "0.8", "--code_manifest", ""])
        assert _numbers(root, "joint")["run"]["pca_arm"] == "joint"
        assert _numbers(root, "joint")["run"]["pca_arms"] == ["joint"]
        assert not _out(root).exists()


def test_arm_tables_read_and_write_their_own_namespace():
    """Per-arm mode, arm joint: everything that is the arm's own comes from
    siamese_sl/results/pca_joint/ and sl_comparison/results/pca_joint/ (the
    markers, whose text is results/pca_joint/..., the combos, selection,
    ablations, cell-line ablation, SLeuth rows, comparison.csv and external
    transfer), its single-modality runs from its owner none's root, the
    shared models' rows from sl_comparison/results/; its outputs go to its
    own paper_tables/ alone and say which arm they are."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        assert _run(root, arm="joint") == INCOMPLETE
        out = _out(root, "joint")
        assert out == root / "sl_comparison" / "results" / "pca_joint" / \
            "paper_tables"
        for f in pt.OUTPUT_FILES:
            assert (out / f).exists(), f
        assert not _out(root).exists()          # arm none's: never written
        nums = _numbers(root, "joint")
        assert all(t["status"] == "ok" for t in nums["tables"].values()), \
            nums["tables"]
        run = nums["run"]
        assert (run["pca_arm"], run["pca"]) == (
            "joint", "PCA on the concatenation of all modalities (variance "
            "0.8)")
        assert run["pca_settings"] == {"modality": "no",
                                       "joint": "variance 0.8"}
        assert run["pca_targets"] == {"PCA_MODALITY_VARIANCE": "0.8",
                                      "PCA_JOINT_VARIANCE": "0.8",
                                      "PCA_JOINT_DIM": None}
        assert (run["sia_results"], run["cmp_results"], run["shared_results"],
                run["step1_results"]) == (
            "siamese_sl/results/pca_joint", "sl_comparison/results/pca_joint",
            "sl_comparison/results", "siamese_sl/results")
        assert run["pca_arms"] == list(ARMS)
        tiers = _rows(out / "tab_tiers.csv")
        r = _one(tiers, cv="cv1", model="siamese_sl")
        _close(r["auprg_mean"],
               CMP["siamese_sl"]["cv1"][2] + ARM_SHIFT["joint"])
        _close(r["params_fold_mean"], 29022 - ARM_PARAMS_CUT["joint"])
        _close(_one(tiers, cv="cv1", model="SLMGAE")["auprg_mean"],
               CMP["SLMGAE"]["cv1"][2])
        # The bilinear probe is trained on arm none's features: another arm's
        # tab_tiers neither lists it nor misses it.
        assert not [t for t in tiers if t["model"] == "BilinearProbe"]
        assert not [m for m in nums["missing"] if "BilinearProbe" in m]
        solo = _one(_rows(out / "tab_solo.csv"), variant="pd:per_head",
                    cv="cv1", embedding="ea")
        assert solo["status"] == "ok" and \
            solo["dir"] == "siamese_sl/results/pcavar0.8_ea_cv1", solo
        _close(solo["combo_auprg_shared"],
               CMP["siamese_sl"]["cv1"][2] + ARM_SHIFT["joint"])
        loco = _rows(out / "tab_loco.csv")
        assert _one(loco, variant="pd:per_head", cv="cv3", dropped="none")[
            "dir"] == "siamese_sl/results/pca_joint/best_cat_ea_ka_pa_cv3"
        lp = _one(loco, variant="pd:per_head", cv="cv3", dropped="ppi")
        assert lp["status"] == "ok" and \
            lp["dir"] == "siamese_sl/results/pca_joint/ablate_no_ppi_cv3"
        rem = _one(_rows(out / "tab_cellline.csv"), variant="pd:per_head",
                   cv="cv1", condition="removed (cell_emb=0)")
        assert rem["status"] == "ok", rem["status"]
        _close(rem["auprg_mean"],
               CMP["siamese_sl_nocell"]["cv1"][2] + ARM_SHIFT["joint"])
        ext = _rows(out / "tab_external.csv")
        _close(_one(ext, screen="Adamson", split="cv1", model="siamese_sl",
                    row_set="matched")["spearman_mean"],
               .10 + ARM_SHIFT["joint"])
        pw = nums["items"]["pca_width"]["value"]["per_variant"]["pd:per_head"]
        assert (pw["min"], pw["max"]) == (1505 - 1000, 1585 - 1000)
        a = nums["items"]["arch"]["value"]["pd:per_head"]
        assert a["pca_arm"] == "joint" and a["pca_settings"] == {
            "modality": "no", "joint": "variance 0.8"}
        assert _no_miss_outside(nums["missing"], "joint") == [], \
            _no_miss_outside(nums["missing"], "joint")
        assert "siamese_sl/results/pca_joint/cellline_ablation_cv3/" \
            "results.json" in nums["missing"], nums["missing"]
        # Every CSV says which arm, in its first column, on every row.
        for t in pt.TABLES:
            with open(out / f"{t}.csv") as fh:
                rows = list(csv.reader(fh))
            assert rows[0][0] == "pca_arm" and len(rows) > 1 and \
                all(x[0] == "joint" for x in rows[1:]), t
        readme = (out / "README.md").read_text()
        assert readme.splitlines()[:3] == [
            "# Paper tables and numbers: PCA arm joint", "", nums["verdict"]]
        assert _section(readme, "Run settings").strip().splitlines()[0] == (
            "- PCA arm: joint (PCA on the concatenation of all modalities "
            "(variance 0.8)); siamese results siamese_sl/results/pca_joint/, "
            "comparison results sl_comparison/results/pca_joint/ (the shared "
            "models' rows from sl_comparison/results/); single-embedding runs "
            "siamese_sl/results/ (single-embedding runs at native dimension "
            "(no PCA), shared by arms none and joint); its combos trained "
            "with PCA within each modality no, on the concatenation variance "
            "0.8 (numbers.json run.pca_settings)")
        # Arm none's tables of the same tree: its own values.
        assert _run(root) == INCOMPLETE
        _close(_one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1",
                    model="siamese_sl")["auprg_mean"],
               CMP["siamese_sl"]["cv1"][2])
        assert _no_miss_outside(_numbers(root)["missing"], "none") == []


def test_arms_never_read_each_others_files():
    """Arm joint's tables are the same with arm none's own files gone (its
    combos, markers, selection, ablations, cell-line ablations, SLeuth rows,
    selected markers, comparison.csv and external transfer: all but the
    single-embedding runs the two share and the shared models' rows), and
    arm none's are the same with arm joint's namespaces gone."""
    import shutil
    shared = ("SLMGAE_", "SLGNN_", "NSF4SL_", "DegreePrior_",
              "BilinearProbe_", "ELISL_")

    def outputs(root, arm):
        nums = _numbers(root, arm)
        return ({t: (_out(root, arm) / f"{t}.csv").read_text()
                 for t in pt.TABLES},
                nums["missing"], nums["absent_by_design"], nums["items"])

    def rm(p):
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        assert _run(root, arm="joint") == INCOMPLETE
        before = outputs(root, "joint")
        for p in list(_sia(root).iterdir()):
            if not p.name.startswith(("pcavar", "pca_")):
                rm(p)
        for p in list(_cmp(root).iterdir()):
            if not p.name.startswith(("pca_",) + shared):
                rm(p)
        assert _run(root, arm="joint") == INCOMPLETE
        assert outputs(root, "joint") == before
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        assert _run(root) == INCOMPLETE
        before = outputs(root, "none")
        shutil.rmtree(_sia(root, "joint"))
        shutil.rmtree(_cmp(root, "joint"))
        assert _run(root) == INCOMPLETE
        assert outputs(root, "none") == before


def test_arm_rows_split_between_arm_and_shared_dirs():
    """compare.py keys rows by summary.model, and arm none's SLeuth rows sit
    in sl_comparison/results/ under every arm's names: an arm reads its
    SLeuth rows from its own directory only (arm none's never stand in for
    one it lacks) and the shared models' from sl_comparison/results/ only (a
    stray copy in the arm's directory is noted, never read). The default
    table's "JSON beside it" rule looks in the row's own directory and
    names it, with the arm in the rerun hint."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        cj = _cmp(root, "joint")
        _rewrite(cj / "comparison.csv",
                 lambda r: (r["cv"], r["model"]) != ("cv2", "siamese_sl"))
        (cj / "siamese_sl_cv2.json").unlink()   # arm none's still beside
        _row_json(cj / "SLGNN_cv2.json", "SLGNN", "cv2", (.1, .1, .1),
                  root / "sl_comparison" / "folds")
        (cj / "siamese_sl_pd_pooled_cv2.json").unlink()   # its CSV row kept
        (_cmp(root) / "SLMGAE_cv3.json").unlink()
        assert _run(root, arm="joint") == INCOMPLETE
        tiers = _rows(_out(root, "joint") / "tab_tiers.csv")
        assert _one(tiers, cv="cv2", model="siamese_sl")["status"] == \
            "missing"
        _close(_one(tiers, cv="cv2", model="SLGNN")["auprg_mean"],
               CMP["SLGNN"]["cv2"][2])
        assert _one(tiers, cv="cv2", model="siamese_sl_pd_pooled")[
            "status"] == "missing"
        assert _one(tiers, cv="cv3", model="SLMGAE")["status"] == "missing"
        nums = _numbers(root, "joint")
        miss = nums["missing"]
        assert "tab_tiers siamese_sl/cv2 (no usable comparison row)" in miss
        assert "siamese_sl_pd_pooled/cv2: comparison.csv has a row but no " \
            "evaluate_model JSON beside it (sl_comparison/results/pca_joint/" \
            "siamese_sl_pd_pooled_cv2.json was moved aside by a rerun, or " \
            "was never graded); not tabulated, rerun sl_comparison/slurm/" \
            "run_compare.sh with PCA_ARM=joint" in miss, miss
        assert any(m.startswith(
            "SLMGAE/cv3: comparison.csv has a row but no evaluate_model JSON "
            "beside it (sl_comparison/results/SLMGAE_cv3.json was moved "
            "aside") for m in miss), miss
        assert "sl_comparison/results/pca_joint/SLGNN_cv2.json is a SLGNN " \
            "row in PCA arm joint's directory: not read (the shared models' " \
            "rows are sl_comparison/results/'s)" in nums["notes"], \
            nums["notes"]
        rows = nums["items"]["fold_provenance"]["value"]["rows"]
        assert "siamese_sl/cv2" not in rows and "SLGNN/cv2" in rows
        # Arm none's own row JSON is still read by arm none.
        assert _run(root) == INCOMPLETE
        _close(_one(_rows(_out(root) / "tab_tiers.csv"), cv="cv2",
                    model="siamese_sl")["auprg_mean"],
               CMP["siamese_sl"]["cv2"][2])


def test_arm_single_embedding_root():
    """An arm's single-modality runs are read from the root its selection's
    _selection_meta.step1_root records (arm both: its pair's owner
    per_modality's), else from config.conf's owner rule over --pca_arms
    (per_modality with all four, both's own root when it is alone), noted. A
    recorded root outside the arm's pair is listed (and read: native runs
    are then stale against both's per-modality PCA); one not inside
    siamese_sl/ is listed and the owner rule read instead."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "per_modality")
        _build_arm(root, "both")

        def ea():
            return _one(_rows(_out(root, "both") / "tab_solo.csv"),
                        variant="pd:per_head", cv="cv1", embedding="ea")

        assert _run(root, arm="both") == INCOMPLETE
        nums = _numbers(root, "both")
        assert nums["run"]["step1_results"] == \
            "siamese_sl/results/pca_per_modality"
        assert ea()["status"] == "ok" and ea()["dir"] == \
            "siamese_sl/results/pca_per_modality/pcavar0.8_ea_cv1", ea()
        # Both alone (its own owner by the rule) but its selection records
        # its pair's other root: read as recorded, noted, nothing listed (a
        # standalone rerun of one arm without $PCA_ARMS stays complete).
        assert _run(root, "--pca_arms", "both", arm="both") == INCOMPLETE
        nums = _numbers(root, "both")
        assert nums["run"]["step1_results"] == \
            "siamese_sl/results/pca_per_modality"
        assert ("single-embedding runs read from siamese_sl/results/"
                "pca_per_modality/ (the selection's _selection_meta."
                "step1_root), not siamese_sl/results/pca_both/ (config.conf's "
                "owner rule over --pca_arms both)") in nums["notes"], \
            nums["notes"]
        assert not any("step1_root" in m for m in nums["missing"]), \
            nums["missing"]
        assert ea()["status"] == "ok", ea()
        sp = _sia(root, "both") / "best_per_category.json"
        sel = json.loads(sp.read_text())
        del sel["_selection_meta"]["step1_root"]
        _wj(sp, sel)
        assert _run(root, arm="both") == INCOMPLETE
        nums = _numbers(root, "both")
        assert nums["run"]["step1_results"] == \
            "siamese_sl/results/pca_per_modality"
        assert "siamese_sl/results/pca_both/best_per_category.json records " \
            "no _selection_meta.step1_root (a selection from before the PCA " \
            "arms): single-embedding runs read from siamese_sl/results/" \
            "pca_per_modality/ (config.conf's owner rule over --pca_arms)" \
            in nums["notes"], nums["notes"]
        # both alone owns its pair's runs, and its root holds none: tab_solo
        # has no source (exit 1).
        assert _run(root, "--pca_arms", "both", arm="both") == 1
        nums = _numbers(root, "both")
        assert nums["run"]["step1_results"] == "siamese_sl/results/pca_both"
        assert nums["tables"]["tab_solo"]["status"] == "missing"
        assert "siamese_sl/results/pca_both/pcavar0.8_ea_cv1/results.json" \
            in nums["missing"], nums["missing"]
        # A root of the other pair: listed, and read as recorded, so its
        # native runs are stale against both's per-modality PCA (every one:
        # tab_solo has no fresh row, exit 1).
        sel["_selection_meta"]["step1_root"] = "results"
        _wj(sp, sel)
        assert _run(root, arm="both") == 1
        nums = _numbers(root, "both")
        assert nums["run"]["step1_results"] == "siamese_sl/results"
        assert "siamese_sl/results/pca_both/best_per_category.json " \
            "_selection_meta.step1_root 'results' is not a single-embedding " \
            "root of PCA arm both's pair (results/pca_per_modality or " \
            "results/pca_both): tab_solo and the selection check read " \
            "siamese_sl/results/, as recorded" in nums["missing"], \
            nums["missing"]
        assert ("siamese_sl/results/pcavar0.8_ea_cv1 is stale "
                "(arch_signature differs from the combo: pca_variance)") in \
            nums["missing"], nums["missing"]
        for bad in ("/abs/results", "../results"):
            sel["_selection_meta"]["step1_root"] = bad
            _wj(sp, sel)
            assert _run(root, arm="both") == INCOMPLETE
            nums = _numbers(root, "both")
            assert nums["run"]["step1_results"] == \
                "siamese_sl/results/pca_per_modality", bad
            assert any(f"_selection_meta.step1_root {bad!r} is not a path "
                       f"inside siamese_sl/" in m for m in nums["missing"]), \
                (bad, nums["missing"])


def test_combo_of_another_arm_or_namespace_is_not_used():
    """A combo whose config.json is another arm's PCA (a POST_PCA_* export
    from before the arms, a hand copy) is listed and not used, as is one a
    marker names outside the arm's own results root (arm none's combo of the
    same name, which IS joint-free); a selection recording another arm is
    listed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        p = _sia(root) / "best_cat_ea_ka_pa_cv1" / "config.json"
        _wj(p, {**json.loads(p.read_text()), **ARM_CFG["joint"]})
        # ...and one whose PCA settings are not train.py's numbers.
        p = _sia(root) / "var_bilinear_pooled__best_cat_ea_ka_pa_cv3" / \
            "config.json"
        _wj(p, {**json.loads(p.read_text()), "pca_dims": ["x"]})
        _w(_sia(root, "joint") / "var_pd_pooled__latest_combo_cv2.txt",
           "results/var_pd_pooled__best_cat_ea_ka_pa_cv2\n")
        sp = _sia(root, "joint") / "best_per_category.json"
        sel = json.loads(sp.read_text())
        sel["_selection_meta"]["pca_arm"] = "none"
        _wj(sp, sel)
        assert _run(root) == INCOMPLETE
        nums = _numbers(root)
        assert "siamese_sl/results/best_cat_ea_ka_pa_cv1 (pd:per_head cv1 " \
            "combo) was trained as PCA arm joint (PCA within each modality " \
            "no, on the concatenation variance 0.8), not none: not used " \
            "(another arm's run, or a hand copy)" in nums["missing"], \
            nums["missing"]
        tiers = _rows(_out(root) / "tab_tiers.csv")
        assert _one(tiers, cv="cv1", model="siamese_sl")["params_fold_mean"] \
            == ""
        assert _one(tiers, cv="cv2", model="siamese_sl")["params_fold_mean"] \
            != ""
        assert nums["items"]["pca_width"]["value"]["per_variant"][
            "pd:per_head"]["min"] is None
        assert _one(_rows(_out(root) / "tab_loco.csv"), variant="pd:per_head",
                    cv="cv1", dropped="none")["status"] == "missing"
        assert any(m.startswith(
            "siamese_sl/results/var_bilinear_pooled__best_cat_ea_ka_pa_cv3 "
            "(bilinear:pooled cv3 combo): its config.json's PCA settings are "
            "unreadable (ValueError: ") and m.endswith(": not used")
            for m in nums["missing"]), nums["missing"]
        # The other (arm none) combos stand, and their PCA is reported.
        assert nums["run"]["pca_settings"] == {"modality": "no", "joint": "no"}
        assert _run(root, arm="joint") == INCOMPLETE
        miss = _numbers(root, "joint")["missing"]
        assert "siamese_sl/results/var_pd_pooled__best_cat_ea_ka_pa_cv2 " \
            "(pd:pooled cv2 combo, named by siamese_sl/results/pca_joint/" \
            "var_pd_pooled__latest_combo_cv2.txt) is not in PCA arm joint's " \
            "results root siamese_sl/results/pca_joint/: not used" in miss, \
            miss
        assert any(m.startswith(
            "siamese_sl/results/pca_joint/best_per_category.json records PCA "
            "arm 'none', not joint: another arm's selection") for m in miss), \
            miss
        assert _one(_rows(_out(root, "joint") / "tab_tiers.csv"), cv="cv2",
                    model="siamese_sl_pd_pooled")["params_fold_mean"] == ""


def test_arm_labels_and_targets():
    """Every output says which arm: README line 1 and its first Run settings
    bullet (config.conf pca_arm_label's words from this job's targets),
    numbers.json run.pca_*, the leading pca_arm column of every CSV.
    run.pca_settings is what the combos trained with; combos trained off
    this job's targets are one Missing entry (the label is the targets')."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "per_modality")
        _build_arm(root, "both")
        label = ("PCA within each modality (variance 0.8), then on their "
                 "concatenation (variance 0.8)")
        assert _run(root, arm="both") == INCOMPLETE
        nums = _numbers(root, "both")
        assert nums["run"]["pca"] == label
        assert nums["run"]["pca_settings"] == {"modality": "variance 0.8",
                                               "joint": "variance 0.8"}
        assert not any("not this job's targets" in m
                       for m in nums["missing"]), nums["missing"]
        readme = (_out(root, "both") / "README.md").read_text()
        assert readme.splitlines()[0] == \
            "# Paper tables and numbers: PCA arm both"
        bullet = _section(readme, "Run settings").strip().splitlines()[0]
        shared = ("single-embedding runs with PCA within the modality "
                  "(variance 0.8), shared by arms per_modality and both")
        assert bullet.startswith(
            f"- PCA arm: both ({label}); siamese results siamese_sl/results/"
            f"pca_both/, comparison results sl_comparison/results/pca_both/ "
            f"(the shared models' rows from sl_comparison/results/); single-"
            f"embedding runs siamese_sl/results/pca_per_modality/ "
            f"({shared})"), bullet
        assert nums["run"]["step1_pca"] == shared
        # A tables run of both alone (a later --skip-siamese with another
        # PCA_ARMS): the label the combo job recorded with those runs still
        # says who shares them; without one, config.conf's wording over this
        # run's --pca_arms.
        assert _run(root, "--pca_arms", "both", arm="both") == INCOMPLETE
        assert _numbers(root, "both")["run"]["step1_pca"] == shared
        sp = _sia(root, "both") / "best_per_category.json"
        sel = json.loads(sp.read_text())
        del sel["_selection_meta"]["step1_pca"]
        _wj(sp, sel)
        assert _run(root, "--pca_arms", "both", arm="both") == INCOMPLETE
        alone = ("single-embedding runs with PCA within the modality "
                 "(variance 0.8), used by arm both only")
        assert _numbers(root, "both")["run"]["step1_pca"] == alone
        assert f"({alone}); its combos trained with" in (
            _out(root, "both") / "README.md").read_text()
        assert _run(root, arm="both") == INCOMPLETE
        assert _numbers(root, "both")["run"]["step1_pca"] == shared
        # The joint target a count: the label follows it, the combos (at
        # variance 0.8) are listed once.
        assert _run(root, arm="both", env={"PCA_JOINT_DIM": "300"}) == \
            INCOMPLETE
        nums = _numbers(root, "both")
        assert nums["run"]["pca"] == (
            "PCA within each modality (variance 0.8), then on their "
            "concatenation (300 components)")
        assert nums["run"]["pca_targets"]["PCA_JOINT_DIM"] == "300"
        assert nums["run"]["pca_settings"] == {"modality": "variance 0.8",
                                               "joint": "variance 0.8"}
        off = [m for m in nums["missing"]
               if m.startswith("PCA arm both's combo(s) ")]
        assert len(off) == 1 and (
            "trained with PCA within each modality variance 0.8, on the "
            "concatenation variance 0.8, not this job's targets (variance "
            "0.8 / count 300: PCA_MODALITY_VARIANCE=0.8, PCA_JOINT_VARIANCE="
            "0.8, PCA_JOINT_DIM=300): the label 'PCA within each modality "
            "(variance 0.8), then on their concatenation (300 components)' "
            "does not describe them") in off[0], off
        # Arm none: no PCA, and so its numbers and CSVs say.
        assert _run(root) == INCOMPLETE
        nums = _numbers(root)
        assert (nums["run"]["pca_arm"], nums["run"]["pca"],
                nums["run"]["pca_settings"]) == (
            "none", "no PCA", {"modality": "no", "joint": "no"})
        for t in pt.TABLES:
            with open(_out(root) / f"{t}.csv") as fh:
                rows = list(csv.reader(fh))
            assert rows[0][0] == "pca_arm" and \
                {x[0] for x in rows[1:]} == {"none"}, t


def test_cellline_source_compared_as_a_path():
    """zero_cell_emb.py records the combo it ablated as the marker's
    siamese_sl-relative text, and arm none's results/best_cat_<names>_<cv>
    shares its NAME with arm X's results/pca_X/best_cat_<names>_<cv>:
    compared as paths, an ablation (or a graded _nocell row) of arm none's
    combo is not arm joint's; a record of the arm's own path, or of the same
    directory spelt from the root, is; a bare name is compared by name."""
    import types
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        ctx = types.SimpleNamespace(root=root)
        d = root / "siamese_sl" / "results" / "pca_joint" / "best_cat_x_cv1"
        f = pt.recorded_dir_differs
        for rec, differs in (("results/pca_joint/best_cat_x_cv1", False),
                             ("siamese_sl/results/pca_joint/best_cat_x_cv1",
                              False),
                             (str(d), False),
                             ("results/best_cat_x_cv1", True),
                             (str(root / "siamese_sl" / "results" /
                                  "best_cat_x_cv1"), True),
                             ("best_cat_x_cv1", False),
                             ("best_cat_y_cv1", True)):
            assert f(ctx, rec, d) is differs, rec
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        p = _sia(root, "joint") / "cellline_ablation_cv1" / "results.json"
        clean = json.loads(p.read_text())
        # cv2 made clean, so the table keeps one tabulated removed row.
        agg2 = {m: [x + ARM_SHIFT["joint"], .01]
                for m, x in zip(pt.METRICS, CMP["siamese_sl"]["cv2"])}
        _wj(_sia(root, "joint") / "cellline_ablation_cv2" / "results.json",
            {**CELLLINE_OK, "agg": {"as-trained": agg2}})

        def rem():
            return _one(_rows(_out(root, "joint") / "tab_cellline.csv"),
                        variant="pd:per_head", cv="cv1",
                        condition="removed (cell_emb=0)")

        assert _run(root, arm="joint") == INCOMPLETE
        assert rem()["status"] == "ok", rem()["status"]
        _wj(p, {**clean, "source_model_dir": "results/best_cat_ea_ka_pa_cv1"})
        assert _run(root, arm="joint") == INCOMPLETE
        assert rem()["status"] == (
            "stale: zero_cell_emb.py ablated results/best_cat_ea_ka_pa_cv1, "
            "not the latest_combo combo siamese_sl/results/pca_joint/"
            "best_cat_ea_ka_pa_cv1"), rem()["status"]
        _wj(p, clean)
        q = _cmp(root, "joint") / "siamese_sl_nocell_cv1.json"
        row = json.loads(q.read_text())
        row["summary"]["source_combo"] = "results/best_cat_ea_ka_pa_cv1"
        _wj(q, row)
        assert _run(root, arm="joint") == INCOMPLETE
        assert rem()["status"] == (
            "stale: siamese_sl_nocell_cv1.json was graded from an ablation of "
            "results/best_cat_ea_ka_pa_cv1, not the latest_combo combo "
            "siamese_sl/results/pca_joint/best_cat_ea_ka_pa_cv1"), \
            rem()["status"]
        row["summary"]["source_combo"] = \
            "results/pca_joint/best_cat_ea_ka_pa_cv1"
        _wj(q, row)
        assert _run(root, arm="joint") == INCOMPLETE
        assert rem()["status"] == "ok", rem()["status"]
        # Arm joint's combo retrained in place since the ablation: both
        # reruns name the arm (without PCA_ARM, run_cell_line_ablation.sh
        # redoes arm none's), and the combo is named by its path.
        cp = _sia(root, "joint") / "best_cat_ea_ka_pa_cv1" / "results.json"
        c = json.loads(cp.read_text())
        for m in c["fold_metrics"]:
            m.update(auroc_any=.90 + m["fold"] / 100,
                     aupr_any=.80 + m["fold"] / 100)
        _wj(cp, c)
        _wj(p, {**clean, "self_check": {
            "ok": True, "per_fold": _cellline_refs(c["fold_metrics"])}})
        assert _run(root, arm="joint") == INCOMPLETE
        assert rem()["status"] == "ok", rem()["status"]
        c["fold_metrics"][0]["auroc_any"] += 1e-9
        _wj(cp, c)
        assert _run(root, arm="joint") == INCOMPLETE
        st = rem()["status"]
        assert st.startswith(
            "stale: zero_cell_emb.py ablated siamese_sl/results/pca_joint/"
            "best_cat_ea_ka_pa_cv1 when its fold 0 auroc_any was "), st
        assert st.endswith(
            "(the combo was retrained in place since: from siamese_sl/ "
            "resubmit cell_line_ablation/run_cell_line_ablation.sh with "
            "PCA_ARM=joint, then from the repo root sl_comparison/slurm/"
            "run_compare.sh with PCA_ARM=joint)"), st


def test_arm_paths_as_the_wrapper_passes_them():
    """run_paper_tables.sh passes the arm's --out_dir / --comparison_csv /
    --external_dir relative to the repo root, its working directory: they
    are read there, and the arm's own comparison.csv so spelt is still the
    DEFAULT table, whose row without its JSON beside it is refused."""
    import contextlib
    import io
    import os
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        (_cmp(root, "joint") / "siamese_sl_pd_pooled_cv2.json").unlink()
        cj = "sl_comparison/results/pca_joint"
        os.chdir(root)
        try:
            with _PcaEnv(), contextlib.redirect_stdout(io.StringIO()):
                rc = pt.main(["--root", str(root), "--variants", *VARIANTS,
                              "--selection_variant", "pd:per_head",
                              "--run_token", TOKEN, "--code_manifest", "",
                              "--pca_arm", "joint", "--pca_arms", *ARMS,
                              "--out_dir", f"{cj}/paper_tables",
                              "--comparison_csv", f"{cj}/comparison.csv",
                              "--external_dir", f"{cj}/external_transfer"])
        finally:
            os.chdir(here)
        assert rc == INCOMPLETE
        nums = _numbers(root, "joint")
        assert all(t["status"] == "ok" for t in nums["tables"].values())
        assert any(m.startswith("siamese_sl_pd_pooled/cv2: comparison.csv "
                                "has a row but no evaluate_model JSON beside "
                                "it") for m in nums["missing"]), \
            nums["missing"]
        assert _one(_rows(_out(root, "joint") / "tab_tiers.csv"), cv="cv2",
                    model="siamese_sl_pd_pooled")["status"] == "missing"


def test_comparison_csv_of_another_arm_is_not_read():
    """compare.py --pca_arm writes the arm in a trailing pca_arm column: a
    comparison.csv naming another arm (one copied into this namespace) is
    listed and not read, its rows falling back to the arm's own JSONs; a
    table from before the column (arm none's fixture) is read as before."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        p = _cmp(root, "joint") / "comparison.csv"
        rows = _rows(p)
        assert {r["pca_arm"] for r in rows} == {"joint"}
        for r in rows:
            r["pca_arm"] = "none"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        assert _run(root, arm="joint") == INCOMPLETE
        nums = _numbers(root, "joint")
        assert "sl_comparison/results/pca_joint/comparison.csv names PCA " \
            "arm(s) none in its pca_arm column, not joint: another arm's " \
            "table, not read (rerun sl_comparison/slurm/run_compare.sh with " \
            "PCA_ARM=joint)" in nums["missing"], nums["missing"]
        r = _one(_rows(_out(root, "joint") / "tab_tiers.csv"), cv="cv1",
                 model="siamese_sl")
        assert r["source"] == "json", r["source"]
        _close(r["auprg_mean"],
               CMP["siamese_sl"]["cv1"][2] + ARM_SHIFT["joint"])


def test_combo_linked_from_other_storage_is_the_arm_s():
    """A combo directory moved to other storage and linked back in place is
    still the arm's, as collect_siamese.py (its name in the arm root) and
    run_external_transfer.sh (its parent, resolved) take it: used, and its
    cell-line ablation's recorded path still matches. A marker whose text
    climbs out of the arm's root with '..' is not the arm's."""
    import os
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        d = _sia(root, "joint") / "best_cat_ea_ka_pa_cv1"
        store = root / "store"
        store.mkdir()
        shutil.move(str(d), str(store / d.name))
        os.symlink(str(store / d.name), str(d))
        _w(_sia(root, "joint") / "latest_combo_cv2.txt",
           "results/pca_joint/../best_cat_ea_ka_pa_cv2\n")
        assert _run(root, arm="joint") == INCOMPLETE
        out = _out(root, "joint")
        miss = _numbers(root, "joint")["missing"]
        assert not any("best_cat_ea_ka_pa_cv1" in m for m in miss), miss
        _close(_one(_rows(out / "tab_tiers.csv"), cv="cv1",
                    model="siamese_sl")["params_fold_mean"],
               29022 - ARM_PARAMS_CUT["joint"])
        assert _one(_rows(out / "tab_loco.csv"), variant="pd:per_head",
                    cv="cv1", dropped="none")["status"] == "ok"
        assert _one(_rows(out / "tab_cellline.csv"), variant="pd:per_head",
                    cv="cv1", condition="removed (cell_emb=0)")[
            "status"] == "ok"
        assert "siamese_sl/results/pca_joint/../best_cat_ea_ka_pa_cv2 " \
            "(pd:per_head cv2 combo, named by siamese_sl/results/pca_joint/" \
            "latest_combo_cv2.txt) is not in PCA arm joint's results root " \
            "siamese_sl/results/pca_joint/: not used" in miss, miss
        assert _one(_rows(out / "tab_tiers.csv"), cv="cv2",
                    model="siamese_sl")["params_fold_mean"] == ""


def test_pca_settings_from_config_conf():
    """The PCA targets and the run's arms come from the environment, else
    siamese_sl/slurm/config.conf (as the variant settings do), else its
    defaults: a standalone run_paper_tables.sh (nothing exported) after a
    default edited there labels and checks the targets every training job
    sourced, and --pca_arms_table compares config.conf's arms. The
    environment still wins. The caller's other pipeline settings are
    cleared for the whole check (an exported ARCH_ARGS, SELECTION_VARIANT or
    POST_PCA_* would reach the sourcing subshell), as in
    test_config_conf_fallback."""
    with _EnvCleared("SIAMESE_VARIANTS_LIST", "SELECTION_VARIANT", "ARCH_ARGS",
                     "RUN_TOKEN", "BESTCAT_EXTRA_VARIANTS",
                     "BEST_CAT_PCA_VARIANCE", "CODE_MANIFEST",
                     "POST_PCA_VARIANCE", "POST_PCA_DIM"):
        _pca_settings_from_config_conf()


def _pca_settings_from_config_conf():
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _build_arm(root, "joint")
        _w(root / "siamese_sl" / "slurm" / "config.conf",
           'SIAMESE_VARIANTS=("pd:per_head" "pd:pooled" "bilinear:per_head" '
           '"bilinear:pooled")\n'
           'SELECTION_VARIANT="${SELECTION_VARIANT:-pd:per_head}"\n'
           'ARCH_ARGS="${ARCH_ARGS:-}"\n'
           'PCA_ARMS="${PCA_ARMS:-joint none}"\n'
           'PCA_JOINT_VARIANCE="${PCA_JOINT_VARIANCE:-0.9}"\n'
           # as the real one refuses the replaced knobs
           '[ -z "${POST_PCA_VARIANCE:-}" ] || return 1\n')
        with _PcaEnv():
            assert pt.read_config_conf(root) == {
                "SIAMESE_VARIANTS_LIST": " ".join(VARIANTS),
                "SELECTION_VARIANT": "pd:per_head", "ARCH_ARGS": "",
                "PCA_ARMS": "joint none", "PCA_JOINT_VARIANCE": "0.9"}
        # Every multi-modal run of arm joint (combos, ablations) at 0.9.
        for p in _sia(root, "joint").glob("*/config.json"):
            _wj(p, {**json.loads(p.read_text()), "post_pca": 0.9,
                    "post_pca_variance": 0.9})

        def run(*extra, **env):
            with _PcaEnv(**env), contextlib.redirect_stdout(io.StringIO()):
                return pt.main(["--root", str(root), "--variants", *VARIANTS,
                                "--selection_variant", "pd:per_head",
                                "--run_token", TOKEN,
                                "--bestcat_extra_variants", "0",
                                "--pca_variance", "0.8", "--code_manifest",
                                "", *extra])

        assert run("--pca_arm", "joint") == INCOMPLETE
        nums = _numbers(root, "joint")
        assert nums["run"]["pca"] == ("PCA on the concatenation of all "
                                      "modalities (variance 0.9)")
        assert nums["run"]["pca_targets"] == {"PCA_MODALITY_VARIANCE": "0.8",
                                              "PCA_JOINT_VARIANCE": "0.9",
                                              "PCA_JOINT_DIM": None}
        assert nums["run"]["pca_settings"] == {"modality": "no",
                                               "joint": "variance 0.9"}
        assert nums["run"]["pca_arms"] == ["none", "joint"]
        assert not any("not this job's targets" in m
                       for m in nums["missing"]), nums["missing"]
        assert all(t["status"] == "ok" for t in nums["tables"].values())
        # The environment wins over config.conf.
        assert run("--pca_arm", "joint", PCA_JOINT_VARIANCE="0.8") == \
            INCOMPLETE
        nums = _numbers(root, "joint")
        assert nums["run"]["pca"] == ("PCA on the concatenation of all "
                                      "modalities (variance 0.8)")
        assert len([m for m in nums["missing"]
                    if "not this job's targets" in m]) == 1, nums["missing"]
        assert run("--pca_arm", "joint", PCA_ARMS="joint") == INCOMPLETE
        assert _numbers(root, "joint")["run"]["pca_arms"] == ["joint"]
        # A config.conf that refuses the environment is not read: noted, and
        # the defaults' label no longer describes the combos (listed).
        import os
        with _EnvCleared("POST_PCA_VARIANCE"):
            os.environ["POST_PCA_VARIANCE"] = "0.8"
            try:
                assert run("--pca_arm", "joint") == INCOMPLETE
            finally:
                os.environ.pop("POST_PCA_VARIANCE", None)
        nums = _numbers(root, "joint")
        assert any(n.startswith("siamese_sl/slurm/config.conf could not be "
                                "sourced") for n in nums["notes"]), \
            nums["notes"]
        assert nums["run"]["pca_targets"]["PCA_JOINT_VARIANCE"] == "0.8"
        assert any("not this job's targets" in m for m in nums["missing"])
        # --pca_arms_table without --pca_arms: config.conf's two arms.
        _arm_readme(root, "none")
        _arm_readme(root, "joint")
        assert run("--pca_arms_table") == 0
        nums = _arms_numbers(root)
        assert nums["run"]["pca_arms"] == ["none", "joint"] and \
            nums["missing"] == [], nums["missing"]
        assert nums["items"]["pca"]["value"]["settings"]["trained"][
            "joint"] == {"modality": "no", "joint": "variance 0.9"}


# =============================================================================
# PCA arms: the cross-arm tab_pca (--pca_arms_table)
# =============================================================================


def _arms_numbers(root: Path) -> dict:
    return json.loads((_arms_out(root) / "numbers.json").read_text())


def _set_graded(root: Path, arm: str, model: str, cv: str, metric: str,
                value: float) -> None:
    """One graded mean of an arm's row in both copies compare.py keeps (its
    comparison.csv cell and its row JSON, whose per-fold records move with
    it, so they still average to it)."""
    p = _cmp(root, arm) / "comparison.csv"
    rows = _rows(p)
    for r in rows:
        if (r["cv"], r["model"], r["metric"]) == (cv, model, metric):
            r["mean"] = repr(value)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    q = _cmp(root, arm) / f"{model}_{cv}.json"
    row = json.loads(q.read_text())
    s = row["summary"]
    for fold in s["folds"]:
        fold[metric] += value - s[f"{metric}_mean"]
    s[f"{metric}_mean"] = value
    _wj(q, row)


def test_pca_arms_delta_reading():
    """tab_pca reads an AUPRG difference as the ablation summaries read
    theirs: |delta| <= 0.005 is no measurable difference (whatever the
    folds); beyond it better / worse only past 2 SE, else within fold noise;
    n/a with fewer than two paired folds, or no difference at all."""
    f = pt.delta_verdict
    assert f(None, None, None) == "n/a (no auprg on one side)"
    assert f(.004, None, None) == "no measurable difference"
    assert f(-.005, .1, 5) == "no measurable difference"
    assert f(.02, None, 1) == "n/a (fewer than 2 paired folds)"
    assert f(.02, .5, 1) == "n/a (fewer than 2 paired folds)"
    assert f(.02, .003, 2) == "better than the reference"
    assert f(-.02, .003, 5) == "worse than the reference"
    assert f(.02, .01, 5) == "within fold noise (|delta| <= 2 SE)"
    assert f(-.02, .01, 5) == "within fold noise (|delta| <= 2 SE)"


def test_pca_arms_table():
    """--pca_arms_table on a four-arm tree whose arms' own paper tables all
    say PAPER COMPLETE: tab_pca holds every variant x tier x arm (in the
    fixed order, arm none the reference) from each arm's GRADED rows, its
    combo's PCA, single-embedding runs, modalities, widths and size, the
    differences from arm none (of means; the AUPRG's spread fold-paired by
    position) and their reading, best_arm; items.pca the 2x2 effects and
    each arm's selection; the README the arms' index. PAPER COMPLETE, exit
    0, and nothing written outside sl_comparison/results/pca_arms/."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        readmes = {a: (_out(root, a) / "README.md").read_text() for a in ARMS}
        rc = _run_arms(root)
        nums = _arms_numbers(root)
        assert rc == 0 and nums["missing"] == [] and \
            nums["verdict"] == pt.VERDICT_COMPLETE, (rc, nums["missing"])
        out = _arms_out(root)
        assert sorted(p.name for p in out.iterdir()) == [
            "README.md", "numbers.json", "tab_pca.csv"]
        assert {a: (_out(root, a) / "README.md").read_text()
                for a in ARMS} == readmes
        rows = _rows(out / "tab_pca.csv")
        assert len(rows) == len(VARIANTS) * len(pt.CVS) * len(ARMS)
        assert all(r["status"] == "ok" for r in rows)
        with open(out / "tab_pca.csv") as fh:
            assert next(csv.reader(fh)) == pt.PCA_ARMS_COLS
        cell = [r for r in rows if (r["variant"], r["cv"]) == (
            "pd:per_head", "cv1")]
        assert [(r["pca_arm"], r["role"]) for r in cell] == [
            ("none", "reference"), ("joint", "arm"), ("per_modality", "arm"),
            ("both", "arm")]
        pca = {"none": ("no", "no", "native"),
               "joint": ("no", "variance 0.8", "native"),
               "per_modality": ("variance 0.8", "no",
                                "per-modality PCA variance 0.8"),
               "both": ("variance 0.8", "variance 0.8",
                        "per-modality PCA variance 0.8")}
        x = CMP["siamese_sl"]["cv1"]
        for r in cell:
            a = r["pca_arm"]
            assert (r["modality_pca"], r["joint_pca"],
                    r["single_embedding"]) == pca[a], r
            assert (r["modalities"], r["n_modalities"],
                    r["same_modalities_as_reference"]) == ("ea+pa+ka", "3",
                                                           "True")
            assert r["dir"] == \
                f"siamese_sl/{_arm_rel(a)}/best_cat_ea_ka_pa_cv1"
            for k, m in enumerate(pt.METRICS):
                _close(r[f"{m}_mean"], x[k] + ARM_SHIFT[a])
                _close(r[f"delta_{m}"], ARM_SHIFT[a])
            _close(r["auprg_sd"], .02)
            cut = ARM_WIDTH_CUT[a]
            _close(r["input_dim_mean"], 1507.5 - cut)
            assert (r["input_dim_min"], r["input_dim_max"]) == (
                str(1505 - cut), str(1510 - cut))
            _close(r["params_fold_mean"], 29022 - ARM_PARAMS_CUT[a])
        assert cell[0]["verdict"] == "n/a (the reference arm)" and \
            cell[0]["n_paired"] == "" and cell[0]["delta_auprg_se"] == ""
        for a, want in (("joint", "no measurable difference"),
                        ("per_modality", "better than the reference"),
                        ("both", "within fold noise (|delta| <= 2 SE)")):
            r = _one(cell, pca_arm=a)
            e = abs(FOLD_SPREAD[a] - FOLD_SPREAD["none"])
            _close(r["delta_auprg_paired_sd"], e)
            _close(r["delta_auprg_se"], e)
            assert r["n_paired"] == "2" and r["verdict"] == want, (a, r)
        assert [r["best_arm"] for r in cell] == ["False", "False", "True",
                                                 "False"]
        # Another variant x tier: its own graded rows, the same reading.
        bp = _one(rows, variant="bilinear:pooled", cv="cv3",
                  pca_arm="per_modality")
        _close(bp["auprg_mean"], CMP["siamese_sl_bilinear_pooled"]["cv3"][2] +
               .02)
        assert bp["verdict"] == "better than the reference" and \
            bp["best_arm"] == "True"
        it = nums["items"]["pca"]
        assert it["status"] == "ok" and \
            it["paper"] == "tab:pca (not yet in paper/main.tex)"
        v = it["value"]
        assert v["reference_arm"] == "none" and v["arms"] == list(ARMS)
        p1 = v["pd:per_head"]["cv1"]
        assert p1["arms_missing"] == [] and p1["best_arm"] == "per_modality" \
            and p1["best_arms"] == ["per_modality"]
        pm = p1["arms"]["per_modality"]
        _close(pm["auprg"]["mean"], x[2] + .02)
        _close(pm["delta_auprg"], .02)
        _close(pm["paired"]["mean"], .02)
        _close(pm["paired"]["sd"], .003)
        _close(pm["paired"]["se"], .003)
        assert pm["paired"]["n"] == 2 and \
            pm["verdict"] == "better than the reference"
        assert pm["modalities"] == ["ea", "pa", "ka"] and \
            pm["input_dim"] == {"min": 1005, "max": 1010, "mean": 1007.5}
        assert p1["arms"]["none"]["paired"] is None
        # The 2x2 on the graded per-fold AUPRG (arm a's fold k at its mean
        # -/+ FOLD_SPREAD[a]; s = ARM_SHIFT): per_modality = ((s_pm + s_both
        # - s_joint) -/+ (e_pm - e_none + e_both - e_joint)) / 2 = .0165 -/+
        # .0055, joint = ((s_joint + s_both - s_pm) -/+ (e_joint - e_none +
        # e_both - e_pm)) / 2 = -.0065 -/+ .0045, interaction = (s_both -
        # s_pm - s_joint) -/+ (e_both - e_pm - e_joint + e_none) = -.007 -/+
        # .005.
        for name, mean, sd in (("per_modality", .0165, .0055),
                               ("joint", -.0065, .0045),
                               ("interaction", -.007, .005)):
            e = p1["effects"][name]
            _close(e["mean"], mean)
            _close(e["sd"], sd)
            _close(e["se"], sd)
            assert e["n"] == 2, name
        sel = v["selection"]
        assert sel["winners"] == {a: WINNERS for a in ARMS}
        assert sel["same_selection"] == [list(ARMS)]
        assert sel["step1_results"] == {
            "none": "siamese_sl/results", "joint": "siamese_sl/results",
            "per_modality": "siamese_sl/results/pca_per_modality",
            "both": "siamese_sl/results/pca_per_modality"}
        assert sel["run_token"] == dict.fromkeys(ARMS, TOKEN)
        assert sel["selection_cv"] == dict.fromkeys(ARMS, "cv1")
        assert (v["settings"]["modality"], v["settings"]["joint"]) == (
            "variance 0.8", "variance 0.8")
        assert v["settings"]["trained"]["joint"] == {
            "modality": "no", "joint": "variance 0.8"}
        assert [r["arm"] for r in nums["arms"]] == list(ARMS)
        assert nums["arms"][2] == {
            "arm": "per_modality",
            "pca": "PCA within each modality (variance 0.8)",
            "sia_results": "siamese_sl/results/pca_per_modality",
            "cmp_results": "sl_comparison/results/pca_per_modality",
            "step1_results": "siamese_sl/results/pca_per_modality",
            "paper_tables": "sl_comparison/results/pca_per_modality/"
                            "paper_tables",
            "verdict": pt.VERDICT_COMPLETE}
        assert nums["run"]["pca_arms"] == list(ARMS) and \
            nums["run"]["reference_arm"] == "none" and \
            nums["mode"] == "pca_arms_table"
        readme = (out / "README.md").read_text()
        assert readme.splitlines()[:3] == [
            "# PCA arms: tab_pca and every arm's verdict", "",
            pt.VERDICT_COMPLETE]
        assert ("| both | PCA within each modality (variance 0.8), then on "
                "their concatenation (variance 0.8) | siamese_sl/results/"
                "pca_both/ | sl_comparison/results/pca_both/ | siamese_sl/"
                "results/pca_per_modality/ | sl_comparison/results/pca_both/"
                "paper_tables/ | PAPER COMPLETE |") in readme, readme
        assert "| tab_pca.csv | ok |" in readme
        assert _section(readme, "Missing inputs and items") == "\n- none\n"


def test_pca_arms_table_best_arm_on_a_tie():
    """An exact tie for the best graded AUPRG flags every tied arm in
    tab_pca.csv; numbers.json names them in best_arms and leaves best_arm
    empty (noted, not missing: the verdict stands). best_arm is decided on
    the exact values: two arms equal to 3 decimals but not exactly (cv3) are
    no tie."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        x = CMP["siamese_sl"]["cv2"][2]
        _set_graded(root, "both", "siamese_sl", "cv2", "auprg", x + .02)
        x3 = CMP["siamese_sl"]["cv3"][2]
        _set_graded(root, "both", "siamese_sl", "cv3", "auprg",
                    x3 + .02 - .0001)
        assert round(x3 + .02, 3) == round(x3 + .02 - .0001, 3)
        assert _run_arms(root) == 0
        cell = [r for r in _rows(_arms_out(root) / "tab_pca.csv")
                if (r["variant"], r["cv"]) == ("pd:per_head", "cv2")]
        assert [r["best_arm"] for r in cell] == ["False", "False", "True",
                                                 "True"]
        cell3 = [r for r in _rows(_arms_out(root) / "tab_pca.csv")
                 if (r["variant"], r["cv"]) == ("pd:per_head", "cv3")]
        assert [r["best_arm"] for r in cell3] == ["False", "False", "True",
                                                  "False"]
        nums = _arms_numbers(root)
        p3 = nums["items"]["pca"]["value"]["pd:per_head"]["cv3"]
        assert p3["best_arm"] == "per_modality" and \
            p3["best_arms"] == ["per_modality"]
        p2 = nums["items"]["pca"]["value"]["pd:per_head"]["cv2"]
        assert p2["best_arm"] is None and \
            p2["best_arms"] == ["per_modality", "both"]
        assert "numbers.json pca pd:per_head/cv2: per_modality, both tie at " \
            "the best auprg_mean; best_arm left empty (best_arms names them)" \
            in nums["notes"], nums["notes"]
        assert nums["missing"] == [], nums["missing"]


def test_pca_arms_table_with_three_arms():
    """A run of three arms, without arm none: the reference is the first in
    the fixed order (joint, however --pca_arms lists them), joint owns its
    pair's single-embedding runs, best_arm is decided among the three, and
    the 2x2 effects, which need all four arms, are absent by design (not
    missing: the verdict is complete)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arms = ("joint", "per_modality", "both")
        _build_arms(root, arms)
        assert _run_arms(root, arms=("both", "joint", "per_modality")) == 0
        nums = _arms_numbers(root)
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        assert len(rows) == len(VARIANTS) * len(pt.CVS) * 3
        cell = [r for r in rows if (r["variant"], r["cv"]) == (
            "pd:per_head", "cv1")]
        assert [(r["pca_arm"], r["role"]) for r in cell] == [
            ("joint", "reference"), ("per_modality", "arm"), ("both", "arm")]
        _close(_one(cell, pca_arm="per_modality")["delta_auprg"],
               ARM_SHIFT["per_modality"] - ARM_SHIFT["joint"])
        assert [r["best_arm"] for r in cell] == ["False", "True", "False"]
        v = nums["items"]["pca"]["value"]
        assert v["reference_arm"] == "joint" and v["arms"] == list(arms)
        assert v["pd:per_head"]["cv1"]["effects"] is None and \
            v["pd:per_head"]["cv1"]["best_arm"] == "per_modality"
        assert v["selection"]["step1_results"]["joint"] == \
            "siamese_sl/results/pca_joint"
        assert nums["absent_by_design"] == [
            "numbers.json pca effects: the 2x2's per-modality / joint / "
            "interaction effects need all four PCA arms; this run compares "
            "joint, per_modality, both (--pca_arms / PCA_ARMS), so they are "
            "left empty"], nums["absent_by_design"]
        assert nums["missing"] == [] and \
            nums["verdict"] == pt.VERDICT_COMPLETE, nums["missing"]


def test_pca_arms_table_stale_and_missing_cells():
    """A cell is not tabulated (no value, listed under its arm, its variant x
    tier's best_arm left empty) when its arm's selection is another
    submission's (run_token), its combo's arch_signature differs from the
    reference arm's outside the PCA keys, it was trained on another content
    of an embedding file the reference combo has, its arm has no combo, the
    combo is another arm's, or its graded row is another run than its combo
    (a *_selected_<cv>.txt pin: that run's AUPRG next to the combo's widths);
    the other cells stand."""
    A, B, C, D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        sp = _sia(root, "joint") / "best_per_category.json"
        sel = json.loads(sp.read_text())
        sel["_selection_meta"]["run_token"] = "T2"
        _wj(sp, sel)
        p = (_sia(root, "per_modality") / "best_cat_ea_ka_pa_cv2" /
             "config.json")
        _wj(p, {**json.loads(p.read_text()), "dropout": 0.5})
        _record_content(_sia(root) / "best_cat_ea_ka_pa_cv3", [A, B, C])
        _record_content(_sia(root, "both") / "best_cat_ea_ka_pa_cv3",
                        [A, D, C])
        (_sia(root, "both") / "var_pd_pooled__latest_combo_cv1.txt").unlink()
        p = (_sia(root, "per_modality") /
             "var_bilinear_pooled__best_cat_ea_ka_pa_cv1" / "config.json")
        cfg = {k: x for k, x in json.loads(p.read_text()).items()
               if k not in ("pca_variance", "pca_dims")}
        _wj(p, {**cfg, **ARM_CFG["none"]})
        # both's cv1 comparison row graded another run (a pin) than its combo
        _selected_marker(root, "pd:per_head", "cv1",
                         "siamese_sl/results/pca_both/pinned_run_cv1", "both")
        assert _run_arms(root) == INCOMPLETE
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        stale = "stale: its selection's run_token T2 is not the reference " \
            "arm none's TOK (another submission)"
        assert all(r["status"] == stale and r["auprg_mean"] == "" and
                   r["verdict"] == "n/a (stale)"
                   for r in rows if r["pca_arm"] == "joint")
        assert _one(rows, variant="pd:per_head", cv="cv2",
                    pca_arm="per_modality")["status"] == (
            "stale: its arch_signature differs from the reference arm none's "
            "combo outside the PCA keys: dropout")
        b = _one(rows, variant="pd:per_head", cv="cv3", pca_arm="both")
        # the comparator by its arm and path: the bare name is both's own too
        assert b["status"].startswith(
            "stale: trained on another content of all_genes_pa.pt than "
            "the reference arm none's combo siamese_sl/results/"
            "best_cat_ea_ka_pa_cv3 (embeddings_sha256"), b["status"]
        m = _one(rows, variant="pd:pooled", cv="cv1", pca_arm="both")
        assert m["status"] == ("missing: no var_pd_pooled__latest_combo_cv1"
                               ".txt (no combo; listed)") and m["dir"] == ""
        g = _one(rows, variant="pd:per_head", cv="cv1", pca_arm="both")
        assert g["status"] == ("stale: its comparison row graded another run "
                               "than this combo (listed)") and \
            g["auprg_mean"] == "", g
        o = _one(rows, variant="bilinear:pooled", cv="cv1",
                 pca_arm="per_modality")
        assert o["status"] == (
            "stale: trained as PCA arm none (PCA within each modality no, on "
            "the concatenation no), not per_modality (listed)"), o["status"]
        assert _one(rows, variant="pd:per_head", cv="cv2",
                    pca_arm="none")["status"] == "ok"
        assert _one(rows, variant="pd:per_head", cv="cv2",
                    pca_arm="both")["status"] == "ok"
        assert all(r["best_arm"] == "" for r in rows)  # joint is everywhere
        nums = _arms_numbers(root)
        miss = nums["missing"]
        for want in (
                "PCA arm joint: tab_pca siamese_sl/cv1: siamese_sl/results/"
                "pca_joint/best_cat_ea_ka_pa_cv1 is stale (its selection's "
                "run_token T2 is not the reference arm none's TOK (another "
                "submission))",
                "PCA arm per_modality: tab_pca siamese_sl/cv2: siamese_sl/"
                "results/pca_per_modality/best_cat_ea_ka_pa_cv2 is stale (its "
                "arch_signature differs from the reference arm none's combo "
                "outside the PCA keys: dropout)",
                "PCA arm both: siamese_sl/results/pca_both/"
                "var_pd_pooled__latest_combo_cv1.txt (pd:pooled cv1 combo)",
                "PCA arm both: tab_pca siamese_sl/cv1: siamese_sl/results/"
                "pca_both/best_cat_ea_ka_pa_cv1 is stale (its comparison row "
                "graded another run than this combo (listed))",
                "numbers.json pca pd:per_head/cv1: no value for arm(s) joint, "
                "both; best_arm and effects left empty"):
            assert want in miss, (want, miss)
        assert any(x.startswith(
            "PCA arm per_modality: siamese_sl/results/pca_per_modality/"
            "var_bilinear_pooled__best_cat_ea_ka_pa_cv1 (bilinear:pooled cv1 "
            "combo) was trained as PCA arm none") for x in miss), miss
        v = nums["items"]["pca"]["value"]
        assert v["pd:per_head"]["cv2"]["arms_missing"] == [
            "joint", "per_modality"]
        assert v["selection"]["run_token"]["joint"] == "T2"


def test_pca_arms_table_without_run_tokens():
    """Selections that record no run_token cannot show their arms to be one
    submission: listed once (the cells stand, compared alike); one arm
    without it next to arms with one is another submission's (stale)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        for a in ARMS:
            sp = _sia(root, a) / "best_per_category.json"
            sel = json.loads(sp.read_text())
            del sel["_selection_meta"]["run_token"]
            _wj(sp, sel)
        assert _run_arms(root) == INCOMPLETE
        nums = _arms_numbers(root)
        assert nums["missing"] == [
            "tab_pca: the selection of PCA arm(s) none, joint, per_modality, "
            "both records no _selection_meta.run_token, so those arms are "
            "not shown to be one submission with the others"], \
            nums["missing"]
        assert all(r["status"] == "ok"
                   for r in _rows(_arms_out(root) / "tab_pca.csv"))
        sp = _sia(root) / "best_per_category.json"
        sel = json.loads(sp.read_text())
        sel["_selection_meta"]["run_token"] = TOKEN
        _wj(sp, sel)
        assert _run_arms(root) == INCOMPLETE
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        assert {r["status"] for r in rows if r["pca_arm"] != "none"} == {
            "stale: its selection's run_token None is not the reference arm "
            "none's TOK (another submission)"}


def test_pca_arms_table_arms_selected_on_two_cvs():
    """One submission selects every arm's modalities on one CV (its
    SELECTION_CV), and that CV is in no signature: an arm whose combo job
    was resubmitted by hand without it selected on another, exited 0 and
    kept the run's token (2026-10-21). Its cells are stale (no value,
    listed under its arm, best_arm and the 2x2 effects left empty), so the
    verdict is not PAPER COMPLETE; numbers.json and the README name each
    arm's selection CV. The reference arm on another CV makes every other
    arm stale, and a selection that records no CV is not shown to be on the
    reference arm's."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)

        def set_cv(arm, cv):
            sp = _sia(root, arm) / "best_per_category.json"
            sel = json.loads(sp.read_text())
            if cv is None:
                del sel["_selection_meta"]["selection_cv"]
            else:
                sel["_selection_meta"]["selection_cv"] = cv
            _wj(sp, sel)

        def statuses():
            return {r["pca_arm"]: r["status"]
                    for r in _rows(_arms_out(root) / "tab_pca.csv")
                    if (r["variant"], r["cv"]) == ("pd:per_head", "cv1")}

        def stale(cv, ref_cv):
            return (f"stale: its modalities were selected on {cv}, the "
                    f"reference arm none's on {ref_cv} (SELECTION_CV: "
                    f"another selection protocol)")

        # per_modality and both re-selected on cv2 by hand, beside arms none
        # and joint on the run's cv1, all under the run's token
        for a in ("per_modality", "both"):
            set_cv(a, "cv2")
        assert _run_arms(root) == INCOMPLETE
        nums = _arms_numbers(root)
        assert nums["verdict"] != pt.VERDICT_COMPLETE, nums["verdict"]
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        want = stale("cv2 validation", "cv1 validation")
        for r in rows:
            if r["pca_arm"] in ("per_modality", "both"):
                assert r["status"] == want and r["auprg_mean"] == "" and \
                    r["verdict"] == "n/a (stale)", r
            else:
                assert r["status"] == "ok", r
        assert all(r["best_arm"] == "" for r in rows)
        v = nums["items"]["pca"]["value"]
        p1 = v["pd:per_head"]["cv1"]
        assert p1["arms_missing"] == ["per_modality", "both"] and \
            p1["best_arm"] is None and p1["effects"] is None, p1
        assert v["selection"]["selection_cv"] == {
            "none": "cv1", "joint": "cv1", "per_modality": "cv2",
            "both": "cv2"}, v["selection"]
        assert v["selection"]["run_token"] == dict.fromkeys(ARMS, TOKEN)
        assert ("PCA arm both: tab_pca siamese_sl/cv1: siamese_sl/results/"
                "pca_both/best_cat_ea_ka_pa_cv1 is stale (its modalities "
                "were selected on cv2 validation, the reference arm none's "
                "on cv1 validation (SELECTION_CV: another selection "
                "protocol))") in nums["missing"], nums["missing"]
        readme = (_arms_out(root) / "README.md").read_text()
        assert ("- Selection CV (each arm's _selection_meta.selection_cv): "
                "none cv1; joint cv1; per_modality cv2; both cv2 (an arm "
                "whose modalities were selected on another CV than the "
                "reference arm's is stale)") in readme, readme
        # the reference arm on another CV, and an arm that records none
        for a in ("per_modality", "both"):
            set_cv(a, "cv1")
        set_cv("none", "cv2")
        set_cv("joint", None)
        assert _run_arms(root) == INCOMPLETE
        assert statuses() == {
            "none": "ok",
            "joint": stale("a CV its selection does not record",
                           "cv2 validation"),
            "per_modality": stale("cv1 validation", "cv2 validation"),
            "both": stale("cv1 validation", "cv2 validation")}, statuses()
        # one CV everywhere again: complete
        set_cv("none", "cv1")
        set_cv("joint", "cv1")
        assert _run_arms(root) == 0 and _arms_numbers(root)["verdict"] == \
            pt.VERDICT_COMPLETE, _arms_numbers(root)["missing"]


def test_pca_arms_table_unpaired_folds():
    """The paired spread needs each graded row's per-fold records, all of
    them (summary.folds that average to its mean) and as many as the
    reference arm's: a row without them, or of another length, is listed;
    its cell keeps its means (no paired spread, the reading n/a), and the
    2x2 of its variant x tier is left empty."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        q = _cmp(root, "both") / "siamese_sl_cv1.json"
        row = json.loads(q.read_text())
        for fold in row["summary"]["folds"]:
            fold["auprg"] = .1          # not what its mean is over
        _wj(q, row)
        q = _cmp(root, "joint") / "siamese_sl_cv2.json"
        row = json.loads(q.read_text())
        x = row["summary"]["auprg_mean"]
        row["summary"]["folds"] = [{"auprg": x}, {"auprg": x - .01},
                                   {"auprg": x + .01}]
        _wj(q, row)
        assert _run_arms(root) == INCOMPLETE
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        b = _one(rows, variant="pd:per_head", cv="cv1", pca_arm="both")
        assert b["status"] == "ok" and b["n_paired"] == "" and \
            b["verdict"] == "n/a (fewer than 2 paired folds)", b
        _close(b["delta_auprg"], ARM_SHIFT["both"])
        j = _one(rows, variant="pd:per_head", cv="cv2", pca_arm="joint")
        assert j["status"] == "ok" and j["n_paired"] == "" and \
            j["verdict"] == "no measurable difference", j
        nums = _arms_numbers(root)
        assert nums["missing"] == [
            "PCA arm both: tab_pca siamese_sl/cv1: no per-fold graded auprg "
            "in sl_comparison/results/pca_both/siamese_sl_cv1.json "
            "(summary.folds, the folds its auprg_mean is over): no paired "
            "spread, no 2x2 effect",
            "PCA arm joint: tab_pca siamese_sl/cv2: its 3 per-fold graded "
            "auprg do not pair with the reference arm none's 2 (no paired "
            "spread)",
            "numbers.json pca pd:per_head/cv1: no fold that all four arms' "
            "graded rows scored (their per-fold auprg, listed above); "
            "effects left empty",
            "numbers.json pca pd:per_head/cv2: no fold that all four arms' "
            "graded rows scored (their per-fold auprg, listed above); "
            "effects left empty"], nums["missing"]
        v = nums["items"]["pca"]["value"]["pd:per_head"]
        assert v["cv1"]["best_arm"] == "per_modality" and \
            v["cv1"]["effects"] is None and v["cv3"]["effects"] is not None


def _set_folds(root: Path, arm: str, model: str, cv: str, offsets) -> None:
    """An arm's graded row with one per-fold record per offset (each metric
    at its mean + offset; the offsets sum to 0, so the records still average
    to the row's means), as evaluate_model.py records a fold set of that
    many folds."""
    q = _cmp(root, arm) / f"{model}_{cv}.json"
    row = json.loads(q.read_text())
    s = row["summary"]
    s["folds"] = [{m: s[f"{m}_mean"] + o for m in pt.METRICS}
                  for o in offsets]
    _wj(q, row)


def test_pca_arms_table_se_over_three_folds():
    """The paired spread's se is the SAMPLE sd / sqrt(n), as the ablation
    summaries' (not the population sd): the two equal each other at two
    folds, the fixture's, and part at three (on 5-fold rows the population
    sd is twice the se), and with them the reading of a difference. Pinned
    on _spread, on a tab_pca cell and on the 2x2 effects over three folds."""
    s = pt._spread([.01, .02, .06])
    _close(s["mean"], .03)
    _close(s["sd"], .0216025, 1e-6)
    _close(s["se"], .0152753, 1e-6)
    assert s["n"] == 3 and pt._spread([.01])["se"] is None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        flat = (-.01, 0, .01)
        for a in ARMS:
            _set_folds(root, a, "siamese_sl", "cv3",
                       (-.016, -.002, .018) if a == "both" else flat)
        assert _run_arms(root) == 0, _arms_numbers(root)["missing"]
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        # both - none per fold: .01 + (-.006, -.002, .008): population sd
        # .0058878, se = sample sd / sqrt(3) = .0041633. The difference .01
        # exceeds 2 se (.0083), not 2 population sd (.0118).
        b = _one(rows, variant="pd:per_head", cv="cv3", pca_arm="both")
        _close(b["delta_auprg_paired_sd"], .0058878, 1e-6)
        _close(b["delta_auprg_se"], .0041633, 1e-6)
        assert b["n_paired"] == "3" and \
            b["verdict"] == "better than the reference", b
        j = _one(rows, variant="pd:per_head", cv="cv3", pca_arm="joint")
        _close(j["delta_auprg_se"], 0)
        assert j["n_paired"] == "3" and \
            j["verdict"] == "no measurable difference", j
        v = _arms_numbers(root)["items"]["pca"]["value"]["pd:per_head"]["cv3"]
        _close(v["arms"]["both"]["paired"]["se"], .0041633, 1e-6)
        # The 2x2 over the three folds (only both's offsets differ from
        # flat): per_modality and joint move by half of both's, the
        # interaction by all of it.
        for name, mean, sd, se in (
                ("per_modality", .0165, .0029439, .0020817),
                ("joint", -.0065, .0029439, .0020817),
                ("interaction", -.007, .0058878, .0041633)):
            e = v["effects"][name]
            _close(e["mean"], mean)
            _close(e["sd"], sd, 1e-6)
            _close(e["se"], se, 1e-6)
            assert e["n"] == 3, name


def test_pca_arms_table_views_are_isolated():
    """Every arm judges its own rows with its own caches (row_usable and
    fold_ok are keyed by (cv, model), the same in every arm): arm none's
    partial siamese_sl/cv1 row leaves the other arms' cv1 cells ok, arm
    joint's partial cv2 row leaves the reference's and the others' cv2 cells
    ok, and per_modality's comparison.csv graded on other folds at cv3 only
    per_modality's cell."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        for arm, cv in (("none", "cv1"), ("joint", "cv2")):
            q = _cmp(root, arm) / f"siamese_sl_{cv}.json"
            row = json.loads(q.read_text())
            row["summary"].update(partial=True, n_folds_scored=1,
                                  n_folds_expected=2)
            _wj(q, row)
        p = _cmp(root, "per_modality") / "comparison.csv"
        rows = _rows(p)
        for r in rows:
            if (r["cv"], r["model"], r["metric"]) == (
                    "cv3", "siamese_sl", "folds_meta.json_sha256"):
                r["mean"] = "0" * 16
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        assert _run_arms(root) == INCOMPLETE
        cells = {(r["pca_arm"], r["cv"]): r["status"]
                 for r in _rows(_arms_out(root) / "tab_pca.csv")
                 if r["variant"] == "pd:per_head"}
        bad = {("none", "cv1"), ("joint", "cv2"), ("per_modality", "cv3")}
        for key, st in cells.items():
            assert (st == "missing: no usable graded row (listed)") is \
                (key in bad), (key, st)
        miss = _arms_numbers(root)["missing"]
        for arm, cv in (("none", "cv1"), ("joint", "cv2")):
            assert f"PCA arm {arm}: siamese_sl/{cv} is not tabulated: " \
                f"partial fold set (1 of 2 folds); a partial mean is not " \
                f"comparable to a complete one" in miss, (arm, miss)
        assert any(m.startswith("PCA arm per_modality: siamese_sl/cv3 "
                                "(comparison.csv) is not on the current "
                                "folds") for m in miss), miss
        assert not any(m.startswith(("PCA arm joint: siamese_sl/cv1 ",
                                     "PCA arm none: siamese_sl/cv2 ",
                                     "PCA arm none: siamese_sl/cv3 "))
                       for m in miss), miss


def test_pca_arms_table_arm_without_a_selection():
    """An arm without its best_per_category.json (a partial copy) has no
    cell ("missing: no best_per_category.json"), is listed once by its own
    loading and is no evidence for the token or selection checks: not
    "another submission", not "records no run_token", not "selected
    differently". The reference arm without one leaves the other arms' cells
    standing (no token to compare with), their differences and best arm
    unknown."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        sj = _sia(root, "joint") / "best_per_category.json"
        text = sj.read_text()
        sj.unlink()
        assert _run_arms(root) == INCOMPLETE
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        assert {r["status"] for r in rows if r["pca_arm"] == "joint"} == {
            "missing: no best_per_category.json (listed)"}
        assert all(r["status"] == "ok" for r in rows
                   if r["pca_arm"] != "joint")
        nums = _arms_numbers(root)
        miss = nums["missing"]
        assert miss[0] == ("PCA arm joint: siamese_sl/results/pca_joint/"
                           "best_per_category.json"), miss
        assert not any("run_token" in m or "selected differently" in m
                       for m in miss), miss
        sel = nums["items"]["pca"]["value"]["selection"]
        assert sel["winners"]["joint"] is None and \
            sel["same_selection"] == [["none", "per_modality", "both"]]
        _w(sj, text)
        (_sia(root) / "best_per_category.json").unlink()
        assert _run_arms(root) == INCOMPLETE
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        assert all(r["status"] == "ok" for r in rows if r["pca_arm"] != "none")
        assert all(r["verdict"] == "n/a (the reference arm's cell is not ok)"
                   and r["best_arm"] == "" for r in rows
                   if r["pca_arm"] != "none")
        miss = _arms_numbers(root)["missing"]
        assert miss[0] == "PCA arm none: siamese_sl/results/" \
            "best_per_category.json", miss
        assert not any("run_token" in m or "selected differently" in m
                       for m in miss), miss


def test_pca_arms_table_modalities_and_selections():
    """same_modalities_as_reference is False for an arm whose combo was
    trained on other files than the reference arm's: per_modality and both
    selected eb (their pair ranks other runs than none and joint: not a
    problem, nothing listed). Two arms that rank ONE single-embedding root
    but selected differently (joint against none) are listed: Step 2 is
    deterministic on the same runs."""
    def reselect(arm):
        sp = _sia(root, arm) / "best_per_category.json"
        sel = json.loads(sp.read_text())
        for cv in pt.CVS:
            sel[cv]["expression"] = {"type": "eb", "file": "all_genes_eb.pt",
                                     "aupr": .9}
        _wj(sp, sel)
        for p in _sia(root, arm).glob("*best_cat_*/config.json"):
            cfg = json.loads(p.read_text())
            cfg["embeddings_paths"] = [
                x.replace("all_genes_ea", "all_genes_eb")
                for x in cfg["embeddings_paths"]]
            _wj(p, cfg)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        reselect("per_modality")
        reselect("both")
        assert _run_arms(root) == 0, _arms_numbers(root)["missing"]
        rows = _rows(_arms_out(root) / "tab_pca.csv")
        for r in rows:
            other = r["pca_arm"] in ("per_modality", "both")
            assert (r["same_modalities_as_reference"], r["modalities"]) == (
                ("False", "eb+pa+ka") if other else ("True", "ea+pa+ka")), r
        sel = _arms_numbers(root)["items"]["pca"]["value"]["selection"]
        assert sel["same_selection"] == [["none", "joint"],
                                         ["per_modality", "both"]]
        reselect("joint")
        assert _run_arms(root) == INCOMPLETE
        nums = _arms_numbers(root)
        assert nums["missing"] == [
            "PCA arms none, joint rank the same single-embedding runs "
            "(siamese_sl/results/) but selected differently (none: "
            '{"expression": "ea", "kg": "ka", "ppi": "pa"}; joint: '
            '{"expression": "eb", "kg": "ka", "ppi": "pa"}): one of their '
            "selections is not of the runs on disk now"], nums["missing"]
        assert nums["items"]["pca"]["value"]["selection"][
            "same_selection"] == [["none"], ["joint", "per_modality", "both"]]


def test_pca_arms_table_ignores_an_arms_tables_of_another_run():
    """(2026-10-24) An arm whose own paper-tables job wrote nothing this time
    (cancelled, refused, failed before clearing its outputs) leaves the
    previous run's README: its PAPER COMPLETE is not certified. A README
    whose numbers.json carries another run token than this job's (or the
    arm's selection's), names another arm, or is missing gives no verdict
    and one Missing entry; the arm's own token (its selection's) is
    accepted."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        assert _run_arms(root) == 0
        _arm_readme(root, "joint", token="OLD-RUN")
        _arm_readme(root, "both")
        nums_both = _out(root, "both") / "numbers.json"
        nums_both.write_text(json.dumps({"run": {"pca_arm": "joint",
                                                 "run_token": TOKEN}}))
        (_out(root, "per_modality") / "numbers.json").unlink()
        assert _run_arms(root) == INCOMPLETE
        nums = _arms_numbers(root)
        assert [r["verdict"] for r in nums["arms"]] == [
            pt.VERDICT_COMPLETE, None, None, None], nums["arms"]
        miss = nums["missing"]
        assert any("pca_joint/paper_tables/README.md is not this run's: its "
                   "numbers.json carries run token 'OLD-RUN'" in m
                   for m in miss), miss
        assert any("pca_both/paper_tables/README.md is not this run's: its "
                   "numbers.json is PCA arm 'joint''s" in m for m in miss), miss
        assert any("pca_per_modality/paper_tables/README.md is not this "
                   "run's: its numbers.json is missing" in m for m in miss), miss


def test_pca_arms_table_reads_every_arm_s_own_verdict():
    """Each arm's own paper tables are read from <its sl_comparison
    root>/paper_tables/README.md: a verdict that is not PAPER COMPLETE is one
    Missing entry naming it, a README that is absent or not that arm's (line
    1) gives none and is listed. PAPER COMPLETE only with every tab_pca cell
    ok and every arm's own complete (exit 0), else PAPER INCOMPLETE (3, the
    last line printed and README line 3); no ok cell at all exits 1."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        assert _run_arms(root) == 0
        bad = "PAPER INCOMPLETE: 2 item(s) (x; y)"
        _arm_readme(root, "both", bad)
        (_out(root, "joint") / "README.md").unlink()
        _w(_out(root, "per_modality") / "README.md",
           "# Paper tables and numbers\n\nPAPER COMPLETE\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _run_arms(root)
        lines = buf.getvalue().splitlines()
        nums = _arms_numbers(root)
        assert rc == INCOMPLETE
        assert nums["missing"] == [
            "PCA arm joint: sl_comparison/results/pca_joint/paper_tables/"
            "README.md not found: its paper tables did not run (or failed "
            "before writing it), so the arm's verdict is unknown",
            "PCA arm per_modality: sl_comparison/results/pca_per_modality/"
            "paper_tables/README.md is not PCA arm per_modality's paper "
            "tables (line 1 '# Paper tables and numbers', not '# Paper "
            "tables and numbers: PCA arm per_modality': written before the "
            "PCA arms, or another arm's): its verdict is not read",
            f"PCA arm both: its paper tables (sl_comparison/results/pca_both/"
            f"paper_tables) say: {bad}"], nums["missing"]
        assert all(r["status"] == "ok"
                   for r in _rows(_arms_out(root) / "tab_pca.csv"))
        assert [r["verdict"] for r in nums["arms"]] == [
            pt.VERDICT_COMPLETE, None, None, bad]
        assert nums["verdict"].startswith("PAPER INCOMPLETE: 3 item(s) (PCA "
                                          "arm joint: ")
        assert lines[-1] == nums["verdict"] == (
            _arms_out(root) / "README.md").read_text().splitlines()[2]
        # No ok cell at all (no arm has a combo): 1, and no tab_pca.csv.
        for a in ARMS:
            for p in _sia(root, a).glob("*latest_combo_*.txt"):
                p.unlink()
        assert _run_arms(root) == 1
        nums = _arms_numbers(root)
        # Its cells were built (each missing, with its reason) but not
        # written: a table without a usable source is absent, as in per-arm
        # mode.
        assert nums["tables"]["tab_pca"]["status"] == "missing"
        assert "tab_pca.csv (no usable source)" in nums["missing"]
        assert "PCA arm both: siamese_sl/results/pca_both/" \
            "latest_combo_cv1.txt (pd:per_head cv1 combo)" in nums["missing"]
        assert not (_arms_out(root) / "tab_pca.csv").exists()


def test_pca_arms_table_after_every_arm_s_tables():
    """The DAG's order: each arm's own paper tables, then the cross-arm
    table, which quotes each arm's verdict (the fixture's are incomplete,
    so the cross-arm verdict is too, with one entry per arm), reads its
    tab_pca cells from the same namespaces and changes no arm's output."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_arms(root)
        for a in ARMS:
            assert _run(root, arm=a) == INCOMPLETE, a
        before = {a: {p.name: p.read_bytes()
                      for p in sorted(_out(root, a).iterdir())} for a in ARMS}
        assert _run_arms(root) == INCOMPLETE
        nums = _arms_numbers(root)
        for a, rec in zip(ARMS, nums["arms"]):
            assert rec["verdict"] == _numbers(root, a)["verdict"], a
            assert rec["verdict"].startswith("PAPER INCOMPLETE: ")
        assert [m.split(":", 1)[0] for m in nums["missing"]] == [
            f"PCA arm {a}" for a in ARMS], nums["missing"]
        assert all(r["status"] == "ok"
                   for r in _rows(_arms_out(root) / "tab_pca.csv"))
        assert {a: {p.name: p.read_bytes()
                    for p in sorted(_out(root, a).iterdir())}
                for a in ARMS} == before


def test_pca_arms_table_needs_two_arms():
    """--pca_arms_table compares arms side by side: with one arm (from
    --pca_arms, from $PCA_ARMS, or --pca_arm alone by default) it refuses
    (exit 1) before reading or writing anything; nor does it write into an
    arm's own paper_tables/ (a per-arm run, not into its directory)."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        base = ["--root", str(root), "--variants", *VARIANTS,
                "--code_manifest", "", "--pca_arms_table"]
        for extra, env in ((["--pca_arms", "joint"], {}),
                           ([], {"PCA_ARMS": "both"}), ([], {})):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), _PcaEnv(**env):
                assert pt.main(base + extra) == 1, (extra, env)
            assert err.getvalue().startswith(
                "ERROR: --pca_arms_table compares PCA arms side by side, but "
                "--pca_arms / $PCA_ARMS lists only "), err.getvalue()
            assert "nothing written" in err.getvalue()
            assert not _arms_out(root).exists(), (extra, env)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), _PcaEnv():
            assert pt.main(base + ["--pca_arms", "none", "joint", "--out_dir",
                                   str(_out(root))]) == 1
        assert "it is PCA arm none's paper tables directory" in \
            err.getvalue(), err.getvalue()
        assert not _out(root).exists()
        # ...nor a per-arm run into it, or into another arm's.
        for out, arm, why in ((_arms_out(root), "none",
                               "it is --pca_arms_table's directory"),
                              (_out(root), "joint",
                               "it is PCA arm none's paper tables directory")):
            try:
                _run(root, "--out_dir", str(out), arm=arm)
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(why)
            assert not out.exists(), out


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


def test_sldb_agreement_table_with_and_without_cis():
    """tab_sldb_agreement: external_transfer's sldb_agreement.csv per screen x
    scope with screen_role; the bootstrap CIs of
    <cmp>/bootstrap/ci_sldb_agreement.csv merged when their point is the
    table's value (absent: a note, not missing; another value: listed)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        _run(root)
        rows = _rows(_out(root) / "tab_sldb_agreement.csv")
        assert {r["screen"] for r in rows} == set(pt.SCREENS)
        g = _one(rows, screen="Gilbert", scope="all_pairs")
        assert g["screen_role"] == "in_distribution_check"
        assert _one(rows, screen="Corn", scope="all_pairs")["screen_role"] \
            == "external"
        _close(g["auroc"], AGREE_AUROC)
        assert g["auroc_ci_low"] == "" and g["ci_source"] == ""
        assert any("tab_sldb_agreement: no " in n
                   for n in _numbers(root)["notes"])
        assert not [m for m in _numbers(root)["missing"]
                    if "sldb_agreement" in m]
        # CIs from bootstrap_ci.py --part external
        bdir = _cmp(root) / "bootstrap"
        bdir.mkdir(parents=True, exist_ok=True)
        ci = [["screen", "scope", "split", "metric", "point", "ci_low",
               "ci_high", "pca_arm"]]
        for s in pt.SCREENS:
            for scope, sp in [("all_pairs", "any")] + [
                    ("test_genes_txt", x) for x in pt.SPLITS]:
                ci.append([s, scope, sp, "auroc_target_vs_label",
                           AGREE_AUROC, .6, .8, "none"])
                ci.append([s, scope, sp, "spearman_target_vs_label",
                           .3, .2, .4, "none"])
        with open(bdir / "ci_sldb_agreement.csv", "w", newline="") as f:
            csv.writer(f).writerows(ci)
        _run(root)
        g = _one(_rows(_out(root) / "tab_sldb_agreement.csv"),
                 screen="Gilbert", scope="all_pairs")
        _close(g["auroc_ci_low"], .6)
        _close(g["spearman_ci_high"], .4)
        assert g["ci_source"].endswith("bootstrap/ci_sldb_agreement.csv")
        # a CI of another value (another run's bootstrap) is not merged
        ci[1][4] = .55
        with open(bdir / "ci_sldb_agreement.csv", "w", newline="") as f:
            csv.writer(f).writerows(ci)
        _run(root)
        rows = _rows(_out(root) / "tab_sldb_agreement.csv")
        bad = [r for r in rows if r["screen"] == ci[1][0] and
               r["scope"] == "all_pairs"]
        assert bad[0]["auroc_ci_low"] == ""
        assert any("tab_sldb_agreement: 1 CI(s)" in m
                   for m in _numbers(root)["missing"])
        # no sldb_agreement.csv: the table is missing
        (_cmp(root) / "external_transfer" / "sldb_agreement.csv").unlink()
        _run(root)
        assert any(m.endswith("sldb_agreement.csv (external_transfer.py)")
                   for m in _numbers(root)["missing"]), \
            _numbers(root)["missing"]


def _allmod_rows(root: Path, record=True) -> None:
    """siamese_sl_allmod (pd:per_head) rows as run_compare.sh leaves them with
    ALLMOD_COMBO=1: comparison.csv lines (with collect_and_regrade's per-fold
    parameter row) and the graded JSON beside it, carrying the
    all_modalities record collect_siamese.py --allmod writes."""
    folds = root / "sl_comparison" / "folds"
    model = "siamese_sl_allmod"
    with open(_cmp(root) / "comparison.csv", newline="") as f:
        rows = list(csv.reader(f))
    for cv, base in CMP["siamese_sl"].items():
        vals = tuple(x - .01 for x in base)
        for m, x in zip(pt.METRICS, vals):
            rows.append([cv, model, m, x, .02])
        rows.append([cv, model, "baseline", .172 if cv == "cv1" else .2, 0])
        rows.append([cv, model, "trainable_params", 61000, ""])
        rows.append([cv, model, "trainable_params_fold", 61000.0, 0.0])
        st = mp.fold_stamp(folds, cv)
        for n in mp.STAMP_FILES:
            rows.append([cv, model, f"folds_{n}_sha256", st[n], ""])
            rows.append([cv, model, f"trained_on_{n}_sha256", st[n], ""])
        rows.append([cv, model, "seed", SEED, ""])
        p = _cmp(root) / f"{model}_{cv}.json"
        _row_json(p, model, cv, vals, folds)
        if record:
            d = json.loads(p.read_text())
            d["summary"]["selection_audit"] = {
                "pca_arm": "none", "all_modalities": {
                    "selection": "none", "modalities": ["ea", "eb", "pa"],
                    "n_modalities": 3}}
            d["summary"]["trainable_params_fold_mean"] = 61000.0
            d["summary"]["trainable_params_fold_std"] = 0.0
            _wj(p, d)
        # The run and marker run_allmod_combo.sh writes: the combo's
        # config.json with --all_modalities (paper_tables checks the two
        # agree outside the modality set).
        sia = root / "siamese_sl"
        combo = (sia / "results" / f"latest_combo_{cv}.txt")
        if combo.exists():
            cfg = json.loads((sia / combo.read_text().strip() /
                              "config.json").read_text())
            cfg["all_modalities"] = True
            name = f"results/allmod__all_modalities_{cv}"
            _wj(sia / name / "config.json", cfg)
            _w(sia / "results" / f"allmod__latest_allmod_{cv}.txt",
               name + "\n")
    with open(_cmp(root) / "comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_allmod_rows_are_an_opt_in_ablation():
    """ALLMOD_COMBO (siamese_sl/slurm/run_allmod_combo.sh): off, a complete
    run stays complete and lists no all-modalities row; on, each variant's
    siamese_sl_allmod[_<tag>] row is tab_tiers kind 'ablation' right after
    the selected SLeuth rows (bold-eligible, never beats_peers_*), missing
    when absent, and refused when its row JSON is not the all-modalities
    combo's."""
    if torch is None:
        return
    import os
    saved = os.environ.pop("ALLMOD_COMBO", None)
    try:
        with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_tree(root)
            rc, _ = _run_complete(root)
            assert rc == 0, _missing(_out(root))
            rows = _rows(_out(root) / "tab_tiers.csv")
            assert not [r for r in rows if "allmod" in r["model"]]
            assert _numbers(root)["run"]["allmod_combo"] is False
            # on, no rows yet: listed, the verdict incomplete
            rc, _ = _run_complete(root, "--allmod_combo", "1")
            miss = _missing(_out(root))
            assert rc == INCOMPLETE and any(
                m.startswith("tab_tiers siamese_sl_allmod/cv1 (no usable")
                for m in miss), miss
            # on, with the rows: complete, kind ablation after SLeuth
            _allmod_rows(root)
            rc, _ = _run_complete(root, "--allmod_combo", "1")
            assert rc == 0, _missing(_out(root))
            rows = _rows(_out(root) / "tab_tiers.csv")
            cv1 = [r["model"] for r in rows if r["cv"] == "cv1"]
            assert cv1.index("siamese_sl") + 1 == \
                cv1.index("siamese_sl_allmod"), cv1
            a = _one(rows, cv="cv1", model="siamese_sl_allmod")
            assert a["kind"] == "ablation" and a["variant"] == "pd:per_head" \
                and a["display"] == "SLeuth, all modalities (pd:per_head)"
            _close(a["auprg_mean"], CMP["siamese_sl"]["cv1"][2] - .01)
            assert a["beats_peers_auprg"] == "" and \
                a["trainable_params"] == "" and \
                float(a["params_fold_mean"]) == 61000.0, a
            assert _numbers(root)["run"]["allmod_combo"] is True
            # ALLMOD_COMBO from the environment, as run_paper_tables.sh gets it
            os.environ["ALLMOD_COMBO"] = "1"
            rc, _ = _run_complete(root)
            os.environ.pop("ALLMOD_COMBO")
            assert rc == 0 and _one(_rows(_out(root) / "tab_tiers.csv"),
                                    cv="cv2", model="siamese_sl_allmod")
            # a row JSON without the record is not the allmod combo's
            for cv in ("cv1", "cv2", "cv3"):
                p = _cmp(root) / f"siamese_sl_allmod_{cv}.json"
                d = json.loads(p.read_text())
                d["summary"].pop("selection_audit")
                _wj(p, d)
            rc, _ = _run_complete(root, "--allmod_combo", "1")
            miss = _missing(_out(root))
            assert rc == INCOMPLETE and any(
                "siamese_sl_allmod/cv1: its row JSON records no "
                "selection_audit.all_modalities" in m for m in miss), miss
            a = _one(_rows(_out(root) / "tab_tiers.csv"), cv="cv1",
                     model="siamese_sl_allmod")
            assert a["auprg_mean"] == "" and a["status"] == "missing", a
    finally:
        if saved is not None:
            os.environ["ALLMOD_COMBO"] = saved


def test_probe_row_built_on_another_combo_is_refused():
    """A BilinearProbe row is tabulated only when its probe_source names the
    config.json of the arm-none pd:pooled combo the marker names now: a combo
    retrained on the same folds passes every fold check, and nothing else
    moves the shared probe row aside (2026-11-13)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        m = (root / "siamese_sl" / "results" /
             "var_pd_pooled__latest_combo_cv2.txt")
        cfg = root / "siamese_sl" / m.read_text().strip() / "config.json"
        o = json.loads(cfg.read_text())
        o["seed"] = 7          # a retrained combo: other config.json bytes
        cfg.write_text(json.dumps(o))
        assert _run(root) == INCOMPLETE
        tiers = _rows(_out(root) / "tab_tiers.csv")
        assert _one(tiers, cv="cv2", model="BilinearProbe")["status"] == \
            "missing"
        assert _one(tiers, cv="cv1", model="BilinearProbe")["status"] == "ok"
        miss = _numbers(root)["missing"]
        assert any("BilinearProbe/cv2: built on another pd:pooled combo" in x
                   for x in miss), miss


def test_allmod_row_with_other_settings_is_refused():
    """An all-modalities run whose arch_signature differs from its selected
    combo's outside the modality set (a hand rerun with other ARCH_ARGS, PCA
    or selection split) is not tabulated (2026-11-13)."""
    if torch is None:
        return
    import os
    saved = os.environ.pop("ALLMOD_COMBO", None)
    try:
        with _FixturePaper16(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_tree(root)
            _allmod_rows(root)
            sia = root / "siamese_sl"
            d = sia / "results" / "allmod__all_modalities_cv2" / "config.json"
            cfg = json.loads(d.read_text())
            cfg["selection_frac"], cfg["selection_seed"] = 0.2, 42
            d.write_text(json.dumps(cfg))
            rc, _ = _run_complete(root, "--allmod_combo", "1")
            assert rc == INCOMPLETE
            rows = _rows(_out(root) / "tab_tiers.csv")
            assert _one(rows, cv="cv2", model="siamese_sl_allmod"
                        )["status"] == "missing"
            assert _one(rows, cv="cv1", model="siamese_sl_allmod"
                        )["status"] == "ok"
            assert any("siamese_sl_allmod/cv2: its arch_signature differs"
                       in m for m in _missing(_out(root)))
    finally:
        if saved is not None:
            os.environ["ALLMOD_COMBO"] = saved


def test_probe_row_on_a_retrained_combo_is_refused():
    """Same settings, new training (config.json bytes equal, results.json
    not): the probe row is refused too (2026-11-13)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build(root)
        m = (root / "siamese_sl" / "results" /
             "var_pd_pooled__latest_combo_cv3.txt")
        res = root / "siamese_sl" / m.read_text().strip() / "results.json"
        res.write_text(res.read_text() + "\n")
        assert _run(root) == INCOMPLETE
        tiers = _rows(_out(root) / "tab_tiers.csv")
        assert _one(tiers, cv="cv3", model="BilinearProbe")["status"] == \
            "missing"
        assert any("BilinearProbe/cv3: built on another training" in x
                   for x in _numbers(root)["missing"])
