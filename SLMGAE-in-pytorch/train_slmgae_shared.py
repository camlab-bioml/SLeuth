#!/usr/bin/env python3
"""
Run SLMGAE on the SLDB 3.0 standard: the shared canonical CV folds used by the
four-model comparison (see ../sl_comparison/prepare_folds.py).

Differences from the standalone train_slmgae.py:
  - Folds come from sl_comparison/folds (given positives + all given negatives,
    one seed, identical to the other models) instead of SLMGAE's own random-
    negative SLDataSplitter.
  - The gene universe is the shared 8,844-gene index (folds/genes.txt), not
    List_Proteins_in_SL.txt (6,375 — see the loader note below; a 6,372-entry
    copy is a stale artefact). SLMGAE's GO / GO-CC / PPI similarity
    support views are therefore REMAPPED from the List_Proteins index into the
    shared index (genes absent from a view get zero similarity rows).
  - UPSTREAM COVERAGE, KEPT ON PURPOSE: the authors' view files exist only for
    List_Proteins_in_SL.txt, which is exactly the gene set of SynLethDB v1 SL
    pairs (SL_Human_Approved.txt). On the wider shared universe, whether a
    gene has a view row therefore correlates with the labels, and a gene with
    no view row and no training positive gets an all-zero embedding (score
    exactly 0). SLMGAE is run on its authors' inputs unchanged; the startup
    diagnostic only MEASURES this (share of val/test positive vs negative
    pairs whose genes both have a view row, per fold) and records it in the
    log and training_summary.json["view_coverage"].
  - A support view whose source file is absent ABORTS the run (see
    build_support_views): SLMGAE runs on its authors' shipped views, and a
    reduced model must not sit in the four-model table under SLMGAE's name.
    --allow_missing_views opts into an ALL-ZERO view instead; the reduced run
    is then printed at startup and recorded in training_summary.json and
    model_params.json ["support_views"]. A view file whose size does not match
    List_Proteins_in_SL.txt also aborts (the loaders would truncate silently).
    If HGNC cannot be loaded, List_Proteins would map through the static
    table (fewer rows, a reduced model): that ABORTS too, unless
    --allow_static_lp_mapping; the reduced run is then recorded as
    ["lp_mapping"]["id_space"] == LP_STATIC_FALLBACK_SPACE in the same files.
  - Each fold's full gene x gene score matrix is saved as
    fold_<k>_<cv>_predictions.npy so sl_comparison/evaluate_model.py scores it
    with the same cal_metrics evaluator as every other model.
  - Early stopping AND checkpoint selection use AUPRG (sl_comparison/auprg.py,
    the shared evaluator's definition) on the fold's VALIDATION split
    (val_pos = 1, val_neg = 0, carved from the fold's training pairs), scored
    with the R[i, j] entries the saved matrix holds after the grader's float32
    shift (score - min + 1, min over the matrix incl. its zero diagonal), so
    selection ranks val pairs as evaluate_model.py does. --select_metric
    auroc restores the old rule. A fold without a two-class val split raises;
    test is never used for selection and is scored once, after the restore.

The SLMGAE model, loss, and per-fold training loop are inherited unchanged
(the loss is taken over the fold's train_pos + train_neg; see objective.py).

Usage (from SLMGAE-in-pytorch/; manual runs default to
results/slmgae_shared_manual_<cv>, never the graded results/slmgae_shared_<cv>):
  python train_slmgae_shared.py --cv_type cv3 --folds_dir ../sl_comparison/folds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from slmgae_pytorch import DataLoader, set_random_seed
from train_slmgae import SELECT_METRICS, Trainer, build_selection_split

# Canonical shared-fold loader lives in ../sl_comparison.
sys.path.insert(0,
                str(Path(__file__).resolve().parent.parent / "sl_comparison"))
from shared_folds import load_shared_folds  # noqa: E402
from model_params import purge_stale_predictions  # noqa: E402


def remap_support(view, lp_genes, shared_g2i, num_nodes):
    """Remap a (L,L) similarity matrix from List_Proteins index to shared index.

    Genes in List_Proteins but not in the shared universe are dropped; shared
    genes with no List_Proteins counterpart get all-zero rows/cols.
    """
    coo = sp.coo_matrix(view)
    lp2s = np.array([shared_g2i.get(g, -1) for g in lp_genes], dtype=np.int64)
    r = lp2s[coo.row]
    c = lp2s[coo.col]
    keep = (r >= 0) & (c >= 0)
    return sp.coo_matrix((coo.data[keep], (r[keep], c[keep])),
                         shape=(num_nodes, num_nodes))


def find_row_collisions(lp_genes, shared_g2i):
    """{entrez: [List_Proteins row, ...]} for shared genes hit by >1 row.

    remap_support has no uniqueness check: two rows landing on one shared gene
    get their view rows SUMMED (coo duplicates), and whichever gene they were
    meant to be is left with an empty row.
    """
    rows = {}
    for i, g in enumerate(lp_genes):
        if g in shared_g2i:
            rows.setdefault(g, []).append(i)
    return {g: r for g, r in rows.items() if len(r) > 1}


def view_coverage(support_adjs, labels, splits):
    """MEASURE ONLY: how view-row coverage splits val/test positives vs negatives.

    A gene "has a view row" when it has >= 1 off-diagonal entry in at least one
    support view as the model receives it (after remapping). For each fold and
    each of val_pos / val_neg / test_pos / test_neg, report the share of pairs
    whose TWO genes both have one. Nothing is altered or filtered; the numbers
    disclose the upstream coverage described in the module docstring.
    """
    n = support_adjs[0].shape[0]
    has_row = np.zeros(n, dtype=bool)
    per_view = {}
    for adj, label in zip(support_adjs, labels):
        a = sp.coo_matrix(adj)
        off = (a.row != a.col) & (a.data != 0)
        m = np.zeros(n, dtype=bool)
        m[a.row[off]] = True
        m[a.col[off]] = True
        per_view[label] = int(m.sum())
        has_row |= m

    def both(pairs):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            return {"n": 0, "both_have_view": None}
        hit = has_row[pairs[:, 0]] & has_row[pairs[:, 1]]
        return {"n": int(len(pairs)), "both_have_view": float(hit.mean())}

    folds = []
    for s in splits:
        te, tl = np.asarray(s["test_edges"]), np.asarray(s["test_labels"])
        folds.append({
            "fold": int(s["fold"]),
            "val_pos": both(s["val_pos"]),
            "val_neg": both(s["val_neg"]),
            "test_pos": both(te[tl == 1]),
            "test_neg": both(te[tl == 0]),
        })
    return {
        "note": ("Upstream data coverage kept on purpose (authors' view "
                 "files cover only List_Proteins_in_SL.txt = genes of "
                 "SynLethDB v1 SL pairs). A positive-vs-negative gap here is "
                 "label information reaching SLMGAE through its own inputs. "
                 "Measured and recorded only; nothing was altered."),
        "definition": ("gene has >= 1 off-diagonal entry in at least one "
                       "support view after remapping; value = share of pairs "
                       "whose two genes both do"),
        "num_genes": int(n),
        "genes_with_view_row": int(has_row.sum()),
        "genes_with_row_per_view": per_view,
        "folds": folds,
    }


# DO NOT "FIX" load_dense_feature's triangular parse. The similarity files are
# relative-triangular (row i has n-i fields), but the loader indexes them as if
# absolute: `for j in range(i, len(parts)): m[i][j+1] = float(parts[j])`. The
# effect is that row i reads only parts[i:] and rows i >= n/2 read NOTHING, so
# roughly half the file never lands. That looks like a bug, and arguably is one
# — but it is verbatim the published implementation
# (original_tensorflow_code/utils.py:156-160, which also hardcodes the 6375x6375
# shape). Correcting it would silently benchmark a DIFFERENT model than the
# SLMGAE paper's. Fidelity wins; the asymmetry is symmetrised away afterwards.
#
# That hardcoded 6375 is also the authority on the gene count: the companion
# List_Proteins_in_SL.txt must have 6,375 entries. A 6,372-entry copy is stale
# and would misalign every similarity row (load_dense_feature just truncates),
# so check_view_shape refuses a view file that does not match the list.
#
# SLMGAE's three support views, in the order the model consumes them.
# (filename, loader kind, display name). The two GO similarity matrices ship
# with the original SLMGAE release and are NOT produced by anything in this
# repo (`grep -rn GOsim` finds only readers) and are git-ignored (*.txt), so
# a checkout can lack them: copy them into data/. biogrid_ppi_sparse.txt is
# tracked.
SUPPORT_VIEWS = [
    ("Human_GOsim.txt", "dense", "GO BP similarity"),
    ("Human_GOsim_CC.txt", "dense", "GO CC similarity"),
    ("biogrid_ppi_sparse.txt", "sparse", "BioGRID PPI"),
]

# biogrid_ppi_sparse.txt as shipped (keyed by the 6,375-row List_Proteins)
# holds 98,581 "i<TAB>j" lines; `wc -l` prints 98,580 because the file has no
# final newline. Its ids are in no order (any prefix longer than ~1,600 lines
# already reaches id 6374), so the line count is the only thing that tells a
# cut-short copy from the whole file. Other list sizes are not pinned.
PPI_EDGES_SHIPPED = {6375: 98581}

# List_Proteins symbols that HGNC lists as a previous/alias symbol of TWO
# genes, pinned to the gene they named when the list was built (pre-2020 HGNC
# approved symbols). Where both genes list it as a PREVIOUS symbol,
# GeneNameMapper keeps the last HGNC row: QARS -> EPRS1, which already has its
# own row here (EPRS), so two of the authors' rows would be merged and QARS1
# left empty; C18orf21 -> the pseudogene RMP24P1 (neither candidate is in the
# current shared universe). A previous symbol outranks another gene's alias,
# so TAZ -> TAFAZZIN and SARS -> SARS1 already; those two pins are redundant
# but guard against a mapper or HGNC change. Kept local to SLMGAE:
# gene_corrections_config also feeds the siamese embedding pipeline.
LP_SYMBOL_OVERRIDES = {
    "QARS": "5859",  # QARS1 (EPRS1 also has QARS as a previous symbol)
    "TAZ": "6901",  # TAFAZZIN (WWTR1 has TAZ as an alias)
    "SARS": "6301",  # SARS1 (SARS2 has SARS as an alias)
    "C18ORF21": "83608",  # RMP24 (RMP24P1 also has it as a previous symbol)
}

# id_space reported when HGNC could not be loaded. The static table maps fewer
# rows (6,156/6,375 vs 6,375 on the shipped list; 4,470 vs 4,587 of them in the
# shared universe), so the run is a REDUCED SLMGAE: SharedFoldTrainer refuses
# it unless --allow_static_lp_mapping, and the distinct name travels into
# model_params.json ["lp_mapping"].
LP_STATIC_FALLBACK_SPACE = ("symbol->entrez "
                            "(static gene_id_mapping.tsv fallback)")


def load_lp_genes(data_path):
    """Read List_Proteins_in_SL.txt and return its genes as ENTREZ ID strings.

    The support views are indexed by this file's line order, and remap_support
    looks each entry up in the shared universe — which prepare_folds.py:13 keys
    by sorted ENTREZ ID. Historically this file held Entrez IDs, but the
    upstream SLMGAE data drop ships gene SYMBOLS ("A2M", "A2ML1", ...). Feeding
    symbols to an Entrez-keyed dict makes EVERY lookup miss, which silently
    empties all three support views instead of raising — the model then trains
    on nothing and still reports plausible numbers. So detect the space and
    convert, and let the caller assert the overlap.

    Returns (genes, id_space, n_unmapped) where genes[i] corresponds to row i
    of the similarity files; unmappable symbols become "" (never matched).
    id_space is LP_STATIC_FALLBACK_SPACE when HGNC was unavailable.
    """
    raw = [
        ln.strip() for ln in open(Path(data_path) / "List_Proteins_in_SL.txt")
        if ln.strip()
    ]
    if not raw:
        raise SystemExit(
            f"SLMGAE input empty: {Path(data_path)/'List_Proteins_in_SL.txt'}")
    # Entrez IDs are all-digits; symbols are not. Sample rather than scan all.
    if sum(g.isdigit() for g in raw[:200]) >= 190:
        return raw, "entrez", 0

    # Symbols -> Entrez. Prefer siamese's mapper (HGNC current + alias + the
    # 86 hand corrections in gene_corrections_config); fall back to the tracked
    # static table, which covers ~96.5% offline.
    genes = None
    id_space = "symbol->entrez"
    sia = Path(__file__).resolve().parent.parent / "siamese_sl"
    try:
        sys.path.insert(0, str(sia))
        from gene_name_utils import GeneNameMapper  # noqa: E402
        mapper = GeneNameMapper(
            cache_dir=str(Path(data_path).resolve() / "cache"))
        # A failed HGNC download or an unreadable cache is not raised by the
        # mapper: every row would come back None and skip the static table
        # below. Raise so the fallback is used instead.
        if not mapper.stats["entrez_ids"]:
            raise RuntimeError("HGNC table not loaded (no cached "
                               "hgnc_complete_set.txt and download failed, "
                               "or the cache is unreadable)")
        # drop_unmapped=False keeps exactly ONE entry per input row (None when
        # a symbol will not map). Row order IS the support-view index, so a
        # shortened list would shift every similarity row by one.
        eids, _ = mapper.symbols_to_entrez_list(raw, drop_unmapped=False)
        if len(eids) != len(raw):
            raise RuntimeError(f"mapper returned {len(eids)} ids for "
                               f"{len(raw)} rows — would shift alignment")
        genes = [e or "" for e in eids]
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: GeneNameMapper unavailable ({type(e).__name__}: "
              f"{e}); falling back to data/gene_id_mapping.tsv. Fewer "
              f"List_Proteins rows map, so this is a REDUCED SLMGAE "
              f"(refused by the trainer unless --allow_static_lp_mapping).")
    if genes is None:
        id_space = LP_STATIC_FALLBACK_SPACE
        static = {}
        tsv = Path(data_path) / "gene_id_mapping.tsv"
        if tsv.exists():
            for line in open(tsv):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    static[p[1].strip()] = p[0].strip()
        genes = [static.get(g, "") for g in raw]
    genes = [LP_SYMBOL_OVERRIDES.get(s.upper(), g) for s, g in zip(raw, genes)]
    n_unmapped = sum(1 for g in genes if not g)
    print(f"  List_Proteins is in SYMBOL space -> mapped to Entrez "
          f"({len(raw) - n_unmapped}/{len(raw)}; {n_unmapped} unmapped; "
          f"id_space={id_space})")
    return genes, id_space, n_unmapped


def check_view_shape(path, kind, n_lp):
    """Raise SystemExit unless a view file matches List_Proteins row for row.

    The views are indexed by List_Proteins line order (remap_support maps row
    i to lp_genes[i]), and the loaders silently truncate a size mismatch
    (load_dense_feature breaks at num_nodes, load_sparse_feature drops
    out-of-range ids), so a stale list would misalign every row without an
    error. The authors' loaders hardcode 6375 and would crash instead.
      dense  (relative-triangular, row i = sim(i, i+1..n-1)): n_lp - 1 rows.
      sparse (0-based "i<TAB>j" edge list): every id < n_lp, at least one
              edge line, and for a list size in PPI_EDGES_SHIPPED exactly
              that many edge lines (a cut-short copy has fewer).
    """
    with open(path) as f:
        if kind == "dense":
            got = sum(1 for ln in f if ln.strip())
            ok, want = got == n_lp - 1, f"{n_lp - 1} rows"
        else:
            got, n_edges = -1, 0
            for ln in f:
                p = ln.strip().split("\t")
                if len(p) >= 2:
                    got = max(got, int(p[0]), int(p[1]))
                    n_edges += 1
            # got == -1: not one "i<TAB>j" line (an empty or cut-short copy, or
            # not tab-separated). load_sparse_feature would return an all-zero
            # view: the REDUCED model of --allow_missing_views, but under the
            # ordinary provenance, so no table would name it.
            ok, want = 0 <= got < n_lp, f"0 <= max gene index < {n_lp}"
            n_want = PPI_EDGES_SHIPPED.get(n_lp)
            if ok and n_want is not None and n_edges != n_want:
                raise SystemExit(
                    f"SLMGAE: {path} holds {n_edges:,} edge lines, the shipped "
                    f"file {n_want:,}: a cut-short copy (upload the tracked "
                    f"file again) or another release. Either is a different "
                    f"PPI view than the authors' under SLMGAE's name.")
    if not ok:
        raise SystemExit(
            f"SLMGAE: {path} does not match List_Proteins_in_SL.txt ({n_lp} "
            f"entries): got {got}, expected {want}. A stale List_Proteins "
            f"copy (e.g. the old 6,372-entry one) or a view file from another "
            f"release would misalign every similarity row; got -1 means the "
            f"edge list holds no 'i<TAB>j' line (an empty or cut-short copy: "
            f"upload the file again).")


def build_support_views(data_path,
                        nn_size,
                        lp_genes,
                        shared_g2i,
                        num_nodes,
                        allow_missing=False):
    """Build the support views in the SHARED gene index.

    Each view is read in the List_Proteins index space and remapped into the
    shared universe. A view whose source file is missing is FATAL: SLMGAE runs
    on its authors' shipped views, and a reduced model would otherwise reach
    the comparison table under SLMGAE's name. allow_missing=True
    (--allow_missing_views) zero-fills it instead: normalize_adj() adds
    self-loops, so a zero view degrades to the identity. Every present file
    must match List_Proteins (check_view_shape).

    Returns (support_adjs, provenance); provenance maps display name -> the
    source path used, or a "ZERO-FILLED" marker naming the missing file.
    """
    missing = [(Path(data_path) / fname, label)
               for fname, _, label in SUPPORT_VIEWS
               if not (Path(data_path) / fname).exists()]
    if missing and not allow_missing:
        raise SystemExit(
            "SLMGAE input missing: " +
            "; ".join(f"{p.resolve()} ({label})" for p, label in missing) +
            ". SLMGAE runs on its authors' shipped support views; copy the "
            "file(s) into " + str(Path(data_path).resolve()) + " (git-ignored "
            "*.txt), or pass --allow_missing_views to run a REDUCED model "
            "(recorded in training_summary.json['support_views']).")

    dl = DataLoader(data_path=data_path, nn_size=nn_size)
    # load_dense_feature/load_sparse_feature size their matrix from num_nodes.
    # Set it directly instead of calling load_data(): that would also parse the
    # SL matrix, which this trainer discards (the folds supply it), and its
    # pure-Python triu edge loop costs ~20M iterations for nothing.
    dl.num_nodes = len(lp_genes)

    support_adjs, provenance = [], {}
    for fname, kind, label in SUPPORT_VIEWS:
        path = Path(data_path) / fname
        if path.exists():
            check_view_shape(path, kind, len(lp_genes))
            view = (dl.load_dense_feature(str(path), knn=True)
                    if kind == "dense" else dl.load_sparse_feature(str(path)))
            support_adjs.append(
                remap_support(view, lp_genes, shared_g2i, num_nodes))
            provenance[label] = str(path)
        else:  # only reachable with allow_missing (checked above)
            print(f"WARNING: {label} source not found ({path}) -> using an "
                  f"all-zero view (--allow_missing_views); SLMGAE runs "
                  f"WITHOUT this modality.")
            support_adjs.append(
                sp.coo_matrix((num_nodes, num_nodes), dtype=np.float32))
            provenance[label] = f"ZERO-FILLED (missing {path})"
    return support_adjs, provenance


class SharedFoldTrainer(Trainer):
    """SLMGAE trainer that consumes the shared folds + remapped support views."""

    def __init__(self, args):
        self.args = args
        # Device + seed + output dirs (mirrors the base __init__ setup).
        import torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (GPU not available)")
        set_random_seed(args.seed)
        print(f"Seed: {args.seed}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{args.output_dir}/checkpoints").mkdir(parents=True,
                                                     exist_ok=True)

        # Shared gene universe + folds.
        self.gene_order, self.num_nodes, self.splits = load_shared_folds(
            args.folds_dir, args.cv_type)
        # The fold set, not a CLI flag, fixes k: every fold_* dir is trained.
        # The inherited "Training fold i/{num_folds}" print and the args
        # recorded in training_summary.json read it from here.
        self.args.num_folds = len(self.splits)
        shared_g2i = {g: i for i, g in enumerate(self.gene_order)}
        self.num_features = self.num_nodes

        # SLMGAE's GO / GO-CC / PPI support views, built in the shared index.
        # Fail fast and by name on the one input that is genuinely required, so
        # a missing file is line 1 of the SLURM log rather than a deep traceback.
        lp_path = Path(args.data_path) / "List_Proteins_in_SL.txt"
        if not lp_path.exists():
            raise SystemExit(
                f"SLMGAE input missing: {lp_path.resolve()} — required to map "
                f"the support views into the shared gene universe.")
        lp_genes, id_space, _unmapped = load_lp_genes(args.data_path)
        # Same policy as a missing view: the static table gives fewer genes a
        # view row (a REDUCED SLMGAE), so it must be asked for, not fallen into.
        if (id_space == LP_STATIC_FALLBACK_SPACE
                and not getattr(args, "allow_static_lp_mapping", False)):
            raise SystemExit(
                f"SLMGAE: HGNC table unavailable "
                f"({Path(args.data_path).resolve() / 'cache'}"
                f"/hgnc_complete_set.txt missing or unreadable, download "
                f"failed); the static gene_id_mapping.tsv fallback maps only "
                f"{len(lp_genes) - _unmapped}/{len(lp_genes)} List_Proteins "
                f"rows, a REDUCED SLMGAE. Restore the tracked file (git "
                f"checkout -- data/cache/hgnc_complete_set.txt), or pass "
                f"--allow_static_lp_mapping to run the reduced model.")
        matched = sum(1 for g in lp_genes if g in shared_g2i)
        # Denominator is min(len(lp_genes), num_nodes), NOT len(lp_genes).
        # `matched` counts List_Proteins genes that landed in the shared
        # universe, so it is bounded above by num_nodes; dividing by the full
        # 6,375-gene list makes the ratio unreachable whenever the universe is
        # smaller than the list. preflight.sh builds a 300-gene universe, so the
        # old form capped frac at 300/6375 = 4.7% and tripped this guard BY
        # ARITHMETIC on a perfectly healthy tree — measured 233/6375 = 3.7%,
        # SystemExit, "PREFLIGHT FAILED". SLGNN and NSF4SL passed the same tiny
        # folds, so SLMGAE alone was permanently red.
        frac = matched / max(min(len(lp_genes), self.num_nodes), 1)
        # HARD GUARD. A near-zero overlap means the support views would remap to
        # all-zero WITHOUT raising, so SLMGAE would train on empty modalities and
        # still report plausible metrics. That is worse than crashing, so crash.
        # Threshold is deliberately loose: the healthy value on the full shared
        # universe is ~72% (4,587/6,375 measured 2026-08-06 — an earlier comment
        # claimed ~96%, which was never observed), and a genuine partial overlap
        # still clears 50%.
        if frac < 0.5:
            raise SystemExit(
                f"SLMGAE support views would be EMPTY: only {matched}/"
                f"{len(lp_genes)} ({frac:.1%}) of List_Proteins_in_SL.txt maps "
                f"into the shared gene universe (id_space={id_space}).\n"
                f"  List_Proteins sample: {lp_genes[:3]}\n"
                f"  shared universe sample: {self.gene_order[:3]}\n"
                f"  Both must be Entrez ID strings (prepare_folds.py indexes "
                f"the universe by sorted Entrez ID). Fix the identifier space "
                f"rather than letting the views silently zero out.")
        # Every one of the authors' rows must keep its own gene: a collision
        # would sum two genes' view rows (see find_row_collisions).
        collisions = find_row_collisions(lp_genes, shared_g2i)
        if collisions:
            raw = [ln.strip() for ln in open(lp_path) if ln.strip()]
            detail = "; ".join(f"{g} <- {[raw[i] for i in rows]}"
                               for g, rows in sorted(collisions.items()))
            raise SystemExit(
                f"SLMGAE: {len(collisions)} shared gene(s) receive more than "
                f"one List_Proteins row ({detail}). remap_support would sum "
                f"their view rows. Pin the ambiguous symbol(s) in "
                f"LP_SYMBOL_OVERRIDES.")
        self.support_adjs, self.support_provenance = build_support_views(
            args.data_path,
            args.nn_size,
            lp_genes,
            shared_g2i,
            self.num_nodes,
            allow_missing=getattr(args, "allow_missing_views", False))
        # Stamped into model_params.json next to the score matrices, which
        # evaluate_model.py copies into results/SLMGAE_<cv>.json, so a
        # ZERO-FILLED (reduced) run stays visible in the graded row itself.
        # lp_mapping does the same for a static-table fallback (fewer rows map,
        # so fewer genes get view rows; only with --allow_static_lp_mapping).
        self.lp_mapping = {
            "id_space": id_space,
            "rows": len(lp_genes),
            "mapped": len(lp_genes) - _unmapped,
            "in_universe": matched,
        }
        self.model_params_extra = {
            "support_views": self.support_provenance,
            "lp_mapping": self.lp_mapping,
        }
        print(f"Shared universe: {self.num_nodes} genes | folds: "
              f"{len(self.splits)} | support views remapped "
              f"({matched}/{len(lp_genes)} List_Proteins genes in universe, "
              f"{frac:.1%}, id_space={id_space})")
        for label, src in self.support_provenance.items():
            print(f"  support view: {label:<22s} {src}")

        # MEASURE-ONLY diagnostic of the upstream view coverage (module
        # docstring). Recorded, never acted on: SLMGAE keeps its own inputs.
        self.view_coverage = view_coverage(
            self.support_adjs, [label for _, _, label in SUPPORT_VIEWS],
            self.splits)
        vc = self.view_coverage
        print(f"View coverage (upstream, kept on purpose): "
              f"{vc['genes_with_view_row']}/{vc['num_genes']} genes have a "
              f"view row; share of pairs whose genes BOTH do:")

        def _f(d):
            v = d["both_have_view"]
            return "  n/a" if v is None else f"{v:.3f}"

        for fc in vc["folds"]:
            print(f"  fold {fc['fold']}: val pos {_f(fc['val_pos'])} / neg "
                  f"{_f(fc['val_neg'])} | test pos {_f(fc['test_pos'])} / "
                  f"neg {_f(fc['test_neg'])}")

    def save_predictions(self, fold_idx, x_idx, y_idx, predictions):
        """Save the full symmetric score matrix as fold_<k>_<cv>_predictions.npy."""
        n = self.num_nodes
        mat = np.zeros((n, n), dtype=np.float32)
        mat[x_idx, y_idx] = predictions
        mat[y_idx, x_idx] = predictions
        out = Path(self.args.output_dir) / \
            f"fold_{fold_idx}_{self.args.cv_type}_predictions.npy"
        np.save(out, mat)
        print(f"  Saved {n}x{n} score matrix -> {out.name}")

    def train(self):
        print("=" * 60)
        print(f"SLMGAE on shared folds ({self.args.cv_type.upper()}), "
              f"{len(self.splits)} folds")
        print("=" * 60)
        # CLEAR EVERY FOLD'S LEFTOVER MATRIX BEFORE FOLD 0.
        #
        # The per-fold clear inside the loop only protects folds this run
        # actually reaches. A run that dies at fold 3 leaves folds 3-4 holding
        # the PREVIOUS run's matrices, and evaluate_model.py grades every
        # matrix it finds in the directory — publishing a mean over fresh and
        # foreign folds at exit 0. A crashed fold must be MISSING, not stale.
        # Every fold must carry a two-class val split (selection is val-only);
        # check them all before anything is deleted, as SLGNN / NSF4SL do.
        for split in self.splits:
            build_selection_split(split)
        purge_stale_predictions(self.args.output_dir, self.args.cv_type)
        # The summary is written only after the last fold: a crashed rerun
        # must not leave the previous run's summary beside this run's matrices.
        (Path(self.args.output_dir) / "training_summary.json").unlink(
            missing_ok=True)

        aucs, aps, f1s = [], [], []
        for split in self.splits:
            auc, ap, f1 = self.train_fold_from_split(split["fold"], split)
            aucs.append(auc)
            aps.append(ap)
            f1s.append(f1)
        summary = {
            "cv_type":
            self.args.cv_type,
            "auc_mean":
            float(np.mean(aucs)),
            "auc_std":
            float(np.std(aucs)),
            "ap_mean":
            float(np.mean(aps)),
            "ap_std":
            float(np.std(aps)),
            "f1_mean":
            float(np.mean(f1s)),
            "f1_std":
            float(np.std(f1s)),
            "note":
            "Informational sklearn metrics; authoritative comparison "
            "metrics come from sl_comparison/evaluate_model.py on the "
            "saved score matrices.",
            # Early stopping + checkpoint selection: metric, and per fold the
            # selected epoch / val value / NaN-evaluation count (ValSelector).
            "select_metric":
            self.args.select_metric,
            "selection":
            getattr(self, "selection_log", []),
            # Which modalities actually fed the model. A ZERO-FILLED entry means
            # that view's source file was absent and SLMGAE ran reduced (only
            # with --allow_missing_views) — read this before interpreting the
            # comparison table.
            "support_views":
            self.support_provenance,
            # How List_Proteins symbols reached Entrez; id_space ==
            # LP_STATIC_FALLBACK_SPACE marks a reduced run (HGNC unavailable;
            # only with --allow_static_lp_mapping).
            "lp_mapping":
            self.lp_mapping,
            # Upstream view coverage vs labels, per fold (measured only).
            "view_coverage":
            self.view_coverage,
            # Full CLI (seed included), as the base Trainer.train() and the
            # other models' results.json record it.
            "args":
            vars(self.args),
        }
        with open(f"{self.args.output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"\nAUC {summary['auc_mean']:.4f} | AP {summary['ap_mean']:.4f} "
            f"| F1 {summary['f1_mean']:.4f}  (self-eval; see evaluate_model.py)"
        )


def main():
    p = argparse.ArgumentParser(description="SLMGAE on shared SLDB 3.0 folds")
    p.add_argument("--folds_dir", type=str, default="../sl_comparison/folds")
    p.add_argument("--cv_type",
                   type=str,
                   default="cv3",
                   choices=["cv1", "cv2", "cv3"])
    p.add_argument("--data_path", type=str, default="../data")
    p.add_argument("--output_dir",
                   type=str,
                   default=None,
                   help="Default: results/slmgae_shared_manual_<cv_type>. "
                   "The comparison's results/slmgae_shared_<cv_type> is "
                   "written only by sl_comparison/slurm/run_shared_models.sh, "
                   "which passes it explicitly.")
    p.add_argument("--allow_missing_views",
                   action="store_true",
                   help="Zero-fill a missing GO/PPI support view instead of "
                   "aborting. Runs a REDUCED SLMGAE, not the published model; "
                   "recorded in training_summary.json and model_params.json.")
    p.add_argument("--allow_static_lp_mapping",
                   action="store_true",
                   help="When the HGNC table cannot be loaded, map "
                   "List_Proteins_in_SL.txt through data/gene_id_mapping.tsv "
                   "(fewer rows) instead of aborting. Runs a REDUCED SLMGAE; "
                   "recorded as lp_mapping.id_space in the same two files.")
    # SLMGAE hyperparameters (defaults match train_slmgae.py).
    p.add_argument("--hidden1", type=int, default=512)
    p.add_argument("--hidden2", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--nn_size", type=int, default=45)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--coe", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eva_epochs", type=int, default=25)
    # NOTE: 20 evaluations x 25 epochs = 500 > 300 epochs, so with these
    # defaults patience can never fire; all 12 evaluations run and selection
    # is a pure val argmax. Kept as-is (matches the published schedule).
    p.add_argument("--early_stopping", type=int, default=20)
    p.add_argument("--select_metric",
                   type=str,
                   default="auprg",
                   choices=list(SELECT_METRICS),
                   help="Val metric for early stopping AND checkpoint "
                   "selection: auprg (default; sl_comparison/auprg.py, same "
                   "as the shared evaluator) or auroc (the old rule).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output_dir is None:
        # Never the graded results/slmgae_shared_<cv>: external transfer scores
        # whatever matrices sit there next to the benchmark's row JSON.
        args.output_dir = f"results/slmgae_shared_manual_{args.cv_type}"

    trainer = SharedFoldTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
