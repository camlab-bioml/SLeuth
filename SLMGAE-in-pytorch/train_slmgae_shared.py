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
  - A support view whose source file is absent becomes an ALL-ZERO view instead
    of aborting the run (see build_support_views). This mirrors the rule the
    shared SL adjacency already uses: SLDB 3.0 positives go in, everything else
    is 0. It keeps SLMGAE in the four-model table when a modality is
    unavailable — at the cost of running a reduced model, which is recorded in
    training_summary.json["support_views"] and printed at startup.
  - Each fold's full gene x gene score matrix is saved as
    fold_<k>_<cv>_predictions.npy so sl_comparison/evaluate_model.py scores it
    with the same cal_metrics evaluator as every other model.

The SLMGAE model, loss, and per-fold training loop are inherited unchanged.

Usage (from SLMGAE-in-pytorch/):
  python train_slmgae_shared.py --cv_type cv3 --folds_dir ../sl_comparison/folds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from slmgae_pytorch import DataLoader, set_random_seed
from train_slmgae import Trainer

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
# and will misalign every similarity row (load_dense_feature just truncates).
#
# SLMGAE's three support views, in the order the model consumes them.
# (filename, loader kind, display name). The two GO similarity matrices ship
# with the original SLMGAE release and are NOT produced by anything in this
# repo (`grep -rn GOsim` finds only readers), so they can legitimately be
# absent; biogrid_ppi_sparse.txt is tracked and normally present.
SUPPORT_VIEWS = [
    ("Human_GOsim.txt", "dense", "GO BP similarity"),
    ("Human_GOsim_CC.txt", "dense", "GO CC similarity"),
    ("biogrid_ppi_sparse.txt", "sparse", "BioGRID PPI"),
]


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
    sia = Path(__file__).resolve().parent.parent / "siamese_sl"
    try:
        sys.path.insert(0, str(sia))
        from gene_name_utils import GeneNameMapper  # noqa: E402
        mapper = GeneNameMapper(
            cache_dir=str(Path(data_path).resolve() / "cache"))
        # drop_unmapped=False keeps exactly ONE entry per input row (None when
        # a symbol will not map). Row order IS the support-view index, so a
        # shortened list would shift every similarity row by one.
        eids, _ = mapper.symbols_to_entrez_list(raw, drop_unmapped=False)
        if len(eids) != len(raw):
            raise RuntimeError(f"mapper returned {len(eids)} ids for "
                               f"{len(raw)} rows — would shift alignment")
        genes = [e or "" for e in eids]
    except Exception as e:  # noqa: BLE001
        print(f"  (GeneNameMapper unavailable: {type(e).__name__}: {e}; "
              f"falling back to data/gene_id_mapping.tsv)")
    if genes is None:
        static = {}
        tsv = Path(data_path) / "gene_id_mapping.tsv"
        if tsv.exists():
            for line in open(tsv):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    static[p[1].strip()] = p[0].strip()
        genes = [static.get(g, "") for g in raw]
    n_unmapped = sum(1 for g in genes if not g)
    print(f"  List_Proteins is in SYMBOL space -> mapped to Entrez "
          f"({len(raw) - n_unmapped}/{len(raw)}; {n_unmapped} unmapped)")
    return genes, "symbol->entrez", n_unmapped


def build_support_views(data_path, nn_size, lp_genes, shared_g2i, num_nodes):
    """Build the support views in the SHARED gene index, zero-filling gaps.

    Each view is read in the List_Proteins index space and remapped into the
    shared universe. A view whose source file is missing becomes an all-zero
    matrix rather than raising: normalize_adj() adds self-loops, so a zero view
    degrades to the identity and the model's view-attention learns to ignore
    it. SLMGAE then still trains and still lands in the comparison table.

    Returns (support_adjs, provenance); provenance maps display name -> the
    source path used, or a "zero-filled" marker naming the missing file.
    """
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
            view = (dl.load_dense_feature(str(path), knn=True)
                    if kind == "dense" else dl.load_sparse_feature(str(path)))
            support_adjs.append(
                remap_support(view, lp_genes, shared_g2i, num_nodes))
            provenance[label] = str(path)
        else:
            print(f"WARNING: {label} source not found ({path}) -> using an "
                  f"all-zero view; SLMGAE runs WITHOUT this modality.")
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
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{args.output_dir}/checkpoints").mkdir(parents=True,
                                                     exist_ok=True)

        # Shared gene universe + folds.
        self.gene_order, self.num_nodes, self.splits = load_shared_folds(
            args.folds_dir, args.cv_type)
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
        self.support_adjs, self.support_provenance = build_support_views(
            args.data_path, args.nn_size, lp_genes, shared_g2i, self.num_nodes)
        print(f"Shared universe: {self.num_nodes} genes | folds: "
              f"{len(self.splits)} | support views remapped "
              f"({matched}/{len(lp_genes)} List_Proteins genes in universe, "
              f"{frac:.1%}, id_space={id_space})")
        for label, src in self.support_provenance.items():
            print(f"  support view: {label:<22s} {src}")

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
        purge_stale_predictions(self.args.output_dir, self.args.cv_type)

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
            # Which modalities actually fed the model. A ZERO-FILLED entry means
            # that view's source file was absent and SLMGAE ran reduced — read
            # this before interpreting the comparison table.
            "support_views":
            self.support_provenance,
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
                   help="Default: results/slmgae_shared_<cv_type>")
    p.add_argument("--num_folds", type=int, default=5)
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
    p.add_argument("--early_stopping", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"results/slmgae_shared_{args.cv_type}"

    trainer = SharedFoldTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
