#!/usr/bin/env python3
"""Train ELISL (context-free: Sequence_1024 + PPI_ec) on the shared folds.

ELISL (Tepeli, Seale & Goncalves, Bioinformatics 2024;
github.com/joanagoncalveslab/ELISL) as the authors ship it, restricted to its
two context-free sources (the user's choice: the benchmark is context-free SL
over SynLethDB, so the three cancer-specific sources are left out):

  * one LightGBM random forest (ELRRF, boosting_type 'rf') per source on
    |e(g1) - e(g2)| features: SeqVec 1,024 dims ("Sequence_1024") and STRING
    node2vec 64 dims ("PPI_ec"), plus one on their concatenation (the early
    integration single_cancer_experiment.py adds whenever a config lists
    more than one source: "Sequence_1024&PPI_ec");
  * StandardScaler + VarianceThreshold(1e-4) fitted on the training rows
    (--process true, as the README's run command);
  * the training set undersampled to 1:1 (balance strategy
    undersample_train_test, train side);
  * Bayesian hyperparameter search (skopt gp_minimize, 20 calls, ELRRF's
    get_tree_search_grid; the README's --grid_search true);
  * late integration: the three models' probabilities averaged with weights
    = each model's mean AUPRC over a 10-fold stratified CV of the training
    set (experiment.fit_predict_2set -> ELRRF.calculate_data_weights).

The model code is the authors' (elisl_orig/, GPL v3, see ELISL/README.md for
the five compatibility lines). What this file changes is the harness, as for
SLGNN / NSF4SL: the shared folds in (--folds_dir), one model per fold (the
authors' 10 undersampling repetitions -> the 5 shared folds, repetition seed
`seed + fold`), the search's objective on the shared VALIDATION split
(val AUPRG of the late-integrated score, sl_comparison/auprg.auprg; the
authors' objective is 1 - mean AUPRC over a 10-fold CV of the training set),
and the 8,844 x 8,844 float32 fold_<k>_<cv>_predictions.npy over every gene
pair, scored by the same fitted transforms and forests.

Usage (repo root or ELISL/; the comparison's results/elisl_<cv> is written
only by ELISL/slurm/run_elisl.sh):
    python ELISL/train_elisl.py --cv_type cv3 --folds_dir sl_comparison/folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO / "sl_comparison"))
from elisl_orig import ELRRF as E  # noqa: E402
from elisl_orig import features as F  # noqa: E402
from shared_folds import load_shared_folds  # noqa: E402
from model_params import purge_stale_predictions, write_model_params  # noqa: E402
from auprg import auprg  # noqa: E402

DEFAULT_FEATURES = REPO / "data" / "elisl" / "gene_features.npz"
SEQ, PPI = "Sequence_1024", "PPI_ec"
SOURCES = (SEQ, PPI)
COMBO = "&".join(SOURCES)          # single_cancer_experiment.py's early set
DATASETS = (SEQ, PPI, COMBO)
# create_seq_features(emb_name='seq') / create_ppi_features(emb_name='ppi_ec')
COL_PREFIX = {SEQ: "seq", PPI: "ppi_ec"}

METRIC_KEYS = ["auroc", "f1", "aupr", "auprg", "ndcg@10", "ndcg@20", "ndcg@50",
               "recall@10", "recall@20", "recall@50", "precision@10",
               "precision@20", "precision@50", "map@10", "map@20", "map@50"]


# ---------------------------------------------------------------------------
# Grading helpers: the same definitions as SLGNN/train_slgnn.py (the shared
# evaluator's float32 shift, cal_metrics on the shifted matrix).
# ---------------------------------------------------------------------------

def graded_pair_scores(mat, pairs):
    mat = np.asarray(mat, dtype=np.float32)
    s = mat[pairs[:, 0], pairs[:, 1]]
    return (s - mat.min() + np.float32(1.0)).astype(np.float32)


def graded_seen(folds_dir, test_pos, train_pos):
    p = Path(folds_dir) / "all_pos.npy"
    if not p.exists():
        return train_pos
    all_pos = np.load(p).astype(np.int64)
    keep = set(map(tuple, np.asarray(test_pos, dtype=np.int64).tolist()))
    mask = np.fromiter((tuple(e) not in keep for e in all_pos.tolist()),
                       dtype=bool, count=len(all_pos))
    return all_pos[mask]


def evaluate_fold(mat, pos, neg, seen_pos):
    from evaluation import evaluate_predictions_dict
    if not np.isfinite(mat).all():
        return {k: float("nan") for k in METRIC_KEYS}
    ranked = (mat - mat.min() + np.float32(1.0)).astype(np.float32)
    metrics = evaluate_predictions_dict(ranked, pos, neg, seen_pos)
    pairs = np.concatenate([pos, neg]).reshape(-1, 2)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    metrics["auprg"] = auprg(y, graded_pair_scores(mat, pairs))
    return metrics


def require_val(split, fold_label):
    vp, vn = split.get("val_pos"), split.get("val_neg")
    if vp is None or vn is None or len(vp) == 0 or len(vn) == 0:
        raise SystemExit(
            f"{fold_label}: no usable validation split "
            f"(val_pos={0 if vp is None else len(vp)}, "
            f"val_neg={0 if vn is None else len(vn)}). ELISL's hyperparameter "
            f"search runs on validation only and will not fall back to TEST. "
            f"Pass --folds_dir with a fold set built by "
            f"sl_comparison/prepare_folds.py with --val_frac > 0.")
    return np.asarray(vp).astype(np.int64), np.asarray(vn).astype(np.int64)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_features(path, gene_order):
    """data/elisl/gene_features.npz (build_elisl_inputs.py features), in the
    shared gene order or refused."""
    z = np.load(path, allow_pickle=False)
    genes = [str(g) for g in z["genes"]]
    if genes != list(gene_order):
        raise SystemExit(
            f"{path} was built for another gene universe ({len(genes)} genes, "
            f"first {genes[:3]}) than the folds' genes.txt "
            f"({len(gene_order)} genes). Rebuild it: "
            f"python ELISL/build_elisl_inputs.py features")
    # The authors key pairs by name ('gene1|gene2') and merge the source
    # frames on it, so two genes sharing a symbol would duplicate rows: such
    # a name is suffixed with its Entrez id (it only labels; the lookup used
    # the plain symbol).
    names = [str(n) for n in z["names"]]
    from collections import Counter
    dup = {n for n, c in Counter(names).items() if c > 1}
    names = [f"{n}#{g}" if n in dup else n for n, g in zip(names, genes)]
    return {"seq": z["seq"], "seq_missing": z["seq_missing"].astype(bool),
            "ppi": z["ppi"], "ppi_missing": z["ppi_missing"].astype(bool),
            "names": names}


def feature_block(feats, a, b):
    """{dataset: (n, d) float64} for pairs (a[k], b[k]), the authors' values
    (create_*_features, then read_csv(...).fillna(0))."""
    seq = F.pair_features(feats["seq"], feats["seq_missing"], a, b)
    ppi = F.pair_features(feats["ppi"], feats["ppi_missing"], a, b)
    for x in (seq, ppi):
        np.nan_to_num(x, copy=False, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return {SEQ: seq, PPI: ppi, COMBO: np.hstack([seq, ppi])}


def sample_frames(feats, pairs, labels):
    """The authors' per-source DataFrames for labelled pairs: columns
    pair_name, class, <prefix>0..; rows sorted by (gene1, gene2) names, as
    experiment.prepare_cancer_dataset sorts them, so the undersampling draw
    and the forests' bagging see the authors' row order. Returns
    (samples, {dataset: frame}, pairs in that order)."""
    names = feats["names"]
    g1 = [names[i] for i in pairs[:, 0]]
    g2 = [names[i] for i in pairs[:, 1]]
    base = pd.DataFrame({"gene1": g1, "gene2": g2,
                         "class": labels.astype(int), "_k": np.arange(len(g1))})
    base = base.sort_values(by=["gene1", "gene2"], kind="mergesort")
    order = base["_k"].to_numpy()
    pairs = pairs[order]
    pair_name = (base["gene1"] + "|" + base["gene2"]).to_numpy()
    samples = pd.DataFrame({"pair_name": pair_name,
                            "class": base["class"].to_numpy()})
    block = feature_block(feats, pairs[:, 0], pairs[:, 1])
    frames = {}
    for src in SOURCES:
        x = block[src]
        cols = [f"{COL_PREFIX[src]}{i}" for i in range(x.shape[1])]
        frames[src] = pd.concat([samples, pd.DataFrame(x, columns=cols)],
                                axis=1)
    # single_cancer_experiment.py: samples merged (inner, on pair_name) with
    # each source's frame minus its class column, in config order.
    combo = samples.copy()
    for src in SOURCES:
        combo = pd.merge(combo, frames[src].drop(columns="class"), how="inner",
                         on=["pair_name"])
    frames[COMBO] = combo
    return samples, frames, pairs


# ---------------------------------------------------------------------------
# The model: the authors' ELRRF, plus what the matrix export needs
# ---------------------------------------------------------------------------

def make_model(seed, grid_search):
    # single_cancer_experiment.py's constructor call for --model ELRRF
    # --balance_strategy undersample_train_test --process true
    # --threshold 0.5, with cancer=None (no per-cancer filter).
    return E.ELRRF(boosting_type="rf", use_single=True,
                   grid_searched=grid_search,
                   balance_strategy="undersample_train_test", use_comb=False,
                   thold=0.5, process=True, cancer=None, random_state=seed)


def fuse(predictions, names=DATASETS):
    """evaluate_all_models' late integration: per pair, sum of probabilities
    weighted by tr_auc / sum(tr_auc) (combine_preds). All models score the
    same rows in the same order, so the groupby is a weighted sum."""
    w = np.asarray([float(predictions[n]["tr_auc"]) for n in names])
    w = w / w.sum()
    p = np.stack([np.asarray(predictions[n]["probabilities"], dtype=np.float64)
                  for n in names])
    return (w[:, None] * p).sum(axis=0), dict(zip(names, w.tolist()))


def val_auprg(prob, y):
    return float(auprg(y, (prob - prob.min() + 1.0).astype(np.float32)))


def grid_search(model, fold, sample_dict, samples_tr, samples_val, y_val, seed,
                n_calls):
    """experiment.fit_predict_2set's gp_minimize over get_tree_search_grid,
    20 calls, random_state = the model's. HARNESS: the objective is
    1 - val AUPRG of the late-integrated validation score (the three forests
    fitted on the undersampled training set, fused with training-AP weights
    as the authors' search fuses them), not 1 - mean AUPRC of a 10-fold CV of
    the training set."""
    from skopt import gp_minimize
    from skopt.space import Categorical
    from skopt.utils import use_named_args
    space = [Categorical(v, name=k)
             for k, v in model.get_tree_search_grid().items()]
    trace = []

    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        # a fold key not in model.contr -> tr_auc is the training AP
        preds = model.fit_predict_all_datasets_2sets(
            sample_dict, samples_tr, samples_val, fold=f"search_{fold}")
        v = val_auprg(fuse(preds)[0], y_val)
        trace.append({"params": {k: _py(x) for k, x in params.items()},
                      "val_auprg": v})
        print(f"    search {len(trace):2d}/{n_calls}: val AUPRG {v:.4f} "
              f"{trace[-1]['params']}")
        return 1.0 - v if np.isfinite(v) else 1.0

    res = gp_minimize(objective, space, n_calls=n_calls, random_state=seed,
                      n_jobs=-1)
    best = {s.name: _py(x) for s, x in zip(space, res.x)}
    model.set_params(**best)
    return best, trace


def _py(x):
    return x.item() if hasattr(x, "item") else x


def fit_for_export(model, sample_dict, fold):
    """Each dataset's fitted (scaler, variance filter, forest, class-1 column,
    weight), from the same calls fit_predict_all_datasets_2sets makes:
    prepare_train_test is deterministic, and fit_predict_report builds and
    fits the forest exactly as there. Also returns those predictions (the
    `test` rows of sample_dict)."""
    fitted, preds = {}, {}
    train_samples = None
    for name, both in model.both_datasets_dict.items():
        train_x, test_x, scaler, varthr = model.prepare_train_test(
            both, sample_dict, return_process_fits=True)
        test_x = varthr.transform(scaler.transform(test_x))
        y = both["train"].iloc[sample_dict["train"]]["class"].values
        probs, _, tr_auc, clf = model.fit_predict_report(train_x, y, test_x)
        one = int(np.where(clf.classes_ == 1)[0][0])
        w = model.contr.get(fold, {}).get(name, tr_auc)
        fitted[name] = {"scaler": scaler, "varthr": varthr, "clf": clf,
                        "one_ind": one, "weight": float(w)}
        preds[name] = {"probabilities": probs, "tr_auc": w}
        if train_samples is None:
            train_samples = len(y)
    return fitted, preds


def predict_pairs(fitted, feats, a, b):
    block = feature_block(feats, a, b)
    out = {}
    for name, f in fitted.items():
        x = f["varthr"].transform(f["scaler"].transform(block[name]))
        out[name] = {"probabilities": f["clf"].predict_proba(x)[:, f["one_ind"]],
                     "tr_auc": f["weight"]}
    return fuse(out)[0]


def score_matrix(fitted, feats, n, chunk):
    """Every unordered gene pair once (i < j), mirrored; diagonal 0. The
    features are symmetric in the two genes, so is the score."""
    mat = np.zeros((n, n), dtype=np.float32)
    rows = np.arange(n)
    i = 0
    t0 = time.time()
    while i < n - 1:
        # rows i..k-1 whose upper-triangle pairs fit in one chunk
        k, size = i, 0
        while k < n - 1 and (size == 0 or size + (n - 1 - k) <= chunk):
            size += n - 1 - k
            k += 1
        a = np.concatenate([np.full(n - 1 - r, r) for r in range(i, k)])
        b = np.concatenate([rows[r + 1:] for r in range(i, k)])
        p = predict_pairs(fitted, feats, a, b).astype(np.float32)
        mat[a, b] = p
        mat[b, a] = p
        i = k
        print(f"    scored rows < {k:,}/{n:,} "
              f"({time.time() - t0:,.0f}s)", flush=True)
    return mat


class ForestSize:
    """write_model_params counts `p.numel()` over `model.parameters()`. For
    the forests that is every fitted value: per tree, one threshold per split
    (num_leaves - 1) and one value per leaf, plus the late-integration
    weights. The scalers' means / scales are preprocessing statistics, not
    counted (as for every other model's feature standardisation)."""

    class _Count:
        requires_grad = True

        def __init__(self, n):
            self.n = n

        def numel(self):
            return self.n

    def __init__(self, fitted):
        self.per_dataset = {}
        for name, f in fitted.items():
            trees = f["clf"].booster_.dump_model()["tree_info"]
            self.per_dataset[name] = int(sum(2 * t["num_leaves"] - 1
                                             for t in trees))
        self.n = sum(self.per_dataset.values()) + len(fitted)

    def parameters(self):
        yield self._Count(self.n)


def train_fold(args, split, n_folds, feats, n_genes):
    fold = split["fold"]
    print(f"\n{'=' * 60}\nFold {fold + 1}/{n_folds}\n{'=' * 60}")
    out = Path(args.output_dir)
    stale = out / f"fold_{fold}_{args.cv_type}_predictions.npy"
    stale.unlink(missing_ok=True)

    tr_pairs = split["train_edges"].astype(np.int64)
    tr_y = split["train_labels"].astype(int)
    val_pos, val_neg = require_val(split, f"Fold {fold + 1}")
    val_pairs = np.vstack([val_pos, val_neg])
    val_y = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])

    samples_tr, frames_tr, _ = sample_frames(feats, tr_pairs, tr_y)
    samples_val, frames_val, val_sorted = sample_frames(feats, val_pairs,
                                                        val_y)
    y_val = samples_val["class"].to_numpy()

    model = make_model(args.seed, not args.no_grid_search)
    for name in DATASETS:
        model.add_dataset(name, frames_tr[name], frames_val[name])

    # experiment.fit_predict_2set: undersample the training rows to 1:1, seed
    # 124 + repetition (here --seed + fold; --seed defaults to 124).
    tr_idx = E.balance_by_index(samples_tr, np.arange(len(samples_tr)),
                                rand_seed=args.seed + fold)
    sample_dict = {"train": tr_idx, "test": np.arange(len(samples_val))}
    n_pos = int(samples_tr["class"].to_numpy()[tr_idx].sum())
    print(f"Train: {int(tr_y.sum())} pos, {int((tr_y == 0).sum())} neg -> "
          f"undersampled {n_pos} + {len(tr_idx) - n_pos} | Val: "
          f"{len(val_pos)} pos, {len(val_neg)} neg | Test: "
          f"{int(split['test_labels'].sum())} pos, "
          f"{int((split['test_labels'] == 0).sum())} neg")

    best, trace = {}, []
    if not args.no_grid_search:
        print(f"  Bayesian search ({args.n_calls} calls, objective val AUPRG)")
        best, trace = grid_search(model, fold, sample_dict, samples_tr,
                                  samples_val, y_val, args.seed, args.n_calls)
        print(f"  best: {best}")

    # experiment.fit_predict_2set, EL models: late-integration weights from a
    # 10-fold stratified CV of the (undersampled) training rows.
    samples_cont = samples_tr.copy().iloc[tr_idx]
    inner = E.fit_predict_cross_validation(model, samples_cont,
                                           fold_type="stratified_shuffled",
                                           n_split=10, train_idx=tr_idx)
    model.calculate_data_weights(inner, fold_main=fold)
    print(f"  late-integration weights (inner 10-fold AUPRC): "
          f"{ {k: round(float(v), 4) for k, v in model.contr[fold].items()} }")

    fitted, preds = fit_for_export(model, sample_dict, fold)
    val_prob, weights = fuse(preds)
    v_auprg = val_auprg(val_prob, y_val)
    print(f"  val AUPRG {v_auprg:.4f} (late-integrated)")

    size = ForestSize(fitted)
    write_model_params(
        args.output_dir, size, folds_dir=args.folds_dir, cv_type=args.cv_type,
        selection_regime="val", selection_metric="auprg", seed=args.seed,
        elisl_sources="|".join(SOURCES), context_free=True,
        grid_search=not args.no_grid_search,
        input_meta_sha256=args.input_meta_sha256,
        params_per_dataset=size.per_dataset)

    print(f"  scoring all {n_genes * (n_genes - 1) // 2:,} gene pairs")
    mat = score_matrix(fitted, feats, n_genes, args.chunk)
    # The export path must reproduce the model's own validation scores.
    graded_val = mat[val_sorted[:, 0], val_sorted[:, 1]].astype(np.float64)
    if not np.allclose(graded_val, val_prob, rtol=0, atol=1e-6):
        raise RuntimeError(
            f"Fold {fold + 1}: the exported matrix does not reproduce the "
            f"model's validation scores (max |diff| "
            f"{np.abs(graded_val - val_prob).max():.3g})")

    test_edges = split["test_edges"].astype(np.int64)
    test_pos = test_edges[split["test_labels"] == 1]
    test_neg = test_edges[split["test_labels"] == 0]
    seen = split["train_edges"][split["train_labels"] == 1].astype(np.int64)
    metrics = evaluate_fold(mat, test_pos, test_neg,
                            graded_seen(args.folds_dir, test_pos, seen))
    metrics.update({"val_auprg": v_auprg, "best_params": best,
                    "search_trace": trace, "fusion_weights": weights,
                    "trainable_params": size.n,
                    "params_per_dataset": size.per_dataset,
                    "n_train_undersampled": int(len(tr_idx))})
    np.save(out / f"fold_{fold}_{args.cv_type}_predictions.npy", mat)
    print(f"  Fold {fold + 1}: AUPRG={metrics['auprg']:.4f} "
          f"AUROC={metrics['auroc']:.4f}")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(HERE / "results" / f"elisl_manual_{args.cv_type}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gene_order, n_genes, splits = load_shared_folds(args.folds_dir,
                                                    args.cv_type)
    for split in splits:
        require_val(split, f"{args.cv_type} fold {split['fold'] + 1}")
    feats = load_features(args.features, gene_order)
    meta = Path(args.features).with_name("meta.json")
    import hashlib
    args.input_meta_sha256 = (hashlib.sha256(meta.read_bytes()).hexdigest()
                              if meta.exists() else None)
    print(f"Shared folds {args.folds_dir}/{args.cv_type}: {len(splits)} folds, "
          f"{n_genes} genes | SeqVec for "
          f"{int((~feats['seq_missing']).sum())}, PPI for "
          f"{int((~feats['ppi_missing']).sum())}")

    purge_stale_predictions(args.output_dir, args.cv_type)
    (out / "results.json").unlink(missing_ok=True)

    fold_metrics = [train_fold(args, s, len(splits), feats, n_genes)
                    for s in splits]
    summary = {"cv_type": args.cv_type, "model": "ELISL",
               "elisl_sources": "|".join(SOURCES), "context_free": True,
               "features": str(args.features),
               "input_meta_sha256": args.input_meta_sha256,
               "folds": fold_metrics}
    for key in METRIC_KEYS + ["val_auprg"]:
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        print(f"  {key:14s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    a = vars(args).copy()
    a["features"] = str(a["features"])
    summary["args"] = a
    with open(out / "results.json", "w") as f:
        json.dump({"summary": summary}, f, indent=2, default=_py)
    print(f"\nResults saved to {out / 'results.json'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ELISL (context-free)")
    p.add_argument("--folds_dir", required=True,
                   help="shared folds with a validation split "
                   "(sl_comparison/prepare_folds.py --val_frac > 0)")
    p.add_argument("--cv_type", default="cv1", choices=["cv1", "cv2", "cv3"])
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                   help="build_elisl_inputs.py's gene_features.npz")
    p.add_argument("--output_dir", default=None,
                   help="default ELISL/results/elisl_manual_<cv>; the "
                   "comparison's ELISL/results/elisl_<cv> is written by "
                   "ELISL/slurm/run_elisl.sh only")
    p.add_argument("--seed", type=int, default=124,
                   help="replaces the authors' fixed 124: model random_state "
                   "(search, inner CV) and the undersampling seed seed + fold")
    p.add_argument("--no_grid_search", action="store_true",
                   help="skip the Bayesian search; ELRRF's constructor "
                   "defaults (the authors' --grid_search false)")
    p.add_argument("--n_calls", type=int, default=20,
                   help="gp_minimize calls (the authors' 20)")
    p.add_argument("--chunk", type=int, default=200_000,
                   help="gene pairs scored per block of the matrix export")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
