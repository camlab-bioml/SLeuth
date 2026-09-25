#!/usr/bin/env python3
"""
Generate ONE canonical set of CV folds shared by all four SL models
(siamese_sl, SLMGAE, SLGNN, NSF4SL) so their comparison is fair: identical
train/test division, identical gene index space, one fixed seed.

Data: SLDB 3.0 given pairs.
  positives = data/SL_SynLethDB_experimental.txt            (~22.2k pairs)
  negatives = data/SL_SynLethDB_experimental_negatives.txt  (~112.2k pairs, ALL used)

Gene universe = union of every gene appearing in a given positive OR negative
pair (so every given negative is scorable). Genes are indexed 0..N-1 by sorted
Entrez ID; that index space is written to genes.txt and every fold's edge arrays
index into it.

Label: a pair is POSITIVE if any screen called it SL in any cell line, NEGATIVE
only if every screen of it called it non-SL. The 5,542 pairs screened both ways
are therefore positive; conflicts.tsv records the cell lines and heads behind
each one (a side naming no usable line is attributed to OTHER). This is the same
relation siamese's *_any OR-collapse is graded on, so the shared binary folds
and the cell-line-conditioned model score identical ground truth.

For each cv in {cv1, cv2, cv3} and each of k folds we write four (M,2) int32
arrays of gene-index pairs:
  train_pos, train_neg, test_pos, test_neg
plus meta.json. GNN-structure models build their graph from train_pos only; the
negatives are label-0 for train/eval (never graph edges).

CV definitions (gene-disjoint for cv2/cv3, leak-free):
  cv1 : edge split  — positives and negatives each KFold'd by pair.
  cv2 : gene split  — hold out a gene group; test = pairs touching >=1 held gene,
                      train = pairs touching none (>=1 unseen gene at test).
  cv3 : pair split  — test = pairs with BOTH genes held out, train = pairs with
                      NEITHER held out (cold-start; single-held-gene pairs dropped).

Usage (from repo root):
  python sl_comparison/prepare_folds.py --verify
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_pairs(path):
    """Read a whitespace-separated Entrez pair file -> list of (str, str)."""
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def sidecar_path(pair_path):
    """data/X.txt -> data/X.sources.tsv (provenance written by
    download_synlethdb.py: entrez1, entrez2, sources, cell_lines, pubmed_ids)."""
    return Path(pair_path).with_suffix(".sources.tsv")


def _vocab():
    """Return siamese's cell_line_vocab module, or None if unavailable.

    The shared folds must read cell lines with the SAME vocabulary siamese
    trains on, otherwise the two label definitions diverge. A raw upper-cased
    string intersection is not equivalent: cell_line_vocab merges aliases
    (X293T -> 293T) and drops placeholders (TBD, NA, cancer-type tokens).
    Measured on the committed data, the alias merge alone moves 31 pairs
    between "same-line contradiction" and "cross-line".
    """
    import sys
    vocab_dir = Path(__file__).resolve().parent.parent / "siamese_sl"
    if str(vocab_dir) not in sys.path:
        sys.path.insert(0, str(vocab_dir))
    try:
        import cell_line_vocab
        return cell_line_vocab
    except ImportError:
        return None


def read_pair_cell_lines(path):
    """Map canonical (entrez1, entrez2) -> set of cell lines it was screened in.

    Returns {} if the sidecar is absent, which callers must treat as "cannot
    resolve conflicts by cell line".
    """
    p = Path(path)
    if not p.exists():
        return {}
    vocab = _vocab()
    out = {}
    with open(p) as f:
        next(f, None)  # header
        for line in f:
            t = line.rstrip("\n").split("\t")
            if len(t) < 4:
                continue
            a, b = t[0], t[1]
            key = (a, b) if a < b else (b, a)
            if vocab is not None:
                cells = vocab.parse_cell_field(t[3])
            else:
                cells = {
                    c.strip().upper()
                    for c in t[3].split(";") if c.strip()
                }
            out.setdefault(key, set()).update(cells)
    return out


MISSING_LINE = "MISSING"


def conflict_cell_lines(pos_path, neg_path, conflict_keys, policy):
    """Attribute every SL/non-SL conflict to the cell lines it was screened in.

    A pair listed as SL in the positives file AND non-SL in the negatives file
    is only a real contradiction when the SAME cell line is asserted both ways.
    When the two assertions come from DIFFERENT lines (SL in A375, non-SL in
    K562) there is no contradiction at all — that is the cell-line specificity
    of synthetic lethality, and the pair is genuinely "SL in at least one line".

    A side that names no usable line is recorded as MISSING and attributed to
    the OTHER head, matching cell_line_vocab.heads_for_field. That is what makes
    every one of these pairs attributable to a cell-line category instead of
    being discarded as unlabelled.

    Returns key -> {sl_lines, ns_lines, sl_heads, ns_heads, same_line,
    same_head}, with lines/heads as sorted lists and the two flags as bools.
    """
    vocab = _vocab()
    if vocab is None:
        raise ImportError(
            f"conflict_policy={policy!r} needs siamese_sl/cell_line_vocab.py "
            "so the shared folds read cell lines with the same vocabulary "
            "siamese trains on. Falling back to raw string matching would give "
            "the two pipelines different labels — use --conflict_policy drop "
            "if you genuinely want the vocabulary-free behaviour.")
    pc = read_pair_cell_lines(sidecar_path(pos_path))
    nc = read_pair_cell_lines(sidecar_path(neg_path))
    if not pc or not nc:
        raise FileNotFoundError(
            f"conflict_policy={policy!r} needs the provenance sidecars "
            f"{sidecar_path(pos_path)} and {sidecar_path(neg_path)} "
            f"(produced by download_synlethdb.py). Not found.")
    out = {}
    for key in conflict_keys:
        sl_lines, ns_lines = pc.get(key, set()), nc.get(key, set())
        sl_heads = {
            vocab.HEADS[h]
            for h in vocab.heads_for_field(";".join(sl_lines), True)
        }
        ns_heads = {
            vocab.HEADS[h]
            for h in vocab.heads_for_field(";".join(ns_lines), True)
        }
        out[key] = {
            "sl_lines": sorted(sl_lines) or [MISSING_LINE],
            "ns_lines": sorted(ns_lines) or [MISSING_LINE],
            "sl_heads": sorted(sl_heads),
            "ns_heads": sorted(ns_heads),
            "same_line": bool(sl_lines & ns_lines),
            "same_head": bool(sl_heads & ns_heads),
        }
    return out


def write_conflict_report(path, gene_order, conflicts, detail, kept):
    """Write the per-pair audit trail for the pairs screened both ways.

    5,542 of ~22k positives are in this file. A comparison table that does not
    say what happened to them is not reproducible, and "we took cell lines into
    account" is not checkable from meta.json counts alone — so every pair is
    listed with the lines and heads on each side and the label it received.
    """
    rows = []
    for e in sorted(conflicts):
        a, b = gene_order[e[0]], gene_order[e[1]]
        key = (a, b) if a < b else (b, a)
        d = detail[key]
        rows.append("\t".join([
            key[0],
            key[1],
            ";".join(d["sl_lines"]),
            ";".join(d["ns_lines"]),
            ";".join(d["sl_heads"]),
            ";".join(d["ns_heads"]),
            "same_line" if d["same_line"] else "cross_line",
            "same_head" if d["same_head"] else "cross_head",
            "positive" if tuple(e) in kept else "dropped",
        ]))
    with open(path, "w") as f:
        f.write("entrez1\tentrez2\tsl_lines\tnonsl_lines\tsl_heads\t"
                "nonsl_heads\tline_agreement\thead_agreement\tshared_label\n")
        f.write("\n".join(rows) + "\n")


def subsample_genes(pos_pairs, neg_pairs, max_genes):
    """Keep the top-`max_genes` most positive-connected genes (for a fast
    preflight smoke test); filter both pair lists to pairs within that set."""
    from collections import Counter
    deg = Counter()
    for a, b in pos_pairs:
        deg[a] += 1
        deg[b] += 1
    top = {g for g, _ in deg.most_common(max_genes)}
    fp = [(a, b) for a, b in pos_pairs if a in top and b in top]
    fn = [(a, b) for a, b in neg_pairs if a in top and b in top]
    return fp, fn


def build_universe(pos_pairs, neg_pairs):
    """Sorted-by-Entrez gene list + Entrez->index map over pos UNION neg genes."""
    genes = set()
    for a, b in pos_pairs + neg_pairs:
        genes.add(a)
        genes.add(b)
    # Sort numerically when possible (Entrez IDs), else lexically — deterministic.
    def key(g):
        return (0, int(g)) if g.isdigit() else (1, g)

    gene_order = sorted(genes, key=key)
    g2i = {g: i for i, g in enumerate(gene_order)}
    return gene_order, g2i


def to_index_pairs(pairs, g2i):
    """Map Entrez pairs to canonical (i<j) index pairs; drop self-loops; dedup."""
    seen = set()
    out = []
    for a, b in pairs:
        ia, ib = g2i[a], g2i[b]
        if ia == ib:
            continue
        e = (ia, ib) if ia < ib else (ib, ia)
        if e not in seen:
            seen.add(e)
            out.append(e)
    return np.asarray(out, dtype=np.int32), seen


def edge_kfold(arr, k, rng):
    """Yield (train, test) index-array pairs for a K-fold edge split."""
    n = len(arr)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for j in range(k):
        test_idx = folds[j]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != j])
        yield arr[train_idx], arr[test_idx]


def gene_kfold_groups(num_nodes, k, rng):
    """Partition gene indices into k disjoint held-out groups (as sets)."""
    perm = rng.permutation(num_nodes)
    return [set(g.tolist()) for g in np.array_split(perm, k)]


def split_by_genes(arr, held, mode):
    """Partition (M,2) pairs by held-out gene membership.

    mode 'cv2': test = touches >=1 held gene; train = touches none.
    mode 'cv3': test = BOTH genes held;       train = NEITHER held (drop the rest).
    """
    if len(arr) == 0:
        return arr.copy(), arr.copy()
    a_in = np.array([x in held for x in arr[:, 0]])
    b_in = np.array([x in held for x in arr[:, 1]])
    if mode == "cv2":
        test_mask = a_in | b_in
        train_mask = ~test_mask
    else:  # cv3
        test_mask = a_in & b_in
        train_mask = ~a_in & ~b_in
    return arr[train_mask], arr[test_mask]


def carve_val(arr, frac, rng):
    """Split (M,2) pairs into (train, val) by a random pair partition."""
    if frac <= 0 or len(arr) == 0:
        return arr, arr[:0].copy()
    n_val = max(1, int(round(len(arr) * frac)))
    perm = rng.permutation(len(arr))
    return arr[perm[n_val:]], arr[perm[:n_val]]


def generate(pos_idx, neg_idx, num_nodes, cv_type, k, seed, val_frac=0.1):
    """Return k dicts {train,val,test}_{pos,neg}.

    The validation set is carved out of TRAIN only — the test partition is
    byte-identical to what this function produced before validation existed
    (cv1 uses a separate RNG stream; cv2/cv3 reuse the same gene groups).

    For cv2/cv3 the val split must be cold-start in the SAME sense as test,
    or selecting on it would not reflect test conditions — so val is defined
    by held-out GENES, not by a random pair split. It is a val_frac-sized
    slice of the non-test genes rather than a whole 1/k group: a full group
    would cost ~40-55% of the training pairs (cv2 fold 0 measured 5,405 train
    vs 6,738 val positives), which is a worse trade than the selection is
    worth.
    """
    rng = np.random.RandomState(seed)
    folds = []
    if cv_type == "cv1":
        pos_splits = list(edge_kfold(pos_idx, k, rng))
        neg_splits = list(edge_kfold(neg_idx, k, rng))
        vrng = np.random.RandomState(seed + 7919)  # separate stream
        for j in range(k):
            tr_p, va_p = carve_val(pos_splits[j][0], val_frac, vrng)
            tr_n, va_n = carve_val(neg_splits[j][0], val_frac, vrng)
            folds.append({
                "train_pos": tr_p,
                "val_pos": va_p,
                "test_pos": pos_splits[j][1],
                "train_neg": tr_n,
                "val_neg": va_n,
                "test_neg": neg_splits[j][1],
            })
    else:
        groups = gene_kfold_groups(num_nodes, k, rng)
        vrng = np.random.RandomState(seed + 7919)  # separate stream
        for j in range(k):
            held = groups[j]
            tr_p, te_p = split_by_genes(pos_idx, held, cv_type)
            tr_n, te_n = split_by_genes(neg_idx, held, cv_type)
            if val_frac > 0:
                avail = np.setdiff1d(
                    np.arange(num_nodes),
                    np.fromiter(held, dtype=np.int64, count=len(held)))
                n_v = max(1, int(round(len(avail) * val_frac)))
                vheld = set(vrng.permutation(avail)[:n_v].tolist())
                tr_p, va_p = split_by_genes(tr_p, vheld, cv_type)
                tr_n, va_n = split_by_genes(tr_n, vheld, cv_type)
            else:
                va_p, va_n = tr_p[:0].copy(), tr_n[:0].copy()
            folds.append({
                "train_pos": tr_p,
                "val_pos": va_p,
                "test_pos": te_p,
                "train_neg": tr_n,
                "val_neg": va_n,
                "test_neg": te_n
            })
    return folds


def split_digest(folds):
    """Canonical fingerprint of the actual train/val/test PARTITION.

    fold_stamp hashes genes.txt, all_pos.npy and meta.json — none of which
    describe how the pairs were split. Two fold sets built from the same genes
    and the same positives but a different partition therefore produced an
    IDENTICAL stamp, and verify_fold_stamp accepted a prediction matrix trained
    on the other one. This closes that: the digest goes into meta.json, which is
    already stamped, so every existing consumer picks it up for free.

    Hashes the sorted unique pair set per partition, so it is invariant to row
    order within an array but sensitive to a single pair moving between train
    and test.
    """
    h = hashlib.sha256()
    for j, fold in enumerate(folds):
        for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg",
                    "test_neg"):
            arr = fold[key]
            h.update(f"{j}/{key}/{len(arr)}|".encode())
            if len(arr):
                canon = np.unique(np.sort(arr.astype(np.int64), axis=1),
                                  axis=0)
                h.update(canon.tobytes())
    return h.hexdigest()[:16]


def verify_fold(cv_type, fold):
    """Assert leak-free gene-disjointness for cv2/cv3; basic non-emptiness.

    Validation is held to the SAME standard as test: for cv2/cv3 it must be
    cold-start w.r.t. train, otherwise selecting on it leaks.
    """
    for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg",
                "test_neg"):
        assert fold[key].ndim == 2 and fold[key].shape[1] == 2, key
    # The docstring has always promised "basic non-emptiness" and nothing
    # enforced it: an entirely empty fold passed --verify and only surfaced
    # later as an unexplainable metric. val_* may legitimately be empty when
    # val_frac == 0, so it is excluded here.
    for key in ("train_pos", "test_pos", "train_neg", "test_neg"):
        assert len(fold[key]) > 0, \
            f"{cv_type}: {key} is empty — the split produced no {key} pairs"
    # No pair may appear in two partitions, in any CV type.
    seen = {}
    for part in ("train", "val", "test"):
        for cls in ("pos", "neg"):
            for e in map(tuple, fold[f"{part}_{cls}"]):
                prev = seen.get(e)
                assert prev is None, f"pair {e} in both {prev} and {part}_{cls}"
                seen[e] = f"{part}_{cls}"
    if cv_type == "cv1":
        return
    train_genes = set(fold["train_pos"].ravel().tolist()) | \
        set(fold["train_neg"].ravel().tolist())
    for part in ("test", "val"):
        held = np.vstack([fold[f"{part}_pos"], fold[f"{part}_neg"]]) \
            if len(fold[f"{part}_pos"]) or len(fold[f"{part}_neg"]) \
            else fold[f"{part}_pos"]
        if len(held) == 0:
            continue
        if cv_type == "cv3":
            # No train gene may appear in any held-out pair (both genes held).
            overlap = train_genes & set(held.ravel().tolist())
            assert not overlap, \
                f"cv3 leak: {len(overlap)} genes in both train and {part}"
        else:  # cv2: every held pair touches >=1 gene absent from train
            for pair in held:
                assert (pair[0] not in train_genes) or \
                       (pair[1] not in train_genes), \
                    f"cv2 leak: {part} pair fully inside train genes"


def main():
    ap = argparse.ArgumentParser(description="Generate shared CV folds")
    ap.add_argument("--pos_path", default="data/SL_SynLethDB_experimental.txt")
    ap.add_argument("--neg_path",
                    default="data/SL_SynLethDB_experimental_negatives.txt")
    ap.add_argument("--out_dir", default="sl_comparison/folds")
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_types", nargs="+", default=["cv1", "cv2", "cv3"])
    ap.add_argument("--verify",
                    action="store_true",
                    help="Run leak-free assertions on every generated fold")
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of TRAIN held out for model selection. "
        "cv1 splits pairs; cv2/cv3 hold out a val_frac-sized "
        "random slice of the NON-TEST genes, so val is cold-start "
        "in the same sense as test. Note this makes val a fraction "
        "of GENES, not of pairs: 0.1 yields ~10%% of pairs on cv1 "
        "but ~18%% on cv2 and ~1%% on cv3, because cv3 keeps only "
        "pairs with BOTH genes held. 0 disables (legacy).")
    ap.add_argument(
        "--conflict_policy",
        choices=["or_collapse", "cell_aware", "drop", "positive"],
        default="or_collapse",
        help="Pairs present in BOTH the SL and non-SL files "
        "(5,542 of them). 'or_collapse' (default) labels every one "
        "of them POSITIVE — SL if screened SL in ANY cell line — "
        "which is exactly the label siamese's *_any OR-collapse is "
        "graded on, so the shared binary relation and the cell-line "
        "model agree; each pair's lines and heads are written to "
        "conflicts.tsv (a side naming no usable line is attributed "
        "to OTHER). 'cell_aware' additionally DROPS the 1,782 "
        "same-cell-line contradictions; 'drop' removes all 5,542; "
        "'positive' is a deprecated alias for 'or_collapse'. Every "
        "policy removes them from the negative set.")
    ap.add_argument(
        "--max_genes",
        type=int,
        default=None,
        help="Subsample to the top-N most positive-connected genes "
        "(for a fast preflight smoke test only)")
    ap.add_argument(
        "--print_defaults",
        action="store_true",
        help="Emit this script's fold-defining defaults as JSON and exit. "
        "run_shared_models.sh uses it to decide whether an existing "
        "sl_comparison/folds is stale, so the staleness check can never drift "
        "from the generator (comparing only the seed silently reused folds "
        "built under a different --conflict_policy).")
    args = ap.parse_args()

    if args.print_defaults:
        print(
            json.dumps({
                "num_folds": args.num_folds,
                "val_frac": args.val_frac,
                "conflict_policy": args.conflict_policy,
                "pos_path": args.pos_path,
                "neg_path": args.neg_path,
                # A real benchmark run never subsamples; a fold set that did is
                # stale by definition.
                "max_genes": None,
            }))
        return

    pos_pairs = load_pairs(args.pos_path)
    neg_pairs = load_pairs(args.neg_path)
    if args.max_genes:
        pos_pairs, neg_pairs = subsample_genes(pos_pairs, neg_pairs,
                                               args.max_genes)
        print(f"[preflight] subsampled to top {args.max_genes} genes -> "
              f"{len(pos_pairs)} pos / {len(neg_pairs)} neg pairs")
    gene_order, g2i = build_universe(pos_pairs, neg_pairs)
    num_nodes = len(gene_order)

    pos_idx, pos_set = to_index_pairs(pos_pairs, g2i)
    neg_idx, neg_set = to_index_pairs(neg_pairs, g2i)

    # SL / non-SL conflicts: pairs screened BOTH ways, 5,542 of them.
    #
    # The shared folds carry ONE unlabelled SL relation, so the only question a
    # binary label can answer is "is this pair SL somewhere". OR-collapse says
    # yes: a screen that detected lethality is positive evidence, while a screen
    # that did not is a failure to detect in one context. That is the same rule
    # siamese's *_any headline is graded on, so both pipelines now score the
    # same ground truth — earlier policies gave them different labels on
    # thousands of pairs, which is not a difference in models.
    #
    # The cell-line attribution behind each decision goes to conflicts.tsv;
    # only the cell-line-conditioned model can act on it, but the shared label
    # is derived from it and must be auditable.
    policy = args.conflict_policy
    if policy == "positive":
        print("NOTE: --conflict_policy positive is a deprecated alias for "
              "or_collapse (identical behaviour); recording 'or_collapse'.")
        policy = "or_collapse"
    conflicts = pos_set & neg_set
    # None, not 0: under --conflict_policy drop the attribution below never
    # runs, and writing 0 into meta.json made "we did not measure this" read as
    # "we measured zero conflicts", which is a different and false claim.
    n_same = n_cross = n_same_head = n_missing_line = None
    detail_by_key = {}
    kept_conflicts = set()
    if conflicts:
        # Conflicted pairs always leave the negative set — whatever else is
        # true, they are not confidently non-SL.
        keep_n = np.array(
            [tuple(e) not in conflicts for e in map(tuple, neg_idx)])
        neg_idx = neg_idx[keep_n]

        def key(e):
            a, b = gene_order[e[0]], gene_order[e[1]]
            return (a, b) if a < b else (b, a)

        if policy in ("or_collapse", "cell_aware"):
            detail_by_key = conflict_cell_lines(args.pos_path, args.neg_path,
                                                {key(e)
                                                 for e in conflicts}, policy)
            n_same = sum(d["same_line"] for d in detail_by_key.values())
            n_cross = len(detail_by_key) - n_same
            n_same_head = sum(d["same_head"] for d in detail_by_key.values())
            n_missing_line = sum(
                MISSING_LINE in d["sl_lines"] or MISSING_LINE in d["ns_lines"]
                for d in detail_by_key.values())

        if policy == "or_collapse":
            kept_conflicts = {tuple(e) for e in conflicts}
        elif policy == "cell_aware":
            # Legacy: only a SAME-cell-line disagreement counts as a real
            # contradiction and is deleted.
            drop_p = {
                e
                for e in conflicts if detail_by_key[key(e)]["same_line"]
            }
            kept_conflicts = {tuple(e) for e in conflicts} - \
                {tuple(e) for e in drop_p}
            keep_p = np.array(
                [tuple(e) not in drop_p for e in map(tuple, pos_idx)])
            pos_idx = pos_idx[keep_p]
        elif policy == "drop":
            keep_p = np.array(
                [tuple(e) not in conflicts for e in map(tuple, pos_idx)])
            pos_idx = pos_idx[keep_p]
    detail = ""
    if detail_by_key:
        detail = (f" [{n_same} same-line, {n_cross} cross-line; "
                  f"{n_same_head} collide in one head; "
                  f"{n_missing_line} name no line on a side -> OTHER; "
                  f"{len(kept_conflicts)} kept positive]")
    print(f"Universe: {num_nodes} genes | positives: {len(pos_idx)} | "
          f"negatives: {len(neg_idx)} | {len(conflicts)} pos/neg conflicts "
          f"({policy}){detail}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if detail_by_key:
        write_conflict_report(out_root / "conflicts.tsv", gene_order,
                              conflicts, detail_by_key, kept_conflicts)
    with open(out_root / "genes.txt", "w") as f:
        f.write("\n".join(gene_order) + "\n")

    # EVERY known SL pair, regardless of fold. The retrieval metrics rank each
    # test gene against the whole universe, so any known positive that is not
    # this fold's answer key must be MASKED OUT — otherwise a model is punished
    # for correctly ranking a genuine SL pair, and the punishment scales with
    # how many known positives the fold happens to withhold. Masking only
    # train_pos left 1,631 such pairs unmasked per cv1 fold but 7,640-11,036 per
    # cv3 fold (against only 409-1,433 cv3 test positives), so it depressed cv3
    # ~10x harder than cv1 for reasons unrelated to cold-start difficulty.
    np.save(out_root / "all_pos.npy", pos_idx.astype(np.int32))

    meta = {
        "seed":
        args.seed,
        "num_folds":
        args.num_folds,
        "num_nodes":
        num_nodes,
        "num_positives":
        int(len(pos_idx)),
        "num_negatives":
        int(len(neg_idx)),
        "cv_types":
        args.cv_types,
        "pos_path":
        args.pos_path,
        "neg_path":
        args.neg_path,
        "val_frac":
        args.val_frac,
        "conflict_policy":
        policy,
        # The single binary relation these folds define, stated in words so a
        # results table can quote it without re-deriving it from a policy name.
        "label_rule":
        "A pair is POSITIVE if any screen called it SL in any cell line "
        "(OR-collapse over cell lines); NEGATIVE only if every screen of it "
        "called it non-SL. Pairs screened both ways are positive, and are "
        "listed with their per-side cell lines and heads in conflicts.tsv."
        if policy == "or_collapse" else
        f"conflict_policy={policy}; see conflicts.tsv where present.",
        "num_conflicts":
        int(len(conflicts)),
        "num_conflicts_kept_positive":
        int(len(kept_conflicts)),
        "num_conflicts_same_line": (None if n_same is None else int(n_same)),
        "num_conflicts_cross_line":
        (None if n_cross is None else int(n_cross)),
        # Same-HEAD is coarser than same-LINE: the 8 heads collapse every rare
        # line into OTHER, so pairs that disagree across two rare lines collide
        # there. This is the count the cell-line model's OR rule acts on.
        "num_conflicts_same_head":
        (None if n_same_head is None else int(n_same_head)),
        "num_conflicts_missing_cell_line":
        (None if n_missing_line is None else int(n_missing_line)),
        # PREFLIGHT MARKER. --max_genes builds a tiny smoke-test fold set. Left
        # in sl_comparison/folds it is indistinguishable from the real thing to
        # any check that looks only at seed/policy, and a whole benchmark would
        # silently run on 300 genes. Recorded so run_shared_models.sh can treat
        # any subsampled fold set as stale. (This is not hypothetical: an audit
        # agent overwrote the real fold set with a --max_genes 300 run.)
        "max_genes":
        args.max_genes,
    }

    for cv in args.cv_types:
        folds = generate(pos_idx,
                         neg_idx,
                         num_nodes,
                         cv,
                         args.num_folds,
                         args.seed,
                         val_frac=args.val_frac)
        # Fingerprint the PARTITION itself. meta.json is already one of
        # model_params.STAMP_FILES, so this reaches fold_stamp,
        # compare_fold_stamp and evaluate_model's provenance guard with no
        # further plumbing — and a fold set that differs only in how the pairs
        # were split now produces a different stamp instead of an identical one.
        meta.setdefault("split_digest", {})[cv] = split_digest(folds)
        for j, fold in enumerate(folds):
            if args.verify:
                verify_fold(cv, fold)
            fdir = out_root / cv / f"fold_{j}"
            fdir.mkdir(parents=True, exist_ok=True)
            for key, arr in fold.items():
                np.save(fdir / f"{key}.npy", arr.astype(np.int32))
        sizes = [(len(f["train_pos"]), len(f["val_pos"]), len(f["test_pos"]),
                  len(f["train_neg"]), len(f["val_neg"]), len(f["test_neg"]))
                 for f in folds]
        print(f"  {cv}: " +
              " | ".join(f"fold{j}: trP{a} vaP{b} teP{c} trN{d} vaN{e} teN{f}"
                         for j, (a, b, c, d, e, f) in enumerate(sizes)))

    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    extra = ", conflicts.tsv" if detail_by_key else ""
    print(f"Folds written to {out_root}/  (genes.txt, meta.json{extra}, "
          f"<cv>/fold_<k>/*.npy)")


if __name__ == "__main__":
    main()
