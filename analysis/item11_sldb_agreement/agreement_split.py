#!/usr/bin/env python3
"""
Plan item 11 (ICLR 2027 revision, Submission45596): agreement between the
SynLethDB SL label and each external screen's phenotype, split into
cell-line-specific SL (reported in the screen's own cell line) and
general / other-context SL (reported only in other cell lines), with
pair-resampling bootstrap CIs and bootstrap tests of the difference.

Mirrors the published computation exactly:
  * rows / target / label: sl_comparison/external_transfer.py sldb_tables()
    (scopes all_pairs, all_pairs_genes_txt, test_genes_txt per screen split;
    label 1 = SynLethDB SL in any cell line, 0 = screened non-SL and not SL:
    the shared folds' OR-collapse; target = -1 x screen phenotype, higher =
    more SL-like; AUROC of the target for SL vs non-SL, Spearman(label,
    target));
  * resampling: sl_comparison/bootstrap_ci.py bootstrap_sldb_agreement()
    (the labelled pairs of each scope, sorted by key, resampled with
    replacement, B = 1000, resample b drawn from
    default_rng([42, crc32(screen|scope|split|sldb_agreement|digest), b])).
The same resamples are used for every subset, so (i) subset 'all' reproduces
bootstrap/ci_sldb_agreement.csv exactly (checked below) and (ii) differences
between subsets are paired (the negatives are shared).

Subsets of the SL pairs (the negatives are always ALL screened non-SL pairs
of the scope, as published):
  context (primary)  own_line      SL cell lines include the screen's line
                     other_line    >= 1 usable line, none is the screen's
                     unknown_line  no usable cell line (TBD, LAML, '', ...)
                     not_own_line  other_line + unknown_line
  conflict (sens.)   other_line_uncontradicted / _contradicted: other_line
                     SL pairs NOT / ALSO screened non-SL in the screen's line
  breadth (second.)  multi_line (>= 2 usable lines), single_line (exactly 1)
  source (diagn.)    src_horlbeck2018 (PMID 30033366, the Gilbert screen's
                     paper), src_gemini2019 (PMID 31300006), src_other
  own-line label     ownline_label: positives = SL in the screen's line,
  (sens.)            negatives = screened non-SL in the screen's line and
                     not SL there (the per-line label; for K562 = the K562
                     head label of cell_line_vocab.pair_label_mask)
Cell lines: data/SL_SynLethDB_experimental[_negatives].sources.tsv column
cell_lines, canonicalised with siamese_sl/cell_line_vocab.parse_cell_field.
Screen lines: Adamson / AdamsonGI / Gilbert = K562, Corn = RPE-1 (RPE1)
(siamese_sl/slurm/eval_finetuning.conf DATASETS comment; paper/main.tex).

Read-only on the repo (imports with -B: no bytecode written). Run:
  python -B agreement_split.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "sl_comparison"))
import bootstrap_ci as bc  # noqa: E402
import external_transfer as et  # noqa: E402


def _load_by_path(name: str, path: Path):
    """Import one module by file path (siamese_sl/ is NOT put on sys.path:
    its retrieval_metrics.py would shadow sl_comparison's and needs torch)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clv = _load_by_path("cell_line_vocab", REPO / "siamese_sl/cell_line_vocab.py")

N_BOOT, SEED, ALPHA = 1000, 42, 0.05
MIN_POS = 10  # report a subset's metric only with >= this many SL pairs
CMP = REPO / "sl_comparison" / "results"            # arm none
EXT = CMP / "external_transfer"
BOOT = CMP / "bootstrap"
FOLDS = REPO / "sl_comparison" / "folds"           # read only
OWN_LINE = {"Adamson": "K562", "AdamsonGI": "K562", "Gilbert": "K562",
            "Corn": "RPE1"}
PMID_HORLBECK, PMID_GEMINI = "30033366", "31300006"

SUBSETS = {
    # name: (definition, description)
    "all": ("published", "every SynLethDB SL pair (published row)"),
    "own_line": ("context", "SL reported in the screen's own cell line"),
    "other_line": ("context", "SL reported only in other cell line(s)"),
    "unknown_line": ("context", "SL with no usable cell line"),
    "not_own_line": ("context", "other_line + unknown_line"),
    "other_line_uncontradicted": ("conflict", "other_line, not screened "
                                  "non-SL in the screen's line"),
    "other_line_contradicted": ("conflict", "other_line, also screened "
                                "non-SL in the screen's line"),
    "multi_line": ("breadth", "SL in >= 2 usable cell lines"),
    "single_line": ("breadth", "SL in exactly 1 usable cell line"),
    "src_horlbeck2018": ("source", "SL evidence incl. PMID 30033366"),
    "src_gemini2019": ("source", "SL evidence PMID 31300006 (not 30033366)"),
    "src_other": ("source", "neither PMID (incl. no PMID)"),
    "ownline_label": ("ownline_label", "own-line label: SL in the line vs "
                      "screened non-SL in the line and not SL there"),
}
CONTRASTS = [
    # (A, B, role)
    ("own_line", "other_line", "primary"),
    ("own_line", "not_own_line", "sensitivity"),
    ("own_line", "other_line_uncontradicted", "sensitivity"),
    ("multi_line", "single_line", "secondary"),
    ("ownline_label", "all", "sensitivity"),
    ("src_horlbeck2018", "src_gemini2019", "diagnostic"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def p_two_sided(p_le0: float, b: int = N_BOOT) -> float:
    """The revision's rule: two-sided p from a one-sided bootstrap fraction."""
    if not np.isfinite(p_le0):
        return float("nan")
    return float(min(1.0, 2.0 * max(min(p_le0, 1.0 - p_le0), 1.0 / (b + 1))))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * q[i]))
        adj[i] = run
    out[ok] = adj
    return out


def auroc_from_inc(ip: np.ndarray, ineg: np.ndarray) -> np.ndarray:
    """AUROC per row from per-segment (descending score) counts: the
    trapezoid of bootstrap_ci.metrics_from_counts (= Mann-Whitney with half
    credit for ties), vectorised over resamples. Empty segments add 0."""
    ip = ip.astype(np.float64)
    ineg = ineg.astype(np.float64)
    cp = np.cumsum(ip, axis=1)
    P = cp[:, -1]
    N = ineg.sum(axis=1)
    tp0 = cp - ip
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (ineg * (cp + tp0)).sum(axis=1) / (2.0 * P * N)
    a[(P <= 0) | (N <= 0)] = np.nan
    return a


def auroc_boot(y: np.ndarray, s: np.ndarray, W: np.ndarray, chunk: int = 64):
    seg = bc.DescSegments(s)
    yi = y.astype(np.int64)
    ones = np.ones((1, len(y)), dtype=np.uint16)
    pt = float(auroc_from_inc(*bc.segment_increments(ones, yi, seg))[0])
    reps = np.full(W.shape[0], np.nan)
    for c0 in range(0, W.shape[0], chunk):
        reps[c0:c0 + chunk] = auroc_from_inc(
            *bc.segment_increments(W[c0:c0 + chunk], yi, seg))
    return pt, reps


def spearman_boot(y: np.ndarray, t: np.ndarray, W: np.ndarray):
    pts, reps = bc.bootstrap_spearman(t, {"label": y.astype(np.float64)}, W)
    return pts["label"], reps["label"]


def stats(pt, reps):
    d = bc.boot_stats(pt, reps, ALPHA)
    return d


def read_sidecar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["key"] = et.pair_keys(df["entrez1"].str.strip(),
                             df["entrez2"].str.strip())
    if df["key"].duplicated().any():
        raise SystemExit(f"{path}: duplicate pair keys")
    return df.set_index("key")


def main() -> int:
    t_start = time.time()
    checks = {}
    # ---------------- provenance of the inputs ----------------
    meta = json.loads((FOLDS / "meta.json").read_text())
    files = {
        "pos": REPO / meta["pos_path"],
        "neg": REPO / meta["neg_path"],
        "pos_sources": REPO / "data/SL_SynLethDB_experimental.sources.tsv",
        "neg_sources": REPO /
        "data/SL_SynLethDB_experimental_negatives.sources.tsv",
        "cell_line_vocab": REPO / "siamese_sl/cell_line_vocab.py",
    }
    for k, p in files.items():
        got = sha256(p)
        want = meta["source_sha256"][k]
        checks[f"sha256 {k} == folds/meta.json"] = got == want
        if got != want:
            raise SystemExit(f"{p}: sha256 {got} != meta.json {want}")

    prov = bc.parse_external_provenance(EXT / "summary.md")
    sia_name, sia_dir = prov["siamese"][0]
    sia_dir = bc.repo_path(sia_dir)
    checks["rows defined by (first SLeuth variant, as published)"] = \
        f"{sia_name} = {sia_dir.relative_to(REPO)}"
    zs_pub = pd.read_csv(EXT / "zero_shot_per_fold.csv")
    screens = list(dict.fromkeys(zs_pub["screen"].astype(str)))
    splits = list(dict.fromkeys(zs_pub["split"].astype(str)))
    agree_pub = pd.read_csv(EXT / "sldb_agreement.csv")
    ci_pub = pd.read_csv(BOOT / "ci_sldb_agreement.csv")
    gene_index = et.load_gene_index(FOLDS / "genes.txt")
    pos_keys = et.load_pair_keys(files["pos"])
    neg_keys = et.load_pair_keys(files["neg"])

    pos_sc = read_sidecar(files["pos_sources"])
    neg_sc = read_sidecar(files["neg_sources"])
    if set(pos_sc.index) != pos_keys or set(neg_sc.index) != neg_keys:
        raise SystemExit("sidecar keys differ from the pair files")
    sl_lines = {k: clv.parse_cell_field(v)
                for k, v in pos_sc["cell_lines"].items()}
    ns_lines = {k: clv.parse_cell_field(v)
                for k, v in neg_sc["cell_lines"].items()}
    sl_pmids = {k: {x.strip() for x in v.split(";") if x.strip()}
                for k, v in pos_sc["pubmed_ids"].items()}

    point_rows, diff_rows, count_rows, val_rows = [], [], [], []
    reps_store = {}
    for screen in screens:
        own = OWN_LINE[screen]
        pairs, reference = None, {}
        for split in splits:
            df = et.read_pair_csv(sia_dir / f"eval_finetuning_{split}" /
                                  f"{screen}_predictions.csv")
            if pairs is None:
                pairs = df
            m = et.row_mask(df, gene_index, et.ROW_MATCHED_INCL)
            reference[(screen, split)] = df.loc[m].sort_values(
                "key").reset_index(drop=True)
        for scope, sp, sub in et.screen_scopes(screen, pairs, reference,
                                               gene_index, splits):
            t0 = time.time()
            if sub["key"].duplicated().any():
                raise SystemExit(f"{screen}/{scope}/{sp}: duplicate keys")
            flagged = sub["sldb_overlap"].to_numpy(dtype=bool)
            allk = sub["key"].to_numpy()
            in_pos = np.fromiter((k in pos_keys for k in allk), bool,
                                 len(allk))
            in_neg = np.fromiter((k in neg_keys for k in allk), bool,
                                 len(allk)) & ~in_pos
            n_bad = int(((in_pos | in_neg) != flagged).sum())
            checks[f"flag == pos|neg  {screen}/{scope}/{sp}"] = n_bad == 0
            if n_bad:
                raise SystemExit(f"{screen}/{scope}/{sp}: {n_bad} flag "
                                 f"mismatches")
            lab = sub.loc[flagged].sort_values("key").reset_index(drop=True)
            keys = lab["key"].to_numpy()
            y = np.fromiter((k in pos_keys for k in keys), bool, len(keys))
            t = lab["target"].to_numpy(dtype=np.float64)
            et.require_finite(t, f"{screen}/{scope}/{sp} target")
            n = len(keys)
            W = bc.resample_counts(
                n, N_BOOT, SEED,
                (bc.unit_hash(screen, scope, sp, "sldb_agreement",
                              et.matched_digest(keys)),))

            # ---- per-pair annotation ----
            L = [sl_lines.get(k, set()) if yy else set()
                 for k, yy in zip(keys, y)]
            NL = [ns_lines.get(k, set()) for k in keys]
            nlines = np.array([len(s) for s in L])
            own_sl = np.array([own in s for s in L]) & y
            other_sl = y & (nlines > 0) & ~own_sl
            unk_sl = y & (nlines == 0)
            contra = np.array([own in s for s in NL])
            pm = [sl_pmids.get(k, set()) if yy else set()
                  for k, yy in zip(keys, y)]
            horl = y & np.array([PMID_HORLBECK in s for s in pm])
            gem = y & np.array([PMID_GEMINI in s for s in pm]) & ~horl
            srco = y & ~horl & ~gem
            neg = ~y
            masks_pos = {
                "all": y,
                "own_line": own_sl,
                "other_line": other_sl,
                "unknown_line": unk_sl,
                "not_own_line": y & ~own_sl,
                "other_line_uncontradicted": other_sl & ~contra,
                "other_line_contradicted": other_sl & contra,
                "multi_line": y & (nlines >= 2),
                "single_line": y & (nlines == 1),
                "src_horlbeck2018": horl,
                "src_gemini2019": gem,
                "src_other": srco,
            }
            assert (own_sl | other_sl | unk_sl).sum() == y.sum()
            assert not (own_sl & other_sl).any()
            assert not (own_sl & unk_sl).any()
            # own-line label: positives SL in `own`; negatives screened
            # non-SL in `own` and not SL there (incl. SL-elsewhere pairs)
            ol_pos = own_sl
            ol_neg = contra & ~own_sl
            count_rows.append({
                "screen": screen, "scope": scope, "split": sp,
                "own_line": own, "n_labelled": n, "n_sl": int(y.sum()),
                "n_nonsl": int(neg.sum()),
                **{f"n_{k}": int(v.sum()) for k, v in masks_pos.items()
                   if k != "all"},
                "n_nonsl_screened_in_own_line": int((neg & contra).sum()),
                "n_ownline_label_pos": int(ol_pos.sum()),
                "n_ownline_label_neg": int(ol_neg.sum()),
            })

            # ---- metrics per subset (same resamples W) ----
            for sname, (defin, _desc) in SUBSETS.items():
                if sname == "ownline_label":
                    cols = ol_pos | ol_neg
                    yy = ol_pos[cols]
                else:
                    cols = masks_pos[sname] | neg
                    yy = y[cols]
                n_p, n_n = int(yy.sum()), int((~yy).sum())
                base = {"screen": screen, "own_line": own, "scope": scope,
                        "split": sp, "definition": defin, "subset": sname,
                        "n_pos": n_p, "n_neg": n_n,
                        "reportable": bool(n_p >= MIN_POS and n_n >= MIN_POS)}
                if n_p == 0 or n_n == 0:
                    point_rows.append({**base, "auroc": np.nan,
                                       "spearman": np.nan})
                    continue
                Wc = W[:, cols]
                tc = t[cols]
                a_pt, a_rep = auroc_boot(yy, tc, Wc)
                s_pt, s_rep = spearman_boot(yy, tc, Wc)
                reps_store[(screen, scope, sp, sname)] = (a_pt, a_rep,
                                                          s_pt, s_rep)
                sa, ss = stats(a_pt, a_rep), stats(s_pt, s_rep)
                pa = float(np.mean(a_rep[np.isfinite(a_rep)] - 0.5 <= 0))
                ps = float(np.mean(s_rep[np.isfinite(s_rep)] <= 0))
                pi = n_p / (n_p + n_n)
                point_rows.append({
                    **base,
                    "auroc": a_pt, "auroc_ci_low": sa["ci_low"],
                    "auroc_ci_high": sa["ci_high"],
                    "auroc_boot_se": sa["boot_se"],
                    "auroc_n_finite": sa["n_finite"],
                    "r_rankbiserial": 2 * a_pt - 1,
                    "r_rb_ci_low": 2 * sa["ci_low"] - 1,
                    "r_rb_ci_high": 2 * sa["ci_high"] - 1,
                    "d_binormal": float(np.sqrt(2) * norm.ppf(a_pt)),
                    "d_ci_low": float(np.sqrt(2) * norm.ppf(sa["ci_low"])),
                    "d_ci_high": float(np.sqrt(2) * norm.ppf(sa["ci_high"])),
                    "p_boot_auroc_le_half": pa,
                    "p_two_sided_vs_chance": p_two_sided(pa),
                    "spearman": s_pt, "spearman_ci_low": ss["ci_low"],
                    "spearman_ci_high": ss["ci_high"],
                    "spearman_boot_se": ss["boot_se"],
                    "p_boot_spearman_le0": ps,
                    "spearman_no_ties_approx": float(
                        2 * np.sqrt(3 * pi * (1 - pi)) * (a_pt - 0.5)),
                    "prevalence": pi,
                })
                # ---- validation against the published row (subset all) --
                if sname == "all":
                    cp, cr = bc.bootstrap_classification(
                        yy.astype(np.int64), {"target": tc}, Wc)
                    v = {"screen": screen, "scope": scope, "split": sp,
                         "auroc_point_vs_bootstrap_ci_impl": abs(
                             a_pt - cp["target"]["auroc"]),
                         "auroc_reps_max_absdiff_vs_impl": float(np.nanmax(
                             np.abs(a_rep - cr["target"]["auroc"])))}
                    hit = agree_pub[(agree_pub.screen == screen) &
                                    (agree_pub.scope == scope) &
                                    (agree_pub.split.astype(str) == sp)]
                    v["n_labelled_pub"] = int(hit["n_labelled"].iloc[0])
                    v["n_pos_pub"] = int(hit["n_pos"].iloc[0])
                    v["auroc_vs_sldb_agreement.csv"] = abs(
                        a_pt - float(hit["auroc_target_vs_label"].iloc[0]))
                    v["spearman_vs_sldb_agreement.csv"] = abs(
                        s_pt - float(hit["spearman_target_vs_label"].iloc[0]))
                    for met, (pt_, st_) in {
                            "auroc_target_vs_label": (a_pt, sa),
                            "spearman_target_vs_label": (s_pt, ss)}.items():
                        h = ci_pub[(ci_pub.screen == screen) &
                                   (ci_pub.scope == scope) &
                                   (ci_pub.split.astype(str) == sp) &
                                   (ci_pub.metric == met)]
                        for c in ("ci_low", "ci_high", "boot_mean",
                                  "boot_se"):
                            v[f"{met}:{c} absdiff vs ci_sldb_agreement.csv"] \
                                = abs(st_[c] - float(h[c].iloc[0]))
                    v["n_ok"] = (v["n_labelled_pub"] == n and
                                 v["n_pos_pub"] == int(y.sum()))
                    val_rows.append(v)

            # ---- contrasts (paired: same resamples) ----
            for A, B, role in CONTRASTS:
                ra = reps_store.get((screen, scope, sp, A))
                rb = reps_store.get((screen, scope, sp, B))
                rowA = [r for r in point_rows if r["screen"] == screen and
                        r["scope"] == scope and r["split"] == sp and
                        r["subset"] == A][0]
                rowB = [r for r in point_rows if r["screen"] == screen and
                        r["scope"] == scope and r["split"] == sp and
                        r["subset"] == B][0]
                rep_ok = bool(rowA["reportable"] and rowB["reportable"])
                for met, idx in (("auroc", 0), ("spearman", 2)):
                    rec = {"screen": screen, "scope": scope, "split": sp,
                           "contrast": f"{A} - {B}", "A": A, "B": B,
                           "role": role, "metric": met,
                           "n_pos_A": rowA["n_pos"], "n_pos_B": rowB["n_pos"],
                           "n_neg_A": rowA["n_neg"], "n_neg_B": rowB["n_neg"],
                           "reportable": rep_ok}
                    if ra is None or rb is None:
                        diff_rows.append({**rec, "point": np.nan})
                        continue
                    d_pt = ra[idx] - rb[idx]
                    d_rep = ra[idx + 1] - rb[idx + 1]
                    ds = bc.diff_stats(d_pt, d_rep, ALPHA)
                    diff_rows.append({**rec, "A_point": ra[idx],
                                      "B_point": rb[idx], **ds,
                                      "p_two_sided": p_two_sided(
                                          ds["p_boot_le0"])})
                # direct AUROC of the target, A's SL pairs vs B's SL pairs
                if role in ("primary", "secondary", "diagnostic"):
                    ma, mb = masks_pos[A], masks_pos[B]
                    cols = ma | mb
                    rec = {"screen": screen, "scope": scope, "split": sp,
                           "contrast": f"{A} vs {B} (direct)", "A": A,
                           "B": B, "role": role, "metric": "auroc_direct",
                           "n_pos_A": int(ma.sum()), "n_pos_B": int(mb.sum()),
                           "reportable": bool(ma.sum() >= MIN_POS and
                                              mb.sum() >= MIN_POS)}
                    if ma.any() and mb.any():
                        a_pt, a_rep = auroc_boot(ma[cols], t[cols],
                                                 W[:, cols])
                        ds = bc.diff_stats(a_pt - 0.5, a_rep - 0.5, ALPHA)
                        rec.update({"A_point": a_pt, "point": a_pt,
                                    "ci_low": ds["ci_low"] + 0.5,
                                    "ci_high": ds["ci_high"] + 0.5,
                                    "boot_se": ds["boot_se"],
                                    "n_finite": ds["n_finite"],
                                    "p_boot_le0": ds["p_boot_le0"],
                                    "p_two_sided": p_two_sided(
                                        ds["p_boot_le0"])})
                    else:
                        rec["point"] = np.nan
                    diff_rows.append(rec)
            print(f"{screen:10s} {scope:20s} {sp:4s} n={n:6d} "
                  f"SL={int(y.sum()):5d} [{time.time() - t0:5.1f}s]",
                  flush=True)

    pts = pd.DataFrame(point_rows)
    dfs = pd.DataFrame(diff_rows)
    cnt = pd.DataFrame(count_rows)
    val = pd.DataFrame(val_rows)

    # ---------------- Holm families (scope all_pairs only) ----------------
    dfs["holm_family"] = ""
    dfs["p_holm"] = np.nan
    fam_defs = {
        "F1_ctx_auroc": ("own_line - other_line", "auroc"),
        "F2_ctx_spearman": ("own_line - other_line", "spearman"),
        "F3_breadth_auroc": ("multi_line - single_line", "auroc"),
        "F4_breadth_spearman": ("multi_line - single_line", "spearman"),
        "S1_notown_auroc": ("own_line - not_own_line", "auroc"),
        "S2_uncontra_auroc": ("own_line - other_line_uncontradicted",
                              "auroc"),
        "S3_ownlabel_auroc": ("ownline_label - all", "auroc"),
    }
    for fam, (con, met) in fam_defs.items():
        sel = ((dfs.scope == "all_pairs") & (dfs.contrast == con) &
               (dfs.metric == met) & dfs.reportable.astype(bool) &
               np.isfinite(dfs.get("p_two_sided", np.nan)))
        if sel.any():
            dfs.loc[sel, "holm_family"] = fam
            dfs.loc[sel, "p_holm"] = holm(dfs.loc[sel, "p_two_sided"])
    pts["holm_family"] = ""
    pts["p_holm_vs_chance"] = np.nan
    sel = ((pts.scope == "all_pairs") & pts.reportable.astype(bool) &
           pts.subset.isin(["all", "own_line", "other_line", "unknown_line"])
           & np.isfinite(pts.get("p_two_sided_vs_chance", np.nan)))
    pts.loc[sel, "holm_family"] = "F5_vs_chance_context_subsets"
    pts.loc[sel, "p_holm_vs_chance"] = holm(
        pts.loc[sel, "p_two_sided_vs_chance"])

    # Spearman-vs-chance and AUROC-vs-chance agree in sign per resample
    both = pts.dropna(subset=["p_boot_auroc_le_half"])
    checks["sign(Spearman) == sign(AUROC-0.5) on every resample (max |p|"
           " diff)"] = float(np.max(np.abs(both["p_boot_auroc_le_half"] -
                                            both["p_boot_spearman_le0"])))

    pts.to_csv(OUT / "agreement_split_points.csv", index=False)
    dfs.to_csv(OUT / "agreement_split_diffs.csv", index=False)
    cnt.to_csv(OUT / "agreement_split_counts.csv", index=False)
    val.to_csv(OUT / "validation_published_rows.csv", index=False)
    num_cols = [c for c in val.columns if "absdiff" in c or "vs_" in c]
    checks["max abs diff of subset 'all' vs published (points + CIs)"] = \
        float(val[num_cols].to_numpy(dtype=float).max())
    checks["n_labelled / n_pos equal published"] = bool(val["n_ok"].all())
    checks["runtime_s"] = round(time.time() - t_start, 1)
    checks["versions"] = {"python": sys.version.split()[0],
                          "numpy": np.__version__, "pandas": pd.__version__}
    checks["n_boot"], checks["seed"], checks["alpha"] = N_BOOT, SEED, ALPHA
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1,
                                                default=str))
    print(json.dumps(checks, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
