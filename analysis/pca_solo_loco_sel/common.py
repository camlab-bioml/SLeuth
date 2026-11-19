"""Shared loaders and statistics for the PCA / single-modality / LOCO /
modality-selection analyses (ICLR 2027 revision, run 20261113T113339-1225758).

Read-only on the repo. Every number comes from a file named in the output.

Statistics (stated once, used everywhere):
  * fold-paired tests over the 5 splits of one tier, the difference taken
    fold by fold (by fold id for results.json, by position for the graded
    row JSONs' summary.folds, which train.py / evaluate_model.py write in
    fold order 0..4; checked where both exist);
      - paired t-test, two-sided, df = n - 1 (sample sd, ddof=1);
      - exact sign-flip permutation test, two-sided, all 2^n sign patterns
        (n = 5: 32 patterns, minimum attainable p = 2/32 = 0.0625);
  * bootstrap two-sided p from a one-sided p_boot_le0 with B resamples:
      p2 = min(1, 2 * max(min(p, 1 - p), 1 / (B + 1)))   (floor ~0.002);
  * Holm step-down correction inside each family, defined where used.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CMP = {"none": REPO / "sl_comparison/results",
       "joint": REPO / "sl_comparison/results/pca_joint"}
SIA = {"none": REPO / "siamese_sl/results",
       "joint": REPO / "siamese_sl/results/pca_joint"}
SHARED_CMP = REPO / "sl_comparison/results"   # SLMGAE/SLGNN/... rows live here
CVS = ["cv1", "cv2", "cv3"]
TIER = {"cv1": "I", "cv2": "II", "cv3": "III"}
SLEUTH_ROWS = ["siamese_sl_pd_pooled", "siamese_sl", "siamese_sl_allmod_pd_pooled",
               "siamese_sl_allmod", "siamese_sl_nocell"]
SHARED_ROWS = ["SLMGAE", "SLGNN", "NSF4SL", "ELISL", "BilinearProbe",
               "DegreePrior"]
ROW_LABEL = {
    "siamese_sl_pd_pooled": "SLeuth pooled (selected combo)",
    "siamese_sl": "SLeuth cell-line (selected combo)",
    "siamese_sl_allmod_pd_pooled": "SLeuth pooled, all 16 embeddings",
    "siamese_sl_allmod": "SLeuth cell-line, all 16 embeddings",
    "siamese_sl_nocell": "SLeuth cell-line, cell_emb zeroed",
}
PAPER16 = ["geneformer", "scgpt", "gene2vec", "esm2", "esm1b", "prot_t5",
           "text_embed", "bioconceptvec", "node2vec_ppi", "ppi_svd", "ppi_raw",
           "mashup", "go", "go2vec", "onto2vec", "kg_complex"]
CATEGORY = {"geneformer": "expression", "scgpt": "expression",
            "gene2vec": "expression", "esm2": "protein_seq",
            "esm1b": "protein_seq", "prot_t5": "protein_seq",
            "text_embed": "text", "bioconceptvec": "text",
            "node2vec_ppi": "ppi", "ppi_svd": "ppi", "ppi_raw": "ppi",
            "mashup": "ppi", "go": "go", "go2vec": "go", "onto2vec": "go",
            "kg_complex": "kg"}
CATS = ["expression", "go", "kg", "ppi", "protein_seq", "text"]


def rel(p):
    """Repo-relative path string for citations."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------- loaders
def row_path(arm, model, cv):
    base = SHARED_CMP if model in SHARED_ROWS else CMP[arm]
    return base / f"{model}_{cv}.json"


def graded_folds(arm, model, cv, metrics=("auprg", "ap", "auroc")):
    """Per-fold graded metrics (shared evaluator) of one comparison row, by
    position in summary.folds. Returns (DataFrame n x metrics, path)."""
    p = row_path(arm, model, cv)
    d = json.loads(p.read_text())["summary"]
    folds = d["folds"]
    if d.get("n_folds_scored") != 5 or d.get("partial"):
        raise RuntimeError(f"{p}: not 5 complete folds")
    df = pd.DataFrame([{m: float(f[m]) for m in metrics} for f in folds])
    df.index.name = "fold"
    return df, p


def graded_summary(arm, model, cv):
    p = row_path(arm, model, cv)
    return json.loads(p.read_text())["summary"], p


def marker_dir(arm, name):
    """The run dir a latest-combo marker names (siamese_sl-relative)."""
    m = SIA[arm] / name
    txt = m.read_text().strip()
    return REPO / "siamese_sl" / txt, m


def self_folds(run_dir, keys=("auprg_any", "auroc_any", "aupr_any",
                              "val_auprg_any", "sel_auprg_any")):
    """Per-fold self-scored metrics from a run's results.json, ordered by
    the 'fold' key train.py records."""
    p = Path(run_dir) / "results.json"
    r = json.loads(p.read_text())
    fm = sorted(r["fold_metrics"], key=lambda m: m["fold"])
    if [m["fold"] for m in fm] != list(range(5)):
        raise RuntimeError(f"{p}: folds {[m['fold'] for m in fm]}")
    rows = []
    for m in fm:
        rows.append({k: (float(m[k]) if k in m and m[k] is not None
                         else np.nan) for k in keys})
    df = pd.DataFrame(rows)
    df.index.name = "fold"
    return df, p, r


def combo_dirs(arm):
    """{(variant, cv): run dir} of the selected combos and the all-embedding
    models of one arm, from the latest_* markers."""
    out = {}
    for cv in CVS:
        out[("pd:per_head", "combo", cv)] = marker_dir(arm, f"latest_combo_{cv}.txt")[0]
        out[("pd:pooled", "combo", cv)] = marker_dir(
            arm, f"var_pd_pooled__latest_combo_{cv}.txt")[0]
        out[("pd:per_head", "allmod", cv)] = marker_dir(
            arm, f"allmod__latest_allmod_{cv}.txt")[0]
        out[("pd:pooled", "allmod", cv)] = marker_dir(
            arm, f"allmod__var_pd_pooled__latest_allmod_{cv}.txt")[0]
    return out


def solo_dir(variant, emb, cv):
    """Single-embedding run dir (shared by arms none and joint: native dim,
    no PCA). pd:pooled = the selection variant's Step-1 runs."""
    pre = "" if variant == "pd:pooled" else "var_pd_per_head__"
    return SIA["none"] / f"{pre}pcavar0.8_{emb}_{cv}"


# --------------------------------------------------------------- statistics
def signflip_p(d, tol=1e-12):
    """Exact two-sided sign-flip permutation p of mean(d) = 0."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = abs(d.sum())
    cnt = 0
    for s in itertools.product([1.0, -1.0], repeat=n):
        if abs(np.dot(s, d)) >= obs - tol:
            cnt += 1
    return cnt / 2 ** n


def paired(d):
    """Summary of fold-paired differences d (A - B)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    if se > 0:
        t = mean / se
        p_t = float(2 * stats.t.sf(abs(t), n - 1))
    else:
        t, p_t = (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    h = float(stats.t.ppf(0.975, n - 1)) * se
    return {"n": n, "mean_diff": mean, "sd_diff": sd,
            "sd_diff_pop": float(d.std(ddof=0)), "se_diff": se,
            "ci95_low": mean - h, "ci95_high": mean + h, "t": t, "p_t": p_t,
            "p_signflip": signflip_p(d), "n_pos": int((d > 0).sum()),
            "n_neg": int((d < 0).sum())}


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def p2_boot(p_le0, B=1000):
    """Two-sided bootstrap p from the one-sided fraction of resamples with
    A - B <= 0 (floor 2/(B+1))."""
    p = float(p_le0)
    return min(1.0, 2 * max(min(p, 1 - p), 1.0 / (B + 1)))


def fmt_p(p, floor=None):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if floor is not None and p <= floor + 1e-12:
        return f"<= {floor:.3f}"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def ms(mean, sd, k=3):
    return f"{mean:.{k}f} ({sd:.{k}f})"
