#!/usr/bin/env python3
"""Re-run, locally, the EXTERNAL part of sl_comparison/bootstrap_ci.py for the
score sources whose inputs exist on the laptop (the SLeuth CSVs of the two cv3
combos and DegreePrior, rebuilt from the shared cv3 folds), with the SAME
resamples (bootstrap_ci.resample_counts / unit_hash, B=1000, seed 42), and
keep the replicates so comparisons the published run did not tabulate can be
tested on those same resamples:

  * siamese_sl (per-head) - siamese_sl_pd_pooled, zero_shot_raw, matched rows
  * LP_last / Full / Scratch - zero_shot_raw (the paper's printed zero-shot
    column), all_sldb_filtered and matched rows
  * one-sided p of each SLeuth / DegreePrior correlation vs 0
  * split drops (cv1 - cv2, cv1 - cv3, cv2 - cv3) of each fine-tuning mode
    (independent resamples: different row sets)

SLMGAE / SLGNN / NSF4SL cannot be re-scored here (their dense fold matrices
are on the cluster only); their published CIs / diffs are used as published.

Every point estimate and every published CI / diff that this script can
rebuild is checked against <cmp>/bootstrap/ci_external*.csv first (atol 1e-9
on points, 1e-9 on CI ends); the script stops on a mismatch.

Usage: python -I repro_boot.py ARM   (ARM = none | joint)
Writes  out_<arm>/replicates.npz, out_<arm>/repro_check.json
Read-only on the repository (no bytecode written there).
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sl_comparison"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bootstrap_ci as bc  # noqa: E402
import external_transfer as et  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREENS = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
SPLITS = ("cv1", "cv2", "cv3")
ROW_SETS = ("matched", "all_sldb_filtered")
METHODS = ("zero_shot_raw", "zero_shot_ols", "LP_last", "Full", "Scratch")
COMBO_NAME = ("best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_"
              "ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3")
N_BOOT, SEED, ALPHA = 1000, 42, 0.05


def arm_dirs(arm):
    sia_root = REPO / "siamese_sl" / "results"
    cmp_dir = REPO / "sl_comparison" / "results"
    if arm != "none":
        sia_root = sia_root / f"pca_{arm}"
        cmp_dir = cmp_dir / f"pca_{arm}"
    return {
        "siamese_sl": sia_root / COMBO_NAME,
        "siamese_sl_pd_pooled": sia_root / f"var_pd_pooled__{COMBO_NAME}",
    }, cmp_dir


def check_provenance(siamese, cmp_dir):
    """The published run's SLeuth dirs (manifest.json) must be these."""
    man = json.load(open(cmp_dir / "bootstrap" / "manifest.json"))
    pub = man["external"]["siamese"]
    for name, d in siamese.items():
        tail = str(d).split("SLMGAE-pytorch/")[1]
        if not pub[name].endswith(tail):
            raise SystemExit(f"{name}: manifest {pub[name]} != {d}")
    if man["n_boot"] != N_BOOT or man["seed"] != SEED:
        raise SystemExit("manifest n_boot / seed differ")
    return man


def main(arm):
    t_start = time.time()
    siamese, cmp_dir = arm_dirs(arm)
    man = check_provenance(siamese, cmp_dir)
    out_dir = HERE / f"out_{arm}"
    out_dir.mkdir(exist_ok=True)
    folds_dir = REPO / "sl_comparison" / "folds"
    gene_index = et.load_gene_index(folds_dir / "genes.txt")
    num_nodes = len(gene_index)
    deg_scorers = et.degree_fold_scorers(folds_dir, "cv3", num_nodes)
    pub = pd.read_csv(cmp_dir / "bootstrap" / "ci_external.csv",
                      dtype={"fold": str})
    pubd = pd.read_csv(cmp_dir / "bootstrap" / "ci_external_diffs.csv",
                       dtype={"fold": str})
    reps_out = {}
    checks = {"n_point": 0, "n_ci": 0, "n_diff": 0, "max_abs_err": 0.0,
              "problems": []}

    def chk(got, want, what):
        if want is None or not np.isfinite(want):
            return
        err = abs(got - want)
        checks["max_abs_err"] = max(checks["max_abs_err"], float(err))
        if err > 1e-9:
            checks["problems"].append(f"{what}: got {got!r} want {want!r}")

    for screen in SCREENS:
        for split in SPLITS:
            t0 = time.time()
            units = {}
            for name, d in siamese.items():
                ed = d / f"eval_finetuning_{split}"
                df = et.read_pair_csv(ed / f"{screen}_predictions.csv")
                logit_cols = et.fold_columns(df, et._LOGIT_COL)
                target_all = df["target"].to_numpy(dtype=np.float64)
                train = (df["split"].astype(str) == "train").to_numpy()
                ols = {k: et.fit_ols_affine(df[c].to_numpy()[train],
                                            target_all[train])
                       for k, c in logit_cols}
                ft = {m: et.read_pair_csv(
                    ed / f"{screen}_finetuned_predictions_{m}.csv"
                ).set_index("key") for m in ("LP_last", "Full", "Scratch")}
                for rs in ROW_SETS:
                    mask = et.row_mask(df, gene_index, rs)
                    sub = df.loc[mask].sort_values("key").reset_index(
                        drop=True)
                    keys = sub["key"].to_numpy()
                    tgt = sub["target"].to_numpy(dtype=np.float64)
                    digest = et.matched_digest(keys)
                    u = units.setdefault(rs, {"keys": keys, "target": tgt,
                                              "digest": digest,
                                              "ref": sub, "vecs": {}})
                    if u["digest"] != digest:
                        raise SystemExit(f"{screen}/{split}/{rs}: variants "
                                         f"differ in rows")
                    for k, c in logit_cols:
                        raw = sub[c].to_numpy(dtype=np.float64)
                        a, b0 = ols[k]
                        u["vecs"][(name, "zero_shot_raw", k)] = raw
                        u["vecs"][(name, "zero_shot_ols", k)] = a * raw + b0
                        u.setdefault("a_ols", {})[(name, k)] = a
                    for m, fdf in ft.items():
                        for k, c in et.fold_columns(fdf.reset_index(),
                                                    et._PRED_COL):
                            u["vecs"][(name, m, k)] = fdf[c].reindex(
                                keys).to_numpy(dtype=np.float64)
            # DegreePrior on the matched rows
            um = units["matched"]
            i, j = et._reference_indices(um["ref"], gene_index)
            for k, scorer in deg_scorers:
                um["vecs"][("DegreePrior", "zero_shot_raw", k)] = np.asarray(
                    scorer(i, j), dtype=np.float64)
            for rs, u in units.items():
                W = bc.resample_counts(len(u["keys"]), N_BOOT, SEED,
                                       (bc.unit_hash(screen, split, rs,
                                                     u["digest"]),))
                pts, reps = bc.bootstrap_spearman(u["target"], u["vecs"], W)
                src = sorted({(n, m) for (n, m, k) in u["vecs"]})
                for (n, m) in src:
                    ks = sorted(k for (nn, mm, k) in u["vecs"]
                                if (nn, mm) == (n, m))
                    fold_pts = np.array([pts[(n, m, k)] for k in ks])
                    fold_reps = np.vstack([reps[(n, m, k)] for k in ks])
                    key = f"{n}|{m}|{screen}|{split}|{rs}"
                    reps_out[key + "|folds_point"] = fold_pts
                    reps_out[key + "|folds_reps"] = fold_reps
                    # check against the published table
                    sel = pub[(pub.model == n) & (pub.method == m) &
                              (pub.screen == screen) & (pub.split == split) &
                              (pub.row_set == rs)]
                    if sel.empty:
                        continue  # e.g. DegreePrior is not on all_sldb rows
                    for k in ks:
                        row = sel[sel.fold == str(k)].iloc[0]
                        chk(pts[(n, m, k)], row.point,
                            f"{key}|fold{k}|point")
                        checks["n_point"] += 1
                    fin = np.isfinite(fold_pts)
                    mean_rep = fold_reps[fin].mean(axis=0)
                    lo, hi, _ = bc.percentile_ci(mean_rep, ALPHA)
                    row = sel[sel.fold == "mean"].iloc[0]
                    chk(float(fold_pts[fin].mean()), row.point,
                        f"{key}|mean|point")
                    chk(lo, row.ci_low, f"{key}|mean|ci_low")
                    chk(hi, row.ci_high, f"{key}|mean|ci_high")
                    checks["n_ci"] += 1
                # published diffs that can be rebuilt here
                for _, row in pubd[(pubd.screen == screen) &
                                   (pubd.split == split) &
                                   (pubd.row_set == rs) &
                                   (pubd.fold == "mean")].iterrows():
                    ka = f"{row.model}|{row.method}|{screen}|{split}|{rs}"
                    kb = f"{row.peer}|{row.peer_method}|{screen}|{split}|{rs}"
                    if kb + "|folds_reps" not in reps_out:
                        continue  # a cluster-only baseline
                    pa = reps_out[ka + "|folds_point"]
                    pb = reps_out[kb + "|folds_point"]
                    both = np.isfinite(pa) & np.isfinite(pb)
                    drep = (reps_out[ka + "|folds_reps"][both] -
                            reps_out[kb + "|folds_reps"][both]).mean(axis=0)
                    st = bc.diff_stats(float((pa[both] - pb[both]).mean()),
                                       drep, ALPHA)
                    for f in ("point", "ci_low", "ci_high", "p_boot_le0"):
                        chk(st[f], getattr(row, f),
                            f"diff {ka} - {kb}|{f}")
                    checks["n_diff"] += 1
            print(f"{arm} {screen}/{split}: "
                  f"{sum(len(u['vecs']) for u in units.values())} vectors "
                  f"[{time.time() - t0:.1f}s]", flush=True)
    if checks["problems"]:
        json.dump(checks, open(out_dir / "repro_check.json", "w"), indent=1)
        raise SystemExit(f"REPRODUCTION FAILED: {len(checks['problems'])} "
                         f"mismatches, first: {checks['problems'][:5]}")
    np.savez_compressed(out_dir / "replicates.npz", **reps_out)
    checks["arm"] = arm
    checks["manifest_started"] = man.get("started")
    checks["seconds"] = time.time() - t_start
    json.dump(checks, open(out_dir / "repro_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in checks.items() if k != "problems"}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
