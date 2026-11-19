"""AdamsonGI screen + the SynLethDB in-distribution / agreement tables
(2026-11-12).

- eval_finetuning._load_adamson_gi reads Simpson et al. 2023 Table S3
  (title row, header row, GeneCombinationName 'A:B'), drops self pairs and
  non-targeting controls and rows without the requested score.
- eval_finetuning.conf carries the AdamsonGI row with sign -1 (negative GI =
  synergistic / SL), next to the Adamson Gamma row.
- prepare_dataset reports SynLethDB positives / screened negatives apart
  (negatives = non-SL and not SL, the folds' OR-collapse).
- external_transfer: sldb_overlap.csv / sldb_agreement.csv counts, fold
  training overlap, screen_role, the SynLethDB_label predictor row, the
  matched_incl_sldb / matched_in_sldb row sets, the negatives-file fallback
  and the flag-vs-files guard. All on small synthetic files.
"""

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SIAMESE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

_spec = importlib.util.spec_from_file_location(
    "external_transfer_sldb",
    os.path.join(_REPO, "sl_comparison", "external_transfer.py"))
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)


def _ef():
    import eval_finetuning as ef
    return ef


# ---------------------------------------------------------------------------
# AdamsonGI loader + DATASETS row
# ---------------------------------------------------------------------------

S3_HEADER = ["GeneCombinationName", "Gamma.GI", "Gamma.Nsupp", "Tau.GI",
             "InteractionCategory"]
S3_ROWS = [
    ["BRCA1:PARP1", -3.5, 8, -4.0, "Interaction, unmodified - synthetic lethal"],
    ["ATM:ATR", 2.0, 8, 1.0, "Interaction, unmodified - buffering"],
    ["ATM:ATM", -9.0, 8, -9.0, "Same gene - no category"],
    ["ATM:non-targeting", 0.1, 8, 0.1, "Control - no category"],
    ["BRCA2:RAD51", "", 8, 0.5, ""],          # no Gamma.GI score
    ["rad51:xrcc3", 0.25, 8, 0.0, "No interaction"],
]


def _write_s3_csv(path: Path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Genetic interaction scores for all gene pairs."] +
                   [""] * (len(S3_HEADER) - 1))
        w.writerow(S3_HEADER)
        w.writerows(S3_ROWS)


def test_adamson_gi_loader_csv():
    ef = _ef()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s3.csv"
        _write_s3_csv(p)
        df = ef._load_adamson_gi(str(p), "Gamma.GI")
    got = {(a, b): v for a, b, v in df[["gene1_symbol", "gene2_symbol",
                                        "outcome_raw"]].itertuples(index=False)}
    assert got == {("BRCA1", "PARP1"): -3.5, ("ATM", "ATR"): 2.0,
                   ("RAD51", "XRCC3"): 0.25}, got


def test_adamson_gi_loader_drops_names_without_a_pair():
    """A GeneCombinationName without ':' (or with an empty side) splits
    into a null / empty gene2: dropped, never kept as a pair with gene2 NaN
    (it passed the self-pair filter before)."""
    import csv
    ef = _ef()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s3.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["title"] + [""] * (len(S3_HEADER) - 1))
            w.writerow(S3_HEADER)
            w.writerows(S3_ROWS + [["CHEK1", -1.0, 8, -1.0, ""],
                                   ["CHEK2:", -2.0, 8, -2.0, ""],
                                   [" :WEE1", -3.0, 8, -3.0, ""]])
        df = ef._load_adamson_gi(str(p), "Gamma.GI")
    assert df["gene1_symbol"].notna().all() and \
        df["gene2_symbol"].notna().all(), df
    assert set(zip(df["gene1_symbol"], df["gene2_symbol"])) == {
        ("BRCA1", "PARP1"), ("ATM", "ATR"), ("RAD51", "XRCC3")}, df


def test_adamson_gi_loader_xlsx_and_bad_column():
    try:
        import openpyxl
    except ImportError:  # the laptop venv may lack it; the cluster has it
        return
    ef = _ef()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s3.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Genetic interaction scores for all gene pairs."])
        ws.append(S3_HEADER)
        for r in S3_ROWS:
            ws.append([None if x == "" else x for x in r])
        wb.save(p)
        df = ef._load_adamson_gi(str(p), "Tau.GI")
        assert len(df) == 4  # BRCA2:RAD51 has a Tau.GI score
        try:
            ef._load_adamson_gi(str(p), "Rho.GI")
        except ValueError as e:
            assert "Rho.GI" in str(e)
        else:
            raise AssertionError("an absent score column was accepted")


def test_adamson_gi_dataset_row():
    ef = _ef()
    assert "AdamsonGI" in ef.DATASET_LOADERS
    conf = Path(_SIAMESE, "slurm", "eval_finetuning.conf").read_text()
    rows = re.findall(r'^\s*"([A-Za-z]+:[^"]+)"', conf, re.M)
    specs = {s["name"]: s for s in ef.parse_datasets_spec(rows)}
    assert {"Adamson", "AdamsonGI", "Corn", "Gilbert"} <= set(specs)
    gi = specs["AdamsonGI"]
    # negative GI = synergistic (SL): target = -GI, higher = more SL
    assert gi["column"] == "Gamma.GI" and gi["sign"] == -1.0
    assert gi["head"] == "K562"
    assert gi["file"] == "external data/Simpson2023_TableS3_GI_scores.xlsx"
    assert specs["Adamson"]["column"] == "Gamma"  # the Gamma row is kept
    assert list(specs) == list(et.SCREENS)  # same screens, same order


def test_prepare_dataset_counts_positives_and_negatives_apart():
    ef = _ef()
    raw = pd.DataFrame({"gene1_symbol": ["A", "A", "B", "C"],
                        "gene2_symbol": ["B", "C", "C", "D"],
                        "outcome_raw": [1.0, 2.0, 3.0, 4.0]})
    ent = {"A": "1", "B": "2", "C": "3", "D": "4"}
    orig = ef._symbols_to_entrez
    ef._symbols_to_entrez = lambda s, m: s.map(ent)
    try:
        pos = {frozenset(["1", "2"]), frozenset(["2", "3"])}
        neg = {frozenset(["2", "3"]), frozenset(["1", "3"])}  # 2|3 conflict
        df, cov = ef.prepare_dataset(raw, {g: i for i, g in enumerate(
            ["1", "2", "3", "4"])}, pos | neg, -1.0, sldb_pos=pos,
            sldb_neg=neg)
    finally:
        ef._symbols_to_entrez = orig
    assert cov["sldb_overlap_count"] == 3
    assert cov["sldb_pos_count"] == 2          # 1|2, 2|3 (OR-collapse)
    assert cov["sldb_neg_count"] == 1          # 1|3 only
    assert abs(cov["sldb_neg_frac"] - 0.25) < 1e-12
    assert list(df.sort_values("outcome_raw")["target"]) == [-1, -2, -3, -4]


# ---------------------------------------------------------------------------
# external_transfer SynLethDB tables
# ---------------------------------------------------------------------------

GENES = [str(100 + g) for g in range(12)]
GIDX = {g: i for i, g in enumerate(GENES)}


def _key(a, b):
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def _tree(root: Path, neg_file=True, flag_extra=()):
    """One SLeuth-style eval dir on a synthetic screen + folds + pair files.

    Screen: every pair of the 12 genes plus 6 pairs with a gene outside
    genes.txt; SynLethDB positives / negatives chosen below; target makes
    the positives high (agreement), with ties.
    """
    rng = np.random.default_rng(3)
    pairs = [(GENES[i], GENES[j]) for i in range(12) for j in range(i + 1, 12)]
    pairs += [(GENES[i], str(900 + i)) for i in range(6)]
    keys = [_key(a, b) for a, b in pairs]
    pos = set(keys[0:12])
    neg = set(keys[10:40]) | {_key("100", "999")}  # 10,11 also SL
    target = rng.normal(size=len(pairs))
    for i, k in enumerate(keys):
        if k in pos:
            target[i] += 2.0
    target = np.round(target, 1)
    flagged = np.array([(k in pos) or (k in neg) or (k in flag_extra)
                        for k in keys])
    split = np.where(np.arange(len(pairs)) % 3 == 0, "test", "train")
    v = root / "v"
    e = v / "eval_finetuning_cv1"
    e.mkdir(parents=True)
    (v / "config.json").write_text(json.dumps({"cv_type": "cv3"}))
    import hashlib
    (v / "checkpoints").mkdir()
    hashes = {}
    for k in range(2):
        f = v / "checkpoints" / f"fold_{k}_best.pt"
        f.write_bytes(f"weights {k}".encode())
        hashes[f"results/v/checkpoints/{f.name}"] = hashlib.sha256(
            f.read_bytes()).hexdigest()
    (e / "config_used.json").write_text(json.dumps(
        {"ft_modes": [], "sldb_overlap_sources": ["p", "n"],
         "fold_ids": [0, 1], "checkpoint_sha256": hashes}))
    df = pd.DataFrame({"gene1_entrez": [a for a, _ in pairs],
                       "gene2_entrez": [b for _, b in pairs],
                       "outcome_raw": -target, "target": target,
                       "sldb_overlap": flagged, "split": split,
                       "logit_fold0": target + rng.normal(size=len(pairs)),
                       "logit_fold1": rng.normal(size=len(pairs))})
    df.to_csv(e / "Toy_predictions.csv", index=False)
    folds = root / "folds"
    (folds / "genes.txt").parent.mkdir(parents=True)
    (folds / "genes.txt").write_text("\n".join(GENES) + "\n")
    train = {0: [keys[0], keys[1], keys[20]], 1: [keys[2], keys[30],
                                                  keys[31], keys[50]]}
    for k, ks in train.items():
        d = folds / "cv3" / f"fold_{k}"
        d.mkdir(parents=True)
        arr = np.array([[GIDX[x.split("|")[0]], GIDX[x.split("|")[1]]]
                        for x in ks], dtype=np.int32)
        np.save(d / "train_pos.npy", arr[:1])
        np.save(d / "train_neg.npy", arr[1:])
    sl = root / "pos.txt"
    sl.write_text("".join(f"{x.split('|')[0]}\t{x.split('|')[1]}\n"
                          for x in sorted(pos)))
    ng = root / "neg.txt"
    if neg_file:
        ng.write_text("".join(f"{x.split('|')[1]} {x.split('|')[0]}\n"
                              for x in sorted(neg)))
    # meta.json records the pair files, as prepare_folds.py does
    (folds / "meta.json").write_text(json.dumps(
        {"pos_path": str(sl), "neg_path": str(ng)}))
    return v, folds, df, pos, neg, train


def _run(root, v, folds, **kw):
    return et.run([("S", v)], [], True, folds / "genes.txt", folds,
                  root / "out", "cv3", ["Toy"], ["cv1"],
                  allow_unstamped=True, allow_legacy_eval=False, **kw)


def _auroc(t, y):
    r = rankdata(t)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def test_sldb_overlap_agreement_and_label_row():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v, folds, df, pos, neg, train = _tree(root)
        res = _run(root, v, folds)
        keys = np.array([_key(a, b) for a, b in
                         zip(df.gene1_entrez, df.gene2_entrez)])
        in_pos = np.isin(keys, list(pos))
        in_neg = np.isin(keys, list(neg)) & ~in_pos
        ov = res["sldb_overlap"].set_index(["scope", "split"])
        a = ov.loc[("all_pairs", "any")]
        assert a["n_pairs"] == len(df)
        assert a["n_sldb_pos"] == in_pos.sum() == 12
        assert a["n_sldb_neg"] == in_neg.sum() == 28   # 30 - 2 SL; 100|999 unscored
        assert a["n_sldb_any"] == (in_pos | in_neg).sum()
        assert a["neg_source"] == "file" and a["n_flag_mismatch"] == 0
        for k, ks in train.items():
            assert a[f"n_train_fold{k}"] == len(ks)
        g = ov.loc[("all_pairs_genes_txt", "any")]
        assert g["n_pairs"] == 66
        t = ov.loc[("test_genes_txt", "cv1")]
        assert t["n_pairs"] == res["row_sets"].iloc[0]["n_matched_incl_sldb"]
        # 40 of 72 pairs in SynLethDB -> in-distribution check
        assert res["screen_roles"] == {"Toy": et.ROLE_IN_DISTRIBUTION}
        assert set(res["zero_shot_summary"]["screen_role"]) == {
            et.ROLE_IN_DISTRIBUTION}

        ag = res["sldb_agreement"].set_index(["scope", "split"])
        lab = in_pos | in_neg
        tg = df["target"].to_numpy()
        r = ag.loc[("all_pairs", "any")]
        assert r["n_labelled"] == lab.sum() and r["n_pos"] == 12
        assert abs(r["auroc_target_vs_label"] -
                   _auroc(tg[lab], in_pos[lab])) < 1e-12
        assert r["auroc_target_vs_label"] > 0.8
        assert abs(r["spearman_target_vs_label"] -
                   spearmanr(in_pos[lab].astype(float), tg[lab])[0]) < 1e-12

        # the label as a predictor on matched_in_sldb (test, genes.txt, SLDB)
        zs = res["zero_shot_summary"]
        lr = zs[zs["model"] == et.SLDB_LABEL_NAME]
        assert list(lr["row_set"]) == [et.ROW_MATCHED_IN_SLDB]
        test = (df["split"] == "test").to_numpy()
        both = np.array([a in GIDX and b in GIDX for a, b in
                         zip(df.gene1_entrez, df.gene2_entrez)])
        m = test & both & df["sldb_overlap"].to_numpy()
        want = spearmanr(in_pos[m].astype(float), tg[m])[0]
        assert abs(lr.iloc[0]["spearman_mean"] - want) < 1e-12
        assert lr.iloc[0]["spearman_sd_pop"] == 0.0
        # every model on the three matched-gene row sets
        for model in ("S", et.DEGREE_PRIOR_NAME):
            got = set(zs.loc[zs["model"] == model, "row_set"])
            assert set(et.MATCHED_ROW_SETS) <= got, (model, got)
        out = root / "out"
        assert (out / "sldb_overlap.csv").exists()
        assert (out / "sldb_agreement.csv").exists()
        md = (out / "summary.md").read_text()
        assert "## Screen role (SynLethDB overlap)" in md
        assert "in_distribution_check" in md


def test_negatives_file_missing_falls_back_to_flag():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v, folds, df, pos, neg, _ = _tree(root, neg_file=False)
        res = _run(root, v, folds)
        a = res["sldb_overlap"].set_index("scope").loc["all_pairs"]
        assert a["neg_source"] != "file"
        assert a["n_sldb_neg"] == 28  # flagged and not SL
        assert "not found" in (root / "out" / "summary.md").read_text()


def test_flag_disagreeing_with_pair_files_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v, folds, *_ = _tree(root, flag_extra={_key("110", "111")})
        try:
            _run(root, v, folds)
        except ValueError as e:
            assert "disagrees" in str(e), e
        else:
            raise AssertionError("an eval matched on other pair files passed")


def test_no_pair_files_recorded_skips_the_tables():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v, folds, *_ = _tree(root)
        (folds / "meta.json").unlink()
        res = _run(root, v, folds)
        assert res["sldb_overlap"].empty and not res["screen_roles"]
        assert et.SLDB_LABEL_NAME not in set(res["zero_shot_summary"]["model"])
        assert not (root / "out" / "sldb_overlap.csv").exists()
