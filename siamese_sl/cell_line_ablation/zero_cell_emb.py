#!/usr/bin/env python3
"""Post-hoc cell-line ablation: zero the learned cell-line embeddings.

WHAT THIS DOES
--------------
Loads the PUBLISHED multi-modal checkpoints (no retraining), sets
`cell_emb.weight` to zero, and re-scores the same test edges. With a zero cell
vector every head encodes a gene pair to the same latent point, so the encoder
can no longer tell which cell line it is scoring. The gap between the two rows
is how much of the model's ranking depended on the cell-line representation it
learned.

WHAT IT DOES NOT DO
-------------------
Nothing is retrained, so this measures how much the TRAINED model leans on the
cell-line embedding at inference — not what a model trained without cell lines
from scratch would achieve. The per-head scoring bias is left in place (it is
an intercept, not a representation); zeroing it as well is reported separately
in the `+bias` row so the two effects are not conflated.

SCORING
-------
Scores come from the same forward pass and the same OR-collapse (max over
heads) that train.py's exporter uses, and the metrics are computed exactly as
sl_comparison/evaluate_model.py computes them on test edges:
    pos_s = S[test_pos], neg_s = S[test_neg]
    y = [1...1, 0...0];  s = [pos_s, neg_s]
    auprg(y, s), average_precision_score(y, s), roc_auc_score(y, s)
Only the test edges are scored, not the full 8,844 x 8,844 matrix: the matrix
exists in the published pipeline so that the RANKING metrics can be computed,
and every metric here is a function of the test-edge scores alone. The
evaluator's affine shift is rank-neutral and therefore omitted.

SELF-CHECK
----------
The as-trained row must reproduce sl_comparison/results/comparison.csv. The
script asserts this and prints the comparison, so a mistake in gene indexing or
in the preprocessing reconstruction shows up as a mismatch rather than as a
plausible ablation result.

USAGE (local, CPU is fine)
--------------------------
    cd siamese_sl
    python3 cell_line_ablation/zero_cell_emb.py                 # all three CVs
    python3 cell_line_ablation/zero_cell_emb.py --cv_type cv3    # one CV
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SIAMESE = HERE.parent
ROOT = SIAMESE.parent
sys.path.insert(0, str(SIAMESE))
sys.path.insert(0, str(ROOT / "sl_comparison"))

from auprg import auprg  # noqa: E402
from data_loader import apply_multimodal_transform, load_raw_multimodal  # noqa: E402
from siamese_esm import SiameseSLMultiCell  # noqa: E402

try:
    from model_params import write_model_params
except ImportError:  # pragma: no cover
    write_model_params = None

COMBO = ("best_cat_geneformer_go2vec_kg_complex_ppi_raw_prot_t5_text_embed_%s")
FOLDS = ROOT / "sl_comparison" / "folds"
# Row labels, in the order they are reported.
VARIANTS = ("as-trained", "cell_emb=0", "cell_emb=0 +bias=0")


def load_shared_genes():
    return [ln.strip() for ln in open(FOLDS / "genes.txt") if ln.strip()]


@torch.no_grad()
def score_pairs(model, E, rows_i, rows_j, batch=20000):
    """OR-collapsed (max-over-heads) logit for each pair, as the exporter does."""
    out = np.empty(len(rows_i), dtype=np.float64)
    for s in range(0, len(rows_i), batch):
        e = min(s + batch, len(rows_i))
        x1 = E[rows_i[s:e]]
        x2 = E[rows_j[s:e]]
        out[s:e] = model(x1, x2).max(dim=1).values.double().cpu().numpy()
    return out


def metrics(y, s):
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "auprg": float(auprg(y, s)),
    }


def run_cv(cv, raw, gene_to_idx, shared_genes, out_dir=None,
           save_matrices=False, verbose=True):
    model_dir = SIAMESE / "results" / (COMBO % cv)
    if not model_dir.is_dir():
        print(f"  {cv}: {model_dir} not found — skipping")
        return None

    # shared fold index -> row of the model's embedding matrix
    smap = np.array([gene_to_idx.get(g, -1) for g in shared_genes], dtype=np.int64)
    n_unmapped = int((smap < 0).sum())
    if verbose:
        print(f"  gene map: {len(shared_genes) - n_unmapped:,}/{len(shared_genes):,} "
              f"shared genes present in the model's universe"
              + (f" ({n_unmapped} missing)" if n_unmapped else ""))

    per_fold = {v: [] for v in VARIANTS}
    ck_dir = None
    if out_dir is not None:
        ck_dir = Path(out_dir) / f"no_cellline_{cv}" / "checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs = sorted((FOLDS / cv).glob("fold_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    for fd in fold_dirs:
        k = int(fd.name.split("_")[1])
        ckpt_path = model_dir / "checkpoints" / f"fold_{k}_best.pt"
        if not ckpt_path.exists():
            print(f"    fold {k}: no checkpoint — skipped")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]

        # Rebuild this fold's feature space from the transform stored WITH the
        # checkpoint. The PCA is refit per fold, so using another fold's
        # transform would silently score in the wrong basis.
        E = apply_multimodal_transform(raw, ck["preprocessing_transform"])
        E = torch.as_tensor(E, dtype=torch.float32)

        sd = ck["model_state_dict"]
        # Infer the architecture from the WEIGHTS, not from config: the two can
        # disagree (the published run passes --no-last-layer-bias, which lands
        # in config as last_layer_bias=False, and an earlier guess at that key
        # name loaded a model with a bias the checkpoint does not carry).
        # predict.load_model cannot be reused here — it raises
        # NotImplementedError on any checkpoint with cell_emb.* keys.
        model = SiameseSLMultiCell(
            input_dim=int(ck["input_dim"]),
            num_heads=int(sd["cell_emb.weight"].shape[0]),
            encoder_dims=cfg["encoder_dims"],
            dropout=cfg.get("dropout", 0.2),
            last_layer_bias="encoder.projection.bias" in sd,
            pd_epsilon=cfg.get("pd_epsilon", 0.001),
            siamese_encoder_type=cfg.get("siamese_encoder_type", "residual"),
            cell_line_dim=int(sd["cell_emb.weight"].shape[1]),
        )
        model.load_state_dict(sd)  # strict: a silent arch mismatch is worse
                                   # than a crash
        model.eval()  # dropout OFF: in train mode row i and column i disagree

        tp = np.load(fd / "test_pos.npy").astype(np.int64)
        tn = np.load(fd / "test_neg.npy").astype(np.int64)
        pairs = np.vstack([tp, tn])
        y = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
        ri, rj = smap[pairs[:, 0]], smap[pairs[:, 1]]
        keep = (ri >= 0) & (rj >= 0)
        if not keep.all():
            print(f"    fold {k}: dropping {int((~keep).sum()):,} edges whose "
                  f"genes are absent from the model's universe")
        ri, rj, yk = ri[keep], rj[keep], y[keep]

        # --- the three variants ---------------------------------------------
        saved_emb = model.cell_emb.weight.detach().clone()
        saved_bias = model.scoring_bias.detach().clone()

        res = {}
        res["as-trained"] = metrics(yk, score_pairs(model, E, ri, rj))

        model.cell_emb.weight.data.zero_()
        res["cell_emb=0"] = metrics(yk, score_pairs(model, E, ri, rj))
        # --- persist the ablated model --------------------------------------
        # Saved in the SAME checkpoint schema as train.py writes, with the
        # preprocessing_transform / gene_order / embeddings_paths carried over
        # verbatim, so predict.py and eval_finetuning.py can load these
        # directly. Only cell_emb is different (all zeros); `config` records
        # the provenance so an ablated checkpoint can never be mistaken for a
        # published one.
        if ck_dir is not None:
            abl_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            abl_cfg = dict(cfg)
            abl_cfg["cell_line_ablation"] = {
                "variant": "cell_emb=0",
                "source_checkpoint": str(ckpt_path.relative_to(ROOT)),
                "note": "cell_emb zeroed post hoc; NOT retrained. "
                        "scoring_bias is left as trained.",
            }
            torch.save({
                "epoch": ck.get("epoch"),
                "model_state_dict": abl_sd,
                "metrics": res["cell_emb=0"],
                "config": abl_cfg,
                "input_dim": ck["input_dim"],
                "preprocessing_transform": ck["preprocessing_transform"],
                "gene_order": ck["gene_order"],
                "embeddings_paths": ck["embeddings_paths"],
            }, ck_dir / f"fold_{k}_best.pt")

        # --- optional dense matrix, so the shared evaluator can grade this ---
        if save_matrices:
            mdir = Path(out_dir) / f"no_cellline_{cv}"
            n = len(shared_genes)
            S = np.full((n, n), np.nan, dtype=np.float32)
            have = np.where(smap >= 0)[0]
            rows = smap[have]
            with torch.no_grad():
                Z, Hd = model.encoder.forward_with_hidden(
                    torch.cat([E[rows],
                               model.cell_emb.weight[0].unsqueeze(0)
                               .expand(len(rows), -1)], dim=1))
                tau = torch.exp(model.log_temperature).float()
                eps = float(model.pd_epsilon)
                # All heads are identical once cell_emb is zero, so the
                # max-over-heads collapse is the shared score plus the LARGEST
                # per-head bias. Exported here, before scoring_bias is zeroed
                # for the third variant, so the matrix matches the cell_emb=0
                # row and not the +bias=0 one.
                bmax = float(model.scoring_bias.max())
                blk = 2048
                sub = np.empty((len(rows), len(rows)), dtype=np.float32)
                for a in range(0, len(rows), blk):
                    b = min(a + blk, len(rows))
                    cur = Z[a:b] @ Z.T
                    if eps:
                        cur.addmm_(Hd[a:b], Hd.T, alpha=eps)
                    sub[a:b] = (cur * tau + bmax).cpu().numpy()
                sub = 0.5 * (sub + sub.T)
            S[np.ix_(have, have)] = sub
            S[np.isnan(S)] = np.nanmin(sub)
            np.save(mdir / f"fold_{k}_{cv}_predictions.npy", S)
            del S, sub

        model.scoring_bias.data.zero_()
        res["cell_emb=0 +bias=0"] = metrics(yk, score_pairs(model, E, ri, rj))



        model.cell_emb.weight.data.copy_(saved_emb)
        model.scoring_bias.data.copy_(saved_bias)

        for v in VARIANTS:
            per_fold[v].append(res[v])
        if verbose:
            print(f"    fold {k}: n={len(yk):,} pi={yk.mean():.4f}  "
                  + "  ".join(f"{v}: AUPRG {res[v]['auprg']:+.4f}" for v in VARIANTS))
        del E

    if save_matrices and out_dir is not None and write_model_params is not None:
        # evaluate_model.py refuses to grade matrices with no fold_stamp, and a
        # stamp MISMATCH is always fatal — so this records the fold set these
        # matrices were scored against.
        mdir = Path(out_dir) / f"no_cellline_{cv}"
        write_model_params(mdir, model, folds_dir=str(FOLDS), cv_type=cv,
                           selection_regime="val")
        print(f"    wrote {mdir / 'model_params.json'}")

    if not per_fold["as-trained"]:
        return None
    # Population sd (ddof=0), matching the paper's tables.
    agg = {v: {m: (float(np.mean([f[m] for f in per_fold[v]])),
                   float(np.std([f[m] for f in per_fold[v]])))
               for m in ("auroc", "ap", "auprg")} for v in VARIANTS}
    # Paired per-fold deltas: prevalence varies a lot across folds and affects
    # every variant identically, so it cancels in the difference. The across-
    # fold sd above does not show that.
    paired = {v: {m: float(np.mean([a[m] - b[m] for a, b in
                                    zip(per_fold[v], per_fold["as-trained"])]))
                  for m in ("auroc", "ap", "auprg")}
              for v in VARIANTS[1:]}
    return {"agg": agg, "paired": paired, "n_folds": len(per_fold["as-trained"])}


def published(cv):
    """The paper's row for this CV, for the self-check."""
    import csv
    path = ROOT / "sl_comparison" / "results" / "comparison.csv"
    if not path.exists():
        return {}
    want = {"auroc": "auroc", "ap": "ap", "auprg": "auprg"}
    out = {}
    for r in csv.DictReader(open(path)):
        if r["cv"] == cv and r["model"] == "siamese_sl" and r["metric"] in want:
            try:
                out[want[r["metric"]]] = float(r["mean"])
            except (TypeError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cv_type", choices=["cv1", "cv2", "cv3"], action="append",
                    help="repeatable; default all three")
    ap.add_argument("--json_out", help="write the numbers here as JSON")
    ap.add_argument("--out_dir", default=str(HERE / "models"),
                    help="where to save the ablated checkpoints "
                         "(default: cell_line_ablation/models)")
    ap.add_argument("--no_save", action="store_true",
                    help="do not save the ablated checkpoints")
    ap.add_argument("--save_matrices", action="store_true",
                    help="also export fold_<k>_<cv>_predictions.npy so "
                         "grade.sh / evaluate_model.py can grade this run. "
                         "~313 MB per fold (8844^2 float32).")
    args = ap.parse_args()
    cvs = args.cv_type or ["cv1", "cv2", "cv3"]

    # The six modality files are identical across folds and CVs, so load once.
    probe = SIAMESE / "results" / (COMBO % cvs[0]) / "checkpoints" / "fold_0_best.pt"
    if not probe.exists():
        sys.exit(f"ERROR: {probe} not found. This script ablates the PUBLISHED "
                 f"checkpoints; train the combo first.")
    paths = torch.load(probe, map_location="cpu", weights_only=False)["embeddings_paths"]
    paths = [str(ROOT / "data" / Path(p).name) for p in paths]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        sys.exit("ERROR: missing embedding files:\n  " + "\n  ".join(missing))
    print("Loading raw embeddings (once):")
    for p in paths:
        print(f"  {Path(p).name}")
    raw, gene_to_idx, _ = load_raw_multimodal(paths)
    print(f"  {len(gene_to_idx):,} genes, dims {[t.shape[1] for t in raw]}\n")

    shared_genes = load_shared_genes()
    results = {}
    for cv in cvs:
        print(f"=== {cv} ===")
        r = run_cv(cv, raw, gene_to_idx, shared_genes,
                   out_dir=None if args.no_save else args.out_dir,
                   save_matrices=args.save_matrices)
        if r:
            results[cv] = r
        print("")

    if not results:
        sys.exit("no CV produced results")

    # ---------------- report -------------------------------------------------
    print("=" * 78)
    print("POST-HOC CELL-LINE ABLATION — mean (population sd) over folds")
    print("=" * 78)
    for cv, r in results.items():
        print(f"\n{cv}  ({r['n_folds']} folds)")
        print(f"  {'variant':22s} {'AUROC':>16s} {'AP':>16s} {'AUPRG':>16s}")
        for v in VARIANTS:
            a = r["agg"][v]
            print(f"  {v:22s} " + " ".join(
                f"{a[m][0]:+.4f} ({a[m][1]:.3f})".rjust(16)
                for m in ("auroc", "ap", "auprg")))
        print(f"  {'-' * 74}")
        for v in VARIANTS[1:]:
            d = r["paired"][v]
            print(f"  {('Δ ' + v):22s} " + " ".join(
                f"{d[m]:+.4f}".rjust(16) for m in ("auroc", "ap", "auprg")))
        print("  (Δ is the mean paired per-fold change vs as-trained)")
        print("  Note: the +bias=0 row is IDENTICAL to cell_emb=0 by "
              "construction, not by accident.")
        print("        With cell_emb zero every head scores a pair alike, so "
              "max-over-heads")
        print("        adds the single largest bias to every pair — a constant "
              "shift, and all")
        print("        three metrics are rank-based. The per-head intercept "
              "cannot matter once")
        print("        the heads are indistinguishable.")

        # self-check against the paper
        pub = published(cv)
        if pub:
            print(f"\n  self-check vs comparison.csv (siamese_sl / {cv}):")
            ok = True
            for m in ("auroc", "ap", "auprg"):
                if m not in pub:
                    continue
                got = r["agg"]["as-trained"][m][0]
                delta = abs(got - pub[m])
                flag = "match" if delta < 5e-3 else f"MISMATCH (Δ={delta:.4f})"
                if delta >= 5e-3:
                    ok = False
                print(f"    {m:6s} here {got:.4f}  published {pub[m]:.4f}  -> {flag}")
            if not ok:
                print("    WARNING: the as-trained row does not reproduce the "
                      "published numbers, so the ablation deltas above are not "
                      "trustworthy. Check the gene map and the per-fold "
                      "preprocessing transform before reading anything into them.")
        else:
            print("\n  self-check skipped: comparison.csv not found")

    if not args.no_save:
        print(f"\nablated checkpoints -> {args.out_dir}/no_cellline_<cv>/checkpoints/")
        if args.save_matrices:
            print(f"score matrices      -> {args.out_dir}/no_cellline_<cv>/")
            print("grade them with:  ./cell_line_ablation/grade.sh")
        else:
            print("re-run with --save_matrices to make this gradeable by "
                  "sl_comparison/evaluate_model.py")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
