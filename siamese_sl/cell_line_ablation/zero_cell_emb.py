#!/usr/bin/env python3
"""Post-hoc cell-line ablation: zero the learned cell-line embeddings.

WHAT THIS DOES
--------------
Loads a trained per_head multi-modal combo's checkpoints (no retraining), sets
`cell_emb.weight` to zero, and re-scores the same test edges. With a zero cell
vector the encoder no longer sees which cell line it is scoring, so the gap
between the rows is how much of the model's ranking depended on the cell-line
representation it learned. The checkpoints are git-ignored local artifacts
under siamese_sl/results/, not part of the repository (see CHECKPOINTS below).

WHAT IT DOES NOT DO
-------------------
Nothing is retrained, so this measures how much the TRAINED model leans on the
cell-line embedding at inference — not what a model trained without cell lines
from scratch would achieve (the pooled variants are that counterpart). The
per-head scoring bias is left in place (it is an intercept, not a
representation); zeroing it as well is reported separately in the `+bias=0`
row so the two effects are not conflated.

WHICH CHECKPOINTS
-----------------
Every per_head SiameseSLMultiCell checkpoint, of either kernel and under any
train.py options (the model is rebuilt from its weights by
eval_finetuning.load_multicell_model: LayerNorm or not, raw or log tau, the
per-head projection bias c_h, kernel_A, ...). A pooled (--cell_lines pooled)
or single-output checkpoint has no cell_emb, so the ablation is NOT
APPLICABLE and the CV is skipped as such.

Under the published architecture (--no_per_head_last_layer_bias, one shared
tau, no per-head offsets) all head identity lives in cell_emb, so once it is
zero every head scores a pair alike up to its scoring-bias intercept. Under
train.py's defaults the per-head c_h (head_proj_bias) still tells the heads
apart after cell_emb is zeroed, so two more rows are reported when it exists:
`+head_proj_bias=mean` (every head gets the head-average c_h: head identity
removed, the shared c_h part kept) and `+head_proj_bias=0` (c_h removed
outright, which also drops its gene-level term c_h^T W h_i). The exported
('removed') row is always `cell_emb=0`, as in the paper.

SCORING
-------
Scores come from the model's forward pass and the same OR-collapse (max over
heads) train.py uses, and the metrics are computed exactly as
sl_comparison/evaluate_model.py computes them on test edges:
    pos_s = S[test_pos], neg_s = S[test_neg]
    y = [1...1, 0...0];  s = [pos_s, neg_s]
    auprg(y, s), average_precision_score(y, s), roc_auc_score(y, s)
Every metric here is a function of the test-edge scores alone. The
evaluator's affine shift is order-preserving and therefore omitted; the
float32 near-ties it can merge may make the self-check differ at the ~1e-6
level. With --save_matrices the `cell_emb=0` model is also exported as the
dense 8,844 x 8,844 matrix (retrieval_metrics.build_any_score_matrix, the
builder train.py exports with) so the shared evaluator can grade it
(grade.sh); its entries are checked against the forward pass on the test
edges before it is written.

SELF-CHECK
----------
The as-trained row must reproduce the source combo's own test metrics:
results.json fold_metrics[k] auroc_any / aupr_any / auprg_any (the same test
edges and OR-collapse, scored by train.py). A difference above
--selfcheck_tol in any fold means the gene map or the preprocessing
reconstruction is wrong: the CV is marked selfcheck_failed, its matrices are
deleted (nothing stale for grade.sh) and the script exits 1. The cross-check
of the graded rows against comparison.csv is left to
sl_comparison/paper_tables.py, which runs after the compare job.

CHECKPOINTS
-----------
Read from <model_dir>/checkpoints/fold_<k>_best.pt, where <model_dir> is
--model_dir CV=DIR (what cell_line_ablation/run_cell_line_ablation.sh passes,
from each variant's results/<prefix>latest_combo_<cv>.txt marker) or
siamese_sl/results/<--combo % cv>. The combo must have been trained on the
CURRENT sl_comparison/folds: the fold stamp train.py wrote into its
model_params.json must match them, or that CV stops before scoring; likewise
when the embedding files are not the ones the combo was trained on (the
content hashes its checkpoints record). A CV that stops this way, or with
any other error, keeps no matrices and no model_params.json; the CVs after
it still run. The
combo job (slurm/run_best_per_category_combo.sh) moves results/best_cat_* to
results/_superseded_combos/job_<id>/ when it starts and deletes them there
once Step 2 selection succeeds, and clean_previous_runs.sh deletes them.

OUTPUTS
-------
--out_dir/<--cv_dir_name % cv>/ (default models/no_cellline_<cv>/):
checkpoints/fold_<k>_best.pt (cell_emb zeroed) plus config.json, results.json
(this CV's rows, per-fold values and self-check; schema in README.md) and,
with --save_matrices, fold_<k>_<cv>_predictions.npy + model_params.json.
Each CV's previous outputs (and any eval_finetuning_*/ or pair_predictions/
scored from them) are deleted first, unless --no_save, which writes and
deletes nothing under --out_dir; so a skipped fold or CV, a crash or a saving
run without --save_matrices leaves grade.sh nothing stale to grade.

EXIT STATUS
-----------
0 when every requested CV was ablated (or is not applicable), 1 when any CV
was missing, incomplete, failed its self-check, or stopped with an error.

USAGE (from siamese_sl/)
------------------------
    python3 cell_line_ablation/zero_cell_emb.py                 # all three CVs
    python3 cell_line_ablation/zero_cell_emb.py --cv_type cv3    # one CV
    python3 cell_line_ablation/zero_cell_emb.py --combo 'best_cat_<winners>_%s'
    python3 cell_line_ablation/zero_cell_emb.py --save_matrices \\
        --model_dir cv1=results/best_cat_<winners>_cv1 --out_dir results \\
        --cv_dir_name 'cellline_ablation_%s' --model_name siamese_sl
"""
import argparse
import importlib.util
import json
import shutil
import sys
import traceback
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
from eval_finetuning import (checkpoint_input_problems,  # noqa: E402
                             load_multicell_model)

try:
    from model_params import (compare_fold_stamp, read_model_params,
                              write_model_params)
except ImportError:  # pragma: no cover
    compare_fold_stamp = read_model_params = write_model_params = None

# siamese_sl/retrieval_metrics.py BY PATH: sl_comparison/ is first on sys.path
# above, and its retrieval_metrics.py (a different module) has no
# build_any_score_matrix.
_rm_spec = importlib.util.spec_from_file_location(
    "_siamese_retrieval_metrics", SIAMESE / "retrieval_metrics.py")
_rm = importlib.util.module_from_spec(_rm_spec)
_rm_spec.loader.exec_module(_rm)
build_any_score_matrix = _rm.build_any_score_matrix

COMBO = ("best_cat_geneformer_go2vec_kg_complex_ppi_raw_prot_t5_text_embed_%s")
FOLDS = ROOT / "sl_comparison" / "folds"
METRICS = ("auroc", "ap", "auprg")
# The source combo's results.json key for each metric (train.py's *_any view).
SELFCHECK_KEYS = {"auroc": "auroc_any", "ap": "aupr_any", "auprg": "auprg_any"}
# Rows, in report order. The +head_proj_bias rows exist only for checkpoints
# with a per-head c_h (train.py's default).
ROW_TRAINED = "as-trained"
ROW_REMOVED = "cell_emb=0"  # the exported / graded 'removed' row
ROW_BIAS0 = "cell_emb=0 +bias=0"
ROW_HPB_MEAN = "cell_emb=0 +head_proj_bias=mean"
ROW_HPB0 = "cell_emb=0 +head_proj_bias=0"


def load_shared_genes():
    return [ln.strip() for ln in open(FOLDS / "genes.txt") if ln.strip()]


def _rel(p):
    """p relative to the repo root when inside it, else absolute."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def not_applicable_reason(sd):
    """Why the ablation cannot apply to this state dict, or None."""
    if "cell_emb.weight" not in sd:
        return ("no cell_emb.weight: a --cell_lines pooled or single-output "
                "checkpoint has no cell-line embedding to zero")
    return None


def residual_head_terms(model):
    """Per-head terms other than cell_emb / scoring_bias that keep the heads
    apart once cell_emb is zero (empty = the published architecture)."""
    terms = []
    if model.head_proj_bias is not None:
        terms.append("head_proj_bias (per-head c_h)")
    if model.head_input_bias is not None:
        terms.append("head_input_bias")
    if model.per_head_temperature:
        terms.append("per-head temperature")
    if model.inner_norm == "batch":
        terms.append("per-head inner-norm statistics")
    return terms


@torch.no_grad()
def score_pairs(model, E, rows_i, rows_j, batch=8192):
    """OR-collapsed (max-over-heads) logit for each pair, as train.py does."""
    out = np.empty(len(rows_i), dtype=np.float64)
    ri = torch.as_tensor(rows_i, dtype=torch.long, device=E.device)
    rj = torch.as_tensor(rows_j, dtype=torch.long, device=E.device)
    for s in range(0, len(rows_i), batch):
        e = min(s + batch, len(rows_i))
        out[s:e] = (model(E[ri[s:e]], E[rj[s:e]]).max(dim=1).values
                    .double().cpu().numpy())
    return out


def metrics(y, s):
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "auprg": float(auprg(y, s)),
    }


def clear_previous_outputs(mdir, cv):
    """Delete a previous run's outputs for this CV from mdir, up front.

    grade.sh grades every fold_*_<cv>_predictions.npy it finds, and nothing in
    them names the combo they came from. A skipped CV or fold, a crash, or a
    run without --save_matrices must therefore leave files MISSING (grade.sh /
    evaluate_model.py refuse), never another combo's matrices next to fresh
    checkpoints. (model_params.purge_stale_predictions removes a checkpoint
    only next to a matrix, which misses a run that saved no matrices.) The
    eval_finetuning_*/ dirs eval_finetuning.py writes here by default were
    scored from the checkpoints deleted now, and the new config.json would
    otherwise name another combo as their source. The same holds for a
    pair_predictions/ export (export_pair_predictions.py) of the old ones.
    """
    stale = [*mdir.glob(f"fold_*_{cv}_predictions.npy"),
             *(mdir / "checkpoints").glob("fold_*_best.pt"),
             mdir / "model_params.json", mdir / "config.json",
             mdir / "results.json",
             *(p for p in mdir.glob("eval_finetuning_*") if p.is_dir()),
             mdir / "pair_predictions"]
    stale = [p for p in stale if p.exists()]
    for p in stale:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    if stale:
        print(f"  cleared {len(stale)} output file(s)/dir(s) of a previous run "
              f"from {mdir}")
    return len(stale)


def require_combo_fold_stamp(model_dir, cv):
    """Refuse a combo that was not trained on the current fold set.

    The test edges scored below come from the CURRENT sl_comparison/folds, and
    --save_matrices stamps its matrices with that fold set. A combo trained on
    an older fold set would score its own training pairs as test edges, and
    the re-stamped matrices would pass evaluate_model.py's fold-stamp check.
    train.py records the fold set it trained on in the combo dir's
    model_params.json (write_model_params with --folds_dir).
    """
    if read_model_params is None:  # pragma: no cover
        return
    recorded = read_model_params(model_dir).get("fold_stamp")
    diffs = (compare_fold_stamp(recorded, FOLDS, cv) if recorded else
             [f"no fold_stamp in {model_dir / 'model_params.json'}"])
    if diffs:
        raise SystemExit(
            f"ERROR: {model_dir} was not trained on the current fold set "
            f"{FOLDS}:\n  " + "\n  ".join(diffs) +
            "\nScoring it on these test edges would score pairs it trained "
            "on. Retrain the combo on the current folds.")


def carried_training_params(model_dir):
    """The SOURCE combo's training provenance, for the ablated row.

    Nothing is retrained here, so this row's training seed and checkpoint
    selection metric / regime ARE the combo's — train.py recorded them in its
    model_params.json (the file require_combo_fold_stamp already reads). Left
    out, compare.py's seed report and paper_tables' item_inputs (D3) file
    every <row>_nocell row under UNRECORDED / rows_without_seed although the
    seed is known. A key the combo does not record stays absent.
    """
    src = read_model_params(model_dir) if read_model_params else {}
    out = {"selection_regime": src.get("selection_regime", "val")}
    out.update({k: src[k] for k in ("seed", "selection_metric")
                if src.get(k) is not None})
    return out


def combo_fold_reference(model_dir):
    """{fold id: {metric: value}} from the combo's results.json, or {}."""
    p = Path(model_dir) / "results.json"
    try:
        fm = json.loads(p.read_text()).get("fold_metrics") or []
    except (OSError, ValueError):
        return {}
    ref = {}
    for pos, m in enumerate(fm):
        k = int(m.get("fold", pos))  # "fold" is recorded by newer train.py
        ref[k] = {ours: m.get(key) for ours, key in SELFCHECK_KEYS.items()}
    return ref


def summarize(per_fold, rows):
    """Population-sd aggregate plus paired per-fold deltas vs as-trained."""
    # [mean, sd] per metric; population sd (ddof=0), matching the paper.
    agg = {r: {m: [float(np.mean([f[m] for f in per_fold[r]])),
                   float(np.std([f[m] for f in per_fold[r]]))]
               for m in METRICS} for r in rows}
    # Paired per-fold deltas: prevalence varies a lot across folds and affects
    # every row identically, so it cancels in the difference. The across-fold
    # sd above does not show that.
    paired = {r: {m: float(np.mean([a[m] - b[m] for a, b in
                                    zip(per_fold[r], per_fold[ROW_TRAINED])]))
                  for m in METRICS}
              for r in rows if r != ROW_TRAINED}
    return agg, paired


def resolve_raw(paths, raw_cache, gene_list_path=None):
    """Raw per-modality embeddings for these checkpoint paths (cached)."""
    paths = [str(ROOT / "data" / Path(p).name) for p in paths]
    key = (tuple(paths), gene_list_path)
    if key not in raw_cache:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise SystemExit("ERROR: missing embedding files:\n  " +
                             "\n  ".join(missing))
        print("  loading raw embeddings:")
        for p in paths:
            print(f"    {Path(p).name}")
        raw, gene_to_idx, idx_to_gene = load_raw_multimodal(paths,
                                                            gene_list_path)
        order = [idx_to_gene[i] for i in range(len(idx_to_gene))]
        print(f"    {len(gene_to_idx):,} genes, dims {[t.shape[1] for t in raw]}")
        raw_cache[key] = (raw, gene_to_idx, order)
    return raw_cache[key]


def embedding_content_problems(ckpts, first, gene_order, data_dir):
    """Why the embedding files this script loads (data_dir/<basename of each
    file the combo was trained on>) are not the trained ones, one line each
    ([] = no evidence they differ).

    The CONTENT guard eval_finetuning.py, predict.py and the pair export
    apply (train.py's embeddings_sha256): a same-path rebuild of the
    embeddings keeps the basenames and the gene order, and the as-trained
    self-check sees it only when it moves a fold's metric by more than
    --selfcheck_tol. Every fold checkpoint is compared (`first` is ckpts[0],
    already loaded), the files are hashed once, and a legacy checkpoint
    without the key is not refused."""
    ckpt_inputs = []
    for p in ckpts:
        c = (first if p == ckpts[0] else
             torch.load(p, map_location="cpu", weights_only=False))
        ckpt_inputs.append((p, c.get("gene_order"), c.get("embeddings_paths"),
                            c.get("embeddings_sha256")))
        del c
    return checkpoint_input_problems(
        ckpt_inputs, [str(Path(data_dir) / Path(p).name)
                      for p in first["embeddings_paths"]], gene_order)


def run_cv(cv, model_dir, raw_cache, shared_genes, out_dir=None,
           cv_dir_name="no_cellline_%s", save_matrices=False, verbose=True,
           device="cpu", model_name=None, selfcheck_tol=1e-3):
    """Ablate one CV. Returns a result dict whose "status" is ok,
    not_applicable, missing, incomplete or selfcheck_failed."""
    model_dir = Path(model_dir)
    mdir = Path(out_dir) / (cv_dir_name % cv) if out_dir is not None else None
    # The clear below deletes checkpoints/, the matrices and eval_finetuning_*/
    # of mdir: pointed at the combo itself it would destroy what it ablates.
    if mdir is not None and mdir.resolve() == model_dir.resolve():
        raise SystemExit(f"ERROR: --out_dir/--cv_dir_name resolve to the "
                         f"source combo {model_dir} for {cv}; refusing to "
                         f"clear it (pick another --cv_dir_name)")
    if mdir is not None:
        clear_previous_outputs(mdir, cv)
    base = {"cv_type": cv, "source_model_dir": str(model_dir),
            "model_name": model_name,
            "graded_row": f"{model_name}_nocell" if model_name else None}
    ckpts = sorted((model_dir / "checkpoints").glob("fold_*_best.pt"))
    if not model_dir.is_dir() or not ckpts:
        print(f"  {cv}: no checkpoints under {model_dir} — skipping")
        return {**base, "status": "missing"}
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    why = not_applicable_reason(first["model_state_dict"])
    if why is not None:
        print(f"  {cv}: SKIP, not applicable ({why})")
        return {**base, "status": "not_applicable", "reason": why}
    require_combo_fold_stamp(model_dir, cv)
    cfg = first.get("config", {})
    raw, gene_to_idx, gene_order = resolve_raw(first["embeddings_paths"],
                                               raw_cache,
                                               cfg.get("gene_list_path"))
    if list(map(str, first.get("gene_order") or gene_order)) != gene_order:
        raise SystemExit(
            f"ERROR: {ckpts[0]} was trained on a different gene order than "
            f"the embedding files now in data/ give; its preprocessing "
            f"transform and weights would address the wrong rows.")
    problems = embedding_content_problems(ckpts, first, gene_order,
                                          ROOT / "data")
    if problems:
        raise SystemExit(
            f"ERROR: the embedding files in {ROOT / 'data'} are not the ones "
            f"{model_dir} was trained on:\n  " + "\n  ".join(problems) +
            "\nRetrain the combo on the current files (or restore the ones "
            "it was trained on).")
    del first
    # eval_finetuning.py refuses a model dir without config.json, so the
    # ablated dir gets the combo's own (train.py's vars(args)) plus the
    # ablation provenance.
    combo_cfg_path = model_dir / "config.json"
    combo_cfg = (json.loads(combo_cfg_path.read_text())
                 if combo_cfg_path.exists() else None)
    ref = combo_fold_reference(model_dir)
    if not ref:
        print(f"  WARNING: {model_dir / 'results.json'} has no fold_metrics; "
              f"the as-trained self-check cannot run")

    # shared fold index -> row of the model's embedding matrix
    smap = np.array([gene_to_idx.get(g, -1) for g in shared_genes], dtype=np.int64)
    n_unmapped = int((smap < 0).sum())
    if verbose:
        print(f"  gene map: {len(shared_genes) - n_unmapped:,}/{len(shared_genes):,} "
              f"shared genes present in the model's universe"
              + (f" ({n_unmapped} missing)" if n_unmapped else ""))

    rows = None
    per_fold = {}
    folds_done, folds_skipped, check = [], [], []
    residual = []
    ck_dir = None
    if mdir is not None:
        ck_dir = mdir / "checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs = sorted((FOLDS / cv).glob("fold_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    model = None
    for fd in fold_dirs:
        k = int(fd.name.split("_")[1])
        ckpt_path = model_dir / "checkpoints" / f"fold_{k}_best.pt"
        if not ckpt_path.exists():
            print(f"    fold {k}: no checkpoint — skipped")
            folds_skipped.append(k)
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        why = not_applicable_reason(ck["model_state_dict"])
        if why is not None:
            raise SystemExit(f"ERROR: {ckpt_path}: {why}, unlike fold 0")

        # Rebuild this fold's feature space from the transform stored WITH the
        # checkpoint. The PCA is refit per fold, so using another fold's
        # transform would silently score in the wrong basis.
        E = torch.as_tensor(
            apply_multimodal_transform(raw, ck["preprocessing_transform"]),
            dtype=torch.float32).to(device)
        # Architecture from the WEIGHTS (kernel, tau form, c_h, LayerNorm,
        # ...), strict load; eval mode (dropout off).
        model = load_multicell_model(str(ckpt_path), device=device)
        model.eval()
        if rows is None:
            residual = residual_head_terms(model)
            rows = [ROW_TRAINED, ROW_REMOVED, ROW_BIAS0]
            if model.head_proj_bias is not None:
                rows += [ROW_HPB_MEAN, ROW_HPB0]
            per_fold = {r: [] for r in rows}

        tp = np.load(fd / "test_pos.npy").astype(np.int64)
        tn = np.load(fd / "test_neg.npy").astype(np.int64)
        pairs = np.vstack([tp, tn])
        y = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
        ri, rj = smap[pairs[:, 0]], smap[pairs[:, 1]]
        keep = (ri >= 0) & (rj >= 0)
        if not keep.all():
            print(f"    fold {k}: dropping {int((~keep).sum()):,} edges whose "
                  f"genes are absent from the model's universe")
        ri, rj, yk, pk = ri[keep], rj[keep], y[keep], pairs[keep]

        saved_emb = model.cell_emb.weight.detach().clone()
        saved_bias = model.scoring_bias.detach().clone()
        saved_hpb = (model.head_proj_bias.detach().clone()
                     if model.head_proj_bias is not None else None)

        res = {ROW_TRAINED: metrics(yk, score_pairs(model, E, ri, rj))}

        model.cell_emb.weight.data.zero_()
        s_removed = score_pairs(model, E, ri, rj)
        res[ROW_REMOVED] = metrics(yk, s_removed)
        # --- persist the ablated model --------------------------------------
        # Saved in the SAME checkpoint schema as train.py writes, with the
        # preprocessing_transform / gene_order / embeddings_paths carried over
        # verbatim, next to a config.json (the combo's, plus provenance), so
        # eval_finetuning.py --per_head (load_multicell_model) can load these
        # directly; predict.py refuses every SiameseSLMultiCell checkpoint.
        # Only cell_emb is different (all zeros); `config` records the
        # provenance so an ablated checkpoint can never be mistaken for a
        # trained one.
        if ck_dir is not None:
            abl_sd = {n: v.detach().cpu().clone()
                      for n, v in model.state_dict().items()}
            provenance = {
                "variant": ROW_REMOVED,
                "source_checkpoint": _rel(ckpt_path),
                "note": "cell_emb zeroed post hoc; NOT retrained. "
                        "scoring_bias and any head_proj_bias are left as "
                        "trained.",
            }
            abl_cfg = dict(ck.get("config", cfg))
            abl_cfg["cell_line_ablation"] = provenance
            torch.save({
                "epoch": ck.get("epoch"),
                "model_state_dict": abl_sd,
                "metrics": res[ROW_REMOVED],
                "config": abl_cfg,
                "input_dim": ck["input_dim"],
                "preprocessing_transform": ck["preprocessing_transform"],
                "gene_order": ck["gene_order"],
                "embeddings_paths": ck["embeddings_paths"],
                # Carried too, or the ablated checkpoint would lose the CONTENT
                # guard (eval_finetuning.checkpoint_input_problems /
                # predict.py) and read as a legacy one: a same-path rebuild of
                # the embeddings would then pass every remaining check.
                "embeddings_sha256": ck.get("embeddings_sha256"),
            }, ck_dir / f"fold_{k}_best.pt")
            # One per CV (identical for every fold): names the source combo dir.
            dir_cfg = dict(combo_cfg if combo_cfg is not None else cfg)
            dir_cfg["cell_line_ablation"] = {
                "variant": provenance["variant"],
                "source_model_dir": _rel(model_dir),
                "note": provenance["note"],
            }
            with open(mdir / "config.json", "w") as f:
                json.dump(dir_cfg, f, indent=2, default=str)

        # --- optional dense matrix of the cell_emb=0 model, for grade.sh ----
        mat_err = None
        if save_matrices:
            n = len(shared_genes)
            have = np.where(smap >= 0)[0]
            S_t = build_any_score_matrix(
                model, E[torch.as_tensor(smap[have], device=E.device)])
            S = S_t.cpu().numpy()
            del S_t
            if not np.isfinite(S).all():
                raise SystemExit(f"ERROR: fold {k}: non-finite entries in the "
                                 f"cell_emb=0 score matrix")
            full = np.full((n, n), np.float32(S.min()), dtype=np.float32)
            full[np.ix_(have, have)] = S
            del S
            # The matrix must reproduce the forward pass on the scored test
            # edges (index space, weights and state all in one check); the
            # bound follows train.py's export check.
            mat_err = float(np.abs(full[pk[:, 0], pk[:, 1]] - s_removed).max())
            tol = max(1e-3, 1e-5 * float(np.abs(s_removed).max()))
            if mat_err > tol:
                raise SystemExit(
                    f"ERROR: fold {k}: the cell_emb=0 matrix disagrees with "
                    f"the forward pass on the test edges (max |diff| "
                    f"{mat_err:.3e} > {tol:.1e}); not exported")
            np.save(mdir / f"fold_{k}_{cv}_predictions.npy", full)
            del full
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.scoring_bias.data.zero_()
        res[ROW_BIAS0] = metrics(yk, score_pairs(model, E, ri, rj))
        model.scoring_bias.data.copy_(saved_bias)
        if saved_hpb is not None:
            model.head_proj_bias.data.copy_(
                saved_hpb.mean(dim=0, keepdim=True).expand_as(saved_hpb))
            res[ROW_HPB_MEAN] = metrics(yk, score_pairs(model, E, ri, rj))
            model.head_proj_bias.data.zero_()
            res[ROW_HPB0] = metrics(yk, score_pairs(model, E, ri, rj))
            model.head_proj_bias.data.copy_(saved_hpb)
        model.cell_emb.weight.data.copy_(saved_emb)

        # --- self-check: as-trained vs the combo's own test metrics ---------
        fc = {"fold": k, "n": int(len(yk)), "pi": float(yk.mean())}
        if k in ref:
            diffs = {m: (abs(res[ROW_TRAINED][m] - float(ref[k][m]))
                         if ref[k][m] is not None else None) for m in METRICS}
            fc["reference"] = ref[k]
            fc["abs_diff"] = diffs
            fc["ok"] = all(d is not None and d <= selfcheck_tol
                           for d in diffs.values())
        else:
            fc["ok"] = None  # no reference for this fold
        if mat_err is not None:
            fc["matrix_vs_forward_max_abs"] = mat_err
        check.append(fc)
        for r in rows:
            per_fold[r].append({"fold": k, **res[r]})
        folds_done.append(k)
        if verbose:
            flag = {True: "self-check ok", False: "SELF-CHECK MISMATCH",
                    None: "no self-check reference"}[fc["ok"]]
            print(f"    fold {k}: n={len(yk):,} pi={yk.mean():.4f}  "
                  + "  ".join(f"{r}: AUPRG {res[r]['auprg']:+.4f}"
                              for r in rows[:2]) + f"  [{flag}]")
        del E, ck

    if not folds_done:
        return {**base, "status": "missing"}
    failed = [c["fold"] for c in check if c["ok"] is False]
    status = ("selfcheck_failed" if failed else
              "incomplete" if folds_skipped else "ok")
    if mdir is not None and save_matrices:
        if status == "ok" and write_model_params is not None:
            # evaluate_model.py refuses to grade matrices with no fold_stamp,
            # and a stamp MISMATCH is always fatal — so this records the fold
            # set these matrices were scored against, which
            # require_combo_fold_stamp has checked is the one the source combo
            # trained on.
            write_model_params(mdir, model, folds_dir=str(FOLDS), cv_type=cv,
                               source_combo=str(model_dir),
                               **carried_training_params(model_dir))
            print(f"    wrote {mdir / 'model_params.json'}")
        elif status != "ok":
            # Nothing gradeable may survive a failed or partial CV.
            for p in mdir.glob(f"fold_*_{cv}_predictions.npy"):
                p.unlink()
            print(f"    {status}: removed this CV's matrices so grade.sh "
                  f"has nothing to grade")
    agg, paired = summarize(per_fold, rows)
    out = {
        **base,
        "status": status,
        "kind": "post_hoc_cell_line_ablation",
        "kernel": (combo_cfg or cfg).get("kernel", "pd"),
        "cell_lines": (combo_cfg or cfg).get("cell_lines", "per_head"),
        "exported_row": ROW_REMOVED,
        "rows": rows,
        "residual_head_terms": residual,
        "n_folds": len(folds_done),
        "folds": folds_done,
        "folds_skipped": folds_skipped,
        "per_fold": per_fold,
        "agg": agg,
        "paired": paired,
        "self_check": {
            "reference": "source model_dir/results.json fold_metrics[k] "
                         "(auroc_any, aupr_any, auprg_any)",
            "tol": selfcheck_tol,
            "ok": (not failed) and all(c["ok"] is True for c in check),
            "failed_folds": failed,
            "per_fold": check,
        },
    }
    if mdir is not None:
        (mdir / "results.json").write_text(json.dumps(out, indent=2))
        print(f"    wrote {mdir / 'results.json'}")
    return out


def parse_model_dirs(specs):
    """--model_dir CV=DIR entries -> {cv: Path}."""
    out = {}
    for s in specs or []:
        cv, sep, d = s.partition("=")
        if not sep or cv not in ("cv1", "cv2", "cv3") or not d:
            raise SystemExit(f"--model_dir wants CV=DIR with CV in cv1..cv3; "
                             f"got {s!r}")
        out[cv] = Path(d)
    return out


def print_report(results):
    print("=" * 78)
    print("POST-HOC CELL-LINE ABLATION — mean (population sd) over folds")
    print("=" * 78)
    for cv, r in results.items():
        if r["status"] in ("missing", "not_applicable", "error"):
            print(f"\n{cv}: {r['status']}")
            continue
        print(f"\n{cv}  ({r['n_folds']} folds, status {r['status']})")
        print(f"  {'row':32s} {'AUROC':>16s} {'AP':>16s} {'AUPRG':>16s}")
        for row in r["rows"]:
            a = r["agg"][row]
            print(f"  {row:32s} " + " ".join(
                f"{a[m][0]:+.4f} ({a[m][1]:.3f})".rjust(16)
                for m in METRICS))
        print(f"  {'-' * 84}")
        for row in r["rows"][1:]:
            d = r["paired"][row]
            print(f"  {('Δ ' + row):32s} " + " ".join(
                f"{d[m]:+.4f}".rjust(16) for m in METRICS))
        print("  (Δ is the mean paired per-fold change vs as-trained)")
        if r["residual_head_terms"]:
            print("  Note: the heads still differ after cell_emb=0 through "
                  + ", ".join(r["residual_head_terms"]) + ",")
            print("        so cell_emb=0 does not make them identical and the "
                  "+bias=0 row can differ")
            print("        from it. The +head_proj_bias rows remove c_h's head "
                  "identity (=mean) or c_h")
            print("        itself (=0).")
        else:
            print("  Note: the +bias=0 row is IDENTICAL to cell_emb=0 by "
                  "construction: with cell_emb")
            print("        zero every head scores a pair alike, so "
                  "max-over-heads adds the largest")
            print("        bias to every pair — a constant shift, and all three "
                  "metrics are rank-based.")
        sc = r["self_check"]
        print(f"  self-check vs {r['source_model_dir']}/results.json: "
              + ("ok" if sc["ok"] else
                 f"FAILED on folds {sc['failed_folds']}" if sc["failed_folds"]
                 else "incomplete (no reference for some folds)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cv_type", choices=["cv1", "cv2", "cv3"], action="append",
                    help="repeatable; default all three (or the --model_dir "
                         "CVs)")
    ap.add_argument("--model_dir", action="append", metavar="CV=DIR",
                    help="repeatable: the combo dir to ablate for CV (e.g. "
                         "the content of results/<prefix>latest_combo_cv1.txt;"
                         " relative to the cwd). Overrides --combo for CV.")
    ap.add_argument("--combo", default=COMBO,
                    help="combo directory name under siamese_sl/results with "
                         "%%s for the CV, for CVs without --model_dir "
                         "(default: the published modality set, "
                         "%(default)s)")
    ap.add_argument("--json_out", help="also write every CV's result here")
    ap.add_argument("--out_dir", default=str(HERE / "models"),
                    help="where to save the ablated checkpoints, matrices and "
                         "per-CV results.json (default: cell_line_ablation/"
                         "models)")
    ap.add_argument("--cv_dir_name", default="no_cellline_%s",
                    help="per-CV directory under --out_dir, %%s = the CV "
                         "(default %(default)s)")
    ap.add_argument("--no_save", action="store_true",
                    help="do not save anything (nothing under --out_dir is "
                         "written or deleted)")
    ap.add_argument("--save_matrices", action="store_true",
                    help="also export the cell_emb=0 fold_<k>_<cv>_"
                         "predictions.npy so grade.sh / evaluate_model.py can "
                         "grade it. ~313 MB per fold (8844^2 float32). "
                         "Written under --out_dir, so it cannot be combined "
                         "with --no_save.")
    ap.add_argument("--model_name",
                    help="the sl_comparison row of the source combo (e.g. "
                         "siamese_sl, siamese_sl_bilinear_per_head); recorded "
                         "in results.json, whose graded row is <name>_nocell")
    ap.add_argument("--selfcheck_tol", type=float, default=1e-3,
                    help="max |as-trained - combo results.json| per metric "
                         "and fold (default %(default)s)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.no_save and args.save_matrices:
        ap.error("--save_matrices writes under --out_dir; it cannot be "
                 "combined with --no_save")
    if "%s" not in args.combo:
        ap.error(f"--combo must contain %s for the CV; got {args.combo!r}")
    if "%s" not in args.cv_dir_name:
        ap.error(f"--cv_dir_name must contain %s; got {args.cv_dir_name!r}")
    explicit = parse_model_dirs(args.model_dir)
    cvs = args.cv_type or (sorted(explicit) if explicit
                           else ["cv1", "cv2", "cv3"])
    print(f"device: {args.device}")

    shared_genes = load_shared_genes()
    raw_cache = {}
    results = {}
    plan = [(cv, Path(explicit.get(cv, SIAMESE / "results" / (args.combo % cv))))
            for cv in cvs]
    mdirs = {}
    if not args.no_save:
        # EVERY requested CV's previous outputs go before ANY CV runs: run_cv
        # clears only the CV it reaches, so a hard stop in one CV would leave
        # the later CVs' old matrices and fold stamp for grade.sh. The
        # out == source refusal comes first, for every CV, so nothing is
        # cleared when one of them would be a combo.
        mdirs = {cv: Path(args.out_dir) / (args.cv_dir_name % cv)
                 for cv, _ in plan}
        for cv, model_dir in plan:
            if mdirs[cv].resolve() == model_dir.resolve():
                raise SystemExit(f"ERROR: --out_dir/--cv_dir_name resolve to "
                                 f"the source combo {model_dir} for {cv}; "
                                 f"refusing to clear it (pick another "
                                 f"--cv_dir_name)")
        for cv, _ in plan:
            clear_previous_outputs(mdirs[cv], cv)
    for cv, model_dir in plan:
        print(f"=== {cv}: {model_dir} ===")
        try:
            results[cv] = run_cv(cv, model_dir, raw_cache, shared_genes,
                                 out_dir=None if args.no_save else args.out_dir,
                                 cv_dir_name=args.cv_dir_name,
                                 save_matrices=args.save_matrices,
                                 device=args.device,
                                 model_name=args.model_name,
                                 selfcheck_tol=args.selfcheck_tol)
        except (SystemExit, Exception) as e:  # noqa: BLE001
            # One CV's hard stop (a stale fold stamp, a matrix that is not the
            # forward pass, an unreadable checkpoint, an OOM) must not cost the
            # CVs after it. Nothing gradeable of the failed CV survives and
            # the exit status is 1. KeyboardInterrupt is not caught.
            if not isinstance(e, SystemExit):
                traceback.print_exc()
            print(f"  {cv}: FAILED ({type(e).__name__}: {e})")
            if cv in mdirs:
                for q in [*mdirs[cv].glob(f"fold_*_{cv}_predictions.npy"),
                          mdirs[cv] / "model_params.json"]:
                    q.unlink(missing_ok=True)
            results[cv] = {"cv_type": cv, "source_model_dir": str(model_dir),
                           "model_name": args.model_name, "status": "error",
                           "error": f"{type(e).__name__}: {e}"}
        print("")

    print_report(results)
    if not args.no_save:
        print(f"\nablated checkpoints -> {args.out_dir}/"
              f"{args.cv_dir_name % '<cv>'}/checkpoints/")
        if args.save_matrices:
            print("grade the matrices with cell_line_ablation/grade.sh")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    bad = {cv: r["status"] for cv, r in results.items()
           if r["status"] not in ("ok", "not_applicable")}
    if bad:
        print(f"\nERROR: not every CV was ablated: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
