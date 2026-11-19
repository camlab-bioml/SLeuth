#!/bin/bash
#SBATCH --job-name=ablation_summary
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm/logs/ablation_summary_%j.out
#SBATCH --error=slurm/logs/ablation_summary_%j.err
# ============================================================================
# Leave-One-Category-Out Ablation — summary
# ============================================================================
# Builds <arm root>/ablation_summary.csv and prints a per-(variant, CV) table,
# for the job's PCA arm (PCA_ARM; <arm root> = its RESULTS_ROOT, config.conf:
# results/ for arm none, results/pca_<arm>/ otherwise; every row's pca_arm
# column says which). Every variant in SIAMESE_VARIANTS (config.conf) is
# paired with ITS OWN reference combo: <arm root>/<prefix>ablate_no_<cat>_<cv>
# against the combo named by <arm root>/<prefix>latest_combo_<cv>.txt (prefix
# "" for the baseline pd:per_head); a (variant, cv) without that marker has no
# reference and is skipped.
#
# Each row pairs the marginal ablation with the standalone score of the
# category that was dropped. Reading only one of the two is how ablation
# tables mislead: under redundancy a category can be individually strong and
# still cost nothing to remove, and the LOCO column alone would report it as
# worthless. The two columns together separate "uninformative" from "already
# covered by the others".
#
# CSV: one dropped="none" row per (variant, CV) carries the reference itself;
# every row has the fold mean and population sd (<metric>, <metric>_std), the
# reference's (ref_<metric>_mean / _std) and the delta. The AUPRG delta also
# carries its paired spread over ablation fold k minus reference fold k (the
# same shared split, matched by fold_metrics[].fold): population sd
# (delta_auprg_any_paired_sd, as paper_tables' tab_loco delta_paired_sd),
# standard error (delta_auprg_any_se, sample sd / sqrt(n)) and n_paired. A
# delta beyond +-0.005 is read as HARMFUL (positive) or contributes
# (negative) only when it also exceeds 2 SE, else as within fold noise, and
# as n/a with fewer than 2 paired folds. An ablation whose
# run_token.txt is not this submission's RUN_TOKEN (config.conf; a task that
# never started keeps its previous run's directory) is a MISMATCH with no
# delta, like one that is not reference-minus-one. Run this job in the same
# submission as the ablation array, or (direct sbatch) export the RUN_TOKEN the
# array was submitted with: an array sbatch'ed without one gave each task its
# own token, and then no row gets a delta (recipe in run_ablation.sh's header).
# `solo` is the dropped category's winner trained alone: under the variant
# itself when BESTCAT_EXTRA_VARIANTS=1 produced that run in the same Step-1
# array as the selection run, else under SELECTION_VARIANT (solo_variant);
# blank when the selection run's run_token.txt is not the one Step 2 recorded
# in best_per_category.json (_selection_meta.run_token). Those single-embedding
# runs are read where Step 2 ranked them (_selection_meta.step1_root, else the
# arm's STEP1_ROOT): arms none / joint and per_modality / both share them.
# ============================================================================

set -e
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_ablation.conf"

# Clean the previous table first (as run_embedding_benchmark_summary.sh does):
# a run that builds no rows (every ablation task failed, no current-winner
# reference) or stops on a missing best_per_category.json must not leave the
# last run's deltas looking like this run's.
rm -f "$RESULTS_ROOT/ablation_summary.csv"

# CV_TYPES is a bash array; hand it to python as JSON rather than re-deriving
# the list there, so adding a CV type in config.conf reaches the summary too.
CV_TYPES_JSON=$(printf '%s\n' "${CV_TYPES[@]}" | $PYTHON_PATH -c "import sys,json;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")
export CV_TYPES_JSON
# The per-embedding runs live in <Step-1 root>/pcavar<PCA_VARIANCE>_<type>_<cv>, and
# PCA_VARIANCE is overridable via BEST_CAT_PCA_VARIANCE. Hardcoding "0.8" here
# blanked the whole `solo` column the moment anyone used that documented
# override — silently, since a missing directory just yields None.
export PCA_VARIANCE
# [[variant, results-dir prefix, single-modality prefix], ...] — each variant's
# rows are read against that variant's own reference combo.
VARIANTS_JSON="[$(for v in "${SIAMESE_VARIANTS[@]}"; do printf '["%s", "%s", "%s"], ' "$v" "$(siamese_variant_prefix "$v")" "$(bestcat_solo_prefix "$v")"; done | sed 's/, $//')]"
export VARIANTS_JSON
export SELECTION_VARIANT RUN_TOKEN
# This job's PCA arm and its roots (config.conf). Exported here only, for the
# quoted heredoc below; config.conf itself never exports the roots.
export PCA_ARM RESULTS_ROOT STEP1_ROOT
PCA_LABEL=$(pca_arm_label "$PCA_ARM")
export PCA_LABEL

$PYTHON_PATH - <<'PYSUM'
import json, math, os, csv, statistics as st
from arch_signature import signature_diff
import fold_selection as fsel

CVS = json.loads(os.environ.get("CV_TYPES_JSON", '["cv1","cv2","cv3"]'))
VARIANTS = json.loads(os.environ.get("VARIANTS_JSON", '[["pd:per_head", "", ""]]'))
SELECTION_VARIANT = os.environ.get("SELECTION_VARIANT", "pd:per_head")
RUN_TOKEN = os.environ.get("RUN_TOKEN", "")
PCA_ARM = os.environ.get("PCA_ARM", "none")
RESULTS = os.environ.get("RESULTS_ROOT", "results")
METRICS = [("auprg_any", "AUPRG"), ("aupr_any", "AP"), ("auroc_any", "AUROC")]
FIELDS = (["pca_arm", "variant", "cv", "dropped", "winner", "n_modalities",
           "solo_auprg", "solo_auprg_std", "solo_variant"]
          + [c for k, _ in METRICS for c in (k, f"{k}_std")]
          + [f"delta_{k}" for k, _ in METRICS]
          + ["delta_auprg_any_paired_sd", "delta_auprg_any_se", "n_paired"]
          + [c for k, _ in METRICS for c in (f"ref_{k}_mean", f"ref_{k}_std")]
          + ["verdict", "reference"])

def foldmean(path, key):
    try:
        fm = json.load(open(path))["fold_metrics"]
        vals = [f[key] for f in fm if key in f]
        return (st.mean(vals), st.pstdev(vals)) if vals else (None, None)
    except Exception:
        return (None, None)

def fold_vals(path, key):
    """{fold_metrics[].fold: key} of a results.json; an entry without an int
    fold id or a finite value is left out."""
    try:
        fm = json.load(open(path))["fold_metrics"]
    except Exception:
        return {}
    out = {}
    for f in fm:
        try:
            v = float(f[key])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(f.get("fold"), int) and math.isfinite(v):
            out[f["fold"]] = v
    return out

def paired_spread(path, ref, key="auprg_any"):
    """(population sd, SE, n) of the fold-by-fold (ablation - reference) key.

    Both runs train on the same shared folds, so the per-split swing they
    share cancels in the paired difference but not in either run's own sd.
    sd is ddof=0 (paper_tables' tab_loco delta_paired_sd); SE = sample sd
    (ddof=1) / sqrt(n), None below 2 pairs.
    """
    a, b = fold_vals(path, key), fold_vals(ref, key)
    d = [a[f] - b[f] for f in sorted(set(a) & set(b))]
    return (st.pstdev(d) if d else None,
            st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None, len(d))

PCA_VARIANCE = os.environ.get("PCA_VARIANCE", "0.8")

def fold_stamp(result_json):
    """model_params.json fold_stamp next to a results.json, or None."""
    mp = os.path.join(os.path.dirname(result_json), "model_params.json")
    try:
        return json.load(open(mp)).get("fold_stamp")
    except Exception:
        return None

def stamp_content(result_json):
    """fold_stamp minus folds_dir — the fold SET, not the path to it.

    model_params.compare_fold_stamp (the rule every other consumer uses, incl.
    paper_tables' tab:loco gate) deliberately ignores folds_dir: the same folds
    are legitimately reached by different spellings of SHARED_FOLDS, and a path
    difference is not a data difference. Comparing the raw dicts made an
    ablation run with SHARED_FOLDS spelled differently a fold_stamp MISMATCH
    with no delta, while paper_tables published that row's delta anyway.
    """
    s = fold_stamp(result_json)
    return None if s is None else {k: v for k, v in s.items()
                                   if k != "folds_dir"}

def run_token(result_json):
    """run_token.txt next to a results.json, or None."""
    try:
        with open(os.path.join(os.path.dirname(result_json), "run_token.txt")) as f:
            return f.read().strip()
    except OSError:
        return None

def names(paths):
    return [os.path.basename(x) for x in paths]

def solo(cv, etype, variant, solo_prefix):
    """(mean, sd, trained-as variant) of etype alone on cv, or Nones."""
    sel = f"{STEP1}/pcavar{PCA_VARIANCE}_{etype}_{cv}/results.json"
    # Only the Step-1 array the winners were selected from: Step 2 refuses a
    # foreign token on SELECTION_CV alone, so a cv2/cv3 task that never started
    # would otherwise fill `solo` (and the "redundant" verdict) from an old run.
    if not os.path.exists(sel) or (SEL_TOKEN and run_token(sel) != SEL_TOKEN):
        return None, None, None
    own = f"{STEP1}/{solo_prefix}pcavar{PCA_VARIANCE}_{etype}_{cv}/results.json"
    # The variant's own run only when it came from the same Step-1 array as the
    # selection run; an older directory would pair two different runs.
    if (solo_prefix and os.path.exists(own)
            and run_token(own) is not None and run_token(own) == run_token(sel)):
        return (*foldmean(own, "auprg_any"), variant)
    return (*foldmean(sel, "auprg_any"), SELECTION_VARIANT)

try:
    winners = json.load(open(os.path.join(RESULTS, "best_per_category.json")))
except Exception:
    print(f"ERROR: {RESULTS}/best_per_category.json missing (PCA arm {PCA_ARM}) "
          f"— run the best-per-category pipeline first.")
    raise SystemExit(1)
# The token of the Step-1 array Step 2 ranked (absent in a pre-token json), and
# where those runs are (absent before 2026-10-20: the arm's own STEP1_ROOT).
SEL_TOKEN = (winners.get("_selection_meta") or {}).get("run_token")
# A per_fold selection (SELECTION_MODE=per_fold, 2026-11-12): each CV's
# combo is the union of its folds' winners, each fold on its own, and each
# ablation every fold's own set minus the category's pick
# (fold_selection.combo_spec); "winner" is then the category's embeddings
# over the CV's folds (most-picked first in the solo column's sense: `solo`
# is the most-picked one alone).
PER_FOLD = fsel.selection_mode(winners) == "per_fold"


def spec_problem(cfg, cv, drop=None):
    """Why cfg is not the per_fold selection's combo (drop=None) or its
    ablation of drop (None: it is)."""
    try:
        return fsel.run_matches_spec(cfg, fsel.combo_spec(winners, cv, drop))
    except (fsel.SelectionError, KeyError, TypeError, ValueError) as e:
        return f"{type(e).__name__}: {e}"


def winner_name(cv, cat):
    w = winners[cv][cat]
    return "/".join(w.get("types") or [w["type"]]) if PER_FOLD else w["type"]
STEP1 = ((winners.get("_selection_meta") or {}).get("step1_root")
         or os.environ.get("STEP1_ROOT", "results"))
print(f"PCA arm: {PCA_ARM} ({os.environ.get('PCA_LABEL', '')}); combos and "
      f"ablations in {RESULTS}/, single-embedding runs in {STEP1}/\n")

rows = []
n_solo_own = 0
for variant, prefix, solo_prefix in VARIANTS:
    for cv in CVS:
        # The reference is the combo the LAST combo job trained for this
        # (variant, cv): its marker, never a glob. That job deletes every
        # marker when it starts and writes one only on success, so a
        # best_cat_* dir without one (a failed train.py that still left a
        # results.json) is not a trained reference; run_ablation.sh skips it
        # too.
        marker = f"{RESULTS}/{prefix}latest_combo_{cv}.txt"
        if not os.path.exists(marker):
            print(f"[{variant} {cv}] no {marker} (the latest combo job did "
                  f"not train it); skipping.\n")
            continue
        ref = os.path.join(open(marker).read().strip(), "results.json")
        if not os.path.exists(ref):
            print(f"[{variant} {cv}] reference {ref} missing; skipping.\n")
            continue
        try:
            ref_cfg = json.load(open(ref))["config"]
            nref = len(ref_cfg["embeddings_paths"])
        except Exception as e:
            # Costs this (variant, cv) only, not every other row of the CSV.
            print(f"[{variant} {cv}] reference {ref} unreadable "
                  f"({type(e).__name__}: {e}); skipping.\n")
            continue
        base = {k: foldmean(ref, k)[0] for k, _ in METRICS}
        base_sd = {k: foldmean(ref, k)[1] for k, _ in METRICS}
        ref_cols = {c: v for k, _ in METRICS
                    for c, v in ((f"ref_{k}_mean", base[k]),
                                 (f"ref_{k}_std", base_sd[k]))}
        ref_stamp = stamp_content(ref)
        # The reference must be the combo of the CURRENT category winners;
        # otherwise every "minus one category" comparison is against a
        # different modality set.
        want = sorted(w["file"] for w in winners.get(cv, {}).values())
        if (spec_problem(ref_cfg, cv) is not None if PER_FOLD else
                sorted(names(ref_cfg["embeddings_paths"])) != want):
            print(f"[{variant} {cv}] reference {ref} was not trained on the "
                  f"current best_per_category.json winners; skipping.\n")
            continue
        if PER_FOLD:
            # the categories each fold trains on (embeddings_paths is their
            # union), as the ablations' num_modalities counts them
            nref = fsel.combo_spec(winners, cv)["n_per_fold"]

        print("=" * 96)
        print(f"[{variant}] {cv}   reference: full {nref}-category combo   " +
              "  ".join(f"{lab}={base[k]:.4f}" for k, lab in METRICS if base[k] is not None))
        print("=" * 96)
        print(f"  {'dropped':<13}{'n':>3}  " +
              "".join(f"{lab:>9}{'Δ':>9}" +
                      (f"{'SE':>9}" if k == "auprg_any" else "")
                      for k, lab in METRICS) +
              f"{'solo':>9}  read as")
        print("  " + "-" * 101)
        rows.append(dict(pca_arm=PCA_ARM, variant=variant, cv=cv,
                         dropped="none", winner="",
                         n_modalities=nref,
                         **{k: base[k] for k, _ in METRICS},
                         **{f"{k}_std": base_sd[k] for k, _ in METRICS},
                         **ref_cols, verdict="reference (full combo)",
                         reference=os.path.dirname(ref)))

        for cat in sorted(winners.get(cv, {})):
            p = f"{RESULTS}/{prefix}ablate_no_{cat}_{cv}/results.json"
            if not os.path.exists(p):
                print(f"  {cat:<13}  -   (not run)")
                continue
            try:
                d = json.load(open(p))
            except Exception as e:
                # A torn or truncated results.json costs this row only. Letting the
                # exception escape would abort before the CSV is written, throwing
                # away every ablation that DID succeed.
                print(f"  {cat:<13}  -   (unreadable: {e})")
                continue
            n = d.get("ablation", {}).get("num_modalities", nref - 1)
            # An ablation is only "the reference minus one category" if the
            # rest of its architecture / selection signature matches; if not
            # (e.g. --ablation-only with other ARCH_ARGS), no delta is reported.
            arch_diff = signature_diff(ref_cfg, d.get("config") or {})
            # ...and it must be exactly the reference's modalities minus this
            # category's winner, on the same fold set. A stale ablate_no_X from
            # an earlier combo (a worker that failed before its cleanup) has
            # the same architecture but a different modality set.
            if PER_FOLD:
                # ...each fold's own winners minus this category's pick
                _why = spec_problem(d.get("config") or {}, cv, cat)
                if _why:
                    arch_diff = dict(arch_diff, embeddings_paths=_why)
            else:
                drop_file = winners[cv][cat]["file"]
                want_mods = [m for m in names(ref_cfg["embeddings_paths"])
                             if m != drop_file]
                got_mods = names((d.get("config") or {}).get(
                    "embeddings_paths") or [])
                if got_mods != want_mods:
                    arch_diff = dict(arch_diff,
                                     embeddings_paths=(want_mods, got_mods))
            if ref_stamp != stamp_content(p):
                arch_diff = dict(arch_diff, fold_stamp="differs")
            # ...and it must be THIS submission's run: a task that never
            # started left its previous run's directory, which can pass every
            # check above (same config, same modality set, same folds).
            _tok = run_token(p)
            if _tok != RUN_TOKEN:
                arch_diff = dict(arch_diff, run_token=(
                    "missing" if _tok is None else f"{_tok} != {RUN_TOKEN}"))
            # The AUPRG delta's fold-by-fold spread; no delta, no spread.
            p_sd, p_se, n_pair = ((None, None, None) if arch_diff
                                  else paired_spread(p, ref))
            cells, deltas, vals = "", {}, {}
            for k, lab in METRICS:
                v, vsd = foldmean(p, k)
                vals[k], vals[f"{k}_std"] = v, vsd
                dv = ((v - base[k]) if (v is not None and base[k] is not None
                                         and not arch_diff) else None)
                deltas[k] = dv
                cells += f"{v:>9.4f}" if v is not None else f"{'--':>9}"
                cells += f"{dv:>+9.4f}" if dv is not None else f"{'--':>9}"
                if k == "auprg_any":
                    cells += (f"{p_se:>9.4f}" if p_se is not None
                              else f"{'--':>9}")
            s, s_sd, s_var = solo(cv, winners[cv][cat]["type"], variant,
                                  solo_prefix)
            n_solo_own += s_var == variant != SELECTION_VARIANT
            socell = f"{s:>9.4f}" if s is not None else f"{'--':>9}"

            # Interpretation. `dg` is the AUPRG delta from removing the
            # category, `p_se` its paired standard error: a delta beyond 0.005
            # is read as a direction only when it also exceeds 2 SE.
            dg = deltas.get("auprg_any")
            if arch_diff:
                verdict = ("MISMATCH vs reference (" +
                           ", ".join(sorted(arch_diff)) +
                           "): stale or not reference-minus-one, no delta")
            elif dg is None:
                verdict = ""
            elif abs(dg) > 0.005 and n_pair < 2:
                verdict = "n/a (fewer than 2 paired folds)"
            elif dg > 0.005 and dg > 2 * p_se:
                verdict = "HARMFUL — removing it helps"
            elif dg < -0.005 and -dg > 2 * p_se:
                verdict = "contributes"
            elif abs(dg) > 0.005:
                verdict = "within fold noise (|Δ| <= 2 SE)"
            elif s is not None and base["auprg_any"] is not None and s > base["auprg_any"]:
                verdict = "redundant (strong alone, free to drop)"
            else:
                verdict = "no measurable effect"
            rows.append(dict(pca_arm=PCA_ARM, variant=variant, cv=cv,
                             dropped=cat, winner=winner_name(cv, cat),
                             n_modalities=n, solo_auprg=s,
                             solo_auprg_std=s_sd, solo_variant=s_var,
                             **vals,
                             **{f"delta_{k}": deltas[k] for k, _ in METRICS},
                             delta_auprg_any_paired_sd=p_sd,
                             delta_auprg_any_se=p_se, n_paired=n_pair,
                             **ref_cols, verdict=verdict,
                             reference=os.path.dirname(ref)))
            print(f"  {cat:<13}{n:>3}  {cells}{socell}  {verdict}")
        print()

if rows:
    out = os.path.join(RESULTS, "ablation_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    print()
    print("Columns: Δ is the ablated score minus the full-combo reference, so a")
    print("POSITIVE Δ means the model scored higher without the category. SE is")
    print("the standard error of the AUPRG Δ over the paired folds (ablation fold")
    print("k minus reference fold k, the same shared split; sample sd / sqrt(n)).")
    print("A |Δ| above 0.005 reads as HARMFUL (Δ > 0) or contributes (Δ < 0)")
    print("only when it also exceeds 2 SE, else as within fold noise, and as n/a")
    print("with fewer than 2 paired folds; a smaller |Δ| is redundant when `solo`")
    print("beats the reference, else no measurable effect. The CSV adds the")
    print("paired population sd (delta_auprg_any_paired_sd) and n_paired.")
    print("`solo` is that category's winning embedding trained on its own, from the")
    print("best-per-category pass — no extra runs needed for it. It is scored")
    print("under the row's own variant only where BESTCAT_EXTRA_VARIANTS=1 trained")
    print(f"it ({n_solo_own} rows); otherwise under SELECTION_VARIANT")
    print(f"({SELECTION_VARIANT}), not under the row's kernel / cell-line pooling")
    print("(the CSV's solo_variant column says which).")
else:
    print(f"No ablation results found; {RESULTS}/ablation_summary.csv was not "
          f"written (the previous one was removed). Submit with: "
          f"PCA_ARMS={PCA_ARM} bash slurm/submit_pipeline.sh --ablation-only")
PYSUM
echo "Completed at: $(date)"
