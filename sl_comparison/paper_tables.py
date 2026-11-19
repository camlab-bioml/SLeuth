#!/usr/bin/env python3
"""
Every table and number of paper/main.tex that one benchmark run can compute,
with all SLeuth variants (SIAMESE_VARIANTS) wherever SLeuth appears: for ONE
PCA arm (per-arm mode, the default), or the PCA arms side by side
(--pca_arms_table, cross-arm mode).

The PCA arms (siamese_sl/slurm/config.conf, 2026-10-20): the whole siamese
pipeline runs once per PCA choice, none (no PCA), joint (PCA on the
concatenation of all modalities), per_modality (PCA within each modality) and
both (per_modality, then joint), and every table this script writes says
which arm it is (so do comparison.csv, the external_transfer/ CSVs, the
SLeuth combo rows, in selection_audit.pca_arm, and the selection it reads;
their *_nocell rows (zero_cell_emb.py records no arm), the markers, stale_*
and run directories are told apart by their directory alone; 2026-10-21:
this had the *_nocell rows name it too). Arm
none keeps the historical paths; every other arm writes the same names one
directory down (config.conf pca_arm_results / pca_arm_cmp):

    arm none:  siamese_sl/results/          sl_comparison/results/
    arm X:     siamese_sl/results/pca_X/    sl_comparison/results/pca_X/

The shared models' rows (SLGNN / NSF4SL / SLMGAE / DegreePrior) exist only in
sl_comparison/results/, and every arm reads them there. An arm's single-
embedding runs are its pair's: none and joint rank the native ones,
per_modality and both the per-modality-PCA ones (config.conf pca_step1_arm).

Run from the repo root after the whole DAG (sl_comparison/slurm/run_paper_tables.sh):

    python sl_comparison/paper_tables.py --pca_arm joint
    python sl_comparison/paper_tables.py --pca_arms_table \\
        --pca_arms none joint per_modality both

PER-ARM MODE. --pca_arm (default $PCA_ARM, else none) writes to --out_dir
(default: the arm's sl_comparison/results[/pca_<arm>]/paper_tables; previous
outputs of this script there are removed first):

  tab_tiers.csv        tab:tiers  AUROC / AP (the paper's "AUPR") / AUPRG mean (sd)
                                  per tier, SLeuth trainable parameters mean (sd)
                                  over folds, best-per-tier flags
  tab_solo.csv         tab:solo   single-modality collapsed test AUPRG per
                                  embedding x tier, per-category best flags
  tab_loco.csv         tab:loco   one category removed + the 'none' reference
  tab_cellline.csv     tab:cellline  cell_emb zeroed at inference vs as trained
                                  (+ the pooled variant as the retrained pairing)
  tab_external.csv     tab:external  zero-shot Spearman, every model
  tab_external_ft.csv  tab:external-ft  zero-shot / LP_last / Full (+ Scratch
                                        when the eval ran it), SLeuth only
  tab_sldb_agreement.csv  (reviewer: do the screens and SynLethDB agree?)
                       per screen x scope: AUROC of the screen target for
                       SynLethDB SL vs screened non-SL and Spearman(label,
                       target) on the pairs SynLethDB labels, with bootstrap
                       CIs (bootstrap_ci.py --part external), screen_role
  numbers.json         every other quantitative claim (see ITEMS below) plus the
                       verdict, the list of missing inputs, what is absent by
                       design and each table's status
  README.md            the verdict (the first line under README.md's title,
                       line 3), paper item -> file / column, run settings,
                       what is missing, what is absent by design

Which arm: every CSV's first column is pca_arm (the arm, on every row);
README.md's title (line 1) is '# Paper tables and numbers: PCA arm <arm>', and
its first Run settings bullet names the arm, its PCA (config.conf
pca_arm_label's wording, from $PCA_MODALITY_VARIANCE / $PCA_JOINT_VARIANCE /
$PCA_JOINT_DIM, else config.conf's values, else its defaults 0.8 / 0.8 /
unset), its roots, and its single-embedding runs' root and what those are
(the label the combo job recorded with them, _selection_meta.step1_pca,
else config.conf pca_step1_label's wording over --pca_arms); numbers.json
run.pca_arm / pca / pca_targets / pca_settings / sia_results / cmp_results /
step1_results / step1_pca say the same.
run.pca_settings is the PCA the combos actually trained with
(arch_signature.pca_settings of their config.json): combos trained with
another setting than those targets are one Missing entry (the label would not
describe them).

Sources of arm X (paths relative to --root; <sia> = siamese_sl/results[/pca_X],
<cmp> = sl_comparison/results[/pca_X]; <step1> = siamese_sl/<the step1_root
best_per_category.json's _selection_meta records> (the Step-1 root its Step 2
ranked), else config.conf's owner rule: the results root of the first arm of
X's pair (none, joint) / (per_modality, both) that --pca_arms lists, else X's):
  <cmp>/comparison.csv (compare.py; a (cv, model) row absent from it falls
      back to its <model>_<cv>.json, and only to one evaluate_model.py
      wrote). The SLeuth rows (siamese_sl[_<tag>] and their _nocell rows:
      compare.py is_siamese_row) are read from <cmp>/<row>_<cv>.json only,
      every other model's from sl_comparison/results/<model>_<cv>.json only
      (for arm none the two are one directory), so another arm's rows of the
      same name never stand in. A comparison.csv whose pca_arm column names
      another arm is not read. A row whose recorded graded-on or trained-on
      fold fingerprints are not the current folds' is left out and listed
      (items.fold_provenance). compare.py writes the table FROM those JSONs,
      so a cell whose two copies disagree was re-graded after the table was
      written (only run_compare.sh rewrites it, while a standalone rerun
      rewrites the row JSON alone): it is left out and listed. So is a row of
      the DEFAULT table with no such JSON in its own directory at all (the
      state a standalone rerun leaves when it moves the previous row aside and
      then fails), a row over a partial fold set (summary.partial) and a
      REDUCED SLMGAE (--allow_missing_views / --allow_static_lp_mapping)
  <sia>/<prefix>latest_combo_<cv>.txt -> the combo (its text is siamese_sl-
      relative, results[/pca_X]/<prefix>best_cat_<names>_<cv>): results.json
      (fold_metrics[].total_params, fold_metrics[].input_dim), config.json;
      used only when it is a direct child of <sia>, its config.json is X's PCA
      (arch_signature.pca_arm: a marker naming another namespace, a hand copy
      or a combo trained under another arm is listed and not used) and its
      model_params.json fold_stamp matches the current folds. A combo that is
      not the run <cmp>/<siamese|row>_selected_<cv>.txt says was graded (a
      SIAMESE_RESULTS_* pin) is listed, and its parameter counts are not put
      on that row
  <sia>/best_per_category.json (the selection; one recording another
      _selection_meta.pca_arm, or a step1_root outside X's pair, is listed)
  <step1>/<solo prefix>pcavar<PCA>_<emb>_<cv>/ (single-modality runs;
      bestcat_solo_prefix: "" for SELECTION_VARIANT, var_pd_per_head__ for a
      non-selection pd:per_head, else the variant prefix). tab:solo also
      covers SELECTION_VARIANT when SIAMESE_VARIANTS omits it (Step 1 trains
      its runs either way); that variant has no combo here, so its
      combo_auprg_* / above_combo_* stay empty
  <sia>/<prefix>ablate_no_<cat>_<cv>/ (leave-one-category-out)
  <sia>/<prefix>cellline_ablation_<cv>/results.json and
      <cmp>/<row>_nocell_<cv>.json (cell-line ablation)
  <cmp>/external_transfer/{zero_shot,finetune}_summary.csv (one whose
      pca_arm column, external_transfer.py --pca_arm, names another arm is
      not read; one without the column is)
  shared by every arm: sl_comparison/folds/ (meta.json, genes.txt,
      <cv>/fold_<k>/*.npy); data/ pair files and .sources.tsv sidecars (paths
      from folds/meta.json), data/synlethkg/meta.json, the catalog's data/*.pt
      (native widths); siamese_sl/slurm/run_best_per_category.conf
      (EMB_CATALOG), run_ablation.conf (ABLATE_CATEGORIES) and
      run_generate_embeddings.conf (OPTIONAL_TYPES)
  --code_manifest (default $CODE_MANIFEST): code_manifest.py's sha256 of the
      code the run's jobs execute, written by the orchestrator at submission.
      The same files under --root are hashed again; any that changed,
      appeared or vanished since is one Missing entry (the jobs start over
      days, so they may have run different versions), and numbers.json
      run.code_drift lists them all. Without a manifest (a standalone run)
      this is only noted

A single-modality or ablation directory is tabulated only when it is from THIS
run: its run_token.txt equals the expected token, its config's kernel /
cell_lines / cv_type are the cell's, its model_params.json fold_stamp matches
the current folds, its arch_signature equals its variant's combo (minus the
post-concat PCA for the single-modality runs: no arm applies it to one
modality, while their per-modality PCA must be the combo's), and it was
trained on the same content of every embedding file it shares with that combo
(both results.json embeddings_sha256; a file rebuilt in place between the two
trainings differs). Otherwise the cell is marked stale and no value is
written. The expected token of a single-modality dir is the one
best_per_category.json's _selection_meta.run_token records (the Step-1 array
Step 2 ranked; checked even without RUN_TOKEN), else RUN_TOKEN; of an ablation
dir, --ablation_run_token, else RUN_TOKEN, else that same
_selection_meta.run_token (every orchestrator path ablates under the
selection's token). '' = unchecked, for either. The orchestrators give both
one token (submit_pipeline.sh --ablation-only reuses the selection's);
--ablation_run_token is for an ablation rerun under a token of its own, and is
CLI-only so that a leftover export cannot mark a fresh run's dirs stale. Every
sd is the population sd (ddof=0), as in the paper.

tab_cellline tabulates zero_cell_emb.py's rows (the graded 'removed' row and
its self-scored rows) only when its results.json is status ok with a passed
self-check over every fold, it ablated the latest_combo combo as that combo
is NOW (the directory it records, compared as a path so that arm none's and
arm X's combos of one name are told apart, and the per-fold test metrics it
copied from the combo's results.json still equal that file's: a combo
retrained in place keeps its directory name), and its as-trained row
reproduces the included comparison row (within 5e-3); otherwise they are
marked stale and listed.

tab_external / tab_external_ft are external_transfer.py's CSVs, which record
no fingerprint of the eval they were computed from. A SLeuth row whose
all_sldb_filtered Spearman (zero_shot_raw, LP_last, Full) is not the per-fold
mean its cv3 combo's eval_finetuning_<split>/ reports NOW (its
zero_shot_summary.csv / finetuned_summary.csv) was built before the combo was
retrained or the eval rerun: it is listed (rerun
sl_comparison/slurm/run_external_transfer.sh for the arm). The peers' rows
have no such second record and are not checked.

A bold, rank or 'best' is left empty (and listed) unless every row it is
decided over is present: a comparison against whatever survived is not the
paper's. So tab_tiers' best_<metric> for a tier with an in-paper row missing,
beats_peers_<metric> / items.margins leads / items.external X5 ranks unless
all three peers have a row, tab_external's best_matched unless every SLeuth
variant and peer has a matched row, tab_external_ft's best_method unless all
three columns are there, tab_solo's best_in_category_paper16 and the AB2
paper-16 mean of a category unless all its paper-16 members have a fresh row,
the AB2 / AB3 best category, paper-16 counts and 'at every tier' set unless
all 16 do, AB5's largest_drop unless every category's ablation does, a
variant's R5 PCA-width range unless every split has a width, I2's min_ratio
unless every tier has a ratio, and (cross-arm mode) tab_pca's best_arm unless
every requested arm's cell is ok and the PCA 2x2 effects unless all four arms'
are.

The paper prints tab:tiers, tab:solo and tab:external to 3 decimals, and its
bold (best_<metric>, best_in_category_paper16, best_matched) marks every row
that prints the best value: ties at 3 decimals are all bold. Every other flag,
rank and margin (best_in_category, beats_peers_*, best_method, the X5 ranks,
items.margins, tab_pca's best_arm) is decided on the exact values.

tab_solo / tab_loco carry train.py's self-scored auprg_any (as the paper's
tables do); --regrade additionally grades each fresh cell whose fold matrices
were kept with evaluate_model.py (the tab_tiers evaluator) into
<out_dir>/regraded/ and adds auprg_shared_mean / auprg_shared_sd.

Verdict (the last line printed, README.md's first line under the title,
numbers.json 'verdict'): PAPER COMPLETE only when nothing is listed under
Missing and the folds can be shown to be built from the current pair files /
sidecars / cell-line vocabulary; otherwise PAPER INCOMPLETE: <n> item(s)
(<the first three>...). It certifies the tables and numbers computed here
only: paper/main.tex is copied by hand and is not checked.
Absences the run's settings intend are listed apart, under 'Absent by design'
(numbers.json 'absent_by_design'), and do not make it incomplete: a catalog
embedding outside the paper's 16 that run_generate_embeddings.conf's
OPTIONAL_TYPES accepts as not generated, and the tab_solo cells of the
non-selection variants under BESTCAT_EXTRA_VARIANTS=0.

Exit status: 1 if any of the six tables could not be built at all (no usable
source); 3 if every table was built but the verdict is PAPER INCOMPLETE (3,
not 2: argparse and Python's can't-open-file already exit 2); 0 only for PAPER
COMPLETE.

CROSS-ARM MODE. --pca_arms_table compares the arms --pca_arms lists (default
$PCA_ARMS, else config.conf's PCA_ARMS, else --pca_arm alone, which it
refuses: it needs two) and writes to
--out_dir (default sl_comparison/results/pca_arms/; its previous outputs there
are removed first):

  tab_pca.csv   tab:pca (not yet in paper/main.tex): per variant x tier x arm
                (in the order none, joint, per_modality, both): the arm's PCA
                (modality_pca / joint_pca of its combo's config.json), its
                single-embedding runs (native / per-modality PCA), its
                selected modalities and whether they are the reference arm's,
                its combo's encoder input width and parameters per fold, the
                AUPRG / AP / AUROC mean (sd) of its GRADED comparison row (the
                shared evaluator, as tab:tiers), the difference from the
                reference arm (the first of --pca_arms, none when listed), the
                fold-paired spread of the AUPRG difference (the graded
                per-fold AUPRG in summary.folds of the two row JSONs, paired
                by position: evaluate_model.py records them in fold order),
                read as the ablation summaries read theirs (|delta| <= 0.005:
                no measurable difference; else better / worse than the
                reference only beyond 2 SE, else within fold noise; n/a below
                two paired folds), and best_arm
  numbers.json  items.pca (per variant x tier every arm's values, best_arm
                and the 2x2 effects; each arm's selection; the PCA settings),
                arms[] (the index below) and the verdict
  README.md     the verdict (line 3), the PCA arms index (each arm's label,
                roots, paper tables and their verdict: line 3 of its
                <cmp>/paper_tables/README.md), run settings, what is missing

The checks every arm shares (code drift, fold sources, catalog,
folds/meta.json) run once; each arm loads its own selection, comparison rows
and combos as per-arm mode does, and every entry an arm adds is prefixed 'PCA
arm <arm>: '. A cell is tabulated only when its combo is that arm's (as
above), its arch_signature equals the reference arm's combo outside the four
PCA keys (arch_signature.PCA_SIGNATURE_KEYS), its selection's run_token is the
reference arm's (one submission), every embedding file it shares with the
reference combo was trained on the same content, and its graded row is usable
(as tab:tiers'); else it is marked missing / stale, no value is written and it
is listed. The 2x2 effects (fold by fold on the graded AUPRG: per_modality =
((per_modality - none) + (both - joint)) / 2, joint = ((joint - none) + (both -
per_modality)) / 2, interaction = (both - per_modality) - (joint - none)) need
all four arms: a run that requests fewer leaves them absent by design. Verdict:
PAPER COMPLETE only when nothing is listed, so only when every tab_pca cell is
ok and every arm's own paper tables say PAPER COMPLETE (an arm whose do not is
one Missing entry naming its verdict). Exit status: 1 if no tab_pca cell is ok
(or fewer than two arms are requested: nothing is written), 3 if incomplete, 0
if complete.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

ALL_VARIANTS = ("pd:per_head", "pd:pooled", "bilinear:per_head",
                "bilinear:pooled")
PEERS = ("SLMGAE", "SLGNN", "NSF4SL")
# tab_tiers' published peers (and the numbers.json margins built from it):
# PEERS plus ELISL (2026-11-12, reviewer request; ELISL/slurm/run_elisl.sh,
# context-free Sequence_1024 + PPI_ec). ELISL has no external-screen
# predictions (external_transfer.py scores SLMGAE / SLGNN / NSF4SL), so the
# tab_external* tables and their numbers keep PEERS.
TIER_PEERS = PEERS + ("ELISL", )
# Peers no orchestrator submits (ELISL/slurm/run_elisl.sh is run by hand, as
# run_probe.sh is): a tier with no row of theirs at all (neither in
# comparison.csv nor as a row JSON) compares SLeuth with the peers present and
# lists them under 'Absent by design'; a row that exists counts as a peer (and
# one refused, e.g. for its recorded sources, leaves beats_peers_* empty).
OPTIONAL_PEERS = ("ELISL", )
# The sources ELISL/slurm/run_elisl.sh trains on (compare.ELISL_SOURCES).
ELISL_SOURCES = "Sequence_1024|PPI_ec"
CONTROL = "DegreePrior"
# Trained baselines that are not published methods (2026-11-12): the bilinear
# probe of reviewer 22DY, tau * (W x_i)ᵀ(W x_j) + b on PCA arm none's pd:pooled
# combo features (siamese_sl/slurm/run_probe.sh, siamese_sl/probe_args.py).
# A shared row (sl_comparison/results/, not compare.is_siamese_row), printed in
# tab_tiers as kind "baseline": it competes for the bold (a trained model on
# the same features is a fair competitor) but is NOT one of PEERS, so
# beats_peers_* / numbers.json margins stay "SLeuth vs the published methods".
# Arm none only: it is trained on arm none's features, so another arm's
# tab_tiers would set it against SLeuth models that read other features.
BASELINES = ("BilinearProbe", )
BASELINE_ARMS = ("none", )
# The all-modalities combo (2026-11-12; siamese_sl/slurm/run_allmod_combo.sh,
# opt-in ALLMOD_COMBO=1): per variant the SLeuth model on every catalog
# embedding with no modality selection, row siamese_sl_allmod[_<tag>]
# (allmod_row) in each arm's own comparison root. In tab_tiers as kind
# "ablation", right after the selected SLeuth rows: eligible for the bold (the
# best printed number), never in beats_peers_* / numbers.json margins /
# param_ratio, which stay the selected model's. Only with ALLMOD_COMBO=1
# (--allmod_combo, else $ALLMOD_COMBO, else config.conf): off, no row is
# built and none is missing.
ALLMOD_KIND = "ablation"
CVS = ("cv1", "cv2", "cv3")
TIER = {"cv1": "I", "cv2": "II", "cv3": "III"}
METRICS = ("auroc", "ap", "auprg")
SCREENS = ("Adamson", "AdamsonGI", "Corn", "Gilbert")
SPLITS = ("cv1", "cv2", "cv3")
TABLES = ("tab_tiers", "tab_solo", "tab_loco", "tab_cellline",
          "tab_external", "tab_external_ft", "tab_sldb_agreement")
OUTPUT_FILES = tuple(f"{t}.csv" for t in TABLES) + ("numbers.json",
                                                    "README.md")
# The fixed-selection PCA ablation's table of 2026-10-19, superseded on
# 2026-10-20 by the PCA arms (the cross-arm tab_pca lives in
# sl_comparison/results/pca_arms/): a per-arm run still removes a leftover
# copy from its out_dir, so that it cannot sit next to this run's tables as
# one of them.
LEGACY_OUTPUT_FILES = ("tab_pca.csv", )
# --pca_arms_table's outputs (default dir sl_comparison/results/pca_arms/).
PCA_ARMS_OUTPUT_FILES = ("tab_pca.csv", "numbers.json", "README.md")
# The PCA arms in their fixed order (siamese_sl/slurm/config.conf PCA_ARMS,
# arch_signature.PCA_ARMS): PCA within each modality (no / yes) x PCA on the
# concatenation of all modalities (no / yes). Copied, not imported, so that
# the CLI validates without siamese_sl/arch_signature.py; Ctx lists a copy
# that disagrees with it.
PCA_ARM_NAMES = ("none", "joint", "per_modality", "both")
# config.conf's PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE defaults (both
# 0.8; PCA_JOINT_DIM is empty by default).
PCA_DEFAULT_VARIANCE = "0.8"
# A cross-arm difference up to this is read as no measurable difference, as
# siamese_sl/slurm/run_ablation_summary.sh (and the PCA ablation summary of
# 2026-10-19) read theirs; a larger one is a direction only beyond 2 SE.
NO_MEASURABLE_DIFFERENCE = 0.005
# verdict() of a run with nothing listed under Missing.
VERDICT_COMPLETE = "PAPER COMPLETE"
# Exit status of a run whose tables were all built but whose verdict is PAPER
# INCOMPLETE (1: a table could not be built; 0: PAPER COMPLETE). 3, not 2:
# argparse and Python's can't-open-file already exit 2.
EXIT_INCOMPLETE = 3
# tab:tiers / tab:solo / tab:external print every value to 3 decimals; their
# bold (best_<metric>, best_in_category_paper16, best_matched) marks every row
# that prints the best value. Every other flag, rank and margin is exact.
PRINTED_DECIMALS = 3

# tab:solo's 16 embeddings (the catalog may hold more; flagged in_paper).
PAPER_SOLO = {
    "geneformer": "expression", "scgpt": "expression",
    "gene2vec": "expression", "esm2": "protein_seq", "esm1b": "protein_seq",
    "prot_t5": "protein_seq", "text_embed": "text",
    "bioconceptvec": "text", "node2vec_ppi": "ppi", "ppi_svd": "ppi",
    "ppi_raw": "ppi", "mashup": "ppi", "go": "go", "go2vec": "go",
    "onto2vec": "go", "kg_complex": "kg",
}
# The six winners main.tex names (E3).
PAPER_WINNERS = {
    "expression": "geneformer", "protein_seq": "prot_t5",
    "text": "text_embed", "ppi": "ppi_raw", "go": "go2vec",
    "kg": "kg_complex",
}
# Keys that legitimately differ between a single-modality run and its combo.
SOLO_SIG_IGNORE = ("post_pca", "post_pca_dim")
FT_METHODS = ("zero_shot_raw", "zero_shot_ols", "LP_last", "Full",
              "Scratch")
# The tab:external-ft columns (zero_shot_ols is a reference only).
FT_PAPER_METHODS = ("zero_shot_raw", "LP_last", "Full")
# The from-scratch baseline (eval_finetuning --ft_modes ... Scratch): opt-in,
# a paper column of a model only when external_transfer.py's summary holds a
# Scratch row of it; without one the table is as before.
FT_SCRATCH = "Scratch"
FT_COLUMN_NAMES = {"zero_shot_raw": "Zero-shot", "LP_last": "LP_last",
                   "Full": "Full", FT_SCRATCH: "Scratch"}


def ft_paper_methods(rows: List[dict], model: str) -> Tuple[str, ...]:
    """The tab:external-ft columns of one model: FT_PAPER_METHODS, plus
    Scratch when any of its rows is a Scratch row."""
    if any(r["model"] == model and r["method"] == FT_SCRATCH for r in rows):
        return FT_PAPER_METHODS + (FT_SCRATCH, )
    return FT_PAPER_METHODS
# external_transfer.SPLIT_LABELS (test_paper_tables checks they agree). The
# cv2 screen split holds out pairs with >=1 unseen gene, both-unseen included:
# the paper's "one gene unseen" row.
SPLIT_LABELS = {"cv1": "random pairs", "cv2": ">=1 gene unseen",
                "cv3": "both genes unseen"}
# comparison.csv metrics kept as strings (a sha256 prefix can be all digits)
# or as exact ints (seed), never parsed as means. n_folds_scored /
# n_folds_expected / partial are compare.py's record of how many folds the
# means beside them are over, not metrics.
RAW_METRIC_PREFIXES = ("folds_", "trained_on_")
# arch / features / legacy_loop say WHICH model ran (compare.py writes them:
# SLGNN --arch port, NSF4SL --features embedding / --legacy_loop). row_usable
# reads them, so they must reach cmp_raw rather than be parsed as means.
RAW_METRICS = ("seed", "git_sha", "git_dirty", "partial", "n_folds_scored",
               "n_folds_expected", "arch", "features", "legacy_loop")
# A REDUCED SLMGAE (train_slmgae_shared.py --allow_missing_views /
# --allow_static_lp_mapping) is not the authors' model, and that script says
# so: "a reduced model must not sit in the four-model table under SLMGAE's
# name". The trainer records the reduction in model_params.json and
# evaluate_model.build_summary copies it into the row; compare.py names such
# rows in comparison_notes.txt ("do not publish them as SLMGAE"), and this is
# the file that publishes the table. Same two markers as compare.py's
# upstream_input_report, copied rather than imported for the same reason
# (SLMGAE-in-pytorch is not importable by name).
ZERO_FILLED_VIEW = "ZERO-FILLED"
LP_STATIC_FALLBACK_MARK = "static gene_id_mapping.tsv fallback"
# Same two values compare.py names (NSF4SL_ORIGINAL_FEATURES / --arch port):
# an NSF4SL row on other features, or on the legacy loop, and an SLGNN row on
# the KG-free port are SUBSTITUTE inputs, not the authors' models. row_usable
# drops them, so tab:tiers never carries one under the authors' name.
NSF4SL_ORIGINAL_FEATURES = "transe"
SLGNN_SUBSTITUTE_ARCH = "port"


# =============================================================================
# Small helpers
# =============================================================================


def _load_module(name: str, path: Path):
    """Import a repo module by file path (keeps sl_comparison/ and siamese_sl/
    off sys.path, where same-named modules would shadow each other)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _optional_module(name: str, path: Path):
    try:
        return _load_module(name, path)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not load {path}: {e}")
        return None


def variant_tag(v: str) -> str:
    return "" if v == "pd:per_head" else v.replace(":", "_")


def variant_prefix(v: str) -> str:
    t = variant_tag(v)
    return f"var_{t}__" if t else ""


def variant_row(v: str) -> str:
    """sl_comparison row name (collect_and_regrade_siamese.sh)."""
    t = variant_tag(v)
    return f"siamese_sl_{t}" if t else "siamese_sl"


def allmod_row(v: str) -> str:
    """The all-modalities row of variant v (config.conf allmod_row)."""
    t = variant_tag(v)
    return f"siamese_sl_allmod_{t}" if t else "siamese_sl_allmod"


def is_sleuth_row(model: str) -> bool:
    """compare.py is_siamese_row: a SLeuth row (every variant's, and their
    <row>_nocell rows), which each PCA arm grades into its own
    sl_comparison/results[/pca_<arm>]/; every other row is a shared model's,
    in sl_comparison/results/ alone."""
    return model == "siamese_sl" or model.startswith("siamese_sl_")


# ---- PCA arms: siamese_sl/slurm/config.conf's helpers, in Python ----------
# test_paper_tables checks them against config.conf itself.


def pca_arm_dir(arm: str) -> str:
    """config.conf pca_arm_dir: '' for none, else 'pca_<arm>'."""
    return "" if arm == "none" else f"pca_{arm}"


def pca_arm_results(arm: str) -> str:
    """config.conf pca_arm_results: the arm's siamese results root, relative
    to siamese_sl/ (results, or results/pca_<arm>)."""
    d = pca_arm_dir(arm)
    return "results" + (f"/{d}" if d else "")


def pca_arm_cmp(arm: str) -> str:
    """config.conf pca_arm_cmp: the arm's sl_comparison results root,
    relative to the repo root."""
    d = pca_arm_dir(arm)
    return "sl_comparison/results" + (f"/{d}" if d else "")


def pca_pair(arm: str) -> Tuple[str, str]:
    """The two arms whose single-embedding runs train alike, arm's among
    them: (none, joint) native, (per_modality, both) under the per-modality
    PCA (config.conf pca_single_args)."""
    return (("per_modality", "both") if arm in ("per_modality", "both")
            else ("none", "joint"))


def pca_step1_arm(arm: str, arms: Sequence[str]) -> str:
    """config.conf pca_step1_arm: the arm whose results root holds arm's
    single-embedding runs, the first of its pair that arms (PCA_ARMS) lists,
    else arm itself."""
    return next((a for a in pca_pair(arm) if a in arms), arm)


def pca_step1_arms(arm: str, arms: Sequence[str]) -> List[str]:
    """config.conf pca_step1_arms: the arms of the run (arms, its PCA_ARMS)
    that read arm's single-embedding runs (pca_step1_arm's root): those of
    arm's pair it lists, and arm itself, in the fixed order."""
    return [a for a in pca_pair(arm) if a in arms or a == arm]


def pca_targets(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    """PCA_MODALITY_VARIANCE / PCA_JOINT_VARIANCE / PCA_JOINT_DIM from env
    (the environment over config.conf's values: read_config_conf), else
    config.conf's defaults (0.8 / 0.8 / unset), validated as config.conf
    validates them. A value config.conf refuses (its every job would have
    stopped on it) cannot be what the run trained with: refused here too,
    before anything is written."""
    t = {"PCA_MODALITY_VARIANCE": env.get("PCA_MODALITY_VARIANCE") or
         PCA_DEFAULT_VARIANCE,
         "PCA_JOINT_VARIANCE": env.get("PCA_JOINT_VARIANCE") or
         PCA_DEFAULT_VARIANCE,
         "PCA_JOINT_DIM": env.get("PCA_JOINT_DIM") or None}
    for k in ("PCA_MODALITY_VARIANCE", "PCA_JOINT_VARIANCE"):
        if not re.fullmatch(r"0\.\d*[1-9]\d*", t[k]):
            raise SystemExit(f"ERROR: ${k}='{t[k]}': want a fraction in "
                             f"(0, 1), e.g. 0.8 (siamese_sl/slurm/"
                             f"config.conf refuses it too)")
    dim = t["PCA_JOINT_DIM"]
    if dim is not None and not re.fullmatch(r"[1-9]\d*", dim):
        raise SystemExit(f"ERROR: $PCA_JOINT_DIM='{dim}': want a positive "
                         f"integer (no leading zero), or empty for "
                         f"PCA_JOINT_VARIANCE (siamese_sl/slurm/config.conf "
                         f"refuses it too)")
    return t


def pca_arm_label(arm: str, t: Dict[str, Optional[str]]) -> str:
    """config.conf pca_arm_label, word for word (the targets as written)."""
    j = (f"{t['PCA_JOINT_DIM']} components" if t["PCA_JOINT_DIM"] else
         f"variance {t['PCA_JOINT_VARIANCE']}")
    m = t["PCA_MODALITY_VARIANCE"]
    return {"joint": f"PCA on the concatenation of all modalities ({j})",
            "per_modality": f"PCA within each modality (variance {m})",
            "both": f"PCA within each modality (variance {m}), then on "
                    f"their concatenation ({j})"}.get(arm, "no PCA")


def pca_step1_label(arm: str, t: Dict[str, Optional[str]],
                    arms: Sequence[str]) -> str:
    """config.conf pca_step1_label, word for word: what arm's
    single-embedding runs train and which arms of the run (arms, its
    PCA_ARMS) read them ('shared by arms none and joint', or 'used by arm
    joint only' for a run of joint alone, whose runs are its own)."""
    who = pca_step1_arms(arm, arms)
    who = (f"shared by arms {who[0]} and {who[1]}" if len(who) == 2 else
           f"used by arm {who[0]} only")
    if arm in ("per_modality", "both"):
        return (f"single-embedding runs with PCA within the modality "
                f"(variance {t['PCA_MODALITY_VARIANCE']}), {who}")
    return f"single-embedding runs at native dimension (no PCA), {who}"


def pca_target_settings(arm: str, t: Dict[str, Optional[str]]) -> dict:
    """The arm's PCA as arch_signature.pca_settings() would read a config
    trained with config.conf's pca_multi_args under these targets."""
    joint = (("count", int(t["PCA_JOINT_DIM"])) if t["PCA_JOINT_DIM"] else
             ("variance", float(t["PCA_JOINT_VARIANCE"])))
    return {"modality": (("variance", float(t["PCA_MODALITY_VARIANCE"]))
                         if arm in ("per_modality", "both") else None),
            "joint": joint if arm in ("joint", "both") else None}


def resolve_pca_arms(args, env: Dict[str, str]) -> Tuple[str, List[str]]:
    """(--pca_arm, --pca_arms): the arm of a per-arm run (CLI, else $PCA_ARM,
    else none) and the arms the run includes (CLI, else PCA_ARMS from env:
    the environment over config.conf's, else that arm alone), the latter in
    the fixed order, as config.conf normalises PCA_ARMS. An unknown or
    repeated arm is refused (config.conf refuses both) before any result is
    read. Read with getattr: a bare Namespace (no PCA arm at all) is arm
    none."""
    arm = getattr(args, "pca_arm", None) or env.get("PCA_ARM") or "none"
    if arm not in PCA_ARM_NAMES:
        raise SystemExit(f"ERROR: PCA arm {arm!r} (--pca_arm / $PCA_ARM): "
                         f"want one of {' '.join(PCA_ARM_NAMES)}")
    given = getattr(args, "pca_arms", None)
    if given is None:
        given = env["PCA_ARMS"].split() if "PCA_ARMS" in env else [arm]
    # A quoted "none joint" from a wrapper is one argument: split it too.
    words = [w for x in given for w in str(x).split()]
    bad = [w for w in words if w not in PCA_ARM_NAMES]
    twice = sorted({w for w in words if words.count(w) > 1})
    if bad or twice or not words:
        why = (f"'{bad[0]}' is not an arm" if bad else
               f"lists {twice[0]} twice" if twice else "lists no arm")
        raise SystemExit(f"ERROR: --pca_arms / $PCA_ARMS "
                         f"'{' '.join(map(str, given))}': {why} (want "
                         f"some of {' '.join(PCA_ARM_NAMES)})")
    return arm, [a for a in PCA_ARM_NAMES if a in words]


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def parse_bash_array(path: Path, name: str) -> List[str]:
    """Elements of a `NAME=( ... )` bash array in a .conf file (comments
    stripped first: they contain parentheses)."""
    text = "\n".join(ln.split("#", 1)[0] for ln in path.read_text().splitlines())
    m = re.search(rf"(?m)^\s*{re.escape(name)}=\((.*?)\)", text, re.S)
    if not m:
        raise ValueError(f"{name}=( ... ) not found in {path}")
    return shlex.split(m.group(1))


# The PCA arms' settings read_config_conf also returns, each only when
# config.conf sets it (PCA_JOINT_DIM is empty by default).
CONF_PCA_KEYS = ("PCA_ARMS", "PCA_MODALITY_VARIANCE", "PCA_JOINT_VARIANCE",
                 "PCA_JOINT_DIM")
# ... and the opt-in switches read the same way (only when set).
CONF_SWITCH_KEYS = ("ALLMOD_COMBO", )


def read_config_conf(root: Path) -> Dict[str, str]:
    """SIAMESE_VARIANTS / SELECTION_VARIANT / ARCH_ARGS, and the PCA arms'
    CONF_PCA_KEYS, as siamese_sl/slurm/config.conf sets them (sourced in a
    bash subshell with SLURM_JOB_ID unset, as collect_and_regrade_siamese.sh
    does; only the last line is read). {} when it cannot be sourced. The PCA
    keys too (2026-10-20): every training job sources config.conf, while
    pca_arm.sh reads it in a subshell and exports nothing, so a standalone
    run_paper_tables.sh after a default edited there would otherwise label
    and check the arms by paper_tables' own copies of the defaults."""
    import subprocess
    conf = root / "siamese_sl" / "slurm" / "config.conf"
    if not conf.exists():
        return {}
    script = ('unset SLURM_JOB_ID; source "$1" >/dev/null 2>&1 || exit 1; '
              'printf "%s|%s|%s|%s|%s|%s|%s|%s\\n" "${SIAMESE_VARIANTS[*]}" '
              '"$SELECTION_VARIANT" "$ARCH_ARGS" "${PCA_ARMS:-}" '
              '"${PCA_MODALITY_VARIANCE:-}" "${PCA_JOINT_VARIANCE:-}" '
              '"${PCA_JOINT_DIM:-}" "${ALLMOD_COMBO:-}"')
    try:
        r = subprocess.run(["bash", "-c", script, "_", str(conf)],
                           capture_output=True, text=True, timeout=120,
                           cwd=str(root / "siamese_sl"))
    except (OSError, subprocess.TimeoutExpired):
        return {}
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines or lines[-1].count("|") != 7:
        return {}
    v, s, a, *pca = lines[-1].split("|")
    out = {"SIAMESE_VARIANTS_LIST": v, "SELECTION_VARIANT": s,
           "ARCH_ARGS": a}
    out.update({k: x for k, x in zip(CONF_PCA_KEYS + CONF_SWITCH_KEYS, pca)
                if x})
    return out


def fold_index(p: Path) -> int:
    return int(p.name.split("_")[1])


def canon(a: str, b: str) -> Tuple[str, str]:
    a, b = str(a).strip(), str(b).strip()
    return (a, b) if a <= b else (b, a)


def load_pair_set(path: Path) -> set:
    out = set()
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2 and t[0] != t[1]:
                out.add(canon(t[0], t[1]))
    return out


def auroc_rank(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(np.asarray(scores, dtype=np.float64))
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def write_csv(path: Path, rows: List[dict], columns: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c))
                        for c in columns})


def reduced_slmgae(summary: dict) -> Optional[str]:
    """Why an SLMGAE row is a REDUCED model (the markers the trainer records
    in model_params.json), or None. Only a PRESENT marker counts: a row from
    before the trainer recorded them is not a reduction."""
    why = []
    views = summary.get("support_views")
    if isinstance(views, dict):
        zero = sorted(k for k, v in views.items()
                      if str(v).startswith(ZERO_FILLED_VIEW))
        if zero:
            why.append(f"ZERO-FILLED support view(s) {', '.join(zero)} "
                       f"(--allow_missing_views)")
    lp = summary.get("lp_mapping")
    if isinstance(lp, dict) and LP_STATIC_FALLBACK_MARK in str(
            lp.get("id_space", "")):
        why.append("List_Proteins mapped through the static table "
                   "(--allow_static_lp_mapping)")
    return "; ".join(why) or None


def _flag_best(rows: List[dict], group_keys: Sequence[str], value_key: str,
               flag_key: str, eligible=lambda r: True,
               min_group: int = 1, decimals: Optional[int] = None) -> None:
    """Set rows[*][flag_key] = True on the max value_key per group (ties all
    flagged), False on other eligible rows, None elsewhere (also on every row
    of a group with fewer than min_group eligible rows). decimals=None ties
    on the exact value; decimals=k on the value printed to k decimals
    (f"{v:.{k}f}", -0.000 equal to 0.000): the paper's bold."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        if eligible(r) and _num(r.get(value_key)) is not None:
            groups.setdefault(tuple(r.get(k) for k in group_keys), []).append(r)
    for r in rows:
        r.setdefault(flag_key, None)
    for grp in groups.values():
        if len(grp) < min_group:
            continue
        best = max(r[value_key] for r in grp)
        for r in grp:
            # round() is format()'s correctly rounded value, so equal
            # roundings are equal printed strings.
            r[flag_key] = bool(r[value_key] == best if decimals is None else
                               round(r[value_key], decimals) ==
                               round(best, decimals))


# =============================================================================
# Run context
# =============================================================================


class Ctx:
    """Resolved paths, run settings and every loaded source of one PCA arm.

    Per-arm mode builds one. Cross-arm mode (--pca_arms_table, cross=True)
    builds one for the reference arm, which runs the checks every arm shares
    (code drift, fold sources, catalog, folds/meta.json) once, plus one view
    per other arm (base=that context) that reloads only what is the arm's
    own: its roots, selection, comparison rows and combos, with fresh
    per-arm caches. In cross-arm mode every entry an arm's loading or tables
    add is prefixed 'PCA arm <arm>: ' (the reference arm's too), and a view's
    entries go into the reference context's lists."""

    # What a cross-arm view shares with its reference context: the run's
    # settings, the shared checks' results and the lists every entry goes to
    # (the same objects, so a view's entries land in the reference's).
    _SHARED = ("root", "folds", "variants", "selection_variant", "run_token",
               "extra_variants", "allmod", "pca_variance", "arch_args_env",
               "regrade",
               "code_manifest", "code_drift", "missing", "absent_by_design",
               "folds_stale", "notes", "_stamps", "model_params", "arch",
               "vocab", "fsel", "catalog", "ablate_categories", "optional_types",
               "meta", "pca_arms", "pca_targets")

    def __init__(self, args, arm: Optional[str] = None,
                 base: Optional["Ctx"] = None, cross: bool = False):
        self.prefix = ""
        if base is None:
            self._settings(args)
            self._shared_checks()
        else:
            for k in self._SHARED:
                setattr(self, k, getattr(base, k))
        self.cross = cross or base is not None
        arm = arm or self.pca_arm
        if self.cross:
            self.prefix = f"PCA arm {arm}: "
        self._arm(args, arm)

    def _settings(self, args) -> None:
        self.root = Path(args.root).resolve()
        self.folds = self.root / "sl_comparison" / "folds"
        # CLI, else the environment the orchestrator exported, else
        # config.conf itself (SELECTION_VARIANT / ARCH_ARGS are not exported,
        # nor anything a standalone job's pca_arm.sh reads there: PCA_ARMS
        # and the PCA targets).
        conf = read_config_conf(self.root)
        env = {**conf, **{k: v for k, v in os.environ.items() if v}}
        # The arm before anything is read: an unknown one is refused.
        self.pca_arm, self.pca_arms = resolve_pca_arms(args, env)
        self.pca_targets = pca_targets(env)
        self.variants = list(args.variants or env.get(
            "SIAMESE_VARIANTS_LIST", " ".join(ALL_VARIANTS)).split())
        self.selection_variant = (args.selection_variant or
                                  env.get("SELECTION_VARIANT") or "pd:per_head")
        # The one SLeuth variant main.tex's tab:tiers prints ("SLeuth is the
        # pooled, no-PCA model"). Only it, the peers and the reference models
        # decide that table's bold: an unprinted variant (or the
        # all-modalities ablation) leading at the printed precision used to
        # take the bold away from every printed cell of its column.
        _pv = getattr(args, "paper_variant", None) or env.get("PAPER_VARIANT")
        self.paper_variant_explicit = bool(_pv)
        self.paper_variant = _pv or ("pd:pooled" if "pd:pooled" in
                                     self.variants else
                                     (self.variants or ["pd:pooled"])[0])
        self.run_token = (args.run_token if args.run_token is not None else
                          env.get("RUN_TOKEN")) or None
        extra = (args.bestcat_extra_variants
                 if args.bestcat_extra_variants is not None else
                 env.get("BESTCAT_EXTRA_VARIANTS", "0"))
        self.extra_variants = str(extra) == "1"
        am = getattr(args, "allmod_combo", None)
        am = am if am is not None else env.get("ALLMOD_COMBO", "0")
        self.allmod = str(am) == "1"
        self.pca_variance = (args.pca_variance or
                             env.get("BEST_CAT_PCA_VARIANCE") or "0.8")
        self.arch_args_env = env.get("ARCH_ARGS")
        self.regrade = bool(getattr(args, "regrade", False))
        # The orchestrator's submission-time code manifest: CLI, else its
        # export ('' = none). code_drift: None until checked.
        cm = getattr(args, "code_manifest", None)
        cm = cm if cm is not None else env.get("CODE_MANIFEST")
        self.code_manifest = str(Path(cm).resolve()) if cm else None
        self.code_drift: Optional[List[str]] = None
        self.missing: List[str] = []
        # Absences the run's settings intend (by_design): listed apart, and
        # they do not make the verdict incomplete.
        self.absent_by_design: List[str] = []
        # What check_sources found stale in the folds' recorded sources.
        self.folds_stale: List[str] = []
        self.notes: List[str] = []
        self._stamps: Dict[str, dict] = {}
        if not conf and (self.root / "siamese_sl" / "slurm" /
                         "config.conf").exists():
            # It refuses an environment it does not accept (an exported
            # POST_PCA_* among them): its variant and PCA settings are then
            # not read, and nothing else would say so.
            self.note("siamese_sl/slurm/config.conf could not be sourced (it "
                      "refused this environment, or failed): the variant and "
                      "PCA settings are the command line's, the "
                      "environment's or the defaults")

    def _shared_checks(self) -> None:
        """What every PCA arm shares: the code, the folds and their sources,
        the catalog and the confs (cross-arm mode runs them once)."""
        self._check_code_drift()

        self.model_params = _load_module("_pt_model_params",
                                         HERE / "model_params.py")
        self.arch = _optional_module(
            "_pt_arch_signature", REPO_ROOT / "siamese_sl" / "arch_signature.py")
        self.vocab = _optional_module(
            "_pt_cell_line_vocab", REPO_ROOT / "siamese_sl" / "cell_line_vocab.py")
        # A per_fold selection (SELECTION_MODE=per_fold, 2026-11-12) is read
        # through it (each arm's Ctx says when it is needed and missing).
        self.fsel = _optional_module(
            "_pt_fold_selection", REPO_ROOT / "siamese_sl" / "fold_selection.py")
        # Without it no single-modality / ablation dir is shown to be its
        # combo's architecture (freshness) and no combo is shown to be the
        # arm's PCA (_load_combo), so the verdict cannot be COMPLETE; the
        # cross-arm table, which compares the arms' PCA, is not built at all.
        if self.arch is None:
            self.miss("siamese_sl/arch_signature.py not loadable: tab_solo / "
                      "tab_loco arch signatures and the combos' PCA arm "
                      "unchecked")
        elif tuple(getattr(self.arch, "PCA_ARMS", ())) != PCA_ARM_NAMES:
            # The 2x2's order (and the arm names) are arch_signature's and
            # config.conf's: a copy here that disagrees would mislabel arms.
            self.miss(f"siamese_sl/arch_signature.py PCA_ARMS "
                      f"{tuple(getattr(self.arch, 'PCA_ARMS', ()))} is not "
                      f"paper_tables' {PCA_ARM_NAMES}: the arms may be "
                      f"mislabelled")

        slurm = self.root / "siamese_sl" / "slurm"
        try:
            self.catalog = [tuple(e.split(":")[:3]) for e in parse_bash_array(
                slurm / "run_best_per_category.conf", "EMB_CATALOG")]
        except (OSError, ValueError) as e:
            self.catalog = []
            self.miss(f"EMB_CATALOG ({e})")
        try:
            self.ablate_categories = parse_bash_array(
                slurm / "run_ablation.conf", "ABLATE_CATEGORIES")
        except (OSError, ValueError) as e:
            self.ablate_categories = []
            self.miss(f"ABLATE_CATEGORIES ({e})")
        # The catalog types whose absence generation accepts: one of them
        # outside the paper's 16 that was not generated is absent by design.
        # Unreadable: none is, so every one not generated stays missing.
        try:
            self.optional_types = set(parse_bash_array(
                slurm / "run_generate_embeddings.conf", "OPTIONAL_TYPES"))
        except (OSError, ValueError) as e:
            self.optional_types = set()
            why = str(e).replace(str(self.root) + os.sep, "")
            self.note(f"OPTIONAL_TYPES not read ({why}): every catalog "
                      f"embedding not generated is listed as missing")

        self.meta = read_json(self.folds / "meta.json") or {}
        if not self.meta:
            self.miss(f"{self.folds / 'meta.json'}")
        else:
            self._check_fold_sources()
        if os.environ.get("ABLATION_RUN_TOKEN"):
            self.note("$ABLATION_RUN_TOKEN is ignored (pass "
                      "--ablation_run_token for an ablation rerun under a "
                      "token of its own)")

    def _arm(self, args, arm: str) -> None:
        """Everything that is one PCA arm's own (siamese_sl/slurm/config.conf
        pca_arm_results / pca_arm_cmp): arm none reads today's paths, arm X
        the same names one directory down. The shared models' rows are
        sl_comparison/results/'s for every arm."""
        self.pca_arm = arm
        self.pca_label = pca_arm_label(arm, self.pca_targets)
        # How a rerun hint names the arm: the orchestrators submit an arm's
        # jobs as PCA_ARM=<arm> sbatch ... --export=ALL. Arm none's too
        # (2026-10-21): a job submitted without PCA_ARM is the arm the
        # submitting shell exports, none only where none or nothing is, and
        # the hand recipes leave one exported (sl_comparison/README.md, "After
        # a combo job submitted by hand": export PCA_ARM=<arm>).
        self.arm_hint = f" with PCA_ARM={arm}"
        self.sia_results = self.root / "siamese_sl" / pca_arm_results(arm)
        self.cmp_results = self.root / pca_arm_cmp(arm)
        self.shared_results = self.root / "sl_comparison" / "results"
        if self.cross:
            # One --comparison_csv / --external_dir cannot be every arm's:
            # each reads its own (run_pca_arms notes a CLI value ignored).
            self.out_dir = None
            self.ext_dir = self.cmp_results / "external_transfer"
            self.comparison_csv = self.cmp_results / "comparison.csv"
        else:
            self.out_dir = Path(args.out_dir or self.cmp_results /
                                "paper_tables")
            self.ext_dir = Path(args.external_dir or self.cmp_results /
                                "external_transfer")
            self.comparison_csv = Path(args.comparison_csv or
                                       self.cmp_results / "comparison.csv")
        # The layout run_compare.sh writes (compare.py --results_dir <the
        # arm's sl_comparison root> --shared_results_dir sl_comparison/results
        # --csv <the arm's root>/comparison.csv): there every table row came
        # from a <model>_<cv>.json in that row's own directory (the arm's for
        # a SLeuth row, sl_comparison/results/ for a shared model's), so one
        # without it is stale (metric()). A hand-passed --comparison_csv need
        # not have its rows' JSONs beside it. Compared resolved (a relative
        # spelling of the arm's own table, as a wrapper may pass it, is the
        # default table; it is read relative to the working directory too).
        self.default_comparison_csv = (
            self.comparison_csv.resolve() ==
            (self.cmp_results / "comparison.csv").resolve())
        self._fold_ok: Dict[tuple, bool] = {}
        self._row_usable: Dict[tuple, bool] = {}
        self._eval_cfg: Dict[tuple, dict] = {}
        if not self.cross and arm not in self.pca_arms:
            self.note(f"PCA arm {arm} is not among --pca_arms "
                      f"({' '.join(self.pca_arms)}): they only decide where "
                      f"its single-embedding runs are when its selection "
                      f"does not record it (config.conf's owner rule)")

        sel_path = self.sia_results / "best_per_category.json"
        self.selection = read_json(sel_path)
        if self.selection is None:
            self.miss(f"{sel_path}")
        meta = (self.selection or {}).get("_selection_meta") or {}
        # "global" (one selection reused for every fold of every CV; every
        # selection before 2026-11-12) or "per_fold" (each CV and fold on its
        # own selection split; each CV's combo on the union of its folds'
        # winners, fold k on its own).
        self.selection_mode = meta.get("selection_mode") or "global"
        if self.selection_mode == "per_fold" and self.fsel is None:
            self.miss("siamese_sl/fold_selection.py not loadable: the per_fold "
                      "selection's combos and ablations cannot be checked "
                      "against it")
        # The combo job records its arm (2026-10-20): a selection of another
        # arm here is a copy from another namespace, and its winners and
        # single-embedding runs are not this arm's.
        if meta.get("pca_arm") not in (None, arm):
            self.miss(f"{sel_path} records PCA arm {meta.get('pca_arm')!r}, "
                      f"not {arm}: another arm's selection in this arm's "
                      f"namespace (a hand copy?); its winners and "
                      f"single-embedding runs are not this arm's")
        self.step1_results = self._step1_results(meta, sel_path)
        # What those runs are and which arms read them: the label the combo
        # job recorded with them (_selection_meta.step1_pca, written under
        # the PCA_ARMS of the run that ranked them), else config.conf's
        # wording over --pca_arms (a later --skip-siamese tables run may
        # list other arms than that run did).
        rec = meta.get("step1_pca")
        self.step1_label = (rec.strip() if isinstance(rec, str) and
                            rec.strip() else
                            pca_step1_label(arm, self.pca_targets,
                                            self.pca_arms))
        # Per table: the single-modality dirs are the Step-1 array Step 2
        # ranked (its recorded token, also when RUN_TOKEN is unset), while an
        # ablation rerun may carry a token of its own.
        sel_tok = meta.get("run_token")
        solo = getattr(args, "solo_run_token", None)
        self.solo_run_token = (solo if solo is not None else
                               sel_tok or self.run_token) or None
        # CLI only: the orchestrators export RUN_TOKEN for every job, so an
        # environment override inherited from an earlier export would mark a
        # fresh run's ablation dirs stale. With no RUN_TOKEN at all (a
        # standalone run_paper_tables.sh) fall back to the selection's token,
        # as the single-modality dirs do: every orchestrator path ablates
        # under it (one submission trains both; --ablation-only and the root
        # --skip-siamese reuse the selection's), so an ablate_no_* dir a
        # SKIPped or cancelled task left from an earlier run is marked stale
        # instead of tabulated as this run's.
        abl = getattr(args, "ablation_run_token", None)
        self.ablation_run_token = (abl if abl is not None else
                                   self.run_token or sel_tok) or None
        self.cmp = self._load_comparison()
        self.json_rows = self._load_result_jsons()
        # (v, cv) of combos trained on another fold set (left out), of
        # combos that are not the run their comparison row graded, and why a
        # (v, cv) has no combo here (the cross-arm table's status).
        self.fold_refused_combos: set = set()
        self.combo_not_graded: Dict[tuple, str] = {}
        self.combo_problem: Dict[tuple, str] = {}
        self.combos = {(v, cv): self._load_combo(v, cv)
                       for v in self.variants for cv in CVS}
        self.trained_pca = self._check_trained_pca()

    def _step1_results(self, meta: dict, sel_path: Path) -> Path:
        """Where this arm's single-embedding runs are (tab_solo,
        selection_problems): siamese_sl/<_selection_meta.step1_root>, the
        $STEP1_ROOT the combo job's Step 2 ranked (2026-10-20), else
        config.conf's owner rule over --pca_arms (a selection from before the
        PCA arms records none). The recorded root is the truth of what was
        ranked, so it is read even where the owner rule would say otherwise
        (noted), or where it is not a root of this arm's pair (listed: the
        two arms of a pair rank the same runs, and no other arm's are this
        arm's PCA)."""
        rule = pca_arm_results(pca_step1_arm(self.pca_arm, self.pca_arms))
        rec = meta.get("step1_root")
        if rec is None:
            if self.selection is not None:
                self.note(f"{sel_path.relative_to(self.root)} records no "
                          f"_selection_meta.step1_root (a selection from "
                          f"before the PCA arms): single-embedding runs read "
                          f"from siamese_sl/{rule}/ (config.conf's owner "
                          f"rule over --pca_arms)")
            return self.root / "siamese_sl" / rule
        p = Path(str(rec))
        if not isinstance(rec, str) or not rec.strip() or p.is_absolute() \
                or ".." in p.parts:
            self.miss(f"{sel_path} _selection_meta.step1_root {rec!r} is not "
                      f"a path inside siamese_sl/: single-embedding runs read "
                      f"from siamese_sl/{rule}/ (config.conf's owner rule) "
                      f"instead")
            return self.root / "siamese_sl" / rule
        norm = p.as_posix()
        pair = [pca_arm_results(a) for a in pca_pair(self.pca_arm)]
        if norm not in pair:
            self.miss(f"{sel_path} _selection_meta.step1_root {rec!r} is not "
                      f"a single-embedding root of PCA arm {self.pca_arm}'s "
                      f"pair ({' or '.join(pair)}): tab_solo and the "
                      f"selection check read siamese_sl/{norm}/, as recorded")
        elif norm != rule:
            self.note(f"single-embedding runs read from siamese_sl/{norm}/ "
                      f"(the selection's _selection_meta.step1_root), not "
                      f"siamese_sl/{rule}/ (config.conf's owner rule over "
                      f"--pca_arms {' '.join(self.pca_arms)})")
        return self.root / "siamese_sl" / norm

    def _check_trained_pca(self) -> Dict[str, Optional[str]]:
        """run.pca_settings: the PCA within each modality / on the
        concatenation every combo used here trained with
        (arch_signature.pca_settings of its config.json, as table cells),
        None while unknown (no combo, no arch_signature.py) or while the
        combos disagree. _load_combo has kept only combos of this arm, so
        only a setting can differ: from another one (two submissions under
        two PCA_* exports), or from this job's targets, the label the README
        and every output print (one Missing entry per setting: the label
        would not describe those combos)."""
        if self.arch is None:
            return {"modality": None, "joint": None}
        tgt = pca_target_settings(self.pca_arm, self.pca_targets)
        want = (pca_setting_text(tgt["modality"]),
                pca_setting_text(tgt["joint"]))
        seen: Dict[tuple, List[str]] = {}
        for (v, cv), c in self.combos.items():
            if c:
                s = self.arch.pca_settings(c["config"])
                seen.setdefault((pca_setting_text(s["modality"]),
                                 pca_setting_text(s["joint"])), []).append(
                                     f"{v} {cv}")
        t = self.pca_targets
        for (mod, joint), where in seen.items():
            if (mod, joint) != want:
                self.miss(f"PCA arm {self.pca_arm}'s combo(s) "
                          f"{', '.join(where[:4])}"
                          f"{', ...' if len(where) > 4 else ''} trained with "
                          f"PCA within each modality {mod}, on the "
                          f"concatenation {joint}, not this job's targets "
                          f"({want[0]} / {want[1]}: PCA_MODALITY_VARIANCE="
                          f"{t['PCA_MODALITY_VARIANCE']}, PCA_JOINT_VARIANCE="
                          f"{t['PCA_JOINT_VARIANCE']}, PCA_JOINT_DIM="
                          f"{t['PCA_JOINT_DIM'] or '(unset)'}): the label "
                          f"'{self.pca_label}' does not describe them; rerun "
                          f"with the run's PCA_* values exported")
        if len(seen) != 1:
            return {"modality": None, "joint": None}
        mod, joint = next(iter(seen))
        return {"modality": mod, "joint": joint}

    def rel(self, p: Path) -> str:
        """p relative to the root when inside it (as the tables print a
        directory), else as given."""
        try:
            return str(Path(p).relative_to(self.root))
        except ValueError:
            return str(p)

    def _check_code_drift(self) -> None:
        """List the code that changed after the run was submitted.

        The jobs of one submission start over days and read the code they
        source or import when they start, while their batch scripts are the
        ones queued at submission. The tree is updated by upload and git does
        not run in the jobs, so a file uploaded in between (on 2026-09-23 a
        whole-tree upload 32 min after the submission) reaches some jobs and
        not others with nothing recording it. The orchestrator hashes that
        code into CODE_MANIFEST at submission (code_manifest.py); the same
        files under root are hashed again here, after every job a table waits
        for (only the all-gene-pair export starts later), and
        every one that changed, appeared or vanished is one Missing entry.
        code_drift: that list ([] = none), None when not checked."""
        path = self.code_manifest
        if path is None:
            self.note("no code manifest (not an orchestrated run): code drift "
                      "not checked")
            return
        cm = _optional_module("_pt_code_manifest", HERE / "code_manifest.py")
        if cm is None:
            self.miss("sl_comparison/code_manifest.py not loadable: code "
                      "drift during the run not checked")
            return
        try:
            why = None if cm.read_manifest(path) else "no entries"
        except OSError as e:
            why = e.strerror or type(e).__name__
        except ValueError:  # UnicodeDecodeError too
            why = "not '<sha256>  <path>' lines"
        if why:
            self.miss(f"code manifest {path} unreadable ({why}): code drift "
                      f"during the run not checked")
            return
        try:
            d = cm.drift(self.root, path)
        except Exception as e:  # noqa: BLE001  (a file of the tree unreadable)
            self.miss(f"code drift during the run not checked: the code could "
                      f"not be hashed again ({type(e).__name__}: {e})")
            return
        self.code_drift = d
        if d:
            head = ", ".join(d[:5]) + (", ..." if len(d) > 5 else "")
            self.miss(f"code changed after submission ({len(d)} file(s): "
                      f"{head}): jobs of this run may have run different "
                      f"versions; check them before publishing")

    def _check_fold_sources(self) -> None:
        """Flag folds whose recorded sources no longer match the files now.

        run_shared_folds.sh is the only other --check_sources caller: under
        the orchestrator (SHARED_FOLDS_FROZEN) it refuses a set whose fold
        files changed since step 0 (and only warns when just a source
        changed), standalone it regenerates. A standalone `sbatch
        sl_comparison/slurm/run_paper_tables.sh`, or a DAG whose source
        changed after step 0, would otherwise republish meta.json's conflict
        attribution (item_folds_meta: num_conflicts_same_line / _cross_line /
        _same_head) next to head statistics item_heads recomputes with the
        LIVE siamese_sl/cell_line_vocab.py — two vocabularies side by side in
        numbers.json with nothing flagged. prepare_folds fingerprints the
        vocabulary too, so check_sources sees exactly that edit. The
        differences are listed under Missing (so the verdict is incomplete)
        and mark items.folds_meta stale.
        """
        pf = _optional_module("_pt_prepare_folds", HERE / "prepare_folds.py")
        if pf is None:
            self.miss("sl_comparison/prepare_folds.py not loadable: the "
                      "folds' source files were NOT checked against "
                      "meta.json source_sha256")
            return
        pos = self.root / self.meta.get(
            "pos_path", "data/SL_SynLethDB_experimental.txt")
        neg = self.root / self.meta.get(
            "neg_path", "data/SL_SynLethDB_experimental_negatives.txt")
        try:
            problems = pf.check_sources(self.folds, pos, neg)
        except Exception as e:  # noqa: BLE001
            self.miss(f"could not check the folds' source files "
                      f"({type(e).__name__}: {e})")
            return
        # Regenerating in place would also break every row's fold stamp, and
        # under the orchestrator run_shared_folds.sh does not regenerate
        # (SHARED_FOLDS_FROZEN): only the run's own fold set (downloaded with
        # its results) or a new run makes these numbers whole.
        self.folds_stale = [p.replace(str(self.root) + os.sep, "")
                            for p in problems]
        if self.folds_stale:
            self.miss("shared folds are STALE: the fold set on disk cannot "
                      "be shown to be built from the current pair files / "
                      "sidecars / cell-line vocabulary (" +
                      "; ".join(self.folds_stale) + "); these numbers are "
                      "not publishable: if this tree is a copy of a "
                      "finished run's results (e.g. the laptop), download "
                      "that run's sl_comparison/folds (and the data/ pair "
                      "files and sidecars it fingerprints) with them instead "
                      "of rerunning; otherwise rerun the orchestrator (bash "
                      "run_benchmarking_siamese_vs_others.sh; step 0 "
                      "regenerates the folds and keeps them read-only for "
                      "the run); and do not upload sl_comparison/folds, the "
                      "data/ pair files or siamese_sl/cell_line_vocab.py "
                      "while a run is live")

    def _entry(self, what: str, prefixed: bool) -> str:
        """An entry as listed: root-relative, and in cross-arm mode named
        after its arm ('PCA arm <arm>: ...') unless it is the comparison's
        own (prefixed=False)."""
        what = what.replace(str(self.root) + os.sep, "")
        return (self.prefix if prefixed else "") + what

    def miss(self, what: str, prefixed: bool = True) -> None:
        what = self._entry(what, prefixed)
        if what not in self.missing:
            self.missing.append(what)

    def by_design(self, what: str, prefixed: bool = True) -> None:
        """An absence the run's settings intend: listed under 'Absent by
        design', not under Missing."""
        what = self._entry(what, prefixed)
        if what not in self.absent_by_design:
            self.absent_by_design.append(what)

    def optional_peer_absent(self, cv: str, model: str) -> bool:
        """True (and listed under 'Absent by design', once) for an
        OPTIONAL_PEERS model with no row of any kind for any CV."""
        if model not in OPTIONAL_PEERS:
            return False
        # Absent by design only when it has no row for ANY tier
        # (2026-11-13): a run that succeeded for some CVs and failed for
        # another compared that tier with fewer peers under PAPER COMPLETE.
        # With a row anywhere, a missing CV is a missing item.
        for c in CVS:
            key = (c, model)
            if key in self.cmp or \
                    key in (getattr(self, "cmp_raw", None) or {}) or \
                    key in self.json_rows:
                return False
        self.by_design(f"{model}/{cv}: absent (optional peer, run by hand: "
                       f"{model}/slurm/run_{model.lower()}.sh); tab_tiers "
                       f"beats_peers_* and numbers.json margins compare the "
                       f"peers present")
        return True

    def tier_peers(self, cv: str) -> Tuple[str, ...]:
        """tab_tiers' peers for cv: TIER_PEERS less an absent optional one."""
        return tuple(p for p in TIER_PEERS
                     if not self.optional_peer_absent(cv, p))

    def note(self, what: str, prefixed: bool = True) -> None:
        what = (self.prefix if prefixed else "") + what
        if what not in self.notes:
            self.notes.append(what)

    # ---- comparison rows ---------------------------------------------------
    def _load_comparison(self) -> Dict[Tuple[str, str], Dict[str, tuple]]:
        rows: Dict[Tuple[str, str], Dict[str, tuple]] = {}
        self.cmp_raw: Dict[Tuple[str, str], Dict[str, str]] = {}
        if not self.comparison_csv.exists():
            self.miss(f"{self.comparison_csv} (compare.py)")
            return rows
        arms = set()
        with open(self.comparison_csv) as f:
            for r in csv.DictReader(f):
                if None in r.values():
                    # Fewer fields than the header: a line cut short
                    # (compare.py writes all five on every row), so the table
                    # was not written to its end (a full disk, a copy cut
                    # short). The rows after the cut fall back to their JSON.
                    self.miss(f"{self.comparison_csv} has a line cut short "
                              f"(compare.py did not finish writing it); rerun "
                              f"sl_comparison/slurm/run_compare.sh"
                              f"{self.arm_hint}")
                    continue
                # compare.py --pca_arm writes the arm in a trailing column
                # (2026-10-20; a table from before has none).
                if r.get("pca_arm"):
                    arms.add(r["pca_arm"])
                key, name = (r["cv"], r["model"]), r["metric"]
                if name.startswith(RAW_METRIC_PREFIXES) or name in RAW_METRICS:
                    self.cmp_raw.setdefault(key, {})[name] = r.get("mean")
                    continue
                rows.setdefault(key, {})[name] = (_num(r.get("mean")),
                                                  _num(r.get("std")))
        other = sorted(arms - {self.pca_arm})
        if other:
            # Another arm's comparison (a copy into this namespace, a hand
            # --comparison_csv): its SLeuth rows are that arm's models under
            # this arm's names. Not read; the rows fall back to this arm's
            # JSONs.
            self.miss(f"{self.comparison_csv} names PCA arm(s) "
                      f"{', '.join(other)} in its pca_arm column, not "
                      f"{self.pca_arm}: another arm's table, not read (rerun "
                      f"sl_comparison/slurm/run_compare.sh{self.arm_hint})")
            self.cmp_raw = {}
            return {}
        return rows

    def _load_result_jsons(self) -> Dict[Tuple[str, str], dict]:
        """Every graded row JSON of this arm, keyed (cv, model). The SLeuth
        rows only from the arm's own sl_comparison root, the shared models'
        only from sl_comparison/results/ (for arm none the two are one
        directory, read once): rows are keyed by summary.model, not by file
        name, and arm none's SLeuth rows sit in sl_comparison/results/ under
        the same names as every other arm's, so a union of the two
        directories would let one arm's row stand in for another's."""
        out = {}
        if self.cmp_results.resolve() == self.shared_results.resolve():
            dirs = [(self.cmp_results, None)]
        else:
            dirs = [(self.cmp_results, True), (self.shared_results, False)]
        for d, sleuth in dirs:
            for p in sorted(d.glob("*.json")):
                s = (read_json(p) or {}).get("summary") or {}
                if not (s.get("model") and s.get("cv_type")):
                    continue
                if sleuth is not None and is_sleuth_row(s["model"]) != sleuth:
                    # In the shared directory: arm none's SLeuth rows, not
                    # this arm's (expected). In the arm's: a shared model's
                    # row that compare.py never reads from there.
                    if sleuth:
                        self.note(f"{self.rel(p)} is a {s['model']} row in "
                                  f"PCA arm {self.pca_arm}'s directory: not "
                                  f"read (the shared models' rows are "
                                  f"{self.rel(self.shared_results)}/'s)")
                    continue
                # Only evaluate_model.build_summary rows (DegreePrior and the
                # _nocell rows are too): a collect_siamese.py row an
                # interrupted compare job never re-graded is self-scored at
                # siamese's own prevalence.
                if not (s.get("provenance") or {}).get("meta.json_sha256"):
                    self.miss(f"{p} (not graded by evaluate_model.py; "
                              f"ignored)")
                    continue
                out[(s["cv_type"], s["model"])] = s
        return out

    def current_stamp(self, cv: Optional[str] = None) -> dict:
        key = cv or ""
        if key not in self._stamps:
            self._stamps[key] = self.model_params.fold_stamp(self.folds, cv)
        return self._stamps[key]

    def fold_record(self, cv: str, model: str, source: str) -> dict:
        """The fold fingerprints a row was graded on (evaluate_model
        provenance) and trained on (model_params fold_stamp; none for a model
        that trains nothing, e.g. DegreePrior), as comparison.csv or the row's
        JSON records them, and every difference from the current folds."""
        names = self.model_params.STAMP_FILES
        if source == "comparison.csv":
            raw = self.cmp_raw.get((cv, model), {})
            graded = {n: raw.get(f"folds_{n}_sha256") or None for n in names}
            trained = {n: raw.get(f"trained_on_{n}_sha256") or None
                       for n in names}
        else:
            s = self.json_rows.get((cv, model)) or {}
            prov, st = s.get("provenance") or {}, s.get("fold_stamp") or {}
            graded = {n: prov.get(f"{n}_sha256") for n in names}
            trained = {n: st.get(n) for n in names}
        cur = self.current_stamp(cv)
        problems = [f"graded on {n} {graded[n]} (current {cur[n]})"
                    for n in names if graded[n] != cur[n]]
        if any(trained.values()):
            problems += [f"trained on {n} {trained[n]} (current {cur[n]})"
                         for n in names if trained[n] != cur[n]]
        else:
            trained = None
        return {"source": source, "graded_on": graded, "trained_on": trained,
                "problems": problems}

    def fold_ok(self, cv: str, model: str, source: str) -> bool:
        """False (and listed as missing, once) for a row not graded and
        trained on the current sl_comparison/folds."""
        key = (cv, model, source)
        if key not in self._fold_ok:
            probs = self.fold_record(cv, model, source)["problems"]
            self._fold_ok[key] = not probs
            if probs:
                self.miss(f"{model}/{cv} ({source}) is not on the current "
                          f"folds, ignored: " + "; ".join(probs))
        return self._fold_ok[key]

    def row_usable(self, cv: str, model: str) -> bool:
        """False (and listed once) for a row that must not be tabulated
        whatever its source: a mean over FEWER folds than the fold set holds
        (evaluate_model.py --allow_partial, or a control that dropped a
        degenerate fold), a REDUCED SLMGAE, or a SUBSTITUTE-INPUT peer that is
        not the model the table names. compare.py copies n_folds_scored /
        n_folds_expected / partial and arch / features / legacy_loop into
        comparison.csv; the SLMGAE reduction markers live in the row's JSON
        only. The peer rules mirror compare.py's model_config_report /
        upstream_input_report, which only WARN in comparison_notes.txt — this
        is the file that publishes tab:tiers under the authors' names."""
        key = (cv, model)
        if key not in self._row_usable:
            raw = self.cmp_raw.get(key, {})
            s = self.json_rows.get(key) or {}

            def _cfg(name):
                """comparison.csv cell, else the row JSON (blank = absent)."""
                v = raw.get(name)
                if v is None or str(v).strip() == "":
                    v = s.get(name)
                return v

            def _true(v):
                return str(v).strip().lower() in ("true", "1")

            partial = (str(raw.get("partial", "")).strip().lower() == "true"
                       or s.get("partial") is True)
            why = None
            if partial:
                got = raw.get("n_folds_scored") or s.get("n_folds_scored")
                want = raw.get("n_folds_expected") or s.get("n_folds_expected")
                n = (f" ({got} of {want} folds)"
                     if got is not None and want is not None else "")
                why = (f"partial fold set{n}; a partial mean is not "
                       f"comparable to a complete one")
            elif model == "SLMGAE":
                r = reduced_slmgae(s)
                why = f"REDUCED SLMGAE: {r}" if r else None
            elif model == "NSF4SL":
                # The authors' NSF4SL reads TransE features and the current
                # training loop; --features embedding / --legacy_loop are a
                # substitute (NSF4SL/slurm/run_nsf4sl.sh).
                feats = _cfg("features")
                if feats is not None and str(feats) != NSF4SL_ORIGINAL_FEATURES:
                    why = (f"--features {feats} is not the authors' TransE "
                           f"input")
                elif _true(_cfg("legacy_loop")):
                    why = "--legacy_loop is not the authors' training loop"
            elif model == "ELISL" and (raw or s):
                # train_elisl.py records its sources; anything but the two
                # context-free ones (or no record) is not the row the paper
                # describes.
                src = _cfg("elisl_sources")
                if src is None or str(src).strip() == "":
                    why = "ELISL sources not recorded (elisl_sources)"
                elif str(src) != ELISL_SOURCES:
                    why = (f"ELISL sources {src}, not the context-free "
                           f"{ELISL_SOURCES}")
            elif model == "SLGNN" and (raw or s):
                # --arch port never reads SynLethKG; a row that EXISTS with no
                # arch record was graded before the trainer recorded one, so
                # WHICH input produced it is undisclosed (compare.py's
                # upstream_input_report says the same). A row that is absent
                # altogether is reported by its own caller, not here.
                arch = _cfg("arch")
                if arch is None or str(arch).strip() == "":
                    why = ("architecture not disclosed (graded before the "
                           "trainer recorded it)")
                elif str(arch) == SLGNN_SUBSTITUTE_ARCH:
                    why = ("--arch port: per-gene --embedding_path features, "
                           "no SynLethKG input")
            self._row_usable[key] = why is None
            if why:
                self.miss(f"{model}/{cv} is not tabulated: {why}")
        return self._row_usable[key]

    def _json_metric(self, cv: str, model: str, key: str) -> Optional[
            Tuple[float, Optional[float]]]:
        """(mean, sd) of key in the row's evaluate_model JSON, or None."""
        s = self.json_rows.get((cv, model))
        if s is None or not self.fold_ok(cv, model, "json"):
            return None
        if key == "trainable_params" and isinstance(
                s.get("trainable_params"), (int, float)):
            return float(s["trainable_params"]), None
        if _num(s.get(f"{key}_mean")) is not None:
            return float(s[f"{key}_mean"]), _num(s.get(f"{key}_std"))
        return None

    def metric(self, cv: str, model: str, key: str) -> Tuple[
            Optional[float], Optional[float], Optional[str]]:
        """(mean, sd, source) from comparison.csv, else the row's JSON; None
        for a row whose fold fingerprints are not the current folds', that is
        not usable at all (row_usable), or whose two sources disagree or whose
        JSON (which the default table was written from) is gone."""
        if not self.row_usable(cv, model):
            return None, None, None
        rec = self.cmp.get((cv, model), {})
        if key in rec and rec[key][0] is not None:
            if not self.fold_ok(cv, model, "comparison.csv"):
                return None, None, None
            # A standalone rerun MOVES the previous row aside before anything
            # else (run_shared_models.sh, run_shared_folds.sh,
            # cell_line_ablation/grade.sh: "a failed task must leave its row
            # MISSING") and none of them rewrites the table, so a row of the
            # default table without its JSON is the pre-rerun number of a row
            # this checkout no longer has. Its JSON is in the row's own
            # directory: the arm's for a SLeuth row, sl_comparison/results/
            # for a shared model's (the only one _load_result_jsons reads it
            # from).
            if self.default_comparison_csv and \
                    (cv, model) not in self.json_rows:
                d = (self.cmp_results if is_sleuth_row(model) else
                     self.shared_results)
                self.miss(f"{model}/{cv}: comparison.csv has a row but no "
                          f"evaluate_model JSON beside it "
                          f"({d / f'{model}_{cv}.json'} was moved aside by a "
                          f"rerun, or was never graded); not tabulated, rerun "
                          f"sl_comparison/slurm/run_compare.sh"
                          f"{self.arm_hint}")
                return None, None, None
            # compare.py writes comparison.csv FROM the row JSONs, so the two
            # agree exactly unless the row was re-graded afterwards: only
            # run_compare.sh rewrites the table, while a standalone rerun
            # (run_shared_models.sh, run_shared_folds.sh, grade.sh) rewrites
            # the JSON alone.
            j = self._json_metric(cv, model, key)
            if j is not None and abs(j[0] - rec[key][0]) > 1e-9:
                self.miss(f"{model}/{cv} {key}: comparison.csv has "
                          f"{rec[key][0]}, its row JSON {j[0]} (re-graded "
                          f"after the table was written); rerun "
                          f"sl_comparison/slurm/run_compare.sh"
                          f"{self.arm_hint}")
                return None, None, None
            return rec[key][0], rec[key][1], "comparison.csv"
        j = self._json_metric(cv, model, key)
        return (j[0], j[1], "json") if j is not None else (None, None, None)

    def allmod_problem(self, v: str, cv: str) -> Optional[str]:
        """None when the all-modalities run its marker names has the
        arch_signature of v's selected combo outside the modality flag;
        else why not (no check without arch_signature.py or a combo)."""
        marker = (self.sia_results /
                  f"allmod__{variant_prefix(v)}latest_allmod_{cv}.txt")
        if not marker.exists():
            return f"no {marker.name}"
        d = self.root / "siamese_sl" / marker.read_text().strip()
        cfg = read_json(d / "config.json")
        if not cfg:
            return f"{self.rel(d / 'config.json')} missing"
        if not cfg.get("all_modalities"):
            return f"{self.rel(d)} is not an --all_modalities run"
        ref = self.combo_config(v, cv)
        if self.arch is None or ref is None:
            return None
        diff = self.arch.signature_diff(cfg, ref, ignore=("all_modalities",))
        if diff:
            return ("its arch_signature differs from the selected combo's: " +
                    ", ".join(sorted(diff)))
        return None

    def probe_source_problem(self, cv: str) -> Optional[str]:
        """None when the BilinearProbe row of `cv` records (probe_source,
        run_probe.sh) the config.json sha256 of the combo the arm-none
        pd:pooled marker names now; else why not."""
        s = self.json_rows.get((cv, "BilinearProbe")) or {}
        ps = s.get("probe_source")
        if not isinstance(ps, dict) or not ps.get("combo_config_sha256"):
            return "its row records no probe_source (not run_probe.sh's row)"
        marker = (self.sia_results /
                  f"{variant_prefix('pd:pooled')}latest_combo_{cv}.txt")
        if not marker.exists():
            return f"no {marker.name}: the combo it probes is gone"
        cfg = self.root / "siamese_sl" / marker.read_text().strip() / \
            "config.json"
        if not cfg.is_file():
            return f"{self.rel(cfg)} missing"
        import hashlib
        cur = hashlib.sha256(cfg.read_bytes()).hexdigest()
        if cur != ps["combo_config_sha256"]:
            return (f"built on another pd:pooled combo (probe_source "
                    f"{ps.get('combo_dir')}, config sha256 "
                    f"{ps['combo_config_sha256'][:12]} != the current "
                    f"{cur[:12]}); rerun siamese_sl/slurm/run_probe.sh")
        # The trained combo, not only its settings: a retrain with the same
        # arguments writes the same config.json bytes.
        res = cfg.parent / "results.json"
        want = ps.get("combo_results_sha256")
        if not want:
            return ("its probe_source records no combo_results_sha256 (a "
                    "run_probe.sh from before 2026-11-13); rerun "
                    "siamese_sl/slurm/run_probe.sh")
        if not res.is_file() or \
                hashlib.sha256(res.read_bytes()).hexdigest() != want:
            return (f"built on another training of the pd:pooled combo "
                    f"({self.rel(res)} differs from the one the probe "
                    f"read); rerun siamese_sl/slurm/run_probe.sh")
        return None

    # ---- siamese dirs ------------------------------------------------------
    def _load_combo(self, v: str, cv: str) -> Optional[dict]:
        marker = self.sia_results / f"{variant_prefix(v)}latest_combo_{cv}.txt"
        if not marker.exists():
            self.miss(f"{marker} ({v} {cv} combo)")
            self.combo_problem[(v, cv)] = (f"missing: no {marker.name} (no "
                                           f"combo; listed)")
            return None
        # siamese_sl-relative: results[/pca_<arm>]/<prefix>best_cat_..._<cv>
        d = self.root / "siamese_sl" / marker.read_text().strip()
        res = read_json(d / "results.json")
        if res is None:
            self.miss(f"{d / 'results.json'} (named by {marker.name})")
            self.combo_problem[(v, cv)] = (f"missing: the combo {marker.name} "
                                           f"names has no results.json "
                                           f"(listed)")
            return None
        cfg = read_json(d / "config.json") or res.get("config") or {}
        # Not this arm's combo (2026-10-20): a marker naming another arm's
        # namespace, or a combo trained under another PCA (another PCA_ARM,
        # a POST_PCA_* export of before the arms, a hand copy into this
        # namespace). Its widths, sizes and graded row would be published as
        # this arm's. The directory does not say which PCA a run applied, its
        # config.json does (train.py resolves the flags into it); both must
        # be this arm's.
        # The combo's PARENT resolved (os.path.normpath collapses a '..' of
        # the marker text first): a combo directory moved to other storage
        # and linked back is still the arm's, as collect_siamese.py (its name
        # in the arm root) and run_external_transfer.sh (its parent,
        # resolved) take it.
        if Path(os.path.normpath(d)).parent.resolve() != \
                self.sia_results.resolve():
            why = (f"the combo {marker.name} names is not in PCA arm "
                   f"{self.pca_arm}'s results root "
                   f"{self.rel(self.sia_results)}/")
            self.miss(f"{d} ({v} {cv} combo, named by {marker}) is not in "
                      f"PCA arm {self.pca_arm}'s results root "
                      f"{self.sia_results}/: not used")
            self.combo_problem[(v, cv)] = f"stale: {why} (listed)"
            return None
        if self.arch is not None:
            try:
                got, s = self.arch.pca_arm(cfg), self.arch.pca_settings(cfg)
            except (TypeError, ValueError) as e:
                # Not what train.py writes (it resolves the flags into
                # numbers): the arm cannot be told, so neither can the combo.
                why = (f"its config.json's PCA settings are unreadable "
                       f"({type(e).__name__}: {e})")
                self.miss(f"{d} ({v} {cv} combo): {why}: not used")
                self.combo_problem[(v, cv)] = f"stale: {why} (listed)"
                return None
            if got != self.pca_arm:
                why = (f"trained as PCA arm {got} (PCA within each modality "
                       f"{pca_setting_text(s['modality'])}, on the "
                       f"concatenation {pca_setting_text(s['joint'])}), not "
                       f"{self.pca_arm}")
                self.miss(f"{d} ({v} {cv} combo) was {why}: not used "
                          f"(another arm's run, or a hand copy)")
                self.combo_problem[(v, cv)] = f"stale: {why} (listed)"
                return None
        # --skip-siamese reuses the markers' combos while step 0 may have
        # rebuilt the folds: widths / parameter counts / the tab_loco
        # reference of another fold set are not this run's.
        stamp = self.model_params.read_model_params(d).get("fold_stamp")
        diffs = (self.model_params.compare_fold_stamp(stamp, self.folds, cv)
                 if stamp else ["no fold_stamp in model_params.json"])
        if diffs:
            self.miss(f"{d} ({v} {cv} combo) is not on the current folds, "
                      f"ignored: " + "; ".join(diffs))
            self.fold_refused_combos.add((v, cv))
            self.combo_problem[(v, cv)] = ("stale: the combo is not on the "
                                           "current folds (listed)")
            return None
        # A SIAMESE_RESULTS_* pin (or SIAMESE_BASELINE_DISCOVER=all) grades
        # another run than this combo: its params / widths / arch would sit
        # next to that run's metrics.
        sel = self.cmp_results / (("siamese" if v == "pd:per_head" else
                                   variant_row(v)) + f"_selected_{cv}.txt")
        if sel.exists():
            graded = (self.root / sel.read_text().strip()).resolve()
            if graded != d.resolve():
                why = (f"comparison row graded {graded}, {marker.name} names "
                       f"{d}")
                self.combo_not_graded[(v, cv)] = why
                self.miss(f"{variant_row(v)}/{cv}: {why}; the combo's "
                          f"parameter counts are not put on its tab_tiers "
                          f"row, and its PCA widths / arch / tab_loco "
                          f"reference row describe the combo, not that run")
        return {"dir": d, "results": res, "config": cfg}

    def eval_config(self, v: str, split: str) -> dict:
        """eval_finetuning_<split>/config_used.json of v's cv3 combo ({} if
        absent)."""
        if (v, split) not in self._eval_cfg:
            c = self.combos.get((v, "cv3"))
            self._eval_cfg[(v, split)] = (read_json(
                c["dir"] / f"eval_finetuning_{split}" / "config_used.json")
                or {}) if c else {}
        return self._eval_cfg[(v, split)]

    def combo_config(self, v: str, cv: str) -> Optional[dict]:
        c = self.combos.get((v, cv))
        if c is None:  # the arch signature is CV-independent
            c = next((self.combos[(v, o)] for o in CVS
                      if self.combos.get((v, o))), None)
        return c["config"] if c else None

    def freshness(self, d: Path, v: str, cv: str, ref_cfg: Optional[dict],
                  token: Optional[str],
                  ignore: Sequence[str] = ()) -> List[str]:
        """Why directory d is NOT this run's (v, cv) result ([] = fresh);
        token is the run_token.txt it must carry (None: unchecked)."""
        problems = []
        if token:
            tp = d / "run_token.txt"
            if not tp.exists():
                problems.append("no run_token.txt")
            elif tp.read_text().strip() != token:
                problems.append("run_token.txt is another run's")
        res = read_json(d / "results.json") or {}
        cfg = read_json(d / "config.json") or res.get("config") or {}
        kernel, cells = v.split(":")
        if (cfg.get("kernel", "pd"), cfg.get("cell_lines", "per_head")) != (
                kernel, cells):
            problems.append(f"trained as {cfg.get('kernel')}:"
                            f"{cfg.get('cell_lines')}, not {v}")
        if cfg.get("cv_type") not in (None, cv):
            problems.append(f"cv_type {cfg.get('cv_type')}")
        stamp = self.model_params.read_model_params(d).get("fold_stamp")
        if not stamp:
            problems.append("no fold_stamp")
        else:
            diffs = self.model_params.compare_fold_stamp(stamp, self.folds, cv)
            if diffs:
                problems.append("fold_stamp: " + "; ".join(diffs))
        if ref_cfg is not None and self.arch is not None:
            diff = {k: d_ for k, d_ in self.arch.signature_diff(
                cfg, ref_cfg).items() if k not in ignore}
            if diff:
                problems.append("arch_signature differs from the combo: " +
                                ", ".join(sorted(diff)))
        return problems


# =============================================================================
# tab_tiers
# =============================================================================


def fold_param_stats(res: dict) -> dict:
    vals = [m.get("total_params") for m in res.get("fold_metrics") or []]
    vals = [int(x) for x in vals if isinstance(x, (int, float))]
    if not vals:
        return {}
    return {"params_fold_mean": float(np.mean(vals)),
            "params_fold_sd": float(np.std(vals, ddof=0)),
            "params_fold_min": int(min(vals)),
            "params_fold_max": int(max(vals)),
            "params_n_folds": len(vals)}


def build_tiers(ctx: Ctx) -> Tuple[List[dict], str]:
    rows = []
    models = [(variant_row(v), "SLeuth", v) for v in ctx.variants] + \
        [(allmod_row(v), ALLMOD_KIND, v) for v in ctx.variants
         if ctx.allmod] + \
        [(p, "peer", "") for p in TIER_PEERS] + \
        [(b, "baseline", "") for b in BASELINES
         if ctx.pca_arm in BASELINE_ARMS and "pd:pooled" in ctx.variants] + \
        [(CONTROL, "control", "")]   # the probe reads pd:pooled's combo
    if ctx.paper_variant_explicit and ctx.paper_variant not in ctx.variants:
        ctx.miss(f"tab_tiers: the printed SLeuth variant "
                 f"{ctx.paper_variant} (--paper_variant / PAPER_VARIANT) is "
                 f"not among the run's variants ({', '.join(ctx.variants)}); "
                 f"no bold")
    for cv in CVS:
        peers_cv = ctx.tier_peers(cv)
        for model, kind, v in models:
            if kind == "peer" and model not in peers_cv:
                continue  # an optional peer not run (Ctx.tier_peers)
            r = {"cv": cv, "tier": TIER[cv], "model": model, "kind": kind,
                 "variant": v,
                 "display": (f"SLeuth, all modalities ({v})"
                             if kind == ALLMOD_KIND else
                             f"SLeuth ({v})" if v else model),
                 "in_paper_table": (kind in ("peer", "baseline") or
                                    (kind == "SLeuth" and
                                     v == ctx.paper_variant))}
            srcs = set()
            for m in METRICS + ("baseline", ):
                mean, sd, src = ctx.metric(cv, model, m)
                r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
                if src:
                    srcs.add(src)
            tp, _, src = ctx.metric(cv, model, "trainable_params")
            # A SLeuth row's comparison.csv count is the LAST fold's only
            # (model_params.json is rewritten per fold); its size is the
            # params_fold_* columns, so this column holds peers' counts only.
            r["trainable_params"] = (int(tp) if tp is not None
                                     and kind not in ("SLeuth", ALLMOD_KIND)
                                     else None)
            if kind == ALLMOD_KIND:
                # Its row must say it is the all-modalities model
                # (collect_siamese.allmod_record, carried by the re-grade):
                # anything else under its name is not published.
                s = ctx.json_rows.get((cv, model)) or {}
                rec = (s.get("selection_audit") or {}).get("all_modalities")
                if s and not isinstance(rec, dict):
                    ctx.miss(f"tab_tiers {model}/{cv}: its row JSON records "
                             f"no selection_audit.all_modalities (not the "
                             f"all-modalities combo's row); not tabulated")
                    for m in METRICS + ("baseline", ):
                        r[f"{m}_mean"] = r[f"{m}_sd"] = None
                    srcs = set()
                # Same settings as the variant's selected combo, apart from
                # the modality set (2026-11-13): a hand rerun of a failed
                # all-modalities CV with other ARCH_ARGS, PCA or selection
                # split was tabulated (and bootstrapped) beside the combo.
                if r["auprg_mean"] is not None:
                    why = ctx.allmod_problem(v, cv)
                    if why:
                        ctx.miss(f"tab_tiers {model}/{cv}: {why}; not "
                                 f"tabulated")
                        for m in METRICS + ("baseline", ):
                            r[f"{m}_mean"] = r[f"{m}_sd"] = None
                # Per-fold sizes, as collect_and_regrade_siamese.sh writes
                # them for every SLeuth row (compare.py's CSV row).
                pm, psd, _ = ctx.metric(cv, model, "trainable_params_fold")
                if pm is not None and r["auprg_mean"] is not None:
                    r["params_fold_mean"], r["params_fold_sd"] = pm, psd
            if v and kind == "SLeuth":
                c = ctx.combos.get((v, cv))
                stats = fold_param_stats(c["results"]) if c else {}
                if c and not stats:
                    ctx.miss(f"{c['dir']}/results.json fold_metrics"
                             f"[].total_params")
                # The params must be the graded run's: comparison.csv's copy
                # (from that run's results.json) must agree.
                pm, _, _ = ctx.metric(cv, model, "trainable_params_fold")
                if stats and pm is not None and \
                        abs(pm - stats["params_fold_mean"]) > 0.5:
                    ctx.miss(f"tab_tiers {model}/{cv}: trainable_params_fold "
                             f"{pm} in comparison.csv != "
                             f"{stats['params_fold_mean']} of {c['dir']}; "
                             f"params_fold_* left empty")
                    stats = {}
                if (v, cv) in ctx.combo_not_graded:
                    stats = {}  # listed by Ctx._load_combo
                r.update(stats)
            r["status"] = "ok" if r["auprg_mean"] is not None else "missing"
            r["source"] = ",".join(sorted(srcs))
            if r["status"] == "missing":
                # The control too: it decides no bold, but the run is not
                # whole without it. 'usable': absent, or refused by
                # Ctx.metric (its reason is listed above).
                ctx.miss(f"tab_tiers {model}/{cv} (no usable comparison row"
                         + ("; control, never bolded)" if kind == "control"
                            else ")"))
            elif kind == "peer" and r["trainable_params"] is None:
                ctx.miss(f"tab_tiers {model}/{cv}: no trainable_params (the "
                         f"paper's parameter column)")
            if kind == "baseline" and r["status"] == "ok":
                # The probe must be built on THIS run's arm-none pd:pooled
                # combo (2026-11-13): nothing else moves its shared row aside,
                # and a combo retrained on the same folds passes the fold
                # check, so an older probe on other features was published.
                why = ctx.probe_source_problem(cv)
                if why:
                    ctx.miss(f"tab_tiers {model}/{cv}: {why}; not tabulated")
                    for m in METRICS + ("baseline", ):
                        r[f"{m}_mean"] = r[f"{m}_sd"] = None
                    r["status"] = "missing"
            if kind == "baseline":
                # Its input width follows the combo's per-fold modalities, so
                # its size is per fold too (run_probe.sh writes the
                # trainable_params_fold_* keys, compare.py the CSV row).
                pm, psd, _ = ctx.metric(cv, model, "trainable_params_fold")
                if pm is not None:
                    r["params_fold_mean"], r["params_fold_sd"] = pm, psd
                if r["status"] == "ok" and r["trainable_params"] is None:
                    ctx.miss(f"tab_tiers {model}/{cv}: no trainable_params "
                             f"(the paper's parameter column)")
            # The paper prints every cell as mean (sd).
            nosd = [m for m in METRICS if r[f"{m}_mean"] is not None and
                    r[f"{m}_sd"] is None]
            if r["in_paper_table"] and nosd:
                ctx.miss(f"tab_tiers {model}/{cv}: no sd for "
                         f"{', '.join(nosd)} (the paper prints mean (sd))")
            rows.append(r)
    for m in METRICS:
        _flag_best(rows, ["cv"], f"{m}_mean", f"best_{m}",
                   eligible=lambda r: r["in_paper_table"],
                   decimals=PRINTED_DECIMALS)
    for cv in CVS:
        tier = [r for r in rows if r["cv"] == cv]
        absent = set()
        for m in METRICS:
            # The bold is the best of the printed columns: unknown, not the
            # best of the rest, while one of them is absent.
            gone = {r["model"] for r in tier
                    if r["in_paper_table"] and r[f"{m}_mean"] is None}
            if gone:
                absent |= gone
                for r in tier:
                    r[f"best_{m}"] = None
            # Each SLeuth variant against the peers alone (the paper's
            # layout, once per variant); None unless every one of this tier's
            # (Ctx.tier_peers: an optional peer not run is not one).
            peers = [p[f"{m}_mean"] for p in tier
                     if p["kind"] == "peer" and p[f"{m}_mean"] is not None]
            for r in tier:
                if r["kind"] == "SLeuth" and r[f"{m}_mean"] is not None:
                    r[f"beats_peers_{m}"] = (
                        r[f"{m}_mean"] > max(peers)
                        if len(peers) == len(ctx.tier_peers(cv)) else None)
        if absent:
            ctx.miss(f"tab_tiers {cv}: best_* left empty (no "
                     f"{', '.join(sorted(absent))} row)" +
                     ("; beats_peers_* too" if absent & set(TIER_PEERS)
                      else ""))
    have = any(r["status"] == "ok" and r["kind"] == "SLeuth" for r in rows)
    return rows, "ok" if have else "missing"


TIERS_COLS = (["cv", "tier", "model", "display", "kind", "variant",
               "in_paper_table", "trainable_params", "params_fold_mean",
               "params_fold_sd", "params_fold_min", "params_fold_max",
               "params_n_folds"] +
              [f"{m}_{s}" for m in METRICS for s in ("mean", "sd")] +
              ["baseline_mean", "baseline_sd"] +
              [f"best_{m}" for m in METRICS] +
              [f"beats_peers_{m}" for m in METRICS] + ["status", "source"])


# =============================================================================
# tab_solo
# =============================================================================


def self_scored_cell(ctx: Ctx, table: str, r: dict, res: dict, d: Path,
                     keys: Sequence[str]) -> None:
    """Fill r from a train.py results.json: auprg_any mean / sd (its nanmean
    over folds), n_folds / n_folds_finite, status. A missing mean, or a mean
    over fewer finite folds than the fold set has, is listed."""
    s = res.get("summary") or {}
    fm = [m for m in res.get("fold_metrics") or [] if isinstance(m, dict)]
    r["auprg_any_mean"] = _num(s.get("auprg_any_mean"))
    r["auprg_any_sd"] = _num(s.get("auprg_any_std"))
    r["n_folds"] = len(res.get("fold_metrics") or [])
    r["n_folds_finite"] = sum(_num(m.get("auprg_any")) is not None
                              for m in fm)
    if r["auprg_any_mean"] is None:
        r["status"] = "missing: no auprg_any_mean"
        ctx.miss(f"{d}/results.json summary.auprg_any_mean")
    else:
        r["status"] = "ok"
        check_folds(ctx, table, r, keys)


def solo_prefix(ctx: Ctx, v: str) -> str:
    """run_best_per_category.conf's bestcat_solo_prefix: SELECTION_VARIANT's
    runs are the unprefixed dirs Step 2 ranks, so a non-selection pd:per_head
    writes var_pd_per_head__ instead (siamese_variant_prefix gives it "")."""
    if v == ctx.selection_variant:
        return ""
    return variant_prefix(v) or "var_pd_per_head__"


def solo_dir(ctx: Ctx, v: str, etype: str, cv: str) -> Path:
    """A single-modality run of the arm: in the root its selection ranked
    (ctx.step1_results, the owner of the arm's pair), not in the arm's own
    results root."""
    return ctx.step1_results / (f"{solo_prefix(ctx, v)}pcavar"
                                f"{ctx.pca_variance}_{etype}_{cv}")


def solo_variants(ctx: Ctx) -> List[str]:
    """The variants tab:solo can hold rows for: this run's, plus
    SELECTION_VARIANT. run_best_per_category.conf's bestcat_task_variants
    makes block 0 of the Step-1 array SELECTION_VARIANT whatever
    SIAMESE_VARIANTS holds (config.conf validates the two independently), and
    those unprefixed runs are the ones Step 2 ranked and the paper's
    tab:solo — a variant list without the selection variant must not drop
    them."""
    return list(ctx.variants) + ([] if ctx.selection_variant in ctx.variants
                                 else [ctx.selection_variant])


def build_solo(ctx: Ctx) -> Tuple[List[dict], str]:
    rows = []
    not_run: Dict[str, int] = {}
    lack = sorted(set(PAPER_SOLO) - {e for e, _, _ in ctx.catalog})
    if ctx.catalog and lack:
        ctx.miss(f"EMB_CATALOG lacks {len(lack)} of tab:solo's 16 embeddings "
                 f"({', '.join(lack)}): no rows, and the paper-16 bold / "
                 f"means / counts are over the rest")
    for v in solo_variants(ctx):
        is_sel = v == ctx.selection_variant
        for cv in CVS:
            ref = ctx.combo_config(v, cv)
            # A variant this run trained no combo for (SELECTION_VARIANT
            # outside SIAMESE_VARIANTS) has no comparison row of this run: a
            # leftover one would be another run's combo.
            shared = (ctx.metric(cv, variant_row(v), "auprg")[0]
                      if v in ctx.variants else None)
            c = ctx.combos.get((v, cv))
            self_auprg = _num(((c or {}).get("results") or {}).get(
                "summary", {}).get("auprg_any_mean"))
            for etype, efile, cat in ctx.catalog:
                d = solo_dir(ctx, v, etype, cv)
                r = {"variant": v, "model": variant_row(v), "cv": cv,
                     "tier": TIER[cv], "category": cat, "embedding": etype,
                     "in_paper": etype in PAPER_SOLO,
                     "selection_variant": is_sel,
                     "combo_auprg_shared": shared,
                     "combo_auprg_any_self": self_auprg,
                     "dir": str(d.relative_to(ctx.root))}
                res = read_json(d / "results.json")
                solo_per_fold_cells(ctx, r, res)
                if res is None:
                    if not (ctx.root / "data" / efile).exists():
                        r["status"] = "not_generated"
                        # Step 1 skips it with exit 0. An OPTIONAL_TYPES
                        # modality outside the paper's 16 is absent by design;
                        # any other (generation fails on it) comes from a
                        # reused / skipped generation, and a paper-16 row is
                        # absent whatever generation accepted.
                        if etype in ctx.optional_types and \
                                etype not in PAPER_SOLO:
                            ctx.by_design(
                                f"data/{efile} not generated (OPTIONAL_TYPES, "
                                f"siamese_sl/slurm/run_generate_embeddings."
                                f"conf): tab_solo {etype} rows absent (not "
                                f"among the paper's 16)")
                        else:
                            ctx.miss(f"data/{efile} not generated: tab_solo "
                                     f"{etype} rows absent" +
                                     ("" if etype in PAPER_SOLO else
                                      " (not among the paper's 16)"))
                    elif not is_sel and not ctx.extra_variants:
                        r["status"] = "not_run (BESTCAT_EXTRA_VARIANTS=0)"
                        not_run[v] = not_run.get(v, 0) + 1
                    else:
                        r["status"] = "missing"
                        ctx.miss(f"{d}/results.json")
                    rows.append(r)
                    continue
                if ref is None:
                    ctx.note(f"tab_solo {v}: no combo config, arch "
                                     f"signature unchecked")
                probs = ctx.freshness(d, v, cv, ref, ctx.solo_run_token,
                                      ignore=SOLO_SIG_IGNORE)
                bad = embedding_content_problem(res, c)
                if bad:
                    probs = probs + [bad]
                if probs:
                    r["status"] = "stale: " + "; ".join(probs)
                    ctx.miss(f"{d} is stale ({'; '.join(probs)})")
                    rows.append(r)
                    continue
                self_scored_cell(ctx, "tab_solo", r, res, d,
                                 ("cv", "embedding"))
                for key, ref_v in (("above_combo_shared", shared),
                                   ("above_combo_self", self_auprg)):
                    r[key] = (None if ref_v is None or
                              r["auprg_any_mean"] is None else
                              bool(r["auprg_any_mean"] > ref_v))
                rows.append(r)
    for v, n in not_run.items():
        ctx.by_design(f"tab_solo {v}: {n} cell(s) not_run (the single-"
                      f"modality runs of variants other than "
                      f"SELECTION_VARIANT {ctx.selection_variant} are off: "
                      f"BESTCAT_EXTRA_VARIANTS=0)")
    for v, cv, cat, gone in flag_solo_best(rows):
        ctx.miss(f"tab_solo {v}/{cv}/{cat}: best_in_category_paper16 left "
                 f"empty (no fresh {', '.join(gone)} row)")
    have = any(r["status"] == "ok" and r["selection_variant"] for r in rows)
    return rows, "ok" if have else "missing"


def flag_solo_best(rows: List[dict]) -> List[tuple]:
    """tab:solo's bold, per variant x tier x category: over every catalog
    embedding that ran (best_in_category, the exact maximum) and over the
    paper's 16 printed rows (best_in_category_paper16, the paper's bold: every
    row printing the best value at PRINTED_DECIMALS). None for a category with
    a single candidate: the caption marks no best there (kg). The paper's bold
    is also None for a group where one of the category's paper-16 members (a
    row of every catalog entry exists, whatever its status) is not fresh
    while another is; those (variant, cv, category, absent) are returned."""
    keys = ["variant", "cv", "category"]
    _flag_best(rows, keys, "auprg_any_mean", "best_in_category",
               eligible=lambda r: r["status"] == "ok", min_group=2)
    _flag_best(rows, keys, "auprg_any_mean", "best_in_category_paper16",
               eligible=lambda r: r["status"] == "ok" and r["in_paper"],
               min_group=2, decimals=PRINTED_DECIMALS)
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    cleared = []
    for (v, cv, cat), grp in groups.items():
        want = {r["embedding"] for r in grp if r["in_paper"]}
        ok = {r["embedding"] for r in grp if r["status"] == "ok"} & want
        gone = sorted(want - ok)
        if gone and ok:
            for r in grp:
                r["best_in_category_paper16"] = None
            cleared.append((v, cv, cat, gone))
    return cleared


REGRADE_COLS = ("auprg_shared_mean", "auprg_shared_sd", "regrade")
SOLO_COLS = ("variant", "model", "cv", "tier", "category", "embedding",
             "in_paper", "selection_variant", "auprg_any_mean",
             "auprg_any_sd", "n_folds", "n_folds_finite", "best_in_category",
             "best_in_category_paper16", "above_combo_shared",
             "above_combo_self", "combo_auprg_shared",
             "combo_auprg_any_self", "status", "dir") + REGRADE_COLS
# A per_fold selection's extra tab_solo columns: how many folds of the tier
# picked the embedding (its category's winner on that fold's selection
# split), which, and its mean score on those splits (the selection metric).
SOLO_PER_FOLD_COLS = ("n_folds_picked", "picked_folds", "sel_score_mean")


def solo_cols(ctx: Ctx) -> Tuple[str, ...]:
    """tab_solo's columns (SOLO_COLS; a per_fold selection adds
    SOLO_PER_FOLD_COLS after selection_variant)."""
    if not per_fold_selection(ctx):
        return SOLO_COLS
    i = SOLO_COLS.index("selection_variant") + 1
    return SOLO_COLS[:i] + SOLO_PER_FOLD_COLS + SOLO_COLS[i:]


def solo_per_fold_cells(ctx: Ctx, r: dict, res: Optional[dict]) -> None:
    """Fill SOLO_PER_FOLD_COLS of a tab_solo row (per_fold selection only;
    a non-selection variant's row is never ranked, so it picks nothing)."""
    if not per_fold_selection(ctx):
        return
    picks = (per_fold_picks(ctx, r["cv"], r["category"])
             if r["selection_variant"] else {})
    mine = sorted(k for k, e in picks.items() if e == r["embedding"])
    r["n_folds_picked"] = len(mine) if r["selection_variant"] else None
    r["picked_folds"] = ";".join(map(str, mine))
    key = ((ctx.selection or {}).get("_selection_meta") or {}).get(
        "selection_metric")
    vals = []
    if key and isinstance(res, dict):
        for m in ctx.fsel.fold_entries(res, 10 ** 6).values():
            x = _num(m.get(key))
            if x is not None:
                vals.append(x)
    r["sel_score_mean"] = float(np.mean(vals)) if vals else None


def _category_means(rows: List[dict]) -> Dict[str, float]:
    cats: Dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r["auprg_any_mean"])
    return {k: float(np.mean(x)) for k, x in cats.items()}


def solo_summaries(ctx: Ctx, rows: List[dict]) -> dict:
    """AB2 category means and AB3 'beats the multi-modal model' counts.

    The paper's are over its 16 (those of the catalog: rows carry every
    catalog entry, whatever its status). With one of them not fresh at a
    tier, that tier's paper-16 mean of its category, best_category_paper16
    and n_above_combo_*_paper16 are None (paper16_absent names them, listed);
    an embedding without a fresh row or reference at some tier is undecided
    in 'at every tier', which is None while one of the 16 is."""
    out = {}
    for v in solo_variants(ctx):
        allr = [r for r in rows if r["variant"] == v]
        vr = [r for r in allr if r["status"] == "ok"]
        if not vr:
            continue
        paper = {r["embedding"]: r["category"] for r in allr if r["in_paper"]}
        per_cv = {}
        # Tiers whose combo reference is known: "at every tier" needs all.
        n_tiers = {"shared": 0, "self": 0}
        # basis -> embedding -> {cv: above the combo; None = unknown}
        state = {b: {e: {} for e in sorted({r["embedding"] for r in allr})}
                 for b in ("shared", "self")}
        for cv in CVS:
            cr = [r for r in vr if r["cv"] == cv]
            if not cr:
                continue
            absent16 = sorted(set(paper) - {r["embedding"] for r in cr})
            short = {paper[e] for e in absent16}
            if absent16:
                ctx.miss(f"numbers.json solo {v}/{cv}: no fresh "
                         f"{', '.join(absent16)} row; the paper-16 means of "
                         f"{', '.join(sorted(short))}, best_category_paper16 "
                         f"and n_above_combo_*_paper16 left empty")
            means = _category_means(cr)
            # The paper averages over its 16 only; the catalog may hold more.
            means16 = {c: (None if c in short else x) for c, x in
                       _category_means([r for r in cr
                                        if r["in_paper"]]).items()}
            rec = {"category_means": means,
                   "best_category": max(means, key=means.get),
                   "category_means_paper16": means16,
                   "best_category_paper16": max(means16, key=means16.get)
                   if means16 and not absent16 else None,
                   "paper16_absent": absent16,
                   "n_embeddings": len(cr),
                   "n_embeddings_in_paper16": sum(r["in_paper"] for r in cr)}
            by_emb = {r["embedding"]: r for r in cr}
            for basis, key in (("shared", "above_combo_shared"),
                               ("self", "above_combo_self")):
                above = sorted(r["embedding"] for r in cr if r.get(key))
                known = [r for r in cr if r.get(key) is not None]
                rec[f"above_combo_{basis}"] = above if known else None
                rec[f"n_above_combo_{basis}"] = len(above) if known else None
                rec[f"n_above_combo_{basis}_paper16"] = (
                    sum(e in paper for e in above)
                    if known and not absent16 else None)
                if known:
                    n_tiers[basis] += 1
                    rec[f"combo_auprg_{basis}"] = cr[0][
                        "combo_auprg_shared" if basis == "shared" else
                        "combo_auprg_any_self"]
                for e, st in state[basis].items():
                    st[cv] = (by_emb.get(e) or {}).get(key)
            per_cv[cv] = rec
        every = {}
        for b, st in state.items():
            yes = sorted(e for e, s in st.items()
                         if all(s.get(cv) is True for cv in CVS))
            open_ = sorted(e for e, s in st.items() if e not in yes and
                           not any(s.get(cv) is False for cv in CVS))
            decided = n_tiers[b] == len(CVS) and not set(open_) & set(paper)
            if n_tiers[b] == len(CVS) and not decided:
                ctx.miss(f"numbers.json solo {v}: above_combo_at_every_tier_"
                         f"{b} left empty (undecided: "
                         f"{', '.join(e for e in open_ if e in paper)})")
            every[f"above_combo_at_every_tier_{b}"] = yes if decided else None
            every[f"above_combo_at_every_tier_{b}_undecided"] = open_
        out[v] = {"per_cv": per_cv, **every}
    return out


# =============================================================================
# tab_loco
# =============================================================================


def per_fold_selection(ctx: Ctx) -> bool:
    """The arm's selection is SELECTION_MODE=per_fold (and readable: its
    absent fold_selection.py is listed by Ctx)."""
    return (getattr(ctx, "selection_mode", "global") == "per_fold"
            and getattr(ctx, "fsel", None) is not None)


def per_fold_picks(ctx: Ctx, cv: str, cat: str) -> Dict[int, str]:
    """{fold: embedding} a per_fold selection picked for cat at cv ({}
    otherwise)."""
    info = ((ctx.selection or {}).get(cv) or {}).get(cat)
    per = info.get("per_fold") if isinstance(info, dict) else None
    if not isinstance(per, dict):
        return {}
    return {int(k): str(w.get("type")) for k, w in per.items()
            if isinstance(w, dict)}


def selection_winners(ctx: Ctx, cv: Optional[str] = None) -> Dict[str, str]:
    """{category: winner}. A per_fold selection names every embedding a fold
    picked, 'a/b' (catalog order), at cv, else over every CV."""
    sel = ctx.selection or {}
    if per_fold_selection(ctx):
        out: Dict[str, List[str]] = {}
        for x in ([cv] if cv else CVS):
            for cat, info in (sel.get(x) or {}).items():
                if isinstance(info, dict):
                    ts = out.setdefault(cat, [])
                    ts += [str(e) for e in info.get("types") or
                           [info.get("type")] if e and str(e) not in ts]
        return {c: "/".join(ts) for c, ts in out.items()}
    for cv in CVS:
        if isinstance(sel.get(cv), dict):
            return {cat: info.get("type") for cat, info in sel[cv].items()
                    if isinstance(info, dict)}
    return {}


def selection_files(ctx: Ctx) -> Dict[str, str]:
    """Each category winner's embedding FILE name (run_ablation.sh drops
    exactly this file from the combo's list)."""
    sel = ctx.selection or {}
    for cv in CVS:
        if isinstance(sel.get(cv), dict):
            return {cat: Path(info["file"]).name
                    for cat, info in sel[cv].items()
                    if isinstance(info, dict) and info.get("file")}
    return {}


def loco_modality_problem(ctx: Ctx, d: Path, res: dict, cat: str,
                          drop_file: Optional[str],
                          combo: Optional[dict],
                          cv: Optional[str] = None) -> Optional[str]:
    """Why d is not 'the combo minus cat's winner' (None = it is, or there is
    nothing to compare it with). freshness() cannot see this: arch_signature
    carries no embeddings_paths, and an ablate_no_<cat>_<cv> name records no
    modality set, so an ablation of an EARLIER selection survives a
    re-selection with the same folds and architecture and would be tabulated
    under this run's winner. run_ablation_summary.sh applies the same rule
    (there in the combo's sorted-category order; compared as a set here, so a
    different ordering of the same modalities is not called stale).
    Under a per_fold selection (cv given): every fold's own winners minus
    cat's pick, over their union, as run_ablation.sh trains it
    (fold_selection.combo_spec / run_matches_spec)."""
    if per_fold_selection(ctx) and cv is not None:
        if not combo:
            return None
        cfg = read_json(d / "config.json") or res.get("config") or {}
        try:
            spec = ctx.fsel.combo_spec(ctx.selection, cv, drop=cat)
            why = ctx.fsel.run_matches_spec(cfg, spec, d)
        except (ctx.fsel.SelectionError, KeyError, TypeError,
                ValueError) as e:
            why = f"{type(e).__name__}: {e}"
        return (None if why is None else
                f"{why} (want each fold's winners minus {cat}'s pick)")
    if not combo or not drop_file:
        return None
    ref = [Path(p).name for p in combo["config"].get("embeddings_paths") or []]
    if drop_file not in ref:
        return None
    cfg = read_json(d / "config.json") or res.get("config") or {}
    used = sorted(Path(p).name for p in cfg.get("embeddings_paths") or [])
    want = sorted(n for n in ref if n != drop_file)
    if used == want:
        return None
    return (f"trained on {', '.join(used) or 'nothing'}, not the combo minus "
            f"{cat}'s winner {drop_file}")


def embedding_content_problem(res: dict,
                              combo: Optional[dict],
                              name: Optional[str] = None) -> Optional[str]:
    """Why a run was not trained on the embedding files its variant's combo
    was trained on (None = it was, or one of the two records nothing to
    compare). train.py records the sha256 of every --embeddings_paths file in
    results.json (embeddings_sha256, in config.embeddings_paths order). A
    file rebuilt at the same path between two trainings (a generation run by
    hand; node2vec / go2vec / onto2vec are not bit-reproducible) keeps its
    name, generator stamp and gene order, so freshness() and
    loco_modality_problem() both pass, and a single-modality or ablation run
    on one content would be tabulated next to a combo on the other
    (eval_finetuning.py, zero_cell_emb.py and the pair export compare the
    same record with the file on disk, but nothing loads these dirs'
    checkpoints). Compared by file name, on the files both runs list (an
    ablation lacks the dropped winner by design); no file is hashed here, so
    a copy of the results without data/ is judged the same way. name: what
    the message calls the combo, by default its directory's bare name; the
    cross-arm table passes the reference arm's combo by its root-relative
    path, as cellline_reference_problem names it: every arm's combo of a
    (variant, CV) can have that bare name (both arms of a pair select the
    same winners), so the bare name read as the stale combo's own
    (2026-10-21)."""
    if not combo:
        return None

    def recorded(r: dict) -> Dict[str, str]:
        paths = (r.get("config") or {}).get("embeddings_paths")
        shas = r.get("embeddings_sha256")
        if not (isinstance(paths, list) and isinstance(shas, list)
                and len(paths) == len(shas)):
            return {}
        return {Path(str(p)).name: str(s) for p, s in zip(paths, shas)}

    mine, ref = recorded(res), recorded(combo["results"])
    diff = sorted(n for n in mine if n in ref and mine[n] != ref[n])
    if not diff:
        return None
    return (f"trained on another content of {', '.join(diff)} than "
            f"{name or combo['dir'].name} (embeddings_sha256: rebuilt in "
            f"place between the two trainings)")


def _fold_maps(fms: Sequence[list],
               key: str) -> Optional[List[Dict[int, Optional[float]]]]:
    """Each run's fold_metrics as {fold: key's value (None unless finite)},
    on the same shared splits: keyed by 'fold' when every entry of every list
    records an int one, else by position when the lists have one length.
    None otherwise, or with a list empty."""
    fms = [[m for m in fm or [] if isinstance(m, dict)] for fm in fms]
    if all(fms) and all(isinstance(m.get("fold"), int)
                        for fm in fms for m in fm):
        return [{m["fold"]: _num(m.get(key)) for m in fm} for fm in fms]
    if fms[0] and all(len(fm) == len(fms[0]) for fm in fms):
        return [{i: _num(m.get(key)) for i, m in enumerate(fm)} for fm in fms]
    return None


def _paired(fm: list, ref_fm: list, key: str) -> Optional[List[float]]:
    """The fold-by-fold differences (run - reference) of key on the same
    shared splits, aligned by _fold_maps; None when no fold pairs up."""
    maps = _fold_maps([fm, ref_fm], key)
    if maps is None:
        return None
    a, b = maps
    d = [a[k] - b[k] for k in sorted(set(a) & set(b))
         if a[k] is not None and b[k] is not None]
    return d or None


def _spread(x: Sequence[float]) -> dict:
    """{mean, sd (population), se (sample sd / sqrt(n); None for one value),
    n} of a list of paired differences."""
    n = len(x)
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=0)),
            "se": (float(np.std(x, ddof=1) / math.sqrt(n)) if n > 1
                   else None),
            "n": n}


def paired_deltas(fm: list, ref_fm: list,
                  key: str = "auprg_any") -> Optional[dict]:
    """Fold-by-fold (ablation - reference) key on the same shared splits, the
    paired spread tab:loco's caption says to read differences against (the
    per-split prevalence swing cancels). Aligned by 'fold', else by position
    when both lists have the same length. None when no fold pairs up."""
    d = _paired(fm, ref_fm, key)
    if d is None:
        return None
    return {"delta_paired_mean": float(np.mean(d)),
            "delta_paired_sd": float(np.std(d, ddof=0)), "n_paired": len(d)}


def build_loco(ctx: Ctx) -> Tuple[List[dict], str]:
    rows = []
    winners = selection_winners(ctx)
    files = selection_files(ctx)
    for v in ctx.variants:
        for cv in CVS:
            c = ctx.combos.get((v, cv))
            ref_fm = (c["results"].get("fold_metrics") or []) if c else []
            ref_mean = None
            ref = {"variant": v, "model": variant_row(v), "cv": cv,
                   "tier": TIER[cv], "dropped": "none",
                   "removed_embedding": "", "status": "missing"}
            if c:
                self_scored_cell(ctx, "tab_loco", ref, c["results"],
                                 c["dir"], ("cv", "dropped"))
                ref_mean = ref["auprg_any_mean"]
                if ref["status"] != "ok":
                    ref["status"] = "missing"
                ref.update({"delta_vs_reference": 0.0 if ref_mean is not None
                            else None,
                            "dir": str(c["dir"].relative_to(ctx.root))})
                ref["shared_eval_auprg"] = ctx.metric(cv, variant_row(v),
                                                      "auprg")[0]
            rows.append(ref)
            pf = per_fold_selection(ctx)
            wcv = selection_winners(ctx, cv) if pf else winners
            for cat in ctx.ablate_categories:
                d = ctx.sia_results / f"{variant_prefix(v)}ablate_no_{cat}_{cv}"
                r = {"variant": v, "model": variant_row(v), "cv": cv,
                     "tier": TIER[cv], "dropped": cat,
                     "removed_embedding": wcv.get(cat, ""),
                     "dir": str(d.relative_to(ctx.root))}
                if pf:
                    # per_fold: what each fold of this tier removed
                    r["removed_per_fold"] = ";".join(
                        f"{k}:{e}" for k, e in sorted(
                            per_fold_picks(ctx, cv, cat).items()))
                res = read_json(d / "results.json")
                if res is None:
                    r["status"] = "missing"
                    ctx.miss(f"{d}/results.json")
                    rows.append(r)
                    continue
                probs = ctx.freshness(d, v, cv, c["config"] if c else None,
                                      ctx.ablation_run_token)
                bad = loco_modality_problem(ctx, d, res, cat,
                                            files.get(cat), c, cv)
                if bad:
                    probs = probs + [bad]
                bad = embedding_content_problem(res, c)
                if bad:
                    probs = probs + [bad]
                if probs:
                    r["status"] = "stale: " + "; ".join(probs)
                    ctx.miss(f"{d} is stale ({'; '.join(probs)})")
                    rows.append(r)
                    continue
                self_scored_cell(ctx, "tab_loco", r, res, d, ("cv", "dropped"))
                if r["status"] != "ok":
                    r["status"] = "missing"
                r["delta_vs_reference"] = (
                    r["auprg_any_mean"] - ref_mean
                    if r["auprg_any_mean"] is not None and ref_mean is not None
                    else None)
                if c and r["status"] == "ok":
                    paired = paired_deltas(res.get("fold_metrics"), ref_fm)
                    if paired is None:
                        ctx.miss(f"{d}/results.json fold_metrics[].auprg_any "
                                 f"pairing with {c['dir']} (paired delta)")
                    else:
                        r.update(paired)
                        if paired["n_paired"] != len(ref_fm):
                            ctx.note(f"tab_loco {v}/{cv}/{cat}: "
                                     f"{paired['n_paired']} of {len(ref_fm)} "
                                     f"folds paired")
                rows.append(r)
    have = any(r["status"] == "ok" and r["dropped"] != "none" for r in rows)
    return rows, "ok" if have else "missing"


LOCO_COLS = ("variant", "model", "cv", "tier", "dropped", "removed_embedding",
             "auprg_any_mean", "auprg_any_sd", "n_folds", "n_folds_finite",
             "delta_vs_reference", "delta_paired_mean", "delta_paired_sd",
             "n_paired", "shared_eval_auprg", "status",
             "dir") + REGRADE_COLS


def loco_cols(ctx: Ctx) -> Tuple[str, ...]:
    """tab_loco's columns: a per_fold selection adds removed_per_fold
    (fold:embedding each fold dropped); a global one's are LOCO_COLS."""
    if not per_fold_selection(ctx):
        return LOCO_COLS
    i = LOCO_COLS.index("removed_embedding") + 1
    return LOCO_COLS[:i] + ("removed_per_fold",) + LOCO_COLS[i:]


# =============================================================================
# PCA settings, widths and the 2x2 (the PCA arms; tab_pca is cross-arm mode's)
# =============================================================================


def pca_setting_text(stage) -> str:
    """One stage of arch_signature.pca_settings() as a table cell: 'no',
    'variance 0.8' (the fraction of the correlation spectrum kept, resolved
    per fold), 'dims 64 64 64' (a count per modality) or 'count 512' (the
    joint width)."""
    if stage is None:
        return "no"
    kind, val = stage
    if isinstance(val, (list, tuple)):
        val = " ".join(str(x) for x in val)
    return f"{kind} {val}"


def fold_widths(res: dict) -> Optional[List[int]]:
    """fold_metrics[].input_dim (the encoder's input width, after every PCA)
    in fold order; None when a fold records none."""
    fm = [m for m in res.get("fold_metrics") or [] if isinstance(m, dict)]
    if not fm or any(not isinstance(m.get("input_dim"), (int, float))
                     for m in fm):
        return None
    if all(isinstance(m.get("fold"), int) for m in fm):
        fm = sorted(fm, key=lambda m: m["fold"])
    return [int(m["input_dim"]) for m in fm]


def pca_effects(fms: Dict[str, list],
                key: str = "auprg_any") -> Optional[dict]:
    """The 2x2's effects, fold by fold on the same shared splits, from the
    four arms' per-fold records ({arm: [per-fold dicts]}; cross-arm mode
    passes each arm's graded summary.folds with key 'auprg'): per_modality =
    ((per_modality - none) + (both - joint)) / 2 and joint = ((joint - none)
    + (both - per_modality)) / 2 (each stage's change, averaged over the
    other stage's two levels), interaction = (both - per_modality) - (joint -
    none). Each {mean, sd (population), se (sample sd / sqrt(n); None for one
    fold), n} over the folds all four scored, aligned as paired_deltas
    aligns two runs. None when no fold has all four."""
    maps = _fold_maps([fms[a] for a in ("none", "joint", "per_modality",
                                        "both")], key)
    if maps is None:
        return None
    none, joint, permod, both = maps
    folds = [k for k in sorted(set(none) & set(joint) & set(permod) &
                               set(both))
             if all(m[k] is not None for m in maps)]
    if not folds:
        return None
    eff = {"per_modality": [((permod[k] - none[k]) + (both[k] - joint[k])) / 2
                            for k in folds],
           "joint": [((joint[k] - none[k]) + (both[k] - permod[k])) / 2
                     for k in folds],
           "interaction": [(both[k] - permod[k]) - (joint[k] - none[k])
                           for k in folds]}
    return {name: _spread(x) for name, x in eff.items()}


# =============================================================================
# Optional shared-evaluator re-grade of tab_solo / tab_loco cells
# =============================================================================

REGRADE_DIR = "regraded"


def _evaluate(pred_dir: Path, folds: Path, cv: str, name: str,
              out: Path) -> bool:
    """Grade one run's fold matrices with evaluate_model.py (as tab_tiers)."""
    import subprocess
    cmd = [sys.executable, str(HERE / "evaluate_model.py"), "--folds_dir",
           str(folds), "--cv_type", cv, "--pred_dir", str(pred_dir),
           "--model_name", name, "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  regrade {pred_dir} failed:\n{(r.stderr or r.stdout)[-800:]}")
    return r.returncode == 0 and out.exists()


def regrade(ctx: Ctx, built: Dict[str, tuple]) -> None:
    """tab_solo / tab_loco are train.py's self-scored auprg_any while
    tab_tiers uses the shared evaluator; put the shared AUPRG next to every
    fresh cell whose score matrices were kept (--regrade). tab_loco's
    reference row is the combo, whose shared AUPRG is tab_tiers' own row."""
    n_folds = int(ctx.meta.get("num_folds") or 0)
    for table in ("tab_solo", "tab_loco"):
        for r in built[table][0]:
            if r.get("status") != "ok" or r.get("dropped") == "none":
                continue
            d = ctx.root / r["dir"]
            mats = sorted(d.glob(f"fold_*_{r['cv']}_predictions.npy"))
            if not n_folds or len(mats) != n_folds:
                r["regrade"] = f"skipped: {len(mats)} of {n_folds} matrices"
                continue
            name = f"{r['model']}__{d.name}"
            out = ctx.out_dir / REGRADE_DIR / f"{name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            s = ((read_json(out) or {}).get("summary") or {}) if _evaluate(
                d, ctx.folds, r["cv"], name, out) else {}
            if _num(s.get("auprg_mean")) is None:
                r["regrade"] = "failed"
                ctx.miss(f"regrade of {d}")
                continue
            r["auprg_shared_mean"] = float(s["auprg_mean"])
            r["auprg_shared_sd"] = _num(s.get("auprg_std"))
            r["regrade"] = "ok"


# =============================================================================
# tab_cellline
# =============================================================================


CELLLINE_TRAINED = "as-trained"      # zero_cell_emb.ROW_TRAINED
CELLLINE_REMOVED = "cell_emb=0"      # zero_cell_emb.ROW_REMOVED (graded row)


def cellline_result(ctx: Ctx, v: str, cv: str) -> Optional[dict]:
    """zero_cell_emb.py's results.json for (v, cv), flat or keyed by the CV;
    None (listed as missing) when absent."""
    p = ctx.sia_results / f"{variant_prefix(v)}cellline_ablation_{cv}" / \
        "results.json"
    data = read_json(p)
    if data is None:
        ctx.miss(f"{p}")
        return None
    return data if isinstance(data.get("agg"), dict) else (data.get(cv) or {})


def _agg_pair(x) -> Tuple[Optional[float], Optional[float]]:
    """(mean, sd) of a zero_cell_emb agg entry ([mean, sd] or a bare mean)."""
    if isinstance(x, (list, tuple)):
        return _num(x[0]) if x else None, _num(x[1]) if len(x) > 1 else None
    return _num(x), None


def cellline_selfcheck(res: Optional[dict], included: dict) -> Tuple[
        Optional[str], Optional[float]]:
    """zero_cell_emb.py's as-trained row vs the comparison row it should
    reproduce (same test edges, same OR-collapse). (status, max |diff|)."""
    if res is None:
        return None, None
    agg = res.get("agg")
    trained = agg.get(CELLLINE_TRAINED) if isinstance(agg, dict) else None
    if not isinstance(trained, dict):
        return "unreadable", None
    diffs = []
    for m in METRICS:
        got = _agg_pair(trained.get(m))[0]
        if got is not None and included.get(f"{m}_mean") is not None:
            diffs.append(abs(got - included[f"{m}_mean"]))
    if not diffs:
        return "unreadable", None
    return ("match" if max(diffs) < 5e-3 else "MISMATCH"), float(max(diffs))


def cellline_source_problem(res: Optional[dict],
                            n_folds: int) -> Optional[str]:
    """Why zero_cell_emb.py's results.json cannot vouch for its rows, or
    None: it still writes one (and exits 1) for a failed self-check or a
    partial fold set, deleting only the matrices."""
    if not res:
        return None
    st = res.get("status")
    if st != "ok":
        return f"zero_cell_emb.py status {st}"
    if (res.get("self_check") or {}).get("ok") is not True:
        return "zero_cell_emb.py's own self-check did not pass"
    if n_folds and res.get("n_folds") != n_folds:
        return (f"zero_cell_emb.py scored {res.get('n_folds')} of {n_folds} "
                f"folds")
    return None


# zero_cell_emb.SELFCHECK_KEYS: its metric name -> the key of the source
# combo's results.json fold_metrics[k] that it copies, as read, into its own
# results.json (self_check.per_fold[k].reference).
CELLLINE_REF_KEYS = {"auroc": "auroc_any", "ap": "aupr_any",
                     "auprg": "auprg_any"}


def cellline_reference_problem(res: Optional[dict],
                               combo: Optional[dict],
                               arm_hint: str = "",
                               name: Optional[str] = None) -> Optional[str]:
    """Why zero_cell_emb.py's results.json is not an ablation of the combo
    NOW on disk, or None. The combo job replaces a combo under the same
    directory name (the winners fix it), so source_model_dir still matches a
    retrained combo, and the 5e-3 cross-check against the included row passes
    any retrain that moves the three fold means by less. What a retrain does
    change is the combo's own results.json: zero_cell_emb.py copied its
    per-fold test metrics from there when it ran (CELLLINE_REF_KEYS), and
    both files hold the same JSON floats, so they are compared exactly, by
    fold id. A fold without a recorded reference (an older zero_cell_emb.py)
    is not judged. arm_hint (Ctx.arm_hint) names the PCA arm BOTH reruns need,
    arm none's included (without PCA_ARM, run_cell_line_ablation.sh redoes
    the ablation of the arm the submitting shell exports, arm none's only
    where none or nothing is), and name the combo (Ctx.rel of its directory:
    its bare name is every arm's; default that name)."""
    if not res or not combo:
        return None
    now = {}
    for pos, m in enumerate(combo["results"].get("fold_metrics") or []):
        if isinstance(m, dict):
            k = m.get("fold")  # recorded by newer train.py, else positional
            now[k if isinstance(k, int) else pos] = m
    for chk in (res.get("self_check") or {}).get("per_fold") or []:
        ref = chk.get("reference") if isinstance(chk, dict) else None
        if not isinstance(ref, dict):
            continue
        cur = now.get(chk.get("fold")) or {}
        for ours, key in CELLLINE_REF_KEYS.items():
            was, got = _num(ref.get(ours)), _num(cur.get(key))
            if was != got:
                return (f"zero_cell_emb.py ablated "
                        f"{name or combo['dir'].name} when its fold "
                        f"{chk.get('fold')} {key} was {was!r}; its "
                        f"results.json now has {got!r} (the combo was "
                        f"retrained in place since: from siamese_sl/ resubmit "
                        f"cell_line_ablation/run_cell_line_ablation.sh"
                        f"{arm_hint}, then from the repo root sl_comparison/"
                        f"slurm/run_compare.sh{arm_hint})")
    return None


def cellline_self_rows(res: Optional[dict], base: dict,
                       row: str) -> List[dict]:
    """zero_cell_emb.py's other rows (e.g. +head_proj_bias=mean / =0, which
    also remove what per-head c_h keeps apart after cell_emb=0): same test
    edges and metrics, forward-pass scores. Deltas are against its own
    as-trained row, i.e. on the same evaluator."""
    agg = (res or {}).get("agg")
    if not isinstance(agg, dict):
        return []
    trained = agg.get(CELLLINE_TRAINED) or {}
    out = []
    for name in (res.get("rows") or list(agg)):
        a = agg.get(name)
        if name in (CELLLINE_TRAINED, CELLLINE_REMOVED) or \
                not isinstance(a, dict):
            continue
        r = {**base, "condition": f"self-scored: {name}",
             "model": f"{row} (zero_cell_emb.py)"}
        for m in METRICS:
            r[f"{m}_mean"], r[f"{m}_sd"] = _agg_pair(a.get(m))
            t = _agg_pair(trained.get(m))[0]
            r[f"delta_{m}_vs_included"] = (
                r[f"{m}_mean"] - t if r[f"{m}_mean"] is not None and
                t is not None else None)
        r["status"] = "ok" if r["auprg_mean"] is not None else "missing"
        out.append(r)
    return out


def recorded_dir_differs(ctx: Ctx, recorded, d: Path) -> bool:
    """Whether a directory zero_cell_emb.py recorded (results.json
    source_model_dir, a row's model_params source_combo) is NOT d, the combo.
    It records the --model_dir it was given, the latest_combo marker's
    siamese_sl-relative text (cell_line_ablation/run_cell_line_ablation.sh
    runs it from siamese_sl/), so a path is compared as a path, resolved
    under siamese_sl/ (else under the root, a hand run from there; else as
    written when absolute): arm none's results/best_cat_<names>_<cv> and arm
    X's results/pca_X/best_cat_<names>_<cv> share their name. A bare name (a
    record that kept no directory) can only be compared by name."""
    p = Path(str(recorded))
    if len(p.parts) <= 1:
        return p.name != d.name
    want = d.resolve()
    cands = [p] if p.is_absolute() else [ctx.root / "siamese_sl" / p,
                                         ctx.root / p]
    return all(q.resolve() != want for q in cands)


def build_cellline(ctx: Ctx) -> Tuple[List[dict], str]:
    rows = []
    n_folds = int(ctx.meta.get("num_folds") or 0)
    for v in ctx.variants:
        kernel, cells = v.split(":")
        row = variant_row(v)
        for cv in CVS:
            base = {"variant": v, "cv": cv, "tier": TIER[cv]}
            # A combo trained without --use_cell_lines is the single-output
            # SiameseSL whatever its config's kernel / cell_lines defaults
            # say: it has no cell-line embedding to zero, and with
            # USE_CELL_LINES=0 the orchestrator submits no ablation either, so
            # nothing is missing here.
            single = (ctx.combo_config(v, cv) or {}).get(
                "use_cell_lines") is False
            if cells == "pooled" or single:
                rows.append({**base, "condition": "n/a", "model": row,
                             "status": "not applicable: trained without "
                             "--use_cell_lines (the single-output SiameseSL "
                             "has no cell-line embedding)" if single else
                             "not applicable: a pooled variant has "
                             "no cell-line embedding (it is the retrained "
                             f"counterpart of {kernel}:per_head)"})
                continue
            def cell(condition, model):
                r = {**base, "condition": condition, "model": model}
                for m in METRICS:
                    r[f"{m}_mean"], r[f"{m}_sd"], _ = ctx.metric(cv, model, m)
                r["status"] = "ok" if r["auprg_mean"] is not None \
                    else "missing"
                return r

            res = cellline_result(ctx, v, cv)
            inc = cell("included", row)
            sc, sc_diff = cellline_selfcheck(res, inc)
            inc["selfcheck_as_trained"], inc["selfcheck_max_abs_diff"] = \
                sc, sc_diff
            rem = cell("removed (cell_emb=0)", f"{row}_nocell")
            if rem["status"] == "missing":
                # Graded into the arm's own sl_comparison root (grade.sh).
                ctx.miss(f"{row}_nocell/{cv} "
                         f"({ctx.cmp_results / f'{row}_nocell_{cv}.json'})")
            # zero_cell_emb.py leaves the cross-check against the graded row
            # to us: unless its run is clean, it ablated this combo AS IT IS
            # NOW ON DISK (the name alone survives a retrain in place) and
            # its as-trained row reproduces the included row, its rows are
            # another model's and removed - included differences two models.
            c = ctx.combos.get((v, cv))
            src = (res or {}).get("source_model_dir")
            why = cellline_source_problem(res, n_folds)
            retrained = cellline_reference_problem(
                res, c, ctx.arm_hint, ctx.rel(c["dir"]) if c else None)
            if why is None and res is None:
                why = "no zero_cell_emb.py results.json to cross-check"
            elif why is None and c and src and recorded_dir_differs(
                    ctx, src, c["dir"]):
                why = (f"zero_cell_emb.py ablated {src}, not the latest_combo "
                       f"combo {ctx.rel(c['dir'])}")
            elif why is None and retrained:
                why = retrained
            elif why is None and sc == "MISMATCH":
                why = (f"zero_cell_emb.py's as-trained row differs from the "
                       f"included row by {sc_diff:.4f} (>= 5e-3)")
            elif why is None and sc != "match":
                why = ("no included row to cross-check"
                       if inc["status"] != "ok" else
                       "no as-trained row in zero_cell_emb.py's results.json")
            # ...and the GRADED row's own provenance: zero_cell_emb.py stamps
            # the combo it ablated into model_params.json (source_combo) and
            # build_summary carries it into the row, so a row graded from
            # another ablation (a hand grade.sh under a pipeline MODEL_NAME)
            # is caught even when this results.json is clean. A row from
            # before source_combo was recorded carries none and is not judged.
            rsrc = (ctx.json_rows.get((cv, f"{row}_nocell")) or {}).get(
                "source_combo")
            if why is None and c and rsrc and recorded_dir_differs(
                    ctx, rsrc, c["dir"]):
                why = (f"{row}_nocell_{cv}.json was graded from an ablation "
                       f"of {rsrc}, not the latest_combo combo "
                       f"{ctx.rel(c['dir'])}")
            if why and (res is not None or rem["status"] == "ok"):
                ctx.miss(f"tab_cellline {row}/{cv}: {why}; zero_cell_emb.py's "
                         f"rows not tabulated")
            if why and rem["status"] == "ok":
                rem["status"] = f"stale: {why}"
            # What still tells the heads apart once cell_emb is zero (per-head
            # c_h under train.py's defaults): the removed row does not
            # withdraw the whole cell-line representation.
            if res is not None and "residual_head_terms" in res:
                rem["residual_head_terms"] = "; ".join(
                    res["residual_head_terms"] or []) or "none"
            group = [inc, rem]
            pooled = f"{kernel}:pooled"
            if pooled in ctx.variants:
                group.append(cell(f"retrained without cell lines ({pooled})",
                                  variant_row(pooled)))
            for r in group:
                for m in METRICS:
                    a, b = r.get(f"{m}_mean"), inc.get(f"{m}_mean")
                    r[f"delta_{m}_vs_included"] = (
                        a - b if a is not None and b is not None and
                        r["status"] == "ok" else None)
            rows.extend(group + ([] if why else
                                 cellline_self_rows(res, base, row)))
    # The table stays "missing" when no removed row was graded (tab:cellline /
    # CL1 are then not produced, whatever the reason), but a run whose
    # variants all lack a cell-line embedding says why: nothing was skipped.
    if not any(r["condition"] == "removed (cell_emb=0)" for r in rows):
        ctx.note("tab_cellline: no variant of this run has a cell-line "
                 "embedding to remove (pooled variants, or combos trained "
                 "without --use_cell_lines), so tab:cellline / CL1 cannot be "
                 "produced by it")
    have = any(r["status"] == "ok" and r["condition"].startswith("removed")
               for r in rows)
    return rows, "ok" if have else "missing"


CELLLINE_COLS = (["variant", "cv", "tier", "condition", "model"] +
                 [f"{m}_{s}" for m in METRICS for s in ("mean", "sd")] +
                 [f"delta_{m}_vs_included" for m in METRICS] +
                 ["residual_head_terms", "selfcheck_as_trained",
                  "selfcheck_max_abs_diff", "status"])


# =============================================================================
# tab_external / tab_external_ft
# =============================================================================


def _read_frame(path: Path):
    import pandas as pd
    if not path.exists():
        return None
    try:
        # A copy cut short (an interrupted download of the results tree) ends
        # inside a line, and pandas would take that line as a row, a number
        # cut inside its digits included; external_transfer.py ends every row
        # with a line end.
        if not path.read_bytes().endswith(b"\n"):
            return None
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001  (an empty file has no columns)
        return None
    return df if len(df) else None


def scored_head(ctx: Ctx, v: str, screen: str, split: str) -> Optional[str]:
    """The head that scored a SLeuth screen (eval_finetuning config_used.json
    effective_head: the screen's cell-line head for per_head, 'ANY (pooled)'
    for pooled); missing is listed."""
    cfg = ctx.eval_config(v, split)
    head = (cfg.get("effective_head") or {}).get(screen)
    # eval_finetuning.effective_head is None on the single-output path (no
    # heads), so a PER_HEAD=0 eval records nulls by design: nothing is absent.
    if head is None and cfg.get("per_head") is False:
        return "n/a (single output)"
    if head is None:
        ctx.miss(f"scored head of {variant_row(v)} {screen}/{split} (no "
                 f"effective_head in its cv3 combo's eval_finetuning_{split}/"
                 f"config_used.json)")
    return head


def check_grid(ctx: Ctx, table: str, rows: List[dict], keys: Sequence[str],
               expected: Dict[str, List[tuple]], p: Path,
               dropped: Collection[str] = ()) -> None:
    """List every expected cell (model -> [key values]) absent from rows; a
    model absent altogether is one entry, and one in dropped (by
    drop_unusable_peers: in p, but refused, the reason listed above) is
    named as dropped rather than as not in p."""
    have = {tuple(r[k] for k in ("model", ) + tuple(keys)) for r in rows}
    for model, cells in expected.items():
        if not any(r["model"] == model for r in rows):
            ctx.miss(f"{table} {model}: dropped (see above)"
                     if model in dropped else f"{table} {model} not in {p}")
            continue
        for c in cells:
            if (model, ) + tuple(c) not in have:
                ctx.miss(f"{table} {model} {'/'.join(c)} not in {p}")


def check_folds(ctx: Ctx, table: str, r: dict, keys: Sequence[str]) -> None:
    """List a cell whose mean is over fewer finite folds than it scored, or
    over fewer folds than the fold set holds."""
    n, nf = _num(r.get("n_folds")), _num(r.get("n_folds_finite"))
    k = int(ctx.meta.get("num_folds") or 0)
    cell = f"{table} {r['model']} " + "/".join(str(r[x]) for x in keys)
    if n is not None and nf is not None and nf < n:
        ctx.miss(f"{cell}: {int(nf)} of {int(n)} folds finite")
    if k and n is not None and n != k:
        ctx.miss(f"{cell}: {int(n)} folds scored, the fold set has {k}")


def external_model_cv(ctx: Ctx) -> str:
    """The CV whose trained models external_transfer.py scored the screens
    with (its --model_cv; the baselines' fold_<k>_<cv>_predictions.npy). Read
    from the run's own summary.md, so a --model_cv other than cv3 gates the
    right row."""
    md = ctx.ext_dir / "summary.md"
    if md.exists():
        m = re.search(r"model CV \(baselines / DegreePrior folds\)[^\n]*?"
                      r"(cv[123])", md.read_text())
        if m:
            return m.group(1)
    return "cv3"


def drop_unusable_peers(ctx: Ctx, rows: List[dict],
                        table: str) -> Tuple[List[dict], set]:
    """A peer whose tab:tiers row at the model CV is not tabulated must not
    be published (bolded, ranked) here either: tab:tiers gates it through
    Ctx.metric (row_usable: a substitute input, a REDUCED SLMGAE, a partial
    fold set; or no row, a foreign fold stamp, no evaluate_model JSON beside
    its comparison row, two copies that disagree), while these tables read
    external_transfer.py's CSVs, which carry no such record, so the same gate
    is applied by model name. Dropped before the completeness check, so the
    cell is reported missing, with Ctx.metric's / tab_tiers' reason above.
    Returns the rows kept and the models dropped (check_grid names those as
    dropped, not as absent from the CSV)."""
    cv = external_model_cv(ctx)
    keep, dropped = [], set()
    for r in rows:
        if r["model"] in PEERS and \
                ctx.metric(cv, r["model"], "auprg")[0] is None:
            dropped.add(r["model"])
            continue
        keep.append(r)
    for m in sorted(dropped):
        ctx.miss(f"{table}: {m} rows dropped (its {cv} row is not tabulated; "
                 f"see the reason above)")
    return keep, dropped


# external_transfer.py's value and the eval's own, when both come from ONE
# eval run, are the same per-fold Spearman values averaged twice (equal to
# 2e-16 on a synthetic run). Not 1e-9: eval_finetuning.py scores the in-memory
# target, external_transfer.py the target read back from
# <Screen>_predictions.csv, and pandas' default read_csv does not round-trip a
# float64 (0.30000000000000004 comes back as 0.3), so two targets one ULP
# apart can become a tie there (none are that close in the three screens).
# 1e-6 is far below the three decimals the paper prints.
EXTERNAL_EVAL_TOL = 1e-6


def eval_own_spearman(ctx: Ctx, v: str, split: str,
                      cv: str = "cv3") -> Dict[tuple, float]:
    """{(screen, method): mean Spearman} that v's <cv> combo's
    eval_finetuning_<split>/ itself reports on the SynLethDB-filtered test
    rows, per-fold mean: zero_shot_summary.csv (calibration raw ->
    zero_shot_raw) and finetuned_summary.csv (mode LP_last / Full). These are
    the same per-fold values external_transfer.py averages into its
    all_sldb_filtered rows. {} when the eval dir or the columns are absent."""
    c = ctx.combos.get((v, cv))
    if not c:
        return {}
    ed, out = c["dir"] / f"eval_finetuning_{split}", {}
    for name, col, rename in (("zero_shot_summary.csv", "calibration",
                               {"raw": "zero_shot_raw"}),
                              ("finetuned_summary.csv", "mode", None)):
        df = _read_frame(ed / name)
        if df is None or not {"dataset", col, "filter", "aggregation",
                              "spearman_mean"} <= set(df.columns):
            continue
        for rec in df.to_dict("records"):
            if rec["filter"] != "sldb_filtered" or \
                    rec["aggregation"] != "per_fold":
                continue
            meth = rename.get(rec[col]) if rename else rec[col]
            if meth and _num(rec["spearman_mean"]) is not None:
                out[(rec["dataset"], meth)] = float(rec["spearman_mean"])
    return out


def check_external_evals(ctx: Ctx, table: str, rows: List[dict],
                         p: Path) -> None:
    """List a SLeuth row of an external_transfer.py CSV that was not computed
    from the eval dirs now under the combo it scored (cv3; its --model_cv).
    Those CSVs record the combo by directory path only, while the combo job
    replaces a retrained combo under the same name and eval_finetuning
    rewrites its outputs in place: after either, scored_head and
    items.screens read the NEW eval while these values are the previous
    one's, with nothing listed. Compared on the rows both sides score
    identically (all_sldb_filtered; zero_shot_raw, LP_last, Full;
    zero_shot_ols is refit there and differs by up to 1e-6), within
    EXTERNAL_EVAL_TOL; a cell with no counterpart (an eval without
    sldb_filtered rows, a row of a variant outside this run, a peer) is not
    judged. One entry per model."""
    cv = external_model_cv(ctx)
    cache: Dict[tuple, dict] = {}
    bad: Dict[str, list] = {}
    for r in rows:
        v = r.get("variant")
        if not v or r["row_set"] != "all_sldb_filtered" or \
                r["spearman_mean"] is None:
            continue
        key = (v, r["split"])
        if key not in cache:
            cache[key] = eval_own_spearman(ctx, v, r["split"], cv)
        meth = r.get("method", "zero_shot_raw")
        own = cache[key].get((r["screen"], meth))
        if own is not None and \
                abs(own - r["spearman_mean"]) > EXTERNAL_EVAL_TOL:
            bad.setdefault(r["model"], []).append(
                (r["screen"], r["split"], meth, r["spearman_mean"], own))
    for model, cells in bad.items():
        s, sp, meth, x, own = cells[0]
        ctx.miss(f"{table} {model}: {len(cells)} value(s) in {p} are not "
                 f"what its {cv} combo's eval_finetuning_<split>/ reports now "
                 f"(e.g. {s}/{sp} {meth}: {x:.6f} there, {own:.6f} in the "
                 f"eval): the tables were built from an earlier eval of this "
                 f"row (the combo or the eval was rerun since); rerun "
                 f"sl_comparison/slurm/run_external_transfer.sh"
                 f"{ctx.arm_hint}")


def external_arm_problem(ctx: Ctx, p: Path, df) -> Optional[str]:
    """Why an external_transfer.py CSV is not this arm's (None = it is, or it
    does not say). external_transfer.py --pca_arm writes the SLeuth combos'
    arm in a trailing pca_arm column (2026-10-21; a table from before has
    none and is read as before), so one naming another arm is that arm's
    tables copied into this arm's namespace (or a hand --external_dir): its
    SLeuth rows are another arm's models under this arm's names, and the
    table is not read, as a comparison.csv of another arm is not."""
    if "pca_arm" not in df.columns:
        return None
    other = sorted({str(a) for a in df["pca_arm"].dropna()} - {ctx.pca_arm})
    if not other:
        return None
    return (f"{p} names PCA arm(s) {', '.join(other)} in its pca_arm column, "
            f"not {ctx.pca_arm}: another arm's external-transfer table, not "
            f"read (rerun sl_comparison/slurm/run_external_transfer.sh"
            f"{ctx.arm_hint})")


def build_external(ctx: Ctx) -> Tuple[List[dict], str]:
    p = ctx.ext_dir / "zero_shot_summary.csv"
    df = _read_frame(p)
    if df is None:
        ctx.miss(f"{p} (external_transfer.py)")
        return [], "missing"
    bad = external_arm_problem(ctx, p, df)
    if bad:
        ctx.miss(bad)
        return [], "missing"
    sleuth = {variant_row(v): v for v in ctx.variants}
    rows = []
    for rec in df.to_dict("records"):
        model = str(rec["model"])
        is_sleuth = model in sleuth or rec.get("kind") == "siamese"
        rs = rec["row_set"]
        if is_sleuth and rs not in ("all_sldb_filtered", "matched") + \
                EXTERNAL_SLDB_ROW_SETS:
            continue
        if not is_sleuth and rs not in ("matched", ) + EXTERNAL_SLDB_ROW_SETS:
            continue
        rows.append({
            "screen": rec["screen"], "split": rec["split"],
            "split_label": SPLIT_LABELS.get(rec["split"], rec["split"]),
            "screen_role": rec.get("screen_role")
            if isinstance(rec.get("screen_role"), str) else None,
            "model": model, "variant": sleuth.get(model, ""),
            "kind": rec.get("kind"), "row_set": rs,
            # The revised paper prints every model on the identical matched
            # rows (2026-11-13; the submitted one put SLeuth on
            # all_sldb_filtered, pairs with a gene outside genes.txt
            # included, next to peers on matched).
            "paper_convention": (rs == "matched"
                                 and rec.get("kind") != "degree_prior"),
            "scored_head": scored_head(ctx, sleuth[model], rec["screen"],
                                       rec["split"])
            if model in sleuth else None,
            "n_rows": rec.get("n_rows"),
            "spearman_mean": _num(rec.get("spearman_mean")),
            "spearman_sd": _num(rec.get("spearman_sd_pop")),
            "n_folds": rec.get("n_folds"),
            "n_folds_finite": rec.get("n_folds_finite"),
        })
    rows, dropped = drop_unusable_peers(ctx, rows, "tab_external")
    keys = ("screen", "split", "row_set")
    grid = {m: [(s, sp, rs) for s in SCREENS for sp in SPLITS
                for rs in ("matched", "all_sldb_filtered", "matched_incl_sldb")]
            for m in sleuth}
    grid.update({m: [(s, sp, rs) for s in SCREENS for sp in SPLITS
                     for rs in ("matched", "matched_incl_sldb")]
                 for m in PEERS})
    check_grid(ctx, "tab_external", rows, keys, grid, p, dropped)
    # screen_role: external_transfer.py's in-distribution label (a screen
    # most of whose pairs SynLethDB lists); absent from a table built before
    # 2026-11-12, which also has no matched_incl_sldb rows (listed above).
    if rows and all(r["screen_role"] is None for r in rows):
        ctx.miss(f"tab_external: no screen_role column in {p} (built before "
                 f"the SynLethDB in-distribution columns; rerun "
                 f"sl_comparison/slurm/run_external_transfer.sh"
                 f"{ctx.arm_hint})")
    check_external_evals(ctx, "tab_external", rows, p)
    for r in rows:
        check_folds(ctx, "tab_external", r, keys)
    # A siamese row the summary holds but this run does not compare (a
    # narrower --variants than external_transfer.py ran with, or one of its
    # standalone --siamese NAME=DIR rows) is tabulated, with an empty variant
    # cell, but never decides the bold of the columns the table IS asked to
    # compare.
    for r in rows:
        if r["kind"] == "siamese" and not r["variant"]:
            ctx.note(f"tab_external: {r['model']} is not among this run's "
                     f"SLeuth variants; tabulated, left out of best_matched")
    # Bold = best on the identical (matched) rows among trained models, ties
    # at the printed precision all bold; unknown, not the best of the rest,
    # while one of them is absent.
    _flag_best(rows, ["screen", "split"], "spearman_mean", "best_matched",
               eligible=lambda r: r["row_set"] == "matched" and
               r["kind"] != "degree_prior" and
               not (r["kind"] == "siamese" and not r["variant"]),
               decimals=PRINTED_DECIMALS)
    want = set(sleuth) | set(PEERS)
    for s in SCREENS:
        for sp in SPLITS:
            cell = [r for r in rows if r["screen"] == s and r["split"] == sp]
            gone = sorted(want - {r["model"] for r in cell
                                  if r["row_set"] == "matched" and
                                  r["spearman_mean"] is not None})
            if cell and gone:
                for r in cell:
                    r["best_matched"] = None
                ctx.miss(f"tab_external {s}/{sp}: best_matched left empty (no "
                         f"matched {', '.join(gone)} row)")
    return rows, "ok" if rows else "missing"


EXTERNAL_COLS = ("screen", "split", "split_label", "screen_role", "model",
                 "variant", "kind", "row_set", "paper_convention",
                 "scored_head", "n_rows", "spearman_mean", "spearman_sd",
                 "n_folds", "n_folds_finite", "best_matched")
# external_transfer.py's row sets that keep the SynLethDB pairs (every model;
# kind sldb_label = the SynLethDB label as a predictor, matched_in_sldb only).
EXTERNAL_SLDB_ROW_SETS = ("matched_incl_sldb", "matched_in_sldb")


# =============================================================================
# tab_sldb_agreement
# =============================================================================

SLDB_AGREEMENT_METRICS = (("auroc", "auroc_target_vs_label"),
                          ("spearman", "spearman_target_vs_label"))


def build_sldb_agreement(ctx: Ctx) -> Tuple[List[dict], str]:
    """Does SynLethDB agree with each screen? external_transfer.py's
    sldb_agreement.csv (per screen x scope: AUROC of the target, higher =
    more SL, for SynLethDB SL vs screened non-SL; Spearman(label, target);
    on the pairs SynLethDB labels), with the bootstrap CIs of
    <cmp>/bootstrap/ci_sldb_agreement.csv (bootstrap_ci.py --part external,
    run by hand: absent CIs are noted, not missing; a CI whose point is not
    the table's value is from another run and is listed)."""
    p = ctx.ext_dir / "sldb_agreement.csv"
    df = _read_frame(p)
    if df is None:
        ctx.miss(f"{p} (external_transfer.py)")
        return [], "missing"
    bad = external_arm_problem(ctx, p, df)
    if bad:
        ctx.miss(bad)
        return [], "missing"
    cip = ctx.cmp_results / "bootstrap" / "ci_sldb_agreement.csv"
    ci = _read_frame(cip)
    if ci is not None and "pca_arm" in ci.columns and \
            set(ci["pca_arm"].astype(str)) - {ctx.pca_arm}:
        ctx.miss(f"{cip} is another PCA arm's; CIs not read")
        ci = None
    if ci is None:
        ctx.note(f"tab_sldb_agreement: no {cip}; CI columns empty (run "
                 f"python sl_comparison/bootstrap_ci.py --part external "
                 f"--pca_arm {ctx.pca_arm})")
    rows, stale = [], 0
    for rec in df.to_dict("records"):
        r = {"screen": rec["screen"],
             "screen_role": rec.get("screen_role")
             if isinstance(rec.get("screen_role"), str) else None,
             "scope": rec["scope"], "split": str(rec["split"]),
             "n_labelled": rec.get("n_labelled"), "n_pos": rec.get("n_pos"),
             "n_neg": rec.get("n_neg")}
        for short, col in SLDB_AGREEMENT_METRICS:
            r[short] = _num(rec.get(col))
            r[f"{short}_ci_low"] = r[f"{short}_ci_high"] = None
            if ci is None:
                continue
            hit = ci[(ci["screen"] == rec["screen"]) &
                     (ci["scope"] == rec["scope"]) &
                     (ci["split"].astype(str) == str(rec["split"])) &
                     (ci["metric"] == col)]
            if len(hit) != 1:
                continue
            pt = _num(hit["point"].iloc[0])
            if (pt is None) != (r[short] is None) or (
                    pt is not None and abs(pt - r[short]) > 1e-9):
                stale += 1
                continue
            r[f"{short}_ci_low"] = _num(hit["ci_low"].iloc[0])
            r[f"{short}_ci_high"] = _num(hit["ci_high"].iloc[0])
        r["ci_source"] = str(cip.relative_to(ctx.root)) \
            if ci is not None and cip.is_relative_to(ctx.root) else \
            (str(cip) if ci is not None else None)
        rows.append(r)
    if stale:
        ctx.miss(f"tab_sldb_agreement: {stale} CI(s) in {cip} are not of the "
                 f"values in {p} (another run's bootstrap); rerun "
                 f"bootstrap_ci.py --part external --pca_arm {ctx.pca_arm}")
    have = {r["screen"] for r in rows}
    for s in SCREENS:
        if s not in have:
            ctx.miss(f"tab_sldb_agreement {s}: not in {p}")
    return rows, "ok" if rows else "missing"


SLDB_AGREEMENT_COLS = ("screen", "screen_role", "scope", "split",
                       "n_labelled", "n_pos", "n_neg", "auroc",
                       "auroc_ci_low", "auroc_ci_high", "spearman",
                       "spearman_ci_low", "spearman_ci_high", "ci_source")


def build_external_ft(ctx: Ctx) -> Tuple[List[dict], str]:
    p = ctx.ext_dir / "finetune_summary.csv"
    df = _read_frame(p)
    if df is None:
        ctx.miss(f"{p} (external_transfer.py)")
        return [], "missing"
    bad = external_arm_problem(ctx, p, df)
    if bad:
        ctx.miss(bad)
        return [], "missing"
    sleuth = {variant_row(v): v for v in ctx.variants}
    rows = []
    for rec in df.to_dict("records"):
        if rec["row_set"] not in ("all_sldb_filtered", "matched") or \
                rec["method"] not in FT_METHODS:
            continue
        model = str(rec["model"])
        rows.append({
            "model": model, "variant": sleuth.get(model, ""),
            "screen": rec["screen"], "split": rec["split"],
            "split_label": SPLIT_LABELS.get(rec["split"], rec["split"]),
            "row_set": rec["row_set"],
            "paper_convention": rec["row_set"] == "all_sldb_filtered",
            "method": rec["method"],
            "column": FT_COLUMN_NAMES.get(rec["method"],
                                          "(reference) zero-shot OLS"),
            "scored_head": scored_head(ctx, sleuth[model], rec["screen"],
                                       rec["split"])
            if model in sleuth else None,
            "n_rows": rec.get("n_rows"),
            "spearman_mean": _num(rec.get("spearman_mean")),
            "spearman_sd": _num(rec.get("spearman_sd_pop")),
            "n_folds": rec.get("n_folds"),
            "n_folds_finite": rec.get("n_folds_finite"),
        })
    rows, dropped = drop_unusable_peers(ctx, rows, "tab_external_ft")
    if any(r["method"] == FT_SCRATCH for r in rows):
        ctx.note("tab_external_ft: column Scratch = the from-scratch baseline "
                 "(eval_finetuning --ft_modes ... Scratch): Full's schedule on "
                 "the same architecture and fold feature space, freshly "
                 "initialised, its head warm-started by OLS on its own "
                 "logits; items.external F2_F3 full_minus_scratch = Full - "
                 "Scratch")
    keys = ("screen", "split", "row_set", "method")
    check_grid(ctx, "tab_external_ft", rows, keys,
               {m: [(s, sp, rs, meth) for s in SCREENS for sp in SPLITS
                    for rs in ("all_sldb_filtered", "matched")
                    for meth in ft_paper_methods(rows, m)] for m in sleuth},
               p, dropped)
    check_external_evals(ctx, "tab_external_ft", rows, p)
    for r in rows:
        check_folds(ctx, "tab_external_ft", r, keys)
    gkeys = ("model", "screen", "split", "row_set")
    _flag_best(rows, gkeys, "spearman_mean", "best_method",
               eligible=lambda r: r["method"] != "zero_shot_ols")
    # The best of the three columns (four with Scratch), not of the ones that
    # survived.
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in gkeys), []).append(r)
    short: Dict[str, list] = {}
    for key, grp in groups.items():
        have = {r["method"] for r in grp if r["spearman_mean"] is not None}
        gone = [m for m in ft_paper_methods(rows, key[0]) if m not in have]
        if gone:
            for r in grp:
                r["best_method"] = None
            n, meths = short.setdefault(key[0], [0, set()])
            short[key[0]] = [n + 1, meths | set(gone)]
    for model, (n, meths) in short.items():
        ctx.miss(f"tab_external_ft {model}: best_method left empty in {n} "
                 f"screen/split/row_set cell(s) (no {', '.join(sorted(meths))} "
                 f"value)")
    return rows, "ok" if rows else "missing"


EXTERNAL_FT_COLS = ("model", "variant", "screen", "split", "split_label",
                    "row_set", "paper_convention", "method", "column",
                    "scored_head", "n_rows", "spearman_mean", "spearman_sd",
                    "n_folds", "n_folds_finite", "best_method")


# =============================================================================
# numbers.json items
# =============================================================================


def item_data_stats(ctx: Ctx) -> dict:
    """I1 / E1: counts and densities over C(num_nodes, 2)."""
    m = ctx.meta
    n = int(m["num_nodes"])
    total = n * (n - 1) // 2
    return {"num_nodes": n, "num_positives": int(m["num_positives"]),
            "num_negatives": int(m["num_negatives"]),
            "all_pairs": total,
            "positive_pct_of_all_pairs": 100 * m["num_positives"] / total,
            "negative_pct_of_all_pairs": 100 * m["num_negatives"] / total}


def item_folds_meta(ctx: Ctx) -> dict:
    """E2 / M8: seed, folds, val fraction, split hashes, conflict counts."""
    keys = ("seed", "num_folds", "val_frac", "conflict_policy", "cv_types",
            "split_digest", "source_sha256", "num_conflicts",
            "num_conflicts_kept_positive", "num_conflicts_same_line",
            "num_conflicts_cross_line", "num_conflicts_same_head",
            "num_conflicts_missing_cell_line", "max_genes")
    return {k: ctx.meta.get(k) for k in keys}


def item_fold_provenance(ctx: Ctx) -> dict:
    """E2 / Reproducibility: the fold set every comparison row (comparison.csv,
    else its JSON) was graded and trained on, against the current folds. A
    differing row is also listed as missing and left out of every table."""
    rows = {}
    for key in sorted(set(ctx.cmp) | set(ctx.cmp_raw) | set(ctx.json_rows)):
        cv, model = key
        src = "comparison.csv" if key in ctx.cmp or key in ctx.cmp_raw \
            else "json"
        ctx.fold_ok(cv, model, src)
        rows[f"{model}/{cv}"] = ctx.fold_record(cv, model, src)
    cur = ctx.current_stamp()
    return {"current": {n: cur[n] for n in ctx.model_params.STAMP_FILES},
            "rows": rows,
            "rows_without_training_stamp": sorted(
                k for k, r in rows.items() if r["trained_on"] is None),
            "all_rows_same_folds": all(not r["problems"]
                                       for r in rows.values())
            if rows else None}


def _embedding_width(path: Path) -> Optional[int]:
    """Native width of a data/*.pt embedding (the matrix data_loader reads),
    without reading the tensor payload where torch allows."""
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=False,
                       mmap=True)
    except (RuntimeError, TypeError, ValueError):
        d = torch.load(path, map_location="cpu", weights_only=False)
    e = d.get("raw_embeddings", d.get("embeddings")) if isinstance(d, dict) \
        else None
    return int(e.shape[1]) if getattr(e, "ndim", 0) == 2 else None


def kg_entity_types(relations: Sequence[str]) -> List[str]:
    """Entity types joined by SynLethKG relation names: each is a Hetionet
    metaedge (PARTICIPATES_GpCC, BINDS_CbG, INCLUDES_PCiC) whose capitalised
    runs are the head and tail types."""
    return sorted({t for r in relations
                   for t in re.findall(r"[A-Z]+", r.split("_", 1)[-1])})


def item_inputs(ctx: Ctx) -> dict:
    """R2: the SynLethKG SLGNN ran on; D3: each comparison row's recorded
    training seed; AP1 / AP2: each catalog embedding's native width."""
    out = {}
    kg = read_json(ctx.root / "data" / "synlethkg" / "meta.json")
    if kg is None:
        ctx.miss("data/synlethkg/meta.json (R2 SynLethKG entity / relation "
                 "types)")
        out["synlethkg"] = None
    else:
        types = kg_entity_types(kg.get("relations") or [])
        out["synlethkg"] = {**{k: kg.get(k) for k in (
            "n_relations", "relations", "removed_label_relations",
            "n_entities", "n_triples")},
            "entity_types_from_relations": types,
            "n_entity_types": len(types) if types else None}
    seeds = {}
    for key in sorted(set(ctx.cmp_raw) | set(ctx.json_rows)):
        cv, model = key
        raw = (ctx.cmp_raw.get(key) or {}).get("seed")
        if raw is None:
            raw = (ctx.json_rows.get(key) or {}).get("seed")
        seeds[f"{model}/{cv}"] = int(float(raw)) if _num(raw) is not None \
            else None
    known = {s for s in seeds.values() if s is not None}
    out["training_seeds"] = seeds
    out["single_training_seed"] = len(known) == 1 if known else None
    out["rows_without_seed"] = sorted(k for k, s in seeds.items() if s is None)
    # D3 is over the trained paper rows: DegreePrior trains nothing, and a
    # <row>_nocell row is its combo ablated at inference, not a training run.
    trained = {variant_row(v) for v in ctx.variants} | set(PEERS)
    for k in out["rows_without_seed"]:
        if k.rsplit("/", 1)[0] in trained:
            ctx.miss(f"numbers.json inputs {k}: no training seed recorded "
                     f"(D3)")
    widths = {}
    try:
        import torch  # noqa: F401
    except ImportError:
        ctx.miss("torch not importable: items.inputs.native_dims not read")
        out["native_dims"] = None
        return out
    for etype, efile, _ in ctx.catalog:
        p = ctx.root / "data" / efile
        if not p.exists():
            continue
        try:
            widths[etype] = _embedding_width(p)
        except Exception as e:  # noqa: BLE001  (one unreadable file)
            widths[etype] = None
            ctx.note(f"native width of data/{efile} unreadable: "
                     f"{type(e).__name__}")
        # AP1 / AP2 quote the paper's 16 widths.
        if widths[etype] is None and etype in PAPER_SOLO:
            ctx.miss(f"native width of data/{efile} not read (AP1/AP2)")
    out["native_dims"] = widths
    return out


def item_fold_geometry(ctx: Ctx) -> dict:
    """D1 / D7 / D8 / D9 / D10 / TT1 from the fold .npy files."""
    total = int(ctx.meta["num_positives"]) + int(ctx.meta["num_negatives"])
    k = int(ctx.meta.get("num_folds") or 5)
    out = {"labelled_pairs": total,
           "D7_analytic_gene_split": {"p_both_train": (1 - 1 / k)**2,
                                      "p_both_test": (1 / k)**2,
                                      "p_mixed": 2 * (1 / k) * (1 - 1 / k)}}
    for cv in CVS:
        per = []
        for fd in sorted((ctx.folds / cv).glob("fold_*"), key=fold_index):
            a = {n: np.load(fd / f"{n}.npy") for n in (
                "train_pos", "val_pos", "test_pos", "train_neg", "val_neg",
                "test_neg")}
            n = {key: len(v) for key, v in a.items()}
            n_tr = n["train_pos"] + n["train_neg"]
            n_va = n["val_pos"] + n["val_neg"]
            n_te = n["test_pos"] + n["test_neg"]
            rec = {"fold": fold_index(fd), **{f"n_{x}": y for x, y in n.items()},
                   "test_share": n_te / total, "train_share": n_tr / total,
                   "val_share": n_va / total,
                   "discarded_share": 1 - (n_tr + n_va + n_te) / total,
                   "test_prevalence": n["test_pos"] / n_te if n_te else None}
            te = np.vstack([a["test_pos"], a["test_neg"]])
            tr = np.vstack([a["train_pos"], a["train_neg"]])
            tv = np.vstack([tr, a["val_pos"], a["val_neg"]])
            for name, ref in (("train", tr), ("train_val", tv)):
                g = np.unique(ref.ravel())
                both = np.isin(te[:, 0], g) & np.isin(te[:, 1], g)
                rec[f"test_both_genes_in_{name}"] = float(both.mean()) \
                    if len(te) else None
            per.append(rec)
        if not per:
            ctx.miss(f"{ctx.folds / cv}/fold_*")
            continue
        # The ranges below are over the fold dirs found, and nothing else here
        # reads the fold arrays (the fold stamp hashes genes.txt / all_pos.npy
        # / meta.json): a set short a fold_<k> dir (a partial copy) is listed.
        nf = int(ctx.meta.get("num_folds") or 0)
        have = [r["fold"] for r in per]
        if nf and have != list(range(nf)):
            ctx.miss(f"numbers.json fold_geometry {cv}: fold dirs {have} "
                     f"under {ctx.folds / cv}, meta.json num_folds is {nf}; "
                     f"its ranges are over those only")

        def rng(key):
            v = [r[key] for r in per if r[key] is not None]
            return {"mean": float(np.mean(v)), "min": float(min(v)),
                    "max": float(max(v))} if v else None

        for r in per:
            r["n_test"] = r["n_test_pos"] + r["n_test_neg"]
        out[cv] = {"per_fold": per,
                   **{key: rng(key) for key in (
                       "test_share", "discarded_share", "test_prevalence",
                       "test_both_genes_in_train",
                       "test_both_genes_in_train_val", "n_test",
                       "n_test_pos")}}
    return out


def item_pca_width(ctx: Ctx) -> dict:
    """R5 / M12: the encoder input width after the arm's PCA per split
    (fold_metrics input_dim: the concatenated width under arm none, the
    width every PCA of the arm retained otherwise), per variant, and whether
    every variant saw the same widths.
    A variant's min / max are None (listed) unless every split of every tier
    has a width: the paper's range is over all fifteen; the comparison across
    variants is None while a variant is incomplete and the rest agree."""
    k = int(ctx.meta.get("num_folds") or 0)
    out, grid = {}, {}
    for v in ctx.variants:
        vals = {}
        for cv in CVS:
            c = ctx.combos.get((v, cv))
            if not c:
                continue
            fm = c["results"].get("fold_metrics") or []
            dims = [m.get("input_dim") for m in fm]
            if not fm or any(not isinstance(x, (int, float)) for x in dims):
                ctx.miss(f"{c['dir']}/results.json fold_metrics[].input_dim")
                continue
            vals[cv] = [int(x) for x in dims]
        if vals:
            flat = [x for cv in vals for x in vals[cv]]
            full = set(vals) == set(CVS) and all(
                not k or len(x) == k for x in vals.values())
            if full:
                grid[v] = vals
            else:
                ctx.miss(f"numbers.json pca_width {v}: widths of {len(flat)} "
                         f"split(s) only; min / max left empty")
            out[v] = {"per_cv": vals, "min": min(flat) if full else None,
                      "max": max(flat) if full else None,
                      "n_splits": len(flat)}
    ref = next(iter(grid.values()), None)
    diffs = [v for v, g in grid.items() if g != ref]
    return {"per_variant": out, "equal_across_variants":
            False if diffs else True if grid and len(grid) ==
            len(ctx.variants) else None,
            "variants_differing_from_first": diffs,
            "variants_incomplete": [v for v in ctx.variants
                                    if v not in grid]}


def item_param_ratio(ctx: Ctx, tiers: List[dict]) -> dict:
    """I2: SLMGAE trainable parameters / SLeuth per-fold mean. min_ratio
    (the paper's '>8,600 times') is None, listed, unless every tier has a
    ratio: the minimum over the tiers that survived can overstate it."""
    out = {}
    for v in ctx.variants:
        per = {}
        for cv in CVS:
            sl = next((r for r in tiers if r["cv"] == cv and
                       r["model"] == "SLMGAE"), {}).get("trainable_params")
            me = next((r for r in tiers if r["cv"] == cv and
                       r["variant"] == v and r["kind"] == "SLeuth"),
                      {}).get("params_fold_mean")
            if sl and me:
                per[cv] = {"slmgae": sl, "sleuth_fold_mean": me,
                           "ratio": sl / me}
        if per:
            full = len(per) == len(CVS)
            if not full:
                ctx.miss(f"numbers.json param_ratio {v}: no ratio at "
                         f"{', '.join(cv for cv in CVS if cv not in per)}; "
                         f"min_ratio left empty")
            out[v] = {"per_cv": per,
                      "min_ratio": min(p["ratio"] for p in per.values())
                      if full else None}
    return out


def item_heads(ctx: Ctx) -> dict:
    """M7 / M9: head vocabulary, cell-line embedding width, and the most
    heads any pair is observed in: cell_line_vocab.pair_label_mask over every
    pair of the pair files, as data_loader._setup_multilabel builds its masks
    (which additionally drops pairs with a gene lacking an embedding)."""
    if ctx.vocab is None:
        raise RuntimeError("siamese_sl/cell_line_vocab.py not importable")
    pos_p = ctx.root / ctx.meta.get("pos_path",
                                    "data/SL_SynLethDB_experimental.txt")
    neg_p = ctx.root / ctx.meta.get(
        "neg_path", "data/SL_SynLethDB_experimental_negatives.txt")

    def sidecar(p):
        out = {}
        with open(p.with_suffix(".sources.tsv")) as f:
            next(f, None)
            for line in f:
                t = line.rstrip("\r\n").split("\t")
                if len(t) >= 4 and t[0] != t[1]:
                    out[canon(t[0], t[1])] = t[3]  # last write wins
        return out

    pos, neg = load_pair_set(pos_p), load_pair_set(neg_p)
    pf, nf = sidecar(pos_p), sidecar(neg_p)
    counts: Dict[int, int] = {}
    for key in pos | neg:
        _, mask = ctx.vocab.pair_label_mask(pf.get(key, ""), nf.get(key, ""),
                                            key in pos, key in neg)
        h = int(mask.sum())
        if h:
            counts[h] = counts.get(h, 0) + 1
    cfgs = {v: (ctx.combo_config(v, "cv1") or {})
            for v in ctx.variants if v.endswith(":per_head")}
    # A combo trained without --use_cell_lines is the single-output SiameseSL
    # (as build_cellline reads it): it has no cell-line embedding, and
    # config.json's cell_line_dim is then train.py's unused argparse default
    # (it dumps vars(args)), not a width this run used.
    dims = {v: c.get("cell_line_dim") for v, c in cfgs.items()
            if c.get("use_cell_lines") is not False}
    return {"heads": list(ctx.vocab.HEADS), "num_heads": len(ctx.vocab.HEADS),
            "cell_line_dim_per_variant": dims,
            "max_heads_per_pair": max(counts) if counts else None,
            "pairs_by_n_heads": {str(k): counts[k] for k in sorted(counts)}}


def item_arch(ctx: Ctx) -> dict:
    """M3-M6 / M10 / D3: each variant's recorded architecture and training
    settings (arch_signature of its combo config) + ARCH_ARGS."""
    out = {"ARCH_ARGS_env": ctx.arch_args_env}
    sel = ctx.selection or {}
    for v in ctx.variants:
        cfgs = {cv: ctx.combos[(v, cv)]["config"] for cv in CVS
                if ctx.combos.get((v, cv))}
        if not cfgs:
            continue
        first = next(iter(cfgs.values()))
        sig = ctx.arch.arch_signature(first) if ctx.arch else {}
        same = (all(ctx.arch.arch_signature(c) == sig for c in cfgs.values())
                if ctx.arch else None)
        # The combo job refuses to train without every winner's file
        # (PYEXTRACT), so a mismatch here means a legacy or foreign combo.
        per_cv, match = {}, {}
        for cv, c in cfgs.items():
            used = sorted(Path(p).name for p in c.get("embeddings_paths") or [])
            per_cv[cv] = used
            if per_fold_selection(ctx):
                # the union of the CV's per-fold winners, each fold on its own
                try:
                    why = ctx.fsel.run_matches_spec(
                        c, ctx.fsel.combo_spec(ctx.selection, cv),
                        ctx.combos[(v, cv)]["dir"])
                except (ctx.fsel.SelectionError, KeyError, TypeError,
                        ValueError) as e:
                    why = f"{type(e).__name__}: {e}"
                match[cv] = why is None
                if why:
                    ctx.miss(f"{v}/{cv} combo is not the per_fold selection's "
                             f"({why})")
                continue
            chosen = sel.get(cv) if isinstance(sel.get(cv), dict) else {}
            want = sorted(Path(i["file"]).name for i in chosen.values()
                          if isinstance(i, dict) and i.get("file"))
            match[cv] = used == want if want else None
            if match[cv] is False:
                ctx.miss(f"{v}/{cv} combo trained on "
                         f"{', '.join(used) or 'nothing'}, not the selection's "
                         f"winners {', '.join(want)}")
        # The PCA the combo applied (its arm and both stages), next to the
        # raw post-concat knobs it recorded.
        pset = ctx.arch.pca_settings(first) if ctx.arch else None
        out[v] = {"arch_signature": sig, "same_across_cvs": same,
                  "pca_arm": ctx.arch.pca_arm(first) if ctx.arch else None,
                  "pca_settings": {k: pca_setting_text(s)
                                   for k, s in pset.items()}
                  if pset is not None else None,
                  "post_pca_variance": first.get("post_pca_variance"),
                  "post_pca": first.get("post_pca"),
                  "embeddings": [Path(p).name for p in
                                 first.get("embeddings_paths") or []],
                  "embeddings_per_cv": per_cv,
                  "embeddings_match_winners": match}
    return out


def catalog_not_in_paper(ctx: Ctx) -> List[str]:
    """EMB_CATALOG entries beyond tab:solo's 16 (independent of any run
    output, so the README can name them without a selection)."""
    return sorted(e for e, _, _ in ctx.catalog if e not in PAPER_SOLO)


def selection_problems(ctx: Ctx) -> List[str]:
    """Categories whose recorded winner is not the one the single-modality
    runs NOW on disk select ([] = none, or nothing to recompute from).

    Only the combo job's Step 2 writes best_per_category.json. A SELECTION_CV
    single-modality task run after it under the same RUN_TOKEN is fresh by
    every check of Ctx.freshness, so tab:solo tabulates it, while the
    selection, every combo and all that is built on them keep the earlier
    ranking: a task that failed, was left unranked under
    BEST_CAT_ALLOW_MISSING=1 (_selection_meta.unranked_failed) and is then
    rerun by hand to fill its tab:solo cell, or any rerun whose validation
    score changed. Recomputed as Step 2 ranks: per category the first
    maximum, in catalog order, of summary[_selection_meta.selection_metric]
    over the fresh SELECTION_VARIANT dirs of _selection_meta.selection_cv
    (the unprefixed pcavar<tag>_<type>_<cv> dirs of the arm's
    single-embedding root, ctx.step1_results). A category is judged only
    when its recorded winner is among them (an absent or stale dir is listed
    by build_solo), so leaving stale candidates out cannot produce an entry:
    the recorded winner was the first maximum of a superset."""
    meta = (ctx.selection or {}).get("_selection_meta") or {}
    if per_fold_selection(ctx):
        return per_fold_selection_problems(ctx, meta.get("selection_metric"))
    cv, key = meta.get("selection_cv"), meta.get("selection_metric")
    sel = (ctx.selection or {}).get(cv) if isinstance(cv, str) else None
    if not key or not isinstance(sel, dict):
        return []
    v = ctx.selection_variant
    ref = ctx.combo_config(v, cv)
    scores: Dict[str, float] = {}
    best: Dict[str, str] = {}
    for etype, _efile, cat in ctx.catalog:
        d = solo_dir(ctx, v, etype, cv)
        res = read_json(d / "results.json")
        if not isinstance(res, dict) or ctx.freshness(
                d, v, cv, ref, ctx.solo_run_token, ignore=SOLO_SIG_IGNORE):
            continue
        s = _num((res.get("summary") or {}).get(key))
        if s is None:
            continue
        scores[etype] = s
        if cat not in best or s > scores[best[cat]]:
            best[cat] = etype
    out = []
    for cat, info in sel.items():
        w = info.get("type") if isinstance(info, dict) else None
        b = best.get(cat)
        if isinstance(w, str) and w in scores and b is not None and b != w:
            out.append(f"{cat}: best_per_category.json selected {w}, whose "
                       f"run on disk has {cv} {key} {scores[w]:.6f}, but "
                       f"{b} has {scores[b]:.6f}")
    return out


def per_fold_selection_problems(ctx: Ctx, key: Optional[str]) -> List[str]:
    """selection_problems of a per_fold selection: per (CV, fold), each
    category's recorded pick against the first maximum, in catalog order, of
    fold_metrics[fold][key] over the fresh SELECTION_VARIANT dirs of that CV
    (judged only where the recorded pick is among them)."""
    if not key:
        return []
    v = ctx.selection_variant
    out = []
    for cv in CVS:
        block = (ctx.selection or {}).get(cv)
        if not isinstance(block, dict):
            continue
        ref = ctx.combo_config(v, cv)
        fs: Dict[str, Dict[int, Optional[float]]] = {}
        for etype, _efile, _cat in ctx.catalog:
            d = solo_dir(ctx, v, etype, cv)
            res = read_json(d / "results.json")
            if not isinstance(res, dict) or ctx.freshness(
                    d, v, cv, ref, ctx.solo_run_token,
                    ignore=SOLO_SIG_IGNORE):
                continue
            fs[etype] = {k: _num(m.get(key)) for k, m in
                         ctx.fsel.fold_entries(res, 10 ** 6).items()}
        for cat in sorted(block):
            for k, w in sorted(per_fold_picks(ctx, cv, cat).items()):
                if fs.get(w, {}).get(k) is None:
                    continue
                best = None
                for etype, _efile, c in ctx.catalog:
                    s = fs.get(etype, {}).get(k)
                    if c == cat and s is not None and (
                            best is None or s > fs[best][k]):
                        best = etype
                if best is not None and best != w:
                    out.append(f"{cat} {cv} fold {k}: best_per_category.json "
                               f"picked {w}, whose run on disk has {key} "
                               f"{fs[w][k]:.6f}, but {best} has "
                               f"{fs[best][k]:.6f}")
    return out


def item_selection(ctx: Ctx) -> dict:
    """E3 / M11: the winners, how they were chosen, vs the paper's six."""
    winners = selection_winners(ctx)
    if not winners:
        raise RuntimeError("best_per_category.json missing or empty")
    # The command, with the token: a combo job submitted without it takes a
    # fresh one from config.conf, so its Step 2 finds every single-modality
    # run "another run's" and stops, the selection and the combo markers
    # already removed and the combos moved aside. The token is the one the
    # runs selection_problems ranks carry (freshness checks it there), which
    # a tables job run by hand has no $RUN_TOKEN for.
    tok = ctx.solo_run_token or ("<the token in the single-modality runs' "
                                 "run_token.txt>")
    # ...and with the arm, arm none's too (2026-10-21): a combo job submitted
    # without PCA_ARM trains the arm the shell it is pasted into exports
    # (sbatch passes the caller's environment, --export=ALL), arm none's only
    # where none or nothing is. After another arm's "After a combo job
    # submitted by hand" recipe (sl_comparison/README.md: export
    # PCA_ARM=<arm>), arm none's command removed and retrained that arm and
    # left arm none's selection as it was.
    arm = f"PCA_ARM={ctx.pca_arm} "
    # ...and with the arms (2026-10-21): config.conf's pca_step1_arm over
    # PCA_ARMS places the single-embedding runs Step 2 ranks (STEP1_ROOT), and
    # a login shell without PCA_ARMS means all four, where joint ranks none's
    # runs and both per_modality's. So in a subset run in which joint or both
    # owned its runs (PCA_ARMS=joint, "joint both", ...), the job, pasted
    # without them, removed the arm's selection and markers and then ranked
    # the partner arm's root. The run's arms (ctx.pca_arms), adjusted only
    # where they would not place the runs in the root this selection RECORDS
    # (ctx.step1_results), so that its owner is the first arm of the pair they
    # list. Not the arm alone whenever it owns them: a tables run under other
    # arms than the selection's (PCA_ARMS=joint --skip-siamese after a
    # four-arm run, its selection recording results/) would then name
    # results/pca_joint. Printed for arm none too, against a PCA_ARMS exported
    # in the shell it is pasted into.
    owner = next((a for a in pca_pair(ctx.pca_arm)
                  if ctx.step1_results ==
                  ctx.root / "siamese_sl" / pca_arm_results(a)), None)
    arms = list(ctx.pca_arms)
    if owner is not None and pca_step1_arm(ctx.pca_arm, arms) != owner:
        first = pca_pair(ctx.pca_arm)[0]
        arms = (arms + [first] if owner == first else
                [a for a in arms if a != first] + [owner])
    arm += f"PCA_ARMS=\"{' '.join(a for a in PCA_ARM_NAMES if a in arms)}\" "
    # ...and with the CV this selection was made on (2026-10-21): SELECTION_CV
    # is in no signature, so a job pasted without the run's selects on
    # run_best_per_category.conf's default and exits 0, and these entries,
    # recomputed on the CV the new selection records, then agree with it
    # (only the table across the arms tells: build_pca_arms).
    sel_cv = selection_cv(ctx)
    if sel_cv:
        arm += f"SELECTION_CV={shlex.quote(sel_cv)} "
    # Step 2's expected signature includes ARCH_ARGS and the PCA targets: a
    # resubmission without the run's stops after the same removals. ARCH_ARGS
    # verbatim when this job knows it (quoted for bash and zsh alike).
    rest = "the run's other PCA_* targets, ARCH_ARGS and SIAMESE_VARIANTS_LIST"
    if ctx.arch_args_env:
        arm += f"ARCH_ARGS={shlex.quote(ctx.arch_args_env)} "
        rest = "the run's other PCA_* targets and SIAMESE_VARIANTS_LIST"
    for why in selection_problems(ctx):
        ctx.miss(f"selection is not the one the single-modality runs on disk "
                 f"give ({why}): a selection-CV single-modality run was "
                 f"trained after the combo job's Step 2 ranked them; rerun "
                 f"siamese_sl/slurm/run_best_per_category_combo.sh under the "
                 f"run's token (from siamese_sl/: RUN_TOKEN={tok} {arm}sbatch "
                 f"--export=ALL slurm/run_best_per_category_combo.sh, with "
                 f"{rest} exported; without the token the job ranks nothing "
                 f"and stops) and every job after it")
    if ctx.fold_refused_combos:
        # The combos a selection built come from its own run, so theirs is the
        # fold set the winners were ranked on.
        raise RuntimeError(
            "the combos built from this selection are not on the current "
            "folds (" + ", ".join(f"{v} {cv}" for v, cv in sorted(
                ctx.fold_refused_combos)) + "), nor were the runs its winners "
            "were ranked on")
    meta = (ctx.selection or {}).get("_selection_meta") or {}
    # A category left without a winner (BEST_CAT_ALLOW_MISSING=1) is absent
    # from every combo.
    lacking = sorted(set(meta.get("missing_categories") or []) |
                     ({c for _, _, c in ctx.catalog} - set(winners)))
    for c in lacking:
        ctx.miss(f"selection has no {c} winner: every combo lacks {c}")
    diff = {c: {"run": w, "paper": PAPER_WINNERS.get(c)}
            for c, w in winners.items() if PAPER_WINNERS.get(c) != w}
    diff.update({c: {"run": None, "paper": PAPER_WINNERS.get(c)}
                 for c in lacking})
    out = {"winners": winners,
           "differs_from_paper": diff,
           "missing_categories": lacking,
           "selection_meta": (ctx.selection or {}).get("_selection_meta"),
           "catalog_size": len(ctx.catalog), "paper_catalog_size":
           len(PAPER_SOLO),
           "catalog_not_in_paper": catalog_not_in_paper(ctx)}
    if per_fold_selection(ctx):
        # per tier: each fold's pick per category, and how often each
        # embedding was picked (winners above: every embedding a fold picked)
        out.update({
            "selection_mode": "per_fold",
            "winners_per_cv": {cv: selection_winners(ctx, cv) for cv in CVS
                               if isinstance((ctx.selection or {}).get(cv),
                                             dict)},
            "per_fold": {cv: {cat: {str(k): e for k, e in sorted(
                per_fold_picks(ctx, cv, cat).items())}
                for cat in sorted(ctx.selection[cv])}
                for cv in CVS if isinstance((ctx.selection or {}).get(cv),
                                            dict)},
            "pick_counts": {cv: ctx.fsel.pick_counts(ctx.selection, cv)
                            for cv in CVS},
        })
    return out


def item_loco(ctx: Ctx, rows: List[dict]) -> dict:
    """AB5: per variant and tier, the change from removing each category
    (difference of means, and fold-by-fold paired mean / sd / n).
    largest_drop is None (listed) unless every ABLATE_CATEGORIES entry has a
    delta: the largest of the ablations that survived is not the paper's."""
    out = {}
    for r in rows:
        if r["dropped"] == "none" or r.get("delta_vs_reference") is None:
            continue
        rec = out.setdefault(r["variant"], {}).setdefault(r["cv"], {
            "deltas": {}, "paired": {}})
        rec["deltas"][r["dropped"]] = r["delta_vs_reference"]
        if r.get("n_paired"):
            rec["paired"][r["dropped"]] = {
                "mean": r["delta_paired_mean"], "sd": r["delta_paired_sd"],
                "n": r["n_paired"]}
    for v, per in out.items():
        for cv, rec in per.items():
            d = rec["deltas"]
            gone = [c for c in ctx.ablate_categories if c not in d]
            if gone:
                ctx.miss(f"numbers.json loco {v}/{cv}: no {', '.join(gone)} "
                         f"ablation delta; largest_drop left empty")
            rec["categories_missing"] = gone
            rec["largest_drop"] = None if gone else min(d, key=d.get)
            rec["no_drop"] = sorted(k for k, x in d.items() if x >= 0)
    return out


def item_cellline(rows: List[dict]) -> dict:
    """CL1: removed - included per per_head variant and tier."""
    out = {}
    for r in rows:
        if r["condition"].startswith("removed") and r["status"] == "ok":
            out.setdefault(r["variant"], {})[r["cv"]] = {
                m: r.get(f"delta_{m}_vs_included") for m in METRICS}
    return out


def _screen_frames(ctx: Ctx):
    """[(variant, combo dir, {screen: (split dir, DataFrame)})] from each
    variant's cv3 combo (all rows of <Screen>_predictions.csv).

    These columns do not depend on the screen split, so EACH screen is taken
    from the first eval_finetuning_<split>/ that holds it: the splits are
    separate array tasks, and one that died (or SKIPped a screen) after
    writing some of them must not hide the screens the others did write.
    A screen no split holds is listed by item_screens."""
    import pandas as pd
    order = [ctx.selection_variant] + [v for v in ctx.variants
                                       if v != ctx.selection_variant]
    out = []
    for v in order:
        c = ctx.combos.get((v, "cv3"))
        if not c:
            continue
        frames = {}
        for split in SPLITS:
            ed = c["dir"] / f"eval_finetuning_{split}"
            for s in SCREENS:
                p = ed / f"{s}_predictions.csv"
                if s not in frames and p.exists():
                    frames[s] = (ed, pd.read_csv(
                        p, dtype={"gene1_entrez": str, "gene2_entrez": str}))
            if len(frames) == len(SCREENS):
                break
        out.append((v, c["dir"], frames))
    return out


def item_screens(ctx: Ctx) -> dict:
    """X2 / X3 / F4 / D11 per external screen.

    Each screen is read from the first eval_finetuning_<split>/ that holds it
    (eval_dir records which) and a screen no split holds is listed, so a task
    that died between two screens costs only that screen's numbers. X2 / X3
    are shares of n_pairs, the rows of <Screen>_predictions.csv: the screen's
    pairs whose genes both map to Entrez and both carry an embedding,
    duplicates averaged (eval_finetuning.prepare_dataset) - not of the screen
    file as published."""
    ext = _load_module("_pt_external_transfer", HERE / "external_transfer.py")
    frames = _screen_frames(ctx)
    for v, model_dir, fr in frames:
        gone = [s for s in SCREENS if s not in fr]
        if gone:
            ctx.miss(f"numbers.json screens {variant_row(v)}: no "
                     f"{', '.join(gone)} _predictions.csv under any "
                     f"eval_finetuning_<split>/ of "
                     f"{model_dir.relative_to(ctx.root)}; X2 / X3 / F4 / D11 "
                     f"left empty for {'it' if len(gone) == 1 else 'them'}")
    frames = [(v, fr) for v, _, fr in frames if fr]
    if not frames:
        raise RuntimeError("no <Screen>_predictions.csv under any variant's "
                           "cv3 combo eval_finetuning_<split>/")
    genes = set(ext.load_gene_index(ctx.folds / "genes.txt"))
    pos = load_pair_set(ctx.root / ctx.meta.get(
        "pos_path", "data/SL_SynLethDB_experimental.txt"))
    neg = load_pair_set(ctx.root / ctx.meta.get(
        "neg_path", "data/SL_SynLethDB_experimental_negatives.txt"))
    per_variant = {}
    for v, fr in frames:
        # Per screen: they may come from different splits' eval dirs.
        rec = {"eval_dir": {s: str(ed.relative_to(ctx.root))
                            for s, (ed, _) in fr.items()},
               "sldb_overlap_sources": {
                   s: ext.read_eval_config(
                       ed.parent, ed.name.split("_")[-1]).get(
                           "sldb_overlap_sources")
                   for s, (ed, _) in fr.items()}}
        for s, (ed, df) in fr.items():
            g1 = df["gene1_entrez"].astype(str).str.strip()
            g2 = df["gene2_entrez"].astype(str).str.strip()
            both = (g1.isin(genes) & g2.isin(genes)).to_numpy()
            ov = ext._as_bool(df["sldb_overlap"])
            keys = [canon(a, b) for a, b in zip(g1, g2)]
            in_pos = np.array([k in pos for k in keys])
            in_neg = np.array([k in neg for k in keys])
            n_both = int(both.sum())
            rec[s] = {
                "n_pairs": int(len(df)), "n_both_in_genes_txt": n_both,
                "X2_share_both_in_genes_txt": n_both / len(df) if len(df)
                else None,
                "X3_sldb_overlap_share_among_both": float(ov[both].mean())
                if n_both else None,
                "recomputed_share_in_pos_or_neg_among_both":
                float((in_pos | in_neg)[both].mean()) if n_both else None,
                "recomputed_share_in_pos_only_among_both":
                float(in_pos[both].mean()) if n_both else None,
                "F4_additive_r2": _additive_r2(g1, g2, df["target"]),
                "D11_sldb_agreement": _sldb_agreement(df["target"], in_pos,
                                                      in_neg),
            }
        per_variant[v] = rec
    first = next(iter(per_variant.values()))
    # n_pairs too: X2 / X3 are shares of it, and a CSV cut short can keep both.
    keys = ("n_pairs", "X2_share_both_in_genes_txt",
            "X3_sldb_overlap_share_among_both")
    # Unknown, not True, while a record is short a screen: two records both
    # missing it would compare equal (None == None). Over the variants of the
    # run as well as the records, so a variant whose eval produced NO screen
    # at all (no record here, named above) also leaves the claim unknown
    # instead of dropping out of it.
    short = [v for v in dict.fromkeys(list(ctx.variants) + list(per_variant))
             if any(s not in per_variant.get(v, {}) for s in SCREENS)]
    same = all(r.get(s, {}).get(k) == first.get(s, {}).get(k)
               for r in per_variant.values() for s in SCREENS
               for k in keys) if not short else None
    # Every variant's eval reads the same screen files through the same gene
    # universe and pair files, so the rows of a screen's CSV cannot depend on
    # the variant: records that differ mean a CSV that is not the whole file
    # (a partial copy of the results tree) or not this run's, and then one
    # variant's X2 / X3 / F4 / D11 are not the screen's. Listed per screen,
    # over the records that hold it (so also next to an absent screen).
    for s in SCREENS:
        seen = {v: tuple(r[s][k] for k in keys)
                for v, r in per_variant.items() if s in r}
        if len(set(seen.values())) > 1:
            ctx.miss(f"numbers.json screens {s}: the variants' "
                     f"{s}_predictions.csv are not the same rows (n_pairs / "
                     f"X2 / X3: " + "; ".join(
                         f"{variant_row(v)} " + " / ".join(
                             f"{x:.6g}" if isinstance(x, float) else str(x)
                             for x in t) for v, t in seen.items()) +
                     "); every variant's eval writes the same rows, so one of "
                     "these files is cut short or is not this run's: X2 / X3 "
                     "/ F4 / D11 are not all the screen's")
    return {"per_variant": per_variant, "equal_across_variants": same,
            "variants_incomplete": short,
            "reported_variant": next(iter(per_variant))}


def _additive_r2(g1, g2, target) -> Optional[dict]:
    """F4: R^2 of target ~ a_i + a_j (least squares over all screen pairs;
    the pair sum of per-gene effects absorbs the intercept)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    y = np.asarray(target, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None
    y = y[ok]
    g = np.concatenate([np.asarray(g1)[ok], np.asarray(g2)[ok]])
    uniq, inv = np.unique(g, return_inverse=True)
    n = len(y)
    X = coo_matrix((np.ones(2 * n), (np.tile(np.arange(n), 2), inv)),
                   shape=(n, len(uniq))).tocsr()
    a = lsqr(X, y, atol=1e-12, btol=1e-12, iter_lim=10000)[0]
    ss_tot = float(((y - y.mean())**2).sum())
    if ss_tot == 0:
        return None
    return {"r2": 1 - float(((y - X @ a)**2).sum()) / ss_tot,
            "n_pairs": int(n), "n_genes": int(len(uniq))}


def _sldb_agreement(target, in_pos, in_neg) -> Optional[dict]:
    """D11: AUROC of the screen target (higher = more SL) against the
    SynLethDB label on the screen pairs SynLethDB also lists (a pair in both
    files is positive, the shared folds' OR-collapse)."""
    t = np.asarray(target, dtype=np.float64)
    known = (in_pos | in_neg) & np.isfinite(t)
    if not known.any():
        return None
    return {"n_overlap": int(known.sum()), "n_sldb_pos": int(in_pos[known].sum()),
            "n_sldb_neg": int((~in_pos[known]).sum()),
            "auroc_target_vs_sldb_label": auroc_rank(t[known], in_pos[known])}


def item_margins(ctx: Ctx, tiers: List[dict]) -> dict:
    """R4 / D2: per variant, lead or deficit against the peers per tier."""
    get = {(r["cv"], r["model"]): r for r in tiers}
    out = {}
    for v in ctx.variants:
        row = variant_row(v)
        rec = {}
        for cv in CVS:
            me = get.get((cv, row))
            if not me or me["auprg_mean"] is None:
                continue
            want = ctx.tier_peers(cv)  # less an optional peer not run
            for m in METRICS:
                peers = {p: get[(cv, p)][f"{m}_mean"] for p in want
                         if (cv, p) in get and
                         get[(cv, p)][f"{m}_mean"] is not None}
                if not peers:
                    continue
                # A lead over the peers that survived is not the paper's.
                full = len(peers) == len(want)
                if not full:
                    ctx.miss(f"numbers.json margins {cv}/{m}: no "
                             f"{', '.join(p for p in want if p not in peers)}"
                             f" row; best_peer / margin / leads left empty")
                bp = max(peers, key=peers.get) if full else None
                sl = get.get((cv, "SLMGAE"), {})
                rec.setdefault(cv, {})[m] = {
                    "sleuth": me[f"{m}_mean"], "sleuth_sd": me[f"{m}_sd"],
                    "peers_compared": sorted(peers),
                    "best_peer": bp,
                    "best_peer_mean": peers[bp] if full else None,
                    "margin_vs_best_peer": (me[f"{m}_mean"] - peers[bp]
                                            if full else None),
                    "leads": me[f"{m}_mean"] > peers[bp] if full else None,
                    "slmgae_minus_sleuth": (sl.get(f"{m}_mean") - me[f"{m}_mean"]
                                            if sl.get(f"{m}_mean") is not None
                                            else None),
                    "slmgae_sd": sl.get(f"{m}_sd")}
        g3 = rec.get("cv3", {}).get("auroc", {})
        g1 = rec.get("cv1", {}).get("auroc", {})
        if g3.get("slmgae_minus_sleuth") is not None:
            gap = g3["slmgae_minus_sleuth"]
            spread = max(x for x in (g3["sleuth_sd"], g3["slmgae_sd"])
                         if x is not None) if (g3["sleuth_sd"] is not None or
                                               g3["slmgae_sd"] is not None) \
                else None
            rec["tier3_auroc_gap_slmgae_minus_sleuth"] = gap
            rec["tier3_auroc_gap_inside_spread"] = (
                abs(gap) <= spread if spread is not None else None)
            if g1.get("slmgae_minus_sleuth") is not None:
                rec["tier1_auroc_margin_slmgae_minus_sleuth"] = \
                    g1["slmgae_minus_sleuth"]
                rec["tier3_gap_not_smaller_than_tier1_margin"] = (
                    gap >= g1["slmgae_minus_sleuth"])
        out[v] = rec
    return out


def item_external(ctx: Ctx, ext_rows: List[dict], ft_rows: List[dict]) -> dict:
    """X5 ranks on identical rows; F2 / F3 fine-tuning vs zero-shot."""
    out = {}
    for v in ctx.variants:
        row = variant_row(v)
        ranks, ft = {}, {}
        for s in SCREENS:
            for sp in SPLITS:
                cell = {r["model"]: r["spearman_mean"] for r in ext_rows
                        if r["screen"] == s and r["split"] == sp and
                        r["row_set"] == "matched" and
                        (r["model"] == row or r["model"] in PEERS) and
                        r["spearman_mean"] is not None}
                if row in cell:
                    order = sorted(cell, key=cell.get, reverse=True)
                    # A rank among the peers that survived is not the paper's.
                    gone = [p for p in PEERS if p not in cell]
                    if gone:
                        ctx.miss(f"numbers.json external X5 {s}/{sp}: no "
                                 f"matched {', '.join(gone)} row; rank left "
                                 f"empty")
                    ranks[f"{s}/{sp}"] = {
                        "rank": None if gone else order.index(row) + 1,
                        "of": len(order), "order": order,
                        "peers_compared": sorted(p for p in PEERS
                                                 if p in cell)}
                m = {r["method"]: r["spearman_mean"] for r in ft_rows
                     if r["model"] == row and r["screen"] == s and
                     r["split"] == sp and r["row_set"] == "all_sldb_filtered"}
                if m:
                    zs = m.get("zero_shot_raw")
                    # ...and against zero_shot_ols, the calibrated head
                    # LP_last / Full are warm-started from (it differs from
                    # raw by the sign of a, a < 0 on Adamson): the paper's F2
                    # quotes the raw column, the gain fine-tuning itself adds
                    # is the OLS delta (external_transfer.py's summary.md).
                    ols = m.get("zero_shot_ols")
                    ft[f"{s}/{sp}"] = {
                        **m, "full_minus_zero_shot_raw":
                        m["Full"] - zs if m.get("Full") is not None and
                        zs is not None else None,
                        "lp_last_minus_zero_shot_raw":
                        m["LP_last"] - zs if m.get("LP_last") is not None and
                        zs is not None else None,
                        "full_minus_zero_shot_ols":
                        m["Full"] - ols if m.get("Full") is not None and
                        ols is not None else None,
                        "lp_last_minus_zero_shot_ols":
                        m["LP_last"] - ols if m.get("LP_last") is not None and
                        ols is not None else None}
                    if FT_SCRATCH in m:
                        # What pretraining adds under the same schedule.
                        ft[f"{s}/{sp}"]["full_minus_scratch"] = (
                            m["Full"] - m[FT_SCRATCH]
                            if m.get("Full") is not None and
                            m[FT_SCRATCH] is not None else None)
        out[v] = {"X5_rank_on_matched_rows": ranks, "F2_F3_all_sldb_filtered":
                  ft}
    return out


# =============================================================================
# README
# =============================================================================

MAPPING = [
    ("The PCA arm of every table and number here", "every CSV / numbers.json "
     "/ README.md", "pca_arm (the first column of every CSV, on every row); "
     "run.pca_arm / pca (its label) / pca_targets / pca_settings (the PCA the "
     "combos trained with) / sia_results / cmp_results / step1_results; "
     "README.md line 1 and the first Run settings bullet. The arms side by "
     "side: sl_comparison/results/pca_arms/ (--pca_arms_table)"),
    ("tab:tiers AUROC / AUPR / AUPRG mean (sd)", "tab_tiers.csv",
     "auroc_mean/_sd, ap_mean/_sd (the paper's AUPR is average precision), "
     "auprg_mean/_sd; bold = best_<metric> (among the SLeuth variants, "
     "the three peers and, in PCA arm none, the BilinearProbe baseline "
     "(kind baseline: siamese_sl/slurm/run_probe.sh); ties at the printed 3 "
     "decimals are all bold; empty for a tier with one of them missing), "
     "beats_peers_<metric> per variant (on the exact values, against the "
     "three published peers only, never the probe; empty unless all three "
     "peers have a row). With ALLMOD_COMBO=1 also kind ablation, rows "
     "siamese_sl_allmod[_<tag>] (siamese_sl/slurm/run_allmod_combo.sh: every "
     "catalog embedding, no modality selection), after the SLeuth rows: "
     "eligible for the bold, never beats_peers_*; params_fold_* from "
     "comparison.csv trainable_params_fold"),
    ("tab:tiers trainable parameters", "tab_tiers.csv",
     "SLeuth: params_fold_mean / params_fold_sd over folds (combo "
     "results.json fold_metrics[].total_params); peers: trainable_params "
     "(empty on SLeuth rows: comparison.csv's copy is the last fold's count "
     "only)"),
    ("tab:tiers caption pi (TT1)", "tab_tiers.csv / numbers.json",
     "baseline_mean (cv1); items.fold_geometry.cv1.test_prevalence"),
    ("tab:tiers / tab:cellline row provenance", "numbers.json",
     "items.fold_provenance (every row's graded-on / trained-on fold "
     "fingerprints vs the current folds); a row not graded by "
     "evaluate_model.py or not on the current folds is left out and listed "
     "under Missing"),
    ("tab:solo (+AB1)", "tab_solo.csv",
     "auprg_any_mean / auprg_any_sd per variant x embedding x tier; bold = "
     "best_in_category_paper16 (over the paper's 16 printed rows; ties at "
     "the printed 3 decimals are all bold; best_in_category is the exact "
     "maximum over every catalog embedding that ran; both empty for a "
     "single-candidate category such as kg, the paper's also for a category "
     "with one of its paper-16 members not fresh); in_paper marks the "
     "paper's 16; status"),
    ("AB2 category means", "numbers.json", "items.solo.<variant>.per_cv.<cv>"
     ".category_means_paper16 / best_category_paper16 (over the paper's 16, "
     "the paper's definition; a category's mean is empty with one of its "
     "members not fresh, the best category with any of the 16 not fresh: "
     "paper16_absent names them); category_means / best_category over every "
     "catalog embedding that ran (unrounded)"),
    ("AB3 / D4 single modalities above the multi-modal model",
     "numbers.json", "items.solo.<variant>.per_cv.<cv>."
     "n_above_combo_shared_paper16 (the paper's 'of the sixteen', vs the "
     "tab:tiers AUPRG; empty unless all 16 are fresh) / _self (vs the "
     "combo's own auprg_any); n_above_combo_* over every catalog embedding "
     "that ran; above_combo_at_every_tier_* (empty while one of the 16 is "
     "undecided, i.e. without a fresh row at a tier; *_undecided)"),
    ("tab:loco (+AB4)", "tab_loco.csv", "auprg_any_mean / auprg_any_sd; "
     "dropped='none' is the reference row (shared_eval_auprg = its "
     "tab:tiers value); delta_vs_reference (difference of means)"),
    ("tab:loco caption: paired per-split spread", "tab_loco.csv",
     "delta_paired_mean / delta_paired_sd / n_paired (fold-by-fold ablation "
     "- reference auprg_any on the same splits, population sd)"),
    ("AB5 LOCO deltas", "numbers.json", "items.loco.<variant>.<cv> (deltas; "
     "paired.<category>.mean / sd / n; largest_drop empty unless every "
     "category's ablation has a delta, categories_missing)"),
    ("tab:solo / tab:loco on the tab:tiers evaluator (optional)",
     "tab_solo.csv / tab_loco.csv", "auprg_shared_mean / "
     "auprg_shared_sd (only with --regrade, for cells whose fold matrices "
     "were kept)"),
    ("tab:cellline (+CL1)", "tab_cellline.csv",
     "condition included / removed (cell_emb=0) / retrained without cell "
     "lines (pooled); <metric>_mean/_sd, delta_<metric>_vs_included; bold = "
     "the sign of the removed row's delta (included vs removed only); "
     "selfcheck_as_trained. A removed row whose zero_cell_emb.py run did not "
     "pass, ablated another combo or does not reproduce the included row is "
     "'stale: ...' (no delta, not in CL1)"),
    ("tab:cellline: what cell_emb=0 leaves", "tab_cellline.csv",
     "residual_head_terms on the removed row (per-head terms that still tell "
     "the heads apart, e.g. per-head c_h); condition 'self-scored: ...' rows "
     "(zero_cell_emb.py, same test edges and metrics, forward-pass scores; "
     "delta vs its own as-trained row), e.g. +head_proj_bias=mean / =0"),
    ("tab:external (+X1, X4)", "tab_external.csv",
     "spearman_mean / spearman_sd per screen x split x model; SLeuth on "
     "row_set matched (paper_convention) and all_sldb_filtered, peers on "
     "matched (paper_convention); bold = best_matched (ties at the printed 3 decimals are all "
     "bold; empty for a screen / split without a matched row of every "
     "SLeuth variant and peer); split_label (the paper's 'one gene "
     "unseen' row is split cv2, >=1 gene unseen); scored_head (SLeuth). "
     "paper_convention: every model on row_set matched, the identical rows "
     "(best_matched and X5 use it too); SLeuth's all_sldb_filtered row (pairs "
     "with a gene outside genes.txt included) is kept for reference, as "
     "the submitted paper printed it"),
    ("tab:external, in-distribution reading", "tab_external.csv",
     "screen_role (in_distribution_check: >= 50% of the screen's scored "
     "pairs in SynLethDB, e.g. Gilbert; else external); row_set "
     "matched_incl_sldb (all matched test pairs, SynLethDB ones kept) next "
     "to matched (held out), matched_in_sldb (the SynLethDB part), every "
     "model, and model SynLethDB_label (kind sldb_label: the label itself "
     "as a predictor on matched_in_sldb); bootstrap CIs in "
     "<cmp>/bootstrap/ci_external.csv; counts in "
     "external_transfer/sldb_overlap.csv"),
    ("SynLethDB vs screen agreement", "tab_sldb_agreement.csv",
     "auroc / spearman (+ _ci_low / _ci_high) per screen x scope "
     "(all_pairs, all_pairs_genes_txt, test_genes_txt per split)"),
    ("X5 ranks", "numbers.json", "items.external.<variant>."
     "X5_rank_on_matched_rows (ranked on the exact values; rank empty unless "
     "all three peers have a matched row; peers_compared)"),
    ("tab:external-ft (+F1)", "tab_external_ft.csv",
     "column Zero-shot (= zero_shot_raw) / LP_last / Full, row_set "
     "all_sldb_filtered (paper_convention) and matched; best_method = the "
     "best of the three columns (the paper prints no bold here; empty with "
     "one of them absent); method zero_shot_ols (column '(reference) "
     "zero-shot OLS') is not a paper column but the OLS-calibrated head "
     "LP_last / Full are warm-started from (it differs from zero_shot_raw by "
     "the sign of a, a < 0 on Adamson), never bolded; "
     "scored_head = the head that scored the screen (eval_finetuning "
     "effective_head: the screen's cell-line head for per_head, OTHER for "
     "Corn / RPE-1, 'ANY (pooled)' for a pooled variant)"),
    ("F2 / F3", "numbers.json",
     "items.external.<variant>.F2_F3_all_sldb_filtered (zero_shot_raw is the "
     "paper's printed Zero-shot column and the baseline of the "
     "*_minus_zero_shot_raw deltas; *_minus_zero_shot_ols measure the same "
     "gain against the calibrated head fine-tuning starts from)"),
    ("I1 / E1 data statistics", "numbers.json", "items.data_stats"),
    ("E2 / M8 fold metadata and conflicts", "numbers.json", "items.folds_meta"),
    ("D1 / D7 / D8 / D9 / D10 fold geometry", "numbers.json",
     "items.fold_geometry.<cv> (test_share, discarded_share, "
     "test_prevalence, test_both_genes_in_train[_val]); D7_analytic_gene_split"),
    ("R5 / M12 input width (after the arm's PCA)", "numbers.json",
     "items.pca_width.per_variant.<variant> (the encoder input width after "
     "the arm's PCA; min, max over the splits; empty unless every split of "
     "every tier has a width)"),
    ("I2 parameter ratio", "numbers.json", "items.param_ratio.<variant> "
     "(min_ratio empty unless every tier has a ratio)"),
    ("M7 / M9 heads", "numbers.json", "items.heads"),
    ("M3-M6 / M10 / D3 architecture and training settings", "numbers.json",
     "items.arch.<variant>.arch_signature; items.arch.<variant>.pca_arm / "
     "pca_settings (the PCA its combo applied)"),
    ("E3 / M11 selected modalities", "numbers.json", "items.selection"),
    ("R4 / D2 prose margins", "numbers.json", "items.margins.<variant> "
     "(best_peer / margin_vs_best_peer / leads empty unless all three peers "
     "have a row; peers_compared)"),
    ("X2 / X3 screen coverage", "numbers.json",
     "items.screens.per_variant.<variant>.<Screen>.X2_share_both_in_genes_txt"
     " / X3_sldb_overlap_share_among_both (shares of n_pairs, the rows of "
     "<Screen>_predictions.csv: the screen's pairs whose genes both map to "
     "Entrez and both carry an embedding, duplicate pairs averaged - not of "
     "the screen file as published; each screen is taken from the first "
     "eval_finetuning_<split>/ that holds it, eval_dir names which)"),
    ("F4 additivity", "numbers.json",
     "items.screens.per_variant.<variant>.<Screen>.F4_additive_r2"),
    ("D11 SynLethDB agreement", "numbers.json",
     "items.screens.per_variant.<variant>.<Screen>.D11_sldb_agreement"),
    ("R2 SynLethKG entity / relation types", "numbers.json",
     "items.inputs.synlethkg (n_entity_types from the relation names, "
     "n_relations, removed_label_relations; data/synlethkg/meta.json, the KG "
     "SLGNN ran on)"),
    ("D3 single training seed", "numbers.json",
     "items.inputs.training_seeds (model_params.json seed per row, via "
     "comparison.csv), single_training_seed"),
    ("AP1 / AP2 embedding widths", "numbers.json",
     "items.inputs.native_dims.<embedding> (real columns: ComplEx's 256 "
     "complex dims are 512)"),
]


def arm_line(ctx: Ctx) -> str:
    """README's first Run settings bullet: which PCA arm, its PCA (config.conf
    pca_arm_label's words, from this job's targets), where its results are
    (its single-embedding runs with ctx.step1_label: the selection's own
    record of who shares them, else config.conf's wording over --pca_arms),
    and the PCA its combos actually trained with."""
    tp = ctx.trained_pca
    trained = ("unknown (no combo of this arm read, or they disagree: listed)"
               if tp["modality"] is None or tp["joint"] is None else
               f"PCA within each modality {tp['modality']}, on the "
               f"concatenation {tp['joint']}")
    return (f"- PCA arm: {ctx.pca_arm} ({ctx.pca_label}); siamese results "
            f"{ctx.rel(ctx.sia_results)}/, comparison results "
            f"{ctx.rel(ctx.cmp_results)}/ (the shared models' rows from "
            f"{ctx.rel(ctx.shared_results)}/); single-embedding runs "
            f"{ctx.rel(ctx.step1_results)}/ "
            f"({ctx.step1_label}); its combos "
            f"trained with {trained} (numbers.json run.pca_settings)")


def render_readme(ctx: Ctx, status: Dict[str, str], numbers: dict) -> str:
    # Line 1 names the arm; the verdict stays on line 3 (run_paper_tables.sh
    # and the orchestrators read it there).
    L = [f"# Paper tables and numbers: PCA arm {ctx.pca_arm}", "",
         numbers["verdict"], "",
         f"Generated {numbers['generated_at']} by sl_comparison/paper_tables.py "
         f"from {ctx.root}.", ""]
    L += ["## Tables", "", "| Table | Status |", "|---|---|"]
    L += [f"| {t}.csv | {status[t]} |" for t in TABLES]
    L += ["", "## Where each paper item is", "",
          "In numbers.json each `items.<key>` is `{paper, value, status}`, so "
          "a key path `items.<key>.<rest>` in this README means "
          "`items.<key>.value.<rest>` (e.g. "
          "`items.fold_geometry.cv1.test_prevalence` is "
          "`items.fold_geometry.value.cv1.test_prevalence`).", "",
          "| Paper item | File | Column / key |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in MAPPING]
    sel = numbers["items"].get("selection", {}).get("value") or {}
    meta = (ctx.selection or {}).get("_selection_meta") or {}
    unchecked = "(unset: run-token check skipped)"
    drift = ("not checked" if ctx.code_drift is None else
             f"{len(ctx.code_drift)} file(s) (numbers.json run.code_drift)"
             if ctx.code_drift else "none")
    L += ["", "## Run settings", "", arm_line(ctx),
          f"- SLeuth variants: {', '.join(ctx.variants)} (rows "
          f"{', '.join(variant_row(v) for v in ctx.variants)})",
          f"- SELECTION_VARIANT: {ctx.selection_variant}; single-modality "
          f"runs for the other variants: "
          f"{'on' if ctx.extra_variants else 'off'} (BESTCAT_EXTRA_VARIANTS)",
          f"- All-modalities combo (no modality selection; tab_tiers kind "
          f"{ALLMOD_KIND}): "
          + (f"on, rows {', '.join(allmod_row(v) for v in ctx.variants)}"
             if ctx.allmod else "off") + " (ALLMOD_COMBO)",
          f"- RUN_TOKEN: {ctx.run_token or '(unset)'}; single-modality dirs "
          f"checked against {ctx.solo_run_token or unchecked}, ablation dirs "
          f"against {ctx.ablation_run_token or unchecked}",
          f"- Code manifest (written at submission): "
          f"{ctx.code_manifest or '(none: not an orchestrated run)'}; code "
          f"changed, added or removed since: {drift}",
          f"- ARCH_ARGS (environment, else config.conf): "
          f"{ctx.arch_args_env!r}; the recorded flags of each variant are "
          f"items.arch.<variant>",
          f"- Embedding catalog: {len(ctx.catalog)} entries "
          f"({len(PAPER_SOLO)} in the paper; extra: "
          f"{', '.join(catalog_not_in_paper(ctx)) or 'none'})",
          f"- Winners: {json.dumps(sel.get('winners'))}; differing from the "
          f"paper: {json.dumps(sel.get('differs_from_paper'))}",
          (f"- Modality selection: PER FOLD of every CV "
           f"(SELECTION_MODE=per_fold): each category's argmax of "
           f"{meta.get('selection_metric')} on that fold's own selection "
           f"split (a random {meta.get('selection_frac')} of its training "
           f"pairs, never trained on; seed {meta.get('selection_seed')}); "
           f"each CV's combo trains every fold on its own winners. Picks per "
           f"tier: numbers.json items.selection.per_fold / pick_counts, "
           f"tab_solo n_folds_picked"
           if meta.get("selection_mode") == "per_fold" else
           f"- Modality selection: once, on "
           f"{meta.get('selection_cv', 'SELECTION_CV')} validation "
           f"({meta.get('selection_metric', 'val_<select_metric>_mean')}, the "
           f"mean over its folds), reused for every CV and variant. Not "
           f"leakage-free across folds: each fold's training / validation "
           f"pairs include other folds' test pairs (accepted; "
           f"siamese_sl/slurm/run_best_per_category.conf)"),
          "- SynLethDB filter: a screen row is SynLethDB-filtered when its "
          "pair is in neither the SL positives nor the screened negatives "
          "(eval_finetuning.py --neg_path; external_transfer.py refuses a "
          "positives-only filter). 'matched' rows also have both genes in "
          "sl_comparison/folds/genes.txt.",
          "- Every sd is the population sd (ddof=0)."]
    L += ["", "## Missing inputs and items", ""]
    L += [f"- {m}" for m in ctx.missing] or ["- none"]
    L += ["", "## Absent by design", "",
          "Intended by the run's settings; not part of the verdict.", ""]
    L += [f"- {m}" for m in ctx.absent_by_design] or ["- none"]
    if ctx.notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in ctx.notes]
    return "\n".join(L) + "\n"


# =============================================================================
# Cross-arm comparison: tab_pca (--pca_arms_table)
# =============================================================================


def selection_token(ctx: Ctx) -> Optional[str]:
    """The run_token an arm's selection records (_selection_meta.run_token):
    the submission whose Step-1 runs it ranked and whose combos it built (one
    token per submission, so the arms of one submission share it)."""
    return ((ctx.selection or {}).get("_selection_meta") or {}).get(
        "run_token")


def selection_cv(ctx: Ctx) -> Optional[str]:
    """The CV whose validation scores an arm's selection ranked
    (_selection_meta.selection_cv: the combo job's SELECTION_CV, one per
    submission), or None when it records none."""
    cv = ((ctx.selection or {}).get("_selection_meta") or {}).get(
        "selection_cv")
    return cv if isinstance(cv, str) and cv else None


def _cv_text(cv: Optional[str]) -> str:
    """A selection CV as a stale tab_pca cell names it."""
    return f"{cv} validation" if cv else "a CV its selection does not record"


def combo_files(c: Optional[dict]) -> Optional[List[str]]:
    """A combo's embedding file names, sorted (its modality set); None
    without a combo."""
    if not c:
        return None
    return sorted(Path(str(p)).name
                  for p in c["config"].get("embeddings_paths") or [])


def combo_modalities(ctx: Ctx, c: dict, cv: str) -> List[str]:
    """The embedding types a combo was trained on, in its arm's selection's
    order (best_per_category.json's categories at cv, else at the first CV
    it records), each named as the selection names it, else as the catalog
    does, else by its file name; a file the selection lacks (a legacy or
    foreign combo, which items.arch lists) after them."""
    files = [Path(str(p)).name
             for p in c["config"].get("embeddings_paths") or []]
    sel = ctx.selection or {}
    chosen = sel.get(cv) if isinstance(sel.get(cv), dict) else next(
        (sel[x] for x in CVS if isinstance(sel.get(x), dict)), {})
    by_file = {Path(str(i["file"])).name: str(i.get("type") or
                                              Path(str(i["file"])).stem)
               for i in chosen.values()
               if isinstance(i, dict) and i.get("file")}
    # a per_fold selection's combo trains on every embedding a fold picked
    for i in chosen.values():
        if isinstance(i, dict) and isinstance(i.get("files"), list):
            by_file.update({Path(str(fl)).name: str(ty) for fl, ty in
                            zip(i["files"], i.get("types") or [])})
    catalog = {Path(efile).name: etype for etype, efile, _ in ctx.catalog}
    return ([t for f, t in by_file.items() if f in files] +
            [catalog.get(f, Path(f).stem) for f in files if f not in by_file])


def graded_folds(ctx: Ctx, cv: str, model: str,
                 key: str = "auprg") -> Optional[List[dict]]:
    """The per-fold records behind a tabulated graded row: its
    evaluate_model JSON's summary.folds (one dict per fold, in fold-dir
    order; evaluate_model.py records no fold id, so they pair by position),
    or None when that JSON is absent or not on the current folds, or when
    those folds are not what its mean is over (their mean of key is not its
    <key>_mean within 1e-9: a list another grading left). Ctx.metric has
    already refused a partial or otherwise unusable row, so both sides of a
    pair hold every fold."""
    s = ctx.json_rows.get((cv, model))
    if s is None or not ctx.fold_ok(cv, model, "json"):
        return None
    folds = s.get("folds")
    if not isinstance(folds, list) or not folds or \
            not all(isinstance(f, dict) for f in folds):
        return None
    vals = [_num(f.get(key)) for f in folds]
    mean = _num(s.get(f"{key}_mean"))
    if mean is None or any(x is None for x in vals) or \
            abs(float(np.mean(vals)) - mean) > 1e-9:
        return None
    return folds


def delta_verdict(delta: Optional[float], se: Optional[float],
                  n: Optional[int]) -> str:
    """How tab_pca reads an arm's AUPRG difference from the reference arm,
    by the rule siamese_sl/slurm/run_ablation_summary.sh reads its ablations
    with (and the PCA ablation summary of 2026-10-19 its arms): a |delta| up
    to NO_MEASURABLE_DIFFERENCE is none; a larger one is a direction only
    when it also exceeds 2 SE of the fold-paired differences, else within
    fold noise; with fewer than two paired folds it is not read."""
    if delta is None:
        return "n/a (no auprg on one side)"
    if abs(delta) <= NO_MEASURABLE_DIFFERENCE:
        return "no measurable difference"
    if not n or n < 2 or se is None:
        return "n/a (fewer than 2 paired folds)"
    if delta > 2 * se:
        return "better than the reference"
    if -delta > 2 * se:
        return "worse than the reference"
    return "within fold noise (|delta| <= 2 SE)"


def _pca_arm_deltas(grp: List[dict], views: Dict[str, Ctx]) -> None:
    """Fill one variant x tier's differences from its reference arm
    (grp[0]): of the graded means, and the AUPRG's fold-paired spread and
    reading. Nothing is differenced against a cell that is not ok."""
    rr = grp[0]
    for r in grp:
        if r["status"] != "ok":
            r["verdict"] = f"n/a ({r['status'].split(':', 1)[0]})"
            continue
        if rr["status"] != "ok":
            r["verdict"] = "n/a (the reference arm's cell is not ok)"
            continue
        for m in METRICS:
            x, y = r.get(f"{m}_mean"), rr.get(f"{m}_mean")
            r[f"delta_{m}"] = (x - y if x is not None and y is not None
                               else None)
        if r is rr:
            r["verdict"] = "n/a (the reference arm)"
            continue
        d = None
        if r["_folds"] and rr["_folds"]:
            d = _paired(r["_folds"], rr["_folds"], "auprg")
            if d is None:
                views[r["pca_arm"]].miss(
                    f"tab_pca {r['model']}/{r['cv']}: its "
                    f"{len(r['_folds'])} per-fold graded auprg do not pair "
                    f"with the reference arm {rr['pca_arm']}'s "
                    f"{len(rr['_folds'])} (no paired spread)")
        if d:
            sp = _spread(d)
            r["_paired"] = sp
            r.update(delta_auprg_paired_sd=sp["sd"], delta_auprg_se=sp["se"],
                     n_paired=sp["n"])
        r["verdict"] = delta_verdict(r["delta_auprg"],
                                     r.get("delta_auprg_se"),
                                     r.get("n_paired"))


def build_pca_arms(ref: Ctx, views: Dict[str, Ctx]) -> Tuple[List[dict], str]:
    """tab:pca: per variant x tier x arm (views: {arm: its context} in the
    order none, joint, per_modality, both; the first is the reference arm)
    the arm's combo (its PCA, single-embedding runs, modalities, widths and
    size) and its GRADED comparison row (Ctx.metric, the shared evaluator,
    as tab:tiers: every arm's rows are graded by one evaluator on the same
    folds, while train.py's self-scored auprg_any is each arm's own), and the
    difference from the reference arm's. A cell is tabulated only when the
    combo is the arm's (Ctx._load_combo), it is the run its row graded, its
    arch_signature equals the reference arm's combo outside the four PCA keys
    (the arms differ there alone by design), its selection's run_token is
    the reference arm's (one submission: an arm reused from another run is
    another selection and another training), so is its selection's
    _selection_meta.selection_cv (2026-10-21: SELECTION_CV is in no
    signature, so a combo job resubmitted by hand without the run's selected
    on another CV, exited 0 and kept the token), every embedding file it
    shares with the reference combo was trained on the same content, and its
    graded row is usable; otherwise it is missing / stale, without values,
    and listed. An arm without its best_per_category.json (listed by its Ctx)
    has no cell at all ("missing: no best_per_category.json"): which
    submission it is and what it selected are unknown, so it is neither
    called another submission's nor compared in the token check. Without
    arch_signature.py (listed by Ctx) no arm can be told apart: no row at
    all."""
    rows: List[dict] = []
    if ref.arch is None:
        return rows, "missing"
    arms = list(views)
    ref_arm = arms[0]
    rv = views[ref_arm]
    no_sel = {a for a, w in views.items() if w.selection is None}
    tokens = {a: selection_token(w) for a, w in views.items()}
    # The CV whose validation scores chose each arm's modalities: one per
    # submission (the orchestrators hand every arm's combo job the same
    # SELECTION_CV), and in no signature, so an arm whose combo job was
    # resubmitted by hand without the run's differs only here.
    sel_cvs = {a: selection_cv(w) for a, w in views.items()}
    # The selection mode too (2026-11-13): in no signature either, so one
    # arm's combo resubmitted with SELECTION_MODE=global sat beside per-fold
    # arms unflagged.
    sel_modes = {a: getattr(w, "selection_mode", None)
                 for a, w in views.items()}
    # Two arms that record no token compare equal: one submission is then
    # not shown, which is listed once (every combo job records one). An arm
    # without a selection records nothing either way (listed apart).
    unknown = [a for a in arms if a not in no_sel and tokens[a] is None]
    if unknown:
        ref.miss(f"tab_pca: the selection of PCA arm(s) {', '.join(unknown)} "
                 f"records no _selection_meta.run_token, so those arms are "
                 f"not shown to be one submission with the others",
                 prefixed=False)
    pca_keys = ref.arch.PCA_SIGNATURE_KEYS
    groups: List[List[dict]] = []
    for v in ref.variants:
        model = variant_row(v)
        for cv in CVS:
            rc = rv.combos.get((v, cv))
            grp = []
            for a in arms:
                w = views[a]
                r = {"pca_arm": a,
                     "role": "reference" if a == ref_arm else "arm",
                     "variant": v, "model": model, "cv": cv,
                     "tier": TIER[cv]}
                grp.append(r)
                if a in no_sel:
                    r["status"] = "missing: no best_per_category.json (listed)"
                    continue
                c = w.combos.get((v, cv))
                if c is None:
                    # Why is listed by its context (Ctx._load_combo).
                    r["status"] = w.combo_problem.get(
                        (v, cv), "missing: no combo (listed)")
                    continue
                cfg = c["config"]
                s = ref.arch.pca_settings(cfg)
                mods = combo_modalities(w, c, cv)
                r.update({
                    "modality_pca": pca_setting_text(s["modality"]),
                    "joint_pca": pca_setting_text(s["joint"]),
                    # Its pair's single-embedding runs: native, or under the
                    # per-modality PCA its combo applies too (Step 2 ranks no
                    # candidate of another PCA, and the arm's tab_solo lists
                    # one of another per-modality setting than its combo's).
                    "single_embedding": (
                        "native" if s["modality"] is None else
                        "per-modality PCA " +
                        pca_setting_text(s["modality"])),
                    "modalities": "+".join(mods),
                    "n_modalities": len(mods),
                    "same_modalities_as_reference": (
                        None if rc is None else
                        combo_files(c) == combo_files(rc)),
                    "dir": w.rel(c["dir"])})
                probs = []
                if (v, cv) in w.combo_not_graded:
                    probs.append("its comparison row graded another run than "
                                 "this combo (listed)")
                if a != ref_arm:
                    if ref_arm not in no_sel and tokens[a] != tokens[ref_arm]:
                        probs.append(f"its selection's run_token {tokens[a]} "
                                     f"is not the reference arm {ref_arm}'s "
                                     f"{tokens[ref_arm]} (another "
                                     f"submission)")
                    if ref_arm not in no_sel and \
                            sel_cvs[a] != sel_cvs[ref_arm]:
                        probs.append(f"its modalities were selected on "
                                     f"{_cv_text(sel_cvs[a])}, the reference "
                                     f"arm {ref_arm}'s on "
                                     f"{_cv_text(sel_cvs[ref_arm])} "
                                     f"(SELECTION_CV: another selection "
                                     f"protocol)")
                    if ref_arm not in no_sel and \
                            sel_modes[a] != sel_modes[ref_arm]:
                        probs.append(f"its modalities were selected in mode "
                                     f"{sel_modes[a]}, the reference arm "
                                     f"{ref_arm}'s in {sel_modes[ref_arm]} "
                                     f"(SELECTION_MODE: another selection "
                                     f"protocol)")
                    if rc is not None:
                        diff = ref.arch.signature_diff(cfg, rc["config"],
                                                       ignore=pca_keys)
                        if diff:
                            probs.append(f"its arch_signature differs from "
                                         f"the reference arm {ref_arm}'s "
                                         f"combo outside the PCA keys: "
                                         f"{', '.join(sorted(diff))}")
                        bad = embedding_content_problem(
                            c["results"], rc,
                            name=f"the reference arm {ref_arm}'s combo "
                                 f"{rv.rel(rc['dir'])}")
                        if bad:
                            probs.append(bad)
                if probs:
                    r["status"] = "stale: " + "; ".join(probs)
                    w.miss(f"tab_pca {model}/{cv}: {c['dir']} is stale "
                           f"({'; '.join(probs)})")
                    continue
                for m in METRICS:
                    r[f"{m}_mean"], r[f"{m}_sd"], _ = w.metric(cv, model, m)
                if r["auprg_mean"] is None:
                    r["status"] = "missing: no usable graded row (listed)"
                    w.miss(f"tab_pca {model}/{cv}: no usable graded {model} "
                           f"row ({w.comparison_csv} / "
                           f"{w.cmp_results / f'{model}_{cv}.json'}; a "
                           f"reason, if any, is listed above)")
                    continue
                r["status"] = "ok"
                dims = fold_widths(c["results"])
                if dims is None:
                    w.miss(f"{c['dir']}/results.json fold_metrics[].input_dim")
                else:
                    r.update(input_dim_mean=float(np.mean(dims)),
                             input_dim_min=min(dims), input_dim_max=max(dims))
                stats = fold_param_stats(c["results"])
                if not stats:
                    w.miss(f"{c['dir']}/results.json fold_metrics[]."
                           f"total_params")
                r["params_fold_mean"] = stats.get("params_fold_mean")
                # The paired spread's and the 2x2's input (not a column).
                r["_folds"] = graded_folds(w, cv, model)
                if r["_folds"] is None:
                    w.miss(f"tab_pca {model}/{cv}: no per-fold graded auprg "
                           f"in {w.cmp_results / f'{model}_{cv}.json'} "
                           f"(summary.folds, the folds its auprg_mean is "
                           f"over): no paired spread, no 2x2 effect")
            _pca_arm_deltas(grp, views)
            groups.append(grp)
            rows.extend(grp)
    # The best of every requested arm (exact values), not of those that
    # survived: item_pca_arms lists the gap.
    _flag_best(rows, ["variant", "cv"], "auprg_mean", "best_arm",
               eligible=lambda r: r["status"] == "ok")
    for grp in groups:
        if any(r["status"] != "ok" for r in grp):
            for r in grp:
                r["best_arm"] = None
    have = any(r["status"] == "ok" for r in rows)
    return rows, "ok" if have else "missing"


PCA_ARMS_COLS = (["pca_arm", "role", "variant", "model", "cv", "tier",
                  "modality_pca", "joint_pca", "single_embedding",
                  "modalities", "n_modalities",
                  "same_modalities_as_reference", "input_dim_mean",
                  "input_dim_min", "input_dim_max", "params_fold_mean"] +
                 [f"{m}_{s}" for m in ("auprg", "ap", "auroc")
                  for s in ("mean", "sd")] +
                 [f"delta_{m}" for m in ("auprg", "ap", "auroc")] +
                 ["delta_auprg_paired_sd", "delta_auprg_se", "n_paired",
                  "verdict", "best_arm", "status", "dir"])


def item_pca_arms(ref: Ctx, views: Dict[str, Ctx], rows: List[dict]) -> dict:
    """items.pca: per variant and tier every ok arm's graded means, its
    differences from the reference arm (of means; fold-paired mean / sd / se
    / n of the AUPRG), widths, size and modalities. best_arm (the exact
    maximum; None on an exact tie, best_arms naming every tied arm) is None,
    and listed, unless every requested arm's cell is ok, and the 2x2 effects
    (pca_effects on the graded per-fold AUPRG) unless all four arms are
    requested and ok: decided over the arms that survived they are not the
    comparison's. A run requesting fewer than four arms has no 2x2 (absent by
    design). selection: each arm's winners (None without its
    best_per_category.json), the groups of arms that selected alike, where
    their single-embedding runs are, their run tokens and the CVs they
    selected on (two arms that rank one root and selected differently are
    listed: Step 2 is deterministic on the same runs). settings: this job's
    PCA targets and the PCA each arm's combos trained with."""
    if not any(r["status"] == "ok" for r in rows):
        return {}  # add() lists that once
    arms = list(views)
    four = tuple(arms) == PCA_ARM_NAMES
    out: dict = {"reference_arm": arms[0], "arms": arms}
    for v in ref.variants:
        for cv in CVS:
            ok = {r["pca_arm"]: r for r in rows
                  if (r["variant"], r["cv"]) == (v, cv) and
                  r["status"] == "ok"}
            gone = [a for a in arms if a not in ok]
            rec = {"arms": {}, "arms_missing": gone, "best_arm": None,
                   "best_arms": None, "effects": None}
            for a in arms:
                if a not in ok:
                    continue
                r = ok[a]
                rec["arms"][a] = {
                    **{m: {"mean": r.get(f"{m}_mean"), "sd": r.get(f"{m}_sd")}
                       for m in METRICS},
                    **{f"delta_{m}": r.get(f"delta_{m}") for m in METRICS},
                    "paired": r.get("_paired"),
                    "verdict": r.get("verdict"),
                    "input_dim": {"min": r["input_dim_min"],
                                  "max": r["input_dim_max"],
                                  "mean": r["input_dim_mean"]}
                    if r.get("input_dim_mean") is not None else None,
                    "params_fold_mean": r.get("params_fold_mean"),
                    "modalities": (r["modalities"].split("+")
                                   if r.get("modalities") else [])}
            if gone:
                ref.miss(f"numbers.json pca {v}/{cv}: no value for arm(s) "
                         f"{', '.join(gone)}; best_arm"
                         f"{' and effects' if four else ''} left empty",
                         prefixed=False)
            else:
                means = {a: ok[a]["auprg_mean"] for a in arms}
                top = max(means.values())
                # Every arm at the exact maximum (tab_pca.csv flags each);
                # best_arm only when that is one arm.
                rec["best_arms"] = [a for a in arms if means[a] == top]
                if len(rec["best_arms"]) == 1:
                    rec["best_arm"] = rec["best_arms"][0]
                else:
                    ref.note(f"numbers.json pca {v}/{cv}: "
                             f"{', '.join(rec['best_arms'])} tie at the best "
                             f"auprg_mean; best_arm left empty (best_arms "
                             f"names them)", prefixed=False)
                if four:
                    fl = {a: ok[a].get("_folds") for a in arms}
                    rec["effects"] = (pca_effects(fl, key="auprg")
                                      if all(fl.values()) else None)
                    if rec["effects"] is None:
                        ref.miss(f"numbers.json pca {v}/{cv}: no fold that "
                                 f"all four arms' graded rows scored (their "
                                 f"per-fold auprg, listed above); effects "
                                 f"left empty", prefixed=False)
            out.setdefault(v, {})[cv] = rec
    if not four:
        ref.by_design(f"numbers.json pca effects: the 2x2's per-modality / "
                      f"joint / interaction effects need all four PCA arms; "
                      f"this run compares {', '.join(arms)} (--pca_arms / "
                      f"PCA_ARMS), so they are left empty", prefixed=False)
    # An arm without its best_per_category.json (listed by its Ctx) selected
    # nothing known: None, in no group and in no same-root comparison (its
    # {} would read as a selection that differs).
    have = [a for a in arms if views[a].selection is not None]
    winners = {a: selection_winners(views[a]) if a in have else None
               for a in arms}
    # A per_fold selection is told apart by its per-(CV, fold) picks, not by
    # the union its winners name.
    for a in have:
        if per_fold_selection(views[a]):
            winners[a] = {cv: {cat: {str(k): e for k, e in sorted(
                per_fold_picks(views[a], cv, cat).items())}
                for cat in sorted(views[a].selection.get(cv) or {})}
                for cv in CVS}
    same: List[List[str]] = []
    for a in have:
        g = next((g for g in same if winners[g[0]] == winners[a]), None)
        if g is None:
            same.append([a])
        else:
            g.append(a)
    by_root: Dict[Path, List[str]] = {}
    for a in have:
        by_root.setdefault(views[a].step1_results.resolve(), []).append(a)
    for d, group in by_root.items():
        if len(group) > 1 and any(winners[a] != winners[group[0]]
                                  for a in group[1:]):
            ref.miss(f"PCA arms {', '.join(group)} rank the same single-"
                     f"embedding runs ({ref.rel(d)}/) but selected "
                     "differently (" + "; ".join(
                         f"{a}: {json.dumps(winners[a], sort_keys=True)}"
                         for a in group) + "): one of their selections is "
                     "not of the runs on disk now", prefixed=False)
    out["selection"] = {
        "winners": winners, "same_selection": same,
        "step1_results": {a: w.rel(w.step1_results)
                          for a, w in views.items()},
        "run_token": {a: selection_token(w) for a, w in views.items()},
        "selection_cv": {a: selection_cv(w) for a, w in views.items()}}
    tgt = pca_target_settings("both", ref.pca_targets)
    out["settings"] = {"modality": pca_setting_text(tgt["modality"]),
                       "joint": pca_setting_text(tgt["joint"]),
                       "targets": ref.pca_targets,
                       "trained": {a: w.trained_pca for a, w in views.items()}}
    return out


def arms_index(ref: Ctx, views: Dict[str, Ctx]) -> List[dict]:
    """The PCA arms' index (README '## PCA arms', numbers.json arms[]): each
    arm's label, roots, paper tables and their verdict, line 3 of
    <cmp>/paper_tables/README.md (run_paper_tables.sh writes each arm's
    before this job runs). A README that is absent, or whose title (line 1)
    is not that arm's, gives no verdict; a verdict that is not PAPER
    COMPLETE is one Missing entry naming it. Nor does one whose numbers.json
    (written with it) names another arm or, when this job has a RUN_TOKEN,
    another submission's token (neither this one's nor the arm's selection's):
    an arm's tables job that wrote nothing this time (cancelled, refused,
    failed before clearing its outputs) left the previous run's README, and
    its PAPER COMPLETE was certified here (2026-10-24)."""
    out = []
    for a, w in views.items():
        pt_dir = w.cmp_results / "paper_tables"
        readme = pt_dir / "README.md"
        rec = {"arm": a, "pca": w.pca_label,
               "sia_results": w.rel(w.sia_results),
               "cmp_results": w.rel(w.cmp_results),
               "step1_results": w.rel(w.step1_results),
               "paper_tables": w.rel(pt_dir), "verdict": None}
        try:
            lines = readme.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            lines = None
        title = f"# Paper tables and numbers: PCA arm {a}"
        if lines is None:
            w.miss(f"{readme} not found: its paper tables did not run (or "
                   f"failed before writing it), so the arm's verdict is "
                   f"unknown")
        elif len(lines) < 3 or lines[0] != title:
            w.miss(f"{readme} is not PCA arm {a}'s paper tables (line 1 "
                   f"{(lines[0] if lines else '')!r}, not {title!r}: written "
                   f"before the PCA arms, or another arm's): its verdict is "
                   f"not read")
        elif _stale_tables(ref, w, a, pt_dir):
            w.miss(f"{readme} is not this run's: {_stale_tables(ref, w, a, pt_dir)} "
                   f"(its paper tables job wrote nothing this time): its "
                   f"verdict is not read")
        else:
            rec["verdict"] = lines[2]
            if lines[2] != VERDICT_COMPLETE:
                w.miss(f"its paper tables ({pt_dir}) say: {lines[2]}")
        out.append(rec)
    return out


def _stale_tables(ref: Ctx, w: Ctx, arm: str, pt_dir: Path) -> Optional[str]:
    """Why an arm's paper tables are not this run's (None: they may be):
    their numbers.json is missing or unreadable, names another arm, or, when
    this job has a RUN_TOKEN, carries a token that is neither this job's nor
    the arm's selection's (an arm reused from its own submission keeps its
    token: the orchestrators hand each arm's jobs that one)."""
    try:
        run = json.loads((pt_dir / "numbers.json").read_text()).get("run") or {}
    except (OSError, ValueError, AttributeError):
        return "its numbers.json is missing or unreadable"
    if run.get("pca_arm") != arm:
        return f"its numbers.json is PCA arm {run.get('pca_arm')!r}'s"
    if ref.run_token:
        ok = {ref.run_token, selection_token(w)} - {None}
        if run.get("run_token") not in ok:
            return (f"its numbers.json carries run token "
                    f"{run.get('run_token')!r}, not {' or '.join(sorted(ok))}")
    return None


PCA_ARMS_MAPPING = [
    ("tab:pca (PCA arms; not yet in paper/main.tex)", "tab_pca.csv",
     "per variant x tier x arm (pca_arm; role reference = the first of "
     "--pca_arms): modality_pca / joint_pca (its combo's PCA), "
     "single_embedding, modalities / n_modalities / "
     "same_modalities_as_reference, input_dim_mean / _min / _max and "
     "params_fold_mean (its combo's folds), auprg / ap / auroc _mean / _sd "
     "(its graded comparison row, the shared evaluator, as tab:tiers), "
     "delta_<metric> (arm - reference), status, dir"),
    ("tab:pca paired spread and reading", "tab_pca.csv",
     "delta_auprg_paired_sd (population sd) / delta_auprg_se (sample sd / "
     "sqrt(n)) / n_paired over the graded per-fold AUPRG paired by "
     "position; verdict (|delta| <= 0.005: no measurable difference; else "
     "better / worse than the reference beyond 2 SE, else within fold "
     "noise; n/a below two paired folds)"),
    ("tab:pca best arm", "tab_pca.csv / numbers.json",
     "best_arm (exact values, every tied arm True; empty unless every "
     "requested arm's cell is ok); items.pca.<variant>.<cv>.best_arm (empty "
     "on a tie) / best_arms"),
    ("PCA 2x2 effects", "numbers.json",
     "items.pca.<variant>.<cv>.effects.per_modality / joint / interaction "
     "(mean, sd, se = sample sd / sqrt(n), n over the folds all four arms' "
     "graded rows scored; empty unless all four arms are requested and "
     "ok); items.pca.<variant>.<cv>.arms.<arm> (means, deltas, paired, "
     "verdict, input_dim, params_fold_mean, modalities)"),
    ("Each arm's modality selection", "numbers.json",
     "items.pca.selection (winners per arm, same_selection groups, "
     "step1_results, run_token, selection_cv)"),
    ("PCA settings", "numbers.json",
     "items.pca.settings (this job's targets; trained: the PCA each arm's "
     "combos trained with)"),
    ("Each arm's paper tables and verdict", "README.md / numbers.json",
     "## PCA arms; arms[] (pca, sia_results, cmp_results, step1_results, "
     "paper_tables, verdict)"),
]


def render_pca_arms_readme(ref: Ctx, views: Dict[str, Ctx],
                           index: List[dict], status: str,
                           numbers: dict) -> str:
    arms = list(views)
    # The verdict on line 3, as in every per-arm README.
    L = ["# PCA arms: tab_pca and every arm's verdict", "",
         numbers["verdict"], "",
         f"Generated {numbers['generated_at']} by sl_comparison/"
         f"paper_tables.py --pca_arms_table from {ref.root}.", ""]
    L += ["## PCA arms", "",
          "Each arm is the whole pipeline under one PCA choice, in its own "
          "namespace; its paper tables are its own (their README.md's line 3 "
          "is its verdict, quoted here).", "",
          "| Arm | PCA | Siamese results | Comparison results | "
          "Single-embedding runs | Paper tables | Verdict |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['arm']} | {r['pca']} | {r['sia_results']}/ | "
          f"{r['cmp_results']}/ | {r['step1_results']}/ | "
          f"{r['paper_tables']}/ | {r['verdict'] or '(not read: listed)'} |"
          for r in index]
    L += ["", "## Tables", "", "| Table | Status |", "|---|---|",
          f"| tab_pca.csv | {status} |"]
    L += ["", "## Where each item is", "",
          "In numbers.json `items.pca` is `{paper, value, status}`, so a key "
          "path `items.pca.<rest>` here means `items.pca.value.<rest>`.", "",
          "| Item | File | Column / key |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in PCA_ARMS_MAPPING]
    t = ref.pca_targets
    drift = ("not checked" if ref.code_drift is None else
             f"{len(ref.code_drift)} file(s) (numbers.json run.code_drift)"
             if ref.code_drift else "none")
    toks = "; ".join(f"{a} {selection_token(w) or '(none recorded)'}"
                     for a, w in views.items())
    cvs = "; ".join(f"{a} {selection_cv(w) or '(none recorded)'}"
                    for a, w in views.items())
    L += ["", "## Run settings", "",
          f"- PCA arms compared: {', '.join(arms)} (--pca_arms / PCA_ARMS); "
          f"the reference arm is {arms[0]}: every delta is arm minus "
          f"{arms[0]}",
          f"- PCA targets (the environment, else siamese_sl/slurm/"
          f"config.conf, else its defaults): PCA_MODALITY_VARIANCE="
          f"{t['PCA_MODALITY_VARIANCE']}, "
          f"PCA_JOINT_VARIANCE={t['PCA_JOINT_VARIANCE']}, PCA_JOINT_DIM="
          f"{t['PCA_JOINT_DIM'] or '(unset)'}; what each arm's combos "
          f"trained with: items.pca.settings.trained",
          f"- SLeuth variants: {', '.join(ref.variants)} (rows "
          f"{', '.join(variant_row(v) for v in ref.variants)})",
          f"- RUN_TOKEN: {ref.run_token or '(unset)'}; each arm's selection "
          f"run_token: {toks} (an arm of another submission is stale)",
          f"- Selection CV (each arm's _selection_meta.selection_cv): {cvs} "
          f"(an arm whose modalities were selected on another CV than the "
          f"reference arm's is stale)",
          "- Metrics: each arm's GRADED comparison row (the shared "
          "evaluator on the shared folds, as tab:tiers), not train.py's "
          "self-scored auprg_any; paired differences over the graded "
          "per-fold AUPRG (summary.folds of the row JSONs, by position)",
          f"- Code manifest (written at submission): "
          f"{ref.code_manifest or '(none: not an orchestrated run)'}; code "
          f"changed, added or removed since: {drift}",
          "- Every sd is the population sd (ddof=0); se is the sample sd / "
          "sqrt(n)."]
    L += ["", "## Missing inputs and items", ""]
    L += [f"- {m}" for m in ref.missing] or ["- none"]
    L += ["", "## Absent by design", "",
          "Intended by the run's settings; not part of the verdict.", ""]
    L += [f"- {m}" for m in ref.absent_by_design] or ["- none"]
    if ref.notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in ref.notes]
    return "\n".join(L) + "\n"


def run_pca_arms(args) -> int:
    """--pca_arms_table: tab_pca.csv, numbers.json and README.md of the arms
    --pca_arms lists (at least two; else refused before any result is read
    or anything is written)."""
    root = Path(args.root).resolve()
    # The arms as Ctx resolves them: the environment over config.conf's.
    env = {**read_config_conf(root),
           **{k: v for k, v in os.environ.items() if v}}
    _, arms = resolve_pca_arms(args, env)
    if len(arms) < 2:
        print(f"ERROR: --pca_arms_table compares PCA arms side by side, but "
              f"--pca_arms / $PCA_ARMS lists only {' '.join(arms)}: give at "
              f"least two (e.g. --pca_arms none joint per_modality both); "
              f"nothing written", file=sys.stderr)
        return 1
    out_dir = Path(getattr(args, "out_dir", None) or
                   root / "sl_comparison" / "results" / "pca_arms")
    why = foreign_out_dir(root, out_dir, None)
    if why:
        print(f"ERROR: --out_dir {out_dir}: {why}; nothing written",
              file=sys.stderr)
        return 1
    ref = Ctx(args, arm=arms[0], cross=True)
    views = {arms[0]: ref}
    for a in arms[1:]:
        views[a] = Ctx(args, arm=a, base=ref)
    for flag in ("comparison_csv", "external_dir"):
        if getattr(args, flag, None):
            ref.note(f"--{flag} ignored: --pca_arms_table reads every arm's "
                     f"own (sl_comparison/results[/pca_<arm>]/)",
                     prefixed=False)
    if getattr(args, "regrade", False):
        ref.note("--regrade ignored: tab_pca is on the graded rows already",
                 prefixed=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(out_dir, PCA_ARMS_OUTPUT_FILES, regraded=False)

    try:
        rows, st = build_pca_arms(ref, views)
    except Exception as e:  # noqa: BLE001  (write the rest)
        rows, st = [], "missing"
        ref.miss(f"tab_pca: {type(e).__name__}: {e}", prefixed=False)
    if st == "missing":
        ref.miss("tab_pca.csv (no usable source)", prefixed=False)
    elif rows:
        write_csv(out_dir / "tab_pca.csv", rows, PCA_ARMS_COLS)
    items: Dict[str, dict] = {}
    paper = "tab:pca (not yet in paper/main.tex)"
    try:
        value = item_pca_arms(ref, views, rows)
    except Exception as e:  # noqa: BLE001
        items["pca"] = {"paper": paper, "value": None,
                        "status": f"missing: {type(e).__name__}: {e}"}
        ref.miss(f"numbers.json pca: {e}", prefixed=False)
    else:
        if is_empty(value):
            items["pca"] = {"paper": paper, "value": value,
                            "status": "missing: no input"}
            ref.miss("numbers.json pca: empty (no input)", prefixed=False)
        else:
            items["pca"] = {"paper": paper, "value": value, "status": "ok"}
    index = arms_index(ref, views)

    # Every entry is in by now.
    numbers = {
        "verdict": verdict(ref),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ref.root),
        "mode": "pca_arms_table",
        "run": {"pca_arms": arms, "reference_arm": arms[0],
                "pca_targets": ref.pca_targets,
                "variants": ref.variants,
                "selection_variant": ref.selection_variant,
                "run_token": ref.run_token,
                "code_manifest": ref.code_manifest,
                "code_drift": ref.code_drift},
        "arms": index,
        "tables": {"tab_pca": {"status": st, "rows": len(rows)}},
        "items": items,
        "missing": ref.missing,
        "absent_by_design": ref.absent_by_design,
        "notes": ref.notes,
    }
    with open(out_dir / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=2, default=str)
    (out_dir / "README.md").write_text(
        render_pca_arms_readme(ref, views, index, st, numbers))

    print(f"PCA arms ({', '.join(arms)}) -> {out_dir}")
    print(f"  {'tab_pca':16s} {st} ({len(rows)} rows)")
    for r in index:
        print(f"  arm {r['arm']:13s} {r['verdict'] or '(verdict not read)'}")
    if ref.missing:
        print(f"  {len(ref.missing)} missing input(s)/item(s); see README.md")
    if ref.absent_by_design:
        print(f"  {len(ref.absent_by_design)} absent by design; see README.md")
    if st == "missing":
        print("ERROR: table(s) not built: tab_pca")
    print(numbers["verdict"])
    if st == "missing":
        return 1
    return 0 if numbers["verdict"] == VERDICT_COMPLETE else EXIT_INCOMPLETE


# =============================================================================
# Main
# =============================================================================


def is_empty(x) -> bool:
    """None, "", an empty container, or a container holding only those
    (0 and False are values)."""
    if x is None or (isinstance(x, str) and not x):
        return True
    if isinstance(x, dict):
        return all(is_empty(v) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return all(is_empty(v) for v in x)
    return False


def verdict(ctx: Ctx) -> str:
    """VERDICT_COMPLETE with nothing listed under Missing and the folds built
    from the current sources (their STALE entry is listed under Missing too);
    else 'PAPER INCOMPLETE: <n> item(s) (<first three>; ...)'. One line. It
    certifies the tables / numbers computed here only: paper/main.tex is
    copied by hand and is not checked."""
    if not ctx.missing and not ctx.folds_stale:
        return VERDICT_COMPLETE
    items = ctx.missing or [f"shared folds are STALE: {p}"
                            for p in ctx.folds_stale]
    head = "; ".join(items[:3]) + ("; ..." if len(items) > 3 else "")
    return " ".join(f"PAPER INCOMPLETE: {len(items)} item(s) ({head})"
                    .splitlines())


def clear_outputs(out_dir: Path,
                  names: Sequence[str] = OUTPUT_FILES + LEGACY_OUTPUT_FILES,
                  regraded: bool = True) -> None:
    """Remove this script's previous outputs so a table that cannot be
    built this time is absent, not stale (a per-arm out_dir also loses a
    leftover tab_pca.csv of the superseded PCA ablation)."""
    import shutil
    for name in names:
        (out_dir / name).unlink(missing_ok=True)
    if regraded:
        shutil.rmtree(out_dir / REGRADE_DIR, ignore_errors=True)


def foreign_out_dir(root: Path, out_dir: Path,
                    arm: Optional[str]) -> Optional[str]:
    """Why out_dir holds another run's outputs of the same names (clearing it
    would delete them: every mode writes numbers.json / README.md), or None.
    A per-arm run (arm) must not write into sl_comparison/results/pca_arms/
    nor into another arm's paper_tables/; a cross-arm run (arm None) into no
    arm's paper_tables/."""
    out = Path(out_dir).resolve()
    if arm is not None and out == (root / "sl_comparison" / "results" /
                                   "pca_arms").resolve():
        return "it is --pca_arms_table's directory"
    for a in PCA_ARM_NAMES:
        if a != arm and out == (root / pca_arm_cmp(a) /
                                "paper_tables").resolve():
            return f"it is PCA arm {a}'s paper tables directory"
    return None


def run(ctx: Ctx) -> int:
    why = foreign_out_dir(ctx.root, ctx.out_dir, ctx.pca_arm)
    if why:
        raise SystemExit(f"ERROR: --out_dir {ctx.out_dir}: {why}; nothing "
                         f"written")
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(ctx.out_dir)
    # The selection variant too: a typo there would silently redirect every
    # single-modality prefix (solo_prefix) instead of failing.
    for v in solo_variants(ctx):
        if v not in ALL_VARIANTS:
            raise SystemExit(f"unknown variant {v!r}")

    built, columns = {}, {}
    for name, fn, cols in (
            ("tab_tiers", build_tiers, TIERS_COLS),
            ("tab_solo", build_solo, solo_cols(ctx)),
            ("tab_loco", build_loco, loco_cols(ctx)),
            ("tab_cellline", build_cellline, CELLLINE_COLS),
            ("tab_external", build_external, EXTERNAL_COLS),
            ("tab_external_ft", build_external_ft, EXTERNAL_FT_COLS),
            ("tab_sldb_agreement", build_sldb_agreement,
             SLDB_AGREEMENT_COLS)):
        try:
            rows, st = fn(ctx)
        except Exception as e:  # noqa: BLE001  (write every other table)
            rows, st = [], "missing"
            ctx.miss(f"{name}: {type(e).__name__}: {e}")
        built[name], columns[name] = (rows, st), cols
    if ctx.regrade:
        regrade(ctx, built)
    for name, (rows, st) in built.items():
        if st == "missing":
            ctx.miss(f"{name}.csv (no usable source)")
        elif rows:
            # Every row says which arm it is (the CSVs of two arms have the
            # same names and columns). TIERS_COLS / CELLLINE_COLS are lists,
            # the rest tuples.
            for r in rows:
                r["pca_arm"] = ctx.pca_arm
            write_csv(ctx.out_dir / f"{name}.csv", rows,
                      ["pca_arm", *columns[name]])
    status = {n: st for n, (_, st) in built.items()}

    tiers = built["tab_tiers"][0]
    items: Dict[str, dict] = {}

    def add(key, paper, fn, *a):
        try:
            value = fn(*a)
        except Exception as e:  # noqa: BLE001
            items[key] = {"paper": paper, "value": None,
                          "status": f"missing: {type(e).__name__}: {e}"}
            ctx.miss(f"numbers.json {key}: {e}")
            return
        # An item whose inputs were all absent returns empty containers, not
        # an error; it must not read as computed.
        if is_empty(value):
            items[key] = {"paper": paper, "value": value,
                          "status": "missing: no input"}
            ctx.miss(f"numbers.json {key}: empty (no input)")
        else:
            items[key] = {"paper": paper, "value": value, "status": "ok"}

    add("data_stats", "I1 (L63), E1 (L401-403)", item_data_stats, ctx)
    add("folds_meta", "E2 (L407-408, L862-866), M8 (L317-331)",
        item_folds_meta, ctx)
    # meta.json's record of a fold set not built from the current sources.
    if ctx.folds_stale and items["folds_meta"]["status"] == "ok":
        items["folds_meta"]["status"] = "stale: " + "; ".join(
            ctx.folds_stale)
    add("fold_provenance", "E2 / Reproducibility (L863-866: folds shared "
        "byte-identically across all four models)", item_fold_provenance, ctx)
    add("inputs", "R2 (L152-153), D3 (L808-811), AP1 (L891-896), AP2 "
        "(L917-922)", item_inputs, ctx)
    add("fold_geometry", "D1 (L803), D7-D10 (L820-835), TT1 (L544-547)",
        item_fold_geometry, ctx)
    add("pca_width", "R5 (L568-574), M12 (L387-394)", item_pca_width, ctx)
    add("param_ratio", "I2 (L109-111)", item_param_ratio, ctx, tiers)
    add("heads", "M7 (L306-309), M9 (L343-346)", item_heads, ctx)
    add("arch", "M3-M6, M10, D3 (L180-374)", item_arch, ctx)
    add("selection", "E3 (L408-413), M11 (L376-385)", item_selection, ctx)
    add("solo", "AB2 (L625-630), AB3 (L632-638), D4 (L812-814)",
        solo_summaries, ctx, built["tab_solo"][0])
    add("loco", "AB5 (L674-681)", item_loco, ctx, built["tab_loco"][0])
    add("cellline", "CL1 (L685, L702-704)", item_cellline,
        built["tab_cellline"][0])
    add("margins", "R4 (L557-561), D2 (L802-806)", item_margins, ctx, tiers)
    add("external", "X5 (L743-747), F2 (L776-780), F3 (L780-783)",
        item_external, ctx, built["tab_external"][0],
        built["tab_external_ft"][0])
    add("screens", "X2 (L715-716), X3 (L717-719), F4 (L783-785), D11 (L835)",
        item_screens, ctx)

    # Every ctx.miss is in by now.
    numbers = {
        "verdict": verdict(ctx),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ctx.root),
        # Which arm first: every arm writes a numbers.json of this layout.
        "run": {"pca_arm": ctx.pca_arm,
                "pca": ctx.pca_label,
                "pca_targets": ctx.pca_targets,
                # the PCA the combos trained with (None: unknown, listed)
                "pca_settings": ctx.trained_pca,
                "pca_arms": ctx.pca_arms,
                "sia_results": ctx.rel(ctx.sia_results),
                "cmp_results": ctx.rel(ctx.cmp_results),
                "shared_results": ctx.rel(ctx.shared_results),
                "step1_results": ctx.rel(ctx.step1_results),
                "step1_pca": ctx.step1_label,
                "variants": ctx.variants,
                "selection_variant": ctx.selection_variant,
                "run_token": ctx.run_token,
                "solo_run_token": ctx.solo_run_token,
                "ablation_run_token": ctx.ablation_run_token,
                "bestcat_extra_variants": ctx.extra_variants,
                "allmod_combo": ctx.allmod,
                "pca_variance_dir_tag": ctx.pca_variance,
                "ARCH_ARGS_env": ctx.arch_args_env,
                # code_drift: files changed / new / removed since the
                # manifest ([] = none), None when not checked.
                "code_manifest": ctx.code_manifest,
                "code_drift": ctx.code_drift},
        "tables": {n: {"status": st, "rows": len(built[n][0])}
                   for n, st in status.items()},
        "items": items,
        "missing": ctx.missing,
        "absent_by_design": ctx.absent_by_design,
        "notes": ctx.notes,
    }
    with open(ctx.out_dir / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=2, default=str)
    (ctx.out_dir / "README.md").write_text(render_readme(ctx, status, numbers))

    print(f"paper tables -> {ctx.out_dir}")
    for n, st in status.items():
        print(f"  {n:16s} {st} ({len(built[n][0])} rows)")
    if ctx.missing:
        print(f"  {len(ctx.missing)} missing input(s)/item(s); see README.md")
    if ctx.absent_by_design:
        print(f"  {len(ctx.absent_by_design)} absent by design; see README.md")
    bad = [n for n, st in status.items() if st == "missing"]
    if bad:
        print(f"ERROR: table(s) not built: {', '.join(bad)}")
    print(numbers["verdict"])
    if bad:
        return 1
    return 0 if numbers["verdict"] == VERDICT_COMPLETE else EXIT_INCOMPLETE


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Every paper table / number from one benchmark run, for "
        "one PCA arm (default), or the PCA arms side by side "
        "(--pca_arms_table).")
    ap.add_argument("--out_dir", default=None,
                    help="default: <root>/sl_comparison/results[/pca_<arm>]/"
                    "paper_tables (the arm's); with --pca_arms_table "
                    "<root>/sl_comparison/results/pca_arms")
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="repo root holding the run's results (default: this "
                    "checkout)")
    ap.add_argument("--pca_arm", default=None,
                    help="the PCA arm whose tables to build (none, joint, "
                    "per_modality, both; default: $PCA_ARM, else none): its "
                    "results are under siamese_sl/results[/pca_<arm>]/ and "
                    "sl_comparison/results[/pca_<arm>]/ "
                    "(siamese_sl/slurm/config.conf)")
    ap.add_argument("--pca_arms", nargs="+", default=None, metavar="ARM",
                    help="the PCA arms of the run (default: $PCA_ARMS, else "
                    "siamese_sl/slurm/config.conf's PCA_ARMS, else --pca_arm "
                    "alone; any order, read in the order none "
                    "joint per_modality both): the arms --pca_arms_table "
                    "compares, the first its reference; in per-arm mode "
                    "config.conf's owner rule over them places an arm's "
                    "single-embedding runs when its selection does not record "
                    "where they are")
    ap.add_argument("--pca_arms_table", action="store_true",
                    help="build the cross-arm table instead (tab_pca.csv, "
                    "numbers.json, README.md: each arm's graded rows, the "
                    "differences from the reference arm, best_arm, the 2x2 "
                    "effects and every arm's own paper-tables verdict); needs "
                    "at least two --pca_arms")
    ap.add_argument("--comparison_csv", default=None,
                    help="default: the arm's sl_comparison/results"
                    "[/pca_<arm>]/comparison.csv (not with --pca_arms_table: "
                    "every arm reads its own)")
    ap.add_argument("--external_dir", default=None,
                    help="default: the arm's sl_comparison/results"
                    "[/pca_<arm>]/external_transfer")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="SLeuth variants (default: $SIAMESE_VARIANTS_LIST, "
                    "else siamese_sl/slurm/config.conf's SIAMESE_VARIANTS, "
                    "else all four)")
    ap.add_argument("--selection_variant", default=None,
                    help="default: $SELECTION_VARIANT, else siamese_sl/slurm/"
                    "config.conf's SELECTION_VARIANT, else pd:per_head")
    ap.add_argument("--paper_variant", default=None,
                    help="the SLeuth variant main.tex's tab:tiers prints "
                    "(default: $PAPER_VARIANT, else pd:pooled when the run "
                    "has it, else the run's first variant); it alone, "
                    "with the peers and the bilinear probe, decides that "
                    "table's bold")
    ap.add_argument("--run_token", default=None,
                    help="this run's token (default: $RUN_TOKEN): the "
                    "expected run_token.txt of the ablation dirs, and of the "
                    "single-modality dirs when best_per_category.json records "
                    "no _selection_meta.run_token. With no run token at all "
                    "BOTH fall back to that recorded token (they are not left "
                    "unchecked); only --solo_run_token '' / "
                    "--ablation_run_token '' turn a check off")
    ap.add_argument("--solo_run_token", default=None,
                    help="expected run_token.txt of the single-modality dirs "
                    "(default: best_per_category.json _selection_meta."
                    "run_token, the Step-1 array Step 2 ranked, else the run "
                    "token; '' = unchecked)")
    ap.add_argument("--ablation_run_token", default=None,
                    help="expected run_token.txt of the ablation dirs "
                    "(default: the run token, else best_per_category.json "
                    "_selection_meta.run_token; for an ablation rerun under a "
                    "token of its own, e.g. submit_pipeline.sh --ablation-only "
                    "on a selection that recorded none; '' = unchecked). "
                    "CLI only: $ABLATION_RUN_TOKEN is ignored")
    ap.add_argument("--bestcat_extra_variants", default=None,
                    help="1 if single-modality runs were requested for the "
                    "non-selection variants (default: "
                    "$BESTCAT_EXTRA_VARIANTS, else 0)")
    ap.add_argument("--allmod_combo", default=None, choices=["0", "1"],
                    help="1: tab_tiers also lists each variant's "
                    "all-modalities row siamese_sl_allmod[_<tag>] (kind "
                    "ablation; missing ones are listed). Default: "
                    "$ALLMOD_COMBO, else config.conf, else 0")
    ap.add_argument("--pca_variance", default=None,
                    help="pcavar<value> directory tag (default: "
                    "$BEST_CAT_PCA_VARIANCE, else 0.8)")
    ap.add_argument("--regrade", action="store_true",
                    help="also grade every fresh tab_solo / tab_loco cell "
                    "whose fold matrices were kept with evaluate_model.py "
                    "(-> <out_dir>/regraded/, column auprg_shared_mean). "
                    "Slow: every fold's dense matrix is read")
    ap.add_argument("--code_manifest", default=None, metavar="PATH",
                    help="code_manifest.py's manifest of the code the run's "
                    "jobs execute, written at submission (default: "
                    "$CODE_MANIFEST, which the orchestrator exports; '' = "
                    "none): code changed, added or removed since is listed "
                    "under Missing")
    args = ap.parse_args(argv)
    if args.pca_arms_table:
        return run_pca_arms(args)
    return run(Ctx(args))


if __name__ == "__main__":
    sys.exit(main())
