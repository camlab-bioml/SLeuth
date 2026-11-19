#!/usr/bin/env python3
"""The SLURM tree's cross-job contracts, checked by sourcing the .conf files.

  - RUN_TOKEN (slurm/config.conf): kept when the orchestrator already
    exported one, so every job of a submission shares it; made fresh
    (YYYYmmddTHHMMSS-PID) and exported otherwise. README's manual Step-1
    block exports one before its first sbatch, so hand-submitted stages
    share it too, and its Step-2a/2b blocks submit each array's afterany
    follow-up (benchmark summary; selection + combos).
  - bestcat_n_tasks (slurm/run_best_per_category.conf): |EMB_CATALOG| x
    |CV_TYPES| on SELECTION_VARIANT by default; BESTCAT_EXTRA_VARIANTS=1 adds
    one block per other SIAMESE_VARIANTS entry (reporting only), with
    SELECTION_VARIANT always block 0 and the only unprefixed one.
  - run_generate_embeddings.sh ends with embedding_versions.py (after the
    catalog-existence check) and fails when it does.
  - submit_pipeline.sh: every afterok edge carries --kill-on-invalid-dep=yes
    (a failed parent cancels its child instead of leaving it — and every
    afterany consumer — pending forever), and one RUN_TOKEN per submission.
  - the PCA arms (config.conf PCA_ARMS): both orchestrators run every arm the
    list names, the shared jobs once, one single-embedding array (and in
    submit_pipeline.sh one benchmark) per OWNER (pca_step1_arm), every other
    stage per arm, each arm's job with PCA_ARM=<arm> in sbatch's own
    environment and a plain --export=ALL (no submission, and no command
    either prints, passes an --export list: Slurm reads one as an implied
    --get-user-env) and the arm in its log names, under the usual job
    names; the root's shared folds job and array right after the
    generation, before any siamese job (FIFO-like schedulers ran them last);
    the root's table across the
    arms (run_paper_tables.sh --pca_arms_table) once, with 2 arms or more;
    every export last. The removal of previous outputs stays in each listed
    arm's roots (rows into that root's own stale_siamese/), the shared rows
    once; a regeneration drops every arm's combo markers, and a root run
    that retrains the shared models the tables of every arm it leaves out
    (comparison.csv + notes, external_transfer/, paper_tables/: built on
    those rows). Such a run names every left-out arm without tables by its
    state (also after a submission that stopped part-way; never an arm it
    runs) and prints, last, the jobs that rebuild them wired to its own,
    free of the pasting shell's RUN_TOKEN / CODE_MANIFEST / compare pins
    (_paste_env, the block's first line), with this run's PCA targets and
    a left-out arm's own where its combos were trained at others (or, for an
    arm without a combo marker, the command that retrains it); every such
    command carries the settings the run took from the calling shell. A
    best-per-category submit_pipeline.sh run ends with its sl_comparison
    jobs, wired the same way (cell-line ablation, compare, external transfer,
    paper tables, the table across the arms). Both banners count the jobs
    and array tasks they queue, as submitted. A pair partner
    left out whose selection ranked the single-embedding runs a run rebuilds
    is warned, before anything changes. An inherited PCA_ARM is ignored, an
    inherited job id (<NAME> or <NAME>_<arm>) never reaches a dependency,
    POST_PCA_* and a bad arm setting are refused before anything changes,
    and the root asks train.py --print_arch once per distinct PCA flag set
    of its arms.
  - submit_pipeline.sh --ablation-only joins each ablated arm's token (its
    best_per_category.json _selection_meta.run_token) and removes the
    previous <prefix>ablate_no_* dirs first; with no selection token it keeps
    a fresh one and warns; arms of different tokens get one each in
    sbatch's environment (PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ...), as
    the root's --skip-siamese does for the arms it reuses.
    An arm with no selection, or no combo marker at all, is refused before
    anything is removed (with the command that trains it, embeddings kept);
    a (variant, CV) without a marker is named (its tasks SKIP); so is an arm
    whose marker-named combos were trained with another PCA than this shell
    gives it (arch_signature.py --check_pca), with what to submit instead,
    each command complete and alone on its line (the settings submit_pipeline
    took from the calling shell, recorded under the root's name list, the
    three targets in full): ONE when one set of targets fits every arm, else
    one per arm or variant group, which run as printed (bash, zsh) one after
    the other (a later one with ALLOW_LIVE_JOBS=1), plus the rebuild of a
    split arm's ablation_summary.csv. The root's --skip-siamese checks its
    reused arms the same way, over every variant and CV (refused when one
    set of targets fits them all, else a WARNING), and names an arm with
    nothing on disk in one NOTE, under the first reused arm's token.
  - every command either orchestrator prints to run after it, but the lines
    of a pasted block, carries the settings it took from the calling shell;
    a block's lines get, through _paste_env, the settings their jobs read:
    in submit_pipeline.sh's sl_comparison block this run's arms, targets,
    variants and USE_CELL_LINES, and the selection variant, Step-1 tag,
    BESTCAT_EXTRA_VARIANTS, ARCH_ARGS (printf %q) and CVS as it resolved
    them; in the root's rebuild of the arms it leaves out, its arms,
    targets, variants, USE_CELL_LINES and CVS. Every orchestrator command
    either orchestrator prints, the tab:loco hint and the ablation_summary
    rebuild also pin the arms, the PCA targets, USE_CELL_LINES and (but the
    --benchmark-only remedy, which trains the selection variant alone) the
    variants as the run resolved them; a refused export's sbatch line pins
    the variants and USE_CELL_LINES. A refused export's command (root,
    submit_pipeline.sh, --eval-only) carries the PAIR_EXPORT_* and variants,
    each pasted into a fresh bash or zsh as its arm's own; so do the
    --benchmark-only remedy, the tab:loco token hint, the venv rebuilds
    (TORCH_SPEC) and the --skip-siamese submission that keeps one Step-1
    token, printed after the scancel line of a run that queued its jobs, and
    with no ALLOW_LIVE_JOBS: it is pasted once those jobs are gone (each
    command alone on its line, with no trailing "#").
  - a best-cat submission removes the replaced combos' cellline_ablation dirs
    and moves their sl_comparison siamese rows (graded against the combos this
    submission replaces) into stale_siamese/, leaving other models' rows.
  - the all-gene-pair export (slurm/export_pair_predictions.sh) is the LAST
    job of both orchestrators, afterany on the others (never afterok: nothing
    reads its output, and a failed parent must not cancel it); with no eval
    it is told to skip the fine-tuned sets; --skip-pair-export omits it;
    --ablation-only and --benchmark-only never submit it; --eval-only chains
    it on the one evaluated combo; every live-job list names its job. Every
    submission of it is also `singleton` (two exports of one combo clear each
    other's files, and its own #SBATCH header says so for a direct sbatch),
    a refused submission is a WARNING that prints the command to run, and
    a mistyped PAIR_EXPORT_* value (dataset and mode names apart: only the
    export job can match those) is refused before anything is queued.
  - the export job itself (bash + export_pair_predictions.conf), run in the
    stub tree against a logging $PYTHON_PATH: by default each variant's cv3
    combo on CUDA, all genes, with the cv1-split Adamson fine-tuned models;
    exit 0 only when every requested combo was exported in full.
  - the eval job without --array runs every cv_type even after one whose
    eval_finetuning.py exits 1 (tables written, a screen or an optional output
    missing), and still ends non-zero.
  - EMB_CATALOG minus OPTIONAL_TYPES == paper_tables.PAPER_SOLO, and
    generation builds no catalog member outside it but scprint / seqvec.
  - run_ablation.sh's header exports one RUN_TOKEN before its sbatch example,
    and its TRAIN_ARGS are the combo's Step-3 call, entry for entry.
  - every submission in both orchestrators passes $GPU_NODE_ARGS and sizes
    its --array through array_throttle; config.conf's check_throttles /
    check_node_selection refuse a non-numeric cap and a GPU_EXCLUDE that
    cannot work (without GPU_NODES, or overlapping it), and submit_pipeline.sh
    stops on them before anything is removed or queued.
  - no job script executes a repo script directly (an uploaded copy can lack
    +x), and the orchestrator's step 0 makes sl_comparison/folds writable
    before prepare_folds.py and read-only after it, before the first job.
  - before the first job the orchestrator also writes and exports the code
    manifest (sl_comparison/code_manifest.py; paper_tables re-checks it) and
    makes data/synlethkg read-only; the manifest tool's scope and drift.

The .conf files are sourced (bash, SLURM_JOB_ID unset, so config.conf's
compute-node header is skipped); the tests that drive documented shell blocks
or submit_pipeline.sh do so in a tempdir against stub sbatch/squeue, so no
SLURM job is ever submitted and nothing outside the tempdir is written.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_slurm_contracts.py -q
    $PYTHON_PATH siamese_sl/tests/test_slurm_contracts.py
"""
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
# A pipeline switch still exported by the calling shell (a mistyped
# SELECTION_CV of a submission the orchestrator has just refused, the
# USE_CELL_LINES=0 of an earlier experiment) must not decide these tests:
# run_best_per_category.conf and config.conf honour them wherever they are
# sourced. A test that means one passes it to its subprocess explicitly.
for _v in ("SELECTION_CV", "USE_CELL_LINES"):
    os.environ.pop(_v, None)


_SIAMESE = os.path.dirname(_HERE)


_ROOT = os.path.dirname(_SIAMESE)


# The pipeline defaults these contracts were written under. On 2026-11-13
# config.conf / run_best_per_category.conf took the paper's protocol as their
# defaults: PCA_ARMS "none joint", the variants "pd:pooled pd:per_head" with
# pd:pooled selecting, BESTCAT_EXTRA_VARIANTS=1, SELECTION_FRAC=0.1
# (per-fold selection) and ALLMOD_COMBO=1. What most tests check (edges,
# tokens, exports, refusals, cleanup, printed commands) does not depend on
# them, and they keep checking it on the four arms and the 2x2 of variants,
# selected globally on pd:per_head without a selection split or the
# all-modalities combo: _bash sets these in its environment, and the
# orchestrator trees (_stub_tree, _root_tree) get them as their copy's
# config.conf defaults (_pin_old_defaults), NOT in the orchestrators'
# environment, where each would be a setting "taken from the calling shell"
# that every printed command repeats. A test passes its own value to
# override one; _defaults=True (_bash, the trees) keeps the shipped defaults,
# which the test_default_* tests check with the DAG they queue.
_VARIANTS_2X2 = "pd:per_head pd:pooled bilinear:per_head bilinear:pooled"
_PINNED = {"SIAMESE_VARIANTS_LIST": _VARIANTS_2X2,
           "SELECTION_VARIANT": "pd:per_head", "BESTCAT_EXTRA_VARIANTS": "0",
           "SELECTION_FRAC": "", "ALLMOD_COMBO": "0"}
_PINNED_NAMES = tuple(_PINNED) + ("SIAMESE_VARIANTS", "SELECTION_SEED",
                                  "SELECTION_MODE")
# (pattern, replacement) per default line of the copy's confs; each must
# match exactly once, so a moved default fails here, loudly.
_OLD_DEFAULT_LINES = {
    "config.conf": (
        (r'(?m)^PCA_ARMS="\$\{PCA_ARMS:-none joint\}"',
         'PCA_ARMS="${PCA_ARMS:-none joint per_modality both}"'),
        (r'(?m)^SIAMESE_VARIANTS_LIST="\$\{SIAMESE_VARIANTS_LIST:-'
         r'\$\{SIAMESE_VARIANTS\[\*\]:-pd:pooled pd:per_head\}\}"',
         'SIAMESE_VARIANTS_LIST="${SIAMESE_VARIANTS_LIST:-'
         '${SIAMESE_VARIANTS[*]:-' + _VARIANTS_2X2 + '}}"'),
        (r'(?m)^SELECTION_VARIANT="\$\{SELECTION_VARIANT:-pd:pooled\}"',
         'SELECTION_VARIANT="${SELECTION_VARIANT:-pd:per_head}"'),
        (r'(?m)^SELECTION_FRAC="\$\{SELECTION_FRAC-0\.1\}"',
         'SELECTION_FRAC="${SELECTION_FRAC-}"'),
        (r'(?m)^ALLMOD_COMBO="\$\{ALLMOD_COMBO:-1\}"',
         'ALLMOD_COMBO="${ALLMOD_COMBO:-0}"')),
    "run_best_per_category.conf": (
        (r'(?m)^BESTCAT_EXTRA_VARIANTS="\$\{BESTCAT_EXTRA_VARIANTS:-1\}"',
         'BESTCAT_EXTRA_VARIANTS="${BESTCAT_EXTRA_VARIANTS:-0}"'),)}


def _pin_old_defaults(slurm_dir):
    """Rewrite the default lines of a COPY's config.conf and
    run_best_per_category.conf to the settings these contracts were written
    under (_PINNED, all four PCA arms)."""
    assert os.path.realpath(slurm_dir) != os.path.realpath(
        os.path.join(_SIAMESE, "slurm")), slurm_dir
    for name, subs in _OLD_DEFAULT_LINES.items():
        path = os.path.join(slurm_dir, name)
        with open(path) as f:
            text = f.read()
        for pat, rep in subs:
            text, n = re.subn(pat, lambda _m, r=rep: r, text)
            assert n == 1, (name, pat, n)
        with open(path, "w") as f:
            f.write(text)


def _bash(script, _defaults=False, **env):
    """Run `script` after sourcing both confs; returns stdout lines."""
    e = {k: v for k, v in os.environ.items()
         if k not in ("SLURM_JOB_ID", "RUN_TOKEN") + _PINNED_NAMES}
    if not _defaults:
        e.update(_PINNED)
    e.update(env)
    e = {k: v for k, v in e.items() if v is not None}
    pre = ("source slurm/config.conf >/dev/null 2>&1 || exit 3; "
           "source slurm/run_best_per_category.conf >/dev/null 2>&1 "
           "|| exit 4; ")
    out = subprocess.run(["bash", "-c", pre + script], cwd=_SIAMESE, env=e,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
    return out.stdout.split("\n")[:-1]


SIZES = ('echo "${#EMB_CATALOG[@]} ${#CV_TYPES[@]} ${#SIAMESE_VARIANTS[@]}"; '
         'bestcat_n_tasks; bestcat_task_variants; '
         'for v in "${BESTCAT_TASK_VARIANTS[@]}"; do '
         'echo "$v=$(bestcat_solo_prefix "$v")"; done')


def _sizes(_defaults=False, **env):
    lines = _bash(SIZES, _defaults, **env)
    n_cat, n_cv, n_var = map(int, lines[0].split())
    blocks = [ln.split("=", 1) for ln in lines[2:]]
    return n_cat, n_cv, n_var, int(lines[1]), blocks


def test_run_token_kept_when_exported_else_fresh():
    assert _bash('echo "$RUN_TOKEN"', RUN_TOKEN="tok-123") == ["tok-123"]
    tok = _bash('echo "$RUN_TOKEN"')[0]
    assert re.fullmatch(r"\d{8}T\d{6}-\d+", tok), tok
    # A token config.conf mints itself is exported too (none inherited, so
    # only its own `export` can reach the child): an sbatch --export=ALL job
    # or a python child sees the same value.
    parent, child = _bash('echo "$RUN_TOKEN"; bash -c \'echo "$RUN_TOKEN"\'')
    assert parent == child, (parent, child)
    assert re.fullmatch(r"\d{8}T\d{6}-\d+", child), child


def _readme_block(heading):
    """The first ```bash block after `heading` in siamese_sl/README.md."""
    with open(os.path.join(_SIAMESE, "README.md")) as f:
        text = f.read()
    return text.split(heading, 1)[1].split("```bash", 1)[1].split("```", 1)[0]


def test_readme_manual_stages_export_one_token_first():
    # README's hand-submitted stages bypass the orchestrators; without an
    # exported token each job (each array task) mints its own and combo
    # Step 2 refuses every candidate as another submission's.
    block = _readme_block("### Step 1")
    lines = [ln.strip() for ln in block.splitlines()]
    first_sbatch = next(i for i, ln in enumerate(lines) if "sbatch" in ln
                        and not ln.startswith("#"))
    exports = [ln for ln in lines[:first_sbatch]
               if ln.startswith("export RUN_TOKEN=")]
    assert len(exports) == 1, lines[:first_sbatch]
    # the exported value is what a job's config.conf then keeps
    got = _bash(exports[0] + '; bash -c \'source slurm/config.conf '
                '>/dev/null 2>&1; echo "$RUN_TOKEN"\'; echo "$RUN_TOKEN"')
    assert len(got) == 2 and got[0] == got[1], got
    assert re.fullmatch(r"\d{8}T\d{6}-\d+", got[0]), got


def test_readme_manual_steps_chain_their_follow_up_jobs():
    # The arrays never submit their follow-ups: selection + combos (and the
    # benchmark table) are separate afterany jobs, as in submit_pipeline.sh.
    # Run README's Step 1/2a/2b blocks as written against a logging sbatch.
    script = "\n".join(_readme_block(h) for h in
                       ("### Step 1", "### Step 2a", "### Step 2b"))
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "sbatch.log")
        stub = os.path.join(tmp, "sbatch")
        with open(stub, "w") as f:
            # prints job id 1000 + call index (0-based) like --parsable
            f.write('#!/bin/bash\necho "$*" >> "$SBATCH_LOG"\n'
                    'echo $(( 999 + $(wc -l < "$SBATCH_LOG") ))\n')
        os.chmod(stub, 0o755)
        e = {k: v for k, v in os.environ.items()
             if k not in ("SLURM_JOB_ID", "RUN_TOKEN",
                          "BESTCAT_EXTRA_VARIANTS", "SIAMESE_VARIANTS_LIST",
                          "SIAMESE_VARIANTS", "SELECTION_VARIANT")}
        e.update(PATH=tmp + os.pathsep + e.get("PATH", ""), SBATCH_LOG=log)
        out = subprocess.run(["bash", "-e", "-c", script],
                             cwd=os.path.dirname(_SIAMESE), env=e,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
        with open(log) as f:
            calls = f.read().splitlines()

    def job(script_name):
        idx = [i for i, c in enumerate(calls)
               if c.split()[-1] == f"slurm/{script_name}"]
        assert len(idx) == 1, (script_name, calls)
        return idx[0], calls[idx[0]]

    i_gen, _ = job("run_generate_embeddings.sh")
    for array, follow_up in (
            ("run_embedding_benchmark.sh",
             "run_embedding_benchmark_summary.sh"),
            ("run_best_per_category.sh", "run_best_per_category_combo.sh")):
        i_arr, arr = job(array)
        _, nxt = job(follow_up)
        assert f"--dependency=afterok:{1000 + i_gen}" in arr, arr
        assert "--parsable" in arr, arr
        assert f"--dependency=afterany:{1000 + i_arr}" in nxt, nxt


def test_generation_job_ends_with_the_provenance_gate():
    # A failed builder falls back to a kept cache file and stamps the .pt
    # non-current; the job must then fail (existence alone passes it), with
    # the same embedding_versions.py check the root orchestrator's gate runs.
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_generate_embeddings.sh")) as f:
        text = f.read()
    i_missing = text.index("REQUIRED_MISSING[@]} -gt 0")
    i_gate = text.index("$SLURM_SUBMIT_DIR/embedding_versions.py")
    assert i_gate > i_missing, "provenance check must follow the catalog check"
    block = text[text.rindex("\n#", 0, i_gate):]
    with tempfile.TemporaryDirectory() as tmp:
        stub = os.path.join(tmp, "py")
        with open(stub, "w") as f:
            f.write('#!/bin/bash\necho "$*" >> "$LOG"\nexit "$RC"\n')
        os.chmod(stub, 0o755)
        log = os.path.join(tmp, "log")
        for rc, want in (("0", 0), ("1", 1)):
            env = dict(os.environ, PYTHON_PATH=stub, SLURM_SUBMIT_DIR=_SIAMESE,
                       DATA_DIR="/d", GENE_UNIVERSE="/d/gene_universe.txt",
                       LOG=log, RC=rc)
            out = subprocess.run(["bash", "-e", "-c", block], env=env,
                                 capture_output=True, text=True, timeout=60)
            assert out.returncode == want, (rc, out.stdout, out.stderr)
        with open(log) as f:
            call = f.readline().split()
        assert call[0].endswith("siamese_sl/embedding_versions.py"), call
        assert call[1:] == ["--data_dir", "/d", "--gene_universe",
                            "/d/gene_universe.txt"], call


def test_generation_retry_rebuilds_what_a_cut_short_attempt_left():
    # STEP 3's one retry must not reuse what a failed attempt left behind: an
    # unstamped (possibly truncated) builder pkl is dropped, a pkl with its
    # build stamp (a finished build of this job) is kept, a half-written
    # output never stays, and a removal that fails does not end the job
    # (set -e is on in the loop).
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_generate_embeddings.sh")) as f:
        text = f.read()
    a = text.index('for etype in "${PRECOMPUTED_TYPES[@]}"; do')
    loop = text[a:text.index("\ndone\n", a) + len("\ndone\n")]
    a = text.index("CLEANED=0\n")
    clean = text[a:text.index("\ndone\n", a) + len("\ndone\n")]
    stub = r"""#!/bin/bash
# stand-in for generate_embeddings.py: what each attempt finds and leaves
while [ $# -gt 0 ]; do
    case "$1" in --type) T="$2" ;; --output) O="$2" ;; --cache_dir) C="$2" ;; esac
    shift
done
PKL="$C/${T}_gene_embeddings.pkl"
N=$(( $(cat "$C/$T.n" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$C/$T.n"
has() { [ -f "$1" ] && echo y || echo n; }
echo "$T $N pkl=$(has "$PKL") stamp=$(has "$PKL.generator_version") part=$(has "$O.part")" >> "$C/log"
case "$T:$N" in
    cutpkl:1|rofs:1) echo half > "$PKL"; exit 1 ;;          # died in pickle.dump
    cutsave:1) echo whole > "$PKL"; : > "$PKL.generator_version"
               echo half > "$O.part"; exit 1 ;;              # died in torch.save
    dead:*|opt:*) echo half > "$O.part"; exit 1 ;;
esac
[ "$(cat "$PKL" 2>/dev/null)" != half ] || exit 1           # a cut pkl cannot load
echo whole > "$PKL"; : > "$PKL.generator_version"; echo pt > "$O"
"""
    with tempfile.TemporaryDirectory() as tmp:
        data, cache = os.path.join(tmp, "data"), os.path.join(tmp, "cache")
        os.makedirs(data)
        os.makedirs(cache)
        py = os.path.join(tmp, "py")
        with open(py, "w") as f:
            f.write(stub)
        os.chmod(py, 0o755)
        pre = ('PRECOMPUTED_TYPES=(cutpkl cutsave dead opt rofs last)\n'
               'OPTIONAL_TYPES=(opt)\nsleep() { :; }\n'
               # a cache directory that turned read-only: removals fail
               'rm() { case "$*" in *rofs*) return 1 ;; esac; '
               'command rm "$@"; }\n')
        env = dict(os.environ, PYTHON_PATH=py, DATA_DIR=data, CACHE_DIR=cache,
                   ESM_PATH=os.path.join(data, "genes.txt"))
        r = subprocess.run(["bash", "-e", "-c",
                            pre + loop + 'echo "LOOP DONE"\n'], cwd=tmp,
                           env=env, capture_output=True, text=True,
                           timeout=60)
        assert r.returncode == 0 and "LOOP DONE" in r.stdout, \
            (r.returncode, r.stdout, r.stderr)
        with open(os.path.join(cache, "log")) as f:
            log = f.read().splitlines()
        assert log == [
            "cutpkl 1 pkl=n stamp=n part=n",
            "cutpkl 2 pkl=n stamp=n part=n",      # the cut pkl was dropped
            "cutsave 1 pkl=n stamp=n part=n",
            "cutsave 2 pkl=y stamp=y part=n",     # the finished build kept
            "dead 1 pkl=n stamp=n part=n",
            "dead 2 pkl=n stamp=n part=n",
            "opt 1 pkl=n stamp=n part=n",         # optional: one attempt
            "rofs 1 pkl=n stamp=n part=n",
            "rofs 2 pkl=y stamp=n part=n",        # could not be removed
            "last 1 pkl=n stamp=n part=n",        # ... and the job went on
        ], log
        for t, done in (("cutpkl", True), ("cutsave", True), ("dead", False),
                        ("opt", False), ("rofs", False), ("last", True)):
            assert os.path.isfile(
                os.path.join(data, f"all_genes_{t}.pt")) == done, t
            assert (f"[{t}] Done" if done
                    else f"[{t}] Skipped (not available)") in r.stdout, t
        assert [n for n in os.listdir(data) if n.endswith(".part")] == [], \
            os.listdir(data)
        # the job's start-up cleaning drops what a killed save left behind
        for n in ("all_genes_x.pt", "all_genes_x.pt.part",
                  "all_genes_y.pt.part", "other.pt.part"):
            with open(os.path.join(data, n), "w") as f:
                f.write("old")
        r = subprocess.run(
            ["bash", "-e", "-c",
             'ALL_EMBEDDINGS=("x:all_genes_x.pt:X" "y:all_genes_y.pt:Y")\n'
             + clean + 'echo "CLEANED=$CLEANED"\n'], cwd=tmp, env=env,
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "CLEANED=1" in r.stdout, \
            (r.stdout, r.stderr)
        left = sorted(n for n in os.listdir(data) if "_x" in n or "_y" in n
                      or n.startswith("other"))
        assert left == ["other.pt.part"], left


def test_bestcat_n_tasks_without_extra_variants_is_selection_variant_only():
    n_cat, n_cv, _, n, blocks = _sizes(BESTCAT_EXTRA_VARIANTS="0")
    assert n_cat > 0 and n_cv > 0
    assert n == n_cat * n_cv
    assert len(blocks) == 1 and blocks[0][1] == ""


def test_default_bestcat_array_is_both_variants_blocks():
    # The paper's protocol (2026-11-13): BESTCAT_EXTRA_VARIANTS=1 on the two
    # variants, pd:pooled (selecting, block 0, unprefixed) then pd:per_head
    # (reporting only, under its own prefix): 19 catalog embeddings x 3 CVs
    # x 2 blocks = 114 tasks.
    n_cat, n_cv, n_var, n, blocks = _sizes(_defaults=True)
    assert (n_cat, n_cv, n_var, n) == (19, 3, 2, 114), (n_cat, n_cv, n_var, n)
    assert blocks == [["pd:pooled", ""], ["pd:per_head", "var_pd_per_head__"]]
    assert _bash('echo "$SELECTION_VARIANT|$BESTCAT_EXTRA_VARIANTS"',
                 _defaults=True) == ["pd:pooled|1"]


def test_bestcat_extra_variants_add_one_block_per_other_variant():
    n_cat, n_cv, n_var, n, blocks = _sizes(BESTCAT_EXTRA_VARIANTS="1")
    assert n == n_cat * n_cv * n_var == n_cat * n_cv * len(blocks)
    sel = _bash('echo "$SELECTION_VARIANT"')[0]
    assert blocks[0] == [sel, ""]
    prefixes = [p for _, p in blocks[1:]]
    assert all(p.startswith("var_") and p.endswith("__") for p in prefixes)
    assert len(set(prefixes)) == len(prefixes)
    # a non-default selection variant: pd:per_head gets a prefix of its own
    # so its reporting runs cannot overwrite the ranked selection runs
    _, _, _, _, blocks = _sizes(BESTCAT_EXTRA_VARIANTS="1",
                                SELECTION_VARIANT="bilinear:pooled")
    assert blocks[0] == ["bilinear:pooled", ""]
    assert ["pd:per_head", "var_pd_per_head__"] in blocks
    # one variant: nothing extra to train
    n_cat, n_cv, _, n, _ = _sizes(BESTCAT_EXTRA_VARIANTS="1",
                                  SIAMESE_VARIANTS_LIST="pd:per_head")
    assert n == n_cat * n_cv


def _stub_tree(tmp, _defaults=False):
    """A siamese_sl/slurm copy plus the files submit_pipeline.sh requires,
    its confs' defaults those of _pin_old_defaults (the shipped ones with
    _defaults=True).

    sbatch logs "<id>|<RUN_TOKEN>|<args>" and prints an incrementing id, and
    logs "<id>|<PCA_ARM>" (<unset> when its environment has none) to
    <log>.arm: the arm the job gets, which rides in sbatch's environment
    (--export=ALL), never on its command line. squeue reports an empty queue
    (submit_pipeline's _check_no_live_jobs needs BOTH on PATH). Everything
    the script writes stays under `tmp`.
    """
    sia = os.path.join(tmp, "siamese_sl")
    shutil.copytree(os.path.join(_SIAMESE, "slurm"),
                    os.path.join(sia, "slurm"),
                    ignore=shutil.ignore_patterns("logs"))
    if not _defaults:
        _pin_old_defaults(os.path.join(sia, "slurm"))
    for f in ("arch_signature.py", "embedding_versions.py",
              "export_pair_predictions.py"):
        open(os.path.join(sia, f), "w").close()
    os.makedirs(os.path.join(tmp, "sl_comparison", "results"))
    for f in ("model_params.py", "auprg.py"):
        open(os.path.join(tmp, "sl_comparison", f), "w").close()
    os.makedirs(os.path.join(sia, "results"))
    b = os.path.join(tmp, "bin")
    os.makedirs(b)
    with open(os.path.join(b, "sbatch"), "w") as f:
        # --test-only (the submitters' pre-flight probe) is logged apart and
        # queues nothing; SB_TEST_FAIL=1 makes it refuse like slurmctld would.
        f.write('#!/bin/bash\n'
                'if [ "$1" = "--test-only" ]; then\n'
                '  echo "$*" >> "${SB_LOG}.test"\n'
                '  if [ "${SB_TEST_FAIL:-0}" = 1 ]; then\n'
                '    echo "sbatch: error: Requested node configuration is not'
                ' available" >&2; exit 1; fi\n'
                '  echo "sbatch: Job 1 to start at now using 12 processors on'
                ' nodes gpu2 in partition p" >&2; exit 0\n'
                'fi\n'
                # SB_PAIR_FAIL=1: slurmctld refuses the pair export alone (a
                # QOS submit limit reached by the jobs queued before it, say)
                'case "$*" in *export_pair_predictions.sh)\n'
                '  if [ "${SB_PAIR_FAIL:-0}" = 1 ]; then\n'
                '    echo "sbatch: error: Batch job submission failed" >&2\n'
                '    exit 1; fi ;;\n'
                'esac\n'
                # SB_FAIL_ON=<script>: slurmctld refuses that submission
                'if [ -n "${SB_FAIL_ON:-}" ]; then case "$*" in\n'
                '  *"$SB_FAIL_ON") echo "sbatch: error: Batch job '
                'submission failed" >&2; exit 1 ;; esac; fi\n'
                'n=$(( $(cat "$SB_CTR" 2>/dev/null || echo 999) + 1 ))\n'
                'echo $n > "$SB_CTR"\n'
                'echo "$n|$RUN_TOKEN|$*" >> "$SB_LOG"\n'
                'echo "$n|${PCA_ARM-<unset>}" >> "${SB_LOG}.arm"\necho $n\n'
                'echo "$n|${PAIR_EXPORT_MODEL_DIR:-}|${PAIR_EXPORT_FT_SPLITS:-}"'
                ' >> "${SB_LOG}.pairenv"\n'
                # the selection settings and ALLMOD_COMBO the job inherits
                'echo "$n|${SELECTION_FRAC-<unset>}|${SELECTION_SEED-<unset>}|'
                '${SELECTION_MODE-<unset>}|${ALLMOD_COMBO-<unset>}|'
                '${SELECTION_VARIANT-<unset>}|'
                '${BESTCAT_EXTRA_VARIANTS-<unset>}" >> "${SB_LOG}.sel"\n')
    with open(os.path.join(b, "squeue"), "w") as f:
        f.write('#!/bin/bash\nexit 0\n')  # no job of an earlier submission
    for f in ("sbatch", "squeue"):
        os.chmod(os.path.join(b, f), 0o755)
    return sia, b


_ARMS = ("none", "joint", "per_modality", "both")


# The jobs of every arm, in either orchestrator: submitted with no PCA_ARM of
# their own (each arm's other jobs with theirs, _as_arm).
_SHARED_SCRIPTS = ("reset_env.sh", "run_generate_embeddings.sh",
                   "run_shared_folds.sh", "run_shared_models.sh",
                   "run_elisl.sh", "build_elisl_inputs.sh")
_ELISL_JOBS = ("elisl", "elisl_inputs")


def _plain_export(args):
    """Every submission passes exactly one --export, and it is plain ALL:
    the arm (and an arm's own RUN_TOKEN) ride in sbatch's environment, as
    PCA_ARM=<arm> in front of it (_as_arm), never in an --export list, which
    Slurm takes as an implied --get-user-env (2026-10-21). Returns args."""
    exp = [a for a in args.split() if a.startswith("--export")]
    assert exp == ["--export=ALL"], args
    return args


class _Job(tuple):
    """One stub sbatch call of _submit, (job id, RUN_TOKEN, arguments) as the
    stub logs them, and .arm: the PCA_ARM of the environment it was
    submitted with, which the job gets ("<unset>" for none)."""

    def __new__(cls, jid, tok, args, arm):
        self = super().__new__(cls, (jid, tok, args))
        self.arm = arm
        return self


# What submit_pipeline.sh takes from the calling shell (its recorded names,
# 2026-10-21: named in a NOTE and carried by every command it prints, so one
# exported by the shell running these tests would change their output), and
# the other switches its jobs honour: none reaches a run of it here unless a
# test passes it.
_CALLER_PREFIXES = ("SLURM_", "PAIR_EXPORT_", "SB_", "PCA_", "POST_PCA_",
                    "SIAMESE_", "BESTCAT_", "BEST_CAT_", "ABLATION_", "TORCH_",
                    "GPU_", "EVAL_")
_CALLER_NAMES = ("RUN_TOKEN", "SELECTION_CV", "SELECTION_VARIANT",
                 "SELECTION_FRAC", "SELECTION_SEED", "SELECTION_MODE",
                 "ALLMOD_COMBO", "ARCH_ARGS",
                 "CVS", "SEED", "ALLOW_LIVE_JOBS", "MAX_CONCURRENT",
                 "SHARED_CONCURRENT", "PREPROCESSING_FIT_SCOPE",
                 "CELL_LINE_DIM", "USE_CELL_LINES", "SHARED_FOLDS",
                 "SQUEUE_OUT")


def _clean_env(**env):
    """This process's environment without any setting of _CALLER_PREFIXES /
    _CALLER_NAMES, plus `env`: the shell a command runs or is pasted in."""
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(_CALLER_PREFIXES) and k not in _CALLER_NAMES}
    e.update(env)
    return e


def _submit(tmp, *args, _rc=0, _defaults=False, **env):
    """Run submit_pipeline.sh in the stub tree; returns its sbatch calls
    (_Job, in submission order), each checked to pass a plain --export=ALL.
    `_rc`: the exit status it must end with. PCA_ARMS defaults to "none":
    the single-arm contracts below are about one arm (every arm runs the
    same code); a test of the arms passes its own, None for the tree's
    config.conf default (all four, _pin_old_defaults). _defaults=True sets
    no PCA_ARMS either (for a tree built with _defaults=True: the shipped
    defaults)."""
    sia, b = os.path.join(tmp, "siamese_sl"), os.path.join(tmp, "bin")
    e = _clean_env()
    if not _defaults:
        env.setdefault("PCA_ARMS", "none")
    e.update(PATH=b + os.pathsep + e["PATH"],
             SB_LOG=os.path.join(tmp, "sb.log"),
             SB_CTR=os.path.join(tmp, "ctr"),
             **{k: v for k, v in env.items() if v is not None})
    for f in (e["SB_LOG"], e["SB_CTR"], e["SB_LOG"] + ".test",
              e["SB_LOG"] + ".pairenv", e["SB_LOG"] + ".arm",
              e["SB_LOG"] + ".sel"):
        if os.path.exists(f):
            os.remove(f)
    out = subprocess.run(["bash", "slurm/submit_pipeline.sh", *args],
                         cwd=sia, env=e, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == _rc, out.stdout[-2000:] + out.stderr[-2000:]
    if not os.path.exists(e["SB_LOG"]):
        return [], out.stdout
    arms = {}
    if os.path.exists(e["SB_LOG"] + ".arm"):
        with open(e["SB_LOG"] + ".arm") as f:
            arms = dict(ln.rstrip("\n").split("|", 1) for ln in f)
    with open(e["SB_LOG"]) as f:
        calls = [ln.rstrip("\n").split("|", 2) for ln in f]
    jobs = []
    for j, t, a in calls:
        if j.isdigit():          # a stub sbatch call: its arm is logged too
            assert j in arms, (j, arms)
            jobs.append(_Job(j, t, _plain_export(a), arms[j]))
        else:                    # another stub a test logs here (its
            jobs.append(_Job(j, t, a, ""))   # $PYTHON_PATH's "py||<args>")
    return jobs, out.stdout


def _paste(tmp, out, **env):
    """Run the one command a WARNING printed for a refused export, as its
    reader would: pasted from siamese_sl/ into a shell that may still export
    PAIR_EXPORT_* (`env`), and exports no PCA_ARM / RUN_TOKEN. Returns the
    stub sbatch call it makes, as (args, (PAIR_EXPORT_MODEL_DIR,
    PAIR_EXPORT_FT_SPLITS), (its PCA_ARM, its RUN_TOKEN)): what the job gets,
    from the command alone."""
    cmd = [ln for ln in out.splitlines()
           if ln.startswith("    ") and "sbatch" in ln
           and ln.endswith(" slurm/" + _PAIR_SH)]
    assert len(cmd) == 1, out
    sia, b = os.path.join(tmp, "siamese_sl"), os.path.join(tmp, "bin")
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "PAIR_EXPORT_", "SB_", "PCA_"))
         and k not in ("RUN_TOKEN", "GPU_NODES", "GPU_EXCLUDE")}
    e.update(PATH=b + os.pathsep + e["PATH"],
             SB_LOG=os.path.join(tmp, "paste.log"),
             SB_CTR=os.path.join(tmp, "paste.ctr"), **env)
    for f in (e["SB_LOG"], e["SB_CTR"], e["SB_LOG"] + ".pairenv",
              e["SB_LOG"] + ".arm"):
        if os.path.exists(f):
            os.remove(f)
    r = subprocess.run(["bash", "-c", cmd[0]], cwd=sia, env=e,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (cmd[0], r.stdout, r.stderr)
    with open(e["SB_LOG"]) as f:
        (_, tok, args), = [ln.rstrip("\n").split("|", 2) for ln in f]
    with open(e["SB_LOG"] + ".pairenv") as f:
        (_, model_dir, splits), = [ln.rstrip("\n").split("|") for ln in f]
    with open(e["SB_LOG"] + ".arm") as f:
        (_, arm), = [ln.rstrip("\n").split("|", 1) for ln in f]
    return _plain_export(args), (model_dir, splits), (arm, tok)


def test_stub_switches_of_the_calling_shell_do_not_reach_the_stub():
    # SB_PAIR_FAIL / SB_TEST_FAIL are this file's own stub switches: only a
    # value a test passes on purpose may reach the stub sbatch.
    saved = {k: os.environ.get(k) for k in ("SB_PAIR_FAIL", "SB_TEST_FAIL")}
    os.environ.update(SB_PAIR_FAIL="1", SB_TEST_FAIL="1")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _stub_tree(tmp)
            base = ("--skip-generate", "--skip-reset-env")
            jobs, out = _submit(tmp, *base)
            assert [a for _, _, a in jobs if a.endswith(_PAIR_SH)], jobs
            assert "WARNING: could not submit" not in out, out
            _, out = _submit(tmp, *base, SB_PAIR_FAIL="1")
            assert "WARNING: could not submit" in out, out
            args, _, _ = _paste(tmp, out)
            assert args.endswith(" slurm/" + _PAIR_SH), args
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_submit_pipeline_names_what_is_queued_when_an_sbatch_is_refused():
    # A refused sbatch ends the submission under set -e with part of the DAG
    # queued (reset_env, which deletes the venv, among it): the script says
    # so and names what to cancel, as the root orchestrator does. Each
    # command alone on its line (a trailing "# before resubmitting" reached
    # scancel as job ids in an interactive zsh), the venv rebuild with the
    # settings taken from the calling shell (a one-shot TORCH_SPEC here,
    # which reset_env.sh reads).
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        jobs, out = _submit(tmp, _rc=1, SB_FAIL_ON="run_ablation.sh",
                            TORCH_SPEC="torch==2.5.1")
        ids = [j for j, _, _ in jobs]
        assert [a.rsplit("/", 1)[-1] for _, _, a in jobs] == [
            "reset_env.sh", "run_generate_embeddings.sh",
            "run_embedding_benchmark.sh", "run_embedding_benchmark_summary.sh",
            "run_best_per_category.sh", "run_best_per_category_combo.sh",
            "eval_finetuning.sh"], jobs
        assert ("submission stopped part-way; already queued: "
                + " ".join(ids)) in out, out[-1500:]
        assert "\n    scancel " + " ".join(ids) + "\n" in out, out[-1500:]
        assert f"reset_env ({ids[0]}) deletes the venv as soon as it starts" \
            in out and "All jobs submitted" not in out, out[-1500:]
        assert ("the venv before resubmitting:\n    TORCH_SPEC=torch==2.5.1 "
                "sbatch slurm/reset_env.sh\n") in out, out[-1500:]
        # no reset job; an id the calling shell exports is not this run's
        jobs, out = _submit(tmp, "--skip-reset-env", "--skip-generate", _rc=1,
                            SB_FAIL_ON="run_ablation.sh",
                            ENV_JOB="inherited_env", ABL_JOB="inherited_abl")
        ids = [j for j, _, _ in jobs]
        assert len(ids) == 5 and "\n    scancel " + " ".join(ids) + "\n" in out \
            and "inherited_" not in out and "deletes the venv" not in out, \
            out[-1500:]
        # nothing queued yet: sbatch's own error is all there is to say
        jobs, out = _submit(tmp, _rc=1, SB_FAIL_ON="reset_env.sh")
        assert not jobs and "part-way" not in out, out[-1500:]
        # the export alone refused stays a WARNING, and a whole submission
        # prints no such report
        _, out = _submit(tmp, SB_PAIR_FAIL="1")
        assert "WARNING: could not submit" in out and "part-way" not in out
        _, out = _submit(tmp)
        assert "All jobs submitted" in out and "part-way" not in out
        # every arm's: the first arm's ablation refused, the other owner's
        # arrays (queued before any arm's stages) are named too (an id
        # inherited under an arm's name: test_inherited_job_ids_never_reach_
        # a_dependency)
        jobs, out = _submit(tmp, "--skip-reset-env", "--skip-generate", _rc=1,
                            PCA_ARMS=None, SB_FAIL_ON="run_ablation.sh")
        ids = [j for j, _, _ in jobs]
        assert _arm_names(jobs) == [
            f"{s}/{o}" for o in ("none", "per_modality")
            for s in ("run_embedding_benchmark.sh",
                      "run_embedding_benchmark_summary.sh",
                      "run_best_per_category.sh")] + [
            "run_best_per_category_combo.sh/none", "eval_finetuning.sh/none"], \
            jobs
        assert ("submission stopped part-way; already queued: "
                + " ".join(ids)) in out \
            and "\n    scancel " + " ".join(ids) + "\n" in out, out[-1500:]


def test_inherited_job_ids_never_reach_a_dependency():
    # Both orchestrators keep each job id in a variable (<NAME>, saved per
    # arm or owner as <NAME>_<arm>), every one blanked at start: an id the
    # calling shell exports under such a name (a previous submission's,
    # sourced or exported by hand) must never become a dependency, a token
    # or a printed command. The paths that READ a per-arm id this run never
    # writes: submit_pipeline.sh's export after --best-cat-only (each
    # owner's benchmark summary, BENCH_SUMM_<owner>, not queued), and the
    # root's --skip-siamese (COMBO_JOB_ / CELL_JOB_ / EVAL_JOB_ /
    # ABL_SUMM_<arm>, none queued, in each arm's compare / external
    # transfer), --skip-shared (SHARED_JOB, in the same), --skip-paper-
    # tables or one arm (PT_ARMS_JOB, in every export) and, in either, a
    # stop part-way without a reset (ENV_JOB and the list of what is
    # queued, _QUEUED, in its report); the stages a flag skips, in either,
    # for good measure.
    def inherited(names):
        env = {n: f"INH.{n}" for n in names}
        env.update({f"{n}_{a}": f"INH.{n}_{a}" for n in names for a in _ARMS})
        env.update({f"_ARM_TOKEN_{a}": f"INH.tok_{a}" for a in _ARMS})
        env["_QUEUED"] = "INH._QUEUED"
        return env

    sp = inherited(("ENV_JOB", "GEN_JOB", "BENCH_JOB", "BENCH_SUMM",
                    "BESTCAT_JOB", "COMBO_JOB", "EVAL_JOB", "ABL_JOB",
                    "ABL_SUMM", "PAIR_JOB"))
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        flat = ("--skip-reset-env", "--skip-generate")
        for args, env, rc in (
                ((*flat, "--best-cat-only"), {}, 0),
                ((*flat, "--best-cat-only", "--skip-eval", "--skip-ablation"),
                 {}, 0),
                (flat, {}, 0), (("--ablation-only",), {}, 0),
                (flat, {"SB_FAIL_ON": "run_ablation.sh"}, 1)):
            for a in _ARMS:          # (for --ablation-only: its selections)
                r = os.path.join(sia, _root_of(a))
                os.makedirs(r, exist_ok=True)
                with open(os.path.join(r, "best_per_category.json"), "w") as f:
                    json.dump({"cv1": {}, "_selection_meta": {
                        "run_token": "T1"}}, f)
                with open(os.path.join(r, "latest_combo_cv3.txt"), "w") as f:
                    f.write(f"{_root_of(a)}/x_cv3\n")
            jobs, out = _submit(tmp, *args, _rc=rc, PCA_ARMS=None, **sp,
                                **env)
            names = _arm_names(jobs)
            assert ("export_pair_predictions.sh/none" in names) \
                == (args != ("--ablation-only",) and not rc), (args, names)
            assert ("submission stopped part-way" in out) == bool(rc) \
                and "reset_env (" not in out, out[-2000:]
            assert not [a for _, _, a in jobs if "INH." in a] \
                and {t for _, t, _ in jobs} != {""} \
                and not [t for _, t, _ in jobs if "INH." in t] \
                and "INH." not in out, (args, jobs, out[-2000:])
    root_env = inherited(("ENV_JOB", "GEN_JOB", "BESTCAT_JOB", "FOLDS_JOB",
                          "SHARED_JOB", "PT_ARMS_JOB", "COMBO_JOB", "EVAL_JOB",
                          "ABL_JOB", "ABL_SUMM", "CELL_JOB", "COMPARE_JOB",
                          "EXT_JOB", "PT_JOB", "PAIR_JOB"))
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        for a in _ARMS:              # each arm's reused selection
            r = os.path.join(root, "siamese_sl", _root_of(a))
            os.makedirs(r, exist_ok=True)
            with open(os.path.join(r, "best_per_category.json"), "w") as f:
                json.dump({"_selection_meta": {"run_token": "T1"}}, f)
        last = {"compare", "ext", "cellabl", "summary", "eval"}
        for args, arms, env, want_rc in (
                (("--skip-siamese",), None, {}, 0),
                (("--skip-generate",), None, {}, 0),
                (("--skip-generate", "--skip-eval", "--skip-ablation",
                  "--skip-cellline-ablation"), None, {}, 0),
                # SHARED_JOB, in every arm's compare / external transfer
                (("--skip-generate", "--skip-shared"), "none joint", {}, 0),
                # PT_ARMS_JOB, in every export: none without paper tables or
                # with one arm (the exports then wait on every arm's last
                # jobs, or on the one arm's paper tables)
                (("--skip-generate", "--skip-paper-tables"), "none joint",
                 {}, 0),
                (("--skip-generate",), "joint", {}, 0),
                # ENV_JOB, in the report of a stop part-way without a reset
                (("--skip-generate", "--skip-reset-env"), "none joint",
                 {"SB_FAIL_ON": "run_ablation.sh"}, 1)):
            rc, out, jobs = _root_run(tmp, *args, PCA_ARMS=arms,
                                      **root_env, **env)
            assert rc == want_rc, (args, arms, out[-2500:])
            leaked = {n: v[1] for n, v in jobs.items()
                      if not n.startswith("_")
                      and [p for p in v[1] if "INH." in p]}
            assert not leaked and not [
                t for t in jobs["_tokens"].values() if "INH." in t] \
                and "INH." not in out, (args, arms, leaked, out[-2500:])
            if rc:
                assert "submission stopped part-way" in out \
                    and "reset_env (" not in out, out[-2500:]
                continue
            assert ("compare/none" in jobs or arms == "joint") and (
                ("combo/joint" in jobs) == ("--skip-siamese" not in args)), \
                (args, list(jobs))
            if "--skip-paper-tables" in args:
                # each arm's export still waits on every arm's last jobs
                for a, b in (("none", "joint"), ("joint", "none")):
                    assert {f"{j}/{b}" for j in last} \
                        <= set(jobs[f"export/{a}"][1]), jobs[f"export/{a}"]
            if arms == "joint":
                assert "tables/joint" in jobs["export/joint"][1], jobs


def test_submit_pipeline_afterok_edges_carry_kill():
    # Without the flag a failed parent leaves its child pending forever as
    # DependencyNeverSatisfied, and every afterany consumer waits on that.
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        for args in ((),
                     ("--skip-generate", "--skip-reset-env", "--best-cat-only"),
                     ("--after", "777"),
                     ("--eval-only", "results/x_cv3", "--reset-env")):
            if args[:1] == ("--eval-only",):
                d = os.path.join(tmp, "siamese_sl", "results", "x_cv3")
                os.makedirs(d, exist_ok=True)
                for f in ("config.json", "results.json"):
                    open(os.path.join(d, f), "w").write("{}")
            for arms in ("none", None):       # one arm; every arm
                jobs, _ = _submit(tmp, *args, PCA_ARMS=arms)
                afterok = [a for _, _, a in jobs if "afterok" in a]
                assert afterok, (args, jobs)
                assert all("--kill-on-invalid-dep=yes" in a
                           for a in afterok), (args, jobs)
                # one token per submission, shared by every job of it
                assert len({t for _, t, _ in jobs}) == 1, jobs


def test_submit_pipeline_runs_every_pca_arm():
    # Every arm of PCA_ARMS (config.conf's default: all four; any subset, in
    # canonical order whatever order the list gives): the env reset and the
    # generation once; the benchmark array + summary and the Step-1 array once
    # per OWNER of single-embedding runs (each pair's first listed arm), as
    # that arm; per arm its combo (afterany its owner's Step-1 array), eval
    # and ablation (afterok the combo, with kill) and the summary; the arms'
    # exports last, singleton, each afterany every arm's combo, eval and
    # ablation summary and every owner's benchmark summary.
    def check(jobs, arms):
        by = _by_arm(jobs)
        owners = _owners(arms)
        want = ["reset_env.sh", "run_generate_embeddings.sh"]
        for o in owners:
            want += [f"run_embedding_benchmark.sh/{o}",
                     f"run_embedding_benchmark_summary.sh/{o}",
                     f"run_best_per_category.sh/{o}"]
        for a in arms:
            want += [f"run_best_per_category_combo.sh/{a}",
                     f"eval_finetuning.sh/{a}", f"run_ablation.sh/{a}",
                     f"run_ablation_summary.sh/{a}"]
        want += [f"{_PAIR_SH}/{a}" for a in arms]
        assert _arm_names(jobs) == want, (arms, _arm_names(jobs))
        gen = by["run_generate_embeddings.sh"][0]
        terminal = []
        for a in arms:
            combo, args = by[f"run_best_per_category_combo.sh/{a}"]
            step1 = by[f"run_best_per_category.sh/{_owner(a, arms)}"][0]
            assert f" --dependency=afterany:{step1} " in args \
                and "--kill-on-invalid-dep" not in args, (a, args)
            for s in ("eval_finetuning.sh", "run_ablation.sh"):
                assert (f" --dependency=afterok:{combo} "
                        "--kill-on-invalid-dep=yes ") in by[f"{s}/{a}"][1], \
                    (a, s, by[f"{s}/{a}"][1])
            assert " --array=0-11 " in by[f"eval_finetuning.sh/{a}"][1] \
                and " --array=0-71 " in by[f"run_ablation.sh/{a}"][1], a
            abl = by[f"run_ablation.sh/{a}"][0]
            assert f" --dependency=afterany:{abl} " in \
                by[f"run_ablation_summary.sh/{a}"][1], a
            terminal += [combo, by[f"eval_finetuning.sh/{a}"][0],
                         by[f"run_ablation_summary.sh/{a}"][0]]
        for o in owners:
            for s, n in (("run_best_per_category.sh", "56"),
                         ("run_embedding_benchmark.sh", "59")):
                arr = by[f"{s}/{o}"][1]
                assert (f" --dependency=afterok:{gen} "
                        "--kill-on-invalid-dep=yes ") in arr \
                    and f" --array=0-{n} " in arr, (o, s, arr)
            terminal.append(by[f"run_embedding_benchmark_summary.sh/{o}"][0])
        for a in arms:
            args = by[f"{_PAIR_SH}/{a}"][1]
            dep = re.search(r"--dependency=afterany:([0-9:]+),singleton ",
                            args)
            assert dep and dep.group(1).split(":") == terminal \
                and "afterok" not in args, (a, args, terminal)

    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        for arms_env, arms in ((None, _ARMS), ("both joint", ("joint", "both")),
                               ("per_modality", ("per_modality",)),
                               ("joint", ("joint",)),
                               ("none both", ("none", "both"))):
            jobs, out = _submit(tmp, PCA_ARMS=arms_env)
            check(jobs, arms)
            # one token for every job (the submission's: no arm passes one
            # of its own; and _submit saw a plain --export=ALL everywhere)
            assert len({t for _, t, _ in jobs}) == 1, jobs
            # the banner names each arm and where it writes, and each owner
            for a in arms:
                assert re.search(rf"(?m)^    {a} +{_root_of(a)}/ +\S", out), \
                    (a, out[:3000])
            assert out.count("    (Step 1)") == len(_owners(arms)), out[:3000]
        # the stage flags act on every arm
        base = ("--skip-generate", "--skip-reset-env")
        jobs, _ = _submit(tmp, *base, "--benchmark-only", PCA_ARMS=None)
        assert _arm_names(jobs) == [
            f"{s}/{o}" for o in ("none", "per_modality")
            for s in ("run_embedding_benchmark.sh",
                      "run_embedding_benchmark_summary.sh")], jobs
        jobs, _ = _submit(tmp, *base, "--best-cat-only", "--skip-eval",
                          "--skip-ablation", PCA_ARMS="none joint")
        assert _arm_names(jobs) == [
            "run_best_per_category.sh/none",
            "run_best_per_category_combo.sh/none",
            "run_best_per_category_combo.sh/joint",
            f"{_PAIR_SH}/none", f"{_PAIR_SH}/joint"], jobs
        # (no eval: every export is told to skip the fine-tuned sets)
        assert {v[1] for k, v in _pair_env(tmp).items()
                if k in {j for j, _, a in jobs if a.endswith(_PAIR_SH)}} \
            == {"none"}, _pair_env(tmp)
        jobs, _ = _submit(tmp, *base, "--skip-pair-export", PCA_ARMS=None)
        assert not [a for _, _, a in jobs if a.endswith(_PAIR_SH)], jobs


_PAIR_SH = "export_pair_predictions.sh"


def _pair_env(tmp):
    """{job id: (PAIR_EXPORT_MODEL_DIR, PAIR_EXPORT_FT_SPLITS)} each stub
    sbatch call was made with."""
    with open(os.path.join(tmp, "sb.log.pairenv")) as f:
        return {a: (b, c) for a, b, c in
                (ln.rstrip("\n").split("|") for ln in f)}


def test_pair_export_is_the_last_job_and_never_afterok():
    # Nothing reads the export, so it must neither be cancelled by a failed
    # parent (the paper tables exit 3 unless complete; an eval task exits 1
    # on a skipped screen) nor run beside a job a result depends on.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        base = ("--skip-generate", "--skip-reset-env")
        jobs, out = _submit(tmp, *base, PAIR_EXPORT_MODEL_DIR="results/old")
        ids = [j for j, _, _ in jobs]
        pair = [(j, a) for j, _, a in jobs if a.endswith(_PAIR_SH)]
        assert len(pair) == 1 and pair[0][0] == ids[-1], jobs
        job, args = pair[0]
        # ... and for any other pair_export job of the user (singleton): a
        # --eval-only export of one of these combos would clear its files
        dep = re.search(r"--dependency=afterany:([0-9:]+),singleton(?=\s)", args)
        assert dep and "afterok" not in args, args
        # it waits for every summary / terminal job queued before it
        by_script = {a.rsplit("/", 1)[-1]: j for j, _, a in jobs}
        for script in ("run_best_per_category_combo.sh", "eval_finetuning.sh",
                       "run_ablation_summary.sh",
                       "run_embedding_benchmark_summary.sh"):
            assert by_script[script] in dep.group(1).split(":"), (script, args)
        # an inherited single-combo override must not shrink the chained export
        assert _pair_env(tmp)[job] == ("", ""), _pair_env(tmp)[job]
        assert "ignoring PAIR_EXPORT_MODEL_DIR" in out, out
        assert "<cv3 combo>/pair_predictions/" in out, out
        assert "(+ fine-tuned Adamson predictions)" in out, out

        # no eval -> no fine-tuned models: probabilities only, not a FAILED job
        jobs, out = _submit(tmp, *base, "--skip-eval")
        job = [j for j, _, a in jobs if a.endswith(_PAIR_SH)]
        assert len(job) == 1 and _pair_env(tmp)[job[0]] == ("", "none"), jobs
        # ... and the closing summary does not promise them
        assert "(no fine-tuned predictions)" in out, out
        assert "fine-tuned Adamson" not in out, out
        # ... nor when no cv3 combo is exported (only that one is asked for
        # fine-tuned sets), and it names the combos and the screen the job
        # will: the environment's, else the defaults of the conf the job reads
        for env, combos, says in (
                (dict(PAIR_EXPORT_CVS="cv1"), "cv1",
                 "(no fine-tuned predictions)"),
                (dict(PAIR_EXPORT_CVS="cv1 cv2"), "cv1 cv2",
                 "(no fine-tuned predictions)"),
                (dict(PAIR_EXPORT_CVS="cv1 cv3"), "cv1 cv3",
                 "(+ fine-tuned Adamson predictions)"),
                (dict(PAIR_EXPORT_FT_SPLITS="none"), "cv3",
                 "(no fine-tuned predictions)"),
                (dict(PAIR_EXPORT_FT_DATASETS="Corn"), "cv3",
                 "(+ fine-tuned Corn predictions)")):
            _, out = _submit(tmp, *base, **env)
            assert f"<{combos} combo>/pair_predictions/" in out \
                and says in out, (env, out)
            assert ("no fine-tuned" in out) == ("no fine-tuned" in says), env
        conf = os.path.join(sia, "slurm", "export_pair_predictions.conf")
        with open(conf) as f:
            conf_text = f.read()
        for old, new, combos, says in (
                (":-Adamson}", ":-Corn}", "cv3",
                 "(+ fine-tuned Corn predictions)"),
                (":-cv3}", ":-cv2}", "cv2", "(no fine-tuned predictions)"),
                (":-cv1}", ":-none}", "cv3", "(no fine-tuned predictions)")):
            assert conf_text.count(old) == 1, old
            with open(conf, "w") as f:
                f.write(conf_text.replace(old, new))
            _, out = _submit(tmp, *base)
            assert f"<{combos} combo>/pair_predictions/" in out \
                and says in out, (new, out)
        with open(conf, "w") as f:
            f.write(conf_text)

        jobs, out = _submit(tmp, *base, "--skip-pair-export")
        assert not [a for _, _, a in jobs if a.endswith(_PAIR_SH)], jobs
        assert "pair_predictions/" not in out, out
        # no combo trained by the run -> nothing to export
        jobs, _ = _submit(tmp, *base, "--benchmark-only")
        assert not [a for _, _, a in jobs if a.endswith(_PAIR_SH)], jobs
        # slurmctld refusing the export alone (a QOS submit limit, say) is a
        # WARNING, not a failed submission: every other job is queued by then
        # (_submit asserts exit 0)
        queued, _ = _submit(tmp, *base, "--skip-pair-export")
        jobs, out = _submit(tmp, *base, SB_PAIR_FAIL="1")
        assert [a for _, _, a in jobs] == [a for _, _, a in queued], jobs
        assert "WARNING: could not submit the all-gene-pair export" in out, out
        assert "pair_predictions/" not in out, out
        # the command it prints keeps the export last: the dependency of the
        # refused submission (the stub hands out the same ids in both runs)
        want = re.search(r"--dependency=\S+", next(
            a for _, _, a in _submit(tmp, *base)[0]
            if a.endswith(_PAIR_SH))).group(0)
        assert want.endswith(",singleton"), want
        # (the whole line: with the eval queued it must NOT switch the
        # fine-tuned sets off, and it clears a PAIR_EXPORT_MODEL_DIR the
        # shell it is pasted into may export, as the refused submission did;
        # the arm in front of sbatch, as the submission passed it)
        v4 = ('USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled '
              'bilinear:per_head bilinear:pooled" ')
        assert (f"\n    {v4}PAIR_EXPORT_MODEL_DIR= PCA_ARM=none sbatch {want} "
                f"--export=ALL slurm/{_PAIR_SH}\n") in out, out
        # ... run as printed, it IS the refused submission, also from a
        # shell exporting USE_CELL_LINES=0 and another variant list (the
        # export loops over the variants; 2026-10-22)
        _env_stubs(tmp)
        args, penv, parm = _paste(tmp, out, PAIR_EXPORT_MODEL_DIR="results/old",
                                  USE_CELL_LINES="0",
                                  SIAMESE_VARIANTS_LIST="bilinear:pooled")
        assert args == f"{want} --export=ALL slurm/{_PAIR_SH}" \
            and penv == ("", "") and parm == ("none", ""), (args, penv, parm)
        jenv = dict(x.split("=", 1) for x in open(
            glob.glob(os.path.join(tmp, "paste.log.env.*"))[0]).read().split("\n")
            if "=" in x)
        assert (jenv["USE_CELL_LINES"], jenv["SIAMESE_VARIANTS_LIST"]) == (
            "1", "pd:per_head pd:pooled bilinear:per_head bilinear:pooled"), jenv
        # ... on the nodes of the submission (and, since 2026-10-21, after the
        # settings it took from the calling shell: GPU_NODES is one)
        _, out = _submit(tmp, *base, SB_PAIR_FAIL="1", GPU_NODES="gpu4")
        assert (f"\n    GPU_NODES=gpu4 {v4}PAIR_EXPORT_MODEL_DIR= PCA_ARM=none "
                f"sbatch --nodelist=gpu4 {want} --export=ALL slurm/{_PAIR_SH}\n") \
            in out, out
        # with no eval queued, it also keeps PAIR_EXPORT_FT_SPLITS=none
        _, out = _submit(tmp, *base, "--skip-eval", SB_PAIR_FAIL="1")
        assert re.search(r"\n    " + re.escape(v4) + r"PAIR_EXPORT_MODEL_DIR= PAIR_EXPORT_FT_SPLITS="
                         r"none PCA_ARM=none sbatch --dependency=afterany:"
                         r"[0-9:]+,singleton --export=ALL slurm/" + _PAIR_SH
                         + r"\n", out), out
        args, penv, parm = _paste(tmp, out, PAIR_EXPORT_MODEL_DIR="results/old")
        assert args.endswith(f",singleton --export=ALL slurm/{_PAIR_SH}") \
            and penv == ("", "none") and parm == ("none", ""), \
            (args, penv, parm)

        # A mistyped PAIR_EXPORT_* value would be seen only by the last job:
        # refused at submission, before anything is removed or queued; not
        # checked when the export is not submitted.
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "PAIR_EXPORT_"))
             and k != "RUN_TOKEN"}
        e.update(PATH=os.path.join(tmp, "bin") + os.pathsep + e["PATH"],
                 SB_LOG=os.path.join(tmp, "sb3.log"),
                 SB_CTR=os.path.join(tmp, "ctr"))
        for bad in (dict(PAIR_EXPORT_UNIVERSE="Shared"),
                    dict(PAIR_EXPORT_CVS="all"),
                    dict(PAIR_EXPORT_FT_SPLITS="all"),
                    dict(PAIR_EXPORT_PER_FOLD="yes"),
                    dict(PAIR_EXPORT_RAM_GB="40G"),
                    dict(PAIR_EXPORT_FT_DATASETS="none")):
            r = subprocess.run(["bash", "slurm/submit_pipeline.sh", *base],
                               cwd=sia, env=dict(e, **bad),
                               capture_output=True, text=True, timeout=120)
            name = next(iter(bad))
            assert r.returncode == 1 and f"ERROR: {name}" in r.stderr, \
                (bad, r.stdout[-600:], r.stderr[-600:])
            assert "--skip-pair-export" in r.stdout, r.stdout[-600:]
            assert not os.path.exists(e["SB_LOG"]), bad
            jobs, _ = _submit(tmp, *base, "--skip-pair-export", **bad)
            assert jobs, bad
        # every documented value passes, and the conf's defaults do not leak
        # into the jobs' environment (the job sources the conf itself)
        jobs, _ = _submit(tmp, *base, PAIR_EXPORT_UNIVERSE="shared",
                          PAIR_EXPORT_DTYPE="float16",
                          PAIR_EXPORT_CVS="cv1 cv2 cv3",
                          PAIR_EXPORT_PER_FOLD="1", PAIR_EXPORT_LOGITS="1",
                          PAIR_EXPORT_FT_DATASETS="all",
                          PAIR_EXPORT_FT_MODES="Full", PAIR_EXPORT_RAM_GB="7.5")
        job = [j for j, _, a in jobs if a.endswith(_PAIR_SH)]
        assert len(job) == 1 and _pair_env(tmp)[job[0]] == ("", ""), jobs

        # --eval-only: the one evaluated combo, after its eval
        d = os.path.join(sia, "results", "x_cv3")
        os.makedirs(d)
        for f in ("config.json", "results.json"):
            open(os.path.join(d, f), "w").write("{}")
        jobs, _ = _submit(tmp, "--eval-only", "results/x_cv3")
        assert [a.rsplit("/", 1)[-1] for _, _, a in jobs] == [
            "eval_finetuning.sh", _PAIR_SH], jobs
        assert f"--dependency=afterany:{jobs[0][0]},singleton " \
            in jobs[1][2], jobs
        assert _pair_env(tmp)[jobs[1][0]] == ("results/x_cv3", ""), jobs
        jobs, _ = _submit(tmp, "--eval-only", "results/x_cv3",
                          "--skip-pair-export")
        assert [a.rsplit("/", 1)[-1] for _, _, a in jobs] == [
            "eval_finetuning.sh"], jobs
        jobs, out = _submit(tmp, "--eval-only", "results/x_cv3",
                            SB_PAIR_FAIL="1")
        assert [a.rsplit("/", 1)[-1] for _, _, a in jobs] == [
            "eval_finetuning.sh"], jobs
        assert "WARNING: could not submit the all-gene-pair export" in out, out
        # the command it prints: this one combo, after its eval, singleton
        want = f"--dependency=afterany:{jobs[0][0]},singleton"
        assert (f"\n    PAIR_EXPORT_MODEL_DIR=results/x_cv3 PCA_ARM=none "
                f"sbatch {want} --export=ALL slurm/{_PAIR_SH}\n") in out, out
        args, penv, parm = _paste(tmp, out)
        assert args == f"{want} --export=ALL slurm/{_PAIR_SH}" \
            and penv == ("results/x_cv3", "") and parm == ("none", ""), \
            (args, penv, parm)
        # ... on the nodes of the submission, after the settings taken from
        # the calling shell
        _, out = _submit(tmp, "--eval-only", "results/x_cv3",
                         SB_PAIR_FAIL="1", GPU_NODES="gpu4")
        assert (f"\n    GPU_NODES=gpu4 PAIR_EXPORT_MODEL_DIR=results/x_cv3 "
                f"PCA_ARM=none sbatch --nodelist=gpu4 {want} --export=ALL "
                f"slurm/{_PAIR_SH}\n") \
            in out, out

        # a checkout without the export's code (any of its three files) is
        # refused before anything
        e = {k: v for k, v in os.environ.items()
             if not k.startswith("SLURM_") and k != "RUN_TOKEN"}
        e.update(PATH=os.path.join(tmp, "bin") + os.pathsep + e["PATH"],
                 SB_LOG=os.path.join(tmp, "sb2.log"),
                 SB_CTR=os.path.join(tmp, "ctr"))
        for rel in ("export_pair_predictions.py", "slurm/" + _PAIR_SH,
                    "slurm/export_pair_predictions.conf"):
            os.rename(os.path.join(sia, rel), os.path.join(tmp, "aside"))
            r = subprocess.run(["bash", "slurm/submit_pipeline.sh", *base],
                               cwd=sia, env=e, capture_output=True, text=True,
                               timeout=120)
            assert r.returncode == 1 and rel.rsplit("/", 1)[-1] + " missing" \
                in r.stdout, (rel, r.stdout[-800:])
            assert not os.path.exists(e["SB_LOG"]), rel
            jobs, _ = _submit(tmp, *base, "--skip-pair-export")
            assert jobs, jobs
            os.rename(os.path.join(tmp, "aside"), os.path.join(sia, rel))


def test_eval_only_takes_the_arm_of_its_directory():
    # --eval-only DIR evaluates (and exports) the one combo it is given and
    # writes inside it, whatever its arm; its jobs are submitted as the arm
    # whose results root holds DIR (results/pca_<arm>/<combo>, a combo being a
    # direct child of its arm's root), so their banners and log names say
    # which, and as arm none anywhere else. A refused export's command keeps
    # the same arm and logs.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        for rel, arm in (("results/x_cv3", "none"),
                         ("results/pca_joint/x_cv3", "joint"),
                         ("results/pca_per_modality/x_cv3", "per_modality"),
                         ("results/pca_both/sub/x_cv3", "none")):
            d = os.path.join(sia, rel)
            os.makedirs(d)
            for f in ("config.json", "results.json"):
                with open(os.path.join(d, f), "w") as fh:
                    fh.write("{}")
            jobs, out = _submit(tmp, "--eval-only", rel, PCA_ARMS=None)
            assert _arm_names(jobs) == [f"eval_finetuning.sh/{arm}",
                                        f"{_PAIR_SH}/{arm}"], (rel, jobs)
            assert f"(target: {rel}; PCA arm {arm})" in out, (rel, out)
            logs = [[x for x in a.split() if x.startswith("--output=")]
                    for _, _, a in jobs]
            assert logs == ([[], []] if arm == "none" else [
                [f"--output=slurm/logs/eval_finetuning_pca_{arm}_%A_%a.out"],
                [f"--output=slurm/logs/pair_export_pca_{arm}_%j.out"]]), \
                (rel, logs)
            assert f"eval_finetuning_{_tag(arm)}{jobs[0][0]}_0.out" in out, out
            _, out = _submit(tmp, "--eval-only", rel, SB_PAIR_FAIL="1",
                             PCA_ARMS=None)
            args, penv, parm = _paste(tmp, out)
            assert parm == (arm, "") and penv == (rel, "") \
                and (f"--output=slurm/logs/pair_export_pca_{arm}_%j.out"
                     in args) == (arm != "none"), (rel, args, penv, parm)


def _root_block(first, last="fi"):
    """The physical lines of run_benchmarking_siamese_vs_others.sh from the
    LAST line `first(line)` accepts through the next line equal to `last`.

    The real root script is never executed here (its step 0 rewrites the
    folds; _root_run runs a copy of the tree); a block that only sources a
    conf and echoes can be. A block naming one of
    the commands below is refused. That is a blacklist, not a proof, so
    _run_root_block also runs a block with an empty PATH (shell builtins
    only), on a copy of the conf, and checks that it left no file."""
    with open(_ORCH) as f:
        raw = f.read().split("\n")
    i = max(k for k, x in enumerate(raw) if first(x))
    while i and raw[i - 1].endswith("\\"):      # a continued first line
        i -= 1
    j = raw.index(last, i)
    block = raw[i:j + 1]
    code = "\n".join(x for x in block if not x.lstrip().startswith("#"))
    assert not re.search(r"\b(sbatch|scancel|rm|mv|cp|mkdir|chmod|touch|cd|"
                         r"tee|python\w*|prepare_folds)\b|PYTHON_PATH",
                         code), code
    return "\n".join(block)


def _run_root_block(block, tmp, slurm_dir=None, **env):
    """Run such a block under the root script's `set -e`, then report what
    it left in the shell. No command of the machine is on its PATH (a block
    that comes to need one fails here with "command not found" and has to be
    given a stub on purpose), $SLURM_DIR holds a copy (under `tmp`) of the
    conf the blocks source unless another directory is given, and a block
    that leaves a file in either directory fails the assertion below."""
    script = ("set -e\nSLURM_DIR=\"$1\"\n" + block + "\n"
              'echo "END U=[${PAIR_EXPORT_UNIVERSE:-}] '
              'C=[${PAIR_EXPORT_CVS:-}] '
              'F=[$(type -t check_pair_export || true)]"\n')
    if slurm_dir is None:
        slurm_dir = os.path.join(tmp, "slurm_repo")
        os.makedirs(slurm_dir, exist_ok=True)
        shutil.copy(os.path.join(_SIAMESE, "slurm",
                                 "export_pair_predictions.conf"), slurm_dir)
    # Nothing of the calling shell is inherited, only what the test passes:
    # the callers assert that stderr is empty, and bash prints there at
    # start-up for an LC_ALL naming a locale the node lacks (bash 4+), a
    # BASH_ENV file that writes, an exported SHELLOPTS=xtrace. bash itself
    # is resolved from this process's PATH: the child has none.
    e = dict(env, PATH=os.path.join(tmp, "no-commands"))
    before = {d: sorted(os.listdir(d)) for d in (tmp, slurm_dir)}
    r = subprocess.run(
        [shutil.which("bash"), "-c", script, "root-block", slurm_dir],
        cwd=tmp, env=e, capture_output=True, text=True, timeout=60)
    after = {d: sorted(os.listdir(d)) for d in before}
    assert after == before, (after, before, r.stdout, r.stderr)
    return r


def test_pair_export_is_wired_into_the_root_orchestrator():
    # The root's text carries the contract (its runs are _root_run's, on a
    # copy of the tree: test_root_refused_exports_print_each_arms_own_command
    # and the DAG tests): the export is submitted after the paper tables,
    # afterany, from siamese_sl/, and only with the siamese stages.
    lines = _logical_lines(_ORCH)
    subs = [i for i, ln in enumerate(lines)
            if re.search(r"\bsbatch\s+--parsable\b", ln)]
    i_pair = [i for i in subs if _PAIR_SH in lines[i]]
    assert len(i_pair) == 1 and i_pair[0] == subs[-1], \
        [lines[i].strip()[:80] for i in subs[-3:]]
    ln = lines[i_pair[0]]
    assert "--dependency=afterany:$PAIR_DEPS,singleton --export=ALL " in ln \
        and "afterok" not in ln and 'cd "$SIAMESE_DIR"' in ln \
        and '_as_arm "$_arm" sbatch --parsable ' in ln, ln
    # one per arm, after its arm's jobs and every other arm's: through the
    # table across the arms, or without one through each arm's last jobs
    deps = next(ln for ln in lines if re.match(r"\s*PAIR_DEPS=", ln))
    for j in ("$PT_JOB", "$COMPARE_JOB", "$EXT_JOB", "$CELL_JOB",
              "$ABL_SUMM", "$EVAL_JOB", "$PT_ARMS_JOB", "$_ALL_LAST"):
        assert f'"{j}"' in deps, (j, deps)
    text = "\n".join(lines)
    assert re.search(r'if \[ "\$RUN_SIAMESE" = true \] && \[ "\$RUN_PAIR" = '
                     r'true \]; then\s+_ALL_LAST=', text)
    assert "--skip-pair-export)       RUN_PAIR=false; shift ;;" in text
    # ... and each of the following WHERE it acts (presence alone would pass
    # an inverted condition or a statement moved below the submission).
    # An inherited one-combo override is cleared before the export is queued:
    i_unset = [i for i, x in enumerate(lines)
               if x.strip() == "unset PAIR_EXPORT_MODEL_DIR"]
    assert len(i_unset) == 1 and i_unset[0] < i_pair[0], (i_unset, i_pair)
    # the fine-tuned sets are skipped only when the eval does not run, which
    # is settled before the submission as well:
    i_none = [i for i, x in enumerate(lines)
              if "PAIR_EXPORT_FT_SPLITS=none" in x]
    assert len(i_none) == 2 and i_none[0] < i_pair[0] < i_none[1] \
        and lines[i_none[0]].strip() == "export PAIR_EXPORT_FT_SPLITS=none" \
        and lines[i_none[0] - 1].strip() == \
        'if [ "$RUN_EVAL" != true ]; then', lines[i_none[0] - 1:i_none[0] + 1]
    # (the second one is the command printed when the submission is refused,
    # which must carry the same setting under the same condition)
    assert lines[i_none[1]].strip() == ('[ "$RUN_EVAL" = true ] || _PAIR_ENV='
                                        '"PAIR_EXPORT_FT_SPLITS=none "'), \
        lines[i_none[1]]
    # a refused submission is a WARNING (the sbatch is an `if` condition, so
    # neither set -e nor the ERR trap fires, and nothing exits before the
    # trap is lifted): the rest of the DAG is queued by then.
    i_trap = next(i for i, x in enumerate(lines) if x.strip() == "trap - ERR")
    assert lines[i_pair[0]].lstrip().startswith("if PAIR_JOB=$(") \
        and i_pair[0] < i_trap and not [
            x for x in lines[i_pair[0]:i_trap]
            if re.search(r"\bexit\b", x)], lines[i_pair[0]:i_trap]
    # the export's three files are checked with the rest of the code (so
    # before step 0), and only when the export is submitted:
    need = [i for i, x in enumerate(lines)
            if "NEED_CODE=" in x and _PAIR_SH in x]
    i_chk = next(i for i, x in enumerate(lines)
                 if x.strip() == "for _f in $NEED_CODE; do")
    assert len(need) == 1 and need[0] < i_chk \
        and '[ "$RUN_PAIR" = true ]' in lines[need[0]] and all(
            f in lines[need[0]] for f in (
                "siamese_sl/export_pair_predictions.py",
                "siamese_sl/slurm/" + _PAIR_SH,
                "siamese_sl/slurm/export_pair_predictions.conf")), need
    # A mistyped PAIR_EXPORT_* value is refused there too: after the files,
    # before step 0 changes anything (so before any job), only when the
    # export is submitted, and the run stops there.
    i_val = [i for i, x in enumerate(lines) if "check_pair_export" in x]
    i_step0 = next(i for i, x in enumerate(lines)
                   if re.search(r"chmod -R u\+w sl_comparison/folds", x))
    assert len(i_val) == 1 and i_chk < i_val[0] < i_step0 < subs[0], \
        (i_chk, i_val, i_step0, subs[0])
    check = _root_block(lambda x: "check_pair_export );" in x)
    assert check.startswith('if [ "$RUN_SIAMESE" = true ] && '
                            '[ "$RUN_PAIR" = true ]'), check
    on = dict(RUN_SIAMESE="true", RUN_PAIR="true")
    with tempfile.TemporaryDirectory() as tmp:
        # (in a subshell: neither the conf's defaults nor its function reach
        # the orchestrator's shell, hence the jobs)
        r = _run_root_block(check, tmp, **on)
        assert r.returncode == 0 and r.stdout == "END U=[] C=[] F=[]\n" \
            and not r.stderr, (r.stdout, r.stderr)
        for bad in (dict(PAIR_EXPORT_UNIVERSE="Shared"),
                    dict(PAIR_EXPORT_CVS="all"),
                    dict(PAIR_EXPORT_FT_SPLITS="all")):
            name = next(iter(bad))
            r = _run_root_block(check, tmp, **on, **bad)
            assert r.returncode == 1 and f"ERROR: {name}" in r.stderr \
                and "Nothing was changed or submitted." in r.stdout \
                and "--skip-pair-export" in r.stdout \
                and "END" not in r.stdout, (bad, r.stdout, r.stderr)
            # not checked when the export is not submitted
            for off in (dict(on, RUN_PAIR="false"),
                        dict(on, RUN_SIAMESE="false")):
                r = _run_root_block(check, tmp, **off, **bad)
                assert r.returncode == 0 and "END" in r.stdout \
                    and not r.stderr, (off, bad, r.stdout, r.stderr)

        # The command printed when the submission is refused: the export
        # stays last and singleton, on the same nodes, with the settings the
        # run took from the calling shell first (_ENV_PREFIX, 2026-10-21),
        # then $_PAIR_ENV and the arm (_arm_env, as _as_arm passed it: the
        # last assignment wins), and without a PAIR_EXPORT_MODEL_DIR the
        # shell it is pasted into may export. Word for word
        # submit_pipeline.sh's, which
        # test_pair_export_is_the_last_job_and_never_afterok runs as printed
        # (and test_root_refused_exports_print_each_arms_own_command runs the
        # root's, per arm).
        # (and, since 2026-10-22, the run's USE_CELL_LINES and variants:
        # the export loops over SIAMESE_VARIANTS, which a pasting shell's
        # USE_CELL_LINES=0 or list changed)
        cmd = [x.strip() for x in lines[i_pair[0] + 1:i_trap] if "sbatch" in x]
        assert cmd == [
            'echo "    ${_ENV_PREFIX}$(_vset "${SIAMESE_VARIANTS[*]}") '
            'PAIR_EXPORT_MODEL_DIR= ${_PAIR_ENV}'
            '$(_arm_env "$_arm") sbatch${GPU_NODE_ARGS:+ $GPU_NODE_ARGS} '
            '--dependency=afterany:$PAIR_DEPS,singleton $_PAIR_OPTS slurm/'
            + _PAIR_SH + '"'], cmd
        assert [x.strip() for x in _logical_lines(_SUBMIT)
                if "PAIR_EXPORT_MODEL_DIR= ${_PAIR_ENV}" in x] == cmd
        # ... and the helpers that build it are text-identical in both (the
        # RUN_TOKEN branch of the root's _arm_env no run of it reaches:
        # per-arm tokens exist only under --skip-siamese, which exports
        # nothing)
        texts = []
        for path in (_ORCH, _SUBMIT):
            with open(path) as f:
                texts.append(f.read())
        for fn in ("_as_arm", "_arm_env", "_logs", "_armtag", "_ucl", "_vset",
                   "_arms_pin"):
            # (a multi-line definition, else a one-line one)
            defs = [re.findall(r"(?ms)^%s\(\) \{\n.*?^\}$" % fn, t)
                    or re.findall(r"(?m)^%s\(\) \{ .*\}$" % fn, t)
                    for t in texts]
            assert len(defs[0]) == 1 and defs[0] == defs[1], (fn, defs)
        # (and the options it prints are the submission's plain --export
        # and the logs, in both)
        for path in (_ORCH, _SUBMIT):
            opts = [x.strip() for x in _logical_lines(path)
                    if re.match(r"\s*_PAIR_OPTS=", x)]
            assert opts == ['_PAIR_OPTS="--export=ALL"'] * (
                2 if path == _SUBMIT else 1), (path, opts)

        # The closing summary says what the job will do: the combos and the
        # screen it will take from the environment, else from its conf, and
        # no fine-tuned predictions unless a cv3 combo is exported with them.
        banner = _root_block(lambda x: x == 'if [ -n "$_PAIR_JOBS" ]; then')
        job = dict(_PAIR_JOBS="none:4242 joint:4243", SIA_LOGS="/logs")
        last = "; the last jobs, one per arm:"     # then one line per arm
        for env, combos, says in (
                ({}, "cv3", " and the fine-tuned Adamson predictions" + last),
                (dict(PAIR_EXPORT_FT_SPLITS="none"), "cv3",
                 ", no fine-tuned predictions" + last),
                (dict(PAIR_EXPORT_CVS="cv1"), "cv1",
                 ", no fine-tuned predictions" + last),
                (dict(PAIR_EXPORT_CVS="cv1 cv3",
                      PAIR_EXPORT_FT_DATASETS="Corn"), "cv1 cv3",
                 " and the fine-tuned Corn predictions" + last)):
            r = _run_root_block(banner, tmp, **job, **env)
            assert r.returncode == 0 and not r.stderr and (
                f"siamese_sl/<arm root>/<{combos} combo>/pair_predictions/"
                in r.stdout) and f"(sl_prob_<HEAD>.npy){says}\n" in r.stdout \
                and ("    none: job 4242 (tail -f /logs/pair_export_4242.out)"
                     "\n    joint: job 4243 (tail -f /logs/pair_export_pca_"
                     "joint_4243.out)\n") in r.stdout, (
                    env, r.stdout, r.stderr)
            # (the conf's defaults stay in the subshell)
            assert r.stdout.endswith(
                "END U=[] C=[%s] F=[]\n" % env.get("PAIR_EXPORT_CVS", "")), \
                (env, r.stdout)
        r = _run_root_block(banner, tmp, _PAIR_JOBS="", SIA_LOGS="/logs")
        assert r.returncode == 0 and r.stdout == "END U=[] C=[] F=[]\n", r.stdout
        # (the defaults are the conf's own, not a copy of them in the script)
        with open(os.path.join(_SIAMESE, "slurm",
                               "export_pair_predictions.conf")) as f:
            conf_text = f.read()
        other = os.path.join(tmp, "slurm")
        os.makedirs(other)
        for old, new, combos, says in (
                (":-Adamson}", ":-Corn}", "cv3",
                 " and the fine-tuned Corn predictions" + last),
                (":-cv3}", ":-cv2}", "cv2",
                 ", no fine-tuned predictions" + last),
                (":-cv1}", ":-none}", "cv3",
                 ", no fine-tuned predictions" + last)):
            assert conf_text.count(old) == 1, old
            with open(os.path.join(other, "export_pair_predictions.conf"),
                      "w") as f:
                f.write(conf_text.replace(old, new))
            r = _run_root_block(banner, tmp, slurm_dir=other, **job)
            assert r.returncode == 0 and (
                f"siamese_sl/<arm root>/<{combos} combo>/pair_predictions/"
                in r.stdout) and f"(sl_prob_<HEAD>.npy){says}\n" in r.stdout, \
                (new, r.stdout, r.stderr)

    # every list of live pipeline jobs knows the export's job name, which is
    # the script's own #SBATCH --job-name; a direct sbatch of it (no
    # --dependency on the command line) is singleton by that header alone
    # (sbatch reads directives up to the first command), and the manual DAG's
    # last step says so on its command line
    with open(os.path.join(_SIAMESE, "slurm", _PAIR_SH)) as f:
        job_text = f.read()
    head = job_text[:re.search(r"^[^#\s]", job_text, re.M).start()]
    name = re.search(r"^#SBATCH --job-name=(\S+)$", head, re.M).group(1)
    assert re.search(r"^#SBATCH --dependency=singleton$", head, re.M), \
        "the job's #SBATCH header must carry --dependency=singleton"
    with open(os.path.join(_ROOT, "sl_comparison", "README.md")) as f:
        by_hand = [x for x in f.read().splitlines()
                   if "sbatch" in x and _PAIR_SH in x]
    assert by_hand and all(
        re.search(r"--dependency=afterany:\S+,singleton\s", x)
        for x in by_hand), by_hand
    for path in (_ORCH, _SUBMIT, os.path.join(_SIAMESE, "slurm", "clean.sh"),
                 os.path.join(_ROOT, "clean_previous_runs.sh")):
        with open(path) as f:
            lists = re.findall(r"'\^\((reset_env\|[^']*)\)\$'", f.read())
        assert lists and all(name in ls.split("|") for ls in lists), path


# bash 3 (macOS) has no mapfile builtin; the cluster's bash does, and the shim
# is defined only where the builtin is missing.

_MAPFILE_SHIM = ('type mapfile >/dev/null 2>&1 || mapfile() { '
                 '[ "$1" = "-t" ] && shift; local __l; eval "$1=()"; '
                 'while IFS= read -r __l; do eval "$1+=(\\"\\$__l\\")"; done; }; ')


# $PYTHON_PATH of the export job: the CUDA probe answers $PY_CUDA_RC, the
# config.json reader runs for real, and an export call is logged (tab-joined
# argv) and leaves the manifest.json a finished export writes.

_PAIR_PY = r'''#!/bin/bash
if [ "$1" = "-c" ]; then
    case "$2" in *cuda.is_available*) exit "${PY_CUDA_RC:-0}" ;; esac
    exec "$PY_REAL" "$@"
fi
( IFS=$'\t'; echo "$*" ) >> "$PY_LOG"
d=""; while [ $# -gt 1 ]; do [ "$1" = "--model_dir" ] && d="$2"; shift; done
rc="${PY_RC:-0}"; [ "$d" = "${PY_FAIL_DIR:-}" ] && rc=1
if [ "$rc" = 0 ] && [ "${PY_MANIFEST:-1}" = 1 ]; then
    mkdir -p "$d/pair_predictions"; echo '{}' > "$d/pair_predictions/manifest.json"
fi
exit "$rc"
'''


def _pair_job(tmp, **env):
    """Run export_pair_predictions.sh itself in the stub tree; returns (exit
    status, output, [{flag: [values]} per export_pair_predictions.py call]).
    The job's arm is the test's alone (PCA_ARM / PCA_ARMS or none): an arm the
    calling shell exports (a hand recipe starts with `export PCA_ARM=<arm>`)
    would move it into another root; and every arm root's pair_predictions/
    goes first, results/pca_<arm>/<combo>/ too, where a manifest.json left
    over would pass for an export of this run (2026-10-21)."""
    sia, log = os.path.join(tmp, "siamese_sl"), os.path.join(tmp, "py.log")
    if os.path.exists(log):
        os.remove(log)
    for d in glob.glob(os.path.join(sia, "results", "**", "pair_predictions"),
                       recursive=True):
        shutil.rmtree(d)
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "PAIR_EXPORT_", "PY_", "PCA_",
                              "POST_PCA_"))
         and k not in ("RUN_TOKEN", "SIAMESE_VARIANTS_LIST",
                       "SIAMESE_VARIANTS", "SHARED_FOLDS")}
    e.update(PY_LOG=log, PY_REAL=sys.executable, **env)
    r = subprocess.run(["bash", "-c", _MAPFILE_SHIM +
                        ". slurm/export_pair_predictions.sh"], cwd=sia, env=e,
                       capture_output=True, text=True, timeout=120)
    # the stub accepts any flag; the real parser must know each one
    with open(os.path.join(_SIAMESE, "export_pair_predictions.py")) as f:
        known = set(re.findall(r'add_argument\("(--\w+)"', f.read()))
    calls = []
    if os.path.exists(log):
        with open(log) as f:
            for argv in (ln.split("\t") for ln in f.read().splitlines()):
                assert argv[0] == "export_pair_predictions.py", argv
                call, cur = {}, None
                for a in argv[1:]:
                    if a.startswith("--"):
                        cur = call.setdefault(a, [])
                    else:
                        cur.append(a)
                assert set(call) <= known, sorted(set(call) - known)
                calls.append(call)
    return r.returncode, r.stdout + r.stderr, calls


def test_pair_export_job_exports_each_variants_combo_and_finetuned_sets():
    # The job script and its conf, which no sbatch stub ever runs. Defaults:
    # every variant's cv3 combo, on CUDA, all genes, plus the Adamson models
    # fine-tuned under the cv1 screen split (both modes); exit 0 only when
    # every requested combo was exported in full.
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        py = os.path.join(b, "py")
        with open(py, "w") as f:
            f.write(_PAIR_PY)
        os.chmod(py, 0o755)
        # the copied config.conf names the cluster venv
        with open(os.path.join(sia, "slurm", "config.conf"), "a") as f:
            f.write(f'\nPYTHON_PATH="{py}"\n')
        os.makedirs(os.path.join(tmp, "data"))
        for n in ("all_genes_a.pt", "all_genes_b.pt"):
            open(os.path.join(tmp, "data", n), "w").close()
        prefixes = [ln[1:-1] for ln in _bash(
            'for v in "${SIAMESE_VARIANTS[@]}"; do '
            'echo "[$(siamese_variant_prefix "$v")]"; done')]
        assert len(prefixes) == 4 and prefixes[0] == "", prefixes
        res = os.path.join(sia, "results")

        def combo(name, marker=None):
            d = os.path.join(res, name)
            os.makedirs(d)
            with open(os.path.join(d, "config.json"), "w") as f:
                # another checkout's paths: rebased under this DATA_DIR
                json.dump({"embeddings_paths": ["/elsewhere/all_genes_a.pt",
                                                "../data/all_genes_b.pt"]}, f)
            open(os.path.join(d, "results.json"), "w").write("{}")
            if marker:
                with open(os.path.join(res, marker), "w") as f:
                    f.write(f"results/{name}\n")

        for p in prefixes:
            for cv in ("cv1", "cv3"):
                combo(f"{p}combo_{cv}", f"{p}latest_combo_{cv}.txt")
        cv3 = [f"results/{p}combo_cv3" for p in prefixes]
        ft = {"--finetuned_eval_dirs": ["eval_finetuning_cv1"],
              "--finetuned_datasets": ["Adamson"]}

        def finetuned(call):
            return {k: v for k, v in call.items()
                    if k.startswith("--finetuned")}

        def dirs(calls):
            return [c["--model_dir"][0] for c in calls]

        rc, out, calls = _pair_job(tmp)
        assert rc == 0 and dirs(calls) == cv3, (calls, out[-1500:])
        for c in calls:
            emb = c["--embeddings_paths"]
            assert [os.path.basename(x) for x in emb] == [
                "all_genes_a.pt", "all_genes_b.pt"], c
            assert all(os.path.samefile(os.path.dirname(x),
                                        os.path.join(tmp, "data"))
                       for x in emb), c
            assert c["--device"] == ["cuda"], c
            assert (c["--universe"], c["--dtype"]) == (["all"], ["float32"]), c
            assert "--per_fold" not in c and "--also_logits" not in c, c
            assert finetuned(c) == ft, c
        # every PAIR_EXPORT_* setting reaches the command line
        rc, out, calls = _pair_job(
            tmp, PAIR_EXPORT_UNIVERSE="shared", PAIR_EXPORT_DTYPE="float16",
            PAIR_EXPORT_RAM_GB="7", PAIR_EXPORT_PER_FOLD="1",
            PAIR_EXPORT_LOGITS="1", PAIR_EXPORT_FT_SPLITS="cv1 cv2",
            PAIR_EXPORT_FT_DATASETS="all", PAIR_EXPORT_FT_MODES="Full")
        assert rc == 0 and dirs(calls) == cv3, (calls, out[-1500:])
        for c in calls:
            assert (c["--universe"], c["--dtype"], c["--ram_gb"]) == (
                ["shared"], ["float16"], ["7"]), c
            assert c["--shared_genes"][0].endswith("/genes.txt"), c
            assert c["--per_fold"] == [] and c["--also_logits"] == [], c
            assert finetuned(c) == {
                "--finetuned_eval_dirs": ["eval_finetuning_cv1",
                                          "eval_finetuning_cv2"],
                "--finetuned_modes": ["Full"]}, c
        # no eval ran (the orchestrators' --skip-eval): probabilities only
        rc, out, calls = _pair_job(tmp, PAIR_EXPORT_FT_SPLITS="none")
        assert rc == 0 and dirs(calls) == cv3, (calls, out[-1500:])
        assert not any(finetuned(c) for c in calls), calls
        # only the cv3 combo is fine-tuned by eval_finetuning.py
        rc, out, calls = _pair_job(tmp, PAIR_EXPORT_CVS="cv1 cv3")
        assert rc == 0 and len(calls) == 8, (calls, out[-1500:])
        for c in calls:
            want = ft if c["--model_dir"][0].endswith("_cv3") else {}
            assert finetuned(c) == want, c
        # one explicit combo (--eval-only), whatever the markers say
        combo("picked")
        rc, out, calls = _pair_job(tmp, PAIR_EXPORT_MODEL_DIR="results/picked")
        assert rc == 0 and dirs(calls) == ["results/picked"], (calls, out)
        assert finetuned(calls[0]) == ft, calls

        # ---- anything short of every combo in full is exit 1 ----
        # no CUDA: nothing is exported on CPU
        rc, out, calls = _pair_job(tmp, PY_CUDA_RC="1")
        assert rc == 1 and not calls and "CUDA" in out, (calls, out[-1500:])
        # a failed / unfinished export: the other variants are still exported
        for env in (dict(PY_FAIL_DIR=cv3[1]), dict(PY_MANIFEST="0")):
            rc, out, calls = _pair_job(tmp, **env)
            assert rc == 1 and dirs(calls) == cv3, (env, out[-1500:])
        # a variant whose combo never trained has no marker
        os.remove(os.path.join(res, f"{prefixes[3]}latest_combo_cv3.txt"))
        rc, out, calls = _pair_job(tmp)
        assert rc == 1 and dirs(calls) == cv3[:3], (calls, out[-1500:])
        # a combo that did not finish training is not scored
        os.remove(os.path.join(res, f"{prefixes[2]}combo_cv3", "results.json"))
        rc, out, calls = _pair_job(tmp)
        assert rc == 1 and dirs(calls) == cv3[:2], (calls, out[-1500:])

        # ---- a mistyped PAIR_EXPORT_* value: refused before any export ----
        for bad in (dict(PAIR_EXPORT_UNIVERSE="Shared"),
                    dict(PAIR_EXPORT_DTYPE="float64"),
                    dict(PAIR_EXPORT_CVS="cv1,cv3"),
                    dict(PAIR_EXPORT_CVS=" "),
                    dict(PAIR_EXPORT_FT_SPLITS="all"),
                    dict(PAIR_EXPORT_FT_MODES="none"),
                    dict(PAIR_EXPORT_LOGITS="yes"),
                    dict(PAIR_EXPORT_RAM_GB="40G")):
            rc, out, calls = _pair_job(tmp, **bad)
            assert rc == 1 and not calls, (bad, calls, out[-800:])
            assert f"ERROR: {next(iter(bad))}" in out, (bad, out[-800:])
        # `true` stays a valid spelling of 1 for the two switches
        rc, out, calls = _pair_job(tmp, PAIR_EXPORT_PER_FOLD="true",
                                   PAIR_EXPORT_LOGITS="true")
        assert calls and all("--per_fold" in c and "--also_logits" in c
                             for c in calls), (calls, out[-800:])


# $PYTHON_PATH of the eval job: the config.json reader runs for real, and an
# eval_finetuning.py call is logged by its --cv_type and exits 1 for the
# cv_types in $PY_FAIL_CVS (a skipped screen, or an optional output missing).

_EVAL_PY = r"""#!/bin/bash
if [ "$1" = "-c" ]; then exec "$PY_REAL" "$@"; fi
cv=""; prev=""
for a in "$@"; do [ "$prev" = "--cv_type" ] && cv="$a"; prev="$a"; done
echo "$cv" >> "$PY_LOG"
case " ${PY_FAIL_CVS:-} " in *" $cv "*) exit 1 ;; esac
exit 0
"""


def test_eval_job_runs_every_cv_type_after_one_that_fails():
    # Without --array the job runs the cv_types back to back under `set -e`.
    # eval_finetuning.py exits 1 once its tables are written when a screen was
    # skipped or only an optional all-pair output is missing: that must not
    # cost the cv_types after it their tables, and the job must still fail.
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        py = os.path.join(b, "py")
        with open(py, "w") as f:
            f.write(_EVAL_PY)
        os.chmod(py, 0o755)
        with open(os.path.join(sia, "slurm", "config.conf"), "a") as f:
            f.write(f'\nPYTHON_PATH="{py}"\n')
        os.makedirs(os.path.join(tmp, "data"))
        open(os.path.join(tmp, "data", "all_genes_a.pt"), "w").close()
        d = os.path.join(sia, "results", "x_cv3")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"embeddings_paths": ["/elsewhere/all_genes_a.pt"]}, f)
        open(os.path.join(d, "results.json"), "w").write("{}")
        cvs = _bash('source slurm/eval_finetuning.conf >/dev/null 2>&1; '
                    'printf "%s\\n" "${EVAL_CV_TYPES[@]}"',
                    EVAL_MODEL_DIR="results")
        assert len(cvs) >= 2, cvs
        log = os.path.join(tmp, "py.log")

        def run(**env):
            if os.path.exists(log):
                os.remove(log)
            e = {k: v for k, v in os.environ.items()
                 if not k.startswith(("SLURM_", "PY_", "EVAL_"))
                 and k not in ("RUN_TOKEN", "SIAMESE_VARIANTS_LIST",
                               "SIAMESE_VARIANTS", "USE_CELL_LINES")}
            e.update(SLURM_SUBMIT_DIR=sia, EVAL_MODEL_DIR="results/x_cv3",
                     PY_LOG=log, PY_REAL=sys.executable, **env)
            # the job's `sleep 5` and (bash 3) mapfile, neither under test
            r = subprocess.run(
                ["bash", "-c", "sleep() { :; }; " + _MAPFILE_SHIM +
                 ". slurm/eval_finetuning.sh"], cwd=sia, env=e,
                capture_output=True, text=True, timeout=120)
            ran = []
            if os.path.exists(log):
                with open(log) as f:
                    ran = f.read().split()
            return r.returncode, ran, r.stdout + r.stderr

        rc, ran, out = run()
        assert rc == 0 and ran == cvs, (rc, ran, out[-1500:])
        for failing in (cvs[0], cvs[-1], " ".join(cvs)):
            rc, ran, out = run(PY_FAIL_CVS=failing)
            assert rc == 1 and ran == cvs, (failing, rc, ran, out[-1500:])
        # an array task is one cv_type and ends with its own status
        rc, ran, out = run(SLURM_ARRAY_TASK_ID="0", PY_FAIL_CVS=cvs[0])
        assert rc == 1 and ran == [cvs[0]], (rc, ran, out[-1500:])
        rc, ran, out = run(SLURM_ARRAY_TASK_ID="1", PY_FAIL_CVS=cvs[0])
        assert rc == 0 and ran == [cvs[1]], (rc, ran, out[-1500:])


def test_ablation_only_joins_the_selection_runs_token():
    # A fresh token would make every ablation row a MISMATCH and leave
    # paper_tables' tab:loco reading the previous run's dirs.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        res = os.path.join(sia, "results")
        with open(os.path.join(res, "best_per_category.json"), "w") as f:
            json.dump({"cv1": {}, "_selection_meta": {"run_token": "T1-sel"}},
                      f)
        for d in ("ablate_no_ppi_cv1", "var_pd_pooled__ablate_no_go_cv2"):
            os.makedirs(os.path.join(res, d))
        open(os.path.join(res, "ablation_summary.csv"), "w").write("x")
        os.makedirs(os.path.join(res, "pcavar0.8_go_cv1"))
        # the combo it ablates (without one, --ablation-only is refused)
        with open(os.path.join(res, "latest_combo_cv1.txt"), "w") as f:
            f.write("results/x_cv1\n")
        # the paper tables built on that ablation (tab:loco), and the table
        # across the arms built on those
        cmp_res = os.path.join(tmp, "sl_comparison", "results")
        for d in ("paper_tables", "pca_arms"):
            os.makedirs(os.path.join(cmp_res, d), exist_ok=True)
            open(os.path.join(cmp_res, d, "README.md"), "w").write("PAPER COMPLETE\n")
        jobs, out = _submit(tmp, "--ablation-only", RUN_TOKEN="inherited")
        assert [t for _, t, _ in jobs] == ["T1-sel", "T1-sel"], jobs
        # the previous run's ablation dirs are gone; nothing else is touched
        assert sorted(os.listdir(res)) == ["best_per_category.json",
                                           "latest_combo_cv1.txt",
                                           "pcavar0.8_go_cv1"]
        # (2026-10-24) and the tables built on them, their rebuild printed
        # after the ablation summary (the last job), with no token flag
        assert not os.path.exists(os.path.join(cmp_res, "paper_tables")) \
            and not os.path.exists(os.path.join(cmp_res, "pca_arms")), out
        summ = jobs[-1][0]
        assert re.search(r"_t_none=\$\(_paste_env PCA_ARM=none sbatch .*--parsable "
                         rf"--dependency=afterany:{summ} --export=ALL .*"
                         r"sl_comparison/slurm/run_paper_tables\.sh\)\n", out) \
            and "--pca_arms_table" not in out, out
        # a pre-token selection file keeps a fresh token, and says so
        with open(os.path.join(res, "best_per_category.json"), "w") as f:
            json.dump({"cv1": {}}, f)
        jobs, out = _submit(tmp, "--ablation-only")
        assert re.fullmatch(r"\d{8}T\d{6}-\d+", jobs[0][1]), jobs
        assert "WARNING" in out, out
        # (its tables rebuild takes that fresh token for tab:loco)
        assert ("run_paper_tables.sh --ablation_run_token "
                f"{jobs[0][1]})\n") in out, out


def test_ablation_only_joins_each_arms_selection_token():
    # --ablation-only ablates every arm of PCA_ARMS from its own selection and
    # joins that selection's token. Arms selected by one submission: that
    # token for the whole submission, the one RUN_TOKEN every job inherits.
    # Arms selected by different submissions: each arm's jobs get theirs in
    # sbatch's environment (PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ...
    # --export=ALL), RUN_TOKEN (what the rest carries, the banner's) is the
    # first arm's, and a WARNING says the table across the
    # arms will flag the others. A selection that records no token keeps a
    # fresh one for its arm (and says which paper_tables flag takes it). An
    # arm with no selection, or no combo marker at all (every task of it
    # would SKIP after its ablation dirs went), stops the submission before
    # anything, with the command that trains what it lacks; a (variant, CV)
    # without a marker is named (its tasks SKIP).
    every = [f"{p}latest_combo_{cv}.txt"
             for p in ("", "var_pd_pooled__", "var_bilinear_per_head__",
                       "var_bilinear_pooled__")
             for cv in ("cv1", "cv2", "cv3")]
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)

        def seed(tokens, markers=None):
            """tokens: {arm: its selection's token ("" none recorded, None
            no selection)}; markers: {arm: its combo markers} (default every
            variant's, every CV's, for an arm with a selection)."""
            res = os.path.join(sia, "results")
            shutil.rmtree(res)
            os.makedirs(res)
            for a, tok in tokens.items():
                r = os.path.join(sia, _root_of(a))
                os.makedirs(os.path.join(r, "ablate_no_go_cv1"))
                if tok is None:
                    continue
                sel = {"cv1": {}}
                if tok:
                    sel["_selection_meta"] = {"run_token": tok}
                with open(os.path.join(r, "best_per_category.json"), "w") as f:
                    json.dump(sel, f)
                for n in (markers or {}).get(a, every):
                    with open(os.path.join(r, n), "w") as f:
                        f.write("results/x_cv1\n")

        def gone(a):
            return not os.path.exists(os.path.join(sia, _root_of(a),
                                                   "ablate_no_go_cv1"))

        seed({"none": "T1-sel", "joint": "T2-sel"})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            RUN_TOKEN="inherited")
        assert _arm_names(jobs) == [
            "run_ablation.sh/none", "run_ablation_summary.sh/none",
            "run_ablation.sh/joint", "run_ablation_summary.sh/joint"], jobs
        assert [t for _, t, _ in jobs] == [
            "T1-sel", "T1-sel", "T2-sel", "T2-sel"], jobs
        assert ("\n  RUN_TOKEN:      T1-sel (PCA arm none's; each arm's jobs get "
                "their own, below)\n") in out, out
        assert "WARNING: the PCA arms to ablate were selected by runs of " \
            "different tokens:\n    none: T1-sel\n    joint: T2-sel\n" in out \
            and "    PCA arm joint: RUN_TOKEN=T2-sel" in out \
            and "has no combo marker" not in out, out
        assert gone("none") and gone("joint"), os.listdir(
            os.path.join(sia, "results"))
        # one submission's selections: its token for every job; an arm
        # PCA_ARMS leaves out keeps its ablation
        seed({"none": "T1-sel", "joint": "T1-sel", "both": "T9"})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint")
        assert {t for _, t, _ in jobs} == {"T1-sel"} \
            and "different tokens" not in out \
            and "records no _selection_meta" not in out, out
        assert gone("none") and gone("joint") and not gone("both")
        # a selection without a token: a fresh one for that arm, and the
        # paper_tables flag that takes it, for that arm
        # (2026-10-21: with the settings taken from the calling shell, the
        # targets in full and the run's arms, which that job reads: pasted
        # bare in a fresh shell, it graded the confs' defaults)
        seed({"none": "T1-sel", "per_modality": ""})
        jobs, out = _submit(tmp, "--ablation-only",
                            PCA_ARMS="none per_modality", SEED="7",
                            ARCH_ARGS="--a --b", PCA_MODALITY_VARIANCE="0.9")
        fresh = jobs[2][1]
        assert re.fullmatch(r"\d{8}T\d{6}-\d+", fresh) \
            and jobs[0][1] == "T1-sel" \
            and [j.arm for j in jobs] == ["none", "none", "per_modality",
                                          "per_modality"], jobs
        hint = (r"env -u RUN_TOKEN ARCH_ARGS=--a\ --b SEED=7 PCA_MODALITY_VARIANCE=0.9 "
                "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM= USE_CELL_LINES=1 "
                'SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled bilinear:per_head '
                'bilinear:pooled" '
                'PCA_ARMS="none per_modality" PCA_ARM=per_modality sbatch '
                "--export=ALL sl_comparison/slurm/run_paper_tables.sh "
                f"--ablation_run_token {fresh}")
        assert ("WARNING: results/pca_per_modality/best_per_category.json "
                "records no _selection_meta.run_token (PCA arm "
                "per_modality);") in out \
            and f"\n    {hint}\n" in out, out
        _env_stubs(tmp)
        # (from a shell exporting a stray RUN_TOKEN too: the hint unsets it,
        # as its own text says to, 2026-10-22)
        rc, o2, calls = _run_printed(tmp, hint, PCA_MODALITY_VARIANCE="0.5",
                                     PCA_ARM="joint", USE_CELL_LINES="0",
                                     RUN_TOKEN="TX")
        assert rc == 0 and [(a.split()[-3], e["PCA_ARM"], e["PCA_ARMS"],
                             e["PCA_MODALITY_VARIANCE"], e["SEED"],
                             e["ARCH_ARGS"], e["USE_CELL_LINES"],
                             e.get("RUN_TOKEN"))
                            for _, a, e in calls] == [
            ("sl_comparison/slurm/run_paper_tables.sh", "per_modality",
             "none per_modality", "0.9", "7", "--a --b", "1", None)], \
            (o2, calls)
        # some (variant, CV) without a marker: submitted, those named
        # (variant-major, in SIAMESE_VARIANTS x CV_TYPES order), per arm
        seed({"none": "T1-sel", "joint": "T1-sel"},
             markers={"none": [m for m in every if m not in (
                 "var_pd_pooled__latest_combo_cv2.txt",
                 "var_bilinear_pooled__latest_combo_cv3.txt")]})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint")
        assert len(jobs) == 4 and gone("none") and gone("joint"), jobs
        assert ("NOTE: PCA arm none has no combo marker "
                "(results/<prefix>latest_combo_<cv>.txt)\n      for: "
                "pd:pooled cv2, bilinear:pooled cv3. Those ablation tasks "
                "will SKIP (no combo to ablate).\n") in out \
            and "PCA arm joint has no combo marker" not in out, out
        # no selection for an arm: refused, nothing removed or queued; the
        # command that trains it names it alone, and keeps the embeddings.
        # Each command alone on its line and complete (2026-10-21): the
        # settings taken from the calling shell (SEED=7 here, one-shot),
        # this shell's targets in full, then its own arms; the others'
        # ablation is this submission again (its flags, --reset-env kept)
        # The retrain line sets its variants (this run's and every other one
        # the arm holds a combo of: it replaces the arm's selection under all
        # of them), after USE_CELL_LINES as this run resolved it, and keeps
        # the venv unless this run asked for a reset.
        t0 = " ".join(_TARGETS0) + " "
        v4 = ('USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled '
              'bilinear:per_head bilinear:pooled" ')
        seed({"none": "T1-sel", "both": None})
        jobs, out = _submit(tmp, "--ablation-only", "--reset-env",
                            PCA_ARMS="none both", SEED="7", _rc=1)
        assert not jobs and ("ERROR: --ablation-only needs the selection of "
                             "every PCA arm it ablates; missing:\n    "
                             "results/pca_both/best_per_category.json  "
                             "(PCA arm both)\n") in out \
            and (f'\n    SEED=7 {t0}{v4}PCA_ARMS="both" bash slurm/submit_'
                 "pipeline.sh --best-cat-only --skip-generate\n") in out \
            and (f'\n  or ablate the others alone:\n    SEED=7 {t0}{v4}PCA_ARMS='
                 '"none" bash slurm/submit_pipeline.sh --ablation-only '
                 "--reset-env\n") in out \
            and "Nothing was changed or submitted." in out, out
        assert not gone("none") and not gone("both")
        # (a variant subset: the others' line pins this run's list, not the
        # four variants arm none holds combos of)
        seed({"none": "T1-sel", "both": None})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none both",
                            SIAMESE_VARIANTS_LIST="pd:per_head", _rc=1)
        assert not jobs and ("\n  or ablate the others alone:\n    "
                             "SIAMESE_VARIANTS_LIST=pd:per_head "
                             f'{t0}USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST='
                             '"pd:per_head" PCA_ARMS="none" bash slurm/'
                             "submit_pipeline.sh --ablation-only\n") in out, out
        # a selection but no combo marker at all (a regeneration removed
        # them): refused the same way, before anything is removed. Arm joint
        # ranks arm none's single-embedding runs, so the retrain is the pair
        # whole (2026-10-24: PCA_ARMS="joint" alone made joint its own Step-1
        # owner) and no arm is left to ablate alone
        seed({"none": "T1-sel", "joint": "T1-sel"}, markers={"joint": []})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            _rc=1)
        assert not jobs and ("ERROR: --ablation-only needs a combo to ablate "
                             "in every PCA arm it ablates; no\n  <prefix>"
                             "latest_combo_<cv>.txt (variants pd:per_head "
                             "pd:pooled bilinear:per_head bilinear:pooled; "
                             "cv1 cv2 cv3) in:\n    results/pca_joint/  "
                             "(PCA arm joint)\n") in out \
            and "both arms of it retrain" in out \
            and _printed_commands(out) == [
                f'{t0}{v4}PCA_ARMS="none joint" bash slurm/submit_pipeline.sh '
                "--best-cat-only --skip-generate --skip-reset-env"] \
            and "the others alone" not in out and "missing:" not in out, out
        assert not gone("none") and not gone("joint")
        # both at once: one command for every arm that lacks either, with
        # its pair partner (both, beside per_modality), and the others alone
        seed({"none": "T1-sel", "per_modality": None, "both": "T1-sel"},
             markers={"none": []})
        jobs, out = _submit(tmp, "--ablation-only",
                            PCA_ARMS="none per_modality both", _rc=1)
        assert not jobs and "results/pca_per_modality/best_per_category.json" \
            "  (PCA arm per_modality)\n" in out \
            and "\n    results/  (PCA arm none)\n" in out \
            and (f'\n    {t0}{v4}PCA_ARMS="none per_modality both" bash slurm/'
                 "submit_pipeline.sh --best-cat-only --skip-generate "
                 "--skip-reset-env\n") \
            in out and "the others alone" not in out, out
        assert not any(gone(a) for a in ("none", "per_modality", "both"))
        seed({"none": "T1-sel"}, markers={"none": []})
        jobs, out = _submit(tmp, "--ablation-only", _rc=1)
        assert not jobs and (f'\n    {t0}{v4}PCA_ARMS="none" bash slurm/'
                             "submit_pipeline.sh --best-cat-only "
                             "--skip-generate --skip-reset-env\n") in out \
            and "the others alone" not in out, out
        # a variant subset in the calling shell, the arm holding combos of
        # another variant only: the retrain lists both (the last assignment
        # wins over the one-shot list in front)
        seed({"none": "T1-sel"},
             markers={"none": [f"var_bilinear_pooled__latest_combo_{cv}.txt"
                               for cv in ("cv1", "cv2", "cv3")]})
        jobs, out = _submit(tmp, "--ablation-only", _rc=1,
                            SIAMESE_VARIANTS_LIST="pd:per_head")
        assert not jobs and ("\n    SIAMESE_VARIANTS_LIST=pd:per_head "
                             f'{t0}USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head '
                             'bilinear:pooled" PCA_ARMS="none" bash slurm/'
                             "submit_pipeline.sh --best-cat-only "
                             "--skip-generate --skip-reset-env\n") in out \
            and "would stay, under a" not in out \
            and "of every other variant they hold a combo of" in out, out
        # two arms without a marker of this run's variant: the line lists
        # every variant either holds a combo of, once, at whatever CV (none:
        # bilinear:per_head cv1-3; joint: bilinear:per_head and
        # bilinear:pooled at cv3 only), and, run as printed from a shell
        # exporting USE_CELL_LINES=0 (2026-10-22: the line did not pin it,
        # and trained the single-output model in place of the cell-line
        # ones), trains them all at USE_CELL_LINES=1, every marker and the
        # selection of both arms removed first
        seed({"none": "T1-sel", "joint": "T1-sel"},
             markers={"none": [f"var_bilinear_per_head__latest_combo_{cv}.txt"
                               for cv in ("cv1", "cv2", "cv3")],
                      "joint": ["var_bilinear_per_head__latest_combo_cv3.txt",
                                "var_bilinear_pooled__latest_combo_cv3.txt"]})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            SIAMESE_VARIANTS_LIST="pd:per_head", _rc=1)
        three = "pd:per_head bilinear:per_head bilinear:pooled"
        cmds = _printed_commands(out)
        assert not jobs and cmds == [
            "SIAMESE_VARIANTS_LIST=pd:per_head " + t0 + "USE_CELL_LINES=1 "
            f'SIAMESE_VARIANTS_LIST="{three}" PCA_ARMS="none joint" bash '
            "slurm/submit_pipeline.sh --best-cat-only --skip-generate "
            "--skip-reset-env"], (cmds, out)
        rc, o2, calls = _run_printed(tmp, cmds[0], USE_CELL_LINES="0")
        assert rc == 0 and calls and {
            (e["SIAMESE_VARIANTS_LIST"], e["USE_CELL_LINES"])
            for _, _, e in calls} == {(three, "1")}, (o2[-2500:], calls)
        for a in ("none", "joint"):
            r = os.path.join(sia, _root_of(a))
            assert not glob.glob(os.path.join(r, "*latest_combo_*")) \
                and not os.path.exists(os.path.join(r,
                                                    "best_per_category.json")), \
                (a, os.listdir(r))
        # USE_CELL_LINES=0 (the calling shell's, on the line once): the line
        # trains the single-output model alone, as pd:per_head, and the
        # arm's cell-line combos are named as left under the new selection,
        # with the two ways out; the one at USE_CELL_LINES=1 holds here: the
        # same --ablation-only, accepted, ablates those combos as they are
        # (2026-10-22: the NOTE pointed at a retrain line never printed)
        seed({"joint": "T1-sel"},
             markers={"joint": [f"var_bilinear_pooled__latest_combo_{cv}.txt"
                                for cv in ("cv1", "cv2", "cv3")]})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint",
                            USE_CELL_LINES="0", _rc=1)
        flat = " ".join(out.split())
        assert not jobs and _printed_commands(out) == [
            "USE_CELL_LINES=0 " + t0 + 'SIAMESE_VARIANTS_LIST="pd:per_head" '
            'PCA_ARMS="joint" bash slurm/submit_pipeline.sh --best-cat-only '
            "--skip-generate --skip-reset-env"] \
            and ("Train them first (their selection, and the single-output "
                 "model's combos, as pd:per_head; the embeddings kept):") \
            in flat and ("The combos of bilinear:pooled they hold (cell-line models) would "
                 "stay, under a selection they were not trained with: remove "
                 "their markers (<arm root>/ var_*__latest_combo_<cv>.txt) "
                 "before that line, or run this --ablation-only at "
                 "USE_CELL_LINES=1 instead (refused, it prints what to submit "
                 "instead; accepted, it ablates those combos as they are).") in flat, out
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint")
        assert _arm_names(jobs) == ["run_ablation.sh/joint",
                                    "run_ablation_summary.sh/joint"], out


def _seed_combos(sia, arm, cfgs, token="T1"):
    """An earlier submission's combos of `arm` in the stub tree: its
    selection (that token), its previous ablation dir, and per
    {(variant prefix, cv): config.json content} a combo dir and the
    <prefix>latest_combo_<cv>.txt marker naming it."""
    r = os.path.join(sia, _root_of(arm))
    os.makedirs(os.path.join(r, "ablate_no_go_cv1"), exist_ok=True)
    with open(os.path.join(r, "best_per_category.json"), "w") as f:
        json.dump({"cv1": {}, "_selection_meta": {"run_token": token}}, f)
    for (pre, cv), cfg in cfgs.items():
        d = f"{_root_of(arm)}/{pre}best_cat_x_{cv}"
        os.makedirs(os.path.join(sia, d), exist_ok=True)
        with open(os.path.join(sia, d, "config.json"), "w") as f:
            json.dump(cfg, f)
        with open(os.path.join(r, f"{pre}latest_combo_{cv}.txt"), "w") as f:
            f.write(d + "\n")


def _job_name(script):
    """The #SBATCH --job-name of one of siamese_sl/slurm's job scripts: what
    squeue lists for a job of it."""
    with open(os.path.join(_SIAMESE, "slurm", script)) as f:
        return re.search(r"^#SBATCH --job-name=(\S+)$", f.read(),
                         re.M).group(1)


def _env_stubs(tmp):
    """Make _stub_tree's stubs show more: sbatch also writes each call's
    whole environment (what the job inherits) to <SB_LOG>.env.<id>, and
    squeue lists the file $SQUEUE_OUT names ("<id> <job name>" lines: the
    jobs a test has had queued so far)."""
    b = os.path.join(tmp, "bin")
    with open(os.path.join(b, "sbatch")) as f:
        text = f.read()
    assert text.count("echo $n\n") == 1, text
    with open(os.path.join(b, "sbatch"), "w") as f:
        f.write(text.replace("echo $n\n",
                             'env > "${SB_LOG}.env.$n"\necho $n\n'))
    with open(os.path.join(b, "squeue"), "w") as f:
        f.write('#!/bin/bash\n[ -z "${SQUEUE_OUT:-}" ] || cat "$SQUEUE_OUT"\n')


def _run_printed(tmp, line, sh="bash", **env):
    """Run one command a refusal printed, as printed (`sh -c LINE`), from
    siamese_sl/ of the stub tree (_env_stubs), in a fresh shell that holds
    none of the refused run's settings (_clean_env; HOME is `tmp`, so no rc
    file of the user's is read); `env` adds the test's own (SQUEUE_OUT).
    Returns (rc, output, [(job id, sbatch args, {the job's environment})])."""
    sia, b = os.path.join(tmp, "siamese_sl"), os.path.join(tmp, "bin")
    e = _clean_env(PATH=b + os.pathsep + os.environ["PATH"], HOME=tmp,
                   SB_LOG=os.path.join(tmp, "pasted.log"),
                   SB_CTR=os.path.join(tmp, "pasted.ctr"), **env)
    for f in glob.glob(e["SB_LOG"] + "*"):
        os.remove(f)
    r = subprocess.run([sh, "-c", line], cwd=sia, env=e, capture_output=True,
                       text=True, timeout=180)
    calls = []
    if os.path.exists(e["SB_LOG"]):
        with open(e["SB_LOG"]) as f:
            for ln in f:
                j, _, a = ln.rstrip("\n").split("|", 2)
                with open(f"{e['SB_LOG']}.env.{j}") as g:
                    jenv = dict(x.split("=", 1) for x in g.read().split("\n")
                                if "=" in x)
                calls.append((j, _plain_export(a), jenv))
    return r.returncode, r.stdout + r.stderr, calls


def _printed_commands(out):
    """The commands a refusal printed: its 4-space-indented lines that run
    submit_pipeline.sh or sbatch, the indent removed. Each must read as ONE
    simple command in any shell (shlex): no token that starts a comment
    (interactive zsh has interactivecomments off) or a subshell."""
    cmds = [ln[4:] for ln in out.splitlines() if ln.startswith("    ")
            and (" bash slurm/submit_pipeline.sh" in ln or " sbatch " in ln)]
    for c in cmds:
        assert not [w for w in shlex.split(c) if w.startswith(("#", "("))], c
    return cmds


def test_ablation_only_refuses_combos_of_another_pca():
    # Every ablation task trains with the PCA this shell gives its arm and
    # stops at its signature check on any other than its reference combo's:
    # --ablation-only compares each marker-named combo's config.json with it
    # first (arch_signature.py --check_pca), and refuses a mismatch before it
    # removes the arm's previous ablation, with what to submit instead, each
    # command complete (the settings taken from the calling shell, the three
    # targets in full, its own variants and arms, nothing after it): ONE
    # when one set of targets trains every arm's combos (a stray or
    # forgotten target, every variant of an arm at one count), else one per
    # arm or per variant group plus the job that rebuilds a split arm's
    # ablation_summary.csv; 0.80 is 0.8; an arm with a combo no target
    # trains in it is retrained whole (first, alone on its line); and with
    # no helper to ask (an empty arch_signature.py) it only warns: every
    # task still checks. test_ablation_only_refusal_runs_as_printed runs the
    # printed commands themselves.
    vs = "pd:per_head pd:pooled"
    pre = r"SIAMESE_VARIANTS_LIST=pd:per_head\ pd:pooled "  # vs, one-shot
    t0 = " ".join(_TARGETS0) + " "
    j12 = {"post_pca": 12, "post_pca_dim": 12}
    j08 = {"post_pca": 0.8}
    every = {("", cv): j12 for cv in ("cv1", "cv2", "cv3")}
    both08 = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8], "post_pca": 0.8}
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        shutil.copy(os.path.join(_SIAMESE, "arch_signature.py"), sia)
        _env_stubs(tmp)

        def seed(cfgs):
            res = os.path.join(sia, "results")
            shutil.rmtree(res)
            os.makedirs(res)
            for arm, c in cfgs.items():
                _seed_combos(sia, arm, c)

        def kept(arm):
            return os.path.isdir(os.path.join(sia, _root_of(arm),
                                              "ablate_no_go_cv1"))

        # joint's pd:per_head combos at 12 components, its pd:pooled one at
        # this shell's 0.8; none's at no PCA: one line per group of joint's
        # variants (each its own targets in full and its own list, after the
        # one-shot one: the last wins), then none's, then the rebuild of
        # joint's ablation_summary.csv under the selection's token
        split = {"none": {("", "cv1"): {}},
                 "joint": {**every, ("var_pd_pooled__", "cv1"): j08}}
        seed(split)
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        assert not jobs and kept("none") and kept("joint"), jobs
        assert ("ERROR: --ablation-only would train the ablations with this "
                "shell's PCA targets,\n    PCA_MODALITY_VARIANCE=0.8 "
                "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=\n") in out \
            and "  PCA arm joint (--post_pca_variance 0.8 here):\n    " \
                "results/pca_joint/best_cat_x_cv1  (pd:per_head: PCA on the " \
                "concatenation of all modalities (12 components))\n" in out \
            and "best_cat_x_cv1  (pd:pooled" not in out, out
        cmds = _printed_commands(out)
        # (each line pins USE_CELL_LINES and its variants, 2026-10-22)
        assert cmds == [
            pre + "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 "
            'PCA_JOINT_DIM=12 USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head" '
            'PCA_ARMS="joint" bash slurm/submit_pipeline.sh --ablation-only',
            pre + t0 + 'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:pooled" '
            'PCA_ARMS="joint" bash slurm/submit_pipeline.sh --ablation-only',
            pre + t0 + f'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            'PCA_ARMS="none" bash slurm/submit_pipeline.sh --ablation-only',
            pre + t0 + 'RUN_TOKEN=T1 PCA_ARMS="joint" PCA_ARM=joint '
            'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled" sbatch '
            "--export=ALL "
            "--output=slurm/logs/ablation_summary_pca_joint_%j.out "
            "--error=slurm/logs/ablation_summary_pca_joint_%j.err "
            "slurm/run_ablation_summary.sh"], cmds
        lines = out.splitlines()
        i = lines.index("    " + cmds[0])
        assert lines[i + 1:i + 3] == ["    " + c for c in cmds[1:3]] \
            and "(this shell's PCA)" not in out \
            and ("  PCA arm joint takes one line per group of its variants, "
                 "and each rewrites\n  results/pca_joint/ablation_summary.csv"
                 " over its own: the file then holds the variants of the\n"
                 "  last line only (no table reads it: tab:loco reads the "
                 "ablation dirs).\n  Once the jobs of every line have "
                 "finished, this writes it for all of them") in out \
            and ("Once squeue\n  lists only those,\n  ALLOW_LIVE_JOBS=1 in "
                 "front of the next line submits it alongside them") in out, \
            out
        assert out.rstrip().endswith("Nothing was changed or submitted."), out
        # with --reset-env: on the first line only (a later line's reset
        # would delete the venv under the earlier ones' jobs), and the prose
        # waits for that reset
        jobs, out = _submit(tmp, "--ablation-only", "--reset-env",
                            PCA_ARMS="none joint", SIAMESE_VARIANTS_LIST=vs,
                            _rc=1)
        cmds = _printed_commands(out)
        assert [c.endswith(" --ablation-only --reset-env") for c in cmds] \
            == [True, False, False, False] and "--reset-env" not in "".join(
                cmds[1:]) and ("lists only those, and the reset_env job of "
                               "the first line has finished,\n") in out, \
            (cmds, out)
        # two arms at targets no one set reproduces (joint's combos at 12
        # components, both's at 150): one line per arm, each with its own
        # targets and pinning this run's variants and USE_CELL_LINES
        # (2026-10-22: from a shell exporting USE_CELL_LINES=0 such a line
        # ablated the single-output model); as printed from that shell, each
        # arm's jobs get them
        # (each arm also holds a bilinear:pooled combo this run does not
        # list: the lines pin this run's variants, not the arms' own)
        b150 = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8], "post_pca": 150,
                "post_pca_dim": 150}
        seed({"none": {("", "cv1"): {}, ("var_bilinear_pooled__", "cv1"): {}},
              "joint": {**every, ("var_bilinear_pooled__", "cv1"): j12},
              "both": {**{("", cv): b150 for cv in ("cv1", "cv2", "cv3")},
                       ("var_bilinear_pooled__", "cv1"): b150}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint both",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        cmds = _printed_commands(out)
        assert not jobs and cmds == [
            pre + "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 "
            f'PCA_JOINT_DIM=12 USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            'PCA_ARMS="joint" bash slurm/submit_pipeline.sh --ablation-only',
            pre + "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 "
            f'PCA_JOINT_DIM=150 USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            'PCA_ARMS="both" bash slurm/submit_pipeline.sh --ablation-only',
            pre + t0 + f'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            'PCA_ARMS="none" bash slurm/submit_pipeline.sh --ablation-only'], \
            (cmds, out)
        rc, o2, calls = _run_printed(tmp, cmds[0], USE_CELL_LINES="0",
                                     SIAMESE_VARIANTS_LIST="bilinear:pooled")
        assert rc == 0 and calls and {
            (e["PCA_ARM"], e["SIAMESE_VARIANTS_LIST"], e["USE_CELL_LINES"],
             e["PCA_JOINT_DIM"]) for _, _, e in calls} == {
            ("joint", vs, "1", "12")}, (o2[-2500:], calls)
        # a selection that records no token: no rebuild command, and why
        seed(split)
        with open(os.path.join(sia, _root_of("joint"),
                               "best_per_category.json"), "w") as f:
            json.dump({"cv1": {}}, f)
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        assert "run_ablation_summary.sh" not in out and len(
            _printed_commands(out)) == 3 and ("Its selection records no run "
                                              "token, so each line keeps a "
                                              "fresh one of its own") in out, \
            out
        # every variant at 12: one set fits, ONE command for the arm (the
        # run's list in its prefix, and pinned: not the bilinear:pooled
        # combo the arm also holds), no rebuild; as printed, accepted
        seed({"joint": {**every, ("var_pd_pooled__", "cv1"): j12,
                        ("var_bilinear_pooled__", "cv1"): j12}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        cmds = _printed_commands(out)
        assert cmds == [
            r"PCA_ARMS=joint SIAMESE_VARIANTS_LIST=pd:per_head\ pd:pooled "
            "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=12 "
            f'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            "bash slurm/submit_pipeline.sh --ablation-only"] \
            and "One set of targets trains the combos of every arm. Submit " \
                "the same again with it:\n" in out, (cmds, out)
        rc, o2, calls = _run_printed(tmp, cmds[0])
        assert rc == 0 and len(calls) == 2 and not kept("joint") and {
            c[2]["PCA_JOINT_DIM"] for c in calls} == {"12"}, (o2, calls)
        # the same with PCA_JOINT_DIM=12 exported: accepted
        seed({"joint": {**every, ("var_pd_pooled__", "cv1"): j12}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint",
                            SIAMESE_VARIANTS_LIST=vs, PCA_JOINT_DIM="12")
        assert len(jobs) == 2 and "ERROR" not in out and not kept("joint"), \
            (jobs, out[-1500:])
        # the same PCA spelled otherwise is the same PCA
        seed({"joint": {("", "cv1"): j08}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint",
                            SIAMESE_VARIANTS_LIST=vs, PCA_JOINT_VARIANCE="0.80")
        assert len(jobs) == 2 and "ERROR" not in out, out[-1500:]
        # a stray target in this shell: one set (this shell's but the stray
        # one) trains every arm's combos: ONE command, the arms of the run
        both08m = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8],
                   "post_pca": 0.8}
        seed({"none": {("", "cv1"): {}},
              "per_modality": {("", "cv1"): {"pca_variance": 0.8,
                                             "pca_dims": [0.8, 0.8]}},
              "both": {("", "cv1"): both08m}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_MODALITY_VARIANCE="0.9",
                            PCA_ARMS="none per_modality both", _rc=1)
        assert not jobs and _printed_commands(out) == [
            r"PCA_ARMS=none\ per_modality\ both " + t0 + 'USE_CELL_LINES=1 '
            'SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled bilinear:per_head '
            'bilinear:pooled" bash slurm/submit_pipeline.sh --ablation-only'], out
        assert all(kept(a) for a in ("none", "per_modality", "both"))
        # a combo of another arm's PCA in this arm's root: no target trains
        # it here, the arm is retrained whole; with another variant of the
        # arm that a target would reproduce, still whole (a retrain of the
        # bad variant alone replaced the selection under the other), first,
        # and the other arms as they are after it
        # (whole: this run's variants and every other one the arm holds a
        # combo of, in the venv the other arms were trained in)
        seed({"joint": {("", "cv1"): both08}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint", _rc=1)
        assert not jobs and _printed_commands(out) == [
            t0 + 'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled '
            'bilinear:per_head bilinear:pooled" PCA_ARMS="joint" bash '
            "slurm/submit_pipeline.sh --best-cat-only --skip-generate "
            "--skip-reset-env"] \
            and ("No targets train some combos of PCA arm(s) joint (the PCA "
                 "of another arm, or CVs\n  that disagree): the "
                 "--best-cat-only line retrains the arm(s) whole") in out \
            and "No one set of targets trains the combos of every arm" \
            not in out and kept("joint"), out
        # (2026-10-24) beside arm none, arm joint ranks none's single-
        # embedding runs: the retrain is the pair whole (PCA_ARMS="joint"
        # alone made joint its own Step-1 owner), and none is not ablated
        # alone after it
        seed({"none": {("", "cv1"): {}},
              "joint": {**every, ("var_pd_pooled__", "cv1"): both08}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        cmds = _printed_commands(out)
        assert not jobs and cmds == [
            pre + t0 + f'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" PCA_ARMS="none joint" bash '
            "slurm/submit_pipeline.sh --best-cat-only --skip-generate "
            "--skip-reset-env"] \
            and "No targets train some combos of PCA arm(s) joint (" in out \
            and "(PCA arm(s) none joint: a pair shares its single-embedding" in out, (cmds, out)
        # an arm whose partner the run leaves out (both, without
        # per_modality) retrains alone, the other arms' ablation after it
        seed({"none": {("", "cv1"): {}}, "both": {("", "cv1"): j08}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none both",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        cmds = _printed_commands(out)
        assert not jobs and cmds == [
            pre + t0 + f'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" PCA_ARMS="both" bash '
            "slurm/submit_pipeline.sh --best-cat-only --skip-generate "
            "--skip-reset-env",
            pre + t0 + f'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            'PCA_ARMS="none" bash slurm/submit_pipeline.sh --ablation-only'] \
            and "the reset_env job of the first line" not in out \
            and ("Once squeue\n  lists only those,\n  ALLOW_LIVE_JOBS=1 in "
                 "front of the next line") in out, (cmds, out)
        # the retrain, as printed: accepted, arm both retrained at this
        # shell's targets over the run's variants, no venv reset
        rc, o2, calls = _run_printed(tmp, cmds[0])
        assert rc == 0 and any(a.endswith("run_best_per_category_combo.sh")
                               and e["PCA_ARM"] == "both"
                               for _, a, e in calls) and {
            (e["SIAMESE_VARIANTS_LIST"], e["PCA_JOINT_VARIANCE"],
             e["PCA_JOINT_DIM"]) for _, _, e in calls} == {(vs, "0.8", "")} \
            and not [a for _, a, _ in calls if a.endswith("/reset_env.sh")], \
            (o2[-2500:], calls)
        # a variant subset in the calling shell over an arm with combos of
        # all four, one of them at another arm's PCA: the retrain line lists
        # all four (until 2026-10-21 it took the subset, and left the other
        # variants' combos and ablations under a selection of another run),
        # and, run as printed, removes every variant's marker and the
        # selection before it retrains them all; with --reset-env the line
        # resets the venv, and the prose waits for that reset
        seed({"joint": {("", "cv1"): j08, ("var_pd_pooled__", "cv1"): both08,
                        ("var_bilinear_per_head__", "cv1"): j08,
                        ("var_bilinear_pooled__", "cv1"): j08}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint",
                            SIAMESE_VARIANTS_LIST=vs, _rc=1)
        cmds = _printed_commands(out)
        assert not jobs and cmds == [
            pre + t0 + 'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled '
            'bilinear:per_head bilinear:pooled" PCA_ARMS="joint" bash '
            "slurm/submit_pipeline.sh --best-cat-only --skip-generate "
            "--skip-reset-env"] and ("retrains the arm(s) whole (the variants\n"
                                     "  of this run and every other variant "
                                     "they hold a combo of)") in out, (cmds, out)
        # (from a shell exporting USE_CELL_LINES=0: the line pins this
        # run's, 2026-10-22)
        rc, o2, calls = _run_printed(tmp, cmds[0], USE_CELL_LINES="0")
        r = os.path.join(sia, _root_of("joint"))
        assert rc == 0 and calls and {(e["SIAMESE_VARIANTS_LIST"],
                                       e["USE_CELL_LINES"])
                                      for _, _, e in calls} == {
            ("pd:per_head pd:pooled bilinear:per_head bilinear:pooled", "1")} \
            and not glob.glob(os.path.join(r, "*latest_combo_*")) \
            and not os.path.exists(os.path.join(r, "best_per_category.json")), \
            (o2[-2500:], calls, os.listdir(r))
        # (with arm none's ablation after it: two lines, the second refused
        # until the first's reset has finished)
        seed({"none": {("", "cv1"): {}}, "both": {("", "cv1"): j08}})
        jobs, out = _submit(tmp, "--ablation-only", "--reset-env",
                            PCA_ARMS="none both", SIAMESE_VARIANTS_LIST=vs,
                            _rc=1)
        cmds = _printed_commands(out)
        assert len(cmds) == 2 and cmds[0].endswith(
            " --best-cat-only --skip-generate") and cmds[1].endswith(
            'PCA_ARMS="none" bash slurm/submit_pipeline.sh --ablation-only') \
            and "--skip-reset-env" not in out \
            and ("Once squeue\n  lists only those, and the reset_env job of the "
                 "first line has finished,\n  ALLOW_LIVE_JOBS=1") in out, \
            (cmds, out)
        # USE_CELL_LINES=0: arm joint's whole retrain is its single-output
        # model alone, as pd:per_head; its cell-line combos are named as left
        seed({"joint": {("", "cv1"): both08,
                        ("var_bilinear_pooled__", "cv1"): j08}})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint",
                            USE_CELL_LINES="0", _rc=1)
        flat = " ".join(out.split())
        assert not jobs and _printed_commands(out) == [
            "USE_CELL_LINES=0 " + t0 + 'SIAMESE_VARIANTS_LIST="pd:per_head" '
            'PCA_ARMS="joint" bash slurm/submit_pipeline.sh --best-cat-only '
            "--skip-generate --skip-reset-env"] \
            and ("the --best-cat-only line retrains the arm(s)' selection and "
                 "the single-output model (USE_CELL_LINES=0, as pd:per_head), "
                 "at the targets of this shell, and ablates it.") in flat \
            and "retrains the arm(s) whole" not in flat \
            and ("The combos of bilinear:pooled they hold (cell-line models) would "
                 "stay, under a selection they were not trained with: remove "
                 "their markers (<arm root>/ var_*__latest_combo_<cv>.txt) "
                 "before that line, or run this --ablation-only at "
                 "USE_CELL_LINES=1 instead (refused, it prints what to submit "
                 "instead; accepted, it ablates those combos as they are).") in flat, out
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint", _rc=1)
        assert not jobs and _printed_commands(out) == [
            t0 + 'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled '
            'bilinear:per_head bilinear:pooled" PCA_ARMS="joint" bash '
            "slurm/submit_pipeline.sh --best-cat-only --skip-generate "
            "--skip-reset-env"] \
            and "retrains the arm(s) whole" in out \
            and "would stay, under a" not in out, out
        # no helper here (an older or missing arch_signature.py): a WARNING,
        # and the tasks check
        open(os.path.join(sia, "arch_signature.py"), "w").close()
        seed({"joint": every})
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="joint")
        assert len(jobs) == 2 and "WARNING: could not compare each PCA arm's " \
            "PCA with its combos' config.json" in out and "ERROR" not in out, \
            out[-1500:]


def test_ablation_only_refusal_runs_as_printed():
    # What the refusal prints is run, here, as printed (bash, and zsh where
    # installed) from a shell that never held the refused run's one-shot
    # settings (SEED=7, ARCH_ARGS="--a --b", a variant subset, a target),
    # and that exports PCA targets of its own (each command sets all three):
    # each command does what its prose says. Until 2026-10-21 none carried
    # them: arm both's line, pasted, was refused again with the other target
    # (endlessly), and an accepted one removed the arm's ablation at seed 42
    # and queued tasks that stop at their signature check.
    # 1. One target forgotten (every arm trained at MV 0.9 / JD 150, the
    #    refused run given MV 0.9 alone): ONE command, accepted, every job of
    #    every arm with the run's seed, ARCH_ARGS, variants and both targets,
    #    each ablation array sized for the two variants, and only those
    #    variants' previous ablation removed.
    # 2. Arms that need submissions of their own (joint's pd:per_head
    #    combos at 12 components, its pd:pooled ones at 0.8): run in order
    #    with squeue listing the jobs each line queued, every later line is
    #    refused while those are live (job names carry no arm) and accepted
    #    with ALLOW_LIVE_JOBS=1 in front, each with its own targets,
    #    variants and array, each removing only its own variants' ablation;
    #    then the split arm's printed rebuild submits its summary over both
    #    variants under the selection's token.
    vs = "pd:per_head pd:pooled"
    m9 = {"pca_variance": 0.9, "pca_dims": [0.9, 0.9]}
    j150 = {"post_pca": 150, "post_pca_dim": 150}
    j12 = {"post_pca": 12, "post_pca_dim": 12}
    one_shot = {"SEED": "7", "ARCH_ARGS": "--a --b",
                "SIAMESE_VARIANTS_LIST": vs}
    # the pasting shell's own targets and USE_CELL_LINES, which no printed
    # command may inherit (each line pins the run's, 2026-10-22: under a
    # stray USE_CELL_LINES=0 the lines ablated the single-output model, its
    # tasks stopping at their signature check after the dirs were removed)
    stray = {"PCA_MODALITY_VARIANCE": "0.5", "PCA_JOINT_VARIANCE": "0.6",
             "PCA_JOINT_DIM": "999", "USE_CELL_LINES": "0"}
    shells = ["bash"] + [z for z in (shutil.which("zsh"),) if z]
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        shutil.copy(os.path.join(_SIAMESE, "arch_signature.py"), sia)
        _env_stubs(tmp)

        def seed(state):
            """{arm: {variant prefix: config.json}} for cv1-3, and a previous
            ablation dir of pd:per_head, pd:pooled and bilinear:pooled."""
            res = os.path.join(sia, "results")
            shutil.rmtree(res)
            os.makedirs(res)
            for arm, by in state.items():
                _seed_combos(sia, arm, {(p, cv): c for p, c in by.items()
                                        for cv in ("cv1", "cv2", "cv3")})
                for p in ("var_pd_pooled__", "var_bilinear_pooled__"):
                    os.makedirs(os.path.join(sia, _root_of(arm),
                                             p + "ablate_no_go_cv1"))

        def left(arm):
            """The variants whose previous ablation dir is still there."""
            r = os.path.join(sia, _root_of(arm))
            return [p for p in ("", "var_pd_pooled__", "var_bilinear_pooled__")
                    if os.path.isdir(os.path.join(r, p + "ablate_no_go_cv1"))]

        def arrays(calls):
            return [re.search(r"--array=(\S+)", a).group(1)
                    for _, a, _ in calls if a.endswith("/run_ablation.sh")]

        # 1. one target forgotten
        forgot = {"none": {"": {}, "var_pd_pooled__": {}},
                  "joint": {"": j150, "var_pd_pooled__": j150},
                  "per_modality": {"": m9, "var_pd_pooled__": m9},
                  "both": {"": {**m9, **j150}, "var_pd_pooled__": {**m9, **j150}}}
        seed(forgot)
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS=None, _rc=1,
                            PCA_MODALITY_VARIANCE="0.9", **one_shot)
        cmds = _printed_commands(out)
        assert not jobs and cmds == [
            r"ARCH_ARGS=--a\ --b SIAMESE_VARIANTS_LIST=pd:per_head\ pd:pooled "
            'SEED=7 PCA_ARMS="none joint per_modality both" '
            "PCA_MODALITY_VARIANCE=0.9 PCA_JOINT_VARIANCE=0.8 "
            f'PCA_JOINT_DIM=150 USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{vs}" '
            "bash slurm/submit_pipeline.sh --ablation-only"], (cmds, out)
        for sh in shells:
            seed(forgot)
            rc, o2, calls = _run_printed(tmp, cmds[0], sh, **stray)
            assert rc == 0 and "ERROR" not in o2 and [
                (e["PCA_ARM"], a.rsplit("/", 1)[1]) for _, a, e in calls] == [
                (arm, s) for arm in _ARMS for s in ("run_ablation.sh",
                                                    "run_ablation_summary.sh")], \
                (sh, o2[-2500:], calls)
            assert {(e["SEED"], e["ARCH_ARGS"], e["SIAMESE_VARIANTS_LIST"],
                     e["PCA_MODALITY_VARIANCE"], e["PCA_JOINT_VARIANCE"],
                     e["PCA_JOINT_DIM"], e["RUN_TOKEN"], e["USE_CELL_LINES"])
                    for _, _, e in calls} == {
                ("7", "--a --b", vs, "0.9", "0.8", "150", "T1", "1")}, \
                (sh, calls)
            assert arrays(calls) == ["0-35"] * 4 and all(
                left(a) == ["var_bilinear_pooled__"] for a in _ARMS), \
                (sh, arrays(calls), [left(a) for a in _ARMS])

        # 2. one submission per group of joint's variants, then none's; the
        # refused run given ALLOW_LIVE_JOBS=1 one-shot, which no line may
        # carry (pasted as a block, each later line would pass a real
        # conflict): the later-line refusals below then see a mutant that
        # kept it
        split = {"none": {"": {}, "var_pd_pooled__": {}},
                 "joint": {"": j12, "var_pd_pooled__": {"post_pca": 0.8}}}
        seed(split)
        jobs, out = _submit(tmp, "--ablation-only", PCA_ARMS="none joint",
                            _rc=1, ALLOW_LIVE_JOBS="1", **one_shot)
        cmds = _printed_commands(out)
        assert not jobs and len(cmds) == 4 \
            and cmds[3].endswith(" slurm/run_ablation_summary.sh") \
            and not [c for c in cmds if "ALLOW_LIVE_JOBS" in c], cmds
        # per line: (arm, its jobs' SIAMESE_VARIANTS_LIST, PCA_JOINT_DIM,
        # array, the previous ablation dirs of the arm left after it)
        want = [("joint", "pd:per_head", "12", "0-17",
                 ["var_pd_pooled__", "var_bilinear_pooled__"]),
                ("joint", "pd:pooled", "", "0-17", ["var_bilinear_pooled__"]),
                ("none", vs, "", "0-35", ["var_bilinear_pooled__"])]
        for sh in shells:
            seed(split)
            queue = os.path.join(tmp, "queue")
            open(queue, "w").close()
            for k, (cmd, (arm, lst, dim, arr, kept_after)) in enumerate(
                    zip(cmds, want)):
                if k:
                    rc, o2, calls = _run_printed(tmp, cmd, sh,
                                                 SQUEUE_OUT=queue, **stray)
                    assert rc == 1 and not calls and (
                        "ERROR: jobs of an earlier submission are still "
                        "queued or running:\n") in o2 \
                        and "Nothing was changed." in o2, (sh, k, o2[-2000:])
                    cmd = "ALLOW_LIVE_JOBS=1 " + cmd
                rc, o2, calls = _run_printed(tmp, cmd, sh, SQUEUE_OUT=queue,
                                             **stray)
                assert rc == 0 and len(calls) == 2 and {
                    (e["PCA_ARM"], e["SIAMESE_VARIANTS_LIST"],
                     e["PCA_JOINT_DIM"], e["SEED"], e["ARCH_ARGS"],
                     e["USE_CELL_LINES"])
                    for _, _, e in calls} == {
                    (arm, lst, dim, "7", "--a --b", "1")} \
                    and arrays(calls) == [arr] and left(arm) == kept_after, \
                    (sh, k, o2[-2500:], calls, left(arm))
                with open(queue, "a") as f:
                    for j, a, _ in calls:
                        f.write(f"{j} {_job_name(a.rsplit('/', 1)[1])}\n")
            # the rebuild of joint's ablation_summary.csv, once they are done
            rc, o2, calls = _run_printed(tmp, cmds[3], sh, **stray)
            assert rc == 0 and [(a.rsplit("/", 1)[1], e["PCA_ARM"],
                                 e["RUN_TOKEN"], e["SIAMESE_VARIANTS_LIST"],
                                 e["SEED"], e["USE_CELL_LINES"])
                                for _, a, e in calls] == [
                ("run_ablation_summary.sh", "joint", "T1", vs, "7", "1")], \
                (sh, o2, calls)


def _names(jobs):
    """The script basenames of _submit's sbatch calls, in submission order
    (the single-arm contracts: one call per script)."""
    return [a.rsplit("/", 1)[-1] for _, _, a in jobs]


def _arm_of(job):
    """The arm one stub sbatch call of _submit (a _Job) hands its job: the
    PCA_ARM of the environment it was submitted with (PCA_ARM=<arm> in front
    of sbatch, _as_arm), '' for a job of every arm. Such a job (reset_env,
    the generation) passes none of its own: it gets the orchestrator's
    exported PCA_ARM=none (config.conf's default), what a job without one
    resolves anyway."""
    script = job[2].rsplit("/", 1)[-1]
    if script in _SHARED_SCRIPTS:
        assert job.arm == "none", job
        return ""
    assert job.arm in _ARMS, (job, job.arm)
    return job.arm


def _arm_names(jobs):
    """<script>/<arm> per sbatch call of _submit (the arm it hands its
    job), the bare <script> for a job of every arm; in submission order."""
    out = []
    for j in jobs:
        arm = _arm_of(j)
        s = j[2].rsplit("/", 1)[-1]
        out.append(f"{s}/{arm}" if arm else s)
    return out


def _by_arm(jobs):
    """{<script>/<arm> (or <script>): (job id, args)}, each queued once."""
    names = _arm_names(jobs)
    assert len(set(names)) == len(names), names
    return {n: (j, a) for n, (j, _, a) in zip(names, jobs)}


def _owners(arms):
    """The owner of each pair's single-embedding runs among `arms`
    (config.conf pca_step1_arm), in canonical order."""
    out = []
    for a in arms:
        pair = (("per_modality", "both") if a in ("per_modality", "both")
                else ("none", "joint"))
        o = next(p for p in pair if p in arms)
        if o not in out:
            out.append(o)
    return out


def _owner(arm, arms):
    pair = (("per_modality", "both") if arm in ("per_modality", "both")
            else ("none", "joint"))
    return next(p for p in pair if p in arms)


def _tag(arm):
    """The arm in a log file name (config.conf pca_arm_dir + '_')."""
    return "" if arm == "none" else f"pca_{arm}_"


def _root_of(arm):
    """An arm's siamese results root, relative to siamese_sl/."""
    return "results" if arm == "none" else f"results/pca_{arm}"


def _cmp_of(arm):
    """An arm's sl_comparison results root, relative to the repo root."""
    return ("sl_comparison/results" if arm == "none"
            else f"sl_comparison/results/pca_{arm}")


def test_best_cat_submission_moves_replaced_combos_rows_aside():
    # The combo this submits replaces the previous one, and no job queued
    # here rebuilds the sl_comparison rows or reruns the cell-line ablation.
    # Left in place, paper_tables would publish the PREVIOUS combos' scores
    # next to this run's parameter counts, PCA widths and tab:loco reference,
    # and pair the stale *_nocell rows into tab:cellline.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        os.makedirs(os.path.join(sia, "results", "cellline_ablation_cv1"))
        scr = os.path.join(tmp, "sl_comparison", "results")
        siamese = ("siamese_sl_nocell_cv1.json", "siamese_sl_cv1.json",
                   "siamese_sl_pd_pooled_nocell_cv2.json")
        for n in siamese + ("slgnn_cv1.json",):
            open(os.path.join(scr, n), "w").write("{}")
        _submit(tmp, "--skip-generate", "--skip-reset-env", "--best-cat-only")
        assert not os.path.exists(
            os.path.join(sia, "results", "cellline_ablation_cv1"))
        # every siamese row moved aside (timestamped), other models' kept
        assert sorted(os.listdir(scr)) == ["slgnn_cv1.json", "stale_siamese"]
        moved = sorted(n.split(".")[0]
                       for n in os.listdir(os.path.join(scr, "stale_siamese")))
        assert moved == sorted(n[:-len(".json")] for n in siamese), moved


def _sp_block(out):
    """The ready-to-paste block a best-per-category submit_pipeline.sh run
    ends with: its lines, from the _paste_env definition to the last sbatch,
    the 6-space indent removed."""
    lines = out.splitlines()
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("      _paste_env() {"))
    j = max(k for k, ln in enumerate(lines)
            if ln.startswith("      ") and "sbatch " in ln)
    assert all(ln.startswith("      ") for ln in lines[i:j + 1]), lines[i:j + 1]
    return [ln[6:] for ln in lines[i:j + 1]]


def test_best_cat_run_prints_its_sl_comparison_jobs_wired():
    # A best-per-category submission removes each arm's sl_comparison
    # outputs and the table across the arms, and queues no job that rebuilds
    # them: it ends with them, ready to paste and wired to its own jobs as
    # the root orchestrator chains them (until 2026-10-21 step 0a printed
    # them unwired, "once the combos land": the external transfer then ran
    # before the eval, the paper tables beside the compare). Per arm: the
    # post-hoc cell-line ablation after its combo (afterok, kill; the NOTE
    # under each combo points here, no command of its own), the compare after
    # the combo and that job, the external transfer after the eval (none
    # without one), the paper tables after all of them and the ablation
    # summary, logs named after the arm; then the table across the arms after
    # every arm's paper tables, as the first arm. Pasted (bash 3.2, zsh) into
    # a shell that exports a stray token, manifest, compare pins, arms,
    # targets, variants and the other settings those jobs read (selection
    # variant, Step-1 tag, BESTCAT_EXTRA_VARIANTS, ARCH_ARGS, CVS), every job
    # gets none of the former and this run's values of the rest (resolved:
    # its defaults here, an empty one set empty) from the block's first line;
    # a left-out cell-line line leaves the compare on the combo alone. Printed
    # on state: a resubmission with nothing left to remove prints it too.
    # What the block does not rebuild is said so at step 0a: with one arm
    # the table across the arms, without the eval external_transfer/.
    polluted = {"RUN_TOKEN": "20261001T090000-777",
                "CODE_MANIFEST": "/old/code.sha256",
                "SIAMESE_RESULTS_cv1": "x/results.json",
                "SIAMESE_BASELINE_DISCOVER": "all", "PCA_ARMS": "joint",
                "PCA_JOINT_DIM": "999", "PCA_MODALITY_VARIANCE": "0.5",
                "SIAMESE_VARIANTS_LIST": "pd:pooled", "USE_CELL_LINES": "0",
                "SELECTION_VARIANT": "bilinear:pooled",
                "BEST_CAT_PCA_VARIANCE": "0.5", "BESTCAT_EXTRA_VARIANTS": "1",
                "ARCH_ARGS": "--from-the-pasting-shell", "CVS": "cv2"}
    seen = ("RUN_TOKEN", "CODE_MANIFEST", "SIAMESE_RESULTS_cv1",
            "SIAMESE_BASELINE_DISCOVER", "PCA_ARM", "PCA_ARMS",
            "PCA_MODALITY_VARIANCE", "PCA_JOINT_VARIANCE", "PCA_JOINT_DIM",
            "SIAMESE_VARIANTS_LIST", "USE_CELL_LINES", "SELECTION_VARIANT",
            "BEST_CAT_PCA_VARIANCE", "BESTCAT_EXTRA_VARIANTS", "ARCH_ARGS",
            "CVS")
    every = "pd:per_head pd:pooled bilinear:per_head bilinear:pooled"
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        real = os.path.realpath(tmp)    # (as the script resolves its dirs)
        sb = os.path.join(tmp, "bin", "sbatch")
        with open(sb) as f:
            text = f.read()
        with open(sb, "w") as f:   # what each pasted job would inherit, and
            f.write(text.replace(  # the directory it is submitted from
                'echo $n\n', 'echo "$n|$PWD|' + "|".join(
                    "${%s-<unset>}" % v for v in seen)
                + '" >> "${SB_LOG}.seen"\necho $n\n', 1))

        def paste(block, sh):
            """Run the pasted lines; (call args, what each job inherits)."""
            log, ctr = os.path.join(tmp, "paste.log"), os.path.join(tmp, "pc")
            for f in (log, log + ".seen", log + ".arm", log + ".pairenv", ctr):
                if os.path.exists(f):
                    os.remove(f)
            r = subprocess.run(
                [sh, "-s"], input="\n".join(block) + "\n", cwd=tmp,
                capture_output=True, text=True, timeout=60,
                env=dict(polluted, HOME=tmp, SB_LOG=log, SB_CTR=ctr,
                         PATH=os.path.join(tmp, "bin") + os.pathsep
                         + os.environ["PATH"]))
            assert r.returncode == 0 and not r.stderr, (sh, r.stdout, r.stderr)
            with open(log) as f:
                calls = [ln.rstrip("\n").split("|", 2) for ln in f]
            with open(log + ".seen") as f:
                inherited = [dict(zip(("id", "pwd") + seen,
                                      ln.rstrip("\n").split("|")))
                             for ln in f]
            return calls, inherited

        shells = ["/bin/bash"] + [z for z in (shutil.which("zsh"),) if z]
        for arms_env, arms, flags in (
                (None, list(_ARMS), ()),
                ("both per_modality", ["per_modality", "both"],
                 ("--skip-eval",)),
                ("none", ["none"], ("--skip-ablation",))):
            for a in arms:
                os.makedirs(os.path.join(tmp, _cmp_of(a), "paper_tables"),
                            exist_ok=True)
            os.makedirs(os.path.join(tmp, "sl_comparison", "results",
                                     "pca_arms"), exist_ok=True)
            jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                                "--best-cat-only", *flags, PCA_ARMS=arms_env)
            for a in arms:
                assert f"previous sl_comparison output(s) of PCA arm {a} " \
                    in out, (a, out)
            assert "once the combos land" not in out and \
                "the jobs that rebuild them, wired\n  to this submission's, " \
                "are printed at the end." in out, out
            assert not os.path.exists(os.path.join(
                tmp, "sl_comparison", "results", "pca_arms"))
            # what the block below does not rebuild is said so: the table
            # across the arms with one arm, the external transfer without
            # the eval
            assert ("No job rebuilds the table across the arms: one PCA "
                    "arm has nothing to compare" in out) == (len(arms) < 2) \
                and ("Not the external-transfer tables: with --skip-eval"
                     in out) == ("--skip-eval" in flags), out
            # and so does the block's own prose (it said external transfer
            # whatever the flags)
            assert ("to rebuild every arm's comparison, external\n    "
                    "transfer and paper tables" in out) == (
                "--skip-eval" not in flags) and (
                "to rebuild every arm's comparison and paper\n    tables (not "
                "the external transfer: with --skip-eval there is no eval to "
                "build it\n    from)" in out) == ("--skip-eval" in flags), out
            by = _by_arm(jobs)
            block = _sp_block(out)
            assert block[0] == _paste_def(
                "PCA_ARMS=" + " ".join(arms), *_TARGETS0,
                "SIAMESE_VARIANTS_LIST=" + every, "USE_CELL_LINES=1",
                "SELECTION_VARIANT=pd:per_head", "BEST_CAT_PCA_VARIANCE=0.8",
                "BESTCAT_EXTRA_VARIANTS=0", "ARCH_ARGS=", "CVS="), block[0]
            assert block[1] == "cd %s && _L=%s/sl_comparison/slurm/logs && %s" % (
                real, real, " ".join(f"_l_{a}=" for a in arms)), block[1]
            # the cell-line ablation has this one command: the NOTE under
            # each combo points at it
            assert len([ln for ln in out.splitlines()
                        if "run_cell_line_ablation.sh" in ln]) == len(arms) \
                and all(f"the _l_{a} line of the block printed at the end"
                        in out for a in arms), out
            for sh in shells:
                calls, inh = paste(block, sh)
                per = 3 + ("--skip-eval" not in flags)
                assert len(calls) == per * len(arms) + (len(arms) > 1), (
                    sh, calls)
                for job in inh:      # none of the shell's; the run's own
                    assert [job[v] for v in seen[:4]] == ["<unset>"] * 4 \
                        and tuple(job[v] for v in seen[5:]) == (
                            " ".join(arms), "0.8", "0.8", "", every, "1",
                            "pd:per_head", "0.8", "0", "", ""), (sh, job)
                for c in calls:
                    _plain_export(c[2])
                k = 0
                for a in arms:
                    combo = by[f"run_best_per_category_combo.sh/{a}"][0]
                    cell, cmp_ = calls[k], calls[k + 1]
                    assert os.path.realpath(inh[k]["pwd"]) == os.path.join(
                        real, "siamese_sl") and os.path.realpath(
                        inh[k + 1]["pwd"]) == real, (sh, inh[k:k + 2])
                    assert cell[2].endswith(
                        "cell_line_ablation/run_cell_line_ablation.sh") \
                        and (f"--dependency=afterok:{combo} "
                             "--kill-on-invalid-dep=yes ") in cell[2] \
                        and ("--output=slurm/logs/cellline_ablation_pca_"
                             f"{a}_%j.out" in cell[2]) == (a != "none"), \
                        (a, cell)
                    assert (f"--dependency=afterany:{combo}:{cell[0]} " in
                            cmp_[2]) and cmp_[2].endswith(
                        "sl_comparison/slurm/run_compare.sh") and (
                        f"--output={real}/sl_comparison/slurm/logs/final_"
                        f"compare_{_tag(a)}%j.out") in cmp_[2], (a, cmp_)
                    deps = [cmp_[0], cell[0]]
                    if "--skip-eval" not in flags:
                        ev = by[f"eval_finetuning.sh/{a}"][0]
                        ext = calls[k + 2]
                        assert f"--dependency=afterany:{ev} " in ext[2] \
                            and ext[2].endswith("run_external_transfer.sh"), \
                            (a, ext)
                        deps += [ext[0], ev]
                    if "--skip-ablation" not in flags:
                        deps.append(by[f"run_ablation_summary.sh/{a}"][0])
                    tab = calls[k + per - 1]
                    assert f"--dependency=afterany:{':'.join(deps)} " in \
                        tab[2] and tab[2].endswith(
                            "sl_comparison/slurm/run_paper_tables.sh"), (a, tab)
                    assert all(inh[k + n]["PCA_ARM"] == a
                               for n in range(per)), (sh, a, inh[k:k + per])
                    k += per
                if len(arms) > 1:
                    x = calls[-1]
                    tables = [calls[per * n + per - 1][0]
                              for n in range(len(arms))]
                    assert x[2].endswith("run_paper_tables.sh "
                                         "--pca_arms_table") \
                        and f"--dependency=afterany:{':'.join(tables)} " in \
                        x[2] and inh[-1]["PCA_ARM"] == arms[0], (sh, x)
            # the cell-line lines left out: the compare waits on the combo
            calls, _ = paste([ln for ln in block if not ln.startswith(
                tuple(f"_l_{a}=$(" for a in arms))], "/bin/bash")
            combo = by[f"run_best_per_category_combo.sh/{arms[0]}"][0]
            assert f"--dependency=afterany:{combo} " in calls[0][2] \
                and calls[0][2].endswith("run_compare.sh"), calls[0]
        # printed on state: a resubmission, nothing left to remove
        jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                            "--best-cat-only", PCA_ARMS="none")
        assert "previous sl_comparison output" not in out \
            and _sp_block(out)[2].startswith("_l_none=$(cd siamese_sl && "), \
            out[-3000:]
        # no cell-line ablation without the cell-line model: no _l_ lines,
        # and the NOTE says so
        jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                            "--best-cat-only", PCA_ARMS="none",
                            USE_CELL_LINES="0")
        block = _sp_block(out)
        assert not [ln for ln in block if "_l_" in ln] and block[1].endswith(
            "/sl_comparison/slurm/logs") and "no post-hoc cell-line " \
            "ablation: no per_head variant with USE_CELL_LINES=1" in out, block
        # nothing trained here: nothing to rebuild, no block
        jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                            "--benchmark-only", PCA_ARMS="none")
        assert "_paste_env" not in out, out[-2000:]


# An earlier run's outputs in one PCA arm's two roots (the siamese results
# root, the sl_comparison one), by name ("/" = a directory): what the two
# orchestrators remove or move aside at submission, and what they leave (the
# combo directories, which the combo job moves aside itself).
_ARM_SIA = ("best_per_category.json", "latest_combo_cv1.txt",
            "var_pd_pooled__latest_combo_cv3.txt", "best_cat_go_kg_cv1/",
            "ablate_no_go_cv1/", "var_pd_pooled__ablate_no_kg_cv2/",
            "ablation_summary.csv", "cellline_ablation_cv1/",
            "pcavar0.8_go_cv1/")
_ARM_ROWS = ("siamese_sl_cv1.json", "siamese_sl_pd_pooled_cv2.json",
             "siamese_selected_cv1.txt", "siamese_sl_nocell_cv1.json")
_ARM_CMP = _ARM_ROWS + ("comparison.csv", "comparison_notes.txt",
                        "external_transfer/", "paper_tables/")


def _seed_arms(base, arms=_ARMS):
    """Every listed arm's earlier outputs under base (the stub tree or the
    root copy), plus a shared row and the table across the arms; whatever
    base held under the two results roots goes first."""
    for rel in ("siamese_sl/results", "sl_comparison/results"):
        p = os.path.join(base, rel)
        if os.path.isdir(p):
            subprocess.run(["chmod", "-R", "u+w", p], check=True)
            shutil.rmtree(p)
        os.makedirs(p)
    for a in arms:
        for root, names in ((os.path.join(base, "siamese_sl", _root_of(a)),
                             _ARM_SIA),
                            (os.path.join(base, _cmp_of(a)), _ARM_CMP)):
            for n in names:
                p = os.path.join(root, n.rstrip("/"))
                if n.endswith("/"):
                    os.makedirs(os.path.join(p, "x"))
                else:
                    os.makedirs(root, exist_ok=True)
                    with open(p, "w") as f:
                        f.write("{}")
    with open(os.path.join(base, "sl_comparison", "results",
                           "SLGNN_cv1.json"), "w") as f:
        f.write("{}")
    os.makedirs(os.path.join(base, "sl_comparison", "results", "pca_arms"))


def _arm_state(base):
    """{arm: (its siamese root's names, its sl_comparison root's names, the
    stems moved into that root's stale_siamese/)}: another arm's root inside
    arm none's is not arm none's."""
    def names(p):
        if not os.path.isdir(p):
            return []
        return sorted(n + ("/" if os.path.isdir(os.path.join(p, n)) else "")
                      for n in os.listdir(p) if not n.startswith("pca_"))
    out = {}
    for a in _ARMS:
        c = os.path.join(base, _cmp_of(a))
        stale = os.path.join(c, "stale_siamese")
        out[a] = (names(os.path.join(base, "siamese_sl", _root_of(a))),
                  names(c),
                  sorted(re.sub(r"\.\d{8}T\d{6}(\.\w+)$", r"\1", n)
                         for n in (os.listdir(stale)
                                   if os.path.isdir(stale) else [])))
    return out


def test_submit_pipeline_cleanup_stays_in_each_arms_roots():
    # A best-per-category submission removes, at submission, what it rebuilds
    # for the arms of PCA_ARMS only, each in its own roots: markers,
    # selection, ablation and cell-line dirs; rows and markers moved into
    # that arm's own stale_siamese/ (every arm's have the same names);
    # comparison table, external transfer and paper tables; and the table
    # across the arms. Another arm's are never touched (arm none's root
    # holds the others' roots: no glob may reach into them, nor theirs into
    # arm none's), but a regeneration drops every arm's combo markers.
    # The single-embedding runs are left to their jobs.
    kept_sia = sorted({"best_cat_go_kg_cv1/", "pcavar0.8_go_cv1/"})
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        _seed_arms(tmp)
        seeded = _arm_state(tmp)
        for arms_env, arms in (("joint", ["joint"]), ("none", ["none"]),
                               ("per_modality both", ["per_modality",
                                                      "both"])):
            _seed_arms(tmp)
            jobs, _ = _submit(tmp, "--skip-generate", "--skip-reset-env",
                              PCA_ARMS=arms_env)
            assert jobs, arms_env
            got = _arm_state(tmp)
            for a in _ARMS:
                if a not in arms:
                    assert got[a] == seeded[a], (arms_env, a, got[a])
                    continue
                cmp_left = ["stale_siamese/"] + (["SLGNN_cv1.json"]
                                                 if a == "none" else [])
                assert got[a] == (kept_sia, sorted(cmp_left),
                                  sorted(_ARM_ROWS)), (arms_env, a, got[a])
            assert not os.path.exists(os.path.join(
                tmp, "sl_comparison", "results", "pca_arms")), arms_env
        # --ablation-only: only its arms' ablation dirs and summaries, and
        # (2026-10-24) their paper tables, built on those (tab:loco)
        _seed_arms(tmp)
        jobs, _ = _submit(tmp, "--ablation-only", PCA_ARMS="both")
        assert jobs
        got = _arm_state(tmp)
        assert got["both"][0] == sorted(set(_ARM_SIA) - {
            "ablate_no_go_cv1/", "var_pd_pooled__ablate_no_kg_cv2/",
            "ablation_summary.csv"}) \
            and got["both"][1] == sorted(set(seeded["both"][1]) - {"paper_tables/"}) \
            and got["both"][2] == seeded["both"][2], got["both"]
        assert {a: got[a] for a in _ARMS if a != "both"} == \
            {a: seeded[a] for a in _ARMS if a != "both"}, got
        # a regeneration: every arm's combo markers go, the rest as above
        _seed_arms(tmp)
        jobs, out = _submit(tmp, "--skip-reset-env", PCA_ARMS="joint")
        assert jobs and "combo marker(s) of variants or PCA arms this run " \
            "does not" in out, out[-1500:]
        got = _arm_state(tmp)
        markers = {"latest_combo_cv1.txt", "var_pd_pooled__latest_combo_cv3.txt"}
        for a in ("none", "per_modality", "both"):
            assert got[a] == (sorted(set(seeded[a][0]) - markers),
                              ) + seeded[a][1:], (a, got[a])


def test_combo_job_removes_what_the_replaced_combos_produced():
    # A combo job replaces its variants' combos under the SAME directory
    # names (the winners fix them), so every path-based check downstream still
    # matches the retrained ones. Both orchestrators remove what was computed
    # from the previous combos at submission; a combo job submitted by hand
    # (its own "resubmit this job with the same RUN_TOKEN" / "Retrain them
    # before publishing") has to do it itself, or paper_tables prints the
    # replaced models' rows under PAPER COMPLETE. Only once the selection has
    # succeeded (a failed Step 2 keeps the previous combos and their rows),
    # only the listed variants' rows and dirs, rows moved aside, not deleted.
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_best_per_category_combo.sh")) as f:
        text = f.read()
    i = text.index('_CMP_RES="$BASE_DIR/$CMP_ROOT"')
    j = text.index("# STEP 3: Train multi-modal combo", i)
    assert text.index('[ -f "$RESULTS_ROOT/best_per_category.json" ] ||') \
        < text.index('echo "Deleted previous combos') < i
    block = text[i:text.rindex("\n# ====", i, j)]

    def run(tmp, variants, arm="none"):
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "PCA_", "POST_PCA_"))
             and k not in ("SIAMESE_VARIANTS", "USE_CELL_LINES")}
        e.update(SIAMESE_VARIANTS_LIST=variants, PCA_ARM=arm)
        r = subprocess.run(
            ["bash", "-c", "source slurm/config.conf >/dev/null 2>&1 "
             "|| exit 3; set -e\n" + block],
            cwd=os.path.join(tmp, "siamese_sl"), env=e, capture_output=True,
            text=True, timeout=60)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        return r.stdout

    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        res = os.path.join(sia, "results")
        scr = os.path.join(tmp, "sl_comparison", "results")
        gone_dirs = ["var_pd_pooled__ablate_no_go_cv1",
                     "var_bilinear_per_head__ablate_no_kg_cv3",
                     "var_bilinear_per_head__cellline_ablation_cv2"]
        kept_dirs = ["ablate_no_go_cv1", "cellline_ablation_cv1",
                     "var_bilinear_pooled__ablate_no_go_cv1",
                     "var_pd_pooled__best_cat_go_kg_complex_cv1",
                     "pcavar0.8_go_cv1"]
        for d in gone_dirs + kept_dirs:
            os.makedirs(os.path.join(res, d))
        moved = ["siamese_sl_pd_pooled_cv1.json",
                 "siamese_sl_bilinear_per_head_cv3.json",
                 "siamese_sl_bilinear_per_head_nocell_cv3.json",
                 "siamese_sl_pd_pooled_selected_cv1.txt",
                 "siamese_sl_bilinear_per_head_selected_cv2.txt"]
        kept = ["siamese_sl_cv1.json", "siamese_sl_nocell_cv1.json",
                "siamese_selected_cv1.txt",
                "siamese_sl_bilinear_pooled_cv1.json",
                "siamese_sl_bilinear_pooled_selected_cv1.txt",
                "SLGNN_cv1.json", "DegreePrior_cv1.json"]
        joint = ["comparison.csv", "comparison_notes.txt"]
        for n in moved + kept + joint:
            with open(os.path.join(scr, n), "w") as f:
                f.write("{}")
        for d in ("external_transfer", "paper_tables", "stale_siamese"):
            os.makedirs(os.path.join(scr, d))
        open(os.path.join(scr, "stale_siamese", "earlier.json"), "w").close()
        open(os.path.join(res, "ablation_summary.csv"), "w").close()

        out = run(tmp, "pd:pooled bilinear:per_head")
        assert "Removed 13 output(s)" in out, out
        assert sorted(os.listdir(res)) == sorted(kept_dirs), os.listdir(res)
        assert sorted(os.listdir(scr)) == sorted(kept + ["stale_siamese"]), \
            os.listdir(scr)
        stale = os.listdir(os.path.join(scr, "stale_siamese"))
        # <stem>.<timestamp>.<ext>, as the orchestrators name their moves
        assert sorted(re.sub(r"\.\d{8}T\d{6}(\.\w+)$", r"\1", n)
                      for n in stale) == sorted(moved + ["earlier.json"]), \
            stale
        # under an orchestrator (everything already removed): nothing, silent
        assert run(tmp, "pd:pooled bilinear:per_head") == ""
        # the baseline's names are prefixes of the others': it takes only its
        # own (results/ablate_no_*, siamese_sl_cv*.json, siamese_selected_*)
        out = run(tmp, "pd:per_head")
        assert "Removed 5 output(s)" in out, out
        assert sorted(os.listdir(res)) == sorted(kept_dirs[2:]), \
            os.listdir(res)
        assert sorted(os.listdir(scr)) == sorted(kept[3:] + ["stale_siamese"]), \
            os.listdir(scr)
        # PCA arm joint (the job's PCA_ARM): its own roots only, its rows into
        # its own stale_siamese/, and the table across the arms (it compares
        # every arm's combos); arm none's outputs are left alone
        jres, jscr = os.path.join(res, "pca_joint"), os.path.join(scr, "pca_joint")
        for d in ("ablate_no_go_cv1", "cellline_ablation_cv2"):
            os.makedirs(os.path.join(jres, d))
        os.makedirs(jscr)
        for n in ("siamese_sl_cv1.json", "siamese_selected_cv1.txt",
                  "comparison.csv"):
            open(os.path.join(jscr, n), "w").close()
        os.makedirs(os.path.join(scr, "pca_arms"))
        none_left = (sorted(os.listdir(res)), sorted(
            n for n in os.listdir(scr) if n != "pca_arms"))
        out = run(tmp, "pd:per_head", arm="joint")
        assert "Removed 6 output(s)" in out and ("  joint: comparison rows -> "
                                                 "sl_comparison/results/"
                                                 "pca_joint/stale_siamese/") \
            in out, out
        assert os.listdir(jres) == [] and os.listdir(jscr) == ["stale_siamese"] \
            and len(os.listdir(os.path.join(jscr, "stale_siamese"))) == 2, \
            (os.listdir(jres), os.listdir(jscr))
        assert (sorted(os.listdir(res)), sorted(os.listdir(scr))) == none_left, \
            (os.listdir(res), os.listdir(scr))


def test_hand_combo_resubmission_names_the_jobs_that_must_be_gone():
    # The combo job removes the selection and its variants' markers and combos
    # the moment it starts, then (Step 2 passed) their ablation dirs and rows.
    # Both orchestrators refuse those removals while a job that reads or
    # writes them is queued or running (_check_no_live_jobs outputs); the job
    # itself checks no queue. Its closing WARNING is printed while the rest of
    # an orchestrated run IS alive (it exits 0 "so the other combos' eval and
    # ablation still run") and names the resubmission: run then, the hand job
    # fails the variant's running eval and ablation tasks and any variant's
    # ablation task that starts before its Step 2 has rewritten the selection.
    # So the WARNING, the header and the README section it points to name the
    # jobs that must be gone first: the orchestrator's own list, minus the
    # Step-1 array and the combo job (siamese_sl/README.md Step 2b chains a
    # hand combo job behind its own Step-1 array).
    with open(_SUBMIT) as f:
        scope = re.search(
            r"outputs\)\s+_after=\"\"; _pipe='\^\(([^)']*)\)\$'",
            f.read()).group(1).split("|")
    assert scope[:2] == ["bestcat_emb", "bestcat_combo"], scope
    names = scope[2:]
    # (allmod_combo and bilinear_probe since 2026-11-13: both read the
    # combo's outputs or write rows beside them)
    assert len(names) == 10 and "ablation" in names \
        and {"allmod_combo", "bilinear_probe"} <= set(names), names
    listed = re.compile(
        r"[,;:]?\s+(?:or\s+|and\s+)?".join(map(re.escape, names)))

    def says(text, where):
        flat = re.sub(r"\s+", " ", text.replace("`", "").replace("#", " "))
        m = listed.search(flat)
        assert m, (where, "does not list", names)
        near = flat[max(0, m.start() - 500):m.end() + 500]
        assert "queued or running" in near and "scancel" in near, (where, near)

    with open(os.path.join(_SIAMESE, "slurm",
                           "run_best_per_category_combo.sh")) as f:
        text = f.read()
    tail = text[text.index("# Exit non-zero only when NOTHING trained"):]

    def end(ok, failed, runs):
        pre = ("set -e\nSELECTION_VARIANT=pd:per_head\nSTEP3_SKIPPED=0\n"
               "STEP3_SELECTION_FAILED=0\n"
               f"STEP3_SUCCESSFUL={ok}\nSTEP3_FAILED={failed}\n"
               f"STEP3_FAILED_RUNS=({runs})\n")
        r = subprocess.run(["bash", "-c", pre + tail], capture_output=True,
                           text=True, timeout=60)
        return r.returncode, r.stdout + r.stderr

    # one failed combo: exit 0 (the run's eval and ablation start now), and
    # the lines that name the resubmission say what has to be gone first
    rc, out = end(11, 1, "pd:pooled/cv2")
    assert rc == 0 and "Failed combos (1): pd:pooled/cv2" in out \
        and "Resubmit" in out and ('sl_comparison/README.md, "After a combo '
                                   'job submitted by hand"') in out, (rc, out)
    says(out, "the combo job's closing WARNING")
    # the exit-status rules are untouched: silent and 0 with no failure,
    # 1 (before the WARNING) when nothing trained
    assert end(12, 0, "") == (0, ""), end(12, 0, "")
    rc, out = end(0, 12, "pd:per_head/cv1")
    assert rc == 1 and "no combo trained successfully" in out \
        and "Resubmit" not in out, (rc, out)
    # the header says it too, in comment lines sbatch cannot read as options
    head = text[:text.index("\nset -e\n")]
    says(head[head.index("# Submitted BY HAND"):], "the combo job's header")
    assert "#SBATCH" not in head[head.index("# ====="):], head
    # ... and the README section the WARNING points to, before its recipe
    with open(os.path.join(_ROOT, "sl_comparison", "README.md")) as f:
        sect = f.read().split("**After a combo job submitted by hand.**", 1)[1]
    says(sect.split("```bash", 1)[0], "sl_comparison/README.md")


def test_live_job_lists_name_every_pipeline_job():
    # Every check that refuses to reset, rebuild or remove under a live job
    # lists the names of the jobs the pipeline submits: each script's own
    # #SBATCH --job-name (no orchestrator passes another, so every PCA arm
    # submits the same names); the root and clean scripts also the
    # standalone SLGNN / NSF4SL launchers. Anchored lists, so a name that
    # only looks alike never matches, and none names a job no script has.
    # ELISL's two jobs (ELISL/slurm, submitted by hand) are in every list too
    # (2026-11-13): step 0 moves an ELISL row aside, which a live ELISL job
    # would then rewrite; so are the bilinear probe and the all-modalities
    # combo the orchestrators queue.
    names = set()
    for d in ("siamese_sl/slurm", "siamese_sl/cell_line_ablation",
              "sl_comparison/slurm", "ELISL/slurm"):
        for p in glob.glob(os.path.join(_ROOT, d, "*.sh")):
            with open(p) as f:
                m = re.search(r"^#SBATCH --job-name=(\S+)$", f.read(), re.M)
            if m:
                names.add(m.group(1))
    assert {"pair_export", "bestcat_combo", "sl_paper_tables", "ablation",
            "ablation_summary", "cellline_ablation", "bilinear_probe",
            "allmod_combo", "ELISL", "elisl_inputs"} <= names \
        and not [n for n in names if "pca" in n], names
    for path, extra in ((_ORCH, {"SLGNN", "NSF4SL"}), (_SUBMIT, set()),
                        (os.path.join(_SIAMESE, "slurm", "clean.sh"),
                         {"SLGNN", "NSF4SL"}),
                        (os.path.join(_ROOT, "clean_previous_runs.sh"),
                         {"SLGNN", "NSF4SL"})):
        with open(path) as f:
            lists = re.findall(r"'\^\((reset_env\|[^']*)\)\$'", f.read())
        assert lists, path
        for ls in lists:
            assert set(ls.split("|")) == names | extra \
                and len(ls.split("|")) == len(names | extra), \
                (path, set(ls.split("|")) ^ (names | extra))
    # submit_pipeline.sh's narrower scopes name jobs of that list only: a
    # best-per-category submission's and --ablation-only's
    with open(_SUBMIT) as f:
        scopes = {k: v.split("|") for k, v in re.findall(
            r"^\s+(\w+)\)\s+_after=\"\"; _pipe='\^\(([^)']*)\)\$'", f.read(),
            re.M)}
    assert sorted(scopes) == ["ablation", "outputs"] \
        and all(set(v) <= names for v in scopes.values()), scopes
    assert scopes["ablation"] == ["bestcat_emb", "bestcat_combo", "ablation",
                                  "ablation_summary"], scopes


def test_a_variant_subset_that_regenerates_drops_every_variants_markers():
    # SIAMESE_VARIANTS_LIST (or USE_CELL_LINES=0) retrains only the listed
    # variants' combos, while the generation job rebuilds the embedding files
    # under every variant's combo. The other variants' markers must go too:
    # a later --ablation-only or compare would read their combos as current
    # against vectors they were not trained on. With --skip-generate or
    # --after nothing is rebuilt and they stay.
    names = ["latest_combo_cv1.txt", "var_pd_pooled__latest_combo_cv1.txt",
             "var_bilinear_per_head__latest_combo_cv3.txt",
             "var_bilinear_pooled__latest_combo_cv2.txt"]
    subset = dict(SIAMESE_VARIANTS_LIST="pd:per_head bilinear:pooled")

    def markers(tmp, *flags, **env):
        sia = os.path.join(tmp, "siamese_sl")
        for n in names:
            with open(os.path.join(sia, "results", n), "w") as f:
                f.write("results/x\n")
        _, out = _submit(tmp, *flags, **env)
        left = sorted(n for n in os.listdir(os.path.join(sia, "results"))
                      if n.endswith(".txt"))
        return left, out

    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        other = sorted(names[1:3])
        # a subset that regenerates: no marker of any variant is left
        left, out = markers(tmp, "--skip-reset-env", **subset)
        assert left == [] and "removed 2 combo marker(s) of variants or PCA " \
            "arms this run does not" in out, (left, out[-1200:])
        left, _ = markers(tmp, "--skip-reset-env", USE_CELL_LINES="0")
        assert left == [], left
        # nothing rebuilt: the other variants' combos stay valid
        for flags in (("--skip-generate", "--skip-reset-env"),
                      ("--skip-reset-env", "--after", "777")):
            left, out = markers(tmp, *flags, **subset)
            assert left == other and "combo marker(s) of variants" \
                not in out, (flags, left)
        # all four variants: every marker goes through the per-variant loop,
        # and the new line is not printed
        left, out = markers(tmp, "--skip-reset-env")
        assert left == [] and "combo marker(s) of variants" not in out, \
            left


def test_benchmark_only_regeneration_guard_sees_every_arms_root():
    # --benchmark-only retrains no combo but still regenerates the
    # embeddings: refused while any arm's combos (a marker or a selection in
    # its results root) were trained on the files it would replace, whatever
    # arms PCA_ARMS names, before anything changes; --skip-generate keeps the
    # files and is accepted.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        res = os.path.join(sia, "results")
        for rel in ("pca_both/var_pd_pooled__latest_combo_cv2.txt",
                    "pca_joint/best_per_category.json",
                    "latest_combo_cv1.txt"):
            shutil.rmtree(res)
            os.makedirs(os.path.join(res, os.path.dirname(rel) or "."),
                        exist_ok=True)
            with open(os.path.join(res, rel), "w") as f:
                f.write("{}")
            for arms in ("none", None):
                jobs, out = _submit(tmp, "--benchmark-only", _rc=1,
                                    PCA_ARMS=arms)
                assert not jobs and "ERROR: this submission would regenerate" \
                    " the embeddings but retrains no combo" in out \
                    and f"\n    results/{rel}\n" in out, (rel, arms, out[-1500:])
                assert os.path.isfile(os.path.join(res, rel)), rel
            jobs, _ = _submit(tmp, "--benchmark-only", "--skip-generate",
                              PCA_ARMS=None)
            assert jobs and os.path.isfile(os.path.join(res, rel)), rel


def test_submit_pipeline_refuses_a_cvs_its_block_cannot_take():
    # CVS reaches no job of submit_pipeline.sh, only the compare of the block
    # a best-per-category run ends with (_paste_env, 2026-10-21): one that is
    # not a list of cv1 / cv2 / cv3, each once, is refused before anything
    # changes (an earlier run's selection, markers, ablation dir, SLeuth row
    # and comparison table all kept), as the root refuses it, and so it is
    # with --ablation-only, whose refusals print --best-cat-only lines that
    # carry it (2026-10-22: such a line, run as printed, was refused here);
    # a valid one is carried; with neither (--benchmark-only, --eval-only)
    # it is not looked at.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        _seed_combos(sia, "none", {("", cv): {} for cv in ("cv1", "cv2", "cv3")})
        cmp_dir = os.path.join(tmp, "sl_comparison", "results")
        os.makedirs(cmp_dir, exist_ok=True)
        for f in ("comparison.csv", "siamese_sl_cv1.json"):
            with open(os.path.join(cmp_dir, f), "w") as fh:
                fh.write("{}\n")
        combo = os.path.join(sia, "results", "x_cv3")
        os.makedirs(combo)
        for f in ("config.json", "results.json"):
            with open(os.path.join(combo, f), "w") as fh:
                fh.write("{}")

        def state():
            out = {}
            for top in (os.path.join(sia, "results"), cmp_dir):
                for dp, dn, fn in os.walk(top):
                    for n in dn + fn:
                        q = os.path.join(dp, n)
                        out[os.path.relpath(q, tmp)] = os.stat(q).st_mtime_ns
            return out

        before = state()
        assert "siamese_sl/results/ablate_no_go_cv1" in before \
            and "sl_comparison/results/siamese_sl_cv1.json" in before, before
        base = ("--skip-generate", "--skip-reset-env")
        for bad in ("cv1,cv2", "cv4", "cv1 cv1", " "):
            jobs, out = _submit(tmp, *base, CVS=bad, _rc=1)
            assert not jobs and f"ERROR: CVS='{bad}': want a space-separated " \
                "subset of cv1 cv2 cv3, each once" in out \
                and "Nothing\n  was changed or submitted." in out \
                and state() == before, (bad, out)
        jobs, out = _submit(tmp, "--ablation-only", CVS="cv1,cv2", _rc=1)
        assert not jobs and ("ERROR: CVS='cv1,cv2': want a space-separated "
                             "subset of cv1 cv2 cv3, each once\n  (the CVs of "
                             "the comparison of the --best-cat-only lines an "
                             "--ablation-only\n  refusal prints, which carry "
                             "it). Nothing was changed or submitted.") in out \
            and state() == before, out
        # (checked before the refusals that print those lines: arm joint
        # has no selection, and no line is printed)
        jobs, out = _submit(tmp, "--ablation-only", CVS="cv1,cv2", _rc=1,
                            PCA_ARMS="none joint")
        assert not jobs and "ERROR: CVS='cv1,cv2'" in out \
            and "needs the selection" not in out \
            and not _printed_commands(out) and state() == before, out
        for args in (("--eval-only", "results/x_cv3"),
                     ("--benchmark-only",) + base):
            jobs, out = _submit(tmp, *args, CVS="cv1,cv2")
            assert jobs and "ERROR: CVS=" not in out, (args, out)
        jobs, out = _submit(tmp, "--eval-only", "results/x_cv3", CVS="cv1,cv2")
        assert _arm_names(jobs) == ["eval_finetuning.sh/none",
                                    f"{_PAIR_SH}/none"], jobs
        jobs, out = _submit(tmp, "--ablation-only", CVS="cv1 cv3")
        assert _arm_names(jobs) == ["run_ablation.sh/none",
                                    "run_ablation_summary.sh/none"], out
        jobs, out = _submit(tmp, *base, CVS="cv1 cv3")
        assert jobs and ' CVS="cv1 cv3" ' in _sp_block(out)[0], _sp_block(out)[0]


def test_benchmark_only_refusal_runs_as_printed():
    # The guard's remedy keeps the embeddings: this submission again with
    # --skip-generate, after the settings taken from the calling shell and
    # the three targets in full (2026-10-21: it printed the bare command,
    # which, pasted into a fresh shell, reset the venv the refused run had
    # kept and benchmarked config.conf's four arms at their default PCA and
    # seed 42; the arm the refused run named, at its own, never ran). Run
    # as printed (bash, and zsh where installed) from a shell that exports
    # other targets: the one owner's benchmark array and its summary, with
    # the refused run's arm, seed and PCA.
    shells = ["bash"] + [z for z in (shutil.which("zsh"),) if z]
    stray = {"PCA_MODALITY_VARIANCE": "0.5", "PCA_JOINT_VARIANCE": "0.6",
             "PCA_JOINT_DIM": "999"}
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        _env_stubs(tmp)
        r = os.path.join(sia, _root_of("joint"))
        os.makedirs(r, exist_ok=True)
        with open(os.path.join(r, "latest_combo_cv1.txt"), "w") as f:
            f.write("results/pca_joint/best_cat_x_cv1\n")
        jobs, out = _submit(tmp, "--benchmark-only", "--skip-reset-env",
                            _rc=1, PCA_ARMS="per_modality", SEED="7",
                            PCA_MODALITY_VARIANCE="0.9")
        cmds = [ln[4:] for ln in out.splitlines() if ln.startswith("    ")
                and "slurm/submit_pipeline.sh" in ln]
        # (USE_CELL_LINES pinned too, 2026-10-22: the benchmark trains the
        # cell-line model, or under a stray USE_CELL_LINES=0 the other one)
        assert not jobs and cmds == [
            "PCA_ARMS=per_modality SEED=7 PCA_MODALITY_VARIANCE=0.9 "
            "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM= USE_CELL_LINES=1 bash slurm/"
            "submit_pipeline.sh --benchmark-only --skip-reset-env "
            "--skip-generate"], (cmds, out[-2000:])
        # (the arms pinned where the refused run took them from config.conf)
        jobs, o3 = _submit(tmp, "--benchmark-only", "--skip-reset-env",
                           _rc=1, PCA_ARMS=None, SEED="7")
        assert [ln[4:] for ln in o3.splitlines() if ln.startswith("    ")
                and "slurm/submit_pipeline.sh" in ln] == [
            'SEED=7 PCA_ARMS="none joint per_modality both" '
            "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM= "
            "USE_CELL_LINES=1 bash slurm/submit_pipeline.sh --benchmark-only "
            "--skip-reset-env --skip-generate"], o3[-2000:]
        for sh in shells:
            rc, o2, calls = _run_printed(tmp, cmds[0], sh, USE_CELL_LINES="0",
                                         **stray)
            assert rc == 0 and [(a.rsplit("/", 1)[1], e["PCA_ARM"], e["SEED"],
                                 e["PCA_MODALITY_VARIANCE"],
                                 e["USE_CELL_LINES"])
                                for _, a, e in calls] == [
                (s, "per_modality", "7", "0.9", "1")
                for s in ("run_embedding_benchmark.sh",
                          "run_embedding_benchmark_summary.sh")], \
                (sh, o2[-2500:], calls)
            assert os.path.isfile(os.path.join(r, "latest_combo_cv1.txt"))


def test_live_jobs_check_covers_what_a_submission_removes():
    # A submission that neither resets the venv nor regenerates still removes,
    # before any sbatch, the markers, selection, ablation dirs and
    # sl_comparison rows a live run's siamese-stage jobs read and write; and
    # --ablation-only retrains the ablation dirs under the ablated run's own
    # token. Both are refused while such a job is queued or running, with
    # nothing removed (ALLOW_LIVE_JOBS=1: a WARNING). Jobs that read none of
    # it never block: sl_comparison/README.md's manual DAG submits with
    # reset_env, siamese_gen_emb, sl_folds and sl_compare queued. (One arm,
    # arm none: the job names carry no arm, so every arm is checked alike.)
    def run(tmp, queue, *args, **env):
        sia, b = os.path.join(tmp, "siamese_sl"), os.path.join(tmp, "bin")
        with open(os.path.join(b, "squeue"), "w") as f:
            f.write("#!/bin/bash\ncat <<'Q'\n" + queue + "Q\n")
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "PAIR_EXPORT_", "SB_", "PCA_",
                                  "POST_PCA_"))
             and k not in ("RUN_TOKEN", "SIAMESE_VARIANTS_LIST",
                           "SIAMESE_VARIANTS", "BESTCAT_EXTRA_VARIANTS",
                           "EVAL_MODEL_DIR", "EVAL_VARIANT", "GPU_NODES",
                           "GPU_EXCLUDE", "ALLOW_LIVE_JOBS",
                           "SELECTION_CV")}
        e.update(PATH=b + os.pathsep + e["PATH"],
                 SB_LOG=os.path.join(tmp, "sb.log"),
                 SB_CTR=os.path.join(tmp, "ctr"), PCA_ARMS="none", **env)
        for f in (e["SB_LOG"], e["SB_CTR"], e["SB_LOG"] + ".test",
                  e["SB_LOG"] + ".pairenv"):
            if os.path.exists(f):
                os.remove(f)
        r = subprocess.run(["bash", "slurm/submit_pipeline.sh", *args],
                           cwd=sia, env=e, capture_output=True, text=True,
                           timeout=120)
        jobs = []
        if os.path.exists(e["SB_LOG"]):
            with open(e["SB_LOG"]) as f:
                jobs = [os.path.basename(ln.split("|", 2)[2].split()[-1])
                        for ln in f]
        return r.returncode, jobs, r.stdout + r.stderr
    kept = ["ablate_no_ppi_cv1", "best_per_category.json",
            "latest_combo_cv3.txt"]
    def seed(res):
        shutil.rmtree(res)
        os.makedirs(os.path.join(res, "ablate_no_ppi_cv1", "checkpoints"))
        with open(os.path.join(res, "best_per_category.json"), "w") as f:
            json.dump({"cv1": {}, "_selection_meta": {"run_token": "T1"}}, f)
        with open(os.path.join(res, "latest_combo_cv3.txt"), "w") as f:
            f.write("results/x_cv3\n")

    abl = "900_[5-71] ablation\n900_4 ablation\n901 ablation_summary\n"
    manual = ("10 reset_env\n5 siamese_gen_emb\n11 sl_folds\n"
              "12_[0-8] sl_compare\n")
    flat = ("--skip-reset-env", "--skip-generate")
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        res = os.path.join(sia, "results")

        def refused(queue, *args):
            seed(res)
            rc, jobs, out = run(tmp, queue, *args)
            assert rc == 1 and jobs == [] and sorted(os.listdir(res)) == kept \
                and "ERROR: jobs of an earlier submission" in out \
                and "Nothing was changed" in out, (args, queue, rc, jobs,
                                                   out[-800:])
            return out

        def accepted(queue, *args, **env):
            seed(res)
            rc, jobs, out = run(tmp, queue, *args, **env)
            assert rc == 0 and jobs, (args, queue, rc, out[-800:])
            return jobs, out

        # --ablation-only beside the array it would duplicate, its summary, or
        # a combo job about to replace the combos it ablates
        for q in (abl, "901 ablation_summary\n", "700 bestcat_combo\n",
                  "699_[3-56] bestcat_emb\n"):
            refused(q, "--ablation-only")
        jobs, out = accepted(abl, "--ablation-only", ALLOW_LIVE_JOBS="1")
        assert jobs == ["run_ablation.sh", "run_ablation_summary.sh"] \
            and "WARNING: jobs of an earlier submission" in out, (jobs, out)
        # ... but not beside jobs that read no ablation dir
        jobs, out = accepted("902_[0-11] eval_finetuning\n903 pair_export\n"
                             "12_[0-8] sl_compare\n", "--ablation-only")
        assert "jobs of an earlier submission" not in out, out
        # a best-per-category submission without reset / generation: every
        # job that reads or writes what it removes blocks it
        for name in ("bestcat_emb", "bestcat_combo", "eval_finetuning",
                     "ablation", "ablation_summary", "cellline_ablation",
                     "pair_export", "sl_final_compare", "sl_ext_transfer",
                     "sl_paper_tables"):
            refused(f"950 {name}\n", *flat)
        refused(abl, *flat, "--best-cat-only", "--skip-ablation",
                "--skip-eval")
        _, out = accepted(abl, *flat, ALLOW_LIVE_JOBS="1")
        assert "WARNING: jobs of an earlier submission" in out, out
        # --after orders this submission's JOBS, not the removal: its job is
        # no exception when it is one of those
        refused("903 pair_export\n", "--skip-reset-env", "--best-cat-only",
                "--after", "903")
        # the manual DAG, and names that only look alike, never block
        for q, args in ((manual, ("--skip-reset-env", "--best-cat-only",
                                  "--skip-pair-export", "--after", "5")),
                        ("60_[0-59] bench_emb\n61 bench_summary\n"
                         "80 ablation2\n81 my_ablation\n78 SLGNN\n", flat)):
            jobs, out = accepted(q, *args)
            assert "run_best_per_category_combo.sh" in jobs \
                and "jobs of an earlier submission" not in out, (args, out)
        # the venv / embedding refusal no longer points at a checked path
        seed(res)
        rc, jobs, out = run(tmp, abl)
        assert rc == 1 and jobs == [] and "touches neither" not in out \
            and "venv reset_env deletes" in out, out[-800:]
        # --eval-only without --reset-env stays unchecked (the recovery step)
        seed(res)
        os.makedirs(os.path.join(res, "x_cv3"))
        for n in ("config.json", "results.json"):
            with open(os.path.join(res, "x_cv3", n), "w") as f:
                f.write("{}")
        rc, jobs, out = run(tmp, abl + "902_[0-11] eval_finetuning\n",
                            "--eval-only", "results/x_cv3")
        assert rc == 0 and jobs[:1] == ["eval_finetuning.sh"], (rc, jobs, out)


def test_login_node_python_runs_before_the_reset_is_queued():
    # reset_env.sh rm -rf's the venv $PYTHON_PATH lives in as soon as it
    # starts, so every interpreter call of a submission must precede that
    # sbatch (the root orchestrator's rule). --skip-generate's embedding check
    # used to follow it, and a reset job that started at once made it report
    # current embeddings as unconfirmed (or each one as STALE "unreadable").
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        py = os.path.join(b, "py")     # logs each call into the sbatch log
        with open(py, "w") as f:
            f.write('#!/bin/bash\necho "py||$*" >> "$SB_LOG"\n')
        os.chmod(py, 0o755)
        conf = os.path.join(sia, "slurm", "config.conf")
        with open(conf) as f:
            text, n = re.subn(r"(?m)^PYTHON_PATH=.*$",
                              f'PYTHON_PATH="{py}"', f.read())
        assert n == 1
        with open(conf, "w") as f:
            f.write(text)
        sel = os.path.join(sia, "results", "best_per_category.json")
        for args, n_reset, n_check in (
                (("--skip-generate",), 1, 1),
                (("--ablation-only", "--reset-env"), 1, 1),
                (("--skip-generate", "--skip-reset-env"), 0, 1),
                (("--ablation-only",), 0, 1),
                (("--after", "777"), 1, 0),      # never with --after ...
                ((), 1, 0)):                     # ... or when generation runs
            with open(sel, "w") as f:
                json.dump({"cv1": {}, "_selection_meta": {"run_token": "T"}},
                          f)
            # the combo --ablation-only ablates (a regeneration removes it)
            with open(os.path.join(os.path.dirname(sel),
                                   "latest_combo_cv1.txt"), "w") as f:
                f.write("results/x_cv1\n")
            calls, _ = _submit(tmp, *args)
            check = [a for j, _, a in calls
                     if j == "py" and "embedding_versions.py" in a]
            assert len(check) == n_check, (args, calls)
            reset = [i for i, (_, _, a) in enumerate(calls)
                     if a.endswith("reset_env.sh")]
            assert len(reset) == n_reset, (args, calls)
            if reset:
                late = [a for j, _, a in calls[reset[0] + 1:] if j == "py"]
                assert not late, (args, late)


def test_each_arms_stages_name_its_pca_once():
    # Every arm's combo and ablation train with its own PCA (config.conf
    # pca_multi_args), its single-embedding runs with their owner's
    # (pca_single_args): the submission names each once, under the arm (or
    # owner) it belongs to, from the PCA_* targets; nothing reads POST_PCA_*.
    def pca_lines(out, header):
        got = {}
        for part in re.split(r"(?m)^=== ", out)[1:]:
            m = re.match(header, part)
            if m:
                got[m.group(1)] = [ln.strip() for ln in part.splitlines()
                                   if re.match(r"\s+(PCA|Single-embedding "
                                               r"runs): ", ln)]
        return got

    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        for env, multi, single in (
                (dict(PCA_ARMS=None),
                 {"none": "--no_post_pca",
                  "joint": "--post_pca_variance 0.8",
                  "per_modality": "--pca_variance 0.8 --no_post_pca",
                  "both": "--pca_variance 0.8 --post_pca_variance 0.8"},
                 {"none": ("results", "--no_post_pca"),
                  "per_modality": ("results/pca_per_modality",
                                   "--pca_variance 0.8 --no_post_pca")}),
                (dict(PCA_ARMS="joint both", PCA_JOINT_DIM="64",
                      PCA_MODALITY_VARIANCE="0.9", PCA_JOINT_VARIANCE="0.7"),
                 {"joint": "--post_pca_dim 64",
                  "both": "--pca_variance 0.9 --post_pca_dim 64"},
                 {"joint": ("results/pca_joint", "--no_post_pca"),
                  "both": ("results/pca_both",
                           "--pca_variance 0.9 --no_post_pca")})):
            _, out = _submit(tmp, "--skip-generate", "--skip-reset-env", **env)
            got = pca_lines(out, r"PCA arm (\S+): ")
            assert got == {a: ["PCA: " + f, "PCA: " + f + " (the arm's, as "
                               "its reference combos')"]
                           for a, f in multi.items()}, (env, got)
            got = pca_lines(out, r"Single-embedding runs of PCA arm\(s\) "
                                 r"(\S+)")
            assert got == {o: [f"Single-embedding runs: {r}/pcavar0.8_* ({f})"]
                           for o, (r, f) in single.items()}, (env, got)
            assert "Post-concat PCA" not in out, out


def _paper_tables():
    # By file path: sl_comparison/ on sys.path would shadow siamese_sl's
    # same-named modules (retrieval_metrics.py) for every later import.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_paper_tables_contract",
        os.path.join(_ROOT, "sl_comparison", "paper_tables.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _catalog(*arrays):
    """EMB_CATALOG types plus the named arrays, as sourced from the confs."""
    body = 'printf "%s\\n" "${EMB_CATALOG[@]%%:*}"'
    for a in arrays:
        body += '; echo --; printf "%%s\\n" "${%s[@]}"' % a
    out = subprocess.run(
        ["bash", "-c",
         "unset SLURM_JOB_ID; source slurm/config.conf >/dev/null 2>&1; "
         "source slurm/run_best_per_category.conf >/dev/null 2>&1; "
         "source slurm/run_generate_embeddings.conf; " + body],
        cwd=_SIAMESE, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    return [set(part.split()) for part in out.stdout.split("--\n")]


def test_generation_requires_exactly_the_papers_single_modality_set():
    # EMB_CATALOG minus OPTIONAL_TYPES is what generation refuses to lose:
    # it must be paper_tables.PAPER_SOLO (the paper's 16 solo modalities).
    cat, optional = _catalog("OPTIONAL_TYPES")
    solo = set(_paper_tables().PAPER_SOLO)
    assert cat - optional == solo, (cat - optional) ^ solo


def test_generation_builds_only_the_papers_catalog_members():
    # The combo ranks every EMB_CATALOG type that has a .pt, so generation
    # (ESM-2 and GO in dedicated steps, the rest from PRECOMPUTED_TYPES) must
    # attempt no catalog member outside the paper's 16 — e.g. a 17th
    # protein_seq ESM-C — except scPRINT / SeqVec, which reset_env.sh's venv
    # cannot build and which OPTIONAL_TYPES therefore accepts as absent.
    cat, precomputed = _catalog("PRECOMPUTED_TYPES")
    built = (precomputed | {"esm2", "go"}) & cat
    solo = set(_paper_tables().PAPER_SOLO)
    assert solo <= built, solo - built
    assert built - solo <= {"scprint", "seqvec"}, built - solo


def test_the_signature_subprocess_keeps_its_error():
    # `train.py --print_arch` gives the signature the combo's Step 2 and each
    # ablation task check their candidates / reference against. When it fails
    # (a mistyped ARCH_ARGS on a hand resubmission, a broken import) its own
    # error must reach the job's .err, and the combo must say THAT, not list
    # every valid Step-1 run as stale with the advice to delete it. `| tail
    # -1` stays: without pipefail it keeps the assignment from ending the job
    # under set -e before the explanation is printed.
    for name in ("run_best_per_category_combo.sh", "run_ablation.sh"):
        calls = [ln for ln in _logical_lines(os.path.join(_SIAMESE, "slurm",
                                                          name))
                 if "--print_arch" in ln and "$PYTHON_PATH train.py" in ln]
        assert len(calls) == 1 and "2>/dev/null" not in calls[0] \
            and calls[0].rstrip().endswith("| tail -1)"), (name, calls)
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_best_per_category_combo.sh")) as f:
        text = f.read()
    i = text.index("if EXPECT_SIG is None:")
    branch = text[i:text.index("\nfor (etype, cv), cfg in", i)]
    assert "raise SystemExit(1)" in branch and "stale.append" not in branch \
        and "Step-1 results are untouched" in branch, branch
    # (the heredoc is unquoted: bash would expand these inside it)
    assert not set("$`\\") & set(branch), branch


def test_run_ablation_header_exports_one_token_before_its_sbatch():
    # The documented direct submission of the array + summary must export ONE
    # RUN_TOKEN first, or the summary marks every row a run_token MISMATCH.
    for name in ("run_ablation.sh",):
        with open(os.path.join(_SIAMESE, "slurm", name)) as f:
            head = f.read().split("\nset -e", 1)[0].splitlines()
        # "sbatch" is lower case, so the #SBATCH directives never match.
        i_sb = next(i for i, ln in enumerate(head)
                    if ln.startswith("#") and "sbatch " in ln
                    and "submit_pipeline.sh" not in ln)
        assert any(ln.lstrip("# ").startswith("export RUN_TOKEN=")
                   for ln in head[:i_sb]), (name, head[:i_sb])


def _train_args(name):
    """The entries of a script's TRAIN_ARGS=( ... ) array, one per line."""
    with open(os.path.join(_SIAMESE, "slurm", name)) as f:
        text = f.read()
    i = text.index("\nTRAIN_ARGS=(\n")
    return [ln.strip() for ln in
            text[i:text.index("\n)\n", i)].splitlines()[2:]]


def test_run_ablation_s_train_args_are_the_combo_s_step3_call():
    # The ablation copies this array, so it must be the call that trained its
    # reference (run_best_per_category_combo.sh Step 3), entry for entry:
    # --num_folds, --sl_path or --pos_neg_ratio are not signature keys, and
    # the worker's --print_arch check would not see them drift. The arm's
    # PCA comes in one slot of both, $PCA_MULTI_ARGS (config.conf), so an
    # arm's ablation trains with its reference combos' PCA.
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_best_per_category_combo.sh")) as f:
        text = f.read()
    # (Step 1's --print_arch probe is the script's other train.py call)
    i = text.index("    $PYTHON_PATH train.py \\\n"
                   "        --embeddings_paths $EMB_PATHS")
    combo = [ln.strip().rstrip("\\").rstrip() for ln in
             text[i:text.index("\n    TRAIN_EXIT=$?", i)].splitlines()[1:]]
    loco = _train_args("run_ablation.sh")
    assert combo == loco, (combo, loco)
    assert loco.count("$PCA_MULTI_ARGS") == 1 \
        and not [a for a in loco if "pca" in a.lower()
                 and a not in ("$PCA_MULTI_ARGS",
                               '--pca_method "${PCA_METHOD:-plain}"')], loco


def test_kernel_and_cell_lines_abbreviations_are_refused():
    """Two independent guards, both required.

    ARCH_ARGS is scanned on PREFIXES in config.conf (a --kernel/--cell_lines
    on the command line would be LAST and silently override the variant, so
    all four 2x2 variants would train identically), and train.py's parser is
    built with allow_abbrev=False, which covers MODEL_ARGS and a hand-run
    train.py that config.conf never sees. Checked by text rather than by
    invoking train.py, which would load data if the flag were ever lost.
    """
    for bad in ("--kern bilinear", "--k pd", "--cell_lines pooled",
                "--layernorm --kernel bilinear"):
        r = subprocess.run(
            ["bash", "-c",
             f'ARCH_ARGS={bad!r} source slurm/config.conf'],
            cwd=_SIAMESE, capture_output=True, text=True, timeout=60)
        assert r.returncode != 0, (bad, r.stdout, r.stderr)
        assert "SIAMESE_VARIANTS" in r.stderr, (bad, r.stderr)
    with open(os.path.join(_SIAMESE, "train.py")) as f:
        src = f.read()
    m = re.search(r"argparse\.ArgumentParser\((.*?)\n    \)\n", src, re.S)
    assert m and "allow_abbrev=False" in m.group(1), \
        "train.py's parser must set allow_abbrev=False"


def test_pca_flags_in_arch_args_are_refused():
    """ARCH_ARGS comes last on every train.py command line, so a PCA flag in
    it would silently override the PCA of the job's arm (config.conf
    PCA_MULTI_ARGS / PCA_SINGLE_ARGS; the logs echo only ARCH_ARGS) and
    every arm would train the same PCA under its own arm's name.
    config.conf refuses the PCA flags on prefixes and in any =value form, as
    it does --kernel / --cell_lines, and names what sets the PCA instead
    (the arms and their targets), by a path that reads right from both
    orchestrators' directories; --pca_method (the preprocessing) and
    look-alike flags stay allowed. Both orchestrators stop on it before
    anything changes."""
    def source(arch):
        e = {k: v for k, v in os.environ.items()
             if k != "SLURM_JOB_ID" and not k.startswith(("PCA_",
                                                          "POST_PCA_"))}
        e["ARCH_ARGS"] = arch
        return subprocess.run(
            ["bash", "-c", 'source slurm/config.conf >/dev/null || exit 3; '
             'printf "%s" "$ARCH_ARGS"'], cwd=_SIAMESE, env=e,
            capture_output=True, text=True, timeout=60)

    for bad in ("--pca_variance 0.8", "--pca_dims 64 64 64",
                "--post_pca_variance=0.8", "--post_pca_dim 512",
                "--no_post_pca", "--layernorm --no_post_pca", "--pca_v 0.8",
                "--post_pca 0.8", "--layernorm\t--post_pca_variance 0.8"):
        r = source(bad)
        assert r.returncode != 0 and "ERROR: ARCH_ARGS must not set a PCA" \
            in r.stderr, (bad, r.returncode, r.stderr)
        msg = " ".join(r.stderr.split())
        assert "the PCA is the PCA arm's (PCA_ARMS / PCA_ARM, " \
            "PCA_MODALITY_VARIANCE, PCA_JOINT_VARIANCE, PCA_JOINT_DIM in " \
            "siamese_sl/slurm/config.conf)" in msg \
            and "POST_PCA" not in msg and "PCA_ABLATION" not in msg, \
            (bad, msg)
    published = ("--layernorm --log_temperature --no_per_head_last_layer_bias"
                 " --select_metric auprg --no_gene_term --no_pair_features")
    for good in ("--pca_method robust", "--pos_neg_ratio 2", published):
        r = source(good)
        assert r.returncode == 0 and r.stdout == good, (good, r.stderr)
    # ... before a submission changes anything
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        jobs, _ = _submit(tmp, "--skip-generate", "--skip-reset-env", _rc=1,
                          ARCH_ARGS="--layernorm --post_pca_dim 64")
        assert not jobs, jobs
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        rc, out, jobs = _root_run(tmp, "--force-generate",
                                  ARCH_ARGS="--layernorm --pca_variance 0.8")
        assert rc != 0 and not jobs \
            and "ERROR: ARCH_ARGS must not set a PCA" in out \
            and not os.path.exists(os.path.join(
                root, "sl_comparison", "folds", "meta.json")), out[-1500:]


# ---- node selection, array caps, direct execution, the step-0 folds freeze --

_ORCH = os.path.join(_ROOT, "run_benchmarking_siamese_vs_others.sh")


_SUBMIT = os.path.join(_SIAMESE, "slurm", "submit_pipeline.sh")


def _logical_lines(path):
    """Non-comment lines, backslash continuations joined into one."""
    out, cur = [], ""
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not cur and ln.lstrip().startswith("#"):
                continue
            if ln.endswith("\\"):
                cur += ln[:-1] + " "
                continue
            out.append(cur + ln)
            cur = ""
    return out


def test_every_submission_passes_node_args_and_throttle():
    # Without $GPU_NODE_ARGS a job ignores GPU_NODES and stays on the
    # directive's gpu3; a raw %${X} breaks on the (default) empty value.
    for path in (_ORCH, _SUBMIT):
        subs = [ln for ln in _logical_lines(path)
                if re.search(r"\bsbatch\s+--parsable\b", ln)]
        assert len(subs) >= 8, (path, len(subs))
        for ln in subs:
            assert "$GPU_NODE_ARGS" in ln, (path, ln.strip()[:160])
            if "--array=" in ln:
                assert "$(array_throttle " in ln, (path, ln.strip()[:160])
                assert not re.search(r"--array=\S*%\$", ln), \
                    (path, ln.strip()[:160])


def _conf_rc(script, **env):
    e = {k: v for k, v in os.environ.items()
         if k not in ("SLURM_JOB_ID", "GPU_NODES", "GPU_EXCLUDE",
                      "MAX_CONCURRENT", "EVAL_CONCURRENT",
                      "SHARED_CONCURRENT")}
    e.update(env)
    r = subprocess.run(["bash", "-c", "source slurm/config.conf >/dev/null "
                        "2>&1 || exit 3; " + script], cwd=_SIAMESE, env=e,
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout.strip(), r.stderr


def test_node_selection_and_throttles_are_validated():
    chk = 'check_throttles && check_node_selection && echo "[$GPU_NODE_ARGS]"'
    for env, args in ((dict(), "[]"),
                      (dict(GPU_NODES="gpu4"), "[--nodelist=gpu4]"),
                      # whitespace would split the unquoted expansion
                      (dict(GPU_NODES="gpu2, gpu4", GPU_EXCLUDE="gpu3"),
                       "[--nodelist=gpu2,gpu4 --exclude=gpu3]"),
                      (dict(MAX_CONCURRENT="4", EVAL_CONCURRENT="0",
                            SHARED_CONCURRENT=""), "[]")):
        rc, out, err = _conf_rc(chk, **env)
        assert rc == 0 and out == args, (env, rc, out, err)
    # --exclude cannot lift the scripts' #SBATCH --nodelist=gpu3, and SLURM
    # rejects overlapping required / excluded nodes; a non-numeric cap would
    # otherwise submit the array uncapped with only a stray ERROR line.
    for env in (dict(GPU_EXCLUDE="gpu2"), dict(GPU_EXCLUDE="gpu3"),
                dict(GPU_NODES="gpu2,gpu4", GPU_EXCLUDE="gpu4"),
                dict(MAX_CONCURRENT="4x"), dict(SHARED_CONCURRENT="-1"),
                dict(EVAL_CONCURRENT=" 4")):
        rc, out, err = _conf_rc(chk, **env)
        assert rc != 0 and "ERROR" in err, (env, rc, out, err)


def test_submit_pipeline_stops_on_a_bad_selection_before_anything():
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        log = os.path.join(tmp, "sb.log")
        for env in (dict(GPU_EXCLUDE="gpu2"), dict(MAX_CONCURRENT="4x"),
                    dict(EVAL_CONCURRENT="04"),
                    # checked by run_best_per_category.conf, which used to be
                    # sourced only after the first jobs were queued
                    dict(BESTCAT_EXTRA_VARIANTS="2"),
                    # ... and a selection CV that names no CV, which only
                    # the combo's Step 2 refused, after the Step-1 array
                    dict(SELECTION_CV="cv4"), dict(SELECTION_CV="cv1 cv2"),
                    # slurmctld refusing a stage's script on the node
                    dict(SB_TEST_FAIL="1")):
            e = {k: v for k, v in os.environ.items()
                 if not k.startswith("SLURM_") and k not in (
                     "RUN_TOKEN", "GPU_NODES", "GPU_EXCLUDE",
                     "MAX_CONCURRENT", "EVAL_CONCURRENT", "SB_TEST_FAIL",
                     "BESTCAT_EXTRA_VARIANTS", "SELECTION_CV")}
            e.update(PATH=b + os.pathsep + e["PATH"], SB_LOG=log,
                     SB_CTR=os.path.join(tmp, "ctr"), **env)
            r = subprocess.run(["bash", "slurm/submit_pipeline.sh"], cwd=sia,
                               env=e, capture_output=True, text=True,
                               timeout=120)
            assert r.returncode == 1, (env, r.stdout[-1500:], r.stderr[-1500:])
            assert "Nothing was changed or submitted" in r.stdout, \
                (env, r.stdout[-1500:])
            assert not os.path.exists(log), (env, open(log).read())
        # submit_pipeline.sh has no shared-model array: SHARED_CONCURRENT is
        # not its business
        jobs, _ = _submit(tmp, SHARED_CONCURRENT="x")
        assert jobs, jobs
        # no job of an --eval-only (or a benchmark that generates nothing)
        # reads run_best_per_category.conf: its value is not their business
        d = os.path.join(sia, "results", "x_cv3")
        os.makedirs(d)
        for f in ("config.json", "results.json"):
            open(os.path.join(d, f), "w").write("{}")
        jobs, _ = _submit(tmp, "--eval-only", "results/x_cv3",
                          BESTCAT_EXTRA_VARIANTS="2")
        assert jobs, jobs
        jobs, _ = _submit(tmp, "--benchmark-only", "--skip-generate",
                          "--skip-reset-env", BESTCAT_EXTRA_VARIANTS="2")
        assert jobs, jobs
        # ... but a benchmark that generates does: the generation job's
        # closing catalog check sources that conf
        jobs, out = _submit(tmp, "--benchmark-only", _rc=1,
                            BESTCAT_EXTRA_VARIANTS="2")
        assert not jobs and "Nothing was changed or submitted" in out, out
        # --after without a job id swallowed the next flag (or an empty $GEN)
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "SB_")) and k != "RUN_TOKEN"}
        e.update(PATH=b + os.pathsep + e["PATH"], SB_LOG=log,
                 SB_CTR=os.path.join(tmp, "ctr"))
        if os.path.exists(log):
            os.remove(log)
        for bad in (["--after", "--skip-eval"], ["--after", ""],
                    ["--after", "siamese_gen_emb"], ["--after", "777_"],
                    ["--after", "_7"], ["--after", "7_7_7"],
                    ["--after", "7 8"], ["--skip-reset-env", "--after"]):
            r = subprocess.run(["bash", "slurm/submit_pipeline.sh", *bad],
                               cwd=sia, env=e, capture_output=True, text=True,
                               timeout=120)
            assert r.returncode == 1 and "--after requires a job ID" \
                in r.stdout, (bad, r.stdout[-800:], r.stderr[-800:])
            assert not os.path.exists(log), (bad, open(log).read())


def test_submit_pipeline_node_selection_reaches_every_job():
    # Every stage's script is probed with --test-only (submits nothing) on
    # the selection before anything changes; the selection then rides on
    # every submission. Without GPU_NODES the probes test the scripts' pin.
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        jobs, out = _submit(tmp, GPU_NODES="gpu4")
        with open(os.path.join(tmp, "sb.log.test")) as f:
            probes = f.read().splitlines()
        names = [p.rsplit("/", 1)[-1] for p in probes]
        assert {"reset_env.sh", "run_generate_embeddings.sh",
                "run_best_per_category_combo.sh"} <= set(names), probes
        assert all(p.startswith("--test-only --nodelist=gpu4 ")
                   for p in probes), probes
        args = [a for _, _, a in jobs]
        assert len(args) > 3 and all("--nodelist=gpu4" in a
                                     for a in args), args
        jobs, _ = _submit(tmp)
        with open(os.path.join(tmp, "sb.log.test")) as f:
            probes = f.read().splitlines()
        assert probes and all("--nodelist" not in p for p in probes), probes


_CMD_POS = re.compile(
    r"(?:^|[;&|(`]|(?<![\w$])!|\b(?:then|do|else|if|elif|while|until|exec"
    r"|time|nohup)\b)\s*"
    r"(?:\w+=(?:\"[^\"]*\"|'[^']*'|[^\s\"']*)\s+)*"
    r"((?:\./|\.\./|\"?\$\{?\w+\}?\"?/|(?<![\w/$\"=])[A-Za-z_][\w.-]*/)"
    r"[\w./-]*\.(?:sh|py))(?=[\s;&|)`\"]|$)")


def test_no_job_script_executes_a_repo_script_directly():
    # An uploaded copy can lack +x: on 2026-09-23 './cell_line_ablation/
    # grade.sh' failed with 'Permission denied' after the ablation finished.
    # Helpers go through bash / source / sbatch / $PYTHON_PATH.
    paths = [_ORCH, os.path.join(_ROOT, "clean_previous_runs.sh")]
    for d in ("siamese_sl/slurm", "siamese_sl/cell_line_ablation",
              "sl_comparison/slurm", "SLGNN/slurm", "NSF4SL/slurm"):
        paths += sorted(glob.glob(os.path.join(_ROOT, d, "*.sh")))
    assert len(paths) > 20, paths
    bad = []
    for p in paths:
        for ln in _logical_lines(p):
            if ln.lstrip().startswith(("echo ", "printf ")):
                continue
            if _CMD_POS.search(ln):
                bad.append(f"{os.path.relpath(p, _ROOT)}: {ln.strip()[:120]}")
    assert not bad, bad
    # the check itself catches the 2026-09-23 form and its relatives ...
    for ln in ('MODEL_NAME="x" GRADE_CVS="cv1 cv2" ./cell_line_ablation/grade.sh',
               'cell_line_ablation/grade.sh', 'if ./x.sh; then', '! ./x.sh',
               'exec ./x.sh', 'time ./x.sh', 'while ./x.sh; do',
               'RC=`./x.sh`', 'elif ./x.sh; then', '"$DIR"/x.sh --flag'):
        assert _CMD_POS.search(ln), ln
    # ... and leaves interpreter / source / sbatch invocations alone
    for ln in ('MODEL_NAME="x" bash cell_line_ablation/grade.sh',
               '. sl_comparison/slurm/python_path.sh',
               'source "$SLURM_DIR/config.conf"',
               '$PYTHON_PATH sl_comparison/paper_tables.py --out_dir x',
               'sbatch --parsable slurm/reset_env.sh',
               'rm -f results/x.sh'):
        assert not _CMD_POS.search(ln), ln


def test_step0_freezes_the_folds_before_the_first_job():
    # 2026-09-23: an upload replaced meta.json under the running DAG, and the
    # folds job's refusal cancelled every shared model. Step 0 unfreezes,
    # regenerates and freezes the set before anything is queued; a standalone
    # run_shared_folds.sh regeneration unfreezes and refreezes likewise.
    def order(lines, *needles):
        return [next(i for i, ln in enumerate(lines) if re.search(n, ln))
                for n in needles]
    got = order(_logical_lines(_ORCH),
                r"chmod -R u\+w sl_comparison/folds",
                r"prepare_folds\.py --seed",
                r"chmod -R a-w sl_comparison/folds",
                r"\bsbatch\s+--parsable\b")
    assert got == sorted(got) and len(set(got)) == 4, got
    got = order(_logical_lines(os.path.join(
                    _ROOT, "sl_comparison", "slurm", "run_shared_folds.sh")),
                r"chmod -R u\+w sl_comparison/folds",
                r"prepare_folds\.py --seed",
                r"chmod -R a-w sl_comparison/folds")
    assert got == sorted(got) and len(set(got)) == 3, got


def test_code_manifest_and_synlethkg_freeze_precede_the_first_job():
    # 2026-09-23: code changed under the queued jobs unrecorded, and
    # data/synlethkg vanished mid-run. The orchestrator records the code
    # (CODE_MANIFEST, re-checked by paper_tables) and makes SynLethKG
    # read-only before anything is queued.
    lines = _logical_lines(_ORCH)
    def first(pat):
        return next(i for i, ln in enumerate(lines) if re.search(pat, ln))
    i_code = first(r"sl_comparison/code_manifest\.py > \"\$CODE_MANIFEST\"")
    i_exp = first(r"^\s*export CODE_MANIFEST\b")
    i_kg = first(r"chmod -R a-w data/synlethkg")
    i_sb = first(r"\bsbatch\s+--parsable\b")
    assert i_code < i_exp < i_sb and i_kg < i_sb, (i_code, i_exp, i_kg, i_sb)
    # the manifest tool itself: scope, format and drift detection
    sys.path.insert(0, os.path.join(_ROOT, "sl_comparison"))
    try:
        import code_manifest as cm
    finally:
        sys.path.pop(0)
    with tempfile.TemporaryDirectory() as tmp:
        for rel, txt in (("siamese_sl/train.py", "a"),
                         ("siamese_sl/slurm/config.conf", "b"),
                         ("siamese_sl/slurm/submit_pipeline.sh", "login"),
                         ("siamese_sl/tests/test_x.py", "test"),
                         ("siamese_sl/results/x/model.py", "output"),
                         ("sl_comparison/evaluate_model.py", "c"),
                         ("paper/main.py", "not code of a job")):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)),
                        exist_ok=True)
            with open(os.path.join(tmp, rel), "w") as f:
                f.write(txt)
        assert cm.code_files(tmp) == ["siamese_sl/slurm/config.conf",
                                      "siamese_sl/train.py",
                                      "sl_comparison/evaluate_model.py"]
        man = os.path.join(tmp, "m.sha256")
        with open(man, "w") as f:
            for rel, h in cm.manifest(tmp).items():
                f.write(f"{h}  {rel}\n")
        assert cm.drift(tmp, man) == []
        with open(os.path.join(tmp, "siamese_sl/train.py"), "a") as f:
            f.write("changed")
        os.remove(os.path.join(tmp, "sl_comparison/evaluate_model.py"))
        with open(os.path.join(tmp, "sl_comparison/new.sh"), "w") as f:
            f.write("new")
        assert cm.drift(tmp, man) == [
            "siamese_sl/train.py (changed)", "sl_comparison/evaluate_model.py (removed)",
            "sl_comparison/new.sh (new)"], cm.drift(tmp, man)


# ---------------------------------------------------------------------------
# The root orchestrator, run whole
# ---------------------------------------------------------------------------
# run_benchmarking_siamese_vs_others.sh in a COPY of the code tree, with stub
# sbatch / squeue / sinfo and made-up pair files: step 0 really runs (it
# builds the copy's folds from them), nothing is queued anywhere, and nothing
# outside the tempdir is written. config.conf's PYTHON_PATH in the copy is a
# stand-in that runs this test's interpreter for the login-node helpers and
# only reports every other call.

_SKIP_COPY = shutil.ignore_patterns(
    ".git", "__pycache__", "results", "results_preflight", "logs", "data",
    "external data", "paper", "timestamps", "folds", "folds_preflight",
    "checkpoints", "embeddings_cache", "tests", "original_tensorflow_code",
    "kg_meta_only", "BC_data", "qmd", "Simpson_qmd", "supplementary material",
    "graph_transformer_code", "GNN_Adamson", "slurm_scripts", "node_modules",
    "*.npy", "*.pt", "*.ipynb", "*.png", "*.pdf", ".DS_Store")


_JOBS = {  # script -> the name the graphs below use (+ "/<arm>" per arm)
    "reset_env.sh": "reset", "run_generate_embeddings.sh": "gen",
    "run_best_per_category.sh": "bestcat",
    "run_best_per_category_combo.sh": "combo", "eval_finetuning.sh": "eval",
    "run_ablation.sh": "ablation", "run_ablation_summary.sh": "summary",
    "run_cell_line_ablation.sh": "cellabl", "run_shared_folds.sh": "folds",
    "run_shared_models.sh": "shared", "run_compare.sh": "compare",
    "run_external_transfer.sh": "ext", "run_paper_tables.sh": "tables",
    "run_allmod_combo.sh": "allmod", "run_probe.sh": "probe",
    "run_elisl.sh": "elisl", "build_elisl_inputs.sh": "elisl_inputs",
    _PAIR_SH: "export"}


def _root_tree(tmp, _defaults=False):
    """The copy: code, made-up pairs (120 genes, 3,000 pairs, 20 listed as
    both SL and non-SL), empty stand-ins for the inputs the script only
    checks for, the stubs; its confs' defaults those of _pin_old_defaults
    (the shipped ones with _defaults=True). Returns its root."""
    import numpy as np
    root = os.path.join(tmp, "repo")
    shutil.copytree(_ROOT, root, ignore=_SKIP_COPY)
    if not _defaults:
        _pin_old_defaults(os.path.join(root, "siamese_sl", "slurm"))
    data = os.path.join(root, "data")
    os.makedirs(os.path.join(data, "embeddings_cache", "scGPT_human"))
    for f in ("best_model.pt", "vocab.json"):
        open(os.path.join(data, "embeddings_cache", "scGPT_human", f),
             "w").close()
    rng = np.random.default_rng(3)
    genes = [str(1000 + k) for k in range(120)]
    iu = np.triu_indices(len(genes), 1)
    pick = rng.choice(len(iu[0]), size=3000, replace=False)
    lines = ["K562", "JURKAT", "HELA", "A549", "RPE1"]
    for stem, sel in (("SL_SynLethDB_experimental", pick[:750]),
                      ("SL_SynLethDB_experimental_negatives", pick[730:])):
        pairs = [(genes[iu[0][k]], genes[iu[1][k]]) for k in sel]
        with open(os.path.join(data, stem + ".txt"), "w") as f:
            f.writelines(f"{a}\t{b}\n" for a, b in pairs)
        with open(os.path.join(data, stem + ".sources.tsv"), "w") as f:
            f.write("entrez1\tentrez2\tsources\tcell_lines\tpubmed_ids\n")
            f.writelines(f"{a}\t{b}\tHigh Throughput\t{lines[k % 5]}\t1\n"
                         for k, (a, b) in enumerate(pairs))
    # the screens eval_finetuning.conf names (existence is all that is checked)
    with open(os.path.join(_SIAMESE, "slurm", "eval_finetuning.conf")) as f:
        screens = re.findall(r'^\s*"[A-Za-z]+:([^:"]+):', f.read(), re.M)
    assert len(screens) == 4, screens  # Adamson, AdamsonGI, Corn, Gilbert
    for rel in screens:
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        open(os.path.join(root, rel), "w").close()
    b = os.path.join(tmp, "bin")
    os.makedirs(b)
    stubs = {
        "py": '#!/bin/bash\n'
              'if [ -n "${PY_BROKEN:-}" ]; then\n'
              '  echo "ModuleNotFoundError: No module named \'torch\'" >&2\n'
              '  exit 1\nfi\n'
              'case "$1" in\n'
              '  -c|-|*prepare_folds.py|*code_manifest.py|*arch_signature.py'
              '|*embedding_versions.py)\n'
              f'    exec "{sys.executable}" "$@" ;;\n'
              'esac\n'
              # PY_ARCH_LOG: each settings probe's arguments, one line each
              'if [ -n "${PY_ARCH_LOG:-}" ]; then case " $* " in\n'
              '  *" --print_arch "*) echo "$*" >> "$PY_ARCH_LOG" ;; esac\n'
              'fi\n'
              # the settings probe (`train.py ... --print_arch`):
              # PY_TRAIN=real runs the copy's real parser,
              # PY_TRAIN=<rc>:<stderr text> makes it fail that way
              'case "${PY_TRAIN:-}:$1: $* " in\n'
              '  real:train.py:*" --print_arch "*)\n'
              f'    exec "{sys.executable}" "$@" ;;\n'
              '  [0-9]*:train.py:*" --print_arch "*)\n'
              '    echo "${PY_TRAIN#*:}" >&2; exit "${PY_TRAIN%%:*}" ;;\n'
              'esac\n'
              # PY_ELISL_STALE=1: ELISL's input check fails (a build is due)
              'if [ -n "${PY_ELISL_STALE:-}" ] && [ "$*" = '
              '"sl_comparison/slurm/check_shared_inputs.py --models ELISL" ]; '
              'then echo "stub-python $*" >&2; exit 1; fi\n'
              'echo "stub-python $*" >&2\n',
        # logs "<id>|<RUN_TOKEN>|<args>" and what the job would inherit
        "sbatch": '#!/bin/bash\n'
                  'case " $* " in *" --test-only "*)\n'
                  '  echo "sbatch: Job 1 to start" >&2; exit 0 ;; esac\n'
                  'if [ -n "${SB_FAIL_ON:-}" ]; then case "$*" in\n'
                  '  *"$SB_FAIL_ON") echo "sbatch: error: Batch job '
                  'submission failed" >&2; exit 1 ;; esac; fi\n'
                  'n=$(( $(cat "$SB_CTR" 2>/dev/null || echo 1000) + 1 ))\n'
                  'echo $n > "$SB_CTR"\n'
                  'echo "$n|$RUN_TOKEN|$*" >> "$SB_LOG"\n'
                  'echo "$n|${PAIR_EXPORT_FT_SPLITS:-}|'
                  '${SHARED_FOLDS_FROZEN:-}|${FOLDS_MANIFEST:-}|'
                  '${CODE_MANIFEST:-}|${PCA_ARM-<unset>}|'
                  '${PCA_ARMS-<unset>}|${PCA_MODALITY_VARIANCE-<unset>}|'
                  '${PCA_JOINT_VARIANCE-<unset>}|${PCA_JOINT_DIM-<unset>}"'
                  ' >> "$SB_LOG.env"\n'
                  'echo "$n|${SIAMESE_VARIANTS_LIST-<unset>}|'
                  '${USE_CELL_LINES-<unset>}" >> "$SB_LOG.vars"\n'
                  'echo "$n|${SELECTION_FRAC-<unset>}|'
                  '${SELECTION_SEED-<unset>}|${SELECTION_MODE-<unset>}|'
                  '${ALLMOD_COMBO-<unset>}|${SELECTION_VARIANT-<unset>}|'
                  '${BESTCAT_EXTRA_VARIANTS-<unset>}" >> "$SB_LOG.sel"\n'
                  'echo $n\n',
        "squeue": '#!/bin/bash\n[ -z "${SQUEUE_OUT:-}" ] || cat "$SQUEUE_OUT"\n',
        "sinfo": '#!/bin/bash\necho "gpu3 mix"\n',
    }
    for name, text in stubs.items():
        with open(os.path.join(b, name), "w") as f:
            f.write(text)
        os.chmod(os.path.join(b, name), 0o755)
    conf = os.path.join(root, "siamese_sl", "slurm", "config.conf")
    with open(conf) as f:
        text = f.read()
    text, n = re.subn(r"(?m)^PYTHON_PATH=.*$",
                      'PYTHON_PATH="%s"' % os.path.join(b, "py"), text)
    assert n == 1
    with open(conf, "w") as f:
        f.write(text)
    return root


def _root_run(tmp, *flags, _line=None, _sh="bash", _defaults=False,
              _elisl=False, **env):
    """One run of the copy's orchestrator: (rc, output, jobs), jobs in
    submission order as {name: (dependency kind, [parent names], kill-on-
    invalid-dep, --array or '', singleton)} plus '_env', '_args', '_logs',
    '_tokens', '_ids' and '_vars' (per name: what the job inherits, the arguments
    after its script, its --output / --error, the RUN_TOKEN it gets, its
    job id, its SIAMESE_VARIANTS_LIST and USE_CELL_LINES). A job of one PCA arm is named <job>/<arm> after the PCA_ARM it
    is submitted with (in sbatch's environment, _as_arm; an owner's Step-1
    array: bestcat/<owner>), a job of every arm (_SHARED_SCRIPTS: none of
    its own, so the orchestrator's PCA_ARM=none) by itself, and the table
    across the arms "pca_arms". Every submission is checked to pass a plain
    --export=ALL. PCA_ARMS defaults to "none" (one arm, the single-arm
    contracts); None leaves the tree's config.conf default (all four,
    _pin_old_defaults); _defaults=True sets no PCA_ARMS either (a tree built
    with _defaults=True: the shipped defaults). _line: a
    command the orchestrator printed, run as pasted into a fresh `_sh` shell
    in the repo root (env sets only the stubs: no setting of a test's, nor
    PCA_ARMS), instead of the script with `flags`. ELISL's jobs (submitted
    with the shared models since 2026-11-13) are moved to jobs['_elisl'] and
    dropped from every other job's parents, so the graphs below stay those
    of the other stages; _elisl=True keeps them in the graph
    (test_root_submits_elisl_with_the_shared_models)."""
    root, b = os.path.join(tmp, "repo"), os.path.join(tmp, "bin")
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "PAIR_EXPORT_", "SB_", "SIAMESE_",
                              "EVAL_", "BESTCAT_", "PCA_", "POST_PCA_"))
         and k not in ("RUN_TOKEN", "GPU_NODES", "GPU_EXCLUDE", "CVS", "SEED",
                       "MAX_CONCURRENT", "SHARED_CONCURRENT", "ARCH_ARGS",
                       "USE_CELL_LINES", "SELECTION_VARIANT", "PYTHON_PATH",
                       "SELECTION_CV", "PREPROCESSING_FIT_SCOPE",
                       "CELL_LINE_DIM", "PER_HEAD", "BEST_CAT_ALLOW_MISSING",
                       "BEST_CAT_PCA_VARIANCE", "ABLATION_PURGE_CHECKPOINTS",
                       "ABLATION_PURGE_PREDICTIONS", "TORCH_SPEC",
                       "TORCH_INDEX_URL",
                       "ALLOW_LIVE_JOBS", "CODE_MANIFEST", "FOLDS_MANIFEST",
                       "SHARED_FOLDS_FROZEN", "PY_BROKEN", "SQUEUE_OUT",
                       "PY_TRAIN", "PY_ARCH_LOG", "SIAMESE_ENCODER_TYPE")
         and k not in _PINNED_NAMES}
    if _line is None and not _defaults:
        env.setdefault("PCA_ARMS", "none")
    e.update(PATH=b + os.pathsep + e["PATH"], PYTHONDONTWRITEBYTECODE="1",
             SB_LOG=os.path.join(tmp, "sb.log"),
             SB_CTR=os.path.join(tmp, "sb.ctr"),
             **{k: v for k, v in env.items() if v is not None})
    for f in (e["SB_LOG"], e["SB_LOG"] + ".env", e["SB_LOG"] + ".vars",
              e["SB_LOG"] + ".sel", e["SB_CTR"]):
        if os.path.exists(f):
            os.remove(f)
    cmd = ([_sh, "-c", _line] if _line is not None else
           ["bash", "./run_benchmarking_siamese_vs_others.sh", *flags])
    r = subprocess.run(cmd, cwd=root, env=e, capture_output=True, text=True,
                       timeout=600)
    jobs, ids, tokens, script_args, logs = {}, {}, {}, {}, {}
    if os.path.exists(e["SB_LOG"]):
        # what each job inherits, by job id (the stub's second log): its
        # PCA_ARM is field 5
        with open(e["SB_LOG"] + ".env") as f:
            env_of = {v[0]: v[1:] for v in
                      (ln.rstrip("\n").split("|") for ln in f)}
        with open(e["SB_LOG"]) as f:
            for ln in f:
                jid, tok, args = ln.rstrip("\n").split("|", 2)
                toks = args.split()
                # sbatch's options, the job script, then the script's own
                # arguments (run_paper_tables.sh passes them to paper_tables)
                i = next(k for k, x in enumerate(toks)
                         if os.path.basename(x) in _JOBS)
                opts, name = toks[:i], _JOBS[os.path.basename(toks[i])]
                _plain_export(" ".join(opts))
                arm = env_of[jid][4]
                assert arm in _ARMS, (arm, ln)
                # The only job that takes an argument: the table across the
                # arms. Anything else after a script (an argument, a
                # misplaced sbatch option) is a bug.
                if name == "tables" and toks[i + 1:] == ["--pca_arms_table"]:
                    name = "pca_arms"
                else:
                    assert toks[i + 1:] == [], ln
                    if os.path.basename(toks[i]) in _SHARED_SCRIPTS:
                        assert arm == "none", ln   # the orchestrator's own
                    else:
                        name = f"{name}/{arm}"
                assert name not in jobs, (name, ln)
                dep = next((x.split("=", 1)[1] for x in opts
                            if x.startswith("--dependency=")), "")
                single = dep.endswith(",singleton")
                kind, _, parents = dep.replace(",singleton", "").partition(":")
                jobs[name] = (
                    kind, [ids.get(p, p) for p in parents.split(":") if p],
                    "--kill-on-invalid-dep=yes" in opts,
                    next((x.split("=", 1)[1] for x in opts
                          if x.startswith("--array=")), ""), single)
                script_args[name] = toks[i + 1:]
                logs[name] = tuple(next((x.split("=", 1)[1] for x in opts
                                         if x.startswith(o)), "")
                                   for o in ("--output=", "--error="))
                ids[jid] = name
                tokens[name] = tok
                # every job name is its script's own #SBATCH --job-name
                assert not [x for x in opts
                            if x.startswith(("--job-name", "-J"))], ln
        # one RUN_TOKEN per submission, unless the arms reuse runs of
        # different tokens: then every job of an arm carries its own
        if len(set(tokens.values())) > 1:
            for name, tok in tokens.items():
                arm = name.split("/", 1)[1] if "/" in name else ""
                assert not arm or tok == tokens[next(
                    k for k in tokens if k.endswith("/" + arm))], tokens
        assert set(env_of) == set(ids), (env_of, ids)
        jobs["_env"] = {ids[j]: v for j, v in env_of.items()}
        # (and the variants and USE_CELL_LINES each job inherits)
        with open(e["SB_LOG"] + ".vars") as f:
            jobs["_vars"] = {ids[v[0]]: tuple(v[1:]) for v in
                             (ln.rstrip("\n").split("|") for ln in f)}
        # (and the selection settings, ALLMOD_COMBO, SELECTION_VARIANT and
        # BESTCAT_EXTRA_VARIANTS)
        with open(e["SB_LOG"] + ".sel") as f:
            jobs["_sel"] = {ids[v[0]]: tuple(v[1:]) for v in
                            (ln.rstrip("\n").split("|") for ln in f)}
        jobs["_args"] = script_args
        jobs["_logs"] = logs
        jobs["_tokens"] = tokens
        jobs["_ids"] = {name: jid for jid, name in ids.items()}
        if not _elisl:
            el = {k: jobs.pop(k) for k in _ELISL_JOBS if k in jobs}
            for k, v in list(jobs.items()):
                if not k.startswith("_"):
                    jobs[k] = (v[0], [p for p in v[1] if p not in el],
                               *v[2:])
            jobs["_elisl"] = el
    return r.returncode, r.stdout + r.stderr, jobs


def _ok(kind, *parents, array=""):
    """An afterok edge (it must carry --kill-on-invalid-dep=yes)."""
    return (kind, list(parents), kind == "afterok", array, False)


# Every compare pin collect_and_regrade_siamese.sh reads (one per variant and
# CV), and config.conf's default PCA targets as the commands printed to paste
# set them.
_PINS = tuple("SIAMESE_RESULTS_" + (f"{t}_" if t else "") + cv
              for t in ("", "pd_pooled", "bilinear_per_head", "bilinear_pooled")
              for cv in ("cv1", "cv2", "cv3"))
_TARGETS0 = ("PCA_MODALITY_VARIANCE=0.8", "PCA_JOINT_VARIANCE=0.8",
             "PCA_JOINT_DIM=")
# ... and the default run's variants and CVS, as the root's rebuild of the
# arms it leaves out sets them (config.conf's four, USE_CELL_LINES=1, since
# 2026-10-21; CVS empty, run_compare.sh's default, since 2026-10-22).
_VARIANTS0 = ("SIAMESE_VARIANTS_LIST=pd:per_head pd:pooled bilinear:per_head "
              "bilinear:pooled", "USE_CELL_LINES=1", "CVS=")


def _paste_def(*settings, allmod="0"):
    """The first line of a block printed to paste (config.conf
    pasted_env_def), derived here on its own: it defines _paste_env, env
    without the pasting shell's RUN_TOKEN, CODE_MANIFEST and every compare
    pin, with these NAME=VALUE settings (a value holding a space quoted),
    then the run's ALLMOD_COMBO, 0 or 1, always last (since 2026-11-13;
    before, ALLMOD_COMBO=1 only when on). `allmod`: its value (0, the
    contracts' pinned default, _pin_old_defaults)."""
    unset = " ".join(f"-u {n}" for n in ("RUN_TOKEN", "CODE_MANIFEST",
                                          "SIAMESE_BASELINE_DISCOVER",
                                          "ALLMOD_COMBO") + _PINS)
    settings = settings + (f"ALLMOD_COMBO={allmod}",)
    words = [f'{k}="{v}"' if " " in v else f"{k}={v}"
             for k, v in (s.split("=", 1) for s in settings)]
    return "_paste_env() { env " + " ".join([unset] + words) + ' "$@"; }'


def _array_tasks(array):
    """How many tasks an --array value (e.g. 0-56, 0-71%8, 2,0) queues; 1
    for no array."""
    if not array:
        return 1
    n = 0
    for part in array.split("%", 1)[0].split(","):
        lo, _, hi = part.partition("-")
        n += int(hi or lo) - int(lo) + 1
    return n


def _full(arms=_ARMS, variants=_VARIANTS_2X2, allmod=False,
          step1="0-56"):
    """The documented root DAG for these PCA arms, in submission order: the
    jobs of every arm once (the env reset, the generation, then the shared
    folds and array, queued before any siamese job since 2026-10-21: under a
    FIFO-like priority, queued after every arm's they ran last), one Step-1
    array per owner (`step1`: its --array), each arm's siamese stages
    (with `allmod`, ALLMOD_COMBO=1, the arm's all-modalities array right
    after its combo, after the generation alone; in arm none, when pd:pooled
    is one of the `variants`, the bilinear probe after it, afterok on that
    arm's combo, 2026-11-13), each arm's compare (after the arm's combo,
    all-modalities array, probe, cell-line ablation and the shared array) /
    external transfer / paper tables, the table across the arms (2 arms or
    more), each arm's export (last, singleton, after its arm's jobs and that
    table)."""
    n_var = len(variants.split())
    probe = "pd:pooled" in variants.split()
    g = {"reset": ("", [], False, "", False), "gen": _ok("afterok", "reset")}
    g["folds"] = _ok("afterok", "gen")
    g["shared"] = _ok("afterok", "folds", array="0-8")
    for o in _owners(arms):
        g[f"bestcat/{o}"] = _ok("afterok", "gen", array=step1)
    for a in arms:
        g[f"combo/{a}"] = _ok("afterany", f"bestcat/{_owner(a, arms)}")
        if allmod:
            g[f"allmod/{a}"] = _ok("afterok", "gen", array="0-2")
        if probe and a == "none":
            g["probe/none"] = _ok("afterok", "combo/none", array="0-2")
        g[f"eval/{a}"] = _ok("afterok", f"combo/{a}",
                             array=f"0-{3 * n_var - 1}")
        g[f"ablation/{a}"] = _ok("afterok", f"combo/{a}",
                                 array=f"0-{18 * n_var - 1}")
        g[f"summary/{a}"] = _ok("afterany", f"ablation/{a}")
        g[f"cellabl/{a}"] = _ok("afterok", f"combo/{a}")

    def extra(a):        # the arm's all-modalities array and probe
        return ([f"allmod/{a}"] if allmod else []) + (
            ["probe/none"] if probe and a == "none" else [])
    for a in arms:
        g[f"compare/{a}"] = _ok("afterany", f"combo/{a}", *extra(a),
                                f"cellabl/{a}", "shared")
        g[f"ext/{a}"] = _ok("afterany", f"eval/{a}", "shared")
        g[f"tables/{a}"] = _ok("afterany", f"compare/{a}", f"ext/{a}",
                               f"cellabl/{a}", f"summary/{a}", f"eval/{a}")
    if len(arms) >= 2:
        g["pca_arms"] = _ok("afterany", *[f"tables/{a}" for a in arms])
    for a in arms:
        g[f"export/{a}"] = ("afterany", [
            f"tables/{a}", f"compare/{a}", f"ext/{a}", f"cellabl/{a}",
            f"summary/{a}", f"eval/{a}"] + (["pca_arms"] if len(arms) >= 2
                                            else []) + extra(a),
            False, "", True)
    return g


_FULL = _full(("none",))     # one arm: the single-arm contracts below


def _graph(jobs):
    return {k: v for k, v in jobs.items() if not k.startswith("_")}


def _queued_count(jobs):
    """(jobs, array tasks) the submission queued: the graph and ELISL's
    jobs (which _root_run sets aside unless _elisl=True)."""
    g = dict(_graph(jobs), **jobs.get("_elisl", {}))
    return len(g), sum(_array_tasks(v[3]) for v in g.values())


def test_root_orchestrator_submits_the_documented_dag():
    # The paper run: every PCA arm of config.conf's default (PCA_ARMS unset),
    # its jobs in the documented order (_full), the exports last and
    # singleton; then the same shape under the stage flags, for every arm.
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        folds = os.path.join(root, "sl_comparison", "folds")
        want = _full()
        # ---- the paper run: 44 jobs (43 + arm none's bilinear probe, since
        # 2026-11-13), the four exports last -------------------------------
        rc, out, jobs = _root_run(tmp, "--force-generate", PCA_ARMS=None)
        assert rc == 0, out[-3000:]
        g = _graph(jobs)
        assert g == want, {k: (g.get(k), want.get(k))
                           for k in set(g) | set(want)
                           if g.get(k) != want.get(k)}
        assert list(g) == list(want) and len(g) == 44, list(g)
        assert list(g)[-4:] == [f"export/{a}" for a in _ARMS], list(g)
        # step 0 ran before the first job: folds built and frozen, both
        # manifests written and exported to every job
        assert os.path.isfile(os.path.join(folds, "meta.json"))
        assert not os.access(os.path.join(folds, "meta.json"), os.W_OK)
        for name, v in jobs["_env"].items():
            frozen, fm, cm = v[1:4]
            assert frozen == "1" and os.path.isfile(fm) \
                and os.path.isfile(cm), (name, v)
        for a in _ARMS:                               # fine-tuned sets wanted
            assert jobs["_env"][f"export/{a}"][0] == "", a
        # ---- a second submission over the frozen folds of the first -------
        rc, out, jobs = _root_run(tmp, PCA_ARMS=None)
        assert rc == 0 and _graph(jobs) == want, out[-3000:]
        # ---- --skip-eval: no eval, so no external-transfer job (its tables
        # are built from this run's eval outputs); the exports are told to
        # skip the fine-tuned sets -----------------------------------------
        rc, out, jobs = _root_run(tmp, "--skip-eval", PCA_ARMS=None)
        g = _graph(jobs)
        assert rc == 0 and not [k for k in g if k.split("/")[0]
                                in ("eval", "ext")], (rc, list(g))
        assert "NOTE: --skip-eval: no external-transfer job either" in out
        # (each arm's header names the stages it queues: 2026-10-22)
        for a in _ARMS:
            assert f"\n--- PCA arm {a}: comparison, paper tables -> " in out, \
                out[-3000:]
        for a in _ARMS:
            assert g[f"tables/{a}"] == _ok("afterany", f"compare/{a}",
                                           f"cellabl/{a}", f"summary/{a}")
            assert g[f"export/{a}"] == ("afterany", [
                f"tables/{a}", f"compare/{a}", f"cellabl/{a}", f"summary/{a}",
                "pca_arms"] + (["probe/none"] if a == "none" else []),
                False, "", True), g[f"export/{a}"]
            assert jobs["_env"][f"export/{a}"][0] == "none"
        assert "external_transfer/" not in out.split("All jobs submitted")[1]
        # ---- --skip-siamese keeps it (the reused combos carry their eval)
        rc, out, jobs = _root_run(tmp, "--skip-siamese", "--skip-eval",
                                  PCA_ARMS=None)
        g = _graph(jobs)
        want_s = {"reset": _FULL["reset"], "folds": _ok("afterok", "reset"),
                  "shared": _ok("afterok", "folds", array="0-8")}
        for a in _ARMS:
            want_s[f"compare/{a}"] = _ok("afterany", "shared")
            want_s[f"ext/{a}"] = _ok("afterany", "shared")
            want_s[f"tables/{a}"] = _ok("afterany", f"compare/{a}", f"ext/{a}")
        want_s["pca_arms"] = _ok("afterany", *[f"tables/{a}" for a in _ARMS])
        assert rc == 0 and g == want_s and list(g) == list(want_s), \
            (rc, g, out[-1500:])
        # (the header names the external transfer it keeps; and none that
        # --skip-external-transfer leaves out: the stages queued, RUN_EXT)
        for a in _ARMS:
            assert (f"\n--- PCA arm {a}: comparison, external transfer, paper "
                    "tables -> ") in out, out[-3000:]
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  "--skip-external-transfer")
        assert rc == 0 and "ext/none" not in jobs \
            and "\n--- PCA arm none: comparison, paper tables -> " in out, \
            out[-3000:]
        # ---- --skip-shared -----------------------------------------------
        rc, out, jobs = _root_run(tmp, "--skip-shared", PCA_ARMS=None)
        g = _graph(jobs)
        assert rc == 0 and "folds" not in g and "shared" not in g
        for a in _ARMS:
            assert g[f"compare/{a}"] == _ok(
                "afterany", f"combo/{a}",
                *(["probe/none"] if a == "none" else []), f"cellabl/{a}")
            assert g[f"ext/{a}"] == _ok("afterany", f"eval/{a}")
        # ---- CVS: a subset, in any order, sizes the shared array -----------
        rc, out, jobs = _root_run(tmp, CVS="cv3  cv1")
        assert rc == 0 and jobs["shared"] == _ok("afterok", "folds",
                                                 array="0-5"), out[-1500:]
        # ---- --after <job>: the env reset waits for it, generation is its
        assert _root_run(tmp, "--after", "555")[2]["reset"] == \
            _ok("afterany", "555")
        rc, out, jobs = _root_run(tmp, "--after", "555_3", "--skip-reset-env",
                                  PCA_ARMS=None)
        g = _graph(jobs)
        assert rc == 0 and "reset" not in g and "gen" not in g
        for o in ("none", "per_modality"):
            assert g[f"bestcat/{o}"] == _ok("afterok", "555_3", array="0-56")
        assert g["folds"] == _ok("afterok", "555_3")


def test_root_submits_elisl_with_the_shared_models():
    # Since 2026-11-13 ELISL is a shared-model job of the root DAG: its array
    # (one task per CV of CVS) afterok on the folds job, every arm's compare
    # afterany on it; its inputs job only when check_shared_inputs.py
    # --models ELISL fails (after the env reset; the download runs on the
    # login node, and a missing SEQVEC_BIN refuses before anything changes).
    # --skip-elisl and --skip-shared queue neither, and --skip-elisl keeps
    # the previous ELISL rows that a plain run moves aside.
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        rc, out, jobs = _root_run(tmp, _elisl=True)
        g = _graph(jobs)
        assert rc == 0, out[-3000:]
        assert "elisl_inputs" not in g and "current: reused" in out
        assert g["elisl"] == _ok("afterok", "folds", array="0,1,2"), g
        assert "elisl" in g["compare/none"][1], g["compare/none"]
        assert "elisl" not in g["ext/none"][1]
        assert jobs["_logs"]["elisl"][0].endswith(
            os.path.join("ELISL", "slurm", "logs", "elisl_%A_%a.out"))
        rc, out, jobs = _root_run(tmp, CVS="cv3 cv1", _elisl=True)
        assert rc == 0 and jobs["elisl"][3] == "2,0", jobs["elisl"]
        # a stale build: the inputs job after the reset, ELISL after both
        seq = os.path.join(tmp, "seqvec")
        with open(seq, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(seq, 0o755)
        rc, out, jobs = _root_run(tmp, PY_ELISL_STALE="1", SEQVEC_BIN=seq,
                                  _elisl=True)
        g = _graph(jobs)
        assert rc == 0, out[-3000:]
        assert "stub-python ELISL/build_elisl_inputs.py download" in out
        assert g["elisl_inputs"] == _ok("afterok", "reset"), g
        assert g["elisl"] == _ok("afterok", "folds", "elisl_inputs",
                                 array="0,1,2"), g
        assert jobs["_env"] and "elisl_inputs" in jobs["_ids"]
        # ... without a working seqvec: refused, nothing queued
        rc, out, jobs = _root_run(tmp, PY_ELISL_STALE="1",
                                  SEQVEC_BIN=os.path.join(tmp, "nope"),
                                  _elisl=True)
        assert rc == 1 and "Nothing was changed or submitted" in out \
            and not _graph(jobs), out[-2000:]
        # --skip-elisl / --skip-shared: no ELISL job; --skip-elisl keeps the
        # previous rows, a plain run moves them aside
        res = os.path.join(root, "sl_comparison", "results")
        os.makedirs(res, exist_ok=True)
        row = os.path.join(res, "ELISL_cv1.json")
        open(row, "w").close()
        for flag in ("--skip-elisl", "--skip-shared"):
            rc, out, jobs = _root_run(tmp, flag, _elisl=True)
            g = _graph(jobs)
            assert rc == 0 and not set(_ELISL_JOBS) & set(g), (flag, g)
            assert all("elisl" not in v[1] for v in g.values())
            assert os.path.exists(row), flag
        rc, out, jobs = _root_run(tmp, _elisl=True)
        assert rc == 0 and not os.path.exists(row) and os.path.exists(
            os.path.join(res, "stale_shared")), out[-2000:]


def test_root_orchestrator_runs_the_pca_arms_it_is_given():
    # PCA_ARMS picks the arms, any subset, run in canonical order whatever
    # order the list gives: the Step-1 array of each pair (none / joint,
    # per_modality / both) is its first listed arm's, the table across the
    # arms exists only with two arms or more, and without it (one arm, or
    # --skip-paper-tables) every export still waits for every arm's last
    # jobs, so none runs beside a job a result depends on.
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        for arms_env, arms in (("both joint", ("joint", "both")),
                               ("per_modality", ("per_modality",)),
                               ("joint", ("joint",)),
                               ("none both", ("none", "both")),
                               # joint beside its owner, none (it owns no
                               # Step-1 array of its own), per_modality
                               # owning its own
                               ("per_modality joint none",
                                ("none", "joint", "per_modality"))):
            rc, out, jobs = _root_run(tmp, PCA_ARMS=arms_env)
            g, want = _graph(jobs), _full(arms)
            assert rc == 0 and g == want and list(g) == list(want), (
                arms_env, rc, {k: (g.get(k), want.get(k))
                               for k in set(g) | set(want)
                               if g.get(k) != want.get(k)}, out[-1500:])
            # every job inherits the list as config.conf validated it
            assert {v[5] for v in jobs["_env"].values()} == \
                {" ".join(arms)}, (arms_env, jobs["_env"])
            assert ("pca_arms" in g) == (len(arms) >= 2), arms_env
            # a non-owner arm's combo waits on its owner's Step-1 array, the
            # only one of its pair (pinned here, not only by _full: M64)
            if "joint" in arms and "none" in arms:
                assert "bestcat/joint" not in g and g["combo/joint"] == \
                    _ok("afterany", "bestcat/none"), (arms_env, g)
        # --skip-paper-tables with two arms: no tables at all, and each
        # export waits for every arm's last jobs instead of the table
        rc, out, jobs = _root_run(tmp, "--skip-paper-tables",
                                  PCA_ARMS="none per_modality")
        g = _graph(jobs)
        assert rc == 0 and not [k for k in g if k.startswith(
            ("tables", "pca_arms"))], list(g)
        last = {f"{s}/{a}" for a in ("none", "per_modality")
                for s in ("compare", "ext", "cellabl", "summary", "eval")}
        for a in ("none", "per_modality"):
            kind, parents, kill, array, single = g[f"export/{a}"]
            # (arm none's export also waits on its own bilinear probe)
            mine = last | ({"probe/none"} if a == "none" else set())
            assert kind == "afterany" and single and not kill \
                and set(parents) == mine and len(parents) == len(mine), \
                (a, parents)
        assert "sl_comparison/results/pca_arms/" not in \
            out.split("All jobs submitted")[1], out[-1500:]
        for a in ("none", "per_modality"):
            assert f"\n--- PCA arm {a}: comparison, external transfer -> " \
                in out, out[-3000:]
    # ... and paper_tables.py takes the flags run_paper_tables.sh hands it for
    # an arm and across the arms: argparse would otherwise fail the job (2)
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "sl_comparison", "paper_tables.py"),
         "--help"], capture_output=True, text=True, timeout=120,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0 and all(
        re.search(rf"(?m)^\s+{f}\b", r.stdout)
        for f in ("--pca_arm", "--pca_arms", "--pca_arms_table")) \
        and "--no_pca_ablation" not in r.stdout, \
        (r.returncode, r.stdout[-1500:], r.stderr[-1500:])


# what every job inherits by default: SELECTION_FRAC, SELECTION_SEED,
# SELECTION_MODE, ALLMOD_COMBO (config.conf exports them; SELECTION_VARIANT
# and BESTCAT_EXTRA_VARIANTS each job derives from the same confs)
_SEL_DEFAULT = ("0.1", "42", "per_fold", "1")


def test_default_root_dag_is_the_papers_protocol():
    # config.conf's defaults since 2026-11-13 (no pipeline setting in the
    # environment, PCA_ARMS included): arms none and joint (one Step-1 array,
    # none's, of 19 catalog embeddings x 3 CVs x 2 variant blocks = 114
    # tasks), the variants pd:pooled and pd:per_head (eval 6, ablation 36
    # tasks per arm), each arm's all-modalities array after the generation,
    # the bilinear probe in arm none (afterok, with kill, on arm none's combo;
    # arm none's compare and export wait for it; never in arm joint), and the
    # selection split and ALLMOD_COMBO=1 exported into every job.
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp, _defaults=True)
        rc, out, jobs = _root_run(tmp, "--force-generate", _defaults=True)
        assert rc == 0, out[-3000:]
        g = _graph(jobs)
        want = _full(("none", "joint"), variants="pd:pooled pd:per_head",
                     allmod=True, step1="0-113")
        assert g == want and list(g) == list(want), {
            k: (g.get(k), want.get(k)) for k in set(g) | set(want)
            if g.get(k) != want.get(k)}
        assert "bestcat/joint" not in g and "probe/joint" not in g
        assert g["probe/none"] == _ok("afterok", "combo/none", array="0-2")
        assert "probe/none" in g["compare/none"][1] \
            and "probe/none" not in g["compare/joint"][1], g
        # the probe is queued after arm none's combo and before its compare
        order = list(g)
        assert order.index("combo/none") < order.index("probe/none") \
            < order.index("compare/none"), order
        # the jobs' own settings: the arms, the variants and the selection
        assert {v[5] for v in jobs["_env"].values()} == {"none joint"}
        assert {v[0] for v in jobs["_vars"].values()} == {
            "pd:pooled pd:per_head"}, jobs["_vars"]
        assert {v[:4] for v in jobs["_sel"].values()} == {_SEL_DEFAULT}, \
            jobs["_sel"]
        # the banner counts what it queues: the probe included
        m = re.search(r"queues: +(\d+) jobs, (\d+) tasks counting every "
                      r"array task", out)
        assert m and (int(m.group(1)), int(m.group(2))) == \
            _queued_count(jobs), (m and m.groups(), len(g))
        # ... and without pd:pooled there is no probe to queue
        rc, out, jobs = _root_run(tmp, "--force-generate", _defaults=True,
                                  SIAMESE_VARIANTS_LIST="pd:per_head",
                                  SELECTION_VARIANT="pd:per_head")
        assert rc == 0 and "probe/none" not in jobs, (rc, out[-2000:])
        assert _graph(jobs) == _full(("none", "joint"),
                                     variants="pd:per_head", allmod=True), \
            out[-2000:]


def test_default_submit_pipeline_dag_is_the_papers_protocol():
    # submit_pipeline.sh under the same defaults: the env reset and the
    # generation; arm none's benchmark + summary and its 114-task Step-1
    # array (it owns joint's single-embedding runs too); per arm its combo
    # (afterany on none's Step-1 array), its all-modalities array (afterok
    # on the generation, with kill, one task per CV), eval (6 tasks) and
    # ablation (36) afterok the combo, the ablation summary; the exports
    # last, each waiting on every arm's all-modalities array too. Every job
    # gets the arms, the variants, the selection split and ALLMOD_COMBO=1.
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp, _defaults=True)
        jobs, out = _submit(tmp, _defaults=True)
        want = ["reset_env.sh", "run_generate_embeddings.sh",
                "run_embedding_benchmark.sh/none",
                "run_embedding_benchmark_summary.sh/none",
                "run_best_per_category.sh/none"]
        for a in ("none", "joint"):
            want += [f"run_best_per_category_combo.sh/{a}",
                     f"run_allmod_combo.sh/{a}", f"eval_finetuning.sh/{a}",
                     f"run_ablation.sh/{a}", f"run_ablation_summary.sh/{a}"]
        want += [f"{_PAIR_SH}/none", f"{_PAIR_SH}/joint"]
        assert _arm_names(jobs) == want, _arm_names(jobs)
        by = _by_arm(jobs)
        gen = by["run_generate_embeddings.sh"][0]
        step1 = by["run_best_per_category.sh/none"]
        assert " --array=0-113 " in step1[1], step1
        for a in ("none", "joint"):
            combo, args = by[f"run_best_per_category_combo.sh/{a}"]
            assert f" --dependency=afterany:{step1[0]} " in args, (a, args)
            am, args = by[f"run_allmod_combo.sh/{a}"]
            assert (f"--dependency=afterok:{gen} --kill-on-invalid-dep=yes "
                    in args) and " --array=0-2 " in args, (a, args)
            for sc, arr in (("eval_finetuning.sh", "0-5"),
                            ("run_ablation.sh", "0-35")):
                args = by[f"{sc}/{a}"][1]
                assert (f" --dependency=afterok:{combo} "
                        "--kill-on-invalid-dep=yes ") in args \
                    and f" --array={arr} " in args, (a, sc, args)
            for x in (f"{_PAIR_SH}/none", f"{_PAIR_SH}/joint"):
                assert re.search(rf"--dependency=afterany:[0-9:]*\b{am}\b",
                                 by[x][1]), (x, am, by[x][1])
        # the bilinear probe is the root orchestrator's alone (by hand after
        # a submit_pipeline.sh run: README)
        assert "run_probe.sh" not in " ".join(a for _, _, a in jobs)
        with open(os.path.join(tmp, "sb.log.sel")) as f:
            sel = {tuple(ln.rstrip("\n").split("|")[1:5]) for ln in f}
        assert sel == {_SEL_DEFAULT}, sel
        assert ("Selection mode: per_fold; selection split: --selection_frac "
                "0.1 --selection_seed 42") in out, out[:4000]
        # the banner counts what it queues
        m = re.search(r"Queues: +(\d+) jobs, (\d+) tasks counting every "
                      r"array task", out)
        tasks = sum(_array_tasks(m2.group(1)) if m2 else 1 for m2 in (
            re.search(r"--array=(\S+)", a) for _, _, a in jobs))
        assert m and (int(m.group(1)), int(m.group(2))) == (
            len(jobs), tasks), (m and m.groups(), len(jobs), tasks)


def test_root_orchestrator_clears_an_inherited_per_head():
    # eval_finetuning.conf honours an exported PER_HEAD (meant for a direct
    # sbatch of one model). The combos a chained run trains load only with
    # the value that conf derives from USE_CELL_LINES, so one inherited from
    # the calling shell is ignored with a NOTE and reaches no job: any other
    # value stops every eval task at its first checkpoint, after the combo.
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        stub = os.path.join(tmp, "bin", "sbatch")
        seen = os.path.join(tmp, "per_head.seen")
        with open(stub) as f:
            text = f.read()
        assert text.endswith("echo $n\n"), text[-60:]
        with open(stub, "w") as f:      # what each job would inherit
            f.write(text[:-len("echo $n\n")]
                    + 'echo "${PER_HEAD-<unset>}" >> "%s"\necho $n\n' % seen)

        def inherited(**env):
            if os.path.exists(seen):
                os.remove(seen)
            rc, out, jobs = _root_run(tmp, "--force-generate", **env)
            assert rc == 0 and "eval/none" in jobs, out[-2000:]
            with open(seen) as f:
                return out, sorted(set(f.read().split("\n")[:-1]))

        for env in (dict(PER_HEAD="false"), dict(PER_HEAD="True"),
                    dict(PER_HEAD="1"),
                    dict(PER_HEAD="true", USE_CELL_LINES="0")):
            out, got = inherited(**env)
            assert got == ["<unset>"], (env, got)
            assert f"NOTE: ignoring PER_HEAD={env['PER_HEAD']} from the " \
                "environment" in out, (env, out[:600])
        out, got = inherited()
        assert got == ["<unset>"] and "PER_HEAD" not in out, (got, out[:600])


def test_submit_pipeline_clears_an_inherited_per_head():
    # As the root does (test_root_orchestrator_clears_an_inherited_per_head):
    # the chained eval of the combos this run trains loads them only with
    # the PER_HEAD eval_finetuning.conf derives from USE_CELL_LINES, so one
    # inherited from the calling shell, or from the shell a printed line is
    # pasted into, is ignored with a NOTE and reaches no job (2026-10-22: it
    # reached every eval task, which then refused each cell-line
    # checkpoint); --eval-only, a submission of one model, keeps it.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        _env_stubs(tmp)

        def eval_env(jobs):
            ev = [j for j, _, a in jobs if a.endswith("/eval_finetuning.sh")]
            assert len(ev) == 1, jobs
            with open(os.path.join(tmp, f"sb.log.env.{ev[0]}")) as f:
                return dict(x.split("=", 1) for x in f.read().split("\n")
                            if "=" in x)

        base = ("--skip-generate", "--skip-reset-env")
        for v in ("false", "true", "0"):
            jobs, out = _submit(tmp, *base, PER_HEAD=v)
            assert f"NOTE: ignoring PER_HEAD={v} from the environment" in out \
                and "PER_HEAD" not in eval_env(jobs), (v, out[-2000:])
        jobs, out = _submit(tmp, *base)
        assert "PER_HEAD" not in out and "PER_HEAD" not in eval_env(jobs), \
            out[-2000:]
        d = os.path.join(sia, "results", "x_cv3")
        os.makedirs(d)
        for f in ("config.json", "results.json"):
            with open(os.path.join(d, f), "w") as fh:
                fh.write("{}")
        jobs, out = _submit(tmp, "--eval-only", "results/x_cv3",
                            PER_HEAD="false")
        assert eval_env(jobs).get("PER_HEAD") == "false" \
            and "ignoring PER_HEAD" not in out, out[-2000:]


def test_root_subset_run_that_regenerates_drops_every_variants_markers():
    # Step 0a removes the markers of the variants this run retrains. When
    # the run also regenerates the embeddings, the markers of the variants it
    # does NOT retrain go too (their combos were trained on the old vectors);
    # with --skip-generate they stay, and --skip-siamese keeps them all.
    names = ["latest_combo_cv1.txt", "var_pd_pooled__latest_combo_cv2.txt",
             "var_bilinear_per_head__latest_combo_cv3.txt",
             "var_bilinear_pooled__latest_combo_cv1.txt"]
    subset = dict(SIAMESE_VARIANTS_LIST="pd:per_head bilinear:pooled")
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        res = os.path.join(root, "siamese_sl", "results")
        os.makedirs(res)

        def left(*flags, **env):
            for n in names:
                with open(os.path.join(res, n), "w") as f:
                    f.write("results/x\n")
            rc, out, _ = _root_run(tmp, *flags, **env)
            assert rc == 0, out[-2500:]
            return sorted(n for n in os.listdir(res) if n.endswith(".txt"))

        assert left("--force-generate", **subset) == []
        assert left("--skip-generate", **subset) == sorted(names[1:3])
        # --after: this script cannot tell whether JOB regenerates, so the
        # other variants' markers are KEPT (the usage lines say to remove
        # them by hand when it does)
        assert left("--after", "555", **subset) == sorted(names[1:3])
        assert left("--force-generate") == []
        assert left("--skip-siamese") == sorted(names)


def test_root_orchestrator_refuses_before_it_changes_anything():
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        rc, out, jobs = _root_run(tmp, "--force-generate")
        assert rc == 0, out[-3000:]
        res = os.path.join(root, "sl_comparison", "results")
        os.makedirs(os.path.join(res, "paper_tables"), exist_ok=True)
        with open(os.path.join(res, "comparison.csv"), "w") as f:
            f.write("previous run\n")

        def state():
            out = {}
            for dp, _, fn in os.walk(os.path.join(root, "sl_comparison")):
                if os.sep + "logs" in dp:
                    continue
                for f in fn:
                    q = os.path.join(dp, f)
                    out[os.path.relpath(q, root)] = os.stat(q).st_mtime_ns
            return out

        before = state()

        def refused(needle, *flags, **env):
            rc, out, jobs = _root_run(tmp, *flags, **env)
            assert rc == 1 and needle in out and not jobs, (rc, flags, env,
                                                            out[-1500:])
            assert state() == before, (flags, env)
            return out

        # --after with no job id swallowed the next flag (or an empty $GEN)
        for bad in (("--after", "--skip-eval"), ("--after", ""),
                    ("--after", "siamese_gen_emb"), ("--after", "555_"),
                    ("--after", "_5"), ("--after", "5_5_5"),
                    ("--after", "5 6"),
                    ("--skip-reset-env", "--after", "--force-generate"),
                    ("--after", )):
            refused("needs a job ID", *bad)
        # the venv is missing or half-installed (a reset_env job that was
        # cancelled, failed or is still rebuilding it): named as that, not
        # as missing SLGNN / NSF4SL inputs
        # (the rebuild it prints carries the settings taken from the
        # calling shell: a one-shot TORCH_SPEC, which reset_env.sh reads)
        out = refused("PYTHON_PATH cannot run the login-node checks",
                      "--force-generate", PY_BROKEN="1",
                      TORCH_SPEC="torch==2.5.1")
        assert "No module named 'torch'" in out \
            and "Nothing was changed or submitted." in out \
            and "rebuild or copy the inputs" not in out \
            and ("\n    (cd siamese_sl && TORCH_SPEC=torch==2.5.1 sbatch "
                 "slurm/reset_env.sh)\n") in out, out[-1500:]
        refused("PYTHON_PATH cannot run the login-node checks",
                "--skip-shared", PY_BROKEN="1")
        # --help needs no interpreter
        rc, out, _ = _root_run(tmp, "--help", PY_BROKEN="1")
        assert rc == 0 and "ROOT pipeline orchestrator" in out
        # a flag that names nothing, and nothing left to run
        refused("unknown option", "--no-such-flag")
        refused("leave nothing to run", "--skip-siamese", "--skip-shared")
        # a mistyped export value
        refused("PAIR_EXPORT_UNIVERSE", PAIR_EXPORT_UNIVERSE="Shared")
        # run_best_per_category.conf's own values (the combo's Step 2 was
        # the only place that refused a selection CV that names no CV)
        refused("BESTCAT_EXTRA_VARIANTS must be 0 or 1",
                BESTCAT_EXTRA_VARIANTS="yes")
        for bad in ("cv4", "CV1", "cv1 cv2"):
            refused(f"SELECTION_CV='{bad}' is not one of CV_TYPES",
                    SELECTION_CV=bad)
        # a sidecar that lost rows (a copy that stopped part-way): the pairs
        # it lost would be trained at the OTHER head
        side = os.path.join(root, "data", "SL_SynLethDB_experimental_negatives"
                            ".sources.tsv")
        with open(side) as f:
            rows = f.readlines()
        try:
            with open(side, "w") as f:
                f.writelines(rows[:-40])
            before = state()
            out = refused("have no row in", "--force-generate")
            assert "40 pair(s)" in out and "Nothing was changed" in out, out
        finally:
            with open(side, "w") as f:
                f.writelines(rows)
        before = state()
        # CVS that is not a list of cv1 / cv2 / cv3, each once (it would
        # fail every shared task and the SLeuth rows of the compare)
        for bad in ("cv1,cv2", "cv4", "CV1", "cv1 cv1", " "):
            assert "CVS='" in refused("want a space-separated subset",
                                      CVS=bad)
        refused("want a space-separated subset", "--skip-shared", CVS="cv9")

        # ---- an sbatch refused part-way: what is queued is named, and the
        # venv reset_env is deleting is not forgotten -----------------------
        # (each command alone on its line: a trailing "# ..." reached
        # scancel as job ids in an interactive zsh; the venv rebuild with the
        # settings taken from the calling shell, TORCH_SPEC here)
        rc, out, jobs = _root_run(tmp, "--force-generate", "--skip-elisl",
                                  SB_FAIL_ON="slurm/run_ablation.sh",
                                  TORCH_SPEC="torch==2.5.1")
        # (arm none's bilinear probe, 2026-11-13, right after its combo)
        assert rc == 1 and list(_graph(jobs)) == [
            "reset", "gen", "folds", "shared", "bestcat/none", "combo/none",
            "probe/none", "eval/none"], (rc, list(jobs))
        assert "submission stopped part-way; already queued: 1001 1002 1003 " \
            "1004 1005 1006 1007 1008" in out \
            and "\n    scancel 1001 1002 1003 1004 1005 1006 1007 1008\n" \
            in out, out[-1500:]
        assert "reset_env (1001) deletes the venv as soon as it starts" in out \
            and ("\n    (cd siamese_sl && TORCH_SPEC=torch==2.5.1 sbatch "
                 "slurm/reset_env.sh)\n") in out, out[-1500:]
        # every arm's: refused at the first arm's ablation, the shared folds
        # and array and both owners' Step-1 arrays (queued before any arm's
        # stages) are named too
        rc, out, jobs = _root_run(tmp, "--force-generate", "--skip-elisl",
                                  PCA_ARMS=None,
                                  SB_FAIL_ON="slurm/run_ablation.sh")
        assert rc == 1 and list(_graph(jobs)) == [
            "reset", "gen", "folds", "shared", "bestcat/none",
            "bestcat/per_modality", "combo/none", "probe/none",
            "eval/none"], (rc, list(jobs))
        assert ("\n    scancel 1001 1002 1003 1004 1005 1006 1007 1008 1009\n"
                in out), out[-1500:]
        rc, out, jobs = _root_run(tmp, "--skip-elisl", "--force-generate", "--skip-reset-env",
                                  SB_FAIL_ON="slurm/run_ablation.sh")
        assert rc == 1 and \
            "\n    scancel 1001 1002 1003 1004 1005 1006 1007\n" in out \
            and "deletes the venv" not in out, out[-1500:]
        # the export alone refused: a WARNING, every other job stays queued
        rc, out, jobs = _root_run(tmp, "--skip-elisl", "--force-generate",
                                  SB_FAIL_ON="slurm/" + _PAIR_SH)
        assert rc == 0 and "export/none" not in jobs \
            and len(_graph(jobs)) == len(_FULL) - 1 == 14 \
            and "WARNING" in out, (rc, list(jobs))


def test_root_orchestrator_asks_train_py_about_the_training_settings():
    # ARCH_ARGS, PCA_METHOD, PREPROCESSING_FIT_SCOPE, SIAMESE_ENCODER_TYPE,
    # CELL_LINE_DIM and the PCA arms' targets come from the
    # environment (config.conf), and only train.py's parser judges them. No
    # job runs it before reset_env and generation have finished: a value it
    # refuses failed every Step-1 task hours in, or (an arm's joint PCA,
    # which only its combo and ablation pass) every combo after the Step-1
    # array, and the failed combo job cancels eval, ablation and the
    # cell-line ablation. The orchestrator asks the parser itself
    # (--print_arch: no data loaded, nothing written) before it changes
    # anything. PY_TRAIN=real: the stand-in interpreter hands that one call
    # to the copy's real train.py.
    published = ("--layernorm --log_temperature --no_per_head_last_layer_bias"
                 " --select_metric auprg --no_gene_term --no_pair_features")
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)

        def state():
            out = {}
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d != "logs"]
                for f in fn:
                    q = os.path.join(dp, f)
                    out[os.path.relpath(q, root)] = os.stat(q).st_mtime_ns
            return out

        before = state()
        for env, said in (
                (dict(ARCH_ARGS="--layernrom"),
                 "unrecognized arguments: --layernrom"),
                (dict(PCA_METHOD="Robust"),
                 "argument --pca_method: invalid choice"),
                (dict(CELL_LINE_DIM="24d"),
                 "argument --cell_line_dim: invalid int value"),
                (dict(SEED="abc"), "argument --seed: invalid int value"),
                (dict(SIAMESE_ENCODER_TYPE="MLP"),
                 "argument --siamese_encoder_type: invalid choice"),
                (dict(PREPROCESSING_FIT_SCOPE="ALL"),
                 "argument --preprocessing_fit_scope: invalid choice"),
                # an arm's target in a form config.conf takes but argparse
                # refuses (it rounds to 1.0): joint's combos, per_modality's
                # every run
                (dict(PCA_ARMS="joint",
                      PCA_JOINT_VARIANCE="0.99999999999999999999"),
                 "--post_pca_variance must be in (0, 1)"),
                (dict(PCA_ARMS="per_modality",
                      PCA_MODALITY_VARIANCE="0.99999999999999999999"),
                 "--pca_variance must be in (0, 1)")):
            rc, out, jobs = _root_run(tmp, "--force-generate",
                                      PY_TRAIN="real", **env)
            assert rc == 1 and not jobs, (env, rc, out[-1500:])
            assert "ERROR: train.py refuses the settings" in out \
                and "train.py: error: " + said in out \
                and "Nothing was changed or submitted." in out, \
                (env, out[-1500:])
            # no folds, no manifest, no __pycache__, no results/_sigcheck
            assert state() == before, env
        # only argparse's own answer (exit status 2 AND "train.py: error:")
        # refuses ...
        rc, out, jobs = _root_run(tmp, "--force-generate",
                                  PY_TRAIN="2:train.py: error: argument --x")
        assert rc == 1 and not jobs and state() == before, (rc, out[-1500:])

        # ... what the parser accepts is submitted, and nothing is printed
        for env, arms in (
                ({}, ("none",)),
                # the published model: arm joint, PCA_JOINT_VARIANCE=0.8
                (dict(ARCH_ARGS=published, PCA_ARMS="joint"), ("joint",)),
                (dict(PCA_METHOD="robust", PREPROCESSING_FIT_SCOPE="all",
                      SIAMESE_ENCODER_TYPE="mlp", CELL_LINE_DIM="8"),
                 ("none",)),
                # an exact count wins (--post_pca_dim): no job passes this
                # fraction
                (dict(PCA_ARMS="joint both", PCA_JOINT_DIM="64",
                      PCA_JOINT_VARIANCE="0.99999999999999999999"),
                 ("joint", "both"))):
            rc, out, jobs = _root_run(tmp, "--force-generate",
                                      PY_TRAIN="real", **env)
            assert rc == 0 and _graph(jobs) == _full(arms), \
                (env, rc, out[-1500:])
            assert "train.py refuses" not in out \
                and "could not check the training settings" not in out, env
        # the single-output model's command line (no --use_cell_lines)
        rc, out, jobs = _root_run(tmp, "--force-generate", PY_TRAIN="real",
                                  USE_CELL_LINES="0")
        assert rc == 0 and "combo/none" in jobs \
            and "cellabl/none" not in jobs \
            and "could not check the training settings" not in out, \
            (rc, out[-1500:])
        # --skip-siamese trains no SLeuth model: nothing to ask
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PY_TRAIN="real",
                                  ARCH_ARGS="--layernrom")
        assert rc == 0 and "combo/none" not in jobs and "shared" in jobs \
            and "train.py refuses" not in out, (rc, out[-1500:])
        # a probe that cannot run (a module this node lacks, no such file, an
        # exit status that is not argparse's) is reported and never stops the
        # submission: the jobs will judge
        for how in ("1:ModuleNotFoundError: No module named 'sklearn'",
                    "2:python3: can't open file 'train.py': [Errno 2] No such"
                    " file or directory",
                    "1:train.py: error: but not argparse's exit status"):
            rc, out, jobs = _root_run(tmp, "--force-generate", PY_TRAIN=how)
            assert rc == 0 and _graph(jobs) == _FULL, (how, rc, out[-1500:])
            assert "WARNING: could not check the training settings" in out \
                and how.split(":", 1)[1] in out, (how, out[-1500:])

    # The probe passes no option the combo's own training call does not (one
    # dropped from the jobs must not linger here and refuse a good run).
    def flags(text):
        return set(re.findall(r"(?<![\w-])--[A-Za-z][\w-]*", text))

    probe = [ln for ln in _logical_lines(os.path.join(
                 _ROOT, "run_benchmarking_siamese_vs_others.sh"))
             if "--print_arch" in ln and " train.py " in ln]
    assert len(probe) == 1, probe
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_best_per_category_combo.sh")) as f:
        combo = f.read()
    call = combo[combo.index("    $PYTHON_PATH train.py \\\n"
                             "        --embeddings_paths $EMB_PATHS"):]
    call = call[:call.index("TRAIN_EXIT=$?")]
    assert flags(probe[0]) - {"--print_arch"} <= flags(call), \
        flags(probe[0]) - flags(call)


def test_root_asks_train_py_once_per_pca_flag_set():
    # Each arm trains with its own PCA flags (config.conf pca_single_args for
    # the Step-1 runs, pca_multi_args for its combos and ablation), and only
    # train.py's parser judges a flag set: the login-node probe asks it once
    # per DISTINCT set of the arms this DAG trains, in PCA_ARMS order (four
    # for the four arms, one for arm none alone), so a value config.conf
    # accepts but argparse refuses stops the run before anything changes,
    # and only when an arm of this run would train with it. A probe that
    # cannot run at all is reported once, not once per set.
    def pca_of(call):
        toks = call.split()
        i = toks.index("--siamese_encoder_type") + 2
        return " ".join(toks[i:toks.index("--kernel", i)])

    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        log = os.path.join(tmp, "arch.log")

        def state():
            out = {}
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d != "logs"]
                for f in fn:
                    q = os.path.join(dp, f)
                    out[os.path.relpath(q, root)] = os.stat(q).st_mtime_ns
            return out

        def probes(*flags, **env):
            if os.path.exists(log):
                os.remove(log)
            rc, out, jobs = _root_run(tmp, *flags, PY_ARCH_LOG=log, **env)
            calls = []
            if os.path.exists(log):
                with open(log) as f:
                    calls = [pca_of(ln) for ln in f.read().splitlines()]
            return rc, out, jobs, calls

        for env, want in (
                (dict(PCA_ARMS=None),
                 ["--no_post_pca", "--post_pca_variance 0.8",
                  "--pca_variance 0.8 --no_post_pca",
                  "--pca_variance 0.8 --post_pca_variance 0.8"]),
                (dict(PCA_ARMS="none"), ["--no_post_pca"]),
                (dict(PCA_ARMS="joint"),
                 ["--no_post_pca", "--post_pca_variance 0.8"]),
                (dict(PCA_ARMS="both", PCA_JOINT_DIM="300",
                      PCA_MODALITY_VARIANCE="0.9"),
                 ["--pca_variance 0.9 --no_post_pca",
                  "--pca_variance 0.9 --post_pca_dim 300"])):
            rc, out, jobs, calls = probes("--force-generate", **env)
            assert rc == 0 and calls == want, (env, calls, out[-1500:])
        # (--skip-siamese trains nothing: nothing to ask)
        rc, out, jobs, calls = probes("--skip-siamese", PCA_ARMS=None)
        assert rc == 0 and calls == [], calls
        # A target config.conf takes (a fraction written 0.<digits>) that
        # argparse refuses (it rounds to 1.0): refused only for the arms that
        # train with it, before step 0, naming them and the flags.
        near_one = "0.99999999999999999999"
        before = state()
        for env, arms, said in (
                (dict(PCA_ARMS="none joint", PCA_JOINT_VARIANCE=near_one),
                 "joint", "--post_pca_variance must be in (0, 1)"),
                (dict(PCA_ARMS="per_modality both",
                      PCA_MODALITY_VARIANCE=near_one), "per_modality both",
                 "--pca_variance must be in (0, 1)")):
            rc, out, jobs = _root_run(tmp, "--force-generate",
                                      PY_TRAIN="real", **env)
            assert rc == 1 and not jobs, (env, rc, out[-1500:])
            flat = " ".join(out.split())
            assert "ERROR: train.py refuses the settings every siamese " \
                f"training of PCA arm(s) {arms} would be started with" \
                in flat and "train.py: error: " + said in out \
                and "Nothing was changed or submitted." in out, \
                (env, out[-1500:])
            assert state() == before, env
        rc, out, jobs = _root_run(tmp, "--force-generate", PY_TRAIN="real",
                                  PCA_ARMS="none per_modality",
                                  PCA_JOINT_VARIANCE=near_one)
        assert rc == 0 and "train.py refuses" not in out, out[-1500:]
        # a probe that cannot run: one WARNING, one probe, the run goes on
        rc, out, jobs, calls = probes(
            "--force-generate", PCA_ARMS=None,
            PY_TRAIN="1:ModuleNotFoundError: No module named 'sklearn'")
        assert rc == 0 and _graph(jobs) == _full() and len(calls) == 1 \
            and out.count("WARNING: could not check the training "
                          "settings") == 1, (calls, out[-1500:])


def test_root_step0a_removes_previous_shared_score_matrices():
    # A submission that retrains the shared models removes the previous run's
    # score matrices where it moves their rows aside: a task that stops
    # before its trainer's purge, or an array cancelled with the folds job,
    # would leave them complete and with a matching fold stamp. Every CV, as
    # for the rows; --skip-shared reuses rows and matrices; a manual run's
    # directory and the other files of a graded directory are never touched.
    graded = ["SLGNN/results/slgnn_cv1", "NSF4SL/results/nsf4sl_cv2",
              "SLMGAE-in-pytorch/results/slmgae_shared_cv3"]
    manual = "NSF4SL/results/nsf4sl_standalone_cv1"
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)

        def left(*flags, **env):
            for d in graded + [manual]:
                os.makedirs(os.path.join(root, d), exist_ok=True)
                for n in ("fold_0_%s_predictions.npy" % d[-3:],
                          "model_params.json"):
                    with open(os.path.join(root, d, n), "w") as f:
                        f.write("x")
            rc, out, _ = _root_run(tmp, *flags, **env)
            assert rc == 0, out[-2500:]
            return sorted(d + "/" + n for d in graded + [manual]
                          for n in os.listdir(os.path.join(root, d)))

        kept = sorted([d + "/model_params.json" for d in graded + [manual]]
                      + [manual + "/fold_0_cv1_predictions.npy"])
        assert left("--skip-siamese") == kept
        assert left("--force-generate", CVS="cv1") == kept
        assert len(left("--skip-shared")) == len(kept) + len(graded)


def test_root_step0a_moves_previous_elisl_rows_aside():
    # ELISL is run by hand (ELISL/slurm/run_elisl.sh), but its row is a
    # tab_tiers peer that passes the fold-stamp gate after the folds are
    # rebuilt byte-identically: a run that retrains the shared tier moves it
    # aside with the other shared rows; --skip-shared keeps it.
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        res = os.path.join(root, "sl_comparison", "results")

        def run(*flags):
            os.makedirs(res, exist_ok=True)
            for n in ("ELISL_cv1.json", "ELISL_cv3.json", "SLGNN_cv2.json"):
                with open(os.path.join(res, n), "w") as f:
                    f.write("{}")
            rc, out, _ = _root_run(tmp, *flags)
            assert rc == 0, out[-2500:]
            stale = os.path.join(res, "stale_shared")
            moved = sorted(n.split(".")[0] for n in os.listdir(stale)) \
                if os.path.isdir(stale) else []
            if os.path.isdir(stale):
                shutil.rmtree(stale)
            return moved

        assert run("--skip-siamese") == ["ELISL_cv1", "ELISL_cv3",
                                         "SLGNN_cv2"]
        assert run("--skip-shared") == []


def test_root_step0a_stays_in_each_arms_roots():
    # Step 0a removes what the DAG rebuilds for the arms of PCA_ARMS, each in
    # its own roots: the siamese tree (markers, selection, ablation and
    # cell-line dirs) with siamese, the single-embedding runs in the roots of
    # this run's owners only (an arm's pair partner owns them when both run),
    # the rows and markers into that arm's own stale_siamese/ and its tables
    # always; the shared rows once (stale_shared/), and the table across the
    # arms always. An arm PCA_ARMS leaves out keeps its siamese outputs and
    # SLeuth rows, but a run that retrains the shared models removes its
    # comparison table, external transfer and paper tables (built on the rows
    # moved aside; named, with what rebuilds them: test_root_wires_the_
    # rebuild_of_the_arms_it_leaves_out), and a regeneration drops its combo
    # markers; with --skip-shared it keeps all.
    gone_sia = {"best_per_category.json", "latest_combo_cv1.txt",
                "var_pd_pooled__latest_combo_cv3.txt", "ablate_no_go_cv1/",
                "var_pd_pooled__ablate_no_kg_cv2/", "ablation_summary.csv",
                "cellline_ablation_cv1/"}
    shared_built = {"comparison.csv", "comparison_notes.txt",
                    "external_transfer/", "paper_tables/"}
    markers = {"latest_combo_cv1.txt", "var_pd_pooled__latest_combo_cv3.txt"}
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        _seed_arms(root)
        seeded = _arm_state(root)

        def left_out(a, keep_shared=False):
            """A left-out arm after a run that retrains the shared models (or
            not): the shared row of arm none's root moved once, the tables
            built on the shared rows gone (or kept)."""
            sia, cmp_, stale = seeded[a]
            cmp_ = set(cmp_)
            if not keep_shared:
                cmp_ -= shared_built
                if a == "none":
                    cmp_ = cmp_ - {"SLGNN_cv1.json"} | {"stale_shared/"}
            return (sia, sorted(cmp_), stale)

        def run(*flags, **env):
            _seed_arms(root)
            rc, out, jobs = _root_run(tmp, *flags, **env)
            assert rc == 0, (flags, env, out[-2500:])
            return _arm_state(root), out

        for arms_env, arms, owners in (
                ("joint", ["joint"], ["joint"]),
                ("none joint", ["none", "joint"], ["none"]),
                ("per_modality both", ["per_modality", "both"],
                 ["per_modality"])):
            got, out = run("--skip-generate", PCA_ARMS=arms_env)
            for a in _ARMS:
                if a not in arms:
                    assert got[a] == left_out(a), (arms_env, a, got[a])
                    assert (f"removed 4 previous-run output(s) of PCA arm {a} "
                            "built on the shared rows") in out, out[-3000:]
                    continue
                sia = sorted(set(_ARM_SIA) - gone_sia
                             - ({"pcavar0.8_go_cv1/"} if a in owners
                                else set()))
                cmp_left = {"stale_siamese/"} | ({"stale_shared/"}
                                                 if a == "none" else set())
                assert got[a] == (sia, sorted(cmp_left), sorted(_ARM_ROWS)), \
                    (arms_env, a, got[a])
            assert not os.path.exists(os.path.join(
                root, "sl_comparison", "results", "pca_arms")), arms_env
        # --skip-shared reuses the shared rows: the other arms keep all
        got, out = run("--skip-generate", "--skip-shared", PCA_ARMS="joint")
        for a in ("none", "per_modality", "both"):
            assert got[a] == left_out(a, keep_shared=True) == seeded[a], \
                (a, got[a])
        assert "built on the shared rows" not in out \
            and "which PCA_ARMS leaves out" not in out, out[-2000:]
        # --skip-siamese: no siamese root is touched, the listed arms'
        # comparison rows (not the *_nocell ones: no cell-line job reruns
        # them) and tables and the table across the arms still go, and the
        # left-out arms' tables built on the shared rows
        got, _ = run("--skip-siamese", PCA_ARMS="both")
        for a in _ARMS:
            want = left_out(a)
            if a == "both":
                want = (seeded[a][0], ["siamese_sl_nocell_cv1.json",
                                       "stale_siamese/"],
                        sorted(set(_ARM_ROWS)
                               - {"siamese_sl_nocell_cv1.json"}))
            assert got[a] == want, (a, got[a])
        assert not os.path.exists(os.path.join(
            root, "sl_comparison", "results", "pca_arms"))
        # a regeneration: every arm's combo markers go, the rest as above
        got, _ = run("--force-generate", PCA_ARMS="joint")
        for a in ("none", "per_modality", "both"):
            assert got[a] == (sorted(set(seeded[a][0]) - markers),) \
                + left_out(a)[1:], (a, got[a])


def test_root_wires_the_rebuild_of_the_arms_it_leaves_out():
    # A run that retrains the shared models names, by their state, the arms
    # it leaves out that have something to tabulate (all without tables by
    # then, whichever submission removed them), in two groups:
    # - with a combo marker: their tables. The final listing prints the jobs
    #   that rebuild them, ready to paste and wired to this run's (ids known
    #   only there): each arm's compare and external transfer afterany the
    #   shared array, its paper tables afterany both, and the table across
    #   every arm afterany those and this run's last table job (its own
    #   table across the arms, which rewrites pca_arms/ over PCA_ARMS only;
    #   else its arms' paper tables; else their compare and external
    #   transfer);
    # - with a selection but no marker (a regeneration removed them): no
    #   combo of theirs counts, so the recipe that retrains them instead,
    #   with what it does to the table across the arms and what a table
    #   across every arm takes.
    # An arm of PCA_ARMS is never named, even when it keeps its state
    # (--skip-siamese; a variant subset, whose other variants' markers
    # stay): this run's own jobs rebuild its tables. A submission that stops
    # part-way says so in its error report, whichever group the arms are in
    # and whether or not anything was queued yet, and its resubmission
    # (nothing left to remove) names them again. The pasted jobs run under
    # _paste_env (the block's first line: none of the pasting shell's
    # RUN_TOKEN, CODE_MANIFEST or compare pins; the arms of the table across
    # the arms, this run's PCA targets and, since 2026-10-21, its variants),
    # a left-out arm's own targets on its lines where its combos were
    # trained at others.
    every = "none joint per_modality both"
    error_note = ("\n  NOTE: left-out PCA arm(s) none per_modality both have "
                  "no tables now (step 0a); resubmitting names them again, "
                  "with what rebuilds them.\n")

    def named(out):
        """(the arms with tables to rebuild, the arms to retrain), as the
        step-0a NOTEs name them."""
        got = []
        for what in ("have no comparison, external-",
                     "have no tables now and no combo"):
            m = re.search(r"NOTE: PCA arm\(s\) ([a-z_ ]+), which PCA_ARMS "
                          r"leaves out, " + re.escape(what), out)
            got.append(m.group(1).split() if m else [])
        return tuple(got)

    def retrain(text, arms):
        """The retraining recipe for `arms`, as the step-0a NOTE or the
        final listing (text) gives it."""
        flat = " ".join(text.split())
        tab = (f"and builds one over {' '.join(arms)} only." if len(arms) > 1
               else "and builds none (one arm).")
        # (each command pins the targets, USE_CELL_LINES and the variants,
        # since 2026-10-22)
        pins = (" ".join(_TARGETS0) + ' USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST='
                '"pd:per_head pd:pooled bilinear:per_head bilinear:pooled" ')
        return (f'{pins}PCA_ARMS="{" ".join(arms)}" bash ./run_benchmarking_'
                "siamese_vs_others.sh --skip-generate --skip-shared That run "
                "removes this run's table across the arms (sl_comparison/"
                f"results/pca_arms/), if any, {tab} A table across every arm, "
                "with its 2x2 effects, needs one submission's selection of "
                "every arm, which retrains this run's arms too (and reuses "
                f"its shared rows): env -u PCA_ARMS {pins}bash ./run_benchmarking_"
                "siamese_vs_others.sh --skip-generate --skip-shared") in flat

    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        sia = os.path.join(root, "siamese_sl")
        real = os.path.realpath(root)

        def run(*flags, _rc=0, **env):
            rc, out, jobs = _root_run(tmp, *flags, **env)
            assert rc == _rc, (flags, env, rc, out[-2500:])
            return out, jobs

        def wired(out, jobs, ran, left, own, targets=None,
                  variants=_VARIANTS0):
            """The final listing's rebuild of `left` (this run: `ran`),
            waiting on this run's shared array and, for the table across
            the arms, on this run's jobs `own`; targets: {arm: the words of
            its own PCA targets on its lines}; variants: this run's, which
            every pasted job gets."""
            tail = out.split("All jobs submitted")[1]
            ids = jobs["_ids"]
            arms = [a for a in _ARMS if a in left or a in ran]
            assert (f"\n      {_paste_def('PCA_ARMS=' + ' '.join(arms), *_TARGETS0, *variants)}"
                    f"\n      cd {real} && _L={real}/sl_comparison/slurm/logs\n"
                    in tail), tail
            for a in left:
                t = _tag(a)
                own_t = "".join(" " + w for w in (targets or {}).get(a, ()))
                for v, dep, log, script in (
                        ("c", ids["shared"] + ":" + ids["elisl"],
                         "final_compare", "run_compare.sh"),
                        ("e", ids["shared"], "external_transfer",
                         "run_external_transfer.sh"),
                        ("t", f"$_c_{a}:$_e_{a}", "paper_tables",
                         "run_paper_tables.sh")):
                    want = (f"\n      _{v}_{a}=$(_paste_env PCA_ARM={a}{own_t} "
                            f"sbatch --parsable --dependency=afterany:{dep} "
                            f"--export=ALL --output=$_L/"
                            f"{log}_{t}%j.out --error=$_L/{log}_{t}%j.err "
                            f"sl_comparison/slurm/{script})\n")
                    assert want in tail, (want, tail)
            for a in ran:      # this run's own jobs rebuild its arms' tables
                assert f"_c_{a}=" not in out, (a, tail)
            deps = ":".join([f"$_t_{a}" for a in left]
                            + [ids[j] for j in own])
            want = (f"\n      _paste_env PCA_ARM={arms[0]} sbatch "
                    f"--dependency=afterany:{deps} "
                    "--export=ALL --output=$_L/paper_tables_pca_arms_%j.out "
                    "--error=$_L/paper_tables_pca_arms_%j.err sl_comparison/"
                    "slurm/run_paper_tables.sh --pca_arms_table\n")
            assert want in tail and ("when another submission than\n    "
                                     f"arm {arms[0]}'s made its selection"
                                     ) in tail, (want, tail)
            return tail

        # two arms: after their own table across the arms; one arm: after
        # its paper tables; without paper tables: after each arm's compare
        # and external transfer
        for ran, flags, own in (
                (["none", "joint"], (), ("pca_arms",)),
                (["joint"], (), ("tables/joint",)),
                (["none", "joint"], ("--skip-paper-tables",),
                 ("compare/none", "ext/none", "compare/joint", "ext/joint"))):
            _seed_arms(root)
            out, jobs = run("--skip-generate", *flags,
                            PCA_ARMS=" ".join(ran))
            left = [a for a in _ARMS if a not in ran]
            assert named(out) == (left, []), (ran, out[-3000:])
            tail = wired(out, jobs, ran, left, own)
            assert ("\n    models once more): PCA_MODALITY_VARIANCE=0.8 "
                    "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM= USE_CELL_LINES=1 "
                    'SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled '
                    f'bilinear:per_head bilinear:pooled" PCA_ARMS="{every}" bash '
                    "./run_benchmarking_siamese_vs_others.sh --skip-siamese\n"
                    ) in tail and "--skip-generate --skip-shared\n" \
                not in tail, tail
        # the arms of PCA_ARMS keep their state, and are still never named:
        # --skip-siamese leaves their siamese roots alone, and a variant
        # subset the other variants' markers
        for flags, env in ((("--skip-siamese",), {}),
                           (("--skip-generate",),
                            {"SIAMESE_VARIANTS_LIST": "pd:per_head"})):
            _seed_arms(root)
            out, jobs = run(*flags, PCA_ARMS="none joint", **env)
            for a in ("none", "joint"):
                assert os.path.exists(os.path.join(
                    sia, _root_of(a), "var_pd_pooled__latest_combo_cv3.txt")), \
                    (flags, a)
            assert named(out) == (["per_modality", "both"], []), \
                (flags, out[-3000:])
            wired(out, jobs, ["none", "joint"], ["per_modality", "both"],
                  ("pca_arms",), variants=(
                      "SIAMESE_VARIANTS_LIST=" + env["SIAMESE_VARIANTS_LIST"],
                      "USE_CELL_LINES=1", "CVS=") if env else _VARIANTS0)
        # --skip-shared: nothing removed, nothing named
        _seed_arms(root)
        out, _ = run("--skip-generate", "--skip-shared", PCA_ARMS="joint")
        assert named(out) == ([], []) and "_c_" not in out, out[-2000:]
        # by state: an arm with nothing to tabulate is not named, and the
        # table across the arms leaves it out
        _seed_arms(root, arms=("none", "joint"))
        out, jobs = run("--skip-generate", PCA_ARMS="joint")
        assert named(out) == (["none"], []), out[-3000:]
        wired(out, jobs, ["joint"], ["none"], ("tables/joint",))
        # a stop part-way: named in the error report; the resubmission finds
        # nothing left to remove and names them again
        _seed_arms(root)
        out, jobs = run("--skip-generate", _rc=1, PCA_ARMS="joint",
                        SB_FAIL_ON="run_paper_tables.sh")
        assert "submission stopped part-way" in out and error_note in out, \
            out[-2500:]
        out, jobs = run("--skip-generate", PCA_ARMS="joint")
        assert "built on the shared rows (comparison" not in out \
            and named(out) == (["none", "per_modality", "both"], []), \
            out[-3000:]
        wired(out, jobs, ["joint"], ["none", "per_modality", "both"],
              ("tables/joint",))
        # ... and when the first sbatch is refused (nothing queued yet)
        _seed_arms(root)
        out, jobs = run("--skip-generate", _rc=1, PCA_ARMS="joint",
                        SB_FAIL_ON="slurm/reset_env.sh")
        assert "part-way" not in out and error_note in out, out[-2500:]
        # a selection without a combo marker: retrained, not tabulated (a
        # rebuild would have no SLeuth rows), and the others as above
        _seed_arms(root)
        for n in ("latest_combo_cv1.txt",
                  "var_pd_pooled__latest_combo_cv3.txt"):
            os.remove(os.path.join(sia, _root_of("both"), n))
        out, jobs = run("--skip-generate", PCA_ARMS="joint")
        assert named(out) == (["none", "per_modality"], ["both"]) \
            and "_c_both" not in out \
            and "a selection without a combo marker" in out, out[-3000:]
        tail = wired(out, jobs, ["joint"], ["none", "per_modality"],
                     ("tables/joint",))
        # (both with per_modality, whose kept selection ranks the Step-1 runs
        # of their pair: retrained alone, both owned runs of its own,
        # 2026-10-24)
        assert retrain(tail, ["per_modality", "both"]) \
            and retrain(out.split("All jobs submitted")[0],
                        ["per_modality", "both"]) \
            and "so both arms of it retrain" in tail, tail
        # a regeneration removed every left-out arm's markers: all retrained
        # (their combos were trained on the files it replaces), no tables
        # rebuild offered, no "SLeuth rows stay"
        _seed_arms(root)
        out, jobs = run("--force-generate", PCA_ARMS="joint")
        head, tail = out.split("All jobs submitted")
        assert named(out) == ([], ["none", "per_modality", "both"]) \
            and "trained on the embeddings this run replaces" in out \
            and "SLeuth rows and siamese outputs stay" not in out \
            and "_c_" not in out and "--pca_arms_table\n" not in tail \
            and retrain(head, _ARMS) and retrain(tail, _ARMS), out[-3000:]
        # ... stopped part-way: the arms to retrain are named in the report
        _seed_arms(root)
        out, jobs = run("--force-generate", _rc=1, PCA_ARMS="joint",
                        SB_FAIL_ON="run_paper_tables.sh")
        assert "submission stopped part-way" in out and error_note in out, \
            out[-2500:]


def test_root_rebuild_block_keeps_the_pasting_shells_token_out():
    # The final listing's ready-to-paste rebuild of the left-out arms runs as
    # printed (bash, and zsh where installed) and is wired as printed: each
    # arm's paper tables after its compare and external transfer (their ids
    # captured), the table across every arm after those and this run's last
    # table job. Pasted in a shell that exports RUN_TOKEN and CODE_MANIFEST
    # (sl_comparison/README.md's hand recipes leave both) and compare pins
    # (SIAMESE_RESULTS_<...>, SIAMESE_BASELINE_DISCOVER: the hand-compare
    # recipe's), no job inherits any of them (_paste_env, the block's first
    # line), so each arm's tables check its dirs against its own selection's
    # token, as a standalone run_paper_tables.sh does, and each arm's compare
    # grades its own combos; and the shell keeps them all. The root sees two
    # of the pins (and ignores them for its own jobs, with a NOTE); the
    # third, exported only in the pasting shell, is kept out all the same.
    # Every pasted job gets the arms of the table across the arms and this
    # run's PCA targets, whatever the pasting shell exports.
    shell_env = {"RUN_TOKEN": "20261001T090000-777",
                 "CODE_MANIFEST": "/old/sl_comparison/slurm/logs/"
                                  "code_20261001T090000-777_x.sha256"}
    pins = {"SIAMESE_RESULTS_cv1":
            "siamese_sl/results/best_cat_go_kg_cv1/results.json",
            "SIAMESE_BASELINE_DISCOVER": "all"}
    late = {"SIAMESE_RESULTS_bilinear_pooled_cv3": "elsewhere/results.json",
            "PCA_JOINT_DIM": "999", "PCA_ARMS": "joint"}
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        _seed_arms(root)
        sb = os.path.join(tmp, "bin", "sbatch")
        with open(sb) as f:
            text = f.read()
        assert text.endswith("echo $n\n"), text[-60:]
        with open(sb, "w") as f:        # what each job would inherit
            f.write(text[:-len("echo $n\n")]
                    + 'echo "$n|' + "|".join(
                        "${%s-<unset>}" % p for p in sorted(pins) + sorted(
                            k for k in late if k.startswith("SIAMESE")))
                    + '" >> "$SB_LOG.pins"\necho $n\n')
        rc, out, jobs = _root_run(tmp, "--skip-generate", PCA_ARMS="joint",
                                  **shell_env, **pins)
        assert rc == 0 and "NOTE: replaced RUN_TOKEN=20261001T090000-777" \
            in out and ("NOTE: ignoring SIAMESE_RESULTS_cv1 "
                        "SIAMESE_BASELINE_DISCOVER from the") in out, \
            out[-2500:]
        with open(os.path.join(tmp, "sb.log.pins")) as f:
            own = [ln.rstrip("\n").split("|")[1:] for ln in f]
        assert own and all(p == ["<unset>"] * 3 for p in own), own
        tail = out.split("All jobs submitted")[1].splitlines()
        i = next(k for k, ln in enumerate(tail)
                 if ln.startswith("      _paste_env() {"))
        j = next(k for k, ln in enumerate(tail)
                 if ln.endswith(" --pca_arms_table"))
        assert j - i == 11 and tail[i + 1].startswith("      cd ") and all(
            ln.startswith("      ") for ln in tail[i:j + 1]), tail[i:j + 1]
        block = "\n".join(ln[6:] for ln in tail[i:j + 1]) + \
            '\necho "kept: $RUN_TOKEN $CODE_MANIFEST $SIAMESE_RESULTS_cv1' \
            ' $SIAMESE_BASELINE_DISCOVER $SIAMESE_RESULTS_bilinear_pooled_cv3"\n'
        kept = " ".join([shell_env["RUN_TOKEN"], shell_env["CODE_MANIFEST"],
                         pins["SIAMESE_RESULTS_cv1"], "all",
                         late["SIAMESE_RESULTS_bilinear_pooled_cv3"]])
        ids = jobs["_ids"]
        log, ctr = os.path.join(tmp, "paste.log"), os.path.join(tmp, "ctr")
        for sh in ["bash"] + [z for z in (shutil.which("zsh"),) if z]:
            for f in (log, log + ".env", log + ".pins"):
                if os.path.exists(f):
                    os.remove(f)
            with open(ctr, "w") as f:
                f.write("2000\n")
            r = subprocess.run(
                [sh, "-s"], input=block, cwd=tmp, capture_output=True,
                text=True, timeout=60,
                env=dict(shell_env, **pins, **late, HOME=tmp, SB_LOG=log,
                         SB_CTR=ctr, PATH=os.path.join(tmp, "bin")
                         + os.pathsep + os.environ["PATH"]))
            assert r.returncode == 0 and f"kept: {kept}\n" in r.stdout, \
                (sh, r.stdout, r.stderr)
            with open(log) as f:
                calls = [ln.rstrip("\n").split("|", 2) for ln in f]
            with open(log + ".env") as f:
                envs = [ln.rstrip("\n").split("|") for ln in f]
            with open(log + ".pins") as f:
                pinned = [ln.rstrip("\n").split("|")[1:] for ln in f]
            assert len(calls) == len(envs) == len(pinned) == 10, (sh, calls)
            # no job inherits the shell's token, manifest or pins
            assert [c[1] for c in calls] == [""] * 10 \
                and [e[4] for e in envs] == [""] * 10 \
                and pinned == [["<unset>"] * 3] * 10, (sh, calls, envs, pinned)
            # ... and each gets the arms of the table across the arms (this
            # run's and the left-out ones it rebuilds) and this run's PCA
            # targets, not the shell's stray ones
            assert {tuple(e[6:]) for e in envs} == {
                ("none joint per_modality both", "0.8", "0.8", "")}, envs
            # every job gets its arm from the command alone (the shell
            # exports none), with a plain --export=ALL
            for c in calls:
                _plain_export(c[2])
            for k, a in enumerate(("none", "per_modality", "both")):
                c, e, t = calls[3 * k:3 * k + 3]
                for job, env, dep, script in (
                        (c, envs[3 * k], ids["shared"] + ":" + ids["elisl"],
                         "run_compare.sh"),
                        (e, envs[3 * k + 1], ids["shared"],
                         "run_external_transfer.sh"),
                        (t, envs[3 * k + 2], f"{c[0]}:{e[0]}",
                         "run_paper_tables.sh")):
                    assert f"--dependency=afterany:{dep} " in job[2] \
                        and env[0] == job[0] and env[5] == a \
                        and job[2].endswith("sl_comparison/slurm/" + script), \
                        (sh, a, job, env)
            x = calls[9]
            assert (f"--dependency=afterany:{calls[2][0]}:{calls[5][0]}:"
                    f"{calls[8][0]}:{ids['tables/joint']} ") in x[2] \
                and x[2].endswith("run_paper_tables.sh --pca_arms_table") \
                and envs[9][5] == "none" \
                and envs[9][6] == "none joint per_modality both", (sh, x)


def test_a_left_out_partner_hears_its_ranked_runs_are_rebuilt():
    # A pair's single-embedding runs (Step 1) live in its owner's root (the
    # first of the pair this run lists: none of none / joint, per_modality
    # of per_modality / both). A run of the owner without its partner
    # rebuilds them while the partner, left out, keeps the selection that
    # ranked them (_selection_meta.step1_root): its tab:solo would flag this
    # run's runs as another run's. Both orchestrators say so, naming the
    # partner, before anything changes, and leave its outputs alone; never
    # when the partner runs too, when its selection ranked other runs (its
    # own root, after a run of it alone), or when no Step 1 runs.
    def seed(base, sels):
        """sels: {arm: its selection's step1_root}; the siamese results
        root emptied first."""
        res = os.path.join(base, "siamese_sl", "results")
        shutil.rmtree(res, ignore_errors=True)
        os.makedirs(res)
        for a, s1 in sels.items():
            r = os.path.join(base, "siamese_sl", _root_of(a))
            os.makedirs(r, exist_ok=True)
            with open(os.path.join(r, "best_per_category.json"), "w") as f:
                json.dump({"cv1": {}, "_selection_meta": {
                    "run_token": "T1", "step1_root": s1}}, f)
            with open(os.path.join(r, "latest_combo_cv1.txt"), "w") as f:
                f.write(f"{_root_of(a)}/x_cv1\n")

    def kept(base, arm):
        with open(os.path.join(base, "siamese_sl", _root_of(arm),
                               "best_per_category.json")) as f:
            return json.load(f)["_selection_meta"]["step1_root"]

    def warned(out):
        return re.findall(r"WARNING: PCA arm (\w+), which PCA_ARMS leaves "
                          r"out, ranked the single-embedding runs in", out)

    sp = ("WARNING: PCA arm joint, which PCA_ARMS leaves out, ranked the "
          "single-embedding runs in\n  results/ (results/pca_joint/"
          "best_per_category.json, _selection_meta.step1_root), which this\n"
          "  submission rebuilds as PCA arm none's Step 1: until joint is "
          "rerun (PCA_ARMS listing\n  it), its tab:solo will be another "
          "run's.\n")
    flat = ("--skip-reset-env", "--skip-generate")
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        seed(tmp, {"none": "results", "joint": "results"})
        jobs, out = _submit(tmp, *flat, PCA_ARMS="none")
        assert jobs and sp in out and warned(out) == ["joint"] \
            and out.index(sp) < out.index("=" * 40) \
            and kept(tmp, "joint") == "results", out[-3000:]
        # (a step1_root written with its trailing slash is the same root)
        seed(tmp, {"per_modality": "results/pca_per_modality",
                   "both": "results/pca_per_modality/"})
        jobs, out = _submit(tmp, *flat, PCA_ARMS="per_modality")
        assert jobs and warned(out) == ["both"] \
            and ("  results/pca_per_modality/ (results/pca_both/"
                 "best_per_category.json, _selection_meta.step1_root), which "
                 "this\n  submission rebuilds as PCA arm per_modality's Step "
                 "1: until both is rerun") in out, out[-3000:]
        for sels, args, arms in (
                ({"joint": "results"}, flat, "none joint"),
                ({"joint": "results/pca_joint"}, flat, "none"),
                ({"none": "results", "joint": "results"}, flat, "joint"),
                ({"both": "results/pca_per_modality"}, flat, "none"),
                ({"none": "results", "joint": "results"}, ("--ablation-only",),
                 "none")):
            seed(tmp, sels)
            jobs, out = _submit(tmp, *args, PCA_ARMS=arms)
            assert jobs and not warned(out), (sels, args, arms, out[-3000:])
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        seed(root, {"none": "results", "joint": "results"})
        rc, out, jobs = _root_run(tmp, "--skip-generate", PCA_ARMS="none")
        assert rc == 0 and "bestcat/none" in jobs, out[-2500:]
        assert warned(out) == ["joint"] and (
            "  siamese_sl/results/ (siamese_sl/results/pca_joint/best_per_"
            "category.json, _selection_meta.step1_root), which\n  this run "
            "removes and rebuilds as PCA arm none's Step 1: until joint is "
            "rerun\n  (PCA_ARMS listing it), its tab:solo will be another "
            "run's.\n") in out, out[-3000:]
        assert out.index("WARNING: PCA arm joint") \
            < out.index("--- previous-run outputs ---") \
            and kept(root, "joint") == "results", out[-3000:]
        seed(root, {"both": "results/pca_per_modality"})
        rc, out, _ = _root_run(tmp, "--skip-generate", PCA_ARMS="per_modality")
        assert rc == 0 and warned(out) == ["both"] \
            and "as PCA arm per_modality's Step 1: until both" in out, \
            out[-3000:]
        for sels, args, arms in (
                ({"joint": "results"}, ("--skip-generate",), "none joint"),
                ({"joint": "results/pca_joint"}, ("--skip-generate",), "none"),
                ({"none": "results", "joint": "results"}, ("--skip-siamese",),
                 "none")):
            seed(root, sels)
            rc, out, _ = _root_run(tmp, *args, PCA_ARMS=arms)
            assert rc == 0 and not warned(out), (sels, args, arms, out[-3000:])


def test_skip_siamese_rerun_names_code_changed_since_the_reused_run():
    # --skip-siamese keeps the reused run's RUN_TOKEN but writes a new code
    # manifest, the only one paper_tables compares with: code uploaded between
    # the two submissions is in no list. The submission names it (a WARNING
    # only; every job still gets this submission's manifest).
    token, warn = "20260101T000000-4242", "code file(s) changed since run"
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        logs = os.path.join(root, "sl_comparison", "slurm", "logs")
        res = os.path.join(root, "siamese_sl", "results")
        os.makedirs(res, exist_ok=True)

        def run(*flags):
            rc, out, jobs = _root_run(tmp, *flags)
            assert rc == 0, out[-2500:]
            return out, jobs

        # nothing to compare with: no selection; the default run; --skip-shared
        for flags in (("--skip-siamese",), ("--skip-generate",),
                      ("--skip-shared", "--skip-generate")):
            out = run(*flags)[0]
            assert warn not in out, flags
            # only a reuse says that it has nothing to compare with
            assert ("no earlier code manifest" in out) == \
                (flags == ("--skip-siamese",)), flags
        # the reused run: its selection's token, its submission's manifest
        with open(os.path.join(res, "best_per_category.json"), "w") as f:
            json.dump({"_selection_meta": {"run_token": token}}, f)
        # ...whose submission left no manifest (submit_pipeline.sh writes
        # none): nothing to compare with, and the submission says so
        out = run("--skip-siamese")[0]
        assert warn not in out and \
            f"NOTE: no earlier code manifest carries run token {token}" in out
        m = subprocess.run([sys.executable, "sl_comparison/code_manifest.py"],
                           cwd=root, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        assert m.returncode == 0 and m.stdout, m.stderr
        old = os.path.join(logs, f"code_{token}_20260101T000003.sha256")
        with open(old, "w") as f:
            f.write(m.stdout)
        out = run("--skip-siamese")[0]                  # unchanged tree
        assert warn not in out and "no earlier code manifest" not in out
        with open(os.path.join(root, "siamese_sl", "siamese_esm.py"),
                  "a") as f:
            f.write("\n# uploaded between the two submissions\n")
        for _ in range(2):      # each rerun adds a manifest: the OLDEST counts
            out, jobs = run("--skip-siamese")
            assert f"WARNING: 1 {warn} {token}" in out \
                and "siamese_sl/siamese_esm.py (changed)" in out \
                and os.path.basename(old) in out, out[-2500:]
            # (by name: the script resolves its own path, tempfile may not)
            given = {os.path.basename(v[3]) for v in jobs["_env"].values()}
            assert len(given) == 1 and os.path.basename(old) not in given \
                and all(g.startswith(f"code_{token}_") for g in given), given
        assert len([n for n in os.listdir(logs)
                    if n.startswith(f"code_{token}_")]) >= 3
        assert "the first 20" not in out
        # more than 20 changed files: the list is cut, and says so
        files = [ln.split(None, 1)[1] for ln in m.stdout.splitlines()]
        more = [p for p in files if p != "siamese_sl/siamese_esm.py"][:24]
        for p in more:
            with open(os.path.join(root, p), "a") as f:
                f.write("\n# changed too\n")
        out, _ = run("--skip-siamese")
        assert f"WARNING: 25 {warn} {token}" in out \
            and "... (the first 20 of 25; all of them:" in out \
            and out.count(" (changed)") == 20, out[-3000:]
        # a fresh run beside those manifests has another token: silent
        assert warn not in run("--skip-generate")[0]


def test_skip_siamese_joins_each_arms_reused_token():
    # --skip-siamese reuses every arm's siamese tree under that arm's own
    # token: its selection's, else the one token of the single-embedding
    # runs in its OWNER's root (an arm alone owns its own). Arms reused from
    # one submission: that token for the whole DAG, the RUN_TOKEN every job
    # inherits. From different ones: each arm's jobs get theirs in sbatch's
    # environment (PCA_ARM=<arm> RUN_TOKEN=<token> sbatch ... --export=ALL),
    # RUN_TOKEN (the shared jobs', the manifests' names, the table across
    # the arms') is the first arm's, with a WARNING; the code each reused
    # run was submitted with is looked up per token.
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        sia = os.path.join(root, "siamese_sl")

        def seed(sel=None, step1=None):
            res = os.path.join(sia, "results")
            shutil.rmtree(res, ignore_errors=True)
            os.makedirs(res)
            for a, tok in (sel or {}).items():
                r = os.path.join(sia, _root_of(a))
                os.makedirs(r, exist_ok=True)
                with open(os.path.join(r, "best_per_category.json"), "w") as f:
                    json.dump({"_selection_meta": {"run_token": tok}}, f)
            for o, toks in (step1 or {}).items():
                for k, tok in enumerate(toks):
                    d = os.path.join(sia, _root_of(o), f"pcavar0.8_t{k}_cv1")
                    os.makedirs(d)
                    with open(os.path.join(d, "run_token.txt"), "w") as f:
                        f.write(tok + "\n")

        # none's selection; per_modality's Step-1 runs, in its own root
        seed({"none": "T1"}, {"per_modality": ["T2", "T2"]})
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none per_modality")
        assert rc == 0, out[-2500:]
        toks = jobs["_tokens"]
        for job in ("compare", "ext", "tables"):
            assert (toks[f"{job}/none"], toks[f"{job}/per_modality"]) == \
                ("T1", "T2"), (job, toks)
        assert toks["shared"] == toks["folds"] == toks["pca_arms"] == "T1" \
            and "\n    PCA arm per_modality: RUN_TOKEN=T2\n" in out, \
            (toks, out[-2500:])
        assert "WARNING: the PCA arms reused here were run by submissions " \
            "of different tokens:\n    none: T1\n    per_modality: T2\n" \
            in out, out[-2500:]
        for tok, arm in (("T1", "none"), ("T2", "per_modality")):
            assert (f"NOTE: no earlier code manifest carries run token {tok} "
                    f"(PCA arm(s) {arm}) under") in out, out[-2500:]
        # (the manifests are named after the first arm's)
        assert {os.path.basename(v[3]).split("_")[1]
                for v in jobs["_env"].values()} == {"T1"}, jobs["_env"]
        # one submission's: its token for every job
        seed({"none": "T1", "per_modality": "T1"})
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none per_modality")
        assert rc == 0 and set(jobs["_tokens"].values()) == {"T1"} \
            and "different tokens" not in out \
            and "    PCA arm none: RUN_TOKEN=" not in out \
            and "run token T1 (the reused selection's" in out, out[-2500:]
        # an arm run alone owns its pair's single-embedding runs (another
        # root's are no run of it)
        seed({}, {"none": ["T3"], "joint": ["T4"]})
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PCA_ARMS="joint")
        assert rc == 0 and set(jobs["_tokens"].values()) == {"T4"}, \
            (jobs.get("_tokens"), out[-2500:])
        seed({}, {"none": ["T3"], "joint": ["T4"]})
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none joint")
        assert rc == 0 and set(jobs["_tokens"].values()) == {"T3"}, \
            (jobs.get("_tokens"), out[-2500:])
        # (no run of several tokens: no NOTE to keep one)
        assert "To keep one of those tokens" not in out, out[-2500:]
        # Step-1 runs of two tokens and no selection: a fresh token; the
        # WARNING names them, and a run that queues its jobs ends with a
        # NOTE: their scancel line, then the submission that keeps one token
        # (the settings taken from the calling shell, a one-shot target once,
        # the arms and targets in full and every flag of this run; 2026-10-21:
        # it printed "RUN_TOKEN=<token above> bash $0 --skip-siamese ..."),
        # with no ALLOW_LIVE_JOBS (2026-10-22: pasted while this run's jobs
        # were live, it passed the guard that keeps its step 0 and reset_env
        # off them; and printed in the WARNING, it named a scancel line a
        # refused run never printed). With the token filled in, run from a
        # fresh shell: refused while one of those jobs is live; once none is,
        # every job gets the token, the one-shot seed and target, and no
        # paper tables are queued (its --skip-paper-tables kept)
        seed({}, {"none": ["T5", "T6", "T6"]})
        rc, out, jobs = _root_run(tmp, "--skip-siamese", "--skip-paper-tables",
                                  SEED="7", ALLOW_LIVE_JOBS="1",
                                  PCA_JOINT_DIM="150")
        line = ("RUN_TOKEN=<token> SEED=7 PCA_MODALITY_VARIANCE=0.8 "
                "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=150 USE_CELL_LINES=1 "
                f'SIAMESE_VARIANTS_LIST="{"pd:per_head pd:pooled bilinear:per_head bilinear:pooled"}" '
                'PCA_ARMS="none" bash '
                "./run_benchmarking_siamese_vs_others.sh --skip-siamese "
                "--skip-paper-tables")
        assert rc == 0 and "carry 2 different run tokens" in out \
            and ("(The submission that keeps one of them instead is printed "
                 "once this run\n         is queued.)") in out \
            and ("NOTE: PCA arm(s) none reuse runs of several tokens (the "
                 "WARNING above) and\n    got this submission's fresh one. To "
                 "keep one of those tokens instead, cancel this\n    "
                 "submission's jobs:\n      scancel "
                 + " ".join(jobs["_ids"].values()) + "\n    and, once squeue "
                 "lists none of them, submit it again with that token in place "
                 "of\n    <token> (one RUN_TOKEN is kept only for the arms "
                 "whose runs carry it; the\n    WARNING's other arms get a "
                 "fresh one again):\n"
                 f"      {line}\n") in out \
            and "tables/none" not in jobs, out[-2500:]
        assert out.split("All jobs submitted")[0].count("RUN_TOKEN=<token>") \
            == 0 and [ln for ln in out.splitlines() if "RUN_TOKEN=<token>" in ln
                      and "ALLOW_LIVE_JOBS" in ln] == [], out[-2500:]
        queue = os.path.join(tmp, "queue")
        with open(queue, "w") as f:
            f.write(jobs["_ids"]["reset"] + " reset_env\n")
        seed({}, {"none": ["T5", "T6", "T6"]})
        rc, o2, j2 = _root_run(tmp, _line=line.replace("<token>", "T6"),
                               SQUEUE_OUT=queue)
        assert rc == 1 and not j2 and "ERROR: jobs of an earlier submission" \
            in o2, o2[-2500:]
        rc, o2, jobs = _root_run(tmp, _line=line.replace("<token>", "T6"))
        assert rc == 0 and set(jobs["_tokens"].values()) == {"T6"} \
            and "| seed 7 |" in o2 and "tables/none" not in jobs \
            and "compare/none" in jobs and {
                tuple(v[5:]) for v in jobs["_env"].values()} == {
                ("none", "0.8", "0.8", "150")} \
            and "To keep one of those tokens" not in o2, \
            (jobs.get("_tokens"), o2[-2500:])
        # ... and from a shell exporting USE_CELL_LINES=0, another variant
        # list and PCA_ARMS: the same arm, variants and USE_CELL_LINES
        # (2026-10-22: the line rebuilt pd:per_head's rows alone, after
        # moving the arm's four SLeuth rows aside)
        seed({}, {"none": ["T5", "T6", "T6"]})
        rc, o2, jobs = _root_run(tmp, _line=line.replace("<token>", "T6"),
                                 USE_CELL_LINES="0", PCA_ARMS="joint",
                                 SIAMESE_VARIANTS_LIST="bilinear:pooled")
        assert rc == 0 and "compare/none" in jobs \
            and "compare/joint" not in jobs and set(jobs["_vars"].values()) \
            == {("pd:per_head pd:pooled bilinear:per_head bilinear:pooled",
                 "1")}, (o2[-2500:], jobs.get("_vars"))
        # (the NOTE's line resubmits every arm of this run, the cancelled
        # jobs of an arm with its own selection included: arm joint here,
        # under its selection's token S9)
        seed({"joint": "S9"}, {"none": ["T5", "T6"]})
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PCA_ARMS="none joint")
        line3 = next(ln.strip() for ln in out.splitlines()
                     if ln.strip().startswith("RUN_TOKEN=<token> "))
        assert rc == 0 and line3.endswith(
            'PCA_ARMS="none joint" bash ./run_benchmarking_siamese_vs_others.sh '
            "--skip-siamese"), (line3, out[-2500:])
        seed({"joint": "S9"}, {"none": ["T5", "T6"]})
        rc, o2, jobs = _root_run(tmp, _line=line3.replace("<token>", "T6"))
        assert rc == 0 and (jobs["_tokens"]["compare/none"],
                            jobs["_tokens"]["compare/joint"]) == ("T6", "S9"), \
            (jobs.get("_tokens"), o2[-2500:])
        # Step-1 dirs that carry no run_token.txt: the WARNING, and nothing
        # to keep
        seed({})
        for k in range(2):
            os.makedirs(os.path.join(sia, "results", f"pcavar0.8_t{k}_cv1"))
        rc, out, jobs = _root_run(tmp, "--skip-siamese")
        assert rc == 0 and "carry no run_token.txt." in out \
            and "To keep one of those tokens" not in out \
            and "RUN_TOKEN=<token>" not in out \
            and "keeps one of them instead" not in out, out[-2500:]
        # two arms of other owners, each of several tokens: the NOTE names
        # both, and one RUN_TOKEN keeps the arm whose runs carry it only
        seed({}, {"none": ["T5", "T6"], "per_modality": ["T7", "T8"]})
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none per_modality")
        assert rc == 0 and ("NOTE: PCA arm(s) none per_modality reuse runs of "
                            "several tokens") in out, out[-2500:]
        line2 = next(ln.strip() for ln in out.splitlines()
                     if ln.strip().startswith("RUN_TOKEN=<token> "))
        seed({}, {"none": ["T5", "T6"], "per_modality": ["T7", "T8"]})
        rc, o2, jobs = _root_run(tmp, _line=line2.replace("<token>", "T6"))
        assert rc == 0 and jobs["_tokens"]["compare/none"] == "T6" \
            and jobs["_tokens"]["compare/per_modality"] not in ("T6", "T7",
                                                                "T8") \
            and ("NOTE: PCA arm(s) per_modality reuse runs of several "
                 "tokens") in o2, (jobs.get("_tokens"), o2[-2500:])
        # refused by the PCA check that follows (joint's combos at 150
        # components): the resubmit line, and the one that keeps a token,
        # with that target (2026-10-22: the WARNING's line, with this
        # shell's targets, was refused again, and the resubmit line ended in
        # a fresh token); an exported RUN_TOKEN naming one of the runs' is
        # kept, and carried on the resubmit line, and no line to keep one is
        # printed; one naming none of them is not
        seed({}, {"none": ["T5", "T6", "T6"]})
        _seed_combos(sia, "joint", {("", cv): {"post_pca": 150,
                                               "post_pca_dim": 150}
                                    for cv in ("cv1", "cv2", "cv3")})
        rc, out, jobs = _root_run(tmp, "--skip-siamese", SEED="7",
                                  PCA_ARMS="none joint")
        fix = (r"PCA_ARMS=none\ joint SEED=7 PCA_MODALITY_VARIANCE=0.8 "
               "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=150 USE_CELL_LINES=1 "
               'SIAMESE_VARIANTS_LIST="pd:per_head pd:pooled bilinear:per_head '
               'bilinear:pooled" bash '
               "./run_benchmarking_siamese_vs_others.sh --skip-siamese")
        keep = ("\n  or, to keep one of the run tokens of PCA arm(s) none (the "
                "WARNING above;\n  one RUN_TOKEN is kept only for the arms whose "
                "runs carry it), with that token in\n  place of <token>:\n    "
                "RUN_TOKEN=<token> " + fix + "\n")
        assert rc == 1 and not jobs and ("resubmit with it,\n    " + fix
                                         + keep) in out \
            and "NOTE: PCA arm(s) none reuse runs" not in out, out[-2500:]
        rc, o2, jobs = _root_run(tmp, _line="RUN_TOKEN=T6 " + fix)
        assert rc == 0 and jobs["_tokens"]["compare/none"] == "T6" \
            and jobs["_tokens"]["compare/joint"] == "T1", \
            (jobs.get("_tokens"), o2[-2500:])
        # (the same from a shell exporting USE_CELL_LINES=0, another list
        # and PCA_ARMS: both arms, all four variants, the cell-line model)
        rc, o2, jobs = _root_run(tmp, _line="RUN_TOKEN=T6 " + fix,
                                 USE_CELL_LINES="0", PCA_ARMS="joint",
                                 SIAMESE_VARIANTS_LIST="bilinear:pooled")
        assert rc == 0 and {"compare/none", "compare/joint"} <= set(jobs) \
            and set(jobs["_vars"].values()) == {
                ("pd:per_head pd:pooled bilinear:per_head bilinear:pooled",
                 "1")}, (o2[-2500:], jobs.get("_vars"))
        for tok, kept in (("T6", True), ("TX", False)):
            seed({}, {"none": ["T5", "T6", "T6"]})
            _seed_combos(sia, "joint", {("", cv): {"post_pca": 150,
                                                   "post_pca_dim": 150}
                                        for cv in ("cv1", "cv2", "cv3")})
            rc, out, jobs = _root_run(tmp, "--skip-siamese", SEED="7",
                                      PCA_ARMS="none joint", RUN_TOKEN=tok)
            want = ("resubmit with it,\n    RUN_TOKEN=T6 " + fix
                    + "\n  Nothing was changed") if kept else \
                ("resubmit with it,\n    " + fix + keep)
            assert rc == 1 and want in out and (
                "carry 2 different run tokens" in out) != kept, \
                (tok, out[-2500:])


def test_skip_siamese_checks_the_reused_arms_pca():
    # --skip-siamese labels every reused arm, and paper_tables checks its
    # combos, with this shell's PCA targets: combos an earlier submission
    # trained at others would be mislabelled and their tables PAPER
    # INCOMPLETE. Each marker-named combo's config.json is compared before
    # anything changes: one set of targets that trains every reused arm's
    # combos is refused with a command that carries it (and the shell's
    # other settings), which then runs; arms that disagree among themselves
    # (no one set) only get a WARNING naming the arms whose tables will be
    # INCOMPLETE; 0.80 is 0.8. A run that leaves such an arm out prints its
    # rebuild with the arm's own targets on its lines.
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        sia = os.path.join(root, "siamese_sl")

        def seed(cfgs):
            res = os.path.join(sia, "results")
            if os.path.isdir(res):
                shutil.rmtree(res)
            os.makedirs(res)
            for arm, cfg in cfgs.items():
                _seed_combos(sia, arm, {("", cv): cfg
                                        for cv in ("cv1", "cv2", "cv3")})

        def state():
            out = {}
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d != "logs"]
                for f in fn:
                    q = os.path.join(dp, f)
                    out[os.path.relpath(q, root)] = os.stat(q).st_mtime_ns
            return out

        c150 = {"post_pca": 150, "post_pca_dim": 150}
        b150 = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8], "post_pca": 150,
                "post_pca_dim": 150}
        b08 = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8], "post_pca": 0.8}
        # joint and both trained at 150 components: one export fits both
        seed({"none": {}, "joint": c150, "both": b150})
        before = state()
        rc, out, jobs = _root_run(tmp, "--skip-siamese", SEED="7",
                                  PCA_ARMS="none joint both")
        assert rc == 1 and not jobs and state() == before, out[-2500:]
        # (a CVS the root refuses is refused first: the resubmit line would
        # carry it, 2026-10-22)
        rc, o2, j2 = _root_run(tmp, "--skip-siamese", SEED="7", CVS="cv1,cv2",
                               PCA_ARMS="none joint both")
        assert rc == 1 and not j2 and "ERROR: CVS='cv1,cv2'" in o2 \
            and "--skip-siamese reuses" not in o2 and state() == before, \
            o2[-2500:]
        assert ("ERROR: --skip-siamese reuses PCA arm(s) joint both, whose "
                "combos were trained at\n  other PCA targets than this shell's "
                "(PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 "
                "PCA_JOINT_DIM=):\n    joint: siamese_sl/results/pca_joint/"
                "best_cat_x_cv1  (pd:per_head: PCA on the concatenation of all "
                "modalities (150 components))\n") in out \
            and "Nothing was changed or submitted." in out, out[-2500:]
        cmd = [ln.strip() for ln in out.splitlines()
               if ln.endswith(" bash ./run_benchmarking_siamese_vs_others.sh "
                              "--skip-siamese")]
        assert cmd == [r"PCA_ARMS=none\ joint\ both SEED=7 PCA_MODALITY_"
                       "VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=150 "
                       'USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="pd:per_head '
                       'pd:pooled bilinear:per_head bilinear:pooled" '
                       "bash ./run_benchmarking_siamese_vs_others.sh "
                       "--skip-siamese"], (cmd, out[-2500:])
        # the printed command, pasted into a shell without those settings,
        # submits: the targets are its, and every job gets them
        rc, out, jobs = _root_run(tmp, _line=cmd[0])
        assert rc == 0 and "ERROR" not in out \
            and "seed 7 | run token T1" in out \
            and "    joint         PCA on the concatenation of all modalities " \
                "(150 components)\n" in out, out[-2500:]
        assert {tuple(v[5:]) for v in jobs["_env"].values()} == {
            ("none joint both", "0.8", "0.8", "150")}, jobs["_env"]
        # PCA_ARMS unset (config.conf's four): the resubmit line pins them,
        # and, from a shell exporting PCA_ARMS=joint, still runs all four
        # (2026-10-22: it carried no PCA_ARMS, and ran arm joint alone)
        seed({"none": {}, "joint": c150, "both": b150})
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PCA_ARMS=None)
        cmd = [ln.strip() for ln in out.splitlines()
               if ln.endswith(" bash ./run_benchmarking_siamese_vs_others.sh "
                              "--skip-siamese")]
        assert rc == 1 and len(cmd) == 1 and cmd[0].startswith(
            'PCA_ARMS="none joint per_modality both" PCA_MODALITY_VARIANCE=0.8 '
            "PCA_JOINT_VARIANCE=0.8 PCA_JOINT_DIM=150 USE_CELL_LINES=1 "), \
            (cmd, out[-2500:])
        rc, o2, jobs = _root_run(tmp, _line=cmd[0], PCA_ARMS="joint")
        assert rc == 0 and {f"compare/{a}" for a in _ARMS} <= set(jobs), \
            (o2[-2500:], list(jobs))
        # ... and so does its line that keeps one of an arm's tokens (arm
        # none: no selection, Step-1 runs of two tokens), pasted with one
        # from a shell exporting PCA_ARMS=joint
        seed({"joint": c150})
        for k, tok in enumerate(("T5", "T6")):
            d = os.path.join(sia, "results", f"pcavar0.8_t{k}_cv1")
            os.makedirs(d)
            with open(os.path.join(d, "run_token.txt"), "w") as f:
                f.write(tok + "\n")
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PCA_ARMS=None)
        keep = [ln.strip() for ln in out.splitlines()
                if ln.strip().startswith("RUN_TOKEN=<token> ")]
        assert rc == 1 and len(keep) == 1 and keep[0].startswith(
            'RUN_TOKEN=<token> PCA_ARMS="none joint per_modality both" '), \
            (keep, out[-2500:])
        rc, o2, jobs = _root_run(tmp, _line=keep[0].replace("<token>", "T6"),
                                 PCA_ARMS="joint")
        assert rc == 0 and {f"compare/{a}" for a in _ARMS} <= set(jobs) \
            and jobs["_tokens"]["compare/none"] == "T6", \
            (o2[-2500:], jobs.get("_tokens"))
        # 0.80 is this shell's 0.8
        seed({"joint": {"post_pca": 0.8}})
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PCA_ARMS="joint",
                                  PCA_JOINT_VARIANCE="0.80")
        assert rc == 0 and "ERROR" not in out \
            and "trained at PCA targets" not in out, out[-2500:]
        # joint at 150, both at this shell's: no one set; a WARNING naming
        # joint, and the run goes on
        seed({"none": {}, "joint": c150, "both": b08})
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none joint both")
        assert rc == 0 and "compare/joint" in jobs and (
            "WARNING: --skip-siamese reuses PCA arms trained at PCA targets "
            "that no one set of\n  exports reproduces for all of them") in out \
            and "the tables of PCA arm(s) joint would be PAPER INCOMPLETE" \
            in out and "\n    joint: PCA_JOINT_DIM=150\n" in out \
            and "both: siamese_sl" not in out, out[-2500:]
        # every variant's and every CV's markers are read (2026-10-21: no
        # case put the odd combo past the first variant and CV, so a check
        # of SIAMESE_VARIANTS[0] or CV_TYPES[0] alone passed): joint's
        # pd:pooled combo of cv3 retrained at 150 components (a variant-
        # subset rerun), its pd:per_head ones at this shell's: no one set, a
        # WARNING naming that combo, and the run goes on; with only pd:pooled
        # combos, at 150: refused with PCA_JOINT_DIM=150 (the combos listed),
        # and a run that leaves joint out rebuilds it with it
        vs = "pd:per_head pd:pooled"
        pooled = "joint: siamese_sl/results/pca_joint/var_pd_pooled__best_" \
                 "cat_x_cv3  (pd:pooled: PCA on the concatenation of all " \
                 "modalities (150 components))\n"
        for cfgs, flags, arms in (
                ({**{("", cv): {"post_pca": 0.8} for cv in ("cv1", "cv2",
                                                          "cv3")},
                  ("var_pd_pooled__", "cv3"): c150},
                 ("--skip-siamese",), "joint"),
                ({("var_pd_pooled__", cv): c150 for cv in ("cv1", "cv2",
                                                           "cv3")},
                 ("--skip-siamese",), "joint"),
                ({("var_pd_pooled__", cv): c150 for cv in ("cv1", "cv2",
                                                           "cv3")},
                 ("--skip-generate",), "none")):
            seed({})
            _seed_combos(sia, "joint", cfgs)
            rc, out, jobs = _root_run(tmp, *flags, PCA_ARMS=arms,
                                      SIAMESE_VARIANTS_LIST=vs)
            if ("", "cv1") in cfgs:
                assert rc == 0 and "compare/joint" in jobs and (
                    "WARNING: --skip-siamese reuses PCA arms trained at PCA "
                    "targets that no one set of\n") in out \
                    and "\n    " + pooled in out, out[-2500:]
            elif arms == "joint":
                assert rc == 1 and not jobs and "\n    " + pooled in out \
                    and re.search(r'PCA_JOINT_DIM=150 USE_CELL_LINES=1 '
                                  r'SIAMESE_VARIANTS_LIST="[^"]+" bash '
                                  r"\./run_benchmarking_siamese_vs_others\.sh "
                                  r"--skip-siamese", out), out[-2500:]
            else:
                assert rc == 0 and ("_c_joint=$(_paste_env PCA_ARM=joint "
                                    "PCA_JOINT_DIM=150 sbatch --parsable") \
                    in out.split("All jobs submitted")[1], out[-2500:]
        # a run that leaves joint out (it retrains the shared models):
        # joint's rebuild carries joint's own targets, both's none
        seed({"none": {}, "joint": c150, "both": b08})
        rc, out, jobs = _root_run(tmp, "--skip-generate", PCA_ARMS="none")
        tail = out.split("All jobs submitted")[1]
        assert rc == 0 and ("_c_joint=$(_paste_env PCA_ARM=joint "
                            "PCA_JOINT_DIM=150 sbatch --parsable") in tail \
            and ("_c_both=$(_paste_env PCA_ARM=both sbatch "
                 "--parsable") in tail \
            and _paste_def("PCA_ARMS=none joint both", *_TARGETS0,
                           *_VARIANTS0) in tail, \
            tail


def test_skip_siamese_names_an_arm_with_nothing_on_disk():
    # An arm of PCA_ARMS with nothing in this checkout (no selection, no
    # combo marker, no single-modality run in its owner's root: every run
    # before 2026-10-20 was arm none's) gets one NOTE, and no WARNING that
    # --skip-siamese "reuses its siamese tree", nor a token of its own that
    # would read as another submission's: it takes the first reused arm's
    # (an empty FIRST arm's is every job's RUN_TOKEN: the reused arm's).
    # Still queued (its tables end PAPER INCOMPLETE). joint beside none's
    # Step-1 runs is no such arm: it takes its owner's token, silently.
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        res = os.path.join(root, "siamese_sl", "results")
        os.makedirs(os.path.join(res, "pcavar0.8_go_cv1"))
        with open(os.path.join(res, "pcavar0.8_go_cv1", "run_token.txt"),
                  "w") as f:
            f.write("T1\n")
        with open(os.path.join(res, "best_per_category.json"), "w") as f:
            json.dump({"_selection_meta": {"run_token": "T1"}}, f)
        with open(os.path.join(res, "latest_combo_cv1.txt"), "w") as f:
            f.write("results/x_cv1\n")
        rc, out, jobs = _root_run(tmp, "--skip-siamese", PCA_ARMS=None)
        assert rc == 0, out[-2500:]
        assert ("NOTE: PCA arm(s) per_modality both have no siamese outputs "
                "here (no best_per_category.json,\n  latest_combo marker or "
                "single-modality run in their roots)") in out \
            and "reuses the siamese tree of PCA arm" not in out \
            and "different tokens" not in out \
            and out.count("no earlier code manifest carries run token") == 1 \
            and "carries run token T1 under" in out, out[-3000:]
        assert set(jobs["_tokens"].values()) == {"T1"} and all(
            f"{j}/{a}" in jobs for j in ("compare", "ext", "tables")
            for a in _ARMS), (jobs.get("_tokens"), out[-2500:])
        # nothing anywhere: every arm named, one fresh token for the run
        shutil.rmtree(res)
        os.makedirs(res)
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none joint")
        toks = set(jobs["_tokens"].values())
        assert rc == 0 and "NOTE: PCA arm(s) none joint have no siamese " \
            "outputs here" in out and len(toks) == 1 and re.fullmatch(
                r"\d{8}T\d{6}-\d+", toks.pop()) \
            and "reuses the siamese tree" not in out, out[-2500:]
        # an empty FIRST arm (a checkout that ran only the PCA arms): its
        # token is RUN_TOKEN, every job's, so it is the reused arm's (a fresh
        # one would have per_modality's tables judge its reused ablation dirs
        # another run's). (The banner, anchored: the manifest NOTE also says
        # "run token T2".) 2026-10-21: no case had an empty first arm.
        os.makedirs(os.path.join(res, "pca_per_modality"))
        with open(os.path.join(res, "pca_per_modality",
                               "best_per_category.json"), "w") as f:
            json.dump({"_selection_meta": {"run_token": "T2"}}, f)
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none per_modality")
        assert rc == 0 and "NOTE: PCA arm(s) none have no siamese outputs " \
            "here" in out and set(jobs["_tokens"].values()) == {"T2"} \
            and "run token T2 (the reused selection's" in out \
            and "different tokens" not in out, \
            (jobs.get("_tokens"), out[-2500:])
        # arms of two submissions: the empty arm's jobs run, and it is
        # listed, under the FIRST reused arm's token, not one of its own (nor
        # the last one's)
        with open(os.path.join(res, "best_per_category.json"), "w") as f:
            json.dump({"_selection_meta": {"run_token": "T1"}}, f)
        rc, out, jobs = _root_run(tmp, "--skip-siamese",
                                  PCA_ARMS="none per_modality both")
        toks = jobs["_tokens"]
        assert rc == 0 and "NOTE: PCA arm(s) both have no siamese outputs " \
            "here" in out \
            and "\n    none: T1\n    per_modality: T2\n    both: T1\n" in out \
            and {toks[f"{j}/both"] for j in ("compare", "ext", "tables")} \
            == {"T1"}, (toks, out[-2500:])


def test_root_names_settings_inherited_from_the_calling_shell():
    """An export left in the login shell is honoured by every job of the run
    (the confs' ${VAR:-default}, sbatch --export=ALL), so the submission names
    each one under the banner; with none set it prints not one line more."""
    mark = "comes from the calling shell"
    stray = {"PCA_JOINT_DIM": "256", "PCA_METHOD": "robust",
             "PREPROCESSING_FIT_SCOPE": "all", "SIAMESE_ENCODER_TYPE": "mlp",
             "CELL_LINE_DIM": "8", "SELECTION_CV": "cv3",
             "ARCH_ARGS": "--layernorm --log_temperature",
             "PCA_JOINT_VARIANCE": "0.7", "PCA_MODALITY_VARIANCE": "0.9"}
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        rc, out, jobs = _root_run(tmp, "--force-generate", PCA_ARMS=None)
        assert rc == 0 and "the calling shell" not in out, out[-3000:]
        # set but empty is the default: nothing to name
        rc, out, _ = _root_run(tmp, "--force-generate", PCA_ARMS=None, ARCH_ARGS="",
                               POST_PCA_DIM="", MAX_CONCURRENT="",
                               PCA_JOINT_DIM="")
        assert rc == 0 and "the calling shell" not in out, out[-3000:]
        rc, out, jobs2 = _root_run(tmp, "--force-generate", PCA_ARMS=None, **stray)
        assert rc == 0 and _graph(jobs2) == _graph(jobs), out[-3000:]
        lines = out.splitlines()
        for k, v in stray.items():
            assert (f"NOTE: {k}='{v}' {mark}: every job of this run gets it."
                    in lines), (k, out[:3000])
        assert sum(mark in ln for ln in lines) == len(stray), out[:3000]
        # ...and once more on the last screen, by name
        last = [ln for ln in lines if ln.startswith(
            "NOTE: taken from the calling shell (values under the banner")]
        assert len(last) == 1 and set(last[0].split(": ", 2)[2].split()) \
            == set(stray) and lines.index(last[0]) > lines.index(
                next(ln for ln in lines if ln.startswith("All jobs "))), \
            out[-3000:]
        # under the banner, before anything is checked, changed or queued
        assert max(i for i, ln in enumerate(lines) if mark in ln) < \
            min(i for i, ln in enumerate(lines) if ln.startswith("---")), \
            out[:3000]
        # what the script replaces or clears keeps its own NOTE only
        rc, out, _ = _root_run(tmp, "--force-generate", PCA_ARMS=None, RUN_TOKEN="OLD",
                               EVAL_VARIANT="pd:pooled", PCA_ARM="joint")
        assert rc == 0 and mark not in out \
            and "NOTE: replaced RUN_TOKEN=OLD" in out \
            and "NOTE: ignoring PCA_ARM=joint from the environment" in out, \
            out[:3000]
        # the arms, picked in the calling shell, are named like the rest
        rc, out, jobs = _root_run(tmp, "--force-generate", PCA_ARMS="both")
        assert rc == 0 and _graph(jobs) == _full(("both",)) \
            and f"NOTE: PCA_ARMS='both' {mark}: every job of this run gets it." \
            in out.splitlines(), out[:3000]
        # the export the script makes itself under --skip-eval is not one
        rc, out, _ = _root_run(tmp, "--force-generate", "--skip-eval",
                               PCA_ARMS=None)
        assert rc == 0 and mark not in out, out[-3000:]


def _folds_state(root):
    """{path: sha256} of every file of the copy's sl_comparison/folds."""
    out = {}
    for dp, _, fn in os.walk(os.path.join(root, "sl_comparison", "folds")):
        for f in fn:
            with open(os.path.join(dp, f), "rb") as fh:
                out[os.path.relpath(os.path.join(dp, f), root)] = \
                    hashlib.sha256(fh.read()).hexdigest()
    return out


def test_root_printed_commands_carry_the_calling_shells_settings():
    # The commands the root prints to finish its work after it (the retrain
    # recipe of the left-out arms, in step 0a's NOTE and the final listing;
    # its one-submission alternative; the rebuild block) carry the settings
    # it took from the calling shell, as given (printf %q): a one-shot SEED=7
    # ARCH_ARGS=... PCA_JOINT_DIM=150 in front of the script never reached
    # the shell the command is pasted into, and pasted there the recipe reran
    # step 0 at seed 42 under the shared rows it reuses, and the arms without
    # ARCH_ARGS. Each command sets its own PCA_ARMS, and none carries
    # ALLOW_LIVE_JOBS: they are for once this run has finished, and carried
    # it would lift the guard that keeps their reset and step 0 off this
    # run's jobs. Pasted into a fresh shell (bash, zsh), the recipe trains at
    # this run's seed, ARCH_ARGS and targets and leaves the fold set as it is.
    one_shot = {"SEED": "7", "ARCH_ARGS": "--layernorm --log_temperature",
                "PCA_JOINT_DIM": "150", "ALLOW_LIVE_JOBS": "1"}
    # (the settings from the calling shell, then the targets, USE_CELL_LINES
    # and the variants in full: since 2026-10-22 every submission the root
    # prints pins them, and its arms)
    four = "pd:per_head pd:pooled bilinear:per_head bilinear:pooled"
    prefix = (r"ARCH_ARGS=--layernorm\ --log_temperature SEED=7 "
              "PCA_MODALITY_VARIANCE=0.8 PCA_JOINT_VARIANCE=0.8 "
              f'PCA_JOINT_DIM=150 USE_CELL_LINES=1 SIAMESE_VARIANTS_LIST="{four}" ')
    script = "bash ./run_benchmarking_siamese_vs_others.sh"
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        _seed_arms(root)
        rc, out, jobs = _root_run(tmp, "--force-generate",
                                  PCA_ARMS="none joint", **one_shot)
        assert rc == 0, out[-2500:]
        folds = _folds_state(root)
        rec = [ln.strip() for ln in out.splitlines()
               if ln.endswith("--skip-generate --skip-shared")]
        want = [prefix + f'PCA_ARMS="per_modality both" {script} '
                "--skip-generate --skip-shared",
                f"env -u PCA_ARMS {prefix}{script} --skip-generate "
                "--skip-shared"]
        assert rec == want * 2, rec          # step 0a's NOTE, the listing
        for sh in ["bash"] + [z for z in (shutil.which("zsh"),) if z]:
            rc, out2, jobs2 = _root_run(tmp, _line=want[0], _sh=sh)
            assert rc == 0 and "| seed 7 |" in out2 and "NOTE: ARCH_ARGS=" \
                "'--layernorm --log_temperature' comes from the calling shell" \
                in out2 and "ALLOW_LIVE_JOBS" not in out2, (sh, out2[-2500:])
            assert "combo/both" in jobs2 and "combo/none" not in jobs2 \
                and {v[8] for v in jobs2["_env"].values()} == {"150"}, \
                (sh, jobs2.get("_env"))
            assert _folds_state(root) == folds, sh
        # ... and from a shell exporting USE_CELL_LINES=0, another variant
        # list and PCA_ARMS (2026-10-22: the recipe then retrained and tabled
        # pd:per_head alone, after moving every arm's SLeuth rows aside)
        rc, out2, jobs2 = _root_run(tmp, _line=want[0], USE_CELL_LINES="0",
                                    SIAMESE_VARIANTS_LIST="bilinear:pooled",
                                    PCA_ARMS="joint")
        assert rc == 0 and "combo/both" in jobs2 \
            and "combo/joint" not in jobs2 \
            and set(jobs2["_vars"].values()) == {(four, "1")}, \
            (out2[-2500:], jobs2.get("_vars"))
        assert _folds_state(root) == folds
        # with --skip-generate the left-out arms keep their markers: the
        # rebuild block carries this run's targets (150 components) and its
        # one-submission alternative the prefix
        _seed_arms(root)
        rc, out, jobs = _root_run(tmp, "--skip-generate",
                                  PCA_ARMS="none joint", **one_shot)
        tail = out.split("All jobs submitted")[1]
        assert rc == 0 and _paste_def(
            "PCA_ARMS=none joint per_modality both",
            "PCA_MODALITY_VARIANCE=0.8", "PCA_JOINT_VARIANCE=0.8",
            "PCA_JOINT_DIM=150", *_VARIANTS0) in tail and (
                f'models once more): {prefix}PCA_ARMS="none joint '
                f'per_modality both" {script} --skip-siamese\n') in tail, tail


def test_banners_count_the_jobs_and_tasks_they_queue():
    # Slurm's MaxSubmitJobs counts every array task, and four arms queue 491
    # tasks from the root (586 from submit_pipeline.sh) where one arm queued
    # 160: each banner says how many jobs and tasks it will queue, before
    # anything changes, and it is the count it submits, under every flag.
    # The root's disk adds the cell-line ablation's fold matrices and gives
    # the single-embedding runs as an upper bound (their optional embeddings'
    # tasks skip when the file is absent). Both say the figure is the fold
    # matrices and pair export the submission keeps, and name what it does
    # not count, under the conditions it is written (2026-10-21): the eval's
    # per-pair CSVs, the peers' fold matrices (root, with the shared array)
    # and the checkpoints.
    def counted(out, word):
        m = re.search(word + r": +(\d+) jobs, (\d+) tasks counting every "
                      r"array task", out)
        assert m, out[:4000]
        return int(m.group(1)), int(m.group(2))

    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        for flags, env in (((), {"PCA_ARMS": None}),
                           (("--skip-reset-env",), {"CVS": "cv1 cv3"}),
                           (("--skip-siamese",), {"PCA_ARMS": None}),
                           (("--skip-shared", "--skip-generate"),
                            {"PCA_ARMS": "joint"}),
                           (("--skip-eval", "--skip-cellline-ablation"),
                            {"PCA_ARMS": "none both"}),
                           (("--skip-ablation", "--skip-pair-export",
                             "--skip-paper-tables"), {"PCA_ARMS": None}),
                           (("--after", "555"),
                            {"PCA_ARMS": "per_modality",
                             "USE_CELL_LINES": "0"})):
            rc, out, jobs = _root_run(tmp, *flags, **env)
            g = _graph(jobs)
            assert rc == 0 and counted(out, "queues") == \
                _queued_count(jobs), (flags, env, counted(out, "queues"),
                                      list(g))
            if flags == ():
                # (43 jobs / 491 tasks + arm none's 3-task bilinear probe
                # + ELISL's 3-task array, 2026-11-13)
                assert counted(out, "queues") == (45, 497)
                flat = " ".join(out.split())
                assert ("disk of the fold matrices and pair export it keeps, "
                        "up to: 910 GB in all; per arm 19 GB of combos, "
                        "42 GB of pair export (PAIR_EXPORT_* defaults), 113 GB "
                        "of ablation fold matrices, 9 GB of cell-line ablation "
                        "fold matrices; per owner up to 89 GB of "
                        "single-embedding fold matrices (75 GB when the "
                        "optional esmc scprint seqvec are absent: their 9 "
                        "Step-1 tasks skip; then 882 GB in all)") in flat, flat
            # what the figure does not count, under its conditions
            if "--skip-siamese" in flags:
                assert "not counted:" not in out, out
            else:
                said = [ln.strip() for ln in out.splitlines()
                        if ln.strip().startswith("not counted: ")]
                want = [x for x, on in (
                    ("the eval's per-pair CSVs (about 5 GB per arm by "
                     "default)", "--skip-eval" not in flags),
                    ("the peers' fold matrices (about %d GB, once)" % (
                        14 if env.get("CVS") is None else 9),
                     "--skip-shared" not in flags)) if on] + ["the checkpoints"]
                assert said == ["not counted: " + ", ".join(want[:-1]) + (
                    " and " if len(want) > 1 else "") + want[-1]], (flags, said)
                assert ("disk of the fold matrices%s it keeps, up to: " % (
                    "" if "--skip-pair-export" in flags
                    else " and pair export")) in out, (flags, out[:4000])
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        for args, arms in (((), None),
                           (("--skip-generate", "--skip-reset-env"), "joint"),
                           (("--best-cat-only", "--skip-eval"), None),
                           (("--benchmark-only", "--skip-generate"),
                            "none both"),
                           (("--skip-ablation", "--skip-pair-export"), None),
                           (("--ablation-only",), "none per_modality")):
            if args == ("--ablation-only",):
                for a in ("none", "per_modality"):
                    _seed_combos(sia, a, {})
                    with open(os.path.join(sia, _root_of(a),
                                           "latest_combo_cv1.txt"), "w") as f:
                        f.write(f"{_root_of(a)}/x_cv1\n")
            jobs, out = _submit(tmp, *args, PCA_ARMS=arms)
            tasks = sum(_array_tasks(m.group(1)) if m else 1 for m in (
                re.search(r"--array=(\S+)", a) for _, _, a in jobs))
            assert counted(out, "Queues") == (len(jobs), tasks), (
                args, arms, counted(out, "Queues"), jobs)
            if args == ():
                assert counted(out, "Queues") == (28, 586) \
                    and ("  Disk, up to:    1062 GB in all, of the fold "
                         "matrices and pair export it keeps\n") in out \
                    and "per owner: up to 183 GB of single-embedding fold " \
                        "matrices (a task\n" in out, out[:4000]
            # what it does not count: the eval's CSVs when a combo is
            # evaluated, the checkpoints always
            evals = "--skip-eval" not in args and not {
                "--benchmark-only", "--ablation-only"} & set(args)
            assert ("\n                  not counted: " + (
                "the eval's per-pair CSVs (about 5 GB per arm by default) and "
                if evals else "") + "the checkpoints\n") in out, (args, out[:4000])
            if args == ("--ablation-only",):
                assert "  Disk, about:    226 GB in all, of the fold matrices " \
                    "it keeps\n" in out, out[:4000]


def test_generation_job_banner_names_the_runs_arms():
    # The generation job serves every arm, and inherits the orchestrator's
    # PCA_ARM=none: its banner named arm none and results/, even in a run
    # without arm none (2026-10-21). run_generate_embeddings.sh marks itself
    # (a plain shell variable, right before it sources config.conf) and
    # config.conf's compute-node banner then names the run's arms; any other
    # job keeps its arm's line.
    def banner(pre="", **env):
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "PCA_", "POST_PCA_"))}
        e.update(SLURM_JOB_ID="77", SLURM_JOB_NAME="x",
                 SLURM_SUBMIT_DIR=_SIAMESE, **env)
        r = subprocess.run(
            ["bash", "-c", "module() { :; }; nvidia-smi() { :; }; " + pre
             + "source slurm/config.conf"], cwd=_SIAMESE, env=e,
            capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        return [ln for ln in r.stdout.splitlines() if ln.startswith("PCA arm")]

    assert banner("PCA_EVERY_ARM=1; ", PCA_ARMS="joint both") == [
        "PCA arms: joint both (a job shared by every arm: no arm's results "
        "root)"]
    assert banner(PCA_ARMS="joint both") == [
        "PCA arm: none (no PCA) | results: results/"]
    assert banner(PCA_ARMS="joint both", PCA_ARM="joint") == [
        "PCA arm: joint (PCA on the concatenation of all modalities (variance "
        "0.8)) | results: results/pca_joint/"]
    with open(os.path.join(_SIAMESE, "slurm",
                           "run_generate_embeddings.sh")) as f:
        code = [ln.strip() for ln in f
                if ln.strip() and not ln.lstrip().startswith("#")]
    i = code.index('source "$SLURM_SUBMIT_DIR/slurm/config.conf"')
    assert code[i - 1] == "PCA_EVERY_ARM=1" and not [
        ln for ln in code if re.search(r"\bexport\b.*PCA_EVERY_ARM", ln)], \
        code[:i + 1]


def test_root_names_an_ignored_variants_alias_as_ignored():
    """SIAMESE_VARIANTS on the command line is only an alias: config.conf
    reads it when SIAMESE_VARIANTS_LIST is empty and ignores it otherwise, and
    the jobs get the variants as SIAMESE_VARIANTS_LIST alone (bash exports no
    array). The NOTE lines must agree with the banner's variant list and with
    what every job inherits: beside a list the alias is named as ignored,
    never as inherited by every job."""
    gets = "comes from the calling shell: every job of this run gets it."
    alias, lst = "pd:per_head pd:pooled", "bilinear:pooled"
    ignored = (f"NOTE: ignoring SIAMESE_VARIANTS='{alias}' from the calling "
               "shell: SIAMESE_VARIANTS_LIST is set too, and wins.")
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        stub = os.path.join(tmp, "bin", "sbatch")
        seen = os.path.join(tmp, "variants.seen")
        with open(stub) as f:
            text = f.read()
        assert text.endswith("echo $n\n"), text[-60:]
        with open(stub, "w") as f:      # what each job would inherit
            f.write(text[:-len("echo $n\n")]
                    + 'echo "${SIAMESE_VARIANTS-<unset>}|'
                      '${SIAMESE_VARIANTS_LIST-<unset>}" >> "%s"\necho $n\n'
                    % seen)

        def run(**env):
            """(NOTE lines, the banner's variants, what the jobs inherit as
            'alias|list', the names of the closing reminder)."""
            if os.path.exists(seen):
                os.remove(seen)
            rc, out, jobs = _root_run(tmp, "--force-generate", PCA_ARMS=None,
                                      **env)
            assert rc == 0 and "combo/none" in jobs, out[-2000:]
            lines = out.splitlines()
            with open(seen) as f:
                inherited = sorted(set(f.read().split("\n")[:-1]))
            banner = [ln for ln in lines if "SL benchmark: SLeuth" in ln]
            last = [ln for ln in lines if ln.startswith(
                "NOTE: taken from the calling shell (values under the banner")]
            assert len(banner) == 1 and len(last) == 1, out[-3000:]
            return ([ln for ln in lines if ln.startswith("NOTE: ")],
                    banner[0].split("variants: ")[1].rstrip(")"),
                    inherited, last[0].split(": ", 2)[2].split())

        # both set: the list is trained and inherited, the alias is neither
        notes, trained, inherited, names = run(SIAMESE_VARIANTS=alias,
                                               SIAMESE_VARIANTS_LIST=lst)
        assert trained == lst and inherited == [f"<unset>|{lst}"], \
            (trained, inherited)
        named = f"NOTE: SIAMESE_VARIANTS_LIST='{lst}' {gets}"
        assert named in notes and ignored in notes \
            and notes.index(ignored) == notes.index(named) + 1, notes
        assert not any(ln.startswith("NOTE: SIAMESE_VARIANTS=")
                       for ln in notes), notes
        assert names == ["SIAMESE_VARIANTS_LIST"], names
        # the alias alone, or beside an EMPTY list, is in force: its value is
        # the list every job inherits, and it keeps the NOTE that says so
        for env in (dict(SIAMESE_VARIANTS=alias),
                    dict(SIAMESE_VARIANTS=alias, SIAMESE_VARIANTS_LIST="")):
            notes, trained, inherited, names = run(**env)
            assert trained == alias and inherited == [f"<unset>|{alias}"], \
                (env, trained, inherited)
            assert f"NOTE: SIAMESE_VARIANTS='{alias}' {gets}" in notes, \
                (env, notes)
            assert not any("ignoring SIAMESE_VARIANTS" in ln
                           for ln in notes), (env, notes)
            assert names == ["SIAMESE_VARIANTS"], (env, names)


def test_an_inherited_pca_arm_is_ignored():
    # PCA_ARM is the arm of ONE job: one left exported in the calling shell
    # (meant for a direct sbatch) neither picks the arms (PCA_ARMS does) nor
    # stops the submission (a mistyped one would make config.conf refuse);
    # each orchestrator names it and drops it, and every arm's jobs get
    # their own. --eval-only takes the arm of its directory, not that one.
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        for inherited in ("joint", "bogus"):
            jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                                PCA_ARMS="none per_modality",
                                PCA_ARM=inherited)
            assert {n.split("/")[1] for n in _arm_names(jobs) if "/" in n} \
                == {"none", "per_modality"}, jobs
            assert f"NOTE: ignoring PCA_ARM={inherited} from the environment" \
                in out, out[:2000]
        d = os.path.join(sia, "results", "x_cv3")
        os.makedirs(d)
        for f in ("config.json", "results.json"):
            with open(os.path.join(d, f), "w") as fh:
                fh.write("{}")
        jobs, _ = _submit(tmp, "--eval-only", "results/x_cv3",
                          PCA_ARM="joint")
        assert _arm_names(jobs) == ["eval_finetuning.sh/none",
                                    f"{_PAIR_SH}/none"], jobs
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        for inherited in ("joint", "bogus"):
            rc, out, jobs = _root_run(tmp, PCA_ARM=inherited)
            assert rc == 0 and _graph(jobs) == _FULL, (inherited, out[-2000:])
            # (named as ignored, never among the settings every job gets)
            assert f"NOTE: ignoring PCA_ARM={inherited} from the environment" \
                in out and "NOTE: PCA_ARM=" not in out, out[:3000]


def _script_log(script):
    """(stage, pattern) of a job script's own #SBATCH --output, e.g.
    ('bestcat_combo', '%j'): the log name arm none keeps."""
    with open(script) as f:
        m = re.search(r"^#SBATCH --output=(?:sl_comparison/)?slurm/logs/"
                      r"(\w+?)_(%j|%A_%a)\.out$", f.read(), re.M)
    assert m, script
    return m.group(1), m.group(2)


def test_every_arm_job_carries_its_arm_and_names_it_in_its_logs():
    # Each arm's jobs are submitted with PCA_ARM=<arm> in sbatch's own
    # environment (a prefix assignment, _as_arm: sbatch snapshots its
    # environment at submission and --export=ALL hands it to the job; never
    # by exporting one in the orchestrator's shell, and never in an --export
    # list, which Slurm reads as an implied --get-user-env), the jobs of
    # every arm with no PCA_ARM of their own: they inherit PCA_ARM=none
    # (config.conf exports its default), as every job the validated
    # PCA_ARMS. The scripts' own job names stand (no --job-name: the
    # live-job lists and singleton match them), and each arm's log files name
    # it: the script's own <stage>_<pattern> name with pca_<arm>_ before the
    # pattern (arm none: unchanged), absolute in the root orchestrator's logs
    # dirs, relative to siamese_sl/ (the submit dir) in submit_pipeline.sh's.
    scripts = {}
    for d in ("siamese_sl/slurm", "siamese_sl/cell_line_ablation",
              "sl_comparison/slurm"):
        for p in glob.glob(os.path.join(_ROOT, d, "*.sh")):
            scripts[os.path.basename(p)] = p
    with tempfile.TemporaryDirectory() as tmp:
        # (resolved: the root prints its logs dirs from its resolved root,
        # and macOS's default TMPDIR lies under the /var symlink)
        tmp = os.path.realpath(tmp)
        root = _root_tree(tmp)
        rc, out, jobs = _root_run(tmp, "--force-generate", PCA_ARMS=None)
        assert rc == 0, out[-2000:]
        by_name = {v: k for k, v in _JOBS.items()}
        logs_dir = {"siamese": os.path.join(root, "siamese_sl", "slurm",
                                            "logs"),
                    "cmp": os.path.join(root, "sl_comparison", "slurm",
                                        "logs")}
        n_arm = 0
        for name, (o, e) in jobs["_logs"].items():
            if name in _ELISL_JOBS:     # ELISL's own logs dir, its own stem
                stem = "elisl_%A_%a" if name == "elisl" else "elisl_inputs_%j"
                d = os.path.join(root, "ELISL", "slurm", "logs")
                assert (o, e) == (os.path.join(d, stem + ".out"),
                                  os.path.join(d, stem + ".err")), (name, o)
                continue
            job, _, arm = name.partition("/")
            script = by_name["tables" if name == "pca_arms" else job]
            if script == "run_generate_embeddings.sh":
                stage, pat = "siamese_gen_emb", "%j"   # its own is %x_%j
            else:
                stage, pat = _script_log(scripts[script])
            tag = "pca_arms_" if name == "pca_arms" else _tag(arm or "none")
            d = logs_dir["cmp" if script.startswith(
                ("run_shared", "run_compare", "run_external", "run_paper"))
                else "siamese"]
            assert (o, e) == (os.path.join(d, f"{stage}_{tag}{pat}.out"),
                              os.path.join(d, f"{stage}_{tag}{pat}.err")), \
                (name, o, e)
            n_arm += bool(arm)
            # what every job gets: its own arm (none for the jobs of every
            # arm, which inherit the orchestrator's, and for the table across
            # the arms, submitted as the first, reference arm), the arms of
            # the run and their targets
            assert jobs["_env"][name][4:] == [arm or "none", " ".join(_ARMS),
                                              "0.8", "0.8", ""], \
                (name, jobs["_env"][name])
        # (Step-1 arrays of 2 owners, 9 stages per arm, arm none's probe)
        assert n_arm == 2 + 9 * 4 + 1, n_arm
        # (_root_run named the jobs of every arm after their scripts, having
        # checked each passes a plain --export=ALL and inherits none)
        assert {"reset", "gen", "folds", "shared", "pca_arms"} <= set(jobs)
        with open(os.path.join(tmp, "sb.log")) as f:
            for ln in f:
                args = ln.split("|", 2)[2]
                assert not re.search(r"(--job-name|\s-J)\b", args), ln
        # the targets of the calling shell reach every job, the tables'
        # included (their labels, and the trained-vs-target check), and each
        # its own arm: joint's and both's theirs (each owns its pair's
        # Step-1 array), the jobs of every arm none, and the reference arm,
        # the first of the list, submits the table across the arms
        rc, out, jobs = _root_run(tmp, PCA_ARMS="joint both",
                                  PCA_MODALITY_VARIANCE="0.9",
                                  PCA_JOINT_VARIANCE="0.7", PCA_JOINT_DIM="300")
        assert rc == 0 and {tuple(v[5:]) for v in jobs["_env"].values()} == {
            ("joint both", "0.9", "0.7", "300")}, jobs.get("_env")
        for name, v in jobs["_env"].items():
            want = "joint" if name == "pca_arms" else (
                name.partition("/")[2] or "none")
            assert v[4] == want, (name, v)
        assert {n.partition("/")[2] for n in jobs["_env"]} == {
            "", "joint", "both"} and "bestcat/joint" in jobs \
            and "bestcat/both" in jobs, list(jobs["_env"])

    # submit_pipeline.sh: the same rule, relative log paths, none for arm none
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        jobs, out = _submit(tmp, PCA_ARMS=None)
        n_arm = 0
        for job in jobs:
            args = job[2]
            script = args.rsplit("/", 1)[-1]
            arm = _arm_of(job)      # (the jobs of every arm: none inherited)
            assert (arm == "") == (script in ("reset_env.sh",
                                              "run_generate_embeddings.sh")), \
                args
            assert not re.search(r"(--job-name|\s-J)\b", args), args
            logs = [x for x in args.split()
                    if x.startswith(("--output=", "--error="))]
            if not arm or arm == "none":
                assert logs == [], args
                continue
            n_arm += 1
            stage, pat = _script_log(scripts[script])
            assert logs == [f"--output=slurm/logs/{stage}_pca_{arm}_{pat}.out",
                            f"--error=slurm/logs/{stage}_pca_{arm}_{pat}.err"], \
                args
        assert n_arm == 3 + 5 * 3, n_arm


def test_no_submission_or_printed_command_passes_an_export_list():
    # Any --export value but ALL / NIL / NONE (--export=ALL,PCA_ARM=<arm>,
    # the arms' form of 2026-10-20) makes Slurm imply --get-user-env: slurmd
    # runs `su - <user> -c env` at every task launch and lays the job's
    # environment over the login one, and from Slurm 25.05 a fetch that fails
    # or takes over 120 s requeues the job held. Each arm's PCA_ARM (and its
    # own RUN_TOKEN) rides in sbatch's environment instead, with a plain
    # --export=ALL (every run in this file checks each submission it makes:
    # _plain_export). Here the code itself: in both orchestrators every
    # submission passes exactly --export=ALL (literally, or through
    # $_PAIR_OPTS, itself exactly that), an arm's job through _as_arm, the
    # jobs of every arm (_SHARED_SCRIPTS) without it; and no line of code in
    # the pipeline's scripts, confs or python (a submission, or a command
    # one prints to paste) holds an --export of anything but ALL.
    for path in (_ORCH, _SUBMIT):
        subs = [ln for ln in _logical_lines(path)
                if re.search(r"\bsbatch\s+--parsable\b", ln)]
        assert len(subs) >= 8, (path, len(subs))
        for ln in subs:
            exp = re.findall(r"--export\S*", ln)
            assert exp == ["--export=ALL"] or (
                exp == [] and " $_PAIR_OPTS " in ln), (path, ln.strip())
            script = os.path.basename(re.findall(r"([\w./$-]+)\.sh\b", ln)[-1]
                                      + ".sh")
            if script in _SHARED_SCRIPTS:
                assert not re.search(r"\b_as_arm\b|PCA_ARM=", ln), \
                    (path, script, ln.strip())
            else:
                assert re.search(r'_as_arm "\$(\{PCA_ARM_LIST\[0\]\}|\w+)" '
                                 r"sbatch --parsable ", ln), \
                    (path, script, ln.strip())
    bad = []
    for d in ("", "siamese_sl", "siamese_sl/slurm", "siamese_sl/cell_line_ablation",
              "sl_comparison", "sl_comparison/slurm"):
        for p in sorted(glob.glob(os.path.join(_ROOT, d, "*"))):
            if not p.endswith((".sh", ".conf", ".py")) \
                    or os.path.basename(p).startswith("test_"):
                continue
            with open(p) as f:
                for n, ln in enumerate(f, 1):
                    if not ln.lstrip().startswith("#") and re.search(
                            r"--export=(?!ALL(?![\w,]))", ln):
                        bad.append(f"{os.path.relpath(p, _ROOT)}:{n}: "
                                   f"{ln.strip()[:120]}")
    assert not bad, "\n".join(bad)


def test_help_describes_the_pca_arms():
    # --help prints each orchestrator's header: it says how the arms are
    # chosen (PCA_ARMS and their targets), where each one writes, that every
    # arm's job carries its PCA_ARM, and (root) the table across the arms;
    # nothing of the fixed-selection PCA ablation it replaced is left; a
    # recipe that trains a subset of the arms keeps the embeddings (a
    # regeneration removes EVERY arm's combo markers) and says why, and the
    # root's keeps the shared rows too (retrained, they cost the arms it
    # leaves out their tables); the env-reset item of submit_pipeline.sh
    # names every mode that skips the reset unless --reset-env is passed; no
    # blank line of either is doubled; and no line keeps its comment marker
    # (2026-10-22: sed 's/^# \?//' is GNU's; BSD sed, macOS's, left every
    # line's "# ").
    gone = re.compile(r"pca[ _-]?abl|tab_pca skipped|no_pca_ablation"
                      r"|pcaabl", re.I)
    for path, cwd, extra, why in (
            (_SUBMIT, _SIAMESE, ("results/pca_<arm>/",
                                 "results/pca_per_modality/",
                                 "--eval-only results/pca_joint/",
                                 "PCA_ARMS=joint bash slurm/submit_"
                                 "pipeline.sh --skip-reset-env "
                                 "--skip-generate"),
             ("without it the run regenerates the embeddings, which removes "
              "EVERY arm's combo markers",)),
            (_ORCH, _ROOT, ("siamese_sl/results/pca_joint/",
                            "sl_comparison/results/pca_arms/",
                            "--pca_arms_table",
                            "PCA_ARMS=joint bash run_benchmarking_siamese_vs_"
                            "others.sh --skip-generate --skip-shared"),
             ("a run that regenerates the embeddings removes EVERY arm's "
              "combo markers", "one that retrains the shared models removes "
              "the other arms' comparison tables, external transfer and "
              "paper tables (the final listing prints the jobs that rebuild "
              "them, wired to its own)", "the jobs that rebuild its tables, "
              "wired to this run's shared models and to its last table job",
              "the command that retrains it once this run has finished"))):
        r = subprocess.run(["bash", path, "--help"], cwd=cwd,
                           capture_output=True, text=True, timeout=60,
                           env={k: v for k, v in os.environ.items()
                                if not k.startswith(("PCA_", "POST_PCA_",
                                                     "SLURM_"))})
        assert r.returncode == 0 and r.stdout, (path, r.stderr)
        assert not [ln for ln in r.stdout.splitlines()
                    if ln.startswith("#")], (path, r.stdout[:600])
        flat = " ".join(r.stdout.split())
        for want in ("PCA_ARMS", "none", "joint", "per_modality", "both",
                     "PCA_ARM=<arm> sbatch ... --export=ALL",
                     "never in an --export list", "PCA_MODALITY_VARIANCE",
                     "PCA_JOINT_VARIANCE", "PCA_JOINT_DIM", "OWNER",
                     "_pca_<arm>_") + extra:
            assert want in flat, (path, want)
        prose = " ".join(r.stdout.replace("#", " ").split())
        for want in why:
            assert want in prose, (path, want)
        for m in re.finditer(r'PCA_ARMS=(?:"[^"]*"|\S+) bash \S+([^\n]*)',
                             r.stdout):
            if not re.search(r"--(ablation|eval)-only|--skip-siamese",
                             m.group(1)):
                assert "--skip-generate" in m.group(1), (path, m.group(0))
        assert not gone.search(r.stdout), (path, gone.findall(r.stdout))
        with open(path) as f:
            head = f.read().split("\nset -e\n", 1)[0].splitlines()
        assert not [i for i in range(1, len(head))
                    if head[i] == head[i - 1] == "#"], path
    with open(_SUBMIT) as f:
        head = f.read().split("\nset -e\n", 1)[0]
    flat = " ".join(ln.lstrip("#").strip() for ln in head.splitlines())
    reset = flat.split("0. Environment reset", 1)[1].split(
        "1. Embedding generation", 1)[0]
    assert "--eval-only" in reset and "--ablation-only" in reset, reset


def test_the_removed_pca_ablation_flags_are_unknown():
    # The fixed-selection PCA ablation is gone (every arm runs the whole
    # pipeline instead): its flags are refused as unknown options, before
    # anything is checked, changed or queued, never silently accepted.
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        for flag in ("--skip-pca-ablation", "--pca-ablation-only"):
            jobs, out = _submit(tmp, flag, _rc=1)
            assert not jobs and f"ERROR: Unknown option: {flag}" in out, out
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        for flag in ("--skip-pca-ablation", "--pca-ablation-only"):
            rc, out, jobs = _root_run(tmp, flag)
            assert rc == 1 and not jobs \
                and f"ERROR: unknown option: {flag}" in out, out[-1500:]
            assert not os.path.exists(os.path.join(root, "sl_comparison",
                                                   "folds", "meta.json"))


def test_pca_settings_are_refused_before_anything_changes():
    # config.conf validates the arms' settings and refuses POST_PCA_* (the
    # reported model's switch until 2026-10-20: now it would train every arm
    # with the same post-concat PCA, arm none included); both orchestrators
    # source it first, so either one stops before a file changes or a job is
    # queued, with config.conf's own reason.
    bad = ((dict(POST_PCA_VARIANCE="0.8"),
            "POST_PCA_VARIANCE / POST_PCA_DIM are replaced by the PCA arms"),
           (dict(POST_PCA_DIM="512"),
            "POST_PCA_VARIANCE / POST_PCA_DIM are replaced by the PCA arms"),
           (dict(PCA_ARMS="none pca"), "'pca' is not an arm"),
           (dict(PCA_ARMS="joint joint"), "lists joint twice"),
           # one arm per line ($(cat arms.txt)): every line is read, the
           # names after the first refused like the rest (a plain `read`
           # dropped them and ran the first arm alone until 2026-10-21)
           (dict(PCA_ARMS="joint\nbogus"), "'bogus' is not an arm"),
           (dict(PCA_ARMS="joint\njoint"), "lists joint twice"),
           (dict(PCA_ARMS=" "), "PCA_ARMS is empty"),
           (dict(PCA_ARMS=""), "PCA_ARMS is set but empty"),
           (dict(PCA_MODALITY_VARIANCE="80"), "PCA_MODALITY_VARIANCE='80'"),
           (dict(PCA_JOINT_VARIANCE="1"), "PCA_JOINT_VARIANCE='1'"),
           (dict(PCA_JOINT_DIM="0"), "PCA_JOINT_DIM='0'"))
    with tempfile.TemporaryDirectory() as tmp:
        sia, b = _stub_tree(tmp)
        res = os.path.join(sia, "results")
        for n in ("latest_combo_cv1.txt", "best_per_category.json"):
            with open(os.path.join(res, n), "w") as f:
                f.write("{}")
        before = sorted(os.listdir(res))
        for env, said in bad:
            env = dict({"PCA_ARMS": None}, **env)
            jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                                _rc=1, **env)
            assert not jobs and sorted(os.listdir(res)) == before \
                and not os.path.exists(os.path.join(tmp, "sb.log.test")), env
            r = subprocess.run(
                ["bash", "slurm/submit_pipeline.sh"], cwd=sia,
                env=dict({k: v for k, v in os.environ.items()
                          if not k.startswith(("PCA_", "POST_PCA_", "SLURM_"))},
                         PATH=b + os.pathsep + os.environ["PATH"],
                         SB_LOG=os.path.join(tmp, "x.log"),
                         SB_CTR=os.path.join(tmp, "x.ctr"),
                         **{k: v for k, v in env.items() if v is not None}),
                capture_output=True, text=True, timeout=120)
            assert r.returncode != 0 and "ERROR" in r.stderr \
                and said in r.stderr, (env, r.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)

        def state():
            out = {}
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d != "logs"]
                for f in fn:
                    q = os.path.join(dp, f)
                    out[os.path.relpath(q, root)] = os.stat(q).st_mtime_ns
            return out

        before = state()
        for env, said in bad:
            env = dict({"PCA_ARMS": None}, **env)
            rc, out, jobs = _root_run(tmp, "--force-generate", **env)
            assert rc != 0 and not jobs and said in out \
                and "SL benchmark" not in out, (env, rc, out[-1500:])
            assert state() == before, env


def test_submit_pipeline_records_the_roots_settings():
    # submit_pipeline.sh records the settings it takes from the calling
    # shell (before config.conf, printf %q; the commands it prints to run
    # after it carry them) under the root orchestrator's name list, in its
    # order, plus SHARED_FOLDS, which it honours and the root refuses: the
    # two lists cannot drift apart (2026-10-21). Each inherited one is named
    # in a NOTE before anything changes (nothing when none is set), the
    # SIAMESE_VARIANTS alias beside a list as ignored, as the root does.
    def names(path):
        line = next(ln for ln in _logical_lines(path)
                    if re.match(r"\s*for _v in ARCH_ARGS ", ln))
        return re.search(r"for _v in (.*?); do", line).group(1).split()
    root, sp = names(_ORCH), names(_SUBMIT)
    assert len(root) == len(set(root)) and "SEED" in root \
        and set(sp) - set(root) == {"SHARED_FOLDS"} \
        and [n for n in sp if n in root] == root, (root, sp)
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        base = ("--skip-generate", "--skip-reset-env")
        _, out = _submit(tmp, *base, PCA_ARMS=None)
        assert "the calling shell" not in out, out[:3000]
        _, out = _submit(tmp, *base, PCA_ARMS=None, SEED="7",
                         SIAMESE_VARIANTS="pd:pooled",
                         SIAMESE_VARIANTS_LIST="pd:per_head")
        lines = out.splitlines()
        notes = [ln for ln in lines if "the calling shell" in ln]
        assert notes == [
            "NOTE: SIAMESE_VARIANTS_LIST='pd:per_head' comes from the calling "
            "shell: every job of this run gets it.",
            "NOTE: ignoring SIAMESE_VARIANTS='pd:pooled' from the calling "
            "shell: SIAMESE_VARIANTS_LIST is set too, and wins.",
            "NOTE: SEED='7' comes from the calling shell: every job of this "
            "run gets it."] and lines.index(notes[-1]) < min(
                i for i, ln in enumerate(lines) if ln.startswith("---")), \
            notes


def test_sp_block_carries_the_runs_one_shot_settings():
    # submit_pipeline.sh's block stands in for the root's sl_comparison
    # jobs, which inherit every setting of the submission (--export=ALL).
    # Pasted into a shell that never had this run's one-shot prefix (and
    # that exports other values of its own), each job must still get the
    # run's, as this run resolved them: the paper tables resolve the
    # selection variant, the Step-1 dir tag and BESTCAT_EXTRA_VARIANTS from
    # the confs (run_paper_tables.sh's SETTINGS block, run here in each
    # paper-tables job's environment) and record ARCH_ARGS, the compare
    # grades CVS. Without them (until 2026-10-21) tab:solo read pcavar0.8_*
    # dirs of no run, every selection cell read as another variant's, and
    # the records said the defaults: false PAPER INCOMPLETEs. A value
    # config.conf does not validate (ARCH_ARGS) is printf %q-quoted: bash
    # and zsh read back the same string, nothing in it expanded.
    one_shot = {"SELECTION_VARIANT": "bilinear:per_head",
                "BEST_CAT_PCA_VARIANCE": "0.9", "BESTCAT_EXTRA_VARIANTS": "1",
                "ARCH_ARGS": "--layernorm --log_temperature", "CVS": "cv1 cv3"}
    stray = {"SELECTION_VARIANT": "bilinear:pooled",
             "BEST_CAT_PCA_VARIANCE": "0.5", "BESTCAT_EXTRA_VARIANTS": "0",
             "ARCH_ARGS": "--from-the-pasting-shell", "CVS": "cv2"}
    every = "pd:per_head pd:pooled bilinear:per_head bilinear:pooled"
    with open(os.path.join(_ROOT, "sl_comparison", "slurm",
                           "run_paper_tables.sh")) as f:
        src = f.read()
    m = re.search(r"^SETTINGS=\$\(bash -c '.*?^SETTING_ARGS=\"\"\n"
                  r"if .*?^fi\n", src, re.S | re.M)
    assert m, "run_paper_tables.sh SETTINGS block not found"
    settings = m.group(0) + 'echo "ARGS=$SETTING_ARGS|ARCH=${ARCH_ARGS-}"\n'
    shells = ["/bin/bash"] + [z for z in (shutil.which("zsh"),) if z]
    with tempfile.TemporaryDirectory() as tmp:
        _stub_tree(tmp)
        _env_stubs(tmp)
        jobs, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                            "--best-cat-only", PCA_ARMS="none joint",
                            **one_shot)
        assert "results/pcavar0.9_*" in out, out[-3000:]
        block = _sp_block(out)
        assert block[0] == _paste_def(
            "PCA_ARMS=none joint", *_TARGETS0, "SIAMESE_VARIANTS_LIST=" + every,
            "USE_CELL_LINES=1", *(f"{k}={v}" for k, v in one_shot.items())), \
            block[0]
        for sh in shells:
            log, ctr = os.path.join(tmp, "pl"), os.path.join(tmp, "pc")
            for f in glob.glob(log + "*") + [ctr]:
                if os.path.exists(f):
                    os.remove(f)
            r = subprocess.run(
                [sh, "-s"], input="\n".join(block) + "\n", cwd=tmp,
                capture_output=True, text=True, timeout=60,
                env=dict(stray, HOME=tmp, SB_LOG=log, SB_CTR=ctr,
                         PATH=os.path.join(tmp, "bin") + os.pathsep
                         + os.environ["PATH"]))
            assert r.returncode == 0 and not r.stderr, (sh, r.stdout, r.stderr)
            with open(log) as f:
                calls = [ln.rstrip("\n").split("|", 2) for ln in f]
            assert len(calls) == 9, (sh, calls)    # 4 per arm, the cross-arm
            tables = []
            for n, _, a in calls:
                with open(f"{log}.env.{n}") as f:
                    jenv = dict(ln.split("=", 1) for ln in f.read().split("\n")
                                if "=" in ln)
                assert {k: jenv.get(k) for k in one_shot} == one_shot, \
                    (sh, a, jenv)
                if a.endswith("run_paper_tables.sh"):
                    tables.append(jenv)
            assert len(tables) == 2, (sh, calls)
            for jenv in tables:
                jenv.pop("SLURM_JOB_ID", None)
                res = subprocess.run(["bash", "-c", settings], cwd=tmp,
                                     env=jenv, capture_output=True,
                                     text=True, timeout=60)
                assert [ln for ln in res.stdout.splitlines()
                        if ln.startswith("ARGS=")] == [
                    "ARGS=--variants " + every + " --selection_variant "
                    "bilinear:per_head --pca_variance 0.9 "
                    "--bestcat_extra_variants 1|ARCH=--layernorm "
                    "--log_temperature"], (sh, res.stdout, res.stderr)
    # the %q branch: a value of characters a shell reads (quotes, $, a
    # backtick, !, a tab) set as given, in bash and zsh alike; a plain one
    # as before, an empty one still set
    odd = "--x 'y' $HOME `id` \"z\" !w\tv"
    r = subprocess.run(
        ["bash", "-c", 'source slurm/config.conf >/dev/null || exit 9; '
         'pasted_env_def "PCA_ARMS=none joint" "ARCH_ARGS=$1" "CVS="',
         "_", odd], cwd=_SIAMESE, capture_output=True, text=True, timeout=60,
        env={k: v for k, v in os.environ.items()
             if not k.startswith(("PCA_", "POST_PCA_", "SLURM_"))})
    line = r.stdout.strip()
    assert r.returncode == 0 and ' PCA_ARMS="none joint" ARCH_ARGS=' in line \
        and " CVS= " in line and "'y'" not in line, (r.stdout, r.stderr)
    for sh in ["/bin/bash"] + [z for z in (shutil.which("zsh"),) if z]:
        p = subprocess.run(
            [sh, "-c", line + '\n_paste_env /bin/sh -c \'printf "%s|%s|%s" '
             '"$ARCH_ARGS" "$PCA_ARMS" "$CVS"\''],
            capture_output=True, text=True, timeout=60,
            env=dict(PATH=os.environ["PATH"], HOME="/nonexistent",
                     ARCH_ARGS="stray", CVS="cv2"))
        assert p.returncode == 0 and p.stdout == odd + "|none joint|", \
            (sh, line, p.stdout, p.stderr)


def test_root_rebuild_block_carries_the_runs_variants():
    # The rebuild of the arms a run leaves out checks their combos' PCA for
    # this run's variants only (_pca_refs: SIAMESE_VARIANTS), so its jobs
    # must table those variants, whatever the pasting shell exports: a
    # one-shot SIAMESE_VARIANTS_LIST (or USE_CELL_LINES=0) in front of the
    # script never reached that shell, whose jobs resolved config.conf's
    # four, unchecked, at targets no line of theirs set (2026-10-21). Arm
    # both: its pd:per_head combos at this run's targets, its pd:pooled ones
    # at 150.
    b08 = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8], "post_pca": 0.8}
    b150 = {"pca_variance": 0.8, "pca_dims": [0.8, 0.8], "post_pca": 150,
            "post_pca_dim": 150}
    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        sia = os.path.join(root, "siamese_sl")
        _seed_combos(sia, "both", {**{("", cv): b08
                                      for cv in ("cv1", "cv2", "cv3")},
                                   **{("var_pd_pooled__", cv): b150
                                      for cv in ("cv1", "cv2", "cv3")}})
        odd = "PCA arm(s) both: combos trained at PCA targets that no one set"
        # every variant: both's two variants disagree, and the NOTE says so
        rc, out, jobs = _root_run(tmp, "--skip-generate", PCA_ARMS="none")
        tail = out.split("All jobs submitted")[1]
        assert rc == 0 and odd in tail and _paste_def(
            "PCA_ARMS=none both", *_TARGETS0, *_VARIANTS0) in tail \
            and ("variants (pd:per_head pd:pooled bilinear:per_head "
                 "bilinear:pooled): the left-out arms are rebuilt over these")\
            in " ".join(tail.split()), tail
        # this run's variant only: both's pd:per_head combos are checked, and
        # every pasted job tables pd:per_head alone
        rc, out, jobs = _root_run(tmp, "--skip-generate", PCA_ARMS="none",
                                  SIAMESE_VARIANTS_LIST="pd:per_head")
        tail = out.split("All jobs submitted")[1]
        assert rc == 0 and odd not in tail and _paste_def(
            "PCA_ARMS=none both", *_TARGETS0,
            "SIAMESE_VARIANTS_LIST=pd:per_head", "USE_CELL_LINES=1",
            "CVS=") in tail \
            and "_c_both=$(_paste_env PCA_ARM=both sbatch --parsable" in tail, \
            tail
        # this run's CVS too: its shared rows cover those CVs only, and the
        # left-out arm's pasted compare grades them (2026-10-22: it graded
        # the pasting shell's, all three by default)
        rc2, out2, _ = _root_run(tmp, "--skip-generate", PCA_ARMS="none",
                                 CVS="cv1 cv3")
        tail2 = out2.split("All jobs submitted")[1]
        assert rc2 == 0 and _paste_def(
            "PCA_ARMS=none both", *_TARGETS0, *_VARIANTS0[:2],
            "CVS=cv1 cv3") in tail2, tail2
        # (the block of the pd:per_head run, pasted)
        lines = tail.splitlines()
        i = next(k for k, ln in enumerate(lines)
                 if ln.startswith("      _paste_env() {"))
        j = next(k for k, ln in enumerate(lines)
                 if ln.endswith(" --pca_arms_table"))
        block = "\n".join(ln[6:] for ln in lines[i:j + 1]) + "\n"
        sb = os.path.join(tmp, "bin", "sbatch")
        with open(sb) as f:
            text = f.read()
        assert text.endswith("echo $n\n"), text
        with open(sb, "w") as f:        # what each job would inherit
            f.write(text[:-len("echo $n\n")] + 'echo "$n|'
                    '${SIAMESE_VARIANTS_LIST-<unset>}|'
                    '${USE_CELL_LINES-<unset>}" >> "$SB_LOG.var"\necho $n\n')
        log, ctr = os.path.join(tmp, "paste.log"), os.path.join(tmp, "ctr")
        for sh in ["bash"] + [z for z in (shutil.which("zsh"),) if z]:
            for f in (log, log + ".env", log + ".var"):
                if os.path.exists(f):
                    os.remove(f)
            with open(ctr, "w") as f:
                f.write("2000\n")
            # a stray list and USE_CELL_LINES=0 in the pasting shell
            r = subprocess.run(
                [sh, "-s"], input=block, cwd=tmp, capture_output=True,
                text=True, timeout=60,
                env=dict(HOME=tmp, SB_LOG=log, SB_CTR=ctr,
                         SIAMESE_VARIANTS_LIST="bilinear:pooled",
                         USE_CELL_LINES="0",
                         PATH=os.path.join(tmp, "bin") + os.pathsep
                         + os.environ["PATH"]))
            assert r.returncode == 0, (sh, r.stdout, r.stderr)
            with open(log + ".var") as f:
                got = [ln.rstrip("\n").split("|")[1:] for ln in f]
            assert got == [["pd:per_head", "1"]] * 4, (sh, got)


def test_a_refused_exports_printed_command_keeps_the_runs_settings():
    # A one-shot PAIR_EXPORT_UNIVERSE=shared (the root's help) or variant
    # subset in front of either orchestrator never reached the shell the
    # command a refused export prints is pasted into: since 2026-10-21 that
    # command starts with them (printf %q, bash and zsh alike), --skip-eval's
    # PAIR_EXPORT_FT_SPLITS=none after them (the last assignment wins, as
    # the refused job got it), never ALLOW_LIVE_JOBS or PCA_ARMS. Pasted into
    # a fresh shell, the job gets what the refused submission would have:
    # the root's, submit_pipeline.sh's, and --eval-only's (its combo, its
    # fine-tuned sets).
    one_shot = {"PAIR_EXPORT_UNIVERSE": "shared", "PAIR_EXPORT_DTYPE": "float16",
                "PAIR_EXPORT_CVS": "cv1 cv3", "PAIR_EXPORT_FT_SPLITS": "cv1",
                "SIAMESE_VARIANTS_LIST": "pd:per_head", "ALLOW_LIVE_JOBS": "1"}
    want = {"PAIR_EXPORT_UNIVERSE": "shared", "PAIR_EXPORT_DTYPE": "float16",
            "PAIR_EXPORT_CVS": "cv1 cv3", "PAIR_EXPORT_FT_SPLITS": "none",
            "PAIR_EXPORT_MODEL_DIR": "", "SIAMESE_VARIANTS_LIST": "pd:per_head",
            "ALLOW_LIVE_JOBS": "<unset>"}

    def pasted(tmp, out, where):
        line, = [ln.strip() for ln in out.splitlines()
                 if ln.startswith("    ") and "sbatch" in ln
                 and ln.endswith(" slurm/" + _PAIR_SH)]
        b = os.path.join(tmp, "bin_pasted")
        os.makedirs(b, exist_ok=True)
        log = os.path.join(tmp, "pasted.env")
        with open(os.path.join(b, "sbatch"), "w") as f:
            f.write('#!/bin/bash\nfor v in ' + " ".join(want) + '; do '
                    'printf "%s=%s|" "$v" "${!v-<unset>}"; done > "$PASTED"\n'
                    'echo 1\n')
        os.chmod(os.path.join(b, "sbatch"), 0o755)
        got = []
        for sh in ["bash"] + [z for z in (shutil.which("zsh"),) if z]:
            e = {"PATH": b + os.pathsep + os.environ["PATH"], "HOME": tmp,
                 "PASTED": log}
            r = subprocess.run([sh, "-c", line], cwd=where, env=e,
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, (sh, line, r.stderr)
            with open(log) as f:
                got.append(dict(kv.split("=", 1)
                                for kv in f.read().split("|") if kv))
        assert r"PAIR_EXPORT_CVS=cv1\ cv3 " in line \
            and "ALLOW_LIVE_JOBS" not in line and "PCA_ARMS" not in line, line
        return line, got

    with tempfile.TemporaryDirectory() as tmp:
        root = _root_tree(tmp)
        rc, out, _ = _root_run(tmp, "--skip-generate", "--skip-eval",
                               PCA_ARMS="joint", SB_FAIL_ON="slurm/" + _PAIR_SH,
                               **one_shot)
        assert rc == 0 and "WARNING: could not submit the all-gene-pair" \
            in out, out[-2500:]
        line, got = pasted(tmp, out, os.path.join(root, "siamese_sl"))
        assert got and all(g == want for g in got), (line, got)
    with tempfile.TemporaryDirectory() as tmp:
        sia, _ = _stub_tree(tmp)
        _, out = _submit(tmp, "--skip-generate", "--skip-reset-env",
                         "--skip-eval", SB_PAIR_FAIL="1", **one_shot)
        line, got = pasted(tmp, out, sia)
        assert got and all(g == want for g in got), (line, got)
        # --eval-only: the evaluated combo, and the fine-tuned sets it asks
        d = os.path.join(sia, "results", "pca_joint", "x_cv3")
        os.makedirs(d)
        for f in ("config.json", "results.json"):
            with open(os.path.join(d, f), "w") as fh:
                fh.write("{}")
        _, out = _submit(tmp, "--eval-only", "results/pca_joint/x_cv3",
                         SB_PAIR_FAIL="1", PCA_ARMS=None, **one_shot)
        line, got = pasted(tmp, out, sia)
        assert got and all(g == dict(want, PAIR_EXPORT_FT_SPLITS="cv1",
                                     PAIR_EXPORT_MODEL_DIR="results/pca_joint/"
                                     "x_cv3") for g in got) \
            and " PCA_ARM=joint sbatch " in line, (line, got)


def test_root_refused_exports_print_each_arms_own_command():
    # Four arms queue 43 jobs (491 tasks) and the exports are the last 4: a
    # submit limit (MaxSubmitJobs counts every array task) refuses them
    # first. Each refusal is a WARNING with the command that queues that
    # arm's export by hand, to paste from siamese_sl/. Pasted into a fresh
    # bash or zsh (no PCA_ARM, no PCA_ARMS), each one IS its arm's refused
    # submission: that arm's PCA_ARM (the job exports the combos of that
    # arm's markers), its log names, the same afterany:...,singleton edge
    # (the stub hands out the same ids with and without the refusal: the
    # exports are the last submissions), a plain --export=ALL (_root_run
    # checks it), no token of its own, and with no eval queued
    # PAIR_EXPORT_FT_SPLITS=none. (No test ran the root's own command for
    # another arm than none until 2026-10-21: an _arm_env that printed none,
    # or swapped two arms, passed them all.)
    with tempfile.TemporaryDirectory() as tmp:
        _root_tree(tmp)
        for flags, splits in (((), ""), (("--skip-eval",), "none")):
            rc, out, queued = _root_run(tmp, "--force-generate", *flags,
                                        PCA_ARMS=None)
            assert rc == 0, out[-2000:]
            rc, out, jobs = _root_run(tmp, "--force-generate", *flags,
                                      PCA_ARMS=None,
                                      SB_FAIL_ON="slurm/" + _PAIR_SH)
            assert rc == 0 and list(_graph(jobs)) == [
                k for k in _graph(queued) if not k.startswith("export/")], \
                (rc, list(jobs), out[-2000:])
            cmds = re.findall(r"^WARNING: could not submit the all-gene-pair "
                              r"export of PCA arm (\w+);.*\n.*\n    (.+)$",
                              out, re.M)
            assert [a for a, _ in cmds] == list(_ARMS), (cmds, out[-3000:])
            for arm, cmd in cmds:
                name = f"export/{arm}"
                want = queued[name][1]       # the export's parents, by name
                logs = (("", "") if arm == "none" else tuple(
                    f"slurm/logs/pair_export_pca_{arm}_%j.{x}"
                    for x in ("out", "err")))
                for sh in ["bash"] + [z for z in (shutil.which("zsh"),) if z]:
                    rc, o2, got = _root_run(tmp, _line="cd siamese_sl && "
                                            + cmd, _sh=sh)
                    assert rc == 0 and list(_graph(got)) == [name], \
                        (sh, arm, cmd, list(got), o2)
                    assert got[name] == ("afterany", [
                        queued["_ids"][p] for p in want], False, "", True) \
                        and got["_logs"][name] == logs \
                        and got["_tokens"][name] == "" \
                        and got["_env"][name][0] == splits, \
                        (sh, arm, cmd, got[name], got["_logs"], got["_env"])


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n==== {passed} passed, {failed} failed ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
