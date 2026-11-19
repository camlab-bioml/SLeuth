#!/bin/bash
#SBATCH --job-name=siamese_gen_emb
#SBATCH --partition=gpu_Prosmn
#SBATCH --nodes=1
# Pinned to gpu3 (2026-10-01, by request). GPU_NODES
# (siamese_sl/slurm/config.conf) moves a whole orchestrated submission: both
# orchestrators pass it as --nodelist on the command line, which replaces this
# line (GPU_EXCLUDE only adds --exclude and cannot lift it).
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
# 14 days: ESM-2 generation on the full proteome can take many hours per GPU,
# and the first run also downloads large pretrained models (~5 GB).
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Generate All Embeddings
# ============================================================================
# Downloads source data and generates all .pt embedding files:
#   0. SynLethDB experimental SL pairs
#   1. ESM-2 protein embeddings (GPU-heavy)
#   2. GO anc2vec embeddings
#   3. All other embedding types (auto-download + extraction)
#
# This is the first step in the pipeline. Training jobs (benchmark and
# best-per-category array jobs) depend on this job completing successfully.
#
# Usage:
#   cd siamese_sl
#
#   # Preferred: use the pipeline orchestrator (handles all dependencies)
#   bash slurm/submit_pipeline.sh
#
#   # Manual: submit directly and chain training jobs
#   GEN=$(sbatch --parsable slurm/run_generate_embeddings.sh)
#   bash slurm/submit_pipeline.sh --after $GEN
#
# Exits 1 when ESM-2 fails, when any EMB_CATALOG type outside OPTIONAL_TYPES
# (run_generate_embeddings.conf) is still missing at the end (each is retried
# once), so the afterok training jobs never select over a smaller catalog, or
# when a catalog .pt is not a current build (embedding_versions.py: a failed
# builder falls back to a cache file this cleanup keeps and stamps it so).
# ============================================================================

set -e
# Every PCA arm reads the files this job builds: config.conf's job banner then
# names the run's arms, not the arm the job inherits from the orchestrator
# (none) and its results root (2026-10-21). A plain shell variable, not
# exported: it marks this job's own banner, nothing the job runs.
PCA_EVERY_ARM=1
source "$SLURM_SUBMIT_DIR/slurm/config.conf"
source "$SLURM_SUBMIT_DIR/slurm/run_generate_embeddings.conf"

echo "SIAMESE SL - EMBEDDING GENERATION"
echo ""

# ============================================================================
# Clean previous generated .pt files (keep downloads in CACHE_DIR intact)
# ============================================================================
echo "--- Cleaning previous generated embeddings ---"
CLEANED=0
for entry in "${ALL_EMBEDDINGS[@]}"; do
    IFS=':' read -r _etype efile _ename <<< "$entry"
    if [ -f "$DATA_DIR/$efile" ]; then
        rm -f "$DATA_DIR/$efile"
        CLEANED=$((CLEANED + 1))
    fi
    # (generate_embeddings.py saves under <file>.part and renames: a job
    # killed during a save leaves one behind)
    rm -f "$DATA_DIR/$efile.part"
done
# Remove known-legacy files that were renamed/removed from the catalog.
# Safe to re-run: each rm is guarded by -f and the file list is fixed.
LEGACY_FILES=(
    "all_genes_esm.pt"          # renamed to all_genes_esm2.pt
    "all_genes_esm.genes.txt"   # sidecar of the renamed ESM file
)
LEGACY_CLEANED=0
for lf in "${LEGACY_FILES[@]}"; do
    if [ -f "$DATA_DIR/$lf" ]; then
        rm -f "$DATA_DIR/$lf"
        echo "  Removed legacy: $lf"
        LEGACY_CLEANED=$((LEGACY_CLEANED + 1))
    fi
done
# Remove GO embeddings and the legacy NCBI sequence caches (no longer written).
# Kept on purpose: protein_sequences_canonical.json (keyed by its source files'
# checksums and the gene set, so a changed universe rebuilds it; deleting it
# would only re-run the flaky NCBI fallback) and
# ncbi_gene_descriptions_by_geneid.json (derived from the kept gene_info.gz).
rm -f "$GO_PATH"
rm -f "$CACHE_DIR/ncbi_gene_sequences.json"
rm -f "$CACHE_DIR/ncbi_protein_sequences.fasta"

# Remove computed per-embedding pickles so generate_embeddings.py regenerates
# them with the current code (e.g., the ComplEx real⊕imag fix needs a rerun).
# The ".pkl" extension is our convention for computed caches; downloaded
# sources use their own extensions (.gz, .obo, .tsv, .fasta, .txt, .pickle,
# .npz, .json) and are preserved.
PKL_CLEANED=$(find "$CACHE_DIR" -maxdepth 1 -name '*_gene_embeddings.pkl' -print -delete 2>/dev/null | wc -l | tr -d ' ')
# ...and their build stamps (generate_embeddings.py _write_build_stamp), which
# would otherwise pile up as orphans.
find "$CACHE_DIR" -maxdepth 1 -name '*_gene_embeddings.pkl.generator_version' -delete 2>/dev/null || true

echo "  Removed $CLEANED old .pt files, $LEGACY_CLEANED legacy files, $PKL_CLEANED cached .pkl embeddings, + NCBI sequence caches"
echo ""

# ============================================================================
# STEP 0: Download SynLethDB SL pairs
# ============================================================================
echo "--- STEP 0: SynLethDB Experimental SL Pairs ---"
sleep 10

# Skip only when BOTH the pairs .txt AND its .sources.tsv sidecar exist. The
# sidecar carries the per-cell-line provenance that USE_CELL_LINES=1 (now the
# default) needs; a .txt left without its sidecar would make every cell-mode
# train.py call crash at _setup_multilabel. Regenerate if either is missing.
SL_SIDECAR="${SL_PATH%.txt}.sources.tsv"
if [ -f "$SL_PATH" ] && [ -f "$SL_SIDECAR" ]; then
    echo "Already exist, skipping"
    wc -l "$SL_PATH" | awk '{print "  " $1 " SL pairs"}'
else
    echo "Downloading SynLethDB and filtering to experimental SL pairs..."
    $PYTHON_PATH download_synlethdb.py \
        --output "$SL_PATH" \
        --cache_dir "$CACHE_DIR"
    { [ -f "$SL_PATH" ] && [ -f "$SL_SIDECAR" ]; } \
        || { echo "FAILED (missing pairs .txt or .sources.tsv sidecar)"; exit 1; }
fi

# Experimentally screened NON-SL pairs + provenance sidecar. These are REQUIRED
# in every mode (STEP 0b and cell-line training) and the download is FATAL on
# failure — cell mode has no random-negative fallback. There is no
# random-negative benchmark any more: every train.py run needs the shared
# folds, and USE_CELL_LINES=0 takes its labels from them (NEG_PATH is then
# ignored). STEP 0b below builds the gene universe from HGNC protein-coding
# genes UNION every gene in a positive or negative pair, so it needs this file
# in every mode.
# Skip only when BOTH the .txt and its sidecar exist; retry a transient
# (network/Drive) failure.
NEG_SIDECAR="${NEG_PATH_FILE%.txt}.sources.tsv"
if [ -f "$NEG_PATH_FILE" ] && [ -f "$NEG_SIDECAR" ]; then
    echo "Negatives already exist, skipping"
    wc -l "$NEG_PATH_FILE" | awk '{print "  " $1 " non-SL pairs"}'
else
    echo "Downloading SynLethDB experimental NON-SL pairs (negatives)..."
    for attempt in 1 2 3; do
        if $PYTHON_PATH download_synlethdb.py \
            --relation NONSL \
            --output "$NEG_PATH_FILE" \
            --cache_dir "$CACHE_DIR"; then
            break
        fi
        echo "WARNING: negatives download attempt $attempt/3 returned non-zero"
        sleep $((attempt * 15))
    done
    if [ ! -f "$NEG_PATH_FILE" ] || [ ! -f "$NEG_SIDECAR" ]; then
        echo "ERROR: negatives pairs+sidecar download failed after retries;" \
             "STEP 0b (gene universe) and cell-line mode both need them."
        exit 1
    fi
fi
echo ""

# ============================================================================
# STEP 0b: Canonical gene universe (HGNC protein-coding UNION all pair genes)
# ============================================================================
# Every embedding is aligned to this list. Seeding it from SL_PATH alone gave
# gene_order = the 5,257 genes in >=1 POSITIVE pair, leaving the 3,587
# negative-only genes with no embedding — see config.conf GENE_UNIVERSE.
echo "--- STEP 0b: Gene universe ---"
$PYTHON_PATH - "$SL_PATH" "$NEG_PATH_FILE" "$GENE_UNIVERSE" "$DATA_DIR" <<'PY'
import sys
from pathlib import Path
pos_p, neg_p, out_p, data_dir = sys.argv[1:5]

# (a) Benchmark genes — every gene in a given SL or non-SL pair. This set is
# MANDATORY and must be fully contained in the universe: sl_comparison/folds
# indexes into it, so a missing gene silently drops pairs.
bench = set()
for p in (pos_p, neg_p):
    if not Path(p).exists():
        sys.exit(f"FATAL: {p} missing — cannot build the gene universe")
    for line in Path(p).read_text().splitlines():
        for tok in line.split():
            if tok.isdigit():
                bench.add(tok)
if not bench:
    sys.exit("FATAL: no genes found in the pair files")

# (b) All protein-coding genes, so any gene pair is scoreable at inference and
# "has an embedding" carries no information about the label.
hgnc = next((c for c in (Path(data_dir) / "cache" / "hgnc_complete_set.txt",
                         Path(data_dir) / "embeddings_cache" /
                         "hgnc_complete_set.txt") if c.exists()), None)
if hgnc is None:
    sys.exit("FATAL: hgnc_complete_set.txt not found under "
             f"{data_dir}/cache or {data_dir}/embeddings_cache. Refusing to "
             "silently fall back to the benchmark-only universe — that is the "
             "confound this step exists to remove.")
coding, hdr = set(), None
for i, line in enumerate(Path(hgnc).read_text().splitlines()):
    f = line.split("\t")
    if i == 0:
        hdr = {n: j for j, n in enumerate(f)}
        continue
    ei = f[hdr["entrez_id"]].strip() if len(f) > hdr["entrez_id"] else ""
    lg = f[hdr["locus_group"]].strip() if len(f) > hdr["locus_group"] else ""
    if ei.isdigit() and lg == "protein-coding gene":
        coding.add(ei)
if len(coding) < 15000:
    sys.exit(f"FATAL: only {len(coding)} protein-coding genes parsed from "
             f"{hgnc} (expected ~19k) — the file looks wrong or truncated.")

genes = coding | bench
Path(out_p).write_text("\n".join(sorted(genes, key=int)) + "\n")
print(f"  protein-coding: {len(coding):,} | benchmark: {len(bench):,} "
      f"(non-coding kept: {len(bench - coding):,})")
print(f"  gene universe: {len(genes):,} genes -> {out_p}")
PY
[ -s "$GENE_UNIVERSE" ] || { echo "FAILED to build $GENE_UNIVERSE"; exit 1; }
echo ""

# ============================================================================
# STEP 1: Generate ESM Embeddings
# ============================================================================
echo "--- STEP 1: ESM Embeddings ---"
sleep 10

echo "Generating ESM embeddings with Pool PaRTI..."
$PYTHON_PATH generate_all_genes_esm.py \
    --output "$ESM_PATH" \
    --gene_list "$GENE_UNIVERSE" \
    --cache_dir "$CACHE_DIR" \
    --pooling pool_parti \
    --batch_size 8 \
    --device cuda:0
[ -f "$ESM_PATH" ] || { echo "FAILED"; exit 1; }
echo ""

# ============================================================================
# STEP 2: Generate GO Embeddings
# ============================================================================
echo "--- STEP 2: GO Embeddings ---"
sleep 10

# Auto-download human GAF if missing or empty (same URL used by
# generate_embeddings.py for kg_complex; we do it here so GO anc2vec
# doesn't depend on kg_complex having run first).
if [ ! -s "$GAF_PATH" ]; then
    echo "  GAF not cached, downloading to $GAF_PATH..."
    mkdir -p "$(dirname "$GAF_PATH")"
    set +e
    curl -fsSL --retry 3 -o "$GAF_PATH" \
        "http://geneontology.org/gene-associations/goa_human.gaf.gz"
    status=$?
    set -e
    if [ $status -ne 0 ]; then
        rm -f "$GAF_PATH"  # clean up partial download
        echo "WARNING: GAF download failed (exit $status), skipping GO"
    fi
fi

if [ -s "$GAF_PATH" ]; then
    # GO failure does not stop the job here (set +e), unlike ESM above, which
    # every later step needs: the other modalities still generate, and the
    # catalog check at the end fails the job for the missing GO file.
    set +e
    $PYTHON_PATH generate_go_esm_embeddings.py \
        --go_only \
        --esm_embeddings "$ESM_PATH" \
        --gaf "$GAF_PATH" \
        --output "$GO_PATH"
    set -e
    [ -f "$GO_PATH" ] || echo "WARNING: GO generation failed, continuing"
else
    echo "WARNING: GAF not present at $GAF_PATH, skipping GO"
fi
echo ""

# ============================================================================
# STEP 3: Generate All Precomputed Embeddings
# ============================================================================
echo "--- STEP 3: Generate All Embeddings ---"
sleep 10

for etype in "${PRECOMPUTED_TYPES[@]}"; do
    OUTPUT="$DATA_DIR/all_genes_${etype}.pt"
    # One retry for a required type (a transient HF / NCBI / network error);
    # an OPTIONAL_TYPES entry that is not available fails the same way twice.
    TRIES=2
    case " ${OPTIONAL_TYPES[*]} " in *" $etype "*) TRIES=1 ;; esac
    for ((attempt = 1; attempt <= TRIES; attempt++)); do
        echo "[$etype] Generating (attempt $attempt/$TRIES)..."
        set +e
        $PYTHON_PATH generate_embeddings.py \
            --type "$etype" \
            --gene_list "$ESM_PATH" \
            --output "$OUTPUT" \
            --cache_dir "$CACHE_DIR"
        set -e
        [ -f "$OUTPUT" ] && break
        # A failed attempt can leave a truncated or unstamped pkl, which the
        # next attempt would reuse instead of rebuilding: drop it. A pkl with
        # its build stamp is a finished build of this job and is kept. Also
        # drop a half-written output. Never fatal (set -e is on here).
        [ -f "$CACHE_DIR/${etype}_gene_embeddings.pkl.generator_version" ] \
            || rm -f "$CACHE_DIR/${etype}_gene_embeddings.pkl" || true
        rm -f "$OUTPUT.part" || true
        if [ "$attempt" -lt "$TRIES" ]; then sleep 60; fi
    done
    [ -f "$OUTPUT" ] && echo "[$etype] Done" || echo "[$etype] Skipped (not available)"
done
echo ""

# ============================================================================
# Summary: list available embeddings
# ============================================================================
echo "--- Available Embeddings ---"
for entry in "${ALL_EMBEDDINGS[@]}"; do
    IFS=':' read -r etype efile ename <<< "$entry"
    if [ -f "$DATA_DIR/$efile" ]; then
        echo "  [FOUND] $ename"
    else
        echo "  [SKIP]  $ename"
    fi
done

# A missing catalog modality fails the job (after every other type has had its
# turn): Step 1 would SKIP it with exit 0 and the combo would silently select
# over a smaller catalog. OPTIONAL_TYPES (run_generate_embeddings.conf) are the
# accepted absences.
source "$SLURM_SUBMIT_DIR/slurm/run_best_per_category.conf"   # EMB_CATALOG
REQUIRED_MISSING=()
for entry in "${EMB_CATALOG[@]}"; do
    IFS=':' read -r etype efile _ecat <<< "$entry"
    [ -f "$DATA_DIR/$efile" ] && continue
    case " ${OPTIONAL_TYPES[*]} " in *" $etype "*) continue ;; esac
    REQUIRED_MISSING+=("$etype")
done
if [ ${#REQUIRED_MISSING[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: required catalog embedding(s) not generated: ${REQUIRED_MISSING[*]}"
    echo "  (see the [type] lines above). The combo would select over a smaller"
    echo "  catalog than the paper's; add a type to OPTIONAL_TYPES in"
    echo "  slurm/run_generate_embeddings.conf to accept its absence."
    exit 1
fi

# Existence is not provenance: a builder that fails falls back to a cache file
# the cleanup above keeps (an alt name such as GenePert's
# gene_to_geneformer_embed.pkl, or a cache_dir/<type>/ copy) and stamps the .pt
# non-current, and with --force-generate nothing downstream checks the stamp.
# Same check as the root orchestrator's generation gate (OPTIONAL_TYPES may be
# absent; present files must be current, on the gene universe, in ESM-2 order).
echo ""
echo "--- Provenance check (embedding_versions.py) ---"
if ! $PYTHON_PATH "$SLURM_SUBMIT_DIR/embedding_versions.py" \
        --data_dir "$DATA_DIR" --gene_universe "$GENE_UNIVERSE"; then
    echo ""
    echo "ERROR: a catalog embedding is not a build of the current generator (a"
    echo "  failed build fell back to a cached file; see the WARNING lines above)."
    exit 1
fi

echo ""
echo "Embedding generation completed at: $(date)"
