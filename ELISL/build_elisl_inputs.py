#!/usr/bin/env python3
"""Build ELISL's context-free inputs (SeqVec + STRING node2vec) for the shared genes.

ELISL (Tepeli et al., Bioinformatics 2024) learns from two context-free
sources: amino-acid sequence (SeqVec, 1,024 dims per protein) and PPI
(node2vec, 64 dims, on STRING v11.0). Its feature for a gene pair is
|e(gene1) - e(gene2)| per source, or -1 everywhere when either gene has no
embedding. This script produces exactly the files the authors' README builds,
from the same sources, then aligns them to sl_comparison/folds/genes.txt.

Steps (each one a subcommand; `gpu` runs fasta, seqvec, node2vec, features):

  download   LOGIN NODE (internet). Into data/elisl/raw/:
               uniprot_reviewed_9606.fasta   UniProtKB reviewed, organism 9606
               9606.protein.links.full.v11.0.txt.gz, human.name_2_string.tsv.gz,
               human.uniprot_2_string.2018.tsv.gz   (STRING, the README's URLs)
             and the SeqVec ELMo weights into data/elisl/seqvec_model/ (the
             seqvec package's own download location and URLs), with a
             downloads.json manifest (URL, size, sha256).
  fasta      Rewrites the FASTA with accession-only headers (seqvec's `--id 0`,
             as the authors' README runs it, keys each embedding by the first
             '|' field: on a raw UniProt header that is 'sp') and writes
             uniprot_reviewed_9606_mappings.csv (gene_name, uniprot_id,
             uniprot_name), the table sequence_features.get_mappings reads.
             The authors ship no code for either file.
  seqvec     GPU. The authors' command,
               seqvec -i <fasta> -o uniprot_1024_embeddings.npz --id 0 --protein
             run with --seqvec_bin (a separate Python 3.7 environment with
             seqvec==0.4.1, the authors' pin: allennlp 0.9, torch < 2,
             numpy < 2; it cannot be installed in the cluster venv).
  node2vec   CPU. ppi_embedding.main(): STRING edges with experiments > 0 OR
             database > 0 ('ec', cutoff 0.0), undirected, node2vec 64 dims,
             walk length 30, 200 walks, p = q = 1, 4 workers, window 10,
             min_count 1, batch_words 4 -> word2vec text file.
  features   Maps every gene of genes.txt (Entrez) to its symbol
             (data/gene_id_mapping.tsv), then through the authors' name-based
             mapping chains (sequence: gene name -> UniProt; PPI: STRING display
             name, then UniProt entry name -> STRING id) to its vectors, and
             writes data/elisl/gene_features.npz + data/elisl/meta.json
             (coverage, unmapped genes, sha256 of every source and of this code).
             sl_comparison/slurm/check_shared_inputs.py --models ELISL checks it.

Usage (repo root; nothing here runs during development, see ELISL/README.md):
    python ELISL/build_elisl_inputs.py download                    # login node
    sbatch ELISL/slurm/build_elisl_inputs.sh                       # GPU: the rest
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from elisl_orig import features as F  # noqa: E402

OUT = REPO / "data" / "elisl"
RAW = OUT / "raw"
SEQVEC_MODEL = OUT / "seqvec_model"
FEATURES = OUT / "gene_features.npz"
META = OUT / "meta.json"

UNIPROT_FASTA = "uniprot_reviewed_9606.fasta"
UNIPROT_URL = ("https://rest.uniprot.org/uniprotkb/stream?format=fasta&"
               "query=%28reviewed%3Atrue%29%20AND%20%28organism_id%3A9606%29")
STRING_FILES = {
    "9606.protein.links.full.v11.0.txt.gz":
    "https://stringdb-static.org/download/protein.links.full.v11.0/"
    "9606.protein.links.full.v11.0.txt.gz",
    "human.name_2_string.tsv.gz":
    "https://version-11-0.string-db.org/mapping_files/STRING_display_names/"
    "human.name_2_string.tsv.gz",
    "human.uniprot_2_string.2018.tsv.gz":
    "https://version-11-0.string-db.org/mapping_files/uniprot/"
    "human.uniprot_2_string.2018.tsv.gz",
}
# The two mapping files left string-db.org/mapping_files (404 as of
# 2026-11-13); STRING's v11.0 archive host serves them (gzip, 162,720 and
# 306,716 bytes). The links file is still on stringdb-static.org.
# seqvec 0.4.1 seqvec.get_elmo_model downloads options.json / weights.hdf5
# from rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec, which now
# 404s (2026-11-13). The same two files are mirrored by bio_embeddings
# (options.json identical; weights.hdf5 374,434,776 bytes); downloads.json
# records each file's sha256.
SEQVEC_REPO = ("http://data.bioembeddings.com/public/embeddings/"
               "embedding_models/seqvec")
SEQVEC_FILES = ("options.json", "weights.hdf5")

ACC_FASTA = OUT / "uniprot_reviewed_9606.acc.fasta"
SEQ_MAPPINGS = OUT / "uniprot_reviewed_9606_mappings.csv"
SEQVEC_NPZ = OUT / "uniprot_1024_embeddings.npz"
NODE2VEC_TXT = OUT / ("embs_ppi_ec=0.0_dim=64_p=1_q=1_wl=30_nw=200_ws=4_"
                      "wd=10_mc=1_bw=4")
GENE_ID_MAPPING = REPO / "data" / "gene_id_mapping.tsv"
DEFAULT_GENES = REPO / "sl_comparison" / "folds" / "genes.txt"

# Code whose bytes decide the features (repo-relative). check_shared_inputs
# refuses a build whose recorded hash of any of them differs from the file.
CODE_FILES = ("ELISL/build_elisl_inputs.py", "ELISL/elisl_orig/features.py")
SOURCES = ("Sequence_1024", "PPI_ec")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  {url}\n    -> {rel(dest)}")
    # data.bioembeddings.com (the SeqVec weights) answers 406 Not Acceptable
    # to a python-urllib or browser User-Agent and serves wget / curl ones
    # (2026-11-13); UniProt and STRING serve either.
    req = urllib.request.Request(url, headers={"User-Agent": "Wget/1.21"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)
    return {"url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)}


def cmd_download(args):
    manifest = {}
    jobs = [(UNIPROT_URL, RAW / UNIPROT_FASTA)] + \
        [(u, RAW / n) for n, u in STRING_FILES.items()] + \
        [(f"{SEQVEC_REPO}/{n}", SEQVEC_MODEL / n) for n in SEQVEC_FILES]
    for url, dest in jobs:
        if dest.exists() and not args.force:
            print(f"  present: {rel(dest)} (--force re-downloads)")
            manifest[rel(dest)] = {"url": url, "bytes": dest.stat().st_size,
                                   "sha256": sha256(dest)}
            continue
        manifest[rel(dest)] = fetch(url, dest)
    (OUT / "downloads.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {rel(OUT / 'downloads.json')}")


def parse_uniprot_fasta(path):
    """[(accession, entry_name_prefix, gene_name or None, sequence)].

    UniProt header: >sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=...
    OX=9606 GN=TP53 PE=1 SV=4. uniprot_name is the entry name before '_',
    the key get_PPI_uniprot_maps / get_uniprot_extra_mapping also cut to.
    """
    out, head, seq = [], None, []

    def flush():
        if head is None:
            return
        fields = head[1:].split(" ", 1)
        parts = fields[0].split("|")
        acc, entry = parts[1], parts[2]
        gn = None
        for tok in (fields[1] if len(fields) > 1 else "").split(" "):
            if tok.startswith("GN="):
                gn = tok[3:]
        out.append((acc, entry.split("_", 1)[0], gn, "".join(seq)))

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                head, seq = line, []
            elif line:
                seq.append(line.strip())
    flush()
    return out


def cmd_fasta(args):
    import pandas as pd
    recs = parse_uniprot_fasta(RAW / UNIPROT_FASTA)
    with open(ACC_FASTA, "w") as f:
        for acc, _, _, s in recs:
            f.write(f">{acc}\n{s}\n")
    pd.DataFrame([(gn, acc, un) for acc, un, gn, _ in recs],
                 columns=["gene_name", "uniprot_id", "uniprot_name"]
                 ).to_csv(SEQ_MAPPINGS, index=False)
    print(f"{len(recs):,} reviewed human proteins "
          f"({sum(r[2] is None for r in recs)} without GN=) -> "
          f"{rel(ACC_FASTA)}, {rel(SEQ_MAPPINGS)}")


def cmd_seqvec(args):
    if not args.seqvec_bin:
        raise SystemExit("--seqvec_bin (or SEQVEC_BIN) is required: the "
                         "seqvec==0.4.1 console script of its own Python 3.7 "
                         "environment (ELISL/README.md, Inputs)")
    for n in SEQVEC_FILES:
        if not (SEQVEC_MODEL / n).is_file():
            raise SystemExit(f"{rel(SEQVEC_MODEL / n)} missing: run "
                             f"`download` on the login node first (the GPU "
                             f"node may have no internet)")
    cmd = [args.seqvec_bin, "-i", str(ACC_FASTA), "-o", str(SEQVEC_NPZ),
           "--model", str(SEQVEC_MODEL), "--id", "0", "--protein"]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_node2vec(args):
    import networkx as nx
    links = RAW / "9606.protein.links.full.v11.0.txt.gz"
    cut_off_df, _ = F.get_PPI_data(links, cutoff_list=[0.0], source='ec')
    g = nx.from_pandas_edgelist(cut_off_df['0.0'], 'protein1',
                                'protein2').to_undirected()
    print(f"  STRING 'ec' graph: {g.number_of_nodes():,} proteins, "
          f"{g.number_of_edges():,} edges")
    opts = F.get_node2vec_emb(g, str(NODE2VEC_TXT))
    (OUT / "node2vec_opts.json").write_text(json.dumps(
        {"opts": opts, "nodes": g.number_of_nodes(),
         "edges": g.number_of_edges()}, indent=2))


def gunzip_to(src, dest):
    # Always (2026-11-13): deciding by mtime kept a stale extract when a
    # replacement .gz arrived with an older preserved timestamp (SFTP), and
    # meta.json hashes the extract, so check_shared_inputs passed on it.
    # Written to a temporary name first, so an interrupted run leaves no
    # truncated extract.
    tmp = dest.with_name(dest.name + ".tmp")
    with gzip.open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    os.replace(tmp, dest)
    return dest


def entrez_symbols(genes, mapping=GENE_ID_MAPPING):
    """Entrez id -> symbol from data/gene_id_mapping.tsv (entrez_id, symbol);
    None where it has none."""
    m = {}
    with open(mapping) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                m[p[0]] = p[1]
    return [m.get(g) for g in genes]


def build_features(genes, symbols, seq_emb, seq_maps, ppi_emb, ppi_maps):
    """Per gene of `genes`: the authors' vectors and missing masks. A gene with
    no symbol is looked up under its Entrez id (it then maps to nothing)."""
    names = [s if s else g for g, s in zip(genes, symbols)]
    seq, seq_miss, seq_nf = F.gene_vectors(names, seq_maps, seq_emb)
    ppi, ppi_miss, ppi_nf = F.gene_vectors(names, ppi_maps, ppi_emb)
    return {"seq": seq, "seq_missing": seq_miss, "ppi": ppi,
            "ppi_missing": ppi_miss, "names": np.asarray(names)}, \
        seq_nf, ppi_nf


def cmd_features(args):
    genes = [ln.strip() for ln in open(args.genes) if ln.strip()]
    symbols = entrez_symbols(genes)
    seq_emb = F.load_seq_embeddings(SEQVEC_NPZ)
    seq_maps = F.get_seq_mappings(SEQ_MAPPINGS, args.seq_extra_mapping)
    name2string = gunzip_to(RAW / "human.name_2_string.tsv.gz",
                            OUT / "human.name_2_string.tsv")
    uniprot2string = gunzip_to(RAW / "human.uniprot_2_string.2018.tsv.gz",
                               OUT / "human.uniprot_2_string.2018.tsv")
    ppi_maps = F.get_ppi_mappings(name2string, uniprot2string,
                                  args.ppi_hand_mapping, args.ppi_extra_mapping)
    ppi_emb = F.get_embs_dict(F.load_embs_from_raw_loc(str(NODE2VEC_TXT)))
    arrays, seq_nf, ppi_nf = build_features(genes, symbols, seq_emb, seq_maps,
                                            ppi_emb, ppi_maps)
    np.savez(FEATURES, genes=np.asarray(genes), **arrays)
    n = len(genes)
    meta = {
        "model": "ELISL",
        "sources": list(SOURCES),
        "context_free": True,
        "genes_txt": rel(args.genes),
        "genes_sha256": sha256(args.genes),
        "n_genes": n,
        "no_symbol": [g for g, s in zip(genes, symbols) if not s],
        "coverage": {
            "Sequence_1024": int(n - arrays["seq_missing"].sum()),
            "PPI_ec": int(n - arrays["ppi_missing"].sum()),
            "both": int((~arrays["seq_missing"] & ~arrays["ppi_missing"]).sum()),
            "neither": int((arrays["seq_missing"] & arrays["ppi_missing"]).sum()),
        },
        "unmapped": {"Sequence_1024": seq_nf, "PPI_ec": ppi_nf},
        "dims": {"Sequence_1024": int(arrays["seq"].shape[1]),
                 "PPI_ec": int(arrays["ppi"].shape[1])},
        "optional_mappings": {
            "sequences/uniprot_extra_mapping.tab": args.seq_extra_mapping,
            "PPI/STRING/hand_mapping.tsv": args.ppi_hand_mapping,
            "PPI/STRING/uniprot_extra_mapping.tab": args.ppi_extra_mapping},
        "input_sha256": {rel(p): sha256(p) for p in (
            SEQVEC_NPZ, SEQ_MAPPINGS, NODE2VEC_TXT, name2string,
            uniprot2string, GENE_ID_MAPPING) + tuple(
            Path(x) for x in (args.seq_extra_mapping, args.ppi_hand_mapping,
                              args.ppi_extra_mapping) if x)},
        "code_sha256": {c: sha256(REPO / c) for c in CODE_FILES},
        "gene_features_sha256": sha256(FEATURES),
    }
    dl = OUT / "downloads.json"
    if dl.exists():
        meta["downloads"] = json.loads(dl.read_text())
    META.write_text(json.dumps(meta, indent=2))   # last: the build is whole
    c = meta["coverage"]
    print(f"{n:,} genes: SeqVec {c['Sequence_1024']:,}, PPI {c['PPI_ec']:,}, "
          f"both {c['both']:,}, neither {c['neither']:,} -> {rel(FEATURES)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["download", "fasta", "seqvec", "node2vec",
                                    "features", "gpu"])
    p.add_argument("--force", action="store_true",
                   help="[download] fetch files that are already present")
    p.add_argument("--seqvec_bin", default=os.environ.get("SEQVEC_BIN"),
                   help="[seqvec] seqvec==0.4.1 console script")
    p.add_argument("--genes", type=Path, default=DEFAULT_GENES,
                   help="[features] gene universe (default the shared folds')")
    p.add_argument("--seq_extra_mapping", default=None,
                   help="[features] the authors' sequences/"
                   "uniprot_extra_mapping.tab, if you have their data release")
    p.add_argument("--ppi_hand_mapping", default=None,
                   help="[features] the authors' PPI/STRING/hand_mapping.tsv")
    p.add_argument("--ppi_extra_mapping", default=None,
                   help="[features] the authors' PPI/STRING/"
                   "uniprot_extra_mapping.tab")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    steps = {"download": [cmd_download], "fasta": [cmd_fasta],
             "seqvec": [cmd_seqvec], "node2vec": [cmd_node2vec],
             "features": [cmd_features],
             "gpu": [cmd_fasta, cmd_seqvec, cmd_node2vec, cmd_features]}
    for fn in steps[args.step]:
        print(f"--- {fn.__name__[4:]} ---")
        fn(args)


if __name__ == "__main__":
    main()
