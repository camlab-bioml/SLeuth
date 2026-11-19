#!/usr/bin/env python3
"""Build the SynLethKG input of the ORIGINAL SLGNN for the shared gene universe.

SLGNN (Zhu et al., Bioinformatics 2023; github.com/zy972014452/SLGNN) runs its
factor-aware message passing over SynLethKG, the SynLethDB 2.0 knowledge graph.
Its own input is data/SL/raw/kg2id.txt in that repository: (head, relation,
tail) triples over 54,012 entities and 24 relation types. This script

  1. downloads kg2id.txt and the files that name its ids (entity2id.txt,
     dbid2name.csv, kg_triplet.csv) at a PINNED commit and checks their sha256;
  2. proves the relation vocabulary of kg2id.txt by joining every triple back
     to kg_triplet.csv, where relations are named, and refuses to write
     anything if a single triple carries a label relation (SL_GsG, SR_GsrG,
     NONSL_GnsG): an SL or non-SL edge inside the KG would hand the model
     val/test answers;
  3. MEASURES, and records in meta.json (relation_checks.label_keyed_deletion),
     what else the shipped kg2id.txt leaves out. It is not kg_triplet.csv minus
     the label relations: it also lacks every INTERACTS/REGULATES/COVARIES edge
     between two genes that SynLethKG labels SL (1,251 rows), so no SL pair
     keeps a direct gene-gene edge while SR and non-SL pairs that are not also
     SL keep theirs. That edit is keyed on the SL label. kg2id.txt is still
     written unchanged: it is the original's input, kept as shipped on
     purpose (README, "SynLethKG");
  4. maps every KG gene entity to an NCBI Entrez ID (the shared folds' gene
     key), from local, tracked files only: SynLethDB's own name/ID export
     (data/cache/Human*.SL.detailed.tsv), then HGNC, then NCBI gene_info
     (MAPPING_FILES; meta.json records their sha256, and this script's own,
     as mapping_sha256);
  5. writes <out_dir>/ kg_triples.npy, entity_entrez.tsv, relations.tsv,
     meta.json and a .gitignore that keeps the directory out of git.

Genes of the shared universe with no KG entity are NOT handled here: the
trainer gives each one a fresh isolated entity (see load_kg_for_genes).

Usage (from SLGNN/; ~100 MB download, ~1 min):
    python build_synlethkg.py --out_dir ../data/synlethkg
    # or from an existing clone of the SLGNN repository
    python build_synlethkg.py --src /path/to/SLGNN/data/SL/raw \\
        --out_dir ../data/synlethkg
"""

import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent

# The SLGNN repository at the commit this port was checked against.
SOURCE_COMMIT = "c56e23fbd20e825c5571ef3e94377944c669ee80"
SOURCE_URL = ("https://raw.githubusercontent.com/zy972014452/SLGNN/"
              f"{SOURCE_COMMIT}/data/SL/raw/")
SOURCE_FILES = {
    "kg2id.txt":
    "51e1f5bf2ffe904191e6f20491fad24e20fad6404cc6342b13b91165028b19fa",
    "entity2id.txt":
    "bec5f2c5c6abd61f4630dddb305a6bc737a28315cb8472acd3e63d1d389450c6",
    "dbid2name.csv":
    "9dcb8ef22c38b0549b32383c2168226952ee106d50d61ea5c5868f6ad3320713",
    "kg_triplet.csv":
    "fb61f918ebe33cba142661764d257b01df43464d3342cf87e7a89e984183243e",
}

# kg2id.txt relation id -> (kg_triplet.csv type name, head is gene, tail is
# gene). The order is the repository's relation_name.txt; step 2 checks every
# triple against the names, so a wrong row here stops the build.
RELATIONS = [
    ("PARTICIPATES_GpCC", True, False),
    ("PARTICIPATES_GpBP", True, False),
    ("EXPRESSES_AeG", False, True),
    ("REGULATES_GrG", True, True),
    ("INTERACTS_GiG", True, True),
    ("ASSOCIATES_DaG", False, True),
    ("PARTICIPATES_GpMF", True, False),
    ("COVARIES_GcG", True, True),
    ("PARTICIPATES_GpPW", True, False),
    ("UPREGULATES_DuG", False, True),
    ("CAUSES_CcSE", False, False),
    ("BINDS_CbG", False, True),
    ("UPREGULATES_AuG", False, True),
    ("PRESENTS_DpS", False, False),
    ("LOCALIZES_DlA", False, False),
    ("TREATS_CtD", False, False),
    ("RESEMBLES_DrD", False, False),
    ("DOWNREGULATES_DdG", False, True),
    ("UPREGULATES_CuG", False, True),
    ("DOWNREGULATES_CdG", False, True),
    ("RESEMBLES_CrC", False, False),
    ("INCLUDES_PCiC", False, False),
    ("PALLIATES_CpD", False, False),
    ("DOWNREGULATES_AdG", False, True),
]
# Gene-gene LABEL relations of SynLethKG: synthetic lethality, synthetic
# rescue and screened non-SL. None may reach the model.
LABEL_RELATIONS = ("SL_GsG", "SR_GsrG", "NONSL_GnsG")

# name -> Entrez resolution, most trusted first (recorded per entity).
METHODS = ("synlethdb", "hgnc_symbol", "ncbi_symbol", "hgnc_prev",
           "hgnc_alias", "ncbi_synonym")
# The gene-identity inputs of map_gene_entities, relative to REPO_DIR. Another
# copy of any of them can attach a KG gene to another Entrez id, so meta.json
# records their sha256 (mapping_sha256, keyed by these paths).
MAPPING_FILES = {
    "synlethdb_sl": "data/cache/Human.SL.detailed.tsv",
    "synlethdb_nonsl": "data/cache/Human.non.SL.detailed.tsv",
    "hgnc": "data/cache/hgnc_complete_set.txt",
    "ncbi_gene_info": "data/embeddings_cache/Homo_sapiens.gene_info.gz",
}
# This script's own rules move KG genes onto other Entrez ids with no
# MAPPING_FILES byte changing (METHODS order, resolve_entrez's collision
# precedence, the SOURCE_FILES pins), so mapping_sha256 records its hash too --
# as NSF4SL/build_transe_features.py does. Repo-relative like the rest, so
# check_shared_inputs.py's `root / key` finds it with no change there.
BUILDER_REL = "SLGNN/build_synlethkg.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mapping_sha256(repo_dir):
    """{repo-relative path: sha256} of the files entity_entrez.tsv is mapped
    from (MAPPING_FILES), plus this script itself. Recorded for provenance: a
    hash that differs from the current file means the build was mapped from
    another copy of that input, or by other mapping code."""
    out = {
        rel: sha256(Path(repo_dir) / rel)
        for rel in MAPPING_FILES.values()
    }
    # From __file__, not repo_dir: a build always records the code that ran.
    out[BUILDER_REL] = sha256(Path(__file__).resolve())
    return out


def fetch_sources(src, raw_dir):
    """Copy/download the pinned source files into raw_dir; verify sha256."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in SOURCE_FILES.items():
        dst = raw_dir / name
        if not dst.exists():
            if src:
                dst.write_bytes((Path(src) / name).read_bytes())
            else:
                print(f"  downloading {name} ...")
                urllib.request.urlretrieve(SOURCE_URL + name, dst)
        got = sha256(dst)
        if got != digest:
            raise SystemExit(f"{dst}: sha256 {got} != pinned {digest}. "
                             f"Delete it and rerun, or check --src.")
    print(f"  sources verified (SLGNN @ {SOURCE_COMMIT[:12]})")


def check_relations(raw_dir, kg):
    """Join kg2id triples to the named kg_triplet.csv; return per-relation
    counts. Raises if any kg2id triple is (or could be) a label relation."""
    ent = pd.read_csv(raw_dir / "entity2id.txt", sep="\t")
    dbid_to_ent = dict(zip(ent["a"].astype(np.int64), ent["b"].astype(np.int64)))
    trip = pd.read_csv(raw_dir / "kg_triplet.csv", encoding="utf-8-sig")
    trip.columns = ["a", "type", "b"]
    trip["h"] = trip["a"].map(dbid_to_ent)
    trip["t"] = trip["b"].map(dbid_to_ent)
    label_counts = {r: int((trip["type"] == r).sum()) for r in LABEL_RELATIONS}
    # Genes that occur ONLY in label relations were never given an entity id
    # (entity2id.txt has 54,012 of kg_triplet's ids); any other unmapped row
    # would mean the files disagree.
    unmapped = trip[["h", "t"]].isna().any(axis=1)
    if (unmapped & ~trip["type"].isin(LABEL_RELATIONS)).any():
        raise SystemExit("kg_triplet.csv names an entity absent from "
                         "entity2id.txt in a non-label relation; the sources "
                         "are inconsistent")
    trip = trip[~unmapped].astype({"h": np.int64, "t": np.int64})
    names = {name for name, _, _ in RELATIONS}
    unknown = set(trip["type"]) - names - set(LABEL_RELATIONS)
    if unknown:
        raise SystemExit(f"kg_triplet.csv has relation types this script does "
                         f"not know: {sorted(unknown)}")

    kg_df = pd.DataFrame(kg, columns=["h", "r", "t"])
    kg_df["type"] = [RELATIONS[r][0] for r in kg_df["r"]]
    # Every kg2id triple must exist in kg_triplet.csv under exactly the name
    # RELATIONS gives its relation id.
    joined = kg_df.merge(trip[["h", "type", "t"]].drop_duplicates(),
                         on=["h", "type", "t"],
                         how="left",
                         indicator=True)
    missing = int((joined["_merge"] != "both").sum())
    if missing:
        raise SystemExit(
            f"{missing} kg2id triples have no kg_triplet.csv row with the "
            f"relation name RELATIONS assigns to their id; the relation "
            f"vocabulary is not what this script assumes")
    # The join pins every id to a non-label name, so no kg2id triple is a
    # label edge. `overlap` counts the gene-gene edges kg2id KEEPS between
    # labelled pairs; it cannot see edges the upstream file dropped, which
    # measure_label_keyed_deletion reports.
    lab = trip[trip["type"].isin(LABEL_RELATIONS)]
    lab_pairs = set(zip(lab["h"], lab["t"])) | set(zip(lab["t"], lab["h"]))
    gg = [i for i, (_, hg, tg) in enumerate(RELATIONS) if hg and tg]
    kg_gg = kg_df[kg_df["r"].isin(gg)]
    overlap = sum((h, t) in lab_pairs for h, t in zip(kg_gg["h"], kg_gg["t"]))

    kept = trip[~trip["type"].isin(LABEL_RELATIONS)]
    kept_set = set(zip(kept["h"], kept["type"], kept["t"]))
    kg_set = set(zip(kg_df["h"], kg_df["type"], kg_df["t"]))
    absent = kept_set - kg_set
    return {
        "label_relations_in_kg_triplet": label_counts,
        "label_relations_in_kg2id": 0,
        "kg2id_triples_verified_by_name": int(len(kg_df)),
        "nonlabel_kg_triplet_rows_absent_from_kg2id": len(absent),
        "gene_gene_kg_edges_between_labelled_pairs": int(overlap),
        "label_keyed_deletion": measure_label_keyed_deletion(
            lab, absent, kept, kg_gg),
    }


def measure_label_keyed_deletion(lab, absent, kept, kg_gg):
    """Which non-label kg_triplet.csv rows kg2id.txt lacks, and on which
    SynLethKG-labelled gene pairs. Measures only; the KG is written as shipped.

    lab     kg_triplet label rows (h, type, t), as entity ids
    absent  {(h, type name, t)} non-label kg_triplet rows missing from kg2id
    kept    kg_triplet non-label rows (h, type, t)
    kg_gg   kg2id gene-gene rows (h, r, t, type)
    Pairs are unordered: an edge in either direction counts for the pair.
    """

    def und(h, t):
        return (h, t) if h <= t else (t, h)

    pair_labels = defaultdict(set)
    for h, r, t in zip(lab["h"], lab["type"], lab["t"]):
        pair_labels[und(h, t)].add(r)
    by_rel, by_label, not_on_sl = defaultdict(int), defaultdict(int), 0
    for h, r, t in absent:
        by_rel[r] += 1
        labels = pair_labels.get(und(h, t), set())
        for name in labels or ("unlabelled", ):
            by_label[name] += 1
        not_on_sl += "SL_GsG" not in labels
    gg_names = {name for name, hg, tg in RELATIONS if hg and tg}
    kept_gg = kept[kept["type"].isin(gg_names)]
    gg_triplet = {und(h, t) for h, t in zip(kept_gg["h"], kept_gg["t"])}
    gg_kg2id = {und(h, t) for h, t in zip(kg_gg["h"], kg_gg["t"])}
    pairs = {}
    for name in LABEL_RELATIONS:
        ps = [p for p, ls in pair_labels.items() if name in ls]
        pairs[name] = {
            "pairs": len(ps),
            "with_gene_gene_edge_in_kg_triplet": sum(p in gg_triplet
                                                     for p in ps),
            "with_gene_gene_edge_in_kg2id": sum(p in gg_kg2id for p in ps),
        }
    return {
        "absent_rows_by_relation": dict(sorted(by_rel.items())),
        # A pair can carry several labels, so these may sum to more than
        # the number of absent rows.
        "absent_rows_by_pair_label": dict(sorted(by_label.items())),
        "absent_rows_not_on_SL_pairs": int(not_on_sl),
        "labelled_pairs": pairs,
    }


def _read_hgnc(path):
    sym, prev, alias = {}, defaultdict(set), defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ez = (row.get("entrez_id") or "").strip()
            if not ez:
                continue
            ez = str(int(float(ez)))
            sym[row["symbol"].strip()] = ez
            for col, dst in (("prev_symbol", prev), ("alias_symbol", alias)):
                for s in (row.get(col) or "").replace('"', "").split("|"):
                    if s.strip():
                        dst[s.strip()].add(ez)
    return sym, prev, alias


def _read_ncbi(path):
    sym, syn = {}, defaultdict(set)
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[0] != "9606":
                continue
            sym[p[2]] = p[1]
            if p[4] != "-":
                for s in p[4].split("|"):
                    syn[s].add(p[1])
    return sym, syn


def _read_synlethdb(paths):
    """SynLethDB's own gene name -> Entrez pairs (the KG's naming source)."""
    out = defaultdict(set)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                for side, idc in (("x", "x:START_ID"), ("y", "y:END_ID")):
                    if row.get(f"{side}_type") == "Gene":
                        out[row[f"{side}_name"].strip()].add(row[idc].strip())
    return out


def map_gene_entities(raw_dir, kg):
    """KG gene entity -> (name, Entrez or '', method)."""
    ent = pd.read_csv(raw_dir / "entity2id.txt", sep="\t")
    names = pd.read_csv(raw_dir / "dbid2name.csv", index_col=0)
    dbid_name = dict(zip(names["_id"].astype(np.int64), names["name"]))
    ent_name = {
        int(b): str(dbid_name.get(int(a), ""))
        for a, b in zip(ent["a"], ent["b"])
    }
    head_gene = np.array([hg for _, hg, _ in RELATIONS])
    tail_gene = np.array([tg for _, _, tg in RELATIONS])
    genes = set(kg[head_gene[kg[:, 1]], 0].tolist())
    genes |= set(kg[tail_gene[kg[:, 1]], 2].tolist())

    sdb = _read_synlethdb([
        REPO_DIR / MAPPING_FILES["synlethdb_sl"],
        REPO_DIR / MAPPING_FILES["synlethdb_nonsl"],
    ])
    h_sym, h_prev, h_alias = _read_hgnc(REPO_DIR / MAPPING_FILES["hgnc"])
    n_sym, n_syn = _read_ncbi(REPO_DIR / MAPPING_FILES["ncbi_gene_info"])
    lookups = (
        lambda n: sdb.get(n, set()),
        lambda n: {h_sym[n]} if n in h_sym else set(),
        lambda n: {n_sym[n]} if n in n_sym else set(),
        lambda n: h_prev.get(n, set()),
        lambda n: h_alias.get(n, set()),
        lambda n: n_syn.get(n, set()),
    )
    out = {}
    for e in sorted(genes):
        name = ent_name.get(e, "")
        ez, how = "", "unmapped"
        for method, look in zip(METHODS, lookups):
            hits = look(name)
            if len(hits) == 1:
                ez, how = next(iter(hits)), method
                break
            if len(hits) > 1 and method == "synlethdb":
                continue  # ambiguous in SynLethDB: fall through to HGNC
        out[e] = (name, ez, how)
    return out


def resolve_entrez(gene_map, degree):
    """Entrez -> one entity. Several KG entities resolving to one Entrez ID
    keep the most trusted method, then the higher degree, then the lower id."""
    best = {}
    rank = {m: i for i, m in enumerate(METHODS)}
    for e, (_, ez, how) in gene_map.items():
        if not ez:
            continue
        key = (rank[how], -int(degree[e]), e)
        if ez not in best or key < best[ez][0]:
            best[ez] = (key, e)
    return {ez: e for ez, (_, e) in best.items()}


def load_kg_for_genes(kg_dir, gene_order, inverse_r=True):
    """What the trainer needs, aligned to gene_order (shared genes.txt).

    Returns a dict with
      edge_index (2, E) int64 and edge_type (E,) int64 over KG entities, the
          inverse edge (t, h, r + R) added for every (h, r, t) when inverse_r
          (the original's --inverse_r True, via its networkx MultiDiGraph, so
          duplicate (h, t, r) collapse);
      n_relations  R, or 2R with inverse_r (the original doubles it);
      n_entities   KG entities + one ISOLATED entity per gene of gene_order
                   that SynLethKG does not contain (own trainable embedding,
                   no KG edges; the original treats every entity without
                   edges this way);
      gene_entity  (n_genes,) entity row of each gene (the original's
                   reindex_dict);
      n_in_kg, stamp.
    """
    kg_dir = Path(kg_dir)
    kg = np.load(kg_dir / "kg_triples.npy")
    meta = json.loads((kg_dir / "meta.json").read_text())
    n_kg = int(meta["n_entities"])
    n_rel = int(meta["n_relations"])
    tab = pd.read_csv(kg_dir / "entity_entrez.tsv", sep="\t", dtype=str)
    tab = tab[tab["selected"] == "1"]
    ez_to_ent = dict(zip(tab["entrez"], tab["entity"].astype(np.int64)))

    gene_entity = np.full(len(gene_order), -1, dtype=np.int64)
    for i, g in enumerate(gene_order):
        gene_entity[i] = ez_to_ent.get(str(g), -1)
    missing = np.where(gene_entity < 0)[0]
    gene_entity[missing] = n_kg + np.arange(len(missing))

    h, r, t = kg[:, 0], kg[:, 1], kg[:, 2]
    if inverse_r:
        trip = np.concatenate([np.stack([h, t, r], 1),
                               np.stack([t, h, r + n_rel], 1)])
        n_relations = 2 * n_rel
    else:
        trip = np.stack([h, t, r], 1)
        n_relations = n_rel
    trip = np.unique(trip, axis=0)
    return {
        "edge_index": trip[:, :2].T.copy(),
        "edge_type": trip[:, 2].copy(),
        "n_relations": n_relations,
        "n_entities": n_kg + len(missing),
        "gene_entity": gene_entity,
        "n_in_kg": int(len(gene_order) - len(missing)),
        "stamp": {
            "kg_dir": str(kg_dir),
            "kg_triples_sha256": sha256(kg_dir / "kg_triples.npy")[:16],
            "entity_entrez_sha256": sha256(kg_dir / "entity_entrez.tsv")[:16],
            "source_commit": meta.get("source_commit"),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out_dir", default=str(REPO_DIR / "data/synlethkg"))
    p.add_argument("--src",
                   default=None,
                   help="Directory holding the SLGNN repository's "
                   "data/SL/raw files; default downloads them at the "
                   "pinned commit.")
    p.add_argument("--genes",
                   default=str(REPO_DIR / "sl_comparison/folds/genes.txt"),
                   help="Gene universe to report coverage for (read only).")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("# built by SLGNN/build_synlethkg.py; "
                                    "never commit\n*\n")
    raw = out / "raw"
    fetch_sources(args.src, raw)

    kg = pd.read_csv(raw / "kg2id.txt", sep="\t", names=["h", "r", "t"])
    kg = kg.drop_duplicates().to_numpy(dtype=np.int64)
    if kg[:, 1].min() < 0 or kg[:, 1].max() >= len(RELATIONS):
        raise SystemExit(f"kg2id.txt relation ids span "
                         f"{kg[:, 1].min()}..{kg[:, 1].max()}, expected "
                         f"0..{len(RELATIONS) - 1}")
    checks = check_relations(raw, kg)
    print(f"  {len(kg):,} triples, {len(RELATIONS)} relations; label "
          f"relations {LABEL_RELATIONS} in kg_triplet.csv: "
          f"{checks['label_relations_in_kg_triplet']}, in kg2id.txt: 0")
    deletion = checks["label_keyed_deletion"]
    print(f"  kg2id.txt (used as shipped) also lacks "
          f"{checks['nonlabel_kg_triplet_rows_absent_from_kg2id']:,} non-label "
          f"kg_triplet.csv rows {deletion['absent_rows_by_relation']}, "
          f"{deletion['absent_rows_not_on_SL_pairs']} of them off SL pairs. "
          f"Labelled pairs with a gene-gene edge, kg_triplet.csv -> kg2id.txt:")
    for name, c in deletion["labelled_pairs"].items():
        print(f"    {name:10s} {c['with_gene_gene_edge_in_kg_triplet']:6,} -> "
              f"{c['with_gene_gene_edge_in_kg2id']:6,} of {c['pairs']:,} pairs")

    n_entities = int(kg[:, [0, 2]].max()) + 1  # the original's n_entities
    degree = np.bincount(kg[:, [0, 2]].ravel(), minlength=n_entities)
    gene_map = map_gene_entities(raw, kg)
    chosen = resolve_entrez(gene_map, degree)
    chosen_ent = set(chosen.values())
    with open(out / "entity_entrez.tsv", "w") as f:
        f.write("entity\tname\tentrez\tmethod\tdegree\tselected\n")
        for e, (name, ez, how) in sorted(gene_map.items()):
            f.write(f"{e}\t{name}\t{ez}\t{how}\t{degree[e]}\t"
                    f"{int(e in chosen_ent)}\n")
    with open(out / "relations.tsv", "w") as f:
        f.write("relation\tname\thead_gene\ttail_gene\tn_triples\n")
        counts = np.bincount(kg[:, 1], minlength=len(RELATIONS))
        for i, (name, hg, tg) in enumerate(RELATIONS):
            f.write(f"{i}\t{name}\t{int(hg)}\t{int(tg)}\t{counts[i]}\n")
    np.save(out / "kg_triples.npy", kg)

    methods = defaultdict(int)
    for _, _, how in gene_map.values():
        methods[how] += 1
    meta = {
        "source": f"github.com/zy972014452/SLGNN@{SOURCE_COMMIT}",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_FILES,
        "n_entities": n_entities,
        "n_relations": len(RELATIONS),
        "n_triples": int(len(kg)),
        "relations": [name for name, _, _ in RELATIONS],
        "removed_label_relations": list(LABEL_RELATIONS),
        # Disclosure, not a change: the triples are kg2id.txt as shipped.
        "kg_source": ("the original's data/SL/raw/kg2id.txt, unchanged. Beyond "
                      "the label relations it lacks the gene-gene edges "
                      "between SynLethKG SL pairs (label-keyed upstream edit, "
                      "kept on purpose); relation_checks.label_keyed_deletion "
                      "measures it"),
        "relation_checks": checks,
        "gene_entities": len(gene_map),
        "gene_entity_mapping": dict(methods),
        "entrez_ids_mapped": len(chosen),
        "mapping_sha256": mapping_sha256(REPO_DIR),
    }
    if Path(args.genes).exists():
        genes = [ln.strip() for ln in open(args.genes) if ln.strip()]
        hit = sum(g in chosen for g in genes)
        meta["coverage"] = {
            "genes_file": str(args.genes),
            "genes_sha256": sha256(args.genes)[:16],
            "n_genes": len(genes),
            "in_kg": hit,
            "isolated": len(genes) - hit,
        }
        print(f"  coverage: {hit:,}/{len(genes):,} shared genes have a KG "
              f"entity; {len(genes) - hit:,} get an isolated entity")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  gene entities: {len(gene_map):,} ({dict(methods)}); "
          f"wrote {out}")


if __name__ == "__main__":
    main()
