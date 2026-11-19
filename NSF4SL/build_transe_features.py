#!/usr/bin/env python3
"""
Build NSF4SL's ORIGINAL gene features: SynLethKG TransE entity embeddings.

The original NSF4SL (github.com/JieZheng-ShanghaiTech/NSF4SL, data_loader.py)
feeds every gene its row of ``data/kg_embed/kg_TransE_l2_entity.npy``: a
400-d TransE_l2 embedding of SynLethKG trained with DGL-KE after the
SL_GsG / NONSL_GnsG / SR_GsrG edges were removed (data/kg_process/
process_kg.ipynb). The only thing it needs from us is the link between a KG
entity and our Entrez-keyed gene universe, which the NSF4SL repo alone does
not give for every gene. It lists names for its 9,258 SL genes only. This
script builds that link:

  TransE row  <- entities.tsv            (row, KG entity id)      NSF4SL
  entity id   <- entity2id.txt           (neo4j db id -> entity)  KG4SL
  db id       -> dbid2name.csv           (db id -> name)          KG4SL
  gene?       <- kg_triplet.csv          (the metaedge of every relation
                                          that touches the db id: a 'G'
                                          endpoint of GpBP, AeG, GiG, ...)
  Entrez      <- Human_SL.csv            (SynLethDB 2.0's own NCBI id for the
                                          name, used first), else
                 siamese_sl/gene_name_utils (the repo's HGNC normaliser,
                                          current symbol -> alias -> Entrez)

KG4SL built SynLethKG and NSF4SL reuses its entity ids; the chain is checked
against NSF4SL's own id_mapping.csv (all 9,258 names must agree). When two KG
genes map to one Entrez id (45 with the pinned inputs, nearly all lncRNA alias
collisions; printed and recorded as meta.n_entrez_collisions), the entity
kept is the one the original uses for its SL genes (id_mapping.csv), else one
whose Entrez came from SynLethDB, else one whose name IS the current HGNC
symbol, else the lowest entity id.

Output (--out, a dict read by sl_data.load_transe_features, i.e.
train_nsf4sl.py --features transe --transe_path):
  embeddings   FloatTensor (G, 400)  raw TransE rows, NOT standardised (the
                                     original feeds them raw)
  gene_order   [Entrez str] * G
  entity_mean  FloatTensor (400,)    column mean over ALL 54,012 KG entities:
                                     the original's kgemb_data_mean, the value
                                     its augmentation writes into masked dims
  kg_entity, kg_name, entrez_source  per-row provenance
  meta         source commits, sha256 of every input (mapping_sha256 also
               hashes this builder), counts

Genes of the fold universe with no KG entity linked to their Entrez id are NOT
filled here; the trainer does that (sl_data.load_transe_features) and prints
the count.

Usage (from NSF4SL/; downloads the pinned inputs into --raw_dir once):
    python build_transe_features.py --download
"""

import argparse
import collections
import hashlib
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parent.parent
# The repo's one symbol -> Entrez normaliser (custom corrections, HGNC current
# symbol, HGNC alias), the same one every all_genes_*.pt went through.
sys.path.insert(0, str(_REPO / "siamese_sl"))
from gene_name_utils import GeneNameMapper  # noqa: E402

NSF4SL_COMMIT = "de000dec9993286e3228331c51a543668f6c832d"
KG4SL_COMMIT = "be3626a6970479affb95fdf6b1d84a6244e52637"
_RAW = "https://raw.githubusercontent.com/JieZheng-ShanghaiTech"

# local name -> (url, sha256). Pinned to commits so a moved upstream file
# fails loudly instead of silently changing the features.
SOURCES = {
    "kg_TransE_l2_entity.npy":
    (f"{_RAW}/NSF4SL/{NSF4SL_COMMIT}/data/kg_embed/kg_TransE_l2_entity.npy",
     "f80435f470a8eed31b5635b6543bf8243ff67e0180a816771131a43348a22f39"),
    "entities.tsv":
    (f"{_RAW}/NSF4SL/{NSF4SL_COMMIT}/data/kg_embed/entities.tsv",
     "30cd62e044ba9bffee864eec6e6d3fb9c751daaa2a3b8cee48a7abf304d2c98a"),
    "Human_SL.csv":
    (f"{_RAW}/NSF4SL/{NSF4SL_COMMIT}/data/sl_data/Human_SL.csv",
     "99ca92fb0fe6d99d8004aec63ff2417267a77aac06091ae783c18b6bde90ce0a"),
    "id_mapping.csv":
    (f"{_RAW}/NSF4SL/{NSF4SL_COMMIT}/data/sl_data/id_mapping.csv",
     "669f1b96dce974a12d282a628c0ecc3ce2456c303383c95f88b64d395f2775a9"),
    "kg_triplet.csv":
    (f"{_RAW}/NSF4SL/{NSF4SL_COMMIT}/data/kg_process/kg_triplet.csv",
     "fb61f918ebe33cba142661764d257b01df43464d3342cf87e7a89e984183243e"),
    "entity2id.txt":
    (f"{_RAW}/KG4SL/{KG4SL_COMMIT}/data/entity2id.txt",
     "bec5f2c5c6abd61f4630dddb305a6bc737a28315cb8472acd3e63d1d389450c6"),
    "dbid2name.csv":
    (f"{_RAW}/KG4SL/{KG4SL_COMMIT}/data/dbid2name.csv",
     "9dcb8ef22c38b0549b32383c2168226952ee106d50d61ea5c5868f6ad3320713"),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_sources(raw_dir, download):
    """Ensure every pinned input is in raw_dir with the pinned sha256.

    With ``download``, a missing input is fetched and a present one whose
    sha256 does not match (e.g. truncated by an interrupted earlier run) is
    fetched again. A download lands in ``<name>.part`` and is moved into place
    only once complete and verified, so the final path never holds a partial
    file this function wrote.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, want) in SOURCES.items():
        path = raw_dir / name
        if path.exists():
            got = _sha256(path)
            if got == want:
                continue
            if not download:
                raise RuntimeError(
                    f"{path}: sha256 {got} != pinned {want}. The file is not "
                    f"the one these features were defined on (a truncated "
                    f"download, a git-lfs pointer or a changed upstream "
                    f"file); rerun with --download to fetch it again.")
            print(f"  {name}: sha256 mismatch, downloading it again ...")
        elif not download:
            raise FileNotFoundError(
                f"{path} missing; rerun with --download to fetch it "
                f"from {url}")
        else:
            print(f"  downloading {name} ...")
        part = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(url, part)
            got = _sha256(part)
            if got != want:
                raise RuntimeError(
                    f"{url}: sha256 {got} != pinned {want}. The downloaded "
                    f"file is not the one these features were defined on (a "
                    f"git-lfs pointer or a changed upstream file).")
            os.replace(part, path)
        finally:
            if part.exists():
                part.unlink()
    return raw_dir


def gene_db_ids(triplet_csv):
    """db ids at a Gene endpoint of any SynLethKG relation.

    Relation names end in a Hetionet-style metaedge, e.g. PARTICIPATES_GpBP
    (Gene -> BP), EXPRESSES_AeG (Anatomy -> Gene), SL_GsG (Gene -> Gene):
    upper-case head code, lower-case verb, upper-case tail code.
    """
    tr = pd.read_csv(triplet_csv, encoding="utf-8-sig")
    tr.columns = ["a", "r", "b"]
    genes = set()
    for rel, sub in tr.groupby("r"):
        m = re.match(r"^([A-Z]+)[a-z]+([A-Z]+)$", rel.split("_", 1)[1])
        if m is None:
            raise ValueError(f"unparseable SynLethKG relation {rel!r}")
        if m.group(1) == "G":
            genes.update(sub["a"].tolist())
        if m.group(2) == "G":
            genes.update(sub["b"].tolist())
    return genes


def synlethdb_entrez(human_sl_csv):
    """Name -> Entrez from SynLethDB 2.0's own table (names with one id)."""
    hs = pd.read_csv(human_sl_csv, sep="\t")
    n2e = collections.defaultdict(set)
    for name_col, id_col in (("gene_a.name", "gene_a.identifier"),
                             ("gene_b.name", "gene_b.identifier")):
        for n, e in zip(hs[name_col], hs[id_col]):
            n2e[str(n)].add(str(e))
    return {n: next(iter(e)) for n, e in n2e.items() if len(e) == 1}


def build(raw_dir, hgnc_cache):
    raw = Path(raw_dir)
    emb = np.load(raw / "kg_TransE_l2_entity.npy").astype(np.float32)

    # entities.tsv: TransE row -> KG entity id (DGL-KE's entity dictionary).
    ent = pd.read_csv(raw / "entities.tsv", sep="\t", header=None)
    if not (ent[0].to_numpy() == np.arange(len(ent))).all() or \
            len(ent) != emb.shape[0]:
        raise RuntimeError("entities.tsv is not one row per TransE row")
    entity_row = {int(e): int(r) for r, e in zip(ent[0], ent[1])}

    e2i = pd.read_csv(raw / "entity2id.txt", sep="\t")
    db_of = {int(b): int(a) for a, b in zip(e2i["a"], e2i["b"])}
    d2n = pd.read_csv(raw / "dbid2name.csv", index_col=0)
    name_of_db = {int(i): str(n) for i, n in zip(d2n["_id"], d2n["name"])}
    if set(db_of) != set(entity_row):
        raise RuntimeError("KG4SL entity2id and NSF4SL entities.tsv cover "
                           "different entity ids")

    # The chain must reproduce NSF4SL's own names for its SL genes.
    idm = pd.read_csv(raw / "id_mapping.csv")
    bad = [(i, n) for i, n in zip(idm["id"], idm["name"])
           if name_of_db.get(db_of.get(int(i))) != str(n)]
    if bad:
        raise RuntimeError(f"{len(bad)}/{len(idm)} NSF4SL id_mapping names "
                           f"disagree with KG4SL's tables, e.g. {bad[:3]}")
    # The entities the original itself uses for its SL genes; they win an
    # Entrez collision (SynLethKG has two gene entities named TEC, and the
    # original's TEC is entity 46110).
    nsf4sl_sl_entities = {int(i) for i in idm["id"]}

    genes_db = gene_db_ids(raw / "kg_triplet.csv")
    gene_entities = sorted(e for e, d in db_of.items() if d in genes_db)
    print(f"SynLethKG: {emb.shape[0]:,} entities x {emb.shape[1]}d, "
          f"{len(gene_entities):,} genes; id chain agrees with NSF4SL's "
          f"id_mapping on {len(idm):,}/{len(idm):,} SL genes")

    sdb = synlethdb_entrez(raw / "Human_SL.csv")
    mapper = GeneNameMapper(cache_dir=str(hgnc_cache))
    # gene_name_utils falls back to "custom corrections only" when the HGNC
    # table is missing and cannot be downloaded. Here that would silently drop
    # thousands of KG genes (they would get the all-entity mean at training),
    # so a missing or unparsable table stops the build.
    mapper._ensure_loaded()
    hgnc_file = Path(hgnc_cache) / "hgnc_complete_set.txt"
    if not hgnc_file.exists() or not mapper._symbol_to_entrez:
        raise SystemExit(
            f"HGNC table {hgnc_file} is missing or has no Entrez ids; the "
            f"KG gene -> Entrez mapping needs it. Restore it (it is tracked "
            f"in git as data/cache/hgnc_complete_set.txt) or pass "
            f"--hgnc_cache.")
    import gene_name_utils  # which local symbol-correction table it loaded
    by_entrez = collections.defaultdict(list)
    n_src = collections.Counter()
    for e in gene_entities:
        name = name_of_db[db_of[e]]
        if name in sdb:
            ez, src = sdb[name], "synlethdb_id"
        else:
            ez, src = mapper.symbol_to_entrez(name), "hgnc"
        if ez is None:
            n_src["unmapped"] += 1
            continue
        n_src[src] += 1
        by_entrez[str(ez)].append((src, name, e))

    n_collide = 0
    rows = []
    for ez in sorted(by_entrez, key=int):
        cands = by_entrez[ez]
        if len(cands) > 1:
            n_collide += 1
            cur = mapper.entrez_to_symbol(ez)
            cands.sort(key=lambda c: (c[2] not in nsf4sl_sl_entities,
                                      c[0] != "synlethdb_id",
                                      c[1].upper() != cur, c[2]))
        src, name, e = cands[0]
        rows.append((ez, e, name, src))
    print(f"Entrez: {n_src['synlethdb_id']:,} via SynLethDB ids, "
          f"{n_src['hgnc']:,} via HGNC symbols, {n_src['unmapped']:,} "
          f"unmapped; {n_collide} Entrez ids claimed by >1 KG gene (kept "
          f"one each) -> {len(rows):,} genes")

    return {
        "embeddings":
        torch.from_numpy(np.stack([emb[entity_row[e]] for _, e, _, _ in rows])),
        "gene_order": [ez for ez, _, _, _ in rows],
        "entity_mean": torch.from_numpy(emb.mean(axis=0)),
        "kg_entity": [e for _, e, _, _ in rows],
        "kg_name": [n for _, _, n, _ in rows],
        "entrez_source": [s for _, _, _, s in rows],
        "meta": {
            "source": "NSF4SL SynLethKG TransE_l2 (DGL-KE), SL/NONSL/SR "
            "edges removed before training",
            "nsf4sl_commit": NSF4SL_COMMIT,
            "kg4sl_commit": KG4SL_COMMIT,
            "sha256": {k: v[1] for k, v in SOURCES.items()},
            # The gene-identity inputs: a different HGNC snapshot, local
            # correction table or normaliser version (e.g. its prev_symbol >
            # alias precedence) can change which Entrez id a KG gene gets.
            # So can this builder's own rules (gene-entity test, collision
            # precedence, SOURCES pins), hence its hash too;
            # sl_comparison/slurm/check_shared_inputs.py compares every key.
            "mapping_sha256": {
                "hgnc_complete_set.txt":
                _sha256(hgnc_file),
                "gene_corrections_config.py":
                (_sha256(gene_name_utils.GENE_CORRECTIONS_FILE)
                 if gene_name_utils.GENE_CORRECTIONS_FILE else None),
                "gene_name_utils.py":
                _sha256(gene_name_utils.__file__),
                "build_transe_features.py":
                _sha256(Path(__file__).resolve()),
            },
            "n_gene_corrections": len(gene_name_utils.GENE_CORRECTIONS),
            "n_entities": int(emb.shape[0]),
            "n_kg_genes": len(gene_entities),
            "n_genes": len(rows),
            "n_entrez_collisions": n_collide,
            "entrez_sources": dict(n_src),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw_dir",
                   default=str(_REPO / "data" / "nsf4sl_kg"),
                   help="Where the pinned upstream inputs live (git-ignored)")
    p.add_argument("--out",
                   default=str(_REPO / "data" / "nsf4sl_transe_features.pt"),
                   help="Output feature file (git-ignored, *.pt)")
    p.add_argument("--hgnc_cache",
                   default=str(_REPO / "data" / "cache"),
                   help="gene_name_utils cache dir (hgnc_complete_set.txt)")
    p.add_argument("--download",
                   action="store_true",
                   help="Fetch missing inputs, and inputs whose sha256 does "
                   "not match (e.g. a truncated download), from the pinned "
                   "GitHub commits")
    args = p.parse_args()

    fetch_sources(args.raw_dir, args.download)
    out = build(args.raw_dir, args.hgnc_cache)
    torch.save(out, args.out)
    print(f"Saved {tuple(out['embeddings'].shape)} TransE features -> "
          f"{args.out}")


if __name__ == "__main__":
    main()
