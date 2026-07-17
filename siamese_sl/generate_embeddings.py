#!/usr/bin/env python3
"""
Generate gene embeddings from precomputed sources.

Supports downloading, processing, aligning, and concatenating gene embeddings
for use with the Siamese SL prediction pipeline.

Supported embedding types (precomputed, auto-download/extract):
  geneformer    (auto)  Geneformer token embeddings (V1: 256d, V2: 1152d)
  scgpt         (512d)  scGPT token embeddings
  gene2vec      (200d)  Gene2Vec co-expression embeddings
  genept        (1536d) GenePT (Data Leakage with SL)
  bioconceptvec (100d)  BioConceptVec PubMed concepts
  node2vec_ppi  (128d)  Node2Vec on STRING PPI
  ppi_svd       (256d)  SVD of STRING PPI adjacency
  prot_t5       (1024d) ProtT5-XL protein sequence embeddings
  esm1b         (1280d) ESM-1b protein sequence embeddings
  esmc          (1152d) ESM Cambrian 600M protein sequence embeddings
  scprint       (auto)  scPRINT single-cell foundation model
  seqvec        (1024d) SeqVec ELMo-style protein embeddings
  text_embed    (1024d) Open-source text embeddings (mxbai, GenePT alternative)
  go2vec        (128d)  GO2Vec: Node2Vec on GO graph
  onto2vec      (128d)  Onto2Vec: Word2Vec on GO axioms
  mashup        (500d)  Mashup diffusion-based network embeddings
  ppi_raw       (1024d) PPI adjacency high-dim SVD baseline
  kg_complex    (512d)  ComplEx KG embeddings (STRING + GO)

Already available via dedicated scripts:
  esm2        (1280d) generate_all_genes_esm.py
  go          (200d)  generate_go_esm_embeddings.py --go_only

Usage:
    # Generate a single embedding type
    python generate_embeddings.py --type geneformer \\
        --gene_list ../data/all_genes_esm2.pt \\
        --output ../data/all_genes_geneformer.pt

    # Concatenate multiple embeddings into one file
    python generate_embeddings.py --concat \\
        ../data/all_genes_geneformer.pt \\
        ../data/all_genes_go.pt \\
        --output ../data/all_genes_geneformer_go.pt

    # List available embedding types
    python generate_embeddings.py --list
"""

import argparse
import os
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gene_name_utils import get_mapper

# Module-level gene list for NCBI sequence fetching (set by generate_embedding)
_ACTIVE_GENE_LIST: Optional[List[str]] = None

# =============================================================================
# Embedding Registry
# =============================================================================

EMBEDDING_REGISTRY = {
    "geneformer": {
        "dim": None,  # auto-detected from model (V1-10M: 256d, V2-104M: 1152d)
        "description": "Geneformer token embeddings (auto-detected dim)",
        "source": "GenePert (https://github.com/zou-group/GenePert)",
        "filename": "geneformer_gene_embeddings.pkl",
        "alt_filenames": ["gene_to_geneformer_embed.pkl"],
        "format": "pkl_dict",
        "download_url": None,  # no public direct download
        "notes": "Transformer embeddings from ~30M single-cell transcriptomes",
    },
    "scgpt": {
        "dim": 512,
        "description": "scGPT token embeddings (512d)",
        "source": "GenePert (https://github.com/zou-group/GenePert)",
        "filename": "scgpt_gene_embeddings.pkl",
        "alt_filenames": ["all_scgpt_gene_embeddings.pkl"],
        "format": "pkl_dict",
        "download_url": None,  # no public direct download
        "notes": "Generative transformer embeddings from 33M+ single cells",
    },
    "gene2vec": {
        "dim": 200,
        "description": "Gene2Vec co-expression embeddings (200d)",
        "source": "Gene2Vec (https://github.com/jingcheng-du/Gene2vec)",
        "filename": "gene2vec_dim_200_iter_9_w2v.txt",
        "alt_filenames": ["gene2vec_dim_200_iter_9.txt"],
        "format": "w2v_text",
        "download_url":
        "https://raw.githubusercontent.com/jingcheng-du/Gene2vec/master/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt",
        "notes": "Word2Vec skip-gram on co-expression contexts",
    },
    "genept": {
        "dim":
        1536,
        "description":
        "GenePT (Data Leakage with SL) text embeddings (1536d)",
        "source":
        "Zenodo (https://zenodo.org/records/10833191)",
        "filename":
        "GenePT_gene_embedding_ada_text.pickle",
        "alt_filenames": [
            "GenePT_NCBI+UniProt_3.5.pkl",
            "GPT_3_5_gene_embeddings_augment.pickle",
        ],
        "format":
        "pkl_dict",
        "download_url":
        "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1",
        "download_type":
        "zip",  # needs unzipping
        "notes":
        ("WARNING: Literature-derived embeddings may encode known SL "
         "relationships, causing data leakage for SL prediction tasks"),
    },
    "bioconceptvec": {
        "dim":
        100,
        "description":
        "BioConceptVec PubMed concept embeddings (100d)",
        "source":
        "NCBI (https://github.com/ncbi/BioConceptVec)",
        "filename":
        "bioconceptvec_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,  # generated from concept_cbow.json + gene_info
        "notes":
        "Word2Vec on PubMed concepts; no SL data leakage (concept-level, not relation-level)",
    },
    "node2vec_ppi": {
        "dim": 128,
        "description": "Node2Vec on STRING PPI network (128d)",
        "source": "STRING v12.0 (https://string-db.org)",
        "filename": "node2vec_ppi_gene_embeddings.pkl",
        "alt_filenames": [],
        "format": "pkl_dict",
        "download_url": None,  # generated from STRING PPI
        "notes": "Requires node2vec package (pip install node2vec)",
    },
    "ppi_svd": {
        "dim":
        256,
        "description":
        "PPI adjacency + SVD (256d)",
        "source":
        "STRING v12.0 (https://string-db.org)",
        "filename":
        "ppi_svd_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,  # generated from STRING PPI
        "notes":
        "SVD of high-confidence PPI adjacency matrix; strong zero-compute baseline",
    },
    "prot_t5": {
        "dim":
        1024,
        "description":
        "ProtT5-XL protein sequence embeddings (1024d)",
        "source":
        "Rostlab (https://huggingface.co/Rostlab/prot_t5_xl_uniref50)",
        "filename":
        "prot_t5_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "T5-based PLM; strong for GO/function prediction. Requires GPU + UniProt sequences.",
    },
    "esm1b": {
        "dim": 1280,
        "description": "ESM-1b protein sequence embeddings (1280d)",
        "source": "Meta/FAIR (https://github.com/facebookresearch/esm)",
        "filename": "esm1b_gene_embeddings.pkl",
        "alt_filenames": [],
        "format": "pkl_dict",
        "download_url": None,
        "notes": "Predecessor to ESM-2; 650M params trained on UniRef50.",
    },
    "esmc": {
        "dim":
        1152,
        "description":
        "ESM Cambrian 600M protein sequence embeddings (1152d)",
        "source":
        "EvolutionaryScale (https://github.com/evolutionaryscale/esm)",
        "filename":
        "esmc_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "ESM C 600M via Synthyra/ESMplusplus_large; matches ESM2-3B quality. GPU + UniProt sequences.",
    },
    "scprint": {
        "dim":
        None,
        "description":
        "scPRINT gene embeddings (auto-detected dim)",
        "source":
        "cantinilab (https://github.com/cantinilab/scPRINT)",
        "filename":
        "scprint_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "Large transformer pretrained on 50M+ cells; ESM2-derived gene IDs.",
    },
    "seqvec": {
        "dim": 1024,
        "description": "SeqVec protein sequence embeddings (1024d)",
        "source": "Heinzinger et al. (https://github.com/mheinzinger/SeqVec)",
        "filename": "seqvec_gene_embeddings.pkl",
        "alt_filenames": [],
        "format": "pkl_dict",
        "download_url": None,
        "notes": "ELMo-style model trained on UniRef50. Requires allennlp.",
    },
    "text_embed": {
        "dim":
        1024,
        "description":
        "Open-source text embeddings of gene summaries (1024d)",
        "source":
        "mxbai-embed-large (https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)",
        "filename":
        "text_embed_gene_embeddings.pkl",
        "alt_filenames": ["mxbai_gene_embeddings.pkl"],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "GenePT alternative: open-source, no API cost, no SL data leakage. "
        "Embeds NCBI gene descriptions via sentence-transformers.",
    },
    "go2vec": {
        "dim": 128,
        "description": "GO2Vec: Node2Vec on Gene Ontology graph (128d)",
        "source": "Gene Ontology (http://geneontology.org)",
        "filename": "go2vec_gene_embeddings.pkl",
        "alt_filenames": [],
        "format": "pkl_dict",
        "download_url": None,
        "notes":
        "Node2Vec on GO DAG, mean-pooled per gene via GO annotations.",
    },
    "onto2vec": {
        "dim":
        128,
        "description":
        "Onto2Vec: Word2Vec on GO axiom sentences (128d)",
        "source":
        "Gene Ontology (http://geneontology.org)",
        "filename":
        "onto2vec_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "GO axioms converted to sentences, embedded with Word2Vec, "
        "mean-pooled per gene.",
    },
    "mashup": {
        "dim":
        500,
        "description":
        "Mashup diffusion-based network embeddings (500d)",
        "source":
        "Cho et al. 2016 (http://cb.csail.mit.edu/cb/mashup/)",
        "filename":
        "mashup_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "Diffusion kernel on STRING PPI via spectral decomposition + log transform.",
    },
    "ppi_raw": {
        "dim":
        1024,
        "description":
        "PPI adjacency high-dim SVD baseline (1024d)",
        "source":
        "STRING v12.0 (https://string-db.org)",
        "filename":
        "ppi_raw_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "Higher-dim SVD of PPI adjacency (cf. ppi_svd at 256d). "
        "Strong zero-compute baseline for interaction tasks.",
    },
    "kg_complex": {
        "dim":
        512,
        "description":
        "ComplEx KG embeddings from biological knowledge graph (512d; real ⊕ imag)",
        "source":
        "STRING PPI + GO annotations via PyKEEN",
        "filename":
        "kg_complex_gene_embeddings.pkl",
        "alt_filenames": [],
        "format":
        "pkl_dict",
        "download_url":
        None,
        "notes":
        "ComplEx trained on heterogeneous bio KG (PPI + GO triples). "
        "Requires pykeen (pip install pykeen).",
    },
}

EXISTING_EMBEDDINGS = {
    "esm2": "generate_all_genes_esm.py  (1280d, Pool PaRTI)",
    "go": "generate_go_esm_embeddings.py --go_only  (200d, anc2vec)",
}

# =============================================================================
# Loaders
# =============================================================================


def load_pkl_dict(filepath: str) -> Dict[str, np.ndarray]:
    """Load embeddings from a pickle file.

    Handles common formats:
      - dict  {gene_symbol: np.array}
      - pandas DataFrame with gene names as index
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        emb_dict = {}
        for gene, vec in data.items():
            key = str(gene)
            if isinstance(vec, np.ndarray):
                emb_dict[key] = vec.astype(np.float32)
            elif hasattr(vec, "numpy"):  # torch tensor
                emb_dict[key] = vec.detach().cpu().numpy().astype(np.float32)
            else:
                emb_dict[key] = np.array(vec, dtype=np.float32)
        return emb_dict

    # Try pandas DataFrame (lazy import to avoid hard dependency)
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return {
                str(gene): row.values.astype(np.float32)
                for gene, row in data.iterrows()
            }
    except ImportError:
        pass

    raise ValueError(f"Unexpected pkl format: {type(data).__name__}. "
                     "Expected dict {{gene: array}} or pandas DataFrame.")


def load_w2v_text(filepath: str) -> Dict[str, np.ndarray]:
    """Load embeddings from word2vec text format.

    Format: one gene per line, ``gene_name dim1 dim2 ... dimN``.
    Optional header line with ``num_words dim``.
    """
    emb_dict: Dict[str, np.ndarray] = {}

    with open(filepath, "r") as f:
        first_line = f.readline().strip().split()

        # Detect header (two integers: num_words dim)
        if not first_line:
            pass  # empty file
        elif (len(first_line) == 2 and first_line[0].isdigit()
              and first_line[1].isdigit()):
            pass  # skip header
        else:
            gene_name = first_line[0]
            vector = np.array([float(x) for x in first_line[1:]],
                              dtype=np.float32)
            emb_dict[gene_name] = vector

        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            gene_name = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            emb_dict[gene_name] = vector

    return emb_dict


_LOADERS = {
    "pkl_dict": load_pkl_dict,
    "w2v_text": load_w2v_text,
}

# =============================================================================
# Shared data helpers
# =============================================================================


def _load_protein_sequences(cache_dir: str,
                            gene_ids: Optional[List[str]] = None,
                            ) -> Dict[str, str]:
    """Load protein sequences from NCBI via Entrez Gene ID → RefSeq protein.

    Uses NCBI elink (Gene → Protein RefSeq) + efetch to retrieve canonical
    protein sequences. Results are cached as JSON (symbol → sequence) to
    avoid repeated downloads and fragile FASTA header re-parsing.

    Falls back to cached UniProt FASTA if NCBI fetch fails.

    Args:
        cache_dir: Directory for caching downloaded data.
        gene_ids: List of Entrez Gene IDs. If None, loads from any existing
                  ESM .pt file in the data directory.

    Returns:
        Dict mapping gene_symbol -> protein_sequence.
    """
    import json
    import time
    import xml.etree.ElementTree as ET

    json_cache = os.path.join(cache_dir, "ncbi_gene_sequences.json")
    fasta_cache = os.path.join(cache_dir, "ncbi_protein_sequences.fasta")

    # Prefer JSON cache (symbol → sequence, no re-parsing needed)
    if os.path.exists(json_cache) and gene_ids is None:
        with open(json_cache) as f:
            gene_seqs = json.load(f)
        print(f"  Loaded {len(gene_seqs)} protein sequences from cache")
        return gene_seqs

    if gene_ids is None:
        # Use module-level gene list if available
        gene_ids = _ACTIVE_GENE_LIST

    if gene_ids is None:
        # Try to find gene list from existing ESM .pt file
        for name in ["all_genes_esm2.pt", "all_genes_esm1b.pt"]:
            pt_path = os.path.join(os.path.dirname(cache_dir), name)
            if os.path.exists(pt_path):
                data = torch.load(pt_path, map_location="cpu",
                                  weights_only=False)
                gene_ids = list(data["gene_order"])
                print(f"  Loaded {len(gene_ids)} gene IDs from {pt_path}")
                break

    if gene_ids is None:
        print("  WARNING: No gene list available for NCBI fetch")
        # Fall back to UniProt if available
        return _load_uniprot_sequences_fallback(cache_dir)

    # Get gene symbol mapping for return value
    from gene_name_utils import get_mapper
    mapper = get_mapper()

    # Step 1: elink Gene ID → RefSeq protein UIDs
    elink_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    gene_to_protein: Dict[str, str] = {}
    batch_size = 200

    print(f"  Linking {len(gene_ids)} Gene IDs → RefSeq proteins...")
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        for attempt in range(3):
            try:
                import requests as _req
                # Use separate id params (not comma-joined) so elink
                # returns one LinkSet per gene instead of merging them.
                params = [("dbfrom", "gene"), ("db", "protein"),
                          ("linkname", "gene_protein_refseq"),
                          ("retmode", "xml")]
                params.extend(("id", gid) for gid in batch)
                resp = _req.post(elink_url, data=params, timeout=60)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                for linkset in root.findall(".//LinkSet"):
                    id_elem = linkset.find("IdList/Id")
                    if id_elem is None:
                        continue
                    gid = id_elem.text
                    link_db = linkset.find(".//LinkSetDb")
                    if link_db is None:
                        continue
                    pids = [l.find("Id").text for l in link_db.findall("Link")
                            if l.find("Id") is not None]
                    if pids:
                        gene_to_protein[gid] = pids[0]
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  elink batch failed: {e}")
        time.sleep(0.35)

    print(f"  Found RefSeq protein links for {len(gene_to_protein)}/{len(gene_ids)} genes")

    # Step 2: Resolve GI numbers → accessions via esummary, so we can
    # match FASTA headers (which use accessions, not GIs).
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    gi_to_acc: Dict[str, str] = {}
    protein_gis = list(gene_to_protein.values())

    print(f"  Resolving {len(protein_gis)} protein GIs → accessions...")
    for i in range(0, len(protein_gis), batch_size):
        batch = protein_gis[i:i + batch_size]
        for attempt in range(3):
            try:
                import requests as _req
                resp = _req.post(esummary_url, data={
                    "db": "protein",
                    "id": ",".join(batch),
                    "retmode": "json",
                }, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                for gi in batch:
                    info = data.get("result", {}).get(gi, {})
                    acc = info.get("accessionversion")
                    if acc:
                        gi_to_acc[gi] = acc
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  esummary batch failed: {e}")
        time.sleep(0.35)

    # Step 3: efetch protein sequences
    all_fasta: List[str] = []

    print(f"  Fetching {len(protein_gis)} protein sequences...")
    for i in range(0, len(protein_gis), batch_size):
        batch = protein_gis[i:i + batch_size]
        for attempt in range(3):
            try:
                import requests as _req
                resp = _req.post(efetch_url, data={
                    "db": "protein", "id": ",".join(batch),
                    "rettype": "fasta", "retmode": "text",
                }, timeout=60)
                resp.raise_for_status()
                all_fasta.append(resp.text)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  efetch batch failed: {e}")
        time.sleep(0.35)

    # Save raw FASTA cache
    with open(fasta_cache, "w") as f:
        f.write("\n".join(all_fasta))

    # Build reverse map: accession (with/without version) → gene_id
    # This handles both old-style ">gi|123|ref|NP_...| ..." and
    # new-style ">NP_... ..." FASTA headers.
    acc_to_gene: Dict[str, str] = {}
    for gid, gi in gene_to_protein.items():
        # Map by GI (for old-style headers)
        acc_to_gene[gi] = gid
        # Map by accession (for new-style headers)
        acc = gi_to_acc.get(gi)
        if acc:
            acc_to_gene[acc] = gid
            if "." in acc:
                acc_to_gene[acc.split(".")[0]] = gid

    gene_seqs: Dict[str, str] = {}
    current_gene_id = None
    current_seq: List[str] = []

    def _save():
        if current_gene_id and current_seq:
            seq = "".join(current_seq)
            # Map Entrez ID → symbol for backward compat with extract functions
            sym = mapper.entrez_to_symbol(current_gene_id)
            if sym:
                if sym not in gene_seqs or len(seq) > len(gene_seqs[sym]):
                    gene_seqs[sym] = seq

    for fasta_text in all_fasta:
        for line in fasta_text.strip().split("\n"):
            line = line.strip()
            if line.startswith(">"):
                _save()
                current_seq = []
                current_gene_id = None
                # Match header tokens against our accession/GI reverse map
                tokens = line[1:].replace("|", " ").split()
                for token in tokens:
                    if token in acc_to_gene:
                        current_gene_id = acc_to_gene[token]
                        break
                    base = token.split(".")[0]
                    if base in acc_to_gene:
                        current_gene_id = acc_to_gene[base]
                        break
            elif line:
                current_seq.append(line)
    _save()

    # Save JSON cache (symbol → sequence) so future loads don't re-parse FASTA
    with open(json_cache, "w") as f:
        json.dump(gene_seqs, f)

    print(f"  Loaded {len(gene_seqs)} protein sequences from NCBI")
    return gene_seqs


def _load_uniprot_sequences_fallback(cache_dir: str) -> Dict[str, str]:
    """Fallback: load from cached UniProt FASTA if available."""
    fasta_path = os.path.join(cache_dir, "human_proteome_reviewed.fasta")
    if not os.path.exists(fasta_path):
        print("  No cached UniProt FASTA available either")
        return {}

    gene_seqs: Dict[str, str] = {}
    current_gene = None
    current_seq: List[str] = []

    def _save():
        if current_gene and current_seq:
            seq = "".join(current_seq)
            if current_gene not in gene_seqs or len(seq) > len(
                    gene_seqs[current_gene]):
                gene_seqs[current_gene] = seq

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                _save()
                current_seq = []
                current_gene = None
                if "GN=" in line:
                    current_gene = line.split("GN=")[1].split()[0]
                elif "|" in line:
                    parts = line.split("|")
                    if len(parts) > 2 and "_HUMAN" in parts[2]:
                        current_gene = parts[2].split()[0].replace(
                            "_HUMAN", "")
            else:
                current_seq.append(line)
        _save()

    print(f"  Loaded {len(gene_seqs)} protein sequences from UniProt (fallback)")
    return gene_seqs



def _load_go_graph_and_annotations(
        cache_dir: str) -> Tuple[Optional[object], Dict[str, set]]:
    """Load GO OBO graph and human GAF annotations. Downloads if needed.

    Returns (networkx DiGraph or None, {gene_symbol: set_of_GO_term_IDs}).
    """
    import gzip as _gzip

    # cache_dir must already exist (server does not allow mkdir)

    obo_path = os.path.join(cache_dir, "go-basic.obo")
    if not os.path.exists(obo_path):
        print("  Downloading GO ontology (go-basic.obo)...")
        _run_download("http://purl.obolibrary.org/obo/go/go-basic.obo",
                      obo_path)

    gaf_path = os.path.join(cache_dir, "goa_human.gaf.gz")
    if not os.path.exists(gaf_path):
        print("  Downloading human GO annotations (GAF)...")
        _run_download(
            "http://geneontology.org/gene-associations/goa_human.gaf.gz",
            gaf_path)

    # Parse GO graph
    go_graph = None
    if os.path.exists(obo_path):
        try:
            import obonet
            go_graph = obonet.read_obo(obo_path)
            print(f"  GO graph: {go_graph.number_of_nodes()} terms, "
                  f"{go_graph.number_of_edges()} edges")
        except ImportError:
            print("  obonet not installed (pip install obonet)")

    # Parse GAF annotations
    gene_to_go: Dict[str, set] = {}
    if os.path.exists(gaf_path):
        with _gzip.open(gaf_path, "rt") as f:
            for line in f:
                if line.startswith("!"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                if any(q.strip() == "NOT"
                       for q in fields[3].upper().split("|")):
                    continue
                gene = fields[2]
                go_term = fields[4]
                gene_to_go.setdefault(gene, set()).add(go_term)
        print(f"  GO annotations: {len(gene_to_go)} genes")

        # Normalize gene names (custom corrections + HGNC)
        mapper = get_mapper()
        normalized: Dict[str, set] = {}
        for gene, terms in gene_to_go.items():
            norm = mapper.gene_name_normalize(gene)
            normalized.setdefault(norm, set()).update(terms)
        if len(normalized) != len(gene_to_go):
            print(
                f"  After gene name normalization: {len(normalized)} unique genes"
            )
        gene_to_go = normalized

    return go_graph, gene_to_go


# =============================================================================
# Gene list helpers
# =============================================================================


def load_gene_list(path: str) -> List[str]:
    """Load gene list from a ``.pt`` embedding file or a plain text file."""
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if "gene_order" in data:
                return list(data["gene_order"])
            if "gene_to_idx" in data:
                g2i = data["gene_to_idx"]
                n = len(g2i)
                gene_order = [""] * n
                for gene, idx in g2i.items():
                    gene_order[idx] = gene
                if "" in gene_order:
                    raise ValueError("gene_to_idx has non-contiguous indices")
                return gene_order
        raise ValueError(f"Cannot extract gene list from {path}")
    else:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


# =============================================================================
# Alignment
# =============================================================================


def align_embeddings(
    emb_dict: Dict[str, np.ndarray],
    gene_list: List[str],
    embedding_dim: int,
    embedding_type: str,
) -> Tuple[torch.Tensor, dict]:
    """Align precomputed embeddings to a reference gene list.

    The reference gene_list contains Entrez Gene IDs (strings).
    The emb_dict keys are gene symbols (from external sources).

    Pipeline:
      1. Convert emb_dict keys: symbol -> normalize -> Entrez ID
      2. Match against gene_list (Entrez IDs)
      3. Genes with no match receive NaN vectors (Huber-imputed at load time)

    Returns ``(embeddings_tensor, stats_dict)``.
    """
    # Convert embedding dict keys from symbols to Entrez IDs
    mapper = get_mapper()
    emb_entrez = mapper.symbols_to_entrez_dict(emb_dict, report=True)

    embeddings = np.full((len(gene_list), embedding_dim),
                         np.nan,
                         dtype=np.float32)
    matched = 0
    missing: List[str] = []

    for i, eid in enumerate(gene_list):
        vec = emb_entrez.get(eid)
        if vec is not None:
            dim = min(len(vec), embedding_dim)
            embeddings[i, :dim] = vec[:dim]
            matched += 1
        else:
            missing.append(eid)

    coverage = matched / len(gene_list) * 100 if gene_list else 0.0

    stats = {
        "matched": matched,
        "missing": len(missing),
        "total": len(gene_list),
        "coverage_pct": coverage,
    }

    print(f"  Gene coverage: {matched}/{len(gene_list)} ({coverage:.1f}%)")
    if gene_list and coverage < 5.0:
        # align_embeddings matches the reference gene_list against Entrez-mapped
        # embedding keys. Near-zero coverage almost always means the --gene_list
        # is not in Entrez IDs (e.g. a raw symbol list), which would otherwise
        # save a silently all-NaN embedding file. Warn loudly.
        print(f"  WARNING: coverage is only {coverage:.1f}% — the output would "
              f"be almost entirely NaN. Is --gene_list in NCBI Entrez IDs? "
              f"align_embeddings matches Entrez IDs, not gene symbols.")
    if missing:
        shown = missing[:20]
        print(f"  Missing Entrez IDs (first {len(shown)}): {shown}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    print(f"  Missing genes receive NaN vectors "
          f"(Huber-imputed per dimension at load time)")

    if embedding_type == "genept":
        print(
            "  WARNING: GenePT (Data Leakage with SL) -- literature embeddings "
            "may encode known SL relationships")

    return torch.tensor(embeddings, dtype=torch.float32), stats


# =============================================================================
# Cache / file resolution
# =============================================================================


def find_precomputed_file(embedding_type: str,
                          cache_dir: str) -> Optional[str]:
    """Search *cache_dir* for the expected precomputed file."""
    info = EMBEDDING_REGISTRY[embedding_type]
    filenames = [info["filename"]] + info.get("alt_filenames", [])
    for fname in filenames:
        for path in [
                os.path.join(cache_dir, fname),
                os.path.join(cache_dir, embedding_type, fname),
        ]:
            if os.path.exists(path):
                return path
    return None


def _run_download(url: str, output_path: str) -> bool:
    """Download a file using wget, curl, or urllib (in that order)."""
    import shutil
    import subprocess

    # Try wget first (most reliable for large files)
    if shutil.which("wget"):
        ret = subprocess.run(["wget", "-q", "-O", output_path, url],
                             capture_output=True)
        if ret.returncode == 0:
            return True

    # Try curl
    if shutil.which("curl"):
        ret = subprocess.run(["curl", "-sL", "-o", output_path, url],
                             capture_output=True)
        if ret.returncode == 0:
            return True

    # Fallback to urllib
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception:
        return False


def download_precomputed(embedding_type: str, cache_dir: str) -> Optional[str]:
    """Auto-download precomputed embeddings if a URL is available.

    Returns the path to the downloaded file, or None if download is not
    available or failed.
    """
    info = EMBEDDING_REGISTRY[embedding_type]
    url = info.get("download_url")
    if not url:
        return None

    # cache_dir must already exist (server does not allow mkdir)
    target = os.path.join(cache_dir, info["filename"])

    print(f"  Downloading {embedding_type} from {url} ...")

    try:
        if info.get("download_type") == "zip":
            # Download zip, extract, find the target file
            zip_path = os.path.join(cache_dir,
                                    f"{embedding_type}_download.zip")
            if not _run_download(url, zip_path):
                print(f"  Download failed")
                return None
            print(
                f"  Downloaded zip ({os.path.getsize(zip_path) / 1e6:.1f} MB)")

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                print(f"  Zip contents: {names}")
                # Find the target file (or alt filenames) inside the zip
                found = None
                search = [info["filename"]] + info.get("alt_filenames", [])
                for member in names:
                    basename = os.path.basename(member)
                    if basename in search:
                        found = member
                        break
                if found is None:
                    # Take the first .pickle/.pkl file
                    for member in names:
                        if member.endswith((".pickle", ".pkl")):
                            found = member
                            break
                if found:
                    zf.extract(found, cache_dir)
                    extracted = os.path.join(cache_dir, found)
                    # Move to expected filename if different
                    if extracted != target:
                        os.rename(extracted, target)
                        # Clean up intermediate directory from zip extraction
                        extracted_parent = os.path.dirname(extracted)
                        if extracted_parent != str(
                                cache_dir) and os.path.isdir(extracted_parent):
                            try:
                                os.rmdir(
                                    extracted_parent)  # only removes if empty
                            except OSError:
                                pass
                    print(f"  Extracted: {target}")
                else:
                    print(f"  WARNING: Could not find embedding file in zip")
                    return None

            # Clean up zip
            os.remove(zip_path)
        else:
            # Direct download
            if not _run_download(url, target):
                print(f"  Download failed")
                return None
            print(
                f"  Downloaded: {target} ({os.path.getsize(target) / 1e6:.1f} MB)"
            )

        return target

    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def extract_geneformer_from_hf(cache_dir: str) -> Optional[str]:
    """Extract Geneformer gene embeddings from HuggingFace model.

    Downloads the model + dictionary files, extracts the token embedding
    layer, and maps Ensembl IDs to gene symbols.

    Requires: transformers, huggingface_hub (pip install transformers).
    Returns path to saved pkl, or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface_hub not installed (pip install transformers)")
        return None

    print("  Extracting Geneformer embeddings from HuggingFace...")
    # cache_dir must already exist (server does not allow mkdir)

    try:
        # Download dictionary files
        print("  Downloading token dictionary...")
        token_dict_path = hf_hub_download(
            "ctheodoris/Geneformer", "geneformer/token_dictionary_gc104M.pkl")
        print("  Downloading gene name dictionary...")
        gene_name_path = hf_hub_download(
            "ctheodoris/Geneformer", "geneformer/gene_name_id_dict_gc104M.pkl")

        with open(token_dict_path, "rb") as f:
            token_dict = pickle.load(f)  # {ensembl_id: token_index}
        with open(gene_name_path, "rb") as f:
            gene_name_dict = pickle.load(f)  # {gene_name: ensembl_id}

        # Invert gene_name_dict: ensembl_id -> gene_name
        ensembl_to_gene = {v: k for k, v in gene_name_dict.items()}
        print(f"  Token dict: {len(token_dict)} tokens")
        print(f"  Gene name dict: {len(gene_name_dict)} genes")

        # Download and load model embedding layer
        print("  Downloading model (this may take a few minutes)...")
        model_path = hf_hub_download("ctheodoris/Geneformer",
                                     "model.safetensors")

        # Load just the embedding weights (avoid loading full model into GPU)
        # Key may be "bert.embeddings.word_embeddings.weight" or
        # "embeddings.word_embeddings.weight" depending on model version
        emb_key_candidates = [
            "bert.embeddings.word_embeddings.weight",
            "embeddings.word_embeddings.weight",
        ]
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in emb_key_candidates:
                    if key in f.keys():
                        emb_weight = f.get_tensor(key).numpy()
                        break
                else:
                    raise KeyError(
                        f"No embedding key found. Available: "
                        f"{[k for k in f.keys() if 'embedding' in k.lower()]}")
        except ImportError:
            import torch as _torch
            state = _torch.load(model_path,
                                map_location="cpu",
                                weights_only=True)
            for key in emb_key_candidates:
                if key in state:
                    emb_weight = state[key].numpy()
                    break
            else:
                raise KeyError(f"No embedding key found in checkpoint")

        print(f"  Embedding matrix shape: {emb_weight.shape}")

        # Build gene_name -> embedding dict
        # Chain: gene_name -> ensembl_id -> token_index -> embedding
        emb_dict = {}
        for ensembl_id, token_idx in token_dict.items():
            if not ensembl_id.startswith("ENSG"):
                continue  # skip special tokens (<pad>, <mask>, etc.)
            if ensembl_id in ensembl_to_gene and token_idx < len(emb_weight):
                gene_name = ensembl_to_gene[ensembl_id]
                emb_dict[gene_name.upper()] = emb_weight[token_idx].astype(
                    np.float32)

        print(f"  Mapped {len(emb_dict)} genes (Ensembl -> gene symbol)")

        output = os.path.join(cache_dir, "geneformer_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  Geneformer extraction failed: {e}")
        return None


def extract_scgpt_from_checkpoint(cache_dir: str) -> Optional[str]:
    """Extract scGPT gene embeddings from a downloaded checkpoint.

    Looks for the checkpoint folder at cache_dir/scGPT_human/ containing
    best_model.pt and vocab.json. Extracts the gene encoder embedding
    weights and maps via vocab (gene_symbol -> token_index).

    Returns path to saved pkl, or None on failure.
    """
    import json as _json

    # Search for checkpoint files
    search_dirs = [
        os.path.join(cache_dir, "scGPT_human"),
        os.path.join(cache_dir, "scgpt_human"),
        cache_dir,
    ]
    model_path = None
    vocab_path = None
    for d in search_dirs:
        mp = os.path.join(d, "best_model.pt")
        vp = os.path.join(d, "vocab.json")
        if os.path.exists(mp) and os.path.exists(vp):
            model_path, vocab_path = mp, vp
            break

    if model_path is None:
        return None

    print(f"  Extracting scGPT embeddings from {model_path} ...")

    try:
        import torch as _torch

        # Load vocab: {gene_symbol: token_index}
        with open(vocab_path, "r") as f:
            vocab = _json.load(f)
        print(f"  Vocab: {len(vocab)} genes")

        # Load checkpoint and find gene encoder embedding
        ckpt = _torch.load(model_path, map_location="cpu", weights_only=False)

        # Find the embedding key (may vary across checkpoint versions)
        emb_key = None
        for k in ckpt.keys():
            if "gene_encoder" in k and "embedding.weight" in k:
                emb_key = k
                break
        if emb_key is None:
            # Fallback: look for any large 2D embedding tensor
            for k, v in ckpt.items():
                if hasattr(v, 'shape') and len(
                        v.shape) == 2 and v.shape[0] > 10000:
                    emb_key = k
                    break
        if emb_key is None:
            print(
                "  ERROR: Could not find gene embedding tensor in checkpoint")
            print(f"  Checkpoint keys: {list(ckpt.keys())[:20]}...")
            return None

        emb_weight = ckpt[emb_key].cpu().numpy()
        print(f"  Embedding key: {emb_key}, shape: {emb_weight.shape}")

        # Map gene symbols to embeddings
        emb_dict = {}
        for gene, idx in vocab.items():
            if isinstance(idx, int) and idx < len(emb_weight):
                emb_dict[gene] = emb_weight[idx].astype(np.float32)

        print(f"  Extracted {len(emb_dict)} gene embeddings")

        output = os.path.join(cache_dir, "scgpt_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  scGPT extraction failed: {e}")
        return None


def _load_string_ppi(cache_dir: str):
    """Load STRING PPI network and protein-to-gene mapping.

    Returns (edges_df, protein_to_gene) or (None, None) if files missing.
    """
    import gzip
    links_path = os.path.join(cache_dir, "9606.protein.links.v12.0.txt.gz")
    info_path = os.path.join(cache_dir, "9606.protein.info.v12.0.txt.gz")

    if not os.path.exists(links_path):
        # Try auto-download
        print("  Downloading STRING PPI network...")
        _run_download(
            "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
            links_path)
    if not os.path.exists(info_path):
        print("  Downloading STRING protein info...")
        _run_download(
            "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
            info_path)

    if not os.path.exists(links_path) or not os.path.exists(info_path):
        print("  ERROR: STRING files not found and download failed")
        return None, None

    # Build protein_id -> gene_symbol mapping
    protein_to_gene = {}
    with gzip.open(info_path, "rt") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                protein_to_gene[parts[0]] = parts[1]  # string_id -> symbol

    # Normalize gene symbols through HGNC (consistent with other extractors)
    mapper = get_mapper()
    protein_to_gene = {
        k: mapper.gene_name_normalize(v)
        for k, v in protein_to_gene.items()
    }
    print(f"  STRING protein info: {len(protein_to_gene)} proteins")

    # Load edges (high confidence >= 700)
    edges = []
    with gzip.open(links_path, "rt") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and int(parts[2]) >= 700:
                g1 = protein_to_gene.get(parts[0])
                g2 = protein_to_gene.get(parts[1])
                if g1 and g2 and g1 != g2:
                    edges.append((g1, g2))
    print(f"  High-confidence edges (score >= 700): {len(edges)}")

    return edges, protein_to_gene


def extract_bioconceptvec(cache_dir: str) -> Optional[str]:
    """Extract gene embeddings from BioConceptVec concept_cbow.json.

    BioConceptVec indexes genes as "Gene_NCBI_ID". We map to gene symbols
    using NCBI Homo_sapiens.gene_info.

    Returns path to saved pkl, or None on failure.
    """
    import gzip
    import json as _json

    concept_path = os.path.join(cache_dir, "concept_cbow.json")
    gene_info_path = os.path.join(cache_dir, "Homo_sapiens.gene_info.gz")

    if not os.path.exists(concept_path):
        print("  Downloading BioConceptVec concepts (798MB)...")
        _run_download(
            "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioConceptVec/concept_cbow.json",
            concept_path)
    if not os.path.exists(gene_info_path):
        print("  Downloading NCBI gene info...")
        _run_download(
            "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            gene_info_path)

    if not os.path.exists(concept_path) or not os.path.exists(gene_info_path):
        print("  ERROR: Required files not found")
        return None

    print("  Loading BioConceptVec concepts...")
    try:
        # Build NCBI Gene ID -> gene symbol mapping
        geneid_to_symbol = {}
        with gzip.open(gene_info_path, "rt") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    geneid_to_symbol[parts[1]] = parts[2]  # GeneID -> Symbol
        print(f"  NCBI gene info: {len(geneid_to_symbol)} genes")

        # Load concept embeddings and extract gene concepts
        with open(concept_path, "r") as f:
            concepts = _json.load(f)
        print(f"  Total concepts: {len(concepts)}")

        emb_dict = {}
        for concept_id, vector in concepts.items():
            # BioConceptVec uses "Gene_NCBI_ID" format
            # Skip multi-gene concepts like "Gene_123_456"
            if concept_id.startswith("Gene_") and concept_id.count("_") == 1:
                ncbi_id = concept_id[5:]  # strip "Gene_" prefix
                symbol = geneid_to_symbol.get(ncbi_id)
                if symbol:
                    emb_dict[symbol.upper()] = np.array(vector,
                                                        dtype=np.float32)

        print(f"  Mapped {len(emb_dict)} gene embeddings")

        output = os.path.join(cache_dir, "bioconceptvec_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  BioConceptVec extraction failed: {e}")
        return None


def extract_node2vec_ppi(cache_dir: str) -> Optional[str]:
    """Train Node2Vec on STRING PPI network and extract gene embeddings.

    Requires: node2vec package (pip install node2vec).
    Returns path to saved pkl, or None on failure.
    """
    try:
        import networkx as nx
        from node2vec import Node2Vec
    except ImportError:
        print(
            "  node2vec/networkx not installed (pip install node2vec networkx)"
        )
        return None

    edges, _ = _load_string_ppi(cache_dir)
    if edges is None:
        return None

    print(f"  Building graph...")
    try:
        G = nx.Graph()
        G.add_edges_from(edges)
        print(
            f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        print("  Training Node2Vec (this may take 10-30 minutes)...")
        try:
            n2v = Node2Vec(G,
                           dimensions=128,
                           walk_length=80,
                           num_walks=10,
                           p=1.0,
                           q=1.0,
                           workers=4,
                           quiet=True)
        except TypeError:
            # older node2vec versions don't support quiet
            n2v = Node2Vec(G,
                           dimensions=128,
                           walk_length=80,
                           num_walks=10,
                           p=1.0,
                           q=1.0,
                           workers=4)
        model = n2v.fit(window=10, min_count=1, batch_words=4)

        emb_dict = {}
        for gene in G.nodes():
            emb_dict[gene.upper()] = model.wv[gene].astype(np.float32)

        print(f"  Extracted {len(emb_dict)} gene embeddings (128d)")

        output = os.path.join(cache_dir, "node2vec_ppi_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  Node2Vec training failed: {e}")
        return None


def extract_ppi_svd(cache_dir: str) -> Optional[str]:
    """Build PPI adjacency matrix and reduce via truncated SVD.

    Returns path to saved pkl, or None on failure.
    """
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import svds
    except ImportError:
        print("  scipy not installed (pip install scipy)")
        return None

    edges, _ = _load_string_ppi(cache_dir)
    if edges is None:
        return None

    print("  Building adjacency matrix...")
    try:
        # Get unique genes and create index
        genes = sorted(set(g for e in edges for g in e))
        gene2idx = {g: i for i, g in enumerate(genes)}
        n = len(genes)
        print(f"  Genes: {n}")

        A = lil_matrix((n, n), dtype=np.float32)
        for g1, g2 in edges:
            i, j = gene2idx[g1], gene2idx[g2]
            A[i, j] = 1
            A[j, i] = 1
        A = A.tocsr()

        svd_dim = 256
        print(f"  Running truncated SVD (k={svd_dim})...")
        U, S, _ = svds(A, k=svd_dim)
        embeddings = U * S  # (n, svd_dim)

        emb_dict = {}
        for gene, idx in gene2idx.items():
            emb_dict[gene.upper()] = embeddings[idx].astype(np.float32)

        print(f"  Extracted {len(emb_dict)} gene embeddings ({svd_dim}d)")

        output = os.path.join(cache_dir, "ppi_svd_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  PPI-SVD extraction failed: {e}")
        return None


def extract_prot_t5(cache_dir: str) -> Optional[str]:
    """Generate ProtT5-XL gene embeddings from UniProt sequences.

    Runs the Rostlab/prot_t5_xl_uniref50 encoder on each protein sequence
    and mean-pools over residue positions.

    Requires: transformers, sentencepiece, torch.
    """
    try:
        from transformers import T5Tokenizer, T5EncoderModel
        import torch as _torch
    except ImportError:
        print("  transformers not installed "
              "(pip install transformers sentencepiece)")
        return None

    gene_seqs = _load_protein_sequences(cache_dir)
    if not gene_seqs:
        return None

    print("  Loading ProtT5-XL model (this downloads ~3 GB on first run)...")
    try:
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                                do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        print(f"  Device: {device}")

        from tqdm import tqdm

        emb_dict: Dict[str, np.ndarray] = {}
        genes = sorted(gene_seqs.keys())
        batch_size = 4  # ProtT5 is memory-heavy

        for i in tqdm(range(0, len(genes), batch_size), desc="  ProtT5"):
            batch_genes = genes[i:i + batch_size]
            # ProtT5 expects space-separated amino acids
            sequences = [
                " ".join(list(gene_seqs[g][:1022])) for g in batch_genes
            ]
            ids = tokenizer(sequences,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=1024)
            ids = {k: v.to(device) for k, v in ids.items()}

            with _torch.no_grad():
                output = model(**ids)

            attn = ids["attention_mask"].unsqueeze(-1).float()
            embs = (output.last_hidden_state * attn).sum(1) / attn.sum(1)

            for j, gene in enumerate(batch_genes):
                emb_dict[gene.upper()] = embs[j].cpu().numpy().astype(
                    np.float32)

        print(f"  Generated {len(emb_dict)} ProtT5 embeddings (1024d)")

        output = os.path.join(cache_dir, "prot_t5_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  ProtT5 generation failed: {e}")
        return None


def extract_esm1b(cache_dir: str) -> Optional[str]:
    """Generate ESM-1b gene embeddings from UniProt sequences.

    Same pipeline as ESM-2 but using the predecessor model
    (esm1b_t33_650M_UR50S).  Mean-pooled over residue positions.

    Requires: fair-esm, torch.
    """
    try:
        import esm
        import torch as _torch
    except ImportError:
        print("  fair-esm not installed (pip install fair-esm)")
        return None

    gene_seqs = _load_protein_sequences(cache_dir)
    if not gene_seqs:
        return None

    print("  Loading ESM-1b model...")
    try:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        print(f"  Device: {device}")

        from tqdm import tqdm

        emb_dict: Dict[str, np.ndarray] = {}
        genes = sorted(gene_seqs.keys())
        batch_size = 8

        for i in tqdm(range(0, len(genes), batch_size), desc="  ESM-1b"):
            batch = [(g, gene_seqs[g][:1022]) for g in genes[i:i + batch_size]]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)

            with _torch.no_grad():
                results = model(tokens,
                                repr_layers=[33],
                                return_contacts=False)
            reps = results["representations"][33]

            for j, (gene, seq) in enumerate(batch):
                seq_len = min(len(seq), 1022)
                emb = reps[j, 1:seq_len + 1].mean(0).cpu().numpy()
                emb_dict[gene.upper()] = emb.astype(np.float32)

        print(f"  Generated {len(emb_dict)} ESM-1b embeddings (1280d)")

        output = os.path.join(cache_dir, "esm1b_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  ESM-1b generation failed: {e}")
        return None


def extract_esmc(cache_dir: str) -> Optional[str]:
    """Generate ESM Cambrian (600M) gene embeddings from UniProt sequences.

    Uses Synthyra/ESMplusplus_large — a faithful HuggingFace-compatible
    reimplementation of ESM-C 600M (same weights, same outputs). This avoids
    the EvolutionaryScale ``esm`` package which conflicts with ``fair-esm``
    (both claim the ``esm`` namespace).

    Mean-pooled over residue positions (excluding BOS/EOS tokens) to get
    1152d per gene.

    Requires: transformers, torch, GPU recommended.
    """
    try:
        from transformers import AutoModelForMaskedLM
        import torch as _torch
    except ImportError:
        print("  transformers not installed (pip install transformers)")
        return None

    gene_seqs = _load_protein_sequences(cache_dir)
    if not gene_seqs:
        return None

    print("  Loading ESM-C 600M via Synthyra/ESMplusplus_large...")
    try:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        model = AutoModelForMaskedLM.from_pretrained(
            "Synthyra/ESMplusplus_large", trust_remote_code=True,
        ).to(device)
        model.eval()
        tokenizer = model.tokenizer
        print(f"  Device: {device}")

        # Use built-in embed_dataset for efficient batched mean pooling.
        # Unlike the original EvolutionaryScale ESMC, ESM++ handles batching
        # correctly so we don't need single-sequence processing.
        genes = sorted(gene_seqs.keys())
        sequences = [gene_seqs[g][:2048] for g in genes]

        # Filter out empty sequences — embed_dataset's Pooler asserts
        # non-zero attention masks and will crash on empty strings.
        gene_seq_pairs = [(g, s) for g, s in zip(genes, sequences) if s]
        if len(gene_seq_pairs) < len(genes):
            print(f"  Skipped {len(genes) - len(gene_seq_pairs)} genes "
                  "with empty sequences")
        genes, sequences = zip(*gene_seq_pairs) if gene_seq_pairs else ([], [])

        seq_to_emb = model.embed_dataset(
            sequences=list(sequences),
            tokenizer=tokenizer,
            batch_size=8,
            max_len=2048,
            full_embeddings=False,
            pooling_types=['mean'],
            save=False,
        )

        if seq_to_emb is None:
            print("  embed_dataset returned None")
            return None

        emb_dict: Dict[str, np.ndarray] = {}
        for gene, seq in zip(genes, sequences):
            if seq in seq_to_emb:
                emb = seq_to_emb[seq]
                if isinstance(emb, _torch.Tensor):
                    emb = emb.float().cpu().numpy()
                emb_dict[gene.upper()] = emb.astype(np.float32)

        dim = next(iter(emb_dict.values())).shape[0] if emb_dict else 1152
        print(f"  Generated {len(emb_dict)} ESM-C embeddings ({dim}d)")

        output_path = os.path.join(cache_dir, "esmc_gene_embeddings.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"  ESM-C generation failed: {e}")
        return None


def extract_scprint(cache_dir: str) -> Optional[str]:
    """Extract scPRINT gene embeddings from a pretrained checkpoint.

    scPRINT uses ESM2-derived protein embeddings as gene identity tokens.
    The gene encoder weights are extracted directly from the model.

    Requires: scprint (pip install scprint).
    Checkpoint must be placed manually in cache_dir.
    """
    try:
        from scprint import scPrint
    except ImportError:
        print("  scprint not installed (pip install scprint)")
        return None

    print("  Looking for scPRINT checkpoint...")
    try:
        ckpt_paths = [
            os.path.join(cache_dir, "scprint_checkpoint.ckpt"),
            os.path.join(cache_dir, "scprint", "scprint_checkpoint.ckpt"),
        ]
        ckpt_path = None
        for p in ckpt_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            # Try loading default model (some versions support this)
            try:
                model = scPrint.load_from_checkpoint()
            except Exception:
                print(f"  ERROR: No scPRINT checkpoint found.")
                print(f"  Place scprint_checkpoint.ckpt in {cache_dir}/")
                print("  Download from: https://github.com/cantinilab/scPRINT")
                return None
        else:
            print(f"  Loading from {ckpt_path}...")
            model = scPrint.load_from_checkpoint(ckpt_path)

        model.eval()

        # Extract gene encoder embeddings
        gene_embeddings = model.gene_encoder.weight.detach().cpu().numpy()

        # Get gene vocabulary — attribute name varies by version
        vocab = None
        for attr in ("gene_vocab", "genes", "gene_names", "vocab"):
            if hasattr(model, attr):
                val = getattr(model, attr)
                if isinstance(val, dict):
                    vocab = val
                elif isinstance(val, (list, tuple)):
                    vocab = {g: i for i, g in enumerate(val)}
                break

        if vocab is None:
            print("  WARNING: Cannot determine gene vocabulary from model")
            print(
                "  Consult scPRINT docs for the gene-to-index mapping attribute"
            )
            return None

        emb_dict: Dict[str, np.ndarray] = {}
        for gene, idx in vocab.items():
            if isinstance(idx, int) and idx < len(gene_embeddings):
                emb_dict[gene.upper()] = gene_embeddings[idx].astype(
                    np.float32)

        dim = gene_embeddings.shape[1]
        print(f"  Extracted {len(emb_dict)} scPRINT embeddings ({dim}d)")

        output = os.path.join(cache_dir, "scprint_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  scPRINT extraction failed: {e}")
        return None


def extract_seqvec(cache_dir: str) -> Optional[str]:
    """Generate SeqVec gene embeddings from UniProt sequences.

    Uses the ELMo-based SeqVec model. Embeddings are mean-pooled over
    all 3 ELMo layers and all residue positions -> 1024d per gene.

    Requires: allennlp (pip install allennlp).
    Model weights are auto-downloaded from rostlab.org.
    """
    try:
        from allennlp.commands.elmo import ElmoEmbedder
    except ImportError:
        print("  allennlp not installed (pip install allennlp)")
        return None

    gene_seqs = _load_protein_sequences(cache_dir)
    if not gene_seqs:
        return None

    # Download SeqVec model weights
    weights_path = os.path.join(cache_dir, "seqvec_weights.hdf5")
    options_path = os.path.join(cache_dir, "seqvec_options.json")

    if not os.path.exists(weights_path):
        print("  Downloading SeqVec weights (~400 MB)...")
        _run_download(
            "https://rostlab.org/~deepppi/seqvec/uniref50_v2/weights.hdf5",
            weights_path)
    if not os.path.exists(options_path):
        print("  Downloading SeqVec options...")
        _run_download(
            "https://rostlab.org/~deepppi/seqvec/uniref50_v2/options.json",
            options_path)

    if not os.path.exists(weights_path) or not os.path.exists(options_path):
        print("  ERROR: Could not download SeqVec model files")
        return None

    print("  Loading SeqVec (ELMo) model...")
    try:
        import torch as _torch
        cuda_device = 0 if _torch.cuda.is_available() else -1
        embedder = ElmoEmbedder(options_path,
                                weights_path,
                                cuda_device=cuda_device)
        print(f"  Device: {'cuda:0' if cuda_device == 0 else 'cpu'}")

        from tqdm import tqdm

        emb_dict: Dict[str, np.ndarray] = {}
        genes = sorted(gene_seqs.keys())

        for gene in tqdm(genes, desc="  SeqVec"):
            seq = gene_seqs[gene][:1022]
            embedding = embedder.embed_sentence(list(seq))
            # embedding shape: (3, seq_len, 1024) — 3 ELMo layers
            emb = np.mean(embedding, axis=(0, 1))
            emb_dict[gene.upper()] = emb.astype(np.float32)

        print(f"  Generated {len(emb_dict)} SeqVec embeddings (1024d)")

        output = os.path.join(cache_dir, "seqvec_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  SeqVec generation failed: {e}")
        return None


def extract_text_embed(cache_dir: str) -> Optional[str]:
    """Generate gene embeddings from NCBI gene descriptions using mxbai.

    Open-source GenePT alternative: embeds NCBI gene functional descriptions
    using mixedbread-ai/mxbai-embed-large-v1 (335M params, 1024d output).

    No API costs, no SL data leakage (uses short functional descriptions,
    not full literature summaries that might mention SL relationships).

    Requires: sentence-transformers (pip install sentence-transformers).
    Gene descriptions are auto-fetched from NCBI gene_info.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed "
              "(pip install sentence-transformers)")
        return None

    import gzip as _gzip
    import json as _json

    # cache_dir must already exist (server does not allow mkdir)
    summaries_path = os.path.join(cache_dir, "ncbi_gene_summaries.json")

    if not os.path.exists(summaries_path):
        print("  Fetching NCBI gene descriptions...")
        gene_info_path = os.path.join(cache_dir, "Homo_sapiens.gene_info.gz")
        if not os.path.exists(gene_info_path):
            _run_download(
                "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/"
                "Mammalia/Homo_sapiens.gene_info.gz", gene_info_path)

        if not os.path.exists(gene_info_path):
            print("  ERROR: Could not download NCBI gene_info")
            return None

        summaries: Dict[str, str] = {}
        with _gzip.open(gene_info_path, "rt") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                symbol = parts[2]
                description = parts[8] if parts[8] != "-" else ""
                other = parts[13] if len(
                    parts) > 13 and parts[13] != "-" else ""
                text = f"{symbol}: {description}"
                if other:
                    text += f". Also known as: {other}"
                summaries[symbol] = text

        with open(summaries_path, "w") as f:
            _json.dump(summaries, f)
        print(f"  Saved {len(summaries)} gene descriptions")
    else:
        with open(summaries_path, "r") as f:
            summaries = _json.load(f)
        print(f"  Loaded {len(summaries)} gene descriptions from cache")

    print("  Loading mxbai-embed-large-v1 model...")
    try:
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

        genes = sorted(summaries.keys())
        texts = [summaries[g] for g in genes]

        print(f"  Encoding {len(texts)} gene descriptions...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        emb_dict: Dict[str, np.ndarray] = {}
        for i, gene in enumerate(genes):
            emb_dict[gene.upper()] = embeddings[i].astype(np.float32)

        dim = embeddings.shape[1]
        print(f"  Generated {len(emb_dict)} text embeddings ({dim}d)")

        output = os.path.join(cache_dir, "text_embed_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  Text embedding generation failed: {e}")
        return None


def extract_go2vec(cache_dir: str) -> Optional[str]:
    """Generate GO2Vec gene embeddings: Node2Vec on GO DAG, mean-pooled per gene.

    Trains Node2Vec on the Gene Ontology directed acyclic graph, then for each
    gene, averages the learned GO term vectors for all its annotated terms.

    Requires: node2vec, networkx, obonet.
    GO OBO + human GAF are auto-downloaded.
    """
    try:
        import networkx as nx
        from node2vec import Node2Vec
    except ImportError:
        print("  node2vec/networkx not installed "
              "(pip install node2vec networkx obonet)")
        return None

    go_graph, gene_to_go = _load_go_graph_and_annotations(cache_dir)
    if go_graph is None or not gene_to_go:
        print("  ERROR: Could not load GO graph or annotations")
        return None

    print("  Training Node2Vec on GO graph...")
    try:
        G = go_graph.to_undirected()
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
        print(f"  GO graph (undirected): {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

        try:
            n2v = Node2Vec(G,
                           dimensions=128,
                           walk_length=30,
                           num_walks=20,
                           workers=4,
                           quiet=True)
        except TypeError:
            n2v = Node2Vec(G,
                           dimensions=128,
                           walk_length=30,
                           num_walks=20,
                           workers=4)
        model = n2v.fit(window=10, min_count=1, batch_words=4)

        emb_dict: Dict[str, np.ndarray] = {}
        for gene, go_terms in gene_to_go.items():
            vectors = [model.wv[t] for t in go_terms if t in model.wv]
            if vectors:
                emb_dict[gene.upper()] = np.mean(vectors,
                                                 axis=0).astype(np.float32)

        print(f"  Generated {len(emb_dict)} GO2Vec embeddings (128d)")

        output = os.path.join(cache_dir, "go2vec_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  GO2Vec generation failed: {e}")
        return None


def extract_onto2vec(cache_dir: str) -> Optional[str]:
    """Generate Onto2Vec gene embeddings: Word2Vec on GO axiom sentences.

    Converts GO graph edges and node attributes into "sentences", trains
    Word2Vec (skip-gram), then mean-pools GO term embeddings per gene.

    Requires: gensim, obonet.
    GO OBO + human GAF are auto-downloaded.
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  gensim not installed (pip install gensim obonet)")
        return None

    go_graph, gene_to_go = _load_go_graph_and_annotations(cache_dir)
    if go_graph is None or not gene_to_go:
        print("  ERROR: Could not load GO graph or annotations")
        return None

    print("  Converting GO axioms to sentences...")
    try:
        sentences: List[List[str]] = []

        # Node identity sentences: [GO_id, word1, word2, ...]
        for node in go_graph.nodes():
            name = go_graph.nodes[node].get("name", "")
            tokens = [node] + name.lower().split()
            sentences.append(tokens)

        # Edge sentences: [source, relation, target]
        # obonet returns a MultiDiGraph; relationship type is the edge key
        for u, v, key in go_graph.edges(keys=True):
            rel = str(key)
            sentences.append([u, rel, v])
            # Extended with human-readable names
            u_name = go_graph.nodes[u].get("name", "")
            v_name = go_graph.nodes[v].get("name", "")
            if u_name and v_name:
                sentences.append(u_name.lower().split() + [rel] +
                                 v_name.lower().split())

        # Gene-GO annotation sentences
        for gene, terms in gene_to_go.items():
            for term in terms:
                sentences.append([gene, "annotated_with", term])

        print(f"  Generated {len(sentences)} sentences from GO axioms")

        print("  Training Word2Vec (skip-gram)...")
        w2v = Word2Vec(sentences,
                       vector_size=128,
                       window=5,
                       min_count=1,
                       workers=4,
                       epochs=10,
                       sg=1)

        emb_dict: Dict[str, np.ndarray] = {}
        for gene, go_terms in gene_to_go.items():
            vectors = [w2v.wv[t] for t in go_terms if t in w2v.wv]
            if vectors:
                emb_dict[gene.upper()] = np.mean(vectors,
                                                 axis=0).astype(np.float32)

        print(f"  Generated {len(emb_dict)} Onto2Vec embeddings (128d)")

        output = os.path.join(cache_dir, "onto2vec_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  Onto2Vec generation failed: {e}")
        return None


def extract_mashup(cache_dir: str) -> Optional[str]:
    """Generate Mashup embeddings via diffusion kernel on STRING PPI.

    Implements the Mashup algorithm (Cho et al., 2016) for a single network:
    1. Compute normalized adjacency eigenvectors (spectral decomposition)
    2. Apply diffusion kernel weighting: alpha / (1 - (1-alpha)*lambda)
    3. Log-transform the kernel weights
    4. Result: 500-dim embedding per gene

    This is the spectral equivalent of the RWR + log + SVD pipeline from
    the original paper, but more memory-efficient for large networks.

    Requires: scipy.  STRING PPI is auto-downloaded.
    """
    try:
        from scipy.sparse import lil_matrix, diags
        from scipy.sparse.linalg import eigsh
    except ImportError:
        print("  scipy not installed (pip install scipy)")
        return None

    edges, _ = _load_string_ppi(cache_dir)
    if edges is None:
        return None

    print("  Building Mashup diffusion embeddings...")
    try:
        genes = sorted(set(g for e in edges for g in e))
        gene2idx = {g: i for i, g in enumerate(genes)}
        n = len(genes)
        print(f"  Genes: {n}")

        # Build sparse adjacency
        A = lil_matrix((n, n), dtype=np.float32)
        for g1, g2 in edges:
            i, j = gene2idx[g1], gene2idx[g2]
            A[i, j] = 1
            A[j, i] = 1
        A = A.tocsr()

        # Normalized adjacency: D^{-1/2} A D^{-1/2}
        degrees = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-10))
        D_inv_sqrt = diags(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        svd_dim = min(500, n - 2)
        alpha = 0.5

        print(f"  Computing spectral decomposition (k={svd_dim})...")
        eigenvalues, eigenvectors = eigsh(A_norm, k=svd_dim, which='LM')

        # Diffusion kernel weights: alpha / (1 - (1-alpha) * lambda)
        # Then log-transform as in original Mashup
        diff_weights = alpha / np.maximum(1.0 -
                                          (1.0 - alpha) * eigenvalues, 1e-10)
        diff_weights = np.log(np.maximum(diff_weights, 1e-10))

        embeddings = (eigenvectors * diff_weights[np.newaxis, :]).astype(
            np.float32)

        emb_dict: Dict[str, np.ndarray] = {}
        for gene, idx in gene2idx.items():
            emb_dict[gene.upper()] = embeddings[idx]

        print(f"  Generated {len(emb_dict)} Mashup embeddings ({svd_dim}d)")

        output = os.path.join(cache_dir, "mashup_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  Mashup generation failed: {e}")
        return None


def extract_ppi_raw(cache_dir: str) -> Optional[str]:
    """Build high-dimensional SVD of PPI adjacency matrix (1024d).

    Same as ppi_svd but retains more dimensions (1024 vs 256) to preserve
    more of the raw adjacency structure.  This is the strong "PPI-RAW"
    baseline from Zhong et al. (2025): AUROC SL=0.79, NG=0.73, TF=0.74.

    Requires: scipy.  STRING PPI is auto-downloaded.
    """
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import svds
    except ImportError:
        print("  scipy not installed (pip install scipy)")
        return None

    edges, _ = _load_string_ppi(cache_dir)
    if edges is None:
        return None

    print("  Building high-dim PPI adjacency SVD...")
    try:
        genes = sorted(set(g for e in edges for g in e))
        gene2idx = {g: i for i, g in enumerate(genes)}
        n = len(genes)
        print(f"  Genes: {n}")

        A = lil_matrix((n, n), dtype=np.float32)
        for g1, g2 in edges:
            i, j = gene2idx[g1], gene2idx[g2]
            A[i, j] = 1
            A[j, i] = 1
        A = A.tocsr()

        svd_dim = min(1024, n - 1)
        print(f"  Running truncated SVD (k={svd_dim})...")
        U, S, _ = svds(A, k=svd_dim)
        embeddings = (U * S).astype(np.float32)

        emb_dict: Dict[str, np.ndarray] = {}
        for gene, idx in gene2idx.items():
            emb_dict[gene.upper()] = embeddings[idx]

        print(f"  Generated {len(emb_dict)} PPI-RAW embeddings ({svd_dim}d)")

        output = os.path.join(cache_dir, "ppi_raw_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  PPI-RAW SVD failed: {e}")
        return None


def extract_kg_complex(cache_dir: str) -> Optional[str]:
    """Train ComplEx KG embeddings on a biological knowledge graph.

    Builds a heterogeneous knowledge graph from:
    1. STRING PPI edges -> (gene, interacts_with, gene) triples
    2. GO annotations   -> (gene, annotated_with, GO_term) triples

    Trains ComplEx (Trouillon et al., 2016) using PyKEEN and extracts
    the learned gene entity embeddings (256d).

    Requires: pykeen (pip install pykeen).
    STRING PPI + GO annotations are auto-downloaded.
    """
    try:
        from pykeen.pipeline import pipeline as pykeen_pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        print("  pykeen not installed (pip install pykeen)")
        return None

    # Load STRING PPI
    edges, _ = _load_string_ppi(cache_dir)
    if edges is None:
        return None

    # Load GO annotations
    _, gene_to_go = _load_go_graph_and_annotations(cache_dir)

    print("  Building biological knowledge graph...")
    try:
        triples: List[List[str]] = []

        # PPI triples: (gene, interacts_with, gene)
        for g1, g2 in edges:
            triples.append([g1.upper(), "interacts_with", g2.upper()])

        ppi_count = len(triples)
        print(f"  PPI triples: {ppi_count}")

        # GO annotation triples: (gene, annotated_with, GO_term)
        if gene_to_go:
            for gene, go_terms in gene_to_go.items():
                for term in go_terms:
                    triples.append([gene.upper(), "annotated_with", term])
            go_count = len(triples) - ppi_count
            print(f"  GO triples: {go_count}")

        print(f"  Total triples: {len(triples)}")

        triples_array = np.array(triples, dtype=str)
        tf = TriplesFactory.from_labeled_triples(triples_array)

        # Use all triples for training so every entity gets an embedding.
        # A 90/10 split would leave ~10% of entities only in the test set,
        # and PyKEEN doesn't learn embeddings for test-only entities.
        # We only care about the learned embeddings, not evaluation metrics.
        training_tf, testing_tf = tf.split([0.99, 0.01], random_state=42)

        print("  Training ComplEx (this may take 10-60 minutes)...")
        import torch as _torch
        result = pykeen_pipeline(
            training=training_tf,
            testing=testing_tf,
            model="ComplEx",
            model_kwargs=dict(embedding_dim=256),
            training_kwargs=dict(num_epochs=100, batch_size=4096),
            random_seed=42,
            device="cuda" if _torch.cuda.is_available() else "cpu",
        )

        # Extract gene entity embeddings (real ⊕ imag for ComplEx)
        model = result.model
        entity_to_id = training_tf.entity_to_id

        emb_dict: Dict[str, np.ndarray] = {}
        for entity, eid in entity_to_id.items():
            # Only keep gene entities (skip GO terms)
            if not entity.startswith("GO:"):
                emb = model.entity_representations[0](_torch.tensor(
                    [eid])).detach().cpu().numpy().flatten()
                # ComplEx returns a complex-dtype array of length embedding_dim.
                # Concatenate [real, imag] to preserve all learned information
                # (final dim = 2 * embedding_dim = 512).
                emb_dict[entity] = np.concatenate(
                    [emb.real, emb.imag]).astype(np.float32)

        dim = len(next(iter(emb_dict.values())))
        print(f"  Generated {len(emb_dict)} KG-ComplEx embeddings ({dim}d)")

        output = os.path.join(cache_dir, "kg_complex_gene_embeddings.pkl")
        with open(output, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved: {output}")
        return output

    except Exception as e:
        print(f"  KG-ComplEx training failed: {e}")
        return None


def print_download_instructions(embedding_type: str, cache_dir: str) -> None:
    """Print instructions for manually obtaining a file that can't be auto-downloaded."""
    info = EMBEDDING_REGISTRY[embedding_type]
    print(f"\n{'=' * 60}")
    print(f"  Cannot auto-download: {embedding_type}")
    print(f"{'=' * 60}")
    print(f"  Source : {info['source']}")
    print(f"  Place any of these files in {cache_dir}/:")
    filenames = [info["filename"]] + info.get("alt_filenames", [])
    for fname in filenames:
        print(f"    - {fname}")
    if embedding_type == "geneformer":
        print()
        print("  To extract from HuggingFace (requires transformers):")
        print("    from transformers import AutoModel")
        print('    model = AutoModel.from_pretrained("ctheodoris/Geneformer")')
        print(
            "    emb = model.embeddings.word_embeddings.weight.detach().numpy()"
        )
        print(
            "    # Then map token indices to gene names via token_dictionary.pkl"
        )
    elif embedding_type == "scgpt":
        print()
        print("  To download scGPT checkpoint from Google Drive:")
        print("    pip install gdown")
        print(
            "    gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
        )
        print(
            "  Then extract gene embeddings from best_model.pt + vocab.json:")
        print("    import torch, json")
        print("    vocab = json.load(open('vocab.json'))")
        print("    ckpt = torch.load('best_model.pt', map_location='cpu')")
        print(
            "    emb = ckpt['encoder.embedding.gene_encoder.embedding.weight']"
        )
        print("    gene_emb = {g: emb[i].numpy() for g, i in vocab.items()}")
    elif embedding_type == "genept":
        print()
        print(
            "  NOTE: GenePT (Data Leakage with SL) -- use with caution for SL tasks"
        )
    elif embedding_type == "scprint":
        print()
        print("  scPRINT requires a checkpoint file placed manually:")
        print(f"    {cache_dir}/scprint_checkpoint.ckpt")
        print("  Download from: https://github.com/cantinilab/scPRINT")
    elif embedding_type in ("prot_t5", "esm1b", "esmc", "seqvec"):
        pkgs = {
            "prot_t5": "transformers sentencepiece",
            "esm1b": "fair-esm",
            "esmc": "transformers",
            "seqvec": "allennlp",
        }
        print()
        print(f"  Requires: pip install {pkgs[embedding_type]}")
        print("  Also needs GPU for practical runtime (~20k proteins).")
        print("  Protein sequences are fetched from NCBI on first run.")
    elif embedding_type == "text_embed":
        print()
        print("  Requires: pip install sentence-transformers")
        print("  NCBI gene descriptions are auto-downloaded.")
    elif embedding_type in ("go2vec", "onto2vec"):
        pkgs = {
            "go2vec": "node2vec networkx obonet",
            "onto2vec": "gensim obonet",
        }
        print()
        print(f"  Requires: pip install {pkgs[embedding_type]}")
        print("  GO OBO + human GAF are auto-downloaded.")
    elif embedding_type == "kg_complex":
        print()
        print("  Requires: pip install pykeen")
        print(
            "  Builds KG from STRING PPI + GO annotations (auto-downloaded).")
    print(f"{'=' * 60}\n")


# =============================================================================
# Generate a single embedding type
# =============================================================================


def generate_embedding(
    embedding_type: str,
    gene_list: List[str],
    output_path: str,
    cache_dir: str,
    precomputed_file: Optional[str] = None,
) -> None:
    """Load a precomputed embedding, align to *gene_list*, and save as ``.pt``."""
    # Make gene_list available for NCBI sequence fetching
    global _ACTIVE_GENE_LIST
    _ACTIVE_GENE_LIST = gene_list

    info = EMBEDDING_REGISTRY[embedding_type]

    label = info["description"]
    print(f"\nGenerating {embedding_type} embeddings")
    print(f"  Description : {label}")
    print(f"  Dimension   : {info['dim'] or 'auto-detect'}")
    print(f"  Source      : {info['source']}")

    # Locate the precomputed file: check cache, then try auto-download
    if precomputed_file and os.path.exists(precomputed_file):
        filepath = precomputed_file
    else:
        filepath = find_precomputed_file(embedding_type, cache_dir)

    if filepath is None:
        # Try auto-download (gene2vec, genept have direct URLs)
        filepath = download_precomputed(embedding_type, cache_dir)

    if filepath is None and embedding_type == "geneformer":
        # Try extracting from HuggingFace model
        filepath = extract_geneformer_from_hf(cache_dir)

    if filepath is None and embedding_type == "scgpt":
        filepath = extract_scgpt_from_checkpoint(cache_dir)

    if filepath is None and embedding_type == "bioconceptvec":
        filepath = extract_bioconceptvec(cache_dir)

    if filepath is None and embedding_type == "node2vec_ppi":
        filepath = extract_node2vec_ppi(cache_dir)

    if filepath is None and embedding_type == "ppi_svd":
        filepath = extract_ppi_svd(cache_dir)

    if filepath is None and embedding_type == "prot_t5":
        filepath = extract_prot_t5(cache_dir)

    if filepath is None and embedding_type == "esm1b":
        filepath = extract_esm1b(cache_dir)

    if filepath is None and embedding_type == "esmc":
        filepath = extract_esmc(cache_dir)

    if filepath is None and embedding_type == "scprint":
        filepath = extract_scprint(cache_dir)

    if filepath is None and embedding_type == "seqvec":
        filepath = extract_seqvec(cache_dir)

    if filepath is None and embedding_type == "text_embed":
        filepath = extract_text_embed(cache_dir)

    if filepath is None and embedding_type == "go2vec":
        filepath = extract_go2vec(cache_dir)

    if filepath is None and embedding_type == "onto2vec":
        filepath = extract_onto2vec(cache_dir)

    if filepath is None and embedding_type == "mashup":
        filepath = extract_mashup(cache_dir)

    if filepath is None and embedding_type == "ppi_raw":
        filepath = extract_ppi_raw(cache_dir)

    if filepath is None and embedding_type == "kg_complex":
        filepath = extract_kg_complex(cache_dir)

    if filepath is None:
        print_download_instructions(embedding_type, cache_dir)
        sys.exit(1)

    print(f"  Loading from: {filepath}")

    # Load raw embeddings dict
    loader = _LOADERS[info["format"]]
    emb_dict = loader(filepath)
    print(f"  Loaded {len(emb_dict)} gene embeddings from file")

    # Infer actual dimension from data
    if not emb_dict:
        print(f"  ERROR: No gene embeddings found in {filepath}")
        sys.exit(1)
    sample_vec = next(iter(emb_dict.values()))
    actual_dim = sample_vec.shape[0]
    expected_dim = info["dim"]
    if expected_dim is not None and actual_dim != expected_dim:
        print(f"  Note: expected dim={expected_dim}, found dim={actual_dim}")
    print(f"  Actual dim  : {actual_dim}")

    # Align to gene list
    embeddings, stats = align_embeddings(emb_dict, gene_list, actual_dim,
                                         embedding_type)

    # Save in standard format (compatible with data_loader.py)
    output = {
        "embeddings": embeddings,
        "raw_embeddings": embeddings.clone(
        ),  # kept for compatibility; data_loader uses raw_embeddings
        "gene_order": gene_list,
        "embedding_type": embedding_type,
        "embedding_dim": actual_dim,
        "num_genes": len(gene_list),
        "num_genes_with_embedding": stats["matched"],
        "coverage_pct": stats["coverage_pct"],
        "source": info["source"],
    }

    if embedding_type == "genept":
        output["WARNING"] = (
            "GenePT (Data Leakage with SL): literature embeddings "
            "may encode known SL relationships")

    # output parent dir must already exist (server does not allow mkdir)
    torch.save(output, output_path)
    print(f"\n  Saved to : {output_path}")
    print(f"  Shape    : {embeddings.shape}")


# =============================================================================
# Concatenation
# =============================================================================


def concatenate_embeddings(input_paths: List[str], output_path: str) -> None:
    """Concatenate multiple ``.pt`` embedding files along the feature axis.

    All files must share the same ``gene_order``.  Missing genes (NaN)
    in individual embeddings are preserved; ``data_loader.py`` handles
    per-dimension Huber imputation at load time.
    """
    print(f"\nConcatenating {len(input_paths)} embedding files ...")

    all_tensors: List[torch.Tensor] = []
    gene_order: Optional[List[str]] = None
    types: List[str] = []
    dims: List[int] = []

    for path in input_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)

        current_order = data.get("gene_order")
        if current_order is None:
            raise ValueError(f"No gene_order in {path}")
        current_order = list(current_order)  # normalize tuple/list

        if gene_order is None:
            gene_order = current_order
        elif current_order != gene_order:
            raise ValueError(
                f"Gene order mismatch between {input_paths[0]} and {path}. "
                "All embeddings must be aligned to the same gene list.")

        emb = data["raw_embeddings"] if "raw_embeddings" in data else data[
            "embeddings"]
        all_tensors.append(emb)

        emb_type = data.get("embedding_type",
                            Path(path).stem.replace("all_genes_", ""))
        types.append(emb_type)
        dim = emb.shape[1]
        dims.append(dim)

        print(f"  {path}: {emb_type} ({dim}d)")

    combined = torch.cat(all_tensors, dim=1)
    combined_type = "+".join(types)
    total_dim = combined.shape[1]

    print(f"  -> Combined: {combined_type} ({total_dim}d)")

    has_leakage = any("genept" in t.lower() for t in types)

    output = {
        "embeddings": combined,
        "raw_embeddings": combined.clone(),
        "gene_order": gene_order,
        "embedding_type": combined_type,
        "embedding_dim": total_dim,
        "component_types": types,
        "component_dims": dims,
        "num_genes": len(gene_order),
    }

    if has_leakage:
        output["WARNING"] = (
            "Contains GenePT (Data Leakage with SL): literature embeddings "
            "may encode known SL relationships")

    # output parent dir must already exist (server does not allow mkdir)
    torch.save(output, output_path)
    print(f"\n  Saved to : {output_path}")
    print(f"  Shape    : {combined.shape}")


# =============================================================================
# CLI
# =============================================================================


def cmd_list() -> None:
    """Print available embedding types."""
    print("\nSupported embedding types (precomputed):")
    print("=" * 70)
    for name, info in EMBEDDING_REGISTRY.items():
        print(f"  {name:<12} {info['description']}")
        print(f"  {'':<12} Source  : {info['source']}")
        print(f"  {'':<12} File    : {info['filename']}")
        if info.get("notes"):
            print(f"  {'':<12} Notes   : {info['notes']}")
        print()
    print("Already available (via dedicated scripts):")
    for name, desc in EXISTING_EMBEDDINGS.items():
        print(f"  {name:<12} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate / concatenate gene embeddings from precomputed sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # Generate geneformer embeddings aligned to the ESM gene list
  python generate_embeddings.py --type geneformer \\
      --gene_list ../data/all_genes_esm2.pt \\
      --output ../data/all_genes_geneformer.pt

  # Concatenate geneformer + GO into a single file
  python generate_embeddings.py --concat \\
      ../data/all_genes_geneformer.pt \\
      ../data/all_genes_go.pt \\
      --output ../data/all_genes_geneformer_go.pt

  # List available embedding types and download instructions
  python generate_embeddings.py --list
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--type",
        choices=list(EMBEDDING_REGISTRY.keys()),
        help="Embedding type to generate",
    )
    mode.add_argument(
        "--concat",
        nargs="+",
        metavar="FILE",
        help="Concatenate multiple .pt files along feature axis",
    )
    mode.add_argument(
        "--list",
        action="store_true",
        help="List available embedding types",
    )

    parser.add_argument(
        "--gene_list",
        type=str,
        help=
        "Gene list source (.pt with gene_order, or .txt one gene per line)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../data/embeddings_cache",
        help=
        "Directory containing precomputed embedding files (default: ../data/embeddings_cache)",
    )
    parser.add_argument(
        "--precomputed_file",
        type=str,
        default=None,
        help="Direct path to precomputed file (overrides cache_dir lookup)",
    )

    args = parser.parse_args()

    # --list mode
    if args.list:
        cmd_list()
        return

    # --concat mode
    if args.concat:
        if not args.output:
            parser.error("--output is required for --concat")
        concatenate_embeddings(args.concat, args.output)
        return

    # --type mode
    if not args.gene_list:
        parser.error("--gene_list is required for --type")
    if not args.output:
        parser.error("--output is required for --type")

    gene_list = load_gene_list(args.gene_list)
    print(f"Gene list: {len(gene_list)} genes from {args.gene_list}")

    generate_embedding(
        embedding_type=args.type,
        gene_list=gene_list,
        output_path=args.output,
        cache_dir=args.cache_dir,
        precomputed_file=args.precomputed_file,
    )


if __name__ == "__main__":
    main()
