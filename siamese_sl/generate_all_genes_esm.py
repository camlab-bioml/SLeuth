#!/usr/bin/env python3
"""
Generate ESM embeddings for ALL human protein-coding genes.

This script:
  1. Resolves one canonical protein sequence per Entrez Gene ID
  2. Generates ESM-2 embeddings using Pool PaRTI (PageRank-based pooling)
  3. Saves embeddings with gene order for reproducibility

Sequence source (resolve_protein_sequences, shared with generate_embeddings.py):
the UniProt reviewed canonical sequence, reached through HGNC entrez_id ->
uniprot_ids (both files tracked in git under data/cache); NCBI RefSeq
(NP_ preferred) only for the ~0.5% of genes UniProt does not cover. The chosen
accession per gene is stored in the output as ``protein_accessions``.

Pool PaRTI Reference:
  Tartici et al. "Pool PaRTI: a PageRank-based pooling method for identifying
  critical residues and enhancing protein sequence representations"
  Bioinformatics, 2025. https://github.com/Helix-Research-Lab/Pool_PaRTI

Requirements:
  pip install fair-esm requests tqdm networkx   (requests: NCBI fallback only)

Usage (a gene list is required: --gene_list or --sl_path):
    # Basic usage with Pool PaRTI (default)
    python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \\
        --output ../data/all_genes_esm2.pt

    # Use mean pooling instead. NOT the pipeline build (see PIPELINE_ESM), so
    # it is stamped a marked-down generator_version that embedding_versions.py
    # reports as stale — write it beside the catalog file, never over it.
    python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \\
        --pooling mean --output ../data/all_genes_esm2_mean.pt

    # GPU with larger batch size
    python generate_all_genes_esm.py --gene_list ../data/gene_universe.txt \\
        --device cuda:0 --batch_size 16 --output out.pt
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm

try:
    import networkx as nx
except ImportError:
    nx = None  # Will be checked if pool_parti is used

# Canonical-protein sources. Both files are tracked in git (data/cache and
# data/embeddings_cache), so this resolution is offline and identical in every
# run; NCBI is asked only about the few genes they do not cover.
UNIPROT_FASTA_NAME = "human_proteome_reviewed.fasta"  # reviewed canonical set
HGNC_TABLE_NAME = "hgnc_complete_set.txt"  # entrez_id -> uniprot_ids
SEQUENCE_CACHE_NAME = "protein_sequences_canonical.json"
# Part of the kept sequence cache's key. Bump it with every change to the
# resolver rules (resolve_protein_sequences, _pick_uniprot, _refseq_rank,
# _ncbi_refseq_fallback): the cache survives regeneration and is otherwise
# keyed only by its source files and gene set, so a bump of
# EMBEDDING_GENERATOR_VERSION alone would re-embed the old sequences under the
# new stamp. (Not keyed by that version: an unrelated bump would re-query the
# NCBI fallback, whose answers change between NCBI releases.)
SEQUENCE_CACHE_FORMAT = "canonical_protein_v1"
SEQUENCE_SOURCE = (
    "UniProt reviewed canonical sequence (HGNC entrez_id -> uniprot_ids; "
    "GN= symbol match breaks ties / fills gaps); NCBI RefSeq fallback "
    "(NP_ > YP_ > XP_, then longest)")
# Stamped into every generated embedding .pt (here, generate_embeddings.py and
# generate_go_esm_embeddings.py) so a set built by older code, e.g. on the
# pre-2026-09-22 newest-RefSeq-isoform sequences or the residue-only Pool
# PaRTI graph, can be told apart. Bump it whenever a generator change alters
# its output (a resolver change also bumps SEQUENCE_CACHE_FORMAT).
EMBEDDING_GENERATOR_VERSION = "2026-10-07-source-symbol-pins"
# The ESM build the pipeline (slurm/run_generate_embeddings.sh STEP 1) and the
# paper mean by data/all_genes_esm2.pt: Pool PaRTI on ESM-2 650M. --pooling and
# --model change the vectors without changing the code, so the same
# EMBEDDING_GENERATOR_VERSION would otherwise describe a mean-pooled (or
# other-model) file; main() marks the stamp down instead, as generate_embeddings
# .py does for a reused cache, and embedding_versions.py reads it as stale.
PIPELINE_ESM = ("pool_parti", "esm2_t33_650M_UR50D")

# NCBI E-utilities base URLs (fallback only)
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_ATTEMPTS = 5


def _find_source_file(name: str, cache_dir: str) -> Optional[Path]:
    """First non-empty ``name`` in cache_dir or its sibling data/cache and
    data/embeddings_cache directories (the SLURM CACHE_DIR is
    data/embeddings_cache; the HGNC table lives in data/cache)."""
    base = Path(cache_dir)
    for d in (base, base.parent / "cache", base.parent / "embeddings_cache"):
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _file_md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_uniprot_fasta(path: Path) -> Dict[str, Tuple[str, str]]:
    """Parse a UniProt FASTA into {accession: (GN= symbol upper-cased, seq)}.

    ``GN`` is "" for entries without a gene name.
    """
    records: Dict[str, Tuple[str, List[str]]] = {}
    acc = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) > 2 else parts[0].split()[0]
                gn = ""
                if " GN=" in line:
                    gn = line.split(" GN=", 1)[1].split()[0].upper()
                records[acc] = (gn, [])
            elif line and acc is not None:
                records[acc][1].append(line)
    return {a: (gn, "".join(chunks)) for a, (gn, chunks) in records.items()}


def _parse_hgnc_uniprot(path: Path) -> Dict[str, Tuple[str, List[str]]]:
    """Parse the HGNC complete set into {entrez_id: (SYMBOL, [uniprot_ids])}."""
    import csv
    table: Dict[str, Tuple[str, List[str]]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = ({"entrez_id", "symbol", "uniprot_ids"} -
                   set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path}: missing HGNC columns {sorted(missing)}")
        for row in reader:
            eid = (row.get("entrez_id") or "").strip()
            if not eid:
                continue
            raw = (row.get("uniprot_ids") or "").strip().strip('"')
            accs = [a.strip().strip('"') for a in raw.split("|")]
            table[eid] = ((row.get("symbol") or "").strip().upper(),
                          [a for a in accs if a])
    return table


def _pick_uniprot(accs: List[str], symbol: str,
                  records: Dict[str, Tuple[str, str]]) -> Optional[str]:
    """One reviewed accession out of ``accs``: the one whose GN= is the HGNC
    symbol, else the longest (then the smallest accession, for determinism).
    """
    cands = sorted({a for a in accs if a in records})
    if not cands:
        return None
    named = [a for a in cands if records[a][0] == symbol]
    return min(named or cands, key=lambda a: (-len(records[a][1]), a))


def _refseq_rank(accession: str, length: int) -> Tuple[int, int, str]:
    """Sort key for RefSeq proteins: curated NP_ first, then YP_ (organelle
    genome), then predicted XP_, then anything else; longest first."""
    prefix = {"NP_": 0, "YP_": 1, "XP_": 2}.get(accession[:3], 3)
    return (prefix, -length, accession)


def _ncbi_request(url: str, params, what: str, parse):
    """POST to an NCBI E-utility and ``parse`` the response; on any failure
    (network, HTTP status, or a response ``parse`` rejects) retry with
    backoff, then RAISE.

    The old loops printed and skipped a failed batch, which silently turned
    up to 200 coding genes into NaN rows (later imputed to one shared vector)
    and cached that gap for every later protein model.
    """
    import time
    import requests
    params = list(params.items()) if isinstance(params, dict) else list(params)
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params.append(("api_key", api_key))
    last_err: Optional[Exception] = None
    for attempt in range(NCBI_ATTEMPTS):
        try:
            resp = requests.post(url, data=params, timeout=120)
            resp.raise_for_status()
            out = parse(resp)
            time.sleep(0.35)  # stay under NCBI's 3 requests/s without a key
            return out
        except Exception as e:
            last_err = e
            if attempt < NCBI_ATTEMPTS - 1:
                wait = min(60, 5 * 2**attempt)
                print(f"\n  NCBI {what} attempt {attempt + 1}/{NCBI_ATTEMPTS} "
                      f"failed ({type(e).__name__}: {e}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"NCBI {what} failed after {NCBI_ATTEMPTS} attempts: "
                       f"{last_err}") from last_err


def _parse_elink(resp) -> Dict[str, List[str]]:
    """elink XML -> {gene_id: [every linked RefSeq protein UID]}."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    err = root.find("ERROR")
    if err is not None:
        raise RuntimeError(f"elink error: {err.text}")
    links: Dict[str, List[str]] = {}
    for linkset in root.findall(".//LinkSet"):
        id_elem = linkset.find("IdList/Id")
        if id_elem is None or not id_elem.text:
            continue
        links[id_elem.text] = [
            link.text for link in linkset.findall("LinkSetDb/Link/Id")
            if link.text
        ]
    return links


def _parse_esummary(resp, uids: List[str]) -> Dict[str, Tuple[str, int]]:
    """esummary JSON -> {uid: (accession.version, length)}.

    A body without ``result`` (NCBI answers some errors with HTTP 200) or
    missing a requested UID is rejected, so the batch is retried. A UID that
    NCBI explicitly reports as an error (suppressed record) is left out.
    """
    data = resp.json()
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        raise RuntimeError(f"esummary returned no result: {str(data)[:200]}")
    out: Dict[str, Tuple[str, int]] = {}
    for uid in uids:
        info = result.get(uid)
        if not isinstance(info, dict):
            raise RuntimeError(f"esummary has no record for protein {uid}")
        acc = info.get("accessionversion")
        if acc:
            out[uid] = (acc, int(info.get("slen") or 0))
        elif "error" not in info:
            raise RuntimeError(f"esummary record for {uid} has no accession")
    return out


def _parse_fasta_accessions(text: str, wanted: List[str]) -> Dict[str, str]:
    """Parse efetch FASTA into {accession: sequence} for ``wanted``; raise if
    any wanted accession is absent or empty (a truncated response)."""
    lookup = {}
    for acc in wanted:
        lookup[acc] = acc
        lookup[acc.split(".")[0]] = acc
    seqs: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            current = None
            for token in line[1:].replace("|", " ").split():
                hit = lookup.get(token) or lookup.get(token.split(".")[0])
                if hit:
                    current = hit
                    seqs[current] = []
                    break
        elif line and current is not None:
            seqs[current].append(line)
    out = {acc: "".join(chunks) for acc, chunks in seqs.items() if chunks}
    missing = [acc for acc in wanted if acc not in out]
    if missing:
        raise RuntimeError(f"efetch response lacks {len(missing)} of "
                           f"{len(wanted)} sequences (e.g. {missing[:3]})")
    return out


def _ncbi_refseq_fallback(
    gene_ids: List[str],
    batch_size: int = 200,
) -> Dict[str, Tuple[str, str]]:
    """Resolve genes with no reviewed UniProt entry through NCBI RefSeq.

    elink (gene -> EVERY linked RefSeq protein; elink lists them newest
    first, so the old ``[0]`` pick was the most recent deposit, often a
    predicted XP_ isoform) + esummary (accession, length) pick one protein
    per gene with _refseq_rank, then efetch fetches its sequence. Genes with
    no RefSeq protein (non-coding loci) are simply absent from the result.

    Returns {gene_id: (accession, sequence)}. Raises RuntimeError when NCBI
    keeps failing (see _ncbi_request).
    """
    gene_to_uids: Dict[str, List[str]] = {}
    print(f"  NCBI RefSeq fallback for {len(gene_ids)} genes...")
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        # Separate id params (not comma-joined) so elink returns one LinkSet
        # per gene instead of merging them.
        params = [("dbfrom", "gene"), ("db", "protein"),
                  ("linkname", "gene_protein_refseq"), ("retmode", "xml")]
        params.extend(("id", gid) for gid in batch)
        gene_to_uids.update(
            _ncbi_request(ELINK_URL, params, f"elink batch {i // batch_size}",
                          _parse_elink))

    wanted_genes = set(gene_ids)
    linked = {g: u for g, u in gene_to_uids.items() if u and g in wanted_genes}
    all_uids = sorted({u for uids in linked.values() for u in uids}, key=int)
    uid_info: Dict[str, Tuple[str, int]] = {}
    for i in range(0, len(all_uids), batch_size):
        batch = all_uids[i:i + batch_size]
        params = {"db": "protein", "id": ",".join(batch), "retmode": "json"}
        uid_info.update(
            _ncbi_request(ESUMMARY_URL, params,
                          f"esummary batch {i // batch_size}",
                          lambda r, b=batch: _parse_esummary(r, b)))

    chosen: Dict[str, Tuple[str, str]] = {}  # gene -> (accession, uid)
    for gid, uids in linked.items():
        cands = [(uid_info[u][0], uid_info[u][1], u) for u in uids
                 if u in uid_info]
        if cands:
            acc, _, uid = min(cands, key=lambda c: _refseq_rank(c[0], c[1]))
            chosen[gid] = (acc, uid)

    uid_to_acc = {uid: acc for acc, uid in chosen.values()}
    uid_list = sorted(uid_to_acc, key=int)
    seq_by_acc: Dict[str, str] = {}
    for i in range(0, len(uid_list), batch_size):
        batch = uid_list[i:i + batch_size]
        wanted = [uid_to_acc[u] for u in batch]
        params = {"db": "protein", "id": ",".join(batch),
                  "rettype": "fasta", "retmode": "text"}
        seq_by_acc.update(
            _ncbi_request(EFETCH_URL, params,
                          f"efetch batch {i // batch_size}",
                          lambda r, w=wanted: _parse_fasta_accessions(r.text,
                                                                      w)))

    print(f"  RefSeq protein found for {len(chosen)}/{len(gene_ids)} "
          f"fallback genes (the rest have no RefSeq protein link)")
    return {g: (acc, seq_by_acc[acc]) for g, (acc, _) in chosen.items()}


def resolve_protein_sequences(
    gene_ids: List[str],
    cache_dir: str,
    ncbi_fallback: bool = True,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map Entrez Gene IDs to one canonical protein sequence each.

    Shared by ESM-2 (this script) and the other protein models
    (generate_embeddings.py), so every protein embedding sees the same
    sequence per gene:
      1. UniProt reviewed canonical sequence of the gene's HGNC ``uniprot_ids``
         (several reviewed accessions: the one whose GN= is the HGNC symbol,
         else the longest).
      2. None of those in the reviewed set: the reviewed entry whose GN= is
         the gene's HGNC symbol.
      3. The rest (~0.5% of the universe, mostly non-coding RNA / pseudogene
         loci): NCBI RefSeq, NP_ preferred over YP_/XP_, then longest.
    ESM truncates to the first 1,022 residues, so the choice matters: the
    previous rule (the first elink hit, i.e. the newest RefSeq record) gave
    about half the genes a predicted XP_ isoform (BRCA1 a 712-aa isoform
    instead of the 1,863-aa canonical protein) and changed with every NCBI
    release.

    The result is cached in ``cache_dir/protein_sequences_canonical.json``,
    keyed by SEQUENCE_CACHE_FORMAT, the two source files' checksums and the
    requested gene set, so ESM-2 (SLURM STEP 1) and the later protein models
    (STEP 3) embed identical sequences even though the NCBI fallback runs only
    once.

    Args:
        gene_ids: Entrez Gene IDs (strings).
        cache_dir: Download/cache directory (the SLURM CACHE_DIR).
        ncbi_fallback: Query NCBI for genes steps 1-2 miss (False: leave them
            unresolved, which is how the offline unit test runs).

    Returns:
        (sequences, accessions): {gene_id: amino-acid sequence} and
        {gene_id: "UniProt:<acc>" | "RefSeq:<acc>"}, for resolved genes only.
    """
    import json

    fasta = _find_source_file(UNIPROT_FASTA_NAME, cache_dir)
    hgnc = _find_source_file(HGNC_TABLE_NAME, cache_dir)
    if fasta is None or hgnc is None:
        raise FileNotFoundError(
            f"Canonical protein sources missing: need {UNIPROT_FASTA_NAME} and "
            f"{HGNC_TABLE_NAME} in {cache_dir} or its sibling data/cache "
            f"(both are tracked in git). Refusing to fall back to NCBI's "
            f"newest RefSeq isoform for every gene.")
    requested = sorted({str(g) for g in gene_ids})
    fingerprint = {
        UNIPROT_FASTA_NAME: _file_md5(fasta),
        HGNC_TABLE_NAME: _file_md5(hgnc),
        "ncbi_fallback": bool(ncbi_fallback),
    }

    cache_path = Path(cache_dir) / SEQUENCE_CACHE_NAME
    if cache_path.is_file():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if (cached.get("format") == SEQUENCE_CACHE_FORMAT
                    and cached.get("fingerprint") == fingerprint
                    and set(requested) <= set(cached.get("requested", []))):
                genes = cached["genes"]
                seqs = {g: genes[g][1] for g in requested if g in genes}
                accs = {g: genes[g][0] for g in requested if g in genes}
                print(f"  Loaded canonical protein sequences for "
                      f"{len(seqs)}/{len(requested)} genes from {cache_path}")
                return seqs, accs
            print(f"  Ignoring stale {cache_path} (different sources, cache "
                  f"format or gene set)")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            print(f"  Ignoring unreadable {cache_path}: {e}")

    print(f"Resolving canonical protein sequences for {len(requested)} genes "
          f"({fasta.name} + {hgnc.name})...")
    records = _parse_uniprot_fasta(fasta)
    table = _parse_hgnc_uniprot(hgnc)
    by_gn: Dict[str, List[str]] = {}
    for acc, (gn, _) in records.items():
        if gn:
            by_gn.setdefault(gn, []).append(acc)

    seqs: Dict[str, str] = {}
    accs: Dict[str, str] = {}
    n_hgnc = n_gn = 0
    rest: List[str] = []
    for gid in requested:
        symbol, uniprot_ids = table.get(gid, ("", []))
        acc = _pick_uniprot(uniprot_ids, symbol, records)
        if acc is not None:
            n_hgnc += 1
        elif symbol:
            acc = _pick_uniprot(by_gn.get(symbol, []), symbol, records)
            n_gn += acc is not None
        if acc is None:
            rest.append(gid)
            continue
        seqs[gid] = records[acc][1]
        accs[gid] = f"UniProt:{acc}"
    print(f"  UniProt reviewed: {n_hgnc} via HGNC uniprot_ids, {n_gn} via "
          f"GN= symbol; {len(rest)} left")

    if rest and ncbi_fallback:
        for gid, (acc, seq) in _ncbi_refseq_fallback(rest).items():
            seqs[gid] = seq
            accs[gid] = f"RefSeq:{acc}"
    n_missing = len(requested) - len(seqs)
    print(f"  Protein sequence for {len(seqs)}/{len(requested)} genes "
          f"({n_missing} without one: non-coding or no protein record)")

    payload = {
        "format": SEQUENCE_CACHE_FORMAT,
        "fingerprint": fingerprint,
        "requested": requested,
        "genes": {g: [accs[g], seqs[g]] for g in seqs},
    }
    try:
        tmp = cache_path.with_name(cache_path.name + ".part")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, cache_path)
    except OSError as e:
        print(f"  WARNING: could not write {cache_path}: {e}")
    return seqs, accs


class StreamingAttentionAggregator:
    """
    Streams attention through ESM-2 layer-by-layer using forward hooks,
    maintaining a running element-wise max. Only one layer's attention is
    held in memory at a time — ~33x less peak memory than materializing the
    full (layers, heads, seq_len, seq_len) tensor.

    Usage:
        agg = StreamingAttentionAggregator(model)
        agg.attach()
        with torch.no_grad():
            results = model(tokens, repr_layers=[...], need_head_weights=False)
        # agg.max_attn is (batch, heads, seq_len, seq_len) on CPU
        agg.detach()
    """

    def __init__(self, model):
        self.model = model
        self.max_attn = None  # running max: (B, H, T, T) on CPU
        self._hooks = []

    def _hook_fn(self, module, inputs, outputs):
        # TransformerLayer.forward returns (x, attn)
        # attn shape from self_attn with need_head_weights=True: (H, B, T, T)
        _, attn = outputs
        if attn is None:
            return
        # (H, B, T, T) → (B, H, T, T), move to CPU immediately
        attn = attn.transpose(0, 1).cpu()
        if self.max_attn is None:
            self.max_attn = attn
        else:
            torch.maximum(self.max_attn, attn, out=self.max_attn)

    def attach(self):
        """Register hooks and patch layers to emit per-head attention."""
        self.max_attn = None
        # Patch each TransformerLayer so its self_attn call uses
        # need_head_weights=True even though the top-level forward
        # passes need_head_weights=False.
        for layer in self.model.layers:
            had_instance_fwd = 'forward' in layer.__dict__
            original_forward = layer.forward

            def make_patched(orig):

                def patched(x,
                            self_attn_mask=None,
                            self_attn_padding_mask=None,
                            need_head_weights=False):
                    return orig(x,
                                self_attn_mask,
                                self_attn_padding_mask,
                                need_head_weights=True)

                return patched

            layer.forward = make_patched(original_forward)
            layer._original_forward = original_forward
            layer._orig_had_instance_forward = had_instance_fwd
            h = layer.register_forward_hook(self._hook_fn)
            self._hooks.append((layer, h))

    def detach(self):
        """Remove hooks and restore original forward methods."""
        for layer, h in self._hooks:
            h.remove()
            if hasattr(layer, '_original_forward'):
                if layer._orig_had_instance_forward:
                    # Restore the instance attribute that existed before attach()
                    layer.forward = layer._original_forward
                else:
                    # forward was resolved via class descriptor — remove instance attr
                    layer.__dict__.pop('forward', None)
                del layer._original_forward
                del layer._orig_had_instance_forward
        self._hooks.clear()

    def get_aggregated(self) -> Optional[torch.Tensor]:
        """Return the head-maxed aggregated attention: (B, T, T) on CPU."""
        if self.max_attn is None:
            return None
        # max over heads: (B, H, T, T) → (B, T, T)
        return self.max_attn.max(dim=1).values


class PoolPaRTI:
    """
    Pool PaRTI: PageRank-based Token Importance pooling.

    Converts attention matrices to a directed graph and uses PageRank
    to compute importance weights for each residue position.

    Reference: Tartici et al., Bioinformatics 2025
    """

    def __init__(self, alpha: float = 0.85):
        """
        Args:
            alpha: PageRank damping factor (default 0.85)
        """
        if nx is None:
            raise ImportError(
                "networkx required for Pool PaRTI. Install: pip install networkx"
            )
        self.alpha = alpha

    def _compute_pagerank(self, attention_matrix: np.ndarray) -> np.ndarray:
        """
        Compute PageRank scores from attention matrix.

        Args:
            attention_matrix: Aggregated attention (n_tokens, n_tokens)

        Returns:
            PageRank score of each token (n_tokens,)

        Raises, as the authors' ``_page_rank`` does, on a graph without edges
        or a PageRank that does not converge (nx.PowerIterationFailedConvergence)
        instead of silently mean-pooling; the caller then records the gene as
        failed (NaN). Neither happens on softmax attention (strictly positive,
        and >= 10 residues converge within max_iter).
        """
        # Create directed graph from attention matrix
        G = nx.from_numpy_array(attention_matrix, create_using=nx.DiGraph)
        if G.number_of_edges() == 0:
            raise ValueError("Pool PaRTI: attention graph has no edges")
        pr = nx.pagerank(G,
                         alpha=self.alpha,
                         tol=1e-06,
                         weight='weight',
                         max_iter=100)
        return np.array([pr[i] for i in range(len(pr))])

    def pool(
        self,
        token_embeddings: torch.Tensor,
        agg_attn: torch.Tensor,
        seq_len: int,
    ) -> np.ndarray:
        """
        Apply Pool PaRTI to get sequence embedding.

        Args:
            token_embeddings: Per-token embeddings (1, full_seq_len, embed_dim)
            agg_attn: Pre-aggregated attention (full_seq_len, full_seq_len),
                      already maxed over layers and heads.
            seq_len: Actual sequence length (excluding special tokens)

        Returns:
            Sequence embedding (embed_dim,)
        """
        # Extract embeddings for actual sequence (exclude BOS/EOS tokens)
        # ESM adds BOS at position 0, sequence is 1:seq_len+1
        embeddings = token_embeddings[
            0, 1:seq_len + 1, :].cpu().numpy()  # (seq_len, embed_dim)

        # As the authors (Pool_PaRTI token_to_sequence_pooler.py): PageRank on
        # the whole token graph, BOS (0) and EOS (seq_len + 1) included, since
        # rank flows through those attention sinks; both are dropped only
        # afterwards and the residue weights renormalised. Columns past EOS
        # are batch padding (zero attention), which the authors never have.
        attn_full = agg_attn[:seq_len + 2, :seq_len + 2].numpy()
        weights = self._compute_pagerank(attn_full)[1:seq_len + 1]
        weights = weights / weights.sum()

        # Weighted average of embeddings
        pooled = np.sum(embeddings * weights[:, np.newaxis], axis=0)

        return pooled


class ESMEmbeddingGenerator:
    """Generate ESM-2 embeddings for protein sequences."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
        batch_size: int = 8,
        pooling: str = "pool_parti",
    ):
        """
        Args:
            model_name: ESM model name
            device: Device to run on
            batch_size: Batch size for processing
            pooling: Pooling method - "pool_parti" (default) or "mean"
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling = pooling

        # Detect + robustly initialize the compute device.
        #
        # model.to(device) below is the first real CUDA use; it triggers torch's
        # built-in lazy init (torch._C._cuda_init()). On a flaky node that can
        # raise a transient "CUDA unknown error ... Setting the available devices
        # to be zero" (a cgroup/driver/stale-context race). We drive torch's
        # built-in CUDA init here with a bounded retry + backoff so a transient
        # clears instead of killing the whole generation job. Cluster runs must
        # stay on GPU, so we RAISE on persistent failure rather than silently
        # dropping to CPU; pass --device cpu explicitly for local CPU debugging.
        # NOTE: the guard also accepts "cuda:0" (the SLURM launcher's value) —
        # the old `device == "cuda"` check silently skipped it.
        dev_str = str(device)
        if dev_str.startswith("cuda"):
            import time as _time
            _last_err = None
            for _attempt in range(1, 4):
                try:
                    torch.cuda.init()
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "torch.cuda.is_available() is False "
                            f"(torch {torch.__version__}, "
                            f"cuda {torch.version.cuda})")
                    _probe = torch.zeros(1,
                                         device=dev_str)  # forces _cuda_init
                    _ = (_probe + 1).item()
                    del _probe
                    _last_err = None
                    if _attempt > 1:
                        print(f"CUDA init recovered on attempt {_attempt}/3")
                    break
                except RuntimeError as _e:
                    _last_err = _e
                    print(f"CUDA init attempt {_attempt}/3 on {dev_str} "
                          f"failed: {_e}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if _attempt < 3:
                        _time.sleep(10 * _attempt)
            if _last_err is not None:
                raise RuntimeError(
                    f"CUDA unavailable on {dev_str} after 3 attempts "
                    f"(torch {torch.__version__}, cuda {torch.version.cuda}). "
                    f"Last error: {_last_err}. This is typically a transient "
                    f"node-local CUDA-context failure; resubmit (SLURM steers "
                    f"off the bad node) or pin a driver-matched torch build.")
        self.device = torch.device(dev_str)

        print(f"Loading ESM model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Pooling method: {pooling}")

        # Initialize Pool PaRTI if needed
        if pooling == "pool_parti":
            self.pooler = PoolPaRTI(alpha=0.85)
        else:
            self.pooler = None

        # Load ESM
        try:
            import esm
        except ImportError:
            raise ImportError(
                "fair-esm not installed. Run: uv pip install fair-esm")

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_name)
        # Context already warmed above; retry the transfer once more in case the
        # move itself hits a transient CUDA error on a flaky node.
        for _attempt in range(1, 4):
            try:
                self.model = self.model.to(self.device)
                break
            except RuntimeError as _e:
                if self.device.type != "cuda" or _attempt == 3:
                    raise
                print(f"model.to({self.device}) attempt {_attempt}/3 failed: "
                      f"{_e}; retrying...")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                import time as _time
                _time.sleep(10 * _attempt)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Get model info
        self.embed_dim = self.model.embed_dim
        self.num_layers = self.model.num_layers
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Number of layers: {self.num_layers}")

    def _generate_batch(self,
                        batch: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for a batch of sequences.

        When pooling == "pool_parti", uses StreamingAttentionAggregator to
        compute the layer/head-wise max attention incrementally via forward
        hooks. Only one layer's attention is in memory at a time (~33x less
        peak memory than materializing the full attention stack).
        """
        labels, strs, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        use_pool_parti = self.pooling == "pool_parti" and self.pooler is not None

        # Attach streaming hooks if Pool PaRTI
        if use_pool_parti:
            agg = StreamingAttentionAggregator(self.model)
            agg.attach()

        try:
            with torch.no_grad():
                results = self.model(
                    tokens,
                    repr_layers=[self.num_layers],
                    need_head_weights=False,  # hooks handle attention
                    return_contacts=False,
                )
                representations = results["representations"][self.num_layers]

            # Get pre-aggregated attention (B, T, T) on CPU
            agg_attn = agg.get_aggregated() if use_pool_parti else None
        finally:
            if use_pool_parti:
                agg.detach()

        embeddings = {}
        for i, (gene, seq) in enumerate(batch):
            seq_len = min(len(seq), 1022)  # ESM max length

            if use_pool_parti and agg_attn is not None:
                # Pool PaRTI with streaming-aggregated attention
                emb_i = representations[i:i + 1]  # (1, seq, dim)
                embedding = self.pooler.pool(emb_i, agg_attn[i], seq_len)
            else:
                # Mean pooling fallback
                embedding = representations[i, 1:seq_len +
                                            1].mean(0).cpu().numpy()

            embeddings[gene] = embedding

        return embeddings

    def generate_embeddings(
        self,
        gene_seqs: Dict[str, str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        """
        Generate embeddings for all genes.

        Args:
            gene_seqs: Dict mapping gene_id -> amino acid sequence

        Returns:
            (embeddings tensor, raw_embeddings tensor, gene_order list, failed_genes list)
        """
        gene_order = sorted(gene_seqs.keys())
        all_embeddings = {}

        # Prepare batches
        batch = []
        failed_genes = []

        print(f"\nGenerating embeddings for {len(gene_order)} genes...")
        pbar = tqdm(gene_order, desc="Embedding")

        for gene in pbar:
            seq = gene_seqs[gene]

            # Skip invalid sequences — mark as NaN; imputed per fold at
            # training time (data_loader, fit on the fold's training rows:
            # Huber in the robust path, mean in plain).
            if not seq or len(seq) < 10:
                failed_genes.append(gene)
                all_embeddings[gene] = np.full(self.embed_dim,
                                               np.nan,
                                               dtype=np.float32)
                continue

            # Truncate long sequences
            if len(seq) > 1022:
                seq = seq[:1022]

            batch.append((gene, seq))

            # Process batch
            if len(batch) >= self.batch_size:
                batch_embs = self._generate_batch_safe(batch)
                all_embeddings.update(batch_embs[0])
                failed_genes.extend(batch_embs[1])
                batch = []

        # Process remaining
        if batch:
            batch_embs = self._generate_batch_safe(batch)
            all_embeddings.update(batch_embs[0])
            failed_genes.extend(batch_embs[1])

        # Stack into tensor (failed genes have NaN, imputed per fold later)
        if not gene_order:
            return torch.empty(0, self.embed_dim), torch.empty(
                0, self.embed_dim), [], []
        embedding_matrix = np.stack([all_embeddings[g] for g in gene_order])

        n_failed = len(failed_genes)
        n_ok = len(gene_order) - n_failed
        print(f"\nGenerated {len(gene_order)} embeddings")
        print(f"  - Successful: {n_ok}")
        print(f"  - Failed (NaN, imputed per fold at training): {n_failed}")

        embeddings = torch.from_numpy(embedding_matrix).float()
        return (
            embeddings,
            embeddings.clone(),
            gene_order,
            failed_genes,
        )

    def _generate_batch_safe(
        self,
        batch: List[Tuple[str, str]],
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Generate embeddings for a batch, falling back to per-sequence on failure."""
        try:
            return self._generate_batch(batch), []
        except Exception as e:
            # Common cause: CUDA OOM on an unusually long sequence. Logged so
            # the user can see it in slurm logs and tune --batch_size if needed.
            print(
                f"  Batch failed ({len(batch)} seqs), retrying per-sequence: "
                f"{type(e).__name__}: {e}")

        # Batch failed — retry each sequence individually
        embeddings = {}
        failed = []
        for gene, seq in batch:
            try:
                emb = self._generate_batch([(gene, seq)])
                embeddings.update(emb)
            except Exception:
                failed.append(gene)
                embeddings[gene] = np.full(self.embed_dim,
                                           np.nan,
                                           dtype=np.float32)
        return embeddings, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM embeddings for all human genes")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output path for embeddings (.pt file)")
    parser.add_argument("--gene_list",
                        type=str,
                        default=None,
                        help="Path to .pt file (uses gene_order), "
                        ".txt file (one Entrez Gene ID per line), or "
                        "SL pairs file (two-column tab-separated Entrez IDs)")
    parser.add_argument("--sl_path",
                        type=str,
                        default=None,
                        help="Path to SL pairs file (two-column tab-separated "
                        "Entrez IDs). Extracts unique gene IDs from both "
                        "columns. Alternative to --gene_list.")
    parser.add_argument("--cache_dir",
                        type=str,
                        default="../data/cache",
                        help="Directory to cache downloaded files")
    parser.add_argument("--model",
                        type=str,
                        default="esm2_t33_650M_UR50D",
                        help="ESM model name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_model_failures",
        type=int,
        default=0,
        help="how many genes that HAVE a protein sequence may fail inside the "
        "model (NaN row) before the run aborts without writing --output "
        "(default 0)")
    parser.add_argument(
        "--pooling",
        type=str,
        default="pool_parti",
        choices=["pool_parti", "mean"],
        help="Pooling method: pool_parti (PageRank-based, default) or mean")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get gene list (Entrez Gene IDs)
    gene_src = args.gene_list or args.sl_path
    if gene_src is None:
        raise ValueError(
            "--gene_list or --sl_path is required. Pass a .pt file "
            "(uses gene_order), .txt file (one Entrez Gene ID per line), "
            "or SL pairs file (two-column tab-separated Entrez IDs).")

    if gene_src.endswith(".pt"):
        d = torch.load(gene_src, map_location="cpu", weights_only=False)
        gene_ids = list(d["gene_order"])
        print(f"Loaded {len(gene_ids)} gene IDs from {gene_src}")
    else:
        # Handles both single-column gene lists and two-column SL pairs
        gene_id_set = set()
        with open(gene_src) as f:
            for line in f:
                for token in line.strip().split():
                    if token.isdigit():
                        gene_id_set.add(token)
        gene_ids = sorted(gene_id_set)
        print(f"Loaded {len(gene_ids)} unique gene IDs from {gene_src}")

    # Canonical protein sequence per gene (UniProt reviewed; RefSeq fallback)
    gene_seqs, accessions = resolve_protein_sequences(gene_ids, str(cache_dir))

    # Build full gene list: genes with sequences + genes without (NaN for imputation)
    all_gene_ids = sorted(set(gene_ids))
    missing = [g for g in all_gene_ids if g not in gene_seqs]
    print(f"\nSequence coverage: {len(gene_seqs)}/{len(all_gene_ids)} genes")
    if missing:
        print(
            f"  {len(missing)} genes have no protein sequence (will be NaN-imputed)"
        )

    # Generate embeddings
    generator = ESMEmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )

    # Add missing genes to gene_seqs with empty sequences so they appear in
    # gene_order. generate_embeddings() will NaN-fill them (seq < 10 AA).
    for gid in missing:
        gene_seqs[gid] = ""

    embeddings, raw_embeddings, gene_order, failed = generator.generate_embeddings(
        gene_seqs)

    print(f"  Final gene_order: {len(gene_order)} Entrez IDs")

    # Two very different NaN rows hide in `failed`. A gene with no protein
    # sequence is an expected, permanent absence (non-coding locus). A gene
    # WITH a sequence that came back NaN means the MODEL failed on it: a CUDA
    # fault poisons the context, so every later batch AND its per-sequence
    # retry in _generate_batch_safe fails and thousands of coding genes go NaN.
    # Saving that would stamp it generator_version = current and pass both the
    # SLURM `[ -f "$ESM_PATH" ]` gate and embedding_versions.py, and
    # data_loader would mean-impute the rows per fold — so the esm2 row and
    # every combo using it would be published from one shared vector with
    # nothing failing.
    no_seq = {g for g in gene_order if len(gene_seqs.get(g) or "") < 10}
    model_failed = [g for g in failed if g not in no_seq]
    print(f"  No protein sequence (expected NaN): {len(no_seq)}")
    print(f"  ESM-2 failed on a gene that has a sequence: {len(model_failed)}")
    if len(model_failed) > args.max_model_failures:
        raise SystemExit(
            f"ERROR: ESM-2 failed on {len(model_failed)} genes that have a "
            f"protein sequence (e.g. {model_failed[:5]}); not writing "
            f"{args.output}. One CUDA fault mid-run fails every later batch "
            f"and its per-sequence retry — rerun (SLURM steers off the bad "
            f"node), or pass --max_model_failures N to accept N such genes.")

    # Save
    output_path = Path(args.output)

    # A non-pipeline --pooling / --model is a different embedding, not a
    # different version of the same one; say so in the stamp (see PIPELINE_ESM).
    generator_version = EMBEDDING_GENERATOR_VERSION
    if (args.pooling, args.model) != PIPELINE_ESM:
        generator_version = (f"{EMBEDDING_GENERATOR_VERSION} (non-pipeline "
                             f"build: pooling={args.pooling}, "
                             f"model={args.model})")
        print(f"WARNING: --pooling {args.pooling} --model {args.model} is not "
              f"the pipeline's ESM build {PIPELINE_ESM}; the output is stamped "
              f"generator_version={generator_version!r}, which "
              f"embedding_versions.py reports as stale.")

    torch.save(
        {
            "embeddings": embeddings,
            "raw_embeddings": raw_embeddings,
            "gene_order": gene_order,
            "model_name": args.model,
            "dimension": embeddings.shape[1],
            "pooling": args.pooling,
            "standardized":
            False,  # Raw embeddings; imputed + standardized per fold at training
            "failed_genes": failed,
            "source": SEQUENCE_SOURCE,
            # gene -> "UniProt:<acc>" / "RefSeq:<acc>" actually embedded
            "protein_accessions":
            {g: accessions[g]
             for g in gene_order if g in accessions},
            "generator_version": generator_version,
            "num_genes": len(gene_order),
        },
        output_path)

    # Also save gene list as text file
    gene_list_path = output_path.with_suffix(".genes.txt")
    with open(gene_list_path, "w") as f:
        for gene in gene_order:
            f.write(f"{gene}\n")

    print(f"\nSaved embeddings to {output_path}")
    print(f"Saved gene list to {gene_list_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
