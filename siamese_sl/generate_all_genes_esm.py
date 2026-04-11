#!/usr/bin/env python3
"""
Generate ESM embeddings for ALL human protein-coding genes.

This script:
  1. Fetches canonical protein sequences from NCBI using Entrez Gene IDs
  2. Generates ESM-2 embeddings using Pool PaRTI (PageRank-based pooling)
  3. Saves embeddings with gene order for reproducibility

Sequence source: NCBI gene2refseq (Gene ID → RefSeq protein → FASTA).
Uses Entrez Gene IDs natively — no gene symbol matching needed.

Pool PaRTI Reference:
  Tartici et al. "Pool PaRTI: a PageRank-based pooling method for identifying
  critical residues and enhancing protein sequence representations"
  Bioinformatics, 2025. https://github.com/Helix-Research-Lab/Pool_PaRTI

Requirements:
  pip install fair-esm requests tqdm networkx

Usage:
    # Basic usage with Pool PaRTI (default)
    python generate_all_genes_esm.py --output ../data/all_genes_esm.pt

    # Use mean pooling instead
    python generate_all_genes_esm.py --pooling mean --output ../data/all_genes_esm.pt

    # GPU with larger batch size
    python generate_all_genes_esm.py --device cuda:0 --batch_size 16 --output out.pt
"""

import argparse
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm

try:
    import networkx as nx
except ImportError:
    nx = None  # Will be checked if pool_parti is used

# NCBI E-utilities base URLs
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _ncbi_batch_elink(gene_ids: List[str],
                      batch_size: int = 200,
                      ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Batch elink: Gene ID → RefSeq protein GI, then esummary GI → accession.

    Uses NCBI elink (gene → protein, RefSeq subset) to find the canonical
    protein for each gene, then esummary to resolve GI numbers to accessions.

    Args:
        gene_ids: List of Entrez Gene IDs (strings)
        batch_size: IDs per request (NCBI recommends ≤200)

    Returns:
        Tuple of (gene_to_gi, gi_to_acc):
          - gene_to_gi: Dict mapping gene_id -> protein GI
          - gi_to_acc: Dict mapping protein GI -> accession (e.g. NP_009225.1)
    """
    import time
    import xml.etree.ElementTree as ET

    gene_to_gi: Dict[str, str] = {}

    print(f"Linking {len(gene_ids)} Gene IDs → RefSeq protein GIs...")
    for i in tqdm(range(0, len(gene_ids), batch_size), desc="NCBI elink"):
        batch = gene_ids[i:i + batch_size]

        for attempt in range(3):
            try:
                # Use separate id params (not comma-joined) so elink
                # returns one LinkSet per gene instead of merging them.
                params = [("dbfrom", "gene"), ("db", "protein"),
                          ("linkname", "gene_protein_refseq"),
                          ("retmode", "xml")]
                params.extend(("id", gid) for gid in batch)
                resp = requests.post(ELINK_URL, data=params, timeout=60)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  elink batch failed: {e}")
                    resp = None

        if resp is None:
            continue

        # Parse XML: extract Gene ID → protein GI mappings
        root = ET.fromstring(resp.text)
        for linkset in root.findall(".//LinkSet"):
            id_elem = linkset.find("IdList/Id")
            if id_elem is None:
                continue
            gene_id = id_elem.text

            link_db = linkset.find(".//LinkSetDb")
            if link_db is None:
                continue

            protein_ids = [link.find("Id").text
                          for link in link_db.findall("Link")
                          if link.find("Id") is not None]

            if protein_ids:
                gene_to_gi[gene_id] = protein_ids[0]

        time.sleep(0.35)

    print(f"  Found protein links for {len(gene_to_gi)}/{len(gene_ids)} genes")

    # Resolve GI numbers → accessions via esummary (FASTA headers use accessions)
    gi_to_acc: Dict[str, str] = {}
    all_gis = list(set(gene_to_gi.values()))

    print(f"  Resolving {len(all_gis)} protein GIs → accessions...")
    for i in tqdm(range(0, len(all_gis), batch_size), desc="NCBI esummary"):
        batch = all_gis[i:i + batch_size]
        for attempt in range(3):
            try:
                resp = requests.post(
                    ESUMMARY_URL,
                    data={
                        "db": "protein",
                        "id": ",".join(batch),
                        "retmode": "json",
                    },
                    timeout=60,
                )
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
                    print(f"\n  esummary batch failed: {e}")
        time.sleep(0.35)

    return gene_to_gi, gi_to_acc


def fetch_protein_sequences(gene_to_protein: Dict[str, str],
                            cache_dir: str,
                            batch_size: int = 200,
                            gi_to_acc: Optional[Dict[str, str]] = None,
                            ) -> Dict[str, str]:
    """Batch fetch protein sequences from NCBI.

    Args:
        gene_to_protein: Dict mapping gene_id -> protein GI
        cache_dir: Directory to cache the downloaded FASTA
        batch_size: GIs per efetch request
        gi_to_acc: Dict mapping protein GI -> accession (from esummary)

    Returns:
        Dict mapping gene_id -> amino acid sequence
    """
    import time

    cache_path = Path(cache_dir) / "ncbi_protein_sequences.fasta"

    # Check cache
    if cache_path.exists():
        print(f"Loading cached protein sequences: {cache_path}")
        return _parse_ncbi_fasta(cache_path, gene_to_protein, gi_to_acc)

    protein_ids = list(gene_to_protein.values())

    print(f"Fetching {len(protein_ids)} protein sequences from NCBI...")
    all_fasta = []

    for i in tqdm(range(0, len(protein_ids), batch_size), desc="NCBI efetch"):
        batch = protein_ids[i:i + batch_size]

        for attempt in range(3):
            try:
                resp = requests.post(
                    EFETCH_URL,
                    data={
                        "db": "protein",
                        "id": ",".join(batch),
                        "rettype": "fasta",
                        "retmode": "text",
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                all_fasta.append(resp.text)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"\n  efetch batch failed: {e}")

        time.sleep(0.35)

    # Save cache
    with open(cache_path, "w") as f:
        f.write("\n".join(all_fasta))
    print(f"  Cached to {cache_path}")

    return _parse_ncbi_fasta(cache_path, gene_to_protein, gi_to_acc)


def _parse_ncbi_fasta(fasta_path: Path,
                      gene_to_protein: Dict[str, str],
                      gi_to_acc: Optional[Dict[str, str]] = None,
                      ) -> Dict[str, str]:
    """Parse NCBI FASTA and map back to gene IDs via protein accession/GI.

    NCBI FASTA headers use accessions (e.g. ">NP_009225.1 ...") but elink
    returns GI numbers. We use the gi_to_acc mapping from esummary to bridge
    both formats.
    """
    # Build reverse map: any identifier → gene_id
    # Includes both GIs and accessions (with/without version) for robustness.
    protein_to_gene: Dict[str, str] = {}
    for gid, gi in gene_to_protein.items():
        protein_to_gene[gi] = gid
        if gi_to_acc:
            acc = gi_to_acc.get(gi)
            if acc:
                protein_to_gene[acc] = gid
                if "." in acc:
                    protein_to_gene[acc.split(".")[0]] = gid

    gene_seqs = {}
    current_gene = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous entry
                if current_gene and current_seq:
                    gene_seqs[current_gene] = "".join(current_seq)

                # Match header to gene ID
                current_gene = None
                current_seq = []
                header_tokens = line[1:].replace("|", " ").split()
                for token in header_tokens:
                    token_base = token.split(".")[0]
                    if token in protein_to_gene:
                        current_gene = protein_to_gene[token]
                        break
                    if token_base in protein_to_gene:
                        current_gene = protein_to_gene[token_base]
                        break
            elif line:
                current_seq.append(line)

        # Last entry
        if current_gene and current_seq:
            gene_seqs[current_gene] = "".join(current_seq)

    print(f"  Parsed sequences for {len(gene_seqs)} genes")
    return gene_seqs


def fetch_sequences_for_genes(gene_ids: List[str],
                              cache_dir: str) -> Dict[str, str]:
    """Fetch protein sequences from NCBI for a list of Entrez Gene IDs.

    Pipeline: Gene ID → elink (gene_protein_refseq) → efetch protein FASTA

    Args:
        gene_ids: List of Entrez Gene IDs (strings)
        cache_dir: Directory to cache downloaded files

    Returns:
        Dict mapping gene_id -> amino acid sequence
    """
    # Step 1: Link Gene IDs to RefSeq protein GIs + resolve to accessions
    gene_to_protein, gi_to_acc = _ncbi_batch_elink(gene_ids)

    missing = set(gene_ids) - set(gene_to_protein.keys())
    if missing:
        print(f"  No RefSeq protein for {len(missing)} genes "
              f"(non-coding or no RefSeq protein)")

    # Step 2: Batch fetch protein sequences
    gene_seqs = fetch_protein_sequences(gene_to_protein, cache_dir,
                                        gi_to_acc=gi_to_acc)

    missing_seq = set(gene_to_protein.keys()) - set(gene_seqs.keys())
    if missing_seq:
        print(f"  Failed to fetch sequence for {len(missing_seq)} genes")

    return gene_seqs


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
        self.max_attn = None        # running max: (B, H, T, T) on CPU
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
                def patched(x, self_attn_mask=None,
                            self_attn_padding_mask=None,
                            need_head_weights=False):
                    return orig(x, self_attn_mask, self_attn_padding_mask,
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
            attention_matrix: Aggregated attention (seq_len, seq_len)

        Returns:
            Importance weights for each position (seq_len,)
        """
        # Create directed graph from attention matrix
        G = nx.from_numpy_array(attention_matrix, create_using=nx.DiGraph)

        # Handle edge case of disconnected graph
        if G.number_of_edges() == 0:
            # Fall back to uniform weights
            n = attention_matrix.shape[0]
            return np.ones(n) / n

        # Compute PageRank
        try:
            pr = nx.pagerank(G,
                             alpha=self.alpha,
                             tol=1e-06,
                             weight='weight',
                             max_iter=100)
            weights = np.array([pr[i] for i in range(len(pr))])
        except nx.PowerIterationFailedConvergence:
            # Fall back to uniform weights on convergence failure
            n = attention_matrix.shape[0]
            weights = np.ones(n) / n

        return weights

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

        # Extract attention for actual sequence positions
        attn_seq = agg_attn[1:seq_len + 1,
                            1:seq_len + 1].numpy()  # (seq_len, seq_len)

        # Compute PageRank importance weights
        weights = self._compute_pagerank(attn_seq)

        # Normalize weights to sum to 1
        weights = weights / (weights.sum() + 1e-8)

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

        # Detect device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

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
        self.model = self.model.to(self.device)
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

            # Skip invalid sequences — mark as NaN for Huber imputation
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

        # Stack into tensor (failed genes have NaN, imputed at load time)
        if not gene_order:
            return torch.empty(0, self.embed_dim), torch.empty(0, self.embed_dim), [], []
        embedding_matrix = np.stack([all_embeddings[g] for g in gene_order])

        n_failed = len(failed_genes)
        n_ok = len(gene_order) - n_failed
        print(f"\nGenerated {len(gene_order)} embeddings")
        print(f"  - Successful: {n_ok}")
        print(f"  - Failed (NaN, Huber-imputed at load time): {n_failed}")

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
            pass

        # Batch failed — retry each sequence individually
        embeddings = {}
        failed = []
        for gene, seq in batch:
            try:
                emb = self._generate_batch([(gene, seq)])
                embeddings.update(emb)
            except Exception:
                failed.append(gene)
                embeddings[gene] = np.full(self.embed_dim, np.nan,
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
            "or SL pairs file (two-column tab-separated Entrez IDs)."
        )

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

    # Fetch protein sequences from NCBI
    gene_seqs = fetch_sequences_for_genes(gene_ids, str(cache_dir))

    # Build full gene list: genes with sequences + genes without (NaN for imputation)
    all_gene_ids = sorted(set(gene_ids))
    missing = [g for g in all_gene_ids if g not in gene_seqs]
    print(f"\nSequence coverage: {len(gene_seqs)}/{len(all_gene_ids)} genes")
    if missing:
        print(f"  {len(missing)} genes have no protein sequence (will be NaN-imputed)")

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

    # Save
    output_path = Path(args.output)

    torch.save(
        {
            "embeddings": embeddings,
            "raw_embeddings": raw_embeddings,
            "gene_order": gene_order,
            "model_name": args.model,
            "dimension": embeddings.shape[1],
            "pooling": args.pooling,
            "standardized":
            False,  # Raw embeddings, Huber-imputed at load time
            "failed_genes": failed,
            "source": "NCBI RefSeq (gene2refseq + efetch)",
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
