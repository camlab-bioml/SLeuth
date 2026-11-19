#!/usr/bin/env python3
"""Embedding-generation helpers, on tiny synthetic inputs (no network).

Covers the canonical protein resolver shared by ESM-2 and the other protein
models (UniProt reviewed via HGNC, GN= tie-break, RefSeq NP_ preference,
failing NCBI raises, cache reuse), the key -> Entrez mapping that lets a gene's
own symbol beat another gene's alias, the validated download cache, the
Mashup port against a direct numpy reference, ProtT5 residue pooling, the
SynLethDB raw-download cache (and its tracked data/cache sibling), Pool PaRTI
against the authors' full-graph PageRank, the pinned Geneformer download, the
builder-before-fallback order (a fallback file is stamped non-current), the
rebuild of a protein pkl built for another gene list, the order-free GO
sum-pool and the ESM-order sort of a gene_universe.txt list.

    $PYTHON_PATH -m pytest siamese_sl/tests/test_embedding_generation.py -q
    $PYTHON_PATH siamese_sl/tests/test_embedding_generation.py
"""
import gzip
import json
import os
import pickle
import sys
import tempfile
import time
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIAMESE = os.path.dirname(_HERE)
if _SIAMESE not in sys.path:
    sys.path.insert(0, _SIAMESE)

import download_synlethdb as dsl
import gene_name_utils
import generate_all_genes_esm as gae
import generate_embeddings as ge
import generate_go_esm_embeddings as gg

HGNC_HEADER = ["hgnc_id", "symbol", "prev_symbol", "alias_symbol",
               "entrez_id", "uniprot_ids", "locus_group"]


def _write_hgnc(path, rows):
    with open(path, "w") as f:
        f.write("\t".join(HGNC_HEADER) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in HGNC_HEADER) + "\n")


def _write_fasta(path, records):
    """records: (accession, GN or None, sequence)."""
    with open(path, "w") as f:
        for acc, gn, seq in records:
            tail = f" GN={gn} PE=1 SV=1" if gn else " PE=1 SV=1"
            f.write(f">sp|{acc}|{acc}_HUMAN Protein OS=Homo sapiens "
                    f"OX=9606{tail}\n{seq}\n")


def _toy_sources(root):
    """data/embeddings_cache/<fasta> + data/cache/<hgnc>, as in the repo."""
    cache = os.path.join(root, "data", "embeddings_cache")
    os.makedirs(cache)
    os.makedirs(os.path.join(root, "data", "cache"))
    _write_fasta(os.path.join(cache, gae.UNIPROT_FASTA_NAME), [
        ("P00001", "AAA1", "M" + "A" * 20),
        ("P00002", "OTHER", "M" + "C" * 40),  # longer, but GN != AAA1
        ("P00003", None, "M" + "D" * 15),
        ("P00004", None, "M" + "E" * 30),  # BBB2 tie: longest wins
        ("P00005", "CCC3", "M" + "F" * 12),  # GN= fills CCC3's empty ids
    ])
    _write_hgnc(os.path.join(root, "data", "cache", gae.HGNC_TABLE_NAME), [
        {"symbol": "AAA1", "entrez_id": "11", "uniprot_ids": "P00002|P00001"},
        {"symbol": "BBB2", "entrez_id": "22", "uniprot_ids": "P00003|P00004"},
        {"symbol": "CCC3", "entrez_id": "33", "uniprot_ids": ""},
        {"symbol": "DDD4", "entrez_id": "44", "uniprot_ids": "Q99999"},
    ])
    return cache


def test_resolver_uniprot_rules_and_cache():
    with tempfile.TemporaryDirectory() as root:
        cache = _toy_sources(root)
        genes = ["11", "22", "33", "44", "55"]
        seqs, accs = gae.resolve_protein_sequences(genes, cache,
                                                   ncbi_fallback=False)
        assert accs == {"11": "UniProt:P00001", "22": "UniProt:P00004",
                        "33": "UniProt:P00005"}, accs
        assert seqs["11"] == "M" + "A" * 20
        assert "44" not in seqs and "55" not in seqs
        # A subset of the cached gene set is served from the cache ...
        cache_file = os.path.join(cache, gae.SEQUENCE_CACHE_NAME)
        assert os.path.exists(cache_file)
        with open(cache_file) as f:
            payload = json.load(f)
        payload["genes"]["11"] = ["UniProt:P00001", "CACHED"]
        with open(cache_file, "w") as f:
            json.dump(payload, f)
        seqs2, _ = gae.resolve_protein_sequences(["11", "22"], cache,
                                                 ncbi_fallback=False)
        assert seqs2["11"] == "CACHED"
        # ... but a new source file (different checksum) rebuilds it
        _write_fasta(os.path.join(cache, gae.UNIPROT_FASTA_NAME),
                     [("P00001", "AAA1", "MKKK" * 5)])
        seqs3, _ = gae.resolve_protein_sequences(["11"], cache,
                                                 ncbi_fallback=False)
        assert seqs3["11"] == "MKKK" * 5


class _Resp:
    def __init__(self, text="", payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _with_fake_requests(post):
    """Install a stub ``requests`` module and a no-op sleep; return a
    restore callback."""
    saved_mod = sys.modules.get("requests")
    saved_sleep = time.sleep
    sys.modules["requests"] = types.SimpleNamespace(post=post)
    time.sleep = lambda s: None

    def restore():
        time.sleep = saved_sleep
        if saved_mod is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = saved_mod
    return restore


def test_refseq_fallback_prefers_np_and_raises_on_failure():
    elink = ("<eLinkResult>"
             "<LinkSet><IdList><Id>44</Id></IdList><LinkSetDb>"
             "<Link><Id>900</Id></Link><Link><Id>800</Id></Link>"
             "</LinkSetDb></LinkSet>"
             "<LinkSet><IdList><Id>55</Id></IdList></LinkSet>"
             "</eLinkResult>")
    summary = {"result": {
        "uids": ["800", "900"],
        # 900 is the newest record (elink lists it first) and the longest
        "900": {"accessionversion": "XP_000900.1", "slen": 50},
        "800": {"accessionversion": "NP_000800.2", "slen": 30},
    }}
    fasta = ">NP_000800.2 protein isoform 1 [Homo sapiens]\nMNNN\nPPP\n"

    def post(url, data=None, timeout=None):
        if url == gae.ELINK_URL:
            return _Resp(text=elink)
        if url == gae.ESUMMARY_URL:
            return _Resp(payload=summary)
        return _Resp(text=fasta)

    restore = _with_fake_requests(post)
    try:
        got = gae._ncbi_refseq_fallback(["44", "55"])
    finally:
        restore()
    assert got == {"44": ("NP_000800.2", "MNNNPPP")}, got

    calls = []

    def failing(url, data=None, timeout=None):
        calls.append(url)
        raise ConnectionError("reset by peer")

    restore = _with_fake_requests(failing)
    try:
        try:
            gae._ncbi_refseq_fallback(["44"])
            raise AssertionError("a persistent NCBI failure must raise")
        except RuntimeError as e:
            assert "elink" in str(e)
    finally:
        restore()
    assert len(calls) == gae.NCBI_ATTEMPTS

    # esummary answering HTTP 200 without "result" is a failure, not "no hits"
    def no_result(url, data=None, timeout=None):
        if url == gae.ELINK_URL:
            return _Resp(text=elink)
        return _Resp(payload={"error": "API rate limit exceeded"})

    restore = _with_fake_requests(no_result)
    try:
        try:
            gae._ncbi_refseq_fallback(["44"])
            raise AssertionError("an esummary error body must raise")
        except RuntimeError as e:
            assert "esummary" in str(e)
    finally:
        restore()


def _install_toy_mapper(root):
    """HGNC rows where ATD is a previous symbol of ATM and COX1 an alias of
    both MT-CO1 and PTGS1 (the later row wins, so COX1 -> PTGS1, as in the
    real table); ATD and COX1 are separate NCBI records."""
    path = os.path.join(root, "hgnc_complete_set.txt")
    _write_hgnc(path, [
        {"symbol": "ATM", "prev_symbol": "ATD", "entrez_id": "472"},
        {"symbol": "MT-CO1", "alias_symbol": "COX1", "entrez_id": "4512"},
        {"symbol": "PTGS1", "alias_symbol": "COX1", "entrez_id": "5742"},
    ])
    mapper = gene_name_utils.GeneNameMapper(cache_dir=root)
    mapper._ensure_loaded()  # parse the toy table now (loading is lazy)
    assert mapper.stats["entrez_ids"] == 3
    mapper._cache_dir = "../data/cache"  # what get_mapper() is called with
    saved = gene_name_utils._default_mapper
    gene_name_utils._default_mapper = mapper
    return saved


def test_align_prefers_exact_symbol_over_alias():
    a, b, c = (np.full(3, v, dtype=np.float32) for v in (1.0, 2.0, 3.0))
    with tempfile.TemporaryDirectory() as root:
        saved = _install_toy_mapper(root)
        try:
            for d in ({"ATD": a, "ATM": b}, {"ATM": b, "ATD": a}):
                emb, _ = ge.align_embeddings(d, ["472"], 3, "toy")
                assert np.allclose(emb[0].numpy(), b), (d.keys(), emb)
            emb, _ = ge.align_embeddings({"COX1": a, "PTGS1": b},
                                         ["5742", "4512"], 3, "toy")
            assert np.allclose(emb[0].numpy(), b)
            assert torch.isnan(emb[1]).all()
            # Entrez-keyed dicts (bioconceptvec, text_embed, protein models)
            # pass straight through, so MT-CO1 keeps its own vector
            emb, _ = ge.align_embeddings({"4512": c, "5742": b},
                                         ["5742", "4512"], 3, "toy")
            assert np.allclose(emb.numpy(), np.stack([b, c]))
        finally:
            gene_name_utils._default_mapper = saved


def test_align_pins_a_source_key_two_genes_claim():
    """gene2vec names two genes QARS and EPRS. HGNC lists QARS as a previous
    symbol of both QARS1 and EPRS1 and the mapper keeps the last row, so EPRS1
    carried the QARS vector (its own, under EPRS, lost the collision) and
    QARS1 had none. The pin gives each key to its own gene in either HGNC row
    order and either key order; a gene's current symbol still beats it."""
    a, b, c = (np.full(3, v, dtype=np.float32) for v in (1.0, 2.0, 3.0))
    rows = [{"symbol": "QARS1", "prev_symbol": "QARS", "entrez_id": "5859"},
            {"symbol": "EPRS1", "prev_symbol": "QPRS|QARS|EPRS",
             "entrez_id": "2058"}]
    for hgnc in (rows, rows[::-1]):
        with tempfile.TemporaryDirectory() as root:
            _write_hgnc(os.path.join(root, "hgnc_complete_set.txt"), hgnc)
            mapper = gene_name_utils.GeneNameMapper(cache_dir=root)
            mapper._ensure_loaded()
            mapper._cache_dir = "../data/cache"
            saved = gene_name_utils._default_mapper
            gene_name_utils._default_mapper = mapper
            try:
                for d in ({"QARS": a, "EPRS": b}, {"EPRS": b, "QARS": a}):
                    emb, _ = ge.align_embeddings(d, ["2058", "5859"], 3, "toy")
                    assert np.allclose(emb.numpy(), np.stack([b, a])), (
                        [r["symbol"] for r in hgnc], list(d), emb)
                emb, _ = ge.align_embeddings({"QARS": a, "QARS1": c},
                                             ["2058", "5859"], 3, "toy")
                assert torch.isnan(emb[0]).all()
                assert np.allclose(emb[1].numpy(), c)
            finally:
                gene_name_utils._default_mapper = saved


def test_download_cache_validation():
    with tempfile.TemporaryDirectory() as root:
        p = lambda name: os.path.join(root, name)
        open(p("empty.json"), "w").close()
        with open(p("page.txt"), "w") as f:
            f.write("<!DOCTYPE html><html>quota exceeded</html>")
        with open(p("fake.gz"), "w") as f:
            f.write("not gzip")
        with gzip.open(p("real.gz"), "wt") as f:
            f.write("x\n")
        with open(p("ok.json"), "w") as f:
            f.write('{"Gene_1": [0.1]}')
        assert not ge._cache_ok(p("empty.json"))
        assert not ge._cache_ok(p("page.txt"))
        assert not ge._cache_ok(p("fake.gz"))
        assert ge._cache_ok(p("real.gz")) and ge._cache_ok(p("ok.json"))
        assert not ge._cache_ok(p("missing.txt"))

        # A failed download leaves nothing behind, and an invalid cached
        # file is discarded rather than trusted
        import shutil
        import urllib.request
        saved_which, saved_get = shutil.which, urllib.request.urlretrieve

        def html_get(url, out):
            with open(out, "w") as f:
                f.write("<html>error</html>")

        shutil.which = lambda name: None
        urllib.request.urlretrieve = html_get
        try:
            assert not ge._ensure_download("http://x/y.json", p("empty.json"),
                                           "toy")
            assert not os.path.exists(p("empty.json"))
            assert not os.path.exists(p("empty.json.part"))
        finally:
            shutil.which, urllib.request.urlretrieve = saved_which, saved_get


def test_reused_pkl_is_not_stamped_as_current():
    """A cached builder pkl carries the current generator_version (and, for
    the protein models, sequence_source) only if built by this version."""
    vecs = {"1": np.ones(4, dtype=np.float32),
            "2": np.zeros(4, dtype=np.float32)}
    current = ge.EMBEDDING_GENERATOR_VERSION
    with tempfile.TemporaryDirectory() as root:
        pkl = os.path.join(root, "esm1b_gene_embeddings.pkl")
        out = os.path.join(root, "out.pt")

        def run():
            ge.generate_embedding("esm1b", ["1", "2"], out, root)
            return torch.load(out, map_location="cpu", weights_only=False)

        def fake_build(cache_dir):
            with open(pkl, "wb") as f:
                pickle.dump(vecs, f)
            return pkl

        saved = ge.extract_esm1b
        ge.extract_esm1b = fake_build
        try:
            # left behind by older code: no sidecar
            with open(pkl, "wb") as f:
                pickle.dump(vecs, f)
            d = run()
            assert d["generator_version"] == (
                "unknown (reused cached esm1b_gene_embeddings.pkl)")
            assert "sequence_source" not in d
            # built in this call, then reused with a matching sidecar
            os.remove(pkl)
            for _ in range(2):
                d = run()
                assert d["generator_version"] == current
                assert d["sequence_source"] == ge.SEQUENCE_SOURCE
            # changed after it was stamped
            with open(pkl, "wb") as f:
                pickle.dump({"1": vecs["2"]}, f)
            assert run()["generator_version"].startswith("unknown (reused")
        finally:
            ge.extract_esm1b = saved


def _toy_graph(seed=0, n=40):
    rng = np.random.RandomState(seed)
    genes = [f"G{i}" for i in range(n)]
    edges = [(genes[i], genes[i + 1]) for i in range(n - 1)]  # connected
    for _ in range(3 * n):
        i, j = rng.choice(n, 2, replace=False)
        edges.append((genes[i], genes[j]))
    return genes, edges


def test_mashup_matches_reference():
    genes, edges = _toy_graph()
    n = len(genes)
    saved = ge._load_string_ppi
    ge._load_string_ppi = lambda cache_dir: (edges, None)
    try:
        with tempfile.TemporaryDirectory() as root:
            out = ge.extract_mashup(root)
            with open(out, "rb") as f:
                emb_dict = pickle.load(f)
    finally:
        ge._load_string_ppi = saved
    order = sorted(genes)
    X = np.stack([emb_dict[g] for g in order]).astype(np.float64)
    assert X.shape == (n, n)  # k = min(500, n)

    # Direct transcription of rwr.m + mashup.m for one network
    idx = {g: i for i, g in enumerate(order)}
    A = np.zeros((n, n))
    for g1, g2 in edges:
        A[idx[g1], idx[g2]] = A[idx[g2], idx[g1]] = 1.0
    P = A / A.sum(axis=0, keepdims=True)
    Q = 0.5 * np.linalg.inv(np.eye(n) - 0.5 * P)
    R = np.log(Q + 1.0 / n)
    U, S, _ = np.linalg.svd(R)
    ref = U * np.sqrt(S)  # V d^(1/4) with d = S^2
    assert np.allclose(X @ X.T, ref @ ref.T, atol=1e-4)
    # Diffusion neighbours are similar (the old eigen-log version made
    # adjacent genes anti-similar)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = [Xn[idx[a]] @ Xn[idx[b]] for a, b in edges]
    assert np.mean(cos) > 0.2, np.mean(cos)


def test_canonical_signs():
    M = np.array([[0.1, -3.0], [-2.0, 1.0], [0.5, 0.2]])
    out = ge._canonical_signs(M)
    assert np.allclose(out, [[-0.1, 3.0], [2.0, -1.0], [-0.5, -0.2]])
    assert np.allclose(ge._canonical_signs(-M), out)


def test_prot_t5_mask_drops_eos_only():
    eos = 1
    ids = torch.tensor([[19, 14, 11, eos, 0, 0], [3, 4, 5, 6, 7, eos]])
    attn = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    mask = ge._prot_t5_residue_mask(ids, attn, eos)
    assert mask.tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]
    h = torch.randn(2, 6, 4)
    pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    assert torch.allclose(pooled[0], h[0, :3].mean(0))
    assert torch.allclose(pooled[1], h[1, :5].mean(0))


def test_synlethdb_bad_cache_is_replaced():
    header = "x:START_ID\tx_name\ty:END_ID\ty_name\trel_source\tcell_line\n"
    row = "672\tBRCA1\t5888\tRAD51\tCRISPR/CRISPRi\tK562\n"
    with tempfile.TemporaryDirectory() as root:
        raw = os.path.join(root, dsl.RAW_NAMES["NONSL"])
        with open(raw, "w") as f:
            f.write("<html>" + "Drive quota exceeded " * 100 + "</html>\n")
        assert not dsl.looks_like_synlethdb(raw)
        calls = []

        def fake_download(path, file_id, label):
            calls.append((os.path.basename(path), file_id))
            with open(path, "w") as f:
                f.write(header + row)
            return True

        saved_dl, saved_argv = dsl.download_tsv, sys.argv
        dsl.download_tsv = fake_download
        out = os.path.join(root, "neg.txt")
        try:
            sys.argv = ["download_synlethdb.py", "--relation", "NONSL",
                        "--output", out, "--cache_dir", root]
            dsl.main()
            assert calls == [(dsl.RAW_NAMES["NONSL"],
                              dsl.DRIVE_IDS["NONSL"])]
            with open(out) as f:
                assert f.read() == "672\t5888\n"
            # an overridden Drive ID never reuses the default cached file
            sys.argv += ["--file_id", "NEWID"]
            dsl.main()
            assert calls[-1] == ("Human.non.SL.detailed.NEWID.tsv", "NEWID")
        finally:
            dsl.download_tsv, sys.argv = saved_dl, saved_argv



def test_pool_parti_full_graph_like_authors():
    """PageRank on BOS+residues+EOS, BOS/EOS dropped afterwards (authors'
    token_to_sequence_pooler.py), not on the residue-only submatrix."""
    import networkx as nx
    rng = np.random.default_rng(0)
    pooler = gae.PoolPaRTI(alpha=0.85)
    for L, pad in [(12, 0), (20, 7)]:
        T, Tb = L + 2, L + 2 + pad
        heads = []
        for _ in range(6):
            logits = rng.normal(0, 1.5, size=(Tb, T))
            logits[:, 0] += 3.0  # BOS / EOS attention sinks
            logits[:, T - 1] += 2.0
            e = np.exp(logits - logits.max(1, keepdims=True))
            a = np.zeros((Tb, Tb))
            a[:, :T] = e / e.sum(1, keepdims=True)  # padded keys get 0
            heads.append(a)
        agg = np.max(np.stack(heads), 0).astype(np.float32)
        emb = rng.normal(size=(1, Tb, 8)).astype(np.float32)
        G = nx.from_numpy_array(agg[:T, :T].astype(np.float64),
                                create_using=nx.DiGraph)
        pr = nx.pagerank(G, alpha=0.85, tol=1e-06, weight="weight",
                         max_iter=100)
        w = np.array([pr[i] for i in range(1, L + 1)])
        w /= w.sum()
        ref = (emb[0, 1:L + 1] * w[:, None]).sum(0)
        got = pooler.pool(torch.from_numpy(emb), torch.from_numpy(agg), L)
        assert np.allclose(got, ref, atol=1e-5), np.abs(got - ref).max()
        old = pooler._compute_pagerank(agg[1:L + 1, 1:L + 1])
        assert not np.allclose(old / old.sum(), w, atol=1e-4)
    try:
        pooler._compute_pagerank(np.zeros((12, 12)))
        raise AssertionError("an edgeless graph must raise, as the authors'")
    except ValueError:
        pass


def test_protein_pkl_is_rebuilt_for_another_gene_list():
    calls = []
    with tempfile.TemporaryDirectory() as root:
        pkl = os.path.join(root, "esm1b_gene_embeddings.pkl")
        out = os.path.join(root, "out.pt")

        def build(cache_dir):
            calls.append(list(ge._ACTIVE_GENE_LIST))
            with open(pkl, "wb") as f:
                pickle.dump({g: np.full(4, float(g), dtype=np.float32)
                             for g in ge._ACTIVE_GENE_LIST}, f)
            return pkl

        saved = ge.extract_esm1b
        ge.extract_esm1b = build
        try:
            ge.generate_embedding("esm1b", ["1", "2"], out, root)
            ge.generate_embedding("esm1b", ["2", "1"], out, root)  # same set
            assert len(calls) == 1
            ge.generate_embedding("esm1b", ["1", "2", "3"], out, root)
            assert calls[-1] == ["1", "2", "3"]
            d = torch.load(out, map_location="cpu", weights_only=False)
            assert d["generator_version"] == ge.EMBEDDING_GENERATOR_VERSION
            assert not torch.isnan(d["embeddings"]).any()
            # builder unavailable: the old pkl is reused, marked down
            ge.extract_esm1b = lambda cache_dir: None
            ge.generate_embedding("esm1b", ["1", "2", "3", "4"], out, root)
            d = torch.load(out, map_location="cpu", weights_only=False)
            assert d["generator_version"].endswith(
                "built for another gene list)"), d["generator_version"]
            assert "sequence_source" not in d
        finally:
            ge.extract_esm1b = saved


def test_builder_runs_before_alt_and_subdir_files():
    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "out.pt")
        for path, v in [(os.path.join(root, "gene_to_geneformer_embed.pkl"),
                         7.0),
                        (os.path.join(root, "geneformer",
                                      "geneformer_gene_embeddings.pkl"), 8.0)]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"672": np.full(3, v, dtype=np.float32)}, f)
        prim = os.path.join(root, "geneformer_gene_embeddings.pkl")

        def build(cache_dir):
            with open(prim, "wb") as f:
                pickle.dump({"672": np.full(3, 1.0, dtype=np.float32)}, f)
            return prim

        saved = ge.extract_geneformer_from_hf
        ge.extract_geneformer_from_hf = build
        try:
            ge.generate_embedding("geneformer", ["672"], out, root)
            d = torch.load(out, map_location="cpu", weights_only=False)
            assert float(d["embeddings"][0, 0]) == 1.0
            assert d["generator_version"] == ge.EMBEDDING_GENERATOR_VERSION
            assert ge.GENEFORMER_REVISION in d["source_model"]
            os.remove(prim)  # build fails: the kept file, marked down
            ge.extract_geneformer_from_hf = lambda cache_dir: None
            import contextlib
            import io
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                ge.generate_embedding("geneformer", ["672"], out, root)
            # deleting the fallback file would not rebuild it: the build failed
            assert "build with the current code failed" in log.getvalue()
            assert "Delete the file" not in log.getvalue()
            d = torch.load(out, map_location="cpu", weights_only=False)
            assert d["generator_version"].startswith("unknown (reused cached")
            assert d["source"].startswith("unverified")
            assert "source_model" not in d
        finally:
            ge.extract_geneformer_from_hf = saved


def test_a_save_cut_short_leaves_nothing_under_the_output_name():
    # The SLURM loop counts `[ -f "$OUTPUT" ]` as generated and retries only
    # when the file is absent: a save that dies midway, or one torch.save
    # reports as done although the archive's directory was not written, must
    # leave no file under the output name.
    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "out.pt")
        prim = os.path.join(root, "geneformer_gene_embeddings.pkl")

        def build(cache_dir):
            with open(prim, "wb") as f:
                pickle.dump({"672": np.full(3, 1.0, dtype=np.float32)}, f)
            return prim

        real_save = torch.save

        def dies(obj, path, *a, **k):
            with open(path, "wb") as f:
                f.write(b"PK\x03\x04 half an archive")
            raise OSError(5, "Input/output error")

        def cut_tail(obj, path, *a, **k):
            real_save(obj, path, *a, **k)
            with open(path, "rb+") as f:
                f.truncate(os.path.getsize(path) - 64)

        saved = ge.extract_geneformer_from_hf
        ge.extract_geneformer_from_hf = build
        try:
            for fake in (dies, cut_tail):
                torch.save = fake
                try:
                    ge.generate_embedding("geneformer", ["672"], out, root)
                except OSError:
                    pass
                else:
                    raise AssertionError(f"{fake.__name__}: no error")
                finally:
                    torch.save = real_save
                assert not os.path.exists(out), fake.__name__
            ge.generate_embedding("geneformer", ["672"], out, root)
            assert os.path.isfile(out) and not os.path.exists(out + ".part")
            d = torch.load(out, map_location="cpu", weights_only=False)
            assert float(d["embeddings"][0, 0]) == 1.0
            assert d["generator_version"] == ge.EMBEDDING_GENERATOR_VERSION
        finally:
            ge.extract_geneformer_from_hf = saved
            torch.save = real_save


def test_geneformer_download_is_pinned():
    with tempfile.TemporaryDirectory() as root:
        good, bad = os.path.join(root, "good"), os.path.join(root, "bad")
        open(good, "wb").write(b"pinned")
        open(bad, "wb").write(b"moved")
        fname = "model.safetensors"
        saved_sha = dict(ge.GENEFORMER_SHA256)
        saved_mod = sys.modules.get("huggingface_hub")
        ge.GENEFORMER_SHA256[fname] = ge._file_sha256(good)
        revs = []

        def hub(pick):
            def hf_hub_download(repo, filename, revision=None):
                revs.append(revision)
                r = pick(revision)
                if isinstance(r, Exception):
                    raise r
                return r
            return types.SimpleNamespace(hf_hub_download=hf_hub_download)

        try:
            sys.modules["huggingface_hub"] = hub(lambda rev: good)
            assert ge._geneformer_file(fname) == good
            assert revs == [ge.GENEFORMER_REVISION]
            sys.modules["huggingface_hub"] = hub(
                lambda rev: OSError("offline") if rev else good)
            assert ge._geneformer_file(fname) == good  # same bytes: accepted
            sys.modules["huggingface_hub"] = hub(
                lambda rev: OSError("offline") if rev else bad)
            try:
                ge._geneformer_file(fname)
                raise AssertionError("other content must be refused")
            except RuntimeError as e:
                assert "sha256" in str(e)
        finally:
            ge.GENEFORMER_SHA256.clear()
            ge.GENEFORMER_SHA256.update(saved_sha)
            if saved_mod is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = saved_mod


def test_go_parser_refuses_to_run_without_the_hgnc_table():
    """Without the HGNC table (missing and not downloadable, or holding no
    parsable row) every GAF symbol maps to None, and the generator wrote an
    all-NaN all_genes_go.pt stamped as a current build, exit 0, which no
    later check rejects. parse_gaf stops instead; with a table it maps as
    before (NOT-qualified annotations left out)."""
    import contextlib
    import io
    import urllib.request
    from unittest import mock

    def offline(*a, **k):
        raise OSError("no network in a test")

    with tempfile.TemporaryDirectory() as tmp:
        gaf = os.path.join(tmp, "g.gaf")
        with open(gaf, "w") as f:
            f.write("!gaf-version: 2.2\n"
                    "UniProtKB\tP1\tBRCA1\tenables\tGO:0000001\n"
                    "UniProtKB\tP2\tATM\tNOT|enables\tGO:0000002\n"
                    "UniProtKB\tP3\tATM\tinvolved_in\tGO:0000003\n")
        tables = {"empty": None, "html": "<html>503</html>\n",
                  "ok": "symbol\tprev_symbol\talias_symbol\tentrez_id\n"
                        "BRCA1\t\t\t672\nATM\t\t\t472\n"}
        got = {}
        for name, text in tables.items():
            d = os.path.join(tmp, name)
            os.mkdir(d)
            if text is not None:
                with open(os.path.join(d, "hgnc_complete_set.txt"), "w") as f:
                    f.write(text)
            m = gene_name_utils.GeneNameMapper(cache_dir=d)
            with mock.patch.object(gg, "get_mapper", return_value=m), \
                    mock.patch.object(urllib.request, "urlretrieve",
                                      offline), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    got[name] = gg.parse_gaf(gaf)
                except RuntimeError as e:
                    got[name] = str(e)
        for name in ("empty", "html"):
            assert isinstance(got[name], str) \
                and "HGNC table not loaded" in got[name], (name, got[name])
        assert got["ok"] == {"672": {"GO:0000001"},
                             "472": {"GO:0000003"}}, got["ok"]


def test_go_sum_pool_is_order_free():
    rng = np.random.default_rng(1)
    terms = [f"GO:{i:07d}" for i in range(300)]
    embeds = {t: rng.normal(size=16).astype(np.float32) for t in terms}
    fwd, rev = set(), set()
    for t in terms:
        fwd.add(t)
    for t in reversed(terms):
        rev.add(t)
    m1, _ = gg.build_go_embeddings(["1"], {"1": fwd}, embeds)
    m2, _ = gg.build_go_embeddings(["1"], {"1": rev}, embeds)
    ref = np.zeros(16, dtype=np.float32)
    for t in sorted(terms):
        ref += embeds[t]
    assert np.array_equal(m1, m2) and np.array_equal(m1[0], ref)


def test_gene_list_txt_is_sorted_like_esm():
    with tempfile.TemporaryDirectory() as root:
        p = os.path.join(root, "gene_universe.txt")
        with open(p, "w") as f:
            f.write("1\n2\n9\n10\n11\n100\n2\n")
        assert ge.load_gene_list(p) == ["1", "10", "100", "11", "2", "9"]


def test_synlethdb_reuses_sibling_cache():
    header = "x:START_ID\tx_name\ty:END_ID\ty_name\trel_source\tcell_line\n"
    row = "672\tBRCA1\t5888\tRAD51\tCRISPR/CRISPRi\tK562\n"
    with tempfile.TemporaryDirectory() as root:
        emb_cache = os.path.join(root, "data", "embeddings_cache")
        os.makedirs(emb_cache)
        os.makedirs(os.path.join(root, "data", "cache"))
        with open(os.path.join(root, "data", "cache",
                               dsl.RAW_NAMES["NONSL"]), "w") as f:
            f.write(header + row)
        calls = []

        def fake_download(path, file_id, label):
            calls.append(path)
            with open(path, "w") as f:
                f.write(header + row)
            return True

        saved_dl, saved_argv = dsl.download_tsv, sys.argv
        dsl.download_tsv = fake_download
        out = os.path.join(root, "neg.txt")
        try:
            sys.argv = ["download_synlethdb.py", "--relation", "NONSL",
                        "--output", out, "--cache_dir", emb_cache]
            dsl.main()
            assert calls == []  # the tracked data/cache copy, no Drive
            with open(out) as f:
                assert f.read() == "672\t5888\n"
            sys.argv += ["--file_id", "NEWID"]  # another ID: never reused
            dsl.main()
            assert calls == [os.path.join(
                emb_cache, "Human.non.SL.detailed.NEWID.tsv")]
        finally:
            dsl.download_tsv, sys.argv = saved_dl, saved_argv


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
