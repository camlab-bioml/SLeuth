# Gene Embeddings Reference Guide

A comprehensive reference of available gene embedding methods, organized by data modality. Based in part on the benchmarking study by Zhong et al. (2025), which evaluated 38 gene embedding methods across multiple functional prediction tasks.

---

## 1. Gene Expression–Based (Single-Cell Foundation Models)

These are trained on scRNA-seq data. The embedding for each gene is a learned token/vocabulary vector from the transformer.

### Geneformer

- **What it encodes:** Co-expression and functional context from ~30 million single-cell transcriptomes
- **Architecture:** Transformer with masked gene prediction objective
- **Dimensions:** 256
- **Reference:** Theodoris et al. (2023). *Transfer learning enables predictions in network biology.* Nature 618, 616–624.
- **GitHub:** <https://github.com/ctheodoris/Geneformer>
- **HuggingFace:** `ctheodoris/Geneformer`

**Loading code:**

```python
# pip install transformers torch

from transformers import AutoModel
import pickle

model = AutoModel.from_pretrained("ctheodoris/Geneformer")

# Gene embeddings are the token embedding layer
gene_embeddings = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
# Shape: (num_tokens, 256)

# To map gene names to token indices, load the token dictionary
# (distributed with the Geneformer repo as token_dictionary.pkl)
with open("token_dictionary.pkl", "rb") as f:
    token_dict = pickle.load(f)

# Look up a specific gene
gene_name = "BRCA1"
if gene_name in token_dict:
    idx = token_dict[gene_name]
    brca1_embedding = gene_embeddings[idx]  # shape: (256,)
```

Alternatively, precomputed embeddings are available from the **GenePert** repo:

```python
# Download from: https://github.com/zou-group/GenePert
import pickle
with open("geneformer_gene_embeddings.pkl", "rb") as f:
    geneformer_embs = pickle.load(f)
# geneformer_embs is a dict: gene_name -> numpy array of shape (256,)
```

---

### scGPT

- **What it encodes:** Gene expression relationships across 33M+ single cells
- **Architecture:** Generative transformer (53M parameters) with masked language modeling
- **Dimensions:** 512
- **Reference:** Cui et al. (2024). *scGPT: towards building a foundation model for single-cell multi-omics using generative AI.* Nature Methods 21, 1470–1480.
- **GitHub:** <https://github.com/bowang-lab/scGPT>

**Loading code:**

```python
import json
import torch

# 1. Load the vocabulary (gene name -> token index)
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# 2. Load the pretrained model checkpoint
# Download from: https://github.com/bowang-lab/scGPT (see model zoo)
checkpoint = torch.load("scGPT_human/model.pt", map_location="cpu")

# 3. Extract gene token embeddings from the checkpoint
# The key name may vary by checkpoint version; common keys:
gene_embeddings = checkpoint["encoder.embedding.gene_encoder.embedding.weight"].numpy()
# Shape: (num_genes, 512)

# 4. Look up a gene
gene_name = "TP53"
if gene_name in vocab:
    idx = vocab[gene_name]
    tp53_embedding = gene_embeddings[idx]  # shape: (512,)
```

Alternatively, precomputed embeddings are available from the **GenePert** repo:

```python
import pickle
with open("scgpt_gene_embeddings.pkl", "rb") as f:
    scgpt_embs = pickle.load(f)
```

---

### Gene2Vec (Du et al., 2019)

- **What it encodes:** Gene co-expression patterns across microarray datasets
- **Architecture:** Word2Vec-style skip-gram on co-expression contexts
- **Dimensions:** 200
- **Reference:** Du et al. (2019). *Gene2vec: distributed representation of genes based on co-expression.* BMC Genomics 20, 82.
- **GitHub:** <https://github.com/jingcheng-du/Gene2vec>

**Loading code:**

```python
import numpy as np

# Download gene2vec_dim_200_iter_9_w2v.txt from the Gene2vec repo
# Format: each line is "gene_name dim1 dim2 ... dim200"

gene_embeddings = {}
with open("gene2vec_dim_200_iter_9_w2v.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        gene_name = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        gene_embeddings[gene_name] = vector

# Look up a gene
brca1_emb = gene_embeddings.get("BRCA1")  # shape: (200,)
```

---

### scPRINT

- **What it encodes:** Gene identity (via ESM2 protein embeddings), expression level, and genomic location
- **Architecture:** Large transformer pretrained on 50M+ cells from cellxgene
- **Reference:** (2025). *scPRINT: pre-training on 50 million cells allows robust gene network predictions.* Nature Communications.
- **GitHub:** <https://github.com/cantinilab/scPRINT>

**Loading code:**

```python
# pip install scprint

from scprint import scPrint

# Load the pretrained model
model = scPrint.load_from_checkpoint("path/to/scprint_checkpoint.ckpt")

# Gene ID embeddings are ESM2-derived protein embeddings stored in the model
# Access via the gene encoder component
gene_embeddings = model.gene_encoder.weight.detach().cpu().numpy()

# The gene-to-index mapping is stored in the model's gene vocabulary.
# Consult the scPRINT documentation for the exact attribute name,
# as it may vary by version.
```

---

## 2. Protein Sequence–Based (Protein Language Models)

These embed the amino acid sequence of each gene's protein product. The general procedure is: gene → canonical protein sequence (via UniProt) → run through PLM → mean-pool amino acid embeddings → one vector per gene.

### ESM-2 (Meta/FAIR)

- **What it encodes:** Protein structure and function from amino acid sequences
- **Architecture:** Transformer-based protein language model; multiple sizes (8M to 15B parameters)
- **Recommended model:** `esm2_t33_650M_UR50D` (650M parameters, good performance/cost tradeoff)
- **Benchmark performance:** Strong for functional and genetic interaction prediction tasks (AUROC: SL=0.82, NG=0.75, TF=0.75 in Zhong et al.)
- **Reference:** Lin et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science 379, 1123–1130.
- **GitHub:** <https://github.com/facebookresearch/esm>

**Loading code:**

```python
# pip install torch fair-esm

import torch
import esm
import numpy as np

# 1. Load the model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# 2. Prepare protein sequences (gene -> UniProt canonical sequence)
# Example: BRCA1 protein sequence (truncated here for illustration)
data = [
    ("BRCA1", "MDLSALREVEN..."),  # full amino acid sequence from UniProt
    ("TP53",  "MEEPQSDPSVE..."),
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 3. Extract embeddings
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)

# 4. Mean-pool over amino acid positions (excluding BOS/EOS tokens)
token_representations = results["representations"][33]  # (batch, seq_len, 1280)

gene_embeddings = {}
for i, (label, seq) in enumerate(data):
    # Mean-pool over actual residue positions (skip BOS at position 0)
    emb = token_representations[i, 1:len(seq)+1].mean(0).numpy()
    gene_embeddings[label] = emb  # shape: (1280,)
```

**For many genes at scale:**

```python
# Process in batches; for ~20,000 human genes this takes ~1-2 hours on an RTX 4090.
# Download all human protein sequences from UniProt:
#   https://www.uniprot.org/uniprotkb?query=organism_id:9606+AND+reviewed:true
# Parse the FASTA, then batch through ESM-2 as above.
```

---

### ESM-1b

- **What it encodes:** Same as ESM-2 (protein sequence → structure/function)
- **Architecture:** 650M parameter transformer, predecessor to ESM-2
- **Training data:** 250M protein sequences (UniRef50)
- **Reference:** Rives et al. (2021). *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* PNAS 118, e2016239118.
- **GitHub:** <https://github.com/facebookresearch/esm>

**Loading code:**

```python
# Same as ESM-2, just swap the model:
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# Everything else (batch_converter, mean-pooling) is identical.
```

---

### ProtTrans / ProtT5

- **What it encodes:** Protein sequence information via T5 architecture
- **Training data:** UniRef and BFD databases
- **Reference:** Elnaggar et al. (2022). *ProtTrans: Toward understanding the language of life through self-supervised learning.* IEEE TPAMI 44(10), 7112–7127.
- **HuggingFace:** `Rostlab/prot_t5_xl_uniref50`

**Loading code:**

```python
# pip install transformers torch sentencepiece

from transformers import T5Tokenizer, T5EncoderModel
import torch

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model.eval()

# Protein sequences must have spaces between each amino acid
sequence = "M D L S A L R E V E N"  # space-separated amino acids

ids = tokenizer(sequence, return_tensors="pt", padding=True)

with torch.no_grad():
    output = model(**ids)

# Mean-pool over residue positions (exclude padding)
attention_mask = ids["attention_mask"].unsqueeze(-1)
emb = (output.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
gene_embedding = emb.squeeze().numpy()  # shape: (1024,)
```

---

### SeqVec

- **What it encodes:** Contextual protein sequence representations
- **Architecture:** ELMo-style model trained on UniRef50
- **Reference:** Heinzinger et al. (2019). *Modeling aspects of the language of life through transfer-learning protein sequences.* BMC Bioinformatics 20, 723.
- **GitHub:** <https://github.com/mheinzinger/SeqVec>

**Loading code:**

```python
# pip install seqvec

from seqvec import SeqVec
import numpy as np

# Download weights from: https://github.com/mheinzinger/SeqVec#download-pre-trained-model
model = SeqVec(
    model_dir="path/to/uniref50_v2/",
    weights="path/to/weights.hdf5"
)

# Input: list of protein sequences (as strings, no spaces)
sequences = ["MDLSALREVEN...", "MEEPQSDPSVE..."]

embeddings = model.embed(sequences)
# Each element is shape (3, seq_len, 1024) — 3 ELMo layers

# Mean-pool over layers and residue positions
gene_embeddings = [np.mean(emb, axis=(0, 1)) for emb in embeddings]  # shape: (1024,)
```

---

## 3. Biomedical Literature / Text–Based

These embed genes via their textual descriptions from databases or scientific literature.

### GenePT (Chen & Zou, 2025)

- **What it encodes:** Literature-derived gene knowledge from NCBI gene summaries (and optionally UniProt protein descriptions)
- **Architecture:** Uses OpenAI GPT-3.5 text embedding API on gene description text
- **Variants:** NCBI summary only; NCBI + UniProt; gene name only
- **Benchmark performance:** Top performer for disease-gene prediction (mean AUROC=0.88 for Model3 in Zhong et al.)
- **Precomputed:** <https://zenodo.org/records/10833191>
- **Reference:** Chen & Zou (2025). *Simple and effective embedding model for single-cell biology built from ChatGPT.* Nature Biomedical Engineering 9, 483–493.
- **GitHub:** <https://github.com/yiqunchen/GenePT>

**Loading precomputed embeddings (recommended):**

```python
import pickle
import numpy as np

# Download from Zenodo: https://zenodo.org/records/10833191
# File: GenePT_NCBI+UniProt_3.5.pkl (or other variant)
with open("GenePT_NCBI+UniProt_3.5.pkl", "rb") as f:
    genept_embs = pickle.load(f)

# genept_embs is a dict: gene_name -> numpy array
brca1_emb = genept_embs["BRCA1"]  # shape: (1536,) for GPT-3.5 embeddings
```

**Generating from scratch (using OpenAI API):**

```python
# pip install openai biopython

from Bio import Entrez
import openai
import numpy as np

Entrez.email = "your@email.com"

# 1. Fetch NCBI gene summary
handle = Entrez.efetch(db="gene", id="672", rettype="docsum", retmode="xml")
record = Entrez.read(handle)
summary = record["DocumentSummarySet"]["DocumentSummary"][0]["Summary"]

# 2. Get embedding from OpenAI
client = openai.OpenAI()
response = client.embeddings.create(
    input=summary,
    model="text-embedding-3-small"  # or text-embedding-ada-002
)
embedding = np.array(response.data[0].embedding)
```

---

### Open-Source LLM Text Embeddings (as GenePT replacements)

- **What they encode:** Same as GenePT — literature-derived gene knowledge — but using open-source models instead of OpenAI
- **Models tested:** `mxbai-embed-large-v1` (335M params), `SFR-Embedding-Mistral` (7B params), and others from HuggingFace
- **Advantage:** No API costs, data privacy preserved, runs locally
- **Reference:** (2025). *Small, open-source text-embedding models as substitutes to OpenAI models for gene analysis.* PMC.

**Loading code:**

```python
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import json

# 1. Load NCBI gene summaries (prepare a JSON: {gene_name: summary_text})
with open("ncbi_gene_summaries.json", "r") as f:
    gene_summaries = json.load(f)

# 2. Load an open-source embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 3. Encode all gene summaries
gene_names = list(gene_summaries.keys())
texts = [gene_summaries[g] for g in gene_names]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# 4. Build a lookup dict
gene_embeddings = dict(zip(gene_names, embeddings))
# Each embedding shape: (1024,) for mxbai-embed-large-v1
```

**To fetch NCBI summaries programmatically:**

```python
from Bio import Entrez
Entrez.email = "your@email.com"

# Fetch summaries for a list of gene IDs
gene_ids = ["672", "7157", "5290"]  # BRCA1, TP53, PIK3CA
handle = Entrez.efetch(db="gene", id=",".join(gene_ids), rettype="docsum", retmode="xml")
records = Entrez.read(handle)
for rec in records["DocumentSummarySet"]["DocumentSummary"]:
    name = rec["NomenclatureSymbol"]
    summary = rec["Summary"]
    print(f"{name}: {summary[:100]}...")
```

---

### BioConceptVec (Chen et al., 2020)

- **What it encodes:** Contextual biomedical concept semantics from PubMed literature
- **Architecture:** NER-based concept extraction + Word2Vec/FastText on ~30 million PubMed abstracts
- **Coverage:** 400,000+ biomedical concepts including genes
- **Benchmark performance:** BioConceptVec-FastText achieved comparable performance to GenePT with significantly fewer dimensions (~16.5x more computationally efficient)
- **Reference:** Chen et al. (2020). *BioConceptVec: Creating and evaluating literature-based biomedical concept embeddings on a large scale.* PLOS Computational Biology 16, e1007617.
- **GitHub:** <https://github.com/ncbi-nlp/BioConceptVec>

**Loading code:**

```python
import numpy as np
from gensim.models import KeyedVectors

# Download from: https://github.com/ncbi-nlp/BioConceptVec
# File: BioConceptVec_word2vec_cbow.bin or FastText variant
# Format: standard Word2Vec binary format

wv = KeyedVectors.load_word2vec_format("BioConceptVec_word2vec_cbow.bin", binary=True)

# Gene concepts are indexed by NCBI Gene ID prefixed with "Gene:"
# e.g., BRCA1 (Gene ID 672) -> "Gene:672"
brca1_emb = wv["Gene:672"]  # shape: (200,) for word2vec, (100,) for fasttext

# To map gene symbols to NCBI IDs, use mygene:
# pip install mygene
import mygene
mg = mygene.MyGeneInfo()
result = mg.query("BRCA1", scopes="symbol", fields="entrezgene", species="human")
gene_id = result["hits"][0]["entrezgene"]  # "672"
emb = wv[f"Gene:{gene_id}"]
```

---

## 4. PPI Network–Based

These embed genes based on their position in protein–protein interaction networks.

### Node2Vec on PPI

- **What it encodes:** Topological position in PPI networks via biased random walks
- **Architecture:** Skip-gram on random walk sequences (same idea as Word2Vec on sentences, but "sentences" are random walks on the graph)
- **Typical dimensions:** 128
- **Benchmark performance:** Mean AUROC=0.74 for disease-gene prediction; strong for pairwise interaction tasks
- **Reference:** Grover & Leskovec (2016). *node2vec: Scalable feature learning for networks.* KDD 2016, 855–864.
- **GitHub:** <https://github.com/aditya-grover/node2vec>

**Loading code:**

```python
# pip install node2vec networkx pandas

import networkx as nx
from node2vec import Node2Vec
import pandas as pd

# 1. Load a PPI network (example: STRING database)
# Download from: https://string-db.org/cgi/download
# Format: protein1 protein2 combined_score
edges = pd.read_csv("9606.protein.links.v12.0.txt.gz", sep=" ")
edges = edges[edges["combined_score"] >= 700]  # high-confidence filter

G = nx.from_pandas_edgelist(edges, "protein1", "protein2")

# 2. Train Node2Vec
node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=80,
    num_walks=10,
    p=1.0,    # return parameter
    q=1.0,    # in-out parameter
    workers=4
)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 3. Extract gene embeddings
gene_embedding = model.wv["9606.ENSP00000350283"]  # STRING protein ID
# shape: (128,)

# To convert STRING IDs to gene symbols, use the STRING alias file
# or the mygene package.
```

---

### Mashup (Cho et al., 2016)

- **What it encodes:** Multi-network gene topology via diffusion-based matrix factorization
- **Input:** Multiple PPI and genetic interaction networks simultaneously
- **Benchmark performance:** Mean AUROC=0.78 for disease-gene prediction, outperforming most non-text methods
- **Reference:** Cho et al. (2016). *Compact integration of multi-network topology for functional analysis of genes.* Cell Systems 3(6), 540–548.
- **Website:** <http://cb.csail.mit.edu/cb/mashup/>

**Loading code (MATLAB — original implementation):**

```matlab
% Download from: http://cb.csail.mit.edu/cb/mashup/

% 1. Load PPI adjacency matrices (one per network)
load('network1.mat');  % adjacency matrix A1
load('network2.mat');  % adjacency matrix A2

% 2. Run Mashup
ndim = 500;  % embedding dimensions
embeddings = mashup({A1, A2}, ndim);
% embeddings: (num_genes x ndim) matrix
```

**Loading code (Python re-implementation):**

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def mashup_embed(adjacency_matrices, ndim=500, alpha=0.5):
    diffusion_profiles = []
    for A in adjacency_matrices:
        D_inv = np.diag(1.0 / np.maximum(A.sum(axis=1), 1e-10))
        W = D_inv @ A
        # RWR: Q = alpha * (I - (1-alpha)*W)^{-1}
        n = A.shape[0]
        Q = alpha * np.linalg.inv(np.eye(n) - (1 - alpha) * W)
        Q = np.log(Q + 1e-10)  # log transform
        diffusion_profiles.append(Q)
    # Stack and SVD
    stacked = np.hstack(diffusion_profiles)
    U, S, Vt = svds(csr_matrix(stacked), k=ndim)
    return U * S  # shape: (num_genes, ndim)
```

---

### PPI-RAW (Adjacency Baseline)

- **What it encodes:** Direct PPI adjacency — each gene's row in the adjacency matrix is its "embedding"
- **No training required**
- **Benchmark performance:** Second-best overall for pairwise interaction tasks in Zhong et al. (AUROCs: SL=0.79, NG=0.73, TF=0.74), surpassing most learned embedding methods
- **Note:** This is a very strong baseline — worth trying before more complex methods.

**Loading code:**

```python
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

# 1. Load PPI edges
edges = pd.read_csv("9606.protein.links.v12.0.txt.gz", sep=" ")
edges = edges[edges["combined_score"] >= 700]

# 2. Build adjacency matrix
proteins = sorted(set(edges["protein1"]) | set(edges["protein2"]))
prot2idx = {p: i for i, p in enumerate(proteins)}
n = len(proteins)

A = lil_matrix((n, n))
for _, row in edges.iterrows():
    i, j = prot2idx[row["protein1"]], prot2idx[row["protein2"]]
    A[i, j] = 1
    A[j, i] = 1

A = A.tocsr()

# 3. Each gene's "embedding" is its row in A
# For gene at index i: embedding = A[i, :].toarray().flatten()
# Dimensionality = number of genes (high-dimensional, sparse)
```

---

## 5. Gene Ontology–Based

These embed genes through their GO annotations or the GO graph structure.

### GO2Vec

- **What it encodes:** GO term semantics from the GO directed acyclic graph (DAG), optionally including protein-GO annotation edges
- **Architecture:** Node2Vec applied to the GO graph (and/or GOA graph including protein annotations)
- **Reference:** Zhong et al. (2020). *GO2Vec: transforming GO terms and proteins to vector representations via graph embeddings.* BMC Genomics 20(Suppl 1), 919.

**Loading code:**

```python
# pip install node2vec networkx obonet

import obonet
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# 1. Load the Gene Ontology graph
url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
go_graph = obonet.read_obo(url)

# Convert to undirected for Node2Vec
G = go_graph.to_undirected()

# 2. Train Node2Vec on the GO DAG
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=20)
model = node2vec.fit()

# 3. GO term embeddings
go_term_emb = model.wv["GO:0008150"]  # biological_process, shape: (128,)

# 4. To get gene embeddings, aggregate GO terms annotated to each gene
# Load GO annotations (GAF format) from:
#   http://geneontology.org/docs/download-go-annotations/
# gene_to_go: dict mapping gene_symbol -> list of GO term IDs
# (parse from the GAF file)

def gene_embedding_from_go(gene, gene_to_go, model):
    go_terms = gene_to_go.get(gene, [])
    vectors = [model.wv[t] for t in go_terms if t in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return None
```

---

### Anc2Vec

- **What it encodes:** GO term structure preserving ontological uniqueness, ancestor hierarchy, and sub-ontology membership
- **Architecture:** Neural network–based embedding of GO terms
- **Reference:** Edera et al. (2022). *Anc2vec: embedding gene ontology terms by preserving ancestors relationships.* Briefings in Bioinformatics 23(2), bbac003.
- **GitHub:** <https://github.com/sinc-lab/anc2vec>

**Loading code:**

```python
# Clone: git clone https://github.com/sinc-lab/anc2vec
# Follow the README to train or load pretrained embeddings

import pickle
import numpy as np

# Pretrained embeddings are provided in the repo
with open("anc2vec/embeddings/anc2vec_bp.pkl", "rb") as f:
    go_embeddings = pickle.load(f)

# go_embeddings: dict of GO_term_id -> numpy array
# To get gene-level embeddings, aggregate over annotated GO terms
# (same procedure as GO2Vec above)
```

---

### Onto2Vec

- **What it encodes:** GO ontology axioms treated as sentences, embedded via Word2Vec
- **Reference:** Smaili et al. (2018). *OPA2Vec: combining formal and informal content of biomedical ontologies to improve similarity-based prediction.* Bioinformatics 35(12), 2133–2140.
- **GitHub:** <https://github.com/bio-ontology-research-group/onto2vec>

**Loading code:**

```python
# Clone: git clone https://github.com/bio-ontology-research-group/onto2vec
# The pipeline:
#   1. Convert GO axioms into "sentences"
#   2. Train Word2Vec on those sentences
#   3. Extract term vectors

# After running the Onto2Vec pipeline (see repo README):
from gensim.models import Word2Vec

model = Word2Vec.load("onto2vec_model")
go_term_emb = model.wv["GO:0008150"]

# Gene embeddings: aggregate annotated GO terms (same as GO2Vec/Anc2Vec)
```

---

## 6. Knowledge Graph–Based

### TransE / ComplEx / DistMult / RotatE on Biological KGs

- **What they encode:** Gene relationships within heterogeneous biological knowledge graphs (gene–disease, gene–pathway, gene–drug edges)
- **Flexibility:** You define the KG schema and edge types; the embedding method is agnostic.
- **Libraries:** PyKEEN (`https://github.com/pykeen/pykeen`), DGL-KE, or LibKGE for training.

**Loading code (using PyKEEN):**

```python
# pip install pykeen torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import pandas as pd

# 1. Prepare triples: (head, relation, tail)
# Example: gene-disease associations from DisGeNET
triples = pd.DataFrame([
    ("BRCA1", "associated_with", "Breast_Cancer"),
    ("TP53",  "associated_with", "Li_Fraumeni_Syndrome"),
    ("BRCA1", "interacts_with",  "BARD1"),
    # ... thousands more from KEGG, Reactome, DisGeNET, etc.
], columns=["head", "relation", "tail"])

tf = TriplesFactory.from_labeled_triples(triples.values)

# 2. Train a KG embedding model
result = pipeline(
    training=tf,
    model="ComplEx",       # or TransE, DistMult, RotatE
    model_kwargs=dict(embedding_dim=256),
    training_kwargs=dict(num_epochs=100),
)

# 3. Extract gene embeddings
model = result.model
entity_to_id = tf.entity_to_id
brca1_id = entity_to_id["BRCA1"]
brca1_emb = model.entity_representations[0](
    torch.tensor([brca1_id])
).detach().numpy().flatten()  # shape: (256,) for ComplEx real part
```

---

## Benchmark Summary by Task Type

Based on Zhong et al. (2025) benchmarking results:

| Task Type | Best Performing Modalities |
|---|---|
| Disease–gene prediction | Biomedical literature (GenePT, BioConceptVec) |
| Gene function (GO) prediction | Amino acid sequence (ESM2, ESM-1b, T5); biomedical literature |
| Pairwise interaction prediction (SL, NG, TF) | PPI-based (Node2Vec, Mashup, PPI-RAW); amino acid sequence (ESM2, ESM-1b) |
| Gene-level attribute prediction | Biomedical literature; PPI-based |

---

## Practical Recommendations for Edge Prediction

For graph-based edge prediction with rare binary outcomes:

1. **Start with GenePT** — precomputed, free download from Zenodo, no GPU needed. Covers literature-derived functional knowledge.
2. **Compare against ESM-2** — requires running protein sequences through the model (GPU helps but 650M model is manageable on an RTX 4090; ~1-2 hours for all human genes).
3. **Try PPI-RAW adjacency** — zero compute cost, surprisingly strong baseline for interaction tasks.
4. **Consider concatenating modalities** — e.g., GenePT + ESM-2 + PPI features. Different modalities capture complementary biological signals.
5. **Beware of data leakage with literature-based embeddings** — if your prediction task overlaps with curated knowledge (e.g., predicting known interactions that appear in PubMed), text-based embeddings may leak label information through the literature.
