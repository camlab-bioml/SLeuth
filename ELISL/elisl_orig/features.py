# -*- coding: utf-8 -*-
"""ELISL's context-free feature sets (Sequence_1024, PPI_ec), vendored.

Source: https://github.com/joanagoncalveslab/ELISL, commit
eaf5f8c502a6db580722d73ceef8b41f4e4fe1b7, files
  src/feature_generation/sequence_features.py (sha256 b2c5e4b4adf2...)
  src/feature_generation/ppi_features.py      (sha256 7f1c611bf5f3...)
  src/embedding/ppi_embedding.py              (sha256 21d1653eeb2d...)
  src/data_functions.py                       (sha256 4acde3a2051f...)
Copyright (C) ytepeli; GNU GPL v3 (ELISL/LICENSE).

The authors' functions read their inputs from argparse globals and
config.DATA_DIR. Here the same bodies take those paths as arguments; every
other line is theirs unless marked `# HARNESS:`. Changes:
  * paths are parameters (no argparse / config globals);
  * get_extra_mapping (PPI/STRING/hand_mapping.tsv) and the two
    uniprot_extra_mapping.tab readers return empty maps when the file is
    absent. Those three hand-curated files are part of the authors' data
    release (SurfDrive / figshare 10.6084/m9.figshare.23607558), not of the
    GitHub repository; get_uniprot_extra_mapping already behaved this way,
    get_extra_mapping raised;
  * get_embs_dict: gensim 4 renamed KeyedVectors.index2word / .syn0 to
    .index_to_key / .vectors (same arrays);
  * gene_vectors + pair_features: a vectorised form of the authors'
    per-row loop in create_seq_features / create_ppi_features (same values;
    test_elisl.py checks it against the authors' functions, kept below
    verbatim), used to score all gene pairs without building a DataFrame
    per pair.
"""
import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# src/data_functions.py
# ----------------------------------------------------------------------------


def get_PPI_data(data_loc, cutoff_list=[0.0], source='ec'):
    # HARNESS: data_loc is a parameter (was config.DATA_DIR / in_file)
    df = pd.read_csv(data_loc, sep=" ")
    cutoff_dict = {}
    for cutoff in cutoff_list:
        if 'e' in source and 'c' in source:
            cutoff_df = df[(df['experiments'] > cutoff * 1000) | (df['database'] > cutoff * 1000)]
        elif 'e' in source:
            cutoff_df = df[df['experiments'] > cutoff * 1000]
        elif 'c' in source:
            cutoff_df = df[df['database'] > cutoff * 1000]

        cutoff_dict[str(cutoff)] = cutoff_df
    # HARNESS: the original's unused summary statistics (unique_genes, mean,
    # max, min of 'experiments') are not computed.
    return cutoff_dict, df


def get_PPI_genename_maps(data_loc):
    df = pd.read_csv(data_loc, sep="\t", header=None, names=['taxid', 'name', 'STRING'], skiprows=1)
    name2string = df.set_index('name')['STRING'].to_dict()
    string2name = df.set_index('STRING')['name'].to_dict()

    return name2string, string2name


def get_PPI_uniprot_maps(data_loc, aim='name'):
    df = pd.read_csv(data_loc, sep="\t", header=None, names=['taxid', 'combination', 'STRING', 's1', 's2'])
    df[['uniprot', 'name']] = df['combination'].str.split('|', n=1, expand=True)
    df['name'] = df['name'].str.split('_', n=1, expand=True)[0]
    df = df.drop(columns=['taxid', 'combination', 's1', 's2'])
    if aim == 'name':
        name2string = df.set_index('name')['STRING'].to_dict()
        string2name = df.set_index('STRING')['name'].to_dict()
    elif aim == 'uniprot':
        name2string = df.set_index('uniprot')['STRING'].to_dict()
        string2name = df.set_index('STRING')['uniprot'].to_dict()

    return name2string, string2name


def get_extra_mapping(loc):
    if loc is None or not os.path.exists(loc):  # HARNESS: absent -> empty
        return {}, {}
    df = pd.read_csv(loc, names=['name','STRING'], header=None, sep='\t')
    name2string = df.set_index('name')['STRING'].to_dict()
    string2name = df.set_index('STRING')['name'].to_dict()

    return name2string, string2name


# ----------------------------------------------------------------------------
# src/embedding/ppi_embedding.py
# ----------------------------------------------------------------------------

# The authors' node2vec options (get_node2vec_emb / main()).
NODE2VEC_OPTS = {'dimensions': 64, 'walk_length': 30, 'num_walks': 200, 'p': 1, 'q': 1, 'workers': 4,
                 'window': 10, 'min_count': 1, 'batch_words': 4}


def get_node2vec_emb(ppi_graph, embed_loc, opts_extra={}):
    # HARNESS: writes to embed_loc (was get_save_loc under config.DATA_DIR);
    # the gensim model file is not saved (nothing reads it).
    from node2vec import Node2Vec
    opts = dict(NODE2VEC_OPTS)
    opts.update(opts_extra)

    node2vec = Node2Vec(ppi_graph, dimensions=opts['dimensions'], p=opts['p'], q=opts['q'],
                        walk_length=opts['walk_length'], num_walks=opts['num_walks'], workers=opts['workers'])
    model = node2vec.fit(window=opts['window'], min_count=opts['min_count'], batch_words=opts['batch_words'])

    # Save embeddings
    model.wv.save_word2vec_format(embed_loc)
    return opts


def load_embs_from_raw_loc(embed_loc=''):
    import gensim
    loaded_vectors = gensim.models.KeyedVectors.load_word2vec_format(embed_loc)
    print(f'Embeddings at {embed_loc} are loaded')
    return loaded_vectors


def get_embs_dict(w2v_vectors):
    # HARNESS: gensim 4 names (index2word -> index_to_key, syn0 -> vectors)
    return dict(zip(w2v_vectors.index_to_key, w2v_vectors.vectors))


# ----------------------------------------------------------------------------
# src/embedding/sequence_embedding.py: load_embeddings (npz branch)
# ----------------------------------------------------------------------------

def load_seq_embeddings(emb_loc):
    import collections as coll
    data = np.load(emb_loc)  # type: Dict[str, np.ndarray]
    data = dict(data)
    data = coll.OrderedDict(sorted(data.items()))
    return data


# ----------------------------------------------------------------------------
# src/feature_generation/{sequence,ppi}_features.py
# ----------------------------------------------------------------------------


def get_uniprot_extra_mapping(loc):
    if loc is None or not os.path.exists(loc):
        return {}, {}
    data = pd.read_csv(loc, sep='\t')
    data.columns = ['query', 'uniprot_id', 'uniprot_name', 'status', 'prot_names', 'gene_names', 'organism', 'primary_gene_names','synonym_gene_names']
    reviewed = data[data['status'] == 'reviewed']

    reviewed['uniprot_name'] = reviewed['uniprot_name'].str.split('_', n=1, expand=True)[0]

    query2uniprot = reviewed.set_index('query')['uniprot_name'].to_dict()
    query2genename = reviewed.set_index('query')['primary_gene_names'].to_dict()
    return query2uniprot, query2genename


def get_seq_mappings(mapping_loc, extra_mapping_loc=None):
    """sequence_features.get_mappings."""
    mappings = []
    uniprot_fasta_mapping_loc = mapping_loc
    fasta_df = pd.read_csv(uniprot_fasta_mapping_loc)
    name2uid = fasta_df.set_index('gene_name')['uniprot_id'].to_dict()
    mappings.append(name2uid)
    uname2uid = fasta_df.set_index('uniprot_name')['uniprot_id'].to_dict()
    mappings.append(uname2uid)
    extra_map1, extra_map2 = get_uniprot_extra_mapping(extra_mapping_loc)
    extra_map1_dict = {}
    for key, val in extra_map1.items():
        if val in mappings[0].keys():
            extra_map1_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map1_dict[key] = mappings[1][val]
    mappings.append(extra_map1_dict)
    extra_map2_dict = {}
    for key, val in extra_map2.items():
        if val in mappings[0].keys():
            extra_map2_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map2_dict[key] = mappings[1][val]
    mappings.append(extra_map2_dict)

    return mappings


def get_ppi_mappings(name2string_loc, uniprot2string_loc, hand_mapping_loc=None,
                     extra_mapping_loc=None):
    """ppi_features.get_mappings."""
    mappings = []
    mappings.append(get_PPI_genename_maps(name2string_loc)[0])
    mappings.append(get_PPI_uniprot_maps(uniprot2string_loc, aim='name')[0])
    mappings.append(get_extra_mapping(hand_mapping_loc)[0])
    extra_map1, extra_map2 = get_uniprot_extra_mapping(extra_mapping_loc)
    extra_map1_dict = {}
    for key, val in extra_map1.items():
        if val in mappings[0].keys():
            extra_map1_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map1_dict[key] = mappings[1][val]
        elif val in mappings[2].keys():
            extra_map1_dict[key] = mappings[2][val]
    mappings.append(extra_map1_dict)
    extra_map2_dict = {}
    for key, val in extra_map2.items():
        if val in mappings[0].keys():
            extra_map2_dict[key] = mappings[0][val]
        elif val in mappings[1].keys():
            extra_map2_dict[key] = mappings[1][val]
        elif val in mappings[2].keys():
            extra_map2_dict[key] = mappings[2][val]
    mappings.append(extra_map2_dict)

    return mappings


def find_mapping(key, mappings=[], target_dict=[], mapping_not_found=None):
    for mapping in mappings:
        if key in mapping.keys() and mapping[key] in target_dict.keys():
            return target_dict[mapping[key]]
    # HARNESS: the per-gene print is dropped (thousands of lines); the
    # unmapped list is a parameter (was a module global)
    if mapping_not_found is not None:
        mapping_not_found.append(key)
    return [-1] * list(target_dict.values())[0].shape[0]


def find_genes_mapping(genes, mappings, emb_dict, mapping_not_found=None):
    name2vec = {}
    for gene in genes:
        name2vec[gene] = find_mapping(gene, mappings, emb_dict, mapping_not_found)

    return name2vec


def create_seq_features(df, emb_dict={}, mappings=[{}], emb_name='', ready_data_dict={}):
    emb_length = list(emb_dict.values())[0].shape[0]
    cols = []
    for i in range(emb_length):
        col_name = emb_name + str(i)
        cols.append(col_name)

    genes = np.union1d(df['gene1'].values, df['gene2'].values)
    name2vector = find_genes_mapping(genes, mappings, emb_dict)
    all_rows = np.zeros(shape=(len(df),emb_length))
    all_rows[:]=-1
    for ind, a_row in enumerate(df[['gene1','gene2']].values):
        if (a_row[0], a_row[1]) in ready_data_dict.keys():
            all_rows[ind] = ready_data_dict[(a_row[0], a_row[1])]
        else:
            g1_emb = name2vector[a_row[0]]
            g2_emb = name2vector[a_row[1]]
            if np.average(g1_emb) == -1.0 or np.average(g2_emb) == -1.0:
                pass
            else:
                all_rows[ind] = abs(np.subtract(g1_emb, g2_emb))
    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)
    return df


def create_ppi_features(df, emb_dict={}, mappings=[{}], emb_name='', ready_data=None):
    emb_length = list(emb_dict.values())[0].shape[0]
    cols = []
    for i in range(emb_length):
        col_name = emb_name + str(i)
        cols.append(col_name)

    genes = np.union1d(df['gene1'].values, df['gene2'].values)
    name2vector = find_genes_mapping(genes, mappings, emb_dict)

    all_rows = np.zeros(shape=(len(df),emb_length))
    all_rows[:]=-1
    for ind, a_row in enumerate(df[['gene1','gene2']].values):
        g1_emb = name2vector[a_row[0]]
        g2_emb = name2vector[a_row[1]]
        if np.average(g1_emb) == -1.0 or np.average(g2_emb) == -1.0:
            pass
        else:
            all_rows[ind] = abs(np.subtract(g1_emb, g2_emb))
    val_df = pd.DataFrame(all_rows, columns=cols, index=df.index.values)
    df = pd.concat([df, val_df], axis=1)
    return df


# ----------------------------------------------------------------------------
# HARNESS: vectorised form of the two loops above
# ----------------------------------------------------------------------------


def gene_vectors(genes, mappings, emb_dict):
    """(len(genes), dim) float64 matrix of find_mapping's vectors, plus the
    boolean `missing` row mask the authors' loop tests as
    np.average(emb) == -1.0 (a not-found gene is [-1] * dim)."""
    not_found = []
    name2vec = find_genes_mapping(list(genes), mappings, emb_dict, not_found)
    # The embeddings' own dtype (SeqVec npz and gensim vectors: float32): the
    # authors subtract in it and store the result in a float64 array.
    dtype = np.asarray(next(iter(emb_dict.values()))).dtype
    mat = np.asarray([np.asarray(name2vec[g], dtype=dtype) for g in genes],
                     dtype=dtype).reshape(len(genes), -1)
    missing = np.asarray([np.average(name2vec[g]) == -1.0 for g in genes],
                         dtype=bool)
    return mat, missing, sorted(set(not_found))


def pair_features(vecs, missing, a, b):
    """create_*_features' value for pairs (a[k], b[k]) of row indices into
    `vecs`: |v_a - v_b| computed in vecs' dtype and returned as float64, or
    -1 everywhere when either gene is missing."""
    f = np.abs(np.subtract(vecs[a], vecs[b])).astype(np.float64)
    f[missing[a] | missing[b]] = -1.0
    return f
