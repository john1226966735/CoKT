from gensim.models import KeyedVectors
import numpy as np
import os


def emb_transfer():  # transfer KeyedVectors format embeddings into a embedding matrix
    wv_from_text = KeyedVectors.load_word2vec_format(wv_path, binary=False)
    max_nodes = max([eval(i) for i in wv_from_text.vocab.keys()])+1
    num_nodes = len(wv_from_text.vocab)
    # vector_list = np.random.normal(size=(max_nodes, dim)).astype(np.float32)
    vector_list = np.zeros(shape=(max_nodes, dim), dtype=np.float32)  # for some question not in train dataset
    for vocab in wv_from_text.vocab.keys():
        vector_list[eval(vocab)] = list(wv_from_text.get_vector(vocab))
    np.save(npy_path, vector_list)
    print("inferred question number：%d, actual question number：%d" % (max_nodes, num_nodes))


# ------------------- main -------------------
for dataset in ["ASSIST09"]:
    for dim in [128]:
        for kg in ['kg_cw']:
            for walk_length in [80]:
                for num_walks in [10]:
                    for window_size in [3, 5, 7]:
                        for epoch in [20]:
                            for p in [1]:
                                for q in [1]:
                                    wv_path = "./emb/%s_%s_%d_%d_%d_%d_%d_%.2f_%.2f.embQ" % (
                                        dataset, kg, dim, walk_length, num_walks, window_size, epoch, p, q)
                                    npy_path = wv_path
                                    emb_transfer()
