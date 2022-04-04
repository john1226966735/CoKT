"""
Refer to https://github.com/aditya-grover/node2vec
"""
import argparse
import numpy as np
import networkx as nx
from Code.node2vec import node2vec
from gensim.models import Word2Vec

import os


def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk_length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num_walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window_size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(input_graph):
    """
    Reads the input network in networkx.
    """
    if args.weighted:
        G = nx.read_edgelist(input_graph, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph(),
                             delimiter=',')
    else:
        G = nx.read_edgelist(input_graph, nodetype=int, create_using=nx.DiGraph(), delimiter=',')
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)


def main():
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    # get sentences
    total_walks = []
    for g in ['kg_correct', 'kg_wrong']:
        input_graph = os.path.join("../../Data", DATASET, g + ".edgelist")

        nx_G = read_graph(input_graph)
        G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
        print("read graph done")
        G.preprocess_transition_probs()
        print("process done")
        walks = G.simulate_walks(args.num_walks, args.walk_length)

        total_walks += walks

    # delete non-question ids, question id: from 0 to max_ques_id, non-question id: start from max_ques_id
    print("deleting user_ids")
    final_walks = []
    MAX_QUES_ID = {'ASSIST09': 17736, 'ASSIST12': 53064, 'EdNet': 12149}
    for walks in total_walks:
        walks = np.array(walks)
        final_walks.append(walks[walks <= MAX_QUES_ID[DATASET]])

    # get question embeddings
    print("start to learn embedding")
    learn_embeddings(final_walks)


if __name__ == "__main__":
    # for learning question embeddings from the student-question graph
    import time
    DATASET = 'ASSIST09'  # ASSIST09, ASSIST12, EdNet
    args = parse_args()
    for kg in ['kg_cw']:
        for dim in [128]:
            args.dimensions = dim
            for walk_length in [80]:
                args.walk_length = walk_length
                for num_walks in [10]:
                    args.num_walks = num_walks
                    for window_size in [3, 5, 7]:
                        args.window_size = window_size
                        for epoch in [20]:
                            args.iter = epoch
                            for p in [0.5, 1, 2]:
                                args.p = p
                                for q in [0.5, 1, 2]:
                                    args.q = q

                                    args.output = os.path.join("emb", "%s_%s_%d_%d_%d_%d_%d_%.2f_%.2f.embQ" % (
                                        DATASET, kg, dim, walk_length, num_walks, window_size, epoch, p, q))
                                    t1 = time.time()
                                    main()
                                    print("spend %d seconds" % (time.time() - t1))
