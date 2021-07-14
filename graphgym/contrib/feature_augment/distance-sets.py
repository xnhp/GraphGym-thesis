import networkx as nx

from graphgym.register import register_feature_augment
from pytictoc import TicToc
from graphgym.contrib.feature_augment.util import bfs_accumulate, compute_stats


def distance_set_sz_func(graph, **kwargs):
    """
    The number of nodes *exactly* at a given distance from the node in question. Here distances 1−5 are considered.
    ↝ [[speed up computation of ego_centralities]]
    :param graph:
    :param kwargs:
    :return:
    """

    def acc_nodes_at_dist(visited, next, encountered, at_distance, acc):
        acc.append(len(encountered))

    def pad(l, sz, val):
        l += [val] * (sz - len(l))
        return l

    t = TicToc()
    t.tic()
    abs_sz = [bfs_accumulate(graph.G, node, 5, acc_nodes_at_dist, []) for node in graph.G.nodes]
    # each element is a list of distance set sizes for resp. distance 1-5
    norms = [4, 8, 12, 16, 20]   # cf nielsen

    feats = [[sz / norm for sz, norm in zip(sizes, norms)] for sizes in abs_sz]
    # stats across different sizes
    feats = [feat + compute_stats(feat) for feat in feats]

    t.toc("distance set sizes")
    return feats

register_feature_augment('node_distance_set_size', distance_set_sz_func)


