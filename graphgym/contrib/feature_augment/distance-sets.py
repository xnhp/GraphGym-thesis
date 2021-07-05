import networkx as nx

from graphgym.register import register_feature_augment
from pytictoc import TicToc
from graphgym.contrib.feature_augment.util import bfs_accumulate


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
    dist_sets = [bfs_accumulate(graph.G, node, 5, acc_nodes_at_dist, []) for node in graph.G.nodes]
    t.toc("distance set sizes")
    return dist_sets

register_feature_augment('node_distance_set_size', distance_set_sz_func)


