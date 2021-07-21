import statistics

import networkx
import numpy as np
import torch

import deepsnap
import os
from graphgym.config import cfg


def check_cache(key):
    cwd = os.getcwd()
    targetpath = os.path.join(cwd, 'feature-augment-cache', cfg.dataset.name + '_' + key + '.pt')
    if os.path.isfile(targetpath):
        print("loading from cache: ", key)
        return torch.tensor(torch.load(targetpath)).to(torch.float32)
    else:
        return None


def put_cache(data, key):
    cwd = os.getcwd()
    print("putting into cache: ", key)
    targetpath = os.path.join(cwd, 'feature-augment-cache', cfg.dataset.name + '_' + key + '.pt')
    torch.save(data, targetpath)

def cache_wrap(key, augment_func):
    """
    Note that this simply caches the result of the feature augment call.
    If the train/test split is different this will return incorrect results
    :param key:
    :param augment_func:
    :return:
    """

    def wrapped(graph, **kwargs):
        if cfg.dataset.feat_cache == 'use_and_update' or cfg.dataset.feat_cache == 'enabled':
            cached = check_cache(key)
            if cached is not None:
                return cached
            # the actual feature augment that is called
            r = augment_func(graph, **kwargs)
            put_cache(r, key)
            return r
        if cfg.dataset.feat_cache == 'update_always':
            r = augment_func(graph, **kwargs)
            put_cache(r, key)
            return r
        if cfg.dataset.feat_cache == 'disabled':
            return augment_func(graph, **kwargs)
    return wrapped


def bfs_accumulate(g, source, max_distance, accumulator, acc):
    """
    Perform BFS and call accumulator function at each step.
    Accumulator function is guaranteed to be called max_distance times, potentially with empty node sets.
    Based on networkx.algorithms.traversal.breadth_first_search.descendants_at_distance.
    ↝ [[speed up computation of ego_centralities]]
    :param g:
    :param source:
    :param max_distance:
    :return:
    """
    current_distance = 0
    queue = {source}
    visited = {source}
    while queue:
        if current_distance == max_distance:
            return acc
        current_distance += 1
        next_vertices = set()  # newly discovered
        encountered_in_step = set()
        for vertex in queue:
            for child in g[vertex]:
                encountered_in_step.add(child)
                if child not in visited:
                    visited.add(child)
                    next_vertices.add(child)
        queue = next_vertices
        accumulator(visited, next_vertices, encountered_in_step, current_distance, acc)
    # BFS ended, pad result
    if current_distance < max_distance:
        encountered_in_step = set()
        next_vertices = set()
        # call consumer once for each remaining step until max_distance
        for i in range(current_distance + 1, max_distance + 1):
            accumulator(visited, next_vertices, encountered_in_step, i, acc)

    return acc


def compute_stats(l):
    if len(l) == 0:
        mean = 0
        minv = 0
        maxv = 0
        stddev = 0
    elif len(l) < 2:
        mean = l[0]
        minv = l[0]
        maxv = l[0]
        stddev = 0
    else:
        mean = statistics.mean(l)
        minv = min(l)
        maxv = max(l)
        stddev = statistics.stdev(l)
    return list([mean, minv, maxv, stddev])


def bipartite_projection_wrap(augment_func):
    # the actual feature augment that is called
    def wrapped(graph, **kwargs):
        return augment_func(get_bip_proj_cached(graph), **kwargs)

    return wrapped


def get_non_rxn_nodes(graph: networkx.Graph):
    """
    :param graph:
    :return: list of node indices
    ↝ SBMLModel#reactions
    """
    return [node for (node, nodeclass) in graph.nodes(data='class') if nodeclass != 'reaction']


def get_bip_proj_cached(graph):
    if graph['bipartite_projection'] is None:
        from networkx.algorithms import bipartite
        non_rxn_nodes = get_non_rxn_nodes(graph.G)
        assert min([deg for (node, deg) in graph.G.degree]) > 0
        assert bipartite.is_bipartite_node_set(graph.G, non_rxn_nodes)
        bipartite_projection = bipartite.projected_graph(graph.G,
                                                         non_rxn_nodes)  # connected if common neighbour in rxn_nodes
        dsG = deepsnap.graph.Graph(bipartite_projection,
                                   # avoid updating internal tensor repr
                                   edge_label_index=[],
                                   node_label_index=[]
                                   )

        # selected nodes in original graph
        # i.e. ids of nodes that we want to consider in this split
        node_ids, a_idx, b_idx = tens_intersect(graph['node_label_index'], torch.tensor(non_rxn_nodes))
        dsG['node_label_index'] = b_idx
        graph['bipartite_projection'] = dsG
    return graph['bipartite_projection']


def tens_intersect(x: torch.Tensor, y: torch.Tensor):
    intersect, x_ind, y_ind = np.intersect1d(x.cpu().numpy(), y.cpu().numpy(), return_indices=True)
    return torch.tensor(intersect), torch.tensor(x_ind), torch.tensor(y_ind)


def collect_feature_augment(graph:deepsnap.graph.Graph):
    """
    Concat information from feature augments into node_feature tensor. Same as `Processing` module does in GNN models.
    Additionally takes a node_index mask of nodes to consider.
    :param graph:
    :return:
    """
    # consider only nodes that are not reactions because
    # - we never want to duplicate reactions
    # - bipartite projection and its node features do not contain reactions
    node_index = get_non_rxn_nodes(graph.G)
    return torch.cat(
        [graph[name][node_index].float() for name in cfg.dataset.augment_feature],
        dim=1)
