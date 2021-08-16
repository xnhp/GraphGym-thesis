import statistics
from typing import Tuple

import igraph
import networkx
import numpy as np
import torch
from networkx.algorithms import bipartite
from pytictoc import TicToc
from sklearn.utils import check_array

import deepsnap
import os
from graphgym.config import cfg


def check_cache(key, graph_name):
    cwd = os.getcwd()
    targetpath = os.path.join(cwd, 'feature-augment-cache', graph_name + '_' + key + '.pt')
    if os.path.isfile(targetpath):
        print("loaded from cache: " + graph_name + "/" + key)
        return torch.tensor(torch.load(targetpath)).to(torch.float32)
    else:
        return None


def put_cache(data, key, graph_name):
    cwd = os.getcwd()
    print("put into cache: " + graph_name + "/" + key)
    targetpath = os.path.join(cwd, 'feature-augment-cache', graph_name + '_' + key + '.pt')
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
            cached = check_cache(key, graph['name'])
            if cached is not None:
                return cached
            # the actual feature augment that is called
            r = augment_func(graph, **kwargs)
            put_cache(r, key, graph['name'])
            return r
        if cfg.dataset.feat_cache == 'update_always' or cfg.dataset.feat_cache == 'update':
            r = augment_func(graph, **kwargs)
            put_cache(r, key, graph['name'])
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


def get_bip_proj_repr(graph):
    if cfg.dataset.graph_interpretation == "simple":
        return get_bip_proj_cached(graph)
    elif cfg.dataset.graph_interpretation == "bipartite_projection":
        return graph
    else:
        raise NotImplementedError()


def projection_wrap(augment_func):
    """
    always call the given augment on the bipartite projection
    :param augment_func:
    :return:
    """
    # wrapper for augment functions that operate on bipartite graph
    def wrapped(graph, **kwargs):
        # ↝ simple_wrap
        # here, we always have fewer nodes; have to rely on downstream
        # components to properly handle this
        nodes_requested = graph.G.nodes
        if cfg.dataset.graph_interpretation == "simple":
            graph_to_use = get_bip_proj_cached(graph)
        elif cfg.dataset.graph_interpretation == "bipartite_projection":
            graph_to_use = graph
        else:
            raise NotImplementedError()
        return augment_func(graph_to_use, nodes_requested=nodes_requested, **kwargs)

    return wrapped


def simple_wrap(augment_func):
    """
    always call the given augment on the simple graph
    :param augment_func:
    :return:
    """
    def wrapped(graph, **kwargs):
        if cfg.dataset.graph_interpretation == "simple":
            graph_to_use = graph
            # node indices for which we want to compute features
            nodes_requested = graph.G.nodes
        elif cfg.dataset.graph_interpretation == "bipartite_projection":
            graph_to_use = graph['simple_graph']  # TODO
            # in case we are given bip-proj as primary graph, compute only
            # features in simple graph for nodes that will appear as nodes
            # in the bipartite projection.
            nodes_requested = get_non_rxn_nodes(graph.G)
        else:
            raise NotImplementedError()
        return augment_func(graph_to_use, nodes_requested=nodes_requested, **kwargs)

    return wrapped


def get_non_rxn_nodes(graph: networkx.Graph):
    """
    :param graph:
    :return: list of node indices
    ↝ SBMLModel#reactions
    """
    return [node for (node, nodeclass) in graph.nodes(data='class') if nodeclass != 'reaction']


def split_by_predicate(l, pred):
    yes = []
    no = []
    for x in l:
        if pred(x):
            yes.append(x)
        else:
            no.append(x)
    return yes, no


def split_rxn_nodes(graph: networkx.Graph):
    """
    :param graph:
    :return: (rxn nodes, non-rxn nodes)
    """
    # ↝ data / util.py: 156
    # ↝ data/util.py:128
    return split_by_predicate(graph.nodes(data='class'),
                              lambda x: x[1] == 'reaction')


def get_bip_proj_cached(graph: deepsnap.graph.Graph):
    # TODO this happens on dsG level but we also want to do the same on nxG level
    if graph['bipartite_projection'] is None:
        t = TicToc()
        t.tic()
        bipartite_projection, non_rxn_nodes = bipartite_projection_onto_non_rxn(graph.G)
        dsG = deepsnap.graph.Graph(bipartite_projection,
                                   # avoid updating internal tensor repr
                                   edge_label_index=[],
                                   node_label_index=[]
                                   )

        # selected nodes in original graph
        # i.e. ids of nodes that we want to consider in this split
        node_ids, a_idx, b_idx = tens_intersect(graph['node_label_index'], torch.tensor(non_rxn_nodes))
        dsG['node_label_index'] = b_idx
        dsG['is_bipartite_projection'] = True
        dsG['name'] = graph['name'] + " (bipartite projection)"
        graph['bipartite_projection'] = dsG
        t.toc("computed bipartite projection of " + graph['name'])
    return graph['bipartite_projection']


def get_simple_graph(nxG: networkx.Graph) -> networkx.Graph:
    """
    Given a graph constructed via pipeline that can either be a simple graph or a bipartite projection,
    return its simple graph representation.
    """
    if nxG.graph['is_bipartite_projection']:
        return nxG.graph['simple_graph']
    else:
        return nxG


def get_interpretations(nxG: networkx.Graph) -> Tuple[networkx.Graph, networkx.Graph]:
    is_proj = nxG.graph['is_bipartite_projection'] if 'is_bipartite_projection' in nxG.graph else False
    if is_proj:
        return nxG.graph['simple_graph'], nxG
    else:
        if 'bipartite_projection' in nxG.graph:
            return nxG, nxG.graph['bipartite_projection']
        else:
            bipartite_projection, _ = bipartite_projection_onto_non_rxn(nxG)
            nxG.graph['bipartite_projection'] = bipartite_projection
            return nxG, nxG.graph['bipartite_projection']

def bipartite_projection_onto_non_rxn(nxG: networkx.Graph) -> Tuple[networkx.Graph, list]:
    """
    :param nxG:
    :return: A networkx graph of the bipartite projection containing non-reaction nodes that are adjacent
        iff they share a common reaction; and a list of node indices into the original graph identifying
        non-reaction nodes.
    """
    non_rxn_nodes = get_non_rxn_nodes(nxG)
    # TODO
    rxn_n, non_rxn_n = split_rxn_nodes(nxG)
    rxn_ids = [n for n, _ in rxn_n]
    rxn_degs_sorted = list(sorted(nxG.degree(rxn_ids), key=lambda x: x[1]))
    rxn_degs = [d for _, d in rxn_degs_sorted]
    assert min(rxn_degs) > 0
    # allow isolated nodes
    # assert min([deg for (node, deg) in nxG.degree]) > 0
    assert bipartite.is_bipartite_node_set(nxG, non_rxn_nodes)
    bipartite_projection = bipartite.projected_graph(nxG,
                                                     non_rxn_nodes)  # connected if common neighbour in rxn_nodes
    return bipartite_projection, non_rxn_nodes


def tens_intersect(x: torch.Tensor, y: torch.Tensor):
    intersect, x_ind, y_ind = np.intersect1d(x.cpu().numpy(), y.cpu().numpy(), return_indices=True)
    return torch.tensor(intersect), torch.tensor(x_ind), torch.tensor(y_ind)


def collect_feature_augment(graph: deepsnap.graph.Graph):
    """
    Concat information from feature augments into node_feature tensor. Same as `Processing` module does in GNN models.
    Considers only nodes that are not reactions because
    - we never want to duplicate reactions
    - bipartite projection and its node features do not contain reactions
    :param graph:
    :return:
    """
    # check for invalid values like NaN or infinity
    for name in cfg.dataset.augment_feature:
        check_array(graph[name])

    # ↝ graphgym.models.feature_augment.Preprocess.forward
    node_index = get_non_rxn_nodes(graph.G)
    return torch.cat(
        [graph[name][node_index].float() for name in cfg.dataset.augment_feature],
        dim=1)


def get_igraph_cached(graph) -> igraph.Graph:
    is_proj = graph['is_bipartite_projection'] if 'is_bipartite_projection' in graph else False
    if is_proj:
        key = 'bipartite_projection_igraph'
    else:
        key = 'igraph'
    if graph[key] is None:
        t = TicToc()
        t.tic()
        iG = igraph.Graph.from_networkx(graph.G)
        assert iG.is_directed() is False
        graph[key] = iG
        t.toc("converted to igraph")
    return graph[key]
