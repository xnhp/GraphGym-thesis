import statistics
from typing import Tuple

import igraph
import networkx
import numpy as np
import torch
from networkx.algorithms import bipartite
from networkx.algorithms.connectivity.edge_kcomponents import _low_degree_nodes
from pytictoc import TicToc
from sklearn.utils import check_array

import deepsnap
import os
from graphgym.config import cfg

from data.models import SpeciesClass, SBMLModel


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


def projection_wrap(augment_func):
    """
    always call the given augment on the bipartite projection
    :param augment_func:
    :return:
    """

    # wrapper for augment functions that operate on bipartite graph
    def wrapped(graph: deepsnap.graph.Graph, **kwargs):
        # ↝ simple_wrap
        _, dsG_bip = ds_get_interpretations(graph)
        return augment_func(dsG_bip, nodes_requested=dsG_bip.G.nodes, **kwargs)

    return wrapped


def simple_wrap(augment_func):
    """
    always call the given augment on the simple graph
    :param augment_func:
    :return:
    """

    def wrapped(graph: deepsnap.graph.Graph, **kwargs):
        dsG_simple, dsG_bip = ds_get_interpretations(graph)
        if cfg.dataset.graph_interpretation == "simple":
            # node indices for which we want to compute features
            nodes_requested = graph.G.nodes
        elif cfg.dataset.graph_interpretation == "bipartite_projection":
            # in case we are given bip-proj as primary graph, compute only
            #   features in simple graph for nodes that will appear as nodes
            #   in the bipartite projection.
            # alternatively we would have to sub-index later but some computations are expensive so it seems
            #   desirable to avoid them
            # note that this means the cache cannot be transferred between different primary representations/interpretations
            #   (as specified by dataset.graph_interpretation) ↝ [[^bbdce0]] and esp. [[^61a2c9]]
            nodes_requested = get_non_rxn_nodes(graph.G)
        else:
            raise NotImplementedError()
        return augment_func(dsG_simple, nodes_requested=nodes_requested, **kwargs)

    return wrapped

def get_nodes_by_class(graph: networkx.Graph, requested_class: SpeciesClass):
    return [n for n, c in graph.nodes(data='class') if c == requested_class.value]

def get_non_rxn_nodes(graph: networkx.Graph):
    """
    :param graph:
    :return: list of node IDs that correspond to entities that are not reactions. Note that this returns the node IDs
        of the given graph. GG performs a relabelling on the networkx graph upon wrapping it in a deepsnap graph.
    ↝ SBMLModel#reactions
    """
    return [node for (node, nodeclass) in graph.nodes(data='class') if nodeclass != SpeciesClass.reaction.value]


def get_prediction_nodes(nxG: networkx.Graph) -> Tuple[np.array, np.array]:
    """
    Get indices of nodes to evaluate the prediction on. Motivation: Things like feature computation and message-passing
    should be based on the entire graph but to improve model performance we can exclude some things from prediction
    that we never want to duplicate anyways. Cf Nielsen et. al sec. 3.4
    :param nxG:
    :return:
    """
    assert not nxG.is_directed()
    # assume that node ids correspond to indices in deepsnap tensors ↝ deepsnap/graph.py:819
    complex_to_exclude = get_nodes_by_class(nxG, SpeciesClass.complex) if cfg.dataset.exclude_complex_species else []
    low_deg_to_exclude = list(_low_degree_nodes(nxG, SBMLModel.min_node_degree)) if cfg.dataset.exclude_low_degree else []
    # node_label_index are ids of nodes appearing in this internal split

    excluded = np.union1d(complex_to_exclude, low_deg_to_exclude)
    included = np.setdiff1d(nxG.nodes, excluded)
    return included, excluded

    # cleanup: also include non-rxn nodes? still need to ensure that outside

    # not all may be present in node_label_index because of internal split?
    # assert np.in1d(to_exclude, dsG.G['node_label_index'])

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
                              lambda x: x[1] == SpeciesClass.reaction.value)


# cleanup
nx_simple_ref_key = 'nx_simple_ref'
nx_bip_ref_key = 'nx_bipartite_projection_ref'
ds_simple_ref_key = 'ds_simple_ref'
ds_bip_ref_key = 'ds_bipartite_projection_ref'


def _ds_get_bip(dsG_simple: deepsnap.graph.Graph) -> deepsnap.graph.Graph:
    if ds_bip_ref_key not in dsG_simple or dsG_simple[ds_bip_ref_key] is None:
        simple_nxG, bip_nxG = nx_get_interpretations(dsG_simple.G)
        dsG_bip = deepsnap.graph.Graph(bip_nxG,
                                       # avoid updating internal tensor repr
                                       edge_label_index=[],
                                       node_label_index=[]
                                       )
        # cleanup: these should already be exactly all the node ids of bip_nxG
        non_rxn_nodes = get_non_rxn_nodes(dsG_simple.G)
        # note that deepsnap.graph.Graph constructor actually renames the nodes of the given nxG
        # ↑ TODO at some point we expected integers there...?
        #    at some points nodes were relabeled? ↝ deepsnap/graph.py:808
        #    the above constructor modifies bip_nxG (relabels its nodes?)
        #    do I have to update the nxGs?
        # selected nodes in original graph
        # i.e. ids of nodes that we want to consider in this split
        node_ids, a_idx, b_idx = tens_intersect(dsG_simple['node_label_index'], torch.tensor(non_rxn_nodes))
        dsG_bip['node_label_index'] = b_idx  # subset of node_label_index that appears in bip proj
        dsG_bip['name'] = dsG_simple['name'] + " (bipartite projection)"
        dsG_simple[ds_bip_ref_key] = dsG_bip
        dsG_bip[ds_simple_ref_key] = dsG_simple
    return dsG_simple[ds_bip_ref_key]


def _nx_get_bip(nxG_simple: networkx.Graph) -> networkx.Graph:
    if nx_bip_ref_key not in nxG_simple.graph or nxG_simple.graph[nx_bip_ref_key] is None:
        bipartite_projection = bipartite_projection_onto_non_rxn(nxG_simple)
        bipartite_projection.graph[nx_simple_ref_key] = nxG_simple
        nxG_simple.graph[nx_bip_ref_key] = bipartite_projection
    return nxG_simple.graph[nx_bip_ref_key]


def ds_get_interpretations(dsG: deepsnap.graph.Graph) -> Tuple[deepsnap.graph.Graph, deepsnap.graph.Graph]:
    """
    Return both simple and bipartite representations of the given graph in an ordered tuple, no matter what
    representation the given graph corresponds to.
    """
    # assume projection if has reference
    if ds_simple_ref_key in dsG and dsG[ds_simple_ref_key] is not None:
        return dsG[ds_simple_ref_key], dsG
    else:
        return dsG, _ds_get_bip(dsG)


def nx_get_interpretations(nxG: networkx.Graph) -> Tuple[networkx.Graph, networkx.Graph]:
    """
    Return both simple and bipartite representations of the given graph in an ordered tuple, no matter what
    representation the given graph corresponds to.
    """
    # assume graph is bipartite projection if it has reference to simple interpretation set
    if nx_simple_ref_key in nxG.graph and nxG.graph[nx_simple_ref_key] is not None:
        return nxG.graph[nx_simple_ref_key], nxG
    else:
        return nxG, _nx_get_bip(nxG)


def bipartite_projection_onto_non_rxn(nxG: networkx.Graph) -> networkx.Graph:
    """
    :param nxG:
    :return: A networkx graph of the bipartite projection containing non-reaction nodes that are adjacent
        iff they share a common reaction
    """
    non_rxn_nodes = get_non_rxn_nodes(nxG)
    # cleanup: is this just debug info and can be removed?
    rxn_n, non_rxn_n = split_rxn_nodes(nxG)
    rxn_ids = [n for n, _ in rxn_n]
    rxn_degs_sorted = list(sorted(nxG.degree(rxn_ids), key=lambda x: x[1]))
    rxn_degs = [d for _, d in rxn_degs_sorted]
    assert min(rxn_degs) > 0
    # ↑ ↝ [[^781aa1]]
    assert bipartite.is_bipartite_node_set(nxG, non_rxn_nodes)
    bipartite_projection = bipartite.projected_graph(nxG,
                                                     non_rxn_nodes)  # connected if common neighbour in rxn_nodes
    return bipartite_projection


def tens_intersect(x: torch.Tensor, y: torch.Tensor):
    intersect, x_ind, y_ind = np.intersect1d(x.cpu().numpy(), y.cpu().numpy(), return_indices=True)
    return torch.tensor(intersect), torch.tensor(x_ind), torch.tensor(y_ind)


def collect_feature_augment(graph: deepsnap.graph.Graph):
    # TODO can we call this in Preprocess.forward?
    """
    Concat information from feature augments into node_feature tensor. Same as `Preprocess` module does in GNN models.
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
    # need to index here because tensors need to be of same size for torch.cat
    node_index = get_non_rxn_nodes(graph.G)
    return torch.cat(
        [graph[name][node_index].float() for name in cfg.dataset.augment_feature],
        dim=1)


def get_igraph_cached(graph: deepsnap.graph.Graph) -> igraph.Graph:
    if ds_simple_ref_key in graph and graph[ds_simple_ref_key] is not None:
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
