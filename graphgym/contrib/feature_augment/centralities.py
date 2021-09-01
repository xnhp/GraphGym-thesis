import igraph
import networkx
import networkx as nx
import numpy
import numpy as np
from graphgym.contrib.feature_augment.util import bfs_accumulate, compute_stats, get_igraph_cached
from graphgym.register import register_feature_augment
from pytictoc import TicToc
import deepsnap


def degree_fun(graph: deepsnap.graph.Graph, **kwargs):
    return [deg for (node, deg) in graph.G.degree(kwargs['nodes_requested'])]
register_feature_augment('node_degree', degree_fun)


def in_degree_fun(graph: deepsnap.graph.Graph, **kwargs):
    # NOTE that this returns always the in/out degrees for simple graph representation, even if
    #   called via node_in_degree_projection
    mDiG: nx.MultiDiGraph = graph.G.graph['nx_multidigraph']  # simple graph interpretation
    # complication: wrapping a nxG in a dsG (using the deepsnap.graph.Graph constructor)
    #   leads to that the nodes in the nxG are relabelled sequentially
    nodes_requested = [graph.mapping_int_to_alias[id] for id in kwargs['nodes_requested']]
    assert all([alias in mDiG for alias in nodes_requested])
    return [deg for (_, deg) in mDiG.in_degree(nodes_requested)]
register_feature_augment('node_in_degree', in_degree_fun)


def out_degree_fun(graph: deepsnap.graph.Graph, **kwargs):
    mDiG: nx.MultiDiGraph = graph.G.graph['nx_multidigraph']
    nodes_requested = [graph.mapping_int_to_alias[id] for id in kwargs['nodes_requested']]
    assert all([alias in mDiG for alias in nodes_requested])
    return [deg for (_, deg) in mDiG.out_degree(nodes_requested)]
register_feature_augment('node_out_degree', out_degree_fun)


def betweenness_centr_igraph(graph: deepsnap.graph.Graph, **kwargs):
    return betweenness_impl(graph, **kwargs)
register_feature_augment('node_betweenness_centrality', betweenness_centr_igraph)


def closeness_centr_func(graph: deepsnap.graph.Graph, **kwargs):
    t = TicToc()
    t.tic()
    r = closeness_impl(graph, **kwargs)
    t.toc("Whole-graph closeness centralities")
    return r
register_feature_augment('node_closeness_centrality', closeness_centr_func)


def eigenvector_centr_func(graph: deepsnap.graph.Graph, **kwargs):
    t = TicToc()
    t.tic()
    r = np.array(list(eigenvector_impl(graph).values()))[kwargs['nodes_requested']]
    t.toc("Whole-graph eigenvector centrality")
    return r
register_feature_augment('node_eigenvector_centrality', eigenvector_centr_func)


def betweenness_impl(graph: deepsnap.graph.Graph, **kwargs):
    iG = get_igraph_cached(graph)
    return iG.betweenness(
        vertices=kwargs['nodes_requested'],  # defaults to all vertices
        directed=False
    )


def closeness_impl(graph: deepsnap.graph.Graph, **kwargs):
    iG = get_igraph_cached(graph)
    r = iG.closeness(vertices=kwargs['nodes_requested'], mode="all", normalized=True)
    npr = numpy.array(r)
    # impl. yields NaN for isolated nodes, replace with 0
    npr = numpy.nan_to_num(npr, nan=0.0, copy=False)
    # to_check = numpy.array(r).reshape(-1, 1)
    # can verify for NaN values like so:
    # to_check = numpy.array(r).reshape(-1, 1)
    # check_array(to_check)
    return list(npr)


def eigenvector_impl(graph: deepsnap.graph.Graph):
    return nx.algorithms.centrality.eigenvector_centrality_numpy(graph.G, max_iter=300)


def ego_graphs_incr(g:networkx.Graph, source, radii):
    """
    Compute undirected ego graphs centered at `source`, including `source`.
    :param g:
    :param source:
    :param radii: List of ints resembling distances at which we want an ego graph
    :return: List of ego graphs of given radii
    """

    # element at index i: nodes encountered in <= i steps
    def ego_acc(visited, next, encountered, at_distance, acc):
        # encountered: nodes encountered at this step
        # newly_discovered: nodes that havent been encountered before
        # seen: accumulator value, used to acc all nodes seen so far
        if at_distance in radii:  # this is dirty but fine assuming that at_distance increases with each call
            acc.append(visited)
        return acc

    # for each entry in radii, a set of nodes describing the induced subgraph of ego graph of that radius
    subgraph_nodesets = bfs_accumulate(g, source, max(radii), ego_acc, [])
    # construct actual induced subgraphs
    # could put this in the above accumulator if we also supplied g
    subgraphs = [g.subgraph(nodes) for nodes in subgraph_nodesets]
    return subgraphs


def ego_centrality_func(graph: deepsnap.graph.Graph, **kwargs):
    """
    Centrality measures for each node in ego graph of radius 3 and 5.
    Use custom BFS instead of networkx.generators.ego.ego_graph because that method takes ~1.2s on PDMap dataset
    ↝ [[speed up computation of ego_centralities]]
    """
    timer_whole = TicToc()
    timer_whole.tic()
    feats = []
    nxG = graph.G
    radii = [3, 5]
    for node in kwargs['nodes_requested']:
        # t = TicToc()
        # t.tic()
        egoGs = ego_graphs_incr(nxG, node, radii)
        assert len(egoGs) == len(radii)
        # t.toc("constructed ego graphs") # constructed ego graphs 0.000276 seconds.
        node_feats = []
        for egoG in egoGs:
            if len(egoG.nodes) < 3:
                # computing centralities on subgraphs with leq two nodes does not make sense
                # we assume values of 0 then
                # have to handle this case explicitly since some method calls fail otherwise
                # (e.g. eigenvector_centrality_numpy)
                eigenvector = betweenness = degree = closeness = 0
            else:
                # note that these are centralities in the ego graphs and different from the centralities computed
                #   in other feature augments
                # Four centrality scores calculated for a sub-graph consisting of all nodes within a given distance of
                #   3 and 5 hops to the node in question
                # calls arpack under the hood, just like igraph
                eigenvector = (nx.algorithms.centrality.eigenvector_centrality_numpy(egoG, max_iter=300)[node])
                # t.toc("eigenvector centrality", restart=True)
                iG = igraph.Graph.from_networkx(egoG)
                betweenness = iG.betweenness(vertices=[node], directed=False)[0]
                # t.toc("betweenness centrality", restart=True)
                degree = (nx.degree_centrality(egoG)[node])
                # t.toc("degree centrality", restart=True)
                closeness = iG.closeness(vertices=[node], mode="all")[0]
                # t.toc("closeness centrality", restart=True)
                # computed centralities 0.315175 seconds.
            node_feats += [eigenvector, betweenness, degree, closeness]
        feats.append(node_feats)

    assert len(feats) == len(graph.G.nodes)
    timer_whole.toc("Computed ego centralities (whole feature augment call)")
    return feats


register_feature_augment('node_ego_centralities', ego_centrality_func)


def neighbour_centrality_statistics_func(graph: deepsnap.graph.Graph, **kwargs):
    """
    Statistics (mean, max, min, stddev) of centralities of neighbour nodes, for each node.
    """
    nxG = graph.G
    t = TicToc()
    t.tic()
    feats = []
    # compute all centralities
    betweenness = betweenness_impl(graph, nodes_requested=graph.G.nodes)
    t.toc("betweeness centrality (igraph, whole graph)", restart=True)
    degree = (nx.degree_centrality(nxG))
    t.toc("degree centrality (nx, whole graph)", restart=True)
    closeness = closeness_impl(graph, nodes_requested=graph.G.nodes)
    t.toc("closeness centrality (igraph, whole graph)", restart=True)
    eigenvector = eigenvector_impl(graph)
    t.toc("eigenvector centrality (nx, whole graph)", restart=True)
    # then, for each neighbourhood, fetch and aggregate according centrs
    for node in kwargs['nodes_requested']:
        neighbs = list(nxG.neighbors(node))
        b = compute_stats([betweenness[neighb] for neighb in neighbs])
        d = compute_stats([degree[neighb] for neighb in neighbs])
        c = compute_stats([closeness[neighb] for neighb in neighbs])
        e = compute_stats([eigenvector[neighb] for neighb in neighbs])
        feats.append(b + c + d + e)
    assert len(feats) == len(kwargs['nodes_requested'])
    t.toc("Computed neighbour centrality statistics")
    return feats


register_feature_augment('node_neighbour_centrality_statistics', neighbour_centrality_statistics_func)
