import igraph
import networkx as nx

from graphgym.register import register_feature_augment
from graphgym.contrib.feature_augment.util import bfs_accumulate, compute_stats, get_igraph_cached

from pytictoc import TicToc


def betweenness_impl(graph):
    iG = get_igraph_cached(graph)
    return iG.betweenness(
        vertices=None,  # defaults to all vertices
        directed=False
    )


def betweenness_centr_igraph(graph, **kwargs):
    return betweenness_impl(graph)


register_feature_augment('node_betweenness_centrality', betweenness_centr_igraph)


def closeness_impl(graph):
    iG = get_igraph_cached(graph)
    return iG.closeness(vertices=None, mode="all", normalized=True)


def closeness_centr_func(graph, **kwargs):
    t = TicToc()
    t.tic()
    r = closeness_impl(graph)
    t.toc("Whole-graph closeness centralities")
    return r


register_feature_augment('node_closeness_centrality', closeness_centr_func)


def eigenvector_centr_func(graph, **kwargs):
    t = TicToc()
    t.tic()
    r = list(nx.algorithms.centrality.eigenvector_centrality_numpy(graph.G, max_iter=300).values())
    t.toc("Whole-graph eigenvector centrality")
    return r


register_feature_augment('node_eigenvector_centrality', eigenvector_centr_func)


def ego_graphs_incr(g, source, radii):
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


def ego_centrality_func(graph, **kwargs):
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
    for node in nxG.nodes:
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
                # have to handle this case explicitly since some method calls fail otherwise (e.g. eigenvector_centrality_numpy)
                # print("number of nodes in ego graph < 3")
                eigenvector = betweenness = degree = closeness = 0
            else:
                # t.tic()
                # Four centrality scores calculated for a sub-graph consisting of all nodes within a given distance of 3 and 5 h
                # ops to the node in question
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


def eigenvector_impl(graph):
    return nx.algorithms.centrality.eigenvector_centrality_numpy(graph.G, max_iter=300)


def neighbour_centrality_statistics_func(graph, **kwargs):
    """
    Statistics (mean, max, min, stddev) of centralities of neighbour nodes, for each node.
    """
    nxG = graph.G
    t = TicToc()
    t.tic()
    feats = []
    # compute all centralities
    betweenness = betweenness_impl(graph)
    t.toc("betweeness centrality (igraph, whole graph)", restart=True)
    degree = (nx.degree_centrality(nxG))
    t.toc("degree centrality (nx, whole graph)", restart=True)
    closeness = closeness_impl(graph)
    t.toc("closeness centrality (igraph, whole graph)", restart=True)
    eigenvector = eigenvector_impl(graph)
    t.toc("eigenvector centrality (nx, whole graph)", restart=True)
    # then, for each neighbourhood, fetch and aggregate according centrs
    for node in nxG.nodes:
        neighbs = list(nxG.neighbors(node))
        b = compute_stats([betweenness[neighb] for neighb in neighbs])
        d = compute_stats([degree[neighb] for neighb in neighbs])
        c = compute_stats([closeness[neighb] for neighb in neighbs])
        e = compute_stats([eigenvector[neighb] for neighb in neighbs])
        feats.append(b + c + d + e)
    assert len(feats) == len(nxG.nodes)
    t.toc("Computed neighbour centrality statistics")
    return feats


register_feature_augment('node_neighbour_centrality_statistics', neighbour_centrality_statistics_func)
