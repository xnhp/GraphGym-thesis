import statistics

import networkx as nx
import networkx.generators.ego

from graphgym.register import register_feature_augment
from graphgym.contrib.feature_augment.util import bfs_accumulate

from pytictoc import TicToc

# betweenness already part of core


# degree already part of core


def closeness_centr_func(graph, **kwargs):
    t = TicToc()
    t.tic()
    r = list(nx.algorithms.centrality.closeness_centrality(graph.G).values())
    t.toc("Whole-graph closeness centralities")
    return r
register_feature_augment('node_closeness_centrality', closeness_centr_func)


def eigenvector_centr_func(graph, **kwargs):
    t = TicToc()
    t.tic()
    r =  list(nx.algorithms.centrality.eigenvector_centrality(graph.G).values())
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
        if at_distance in radii: # this is dirty but fine assuming that at_distance increases with each call
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
    radii = [3,5]
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
                betweenness = (nx.betweenness_centrality(egoG)[node])
                # t.toc("betweeness centrality", restart=True)
                degree = (nx.degree_centrality(egoG)[node])
                # t.toc("degree centrality", restart=True)
                closeness = (nx.algorithms.centrality.closeness_centrality(egoG)[node])
                # t.toc("closeness centrality", restart=True)
                # computed centralities 0.315175 seconds.
            node_feats += [eigenvector, betweenness, degree, closeness]
        feats.append(node_feats)

    assert len(feats) == len(graph.G.nodes)
    timer_whole.toc("Computed ego centralities (whole feature augment call)")
    return feats

register_feature_augment('node_ego_centralities', ego_centrality_func)


def neighbour_centrality_statistics_func(nxG):
    """
    Statistics (mean, max, min, stddev) or centralities of neighbour nodes, for each node.
    """
    t = TicToc()
    t.tic()
    feats = []
    for node in nxG.nodes:
        egoG = networkx.generators.ego.ego_graph(nxG,
                                                 node,
                                                 radius=1,
                                                 center=False,
                                                 undirected=True)

        betweenness = list(nx.betweenness_centrality(egoG).values())
        degree = list(nx.degree_centrality(egoG).values())
        closeness = list(nx.algorithms.centrality.closeness_centrality(egoG).values())
        eigenvector = list(nx.algorithms.centrality.eigenvector_centrality(egoG).values())

        def compute_stats(l):
            return [statistics.mean(l), min(l), max(l), statistics.stdev(l)]

        feats.append([
            compute_stats(closeness),
            compute_stats(eigenvector),
            compute_stats(degree),
            compute_stats(betweenness)
        ])
    assert len(feats) == len(nxG.nodes)
    t.toc("Computed neighbour centrality statistics")
    return feats

register_feature_augment('node_neighbour_centrality_statistics', neighbour_centrality_statistics_func)
