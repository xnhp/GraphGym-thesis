import statistics

import networkx as nx
import networkx.generators.ego

from graphgym.register import register_feature_augment

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


def ego_centrality_func(graph, **kwargs):
    """
    Centrality measures for each node in ego graph of radius 3 and 5.
    """
    t = TicToc()
    t.tic()
    def ego_centralities(nxG, radius):
        feats = []
        for node in nxG.nodes:
            t1 = TicToc()
            t1.tic()
            # computing a single one these ego graphs takes ~1.2 seconds on PDMap → >= 1h for all nodes
            # TODO either find a smarter way to do this (subset methods of centrality?)
            #       but apparently not supported for eigenvector
            #   or do the wait but avoid doing this on the fly
            #   or consider using igraph
            egoG = networkx.generators.ego.ego_graph(nxG,
                                                     node,
                                                     radius=radius,
                                                     center=True,
                                                     undirected=True)
            t1.toc("build ego graph")
            if len(egoG.nodes) < 2:
                # computing centralities on subgraphs with a single node does not make sense
                # we assume values of 0 then
                # have to handle this case explicitly since some method calls fail otherwise (e.g. eigenvector_centrality_numpy)
                print("number of nodes in ego graph < 2")
                eigenvector = betweenness = degree = closeness = 0
            else:
                # Four centrality scores calculated for a sub-graph consisting of all nodes within a given distance of 3 and 5 h
                # ops to the node in question
                eigenvector = (nx.algorithms.centrality.eigenvector_centrality_numpy(egoG, max_iter=300)[node])
                t1.toc("Eigenvector centrality for node")
                betweenness = (nx.betweenness_centrality(egoG)[node])
                degree = (nx.degree_centrality(egoG)[node])
                closeness = (nx.algorithms.centrality.closeness_centrality(egoG)[node])
                # eigenvector = (nx.algorithms.centrality.eigenvector_centrality(egoG, max_iter=300)[node])
                t1.toc("all other statistics for node")

            feats.append([betweenness, degree, closeness, eigenvector])

        assert len(feats) == len(nxG.nodes)
        return feats

    # concat the list so we have a 2-dimensional list in the end
    feats = [e1+e2 for (e1, e2) in zip(
        ego_centralities(graph.G, 3),
        ego_centralities(graph.G, 5)
    )]
    assert len(feats) == len(graph.G.nodes)
    t.toc("Computed ego centralities")
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
