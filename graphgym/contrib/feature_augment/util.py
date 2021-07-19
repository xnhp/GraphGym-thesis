import statistics

import networkx

import deepsnap


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
    assert len(l) > 0  # isolated node
    if len(l) < 2:
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
    def wrap(graph, **kwargs):
        return augment_func(get_bip_proj_cached(graph), **kwargs)

    return wrap


def get_non_rxn_nodes(graph : networkx.Graph):
    """
    :param graph:
    :return: list of node indices
    ↝ SBMLModel#reactions
    """
    return [node for (node, nodeclass) in graph.nodes(data='class') if nodeclass != 'reaction']


def get_bip_proj_cached(graph):
    if graph['bipartite_projection'] is None:
        from networkx.algorithms import bipartite
        bipartite_projection = bipartite.projected_graph(graph.G, get_non_rxn_nodes(graph.G))  # connected if common neighbour in rxn_nodes
        dsG = deepsnap.graph.Graph(bipartite_projection,
                                   # avoid updating internal tensor repr
                                   edge_label_index=[],
                                   node_label_index=[]
                                   )
        graph['bipartite_projection'] = dsG
    return graph['bipartite_projection']