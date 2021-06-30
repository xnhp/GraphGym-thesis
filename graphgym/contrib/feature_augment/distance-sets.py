import networkx as nx

from graphgym.register import register_feature_augment
from pytictoc import TicToc


def distance_set_sz_func(graph, **kwargs):
    """
    The number of nodes *exactly* at a given distance from the node in question. Here distances 1−5 are considered.
    :param graph:
    :param kwargs:
    :return:
    """

    def distance_set_sz_incr(g, source, max_distance):
        """
        Based on networkx.algorithms.traversal.breadth_first_search.descendants_at_distance with the only difference
        being that this incrementally accumulates neighbourhood sizes
        :param g:
        :param source:
        :param max_distance:
        :return:
        """
        current_distance = 0
        queue = {source}
        visited = {source}
        result = []  # element at index i denotes number of nodes at distance exactly i
        while queue:
            if current_distance == max_distance:
                return result
            current_distance += 1
            next_vertices = set()  # newly discovered
            for vertex in queue:
                for child in g[vertex]:
                    if child not in visited:
                        visited.add(child)
                        next_vertices.add(child)
            queue = next_vertices
            result.append(len(next_vertices))
        result += [0] * (max_distance - len(result))  # pad with zeroes in case BFS terminated early
        return result

    t = TicToc()
    t.tic()
    r =  [distance_set_sz_incr(graph.G, node, 5) for node in graph.G.nodes]
    t.toc("distance set sizes")
    return r

register_feature_augment('node_distance_set_size', distance_set_sz_func)


