import networkx as nx
import numpy as np
from graphgym.register import register_feature_augment

from graphgym.config import cfg

def node_class_onehot_func(graph, **kwargs):
    return onehot_attr(graph, "class")
register_feature_augment('node_class_onehot', node_class_onehot_func)

def onehot_attr(graph, key):


    attributes = nx.get_node_attributes(graph.G, key)
    possible_values = cfg.dataset.possible_classes
    def onehot_enc(value, possible_values):
        f = np.zeros(len(possible_values))
        f[possible_values.index(value)] = 1  # probably faster to use a dict to get the int
        return f
    r = [
        onehot_enc(value, possible_values) for value in attributes.values()
    ]
    assert len(r) == len(graph.G.nodes)
    return r
