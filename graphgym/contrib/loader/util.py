import networkx as nx
import numpy as np
import torch

import deepsnap.graph
from data.graphs import construct_alias_graph, construct_collapsed_graph

from data.models import SBMLModel


def nxG_to_dsG(nxG):
    """
    Construct a DeepSNAP graph suited for use with GG
    :param nxG:
    :return:
    """
    # do not really have extracted real node features yet but have to set some
    for nodeIx in nxG.nodes:
        feat = np.zeros(1)
        feat = torch.from_numpy(feat).to(torch.float)
        nxG.nodes[nodeIx]['node_feature'] = feat
        # node label should be set
    return deepsnap.graph.Graph(nxG)


def load_nxG(path, model_class: SBMLModel, name: str, **kwargs) -> nx.Graph:
    model = model_class(path)
    nxG: nx.Graph
    if 'collapsed' in kwargs and kwargs['collapsed']:
        nxG = construct_collapsed_graph(model, name=name)
    else:
        if 'collapsed' not in kwargs:
            raise RuntimeWarning("not specified whether collapsed or not, assuming not")
        nxG = construct_alias_graph(model, name=name)
    return nxG


def upsert_node(nxG: nx.Graph, node, **nodeattribs):
    # in networkx, a node can be any hashable object
    if node not in nxG.nodes():
        nxG.add_node(node, **nodeattribs)
    else:
        pass


def contains_node(nxG: nx.Graph, node):
    return node in nxG.nodes()
