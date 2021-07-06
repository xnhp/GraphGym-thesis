import deepsnap.dataset
import networkx as nx
from data.util import get_dataset, CellDesignerModel, SBMLModel

from graphgym.register import register_loader
from graphgym.config import cfg
import os
import numpy as np
import torch


def SBML_single(format, name, dataset_dir) -> list[deepsnap.graph.Graph]:
    if cfg.dataset.format != "SBML":
        return None

    path, model_class = get_dataset(name)
    model = model_class(path)

    nxG : nx.Graph
    nxG = graph_from_model(model)

    # do not really have extracted real node features yet but have to set some
    for nodeIx in nxG.nodes:
        feat = np.zeros(1)
        feat = torch.from_numpy(feat).to(torch.float)
        nxG.nodes[nodeIx]['node_feature'] = feat
        # node label should be set

    dsG = deepsnap.graph.Graph(nxG)
    return [dsG]

register_loader('SBML_single', SBML_single)

def graph_from_model(model: SBMLModel) -> nx.Graph:
    """
    :param path:
    :return:
    """
    G = nx.Graph()
    G.graph['name'] = model.path

    # fetch full info about species and already add as nodes
    for species in model.species:
        G.add_node(species['id'], **species)

    # add nodes for reactions and edges from/to reactions
    for rxn in model.reactions:
        rxn : Dict #  of info about this reaction
        rxn_data = rxn.copy()
        del rxn_data['reactants']
        del rxn_data['products']
        del rxn_data['modifiers']
        reaction_node = G.add_node(rxn['id'], **rxn_data)  # we use id as key and the entire dict (containg id) as additional attributes
        # do not need to explicitly denote here that this is a reaction node because we already set its `type` attribute
        for neighbour_id in rxn['reactants'] + rxn['products'] + rxn['modifiers']:
            if neighbour_id in G.nodes:  # `add_edge` adds nodes if they don't exist yet
                                         # there might be excluded species that do not appear in model.species
                                         # but are referenced in a reaction in model.reactions
                G.add_edge(rxn['id'], neighbour_id)  # disregard direction for now

    return G


def upsert_node(nxG:nx.Graph, node, **nodeattribs):
    # in networkx, a node can be any hashable object
    if node not in nxG.nodes():
        nxG.add_node(node, **nodeattribs)
    else:
        pass

def contains_node(nxG : nx.Graph, node):
    return node in nxG.nodes()


