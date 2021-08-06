import networkx
from cachier import cachier
from graphgym.contrib.feature_augment.util import split_rxn_nodes, bipartite_projection_onto_non_rxn

from deepsnap.hetero_graph import HeteroGraph
from pytictoc import TicToc

import deepsnap.dataset
import networkx as nx
from data.util import get_dataset, CellDesignerModel, SBMLModel

from graphgym.register import register_loader
from graphgym.config import cfg
import os
import numpy as np
import torch


def SBML_multi(format, name, dataset_dir) -> list[deepsnap.graph.Graph]:
    if cfg.dataset.format != "SBML_multi":
        return None
    if len(cfg.dataset.train_names) < 1 or len(cfg.dataset.test_names) < 1:
        raise Exception("No train and/or test datasets provided")

    loader_fns = {
        'simple': sbml_single_impl,
        'het_bipartite': sbml_single_heterogeneous,
        'bipartite_projection': sbml_single_bipartite_projection
    }
    loader = loader_fns[cfg.dataset.graph_interpretation]

    def mark(split, graph):
        graph['is_' + split] = True
        return graph

    train_graphs = [mark('train', loader(name, verbose_cache=True)) for name in cfg.dataset.train_names]
    test_graphs = [mark('test', loader(name, verbose_cache=True)) for name in cfg.dataset.test_names]

    return train_graphs + test_graphs


register_loader('SBML_multi', SBML_multi)


def SBML_single(format, name, dataset_dir) -> list[deepsnap.graph.Graph]:
    if cfg.dataset.format != "SBML":
        return None

    return [sbml_single_impl(name, verbose_cache=True)]


def SBML_single_bip_loader(format, name, dataset_dir) -> list[deepsnap.graph.Graph]:
    if cfg.dataset.format != "SBML_bip":
        return None
    return [sbml_single_heterogeneous(name)]


def sbml_single_bipartite_projection(name, **kwargs) -> deepsnap.graph.Graph:

    # load graph as usual
    nxG = load_nxG(name)
    # compute bipartite projection
    bipartite_projection: networkx.Graph
    bipartite_projection, _ = bipartite_projection_onto_non_rxn(nxG)

    dsG = nxG_to_dsG(bipartite_projection)
    # attach simple graph as attribute
    dsG['is_bipartite_projection'] = True
    dsG['simple_graph'] = nxG_to_dsG(nxG)

    return dsG
    # in feature augments, switch which graph to use


def sbml_single_heterogeneous(name, **_kwargs) -> deepsnap.hetero_graph.HeteroGraph:
    """
    :param name:  identifier of the dataset to load from
    :return: a heterogeneous graph with only two node types: reactions and species (all others)
    """
    dsG: deepsnap.graph.Graph
    dsG = sbml_single_impl(name, verbose_cache=True)
    nxG = dsG.G
    rxn_nodes, non_rxn_nodes = split_rxn_nodes(nxG)
    for node_id, _ in rxn_nodes:
        nxG.nodes[node_id]['node_type'] = "het_reaction_t"
    for node_id, _ in non_rxn_nodes:
        nxG.nodes[node_id]['node_type'] = "het_species_t"

    # TODO do we need to set edge types?
    return HeteroGraph(nxG)


@cachier()
def sbml_single_impl(name) -> deepsnap.graph.Graph:
    nxG = load_nxG(name)
    return nxG_to_dsG(nxG)


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


@cachier()
def load_nxG(name):
    """
    Provides the networkx graph that can be wrapped into a DeepSNAP graph
    for use in a DeepSNAP dataset.
    :param name:
    :return:
    """
    path, model_class = get_dataset(name)
    model = model_class(path)
    nxG: nx.Graph
    nxG = graph_from_model(model, name=name)
    return nxG


register_loader('SBML_single', SBML_single)


def graph_from_model(model: SBMLModel, name=None) -> nx.Graph:
    """
    :param name:
    :param model:
    :return:
    """
    G = nx.Graph()
    G.graph['name'] = name if name is not None else model.path

    # fetch full info about species and already add as nodes
    for species in model.species:
        G.add_node(species['id'], **species)

    # add nodes for reactions and edges from/to reactions
    for rxn in model.reactions:
        rxn: dict  # of info about this reaction
        rxn_data = rxn.copy()
        del rxn_data['reactants']
        del rxn_data['products']
        del rxn_data['modifiers']
        reaction_node = G.add_node(rxn['id'],
                                   **rxn_data)  # we use id as key and the entire dict (containg id) as additional attributes
        # do not need to explicitly denote here that this is a reaction node because we already set its `type` attribute
        for neighbour_id in rxn['reactants'] + rxn['products'] + rxn['modifiers']:
            if neighbour_id in G.nodes:  # `add_edge` adds nodes if they don't exist yet
                # there might be excluded species that do not appear in model.species
                # but are referenced in a reaction in model.reactions
                G.add_edge(rxn['id'], neighbour_id)  # disregard direction for now

    low_deg_nodes = [node for (node, degree) in G.degree if degree < SBMLModel.min_node_degree]
    G.remove_nodes_from(low_deg_nodes)
    # note that this does not necessarily mean that after this G will contain no nodes of low degree
    # we have to accept this...
    # ... but we need to remove isolated nodes because we cannot compute neighbourhood statistics for them
    #     and in any case duplicating isolated nodes seems out of scope
    no_deg_nodes = [node for (node, degree) in G.degree if degree == 0]
    G.remove_nodes_from(no_deg_nodes)
    degrees = [deg for (node, deg) in G.degree()]
    # assert max(degrees) >= SBMLModel.min_node_degree
    assert min(degrees) > 0

    return G


def upsert_node(nxG: nx.Graph, node, **nodeattribs):
    # in networkx, a node can be any hashable object
    if node not in nxG.nodes():
        nxG.add_node(node, **nodeattribs)
    else:
        pass


def contains_node(nxG: nx.Graph, node):
    return node in nxG.nodes()
