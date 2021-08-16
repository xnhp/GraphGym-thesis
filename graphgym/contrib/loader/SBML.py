import networkx
import networkx as nx
from cachier import cachier
from graphgym.contrib.feature_augment.util import split_rxn_nodes, bipartite_projection_onto_non_rxn
from graphgym.contrib.loader.util import nxG_to_dsG, load_nxG

import deepsnap.dataset
from deepsnap.hetero_graph import HeteroGraph


def sbml_single_bipartite_projection_impl(path, model_class, name, **kwargs) -> nx.Graph:
    nxG = load_nxG(path, model_class, name, **kwargs)
    # compute bipartite projection
    bipartite_projection: networkx.Graph
    bipartite_projection, _ = bipartite_projection_onto_non_rxn(nxG)

    # graph attributes are copied over to dsG
    bipartite_projection.graph['is_bipartite_projection'] = True
    bipartite_projection.graph['simple_graph'] = nxG_to_dsG(nxG)

    return bipartite_projection

    # dsG = nxG_to_dsG(bipartite_projection)
    # # attach simple graph as attribute
    # dsG['is_bipartite_projection'] = True
    # dsG['simple_graph'] = nxG_to_dsG(nxG)
    # return dsG


def sbml_single_heterogeneous_impl(path, model_class, name, **_kwargs) -> nx.Graph:
    """
    :param name:  identifier of the dataset to load from
    :return: a heterogeneous graph with only two node types: reactions and species (all others)
    """
    raise NotImplementedError   # needs more work
    dsG: deepsnap.graph.Graph
    nxG = sbml_single_impl(path, model_class, name, verbose_cache=True, **_kwargs)
    rxn_nodes, non_rxn_nodes = split_rxn_nodes(nxG)
    for node_id, _ in rxn_nodes:
        nxG.nodes[node_id]['node_type'] = "het_reaction_t"
    for node_id, _ in non_rxn_nodes:
        nxG.nodes[node_id]['node_type'] = "het_species_t"
    # TODO do we need to set edge types?
    return nxG
    # return HeteroGraph(nxG)


def sbml_single_impl(path, model_class, name, **kwargs) -> nx.Graph:
    nxG = load_nxG(path, model_class, name, **kwargs)
    return nxG
    # return nxG_to_dsG(nxG)
