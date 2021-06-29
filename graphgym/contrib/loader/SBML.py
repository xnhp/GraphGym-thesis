import functools

import deepsnap.dataset
import networkx as nx
from data.util import get_dataset

from graphgym.register import register_loader
from graphgym.config import cfg
from lxml import etree
import os
import itertools
import numpy as np
import torch


def SBML_single(format, name, dataset_dir) -> list[deepsnap.graph.Graph]:
    if cfg.dataset.format != "SBML":
        return None

    nxG : nx.Graph
    nxG = graph_from_celldesigner(get_dataset(name))

    # do not really have extracted real node features yet but have to set some
    # do not
    for nodeIx in nxG.nodes:
        feat = np.zeros(1)
        feat = torch.from_numpy(feat).to(torch.float)
        nxG.nodes[nodeIx]['node_feature'] = feat
        # node label should be set

    dsG = deepsnap.graph.Graph(nxG)
    return [dsG]

register_loader('SBML_single', SBML_single)

def graph_from_celldesigner(path) -> nx.Graph:
    """
    Read a CellDesigner SBML file and construct a networkx graph
    :param path:
    :return:
    """
    G = nx.Graph()
    G.graph['name'] = os.path.basename(path)

    model = CellDesignerModel(path)

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
        reaction_node = upsert_node(G, rxn['id'], **rxn_data)  # we use id as key and the entire dict (containg id) as additional attributes
        # do not need to explicitly denote here that this is a reaction node because we already set its `type` attribute
        for neighbour_id in rxn['reactants'] + rxn['products'] + rxn['modifiers']:
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

class CellDesignerModel:
    def __init__(self, filepath):
        self.tree = etree.parse(filepath)
        self.root = self.tree.getroot()
        # need to explicitly add these namespaces
        self.nsmap = self.root.nsmap.copy()
        self.nsmap['rdf'] = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        self.nsmap['dc'] = "http://purl.org/dc/elements/1.1/"
        self.nsmap['dcterms'] = "http://purl.org/dc/terms/"
        self.nsmap['vCard'] = "http://www.w3.org/2001/vcard-rdf/3.0#"
        self.nsmap['bqbiol'] = "http://biomodels.net/biology-qualifiers/"
        self.nsmap['bqmodel'] = "http://biomodels.net/model-qualifiers/"

    @functools.cached_property
    def species_aliases(self) -> list[dict]:
        cd_extension = self.tree.find("/model/annotation/celldesigner:extension", self.nsmap)
        assert cd_extension is not None
        speciesAliases = cd_extension.findall("celldesigner:listOfSpeciesAliases/celldesigner:speciesAlias", self.nsmap)
        return [alias.attrib for alias in speciesAliases]

    @functools.cached_property
    def duplicate_aliases(self) -> list[dict]:
        # sort by key required before groupby
        aliases_sorted = sorted(self.species_aliases, key=lambda x:x['species'])
        # could also operate on the iterator that groupby returns but
        # for values to persist (not be shared), we have to put them into a list
        grouped = {}
        for key, group in itertools.groupby(aliases_sorted, lambda x:x['species']):
            grouped[key] = list(group)
        duplicates = {key: group for key, group in grouped.items() if len(group) > 1}
        return duplicates

    @functools.cached_property
    def species(self) -> list[dict]:
        listOfSpeciesEl = self.tree.find("/model/listOfSpecies", self.nsmap)
        assert listOfSpeciesEl is not None
        r = []
        for species in listOfSpeciesEl.findall('species', self.nsmap):
            d = {}
            # species id (or should we use the `metaid` attrib instead?)
            d['id'] = species.attrib['id']
            d['type'] = 'species'
            # species/node class (as per [[^2e2cfd]])
            cd_annots = species.find("annotation/celldesigner:extension", self.nsmap)
            d['class'] = cd_annots.find("celldesigner:speciesIdentity/celldesigner:class", self.nsmap).text
            # TODO reconsider characterisation of duplicates
            d['is_duplicate'] = species.attrib['id'] in self.duplicate_aliases
            d['node_label'] = d['is_duplicate']  # GG expects this name
            # TODO annotations, ↝ read-annotations.ipynb
            r.append(d)
        return r


    @functools.cached_property
    def reactions(self):
        def extract_species_reference(el):
            return el.attrib['species']
        listOfRxnEl = self.tree.find("/model/listOfReactions", self.nsmap)
        assert listOfRxnEl is not None
        r = []
        for rxn in listOfRxnEl.findall('reaction', self.nsmap):
            d = {}
            d['id'] = rxn.attrib['id']
            d['type'] = 'reaction'
            d['node_label'] = 0  # placeholder, should not be used
            d['reactants'] = [extract_species_reference(el) for el in rxn.findall("listOfReactants/speciesReference", self.nsmap)]
            d['products'] = [extract_species_reference(el) for el in rxn.findall("listOfProducts/speciesReference", self.nsmap)]
            d['modifiers'] = [extract_species_reference(el) for el in rxn.findall("listOfModifiers/speciesReference", self.nsmap)]
            # TODO need to set node_label here aswell?
            # TODO annotations from CellDesigner and RDF annotations ↝ read-annotations.ipynb
            r.append(d)
        return r
