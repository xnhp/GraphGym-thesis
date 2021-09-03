import itertools
import os
import warnings
from typing import Optional

import networkx as nx
from data.models import get_dataset
from data.util import is_collection_dataset, is_model_file
from graphgym.config import cfg
from graphgym.contrib.feature_augment.util import split_by_predicate
from graphgym.contrib.loader.SBML import sbml_single_impl, sbml_single_bipartite_projection_impl, \
    sbml_single_heterogeneous_impl
from graphgym.contrib.loader.reorganisation import set_labels_by_duplicate_alias, load_reorganisation_steps, \
    set_labels_by_step
from graphgym.contrib.loader.util import nxG_to_dsG
from graphgym.register import register_loader
from more_itertools import pairwise

import deepsnap.graph


def SBML_single_loader(_, name, __) -> Optional[list[deepsnap.graph.Graph]]:
    if cfg.dataset.format != "SBML":
        return None
    raise DeprecationWarning("no longer considering this approach, would need changes")
    path, model_class = get_dataset(name)
    nxG = sbml_single_impl(path, model_class, name, verbose_cache=True)
    # assume species graph approach
    set_labels_by_duplicate_alias(nxG)
    # TODO convert to dsG
    dsG._update_tensors()
    return [dsG]


register_loader('SBML_single', SBML_single_loader)


def SBML_single_bip_loader(_, name, __) -> Optional[list[deepsnap.graph.Graph]]:
    if cfg.dataset.format != "SBML_bip":
        return None
    raise DeprecationWarning("no longer considering this approach, would need changes")
    path, model_class = get_dataset(name)
    nxG = sbml_single_bipartite_projection_impl(path, model_class, name)
    set_labels_by_duplicate_alias(nxG)
    # TODO convert to dsG
    dsG._update_tensors()
    return [dsG]


register_loader('SBML_bip', SBML_single_loader)


def print_label_sums(graphs):
    for graph in graphs:
        labels = [
            label for _, label in
            list(graph.nodes(data="node_label"))
        ]
        print(f"{graph.name} \t {sum(labels)} sum of labels")
    pass


def SBML_multi(_, __, ___) -> Optional[list[deepsnap.graph.Graph]]:
    if cfg.dataset.format != "SBML_multi":
        return None
    if len(cfg.dataset.train_names) < 1 or len(cfg.dataset.test_names) < 1:
        raise Exception("No train and/or test datasets provided")

    loader_impl_fns = {
        'simple': sbml_single_impl,
        'het_bipartite': sbml_single_heterogeneous_impl,
        'bipartite_projection': sbml_single_bipartite_projection_impl
    }

    def mark(split:str, graph:nx.Graph):
        graph.graph['is_' + split] = True
        return graph

    def load_graphs(names):

        loader_impl = loader_impl_fns[cfg.dataset.graph_interpretation]

        collection_datasets_names, simple_datasets_names = split_by_predicate(
            zip(names, [get_dataset(name) for name in names]),
            lambda x: is_collection_dataset(x[1])
        )

        # TODO associate name with dataset and clean up this parameter mess
        def prepare_collection(name, dataset):
            collection_path, model_class = dataset
            dir_entries = sorted(
                [e for e in os.scandir(collection_path) if is_model_file(e)],
                # assume file names begin with a 3-digit int
                key = lambda e: e.name[0:3]
            )
            # construct collapsed graph based on first graph in sequence
            collapsed_graph = loader_impl(dir_entries[0].path, model_class, dir_entries[0].name, collapsed=True)
            # first graph in sequence is still part of sequence
            graphs = [loader_impl(entry.path, model_class, entry.name, collapsed=False)
                      for entry in dir_entries]
            for step in pairwise([collapsed_graph] + graphs):
                set_labels_by_step(*step)
            # additionally return collapsed graph
            # cannot infer labels for last graph in sequence, drop it
            reorg_seq = [collapsed_graph] + graphs[:-1]

            have_pos_labels, no_pos_labels = split_by_predicate(reorg_seq,
                    lambda g: sum([label for _, label in list(g.nodes(data="node_label"))]) > 0)

            if len(no_pos_labels) > 0:
                warnings.warn(f"{len(no_pos_labels)} graphs were excluded from sequence because no positive labels were determined")

            # return only graphs with positive labels, i.e. in which nodes were duplicated
            return have_pos_labels

        def prepare_simple(name, dataset):
            collapsed_graph = loader_impl(*dataset, name, collapsed=True)
            graph = loader_impl(*dataset, name, collapsed=False)
            set_labels_by_step(collapsed_graph, graph)
            print_label_sums([collapsed_graph])
            return collapsed_graph

        return [prepare_simple(n, ds) for n, ds in simple_datasets_names] \
               + list(itertools.chain(*
                                      [prepare_collection(n, ds) for n, ds in collection_datasets_names
                                       ]))

    # external split
    train_graphs = [mark('train', graph) for graph in load_graphs(cfg.dataset.train_names)]
    # note that by interface this supports reorganisation sequence datasets also as external test split
    #   but actually using these will require further consideration (e.g. [[#^2ed5e1]])
    test_graphs = [mark('test', graph) for graph in load_graphs(cfg.dataset.test_names)]

    # GG does not support such a "manual external split" of datasets
    return list(map(nxG_to_dsG, train_graphs + test_graphs))


register_loader('SBML_multi', SBML_multi)
