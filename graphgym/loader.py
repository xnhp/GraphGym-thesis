import functools
import os
import warnings
from importlib.resources import files

import networkx
import networkx as nx
import time
import logging
import pickle

import numpy as np
import pandas
import pandas as pd
from graphgym.contrib.feature_augment.util import get_prediction_nodes, get_non_rxn_nodes
from graphgym.contrib.train.util import get_external_split_graphs
from graphgym.contrib.transform.normalize import normalize_scale, normalize_fit
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

import deepsnap
from data.util import groupby
from deepsnap.dataset import GraphDataset
import torch
from torch.utils.data import DataLoader

from torch_geometric.datasets import *
import torch_geometric.transforms as T

from graphgym.config import cfg
import graphgym.models.feature_augment as preprocess
from graphgym.models.transform import (ego_nets, remove_node_feature,
                                       edge_nets, path_len)
from graphgym.contrib.loader import *
import graphgym.register as register

from ogb.graphproppred import PygGraphPropPredDataset
from deepsnap.batch import Batch
from deepsnap.graph import Graph


def load_pyg(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset_raw = TUDataset(dataset_dir, name,
                                    transform=T.Constant())
        else:
            dataset_raw = TUDataset(dataset_dir, name[3:])
        # TU_dataset only has graph-level label
        # The goal is to have synthetic tasks
        # that select smallest 100 graphs that have more than 200 edges
        if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
            size = []
            for data in dataset_raw:
                edge_num = data.edge_index.shape[1]
                edge_num = 9999 if edge_num < 200 else edge_num
                size.append(edge_num)
            size = torch.tensor(size)
            order = torch.argsort(size)[:100]
            dataset_raw = dataset_raw[order]
    elif name == 'Karate':
        dataset_raw = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset_raw = Coauthor(dataset_dir, name='CS')
        else:
            dataset_raw = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset_raw = Amazon(dataset_dir, name='Computers')
        else:
            dataset_raw = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset_raw = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset_raw = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset_raw = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))
    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs


def load_nx(name, dataset_dir):
    '''
    load networkx format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    try:
        with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
            graphs = pickle.load(file)
    except:
        graphs = nx.read_gpickle('{}/{}.gpickle'.format(dataset_dir, name))
        if not isinstance(graphs, list):
            graphs = [graphs]
    return graphs


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        graphs = func(format, name, dataset_dir)
        if graphs is not None:
            return graphs
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        graphs = load_pyg(name, dataset_dir)
    # Load from networkx formatted data
    # todo: clean nx dataloader
    elif format == 'nx':
        graphs = load_nx(name, dataset_dir)
    # Load from OGB formatted data
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset)
        # Note this is only used for custom splits from OGB
        split_idx = dataset.get_idx_split()
        return graphs, split_idx
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node


def transform_before_split(dataset):
    '''
    Dataset transformation before train/val/test split
    :param dataset: A DeepSNAP dataset object
    :return: A transformed DeepSNAP dataset object
    '''
    if cfg.dataset.remove_feature:
        dataset.apply_transform(remove_node_feature,
                                update_graph=True, update_tensor=False)
    augmentation = preprocess.FeatureAugment()
    actual_feat_dims, actual_label_dim = augmentation.augment(dataset)
    if cfg.dataset.augment_label:
        dataset.apply_transform(preprocess._replace_label,
                                update_graph=True, update_tensor=False)
    # Update augmented feature/label dims by real dims (user specified dims
    # may not be realized)
    cfg.dataset.augment_feature_dims = actual_feat_dims
    if cfg.dataset.augment_label:
        cfg.dataset.augment_label_dims = actual_label_dim

    # Temporary for ID-GNN path prediction task
    if cfg.dataset.task == 'edge' and 'id' in cfg.gnn.layer_type:
        dataset.apply_transform(path_len, update_graph=False,
                                update_tensor=False)

    # note we cannot modify node_label or node_label_index here
    # (e.g. to exclude nodes or under/oversample) without further adjustments
    # since internal splitting assumes both to be of length n
    # although we don't do internal splits right now we still run through that code and it
    # would break
    return dataset


def exclude_graphs_with_no_pos_label(datasets: list[GraphDataset]) -> list[GraphDataset]:
    """
    Replaces the contained GraphDataset with filtered versions.
    using ↝ deepsnap.dataset.GraphDataset.filter
    :param datasets:
    :return:
    """

    def no_pos_label_filter(dsG):
        if len(dsG['node_label']) == 0:
            return True  # do not exclude these cases (internal split or sth)
        if 'is_test' in dsG.G.graph and dsG.G.graph['is_test']:
            return True  # do not exclude graphs external validation split
            # TODO problematic if we'd really give a seq there.
        r = sum(dsG['node_label']) > 0  # node label attributes in nxG graph are obsolete now
        if not r:
            print(f"{dsG['name']} \t no positive labels, dropping")
        return r

    return [ds.filter(no_pos_label_filter) for ds in datasets]


def transform_after_split(datasets):
    """
    Dataset transformation after train/val/test split
    :param datasets: A list of DeepSNAP dataset objects. Each GraphDataset contains *all* given graphs, only with
        node_label and node_label_index according to internal split
    :return: A list of transformed DeepSNAP dataset objects
    """
    if cfg.dataset.transform == 'ego':
        # seemingly computes ego graphs, but then each `dataset` here
        # is still one big graph object, probably containing all the little ego graphs
        # as disconnected subgraphs
        for split_dataset in datasets:
            split_dataset.apply_transform(ego_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
    elif cfg.dataset.transform == 'edge':
        for split_dataset in datasets:
            split_dataset.apply_transform(edge_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
            split_dataset.task = 'node'
        cfg.dataset.task = 'node'

    # Normalise node features
    if cfg.dataset.transform == 'normalize':
        fit_apply_normalization(datasets)

    datasets = exclude_graphs_with_no_pos_label(datasets)

    # graph contents (nodes/edges) determines which nodes are used for message-passing
    # node_label_index determines for which nodes we try to make a prediction

    # When using annotations:
    #   - Obtain mapping from aliases to annotations from disk (precomputed)
    #   - Obtain mapping from annotations to their embeddings (feature vector representation)
    #   - Take subgraph of given graph for aliases with available
    #   - Aggregate and attach embeddings as node attributes (node_GO_embedding)
    if cfg.dataset.use_annotated_subgraph:
        def get_subgraph(graph) -> nx.Graph:
            # assert that node_label_index was not restricted/modified yet
            assert len(graph['node_label_index']) == len(graph['node_label'])
            # construct subgraph and replace
            nx_subgraph = annotated_subgraph_transform(graph)
            return nx_subgraph

        for i, dataset in enumerate(datasets):  # internal train/test split
            # TODO only need to do this for one i
            # construct new GraphDataset aswell because it e.g. contains info on number of nodes of
            # contained graphs (its state depends on its contents)
            datasets[i] = GraphDataset([
                # create new dsG, relabeling nodes, recreating feature/attribute tensors,
                # node_label_index being range(n)
                deepsnap.graph.Graph(get_subgraph(graph))
                for graph in dataset
            ])

    # NOTE the below routines will modify node_label_index
    # Until above, we assume that no restrictions were made, i.e. node_label_index == range(num_nodes)

    # Exclude some classes of nodes from prediction. They will still be present in the graph and be used
    #   for message-passing in GNNs, but we will not try to make a prediction for these (or use their target class
    #   information for training)
    for dataset in datasets:
        subset_prediction_nodes(dataset)

    # Undersample the negative class, if enabeld.
    # Note that this still operates on single graphs in an insolated manner, if we are given a reorg seq
    #   and each is imbalanced, stacking them will further amplify the imbalance.
    # This is problematic if we ever re-enable internal split because it will undersample from each split
    if cfg.dataset.undersample_negatives_ratio not in [0, None]:
        for dataset in datasets:
            for dsG in dataset:
                undersample_negatives(dsG)

    return datasets


def read_annotation_data_for_graph(graph):
    # how to obtain the previously written information here?
    # ... need to get the right one for the current graph
    if "AlzPathway" in graph['name']:
        graph_dir = "mizuno_AlzPathwayComprehensiveMap_2021"
    elif "pd" in graph['name'] or "PD" in graph['name']:
        graph_dir = "pd_map_autumn_19"
    else:
        graph_dir = None
        return graph.G
    computed_subdir = os.path.join(files('computed'), graph_dir)
    target_path = os.path.join(computed_subdir, 'alias-embeddings.pickle')
    df = pandas.read_pickle(target_path)
    return df

def annotated_subgraph_transform(graph: deepsnap.graph.Graph) -> nx.Graph:
    """
    Replace graphs with subgraphs induced by available annotation data
    :param datasets:
    :return:
    """

    df: pd.DataFrame = read_annotation_data_for_graph(graph)
    nxG: networkx.Graph = graph.G

    def aggregate_embeddings(embs):
        embs = [e for e in embs if e is not None]
        if len(embs) == 0:
            return None
        else:
            return np.mean(embs, axis=0)

    def get_embedding(df, sa_id):
        # need to check if embedding is available in case we are trying to aggregate info of complex
        rows = df[df['Element external id'] == sa_id]
        if len(rows) != 1:  # cannot find alias id in df (or multiple matches)
            return None
        else:
            series = rows.iloc[0]
            emb = series['aggregated']
            return emb

    def set_embedding_for_node(graph, node_id, emb):
        graph.G.nodes[node_id]['node_GO_embedding'] = torch.tensor(emb)

    # node ids for which annotations are available, thus those which should make up the resulting subgraph
    subgraph_node_ids = set()

    # Iterate over all aliases for which we have embedding info. Try to map these to a node in the graph. If
    #   the alias does not exist as a node, disregard for now.
    for row in df.itertuples(name="Row", index=False):
        sa_id = row[1]
        emb = row.aggregated
        assert emb is not None
        try:
            sa_node_id = graph['mapping_alias_to_int'][sa_id]
            set_embedding_for_node(graph, sa_node_id, emb)
            subgraph_node_ids.add(sa_node_id)
        except KeyError:
            # otherwise the speciesAlias is assumed to be contained in a complex. These cases we handle below
            pass

    # For each root (top-level) complex species alias, lookup embedding info for its contained species aliases,
    #   aggregate it into a single vector and map that to the root complex species alias.
    csa_to_contained_sa = groupby(graph.G.graph['sa_root_map'].items(), lambda e: e[1]['id'])
    csa_to_aggregated_emb = {
        csa_id: aggregate_embeddings([get_embedding(df, sa[0]) for sa in contained_sa])
        for csa_id, contained_sa in csa_to_contained_sa.items()
    }
    for csa_id, emb in csa_to_aggregated_emb.items():
        if emb is not None:
            try:
                effective_node_id = graph['mapping_alias_to_int'][csa_id]
            except KeyError:
                # when dealing with collapsed graph and there are multiple aliases per species (*),
                # we pick one representative alias to put in the graph. Thus it may be the case here that a species
                # can not be found in the graph via its species alias id. Hence we need to check in the mapping.
                # * currently(?) only considering top-level species
                if not 'is_collapsed' in graph.G.graph or not graph.G.graph['is_collapsed']:
                    raise KeyError
                # effective_node_id = graph.G.graph['species_to_representative'][]
                tla2r = graph.G.graph['top_level_alias_to_representative']
                repr = tla2r[csa_id]
                effective_node_id = graph['mapping_alias_to_int'][repr]

            # TODO not properly coalesced into batch?
            set_embedding_for_node(graph, effective_node_id, emb)
            subgraph_node_ids.add(effective_node_id)

    # take subgraph because we can only do message-passing on nodes for which we have features
    return nxG.subgraph(subgraph_node_ids)


def fit_apply_normalization(datasets):
    """
    Fit normalizer to external train split, apply to all splits
    :param datasets:
    :return:
    """
    # TODO BUG: determine train_graphs like in create_loader!
    # this will respect the internal split but not the external
    train_graphs = datasets[0]
    scalers: dict
    # initialise the scaler(s) using train split
    scalers = normalize_fit(train_graphs)
    # datasets contain the same graph, only node_label_index is different
    for dataset in datasets:
        # apply scaler based on statistics from train split
        # apply_transform will act on each graph in the dataset independently
        dataset.apply_transform(normalize_scale, update_graph=True, update_tensor=False,
                                scalers=scalers)


def undersample_negatives(dsG):
    node_label = dsG['node_label']
    node_label_index = dsG['node_label_index']
    if len(node_label) <= 1:
        return
    # subset of node_label_index that has node_label 0
    # indices i with node_label[i] = 0
    zero_ix = node_label_index[(node_label == 0).nonzero()]  # create boolean mask, then call nonzero
    zero_ix = zero_ix.cpu().numpy()[:, 0]
    one_ix = node_label_index[(node_label == 1).nonzero()]  # positive class, should be equiv to node_label.nonzero()
    one_ix = one_ix.cpu().numpy()[:, 0]
    if len(zero_ix) == 0 or len(one_ix) == 0:
        warnings.warn(f"{dsG.G.graph['name']} \t only one class in labels!")
        return  # may happen e.g. for absurdly small internal train split
    if len(zero_ix) < len(one_ix) * cfg.dataset.undersample_negatives_ratio:
        # cannot "under"sample enough — this happens e.g. for absurdly small internal split
        sampling_strat = 'auto'  # take as many as we have, will result in balanced classes
    else:
        sampling_strat = 1 / cfg.dataset.undersample_negatives_ratio

    rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strat, replacement=False)
    _, y_sampled = rus.fit_resample(np.ones(len(node_label)).reshape(-1, 1), node_label)  # cleanup
    indices_sampled = rus.sample_indices_

    node_label_new = dsG['node_label'][indices_sampled]
    node_label_index_new = dsG['node_label_index'][indices_sampled]

    dsG['node_label'] = node_label_new
    dsG['node_label_index'] = node_label_index_new
    assert sum(node_label_new) == sum(node_label)
    assert sum(node_label_new) * (cfg.dataset.undersample_negatives_ratio + 1) == len(node_label_new)


def subset_prediction_nodes(dataset):
    """
    Updates node label tensors to exclude specific nodes.
    :param dataset:
    :return:
    """
    for dsG in dataset:
        # ↝ GraphGym/graphgym/contrib/train/SVM.py:41 (collect_per_graph)
        # cleanup: can most probably avoid some computations here
        print(f"{dsG.G.graph['name']} \t {len(dsG['node_label_index'])} number of labels (before exclude)")
        print(f"{dsG.G.graph['name']} \t {sum(dsG['node_label'])} label sum (before exclude)")
        included, _ = get_prediction_nodes(dsG.G)
        a = np.intersect1d(included, get_non_rxn_nodes(dsG.G))
        label_index_subset(dsG, a)
        print(f"{dsG.G.graph['name']} \t {len(dsG['node_label_index'])} number of labels (after exclude)")
        print(f"{dsG.G.graph['name']} \t {sum(dsG['node_label'])} label sum (after exclude)")


def label_index_subset(graph, selected_node_ids):
    intersection, picked, _ = np.intersect1d(graph['node_label_index'], list(selected_node_ids), return_indices=True)
    graph['node_label_index'] = torch.tensor(intersection)
    graph['node_label'] = graph['node_label'][picked]


def create_dataset():
    ## Load dataset
    time1 = time.time()
    graphs: list[deepsnap.graph.Graph]
    graphs = load_dataset()

    ## Filter graphs
    time2 = time.time()
    min_node = filter_graphs()

    ## Create whole dataset
    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        resample_disjoint=cfg.dataset.resample_disjoint,
        minimum_node_per_graph=min_node)

    ## Transform the whole dataset
    # apply feature augments
    dataset = transform_before_split(dataset)

    time3 = time.time()
    ## Split dataset
    # # Use custom data splits
    # # dataset.split assumes that node_label and node_label_index is untouched (of length n)
    # datasets = dataset.split(
    #     transductive=cfg.dataset.transductive,
    #     split_ratio=cfg.dataset.split,
    #     shuffle=cfg.dataset.shuffle_split)
    # # We only change the training negative sampling ratio
    # for i in range(1, len(datasets)):
    #     dataset.edge_negative_sampling_ratio = 1
    #

    datasets = [dataset]

    ## Transform each split dataset
    time4 = time.time()
    datasets = transform_after_split(datasets)

    time5 = time.time()
    logging.info('Load: {:.4}s, Before split: {:.4}s, '
                 'Split: {:.4}s, After split: {:.4}s'.format(
        time2 - time1, time3 - time2, time4 - time3, time5 - time4))

    return datasets


def create_loader(datasets):
    # datasets is python list of GraphDataset, the i-th one containing *all* given graphs but with node_label and
    # node_label_index corresponding to internal split i.
    if cfg.dataset.format == "SBML_multi":
        # consider explicit external split
        train_graphs, _, val_graphs = get_external_split_graphs(datasets)

        loader_train = DataLoader(train_graphs, collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=False)

        # loader_test = DataLoader([], collate_fn=Batch.collate(),
        #                          batch_size=cfg.train.batch_size,
        #                          shuffle=False,
        #                          num_workers=cfg.num_workers,
        #                          pin_memory=False)

        loader_val = DataLoader(val_graphs, collate_fn=Batch.collate(),
                                batch_size=cfg.train.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                pin_memory=False)

        return [loader_train, loader_val]
    else:
        # default behaviour
        loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=False)

        loaders = [loader_train]
        for i in range(1, len(datasets)):
            loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                      batch_size=cfg.train.batch_size,
                                      shuffle=False,
                                      num_workers=cfg.num_workers,
                                      pin_memory=False))

        return loaders
