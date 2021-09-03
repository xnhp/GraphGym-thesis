import csv
import os
from typing import Tuple

import pandas as pd
import torch

import deepsnap.graph


def save_labels(tens: torch.Tensor, filename, out_dir):
    df = pd.DataFrame(tens.numpy())
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, filename + ".csv"))
    # previous approach:
    # torch.save(Y_train, os.path.join(logger_train.out_dir, 'Y_train.pt'))

def save_dict(d:dict, filename, out_dir):
    target = os.path.join(out_dir, f"{filename}.csv")
    with open(target, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerows(list(d.items()))


def get_external_split_graphs(datasets) -> Tuple[list[deepsnap.graph.Graph], list[deepsnap.graph.Graph], list[deepsnap.graph.Graph]]:
    # these are the graphs in external train split, with node idx for internal train/test split
    # is_train / is_test attributes are being set by loader based on what is specified in config yaml
    # internal train/test split is reflected in datasets[0] and datasets[1], the is_train flag
    # describes the external train/test split (TODO rename)
    train_graphs = [graph for graph in datasets[0].graphs if graph['is_train']]
    test_graphs = [graph for graph in datasets[1].graphs if graph['is_train']]
    # these are the graphs in the external test split. here, we want to consider the entire graph.
    val_graphs = collect_val_graphs(datasets)
    return train_graphs, test_graphs, val_graphs


def logical_and_pad(a, b):
    """
    perform logical-and between a and b, potentially padding one of either
    :param a:
    :param b:
    :return:
    """
    if a.shape[0] == b.shape[0]:
        return torch.logical_and(a, b)
    if a.shape[0] < b.shape[0]:
        smaller = a
        greater = b
    else:
        smaller = b
        greater = a
    diff = abs(a.shape[0] - b.shape[0])
    padding = torch.zeros(diff)
    padded = torch.cat([smaller, padding], dim=0)
    return torch.logical_and(greater, padded)


def collect_val_graphs(datasets):
    """
    Collect validation graphs.
    complication is that also these graphs are being split internally
    and each representation only has node labels for its subset.
    However, we want to consider the entire graph.
    Hence, here, recollect node_labels across internal splits
    :param datasets:
    :return:
    """
    val_graphs_train = [graph for graph in datasets[0].graphs if graph['is_test']]
    val_graphs_test = [graph for graph in datasets[1].graphs if graph['is_test']]
    val_graphs_full = []
    for train_graph, test_graph in zip(val_graphs_train, val_graphs_test):
        dummy = train_graph  # mind we modify this, but eh should be fine
        dummy['node_label'] = torch.cat([train_graph['node_label'], test_graph['node_label']], dim=0)
        # this below is nonsensical because this will just be *all* nodes but its the simplest for now
        dummy['node_label_index'] = torch.cat([train_graph['node_label_index'], test_graph['node_label_index']], dim=0)
        val_graphs_full.append(dummy)
    return val_graphs_full