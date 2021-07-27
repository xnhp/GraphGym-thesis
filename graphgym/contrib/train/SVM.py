import os

from graphgym.logger import Logger

import deepsnap.graph
import pandas as pd
import torch
from graphgym.config import cfg
from graphgym.contrib.feature_augment.util import get_non_rxn_nodes, tens_intersect, collect_feature_augment
from graphgym.register import register_train
# Do not train a neural network but invoke some other algorithm
# e.g. baseline classifier such as SVM
from sklearn import svm


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


def collect_per_graph(graph:deepsnap.graph.Graph):
    """
    Obtain features and labels of nodes of the given graph, constrained to the current internal
    split (if desired) and non-reaction nodes.
    """
    train_label_index, picked_test, _ = tens_intersect(
        graph['node_label_index'],
        torch.tensor(get_non_rxn_nodes(graph.G))
    )
    feats = collect_feature_augment(graph)[train_label_index]
    labels = graph['node_label'][picked_test]
    return feats, labels


def collect_across_graphs(graphs:list[deepsnap.graph.Graph]):
    """
    Collect features and labels of the given graphs, constrained to the current internal
    split as indicated by the graph's node_label_index attribute and only for non-rxn nodes.
    """
    l = [collect_per_graph(graph) for graph in graphs]
    feats = torch.cat([x[0] for x in l], dim=0)
    labels = torch.cat([x[1] for x in l], dim=0)
    return feats, labels


def save_labels(tens: torch.Tensor, filename, out_dir):
    df = pd.DataFrame(tens.numpy())
    df.to_csv(os.path.join(out_dir, filename + ".csv"))
    # previous approach:
    # torch.save(Y_train, os.path.join(logger_train.out_dir, 'Y_train.pt'))


def run_svm(loggers, loaders, model, optimizer, scheduler, datasets):
    # batch_train and batch_test contain *all* given graphs, each of these having a resp. node_label_index mask
    # for *internal* train/test split.
    # the batch contains as attributes (features) of *all* graphs in a stacked (indistinguishable) shape.
    #   The reference to the graph in the batch points to the networkx graph.
    #   This is why we need to additionally pass in `datasets` from main.py
    from pytictoc import TicToc
    t = TicToc()
    t.tic()
    logger_train = loggers[0]
    logger_test = loggers[1]
    logger_val = Logger(name="val-graph", task_type='classification_binary')
    assert logger_val.task_type == logger_train.task_type

    # There is two kinds of splits happening in our pipeline:
    #   - "internal" split: a graphs node set is partitioned into train/test split. This is indicated by the
    #      graph['node_label_index'] attribute, graphs are the same otherwise
    #   - "external" split (added by us): we explicitly give some datasets that should be used for training and
    #      some for evaluating the model.
    # Let's call the graphs of the external split "train" and "validation" graphs.

    # `datasets` contains two instances of `GraphDataset`, each containing *each* input graph; each
    #   of these graphs has node_label_index set according to internal split
    # So, to recover the external (explicitly given) split, we have to do our own thing.

    # these are the graphs in external train split, with node idx for internal train/test split
    # is_train / is_test attributes are being set by loader based on what is specified in config yaml
    train_graphs = [graph for graph in datasets[0].graphs if graph['is_train']]
    test_graphs = [graph for graph in datasets[1].graphs if graph['is_train']]

    # these are the graphs in the external test split. here, we want to consider the entire graph.
    val_graphs_full = collect_val_graphs(datasets)

    # to feed this into a generic ML model, we need to collect and concat/stack all the features and labels from the
    #   resp. graphs in there
    # this is what we would otherwise obtain from `DataLoader` and `batch` if we did a custom
    #   split like in GraphGym/graphgym/loader.py:239 (special case for OGB). However, then we would not immediately
    #   have the internal splitting. We could still use that but would have to call some deeper method like
    #   deepsnap.dataset.GraphDataset._split_inductive which is nasty and full of special cases.

    # collect features from all train graphs
    # we additionally need to select the right nodes of the internal split since indices are relative to the single
    # graph
    X_train, Y_train = collect_across_graphs(train_graphs)
    X_test, Y_test = collect_across_graphs(test_graphs)
    # internal "split" will point to *all* nodes
    X_val, Y_val = collect_across_graphs(val_graphs_full)  # no internal split (all nodes)

    # train ("fit") SVM model
    rbf_svc = svm.SVC(
        probability=True,
        kernel=cfg.model.svm_kernel,
        class_weight={
            0: 1,
            1: cfg.model.class_weights
        },
        gamma=cfg.model.svm_gamma,
        C=cfg.model.svm_cost
    )
    rbf_svc.fit(X_train, Y_train)

    pred_train = predict(X_train, Y_train, logger_train, rbf_svc)
    pred_test = predict(X_test, Y_test, logger_test, rbf_svc)
    pred_val = predict(X_val, Y_val, logger_val, rbf_svc)

    # write model prediction. Doing so here is simpler than saving/loading the model as we would do
    #   with the checkpointing facility and pytorch models.
    # in case of pytorch modules we can load the checkpoint as we did in project?
    # output only probabilities of "larger" class, 1d expected downstream
    save_labels(Y_test, 'Y_test', logger_test.out_dir)
    save_labels(pred_test[:, 1], 'pred_test', logger_test.out_dir)
    save_labels(Y_train, 'Y_train', logger_train.out_dir)
    save_labels(pred_train[:, 1], 'pred_train', logger_train.out_dir)
    save_labels(Y_val, 'Y_val', logger_val.out_dir)
    save_labels(pred_val, 'pred_val', logger_val.out_dir)

    for logger in loggers:
        logger.close()


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


def predict(X, Y, logger, model):
    # evaluate on internal train split
    pred = model.predict_proba(X)  # will output probability scores!
    # class prediction obtained via cfg.model.thresh (default 0.5)
    pred = torch.from_numpy(pred).to(torch.float)
    # TODO time
    logger.update_stats(true=Y, pred=pred[:, 1], loss=0, lr=0, time_used=0, params=1)
    logger.write_epoch(0)
    return pred


register_train('run_svm', run_svm)
