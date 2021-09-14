import numpy as np
from graphgym.contrib.train.util import save_labels, get_external_split_graphs
from graphgym.logger import Logger
from graphgym.train import write_node_id_mappings, write_node_label_index

import deepsnap.graph
import torch
from graphgym.config import cfg
from graphgym.contrib.feature_augment.util import get_non_rxn_nodes, tens_intersect, collect_feature_augment, \
    get_prediction_nodes
from graphgym.register import register_train
from sklearn import svm


def collect_per_graph(graph:deepsnap.graph.Graph):
    """
    Obtain features and labels of nodes of the given graph, constrained to the current internal
    split (if desired) and non-reaction nodes.
    """

    included, _ = get_prediction_nodes(graph.G)
    # cleanup: could probably do this more elegantly by having above function return a mask tensor aswell
    #   it basically already does? could I then just subindex further?
    node_index = np.intersect1d(
        get_non_rxn_nodes(graph.G),
        included
    )
    # n_excluded = len(get_non_rxn_nodes(graph.G)) - len(node_index)
    # print(f"excluded {n_excluded} nodes from {graph['name']}")

    train_label_index, picked_test, _ = tens_intersect(
        graph['node_label_index'],
        torch.tensor(node_index)
    )

    feats = collect_feature_augment(graph)[train_label_index]
    labels = graph['node_label'][picked_test]
    print(f"{graph['name']} \t {graph['node_label_index'].cpu().numpy().shape} number of labels (after exclude)")
    print(f"{graph['name']} \t {graph['node_label'].cpu().numpy().sum()} label sum (after exclude)")

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


def run_svm(loggers, loaders, model, optimizer, scheduler, datasets):
    # expects to be given simple graphs as primary format, will construct and access bipartite projection
    # via attribute.

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

    train_graphs, _, val_graphs = get_external_split_graphs(datasets)

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
    # X_test, Y_test = collect_across_graphs(test_graphs)
    # internal "split" will point to *all* nodes
    X_val, Y_val = collect_across_graphs(val_graphs)  # no internal split (all nodes)

    # train ("fit") SVM model
    rbf_svc = svm.SVC(
        probability=True,
        kernel=cfg.model.svm_kernel,
        class_weight={
            0: 1,
            1: cfg.model.svm_class_weights
        },
        gamma=cfg.model.svm_gamma,
        C=cfg.model.svm_cost
    )
    rbf_svc.fit(X_train, Y_train)

    # write model predictions
    pred_train = predict(X_train, Y_train, logger_train, rbf_svc)
    save_labels(Y_train, 'Y_train', logger_train.out_dir)
    save_labels(pred_train[:, 1], 'pred_train', logger_train.out_dir)
    # if X_test.shape[0] > 0:
    #     pred_test = predict(X_test, Y_test, logger_test, rbf_svc)
    #     save_labels(Y_test, 'Y_test', logger_test.out_dir)
    #     save_labels(pred_test[:, 1], 'pred_test', logger_test.out_dir)
    # else:
    #     pred_test = torch.tensor([0]).to(torch.float)
    #     Y_test = torch.tensor([0]).to(torch.float)
    #     save_labels(Y_test, 'Y_test', logger_test.out_dir)
    #     save_labels(pred_test, 'pred_test', logger_test.out_dir)
    pred_val = predict(X_val, Y_val, logger_val, rbf_svc)
    save_labels(Y_val, 'Y_val', logger_val.out_dir)
    save_labels(pred_val[:, 1], 'pred_val', logger_val.out_dir)

    write_node_id_mappings(loaders, loggers)
    write_node_label_index(loaders, loggers)

    for logger in loggers:
        logger.close()


def predict(X, Y, logger, model):
    # evaluate on internal train split
    probas = model.predict_proba(X)  # will output probability scores!
    # class prediction obtained via cfg.model.thresh (default 0.5)
    probas = torch.from_numpy(probas).to(torch.float)
    # TODO time
    logger.update_stats(true=Y, pred=probas[:, 1], loss=0, lr=0, time_used=0, params=1)
    logger.write_epoch(0)
    return probas


register_train('run_svm', run_svm)
