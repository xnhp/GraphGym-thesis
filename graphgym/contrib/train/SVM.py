import numpy as np
import torch
from graphgym.config import cfg
from graphgym.contrib.feature_augment.util import get_non_rxn_nodes
from graphgym.register import register_train
# Do not train a neural network but invoke some other algorithm
# e.g. baseline classifier such as SVM
from sklearn import svm


def collect_feature_augment(batch, node_index):
    """
    Concat information from feature augments into node_feature tensor. Same as `Processing` module does in GNN models.
    Additionally takes a node_index mask of nodes to consider.
    :param batch:
    :param node_index:
    :return:
    """
    dim_dict = {name: dim
                for name, dim in zip(cfg.dataset.augment_feature,
                                     cfg.dataset.augment_feature_dims)}
    return torch.cat(
        [batch[name][node_index].float() for name in dim_dict],
        dim=1)


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


def tens_intersect(x: torch.Tensor, y: torch.Tensor):
    intersect, x_ind, y_ind = np.intersect1d(x.cpu().numpy(), y.cpu().numpy(), return_indices=True)
    return torch.tensor(intersect), torch.tensor(x_ind), torch.tensor(y_ind)


def run_svm(loggers, loaders, model, optimizer, scheduler):
    from pytictoc import TicToc
    t = TicToc()

    # we abuse the batch / dataloader abstractions and not make
    # no use of them at all. instead, we only use them to access
    # the data directly.
    # we have one loader per train/test/val split.
    # in context of GG a "batch" is always a collection of graphs?
    # -> in semi-supervised node classification, this will always be a single graph?
    assert len(loaders[0]) == 1  # only one batch in loader
    batch_train = loaders[0]._get_iterator().next()
    batch_test = loaders[1]._get_iterator().next()
    some_batch = batch_train
    graph = some_batch.G[0]

    # consider only nodes that are not reactions because
    # - we never want to duplicate reactions
    # - bipartite projection and its node features do not contain reactions
    non_rxn_index = torch.tensor(get_non_rxn_nodes(graph))
    X_augm = collect_feature_augment(some_batch, non_rxn_index)  # feature matrix

    # node_label_index is the index mask of nodes to consider in this split
    #   e.g. in semi-supervised node classification we would know only some labels
    #   and try to predict the others
    # tensors node_label_index (identifying nodes in train/test split) and
    #   node_label have an implicit mapping, i.e. their order corresponds.
    # When selecting non-rxn nodes from node_label_index we need to do the same
    #   for node_label. We remember the indices we picked from node_label_index
    #   and use these to index into node_label.
    train_label_index, picked_test, _ = tens_intersect(batch_train.node_label_index, non_rxn_index)
    test_label_index, picked_train, _ = tens_intersect(batch_test.node_label_index, non_rxn_index)
    X_train = X_augm[train_label_index]  # this is essentially our feature matrix
    X_test = X_augm[test_label_index]
    Y_train = batch_train.node_label[picked_test]
    Y_test = batch_test.node_label[picked_train]

    # train ("fit") SVM model
    rbf_svc = svm.SVC(
        kernel=cfg.model.svm_kernel,
        class_weight={
            label: weight
            for label, weight in enumerate(cfg.model.class_weights)
        },
        gamma=cfg.model.svm_gamma,
        C=cfg.model.svm_cost
    )
    rbf_svc.fit(X_train, Y_train)

    # evaluate on train split
    pred_train = rbf_svc.predict(X_train)
    pred_train = torch.from_numpy(pred_train).to(torch.float)
    logger_train = loggers[0]
    logger_train.update_stats(true=Y_train, pred=pred_train, loss=0, lr=0, time_used=t.tocvalue(), params=1)
    logger_train.write_epoch(0)

    # evaluate on test split
    pred_test = rbf_svc.predict(X_test)
    pred_test = torch.from_numpy(pred_test).to(torch.float)
    logger_test = loggers[1]
    logger_test.update_stats(true=Y_test, pred=pred_test, loss=0, lr=0, time_used=t.tocvalue(), params=1)
    logger_test.write_epoch(0)

    for logger in loggers:
        logger.close()


register_train('run_svm', run_svm)
