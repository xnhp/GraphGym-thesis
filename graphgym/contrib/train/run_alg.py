import torch
from graphgym.config import cfg
from graphgym.register import register_train
# Do not train a neural network but invoke some other algorithm
# e.g. baseline classifier such as SVM
from sklearn import svm


def collect_feature_augment(batch):
    # collect information from feature augments to node_feature tensor
    # these lines are based on `Preprocess` in feature_augment.py
    dim_dict = {name: dim
                for name, dim in zip(cfg.dataset.augment_feature,
                                     cfg.dataset.augment_feature_dims)}
    return torch.cat(
        [batch[name].float() for name in dim_dict],
        dim=1)


def run_alg(loggers, loaders, model, optimizer, scheduler):
    global comms
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
    X_augm = collect_feature_augment(batch_train)  # can use any batch here
    X_train = X_augm[batch_train.node_label_index]  # this is essentially our feature matrix
    # node_label_index is the index mask of nodes to consider in this split
    # e.g. in semi-supervised node classification we would know only some labels
    # and try to predict the others
    Y_train = batch_train.node_label  # labels of same size as node_label_index
    X_test = X_augm[batch_test.node_label_index]  # this is essentially our feature matrix
    Y_test = batch_test.node_label

    # train SVM model
    rbf_svc = svm.SVC(
        kernel=cfg.model.svm_kernel,
        class_weight={
            label: weight
            for label, weight in enumerate(cfg.model.class_weights)
        },
        gamma=cfg.model.svm_gamma,
        C=cfg.model.svm_cost
    )
    # evaluate on train split
    rbf_svc.fit(X_train, Y_train)
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


register_train('run_alg', run_alg)
