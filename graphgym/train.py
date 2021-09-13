import logging
import os
import time

import torch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.config import cfg
from graphgym.contrib.train.util import save_labels, save_dict
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    assert len(loader) == 1
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()

@torch.no_grad()
def write_predictions_for_all_splits(model, loaders, loggers, epoch):
    # assume only single batch in loader, else we have to approach this differently
    assert len(loaders[0]) == 1
    names = ['train', 'test', 'val']  # ↝ compare-models.get_prediction_and_truth

    for loader, logger, name in zip(loaders, loggers, names):
        assert len(loader) <= 1  # == 1 probably suffices currently, think i set that to <= somewhen when messing with internal splits
        for batch in loader:
            batch.to(torch.device(cfg.device))
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            targetdir = os.path.join(logger.out_dir, "preds", str(epoch))
            save_labels(true.cpu(), "Y_" + name, targetdir)
            save_labels(pred_score.cpu(), 'pred_' + name, targetdir)

def train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    # cleanup: could also do this way sooner? particularly if we only consider
    # case of a single graph, i.e. dont have to consider what batching does.
    # (e.g. in create_loader or shortly after)
    write_node_id_mappings(loaders, loggers)
    write_node_label_index(loaders, loggers)

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        # at each k-th epoch, evaluate the model on the remaining internal splits
        # this is a bit weird and would only make sense in the case of having exactly two internal splits
        # also cannot find any example that uses more than two internal splits
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                if loggers[i].name == 'val':
                    continue
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
            write_predictions_for_all_splits(model, loaders, loggers, cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


def write_node_id_mappings(loaders, loggers):
    # loaders[2] is external test split
    # assume that there is only one graph in external test split
    # 607db5: at least for this one we want to manually assess predictions and thus need to map back to alias ids
    # note that the one graph in external test split will correspond to a collapsed version, hence we cannot
    # conveniently open up a CD/SBML drawing of it. But we can still look at the "next"/G_{t+1} which is an actual map.
    ext_train_batch = loaders[2].__iter__().next()
    save_dict(ext_train_batch['mapping_int_to_alias'], "mapping_int_to_alias", loggers[2].out_dir)

def write_node_label_index(loaders, loggers):
    ext_train_batch = loaders[2].__iter__().next()
    save_labels(ext_train_batch['node_label_index'], 'node_label_index', loggers[2].out_dir)
