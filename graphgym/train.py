import os

import torch
import time
import logging

from graphgym.config import cfg
from graphgym.contrib.train.util import save_labels
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt


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
        assert len(loader) == 1
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

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        # at each k-th epoch, evaluate the model on the remaining internal splits
        # this is a bit weird and would only make sense in the case of having exactly two internal splits
        # also cannot find any example that uses more than two internal splits
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
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
