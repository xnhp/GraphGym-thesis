import torch
import torch.nn as nn

from graphgym.register import register_loss

from graphgym.config import cfg


def loss_bce_weighted(pred, true):
    if cfg.model.loss_fun == 'cross_entropy_weighted':
        pos_weight = cfg.dataset.minority_loss_weight
        bce_loss = nn.BCEWithLogitsLoss(
            reduction=cfg.model.size_average,   # same as in original approach
            pos_weight=torch.tensor([pos_weight]).to(torch.device(cfg.device))
        )
        true = true.float()
        return bce_loss(pred, true), torch.sigmoid(pred)


register_loss('cross_entropy_weighted', loss_bce_weighted)
