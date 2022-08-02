# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午4:13
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn
from yacs.config import CfgNode

from zcls2.model.criterion import cross_entropy_loss, large_margin_softmax_loss, soft_target_cross_entropy_loss

from . import mse_loss, triplet_margin_loss

__all__ = ['build_criterion']


def build_criterion(cfg: CfgNode) -> nn.Module:
    loss_name = cfg.MODEL.CRITERION.NAME
    reduction = cfg.MODEL.CRITERION.REDUCTION

    margin = cfg.MODEL.CRITERION.MARGIN
    p = cfg.MODEL.CRITERION.P
    mining = cfg.MODEL.CRITERION.MINING

    if loss_name in cross_entropy_loss.__all__:
        label_smoothing = cfg.MODEL.CRITERION.LABEL_SMOOTHING
        criterion = cross_entropy_loss.__dict__[loss_name](reduction=reduction, label_smoothing=label_smoothing)
    elif loss_name in large_margin_softmax_loss.__all__:
        criterion = large_margin_softmax_loss.__dict__[loss_name](reduction=reduction)
    elif loss_name in soft_target_cross_entropy_loss.__all__:
        criterion = soft_target_cross_entropy_loss.__dict__[loss_name]()
    elif loss_name in mse_loss.__all__:
        criterion = mse_loss.__dict__[loss_name]()
    elif loss_name in triplet_margin_loss.__all__:
        criterion = triplet_margin_loss.__dict__[loss_name](margin, p, mining)
    else:
        raise ValueError(f"{loss_name} does not support")

    return criterion
