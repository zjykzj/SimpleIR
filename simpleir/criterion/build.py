# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午4:13
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn
from yacs.config import CfgNode

from zcls2.model.criterion.cross_entropy_loss import build_cross_entropy_loss
from zcls2.model.criterion.large_margin_softmax_loss import build_large_margin_softmax_loss

from .mse_loss import build_mse_loss

__supported_criterion__ = [
    'CrossEntropyLoss',
    'LargeMarginSoftmaxV1',
    'MSELoss'
]


def build_criterion(cfg: CfgNode) -> nn.Module:
    loss_name = cfg.MODEL.CRITERION.NAME
    reduction = cfg.MODEL.CRITERION.REDUCTION

    assert loss_name in __supported_criterion__

    if loss_name == 'CrossEntropyLoss':
        return build_cross_entropy_loss(reduction=reduction)
    elif loss_name == 'LargeMarginSoftmaxV1':
        return build_large_margin_softmax_loss(reduction=reduction)
    elif loss_name == 'MSELoss':
        return build_mse_loss(reduction=reduction)
    else:
        raise ValueError(f"{loss_name} does not support")
