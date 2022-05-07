# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午4:14
@file: mse_loss.py
@author: zj
@description: 
"""
from typing import Dict

from torch import nn, Tensor
from zcls2.config.key_word import KEY_OUTPUT
from simpleir.configs.key_words import KEY_INPUT

__all__ = ['MSELoss', 'mse_loss']


class MSELoss(nn.MSELoss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input_dict: Dict, target: Tensor) -> Tensor:
        inputs = input_dict[KEY_OUTPUT]
        targets = input_dict[KEY_INPUT]

        return super().forward(inputs, targets)


def mse_loss(reduction: str = 'mean') -> nn.Module:
    return MSELoss(reduction=reduction)
