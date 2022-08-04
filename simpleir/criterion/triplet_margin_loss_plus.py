# -*- coding: utf-8 -*-

"""
@date: 2022/8/2 下午5:19
@file: triplet_margin_loss.py
@author: zj
@description: See https://github.com/pytorch/vision/blob/main/references/similarity/loss.py

Pytorch adaptation of https://omoindrot.github.io/triplet-loss
https://github.com/omoindrot/tensorflow-triplet-loss
"""
from typing import Dict

import torch.nn as nn
from torch import Tensor

from zcls2.config.key_word import KEY_OUTPUT
from simpleir.configs.key_words import KEY_FEAT

from .triplet_margin_loss import batch_all_triplet_loss, batch_hard_triplet_loss

__all__ = ['triplet_margin_loss_plus', 'TripletMarginLossPlus']


class TripletMarginLossPlus(nn.Module):
    def __init__(self, margin=1.0, p=2.0, mining="batch_all"):
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining = mining

        if mining == "batch_all":
            self.loss_fn = batch_all_triplet_loss
        if mining == "batch_hard":
            self.loss_fn = batch_hard_triplet_loss

        self.loss_cls = nn.CrossEntropyLoss()

    def _forward(self, embeddings, labels):
        return self.loss_fn(labels, embeddings, self.margin, self.p)

    def forward(self, input_dict: Dict, target: Tensor) -> Tensor:
        embeddings = input_dict[KEY_FEAT]
        if len(embeddings.shape) != 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        triplet_loss, fraction_positive_triplets = self._forward(embeddings, target)

        outputs = input_dict[KEY_OUTPUT]
        softmax_loss = self.loss_cls(outputs, target)
        return triplet_loss + softmax_loss


def triplet_margin_loss_plus(margin: float = 1.0, p: float = 2.0, mining: str = "batch_all") -> nn.Module:
    return TripletMarginLossPlus(margin=margin, p=p, mining=mining)
