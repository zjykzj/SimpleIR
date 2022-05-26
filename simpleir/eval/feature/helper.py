# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午3:21
@file: helper.py
@author: zj
@description: 
"""

import torch

from .aggregator import do_aggregate, AggregateType
from .enhancer import do_enhance, EnhanceType


class FeatureHelper:
    """
    Feature enhancement
    """

    def __init__(self, aggregate_type='IDENTITY', enhance_type='IDENTITY') -> None:
        super().__init__()

        self.aggregate_type = AggregateType[aggregate_type]
        self.enhance_type = EnhanceType[enhance_type]

    def run(self, feats: torch.Tensor):
        feats = do_aggregate(feats, aggregate_type=self.aggregate_type)

        # Flatten the eigenvector into a one-dimensional vector
        feats = feats.reshape(feats.shape[0], -1)

        feats = do_enhance(feats, enhance_type=self.enhance_type)
        return feats
