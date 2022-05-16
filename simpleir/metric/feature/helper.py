# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午3:21
@file: helper.py
@author: zj
@description: 
"""

import torch

from .aggregator import do_aggregate
from .enhancer import do_enhance


class FeatureHelper:

    def __init__(self) -> None:
        super().__init__()

    def run(self, feats: torch.Tensor, targets: torch.Tensor,
            aggregate_type='identity', enhance_type='identity'):
        feats = do_aggregate(feats, aggregate_type=aggregate_type)
        # Flatten the eigenvector into a one-dimensional vector
        feats = feats.reshape(feats.shape[0], -1)
        feats = do_enhance(feats, enhance_type=enhance_type)

        return feats
