# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:09
@file: helper.py
@author: zj
@description: 
"""
from typing import List, Tuple

import torch

from .feature.helper import FeatureHelper
from .index.helper import IndexHelper
from .evaluate.helper import EvaluateHelper


class MetricHelper:
    """
    Calculation accuracy. 
    """

    def __init__(self,
                 max_num: int = 5,
                 top_k_list: Tuple = (1, 5),
                 aggregate_type='IDENTITY', enhance_type='IDENTITY',
                 distance_type: str = 'EUCLIDEAN', rank_type='NORMAL', re_rank_type='IDENTITY',
                 train_dir: str = '',
                 eval_type='ACCURACY') -> None:
        super().__init__()
        self.feature = FeatureHelper(aggregate_type=aggregate_type, enhance_type=enhance_type)
        assert len(top_k_list) >= 1
        self.index = IndexHelper(top_k=top_k_list[-1], max_num=max_num,
                                 distance_type=distance_type,
                                 rank_type=rank_type,
                                 re_rank_type=re_rank_type,
                                 train_dir=train_dir)
        self.eval = EvaluateHelper(top_k_list=top_k_list, eval_type=eval_type)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, feats: torch.Tensor, targets: torch.Tensor) -> List:
        new_feats = self.feature.run(feats)
        pred_top_k_list = self.index.run(new_feats, targets)
        res = self.eval.run(pred_top_k_list, targets.numpy())
        return res

    def clear(self) -> None:
        self.index.clear()
