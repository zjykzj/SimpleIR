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
from .metric.helper import MetricHelper


class EvalHelper:
    """
    Calculation accuracy. 
    """

    def __init__(self,
                 top_k_list: Tuple = (1, 5),
                 aggregate_type='IDENTITY', enhance_type='IDENTITY',
                 distance_type: str = 'EUCLIDEAN', rank_type='NORMAL', re_rank_type='IDENTITY',
                 gallery_dir: str = '', max_num: int = 0, index_mode: int = 0,
                 eval_type='ACCURACY') -> None:
        super().__init__()
        self.feature = FeatureHelper(aggregate_type=aggregate_type, enhance_type=enhance_type)
        assert len(top_k_list) >= 1
        self.index = IndexHelper(distance_type=distance_type,
                                 rank_type=rank_type,
                                 re_rank_type=re_rank_type,
                                 gallery_dir=gallery_dir,
                                 max_num=max_num,
                                 index_mode=index_mode
                                 )
        self.metric = MetricHelper(top_k_list=top_k_list, eval_type=eval_type)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, query_feats: torch.Tensor, query_targets: torch.Tensor) -> List:
        new_query_feats = self.feature.run(query_feats)
        rank_list, gallery_dict = self.index.run(new_query_feats, query_targets)
        res = self.metric.run(rank_list, gallery_dict, query_targets.numpy())
        return res

    def init(self) -> None:
        self.index.init()

    def clear(self) -> None:
        self.index.clear()
