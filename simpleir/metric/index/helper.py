# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: helper.py
@author: zj
@description: 
"""
from typing import Dict

import torch

from .distancer import DistanceType
from .ranker import do_rank, do_re_rank, RankType, ReRankType


class IndexHelper:
    """
    Object index. Including Rank and Re_Rank module
    """

    def __init__(self, top_k: int = 10, distance_type='EUCLIDEAN',
                 rank_type: str = 'NORMAL', re_rank_type='IDENTITY') -> None:
        super().__init__()

        self.distance_type = DistanceType[distance_type]
        self.top_k = top_k
        self.rank_type = RankType[rank_type]
        self.re_rank_type = ReRankType[re_rank_type]

    def run(self, feats: torch.Tensor, gallery_dict: Dict):
        # rank
        pred_top_k_list = do_rank(feats, gallery_dict,
                                  distance_type=self.distance_type, top_k=self.top_k, rank_type=self.rank_type)

        # re_rank
        if self.re_rank_type == 'identity':
            pass
        else:
            pred_top_k_list = do_re_rank(feats, gallery_dict, distance_type=self.distance_type,
                                         top_k=self.top_k, rank_type=self.rank_type, re_rank_type=self.re_rank_type)

        return pred_top_k_list
