# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: helper.py
@author: zj
@description: 
"""
from typing import Dict

import torch

from .ranker import do_rank, do_re_rank


class IndexHelper:
    """
    Object index. Including Rank and Re_Rank module
    """

    def __init__(self, top_k: int = 10, distance_type='euclidean',
                 rank_type: str = 'normal', re_rank_type='identity') -> None:
        super().__init__()

        self.distance_type = distance_type
        self.top_k = top_k
        self.rank_type = rank_type
        self.re_rank_type = re_rank_type

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
