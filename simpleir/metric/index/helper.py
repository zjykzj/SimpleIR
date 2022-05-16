# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: helper.py
@author: zj
@description: 
"""
from typing import Dict

import torch

from .distancer import DistanceType, do_distance
from .ranker import do_rank, RankType
from .re_ranker import do_re_rank, ReRankType


class IndexHelper:
    """
    Object index. Including Rank and Re_Rank module
    """

    def __init__(self, top_k: int = 10, distance_type='EUCLIDEAN',
                 rank_type: str = 'NORMAL', re_rank_type='IDENTITY') -> None:
        super().__init__()
        self.top_k = top_k

        self.distance_type = DistanceType[distance_type]
        self.rank_type = RankType[rank_type]
        self.re_rank_type = ReRankType[re_rank_type]

    def run(self, feats: torch.Tensor, gallery_dict: Dict):
        gallery_key_list = list()
        gallery_value_list = list()

        for idx, (key, values) in enumerate(gallery_dict.items()):
            if len(values) == 0:
                continue

            gallery_key_list.extend([key for _ in range(len(values))])
            gallery_value_list.extend(values)

        pred_top_k_list = None
        if len(gallery_value_list) != 0:
            # distance
            distance_array = do_distance(feats, torch.stack(gallery_value_list), distance_type=self.distance_type)

            # rank
            sort_array, pred_top_k_list = do_rank(distance_array, gallery_key_list, top_k=self.top_k,
                                                  rank_type=self.rank_type)

            # re_rank
            if self.re_rank_type != 'identity':
                sort_array, pred_top_k_list = do_re_rank(feats.numpy(), torch.stack(gallery_value_list).numpy(),
                                                         gallery_key_list, sort_array,
                                                         top_k=self.top_k, rank_type=self.rank_type,
                                                         re_rank_type=self.re_rank_type)

        return pred_top_k_list
