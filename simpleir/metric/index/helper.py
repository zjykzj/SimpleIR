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
    object index. Including Rank and Re_Rank module
    """

    def __init__(self) -> None:
        super().__init__()

    def run(self, feats: torch.Tensor, gallery_dict: Dict, distance_type='euclidean',
            top_k: int = 10, rank_type: str = 'normal', re_rank_type='identity'):
        # rank
        pred_top_k_list = do_rank(feats, gallery_dict, distance_type=distance_type, top_k=top_k, rank_type=rank_type)

        # re_rank
        if re_rank_type == 'identity':
            pass
        else:
            pred_top_k_list = do_re_rank(feats, gallery_dict, distance_type=distance_type,
                                         top_k=top_k, rank_type=rank_type, re_rank_type=re_rank_type)

        return pred_top_k_list
