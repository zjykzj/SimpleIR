# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午5:11
@file: re_ranker.py
@author: zj
@description: 
"""

from typing import Any, List

import torch
import numpy as np
from enum import Enum

from .ranker import RankType, do_rank


class ReRankType(Enum):
    IDENTITY = "IDENTITY"
    QE = 'QE'


def qe_re_rank(query_feats: np.ndarray, gallery_feats: np.ndarray, gallery_targets: list, sort_array: np.ndarray,
               top_k: int = 10, rank_type: RankType = RankType.NORMAL) -> Any:
    sorted_index = np.array(sort_array)[:, :2]
    sorted_index = np.array(sorted_index).reshape(-1)
    requery_feats = gallery_feats[sorted_index].reshape(query_feats.shape[0], -1, query_feats.shape[1]).sum(axis=1)
    requery_feats = requery_feats + query_feats
    query_feats = requery_feats

    from .distancer import cosine_distance
    distance_array = cosine_distance(torch.from_numpy(query_feats), torch.from_numpy(gallery_feats)).numpy()

    sort_array, pred_top_k_list = do_rank(distance_array, gallery_targets, top_k=top_k, rank_type=rank_type)
    return sort_array, pred_top_k_list


def do_re_rank(query_feats: np.ndarray, gallery_feats: np.ndarray, gallery_targets: list, sort_array: np.ndarray,
               top_k: int = 10, rank_type: RankType = RankType.NORMAL,
               re_rank_type: ReRankType = ReRankType.IDENTITY) -> Any:
    pred_top_k_list = list()
    if re_rank_type is ReRankType.IDENTITY:
        pass
    elif re_rank_type is ReRankType.QE:
        sort_array, pred_top_k_list = qe_re_rank(query_feats, gallery_feats, gallery_targets, sort_array,
                                                 top_k=top_k, rank_type=rank_type)
    else:
        raise ValueError(f'{re_rank_type} does not support')

    return sort_array, pred_top_k_list
