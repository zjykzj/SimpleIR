# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:10
@file: rank.py
@author: zj
@description: 
"""
from typing import Any, List

import torch
import numpy as np
from enum import Enum

from simpleir.utils.count import count_frequency_v3


class RankType(Enum):
    NORMAL = 'NORMAL'
    KNN = 'KNN'


def normal_rank(sort_array: np.ndarray, gallery_targets: List, top_k: int = 10) -> List:
    pred_top_k_list = list()
    for sort_arr in sort_array:
        tmp_top_list = list(np.array(gallery_targets)[sort_arr].astype(int))

        # Returns only the sort results needed to calculate the hit rate
        top_list = [-1 for _ in range(top_k)]
        if len(tmp_top_list) < top_k:
            top_list[:len(tmp_top_list)] = tmp_top_list[:]
        else:
            top_list[:] = tmp_top_list[:top_k]

        pred_top_k_list.append(top_list)
    return pred_top_k_list


def knn_rank(sort_array: np.ndarray, gallery_targets: List, top_k: int = 10) -> List:
    # sqrt_len = int(np.sqrt(len(candidate_target_list)))
    pred_top_k_list = list()
    for sort_arr in sort_array:
        tmp_top_list = count_frequency_v3(
            list(np.array(gallery_targets)[sort_arr].astype(int))[:top_k])

        top_list = [-1 for _ in range(top_k)]
        if len(tmp_top_list) < top_k:
            top_list[:len(tmp_top_list)] = tmp_top_list[:]
        else:
            top_list[:] = tmp_top_list[:top_k]

        pred_top_k_list.append(top_list)
    return pred_top_k_list


def do_rank(distance_array: np.ndarray, gallery_targets: list,
            top_k: int = 10, rank_type: RankType = RankType.NORMAL) -> Any:
    if len(distance_array.shape) == 1:
        distance_array = distance_array.reshape(1, -1)

    # The more smaller distance, the more similar object
    sort_array = np.argsort(distance_array, axis=1)

    if rank_type is RankType.NORMAL:
        pred_top_k_list = normal_rank(sort_array, gallery_targets, top_k)
    elif rank_type is RankType.KNN:
        pred_top_k_list = knn_rank(sort_array, gallery_targets, top_k)
    else:
        raise ValueError(f'{rank_type} does not support')

    return sort_array, pred_top_k_list
