# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:10
@file: rank.py
@author: zj
@description: 
"""
from typing import Any, List

import numpy as np

from simpleir.utils.count import count_frequency


def normal_rank(distance_array: np.ndarray, candidate_target_list: List, top_k: int = 10) -> List:
    if len(distance_array.shape) == 1:
        distance_array = distance_array.reshape(1, -1)

    # The more smaller distance, the more similar object
    sort_array = np.argsort(distance_array, axis=1)

    pred_top_k_list = list()
    for sort_arr in sort_array:
        tmp_top_list = list(np.array(candidate_target_list)[sort_arr].astype(int))

        # Returns only the sort results needed to calculate the hit rate
        top_list = [-1 for _ in range(top_k)]
        if len(tmp_top_list) < top_k:
            top_list[:len(tmp_top_list)] = tmp_top_list[:]
        else:
            top_list = tmp_top_list[:top_k]

        pred_top_k_list.append(top_list)
    return pred_top_k_list


def knn_rank(distance_array: np.ndarray, candidate_target_list: List, top_k: int = 10) -> List:
    if len(distance_array.shape) == 1:
        distance_array = distance_array.reshape(1, -1)

    # The more smaller distance, the more similar object
    sort_indexes = np.argsort(distance_array, axis=1)

    sqrt_len = int(np.sqrt(len(candidate_target_list)))
    pred_top_k_list = list()
    for idx, sort_idx in enumerate(sort_indexes):
        tmp_top_list = count_frequency(
            list(np.array(candidate_target_list)[sort_idx])[:sqrt_len])

        top_list = [-1 for _ in range(top_k)]
        if len(tmp_top_list) < top_k:
            top_list[:len(tmp_top_list)] = tmp_top_list[:]
        else:
            top_list = tmp_top_list[:top_k]

        pred_top_k_list.append(top_list)
    return pred_top_k_list


def do_rank(distance_array: np.ndarray, candidate_target_list: List,
            top_k: int = 10, rank_type: str = 'normal') -> Any:
    if distance_array is None:
        return None

    if rank_type == 'normal':
        pred_top_k_list = normal_rank(distance_array, candidate_target_list, top_k)
    elif rank_type == 'knn':
        pred_top_k_list = knn_rank(distance_array, candidate_target_list, top_k)
    else:
        raise ValueError(f'{rank_type} does not support')

    return pred_top_k_list