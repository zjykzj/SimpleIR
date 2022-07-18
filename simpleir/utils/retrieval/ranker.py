# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:10
@file: rank.py
@author: zj
@description: 
"""
from typing import Any, List, Tuple

from torch import Tensor
from enum import Enum

from simpleir.utils.count import count_frequency_v3
from simpleir.utils.sort import argsort


class RankType(Enum):
    NORMAL = 'NORMAL'
    KNN = 'KNN'


def normal_rank(batch_sorts: Tensor, gallery_targets: Tensor) -> List[List]:
    rank_list = list()
    for sort_arr in batch_sorts:
        sorted_list = gallery_targets[sort_arr].int().tolist()

        rank_list.append(sorted_list)

    return rank_list


def knn_rank(batch_sorts: Tensor, gallery_targets: Tensor) -> List[List]:
    # sqrt_len = int(np.sqrt(len(candidate_target_list)))
    rank_list = list()
    for sort_arr in batch_sorts:
        sorted_list = count_frequency_v3(gallery_targets[sort_arr].int().tolist())

        rank_list.append(sorted_list)

    return rank_list


def do_rank(batch_dists: Tensor, gallery_targets: Tensor,
            rank_type: RankType = RankType.NORMAL) -> Tuple[Tensor, List[List]]:
    if len(batch_dists.shape) == 1:
        batch_dists = batch_dists.reshape(1, -1)

    # The more smaller distance, the more similar object
    batch_sorts = argsort(batch_dists)

    if rank_type is RankType.NORMAL:
        rank_list = normal_rank(batch_sorts, gallery_targets)
    elif rank_type is RankType.KNN:
        rank_list = knn_rank(batch_sorts, gallery_targets)
    else:
        raise ValueError(f'{rank_type} does not support')

    return batch_sorts, rank_list
