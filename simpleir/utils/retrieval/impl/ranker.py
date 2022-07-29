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


def normal_rank(batch_dists_tensor: Tensor, gallery_targets: Tensor) -> Tuple[List[List], List[List]]:
    # The more smaller distance, the more similar object
    batch_sort_idx_tensor = argsort(batch_dists_tensor)

    batch_rank_label_list = list()
    for sort_idx_tensor in batch_sort_idx_tensor:
        sorted_label_list = gallery_targets[sort_idx_tensor].int().tolist()

        batch_rank_label_list.append(sorted_label_list)

    return batch_sort_idx_tensor.tolist(), batch_rank_label_list


def knn_rank(batch_dists_tensor: Tensor, gallery_targets: Tensor) -> Tuple[List[List], List[List]]:
    # The more smaller distance, the more similar object
    batch_sort_idx_tensor = argsort(batch_dists_tensor)

    # sqrt_len = int(np.sqrt(len(candidate_target_list)))
    batch_rank_label_list = list()
    for sort_idx_tensor in batch_sort_idx_tensor:
        rank_label_list = count_frequency_v3(gallery_targets[sort_idx_tensor].int().tolist())

        batch_rank_label_list.append(rank_label_list)

    return batch_sort_idx_tensor.tolist(), batch_rank_label_list


def do_rank(batch_dists_tensor: Tensor, gallery_targets_tensor: Tensor,
            rank_type: RankType = RankType.NORMAL) -> Tuple[List[List], List[List]]:
    if len(batch_dists_tensor.shape) == 1:
        batch_dists_tensor = batch_dists_tensor.reshape(1, -1)

    if rank_type is RankType.NORMAL:
        batch_sort_idx_list, batch_rank_label_list = normal_rank(batch_dists_tensor, gallery_targets_tensor)
    elif rank_type is RankType.KNN:
        batch_sort_idx_list, batch_rank_label_list = knn_rank(batch_dists_tensor, gallery_targets_tensor)
    else:
        raise ValueError(f'{rank_type} does not support')

    return batch_sort_idx_list, batch_rank_label_list


class Ranker:

    def __init__(self, rank_type: str = 'NORMAL'):
        self.rank_type = RankType[rank_type]

    def run(self, batch_dists_tensor: Tensor, gallery_targets_tensor: Tensor) \
            -> Tuple[List[List], List[List]]:
        return do_rank(batch_dists_tensor, gallery_targets_tensor, self.rank_type)
