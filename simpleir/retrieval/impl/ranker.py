# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:10
@file: rank.py
@author: zj
@description: 
"""
from typing import List, Tuple

import torch
from torch import Tensor

import numpy as np
from enum import Enum

from simpleir.utils.sort import argsort


class RankType(Enum):
    NORMAL = 'NORMAL'
    KNN = 'KNN'


def normal_rank(batch_dists_tensor: Tensor, gallery_targets_tensor: Tensor) -> Tuple[List[List], List[List]]:
    """
    Sort according to the distance from small to large, and adjust the corresponding gallery label
    """
    # The more smaller distance, the more similar object
    batch_sort_idx_tensor = argsort(batch_dists_tensor)

    batch_rank_label_list = list()
    for sort_idx_tensor in batch_sort_idx_tensor:
        sorted_label_list = gallery_targets_tensor[sort_idx_tensor].int().tolist()

        batch_rank_label_list.append(sorted_label_list)

    return batch_sort_idx_tensor.tolist(), batch_rank_label_list


def knn_rank(batch_dists_tensor: Tensor, gallery_targets_tensor: Tensor, top_k: int = None) \
        -> Tuple[List[List], List[List]]:
    """
    Execute normal_rank first, then cut the top_k labels and sort them according to the frequency of occurrence
    """
    batch_sort_idx_list, batch_rank_label_list = normal_rank(batch_dists_tensor, gallery_targets_tensor)

    if top_k is None:
        top_k = int(torch.sqrt(gallery_targets_tensor.size(0)))

    knn_batch_sort_idx_list = list()
    knn_batch_rank_label_list = list()
    for sort_idx_list, rank_label_list in zip(batch_sort_idx_list, batch_rank_label_list):
        tmp_idx_list = sort_idx_list[:top_k]
        tmp_label_list = rank_label_list[:top_k]

        count_list = list()
        # for u in set(data_list):
        for u in tmp_label_list:
            count_list.append(tmp_label_list.count(u))

        sorted_indices = list(np.argsort(-1 * np.array(count_list)))

        tmp_sort_idx_list = list(np.array(tmp_idx_list)[sorted_indices])
        tmp_sort_label_list = list(np.array(tmp_label_list)[sorted_indices])

        sort_idx_list[:top_k] = tmp_sort_idx_list
        rank_label_list[:top_k] = tmp_sort_label_list

        knn_batch_sort_idx_list.append(sort_idx_list)
        knn_batch_rank_label_list.append(rank_label_list)

    return knn_batch_sort_idx_list, knn_batch_rank_label_list


def do_rank(batch_dists_tensor: Tensor, gallery_targets_tensor: Tensor,
            rank_type: RankType = RankType.NORMAL, top_k=None) -> Tuple[List[List], List[List]]:
    if len(batch_dists_tensor.shape) == 1:
        batch_dists_tensor = batch_dists_tensor.reshape(1, -1)

    if rank_type is RankType.NORMAL:
        batch_sort_idx_list, batch_rank_label_list = normal_rank(batch_dists_tensor, gallery_targets_tensor)
    elif rank_type is RankType.KNN:
        batch_sort_idx_list, batch_rank_label_list = knn_rank(batch_dists_tensor, gallery_targets_tensor, top_k=top_k)
    else:
        raise ValueError(f'{rank_type} does not support')

    return batch_sort_idx_list, batch_rank_label_list


class Ranker:

    def __init__(self, rank_type: str = 'NORMAL', top_k=None):
        self.rank_type = RankType[rank_type]
        self.top_k = top_k

    def run(self, batch_dists_tensor: Tensor, gallery_targets_tensor: Tensor) \
            -> Tuple[List[List], List[List]]:
        return do_rank(batch_dists_tensor, gallery_targets_tensor, self.rank_type, self.top_k)
