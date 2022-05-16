# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:10
@file: rank.py
@author: zj
@description: 
"""
from typing import Any, List, Dict

import torch

import numpy as np

from .distancer import do_distance
from simpleir.utils.count import count_frequency_v3


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
            top_list[:] = tmp_top_list[:top_k]

        pred_top_k_list.append(top_list)
    return pred_top_k_list


def knn_rank(distance_array: np.ndarray, candidate_target_list: List, top_k: int = 10) -> List:
    if len(distance_array.shape) == 1:
        distance_array = distance_array.reshape(1, -1)

    # The more smaller distance, the more similar object
    sort_array = np.argsort(distance_array, axis=1)

    # sqrt_len = int(np.sqrt(len(candidate_target_list)))
    pred_top_k_list = list()
    for sort_arr in sort_array:
        tmp_top_list = count_frequency_v3(
            list(np.array(candidate_target_list)[sort_arr].astype(int))[:top_k])

        top_list = [-1 for _ in range(top_k)]
        if len(tmp_top_list) < top_k:
            top_list[:len(tmp_top_list)] = tmp_top_list[:]
        else:
            top_list[:] = tmp_top_list[:top_k]

        pred_top_k_list.append(top_list)
    return pred_top_k_list


def do_rank(feats: torch.Tensor, gallery_dict: Dict, distance_type='euclidean',
            top_k: int = 10, rank_type: str = 'normal', re_rank_type='identity') -> Any:
    distance_array, candidate_target_list = do_distance(feats, gallery_dict, distance_type=distance_type)
    if distance_array is None:
        return None

    if rank_type == 'normal':
        pred_top_k_list = normal_rank(distance_array, candidate_target_list, top_k)
    elif rank_type == 'knn':
        pred_top_k_list = knn_rank(distance_array, candidate_target_list, top_k)
    else:
        raise ValueError(f'{rank_type} does not support')

    if re_rank_type == 'identity':
        pass
    elif re_rank_type == 'qe':
        # The more smaller distance, the more similar object
        sort_array = np.argsort(distance_array, axis=1)
        gallery_fea = list()
        for key, values in gallery_dict.items():
            if len(values) == 0:
                continue
            gallery_fea.extend(values)
        gallery_fea = torch.stack(gallery_fea).numpy()

        # sorted_index = np.array(sort_array)[:, :10]
        sorted_index = np.array(sort_array)[:, :2]
        # sorted_index = np.array(sort_array)[:, :3]
        # sorted_index = np.array(sort_array)[:, :5]
        sorted_index = np.array(sorted_index).reshape(-1)
        # requery_fea = gallery_fea[sorted_index].reshape(feats.shape[0], -1, feats.shape[1]).sum(dim=1)
        requery_fea = gallery_fea[sorted_index].reshape(feats.shape[0], -1, feats.shape[1]).sum(axis=1)
        requery_fea = requery_fea + feats.numpy()
        query_fea = requery_fea
        # query_fea = requery_fea / (10 + 1)
        # query_fea = requery_fea / (5 + 1)

        # 10
        # 不取平均
        # Prec@1 42.871 Prec@3 61.019 Prec@5 67.274 Prec@10 73.043
        # Prec@1 42.403 Prec@3 57.952 Prec@5 65.750 Prec@10 75.344
        # 取平均
        # Prec@1 42.403 Prec@3 57.952 Prec@5 65.750 Prec@10 75.344

        # 5
        # cosine top=5
        # 不取平均
        # Prec@1 45.414 Prec@3 60.561 Prec@5 68.387 Prec@10 74.783
        # 取平均
        # Prec@1 45.423 Prec@3 60.561 Prec@5 68.396 Prec@10 74.773

        from .distancer import cosine_distance
        distance_array = cosine_distance(torch.from_numpy(query_fea), torch.from_numpy(gallery_fea)).numpy()
        if distance_array is None:
            return None

        if rank_type == 'normal':
            pred_top_k_list = normal_rank(distance_array, candidate_target_list, top_k)
        elif rank_type == 'knn':
            pred_top_k_list = knn_rank(distance_array, candidate_target_list, top_k)
        else:
            raise ValueError(f'{rank_type} does not support')
    else:
        raise ValueError(f'{re_rank_type} does not support')

    return pred_top_k_list
