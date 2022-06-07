# -*- coding: utf-8 -*-

"""
@date: 2022/6/7 下午4:17
@file: qe.py
@author: zj
@description: 
"""

from typing import Any

import torch
from torch import Tensor
import numpy as np

from ..ranker import RankType, do_rank
from ..distancer import cosine_distance


def qe_re_rank(query_feats: Tensor, gallery_feats: Tensor, gallery_targets: Tensor, batch_sorts: Tensor,
               rank_type: RankType = RankType.NORMAL) -> Any:
    sorted_index = batch_sorts[:, :2]
    sorted_index = sorted_index.reshape(-1)
    requery_feats = gallery_feats[sorted_index].reshape(query_feats.shape[0], -1, query_feats.shape[1]).sum(axis=1)
    requery_feats = requery_feats + query_feats
    query_feats = requery_feats

    batch_dists = cosine_distance(query_feats, gallery_feats)

    batch_sorts, rank_list = do_rank(batch_dists, gallery_targets, rank_type=rank_type)
    return batch_sorts, rank_list


def qe_re_rank_v2(query_feats: np.ndarray, gallery_feats: np.ndarray, gallery_targets: list, sort_array: np.ndarray,
                  rank_type: RankType = RankType.NORMAL) -> Any:
    sorted_index = np.array(sort_array)[:, :2]
    sorted_index = np.array(sorted_index).reshape(-1)
    requery_feats = gallery_feats[sorted_index].reshape(query_feats.shape[0], -1, query_feats.shape[1]).sum(axis=1)
    requery_feats = requery_feats + query_feats
    query_feats = requery_feats

    batch_dists = cosine_distance(torch.from_numpy(query_feats), torch.from_numpy(gallery_feats))

    batch_sorts, rank_list = do_rank(batch_dists, torch.tensor(gallery_targets), rank_type=rank_type)
    return batch_sorts, rank_list
