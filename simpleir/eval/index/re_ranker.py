# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午5:11
@file: re_ranker.py
@author: zj
@description: 
"""

from typing import Any, List, Tuple

import torch
from torch import Tensor
import numpy as np
from enum import Enum

from .ranker import RankType, do_rank


class ReRankType(Enum):
    IDENTITY = "IDENTITY"
    QE = 'QE'


def qe_re_rank(query_feats: np.ndarray, gallery_feats: np.ndarray, gallery_targets: list, sort_array: np.ndarray,
               rank_type: RankType = RankType.NORMAL) -> Any:
    sorted_index = np.array(sort_array)[:, :2]
    sorted_index = np.array(sorted_index).reshape(-1)
    requery_feats = gallery_feats[sorted_index].reshape(query_feats.shape[0], -1, query_feats.shape[1]).sum(axis=1)
    requery_feats = requery_feats + query_feats
    query_feats = requery_feats

    from .distancer import cosine_distance
    batch_dists = cosine_distance(torch.from_numpy(query_feats), torch.from_numpy(gallery_feats)).numpy()

    batch_sorts, rank_list = do_rank(batch_dists, torch.tensor(gallery_targets), rank_type=rank_type)
    return batch_sorts, rank_list


def do_re_rank(query_feats: Tensor, gallery_feats: Tensor, gallery_targets: Tensor, batch_sorts: Tensor,
               rank_type: RankType = RankType.NORMAL,
               re_rank_type: ReRankType = ReRankType.IDENTITY) -> Tuple[np.ndarray, List[List]]:
    if len(query_feats.shape) == 1:
        query_feats = query_feats.reshape(1, -1)

    rank_list = list()
    if re_rank_type is ReRankType.IDENTITY:
        pass
    elif re_rank_type is ReRankType.QE:
        batch_sorts, rank_list = qe_re_rank(query_feats.numpy(),
                                            gallery_feats.numpy(),
                                            gallery_targets.tolist(),
                                            batch_sorts.numpy(),
                                            rank_type=rank_type)
    else:
        raise ValueError(f'{re_rank_type} does not support')

    return batch_sorts, rank_list
