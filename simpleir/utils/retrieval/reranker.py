# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午5:11
@file: re_ranker.py
@author: zj
@description: 
"""

from typing import Any, List, Tuple

from torch import Tensor
import numpy as np
from enum import Enum

from .ranker import RankType
from .rerank.qe import qe_re_rank


class ReRankType(Enum):
    IDENTITY = "IDENTITY"
    QE = 'QE'


def do_rerank(query_feats: Tensor, gallery_feats: Tensor, gallery_targets: Tensor, batch_sorts: Tensor,
              rank_type: RankType = RankType.NORMAL,
              re_rank_type: ReRankType = ReRankType.IDENTITY) -> Tuple[np.ndarray, List[List]]:
    if len(query_feats.shape) == 1:
        query_feats = query_feats.reshape(1, -1)

    rank_list = list()
    if re_rank_type is ReRankType.IDENTITY:
        pass
    elif re_rank_type is ReRankType.QE:
        batch_sorts, rank_list = qe_re_rank(query_feats,
                                            gallery_feats,
                                            gallery_targets,
                                            batch_sorts,
                                            rank_type=rank_type)
    else:
        raise ValueError(f'{re_rank_type} does not support')

    return batch_sorts, rank_list


class ReRanker:

    def __init__(self, re_rank_type='IDENTITY'):
        self.re_rank_type = ReRankType[re_rank_type]

    def run(self):
        pass
