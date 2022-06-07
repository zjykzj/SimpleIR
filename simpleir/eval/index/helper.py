# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: helper.py
@author: zj
@description: 
"""
from typing import Dict, List, Tuple, Union, Any

import torch
from torch import Tensor

from enum import Enum
from zcls2.util import logging

logger = logging.get_logger(__name__)

from .distancer import DistanceType, do_distance
from .ranker import do_rank, RankType
from .re_ranker import do_re_rank, ReRankType
from simpleir.utils.util import load_feats


class IndexMode(Enum):
    """
    Index mode
    mode = 0: Make query as gallery and batch update gallery set
    mode = 1: Make query as gallery and single update gallery set
    mode = 2: Set gallery set and no update
    mode = 3: Set gallery set and batch update gallery set
    mode = 4: Set gallery set and single update gallery set
    """
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class IndexHelper:
    """
    Object retrieval. Including Rank and Re_Rank module
    """

    def __init__(self, distance_type='EUCLIDEAN',
                 rank_type: str = 'NORMAL', re_rank_type='IDENTITY',
                 gallery_dir: str = '', max_num: int = 0, index_mode: int = 0) -> None:
        super().__init__()

        self.distance_type = DistanceType[distance_type]
        self.rank_type = RankType[rank_type]
        self.re_rank_type = ReRankType[re_rank_type]

        self.is_re_rank = re_rank_type != 'IDENTITY'

        # Feature set, each category saves N features, first in first out
        self.gallery_dict = dict()
        self.max_num = max_num
        self.gallery_dir = gallery_dir

        self.index_mode = IndexMode(index_mode)

    def init(self):
        if self.index_mode in [
            IndexMode.TWO,
            IndexMode.THREE,
            IndexMode.FOUR
        ] and self.gallery_dir != '':
            logger.info(f"Loaded feats from {self.gallery_dir}")
            self.gallery_dict = load_feats(self.gallery_dir)

            if self.max_num > 0:
                for key in self.gallery_dict.keys():
                    if len(self.gallery_dict[key]) > self.max_num:
                        self.gallery_dict[key] = self.gallery_dict[key][:self.max_num]

    def get_gallery_set(self) -> Tuple[Tensor, Tensor]:
        gallery_key_list = list()
        gallery_value_list = list()

        for idx, (key, values) in enumerate(self.gallery_dict.items()):
            if len(values) == 0:
                continue

            gallery_key_list.extend([key for _ in range(len(values))])
            gallery_value_list.extend(values)

        gallery_targets = torch.tensor(gallery_key_list).int() if len(gallery_key_list) > 0 else list()
        gallery_feats = torch.stack(gallery_value_list).float() if len(gallery_key_list) > 0 else list()
        return gallery_targets, gallery_feats

    def update_gallery_set(self, feat: torch.Tensor, target: torch.Tensor) -> None:
        truth_key = int(target)
        # Add feat to the gallery every time.
        if truth_key not in self.gallery_dict.keys():
            self.gallery_dict[truth_key] = list()

        self.gallery_dict[truth_key].append(feat)

        if 0 < self.max_num < len(self.gallery_dict[truth_key]):
            # If the category is full, the data added at the beginning will pop up
            self.gallery_dict[truth_key].pop(0)

    def batch_update(self, query_feats: torch.Tensor, query_targets: torch.Tensor) -> Tuple[List[List], Dict]:
        gallery_targets, gallery_feats = self.get_gallery_set()

        rank_list = None
        if len(gallery_targets) != 0:
            # distance
            batch_dists = do_distance(query_feats, gallery_feats, distance_type=self.distance_type)

            # rank
            batch_sorts, rank_list = do_rank(batch_dists, gallery_targets, rank_type=self.rank_type)

            # re_rank
            if self.is_re_rank:
                _, rank_list = do_re_rank(query_feats, gallery_feats, gallery_targets, batch_sorts,
                                          rank_type=self.rank_type, re_rank_type=self.re_rank_type)

        # update
        if self.index_mode is IndexMode.ZERO or self.index_mode is IndexMode.THREE:
            # Update gallery dict
            for idx, (feat, target) in enumerate(zip(query_feats, query_targets)):
                self.update_gallery_set(feat, target)
        else:
            assert self.index_mode is IndexMode.TWO
            pass

        return rank_list, self.gallery_dict

    def single_update(self, query_feats: torch.Tensor, query_targets: torch.Tensor) -> Tuple[List[List], Dict]:
        assert self.index_mode is IndexMode.ONE or self.index_mode is IndexMode.FOUR

        # Update gallery dict
        rank_list = list()
        for idx, (feat, target) in enumerate(zip(query_feats, query_targets)):
            gallery_targets, gallery_feats = self.get_gallery_set()

            if len(gallery_feats) != 0:
                # distance
                batch_dists = do_distance(feat, gallery_feats, distance_type=self.distance_type)

                # rank
                batch_sorts, tmp_rank_list = do_rank(batch_dists, gallery_targets, rank_type=self.rank_type)

                # re_rank
                if self.is_re_rank:
                    _, tmp_rank_list = do_re_rank(feat, gallery_feats, gallery_targets, batch_sorts,
                                                  rank_type=self.rank_type, re_rank_type=self.re_rank_type)

                rank_list.append(tmp_rank_list[0])
            else:
                rank_list.append([-1 for _ in range(gallery_targets)])

            self.update_gallery_set(feat, target)

        return rank_list, self.gallery_dict

    def run(self, query_feats: torch.Tensor, query_targets: torch.Tensor) -> Tuple[Any, Any]:
        if self.index_mode is IndexMode.ZERO:
            return self.batch_update(query_feats, query_targets)
        if self.index_mode is IndexMode.ONE:
            return self.single_update(query_feats, query_targets)
        if self.index_mode is IndexMode.TWO:
            return self.batch_update(query_feats, query_targets)
        if self.index_mode is IndexMode.THREE:
            return self.batch_update(query_feats, query_targets)
        if self.index_mode is IndexMode.FOUR:
            return self.single_update(query_feats, query_targets)
        return None, None

    def clear(self) -> None:
        del self.gallery_dict
        self.gallery_dict = dict()
