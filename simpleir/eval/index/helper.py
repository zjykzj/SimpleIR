# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: helper.py
@author: zj
@description: 
"""
import glob
from typing import Dict, List

import os
import torch
import pickle

from enum import Enum

from .distancer import DistanceType, do_distance
from .ranker import do_rank, RankType
from .re_ranker import do_re_rank, ReRankType


class IndexMode(Enum):
    """
    Index mode
    mode = 0: Make query as gallery and Batch update gallery set
    mode = 1: Make query as gallery and single update gallery set
    mode = 2: Set gallery set and No update
    mode = 3: Set gallery set and Batch update gallery set
    mode = 4: Set gallery set and single update gallery set
    """
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


def load_feats(feat_dir: str, prefix='part_') -> Dict:
    assert os.path.isdir(feat_dir), feat_dir

    gallery_dict = dict()

    file_list = glob.glob(os.path.join(feat_dir, f'{prefix}*.pkl'))
    for file_path in file_list:
        with open(file_path, 'rb') as f:
            tmp_dict = pickle.load(f)

            gallery_dict.update(tmp_dict['feats'])

    return gallery_dict


class IndexHelper:
    """
    Object index. Including Rank and Re_Rank module
    """

    def __init__(self, top_k: int = 10, distance_type='EUCLIDEAN',
                 rank_type: str = 'NORMAL', re_rank_type='IDENTITY',
                 gallery_dir: str = '', max_num: int = 0, index_mode: int = 0) -> None:
        super().__init__()
        self.top_k = top_k

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
            self.gallery_dict = load_feats(self.gallery_dir)

    def get_gallery_set(self):
        gallery_key_list = list()
        gallery_value_list = list()

        for idx, (key, values) in enumerate(self.gallery_dict.items()):
            if len(values) == 0:
                continue

            gallery_key_list.extend([key for _ in range(len(values))])
            gallery_value_list.extend(values)

        return gallery_key_list, gallery_value_list

    def batch_update(self, feats: torch.Tensor, targets: torch.Tensor) -> List:
        gallery_key_list, gallery_value_list = self.get_gallery_set()

        pred_top_k_list = None
        if len(gallery_value_list) != 0:
            # distance
            distance_array = do_distance(feats, torch.stack(gallery_value_list), distance_type=self.distance_type)

            # rank
            sort_array, pred_top_k_list = do_rank(distance_array, gallery_key_list, top_k=self.top_k,
                                                  rank_type=self.rank_type)

            # re_rank
            if self.is_re_rank:
                sort_array, pred_top_k_list = do_re_rank(feats.numpy(), torch.stack(gallery_value_list).numpy(),
                                                         gallery_key_list, sort_array,
                                                         top_k=self.top_k, rank_type=self.rank_type,
                                                         re_rank_type=self.re_rank_type)

        # update
        if self.index_mode is IndexMode.ZERO or self.index_mode is IndexMode.THREE:
            # Update gallery dict
            for idx, (feat, target) in enumerate(zip(feats, targets)):
                truth_key = int(target)

                # Add feat to the gallery every time.
                if truth_key not in self.gallery_dict.keys():
                    self.gallery_dict[truth_key] = list()
                if self.max_num > 0 and len(self.gallery_dict[truth_key]) > self.max_num:
                    # If the category is full, the data added at the beginning will pop up
                    self.gallery_dict[truth_key].pop(0)
                self.gallery_dict[truth_key].append(feat)
        else:
            assert self.index_mode is IndexMode.TWO
            pass

        return pred_top_k_list

    def single_update(self, feats: torch.Tensor, targets: torch.Tensor) -> List:
        assert self.index_mode is IndexMode.ONE or self.index_mode is IndexMode.FOUR

        # Update gallery dict
        pred_top_k_list = list()
        for idx, (feat, target) in enumerate(zip(feats, targets)):
            gallery_key_list, gallery_value_list = self.get_gallery_set()

            if len(gallery_value_list) != 0:
                # distance
                distance_array = do_distance(feat, torch.stack(gallery_value_list), distance_type=self.distance_type)

                # rank
                sort_array, tmp_pred_top_k_list = do_rank(distance_array, gallery_key_list, top_k=self.top_k,
                                                          rank_type=self.rank_type)

                # re_rank
                if self.is_re_rank:
                    sort_array, tmp_pred_top_k_list = do_re_rank(feat.numpy(), torch.stack(gallery_value_list).numpy(),
                                                                 gallery_key_list, sort_array,
                                                                 top_k=self.top_k, rank_type=self.rank_type,
                                                                 re_rank_type=self.re_rank_type)
                pred_top_k_list.append(tmp_pred_top_k_list[0])
            else:
                pred_top_k_list.append([-1 for _ in range(self.top_k)])

            truth_key = int(target)
            # Add feat to the gallery every time.
            if truth_key not in self.gallery_dict.keys():
                self.gallery_dict[truth_key] = list()
            if self.max_num > 0 and len(self.gallery_dict[truth_key]) > self.max_num:
                # If the category is full, the data added at the beginning will pop up
                self.gallery_dict[truth_key].pop(0)
            self.gallery_dict[truth_key].append(feat)

        return pred_top_k_list

    def run(self, feats: torch.Tensor, targets: torch.Tensor) -> List[List]:
        if self.index_mode is IndexMode.ZERO:
            pred_top_k_list = self.batch_update(feats, targets)
        elif self.index_mode is IndexMode.ONE:
            pred_top_k_list = self.single_update(feats, targets)
        elif self.index_mode is IndexMode.TWO:
            pred_top_k_list = self.batch_update(feats, targets)
        elif self.index_mode is IndexMode.THREE:
            pred_top_k_list = self.batch_update(feats, targets)
        else:
            assert self.index_mode is IndexMode.FOUR
            pred_top_k_list = self.single_update(feats, targets)

        return pred_top_k_list

    def clear(self) -> None:
        del self.gallery_dict
        self.gallery_dict = dict()
