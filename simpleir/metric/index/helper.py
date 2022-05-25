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

from .distancer import DistanceType, do_distance
from .ranker import do_rank, RankType
from .re_ranker import do_re_rank, ReRankType


def load_feats(feat_dir: str) -> Dict:
    assert os.path.isdir(feat_dir), feat_dir

    gallery_dict = dict()

    file_list = glob.glob(os.path.join(feat_dir, 'part_*.pkl'))
    for file_path in file_list:
        with open(file_path, 'rb') as f:
            tmp_dict = pickle.load(f)

            gallery_dict.update(tmp_dict['feats'])

    return gallery_dict


class IndexHelper:
    """
    Object index. Including Rank and Re_Rank module
    """

    def __init__(self, top_k: int = 10, max_num: int = 5, distance_type='EUCLIDEAN',
                 rank_type: str = 'NORMAL', re_rank_type='IDENTITY', train_dir: str = '') -> None:
        super().__init__()
        self.top_k = top_k

        self.distance_type = DistanceType[distance_type]
        self.rank_type = RankType[rank_type]
        self.re_rank_type = ReRankType[re_rank_type]

        self.is_re_rank = re_rank_type != 'IDENTITY'

        # Feature set, each category saves N features, first in first out
        self.gallery_dict = dict()
        self.max_num = max_num

        if train_dir != '':
            self.gallery_dict = load_feats(train_dir)

    def run(self, feats: torch.Tensor, targets: torch.Tensor) -> List[List]:
        # Get gallery set
        gallery_key_list = list()
        gallery_value_list = list()

        for idx, (key, values) in enumerate(self.gallery_dict.items()):
            if len(values) == 0:
                continue

            gallery_key_list.extend([key for _ in range(len(values))])
            gallery_value_list.extend(values)

        # Index
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

        # Update gallery dict
        for idx, (feat, target) in enumerate(zip(feats, targets)):
            truth_key = int(target)

            # Add feat to the gallery every time. If the category is full, the data added at the beginning will pop up
            if truth_key not in self.gallery_dict.keys():
                self.gallery_dict[truth_key] = list()
            if len(self.gallery_dict[truth_key]) > self.max_num:
                self.gallery_dict[truth_key].pop(0)
            self.gallery_dict[truth_key].append(feat)

        return pred_top_k_list

    def clear(self) -> None:
        del self.gallery_dict
        self.gallery_dict = dict()
