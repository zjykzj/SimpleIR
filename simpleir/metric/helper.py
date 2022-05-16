# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:09
@file: helper.py
@author: zj
@description: 
"""
from typing import List, Tuple

import torch

from .feature.helper import FeatureHelper
from .index.helper import IndexHelper


class MetricHelper:
    """
    Calculation accuracy. 
    """

    def __init__(self,
                 max_num: int = 5,
                 top_k_list: Tuple = (1, 5),
                 aggregate_type='IDENTITY', enhance_type='IDENTITY',
                 distance_type: str = 'EUCLIDEAN', rank_type='NORMAL', re_rank_type='IDENTITY') -> None:
        super().__init__()
        self.max_num = max_num
        self.top_k_list = top_k_list

        # Feature set, each category saves N features, first in first out
        self.gallery_dict = dict()

        self.feature = FeatureHelper(aggregate_type=aggregate_type, enhance_type=enhance_type)
        assert len(top_k_list) >= 1
        self.index = IndexHelper(top_k=self.top_k_list[-1], distance_type=distance_type,
                                 rank_type=rank_type, re_rank_type=re_rank_type)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, feats: torch.Tensor, targets: torch.Tensor) -> List:
        feats = self.feature.run(feats)
        pred_top_k_list = self.index.run(feats, self.gallery_dict)

        top_k_similarity_list = [0 for _ in self.top_k_list]
        for idx, (feat, target) in enumerate(zip(feats, targets)):
            truth_key = int(target)
            if pred_top_k_list is None:
                pass
            else:
                sorted_list = pred_top_k_list[idx]

                for i, k in enumerate(self.top_k_list):
                    if truth_key in sorted_list[:k]:
                        top_k_similarity_list[i] += 1

            # Add feat to the gallery every time. If the category is full, the data added at the beginning will pop up
            if truth_key not in self.gallery_dict.keys():
                self.gallery_dict[truth_key] = list()
            if len(self.gallery_dict[truth_key]) > self.max_num:
                self.gallery_dict[truth_key].pop(0)
            self.gallery_dict[truth_key].append(feat)

        total_num = len(feats)
        res = []
        for k in top_k_similarity_list:
            res.append(100.0 * k / total_num)
        return res

    def clear(self) -> None:
        self.gallery_dict = dict()
