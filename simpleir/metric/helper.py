# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:09
@file: helper.py
@author: zj
@description: 
"""
from typing import List, Tuple

import torch

from .aggregator import do_aggregate
from .enhancer import do_enhance
from .distancer import do_distance
from .rank import do_rank


class MetricHelper:
    """
    Calculation accuracy. Based on similarity measurement and rank
    """

    def __init__(self, max_num: int = 5) -> None:
        super().__init__()

        # Feature set, each category saves N features, first in first out
        self.gallery_dict = dict()
        self.max_num = max_num

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, feats: torch.Tensor, targets: torch.Tensor, top_k_list: Tuple = (1, 5),
            aggregate_type='identity', enhance_type='identity', distance_type: str = 'euclidean',
            rank_type='normal') -> List:
        feats = do_aggregate(feats, aggregate_type=aggregate_type)
        # Flatten the eigenvector into a one-dimensional vector
        feats = feats.reshape(feats.shape[0], -1)
        feats = do_enhance(feats, enhance_type=enhance_type)

        distance_array, candidate_target_list = do_distance(feats, self.gallery_dict, distance_type=distance_type)
        pred_top_k_list = do_rank(distance_array, candidate_target_list,
                                  top_k=top_k_list[-1], rank_type=rank_type)

        top_k_similarity_list = [0 for _ in top_k_list]
        for idx, (feat, target) in enumerate(zip(feats, targets)):
            truth_key = int(target)
            if pred_top_k_list is None:
                pass
            else:
                sorted_list = pred_top_k_list[idx]

                for i, k in enumerate(top_k_list):
                    if truth_key in sorted_list[:k]:
                        top_k_similarity_list[i] += 1

            # Add feat to the atlas every time. If the category is full, the data added at the beginning will pop up
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
