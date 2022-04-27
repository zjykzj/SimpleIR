# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:09
@file: helper.py
@author: zj
@description: 
"""
from typing import List, Tuple

import numpy as np

from .similarity import similarity
from .rank import rank


class MetricHelper:
    """
    计算准确率，基于相似度度量方式以及排序规则

    输入度量方式以及排序方式
    """

    def __init__(self, max_num: int = 20) -> None:
        super().__init__()

        # 特征集，每个类别保存20条特征，先进先出
        self.gallery_dict = dict()
        self.max_num = max_num

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, feats: np.ndarray, targets: np.ndarray, top_k_list: Tuple = (1, 5),
            similarity_type: str = 'euclidean', rank_type='normal') -> List:
        top_k_similarity_list = [0 for _ in top_k_list]
        for feat, target in zip(feats, targets):
            # 将特征向量拉平为一维向量
            feat = feat.reshape(-1)

            truth_key = int(target)
            similarity_list = similarity(feat, self.gallery_dict, similarity_type=similarity_type)
            if len(similarity_list) == 0:
                pass
            else:
                sorted_list = rank(similarity_list, top_k=top_k_list[-1], rank_type=rank_type)

                for i, k in enumerate(top_k_list):
                    if truth_key in sorted_list[:k]:
                        top_k_similarity_list[i] += 1

            # 每次都将feat加入图集，如果该类别保存已满，那么弹出最开始加入的数据
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
