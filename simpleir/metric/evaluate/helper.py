# -*- coding: utf-8 -*-

"""
@date: 2022/5/25 下午4:31
@file: helper.py
@author: zj
@description: 
"""

from typing import Tuple, List

from enum import Enum

import numpy as np


class EvaluateType(Enum):
    ACCURACY = 'ACCURACY'
    MAP = 'MAP'


class EvaluateHelper:
    """
    Feature evaluate
    """

    def __init__(self, top_k_list: Tuple = (1, 5), eval_type='ACCURACY') -> None:
        super().__init__()

        self.top_k_list = top_k_list
        self.eval_type = EvaluateType[eval_type]

    def accuracy(self, pred_top_k_list: List[List], targets: np.ndarray) -> List:
        top_k_similarity_list = [0 for _ in self.top_k_list]
        if pred_top_k_list is None:
            pass
        else:
            assert len(pred_top_k_list) == len(targets)
            for idx, target in enumerate(targets):
                truth_key = int(target)
                sorted_list = pred_top_k_list[idx]

                for i, k in enumerate(self.top_k_list):
                    if truth_key in sorted_list[:k]:
                        top_k_similarity_list[i] += 1

        total_num = len(targets)
        res = []
        for k in top_k_similarity_list:
            res.append(100.0 * k / total_num)
        return res

    def run(self, pred_top_k_list: List[List], targets: np.ndarray) -> List:
        if self.eval_type is EvaluateType.ACCURACY:
            return self.accuracy(pred_top_k_list, targets)
        if self.eval_type is EvaluateType.MAP:
            return list()
