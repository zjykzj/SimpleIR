# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:17
@file: precision.py
@author: zj
@description: 
"""

import torch

import numpy as np

from .functional import precision

__all__ = ['Precision']


class Precision:

    def __init__(self, batch_rank_label_list, query_label_list, top_k_list=(1, 3, 5, 10)):
        self.batch_rank_label_list = batch_rank_label_list
        self.query_label_list = query_label_list
        self.top_k_list = top_k_list

    def run(self):
        batch_rank_label_tensor = torch.from_numpy(np.array(self.batch_rank_label_list))
        query_label_tensor = torch.from_numpy(np.array(self.query_label_list))
        assert len(query_label_tensor) == len(batch_rank_label_tensor)

        return precision(batch_rank_label_tensor, query_label_tensor, topk=self.top_k_list)
