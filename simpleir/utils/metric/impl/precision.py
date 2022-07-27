# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:17
@file: precision.py
@author: zj
@description: 
"""

import torch

import numpy as np

from ..metric_base import MetricBase
from .functional import precision

__all__ = ['Precision']


class Precision(MetricBase):

    def __init__(self, retrieval_dir, top_k_list=(1, 3, 5, 10)):
        super().__init__(retrieval_dir, top_k_list)

    def run(self):
        super().run()

        batch_rank_label_tensor = torch.from_numpy(np.array(self.batch_rank_label_list))
        query_label_tensor = torch.from_numpy(np.array(self.query_label_list))
        assert len(query_label_tensor) == len(batch_rank_label_tensor)

        return precision(batch_rank_label_tensor, query_label_tensor, topk=self.top_k_list)
