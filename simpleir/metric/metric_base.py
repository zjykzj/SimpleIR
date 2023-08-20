# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:17
@file: metric_base.py
@author: zj
@description: 
"""

import os

from .impl.functional import load_retrieval


class MetricBase(object):

    def __init__(self, retrieval_dir, top_k_list=(1, 3, 5, 10)):
        self.retrieval_dir = retrieval_dir
        assert os.path.isdir(self.retrieval_dir), self.retrieval_dir

        self.batch_rank_name_list, self.batch_rank_label_list, self.query_name_list, self.query_label_list = \
            load_retrieval(self.retrieval_dir)
        self.top_k_list = top_k_list

    def run(self):
        pass
