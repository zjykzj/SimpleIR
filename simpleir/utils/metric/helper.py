# -*- coding: utf-8 -*-

"""
@date: 2022/5/25 下午4:31
@file: Retriever.py
@author: zj
@description: 
"""
import os

from enum import Enum

from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['MetricHelper', 'EvaluateType']

from .impl import Accuracy, Precision, Map, MapForOxford


class EvaluateType(Enum):
    ACCURACY = 'ACCURACY'
    PRECISION = 'PRECISION'
    MAP = 'MAP'
    MAP_OXFORD = 'MAP_OXFORD'


class MetricHelper:

    def __init__(self, retrieval_dir, eval_type='ACCURACY', top_k_list=(1, 3, 5, 10), data_root=None):
        self.retrieval_dir = retrieval_dir
        assert os.path.isdir(self.retrieval_dir), self.retrieval_dir

        self.eval_type = EvaluateType[eval_type]

        if self.eval_type == EvaluateType.ACCURACY:
            self.model = Accuracy(retrieval_dir, top_k_list=top_k_list)
        if self.eval_type == EvaluateType.PRECISION:
            self.model = Precision(retrieval_dir, top_k_list=top_k_list)
        if self.eval_type == EvaluateType.MAP:
            self.model = Map(retrieval_dir, top_k_list=top_k_list)
        if self.eval_type == EvaluateType.MAP_OXFORD:
            self.model = MapForOxford(data_root, retrieval_dir)

    def run(self):
        return self.model.run()
