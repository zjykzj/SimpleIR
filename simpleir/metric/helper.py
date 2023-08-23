# -*- coding: utf-8 -*-

"""
@date: 2022/5/25 下午4:31
@file: Retriever.py
@author: zj
@description: 
"""
from enum import Enum

__all__ = ['MetricHelper', 'EvaluateType']

from .impl.accuracy import Accuracy
from .impl.precision import Precision
from .impl.map import MAP


class EvaluateType(Enum):
    ACCURACY = 'ACCURACY'
    PRECISION = 'PRECISION'
    MAP = 'MAP'
    MAP_OXFORD = 'MAP_OXFORD'
    MAP_ROXFORD = 'MAP_ROXFORD'


class MetricHelper:
    """
    度量标准有很多种，但是应该支持两种方式，

    1. 检索结果在本地，那么你需要把数据从本地读取，然后进行评估
    2. 支持全程在线评估，并不需要读取到本地的过程！

    包括特征检索和特征评估，都不应该和本地操作联系在一起，支持直接的数据写入操作。

    集成在一起运行时，可以选择保存到本地进行数据分析。但是应该支持直接导入操作。
    """

    def __init__(self, batch_rank_name_list=None,
                 batch_rank_label_list=None,
                 query_name_list=None,
                 query_label_list=None,
                 eval_type='ACCURACY', top_k_list=(1, 3, 5, 10),
                 data_root=None, dataset='oxford5k'):
        self.eval_type = EvaluateType[eval_type]

        if self.eval_type == EvaluateType.ACCURACY:
            self.model = Accuracy(batch_rank_label_list, query_label_list, top_k_list)
        if self.eval_type == EvaluateType.PRECISION:
            self.model = Precision(batch_rank_label_list, query_label_list, top_k_list)
        if self.eval_type == EvaluateType.MAP:
            self.model = MAP(batch_rank_label_list, query_label_list, top_k_list)
        # if self.eval_type == EvaluateType.MAP_OXFORD:
        #     self.model = MapForOxford(data_root, retrieval_dir)
        # if self.eval_type == EvaluateType.MAP_ROXFORD:
        #     self.model = MapForROxford(data_root, retrieval_dir, top_k_list=top_k_list, dataset=dataset)

    def run(self):
        return self.model.run()
