# -*- coding: utf-8 -*-

"""
@date: 2022/5/25 下午4:31
@file: Retriever.py
@author: zj
@description: 
"""
import os
import pickle

from enum import Enum
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor

from zcls2.config.key_word import KEY_SEP
from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['MetricHelper', 'EvaluateType']


class EvaluateType(Enum):
    ACCURACY = 'ACCURACY'
    PRECISION = 'PRECISION'
    MAP = 'MAP'


def load_retrieval(retrieval_dir):
    assert os.path.isdir(retrieval_dir), retrieval_dir

    info_path = os.path.join(retrieval_dir, 'info.pkl')
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    batch_rank_label_list = list()
    batch_rank_name_list = list()
    query_label_list = list()
    for idx, (img_name, label) in enumerate(tqdm(info_dict['content'].items())):
        rank_path = os.path.join(retrieval_dir, f'{img_name}.csv')
        rank_array = np.loadtxt(rank_path, dtype=np.str, delimiter=KEY_SEP)

        batch_rank_name_list.append(list(rank_array[:, 0].astype(str)))
        batch_rank_label_list.append(list(rank_array[:, 1].astype(int)))
        query_label_list.append(label)

    return batch_rank_name_list, batch_rank_label_list, query_label_list


def accuracy(pred: Tensor, target: Tensor, topk=(1,)) -> list:
    """Computes the ACC@K for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred = pred[:, :maxk]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        tmp_k = correct[:k].t()

        correct_k = 0.
        for tmp in tmp_k:
            if tmp.float().sum(0) >= 1:
                correct_k += 1
        res.append(correct_k * (100.0 / batch_size))
    return res


def precision(pred: Tensor, target: Tensor, topk=(1,)) -> list:
    """Computes the Pre@K for the specified values of k"""
    maxk = max(topk)

    pred = pred[:, :maxk]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum(0).mul_(1.0 / k).mean()
        res.append(correct_k * 100.0)
    return res


class MetricHelper:

    def __init__(self, retrieval_dir, eval_type='ACCURACY', top_k_list=(1, 3, 5, 10)):
        self.retrieval_dir = retrieval_dir
        assert os.path.isdir(self.retrieval_dir), self.retrieval_dir

        self.eval_type = EvaluateType[eval_type]
        self.top_k_list = top_k_list

    def run(self):
        rank_name_list, batch_rank_label_list, query_label_list = load_retrieval(self.retrieval_dir)

        batch_rank_label_tensor = torch.from_numpy(np.array(batch_rank_label_list))
        query_label_tensor = torch.from_numpy(np.array(query_label_list))
        assert len(query_label_tensor) == len(batch_rank_label_tensor)

        if self.eval_type is EvaluateType.ACCURACY:
            return accuracy(batch_rank_label_tensor, query_label_tensor, topk=self.top_k_list)
        elif self.eval_type is EvaluateType.PRECISION:
            return precision(batch_rank_label_tensor, query_label_tensor, topk=self.top_k_list)
        elif self.eval_type is EvaluateType.MAP:
            pass
        else:
            raise ValueError('ERROR')
