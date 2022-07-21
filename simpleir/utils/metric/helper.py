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

from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['MetricHelper', 'EvaluateType']


class EvaluateType(Enum):
    ACCURACY = 'ACCURACY'
    MAP = 'MAP'


def load_retrieval(retrieval_dir):
    assert os.path.isdir(retrieval_dir), retrieval_dir

    info_path = os.path.join(retrieval_dir, 'info.pkl')
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    batch_rank_list = list()
    label_list = list()
    for idx, (img_name, label) in tqdm(enumerate(info_dict['content'].items())):
        rank_path = os.path.join(retrieval_dir, f'{img_name}.csv')
        rank_list = np.loadtxt(rank_path, dtype=int, delimiter=' ')

        batch_rank_list.append(rank_list)
        label_list.append(label)

    return batch_rank_list, label_list


def accuracy(pred: Tensor, target: Tensor, topk=(1,)) -> list:
    """Computes the precision@k for the specified values of k"""
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


class MetricHelper:

    def __init__(self, retrieval_dir, eval_type='ACCURACY', top_k_list=(1, 3, 5, 10)):
        self.retrieval_dir = retrieval_dir
        assert os.path.isdir(self.retrieval_dir), self.retrieval_dir

        self.eval_type = EvaluateType[eval_type]
        self.top_k_list = top_k_list

    def run(self):
        rank_list, label_list = load_retrieval(self.retrieval_dir)

        rank_tensor = torch.from_numpy(np.array(rank_list))
        label_tensor = torch.from_numpy(np.array(label_list))

        if self.eval_type is EvaluateType.ACCURACY:
            return accuracy(rank_tensor, label_tensor, topk=self.top_k_list)
        elif self.eval_type is EvaluateType.MAP:
            pass
        else:
            raise ValueError('ERROR')
