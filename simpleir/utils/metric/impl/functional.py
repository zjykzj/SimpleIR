# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:18
@file: functional.py
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

__all__ = ['load_retrieval', 'accuracy', 'precision']


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
    """Computes the Prec@K for the specified values of k"""
    maxk = max(topk)

    pred = pred[:, :maxk]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].t().float().sum(0).mul_(1.0 / k).mean()
        res.append(correct_k * 100.0)
    return res
