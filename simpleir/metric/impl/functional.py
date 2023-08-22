# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:18
@file: functional.py
@author: zj
@description: 
"""
from typing import Tuple, List
import os
import pickle

import numpy as np
from tqdm import tqdm

from torch import Tensor

from zcls2.config.key_word import KEY_SEP

__all__ = ['accuracy', 'precision', 'compute_ap', 'compute_map']


def accuracy(pred: Tensor, target: Tensor, topk=(1,)) -> List[float]:
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


def precision(pred: Tensor, target: Tensor, topk=(1,)) -> List[float]:
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


def compute_ap(rank_label_tensor: Tensor, query_label: Tensor, topk=(1,)) -> List[float]:
    """
    See:

    1. https://blog.zhujian.life/posts/b9ab5b18.html
    2. https://yongyuan.name/blog/evaluation-of-information-retrieval.html
    """
    maxk = max(topk)
    gt_num = (rank_label_tensor == query_label).float().sum().item()
    if gt_num == 0:
        return [0.] * len(topk)

    ap = 0.
    ap_list = list()
    intersect_size = 0.
    for idx, rank_label in enumerate(rank_label_tensor[:maxk]):
        if rank_label == query_label:
            intersect_size += 1
        else:
            continue

        precision = intersect_size * 1.0 / (idx + 1)
        ap += precision

        if (idx + 1) in topk:
            ap_list.append(ap * 100.0 / gt_num)

    return ap_list


def compute_map(batch_rank_label_tensor: Tensor, query_label_tensor: Tensor, topk=(1,)) -> List[float]:
    assert len(batch_rank_label_tensor) == len(query_label_tensor)

    map_list = list()
    for rank_label_tensor, query_label in zip(batch_rank_label_tensor, query_label_tensor):
        ap_list = compute_ap(rank_label_tensor, query_label, topk=topk)
        map_list.append(ap_list)

    return np.array(map_list).mean(0)
