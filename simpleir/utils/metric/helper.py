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


class EvaluateType(Enum):
    ACC = 'ACC'
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


def compute_acc(rank_tensor, label_tensor, topk=(1, 5,)):
    acc_list = accuracy(rank_tensor, label_tensor, topk=topk)

    print()
    for acc, k in zip(acc_list, topk):
        print(f"[{k}] ACC: {acc}%")


class MetricHelper:

    def __init__(self, retrieval_dir, eval_type='ACC', top_k_list=(1, 5,)):
        self.retrieval_dir = retrieval_dir
        assert os.path.isdir(self.retrieval_dir), self.retrieval_dir

        self.eval_type = EvaluateType[eval_type]
        self.top_k_list = top_k_list

    def run(self):
        rank_list, label_list = load_retrieval(self.retrieval_dir)

        rank_tensor = torch.from_numpy(np.array(rank_list))
        label_tensor = torch.from_numpy(np.array(label_list))

        if self.eval_type is EvaluateType.ACC:
            compute_acc(rank_tensor, label_tensor)
        elif self.eval_type is EvaluateType.MAP:
            pass
        else:
            raise ValueError('ERROR')
