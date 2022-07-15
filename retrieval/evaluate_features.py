# -*- coding: utf-8 -*-

"""
@date: 2022/7/15 上午11:54
@file: evaluate_features.py
@author: zj
@description: Evaluate retrieval results. You should set --retrieval-dir and --eval-type
for example,
1. python evaluate_features.py --retrieval-dir data/retrieval_fc --eval-type ACC
"""

import os
import pickle
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch import Tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval features")
    parser.add_argument('--retrieval-dir', metavar='RETRIEVAL', default=None, type=str,
                        help='Dir for loading retrieval results. Default: None')
    parser.add_argument('--eval-type', metavar='EVAL', default='ACC', type=str,
                        help='Which evaluation method. Default: ACC')

    return parser.parse_args()


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


def compute_acc(rank_tensor, label_tensor, topk=(1, 3, 5, 10)):
    acc_list = accuracy(rank_tensor, label_tensor, topk=topk)

    print()
    for acc, k in zip(acc_list, topk):
        print(f"[{k}] ACC: {acc}")


def main():
    args = parse_args()
    print('args:', args)

    rank_list, label_list = load_retrieval(args.retrieval_dir)

    rank_tensor = torch.from_numpy(np.array(rank_list))
    label_tensor = torch.from_numpy(np.array(label_list))

    if args.eval_type == 'ACC':
        compute_acc(rank_tensor, label_tensor)
    else:
        raise ValueError('ERROR')


if __name__ == '__main__':
    main()
