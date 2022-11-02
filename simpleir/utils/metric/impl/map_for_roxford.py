# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:33
@file: map_for_oxford.py
@author: zj
@description: 
"""
from typing import List

import os
import pickle

import numpy as np

from ..metric_base import MetricBase

__all__ = ["MapForROxford"]


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


class MapForROxford(MetricBase):
    """
    Refer to
    https://github.com/filipradenovic/revisitop/blob/master/python/evaluate.py
    https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/evaluate.py
    """

    def __init__(self, data_root: str, retrieval_dir: str, top_k_list=[1, 5, 10], dataset='oxford5k'):
        super().__init__(retrieval_dir, None)

        self.data_root = data_root
        self.retrieval_dir = retrieval_dir
        self.top_k_list = top_k_list
        self.dataset = dataset

    def load_ranks(self):
        info_path = os.path.join(self.retrieval_dir, 'info.pkl')
        with open(info_path, 'rb') as f:
            query_info_dict = pickle.load(f)

        rank_list = []
        for query_name in query_info_dict['content'].keys():
            query_info_path = os.path.join(self.retrieval_dir, f'{query_name}.csv')
            rank_list.append(np.loadtxt(query_info_path, delimiter=",,", dtype=str)[:, 1].astype(int))

        return np.array(rank_list).T

    def load_gnd(self, gnd: List, mode='easy') -> List:
        gnd_t = []
        if mode == 'easy':
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
                gnd_t.append(g)
        elif mode == 'medium':
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk']])
                gnd_t.append(g)
        elif mode == 'hard':
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
                gnd_t.append(g)
        else:
            raise ValueError("ERROR")

        return gnd_t

    def run(self):
        ranks = self.load_ranks()

        gt_file = os.path.join(self.data_root, f"gnd_{self.dataset}.pkl")
        assert os.path.isfile(gt_file), gt_file
        with open(gt_file, 'rb') as f:
            cfg = pickle.load(f)
        gnd = cfg['gnd']

        # old evaluation protocol
        if self.dataset.startswith('oxford5k') or self.dataset.startswith('paris6k'):
            map, aps, _, _ = compute_map(ranks, gnd)
            # print('>> {}: mAP {:.2f}'.format(dataset, np.around(map * 100, decimals=2)))

            return [map, aps]
        else:
            # Easy
            gnd_list = self.load_gnd(gnd, mode='easy')
            assert ranks.shape[1] == len(gnd_list)
            mapE, apsE, mprE, prsE = compute_map(ranks, gnd_list, kappas=self.top_k_list)

            # Medium
            gnd_list = self.load_gnd(gnd, mode='medium')
            assert ranks.shape[1] == len(gnd_list)
            mapM, apsM, mprM, prsM = compute_map(ranks, gnd_list, kappas=self.top_k_list)

            # Hard
            gnd_list = self.load_gnd(gnd, mode='hard')
            assert ranks.shape[1] == len(gnd_list)
            mapH, apsH, mprH, prsH = compute_map(ranks, gnd_list, kappas=self.top_k_list)

            # print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
            #                                               np.around(mapM * 100, decimals=2),
            #                                               np.around(mapH * 100, decimals=2)))
            # print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE * 100, decimals=2),
            #                                                  np.around(mprM * 100, decimals=2),
            #                                                  np.around(mprH * 100, decimals=2)))

            return [
                [mapE, apsE, mprE, prsE],
                [mapM, apsM, mprM, prsM],
                [mapH, apsH, mprH, prsH],
            ]
