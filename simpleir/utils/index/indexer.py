# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午8:18
@file: helper.py
@author: zj
@description: 
"""

import os
import time

from yacs.config import CfgNode

from tqdm import tqdm
import numpy as np


def calculate_similarity(feat_list_1, feat_list_2):
    # abs_diff
    sum = np.sum(np.abs(feat_list_1 - feat_list_2))
    return sum


def process(query_item, gallery_set):
    metric_list = list()
    for gallery_item in gallery_set:
        sim = calculate_similarity(query_item[1], gallery_item[1])
        metric_list.append([gallery_item[0], sim])

    # 按照相似度进行排序
    indices = np.argsort(np.array(metric_list)[:, 1])
    sorted_list = list(reversed(np.array(metric_list)[indices]))

    is_top1 = 0
    is_top5 = 0
    query_path = query_item[0]
    query_cls = os.path.split(os.path.split(query_path)[0])[1]
    for i in range(5):
        gallery_path = sorted_list[i][0]
        gallery_cls = os.path.split(os.path.split(gallery_path)[0])[1]

        if gallery_cls == query_cls:
            if i == 0:
                is_top1 = 1
            is_top5 = 1

    return is_top1, is_top5


class Indexer:

    def __init__(self, cfg: CfgNode):


        assert len(top_k_list) >= 1
        self.index = IndexHelper(top_k=top_k_list[-1], max_num=max_num,
                                 distance_type=distance_type,
                                 rank_type=rank_type,
                                 re_rank_type=re_rank_type,
                                 gallery_dir=gallery_dir)
        self.metric = MetricHelper(top_k_list=top_k_list, eval_type=eval_type)

    def run(self, similarity='abs_diff'):
        top1 = 0
        top5 = 0

        start = time.time()
        for query_item in tqdm(self.query_set):
            is_top1, is_top5 = process(query_item, self.gallery_set)
            top1 += is_top1
            top5 += is_top5
        end = time.time()
        print('time:', (end - start))

        total_query_num = len(self.query_set)
        print("top1:", 1.0 * top1 / total_query_num)
        print("top5:", 1.0 * top5 / total_query_num)
