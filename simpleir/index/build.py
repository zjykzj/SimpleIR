# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午4:15
@file: build.py
@author: zj
@description: 创建检索器
1. 评估标准：欧式距离？余弦距离？
2. 排序标准：按最大值排序
"""
import glob
import os

from tqdm import tqdm
import numpy as np
from zcls.config.key_word import KEY_SEP


def load_feats(feat_root):
    assert os.path.isdir(feat_root), feat_root

    part_list = glob.glob(os.path.join(feat_root, 'part_*.csv'))
    assert len(part_list) > 0

    feat_list = list()
    for part_path in part_list:
        with open(part_path, 'r') as f:
            for line in f:
                tmp_list = line.strip().split(KEY_SEP)

                img_path = tmp_list[0].strip()
                feats = np.array(tmp_list[1:], dtype=float)
                feat_list.append([img_path, feats])

    return feat_list


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


def build_indexer(root_gallery, root_query):
    assert os.path.isdir(root_gallery), root_gallery
    assert os.path.isdir(root_query), root_query

    gallery_set = load_feats(root_gallery)
    query_set = load_feats(root_query)

    top1 = 0
    top5 = 0
    for query_item in tqdm(query_set):
        is_top1, is_top5 = process(query_item, gallery_set)
        top1 += is_top1
        top5 += is_top5

    total_query_num = len(query_set)
    print("top1:", 1.0 * top1 / total_query_num)
    print("top5:", 1.0 * top5 / total_query_num)
