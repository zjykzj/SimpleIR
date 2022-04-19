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

import numpy as np
from zcls.config.key_word import KEY_SEP

from .helper import IndexHelper


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


def build_indexer(root_gallery, root_query):
    assert os.path.isdir(root_gallery), root_gallery
    assert os.path.isdir(root_query), root_query

    gallery_set = load_feats(root_gallery)
    query_set = load_feats(root_query)

    return IndexHelper(gallery_set, query_set)
