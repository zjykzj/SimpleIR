# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 上午11:55
@file: similarity.py
@author: zj
@description: 
对于欧式距离或者余弦距离，应该拥有相同的输入
查询图像特征，以及检索集特征

应该拥有相同的输出，查询图像与检索集每张图像之间的相似度
返回一个列表，第一维表示target，第二维是相似度
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def euclidean_distance(query_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    """
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
    """
    if len(query_feats.shape) == 1:
        query_feats = [query_feats]

    return euclidean_distances(query_feats, gallery_feats)
    # return np.linalg.norm(query_feat - gallery_feats, axis=1)


def cosine_distance(query_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    if len(query_feats.shape) == 1:
        query_feats = [query_feats]

    return cosine_distances(query_feats, gallery_feats)


def similarity(feat: np.ndarray, gallery_dict: Dict, similarity_type='euclidean') \
        -> List:
    """
    计算相似度（欧式距离/余弦距离）
    """
    assert similarity_type in ['euclidean', 'cosine']
    feat_array = np.array(feat)

    sim_list = list()
    for idx, (key, values) in enumerate(gallery_dict.items()):
        if len(values) == 0:
            continue

        if similarity_type == 'euclidean':
            tmp_sim_array = euclidean_distance(feat_array, values)[0]
        elif similarity_type == 'cosine':
            tmp_sim_array = cosine_distance(feat_array, values)[0]
        else:
            raise ValueError(f'{similarity_type} does not support')

        sim_list.extend([[key, score] for score in tmp_sim_array])

    return sim_list
