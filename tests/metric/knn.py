# -*- coding: utf-8 -*-

"""
@date: 2022/5/9 上午9:52
@file: knn.py
@author: zj
@description: 
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def cal_dis(query_fea: torch.tensor, gallery_fea: torch.tensor) -> torch.tensor:
    """
    Calculate the distance between query set features and gallery set features.

    Args:
        query_fea (torch.tensor): query set features.
        gallery_fea (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the distance between query set features and gallery set features.
    """
    query_fea = query_fea.transpose(1, 0)
    inner_dot = gallery_fea.mm(query_fea)
    dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
    dis = dis - 2 * inner_dot
    dis = dis.transpose(1, 0)
    # return dis
    return torch.sqrt(dis)


def euclidean_distance(query_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    """
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
    """
    if len(query_feats.shape) == 1:
        query_feats = [query_feats]

    return euclidean_distances(query_feats, gallery_feats)
    # return np.linalg.norm(query_feat - gallery_feats, axis=1)


if __name__ == '__main__':
    torch.set_printoptions(7)
    # query_fea = torch.randn(256, 1000)
    # gallery_fea = torch.randn(502 * 20, 1000)
    #
    # import time
    #
    # t1 = time.time()
    # res = cal_dis(query_fea, gallery_fea)
    # print(res, time.time() - t1)
    #
    # t1 = time.time()
    # sorted_res = torch.argsort(res, dim=1)
    # print(sorted_res, time.time() - t1)
    #
    # t2 = time.time()
    # res2 = euclidean_distance(query_fea.numpy(), gallery_fea.numpy())
    # print(res2, time.time() - t2)
    #
    # t2 = time.time()
    # sorted_res2 = np.argsort(res2, axis=1)
    # print(sorted_res2, time.time() - t2)

    import torch
    import torch.nn.functional as F

    features_a = torch.rand((4, 64))
    features_b = torch.rand((5, 64))

    similarity_matrix = F.cosine_similarity(features_a.unsqueeze(1),
                                            features_b.unsqueeze(0), dim=2)

    print(similarity_matrix.shape)
    print(similarity_matrix)

    print(1- similarity_matrix)

    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity

    print(1 - cosine_distances(features_a.numpy(), features_b.numpy()))
