# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 上午11:55
@file: similarity.py
@author: zj
@description: 
Input query image features and search set features.
Calculate the similarity between the query image and each image in the retrieval set
Returns a list. The first dimension represents target and the second dimension is similarity
"""
from typing import Dict, List

import torch
import torch.nn.functional as F


def euclidean_distance(query_feats: torch.Tensor, gallery_feats: torch.Tensor) -> torch.Tensor:
    """
    Calculate the distance between query set features and gallery set features. Derived from PyRetri

    Args:
        query_feats (torch.tensor): query set features.
        gallery_feats (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the euclidean distance between query set features and gallery set features.
    """
    query_fea = query_feats.transpose(1, 0)
    inner_dot = gallery_feats.mm(query_fea)
    dis = (gallery_feats ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
    dis = dis - 2 * inner_dot
    dis = dis.transpose(1, 0)
    # return dis
    return torch.sqrt(dis)


def cosine_distance(query_feats: torch.Tensor, gallery_feats: torch.Tensor) -> torch.Tensor:
    """
    Calculate the distance between query set features and gallery set features.

    Args:
        query_feats (torch.tensor): query set features.
        gallery_feats (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the cosine distance between query set features and gallery set features.
    """
    similarity_matrix = F.cosine_similarity(query_feats.unsqueeze(1),
                                            gallery_feats.unsqueeze(0), dim=2)

    return 1 - similarity_matrix


def similarity(feat: torch.Tensor, gallery_dict: Dict, similarity_type='euclidean') \
        -> List:
    """
    Calculate similarity (Euclidean distance / Cosine distance)
    """
    assert similarity_type in ['euclidean', 'cosine']
    if len(feat.shape) == 1:
        feat = feat.reshape(1, -1)

    sim_list = list()
    key_list = list()
    value_list = list()

    for idx, (key, values) in enumerate(gallery_dict.items()):
        if len(values) == 0:
            continue

        key_list.extend([key for _ in range(len(values))])
        value_list.extend(values)

    if len(value_list) == 0:
        pass
    else:
        if similarity_type == 'euclidean':
            tmp_sim_array = euclidean_distance(feat, torch.stack(value_list))[0]
        elif similarity_type == 'cosine':
            tmp_sim_array = cosine_distance(feat, torch.stack(value_list))[0]
        else:
            raise ValueError(f'{similarity_type} does not support')

        sim_list = [[key, score.item()] for key, score in zip(key_list, tmp_sim_array)]

    return sim_list
