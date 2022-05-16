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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from enum import Enum


class DistanceType(Enum):
    EUCLIDEAN = 'EUCLIDEAN'
    COSINE = 'COSINE'


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


def do_distance(query_feats: torch.Tensor, gallery_feats: torch.Tensor,
                distance_type: DistanceType = DistanceType.EUCLIDEAN) \
        -> np.ndarray:
    """
    Calculate distance (Euclidean distance / Cosine distance)
    """
    if len(query_feats.shape) == 1:
        query_feats = query_feats.reshape(1, -1)

    if distance_type is DistanceType.EUCLIDEAN:
        distance_array = euclidean_distance(query_feats, gallery_feats).numpy()
    elif distance_type is DistanceType.COSINE:
        distance_array = cosine_distance(query_feats, gallery_feats).numpy()
    else:
        raise ValueError(f'{distance_type} does not support')

    return distance_array
