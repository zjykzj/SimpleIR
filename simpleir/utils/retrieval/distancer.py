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

import torch

from enum import Enum
from simpleir.utils.distance import euclidean_distance, cosine_distance


class DistanceType(Enum):
    EUCLIDEAN = 'EUCLIDEAN'
    COSINE = 'COSINE'


def do_distance(query_feats: torch.Tensor, gallery_feats: torch.Tensor,
                distance_type: DistanceType = DistanceType.EUCLIDEAN) -> torch.Tensor:
    """
    Calculate distance (Euclidean distance / Cosine distance)
    """
    if len(query_feats.shape) == 1:
        query_feats = query_feats.reshape(1, -1)

    if distance_type is DistanceType.EUCLIDEAN:
        batch_dists = euclidean_distance(query_feats, gallery_feats)
    elif distance_type is DistanceType.COSINE:
        batch_dists = cosine_distance(query_feats, gallery_feats)
    else:
        raise ValueError(f'{distance_type} does not support')

    return batch_dists
