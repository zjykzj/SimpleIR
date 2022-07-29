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

from enum import Enum
from torch import Tensor
from simpleir.utils.distance import euclidean_distance, cosine_distance


class DistanceType(Enum):
    EUCLIDEAN = 'EUCLIDEAN'
    COSINE = 'COSINE'


def do_distance(query_feat_tensor: Tensor, gallery_feat_tensor: Tensor,
                distance_type: DistanceType = DistanceType.EUCLIDEAN) -> Tensor:
    """
    Calculate distance (Euclidean distance / Cosine distance)
    """
    if len(query_feat_tensor.shape) == 1:
        query_feat_tensor = query_feat_tensor.reshape(1, -1)

    if distance_type is DistanceType.EUCLIDEAN:
        batch_dists_tensor = euclidean_distance(query_feat_tensor, gallery_feat_tensor)
    elif distance_type is DistanceType.COSINE:
        batch_dists_tensor = cosine_distance(query_feat_tensor, gallery_feat_tensor)
    else:
        raise ValueError(f'{distance_type} does not support')

    return batch_dists_tensor


class Distancer:

    def __init__(self, distance_type: str = 'EUCLIDEAN'):
        self.distance_type = DistanceType[distance_type]

    def run(self, query_feat_tensor: Tensor, gallery_feat_tensor: Tensor) -> Tensor:
        return do_distance(query_feat_tensor, gallery_feat_tensor, self.distance_type)
