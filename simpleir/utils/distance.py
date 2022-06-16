# -*- coding: utf-8 -*-

"""
@date: 2022/6/16 上午11:24
@file: distance.py
@author: zj
@description: Distance calculation in different ways. For example, the euclidean distance / cosine distance in Pytorch
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """
    refer to [TORCH.CDIST](https://pytorch.org/docs/stable/generated/torch.cdist.html)

    torch.cdist(B×P×M, B×R×M) -> (BxPxR)
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]

    res = torch.cdist(x1, x2, p=2)
    return res


def euclidean_distance_v2(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Calculate the distance between query set features and gallery set features. Derived from PyRetri

    Args:
        x1 (torch.tensor): query set features.
        x2 (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the euclidean distance between query set features and gallery set features.
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]

    x1 = x1.transpose(1, 0)
    inner_dot = x2.mm(x1)
    dis = (x2 ** 2).sum(dim=1, keepdim=True) + (x1 ** 2).sum(dim=0, keepdim=True)
    dis = dis - 2 * inner_dot
    dis = dis.transpose(1, 0)
    # return dis
    return torch.sqrt(dis)


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Calculate the distance between query set features and gallery set features.

    Args:
        x1 (torch.tensor): query set features.
        x2 (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the cosine distance between query set features and gallery set features.
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]
    similarity_matrix = F.cosine_similarity(x1.unsqueeze(1),
                                            x2.unsqueeze(0), dim=2)

    return 1 - similarity_matrix


def cosine_distance_v2(x1: Tensor, x2: Tensor) -> Tensor:
    """
    refer to [COSINESIMILARITY](https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html)
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]
    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-8)

    return 1 - cos(x1.unsqueeze(1), x2.unsqueeze(0))


def cosine_distance_v3(x1, x2=None, eps=1e-8):
    """
    refer to https://github.com/pytorch/pytorch/issues/11202#issue-356530949
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


if __name__ == '__main__':
    torch.manual_seed(0)

    x1 = torch.randn(3, 5)
    print(f"x1({x1.shape}):\n", x1)
    x2 = torch.randn(4, 5)
    print("x2({x2.shape}):\n", x2)

    print('euclidean distance ...')
    res1 = euclidean_distance_v2(x1, x2)
    print(f"res1({res1.shape}):\n", res1)

    res2 = euclidean_distance_v2(x1, x2)
    assert torch.allclose(res1, res2)

    print('cosine distance ...')
    res3 = cosine_distance(x1, x2)
    print(res3)

    res4 = cosine_distance_v2(x1, x2)
    print(res4)
    assert torch.allclose(res3, res4)

    res5 = cosine_distance_v3(x1, x2)
    print(res5)
    assert torch.allclose(res3, res5)
