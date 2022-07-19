# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午4:32
@file: norm.py
@author: zj
@description: 
"""

import torch

import numpy as np
from sklearn.preprocessing import normalize as sknormalize


def l2_norm(feats: torch.Tensor) -> torch.Tensor:
    if len(feats.shape) == 1:
        feats = feats.reshape(1, -1)

    return feats / torch.norm(feats, dim=1, keepdim=True)


def l2_norm_v2(feats: torch.Tensor) -> torch.Tensor:
    if len(feats.shape) == 1:
        feats = feats.reshape(1, -1)

    return feats / torch.linalg.norm(feats, dim=1, keepdim=True)


def l2_norm_v3(feats: torch.Tensor) -> torch.Tensor:
    if len(feats.shape) == 1:
        feats = feats.reshape(1, -1)

    norm_feats = feats.numpy() / np.linalg.norm(feats.numpy(), axis=1, keepdims=True)
    return torch.from_numpy(norm_feats)


def l2_norm_v4(feats: torch.Tensor) -> torch.Tensor:
    if len(feats.shape) == 1:
        feats = feats.reshape(1, -1)

    norm_feats = sknormalize(feats.numpy(), norm="l2", axis=1, return_norm=False)
    return torch.from_numpy(norm_feats)


if __name__ == '__main__':
    torch.manual_seed(0)

    data = torch.randn(3, 6)
    res = l2_norm(data)
    print(res)

    res2 = l2_norm_v2(data)
    print(res2)
    assert torch.allclose(res, res2)

    res3 = l2_norm_v3(data)
    print(res3)
    assert torch.allclose(res2, res3)

    res4 = l2_norm_v4(data)
    print(res4)
    assert torch.allclose(res3, res4)
