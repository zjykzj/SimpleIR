# -*- coding: utf-8 -*-

"""
@date: 2022/6/7 上午11:57
@file: sort.py
@author: zj
@description: 
"""

import torch
from torch import Tensor

import numpy as np


def argsort(data: Tensor) -> Tensor:
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    return torch.argsort(data, dim=1)


def argsort_v2(data: Tensor) -> Tensor:
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    return torch.from_numpy(np.argsort(data.numpy(), axis=1))


if __name__ == '__main__':
    torch.manual_seed(0)

    data = torch.randn(3, 6)
    print(data)

    res = argsort(data)
    print(res)

    res2 = argsort_v2(data)
    print(res)
    assert torch.allclose(res, res2)
