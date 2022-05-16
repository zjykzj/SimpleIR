# -*- coding: utf-8 -*-

"""
@date: 2022/5/9 下午3:48
@file: spoc.py
@author: zj
@description: 
"""

import torch

spatial_weight_cache = dict()


def get_spatial_weight(h, w):
    """
    Spatial weight with center prior.
    """
    if (h, w) in spatial_weight_cache:
        spatial_weight = spatial_weight_cache[(h, w)]
    else:
        sigma = min(h, w) / 2.0 / 3.0
        x = torch.Tensor(range(w))
        y = torch.Tensor(range(h))[:, None]
        spatial_weight = torch.exp(-((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2) / 2.0 / (sigma ** 2))

        spatial_weight = spatial_weight[None, None, :, :]
        spatial_weight_cache[(h, w)] = spatial_weight

    return spatial_weight
