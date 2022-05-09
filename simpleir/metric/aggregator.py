# -*- coding: utf-8 -*-

"""
@date: 2022/5/9 下午2:47
@file: aggregator.py
@author: zj
@description: 
"""
import torch


def gap(feats: torch.Tensor) -> torch.Tensor:
    """
    Global average pooling.

    @param feats: (N, C, H, W)
    @outputs: (N, C)
    """
    assert feats.ndimension() == 4
    feats = feats.mean(dim=3).mean(dim=2)

    return feats


def gmp(feats: torch.Tensor) -> torch.Tensor:
    """
    Global maximum pooling.

    @param feats: (N, C, H, W)
    @outputs: (N, C)
    """
    assert feats.ndimension() == 4
    feats = (feats.max(dim=3)[0]).max(dim=2)[0]

    return feats


def do_aggregate(feats: torch.Tensor, aggregate_type='identity') -> torch.Tensor:
    """
    Calculate similarity (Euclidean distance / Cosine distance)
    """
    assert aggregate_type in ['identity', 'gap', 'gmp']

    if aggregate_type == 'identity':
        return feats
    elif aggregate_type == 'gap':
        return gap(feats)
    elif aggregate_type == 'gmp':
        return gmp(feats)
    else:
        raise ValueError(f'{aggregate_type} does not support')
