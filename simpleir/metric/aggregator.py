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


def gem(feats: torch.Tensor, p: float = 3.0) -> torch.Tensor:
    """
    Generalized-mean pooling.
    If p = 1, GeM is equal to global average pooling;
    if p = +infinity, GeM is equal to global max pooling.

    @param feats: (N, C, H, W)
    @outputs: (N, C)
    """
    assert feats.ndimension() == 4
    feats = feats ** p
    h, w = feats.shape[2:]
    feats = feats.sum(dim=(2, 3)) * 1.0 / w / h
    feats = feats ** (1.0 / p)

    return feats


def do_aggregate(feats: torch.Tensor, aggregate_type='identity') -> torch.Tensor:
    """
    Feature aggregate. Specifically for conv features
    """
    assert aggregate_type in ['identity', 'gap', 'gmp', 'gem']

    if aggregate_type == 'identity':
        return feats
    elif aggregate_type == 'gap':
        return gap(feats)
    elif aggregate_type == 'gmp':
        return gmp(feats)
    elif aggregate_type == 'gem':
        return gem(feats)
    else:
        raise ValueError(f'{aggregate_type} does not support')
