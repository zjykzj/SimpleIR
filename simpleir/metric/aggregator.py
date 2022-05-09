# -*- coding: utf-8 -*-

"""
@date: 2022/5/9 下午2:47
@file: aggregator.py
@author: zj
@description: 
"""
import torch

from .r_mac import get_regions
from .spoc import get_spatial_weight


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


def r_mac(feats: torch.Tensor, level_n: int = 3) -> torch.Tensor:
    """
    Regional Maximum activation of convolutions (R-MAC).

    @param feats (torch.Tensor): conv feats
    @param level_n (int): number of levels for selecting regions.
    """
    assert feats.ndimension() == 4
    h, w = feats.shape[2:]
    final_fea = None
    regions = get_regions(h, w, level_n=level_n)
    for _, r in enumerate(regions):
        st_x, st_y, ed_x, ed_y = r
        region_fea = (feats[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]
        region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)
        if final_fea is None:
            final_fea = region_fea
        else:
            final_fea = final_fea + region_fea

    return final_fea


def spoc(feats: torch.Tensor, use_prior: bool = True) -> torch.Tensor:
    """
    SPoC with center prior.
    """
    assert feats.ndimension() == 4

    if use_prior:
        h, w = feats.shape[2:]
        spatial_weight = get_spatial_weight(h, w)

        feats = feats * spatial_weight
    feats = feats.sum(dim=(2, 3))

    return feats


def do_aggregate(feats: torch.Tensor, aggregate_type='identity') -> torch.Tensor:
    """
    Feature aggregate. Specifically for conv features
    """
    assert aggregate_type in ['identity', 'gap', 'gmp', 'gem', 'r_mac', 'spoc']

    if aggregate_type == 'identity':
        return feats
    elif aggregate_type == 'gap':
        return gap(feats)
    elif aggregate_type == 'gmp':
        return gmp(feats)
    elif aggregate_type == 'gem':
        return gem(feats)
    elif aggregate_type == 'r_mac':
        return r_mac(feats)
    elif aggregate_type == 'spoc':
        return spoc(feats)
    else:
        raise ValueError(f'{aggregate_type} does not support')
