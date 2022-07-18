# -*- coding: utf-8 -*-

"""
@date: 2022/5/9 下午2:47
@file: aggregator.py
@author: zj
@description: 
"""
import torch
from enum import Enum

from .aggregate.r_mac import get_regions
from .aggregate.spoc import get_spatial_weight


class AggregateType(Enum):
    IDENTITY = "IDENTITY"
    GAP = 'GAP'
    GMP = 'GMP'
    GEM = 'GEM'
    R_MAC = 'R_MAC'
    SPOC = 'SPOC'
    CROW = 'CROW'


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


def crow(feats: torch.Tensor, spatial_a: float = 2.0, spatial_b: float = 2.0) -> torch.Tensor:
    """
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    """
    assert feats.ndimension() == 4

    spatial_weight = feats.sum(dim=1, keepdim=True)
    z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdims=True)
    z = z ** (1.0 / spatial_a)
    spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

    c, w, h = feats.shape[1:]
    nonzeros = (feats != 0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
    channel_weight = torch.log(nonzeros.sum(dim=1, keepdims=True) / nonzeros)

    feats = feats * spatial_weight
    feats = feats.sum(dim=(2, 3))
    feats = feats * channel_weight

    return feats


def do_aggregate(feats: torch.Tensor, aggregate_type: AggregateType = AggregateType.IDENTITY) -> torch.Tensor:
    """
    Feature aggregate. Specifically for conv features
    """
    if aggregate_type is AggregateType.IDENTITY:
        return feats
    elif aggregate_type is AggregateType.GAP:
        return gap(feats)
    elif aggregate_type is AggregateType.GMP:
        return gmp(feats)
    elif aggregate_type is AggregateType.GEM:
        return gem(feats)
    elif aggregate_type is AggregateType.R_MAC:
        return r_mac(feats)
    elif aggregate_type is AggregateType.SPOC:
        return spoc(feats)
    elif aggregate_type is AggregateType.CROW:
        return crow(feats)
    else:
        raise ValueError(f'{aggregate_type} does not support')
