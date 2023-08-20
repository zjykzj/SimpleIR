# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 上午10:04
@file: buld.py
@author: zj
@description:
"""

from .helper import ExtractHelper

__all__ = ['build_extract_helper']


def build_extract_helper(cfg, model, device, data_loader, is_gallery):
    aggregate = cfg.RETRIEVAL.EXTRACT.AGGREGATE_TYPE
    enhance = cfg.RETRIEVAL.EXTRACT.ENHANCE_TYPE
    learn_pca = cfg.RETRIEVAL.EXTRACT.LEARN_PCA
    pca_path = cfg.RETRIEVAL.EXTRACT.PCA_PATH
    rd = cfg.RETRIEVAL.EXTRACT.REDUCE_DIMENSION

    extract_helper = ExtractHelper(model=model, device=device,
                                   aggregate_type=aggregate, enhance_type=enhance,
                                   is_gallery=is_gallery,
                                   learn_pca=learn_pca, pca_path=pca_path, reduce_dimension=rd)
    return extract_helper
