# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午7:11
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode


def add_custom_config(_C: CfgNode) -> None:
    # Directory of query set images
    _C.DATASET.QUERY_DIR = ''
    # Directory of gallery set images
    _C.DATASET.GALLERY_DIR = ''

    # ---------------------------------------------------------------------------- #
    # Retrieval
    # ---------------------------------------------------------------------------- #
    _C.RETRIEVAL = CfgNode()

    _C.RETRIEVAL.EXTRACT = CfgNode()
    # Feat type
    _C.RETRIEVAL.EXTRACT.FEAT_TYPE = 'avgpool'
    # Aggregate type
    _C.RETRIEVAL.EXTRACT.AGGREGATE_TYPE = 'IDENTITY'
    # Enhance type
    _C.RETRIEVAL.EXTRACT.ENHANCE_TYPE = 'IDENTITY'
    # Reduce dimension
    _C.RETRIEVAL.EXTRACT.REDUCE_DIMENSION = 512
    # Directory of query set features
    _C.RETRIEVAL.EXTRACT.QUERY_DIR = ''
    # Directory of gallery set features
    _C.RETRIEVAL.EXTRACT.GALLERY_DIR = ''

    _C.RETRIEVAL.INDEX = CfgNode()
    # Distance type
    _C.RETRIEVAL.INDEX.DISTANCE_TYPE = 'EUCLIDEAN'
    # Rank type
    _C.RETRIEVAL.INDEX.RANK_TYPE = 'NORMAL'
    # ReRank type
    _C.RETRIEVAL.INDEX.RERANK_TYPE = 'IDENTITY'
    # Directory of retrieval results
    _C.RETRIEVAL.INDEX.RETRIEVAL_DIR = ''
    # Save top-N sorting results
    _C.RETRIEVAL.INDEX.TOP_K = None

    _C.RETRIEVAL.METRIC = CfgNode()
    # ReRank type
    _C.RETRIEVAL.METRIC.EVAL_TYPE = 'ACCURACY'
    # Get the Top-k result
    _C.RETRIEVAL.METRIC.TOP_K = (1, 3, 5, 10)


def get_cfg_defaults() -> CfgNode:
    from zcls2 import config

    cfg = config.get_cfg_defaults()
    add_custom_config(cfg)

    return cfg
