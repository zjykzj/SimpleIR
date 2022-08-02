# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午7:11
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode


def add_custom_config(_C: CfgNode) -> None:
    # ---------------------------------------------------------------------------- #
    # Dataset
    # ---------------------------------------------------------------------------- #
    # Dataset type for gallery / query data
    _C.DATASET.RETRIEVAL_NAME = 'General'
    # Directory of query set images
    _C.DATASET.QUERY_DIR = ''
    # Directory of gallery set images
    _C.DATASET.GALLERY_DIR = ''

    # ---------------------------------------------------------------------------- #
    # Sampler
    # ---------------------------------------------------------------------------- #
    # Sampler type
    _C.SAMPLER = CfgNode()
    _C.SAMPLER.NAME = ''
    # Number of unique labels/classes per batch
    _C.SAMPLER.LABELS_PER_BATCH = 8
    # Number of samples per label in a batch
    _C.SAMPLER.SAMPLES_PER_LABEL = 8

    # ---------------------------------------------------------------------------- #
    # Criterion
    # ---------------------------------------------------------------------------- #
    # Triplet loss margin
    _C.MODEL.CRITERION.MARGIN = 1.0
    # p value for the p-norm distance to calculate between each vector pair
    _C.MODEL.CRITERION.P = 2.0
    # triplet loss mining way, now supports
    # 1. "batch_all"
    # 2. "batch_hard"
    _C.MODEL.CRITERION.MINING = "batch_all"

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
    # Whether to learn PCA / PCA_W in gallery. Default: True
    # Note: If False, PCA path needs to be specified
    _C.RETRIEVAL.EXTRACT.LEARN_PCA = True
    # PCA model path
    _C.RETRIEVAL.EXTRACT.PCA_PATH = None
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
