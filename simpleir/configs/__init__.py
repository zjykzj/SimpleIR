# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午7:11
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode


def add_custom_config(_C: CfgNode) -> None:
    # Add your own customized configs.
    _C.METRIC = CfgNode()

    _C.METRIC.FEATURE = CfgNode()
    # Feat type
    _C.METRIC.FEATURE.FEAT_TYPE = 'avgpool'
    # Aggregate type
    _C.METRIC.FEATURE.AGGREGATE_TYPE = 'IDENTITY'
    # Enhance type
    _C.METRIC.FEATURE.ENHANCE_TYPE = 'IDENTITY'

    _C.METRIC.INDEX = CfgNode()
    # Maximum number of each category saved in the gallery
    _C.METRIC.INDEX.MAX_CATE_NUM = 5
    # Distance type
    _C.METRIC.INDEX.DISTANCE_TYPE = 'EUCLIDEAN'
    # Rank type
    _C.METRIC.INDEX.RANK_TYPE = 'NORMAL'
    # Re_rank type
    _C.METRIC.INDEX.RE_RANK_TYPE = 'IDENTITY'
    # Pretrained feats
    _C.METRIC.INDEX.TRAIN_DIR = ''


def get_cfg_defaults() -> CfgNode:
    from zcls2 import config

    cfg = config.get_cfg_defaults()
    add_custom_config(cfg)

    return cfg
