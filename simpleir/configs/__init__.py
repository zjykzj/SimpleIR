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
    # Maximum number of each category saved in the gallery
    _C.METRIC.MAX_CATE_NUM = 5
    # feat type
    _C.METRIC.FEAT_TYPE = 'avgpool'
    # aggregate type
    _C.METRIC.AGGREGATE_TYPE = 'IDENTITY'
    # enhance type
    _C.METRIC.ENHANCE_TYPE = 'IDENTITY'
    # distance type
    _C.METRIC.DISTANCE_TYPE = 'EUCLIDEAN'
    # rank type
    _C.METRIC.RANK_TYPE = 'NORMAL'
    # re_rank type
    _C.METRIC.RE_RANK_TYPE = 'IDENTITY'


def get_cfg_defaults() -> CfgNode:
    from zcls2 import config

    cfg = config.get_cfg_defaults()
    add_custom_config(cfg)

    return cfg
