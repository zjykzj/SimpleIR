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
    # Perform accuracy calculation in the training phase
    # Applicable to classification model
    _C.TRAIN.CALCULATE_ACCURACY = True

    _C.METRIC = CfgNode()
    # similarity type
    _C.METRIC.SIMILARITY_TYPE = 'euclidean'
    # sort type
    _C.METRIC.RANK_TYPE = 'normal'


def get_cfg_defaults() -> CfgNode:
    from zcls2 import config

    cfg = config.get_cfg_defaults()
    add_custom_config(cfg)

    return cfg
