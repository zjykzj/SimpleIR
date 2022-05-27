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
    _C.EVAL = CfgNode()

    _C.EVAL.FEATURE = CfgNode()
    # Feat type
    _C.EVAL.FEATURE.FEAT_TYPE = 'avgpool'
    # Aggregate type
    _C.EVAL.FEATURE.AGGREGATE_TYPE = 'IDENTITY'
    # Enhance type
    _C.EVAL.FEATURE.ENHANCE_TYPE = 'IDENTITY'
    # Feats dir of Query set
    _C.EVAL.FEATURE.QUERY_DIR = ''

    _C.EVAL.INDEX = CfgNode()
    # Distance type
    _C.EVAL.INDEX.DISTANCE_TYPE = 'EUCLIDEAN'
    # Rank type
    _C.EVAL.INDEX.RANK_TYPE = 'NORMAL'
    # Re_rank type
    _C.EVAL.INDEX.RE_RANK_TYPE = 'IDENTITY'
    # Feats dir of gallery set
    _C.EVAL.INDEX.GALLERY_DIR = ''
    # Maximum number of each category saved in the gallery if MAX_CATE_NUM > 0
    _C.EVAL.INDEX.MAX_CATE_NUM = 0
    # Index mode
    # mode = 0: Make query as gallery and Batch update gallery set
    # mode = 1: Make query as gallery and single update gallery set
    # mode = 2: Set gallery set and No update
    # mode = 3: Set gallery set and Batch update gallery set
    # mode = 4: Set gallery set and single update gallery set
    _C.EVAL.INDEX.MODE = 0

    _C.EVAL.METRIC = CfgNode()
    _C.EVAL.METRIC.EVAL_TYPE = 'ACCURACY'


def get_cfg_defaults() -> CfgNode:
    from zcls2 import config

    cfg = config.get_cfg_defaults()
    add_custom_config(cfg)

    return cfg
