# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 下午7:44
@file: build.py
@author: zj
@description: 
"""

from yacs.config import CfgNode
from argparse import Namespace

from simpleir.configs import get_cfg_defaults
from .helper import MetricHelper

__all__ = ['build_args', 'build_cfg']


def build_args(args: Namespace, top_k_list=(1, 3, 5, 10)):
    cfg = get_cfg_defaults()
    cfg.DATASET.QUERY_DIR = args.query_dir
    cfg.RETRIEVAL.INDEX.RETRIEVAL_DIR = args.retrieval_dir
    cfg.RETRIEVAL.METRIC.EVAL_TYPE = args.retrieval_type
    cfg.RETRIEVAL.METRIC.TOP_K = top_k_list
    cfg.freeze()

    return build_cfg(cfg)


def build_cfg(cfg: CfgNode):
    data_root = cfg.DATASET.QUERY_DIR
    retrieval_dir = cfg.RETRIEVAL.INDEX.RETRIEVAL_DIR
    retrieval_type = cfg.RETRIEVAL.METRIC.EVAL_TYPE
    top_k_list = cfg.RETRIEVAL.METRIC.TOP_K

    metric_helper = MetricHelper(retrieval_dir, eval_type=retrieval_type, top_k_list=top_k_list, data_root=data_root)
    return metric_helper
