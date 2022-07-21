# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 下午3:19
@file: build.py
@author: zj
@description: 
"""

import os

from yacs.config import CfgNode
from argparse import Namespace

from simpleir.configs import get_cfg_defaults
from .helper import RetrievalHelper

__all__ = ['build_args', 'build_cfg']


def build_args(args: Namespace):
    cfg = get_cfg_defaults()
    cfg.RETRIEVAL.EXTRACT.QUERY_DIR = args.query_dir
    cfg.RETRIEVAL.EXTRACT.GALLERY_DIR = args.gallery_dir

    cfg.RETRIEVAL.INDEX.RETRIEVAL_DIR = args.save_dir
    cfg.RETRIEVAL.INDEX.TOP_K = args.topk

    cfg.RETRIEVAL.INDEX.DISTANCE_TYPE = args.distance
    cfg.RETRIEVAL.INDEX.RANK_TYPE = args.rank
    cfg.RETRIEVAL.INDEX.RERANK_TYPE = args.rerank
    cfg.freeze()

    return build_cfg(cfg)


def build_cfg(cfg: CfgNode):
    query_dir = cfg.RETRIEVAL.EXTRACT.QUERY_DIR
    gallery_dir = cfg.RETRIEVAL.EXTRACT.GALLERY_DIR

    save_dir = cfg.RETRIEVAL.INDEX.RETRIEVAL_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    topk = cfg.RETRIEVAL.INDEX.TOP_K

    distance_type = cfg.RETRIEVAL.INDEX.DISTANCE_TYPE
    rank_type = cfg.RETRIEVAL.INDEX.RANK_TYPE
    rerank_type = cfg.RETRIEVAL.INDEX.RERANK_TYPE

    retrieval_helper = RetrievalHelper(query_dir, gallery_dir, save_dir, topk, distance_type, rank_type, rerank_type)
    return retrieval_helper
