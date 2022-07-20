# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 下午3:19
@file: build.py
@author: zj
@description: 
"""

import os
from argparse import Namespace

from .helper import RetrievalHelper


def build_args(args: Namespace):
    query_dir = args.query_dir
    gallery_dir = args.gallery_dir

    distance_type = args.distance
    retrieval_type = args.retrieval

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    topk = args.topk

    retrieval_helper = RetrievalHelper(query_dir, gallery_dir, save_dir, topk, distance_type, retrieval_type)
    return retrieval_helper


def build_cfg():
    pass
