# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 下午7:44
@file: build.py
@author: zj
@description: 
"""

from argparse import Namespace

from .helper import MetricHelper


def build_args(args: Namespace):
    retrieval_dir = args.retrieval_dir
    retrieval_type = args.retrieval_type

    metric_helper = MetricHelper(retrieval_dir, eval_type=retrieval_type)
    return metric_helper


def build_cfg():
    pass
