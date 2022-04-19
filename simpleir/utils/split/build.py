# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:12
@file: build.py
@author: zj
@description: 
"""

from . import caltech101

__support__ = caltech101.__all__


def build_splitter(dataset_type):
    assert dataset_type in __support__

    if dataset_type == caltech101.__all__:
        return caltech101.__dict__[dataset_type]()
    else:
        raise ValueError(f'{dataset_type} does not support')
