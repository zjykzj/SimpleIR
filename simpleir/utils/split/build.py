# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:12
@file: build.py
@author: zj
@description: 
"""

from . import caltech101, general


def build_splitter(dataset_type, **kwargs):
    if dataset_type in caltech101.__all__:
        return caltech101.__dict__[dataset_type]()
    elif dataset_type in general.__all__:
        return general.__dict__[dataset_type](**kwargs)
    else:
        raise ValueError(f'{dataset_type} does not support')
