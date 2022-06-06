# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:12
@file: build.py
@author: zj
@description: 
"""

from . import caltech101, general, oxford5k


def build_splitter(dataset_type, **kwargs):
    if dataset_type in general.__all__:
        return general.__dict__[dataset_type](**kwargs)
    elif dataset_type in caltech101.__all__:
        return caltech101.__dict__[dataset_type]()
    elif dataset_type in oxford5k.__all__:
        return oxford5k.__dict__[dataset_type]()
    else:
        raise ValueError(f'{dataset_type} does not support')
