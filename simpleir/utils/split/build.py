# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:12
@file: build.py
@author: zj
@description: 
"""

from .caltech101 import Caltech101

__support__ = [
    'Caltech101'
]


def build_splitter(root, type):
    assert type in __support__

    if type == 'Caltech101':
        return Caltech101(root)
    else:
        raise ValueError(f'{type} does not support')
