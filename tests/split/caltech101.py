# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午7:15
@file: caltech101.py
@author: zj
@description: 
"""

from simpleir.utils.split import caltech101

print(caltech101.__all__)
# print(caltech101.__dict__)

dataset_type = 'Caltech101'

m = caltech101.__dict__[dataset_type]('./data/')
print(m)
