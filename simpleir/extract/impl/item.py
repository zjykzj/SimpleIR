# -*- coding: utf-8 -*-

"""
@date: 2023/8/20 下午1:35
@file: item.py
@author: zj
@description: 
"""


class ExtractItem:

    def __init__(self, image_name, target, feat_tensor):
        self.image_name = image_name
        self.target = target
        self.feat_tensor = feat_tensor
