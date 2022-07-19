# -*- coding: utf-8 -*-

"""
@date: 2022/7/18 下午9:27
@file: model_base.py
@author: zj
@description: 
"""

from functools import partial

__all__ = ['ModelBase']


class ModelBase(object):
    _feat_list = []

    def support_feat(self, feat_type):
        return feat_type in self._feat_list

    def _register_hook(self) -> None:
        """
        Register hooks to output inner feature map.
        """

        def hook(feature_buffer, fea_name, module, input, output):
            feature_buffer[fea_name] = output.data

        for fea_name in self._feat_list:
            assert fea_name in self.feature_modules, 'unknown feature {}!'.format(fea_name)
            self.feature_modules[fea_name].register_forward_hook(partial(hook, self.feature_buffer, fea_name))