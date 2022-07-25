# -*- coding: utf-8 -*-

"""
@date: 2022/5/6 下午8:36
@file: ghostnet.py
@author: zj
@description: Custom MobileNetV3, derived from zcls2
"""
from zcls2.model.model.ghostnet import default_cfgs, build_model_with_cfg
from zcls2.model.model.ghostnet import GhostNet as ZGhostNet

from simpleir.configs.key_words import KEY_FEAT
from simpleir.models.model_base import ModelBase

__all__ = ["GhostNet", "ghostnet_050", "ghostnet_100", "ghostnet_130"]


class GhostNet(ZGhostNet, ModelBase):
    _feat_list = [
        'blocks', 'global_pool', 'conv_head', 'act2', 'fc'
    ]

    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, in_chans=3, output_stride=32,
                 feat_type='act2'):
        super().__init__(cfgs, num_classes, width, dropout, in_chans, output_stride)
        assert feat_type in self._feat_list

        self.feature_modules = {
            "blocks": self.blocks[9][0].act1,
            "global_pool": self.global_pool,
            "conv_head": self.conv_head,
            "act2": self.act2,
            "fc": self.classifier,
        }
        self.feature_buffer = dict()
        self.feat_type = feat_type
        self._register_hook()

    def forward(self, x):
        res = super(GhostNet, self).forward(x)
        res[KEY_FEAT] = self.feature_buffer[self.feat_type]

        return res


def _create_ghostnet(variant, width=1.0, pretrained=False, **kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        GhostNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs)


def ghostnet_050(pretrained=False, **kwargs):
    """ GhostNet-0.5x """
    model = _create_ghostnet('ghostnet_050', width=0.5, pretrained=pretrained, **kwargs)
    return model


def ghostnet_100(pretrained=False, **kwargs):
    """ GhostNet-1.0x """
    model = _create_ghostnet('ghostnet_100', width=1.0, pretrained=pretrained, **kwargs)
    return model


def ghostnet_130(pretrained=False, **kwargs):
    """ GhostNet-1.3x """
    model = _create_ghostnet('ghostnet_130', width=1.3, pretrained=pretrained, **kwargs)
    return model
