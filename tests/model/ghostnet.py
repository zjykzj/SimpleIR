# -*- coding: utf-8 -*-

"""
@date: 2022/5/8 下午2:59
@file: ghostnet.py
@author: zj
@description: 
"""

from simpleir.models.impl.ghostnet import ghostnet_100

if __name__ == '__main__':
    # m = timm.models.__dict__['ghostnet_100']()
    m = ghostnet_100()
    print(m)

    # import torch
    #
    # a = torch.randn(1, 3, 224, 224)
    # res = m(a)
    # print(res.keys())
    # print(res[KEY_OUTPUT].shape)
    # print(res[KEY_FEAT].shape)

    # print(m.default_cfg)

    # children = list(m.children())
    # print(children)
    # print(m.blocks[9])
    # print(m.blocks[9][0])
    # print(m.blocks[9][0].act1)
    #
    # print(m.conv_head)
    # print(m.classifier)
    # blocks
    # pool
    # conv_head
    # fc
