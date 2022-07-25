# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午4:06
@file: tiny_autocoder.py
@author: zj
@description: 
"""

import torch

from zcls2.config.key_word import KEY_OUTPUT

from simpleir.configs.key_words import KEY_FEAT
from simpleir.models.impl.tiny_autocoder import TinyAutoCoder

if __name__ == '__main__':
    img = torch.randn((10, 1, 28, 28))
    # img = torch.randn((10, 1, 224, 224))
    print(img.shape)
    model = TinyAutoCoder(in_channels=1)

    # encoded, decoded = model(img)
    res_dict = model(img)
    encoded = res_dict[KEY_FEAT]
    decoded = res_dict[KEY_OUTPUT]

    print(encoded.shape)
    print(decoded.shape)
