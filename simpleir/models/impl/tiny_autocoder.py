# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: model.py
@author: zj
@description:
"""
from typing import Dict

import torch
import torch.nn as nn

from zcls2.config.key_word import KEY_OUTPUT
from simpleir.configs.key_words import KEY_INPUT, KEY_FEAT

__all__ = ['TinyAutoCoder']


class TinyAutoCoder(nn.Module):
    def __init__(self, in_channels=3):
        super(TinyAutoCoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x) -> Dict:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return {
            KEY_INPUT: x,
            KEY_FEAT: encoded,
            KEY_OUTPUT: decoded
        }
        # return encoded, decoded
        # return decoded
