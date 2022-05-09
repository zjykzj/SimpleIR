# -*- coding: utf-8 -*-

"""
@date: 2022/5/8 下午3:57
@file: resnet.py
@author: zj
@description: 
"""

from simpleir.models.resnet import resnet50

if __name__ == '__main__':
    m = resnet50()
    print(m)

    print(m.layer4)
    print(m.layer4[2])
    print(m.layer4[2].relu)
    print(m.avgpool)
    print(m.fc)

