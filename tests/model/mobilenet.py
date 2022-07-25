# -*- coding: utf-8 -*-

"""
@date: 2022/5/11 下午10:45
@file: mobilenet.py
@author: zj
@description: 
"""

from simpleir.models.impl.mobilenet import mobilenet_v3_large

if __name__ == '__main__':
    m = mobilenet_v3_large()
    print(m)

    state_dict = m.state_dict()
    print(state_dict.keys())
