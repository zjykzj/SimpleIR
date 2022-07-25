# -*- coding: utf-8 -*-

"""
@date: 2022/6/6 下午5:24
@file: prefetcher.py
@author: zj
@description: 
"""

from simpleir.utils.prefetcher import data_prefetcher
from simpleir.utils.extract.impl.extractor import create_loader

if __name__ == '__main__':
    from simpleir.configs import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file('tools/retrieval/configs/oxford/ghostnet_100_act2_c5_oxford_224_b256.yaml')

    loader = create_loader(cfg)
    res = data_prefetcher(cfg, loader)
    print(type(res))

    from collections.abc import Iterable, Iterator

    print(isinstance(res, Iterable))
    print(isinstance(res, Iterator))

    res = iter(res)
    print(isinstance(res, Iterable))
    print(isinstance(res, Iterator))
