# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午4:01
@file: index.py
@author: zj
@description: 图像检索
1. 依次计算每张查询集图像和图库图像之间的相似度
2. 按照相似度进行排序，计算top1/top5结果
"""

import os
import argparse

from simpleir.index.build import build_indexer


def parse_args():
    parser = argparse.ArgumentParser(description="Make Query and Gallery Set")
    parser.add_argument('-g', '--gallery', metavar='GALLERY', default='./data/extract/caltech101/gallery', type=str,
                        help='Path to gallery features')
    parser.add_argument('-q', '--query', metavar='QUERY', default='./data/extract/caltech101/query', type=str,
                        help='Path to query features')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    root_gallery = args.gallery
    root_query = args.query

    build_indexer(root_gallery, root_query)