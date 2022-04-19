# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:55
@file: extract_features.py
@author: zj
@description:
1. 加载数据
2. 加载图像处理器
3. 加载模型
4. 提取特征
"""

import argparse

from simpleir.utils.extract.build import build_extractor


def parse_args():
    parser = argparse.ArgumentParser(description="Make Query and Gallery Set")
    parser.add_argument('-d', '--dataset', metavar='DATASET', default='Caltech101', type=str, help='Dataset type')
    parser.add_argument('SRC', default='./data/caltech101/gallery', type=str, help='Path to the dataset')
    parser.add_argument('DST', default='./data/extract/caltech101/gallery/', type=str, help='Path to the save feature')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    src_root = args.SRC
    dst_root = args.DST
    dataset_type = args.dataset

    extractor = build_extractor(src_root, dataset_type)
    extractor.run(dst_root)
