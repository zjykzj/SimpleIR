# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午9:59
@file: make_query_gallery_set.py
@author: zj
@description: Create query sets and gallery sets
1. Specify dataset type。for example, `Caltech101` / `Oxford5k`;
2. Specify dataset path and result path;
3. For `General` datasets, whether to set the maximum number of saves per category.
"""

import argparse

from simpleir.utils.split.build import build_splitter


def parse_args():
    parser = argparse.ArgumentParser(description="Make Query and Gallery Set")
    parser.add_argument('-d', '--dataset', metavar='DATASET', default='Caltech101',
                        type=str, help='Dataset type. Default Caltech101')
    parser.add_argument('src', metavar="SRC", default='./data/caltech101/',
                        type=str, help='Path to the src dataset')
    parser.add_argument('dst', metavar='DST', default='./data/caltech101/',
                        type=str, help='Path to the dst dataset')
    parser.add_argument('-max', metavar='MAX', default=20,
                        type=int, help="Max num each class in dataset. Just used for General")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('args:', args)

    src_root = args.src
    dst_root = args.dst
    dataset_type = args.dataset
    max_num = args.max

    splitter = build_splitter(dataset_type, max_num=max_num)
    splitter.run(src_root, dst_root)
