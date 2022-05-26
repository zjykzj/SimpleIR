# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午4:01
@file: index.py
@author: zj
@description: 图像检索
1. 依次计算每张查询集图像和图库图像之间的相似度
2. 按照相似度进行排序，计算top1/top5结果
"""

import argparse

from simpleir.configs import get_cfg_defaults
from simpleir.utils.index.indexer import Indexer

from zcls2.util import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Index and eval query set")
    parser.add_argument('cfg',
                        type=str,
                        metavar="CONFIG",
                        help="path to config file")

    return parser.parse_args()


if __name__ == '__main__':
    # Parameter configuration
    args = parse_args()
    print('args:', args)
    save_interval = args.save_interval

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    logging.setup_logging(local_rank=cfg.RANK_ID, output_dir=None)

    indexer = Indexer(cfg)
    indexer.run()
