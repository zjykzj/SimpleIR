# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:55
@file: extract_features.py
@author: zj
@description:
"""

import os
import argparse

from simpleir.configs import get_cfg_defaults
from simpleir.utils.extract.helper import ExtractHelper

from zcls2.util import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features")
    parser.add_argument('cfg',
                        type=str, metavar="CONFIG", help="path to config file")
    parser.add_argument('--gallery', default=False, action="store_true",
                        help='Path to the query set or galley set. Default: False (for query set)')
    parser.add_argument('-s', '--save-interval', metavar='INTERVAL',
                        default=5000, type=int, help='Save interval. Default: 5000')

    return parser.parse_args()


if __name__ == '__main__':
    # Parameter configuration
    args = parse_args()
    print('args:', args)
    save_interval = args.save_interval

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    if args.gallery:
        dst_root = cfg.EVAL.INDEX.GALLERY_DIR
    else:
        dst_root = cfg.EVAL.FEATURE.QUERY_DIR
    print(f'extract feats to {dst_root}')

    logging.setup_logging(local_rank=cfg.RANK_ID, output_dir=None)

    # Extract
    extractor = ExtractHelper(cfg, is_gallery=args.gallery)
    print('extract ...')
    extractor.run(dst_root, save_prefix='part_', save_interval=save_interval)
