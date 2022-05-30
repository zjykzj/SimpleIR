# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午4:01
@file: retrieval_images.py
@author: zj
@description: Image Retrieval
"""

import argparse
from argparse import RawTextHelpFormatter

from simpleir.configs import get_cfg_defaults
from simpleir.utils.retrieval.helper import RetrievalHelper

from zcls2.util import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Index and eval query set", formatter_class=RawTextHelpFormatter)
    parser.add_argument('cfg',
                        type=str, metavar="CONFIG", help="path to config file")
    parser.add_argument('-m', '--mode', metavar='MODE', type=int, default=0,
                        help='Index Mode.\n'
                             'Mode = 0: Make query as gallery and batch update gallery set\n'
                             'Mode = 1: Make query as gallery and single update gallery set\n'
                             'Mode = 2: Set gallery set and no update\n'
                             'Mode = 3: Set gallery set and batch update gallery set\n'
                             'Mode = 4: Set gallery set and single update gallery set\n'
                             'Default: -1 (use CONFIG setting)'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Parameter configuration
    args = parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    logging.setup_logging(local_rank=cfg.RANK_ID, output_dir=None)
    logger.info(f"Loaded args: {args}")

    if args.mode != -1:
        cfg.EVAL.INDEX.MODE = args.mode
    logger.info(f'Index Mode: {cfg.EVAL.INDEX.MODE}')

    # Retrieval
    indexer = RetrievalHelper(cfg)
    logger.info('Retrieval ...')
    indexer.run()
