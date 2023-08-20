# -*- coding: utf-8 -*-

"""
@date: 2022/7/16 下午3:44
@file: retrieval_features.py
@author: zj
@description: Retrieval features
"""

import argparse

from simpleir.retrieval import build_args
from simpleir.retrieval.impl.distancer import DistanceType
from simpleir.retrieval.impl.ranker import RankType
from simpleir.retrieval.impl.reranker import ReRankType

from zcls2.util import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval features.")
    parser.add_argument('--query-dir', metavar='QUERY', default=None, type=str,
                        help='Dir for loading query features. Default: None')
    parser.add_argument('--gallery-dir', metavar='GALLERY', default=None, type=str,
                        help='Dir for loading gallery features. Default: None')

    parser.add_argument('--distance', metavar='DISTANCE', default='EUCLIDEAN', type=str,
                        choices=list(DistanceType.__members__.keys()),
                        help='The way to compute distance. Default: EUCLIDEAN')
    parser.add_argument('--rank', metavar='RANK', default='NORMAL', type=str,
                        choices=list(RankType.__members__.keys()),
                        help='The way to retrieval. Default: NORMAL')
    parser.add_argument('--rerank', metavar='RERANK', default='IDENTITY', type=str,
                        choices=list(ReRankType.__members__.keys()),
                        help='The way to retrieval. Default: IDENTITY')

    parser.add_argument('--save-dir', metavar='SAVE', default=None, type=str,
                        help='Dir for saving retrieval results. Default: None')
    parser.add_argument('--topk', metavar='TOPK', default=None, type=int,
                        help='Saving topk results. Default: None (Save all)')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.setup_logging(local_rank=0, output_dir=None)
    logger.info(f'args: {args}')

    retrieval_helper = build_args(args)
    retrieval_helper.run()


if __name__ == '__main__':
    main()
