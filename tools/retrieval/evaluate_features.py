# -*- coding: utf-8 -*-

"""
@date: 2022/7/16 上午11:54
@file: evaluate_features.py
@author: zj
@description: Evaluate retrieval results
"""

import argparse

from simpleir.utils.metric.build import build_args
from simpleir.utils.metric.helper import EvaluateType

from zcls2.util import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval features")
    parser.add_argument('--retrieval-dir', metavar='RETRIEVAL', default=None, type=str,
                        help='Dir for loading retrieval results. Default: None')
    parser.add_argument('--retrieval-type', metavar='EVAL', default='ACCURACY', type=str,
                        choices=list(EvaluateType.__members__.keys()),
                        help='Which evaluation method. Default: ACCURACY')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.setup_logging(local_rank=0, output_dir=None)
    logger.info(f'args: {args}')

    top_k_list = (1, 3, 5, 10)
    metric_helper = build_args(args, top_k_list=top_k_list)
    top_list = metric_helper.run()

    for top, k in zip(top_list, top_k_list):
        if args.retrieval_type == EvaluateType.ACCURACY.value:
            logger.info(f"ACC@{k}: {top}%")
        else:
            logger.info(f"MAP@{k}: {top}%")


if __name__ == '__main__':
    main()
