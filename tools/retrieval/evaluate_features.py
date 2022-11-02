# -*- coding: utf-8 -*-

"""
@date: 2022/7/16 上午11:54
@file: evaluate_features.py
@author: zj
@description: Evaluate retrieval results
"""

import argparse

import numpy as np

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

    parser.add_argument('--query-dir', metavar='QUERY', default='./data/oxford5k', type=str,
                        help='Original image data set path, used for Oxford5k/Paris6k evaluation.'
                             ' Default: ./data/oxford5k')
    parser.add_argument('--dataset', metavar='DATASET', default='oxford5k', type=str,
                        help='Dataset name, used for Oxford5k/Paris6k/ROxford5k/RParis6k evaluation.'
                             ' Default: oxford5k')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.setup_logging(local_rank=0, output_dir=None)
    logger.info(f'args: {args}')

    top_k_list = (1, 3, 5, 10)
    metric_helper = build_args(args, top_k_list=top_k_list)
    top_list = metric_helper.run()

    if args.retrieval_type == 'MAP_OXFORD':
        logger.info(f"=> MAP: {top_list[0]:.3f}%")
    elif args.retrieval_type == 'MAP_ROXFORD':
        if args.dataset.startswith('oxford5k') or args.dataset.startswith('paris6k'):
            assert len(top_list) == 2
            map, aps = top_list[:2]
            logger.info('>> {}: mAP {:.2f}'.format(args.dataset, np.around(map * 100, decimals=2)))
        else:
            assert len(top_list) == 3 and len(top_list[0]) == 4
            mapE, apsE, mprE, prsE = top_list[0]
            mapM, apsM, mprM, prsM = top_list[1]
            mapH, apsH, mprH, prsH = top_list[2]
            kappas = top_k_list

            logger.info('>> {}: mAP E: {}, M: {}, H: {}'.format(args.dataset, np.around(mapE * 100, decimals=2),
                                                                np.around(mapM * 100, decimals=2),
                                                                np.around(mapH * 100, decimals=2)))
            logger.info(
                '>> {}: mP@k{} E: {}, M: {}, H: {}'.format(args.dataset, kappas, np.around(mprE * 100, decimals=2),
                                                           np.around(mprM * 100, decimals=2),
                                                           np.around(mprH * 100, decimals=2)))
    else:
        for top, k in zip(top_list, top_k_list):
            if args.retrieval_type == EvaluateType.ACCURACY.value:
                logger.info(f"=> ACC@{k}: {top}:.3f%")
            elif args.retrieval_type == EvaluateType.PRECISION.value:
                logger.info(f"=> Pre@{k}: {top}:.3f%")
            else:
                logger.info(f"=> MAP@{k}: {top}:.3f%")


if __name__ == '__main__':
    main()
