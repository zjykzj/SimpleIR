# -*- coding: utf-8 -*-

"""
@date: 2022/7/16 上午11:45
@file: extract_features.py
@author: zj
@description: Extract features
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from simpleir.utils.extract.build import build_args
from simpleir.utils.extract.impl.aggregator import AggregateType
from simpleir.utils.extract.impl.enhancer import EnhanceType

from zcls2.util import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = ArgumentParser(description="Extract features.", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model-arch', metavar="ARCH", default='resnet50', type=str,
                        help='Model arch for extracting features. Default: resnet50')
    parser.add_argument('--pretrained', metavar='PRETRAINED', default=None, type=str,
                        help='Pretrained model params path. Default: None')
    parser.add_argument('--layer', metavar='LAYER', default='fc', type=str,
                        help='Location of model extracted features. Default: fc')

    parser.add_argument('--gallery', action='store_true',
                        help='Is it gallery. Default: True')
    parser.add_argument('--dataset', metavar='DATASET', default='General',
                        help='Dataset type for image processing. Default: General')
    parser.add_argument('--image-dir', metavar='IMAGE', default=None,
                        help='Dir for loading images. Default: None')
    parser.add_argument('--save-dir', metavar='SAVE', default=None,
                        help='Dir for saving features. Default: None')

    parser.add_argument('--aggregate', dest='aggregate',
                        metavar='AGGREGATE', default='IDENTITY', type=str,
                        choices=list(AggregateType.__members__.keys()),
                        help='The way to aggregate features. Default: IDENTITY')
    parser.add_argument('--enhance', dest='enhance',
                        metavar='PROCESS', default='IDENTITY', type=str, choices=list(EnhanceType.__members__.keys()),
                        help='The way to post process. Default: IDENTITY')
    parser.add_argument('--learn-pca', dest='learn',
                        metavar='LEARN', default=True, type=bool,
                        help='Whether to learn PCA / PCA_W in gallery. Default: True')
    parser.add_argument('--pca-path', dest='pca',
                        metavar='PCA', default=None, type=str,
                        help='Path of PCA model. Default: None')
    parser.add_argument('-rd', '--reduce-dimension', dest='rd',
                        metavar='DIMENSION', default=512, type=int,
                        help='Dimension after dimension reduction. Default: 512')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.setup_logging(local_rank=0, output_dir=None)
    logger.info(f'args: {args}')

    extract_helper = build_args(args)
    extract_helper.run()


if __name__ == '__main__':
    main()
