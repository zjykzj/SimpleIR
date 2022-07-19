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


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval features")
    parser.add_argument('--retrieval-dir', metavar='RETRIEVAL', default=None, type=str,
                        help='Dir for loading retrieval results. Default: None')
    parser.add_argument('--retrieval-type', metavar='EVAL', default='ACC', type=str,
                        choices=list(EvaluateType.__members__.keys()),
                        help='Which evaluation method. Default: ACC')

    return parser.parse_args()


def main():
    args = parse_args()
    print('args:', args)

    metric_helper = build_args(args)
    metric_helper.run()


if __name__ == '__main__':
    main()
