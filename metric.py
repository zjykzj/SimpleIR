# -*- coding: utf-8 -*-

"""
@date: 2023/8/23 上午9:57
@file: metric.py
@author: zj
@description:

Usage - Metric Features:
    $ python metric.py /path/to/retrieval.pkl --evaluate ACCURACY
    $ python metric.py /path/to/retrieval.pkl --evaluate PRECISION
    $ python metric.py /path/to/retrieval.pkl --evaluate MAP

"""
import argparse
from argparse import Namespace

from simpleir.metric.helper import MetricHelper, EvaluateType
from simpleir.utils.logger import LOGGER
from simpleir.utils.general import load_retrieval


def parse_opt():
    evalaute_types = [e.value for e in EvaluateType]

    parser = argparse.ArgumentParser()
    parser.add_argument('retrieval', type=str, help='retrieval info path')

    parser.add_argument('--evaluate', type=str, default='ACCURACY', choices=evalaute_types,
                        help='evaluate type: ' +
                             ' | '.join(evalaute_types) +
                             ' (default: ACCURACY)')

    opt = parser.parse_args()
    return opt


def main(opt: Namespace):
    # Data
    batch_rank_name_list, batch_rank_label_list, query_name_list, query_label_list = load_retrieval(opt.retrieval)

    # Metric
    metric_helper = MetricHelper(batch_rank_name_list=batch_rank_name_list,
                                 batch_rank_label_list=batch_rank_label_list,
                                 query_name_list=query_name_list,
                                 query_label_list=query_label_list,
                                 eval_type=opt.evaluate, top_k_list=(1, 3, 5, 10), )

    LOGGER.info("Metric")
    res = metric_helper.run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
