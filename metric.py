# -*- coding: utf-8 -*-

"""
@date: 2023/8/23 上午9:57
@file: metric.py
@author: zj
@description: 
"""

import os
import sys

import argparse
from argparse import Namespace

from pathlib import Path

from simpleir.metric.helper import MetricHelper, EvaluateType
from simpleir.utils.logger import LOGGER
from simpleir.utils.misc import print_args, colorstr
from simpleir.utils.fileutil import increment_path
from simpleir.utils.general import load_retrieval

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    evalaute_types = [e.value for e in EvaluateType]

    parser = argparse.ArgumentParser()
    parser.add_argument('retrieval', type=str, help='retrieval info path')

    parser.add_argument('--evaluate', type=str, default='ACCURACY', choices=evalaute_types,
                        help='evaluate type: ' +
                             ' | '.join(evalaute_types) +
                             ' (default: ACCURACY)')

    parser.add_argument('--project', default=ROOT / 'runs/metric', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    return opt


def main(opt: Namespace):
    # Config
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print_args(vars(opt))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Data
    batch_rank_name_list, batch_rank_label_list, query_name_list, query_label_list = load_retrieval(opt.retrieval)

    # Metric
    metric_helper = MetricHelper(batch_rank_name_list=batch_rank_name_list,
                                 batch_rank_label_list=batch_rank_label_list,
                                 query_name_list=query_name_list,
                                 query_label_list=query_label_list,
                                 eval_type=opt.evaluate, top_k_list=(1, 3, 5, 10), )

    LOGGER.info("Metric")
    acc = metric_helper.run()
    LOGGER.info(f"Accuracy: {acc}")

    # Save
    LOGGER.info(f"Save to {colorstr(opt.save_dir)}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
