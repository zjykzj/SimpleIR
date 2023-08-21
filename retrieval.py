# -*- coding: utf-8 -*-

"""
@date: 2023/8/21 上午11:17
@file: retrieval.py
@author: zj
@description: 
"""

import os
import sys
import argparse

from pathlib import Path

import torch

from simpleir.retrieval.helper import Distancer
from simpleir.utils.logger import LOGGER
from simpleir.utils.misc import print_args, colorstr
from simpleir.data.build import build_data
from simpleir.utils.fileutil import increment_path, check_yaml, yaml_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    distance_types = [e.value for e in AggregateType]


    parser = argparse.ArgumentParser()
    parser.add_argument('gallery', type=str, help='dataset.yaml path')
    parser.add_argument('query', type=str, help='dataset.yaml path')

    parser.add_argument("--distance", type=str, default='EUCLIDEAN',)
    parser.add_argument('--rank', type=str, default='NORMAL',
                        help='aggregate type: ' +
                             ' | '.join(aggregate_types) +
                             ' (default: identity)')
    parser.add_argument('--enhance', type=str, default='IDENTITY',
                        help='enhance type: ' +
                             ' | '.join(enhance_types) +
                             ' (default: identity)')

    parser.add_argument('--project', default=ROOT / 'runs/retrieval', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    return opt


def process(opt):
    """
    加载模型，加载数据，批量处理
    """
    # Data
    gallery_loader = build_data(opt.data, is_gallery=True)
    query_loader = build_data(opt.data, is_gallery=False)

    # Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = eval(opt.arch)(pretrained=True).to(device)
    model.eval()

    if isinstance(model, ResNet):
        target_layer = model.fc
    elif isinstance(model, MobileNetV2):
        target_layer = model.classifier[-1]
    else:
        assert isinstance(model, MobileNetV3)
        target_layer = model.classifier[-1]

    # Extractor
    extract_helper = ExtractHelper(save_dir=opt.save_dir,
                                   model=model,
                                   target_layer=target_layer,
                                   device=device,
                                   aggregate_type=opt.aggregate,
                                   enhance_type=opt.enhance,
                                   reduce_dimension=opt.reduce,
                                   learn_pca=opt.learn_pca,
                                   pca_path=opt.pca_path)

    LOGGER.info("Extract query")
    extract_helper.run(query_loader, is_gallery=False)
    LOGGER.info("Extract gallery")
    extract_helper.run(gallery_loader, is_gallery=True)


def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print_args(vars(opt))

    opt.data = check_yaml(opt.data)
    opt.data = yaml_load(opt.data)
    process(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
