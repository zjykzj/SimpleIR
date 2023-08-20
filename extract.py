# -*- coding: utf-8 -*-

"""
@date: 2023/8/20 下午12:17
@file: extract.py
@author: zj
@description: 
"""

import os
import sys
import argparse

from pathlib import Path

import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.mobilenet import MobileNetV2, MobileNetV3, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from simpleir.utils.logger import LOGGER
from simpleir.data.build import build_data
from simpleir.extract.helper import ExtractHelper
from simpleir.utils.general import increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    model_names = sorted(name for name in models.resnet.__dict__ if name.islower() and name.startswith('resnet'))
    model_names += sorted(
        name for name in models.mobilenet.__dict__ if name.islower() and name.startswith('mobilenet_'))
    # print(model_names)

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--data', type=str, default=ROOT / 'configs' / 'data/coco128.yaml', help='dataset.yaml path')

    parser.add_argument('--aggregate', type=str, default='identity', help='aggregate type')
    parser.add_argument('--enhance', type=str, default='identity', help='enhance type')
    parser.add_argument('--reduce', type=str, default='identity', help='reduce dimension')
    parser.add_argument('--learn-pca', type=bool, action='store_true', default=False,
                        help='Whether to perform PCA learning')
    parser.add_argument('--pca-path', type=str, default=None, help='Whether to perform PCA learning')

    parser.add_argument('--project', default=ROOT / 'runs/extract', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    return opt


def process(opt, cfg):
    """
    加载模型，加载数据，批量处理
    """
    # Data
    gallery_loader = build_data(cfg, is_gallery=True)
    query_loader = build_data(cfg, is_gallery=False)
    classes = cfg['names']

    # Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = eval(opt.arch)(pretrained=True).to(device)
    model.eval()

    if isinstance(model, ResNet):
        target_layer = model.layer4[-1]
    else:
        pass

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

    LOGGER.info("Extract gallery")
    extract_helper.run(gallery_loader, is_gallery=True)
    LOGGER.info("Extract query")
    extract_helper.run(query_loader, is_gallery=False)


def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print(f"opt: {opt}")

    # 加载配置文件
    import yaml  # for torch hub
    yaml_file = Path(opt.cfg).name
    with open(yaml_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict

    process(opt, cfg)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
