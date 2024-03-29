# -*- coding: utf-8 -*-

"""
@date: 2023/8/20 下午12:17
@file: extract.py
@author: zj
@description:

Usage - Extract Features:
    $ python extract.py --arch resnet18 --data toy.yaml

Usage - Reduce dimension:
    $ python extract.py --arch resnet18 --data toy.yaml --enhance PCA --reduce 50 --learn-pca

"""

import os
import sys

import argparse
from argparse import Namespace

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.mobilenet import MobileNetV2, MobileNetV3, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from simpleir.utils.logger import LOGGER
from simpleir.utils.misc import print_args, colorstr
from simpleir.data.build import build_data
from simpleir.extract.helper import ExtractHelper, AggregateType, EnhanceType
from simpleir.utils.fileutil import increment_path, check_yaml, yaml_load
from simpleir.utils.general import save_features

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # SimpleIR root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    model_names = sorted(name for name in models.resnet.__dict__ if name.islower() and name.startswith('resnet'))
    model_names += sorted(
        name for name in models.mobilenet.__dict__ if name.islower() and name.startswith('mobilenet_'))
    # print(model_names)

    aggregate_types = [e.value for e in AggregateType]
    enhance_types = [e.value for e in EnhanceType]

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--data', type=str, default=ROOT / 'configs/data/toy.yaml', help='dataset.yaml path')

    parser.add_argument('--aggregate', type=str, default='IDENTITY', choices=aggregate_types,
                        help='aggregate type: ' +
                             ' | '.join(aggregate_types) +
                             ' (default: IDENTITY)')
    parser.add_argument('--enhance', type=str, default='IDENTITY', choices=enhance_types,
                        help='enhance type: ' +
                             ' | '.join(enhance_types) +
                             ' (default: IDENTITY)')
    parser.add_argument('--reduce', type=int, default=512, help='reduce dimension')
    parser.add_argument('--learn-pca', action='store_true', default=False,
                        help='whether to perform PCA learning')
    parser.add_argument('--pca-path', type=str, default=None, help='load the learned PCA model')

    parser.add_argument('--project', default=ROOT / 'runs/extract', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    return opt


def do_extract(data_loader: DataLoader, extract_helper: ExtractHelper, save_dir: str, is_gallery: bool = False):
    image_name_list, label_list, feat_tensor_list = extract_helper.run(data_loader, is_gallery=is_gallery)

    # Save
    if is_gallery:
        feat_dir = os.path.join(save_dir, 'gallery')
        info_path = os.path.join(save_dir, "gallery.pkl")
    else:
        feat_dir = os.path.join(save_dir, 'query')
        info_path = os.path.join(save_dir, "query.pkl")

    assert hasattr(data_loader.dataset, 'classes')
    classes = data_loader.dataset.classes
    save_features(classes, image_name_list, label_list, feat_tensor_list, feat_dir, info_path)


def main(opt: Namespace):
    # Config
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print_args(vars(opt))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    opt.data = check_yaml(opt.data)
    opt.data = yaml_load(opt.data)

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

    # Extract
    extract_helper = ExtractHelper(model=model,
                                   target_layer=target_layer,
                                   device=device,
                                   aggregate_type=opt.aggregate,
                                   enhance_type=opt.enhance,
                                   reduce_dimension=opt.reduce,
                                   learn_pca=opt.learn_pca,
                                   pca_path=opt.pca_path,
                                   save_dir=opt.save_dir
                                   )

    LOGGER.info("Extract Gallery")
    do_extract(gallery_loader, extract_helper, opt.save_dir, is_gallery=True)

    LOGGER.info("Extract Query")
    do_extract(query_loader, extract_helper, opt.save_dir, is_gallery=False)

    LOGGER.info(f"Save to {colorstr(opt.save_dir)}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
