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

from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.mobilenet import MobileNetV2, MobileNetV3, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from simpleir.data.build import build_data

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

"""
data指的是数据集配置
hyp指的是提取、检索、评估配置
还有一个模型加载，这个默认设置为torchvision提供的模型

批量这些都设置为默认的先

按照文件名保存，

gallery/
    cate1/
        xxx.npy
    cate2/
        xxx.npy
query/
    cate1/
        xxx.npy
    cate2/
        xxx.npy

在提取过程中完成特征的提取、融合和增强。不涉及保存。把数据返回后再进行保存

提取器是要保存数据的，所以需要输入保存路径

pca.pkl
pcaw.pkl
"""


def parse_opt(known=False):
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
    parser.add_argument('--hyp', type=str, default=ROOT / 'configs' / 'hyps/hyp.scratch-low.yaml',
                        help='hyperparameters path')

    parser.add_argument('--project', default=ROOT / 'runs/', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    return opt


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break

        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class ExtractItem:

    def __init__(self, image_name, target, feat_tensor):
        self.image_name = image_name
        self.target = target
        self.feat_tensor = feat_tensor


def process(opt, cfg):
    """
    加载模型，加载数据，批量处理
    """
    # Data
    gallery_loader = build_data(cfg, is_gallery=True)
    query_loader = build_data(cfg, is_gallery=False)

    # Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = eval(opt.arch)(pretrained=True).to(device)
    model.eval()
    global batch_feat_tensor
    batch_feat_tensor = None

    def forward_hook(module, input, output):
        global batch_feat_tensor
        batch_feat_tensor = output.detach().clone().cpu()

    if isinstance(model, ResNet):
        model.layer4[-1].register_forward_hook(forward_hook)
    elif isinstance(model, (MobileNetV2, MobileNetV3)):
        model.features[-1].register_forward_hook(forward_hook)
    else:
        pass

    # Extract
    extract_items = list()
    for images, targets, paths in tqdm(dataloader):
        _ = model.forward(images.to(device))

        for path, target, feat_tensor in zip(paths, targets.numpy(), batch_feat_tensor):
            image_name = os.path.basename(path).split('.')[0]

            extract_items.append(ExtractItem(image_name, target, feat_tensor))

    # Aggregate

    return image_name_list, target_list, torch.stack(feat_tensor_list)


def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print(f"opt: {opt}")

    process(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
