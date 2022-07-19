# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 上午10:04
@file: buld.py
@author: zj
@description:
"""

import os
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from simpleir.models import *

from .helper import ExtractHelper


def load_model(arch='resnet50', pretrained=None, layer='fc'):
    model = eval(arch)()
    assert isinstance(model, torch.nn.Module)

    assert isinstance(model, ModelBase)
    if not model.support_feat(layer):
        raise ValueError(f'{arch} does not support {layer}')

    model = eval(arch)(pretrained=True, feat_type=layer)
    assert isinstance(model, torch.nn.Module)

    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    return model


def custom_fn(batches):
    images = [batch[0] for batch in batches]
    targets = [batch[1] for batch in batches]
    paths = [batch[2] for batch in batches]

    return torch.stack(images), torch.stack(targets), paths


def load_data(data_root, dataset='General', transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    data_set = eval(dataset)(root=data_root, transform=transform, w_path=True)

    return DataLoader(data_set, collate_fn=custom_fn, shuffle=False, batch_size=32, num_workers=0, pin_memory=True)


def build_args(args: Namespace) -> ExtractHelper:
    model_arch = args.model_arch
    pretrained = args.pretrained
    layer = args.layer

    image_dir = args.image_dir
    assert os.path.isdir(image_dir), image_dir
    dataset = args.dataset
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    aggregate = args.aggregate
    enhance = args.enhance
    rd = args.rd

    model = load_model(arch=model_arch, pretrained=pretrained, layer=layer)
    print(model)

    data_loader = load_data(image_dir, dataset=dataset)
    print(data_loader)

    extract_helper = ExtractHelper(model=model, model_arch=model_arch, pretrained=pretrained, layer=layer,
                                   data_loader=data_loader, save_dir=save_dir,
                                   aggregate_type=aggregate, enhance_type=enhance, reduce_dimension=rd)
    return extract_helper


def build_cfg():
    pass
