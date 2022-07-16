# -*- coding: utf-8 -*-

"""
@date: 2022/7/14 上午11:45
@file: extract_features.py
@author: zj
@description: Extract features. You should set --image-dir and --save-dir
for example:
1. python extract_features.py --image-dir data/train/ --save-dir data/gallery_fc
2. python extract_features.py --image-dir data/test/ --save-dir data/query_fc

in save-dir, you can find info.pkl. It's a dict and the key/value like this
1. cls_list: [cls1, cls2, ...]
2. description: ''
3. content: {'img_name1": label, 'img_name2': label}
4. feat: 'fc'
5. model: 'resnet50'
6. pretrained: ''
"""

import os
import pickle

import torch
import argparse

from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from simpleir.configs.key_words import KEY_FEAT

from model import *
from dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features")
    parser.add_argument('--model-arch', metavar="ARCH", default='resnet50',
                        help='Model arch for extracting features. Default: resnet50')
    parser.add_argument('--pretrained', metavar='PRETRAINED', default=None,
                        help='Pretrained model params path. Default: None')
    parser.add_argument('--layer', metavar='LAYER', default='fc',
                        help='Location of model extracted features. Default: fc')

    parser.add_argument('--dataset', metavar='DATASET', default='General',
                        help='Dataset type for image processing. Default: General')
    parser.add_argument('--image-dir', metavar='IMAGE', default=None,
                        help='Dir for loading images. Default: None')
    parser.add_argument('--save-dir', metavar='SAVE', default=None,
                        help='Dir for saving features. Default: None')

    return parser.parse_args()


def load_model(arch='resnet50', pretrained=None, layer='fc'):
    model = eval(arch)()
    assert isinstance(model, torch.nn.Module)

    assert hasattr(model, 'support_feat')
    if not model.support_feat(layer):
        raise ValueError(f'{arch} does not support {layer}')

    model = eval(arch)(pretrained=True, feat_type=layer)
    assert isinstance(model, torch.nn.Module)

    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)

    return model


def custom_fn(batches):
    images = [batch[0] for batch in batches]
    targets = [batch[1] for batch in batches]
    paths = [batch[2] for batch in batches]

    return torch.stack(images), torch.stack(targets), paths


def load_data(data_root, dataset='GeneralDataset'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_set = eval(dataset)(root=data_root, transform=transform, w_path=True)

    return DataLoader(data_set, collate_fn=custom_fn, shuffle=False, batch_size=32, num_workers=0, pin_memory=True)


def process(data_loader, model, save_dir=None):
    assert os.path.isdir(save_dir), save_dir

    content_dict = OrderedDict()
    for images, targets, paths in tqdm(data_loader):
        res_dict = model.forward(images)
        feats = res_dict[KEY_FEAT].detach().cpu().numpy()

        for path, target, feat in zip(paths, targets, feats):
            name = os.path.splitext(os.path.split(path)[1])[0]
            save_path = os.path.join(save_dir, f'{name}.npy')

            np.save(save_path, feat)
            content_dict[name] = target.item()

    return content_dict


def main():
    args = parse_args()
    print('args:', args)

    model = load_model(arch=args.model_arch, pretrained=args.pretrained, layer=args.layer)
    print(model)

    data_loader = load_data(args.image_dir, dataset=args.dataset)
    print(data_loader)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    res_dict = process(data_loader, model, save_dir=args.save_dir)

    info_dict = {
        'feat': args.layer,
        'model': args.model_arch,
        'pretrained': args.pretrained,
        'classes': data_loader.dataset.classes,
        'content': res_dict
    }
    info_path = os.path.join(args.save_dir, 'info.pkl')
    print(f'save to {info_path}')
    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == '__main__':
    main()
