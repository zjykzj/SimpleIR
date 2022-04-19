# -*- coding: utf-8 -*-

"""
@date: 2022/3/28 下午3:25
@file: infer.py
@author: zj
@description: 
"""
import torch
import argparse

import numpy as np
from PIL import Image

from simpleir.data.build import build_dataset, build_transform
from simpleir.models.tiny_autocoder import TinyAutoCoder
from simpleir.utils.util import load_model


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=100, metavar='N',
                        help='image index in test MNIST (default: 100)')
    parser.add_argument('-m', '--path', default='./outputs/model_e9.pt', type=str, metavar='PATH',
                        help='pretrained model path (default: ./outputs/model_e9.pt)')
    return parser.parse_args()


def get_img(img_idx=1000):
    m = build_dataset('./data', is_train=False)
    # print(m)

    img, target = m.__getitem__(img_idx)
    # print(type(img), target)

    return img, target


def get_model(model_path):
    model = TinyAutoCoder(in_channels=1)

    model = load_model(model, model_path)
    return model


def main(args):
    img_idx = args.idx
    img, target = get_img(img_idx)
    assert isinstance(img, Image.Image)
    src_img_path = f'outputs/mnist_i{img_idx}_t{target}.png'
    print(f'save src img: {src_img_path}')
    img.save(src_img_path)

    tr = build_transform()
    input_img = tr(img)

    model = get_model(args.path)
    model.eval()

    decoded = model(input_img.unsqueeze(0))[0]
    assert isinstance(decoded, torch.Tensor)

    decoded_np = decoded.detach().squeeze(0).numpy()

    decoded_np = decoded_np * 255
    decoded_np = decoded_np.astype(np.uint8)

    res_img = Image.fromarray(decoded_np)
    dst_img_path = f'outputs/mnist_i{img_idx}_t{target}_de.png'
    print(f'save dst img: {dst_img_path}')
    res_img.save(dst_img_path)

    print('done')


if __name__ == '__main__':
    args = load_args()
    main(args)
