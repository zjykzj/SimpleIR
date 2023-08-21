# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午6:30
@file: extract_mnist_dataset.py
@author: zj
@description: 
"""

import os
import argparse
from argparse import Namespace

from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset
import torchvision.datasets as datasets

__supported__ = [
    'MNIST',
    'CIFAR10',
    'CIFAR100',
    'FashionMNIST'
]


def parse() -> Namespace:
    parser = argparse.ArgumentParser(description='Extract MNIST/CIFAR10/CIFAR100/FASHIONMNIST dataset')
    parser.add_argument('root', metavar='DIR', help='Path to dataset')
    parser.add_argument('--dataset', '-data', metavar='DATASET', default='MNIST',
                        choices=__supported__,
                        help='Dataset type: ' +
                             ' | '.join(__supported__) +
                             ' (default: CIFAR10)')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


def process(data_root: str, dataset: Dataset) -> None:
    assert isinstance(dataset, datasets.VisionDataset)

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    classes = dataset.classes
    dataset_len = len(dataset)
    for idx in tqdm(range(dataset_len)):
        image, target = dataset.__getitem__(idx)
        assert isinstance(image, Image.Image)

        class_name = classes[target]
        cls_dir = os.path.join(data_root, class_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        img_path = os.path.join(cls_dir, f'{idx}.jpg')
        image.save(img_path)


def main(data_root: str, dataset_name: str) -> None:
    assert isinstance(dataset_name, str) and dataset_name in __supported__

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(data_root, train=True, download=True)
        val_dataset = datasets.MNIST(data_root, train=False, download=True)
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True)
        val_dataset = datasets.CIFAR10(data_root, train=False, download=True)
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_root, train=True, download=True)
        val_dataset = datasets.CIFAR100(data_root, train=False, download=True)
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_root, train=True, download=True)
        val_dataset = datasets.FashionMNIST(data_root, train=False, download=True)
    else:
        raise ValueError(f"{dataset_name} does not support")

    process(train_dir, train_dataset)
    process(val_dir, val_dataset)


if __name__ == '__main__':
    args = parse()
    print(args)

    main(args.data, args.dataset)
