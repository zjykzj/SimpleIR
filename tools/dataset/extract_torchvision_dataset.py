# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午6:30
@file: extract_mnist_dataset.py
@author: zj
@description:

Usage - Create Dataset:
    $ python tools/dataset/extract_torchvision_dataset.py --dataset CIFAR10 ../datasets/cifar10
    $ python tools/dataset/extract_torchvision_dataset.py --dataset CIFAR100 ../datasets/cifar100
    $ python tools/dataset/extract_torchvision_dataset.py --dataset MNIST ../datasets/mnist
    $ python tools/dataset/extract_torchvision_dataset.py --dataset FashionMNIST ../datasets/fashionmnist

Usage - Create Toy Dataset:
    $ python tools/dataset/extract_torchvision_dataset.py --dataset CIFAR100 --toy ../datasets/cifar100

"""
import glob
import os
import argparse
import shutil
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
    parser.add_argument('data_root', metavar='DIR', help='Path to dataset')
    parser.add_argument('--dataset', '-data', metavar='DATASET', default='MNIST',
                        choices=__supported__,
                        help='Dataset type: ' +
                             ' | '.join(__supported__) +
                             ' (default: CIFAR10)')
    parser.add_argument('--toy', action='store_true', default=False, help='Create toy dataset. (default: False)')
    parser.add_argument('--toy-num', type=int, default=10,
                        help='How many images per category are in the toy dataset. (default: 10)')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


def process(data_root: str, dataset: Dataset, is_train: bool = True, is_toy: bool = False, toy_num: int = 10) -> None:
    assert isinstance(dataset, datasets.VisionDataset)

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    print(f"Create dataset: {data_root}")
    classes = dataset.classes
    if is_train:
        for idx, class_name in enumerate(classes):
            print(f"{idx}: {class_name}")

    cls_dict = {class_name: 0 for class_name in classes}
    for idx in tqdm(range(len(dataset))):
        image, target = dataset.__getitem__(idx)
        assert isinstance(image, Image.Image)

        class_name = classes[target]
        cls_dir = os.path.join(data_root, class_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        cls_dict[class_name] = cls_dict[class_name] + 1
        prefix = 'train' if is_train else 'val'
        img_path = os.path.join(cls_dir, f'{prefix}_{class_name}-{cls_dict[class_name]}.jpg')
        image.save(img_path)

    if is_toy:
        dataset_name = os.path.basename(data_root)
        toy_dataset_name = 'toy_' + dataset_name
        toy_data_root = os.path.join(os.path.dirname(data_root), toy_dataset_name)
        if not os.path.exists(toy_data_root):
            os.makedirs(toy_data_root)

        print(f"Create Toy: {toy_data_root}")
        for class_name in classes:
            src_cls_dir = os.path.join(data_root, class_name)
            dst_cls_dir = os.path.join(toy_data_root, class_name)
            if not os.path.exists(dst_cls_dir):
                os.makedirs(dst_cls_dir)

            file_list = list(glob.glob(os.path.join(src_cls_dir, "*.jpg")))[:toy_num]
            for file_path in file_list:
                dst_file_path = file_path.replace(src_cls_dir, dst_cls_dir)
                shutil.copyfile(file_path, dst_file_path)


def main(args: Namespace) -> None:
    data_root = args.data_root
    dataset = args.dataset

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if dataset == 'MNIST':
        train_dataset = datasets.MNIST(data_root, train=True, download=True)
        val_dataset = datasets.MNIST(data_root, train=False, download=True)
    elif dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True)
        val_dataset = datasets.CIFAR10(data_root, train=False, download=True)
    elif dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_root, train=True, download=True)
        val_dataset = datasets.CIFAR100(data_root, train=False, download=True)
    elif dataset == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_root, train=True, download=True)
        val_dataset = datasets.FashionMNIST(data_root, train=False, download=True)
    else:
        raise ValueError(f"{dataset} does not support")

    process(train_dir, train_dataset, is_train=True, is_toy=args.toy, toy_num=args.toy_num)
    process(val_dir, val_dataset, is_train=False, is_toy=args.toy, toy_num=args.toy_num)


if __name__ == '__main__':
    args = parse()
    main(args)
