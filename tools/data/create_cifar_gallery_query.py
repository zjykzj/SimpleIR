# -*- coding: utf-8 -*-

"""
@date: 2022/9/1 上午11:21
@file: create_cifar_gallery_query.py
@author: zj
@description: Download the data and parse the train / test data set.
Split the test set, extract 20 pieces of each class as the search set, and the rest as the query set.

>>>python create_cifar_gallery_query.py --root ./data/cifar --cifar100
"""
import glob
import os
import random
import shutil
import argparse

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.datasets import cifar
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Create cifar gallery/query')
    parser.add_argument('--root', type=str, default='./data/cifar',
                        help='Data root path')
    parser.add_argument('--cifar100', action='store_true',
                        help='Whether to use cifar100. Default: false')

    args = parser.parse_args()

    return args


def init():
    seed = 100

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def custom_collate(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    return images, targets


def create_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def create_train_test(data_root, is_cifar100=True, is_train=True):
    dataset = cifar.CIFAR100(data_root, download=True, train=is_train)
    if is_cifar100:
        cifar_root = os.path.join(data_root, 'cifar100')
    else:
        cifar_root = os.path.join(data_root, 'cifar10')
    create_folder(cifar_root)

    classes = dataset.classes
    if is_train:
        print(classes)
        class_path = os.path.join(cifar_root, 'class.txt')
        print(f'save classes to {class_path}')
        np.savetxt(class_path, classes, fmt='%s', delimiter=' ')

    dataloader = DataLoader(dataset, shuffle=False, batch_size=32, num_workers=4, pin_memory=True,
                            collate_fn=custom_collate)

    if is_train:
        image_root = os.path.join(cifar_root, 'train')
    else:
        image_root = os.path.join(cifar_root, 'test')
    print(f'process images: {image_root}')
    idx = 0
    for images, targets in tqdm(dataloader):
        for pil_image, target in zip(images, targets):
            assert isinstance(pil_image, Image.Image)
            cls_name = classes[target]
            cls_dir = os.path.join(image_root, cls_name)
            if not os.path.exists(cls_dir):
                os.makedirs(cls_dir)

            img_path = os.path.join(cls_dir, f'{idx}.jpg')
            pil_image.save(img_path)

            idx += 1

    return cifar_root, image_root


def create_gallery_query(cifar_root, image_root):
    gallery_dir = os.path.join(cifar_root, 'gallery')
    create_folder(gallery_dir)
    query_dir = os.path.join(cifar_root, 'query')
    create_folder(query_dir)

    gallery_num_per_cls = 20

    cls_list = os.listdir(image_root)
    for cls_name in tqdm(cls_list):
        cls_dir = os.path.join(image_root, cls_name)

        img_list = glob.glob(os.path.join(cls_dir, '*.jpg'))
        assert len(img_list) > gallery_num_per_cls, img_list
        idx_list = list(range(len(img_list)))
        np.random.shuffle(idx_list)

        gallery_idx_list = idx_list[:gallery_num_per_cls]
        query_idx_list = idx_list[gallery_num_per_cls:]

        gallery_img_list = np.array(img_list)[gallery_idx_list]
        query_img_list = np.array(img_list)[query_idx_list]

        gallery_cls_dir = os.path.join(gallery_dir, cls_name)
        create_folder(gallery_cls_dir)
        for img_path in gallery_img_list:
            img_name = os.path.basename(img_path)
            gallery_img_path = os.path.join(gallery_cls_dir, img_name)

            shutil.copy(img_path, gallery_img_path)

        query_cls_dir = os.path.join(query_dir, cls_name)
        create_folder(query_cls_dir)
        for img_path in query_img_list:
            img_name = os.path.basename(img_path)
            query_img_path = os.path.join(query_cls_dir, img_name)

            shutil.copy(img_path, query_img_path)


def main(args):
    init()

    data_root = args.root
    _, _ = create_train_test(data_root, is_cifar100=args.cifar100, is_train=True)
    cifar_root, image_root = create_train_test(data_root, is_cifar100=args.cifar100, is_train=False)

    create_gallery_query(cifar_root, image_root)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
