# -*- coding: utf-8 -*-

"""
@date: 2022/7/25 下午8:49
@file: oxford.py
@author: zj
@description: 
"""
from typing import Optional, Callable

import os.path
from glob import glob

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.datasets.folder import default_loader

__all__ = ['Oxford']


class Oxford(Dataset):

    def __init__(self, root: str, is_gallery: bool = False, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, w_path: bool = False) -> None:
        assert os.path.isdir(root), root

        self.gt_root = os.path.join(root, 'groundtruth')
        assert os.path.isdir(self.gt_root), self.gt_root
        self.images_root = os.path.join(root, 'images')
        assert os.path.isdir(self.images_root), self.images_root

        self.image_list = self.create_gallery() if is_gallery else self.create_query()
        self.length = len(self.image_list)

        self.transform = transform
        self.target_transform = target_transform
        self.w_path = w_path

        self.classes = list(range(11))

    def create_gallery(self):
        image_list = glob(os.path.join(self.images_root, '*.jpg'))

        return image_list

    def create_query(self):
        query_file_list = glob(os.path.join(self.gt_root, '*_query.txt'))

        image_list = list()
        for query_file_path in query_file_list:
            assert os.path.isfile(query_file_path), query_file_path

            with open(query_file_path, 'r') as f:
                for line in f:
                    tmp_list = line.strip().split(' ')
                    assert len(tmp_list) == 5

                    image_name = tmp_list[0]
                    if image_name.startswith('oxc1_'):
                        image_name = image_name.replace('oxc1_', '')
                    query_image_path = os.path.join(self.images_root, f'{image_name}.jpg')
                    assert os.path.join(query_image_path), query_file_path

                    image_list.append(query_image_path)
        return image_list

    def __getitem__(self, index) -> T_co:
        img_path = self.image_list[index]
        target = 0

        image = default_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.w_path:
            return image, torch.tensor(int(target)), img_path
        else:
            return image, torch.tensor(int(target))

    def __len__(self):
        return self.length
