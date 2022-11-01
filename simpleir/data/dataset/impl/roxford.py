# -*- coding: utf-8 -*-

"""
@date: 2022/10/30 下午2:06
@file: roxford.py
@author: zj
@description: 
"""
import pickle
from typing import Optional, Callable

import os.path

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.datasets.folder import default_loader

__all__ = ['Oxford5k', 'Paris6k', 'ROxford', 'ROxford5k', 'RParis6k']


class ROxford(Dataset):

    def __init__(self, root: str, dataset='oxford5k', is_gallery: bool = False, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, w_path: bool = False) -> None:
        assert os.path.isdir(root), root

        img_root = os.path.join(root, 'jpg')
        assert os.path.isdir(img_root), img_root

        gnd_file_path = os.path.join(root, f'gnd_{dataset.lower()}.pkl')
        assert os.path.isfile(gnd_file_path), gnd_file_path
        with open(gnd_file_path, 'rb') as f:
            gnd_info_dict = pickle.load(f)

        if is_gallery is True:
            self.img_list = [os.path.join(img_root, f'{img_name}.jpg') for img_name in gnd_info_dict['imlist']]
        else:
            self.img_list = [os.path.join(img_root, f'{img_name}.jpg') for img_name in gnd_info_dict['qimlist']]

        self.is_gallery = is_gallery
        self.transform = transform
        self.target_transform = target_transform
        self.w_path = w_path

    def __getitem__(self, index) -> T_co:
        img_path = self.img_list[index]
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
        return len(self.img_list)


Oxford5k = ROxford
Paris6k = ROxford
ROxford5k = ROxford
RParis6k = ROxford
