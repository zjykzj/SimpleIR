# -*- coding: utf-8 -*-

"""
@date: 2022/5/20 下午3:10
@file: cccf.py
@author: zj
@description: Custom CCCF. Derived from zcls2.data.dataset.cccf.CCCF
"""
from typing import Optional, Callable, Any

import os

from PIL import Image

from zcls2.config.key_word import KEY_SEP
from zcls2.data.dataset import cccf

__all__ = ['CCCF']


class CCCF(cccf.CCCF):

    def __init__(self, root: str, train: Optional[bool] = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, w_path: bool = False, is_gallery=True) -> None:
        if train:
            super().__init__(root, train, transform, target_transform)
        else:
            assert os.path.isdir(root), root

            class_path = os.path.join(root, 'classes.txt')
            assert os.path.isfile(class_path), class_path
            gallery_path = os.path.join(root, 'gallery.txt')
            assert os.path.isfile(gallery_path), gallery_path
            query_path = os.path.join(root, 'query.txt')
            assert os.path.isfile(query_path), query_path

            classes = cccf.load_classes(class_path, delimiter=' ')
            data_list = cccf.load_txt(gallery_path, delimiter=KEY_SEP) if is_gallery else \
                cccf.load_txt(query_path, delimiter=KEY_SEP)

            self.classes = classes
            self.data = [os.path.join(root, str(img_path)) for img_path, target in data_list]
            self.targets = [int(target) for img_path, target in data_list]

            self.root = root
            self.transform = transform
            self.target_transform = target_transform

        self.w_path = w_path

    def __getitem__(self, index) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.w_path:
            # In order to better distinguish image names, add dataset and class as prefix to image name
            img_dir = os.path.dirname(img_path)
            img_name = os.path.basename(img_path)

            tmp = img_path.replace(self.root, "")
            tmp = tmp.lstrip('/')
            dataset_name = tmp.split('/')[0]

            cls_name = os.path.basename(img_dir)
            new_img_path = os.path.join(img_dir, f"{dataset_name}_{cls_name}_{img_name}")
            return img, target, new_img_path
        else:
            return img, target
