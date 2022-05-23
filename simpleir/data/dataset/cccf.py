# -*- coding: utf-8 -*-

"""
@date: 2022/5/20 下午3:10
@file: cccf.py
@author: zj
@description: Custom CCCF. Derived from zcls2.data.dataset.cccf.CCCF
"""
from typing import Optional, Callable, Any

from PIL import Image

from zcls2.data.dataset import cccf

__all__ = ['CCCF']


class CCCF(cccf.CCCF):

    def __init__(self, root: str, train: Optional[bool] = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, w_path: bool = False) -> None:
        super().__init__(root, train, transform, target_transform)

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
            return img, target, img_path
        else:
            return img, target
