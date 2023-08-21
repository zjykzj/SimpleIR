# -*- coding: utf-8 -*-

"""
@date: 2023/8/21 上午10:22
@file: imagefolder.py
@author: zj
@description: 
"""
from typing import Optional, Callable, Any, Tuple

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class ImageFolder(datasets.ImageFolder):

    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # return super().__getitem__(index)
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self) -> int:
        return super().__len__()
