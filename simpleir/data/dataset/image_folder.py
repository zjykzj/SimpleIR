# -*- coding: utf-8 -*-

"""
@date: 2022/5/20 下午4:10
@file: image_folder.py
@author: zj
@description: Custom ImageFolder, derived from torchvision.datasets.ImageFolder
"""
from typing import Optional, Callable, Any

from torchvision import datasets
from torchvision.datasets.folder import default_loader

__all__ = ['ImageFolder']


class ImageFolder(datasets.ImageFolder):

    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None,
                 w_path: bool = False) -> None:
        super().__init__(root, transform, target_transform, loader, is_valid_file)

        self.w_path = w_path

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.w_path:
            return sample, target, path
        else:
            return sample, target
