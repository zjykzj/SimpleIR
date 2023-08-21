# -*- coding: utf-8 -*-

"""
@date: 2023/8/21 上午10:22
@file: imagefolder.py
@author: zj
@description: 
"""
from typing import Optional, Callable, Any, Tuple

import numpy as np

import torch
import torchvision.datasets as datasets


class FeatureFolder(datasets.ImageFolder):

    def __init__(self, root: str):
        super().__init__(root)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # return super().__getitem__(index)
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = torch.from_numpy(np.load(path))

        return sample, target, path

    def __len__(self) -> int:
        return super().__len__()
