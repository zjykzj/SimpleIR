# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: data.py
@author: zj
@description: 
"""

from typing import Tuple, Optional, Any, List

from yacs.config import CfgNode
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import Compose

__all__ = ['build_data']


def custom_fn(batches: List) -> Tuple[Tensor, Tensor, List]:
    images = [batch[0] for batch in batches]
    targets = [batch[1] for batch in batches]
    paths = [batch[2] for batch in batches]

    if isinstance(targets[0], int):
        targets = torch.from_numpy(np.array(targets))
        return torch.stack(images), targets, paths
    else:
        return torch.stack(images), torch.stack(targets), paths


def build_data(cfg: CfgNode,
               is_gallery: bool = False,
               transform: Compose = None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    if is_gallery:
        dataset = ImageFolder(cfg['gallery'], transform=transform)
    else:
        dataset = ImageFolder(cfg['query'], transform=transform)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader
